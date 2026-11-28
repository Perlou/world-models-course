"""CarRacing World Models (V-M-C) 完整管线实战入口。

支持从头运行四阶段（采集数据 → 训练 V → 训练 M → CMA-ES 训练 C），
或在默认情况下自动复用 outputs/phase-02/ 下已生成的 V/M/C 产物，并输出：
    - real-recon-dream.mp4   左：真实赛道画面 ｜ 中：VAE 单帧重建 ｜ 右：MDN-RNN 开环梦境
    - learning-curve.png     [z_t ; h_t] 与 只有 z_t 控制器的 CMA-ES 学习曲线与独立赛道得分分布
    - results.json           全部评测指标
"""

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", ".."))
for p in (os.path.join(REPO_ROOT, "src"), PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import gymnasium as gym
import numpy as np
import torch

from cmaes_controller import LinearController
from memory_mdn import HIDDEN_DIM, MDNRNN, sample_mdn
from utils import get_device, is_quick, output_dir, save_figure, save_video, seed_everything, setup_matplotlib
from vision_vae import ZOOM_FRAMES, ConvVAE, preprocess_frame, to_tensor_obs

plt = setup_matplotlib()
OUT = output_dir("phase-02-project")
PHASE2_OUT = output_dir("phase-02")
Phase2Src = os.path.join(REPO_ROOT, "src", "phase-02-world-models-2018")
SEED = 0
DEVICE = get_device()


def ensure_phase2_artifacts(suffix):
    """若 V / M / C 产物尚未生成，按顺序调用对应讲义生成。"""
    needed = [
        (PHASE2_OUT / f"vae{suffix}.pt", "02-conv-vae-vision.py"),
        (PHASE2_OUT / f"mdnrnn{suffix}.pt", "04-mdn-rnn-memory.py"),
        (PHASE2_OUT / f"controller-real{suffix}.npz", "07-controller-in-real-env.py"),
    ]
    for path, script in needed:
        if not path.exists():
            print(f"📦 未找到 {path.name}，正在运行 {script} 生成……")
            subprocess.run([sys.executable, os.path.join(Phase2Src, script)], check=True)


@torch.no_grad()
def evaluate_and_make_triptych(vae, rnn, ctrl_zh, seed, warmup_steps=25, total_steps=220, temperature=1.0):
    """跑一条完整评估赛道，并在第 warmup_steps 步分叉出开环梦境，合成三联对比视频帧。"""
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    obs, _ = env.reset(seed=int(seed))
    total_reward = 0.0
    for _ in range(ZOOM_FRAMES):
        obs, r, term, trunc, _ = env.step(np.zeros(3, dtype=np.float32))
        total_reward += float(r)
        if term or trunc:
            break

    real_state = (
        torch.zeros(1, 1, HIDDEN_DIM, device=DEVICE),
        torch.zeros(1, 1, HIDDEN_DIM, device=DEVICE),
    )
    dream_z, dream_state = None, None
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    video_frames = []
    recon_mses, dream_mses = [], []

    for step in range(total_steps):
        small_u8 = preprocess_frame(obs)
        obs_t = to_tensor_obs(small_u8, device=DEVICE).unsqueeze(0)  # [1, 3, 64, 64]
        mu, _ = vae.encode(obs_t)
        recon_t = vae.decode(mu)

        z_np = mu.squeeze(0).cpu().numpy()
        h_np = real_state[0].squeeze(0).squeeze(0).cpu().numpy()
        action = ctrl_zh.act(z_np, h_np)
        a_t = torch.from_numpy(action).to(DEVICE).unsqueeze(0)

        if step < warmup_steps:
            # 热身阶段：梦境跟随真实编码，青色边框标记
            dream_img_t = recon_t
            border = np.array([0, 200, 200], dtype=np.uint8)
        else:
            if dream_z is None:
                dream_z = mu.clone()
                dream_state = (real_state[0].clone(), real_state[1].clone())
            dream_img_t = vae.decode(dream_z)
            border = np.array([225, 80, 55], dtype=np.uint8)
            dream_mses.append(float((dream_img_t - obs_t).pow(2).mean().item()))

        recon_mses.append(float((recon_t - obs_t).pow(2).mean().item()))

        # 拼装左（真实 64×64）、中（VAE 重建）、右（开环梦境）三联帧
        recon_u8 = (recon_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        dream_u8 = (dream_img_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        panels = []
        for idx_p, img in enumerate((small_u8, recon_u8, dream_u8)):
            big = cv2.resize(img, (160, 160), interpolation=cv2.INTER_NEAREST)
            if idx_p == 2:
                big[:4] = big[-4:] = big[:, :4] = big[:, -4:] = border
            panels.append(big)
        spacer = np.full((160, 6, 3), 245, dtype=np.uint8)
        video_frames.append(np.concatenate([panels[0], spacer, panels[1], spacer, panels[2]], axis=1))

        # 推进真实环境与真实 LSTM
        obs, r, term, trunc, _ = env.step(action)
        total_reward += float(r)
        real_state = rnn.step_hidden(mu, a_t, real_state)

        # 若已进入开环阶段，用同一真实动作推进梦境（检验 M 对动作响应的一致性）
        if dream_z is not None:
            (lp, m_d, ls_d), dream_state = rnn.step(dream_z, a_t, dream_state)
            dream_z = sample_mdn(lp, m_d, ls_d, temperature=temperature, generator=gen)
        if term or trunc:
            break

    env.close()
    return total_reward, video_frames, float(np.mean(recon_mses)), float(np.mean(dream_mses) if dream_mses else 0.0)


def parse_args():
    p = argparse.ArgumentParser(description="CarRacing World Models (V-M-C) 管线")
    p.add_argument("--temperature", type=float, default=1.0, help="三联视频里梦境分支的采样温度")
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(SEED)
    quick = is_quick()
    suffix = "-quick" if quick else ""
    tag = f"-{args.tag}" if args.tag else ""

    t0 = time.time()
    ensure_phase2_artifacts(suffix)

    vae = ConvVAE().to(DEVICE)
    vae.load_state_dict(torch.load(PHASE2_OUT / f"vae{suffix}.pt", map_location=DEVICE, weights_only=True))
    vae.eval()

    rnn = MDNRNN().to(DEVICE)
    rnn.load_state_dict(torch.load(PHASE2_OUT / f"mdnrnn{suffix}.pt", map_location=DEVICE, weights_only=True))
    rnn.eval()

    with np.load(PHASE2_OUT / f"controller-real{suffix}.npz") as f:
        ctrl_data = {k: f[k] for k in f.files}

    ctrl_z = LinearController(use_hidden_state=False)
    ctrl_z.set_params(ctrl_data["params_z"])
    ctrl_zh = LinearController(use_hidden_state=True)
    ctrl_zh.set_params(ctrl_data["params_zh"])

    scores_z = ctrl_data["scores_z"]
    scores_zh = ctrl_data["scores_zh"]
    logs_z = ctrl_data["logs_z"]
    logs_zh = ctrl_data["logs_zh"]

    print("=" * 60)
    print("Phase 2 实战项目：CarRacing World Models（V + M + C）")
    print("=" * 60)
    print(f"  只有 z_t (99 参数) 独立赛道评估：  {scores_z.mean():6.1f} ± {scores_z.std():5.1f}")
    print(f"  [z_t ; h_t] (867 参数) 独立赛道评估：{scores_zh.mean():6.1f} ± {scores_zh.std():5.1f}")

    warmup, steps = (8, 50) if quick else (25, 240)
    vid_seed = 3000 + int(np.argmax(scores_zh))
    demo_ret, vframes, recon_mse, dream_mse = evaluate_and_make_triptych(
        vae, rnn, ctrl_zh, seed=vid_seed, warmup_steps=warmup, total_steps=steps, temperature=args.temperature
    )
    save_video(vframes, OUT / f"real-recon-dream{tag}.mp4", fps=25)
    print(f"  三联对比视频（赛道 {vid_seed}，前 {len(vframes)} 步回报 {demo_ret:.1f}）："
          f"单帧重建 MSE = {recon_mse:.5f}，开环梦境 MSE = {dream_mse:.5f}")

    # 绘制学习曲线 + 箱线/散点对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].plot(logs_z[:, 0], logs_z[:, 1], color="C1", linewidth=2, label="只有 z_t（当代均值）")
    axes[0].plot(logs_zh[:, 0], logs_zh[:, 1], color="C0", linewidth=2, label="[z_t ; h_t]（当代均值）")
    axes[0].plot(logs_zh[:, 0], logs_zh[:, 2], color="C0", linestyle=":", alpha=0.7, label="[z_t ; h_t]（当代最高）")
    axes[0].set_xlabel("CMA-ES 代数")
    axes[0].set_ylabel("训练期平均回报")
    axes[0].set_title("CMA-ES 训练曲线")
    axes[0].legend(fontsize=8.5)

    axes[1].boxplot([scores_z, scores_zh], tick_labels=["只有 z_t\n(99 参数)", "[z_t ; h_t]\n(867 参数)"], widths=0.45)
    for idx_b, arr in enumerate((scores_z, scores_zh), start=1):
        axes[1].scatter(np.full_like(arr, idx_b) + np.linspace(-0.06, 0.06, len(arr)), arr, alpha=0.65)
    axes[1].axhline(-43.0, color="tomato", linestyle=":", label="随机策略 (-43.0)")
    axes[1].set_ylabel("完整赛道（950 步）独立评估得分")
    axes[1].set_title("M 的隐藏状态 h 带来的性能提升")
    axes[1].legend(fontsize=8.5)
    save_figure(fig, OUT / f"learning-curve{tag}.png")

    summary = dict(
        scores_z_mean=float(scores_z.mean()),
        scores_z_std=float(scores_z.std()),
        scores_zh_mean=float(scores_zh.mean()),
        scores_zh_std=float(scores_zh.std()),
        scores_z=scores_z.tolist(),
        scores_zh=scores_zh.tolist(),
        demo_seed=vid_seed,
        demo_partial_return=demo_ret,
        recon_mse=recon_mse,
        dream_mse=dream_mse,
        elapsed_seconds=time.time() - t0,
    )
    with open(OUT / f"results{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📄 结果已保存：outputs/phase-02-project/results{tag}.json")


if __name__ == "__main__":
    main()
