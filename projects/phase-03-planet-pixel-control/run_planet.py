"""PlaNet 像素控制实战项目主入口：训练、评估、开环预测视频与控制对比视频。"""

import argparse
import json
import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", ".."))
for p in (os.path.join(REPO_ROOT, "src"), PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import torch
import torch.nn as nn

from latent_planner import run_episode
from pixel_env import DEFAULT_ACTION_REPEAT, SequenceReplay, postprocess_obs, preprocess_obs
from rssm_model import WorldModel, world_model_loss
from utils import get_device, is_quick, output_dir, save_figure, save_video, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-03-project")
DEVICE = get_device()
SEED = 0


@torch.no_grad()
def open_loop_metrics_and_video(model, episode, video_path, context=10):
    """在一条评估回合上计算开环预测误差并生成真实 vs 想象对比视频。"""
    obs_t = preprocess_obs(torch.from_numpy(episode["obs"][1:]).to(DEVICE)).unsqueeze(0)  # [1, L, 3, 64, 64]
    act_t = torch.from_numpy(episode["actions"][1:]).to(DEVICE).unsqueeze(0)              # [1, L, 1]
    rew_t = torch.from_numpy(episode["rewards"][1:]).to(DEVICE).unsqueeze(0)              # [1, L]

    ctx_emb = model.encode_seq(obs_t[:, :context])
    ctx_out = model.rssm.observe(ctx_emb, act_t[:, :context], sample=False)
    h_c, z_c = ctx_out["h"][:, -1], ctx_out["z"][:, -1]

    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    imag_samp = model.rssm.imagine(h_c, z_c, act_t[:, context:], sample=True, generator=gen)
    imag_mean = model.rssm.imagine(h_c, z_c, act_t[:, context:], sample=False)

    pred_obs_samp = model.decode_seq(imag_samp)
    pred_obs_mean = model.decode_seq(imag_mean)
    pred_rew_mean = model.reward_head(imag_mean)

    pix_mse = (pred_obs_mean - obs_t[:, context:]).pow(2).mean(dim=(0, 2, 3, 4)).cpu().numpy()
    rew_rmse = (pred_rew_mean - rew_t[:, context:]).pow(2).mean(dim=0).sqrt().cpu().numpy()

    if video_path is not None:
        true_seq = postprocess_obs(obs_t[0])
        ctx_rec = postprocess_obs(model.decode_seq(ctx_out["features"])[0])
        samp_seq = np.concatenate([ctx_rec, postprocess_obs(pred_obs_samp[0])], axis=0)
        mean_seq = np.concatenate([ctx_rec, postprocess_obs(pred_obs_mean[0])], axis=0)
        frames = []
        for idx in range(len(true_seq)):
            border = np.array([0, 210, 210], dtype=np.uint8) if idx < context else np.array([220, 80, 60], dtype=np.uint8)
            cols = []
            for img in (true_seq[idx], samp_seq[idx], mean_seq[idx]):
                u8 = (img * 255.0).clip(0, 255).astype(np.uint8)
                big = cv2.resize(u8, (128, 128), interpolation=cv2.INTER_NEAREST)
                big[:3] = big[-3:] = big[:, :3] = big[:, -3:] = border
                cols.append(big)
            spacer = np.full((128, 6, 3), 255, dtype=np.uint8)
            frames.append(np.concatenate([cols[0], spacer, cols[1], spacer, cols[2]], axis=1))
        save_video(frames, video_path, fps=12)

    return pix_mse.tolist(), rew_rmse.tolist()


def parse_args():
    p = argparse.ArgumentParser(description="PlaNet 像素版 Pendulum-v1 控制")
    p.add_argument("--init-episodes", type=int, default=None)
    p.add_argument("--online-episodes", type=int, default=None)
    p.add_argument("--action-repeat", type=int, default=DEFAULT_ACTION_REPEAT)
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(SEED)
    if DEVICE.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    quick = is_quick()

    if quick:
        cfg = dict(n_init=2, n_online=2, upd_init=15, upd_per_ep=10, batch_size=4, chunk_len=12,
                   horizon=10, n_candidates=50, n_iters=3, eval_seeds=[1000], eval_every=1)
    else:
        cfg = dict(n_init=5, n_online=10, upd_init=150, upd_per_ep=60, batch_size=12, chunk_len=25,
                   horizon=15, n_candidates=300, n_iters=6, eval_seeds=[1000, 1001, 1002, 1003, 1004], eval_every=3)
    if args.init_episodes is not None:
        cfg["n_init"] = args.init_episodes
    if args.online_episodes is not None:
        cfg["n_online"] = args.online_episodes
    tag = f"-{args.tag}" if args.tag else ""

    print("=" * 60)
    print("Phase 3 实战项目：PlaNet 像素控制（Pendulum-v1，64×64 RGB）")
    print("=" * 60)
    print(f"  设备：{DEVICE} ｜ 动作重复 R = {args.action_repeat} ｜ 初始回合 S = {cfg['n_init']} ｜ 在线回合 = {cfg['n_online']}")

    rand_eval = [run_episode(s, model=None, action_repeat=args.action_repeat) for s in cfg["eval_seeds"]]
    random_return = float(np.mean([e["ret"] for e in rand_eval]))
    print(f"  参照：随机策略平均回报 = {random_return:.1f}")

    replay = SequenceReplay()
    for i in range(cfg["n_init"]):
        ep = run_episode(SEED + i, model=None, action_repeat=args.action_repeat)
        replay.add_episode(ep["obs"], ep["actions"], ep["rewards"])

    torch.manual_seed(SEED)
    model = WorldModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(SEED)

    history = []
    early_eval_ep, final_eval_ep = None, None
    t_start = time.time()
    train_secs, plan_secs = 0.0, 0.0

    for r_idx in range(1, cfg["n_online"] + 1):
        n_upd = cfg["upd_init"] if r_idx == 1 else cfg["upd_per_ep"]
        model.train()
        t0 = time.time()
        last_m = None
        for _ in range(n_upd):
            obs_b, act_b, rew_b = replay.sample(cfg["batch_size"], cfg["chunk_len"], DEVICE, generator=gen)
            loss, last_m = world_model_loss(model, obs_b, act_b, rew_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
            optimizer.step()
        train_secs += time.time() - t0

        ep_train = run_episode(
            SEED + 200 + r_idx, model=model, device=DEVICE, action_repeat=args.action_repeat,
            explore_std=0.25, horizon=cfg["horizon"], n_candidates=cfg["n_candidates"], n_iters=cfg["n_iters"],
        )
        replay.add_episode(ep_train["obs"], ep_train["actions"], ep_train["rewards"])
        plan_secs += ep_train["plan_time"]
        env_steps = len(replay) * 200

        if r_idx == 1 or r_idx % cfg["eval_every"] == 0 or r_idx == cfg["n_online"]:
            eval_eps = [
                run_episode(s, model=model, device=DEVICE, action_repeat=args.action_repeat,
                            explore_std=0.0, horizon=cfg["horizon"], n_candidates=cfg["n_candidates"],
                            n_iters=cfg["n_iters"])
                for s in cfg["eval_seeds"]
            ]
            plan_secs += sum(e["plan_time"] for e in eval_eps)
            rets = [e["ret"] for e in eval_eps]
            pix_mse, rew_rmse = open_loop_metrics_and_video(model, eval_eps[0], video_path=None)
            rec = dict(
                round=r_idx, env_steps=env_steps, ret_mean=float(np.mean(rets)), ret_std=float(np.std(rets)),
                recon_loss=last_m["recon"], raw_kl=last_m["raw_kl"],
                open_loop_pix_mse_15=float(pix_mse[min(14, len(pix_mse) - 1)]),
                open_loop_rew_rmse_15=float(rew_rmse[min(14, len(rew_rmse) - 1)]),
            )
            history.append(rec)
            if early_eval_ep is None:
                early_eval_ep = eval_eps[0]
            final_eval_ep = eval_eps[0]
            print(f"  第 {r_idx:>2}/{cfg['n_online']} 轮（{env_steps:>4} 环境步）："
                  f"评估回报 {rec['ret_mean']:6.1f} ± {rec['ret_std']:4.1f} ｜ "
                  f"15步奖励RMSE {rec['open_loop_rew_rmse_15']:4.2f} ｜ 重建 {rec['recon_loss']:5.1f}, KL {rec['raw_kl']:4.1f}")

    total_secs = time.time() - t_start
    pix_mse_final, rew_rmse_final = open_loop_metrics_and_video(
        model, final_eval_ep, video_path=OUT / f"open-loop-prediction{tag}.mp4"
    )

    # 控制对比视频（左：第 1 轮，右：最终轮）
    vframes = []
    for f1, f2 in zip(early_eval_ep["obs"], final_eval_ep["obs"]):
        l_img = cv2.resize(f1, (128, 128), interpolation=cv2.INTER_NEAREST)
        r_img = cv2.resize(f2, (128, 128), interpolation=cv2.INTER_NEAREST)
        vframes.append(np.concatenate([l_img, np.full((128, 8, 3), 240, dtype=np.uint8), r_img], axis=1))
    save_video(vframes, OUT / f"control-progress{tag}.mp4", fps=15)

    # 学习曲线 + 开环误差图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    xs = [h["env_steps"] for h in history]
    axes[0].errorbar(xs, [h["ret_mean"] for h in history], yerr=[h["ret_std"] for h in history],
                     fmt="o-", linewidth=2, capsize=4, label="PlaNet（64×64 像素）")
    axes[0].axhline(random_return, color="tomato", linestyle=":", label=f"随机策略（{random_return:.0f}）")
    axes[0].set_xlabel("累计真实环境步数")
    axes[0].set_ylabel("评估回报")
    axes[0].set_title("学习曲线")
    axes[0].legend()

    d_ax = np.arange(1, len(rew_rmse_final) + 1)
    axes[1].plot(d_ax, rew_rmse_final, linewidth=2, color="C2", label="开环奖励预测 RMSE")
    axes[1].axvline(cfg["horizon"], color="gray", linestyle="--", label=f"规划视野 H = {cfg['horizon']}")
    axes[1].set_xlabel("开环预测步数 d")
    axes[1].set_ylabel("奖励 RMSE")
    axes[1].set_title("最终模型的开环奖励预测精度")
    axes[1].legend()
    save_figure(fig, OUT / f"learning-curve{tag}.png")

    summary = dict(
        device=str(DEVICE), total_seconds=total_secs, train_seconds=train_secs, plan_seconds=plan_secs,
        random_return=random_return, config=cfg, history=history,
        final_open_loop_pix_mse=pix_mse_final, final_open_loop_rew_rmse=rew_rmse_final,
    )
    with open(OUT / f"results{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📄 结果已保存：outputs/phase-03-project/results{tag}.json（总用时 {total_secs:.1f}s）")


if __name__ == "__main__":
    main()
