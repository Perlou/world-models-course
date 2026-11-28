"""
开环预测与 latent overshooting
==============================

学习目标：
    1. 理解训练（一步先验，站在后验上）与规划（连续多步先验，无观测校准）之间的落差
    2. 复现线性高斯系统的多步先验手算例子，理解为什么即使均值预测完美，多步 KL 也随步数自然上升
    3. 实现 open_loop_predict()：前 C 帧用后验热身，之后只喂动作走先验，对比"先验均值解码"与"先验采样解码"，生成对比视频
    4. 实现批量化的 multi_step_kl() 与 latent overshooting 训练，并和标准一步 ELBO 模型同台对比

核心概念：
    - 开环预测（open-loop prediction）：世界模型最重要的诊断手段——给上下文建立信念后，不看观测预测未来
    - 多步先验（multi-step prior）：p(z_t | z_{t-d})，连续走 d 步先验得到的隐状态分布
    - 多步 KL：KL( sg(q_t) ‖ p(z_t | z_{t-1}^{(t-d)}) )，在隐空间度量 d 步预测的漂移与不确定性累积
    - latent overshooting：把 d = 1..D 的隐空间多步 KL 加入训练损失，全程不多做一次图像解码

前置知识：
    - Phase 1 的 04-compounding-error.py（复合误差与多步损失）
    - 04-rssm-from-scratch.py、05-world-model-training.py（训练好的像素世界模型）

对应概念文档：
    - CONCEPT.md 第 8 节「开环预测与 latent overshooting」

预计运行时间：约 7 分钟（CPU）/ 约 2 分钟（MPS / GPU）
"""

import importlib.util
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import numpy as np
import torch
import torch.nn as nn

from utils import get_device, is_quick, output_dir, save_figure, save_video, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lesson04 = _load_module("lesson04", "04-rssm-from-scratch.py")
_lesson05 = _load_module("lesson05", "05-world-model-training.py")

WorldModel = _lesson04.WorldModel
gaussian_kl = _lesson04.gaussian_kl
H_DIM = _lesson04.H_DIM
Z_DIM = _lesson04.Z_DIM
ACTION_DIM = _lesson04.ACTION_DIM

SequenceReplay = _lesson05.SequenceReplay
preprocess_obs = _lesson05.preprocess_obs
postprocess_obs = _lesson05.postprocess_obs
world_model_loss = _lesson05.world_model_loss
FREE_NATS = _lesson05.FREE_NATS

plt = setup_matplotlib()
OUT = output_dir("phase-03")
SEED = 0
DEVICE = get_device()
SUFFIX = "-quick" if is_quick() else ""
DATASET_PATH = OUT / f"pendulum-pixel-dataset{SUFFIX}.npz"
MODEL_PATH = OUT / f"world_model{SUFFIX}.pt"
CONTEXT_LEN = 10


# ==================== 第一部分：导读与手算核对 ====================


def introduction():
    print("=" * 60)
    print("第一部分：重建清晰不等于能预测——用开环预测验真章")
    print("=" * 60)
    print("""
上一节的模型重建画面已经很清晰，但这只证明了"刚看过真图 o_t 的后验 z_t 能把 o_t 画回来"。
规划时面对的是完全不同的处境：

    │◀──── 上下文 C = 10 帧 ────▶│◀──────── 开环预测 d = 1, 2, …, 40 ────────▶│
    │   observe()：看真图校准    │   imagine()：一帧图都不给，只喂真实动作     │
    │   建立起含角速度的信念     │   完全靠先验往前推，解码画面与奖励做检验    │

标准序列 ELBO 只直接约束了 d = 1（站在校准过的后验状态上往前猜一步）。
把约束推广到 d = 1..D 步，又不在像素空间重复解码，就是 latent overshooting（CONCEPT.md 第 8.3 节）：

    q_t ──imagine──▶ p_{t+1|t} ──采样再imagine──▶ p_{t+2|t} ──采样再imagine──▶ p_{t+3|t}
                         │                            │                            │
                     KL(q_{t+1} ‖ ·)             KL(sg(q_{t+2}) ‖ ·)         KL(sg(q_{t+3}) ‖ ·)
                       d = 1                        d = 2                        d = 3

注意 CONCEPT.md 第 8.6 节专门指出的事实：PlaNet 论文附录写明最终版智能体没有启用 latent overshooting，
Dreamer 系列也没用——因为 RSSM 的确定性路径 h_t 已经让多步想象相当稳定。本节用实测数据验证这一点。
    """)


def verify_hand_calculation():
    """复现 CONCEPT.md 第 8.4 节：线性高斯系统多步先验的方差与 KL。"""
    print("\n" + "=" * 60)
    print("第二部分：核对 CONCEPT.md 第 8.4 节的多步 KL 手算例子")
    print("=" * 60)
    print("  设定：一步先验 p(z_t | z_{t-1}) = N(0.9 z_{t-1}, 0.5²)，假设均值预测完美，后验标准差恒为 0.2。")
    print(f"  {'距离 d':<10}{'多步先验方差':>14}{'多步先验标准差':>16}{'KL(后验 ‖ d步先验)':>20}")
    var = 0.0
    results = []
    for d in (1, 2, 3):
        var = 0.81 * var + 0.25
        std = math.sqrt(var)
        kl = gaussian_kl(torch.tensor([0.0]), torch.tensor([0.2]),
                         torch.tensor([0.0]), torch.tensor([std])).item()
        results.append((var, std, kl))
        print(f"  {d:<10}{var:>14.4f}{std:>16.3f}{kl:>20.3f}")
    inf_var = 0.25 / (1.0 - 0.81)
    print(f"  {'∞':<10}{inf_var:>14.3f}{math.sqrt(inf_var):>16.3f}{'—':>20}")
    assert abs(results[0][2] - 0.496) < 2e-3 and abs(results[1][2] - 0.757) < 2e-3 and abs(results[2][2] - 0.900) < 2e-3
    print("  ✓ 与 CONCEPT.md 第 8.4 节一致：即使均值零偏差，多步先验因不确定性累积变宽，KL 也自然上升。")


# ==================== 第二部分：开环预测与多步 KL ====================


def ensure_artifacts():
    if DATASET_PATH.exists() and MODEL_PATH.exists():
        return
    print("⚠️  未找到 05 节的数据集或模型权重，先运行 05-world-model-training.py 生成……")
    script = os.path.join(_DIR, "05-world-model-training.py")
    subprocess.run([sys.executable, script], check=True)


@torch.no_grad()
def open_loop_predict(model, context_obs, actions, sample=True, generator=None):
    """前 C 帧滤波建立信念，剩余 T − C 步只凭动作在隐空间开环想象（CONCEPT.md 第 8.5 节）。

    context_obs [B, C, 3, 64, 64]，actions [B, T, A]
    返回字典：context_recon [B, C, 3, 64, 64]，pred_obs [B, T−C, 3, 64, 64]，pred_rewards [B, T−C]
    """
    c = context_obs.shape[1]
    ctx_embeds = model.encode_seq(context_obs)
    ctx_out = model.rssm.observe(ctx_embeds, actions[:, :c], sample=False)
    h_c, z_c = ctx_out["h"][:, -1], ctx_out["z"][:, -1]

    imag_feats = model.rssm.imagine(h_c, z_c, actions[:, c:], sample=sample, generator=generator)
    return dict(
        context_recon=model.decode_seq(ctx_out["features"]),
        pred_obs=model.decode_seq(imag_feats),
        pred_rewards=model.reward_head(imag_feats),
    )


def multi_step_kl(rssm, obs_out, actions, max_d=8, detach_posterior=True):
    """计算距离 d = 1..max_d 的隐空间多步 KL（CONCEPT.md 第 8.3、8.5 节）。

    返回列表 kls_by_d，第 d−1 个元素的形状为 [B, T − d + 1]（d = 1 时为 [B, T]）。
    全程只在 230 维隐空间运行 D − 1 次批量 GRU 更新，不调用编码器或解码器。
    """
    kl_1 = gaussian_kl(obs_out["post_mean"], obs_out["post_std"], obs_out["prior_mean"], obs_out["prior_std"])
    kls = [kl_1]
    if max_d <= 1:
        return kls

    # 从每个时刻 t = 0..T-2 的后验状态出发，先走第 1 步想象到达时刻 1..T-1
    b, seq_len, _ = actions.shape
    h_curr = obs_out["h"][:, :-1].reshape(-1, H_DIM)
    z_curr = obs_out["z"][:, :-1].reshape(-1, Z_DIM)
    h_curr, z_curr, _ = rssm.imagine_step(h_curr, z_curr, actions[:, 1:].reshape(-1, ACTION_DIM), sample=True)
    h_curr = h_curr.view(b, seq_len - 1, H_DIM)
    z_curr = z_curr.view(b, seq_len - 1, Z_DIM)

    for d in range(2, min(max_d, seq_len - 1) + 1):
        # 丢掉末尾越界的起点，往前再迈一步，落点对应时刻下标 d..T-1
        h_in = h_curr[:, :-1].reshape(-1, H_DIM)
        z_in = z_curr[:, :-1].reshape(-1, Z_DIM)
        act_in = actions[:, d:].reshape(-1, ACTION_DIM)
        h_next, z_next, (pm_d, ps_d) = rssm.imagine_step(h_in, z_in, act_in, sample=True)

        t_rem = seq_len - d
        h_curr = h_next.view(b, t_rem, H_DIM)
        z_curr = z_next.view(b, t_rem, Z_DIM)
        pm_d = pm_d.view(b, t_rem, Z_DIM)
        ps_d = ps_d.view(b, t_rem, Z_DIM)

        qm = obs_out["post_mean"][:, d:].detach() if detach_posterior else obs_out["post_mean"][:, d:]
        qs = obs_out["post_std"][:, d:].detach() if detach_posterior else obs_out["post_std"][:, d:]
        kls.append(gaussian_kl(qm, qs, pm_d, ps_d))
    return kls


# ==================== 第三部分：训练带 latent overshooting 的对照模型 ====================


def train_overshooting_model(replay, n_iters, batch_size, chunk_len, max_d=6, beta_os=0.1):
    """在标准世界模型损失上附加 d = 2..max_d 的 latent overshooting 项。"""
    torch.manual_seed(SEED)
    model = WorldModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(SEED)
    t0 = time.time()

    for it in range(1, n_iters + 1):
        obs, act, rew = replay.sample(batch_size, chunk_len, DEVICE, generator=gen)
        embeds = model.encode_seq(obs)
        out = model.rssm.observe(embeds, act)
        recon = model.decode_seq(out["features"])
        pred_rew = model.reward_head(out["features"])

        recon_loss = 0.5 * (recon - obs).pow(2).sum(dim=(2, 3, 4))
        rew_loss = 0.5 * (pred_rew - rew).pow(2)
        kls = multi_step_kl(model.rssm, out, act, max_d=max_d, detach_posterior=True)
        kl_1_loss = kls[0].clamp_min(FREE_NATS)
        os_loss = sum(k.clamp_min(FREE_NATS).mean() for k in kls[1:]) / max(1, len(kls) - 1)

        loss = (recon_loss + rew_loss + kl_1_loss).mean() + beta_os * os_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
        optimizer.step()

        if it % max(1, n_iters // 4) == 0 or it == 1:
            print(f"    [Overshooting D={max_d}] 步 {it:>4}/{n_iters}  重建 {recon_loss.mean().item():7.2f}  "
                  f"1步KL {kls[0].mean().item():5.2f}  {max_d}步KL {kls[-1].mean().item():5.2f}  用时 {time.time() - t0:5.1f}s")
    return model, time.time() - t0


# ==================== 第四部分：评估、可视化与对比视频 ====================


@torch.no_grad()
def evaluate_open_loop(models, val_data, horizon, max_kl_d=15):
    obs_all = preprocess_obs(torch.from_numpy(val_data["obs"][:, 1 : 1 + CONTEXT_LEN + horizon]).to(DEVICE))
    act_all = torch.from_numpy(val_data["actions"][:, 1 : 1 + CONTEXT_LEN + horizon]).to(DEVICE)
    rew_all = torch.from_numpy(val_data["rewards"][:, 1 : 1 + CONTEXT_LEN + horizon]).to(DEVICE)

    target_obs = obs_all[:, CONTEXT_LEN:]
    target_rew = rew_all[:, CONTEXT_LEN:]
    copy_last_mse = (obs_all[:, CONTEXT_LEN - 1 : CONTEXT_LEN] - target_obs).pow(2).mean(dim=(0, 2, 3, 4)).cpu().numpy()

    results = {}
    for name, model in models.items():
        gen = torch.Generator(device=DEVICE).manual_seed(SEED)
        pred_sample = open_loop_predict(model, obs_all[:, :CONTEXT_LEN], act_all, sample=True, generator=gen)
        pred_mean = open_loop_predict(model, obs_all[:, :CONTEXT_LEN], act_all, sample=False)

        pix_mse_sample = (pred_sample["pred_obs"] - target_obs).pow(2).mean(dim=(0, 2, 3, 4)).cpu().numpy()
        pix_mse_mean = (pred_mean["pred_obs"] - target_obs).pow(2).mean(dim=(0, 2, 3, 4)).cpu().numpy()
        rew_rmse = (pred_mean["pred_rewards"] - target_rew).pow(2).mean(dim=0).sqrt().cpu().numpy()

        torch.manual_seed(SEED)
        out_full = model.rssm.observe(model.encode_seq(obs_all), act_all, sample=True)
        kls_d = [k.mean().item() for k in multi_step_kl(model.rssm, out_full, act_all, max_d=max_kl_d)]

        results[name] = dict(
            pix_sample=pix_mse_sample, pix_mean=pix_mse_mean, rew_rmse=rew_rmse, kls_d=np.array(kls_d),
            sample_out=pred_sample, mean_out=pred_mean,
        )

    steps_show = [1, 5, 10, 15, min(30, horizon), horizon]
    steps_show = sorted(set(steps_show))
    print("\n  开环预测像素 MSE（均值解码）与奖励 RMSE 随步数 d 的变化：")
    header = f"  {'模型 / 基线':<22}" + "".join(f"{f'd={d}':>10}" for d in steps_show)
    print(header)
    for name, r in results.items():
        print(f"  {name + '（像素MSE）':<20}" + "".join(f"{r['pix_mean'][d - 1]:>10.5f}" for d in steps_show))
    print(f"  {'复制上下文最后一帧':<18}" + "".join(f"{copy_last_mse[d - 1]:>10.5f}" for d in steps_show))
    for name, r in results.items():
        print(f"  {name + '（奖励RMSE）':<20}" + "".join(f"{r['rew_rmse'][d - 1]:>10.2f}" for d in steps_show))

    # 图 1：单条序列的帧条带对比（真实 vs 先验采样解码 vs 先验均值解码）
    std_res = results["标准一步 ELBO"]
    pick_d = [1, 4, 8, 12, min(20, horizon), horizon]
    true_imgs = postprocess_obs(target_obs[0])
    samp_imgs = postprocess_obs(std_res["sample_out"]["pred_obs"][0])
    mean_imgs = postprocess_obs(std_res["mean_out"]["pred_obs"][0])

    fig, axes = plt.subplots(3, len(pick_d), figsize=(2.2 * len(pick_d), 5.6))
    for col, d in enumerate(pick_d):
        for row, arr in enumerate((true_imgs, samp_imgs, mean_imgs)):
            axes[row, col].imshow(arr[d - 1])
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].grid(False)
        axes[0, col].set_title(f"开环 d = {d}", fontsize=9.5)
    for row, label in enumerate(["真实未来", "先验采样解码\n（一种可能）", "先验均值解码\n（平均未来）"]):
        axes[row, 0].set_ylabel(label, fontsize=9)
    fig.suptitle(f"给前 {CONTEXT_LEN} 帧建立信念后，不看图像连续预测未来", fontsize=11)
    save_figure(fig, OUT / "06-open-loop-frames.png")

    # 图 2：标准 ELBO vs Latent Overshooting 的三条定量曲线
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    d_axis = np.arange(1, horizon + 1)
    for name, r in results.items():
        axes[0].plot(d_axis, r["pix_mean"], linewidth=2, label=f"{name}（均值解码）")
        axes[1].plot(d_axis, r["rew_rmse"], linewidth=2, label=name)
        axes[2].plot(np.arange(1, len(r["kls_d"]) + 1), r["kls_d"], "o-", markersize=4, linewidth=2, label=name)
    axes[0].plot(d_axis, copy_last_mse, "k--", alpha=0.6, label="基线：停在最后一帧")
    axes[0].set_xlabel("开环预测步数 d")
    axes[0].set_ylabel("单像素均方误差")
    axes[0].set_title("像素开环预测误差")
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("开环预测步数 d")
    axes[1].set_ylabel("奖励 RMSE")
    axes[1].set_title("奖励头开环预测误差（规划真正用的量）")
    axes[1].legend(fontsize=8.5)

    axes[2].set_xlabel("预测距离 d")
    axes[2].set_ylabel("多步 KL（nat）")
    axes[2].set_title("隐空间多步 KL 随距离变化")
    axes[2].legend(fontsize=8.5)
    save_figure(fig, OUT / "06-overshooting-comparison.png")

    # 视频：左列真实画面、中列先验采样、右列先验均值（前 C 帧加青色边框表示有观测输入）
    ctx_true = postprocess_obs(obs_all[0, :CONTEXT_LEN])
    ctx_rec = postprocess_obs(std_res["mean_out"]["context_recon"][0])
    seq_true = np.concatenate([ctx_true, true_imgs], axis=0)
    seq_samp = np.concatenate([ctx_rec, samp_imgs], axis=0)
    seq_mean = np.concatenate([ctx_rec, mean_imgs], axis=0)

    video_frames = []
    scale = 2
    for t_idx in range(len(seq_true)):
        panels = []
        is_ctx = t_idx < CONTEXT_LEN
        border_color = np.array([0, 210, 210], dtype=np.uint8) if is_ctx else np.array([220, 80, 60], dtype=np.uint8)
        for panel in (seq_true[t_idx], seq_samp[t_idx], seq_mean[t_idx]):
            u8 = (panel * 255.0).clip(0, 255).astype(np.uint8)
            big = cv2.resize(u8, (128, 128), interpolation=cv2.INTER_NEAREST)
            big[:3] = big[-3:] = big[:, :3] = big[:, -3:] = border_color
            panels.append(big)
        spacer = np.full((128, 6, 3), 255, dtype=np.uint8)
        row_img = np.concatenate([panels[0], spacer, panels[1], spacer, panels[2]], axis=1)
        video_frames.append(row_img)
    save_video(video_frames, OUT / "06-open-loop.mp4", fps=10)
    print("    （视频三栏从左到右：真实画面 ｜ 先验采样开环 ｜ 先验均值开环；青框 = 有观测热身，红框 = 纯想象）")

    return results, copy_last_mse


# ==================== 第五部分：练习与思考 ====================


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把上下文长度 CONTEXT_LEN 从 10 改成 1 重跑开环预测，误差曲线会发生什么变化？为什么？

练习 1 参考答案：
    只看 1 帧时，模型知道起始角度但不知道起始角速度，第 1 步起的开环误差就会明显变大，
    均值解码的摆杆会迅速散开成一片扇形模糊（同时向顺时针和逆时针两种可能下注）。
    至少给 2–3 帧上下文，后验才能把角速度校准出来。

练习 2：为什么 latent overshooting 里对 d > 1 的后验 q_t 要加 detach()（停止梯度）？

练习 2 参考答案：
    我们的目标是"让多步先验追上真实的滤波后验"，而不是反过来让表示网络为了迎合粗糙的远期预测，
    故意把当前帧的细节抹掉（表示退化）。d = 1 保留双向梯度是为了维持正常的序列 ELBO 正则化。

思考题 1：开环预测到第 30–40 步时，均值解码的摆杆变淡、变模糊，这说明解码器坏了吗？

思考题 1 参考答案：
    恰恰说明模型对不确定性的表达是诚实的。几十步之后微小的初始速度误差已经累积成显著的相位差，
    隐状态的分布变宽，位置分散的潜变量取均值解码自然呈现模糊扇形，而不是自信地画出一根位置完全错误的清晰摆杆。

思考题 2：既然视野 15 步之内的奖励开环预测已经比较准，这对下一节的 CEM 规划意味着什么？

思考题 2 参考答案：
    规划器每次决策只需要看未来 H = 12–15 个决策步（对应 24–30 个环境步），然后只执行第 1 个动作，
    下一步立刻用新观测重新校准信念。所以模型不需要在 100 步上都不漂移，只要在 H 步之内把奖励排对先后就够了。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    if DEVICE.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    quick = is_quick()
    if quick:
        print("（冒烟模式：步数缩小，仅验证开环预测与 overshooting 流程跑通）\n")

    n_episodes, n_val = (8, 2) if quick else (60, 10)
    n_iters_os, batch_size, chunk_len = (20, 4, 12) if quick else (500, 12, 25)
    horizon, max_d, max_kl_d = (12, 4, 8) if quick else (40, 6, 15)

    introduction()
    verify_hand_calculation()
    ensure_artifacts()

    with np.load(DATASET_PATH) as f:
        data = {k: f[k] for k in f.files}
    n_train = n_episodes - n_val
    replay = SequenceReplay()
    for ep in range(n_train):
        replay.add_episode(data["obs"][ep], data["actions"][ep], data["rewards"][ep])
    val_data = {k: data[k][n_train:] for k in ("obs", "actions", "rewards", "states")}

    std_model = WorldModel().to(DEVICE)
    std_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    std_model.eval()

    print("\n" + "=" * 60)
    print(f"第三部分：训练带 latent overshooting（D = {max_d}）的对照模型")
    print("=" * 60)
    os_model, elapsed_os = train_overshooting_model(replay, n_iters_os, batch_size, chunk_len, max_d=max_d)
    os_model.eval()
    print(f"  Overshooting 模型训练完成，耗时 {elapsed_os:.1f}s。")

    print("\n" + "=" * 60)
    print("第四部分：开环预测对比评估")
    print("=" * 60)
    evaluate_open_loop({"标准一步 ELBO": std_model, f"Overshooting (D={max_d})": os_model},
                       val_data, horizon=horizon, max_kl_d=max_kl_d)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：07-latent-cem-planning.py（把 CEM 规划器搬进隐空间）")
    print("=" * 60)


if __name__ == "__main__":
    main()
