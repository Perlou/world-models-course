"""
Phase 9 · 第 07 课：隐动作模型（LAM）—— 从无标注视频里学出离散动作
==================================================================

学习目标：
    1. 实现不对称结构的隐动作模型：编码器看 (o_t, o_{t+1})，经极窄 VQ 瓶颈压缩成 M 个隐动作之一，
       解码器只看 (o_t, ā_t) 重建下一帧 o_{t+1}
    2. 全程不喂任何动作标签，只在带真实 3 动作（左/停/右）的 PixelCatchEnv 视频上训练，
       统计隐动作码与真实动作的对应关系表（纯度）与 ΔPSNR 可控性指标
    3. 对比码本大小 M ∈ {2, 4, 8, 64}：验证过窄会合并动作、过宽会泄露未来画面导致可控性下降

对应概念文档：
    src/phase-09-diffusion-video-world-models/CONCEPT.md 第 7 节

运行方式：
    uv run python src/phase-09-diffusion-video-world-models/07-latent-action-models.py
    WM_QUICK=1 uv run python src/phase-09-diffusion-video-world-models/07-latent-action-models.py

预计运行时间：完整约 90 秒（M1 Max CPU），冒烟约 10 秒
"""

import importlib.util
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_DIR)
_spec_env = importlib.util.spec_from_file_location(
    "p8_05_lam_env", os.path.join(_SRC, "phase-08-transformer-world-models", "05-iris.py")
)
_mod_env = importlib.util.module_from_spec(_spec_env)
_spec_env.loader.exec_module(_mod_env)
PixelCatchEnv = _mod_env.PixelCatchEnv

plt = setup_matplotlib()
OUT = output_dir("phase-09")
SEED = 0


# ==================== 第一部分：隐动作模型（Encoder + 小码本 VQ + Decoder） ====================


class SmallActionVQ(nn.Module):
    """CONCEPT.md 第 7.2 节：极窄 VQ 码本（M 个离散隐动作，straight-through 传梯度）。"""

    def __init__(self, num_codes=4, dim=16, beta=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, dim)
        nn.init.uniform_(self.codebook.weight, -0.5, 0.5)

    def forward(self, e):
        # e: [B, D]
        dists = torch.cdist(e, self.codebook.weight)  # [B, M]
        idx = dists.argmin(dim=-1)                    # [B]
        c = self.codebook(idx)
        cb_loss = F.mse_loss(c, e.detach())
        commit_loss = F.mse_loss(e, c.detach())
        c_st = e + (c - e).detach()
        return c_st, idx, cb_loss + self.beta * commit_loss


class LatentActionModel(nn.Module):
    """编码器看 (o_t, o_{t+1})；解码器只能看 (o_t, ā_t) 预测 o_{t+1}。"""

    def __init__(self, num_codes=4, act_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 24, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(24, 48, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(48, 48, 3, 1, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(48 * 2 * 2, act_dim),
        )
        self.vq = SmallActionVQ(num_codes=num_codes, dim=act_dim)
        # 解码器：用空间特征 + 隐动作条件预测帧差 Δo（残差式，迫使码本聚焦运动而非静态背景）
        self.dec_enc = nn.Sequential(
            nn.Conv2d(3, 24, 4, 2, 1), nn.SiLU(),   # 16
            nn.Conv2d(24, 48, 4, 2, 1), nn.SiLU(),  # 8
        )
        self.act_film = nn.Linear(act_dim, 48 * 2)
        self.dec_up = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(48, 24, 4, 2, 1), nn.SiLU(),  # 16
            nn.ConvTranspose2d(24, 3, 4, 2, 1),              # 32
        )

    def encode_action(self, o_t, o_next):
        e = self.encoder(torch.cat([o_t, o_next], dim=1))
        return self.vq(e)

    def decode_next(self, o_t, c_vec):
        h = self.dec_enc(o_t)
        g, b = self.act_film(c_vec).chunk(2, dim=-1)
        h = h * (1.0 + g[:, :, None, None]) + b[:, :, None, None]
        return (o_t + self.dec_up(h)).clamp(0.0, 1.0)

    def forward(self, o_t, o_next):
        c_st, idx, vq_loss = self.encode_action(o_t, o_next)
        pred_next = self.decode_next(o_t, c_st)
        recon_loss = F.mse_loss(pred_next, o_next)
        return pred_next, idx, recon_loss + vq_loss, recon_loss


def collect_unlabeled_pairs(n_episodes=60, seed=0):
    """采集 PixelCatchEnv 的相邻帧对；保留真实动作仅用于训练后评估，训练过程完全不可见。"""
    env = PixelCatchEnv(seed=seed)
    o_ts, o_nxts, true_acts = [], [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            a = int(np.random.randint(0, 3))
            nxt, _, done = env.step(a)
            o_ts.append(obs); o_nxts.append(nxt); true_acts.append(a)
            obs = nxt
    return (
        torch.from_numpy(np.asarray(o_ts, dtype=np.float32)),
        torch.from_numpy(np.asarray(o_nxts, dtype=np.float32)),
        np.asarray(true_acts, dtype=np.int64),
    )


def psnr01(pred, tgt):
    mse = F.mse_loss(pred.clamp(0, 1), tgt.clamp(0, 1)).item()
    return float(-10.0 * math.log10(max(mse, 1e-8)))


@torch.no_grad()
def evaluate_lam(model, o_t, o_next, true_acts):
    """计算 CONCEPT.md 第 7.4/7.6 节：对应关系纯度、ΔPSNR 可控性、随机施码时挡板运动方向一致性。"""
    model.eval()
    c_st, idx, _ = model.encode_action(o_t, o_next)
    pred_inf = model.decode_next(o_t, c_st)

    # 随机隐动作对照（固定测试种子）
    gen = torch.Generator().manual_seed(123)
    rand_idx = torch.randint(0, model.vq.num_codes, (len(o_t),), generator=gen)
    pred_rand = model.decode_next(o_t, model.vq.codebook(rand_idx))

    psnr_inf = psnr01(pred_inf, o_next)
    psnr_rand = psnr01(pred_rand, o_next)
    d_psnr = psnr_inf - psnr_rand

    # 对应关系表 [3, M] 与聚类纯度（每个隐码对应最多的真实动作占比）
    idx_np = idx.numpy()
    M = model.vq.num_codes
    table = np.zeros((3, M), dtype=np.int64)
    for ta, la in zip(true_acts, idx_np):
        table[ta, la] += 1
    purity = float(table.max(axis=0).sum() / max(table.sum(), 1))
    used_codes = int((table.sum(axis=0) > 0).sum())

    # 动作一致性：固定同一个常用码，在不同初始帧上施加该码，挡板位移符号的一致率
    # PixelCatchEnv 挡板位于最底部行，绿色通道高（自适应图像宽高）
    def paddle_x(frames):
        W = frames.shape[-1]
        green = frames[:, 1, -2:, :].mean(dim=1)  # [B, W]
        xs = torch.arange(W, dtype=torch.float32).unsqueeze(0)
        return (green * xs).sum(dim=1) / green.sum(dim=1).clamp_min(1e-4)

    x0 = paddle_x(o_t[:128])
    consist_list = []
    top_codes = np.argsort(-table.sum(axis=0))[: min(M, 8)]
    for c_id in top_codes:
        if table[:, c_id].sum() < 5:
            continue
        c_vec = model.vq.codebook(torch.full((128,), int(c_id), dtype=torch.long))
        p_c = model.decode_next(o_t[:128], c_vec)
        dx = (paddle_x(p_c) - x0).numpy()
        # 归类为左 (<-0.25) / 停 / 右 (>+0.25)
        signs = np.where(dx < -0.25, -1, np.where(dx > 0.25, 1, 0))
        _, cnts = np.unique(signs, return_counts=True)
        consist_list.append(float(cnts.max() / len(signs)))
    consistency = float(np.mean(consist_list)) if consist_list else 0.0
    return {
        "psnr_inf": psnr_inf,
        "psnr_rand": psnr_rand,
        "delta_psnr": d_psnr,
        "purity": purity,
        "used_codes": used_codes,
        "consistency": consistency,
        "table": table,
    }


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    print("=" * 60)
    print("第一部分：手算核对 CONCEPT.md 第 7.6 节 ΔPSNR")
    print("=" * 60)
    dp_hand = (-10.0 * math.log10(0.002)) - (-10.0 * math.log10(0.004))
    print(f"  MSE 0.002 vs 0.004：ΔPSNR = {dp_hand:.2f} dB（手算 3.01 dB，误差差一倍对应 3 dB ✓）")

    print("\n" + "=" * 60)
    print("第二部分：无标注 PixelCatchEnv 视频上训练 LAM（码本大小 M ∈ {2, 4, 8, 64}）")
    print("=" * 60)
    tr_ot, tr_on, _ = collect_unlabeled_pairs(n_episodes=20 if quick else 80, seed=SEED)
    te_ot, te_on, te_ta = collect_unlabeled_pairs(n_episodes=10 if quick else 30, seed=SEED + 99)
    print(f"  训练帧对：{len(tr_ot)}（无动作标签），测试帧对：{len(te_ot)}（含真实 3 动作仅用于事后对照）")

    M_list = [2, 4, 8, 64]
    n_iters = 100 if quick else 500
    results = {}
    print(f"\n  {'码本 M':>6s} | {'活跃码':>6s} | {'推断PSNR':>8s} | {'随机PSNR':>8s} | {'ΔPSNR':>7s} | {'动作纯度':>8s} | {'方向一致率':>10s}")
    for M in M_list:
        seed_everything(SEED)
        lam = LatentActionModel(num_codes=M, act_dim=16)
        opt = torch.optim.Adam(lam.parameters(), lr=2e-3)
        for _ in range(n_iters):
            idx = torch.randint(0, len(tr_ot), (64,))
            opt.zero_grad()
            _, _, loss, _ = lam(tr_ot[idx], tr_on[idx])
            loss.backward()
            opt.step()
        m = evaluate_lam(lam, te_ot, te_on, te_ta)
        results[M] = m
        print(
            f"  {M:>6d} | {m['used_codes']:>6d} | {m['psnr_inf']:8.2f} | {m['psnr_rand']:8.2f} | "
            f"{m['delta_psnr']:+7.2f} | {m['purity']*100:7.1f}% | {m['consistency']*100:9.1f}%"
        )

    print("\n  M = 4 时「真实动作 × 隐动作码」对应表（行：左 / 停 / 右）：")
    tbl4 = results[4]["table"]
    act_names = ["左 (0)", "停 (1)", "右 (2)"]
    print("  " + f"{'真实动作':8s} | " + " | ".join(f"码 {j:2d}" for j in range(4)))
    for i, nm in enumerate(act_names):
        print("  " + f"{nm:8s} | " + " | ".join(f"{tbl4[i, j]:4d}" for j in range(4)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(M_list, [results[m]["delta_psnr"] for m in M_list], "o-")
    axes[0].set_xscale("log", base=2); axes[0].set_title("可控性 ΔPSNR 随码本 M 变化"); axes[0].set_xlabel("码本大小 M"); axes[0].set_ylabel("ΔPSNR (dB)")

    axes[1].plot(M_list, [results[m]["purity"] * 100 for m in M_list], "o-", label="动作纯度 (%)")
    axes[1].plot(M_list, [results[m]["consistency"] * 100 for m in M_list], "s--", label="跨状态方向一致率 (%)")
    axes[1].set_xscale("log", base=2); axes[1].set_title("隐动作质量 vs 码本宽度"); axes[1].set_xlabel("码本大小 M"); axes[1].legend()

    im = axes[2].imshow(tbl4, cmap="Blues")
    axes[2].set_yticks([0, 1, 2]); axes[2].set_yticklabels(["左", "停", "右"])
    axes[2].set_xticks(range(4)); axes[2].set_xticklabels([f"码{j}" for j in range(4)])
    axes[2].set_title("M=4 真实动作 vs 隐动作热力图")
    for r in range(3):
        for c in range(4):
            axes[2].text(c, r, str(tbl4[r, c]), ha="center", va="center", color="black")
    fig.tight_layout()
    save_figure(fig, OUT / "07-latent-action-models.png")

    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：08-mini-genie.py")


if __name__ == "__main__":
    main()
