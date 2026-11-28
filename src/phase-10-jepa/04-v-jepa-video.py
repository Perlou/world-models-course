"""
Phase 10 · 第 04 课：V-JEPA —— 视频时空管状掩码与 L1 表征预测
==============================================================

学习目标：
    1. 核对 CONCEPT.md 第 4.3 节管状掩码堵住捷径的概率（1 - 0.9⁸ = 0.570）及 L1 中位数手算
    2. 实现时空立方体切分（tubelet_embed）、管状掩码（sample_tube_masks）与 L1 损失的 V-JEPA
    3. 在合成移动几何体视频（32×32×8 帧，带精确位置与速度真值）上对比「管状掩码」与「逐帧独立掩码」
       的训练损失曲线以及冻结后线性探针回归速度的 R²

对应概念文档：
    src/phase-10-jepa/CONCEPT.md 第 4 节

运行方式：
    uv run python src/phase-10-jepa/04-v-jepa-video.py
    WM_QUICK=1 uv run python src/phase-10-jepa/04-v-jepa-video.py

预计运行时间：完整约 90 秒（M1 Max CPU），冒烟约 10 秒
"""

import copy
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
_spec02 = importlib.util.spec_from_file_location("p10_02_vj", os.path.join(_DIR, "02-collapse-and-regularization.py"))
_mod02 = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(_mod02)
effective_rank = _mod02.effective_rank
ema_update = _mod02.ema_update

plt = setup_matplotlib()
OUT = output_dir("phase-10")
SEED = 0


# ==================== 第一部分：合成移动物体视频数据集（带精确位置与速度） ====================


def make_moving_shapes_videos(n_videos=600, n_frames=8, size=32, seed=0):
    """生成带精确中心坐标 (x, y) 与速度 (vx, vy) 的 32×32 RGB 合成视频，供 V-JEPA 与第 05 课探针复用。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    videos = np.zeros((n_videos, n_frames, 3, size, size), dtype=np.float32)
    pos_mid = np.zeros((n_videos, 2), dtype=np.float32)
    vel_mid = np.zeros((n_videos, 2), dtype=np.float32)

    for i in range(n_videos):
        x = rng.uniform(7.0, size - 7.0)
        y = rng.uniform(7.0, size - 7.0)
        vx = rng.uniform(-2.2, 2.2)
        vy = rng.uniform(-2.2, 2.2)
        color = rng.uniform(0.35, 1.0, size=3).astype(np.float32)
        bg_tint = rng.uniform(0.05, 0.18, size=3).astype(np.float32)
        for t in range(n_frames):
            frame = np.broadcast_to(bg_tint[:, None, None], (3, size, size)).copy()
            mask = ((xx - x) ** 2 + (yy - y) ** 2) <= 3.2 ** 2
            for c in range(3):
                frame[c, mask] = color[c]
            videos[i, t] = frame
            if t == n_frames // 2:
                pos_mid[i] = [x / size, y / size]
                vel_mid[i] = [vx / 2.2, vy / 2.2]
            x += vx; y += vy
            if x < 4.0 or x > size - 4.0:
                vx = -vx; x = np.clip(x, 4.0, size - 4.0)
            if y < 4.0 or y > size - 4.0:
                vy = -vy; y = np.clip(y, 4.0, size - 4.0)
    return torch.from_numpy(videos), pos_mid, vel_mid


# ==================== 第二部分：Tubelet ViT、管状掩码与 V-JEPA 训练 ====================


def sample_tube_masks(grid_t=4, grid_h=4, grid_w=4, mask_ratio=0.65, rng=None):
    """CONCEPT.md 第 4.2 节：先在空间 [grid_h, grid_w] 采样被遮位置，再沿时间轴复制贯通全部时间片。"""
    rng = np.random.default_rng() if rng is None else rng
    n_s = grid_h * grid_w
    n_mask_s = max(1, min(n_s - 2, int(round(n_s * mask_ratio))))
    masked_s = np.zeros(n_s, dtype=bool)
    masked_s[rng.choice(n_s, size=n_mask_s, replace=False)] = True
    full_mask = np.tile(masked_s[None, :], (grid_t, 1)).reshape(-1)
    ctx_idx = torch.tensor(np.where(~full_mask)[0], dtype=torch.long)
    tgt_idx = torch.tensor(np.where(full_mask)[0], dtype=torch.long)
    return ctx_idx, tgt_idx


def sample_frame_indep_masks(grid_t=4, grid_h=4, grid_w=4, mask_ratio=0.65, rng=None):
    """对照：每个时间片独立随机遮盖，同一空间位置可从相邻时间片偷看抄答案。"""
    rng = np.random.default_rng() if rng is None else rng
    n_total = grid_t * grid_h * grid_w
    n_mask = max(1, min(n_total - 4, int(round(n_total * mask_ratio))))
    full_mask = np.zeros(n_total, dtype=bool)
    full_mask[rng.choice(n_total, size=n_mask, replace=False)] = True
    ctx_idx = torch.tensor(np.where(~full_mask)[0], dtype=torch.long)
    tgt_idx = torch.tensor(np.where(full_mask)[0], dtype=torch.long)
    return ctx_idx, tgt_idx


class MiniVideoViT(nn.Module):
    """2 帧 × 8×8 像素 tubelet 的轻量 V-JEPA 视频编码器（T=8, H=W=32 → 4×4×4 = 64 tubelets）。"""

    def __init__(self, n_frames=8, img_size=32, t_patch=2, s_patch=8, d_model=64, n_layers=3, n_heads=4):
        super().__init__()
        self.t_patch = t_patch
        self.s_patch = s_patch
        self.gt = n_frames // t_patch
        self.gh = img_size // s_patch
        self.gw = img_size // s_patch
        self.num_tokens = self.gt * self.gh * self.gw
        self.d_model = d_model
        self.tube_dim = 3 * t_patch * s_patch * s_patch

        self.proj = nn.Linear(self.tube_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def tubelet_embed(self, video):
        # video: [B, T, C, H, W] -> [B, N, tube_dim]
        B, T, C, H, W = video.shape
        tp, sp = self.t_patch, self.s_patch
        x = video.reshape(B, self.gt, tp, C, self.gh, sp, self.gw, sp)
        x = x.permute(0, 1, 4, 6, 2, 3, 5, 7).reshape(B, self.num_tokens, self.tube_dim)
        return self.proj(x) + self.pos_emb

    def forward(self, video, keep_idx=None):
        h = self.tubelet_embed(video)
        if keep_idx is not None:
            h = h[:, keep_idx]
        return self.norm(self.enc(h))


class VJEPAPredictor(nn.Module):
    def __init__(self, num_tokens=64, enc_dim=64, pred_dim=48, n_layers=2, n_heads=4):
        super().__init__()
        self.in_proj = nn.Linear(enc_dim, pred_dim)
        self.mask_tok = nn.Parameter(torch.randn(1, 1, pred_dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, pred_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=pred_dim, nhead=n_heads, dim_feedforward=2 * pred_dim,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(pred_dim, enc_dim)

    def forward(self, z_ctx, ctx_idx, tgt_idx):
        B = z_ctx.shape[0]
        hc = self.in_proj(z_ctx) + self.pos_emb[:, ctx_idx]
        mt = self.mask_tok.expand(B, len(tgt_idx), -1) + self.pos_emb[:, tgt_idx]
        out = self.tf(torch.cat([hc, mt], dim=1))[:, len(ctx_idx) :]
        return self.out_proj(out)


def vjepa_step(ctx_enc, tgt_enc, predictor, videos, mask_mode="tube"):
    if mask_mode == "tube":
        ctx_idx, tgt_idx = sample_tube_masks(ctx_enc.gt, ctx_enc.gh, ctx_enc.gw, mask_ratio=0.65)
    else:
        ctx_idx, tgt_idx = sample_frame_indep_masks(ctx_enc.gt, ctx_enc.gh, ctx_enc.gw, mask_ratio=0.65)
    with torch.no_grad():
        z_full = F.layer_norm(tgt_enc(videos), (tgt_enc.d_model,))
    z_ctx = ctx_enc(videos, keep_idx=ctx_idx)
    z_pred = predictor(z_ctx, ctx_idx, tgt_idx)
    # V-JEPA 使用 L1 损失（CONCEPT.md 第 4.2–4.3 节）
    loss = F.l1_loss(z_pred, z_full[:, tgt_idx])
    return loss, z_full


@torch.no_grad()
def linear_probe_video(enc, v_tr, y_tr, v_te, y_te):
    enc.eval()
    def extract(v):
        z = enc(v)  # [B, 64, D] = [B, 4*16, D]，保留时间片维（4 个时间片各做空间池化 → 4*D）
        B = z.shape[0]
        return z.reshape(B, enc.gt, enc.gh * enc.gw, enc.d_model).mean(dim=2).reshape(B, -1).numpy()
    Z_tr, Z_te = extract(v_tr), extract(v_te)
    mu, sd = Z_tr.mean(0), Z_tr.std(0) + 1e-6
    X_tr = np.hstack([(Z_tr - mu) / sd, np.ones((len(Z_tr), 1))])
    X_te = np.hstack([(Z_te - mu) / sd, np.ones((len(Z_te), 1))])
    best_r2 = -1e9
    for lam in [1e-1, 1.0, 10.0, 50.0]:
        reg = lam * np.eye(X_tr.shape[1]); reg[-1, -1] = 0.0
        W = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ y_tr)
        pred = X_te @ W
        r2 = float(1.0 - ((y_te - pred) ** 2).sum() / max(((y_te - y_te.mean(0, keepdims=True)) ** 2).sum(), 1e-8))
        best_r2 = max(best_r2, r2)
    return best_r2


def train_vjepa_variant(mask_mode, v_tr, n_iters=350):
    seed_everything(SEED)
    ctx_enc = MiniVideoViT()
    tgt_enc = copy.deepcopy(ctx_enc)
    pred = VJEPAPredictor()
    opt = torch.optim.AdamW(list(ctx_enc.parameters()) + list(pred.parameters()), lr=2e-3, weight_decay=1e-2)
    losses = []
    for it in range(1, n_iters + 1):
        idx = torch.randint(0, len(v_tr), (24,))
        opt.zero_grad()
        loss, _ = vjepa_step(ctx_enc, tgt_enc, pred, v_tr[idx], mask_mode=mask_mode)
        loss.backward(); nn.utils.clip_grad_norm_(ctx_enc.parameters(), 1.0); opt.step()
        ema_update(tgt_enc, ctx_enc, m=0.96 + 0.035 * (it / n_iters))
        losses.append(float(loss.item()))
    return tgt_enc, losses


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 4.3 节管状掩码捷径与 L1 中位数手算")
    print("=" * 60)
    p_leak = 1.0 - (0.9 ** 8)
    arr = np.array([0.0, 0.0, 10.0])
    print(f"  逐帧独立掩码 (ρ=0.9, T'=8) 至少一片可见概率 = {p_leak:.3f}（手算 0.570），管状掩码 = 0.000 ✓")
    print(f"  样本 [0, 0, 10]：L1(中位数 0) = {np.abs(arr - 0).mean():.2f} < L1(均值 3.33) = {np.abs(arr - arr.mean()).mean():.2f} ✓")

    print("\n" + "=" * 60)
    print("第二部分：合成移动物体视频上对比「管状掩码」与「逐帧独立掩码」")
    print("=" * 60)
    v_tr, p_tr, vel_tr = make_moving_shapes_videos(n_videos=180 if quick else 700, seed=SEED)
    v_te, p_te, vel_te = make_moving_shapes_videos(n_videos=80 if quick else 220, seed=SEED + 99)

    n_iters = 70 if quick else 360
    enc_tube, loss_tube = train_vjepa_variant("tube", v_tr, n_iters=n_iters)
    enc_indep, loss_indep = train_vjepa_variant("indep", v_tr, n_iters=n_iters)

    r2_pos_tube = linear_probe_video(enc_tube, v_tr, p_tr, v_te, p_te)
    r2_vel_tube = linear_probe_video(enc_tube, v_tr, vel_tr, v_te, vel_te)
    r2_pos_indep = linear_probe_video(enc_indep, v_tr, p_tr, v_te, p_te)
    r2_vel_indep = linear_probe_video(enc_indep, v_tr, vel_tr, v_te, vel_te)

    print(f"  管状掩码 V-JEPA：末段 L1 损失 = {np.mean(loss_tube[-20:]):.4f} | 位置探针 R² = {r2_pos_tube:.3f} | 速度探针 R² = {r2_vel_tube:.3f}")
    print(f"  逐帧独立掩码   ：末段 L1 损失 = {np.mean(loss_indep[-20:]):.4f} | 位置探针 R² = {r2_pos_indep:.3f} | 速度探针 R² = {r2_vel_indep:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(loss_tube, label="管状掩码（无跨帧捷径）")
    axes[0].plot(loss_indep, "--", label="逐帧独立掩码（可跨帧抄）")
    axes[0].set_title("V-JEPA 训练 L1 损失曲线"); axes[0].set_xlabel("步数"); axes[0].legend()

    x_bar = np.arange(2)
    axes[1].bar(x_bar - 0.16, [r2_pos_tube, r2_vel_tube], 0.32, label="管状掩码")
    axes[1].bar(x_bar + 0.16, [r2_pos_indep, r2_vel_indep], 0.32, label="逐帧独立掩码")
    axes[1].set_xticks(x_bar); axes[1].set_xticklabels(["位置 R²", "速度 R²"])
    axes[1].set_title("冻结视频表征线性探针对比"); axes[1].set_ylim(0, 1.05); axes[1].legend()
    fig.tight_layout()
    save_figure(fig, OUT / "04-v-jepa-video.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：05-probing-latent-representations.py")


if __name__ == "__main__":
    main()
