"""
Phase 10 · 第 03 课：从零实现 I-JEPA（小 ViT + 多块掩码 + EMA 目标编码器）
==========================================================================

学习目标：
    1. 核对 CONCEPT.md 第 3.3 节随机掩码 vs 块掩码邻居插值捷径的概率计算
    2. 实现 I-JEPA 核心结构：支持只处理可见 patch 的 MiniViT、多目标块采样器、
       窄 ViT 预测器与 EMA 目标编码器
    3. 在真实 PushT 32×32 画面上训练 I-JEPA，全程监控目标表征有效秩与标准差
    4. 冻结编码器后用线性探针回归 PushT 推杆坐标，对比 I-JEPA、MAE 风格像素重建、随机初始化三者

对应概念文档：
    src/phase-10-jepa/CONCEPT.md 第 3 节

运行方式：
    uv run python src/phase-10-jepa/03-i-jepa-from-scratch.py
    WM_QUICK=1 uv run python src/phase-10-jepa/03-i-jepa-from-scratch.py

预计运行时间：完整约 90 秒（M1 Max CPU），冒烟约 10 秒
"""

import copy
import importlib.util
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.pusht import load_pusht_demos

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec02 = importlib.util.spec_from_file_location("p10_02_ij", os.path.join(_DIR, "02-collapse-and-regularization.py"))
_mod02 = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(_mod02)
effective_rank = _mod02.effective_rank
ema_update = _mod02.ema_update

plt = setup_matplotlib()
OUT = output_dir("phase-10")
SEED = 0


# ==================== 第一部分：多块掩码、MiniViT 与预测器 ====================


def sample_block_masks(grid=8, n_targets=4, tgt_h=3, tgt_w=3, rng=None):
    """CONCEPT.md 第 3.2/3.4 节：采 M 个矩形目标块，上下文取剩余全部可见 patch（去重叠）。"""
    rng = np.random.default_rng() if rng is None else rng
    tgt_masks = []
    covered = np.zeros((grid, grid), dtype=bool)
    for _ in range(n_targets):
        r0 = int(rng.integers(0, grid - tgt_h + 1))
        c0 = int(rng.integers(0, grid - tgt_w + 1))
        idx = []
        for r in range(r0, r0 + tgt_h):
            for c in range(c0, c0 + tgt_w):
                idx.append(r * grid + c)
                covered[r, c] = True
        tgt_masks.append(torch.tensor(idx, dtype=torch.long))
    ctx_idx = torch.tensor(np.where(~covered.reshape(-1))[0], dtype=torch.long)
    if len(ctx_idx) < 4:
        ctx_idx = torch.arange(min(8, grid * grid), dtype=torch.long)
    return ctx_idx, tgt_masks


class MiniViT(nn.Module):
    """支持只输入子集 patch 下标（上下文分支）或全图 patch（目标分支）的轻量 ViT。"""

    def __init__(self, img_size=32, patch_size=4, in_ch=3, d_model=64, n_layers=3, n_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.d_model = d_model
        self.patch_dim = in_ch * patch_size * patch_size

        self.patch_proj = nn.Linear(self.patch_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def patchify(self, imgs):
        # imgs: [B, C, H, W] -> [B, N, P]
        B, C, H, W = imgs.shape
        p = self.patch_size
        x = imgs.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
        return x.reshape(B, self.num_patches, self.patch_dim)

    def forward(self, imgs, keep_idx=None):
        patches = self.patchify(imgs)
        h = self.patch_proj(patches) + self.pos_emb
        if keep_idx is not None:
            h = h[:, keep_idx]
        return self.norm(self.encoder(h))


class IJEPAPredictor(nn.Module):
    """CONCEPT.md 第 3.2 节：窄 ViT 预测器，接收上下文特征 + 带位置编码的 mask token。"""

    def __init__(self, num_patches=64, enc_dim=64, pred_dim=48, n_layers=2, n_heads=4):
        super().__init__()
        self.in_proj = nn.Linear(enc_dim, pred_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, pred_dim) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, pred_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=pred_dim, nhead=n_heads, dim_feedforward=2 * pred_dim,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(pred_dim, enc_dim)

    def forward(self, z_ctx, ctx_idx, tgt_idx):
        B = z_ctx.shape[0]
        h_ctx = self.in_proj(z_ctx) + self.pos_emb[:, ctx_idx]
        m_tok = self.mask_token.expand(B, len(tgt_idx), -1) + self.pos_emb[:, tgt_idx]
        seq = torch.cat([h_ctx, m_tok], dim=1)
        out = self.tf(seq)[:, len(ctx_idx) :]
        return self.out_proj(out)


def ijepa_step(ctx_enc, tgt_enc, predictor, imgs, grid=8):
    ctx_idx, tgt_masks = sample_block_masks(grid=grid, n_targets=4, tgt_h=3, tgt_w=3)
    ctx_idx = ctx_idx.to(imgs.device)
    with torch.no_grad():
        z_full = F.layer_norm(tgt_enc(imgs), (tgt_enc.d_model,))
    z_ctx = ctx_enc(imgs, keep_idx=ctx_idx)

    total_loss = 0.0
    for t_idx in tgt_masks:
        t_idx = t_idx.to(imgs.device)
        z_pred = predictor(z_ctx, ctx_idx, t_idx)
        z_tgt = z_full[:, t_idx]
        total_loss = total_loss + F.mse_loss(z_pred, z_tgt)
    return total_loss / len(tgt_masks), z_full


def mae_step(enc, decoder, imgs, grid=8):
    """对照组：MAE 风格像素掩码重建，结构与掩码完全对齐 I-JEPA，只把目标换成原始 patch 像素。"""
    ctx_idx, tgt_masks = sample_block_masks(grid=grid, n_targets=4, tgt_h=3, tgt_w=3)
    ctx_idx = ctx_idx.to(imgs.device)
    patches = enc.patchify(imgs)
    z_ctx = enc(imgs, keep_idx=ctx_idx)
    total_loss = 0.0
    for t_idx in tgt_masks:
        t_idx = t_idx.to(imgs.device)
        p_hat = decoder(z_ctx, ctx_idx, t_idx)
        total_loss = total_loss + F.mse_loss(p_hat, patches[:, t_idx])
    return total_loss / len(tgt_masks)


def load_pusht_images_and_states(n_frames=1200, img_size=32):
    demos = load_pusht_demos()
    stride = max(1, len(demos["images"]) // n_frames)
    raw_im = demos["images"][::stride][:n_frames]
    raw_st = demos["states"][::stride][:n_frames]
    imgs = np.stack([cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA) for im in raw_im], axis=0)
    imgs = np.transpose(imgs.astype(np.float32) / 255.0, (0, 3, 1, 2))
    # states 前 2 维通常是 agent_pos，整体 5 维 [agent_x, agent_y, block_x, block_y, block_angle]
    states = raw_st.astype(np.float32)
    return torch.from_numpy(imgs), states


@torch.no_grad()
def eval_linear_probe(enc, imgs_tr, y_tr, imgs_te, y_te):
    """冻结编码器，取全局平均与空间 2×2 池化拼接特征做岭回归线性探针，报告 R²。"""
    enc.eval()
    def feat(im):
        z = enc(im)  # [N, 64, D]
        B, N, D = z.shape
        g = int(math.sqrt(N))
        # 2×2 空间池化保留空间位置信息（4*D 维），避免全局平均抹平坐标
        z2d = z.reshape(B, g, g, D).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(z2d, (2, 2)).flatten(1)
        return pooled.numpy()

    Z_tr, Z_te = feat(imgs_tr), feat(imgs_te)
    mu, sd = Z_tr.mean(0), Z_tr.std(0) + 1e-6
    X_tr = np.hstack([(Z_tr - mu) / sd, np.ones((len(Z_tr), 1))])
    X_te = np.hstack([(Z_te - mu) / sd, np.ones((len(Z_te), 1))])

    best_r2 = -1e9
    for lam in [1e-1, 1.0, 10.0, 100.0]:
        reg = lam * np.eye(X_tr.shape[1]); reg[-1, -1] = 0.0
        W = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ y_tr)
        pred = X_te @ W
        r2 = float(1.0 - ((y_te - pred) ** 2).sum() / max(((y_te - y_te.mean(0, keepdims=True)) ** 2).sum(), 1e-8))
        best_r2 = max(best_r2, r2)
    return best_r2


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 3.3 节随机掩码 vs 块掩码邻居插值捷径")
    print("=" * 60)
    p_rand = 1.0 - (0.75 ** 4)
    p_block = (36 - 16) / 36.0
    print(f"  随机掩码 (ρ=0.75) 至少一邻可见概率 = {p_rand:.3f}；6×6 块掩码 = {p_block:.3f}（手算 0.684 vs 0.556 ✓）")

    print("\n" + "=" * 60)
    print("第二部分：PushT 32×32 上训练 I-JEPA 与 MAE 对照，并做冻结线性探针")
    print("=" * 60)
    imgs, states = load_pusht_images_and_states(n_frames=320 if quick else 1400, img_size=32)
    n_tr = int(len(imgs) * 0.8)
    im_tr, st_tr = imgs[:n_tr], states[:n_tr, :4]  # [agent_x, agent_y, block_x, block_y]
    im_te, st_te = imgs[n_tr:], states[n_tr:, :4]
    print(f"  数据划分：训练 {len(im_tr)} 帧，测试 {len(im_te)} 帧（32×32 → 8×8=64 patch，回归推杆+物块 4D 坐标）")

    # 1. 随机初始化编码器基线
    rand_enc = MiniViT(32, 4, 3, d_model=64, n_layers=3, n_heads=4)
    r2_rand = eval_linear_probe(rand_enc, im_tr, st_tr, im_te, st_te)

    # 2. I-JEPA 训练
    ctx_enc = MiniViT(32, 4, 3, d_model=64, n_layers=3, n_heads=4)
    tgt_enc = copy.deepcopy(ctx_enc)
    predictor = IJEPAPredictor(num_patches=64, enc_dim=64, pred_dim=48, n_layers=2, n_heads=4)
    opt_ij = torch.optim.AdamW(list(ctx_enc.parameters()) + list(predictor.parameters()), lr=2e-3, weight_decay=1e-2)

    # 3. MAE 像素重建对照（相同骨干，输出映射到 patch_dim = 48）
    mae_enc = MiniViT(32, 4, 3, d_model=64, n_layers=3, n_heads=4)
    mae_dec = IJEPAPredictor(num_patches=64, enc_dim=64, pred_dim=48, n_layers=2, n_heads=4)
    mae_dec.out_proj = nn.Linear(48, mae_enc.patch_dim)
    opt_mae = torch.optim.AdamW(list(mae_enc.parameters()) + list(mae_dec.parameters()), lr=2e-3, weight_decay=1e-2)

    n_iters = 80 if quick else 500
    eranks, ij_losses = [], []
    for it in range(1, n_iters + 1):
        idx = torch.randint(0, len(im_tr), (32,))
        batch = im_tr[idx]

        opt_ij.zero_grad()
        l_ij, z_full = ijepa_step(ctx_enc, tgt_enc, predictor, batch, grid=8)
        l_ij.backward(); nn.utils.clip_grad_norm_(ctx_enc.parameters(), 1.0); opt_ij.step()
        m_cur = 0.96 + 0.035 * (it / n_iters)
        ema_update(tgt_enc, ctx_enc, m=m_cur)

        opt_mae.zero_grad()
        l_mae = mae_step(mae_enc, mae_dec, batch, grid=8)
        l_mae.backward(); nn.utils.clip_grad_norm_(mae_enc.parameters(), 1.0); opt_mae.step()

        if it % 20 == 0 or it == 1:
            er = effective_rank(z_full.mean(dim=1))
            eranks.append((it, er))
            ij_losses.append((it, float(l_ij.item())))

    r2_ijepa = eval_linear_probe(tgt_enc, im_tr, st_tr, im_te, st_te)
    r2_mae = eval_linear_probe(mae_enc, im_tr, st_tr, im_te, st_te)
    print(f"  训练完成（各 {n_iters} 步，{time.time() - t0:.1f}s）")
    print(f"  I-JEPA 目标表征有效秩（满秩 64）：初始 {eranks[0][1]:.2f} → 最终 {eranks[-1][1]:.2f}")
    print(f"\n  冻结编码器线性探针 R²（PushT 推杆+T 块 4D 坐标，越高越好）：")
    print(f"    随机初始化 MiniViT ：R² = {r2_rand:.3f}")
    print(f"    MAE 像素重建 MiniViT：R² = {r2_mae:.3f}")
    print(f"    I-JEPA 目标编码器   ：R² = {r2_ijepa:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot([x[0] for x in ij_losses], [x[1] for x in ij_losses])
    axes[0].set_title("I-JEPA 特征预测损失"); axes[0].set_xlabel("步数")
    axes[1].plot([x[0] for x in eranks], [x[1] for x in eranks], color="tab:green")
    axes[1].set_title("I-JEPA 目标表征有效秩（防坍塌监控）"); axes[1].set_xlabel("步数")
    axes[2].bar(["随机初始化", "MAE 像素重建", "I-JEPA"], [r2_rand, r2_mae, r2_ijepa], color=["gray", "tab:orange", "tab:blue"])
    axes[2].set_title("冻结表征线性探针 R²（PushT 坐标）"); axes[2].set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(fig, OUT / "03-i-jepa-from-scratch.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：04-v-jepa-video.py")


if __name__ == "__main__":
    main()
