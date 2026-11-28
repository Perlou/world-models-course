"""
Phase 10 · 第 05 课：探针 —— 表征里到底有没有位置和速度
=======================================================

学习目标：
    1. 核对 CONCEPT.md 第 5.3 节岭回归线性探针闭式解手算（w=1.993, R²=0.997）
    2. 实现闭式岭回归线性探针 fit_linear_probe()（验证集选 λ）、r2_score() 与可视化解码器（带 sg）
    3. 在合成移动几何体视频上，对 6 组表征做位置与速度的线性探针：
       I-JEPA（单帧 / 两帧拼接）、V-JEPA（管状掩码 / 逐帧掩码）、随机初始化、原始像素 + 打乱标签对照
    4. 验证单帧图像表征为何读不出速度、两帧拼接与视频 tubelet 为何能读出速度

对应概念文档：
    src/phase-10-jepa/CONCEPT.md 第 5 节

运行方式：
    uv run python src/phase-10-jepa/05-probing-latent-representations.py
    WM_QUICK=1 uv run python src/phase-10-jepa/05-probing-latent-representations.py

预计运行时间：完整约 80 秒（M1 Max CPU），冒烟约 10 秒
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
_spec02 = importlib.util.spec_from_file_location("p10_02_pr", os.path.join(_DIR, "02-collapse-and-regularization.py"))
_mod02 = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(_mod02)

_spec03 = importlib.util.spec_from_file_location("p10_03_pr", os.path.join(_DIR, "03-i-jepa-from-scratch.py"))
_mod03 = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(_mod03)

_spec04 = importlib.util.spec_from_file_location("p10_04_pr", os.path.join(_DIR, "04-v-jepa-video.py"))
_mod04 = importlib.util.module_from_spec(_spec04)
_spec04.loader.exec_module(_mod04)

ema_update = _mod02.ema_update
MiniViT = _mod03.MiniViT
IJEPAPredictor = _mod03.IJEPAPredictor
ijepa_step = _mod03.ijepa_step
make_moving_shapes_videos = _mod04.make_moving_shapes_videos
MiniVideoViT = _mod04.MiniVideoViT
train_vjepa_variant = _mod04.train_vjepa_variant

plt = setup_matplotlib()
OUT = output_dir("phase-10")
SEED = 0


# ==================== 第一部分：闭式线性探针与可视化解码器 ====================


def r2_score(y_true, y_pred):
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-8)


def fit_linear_probe(Z_tr, Y_tr, Z_val, Y_val, Z_te, Y_te, lambdas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """CONCEPT.md 第 5.3/5.6 节：按训练集标准化，验证集选最优正则 λ，闭式岭回归求 W 并在测试集评估 R²。"""
    mu_z, sd_z = Z_tr.mean(0), Z_tr.std(0) + 1e-6
    mu_y, sd_y = Y_tr.mean(0), Y_tr.std(0) + 1e-6
    def _prep(Z):
        return np.hstack([(Z - mu_z) / sd_z, np.ones((len(Z), 1), dtype=np.float64)])
    X_tr, X_val, X_te = _prep(Z_tr.astype(np.float64)), _prep(Z_val.astype(np.float64)), _prep(Z_te.astype(np.float64))
    y_tr_n = (Y_tr - mu_y) / sd_y
    y_val_n = (Y_val - mu_y) / sd_y

    best_lam, best_val, best_W = lambdas[0], -1e9, None
    D1 = X_tr.shape[1]
    eye = np.eye(D1, dtype=np.float64); eye[-1, -1] = 0.0
    XtX = X_tr.T @ X_tr
    XtY = X_tr.T @ y_tr_n
    for lam in lambdas:
        W = np.linalg.solve(XtX + lam * eye, XtY)
        val_r2 = r2_score(y_val_n, X_val @ W)
        if val_r2 > best_val:
            best_val, best_lam, best_W = val_r2, lam, W
    pred_te = (X_te @ best_W) * sd_y + mu_y
    return r2_score(Y_te, pred_te), best_lam, best_W


class VizDecoder(nn.Module):
    """CONCEPT.md 第 5.4 节：只用于可视化的解码器，输入带 stop-gradient，绝不影响编码器。"""

    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.SiLU(),
            nn.Linear(256, 32 * 4 * 4), nn.SiLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(self.net(z.detach()).view(-1, 32, 4, 4))


def verify_section_5_3_hand_calc():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 5.3 节线性探针闭式解手算")
    print("=" * 60)
    z = np.array([1.0, 2.0, 3.0])
    y = np.array([2.1, 3.9, 6.0])
    w = float((z @ y) / (z @ z))
    r2 = r2_score(y[:, None], (w * z)[:, None])
    print(f"  无偏置闭式解：w = {w:.3f}（手算 1.993），R² = {r2:.3f}（手算 0.997 ✓）")
    assert abs(w - 1.993) < 0.002 and abs(r2 - 0.997) < 0.002


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()
    verify_section_5_3_hand_calc()

    print("\n" + "=" * 60)
    print("第二部分：训练小 I-JEPA 与 V-JEPA，对 6 组冻结表征做位置与速度探针")
    print("=" * 60)
    n_tot = 280 if quick else 900
    videos, pos, vel = make_moving_shapes_videos(n_videos=n_tot, n_frames=8, size=32, seed=SEED)
    n1, n2 = int(n_tot * 0.6), int(n_tot * 0.8)
    v_tr, v_val, v_te = videos[:n1], videos[n1:n2], videos[n2:]
    p_tr, p_val, p_te = pos[:n1], pos[n1:n2], pos[n2:]
    vel_tr, vel_val, vel_te = vel[:n1], vel[n1:n2], vel[n2:]

    # 训练图像 I-JEPA（在视频单帧上）
    ij_ctx = MiniViT(32, 4, 3, d_model=64, n_layers=2, n_heads=4)
    ij_tgt = copy.deepcopy(ij_ctx)
    ij_pred = IJEPAPredictor(64, 64, 48, n_layers=2, n_heads=4)
    opt_ij = torch.optim.AdamW(list(ij_ctx.parameters()) + list(ij_pred.parameters()), lr=2e-3)
    n_iters = 60 if quick else 300
    frames_flat = v_tr.reshape(-1, 3, 32, 32)
    for it in range(1, n_iters + 1):
        idx = torch.randint(0, len(frames_flat), (32,))
        opt_ij.zero_grad()
        loss, _ = ijepa_step(ij_ctx, ij_tgt, ij_pred, frames_flat[idx], grid=8)
        loss.backward(); opt_ij.step()
        ema_update(ij_tgt, ij_ctx, m=0.97)

    # 训练视频 V-JEPA（管状掩码 & 逐帧独立掩码）
    vj_tube, _ = train_vjepa_variant("tube", v_tr, n_iters=n_iters)
    vj_indep, _ = train_vjepa_variant("indep", v_tr, n_iters=n_iters)
    rand_img_vit = MiniViT(32, 4, 3, d_model=64, n_layers=2, n_heads=4)

    @torch.no_grad()
    def extract_img_2x2(enc, ims):
        enc.eval()
        z = enc(ims)  # [B, 64, 64]
        z2d = z.reshape(len(ims), 8, 8, 64).permute(0, 3, 1, 2)
        return F.adaptive_avg_pool2d(z2d, (2, 2)).flatten(1).numpy()

    @torch.no_grad()
    def extract_vid_feat(enc, vids):
        enc.eval()
        z = enc(vids)  # [B, 64, 64] = [B, 4*16, 64]
        B = len(vids)
        return z.reshape(B, enc.gt, enc.gh * enc.gw, enc.d_model).mean(dim=2).reshape(B, -1).numpy()

    mid = 4  # 对应 pos/vel 标注所在帧
    extractors = {
        "I-JEPA（单帧）": lambda v: extract_img_2x2(ij_tgt, v[:, mid]),
        "I-JEPA（两帧拼接）": lambda v: np.hstack([extract_img_2x2(ij_tgt, v[:, mid - 1]), extract_img_2x2(ij_tgt, v[:, mid])]),
        "V-JEPA（管状掩码）": lambda v: extract_vid_feat(vj_tube, v),
        "V-JEPA（逐帧掩码）": lambda v: extract_vid_feat(vj_indep, v),
        "随机初始化 ViT（两帧）": lambda v: np.hstack([extract_img_2x2(rand_img_vit, v[:, mid - 1]), extract_img_2x2(rand_img_vit, v[:, mid])]),
        "原始像素（单帧降采样）": lambda v: F.avg_pool2d(v[:, mid], 2).flatten(1).numpy(),
    }

    print(f"\n  {'表征来源':22s} | {'位置探针 R²':>10s} | {'速度探针 R²':>10s} | {'打乱标签对照 R²':>14s}")
    rng = np.random.default_rng(123)
    shuf_perm = rng.permutation(len(p_tr))
    table_res = {}
    for name, fn in extractors.items():
        Ztr, Zval, Zte = fn(v_tr), fn(v_val), fn(v_te)
        r2_p, _, _ = fit_linear_probe(Ztr, p_tr, Zval, p_val, Zte, p_te)
        r2_v, _, _ = fit_linear_probe(Ztr, vel_tr, Zval, vel_val, Zte, vel_te)
        r2_shuf, _, _ = fit_linear_probe(Ztr, p_tr[shuf_perm], Zval, p_val, Zte, p_te)
        table_res[name] = (r2_p, r2_v, r2_shuf)
        print(f"  {name:22s} | {r2_p:10.3f} | {r2_v:10.3f} | {r2_shuf:14.3f}")

    # 额外演示第 5.4 节：只用于可视化的解码器（不回传梯度到 I-JEPA）
    z_tr_viz = torch.from_numpy(extract_img_2x2(ij_tgt, v_tr[:, mid]))
    z_te_viz = torch.from_numpy(extract_img_2x2(ij_tgt, v_te[:4, mid]))
    viz_dec = VizDecoder(z_tr_viz.shape[1])
    opt_dec = torch.optim.Adam(viz_dec.parameters(), lr=3e-3)
    for _ in range(60 if quick else 250):
        idx = torch.randint(0, len(z_tr_viz), (32,))
        opt_dec.zero_grad()
        F.mse_loss(viz_dec(z_tr_viz[idx]), v_tr[idx, mid]).backward()
        opt_dec.step()
    with torch.no_grad():
        viz_out = viz_dec(z_te_viz)
        viz_psnr = float(-10.0 * math.log10(max(F.mse_loss(viz_out, v_te[:4, mid]).item(), 1e-8)))
    print(f"\n  只用于可视化的冻结解码器（sg 阻断梯度）：测试集重建 PSNR = {viz_psnr:.2f} dB")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    names = list(table_res.keys())
    x = np.arange(len(names))
    axes[0].bar(x - 0.18, [table_res[n][0] for n in names], 0.35, label="位置 R²")
    axes[0].bar(x + 0.18, [table_res[n][1] for n in names], 0.35, label="速度 R²")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=22, ha="right", fontsize=8)
    axes[0].set_title("各冻结表征的物理量线性可读性对比"); axes[0].set_ylim(-0.15, 1.05); axes[0].legend()

    # 右图拼接真实帧与可视化解码器重建帧
    grid_im = np.concatenate([
        np.concatenate([v_te[i, mid].permute(1, 2, 0).numpy() for i in range(4)], axis=1),
        np.concatenate([viz_out[i].permute(1, 2, 0).numpy() for i in range(4)], axis=1),
    ], axis=0)
    axes[1].imshow(grid_im.clip(0, 1))
    axes[1].set_title(f"可视化解码器对照（上：原图，下：从冻结 I-JEPA 解码，{viz_psnr:.1f}dB）")
    axes[1].axis("off")
    fig.tight_layout()
    save_figure(fig, OUT / "05-probing-representations.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：06-action-conditioned-latent-dynamics.py")


if __name__ == "__main__":
    main()
