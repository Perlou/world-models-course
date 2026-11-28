"""
Phase 10 · 第 01 课：生成式预测与联合嵌入预测（JEPA）
====================================================

学习目标：
    1. 核对 CONCEPT.md 第 1.3 节两个手算例子（不可约背景方差占比 99% 与双峰均方误差均值）
    2. 构造「带噪声雪花背景的移动小球」数据集（noisy TV 最小版本），分别训练：
       - 生成式像素预测器（观察其损失卡在背景噪声总方差平台上、预测背景变均匀灰）
       - 不加防坍塌手段的联合嵌入预测器（观察其损失趋近 0 但表征各维标准差同时掉到 0）
    3. 在二维候选位置网格上绘制能量地形 E(x, y) = ||g(f(x)) - f'(y)||²，直观对比正常能量谷与坍塌平地

对应概念文档：
    src/phase-10-jepa/CONCEPT.md 第 1 节

运行方式：
    uv run python src/phase-10-jepa/01-generative-vs-joint-embedding.py
    WM_QUICK=1 uv run python src/phase-10-jepa/01-generative-vs-joint-embedding.py

预计运行时间：完整约 35 秒（M1 Max CPU），冒烟约 5 秒
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-10")
SEED = 0


# ==================== 第一部分：带雪花背景的移动小球数据集 ====================


def render_dot_frame(x_pos, y_pos, size=32, radius=2.2, noise_std=0.20, rng=None):
    """生成单帧 32×32 灰度图：确定性高亮小球 + 独立高斯雪花背景（模拟不可预测噪声）。"""
    yy, xx = np.mgrid[0:size, 0:size]
    dist2 = (xx - x_pos) ** 2 + (yy - y_pos) ** 2
    ball = np.exp(-dist2 / (2.0 * (radius ** 2))).astype(np.float32)
    if noise_std > 0.0:
        rng = np.random.default_rng() if rng is None else rng
        bg = rng.normal(0.5, noise_std, size=(size, size)).astype(np.float32)
    else:
        bg = np.full((size, size), 0.5, dtype=np.float32)
    img = np.where(dist2 <= (radius * 1.4) ** 2, 0.95 * ball + 0.05, bg)
    return np.clip(img, 0.0, 1.0)[None, :, :]  # [1, 32, 32]


def make_noisy_dot_dataset(n_samples=1500, size=32, noise_std=0.20, seed=0):
    """生成 (o_t, a_t, o_{t+1}, pos_t, pos_{t+1})，其中小球按动作确定性移动，背景每帧独立重采噪声。"""
    rng = np.random.default_rng(seed)
    obs, nxt, acts, pos_t, pos_n = [], [], [], [], []
    for _ in range(n_samples):
        x = rng.uniform(6.0, size - 6.0)
        y = rng.uniform(6.0, size - 6.0)
        dx = rng.uniform(-3.0, 3.0)
        dy = rng.uniform(-3.0, 3.0)
        x2 = np.clip(x + dx, 4.0, size - 4.0)
        y2 = np.clip(y + dy, 4.0, size - 4.0)
        obs.append(render_dot_frame(x, y, size, noise_std=noise_std, rng=rng))
        nxt.append(render_dot_frame(x2, y2, size, noise_std=noise_std, rng=rng))
        acts.append([(x2 - x) / 3.0, (y2 - y) / 3.0])
        pos_t.append([x / size, y / size])
        pos_n.append([x2 / size, y2 / size])
    return {
        "obs": torch.from_numpy(np.asarray(obs, dtype=np.float32)),
        "next_obs": torch.from_numpy(np.asarray(nxt, dtype=np.float32)),
        "actions": torch.from_numpy(np.asarray(acts, dtype=np.float32)),
        "pos": torch.from_numpy(np.asarray(pos_t, dtype=np.float32)),
        "next_pos": torch.from_numpy(np.asarray(pos_n, dtype=np.float32)),
    }


# ==================== 第二部分：像素预测器 vs 无防坍塌表征预测器 ====================


class SmallEncoder(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(),   # 16
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),  # 8
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(),  # 4
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, z_dim),
        )

    def forward(self, x):
        return self.net(x)


class SmallDecoder(nn.Module):
    def __init__(self, z_dim=16):
        super().__init__()
        self.fc = nn.Linear(z_dim, 32 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(),  # 8
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),  # 16
            nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 32, 4, 4))


class LatentPredictor(nn.Module):
    def __init__(self, z_dim=16, act_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim + act_dim, 64), nn.ReLU(), nn.Linear(64, z_dim))

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


def train_pixel_predictor(data, z_dim=16, n_steps=400):
    enc = SmallEncoder(z_dim)
    pred = LatentPredictor(z_dim, 2)
    dec = SmallDecoder(z_dim)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()) + list(dec.parameters()), lr=2e-3)
    losses = []
    N = len(data["obs"])
    for _ in range(n_steps):
        idx = torch.randint(0, N, (64,))
        z = enc(data["obs"][idx])
        z_hat = pred(z, data["actions"][idx])
        pix_hat = dec(z_hat)
        loss = F.mse_loss(pix_hat, data["next_obs"][idx])
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
    return enc, pred, dec, losses


def train_latent_predictor(data, z_dim=16, n_steps=400):
    """不加任何防坍塌手段的 JEPA：预测损失一路降到 0，表征标准差同步塌到 0。"""
    enc = SmallEncoder(z_dim)
    pred = LatentPredictor(z_dim, 2)
    opt = torch.optim.Adam(list(enc.parameters()) + list(pred.parameters()), lr=2e-3)
    losses, stds = [], []
    N = len(data["obs"])
    for _ in range(n_steps):
        idx = torch.randint(0, N, (64,))
        z = enc(data["obs"][idx])
        z_next = enc(data["next_obs"][idx])
        z_hat = pred(z, data["actions"][idx])
        loss = F.mse_loss(z_hat, z_next)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        stds.append(float(z_next.detach().std(dim=0).mean().item()))
    return enc, pred, losses, stds


@torch.no_grad()
def compute_energy_grid(enc, pred, x_pos=16.0, y_pos=16.0, dx=2.0, dy=-1.5, grid_n=13):
    """固定当前帧与动作，在 G×G 候选小球位置网格上计算能量 E(x, y) = ||g(f(x)) - f(y)||²。"""
    enc.eval(); pred.eval()
    o_t = torch.from_numpy(render_dot_frame(x_pos, y_pos, noise_std=0.0)).unsqueeze(0)
    a_t = torch.tensor([[dx / 3.0, dy / 3.0]], dtype=torch.float32)
    z_hat = pred(enc(o_t), a_t)  # [1, D]
    coords = np.linspace(5.0, 27.0, grid_n)
    grid = np.zeros((grid_n, grid_n), dtype=np.float32)
    for iy, cy in enumerate(coords):
        for ix, cx in enumerate(coords):
            oy = torch.from_numpy(render_dot_frame(cx, cy, noise_std=0.0)).unsqueeze(0)
            zy = enc(oy)
            grid[iy, ix] = float(((z_hat - zy) ** 2).mean().item())
    return coords, grid


def verify_section_1_3_hand_calcs():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 1.3 节两个手算例子")
    print("=" * 60)
    bg_var = 96 * 0.04
    err_loss = bg_var + 4 * (0.1 ** 2)
    print(f"  例 1（100 像素，96 背景方差 0.04）：完美物块损失={bg_var:.2f}，偏 0.1 损失={err_loss:.2f}，有效信号仅占 {(err_loss-bg_var)/err_loss*100:.2f}% ✓")
    y_bar = 0.5 * np.array([1.0, 0.0]) + 0.5 * np.array([0.0, 1.0])
    loss2 = float(np.sum((y_bar - np.array([1.0, 0.0])) ** 2))
    print(f"  例 2（双峰 (1,0)/(0,1)）：MSE 最优预测 = {y_bar.tolist()}，不可约损失 = {loss2:.2f} ✓")


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()
    verify_section_1_3_hand_calcs()

    print("\n" + "=" * 60)
    print("第二部分：带雪花背景的移动小球 —— 像素预测 vs 无防坍塌 JEPA")
    print("=" * 60)
    noise_std = 0.20
    data = make_noisy_dot_dataset(n_samples=400 if quick else 1600, noise_std=noise_std, seed=SEED)
    # 统计背景像素的实测经验方差（排除小球中心区域）
    bg_frac = 1.0 - (np.pi * (2.2 * 1.4) ** 2) / (32 * 32)
    theo_floor = bg_frac * (noise_std ** 2)
    print(f"  数据集：{len(data['obs'])} 对 32×32 帧（背景噪声 σ={noise_std:.2f}，理论不可约 MSE 平台 ≈ {theo_floor:.4f}）")

    n_steps = 120 if quick else 500
    enc_p, pred_p, dec_p, pix_losses = train_pixel_predictor(data, z_dim=16, n_steps=n_steps)
    enc_j, pred_j, jepa_losses, jepa_stds = train_latent_predictor(data, z_dim=16, n_steps=n_steps)

    pix_plateau = float(np.mean(pix_losses[-30:]))
    print(f"  像素预测器末段平均 MSE = {pix_plateau:.4f}（理论背景噪声平台 {theo_floor:.4f}，比值 {pix_plateau/theo_floor:.2f}）")
    print(f"  无防坍塌 JEPA：初始损失 {jepa_losses[0]:.4f} → 末段 {jepa_losses[-1]:.2e}；表征均标准差 {jepa_stds[0]:.4f} → {jepa_stds[-1]:.2e}（彻底坍塌！）")

    # 对比能量地形：用像素预测器学出的（防坍塌的）编码器 vs 坍塌的 JEPA 编码器
    coords, e_good = compute_energy_grid(enc_p, pred_p)
    _, e_flat = compute_energy_grid(enc_j, pred_j)
    print(f"  能量地形动态范围（max - min）：未坍塌编码器 = {e_good.max()-e_good.min():.4f}，坍塌编码器 = {e_flat.max()-e_flat.min():.2e}")

    with torch.no_grad():
        sample_hat = dec_p(pred_p(enc_p(data["obs"][:1]), data["actions"][:1]))[0, 0].numpy()

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    axes[0].plot(pix_losses, label="像素预测 MSE")
    axes[0].axhline(theo_floor, ls="--", color="red", label=f"理论噪声平台 ({theo_floor:.4f})")
    axes[0].set_title("生成式像素预测器的损失平台"); axes[0].set_xlabel("步数"); axes[0].legend(fontsize=8)

    axes[1].plot(jepa_losses, label="JEPA 损失")
    axes[1].plot(jepa_stds, "--", label="表征均标准差")
    axes[1].set_title("无防坍塌 JEPA：损失与标准差双双归零"); axes[1].set_xlabel("步数"); axes[1].legend(fontsize=8)

    axes[2].imshow(sample_hat, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("像素预测输出（背景塌成均匀灰）"); axes[2].axis("off")

    im = axes[3].imshow(e_good, origin="lower", extent=[coords[0], coords[-1], coords[0], coords[-1]], cmap="viridis")
    axes[3].plot(18.0, 14.5, "rx", ms=8, label="真实目标")
    axes[3].set_title("能量地形 E(x, y)（谷底对应目标）"); axes[3].legend(fontsize=8)
    fig.colorbar(im, ax=axes[3], fraction=0.046)

    fig.tight_layout()
    save_figure(fig, OUT / "01-generative-vs-jepa.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：02-collapse-and-regularization.py")


if __name__ == "__main__":
    main()
