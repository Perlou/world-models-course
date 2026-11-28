"""
Phase 9 · 第 01 课：从 DDPM 到 EDM —— σ 参数化、预条件与少步采样器
==================================================================

学习目标：
    1. 实现 DDPM 离散日程到连续噪声标准差 σ 的换算，理解 σ 的物理含义
    2. 实现 EDM 的四个预条件系数 (c_in, c_skip, c_out, c_noise) 与 F 空间训练损失，
       逐项核对 CONCEPT.md 第 1.6、1.8、1.9 节的手算数字
    3. 实现 Karras 噪声日程与 Euler / Heun ODE 采样器
    4. 在二维多模态分布（8-高斯环）与 32×32 PushT 真实帧上对比 EDM 与朴素噪声预测
       在 N ∈ {1, 2, 3, 5, 10, 50} 步下的样本质量

对应概念文档：
    src/phase-09-diffusion-video-world-models/CONCEPT.md 第 1 节

运行方式：
    uv run python src/phase-09-diffusion-video-world-models/01-ddpm-to-edm.py
    WM_QUICK=1 uv run python src/phase-09-diffusion-video-world-models/01-ddpm-to-edm.py

预计运行时间：完整约 45 秒（M1 Max CPU），冒烟约 6 秒
"""

import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.pusht import load_pusht_demos

plt = setup_matplotlib()
OUT = output_dir("phase-09")
SEED = 0


# ==================== 第一部分：DDPM→σ 换算与 EDM 核心组件 ====================


def ddpm_alpha_bar_to_sigma(n_steps=1000, beta_start=1e-4, beta_end=0.02):
    """CONCEPT.md 第 1.2 节：DDPM 线性 β 日程换算为等价连续噪声标准差 σ_k。"""
    betas = np.linspace(beta_start, beta_end, n_steps, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    sigmas = np.sqrt((1.0 - alpha_bars) / alpha_bars)
    return alpha_bars, sigmas


def edm_precond_coeffs(sigma, sigma_data=0.5):
    """CONCEPT.md 第 1.7 节：EDM 四个预条件系数与等价损失权重 λ(σ)。"""
    sd2 = sigma_data ** 2
    s2 = sigma ** 2
    denom = torch.sqrt(s2 + sd2)
    c_in = 1.0 / denom
    c_skip = sd2 / (s2 + sd2)
    c_out = (sigma * sigma_data) / denom
    c_noise = 0.25 * torch.log(sigma.clamp_min(1e-8))
    lam = (s2 + sd2) / ((sigma * sigma_data).clamp_min(1e-8) ** 2)
    return c_in, c_skip, c_out, c_noise, lam


def karras_sigmas(n_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0, device="cpu"):
    """CONCEPT.md 第 1.9 节：Karras 噪声日程 σ_0 > σ_1 > ... > σ_N = 0。"""
    if n_steps == 1:
        return torch.tensor([sigma_max, 0.0], dtype=torch.float32, device=device)
    ramp = torch.linspace(0.0, 1.0, n_steps, device=device)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)])


class SinusoidalEmb(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0.0, -math.log(1000.0), half, device=x.device))
        ang = x.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class ToyMLP(nn.Module):
    """用于 2D 玩具分布的轻量条件 MLP F_θ(x_in, c_noise)。"""

    def __init__(self, data_dim=2, hidden=128):
        super().__init__()
        self.t_emb = SinusoidalEmb(64)
        self.net = nn.Sequential(
            nn.Linear(data_dim + 64, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x, c_noise):
        return self.net(torch.cat([x, self.t_emb(c_noise)], dim=-1))


class EDMDenoiser(nn.Module):
    """用 EDM 四个预条件系数包装底层网络 F_θ 得到去噪器 D_θ(x_σ; σ)。"""

    def __init__(self, net, sigma_data=0.5):
        super().__init__()
        self.net = net
        self.sigma_data = float(sigma_data)

    def forward(self, x_noisy, sigma):
        sigma = sigma.reshape(-1)
        c_in, c_skip, c_out, c_noise, _ = edm_precond_coeffs(sigma, self.sigma_data)
        view = [-1] + [1] * (x_noisy.ndim - 1)
        f_out = self.net(c_in.view(*view) * x_noisy, c_noise)
        return c_skip.view(*view) * x_noisy + c_out.view(*view) * f_out

    def f_loss(self, x, sigma):
        """CONCEPT.md 第 1.10 节：直接在单位方差的 F 空间算损失，数值最稳。"""
        eps = torch.randn_like(x)
        view = [-1] + [1] * (x.ndim - 1)
        x_noisy = x + sigma.view(*view) * eps
        c_in, c_skip, c_out, c_noise, _ = edm_precond_coeffs(sigma, self.sigma_data)
        f_pred = self.net(c_in.view(*view) * x_noisy, c_noise)
        f_target = (x - c_skip.view(*view) * x_noisy) / c_out.view(*view).clamp_min(1e-6)
        return F.mse_loss(f_pred, f_target)


class EpsDenoiser(nn.Module):
    """对照基线：DDPM 式朴素噪声预测 D_θ(x_σ; σ) = x_σ - σ · ε_θ(x_σ / √(σ²+σ_d²), c_noise)。"""

    def __init__(self, net, sigma_data=0.5):
        super().__init__()
        self.net = net
        self.sigma_data = float(sigma_data)

    def forward(self, x_noisy, sigma):
        sigma = sigma.reshape(-1)
        c_in, _, _, c_noise, _ = edm_precond_coeffs(sigma, self.sigma_data)
        view = [-1] + [1] * (x_noisy.ndim - 1)
        eps_pred = self.net(c_in.view(*view) * x_noisy, c_noise)
        return x_noisy - sigma.view(*view) * eps_pred

    def eps_loss(self, x, sigma):
        eps = torch.randn_like(x)
        view = [-1] + [1] * (x.ndim - 1)
        x_noisy = x + sigma.view(*view) * eps
        c_in, _, _, c_noise, _ = edm_precond_coeffs(sigma, self.sigma_data)
        eps_pred = self.net(c_in.view(*view) * x_noisy, c_noise)
        return F.mse_loss(eps_pred, eps)


def sample_train_sigmas(batch_size, p_mean=-1.2, p_std=1.2, device="cpu"):
    return torch.exp(torch.randn(batch_size, device=device) * p_std + p_mean)


@torch.no_grad()
def sample_euler(denoiser, shape, n_steps, sigma_min=0.002, sigma_max=10.0, rho=7.0, device="cpu", init_eps=None):
    """CONCEPT.md 第 1.9–1.10 节：一阶 Euler 概率流 ODE 采样器（NFE = N）。"""
    denoiser.eval()
    sigmas = karras_sigmas(n_steps, sigma_min, sigma_max, rho, device=device)
    eps = torch.randn(shape, device=device) if init_eps is None else init_eps.to(device)
    x = eps * sigmas[0]
    B = shape[0]
    for i in range(n_steps):
        s_cur, s_next = sigmas[i], sigmas[i + 1]
        x_hat = denoiser(x, torch.full((B,), s_cur.item(), device=device))
        x = x_hat + (s_next / s_cur) * (x - x_hat)
    return x


@torch.no_grad()
def sample_heun(denoiser, shape, n_steps, sigma_min=0.002, sigma_max=10.0, rho=7.0, device="cpu", init_eps=None):
    """CONCEPT.md 第 1.9 节：EDM 二阶 Heun 采样器（NFE = 2N - 1）。"""
    denoiser.eval()
    sigmas = karras_sigmas(n_steps, sigma_min, sigma_max, rho, device=device)
    eps = torch.randn(shape, device=device) if init_eps is None else init_eps.to(device)
    x = eps * sigmas[0]
    B = shape[0]
    for i in range(n_steps):
        s_cur, s_next = sigmas[i], sigmas[i + 1]
        x_hat = denoiser(x, torch.full((B,), s_cur.item(), device=device))
        d_cur = (x - x_hat) / s_cur
        x_euler = x + (s_next - s_cur) * d_cur
        if s_next.item() > 0.0:
            x_hat_next = denoiser(x_euler, torch.full((B,), s_next.item(), device=device))
            d_next = (x_euler - x_hat_next) / s_next
            x = x + 0.5 * (s_next - s_cur) * (d_cur + d_next)
        else:
            x = x_euler
    return x


# ==================== 第二部分：手算核对 ====================


def verify_section_1_hand_calcs():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 1.2 / 1.6 / 1.8 / 1.9 节手算")
    print("=" * 60)
    alpha_bars, sigmas = ddpm_alpha_bar_to_sigma()
    print(f"  DDPM (K=1000, β: 1e-4→0.02)：ᾱ_K = {alpha_bars[-1]:.2e}，σ_K = {sigmas[-1]:.1f}（手算约 157）")
    assert 150.0 < sigmas[-1] < 165.0

    for s_val in [0.5, 0.02, 20.0]:
        ci, cs, co, cn, lam = edm_precond_coeffs(torch.tensor([s_val]), sigma_data=0.5)
        print(f"  σ={s_val:5.2f} | c_in={ci.item():.4f}  c_skip={cs.item():.6f}  c_out={co.item():.4f}  λ={lam.item():.2f}")

    sched3 = karras_sigmas(3, sigma_min=0.002, sigma_max=5.0, rho=7.0).tolist()
    print(f"  Karras 日程 (N=3, σ_max=5, ρ=7)：{[round(v, 3) for v in sched3]}（手算 [5.0, 0.283, 0.002, 0.0]）")
    assert abs(sched3[1] - 0.283) < 0.01

    # 第 1.9 节 1D 高斯 N(2, 1) 最优去噪器 Euler 验证
    d_star = lambda x, s: 2.0 + (x - 2.0) / (1.0 + s * s)
    x1_1step = d_star(4.5, 2.0)
    x_mid = d_star(4.5, 2.0) + 0.5 * (4.5 - d_star(4.5, 2.0))
    x1_2step = d_star(x_mid, 1.0)
    x_exact = 2.0 + (4.5 - 2.0) * math.sqrt(1.0 / 5.0)
    print(f"  1D 高斯 (x0=4.5, σ=2→0)：1 步={x1_1step:.2f}，2 步={x1_2step:.2f}，精确解={x_exact:.2f}（手算 2.50 / 2.75 / 3.12 ✓）")
    return alpha_bars, sigmas


# ==================== 第三部分：2D 八峰分布对比（EDM vs 朴素 ε 预测） ====================


def sample_8gaussians(n, radius=1.0, std=0.06, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * radius
    idx = rng.integers(0, 8, size=n)
    pts = centers[idx] + rng.normal(0.0, std, size=(n, 2))
    return pts.astype(np.float32), centers.astype(np.float32)


def eval_2d_samples(samples_np, centers, radius=1.0):
    """评估生成点到最近模态中心的平均距离、高置信命中率与模态覆盖熵。"""
    dists = np.linalg.norm(samples_np[:, None, :] - centers[None, :, :], axis=-1)
    min_dist = dists.min(axis=-1)
    nearest = dists.argmin(axis=-1)
    counts = np.bincount(nearest, minlength=len(centers)).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(centers)))
    hit_rate = float((min_dist < 0.25).mean())
    return float(min_dist.mean()), hit_rate, entropy


def run_2d_experiment(quick=False):
    print("\n" + "=" * 60)
    print("第二部分：二维 8-高斯环上对比 EDM 与朴素噪声预测（不同步数）")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    data_np, centers = sample_8gaussians(4000, rng=rng)
    sigma_data = float(data_np.std())
    data_t = torch.from_numpy(data_np)
    print(f"  数据集：8-高斯环（4000 点，σ_data = {sigma_data:.3f}）")

    edm = EDMDenoiser(ToyMLP(2, 128), sigma_data=sigma_data)
    eps_m = EpsDenoiser(ToyMLP(2, 128), sigma_data=sigma_data)
    opt_edm = torch.optim.Adam(edm.parameters(), lr=2e-3)
    opt_eps = torch.optim.Adam(eps_m.parameters(), lr=2e-3)

    steps = 400 if quick else 2500
    t0 = time.time()
    for _ in range(steps):
        idx = torch.randint(0, len(data_t), (256,))
        x = data_t[idx]
        sig = sample_train_sigmas(256, p_mean=-1.0, p_std=1.2)

        opt_edm.zero_grad()
        edm.f_loss(x, sig).backward()
        opt_edm.step()

        opt_eps.zero_grad()
        eps_m.eps_loss(x, sig).backward()
        opt_eps.step()
    print(f"  训练完成（各 {steps} 步，共 {time.time() - t0:.1f}s）")

    step_list = [1, 2, 3, 5, 10, 50]
    init_eps = torch.randn(1000, 2)
    results = {"edm_euler": {}, "edm_heun": {}, "eps_euler": {}}
    print(f"\n  {'步数 N':>6s} | {'EDM-Euler 距中心':>16s} {'命中率':>7s} | {'EDM-Heun 距中心':>15s} {'命中率':>7s} | {'ε-预测 Euler 距中心':>19s} {'命中率':>7s}")
    for n in step_list:
        s_ee = sample_euler(edm, (1000, 2), n, sigma_max=8.0, init_eps=init_eps).numpy()
        s_eh = sample_heun(edm, (1000, 2), n, sigma_max=8.0, init_eps=init_eps).numpy()
        s_pe = sample_euler(eps_m, (1000, 2), n, sigma_max=8.0, init_eps=init_eps).numpy()
        m_ee = eval_2d_samples(s_ee, centers)
        m_eh = eval_2d_samples(s_eh, centers)
        m_pe = eval_2d_samples(s_pe, centers)
        results["edm_euler"][n] = (m_ee, s_ee)
        results["edm_heun"][n] = (m_eh, s_eh)
        results["eps_euler"][n] = (m_pe, s_pe)
        print(f"  {n:>6d} | {m_ee[0]:16.4f} {m_ee[1]*100:6.1f}% | {m_eh[0]:15.4f} {m_eh[1]*100:6.1f}% | {m_pe[0]:19.4f} {m_pe[1]*100:6.1f}%")
    return data_np, centers, results


# ==================== 第四部分：32×32 PushT 图像上的小规模 EDM 实验 ====================


class TinyImageUNet(nn.Module):
    """极轻量 32×32 U-Net，供单帧图像 EDM 去噪演示。"""

    def __init__(self, in_ch=3, out_ch=3, base=32, cond_dim=64):
        super().__init__()
        self.t_emb = nn.Sequential(SinusoidalEmb(cond_dim), nn.Linear(cond_dim, cond_dim), nn.SiLU())
        self.enc1 = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.enc2 = nn.Conv2d(base, base * 2, 4, 2, 1)      # 16×16
        self.mid = nn.Conv2d(base * 2, base * 2, 3, 1, 1)
        self.film_mid = nn.Linear(cond_dim, base * 4)
        self.dec2 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.film_dec = nn.Linear(cond_dim, base * 2)
        self.out = nn.Conv2d(base * 2, out_ch, 3, 1, 1)

    def forward(self, x, c_noise):
        c = self.t_emb(c_noise)
        h1 = F.silu(self.enc1(x))
        h2 = F.silu(self.enc2(h1))
        g1, b1 = self.film_mid(c).chunk(2, dim=-1)
        hm = F.silu(self.mid(h2) * (1.0 + g1[:, :, None, None]) + b1[:, :, None, None])
        u = self.dec2(hm)
        g2, b2 = self.film_dec(c).chunk(2, dim=-1)
        u = F.silu(u * (1.0 + g2[:, :, None, None]) + b2[:, :, None, None])
        return self.out(torch.cat([u, h1], dim=1))


def run_image_experiment(quick=False):
    print("\n" + "=" * 60)
    print("第三部分：32×32 PushT 真实帧上的 EDM 无条件生成与多步对比")
    print("=" * 60)
    demos = load_pusht_demos()
    n_imgs = 128 if quick else 1024
    stride = max(1, len(demos["images"]) // n_imgs)
    raw = demos["images"][::stride][:n_imgs]
    imgs = np.stack([cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA) for im in raw], axis=0)
    imgs = (imgs.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2)  # [N, 3, 32, 32]
    sigma_data = float(imgs_t.std().item())

    edm_img = EDMDenoiser(TinyImageUNet(3, 3, base=32), sigma_data=sigma_data)
    opt = torch.optim.Adam(edm_img.parameters(), lr=1e-3)
    n_iters = 60 if quick else 450
    t0 = time.time()
    for _ in range(n_iters):
        idx = torch.randint(0, len(imgs_t), (32,))
        x = imgs_t[idx]
        sig = sample_train_sigmas(32, p_mean=-1.2, p_std=1.2)
        opt.zero_grad()
        edm_img.f_loss(x, sig).backward()
        opt.step()
    print(f"  PushT 32×32 EDM 训练 {n_iters} 步完成（{time.time() - t0:.1f}s，σ_data = {sigma_data:.3f}）")

    init_eps = torch.randn(8, 3, 32, 32)
    img_samples = {}
    print(f"  {'步数 N':>6s} | {'采样耗时(ms/帧)':>14s} | {'像素标准差(真实≈%.3f)' % sigma_data:>22s} | {'最近邻 L1 误差':>14s}")
    for n in [1, 2, 3, 5, 10, 50]:
        ts = time.time()
        gen = sample_euler(edm_img, (8, 3, 32, 32), n, sigma_max=20.0, init_eps=init_eps).clamp(-1.0, 1.0)
        ms = (time.time() - ts) * 1000.0 / 8.0
        # 衡量生成样本到真实数据集最近邻的 L1 距离（越小说明越像真实 PushT 画面）
        nn_l1 = (gen[:, None] - imgs_t[:128][None]).abs().mean(dim=(2, 3, 4)).min(dim=1).values.mean().item()
        std_gen = float(gen.std().item())
        img_samples[n] = (gen[0].permute(1, 2, 0).numpy() + 1.0) * 0.5
        print(f"  {n:>6d} | {ms:14.2f} | {std_gen:22.3f} | {nn_l1:14.4f}")
    return img_samples


def main():
    t_start = time.time()
    seed_everything(SEED)
    quick = is_quick()

    alpha_bars, sigmas = verify_section_1_hand_calcs()
    data_np, centers, res_2d = run_2d_experiment(quick=quick)
    img_samples = run_image_experiment(quick=quick)

    # 绘图 1：DDPM σ 曲线 + EDM 预条件系数 + 2D 采样对比
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes[0, 0].semilogy(sigmas)
    axes[0, 0].set_title("DDPM 日程换算为 σ_k")
    axes[0, 0].set_xlabel("步数 k"); axes[0, 0].set_ylabel("σ (对数)")

    s_grid = torch.logspace(-2, 1.5, 200)
    ci, cs, co, _, _ = edm_precond_coeffs(s_grid, 0.5)
    axes[0, 1].semilogx(s_grid.numpy(), cs.numpy(), label="c_skip")
    axes[0, 1].semilogx(s_grid.numpy(), co.numpy(), label="c_out")
    axes[0, 1].semilogx(s_grid.numpy(), ci.numpy(), label="c_in")
    axes[0, 1].axvline(0.5, ls="--", color="gray", alpha=0.6, label="σ_data=0.5")
    axes[0, 1].set_title("EDM 预条件系数"); axes[0, 1].legend(fontsize=8)

    ns = [1, 2, 3, 5, 10, 50]
    axes[0, 2].plot(ns, [res_2d["edm_euler"][n][0][0] for n in ns], "o-", label="EDM Euler")
    axes[0, 2].plot(ns, [res_2d["edm_heun"][n][0][0] for n in ns], "s-", label="EDM Heun")
    axes[0, 2].plot(ns, [res_2d["eps_euler"][n][0][0] for n in ns], "^--", label="ε-预测 Euler")
    axes[0, 2].set_xscale("log"); axes[0, 2].set_title("2D 距最近中心误差 (越低越好)"); axes[0, 2].legend(fontsize=8)

    axes[0, 3].scatter(data_np[:500, 0], data_np[:500, 1], s=4, alpha=0.5, label="真实")
    axes[0, 3].set_title("真实 8-高斯环"); axes[0, 3].set_xlim(-1.6, 1.6); axes[0, 3].set_ylim(-1.6, 1.6)

    for idx, (title, pts) in enumerate([
        ("EDM Euler 1 步", res_2d["edm_euler"][1][1]),
        ("EDM Euler 5 步", res_2d["edm_euler"][5][1]),
        ("EDM Heun 5 步", res_2d["edm_heun"][5][1]),
        ("ε-预测 Euler 5 步", res_2d["eps_euler"][5][1]),
    ]):
        ax = axes[1, idx]
        ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=40, zorder=5)
        ax.scatter(pts[:500, 0], pts[:500, 1], s=5, alpha=0.5)
        ax.set_title(title); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)

    fig.tight_layout()
    save_figure(fig, OUT / "01-ddpm-to-edm.png")

    # 绘图 2：PushT 32×32 不同采样步数的画面
    fig2, axes2 = plt.subplots(1, 6, figsize=(13, 2.6))
    for ax, n in zip(axes2, [1, 2, 3, 5, 10, 50]):
        ax.imshow(np.clip(img_samples[n], 0.0, 1.0))
        ax.set_title(f"N = {n} 步")
        ax.axis("off")
    fig2.suptitle("PushT 32×32 EDM 采样随步数 N 的变化（固定初始噪声）")
    fig2.tight_layout()
    save_figure(fig2, OUT / "01-pusht-edm-steps.png")

    print(f"\n本节完成（总用时 {time.time() - t_start:.1f}s）。下一节：02-flow-matching.py")


if __name__ == "__main__":
    main()
