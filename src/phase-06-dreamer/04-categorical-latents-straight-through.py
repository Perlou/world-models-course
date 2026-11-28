"""
Dreamer V2：类别隐变量（32×32）与 Straight-Through 梯度
======================================================

学习目标：
    1. 实现 sample_categorical_st(logits)：前向输出货真价实的硬 one-hot，反向借用 softmax 概率的梯度
    2. 严格核对 CONCEPT.md 第 4.3 节的两个手算例子：
       - 下游线性 L(z)=w^T z：straight-through 精确等于真实梯度 (-0.28, -0.20, 0.48)
       - 下游非线性 L(z)=(w^T z)^2：多次采样均值收敛到 (-1.344, -0.96, 2.304)，对比真实梯度 (-1.2, -1.5, 2.7) 看清有偏低方差本质
    3. 复现 CONCEPT.md 第 4.4 节的双峰先验玩具实验：比较高斯先验 vs 类别先验拟合双峰后验收敛后的 KL（~2.997 vs ~0.693 nat）
    4. 实现完整的 CategoricalRSSM（32×32 类别隐变量，展平 1024 维）并校验前向、想象与梯度连通性

核心概念：
    - 类别隐变量（categorical latents）：32 个独立 32 类变量，可表达 2^160 种局面且天然支持多峰先验
    - straight-through 估计器：z = onehot + probs − sg(probs)

前置知识：
    - Phase 3 的高斯 RSSM（04 节）
    - Phase 5 的三种梯度估计器（05 节）

对应概念文档：
    - CONCEPT.md 第 4 节「Dreamer V2：类别隐变量与 straight-through」

预计运行时间：约 15 秒（CPU）
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-06")
SEED = 0


# ==================== 第一部分：straight-through 采样与手算核对 ====================


def sample_categorical_st(logits, unimix=0.0):
    """CONCEPT.md 第 4.7 节：类别隐变量的 straight-through 采样。

    logits: [..., N_z, K] 或 [..., K]。
    返回 (sample_onehot_st, probs)，二者形状与 logits 相同；前向严格为 one-hot，反向梯度流经 probs。
    """
    probs = F.softmax(logits, dim=-1)
    if unimix > 0.0:
        K = logits.shape[-1]
        probs = (1.0 - unimix) * probs + unimix / K
    idx = torch.distributions.Categorical(probs=probs).sample()
    onehot = F.one_hot(idx, num_classes=logits.shape[-1]).to(probs.dtype)
    sample_st = onehot + probs - probs.detach()
    return sample_st, probs


def verify_st_hand_calculations():
    """复现 CONCEPT.md 第 4.3 节的线性与非线性两个手算例子。"""
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 4.3 节的 straight-through 手算例子")
    print("=" * 60)

    p_target = torch.tensor([0.2, 0.5, 0.3])
    base_logits = torch.log(p_target)
    w = torch.tensor([1.0, 2.0, 4.0])

    # 1. 线性下游 L(z) = w^T z：无论采到哪一类，ST 梯度都精确等于 (-0.28, -0.20, 0.48)
    logits = base_logits.clone().requires_grad_(True)
    z_st, _ = sample_categorical_st(logits)
    assert torch.allclose(z_st.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(z_st, (z_st > 0.5).float(), atol=1e-6)
    loss_lin = (w * z_st).sum()
    loss_lin.backward()
    g_lin = logits.grad.detach().numpy()
    print(f"  1. 线性下游 L(z) = w^T z：")
    print(f"     ST 单次采样梯度 = ({g_lin[0]:+.4f}, {g_lin[1]:+.4f}, {g_lin[2]:+.4f})  ｜  解析真实梯度 = (-0.2800, -0.2000, +0.4800)")
    assert np.allclose(g_lin, [-0.28, -0.20, 0.48], atol=1e-5)

    # 2. 非线性下游 L(z) = (w^T z)^2：多次采样平均收敛到 (-1.344, -0.960, +2.304)
    n_mc = 4000 if is_quick() else 30000
    logits_batch = base_logits.unsqueeze(0).expand(n_mc, -1).clone().detach().requires_grad_(True)
    z_batch, _ = sample_categorical_st(logits_batch)
    loss_nl = ((z_batch * w).sum(dim=-1) ** 2).sum()
    loss_nl.backward()
    g_nl_mean = logits_batch.grad.mean(dim=0).numpy()
    print(f"\n  2. 非线性下游 L(z) = (w^T z)^2（{n_mc:,} 次采样均值）：")
    print(f"     ST 梯度均值     = ({g_nl_mean[0]:+.4f}, {g_nl_mean[1]:+.4f}, {g_nl_mean[2]:+.4f})")
    print(f"     手算 ST 期望    = (-1.3440, -0.9600, +2.3040)  ｜  真实无偏梯度 = (-1.2000, -1.5000, +2.7000)")
    assert np.allclose(g_nl_mean, [-1.344, -0.96, 2.304], atol=0.06)
    print("  ✓ 两个手算例子全部验证通过：ST 在非线性下游上有偏，但符号全对且方差极低！")


# ==================== 第二部分：双峰未来玩具实验 ====================


def bimodal_prior_experiment():
    """CONCEPT.md 第 4.4 节：对比高斯先验与类别先验拟合双峰后验时收敛后的 KL。"""
    print("\n" + "=" * 60)
    print("第二部分：双峰未来玩具实验 — 高斯先验 vs 类别先验的 KL 代价")
    print("=" * 60)

    n_steps = 400 if is_quick() else 1200

    # A. 高斯先验 N(μ_p, σ_p²) 拟合 50% N(-2, 0.1²) + 50% N(+2, 0.1²)
    mu_p = nn.Parameter(torch.tensor(0.3))
    log_std_p = nn.Parameter(torch.tensor(0.0))
    opt_g = torch.optim.Adam([mu_p, log_std_p], lr=2e-2)
    q_mus = torch.tensor([-2.0, 2.0])
    q_std = torch.tensor(0.1)
    kl_gauss_hist = []

    for _ in range(n_steps):
        std_p = log_std_p.exp().clamp_min(1e-3)
        kl = (torch.log(std_p / q_std) + (q_std**2 + (q_mus - mu_p) ** 2) / (2 * std_p**2) - 0.5).mean()
        opt_g.zero_grad()
        kl.backward()
        opt_g.step()
        kl_gauss_hist.append(kl.item())

    # B. 类别先验 softmax(ℓ_p) 拟合 50% (1, 0) + 50% (0, 1)
    logits_p = nn.Parameter(torch.tensor([0.8, -0.5]))
    opt_c = torch.optim.Adam([logits_p], lr=2e-2)
    q_cats = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    kl_cat_hist = []

    for _ in range(n_steps):
        log_p = F.log_softmax(logits_p, dim=-1)
        # 0 ln 0 = 0
        kl = -(q_cats * log_p.unsqueeze(0)).sum(dim=-1).mean()
        opt_c.zero_grad()
        kl.backward()
        opt_c.step()
        kl_cat_hist.append(kl.item())

    g_final, c_final = kl_gauss_hist[-1], kl_cat_hist[-1]
    print(f"  高斯先验收敛后：μ_p = {mu_p.item():+.3f}, σ_p = {log_std_p.exp().item():.3f}  →  KL = {g_final:.4f} nat（解析 2.9970）")
    print(f"  类别先验收敛后：p   = {[round(x, 3) for x in F.softmax(logits_p, -1).tolist()]}  →  KL = {c_final:.4f} nat（解析 ln 2 = 0.6931）")
    print(f"  同样是 1 bit（二选一）不确定性，单峰高斯被迫多付 {g_final - c_final:.3f} nat 的表达代价！")
    assert abs(g_final - 2.997) < 0.02 and abs(c_final - math.log(2.0)) < 0.01

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(kl_gauss_hist, linewidth=2.2, label=f"高斯先验（收敛 {g_final:.3f} nat）")
    axes[0].plot(kl_cat_hist, linewidth=2.2, label=f"类别先验（收敛 {c_final:.3f} nat = ln 2）")
    axes[0].axhline(math.log(2.0), color="gray", linestyle=":", label="理论下界 1 bit = 0.693 nat")
    axes[0].set_xlabel("优化步数")
    axes[0].set_ylabel("平均 KL(q ‖ p) (nat)")
    axes[0].set_title("双峰后验下，高斯先验多付出 2.3 nat 无法消除的 KL")
    axes[0].legend(fontsize=8.5)

    xs = np.linspace(-3.5, 3.5, 400)
    mix_pdf = 0.5 * (np.exp(-0.5 * ((xs + 2) / 0.1) ** 2) + np.exp(-0.5 * ((xs - 2) / 0.1) ** 2)) / (0.1 * np.sqrt(2 * np.pi))
    sp = log_std_p.exp().item()
    g_pdf = np.exp(-0.5 * ((xs - mu_p.item()) / sp) ** 2) / (sp * np.sqrt(2 * np.pi))
    axes[1].plot(xs, mix_pdf, color="C0", linewidth=2, label="真实双峰后验混合")
    axes[1].plot(xs, g_pdf, color="C3", linewidth=2.2, linestyle="--", label=f"最优单峰高斯先验 N(0, {sp:.2f}²)")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("概率密度")
    axes[1].set_title("单峰高斯只能拉大方差盖住两峰，把大量质量浪费在中间空白")
    axes[1].legend(fontsize=8.5)
    save_figure(fig, OUT / "04-categorical-vs-gaussian.png")


# ==================== 第三部分：CategoricalRSSM ====================


class CategoricalRSSM(nn.Module):
    """Dreamer V2 / V3 类别隐变量 RSSM（默认 N_z=32 个变量 × K=32 类，展平 1024 维）。"""

    def __init__(self, embed_dim, act_dim, deter_dim=256, n_stoch=32, n_classes=32, hidden=256, unimix=0.01):
        super().__init__()
        self.deter_dim = deter_dim
        self.n_stoch = n_stoch
        self.n_classes = n_classes
        self.stoch_dim = n_stoch * n_classes
        self.feat_dim = deter_dim + self.stoch_dim
        self.unimix = unimix

        self.in_proj = nn.Sequential(nn.Linear(self.stoch_dim + act_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, self.stoch_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, self.stoch_dim),
        )

    def initial_state(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.deter_dim, device=device),
            torch.zeros(batch_size, self.stoch_dim, device=device),
        )

    def _sample(self, logits, sample=True):
        if sample:
            z_oh, _ = sample_categorical_st(logits, unimix=self.unimix)
        else:
            probs = F.softmax(logits, dim=-1)
            idx = probs.argmax(dim=-1)
            z_oh = F.one_hot(idx, num_classes=self.n_classes).to(probs.dtype)
        return z_oh.reshape(*logits.shape[:-2], self.stoch_dim)

    def imagine_step(self, h, z, act, sample=True):
        x = self.in_proj(torch.cat([z, act], dim=-1))
        h = self.gru(x, h)
        prior_logits = self.prior_net(h).reshape(-1, self.n_stoch, self.n_classes)
        z = self._sample(prior_logits, sample=sample)
        return h, z, prior_logits

    def observe_step(self, h, z, act, embed, sample=True):
        x = self.in_proj(torch.cat([z, act], dim=-1))
        h = self.gru(x, h)
        prior_logits = self.prior_net(h).reshape(-1, self.n_stoch, self.n_classes)
        post_logits = self.post_net(torch.cat([h, embed], dim=-1)).reshape(-1, self.n_stoch, self.n_classes)
        z = self._sample(post_logits, sample=sample)
        return h, z, post_logits, prior_logits

    def observe(self, embeds, actions, is_first=None, sample=True):
        B, T, _ = embeds.shape
        h, z = self.initial_state(B, embeds.device)
        hs, zs, posts, priors = [], [], [], []
        for t in range(T):
            if is_first is not None:
                keep = (1.0 - is_first[:, t].float()).unsqueeze(-1)
                h = h * keep
                z = z * keep
                act_t = actions[:, t] * keep
            else:
                act_t = actions[:, t]
            h, z, post_l, prior_l = self.observe_step(h, z, act_t, embeds[:, t], sample=sample)
            hs.append(h)
            zs.append(z)
            posts.append(post_l)
            priors.append(prior_l)
        hs = torch.stack(hs, dim=1)
        zs = torch.stack(zs, dim=1)
        return dict(
            h=hs,
            z=zs,
            features=torch.cat([hs, zs], dim=-1),
            post_logits=torch.stack(posts, dim=1),
            prior_logits=torch.stack(priors, dim=1),
        )


def verify_categorical_rssm():
    print("\n" + "=" * 60)
    print("第三部分：验证 CategoricalRSSM（32×32 类别隐变量）前向与 ST 梯度连通")
    print("=" * 60)
    rssm = CategoricalRSSM(embed_dim=128, act_dim=1, deter_dim=256, n_stoch=32, n_classes=32)
    embeds = torch.randn(4, 10, 128, requires_grad=True)
    acts = torch.randn(4, 10, 1)
    out = rssm.observe(embeds, acts)
    # 校验每个时间步展平的 z_t 恰好有 32 个 1（稀疏度 32 / 1024 = 3.125%）
    ones_per_step = out["z"].detach().sum(dim=-1)
    assert torch.allclose(ones_per_step, torch.full_like(ones_per_step, 32.0))
    # 校验梯度能通过 straight-through 回到编码器输入 embeds
    out["features"].pow(2).mean().backward()
    assert embeds.grad is not None and embeds.grad.norm().item() > 0
    print(f"  features 形状：{list(out['features'].shape)}（h=256 + z=32×32=1024 → 1280）")
    print(f"  post_logits 形状：{list(out['post_logits'].shape)} ｜ 每步 z 中 1 的个数 = {ones_per_step.mean().item():.1f}")
    print(f"  经 straight-through 反传到 embeds 的梯度范数 = {embeds.grad.norm().item():.4f} ✓")


def main():
    seed_everything(SEED)
    verify_st_hand_calculations()
    bimodal_prior_experiment()
    verify_categorical_rssm()
    print("\n" + "=" * 60)
    print("本节完成！下一节：05-kl-balancing-and-free-bits.py（KL balancing 与 free bits）")
    print("=" * 60)


if __name__ == "__main__":
    main()
