"""
从零实现 RSSM
=============

学习目标：
    1. 手写两个对角高斯之间 KL 散度的闭式解 gaussian_kl()，与 PyTorch 官方实现核对，并复现四个手算例子
    2. 看清 KL 的梯度往哪推：先验均值追后验均值，先验方差自动学出"均值预测误差的平方 + 后验方差"
    3. 从零实现完整的像素级 RSSM（确定性 GRU + 先验 MLP + 后验 MLP）与编码器、解码器、奖励头
    4. 实现两种工作模式：有观测时的 observe()（走后验）与无观测时的 imagine()（走先验）
    5. 用自动求导做三项结构自检：先验网络只从 KL 拿梯度、h_t 没看过当前帧 o_t、未来重建的梯度会回传给过去的编码器

核心概念：
    - 循环状态空间模型（Recurrent State-Space Model, RSSM）：(h_t, z_t) 共同构成模型状态
    - 先验 p(z_t | h_t)：没看 o_t 时对 z_t 的预测（动力学模型）
    - 后验 q(z_t | h_t, o_t)：看了 o_t 后对 z_t 的修正（表示模型）
    - 对角高斯的 KL 闭式解：均值错开项 + 宽窄不同项，对隐状态维度求和

前置知识：
    - 02-sequential-vae-elbo.py（序列 ELBO）
    - 03-deterministic-vs-stochastic.py（为什么要同时保留确定性路径 h_t 和随机路径 z_t）

对应概念文档：
    - CONCEPT.md 第 4 节「两个高斯之间的 KL 散度」（4.1–4.6）、第 6 节「RSSM：结构与数据流」

预计运行时间：约 15 秒（CPU / MPS）
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-03")
SEED = 0
DEVICE = get_device()

# PlaNet 论文（Hafner et al. 2019）的模型维度，04–08 节统一使用这一组常数
IMAGE_CHANNELS = 3
IMAGE_SIZE = 64
EMBED_DIM = 1024
H_DIM = 200  # 确定性状态维数
Z_DIM = 30   # 随机状态维数
HIDDEN_DIM = 200
ACTION_DIM = 1  # Pendulum-v1 的力矩是 1 维
MIN_STD = 0.1


# ==================== 第一部分：RSSM 的五个部件与两种模式 ====================


def introduction():
    print("=" * 60)
    print("第一部分：RSSM 的五个部件与两种工作模式")
    print("=" * 60)
    print("""
上一节看到：状态传递既要"记得住"（确定性的 h_t），又要"容得下多种可能"（随机采样的 z_t）。
把两条路径合在一起，配上图像编码器、解码器和奖励头，就是 PlaNet 与后续 Dreamer 系列共用的核心结构——
循环状态空间模型（Recurrent State-Space Model, RSSM）。

五个部件各司其职：
    1. 确定性状态模型   h_t = f_θ(h_{t-1}, z_{t-1}, a_{t-1})     GRU 单元，负责保存历史
    2. 先验（转移模型） p_θ(z_t | h_t)                           MLP，没看 o_t 时预测 z_t
    3. 后验（表示模型） q_θ(z_t | h_t, o_t)                      卷积编码器 + MLP，看了 o_t 后修正 z_t
    4. 观测模型         p_θ(o_t | h_t, z_t)                      转置卷积解码器，还原 64×64 画面
    5. 奖励模型         p_θ(r_{t-1} | h_t, z_t)                  MLP，预测到达这一步拿到的奖励

它只有两个入口，差别只在 z_t 从哪采：

    observe（有观测：训练、真实交互时跟踪信念）          imagine（无观测：开环预测、规划）
    ─────────────────────────────────────────          ─────────────────────────────────
    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])             h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    先验 (μ_p, σ_p) = PriorMLP(h_t)                    先验 (μ_p, σ_p) = PriorMLP(h_t)
    后验 (μ_q, σ_q) = PostMLP(h_t, Enc(o_t))           （没有 o_t，不走后验）
    z_t ~ N(μ_q, σ_q²)   ← 从后验采                    z_t ~ N(μ_p, σ_p²)   ← 从先验采
    （先验参数留着算 KL）                               （全程不需要解码图像）
    """)


# ==================== 第二部分：两个对角高斯的 KL 闭式解 ====================


def gaussian_kl(post_mean, post_std, prior_mean, prior_std):
    """计算对角高斯 KL( N(μ_q, σ_q²) ‖ N(μ_p, σ_p²) )，对最后一维（隐变量维度）求和。

    输入形状均为 [..., Z]，返回形状为 [...]。公式见 CONCEPT.md 第 4.3–4.4 节：
        KL = Σ_i [ log(σ_{p,i} / σ_{q,i}) + (σ_{q,i}² + (μ_{q,i} − μ_{p,i})²) / (2 σ_{p,i}²) − 1/2 ]
    """
    var_ratio = (post_std / prior_std).pow(2)
    mean_shift = ((post_mean - prior_mean) / prior_std).pow(2)
    kl_per_dim = torch.log(prior_std / post_std) + 0.5 * (var_ratio + mean_shift - 1.0)
    return kl_per_dim.sum(dim=-1)


def gaussian_kl_decomposed(post_mean, post_std, prior_mean, prior_std):
    """把 KL 拆成"均值错开"和"宽窄不同"两部分（CONCEPT.md 第 4.3 节），用于教学诊断。"""
    mean_term = (0.5 * ((post_mean - prior_mean) / prior_std).pow(2)).sum(dim=-1)
    shape_term = (torch.log(prior_std / post_std) + 0.5 * (post_std / prior_std).pow(2) - 0.5).sum(dim=-1)
    return mean_term, shape_term


def verify_kl_and_gradients(n_steps):
    print("\n" + "=" * 60)
    print("第二部分：对角高斯的 KL 闭式解与梯度方向")
    print("=" * 60)

    # 1. 复现 CONCEPT.md 第 4.5 节的四个手算例子
    t = lambda x: torch.tensor(x, dtype=torch.float32)
    kl_1 = gaussian_kl(t([1.0]), t([0.5]), t([0.0]), t([1.0])).item()
    m_1, s_1 = gaussian_kl_decomposed(t([1.0]), t([0.5]), t([0.0]), t([1.0]))
    kl_rev = gaussian_kl(t([0.0]), t([1.0]), t([1.0]), t([0.5])).item()
    kl_2d = gaussian_kl(t([1.0, 0.2]), t([0.5, 0.9]), t([0.0, 0.0]), t([1.0, 1.0])).item()
    kl_modest = gaussian_kl(t([0.1]), t([0.1]), t([0.0]), t([0.1])).item()
    kl_overconf = gaussian_kl(t([0.1]), t([0.1]), t([0.0]), t([0.01])).item()

    print(f"""
核对 CONCEPT.md 第 4.5 节的四个手算例子：
    例一 单维 KL(N(1, 0.5²) ‖ N(0, 1²))       = {kl_1:.3f} nat（均值错开 {m_1.item():.3f} + 宽窄不同 {s_1.item():.3f}）
    例二 对调方向 KL(N(0, 1²) ‖ N(1, 0.5²))   = {kl_rev:.3f} nat（是正方向的 {kl_rev / kl_1:.1f} 倍，KL 不对称）
    例三 两维独立高斯相加                      = {kl_2d:.3f} nat（第一维 0.818 + 第二维 0.030）
    例四 先验标准差 0.1 → 0.01（偏差同为 0.1） = {kl_modest:.1f} nat → {kl_overconf:.1f} nat（过度自信的先验代价暴涨）
    """)
    assert abs(kl_1 - 0.818) < 1e-3 and abs(kl_rev - 2.807) < 1e-3
    assert abs(kl_2d - 0.848) < 1e-3 and abs(kl_overconf - 97.2) < 0.1

    # 2. 与 PyTorch 官方实现对比 + 相同分布为 0
    gen = torch.Generator().manual_seed(SEED)
    qm = torch.randn(4, 10, Z_DIM, generator=gen)
    qs = torch.rand(4, 10, Z_DIM, generator=gen) + 0.1
    pm = torch.randn(4, 10, Z_DIM, generator=gen)
    ps = torch.rand(4, 10, Z_DIM, generator=gen) + 0.1
    ref = kl_divergence(Normal(qm, qs), Normal(pm, ps)).sum(dim=-1)
    max_diff = (gaussian_kl(qm, qs, pm, ps) - ref).abs().max().item()
    zero_kl = gaussian_kl(qm, qs, qm, qs).abs().max().item()
    print(f"  ✓ 与 torch.distributions.kl_divergence 随机张量 [4, 10, 30] 对比：最大绝对误差 {max_diff:.2e}")
    print(f"  ✓ 后验等于先验时 gaussian_kl 的最大绝对值：{zero_kl:.2e}")
    assert max_diff < 1e-4 and zero_kl < 1e-6

    # 3. 验证第 4.6 节：固定一批后验分布，优化先验的 (μ_p, σ_p)，看收敛到哪里
    # 设后验均值 μ_q 在不同样本间服从均值 1.2、标准差 0.6 的分布，后验本身的标准差固定 σ_q = 0.4
    # 理论最优：μ_p* = E[μ_q]，(σ_p*)² = σ_q² + E[(μ_q − μ_p*)²]
    mu_q_samples = 1.2 + 0.6 * torch.randn(2048, 1, generator=gen)
    sigma_q = torch.full_like(mu_q_samples, 0.4)
    theory_mu = mu_q_samples.mean().item()
    theory_std = math.sqrt(0.4**2 + ((mu_q_samples - theory_mu) ** 2).mean().item())

    mu_p = nn.Parameter(torch.tensor([-0.5]))
    raw_std_p = nn.Parameter(torch.tensor([0.0]))
    opt = torch.optim.Adam([mu_p, raw_std_p], lr=0.05)
    traj_mu, traj_std = [], []
    for _ in range(n_steps):
        std_p = F.softplus(raw_std_p) + 1e-3
        loss = gaussian_kl(mu_q_samples, sigma_q, mu_p.expand_as(mu_q_samples), std_p.expand_as(sigma_q)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        traj_mu.append(mu_p.item())
        traj_std.append((F.softplus(raw_std_p) + 1e-3).item())

    print(f"""
验证 CONCEPT.md 第 4.6 节"先验方差在学习自己的预测误差"：
    优化得到的先验：μ_p = {traj_mu[-1]:.3f}，σ_p = {traj_std[-1]:.3f}
    闭式推导最优解：μ_p* = {theory_mu:.3f}，√(σ_q² + E[(μ_q − μ_p*)²]) = {theory_std:.3f}
    """)

    # 画 KL 解剖图：左图是 KL 随 (μ_q − μ_p) 和 σ_p 的等高线，右图是先验参数收敛轨迹
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    deltas = np.linspace(-2.0, 2.0, 120)
    for sp, ls in [(0.3, "-"), (0.6, "--"), (1.2, ":")]:
        kl_curve = [gaussian_kl(t([d]), t([0.4]), t([0.0]), t([sp])).item() for d in deltas]
        axes[0].plot(deltas, kl_curve, linestyle=ls, linewidth=2, label=f"先验 σ_p = {sp}（后验 σ_q = 0.4）")
    axes[0].set_ylim(0, 12)
    axes[0].set_xlabel("均值偏差 μ_q − μ_p")
    axes[0].set_ylabel("KL(q ‖ p)（nat）")
    axes[0].set_title("先验越窄（越自信），猜错均值时代价越陡")
    axes[0].legend(fontsize=8.5)

    steps_x = np.arange(1, n_steps + 1)
    axes[1].plot(steps_x, traj_mu, linewidth=2, label="学到的先验均值 μ_p")
    axes[1].axhline(theory_mu, color="C0", linestyle="--", alpha=0.7, label=f"理论均值 E[μ_q] = {theory_mu:.2f}")
    axes[1].plot(steps_x, traj_std, linewidth=2, color="C1", label="学到的先验标准差 σ_p")
    axes[1].axhline(theory_std, color="C1", linestyle="--", alpha=0.7, label=f"理论标准差 = {theory_std:.2f}")
    axes[1].set_xlabel("梯度下降步数")
    axes[1].set_ylabel("参数值")
    axes[1].set_title("KL 把先验推向\"猜对均值 + 如实报告误差\"")
    axes[1].legend(fontsize=8.5)
    save_figure(fig, OUT / "04-kl-anatomy.png")


# ==================== 第三部分：标量手算核对 ====================


def verify_scalar_rssm():
    """复现 CONCEPT.md 第 6.5 节的标量两步手算例子。"""
    print("\n" + "=" * 60)
    print("第三部分：复现 CONCEPT.md 第 6.5 节的标量手算例子")
    print("=" * 60)

    # 第 1 步：有观测（h_0 = 0, z_0 = 0, a_0 = 1, e_1 = 1.0，抽到 ε_1 = 0.5）
    h0, z0, a0, e1 = 0.0, 0.0, 1.0, 1.0
    h1 = math.tanh(0.5 * h0 + 0.8 * z0 + 0.3 * a0)
    mu_p1, std_p1 = 0.9 * h1, 0.5
    mu_q1, std_q1 = 0.5 * h1 + 0.5 * e1, 0.2
    kl1 = gaussian_kl(torch.tensor([mu_q1]), torch.tensor([std_q1]),
                      torch.tensor([mu_p1]), torch.tensor([std_p1])).item()
    z1 = mu_q1 + std_q1 * 0.5

    # 第 2 步：无观测（a_1 = -1，抽到 ε_2 = -1.0）
    a1 = -1.0
    h2 = math.tanh(0.5 * h1 + 0.8 * z1 + 0.3 * a1)
    mu_p2, std_p2 = 0.9 * h2, 0.5
    z2 = mu_p2 + std_p2 * (-1.0)

    print(f"  第 1 步（observe）：h_1 = {h1:.3f}，先验 μ_p = {mu_p1:.3f}，后验 μ_q = {mu_q1:.3f}，"
          f"KL = {kl1:.3f} nat，采样 z_1 = {z1:.3f}")
    print(f"  第 2 步（imagine）：h_2 = {h2:.3f}，先验 μ_p = {mu_p2:.3f}，采样 z_2 = {z2:.3f}")
    assert abs(h1 - 0.291) < 1e-3 and abs(kl1 - 0.790) < 2e-3 and abs(z1 - 0.746) < 1e-3
    assert abs(h2 - 0.416) < 1e-3 and abs(mu_p2 - 0.374) < 1e-3 and abs(z2 - (-0.126)) < 2e-3
    print("  ✓ 全部数值与 CONCEPT.md 第 6.5 节一致。")


# ==================== 第四部分：从零实现完整的 RSSM 与世界模型 ====================


class ConvEncoder(nn.Module):
    """64×64×3 图像编码器（沿用 Ha & Schmidhuber 2018 / PlaNet 结构）。

    输入 [B, 3, 64, 64]（像素已缩放到 [-0.5, 0.5]） → 输出 e_t [B, 1024]
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=4, stride=2), nn.ReLU(),   # [B, 32, 31, 31]
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),               # [B, 64, 14, 14]
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU(),              # [B, 128, 6, 6]
            nn.Conv2d(128, 256, kernel_size=4, stride=2), nn.ReLU(),             # [B, 256, 2, 2]
            nn.Flatten(),                                                        # [B, 1024]
        )

    def forward(self, obs):
        return self.conv(obs)


class ConvDecoder(nn.Module):
    """从模型状态 (h_t, z_t) 解码出 64×64×3 图像的均值。

    注意最后一层不加 Sigmoid：观测模型 p(o_t | h_t, z_t) 是方差为 1 的高斯分布，
    网络输出的是均值 ô_t ∈ R^{3×64×64}，目标像素范围是 [-0.5, 0.5]（CONCEPT.md 第 7.3 节）。
    """

    def __init__(self, state_dim=H_DIM + Z_DIM):
        super().__init__()
        self.fc = nn.Linear(state_dim, EMBED_DIM)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(EMBED_DIM, 128, kernel_size=5, stride=2), nn.ReLU(),  # [B, 128, 5, 5]
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), nn.ReLU(),         # [B, 64, 13, 13]
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), nn.ReLU(),          # [B, 32, 30, 30]
            nn.ConvTranspose2d(32, IMAGE_CHANNELS, kernel_size=6, stride=2),         # [B, 3, 64, 64]
        )

    def forward(self, features):
        x = self.fc(features).view(-1, EMBED_DIM, 1, 1)
        return self.deconv(x)


class RewardHead(nn.Module):
    """奖励预测头：features [..., H+Z] → 标量奖励均值 [...]"""

    def __init__(self, state_dim=H_DIM + Z_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class RSSM(nn.Module):
    """循环状态空间模型（CONCEPT.md 第 6.2、6.6 节）。

    observe_step 与 imagine_step 共用 _deterministic_step 和 prior，
    从结构上杜绝"训练和想象的输入拼接顺序不一致"这一类隐蔽 bug。
    """

    def __init__(self, action_dim=ACTION_DIM, h_dim=H_DIM, z_dim=Z_DIM,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, min_std=MIN_STD):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.min_std = min_std

        self.inp_fc = nn.Linear(z_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, h_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim),
        )
        self.post_net = nn.Sequential(
            nn.Linear(h_dim + embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * z_dim),
        )

    def initial_state(self, batch_size, device):
        return (torch.zeros(batch_size, self.h_dim, device=device),
                torch.zeros(batch_size, self.z_dim, device=device))

    def _split_dist(self, raw):
        mean, raw_std = raw.chunk(2, dim=-1)
        return mean, F.softplus(raw_std) + self.min_std

    def _deterministic_step(self, h_prev, z_prev, a_prev):
        x = F.relu(self.inp_fc(torch.cat([z_prev, a_prev], dim=-1)))
        return self.gru(x, h_prev)

    def prior(self, h_t):
        return self._split_dist(self.prior_net(h_t))

    def posterior(self, h_t, embed_t):
        return self._split_dist(self.post_net(torch.cat([h_t, embed_t], dim=-1)))

    @staticmethod
    def _sample(mean, std, sample, generator=None):
        if not sample:
            return mean
        eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        return mean + std * eps

    def observe_step(self, h_prev, z_prev, a_prev, embed_t, sample=True, generator=None):
        """单步有观测更新：返回 h_t, z_t, (prior_mean, prior_std), (post_mean, post_std)"""
        h_t = self._deterministic_step(h_prev, z_prev, a_prev)
        prior_mean, prior_std = self.prior(h_t)
        post_mean, post_std = self.posterior(h_t, embed_t)
        z_t = self._sample(post_mean, post_std, sample, generator)
        return h_t, z_t, (prior_mean, prior_std), (post_mean, post_std)

    def imagine_step(self, h_prev, z_prev, a_prev, sample=True, generator=None):
        """单步无观测想象：返回 h_t, z_t, (prior_mean, prior_std)"""
        h_t = self._deterministic_step(h_prev, z_prev, a_prev)
        prior_mean, prior_std = self.prior(h_t)
        z_t = self._sample(prior_mean, prior_std, sample, generator)
        return h_t, z_t, (prior_mean, prior_std)

    def observe(self, embeds, actions, state=None, sample=True):
        """对整段编码序列运行滤波后验。

        embeds [B, T, 1024]，actions [B, T, A]（actions[:, t] 是到达第 t 帧之前执行的动作）
        返回字典，每个张量形状均为 [B, T, ·]。
        """
        batch_size, seq_len, _ = embeds.shape
        h, z = self.initial_state(batch_size, embeds.device) if state is None else state
        keys = ("h", "z", "prior_mean", "prior_std", "post_mean", "post_std")
        lists = {k: [] for k in keys}
        for t in range(seq_len):
            h, z, (pm, ps), (qm, qs) = self.observe_step(h, z, actions[:, t], embeds[:, t], sample=sample)
            for k, val in zip(keys, (h, z, pm, ps, qm, qs)):
                lists[k].append(val)
        out = {k: torch.stack(v, dim=1) for k, v in lists.items()}
        out["features"] = torch.cat([out["h"], out["z"]], dim=-1)  # [B, T, 230]
        return out

    def imagine(self, h, z, actions, sample=True, generator=None):
        """从状态 (h, z) 出发，不看观测连续想象 H 步。

        h [N, 200]，z [N, 30]，actions [N, H, A] → features [N, H, 230]
        """
        feats = []
        for k in range(actions.shape[1]):
            h, z, _ = self.imagine_step(h, z, actions[:, k], sample=sample, generator=generator)
            feats.append(torch.cat([h, z], dim=-1))
        return torch.stack(feats, dim=1)


class WorldModel(nn.Module):
    """完整的 PlaNet 世界模型：编码器 + RSSM + 解码器 + 奖励头。"""

    def __init__(self, action_dim=ACTION_DIM):
        super().__init__()
        self.encoder = ConvEncoder()
        self.rssm = RSSM(action_dim=action_dim)
        self.decoder = ConvDecoder()
        self.reward_head = RewardHead()

    def encode_seq(self, obs_seq):
        """obs_seq [B, T, 3, 64, 64] → embeds [B, T, 1024]"""
        b, t = obs_seq.shape[:2]
        return self.encoder(obs_seq.flatten(0, 1)).view(b, t, -1)

    def decode_seq(self, features):
        """features [B, T, 230] → recon [B, T, 3, 64, 64]"""
        b, t = features.shape[:2]
        return self.decoder(features.flatten(0, 1)).view(b, t, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


# ==================== 第五部分：形状、参数量与自动求导自检 ====================


def verify_rssm_mechanics():
    print("\n" + "=" * 60)
    print("第四部分：形状核对、可复现性与梯度通路验证")
    print("=" * 60)

    wm = WorldModel().to(DEVICE)
    b, seq_len, n_cand, horizon = 2, 5, 8, 12
    obs = torch.randn(b, seq_len, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE) * 0.3
    actions = torch.randn(b, seq_len, ACTION_DIM, device=DEVICE)

    embeds = wm.encode_seq(obs)
    out = wm.rssm.observe(embeds, actions)
    recon = wm.decode_seq(out["features"])
    rewards = wm.reward_head(out["features"])

    h_last = out["h"][:1, -1].expand(n_cand, -1)
    z_last = out["z"][:1, -1].expand(n_cand, -1)
    cand_actions = torch.randn(n_cand, horizon, ACTION_DIM, device=DEVICE)
    gen1 = torch.Generator(device=DEVICE).manual_seed(42)
    gen2 = torch.Generator(device=DEVICE).manual_seed(42)
    imag_1 = wm.rssm.imagine(h_last, z_last, cand_actions, generator=gen1)
    imag_2 = wm.rssm.imagine(h_last, z_last, cand_actions, generator=gen2)

    print("\n  1. 关键张量形状核对（对照 CONCEPT.md 第 6.7 节表）：")
    for name, tensor, expected in [
        ("obs", obs, [b, seq_len, 3, 64, 64]),
        ("embeds", embeds, [b, seq_len, 1024]),
        ("h", out["h"], [b, seq_len, 200]),
        ("z", out["z"], [b, seq_len, 30]),
        ("prior_mean", out["prior_mean"], [b, seq_len, 30]),
        ("post_mean", out["post_mean"], [b, seq_len, 30]),
        ("features", out["features"], [b, seq_len, 230]),
        ("recon", recon, [b, seq_len, 3, 64, 64]),
        ("rewards", rewards, [b, seq_len]),
        ("imagine 输出", imag_1, [n_cand, horizon, 230]),
    ]:
        assert list(tensor.shape) == expected, f"{name} 形状不符：{tensor.shape} vs {expected}"
        print(f"     {name:<14} {str(list(tensor.shape)):<22} ✓")

    # 参数量拆解：直观看到"为什么隐空间规划不解码能快一个数量级"
    counts = {
        "卷积编码器": sum(p.numel() for p in wm.encoder.parameters()),
        "转置卷积解码器": sum(p.numel() for p in wm.decoder.parameters()),
        "RSSM（GRU + 先验 + 后验）": sum(p.numel() for p in wm.rssm.parameters()),
        "奖励头": sum(p.numel() for p in wm.reward_head.parameters()),
    }
    total_params = sum(counts.values())
    imag_params = (sum(p.numel() for p in wm.rssm.inp_fc.parameters())
                   + sum(p.numel() for p in wm.rssm.gru.parameters())
                   + sum(p.numel() for p in wm.rssm.prior_net.parameters())
                   + counts["奖励头"])
    print(f"\n  2. 参数量拆解（共 {total_params:,}）：")
    for name, c in counts.items():
        print(f"     {name:<26} {c:>10,}  ({c / total_params:>5.1%})")
    print(f"     ──▶ 规划时 imagine + 奖励头只用到其中 {imag_params:,} 个参数（占总模型的 {imag_params / total_params:.1%}）！")

    # 可复现性与先验一致性自检
    repro_diff = (imag_1 - imag_2).abs().max().item()
    _, _, (pm_imag, ps_imag) = wm.rssm.imagine_step(out["h"][:, 2], out["z"][:, 2], actions[:, 3], sample=False)
    prior_match = (pm_imag - out["prior_mean"][:, 3]).abs().max().item()
    assert repro_diff == 0.0 and prior_match < 1e-6
    print(f"\n  3. 一致性自检：同种子 imagine 最大差 = {repro_diff:.1e}；observe 与 imagine_step 先验差 = {prior_match:.1e} ✓")

    # 4. 自动求导验证 CONCEPT.md 第 6.8、7.4 节的三条核心性质
    wm_cpu = WorldModel()
    obs_g = (torch.randn(2, seq_len, 3, 64, 64) * 0.3).requires_grad_(True)
    act_g = torch.randn(2, seq_len, ACTION_DIM)
    emb_g = wm_cpu.encode_seq(obs_g)
    out_g = wm_cpu.rssm.observe(emb_g, act_g)

    # (a) h_t 没看过当前帧 o_t，z_t 看过 o_t
    t_idx = 3  # 第 4 步（0-based 下标 3）
    grad_h_wrt_obs = torch.autograd.grad(out_g["h"][:, t_idx].sum(), obs_g, retain_graph=True)[0]
    grad_z_wrt_obs = torch.autograd.grad(out_g["z"][:, t_idx].sum(), obs_g, retain_graph=True)[0]
    h_norms = [grad_h_wrt_obs[:, k].norm().item() for k in range(seq_len)]
    z_norms = [grad_z_wrt_obs[:, k].norm().item() for k in range(seq_len)]
    assert h_norms[t_idx] == 0.0 and h_norms[t_idx - 1] > 0.0 and z_norms[t_idx] > 0.0

    # (b) 第 4 步的重建损失对过去各帧 o_1..o_4 的梯度（验证"表示为预测服务"跨时间回传）
    recon_t = wm_cpu.decoder(out_g["features"][:, t_idx])
    loss_recon_t = 0.5 * (recon_t - obs_g[:, t_idx].detach()).pow(2).sum(dim=(1, 2, 3)).mean()
    grad_recon_wrt_obs = torch.autograd.grad(loss_recon_t, obs_g, retain_graph=True)[0]
    recon_obs_norms = [grad_recon_wrt_obs[:, k].norm().item() for k in range(seq_len)]

    # (c) 比较"只有重建+奖励"与"加上 KL"时，先验 MLP 拿到的梯度范数
    wm_cpu.zero_grad()
    rew_t = wm_cpu.reward_head(out_g["features"]).pow(2).mean()
    (loss_recon_t + rew_t).backward(retain_graph=True)
    prior_grad_no_kl = sum(0.0 if p.grad is None else p.grad.norm().item() for p in wm_cpu.rssm.prior_net.parameters())
    post_grad_no_kl = sum(0.0 if p.grad is None else p.grad.norm().item() for p in wm_cpu.rssm.post_net.parameters())

    wm_cpu.zero_grad()
    kl_loss = gaussian_kl(out_g["post_mean"], out_g["post_std"], out_g["prior_mean"], out_g["prior_std"]).mean()
    (loss_recon_t + rew_t + kl_loss).backward()
    prior_grad_with_kl = sum(p.grad.norm().item() for p in wm_cpu.rssm.prior_net.parameters())
    assert prior_grad_no_kl == 0.0 and prior_grad_with_kl > 0.0

    print(f"""
  4. 自动求导验证三条梯度性质（CONCEPT.md 第 6.8、7.4 节）：
     ① h_4 对当前帧 o_4 的梯度范数 = {h_norms[t_idx]:.1e}（没看过 o_4！），对上一帧 o_3 = {h_norms[t_idx - 1]:.2e}；
       z_4 对当前帧 o_4 的梯度范数 = {z_norms[t_idx]:.2e}（看过 o_4！）。因此解码器和奖励头必须同时读 [h_t, z_t]。
     ② 只有重建 + 奖励损失时：后验 MLP 梯度范数 = {post_grad_no_kl:.2e}，先验 MLP 梯度范数 = {prior_grad_no_kl:.1e}（精确为 0！）；
       加上 KL 项之后：先验 MLP 梯度范数 = {prior_grad_with_kl:.2e}。先验网络只靠 KL 项学习动力学。
     ③ 第 4 步单步重建损失对输入帧 o_1..o_5 的梯度范数：{[f'{v:.2e}' for v in recon_obs_norms]}
       不仅对当前帧 o_4 非零，还沿着 h_4 ← z_3 ← Enc(o_3) 回传到了 o_1..o_3（而未来的 o_5 精确为 0）——
       编码器因此被迫提取对未来预测有用的信息。
    """)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(1, seq_len + 1)
    width = 0.26
    # 归一化到各自最大值，方便在同一张图上看清"哪些时间步严格为 0、哪些非零"
    norm_bar = lambda arr: np.array(arr) / max(max(arr), 1e-12)
    axes[0].bar(x - width, norm_bar(h_norms), width=width, label="∂h_4 / ∂o_k（相对值）")
    axes[0].bar(x, norm_bar(z_norms), width=width, label="∂z_4 / ∂o_k（相对值）")
    axes[0].bar(x + width, norm_bar(recon_obs_norms), width=width, label="∂L_重建(t=4) / ∂o_k（相对值）")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"o_{k}" for k in x])
    axes[0].set_ylabel("归一化梯度范数（最高为 1）")
    axes[0].set_title("第 4 步的状态与损失依赖哪几帧观测")
    axes[0].legend(fontsize=8.5)

    bars = ["先验 MLP\n（无 KL）", "先验 MLP\n（有 KL）", "后验 MLP\n（无 KL）"]
    vals = [prior_grad_no_kl, prior_grad_with_kl, post_grad_no_kl]
    colors = ["lightgray", "C0", "C1"]
    axes[1].bar(bars, vals, color=colors, width=0.5)
    for idx, v in enumerate(vals):
        axes[1].text(idx, v + max(vals) * 0.02, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_ylabel("参数梯度范数之和")
    axes[1].set_title("先验网络只有 KL 一个训练信号来源")
    save_figure(fig, OUT / "04-gradient-flow.png")

    print("  注：线性探针检验\"训练后的 (h_t, z_t) 里确实包含角速度\"需要一个在像素摆上训练好的模型，")
    print("      放在下一节 05-world-model-training.py 训练完成后立即运行。")


# ==================== 第六部分：练习与思考 ====================


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    kl_wrong_mean = gaussian_kl(
        torch.tensor([1.0, 0.2]), torch.tensor([0.5, 0.9]),
        torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])
    ).item() / 2.0
    print(f"""
练习 1：把求和错写成求平均会怎样
    gaussian_kl() 最后一步是对 Z_DIM = 30 求和。如果错写成 .mean(dim=-1)，
    两维手算例子（真实 KL = 0.848 nat）会算出 {kl_wrong_mean:.3f}；30 维时 KL 会被缩小 30 倍。

练习 1 参考答案：
    独立维度的联合密度相乘，对数密度相加，所以多维 KL 必须对维度求和（CONCEPT.md 第 4.4 节）。
    错写成 mean 相当于把 β 偷偷改成了 1/30，先验受到的约束变弱 30 倍，开环预测会严重跑偏。

练习 2：为什么训练时不能从先验采样 z_t 去解码
    试想把 observe_step 里的 z_t 改成从 (prior_mean, prior_std) 采样，再送进解码器算重建损失，会发生什么？

练习 2 参考答案：
    训练刚开始时先验一无所知，采样出的 z_t 跟当前帧 o_t 毫无关系，解码器拿到噪声只能输出模糊平均图；
    更糟的是，后验网络 q(z_t | h_t, o_t) 将完全失去重建梯度，KL 只会把后验拉向同样随机的先验，整个模型坍塌。

练习 3：解码器只吃 h_t、不吃 z_t 行不行
    结合第四部分自动求导测出的 ∂h_4 / ∂o_4 = 0，回答为什么解码器和奖励头的输入必须是 [h_t, z_t]。

练习 3 参考答案：
    h_t 只汇总了到 t−1 为止的信息，还没看过第 t 步的观测 o_t，也没有包含第 t 步刚掷出来的随机分支。
    只喂 h_t 等于逼着模型在没看 o_t 的情况下"盲画" o_t，退化成了单步确定性预测。

思考题 1：为什么 ConvDecoder 的最后一层不加 Sigmoid 或 Tanh？

思考题 1 参考答案：
    观测模型 p(o_t | h_t, z_t) = N(ô_t, I) 把网络输出当作高斯分布的均值。
    如果加了 Sigmoid 或 Tanh，边界附近的梯度会饱和变小，而且高斯均值在数学上不需要硬截断；
    展示图像时再把 [-0.5, 0.5] 加 0.5 并裁剪到 [0, 1] 即可。

思考题 2：RSSM 的 ELBO 里为什么没有关于 h_t 的 KL 项？模型会不会把信息全藏进 h_t 来逃避 KL？

思考题 2 参考答案：
    h_t 在生成模型和推断模型里由同一个确定性函数 f_θ 从同样的输入算出，分布没有分歧，KL 恒为 0。
    但模型逃避不了 KL：观测 o_t 没有任何直接进入 h 的通路，任何来自画面的信息都必须先写入某一步的后验 z，
    在写入的那一步就已经交过 KL "过路费"了；h 只是把已经交过费的信息免费保存下去。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    quick = is_quick()
    if quick:
        print("（冒烟模式：迭代步数缩小）\n")
    n_steps = 80 if quick else 400

    introduction()
    verify_kl_and_gradients(n_steps)
    verify_scalar_rssm()
    verify_rssm_mechanics()
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：05-world-model-training.py（序列回放、三项损失、free nats 与角速度探针）")
    print("=" * 60)


if __name__ == "__main__":
    main()
