"""
序列 VAE 与序列 ELBO
====================

学习目标：
    1. 用两个能手算的例子核对 CONCEPT.md 第 2、3 节的数字：两状态隐马尔可夫模型的似然、一维高斯的 ELBO
    2. 从零实现一个序列 VAE（纯随机的隐状态空间模型）：学出来的先验 + 滤波后验 + 解码器
    3. 实现 sequence_elbo()：每一步"重建 − KL"，看清哪里求和、哪里求平均
    4. 对比"固定先验 N(0, I)"与"学出来的先验"：两者都能重建，只有后者能预测
    5. 看到 KL 项的读法：这一步的观测让模型有多意外

核心概念：
    - 隐状态空间模型（latent state-space model）：转移 p(z_t | z_{t-1}, a_{t-1}) + 观测 p(o_t | z_t)
    - 滤波后验（filtering posterior）：q(z_t | z_{t-1}, a_{t-1}, o_t)，只看过去和现在
    - 序列 ELBO：Σ_t [ E_q log p(o_t | z_t) − KL( q(z_t | ·) ‖ p(z_t | ·) ) ]
    - 在序列模型里，"先验"就是世界模型的预测部分

前置知识：
    - 01-filtering-and-belief-state.py（预测—修正）
    - 旧课的 VAE（ELBO、重参数化）

对应概念文档：
    - CONCEPT.md 第 2 节「隐状态空间模型」、第 3 节「序列 ELBO」

预计运行时间：约 2 分钟（CPU）
"""

import itertools
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-03")
SEED = 0
DEVICE = torch.device("cpu")  # 玩具模型很小、序列循环很短，CPU 比 MPS 的启动开销更划算

# 玩具任务：一维"像素条"上的弹簧小球
O_DIM, A_DIM, Z_DIM, HIDDEN = 16, 1, 4, 128
CELLS = torch.linspace(-1.2, 1.2, O_DIM)  # 16 个"像素"的位置
BUMP_WIDTH = 0.15
OBS_STD = 0.1  # 观测噪声的标准差；解码器的方差也固定为它
DT, SPRING, DAMPING, FORCE = 0.1, 4.0, 0.2, 2.0
MIN_STD = 0.1  # 先验 / 后验标准差的下限
T_TRAIN, T_CONTEXT, T_PREDICT = 30, 10, 40
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


# ==================== 第一部分：问题从哪来 ====================


def introduction():
    print("=" * 60)
    print("第一部分：把 VAE 推广到序列")
    print("=" * 60)
    print("""
Phase 2 的 V（VAE）每张图各压各的，先验固定是 N(0, I)；M（MDN-RNN）另外训练，负责预测。
这一节把两件事写进同一个概率模型：

      z_0 ──▶ z_1 ──▶ z_2 ──▶ z_3 ──▶ ···        隐状态，沿时间传递（转移 = 先验）
          ▲    │   ▲    │   ▲    │
         a_0   ▼  a_1   ▼  a_2   ▼
              o_1      o_2      o_3              观测由隐状态生成（解码器）

训练目标是序列 ELBO（推导见 CONCEPT.md 第 3.3 节）：

    log p(o_1:T | a_<T) ≥ Σ_t [ E_q log p(o_t | z_t)  −  KL( q(z_t | z_{t-1}, a_{t-1}, o_t) ‖ p(z_t | z_{t-1}, a_{t-1}) ) ]
                               └──── 重建项 ────┘        └──── 后验：看了 o_t ────┘   └──── 先验：没看 o_t ────┘

和单步 VAE 唯一的区别：KL 比较的对象从固定的 N(0, I) 换成了"模型上一步对这一步的预测"。

本节分三块：
    1. 两个能手算的例子，核对 CONCEPT.md 里的数字
    2. 在一个低维的玩具序列上从零实现序列 VAE，对比固定先验与学出来的先验
    3. 故意让小球"瞬移"一次，看 KL 怎样报告"意外"
    """)


# ==================== 第二部分：两个手算例子 ====================


def hmm_likelihood_bruteforce():
    """CONCEPT.md 第 2.5 节：两状态隐马尔可夫模型，枚举全部隐状态路径求似然。"""
    print("\n" + "=" * 60)
    print("第二部分：两个手算例子")
    print("=" * 60)
    print("""
例一（第 2.5 节）：隐状态只有 A、B 两个取值，序列长 2，观测到 o_1 = 1, o_2 = 1。
似然 = 把所有隐状态路径的联合概率加起来。路径只有 4 条，可以直接枚举：
    """)
    init = {"A": 0.5, "B": 0.5}
    stay = 0.9
    p_one = {"A": 0.8, "B": 0.2}  # 在该状态下观测到 1 的概率
    joint = {}
    for path in itertools.product("AB", repeat=2):
        p = init[path[0]] * p_one[path[0]]
        p *= (stay if path[1] == path[0] else 1 - stay) * p_one[path[1]]
        joint[path] = p
        print(f"    {path[0]} → {path[1]}：{p:.3f}")
    likelihood = sum(joint.values())
    post_a = (joint[("A", "A")] + joint[("B", "A")]) / likelihood
    print(f"    合计 p(o_1 = 1, o_2 = 1) = {likelihood:.3f}，第 2 步在 A 的后验概率 = {post_a:.3f}")
    assert abs(likelihood - 0.322) < 1e-3 and abs(post_a - 0.919) < 1e-3
    print(f"""
  ✓ 与 CONCEPT.md 一致。2 个取值、2 步只有 4 条路径；30 维连续隐状态、50 步，这个求和就变成了
    没有闭式解的高维积分。所以要换成优化一个算得出来的下界：ELBO。
    """)
    return likelihood, post_a


def gaussian_elbo_table(n_mc):
    """CONCEPT.md 第 3.5 节：先验 N(0,1)、观测 N(z,1)、o = 1，三个不同后验的 ELBO。"""
    print(f"""例二（第 3.5 节）：单独拿出一个时间步。先验 N(0, 1)，观测模型 N(z, 1)，观测到 o = 1。
真实的对数似然可以直接算：o ~ N(0, 2)，log p(o = 1) = −1.516。真正的后验是 N(0.5, 0.5)。
用三个不同的后验算 ELBO：闭式解（重建项的期望 + KL 的闭式解），再用 {n_mc} 个样本的蒙特卡洛估计核对。
    """)
    obs = torch.tensor(1.0)
    prior = Normal(0.0, 1.0)
    true_logp = Normal(0.0, math.sqrt(2.0)).log_prob(obs).item()
    true_post = Normal(0.5, math.sqrt(0.5))
    rows = [("N(0, 1)：等于先验", 0.0, 1.0), ("N(0.8, 0.5²)：过于自信", 0.8, 0.5),
            ("N(0.5, 0.5)：真正的后验", 0.5, math.sqrt(0.5))]
    print(f"  {'后验 q':<22}{'重建项':>8}{'KL':>8}{'ELBO':>9}{'蒙特卡洛':>10}{'与真实值的差':>12}{'KL(q‖真后验)':>14}")
    gen = torch.Generator().manual_seed(SEED)
    results = []
    for name, mu, std in rows:
        q = Normal(mu, std)
        recon = -HALF_LOG_2PI - 0.5 * ((obs - mu) ** 2 + std**2)  # E_q[log N(o; z, 1)] 的闭式解
        kl = kl_divergence(q, prior)
        elbo = (recon - kl).item()
        z = mu + std * torch.randn(n_mc, generator=gen)  # 从后验采样
        mc = (Normal(z, 1.0).log_prob(obs) + prior.log_prob(z) - q.log_prob(z)).mean().item()
        gap = kl_divergence(q, true_post).item()
        results.append((elbo, mc, gap))
        print(f"  {name:<20}{recon.item():>10.3f}{kl.item():>8.3f}{elbo:>9.3f}{mc:>10.3f}{true_logp - elbo:>12.3f}{gap:>14.3f}")
    assert abs(results[2][0] - true_logp) < 1e-3
    print(f"""
  ✓ 三行的 ELBO 分别是 {results[0][0]:.3f}、{results[1][0]:.3f}、{results[2][0]:.3f}，与 CONCEPT.md 的表一致。
    最后两列逐行相等："ELBO 与真实对数似然的差"恰好等于"我们的后验与真正后验之间的 KL"（第 3.4 节）。
    后验等于真正的后验时，下界是紧的。KL 为 0 的第一行不是好事：它根本没用到观测。
    """)
    return results


# ==================== 第三部分：玩具序列 ====================


def render_strip(x):
    """把位置 x 画成 16 个像素上的一个高斯小鼓包。x [...] → [..., 16]"""
    return torch.exp(-0.5 * ((x[..., None] - CELLS) / BUMP_WIDTH) ** 2)


def simulate(n, steps, gen, teleport_at=None):
    """弹簧上的小球，受随机力推动。只看像素条，看不到位置和速度。

    时间对齐（与 CONCEPT.md 第 0.5 节一致）：下标 i 处存的是 a_i 和 o_{i+1}，
    即"先执行动作 a_i，再看到观测 o_{i+1}"。
    teleport_at：在这一步把小球瞬移到 −x（速度不变），用来制造"意外"。
    返回 obs [N, T, 16]（带噪），clean [N, T, 16]（无噪），actions [N, T, 1]
    """
    x = (torch.rand(n, generator=gen) - 0.5) * 1.2
    v = (torch.rand(n, generator=gen) - 0.5) * 2.4
    actions = torch.rand(n, steps, A_DIM, generator=gen) * 2 - 1
    positions = []
    for i in range(steps):
        v = v + DT * (-SPRING * x - DAMPING * v + FORCE * actions[:, i, 0])
        x = x + DT * v
        if teleport_at is not None and i == teleport_at:
            x = -x
        positions.append(x)
    clean = render_strip(torch.stack(positions, dim=1))
    obs = clean + OBS_STD * torch.randn(clean.shape, generator=gen)
    return obs, clean, actions


# ==================== 第四部分：序列 VAE ====================


def mlp(inp, out):
    return nn.Sequential(nn.Linear(inp, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, out))


class SequentialVAE(nn.Module):
    """纯随机的隐状态空间模型（CONCEPT.md 第 2.7 节）。

    learned_prior=True：  先验 p(z_t | z_{t-1}, a_{t-1}) 是一个网络（这就是动力学模型）
    learned_prior=False： 先验固定为 N(0, I)，等于每一帧各自做 VAE，模型里没有动力学
    两者的后验网络和解码器完全相同。
    """

    def __init__(self, learned_prior=True):
        super().__init__()
        self.learned_prior = learned_prior
        self.prior_net = mlp(Z_DIM + A_DIM, 2 * Z_DIM)  # [B, Z+A] → [B, 2Z]
        self.post_net = mlp(Z_DIM + A_DIM + O_DIM, 2 * Z_DIM)  # [B, Z+A+O] → [B, 2Z]
        self.decoder = mlp(Z_DIM, O_DIM)  # [B, Z] → [B, O]

    @staticmethod
    def _split(out):
        mean, raw = out.chunk(2, dim=-1)
        return mean, F.softplus(raw) + MIN_STD

    def prior(self, z_prev, a_prev):
        if not self.learned_prior:
            return torch.zeros_like(z_prev), torch.ones_like(z_prev)
        return self._split(self.prior_net(torch.cat([z_prev, a_prev], dim=-1)))

    def posterior(self, z_prev, a_prev, o_t):
        return self._split(self.post_net(torch.cat([z_prev, a_prev, o_t], dim=-1)))

    def forward(self, obs, actions):
        """沿时间展开：每一步算先验（没看 o_t）和后验（看了 o_t），从后验采样，传给下一步。

        obs [B, T, O]，actions [B, T, A] → 各 [B, T, Z] 的 prior_mean/std、post_mean/std、z
        """
        z = torch.zeros(obs.shape[0], Z_DIM, device=obs.device)  # z_0 = 0
        keys = ("prior_mean", "prior_std", "post_mean", "post_std", "z")
        out = {k: [] for k in keys}
        for t in range(obs.shape[1]):
            prior_mean, prior_std = self.prior(z, actions[:, t])
            post_mean, post_std = self.posterior(z, actions[:, t], obs[:, t])
            z = post_mean + post_std * torch.randn_like(post_mean)  # 重参数化：从后验采样
            for k, v in zip(keys, (prior_mean, prior_std, post_mean, post_std, z)):
                out[k].append(v)
        return {k: torch.stack(v, dim=1) for k, v in out.items()}

    @torch.no_grad()
    def predict(self, z, actions):
        """开环预测：没有观测，只走先验。z [B, Z]，actions [B, D, A] → 解码出的观测 [B, D, O]"""
        frames = []
        for d in range(actions.shape[1]):
            mean, std = self.prior(z, actions[:, d])
            z = mean + std * torch.randn_like(mean)
            frames.append(self.decoder(z))
        return torch.stack(frames, dim=1)


def gaussian_logp(x, mean, std):
    """对角高斯的对数密度，逐元素。"""
    return -0.5 * ((x - mean) / std) ** 2 - math.log(std) - HALF_LOG_2PI


def sequence_elbo(model, obs, actions):
    """序列 ELBO 的单样本估计（CONCEPT.md 第 3.7 节）。

    返回：
        elbo        标量，每个时间步的平均 ELBO（nat）：对观测维、隐状态维求和，对批次、时间求平均
        recon_logp  [B, T]，每一步的重建对数似然（对 16 个像素求和）
        kl          [B, T]，每一步的 KL（对 Z 维求和）
    """
    out = model(obs, actions)
    recon = model.decoder(out["z"])  # [B, T, O]
    recon_logp = gaussian_logp(obs, recon, OBS_STD).sum(-1)  # [B, T]
    kl = kl_divergence(Normal(out["post_mean"], out["post_std"]),
                       Normal(out["prior_mean"], out["prior_std"])).sum(-1)  # [B, T]
    elbo = (recon_logp - kl).mean()
    return elbo, recon_logp, kl


def train_model(learned_prior, data, n_iters, batch_size=64):
    obs, _, actions = data
    model = SequentialVAE(learned_prior).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(SEED)
    history = []
    for it in range(n_iters):
        idx = torch.randint(0, obs.shape[0], (batch_size,), generator=gen)
        elbo, recon_logp, kl = sequence_elbo(model, obs[idx], actions[idx])
        optimizer.zero_grad()
        (-elbo).backward()  # 最大化 ELBO = 最小化它的相反数
        nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
        history.append((elbo.item(), recon_logp.mean().item(), kl.mean().item()))
    return model, np.array(history)


def train_both(n_iters, n_train):
    print("\n" + "=" * 60)
    print("第三部分：玩具序列上的序列 VAE")
    print("=" * 60)
    print(f"""
玩具任务：一个小球挂在弹簧上，每步受一个随机的推力 a_t ∈ [−1, 1]。模型看不到位置和速度，
只看到一条 16 个"像素"的条带，小球所在的地方亮起一个小鼓包，再叠加标准差 {OBS_STD} 的噪声。

    像素条（某一时刻）： ░░░░░▒▓█▓▒░░░░░░        亮的位置 = 小球的位置
                         ← 位置 −1.2 … +1.2 →

和像素版的摆一样：单帧里只有位置，没有速度。

训练两个模型，后验网络和解码器完全相同，只差先验：
    学出来的先验   p(z_t | z_{{t-1}}, a_{{t-1}}) 是一个 MLP（动力学模型）
    固定先验       p(z_t) = N(0, I)，每一步都一样（Phase 2 的 V 就是这样）

数据：{n_train} 条长 {T_TRAIN} 步的序列。隐状态 Z = {Z_DIM} 维，训练 {n_iters} 步，Adam，批大小 64。
    """)
    gen = torch.Generator().manual_seed(SEED)
    data = simulate(n_train, T_TRAIN, gen)
    models, histories = {}, {}
    for name, learned in [("学出来的先验", True), ("固定先验", False)]:
        start = time.time()
        models[name], histories[name] = train_model(learned, data, n_iters)
        print(f"  {name}：训练用时 {time.time() - start:.0f} 秒，最后 100 步平均 ELBO {histories[name][-100:, 0].mean():.2f} nat/步")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, h in histories.items():
        for ax, col, title in zip(axes, range(3), ["ELBO（越高越好）", "重建项 E log p(o|z)", "KL 项"]):
            ax.plot(np.convolve(h[:, col], np.ones(20) / 20, mode="valid"), label=name)
            ax.set_title(title)
            ax.set_xlabel("训练步数")
            ax.set_ylabel("nat / 步")
    axes[0].legend()
    save_figure(fig, OUT / "02-training-curves.png")
    return models


# ==================== 第五部分：重建与预测 ====================


@torch.no_grad()
def evaluate(models, n_test):
    print(f"""
在 {n_test} 条新序列上评估。每条长 {T_CONTEXT + T_PREDICT} 步：
    前 {T_CONTEXT} 步有观测，走后验（滤波）；
    后 {T_PREDICT} 步不给观测，只给动作，走先验（开环预测），把采样出的 z 解码成像素条。
预测误差是解码结果和无噪声画面之间的均方误差（每个像素），单次采样，对 {n_test} 条序列平均。
    """)
    gen = torch.Generator().manual_seed(SEED + 1)
    obs, clean, actions = simulate(n_test, T_CONTEXT + T_PREDICT, gen)
    train_obs, train_clean, _ = simulate(2000, T_TRAIN, torch.Generator().manual_seed(SEED + 2))
    mean_frame = train_clean.mean(dim=(0, 1))
    future = clean[:, T_CONTEXT:]
    torch.manual_seed(SEED)

    results = {}
    print(f"  {'模型':<10}{'ELBO':>8}{'重建项':>9}{'KL':>7}{'重建误差':>10}" + "".join(f"{f'第{d}步':>9}" for d in (1, 5, 10, 20, 40)))
    for name, model in models.items():
        elbo, recon_logp, kl = sequence_elbo(model, obs[:, :T_TRAIN], actions[:, :T_TRAIN])
        out = model(obs[:, :T_CONTEXT], actions[:, :T_CONTEXT])
        recon_mse = (model.decoder(out["z"]) - clean[:, :T_CONTEXT]).pow(2).mean().item()
        pred = model.predict(out["z"][:, -1], actions[:, T_CONTEXT:])  # [N, D, O]
        mse = (pred - future).pow(2).mean(dim=(0, 2)).numpy()  # [D]
        results[name] = dict(elbo=elbo.item(), recon=recon_logp.mean().item(), kl=kl.mean().item(),
                             recon_mse=recon_mse, mse=mse, pred=pred[0].numpy())
        print(f"  {name:<8}{elbo.item():>10.2f}{recon_logp.mean().item():>9.2f}{kl.mean().item():>7.2f}{recon_mse:>10.4f}"
              + "".join(f"{mse[d - 1]:>10.4f}" for d in (1, 5, 10, 20, 40)))
    copy_last = (clean[:, T_CONTEXT - 1 : T_CONTEXT] - future).pow(2).mean(dim=(0, 2)).numpy()
    mean_mse = (mean_frame - future).pow(2).mean().item()
    print(f"  {'复制最后一帧':<8}{'':>34}" + "".join(f"{copy_last[d - 1]:>10.4f}" for d in (1, 5, 10, 20, 40)))
    print(f"  {'永远输出平均帧':<7}{'':>34}" + "".join(f"{mean_mse:>10.4f}" for _ in range(5)))
    print("  （ELBO、重建项、KL 的单位是 nat/步；重建误差和预测误差是每个像素的均方误差）")

    d_axis = np.arange(1, T_PREDICT + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, r in results.items():
        ax.plot(d_axis, r["mse"], linewidth=2, label=name)
    ax.plot(d_axis, copy_last, "--", label="基线：复制最后一帧")
    ax.axhline(mean_mse, color="gray", linestyle=":", label="基线：永远输出平均帧")
    ax.set_xlabel("开环预测的步数 d")
    ax.set_ylabel("每个像素的均方误差")
    ax.set_title("固定先验能重建，不能预测")
    ax.legend()
    save_figure(fig, OUT / "02-open-loop-error.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    ctx_recon = {name: m.decoder(m(obs[:1, :T_CONTEXT], actions[:1, :T_CONTEXT])["z"])[0].numpy() for name, m in models.items()}
    panels = [("真实画面（无噪声）", clean[0].numpy())]
    panels += [(f"{name}：前 {T_CONTEXT} 步重建，之后预测", np.concatenate([ctx_recon[name], results[name]["pred"]]))
               for name in models]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
                  extent=[0.5, T_CONTEXT + T_PREDICT + 0.5, -1.2, 1.2])
        ax.axvline(T_CONTEXT + 0.5, color="cyan", linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("时间步")
        ax.grid(False)
    axes[0].set_ylabel("像素条上的位置")
    fig.suptitle("每一列是一帧像素条；青色虚线右边不再给观测")
    save_figure(fig, OUT / "02-open-loop-strips.png")

    learned, fixed = results["学出来的先验"], results["固定先验"]
    print("\n怎么读这张表：\n")
    print(f"    1. 重建：两个模型都能把看过的帧画回来（重建误差 {learned['recon_mse']:.4f} vs {fixed['recon_mse']:.4f}，"
          f"噪声本身的方差是 {OBS_STD**2:.2f}）。")
    if fixed["mse"][0] > mean_mse * 0.8:
        print(f"    2. 预测：固定先验第 1 步的误差就是 {fixed['mse'][0]:.4f}，和\"永远输出平均帧\"（{mean_mse:.4f}）同一个量级。\n"
              f"       它的先验每一步都是 N(0, I)，和历史无关，采样出来的是一张随便的画面。模型里根本没有动力学。")
    else:
        print(f"    2. 预测：固定先验第 1 步的误差是 {fixed['mse'][0]:.4f}，平均帧是 {mean_mse:.4f}。")
    better = learned["mse"][0] < copy_last[0]
    print(f"    3. 学出来的先验第 1 步误差 {learned['mse'][0]:.4f}，"
          + ("比\"复制最后一帧\"还好" if better else "和\"复制最后一帧\"接近") +
          f"（{copy_last[0]:.4f}）：它知道小球在往哪边走。\n"
          f"       之后误差随步数增大（第 {T_PREDICT} 步 {learned['mse'][-1]:.4f}）：开环预测的误差会累积（第 8 节）。")
    print(f"    4. ELBO：学出来的先验 {learned['elbo']:.2f}，固定先验 {fixed['elbo']:.2f}。差别几乎全在 KL 项"
          f"（{learned['kl']:.2f} vs {fixed['kl']:.2f}）：\n"
          f"       能预测的模型，观测带来的\"意外\"少，每一步要付的码率就低。")
    return results, dict(copy_last=copy_last, mean_mse=mean_mse)


# ==================== 第六部分：KL = 意外 ====================


@torch.no_grad()
def surprise_experiment(models, n_test, teleport_at=15):
    print("\n" + "=" * 60)
    print("第四部分：KL 项在报告\"意外\"")
    print("=" * 60)
    print(f"""
CONCEPT.md 第 3.1 节说 KL 项可以读成"这一步的观测让模型有多意外"。做一个直接的检验：
同样的 {n_test} 条序列跑两遍，第二遍在第 {teleport_at + 1} 步让小球瞬移到镜像位置（x → −x），物理上不可能发生。
看每一步的 KL（对序列平均）。
    """)
    kls = {}
    for tele in (None, teleport_at):
        obs, _, actions = simulate(n_test, T_TRAIN, torch.Generator().manual_seed(SEED + 3), teleport_at=tele)
        for name, model in models.items():
            torch.manual_seed(SEED)
            _, _, kl = sequence_elbo(model, obs, actions)
            kls[(name, tele)] = kl.mean(dim=0).numpy()
    steps = range(teleport_at - 2, teleport_at + 5)
    print(f"  {'模型':<10}{'瞬移？':<6}" + "".join(f"{f't={s + 1}':>8}" for s in steps))
    for name in models:
        for tele in (None, teleport_at):
            print(f"  {name:<8}{'是' if tele else '否':<6}" + "".join(f"{kls[(name, tele)][s]:>8.2f}" for s in steps))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = np.arange(1, T_TRAIN + 1)
    for name, color in zip(models, ("C0", "C1")):
        ax.plot(t, kls[(name, None)], color=color, linestyle="--", label=f"{name}：正常序列")
        ax.plot(t, kls[(name, teleport_at)], color=color, linewidth=2, label=f"{name}：第 {teleport_at + 1} 步瞬移")
    ax.axvline(teleport_at + 1, color="gray", alpha=0.4)
    ax.set_xlabel("时间步 t")
    ax.set_ylabel("KL（nat）")
    ax.set_title("KL 项：这一步的观测让模型有多意外")
    ax.legend(fontsize=8)
    save_figure(fig, OUT / "02-kl-surprise.png")

    learned_n, learned_t = kls[("学出来的先验", None)], kls[("学出来的先验", teleport_at)]
    fixed_n, fixed_t = kls[("固定先验", None)], kls[("固定先验", teleport_at)]
    spike = learned_t[teleport_at] / max(learned_n[teleport_at], 1e-6)
    print(f"""
    学出来的先验：瞬移那一步的 KL 从 {learned_n[teleport_at]:.2f} 跳到 {learned_t[teleport_at]:.2f}（{spike:.1f} 倍），
    之后几步回落：后验把信念改到新位置之后，先验又能接着预测了。
    固定先验：瞬移那一步是 {fixed_n[teleport_at]:.2f} → {fixed_t[teleport_at]:.2f}。它的先验从来不预测，
    每一帧的码率都是"完整描述这一帧"的价钱，瞬移不瞬移对它没什么区别。
    这正是视频编码的类比：有运动预测的编码器，画面突变的那一帧码率暴涨；不做预测的编码器每帧都一样贵。
    """)
    return kls


# ==================== 第七部分：练习与思考 ====================


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    q, p = Normal(1.0, 0.5), Normal(0.0, 1.0)
    obs = 1.0
    best_std = math.sqrt(0.5)
    print(f"""
练习 1：T = 1 时序列 ELBO 退化成什么
    把 T_TRAIN 改成 1 重新训练"学出来的先验"。此时先验的输入是 z_0 = 0 和 a_0，模型和单步 VAE 有什么区别？

练习 1 参考答案：
    z_0 是固定的零向量，先验只剩一个依赖 a_0 的分布，没有任何"历史"可用，
    式子就是单步 VAE 的 ELBO（CONCEPT.md 第 3.3 节末尾）。只有一帧的序列里没有动力学可学。

练习 2：手算 KL 的方向
    后验 q = N(1, 0.5²)，先验 p = N(0, 1)。用 torch.distributions.kl_divergence 算 KL(q‖p) 和 KL(p‖q)。

练习 2 参考答案：
    KL(q‖p) = {kl_divergence(q, p).item():.3f}，KL(p‖q) = {kl_divergence(p, q).item():.3f}。两个方向不一样。
    序列 ELBO 里固定是 KL(后验 ‖ 先验)，期望对后验取。闭式解的推导是 CONCEPT.md 第 4.3 节，讲义 04 会手写它。

练习 3：哪个后验让 ELBO 最大
    沿用第二部分例二的设定（先验 N(0,1)，观测 N(z,1)，o = {obs}），把后验的均值固定为 0.5，
    标准差从 0.3 扫到 1.0，ELBO 在哪里最大？

练习 3 参考答案：
    在标准差 = √0.5 ≈ {best_std:.3f} 处最大，此时 ELBO = −1.516 恰好等于真实的对数似然。
    重建项 −0.919 − (0.25 + σ²)/2 和 KL 项 log(1/σ) + (σ² + 0.25)/2 − 0.5 对 σ 求导：
    −σ + 1/σ − σ = 0 → σ² = 0.5。这正是贝叶斯公式给出的后验方差。

思考题 1：固定先验的模型 ELBO 也不差，为什么说它"不是世界模型"？

思考题 1 参考答案：
    ELBO 只衡量"能不能把看到的序列解释好"。固定先验的模型靠后验把每一帧都编码进 z，重建得很好，
    只是每一步多付了一些 KL。可它的先验不依赖历史，没有观测时只能瞎猜。
    规划需要的恰恰是"没有观测时往前推演"，这只能靠学出来的先验。

思考题 2：后验网络的输入里有 z_{{t-1}}，为什么不只给 o_t？

思考题 2 参考答案：
    单帧里没有速度。只看 o_t 的后验只能编码位置，速度信息无处可来；
    给了 z_{{t-1}}，后验就是在"根据历史的预测"上做修正（01 节的修正步），z_t 才能是一个完整的状态。
    第 6 节的 RSSM 里，这个角色由 h_t 承担。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    torch.set_num_threads(4)
    quick = is_quick()
    if quick:
        print("（冒烟模式：样本数和训练步数都缩得很小，下面的数字没有参考价值）\n")
    n_iters, n_train, n_test, n_mc = (60, 200, 100, 2000) if quick else (3000, 3000, 500, 200_000)

    introduction()
    hmm_likelihood_bruteforce()
    gaussian_elbo_table(n_mc)
    models = train_both(n_iters, n_train)
    evaluate(models, n_test)
    surprise_experiment(models, n_test)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：03-deterministic-vs-stochastic.py（状态该用确定性还是随机的方式传递）")
    print("=" * 60)


if __name__ == "__main__":
    main()
