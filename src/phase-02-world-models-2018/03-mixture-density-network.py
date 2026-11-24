"""
混合密度网络（MDN）：预测一个分布而不是一个值
============================================

学习目标：
    1. 用一个能手算的例子看清：均方误差的最优解是条件均值，多峰时它落在两个峰中间
    2. 手写 MDN 的负对数似然（log-softmax + 高斯对数密度 + log-sum-exp），和手算结果逐位核对
    3. 用自动求导验证三组梯度都由"责任度"决定
    4. 在一个"一个 x 对应多个 y"的玩具问题上，对比 MSE 回归、单高斯、MDN
    5. 知道 MDN 的输出该怎么用：采样，而不是取加权平均

核心概念：
    - 混合密度网络（Mixture Density Network，MDN）：网络输出高斯混合的权重 π、均值 μ、标准差 σ
    - 负对数似然（negative log-likelihood，NLL）：真实值在预测分布下的概率密度，取对数再取负
    - log-sum-exp：在对数空间里对概率求和，避免下溢
    - 责任度（responsibility）：看到 y 之后，"它来自第 k 个成分"的后验概率

前置知识：
    - Phase 1 的 07-uncertainty-and-ensembles.py（高斯负对数似然）

对应概念文档：
    - CONCEPT.md 第 3 节「混合密度网络（MDN）：预测一个分布而不是一个值」

预计运行时间：约 20 秒（CPU）
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-02")
SEED = 0
NOISE_STD = 0.03                  # 玩具数据里 x 方向的噪声
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -5.0, 2.0
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


# ==================== 第一部分：平均的未来 ====================


def introduction():
    print("=" * 60)
    print("第一部分：平均的未来不是未来")
    print("=" * 60)
    print("""
M 要预测"下一个 z"。站在岔路口，下一刻可能左转也可能右转，未来有两个峰：

        概率密度
          │     ▲                         ▲
          │    ╱ ╲                       ╱ ╲
          │   ╱   ╲          ▲          ╱   ╲
          └─────────────────────────────────────▶ y
              左转          │           右转
                     均方误差的最优预测落在这里

只输出一个数、用均方误差训练的网络，学到的是条件均值——两个峰的中间，哪种可能都不是。
MDN 的办法是让网络输出一个能有多个峰的分布：K 个高斯按权重叠在一起。

                              ┌─▶ logits     ─ softmax ─▶ π_1 … π_K    [B, K]  是哪种情况
              ┌──────────┐    │
   x ───────▶ │  网络主体  │ ───┼─▶ μ_1 … μ_K                           [B, K]  这种情况下的值
              └──────────┘    │
                              └─▶ log σ_1 … log σ_K ─ exp ─▶ σ         [B, K]  这种情况下有多不确定

这一节不碰赛车，只在一维玩具问题上把 MDN 这个工具弄清楚。下一节再把它接到 LSTM 上。
    """)


def mse_gives_the_mean():
    """手算：均方误差明确地奖励"站在两个峰中间"。"""
    # 接口响应时间：缓存命中 10 毫秒（90%），没命中 200 毫秒（10%）
    values, probs = np.array([10.0, 200.0]), np.array([0.9, 0.1])
    best = float((values * probs).sum())
    # 岔路口：y = +1 和 y = -1 各一半
    fork = np.array([1.0, -1.0])
    loss_at = lambda c: float(((fork - c) ** 2).mean())
    print(f"""例一：接口响应时间。缓存命中 10 毫秒（90%），没命中 200 毫秒（10%）。
    均方误差下的最优预测 = 0.9 × 10 + 0.1 × 200 = {best:.0f} 毫秒——没有任何一次请求真的花了 {best:.0f} 毫秒。

例二：岔路口。y = +1（右转）和 y = -1（左转）各一半。
    预测 c =  0（直行撞墙）：均方误差 = 0.5 × (1-0)² + 0.5 × (-1-0)² = {loss_at(0.0):.1f}
    预测 c = +1（真实的峰）：均方误差 = 0.5 × 0 + 0.5 × (-2)²       = {loss_at(1.0):.1f}

选一个真实会出现的值，损失反而是站在中间的 {loss_at(1.0) / loss_at(0.0):.0f} 倍。这不是没训好，是这个损失的最优解本来如此。
    """)
    assert loss_at(0.0) < loss_at(1.0)


# ==================== 第二部分：MDN 的损失、采样、责任度 ====================


def mdn_log_components(log_pi, mu, log_sigma, y):
    """每个成分的 b_k = log π_k + log N(y; μ_k, σ_k²)。

    log_pi、mu、log_sigma：[..., K]；y：[..., 1]（和成分维广播）。返回 [..., K]。
    """
    log_sigma = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)          # 防止 σ 塌成 0 或爆炸
    log_normal = -HALF_LOG_2PI - log_sigma - 0.5 * ((y - mu) / log_sigma.exp()) ** 2
    return log_pi + log_normal


def mdn_nll(log_pi, mu, log_sigma, y):
    """MDN 的负对数似然：-log Σ_k π_k N(y; μ_k, σ_k²)。返回每个样本的损失 [...]。"""
    return -torch.logsumexp(mdn_log_components(log_pi, mu, log_sigma, y), dim=-1)


def responsibilities(log_pi, mu, log_sigma, y):
    """责任度 γ_k = π_k N_k / Σ_j π_j N_j，即对 b_k 做 softmax。返回 [..., K]。"""
    return torch.softmax(mdn_log_components(log_pi, mu, log_sigma, y), dim=-1)


def mdn_sample(log_pi, mu, log_sigma):
    """从 MDN 采样：先按 π 抽一个成分，再在这个成分的高斯里采一个值。返回 [B]。"""
    k = torch.distributions.Categorical(logits=log_pi).sample()        # [B]
    mu_k = mu.gather(-1, k.unsqueeze(-1)).squeeze(-1)
    sigma_k = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX).exp().gather(-1, k.unsqueeze(-1)).squeeze(-1)
    return mu_k + sigma_k * torch.randn_like(mu_k)


def check_by_hand():
    print("=" * 60)
    print("第二部分：MDN 的损失——和手算逐位核对")
    print("=" * 60)

    # CONCEPT.md 第 3.5 节例一：K = 2，π = (0.6, 0.4)，μ = (0, 2)，σ = (1, 0.5)，y = 1.5
    pi, mu, sigma, y = [0.6, 0.4], [0.0, 2.0], [1.0, 0.5], 1.5
    logits = torch.tensor([pi]).log().requires_grad_()                 # logits 取 log π，softmax 后正好是 π
    mu_t = torch.tensor([mu], requires_grad=True)
    log_sigma_t = torch.tensor([sigma]).log().requires_grad_()
    y_t = torch.tensor([[y]])
    log_pi = torch.log_softmax(logits, dim=-1)
    b = mdn_log_components(log_pi, mu_t, log_sigma_t, y_t)
    loss = mdn_nll(log_pi, mu_t, log_sigma_t, y_t)
    gamma = responsibilities(log_pi, mu_t, log_sigma_t, y_t)[0].detach()
    loss.sum().backward()

    density = [p / (math.sqrt(2 * math.pi) * s) * math.exp(-0.5 * ((y - m) / s) ** 2) for p, m, s in zip(pi, mu, sigma)]
    print(f"""
手算例子：两个成分，π = {tuple(pi)}，μ = {tuple(mu)}，σ = {tuple(sigma)}，真实值 y = {y}

    b_1 = log π_1 + log N_1 = {b[0, 0].item():.4f}
    b_2 = log π_2 + log N_2 = {b[0, 1].item():.4f}
    损失 = -logsumexp(b)    = {loss.item():.4f}

    不取对数验算：π_1 N_1 + π_2 N_2 = {density[0]:.4f} + {density[1]:.4f} = {sum(density):.4f}，-log = {-math.log(sum(density)):.4f}
    """)
    assert abs(loss.item() - 1.3046) < 1e-3 and abs(loss.item() + math.log(sum(density))) < 1e-5

    # 三组梯度的解析式（CONCEPT.md 第 3.6 节），和自动求导对比
    pi_t, mu_v, sigma_v = torch.tensor(pi), torch.tensor(mu), torch.tensor(sigma)
    grad_mu = gamma * (mu_v - y) / sigma_v**2
    grad_log_sigma = gamma * (1 - (y - mu_v) ** 2 / sigma_v**2)
    grad_logits = pi_t - gamma
    print(f"责任度：γ = ({gamma[0]:.3f}, {gamma[1]:.3f})。看到 y = {y} 之前成分 2 的权重是 {pi[1]}，"
          f"看到之后升到 {gamma[1]:.3f}（{y} 离它的中心 {mu[1]:.0f} 更近）。\n")
    print("    参数        解析式                              解析梯度    自动求导")
    rows = [("μ_1", "γ_1 (μ_1 - y) / σ_1²", grad_mu[0], mu_t.grad[0, 0]),
            ("μ_2", "γ_2 (μ_2 - y) / σ_2²", grad_mu[1], mu_t.grad[0, 1]),
            ("log σ_1", "γ_1 (1 - (y - μ_1)² / σ_1²)", grad_log_sigma[0], log_sigma_t.grad[0, 0]),
            ("log σ_2", "γ_2 (1 - (y - μ_2)² / σ_2²)", grad_log_sigma[1], log_sigma_t.grad[0, 1]),
            ("logit_1", "π_1 - γ_1", grad_logits[0], logits.grad[0, 0]),
            ("logit_2", "π_2 - γ_2", grad_logits[1], logits.grad[0, 1])]
    for name, formula, analytic, auto in rows:
        print(f"    {name:<10}  {formula:<32}  {analytic.item():+8.3f}   {auto.item():+8.3f}")
        assert abs(analytic.item() - auto.item()) < 1e-4
    print("""
读这张表：成分 2 的责任度大，μ_2 的梯度也大——每个样本主要更新离它最近的成分。
log σ_2 的梯度是 0：误差 0.5 正好等于 σ_2，不用调。两个 logit 的梯度一正一负、大小相等。
    """)

    # 为什么必须用 log-sum-exp
    b_small = torch.tensor([-800.0, -805.0])
    naive = torch.log(torch.exp(b_small).sum())
    stable = torch.logsumexp(b_small, dim=0)
    print(f"""数值稳定性。两个成分的 b = (-800, -805)（密度极小，序列长了、维度多了很常见）：
    直接算 log(exp(b_1) + exp(b_2)) = {naive.item()}     ← exp(-800) 在 float32 里是 0
    torch.logsumexp                 = {stable.item():.4f}
    """)
    assert torch.isinf(naive) and abs(stable.item() + 799.9933) < 1e-3

    # K = 1 退化成高斯负对数似然
    k1 = mdn_nll(torch.zeros(1, 1), torch.tensor([[0.3]]), torch.tensor([[math.log(0.7)]]), torch.tensor([[1.0]]))
    gauss = HALF_LOG_2PI + math.log(0.7) + (1.0 - 0.3) ** 2 / (2 * 0.7**2)
    print(f"K = 1 时退化成 Phase 1 的高斯负对数似然：MDN 给出 {k1.item():.4f}，高斯公式给出 {gauss:.4f}。\n")
    assert abs(k1.item() - gauss) < 1e-5


# ==================== 第三部分：玩具问题 ====================


def forward_curve(y):
    """单值的正向函数 x = g(y)。把输入输出对调，就得到一个 x 对应多个 y 的反问题。"""
    return y + 0.3 * np.sin(2 * np.pi * y)


def make_inverse_dataset(n, rng):
    """Bishop 1994 的经典玩具问题。返回 x、y，形状 [n, 1]，float32。"""
    y = rng.uniform(0, 1, size=n)
    x = forward_curve(y) + NOISE_STD * rng.normal(size=n)
    return (torch.from_numpy(x).float().unsqueeze(1), torch.from_numpy(y).float().unsqueeze(1))


def count_branches(x, grid=2001):
    """在给定的 x 上，y = g⁻¹(x) 有几个解（数 g(y) - x 变号的次数）。"""
    ys = np.linspace(0, 1, grid)
    diff = forward_curve(ys) - x
    return int((np.sign(diff[:-1]) != np.sign(diff[1:])).sum())


class PointRegressor(nn.Module):
    """只输出一个数的普通回归网络，用均方误差训练。"""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)                                             # [B, 1]


class MixtureDensityNetwork(nn.Module):
    """输出 K 个成分的 log π、μ、log σ。"""

    def __init__(self, n_components, hidden=64):
        super().__init__()
        self.n_components = n_components
        self.body = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 3 * n_components)
        with torch.no_grad():
            # μ 的初始偏置铺开在数据范围 [0, 1] 上：成分一开始就各管一段，不会全挤在一起
            self.head.bias[n_components:2 * n_components] = torch.linspace(0, 1, n_components + 2)[1:-1]
            self.head.bias[2 * n_components:] = math.log(0.2)

    def forward(self, x):
        """x [B, 1] → log_pi、mu、log_sigma，形状都是 [B, K]。"""
        logits, mu, log_sigma = self.head(self.body(x)).chunk(3, dim=-1)
        return torch.log_softmax(logits, dim=-1), mu, log_sigma


def train(model, loss_fn, x, y, steps, lr=5e-3):
    """全批量训练。玩具数据只有几千个点，不需要分批。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = loss_fn(model, x, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    return loss.item()


def mse_loss(model, x, y):
    return ((model(x) - y) ** 2).mean()


def mdn_loss(model, x, y):
    return mdn_nll(*model(x), y).mean()


def on_branch_rate(x, y_pred):
    """预测点 (x, ŷ) 有多大比例落在真实的曲线上：|x - g(ŷ)| 小于 3 倍噪声标准差就算。"""
    x, y_pred = x.squeeze(-1).numpy(), y_pred.squeeze(-1).numpy()
    return float((np.abs(x - forward_curve(y_pred)) < 3 * NOISE_STD).mean())


def toy_experiment(steps):
    print("=" * 60)
    print("第三部分：一个 x 对应多个 y 的玩具问题")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    x_train, y_train = make_inverse_dataset(2000, rng)
    x_test, y_test = make_inverse_dataset(2000, rng)
    branches = {x: count_branches(x) for x in (0.1, 0.5, 0.9)}
    print(f"""
数据：正向函数 x = y + 0.3 sin(2πy) 是单值的；把它反过来，从 x 预测 y。
中间一段的 x 对应 3 个 y：x = 0.1 有 {branches[0.1]} 个解，x = 0.5 有 {branches[0.5]} 个解，x = 0.9 有 {branches[0.9]} 个解。
训练集、测试集各 {len(x_train)} 个点。三个模型的网络主体相同（两层 64 单元），只有输出层和损失不同。
    """)

    start = time.time()
    torch.manual_seed(SEED)
    regressor = PointRegressor()
    train(regressor, mse_loss, x_train, y_train, steps)
    torch.manual_seed(SEED)
    gaussian = MixtureDensityNetwork(1)
    train(gaussian, mdn_loss, x_train, y_train, steps)
    torch.manual_seed(SEED)
    mdn = MixtureDensityNetwork(5)
    train(mdn, mdn_loss, x_train, y_train, steps)
    print(f"三个模型各训练 {steps} 步，共用时 {time.time() - start:.1f} 秒。\n")

    with torch.no_grad():
        # MSE 网络没有方差。给它配一个在训练集上统计出来的常数 σ，才能算负对数似然
        sigma_const = (regressor(x_train) - y_train).pow(2).mean().sqrt().item()
        pred = regressor(x_test)
        nll_mse = (HALF_LOG_2PI + math.log(sigma_const) + (y_test - pred) ** 2 / (2 * sigma_const**2)).mean().item()
        nll_gauss = mdn_loss(gaussian, x_test, y_test).item()
        nll_mdn = mdn_loss(mdn, x_test, y_test).item()
        torch.manual_seed(SEED)
        sample_gauss = mdn_sample(*gaussian(x_test)).unsqueeze(-1)
        sample_mdn = mdn_sample(*mdn(x_test)).unsqueeze(-1)
        rate_mse, rate_gauss, rate_mdn = (on_branch_rate(x_test, p) for p in (pred, sample_gauss, sample_mdn))
        rate_data = on_branch_rate(x_test, y_test)

    print(f"""                              测试集负对数似然     输出落在真实曲线上的比例
    MSE 回归（配常数 σ = {sigma_const:.3f}）     {nll_mse:+8.3f}              {rate_mse * 100:5.1f}%   （点预测）
    单高斯（K = 1，σ 可学）            {nll_gauss:+8.3f}              {rate_gauss * 100:5.1f}%   （采样）
    MDN（K = 5）                      {nll_mdn:+8.3f}              {rate_mdn * 100:5.1f}%   （采样）
    参照：测试数据本身                                           {rate_data * 100:5.1f}%
    """)
    if nll_mdn < nll_gauss < nll_mse and rate_mdn > max(rate_mse, rate_gauss) + 0.2:
        print(f"""三级台阶：MSE → 单高斯 → MDN，负对数似然一级比一级低。
MSE 的预测曲线在多值区域穿过几条分支的中间，只有 {rate_mse * 100:.0f}% 的预测落在真实曲线上；
单高斯学会了"这一段我不确定"（σ 变大），但只有一个峰，采样仍然大多落在分支之间；
MDN 的采样有 {rate_mdn * 100:.0f}% 落在曲线上，接近数据本身的 {rate_data * 100:.0f}%。
MDN 的负对数似然是负数：概率密度可以大于 1，分布越窄、预测越准，损失越负。
        """)
    else:
        print("这次运行里三个模型的差距没有预期的明显（冒烟模式训练步数太少时会这样）。\n")

    plot_fits(x_train, y_train, x_test, regressor, sample_gauss, sample_mdn)
    return mdn, regressor, (x_train, y_train, x_test, y_test)


def plot_fits(x_train, y_train, x_test, regressor, sample_gauss, sample_mdn):
    xs = torch.linspace(-0.05, 1.05, 300).unsqueeze(1)
    with torch.no_grad():
        curve = regressor(xs)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax in axes:
        ax.scatter(x_train[:800], y_train[:800], s=5, color="lightgray", label="训练数据")
        ax.set_xlabel("x")
        ax.set_ylim(-0.25, 1.25)
    axes[0].plot(xs, curve, color="tab:red", lw=2, label="MSE 回归的预测")
    axes[0].set_title("MSE 回归：穿过分支的中间")
    axes[1].scatter(x_test[:800], sample_gauss[:800], s=5, color="tab:orange", label="单高斯的采样")
    axes[1].set_title("单高斯（K = 1）：知道不确定，但只有一个峰")
    axes[2].scatter(x_test[:800], sample_mdn[:800], s=5, color="tab:blue", label="MDN 的采样")
    axes[2].set_title("MDN（K = 5）：采样落在各条分支上")
    axes[0].set_ylabel("y")
    for ax in axes:
        ax.legend(loc="upper left", fontsize=8)
    save_figure(fig, OUT / "03-mse-vs-mdn.png")


# ==================== 第四部分：MDN 内部是怎么分工的 ====================


@torch.no_grad()
def anatomy(mdn, regressor, data):
    print("=" * 60)
    print("第四部分：MDN 内部——成分怎么分工，输出该怎么用")
    print("=" * 60)
    x_train, y_train, x_test, _ = data
    k_total = mdn.n_components

    # 在 x = 0.5 这一刀上看：三条分支各取一个点，谁负责
    x0 = 0.5
    ys = np.linspace(0, 1, 2001)
    diff = forward_curve(ys) - x0
    roots = ys[:-1][np.sign(diff[:-1]) != np.sign(diff[1:])]
    log_pi, mu, log_sigma = mdn(torch.tensor([[x0]]))
    pi, sigma = log_pi.exp()[0], log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX).exp()[0]
    print(f"\n在 x = {x0} 这一刀上，MDN 的 {k_total} 个成分：")
    print("    成分    权重 π     均值 μ     标准差 σ")
    for k in range(k_total):
        print(f"    {k + 1:>3}    {pi[k].item():6.3f}    {mu[0, k].item():+6.3f}    {sigma[k].item():6.3f}")
    print(f"\n真实曲线在 x = {x0} 处的 {len(roots)} 个解：{', '.join(f'{r:.3f}' for r in roots)}。各个解由哪个成分负责（责任度 γ）：")
    owners = []
    for root in roots:
        gamma = responsibilities(log_pi, mu, log_sigma, torch.tensor([[float(root)]]))[0]
        owners.append(int(gamma.argmax()) + 1)
        print(f"    y = {root:.3f}：γ = ({', '.join(f'{g:.2f}' for g in gamma.tolist())})  → 主要是成分 {owners[-1]}")
    active = int((pi > 0.02).sum())
    print(f"""
在这一刀上权重超过 0.02 的成分有 {active} 个。"是哪种情况"没有标签，
分工是训练中自己形成的：每个样本主要更新离它最近的成分（第二部分的梯度表）。
    """)

    # 输出该怎么用：三种用法
    log_pi_t, mu_t, log_sigma_t = mdn(x_test)
    weighted_mean = (log_pi_t.exp() * mu_t).sum(-1, keepdim=True)
    top_component = mu_t.gather(-1, log_pi_t.argmax(-1, keepdim=True))
    torch.manual_seed(SEED)
    sampled = mdn_sample(log_pi_t, mu_t, log_sigma_t).unsqueeze(-1)
    gap = (weighted_mean - regressor(x_test)).abs().mean().item()
    # 多值区域里，各条分支被"取到"的情况：按 y 分成下、中、上三段
    multi = ((x_test > 0.35) & (x_test < 0.65)).squeeze(-1)
    share = lambda y: [float(((y[multi] >= lo) & (y[multi] < hi)).float().mean()) for lo, hi in ((-9, 0.3), (0.3, 0.7), (0.7, 9))]
    share_top, share_sample = share(top_component), share(sampled)
    print(f"""MDN 的输出有三种用法，在测试集上比较：

    用法                         落在真实曲线上的比例    多值区域里取到 下 / 中 / 上 三条分支的比例
    按 π 加权平均各成分的 μ          {on_branch_rate(x_test, weighted_mean) * 100:5.1f}%
    只取 π 最大的成分的 μ            {on_branch_rate(x_test, top_component) * 100:5.1f}%            {share_top[0] * 100:3.0f}% / {share_top[1] * 100:3.0f}% / {share_top[2] * 100:3.0f}%
    采样                           {on_branch_rate(x_test, sampled) * 100:5.1f}%            {share_sample[0] * 100:3.0f}% / {share_sample[1] * 100:3.0f}% / {share_sample[2] * 100:3.0f}%

加权平均和 MSE 回归的预测平均只差 {gap:.3f}：它就是又算了一次条件均值，白费了混合。
只取权重最大的成分，每个点都落在曲线上，但多值区域里它几乎总是挑同一条分支，其他可能被丢掉了。
采样才是 MDN 的正确用法：先抽"是哪种情况"，再在这种情况里采一个值。
    """)

    xs = torch.linspace(0, 1, 300).unsqueeze(1)
    lp, m, ls = mdn(xs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].scatter(x_train[:800], y_train[:800], s=5, color="lightgray")
    colors = plt.cm.tab10(np.arange(k_total))
    for k in range(k_total):
        # 点的大小和透明度按权重画：权重接近 0 的成分几乎看不见
        w = lp.exp()[:, k].numpy()
        axes[0].scatter(xs.squeeze(1), m[:, k], s=1 + 30 * w, color=colors[k], alpha=0.8, label=f"成分 {k + 1}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_ylim(-0.25, 1.25)
    axes[0].set_title("各成分的均值 μ_k(x)，点越大权重 π_k(x) 越大")
    axes[0].legend(fontsize=8, loc="upper left")

    grid = torch.linspace(-0.1, 1.1, 400)
    b = mdn_log_components(log_pi, mu, log_sigma, grid.view(-1, 1))    # [400, K]
    near = (x_train.squeeze(1) - x0).abs() < 0.03
    axes[1].hist(y_train[near].squeeze(1).numpy(), bins=40, density=True, color="lightgray", label=f"|x - {x0}| < 0.03 的训练数据")
    axes[1].plot(grid, torch.logsumexp(b, dim=-1).exp(), color="black", lw=2, label="MDN 预测的密度 p(y | x)")
    for k in range(k_total):
        if pi[k] > 0.02:
            axes[1].plot(grid, b[:, k].exp(), color=colors[k], ls="--", label=f"成分 {k + 1}（π = {pi[k].item():.2f}）")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("概率密度")
    axes[1].set_title(f"x = {x0} 这一刀上的预测分布")
    axes[1].legend(fontsize=8)
    save_figure(fig, OUT / "03-mdn-anatomy.png")


# ==================== 第五部分：K 取多少 ====================


def component_sweep(data, steps):
    print("=" * 60)
    print("第五部分：成分个数 K 取多少")
    print("=" * 60)
    x_train, y_train, x_test, y_test = data
    print("\n    K     测试集负对数似然     用到的成分数（在某个 x 上权重超过 0.05）")
    results = {}
    for k in (1, 2, 3, 5, 10):
        torch.manual_seed(SEED)
        model = MixtureDensityNetwork(k)
        train(model, mdn_loss, x_train, y_train, steps)
        with torch.no_grad():
            nll = mdn_loss(model, x_test, y_test).item()
            used = int((model(x_test)[0].exp().max(dim=0).values > 0.05).sum())
        results[k] = (nll, used)
        print(f"   {k:>2}        {nll:+8.3f}                {used}")
    print()
    if results[3][0] < results[2][0] < results[1][0] and abs(results[10][0] - results[3][0]) < 0.25 * abs(results[3][0] - results[1][0]):
        print(f"""这个问题最多有 3 条分支。K 从 1 加到 3，负对数似然从 {results[1][0]:+.2f} 降到 {results[3][0]:+.2f}；
再加到 5、10，变化就小了（{results[5][0]:+.2f}、{results[10][0]:+.2f}）。
K 取少了才有问题；取多了，多出来的成分要么权重趋近 0，要么几个成分合起来拟合同一条分支。
        """)
    else:
        print("这次运行里 K 的趋势不典型（冒烟模式训练步数太少时会这样），完整运行时再看。\n")


# ==================== 第六部分：练习 ====================


def exercises():
    print("=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把 mdn_nll 里的 torch.logsumexp 换成 torch.log(torch.exp(b).sum(-1))，重新训练 K = 5 的 MDN。
        训练还能进行吗？在哪一步出问题？

练习 2：把 LOG_SIGMA_MIN 改成 -20，把 NOISE_STD 改成 0（数据没有噪声），训练 5000 步，观察损失。
        再把 LOG_SIGMA_MIN 改回 -5。损失的下限分别是多少？

练习 3：把 MixtureDensityNetwork 里"μ 的初始偏置铺开"那两行删掉，重新运行第五部分。
        用到的成分数怎么变？对应 CONCEPT.md 第 3.8 节的哪一种失败模式？

思考题 1：单高斯（K = 1）的负对数似然比"MSE 配常数 σ"低，它比 MSE 多知道了什么？
          它仍然不知道什么？

思考题 2：MDN 给出的 σ 很大，能说明"模型在这里没见过数据"吗？（回忆 Phase 1 的两类不确定性）

参考答案：
    练习 1：b 大约在 -1 到 3 之间时不会下溢，玩具问题上多半能训练。但只要有一个样本离所有成分都很远
            （比如 σ 被学得很小之后的离群点），exp(b) 全为 0，log 0 = -inf，梯度是 NaN，训练随即崩溃。
            维度多、序列长的时候（下一节）这几乎必然发生。
    练习 2：没有噪声时，把 σ 缩得越小损失越低，没有下限；下限由 LOG_SIGMA_MIN 决定：
            损失的最小值约为 0.5 log(2π) + LOG_SIGMA_MIN + log(分支数)。-20 时损失会掉到 -18 左右，
            这种"损失突然变成很大的负数"就是方差塌缩。-5 时停在 -3 左右。
    练习 3：所有成分初始时 μ 几乎相同，责任度平分，大家收到几乎一样的梯度，很难分开。
            常见的结果是用到的成分变少、负对数似然变差，即"退化成单峰"或"成分闲置"。
    思考题 1：它多知道了"哪一段 x 上 y 的散布大"（σ 随 x 变化），所以在单值区域可以把 σ 收窄。
              它仍然不知道散布的形状：多值区域里它只能用一个很宽的峰盖住三条分支。
    思考题 2：不能。MDN 的 σ 和 π 表达的是偶然不确定性（数据本身的多值和噪声）。
              在训练数据范围之外，π、μ、σ 都是外推出来的，可大可小。认知不确定性要靠集成（Phase 1 第 8 节）。
    """)


def main():
    seed_everything(SEED)
    steps = 300 if is_quick() else 3000

    introduction()
    mse_gives_the_mean()
    check_by_hand()
    mdn, regressor, data = toy_experiment(steps)
    anatomy(mdn, regressor, data)
    component_sweep(data, steps)
    exercises()

    print("=" * 60)
    print("本节小结")
    print("=" * 60)
    print("""
    1. 均方误差的最优解是条件均值。未来多峰时，它落在峰与峰之间，是一个不存在的未来。
    2. MDN 输出高斯混合的参数：π（是哪种情况）、μ（这种情况下的值）、σ（这种情况下有多不确定）。
    3. 损失是负对数似然，实现成 log-softmax + 高斯对数密度 + logsumexp，全程在对数空间里算。
    4. 三组梯度都带一个责任度 γ：每个样本主要更新离它最近的成分，成分因此自己分工。
    5. MDN 的输出要采样着用。加权平均等于退回 MSE；只取最大的成分会丢掉其他可能。

下一节：把这个输出层接到 LSTM 上，让它预测"下一个 z 的分布"——这就是 M。
    """)
    if is_quick():
        print("（当前是冒烟模式 WM_QUICK=1：训练步数很少，上面的数字没有参考价值。）")


if __name__ == "__main__":
    main()
