"""
不确定性与概率集成
==================

学习目标：
    1. 区分两种不确定性：偶然不确定性（世界本身的随机）和认知不确定性（模型没见过）
    2. 推导并实现高斯负对数似然（NLL），让网络在输出预测的同时输出方差
    3. 实现向量化的概率集成：一次前向计算同时跑完所有成员
    4. 用实验确认：单个网络输出的方差发现不了"没见过"，成员之间的分歧可以
    5. 看到"模型被钻空子"：规划器专挑模型高估的地方走，想象的回报系统性偏高

核心概念：
    - 概率网络：输出均值 μ 和对数方差 log σ²，用 NLL 训练
    - 高斯 NLL：(y-μ)²/(2σ²) + ½·log σ²，两项相互制约，最优的 σ² 等于平方误差
    - 集成（ensemble）：B 个结构相同、初始化不同、训练数据不同（自助采样）的网络
    - 分歧（disagreement）：成员预测之间的标准差，用来度量认知不确定性
    - 模型被钻空子（model exploitation）：对一个带误差的估计做最大化，结果系统性高估

前置知识：
    - 03-learning-dynamics-mlp.py（动力学模型）、06-cem-and-mppi.py（CEM）

对应概念文档：
    - CONCEPT.md 第 8 节「模型的不确定性与概率集成」

预计运行时间：约 6 分钟
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-01")
SEED = 0
MAX_TORQUE = 2.0
MAX_SPEED = 8.0
DT = 0.05
HORIZON = 30
N_MODELS = 5
TRAIN_DEVICE = get_device()
PLAN_DEVICE = torch.device("cpu")  # 规划的批量小，CPU 就够（见 06 课）


# ==================== 第一部分：两种不确定性 ====================


def introduction():
    """为什么模型要会说"我不确定"。"""
    print("=" * 60)
    print("第一部分：模型要会说「我不确定」")
    print("=" * 60)

    print("""
到目前为止，模型对任何输入都给出一个确定的预测，不管这个输入它见没见过。
这很危险：规划器是一个优化器，专门去找分数高的地方。模型在没见过的区域
瞎猜出一个高分，规划器会毫不犹豫地冲过去。

"不确定"有两个来源，处理方式不同：

    ┌──────────────┬────────────────────────┬──────────────────────────┐
    │              │ 偶然不确定性（aleatoric） │ 认知不确定性（epistemic）   │
    ├──────────────┼────────────────────────┼──────────────────────────┤
    │ 来源          │ 世界本身的随机性          │ 模型的知识不足              │
    │ 例子          │ 传感器噪声、地面打滑       │ 没见过这个区域的数据         │
    │ 多采数据能减少 │ 不能                    │ 能                        │
    │ 怎么建模      │ 让网络输出均值 + 方差      │ 训练多个模型，看是否一致     │
    └──────────────┴────────────────────────┴──────────────────────────┘

一枚验过的均匀硬币，抛之前不知道正反：偶然不确定性，再抛一万次也消除不了。
一枚没见过的硬币，不知道它均不均匀：认知不确定性，抛几次看看就知道了。

这一节先在一维的回归问题上把两种不确定性各自的建模办法看清楚，再搬到摆上。
    """)
    print(f"  使用设备：训练 {TRAIN_DEVICE}，规划 {PLAN_DEVICE}")


# ==================== 向量化的概率集成 ====================


class EnsembleLinear(nn.Module):
    """B 个线性层并排放。权重形状 [B, in, out]，输入 [B, N, in] → 输出 [B, N, out]。

    等价于 B 个独立的 nn.Linear，但一次矩阵乘法全部算完，不用写循环。
    members 用来只算其中几个成员（规划时用单个模型就靠它）。
    """

    def __init__(self, n_models, in_dim, out_dim):
        super().__init__()
        bound = 1.0 / np.sqrt(in_dim)  # 和 nn.Linear 默认初始化的范围一致；每个成员的初值各不相同
        self.weight = nn.Parameter(torch.empty(n_models, in_dim, out_dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(n_models, 1, out_dim).uniform_(-bound, bound))

    def forward(self, x, members=None):
        weight, bias = (self.weight, self.bias) if members is None else (self.weight[members], self.bias[members])
        return torch.baddbmm(bias, x, weight)


class ProbabilisticEnsemble(nn.Module):
    """概率集成：每个成员输出标准化空间里的均值和对数方差。

    in_dim / out_dim 让它既能用于一维回归的例子，也能用于摆的动力学。
    """

    def __init__(self, in_dim, out_dim, n_models=N_MODELS, hidden=200):
        super().__init__()
        self.n_models, self.out_dim = n_models, out_dim
        self.layers = nn.ModuleList([
            EnsembleLinear(n_models, in_dim, hidden),
            EnsembleLinear(n_models, hidden, hidden),
            EnsembleLinear(n_models, hidden, 2 * out_dim),  # 前一半是均值，后一半是对数方差
        ])
        # 对数方差的上下界，可学习（PETS 的做法）
        self.max_logvar = nn.Parameter(torch.full((out_dim,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((out_dim,), -10.0))
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(out_dim))
        self.register_buffer("y_std", torch.ones(out_dim))

    def fit_normalizer(self, x, y):
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp_min(1e-6))
        self.y_mean.copy_(y.mean(0))
        self.y_std.copy_(y.std(0).clamp_min(1e-6))

    def forward(self, x, members=None):
        """x [B, N, in]（物理单位）→ 标准化空间的均值 [B, N, out] 和对数方差 [B, N, out]。"""
        h = (x - self.x_mean) / self.x_std
        h = F.silu(self.layers[0](h, members))
        h = F.silu(self.layers[1](h, members))
        out = self.layers[2](h, members)
        mean, logvar = out[..., :self.out_dim], out[..., self.out_dim:]
        # 软夹紧：用 softplus 而不是硬截断，梯度在边界附近不会断
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def nll_loss(self, x, y):
        """高斯负对数似然。x [B, N, in]，y [B, N, out]，都是物理单位。"""
        mean, logvar = self(x)
        target = (y - self.y_mean) / self.y_std
        nll = 0.5 * ((mean - target) ** 2 * torch.exp(-logvar) + logvar)  # (y-μ)²/(2σ²) + ½·log σ²
        # 末项是对上下界的轻微正则，防止它们越撑越宽
        return nll.mean() + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())

    @torch.no_grad()
    def predict(self, x, members=None):
        """返回物理单位下的均值 [B, N, out] 和标准差 [B, N, out]。"""
        mean, logvar = self(x, members)
        return mean * self.y_std + self.y_mean, torch.exp(0.5 * logvar) * self.y_std


def train_ensemble(model, x, y, n_steps, batch_size=256, lr=1e-3, seed=SEED):
    """训练集成。每个成员看到的是自助采样（bootstrap）出来的数据，各不相同。"""
    model.to(TRAIN_DEVICE)
    x, y = x.to(TRAIN_DEVICE), y.to(TRAIN_DEVICE)
    model.fit_normalizer(x, y)
    n = x.shape[0]
    generator = torch.Generator().manual_seed(seed)
    # 自助采样：每个成员从 n 条数据里有放回地抽 n 条，作为它自己的训练集
    bootstrap = torch.randint(0, n, (model.n_models, n), generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    for _ in range(n_steps):
        pick = torch.randint(0, n, (model.n_models, batch_size), generator=generator)
        idx = torch.gather(bootstrap, 1, pick).to(TRAIN_DEVICE)  # [B, batch]，每个成员一批
        loss = model.nll_loss(x[idx], y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model.to(PLAN_DEVICE)


# ==================== 第二部分：一维回归上看两种不确定性 ====================


def toy_regression(n_steps):
    """有噪声、有空白的一维数据：方差学到了噪声，分歧发现了空白。"""
    print("\n" + "=" * 60)
    print("第二部分：在一维回归上看两种不确定性")
    print("=" * 60)

    print("""
【高斯 NLL 的两项】让网络除了均值 μ，再输出一个标准差 σ，损失换成负对数似然：

    L = (y - μ)² / (2σ²) + log σ

第一项：误差被 σ 缩放，宣称"我不确定"（σ 大）能减轻对误差的惩罚。
第二项：惩罚 σ 本身，不能对什么都宣称不确定。
假设某个样本的预测误差 y - μ = 0.2，看不同的 σ 对应的损失（CONCEPT.md 第 8.3 节的例子）：
    """)
    print(f"  {'σ':<8}{'第一项':>10}{'第二项':>10}{'损失':>10}")
    for sigma in (0.05, 0.1, 0.2, 0.5):
        first, second = 0.2**2 / (2 * sigma**2), np.log(sigma)
        print(f"  {sigma:<8}{first:>12.2f}{second:>12.2f}{first + second:>12.2f}")
    print("""
损失在 σ = 0.2 处最小，正好等于误差的大小：两项的拉锯逼着网络如实报告自己的误差。
而且报小了（过度自信，5.00）受到的惩罚远大于报大了（过度保守，-0.61）。

【实验】数据来自 y = sin(x) + 噪声：
    - x 只在 [-4, -2] 和 [2, 4] 两段里有，中间 (-2, 2) 和两头之外都没有数据
    - 左边一段的噪声标准差固定是 0.05，右边一段从 0.05 逐渐涨到 0.25
训练一个 5 个成员的概率集成。
    """)

    rng = np.random.default_rng(SEED)
    n = 2000
    x = np.concatenate([rng.uniform(-4, -2, n // 2), rng.uniform(2, 4, n // 2)])

    def noise_std(x):
        return np.where(x > 0, 0.05 + 0.1 * (x - 2), 0.05)

    y = np.sin(x) + rng.standard_normal(n) * noise_std(x)
    x_t = torch.as_tensor(x, dtype=torch.float32)[:, None]
    y_t = torch.as_tensor(y, dtype=torch.float32)[:, None]

    torch.manual_seed(SEED)
    model = train_ensemble(ProbabilisticEnsemble(1, 1, hidden=64), x_t, y_t, n_steps)

    grid = torch.linspace(-6, 6, 241)[None, :, None].expand(N_MODELS, -1, -1)
    mean, std = model.predict(grid)  # [B, 241, 1]
    mean, std = mean[..., 0].numpy(), std[..., 0].numpy()
    grid_x = grid[0, :, 0].numpy()
    disagreement = mean.std(axis=0)  # 成员之间的分歧 [241]

    print(f"  {'x':<7}{'有数据吗':<8}{'sin(x)':>8}{'成员 0 的 μ':>12}{'成员 0 的 σ':>12}{'真实噪声':>10}{'5 个成员的分歧':>16}")
    rows = {}
    for xv in (-3.0, 3.0, 3.8, 0.0, 6.0):
        i = int(np.argmin(np.abs(grid_x - xv)))
        has_data = 2 <= abs(xv) <= 4
        true_noise = f"{float(noise_std(np.array(xv))):.3f}" if has_data else "—"
        rows[xv] = (mean[0, i], std[0, i], disagreement[i])
        print(f"  {xv:<+7.1f}{'有' if has_data else '没有':<10}{np.sin(xv):>8.3f}{mean[0, i]:>12.3f}{std[0, i]:>13.3f}"
              f"{true_noise:>12}{disagreement[i]:>16.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    axes[0].scatter(x, y, s=3, alpha=0.3, color="gray", label="训练数据")
    axes[0].plot(grid_x, np.sin(grid_x), "k--", linewidth=1, label="sin(x)")
    axes[0].plot(grid_x, mean[0], color="C0", linewidth=2, label="成员 0 的均值")
    axes[0].fill_between(grid_x, mean[0] - 2 * std[0], mean[0] + 2 * std[0], color="C0", alpha=0.2, label="成员 0 的 ±2σ")
    axes[0].set_ylim(-3, 3)
    axes[0].set_title("单个概率网络：σ 学到了噪声的大小")
    axes[0].set_xlabel("x")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].scatter(x, y, s=3, alpha=0.3, color="gray")
    axes[1].plot(grid_x, np.sin(grid_x), "k--", linewidth=1)
    for b in range(N_MODELS):
        axes[1].plot(grid_x, mean[b], linewidth=1.5, label=f"成员 {b}")
    axes[1].set_ylim(-3, 3)
    axes[1].set_title("集成：有数据的地方一致，没数据的地方各说各的")
    axes[1].set_xlabel("x")
    axes[1].legend(fontsize=8, loc="lower left")
    save_figure(fig, OUT / "07-toy-uncertainty.png")

    in_data = rows[-3.0][2]
    print(f"""
怎么读这张表：

    1. 有数据的地方，成员 0 输出的 σ 跟着真实噪声走（x = -3 处 {rows[-3.0][1]:.3f}，x = 3.8 处 {rows[3.8][1]:.3f}）。
       没有人告诉它噪声是多少，这是 NLL 损失的两项拉锯出来的。这是偶然不确定性。
    2. 没数据的地方，成员 0 的预测已经错了：x = 0 处它预测 {rows[0.0][0]:+.2f}（应该是 0），
       x = 6 处预测 {rows[6.0][0]:+.2f}（应该是 {np.sin(6.0):+.2f}）。
       但它在这两处报的 σ 是 {rows[0.0][1]:.2f} 和 {rows[6.0][1]:.2f}，只是把有数据的地方的趋势延伸了过去，
       和真实的误差大小没有关系。单个网络的方差发现不了"我没见过这里"。
    3. 成员之间的分歧：有数据的地方约 {in_data:.3f}，x = 0 处 {rows[0.0][2]:.3f}，x = 6 处 {rows[6.0][2]:.3f}。
       没数据的地方没有东西约束它们，各自按自己的初始化去外推，就各说各的。
       这是认知不确定性，要靠集成才看得见。
    """)
    return rows


# ==================== 摆的数据、物理、奖励 ====================


def wrap_angle(theta):
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def true_step(states, actions):
    """真实物理方程。支持任意的前置维度：states [..., 2]，actions [..., 1]。"""
    theta, theta_dot = states[..., 0], states[..., 1]
    u = actions[..., 0].clamp(-MAX_TORQUE, MAX_TORQUE)
    theta_dot_new = (theta_dot + (15.0 * torch.sin(theta) + 3.0 * u) * DT).clamp(-MAX_SPEED, MAX_SPEED)
    return torch.stack([theta + theta_dot_new * DT, theta_dot_new], dim=-1)


def reward_fn(states, actions):
    theta = wrap_angle(states[..., 0])
    return -(theta**2 + 0.1 * states[..., 1] ** 2 + 0.001 * actions[..., 0] ** 2)


def collect_random_dataset(n_episodes, seed):
    """随机策略采集转移，返回 CPU 上的 (s, a, s')。"""
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)
    s, a, s_next = [], [], []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        for _ in range(200):
            s.append(env.unwrapped.state.copy())
            action = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=1).astype(np.float32)
            env.step(action)
            a.append(action)
            s_next.append(env.unwrapped.state.copy())
    env.close()
    return [torch.as_tensor(np.array(x, dtype=np.float32)) for x in (s, a, s_next)]


def features(states, actions):
    """[..., 2], [..., 1] → [..., 4]：(cos θ, sin θ, θ̇, u)。"""
    return torch.cat([torch.cos(states[..., :1]), torch.sin(states[..., :1]), states[..., 1:], actions], dim=-1)


def train_dynamics(data, n_steps, seed=SEED):
    """在转移数据上训练摆的概率集成：输入特征 [.., 4]，目标是状态变化量 [.., 2]。"""
    s, a, s_next = data
    torch.manual_seed(seed)
    return train_ensemble(ProbabilisticEnsemble(4, 2), features(s, a), s_next - s, n_steps, seed=seed)


@torch.no_grad()
def ensemble_step(model, states, actions, members=None):
    """各成员预测的下一状态（取均值，不采样）。states [B', N, 2]，B' 是用到的成员数。"""
    delta, _ = model.predict(features(states, actions), members)
    return states + delta


# ==================== 第三部分：摆上的认知不确定性 ====================


def pendulum_uncertainty(models, datasets):
    """两份数据各训练一个集成，在整个状态空间上比较分歧、预测的 σ 和真实误差。"""
    print("\n" + "=" * 60)
    print("第三部分：摆上的认知不确定性")
    print("=" * 60)

    print("""
搬到摆上。Pendulum-v1 是确定性环境，没有偶然不确定性，要对付的是认知不确定性。
用两份数据各训练一个 5 个成员的集成：

    数据充足     40 个回合的随机数据（8000 条转移），前几节一直在用
    数据很少     只有 1 个回合（200 条转移），而且运气不好：这个回合里摆只在底部附近晃

然后在整个状态空间上均匀撒点，用真实物理方程算出正确答案，统计三个量：
成员之间的分歧、成员 0 自己报的 σ、成员 0 的真实误差（都是对角速度的预测）。
    """)

    rng = np.random.default_rng(SEED)
    n_test = 100_000
    test_s = torch.as_tensor(np.stack([rng.uniform(-np.pi, np.pi, n_test), rng.uniform(-8, 8, n_test)], -1), dtype=torch.float32)
    test_a = torch.as_tensor(rng.uniform(-2, 2, (n_test, 1)), dtype=torch.float32)
    truth = true_step(test_s, test_a)
    x = features(test_s, test_a).expand(N_MODELS, -1, -1)

    def spearman(a, b):
        """秩相关系数：把两组数各自换成名次，再算普通的相关系数。"""
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return np.corrcoef(ra, rb)[0, 1]

    stats, maps = {}, {}
    n_bins = 32
    ti = np.clip(np.digitize(test_s[:, 0].numpy(), np.linspace(-np.pi, np.pi, n_bins + 1)) - 1, 0, n_bins - 1)
    vi = np.clip(np.digitize(test_s[:, 1].numpy(), np.linspace(-8, 8, n_bins + 1)) - 1, 0, n_bins - 1)
    for name, model in models.items():
        delta, std = model.predict(x)  # [B, n_test, 2]
        disagreement = delta[..., 1].std(dim=0).numpy()            # 成员之间对 Δθ̇ 的分歧
        own_sigma = std[0, :, 1].numpy()                           # 成员 0 自己报的 σ
        error = (test_s + delta[0] - truth)[:, 1].abs().numpy()    # 成员 0 的真实误差

        data_s = datasets[name][0]
        d_theta, d_speed = wrap_angle(data_s[:, 0]).numpy(), data_s[:, 1].numpy()
        # "数据覆盖的范围"：训练数据的角度和角速度各自的最小值到最大值围成的矩形
        covered = ((test_s[:, 1].numpy() >= d_speed.min()) & (test_s[:, 1].numpy() <= d_speed.max()))
        if name == "数据很少":
            covered &= np.abs(test_s[:, 0].numpy()) >= np.abs(d_theta).min()
        stats[name] = dict(
            covered=covered.mean(),
            dis_in=disagreement[covered].mean(), dis_out=disagreement[~covered].mean() if (~covered).any() else np.nan,
            sig_in=own_sigma[covered].mean(), sig_out=own_sigma[~covered].mean() if (~covered).any() else np.nan,
            err_in=error[covered].mean(), err_out=error[~covered].mean() if (~covered).any() else np.nan,
            rho_dis=spearman(disagreement, error), rho_sig=spearman(own_sigma, error),
        )
        grid = np.zeros((n_bins, n_bins))
        np.add.at(grid, (ti, vi), disagreement)
        count = np.zeros((n_bins, n_bins))
        np.add.at(count, (ti, vi), 1)
        maps[name] = (grid / np.maximum(count, 1), d_theta, d_speed)

    few = stats["数据很少"]
    print(f"  「数据很少」的训练数据只覆盖了 |θ| ≥ {np.abs(maps['数据很少'][1]).min():.2f}、|θ̇| ≤ {np.abs(maps['数据很少'][2]).max():.1f} 这一小块，"
          f"占状态空间的 {few['covered']:.0%}。")
    print(f"\n  {'数据很少的模型':<14}{'成员之间的分歧':>14}{'成员 0 自己报的 σ':>18}{'成员 0 的真实误差':>18}")
    print(f"  {'数据覆盖的范围内':<12}{few['dis_in']:>16.4f}{few['sig_in']:>20.4f}{few['err_in']:>20.4f}")
    print(f"  {'数据覆盖的范围外':<12}{few['dis_out']:>16.4f}{few['sig_out']:>20.4f}{few['err_out']:>20.4f}")
    print(f"  {'范围外 / 范围内':<13}{few['dis_out'] / few['dis_in']:>15.0f} 倍{few['sig_out'] / few['sig_in']:>18.1f} 倍{few['err_out'] / few['err_in']:>18.0f} 倍")

    print(f"\n  {'在全部测试点上，与成员 0 真实误差的秩相关':<30}{'分歧':>10}{'自己报的 σ':>14}")
    for name, st in stats.items():
        print(f"  {name:<34}{st['rho_dis']:>12.2f}{st['rho_sig']:>14.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    vmax = np.log10(max(m[0].max() for m in maps.values()))
    vmin = np.log10(min(m[0][m[0] > 0].min() for m in maps.values()))
    for ax, (name, (grid, d_theta, d_speed)) in zip(axes, maps.items()):
        im = ax.imshow(np.log10(grid.T + 1e-9), origin="lower", extent=[-np.pi, np.pi, -8, 8], aspect="auto",
                       cmap="magma", vmin=vmin, vmax=vmax)
        ax.scatter(d_theta, d_speed, s=1, color="cyan", alpha=0.25 if len(d_theta) > 1000 else 0.8)
        fig.colorbar(im, ax=ax, label="log10(成员之间的分歧)")
        ax.set_title(f"{name}（青色点是训练数据）")
        ax.set_xlabel("角度 θ（0 是正上方）")
        ax.set_ylabel("角速度 θ̇")
        ax.grid(False)
    save_figure(fig, OUT / "07-epistemic-uncertainty-map.png")

    full = stats["数据充足"]
    print(f"""
三点观察：

    1. 分歧发现了"没见过"。数据很少的模型，在数据范围之外分歧涨了 {few['dis_out'] / few['dis_in']:.0f} 倍，
       真实误差涨了 {few['err_out'] / few['err_in']:.0f} 倍。图上数据点所在的地方最暗，离数据越远越亮。
    2. 成员自己报的 σ 没有发现。范围外的 σ 只是范围内的 {few['sig_out'] / few['sig_in']:.1f} 倍，
       和真实误差的秩相关是 {few['rho_sig']:.2f}；分歧和真实误差的秩相关是 {few['rho_dis']:.2f}。
       和一维例子的结论一样：没见过的地方，网络报的方差也是瞎猜的。
    3. 数据充足的模型，整张图都暗下去了，分歧和误差的秩相关还有 {full['rho_dis']:.2f}。
       分歧不只是"有没有数据"的开关，它在有数据的地方也大致反映了哪里拟合得差。
    """)
    return stats


# ==================== 第四部分：模型被钻空子 ====================


@torch.no_grad()
def evaluate_sequences(model, members, state, actions):
    """在选定的成员里各展开一遍，回报对成员取平均。actions [N, H, 1] → returns [N]。

    members 只有一个元素时，就是普通的单模型规划。
    model 为 None 时用真实物理方程展开，用来核对想象出来的回报。
    """
    n_members = 1 if model is None else len(members)
    n = actions.shape[0]
    states = state.expand(n_members, n, 2)
    returns = torch.zeros(n_members, n)
    for k in range(actions.shape[1]):
        a = actions[:, k].expand(n_members, n, 1)
        returns += reward_fn(states, a)
        states = true_step(states, a) if model is None else ensemble_step(model, states, a, members)
    return returns.mean(dim=0)


def cem_plan(score_fn, mean, generator, n_candidates=200, n_elites=20, n_iters=5):
    """06 课的 CEM。score_fn 把候选序列 [N, H, 1] 映射成分数 [N]。"""
    std = torch.ones_like(mean)
    for _ in range(n_iters):
        actions = (mean + std * torch.randn(n_candidates, *mean.shape, generator=generator)).clamp(-MAX_TORQUE, MAX_TORQUE)
        elites = actions[score_fn(actions).topk(n_elites).indices]
        mean = 0.1 * mean + 0.9 * elites.mean(dim=0)
        std = elites.std(dim=0, correction=0).clamp_min(0.1)
    return mean


def run_mpc(model, members, seed):
    """跑一个回合的 CEM-MPC，同时记录每一步"想象的回报 − 真实的回报"。

    每一步规划完，把规划出的那条 30 步动作序列分别在模型里和真实物理方程里展开，
    比较两个累计奖励。模型准的话差值应该在 0 附近；系统性大于 0 就是被钻了空子。
    """
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    generator = torch.Generator().manual_seed(seed)
    mean = torch.zeros(HORIZON, 1)
    total_reward, gaps = 0.0, []
    for _ in range(200):
        state = torch.as_tensor(env.unwrapped.state, dtype=torch.float32)
        mean = cem_plan(lambda a: evaluate_sequences(model, members, state, a), mean, generator)
        imagined = evaluate_sequences(model, members, state, mean[None]).item()
        real = evaluate_sequences(None, None, state, mean[None]).item()
        gaps.append(imagined - real)
        _, reward, _, _, _ = env.step(mean[0].numpy().astype(np.float32))
        total_reward += reward
        mean = torch.cat([mean[1:], torch.zeros(1, 1)])
    env.close()
    return total_reward, np.array(gaps)


def model_exploitation(models, seeds):
    """单模型 vs 集成：想象的回报比真实的高多少，真实回报又是多少。"""
    print("\n" + "=" * 60)
    print("第四部分：模型被钻空子")
    print("=" * 60)

    print(f"""
规划器在做 max J(A)，而 J 是模型算出来的。模型的误差有正有负，
优化器对两者的态度不对称：它专挑预测得偏好的地方走。

       真实的回报         ────────╲                    ╱────────
                                  ╲__________________╱

       模型预测的回报      ────────╲         ▲         ╱────────
                                  ╲_______╱ ╲_______╱
                                            │
                                  这里没有数据，模型瞎猜出一个高分。
                                  规划器选了这里，真实结果却很差。

实验：用 CEM-MPC 控制摆（设置同 06 课），{len(seeds)} 个起点。每一步规划完，把规划出的那条
{HORIZON} 步动作序列分别在模型里和真实物理方程里展开，记下"想象的回报 − 真实的回报"。

    单模型    规划时只用集成里的一个成员。5 个成员逐个试一遍
    集成      每条候选序列在 5 个成员里各展开一遍，回报取平均
    """)

    configs = [("数据充足 · 单模型（成员 0）", "数据充足", [0])]
    configs += [(f"数据很少 · 单模型（成员 {b}）", "数据很少", [b]) for b in range(N_MODELS)]
    configs += [("数据很少 · 集成（5 个成员）", "数据很少", list(range(N_MODELS)))]
    results = {}
    print(f"  {'规划用的模型':<24}{'真实回报':>10}{'想象 − 真实（每步平均）':>22}{'耗时':>8}")
    for label, data_name, members in configs:
        start = time.time()
        runs = [run_mpc(models[data_name], members, seed) for seed in seeds]
        returns = np.array([r[0] for r in runs])
        gap = np.mean([r[1].mean() for r in runs])
        results[label] = (returns, gap, runs[0][1])
        print(f"  {label:<22}{returns.mean():>12.1f}{gap:>20.1f}{time.time() - start:>12.0f} 秒")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (label, (_, _, gaps)) in enumerate(results.items()):
        if 2 <= i <= N_MODELS - 1:
            continue  # 单模型只画成员 0 和最后一个，图上不至于太挤
        ax.plot(gaps, linewidth=1.8, label=label)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xlabel("回合里的时间步")
    ax.set_ylabel(f"想象的回报 − 真实的回报（{HORIZON} 步）")
    ax.set_title("规划器对自己的计划有多乐观（第一个起点）")
    ax.legend(fontsize=8)
    save_figure(fig, OUT / "07-model-exploitation.png")
    return results


def exploitation_summary(results):
    """根据实测数字写结论。"""
    values = list(results.values())
    full, singles, ens = values[0], values[1:-1], values[-1]
    single_returns = np.array([s[0].mean() for s in singles])
    single_gaps = np.array([s[1] for s in singles])
    print(f"""
怎么读这张表：

    1. 数据充足时没有这个问题：想象和真实的差只有 {full[1]:+.1f}，真实回报 {full[0].mean():.0f}。
    2. 数据很少时，单模型被钻了空子。5 个成员对自己的计划平均乐观了 {single_gaps.min():.0f} 到 {single_gaps.max():.0f}
       （30 步的累计奖励）。规划器在模型里找到了"能拿高分"的动作序列，
       拿到真实环境里一执行，根本不是那回事。
    3. 5 个成员用的是同一份数据，只是初始化和自助采样不同，真实回报却从 {single_returns.min():.0f} 到 {single_returns.max():.0f}，
       差出几倍。数据范围之外模型怎么外推纯属偶然，有一个成员碰巧外推得还行。
       只训练一个模型的话，你不知道自己拿到的是哪一个。
    4. 集成把乐观的程度压到了 {ens[1]:+.0f}，是单模型的四分之一左右。
       一条序列要在集成里拿高分，得在 5 个成员里平均下来都高；
       某一个成员瞎猜出来的高分，会被其他成员拉下来。
    5. 但集成的真实回报是 {ens[0].mean():.0f}，和 5 个单模型的平均（{single_returns.mean():.0f}）差不多。
       集成让规划器对自己的计划不再那么自欺，却变不出模型没学到的知识：
       200 条全在底部附近的转移里，没有任何关于"摆在上半圈怎么运动"的信息。
       缺的知识只能靠数据补。

所以下一节把最后一环接上：让控制器自己去采数据。它在哪里被骗，就在哪里碰壁，
碰壁的经历进了数据集，下一轮训练后那个空子就被补上了。这就是 PETS。
    """)


# ==================== 第五部分：练习与思考 ====================


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：手算全方差分解
    3 个成员在某个输入上的预测：均值 0.20、0.80、0.50，方差都是 0.01。
    偶然项（方差的平均）和认知项（均值之间的方差）各是多少？

练习 1 参考答案：
    偶然项 = (0.01 + 0.01 + 0.01) / 3 = 0.01。
    均值的平均是 0.50，偏差是 -0.30、0.30、0，
    认知项 = (0.09 + 0.09 + 0) / 3 = 0.06。
    每个成员都自信地报了很小的方差，但它们彼此说的不一样。
    单看任何一个都发现不了问题，放在一起才看得出来。

练习 2：NLL 的最优方差
    对 L = (y - μ)² / (2σ²) + ½·log σ²，固定 μ，对 σ² 求导并令其为零，
    验证最优的 σ² 等于 (y - μ)²。

练习 2 参考答案：
    记 v = σ²，e = (y - μ)²。L = e / (2v) + ½·log v。
    dL/dv = -e / (2v²) + 1 / (2v) = 0，两边乘 2v² 得 -e + v = 0，所以 v = e。
    对很多样本取平均后，网络学到的 σ²(x) 就是它在 x 附近的平均平方误差。

练习 3：不用自助采样
    把 train_ensemble() 里的 bootstrap 换成"所有成员用同一份数据"
    （只保留初始化不同、每一步抽到的小批量不同）。分歧会消失吗？

练习 3 参考答案：
    不会消失。实测（数据很少的那个集成）：数据范围外的分歧从 0.134 变成 0.128，
    范围内从 0.0004 变成 0.0003，几乎没有区别。
    在数据范围之外，成员的预测由各自的随机初始化决定，和有没有自助采样关系不大。
    Deep Ensembles 论文（Lakshminarayanan et al. 2017）的结论就是：
    对神经网络，光靠随机初始化就能得到足够的多样性。

思考题 1：5 个成员在某处的预测完全一致，能说明预测是对的吗？

思考题 1 参考答案：
    不能。分歧小只说明成员们意见一致。如果它们共享同一个系统性的偏差，会一致地错。
    本节就有一个例子：角速度撞到上限 ±8 的地方（03 课讲过的"折角"）。
    实测数据充足的那个集成：在这些点上真实误差是 0.32，是其他地方的 800 倍；
    分歧只有 0.002，是其他地方的 5 倍。5 个成员都用光滑函数去拟合折角，错得差不多。
    集成能发现"没数据"，发现不了"模型的形式不对"。

思考题 2：第四部分里集成取的是 5 个成员回报的平均。还可以怎么用分歧？

思考题 2 参考答案：
    两个方向，取决于你想让智能体做什么：
    - 保守：从平均回报里减去分歧（回报的标准差）作为惩罚，让规划器避开没把握的区域。
      离线强化学习里的 MOPO、MOReL 是这个思路。适合不允许试错的场合。
    - 好奇：把分歧当作奖励加上去，让规划器主动去没把握的区域采数据。
      Plan2Explore（Phase 6 的选读论文）是这个思路。适合可以自由探索的阶段。
    同一个量，符号相反，对应"利用"和"探索"两种目的。

思考题 3：集成要训练 5 个模型，计算量是 5 倍。有没有更便宜的办法估计认知不确定性？

思考题 3 参考答案：
    有几种，各有代价：MC Dropout（推理时保留 dropout，多次前向看差异，便宜但不太准）；
    共享主干、只在最后几层分成多个头；随机网络蒸馏（看一个固定的随机网络能不能被预测）。
    从实践结果看，独立训练的集成仍然是最可靠的基线，Phase 5 的 MBPO、
    Phase 7 的 TD-MPC2（Q 函数集成）都在用。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()

    if is_quick():
        toy_steps, full_episodes, full_steps, few_steps, seeds = 200, 5, 100, 100, [100]
    else:
        toy_steps, full_episodes, full_steps, few_steps, seeds = 4000, 40, 3000, 2000, [100, 101]

    toy_regression(toy_steps)

    datasets = {
        "数据充足": collect_random_dataset(full_episodes, seed=SEED),
        "数据很少": collect_random_dataset(1, seed=10),  # 种子 10 的这个回合，摆一直在底部附近
    }
    start = time.time()
    models = {
        "数据充足": train_dynamics(datasets["数据充足"], full_steps),
        "数据很少": train_dynamics(datasets["数据很少"], few_steps),
    }
    print(f"\n  两个集成训练完毕，共用时 {time.time() - start:.1f} 秒")

    pendulum_uncertainty(models, datasets)
    results = model_exploitation(models, seeds)
    exploitation_summary(results)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：08-pets.py（PETS：把所有部件接成闭环）")
    print("=" * 60)


if __name__ == "__main__":
    main()
