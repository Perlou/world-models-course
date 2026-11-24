"""
更聪明的采样：CEM 与 MPPI
=========================

学习目标：
    1. 理解随机打靶的毛病：不会从已经打过分的样本里学习
    2. 手算并实现交叉熵方法（CEM）：保留精英，用精英重新估计采样分布
    3. 手算并实现 MPPI：按分数给所有样本加权，温度控制权重的集中程度
    4. 掌握热启动、方差下限、平滑更新这些工程细节各自防的是什么
    5. 在相同的模型调用预算下对比三种规划器

核心概念：
    - 采样分布：规划器维护一个高斯分布 N(μ, σ²)，形状 [H, A]，从里面采候选序列
    - CEM：μ, σ ← 分数最高的 K 条序列（精英）的均值、标准差
    - MPPI：w_i ∝ exp(J_i / λ)，μ ← Σ w_i · A_i；λ 是温度
    - 热启动（warm start）：把上一步规划出的均值向前平移一格，作为这一步的起点
    - 有效样本数：1 / Σ w_i²，用来检查 MPPI 的温度设得合不合适

前置知识：
    - 05-random-shooting-mpc.py（批量展开、滚动时域）

对应概念文档：
    - CONCEPT.md 第 7 节「更聪明的采样：CEM 与 MPPI」

预计运行时间：约 5 分钟
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-01")
SEED = 0
MAX_TORQUE = 2.0
MAX_SPEED = 8.0
DT = 0.05
HORIZON = 30
TRAIN_DEVICE = get_device()
# 规划时每次只算几百条序列，批量小。实测这种规模下 CPU 和 MPS 一样快，统一用 CPU。
PLAN_DEVICE = torch.device("cpu")


# ==================== 第一部分：从结果里学习 ====================


def introduction():
    """随机打靶的毛病，以及两种改进。"""
    print("=" * 60)
    print("第一部分：让采样分布跟着分数走")
    print("=" * 60)

    print("""
随机打靶的毛病是不长记性：第 1000 条序列和第 1 条是从同一个分布里采的，
前面 999 次打分的结果完全没用上。

改进的想法：先撒一批点，看哪一片分数高，下一批就往那一片多撒。

    第 1 轮：大范围撒点          第 2 轮：往高分区集中         第 3 轮：进一步收紧

    ·   ·    ·   ·   ·              ·  · ·                       ···
      ·    ★    ·                  · ·★· ·                       ·★·
    ·    ·    ·    ·                 · · ·                        ···
       ·    ·   ·   ·
                                                            ★ = 最优解的位置

做法是维护一个高斯采样分布 N(μ, σ²)，每一轮：采样 → 打分 → 更新分布。
CEM 和 MPPI 只差在"怎么根据分数更新分布"这一步：

    ┌──────┬──────────────────────────────────────────┬──────────────┐
    │      │ 怎么更新均值 μ                             │ 关键超参数     │
    ├──────┼──────────────────────────────────────────┼──────────────┤
    │ CEM  │ 只留分数最高的 K 条（精英），取它们的平均     │ 精英数 K      │
    │ MPPI │ 所有样本都留，按 exp(分数/λ) 加权平均        │ 温度 λ        │
    └──────┴──────────────────────────────────────────┴──────────────┘

先在一维问题上把两种更新手算一遍，再搬到摆上。
    """)


# ==================== 第二部分：一维问题上的手算 ====================


def toy_examples():
    """用代码复核 CONCEPT.md 里的两个手算例子，再看 CEM 多轮迭代的过程。"""
    print("\n" + "=" * 60)
    print("第二部分：在一维问题上手算")
    print("=" * 60)

    print("""
【CEM 走一轮】目标：最大化 J(x) = -(x - 3)²，最优解 x = 3。
初始分布 μ = 0、σ = 2。假设采到 6 个样本，保留 K = 3 个精英
（这是 CONCEPT.md 第 7.3 节的例子）：
    """)
    x = np.array([-1.5, 0.3, 1.8, 2.9, -0.4, 4.1])
    scores = -(x - 3) ** 2
    elite_idx = np.argsort(scores)[-3:]
    elites = x[elite_idx]
    print(f"  样本 x：  {x}")
    print(f"  分数 J：  {', '.join(f'{s:.2f}' for s in scores)}")
    print(f"  精英：    {np.sort(elites)}（分数最高的 3 个）")
    print(f"  新的 μ = 精英的均值   = {elites.mean():.2f}")
    print(f"  新的 σ = 精英的标准差 = {elites.std():.2f}")
    print("  一轮之后，均值从 0 移到 2.93（最优解是 3），标准差从 2 缩到 0.94。")

    print("""
【MPPI 的权重】三条序列的分数是 -10、-12、-20（CONCEPT.md 第 7.6 节的例子）：
    """)
    scores = np.array([-10.0, -12.0, -20.0])
    for lam in (2.0, 10.0):
        shifted = (scores - scores.max()) / lam  # 先减最大值再除以温度，exp 不会溢出
        weights = np.exp(shifted) / np.exp(shifted).sum()
        ess = 1.0 / (weights**2).sum()
        print(f"  温度 λ = {lam:>4}：权重 = {np.round(weights, 3)}，有效样本数 = {ess:.2f}")
    print("""
温度低，权重集中在最好的一条上；温度高，权重摊得均匀。
"有效样本数" 1/Σw² 把这件事变成一个数：3 条样本里实际起作用的相当于几条。

【CEM 多轮迭代】还是最大化 J(x) = -(x - 3)²，每轮 50 个样本、10 个精英：
    """)
    rng = np.random.default_rng(SEED)
    history = {}
    for name, min_std in [("不设方差下限", 0.0), ("方差下限 0.05", 0.05)]:
        mu, sigma, trace = 0.0, 2.0, []
        for it in range(8):
            samples = mu + sigma * rng.standard_normal(50)
            elites = samples[np.argsort(-(samples - 3) ** 2)[-10:]]
            mu, sigma = elites.mean(), max(elites.std(), min_std)
            trace.append((mu, sigma))
        history[name] = trace
    print(f"  {'轮次':<6}{'μ':>8}{'σ（不设下限）':>16}{'σ（下限 0.05）':>18}")
    for it in range(8):
        print(f"  {it + 1:<8}{history['不设方差下限'][it][0]:>8.3f}{history['不设方差下限'][it][1]:>14.4f}"
              f"{history['方差下限 0.05'][it][1]:>18.4f}")
    final_sigma = history["不设方差下限"][-1][1]
    print(f"""
均值两三轮就到了 3 附近。标准差每轮缩小好几倍，8 轮之后只剩 {final_sigma:.0e}。
对这个只有一个峰的函数，收得越紧越好。但规划时目标函数每一步都在变
（状态变了），分布收得太死就再也采不到别处。所以实际使用时要给 σ 设一个下限。
    """)


# ==================== 模型（与 05 课相同）====================


def wrap_angle(theta):
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def true_step(states, actions):
    """完美模型：摆的物理方程。states [N, 2]，actions [N, 1] → [N, 2]。"""
    theta, theta_dot = states[:, 0], states[:, 1]
    u = actions[:, 0].clamp(-MAX_TORQUE, MAX_TORQUE)
    theta_dot_new = (theta_dot + (15.0 * torch.sin(theta) + 3.0 * u) * DT).clamp(-MAX_SPEED, MAX_SPEED)
    return torch.stack([theta + theta_dot_new * DT, theta_dot_new], dim=-1)


def reward_fn(states, actions):
    theta = wrap_angle(states[:, 0])
    return -(theta**2 + 0.1 * states[:, 1] ** 2 + 0.001 * actions[:, 0] ** 2)


def collect_random_dataset(n_episodes, seed=SEED):
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
    return [torch.as_tensor(np.array(x, dtype=np.float32), device=TRAIN_DEVICE) for x in (s, a, s_next)]


class DynamicsModel(nn.Module):
    """03 课的动力学模型。"""

    def __init__(self, hidden=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        self.register_buffer("x_mean", torch.zeros(4))
        self.register_buffer("x_std", torch.ones(4))
        self.register_buffer("y_mean", torch.zeros(2))
        self.register_buffer("y_std", torch.ones(2))

    def features(self, states, actions):
        theta, theta_dot = states[:, :1], states[:, 1:]
        return torch.cat([torch.cos(theta), torch.sin(theta), theta_dot, actions], dim=-1)

    def fit_normalizer(self, states, actions, next_states):
        x, y = self.features(states, actions), next_states - states
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp_min(1e-6))
        self.y_mean.copy_(y.mean(0))
        self.y_std.copy_(y.std(0).clamp_min(1e-6))

    def predict(self, states, actions):
        x = (self.features(states, actions) - self.x_mean) / self.x_std
        return states + self.net(x) * self.y_std + self.y_mean


def train_model(model, transitions, epochs, batch_size=256, lr=1e-3):
    model.to(TRAIN_DEVICE)
    model.fit_normalizer(*transitions)
    n = transitions[0].shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * ((n + batch_size - 1) // batch_size))
    for _ in range(epochs):
        perm = torch.randperm(n, device=TRAIN_DEVICE)
        for i in range(0, n, batch_size):
            s, a, s_next = [t[perm[i:i + batch_size]] for t in transitions]
            loss = (((model.predict(s, a) - s_next) / model.y_std) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model.to(PLAN_DEVICE)


# ==================== 第三部分：三个规划器 ====================


@torch.no_grad()
def evaluate_sequences(step_fn, state, actions):
    """在模型里批量展开，返回每条序列的累计奖励。actions [N, H, 1] → returns [N]。"""
    n, horizon, _ = actions.shape
    states = state.expand(n, 2)
    returns = torch.zeros(n, device=actions.device)
    for k in range(horizon):
        returns += reward_fn(states, actions[:, k])
        states = step_fn(states, actions[:, k])
    return returns


def random_shooting_plan(step_fn, state, mean, generator, n_candidates=1000):
    """随机打靶：均匀采样，返回分数最高的序列。不使用传入的 mean。"""
    actions = (torch.rand(n_candidates, *mean.shape, generator=generator) * 2 - 1) * MAX_TORQUE
    returns = evaluate_sequences(step_fn, state, actions)
    return actions[returns.argmax()]


def cem_plan(step_fn, state, mean, generator, n_candidates=200, n_elites=20, n_iters=5,
             init_std=1.0, min_std=0.1, smoothing=0.1):
    """交叉熵方法。

    参数：
        mean: 采样分布的初始均值 [H, 1]（热启动时是上一步的结果平移一格）
    返回：
        优化后的均值 [H, 1]，它的第一个元素就是要执行的动作
    """
    std = torch.full_like(mean, init_std)
    for _ in range(n_iters):
        noise = torch.randn(n_candidates, *mean.shape, generator=generator)
        actions = (mean + std * noise).clamp(-MAX_TORQUE, MAX_TORQUE)  # [N, H, 1]
        returns = evaluate_sequences(step_fn, state, actions)           # [N]
        elites = actions[returns.topk(n_elites).indices]                # [K, H, 1]
        mean = smoothing * mean + (1 - smoothing) * elites.mean(dim=0)  # 平滑更新
        std = elites.std(dim=0, correction=0).clamp_min(min_std)        # 方差下限
    return mean


def mppi_plan(step_fn, state, mean, generator, n_candidates=200, n_iters=5,
              noise_std=0.5, temperature=1.0, ess_log=None):
    """MPPI。标准差是固定的超参数，只更新均值。

    ess_log 传入一个列表时，会把每一轮的有效样本数记进去。
    """
    for _ in range(n_iters):
        noise = torch.randn(n_candidates, *mean.shape, generator=generator)
        actions = (mean + noise_std * noise).clamp(-MAX_TORQUE, MAX_TORQUE)  # [N, H, 1]
        returns = evaluate_sequences(step_fn, state, actions)                 # [N]
        weights = torch.softmax((returns - returns.max()) / temperature, dim=0)  # [N]，和为 1
        mean = (weights[:, None, None] * actions).sum(dim=0)
        if ess_log is not None:
            ess_log.append(1.0 / (weights**2).sum().item())
    return mean


def run_mpc(plan_fn, step_fn, seed, warm_start=True, **plan_kwargs):
    """在真实环境里跑一个回合。三个规划器共用这一个循环。

    规划器的状态只有一个 mean [H, 1]。每个时间步：
        1. 用 plan_fn 优化 mean
        2. 执行 mean 的第一个动作
        3. 热启动：把 mean 向前平移一格，末尾补 0，留给下一步
    """
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    generator = torch.Generator().manual_seed(seed)
    mean = torch.zeros(HORIZON, 1)
    total_reward, heights = 0.0, []
    for _ in range(200):
        state = env.unwrapped.state
        heights.append(np.cos(state[0]))
        state_t = torch.as_tensor(state, dtype=torch.float32)
        mean = plan_fn(step_fn, state_t, mean, generator, **plan_kwargs)
        _, reward, _, _, _ = env.step(mean[0].numpy().astype(np.float32))
        total_reward += reward
        if warm_start:
            mean = torch.cat([mean[1:], torch.zeros(1, 1)])
        else:
            mean = torch.zeros(HORIZON, 1)
    env.close()
    return total_reward, np.array(heights)


def steps_to_settle(heights):
    """摆从第几步开始一直稳在顶部（cos θ > 0.95）。没稳住返回 200。"""
    below = np.where(heights <= 0.95)[0]
    return 0 if len(below) == 0 else int(below[-1]) + 1


def evaluate(plan_fn, step_fn, seeds, **kwargs):
    """在一组固定的起点上评估，返回 (各回合回报, 各回合稳住所用步数, 每步耗时毫秒)。"""
    start = time.time()
    runs = [run_mpc(plan_fn, step_fn, seed, **kwargs) for seed in seeds]
    ms = (time.time() - start) / (len(seeds) * 200) * 1000
    return np.array([r[0] for r in runs]), np.array([steps_to_settle(r[1]) for r in runs]), ms


def handcrafted_returns(seeds):
    """02 课的手写控制器在同一组起点上的回报，作为参照。"""
    out = []
    for seed in seeds:
        env = gym.make("Pendulum-v1")
        env.reset(seed=seed)
        total = 0.0
        for _ in range(200):
            theta, theta_dot = wrap_angle(env.unwrapped.state[0]), env.unwrapped.state[1]
            if abs(theta) < 0.4 and abs(theta_dot) < 3.0:
                u = -10.0 * theta - 2.0 * theta_dot
            else:
                energy = 0.5 * theta_dot**2 + 15.0 * np.cos(theta)
                direction = 1.0 if theta_dot >= 0 else -1.0
                u = MAX_TORQUE * direction if energy < 15.0 else -MAX_TORQUE * direction
            _, reward, _, _, _ = env.step(np.array([np.clip(u, -MAX_TORQUE, MAX_TORQUE)], dtype=np.float32))
            total += reward
        env.close()
        out.append(total)
    return np.array(out)


# ==================== 第四部分：同等预算下的对比 ====================


def compare_planners(model, seeds, learned_seeds):
    """三种规划器，每个真实步都用 1000 次序列评估。"""
    print("\n" + "=" * 60)
    print("第三部分：同等预算下对比三种规划器")
    print("=" * 60)

    print(f"""
公平的比法是让三者用同样多的模型调用。每个真实步的预算定为 1000 条序列（× {HORIZON} 步）：

    随机打靶   1000 条，一次采完
    CEM       每轮 200 条 × 5 轮，精英 20 条，初始 σ = 1.0，方差下限 0.1，平滑 0.1
    MPPI      每轮 200 条 × 5 轮，噪声 σ = 0.5，温度 λ = 1.0

CEM 和 MPPI 都用热启动。先用完美模型（{len(seeds)} 个起点），再换学到的模型（{len(learned_seeds)} 个起点）。
    """)

    planners = [("随机打靶", random_shooting_plan), ("CEM", cem_plan), ("MPPI", mppi_plan)]
    hand = handcrafted_returns(seeds)
    results = {}
    print(f"  {'完美模型':<10}{'平均回报':>10}{'标准差':>8}{'平均第几步稳住':>14}{'比随机打靶好的回合':>18}{'每步耗时':>10}")
    for name, plan_fn in planners:
        returns, settle, ms = evaluate(plan_fn, true_step, seeds)
        results[name] = returns
        wins = "—" if name == "随机打靶" else f"{(returns > results['随机打靶']).sum()}/{len(seeds)}"
        print(f"  {name:<10}{returns.mean():>12.1f}{returns.std():>10.1f}{settle.mean():>14.0f}{wins:>20}{ms:>10.1f} ms")
    print(f"  {'手写控制器':<8}{hand.mean():>12.1f}{hand.std():>10.1f}")

    learned = {}
    print(f"\n  {'学到的模型':<9}{'平均回报':>10}{'标准差':>8}{'平均第几步稳住':>14}{'每步耗时':>12}")
    for name, plan_fn in planners:
        returns, settle, ms = evaluate(plan_fn, model.predict, learned_seeds)
        learned[name] = returns
        print(f"  {name:<10}{returns.mean():>12.1f}{returns.std():>10.1f}{settle.mean():>14.0f}{ms:>12.1f} ms")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [n for n, _ in planners]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [results[n].mean() for n in names], width=0.4, yerr=[results[n].std() for n in names],
           capsize=4, label=f"完美模型（{len(seeds)} 个回合）")
    ax.bar(xs + 0.2, [learned[n].mean() for n in names], width=0.4, yerr=[learned[n].std() for n in names],
           capsize=4, label=f"学到的模型（{len(learned_seeds)} 个回合）")
    ax.axhline(hand.mean(), color="gray", linestyle="--", label="手写控制器")
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylabel("平均回报（误差棒是标准差）")
    ax.set_title(f"同等预算（每步 1000 条序列）下的三种规划器")
    ax.legend(fontsize=9)
    save_figure(fig, OUT / "06-planner-comparison.png")

    rs, cem, mppi = [results[n].mean() for n in names]
    print(f"""
怎么读这两张表：

    - 同样的预算，CEM（{cem:.0f}）和 MPPI（{mppi:.0f}）都比随机打靶（{rs:.0f}）好，
      已经接近手写控制器（{hand.mean():.0f}）。别忘了手写控制器用了摆的能量公式，
      而规划器只有一个模型和一个奖励函数。
    - 各回合之间的标准差很大，那主要是起点不同造成的（起点离顶部近的回合天然得分高）。
      所以要逐回合配对比较："比随机打靶好的回合"那一列比平均值更说明问题。
    - 换成学到的模型，三者的排序不变。
    - 预算相同不等于耗时相同。CEM 和 MPPI 要串行跑 5 轮，每轮的批量只有 200；
      随机打靶一次算完 1000 条。批量越大越划算，所以随机打靶每步更快。
      在 GPU 上、模型更大时，这个差别会更明显。
    """)
    return results, learned, hand


def budget_sweep(seeds, full_results):
    """把预算砍到很小，看谁先撑不住。"""
    print("\n" + "=" * 60)
    print("第四部分：预算小的时候差距有多大")
    print("=" * 60)

    print("""
上一部分每步有 1000 条序列可用，摆又只有一维动作，随机打靶也能凑合。
把预算砍下去才看得出效率的差别（完美模型；CEM 和 MPPI 仍然分 5 轮，精英取十分之一）：
    """)

    budgets = [50, 100, 300]
    table = {"随机打靶": [], "CEM": [], "MPPI": []}
    print(f"  {'每步预算':<10}{'随机打靶':>10}{'CEM':>10}{'MPPI':>10}")
    for b in budgets:
        n = b // 5
        table["随机打靶"].append(evaluate(random_shooting_plan, true_step, seeds, n_candidates=b)[0].mean())
        table["CEM"].append(evaluate(cem_plan, true_step, seeds, n_candidates=n, n_elites=max(2, n // 10))[0].mean())
        table["MPPI"].append(evaluate(mppi_plan, true_step, seeds, n_candidates=n)[0].mean())
        print(f"  {b:<12}{table['随机打靶'][-1]:>10.1f}{table['CEM'][-1]:>12.1f}{table['MPPI'][-1]:>10.1f}")
    budgets.append(1000)
    for name in table:
        table[name].append(full_results[name][:len(seeds)].mean())
    print(f"  {1000:<12}{table['随机打靶'][-1]:>10.1f}{table['CEM'][-1]:>12.1f}{table['MPPI'][-1]:>10.1f}   （上一部分的结果，取同样的起点）")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, values in table.items():
        ax.plot(budgets, values, marker="o", linewidth=2, label=name)
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("每个真实步的预算（序列条数）")
    ax.set_ylabel(f"平均回报（{len(seeds)} 个回合）")
    ax.set_title("预算越小，会学习的采样方法优势越大")
    ax.legend()
    save_figure(fig, OUT / "06-budget-sweep.png")

    print(f"""
每步只有 50 条序列时，随机打靶掉到 {table['随机打靶'][0]:.0f}，CEM 和 MPPI 还在 {table['CEM'][0]:.0f} 和 {table['MPPI'][0]:.0f}。
用 50 条序列的 CEM，比用 1000 条序列的随机打靶（{table['随机打靶'][-1]:.0f}）还好。
（CEM 和 MPPI 各自那一列里几十分的起伏不用解读：每个格子只有 {len(seeds)} 个回合，
这就是噪声的大小。能下结论的是它们和随机打靶之间上百分的差距。）

省下的这 20 倍预算从哪来？两个地方：
    - 一个时间步之内：后几轮的样本集中在前几轮发现的高分区域，不再浪费在明显差的地方。
    - 时间步之间：热启动把上一步优化好的均值传给下一步。相邻两步的最优序列
      几乎一样，等于每一步都站在上一步的肩膀上。随机打靶每一步都从零开始。

动作维度越高，这个差距越大。7 个关节的机械臂、30 步视野是 210 维，
随机打靶基本无从下手，CEM 和 MPPI 是这类问题上的默认选择。
    """)
    return table


# ==================== 第五部分：工程细节各管什么 ====================


def engineering_details(seeds, full_results):
    """热启动、迭代轮数、MPPI 的温度。"""
    print("\n" + "=" * 60)
    print("第五部分：工程细节各管什么")
    print("=" * 60)

    base_cem = full_results["CEM"][:len(seeds)].mean()
    base_mppi = full_results["MPPI"][:len(seeds)].mean()

    print("\n  【热启动】同样的设置，关掉热启动（每一步都从 μ = 0 开始）：")
    no_warm_cem = evaluate(cem_plan, true_step, seeds, warm_start=False)[0].mean()
    no_warm_mppi = evaluate(mppi_plan, true_step, seeds, warm_start=False)[0].mean()
    print(f"    CEM ：有热启动 {base_cem:8.1f}   没有 {no_warm_cem:8.1f}")
    print(f"    MPPI：有热启动 {base_mppi:8.1f}   没有 {no_warm_mppi:8.1f}")

    print("\n  【迭代轮数】CEM 每轮固定 200 条，只改轮数（预算随之变化）：")
    iter_results = []
    for n_iters in (1, 2, 5, 10):
        r = evaluate(cem_plan, true_step, seeds, n_iters=n_iters)[0].mean()
        iter_results.append(r)
        print(f"    {n_iters:>2} 轮（每步 {200 * n_iters:>4} 条）：{r:8.1f}")

    print("\n  【MPPI 的温度】其余不变，只改 λ，同时记录有效样本数（每轮 200 条里实际起作用的条数）：")
    temp_results = []
    for lam in (0.1, 1.0, 10.0, 100.0):
        ess_log = []
        r = evaluate(mppi_plan, true_step, seeds, temperature=lam, ess_log=ess_log)[0].mean()
        temp_results.append((lam, r, np.mean(ess_log)))
        print(f"    λ = {lam:>5}：平均回报 {r:8.1f}   平均有效样本数 {np.mean(ess_log):6.1f} / 200")

    ess_low, ess_high = temp_results[0][2], temp_results[-1][2]
    print(f"""
三点观察：

    1. 热启动：关掉之后 CEM 从 {base_cem:.0f} 变到 {no_warm_cem:.0f}，MPPI 从 {base_mppi:.0f} 变到 {no_warm_mppi:.0f}。
       每一步都从 μ = 0 重新搜，5 轮迭代不够把分布挪到好的位置。
       热启动不花任何额外的计算，是这两个规划器里性价比最高的一个细节。
    2. 迭代轮数：从 1 轮到 5 轮，回报从 {iter_results[0]:.0f} 涨到 {iter_results[2]:.0f}，改善了 {iter_results[2] - iter_results[0]:.0f} 分。
       再从 5 轮加到 10 轮，预算翻了一倍，回报变化 {iter_results[3] - iter_results[2]:+.0f} 分。
       后面这个数和 {len(seeds)} 个回合的噪声是一个量级（上一部分的表里，同一个规划器
       相邻两档预算之间就能差出几十分），所以只能说：头几轮最值钱，之后收益很快变小。
       有热启动时每一步只需要在上一步的基础上微调，不需要迭代到收敛。
    3. 温度要配合分数的尺度。λ = 0.1 时 200 条样本里只有 {ess_low:.1f} 条在起作用，
       接近"只取最好的几条"；λ = 100 时是 {ess_high:.0f} 条，接近全部等权，
       等于没在优化，回报掉到 {temp_results[-1][1]:.0f}。
       这里效果最好的 λ = 1 对应约 {temp_results[1][2]:.0f} 条。调温度时盯着有效样本数，
       比盯着 λ 本身的数值有用：换一个任务，分数的尺度变了，合适的 λ 也跟着变，
       而"有效样本数占一成左右"这个经验可以带着走。
    """)
    return no_warm_cem, no_warm_mppi, iter_results, temp_results


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：手算一轮 CEM
    最大化 J(x) = -|x - 1|。当前 μ = -1、σ = 1，采到 5 个样本：
    -2.1、-0.6、0.2、-1.3、0.9，保留 K = 2 个精英。算出新的 μ 和 σ。

练习 1 参考答案：
    分数依次是 -3.1、-1.6、-0.8、-2.3、-0.1。精英是 0.9 和 0.2。
    μ = (0.9 + 0.2) / 2 = 0.55。
    σ² = ((0.9 - 0.55)² + (0.2 - 0.55)²) / 2 = 0.1225，σ = 0.35。
    均值从 -1 移到 0.55，朝最优解 1 走了一大步。

练习 2：手算 MPPI 的权重
    两条序列的分数是 -5 和 -8。温度取多少时，好的那条权重恰好是 0.9？

练习 2 参考答案：
    减去最大值后是 0 和 -3。权重之比 w_1 / w_2 = exp(3/λ) = 0.9 / 0.1 = 9。
    所以 3/λ = ln 9 = 2.197，λ = 1.365。
    分数差距不变时，温度越低，好样本的权重越大。

练习 3：混合两种做法
    TD-MPC2 的规划器先取前 K 个精英，再在精英内部按 exp(J/λ) 加权算均值和标准差。
    在 cem_plan() 里实现这个变体（把 elites.mean / elites.std 换成加权版本）。

练习 3 参考答案：
    top = returns.topk(n_elites)
    elites = actions[top.indices]
    w = torch.softmax((top.values - top.values.max()) / temperature, dim=0)[:, None, None]
    mean = (w * elites).sum(dim=0)
    std = ((w * (elites - mean) ** 2).sum(dim=0)).sqrt().clamp_min(min_std)
    这样既有 CEM 的"只看排名靠前的"（对分数尺度不敏感），
    又有 MPPI 的"更好的样本说话更响"。Phase 7 会用到它。

思考题 1：CEM 返回的是均值 μ 的第一个动作，而不是分数最高的那条样本的第一个动作。为什么？

思考题 1 参考答案：
    均值是很多条好样本的平均，单条样本里碰巧的坏动作被平均掉了，更稳定。
    最好的那条样本只是在"模型里"分数最高，而模型有误差；
    押在单条样本上，等于把模型的每一个小误差都当真。
    代价是：如果好的解有两个相隔很远的峰（往左甩和往右甩都行），
    平均值可能落在两个峰中间，是一个哪边都不靠的坏动作。

思考题 2：摆在正下方静止时，往左甩和往右甩完全对称。CEM 的高斯分布只有一个峰，
    它是怎么"选边"的？

思考题 2 参考答案：
    靠采样的随机性。第一轮的精英里，碰巧偏左的多一点，均值就往左偏一点；
    下一轮以这个偏左的均值为中心采样，偏左的精英更多，越来越偏。
    这是一种自发的对称破缺。它通常能工作，但偶尔会在中间犹豫几步。
    更稳妥的办法是用多峰的采样分布（比如高斯混合），代价是实现更复杂。

思考题 3：为什么不直接对动作序列做梯度上升？模型是可微的，∂J/∂A 算得出来。

思考题 3 参考答案：
    可以，这叫基于梯度的规划（shooting with gradients）。但有三个麻烦：
    - 梯度要沿模型反传 H 步，和 RNN 一样会爆炸或消失，在不稳定的系统上尤其严重；
    - 目标函数不是凸的，梯度法容易卡在局部最优，而采样方法自带探索；
    - 动作有上下界、奖励可能不可导（比如稀疏奖励）。
    采样方法不需要梯度，对这些都不敏感，所以在视野不长时是更省心的选择。
    "沿模型反传梯度"这个想法没有被放弃：它用在了训练策略上，
    Phase 5 的 SVG 和 Phase 6 的 Dreamer V1 都是这么做的。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()
    toy_examples()

    if is_quick():
        n_episodes, epochs, seeds, learned_seeds, sweep_seeds = 10, 5, [100], [100], [100]
    else:
        n_episodes, epochs = 40, 100
        seeds, learned_seeds, sweep_seeds = list(range(100, 110)), list(range(100, 105)), list(range(100, 105))

    torch.manual_seed(SEED)
    model = train_model(DynamicsModel(), collect_random_dataset(n_episodes), epochs)

    results, _, _ = compare_planners(model, seeds, learned_seeds)
    budget_sweep(sweep_seeds, results)
    engineering_details(sweep_seeds, results)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：07-uncertainty-and-ensembles.py（不确定性与概率集成）")
    print("=" * 60)


if __name__ == "__main__":
    main()
