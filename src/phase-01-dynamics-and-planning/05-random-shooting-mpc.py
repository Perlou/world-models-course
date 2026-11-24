"""
随机打靶与模型预测控制（MPC）
============================

学习目标：
    1. 把"规划"写成一个优化问题：找一条让模型里的累计奖励最大的动作序列
    2. 实现最简单的解法——随机打靶，并用批量展开让它跑得动
    3. 理解滚动时域：只执行第一个动作，然后从真实状态重新规划
    4. 先用完美模型验证规划器，再换上学到的模型（调试时先排除一半的嫌疑）
    5. 实测规划视野 H、候选数 N、重新规划的间隔各自怎么影响控制效果

核心概念：
    - 规划即优化：A* = argmax J(A)，J(A) 是动作序列 A 在模型里展开得到的累计奖励
    - 随机打靶（random shooting）：随机采 N 条序列，选分数最高的
    - 滚动时域（receding horizon）：单次规划是开环的，每步重新规划使整体成为闭环
    - 规划视野 H：太短看不到远处的奖励，太长则模型误差累积、搜索变难

前置知识：
    - 03-learning-dynamics-mlp.py（动力学模型）、04-compounding-error.py（开环展开）

对应概念文档：
    - CONCEPT.md 第 6 节「规划即优化：模型预测控制（MPC）」

预计运行时间：约 3 分钟
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
DEVICE = get_device()


# ==================== 第一部分：规划即优化 ====================


def introduction():
    """把规划写成优化问题。"""
    print("=" * 60)
    print("第一部分：规划即优化")
    print("=" * 60)

    print("""
有了模型，选动作最直接的办法：把各种做法在模型里试一遍，挑结果最好的。
01 课比较了 3 种做法，这一节把它换成 1000 条随机动作序列。

站在当前状态 s_t，要找一条长度为 H 的动作序列 A = (a_0, ..., a_{H-1})，
让它在模型里展开得到的累计奖励 J(A) 最大。随机打靶的解法：

            当前状态 s_t
                 │
         ┌───────┼───────┬─────── ··· ───────┐
         ▼       ▼       ▼                   ▼
       序列 1   序列 2   序列 3      ···     序列 N       每条都是 H 个随机动作
         │       │       │                   │
       模型展开  模型展开  模型展开            模型展开      在模型里各推 H 步
         │       │       │                   │
       J = -85  J = -42  J = -97    ···    J = -63      累计奖励
                 │
                 └──── 最高分 ────▶ 只执行序列 2 的第一个动作

然后观察真实的新状态，从头再来一遍。规划的窗口随时间向前滑动，叫滚动时域：

    时刻 t：    规划 [a_t, a_t+1, ..., a_t+H-1]，执行 a_t，其余丢弃
    时刻 t+1：  拿到真实的 s_t+1，重新规划 [a_t+1, ..., a_t+H]，执行 a_t+1
    ...

规划了 H 步只用 1 步，换来的是：模型的误差最多累积 H 步（上一节的结论），
而且每一步都能根据真实情况修正。
    """)
    print(f"  使用设备：{DEVICE}")


# ==================== 模型：完美模型与学到的模型 ====================


def wrap_angle(theta):
    """把角度折到 [-π, π)。NumPy 数组和 PyTorch 张量都能用。"""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def true_step(states, actions):
    """完美模型：摆的物理方程（PyTorch 批量版）。states [N, 2]，actions [N, 1] → [N, 2]。"""
    theta, theta_dot = states[:, 0], states[:, 1]
    u = actions[:, 0].clamp(-MAX_TORQUE, MAX_TORQUE)
    theta_dot_new = (theta_dot + (15.0 * torch.sin(theta) + 3.0 * u) * DT).clamp(-MAX_SPEED, MAX_SPEED)
    return torch.stack([theta + theta_dot_new * DT, theta_dot_new], dim=-1)


def reward_fn(states, actions):
    """奖励函数（已知，不用学）。states [N, 2]，actions [N, 1] → [N]。"""
    theta = wrap_angle(states[:, 0])
    return -(theta**2 + 0.1 * states[:, 1] ** 2 + 0.001 * actions[:, 0] ** 2)


def collect_random_dataset(n_episodes, seed=SEED):
    """用随机策略采集转移 (s, a, s')，返回放在设备上的三个张量。"""
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
    return [torch.as_tensor(np.array(x, dtype=np.float32), device=DEVICE) for x in (s, a, s_next)]


class DynamicsModel(nn.Module):
    """03 课的动力学模型：输入 (cos θ, sin θ, θ̇, u)，输出标准化后的 (Δθ, Δθ̇)。"""

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
    """单步损失训练（与 03 课相同）。"""
    model.to(DEVICE)
    model.fit_normalizer(*transitions)
    n = transitions[0].shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * ((n + batch_size - 1) // batch_size))
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            s, a, s_next = [t[perm[i:i + batch_size]] for t in transitions]
            loss = (((model.predict(s, a) - s_next) / model.y_std) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model


# ==================== 第二部分：随机打靶规划器 ====================


@torch.no_grad()
def evaluate_sequences(step_fn, state, actions):
    """在模型里批量展开 N 条动作序列，返回每条的累计奖励。

    参数：
        step_fn: 模型的单步函数，(states [N, 2], actions [N, 1]) → next_states [N, 2]
        state:   当前的真实状态 [2]
        actions: 候选动作序列 [N, H, 1]
    返回：
        returns [N]，每条序列在模型里的累计奖励 J
    """
    n, horizon, _ = actions.shape
    states = state.expand(n, 2)  # 把当前状态复制 N 份作为起点
    returns = torch.zeros(n, device=actions.device)
    for k in range(horizon):
        returns += reward_fn(states, actions[:, k])
        states = step_fn(states, actions[:, k])
    return returns


def random_shooting(step_fn, state, horizon, n_candidates, generator):
    """随机打靶：均匀随机采 N 条序列，返回分数最高的那一条 [H, 1] 和它的分数。"""
    actions = (torch.rand(n_candidates, horizon, 1, generator=generator, device=state.device) * 2 - 1) * MAX_TORQUE
    returns = evaluate_sequences(step_fn, state, actions)
    best = returns.argmax()
    return actions[best], returns[best].item()


def run_mpc(step_fn, horizon, n_candidates, seed, replan_every=1, device=DEVICE):
    """在真实环境里跑一个回合的 MPC。返回真实回报和摆尖高度序列。

    replan_every=1 是标准的 MPC：每一步都重新规划，只执行第一个动作。
    replan_every=k 时每 k 步才规划一次，中间照着上次的计划开环执行。
    """
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    total_reward, heights, plan = 0.0, [], None
    for t in range(200):
        state = env.unwrapped.state
        heights.append(np.cos(state[0]))
        if t % replan_every == 0:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
            plan, _ = random_shooting(step_fn, state_t, max(horizon, replan_every), n_candidates, generator)
            plan = plan.cpu().numpy()
        action = plan[t % replan_every]
        _, reward, _, _, _ = env.step(action.astype(np.float32))
        total_reward += reward
    env.close()
    return total_reward, np.array(heights)


def handcrafted_policy(state):
    """02 课的手写控制器，作为参照。"""
    theta, theta_dot = wrap_angle(state[0]), state[1]
    if abs(theta) < 0.4 and abs(theta_dot) < 3.0:
        u = -10.0 * theta - 2.0 * theta_dot
    else:
        energy = 0.5 * theta_dot**2 + 15.0 * np.cos(theta)
        direction = 1.0 if theta_dot >= 0 else -1.0
        u = MAX_TORQUE * direction if energy < 15.0 else -MAX_TORQUE * direction
    return float(np.clip(u, -MAX_TORQUE, MAX_TORQUE))


def run_reference(policy, seed):
    """跑一个回合的参照策略（随机或手写），返回真实回报和摆尖高度序列。"""
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total_reward, heights = 0.0, []
    for _ in range(200):
        state = env.unwrapped.state
        heights.append(np.cos(state[0]))
        u = rng.uniform(-MAX_TORQUE, MAX_TORQUE) if policy == "random" else handcrafted_policy(state)
        _, reward, _, _, _ = env.step(np.array([u], dtype=np.float32))
        total_reward += reward
    env.close()
    return total_reward, np.array(heights)


def steps_to_settle(heights):
    """摆从第几步开始一直稳在顶部（cos θ > 0.95）。没稳住返回 200。"""
    below = np.where(heights <= 0.95)[0]
    return 0 if len(below) == 0 else int(below[-1]) + 1


def evaluate(step_fn, horizon, n_candidates, seeds, **kwargs):
    """在一组固定的起点（种子）上跑 MPC，返回各回合的回报。"""
    return np.array([run_mpc(step_fn, horizon, n_candidates, seed, **kwargs)[0] for seed in seeds])


# ==================== 第三部分：先用完美模型验证规划器 ====================


def perfect_model_mpc(seeds, horizon, n_candidates):
    """用物理方程当模型，确认规划器本身能工作。"""
    print("\n" + "=" * 60)
    print("第二部分：先用完美模型验证规划器")
    print("=" * 60)

    print(f"""
新写的规划器接上新训练的模型，效果不好时分不清是谁的问题。
所以先用 01 课的物理方程当模型。模型没有误差，效果不好就一定是规划器的问题。

设置：视野 H = {horizon}，每步采 N = {n_candidates} 条候选序列，在 {len(seeds)} 个固定的起点上各跑一个回合。
    """)

    refs = {name: [run_reference(name, seed) for seed in seeds] for name in ("random", "hand")}
    mpc_runs = [run_mpc(true_step, horizon, n_candidates, seed) for seed in seeds]
    returns = {
        "随机策略": np.array([r[0] for r in refs["random"]]),
        "手写控制器（02 课）": np.array([r[0] for r in refs["hand"]]),
        "随机打靶 MPC + 完美模型": np.array([r[0] for r in mpc_runs]),
    }
    settle = {
        "随机策略": np.mean([steps_to_settle(r[1]) for r in refs["random"]]),
        "手写控制器（02 课）": np.mean([steps_to_settle(r[1]) for r in refs["hand"]]),
        "随机打靶 MPC + 完美模型": np.mean([steps_to_settle(r[1]) for r in mpc_runs]),
    }
    print(f"  {'控制器':<22}{'平均回报':>10}{'标准差':>8}{'最差的回合':>12}{'平均第几步稳住':>16}")
    for name, r in returns.items():
        print(f"  {name:<20}{r.mean():>12.1f}{r.std():>10.1f}{r.min():>14.1f}{settle[name]:>16.0f}")

    mpc_mean = returns["随机打靶 MPC + 完美模型"].mean()
    hand_mean = returns["手写控制器（02 课）"].mean()
    # 最后 50 步摆尖高度都在 0.95 以上，算作"稳在了顶部"
    n_stable = sum(r[1][-50:].min() > 0.95 for r in mpc_runs)
    print(f"\n  MPC 的 {len(seeds)} 个回合里，有 {n_stable} 个在最后 50 步稳在了顶部（cos θ > 0.95）。")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = np.arange(200) * DT
    ax.plot(t, refs["random"][0][1], color="gray", alpha=0.7, label="随机策略")
    ax.plot(t, refs["hand"][0][1], linewidth=2, label="手写控制器")
    ax.plot(t, mpc_runs[0][1], linewidth=2, label="随机打靶 MPC + 完美模型")
    ax.axhline(1.0, color="gray", linestyle=":")
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("摆尖高度 cos θ")
    ax.set_title("同一个起点，三种控制器")
    ax.legend()
    save_figure(fig, OUT / "05-mpc-perfect-model.png")

    print(f"""
规划器能工作：MPC 把摆甩上去并稳住了，平均回报 {mpc_mean:.0f}，手写控制器是 {hand_mean:.0f}。
差距出在甩上去的过程：手写控制器平均第 {settle["手写控制器（02 课）"]:.0f} 步稳住，MPC 要到第 {settle["随机打靶 MPC + 完美模型"]:.0f} 步。
摆没到顶部时每一步都在扣分，晚到一步就多扣一步。随机撒出来的 1000 条序列里
最好的那条，也比不上按能量公式精确算出来的甩法。下一节换更好的优化方法来缩小这个差距。

注意 MPC 没有用到任何"怎么甩摆"的知识。手写控制器里的能量公式、
比例—微分增益、切换条件，MPC 一样都不知道。它只有两样东西：
一个能预测"接下来会怎样"的模型，和一个给结果打分的奖励函数。
换一个系统，把模型和奖励函数换掉就行，规划器一行不用改。
    """)
    return returns, n_stable


# ==================== 第四部分：换上学到的模型 ====================


def learned_model_mpc(model, seeds, horizon, n_candidates, perfect_returns):
    """把物理方程换成神经网络。"""
    print("\n" + "=" * 60)
    print("第三部分：换上学到的模型")
    print("=" * 60)

    print("""
现在把物理方程换成 03 课的神经网络。规划器的代码不变，只是 step_fn 换了一个。
这个模型只见过随机策略采的 40 个回合（8000 条转移），从没见过"摆稳在顶部"是什么样。
    """)

    start = time.time()
    learned = evaluate(model.predict, horizon, n_candidates, seeds)
    elapsed = time.time() - start
    perfect = perfect_returns["随机打靶 MPC + 完美模型"]
    print(f"  {'模型':<16}{'平均回报':>10}{'标准差':>8}{'最差的回合':>12}")
    print(f"  {'完美模型':<16}{perfect.mean():>12.1f}{perfect.std():>10.1f}{perfect.min():>14.1f}")
    print(f"  {'学到的模型':<15}{learned.mean():>12.1f}{learned.std():>10.1f}{learned.min():>14.1f}")
    ms_per_step = elapsed / (len(seeds) * 200) * 1000
    print(f"\n  用学到的模型规划，每个真实步耗时 {ms_per_step:.1f} 毫秒"
          f"（{n_candidates} 条候选 × {horizon} 步 = {n_candidates * horizon:,} 次模型预测）。")
    print(f"  环境的一步对应真实时间 {DT * 1000:.0f} 毫秒。")

    gap = learned.mean() - perfect.mean()
    print(f"""
学到的模型和完美模型的平均回报相差 {gap:+.1f}，在回合之间的波动范围以内。
上一节看到这个模型开环 50 步的误差被放大了上千倍，为什么控制效果几乎不受影响？

    - 每一步都从真实状态重新规划，模型最多被开环使用 {horizon} 步。
    - 规划结果里只有第一个动作被执行。第一个动作对不对，主要取决于前几步的预测，
      而前几步的误差只有千分之几弧度。
    - 规划器要的不是"每条轨迹都预测得准"，而是"好序列的分数比坏序列高"。
      排序比绝对数值宽容得多。

这是 MPC 和世界模型搭配的根本原因：模型只在短视野内可信，MPC 恰好只用短视野。
    """)
    return learned, ms_per_step


# ==================== 第五部分：三个超参数 ====================


def horizon_sweep(model, seeds, n_candidates, horizons):
    """规划视野 H 的影响。"""
    print("\n" + "=" * 60)
    print("第四部分：规划视野 H 怎么选")
    print("=" * 60)

    print("""
视野 H 是 MPC 最重要的超参数，受两个方向的挤压：

       H 太短                                          H 太长
       ◀─────────────────────────────────────────────────▶
       看不到远处的奖励                      模型误差累积，想象的结果不可信
       （摆要先往反方向甩才能上去，            搜索空间维度变高，随机撒点更难撒中
         短视的规划器不会这么做）              计算量随 H 线性增长
    """)

    results = {"完美模型": [], "学到的模型": []}
    print(f"  {'视野 H':<8}{'完美模型：平均回报':>18}{'学到的模型：平均回报':>20}")
    for h in horizons:
        perfect = evaluate(true_step, h, n_candidates, seeds)
        learned = evaluate(model.predict, h, n_candidates, seeds)
        results["完美模型"].append(perfect)
        results["学到的模型"].append(learned)
        print(f"  {h:<10}{perfect.mean():>16.1f} ± {perfect.std():<6.0f}{learned.mean():>16.1f} ± {learned.std():<6.0f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, runs in results.items():
        means = [r.mean() for r in runs]
        stds = [r.std() for r in runs]
        ax.errorbar(horizons, means, yerr=stds, marker="o", capsize=4, linewidth=2, label=name)
    ax.set_xscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("规划视野 H（步）")
    ax.set_ylabel(f"平均回报（{len(seeds)} 个回合，误差棒是标准差）")
    ax.set_title("规划视野对 MPC 的影响")
    ax.legend()
    save_figure(fig, OUT / "05-horizon-sweep.png")

    perfect = [r.mean() for r in results["完美模型"]]
    learned = [r.mean() for r in results["学到的模型"]]
    best = horizons[int(np.argmax(learned))]
    print(f"""
怎么读这张表：

    - H = {horizons[0]}：回报 {learned[0]:.0f}，和随机策略差不多。{horizons[0]} 步只有 {horizons[0] * DT:.2f} 秒，
      在这么短的时间里，"先往反方向甩"只会让奖励变差，规划器永远不会选它，
      摆也就一直甩不上去。
    - H 增加到 {best} 左右效果最好（{max(learned):.0f}）。把摆从底部甩上去需要一秒左右，
      视野要长到能看见"现在往反方向甩，待会儿能上去"。
    - H = {horizons[-1]}：两个模型都变差了（完美模型 {perfect[-1]:.0f}，学到的模型 {learned[-1]:.0f}）。
      完美模型没有预测误差，变差的原因只能是搜索：动作序列从 {best} 维变成 {horizons[-1]} 维，
      同样撒 1000 个点，撒中好序列的机会更小了。

关于最后一行要多说一句。这次运行里学到的模型在 H = {horizons[-1]} 时比完美模型差得更多，
看上去像是"视野太长，模型误差开始起作用"。但每个配置只跑了 {len(seeds)} 个回合，
标准差有好几百。我单独用 20 个起点复核过：学到的模型 -431，完美模型 -401，
逐回合比较是 6 个更差、5 个更好，分不出高下。
所以这里能下的结论只有：在这个任务上，长视野的瓶颈是随机打靶搜不动，不是模型不准。
回合数少的时候，别急着给一个差距编故事。
    """)
    return results


def candidates_and_replan(model, seeds, horizon, candidate_counts, replan_intervals):
    """候选数 N 和重新规划间隔的影响。"""
    print("\n" + "=" * 60)
    print("第五部分：候选数 N，以及为什么每一步都要重新规划")
    print("=" * 60)

    print(f"\n  候选数 N 的影响（学到的模型，H = {horizon}）：")
    print(f"  {'候选数 N':<10}{'平均回报':>10}{'标准差':>8}")
    n_results = []
    for n in candidate_counts:
        r = evaluate(model.predict, horizon, n, seeds)
        n_results.append(r)
        print(f"  {n:<12}{r.mean():>10.1f}{r.std():>10.1f}")

    print(f"\n  重新规划间隔的影响（学到的模型，H = {horizon}，N = {candidate_counts[-1]}）：")
    print(f"  {'每几步规划一次':<12}{'平均回报':>10}{'标准差':>8}")
    k_results = []
    for k in replan_intervals:
        r = evaluate(model.predict, horizon, candidate_counts[-1], seeds, replan_every=k)
        k_results.append(r)
        print(f"  {k:<16}{r.mean():>10.1f}{r.std():>10.1f}")

    n_means = [r.mean() for r in n_results]
    k_means = [r.mean() for r in k_results]
    print(f"""
两点观察：

    1. 候选越多越好，但收益递减：N 从 {candidate_counts[0]} 到 {candidate_counts[1]}，回报从 {n_means[0]:.0f} 涨到 {n_means[1]:.0f}；
       再到 {candidate_counts[-1]}，只涨到 {n_means[-1]:.0f}。随机撒点的效率很低，
       想再好一点就得多撒十倍。下一节的 CEM 用同样的预算能搜得更好。
    2. 规划一次、照着执行 {replan_intervals[-1]} 步再重新规划，回报从 {k_means[0]:.0f} 掉到 {k_means[-1]:.0f}。
       一条随机打靶找到的序列，只有开头几步是靠谱的；后面的动作
       只是"碰巧没有太拖后腿"。每一步都重新规划，等于每一步都只用最靠谱的那个动作。
    """)
    return n_results, k_results


# ==================== 第六部分：计算量 ====================


def timing(model, horizon, n_candidates):
    """一次规划要多久。"""
    print("\n" + "=" * 60)
    print("第六部分：计算量")
    print("=" * 60)

    print(f"""
MPC 的代价是每一个真实的时间步都要解一次优化问题：

    每个真实步的模型调用次数 = N × H = {n_candidates} × {horizon} = {n_candidates * horizon:,}

能承受这个量靠的是批量化：把 N 条序列的状态堆成一个 [N, 2] 的张量，
每一步对整个批次做一次前向计算。于是每个真实步只需要 H 次前向计算，每次的批大小是 N。
    """)

    devices = ["cpu"] + ([DEVICE.type] if DEVICE.type != "cpu" else [])
    results = {}
    for dev in devices:
        device = torch.device(dev)
        m = DynamicsModel().to(device)
        m.load_state_dict(model.state_dict())
        state = torch.tensor([np.pi, 0.0], device=device)
        gen = torch.Generator(device=device).manual_seed(0)
        for _ in range(5):  # 预热
            random_shooting(m.predict, state, horizon, n_candidates, gen)
        n_runs = 5 if is_quick() else 50
        start = time.time()
        for _ in range(n_runs):
            random_shooting(m.predict, state, horizon, n_candidates, gen)
        results[dev] = (time.time() - start) / n_runs * 1000
        print(f"  {dev:<5} 一次规划（{n_candidates} 条候选 × {horizon} 步）：{results[dev]:6.1f} 毫秒，每秒能规划 {1000 / results[dev]:.0f} 次")

    print(f"""
环境的一步是 {DT * 1000:.0f} 毫秒，所以这个规划器在本机上能实时跑。
这个余量来自模型很小（两层 200 单元的 MLP）。模型一大，账就不好看了：
换成每次预测要几十毫秒的模型，一次规划就是几十秒。
这是"决策时规划"这条路线的主要限制，也是 Phase 6 的 Dreamer
改成"提前在模型里训练一个策略"的动机。
    """)
    return results


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：换一个奖励函数
    把 reward_fn 里的 0.1·θ̇² 去掉（不惩罚转速），用完美模型跑 MPC。
    摆还能稳在顶部吗？

练习 1 参考答案：
    能，而且没有变差。实测（5 个回合，仍按环境原本的奖励计分）：
    平均回报 -265（原来 -290），平均第 43 步稳住（原来第 51 步）。
    稳在顶部并不需要转速惩罚：转得快就会离开顶部，角度项自己就会扣分。
    去掉转速惩罚后，规划器在甩的阶段更放得开，反而早到了几步。
    （5 个回合的差距不大，只能说"没有变差"。）
    这个练习的重点是体会 MPC 的用法：想改变行为，改奖励函数就行，
    不用重新训练任何东西。这是决策时规划相对于"训练一个策略"的一大优势。

练习 2：动作序列的采样方式
    random_shooting() 里每一步的动作独立均匀采样。改成"每 5 步采一个动作，
    保持 5 步不变"（动作重复），其余不变。同样的 N 和 H，回报会怎么变？

练习 2 参考答案：
    变好。实测（完美模型，5 个回合）：平均回报从 -290 涨到 -247，照样能稳在顶部。
    独立采样的序列在时间上是白噪声，正负力矩互相抵消，很难攒出能量。
    保持几步不变的序列更"有主见"，而且 30 步只剩 6 个自由度，
    同样 1000 条候选能把这个小得多的空间覆盖得更好。
    这说明采样分布的设计很重要。下一节的 CEM 和 MPPI 走得更远：
    让采样分布自己朝着好序列的方向移动。

练习 3：闭环的作用
    在 run_mpc() 里给真实环境加扰动：每 50 步把角速度突然加 2。
    比较 replan_every = 1 和 replan_every = 30 的表现。

练习 3 参考答案：
    每步重新规划的控制器在下一步就看到了扰动后的状态，立刻修正。
    每 30 步规划一次的控制器要等到下一次规划才知道出了事，
    在这之前一直照着已经作废的计划执行。
    扰动越频繁，两者差距越大。真实机器人上扰动无处不在，所以重新规划的频率
    通常由"算得过来的最快速度"决定。

思考题 1：随机打靶在 30 维的空间里随机撒 1000 个点，为什么居然够用？

思考题 1 参考答案：
    因为 MPC 只执行第一个动作。它要的不是一条 30 步都恰到好处的序列，
    而是"第一个动作的方向对"。1000 条里总有一些序列的前几步碰巧不错。
    而且奖励是累加的，前几步的好坏对总分的影响和后面一样大。
    动作维度变高时这个运气就没了：7 个关节的机械臂，H = 30 就是 210 维，
    随机撒点基本撒不中，必须用下一节的方法。

思考题 2：规划视野之外的事，规划器完全看不见。有什么办法补救？

思考题 2 参考答案：
    在展开的末端加一个"终端价值"：用一个函数估计"从 ŝ_H 出发，
    之后还能拿多少奖励"，加到 J 上。这样短视野也能做出有远见的决定。
    这个函数就是 Phase 4 要学的价值函数。Phase 7 的 TD-MPC2 正是
    "短视野 MPPI + 学到的终端价值"。

思考题 3：MPC 每一步都要算几万次模型预测。如果模型是一个要跑几十步去噪的
    扩散模型（Phase 9），还能这么用吗？

思考题 3 参考答案：
    很难。单次预测的代价高出几个数量级，实时规划基本做不到。
    两条出路：一是提前在模型里训练一个策略，决策时只需一次前向计算
    （Phase 6 的 Dreamer）；二是换一个便宜得多的预测空间
    （Phase 10 的 DINO-WM 在预训练特征上规划）。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()

    if is_quick():
        n_episodes, epochs, seeds, n_candidates, horizon = 10, 5, [100], 100, 15
        horizons, candidate_counts, replan_intervals = [5, 15], [10, 100], [1, 15]
    else:
        n_episodes, epochs, seeds, n_candidates, horizon = 40, 100, list(range(100, 105)), 1000, 30
        horizons, candidate_counts, replan_intervals = [5, 10, 20, 30, 50], [10, 100, 1000], [1, 5, 15, 30]

    perfect_returns, n_stable = perfect_model_mpc(seeds, horizon, n_candidates)

    torch.manual_seed(SEED)
    model = train_model(DynamicsModel(), collect_random_dataset(n_episodes), epochs)
    learned, ms_per_step = learned_model_mpc(model, seeds, horizon, n_candidates, perfect_returns)
    sweep = horizon_sweep(model, seeds, n_candidates, horizons)
    n_results, k_results = candidates_and_replan(model, seeds, horizon, candidate_counts, replan_intervals)
    times = timing(model, horizon, n_candidates)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：06-cem-and-mppi.py（更聪明的采样：CEM 与 MPPI）")
    print("=" * 60)


if __name__ == "__main__":
    main()
