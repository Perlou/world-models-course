"""
蒙特卡洛与时序差分（TD）
========================

学习目标：
    1. 把模型拿走之后，会用两种办法从经验里估计 V^π：蒙特卡洛（等回合结束）和 TD(0)（走一步改一步）
    2. 认出两者共用同一个更新模板"新估计 ← 旧估计 + 步长 ×（目标 − 旧估计）"，区别只在目标
    3. 用数字看清偏差与方差：TD 目标有偏（偏差来自自举用的 V 不准）、方差小；蒙特卡洛目标无偏、方差大
    4. 看到 TD 和蒙特卡洛在同一批数据上学到收敛，答案也可以不同（TD 隐含着一个模型）
    5. 自举时正确处理 terminated 与 truncated

核心概念：
    - 蒙特卡洛（Monte Carlo，MC）：目标是这个回合实际拿到的回报 G_t
    - 时序差分（Temporal Difference，TD）：目标是 r_t + γ·V(s_{t+1})
    - 自举（bootstrapping）：用一个估计去修正另一个估计
    - TD 误差（TD error）：δ_t = r_t + γ·V(s_{t+1}) − V(s_t)

前置知识：
    - 01-returns-and-value-functions.py：回报的递推、按定义估计价值
    - 02-dynamic-programming.py：Bellman 期望方程、解析法策略评估（这里用来算标准答案）

对应概念文档：
    - CONCEPT.md 第 5 节「蒙特卡洛与时序差分（TD）」

预计运行时间：尚未实测（本文件是还没有运行过的草稿，定稿前要完整跑一遍并改写这一行）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-04")
SEED = 0
# 这一节全是表格运算，用 NumPy 在 CPU 上算


# ==================== 第一部分：两种办法 ====================


def introduction():
    """蒙特卡洛与 TD 的直觉、更新模板。"""
    print("=" * 60)
    print("第一部分：没有模型时，怎么从经验里估计价值")
    print("=" * 60)

    print("""
上一节的动态规划要用完整的模型（P 和 R 两张表）。现在把模型拿走：
智能体只能按策略 π 去环境里跑，拿到一条条经验 (s_t, r_t, s_{t+1})。怎么估计 V^π？

打个比方：估计一条 CI 流水线从各个阶段开始"还要多久跑完"。

    办法一（蒙特卡洛）：等整条流水线跑完，用实际的剩余时间去修正每个阶段的估计。
    办法二（时序差分）：编译实际花了 12 分钟，现在进入单测；"从单测开始还要多久"本来就有个估计
                      （比如 30 分钟），那么"从编译开始"的新证据就是 12 + 30 = 42 分钟，立刻就能改。

办法二用一个估计去修正另一个估计，这叫自举（bootstrapping）。

两种办法共用同一个更新模板，全章的算法都长这样：

    V(s_t)  ←  V(s_t)  +  α · ( 目标 − V(s_t) )

    ┌──────────┬─────────────────────────────┬────────────────────┐
    │ 方法      │ 目标                         │ 需要什么            │
    ├──────────┼─────────────────────────────┼────────────────────┤
    │ 动态规划  │ 对所有动作、所有 s' 求期望     │ 完整的模型          │
    │ 蒙特卡洛  │ G_t（实际拿到的回报）          │ 跑完整个回合        │
    │ TD(0)    │ r_t + γ·V(s_{t+1})           │ 一步转移            │
    └──────────┴─────────────────────────────┴────────────────────┘

                          不自举                    自举
                    ┌─────────────────────┬─────────────────────┐
   对下一步求期望     │   穷举搜索            │   动态规划            │
   （需要模型）       │                     │                     │
                    ├─────────────────────┼─────────────────────┤
   用采样的一条经验    │   蒙特卡洛            │   TD                 │
   （不需要模型）      │                     │                     │
                    └─────────────────────┴─────────────────────┘

TD 从蒙特卡洛借了"采样"（所以不需要模型），从动态规划借了"自举"（所以不用等回合结束）。
    """)


# ==================== 第二部分：两个算法 ====================
#
# 一个回合的记录写成三元组 (states, rewards, terminated)：
#     states:     长度 T+1，s_0 … s_T（最后一个是回合结束时所在的状态）
#     rewards:    长度 T，r_0 … r_{T-1}
#     terminated: 最后一步之后环境是否真的终止（False 表示被步数上限掐断）


def mc_prediction(episodes, n_states, alpha, gamma, v_true=None):
    """蒙特卡洛策略评估（每次访问都更新，CONCEPT.md 第 5.7 节）。

    回合结束后从后往前算回报 G_t，把经过的每个状态的价值往 G_t 挪一步。

    返回：
        V:      [S]
        errors: [回合数]，每个回合之后 V 与 v_true 的均方根误差（没给 v_true 时为空）
    """
    V = np.zeros(n_states)
    errors = []
    for states, rewards, _ in episodes:
        G = 0.0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            V[states[t]] += alpha * (G - V[states[t]])  # 目标是 G_t
        if v_true is not None:
            errors.append(np.sqrt(np.mean((V - v_true) ** 2)))
    return V, np.array(errors)


def td0_prediction(episodes, n_states, alpha, gamma, v_true=None, truncation_as_termination=False):
    """TD(0) 策略评估（CONCEPT.md 第 5.7、5.8 节）。

    每走一步就更新一次：target = r + γ·(1 − d)·V[s']，d 只在 terminated 时为 1。
    truncation_as_termination=True 是故意写错的版本：把被掐断也当成终止（第六部分用来对比）。
    """
    V = np.zeros(n_states)
    errors = []
    for states, rewards, terminated in episodes:
        for t in range(len(rewards)):
            is_last = t == len(rewards) - 1
            done = is_last and (terminated or truncation_as_termination)
            bootstrap = 0.0 if done else V[states[t + 1]]  # 终止状态的价值按定义是 0
            target = rewards[t] + gamma * bootstrap  # 目标是 r_t + γ V(s_{t+1})
            V[states[t]] += alpha * (target - V[states[t]])
        if v_true is not None:
            errors.append(np.sqrt(np.mean((V - v_true) ** 2)))
    return V, np.array(errors)


def td_errors(V, states, rewards, terminated, gamma):
    """一个回合里每一步的 TD 误差 δ_t = r_t + γ V(s_{t+1}) − V(s_t)，V 在回合内不变。返回 [T]。"""
    T = len(rewards)
    deltas = np.zeros(T)
    for t in range(T):
        done = terminated and t == T - 1
        bootstrap = 0.0 if done else V[states[t + 1]]
        deltas[t] = rewards[t] + gamma * bootstrap - V[states[t]]
    return deltas


def by_hand():
    """复现 CONCEPT.md 第 5.5 节的手算例子。"""
    print("\n" + "=" * 60)
    print("第二部分：在三格走廊上手算一遍")
    print("=" * 60)

    gamma, alpha = 0.9, 0.5
    # 走廊，策略"一直向右"：A → B → C → T，奖励 -1, -1, +10。状态 0、1、2 = A、B、C，3 = 充电桩
    episode = ([0, 1, 2, 3], [-1.0, -1.0, 10.0], True)
    v_true = np.array([6.2, 8.0, 10.0])

    mc1, _ = mc_prediction([episode], 3, alpha, gamma)
    mc2, _ = mc_prediction([episode] * 2, 3, alpha, gamma)
    td1, _ = td0_prediction([episode], 3, alpha, gamma)
    td2, _ = td0_prediction([episode] * 2, 3, alpha, gamma)
    assert np.allclose(mc1, [3.1, 4.0, 5.0]) and np.allclose(mc2, [4.65, 6.0, 7.5])
    assert np.allclose(td1, [-0.5, -0.5, 5.0]) and np.allclose(td2, [-0.975, 1.5, 7.5])

    print(f"""
策略"一直向右"，真实价值 (6.2, 8, 10)。初始估计全 0，步长 α = {alpha}，γ = {gamma}。
把同一个回合 A → B → C → T（奖励 -1, -1, +10）喂给两种方法：

    {'':<22}{'V(A)':>8}{'V(B)':>8}{'V(C)':>8}
    {'真实价值':<18}{v_true[0]:>8.3f}{v_true[1]:>8.3f}{v_true[2]:>8.3f}
    {'蒙特卡洛，1 个回合后':<14}{mc1[0]:>8.3f}{mc1[1]:>8.3f}{mc1[2]:>8.3f}
    {'蒙特卡洛，2 个回合后':<14}{mc2[0]:>8.3f}{mc2[1]:>8.3f}{mc2[2]:>8.3f}
    {'TD(0)，1 个回合后':<16}{td1[0]:>8.3f}{td1[1]:>8.3f}{td1[2]:>8.3f}
    {'TD(0)，2 个回合后':<16}{td2[0]:>8.3f}{td2[1]:>8.3f}{td2[2]:>8.3f}

  和 CONCEPT.md 第 5.5 节的表逐格一致。两种方法在 C 上完全一样（C 的下一步就是终止，
  TD 目标和回报是同一个数）。差别在 A 和 B：
    - 蒙特卡洛在第一个回合就把 +10 的信息传到了所有状态。
    - TD(0) 第一个回合只有 C 知道 +10，A 和 B 只看到眼前的 -1；第二个回合 B 才从 C 那里"听说"。
  这和上一节值迭代"每轮往回传一格"是同一个现象。
    """)

    # 蒙特卡洛误差 = TD 误差的折扣和（第 5.3 节（四）），任取一个 V 都成立
    rng = np.random.default_rng(SEED)
    V = rng.normal(size=3)
    states, rewards, terminated = episode
    deltas = td_errors(V, states, rewards, terminated, gamma)
    lhs = sum(gamma**k * rewards[k] for k in range(3)) - V[0]  # G_0 − V(A)
    rhs = sum(gamma**k * deltas[k] for k in range(3))
    assert np.isclose(lhs, rhs)
    print(f"  两种目标之间有一个精确的关系：G_t − V(s_t) = δ_t + γ δ_{{t+1}} + γ² δ_{{t+2}} + …")
    print(f"  随便取一组 V = ({V[0]:.3f}, {V[1]:.3f}, {V[2]:.3f}) 验证：")
    print(f"      G_0 − V(A) = {lhs:.4f}        δ_0 + γ δ_1 + γ² δ_2 = {rhs:.4f}")
    print("  蒙特卡洛一次性用掉了之后所有步的 TD 误差，TD(0) 只用第一个。")
    print("  这个恒等式下一节会升级成 λ-return 的版本，到 Phase 5 就是 GAE 的公式。")


# ==================== 第三部分：随机游走上的对比 ====================


class RandomWalk:
    """随机游走：n 个状态排成一排，从正中间出发，每一步等概率向左或向右走一格。

        [左出口] ◀─ 0 ─ 1 ─ … ─ n-1 ─▶ [右出口]
         奖励 -1                          奖励 +1         其余每一步奖励 0，不折扣（γ = 1）

    没有动作可选（或者说策略固定为"左右各一半"），是一个纯粹的预测问题。
    回合记录里用编号 n 表示终止。
    """

    def __init__(self, n_states=5, left_reward=-1.0, right_reward=1.0):
        self.n_states = n_states
        self.start = n_states // 2
        self.left_reward, self.right_reward = left_reward, right_reward

    def sample_episode(self, rng):
        """跑一个回合，返回 (states, rewards, terminated)。"""
        s = self.start
        states, rewards = [s], []
        while True:
            s = s + (1 if rng.random() < 0.5 else -1)
            if s < 0 or s >= self.n_states:
                rewards.append(self.left_reward if s < 0 else self.right_reward)
                states.append(self.n_states)
                return states, rewards, True
            rewards.append(0.0)
            states.append(s)

    def transition_matrix(self):
        """策略下的状态转移矩阵 P_π [S, S] 和期望奖励 r_π [S]（走出两端的概率不计入矩阵）。"""
        n = self.n_states
        P = np.zeros((n, n))
        r = np.zeros(n)
        for s in range(n):
            if s > 0:
                P[s, s - 1] = 0.5
            if s < n - 1:
                P[s, s + 1] = 0.5
        r[0] += 0.5 * self.left_reward
        r[-1] += 0.5 * self.right_reward
        return P, r

    def true_values(self, gamma=1.0):
        """标准答案：上一节的解析法，解 (I − γ P_π) v = r_π。"""
        P, r = self.transition_matrix()
        return np.linalg.solve(np.eye(self.n_states) - gamma * P, r)


def error_curve_experiment(n_runs, n_episodes):
    """同一批经验上，蒙特卡洛与 TD(0) 在不同步长下的误差曲线。"""
    print("\n" + "=" * 60)
    print("第三部分：随机游走——同一批经验，两种方法谁学得快")
    print("=" * 60)

    env = RandomWalk(5)
    v_true = env.true_values()
    alphas = [0.02, 0.05, 0.1, 0.2, 0.4]
    initial_error = np.sqrt(np.mean(v_true**2))
    print(f"""
手算例子里没有任何随机性，TD(0) 显得很吃亏。换一个有随机性的环境：5 个状态的随机游走。

        [左出口] ◀─ 0 ─ 1 ─ 2 ─ 3 ─ 4 ─▶ [右出口]          从状态 2 出发，每一步左右各一半
         奖励 -1                           奖励 +1          其余奖励 0，γ = 1

  标准答案（解析法）：V^π = ({', '.join(f'{v:.3f}' for v in v_true)})
  实验：每次运行生成 {n_episodes} 个回合，把同一批回合分别喂给两种方法、{len(alphas)} 种步长；
        重复 {n_runs} 次运行取平均。误差 = 5 个状态上的均方根误差，初始（V 全 0）是 {initial_error:.3f}。
    """)

    curves = {"MC": np.zeros((len(alphas), n_episodes)), "TD": np.zeros((len(alphas), n_episodes))}
    for run in range(n_runs):
        rng = np.random.default_rng(SEED + run)
        episodes = [env.sample_episode(rng) for _ in range(n_episodes)]
        for i, alpha in enumerate(alphas):
            curves["MC"][i] += mc_prediction(episodes, 5, alpha, 1.0, v_true)[1] / n_runs
            curves["TD"][i] += td0_prediction(episodes, 5, alpha, 1.0, v_true)[1] / n_runs

    head, tail = min(10, n_episodes), min(10, n_episodes)
    print(f"  {'步长 α':<8}{f'MC：前 {head} 回合':>14}{f'MC：最后 {tail} 回合':>16}{'MC：全程平均':>14}"
          f"{f'TD：前 {head} 回合':>16}{f'TD：最后 {tail} 回合':>16}{'TD：全程平均':>14}")
    for i, alpha in enumerate(alphas):
        mc, td = curves["MC"][i], curves["TD"][i]
        print(f"  {alpha:<10}{mc[:head].mean():>14.3f}{mc[-tail:].mean():>16.3f}{mc.mean():>14.3f}"
              f"{td[:head].mean():>16.3f}{td[-tail:].mean():>16.3f}{td.mean():>14.3f}")

    # 结论都从数据里读，不预设谁赢
    area = {m: curves[m].mean(axis=1) for m in curves}  # 每个步长的全程平均误差
    final = {m: curves[m][:, -tail:].mean(axis=1) for m in curves}
    early = {m: curves[m][:, :head].mean(axis=1) for m in curves}
    best = {m: int(np.argmin(area[m])) for m in curves}
    print(f"\n  按全程平均误差挑各自最好的步长：MC 是 α = {alphas[best['MC']]}（{area['MC'][best['MC']]:.3f}），"
          f"TD(0) 是 α = {alphas[best['TD']]}（{area['TD'][best['TD']]:.3f}）。")
    winner = "TD(0)" if area["TD"][best["TD"]] < area["MC"][best["MC"]] else "蒙特卡洛"
    print(f"  各自用最好的步长时，全程平均误差更低的是 {winner}。")
    e_mc, e_td = early["MC"][best["MC"]], early["TD"][best["TD"]]
    print(f"  只看前 {head} 个回合：MC {e_mc:.3f}，TD(0) {e_td:.3f}，"
          f"{'TD(0) 前期下降更快' if e_td < e_mc else '蒙特卡洛前期下降更快'}。")
    big = len(alphas) - 1
    for m, name in (("MC", "蒙特卡洛"), ("TD", "TD(0)")):
        ratio = final[m][big] / final[m].min()
        print(f"  步长取到最大的 α = {alphas[big]} 时，{name} 最后 {tail} 个回合的平均误差是 {final[m][big]:.3f}，"
              f"是它在各步长里最低值（{final[m].min():.3f}）的 {ratio:.1f} 倍。")
    if final["MC"][big] > final["MC"].min() * 1.2 and final["TD"][big] > final["TD"].min() * 1.2:
        print("  步长太大时，两种方法都学得快但停不稳：每个新样本都把估计拽走一大截，误差停在一个较高的水平上波动。")
    print("  固定步长下误差不会降到 0：估计一直在跟着最新的样本晃，步长越小晃得越轻，但学得越慢。")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(alphas)))
    for i, alpha in enumerate(alphas):
        ax.plot(np.arange(1, n_episodes + 1), curves["TD"][i], color=colors[i], linewidth=2, label=f"TD(0)，α = {alpha}")
        ax.plot(np.arange(1, n_episodes + 1), curves["MC"][i], color=colors[i], linewidth=1.5, linestyle="--",
                label=f"MC，α = {alpha}")
    ax.set_xlabel("回合数")
    ax.set_ylabel(f"均方根误差（{n_runs} 次运行的平均）")
    ax.set_title("5 状态随机游走：蒙特卡洛（虚线）与 TD(0)（实线）")
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, OUT / "03-mc-vs-td-error-curves.png")


# ==================== 第四部分：偏差与方差 ====================


def bias_and_variance(n_episodes):
    """两种目标的偏差和方差，各用数字验证一次。"""
    print("\n" + "=" * 60)
    print("第四部分：偏差与方差从哪来")
    print("=" * 60)

    # ---- 偏差：TD 目标的期望 − 真值 = γ × 下一状态估计的平均误差 ----
    env = RandomWalk(5)
    P, r = env.transition_matrix()
    v_true = env.true_values()
    rng = np.random.default_rng(SEED)
    V = rng.uniform(-1, 1, size=5)  # 一组不准的估计
    td_target_mean = r + P @ V  # 每个状态上 TD 目标的期望（γ = 1），这里借模型精确算出来
    bias = td_target_mean - v_true
    predicted = P @ (V - v_true)  # CONCEPT.md 第 5.4 节的公式：γ·E[V(s') − V^π(s')]
    assert np.allclose(bias, predicted)
    print("""
偏差。蒙特卡洛目标 G_t 的期望按定义就是 V^π(s_t)，无偏。
TD 目标 r_t + γ V(s_{t+1}) 用了自己的估计 V，V 不准时目标的期望也跟着偏：

    E[TD 目标] − V^π(s) = γ · E[ V(s') − V^π(s') ]        偏差 = γ × 当前估计在下一个状态上的平均误差

在 5 状态随机游走上取一组不准的 V 验证（期望借模型精确算出，只为核对公式）：
    """)
    print(f"  {'状态':<6}{'当前估计 V':>12}{'真值 V^π':>12}{'TD 目标的期望':>14}{'偏差':>10}{'公式预测的偏差':>16}")
    for s in range(5):
        print(f"  {s:<8}{V[s]:>12.3f}{v_true[s]:>12.3f}{td_target_mean[s]:>14.3f}{bias[s]:>10.3f}{predicted[s]:>16.3f}")
    print("""
  最后两列逐行相等。偏差完全来自自举用的 V 不准：V 准了，偏差就没了。
  所以有偏的是训练过程中每一步的目标，不是最终的结果。
    """)

    # ---- 方差：从起点出发，两种目标各自有多分散 ----
    print("""方差。蒙特卡洛目标累积了整条轨迹上的随机性，TD 目标只含一步的随机性。
实测：把 V 取成真值（这样两种目标都无偏，只比方差），从起点出发采样很多个回合，
记录两种目标的取值。走廊越长，回合越长。
    """)
    sizes = [5, 11, 19, 31]
    rows = []
    print(f"  {'状态数 n':<10}{'平均回合长度':>12}{'MC 目标的方差':>14}{'TD 目标的方差':>14}{'方差之比':>10}{'理论值 (n+1)²/4':>18}")
    for n in sizes:
        walk = RandomWalk(n)
        v_n = walk.true_values()
        mc_targets, td_targets, lengths = [], [], []
        for _ in range(n_episodes):
            states, rewards, _ = walk.sample_episode(rng)
            mc_targets.append(sum(rewards))  # γ = 1，回报就是奖励之和
            next_value = v_n[states[1]] if states[1] < n else 0.0
            td_targets.append(rewards[0] + next_value)
            lengths.append(len(rewards))
        var_mc, var_td = np.var(mc_targets), np.var(td_targets)
        rows.append((n, np.mean(lengths), var_mc, var_td, var_mc / var_td))
        print(f"  {n:<12}{np.mean(lengths):>12.1f}{var_mc:>14.4f}{var_td:>14.4f}{var_mc / var_td:>10.1f}{(n + 1) ** 2 / 4:>18.1f}")

    ratios = [row[4] for row in rows]
    print(f"\n  这个环境里两个方差都能直接算：从正中间出发，回报不是 -1 就是 +1（各一半），方差是 1；")
    print(f"  TD 目标是相邻状态的真值 ±2/(n+1)，方差是 4/(n+1)²。上表最后一列就是两者之比。")
    if all(ratio > 1 for ratio in ratios) and np.all(np.diff(ratios) > 0):
        print(f"  实测：蒙特卡洛目标的方差在每一行都更大，而且回合越长差得越多"
              f"（n = {sizes[0]} 时 {ratios[0]:.0f} 倍，n = {sizes[-1]} 时 {ratios[-1]:.0f} 倍）。")
    else:
        print(f"  实测的方差之比依次是 {'、'.join(f'{x:.1f}' for x in ratios)}，没有表现出随回合变长而单调增大，见上表。")
    print("  方差大意味着要很多样本才能平均掉噪声。这就是 TD 在长回合、有随机性的任务上通常学得更快的原因。")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([row[1] for row in rows], ratios, "o-", linewidth=2, label="实测")
    ax.plot([row[1] for row in rows], [(n + 1) ** 2 / 4 for n in sizes], "k--", linewidth=1, label="理论值 (n+1)²/4")
    for row in rows:
        ax.annotate(f"n = {row[0]}", (row[1], row[4]), textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("平均回合长度（步，对数坐标）")
    ax.set_ylabel("MC 目标的方差 / TD 目标的方差")
    ax.set_title("回合越长，蒙特卡洛目标相对越吵")
    ax.legend()
    save_figure(fig, OUT / "03-target-variance.png")


# ==================== 第五部分：TD 隐含着一个模型 ====================


def batch_fit(episodes, n_states, method, gamma=1.0, alpha=0.01, n_sweeps=5000):
    """批量更新：把同一批数据反复扫，每一遍先用固定的 V 算出所有更新量，加总后再一起改 V，直到收敛。"""
    V = np.zeros(n_states)
    for _ in range(n_sweeps):
        increment = np.zeros(n_states)
        for states, rewards, terminated in episodes:
            if method == "mc":
                G = 0.0
                for t in reversed(range(len(rewards))):
                    G = rewards[t] + gamma * G
                    increment[states[t]] += G - V[states[t]]
            else:
                deltas = td_errors(V, states, rewards, terminated, gamma)
                for t in range(len(rewards)):
                    increment[states[t]] += deltas[t]
        V += alpha * increment
    return V


def implicit_model_demo():
    """Sutton & Barto 例 6.4：同一批数据学到收敛，TD 和蒙特卡洛的答案不同。"""
    print("\n" + "=" * 60)
    print("第五部分：同一批数据，两个答案")
    print("=" * 60)

    A, B, END = 0, 1, 2
    episodes = [([A, B, END], [0.0, 0.0], True)]
    episodes += [([B, END], [1.0], True)] * 6
    episodes += [([B, END], [0.0], True)]
    v_mc = batch_fit(episodes, 2, "mc")
    v_td = batch_fit(episodes, 2, "td")
    assert np.allclose(v_mc, [0.0, 0.75], atol=1e-3) and np.allclose(v_td, [0.75, 0.75], atol=1e-3)

    print(f"""
只有两个状态 A 和 B，不折扣。一共观察到 8 个回合（Sutton & Barto 例 6.4）：

    回合 1：    A ──(奖励 0)──▶ B ──(奖励 0)──▶ 结束
    回合 2–7：  B ──(奖励 1)──▶ 结束                      （共 6 个）
    回合 8：    B ──(奖励 0)──▶ 结束

把这 8 个回合反复学到收敛（batch_fit）：

    {'方法':<12}{'V(A)':>8}{'V(B)':>8}
    {'蒙特卡洛':<10}{v_mc[A]:>8.3f}{v_mc[B]:>8.3f}        A 只出现过一次，那次的回报是 0
    {'TD(0)':<12}{v_td[A]:>8.3f}{v_td[B]:>8.3f}        A 之后总是到 B、奖励 0，所以 V(A) = 0 + V(B)

  V(B) = 6/8 = 0.75 没有争议。V(A) 两种方法给出不同的答案，各有道理：
    - 蒙特卡洛的答案在这批数据上的平方误差最小。
    - TD 的答案相当于先从数据里估计出一个模型（"A 以概率 1 转到 B，B 以 0.75 的概率拿到 1"），
      再在这个模型上精确求解。A 只出现过一次，但 B 出现过 8 次，TD 让 A 借用了 B 的统计量。
  如果环境满足马尔可夫性，TD 的答案对未来的数据更准。

  TD 学习在不显式写出模型的情况下，利用了和模型相同的结构。
  第 08 课的 Dyna 把这件事做成显式的：真的学一个模型，再用它生成转移来做 TD 更新。
    """)


# ==================== 第六部分：终止与截断 ====================


def truncation_demo(n_episodes):
    """自举时把 truncated 当成 terminated 会怎样（CONCEPT.md 第 5.8 节）。"""
    print("\n" + "=" * 60)
    print("第六部分：回合被掐断时，TD 目标怎么写")
    print("=" * 60)

    gamma, alpha, limit, n = 0.9, 0.05, 10, 5
    # 一个永远不会终止的任务：在 5 个格子里随机走（撞墙留在原地），每一步奖励 +1。
    # 每个回合从中间出发，走满 limit 步被掐断。真实价值处处是 1/(1-γ)。
    rng = np.random.default_rng(SEED)
    episodes = []
    for _ in range(n_episodes):
        s = n // 2
        states = [s]
        for _ in range(limit):
            s = int(np.clip(s + (1 if rng.random() < 0.5 else -1), 0, n - 1))
            states.append(s)
        episodes.append((states, [1.0] * limit, False))  # terminated = False：是被掐断的
    v_true = np.full(n, 1.0 / (1.0 - gamma))
    v_right, _ = td0_prediction(episodes, n, alpha, gamma)
    v_wrong, _ = td0_prediction(episodes, n, alpha, gamma, truncation_as_termination=True)

    print(f"""
TD 目标里的 V(s') 在回合结束处要小心。记 d 为"走完这一步后环境是否 terminated"：

    target = r + γ · (1 − d) · V(s')

    这一步之后        d      目标               理由
    回合继续          0      r + γ V(s')       正常情况
    terminated       1      r                 终止状态之后没有奖励
    truncated        0      r + γ V(s')       环境还能继续，未来的奖励还在，照常自举

实验：一个永远不会终止的任务（在 {n} 个格子里随机走，每一步奖励 +1，γ = {gamma}），
真实价值处处是 1/(1-γ) = {v_true[0]:.0f}。每个回合走满 {limit} 步被掐断，一共 {n_episodes} 个回合。

    {'做法':<34}{'学到的 V（5 个状态）':<40}{'平均':>6}
    {'只有 terminated 才不自举（正确）':<26}{'  '.join(f'{v:6.2f}' for v in v_right):<44}{v_right.mean():>6.2f}
    {'把 truncated 也当成终止（错误）':<26}{'  '.join(f'{v:6.2f}' for v in v_wrong):<44}{v_wrong.mean():>6.2f}
    """)
    if v_wrong.mean() < v_right.mean() - 0.5 and abs(v_right.mean() - v_true[0]) < abs(v_wrong.mean() - v_true[0]):
        print(f"  正确的写法学到的价值接近真值 {v_true[0]:.0f}；把掐断当成终止，价值被系统性地压低到平均 {v_wrong.mean():.2f}。")
        print("  原因：状态里没有'现在是第几步'这个信息，同一个格子有时被标成'后面还有奖励'，")
        print("  有时（恰好是最后一步时）被标成'后面是 0'，学到的是两者的混合。")
    else:
        print(f"  这次运行里两种写法的平均价值分别是 {v_right.mean():.2f} 和 {v_wrong.mean():.2f}（真值 {v_true[0]:.0f}），见上表。")
    print("  写成代码只有一条规则：done 标志里只放 terminated，不要放 terminated or truncated。")
    print("  第 06 课的 DQN 会遇到同样的事：CartPole-v1 撑满 500 步是 truncated，杆倒了才是 terminated。")


# ==================== 第七部分：练习与思考 ====================


def exercises():
    """练习题与参考答案。答案里的数字都是下面的代码现算的。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    gamma, alpha = 0.9, 0.5
    episode = ([0, 1, 2, 3], [-1.0, -1.0, 10.0], True)
    # 练习 1：TD(0) 的第三个回合
    v2, _ = td0_prediction([episode] * 2, 3, alpha, gamma)
    delta0 = td_errors(v2, *episode, gamma)[0]
    v3, _ = td0_prediction([episode] * 3, 3, alpha, gamma)
    assert np.isclose(delta0, 1.325)
    # 练习 2：各自要多少个相同的回合，V(A) 才进入真值 6.2 的 ±0.1 以内
    needed = {}
    for name, fn in (("mc", mc_prediction), ("td", td0_prediction)):
        for k in range(1, 200):
            if abs(fn([episode] * k, 3, alpha, gamma)[0][0] - 6.2) < 0.1:
                needed[name] = k
                break
    # 练习 3：步长取 1/n 的蒙特卡洛就是算术平均
    rng = np.random.default_rng(SEED + 7)
    samples = rng.normal(loc=3.0, scale=2.0, size=50)
    v_inc = 0.0
    for i, g in enumerate(samples, start=1):
        v_inc += (g - v_inc) / i
    assert np.isclose(v_inc, samples.mean())

    print(f"""
练习 1：TD(0) 的第三个回合
    接着第二部分的手算：两个回合后 V = (-0.975, 1.5, 7.5)。再来一个相同的回合，
    A → B 这一步的 TD 误差 δ_0 是多少？这个回合结束后 V 是多少？

练习 1 参考答案：
    δ_0 = -1 + 0.9 × 1.5 − (−0.975) = {delta0:.3f}，第一次是正的：V(A) 终于开始往上走。
    程序算出来的三个回合后的 V = ({v3[0]:.4f}, {v3[1]:.4f}, {v3[2]:.4f})。
    +10 的信息第一个回合到 C，第二个回合到 B，第三个回合才到 A。

练习 2：要多少个回合
    同一个回合反复喂（α = 0.5），蒙特卡洛和 TD(0) 各需要多少个回合，V(A) 才进入 6.2 ± 0.1？

练习 2 参考答案：
    程序算出来的：蒙特卡洛 {needed.get('mc', '超过 200')} 个，TD(0) {needed.get('td', '超过 200')} 个。
    这个环境没有任何随机性，蒙特卡洛的目标就是真值，每次更新把误差减半；
    TD(0) 的目标要等后面的状态先变准。没有噪声时 TD 没有优势，它的优势来自方差小（第四部分）。

练习 3：步长 1/n 就是取平均
    把 mc_prediction 里的固定步长 α 换成 1/n（n 是这个状态被访问的次数），得到的是什么？

练习 3 参考答案：
    就是到目前为止所有回报的算术平均：V_n = V_(n-1) + (G_n − V_(n-1))/n。
    程序用 50 个随机数验证：增量算出来的是 {v_inc:.6f}，直接取平均是 {samples.mean():.6f}。
    固定步长则是指数滑动平均，越新的样本权重越大。强化学习里基本都用固定步长，
    因为目标本身在变（策略在改进，自举用的估计也在变），旧样本应该被逐渐遗忘。

思考题 1：TD 误差很小，能说明价值函数学准了吗？

思考题 1 参考答案：
    不能。TD 误差是拿自己的估计当参照算出来的。V 恒等于 0、奖励也恒为 0 时 TD 误差是 0；
    V 整体偏高但相邻状态之间自洽时，TD 误差同样很小。要判断 V 准不准，得拿真实的回报来对照
    （第 07 课测量 DQN 的过估计时就是这么做的）。

思考题 2：蒙特卡洛不依赖马尔可夫性，TD 依赖。什么情况下这个区别要紧？

思考题 2 参考答案：
    状态不完整的时候。比如只给一帧图像、看不出速度，那么"从这个观测出发之后会怎样"
    还取决于之前发生了什么，Bellman 方程在观测上不成立，TD 的自举会引入去不掉的偏差。
    蒙特卡洛只是对实际回报取平均，不受影响。
    Phase 3 的 RSSM 用循环状态把历史记进状态里，目的之一就是让状态重新满足马尔可夫性。

思考题 3：想要蒙特卡洛的"传得快"，又想要 TD 的"方差小"，有没有折中？

思考题 3 参考答案：
    有：前 n 步用真实奖励，之后用 V 补上（n 步回报）；再把不同 n 的结果加权平均（λ-return）。
    下一节的内容，也是 Dreamer 训练价值函数用的目标。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()
    by_hand()

    n_runs, n_episodes = (10, 50) if is_quick() else (100, 100)
    error_curve_experiment(n_runs, n_episodes)
    bias_and_variance(n_episodes=300 if is_quick() else 5000)
    implicit_model_demo()
    truncation_demo(n_episodes=300 if is_quick() else 3000)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：04-n-step-and-lambda-returns.py（n 步回报与 λ-return）")
    print("=" * 60)


if __name__ == "__main__":
    main()
