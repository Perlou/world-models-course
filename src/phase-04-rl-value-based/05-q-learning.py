"""
表格 Q-learning 与 SARSA
========================

学习目标：
    1. 从"预测"转到"控制"：没有模型时为什么要学 Q 而不是 V，为什么必须探索
    2. 实现 ε-greedy、SARSA 和 Q-learning，看清两者只差目标里的一项
    3. 说清在策略（on-policy）与离策略（off-policy）的区别，以及 Q-learning 为什么可以用任何策略采的数据
    4. 在 CliffWalking-v1 上复现经典现象：训练期间 SARSA 回报更高，关掉探索后 Q-learning 的路更短

核心概念：
    - 探索（exploration）与 ε-greedy 策略
    - 行为策略（behavior policy）与目标策略（target policy）
    - SARSA：目标用下一步实际选的动作，学的是"带着探索行动"的价值
    - Q-learning：目标用下一步最好的动作，学的是最优策略的价值

前置知识：
    - 03-monte-carlo-vs-td.py：TD(0)、TD 目标、终止时不自举
    - 02-dynamic-programming.py：Bellman 最优方程、广义策略迭代

对应概念文档：
    - CONCEPT.md 第 7 节「Q-learning 与离策略学习」

预计运行时间：尚未实测（本文件是还没有运行过的草稿，定稿前要完整跑一遍并改写这一行）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-04")
SEED = 0
LEFT, RIGHT = 0, 1
# CliffWalking-v1 的动作编号：0 上、1 右、2 下、3 左；状态编号 = 行 × 12 + 列
CLIFF_ARROWS = ["↑", "→", "↓", "←"]
CLIFF_ROWS, CLIFF_COLS = 4, 12
# 这一节全是表格运算，用 NumPy 在 CPU 上算


# ==================== 第一部分：从预测到控制 ====================


def introduction():
    """控制问题的三个新麻烦，以及两个算法。"""
    print("=" * 60)
    print("第一部分：从预测到控制")
    print("=" * 60)

    print("""
前两节解决的是预测：给定一个策略，估计它的价值。现在转向控制：找到一个好策略。
框架还是第 02 课的广义策略迭代（评估、改进、交替进行），但没有模型时有三个新问题：

    问题一：只有 V 没法改进策略。对 V 贪心要做一步前瞻，一步前瞻要用模型。
            → 直接学 Q(s, a)，贪心只是一次 argmax。
    问题二：不试就不知道。总选当前看起来最好的动作，别的动作永远没有数据。
            → 探索：以 ε 的概率随机选一个动作，其余时候选 Q 最大的（ε-greedy）。
    问题三：为了探索，实际行为里掺了随机动作；但最终想要的是不掺随机的那个策略的价值。
            → 能不能用一个策略的数据学另一个策略的价值？能，这叫离策略（off-policy）学习。

两个算法都是 TD(0) 的 Q 版本，只差目标里的一项：

    SARSA：       Q(s, a) ← Q(s, a) + α·[ r + γ·Q(s', a')          − Q(s, a) ]
                                               └ a' 是行为策略下一步实际选的动作

    Q-learning：  Q(s, a) ← Q(s, a) + α·[ r + γ·max_a' Q(s', a')   − Q(s, a) ]
                                               └ 当前 Q 认为最好的动作，不管实际选了什么

    ┌──────────────┬──────────────────────────────┬──────────────────────────────┐
    │              │ SARSA                        │ Q-learning                   │
    ├──────────────┼──────────────────────────────┼──────────────────────────────┤
    │ 需要的数据    │ (s, a, r, s', a')            │ (s, a, r, s')                │
    │ 在逼近的方程  │ Q^π 的 Bellman 期望方程       │ Q* 的 Bellman 最优方程        │
    │ 类型         │ 在策略（on-policy）           │ 离策略（off-policy）          │
    │ 学到的是     │ "带着探索行动"时的价值         │ 最优策略的价值                │
    └──────────────┴──────────────────────────────┴──────────────────────────────┘
    """)


# ==================== 第二部分：三个函数 ====================


def epsilon_greedy(q_values, epsilon, rng):
    """ε-greedy：以 ε 的概率均匀随机选一个动作，否则选 Q 最大的（并列时随机挑一个）。

    参数：
        q_values: [A]，一个状态下各动作的价值
    """
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    best = np.flatnonzero(q_values == q_values.max())
    return int(best[rng.integers(len(best))])


def td_control(env, method, n_states, n_actions, n_episodes, alpha, gamma, epsilon, rng, max_steps=10000):
    """SARSA 和 Q-learning 共用的训练循环，只有算自举项的那一行不同（CONCEPT.md 第 7.6 节）。

    env 用 Gymnasium 的接口：reset() → (s, info)，step(a) → (s', r, terminated, truncated, info)。
    返回：
        Q:       [S, A]
        returns: [n_episodes]，训练期间每个回合的奖励之和（不折扣，包含探索造成的损失）
    """
    Q = np.zeros((n_states, n_actions))
    returns = np.zeros(n_episodes)
    for ep in range(n_episodes):
        s, _ = env.reset()
        a = epsilon_greedy(Q[s], epsilon, rng)
        for _ in range(max_steps):
            s2, r, terminated, truncated, _ = env.step(a)
            returns[ep] += r
            if method == "sarsa":
                a2 = epsilon_greedy(Q[s2], epsilon, rng)  # 先选好下一步的动作，目标里用的就是它
                bootstrap = Q[s2, a2]
            else:
                bootstrap = Q[s2].max()  # 目标里用最好的动作，不管下一步实际选什么
            target = r + gamma * (1.0 - terminated) * bootstrap  # 终止之后没有奖励，不自举
            Q[s, a] += alpha * (target - Q[s, a])
            if method != "sarsa":
                a2 = epsilon_greedy(Q[s2], epsilon, rng)  # Q-learning：更新完再选下一步的动作
            s, a = s2, a2
            if terminated or truncated:
                break
    return Q, returns


def sarsa(env, n_states, n_actions, n_episodes, alpha, gamma, epsilon, rng):
    """SARSA（在策略）。"""
    return td_control(env, "sarsa", n_states, n_actions, n_episodes, alpha, gamma, epsilon, rng)


def q_learning(env, n_states, n_actions, n_episodes, alpha, gamma, epsilon, rng):
    """Q-learning（离策略）。"""
    return td_control(env, "q_learning", n_states, n_actions, n_episodes, alpha, gamma, epsilon, rng)


def epsilon_greedy_check():
    """核对 ε-greedy 下各动作被选中的概率。"""
    print("\n" + "=" * 60)
    print("第二部分：ε-greedy")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    epsilon, n = 0.1, 100000
    q_values = np.array([0.3, 1.2, -0.5, 0.0])  # 贪心动作是 1
    counts = np.bincount([epsilon_greedy(q_values, epsilon, rng) for _ in range(n)], minlength=4)
    p_greedy = 1 - epsilon + epsilon / 4
    assert abs(counts[1] / n - p_greedy) < 0.01
    print(f"""
ε = {epsilon}、4 个动作时，随机的那部分也可能抽到贪心动作，所以

    P(贪心动作)   = 1 − ε + ε/4 = {p_greedy:.3f}
    P(其他每个动作) = ε/4         = {epsilon / 4:.3f}

采样 {n} 次的实测频率：贪心动作 {counts[1] / n:.3f}，其他三个 {counts[0] / n:.3f}、{counts[2] / n:.3f}、{counts[3] / n:.3f}。

ε-greedy 的探索没有方向：它不知道哪里没去过，只是偶尔乱走一步。
训练时用它采数据；评估时要把 ε 设成 0，否则分数里混进了随机动作的损失。
    """)


# ==================== 第三部分：走廊上的手算与离策略 ====================


class Corridor:
    """三格走廊（CONCEPT.md 第 2.4 节），包装成 Gymnasium 的接口。状态 0、1、2 = A、B、C，3 = 充电桩。"""

    def reset(self):
        self.s = 0
        return self.s, {}

    def step(self, action):
        if action == RIGHT and self.s == 2:
            return 3, 10.0, True, False, {}
        self.s = self.s + 1 if action == RIGHT else max(self.s - 1, 0)
        return self.s, -1.0, False, False, {}


def by_hand():
    """复现 CONCEPT.md 第 7.4 节的手算。"""
    print("\n" + "=" * 60)
    print("第三部分：同一个转移，两个目标")
    print("=" * 60)

    alpha, gamma = 0.5, 0.9
    A, B, C = 0, 1, 2
    Q = np.zeros((4, 2))  # 第 4 行是充电桩（终止状态），恒为 0

    # 转移一：(C, 右, +10, 终止)
    Q[C, RIGHT] += alpha * (10.0 + gamma * 0.0 - Q[C, RIGHT])
    assert np.isclose(Q[C, RIGHT], 5.0)

    # 转移二：(B, 右, -1, C)，两种目标
    y_q = -1.0 + gamma * Q[C].max()
    y_sarsa = -1.0 + gamma * Q[C, LEFT]  # 假设到 C 之后 ε-greedy 正好抽到探索，选了"左"
    q_new, sarsa_new = alpha * y_q, alpha * y_sarsa  # 旧值是 0
    assert np.isclose(y_q, 3.5) and np.isclose(q_new, 1.75)
    assert np.isclose(y_sarsa, -1.0) and np.isclose(sarsa_new, -0.5)

    print(f"""
走廊，Q 表全 0，α = {alpha}，γ = {gamma}。

  转移一：(C, 右, +10, 终止)        Q(C, 右) ← 0 + 0.5 × (10 − 0) = {Q[C, RIGHT]:.2f}

  转移二：(B, 右, -1, C)。此时 Q(C, 左) = 0，Q(C, 右) = 5。
          假设智能体到 C 之后正好抽到探索，实际选了"左"。

      Q-learning：目标 = -1 + 0.9 × max(0, 5) = {y_q:.2f}      Q(B, 右) ← {q_new:.2f}
      SARSA：     目标 = -1 + 0.9 × Q(C, 左)  = {y_sarsa:.2f}     Q(B, 右) ← {sarsa_new:.2f}

  同一个转移，Q-learning 认为"从 B 向右"是好的（到了 C 之后可以向右拿 +10），
  SARSA 认为不好（到了 C 之后实际向左走了）。SARSA 把探索时犯的错算进了价值里，Q-learning 没有。
    """)


def off_policy_demo(n_episodes_q, n_episodes_sarsa):
    """行为策略完全随机时，两个算法各自学到什么。"""
    gamma = 0.9
    env = Corridor()
    rng = np.random.default_rng(SEED)
    # 行为策略：ε = 1，完全随机。Q-learning 的目标是确定的，步长可以大；SARSA 的目标带着 a' 的随机性，步长要小
    Q_q, _ = q_learning(env, 4, 2, n_episodes_q, alpha=0.1, gamma=gamma, epsilon=1.0, rng=rng)
    Q_s, _ = sarsa(env, 4, 2, n_episodes_sarsa, alpha=0.005, gamma=gamma, epsilon=1.0, rng=rng)

    # 两个标准答案（第 02 课的动态规划）：最优策略的 Q*，和随机策略的 Q^π
    q_star = np.array([[4.58, 6.2], [4.58, 8.0], [6.2, 10.0]])
    P_pi = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
    v_random = np.linalg.solve(np.eye(3) - gamma * P_pi, np.array([-1.0, -1.0, 4.5]))
    q_random = np.array([[-1 + gamma * v_random[0], -1 + gamma * v_random[1]],
                         [-1 + gamma * v_random[0], -1 + gamma * v_random[2]],
                         [-1 + gamma * v_random[1], 10.0]])
    gap_q = np.abs(Q_q[:3] - q_star).max()
    gap_s = np.abs(Q_s[:3] - q_random).max()
    assert gap_q < 1e-3, "转移是确定性的，Q-learning 应该精确收敛到 Q*"

    print(f"""  把这个区别推到极端：让行为策略完全随机（ε = 1，左右各一半），两个算法各自学到收敛。

    {'':<8}{'Q-learning 学到的':^22}{'最优策略的 Q*':^20}{'SARSA 学到的':^22}{'随机策略的 Q^π':^20}
    {'状态':<6}{'左':>10}{'右':>10}{'左':>10}{'右':>10}{'左':>10}{'右':>10}{'左':>10}{'右':>10}""")
    for s, name in enumerate("ABC"):
        print(f"    {name:<7}{Q_q[s, 0]:>10.2f}{Q_q[s, 1]:>10.2f}{q_star[s, 0]:>10.2f}{q_star[s, 1]:>10.2f}"
              f"{Q_s[s, 0]:>10.2f}{Q_s[s, 1]:>10.2f}{q_random[s, 0]:>10.2f}{q_random[s, 1]:>10.2f}")
    print(f"""
  数据是同一个随机策略采的，但两个算法在逼近不同的东西：
    - Q-learning 学到了最优策略的价值，与 Q* 最大相差 {gap_q:.1e}。它从头到尾没有执行过一次最优策略。
    - SARSA 学到的是"一直这样随机走下去"的价值，与随机策略的 Q^π 最大相差 {gap_s:.2f}
      （目标里的 a' 是随机的，固定步长下估计会在真值附近小幅晃动）。

  Q-learning 为什么可以这样？看最优方程 Q*(s, a) = E[ r + γ max_a' Q*(s', a') ]：
  期望只对环境给出的 s' 和 r 取，方程里没有任何一处出现采数据的策略。
  给定 (s, a) 之后，(r, s') 是环境决定的，不管这个 a 是贪心选的、随机选的还是人手动按的。
  行为策略只影响一件事：哪些 (s, a) 被更新、更新得多频繁。
  第 06 课的经验回放（用很久以前的策略采的数据来训练）靠的就是这一点。
    """)


# ==================== 第四部分：悬崖行走 ====================


def greedy_rollout(env, Q, max_steps=200):
    """关掉探索，按 Q 贪心走一个回合。返回 (经过的状态, 回报, 是否走到了终点)。"""
    s, _ = env.reset()
    path, total = [s], 0.0
    for _ in range(max_steps):
        s, r, terminated, _, _ = env.step(int(np.argmax(Q[s])))
        path.append(s)
        total += r
        if terminated:
            return path, total, True
    return path, total, False


def render_cliff_path(path):
    """把一条路径画成字符网格：经过的格子标上离开时的方向。"""
    grid = [["·"] * CLIFF_COLS for _ in range(CLIFF_ROWS)]
    for c in range(1, CLIFF_COLS - 1):
        grid[3][c] = "崖"
    for s, s2 in zip(path[:-1], path[1:]):
        dr, dc = s2 // CLIFF_COLS - s // CLIFF_COLS, s2 % CLIFF_COLS - s % CLIFF_COLS
        arrow = {(-1, 0): "↑", (0, 1): "→", (1, 0): "↓", (0, -1): "←"}.get((dr, dc), "?")  # ? = 原地不动或掉崖
        grid[s // CLIFF_COLS][s % CLIFF_COLS] = arrow
    grid[3][CLIFF_COLS - 1] = "终"
    return ["      " + " ".join(f"{ch:<2}" if ch in "·↑→↓←?" else ch for ch in row) for row in grid]


def cliff_walking_experiment(n_runs, n_episodes):
    """CliffWalking-v1 上 Q-learning 与 SARSA 的对比（Sutton & Barto 例 6.6）。"""
    print("\n" + "=" * 60)
    print("第四部分：悬崖行走——区别变成了行为上的差异")
    print("=" * 60)

    alpha, gamma, epsilon = 0.5, 1.0, 0.1
    print(f"""
CliffWalking-v1：4 行 12 列，从左下角走到右下角，中间一排是悬崖。

      ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
      ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
      ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
      起 崖 崖 崖 崖 崖 崖 崖 崖 崖 崖 终

    每走一步 -1；掉下悬崖 -100 并被送回起点（回合不结束）；走到终点回合结束。
    贴着悬崖走最短：13 步，回报 -13。

设置：α = {alpha}，γ = {gamma:.0f}，ε = {epsilon}（训练全程固定），每次运行 {n_episodes} 个回合，重复 {n_runs} 次。
    """)

    env = gym.make("CliffWalking-v1")
    n_states, n_actions = env.observation_space.n, env.action_space.n
    env.reset(seed=SEED)  # 这个环境的转移是确定性的，种子只是为了规范
    algos = {"sarsa": sarsa, "q_learning": q_learning}
    names = {"sarsa": "SARSA", "q_learning": "Q-learning"}
    curves = {m: np.zeros((n_runs, n_episodes)) for m in algos}
    greedy = {m: [] for m in algos}  # 每次运行结束后，贪心策略的 (路径, 回报, 是否到终点)
    for run in range(n_runs):
        for m, algo in algos.items():
            rng = np.random.default_rng(SEED + run)
            Q, returns = algo(env, n_states, n_actions, n_episodes, alpha, gamma, epsilon, rng)
            curves[m][run] = returns
            greedy[m].append(greedy_rollout(env, Q))
    env.close()

    tail = min(100, n_episodes)
    stats = {}
    print(f"  {'算法':<12}{f'训练期间最后 {tail} 回合的平均回报':>24}{'贪心策略走到终点的次数':>20}{'贪心策略的平均回报':>18}{'其中走出最短路 (-13) 的次数':>26}")
    for m in algos:
        train_mean = curves[m][:, -tail:].mean()
        reached = [g for g in greedy[m] if g[2]]
        greedy_mean = np.mean([g[1] for g in reached]) if reached else float("nan")
        n_optimal = sum(1 for g in reached if g[1] == -13)
        stats[m] = (train_mean, len(reached), greedy_mean, n_optimal)
        print(f"  {names[m]:<12}{train_mean:>24.1f}{f'{len(reached)} / {n_runs}':>20}{greedy_mean:>18.1f}{n_optimal:>26}")

    print()
    for m in algos:
        path, total, reached = greedy[m][0]
        rows = [s // CLIFF_COLS for s in path[1:-1]]
        main_row = int(np.bincount(rows, minlength=CLIFF_ROWS).argmax()) + 1 if rows else 0
        status = f"回报 {total:.0f}，主要走在第 {main_row} 行（悬崖在第 4 行）" if reached else "没有走到终点"
        print(f"  第 1 次运行结束后，{names[m]} 的贪心策略走的路（{status}）：")
        for line in render_cliff_path(path if reached else path[:30]):
            print(line)
        print()

    s_train, q_train = stats["sarsa"][0], stats["q_learning"][0]
    s_greedy, q_greedy = stats["sarsa"][2], stats["q_learning"][2]
    if s_train > q_train:
        print(f"  训练期间（带着 ε = {epsilon} 的探索）：SARSA 的平均回报 {s_train:.1f} 高于 Q-learning 的 {q_train:.1f}。")
        print("  Q-learning 学的是最优策略的价值，最优策略贴着悬崖走；但训练时它的实际行为掺了随机动作，")
        print("  贴边走时一次随机的「向下」就掉下去，-100。SARSA 把这个风险算进了价值，所以绕远一点。")
    else:
        print(f"  训练期间：SARSA 的平均回报 {s_train:.1f}，Q-learning {q_train:.1f}。这次实验里 SARSA 没有表现出优势。")
    if q_greedy > s_greedy:
        print(f"  关掉探索之后：Q-learning 的贪心策略平均回报 {q_greedy:.1f}，好于 SARSA 的 {s_greedy:.1f}。")
        print("  不再有随机动作时，贴边走不再危险，最短的路就是最好的路。")
    elif np.isnan(q_greedy) or np.isnan(s_greedy):
        print("  关掉探索之后，有一种算法的贪心策略在所有运行里都没能走到终点，见上表。")
    else:
        print(f"  关掉探索之后：Q-learning 的贪心策略平均回报 {q_greedy:.1f}，SARSA {s_greedy:.1f}。")
    print("  两个算法都没有错，它们回答的是不同的问题：")
    print("  SARSA 回答「带着探索行动时怎么走最好」，Q-learning 回答「不犯错的话怎么走最好」。")
    print("  真机上摔一次很贵时，前者可能正是你想要的；训练完要部署一个不带探索的策略时，后者更合适。")

    # 图：训练期间的回报曲线
    fig, ax = plt.subplots(figsize=(9, 4.8))
    window = min(10, n_episodes)
    kernel = np.ones(window) / window
    for m, color in (("sarsa", "tab:blue"), ("q_learning", "tab:red")):
        mean_curve = curves[m].mean(axis=0)
        smooth = np.convolve(mean_curve, kernel, mode="valid")
        ax.plot(np.arange(window, n_episodes + 1), smooth, color=color, linewidth=2, label=names[m])
    ax.axhline(-13, color="gray", linestyle=":", label="最短路的回报 -13")
    ax.set_ylim(-120, 0)
    ax.set_xlabel("回合数")
    ax.set_ylabel(f"训练期间每回合的奖励之和（{n_runs} 次运行平均，{window} 回合滑动平均）")
    ax.set_title(f"CliffWalking-v1：带着 ε = {epsilon} 探索时的表现")
    ax.legend()
    save_figure(fig, OUT / "05-cliff-walking-returns.png")
    return stats


# ==================== 第五部分：练习与思考 ====================


def exercises(n_runs, n_episodes):
    """练习题与参考答案。答案里的数字都是下面的代码现算的。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    # 练习 2：SARSA 的表现怎样随 ε 变化
    env = gym.make("CliffWalking-v1")
    env.reset(seed=SEED)
    tail = min(100, n_episodes)
    ex2 = {}
    for epsilon in (0.01, 0.3):
        trains, greedys = [], []
        for run in range(n_runs):
            rng = np.random.default_rng(SEED + 100 + run)
            Q, returns = sarsa(env, 48, 4, n_episodes, 0.5, 1.0, epsilon, rng)
            trains.append(returns[-tail:].mean())
            _, total, reached = greedy_rollout(env, Q)
            if reached:
                greedys.append(total)
        ex2[epsilon] = (np.mean(trains), np.mean(greedys) if greedys else float("nan"), len(greedys))
    env.close()
    if ex2[0.01][0] > ex2[0.3][0]:
        ex2_comment = "实测里 ε 小的那组训练回报更高：随机动作少，掉下悬崖的次数也少。"
    else:
        ex2_comment = "这次实测里 ε 小的那组训练回报并不更高（探索太少时前期学得慢，回合数不多时可能还没走顺）。"

    print(f"""
练习 1：ε-greedy 的概率
    ε = 0.2、2 个动作（CartPole 的情况）时，贪心动作被选中的概率是多少？

练习 1 参考答案：
    1 − ε + ε/2 = {1 - 0.2 + 0.2 / 2:.1f}。随机的那一半也有一半落在贪心动作上，
    所以"实际做了非贪心动作"的概率只有 ε/2 = {0.2 / 2:.1f}，不是 ε。

练习 2：SARSA 绕多远取决于 ε
    把第四部分 SARSA 的 ε 改成 0.01 和 0.3（各 {n_runs} 次运行），训练回报和贪心策略的路各有什么变化？

练习 2 参考答案（程序算出来的；训练回报是最后 {tail} 个回合的平均）：
    ε = 0.01：训练回报 {ex2[0.01][0]:.1f}，贪心策略的平均回报 {ex2[0.01][1]:.1f}（{ex2[0.01][2]} / {n_runs} 次走到终点）
    ε = 0.3： 训练回报 {ex2[0.3][0]:.1f}，贪心策略的平均回报 {ex2[0.3][1]:.1f}（{ex2[0.3][2]} / {n_runs} 次走到终点）
    {ex2_comment}
    原理上，SARSA 评估的是"带着 ε 的随机性行动"的策略。ε 越小，随机失足的风险越小，贴边走的代价越低，
    它要学的价值就越接近 Q*；ε 越大，越值得离悬崖远一些。
    ε 逐渐衰减到 0 时，SARSA 也收敛到最优策略。

练习 3：把 Q-learning 改成"期望 SARSA"
    把目标里的 Q(s', a') 换成它在 ε-greedy 策略下的期望 Σ_a' π(a'|s')·Q(s', a')，要改哪一行？

练习 3 参考答案：
    只改算 bootstrap 的那一行（n 是动作个数）：
        probs = np.full(n, epsilon / n); probs[np.argmax(Q[s2])] += 1 − epsilon
        bootstrap = np.dot(probs, Q[s2])
    它和 SARSA 逼近的是同一个方程，但把"a' 抽到哪个"的随机性提前平均掉了，目标的方差更小。
    ε = 0 时它就是 Q-learning。

思考题 1：Q-learning 是离策略的，是不是用什么数据都能学好？

思考题 1 参考答案：
    不是。离策略只是说"目标不依赖行为策略"，数据的覆盖仍然由行为策略决定：
    没被访问过的 (s, a) 不会被更新，它的 Q 值是瞎的，而目标里的 max 偏偏喜欢挑这种虚高的值。
    第三部分的例子里随机策略把每个 (s, a) 都访问了成千上万次，所以没问题。

思考题 2：能不能把 Q-learning 的一步目标直接换成上一节的 n 步回报或 λ-return？

思考题 2 参考答案：
    不能随意换。n 步回报里中间那几步的奖励是行为策略的动作带来的，目标策略在这些步上如果会选别的动作，
    这些奖励就不是它会拿到的，目标就有偏。一步目标没有这个问题（中间没有别的动作）。
    Dreamer 能放心地用 15 步的 λ-return，是因为它的轨迹不是回放里的旧数据，
    而是世界模型按当前策略现场想象出来的：在想象里，数据永远是在策略的。

思考题 3：CliffWalking 只有 48 个状态，一张 [48, 4] 的表就够了。换成 CartPole 的 4 个连续实数呢？

思考题 3 参考答案：
    表格放不下，而且表格不会举一反三：两个几乎一样的状态是两个不相干的格子。
    下一节把查表换成函数调用，用神经网络近似 Q，这就是 DQN。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()
    epsilon_greedy_check()
    by_hand()
    if is_quick():
        off_policy_demo(n_episodes_q=500, n_episodes_sarsa=2000)
        cliff_walking_experiment(n_runs=2, n_episodes=150)
        exercises(n_runs=1, n_episodes=150)
    else:
        off_policy_demo(n_episodes_q=5000, n_episodes_sarsa=40000)
        cliff_walking_experiment(n_runs=10, n_episodes=500)
        exercises(n_runs=3, n_episodes=500)

    print("\n" + "=" * 60)
    print("本节完成！下一节：06-dqn-from-scratch.py（DQN：经验回放与目标网络）")
    print("=" * 60)


if __name__ == "__main__":
    main()
