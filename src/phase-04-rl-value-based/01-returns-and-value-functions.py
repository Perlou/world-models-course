"""
回报、折扣与价值函数
====================

学习目标：
    1. 说清楚 Phase 1–3 的规划器的两个硬伤（慢、近视），以及价值函数怎样同时解决它们
    2. 会从后往前算折扣回报 G_t = r_t + γ·G_{t+1}，理解折扣因子 γ 和有效视野 1/(1-γ)
    3. 按定义（大量回合取平均）估计状态价值 V^π 和动作价值 Q^π，并和手算结果对上
    4. 看到"价值取决于策略""奖励低的状态价值可以很高"这两件事
    5. 分清 terminated 和 truncated：回合被掐断时，剩下的回报要用价值补上

核心概念：
    - 回报（return）：从某一步起的折扣奖励之和 G_t
    - 折扣因子（discount factor）γ 与有效视野（effective horizon）1/(1-γ)
    - 状态价值函数 V^π(s)、动作价值函数 Q^π(s, a)：回报的期望，总是相对于某个策略
    - 一步前瞻（one-step lookahead）：用模型看一步，再用 V 补上之后的全部

前置知识：
    - Phase 1 的 MDP 词汇（状态、动作、奖励、策略、轨迹）和 MPC 的规划视野
    - NumPy 基础。本章从零讲强化学习，不需要别的基础

对应概念文档：
    - CONCEPT.md 第 1 节「为什么现在要学强化学习」、第 2 节「回报、折扣与价值函数」
      （一步前瞻的手算例子在第 3.7 节）

预计运行时间：尚未实测（本文件是还没有运行过的草稿，定稿前要完整跑一遍并改写这一行）
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-04")
SEED = 0
GAMMA = 0.9
LEFT, RIGHT = 0, 1
ACTION_NAMES = ["左", "右"]
# 这一节全是表格和标量运算，用 NumPy 在 CPU 上算（数值用默认的 float64，方便和手算逐位对照）


# ==================== 第一部分：为什么需要价值函数 ====================


def introduction():
    """规划器的两个硬伤，以及贯穿全章的走廊环境。"""
    print("=" * 60)
    print("第一部分：为什么现在要学强化学习")
    print("=" * 60)

    print("""
Phase 1–3 的决策方式都是规划：每走一步，在模型里试成百上千条动作序列，选最好的。
它有两个硬伤：

    ① 慢：  每个真实时间步都要从头搜索一遍
    ② 近视：只看得到规划视野 H 以内的奖励，视野之外完全看不见

强化学习（Reinforcement Learning，RL）给的办法是把搜索的结果存下来。
这一章讲其中的一半：价值函数（value function）。它回答的问题是

    "从这个状态出发，之后一共还能拿多少奖励？"

有了它，决策时只需要看一步：

    纯规划：    比较  r_0 + γ r_1 + … + γ^(H-1) r_(H-1)              （H 要取得很长）
    规划 + 价值：比较  r_0 + … + γ^(H-1) r_(H-1) + γ^H · V(s_H)      （H 可以很短，甚至是 1）
                                                  └─ 视野之外的部分由 V 补上

贯穿全章的例子：三格走廊。一个扫地机器人要回到最右边的充电桩。

      ┌───────┬───────┬───────┬───────────┐
      │   A   │   B   │   C   │  T 充电桩  │
      └───────┴───────┴───────┴───────────┘
                 ◀── 左      右 ──▶

      状态：A、B、C；T 是终止状态          动作：左、右（转移是确定性的）
      奖励：每走一步 -1（耗电）；从 C 向右走进充电桩 +10，回合结束
           在 A 向左：撞墙，留在 A，奖励 -1
      折扣：γ = 0.9

这一节做四件事：算回报 → 按定义估计价值 → 把价值接到规划器上 → 处理回合被掐断的情况。
    """)


class Corridor:
    """一条 n 格的走廊，最右边是充电桩（终止状态）。

    状态：0 … n-1（三格时就是 A、B、C）；编号 n 表示充电桩 T
    动作：0 = 左，1 = 右
    奖励：每走一步 step_reward；从最右一格向右走进充电桩得 goal_reward，回合结束
    """

    def __init__(self, n_cells=3, step_reward=-1.0, goal_reward=10.0):
        self.n_states = n_cells
        self.n_actions = 2
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.state = 0

    def model(self, state, action):
        """环境的规律：给定 (s, a)，返回 (s', r, terminated)。不改变环境当前的状态。

        规划器调用的就是这个函数（Phase 1 的说法：一个完美的世界模型）。
        """
        if action == RIGHT:
            if state == self.n_states - 1:
                return self.n_states, self.goal_reward, True
            return state + 1, self.step_reward, False
        return max(state - 1, 0), self.step_reward, False

    def reset(self, state=0):
        self.state = state
        return state

    def step(self, action):
        self.state, reward, terminated = self.model(self.state, action)
        return self.state, reward, terminated

    def name(self, state):
        if state == self.n_states:
            return "T"
        return "ABCDEFGHIJKLMNOP"[state]


# ==================== 第二部分：回报 ====================


def discounted_returns(rewards, gamma):
    """从后往前算每一步的折扣回报（CONCEPT.md 第 2.5 节）。

    递推式：G_t = r_t + γ · G_{t+1}，回合结束之后的回报是 0。

    参数：
        rewards: [T] 或 [B, T]，一条或一批轨迹的奖励
    返回：
        returns: 形状同 rewards，每个位置是 G_t
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    returns = np.zeros_like(rewards)
    g = np.zeros(rewards.shape[:-1])  # [] 或 [B]
    for t in reversed(range(rewards.shape[-1])):
        g = rewards[..., t] + gamma * g
        returns[..., t] = g
    return returns


def discounted_returns_by_definition(rewards, gamma):
    """按定义从前往后算：每个 t 都重新求一次和，复杂度 O(T²)。只用来对照。"""
    rewards = np.asarray(rewards, dtype=np.float64)
    T = len(rewards)
    return np.array([sum(gamma**k * rewards[t + k] for k in range(T - t)) for t in range(T)])


def returns_by_hand():
    """复现 CONCEPT.md 第 2.4 节的手算一。"""
    print("\n" + "=" * 60)
    print("第二部分：回报——把未来的奖励加起来")
    print("=" * 60)

    print("""
奖励（reward）说的是眼前这一步，回报（return）说的是从现在起的全部：

    G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …

把第二项起提出一个 γ，就得到全章用得最多的恒等式：

    G_t = r_t + γ · G_{t+1}          现在的回报 = 这一步的奖励 + 折扣后的下一步的回报

所以回报可以从后往前一遍算完（discounted_returns）。

手算一：策略"一直向右"，从 A 出发，轨迹 A → B → C → T，奖励 -1, -1, +10。
    """)

    rewards = [-1.0, -1.0, 10.0]
    returns = discounted_returns(rewards, GAMMA)
    by_def = discounted_returns_by_definition(rewards, GAMMA)
    print(f"  {'步 t':<6}{'状态':<6}{'奖励 r_t':>10}{'从后往前递推 G_t':>18}{'按定义求和 G_t':>16}")
    for t, s in enumerate("ABC"):
        print(f"  {t:<7}{s:<7}{rewards[t]:>10.1f}{returns[t]:>18.3f}{by_def[t]:>16.3f}")
    assert np.allclose(returns, [6.2, 8.0, 10.0]), "应该和 CONCEPT.md 第 2.4 节的手算一致"
    assert np.allclose(returns, by_def)
    print("""
  和手算一致：G_2 = 10，G_1 = -1 + 0.9×10 = 8，G_0 = -1 + 0.9×8 = 6.2。
  两种算法结果相同，区别在计算量：递推扫一遍是 O(T)，按定义对每个 t 重新求和是 O(T²)。
  "从后往前递推"这个写法后面会反复出现：第 04 课的 λ-return、Phase 5 的 GAE、Phase 6 的 Dreamer。
    """)

    # 同一个函数也能处理一批轨迹 [B, T]
    batch = np.array([[-1.0, -1.0, 10.0], [2.0, 0.0, 3.0]])
    print(f"  一批轨迹 rewards 的形状 {batch.shape} → returns 的形状 {discounted_returns(batch, 0.5).shape}")
    print(f"  第二条轨迹（奖励 2, 0, 3，γ = 0.5）的回报：{discounted_returns(batch, 0.5)[1]}")


def discount_and_horizon():
    """折扣因子决定价值函数看多远。"""
    print("\n" + "=" * 60)
    print("第三部分：折扣因子与有效视野")
    print("=" * 60)

    print("""
第 k 步之后的奖励，权重是 γ^k。γ 越接近 1，价值函数"看"得越远。
习惯上把 1/(1-γ) 叫作有效视野（effective horizon）。
下表每一行都是算出来的：假设每一步奖励恒为 1，实际把 20000 步的回报加起来。
    """)

    gammas = [0.9, 0.99, 0.997, 0.999]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = np.arange(0, 1500)
    print(f"  {'γ':<8}{'有效视野 1/(1-γ)':>16}{'奖励恒为 1 的回报':>18}{'前 1/(1-γ) 步的权重占比':>24}{'第 50 步的权重 γ^50':>20}")
    for gamma in gammas:
        horizon = 1.0 / (1.0 - gamma)
        total = discounted_returns(np.ones(20000), gamma)[0]
        n = int(round(horizon))
        share = discounted_returns(np.ones(n), gamma)[0] / total  # 前 n 步的权重和 / 总权重
        print(f"  {gamma:<8}{horizon:>16.0f}{total:>18.1f}{share:>24.1%}{gamma**50:>20.4f}")
        ax.plot(steps, gamma**steps, linewidth=2, label=f"γ = {gamma}（有效视野 {horizon:.0f} 步）")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("多少步之后（k）")
    ax.set_ylabel("这一步奖励的权重 γ^k")
    ax.set_title("折扣因子决定价值函数看多远")
    ax.legend(fontsize=9)
    save_figure(fig, OUT / "01-discount-weights.png")

    print("""
三点观察：

    1. 回报有上界：每步奖励不超过 r_max 时，|G| ≤ r_max / (1-γ)。表里"奖励恒为 1 的回报"
       正好等于有效视野。调试时如果网络输出的价值超过这个上界，一定是哪里错了。
    2. 不管 γ 取多少，前 1/(1-γ) 步都占总权重的 63% 左右（1 - 1/e）。
    3. γ = 0.9 时第 50 步的权重已经不到 1%，等于看不见。γ 是任务定义的一部分，不是随便取的。
       γ = 0.997 是 Dreamer V3 的默认折扣，有效视野约 333 步；
       对比 Phase 1 的规划视野 H = 30，价值函数看到的未来比规划器远一个量级。
    """)


# ==================== 第三部分：按定义估计价值 ====================


def run_episode(env, policy, rng, start_state, first_action=None, max_steps=500):
    """按策略跑一个回合。

    参数：
        policy:       [S, A]，每行是该状态下各动作的概率，行和为 1
        first_action: 不为 None 时，第一步强制执行这个动作（估计 Q 时用），之后再按 policy
        max_steps:    步数上限。到了上限还没结束就掐断（truncated）
    返回：
        states [T+1]（含最后到达的状态）、actions [T]、rewards [T]、terminated
    """
    cum_policy = np.cumsum(policy, axis=1)  # [S, A]，用来按概率抽动作
    s = env.reset(start_state)
    states, actions, rewards = [s], [], []
    terminated = False
    for t in range(max_steps):
        if t == 0 and first_action is not None:
            a = first_action
        else:
            # 按概率 policy[s] 抽一个动作（比 rng.choice 快得多，这个循环要跑上百万次）
            a = min(int(np.searchsorted(cum_policy[s], rng.random(), side="right")), env.n_actions - 1)
        s, r, terminated = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
        if terminated:
            break
    return states, actions, rewards, terminated


def estimate_value_by_rollouts(env, policy, gamma, n_episodes, rng):
    """按定义硬算 V^π：从每个状态出发跑 n_episodes 个回合，把回报取平均。

    这是最笨的办法，但它不依赖任何理论，后面可以用来检验别的算法。

    返回：
        V:       [S]
        returns: [S, n_episodes]，每个回合从起点算起的回报 G_0
    """
    returns = np.zeros((env.n_states, n_episodes))
    for s in range(env.n_states):
        for i in range(n_episodes):
            _, _, rewards, _ = run_episode(env, policy, rng, start_state=s)
            returns[s, i] = discounted_returns(rewards, gamma)[0]
    return returns.mean(axis=1), returns


def estimate_q_by_rollouts(env, policy, gamma, n_episodes, rng):
    """按定义硬算 Q^π(s, a)：第一步强制做 a，之后按 π。返回 Q [S, A]。"""
    Q = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        for a in range(env.n_actions):
            total = 0.0
            for _ in range(n_episodes):
                _, _, rewards, _ = run_episode(env, policy, rng, start_state=s, first_action=a)
                total += discounted_returns(rewards, gamma)[0]
            Q[s, a] = total / n_episodes
    return Q


def random_policy_reference(gamma):
    """走廊在随机策略下的精确价值：解三元一次方程组（CONCEPT.md 第 3.5 节例二）。

    方程从哪来是下一节（Bellman 方程）的内容，这里先把它当标准答案用。
        V(A) = -1 + 0.5γ V(A) + 0.5γ V(B)
        V(B) = -1 + 0.5γ V(A) + 0.5γ V(C)
        V(C) = 0.5·(-1 + γ V(B)) + 0.5·10
    """
    h = 0.5 * gamma
    lhs = np.array([[1 - h, -h, 0.0], [-h, 1.0, -h], [0.0, -h, 1.0]])
    rhs = np.array([-1.0, -1.0, 0.5 * (-1.0) + 0.5 * 10.0])
    return np.linalg.solve(lhs, rhs)


def values_by_definition():
    """复现 CONCEPT.md 第 2.4 节的手算二、手算三。"""
    print("\n" + "=" * 60)
    print("第四部分：价值函数——回报的期望")
    print("=" * 60)

    print("""
回报是一条轨迹上的数。同一个状态出发可能有很多条轨迹（策略有随机性、环境有随机性），
把回报对所有可能的轨迹取期望，就是价值：

    状态价值  V^π(s)    = E_π[ G_t | s_t = s ]            从 s 出发，一直按 π 行动
    动作价值  Q^π(s, a) = E_π[ G_t | s_t = s, a_t = a ]   先做 a（不管 π 怎么说），之后按 π

上标 π 不能省：价值总是"按某个策略行动"的价值。
最直接的估计办法就是按定义来：跑很多回合，把回报取平均。

手算二、三：策略"一直向右"。环境和策略都是确定性的，每个状态出发只有一条轨迹，跑 1 个回合就够。
    """)

    env = Corridor()
    rng = np.random.default_rng(SEED)
    always_right = np.tile([0.0, 1.0], (env.n_states, 1))  # [S, A]，每行是 (P(左), P(右))
    V, _ = estimate_value_by_rollouts(env, always_right, GAMMA, n_episodes=1, rng=rng)
    Q = estimate_q_by_rollouts(env, always_right, GAMMA, n_episodes=1, rng=rng)

    print(f"  {'状态':<6}{'Q(s, 左)':>10}{'Q(s, 右)':>10}{'V(s)':>10}{'  这一步的奖励'}")
    for s in range(env.n_states):
        r_right = env.model(s, RIGHT)[1]
        print(f"  {env.name(s):<7}{Q[s, LEFT]:>10.2f}{Q[s, RIGHT]:>10.2f}{V[s]:>10.2f}{r_right:>10.0f}（向右）")
    assert np.allclose(V, [6.2, 8.0, 10.0])
    assert np.allclose(Q, [[4.58, 6.2], [4.58, 8.0], [6.2, 10.0]])
    print(f"""
  和 CONCEPT.md 第 2.4 节的表一致。读这张表：

    - V(s) 等于 Q(s, 右)：策略在每个状态都选"右"，V 是 Q 按策略的概率加权平均。
    - 每一行"右"都比"左"大：在任何状态偏离这个策略都会变差，所以它就是最优策略。
    - Q(B, 左) = {Q[1, LEFT]:.2f} 并不小。Q 衡量的是"这一步走错、之后改回来"的代价
      （{Q[1, RIGHT]:.2f} - {Q[1, LEFT]:.2f} = {Q[1, RIGHT] - Q[1, LEFT]:.2f}），不是"一直走错"的代价。
    - 状态 B 每一步的奖励都是 -1，但 V(B) = {V[1]:.0f}。奖励低的状态价值可以很高，价值看的是之后的全部。
    """)
    return V


def convergence_experiment(n_max, n_repeats):
    """随机策略：回合平均估计的 V 怎样随回合数收敛。"""
    print("\n" + "=" * 60)
    print("第五部分：换一个策略，价值就变了")
    print("=" * 60)

    print(f"""
现在让机器人在每个状态都以各一半的概率向左或向右（随机策略）。
这时从一个状态出发有无穷多条可能的轨迹，一个回合的回报是随机的，必须跑很多回合取平均。

标准答案用解方程组得到（方程从哪来是下一节的内容）。
实验：每个状态出发跑 {n_max} 个回合，看前 n 个回合的平均离标准答案多远；重复 {n_repeats} 次。
    """)

    env = Corridor()
    random_policy = np.full((env.n_states, env.n_actions), 0.5)
    v_ref = random_policy_reference(GAMMA)
    checkpoints = [n for n in (10, 100, 1000, 10000) if n <= n_max]

    errors = np.zeros((n_repeats, len(checkpoints)))  # 每次重复、每个回合数下的最大误差
    first_returns = None
    for rep in range(n_repeats):
        rng = np.random.default_rng(SEED + rep)
        _, returns = estimate_value_by_rollouts(env, random_policy, GAMMA, n_max, rng)  # [S, n_max]
        if rep == 0:
            first_returns = returns
        for j, n in enumerate(checkpoints):
            errors[rep, j] = np.abs(returns[:, :n].mean(axis=1) - v_ref).max()
    mean_err = errors.mean(axis=0)

    print(f"  {'回合数 n':<10}{'V(A)':>9}{'V(B)':>9}{'V(C)':>9}{'最大误差（第 1 次）':>18}{f'最大误差（{n_repeats} 次平均）':>20}")
    for j, n in enumerate(checkpoints):
        est = first_returns[:, :n].mean(axis=1)
        print(f"  {n:<12}{est[0]:>9.3f}{est[1]:>9.3f}{est[2]:>9.3f}{errors[0, j]:>18.3f}{mean_err[j]:>20.3f}")
    print(f"  {'解方程组':<10}{v_ref[0]:>9.3f}{v_ref[1]:>9.3f}{v_ref[2]:>9.3f}")

    slope = np.polyfit(np.log(checkpoints), np.log(mean_err), 1)[0]
    std = first_returns.std(axis=1)  # [S]，单个回合回报的标准差
    predicted = std.max() / np.sqrt(checkpoints[-1])
    if np.all(np.diff(mean_err) < 0):
        print(f"\n  回合数每增加 10 倍，平均最大误差都在变小；对数坐标下的斜率是 {slope:.2f}"
              f"（独立样本取平均的理论值是 -0.5，即误差 ∝ 1/√n）。")
    else:
        print(f"\n  这次运行里平均最大误差没有随回合数单调下降（对数坐标下的斜率 {slope:.2f}）。"
              f"重复次数少时会出现这种情况，趋势仍应是 1/√n。")
    print(f"  单个回合的回报很分散：从 A、B、C 出发的标准差分别是 {std[0]:.2f}、{std[1]:.2f}、{std[2]:.2f}。")
    print(f"  按「标准差 / √n」估算，{checkpoints[-1]} 个回合时误差的量级是 {predicted:.3f}，"
          f"实测的平均最大误差是 {mean_err[-1]:.3f}。")

    # 图：左，前 n 个回合的平均；中，误差随回合数；右，从 A 出发的回报的分布
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    n_axis = np.arange(1, n_max + 1)
    for s in range(env.n_states):
        running = np.cumsum(first_returns[s]) / n_axis
        line, = axes[0].plot(n_axis, running, linewidth=1.5, label=f"V({env.name(s)}) 的估计")
        axes[0].axhline(v_ref[s], color=line.get_color(), linestyle="--", linewidth=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("用了多少个回合（对数坐标）")
    axes[0].set_ylabel("前 n 个回合的平均回报")
    axes[0].set_title("随机策略：回合平均逐渐靠近精确值（虚线）")
    axes[0].legend(fontsize=9)

    axes[1].plot(checkpoints, mean_err, "o-", linewidth=2, label=f"实测（{n_repeats} 次重复的平均）")
    axes[1].plot(checkpoints, mean_err[0] * np.sqrt(checkpoints[0] / np.array(checkpoints)), "k--", label="1/√n 参考线")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("回合数 n")
    axes[1].set_ylabel("三个状态上的最大误差")
    axes[1].set_title("估计误差随回合数的变化")
    axes[1].legend(fontsize=9)

    axes[2].hist(first_returns[0], bins=40, color="tab:blue", alpha=0.8)
    axes[2].axvline(v_ref[0], color="k", linestyle="--", label=f"期望 V(A) = {v_ref[0]:.2f}")
    axes[2].set_xlabel("一个回合的回报 G_0")
    axes[2].set_ylabel("回合数")
    axes[2].set_title("随机策略下从 A 出发的回报分布")
    axes[2].legend(fontsize=9)
    save_figure(fig, OUT / "01-rollout-value-estimates.png")

    print(f"""
对比两个策略在同一个环境、同一批状态上的价值：

    策略                    V(A)      V(B)      V(C)
    一直向右                6.200     8.000    10.000
    随机（左右各一半）     {v_ref[0]:>6.3f}    {v_ref[1]:>6.3f}    {v_ref[2]:>6.3f}

价值不是环境的属性，是"环境 + 策略"的属性。

右图还说明一件事：价值只是这个分布的均值，一个数。回报本身的形状被平均掉了。
第 07 课的分布式价值（C51）会把整个分布学出来。

按定义硬算有两个明显的问题，后面几节逐个解决：
    - 要跑很多完整的回合才能平均掉噪声 → 第 03 课的时序差分（TD）
    - 这里是从每个状态各自出发重复采样，真实环境里做不到"想从哪开始就从哪开始"
    """)
    return v_ref


# ==================== 第四部分：把价值接到规划器上 ====================


def plan_by_enumeration(env, state, horizon, gamma, terminal_value=None):
    """Phase 1 式的规划：枚举所有长度为 horizon 的动作序列，在模型里展开并打分。

    参数：
        terminal_value: None 表示末端不加价值；否则是 V [S]，分数里加上 γ^H · V(s_H)
    返回：
        results:       [(动作序列, 分数)]
        n_model_calls: 一共调用了多少次模型
    """
    results = []
    n_model_calls = 0
    for seq in itertools.product(range(env.n_actions), repeat=horizon):
        s, score, discount, terminated = state, 0.0, 1.0, False
        for a in seq:
            s, r, terminated = env.model(s, a)
            n_model_calls += 1
            score += discount * r
            discount *= gamma
            if terminated:
                break
        if terminal_value is not None and not terminated:
            score += discount * terminal_value[s]  # 视野之外的部分由价值函数补上
        results.append((seq, score))
    return results, n_model_calls


def best_first_actions(results):
    """分数并列最高的那些序列，第一个动作有哪些。只有一个才算"规划器选得出来"。"""
    best = max(score for _, score in results)
    return sorted({seq[0] for seq, score in results if score > best - 1e-9})


def planning_with_value(v_star):
    """复现 CONCEPT.md 第 3.7 节：短视野规划分不出好坏，接上价值就能选对。"""
    print("\n" + "=" * 60)
    print("第六部分：给规划器接上价值")
    print("=" * 60)

    print("""
机器人在 A。用 Phase 1 的办法规划：枚举所有长度为 H 的动作序列，在模型里展开，选分数最高的。
充电桩在 3 步之外。先看视野 H = 2：
    """)

    env = Corridor()
    plain, _ = plan_by_enumeration(env, 0, 2, GAMMA)
    with_v, _ = plan_by_enumeration(env, 0, 2, GAMMA, terminal_value=v_star)
    print(f"  {'动作序列':<10}{'累计奖励（末端不加价值）':>20}{'累计奖励 + γ²·V(末端)':>24}")
    for (seq, score_plain), (_, score_v) in zip(plain, with_v):
        name = "、".join(ACTION_NAMES[a] for a in seq)
        print(f"  {name:<12}{score_plain:>20.2f}{score_v:>24.2f}")
    assert len(best_first_actions(plain)) == 2, "末端不加价值时，四条序列应该并列"
    assert best_first_actions(with_v) == [RIGHT]

    lookahead = [env.model(0, a)[1] + GAMMA * v_star[env.model(0, a)[0]] for a in (LEFT, RIGHT)]
    print(f"""
  末端不加价值：四条序列分数一模一样，规划器分不出哪条路在靠近充电桩。这就是"近视"。
  末端加上 V：  "右、右"以 {max(s for _, s in with_v):.2f} 胜出，正好等于 V(A)。

  视野还可以更短。H = 1 时只比较两个动作（一步前瞻，one-step lookahead）：
      左：-1 + 0.9 × V(A) = {lookahead[LEFT]:.2f}        右：-1 + 0.9 × V(B) = {lookahead[RIGHT]:.2f}
  这两个数就是 Q(A, 左) 和 Q(A, 右)。如果直接把 Q 学出来，连这一次模型调用都省了。
    """)

    # 走廊变长：纯规划的计算量指数增长，而一步前瞻的代价不变
    n_cells = 8 if not is_quick() else 6
    long_env = Corridor(n_cells=n_cells)
    # "一直向右"在长走廊里的价值：奖励序列是 n-1 个 -1 加一个 +10，从后往前算一遍就是每一格的价值
    long_v = discounted_returns([-1.0] * (n_cells - 1) + [10.0], GAMMA)
    print(f"  把走廊加长到 {n_cells} 格，机器人在最左边，充电桩在 {n_cells} 步之外：\n")
    print(f"  {'视野 H':<8}{'候选序列数':>10}{'模型调用次数':>14}{'  末端不加价值':<16}{'末端加 V'}")
    first_ok = None
    for horizon in range(1, n_cells + 3):
        plain, calls = plan_by_enumeration(long_env, 0, horizon, GAMMA)
        with_v, _ = plan_by_enumeration(long_env, 0, horizon, GAMMA, terminal_value=long_v)
        choice_plain = best_first_actions(plain)
        choice_v = best_first_actions(with_v)
        text_plain = "选「右」" if choice_plain == [RIGHT] else "分不出（并列）"
        text_v = "选「右」" if choice_v == [RIGHT] else "分不出（并列）"
        if first_ok is None and choice_plain == [RIGHT]:
            first_ok = (horizon, calls)
        print(f"  {horizon:<10}{2**horizon:>10}{calls:>14}    {text_plain:<16}{text_v}")
    _, calls_h1 = plan_by_enumeration(long_env, 0, 1, GAMMA, terminal_value=long_v)

    if first_ok is not None:
        print(f"""
  末端不加价值的规划器，视野要到 H = {first_ok[0]} 才第一次选得出来，那时每一步决策要调用 {first_ok[1]} 次模型；
  视野不够长时，它看到的全是耗电。
  末端加上 V 的规划器从 H = 1 起就选对了，每一步决策只调用 {calls_h1} 次模型。
        """)
    print("""  一个式子同时治两个硬伤：末端加 V 治"近视"；既然 V 概括了远处，H 就可以取短，治"慢"。

      视野 H 长，不用价值       PETS、PlaNet（Phase 1、3）
      视野 H 短 + 学到的价值    TD-MPC2（Phase 7）
      H = 1 + 学到的 V          一步前瞻（需要模型）
      H = 0 + 学到的 Q 或策略   DQN（本章）、Dreamer 的 actor（Phase 6）

  当然，这里用的 V 是精确的。它从哪来？下一节（Bellman 方程、动态规划）先讲模型已知时怎么算，
  第 03 课起讲只有交互经验时怎么学。
    """)


# ==================== 第五部分：回合被掐断怎么办 ====================


def truncation_demo(v_star):
    """terminated 与 truncated：被掐断的回合，剩下的回报要用价值补上（CONCEPT.md 第 2.7 节）。"""
    print("\n" + "=" * 60)
    print("第七部分：回合结束的两种方式")
    print("=" * 60)

    print("""
Phase 1 说过回合有两种结束方式，学价值函数时必须分开：

    terminated：环境进入终止状态（走进充电桩），之后真的没有奖励了，终止状态的价值按定义是 0
    truncated： 到了步数上限被外部掐断，环境本身还能继续，未来的奖励还在，只是数据里没有

演示：策略"一直向右"，从 A 出发，但步数上限设成 2。轨迹是 A → B → C，然后被掐断。
    """)

    env = Corridor()
    rng = np.random.default_rng(SEED)
    always_right = np.tile([0.0, 1.0], (env.n_states, 1))
    states, _, rewards, terminated = run_episode(env, always_right, rng, start_state=0, max_steps=2)
    last = states[-1]
    as_terminal = discounted_returns(rewards, GAMMA)[0]
    bootstrapped = as_terminal + GAMMA ** len(rewards) * v_star[last]
    print(f"  拿到的奖励：{rewards}，terminated = {terminated}，最后停在 {env.name(last)}")
    print(f"  把掐断当成终止：        G_0 = {as_terminal:.2f}")
    print(f"  掐断处用 V({env.name(last)}) 补上：  G_0 = {as_terminal:.2f} + 0.9² × {v_star[last]:.0f} = {bootstrapped:.2f}")
    print(f"  真实的 V(A)：                  {v_star[0]:.2f}")
    assert not terminated and np.isclose(bootstrapped, v_star[0])
    print(f"""
  把掐断当成终止，等于告诉机器人"第 2 步之后什么都拿不到"，A 的价值被算成 {as_terminal:.2f}，
  而它实际是 {v_star[0]:.2f}。差掉的正是视野之外的那部分。

  规则只有一条：只有 terminated 才把后面的价值当成 0；truncated 时照常用价值补上。
  "前几步用真实奖励，之后用 V 补上"其实就是第 04 课的 n 步回报。
  这条规则在第 03 课落到 TD 目标的公式里，在第 06 课落到 DQN 的代码里
  （CartPole-v1 撑满 500 步是 truncated，杆倒了才是 terminated）。
    """)


# ==================== 第六部分：练习与思考 ====================


class SlipperyCorridor(Corridor):
    """练习 3 用：地面打滑，每一步有 slip 的概率朝相反方向走。"""

    def __init__(self, slip, rng):
        super().__init__()
        self.slip = slip
        self.rng = rng

    def step(self, action):
        if self.rng.random() < self.slip:
            action = 1 - action
        return super().step(action)


def exercises():
    """练习题与参考答案。答案里的数字都是下面的代码现算的。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    env = Corridor()
    rng = np.random.default_rng(SEED)

    # 练习 1：γ = 0.5
    ex1 = discounted_returns([-1.0, -1.0, 10.0], 0.5)
    # 练习 2：一直向左，永远到不了充电桩（回合在 500 步处被掐断）
    always_left = np.tile([1.0, 0.0], (env.n_states, 1))
    ex2, _ = estimate_value_by_rollouts(env, always_left, GAMMA, n_episodes=1, rng=rng)
    # 练习 3：打滑的走廊
    n_ex3 = 300 if is_quick() else 3000
    slippery = SlipperyCorridor(slip=0.2, rng=np.random.default_rng(SEED + 1))
    always_right = np.tile([0.0, 1.0], (env.n_states, 1))
    ex3, ex3_returns = estimate_value_by_rollouts(slippery, always_right, GAMMA, n_episodes=n_ex3, rng=rng)
    ex3_se = ex3_returns.std(axis=1) / np.sqrt(n_ex3)

    print(f"""
练习 1：换一个折扣
    把 γ 从 0.9 改成 0.5，"一直向右"的 V(A)、V(B)、V(C) 各是多少？先手算，再调用 discounted_returns() 验证。

练习 1 参考答案：
    G_2 = 10，G_1 = -1 + 0.5×10 = 4，G_0 = -1 + 0.5×4 = 1。
    程序算出来的：V = ({ex1[0]:.1f}, {ex1[1]:.1f}, {ex1[2]:.1f})。
    γ 变小后，远处的 +10 被打了更重的折扣，离充电桩越远的状态价值掉得越多。

练习 2：一个永远到不了终点的策略
    "一直向左"的价值是多少？这个策略的回合不会结束，run_episode() 会在 500 步处掐断。
    这里把掐断当成终止来算，误差有多大？

练习 2 参考答案：
    每一步都是 -1，回报是 -(1 + γ + γ² + …) = -1/(1-γ) = -10，三个状态都一样
    （从 B、C 出发先往左走几步，奖励同样是 -1）。
    程序算出来的：V = ({ex2[0]:.4f}, {ex2[1]:.4f}, {ex2[2]:.4f})。
    掐断丢掉的部分是 γ^500 × 10，约 {GAMMA**500 * 10:.1e}，可以忽略。
    所以"掐断当终止"的误差取决于 γ^(步数上限)：上限远大于有效视野时没问题，
    上限和有效视野差不多时（第七部分的例子）就错得离谱。

练习 3：地面打滑
    SlipperyCorridor 里每一步有 20% 的概率朝相反方向走。
    用 estimate_value_by_rollouts() 估计"一直向右"在这个环境里的价值（{n_ex3} 个回合）。

练习 3 参考答案：
    程序算出来的：V ≈ ({ex3[0]:.2f}, {ex3[1]:.2f}, {ex3[2]:.2f})，
    标准误差约 ({ex3_se[0]:.2f}, {ex3_se[1]:.2f}, {ex3_se[2]:.2f})。不打滑时是 (6.20, 8.00, 10.00)。
    策略没变，环境的转移变成随机的，价值跟着变了。期望是对"策略的随机性"和
    "环境的随机性"一起取的。注意 V(C) 也不再是 10：在 C 向右也可能滑回 B。

思考题 1：V 和 Q，决策时哪个更方便？

思考题 1 参考答案：
    有 Q 的话，取 argmax_a Q(s, a) 就能选动作，不需要别的东西。
    只有 V 的话，要比较动作得做一步前瞻（第六部分），需要知道每个动作通向哪个状态，也就是需要模型。
    所以无模型的方法（第 05、06 课的 Q-learning、DQN）学 Q；
    Dreamer 手上有世界模型，又另有一个 actor 选动作，critic 学 V 就够了。

思考题 2：第五部分是"从每个状态各自出发跑很多回合"。如果只能从 A 出发，还能估计 V(B)、V(C) 吗？

思考题 2 参考答案：
    能。一条从 A 出发的轨迹会经过 B 和 C，从经过 B 的那一步往后看，就是一条从 B 出发的轨迹
    （马尔可夫性保证了"之后会怎样"和"怎么来到 B"无关）。
    discounted_returns() 返回的是轨迹上每一步的回报，正好对应每个经过的状态。
    第 03 课的蒙特卡洛方法就是这么用的。

思考题 3：既然"按定义硬算"就能得到价值，为什么还需要后面几节的 Bellman 方程和 TD？

思考题 3 参考答案：
    三个原因。一是慢：误差按 1/√n 下降，精度提高 10 倍要 100 倍的回合。
    二是要等回合结束才能算回报，回合很长或不会结束的任务用不了。
    三是它没有利用状态之间的关系：V(A) 和 V(B) 明明满足 V(A) = -1 + 0.9 V(B)（一直向右时），
    硬算却把它们当成两个互不相干的数分别估计。Bellman 方程说的正是这个关系。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()
    returns_by_hand()
    discount_and_horizon()
    v_star = values_by_definition()

    n_max, n_repeats = (1000, 2) if is_quick() else (10000, 5)
    convergence_experiment(n_max, n_repeats)

    planning_with_value(v_star)
    truncation_demo(v_star)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：02-dynamic-programming.py（Bellman 方程与动态规划）")
    print("=" * 60)


if __name__ == "__main__":
    main()
