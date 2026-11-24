"""
Bellman 方程与动态规划
======================

学习目标：
    1. 写出 V^π 的 Bellman 期望方程和 V* 的 Bellman 最优方程，并在三格走廊上验证手算结果
    2. 把期望方程当成线性方程组直接解出 V^π（solve_bellman_linear）
    3. 实现迭代策略评估、策略改进、值迭代、策略迭代，核心都是同一行 Q = R + γ·(P @ V)
    4. 用数字验证"每迭代一轮，误差至少缩小到 γ 倍"（压缩性）
    5. 看到值迭代里"奖励的信息从目标格一轮一格地向外传"

核心概念：
    - Bellman 期望方程 / 最优方程：真实价值函数满足的等式（是性质，不是算法）
    - Bellman 算子 T^π、T*：把一组价值代进方程右边，得到新的一组价值
    - 策略评估（policy evaluation）、策略改进（policy improvement）、
      值迭代（value iteration）、策略迭代（policy iteration）
    - 广义策略迭代（Generalized Policy Iteration，GPI）：评估和改进交替进行

前置知识：
    - 上一节（01-returns-and-value-functions.py）：回报、V^π、Q^π、三格走廊
    - NumPy 的矩阵乘法和解线性方程组

对应概念文档：
    - CONCEPT.md 第 3 节「Bellman 方程」、第 4 节「动态规划：模型已知时的精确解法」

预计运行时间：尚未实测（本文件是还没有运行过的草稿，定稿前要完整跑一遍并改写这一行）
"""

import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-04")
SEED = 0
GAMMA = 0.9
LEFT, RIGHT = 0, 1
# 这一节全是表格运算，用 NumPy 在 CPU 上算（float64，方便和手算逐位对照）


# ==================== 第一部分：Bellman 方程 ====================


def introduction():
    """Bellman 方程说了什么，以及模型怎样写成两张表。"""
    print("=" * 60)
    print("第一部分：Bellman 方程——价值之间的关系")
    print("=" * 60)

    print("""
上一节按定义硬算价值：跑很多回合取平均。它把每个状态的价值当成互不相干的数分别估计。
但价值之间是有关系的。对回报的递推式 G_t = r_t + γ·G_{t+1} 两边取期望，就得到

    Bellman 期望方程：
        V^π(s) = Σ_a π(a|s) · [ r(s, a) + γ · Σ_s' p(s'|s, a) · V^π(s') ]
                  └ 按策略选动作 ┘   └ 这一步的奖励 ┘   └ 下一个状态的价值，按转移概率加权 ┘

    Bellman 最优方程：把"按策略加权平均"换成"取最好的动作"
        V*(s)  = max_a    [ r(s, a) + γ · Σ_s' p(s'|s, a) · V*(s') ]

画成图（回溯图，backup diagram）：一个状态的价值由它"下面一层"的价值算出来。

              s                         期望方程：对动作按 π 加权平均
            ╱   ╲                       最优方程：对动作取 max
          a₁     a₂        ← 选哪个动作
         ╱ ╲    ╱ ╲
       s'   s' s'   s'     ← 环境把我带到哪（按转移概率加权平均）

要用这两个方程，必须知道完整的模型：每个 (s, a) 的转移概率和期望奖励。
状态有限时，模型就是两张表：

    P[s, a, s']   形状 [S, A, S]   转移概率 p(s'|s, a)。终止状态不占行，走进终止状态的概率不计入
    R[s, a]       形状 [S, A]      期望奖励 r(s, a)

方程右边方括号里的东西，对所有 (s, a) 一次算完只要一行：

    Q = R + γ · (P @ V)            P @ V 在最后一维上求和，[S, A, S] @ [S] → [S, A]

这一节的所有算法都围绕这一行。
    """)


def corridor_model():
    """三格走廊（CONCEPT.md 第 2.4 节）的模型，写成两张表。

    状态 0、1、2 = A、B、C；动作 0 = 左，1 = 右。
    返回：
        P: [3, 2, 3]，R: [3, 2]
    """
    P = np.zeros((3, 2, 3))
    R = np.full((3, 2), -1.0)
    for s in range(3):
        P[s, LEFT, max(s - 1, 0)] = 1.0  # 在 A 向左：撞墙，留在 A
        if s < 2:
            P[s, RIGHT, s + 1] = 1.0
    R[2, RIGHT] = 10.0  # 从 C 向右走进充电桩。P[2, RIGHT] 整行是 0：终止状态的价值是 0，不用出现在表里
    return P, R


def q_from_v(V, P, R, gamma):
    """一步前瞻：Q(s, a) = r(s, a) + γ Σ_s' p(s'|s, a) V(s')。

    参数：
        V: [S]，P: [S, A, S]，R: [S, A]
    返回：
        Q: [S, A]
    """
    return R + gamma * (P @ V)


def solve_bellman_linear(policy, P, R, gamma):
    """解析法策略评估：把期望方程写成 (I − γ P_π) v = r_π，直接解（CONCEPT.md 第 3.6 节）。

    参数：
        policy: [S, A]，每行和为 1
    返回：
        V: [S]
    """
    r_pi = (policy * R).sum(axis=1)  # [S]，策略下的期望奖励
    P_pi = np.einsum("sa,sat->st", policy, P)  # [S, S]，策略下的状态转移矩阵
    return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)


def bellman_by_hand():
    """复现 CONCEPT.md 第 3.5 节的三个手算例子。"""
    print("\n" + "=" * 60)
    print("第二部分：在三格走廊上验证 Bellman 方程")
    print("=" * 60)

    P, R = corridor_model()
    always_right = np.tile([0.0, 1.0], (3, 1))  # [S, A]
    random_policy = np.full((3, 2), 0.5)

    # 例一："一直向右"
    v_right = solve_bellman_linear(always_right, P, R, GAMMA)
    assert np.allclose(v_right, [6.2, 8.0, 10.0]), "应该和上一节按定义算出来的一致"

    # 例二：随机策略
    r_pi = (random_policy * R).sum(axis=1)
    P_pi = np.einsum("sa,sat->st", random_policy, P)
    v_random = solve_bellman_linear(random_policy, P, R, GAMMA)
    assert np.allclose(P_pi, [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.0]])
    assert np.allclose(r_pi, [-1.0, -1.0, 4.5])
    assert np.allclose(v_random, [-1.42, 0.48, 4.72], atol=0.005), "应该和 CONCEPT.md 第 3.5 节例二的手算一致"
    residual = np.abs((random_policy * q_from_v(v_random, P, R, GAMMA)).sum(axis=1) - v_random).max()

    print(f"""
随机策略（左右各一半）下，模型按策略"压扁"成一个矩阵和一个向量：

    P_π = {P_pi[0]}        r_π = {r_pi}
          {P_pi[1]}
          {P_pi[2]}

    （第三行的和只有 0.5：另外 0.5 的概率走进了充电桩，终止状态的价值是 0，不出现在矩阵里。
      r_π 的第三个分量是 0.5×(-1) + 0.5×10 = 4.5。）

解线性方程组 (I − γ P_π) v = r_π：

    策略                    V(A)      V(B)      V(C)
    一直向右              {v_right[0]:>7.3f}   {v_right[1]:>7.3f}   {v_right[2]:>7.3f}
    随机（左右各一半）    {v_random[0]:>7.3f}   {v_random[1]:>7.3f}   {v_random[2]:>7.3f}

  和 CONCEPT.md 第 3.5 节的手算一致（6.2, 8, 10 和 -1.42, 0.48, 4.72）。
  把解代回期望方程，左右两边的最大差距是 {residual:.1e}：方程成立。
  上一节对随机策略要跑上万个回合取平均，得到的还只是近似值；这里解一次三元一次方程组就得到精确值。
    """)

    # 例三：验证最优方程
    Q = q_from_v(v_right, P, R, GAMMA)
    assert np.allclose(Q, [[4.58, 6.2], [4.58, 8.0], [6.2, 10.0]])
    assert np.allclose(Q.max(axis=1), v_right)
    print("  例三：把 (6.2, 8, 10) 代进最优方程的右边，对每个动作算一步前瞻：\n")
    print(f"  {'状态':<6}{'左：r + γV(s′)':>16}{'右：r + γV(s′)':>16}{'两者取 max':>12}{'V(s)':>8}")
    for s, name in enumerate("ABC"):
        print(f"  {name:<7}{Q[s, LEFT]:>16.2f}{Q[s, RIGHT]:>16.2f}{Q[s].max():>12.2f}{v_right[s]:>8.2f}")
    print("""
  每一行"取 max"都等于 V(s) 本身，所以 (6.2, 8, 10) 满足最优方程，它就是 V*，"一直向右"是最优策略。
  注意 max 是对"这一步的动作"取的。"在所有策略里找最好的"这个巨大的搜索，
  被最优方程化简成了"每个状态各自对动作取一次 max"。
    """)
    return v_random


# ==================== 第二部分：一个大一点的网格世界 ====================

GRID_MAP = [
    "....#G",
    ".##.#.",
    ".#....",
    ".#.##.",
    "......",
    "S..#..",
]
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
ARROWS = ["↑", "↓", "←", "→"]


class GridWorld:
    """网格世界：走到 G 得 goal_reward 并结束，其余每一步 step_reward；# 是墙，撞墙或出界留在原地。

    slip > 0 时地面打滑：以 slip 的概率不听指挥，改成从四个方向里均匀地随机选一个走。
    模型写成两张表 P [S, A, S] 和 R [S, A]，状态只给能站的格子编号（墙和 G 不算）。
    """

    def __init__(self, grid_map=GRID_MAP, step_reward=-1.0, goal_reward=10.0, slip=0.0):
        self.grid_map = grid_map
        self.n_rows, self.n_cols = len(grid_map), len(grid_map[0])
        self.cells = []  # 状态编号 → (行, 列)
        self.index = {}  # (行, 列) → 状态编号
        self.goal = None
        self.start = 0
        for r, row in enumerate(grid_map):
            for c, ch in enumerate(row):
                if ch == "G":
                    self.goal = (r, c)
                elif ch != "#":
                    self.index[(r, c)] = len(self.cells)
                    self.cells.append((r, c))
                    if ch == "S":
                        self.start = self.index[(r, c)]
        self.n_states, self.n_actions = len(self.cells), 4

        self.P = np.zeros((self.n_states, 4, self.n_states))
        self.R = np.zeros((self.n_states, 4))
        for s, (r, c) in enumerate(self.cells):
            for a in range(4):
                for actual in range(4):  # 实际走的方向
                    prob = (1.0 - slip) * (actual == a) + slip / 4
                    if prob == 0.0:
                        continue
                    nxt = (r + MOVES[actual][0], c + MOVES[actual][1])
                    if nxt == self.goal:
                        self.R[s, a] += prob * goal_reward  # 走进目标：回合结束，这部分概率不计入 P
                    else:
                        if nxt not in self.index:  # 撞墙或出界
                            nxt = (r, c)
                        self.P[s, a, self.index[nxt]] += prob
                        self.R[s, a] += prob * step_reward

    def to_grid(self, values):
        """把按状态编号排的 [S] 数组摆回网格 [行, 列]，墙和目标填 NaN（画图用）。"""
        grid = np.full((self.n_rows, self.n_cols), np.nan)
        for s, (r, c) in enumerate(self.cells):
            grid[r, c] = values[s]
        return grid

    def distances_to_goal(self):
        """每个状态到目标最少要走几步（广度优先搜索，只在不打滑时有意义）。返回 [S]。"""
        dist = np.full(self.n_states, -1)
        queue = deque()
        for a in range(4):  # 先找紧挨着目标的格子
            cell = (self.goal[0] - MOVES[a][0], self.goal[1] - MOVES[a][1])
            if cell in self.index:
                dist[self.index[cell]] = 1
                queue.append(cell)
        while queue:
            r, c = queue.popleft()
            for dr, dc in MOVES:
                cell = (r + dr, c + dc)
                if cell in self.index and dist[self.index[cell]] < 0:
                    dist[self.index[cell]] = dist[self.index[(r, c)]] + 1
                    queue.append(cell)
        return dist

    def render_policy(self, policy):
        """把确定性策略 [S, A]（one-hot）画成箭头，返回若干行字符串。"""
        rows = []
        for r in range(self.n_rows):
            chars = []
            for c in range(self.n_cols):
                if (r, c) == self.goal:
                    chars.append("G")
                elif (r, c) in self.index:
                    chars.append(ARROWS[int(np.argmax(policy[self.index[(r, c)]]))])
                else:
                    chars.append("█")
            rows.append(" ".join(chars))
        return rows


def estimate_value_by_rollouts(P, R, policy, gamma, n_episodes, rng, max_steps=200):
    """按定义估计 V^π：从每个状态出发各跑 n_episodes 个回合，回报取平均。

    和上一节的同名函数是同一个办法，这里所有回合一起推进（批量版本），直接从 P 里抽下一个状态。
    要求奖励是 (s, a) 的确定函数（不打滑的网格满足）。
    回合在 max_steps 处掐断：丢掉的部分不超过 γ^max_steps · r_max/(1-γ)，γ = 0.9 时可以忽略。

    返回：
        returns: [S, n_episodes]，每个回合从起点算起的回报
    """
    S, A, _ = P.shape
    # 把"走进终止状态"补成第 S 列，方便按概率抽下一个状态
    cum_next = np.cumsum(np.concatenate([P, 1.0 - P.sum(axis=2, keepdims=True)], axis=2), axis=2)  # [S, A, S+1]
    cum_policy = np.cumsum(policy, axis=1)  # [S, A]
    state = np.repeat(np.arange(S), n_episodes)  # [S·n]
    alive = np.ones(S * n_episodes, dtype=bool)
    returns = np.zeros(S * n_episodes)
    discount = 1.0
    for _ in range(max_steps):
        u = rng.random(len(state))
        action = np.minimum((u[:, None] >= cum_policy[state]).sum(axis=1), A - 1)  # 按 π 抽动作
        returns += discount * R[state, action] * alive  # 已经结束的回合不再累加
        u = rng.random(len(state))
        nxt = np.minimum((u[:, None] >= cum_next[state, action]).sum(axis=1), S)  # 按 P 抽下一个状态
        alive &= nxt < S
        state = np.where(alive, nxt, 0)  # 结束了的回合随便放在状态 0，反正 alive 会把它屏蔽掉
        discount *= gamma
        if not alive.any():
            break
    return returns.reshape(S, n_episodes)


def analytic_vs_rollouts(n_max):
    """网格世界上，解析解和回合平均的对比。"""
    print("\n" + "=" * 60)
    print("第三部分：换到网格世界——解方程 vs 跑回合")
    print("=" * 60)

    env = GridWorld()
    print(f"""
三格走廊太小，换一个 {env.n_rows}×{env.n_cols} 的网格世界（S 起点，G 目标，█ 墙）：
    """)
    for row in GRID_MAP:
        print("      " + " ".join("█" if ch == "#" else ch for ch in row))
    print(f"""
    状态：{env.n_states} 个能站的格子          动作：上、下、左、右（撞墙或出界留在原地）
    奖励：每走一步 -1；走进 G 得 +10，回合结束          折扣：γ = {GAMMA}

    模型的两张表：P 的形状 {env.P.shape}，R 的形状 {env.R.shape}

先看随机策略（四个方向各 1/4）的价值。两种办法：
    解析解：solve_bellman_linear()，解一个 {env.n_states} 元线性方程组
    回合平均：上一节的办法，从每个状态出发各跑 n 个回合
    """)

    rng = np.random.default_rng(SEED)
    random_policy = np.full((env.n_states, 4), 0.25)
    v_exact = solve_bellman_linear(random_policy, env.P, env.R, GAMMA)
    returns = estimate_value_by_rollouts(env.P, env.R, random_policy, GAMMA, n_max, rng)  # [S, n_max]

    checkpoints = [n for n in (10, 100, 1000, 10000) if n <= n_max]
    gaps = []
    print(f"  {'每个状态的回合数 n':<16}{'与解析解的最大差距':>16}{'平均差距':>12}{'V(起点) 的估计':>16}")
    for n in checkpoints:
        est = returns[:, :n].mean(axis=1)
        gaps.append(np.abs(est - v_exact).max())
        print(f"  {n:<20}{gaps[-1]:>16.3f}{np.abs(est - v_exact).mean():>12.3f}{est[env.start]:>16.3f}")
    print(f"  {'解析解':<20}{'':>16}{'':>12}{v_exact[env.start]:>16.3f}")

    if np.all(np.diff(gaps) < 0):
        print(f"\n  回合数每增加 10 倍，最大差距都在变小：从 n = {checkpoints[0]} 时的 {gaps[0]:.3f} "
              f"降到 n = {checkpoints[-1]} 时的 {gaps[-1]:.3f}。")
    else:
        print(f"\n  这次运行里最大差距没有随回合数单调下降（{'、'.join(f'{g:.3f}' for g in gaps)}），"
              f"不过 n = {checkpoints[-1]} 时的 {gaps[-1]:.3f} 仍比 n = {checkpoints[0]} 时的 {gaps[0]:.3f} "
              f"{'小' if gaps[-1] < gaps[0] else '大'}。")
    print(f"  两种办法互相印证：回合平均不依赖任何理论，它向解析解靠拢，说明 Bellman 方程没有列错。")
    print(f"  代价差得很远：回合平均一共跑了 {env.n_states * n_max} 个回合，解析解只解了一次方程组。")
    lower_bound = -1.0 / (1.0 - GAMMA)  # 永远到不了目标时的回报
    print(f"  随机策略下起点的价值是 {v_exact[env.start]:.2f}，回报的下界（永远到不了目标）是 -1/(1-γ) = {lower_bound:.0f}。")
    if v_exact[env.start] < 0.9 * lower_bound:
        print("  两者很接近：乱走的话，在折扣还没把 +10 打没之前几乎碰不到目标，看到的全是耗电。")
    return env, v_exact


# ==================== 第三部分：迭代策略评估 ====================


def policy_evaluation(policy, P, R, gamma, tol=1e-10, max_iters=100000, trace=None):
    """迭代策略评估：从全 0 出发，反复做 V ← T^π V（CONCEPT.md 第 4.5 节）。

    参数：
        policy: [S, A]，P: [S, A, S]，R: [S, A]
        trace:  传一个列表进来，每一轮算出的 V 会被追加进去（画图、分析用）
    返回：
        V: [S]，约等于 V^π
    """
    V = np.zeros(P.shape[0])
    for _ in range(max_iters):
        Q = q_from_v(V, P, R, gamma)  # [S, A]
        V_new = (policy * Q).sum(axis=1)  # 对动作按 π 加权平均
        if trace is not None:
            trace.append(V_new)
        if np.abs(V_new - V).max() < tol:
            return V_new
        V = V_new
    return V


def contraction_experiment(env, v_exact):
    """验证压缩性：每迭代一轮，误差至少缩小到 γ 倍。"""
    print("\n" + "=" * 60)
    print("第四部分：迭代策略评估——不解方程，反复代入")
    print("=" * 60)

    print("""
解线性方程组的代价是 O(S³)，状态一多就解不动。更不费脑子的办法：
先猜 V = 0，把它代进期望方程的右边，算出来的结果当作新的猜测，反复做。

    V_{k+1} = T^π V_k        （T^π 就是"代进期望方程右边"这个操作）

CONCEPT.md 第 4.3 节证明了每做一次，离真实价值的最大误差至少缩小到 γ 倍：
    ‖V_k − V^π‖∞ ≤ γ^k · ‖V_0 − V^π‖∞
下面在网格世界的随机策略上实测。
    """)

    random_policy = np.full((env.n_states, 4), 0.25)
    trace = []
    v_iter = policy_evaluation(random_policy, env.P, env.R, GAMMA, trace=trace)
    errors = np.array([np.abs(v_exact).max()] + [np.abs(v - v_exact).max() for v in trace])  # errors[k] = ‖V_k − V^π‖
    bounds = GAMMA ** np.arange(len(errors)) * errors[0]
    # 理论保证：实测误差不超过上界（留一点浮点误差的余量）
    assert np.all(errors <= bounds + 1e-9), "压缩性被违反了，说明实现有错"

    print(f"  {'轮数 k':<8}{'实测误差 ‖V_k − V^π‖':>22}{'理论上界 γ^k·‖V_0 − V^π‖':>26}{'相邻两轮误差之比':>18}")
    for k in [0, 1, 2, 5, 10, 20, 50, 100]:
        if k < len(errors):
            ratio = f"{errors[k] / errors[k - 1]:.3f}" if k > 0 and errors[k - 1] > 0 else "—"
            print(f"  {k:<10}{errors[k]:>22.6f}{bounds[k]:>26.6f}{ratio:>18}")
    k_001 = int(np.argmax(errors < 0.01))
    k_theory = int(np.ceil(np.log(0.01 / errors[0]) / np.log(GAMMA)))
    late_ratio = errors[40] / errors[39] if len(errors) > 40 else float("nan")
    print(f"""
  每一轮的实测误差都没有超过理论上界（程序里用 assert 检查了全部 {len(errors) - 1} 轮）。
  误差降到 0.01 以下用了 {k_001} 轮；按上界估算最多需要 ln(0.01/{errors[0]:.2f}) / ln({GAMMA}) ≈ {k_theory} 轮。
  第 40 轮前后，相邻两轮的误差之比是 {late_ratio:.3f}（上界是 γ = {GAMMA}）。
  迭代到相邻两轮的变化小于 1e-10 一共用了 {len(trace)} 轮，结果和解析解的最大差距是 {np.abs(v_iter - v_exact).max():.1e}。
    """)
    return errors, bounds


# ==================== 第四部分：策略改进与值迭代 ====================


def policy_improvement(V, P, R, gamma):
    """策略改进：对 V 做一步前瞻得到 Q，每个状态选 Q 最大的动作。

    返回：
        policy: [S, A]，one-hot 的贪心策略
        Q:      [S, A]
    """
    Q = q_from_v(V, P, R, gamma)
    # 先四舍五入再取 argmax：两个动作一样好时，不让 1e-16 量级的浮点误差决定选哪个
    greedy = np.argmax(np.round(Q, 10), axis=1)
    return np.eye(P.shape[1])[greedy], Q


def value_iteration(P, R, gamma, tol=1e-10, max_iters=100000, trace=None):
    """值迭代：从全 0 出发，反复做 V ← T* V（CONCEPT.md 第 4.5 节）。

    和 policy_evaluation() 只差一行：对动作取 max，而不是按 π 加权平均。

    返回：
        V:      [S]，约等于 V*
        policy: [S, A]，对 V 贪心的策略
    """
    V = np.zeros(P.shape[0])
    for _ in range(max_iters):
        V_new = q_from_v(V, P, R, gamma).max(axis=1)  # 对动作取 max
        if trace is not None:
            trace.append(V_new)
        converged = np.abs(V_new - V).max() < tol
        V = V_new
        if converged:
            break
    policy, _ = policy_improvement(V, P, R, gamma)
    return V, policy


def value_iteration_by_hand(v_random):
    """复现 CONCEPT.md 第 4.4 节的两个手算例子。"""
    print("\n" + "=" * 60)
    print("第五部分：策略改进与值迭代（先在走廊上手算）")
    print("=" * 60)

    P, R = corridor_model()

    # 例二：一次策略改进
    greedy, Q = policy_improvement(v_random, P, R, GAMMA)
    assert np.allclose(Q, [[-2.28, -0.57], [-2.28, 3.25], [-0.57, 10.0]], atol=0.01)
    assert np.array_equal(np.argmax(greedy, axis=1), [RIGHT, RIGHT, RIGHT])
    print("""
策略改进：有了某个策略的价值 V^π，对它做一步前瞻，每个状态改选 Q^π 最大的动作。
从随机策略的价值 (-1.42, 0.48, 4.72) 出发：
    """)
    print(f"  {'状态':<6}{'Q(s, 左)':>10}{'Q(s, 右)':>10}{'  贪心动作'}")
    for s, name in enumerate("ABC"):
        print(f"  {name:<7}{Q[s, LEFT]:>10.2f}{Q[s, RIGHT]:>10.2f}{'    ' + ['左', '右'][int(np.argmax(greedy[s]))]}")
    print("""
  一次改进就得到了"一直向右"。随机策略的价值在数值上离 V* = (6.2, 8, 10) 很远，
  但它已经包含了"越往右越好"这个排序，而贪心只需要排序是对的。
    """)

    # 例一：值迭代
    trace = []
    v_star, policy = value_iteration(P, R, GAMMA, trace=trace)
    assert np.allclose(trace[0], [-1.0, -1.0, 10.0])
    assert np.allclose(trace[1], [-1.9, 8.0, 10.0])
    assert np.allclose(trace[2], [6.2, 8.0, 10.0])
    assert np.allclose(trace[3], [6.2, 8.0, 10.0])
    print("值迭代：不等评估做完，每一轮直接对动作取 max，V ← T* V。从 V = (0, 0, 0) 开始：\n")
    print(f"  {'轮数':<6}{'V(A)':>8}{'V(B)':>8}{'V(C)':>8}")
    print(f"  {0:<8}{0.0:>8.2f}{0.0:>8.2f}{0.0:>8.2f}")
    for k, v in enumerate(trace[:4], start=1):
        print(f"  {k:<8}{v[0]:>8.2f}{v[1]:>8.2f}{v[2]:>8.2f}")
    print(f"""
  和 CONCEPT.md 第 4.4 节的表逐格一致，第 4 轮和第 3 轮相同，收敛到 V*。
  充电桩的 +10 每一轮往左传一格：第 1 轮只有紧挨着的 C 知道，第 2 轮传到 B，第 3 轮传到 A。
  奖励的信息是从它出现的地方一步一步向后传播的。
    """)


def value_iteration_on_grid(env):
    """网格世界上的值迭代：每一轮的变化量、价值向外扩散的过程。"""
    print("\n" + "=" * 60)
    print("第六部分：网格世界上的值迭代")
    print("=" * 60)

    trace = []
    v_star, policy = value_iteration(env.P, env.R, GAMMA, trace=trace)
    history = np.array([np.zeros(env.n_states)] + trace)  # [轮数+1, S]，history[k] = V_k
    deltas = np.abs(np.diff(history, axis=0)).max(axis=1)  # deltas[k-1] = 第 k 轮的最大变化量
    n_exact = (np.abs(history - v_star) < 1e-9).sum(axis=1)  # 每一轮之后，已经等于最终值的格子数
    dist = env.distances_to_goal()  # [S]
    d_max = int(dist.max())
    # 压缩性的直接推论：相邻两轮的变化量，后一轮不超过前一轮的 γ 倍
    assert np.all(deltas[1:] <= GAMMA * deltas[:-1] + 1e-9)

    print(f"\n  {'轮数 k':<8}{'这一轮的最大变化量':>16}{'参考：10·γ^(k-1)':>18}{'已等于最终值的格子数':>20}{'离目标不超过 k 步的格子数':>24}")
    for k in range(1, len(deltas) + 1):
        print(f"  {k:<10}{deltas[k - 1]:>16.4f}{10 * GAMMA ** (k - 1):>18.4f}{n_exact[k]:>20}{int((dist <= k).sum()):>24}")

    spreads_one_cell = all(n_exact[k] == (dist <= k).sum() for k in range(1, len(deltas) + 1))
    geometric = np.allclose(deltas[:d_max], 10 * GAMMA ** np.arange(d_max))
    print(f"\n  离目标最远的格子要走 {d_max} 步。值迭代一共做了 {len(deltas)} 轮，"
          f"第 {int(np.argmax(deltas < 1e-10)) + 1} 轮起不再有变化。")
    if spreads_one_cell:
        print("  每一轮，「已等于最终值的格子数」都正好等于「离目标不超过 k 步的格子数」：")
        print("  目标的信息每一轮向外传一格，传到的格子当轮就得到了最终的值。")
    else:
        print("  「已等于最终值的格子数」和「离目标不超过 k 步的格子数」不完全相同，见上表。")
    if geometric:
        print(f"  前 {d_max} 轮的最大变化量正好是 10·γ^(k-1)：第 k 轮变化最大的是刚被传到的那一圈格子，")
        print(f"  它们的价值里多出了打过 k-1 次折扣的 +10。变化量按 γ 的等比数列下降，传完之后直接变成 0。")
    else:
        print(f"  最大变化量逐轮下降，相邻两轮之比最大为 {np.max(deltas[1:d_max] / deltas[:d_max - 1]):.3f}"
              f"（压缩性保证它不超过 γ = {GAMMA}）。")

    # 打滑的网格：转移是随机的，值迭代不会在有限轮内精确收敛
    slippery = GridWorld(slip=0.2)
    slip_trace = []
    v_slip, policy_slip = value_iteration(slippery.P, slippery.R, GAMMA, trace=slip_trace)
    slip_history = np.array([np.zeros(slippery.n_states)] + slip_trace)
    slip_deltas = np.abs(np.diff(slip_history, axis=0)).max(axis=1)
    assert np.all(slip_deltas[1:] <= GAMMA * slip_deltas[:-1] + 1e-9)
    slip_ratios = slip_deltas[1:] / slip_deltas[:-1]
    k_mid = min(30, len(slip_ratios) - 1)
    # 打滑环境里的最优动作，放回不打滑的环境里还是不是最优的（并列最优也算）
    q_star = q_from_v(v_star, env.P, env.R, GAMMA)  # [S, A]
    slip_actions = np.argmax(policy_slip, axis=1)
    n_changed = int((q_star[np.arange(env.n_states), slip_actions] < q_star.max(axis=1) - 1e-9).sum())
    print(f"""
  对比：让地面打滑（每一步有 20% 的概率随机朝一个方向走）。这时没有哪个格子能"确定地"走到目标，
  值迭代不再在有限轮内精确收敛：变化量降到 1e-10 以下用了 {len(slip_deltas)} 轮（不打滑时是 {len(deltas)} 轮）。
  第 {k_mid + 2} 轮与前一轮的变化量之比是 {slip_ratios[k_mid]:.3f}，全程最大的比值是 {slip_ratios.max():.3f}，都没有超过 γ = {GAMMA}。
  起点的最优价值从不打滑时的 {v_star[env.start]:.3f} 变成 {v_slip[slippery.start]:.3f}。
  打滑环境的最优动作里，有 {n_changed} 个格子的动作放回不打滑的环境里不再是最优的。
    """)

    print("  不打滑时值迭代得到的最优策略（每个格子的贪心动作；有并列时只画其中一个）：\n")
    for row in env.render_policy(policy):
        print("      " + row)

    # 图一：价值逐轮扩散
    rounds = sorted({1, 2, 4, 7, min(d_max, len(trace))})
    fig, axes = plt.subplots(1, len(rounds), figsize=(3.6 * len(rounds), 3.9))
    for ax, k in zip(axes, rounds):
        ax.imshow(env.to_grid(history[k]), cmap="viridis", vmin=-10, vmax=10)  # 墙是 NaN，显示成空白
        for (r, c), s in env.index.items():
            ax.text(c, r, f"{history[k][s]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if history[k][s] < 4 else "black")
        ax.text(env.goal[1], env.goal[0], "G", ha="center", va="center", fontsize=12, color="red", fontweight="bold")
        ax.set_title(f"第 {k} 轮之后的 V")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.suptitle("值迭代：价值从目标格向外逐轮扩散（空白是墙，颜色越亮价值越高）")
    print()
    save_figure(fig, OUT / "02-value-iteration-spread.png")
    return v_star, policy, deltas, slip_deltas


# ==================== 第五部分：策略迭代 ====================


def policy_iteration(P, R, gamma, max_iters=1000):
    """策略迭代：评估当前策略 → 对评估结果贪心 → 再评估……直到策略不再变化。

    评估这一步直接用解析法（solve_bellman_linear）。
    返回：
        V [S]、policy [S, A]、每一轮评估得到的 V 的列表
    """
    S, A, _ = P.shape
    policy = np.full((S, A), 1.0 / A)  # 从随机策略出发
    values = []
    for _ in range(max_iters):
        V = solve_bellman_linear(policy, P, R, gamma)  # 评估
        values.append(V)
        new_policy, _ = policy_improvement(V, P, R, gamma)  # 改进
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return V, policy, values


def policy_iteration_experiment(env, v_star, deltas, slip_deltas, errors, bounds):
    """策略迭代，并把三种迭代的收敛过程画在一起。"""
    print("\n" + "=" * 60)
    print("第七部分：策略迭代与广义策略迭代")
    print("=" * 60)

    v_pi, policy, values = policy_iteration(env.P, env.R, GAMMA)
    gap = np.abs(v_pi - v_star).max()
    # 策略改进定理：每一轮的新策略在每个状态上都不比旧策略差
    monotone = all(np.all(values[i + 1] >= values[i] - 1e-7) for i in range(len(values) - 1))
    assert monotone, "策略改进定理被违反了，说明实现有错"

    print(f"""
策略迭代把"评估"和"改进"交替做到底：

        ┌── 评估：让 V 去贴合当前的 π ──┐
        │                              ▼
      策略 π                         价值 V
        ▲                              │
        └── 改进：让 π 对当前的 V 贪心 ──┘

从随机策略出发，在网格世界上每一轮评估得到的 V(起点)：
    """)
    for i, v in enumerate(values):
        note = "（随机策略）" if i == 0 else ("（策略不再变化，停止）" if i == len(values) - 1 else "")
        print(f"      第 {i + 1} 次评估：V(起点) = {v[env.start]:>8.3f}，所有状态的平均价值 = {v.mean():>8.3f} {note}")
    print(f"""
  一共评估了 {len(values)} 次就停了，得到的价值和值迭代的 V* 最大相差 {gap:.1e}。
  每一轮的新策略在每个状态上的价值都不低于上一轮（策略改进定理，程序里用 assert 检查了）。

  值迭代可以看成"评估只做一轮就改进"的策略迭代。两步都不必做到底，只要持续交替就会一起走向最优，
  这个模式叫广义策略迭代（GPI）。后面的 Q-learning、Actor-Critic、Dreamer 都是它的实例：
      Q-learning：评估 = 一次 TD 更新；          改进 = 目标里的 max
      Dreamer：   评估 = critic 回归 λ-return；  改进 = actor 在想象的轨迹上做梯度更新
    """)

    # 图二：三种迭代的收敛过程
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    k_axis = np.arange(len(errors))
    axes[0].plot(k_axis, errors, linewidth=2, label="实测误差 ‖V_k − V^π‖")
    axes[0].plot(k_axis, bounds, "k--", linewidth=1, label="理论上界 γ^k·‖V_0 − V^π‖")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("轮数 k")
    axes[0].set_ylabel("最大误差（对数坐标）")
    axes[0].set_title("迭代策略评估（随机策略）：误差按 γ^k 下降")
    axes[0].legend(fontsize=9)

    positive = deltas > 0  # 对数坐标画不了 0
    axes[1].plot(np.arange(1, len(deltas) + 1)[positive], deltas[positive], "o-", linewidth=2, label="不打滑的网格")
    axes[1].plot(np.arange(1, len(slip_deltas) + 1), slip_deltas, linewidth=2, label="打滑 20% 的网格")
    ref_k = np.arange(1, len(slip_deltas) + 1)
    axes[1].plot(ref_k, 10 * GAMMA ** (ref_k - 1), "k--", linewidth=1, label="参考线 10·γ^(k-1)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("轮数 k")
    axes[1].set_ylabel("这一轮的最大变化量（对数坐标）")
    axes[1].set_title("值迭代：每一轮的最大变化量")
    axes[1].legend(fontsize=9)
    save_figure(fig, OUT / "02-dp-convergence.png")

    print("""
动态规划有两个限制，本章剩下的内容就是把它们逐个解除：

    限制 ①：需要完整的模型 P 和 R        → 第 03 课：用采样到的一个 s' 代替对所有 s' 的求和（TD）
    限制 ②：每一轮要扫遍所有状态          → 第 03 课：只更新实际访问到的状态；
                                           第 06 课：用神经网络在状态之间泛化（DQN）

这一节的算法完全不和环境交互，只是在已知的模型上做计算，属于"规划"。
它的价值在于给出标准答案：后面每一个从经验里学价值的算法，都拿它的结果来检验。
    """)


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题与参考答案。答案里的数字都是下面的代码现算的。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    P, R = corridor_model()
    # 练习 1：γ = 0.5 的走廊
    ex1_v, ex1_policy = value_iteration(P, R, 0.5)
    ex1_actions = "、".join(["左", "右"][a] for a in np.argmax(ex1_policy, axis=1))
    # 练习 2：把充电桩的奖励降到 +1，折扣 0.5
    R_small = R.copy()
    R_small[2, RIGHT] = 1.0
    ex2_v, ex2_policy = value_iteration(P, R_small, 0.5)
    ex2_actions = "、".join(["左", "右"][a] for a in np.argmax(ex2_policy, axis=1))
    assert ex1_actions == ex2_actions == "右、右、右", "下面两道练习的答案是按「最优策略仍是一直向右」写的"
    # 练习 3：γ 越大收敛越慢
    env = GridWorld(slip=0.2)
    ex3 = {}
    for gamma in (0.5, 0.9, 0.99):
        trace = []
        value_iteration(env.P, env.R, gamma, tol=1e-6, trace=trace)
        ex3[gamma] = len(trace)

    print(f"""
练习 1：换一个折扣
    把走廊的 γ 改成 0.5，用 value_iteration() 求 V* 和最优策略。先手算，再运行验证。

练习 1 参考答案：
    程序算出来的：V* = ({ex1_v[0]:.2f}, {ex1_v[1]:.2f}, {ex1_v[2]:.2f})，最优动作（A、B、C）：{ex1_actions}。
    手算：V*(C) = 10，V*(B) = -1 + 0.5×10 = 4，V*(A) = -1 + 0.5×4 = 1。
    折扣变重后价值变小了，但"越往右越好"的排序没变，最优策略也没变。

练习 2：最优策略一定是"朝目标走"吗？
    在练习 1 的基础上把走进充电桩的奖励从 +10 降到 +1（R[2, RIGHT] = 1），再求一次。

练习 2 参考答案：
    程序算出来的：V* = ({ex2_v[0]:.3f}, {ex2_v[1]:.3f}, {ex2_v[2]:.3f})，最优动作（A、B、C）：{ex2_actions}。
    不管怎么走，没到充电桩之前每一步都是 -1；到了充电桩回合结束，之后不再扣分。
    所以即使终点的奖励很小，尽快走到终点仍然是最优的："结束回合"本身就有价值，因为它让耗电停下来。
    反过来，如果每一步的奖励是正的（比如 CartPole 每撑一步 +1），终止就成了要避免的事。
    奖励的符号和终止条件合在一起，才决定策略该"尽快结束"还是"尽量不结束"。

练习 3：γ 和收敛速度
    在打滑 20% 的网格上，分别用 γ = 0.5、0.9、0.99 做值迭代（容差 1e-6），各需要多少轮？

练习 3 参考答案：
    程序算出来的：γ = 0.5 用了 {ex3[0.5]} 轮，γ = 0.9 用了 {ex3[0.9]} 轮，γ = 0.99 用了 {ex3[0.99]} 轮。
    每一轮误差最多缩小到 γ 倍，γ 越接近 1，价值看得越远，收敛也越慢。
    这个规律对后面基于采样的方法同样成立：γ 大的任务更难学。

思考题 1：值迭代每一轮都要对所有状态做一次更新。如果只更新一部分状态（比如随机挑一个），还能收敛吗？

思考题 1 参考答案：
    能，只要每个状态都会被反复更新到（这叫异步动态规划）。更新的顺序会影响速度：
    从目标格往外一圈一圈更新，一遍就够；从起点往目标更新，前面的更新都是白做。
    "先更新哪些状态最划算"这个问题，在第 08 课的 Dyna 里会再次出现。

思考题 2：policy_iteration() 里的评估用的是解析法。换成只迭代一轮的 policy_evaluation 会怎样？

思考题 2 参考答案：
    评估只做一轮（V ← T^π V）然后立刻贪心，合起来就是 V ← T* V，也就是值迭代。
    评估做 m 轮再改进，介于两者之间（叫改进的策略迭代）。它们都收敛到同一个 V*。

思考题 3：Phase 1–3 学出来的世界模型，能直接拿来做值迭代吗？

思考题 3 参考答案：
    两个障碍。一是值迭代要对 s' 求和（需要完整的转移分布），而神经网络模型通常只能采样出一个 s'；
    二是要扫遍所有状态，连续的状态空间没法枚举。
    解除第一个障碍是第 03 课的 TD（用一个采样代替期望），
    解除第二个是第 06 课的 DQN（用网络代替表格）。两个都解除之后再把模型接回来，
    就是第 08 课的 Dyna 和 Phase 6 的 Dreamer。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()
    v_random = bellman_by_hand()

    n_max = 300 if is_quick() else 10000
    env, v_exact = analytic_vs_rollouts(n_max)
    errors, bounds = contraction_experiment(env, v_exact)

    value_iteration_by_hand(v_random)
    v_star, _, deltas, slip_deltas = value_iteration_on_grid(env)
    policy_iteration_experiment(env, v_star, deltas, slip_deltas, errors, bounds)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：03-monte-carlo-vs-td.py（蒙特卡洛与时序差分）")
    print("=" * 60)


if __name__ == "__main__":
    main()
