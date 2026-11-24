"""
Dyna-Q：用模型生成经验
======================

学习目标：
    1. 说出 Dyna 的三个环节：直接强化学习、模型学习、规划（用模拟经验做同样的更新）
    2. 实现表格模型 TabularModel 和 dyna_q()，看到它只是在 Q-learning 上加了两段
    3. 用实验看清"用计算换样本"：规划次数 n_p 越大，需要的真实交互越少
    4. 看到模型过时的后果：通路被堵能适应，出现近路却很难发现
    5. 分清后台规划（Dyna、Dreamer）与决策时规划（Phase 1 的 MPC），知道 Dyna 怎样演变成 MBPO 和 Dreamer

核心概念：
    - Dyna 架构：真实经验既更新价值，又用来学模型；模型再生成模拟经验来更新价值
    - 后台规划（background planning）与决策时规划（decision-time planning）
    - 规划次数 n_p：每个真实步之后做多少次模拟更新（Dreamer 里叫训练比）
    - "在想象中学习"的雏形：从真实访问过的状态出发，让模型走一步，再用价值自举

前置知识：
    - 05-q-learning.py：Q-learning 的更新、ε-greedy、离策略
    - Phase 1：学一个模型，用模型做规划（MPC）

对应概念文档：
    - CONCEPT.md 第 10 节「Dyna：用模型生成经验」

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
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
# 这一节全是表格运算，用 NumPy 在 CPU 上算


# ==================== 第一部分：把模型接回来 ====================


def introduction():
    """Dyna 的想法和三个环节。"""
    print("=" * 60)
    print("第一部分：把模型接回来")
    print("=" * 60)

    print("""
第 03–07 课的算法都没有用模型。它们有一个共同的毛病：奖励的信息每次更新只往回传一步。
走廊里机器人第一次碰到充电桩之后，只有 C 知道这件事；要让 A 也知道，还得再实际走两趟。

可是第一趟走完时，机器人已经见过"A 向右到 B""B 向右到 C""C 向右拿 +10"。
把这几件事在脑子里串一遍，就能推出 A 的价值，根本不用再走。棋手管这个叫复盘。

Dyna（Sutton 1991）把这个想法做成了算法：

                            ┌────────────────────┐
              ┌───────────▶ │   价值函数 / 策略     │ ◀───────────┐
              │             └────────────────────┘             │
        ① 直接强化学习                 │ 选动作                ③ 规划
        用真实经验更新                  ▼                  用模拟经验做同样的更新
              │             ┌────────────────────┐             │
              │             │        环境         │             │
              │             └────────────────────┘             │
      ┌───────────────┐               │              ┌───────────────┐
      │    真实经验     │ ◀─────────────┘              │     模型       │
      │ (s, a, r, s')  │ ──────── ② 模型学习 ────────▶ │ (s,a) → (r,s') │
      └───────────────┘                              └───────────────┘

真实交互很贵，计算很便宜。Dyna 用计算换样本：每一步真实经验，配上 n_p 步模拟经验。

注意模型在什么时候被调用，这是它和 Phase 1 的根本区别：

    ┌────────────────┬──────────────────────────┬──────────────────────────┐
    │                │ 决策时规划（Phase 1 的 MPC）│ 后台规划（Dyna、Dreamer）  │
    ├────────────────┼──────────────────────────┼──────────────────────────┤
    │ 什么时候用模型   │ 要选动作的那一刻            │ 训练的时候，和选动作无关    │
    │ 算出来的是      │ 当前这一个状态下的动作       │ 对很多状态都有效的价值函数  │
    │ 结果留下来吗    │ 用完就丢，下一步重新算       │ 留在 Q 表或网络参数里      │
    │ 决策的速度      │ 慢                        │ 快（查表或一次前向）       │
    └────────────────┴──────────────────────────┴──────────────────────────┘
    """)


# ==================== 第二部分：Dyna-Q ====================


def epsilon_greedy(q_values, epsilon, rng):
    """ε-greedy，并列时随机挑一个（迷宫里 Q 一开始全是 0，不随机挑的话会一直朝同一个方向走）。"""
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    best = np.flatnonzero(q_values == q_values.max())
    return int(best[rng.integers(len(best))])


def q_update(Q, s, a, r, s2, terminated, alpha, gamma):
    """一步 Q-learning 更新。真实的转移和模型生成的转移用的是同一个函数。"""
    target = r + gamma * (0.0 if terminated else Q[s2].max())
    Q[s, a] += alpha * (target - Q[s, a])


class TabularModel:
    """表格模型：记下每个试过的 (s, a) 最近一次得到的 (r, s', terminated)。只适用于确定性环境。"""

    def __init__(self, n_states, n_actions):
        self.next_state = np.zeros((n_states, n_actions), dtype=np.int64)  # [S, A]
        self.reward = np.zeros((n_states, n_actions), dtype=np.float32)  # [S, A]
        self.terminated = np.zeros((n_states, n_actions), dtype=bool)  # [S, A]
        self.visited = np.zeros((n_states, n_actions), dtype=bool)  # [S, A]，这个 (s, a) 试过没有
        self.pairs = []  # 试过的 (s, a) 的列表，规划时从里面抽

    def update(self, s, a, r, s2, terminated):
        """② 模型学习：对表格模型来说就是把这条转移记下来（覆盖旧的记录）。"""
        if not self.visited[s, a]:
            self.visited[s, a] = True
            self.pairs.append((s, a))
        self.next_state[s, a], self.reward[s, a], self.terminated[s, a] = s2, r, terminated

    def sample(self, n, rng):
        """随机抽 n 个试过的 (s, a)，返回模型对它们的预测 [(s, a, r, s', terminated), …]。"""
        out = []
        for i in rng.integers(len(self.pairs), size=n):
            s, a = self.pairs[i]
            out.append((s, a, float(self.reward[s, a]), int(self.next_state[s, a]), bool(self.terminated[s, a])))
        return out


def dyna_q(env, n_planning, alpha, gamma, epsilon, rng, n_episodes=None, total_steps=None, on_step=None):
    """Dyna-Q（CONCEPT.md 第 10.3 节）。n_planning = 0 时就是普通的 Q-learning。

    参数：
        n_planning:  每个真实步之后的规划更新次数 n_p
        n_episodes / total_steps: 跑满多少个回合或多少个真实步就停（给哪个用哪个）
        on_step:     每个真实步之前调用一次 on_step(已走的步数, env)，用来中途改变环境
    返回一个字典：
        Q [S, A]、model、steps_per_episode（每个回合用了多少真实步）、
        cumulative_reward（[真实步数]，到每一步为止的累计奖励）
    """
    Q = np.zeros((env.n_states, env.n_actions))
    model = TabularModel(env.n_states, env.n_actions)
    steps_per_episode, cumulative_reward = [], []
    step, episode_steps, total_reward = 0, 0, 0.0
    s = env.reset()
    while True:
        if on_step is not None:
            on_step(step, env)
        a = epsilon_greedy(Q[s], epsilon, rng)
        s2, r, terminated = env.step(a)
        step += 1
        episode_steps += 1
        total_reward += r
        cumulative_reward.append(total_reward)

        q_update(Q, s, a, r, s2, terminated, alpha, gamma)  # ① 直接强化学习：用真实的转移更新 Q
        model.update(s, a, r, s2, terminated)  # ② 模型学习
        for ps, pa, pr, ps2, pd in model.sample(n_planning, rng):  # ③ 规划：用模型生成的转移做同样的更新
            q_update(Q, ps, pa, pr, ps2, pd, alpha, gamma)

        s = s2
        if terminated:
            steps_per_episode.append(episode_steps)
            episode_steps = 0
            s = env.reset()
            if n_episodes is not None and len(steps_per_episode) >= n_episodes:
                break
        if total_steps is not None and step >= total_steps:
            break
    return {"Q": Q, "model": model, "steps_per_episode": np.array(steps_per_episode),
            "cumulative_reward": np.array(cumulative_reward)}


def by_hand():
    """复现 CONCEPT.md 第 10.4 节的手算例子。"""
    print("\n" + "=" * 60)
    print("第二部分：在三格走廊上手算一遍")
    print("=" * 60)

    gamma, alpha = 0.9, 1.0  # 环境是确定性的，α = 1 时更新就是"直接把 Q 设成目标"
    A, B, C, T, RIGHT = 0, 1, 2, 3, 1
    episode = [(A, RIGHT, -1.0, B, False), (B, RIGHT, -1.0, C, False), (C, RIGHT, 10.0, T, True)]

    # 一个真实回合：① 更新 Q，② 记进模型
    Q = np.zeros((4, 2))
    model = TabularModel(4, 2)
    for s, a, r, s2, d in episode:
        q_update(Q, s, a, r, s2, d, alpha, gamma)
        model.update(s, a, r, s2, d)
    after_real = Q[[A, B, C], RIGHT].copy()
    assert np.allclose(after_real, [-1.0, -1.0, 10.0])

    # ③ 两次规划更新：先抽到 (B, 右)，再抽到 (A, 右)
    for s, a in [(B, RIGHT), (A, RIGHT)]:
        q_update(Q, s, a, float(model.reward[s, a]), int(model.next_state[s, a]), bool(model.terminated[s, a]), alpha, gamma)
    after_planning = Q[[A, B, C], RIGHT].copy()
    assert np.allclose(after_planning, [6.2, 8.0, 10.0])

    # 对比：不做规划，同一条路走三个真实回合
    Q_plain = np.zeros((4, 2))
    plain = []
    for _ in range(3):
        for s, a, r, s2, d in episode:
            q_update(Q_plain, s, a, r, s2, d, alpha, gamma)
        plain.append(Q_plain[[A, B, C], RIGHT].copy())
    assert np.allclose(plain[1], [-1.0, 8.0, 10.0]) and np.allclose(plain[2], [6.2, 8.0, 10.0])

    print(f"""
走廊，Q 表全 0，γ = {gamma}，α = {alpha:.0f}。一个真实回合 A → B → C → T（每一步都向右）。

                                              Q(A, 右)   Q(B, 右)   Q(C, 右)
    普通 Q-learning，第 1 个真实回合后          {plain[0][0]:>6.1f}     {plain[0][1]:>6.1f}     {plain[0][2]:>6.1f}
    普通 Q-learning，第 2 个真实回合后          {plain[1][0]:>6.1f}     {plain[1][1]:>6.1f}     {plain[1][2]:>6.1f}
    普通 Q-learning，第 3 个真实回合后          {plain[2][0]:>6.1f}     {plain[2][1]:>6.1f}     {plain[2][2]:>6.1f}
    Dyna-Q：第 1 个真实回合 + 2 次规划更新      {after_planning[0]:>6.1f}     {after_planning[1]:>6.1f}     {after_planning[2]:>6.1f}

  第 1 个真实回合结束时，只有 Q(C, 右) 反映了充电桩的存在。然后做规划（模型里有 3 条记录）：
      抽到 (B, 右)：模型说得到 -1、到达 C      Q(B, 右) = -1 + 0.9 × max(0, 10) = 8
      抽到 (A, 右)：模型说得到 -1、到达 B      Q(A, 右) = -1 + 0.9 × max(0, 8)  = 6.2
  两次规划更新之后，沿途的 Q 值已经是最优值 (6.2, 8, 10)，没有多走一步真实的路。
  普通 Q-learning 要 3 个真实回合、9 步真实交互；Dyna-Q 用 3 步真实交互加 2 次计算。

  ① 和 ③ 调用的是同一个 q_update()，区别只是转移从哪来。
  这能成立是因为 Q-learning 是离策略的：模拟的转移不是当前策略在真实环境里采的，一步的目标对此不介意。
    """)


# ==================== 第三部分：迷宫实验 ====================


class Maze:
    """Sutton & Barto 第 8 章的迷宫：6 行 9 列，走到 G 得 +1 并结束回合，其余奖励都是 0。

    状态编号 = 行 × 列数 + 列；动作 0 上、1 下、2 左、3 右；撞墙或出界留在原地。
    walls 可以在中途用 set_walls() 换掉，用来模拟环境变化。
    """

    def __init__(self, walls, start, goal, n_rows=6, n_cols=9):
        self.n_rows, self.n_cols = n_rows, n_cols
        self.n_states, self.n_actions = n_rows * n_cols, 4
        self.start, self.goal = start, goal
        self.set_walls(walls)
        self.pos = start

    def set_walls(self, walls):
        self.walls = set(walls)

    def reset(self):
        self.pos = self.start
        return self.pos[0] * self.n_cols + self.pos[1]

    def _move(self, pos, action):
        nxt = (pos[0] + MOVES[action][0], pos[1] + MOVES[action][1])
        inside = 0 <= nxt[0] < self.n_rows and 0 <= nxt[1] < self.n_cols
        return nxt if inside and nxt not in self.walls else pos

    def step(self, action):
        """返回 (s', r, terminated)。"""
        self.pos = self._move(self.pos, action)
        done = self.pos == self.goal
        return self.pos[0] * self.n_cols + self.pos[1], (1.0 if done else 0.0), done

    def shortest_path_length(self):
        """从起点到终点最少要走几步（广度优先搜索）。"""
        dist = {self.start: 0}
        queue = deque([self.start])
        while queue:
            pos = queue.popleft()
            if pos == self.goal:
                return dist[pos]
            for a in range(4):
                nxt = self._move(pos, a)
                if nxt not in dist:
                    dist[nxt] = dist[pos] + 1
                    queue.append(nxt)
        return None

    def greedy_path_length(self, Q, max_steps=200):
        """按 Q 贪心从起点走，返回走到终点用的步数；max_steps 步内没走到返回 None。不改变环境当前的位置。"""
        pos = self.start
        for t in range(1, max_steps + 1):
            pos = self._move(pos, int(np.argmax(Q[pos[0] * self.n_cols + pos[1]])))
            if pos == self.goal:
                return t
        return None


DYNA_MAZE_WALLS = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]


def planning_steps_experiment(n_runs, n_episodes):
    """不同规划次数 n_p 下，每个回合用的真实步数（Sutton & Barto 例 8.1）。"""
    print("\n" + "=" * 60)
    print("第三部分：规划次数与样本效率")
    print("=" * 60)

    alpha, gamma, epsilon = 0.1, 0.95, 0.1
    env = Maze(DYNA_MAZE_WALLS, start=(2, 0), goal=(0, 8))
    optimal = env.shortest_path_length()
    print(f"""
迷宫（Sutton & Barto 例 8.1），S 起点，G 终点，█ 墙：

      ·  ·  ·  ·  ·  ·  ·  █  G
      ·  ·  █  ·  ·  ·  ·  █  ·
      S  ·  █  ·  ·  ·  ·  █  ·
      ·  ·  █  ·  ·  ·  ·  ·  ·
      ·  ·  ·  ·  ·  █  ·  ·  ·
      ·  ·  ·  ·  ·  ·  ·  ·  ·

    走到 G 奖励 +1，其余都是 0；γ = {gamma}，α = {alpha}，ε = {epsilon}。最短路是 {optimal} 步。
    奖励只在终点出现：第一个回合纯靠乱走碰到 G，之后要把这个信息一格一格传回起点。

对比 n_p = 0（普通 Q-learning）、5、50，各跑 {n_episodes} 个回合，重复 {n_runs} 次取平均。
    """)

    plannings = [0, 5, 50]
    steps = {}  # n_p → [n_runs, n_episodes]
    nonzero_after_first = {}  # 第 1 个回合结束时，有多少个 (s, a) 的 Q 值不是 0
    q_after_first = {}
    for n_p in plannings:
        runs = []
        for run in range(n_runs):
            rng = np.random.default_rng(SEED + run)
            runs.append(dyna_q(env, n_p, alpha, gamma, epsilon, rng, n_episodes=n_episodes)["steps_per_episode"])
        steps[n_p] = np.array(runs)
        first = dyna_q(env, n_p, alpha, gamma, epsilon, np.random.default_rng(SEED), n_episodes=1)
        nonzero_after_first[n_p] = int((first["Q"] != 0).sum())
        q_after_first[n_p] = first["Q"]
    assert nonzero_after_first[0] == 1, "不做规划时，第 1 个回合只有走进终点的那一步拿到了非零的目标"

    shown = [e for e in (1, 2, 3, 5, 10, 20, 30, 50) if e <= n_episodes]
    print(f"  每个回合用的真实步数（{n_runs} 次运行的平均）：\n")
    print(f"  {'n_p':<6}" + "".join(f"{f'第 {e} 回合':>10}" for e in shown) + f"{'全部回合的真实步数':>18}{'Q 更新的总次数':>16}")
    totals = {}
    for n_p in plannings:
        mean = steps[n_p].mean(axis=0)
        totals[n_p] = steps[n_p].sum(axis=1).mean()
        print(f"  {n_p:<6}" + "".join(f"{mean[e - 1]:>10.0f}" for e in shown)
              + f"{totals[n_p]:>18.0f}{totals[n_p] * (1 + n_p):>16.0f}")

    def episodes_to_reach(mean_curve, threshold):
        below = np.flatnonzero(mean_curve <= threshold)
        return int(below[0]) + 1 if len(below) else None

    threshold = 2 * optimal
    reach = {n_p: episodes_to_reach(steps[n_p].mean(axis=0), threshold) for n_p in plannings}
    reach_text = "、".join(f"n_p = {n_p} 是{'第 ' + str(reach[n_p]) + ' 个回合' if reach[n_p] else f'{n_episodes} 个回合内没有达到'}"
                          for n_p in plannings)
    print(f"\n  平均步数第一次降到最短路的 2 倍（{threshold} 步）以内：{reach_text}。")
    print(f"  第 1 个回合大家都一样靠乱走（还没有任何奖励信息，规划也无从算起）。差别从第 2 个回合开始。")
    print(f"  第 1 个回合结束时，Q 值不为 0 的 (s, a) 个数：" + "、".join(f"n_p = {n_p} 时 {nonzero_after_first[n_p]} 个" for n_p in plannings) + "。")
    print("  不做规划时只有走进终点的那一步学到了东西；做了规划，同一个回合的经验被反复使用，")
    print("  终点的信息在这一个回合之内就沿着模型里记下的转移往回传开了。")
    if totals[0] > totals[5] > totals[50]:
        print(f"\n  n_p 越大，需要的真实交互越少：全部 {n_episodes} 个回合的真实步数从 {totals[0]:.0f}（n_p = 0）"
              f"降到 {totals[5]:.0f}（n_p = 5）和 {totals[50]:.0f}（n_p = 50）。")
        print(f"  代价是计算：n_p = 50 时 Q 更新的总次数是 n_p = 0 的 {totals[50] * 51 / totals[0]:.1f} 倍。用计算换样本。")
    else:
        print(f"\n  这次实验里真实步数没有随 n_p 单调减少："
              + "、".join(f"n_p = {n_p} 时 {totals[n_p]:.0f}" for n_p in plannings) + "。")
    print("  这个比值后面一直存在，只是换了名字：MBPO 里是每个真实步生成多少条模型数据，Dreamer 里叫训练比。")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2), gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
    for n_p in plannings:
        axes[0].plot(np.arange(2, n_episodes + 1), steps[n_p].mean(axis=0)[1:], linewidth=2, label=f"n_p = {n_p}")
    axes[0].axhline(optimal, color="gray", linestyle=":", label=f"最短路 {optimal} 步")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("回合数（从第 2 个回合画起）")
    axes[0].set_ylabel("这个回合用的真实步数（对数坐标）")
    axes[0].set_title(f"规划次数越多，需要的真实回合越少（{n_runs} 次运行平均）")
    axes[0].legend(fontsize=9)
    for ax, n_p in zip(axes[1:], plannings):
        value = q_after_first[n_p].max(axis=1).reshape(env.n_rows, env.n_cols)
        ax.imshow(value, cmap="viridis", vmin=0, vmax=max(q_after_first[50].max(), 1e-6))
        for r, c in env.walls:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="dimgray"))
        ax.text(env.goal[1], env.goal[0], "G", ha="center", va="center", color="red", fontweight="bold")
        ax.text(env.start[1], env.start[0], "S", ha="center", va="center", color="white", fontweight="bold")
        ax.set_title(f"n_p = {n_p}：第 1 个回合结束时的 max_a Q")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    save_figure(fig, OUT / "08-dyna-planning-steps.png")


# ==================== 第四部分：模型过时了怎么办 ====================


def changing_maze_experiment(n_runs, n_planning):
    """环境中途改变：通路被堵、出现近路（Sutton & Barto 例 8.2、8.3 的设定，这里只用 Dyna-Q）。"""
    print("\n" + "=" * 60)
    print("第四部分：模型过时了怎么办")
    print("=" * 60)

    alpha, gamma, epsilon = 0.5, 0.95, 0.1
    start, goal = (5, 3), (0, 8)
    gap_right = [(3, c) for c in range(0, 8)]  # 第 4 行是一堵墙，只在最右边留一个口
    gap_left = [(3, c) for c in range(1, 9)]  # 口在最左边
    gap_both = [(3, c) for c in range(1, 8)]  # 两边都有口
    tasks = {
        "通路被堵": dict(before=gap_right, after=gap_left, change_at=1000, total=3000),
        "出现近路": dict(before=gap_left, after=gap_both, change_at=3000, total=6000),
    }
    print(f"""
Dyna 的好处来自模型，风险也来自模型。表格模型只记"上一次在这里这样做的结果"，环境变了它不知道。

    通路被堵（第 1000 步改变）                      出现近路（第 3000 步改变）
      之前              之后                         之前              之后
    · · · · · · · · G   · · · · · · · · G          · · · · · · · · G   · · · · · · · · G
    · · · · · · · · ·   · · · · · · · · ·          · · · · · · · · ·   · · · · · · · · ·
    · · · · · · · · ·   · · · · · · · · ·          · · · · · · · · ·   · · · · · · · · ·
    █ █ █ █ █ █ █ █ ·   · █ █ █ █ █ █ █ █          · █ █ █ █ █ █ █ █   · █ █ █ █ █ █ █ ·
    · · · · · · · · ·   · · · · · · · · ·          · · · · · · · · ·   · · · · · · · · ·
    · · · S · · · · ·   · · · S · · · · ·          · · · S · · · · ·   · · · S · · · · ·

设置：Dyna-Q，n_p = {n_planning}，α = {alpha}，γ = {gamma}，ε = {epsilon}，各重复 {n_runs} 次。
    """)

    results = {}
    for name, task in tasks.items():
        env_before = Maze(task["before"], start, goal)
        env_after = Maze(task["after"], start, goal)
        len_before, len_after = env_before.shortest_path_length(), env_after.shortest_path_length()

        def change(step, env, task=task):
            if step == task["change_at"]:
                env.set_walls(task["after"])

        curves, final_lengths, stall = [], [], []
        for run in range(n_runs):
            env = Maze(task["before"], start, goal)
            out = dyna_q(env, n_planning, alpha, gamma, epsilon, np.random.default_rng(SEED + run),
                         total_steps=task["total"], on_step=change)
            cum = out["cumulative_reward"]
            curves.append(cum)
            final_lengths.append(env.greedy_path_length(out["Q"]))
            # 环境改变之后，过了多少步才第一次重新走到终点
            after = np.flatnonzero(cum[task["change_at"]:] > cum[task["change_at"] - 1])
            stall.append(int(after[0]) + 1 if len(after) else None)
        curves = np.array(curves)  # [n_runs, total]
        c, total = task["change_at"], task["total"]
        w = min(500, c)
        rate_before = (curves[:, c - 1] - curves[:, c - 1 - w]).mean() / w  # 改变前 w 步里，平均每步拿多少奖励
        rate_late = (curves[:, -1] - curves[:, -1 - w]).mean() / w  # 最后 w 步
        results[name] = dict(curves=curves, change_at=c, len_before=len_before, len_after=len_after,
                             final_lengths=final_lengths, stall=stall, rate_before=rate_before, rate_late=rate_late)

        reached = [x for x in stall if x is not None]
        n_new_optimal = sum(1 for x in final_lengths if x == len_after)
        n_old_path = sum(1 for x in final_lengths if x == len_before)
        print(f"  【{name}】最短路：改变前 {len_before} 步，改变后 {len_after} 步。")
        print(f"      改变前 {w} 步的平均奖励速度：每 100 步 {100 * rate_before:.2f} 次到达终点（按最短路走的上限是 {100 / len_before:.2f}）")
        print(f"      最后 {w} 步的平均奖励速度：  每 100 步 {100 * rate_late:.2f} 次（改变后的最短路对应的上限是 {100 / len_after:.2f}）")
        if reached:
            print(f"      改变之后第一次重新走到终点：{len(reached)} / {n_runs} 次运行做到了，平均用了 {np.mean(reached):.0f} 步")
        else:
            print(f"      改变之后，{n_runs} 次运行都没能再走到终点")
        print(f"      训练结束时贪心策略走的路：{n_new_optimal} / {n_runs} 次是改变后的最短路（{len_after} 步），"
              f"{n_old_path} / {n_runs} 次是 {len_before} 步的路")

    blocked, shortcut = results["通路被堵"], results["出现近路"]
    print()
    n_runs_recovered = sum(1 for x in blocked["stall"] if x is not None)
    if n_runs_recovered == n_runs and blocked["rate_late"] > 0.5 / blocked["len_after"]:
        print(f"  通路被堵：智能体按旧模型走到原来的口，发现走不通。真实经验会改写模型里那一条记录，")
        print(f"  ε-greedy 的随机动作迟早会碰到新的口，之后规划很快把新路线传开。平均 {np.mean([x for x in blocked['stall'] if x is not None]):.0f} 步之后重新到达终点，")
        print(f"  最后的奖励速度恢复到新最短路上限的 {blocked['rate_late'] * blocked['len_after']:.0%}。模型错得「悲观」时，现实会纠正它。")
    else:
        print(f"  通路被堵：{n_runs_recovered} / {n_runs} 次运行在改变后重新到达过终点，最后的奖励速度是新最短路上限的 "
              f"{blocked['rate_late'] * blocked['len_after']:.0%}。")
    n_found = sum(1 for x in shortcut["final_lengths"] if x == shortcut["len_after"])
    if n_found <= n_runs // 2:
        print(f"  出现近路：{n_runs} 次运行里只有 {n_found} 次在训练结束时走上了近路。旧的路仍然走得通，模型在它记录过的地方没有出错，")
        print("  智能体没有理由怀疑；右边那个口附近的记录还是「撞墙、留在原地」，而它很少再去那里试。")
        print("  模型错得「乐观」会被现实纠正，错在「不知道有更好的」却不会。这需要专门的探索机制")
        print("  （比如给很久没试过的 (s, a) 加奖励，Sutton & Barto 的 Dyna-Q+），本课不展开。")
    else:
        print(f"  出现近路：这次实验里 {n_runs} 次运行有 {n_found} 次在训练结束时走上了近路，")
        print("  靠的是 ε-greedy 的随机动作碰巧走到新开的口。口离常走的路线越远，这种运气越难碰到。")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (name, res) in zip(axes, results.items()):
        steps_axis = np.arange(1, res["curves"].shape[1] + 1)
        ax.plot(steps_axis, res["curves"].mean(axis=0), linewidth=2, label=f"Dyna-Q（n_p = {n_planning}）")
        ax.axvline(res["change_at"], color="gray", linestyle="--", label="环境在这里改变")
        ax.plot(steps_axis, np.where(steps_axis <= res["change_at"], steps_axis / res["len_before"],
                                     res["change_at"] / res["len_before"] + (steps_axis - res["change_at"]) / res["len_after"]),
                "k:", linewidth=1, label="一直走最短路的上限")
        ax.set_xlabel("真实环境步数")
        ax.set_ylabel(f"累计奖励（{n_runs} 次运行平均）")
        ax.set_title(name)
        ax.legend(fontsize=9)
    save_figure(fig, OUT / "08-dyna-changing-maze.png")


# ==================== 第五部分：从 Dyna 到 Dreamer ====================


def road_to_dreamer():
    """Dyna 是一个架构：三个环节各自换成更强的部件，就是后面两章的主角。"""
    print("\n" + "=" * 60)
    print("第五部分：从 Dyna-Q 到 MBPO 和 Dreamer")
    print("=" * 60)

    print("""
Dyna-Q 的设计里藏着两个保守的选择，后来的方法都继承了：

    选择一：模拟只走一步，然后立刻用 Q 自举。模型的输出不会被喂回模型，没有 Phase 1 的复合误差。
    选择二：模拟从真实访问过的 (s, a) 出发。模型只在数据覆盖的区域被调用。

    真实轨迹：  s_0 ───▶ s_1 ───▶ s_2 ───▶ s_3 ───▶ ···
                │        │        │        │
                ▼        ▼        ▼        ▼            从每个真实状态出发，用模型往外走很短的几步
               ŝ        ŝ        ŝ        ŝ

    Dyna-Q：走 1 步          MBPO：走很短的 k 步          Dreamer：在隐空间里走 H 步

Dyna 是一个架构，不是一个具体的算法。三个环节各自换成更强的部件：

    ┌──────────────┬────────────────────┬──────────────────────┬────────────────────────┐
    │              │ Dyna-Q（本节）      │ MBPO（Phase 5）       │ Dreamer（Phase 6）      │
    ├──────────────┼────────────────────┼──────────────────────┼────────────────────────┤
    │ 模型          │ 表格，记住见过的转移 │ 概率神经网络的集成     │ RSSM，在隐空间里预测     │
    │ 模拟从哪出发   │ 访问过的真实状态    │ 回放里的真实状态       │ 回放数据编码成的隐状态   │
    │ 模拟走多长     │ 1 步              │ 很短的 k 步           │ H 步（V3 默认 15）      │
    │ 模拟里谁选动作 │ 以前试过的动作      │ 当前策略              │ 当前 actor             │
    │ 用什么算法学   │ 一步 Q-learning    │ SAC                  │ actor-critic + λ-return │
    └──────────────┴────────────────────┴──────────────────────┴────────────────────────┘

表格 Dyna-Q 和经验回放很像：都是把存下来的转移再用一遍。区别出现在模型能泛化的时候：
学出来的模型可以回答"在见过的状态下，换一个没试过的动作会怎样"，也可以按当前的策略生成新的多步轨迹；
回放缓冲区只能重放旧策略实际做过的事。

到这里，Dreamer 需要的部件本章都见过了：
    模型生成轨迹（本节）→ 在轨迹上算 λ-return（第 04 课）→ critic 回归这个目标并带停止梯度（第 06 课）
    → critic 配一个变化更慢的副本（第 06 课的目标网络）→ 输出用 two-hot 表示（第 07 课）
缺的只有 actor 那一半：怎么直接学一个策略。那是 Phase 5 的内容。
    """)


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题与参考答案。答案里的数字都是下面的代码现算的。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    # 练习 1：随机抽样时，平均要多少次规划更新才能把走廊上的 Q 传到最优
    gamma, alpha = 0.9, 1.0
    transitions = [(0, 1, -1.0, 1, False), (1, 1, -1.0, 2, False), (2, 1, 10.0, 3, True)]
    rng = np.random.default_rng(SEED)
    counts = []
    for _ in range(5000):
        Q = np.zeros((4, 2))
        for s, a, r, s2, d in transitions:  # 一个真实回合
            q_update(Q, s, a, r, s2, d, alpha, gamma)
        n = 0
        while not np.isclose(Q[0, 1], 6.2):
            s, a, r, s2, d = transitions[rng.integers(3)]
            q_update(Q, s, a, r, s2, d, alpha, gamma)
            n += 1
        counts.append(n)
    # 练习 2：α = 1 的表格 Dyna-Q 在随机环境里会怎样——模型只记最后一次的结果
    model = TabularModel(1, 1)
    outcomes = rng.random(1000) < 0.8  # 同一个 (s, a)，80% 的时候奖励 1，20% 的时候奖励 0
    for o in outcomes:
        model.update(0, 0, float(o), 0, True)
    last_reward, true_mean = float(model.reward[0, 0]), 0.8

    print(f"""
练习 1：抽样的顺序
    第二部分的手算里，规划时先抽到 (B, 右) 再抽到 (A, 右)，两次更新就够了。
    如果是从 3 条记录里均匀随机地抽，平均要多少次规划更新，Q(A, 右) 才等于 6.2？

练习 1 参考答案：
    程序模拟 5000 次的平均：{np.mean(counts):.2f} 次（最少 {min(counts)} 次，最多 {max(counts)} 次）。
    理论值是 6：先要抽到 (B, 右)（每次 1/3 的概率，平均 3 次），之后再抽到 (A, 右)（又是平均 3 次）。
    顺序反了的更新是白做的：先抽到 (A, 右) 时 Q(B, 右) 还是 -1，Q(A, 右) 不会变。
    从价值刚发生变化的状态往回推（优先扫描，Sutton & Barto 第 8.4 节）可以省掉这些无效更新。

练习 2：表格模型遇到随机环境
    同一个 (s, a)，80% 的时候奖励是 1，20% 的时候是 0。TabularModel 观察 1000 次之后，它预测的奖励是多少？

练习 2 参考答案：
    程序算出来的：模型记下的是 {last_reward:.0f}（最后一次观察到的值），而期望奖励是 {true_mean}。
    只记最后一次结果的模型把偶然的结果当成了必然。随机环境里模型要么记录每种结果出现的次数，
    要么像 Phase 1 那样学一个概率模型。Dreamer 的 RSSM 输出的就是分布。

练习 3：把规划改成多步
    dyna_q() 的规划只让模型走一步。改成"从抽到的 (s, a) 出发，按 ε-greedy 在模型里连走 k 步，每一步都做 Q 更新"，
    会遇到什么新问题？

练习 3 参考答案：
    第二步起，模型会被问到"在 s' 做某个动作"，而这个 (s', a') 可能从来没试过，表格模型答不上来。
    换成能泛化的神经网络模型就能回答，但答案有误差，而且误差会沿着展开累积（Phase 1 的复合误差）。
    所以多步展开的长度要按模型的可靠程度来定：MBPO 只走很短的几步，Dreamer 靠隐空间模型走到 15 步。
    本章的实战项目 Dyna-DQN 会在 CartPole-v1 上碰到这个问题。

思考题 1：模型数据是"白来的"，规划次数是不是越多越好？

思考题 1 参考答案：
    模型准的时候是。模型不准时，更多的规划更新只是让 Q 更彻底地拟合一个错误的世界；
    而且它会和上一节的过估计叠加：模型在某处预测得偏好 → 那里的 Q 偏高 → 目标里的 max 专挑它。
    这就是 Phase 1 "模型被钻空子"在 Dyna 架构里的版本。

思考题 2：Dyna 的"规划"和 Phase 1 的 MPC 的"规划"，各自适合什么场合？

思考题 2 参考答案：
    MPC 在决策时搜索，结果用完就丢：决策慢，但换一个目标（换奖励函数）立刻就能用，不用重新训练。
    Dyna 在后台把模型里的信息"编译"进价值函数：决策快，但适应新目标要靠再训练。
    两者可以结合：短视野的规划加上学到的价值函数，这是 Phase 7 的 TD-MPC2。

思考题 3：Dreamer 的 actor 完全不看真实的转移，只在想象的轨迹上训练。真实数据还有什么用？

思考题 3 参考答案：
    用来学世界模型，并且提供想象的起点（把真实观测编码成隐状态，从那里开始想象）。
    这对应 Dyna 的第 ② 个环节和"模拟从真实访问过的状态出发"这个选择。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()
    by_hand()
    if is_quick():
        planning_steps_experiment(n_runs=3, n_episodes=20)
        changing_maze_experiment(n_runs=3, n_planning=10)
    else:
        planning_steps_experiment(n_runs=20, n_episodes=50)
        changing_maze_experiment(n_runs=20, n_planning=10)
    road_to_dreamer()
    exercises()

    print("\n" + "=" * 60)
    print("本章讲义全部完成！接下来：实战项目 projects/phase-04-dyna-dqn/，然后进入 Phase 5（策略梯度）")
    print("=" * 60)


if __name__ == "__main__":
    main()
