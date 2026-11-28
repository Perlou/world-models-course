"""
蒙特卡洛树搜索（MCTS）从零实现：从 UCB1 到 PUCT
===============================================

学习目标：
    1. 核对 CONCEPT.md 第 2.3–2.4 节手算例子：UCB1（甲 1.359 vs 乙 1.917）与 MuZero PUCT 系数（N=800 → 1.290）
    2. 从零实现井字棋环境 TicTacToe 与最优极小极大（Minimax）对手，作为 MCTS 和 AlphaZero 的精确裁判
    3. 实现 MCTS 全部四步：ucb1_score()、puct_score()、select_child()、expand()、backup()、run_mcts()
    4. 在井字棋上扫描模拟次数（5, 15, 30, 80, 200），对比随机模拟版 UCB1 与均匀先验 PUCT 对随机对手的胜率

核心概念：
    - 四步循环：选择（Selection）→ 扩展（Expansion）→ 模拟/评估（Simulation）→ 回传（Backup）
    - UCB1 / UCT：面对不确定时保持乐观，Q + c √(ln N / n)
    - PUCT：引入先验策略 P(s, a)，Q + c_puct · P · √N / (1 + n)，成为 AlphaZero 与 MuZero 的核心搜索规则

前置知识：
    - 01-decoder-free-and-value-equivalence.py

对应概念文档：
    - CONCEPT.md 第 2 节「蒙特卡洛树搜索（MCTS）」

预计运行时间：约 20 秒（CPU）
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-07")
SEED = 0


# ==================== 第一部分：井字棋环境与最优 Minimax 裁判 ====================


WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


class TicTacToe:
    """井字棋（3×3）：board 中 +1 表示先手 X，-1 表示后手 O，0 表示空格；player 表示当前轮到谁落子。"""

    def __init__(self, board=None, player=1):
        self.board = np.zeros(9, dtype=np.int8) if board is None else np.array(board, dtype=np.int8, copy=True)
        self.player = int(player)

    def clone(self):
        return TicTacToe(self.board, self.player)

    def legal_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, action):
        assert self.board[action] == 0
        nxt = self.clone()
        nxt.board[action] = self.player
        nxt.player = -self.player
        return nxt

    def winner(self):
        """返回 +1（X 胜）、-1（O 胜）、0（和棋）或 None（未结束）。"""
        b = self.board
        for i, j, k in WIN_LINES:
            s = int(b[i]) + int(b[j]) + int(b[k])
            if s == 3:
                return 1
            if s == -3:
                return -1
        if 0 not in b:
            return 0
        return None

    def is_terminal(self):
        return self.winner() is not None

    def encode_canonical(self):
        """返回 [2, 3, 3] 张量：通道 0 为当前走棋方棋子，通道 1 为对手棋子（CONCEPT.md 第 3.6 节）。"""
        me = (self.board == self.player).astype(np.float32).reshape(3, 3)
        opp = (self.board == -self.player).astype(np.float32).reshape(3, 3)
        return np.stack([me, opp], axis=0)


_MINIMAX_CACHE = {}


def minimax_value(state):
    """带记忆化的精确极小极大值（从当前走棋方视角：赢 +1，和 0，输 -1）。"""
    key = (state.board.tobytes(), state.player)
    if key in _MINIMAX_CACHE:
        return _MINIMAX_CACHE[key]
    w = state.winner()
    if w is not None:
        val = float(w * state.player)
        _MINIMAX_CACHE[key] = val
        return val
    best = -2.0
    for a in state.legal_actions():
        v = -minimax_value(state.step(a))
        if v > best:
            best = v
            if best == 1.0:
                break
    _MINIMAX_CACHE[key] = best
    return best


def minimax_action(state, rng=None):
    """返回所有最优极小极大动作中的一个（保证永不输棋）。"""
    best_v, best_acts = -2.0, []
    for a in state.legal_actions():
        v = -minimax_value(state.step(a))
        if v > best_v:
            best_v, best_acts = v, [a]
        elif v == best_v:
            best_acts.append(a)
    return int(best_acts[0] if rng is None else rng.choice(best_acts))


# ==================== 第二部分：UCB1、PUCT 与完整 MCTS 实现 ====================


def ucb1_score(q_val, n_child, n_parent, c=math.sqrt(2.0)):
    if n_child == 0:
        return float("inf")
    return q_val + c * math.sqrt(math.log(max(1, n_parent)) / n_child)


def puct_score(q_val, prior, n_child, n_parent, c_puct=1.25):
    return q_val + c_puct * prior * math.sqrt(max(1, n_parent)) / (1 + n_child)


class MCTSNode:
    def __init__(self, state, prior=1.0):
        self.state = state
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action -> MCTSNode

    @property
    def q_value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def is_expanded(self):
        return len(self.children) > 0


def select_child(node, rule="puct", c=1.25):
    """从父节点视角选子节点：注意子节点 q_value 是对手视角，所以利用项取 -child.q_value！"""
    n_parent = max(1, sum(ch.visit_count for ch in node.children.values()))
    best_score, best_a, best_child = -float("inf"), None, None
    for a, ch in node.children.items():
        q_from_parent = -ch.q_value
        if rule == "ucb1":
            sc = ucb1_score(q_from_parent, ch.visit_count, n_parent, c=c)
        else:
            sc = puct_score(q_from_parent, ch.prior, ch.visit_count, n_parent, c_puct=c)
        if sc > best_score:
            best_score, best_a, best_child = sc, a, ch
    return best_a, best_child


def expand(node, priors=None):
    legal = node.state.legal_actions()
    if priors is None:
        p = 1.0 / len(legal)
        for a in legal:
            node.children[a] = MCTSNode(node.state.step(a), prior=p)
    else:
        for a in legal:
            node.children[a] = MCTSNode(node.state.step(a), prior=float(priors[a]))


def random_rollout(state, rng):
    """随机模拟到终局，返回从 state 当前走棋方视角的结果 (+1 / 0 / -1)。"""
    cur = state
    root_player = state.player
    while not cur.is_terminal():
        a = int(rng.choice(cur.legal_actions()))
        cur = cur.step(a)
    return float(cur.winner() * root_player)


def backup(path, leaf_value):
    """沿搜索路径回传价值：双人零和博弈每往上一层符号取反！"""
    v = float(leaf_value)
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += v
        v = -v


def run_mcts(root_state, n_simulations=50, rule="puct", c=1.25, eval_fn=None, rng=None,
             dirichlet_alpha=None, dirichlet_eps=0.25):
    """通用 MCTS 搜索：eval_fn=None 时退化为经典随机模拟 MCTS；提供 eval_fn(state)->(priors, val) 即成为 AlphaZero 搜索！"""
    if rng is None:
        rng = np.random.default_rng(SEED)
    root = MCTSNode(root_state)

    if eval_fn is None:
        expand(root, priors=None)
    else:
        priors, _ = eval_fn(root_state)
        expand(root, priors=priors)

    if dirichlet_alpha is not None and len(root.children) > 0:
        acts = list(root.children.keys())
        noise = rng.dirichlet([dirichlet_alpha] * len(acts))
        for a, n_val in zip(acts, noise):
            root.children[a].prior = (1.0 - dirichlet_eps) * root.children[a].prior + dirichlet_eps * float(n_val)

    for _ in range(n_simulations):
        node = root
        path = [node]
        while node.is_expanded() and not node.state.is_terminal():
            _, node = select_child(node, rule=rule, c=c)
            path.append(node)

        w = node.state.winner()
        if w is not None:
            leaf_val = float(w * node.state.player)
        elif eval_fn is None:
            expand(node, priors=None)
            leaf_val = random_rollout(node.state, rng)
        else:
            priors, leaf_val = eval_fn(node.state)
            expand(node, priors=priors)

        backup(path, leaf_val)

    counts = np.zeros(9, dtype=np.float64)
    for a, ch in root.children.items():
        counts[a] = ch.visit_count
    return counts, root.q_value


# ==================== 第三部分：手算核对与对比实验 ====================


def verify_hand_calculations():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 2.3–2.4 节 UCB1 与 PUCT 手算例子")
    print("=" * 60)
    u_a = ucb1_score(0.60, 8, 10, c=math.sqrt(2.0))
    u_b = ucb1_score(0.40, 2, 10, c=math.sqrt(2.0))
    print(f"  1. UCB1（t=10）：甲(n=8, x̄=0.60) = {u_a:.3f}（手算 1.359）｜ 乙(n=2, x̄=0.40) = {u_b:.3f}（手算 1.917）→ 选乙 ✓")
    assert abs(u_a - 1.359) < 2e-3 and abs(u_b - 1.917) < 2e-3 and u_b > u_a

    c1, c2, N = 1.25, 19652.0, 800.0
    mu_c = c1 + math.log((N + c2 + 1.0) / c2)
    print(f"  2. MuZero PUCT 系数（N=800, c1=1.25, c2=19652）：{mu_c:.4f}（手算 1.290 ✓）")
    assert abs(mu_c - 1.290) < 2e-3


def play_match(n_games, rule, n_sims, c_val, seed=0):
    """MCTS 执先/后各一半对阵随机对手，统计胜/和/负率。"""
    rng = np.random.default_rng(seed)
    wins, draws, losses = 0, 0, 0
    for g in range(n_games):
        state = TicTacToe()
        mcts_player = 1 if g % 2 == 0 else -1
        while not state.is_terminal():
            if state.player == mcts_player:
                counts, _ = run_mcts(state, n_simulations=n_sims, rule=rule, c=c_val, rng=rng)
                a = int(np.argmax(counts))
            else:
                a = int(rng.choice(state.legal_actions()))
            state = state.step(a)
        w = state.winner()
        if w == mcts_player:
            wins += 1
        elif w == 0:
            draws += 1
        else:
            losses += 1
    return wins / n_games, draws / n_games, losses / n_games


def run_benchmark():
    print("\n" + "=" * 60)
    print("第二部分：井字棋随机模拟版 MCTS — UCB1 vs 均匀先验 PUCT 随模拟次数的胜率")
    print("=" * 60)
    quick = is_quick()
    sim_list = [5, 25, 80] if quick else [5, 15, 30, 80, 200]
    n_games = 16 if quick else 60

    ucb_res, puct_res = [], []
    print(f"  {'模拟次数':<10} {'UCB1 胜/和/负':>22} {'PUCT(均匀先验) 胜/和/负':>26}")
    for n_sim in sim_list:
        w1, d1, l1 = play_match(n_games, "ucb1", n_sim, math.sqrt(2.0), seed=SEED + n_sim)
        w2, d2, l2 = play_match(n_games, "puct", n_sim, 1.5, seed=SEED + 100 + n_sim)
        ucb_res.append((w1, d1, l1))
        puct_res.append((w2, d2, l2))
        print(f"  {n_sim:<10} {w1*100:5.1f}% / {d1*100:4.1f}% / {l1*100:4.1f}%      "
              f"{w2*100:5.1f}% / {d2*100:4.1f}% / {l2*100:4.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(sim_list, [r[0] * 100 for r in ucb_res], "o-", linewidth=2.2, label="UCB1 胜率")
    axes[0].plot(sim_list, [r[0] * 100 for r in puct_res], "s--", linewidth=2.2, label="PUCT (均匀先验) 胜率")
    axes[0].set_xlabel("每步 MCTS 模拟次数")
    axes[0].set_ylabel("对随机对手胜率 (%)")
    axes[0].set_title("随模拟次数增加，胜率迅速逼近上限")
    axes[0].legend(fontsize=9)

    axes[1].plot(sim_list, [(r[0] + r[1]) * 100 for r in ucb_res], "o-", linewidth=2.2, label="UCB1 不败率 (胜+和)")
    axes[1].plot(sim_list, [(r[0] + r[1]) * 100 for r in puct_res], "s--", linewidth=2.2, label="PUCT 不败率 (胜+和)")
    axes[1].set_xlabel("每步 MCTS 模拟次数")
    axes[1].set_ylabel("不败率 (%)")
    axes[1].set_title("80 次以上模拟几乎对随机对手零败绩")
    axes[1].legend(fontsize=9)
    save_figure(fig, OUT / "02-mcts-tictactoe.png")


def main():
    seed_everything(SEED)
    verify_hand_calculations()
    run_benchmark()
    print("\n" + "=" * 60)
    print("本节完成！下一节：03-alphazero-mini.py（策略/价值网络 + MCTS 自我对弈学会井字棋）")
    print("=" * 60)


if __name__ == "__main__":
    main()
