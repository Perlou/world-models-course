"""
AlphaZero 迷你版：已知规则下的策略/价值网络 + MCTS 自我对弈
===========================================================

学习目标：
    1. 核对 CONCEPT.md 第 3.3 节的手算例子（访问次数 [60, 30, 10] → 温度 1 与 0.5 的搜索策略；策略 CE 0.970、熵 0.898、KL 0.072、价值 MSE 0.64）
    2. 实现 PolicyValueNet（[B, 2, 3, 3] 规范视角输入 → 掩码策略 logits [B, 9] + tanh 价值 [B]）
    3. 实现 visit_counts_to_policy()、add_dirichlet_noise()、self_play_game() 与 alphazero_loss()
    4. 对比有无根节点 Dirichlet 噪声的学习曲线，并评估训练后对随机对手与完美极小极大对手（Minimax）的战绩

核心概念：
    - 广义策略迭代缓存：MCTS 搜索产出更优的 π^mcts → 蒸馏进策略网络；终局结果 G_t → 监督价值网络
    - 策略损失下限为目标分布的熵 H(π^mcts)，而非 0
    - 根节点 Dirichlet 噪声：(1−ε) p̂ + ε η，保证自我对弈的数据多样性

前置知识：
    - 02-mcts-from-scratch.py（TicTacToe、minimax_action、run_mcts）

对应概念文档：
    - CONCEPT.md 第 3 节「AlphaZero：策略 / 价值网络 + 搜索」

预计运行时间：约 40 秒（CPU）
"""

import importlib.util
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec02 = importlib.util.spec_from_file_location("p7_02", os.path.join(_DIR, "02-mcts-from-scratch.py"))
_mod02 = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(_mod02)

TicTacToe = _mod02.TicTacToe
minimax_action = _mod02.minimax_action
run_mcts = _mod02.run_mcts

plt = setup_matplotlib()
OUT = output_dir("phase-07")
SEED = 0


# ==================== 第一部分：核心网络与工具函数 ====================


class PolicyValueNet(nn.Module):
    """井字棋双头网络：输入 [B, 2, 3, 3] → (policy_logits [B, 9], value [B] ∈ [-1, 1])。"""

    def __init__(self, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(18, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, 9)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy_head(h), torch.tanh(self.value_head(h)).squeeze(-1)


def visit_counts_to_policy(counts, temperature=1.0):
    counts = np.asarray(counts, dtype=np.float64)
    if temperature <= 1e-4:
        pi = np.zeros_like(counts)
        pi[int(np.argmax(counts))] = 1.0
        return pi
    scaled = counts ** (1.0 / temperature)
    return scaled / scaled.sum()


def add_dirichlet_noise(priors, legal_actions, alpha=0.8, eps=0.25, rng=None):
    rng = np.random.default_rng(SEED) if rng is None else rng
    out = np.array(priors, dtype=np.float64, copy=True)
    noise = rng.dirichlet([alpha] * len(legal_actions))
    for a, n_val in zip(legal_actions, noise):
        out[a] = (1.0 - eps) * out[a] + eps * float(n_val)
    return out


def alphazero_loss(policy_logits, values, target_pis, target_vs, legal_masks=None):
    if legal_masks is not None:
        policy_logits = policy_logits.masked_fill(~legal_masks, -1e9)
    log_p = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_pis * log_p).sum(dim=-1).mean()
    value_loss = F.mse_loss(values, target_vs)
    return policy_loss + value_loss, policy_loss.item(), value_loss.item()


def make_net_eval_fn(net):
    @torch.no_grad()
    def eval_fn(state):
        x = torch.from_numpy(state.encode_canonical()[None])
        logits, val = net(x)
        legal = state.legal_actions()
        mask = torch.zeros(9, dtype=torch.bool)
        mask[legal] = True
        logits_m = logits[0].masked_fill(~mask, -1e9)
        priors = F.softmax(logits_m, dim=-1).numpy()
        return priors, float(val.item())
    return eval_fn


def self_play_game(net, n_sims=30, dirichlet_alpha=0.8, dirichlet_eps=0.25, rng=None):
    """用当前网络 + MCTS 自我对弈一盘，返回 [(obs_2x3x3, pi_mcts, G_t, legal_mask), ...]。"""
    net.eval()
    eval_fn = make_net_eval_fn(net)
    state = TicTacToe()
    traj = []
    step_idx = 0
    while not state.is_terminal():
        counts, _ = run_mcts(
            state, n_simulations=n_sims, rule="puct", c=1.25, eval_fn=eval_fn, rng=rng,
            dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
        )
        temp = 1.0 if step_idx < 4 else 0.1
        pi = visit_counts_to_policy(counts, temperature=temp)
        mask = np.zeros(9, dtype=bool)
        mask[state.legal_actions()] = True
        traj.append((state.encode_canonical(), pi.astype(np.float32), state.player, mask))
        a = int(rng.choice(9, p=pi))
        state = state.step(a)
        step_idx += 1

    winner = state.winner()
    samples = []
    for obs, pi, player, mask in traj:
        g = float(winner * player)
        samples.append((obs, pi, g, mask))
    return samples


def evaluate_agent(net, opponent="random", n_games=30, n_sims=25, seed=900):
    net.eval()
    eval_fn = make_net_eval_fn(net)
    rng = np.random.default_rng(seed)
    wins, draws, losses = 0, 0, 0
    for g in range(n_games):
        state = TicTacToe()
        az_player = 1 if g % 2 == 0 else -1
        while not state.is_terminal():
            if state.player == az_player:
                counts, _ = run_mcts(state, n_simulations=n_sims, rule="puct", c=1.25, eval_fn=eval_fn, rng=rng)
                a = int(np.argmax(counts))
            elif opponent == "minimax":
                a = minimax_action(state, rng=rng)
            else:
                a = int(rng.choice(state.legal_actions()))
            state = state.step(a)
        w = state.winner()
        if w == az_player:
            wins += 1
        elif w == 0:
            draws += 1
        else:
            losses += 1
    return wins / n_games, draws / n_games, losses / n_games


# ==================== 第二部分：手算核对与训练对照 ====================


def verify_hand_calculations():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 3.3 节搜索策略与损失手算例子")
    print("=" * 60)
    counts = np.array([60, 30, 10])
    pi1 = visit_counts_to_policy(counts, temperature=1.0)
    pi05 = visit_counts_to_policy(counts, temperature=0.5)
    p_hat = np.array([0.5, 0.25, 0.25])
    ce = -float(np.sum(pi1 * np.log(p_hat)))
    ent = -float(np.sum(pi1 * np.log(pi1)))
    kl = ce - ent
    v_loss = (1.0 - 0.2) ** 2
    print(f"  1. 访问次数 [60, 30, 10] → T=1: {[round(x, 3) for x in pi1]} ｜ T=0.5: {[round(x, 3) for x in pi05]}")
    print(f"  2. 策略 CE = {ce:.3f}（手算 0.970）= 熵 {ent:.3f}（手算 0.898）+ KL {kl:.3f}（手算 0.072）")
    print(f"  3. 价值 MSE = {v_loss:.2f}（手算 0.64）✓")
    assert np.allclose(pi05, [0.783, 0.196, 0.022], atol=2e-3)
    assert abs(ce - 0.970) < 2e-3 and abs(ent - 0.898) < 2e-3 and abs(kl - 0.072) < 2e-3


def train_alphazero(use_dirichlet=True, n_iters=16, games_per_iter=15, n_sims=30, seed=0):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    net = PolicyValueNet(hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    replay = deque(maxlen=4000)

    pi_losses, v_losses = [], []
    rand_win_hist, mm_draw_hist = [], []
    iters_x = []
    alpha = 0.8 if use_dirichlet else None

    for it in range(1, n_iters + 1):
        for _ in range(games_per_iter):
            replay.extend(self_play_game(net, n_sims=n_sims, dirichlet_alpha=alpha, dirichlet_eps=0.25, rng=rng))

        net.train()
        pl_acc, vl_acc = [], []
        n_upd = min(40, max(10, len(replay) // 16))
        for _ in range(n_upd):
            idx = rng.integers(0, len(replay), size=min(64, len(replay)))
            obs_b = torch.from_numpy(np.stack([replay[i][0] for i in idx]))
            pi_b = torch.from_numpy(np.stack([replay[i][1] for i in idx]))
            v_b = torch.tensor([replay[i][2] for i in idx], dtype=torch.float32)
            m_b = torch.from_numpy(np.stack([replay[i][3] for i in idx]))

            logits, vals = net(obs_b)
            loss, pl, vl = alphazero_loss(logits, vals, pi_b, v_b, legal_masks=m_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pl_acc.append(pl)
            vl_acc.append(vl)

        pi_losses.append(float(np.mean(pl_acc)))
        v_losses.append(float(np.mean(vl_acc)))

        if it == 1 or it % max(1, n_iters // 4) == 0 or it == n_iters:
            rw, rd, _ = evaluate_agent(net, "random", n_games=12 if is_quick() else 40, n_sims=n_sims)
            mw, md, ml = evaluate_agent(net, "minimax", n_games=10 if is_quick() else 20, n_sims=n_sims)
            iters_x.append(it * games_per_iter)
            rand_win_hist.append((rw, rd))
            mm_draw_hist.append((mw, md, ml))

    return net, iters_x, rand_win_hist, mm_draw_hist, pi_losses, v_losses


def main():
    seed_everything(SEED)
    verify_hand_calculations()

    quick = is_quick()
    n_iters = 3 if quick else 14
    games_per_iter = 6 if quick else 15
    n_sims = 15 if quick else 30

    print("\n" + "=" * 60)
    print(f"第二部分：井字棋 AlphaZero 自我对弈（每步 {n_sims} 次模拟，α=0.8，共 {n_iters * games_per_iter} 盘）")
    print("=" * 60)
    t0 = time.time()
    net_d, xs_d, rw_d, mm_d, pl_d, vl_d = train_alphazero(True, n_iters, games_per_iter, n_sims, seed=SEED)
    net_nd, xs_nd, rw_nd, mm_nd, pl_nd, vl_nd = train_alphazero(False, n_iters, games_per_iter, n_sims, seed=SEED)
    print(f"  训练总用时：{time.time() - t0:.1f}s")
    print(f"  有 Dirichlet 噪声（α=0.8, ε=0.25）："
          f"对随机 胜 {rw_d[-1][0]*100:4.1f}% / 和 {rw_d[-1][1]*100:4.1f}% ｜ "
          f"对 Minimax 胜/和/负 = {mm_d[-1][0]*100:3.0f}% / {mm_d[-1][1]*100:4.1f}% / {mm_d[-1][2]*100:4.1f}%")
    print(f"  无 Dirichlet 噪声               ："
          f"对随机 胜 {rw_nd[-1][0]*100:4.1f}% / 和 {rw_nd[-1][1]*100:4.1f}% ｜ "
          f"对 Minimax 胜/和/负 = {mm_nd[-1][0]*100:3.0f}% / {mm_nd[-1][1]*100:4.1f}% / {mm_nd[-1][2]*100:4.1f}%")
    print(f"  末轮损失（有噪声）：策略 CE = {pl_d[-1]:.3f}，价值 MSE = {vl_d[-1]:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    axes[0].plot(xs_d, [r[0] * 100 for r in rw_d], "o-", linewidth=2.2, label="有 Dirichlet (α=0.8)")
    axes[0].plot(xs_nd, [r[0] * 100 for r in rw_nd], "s--", linewidth=2.2, label="无 Dirichlet")
    axes[0].set_xlabel("累计自我对弈盘数")
    axes[0].set_ylabel("对随机对手胜率 (%)")
    axes[0].set_title("对随机对手胜率曲线")
    axes[0].legend(fontsize=8.5)

    axes[1].plot(xs_d, [(r[0] + r[1]) * 100 for r in mm_d], "o-", linewidth=2.2, label="有 Dirichlet 不败率")
    axes[1].plot(xs_nd, [(r[0] + r[1]) * 100 for r in mm_nd], "s--", linewidth=2.2, label="无 Dirichlet 不败率")
    axes[1].set_xlabel("累计自我对弈盘数")
    axes[1].set_ylabel("对完美 Minimax 不败率 (%)")
    axes[1].set_title("对完美极小极大对手的不败率")
    axes[1].legend(fontsize=8.5)

    axes[2].plot(pl_d, linewidth=2.0, label="策略交叉熵（有噪声）")
    axes[2].plot(vl_d, linewidth=2.0, label="价值 MSE（有噪声）")
    axes[2].set_xlabel("训练轮次")
    axes[2].set_ylabel("损失")
    axes[2].set_title("策略损失与价值损失曲线")
    axes[2].legend(fontsize=8.5)
    save_figure(fig, OUT / "03-alphazero-tictactoe.png")

    print("\n" + "=" * 60)
    print("本节完成！下一节：04-muzero-model.py（MuZero 的三个网络与推理接口）")
    print("=" * 60)


if __name__ == "__main__":
    main()
