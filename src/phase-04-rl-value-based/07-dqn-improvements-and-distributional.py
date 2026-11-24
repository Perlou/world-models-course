"""
DQN 的改进与分布式价值
======================

学习目标：
    1. 用随机数复现"对带噪声的估计取 max 会高估"，理解 Q-learning 的过估计从哪来
    2. 实现 Double DQN 的目标（代码只改一行），在 CartPole-v1 上测量 Q 值和实际回报的差距
    3. 实现 C51 的投影 project_distribution()：把挪动后的回报分布分回固定的格点，并验证均值不变
    4. 实现 two_hot()：投影的单点版本，把回归改写成分类。这是 Dreamer V3 训练 critic 的做法
    5. 训练一个 C51 智能体，看它学到的回报分布

核心概念：
    - 过估计（overestimation）/ 赢家的诅咒：E[max_a X_a] ≥ max_a E[X_a]
    - Double DQN：在线网络选动作，目标网络评估这个动作
    - 分布式价值（distributional RL）：预测回报的整个分布，而不只是期望
    - 支撑点（atom）、投影（projection）、two-hot 编码

前置知识：
    - 06-dqn-from-scratch.py：DQN 的损失、回放缓冲区、目标网络
    - deep-learning-course：softmax、交叉熵

对应概念文档：
    - CONCEPT.md 第 9 节「DQN 的改进与分布式价值」

预计运行时间：尚未实测（本文件是还没有运行过的草稿，定稿前要完整跑一遍并改写这一行）
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

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-04")
SEED = 0
DEVICE = torch.device("cpu")  # 和上一节一样：小 MLP、逐步选动作，CPU 单线程最合适
torch.set_num_threads(1)

# 训练设置沿用上一节（06-dqn-from-scratch.py）
CONFIG = dict(gamma=0.99, hidden=128, lr=5e-4, batch_size=64, buffer_size=20000, learning_starts=1000,
              target_sync=500, eps_start=1.0, eps_end=0.05, eps_decay_steps=8000, grad_clip=10.0,
              eval_interval=2000, eval_episodes=5)
# C51 的支撑点：CartPole 每步奖励 1、γ = 0.99，回报落在 [0, 100] 之内
V_MIN, V_MAX, N_ATOMS = 0.0, 100.0, 51


# ==================== 第一部分：赢家的诅咒 ====================


def introduction():
    """过估计的直觉，以及这一节的两条线。"""
    print("=" * 60)
    print("第一部分：Q-learning 的一个系统性毛病")
    print("=" * 60)

    print("""
同时试了 20 个方案，挑出指标最好的那个上线，上线后的实际效果却比实验里测到的差一截。
原因：每个方案的测量值 = 真实效果 + 噪声。"挑测量值最高的"既偏爱真实效果好的，
也偏爱噪声碰巧为正的。这叫赢家的诅咒（winner's curse）。

Q-learning 的目标里正好有这个操作：

    y = r + γ · max_a' Q(s', a')        Q(s', ·) 是带噪声的估计，对它们取 max，结果系统性地偏高

偏高的目标还会被自举回传，一层一层累积。
Phase 1 的"模型被钻空子"是同一件事：规划器对带误差的预测取 max，挑中的往往是模型高估的地方。

这一节有两条线：
    前半：过估计和它的修法（Double DQN）
    后半：把价值预测成一个分布（C51），以及它的单点版本 two-hot —— Dreamer V3 的 critic 用的就是它
    """)


def maximization_bias_demo(n_samples):
    """不训练网络，直接用随机数复现 CONCEPT.md 第 9.2 节的高估量。"""
    print("\n" + "=" * 60)
    print("第二部分：对带噪声的估计取 max 会高估")
    print("=" * 60)

    print("""
设定：m 个动作的真实 Q 值都是 0。每个估计独立地以各一半的概率取 +1 或 -1（期望是 0，无偏）。

    单估计（Q-learning 的做法）：用同一组数既挑出最大的，又读它的值         → max_a X_a
    双估计（Double 的做法）：    用 A 组选动作 â = argmax X^A，用独立的 B 组读值 → X^B_â
    """)
    rng = np.random.default_rng(SEED)
    print(f"  {'动作数 m':<10}{'单估计的平均值':>14}{'理论值 1 − 2^(1−m)':>20}{'双估计的平均值':>16}")
    for m in (1, 2, 4, 8):
        x_a = rng.choice([-1.0, 1.0], size=(n_samples, m))
        x_b = rng.choice([-1.0, 1.0], size=(n_samples, m))
        single = x_a.max(axis=1).mean()
        chosen = x_a.argmax(axis=1)  # A 组选动作
        double = x_b[np.arange(n_samples), chosen].mean()  # B 组评估
        theory = 1.0 - 2.0 ** (1 - m)
        tolerance = 4.0 / np.sqrt(n_samples)  # 平均值的标准误差约 1/√n，留 4 倍余量
        assert abs(single - theory) < tolerance and abs(double) < tolerance
        print(f"  {m:<12}{single:>14.3f}{theory:>20.3f}{double:>16.3f}")
    print("""
  每个估计单独看都是无偏的，取 max 之后就高估了，动作越多越严重。
  双估计的平均值回到了 0 附近：B 组和"选谁"这件事独立，不管选中的是谁，读出来的值的期望都是它的真实值。

  高估为什么有害？如果所有 (s, a) 被高估得一样多，argmax 的结果不变，策略不受影响。
  问题在于高估不均匀：数据少、估计噪声大的 (s, a) 被高估得多，智能体会偏爱它了解得最少的动作。
    """)


# ==================== 第二部分：Double DQN ====================


def dqn_target(rewards, terminated, next_q_target, gamma):
    """DQN 的目标：目标网络自己选、自己评。next_q_target 是 Q_ψ⁻(s', ·)，形状 [B, A]。返回 [B]。"""
    return rewards + gamma * (1.0 - terminated) * next_q_target.max(dim=1).values


def double_dqn_target(rewards, terminated, next_q_online, next_q_target, gamma):
    """Double DQN 的目标（CONCEPT.md 第 9.3、9.8 节）：在线网络选动作，目标网络评估。

    参数：
        next_q_online: [B, A]，在线网络的 Q_ψ(s', ·)
        next_q_target: [B, A]，目标网络的 Q_ψ⁻(s', ·)
    返回：
        [B]
    """
    a_star = next_q_online.argmax(dim=1)  # [B]，和 DQN 相比只改了这一行：argmax 用在线网络
    q_next = next_q_target.gather(1, a_star[:, None]).squeeze(1)  # [B]
    return rewards + gamma * (1.0 - terminated) * q_next


def double_dqn_by_hand():
    """复现 CONCEPT.md 第 9.3 节的手算例子。"""
    online = torch.tensor([[1.0, 2.0, 1.5]])
    target = torch.tensor([[1.2, 1.4, 2.5]])
    r, d = torch.zeros(1), torch.zeros(1)
    y_dqn = dqn_target(r, d, target, 0.9).item()
    y_double = double_dqn_target(r, d, online, target, 0.9).item()
    assert abs(y_dqn - 2.25) < 1e-5 and abs(y_double - 1.26) < 1e-5
    print(f"""  Double DQN 的观察：DQN 里现成就有两个网络。用在线网络选动作，用目标网络评估，代码只改一行。

  手算例子（三个动作，r = 0，γ = 0.9）：

                              动作 0    动作 1    动作 2
      在线网络 Q_ψ(s', ·)      1.0      [2.0]     1.5
      目标网络 Q_ψ⁻(s', ·)     1.2       1.4     [2.5]

      DQN：        目标网络自己选、自己评     y = 0.9 × 2.5 = {y_dqn:.2f}
      Double DQN： 在线网络选动作 1，目标网络评估它   y = 0.9 × 1.4 = {y_double:.2f}

  目标网络给动作 2 的 2.5 如果是噪声造成的虚高，DQN 会照单全收；
  Double DQN 要求在线网络也认为动作 2 最好才会用到它。两个网络同时在同一个动作上虚高的概率小得多。
    """)


# ==================== 第三部分：C51 的投影与 two-hot ====================


def project_distribution(next_probs, rewards, terminated, support, gamma):
    """C51 的投影（CONCEPT.md 第 9.6、9.8 节）。

    下一状态的回报分布（支撑点 b_j 上有概率 p'_j）经过 r + γ·(·) 之后，每个支撑点挪到 r + γ b_j。
    新位置一般不在格点上，把它带的概率按距离分给左右两个相邻格点：离哪边近，哪边分得多。

    参数：
        next_probs: [B, M]，目标网络在选定动作上给出的分布
        rewards, terminated: [B]
        support:    [M]，等间隔的支撑点
    返回：
        m: [B, M]，投影后的目标分布，每行和为 1
    """
    n_atoms = support.shape[0]
    v_min, v_max = support[0], support[-1]
    delta = (v_max - v_min) / (n_atoms - 1)
    # 挪动后的位置。终止时没有后续回报，所有支撑点都挪到 r。超出范围的截到边界上
    x = (rewards[:, None] + gamma * (1.0 - terminated[:, None]) * support[None, :]).clamp(v_min, v_max)  # [B, M]
    pos = ((x - v_min) / delta).clamp(0, n_atoms - 1)  # 落在第几个格点（带小数）
    lower = pos.floor().long()  # 左边的格点
    upper = (lower + 1).clamp(max=n_atoms - 1)  # 右边的格点
    w_upper = pos - lower.float()  # 离左边越远，右边分得越多
    w_lower = 1.0 - w_upper  # 正好落在格点上时 w_upper = 0，概率全给 lower，不会凭空消失
    m = torch.zeros_like(next_probs)
    m.scatter_add_(1, lower, next_probs * w_lower)
    m.scatter_add_(1, upper, next_probs * w_upper)
    return m


def two_hot(y, support):
    """two-hot 编码（CONCEPT.md 第 9.7 节）：把标量 y 编码成相邻两个格点上的概率，解码（求均值）得回 y。

    等价于对"全部概率集中在 y 上"的分布做投影。参数 y: [B]；返回 [B, M]。
    """
    n_atoms = support.shape[0]
    v_min, v_max = support[0], support[-1]
    delta = (v_max - v_min) / (n_atoms - 1)
    pos = ((y.clamp(v_min, v_max) - v_min) / delta).clamp(0, n_atoms - 1)  # [B]
    lower = pos.floor().long()
    upper = (lower + 1).clamp(max=n_atoms - 1)
    w_upper = pos - lower.float()
    out = torch.zeros(y.shape[0], n_atoms)
    out.scatter_add_(1, lower[:, None], (1.0 - w_upper)[:, None])
    out.scatter_add_(1, upper[:, None], w_upper[:, None])
    return out


def projection_tests():
    """用 CONCEPT.md 第 9.6、9.7 节的数字做单元测试，再检查两条性质。"""
    print("\n" + "=" * 60)
    print("第四部分：把价值预测成分布——投影与 two-hot")
    print("=" * 60)

    support = torch.arange(5.0)  # 支撑点 (0, 1, 2, 3, 4)，间隔 1
    next_probs = torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0]])
    r, d, gamma = torch.tensor([0.3]), torch.zeros(1), 0.9
    m = project_distribution(next_probs, r, d, support, gamma)
    pred = torch.tensor([[0.1, 0.1, 0.4, 0.3, 0.1]])
    cross_entropy = -(m * pred.log()).sum().item()
    assert torch.allclose(m, torch.tensor([[0.0, 0.0, 0.45, 0.55, 0.0]]), atol=1e-5)
    assert abs((m * support).sum().item() - 2.55) < 1e-5 and abs(cross_entropy - 1.0745) < 1e-3
    m_done = project_distribution(next_probs, r, torch.ones(1), support, gamma)
    code = two_hot(torch.tensor([2.3]), support)
    assert torch.allclose(code, torch.tensor([[0.0, 0.0, 0.7, 0.3, 0.0]]), atol=1e-5)

    print(f"""
Q(s, a) 是回报的期望，一个数。分布式价值预测的是回报的整个分布，用固定格点上的直方图表示：
网络对每个动作输出 M 个 logit，softmax 后是每个支撑点上的概率，Q 值是分布的均值 Σ_i p_i·b_i。

难点在构造目标。下一状态的分布经过 r + γ·(·) 之后，整个直方图被压缩了 γ 倍、平移了 r，
支撑点不在原来的格点上了，要投影（project）回去。

手算例子（CONCEPT.md 第 9.6 节）：支撑点 (0, 1, 2, 3, 4)，下一状态的分布 (0, 0, 0.5, 0.5, 0)，r = 0.3，γ = 0.9

    格点 2 挪到 0.3 + 0.9 × 2 = 2.1：落在格点 2 和 3 之间，0.5 的概率按 0.9 : 0.1 分给两边
    格点 3 挪到 0.3 + 0.9 × 3 = 3.0：正好在格点 3 上，0.5 的概率全给它

    project_distribution() 的输出：{[round(v, 3) for v in m[0].tolist()]}
    均值：{(m * support).sum().item():.2f}，等于 r + γ × 2.5 = 2.55 —— 投影不改变目标的期望
    和预测 (0.1, 0.1, 0.4, 0.3, 0.1) 的交叉熵：{cross_entropy:.3f}（手算 1.074）
    如果这一步终止了：所有概率都挪到 r = 0.3 上，投影成 {[round(v, 3) for v in m_done[0].tolist()]}

two-hot 是投影的单点版本：全部概率集中在一个数 y 上，投影后只有相邻两个格点非零。

    two_hot(2.3) = {[round(v, 3) for v in code[0].tolist()]}        解码：0.7 × 2 + 0.3 × 3 = {(code * support).sum().item():.1f}
    """)

    # 性质一：随机数据上，投影后每行和为 1，均值等于 r + γ·(1−d)·E[下一状态的分布]
    torch.manual_seed(SEED)
    support = torch.linspace(-10.0, 10.0, 51)
    probs = torch.rand(256, 51).softmax(dim=-1)
    rewards = torch.rand(256) * 2 - 1  # |r| ≤ 1，挪动后不会超出 [-10, 10]，不触发截断
    terminated = (torch.rand(256) < 0.2).float()
    m = project_distribution(probs, rewards, terminated, support, 0.9)
    expected_mean = rewards + 0.9 * (1 - terminated) * (probs * support).sum(dim=1)
    row_gap = (m.sum(dim=1) - 1).abs().max().item()
    mean_gap = ((m * support).sum(dim=1) - expected_mean).abs().max().item()
    assert row_gap < 1e-4 and mean_gap < 1e-3
    # 性质二：two-hot 编码再解码得回原数
    y = torch.rand(1000) * 20 - 10
    decode_gap = ((two_hot(y, support) * support).sum(dim=1) - y).abs().max().item()
    assert decode_gap < 1e-3
    print(f"  随机数据检查（256 个样本，51 个支撑点）：每行之和与 1 最大相差 {row_gap:.1e}；"
          f"均值与 r + γ(1−d)·Q(s′, a′) 最大相差 {mean_gap:.1e}。")
    print(f"  two-hot 编码再解码（1000 个随机标量）：最大误差 {decode_gap:.1e}。")

    # two-hot 回归的梯度有界，平方误差的梯度随目标的尺度变化
    print("\n  用 two-hot 把回归改写成分类，好处在梯度上。同一个初始预测，目标的尺度从 1 变到 10000：\n")
    print(f"  {'目标 y':<12}{'平方误差对输出的梯度':>20}{'two-hot 交叉熵对 logit 的最大梯度':>32}")
    wide_support = torch.linspace(-20000.0, 20000.0, 255)
    for scale in (1.0, 100.0, 10000.0):
        target = torch.tensor([scale])
        out = torch.zeros(1, requires_grad=True)  # 普通回归：网络输出一个实数，初始为 0
        F.mse_loss(out, target).backward()
        logits = torch.zeros(1, 255, requires_grad=True)  # two-hot 回归：255 个 logit，初始为均匀分布
        (-(two_hot(target, wide_support) * F.log_softmax(logits, dim=-1)).sum()).backward()
        max_grad = logits.grad.abs().max().item()
        assert max_grad <= 1.0
        print(f"  {scale:<12.0f}{out.grad.item():>20.1f}{max_grad:>32.3f}")
    print("""
  交叉熵对 logit 的梯度是"预测概率 − 目标概率"，每个分量都在 [-1, 1] 之内，和目标的数值大小无关。
  平方误差的梯度随目标的尺度成比例变化，奖励尺度差几个数量级的任务很难共用一套学习率。
  Dreamer V3 用 two-hot 回归训练 critic 和奖励预测头（再配合 symlog 压缩数值范围），原因就在这里。
  注意区别：C51 的目标是一个分布（用了分布版的 Bellman 方程）；Dreamer V3 的目标是标量 λ-return 的 two-hot 编码，
  它从 C51 这条线继承的是表示和损失，不是分布式的 Bellman 更新。
    """)


# ==================== 第四部分：三种智能体 ====================


class ReplayBuffer:
    """和上一节相同的环形回放缓冲区（写得紧凑一些）。"""

    def __init__(self, capacity, obs_dim):
        self.data = {"obs": np.zeros((capacity, obs_dim), np.float32), "actions": np.zeros(capacity, np.int64),
                     "rewards": np.zeros(capacity, np.float32), "next_obs": np.zeros((capacity, obs_dim), np.float32),
                     "terminated": np.zeros(capacity, np.float32)}
        self.capacity, self.pos, self.size = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, terminated):
        for key, value in zip(self.data, (obs, action, reward, next_obs, float(terminated))):
            self.data[key][self.pos] = value
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(self.size, size=batch_size)
        return {key: torch.from_numpy(value[idx]) for key, value in self.data.items()}


def mlp(in_dim, out_dim, hidden):
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, out_dim))


class QNetwork(nn.Module):
    """普通的 Q 网络：输出 [B, A]。"""

    def __init__(self, obs_dim, n_actions, hidden):
        super().__init__()
        self.net = mlp(obs_dim, n_actions, hidden)

    def q_values(self, obs):
        return self.net(obs)


class C51Network(nn.Module):
    """分布式 Q 网络：对每个动作输出 M 个 logit，形状 [B, A, M]。"""

    def __init__(self, obs_dim, n_actions, hidden, support):
        super().__init__()
        self.n_actions, self.n_atoms = n_actions, support.shape[0]
        self.register_buffer("support", support)  # [M]
        self.net = mlp(obs_dim, n_actions * self.n_atoms, hidden)

    def forward(self, obs):
        return self.net(obs).view(-1, self.n_actions, self.n_atoms)  # logits [B, A, M]

    def q_values(self, obs):
        return (self(obs).softmax(dim=-1) * self.support).sum(dim=-1)  # 分布的均值，[B, A]


def q_loss(net, target_net, batch, gamma, double):
    """DQN / Double DQN 的损失，两者只差算目标的那一个函数。"""
    q = net.q_values(batch["obs"]).gather(1, batch["actions"][:, None]).squeeze(1)  # [B]
    with torch.no_grad():
        next_q_target = target_net.q_values(batch["next_obs"])  # [B, A]
        if double:
            target = double_dqn_target(batch["rewards"], batch["terminated"], net.q_values(batch["next_obs"]),
                                       next_q_target, gamma)
        else:
            target = dqn_target(batch["rewards"], batch["terminated"], next_q_target, gamma)
    return F.huber_loss(q, target)


def c51_loss(net, target_net, batch, gamma):
    """C51 的损失：投影后的目标分布与预测分布的交叉熵。"""
    rows = torch.arange(batch["obs"].shape[0])
    log_probs = F.log_softmax(net(batch["obs"]), dim=-1)[rows, batch["actions"]]  # [B, A, M] → [B, M]
    with torch.no_grad():
        next_probs = target_net(batch["next_obs"]).softmax(dim=-1)  # [B, A, M]
        a_star = (next_probs * target_net.support).sum(dim=-1).argmax(dim=1)  # [B]，按分布的均值选动作
        m = project_distribution(next_probs[rows, a_star], batch["rewards"], batch["terminated"],
                                 target_net.support, gamma)  # [B, M]
    return -(m * log_probs).sum(dim=1).mean()


def evaluate(net, env, n_episodes, seed, gamma, n_first=100):
    """按贪心策略跑几个回合。除了平均回报，还比较"网络预测的 Q"和"实际拿到的折扣回报"。

    只统计每个回合前 n_first 步的状态：撑满 500 步被掐断的回合，后面的状态实际还能继续拿奖励，
    但数据里没有；对前 100 步来说，漏掉的部分不超过 γ^400 / (1 − γ) ≈ 1.8，可以忽略。
    返回：(平均回报, 预测 Q 的平均, 实际折扣回报的平均)
    """
    returns, q_preds, g_actuals = [], [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        rewards, qs, done = [], [], False
        while not done:
            with torch.no_grad():
                q = net.q_values(torch.as_tensor(obs)[None])[0]  # [A]
            action = int(q.argmax().item())
            qs.append(q[action].item())
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
        g, discounted = 0.0, np.zeros(len(rewards))
        for t in reversed(range(len(rewards))):  # 第 01 课的从后往前递推
            g = rewards[t] + gamma * g
            discounted[t] = g
        returns.append(sum(rewards))
        q_preds.extend(qs[:n_first])
        g_actuals.extend(discounted[:n_first].tolist())
    return float(np.mean(returns)), float(np.mean(q_preds)), float(np.mean(g_actuals))


def train_agent(kind, seed, total_steps, cfg=CONFIG):
    """训练一个智能体。kind 取 "dqn"、"double" 或 "c51"，训练循环和上一节相同。

    返回 (日志字典, 训练好的网络)。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env, eval_env = gym.make("CartPole-v1"), gym.make("CartPole-v1")
    obs_dim, n_actions = env.observation_space.shape[0], int(env.action_space.n)
    if kind == "c51":
        support = torch.linspace(V_MIN, V_MAX, N_ATOMS)
        net, target_net = (C51Network(obs_dim, n_actions, cfg["hidden"], support) for _ in range(2))
    else:
        net, target_net = (QNetwork(obs_dim, n_actions, cfg["hidden"]) for _ in range(2))
    target_net.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    buffer = ReplayBuffer(cfg["buffer_size"], obs_dim)

    log = {"steps": [], "returns": [], "q_pred": [], "g_actual": []}
    start = time.time()
    obs, _ = env.reset(seed=seed)
    for step in range(1, total_steps + 1):
        epsilon = cfg["eps_start"] + min(step / cfg["eps_decay_steps"], 1.0) * (cfg["eps_end"] - cfg["eps_start"])
        if rng.random() < epsilon:
            action = int(rng.integers(n_actions))
        else:
            with torch.no_grad():
                action = int(net.q_values(torch.as_tensor(obs)[None]).argmax(dim=1).item())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, terminated)  # 只存 terminated
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        if step >= cfg["learning_starts"]:
            batch = buffer.sample(cfg["batch_size"], rng)
            if kind == "c51":
                loss = c51_loss(net, target_net, batch, cfg["gamma"])
            else:
                loss = q_loss(net, target_net, batch, cfg["gamma"], double=(kind == "double"))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            optimizer.step()
            if step % cfg["target_sync"] == 0:
                target_net.load_state_dict(net.state_dict())

        if step % cfg["eval_interval"] == 0:
            ret, q_pred, g_actual = evaluate(net, eval_env, cfg["eval_episodes"], 10000 + seed, cfg["gamma"])
            for key, value in zip(("steps", "returns", "q_pred", "g_actual"), (step, ret, q_pred, g_actual)):
                log[key].append(value)
    env.close()
    eval_env.close()
    log["seconds"] = time.time() - start
    return log, net


def cartpole_experiment(total_steps, seeds):
    """CartPole-v1 上对比 DQN、Double DQN、C51：Q 值准不准、学得怎么样。"""
    print("\n" + "=" * 60)
    print("第五部分：在 CartPole-v1 上测量过估计，并训练 C51")
    print("=" * 60)

    names = {"dqn": "DQN", "double": "Double DQN", "c51": "C51"}
    print(f"""
三种智能体用同一套超参数（沿用上一节），各训练 {total_steps} 步、{len(seeds)} 个种子。
C51 用 {N_ATOMS} 个支撑点，范围 [{V_MIN:.0f}, {V_MAX:.0f}]（每步奖励 1、γ = {CONFIG['gamma']} 时回报不超过 100）。

怎么测过估计：每次评估时按贪心策略跑几个回合，对经过的状态同时记下
    网络预测的 Q(s, 贪心动作)        和        从这个状态起实际拿到的折扣回报
前者比后者高，就是高估。判断 Q 准不准必须拿真实回报来对照，TD 误差小说明不了问题（第 03 课）。
    """)
    logs, nets = {kind: [] for kind in names}, {}
    for kind in names:
        for seed in seeds:
            log, net = train_agent(kind, seed, total_steps)
            logs[kind].append(log)
            nets.setdefault(kind, net)  # 留下第一个种子的网络，后面画分布用
            print(f"  [{names[kind]}，种子 {seed}] 用时 {log['seconds']:.0f} 秒，最后一次评估的平均回报 {log['returns'][-1]:.0f}")

    steps = np.array(logs["dqn"][0]["steps"])
    arr = {kind: {key: np.array([log[key] for log in runs]) for key in ("returns", "q_pred", "g_actual")}
           for kind, runs in logs.items()}  # 每项 [种子, 评估次数]
    gap = {kind: arr[kind]["q_pred"] - arr[kind]["g_actual"] for kind in names}  # 预测 − 实际，正数表示高估

    print(f"\n  预测的 Q 与实际折扣回报（{len(seeds)} 个种子的平均）：\n")
    print(f"  {'环境步数':<10}" + "".join(f"{names[k] + '：预测 / 实际 / 差':>30}" for k in names))
    for i in range(len(steps)):
        row = f"  {steps[i]:<12}"
        for k in names:
            cell = f"{arr[k]['q_pred'][:, i].mean():.1f} / {arr[k]['g_actual'][:, i].mean():.1f} / {gap[k][:, i].mean():+.1f}"
            row += f"{cell:>30}"
        print(row)

    tail = max(len(steps) // 3, 1)
    late_gap = {k: gap[k][:, -tail:].mean() for k in names}
    max_gap = {k: gap[k].mean(axis=0).max() for k in names}
    late_ret = {k: arr[k]["returns"][:, -tail:].mean() for k in names}
    print(f"\n  {'':<14}{f'最后 {tail} 次评估的平均差':>22}{'全程最大的平均差':>18}{f'最后 {tail} 次评估的平均回报':>24}")
    for k in names:
        print(f"  {names[k]:<14}{late_gap[k]:>+22.1f}{max_gap[k]:>+18.1f}{late_ret[k]:>24.0f}")

    print()
    if max_gap["dqn"] > 1.0:
        print(f"  DQN 的预测在训练过程中出现过明显高于实际回报的阶段（各种子平均最多高出 {max_gap['dqn']:.1f}）。")
    else:
        print(f"  这次实验里 DQN 的预测没有明显高于实际回报（各种子平均的差最大只有 {max_gap['dqn']:+.1f}）。")
    if max_gap["double"] < max_gap["dqn"] and abs(late_gap["double"]) <= abs(late_gap["dqn"]):
        print(f"  Double DQN 的差距更小：全程最大 {max_gap['double']:+.1f}（DQN {max_gap['dqn']:+.1f}），"
              f"后期平均 {late_gap['double']:+.1f}（DQN {late_gap['dqn']:+.1f}）。")
    elif max_gap["double"] < max_gap["dqn"]:
        print(f"  Double DQN 全程最大的差 {max_gap['double']:+.1f} 小于 DQN 的 {max_gap['dqn']:+.1f}，"
              f"但后期平均的差（{late_gap['double']:+.1f}）并不比 DQN（{late_gap['dqn']:+.1f}）更接近 0。")
    else:
        print(f"  这次实验里 Double DQN 没有表现出更小的差距：全程最大 {max_gap['double']:+.1f}（DQN {max_gap['dqn']:+.1f}），"
              f"后期平均 {late_gap['double']:+.1f}（DQN {late_gap['dqn']:+.1f}）。")
        print("  CartPole 只有 2 个动作，取 max 带来的高估本来就不大（第二部分的表：动作越多越严重）。")
    if any(g < -1.0 for g in late_gap.values()):
        print("  差为负数表示预测低于实际：评估时的贪心策略比训练数据里带探索的行为撑得更久，")
        print("  Q 网络在那些很少见到的好状态上还没学准时，会出现这种情况。")
    best = max(late_ret, key=late_ret.get)
    print(f"  后期平均回报最高的是 {names[best]}（{late_ret[best]:.0f}）；三者分别是 "
          + "、".join(f"{names[k]} {late_ret[k]:.0f}" for k in names) + "。只有几个种子，差别不大时不要过度解读。")

    # ---- C51 学到的分布 ----
    c51 = nets["c51"]
    states = {
        "杆竖直、静止": [0.0, 0.0, 0.0, 0.0],
        "杆向右倾斜 0.18 弧度并继续倒": [0.0, 0.0, 0.18, 1.5],
    }
    print("\n  C51（第一个种子）学到的回报分布，两个手工构造的状态：\n")
    print(f"  {'状态':<24}{'贪心动作':>8}{'分布的均值 Q':>14}{'回报 < 20 的概率':>18}{'回报 > 80 的概率':>18}")
    dists = {}
    for label, state in states.items():
        with torch.no_grad():
            probs = c51(torch.tensor([state], dtype=torch.float32)).softmax(dim=-1)[0]  # [A, M]
        q = (probs * c51.support).sum(dim=-1)
        a = int(q.argmax().item())
        dists[label] = (probs[a].numpy(), q[a].item())
        low = probs[a][c51.support < 20].sum().item()
        high = probs[a][c51.support > 80].sum().item()
        print(f"  {label:<24}{['向左推', '向右推'][a]:>8}{q[a].item():>14.1f}{low:>18.2f}{high:>18.2f}")
    q_good, q_bad = dists["杆竖直、静止"][1], dists["杆向右倾斜 0.18 弧度并继续倒"][1]
    if q_good > q_bad + 10:
        print(f"\n  两个状态的分布明显不同：稳定的状态均值 {q_good:.1f}，快倒的状态均值 {q_bad:.1f}。")
        print("  网络输出的不是一个数，而是整个直方图；选动作时仍然只用它的均值。")
    else:
        print(f"\n  这次训练出来的网络对两个状态给出的均值相差不大（{q_good:.1f} 和 {q_bad:.1f}），分布的形状见图。")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    colors = {"dqn": "tab:blue", "double": "tab:orange", "c51": "tab:green"}
    for k in names:
        axes[0].plot(steps, gap[k].mean(axis=0), color=colors[k], linewidth=2, label=names[k])
        axes[0].fill_between(steps, gap[k].min(axis=0), gap[k].max(axis=0), color=colors[k], alpha=0.15)
        axes[1].plot(steps, arr[k]["returns"].mean(axis=0), color=colors[k], linewidth=2, label=names[k])
    axes[0].axhline(0, color="gray", linestyle=":")
    axes[0].set_xlabel("环境步数")
    axes[0].set_ylabel("预测的 Q − 实际折扣回报")
    axes[0].set_title("Q 值准不准（正数是高估；阴影是各种子的范围）")
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel("环境步数")
    axes[1].set_ylabel("评估回报（不带探索）")
    axes[1].set_title(f"学习曲线（{len(seeds)} 个种子的平均）")
    axes[1].legend(fontsize=9)
    width = (V_MAX - V_MIN) / (N_ATOMS - 1) * 0.45
    for i, (label, (probs, q)) in enumerate(dists.items()):
        axes[2].bar(c51.support.numpy() + (i - 0.5) * width, probs, width=width, label=f"{label}（均值 {q:.1f}）")
    axes[2].set_xlabel("回报（支撑点）")
    axes[2].set_ylabel("概率")
    axes[2].set_title("C51 学到的回报分布（贪心动作）")
    axes[2].legend(fontsize=8)
    save_figure(fig, OUT / "07-double-dqn-and-c51.png")


# ==================== 第五部分：练习与思考 ====================


def exercises():
    """练习题与参考答案。答案里的数字都是下面的代码现算的。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    rng = np.random.default_rng(SEED + 1)
    ex1 = {m: rng.normal(size=(200000, m)).max(axis=1).mean() for m in (2, 4, 8, 18)}
    support = torch.arange(5.0)
    ex2 = two_hot(torch.tensor([7.0, -3.0]), support)
    ex2_decoded = (ex2 * support).sum(dim=1).tolist()
    ex3 = project_distribution(torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.5]]), torch.tensor([1.0]), torch.zeros(1), support, 0.9)
    ex3_mean, ex3_true = (ex3 * support).sum().item(), 1.0 + 0.9 * 3.5

    print(f"""
练习 1：换成高斯噪声
    m 个动作的真实 Q 值都是 0，估计的噪声是标准正态分布。E[max] 是多少？（m = 2、4、8、18，18 是 Atari 的动作数上限）

练习 1 参考答案：
    程序用 20 万次采样算出来的：{'，'.join(f'm = {m} 时 {v:.2f}' for m, v in ex1.items())}。
    噪声的标准差是 1 时，18 个动作的高估量已经超过了一个标准差。动作越多过估计越严重；
    CartPole 只有 2 个动作，单看"取 max"这一步，高估量是这几种情况里最小的。

练习 2：超出范围的标量
    支撑点是 (0, 1, 2, 3, 4)。two_hot(7.0) 和 two_hot(-3.0) 各是什么？解码得到多少？

练习 2 参考答案：
    程序算出来的：two_hot(7.0) = {[round(v, 2) for v in ex2[0].tolist()]}，解码 {ex2_decoded[0]:.1f}；
    two_hot(-3.0) = {[round(v, 2) for v in ex2[1].tolist()]}，解码 {ex2_decoded[1]:.1f}。
    超出范围的数被截到边界上，编码再解码得不回原数。所以支撑点的范围必须盖住目标的实际范围。
    范围设得太宽，格点又太稀。Dreamer V3 的办法是先对目标做 symlog 变换（压缩大数）再放格点（Phase 6）。

练习 3：投影时被截断会怎样
    支撑点 (0, 1, 2, 3, 4)，下一状态的分布 (0, 0, 0, 0.5, 0.5)，r = 1，γ = 0.9。投影后的均值还等于 r + γ·Q 吗？

练习 3 参考答案：
    r + γ·Q = 1 + 0.9 × 3.5 = {ex3_true:.2f}，但投影后的分布是 {[round(v, 3) for v in ex3[0].tolist()]}，均值只有 {ex3_mean:.2f}。
    格点 3、4 分别挪到 3.7 和 4.6，后者超出了上界 4，被截到 4 上，均值因此变小。
    回报超出 [V_min, V_max] 时价值会被系统性地压低，这类错误不会报错。

思考题 1：C51 学到的分布，表示的是"模型对价值有多不确定"吗？

思考题 1 参考答案：
    不是。它学的是回报本身的随机性（来自环境和策略），属于 Phase 1 说的偶然不确定性。
    "估计得准不准"是认知不确定性，要靠集成之类的办法（Phase 1 第 07 课）。

思考题 2：Double DQN 需要多训练一个网络吗？

思考题 2 参考答案：
    不需要。它复用 DQN 已有的在线网络和目标网络，只改算目标的一行。
    需要真正训练两个 Q 网络的是 Phase 5 的 TD3、SAC：目标里取两个 Q 中较小的那个，比 Double DQN 更保守。
    共同的原则是：对带噪声的估计做最大化之前，先想办法让估计偏保守。

思考题 3：C51 选动作时只用了分布的均值，那学整个分布的好处是什么？

思考题 3 参考答案：
    没有完全的定论。常见的解释是学一个分布提供了更丰富、更稳定的训练信号；
    另一个确定的好处是损失变成了交叉熵，梯度有界（第四部分的表）。
    Dreamer V3 和 TD-MPC2 借用的正是"把价值回归做成分类"这一点。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()
    maximization_bias_demo(n_samples=20000 if is_quick() else 200000)
    print("=" * 60)
    print("第三部分：Double DQN——选动作和评估动作分开")
    print("=" * 60 + "\n")
    double_dqn_by_hand()
    projection_tests()
    if is_quick():
        CONFIG.update(learning_starts=200, eps_decay_steps=1000, eval_interval=1000, eval_episodes=2)
        cartpole_experiment(total_steps=2000, seeds=[0])
    else:
        cartpole_experiment(total_steps=30000, seeds=[0, 1, 2])
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：08-dyna-q.py（Dyna-Q：用模型生成经验）")
    print("=" * 60)


if __name__ == "__main__":
    main()
