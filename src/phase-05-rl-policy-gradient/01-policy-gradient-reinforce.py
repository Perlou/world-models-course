"""
策略梯度定理与 REINFORCE
========================

学习目标：
    1. 用两臂老虎机数值验证策略梯度定理：采样估计的梯度收敛到直接求导的解析值 (-0.5, +0.5)
    2. 亲眼看到策略梯度的方差来源一：把两个动作奖励同时加 100，真实梯度丝毫不变，单样本方差却暴涨 2601 倍
    3. 实现 compute_returns_to_go() 与 reinforce_update()，注意代理损失数值本身没有单调下降意义
    4. 在 CartPole-v1 上运行多随机种子 REINFORCE，观察它的无偏学习能力与剧烈的种子间波动

核心概念：
    - 策略梯度定理（Policy Gradient Theorem）：∇_φ J(φ) = E[ Σ_t ∇_φ log π_φ(a_t | s_t) · G_t ]
    - 对数导数技巧（log-derivative trick）：∇ p = p · ∇ log p，让未知的环境转移概率从梯度里消失
    - reward-to-go（G_t）：动作不能改变过去，删掉 t 之前的奖励不改期望、只降方差
    - 代理损失（surrogate loss）：L_PG = −mean(log π · sg(G_t))，只借用自动求导算梯度，其数值不代表好坏

前置知识：
    - Phase 4 的 01-returns-and-value-functions.py（折扣回报 G_t）
    - deep-learning-course（softmax、自动求导）

对应概念文档：
    - CONCEPT.md 第 1 节「直接优化策略：策略梯度定理」、第 2 节「REINFORCE 与方差问题」

预计运行时间：约 25 秒（CPU）
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-05")
SEED = 0
GAMMA = 0.99


# ==================== 第一部分：导读 ====================


def introduction():
    print("=" * 60)
    print("第一部分：直接优化策略——用事后回报给动作概率加权")
    print("=" * 60)
    print("""
Phase 1–3 用规划选动作（慢，每步重搜），Phase 4 用 argmax_a Q(s, a) 选动作（连续动作没法直接取 max）。
从本章开始直接学一个策略网络 π_φ(a | s)：决策时只需一次前向传播。

难点在于回报 G_0(τ) 对参数 φ 不可导（中间隔着动作采样和黑盒环境）。
对数导数技巧 ∇_φ p_φ(τ) = p_φ(τ) ∇_φ log p_φ(τ) 把"对分布求导"变成了"在分布下求期望"：

    轨迹采样 τ ~ p_φ  ──▶  算每步往后的折扣回报 G_t = r_t + γ r_{t+1} + ···
                              │
                              ▼
    代理损失 L(φ) = − (1/M) Σ_t log π_φ(a_t | s_t) · sg(G_t)
                              │
                              ▼  反向传播
    G_t 高的动作概率被推高，G_t 低（或负）的动作概率被压低；环境转移概率完全不需要知道！
    """)


# ==================== 第二部分：两臂老虎机手算与采样核对 ====================


def check_bandit_gradient(n_samples=10_000):
    """CONCEPT.md 第 1.5、2.3 节：两臂老虎机解析梯度 vs 蒙特卡洛估计，以及常数偏移对方差的放大。"""
    print("\n" + "=" * 60)
    print("第二部分：两臂老虎机——核对策略梯度与方差暴涨现象")
    print("=" * 60)

    gen = torch.Generator().manual_seed(SEED)
    logits = torch.zeros(2)  # π = (0.5, 0.5)
    probs = torch.softmax(logits, dim=0)

    print(f"  设定：两个动作，logits = (0, 0) → π = (0.5, 0.5)，解析真实梯度 ∇_z J = (-0.500, +0.500)，采样数 = {n_samples}\n")
    print(f"  {'奖励设定':<16}{'估计梯度 (∂J/∂z_1, ∂J/∂z_2)':>30}{'z_1 分量单样本方差':>22}{'理论方差':>12}")

    stats = {}
    for label, rewards_vec, theory_var in [
        ("r = (1, 3)", torch.tensor([1.0, 3.0]), 1.0),
        ("r = (101, 103)", torch.tensor([101.0, 103.0]), 2601.0),
    ]:
        actions = torch.multinomial(probs, n_samples, replacement=True, generator=gen)  # [N]
        r = rewards_vec[actions]  # [N]
        # softmax 得分函数（第 1.4 节）：∂ log π(a) / ∂ z_b = 1[a=b] − π(b)
        one_hot = torch.nn.functional.one_hot(actions, num_classes=2).float()
        score_fn = one_hot - probs.unsqueeze(0)  # [N, 2]
        grad_samples = score_fn * r.unsqueeze(1)  # [N, 2]

        est_mean = grad_samples.mean(dim=0)
        est_var_z1 = grad_samples[:, 0].var(correction=0).item()
        stats[label] = (est_mean.tolist(), est_var_z1)
        print(f"  {label:<16}({est_mean[0]:+7.3f}, {est_mean[1]:+7.3f}){'':>12}{est_var_z1:>22.2f}{theory_var:>12.1f}")

    # 顺手核对自动求导代理损失给出的梯度完全一致
    z_param = torch.zeros(2, requires_grad=True)
    dist = Categorical(logits=z_param)
    acts_small = torch.tensor([0, 1])  # 两个动作各出现一次，等权平均即期望
    rews_small = torch.tensor([1.0, 3.0])
    surrogate = -(dist.log_prob(acts_small) * rews_small).mean()
    surrogate.backward()
    # 注意最小化代理损失 L = -J_surr，所以 -z_param.grad 才是 ∇J
    exact_via_autograd = (-z_param.grad).tolist()
    assert abs(exact_via_autograd[0] - (-0.5)) < 1e-6 and abs(exact_via_autograd[1] - 0.5) < 1e-6
    print(f"\n  ✓ 自动求导核对 -∇L_surrogate = ({exact_via_autograd[0]:+.3f}, {exact_via_autograd[1]:+.3f})，与手算完全一致。")
    print("  ✓ 奖励加 100 不改变哪个动作更好，真实梯度不变，但单样本方差从 ~1 暴涨到 ~2601（下一节用基线解决它）。")
    return stats


# ==================== 第三部分：REINFORCE 实现 ====================


def compute_returns_to_go(rewards, gamma=GAMMA):
    """从后往前递推计算单个回合的折扣回报 G_t（CONCEPT.md 第 2.4 节）。"""
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = float(rewards[t]) + gamma * running
        out[t] = running
    return out


class PolicyNet(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


def reinforce_update(policy, optimizer, obs_batch, act_batch, ret_batch):
    """执行一次 REINFORCE 参数更新（CONCEPT.md 第 1.7、2.5 节）。"""
    dist = Categorical(logits=policy(obs_batch))
    log_probs = dist.log_prob(act_batch)
    loss = -(log_probs * ret_batch.detach()).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_reinforce_seed(seed, n_updates, episodes_per_update=4, lr=3e-3):
    """在 CartPole-v1 上用纯 REINFORCE（不减基线、不批内归一化，原汁原味展示高方差）训练一个种子。"""
    seed_everything(seed)
    env = gym.make("CartPole-v1")
    policy = PolicyNet()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    ep_returns = []
    batch_means = []
    first_500_ep = None
    ep_counter = 0

    for upd in range(n_updates):
        obs_list, act_list, ret_list = [], [], []
        upd_rets = []
        for _ in range(episodes_per_update):
            obs, _ = env.reset(seed=seed * 10_000 + ep_counter)
            ep_counter += 1
            ep_obs, ep_act, ep_rew = [], [], []
            while True:
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    a = int(Categorical(logits=policy(obs_t)).sample().item())
                next_obs, r, term, trunc, _ = env.step(a)
                ep_obs.append(obs)
                ep_act.append(a)
                ep_rew.append(r)
                obs = next_obs
                if term or trunc:
                    break
            ep_total = float(sum(ep_rew))
            ep_returns.append(ep_total)
            upd_rets.append(ep_total)
            if first_500_ep is None and ep_total >= 500.0:
                first_500_ep = ep_counter
            # 注意：逐回合单独计算 G_t 后再拼接，绝不能跨回合递推！
            obs_list.append(np.asarray(ep_obs, dtype=np.float32))
            act_list.append(np.asarray(ep_act, dtype=np.int64))
            ret_list.append(compute_returns_to_go(ep_rew, GAMMA))

        obs_b = torch.from_numpy(np.concatenate(obs_list, axis=0))
        act_b = torch.from_numpy(np.concatenate(act_list, axis=0))
        ret_b = torch.from_numpy(np.concatenate(ret_list, axis=0))
        reinforce_update(policy, optimizer, obs_b, act_b, ret_b)
        batch_means.append(float(np.mean(upd_rets)))

    env.close()
    return np.array(batch_means), first_500_ep


def run_cartpole_experiment(n_seeds, n_updates, episodes_per_update):
    print("\n" + "=" * 60)
    print("第三部分：CartPole-v1 上的多随机种子 REINFORCE")
    print("=" * 60)

    # 先核对第 2.4 节三步手算例子
    hand_g = compute_returns_to_go([1.0, 0.0, 2.0], gamma=0.9)
    assert np.allclose(hand_g, [2.62, 1.8, 2.0], atol=1e-5)
    print(f"  ✓ 核对第 2.4 节手算例子：奖励 (1, 0, 2), γ=0.9 → G = {np.round(hand_g, 2).tolist()}")

    t0 = time.time()
    curves = []
    first_hits = []
    for s in range(n_seeds):
        curve, hit = train_reinforce_seed(s, n_updates=n_updates, episodes_per_update=episodes_per_update)
        curves.append(curve)
        first_hits.append(hit)
        hit_str = f"第 {hit} 回合" if hit is not None else "未达到"
        print(f"    种子 {s}：首次触及满分 500 = {hit_str:<10} ｜ 最后 10 批平均回报 = {curve[-10:].mean():6.1f}")

    arr = np.stack(curves, axis=0)  # [S, U]
    ep_axis = np.arange(1, n_updates + 1) * episodes_per_update
    elapsed = time.time() - t0
    print(f"  共运行 {n_seeds} 个种子 × {n_updates * episodes_per_update} 回合，总耗时 {elapsed:.1f}s。")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for s in range(n_seeds):
        axes[0].plot(ep_axis, arr[s], alpha=0.55, linewidth=1.3, label=f"种子 {s}")
    axes[0].axhline(500.0, color="gray", linestyle=":", label="满分 500")
    axes[0].set_xlabel("训练回合数")
    axes[0].set_ylabel("每批平均回合回报")
    axes[0].set_title("各随机种子独立曲线：抖动与分化明显")
    axes[0].legend(fontsize=8)

    mean_c = arr.mean(axis=0)
    axes[1].plot(ep_axis, mean_c, color="C0", linewidth=2.2, label="均值")
    axes[1].fill_between(ep_axis, arr.min(axis=0), arr.max(axis=0), color="C0", alpha=0.2, label="最小 ~ 最大范围")
    axes[1].axhline(500.0, color="gray", linestyle=":")
    axes[1].set_xlabel("训练回合数")
    axes[1].set_ylabel("每批平均回合回报")
    axes[1].set_title(f"纯 REINFORCE 聚合学习曲线（{n_seeds} 种子）")
    axes[1].legend(fontsize=8.5)
    save_figure(fig, OUT / "01-reinforce-cartpole.png")
    return arr, first_hits


# ==================== 第四部分：练习与思考 ====================


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：如果把 reinforce_update 里的 G_t 换成整条轨迹的总回报 G_0（即动作也为它发生之前的奖励背锅），曲线会怎样？

练习 1 参考答案：
    梯度的数学期望不变（第 1.3 节第六步证明过 t' < t 项期望为 0），但方差大约翻倍，
    同样学习率下曲线抖动更剧烈、收敛更慢。下一节我们会把这两者的梯度方差精确测出来。

练习 2：为什么 CartPole-v1 上纯 REINFORCE 经常走到三四百分又突然掉回几十？

练习 2 参考答案：
    CartPole 每步奖励都是 +1，G_t 全是正数（高达几百），没有减基线时所有采到的动作都在被"鼓励"，
    只是好动作被推得多一点、坏动作推得少一点。一旦某批数据方差大、方向偏了，一次过猛的步子就会把策略推坏，
    而策略变坏后采到的新轨迹又全是短命轨迹，很难立刻恢复。

思考题 1：为什么不能把上一轮更新用过的轨迹存进回放池，像 DQN 那样反复抽出来跑 reinforce_update？

思考题 1 参考答案：
    策略梯度公式里的期望是在当前策略分布 p_φ 下取的（在策略 on-policy）。
    φ 更新之后，旧轨迹里选出的动作不再代表新策略的采样概率；直接复用旧轨迹会让梯度严重偏离，
    要复用几轮就必须引入重要性采样比率与裁剪——那就是第 04 节的 PPO。
    """)


def main():
    seed_everything(SEED)
    quick = is_quick()
    if quick:
        print("（冒烟模式：样本数与回合数缩小，仅验证流程跑通）\n")
    n_samples = 1000 if quick else 10_000
    n_seeds, n_updates, ep_per_upd = (2, 20, 2) if quick else (5, 125, 4)  # 完整模式每种子 500 回合

    introduction()
    check_bandit_gradient(n_samples)
    run_cartpole_experiment(n_seeds, n_updates, ep_per_upd)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：02-baselines-and-advantage.py（用基线砍掉梯度方差）")
    print("=" * 60)


if __name__ == "__main__":
    main()
