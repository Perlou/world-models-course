"""
基线与优势函数：不引入偏差地降低方差
====================================

学习目标：
    1. 用两个手算例子核对 CONCEPT.md 第 3.4 节：基线把 (101, 103) 老虎机的方差从 2601 直接降到 0；
       在不均匀策略下验证状态价值 V 与理论最优基线 b* 的差距
    2. 实现状态价值网络 ValueNet 与梯度方差测量器 measure_gradient_variance()
    3. 在同一个冻结策略上实测三种权重（整条轨迹 G_0、reward-to-go G_t、蒙特卡洛优势 G_t − V_ψ(s_t)）
       的梯度均值一致性与参数梯度总方差之比
    4. 在 CartPole-v1 上用同组随机种子对比三种权重的学习曲线

核心概念：
    - 基线（baseline）b(s_t)：只要不依赖当前动作 a_t，E[∇ log π(a_t|s_t) · b(s_t)] 恒等于 0（零偏差）
    - 优势函数（advantage function）：A^π(s, a) = Q^π(s, a) − V^π(s)，衡量动作比当前状态的平均水平好多少
    - 停止梯度：算策略损失时 (G_t − V_ψ(s_t)) 必须 detach，价值网络单独用回归损失拟合 G_t

前置知识：
    - 01-policy-gradient-reinforce.py（策略梯度与 REINFORCE）

对应概念文档：
    - CONCEPT.md 第 3 节「基线与优势函数」

预计运行时间：约 35 秒（CPU）
"""

import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec01 = importlib.util.spec_from_file_location("p5_01", os.path.join(_DIR, "01-policy-gradient-reinforce.py"))
_mod01 = importlib.util.module_from_spec(_spec01)
_spec01.loader.exec_module(_mod01)

PolicyNet = _mod01.PolicyNet
compute_returns_to_go = _mod01.compute_returns_to_go

plt = setup_matplotlib()
OUT = output_dir("phase-05")
SEED = 0
GAMMA = 0.99


# ==================== 第一部分：导读与手算核对 ====================


def introduction():
    print("=" * 60)
    print("第一部分：考试看排名不看绝对分——基线为什么无偏且降方差")
    print("=" * 60)
    print("""
上一节看到：CartPole 每步奖励 +1，所有动作乘的 G_t 都是正数，坏动作也在被推高概率。
解决办法是减去当前状态的平均水平 b(s_t) = V_ψ(s_t)，得到蒙特卡洛优势估计：

    Â_t = G_t − V_ψ(s_t)       > 0 表示比预期好（鼓励），< 0 表示不如预期（抑制）

为什么减掉任意 b(s_t) 都不会让梯度变偏？因为 b(s_t) 不依赖当前动作 a_t，可以从内层期望提出来：
    E_{a_t ~ π}[ ∇_φ log π_φ(a_t | s_t) · b(s_t) ] = b(s_t) · E_{a_t ~ π}[ ∇_φ log π_φ(a_t | s_t) ] = b(s_t) · 0 = 0
    """)


def verify_hand_calculations():
    """复现 CONCEPT.md 第 3.4 节的两个手算例子。"""
    print("\n" + "=" * 60)
    print("第二部分：核对 CONCEPT.md 第 3.4 节的手算例子")
    print("=" * 60)

    # 例一：π = (0.5, 0.5)，奖励 (101, 103)，基线 b = V = 102
    g1 = 0.5 * (101.0 - 102.0)
    g2 = -0.5 * (103.0 - 102.0)
    var_ex1 = 0.5 * (g1 - (-0.5)) ** 2 + 0.5 * (g2 - (-0.5)) ** 2
    print(f"  例一（r=(101,103), π=(0.5,0.5), b=102）：两种采样的单样本梯度分别为 {g1:.2f} 和 {g2:.2f}，方差 = {var_ex1:.4f}（从 2601 降到 0！）")
    assert abs(var_ex1) < 1e-8

    # 例二：π = (0.8, 0.2)，奖励 (1, 3)，对比 b = 0、b = V = 1.4、b = b* = 2.6
    p = np.array([0.8, 0.2])
    r = np.array([1.0, 3.0])
    scores = np.array([1.0 - 0.8, -0.8])  # 第一个 logit 的得分函数：(0.2, -0.8)
    v_val = float((p * r).sum())
    b_star = float((p * scores**2 * r).sum() / (p * scores**2).sum())

    print(f"\n  例二（r=(1,3), π=(0.8,0.2)，V={v_val:.1f}，最优基线 b*={b_star:.1f}）：")
    print(f"  {'基线 b':<16}{'动作1估计':>12}{'动作2估计':>12}{'期望梯度':>12}{'方差':>12}")
    for name, b in [("b = 0", 0.0), (f"b = V = {v_val:.1f}", v_val), (f"b* = {b_star:.1f}", b_star)]:
        est = scores * (r - b)
        mean = float((p * est).sum())
        var = float((p * (est - mean) ** 2).sum())
        print(f"  {name:<16}{est[0]:>12.2f}{est[1]:>12.2f}{mean:>12.2f}{var:>12.4f}")
        assert abs(mean - (-0.32)) < 1e-6


# ==================== 第二部分：价值网络与梯度方差实测 ====================


class ValueNet(nn.Module):
    """状态价值网络 V_ψ(s)：输入 [M, 4] → 输出 [M]。"""

    def __init__(self, obs_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def collect_batch(env, policy, n_episodes, rng):
    """用当前策略采集 n_episodes 个完整回合，返回 obs, actions, G_0（全轨迹总回报）, G_t（reward-to-go）。"""
    obs_all, act_all, g0_all, gt_all, ep_lens = [], [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 10_000_000)))
        ep_obs, ep_act, ep_rew = [], [], []
        while True:
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                a = int(Categorical(logits=policy(obs_t)).sample().item())
            obs, r, term, trunc, _ = env.step(a)
            ep_obs.append(obs_t.squeeze(0).numpy())
            ep_act.append(a)
            ep_rew.append(r)
            if term or trunc:
                break
        gt = compute_returns_to_go(ep_rew, GAMMA)
        g0 = np.full_like(gt, gt[0])  # 每一步都乘整条轨迹的总折扣回报 G_0
        obs_all.append(np.asarray(ep_obs, dtype=np.float32))
        act_all.append(np.asarray(ep_act, dtype=np.int64))
        gt_all.append(gt)
        g0_all.append(g0)
        ep_lens.append(float(sum(ep_rew)))

    return (
        torch.from_numpy(np.concatenate(obs_all, axis=0)),
        torch.from_numpy(np.concatenate(act_all, axis=0)),
        torch.from_numpy(np.concatenate(g0_all, axis=0)),
        torch.from_numpy(np.concatenate(gt_all, axis=0)),
        float(np.mean(ep_lens)),
    )


def flat_grad(policy, loss):
    policy.zero_grad()
    loss.backward(retain_graph=True)
    return torch.cat([p.grad.view(-1).clone() for p in policy.parameters()])


def measure_gradient_variance(n_batches=80, episodes_per_batch=4):
    """CONCEPT.md 第 3.6 节：固定同一个中等水平的策略与训练好的基线网络，重复采 K 批测三种权重的梯度方差。"""
    print("\n" + "=" * 60)
    print("第三部分：在 CartPole-v1 上实测三种权重的梯度方差")
    print("=" * 60)

    seed_everything(SEED)
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(SEED)
    policy = PolicyNet()
    value_net = ValueNet()
    p_opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    v_opt = torch.optim.Adam(value_net.parameters(), lr=5e-3)

    # 先预热 35 轮，让策略处于正在上升的中期阶段、V_ψ 对当前策略已有合理估计
    for _ in range(10 if is_quick() else 35):
        obs_b, act_b, _, gt_b, _ = collect_batch(env, policy, 4, rng)
        for _ in range(5):
            v_loss = F.mse_loss(value_net(obs_b), gt_b)
            v_opt.zero_grad()
            v_loss.backward()
            v_opt.step()
        with torch.no_grad():
            adv = gt_b - value_net(obs_b)
        dist = Categorical(logits=policy(obs_b))
        loss_p = -(dist.log_prob(act_b) * adv).mean()
        p_opt.zero_grad()
        loss_p.backward()
        p_opt.step()

    # 冻结策略与价值网络，独立采样 n_batches 批轨迹计算三种梯度向量 [K, P]
    grads = {"① 全轨迹回报 G_0": [], "② reward-to-go G_t": [], "③ 带基线 G_t − V_ψ(s_t)": []}
    sample_rets = []
    for _ in range(n_batches):
        obs_b, act_b, g0_b, gt_b, ep_ret = collect_batch(env, policy, episodes_per_batch, rng)
        sample_rets.append(ep_ret)
        with torch.no_grad():
            adv_b = gt_b - value_net(obs_b)
        dist = Categorical(logits=policy(obs_b))
        lp = dist.log_prob(act_b)
        # 注意：对回合批次求 sum / episodes_per_batch，使三种估计严格对应同一目标 J(φ) 的无偏梯度
        scale = 1.0 / episodes_per_batch
        grads["① 全轨迹回报 G_0"].append(-flat_grad(policy, -(lp * g0_b).sum() * scale))
        grads["② reward-to-go G_t"].append(-flat_grad(policy, -(lp * gt_b).sum() * scale))
        grads["③ 带基线 G_t − V_ψ(s_t)"].append(-flat_grad(policy, -(lp * adv_b).sum() * scale))
    env.close()

    stacked = {k: torch.stack(v, dim=0) for k, v in grads.items()}  # 各 [K, P]
    ref_mean = stacked["③ 带基线 G_t − V_ψ(s_t)"].mean(dim=0)
    base_var = stacked["③ 带基线 G_t − V_ψ(s_t)"].var(dim=0, correction=1).sum().item()

    print(f"  冻结策略当前平均回报 ≈ {np.mean(sample_rets):.1f}，重复采样 K = {n_batches} 批（每批 {episodes_per_batch} 回合）：")
    print(f"  {'权重形式':<26}{'参数梯度总方差 Σ Var':>22}{'相对带基线的方差倍数':>22}{'均值方向余弦相似度':>20}")
    var_summary = {}
    for name, mat in stacked.items():
        tot_var = mat.var(dim=0, correction=1).sum().item()
        cos = F.cosine_similarity(mat.mean(dim=0).unsqueeze(0), ref_mean.unsqueeze(0)).item()
        var_summary[name] = (tot_var, tot_var / base_var, cos)
        print(f"  {name:<26}{tot_var:>22.1f}{tot_var / base_var:>21.2f}x{cos:>20.4f}")
    return var_summary


# ==================== 第三部分：三种权重的学习曲线对比 ====================


def train_mode(mode, seed, n_updates, episodes_per_update=4, lr_p=3e-3, lr_v=5e-3):
    seed_everything(seed)
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed + 100)
    policy = PolicyNet()
    value_net = ValueNet()
    p_opt = torch.optim.Adam(policy.parameters(), lr=lr_p)
    v_opt = torch.optim.Adam(value_net.parameters(), lr=lr_v)

    curve = []
    for _ in range(n_updates):
        obs_b, act_b, g0_b, gt_b, ep_ret = collect_batch(env, policy, episodes_per_update, rng)
        curve.append(ep_ret)
        if mode == "g0":
            weights = g0_b
        elif mode == "gt":
            weights = gt_b
        else:
            for _ in range(5):
                v_loss = F.mse_loss(value_net(obs_b), gt_b)
                v_opt.zero_grad()
                v_loss.backward()
                v_opt.step()
            with torch.no_grad():
                weights = gt_b - value_net(obs_b)

        dist = Categorical(logits=policy(obs_b))
        p_loss = -(dist.log_prob(act_b) * weights.detach()).mean()
        p_opt.zero_grad()
        p_loss.backward()
        p_opt.step()

    env.close()
    return np.array(curve)


def compare_learning_curves(var_summary, n_seeds, n_updates, ep_per_upd):
    print("\n" + "=" * 60)
    print("第四部分：三种权重在 CartPole-v1 上的学习曲线对比")
    print("=" * 60)
    modes = [("① 全轨迹回报 G_0", "g0"), ("② reward-to-go G_t", "gt"), ("③ 带基线 G_t − V_ψ(s_t)", "baseline")]
    ep_axis = np.arange(1, n_updates + 1) * ep_per_upd
    results = {}

    for label, key in modes:
        t0 = time.time()
        runs = np.stack([train_mode(key, s, n_updates, ep_per_upd) for s in range(n_seeds)], axis=0)
        results[label] = runs
        print(f"  {label:<26} 最后 10 批均值 = {runs[:, -10:].mean():6.1f} ± {runs[:, -10:].mean(axis=1).std():5.1f}  [{time.time() - t0:.1f}s]")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))
    names = list(var_summary.keys())
    ratios = [var_summary[k][1] for k in names]
    short_names = ["G_0\n(全轨迹)", "G_t\n(reward-to-go)", "G_t − V_ψ\n(带基线)"]
    bars = axes[0].bar(short_names, ratios, color=["C3", "C1", "C0"], width=0.5)
    for idx, v in enumerate(ratios):
        axes[0].text(idx, v + max(ratios) * 0.02, f"{v:.1f}x", ha="center", fontsize=9.5)
    axes[0].set_ylim(0, max(ratios) * 1.18)
    axes[0].set_ylabel("参数梯度总方差（以带基线 = 1x 为基准）")
    axes[0].set_title("同一策略下三种估计器的梯度方差之比")

    for (label, _), color in zip(modes, ("C3", "C1", "C0")):
        arr = results[label]
        m, sd = arr.mean(axis=0), arr.std(axis=0)
        axes[1].plot(ep_axis, m, linewidth=2, color=color, label=label)
        axes[1].fill_between(ep_axis, m - sd, m + sd, color=color, alpha=0.16)
    axes[1].axhline(500.0, color="gray", linestyle=":")
    axes[1].set_xlabel("训练回合数")
    axes[1].set_ylabel("每批平均回报")
    axes[1].set_title(f"学习曲线对比（{n_seeds} 种子均值 ± 标准差）")
    axes[1].legend(fontsize=8.5)
    save_figure(fig, OUT / "02-variance-and-curves.png")
    return results


# ==================== 第四部分：练习与思考 ====================


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：如果不小心把 adv = gt_b - value_net(obs_b) 写在了 no_grad 外面，并且把 p_loss + v_loss 加在一起 backward，会发生什么？

练习 1 参考答案：
    p_loss = -mean(log π · (G_t - V_ψ)) 对 V_ψ 也有梯度 ∂p_loss / ∂V_ψ = +log π / M（因为 log π < 0，它会拼命把 V_ψ 往下拽！）。
    价值网络不再单纯拟合真实回报，而是被策略损失带偏。所以优势计算里的 V_ψ 必须 detach（CONCEPT.md 第 3.5 节）。

思考题 1：既然减掉状态价值 V(s_t) 已经把方差降了这么多，为什么下一节还要学 Actor-Critic？

思考题 1 参考答案：
    G_t − V_ψ(s_t) 里的 G_t 仍然是第 t 步往后直到回合结束所有随机奖励的累加，
    回合长到几百上千步时 G_t 本身的波动依然很大，而且必须等整个回合跑完才能更新。
    下一节把 G_t 换成一步自举 r_t + γ V_ψ(s_{t+1})，方差再降一截，还能不等回合结束在线更新。
    """)


def main():
    seed_everything(SEED)
    quick = is_quick()
    if quick:
        print("（冒烟模式：采样批数与训练轮数缩小，仅验证流程跑通）\n")
    n_batches = 12 if quick else 80
    n_seeds, n_updates, ep_per_upd = (2, 20, 2) if quick else (5, 110, 4)

    introduction()
    verify_hand_calculations()
    var_summary = measure_gradient_variance(n_batches=n_batches)
    compare_learning_curves(var_summary, n_seeds=n_seeds, n_updates=n_updates, ep_per_upd=ep_per_upd)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：03-actor-critic.py（用自举代替蒙特卡洛回报 + 熵正则）")
    print("=" * 60)


if __name__ == "__main__":
    main()
