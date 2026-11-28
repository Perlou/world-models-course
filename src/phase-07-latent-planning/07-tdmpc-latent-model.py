"""
TD-MPC：无解码器隐空间模型 + 带终端价值的 MPPI 规划
==================================================

学习目标：
    1. 核对 CONCEPT.md 第 7.3、7.5、7.6 节的全部手算例子：
       - TD 目标 5.554 与双 Q 回归损失 0.368；H=2 加权总损失 0.486
       - 带终端价值的规划打分：序列甲 10.36 胜过短期奖励更高的序列乙 6.63（终端价值占比 70%）
       - 精英内 MPPI 加权均值 0.360 与加权标准差 0.240
    2. 实现完整 TD-MPC（LatentModel、td_target()、tdmpc_loss()、update_policy()、estimate_value()、plan()），
       明确落地三处 stop-gradient（TD 目标、EMA 编码器一致性目标、策略输入 detach）
    3. 在 Pendulum-v1 上完整训练 TD-MPC，并与两个关键消融对照：
       去掉终端价值（退化为纯短视野 MPC）与去掉隐状态一致性损失

核心概念：
    - 面向任务的隐动力学（TOLD）：不重建观测，靠奖励 + 双 Q TD + EMA 编码器隐态一致性联合塑造表征
    - 带终端价值与策略先验的 MPPI：前 H 步靠模型展开累加奖励，H 步之后靠 Q(z^H, π(z^H)) 兜底，策略提供部分候选序列

前置知识：
    - Phase 1 的 06-cem-and-mppi.py（MPPI 与热启动）
    - 01-decoder-free-and-value-equivalence.py

对应概念文档：
    - CONCEPT.md 第 7 节「TD-MPC：无解码器隐模型 + MPPI」

预计运行时间：约 2 分钟（CPU）
"""

import copy
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
from utils.planning import shift_plan

plt = setup_matplotlib()
OUT = output_dir("phase-07")
SEED = 0


# ==================== 第一部分：手算单元测试 ====================


def verify_section_7_hand_calcs():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 7.3、7.5、7.6 节手算例子")
    print("=" * 60)
    # 1. 第 7.3 节 TD 目标与损失
    y = 1.0 + 0.99 * min(5.0, 4.6)
    lq = (5.0 - y) ** 2 + (5.8 - y) ** 2
    tot = (0.5 * 0.04 + 0.1 * 0.25 + 2.0 * 0.10) + 0.5 * (0.5 * 0.09 + 0.1 * 0.36 + 2.0 * 0.20)
    print(f"  1. TD 目标 y = {y:.3f}（手算 5.554），双 Q 损失 = {lq:.3f}（手算 0.368），H=2 总损失 = {tot:.3f}（手算 0.486）✓")
    assert abs(y - 5.554) < 1e-3 and abs(lq - 0.368) < 2e-3 and abs(tot - 0.486) < 2e-3

    # 2. 第 7.5 节终端价值逆转排序
    j_a = 1.0 + 0.9 * 0.5 + 0.81 * 2.0 + 0.729 * 10.0
    j_b = 2.0 + 0.9 * 1.0 + 0.81 * 1.0 + 0.729 * 4.0
    print(f"  2. 规划打分：序列甲 = {j_a:.2f}（手算 10.36）> 序列乙 = {j_b:.2f}（手算 6.63），终端占比 {0.729 * 10 / j_a * 100:.1f}% ✓")
    assert abs(j_a - 10.36) < 0.02 and abs(j_b - 6.63) < 0.02

    # 3. 第 7.6 节精英内 MPPI 加权更新
    scores = torch.tensor([10.0, 9.0, 6.0])
    acts = torch.tensor([0.5, 0.3, -0.4])
    w = torch.softmax((scores - scores.max()) / 2.0, dim=0)
    mu = (w * acts).sum().item()
    std = torch.sqrt((w * (acts - mu).pow(2)).sum()).item()
    print(f"  3. MPPI 加权更新（λ=2）：μ = {mu:.3f}（手算 0.360），σ = {std:.3f}（手算 0.240）✓")
    assert abs(mu - 0.360) < 2e-3 and abs(std - 0.240) < 2e-3


# ==================== 第二部分：TD-MPC 模型与 MPPI 规划器 ====================


class LatentModel(nn.Module):
    """CONCEPT.md 第 7.2 节：enc、dyn、reward、双 Q 与确定性策略先验 pi。"""

    def __init__(self, obs_dim, act_dim, latent_dim=48, hidden=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ELU(), nn.Linear(hidden, latent_dim))
        self.dyn = nn.Sequential(nn.Linear(latent_dim + act_dim, hidden), nn.ELU(), nn.Linear(hidden, latent_dim))
        self.reward = nn.Sequential(nn.Linear(latent_dim + act_dim, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.q1 = nn.Sequential(nn.Linear(latent_dim + act_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(nn.Linear(latent_dim + act_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.pi = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ELU(), nn.Linear(hidden, act_dim), nn.Tanh())

    def encode(self, obs):
        return self.enc(obs)

    def next_latent(self, z, a):
        return self.dyn(torch.cat([z, a], dim=-1))

    def pred_reward(self, z, a):
        return self.reward(torch.cat([z, a], dim=-1)).squeeze(-1)

    def pred_q(self, z, a):
        za = torch.cat([z, a], dim=-1)
        return self.q1(za).squeeze(-1), self.q2(za).squeeze(-1)

    def act_prior(self, z):
        return self.pi(z)


class SeqReplayBuffer:
    """按回合存轨迹，随机抽取长度为 H 的连续片段（obs [B, H+1, O], act [B, H, A], rew [B, H]）。"""

    def __init__(self):
        self.episodes = []

    def add_episode(self, obs_arr, act_arr, rew_arr):
        self.episodes.append((
            np.asarray(obs_arr, dtype=np.float32),
            np.asarray(act_arr, dtype=np.float32),
            np.asarray(rew_arr, dtype=np.float32),
        ))

    def sample(self, batch_size, horizon, rng):
        ob, ab, rb = [], [], []
        for _ in range(batch_size):
            o_e, a_e, r_e = self.episodes[int(rng.integers(0, len(self.episodes)))]
            s = int(rng.integers(0, len(r_e) - horizon + 1))
            ob.append(o_e[s : s + horizon + 1])
            ab.append(a_e[s : s + horizon])
            rb.append(r_e[s : s + horizon])
        return torch.from_numpy(np.stack(ob)), torch.from_numpy(np.stack(ab)), torch.from_numpy(np.stack(rb))


@torch.no_grad()
def td_target(model, target_model, next_obs, rew, gamma=0.99):
    z_tilde = model.encode(next_obs)
    a_next = model.act_prior(z_tilde)
    q1, q2 = target_model.pred_q(z_tilde, a_next)
    return rew + gamma * torch.min(q1, q2)


def tdmpc_loss(model, target_model, obs_seq, act_seq, rew_seq, gamma=0.99, rho=0.5, c_r=0.5, c_q=0.1, c_z=2.0):
    """CONCEPT.md 第 7.3 节：展开 H 步累加奖励、双 Q TD 与隐态一致性三项联合损失。"""
    H = act_seq.shape[1]
    z = model.encode(obs_seq[:, 0])
    zs_detached = []
    total_loss = 0.0

    for i in range(H):
        zs_detached.append(z.detach())
        a_i = act_seq[:, i]
        r_i = rew_seq[:, i]
        o_nxt = obs_seq[:, i + 1]

        with torch.no_grad():
            y_i = td_target(model, target_model, o_nxt, r_i, gamma=gamma)
            z_tgt = target_model.encode(o_nxt)

        r_pred = model.pred_reward(z, a_i)
        q1, q2 = model.pred_q(z, a_i)
        z = model.next_latent(z, a_i)

        l_r = F.mse_loss(r_pred, r_i)
        l_q = F.mse_loss(q1, y_i) + F.mse_loss(q2, y_i)
        l_z = F.mse_loss(z, z_tgt)
        total_loss = total_loss + (rho**i) * (c_r * l_r + c_q * l_q + c_z * l_z)

    return total_loss, torch.stack(zs_detached, dim=1)  # [B, H, D]


def update_policy(model, zs_detached, pi_opt, rho=0.5):
    """CONCEPT.md 第 7.4 节：沿冻结的 Q 反向传播更新策略先验 π，输入的隐状态已 detach。"""
    for p in list(model.q1.parameters()) + list(model.q2.parameters()):
        p.requires_grad_(False)
    H = zs_detached.shape[1]
    loss = 0.0
    for i in range(H):
        z_i = zs_detached[:, i]
        a_pi = model.act_prior(z_i)
        q1, q2 = model.pred_q(z_i, a_pi)
        loss = loss - (rho**i) * torch.min(q1, q2).mean()
    pi_opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.pi.parameters(), 10.0)
    pi_opt.step()
    for p in list(model.q1.parameters()) + list(model.q2.parameters()):
        p.requires_grad_(True)


@torch.no_grad()
def estimate_value(model, z0, actions, gamma=0.99, use_terminal_q=True):
    """CONCEPT.md 第 7.5 节：在隐空间展开 H 步并加上终端价值 γ^H Q(z^H, π(z^H))。"""
    N, H, _ = actions.shape
    z = z0.expand(N, -1)
    total = torch.zeros(N, device=z.device)
    disc = 1.0
    for k in range(H):
        a_k = actions[:, k]
        total = total + disc * model.pred_reward(z, a_k)
        z = model.next_latent(z, a_k)
        disc *= gamma
    if use_terminal_q:
        a_end = model.act_prior(z)
        q1, q2 = model.pred_q(z, a_end)
        total = total + disc * torch.min(q1, q2)
    return total


@torch.no_grad()
def plan(model, obs_np, prev_mean=None, horizon=5, n_samples=160, n_pi=16, n_elites=24,
         n_iters=4, lam=2.0, gamma=0.99, use_terminal_q=True, deterministic=False):
    """CONCEPT.md 第 7.7 节：带策略先验候选、终端价值与热启动的精英 MPPI 规划。"""
    z0 = model.encode(torch.from_numpy(np.asarray(obs_np, dtype=np.float32)[None]))  # [1, D]
    act_dim = model.act_dim

    # 1. 用策略先验在隐空间生成 n_pi 条候选轨迹
    pi_actions = []
    if n_pi > 0:
        z_p = z0.expand(n_pi, -1)
        for _ in range(horizon):
            a_p = (model.act_prior(z_p) + 0.1 * torch.randn(n_pi, act_dim)).clamp(-1.0, 1.0)
            pi_actions.append(a_p)
            z_p = model.next_latent(z_p, a_p)
        pi_actions = torch.stack(pi_actions, dim=1)  # [n_pi, H, A]

    # 2. 热启动均值与初始标准差
    mean = torch.zeros(horizon, act_dim) if prev_mean is None else shift_plan(prev_mean)
    std = torch.full((horizon, act_dim), 0.7)

    for _ in range(n_iters):
        gauss_a = (mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(n_samples, horizon, act_dim)).clamp(-1.0, 1.0)
        cand = torch.cat([gauss_a, pi_actions], dim=0) if n_pi > 0 else gauss_a
        scores = estimate_value(model, z0, cand, gamma=gamma, use_terminal_q=use_terminal_q)
        top_vals, top_idx = scores.topk(min(n_elites, cand.shape[0]))
        elites = cand[top_idx]
        weights = torch.softmax((top_vals - top_vals.max()) / lam, dim=0).view(-1, 1, 1)
        mean = (weights * elites).sum(dim=0)
        std = torch.sqrt((weights * (elites - mean.unsqueeze(0)).pow(2)).sum(dim=0)).clamp(0.05, 1.0)

    a0 = mean[0] if deterministic else (mean[0] + std[0] * torch.randn(act_dim)).clamp(-1.0, 1.0)
    return a0.numpy(), mean


# ==================== 第三部分：Pendulum-v1 训练与双消融 ====================


def train_tdmpc(n_episodes=20, use_terminal_q=True, c_z=2.0, seed=0):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = gym.make("Pendulum-v1")
    act_scale = float(env.action_space.high[0])

    model = LatentModel(obs_dim=3, act_dim=1, latent_dim=48, hidden=128)
    target = copy.deepcopy(model)
    wm_params = [p for n, p in model.named_parameters() if not n.startswith("pi.")]
    wm_opt = torch.optim.Adam(wm_params, lr=2e-3)
    pi_opt = torch.optim.Adam(model.pi.parameters(), lr=1e-3)
    replay = SeqReplayBuffer()

    horizon = 5
    eval_steps, eval_rets = [], []
    init_eps = 3 if is_quick() else 5

    for ep_idx in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed * 100 + ep_idx)
        o_l, a_l, r_l = [obs.astype(np.float32)], [], []
        prev_mean = None
        for _ in range(200):
            if ep_idx <= init_eps:
                a = rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
            else:
                a, prev_mean = plan(model, obs, prev_mean=prev_mean, horizon=horizon,
                                    n_samples=60 if is_quick() else 120, n_pi=8 if is_quick() else 16,
                                    n_elites=12 if is_quick() else 20, n_iters=3 if is_quick() else 4,
                                    use_terminal_q=use_terminal_q, deterministic=False)
            nxt, r, term, trunc, _ = env.step(a * act_scale)
            o_l.append(nxt.astype(np.float32))
            a_l.append(a)
            r_l.append(float(r))
            obs = nxt
            if term or trunc:
                break
        replay.add_episode(o_l, a_l, r_l)

        if ep_idx >= init_eps:
            n_upd = 30 if is_quick() else 80
            for _ in range(n_upd):
                obs_b, act_b, rew_b = replay.sample(64, horizon, rng)
                loss, zs_det = tdmpc_loss(model, target, obs_b, act_b, rew_b, c_z=c_z)
                wm_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(wm_params, 10.0)
                wm_opt.step()
                update_policy(model, zs_det, pi_opt)
                with torch.no_grad():
                    for p, pt in zip(model.parameters(), target.parameters()):
                        pt.data.mul_(0.99).add_(p.data, alpha=0.01)

        if ep_idx >= init_eps and (ep_idx % max(1, (n_episodes - init_eps) // 4) == 0 or ep_idx == n_episodes):
            ev_env = gym.make("Pendulum-v1")
            scores = []
            for ei in range(2 if is_quick() else 5):
                eo, _ = ev_env.reset(seed=900 + ei)
                pm = None
                tot = 0.0
                for _ in range(200):
                    ea, pm = plan(model, eo, prev_mean=pm, horizon=horizon,
                                  n_samples=60 if is_quick() else 120, n_pi=8 if is_quick() else 16,
                                  n_elites=12 if is_quick() else 20, n_iters=3 if is_quick() else 4,
                                  use_terminal_q=use_terminal_q, deterministic=True)
                    eo, er, _, _, _ = ev_env.step(ea * act_scale)
                    tot += float(er)
                scores.append(tot)
            ev_env.close()
            eval_steps.append(ep_idx * 200)
            eval_rets.append(float(np.mean(scores)))

    env.close()
    return np.array(eval_steps), np.array(eval_rets)


def main():
    seed_everything(SEED)
    verify_section_7_hand_calcs()

    quick = is_quick()
    n_eps = 6 if quick else 18

    print("\n" + "=" * 60)
    print(f"第二部分：Pendulum-v1 上运行 TD-MPC 与双消融（共 {n_eps * 200} 真实步，视野 H=5）")
    print("=" * 60)
    t0 = time.time()
    configs = [
        ("完整 TD-MPC (H=5 + 终端 Q + 一致性)", True, 2.0),
        ("去掉终端价值 (仅累加 5 步奖励)", False, 2.0),
        ("去掉隐状态一致性 (c_z = 0)", True, 0.0),
    ]
    curves = {}
    for label, use_q, cz in configs:
        st, rt = train_tdmpc(n_episodes=n_eps, use_terminal_q=use_q, c_z=cz, seed=SEED)
        curves[label] = (st, rt)
        print(f"  {label:<36} → 最高评估回报 {rt.max():7.1f} ｜ 最终回报 {rt[-1]:7.1f}")
    print(f"  总耗时 {time.time() - t0:.1f}s")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for label, (st, rt) in curves.items():
        ax.plot(st, rt, "o-", linewidth=2.2, label=f"{label} ({rt[-1]:.0f})")
    ax.axhline(-200.0, color="seagreen", linestyle="--", label="优秀参考线（−200，对照 PETS）")
    ax.set_xlabel("真实环境交互步数")
    ax.set_ylabel("无噪评估回报")
    ax.set_title("Pendulum-v1：TD-MPC 及其终端价值与一致性消融")
    ax.legend(fontsize=8.5)
    save_figure(fig, OUT / "07-tdmpc-pendulum.png")

    print("\n" + "=" * 60)
    print("本节完成！下一节：08-tdmpc2.py（SimNorm、Q 集成与离散回归：TD-MPC2）")
    print("=" * 60)


if __name__ == "__main__":
    main()
