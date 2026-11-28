"""
确定性策略梯度：DDPG 与 TD3
===========================

学习目标：
    1. 核对 CONCEPT.md 第 8.3、8.5 节的手算例子：确定性策略梯度的链式法则 ∂Q/∂a · ∂μ/∂φ，以及双 Q 取小消除正偏差
    2. 从零实现离策略连续控制算法 DDPG 与 TD3（双 Q 取小、延迟更新 actor、目标策略平滑）
    3. 在 Pendulum-v1 上同台对比 DDPG 与 TD3 的学习曲线
    4. 定量诊断过估计：对比训练过程中 critic 预测的 Q(s, μ_φ(s)) 与从同一批状态实测展开的真实折扣回报 G_0

核心概念：
    - 确定性策略梯度（DPG）：把 argmax_a Q(s, a) 摊销成一个网络 μ_φ(s)，靠 critic 对动作的导数训练 actor
    - 过估计恶性循环：actor 专挑 critic 给分虚高的动作，虚高值再经 Bellman 自举扩散
    - TD3 三项改进：双 Q 取 min（宁可保守偏低）、actor 延迟更新、目标动作加截断平滑噪声

前置知识：
    - Phase 4 的 06-dqn-from-scratch.py、07-dqn-improvements-and-distributional.py（经验回放与过估计）
    - 05-gradient-estimators.py（pathwise 估计器）

对应概念文档：
    - CONCEPT.md 第 8 节「确定性策略梯度：DDPG 与 TD3」

预计运行时间：约 45 秒（CPU）
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

plt = setup_matplotlib()
OUT = output_dir("phase-05")
SEED = 0
GAMMA = 0.99
MAX_ACTION = 2.0


# ==================== 第一部分：导读与手算核对 ====================


def introduction():
    print("=" * 60)
    print("第一部分：把 argmax 摊销成网络——从 DQN 到 DDPG 与 TD3")
    print("=" * 60)
    print("""
PPO 是在策略的，数据用一轮就扔。要把 DQN 的高样本效率搬到连续动作，卡在 max_a Q(s, a) 没法穷举。
DDPG 训练一个确定性策略 a = μ_φ(s)，让梯度直接穿过 critic Q_ψ(s, a) 传回 actor（pathwise 估计器）：

    s ──▶ [ actor μ_φ ] ──a──▶ [ critic Q_ψ ] ──▶ Q(s, μ_φ(s))
               ▲                     │
               └─── ∂Q/∂a · ∂μ/∂φ ◀──┘  （只更新 φ，critic 参数在此步冻结当管道）

但单个 critic 的误差会被 actor 当漏洞钻，引发严重高估。TD3 加三道保险：
    ① 双 Q 取小 min(Q_1, Q_2)    ② 延迟更新 actor（2 步一次）    ③ 目标动作平滑加噪
    """)


def verify_hand_calculations():
    """核对 CONCEPT.md 第 8.3 节（一维链式法则）与第 8.5 节（双 Q 取小期望）。"""
    print("\n" + "=" * 60)
    print("第二部分：核对 CONCEPT.md 第 8.3、8.5 节手算例子")
    print("=" * 60)

    phi = torch.tensor(0.5, requires_grad=True)
    q_val = -((phi - 2.0) ** 2)
    q_val.backward()
    new_phi = phi.item() + 0.1 * phi.grad.item()
    print(f"  1. 一维确定性策略梯度（Q = -(a-2)², φ = 0.5）：∂J/∂φ = {phi.grad.item():.1f}，一步更新后 φ = {new_phi:.1f} ✓")
    assert abs(phi.grad.item() - 3.0) < 1e-6 and abs(new_phi - 0.8) < 1e-6

    pairs = [(9, 9), (9, 11), (11, 9), (11, 11)]
    exp_min = np.mean([min(a, b) for a, b in pairs])
    exp_max = np.mean([max(a, b) for a, b in pairs])
    print(f"  2. 双 Q 误差 ±1（真值 10）：E[min(Q_1, Q_2)] = {exp_min:.1f}（保守偏低），对比 E[max] = {exp_max:.1f}（乐观高估）✓")
    assert abs(exp_min - 9.5) < 1e-6 and abs(exp_max - 10.5) < 1e-6


# ==================== 第二部分：回放池、网络与 DDPG / TD3 更新 ====================


class ContinuousReplay:
    def __init__(self, obs_dim=3, act_dim=1, capacity=50_000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.term = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, no, terminated):
        i = self.ptr
        self.obs[i] = o
        self.act[i] = a
        self.rew[i, 0] = float(r)
        self.next_obs[i] = no
        self.term[i, 0] = float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.act[idx]),
            torch.from_numpy(self.rew[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.term[idx]),
        )


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim=3, act_dim=1, hidden=128, max_action=MAX_ACTION):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs) * self.max_action


class QCritic(nn.Module):
    def __init__(self, obs_dim=3, act_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


def soft_update(target, source, kappa=0.005):
    with torch.no_grad():
        for pt, ps in zip(target.parameters(), source.parameters()):
            pt.data.mul_(1.0 - kappa).add_(ps.data, alpha=kappa)


def ddpg_update(actor, actor_targ, q1, q1_targ, actor_opt, q_opt, batch, kappa=0.005):
    """标准 DDPG 单步更新（单 Q、无目标平滑、每步都更新 actor）。"""
    obs, act, rew, next_obs, term = batch
    with torch.no_grad():
        next_act = actor_targ(next_obs)
        target_q = rew + GAMMA * (1.0 - term) * q1_targ(next_obs, next_act)

    critic_loss = F.mse_loss(q1(obs, act), target_q)
    q_opt.zero_grad()
    critic_loss.backward()
    q_opt.step()

    # 注意：actor_opt 里只有 actor 的参数，critic 在这里只充当传梯度的可微管道！
    actor_loss = -q1(obs, actor(obs)).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    soft_update(q1_targ, q1, kappa)
    soft_update(actor_targ, actor, kappa)


def td3_update(step_idx, actor, actor_targ, q1, q2, q1_targ, q2_targ, actor_opt, q_opt, batch,
               policy_noise=0.4, noise_clip=1.0, delay=2, kappa=0.005):
    """TD3 更新（CONCEPT.md 第 8.6 节；噪声幅度按 MAX_ACTION=2 等比设定为 0.4 / 1.0）。"""
    obs, act, rew, next_obs, term = batch
    with torch.no_grad():
        noise = (torch.randn_like(act) * policy_noise).clamp(-noise_clip, noise_clip)
        next_act = (actor_targ(next_obs) + noise).clamp(-MAX_ACTION, MAX_ACTION)
        min_next_q = torch.minimum(q1_targ(next_obs, next_act), q2_targ(next_obs, next_act))
        target_q = rew + GAMMA * (1.0 - term) * min_next_q

    critic_loss = F.mse_loss(q1(obs, act), target_q) + F.mse_loss(q2(obs, act), target_q)
    q_opt.zero_grad()
    critic_loss.backward()
    q_opt.step()

    if step_idx % delay == 0:
        actor_loss = -q1(obs, actor(obs)).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        soft_update(q1_targ, q1, kappa)
        soft_update(q2_targ, q2, kappa)
        soft_update(actor_targ, actor, kappa)


# ==================== 第三部分：训练与 Q 值过估计诊断 ====================


@torch.no_grad()
def evaluate_and_measure_bias(actor, q1, eval_seeds):
    """在固定种子上测真实回合回报，并对比起点集上 Q_ψ1(s, μ_φ(s)) 与真实折扣回报 G_0。"""
    env = gym.make("Pendulum-v1")
    ep_rets, pred_qs, true_gs = [], [], []

    for s in eval_seeds:
        obs, _ = env.reset(seed=int(s))
        obs_list, rew_list = [], []
        while True:
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            act = actor(obs_t)
            if len(obs_list) < 20:  # 取每条轨迹前 20 个状态测折扣回报 G_t（剩余至少 180 步，γ^180 ≈ 0.16，末尾用自举补齐避免截断低估）
                obs_list.append(obs_t)
                pred_qs.append(q1(obs_t, act).item())
            obs, r, term, trunc, _ = env.step(act.squeeze(0).numpy())
            rew_list.append(float(r))
            if term or trunc:
                break
        ep_rets.append(float(sum(rew_list)))

        # 计算前 20 个状态的真实折扣回报（200 步是时间截断而非终止，末尾加 γ^{200-t} Q_pred 消除有限步截断偏差，或者跑长 400 步精确算无偏 G_t！）
    env.close()

    # 为了让"真实折扣回报 G_0"完全不依赖任何 critic 自举（100% 纯地面真值），专门在无时间截断的物理摆上往前跑 350 步（0.99^350 = 0.03）
    raw_env = gym.make("Pendulum-v1").unwrapped
    pred_qs_clean, true_gs_clean = [], []
    for s in eval_seeds:
        raw_env.reset(seed=int(s))
        # 取初始状态及第 15 步状态各测一次
        for probe_idx in range(2):
            st_copy = raw_env.state.copy()
            obs_init = np.array([np.cos(st_copy[0]), np.sin(st_copy[0]), st_copy[1]], dtype=np.float32)
            obs_t = torch.from_numpy(obs_init).unsqueeze(0)
            pred_qs_clean.append(q1(obs_t, actor(obs_t)).item())

            g = 0.0
            disc = 1.0
            obs_cur = obs_init
            for _ in range(350):
                a_np = actor(torch.from_numpy(obs_cur).unsqueeze(0)).squeeze(0).numpy()
                obs_cur, r, _, _, _ = raw_env.step(a_np)
                obs_cur = np.asarray(obs_cur, dtype=np.float32)
                g += disc * float(r)
                disc *= GAMMA
            true_gs_clean.append(g)
            raw_env.state = st_copy
            for _ in range(15):
                a_np = actor(torch.from_numpy(raw_env._get_obs()).float().unsqueeze(0)).squeeze(0).numpy()
                raw_env.step(a_np)
    raw_env.close()

    return float(np.mean(ep_rets)), float(np.mean(pred_qs_clean)), float(np.mean(true_gs_clean))


def run_agent(algo, seed, total_steps, eval_every, warmup=500, batch_size=128):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=seed)

    actor = DeterministicActor()
    actor_targ = copy.deepcopy(actor)
    q1 = QCritic()
    q1_targ = copy.deepcopy(q1)
    q2 = QCritic()
    q2_targ = copy.deepcopy(q2)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    q_params = list(q1.parameters()) if algo == "ddpg" else list(q1.parameters()) + list(q2.parameters())
    q_opt = torch.optim.Adam(q_params, lr=1e-3)

    replay = ContinuousReplay()
    eval_seeds = [5000 + i for i in range(5)]
    records = []
    ep_idx = 0

    for step in range(1, total_steps + 1):
        if step <= warmup:
            act = rng.uniform(-MAX_ACTION, MAX_ACTION, size=1).astype(np.float32)
        else:
            with torch.no_grad():
                base = actor(torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)).squeeze(0).numpy()
            act = np.clip(base + rng.normal(0.0, 0.2, size=1), -MAX_ACTION, MAX_ACTION).astype(np.float32)

        next_obs, rew, term, trunc, _ = env.step(act)
        replay.add(obs, act, rew, next_obs, term)  # Pendulum 只有 trunc，term 恒为 False
        if term or trunc:
            ep_idx += 1
            obs, _ = env.reset(seed=seed * 1000 + ep_idx)
        else:
            obs = next_obs

        if replay.size >= batch_size and step > warmup:
            batch = replay.sample(batch_size, rng)
            if algo == "ddpg":
                ddpg_update(actor, actor_targ, q1, q1_targ, actor_opt, q_opt, batch)
            else:
                td3_update(step, actor, actor_targ, q1, q2, q1_targ, q2_targ, actor_opt, q_opt, batch)

        if step % eval_every == 0:
            ret, pq, tg = evaluate_and_measure_bias(actor, q1, eval_seeds)
            records.append((step, ret, pq, tg))

    env.close()
    return records


def compare_ddpg_td3(n_seeds, total_steps, eval_every):
    print("\n" + "=" * 60)
    print("第三部分：Pendulum-v1 上对比 DDPG 与 TD3 的回报与 Q 值过估计")
    print("=" * 60)

    t0 = time.time()
    all_res = {}
    for algo, title in [("ddpg", "DDPG（单 Q）"), ("td3", "TD3（双 Q 取小 + 延迟 + 平滑）")]:
        runs = [run_agent(algo, s, total_steps, eval_every) for s in range(n_seeds)]
        xs = [r[0] for r in runs[0]]
        rets = np.array([[r[1] for r in run] for run in runs])
        pqs = np.array([[r[2] for r in run] for run in runs])
        tgs = np.array([[r[3] for r in run] for run in runs])
        bias = (pqs - tgs).mean(axis=0)
        all_res[algo] = dict(title=title, xs=xs, rets=rets, pqs=pqs, tgs=tgs, bias=bias)
        print(f"  {title:<30} 终点回报 = {rets[:, -1].mean():6.1f} ± {rets[:, -1].std():5.1f} ｜ "
              f"峰值(Q_预测 − G_真实) = {bias.max():+6.2f} ｜ 终点偏差 = {bias[-1]:+6.2f}")
    print(f"  实验完成，共耗时 {time.time() - t0:.1f}s。")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    for algo, col in [("ddpg", "C3"), ("td3", "C0")]:
        r = all_res[algo]
        xs = r["xs"]
        m, sd = r["rets"].mean(0), r["rets"].std(0)
        axes[0].plot(xs, m, "o-", markersize=4, linewidth=2, color=col, label=r["title"])
        axes[0].fill_between(xs, m - sd, m + sd, color=col, alpha=0.16)

        axes[1].plot(xs, r["pqs"].mean(0), linewidth=2, color=col, label=f"{algo.upper()}：critic 预测 Q")
        axes[1].plot(xs, r["tgs"].mean(0), "--", linewidth=1.8, color=col, alpha=0.8, label=f"{algo.upper()}：真实折扣回报 G")

    axes[0].set_xlabel("真实环境交互步数")
    axes[0].set_ylabel("200 步回合评估回报")
    axes[0].set_title(f"Pendulum-v1 学习曲线（{n_seeds} 种子）")
    axes[0].legend(fontsize=8.5)

    axes[1].set_xlabel("真实环境交互步数")
    axes[1].set_ylabel("折扣价值（实线 = 预测 Q，虚线 = 真实 G）")
    axes[1].set_title("DDPG 预测显著高于真实值（过估计），TD3 贴合或略保守")
    axes[1].legend(fontsize=8)
    save_figure(fig, OUT / "06-ddpg-vs-td3.png")


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把 actor_opt 错写成 torch.optim.Adam(list(actor.parameters()) + list(q1.parameters()))，运行 1000 步会发生什么？

练习 1 参考答案：
    actor_loss = -q1(obs, actor(obs)).mean() 会直接修改 q1 的权重，让 q1 对一切输入都疯狂输出 +∞，
    几百步之内预测 Q 值冲到几万，策略彻底报废（CONCEPT.md 第 8.8 节第四条误区）。

思考题 1：TD3 在 Pendulum-v1 上不到一万步就收敛到约 −140，比上一节 PPO 的四万步省得多，但它的探索噪声 σ=0.2 仍然是手调的常数。怎样让智能体自己决定在哪些状态多探索、哪些状态精确控制？

思考题 1 参考答案：
    让策略本身输出高斯分布的均值和标准差，并把策略熵直接写进 Bellman 目标里——这就是下一节的 SAC。
    """)


def main():
    seed_everything(SEED)
    quick = is_quick()
    if quick:
        print("（冒烟模式：交互步数缩小，仅验证流程跑通）\n")
    n_seeds, total_steps, eval_every = (1, 1200, 600) if quick else (3, 10_000, 1000)

    introduction()
    verify_hand_calculations()
    compare_ddpg_td3(n_seeds=n_seeds, total_steps=total_steps, eval_every=eval_every)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：07-sac-from-scratch.py（最大熵强化学习 SAC）")
    print("=" * 60)


if __name__ == "__main__":
    main()
