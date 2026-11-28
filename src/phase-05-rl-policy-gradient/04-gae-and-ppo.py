"""
广义优势估计（GAE）与近端策略优化（PPO）
========================================

学习目标：
    1. 实现 GAE 的反向递推 compute_gae()，严格区分 terminated 与 done 两个掩码，通过第 5.4 节手算单元测试
    2. 实现 ppo_clip_loss()，通过第 6.4 节五样本手算单元测试，理解悲观裁剪为什么允许同一批数据多轮复用
    3. 在同一批轨迹上实测 λ ∈ {0, 0.8, 0.95, 1.0} 的优势方差与学习表现
    4. 在离散任务 CartPole-v1 与连续任务 Pendulum-v1 上跑通 PPO，并对照 epoch 数 K ∈ {1, 4, 10} 与去掉裁剪的后果

核心概念：
    - 广义优势估计（GAE）：Â_t = δ_t + γλ (1 − done_t) Â_{t+1}，在一步 TD（λ=0）与蒙特卡洛优势（λ=1）间平滑折中
    - GAE 与 λ-return 恒等式：G_t^λ = Â_t^GAE + V(s_t)
    - 重要性采样比率：ρ_t(φ) = exp(log π_φ(a_t|s_t) − log π_old(a_t|s_t))
    - PPO-clip：min(ρ Â, clip(ρ, 1−ε, 1+ε) Â)，朝着有利方向走出信任区间后停止提供梯度

前置知识：
    - Phase 4 的 04-n-step-and-lambda-returns.py（λ-return）
    - 03-actor-critic.py（同步并行环境与截断自举）

对应概念文档：
    - CONCEPT.md 第 5 节「广义优势估计（GAE）」、第 6 节「PPO」

预计运行时间：约 1 分钟（CPU）
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
from torch.distributions import Categorical, Normal

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-05")
SEED = 0


# ==================== 第一部分：导读与手算单元测试 ====================


def introduction():
    print("=" * 60)
    print("第一部分：GAE 调好优势估计，PPO 榨干每批在策略数据")
    print("=" * 60)
    print("""
前几节留下的两个痛点，这一节一次性补齐：
  1. 优势估计的偏差—方差旋钮（GAE）：
       δ_t = r_t + γ (1 − terminated_t) V(s_{t+1}) − V(s_t)
       Â_t = δ_t + γλ (1 − done_t) Â_{t+1}，   价值目标 R̂_t = Â_t + V(s_t) = G_t^λ

  2. 在同一批数据上安全更新 K 个 epoch（PPO-clip）：
       ρ_t = π_φ(a_t | s_t) / π_old(a_t | s_t)
       L^CLIP = E[ min( ρ_t Â_t,  clip(ρ_t, 1−ε, 1+ε) Â_t ) ]
     超过 1±ε 的好处一律封顶（梯度归零），超过范围的坏处照单全收（还能拉回来）。
    """)


def compute_gae(rewards, values, next_values, terminated, done, gamma=0.99, gae_lambda=0.95):
    """计算一段 [T, N] 轨迹的 GAE 优势与 λ-return 价值目标（CONCEPT.md 第 5.6、5.7 节）。

    注意此处把 next_values [T, N] 显式传入（而不是直接用 values[t+1]），
    这样当第 t 步发生 truncated 自动重置时，next_values[t] 可以精确填入真实末态的价值，
    同时 1 - done[t] = 0 会把递推的 adv 清零，防止下一个回合的 TD 误差倒灌回上一个回合！
    """
    t_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    adv = torch.zeros_like(rewards[0])
    for t in range(t_steps - 1, -1, -1):
        delta = rewards[t] + gamma * (1.0 - terminated[t]) * next_values[t] - values[t]
        adv = delta + gamma * gae_lambda * (1.0 - done[t]) * adv
        advantages[t] = adv
    value_targets = advantages + values
    return advantages, value_targets


def ppo_clip_loss(log_probs, log_probs_old, advantages, clip_eps=0.2):
    """PPO 裁剪策略损失（要最小化的负目标，CONCEPT.md 第 6.6 节），同时返回近似 KL 与被裁剪比例。"""
    ratio = torch.exp(log_probs - log_probs_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        approx_kl = (log_probs_old - log_probs).mean().item()
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
    return loss, approx_kl, clip_frac


def verify_hand_calculations():
    """严格核对 CONCEPT.md 第 5.4 节（GAE）与第 6.4 节（PPO 裁剪）的手算例子。"""
    print("\n" + "=" * 60)
    print("第二部分：核对 CONCEPT.md 第 5.4、6.4 节手算单元测试")
    print("=" * 60)

    rew = torch.tensor([[1.0], [0.0], [2.0]])
    val = torch.tensor([[2.0], [1.5], [1.8]])
    nxt_val = torch.tensor([[1.5], [1.8], [0.0]])
    term = torch.tensor([[0.0], [0.0], [1.0]])
    done = term.clone()

    for lam, expected in [(0.0, [0.35, 0.12, 0.20]), (0.8, [0.540, 0.264, 0.200]), (1.0, [0.62, 0.30, 0.20])]:
        adv, v_targ = compute_gae(rew, val, nxt_val, term, done, gamma=0.9, gae_lambda=lam)
        vals_list = [round(x, 3) for x in adv.squeeze(1).tolist()]
        print(f"  GAE (λ = {lam:<3})：Â = {vals_list}")
        assert torch.allclose(adv.squeeze(1), torch.tensor(expected), atol=1e-3)
        if lam == 0.8:
            assert torch.allclose(v_targ.squeeze(1), torch.tensor([2.540, 1.764, 2.000]), atol=1e-3)
            print(f"               R̂ = Â + V = {[round(x, 3) for x in v_targ.squeeze(1).tolist()]}（与 Phase 4 λ-return 递推完全一致 ✓）")

    # 第 6.4 节五样本 PPO 裁剪例子
    adv_5 = torch.tensor([2.0, 2.0, -2.0, -2.0, 2.0])
    rho_5 = torch.tensor([1.5, 0.5, 0.5, 1.5, 1.1])
    lp_old = torch.zeros(5)
    lp_new = torch.log(rho_5).requires_grad_(True)
    loss_5, _, cfrac_5 = ppo_clip_loss(lp_new, lp_old, adv_5, clip_eps=0.2)
    loss_5.backward()
    grads_zero = [i + 1 for i, g in enumerate(lp_new.grad.tolist()) if abs(g) < 1e-7]
    print(f"\n  PPO-clip 五样本例子：L_clip 目标值 = {-loss_5.item():.2f}（对应损失 {loss_5.item():+.2f}），"
          f"梯度归零的样本编号 = {grads_zero} ✓")
    assert abs(loss_5.item() - (-0.20)) < 1e-6 and grads_zero == [1, 3]


# ==================== 第二部分：离散与连续通用 PPO 实现 ====================


class DiscreteActorCritic(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_actions))
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def get_dist_and_value(self, obs):
        return Categorical(logits=self.actor(obs)), self.critic(obs).squeeze(-1)

    def act_deterministic(self, obs):
        return self.actor(obs).argmax(dim=-1)


class ContinuousActorCritic(nn.Module):
    """连续动作高斯策略（Pendulum-v1 动作范围 [-2, 2]；策略输出均值后乘 2，或环境外缩放）。"""

    def __init__(self, obs_dim=3, action_dim=1, hidden=64):
        super().__init__()
        self.actor_mean = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, action_dim))
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def get_dist_and_value(self, obs):
        mean = self.actor_mean(obs)
        std = self.log_std.clamp(-2.5, 1.0).exp().expand_as(mean)
        return Normal(mean, std), self.critic(obs).squeeze(-1)

    def act_deterministic(self, obs):
        return self.actor_mean(obs).clamp(-2.0, 2.0)


def collect_rollouts(envs, model, rollout_steps, continuous=False):
    obs = envs._last_obs
    buf = {k: [] for k in ("obs", "actions", "log_probs", "values", "next_values", "rewards", "terminated", "done")}
    for _ in range(rollout_steps):
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        with torch.no_grad():
            dist, val = model.get_dist_and_value(obs_t)
            act = dist.sample()
            lp = dist.log_prob(act).sum(-1) if continuous else dist.log_prob(act)

        env_act = act.clamp(-2.0, 2.0).numpy() if continuous else act.numpy()
        next_obs, rew, term, trunc, infos = envs.step(env_act)
        done = term | trunc

        real_next = next_obs.copy()
        if np.any(done) and "final_obs" in infos:
            for i in np.where(done)[0]:
                if infos["final_obs"][i] is not None:
                    real_next[i] = infos["final_obs"][i]
        with torch.no_grad():
            _, nxt_val = model.get_dist_and_value(torch.from_numpy(real_next.astype(np.float32)))

        for k, v in [
            ("obs", obs_t), ("actions", act), ("log_probs", lp), ("values", val),
            ("next_values", nxt_val), ("rewards", torch.from_numpy(rew.astype(np.float32))),
            ("terminated", torch.from_numpy(term.astype(np.float32))),
            ("done", torch.from_numpy(done.astype(np.float32))),
        ]:
            buf[k].append(v)
        obs = next_obs

    envs._last_obs = obs
    return {k: torch.stack(v, dim=0) for k, v in buf.items()}


def ppo_update(model, optimizer, buf, continuous=False, gamma=0.99, gae_lambda=0.95,
               n_epochs=4, batch_size=64, clip_eps=0.2, use_clip=True, c_ent=0.01):
    """CONCEPT.md 第 6.5 节完整 PPO 更新。use_clip=False 时退化为无裁剪的重要性采样目标 L^IS。"""
    adv, v_targ = compute_gae(
        buf["rewards"], buf["values"], buf["next_values"], buf["terminated"], buf["done"],
        gamma=gamma, gae_lambda=gae_lambda,
    )
    obs_f = buf["obs"].flatten(0, 1)
    act_f = buf["actions"].flatten(0, 1)
    lp_old_f = buf["log_probs"].flatten(0, 1).detach()
    adv_f = adv.flatten(0, 1).detach()
    vtarg_f = v_targ.flatten(0, 1).detach()

    n_total = obs_f.shape[0]
    kls, fracs = [], []
    for _ in range(n_epochs):
        perm = torch.randperm(n_total)
        for start in range(0, n_total, batch_size):
            idx = perm[start : start + batch_size]
            mb_adv = adv_f[idx]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            dist, vals = model.get_dist_and_value(obs_f[idx])
            lp = dist.log_prob(act_f[idx]).sum(-1) if continuous else dist.log_prob(act_f[idx])
            ent = dist.entropy().sum(-1).mean() if continuous else dist.entropy().mean()

            if use_clip:
                p_loss, kl, cfrac = ppo_clip_loss(lp, lp_old_f[idx], mb_adv, clip_eps=clip_eps)
            else:
                ratio = torch.exp((lp - lp_old_f[idx]).clamp(-10.0, 10.0))
                p_loss = -(ratio * mb_adv).mean()
                kl = (lp_old_f[idx] - lp).detach().mean().item()
                cfrac = ((ratio.detach() - 1.0).abs() > clip_eps).float().mean().item()

            v_loss = F.mse_loss(vals, vtarg_f[idx])
            loss = p_loss + 0.5 * v_loss - c_ent * ent
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            kls.append(kl)
            fracs.append(cfrac)
    return float(adv_f.var().item()), float(np.mean(kls)), float(np.mean(fracs))


@torch.no_grad()
def evaluate_env(env_id, model, eval_seeds, continuous=False):
    env = gym.make(env_id)
    rets = []
    for s in eval_seeds:
        obs, _ = env.reset(seed=int(s))
        tot = 0.0
        while True:
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            a = model.act_deterministic(obs_t).squeeze(0).numpy()
            if not continuous:
                a = int(a)
            obs, r, term, trunc, _ = env.step(a)
            tot += float(r)
            if term or trunc:
                break
        rets.append(tot)
    env.close()
    return float(np.mean(rets))


def train_ppo_run(env_id, seed, total_steps, gae_lambda=0.95, n_epochs=4, use_clip=True,
                  n_envs=8, rollout_steps=128, lr=1e-3):
    seed_everything(seed)
    continuous = env_id == "Pendulum-v1"
    envs = gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(n_envs)])
    obs0, _ = envs.reset(seed=seed * 100)
    envs._last_obs = obs0

    model = ContinuousActorCritic() if continuous else DiscreteActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    eval_seeds = [4000 + i for i in range(6)]
    c_ent = 0.0 if continuous else 0.01
    # Pendulum 奖励范围大（每步约 -16 ~ 0），缩放奖励让 critic 快速拟合
    rew_scale = 0.1 if continuous else 1.0

    steps = 0
    curve, kl_hist = [], []
    while steps < total_steps:
        buf = collect_rollouts(envs, model, rollout_steps, continuous=continuous)
        buf_scaled = dict(buf, rewards=buf["rewards"] * rew_scale)
        _, kl, _ = ppo_update(
            model, optimizer, buf_scaled, continuous=continuous, gae_lambda=gae_lambda,
            n_epochs=n_epochs, use_clip=use_clip, c_ent=c_ent,
        )
        steps += n_envs * rollout_steps
        curve.append((steps, evaluate_env(env_id, model, eval_seeds, continuous=continuous)))
        kl_hist.append(kl)
    envs.close()
    return curve, float(np.mean(kl_hist))


# ==================== 第三部分：两组核心消融实验 ====================


def run_all_experiments(quick):
    print("\n" + "=" * 60)
    print("第三部分：实验一——GAE 的 λ 对优势方差与最终回报的影响")
    print("=" * 60)

    n_seeds = 1 if quick else 3
    cp_steps = 2048 if quick else 20_480
    pend_steps = 2048 if quick else 40_960

    # 先固定同一批刚采出的轨迹，直接对比 λ ∈ {0, 0.8, 0.95, 1.0} 算出的 Â_t 方差
    envs_tmp = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(8)])
    o0, _ = envs_tmp.reset(seed=0)
    envs_tmp._last_obs = o0
    m_tmp = DiscreteActorCritic()
    same_buf = collect_rollouts(envs_tmp, m_tmp, 128, continuous=False)
    envs_tmp.close()

    lambdas = [0.0, 0.8, 0.95, 1.0]
    lam_vars, lam_finals = {}, {}
    print(f"  {'GAE λ':<10}{'同一批数据上 Var(Â_t)':>24}{'CartPole 终点评估回报':>26}")
    for lam in lambdas:
        adv, _ = compute_gae(same_buf["rewards"], same_buf["values"], same_buf["next_values"],
                             same_buf["terminated"], same_buf["done"], gae_lambda=lam)
        lam_vars[lam] = adv.var().item()
        finals = [train_ppo_run("CartPole-v1", s, cp_steps, gae_lambda=lam)[0][-1][1] for s in range(n_seeds)]
        lam_finals[lam] = (float(np.mean(finals)), float(np.std(finals)))
        print(f"  {lam:<10.2f}{lam_vars[lam]:>24.3f}{lam_finals[lam][0]:>20.1f} ± {lam_finals[lam][1]:.1f}")

    print("\n" + "=" * 60)
    print("第四部分：实验二——Epoch 数 K 与裁剪开关对比，及连续控制 Pendulum-v1")
    print("=" * 60)
    ppo_configs = [
        ("PPO (K=1, clip)", 1, True),
        ("PPO (K=4, clip)", 4, True),
        ("PPO (K=10, clip)", 10, True),
        ("无裁剪 IS (K=10, no clip)", 10, False),
    ]
    epoch_curves = {}
    for label, k_ep, clip_on in ppo_configs:
        runs, kls = [], []
        for s in range(n_seeds):
            c, m_kl = train_ppo_run("CartPole-v1", s, cp_steps, n_epochs=k_ep, use_clip=clip_on, lr=2.5e-3)
            xs = [st for st, _ in c]
            runs.append([v for _, v in c])
            kls.append(m_kl)
        arr = np.array(runs)
        epoch_curves[label] = (xs, arr)
        print(f"  {label:<28} 终点回报 = {arr[:, -1].mean():6.1f} ± {arr[:, -1].std():5.1f} ｜ 平均近似 KL = {np.mean(kls):+.4f}")

    t_p = time.time()
    pend_runs = []
    for s in range(n_seeds):
        c, _ = train_ppo_run("Pendulum-v1", s, pend_steps, n_epochs=8, rollout_steps=256, lr=2e-3)
        p_xs = [st for st, _ in c]
        pend_runs.append([v for _, v in c])
    pend_arr = np.array(pend_runs)
    print(f"\n  连续控制 Pendulum-v1 PPO（{pend_steps} 步）：起点 {pend_arr[:, 0].mean():.1f} → "
          f"终点 {pend_arr[:, -1].mean():.1f} ± {pend_arr[:, -1].std():.1f}  [{time.time() - t_p:.1f}s]")

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))
    axes[0].bar([f"λ={l}" for l in lambdas], [lam_vars[l] for l in lambdas], color="C2", width=0.5)
    axes[0].set_ylabel("同一批轨迹上优势估计的方差")
    axes[0].set_title("GAE：λ 越大，累积的远期方差越大")

    for (label, _, _), col in zip(ppo_configs, ("C1", "C0", "C2", "C3")):
        xs, arr = epoch_curves[label]
        ls = "--" if "no clip" in label else "-"
        axes[1].plot(xs, arr.mean(0), linestyle=ls, linewidth=2, color=col, label=label)
    axes[1].set_xlabel("CartPole-v1 环境步数")
    axes[1].set_ylabel("评估回报")
    axes[1].set_title("复用次数 K 与悲观裁剪的作用")
    axes[1].legend(fontsize=7.8)

    axes[2].plot(p_xs, pend_arr.mean(0), color="C0", linewidth=2.2, label="PPO（高斯策略）")
    axes[2].fill_between(p_xs, pend_arr.mean(0) - pend_arr.std(0), pend_arr.mean(0) + pend_arr.std(0), color="C0", alpha=0.2)
    axes[2].set_xlabel("Pendulum-v1 环境步数")
    axes[2].set_ylabel("评估回报（越高越好）")
    axes[2].set_title("PPO 在连续动作任务上的学习曲线")
    axes[2].legend(fontsize=8.5)
    save_figure(fig, OUT / "04-gae-and-ppo.png")


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把 compute_gae 里的 (1 - done[t]) 错写成 (1 - terminated[t])，在 200 步固定长度的 Pendulum-v1（永远 truncated、从不 terminated）上会发生什么？

练习 1 参考答案：
    terminated 恒为 0，1 - terminated 恒为 1，于是第 1 个回合第 200 步的优势会乘上 γλ 之后直接把第 2 个回合第 1 步（随机 reset 起点！）的 TD 误差加进来。
    跨回合污染了末尾十来步的优势估计。

思考题 1：对比 Phase 1 的 PETS（约 1200 环境步把 Pendulum 甩到 −150）与本节的 PPO（需要几万环境步才逐渐爬升），为什么 PPO 费这么多样本？

思考题 1 参考答案：
    PPO 每轮收集的几千步数据用完 8 个 epoch 就彻底丢弃，而且它只看标量奖励，完全不利用状态转移规律；
    但它的每次决策只需一次极小 MLP 前向（微秒级），仿真器够快时墙上时间反而非常短。
    """)


def main():
    seed_everything(SEED)
    quick = is_quick()
    if quick:
        print("（冒烟模式：环境步数缩小，仅验证流程跑通）\n")
    introduction()
    verify_hand_calculations()
    run_all_experiments(quick)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：05-gradient-estimators.py（三种梯度估计器统一剖析）")
    print("=" * 60)


if __name__ == "__main__":
    main()
