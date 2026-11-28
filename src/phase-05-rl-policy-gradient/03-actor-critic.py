"""
Actor-Critic 与熵正则
=====================

学习目标：
    1. 核对 CONCEPT.md 第 4.3、4.4 节的手算例子：TD 优势的正负判据、类别分布与高斯分布的熵
    2. 实现同步优势 Actor-Critic（A2C）：N 个并行环境每次走 T 步即可更新，无需等回合结束
    3. 严格区分 terminated（真正失败倒杆，未来价值为 0）与 truncated（达到 500 步时间上限，仍需 V(s_{t+1}) 自举）
    4. 在相同环境交互步数下对比 A2C 与带基线 REINFORCE，并实测熵正则系数 c_ent 对策略熵与防过早收敛的作用

核心概念：
    - Actor-Critic：actor π_φ 选动作，critic V_ψ 既当基线又用来自举（δ_t = r_t + γ V_ψ(s_{t+1}) − V_ψ(s_t)）
    - 两处停止梯度：actor 损失里的 δ_t 要 detach；critic 损失里的 TD 目标 r_t + γ V_ψ(s_{t+1}) 也要 detach
    - 熵正则（entropy bonus）：在损失中减去 c_ent · H(π)，推迟策略坍缩成确定性动作，给探索留出空间
    - 截断处理：truncated 步不是终止状态，必须拿 final_obs 的价值做自举

前置知识：
    - Phase 4 的 03-monte-carlo-vs-td.py（TD 自举）
    - 02-baselines-and-advantage.py（基线与优势）

对应概念文档：
    - CONCEPT.md 第 4 节「Actor-Critic」

预计运行时间：约 35 秒（CPU）
"""

import math
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
GAMMA = 0.99


# ==================== 第一部分：导读与手算核对 ====================


def introduction():
    print("=" * 60)
    print("第一部分：从蒙特卡洛优势到自举 Actor-Critic")
    print("=" * 60)
    print("""
上一节的权重 G_t − V_ψ(s_t) 只把 V_ψ 当基线，G_t 还是要等回合跑完。
Actor-Critic 把 G_t 换成一步（或 n 步）TD 自举目标，每走几步就能直接更新：

    δ_t = r_t + γ (1 − terminated_t) V_ψ(s_{t+1}) − V_ψ(s_t)
          └────────── TD 目标（critic 追它）──────────┘

    总损失 L = − mean( log π_φ(a_t | s_t) · sg(δ_t) )
              + c_v · mean( (V_ψ(s_t) − sg(r_t + γ (1 − term_t) V_ψ(s_{t+1})))² )
              − c_ent · mean( H(π_φ(· | s_t)) )
    """)


def verify_hand_calculations():
    """核对 CONCEPT.md 第 4.3 节（TD 误差）与第 4.4 节（类别熵与高斯微分熵）。"""
    print("\n" + "=" * 60)
    print("第二部分：核对 CONCEPT.md 第 4.3、4.4 节手算例子")
    print("=" * 60)

    delta_good = 1.0 + 0.9 * 6.0 - 5.0
    delta_bad = 1.0 + 0.9 * 4.0 - 5.0
    print(f"  1. TD 优势例子（V(s)=5.0, r=1, γ=0.9）：")
    print(f"     通向 V(s')=6.0 的动作：δ = {delta_good:+.1f}（优于预期，提高概率）")
    print(f"     通向 V(s')=4.0 的动作：δ = {delta_bad:+.1f}（差于预期，压低概率）")
    assert abs(delta_good - 1.4) < 1e-6 and abs(delta_bad - (-0.4)) < 1e-6

    print("\n  2. 策略熵核对（自然对数 nat）：")
    for probs, ref in [([0.5, 0.5], 0.693), ([0.9, 0.1], 0.325), ([0.99, 0.01], 0.056)]:
        ent = Categorical(probs=torch.tensor(probs)).entropy().item()
        print(f"     类别策略 {str(probs):<14} → H = {ent:.3f} nat")
        assert abs(ent - ref) < 1e-3

    h_g1 = Normal(0.0, 1.0).entropy().item()
    h_g01 = Normal(0.0, 0.1).entropy().item()
    print(f"     一维高斯 σ=1.0 → H = {h_g1:.3f}；σ=0.1 → H = {h_g01:.3f}（连续微分熵可以为负！）✓")
    assert abs(h_g1 - 1.419) < 1e-3 and abs(h_g01 - (-0.884)) < 2e-3


# ==================== 第二部分：ActorCritic 模型与更新 ====================


class ActorCritic(nn.Module):
    """独立的 actor 与 critic 网络（CONCEPT.md 第 4.7 节：低维状态输入下独立网络免受梯度尺度互相干扰）。"""

    def __init__(self, obs_dim=4, n_actions=2, hidden=64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return Categorical(logits=self.actor(obs)), self.critic(obs).squeeze(-1)


def a2c_update(model, optimizer, obs, actions, rewards, next_obs, terminated,
               gamma=GAMMA, c_v=0.5, c_ent=0.01):
    """执行一次 n 步 / 一步同步 Actor-Critic 更新（CONCEPT.md 第 4.5、4.6 节）。

    obs [T, N, 4]，actions [T, N]，rewards [T, N]，next_obs [T, N, 4]，terminated [T, N]
    注意：next_obs 在环境自动 reset 的那一步保存的是 reset 之前的真实末态（final_obs），
    所以当 truncated=True、terminated=False 时，1 - terminated = 1，正好用真实末态的价值正确自举！
    """
    t_steps, n_envs = actions.shape
    dist, values = model(obs)  # values [T, N]

    with torch.no_grad():
        # 从片段末端自举递推 T 步回报目标（T=1 时就是严格的一步 TD：r_t + γ V(s_{t+1})）
        _, last_next_v = model(next_obs[-1])
        returns = torch.zeros_like(rewards)
        running = last_next_v
        for t in range(t_steps - 1, -1, -1):
            # 若第 t 步发生了回合切换（无论是 terminated 还是 truncated），该回合的末端价值都取自它自己的 next_obs[t]
            _, step_next_v = model(next_obs[t])
            # 平时相连步 running 沿时间反传；此处直接用展开回报，T=1 时 running 即 step_next_v
            if t == t_steps - 1:
                boot = step_next_v
            else:
                boot = running
            running = rewards[t] + gamma * (1.0 - terminated[t]) * boot
            returns[t] = running
        advantages = returns - values

    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()

    actor_loss = -(log_probs * advantages.detach()).mean()
    critic_loss = F.mse_loss(values, returns.detach())
    loss = actor_loss + c_v * critic_loss - c_ent * entropy.mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return dict(actor=actor_loss.item(), critic=critic_loss.item(), entropy=entropy.mean().item())


@torch.no_grad()
def evaluate_policy(actor_fn, eval_seeds):
    env = gym.make("CartPole-v1")
    rets = []
    for s in eval_seeds:
        obs, _ = env.reset(seed=int(s))
        tot = 0.0
        while True:
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            a = int(actor_fn(obs_t).argmax(dim=-1).item())
            obs, r, term, trunc, _ = env.step(a)
            tot += float(r)
            if term or trunc:
                break
        rets.append(tot)
    env.close()
    return float(np.mean(rets))


def train_a2c(seed, total_steps, n_envs=8, rollout_len=5, c_ent=0.01, lr=3e-3, eval_every=2000):
    """使用 Gymnasium SyncVectorEnv 运行同步 A2C，妥善处理自动 reset 时的 final_obs。"""
    seed_everything(seed)
    envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(n_envs)])
    obs, _ = envs.reset(seed=seed * 100)

    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    eval_seeds = [3000 + i for i in range(8)]

    steps_done = 0
    eval_log = []
    ent_log = []
    next_eval = eval_every

    while steps_done < total_steps:
        obs_buf, act_buf, rew_buf, next_obs_buf, term_buf, done_any_buf = [], [], [], [], [], []
        for _ in range(rollout_len):
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            with torch.no_grad():
                dist, _ = model(obs_t)
                actions = dist.sample()
            act_np = actions.numpy()
            next_obs, rewards, terminated, truncated, infos = envs.step(act_np)

            # Gymnasium VectorEnv 在回合结束时会自动把 next_obs 重置为新回合起点，真实末态保存在 infos["final_obs"]
            real_next = next_obs.copy()
            done_mask = terminated | truncated
            if np.any(done_mask) and "final_obs" in infos:
                for i in np.where(done_mask)[0]:
                    if infos["final_obs"][i] is not None:
                        real_next[i] = infos["final_obs"][i]

            obs_buf.append(obs_t)
            act_buf.append(actions)
            rew_buf.append(torch.from_numpy(rewards.astype(np.float32)))
            next_obs_buf.append(torch.from_numpy(real_next.astype(np.float32)))
            term_buf.append(torch.from_numpy(terminated.astype(np.float32)))
            done_any_buf.append(torch.from_numpy(done_mask.astype(np.float32)))
            obs = next_obs
            steps_done += n_envs

        # 逐环境、逐时间步精确处理中间截断边界的多步回报递推
        obs_s = torch.stack(obs_buf)          # [T, N, 4]
        act_s = torch.stack(act_buf)          # [T, N]
        rew_s = torch.stack(rew_buf)          # [T, N]
        next_s = torch.stack(next_obs_buf)    # [T, N, 4]
        term_s = torch.stack(term_buf)        # [T, N]
        done_s = torch.stack(done_any_buf)    # [T, N]

        dist_all, values_all = model(obs_s)
        with torch.no_grad():
            _, next_vals_all = model(next_s)  # [T, N]
            returns = torch.zeros_like(rew_s)
            running = next_vals_all[-1]
            for t in range(rollout_len - 1, -1, -1):
                # 如果第 t 步回合结束（terminated 或 truncated），不能跨到 t+1 的新回合，而应改用本回合自己的末态价值 next_vals_all[t]
                boot = torch.where(done_s[t] > 0.5, next_vals_all[t], running)
                running = rew_s[t] + GAMMA * (1.0 - term_s[t]) * boot
                returns[t] = running
            adv = returns - values_all

        actor_loss = -(dist_all.log_prob(act_s) * adv.detach()).mean()
        critic_loss = F.mse_loss(values_all, returns.detach())
        ent = dist_all.entropy().mean()
        loss = actor_loss + 0.5 * critic_loss - c_ent * ent
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if steps_done >= next_eval or steps_done >= total_steps:
            ev = evaluate_policy(model.actor, eval_seeds)
            eval_log.append((steps_done, ev))
            ent_log.append((steps_done, ent.item()))
            next_eval += eval_every

    envs.close()
    return eval_log, ent_log


def train_reinforce_baseline_steps(seed, total_steps, eval_every=2000):
    """按相同真实环境交互步数统计带基线的 REINFORCE，用于公平对比。"""
    seed_everything(seed)
    env = gym.make("CartPole-v1")
    model = ActorCritic()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    eval_seeds = [3000 + i for i in range(8)]

    steps_done = 0
    next_eval = eval_every
    eval_log = []
    ep_idx = 0

    while steps_done < total_steps:
        obs_l, act_l, gt_l = [], [], []
        batch_steps = 0
        while batch_steps < 400 and steps_done < total_steps:
            obs, _ = env.reset(seed=seed * 10_000 + ep_idx)
            ep_idx += 1
            ep_o, ep_a, ep_r = [], [], []
            while True:
                obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    dist, _ = model(obs_t)
                    a = int(dist.sample().item())
                obs, r, term, trunc, _ = env.step(a)
                ep_o.append(obs_t.squeeze(0))
                ep_a.append(a)
                ep_r.append(r)
                steps_done += 1
                batch_steps += 1
                if term or trunc:
                    break
            gt = np.zeros(len(ep_r), dtype=np.float32)
            g = 0.0
            for k in range(len(ep_r) - 1, -1, -1):
                g = float(ep_r[k]) + GAMMA * g
                gt[k] = g
            obs_l.append(torch.stack(ep_o))
            act_l.append(torch.tensor(ep_a, dtype=torch.long))
            gt_l.append(torch.from_numpy(gt))

        obs_b = torch.cat(obs_l, dim=0)
        act_b = torch.cat(act_l, dim=0)
        gt_b = torch.cat(gt_l, dim=0)
        dist, vals = model(obs_b)
        adv = (gt_b - vals).detach()
        loss = -(dist.log_prob(act_b) * adv).mean() + 0.5 * F.mse_loss(vals, gt_b)
        opt.zero_grad()
        loss.backward()
        opt.step()

        while steps_done >= next_eval or (steps_done >= total_steps and len(eval_log) == 0):
            eval_log.append((next_eval, evaluate_policy(model.actor, eval_seeds)))
            next_eval += eval_every
            if next_eval > total_steps:
                break

    env.close()
    return eval_log


def run_experiments(n_seeds, total_steps, eval_every):
    print("\n" + "=" * 60)
    print("第三部分：CartPole-v1 上对比 A2C vs 带基线 REINFORCE，以及熵正则的作用")
    print("=" * 60)

    t0 = time.time()
    # 一步 A2C（rollout_len=1）、5 步 A2C（c_ent=0.02 vs c_ent=0.0）与带基线 REINFORCE
    configs = [
        ("一步 A2C (c_ent=0.02)", lambda s: train_a2c(s, total_steps, rollout_len=1, c_ent=0.02, eval_every=eval_every)),
        ("5 步 A2C (c_ent=0.02)", lambda s: train_a2c(s, total_steps, rollout_len=5, c_ent=0.02, eval_every=eval_every)),
        ("5 步 A2C (无熵正则 c_ent=0)", lambda s: train_a2c(s, total_steps, rollout_len=5, c_ent=0.0, eval_every=eval_every)),
    ]
    curves, ents = {}, {}
    for name, fn in configs:
        c_list, e_list = [], []
        for s in range(n_seeds):
            ev_log, en_log = fn(s)
            xs = [k for k, _ in ev_log]
            c_list.append([v for _, v in ev_log])
            e_list.append([v for _, v in en_log])
        curves[name] = (xs, np.array(c_list))
        ents[name] = (xs, np.array(e_list))
        print(f"  {name:<28} 终点评估回报 = {curves[name][1][:, -1].mean():6.1f} ± {curves[name][1][:, -1].std():5.1f}  "
              f"终点策略熵 = {ents[name][1][:, -1].mean():.3f} nat")

    rf_list = []
    for s in range(n_seeds):
        ev_log = train_reinforce_baseline_steps(s, total_steps, eval_every=eval_every)
        rf_xs = [k for k, _ in ev_log]
        rf_list.append([v for _, v in ev_log])
    min_len = min(len(x) for x in rf_list)
    rf_arr = np.array([x[:min_len] for x in rf_list])
    rf_xs = rf_xs[:min_len]
    print(f"  {'带基线 REINFORCE':<28} 终点评估回报 = {rf_arr[:, -1].mean():6.1f} ± {rf_arr[:, -1].std():5.1f}  [总耗时 {time.time() - t0:.1f}s]")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    for (name, _), color in zip(configs, ("C1", "C0", "C3")):
        xs, arr = curves[name]
        axes[0].plot(xs, arr.mean(0), linewidth=2, color=color, label=name)
        axes[0].fill_between(xs, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0), color=color, alpha=0.14)
    axes[0].plot(rf_xs, rf_arr.mean(0), "k--", linewidth=1.8, label="带基线 REINFORCE")
    axes[0].set_xlabel("真实环境交互步数")
    axes[0].set_ylabel("贪心评估平均回报")
    axes[0].set_title(f"学习曲线对比（{n_seeds} 种子均值 ± 标准差）")
    axes[0].legend(fontsize=8)

    for name, color in [("5 步 A2C (c_ent=0.02)", "C0"), ("5 步 A2C (无熵正则 c_ent=0)", "C3"), ("一步 A2C (c_ent=0.02)", "C1")]:
        xs, earr = ents[name]
        axes[1].plot(xs, earr.mean(0), linewidth=2, color=color, label=name)
    axes[1].axhline(math.log(2.0), color="gray", linestyle=":", label="均匀分布最大熵 ln 2 ≈ 0.693")
    axes[1].set_xlabel("真实环境交互步数")
    axes[1].set_ylabel("策略熵 H(π)（nat）")
    axes[1].set_title("熵正则推迟策略过早变确定")
    axes[1].legend(fontsize=8)
    save_figure(fig, OUT / "03-a2c-and-entropy.png")


# ==================== 第四部分：练习与思考 ====================


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：如果把 truncated 错当成 terminated（即 500 步上限到来时把未来价值强行清零），critic 会学出什么怪现象？

练习 1 参考答案：
    第 499 步的状态跟第 10 步完全一样（杆子竖直稳在中间），但第 10 步的回报目标是 ~100，
    第 499 步如果清零目标就突然变成 1！观测里并没有当前步数 t，critic 面对同一个状态收到两个矛盾标签，
    价值预测误差降不下去，连带把那一帧的优势 δ_499 算成大约 −99，无辜惩罚了一个完美的平衡动作。

思考题 1：一步 A2C 方差最小，带基线 REINFORCE 无偏但方差大，5 步 A2C 落在了中间——能不能把所有步数的优势按指数权重平滑地混在一起？

思考题 1 参考答案：
    能！把 1 步、2 步、3 步……优势按 (1−λ) λ^{n-1} 加权求和，化简出来恰好是 TD 误差 δ_t 的指数移动平均。
    这就是下一节（04-gae-and-ppo.py）的广义优势估计（GAE），也是Phase 4 λ-return 减掉 V(s_t) 后的孪生形式。
    """)


def main():
    seed_everything(SEED)
    quick = is_quick()
    if quick:
        print("（冒烟模式：交互步数缩小，仅验证流程跑通）\n")
    n_seeds, total_steps, eval_every = (2, 1600, 800) if quick else (4, 24_000, 2000)

    introduction()
    verify_hand_calculations()
    run_experiments(n_seeds=n_seeds, total_steps=total_steps, eval_every=eval_every)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：04-gae-and-ppo.py（广义优势估计 GAE 与近端策略优化 PPO）")
    print("=" * 60)


if __name__ == "__main__":
    main()
