"""
沿可微模型反向传播：从 SVG 到 Dreamer 的桥梁
============================================

学习目标：
    1. 用 PyTorch 自动求导严格复现 CONCEPT.md 第 11.3 节的一维线性模型 2 步手算梯度（J = -0.3125, dJ/dφ = -1.5）
    2. 在 Pendulum-v1 上实现可微动力学模型 f_θ(s, a) 与解析可微奖励 r(s, a)
    3. 实现 imagine_rollout() 与 actor_loss_through_model()，清晰落地三处停止梯度（sg）边界：
       梯度流过冻结的模型和 critic 只更新 actor；critic 的输入和 λ-return 目标均 detach
    4. 两个核心诊断实验：
       (a) 在同一批想象轨迹上比较 pathwise（沿模型反传）与 score-function（想象轨迹上的 REINFORCE）的梯度方差比
       (b) 扫描想象视野 H ∈ {1, 3, 5, 10, 15, 25, 40}，实测梯度范数随 H 爆炸的现象
    5. 在 Pendulum-v1 上跑通完整的"采集 → 拟合模型 → 沿模型反传更新 actor + λ-return 更新 critic"闭环（SVG(H)）

核心概念：
    - 沿时间反向传播（BPTT through model）：把策略 π_φ、模型 f_θ、奖励 r_θ 和末端价值 V_ψ 接成一条可微计算图
    - 雅可比连乘与梯度爆炸：误差按 L^H 增长的系统，反向传播梯度也按闭环雅可比连乘增长
    - 短视野 + 末端价值 + λ-return：SVG(H) 到 Dreamer V1（Phase 6）的统一架构

前置知识：
    - 05-gradient-estimators.py（pathwise vs score-function）
    - 04-gae-and-ppo.py（λ-return）
    - 08-mbpo.py（黑盒模型生成数据）

对应概念文档：
    - CONCEPT.md 第 11 节「沿模型反向传播：从 SVG 到 Dreamer」

预计运行时间：约 1.5 分钟（CPU）
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
OUT = output_dir("phase-05")
SEED = 0
GAMMA = 0.99
LAM = 0.95


# ==================== 第一部分：导读与手算核对 ====================


def introduction():
    print("=" * 60)
    print("第一部分：把世界模型当白盒——让梯度穿过模型直接流回策略")
    print("=" * 60)
    print("""
上一节的 MBPO 把动力学模型当黑盒仿真器，只拿它生成数据喂给 SAC。
但神经网络世界模型天生对动作和状态可微！把 π_φ、f_θ、r_θ 和末端价值 V_ψ 串成一条可微计算图：

    s_0 ──▶ π_φ ──▶ a_0 ──▶ f_θ ──▶ ŝ_1 ──▶ π_φ ──▶ a_1 ──▶ ··· ──▶ ŝ_H ──▶ V_ψ
                     │       │                       │                        │
                     └──▶ r̂_0                        └──▶ r̂_1                终端价值

梯度从所有奖励 r̂_k 和末端价值 V_ψ 出发，穿过冻结的模型参数 θ 和价值参数 ψ，只更新策略参数 φ。
把这里的真实状态 s 换成 Phase 3 的 RSSM 隐状态 (h, z)，就是下一章（Phase 6）的 Dreamer！
    """)


def verify_hand_calculation():
    """复现 CONCEPT.md 第 11.3 节的手算例子：s' = s + a, a = φ s, r = -s'^2, s_0 = 1, φ = -0.5, H = 2。"""
    print("=" * 60)
    print("第二部分：核对 CONCEPT.md 第 11.3 节的 2 步手算梯度")
    print("=" * 60)
    phi = torch.tensor(-0.5, requires_grad=True)
    s = torch.tensor(1.0)
    rewards = []
    for _ in range(2):
        a = phi * s
        s = s + a
        rewards.append(-(s ** 2))
    J = sum(rewards)
    J.backward()
    print(f"  前向回报 J(φ=-0.5) = {J.item():.4f}（手算 −0.3125）")
    print(f"  自动求导 dJ/dφ     = {phi.grad.item():.4f}（手算 −1.5000）")
    assert abs(J.item() - (-0.3125)) < 1e-6 and abs(phi.grad.item() - (-1.5)) < 1e-6
    print("  ✓ 与 CONCEPT.md 第 11.3 节完全一致。")


# ==================== 第二部分：可微模型、策略、价值与 λ-return ====================


def pendulum_reward_torch(obs, act_norm):
    """Pendulum-v1 的解析可微奖励函数：obs = [cos θ, sin θ, θ̇]，act_norm ∈ [-1, 1]。"""
    cos_th = obs[..., 0].clamp(-1.0, 1.0)
    sin_th = obs[..., 1].clamp(-1.0, 1.0)
    thdot = obs[..., 2]
    theta = torch.atan2(sin_th, cos_th)
    u = act_norm.squeeze(-1) * 2.0
    return -(theta.pow(2) + 0.1 * thdot.pow(2) + 0.001 * u.pow(2))


class DiffDynamicsModel(nn.Module):
    """确定性可微残差动力学模型：输入 [cos θ, sin θ, θ̇, a_norm]，输出下一步状态并投影回单位圆。"""

    def __init__(self, obs_dim=3, act_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, obs_dim),
        )
        self.register_buffer("in_mean", torch.zeros(obs_dim + act_dim))
        self.register_buffer("in_std", torch.ones(obs_dim + act_dim))

    def forward(self, obs, act_norm):
        x = (torch.cat([obs, act_norm], dim=-1) - self.in_mean) / self.in_std
        delta = self.net(x)
        nxt = obs + delta
        norm = torch.sqrt(nxt[..., 0:1].pow(2) + nxt[..., 1:2].pow(2) + 1e-6)
        cos_sin = nxt[..., 0:2] / norm
        thdot = torch.tanh(nxt[..., 2:3] / 8.0) * 8.0  # 光滑有界，比硬 clamp 更适合反传
        return torch.cat([cos_sin, thdot], dim=-1)

    def fit(self, obs_np, act_np, next_obs_np, epochs=40, batch_size=128, lr=2e-3):
        obs_t = torch.from_numpy(obs_np)
        act_t = torch.from_numpy(act_np)
        nxt_t = torch.from_numpy(next_obs_np)
        x_raw = torch.cat([obs_t, act_t], dim=-1)
        self.in_mean.copy_(x_raw.mean(dim=0))
        self.in_std.copy_(x_raw.std(dim=0).clamp_min(1e-3))

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n = len(obs_t)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch_size):
                idx = perm[s : s + batch_size]
                pred = self(obs_t[idx], act_t[idx])
                loss = F.mse_loss(pred, nxt_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()


class StochasticTanhActor(nn.Module):
    """重参数化高斯策略，支持同时返回 rsample 动作与 log_prob（供 pathwise vs score-function 对比）。"""

    def __init__(self, obs_dim=3, act_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs, deterministic=False):
        h = self.net(obs)
        mean = self.mean_head(h)
        if deterministic:
            return torch.tanh(mean), None, None
        log_std = self.log_std_head(h).clamp(-4.0, 1.0)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        u = mean + std * eps
        act = torch.tanh(u)
        base_lp = torch.distributions.Normal(mean, std).log_prob(u)
        corr = 2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))
        log_prob = (base_lp - corr).sum(dim=-1)
        entropy = (0.5 + 0.5 * np.log(2 * np.pi) + log_std).sum(dim=-1)
        return act, log_prob, entropy


class StateValueCritic(nn.Module):
    def __init__(self, obs_dim=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def compute_lambda_returns(rewards, values, gamma=GAMMA, lam=LAM):
    """在想象轨迹上从后往前计算 λ-return（CONCEPT.md 第 5 节与第 11.6 节）。

    rewards: [B, H]，values: [B, H + 1]（含末端 V(ŝ_H)）。返回 [B, H]。
    """
    B, H = rewards.shape
    targets = torch.zeros(B, H, device=rewards.device, dtype=rewards.dtype)
    next_ret = values[:, -1]
    for k in reversed(range(H)):
        next_ret = rewards[:, k] + gamma * ((1.0 - lam) * values[:, k + 1] + lam * next_ret)
        targets[:, k] = next_ret
    return targets


def imagine_rollout(model, actor, start_obs, horizon):
    """CONCEPT.md 第 11.6 节第 2 步：沿可微模型展开 H 步，保留完整计算图。"""
    s = start_obs.detach()
    obs_list = [s]
    act_list, rew_list, logp_list, ent_list = [], [], [], []
    for _ in range(horizon):
        a, logp, ent = actor(s, deterministic=False)
        r = pendulum_reward_torch(s, a)
        s = model(s, a)
        obs_list.append(s)
        act_list.append(a)
        rew_list.append(r)
        logp_list.append(logp)
        ent_list.append(ent)
    return (
        torch.stack(obs_list, dim=1),   # [B, H+1, S]
        torch.stack(act_list, dim=1),   # [B, H, A]
        torch.stack(rew_list, dim=1),   # [B, H]
        torch.stack(logp_list, dim=1),  # [B, H]
        torch.stack(ent_list, dim=1),   # [B, H]
    )


def actor_loss_through_model(model, actor, critic, start_obs, horizon, ent_coef=1e-3):
    """CONCEPT.md 第 11.6 节第 3 步：冻结 model 与 critic 参数（梯度只流过、不写入其 .grad）。"""
    for p in model.parameters():
        p.requires_grad_(False)
    for p in critic.parameters():
        p.requires_grad_(False)

    imag_obs, _, imag_rew, _, imag_ent = imagine_rollout(model, actor, start_obs, horizon)
    imag_vals = critic(imag_obs)  # [B, H+1]，梯度穿过 critic 流回 imag_obs → actor
    lam_returns = compute_lambda_returns(imag_rew, imag_vals)
    # 与 Dreamer 相同：最大化每个想象步的 λ-return 均值 + 少量熵正则
    actor_loss = -(lam_returns.mean() + ent_coef * imag_ent.mean())

    for p in model.parameters():
        p.requires_grad_(True)
    for p in critic.parameters():
        p.requires_grad_(True)
    return actor_loss, imag_obs.detach(), imag_rew.detach()


# ==================== 第三部分：两个核心梯度诊断实验 ====================


def run_gradient_diagnostics(model, actor, critic, real_states):
    print("\n" + "=" * 60)
    print("第三部分：梯度诊断 — Pathwise vs Score-Function 方差比 & 梯度范数随 H 的变化")
    print("=" * 60)

    start_batch = torch.from_numpy(real_states[:64])

    # 诊断 1：在 H = 5 与 H = 15 上，分别采样 40 次单批估计，比较 pathwise 与 score-function 梯度的总方差
    n_trials = 16 if is_quick() else 40
    var_ratios = {}
    for H in [5, 15]:
        pw_grads, sf_grads = [], []
        for _ in range(n_trials):
            # Pathwise（沿模型反向传播）
            actor.zero_grad()
            imag_obs, _, imag_rew, _, _ = imagine_rollout(model, actor, start_batch, H)
            discounts = GAMMA ** torch.arange(H, dtype=torch.float32)
            J_pw = (imag_rew * discounts).sum(dim=1).mean() + (GAMMA ** H) * critic(imag_obs[:, -1]).mean()
            (-J_pw).backward()
            pw_grads.append(torch.cat([p.grad.detach().flatten() for p in actor.parameters()]))

            # Score-Function（想象轨迹上的 REINFORCE，带 V(ŝ_k) 基线，严格截断经模型状态的梯度）
            actor.zero_grad()
            s_cur = start_batch.detach()
            lps, rs, obs_sf = [], [], [s_cur]
            for _ in range(H):
                a_k, lp_k, _ = actor(s_cur.detach(), deterministic=False)
                lps.append(lp_k)
                with torch.no_grad():
                    rs.append(pendulum_reward_torch(s_cur, a_k))
                    s_cur = model(s_cur, a_k)
                    obs_sf.append(s_cur)
            with torch.no_grad():
                r_mat = torch.stack(rs, dim=1)
                v_mat = critic(torch.stack(obs_sf, dim=1))
                ret_mat = compute_lambda_returns(r_mat, v_mat)
                adv_mat = ret_mat - v_mat[:, :-1]
            loss_sf = -(torch.stack(lps, dim=1) * adv_mat).mean()
            loss_sf.backward()
            sf_grads.append(torch.cat([p.grad.detach().flatten() for p in actor.parameters()]))

        pw_mat = torch.stack(pw_grads, dim=0)
        sf_mat = torch.stack(sf_grads, dim=0)
        pw_var = pw_mat.var(dim=0).sum().item()
        sf_var = sf_mat.var(dim=0).sum().item()
        ratio = sf_var / max(pw_var, 1e-12)
        var_ratios[H] = (pw_var, sf_var, ratio)
        print(f"  1. 视野 H = {H:<2}：Pathwise 梯度总方差 = {pw_var:8.4f} ｜ "
              f"Score-Function 总方差 = {sf_var:8.4f} ｜ 方差比 (SF / PW) = {ratio:6.1f}×")

    # 诊断 2：扫描视野 H，观测沿模型反向传播的梯度范数变化（用随机初始化模型 + 训练后模型两条曲线）
    horizons = [1, 3, 5, 10, 15, 25, 40]
    grad_norms = []
    for H in horizons:
        actor.zero_grad()
        imag_obs, _, imag_rew, _, _ = imagine_rollout(model, actor, start_batch, H)
        discounts = GAMMA ** torch.arange(H, dtype=torch.float32)
        J = (imag_rew * discounts).sum(dim=1).mean() + (GAMMA ** H) * critic(imag_obs[:, -1]).mean()
        (-J).backward()
        gnorm = torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in actor.parameters())).item()
        grad_norms.append(gnorm)
    print("\n  2. 沿模型反传的 Actor 梯度范数随视野 H 变化：")
    print("     " + "  ".join(f"H={h}:{g:.1f}" for h, g in zip(horizons, grad_norms)))

    # 画诊断图
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    hs_bar = list(var_ratios.keys())
    x = np.arange(len(hs_bar))
    axes[0].bar(x - 0.18, [var_ratios[h][0] for h in hs_bar], width=0.34, label="Pathwise（沿模型反传）")
    axes[0].bar(x + 0.18, [var_ratios[h][1] for h in hs_bar], width=0.34, label="Score-Function（REINFORCE）")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"H = {h}\n(比值 {var_ratios[h][2]:.1f}×)" for h in hs_bar])
    axes[0].set_ylabel("梯度总方差（对数尺度）")
    axes[0].set_title("短视野下 Pathwise 方差显著低于 Score-Function")
    axes[0].legend(fontsize=8.5)

    axes[1].plot(horizons, grad_norms, "o-", color="C3", linewidth=2.2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("想象视野 H")
    axes[1].set_ylabel("Actor 梯度 L2 范数（对数尺度）")
    axes[1].set_title("视野过长导致雅可比连乘放大梯度")
    save_figure(fig, OUT / "09-grad-diagnostics.png")
    return var_ratios, horizons, grad_norms


# ==================== 第四部分：完整 SVG(H) 闭环训练 ====================


@torch.no_grad()
def evaluate_actor(actor, n_episodes=5, seed_base=900):
    env = gym.make("Pendulum-v1")
    rets = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        tot = 0.0
        for _ in range(200):
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
            act, _, _ = actor(obs_t, deterministic=True)
            obs, r, term, trunc, _ = env.step(act.squeeze(0).numpy() * 2.0)
            tot += float(r)
            if term or trunc:
                break
        rets.append(tot)
    env.close()
    return float(np.mean(rets))


def train_svg_pendulum(n_rounds=16, init_episodes=5, horizon=12, ac_updates=80):
    print("\n" + "=" * 60)
    print(f"第四部分：在 Pendulum-v1 上跑通 SVG(H={horizon}) 沿模型反向传播闭环")
    print("=" * 60)
    t0 = time.time()
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(SEED)

    obs_buf, act_buf, nxt_buf = [], [], []

    def collect(n_eps, actor_model=None, explore_std=0.25):
        for _ in range(n_eps):
            o, _ = env.reset()
            for _ in range(200):
                if actor_model is None:
                    a = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
                else:
                    with torch.no_grad():
                        a_t, _, _ = actor_model(torch.from_numpy(o.astype(np.float32)).unsqueeze(0))
                        a = np.clip(a_t.squeeze(0).numpy() + rng.normal(0.0, explore_std, size=(1,)), -1.0, 1.0).astype(np.float32)
                no, _, term, trunc, _ = env.step(a * 2.0)
                obs_buf.append(o.astype(np.float32))
                act_buf.append(a)
                nxt_buf.append(no.astype(np.float32))
                o = no
                if term or trunc:
                    break

    collect(init_episodes, actor_model=None)

    model = DiffDynamicsModel()
    actor = StochasticTanhActor()
    critic = StateValueCritic()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=2e-3)

    eval_steps, eval_rets = [], []

    for r_idx in range(1, n_rounds + 1):
        obs_arr = np.asarray(obs_buf, dtype=np.float32)
        act_arr = np.asarray(act_buf, dtype=np.float32)
        nxt_arr = np.asarray(nxt_buf, dtype=np.float32)

        model.fit(obs_arr, act_arr, nxt_arr, epochs=25 if r_idx == 1 else 12, batch_size=128)

        # 在第 1 轮模型刚训好后跑梯度诊断，确保诊断对象是真实的非平凡动力学模型
        if r_idx == 1:
            run_gradient_diagnostics(model, actor, critic, obs_arr)

        # 在模型想象中交替更新 actor（沿模型反传）与 critic（λ-return 回归）
        for _ in range(ac_updates):
            idx = np.random.randint(0, len(obs_arr), size=64)
            start_obs = torch.from_numpy(obs_arr[idx])

            # 1. Actor 更新：梯度流过冻结的 model 和 critic
            a_loss, det_obs, det_rew = actor_loss_through_model(model, actor, critic, start_obs, horizon)
            actor_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            actor_opt.step()

            # 2. Critic 更新：输入与目标均 detach（CONCEPT.md 第 11.6 节第 4 步）
            with torch.no_grad():
                v_all = critic(det_obs)
                v_targets = compute_lambda_returns(det_rew, v_all)
            v_pred = critic(det_obs[:, :-1])
            c_loss = F.mse_loss(v_pred, v_targets)
            critic_opt.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
            critic_opt.step()

        # 采集 1 个新回合补充真实数据
        collect(1, actor_model=actor, explore_std=0.2)
        steps_so_far = len(obs_buf)
        if r_idx == 1 or r_idx % 2 == 0 or r_idx == n_rounds:
            ev_r = evaluate_actor(actor, n_episodes=5)
            eval_steps.append(steps_so_far)
            eval_rets.append(ev_r)
            print(f"  第 {r_idx:>2}/{n_rounds} 轮（累计 {steps_so_far:>4} 真实步）：5 回合评估回报 = {ev_r:7.1f}")

    env.close()
    dt = time.time() - t0
    print(f"\n  SVG(H={horizon}) 闭环训练完成，最终评估回报 {eval_rets[-1]:.1f}（总用时 {dt:.1f}s）。")

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(eval_steps, eval_rets, "o-", color="C0", linewidth=2.4, label=f"SVG(H={horizon}) 沿模型反传")
    ax.axhline(-200.0, color="seagreen", linestyle="--", label="优秀参考线（−200）")
    ax.axhline(-1150.0, color="tomato", linestyle=":", label="随机策略参考（≈−1150）")
    ax.set_xlabel("累计真实环境交互步数")
    ax.set_ylabel("5 回合平均评估回报")
    ax.set_title("Pendulum-v1：沿可微世界模型反向传播学习控制策略")
    ax.legend(fontsize=9)
    save_figure(fig, OUT / "09-svg-learning-curve.png")


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把 actor_loss_through_model() 里的冻结模型代码删掉，让优化器同时更新 actor 和 model，会发生什么？

练习 1 参考答案：
    模型不再忠实拟合真实物理，而是学会"无论 actor 输出什么动作，我都立刻把摆瞬移到正上方 θ=0"——
    想象回报瞬间拉满到 0，但策略放到真实环境里完全无效（退化回随机水平）。

思考题 1：从本节到 Phase 6 的 Dreamer V1，还差哪两块拼图？

思考题 1 参考答案：
    1. 把本节的状态空间确定性 MLP f_θ(s, a) 换成 Phase 3 的像素级 RSSM (h_t, z_t)，让视觉输入也能在隐空间想象；
    2. 把解析奖励公式换成世界模型一起学出来的奖励头 r̂(h_t, z_t)（以及 Phase 6 Dreamer V2/V3 里的离散分类潜变量、
       straight-through 梯度、KL balancing、symlog 与两热分布等稳定化设计）。
    """)


def main():
    seed_everything(SEED)
    quick = is_quick()
    if quick:
        print("（冒烟模式：轮数与更新步数缩小，验证梯度诊断与闭环训练跑通）\n")
    introduction()
    verify_hand_calculation()
    train_svg_pendulum(
        n_rounds=4 if quick else 16,
        init_episodes=2 if quick else 5,
        horizon=10 if quick else 12,
        ac_updates=20 if quick else 80,
    )
    exercises()
    print("\n" + "=" * 60)
    print("🎉 Phase 5 全部 9 节讲义完成！下一步：运行实战项目 projects/phase-05-control-benchmark/，然后进入 Phase 6 Dreamer！")
    print("=" * 60)


if __name__ == "__main__":
    main()
