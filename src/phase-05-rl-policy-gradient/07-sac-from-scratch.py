"""
SAC：最大熵强化学习从零实现
===========================

学习目标：
    1. 核对 CONCEPT.md 第 9.3、9.5、9.6 节的手算例子：软价值 log-sum-exp、tanh 雅可比对数概率修正、自动调温梯度方向
    2. 实现数值稳定的 TanhGaussianActor：用 2(ln 2 − u − softplus(−2u)) 避免 a → ±1 时 log(1 − a²) 溢出成 -inf
    3. 实现完整 sac_update()：软 Bellman 双 Q 回归、重参数化 actor 更新、拉格朗日对偶自动调节温度 α
    4. 在 Pendulum-v1 上对比自适应 α 与固定 α ∈ {0.01, 0.2, 2.0}，观察 α 与策略熵向目标熵 H̄ = −dim(A) 收敛的过程
    5. 在 MuJoCo 连续控制基准 HalfCheetah-v5（17 维状态、6 维动作）上验证同一套 SAC 超参数开箱即用

核心概念：
    - 最大熵目标：J(π) = E[ Σ γ^t (r_t + α H(π(·|s_t))) ]，把探索写进价值定义本身
    - tanh 高斯换元修正：log π(a|s) = log N(u; μ, σ²) − Σ_i log(1 − tanh²(u_i))
    - 自动调温：最小化 E[ −log α · sg(log π(a|s) + H̄) ]，目标熵默认取 H̄ = −dim(A)

前置知识：
    - 05-gradient-estimators.py（重参数化 pathwise 估计器）
    - 06-ddpg-td3.py（离策略回放与双 Q 取小）

对应概念文档：
    - CONCEPT.md 第 9 节「SAC：最大熵强化学习」

预计运行时间：约 1.5 分钟（CPU）
"""

import copy
import importlib.util
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
from torch.distributions import Normal

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec06 = importlib.util.spec_from_file_location("p5_06", os.path.join(_DIR, "06-ddpg-td3.py"))
_mod06 = importlib.util.module_from_spec(_spec06)
_spec06.loader.exec_module(_mod06)

ContinuousReplay = _mod06.ContinuousReplay
QCritic = _mod06.QCritic
soft_update = _mod06.soft_update

plt = setup_matplotlib()
OUT = output_dir("phase-05")
SEED = 0
GAMMA = 0.99
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


# ==================== 第一部分：导读与手算单元测试 ====================


def introduction():
    print("=" * 60)
    print("第一部分：SAC——把\"保持适度随机\"写进 Bellman 方程")
    print("=" * 60)
    print("""
SAC 把 TD3 的确定性策略换成了带重参数化的 tanh 高斯随机策略，并在每一步奖励里加上熵奖金 α H(π)：

    软 Bellman 目标： y = r + γ (1 − d) [ min(Q_ψ̄1, Q_ψ̄2)(s', a') − α log π_φ(a' | s') ]， a' ~ π_φ
    Actor 损失：      L_π(φ) = E_{s ~ D, ε ~ N(0,I)} [ α log π_φ(ã | s) − min(Q_ψ1, Q_ψ2)(s, ã) ]
    自动调温损失：    L(α)   = E [ −log α · sg(log π_φ(ã | s) + H̄) ]，     H̄ = −dim(A)

注意：动作范围是 [-M, M]（例如 Pendulum-v1 是 [-2, 2]）时，为避免尺度换算混淆，
最干净的工程约定是 actor 与 critic 内部一律使用 [-1, 1] 的归一化动作，只在送进 env.step 时乘以 M！
    """)


def tanh_gaussian_log_prob(mean, std, u):
    """给定高斯参数 (mean, std) 与 tanh 前的样本 u [B, A]，计算 a = tanh(u) 的精确对数概率密度 [B, 1]。

    使用恒等式 log(1 − tanh² u) = 2 (ln 2 − u − softplus(−2u))（CONCEPT.md 第 9.5 节），
    在大 |u| 处绝不会因为 tanh(u) 四舍五入为 ±1 而产生 -inf / NaN。
    """
    base_logp = Normal(mean, std).log_prob(u)
    squash_corr = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
    return (base_logp - squash_corr).sum(dim=-1, keepdim=True)


def verify_hand_calculations():
    """严格核对 CONCEPT.md 第 9.3、9.5、9.6 节的手算例子。"""
    print("\n" + "=" * 60)
    print("第二部分：核对 CONCEPT.md 第 9.3、9.5、9.6 节手算单元测试")
    print("=" * 60)

    # 1. 第 9.3 节软价值例子：Q = (1, 2)
    q = torch.tensor([1.0, 2.0])
    for alpha, exp_v in [(1.0, 2.313), (0.1, 2.000), (10.0, 8.444)]:
        pi_star = torch.softmax(q / alpha, dim=0)
        v_soft = alpha * torch.logsumexp(q / alpha, dim=0).item()
        print(f"  软价值核对 α={alpha:<4}：π* = ({pi_star[0]:.4f}, {pi_star[1]:.4f})，V_soft = {v_soft:.3f}")
        assert abs(v_soft - exp_v) < 1e-2

    # 2. 第 9.5 节 tanh 高斯对数概率修正：μ=0, σ=1, u=0.5 → a=0.4621, log π(a) = -0.804
    lp_05 = tanh_gaussian_log_prob(torch.tensor([[0.0]]), torch.tensor([[1.0]]), torch.tensor([[0.5]])).item()
    u_extreme = torch.tensor([[10.0]])
    naive_corr = torch.log(1.0 - torch.tanh(u_extreme).pow(2)).item()  # float32 下 tanh(10)==1 → log(0) = -inf
    stable_lp = tanh_gaussian_log_prob(torch.tensor([[0.0]]), torch.tensor([[1.0]]), u_extreme).item()
    print(f"\n  tanh 修正核对（u = 0.5）：a = {math.tanh(0.5):.4f}，log π(a) = {lp_05:.3f}（手算 −0.804）✓")
    print(f"  极端值 u = 10.0 稳定性：直接算 log(1 − tanh² u) 得到 {naive_corr}，恒等式算出有限值 log π = {stable_lp:.2f} ✓")
    assert abs(lp_05 - (-0.804)) < 2e-3 and math.isfinite(stable_lp)


# ==================== 第二部分：TanhGaussianActor 与 SAC 更新 ====================


class TanhGaussianActor(nn.Module):
    """CONCEPT.md 第 9.8 节：输出 [-1, 1] 内的重参数化动作与对动作维度求和后的 log_prob [B, 1]。"""

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs, deterministic=False):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        if deterministic:
            return torch.tanh(mean), None
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)  # rsample：保留路径梯度
        action = torch.tanh(u)
        log_prob = tanh_gaussian_log_prob(mean, std, u)
        return action, log_prob


class SACAgent:
    def __init__(self, obs_dim, act_dim, hidden=128, lr=1e-3, kappa=0.005, fixed_alpha=None, device=torch.device("cpu")):
        self.act_dim = act_dim
        self.kappa = kappa
        self.device = device
        self.target_entropy = -float(act_dim)
        self.fixed_alpha = fixed_alpha

        self.actor = TanhGaussianActor(obs_dim, act_dim, hidden).to(device)
        self.q1 = QCritic(obs_dim, act_dim, hidden).to(device)
        self.q2 = QCritic(obs_dim, act_dim, hidden).to(device)
        self.q1_targ = copy.deepcopy(self.q1)
        self.q2_targ = copy.deepcopy(self.q2)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        init_alpha = 0.2 if fixed_alpha is None else float(fixed_alpha)
        self.log_alpha = torch.tensor(math.log(init_alpha), dtype=torch.float32, device=device, requires_grad=(fixed_alpha is None))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr) if fixed_alpha is None else None

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def select_action(self, obs_np, deterministic=False):
        obs_t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).unsqueeze(0).to(self.device)
        act, _ = self.actor(obs_t, deterministic=deterministic)
        return act.squeeze(0).cpu().numpy()

    def sac_update(self, batch):
        """CONCEPT.md 第 9.7 节第 3–6 步。batch 内动作均已归一化到 [-1, 1]。"""
        obs, act, rew, next_obs, term = [x.to(self.device) for x in batch]
        alpha_val = self.log_alpha.detach().exp()

        # 3. 更新两个 critic
        with torch.no_grad():
            next_act, next_lp = self.actor(next_obs)
            min_q_next = torch.minimum(self.q1_targ(next_obs, next_act), self.q2_targ(next_obs, next_act))
            target_y = rew + GAMMA * (1.0 - term) * (min_q_next - alpha_val * next_lp)

        q_loss = F.mse_loss(self.q1(obs, act), target_y) + F.mse_loss(self.q2(obs, act), target_y)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # 4. 重参数化更新 actor（只更新 actor 参数！）
        new_act, new_lp = self.actor(obs)
        min_q_pi = torch.minimum(self.q1(obs, new_act), self.q2(obs, new_act))
        actor_loss = (alpha_val * new_lp - min_q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 5. 自动调节温度 α
        if self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (new_lp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # 6. 软更新目标 critic
        soft_update(self.q1_targ, self.q1, self.kappa)
        soft_update(self.q2_targ, self.q2, self.kappa)
        return dict(alpha=self.alpha, entropy=(-new_lp.detach()).mean().item())


# ==================== 第三部分：Pendulum 调温对比与 HalfCheetah 实战 ====================


@torch.no_grad()
def evaluate_sac(env_id, agent, eval_seeds, act_scale):
    env = gym.make(env_id)
    rets = []
    for s in eval_seeds:
        obs, _ = env.reset(seed=int(s))
        tot = 0.0
        while True:
            a = agent.select_action(obs, deterministic=True) * act_scale
            obs, r, term, trunc, _ = env.step(a)
            tot += float(r)
            if term or trunc:
                break
        rets.append(tot)
    env.close()
    return float(np.mean(rets))


def train_sac_env(env_id, seed, total_steps, eval_every, fixed_alpha=None, warmup=500, batch_size=128, hidden=128, lr=1e-3):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = env.action_space.high.astype(np.float32)

    agent = SACAgent(obs_dim, act_dim, hidden=hidden, lr=lr, fixed_alpha=fixed_alpha)
    replay = ContinuousReplay(obs_dim=obs_dim, act_dim=act_dim, capacity=max(50_000, total_steps + 1000))
    eval_seeds = [6000 + i for i in range(4)]

    obs, _ = env.reset(seed=seed)
    ep_idx = 0
    eval_log, alpha_log, ent_log = [], [], []
    last_info = dict(alpha=agent.alpha, entropy=0.0)

    for step in range(1, total_steps + 1):
        if step <= warmup:
            norm_a = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            norm_a = agent.select_action(obs, deterministic=False)

        next_obs, rew, term, trunc, _ = env.step(norm_a * act_scale)
        replay.add(obs, norm_a, rew, next_obs, term)
        if term or trunc:
            ep_idx += 1
            obs, _ = env.reset(seed=seed * 1000 + ep_idx)
        else:
            obs = next_obs

        if replay.size >= batch_size and step > warmup:
            last_info = agent.sac_update(replay.sample(batch_size, rng))

        if step % eval_every == 0:
            ev = evaluate_sac(env_id, agent, eval_seeds, act_scale)
            eval_log.append((step, ev))
            alpha_log.append((step, last_info["alpha"]))
            ent_log.append((step, last_info["entropy"]))

    env.close()
    return eval_log, alpha_log, ent_log


def run_experiments(quick):
    print("\n" + "=" * 60)
    print("第三部分：Pendulum-v1 上自动调温 vs 固定温度 α，以及 HalfCheetah-v5 连续控制")
    print("=" * 60)

    n_seeds = 1 if quick else 3
    pend_steps, pend_eval = (1000, 500) if quick else (8000, 1000)
    cheetah_steps, cheetah_eval = (1000, 500) if quick else (12_000, 2000)

    t0 = time.time()
    alpha_configs = [
        ("自动调温 α (H̄ = -1)", None),
        ("固定 α = 0.02（偏小）", 0.02),
        ("固定 α = 0.20（适中）", 0.20),
        ("固定 α = 3.00（过大：只顾随机）", 3.00),
    ]
    pend_curves = {}
    auto_alpha_trace, auto_ent_trace = None, None
    for label, fa in alpha_configs:
        runs = []
        for s in range(n_seeds):
            ev_l, al_l, en_l = train_sac_env("Pendulum-v1", s, pend_steps, pend_eval, fixed_alpha=fa)
            xs = [k for k, _ in ev_l]
            runs.append([v for _, v in ev_l])
            if fa is None and s == 0:
                auto_alpha_trace = ([k for k, _ in al_l], [v for _, v in al_l])
                auto_ent_trace = ([k for k, _ in en_l], [v for _, v in en_l])
        arr = np.array(runs)
        pend_curves[label] = (xs, arr)
        print(f"  [Pendulum] {label:<28} 终点回报 = {arr[:, -1].mean():6.1f} ± {arr[:, -1].std():5.1f}")

    # MuJoCo HalfCheetah-v5（17 维观测、6 维连续力矩，标准机器人跑动基准）
    t_c = time.time()
    hc_runs = []
    for s in range(min(2, n_seeds)):
        ev_l, _, _ = train_sac_env("HalfCheetah-v5", s, cheetah_steps, cheetah_eval,
                                   fixed_alpha=None, warmup=400 if quick else 1000, hidden=256, lr=3e-4)
        hc_xs = [k for k, _ in ev_l]
        hc_runs.append([v for _, v in ev_l])
    hc_arr = np.array(hc_runs)
    print(f"\n  [HalfCheetah-v5] 自动调温 SAC（{cheetah_steps} 步，H̄ = -6）："
          f"第 1 次评估 {hc_arr[:, 0].mean():.1f} → 终点 {hc_arr[:, -1].mean():.1f} ± {hc_arr[:, -1].std():.1f}  [{time.time() - t_c:.1f}s]")
    print(f"  全部实验完成，共耗时 {time.time() - t0:.1f}s。")

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.3))
    for (label, _), col in zip(alpha_configs, ("C0", "C1", "C2", "C3")):
        xs, arr = pend_curves[label]
        axes[0].plot(xs, arr.mean(0), "o-", markersize=3.5, linewidth=2, color=col, label=label)
    axes[0].set_xlabel("Pendulum-v1 环境步数")
    axes[0].set_ylabel("评估回报")
    axes[0].set_title("不同温度设置下的学习曲线")
    axes[0].legend(fontsize=8)

    ax_a = axes[1]
    ax_a.plot(auto_alpha_trace[0], auto_alpha_trace[1], "o-", color="C0", linewidth=2, label="温度 α")
    ax_a.set_xlabel("Pendulum-v1 环境步数")
    ax_a.set_ylabel("温度 α", color="C0")
    ax_e = ax_a.twinx()
    ax_e.plot(auto_ent_trace[0], auto_ent_trace[1], "s--", color="C2", linewidth=1.8, label="策略微分熵 H")
    ax_e.axhline(-1.0, color="C2", linestyle=":", alpha=0.7, label="目标熵 H̄ = -1.0")
    ax_e.set_ylabel("策略熵 H(π)", color="C2")
    ax_e.grid(False)
    ax_a.set_title("自动调温反馈：熵贴近目标 H̄ = -1")

    axes[2].plot(hc_xs, hc_arr.mean(0), "o-", color="C0", linewidth=2.2, label="SAC (HalfCheetah-v5)")
    axes[2].fill_between(hc_xs, hc_arr.mean(0) - hc_arr.std(0), hc_arr.mean(0) + hc_arr.std(0), color="C0", alpha=0.2)
    axes[2].set_xlabel("HalfCheetah-v5 环境步数")
    axes[2].set_ylabel("1000 步回合评估回报")
    axes[2].set_title("同一套 SAC 在 6 自由度猎豹跑动上的表现")
    axes[2].legend(fontsize=8.5)
    save_figure(fig, OUT / "07-sac-results.png")


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：如果把 tanh_gaussian_log_prob 里的雅可比修正项 −log(1 − tanh² u) 删掉，只留高斯本身的 log_prob，会发生什么？

练习 1 参考答案：
    当 |u| 很大（动作贴在 ±1 边界）时，真实密度因为挤压变高，不减修正项会让算出来的 log π 严重偏负（误以为熵很大），
    actor 为赚取"假熵奖金"会把 μ 推向无穷大，动作彻底卡死在 ±1 饱和区（CONCEPT.md 第 9.9 节第三条误区）。

思考题 1：SAC 在 HalfCheetah-v5 上 1.2 万步已经学会往前跑，但要到五六千分通常还要几十万步。怎样只用一两万步就大幅提速？

思考题 1 参考答案：
    把 Phase 1 的概率集成动力学模型接进来：每走一步真实环境，就从真实状态分叉出几百条 1~5 步的短模型轨迹喂给 SAC！
    这就是下一节的 MBPO。
    """)


def main():
    seed_everything(SEED)
    if torch.device("cpu").type == "cpu":
        torch.set_num_threads(min(6, os.cpu_count() or 4))
    quick = is_quick()
    if quick:
        print("（冒烟模式：环境步数大幅缩小，仅验证流程跑通）\n")
    introduction()
    verify_hand_calculations()
    run_experiments(quick)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：08-mbpo.py（概率集成短分支展开 + SAC = MBPO）")
    print("=" * 60)


if __name__ == "__main__":
    main()
