"""
TD-MPC2：SimNorm、5-Q 集成、离散回归与最大熵策略先验
===================================================

学习目标：
    1. 核对 CONCEPT.md 第 8.2–8.4 节的全部手算例子：
       - SimNorm：(2, 0, -1, 1)（L=2, V=2）→ (0.881, 0.119, 0.119, 0.881)，||z||² = 1.58 ∈ [1, 2]
       - 101 桶 symlog two-hot（[-10, 10] 步长 0.2）：x=20 → symlog=3.0445，权重 (0.7775, 0.2225) 还原 20.0
       - 5-Q 集成随机抽 2 取小：[4.0, 4.5, 5.0, 5.5, 6.0] 的精确期望 4.50（介于全均 5.0 与全小 4.0 之间）
    2. 实现 simnorm()、symlog()、symexp()、two_hot()、two_hot_inv()、soft_cross_entropy()、QEnsemble 与完整 TD-MPC2
    3. 在 Pendulum-v1 上逐项加入 SimNorm、Q 集成、离散回归做对比实验，并监控有无 SimNorm 时隐状态范数 ||z|| 的演进
    4. 保存训练好的 TD-MPC2 模型至 outputs/phase-07/tdmpc2_pendulum.pt 供 09 节做"规划 vs 策略"诊断

核心概念：
    - SimNorm（单纯形归一化）：按组做 softmax，把 ||z||² 硬夹在 [L/V, L] 之间，防止隐状态在展开中数值漂移
    - 随机 2-of-5 Q 集成：温和悲观，既压制过估计又不过度保守
    - 离散回归 + 最大熵策略先验：尺度无关梯度 + 分位数 Q 缩放 S，实现一套超参数通吃多任务

前置知识：
    - Phase 6 的 symlog 与 two-hot（06 节）
    - 07-tdmpc-latent-model.py

对应概念文档：
    - CONCEPT.md 第 8 节「TD-MPC2：稳健与可扩展」

预计运行时间：约 2 分钟（CPU）
"""

import copy
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
from utils.planning import shift_plan

plt = setup_matplotlib()
OUT = output_dir("phase-07")
SEED = 0


# ==================== 第一部分：TD-MPC2 四大核心部件 ====================


def simnorm(x, simnorm_dim=8, tau=1.0):
    """CONCEPT.md 第 8.2 节：把最后一维切成 L 组、每组 V=simnorm_dim 维，各做 softmax。"""
    orig_shape = x.shape
    x = x.reshape(*orig_shape[:-1], -1, simnorm_dim)
    x = F.softmax(x / tau, dim=-1)
    return x.reshape(*orig_shape)


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


def make_bins(n_bins=101, low=-10.0, high=10.0, device=None):
    return torch.linspace(low, high, n_bins, device=device)


def two_hot(x_scalar, bins):
    """把原始标量 x_scalar 经 symlog 后编码为 bins 上的两点分布（TD-MPC2 101 桶默认方式）。"""
    z = symlog(x_scalar).clamp(bins[0], bins[-1])
    idx = (torch.searchsorted(bins, z, right=True) - 1).clamp(0, len(bins) - 2)
    b0, b1 = bins[idx], bins[idx + 1]
    w1 = (z - b0) / (b1 - b0).clamp_min(1e-8)
    w0 = 1.0 - w1
    out = torch.zeros(*x_scalar.shape, len(bins), device=x_scalar.device, dtype=x_scalar.dtype)
    out.scatter_(-1, idx.unsqueeze(-1), w0.unsqueeze(-1))
    out.scatter_(-1, (idx + 1).unsqueeze(-1), w1.unsqueeze(-1))
    return out


def two_hot_inv(logits, bins):
    """logits [..., M] → symlog 空间期望 → symexp 还原原始标量。"""
    probs = F.softmax(logits, dim=-1)
    return symexp((probs * bins).sum(dim=-1))


def soft_cross_entropy(logits, target_probs):
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def _mlp_block(in_d, hidden, out_d):
    return nn.Sequential(
        nn.Linear(in_d, hidden), nn.LayerNorm(hidden), nn.Mish(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish(),
        nn.Linear(hidden, out_d),
    )


class QEnsemble(nn.Module):
    """CONCEPT.md 第 8.4 节：E 个独立 Q 网络，支持标量回归或离散 two-hot 回归。"""

    def __init__(self, latent_dim, act_dim, hidden=128, n_ensemble=5, discrete=True, n_bins=101):
        super().__init__()
        self.n_ensemble = n_ensemble
        self.discrete = discrete
        out_d = n_bins if discrete else 1
        self.qs = nn.ModuleList([_mlp_block(latent_dim + act_dim, hidden, out_d) for _ in range(n_ensemble)])
        if discrete:
            for q in self.qs:
                nn.init.zeros_(q[-1].weight)
                nn.init.zeros_(q[-1].bias)
        self.register_buffer("bins", make_bins(n_bins))

    def forward_all(self, z, a):
        za = torch.cat([z, a], dim=-1)
        return torch.stack([q(za) for q in self.qs], dim=0)  # [E, B, out_d]

    def values_all(self, z, a):
        out = self.forward_all(z, a)
        return two_hot_inv(out, self.bins) if self.discrete else out.squeeze(-1)  # [E, B]

    def min_two(self, z, a):
        vals = self.values_all(z, a)  # [E, B]
        if self.n_ensemble <= 2:
            return vals.min(dim=0).values
        perm = torch.randperm(self.n_ensemble, device=z.device)[:2]
        return torch.min(vals[perm[0]], vals[perm[1]])

    def mean_q(self, z, a):
        return self.values_all(z, a).mean(dim=0)


class TDMPC2Agent(nn.Module):
    """精简版 TD-MPC2，可用开关逐项控制 SimNorm、Q 集成数与离散回归（用于消融实验）。"""

    def __init__(
        self,
        obs_dim,
        act_dim,
        latent_dim=64,
        hidden=128,
        simnorm_dim=8,
        use_simnorm=True,
        n_q=5,
        use_discrete=True,
        n_bins=101,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.simnorm_dim = simnorm_dim
        self.use_simnorm = use_simnorm
        self.use_discrete = use_discrete

        self.enc = _mlp_block(obs_dim, hidden, latent_dim)
        self.dyn = _mlp_block(latent_dim + act_dim, hidden, latent_dim)
        self.reward_head = _mlp_block(latent_dim + act_dim, hidden, n_bins if use_discrete else 1)
        if use_discrete:
            nn.init.zeros_(self.reward_head[-1].weight)
            nn.init.zeros_(self.reward_head[-1].bias)
        self.register_buffer("bins", make_bins(n_bins))

        self.q_ens = QEnsemble(latent_dim, act_dim, hidden=hidden, n_ensemble=n_q, discrete=use_discrete, n_bins=n_bins)
        self.q_targ = copy.deepcopy(self.q_ens)

        # 最大熵 tanh-高斯策略先验
        self.pi_trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish(),
        )
        self.pi_mean = nn.Linear(hidden, act_dim)
        self.pi_log_std = nn.Linear(hidden, act_dim)

        self.q_scale = 1.0
        self.q_scale_init = False

    def encode(self, obs):
        z = self.enc(obs)
        return simnorm(z, self.simnorm_dim) if self.use_simnorm else z

    def next_latent(self, z, a):
        zn = self.dyn(torch.cat([z, a], dim=-1))
        return simnorm(zn, self.simnorm_dim) if self.use_simnorm else zn

    def pred_reward(self, z, a):
        out = self.reward_head(torch.cat([z, a], dim=-1))
        return two_hot_inv(out, self.bins) if self.use_discrete else out.squeeze(-1)

    def sample_policy(self, z, deterministic=False):
        h = self.pi_trunk(z)
        mean = self.pi_mean(h)
        if deterministic:
            return torch.tanh(mean), None
        log_std = self.pi_log_std(h).clamp(-4.0, 1.0)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        act = torch.tanh(u)
        base_lp = Normal(mean, std).log_prob(u)
        corr = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
        log_prob = (base_lp - corr).sum(dim=-1)
        return act, log_prob

    @torch.no_grad()
    def estimate_value(self, z0, actions, gamma=0.99, use_terminal_q=True):
        N, H, _ = actions.shape
        z = z0.expand(N, -1)
        total = torch.zeros(N, device=z.device)
        disc = 1.0
        for k in range(H):
            a_k = actions[:, k]
            total = total + disc * self.pred_reward(z, a_k)
            z = self.next_latent(z, a_k)
            disc *= gamma
        if use_terminal_q:
            a_end, _ = self.sample_policy(z, deterministic=True)
            total = total + disc * self.q_ens.mean_q(z, a_end)
        return total

    @torch.no_grad()
    def plan(self, obs_np, prev_mean=None, horizon=3, n_samples=128, n_pi=16, n_elites=24,
             n_iters=4, lam=2.0, gamma=0.99, use_terminal_q=True, deterministic=False):
        z0 = self.encode(torch.from_numpy(np.asarray(obs_np, dtype=np.float32)[None]))
        if horizon == 0:
            a, _ = self.sample_policy(z0, deterministic=deterministic)
            return a.squeeze(0).numpy(), None

        pi_actions = None
        if n_pi > 0:
            z_p = z0.expand(n_pi, -1)
            acts = []
            for _ in range(horizon):
                a_p, _ = self.sample_policy(z_p, deterministic=False)
                acts.append(a_p)
                z_p = self.next_latent(z_p, a_p)
            pi_actions = torch.stack(acts, dim=1)

        if prev_mean is not None and prev_mean.shape[0] == horizon:
            mean = shift_plan(prev_mean)
        else:
            mean = torch.zeros(horizon, self.act_dim)
        std = torch.full((horizon, self.act_dim), 0.7)

        for _ in range(n_iters):
            gauss = (mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(n_samples, horizon, self.act_dim)).clamp(-1.0, 1.0)
            cand = torch.cat([gauss, pi_actions], dim=0) if pi_actions is not None else gauss
            scores = self.estimate_value(z0, cand, gamma=gamma, use_terminal_q=use_terminal_q)
            top_v, top_i = scores.topk(min(n_elites, cand.shape[0]))
            elites = cand[top_i]
            w = torch.softmax((top_v - top_v.max()) / lam, dim=0).view(-1, 1, 1)
            mean = (w * elites).sum(dim=0)
            std = torch.sqrt((w * (elites - mean.unsqueeze(0)).pow(2)).sum(dim=0)).clamp(0.05, 1.0)

        a0 = mean[0] if deterministic else (mean[0] + std[0] * torch.randn(self.act_dim)).clamp(-1.0, 1.0)
        return a0.numpy(), mean


# ==================== 第二部分：手算核对 ====================


def verify_section_8_hand_calcs():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 8.2–8.4 节手算例子")
    print("=" * 60)
    # 1. SimNorm 手算
    x = torch.tensor([[2.0, 0.0, -1.0, 1.0]])
    z = simnorm(x, simnorm_dim=2)
    norm2 = z.pow(2).sum().item()
    print(f"  1. SimNorm (L=2, V=2)：(2,0,-1,1) → {[round(v, 3) for v in z[0].tolist()]}，||z||² = {norm2:.2f} ∈ [1, 2] ✓")
    assert torch.allclose(z[0], torch.tensor([0.8808, 0.1192, 0.1192, 0.8808]), atol=1e-3) and 1.0 <= norm2 <= 2.0

    # 2. 101 桶 two-hot 手算
    bins = make_bins(101, -10.0, 10.0)
    th = two_hot(torch.tensor([20.0]), bins)[0]
    nz = torch.nonzero(th > 0).squeeze(-1).tolist()
    dec = symexp((th * bins).sum()).item()
    print(f"  2. 101 桶 two-hot (x=20)：非零桶值 {[round(bins[i].item(), 1) for i in nz]}，"
          f"权重 {[round(th[i].item(), 4) for i in nz]}（手算 [0.7775, 0.2225]），还原 = {dec:.2f} ✓")
    assert abs(dec - 20.0) < 1e-3 and abs(th[nz[0]].item() - 0.7775) < 2e-3

    # 3. 5-Q 随机抽 2 取小期望手算
    qs = [4.0, 4.5, 5.0, 5.5, 6.0]
    pairs = [min(qs[i], qs[j]) for i in range(5) for j in range(i + 1, 5)]
    exp_min2 = sum(pairs) / len(pairs)
    print(f"  3. 5-Q 集成 {qs}：全均 {np.mean(qs):.1f} ｜ 随机抽 2 取小精确期望 {exp_min2:.2f}（手算 4.50）｜ 全小 {min(qs):.1f} ✓")
    assert abs(exp_min2 - 4.5) < 1e-6


# ==================== 第三部分：训练与三项改进逐项消融 ====================


def train_tdmpc2_variant(use_simnorm=True, n_q=5, use_discrete=True, n_episodes=18, seed=0):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = gym.make("Pendulum-v1")
    act_scale = float(env.action_space.high[0])

    agent = TDMPC2Agent(obs_dim=3, act_dim=1, latent_dim=64, hidden=128,
                        use_simnorm=use_simnorm, n_q=n_q, use_discrete=use_discrete)
    wm_params = (
        list(agent.enc.parameters()) + list(agent.dyn.parameters())
        + list(agent.reward_head.parameters()) + list(agent.q_ens.parameters())
    )
    pi_params = list(agent.pi_trunk.parameters()) + list(agent.pi_mean.parameters()) + list(agent.pi_log_std.parameters())
    wm_opt = torch.optim.Adam(wm_params, lr=2e-3)
    pi_opt = torch.optim.Adam(pi_params, lr=1e-3)

    episodes = []
    horizon = 3
    rho = 0.5
    gamma = 0.99
    init_eps = 3 if is_quick() else 5

    eval_steps, eval_rets, z_norms = [], [], []

    for ep_idx in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed * 100 + ep_idx)
        o_l, a_l, r_l = [obs.astype(np.float32)], [], []
        prev_mean = None
        for _ in range(200):
            if ep_idx <= init_eps:
                a = rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
            else:
                a, prev_mean = agent.plan(obs, prev_mean=prev_mean, horizon=horizon,
                                          n_samples=64 if is_quick() else 128, n_pi=8 if is_quick() else 16,
                                          n_elites=16 if is_quick() else 24, n_iters=3 if is_quick() else 4,
                                          deterministic=False)
            nxt, r, term, trunc, _ = env.step(a * act_scale)
            o_l.append(nxt.astype(np.float32))
            a_l.append(a)
            r_l.append(float(r))
            obs = nxt
            if term or trunc:
                break
        episodes.append((np.asarray(o_l), np.asarray(a_l), np.asarray(r_l, dtype=np.float32)))

        if ep_idx >= init_eps:
            n_upd = 30 if is_quick() else 80
            for _ in range(n_upd):
                ob, ab, rb = [], [], []
                for _ in range(64):
                    oe, ae, re = episodes[int(rng.integers(0, len(episodes)))]
                    s = int(rng.integers(0, len(re) - horizon + 1))
                    ob.append(oe[s : s + horizon + 1])
                    ab.append(ae[s : s + horizon])
                    rb.append(re[s : s + horizon])
                obs_seq = torch.from_numpy(np.stack(ob))
                act_seq = torch.from_numpy(np.stack(ab))
                rew_seq = torch.from_numpy(np.stack(rb))

                z = agent.encode(obs_seq[:, 0])
                zs_det = []
                wm_loss = 0.0
                for i in range(horizon):
                    zs_det.append(z.detach())
                    a_i, r_i, o_nxt = act_seq[:, i], rew_seq[:, i], obs_seq[:, i + 1]
                    with torch.no_grad():
                        z_nxt_tgt = agent.encode(o_nxt)
                        a_nxt, _ = agent.sample_policy(z_nxt_tgt, deterministic=False)
                        y_i = r_i + gamma * agent.q_targ.min_two(z_nxt_tgt, a_nxt)

                    r_out = agent.reward_head(torch.cat([z, a_i], dim=-1))
                    q_out = agent.q_ens.forward_all(z, a_i)
                    z = agent.next_latent(z, a_i)

                    l_z = F.mse_loss(z, z_nxt_tgt)
                    if use_discrete:
                        l_r = soft_cross_entropy(r_out, two_hot(r_i, agent.bins)).mean()
                        y_th = two_hot(y_i, agent.bins).unsqueeze(0).expand(agent.q_ens.n_ensemble, -1, -1)
                        l_q = soft_cross_entropy(q_out, y_th).mean()
                    else:
                        l_r = F.mse_loss(r_out.squeeze(-1), r_i)
                        l_q = F.mse_loss(q_out.squeeze(-1), y_i.unsqueeze(0).expand(agent.q_ens.n_ensemble, -1))
                    wm_loss = wm_loss + (rho**i) * (2.0 * l_z + 0.5 * l_r + 0.5 * l_q)

                wm_opt.zero_grad()
                wm_loss.backward()
                nn.utils.clip_grad_norm_(wm_params, 20.0)
                wm_opt.step()

                # 最大熵策略更新（冻结 Q）
                for p in agent.q_ens.parameters():
                    p.requires_grad_(False)
                pi_loss = 0.0
                for i, zi in enumerate(zs_det):
                    a_pi, lp = agent.sample_policy(zi, deterministic=False)
                    q_val = agent.q_ens.mean_q(zi, a_pi)
                    with torch.no_grad():
                        span = (torch.quantile(q_val, 0.95) - torch.quantile(q_val, 0.05)).item()
                        agent.q_scale = span if not agent.q_scale_init else 0.99 * agent.q_scale + 0.01 * span
                        agent.q_scale_init = True
                    scale = max(1.0, agent.q_scale)
                    pi_loss = pi_loss - (rho**i) * (q_val / scale - 1e-3 * lp).mean()
                pi_opt.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(pi_params, 20.0)
                pi_opt.step()
                for p in agent.q_ens.parameters():
                    p.requires_grad_(True)

                with torch.no_grad():
                    for p, pt in zip(agent.q_ens.parameters(), agent.q_targ.parameters()):
                        pt.data.mul_(0.99).add_(p.data, alpha=0.01)

            with torch.no_grad():
                sample_o = torch.from_numpy(episodes[-1][0][:32])
                z_norms.append(agent.encode(sample_o).norm(dim=-1).mean().item())

        if ep_idx >= init_eps and (ep_idx % max(1, (n_episodes - init_eps) // 4) == 0 or ep_idx == n_episodes):
            ev_env = gym.make("Pendulum-v1")
            scores = []
            for ei in range(2 if is_quick() else 5):
                eo, _ = ev_env.reset(seed=900 + ei)
                pm = None
                tot = 0.0
                for _ in range(200):
                    ea, pm = agent.plan(eo, prev_mean=pm, horizon=horizon,
                                        n_samples=64 if is_quick() else 128, n_pi=8 if is_quick() else 16,
                                        n_elites=16 if is_quick() else 24, n_iters=3 if is_quick() else 4,
                                        deterministic=True)
                    eo, er, _, _, _ = ev_env.step(ea * act_scale)
                    tot += float(er)
                scores.append(tot)
            ev_env.close()
            eval_steps.append(ep_idx * 200)
            eval_rets.append(float(np.mean(scores)))

    env.close()
    return agent, np.array(eval_steps), np.array(eval_rets), np.array(z_norms)


def main():
    seed_everything(SEED)
    verify_section_8_hand_calcs()

    quick = is_quick()
    n_eps = 6 if quick else 16

    print("\n" + "=" * 60)
    print(f"第二部分：Pendulum-v1 上逐项加入 TD-MPC2 三项改动（共 {n_eps * 200} 真实步，H=3）")
    print("=" * 60)
    t0 = time.time()
    configs = [
        ("基础无解码器 (无SimNorm, 2Q, MSE)", False, 2, False),
        ("+ SimNorm (L=8, V=8)", True, 2, False),
        ("+ SimNorm + 5-Q 集成", True, 5, False),
        ("完整 TD-MPC2 (+ 101桶离散回归)", True, 5, True),
    ]
    if quick:
        configs = [configs[0], configs[-1]]

    results = {}
    full_agent = None
    for name, sn, nq, disc in configs:
        ag, st, rt, zn = train_tdmpc2_variant(use_simnorm=sn, n_q=nq, use_discrete=disc, n_episodes=n_eps, seed=SEED)
        results[name] = (st, rt, zn)
        if disc and sn and nq == 5:
            full_agent = ag
        print(f"  {name:<34} → 最高 {rt.max():7.1f} ｜ 最终 {rt[-1]:7.1f} ｜ 末段 ||z|| = {zn[-1]:.2f} (max {zn.max():.2f})")

    if full_agent is not None and not quick:
        ckpt_path = OUT / "tdmpc2_pendulum.pt"
        torch.save(full_agent.state_dict(), ckpt_path)
        print(f"\n  💾 完整 TD-MPC2 模型已保存至 {ckpt_path}（供 09 节直接诊断）")
    print(f"  总耗时 {time.time() - t0:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))
    for name, (st, rt, _) in results.items():
        axes[0].plot(st, rt, "o-", linewidth=2.2, label=f"{name} ({rt[-1]:.0f})")
    axes[0].set_xlabel("真实环境交互步数")
    axes[0].set_ylabel("无噪评估回报")
    axes[0].set_title("Pendulum-v1：TD-MPC2 逐项改进消融")
    axes[0].legend(fontsize=8)

    for name, (_, _, zn) in results.items():
        axes[1].plot(zn, linewidth=2.0, label=name)
    axes[1].axhline(math.sqrt(8.0), color="gray", linestyle=":", label="SimNorm 理论上限 √L = 2.83")
    axes[1].axhline(1.0, color="gray", linestyle="--", label="SimNorm 理论下限 √(L/V) = 1.0")
    axes[1].set_xlabel("训练回合")
    axes[1].set_ylabel("隐状态平均 L2 范数 ||z||")
    axes[1].set_title("SimNorm 把隐状态范数严格锁死在 [1, √8] 区间内")
    axes[1].legend(fontsize=7.5)
    save_figure(fig, OUT / "08-tdmpc2-ablation-and-simnorm.png")

    print("\n" + "=" * 60)
    print("本节完成！下一节：09-planning-vs-policy.py（同一个模型：规划 vs 策略全面拆解）")
    print("=" * 60)


if __name__ == "__main__":
    main()
