"""
MBPO：把概率集成模型的短分支展开喂给 SAC
========================================

学习目标：
    1. 核对 CONCEPT.md 第 10.3–10.4 节的复合误差放大倍数与更新数据比（UTD）记账
    2. 实现带奖励预测的概率动力学集成 ProbDynamicsEnsemble（输出 [E, M, S+1] 均值与方差）
    3. 实现核心函数 branched_rollout() 与 mbpo_train()：从真实状态出发走短分支 k 步、每步随机换集成成员、
       把模型转移写入固定容量的滚动模型池 D_model，再按比例混合真实数据训练 SAC
    4. 在 Pendulum-v1 上对比 MBPO 与纯 SAC 在相同真实交互步数下的样本效率
    5. 做两组消融实验：展开长度 k ∈ {1, 5, 20}（并对照模型开环误差曲线），以及模型数据占比的影响

核心概念：
    - 分支展开（branched rollout）：从回放池里的真实状态分叉，只走很短的 k 步，覆盖靠起点、远期靠 critic 自举
    - 更新数据比（update-to-data ratio, UTD）：每走 1 步真实环境做 G 次梯度更新，用计算换真实样本
    - 滚动模型池（D_model）：模型和策略都在更新，只保留最近几轮生成的数据，丢弃过期的旧模型样本

前置知识：
    - Phase 1 的 07-uncertainty-and-ensembles.py（概率集成与 TS1 采样）
    - 07-sac-from-scratch.py（SAC 算法）

对应概念文档：
    - CONCEPT.md 第 10 节「MBPO：把模型的短展开喂给 SAC」

预计运行时间：约 2.5 分钟（CPU）
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

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod06 = _load("p5_06", "06-ddpg-td3.py")
_mod07 = _load("p5_07", "07-sac-from-scratch.py")

ContinuousReplay = _mod06.ContinuousReplay
SACAgent = _mod07.SACAgent
evaluate_sac = _mod07.evaluate_sac

plt = setup_matplotlib()
OUT = output_dir("phase-05")
SEED = 0


# ==================== 第一部分：导读与手算核对 ====================


def introduction():
    print("=" * 60)
    print("第一部分：MBPO——用短分支展开把真实样本放大几十倍")
    print("=" * 60)
    print("""
MBPO 把三块已学组件拼在一起：
    1. 概率动力学集成 p_θ(Δs, r | s, a)（Phase 1 第 8 节）
    2. 从真实回放池 D_env 随机挑起点出发的 k 步分支展开（本节）
    3. 离策略最大熵算法 SAC（第 07 节），主要在 D_model 上高 UTD 更新

核心权衡：
    k 太小 → 数据停留在旧策略访问过的真实状态附近
    k 太大 → 复合误差按 L^k 放大，策略迅速钻模型漏洞
    """)


def verify_theory_numbers():
    """复算 CONCEPT.md 第 10.3 节复合误差倍数与第 10.4 节 UTD 记账。"""
    print("=" * 60)
    print("第二部分：核对 CONCEPT.md 第 10.3–10.4 节的记账与误差放大表")
    print("=" * 60)
    eps, L = 0.01, 1.2
    print("  1. 复合误差上界 e_k = ε (L^k − 1) / (L − 1)（ε=0.01, L=1.2）：")
    for k in [1, 5, 10, 30]:
        ek = eps * (L**k - 1.0) / (L - 1.0)
        print(f"     k = {k:<2}  L^k = {L**k:6.2f}  误差上界 = {ek:6.3f}（单步误差的 {ek / eps:6.1f} 倍）")

    M, k_len, G, B = 400, 1, 20, 256
    print(f"\n  2. UTD 记账示例（M={M}, k={k_len}, G={G}, B={B}）：")
    print(f"     每 1 真实步生成模型转移 {M * k_len} 条，SAC 消耗样本 {G * B}，UTD = {G} ✓")


# ==================== 第二部分：概率动力学 + 奖励集成 ====================


class ProbDynamicsMember(nn.Module):
    """单个高斯动力学 + 奖励模型：输入 (s, a)，输出 (Δs, r) 的均值与对数方差。"""

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        out_dim = obs_dim + 1
        self.mean_head = nn.Linear(hidden, out_dim)
        self.logvar_head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h)
        return mean, logvar


class ProbDynamicsEnsemble(nn.Module):
    """E 个独立的高斯动力学成员，带输入/目标标准化与软方差上下界（CONCEPT.md 第 10.7 节）。"""

    def __init__(self, obs_dim, act_dim, n_ensemble=5, hidden=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_ensemble = n_ensemble
        self.members = nn.ModuleList([ProbDynamicsMember(obs_dim, act_dim, hidden) for _ in range(n_ensemble)])
        self.max_logvar = nn.Parameter(torch.full((obs_dim + 1,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((obs_dim + 1,), -6.0))
        self.register_buffer("in_mean", torch.zeros(obs_dim + act_dim))
        self.register_buffer("in_std", torch.ones(obs_dim + act_dim))
        self.register_buffer("tgt_mean", torch.zeros(obs_dim + 1))
        self.register_buffer("tgt_std", torch.ones(obs_dim + 1))

    def _bound_logvar(self, raw_logvar):
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return logvar

    def predict_all(self, obs, act):
        """返回所有 E 个成员的标准化均值与对数方差，形状均为 [E, M, S + 1]。"""
        x = (torch.cat([obs, act], dim=-1) - self.in_mean) / self.in_std
        means, logvars = [], []
        for m in self.members:
            mu, lv = m(x)
            means.append(mu)
            logvars.append(self._bound_logvar(lv))
        return torch.stack(means, dim=0), torch.stack(logvars, dim=0)

    def fit(self, replay, epochs=12, batch_size=128, lr=2e-3):
        """用真实数据池 D_env 拟合集成（每个成员使用独立重排小批量）。"""
        n = len(replay)
        obs = torch.from_numpy(replay.obs[:n])
        act = torch.from_numpy(replay.act[:n])
        rew = torch.from_numpy(replay.rew[:n])
        nxt = torch.from_numpy(replay.next_obs[:n])

        x_raw = torch.cat([obs, act], dim=-1)
        y_raw = torch.cat([nxt - obs, rew], dim=-1)
        self.in_mean.copy_(x_raw.mean(dim=0))
        self.in_std.copy_(x_raw.std(dim=0).clamp_min(1e-3))
        self.tgt_mean.copy_(y_raw.mean(dim=0))
        self.tgt_std.copy_(y_raw.std(dim=0).clamp_min(1e-3))

        x_norm = (x_raw - self.in_mean) / self.in_std
        y_norm = (y_raw - self.tgt_mean) / self.tgt_std

        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        for _ in range(epochs):
            # 每个成员独立打乱下标，保持集成多样性
            perms = [torch.randperm(n) for _ in range(self.n_ensemble)]
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                loss = 0.0
                for e_idx, member in enumerate(self.members):
                    idx = perms[e_idx][start:end]
                    mu, raw_lv = member(x_norm[idx])
                    lv = self._bound_logvar(raw_lv)
                    inv_var = torch.exp(-lv)
                    nll = ((mu - y_norm[idx]).pow(2) * inv_var + lv).mean()
                    loss = loss + nll
                loss = loss / self.n_ensemble + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
                opt.zero_grad()
                loss.backward()
                opt.step()

    @torch.no_grad()
    def sample_step(self, obs, act, generator=None):
        """TS1：为每个分支随机选一个集成成员并从高斯采样，返回 (next_obs [M, S], reward [M, 1])。"""
        means_n, logvars_n = self.predict_all(obs, act)  # [E, M, S+1]
        M = obs.shape[0]
        member_idx = torch.randint(0, self.n_ensemble, (M,), generator=generator)
        b_idx = torch.arange(M)
        mu_n = means_n[member_idx, b_idx]
        std_n = torch.exp(0.5 * logvars_n[member_idx, b_idx])
        noise = torch.randn(mu_n.shape, generator=generator)
        pred = (mu_n + std_n * noise) * self.tgt_std + self.tgt_mean
        delta_s = pred[:, : self.obs_dim]
        reward = pred[:, self.obs_dim :]
        next_obs = obs + delta_s
        # Pendulum 的前两维是 (cos θ, sin θ)，重归一化到单位圆、角速度裁剪到合法范围防止离谱发散
        if self.obs_dim == 3:
            norm = torch.sqrt(next_obs[:, 0:1].pow(2) + next_obs[:, 1:2].pow(2)).clamp_min(1e-5)
            next_obs = torch.cat([next_obs[:, 0:2] / norm, next_obs[:, 2:3].clamp(-8.0, 8.0)], dim=-1)
        return next_obs, reward


# ==================== 第三部分：分支展开与 MBPO 训练循环 ====================


@torch.no_grad()
def branched_rollout(ensemble, actor, start_obs, k, model_replay, generator=None):
    """CONCEPT.md 第 10.6 节第 2b 步：从真实状态出发按当前策略走 k 步，写入模型回放池 D_model。"""
    curr_obs = start_obs.clone()
    for _ in range(k):
        act, _ = actor(curr_obs, deterministic=False)
        next_obs, rew = ensemble.sample_step(curr_obs, act, generator=generator)
        # Pendulum-v1 无早停终止（terminated 恒为 0）
        obs_np = curr_obs.cpu().numpy()
        act_np = act.cpu().numpy()
        rew_np = rew.squeeze(-1).cpu().numpy()
        nxt_np = next_obs.cpu().numpy()
        for i in range(obs_np.shape[0]):
            model_replay.add(obs_np[i], act_np[i], float(rew_np[i]), nxt_np[i], False)
        curr_obs = next_obs


def sample_mixed_batch(env_replay, model_replay, batch_size, model_ratio):
    """按 model_ratio 从 D_model 采样、其余从 D_env 采样并拼接成一个小批量。"""
    if len(model_replay) == 0 or model_ratio <= 0.0:
        return env_replay.sample(batch_size)
    if model_ratio >= 1.0:
        return model_replay.sample(batch_size)
    n_mod = max(1, min(batch_size - 1, int(round(batch_size * model_ratio))))
    n_env = batch_size - n_mod
    b_mod = model_replay.sample(n_mod)
    b_env = env_replay.sample(n_env)
    return tuple(torch.cat([xm, xe], dim=0) for xm, xe in zip(b_mod, b_env))


def mbpo_train(total_steps, rollout_k=1, model_ratio=0.9, utd=10, rollouts_per_fit=400,
               fit_every=200, init_random=400, seed=0, eval_every=400, use_model=True):
    """在 Pendulum-v1 上运行 MBPO（或 use_model=False 退化为纯 SAC 对照）。"""
    seed_everything(seed)
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    agent = SACAgent(obs_dim, act_dim, hidden=128, lr=1e-3)
    env_replay = ContinuousReplay(obs_dim, act_dim, capacity=max(20_000, total_steps + 500))
    # 模型池只保留最近 2 轮拟合生成的数据，自动淘汰过期样本
    model_cap = max(2000, rollouts_per_fit * max(1, rollout_k) * 2)
    model_replay = ContinuousReplay(obs_dim, act_dim, capacity=model_cap)
    ensemble = ProbDynamicsEnsemble(obs_dim, act_dim, n_ensemble=5, hidden=128) if use_model else None

    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    eval_steps, eval_rets = [], []

    for step in range(1, total_steps + 1):
        if step <= init_random:
            act_norm = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            act_norm = agent.select_action(obs, deterministic=False)

        next_obs, rew, terminated, truncated, _ = env.step(act_norm * act_scale)
        env_replay.add(obs, act_norm, rew, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        if step >= init_random:
            # 定期用真实池重训集成，并做一批短分支展开刷新 D_model
            if use_model and (step == init_random or (step - init_random) % fit_every == 0):
                ensemble.fit(env_replay, epochs=12, batch_size=128)
                n_starts = min(rollouts_per_fit, len(env_replay))
                idx = np.random.randint(0, len(env_replay), size=n_starts)
                start_obs = torch.from_numpy(env_replay.obs[idx])
                branched_rollout(ensemble, agent.actor, start_obs, rollout_k, model_replay, generator=gen)

            n_updates = utd if use_model else 1
            for _ in range(n_updates):
                batch = sample_mixed_batch(env_replay, model_replay, 128, model_ratio if use_model else 0.0)
                agent.sac_update(batch)

        if step % eval_every == 0 or step == total_steps:
            ev_r = evaluate_sac(agent, "Pendulum-v1", act_scale, n_episodes=5, seed_base=900)
            eval_steps.append(step)
            eval_rets.append(ev_r)

    env.close()
    return np.array(eval_steps), np.array(eval_rets), ensemble, env_replay


# ==================== 第四部分：诊断开环误差与对比实验 ====================


@torch.no_grad()
def measure_open_loop_error(ensemble, max_k=20, n_traj=30):
    """测量训练好的集成在不同开环步数 k 下的状态预测 RMSE，对照 k 消融曲线。"""
    env = gym.make("Pendulum-v1")
    act_scale = float(env.action_space.high[0])
    rng = np.random.default_rng(77)
    sq_errs = np.zeros(max_k, dtype=np.float64)

    for i in range(n_traj):
        obs, _ = env.reset(seed=700 + i)
        pred_obs = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        for k in range(max_k):
            act_norm = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
            true_next, _, _, _, _ = env.step(act_norm * act_scale)
            act_t = torch.from_numpy(act_norm).unsqueeze(0)
            pred_obs, _ = ensemble.sample_step(pred_obs, act_t)
            err = (pred_obs.squeeze(0).numpy() - true_next) ** 2
            sq_errs[k] += float(np.mean(err))
    env.close()
    return np.sqrt(sq_errs / n_traj)


def run_experiments():
    quick = is_quick()
    total_steps = 1200 if quick else 4000
    eval_every = 400
    rollouts_per_fit = 150 if quick else 400
    utd = 6 if quick else 10

    print("\n" + "=" * 60)
    print(f"第三部分：实验 1 — MBPO vs 纯 SAC（Pendulum-v1，总预算 {total_steps} 真实步）")
    print("=" * 60)
    t0 = time.time()
    s_sac, r_sac, _, _ = mbpo_train(
        total_steps=total_steps, use_model=False, eval_every=eval_every, seed=SEED
    )
    t_sac = time.time() - t0
    print(f"  纯 SAC（UTD=1）     ：2000 步附近回报 {r_sac[min(len(r_sac)-1, len(r_sac)//2)]:6.1f} ｜ "
          f"最终（{total_steps} 步）{r_sac[-1]:6.1f}（用时 {t_sac:.1f}s）")

    t1 = time.time()
    s_mbpo, r_mbpo, fitted_ens, _ = mbpo_train(
        total_steps=total_steps, rollout_k=1, model_ratio=0.9, utd=utd,
        rollouts_per_fit=rollouts_per_fit, eval_every=eval_every, seed=SEED, use_model=True
    )
    t_mbpo = time.time() - t1
    print(f"  MBPO（k=1, UTD={utd}）：2000 步附近回报 {r_mbpo[min(len(r_mbpo)-1, len(r_mbpo)//2)]:6.1f} ｜ "
          f"最终（{total_steps} 步）{r_mbpo[-1]:6.1f}（用时 {t_mbpo:.1f}s）")

    # 图 1：MBPO vs SAC 学习曲线
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(s_sac, r_sac, "o--", linewidth=2, label=f"纯 SAC（UTD=1，最终 {r_sac[-1]:.0f}）")
    ax.plot(s_mbpo, r_mbpo, "s-", linewidth=2.4, label=f"MBPO（k=1, UTD={utd}，最终 {r_mbpo[-1]:.0f}）")
    ax.axhline(-200.0, color="gray", linestyle=":", label="优秀参考线（−200）")
    ax.set_xlabel("真实环境交互步数")
    ax.set_ylabel("5 回合平均评估回报")
    ax.set_title("Pendulum-v1：MBPO 用模型短展开大幅领先纯 SAC 的早期样本效率")
    ax.legend(fontsize=9)
    save_figure(fig, OUT / "08-mbpo-vs-sac.png")

    # 实验 2：展开长度 k ∈ {1, 5, 20} 与开环误差
    print("\n" + "=" * 60)
    print("第四部分：实验 2 — 分支展开长度 k 消融与模型数据占比消融")
    print("=" * 60)
    k_values = [1, 5, 20]
    k_curves = {1: (s_mbpo, r_mbpo)}
    for k_val in [5, 20]:
        sk, rk, _, _ = mbpo_train(
            total_steps=total_steps, rollout_k=k_val, model_ratio=0.9, utd=utd,
            rollouts_per_fit=max(80, rollouts_per_fit // (1 if k_val <= 5 else 4)),
            eval_every=eval_every, seed=SEED, use_model=True,
        )
        k_curves[k_val] = (sk, rk)

    print("  1. 分支展开长度 k 对比（固定模型数据占比 90%）：")
    for k_val in k_values:
        sk, rk = k_curves[k_val]
        print(f"     k = {k_val:<2}：最高评估回报 {rk.max():6.1f} ｜ 最终回报 {rk[-1]:6.1f}")

    ol_rmse = measure_open_loop_error(fitted_ens, max_k=20)
    print(f"     对照：集成在 Pendulum-v1 上的开环状态 RMSE："
          f"k=1 为 {ol_rmse[0]:.4f}，k=5 为 {ol_rmse[4]:.4f}，k=20 为 {ol_rmse[19]:.4f}")

    # 实验 3：模型数据占比消融
    ratios = [0.0, 0.5, 0.9, 1.0] if not quick else [0.0, 0.9]
    ratio_results = {}
    print("\n  2. 模型数据占比对比（固定 k = 1, UTD = " + str(utd) + "）：")
    for ratio in ratios:
        if ratio == 0.9:
            sr, rr = s_mbpo, r_mbpo
        else:
            sr, rr, _, _ = mbpo_train(
                total_steps=total_steps, rollout_k=1, model_ratio=ratio, utd=utd,
                rollouts_per_fit=rollouts_per_fit, eval_every=eval_every, seed=SEED, use_model=True,
            )
        ratio_results[ratio] = (sr, rr)
        print(f"     model_ratio = {ratio:<3.1f}：最高回报 {rr.max():6.1f} ｜ 最终回报 {rr[-1]:6.1f}")

    # 图 2：三栏（k 曲线、开环误差、model_ratio 对比）
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))
    for k_val in k_values:
        sk, rk = k_curves[k_val]
        axes[0].plot(sk, rk, "o-", linewidth=2, label=f"k = {k_val}（最终 {rk[-1]:.0f}）")
    axes[0].set_xlabel("真实环境步数")
    axes[0].set_ylabel("评估回报")
    axes[0].set_title("展开长度 k 过大时性能下降")
    axes[0].legend(fontsize=8.5)

    axes[1].plot(np.arange(1, len(ol_rmse) + 1), ol_rmse, "o-", color="C3", linewidth=2)
    axes[1].axvline(1, color="C0", linestyle=":", label="k=1")
    axes[1].axvline(5, color="C1", linestyle=":", label="k=5")
    axes[1].axvline(20, color="C2", linestyle=":", label="k=20")
    axes[1].set_xlabel("开环展开步数 k")
    axes[1].set_ylabel("状态预测 RMSE")
    axes[1].set_title("集成开环误差随 k 迅速累积")
    axes[1].legend(fontsize=8.5)

    for ratio, (sr, rr) in ratio_results.items():
        axes[2].plot(sr, rr, "o-", linewidth=2, label=f"模型占比 {int(ratio * 100)}%")
    axes[2].set_xlabel("真实环境步数")
    axes[2].set_ylabel("评估回报")
    axes[2].set_title("高模型占比 + 少量真实锚点最稳")
    axes[2].legend(fontsize=8.5)
    save_figure(fig, OUT / "08-rollout-length-and-ratio.png")


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把模型池容量 model_cap 改成无限大（永远不淘汰旧模型生成的样本），训练曲线会变差多少？

练习 1 参考答案：
    早期模型极不准确，生成的幻觉转移如果一直留在 D_model 里被反复采样，就会持续污染后期 critic；
    只保留最近 1–2 轮重训后的模型转移，能保证 D_model 的分布始终跟随最新模型和最新策略。

思考题 1：MBPO 用 k = 1 只展开一步，为什么学出来的策略不会变得短视？

思考题 1 参考答案：
    因为 SAC 的 critic 用 γ = 0.99 做 TD 自举，把未来约 100 步的价值压进了 Q(s', a') 里；
    短展开限制的是"数据分布偏离真实动力学的程度"，而不是策略的远见。
    """)


def main():
    seed_everything(SEED)
    if is_quick():
        print("（冒烟模式：交互步数与消融规模缩小，验证完整流程）\n")
    introduction()
    verify_theory_numbers()
    run_experiments()
    exercises()
    print("\n" + "=" * 60)
    print("本节完成！下一节：09-backprop-through-model.py（沿可微模型反向传播：从 SVG 到 Dreamer）")
    print("=" * 60)


if __name__ == "__main__":
    main()
