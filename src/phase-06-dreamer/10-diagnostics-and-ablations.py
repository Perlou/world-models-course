"""
Dreamer 诊断与系统消融：从上游到下游定位问题
============================================

学习目标：
    1. 核对 CONCEPT.md 第 10.3 节的"最大化偏差"手算例子（真实价值全为 0，带 ±1 噪声的 2 选 1 最大值期望 = +0.5）
    2. 实现核心诊断函数 imagined_vs_real_return(agent, env, episodes)：
       在同一策略的新评估回合上比较 Critic 估计 V_ψ(m_t) 与真实折扣回报 G_t，画价值校准散点图并监控偏差演进
    3. 实现 run_ablation(config_overrides, seeds)，严格执行多随机种子（每组 3 种子）消融纪律，
       完整运行 CONCEPT.md 第 10.5 节的全部消融项（含两条"故意引入 bug"的反面教材：起点不 detach 与奖励错开一位）

核心概念：
    - 价值校准（value calibration）：bias = mean(V_ψ(m_t) − G_t)，检验想象与现实是否对得上
    - 最大化偏差：E[max(μ̂_1, …, μ̂_n)] ≥ max μ_i，轻微高估是常态，持续发散才是钻模型漏洞
    - 多随机种子消融纪律：单种子对比没有统计意义，每次只改一处、固定种子集合

前置知识：
    - 08-dreamer-v3-world-model.py、09-dreamer-v3-agent.py

对应概念文档：
    - CONCEPT.md 第 10 节「诊断与消融：怎么知道哪里出了问题」

预计运行时间：约 3 分钟（CPU，9 组消融 × 3 种子）
"""

import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec09 = importlib.util.spec_from_file_location("p6_09", os.path.join(_DIR, "09-dreamer-v3-agent.py"))
_mod09 = importlib.util.module_from_spec(_spec09)
_spec09.loader.exec_module(_mod09)

DreamerV3Agent = _mod09.DreamerV3Agent
EpisodeReplay = _mod09.EpisodeReplay
collect_vector_episode = _mod09.collect_vector_episode

plt = setup_matplotlib()
OUT = output_dir("phase-06")
SEED = 0


# ==================== 第一部分：手算核对与价值校准函数 ====================


def verify_overestimation_hand_calc():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 10.3 节最大化偏差手算例子")
    print("=" * 60)
    combos = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    exp_max = sum(max(a, b) for a, b in combos) / len(combos)
    print(f"  两个真实价值为 0 的动作，估计各自等概率 ±1 → E[max(μ̂_1, μ̂_2)] = {exp_max:.2f}（手算 +0.50 ✓）")
    assert abs(exp_max - 0.5) < 1e-8


@torch.no_grad()
def imagined_vs_real_return(agent, env_id="Pendulum-v1", episodes=4, gamma=0.99, reward_scale=1.0, seed_base=950):
    """CONCEPT.md 第 10.6 节：评估当前策略的 Critic 价值校准（V_ψ(m_t) vs 真实折扣回报 G_t）。"""
    env = gym.make(env_id)
    act_scale = float(env.action_space.high[0])
    all_v, all_g = [], []

    for i in range(episodes):
        obs, _ = env.reset(seed=seed_base + i)
        state = agent.init_online_state()
        vs, rs = [], []
        terminated = False
        while True:
            a, state, v_est = agent.act(obs, state, deterministic=True)
            vs.append(v_est)
            obs, r, term, trunc, _ = env.step(a * act_scale)
            rs.append(float(r) * reward_scale)
            if term or trunc:
                terminated = term
                break
        # 从后往前算真实折扣回报 G_t
        gs = np.zeros(len(rs), dtype=np.float64)
        g = 0.0
        for t in reversed(range(len(rs))):
            g = rs[t] + gamma * g
            gs[t] = g
        # 若被步数上限截断，丢弃最后 80 步（≈ 接近 1/(1-γ) 的截断边界）以防真值低估
        keep = len(rs) if terminated else max(1, len(rs) - 80)
        all_v.extend(vs[:keep])
        all_g.extend(gs[:keep])

    env.close()
    v_arr = np.asarray(all_v, dtype=np.float64)
    g_arr = np.asarray(all_g, dtype=np.float64)
    bias = float(np.mean(v_arr - g_arr))
    corr = float(np.corrcoef(v_arr, g_arr)[0, 1]) if np.std(v_arr) > 1e-8 and np.std(g_arr) > 1e-8 else 0.0
    return v_arr, g_arr, bias, corr


# ==================== 第二部分：多随机种子消融框架 ====================


def run_ablation(config_overrides, seeds, n_online=10, upd_per_ep=30):
    """CONCEPT.md 第 10.7 节：在固定随机种子集合上跑同一配置，返回 [n_seeds, n_evals] 回报曲线与末轮校准信息。"""
    cfg = dict(
        deter_dim=96, n_stoch=8, n_classes=8, hidden=96,
        horizon=12, actor_mode="reinforce",
        use_ret_norm=True, use_ema_reg=True, use_twohot_critic=True,
        beta_dyn=1.0, beta_rep=0.1, free_nats=1.0, detach_start=True,
        reward_scale=1.0, shift_reward_bug=False,
    )
    cfg.update(config_overrides)
    reward_scale = cfg.pop("reward_scale")
    shift_bug = cfg.pop("shift_reward_bug")

    all_curves = []
    eval_steps_ref = None
    cal_info = None

    for s_idx, seed in enumerate(seeds):
        seed_everything(seed)
        rng = np.random.default_rng(seed)
        replay = EpisodeReplay(obs_mode="vector")
        for i in range(4):
            ep = collect_vector_episode("Pendulum-v1", seed * 10 + i, reward_scale=reward_scale, shift_reward_bug=shift_bug)
            replay.add_episode(ep["obs"], ep["actions"], ep["rewards"], ep["conts"])

        agent = DreamerV3Agent(act_dim=1, obs_mode="vector", obs_dim=3, **cfg)
        ev_steps, ev_rets = [], []
        cal_bias_hist = []

        for r_idx in range(1, n_online + 1):
            n_u = upd_per_ep * 2 if r_idx == 1 else upd_per_ep
            for _ in range(n_u):
                batch = replay.sample(16, 16, torch.device("cpu"), rng)
                agent.train_step(batch)
            ep = collect_vector_episode(
                "Pendulum-v1", seed * 100 + r_idx, agent=agent, deterministic=False,
                reward_scale=reward_scale, shift_reward_bug=shift_bug,
            )
            replay.add_episode(ep["obs"], ep["actions"], ep["rewards"], ep["conts"])

            if r_idx == 1 or r_idx % 2 == 0 or r_idx == n_online:
                evs = [collect_vector_episode("Pendulum-v1", 900 + k, agent=agent, deterministic=True)["ret"] for k in range(3)]
                ev_steps.append(len(replay) * 200)
                ev_rets.append(float(np.mean(evs)))
                if s_idx == 0:
                    va, ga, b, c = imagined_vs_real_return(agent, episodes=3, reward_scale=reward_scale)
                    cal_bias_hist.append((len(replay) * 200, b, c))
                    cal_info = (va, ga, b, c, cal_bias_hist)

        eval_steps_ref = ev_steps
        all_curves.append(ev_rets)

    return np.asarray(eval_steps_ref), np.asarray(all_curves, dtype=np.float64), cal_info


def main():
    seed_everything(SEED)
    verify_overestimation_hand_calc()

    quick = is_quick()
    seeds = [0] if quick else [0, 1, 2]
    n_online = 3 if quick else 10
    upd_per_ep = 10 if quick else 30

    print("\n" + "=" * 60)
    print(f"第二部分：运行 CONCEPT.md 第 10.5 节系统消融实验（每组 {len(seeds)} 个种子，每种子 {(4 + n_online) * 200} 步）")
    print("=" * 60)

    ablations = [
        ("完整 Dreamer V3（基线）", {}),
        ("无 free bits (free=0)", {"free_nats": 0.0}),
        ("无 KL balancing (β_dyn=β_rep=0.5)", {"beta_dyn": 0.5, "beta_rep": 0.5}),
        ("短想象视野 (H=5)", {"horizon": 5}),
        ("长想象视野 (H=25)", {"horizon": 25}),
        ("放大奖励10×：Two-Hot + 归一化", {"reward_scale": 10.0}),
        ("放大奖励10×：改用 MSE Critic", {"reward_scale": 10.0, "use_twohot_critic": False}),
        ("放大奖励10×：去掉回报归一化", {"reward_scale": 10.0, "use_ret_norm": False}),
        ("故意 Bug 1：想象起点不 detach", {"detach_start": False, "actor_mode": "pathwise"}),
        ("故意 Bug 2：奖励错开一位", {"shift_reward_bug": True}),
    ]
    if quick:
        ablations = [ablations[0], ablations[1], ablations[-2], ablations[-1]]

    results = {}
    base_cal = None
    t0 = time.time()

    for name, overrides in ablations:
        st, curves, cal = run_ablation(overrides, seeds=seeds, n_online=n_online, upd_per_ep=upd_per_ep)
        if base_cal is None:
            base_cal = cal
        mean_f = float(curves[:, -1].mean())
        std_f = float(curves[:, -1].std())
        results[name] = (st, curves)
        print(f"  {name:<34} → 最终回报 = {mean_f:7.1f} ± {std_f:5.1f}（各子: {[round(x, 0) for x in curves[:, -1].tolist()]}）")

    print(f"\n  全部消融完成，总耗时 {time.time() - t0:.1f}s。")

    # 图 1：价值校准散点图与偏差随训练的变化
    va, ga, b_final, c_final, cal_hist = base_cal
    print(f"  基线价值校准（末轮）：平均偏差 V − G = {b_final:+.2f} ｜ 相关系数 r = {c_final:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    lo = min(va.min(), ga.min()) - 5
    hi = max(va.max(), ga.max()) + 5
    axes[0].scatter(ga, va, s=14, alpha=0.5)
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="完美校准 V = G")
    axes[0].set_xlabel("真实折扣回报 G_t")
    axes[0].set_ylabel("Critic 预测价值 V_ψ(m_t)")
    axes[0].set_title(f"Dreamer V3 价值校准散点（偏差 {b_final:+.1f}，r = {c_final:.2f}）")
    axes[0].legend(fontsize=9)

    axes[1].plot([x[0] for x in cal_hist], [x[1] for x in cal_hist], "o-", color="C3", linewidth=2)
    axes[1].axhline(0.0, color="gray", linestyle=":")
    axes[1].set_xlabel("真实环境交互步数")
    axes[1].set_ylabel("平均偏差 mean(V − G)")
    axes[1].set_title("价值校准偏差随训练进程的变化")
    save_figure(fig, OUT / "10-value-calibration.png")

    # 图 2：消融汇总柱状图与学习曲线
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    names = list(results.keys())
    means = [results[n][1][:, -1].mean() for n in names]
    stds = [results[n][1][:, -1].std() for n in names]
    y_pos = np.arange(len(names))
    axes[0].barh(y_pos, means, xerr=stds, capsize=3, color=["seagreen" if i == 0 else ("tomato" if "Bug" in n else "C0") for i, n in enumerate(names)])
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names, fontsize=8.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("最终评估回报（均值 ± 种子标准差）")
    axes[0].set_title(f"Dreamer V3 消融与反面 Bug 实验汇总（{len(seeds)} 种子）")

    for i, n in enumerate(names[:5]):
        st, cur = results[n]
        axes[1].plot(st, cur.mean(axis=0), "o-", linewidth=2, label=n)
    axes[1].set_xlabel("真实环境交互步数")
    axes[1].set_ylabel("平均评估回报")
    axes[1].set_title("核心结构消融学习曲线")
    axes[1].legend(fontsize=8)
    save_figure(fig, OUT / "10-ablations.png")

    print("\n" + "=" * 60)
    print("🎉 Phase 6 全部 10 节讲义完成！下一步：运行实战项目 projects/phase-06-mini-dreamer/ 并加入连续控制排行榜！")
    print("=" * 60)


if __name__ == "__main__":
    main()
