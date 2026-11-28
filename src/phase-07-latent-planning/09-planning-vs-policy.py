"""
规划还是策略：在同一个 TD-MPC2 模型上拆解每块组件的真实贡献
============================================================

学习目标：
    1. 加载 08 节训练好的同一个 TD-MPC2 模型（保证完全相同的动力学、奖励、Q 集成与策略权重）
    2. 实现 compare_horizons()：扫描规划视野 H ∈ {0（只用策略）, 1, 3, 5, 8, 12}，
       观察"短视野下规划改进策略 → 过长视野受模型复合误差拖累"的经典倒 U 型曲线
    3. 实现 policy_only_action() 与 plan_with_budget()，同台对比四种决策配置的真实回报与单步决策耗时：
       (1) 只用策略先验（H=0）
       (2) 规划但去掉策略候选（n_pi=0）
       (3) 规划但去掉终端价值（use_terminal_q=False）
       (4) 完整 TD-MPC2 规划（策略候选 + 终端价值）

核心概念：
    - 决策时计算（test-time compute）的边际收益：在合适视野 H 下，规划能稳定改进策略先验
    - 终端价值的关键地位：短视野规划的大部分分数来自终端 Q，去掉终端价值性能骤降

前置知识：
    - 08-tdmpc2.py

对应概念文档：
    - CONCEPT.md 第 9 节「规划还是策略：怎么选，怎么结合」

预计运行时间：约 40 秒（CPU）
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
_spec08 = importlib.util.spec_from_file_location("p7_08", os.path.join(_DIR, "08-tdmpc2.py"))
_mod08 = importlib.util.module_from_spec(_spec08)
_spec08.loader.exec_module(_mod08)

TDMPC2Agent = _mod08.TDMPC2Agent
train_tdmpc2_variant = _mod08.train_tdmpc2_variant

plt = setup_matplotlib()
OUT = output_dir("phase-07")
SEED = 0


def policy_only_action(agent, obs_np):
    a, _ = agent.plan(obs_np, horizon=0, deterministic=True)
    return a


def plan_with_budget(agent, eval_seeds, horizon=3, n_samples=128, n_pi=16, n_elites=24, n_iters=4, use_terminal_q=True):
    """在固定评估种子上按指定规划配置运行完整回合，返回 (均值回报, 标准差, 单步决策耗时 ms)。"""
    env = gym.make("Pendulum-v1")
    act_scale = float(env.action_space.high[0])
    rets = []
    t_dec = 0.0
    n_steps = 0

    for s in eval_seeds:
        obs, _ = env.reset(seed=int(s))
        prev_mean = None
        tot = 0.0
        for _ in range(200):
            t0 = time.time()
            a, prev_mean = agent.plan(
                obs, prev_mean=prev_mean, horizon=horizon,
                n_samples=n_samples, n_pi=n_pi, n_elites=n_elites,
                n_iters=n_iters, use_terminal_q=use_terminal_q, deterministic=True,
            )
            t_dec += time.time() - t0
            n_steps += 1
            obs, r, term, trunc, _ = env.step(a * act_scale)
            tot += float(r)
            if term or trunc:
                break
        rets.append(tot)
    env.close()
    return float(np.mean(rets)), float(np.std(rets)), (t_dec / max(1, n_steps)) * 1000.0


def compare_horizons(agent, eval_seeds, horizons):
    print("\n" + "=" * 60)
    print("第一部分：同一个 TD-MPC2 模型，扫描规划视野 H ∈ " + str(horizons))
    print("=" * 60)
    means, stds, ms_list = [], [], []
    n_s = 64 if is_quick() else 128
    n_it = 3 if is_quick() else 4
    for h in horizons:
        m, sd, ms = plan_with_budget(agent, eval_seeds, horizon=h, n_samples=n_s, n_pi=16, n_elites=20, n_iters=n_it, use_terminal_q=True)
        means.append(m)
        stds.append(sd)
        ms_list.append(ms)
        tag = "（纯策略先验）" if h == 0 else ""
        print(f"  H = {h:<2} {tag:<12} → 回合回报 = {m:7.1f} ± {sd:5.1f} ｜ 单步决策耗时 = {ms:5.2f} ms")
    return np.array(means), np.array(stds), np.array(ms_list)


def main():
    seed_everything(SEED)
    quick = is_quick()
    ckpt_path = OUT / "tdmpc2_pendulum.pt"

    if ckpt_path.exists():
        agent = TDMPC2Agent(obs_dim=3, act_dim=1, latent_dim=64, hidden=128, use_simnorm=True, n_q=5, use_discrete=True)
        agent.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        print(f"已加载 08 节训练好的 TD-MPC2 模型：{ckpt_path}")
    else:
        print("未找到预训练权重，先快速训练一个 TD-MPC2 模型……")
        agent, _, _, _ = train_tdmpc2_variant(use_simnorm=True, n_q=5, use_discrete=True, n_episodes=5 if quick else 16, seed=SEED)
    agent.eval()

    eval_seeds = [900, 901] if quick else [900, 901, 902, 903, 904, 905, 906, 907]
    horizons = [0, 1, 3, 8] if quick else [0, 1, 3, 5, 8, 12]
    h_means, h_stds, h_ms = compare_horizons(agent, eval_seeds, horizons)

    print("\n" + "=" * 60)
    print("第二部分：四种决策配置对比（回报与每步决策耗时）")
    print("=" * 60)
    n_s = 64 if is_quick() else 128
    n_it = 3 if is_quick() else 4
    configs = [
        ("只用策略先验 (H=0)", dict(horizon=0, n_samples=n_s, n_pi=0, n_iters=n_it, use_terminal_q=True)),
        ("规划但不用策略候选 (n_pi=0)", dict(horizon=3, n_samples=n_s, n_pi=0, n_iters=n_it, use_terminal_q=True)),
        ("规划但去掉终端价值 (无Q)", dict(horizon=3, n_samples=n_s, n_pi=16, n_iters=n_it, use_terminal_q=False)),
        ("完整 TD-MPC2 规划 (H=3)", dict(horizon=3, n_samples=n_s, n_pi=16, n_iters=n_it, use_terminal_q=True)),
    ]
    cfg_names, cfg_means, cfg_stds, cfg_ms = [], [], [], []
    for name, kw in configs:
        m, sd, ms = plan_with_budget(agent, eval_seeds, **kw)
        cfg_names.append(name)
        cfg_means.append(m)
        cfg_stds.append(sd)
        cfg_ms.append(ms)
        print(f"  {name:<28} → 回合回报 = {m:7.1f} ± {sd:5.1f} ｜ 单步耗时 = {ms:5.2f} ms")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    axes[0].errorbar(horizons, h_means, yerr=h_stds, fmt="o-", color="C0", linewidth=2.2, capsize=4)
    axes[0].set_xlabel("规划视野 H（H = 0 为纯策略前向）")
    axes[0].set_ylabel("真实回合回报（越高越好）")
    axes[0].set_title("固定同一个模型：短视野规划改进策略，过长受模型误差拖累")

    y_pos = np.arange(len(cfg_names))
    colors = ["C0", "C1", "tomato", "seagreen"]
    axes[1].barh(y_pos, cfg_means, xerr=cfg_stds, capsize=4, color=colors)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"{n}\n({ms:.2f} ms/步)" for n, ms in zip(cfg_names, cfg_ms)], fontsize=8.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("真实回合回报")
    axes[1].set_title("四种决策配置对比（去掉终端价值损失最大）")
    save_figure(fig, OUT / "09-planning-vs-policy.png")

    print("\n" + "=" * 60)
    print("🎉 Phase 7 全部 9 节讲义完成！下一步：运行实战项目 projects/phase-07-tdmpc2-control/ 并加入连续控制排行榜！")
    print("=" * 60)


if __name__ == "__main__":
    main()
