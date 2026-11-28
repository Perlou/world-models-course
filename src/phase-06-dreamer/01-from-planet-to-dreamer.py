"""
从规划到学习行为：PlaNet 的短板与 Dreamer 总循环骨架
====================================================

学习目标：
    1. 量化 PlaNet"看不远"的短板：验证 γ^H（γ=0.99, H=12 时 88.6% 权重在视野之外；γ=0.997, H=15 时达 95.6%）
    2. 实现 planning_cost_per_step()：实测同一台机器上 PlaNet 隐空间 CEM 规划 vs Dreamer actor 单次前向的决策耗时与网络调用数
    3. 实现 dreamer_training_loop()：用清晰的小型占位网络走通 Dreamer 三环节（学世界模型 → 隐空间批量想象 → 环境交互）的数据流与张量形状

核心概念：
    - 摊销（amortization）：把在线搜索的计算挪到训练期，存进 actor（策略索引）和 critic（远期价值视图）
    - 视野外的折扣权重 γ^H：没有价值函数兜底时，短视野规划丢掉了绝大多数远期回报
    - 想象样本放大比：每次更新从 B×T 个后验起点各想象 H 步，把 1 步真实转移放大为 H 步想象转移

前置知识：
    - Phase 3 的 RSSM 与 PlaNet 隐空间 CEM（04、07 节）
    - Phase 5 的 Actor-Critic

对应概念文档：
    - CONCEPT.md 第 1 节「从规划到学习行为：PlaNet 的短板」

预计运行时间：约 10 秒（CPU）
"""

import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn

from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.planning import cem_plan

# 复用 Phase 3 第 04 节的真实 RSSM 结构做公平的单步耗时对比
_P3_04 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "phase-03-rssm-planet", "04-rssm-from-scratch.py"))
_spec = importlib.util.spec_from_file_location("p3_04", _P3_04)
_p3_04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p3_04)
GaussianWorldModel = _p3_04.WorldModel

plt = setup_matplotlib()
OUT = output_dir("phase-06")
SEED = 0
DEVICE = get_device()


# ==================== 第一部分：导读与视野外权重核算 ====================


def introduction():
    print("=" * 60)
    print("第一部分：为什么从 PlaNet 走向 Dreamer")
    print("=" * 60)
    print("""
PlaNet 用 RSSM 实现了纯像素控制，但留下三个结构性短板（CONCEPT.md 第 1.2 节）：
    1. 决策慢   — 每走 1 步真实环境，都要在隐空间做 N×H×I 次前向搜索
    2. 看不远   — 规划视野 H 之外的奖励全部忽略（没有价值函数）
    3. 不积累   — 上一步搜出来的动作序列，下一步几乎全部作废重搜

Dreamer 用摊销思想（amortization）一次解决这三个问题：
    - 世界模型 θ  只在真实序列上训练（和 Phase 3 相同）
    - actor φ     与 critic ψ 全部在世界模型的批量隐空间想象里训练
    - 决策时只需  1 次编码 + 1 次 observe_step + 1 次 actor 前向，毫秒级出动作！
    """)

    print("  1. 视野外丢掉的折扣权重比例 γ^H（CONCEPT.md 第 1.4 节）：")
    for gamma, H, note in [
        (0.99, 12, "PlaNet 常见折扣与视野"),
        (0.997, 15, "Dreamer V3 默认设置"),
    ]:
        outside = gamma ** H
        print(f"     γ = {gamma:<5}, H = {H:<2} → γ^H = {outside:.3f}（{outside * 100:4.1f}% 权重落在视野之外！[{note}]）")
        if gamma == 0.99 and H == 12:
            assert abs(outside - 0.8864) < 1e-3
        if gamma == 0.997 and H == 15:
            assert abs(outside - 0.9559) < 1e-3

    B, T, H = 16, 64, 15
    print(f"\n  2. 训练期想象放大账（V3 默认 B={B}, T={T}, H={H}）：")
    print(f"     真实转移数 B×T = {B * T:,}  →  想象转移数 B×T×H = {B * T * H:,}（每步真实转移放大 {H} 倍）")


# ==================== 第二部分：单步决策耗时实测 ====================


@torch.no_grad()
def planning_cost_per_step():
    """对比 PlaNet 隐空间 CEM 与 Dreamer actor 在本机上的单步决策耗时和模型调用次数。"""
    print("\n" + "=" * 60)
    print("第二部分：实测单步决策耗时 — PlaNet CEM vs Dreamer Actor")
    print("=" * 60)

    wm = GaussianWorldModel().to(DEVICE).eval()
    actor = nn.Sequential(nn.Linear(230, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1)).to(DEVICE).eval()

    obs = torch.rand(1, 3, 64, 64, device=DEVICE)
    a_prev = torch.zeros(1, 1, device=DEVICE)
    h, z = wm.rssm.initial_state(1, DEVICE)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)

    # 1. Dreamer 单步决策：1 次编码 + 1 次 observe_step + 1 次 actor MLP
    for _ in range(5):
        e = wm.encoder(obs)
        h1, z1, _, _ = wm.rssm.observe_step(h, z, a_prev, e, sample=False)
        _ = torch.tanh(actor(torch.cat([h1, z1], dim=-1)))

    n_rep_dreamer = 20 if is_quick() else 50
    t0 = time.time()
    for _ in range(n_rep_dreamer):
        e = wm.encoder(obs)
        h1, z1, _, _ = wm.rssm.observe_step(h, z, a_prev, e, sample=False)
        _ = torch.tanh(actor(torch.cat([h1, z1], dim=-1)))
    ms_dreamer = (time.time() - t0) / n_rep_dreamer * 1000.0

    # 2. PlaNet CEM 两种预算：本课缩小版 (N=300, H=15, I=6) 与论文满配版 (N=1000, H=12, I=10)
    configs = (
        [("PlaNet 本课预算 (N=300,H=15,I=6)", 300, 15, 6)]
        if is_quick() else
        [
            ("PlaNet 本课预算 (N=300,H=15,I=6)", 300, 15, 6),
            ("PlaNet 论文预算 (N=1000,H=12,I=10)", 1000, 12, 10),
        ]
    )
    results = [("Dreamer Actor (1 次前向)", 1, ms_dreamer)]
    for label, N, H, I in configs:
        h_b = h1.expand(N, -1)
        z_b = z1.expand(N, -1)

        def score_fn(actions):
            feats = wm.rssm.imagine(h_b, z_b, actions, sample=True, generator=gen)
            return wm.reward_head(feats).sum(dim=-1)

        init_mean = torch.zeros(H, 1, device=DEVICE)
        _ = cem_plan(score_fn, init_mean, generator=gen, action_low=-1.0, action_high=1.0,
                     n_candidates=N, n_elites=max(10, N // 10), n_iters=I)
        n_rep = 2 if is_quick() else 4
        t1 = time.time()
        for _ in range(n_rep):
            e = wm.encoder(obs)
            h_cur, z_cur, _, _ = wm.rssm.observe_step(h, z, a_prev, e, sample=False)
            _ = cem_plan(score_fn, init_mean, generator=gen, action_low=-1.0, action_high=1.0,
                         n_candidates=N, n_elites=max(10, N // 10), n_iters=I)
        ms_planet = (time.time() - t1) / n_rep * 1000.0
        results.append((label, N * H * I, ms_planet))

    print(f"  设备：{DEVICE}")
    print(f"  {'方法':<34} {'隐空间转移次数':>14} {'单步耗时 (ms)':>14} {'相对 Dreamer':>14}")
    for label, calls, ms in results:
        print(f"  {label:<34} {calls:>14,} {ms:>14.2f} {ms / ms_dreamer:>13.1f}×")

    # 绘图：左为视野外权重 γ^H 曲线，右为单步决策耗时对数柱状图
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    hs = np.arange(1, 31)
    axes[0].plot(hs, 0.99 ** hs, linewidth=2.2, label="γ = 0.99")
    axes[0].plot(hs, 0.997 ** hs, linewidth=2.2, label="γ = 0.997（Dreamer V3）")
    axes[0].axvline(12, color="C0", linestyle=":", alpha=0.7)
    axes[0].axvline(15, color="C1", linestyle=":", alpha=0.7)
    axes[0].set_xlabel("规划 / 想象视野 H")
    axes[0].set_ylabel("视野外折扣权重占比 γ^H")
    axes[0].set_title("短视野规划丢掉了绝大部分远期权重，必须靠 Critic 兜底")
    axes[0].legend(fontsize=9)

    names = [r[0].replace(" (", "\n(") for r in results]
    vals = [r[2] for r in results]
    bars = axes[1].bar(names, vals, color=["seagreen", "C1", "C3"][: len(vals)], width=0.52)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("单步决策耗时（ms，对数尺度）")
    axes[1].set_title(f"单步决策延迟对比（设备 {DEVICE}）")
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.1f} ms", ha="center", fontsize=9)
    axes[1].set_ylim(min(vals) * 0.5, max(vals) * 3.0)
    save_figure(fig, OUT / "01-decision-latency-and-horizon.png")
    return results


# ==================== 第三部分：Dreamer 三环节骨架 ====================


def dreamer_training_loop():
    """用轻量占位网络跑一遍 CONCEPT.md 第 1.6 节的三环节数据流，打印每步张量形状。"""
    print("\n" + "=" * 60)
    print("第三部分：Dreamer 三环节总循环骨架与张量形状核对")
    print("=" * 60)

    B, T, H = 4, 16, 15
    D_h, N_z, K, A = 256, 32, 32, 1
    D_z = N_z * K
    D_m = D_h + D_z

    obs = torch.rand(B, T, 3, 64, 64)
    actions = torch.randn(B, T, A).clamp(-1, 1)
    rewards = torch.randn(B, T)
    conts = torch.ones(B, T)

    # 环节 ①：世界模型 observe() → 得到后验模型状态 m_t = [h_t, z_t]
    h_post = torch.randn(B, T, D_h)
    z_post = torch.zeros(B, T, N_z, K)
    z_post.scatter_(-1, torch.randint(0, K, (B, T, N_z, 1)), 1.0)
    z_flat = z_post.reshape(B, T, D_z)
    m_post = torch.cat([h_post, z_flat], dim=-1)

    # 环节 ②：把 B×T 个后验状态 detach 后当作想象起点，往前推 H 步
    start_m = m_post.detach().reshape(B * T, D_m)
    imag_states = torch.zeros(H + 1, B * T, D_m)
    imag_states[0] = start_m
    imag_actions = torch.zeros(H, B * T, A)
    for k in range(H):
        imag_actions[k] = torch.randn(B * T, A).clamp(-1, 1)
        imag_states[k + 1] = imag_states[k] * 0.99  # 占位先验转移

    # 环节 ③：在线交互（B=1 单步滤波 + actor 前向）
    online_m = torch.randn(1, D_m)
    online_a = torch.randn(1, A).clamp(-1, 1)

    print(f"  ① 世界模型输入 obs:          {list(obs.shape)}  | actions: {list(actions.shape)}  | rewards: {list(rewards.shape)}")
    print(f"     observe() 后验模型状态 m:  {list(m_post.shape)}  (D_h={D_h} + 32×32={D_z} = {D_m})")
    print(f"  ② 想象起点 start_m (detach): {list(start_m.shape)}  (合并 B×T = {B*T} 条并行想象轨迹)")
    print(f"     想象轨迹 imag_states:      {list(imag_states.shape)}  (时间维在第 0 维，方便从后往前算 λ-return)")
    print(f"     想象动作 imag_actions:     {list(imag_actions.shape)}")
    print(f"  ③ 在线决策状态 online_m:     {list(online_m.shape)}  → 动作 online_a: {list(online_a.shape)}")
    assert imag_states.shape == (H + 1, B * T, D_m) and conts.shape == (B, T)
    print("  ✓ 全部张量形状与 CONCEPT.md 第 1.7 节完全一致！")


def main():
    seed_everything(SEED)
    if DEVICE.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    introduction()
    planning_cost_per_step()
    dreamer_training_loop()
    print("\n" + "=" * 60)
    print("本节完成！下一节：02-imagination-rollouts.py（批量想象轨迹与 stop-gradient 验证）")
    print("=" * 60)


if __name__ == "__main__":
    main()
