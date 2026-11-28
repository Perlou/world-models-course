"""
Phase 11 · 第 01 课：机器人学习的全景 —— 数据来源、双层控制栈与成功率误差
=========================================================================

学习目标：
    1. 对比在线交互、人类遥操作演示、无标签视频三种数据来源的字段与瓶颈
    2. 实现双层控制栈演示（10 Hz 策略目标位置 + 100 Hz 底层 PD 伺服跟踪）
    3. 核对 CONCEPT.md 第 1.7 节伯努利成功率标准误差与 95% 置信区间（正态近似 & Wilson 区间）
    4. 在真实 PushT 仿真器上评估固定策略，通过重复抽样测量 n ∈ {10, 20, 50, 100} 回合的
       成功率波动范围，与理论值 √(y(1-y)/n) 对照

对应概念文档：
    src/phase-11-robot-world-models/CONCEPT.md 第 1 节

运行方式：
    uv run python src/phase-11-robot-world-models/01-robot-learning-landscape.py
    WM_QUICK=1 uv run python src/phase-11-robot-world-models/01-robot-learning-landscape.py

预计运行时间：完整约 30 秒（M1 Max CPU），冒烟约 6 秒
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.pusht import make_pusht_env

plt = setup_matplotlib()
OUT = output_dir("phase-11")
SEED = 0


def compare_data_sources():
    print("=" * 60)
    print("第一部分：三种机器人学习数据来源的字段对比")
    print("=" * 60)
    rows = [
        ("在线交互 (RL)", "o_t, a_t, r_t, o_{t+1}, d_t", "有奖励，可自主试错", "真机极慢、不安全、需重置"),
        ("遥操作演示 (IL)", "o_t, s_prop, a_t, 指令 ℓ", "无奖励，示范质量高", "人工采集贵，单任务仅几十~几百条"),
        ("互联网无标签视频", "o_1 ... o_T (+ 文本)", "无奖励也无动作标签", "规模大几个量级，存在具身与动作缺口"),
    ]
    print(f"  {'数据来源':16s} | {'字段':28s} | {'核心特点':18s} | {'主要瓶颈'}")
    for r in rows:
        print(f"  {r[0]:16s} | {r[1]:28s} | {r[2]:18s} | {r[3]}")


def control_stack_demo():
    """CONCEPT.md 第 1.2 节：10 Hz 策略输出目标位置，100 Hz 底层 PD 控制器生成电机力矩跟踪。"""
    dt_low = 0.01   # 100 Hz 底层伺服
    dt_high = 0.10  # 10 Hz 策略网络
    T_sec = 3.0
    n_steps = int(T_sec / dt_low)
    ts = np.arange(n_steps) * dt_low

    q, qd = 0.0, 0.0
    kp, kd = 120.0, 22.0
    q_hist, target_hist, tau_hist = [], [], []
    cur_target = 0.0

    for i, t in enumerate(ts):
        if i % int(round(dt_high / dt_low)) == 0:
            # 10 Hz 策略更新目标位置（阶跃式）
            cur_target = 0.8 * math.sin(1.6 * t) + 0.25 * (1.0 if t > 1.5 else 0.0)
        tau = kp * (cur_target - q) - kd * qd
        tau = float(np.clip(tau, -30.0, 30.0))
        qd += tau * dt_low
        q += qd * dt_low
        q_hist.append(q); target_hist.append(cur_target); tau_hist.append(tau)

    track_err = float(np.mean(np.abs(np.asarray(q_hist) - np.asarray(target_hist))))
    print(f"\n第二部分：双层控制栈演示（策略 10 Hz → PD 伺服 100 Hz）")
    print(f"  3 秒轨迹平均位置跟踪误差 = {track_err:.4f} rad，最大电机力矩 = {max(abs(x) for x in tau_hist):.1f} N·m")
    return ts, np.asarray(target_hist), np.asarray(q_hist), np.asarray(tau_hist)


def success_rate_interval(k, n, z=1.96):
    """CONCEPT.md 第 1.7 节：返回成功率、正态近似 95% 区间与 Wilson 95% 区间。"""
    p = k / max(n, 1)
    se = math.sqrt(p * (1.0 - p) / max(n, 1))
    norm_ci = (max(0.0, p - z * se), min(1.0, p + z * se))
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    wilson_ci = (max(0.0, center - half), min(1.0, center + half))
    return p, se, norm_ci, wilson_ci


def eval_fixed_pusht_policy_pool(n_pool=120, max_steps=120):
    """在真实 gym-pusht 上运行一个固定启发式推块策略，收集 n_pool 个回合的二值成功结果用于抽样方差验证。"""
    env = make_pusht_env(obs_type="environment_state_agent_pos")
    outcomes = []
    coverages = []
    for ep in range(n_pool):
        obs, _ = env.reset(seed=1000 + ep)
        max_cov = 0.0
        for step in range(max_steps):
            # 固定比例控制策略：先绕到 T 块后方再朝画布中心目标 (256, 256) 推
            ag = obs["agent_pos"]
            st = obs["environment_state"]  # [block_x, block_y, block_angle]
            blk = st[:2]
            goal = np.array([256.0, 256.0], dtype=np.float32)
            push_dir = goal - blk
            norm = float(np.linalg.norm(push_dir)) + 1e-6
            behind = blk - (push_dir / norm) * 45.0
            if np.linalg.norm(ag - behind) > 35.0 and step < 30:
                act = behind
            else:
                act = blk + (push_dir / norm) * 35.0
            act = np.clip(act, 10.0, 502.0)
            obs, r, term, trunc, info = env.step(act)
            cov = float(info.get("coverage", r))
            max_cov = max(max_cov, cov)
            if term or trunc:
                break
        # 以覆盖度 >= 0.45 作为该启发式策略的二值任务达标线（产生 ~40–60% 真实成功率，方差最具代表性）
        outcomes.append(1 if max_cov >= 0.45 else 0)
        coverages.append(max_cov)
    env.close()
    return np.asarray(outcomes, dtype=np.int32), np.asarray(coverages, dtype=np.float32)


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    compare_data_sources()
    ts, tgt_q, act_q, tau = control_stack_demo()

    print("\n" + "=" * 60)
    print("第三部分：核对 CONCEPT.md 第 1.7 节理论误差表 & PushT 实测波动")
    print("=" * 60)
    print(f"  {'回合数 n':>8s} | {'理论 SE (y=0.5)':>15s} | {'理论 95% 半宽':>13s}")
    for n in [10, 20, 50, 100]:
        _, se, _, _ = success_rate_interval(n // 2, n)
        print(f"  {n:>8d} | {se:15.3f} | ±{1.96 * se:12.3f}")

    n_pool = 30 if quick else 160
    outcomes, coverages = eval_fixed_pusht_policy_pool(n_pool=n_pool, max_steps=60 if quick else 100)
    p_true = float(outcomes.mean())
    print(f"\n  PushT 固定策略实测样本池（{n_pool} 回合）：总体达标率 y = {p_true:.3f}（平均最高覆盖度 = {coverages.mean():.3f}）")

    rng = np.random.default_rng(SEED)
    n_list = [10, 20, 50, 100]
    emp_stds, theo_stds, spreads_95 = [], [], []
    print(f"  {'评估回合 n':>10s} | {'理论 SE':>9s} | {'实测标准差':>10s} | {'实测 95% 波动区间 [P2.5, P97.5]':>31s}")
    for n in n_list:
        draws = rng.choice(outcomes, size=(1000, n), replace=True).mean(axis=1)
        th_se = math.sqrt(p_true * (1.0 - p_true) / n)
        em_sd = float(draws.std())
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        emp_stds.append(em_sd); theo_stds.append(th_se); spreads_95.append((lo, hi))
        print(f"  {n:>10d} | {th_se:9.3f} | {em_sd:10.3f} | [{lo:.3f}, {hi:.3f}] (半宽 ±{(hi-lo)/2:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    axes[0].step(ts, tgt_q, where="post", label="10 Hz 策略目标位置", lw=1.6)
    axes[0].plot(ts, act_q, label="100 Hz PD 伺服实际位置", lw=1.4)
    axes[0].set_title("双层控制栈：低频策略目标 vs 高频伺服跟踪")
    axes[0].set_xlabel("时间 (s)"); axes[0].legend()

    axes[1].plot(n_list, theo_stds, "o--", label="理论标准误差 √(y(1-y)/n)")
    axes[1].plot(n_list, emp_stds, "s-", label="PushT 实测抽样标准差")
    axes[1].set_title(f"评估回合数与成功率波动（池达标率 y={p_true:.2f}）")
    axes[1].set_xlabel("评估回合数 n"); axes[1].set_ylabel("标准差"); axes[1].legend()

    fig.tight_layout()
    save_figure(fig, OUT / "01-robot-landscape.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：02-lerobot-data-and-action-chunks.py")


if __name__ == "__main__":
    main()
