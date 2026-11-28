"""
Phase 11 · 第 02 课：LeRobot 数据格式、动作分块与时间集成
=========================================================

学习目标：
    1. 直接读取本地缓存的 LeRobot 格式 PushT 元数据（data/pusht/raw/meta/info.json）与演示数据，
       核实回合数、总帧数、帧率、图像分辨率及动作各维范围
    2. 实现带边界填充与掩码的动作分块采样器 make_chunked_batch()
    3. 核对 CONCEPT.md 第 2.5 节决策次数（0.99^100 vs 0.99^10）与时间集成（0.541）手算
    4. 实现 replay_open_loop()：把演示动作原样在 gym-pusht 里重放，检验环境与数据集约定是否一致

对应概念文档：
    src/phase-11-robot-world-models/CONCEPT.md 第 2 节

运行方式：
    uv run python src/phase-11-robot-world-models/02-lerobot-data-and-action-chunks.py
    WM_QUICK=1 uv run python src/phase-11-robot-world-models/02-lerobot-data-and-action-chunks.py

预计运行时间：完整约 20 秒（M1 Max CPU），冒烟约 5 秒
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.pusht import load_pusht_demos, make_pusht_env

plt = setup_matplotlib()
OUT = output_dir("phase-11")
SEED = 0


def load_pusht_dataset():
    """读取本地缓存的 LeRobot PushT 元数据与 npz 演示张量。"""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    info_path = os.path.join(repo_root, "data", "pusht", "raw", "meta", "info.json")
    meta_info = {}
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            meta_info = json.load(f)
    demos = load_pusht_demos()
    return meta_info, demos


def make_chunked_batch(demos, batch_size=32, T_o=2, K=16, rng=None):
    """CONCEPT.md 第 2.6 节：切出 [B, T_o, ...] 观测窗口与 [B, K, 2] 归一化动作块及 pad_mask。"""
    rng = np.random.default_rng() if rng is None else rng
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    states, actions = demos["states"], demos["actions"]
    act_mean, act_std = actions.mean(axis=0), actions.std(axis=0) + 1e-6

    obs_st, act_ck, pad_mk = [], [], []
    for _ in range(batch_size):
        ep = int(rng.integers(0, len(starts)))
        s, e = int(starts[ep]), int(ends[ep])
        L = e - s
        t = int(rng.integers(0, L))

        # 观测窗口：越过回合开头用首帧填充
        o_idxs = [s + max(0, t - (T_o - 1 - k)) for k in range(T_o)]
        obs_st.append(states[o_idxs])

        # 动作块：越过回合结尾用末帧填充并置 pad_mask = True
        a_idxs, mask = [], []
        for k in range(K):
            if t + k < L:
                a_idxs.append(s + t + k); mask.append(False)
            else:
                a_idxs.append(e - 1); mask.append(True)
        act_ck.append((actions[a_idxs] - act_mean) / act_std)
        pad_mk.append(mask)

    return (
        torch.from_numpy(np.asarray(obs_st, dtype=np.float32)),
        torch.from_numpy(np.asarray(act_ck, dtype=np.float32)),
        torch.from_numpy(np.asarray(pad_mk, dtype=np.bool_)),
    )


def temporal_ensemble(predictions, m=0.5):
    """CONCEPT.md 第 2.5 节：按 ACT 约定，i=0 为最早预测，权重 w_i = exp(-m * i)。"""
    preds = np.asarray(predictions, dtype=np.float64)
    weights = np.exp(-m * np.arange(len(preds), dtype=np.float64))
    return (weights[:, None] * preds).sum(axis=0) / weights.sum() if preds.ndim > 1 else float((weights * preds).sum() / weights.sum())


def replay_open_loop(demos, n_episodes=20):
    """CONCEPT.md 第 2.7 节：在 gym-pusht 中把录下的演示动作原样重放，检查动作约定与初始状态复现。"""
    env = make_pusht_env(obs_type="environment_state_agent_pos")
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    actions = demos["actions"]
    states = demos["states"]  # [agent_x, agent_y, block_x, block_y, block_angle]

    succ_exact, succ_rand = 0, 0
    cov_exact, cov_rand = [], []

    for ep in range(min(n_episodes, len(starts))):
        s, e = int(starts[ep]), int(ends[ep])
        a_seq = actions[s:e]
        init_st = states[s]  # 若状态含 5 维 [ax, ay, bx, by, ang] 则精确注入初始状态

        # 1. 精确复现初始状态后重放
        env.reset(seed=ep)
        if hasattr(env.unwrapped, "_set_state") and len(init_st) >= 5:
            env.unwrapped._set_state(init_st[:5])
        max_cov = 0.0
        done_succ = False
        for a in a_seq:
            _, r, term, trunc, info = env.step(a)
            cov = float(info.get("coverage", r))
            max_cov = max(max_cov, cov)
            if info.get("is_success", False) or cov >= 0.95:
                done_succ = True
            if term or trunc:
                break
        cov_exact.append(max_cov)
        succ_exact += int(done_succ)

        # 2. 不注入初始状态（仅随机 reset）直接重放作为对照
        env.reset(seed=ep + 999)
        max_cov_r, done_r = 0.0, False
        for a in a_seq:
            _, r, term, trunc, info = env.step(a)
            cov = float(info.get("coverage", r))
            max_cov_r = max(max_cov_r, cov)
            if info.get("is_success", False) or cov >= 0.95:
                done_r = True
            if term or trunc:
                break
        cov_rand.append(max_cov_r)
        succ_rand += int(done_r)

    env.close()
    n = min(n_episodes, len(starts))
    return (succ_exact / n, float(np.mean(cov_exact))), (succ_rand / n, float(np.mean(cov_rand)))


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    print("=" * 60)
    print("第一部分：读取本地 LeRobot PushT 数据集规格")
    print("=" * 60)
    meta, demos = load_pusht_dataset()
    n_eps = len(demos["episode_starts"])
    n_frames = len(demos["images"])
    img_shape = demos["images"].shape[1:]
    fps = meta.get("fps", 10)
    a_min = demos["actions"].min(axis=0)
    a_max = demos["actions"].max(axis=0)
    s_shape = demos["states"].shape[1:]
    print(f"  回合数 = {n_eps}，总帧数 = {n_frames}，帧率 = {fps} Hz，图像分辨率 = {img_shape}，状态维度 = {s_shape}")
    print(f"  action 维度 0 范围：[{a_min[0]:.2f}, {a_max[0]:.2f}]；维度 1 范围：[{a_min[1]:.2f}, {a_max[1]:.2f}]")
    if meta:
        print(f"  LeRobot meta/info.json 字段数 = {len(meta.get('features', {}))}，codebase_version = {meta.get('codebase_version', 'N/A')}")

    print("\n" + "=" * 60)
    print("第二部分：动作分块采样与 CONCEPT.md 第 2.5 节手算核对")
    print("=" * 60)
    obs_st, act_ck, pad_mk = make_chunked_batch(demos, batch_size=16, T_o=2, K=16, rng=np.random.default_rng(SEED))
    print(f"  make_chunked_batch 形状：obs_state={tuple(obs_st.shape)}，action_chunk={tuple(act_ck.shape)}，pad_mask={tuple(pad_mk.shape)}（填充占比 {pad_mk.float().mean().item()*100:.1f}%）")
    p_100 = 0.99 ** 100
    p_10 = 0.99 ** 10
    te_val = temporal_ensemble([0.50, 0.56, 0.62], m=0.5)
    print(f"  零失误概率 (T=100, ε=0.01)：T_a=1 → {p_100:.3f}，T_a=10 → {p_10:.3f}（手算 0.366 / 0.904 ✓）")
    print(f"  时间集成 ([0.50, 0.56, 0.62], m=0.5)：{te_val:.3f}（手算 0.541 ✓）")
    assert abs(te_val - 0.541) < 0.002

    print("\n" + "=" * 60)
    print("第三部分：在 gym-pusht 中原样重放演示动作（检查数据与环境对齐）")
    print("=" * 60)
    n_rep = 8 if quick else 30
    (sr_ex, cov_ex), (sr_rn, cov_rn) = replay_open_loop(demos, n_episodes=n_rep)
    print(f"  精确复现初始状态原样重放（{n_rep} 回合）：成功率 = {sr_ex*100:5.1f}%，平均最高覆盖度 = {cov_ex:.3f}")
    print(f"  随机初始状态开环重放对照（{n_rep} 回合）：成功率 = {sr_rn*100:5.1f}%，平均最高覆盖度 = {cov_rn:.3f}")

    # 绘图：演示动作平滑曲线 vs 时间集成滤波效果
    s0, e0 = int(demos["episode_starts"][0]), int(demos["episode_ends"][0])
    raw_a = demos["actions"][s0 : min(e0, s0 + 60), 0]
    noisy_preds = raw_a + np.random.default_rng(0).normal(0, 8.0, size=len(raw_a))
    smoothed = [temporal_ensemble(noisy_preds[max(0, i - 3) : i + 1], m=0.5) for i in range(len(raw_a))]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(raw_a, label="演示原始动作 x", lw=1.6)
    axes[0].plot(noisy_preds, "--", alpha=0.5, label="带抖动预测")
    axes[0].plot(smoothed, label="时间集成平滑 (m=0.5)", lw=1.6)
    axes[0].set_title("时间集成对逐帧动作抖动的平滑效果"); axes[0].set_xlabel("步数"); axes[0].legend(fontsize=8)

    axes[1].bar(["精确初态重放", "随机初态对照"], [cov_ex, cov_rn], color=["tab:green", "gray"])
    axes[1].set_title(f"PushT 开环重放检验（精确初态成功率 {sr_ex*100:.0f}%）"); axes[1].set_ylabel("平均最高覆盖度"); axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(fig, OUT / "02-lerobot-action-chunks.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：03-behavior-cloning-and-diffusion-policy.py")


if __name__ == "__main__":
    main()
