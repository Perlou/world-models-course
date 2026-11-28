"""
Phase 10 · 第 07 课：在特征空间里做目标条件 CEM 规划（DINO-WM）
==============================================================

学习目标：
    1. 核对 CONCEPT.md 第 7.2 节（像素距离平坦 vs 表征距离单调）与第 7.3 节（特征空间一轮 CEM 手算）
    2. 原样复用 Phase 1 的 utils.planning.cem_plan，把展开函数换为冻结 DINOv2 + LatentDynamics，
       把分数换为到目标图像表征的负均方距离
    3. 测量本机 CPU 上单次规划耗时随候选数 S 与视野 H 的变化
    4. 按第 7.7 节三步隔离法对比：
       ① 完美仿真器模型 + 状态代价
       ② 学到的 DINO-WM 预测器 + 线性探针位置代价
       ③ 学到的 DINO-WM 预测器 + 纯 DINOv2 特征距离代价
       并报告想象代价与真实执行后代价的差距

对应概念文档：
    src/phase-10-jepa/CONCEPT.md 第 7 节

运行方式：
    uv run python src/phase-10-jepa/07-planning-in-feature-space.py
    WM_QUICK=1 uv run python src/phase-10-jepa/07-planning-in-feature-space.py

预计运行时间：完整约 2 分钟（M1 Max CPU），冒烟约 15 秒
"""

import importlib.util
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.planning import cem_plan
from utils.pusht import make_pusht_env

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec06 = importlib.util.spec_from_file_location("p10_06_pl", os.path.join(_DIR, "06-action-conditioned-latent-dynamics.py"))
_mod06 = importlib.util.module_from_spec(_spec06)
_spec06.loader.exec_module(_mod06)

load_frozen_dinov2 = _mod06.load_frozen_dinov2
encode_images_dinov2 = _mod06.encode_images_dinov2
precompute_features = _mod06.precompute_features
LatentDynamics = _mod06.LatentDynamics
rollout = _mod06.rollout

plt = setup_matplotlib()
OUT = output_dir("phase-10")
SEED = 0


# ==================== 第一部分：目标编码、特征代价与线性位置探针 ====================


@torch.no_grad()
def encode_goal(encoder, goal_img_uint8, img_size=112):
    return encode_images_dinov2(encoder, goal_img_uint8[None], img_size=img_size)[0]  # [64, 384]


def feature_cost(z_pred, z_goal, mode="mse"):
    """CONCEPT.md 第 7.2 节：z_pred [S, H, N, D] 终点帧到 z_goal [N, D] 的均方或 L1 距离。"""
    diff = z_pred[:, -1] - z_goal.unsqueeze(0)
    if mode == "l1":
        return diff.abs().mean(dim=(1, 2))
    return (diff ** 2).mean(dim=(1, 2))


def fit_state_probe_on_dinov2(episodes):
    """在 DINOv2 特征的 2×2 空间池化上拟合线性探针，映射回 PushT 4D 状态 [agent_x, y, block_x, y]。"""
    Zs, Ys = [], []
    for f, _, st in episodes:
        # f: [L, 64, 384] -> [L, 8, 8, 384] -> 2×2 pool -> [L, 1536]
        t = torch.from_numpy(f).reshape(len(f), 8, 8, 384).permute(0, 3, 1, 2)
        p = F.adaptive_avg_pool2d(t, (2, 2)).flatten(1).numpy()
        Zs.append(p)
        Ys.append(st[:, :4] / 512.0)
    Z = np.concatenate(Zs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    X = np.hstack([(Z - mu) / sd, np.ones((len(Z), 1), dtype=np.float32)])
    reg = 50.0 * np.eye(X.shape[1], dtype=np.float32); reg[-1, -1] = 0.0
    W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    return torch.from_numpy(mu), torch.from_numpy(sd), torch.from_numpy(W)


def apply_probe(z_batch, mu, sd, W):
    # z_batch: [S, 64, 384]
    t = z_batch.reshape(len(z_batch), 8, 8, 384).permute(0, 3, 1, 2)
    p = F.adaptive_avg_pool2d(t, (2, 2)).flatten(1)
    x = torch.cat([(p - mu) / sd, torch.ones(len(p), 1)], dim=-1)
    return x @ W  # [S, 4] in [0, 1]


def verify_section_7_hand_calcs():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 7.2–7.3 节手算")
    print("=" * 60)
    # 7.3 节一轮 CEM 手算
    cands = np.array([[0.8, 0.5], [-0.5, 0.9], [1.0, 0.7], [0.2, -0.6], [0.6, 0.4]])
    costs = (cands.sum(axis=1) - 1.5) ** 2
    elites = cands[np.argsort(costs)[:3]]
    mu = elites.mean(axis=0)
    sd = elites.std(axis=0)
    print(f"  一轮 CEM 手算：精英均值 μ = {mu.round(3).tolist()}（手算 [0.8, 0.533]），σ = {sd.round(3).tolist()}（手算 [0.163, 0.125] ✓）")
    assert np.allclose(mu, [0.8, 0.5333], atol=1e-3) and np.allclose(sd, [0.1633, 0.1247], atol=1e-3)


def benchmark_planning_latency(model, ctx_len=3, quick=False):
    print("\n" + "=" * 60)
    print("第二部分：单次特征空间 CEM 规划耗时随候选数 S 与视野 H 的变化（CPU）")
    print("=" * 60)
    z_hist = torch.randn(1, ctx_len, 64, 384)
    z_goal = torch.randn(64, 384)
    a_prev = torch.zeros(1, ctx_len - 1, 8)
    print(f"  {'候选数 S':>8s} | {'视野 H':>6s} | {'迭代 I':>6s} | {'单次规划耗时 (ms)':>16s}")
    grid_lat = {}
    S_list = [8, 16] if quick else [16, 32, 64]
    H_list = [2, 3] if quick else [2, 4, 6]
    I_iters = 2 if quick else 3
    for S in S_list:
        for H in H_list:
            t0 = time.perf_counter()

            def score_fn(cand_t):
                # cand_t: [S, H, 2] -> 每模型步重复 4 次组成 8D 跳帧动作
                cand_rep = cand_t.repeat(1, 1, 4)
                a_full = torch.cat([a_prev.repeat(len(cand_rep), 1, 1), cand_rep], dim=1)
                z_out = rollout(model, z_hist.repeat(len(cand_rep), 1, 1, 1), a_full, ctx_len=ctx_len)
                return -feature_cost(z_out, z_goal)

            cem_plan(score_fn, torch.zeros(H, 2), action_low=-1.0, action_high=1.0, n_candidates=S, n_elites=max(2, S // 4), n_iters=I_iters)
            ms = (time.perf_counter() - t0) * 1000.0
            grid_lat[(S, H)] = ms
            print(f"  {S:>8d} | {H:>6d} | {I_iters:>6d} | {ms:16.1f}")
    return grid_lat


def run_three_way_planning_ablation(encoder, model, probe_params, ctx_len=3, frame_skip=4, quick=False):
    """CONCEPT.md 第 7.7 节：在真实 PushT 目标到达任务上做三组隔离对照实验。"""
    print("\n" + "=" * 60)
    print("第三部分：PushT 目标到达规划三组隔离对照（完美模型 vs 探针代价 vs 特征代价）")
    print("=" * 60)
    mu_p, sd_p, W_p = probe_params
    env = make_pusht_env(obs_type="pixels_agent_pos")
    n_tasks = 2 if quick else 5
    goal_horizon_steps = 12  # 目标由当前起点按一段真实随机/演示推杆动作前推 12 步生成，保证严格可达

    # 构造 n_tasks 个可达 (起点种子, 目标图, 目标推杆坐标)
    tasks = []
    rng = np.random.default_rng(777)
    for k in range(n_tasks):
        seed_k = 100 + k
        obs0, _ = env.reset(seed=seed_k)
        start_pos = obs0["agent_pos"].copy()
        # 走一条平滑可控的动作轨 12 步到达目标
        dir_vec = rng.uniform(-1.0, 1.0, size=2)
        dir_vec = dir_vec / max(np.linalg.norm(dir_vec), 1e-4)
        cur_p = start_pos.copy()
        for _ in range(goal_horizon_steps):
            cur_p = np.clip(cur_p + dir_vec * 18.0, 40.0, 472.0)
            obs_g, _, _, _, _ = env.step(cur_p)
        goal_img = obs_g["pixels"].copy()
        goal_pos = obs_g["agent_pos"].copy()
        z_goal = encode_goal(encoder, goal_img, img_size=112)
        tasks.append((seed_k, goal_img, goal_pos, z_goal, float(np.linalg.norm(goal_pos - start_pos))))

    S = 12 if quick else 24
    H = 3
    I_iters = 2 if quick else 3
    n_mpc_steps = 3 if quick else 5

    summary = {}
    for mode in ["oracle_state", "learned_probe", "learned_feature"]:
        succ = 0
        final_dists, init_dists, imag_gaps = [], [], []
        for seed_k, goal_img, goal_pos, z_goal, d0 in tasks:
            obs, _ = env.reset(seed=seed_k)
            z0 = encode_goal(encoder, obs["pixels"], img_size=112)
            z_hist = z0.unsqueeze(0).unsqueeze(0).repeat(1, ctx_len, 1, 1)
            a_hist = torch.from_numpy((obs["agent_pos"].astype(np.float32) / 256.0 - 1.0)).view(1, 1, 2).repeat(1, ctx_len - 1, 4)

            for step_i in range(n_mpc_steps):
                if mode == "oracle_state":
                    # 完美一阶运动学代理：在 PushT 中推杆朝动作坐标移动（验证 CEM 配置本身能轻松到达目标）
                    cur_pos = obs["agent_pos"].copy()

                    def score_oracle(cand_t):
                        cand_np = cand_t.cpu().numpy()
                        pos_sim = np.repeat(cur_pos[None, :], len(cand_np), axis=0)
                        for h in range(H):
                            tgt = (cand_np[:, h, :] + 1.0) * 256.0
                            # 4 步内推杆向目标指令逼近
                            pos_sim = pos_sim + 0.8 * (tgt - pos_sim)
                        return torch.from_numpy(-np.linalg.norm(pos_sim - goal_pos[None, :], axis=-1).astype(np.float32))

                    best_seq = cem_plan(score_oracle, torch.zeros(H, 2), action_low=-1.0, action_high=1.0, n_candidates=S, n_elites=max(2, S // 4), n_iters=I_iters).cpu().numpy()
                else:
                    def score_wm(cand_t):
                        cand_rep = cand_t.repeat(1, 1, 4)
                        a_full = torch.cat([a_hist.repeat(len(cand_rep), 1, 1), cand_rep], dim=1)
                        z_out = rollout(model, z_hist.repeat(len(cand_rep), 1, 1, 1), a_full, ctx_len=ctx_len)
                        if mode == "learned_probe":
                            p_end = apply_probe(z_out[:, -1], mu_p, sd_p, W_p)[:, :2]
                            g_norm = torch.from_numpy(goal_pos.astype(np.float32) / 512.0).unsqueeze(0)
                            return -((p_end - g_norm) ** 2).sum(dim=-1)
                        return -feature_cost(z_out, z_goal)

                    best_seq = cem_plan(score_wm, torch.zeros(H, 2), action_low=-1.0, action_high=1.0, n_candidates=S, n_elites=max(2, S // 4), n_iters=I_iters).cpu().numpy()

                    # 记录单步想象代价 vs 真实执行后代价（仅在 learned_feature 记录）
                    if mode == "learned_feature" and step_i == 0:
                        a1_t = torch.from_numpy(best_seq[:1].astype(np.float32)).repeat(1, 1, 4)
                        z_im1 = rollout(model, z_hist, torch.cat([a_hist, a1_t], dim=1), ctx_len=ctx_len)[:, 0]
                        imag_cost_1 = float(((z_im1 - z_goal.unsqueeze(0)) ** 2).mean().item())

                act_env = np.clip((best_seq[0] + 1.0) * 256.0, 0.0, 512.0)
                for _ in range(frame_skip):
                    obs, _, term, trunc, _ = env.step(act_env)
                    if term or trunc:
                        break
                z_new = encode_goal(encoder, obs["pixels"], img_size=112)
                if mode == "learned_feature" and step_i == 0:
                    real_cost_1 = float(((z_new - z_goal) ** 2).mean().item())
                    imag_gaps.append(real_cost_1 - imag_cost_1)
                z_hist = torch.cat([z_hist[:, 1:], z_new.view(1, 1, 64, 384)], dim=1)
                a_new = torch.from_numpy(best_seq[0].astype(np.float32)).view(1, 1, 2).repeat(1, 1, 4)
                a_hist = torch.cat([a_hist[:, 1:], a_new], dim=1) if ctx_len > 2 else a_new

            df = float(np.linalg.norm(obs["agent_pos"] - goal_pos))
            init_dists.append(d0); final_dists.append(df)
            if df < 45.0:
                succ += 1
        summary[mode] = {
            "success_rate": succ / n_tasks,
            "init_dist": float(np.mean(init_dists)),
            "final_dist": float(np.mean(final_dists)),
            "imag_gap": float(np.mean(imag_gaps)) if imag_gaps else None,
        }
        print(
            f"  [{mode:16s}] 到达成功率(<45px) = {succ}/{n_tasks} ({succ/n_tasks*100:5.1f}%) | "
            f"起点均距 {np.mean(init_dists):5.1f}px → 终点均距 {np.mean(final_dists):5.1f}px"
            + (f" | 真实−想象特征代价差 = {np.mean(imag_gaps):+.4f}" if imag_gaps else "")
        )
    env.close()
    return summary


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()
    verify_section_7_hand_calcs()

    # 读取第 06 课缓存的 DINOv2 特征并快速训练预测器 + 状态探针
    n_eps = 10 if quick else 36
    cache_file = str(OUT / ("pusht_dinov2_112_quick.npz" if quick else "pusht_dinov2_112.npz"))
    episodes, _, _ = precompute_features(n_episodes=n_eps, frame_skip=4, img_size=112, cache_path=cache_file)
    encoder = load_frozen_dinov2()

    model = LatentDynamics(feat_dim=384, act_dim=8, num_patches=64, d_model=128, n_layers=2, n_heads=4, max_frames=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n_iters = 50 if quick else 350
    for _ in range(n_iters):
        zb, ab, z_tgt = [], [], []
        for _ in range(16):
            f, a, _ = episodes[np.random.randint(len(episodes))]
            t = np.random.randint(0, len(f) - 4)
            zb.append(f[t : t + 4]); ab.append(a[t : t + 4]); z_tgt.append(f[t + 1 : t + 5])
        opt.zero_grad()
        F.mse_loss(model(torch.from_numpy(np.asarray(zb)), torch.from_numpy(np.asarray(ab))), torch.from_numpy(np.asarray(z_tgt))).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    probe_params = fit_state_probe_on_dinov2(episodes)
    grid_lat = benchmark_planning_latency(model, ctx_len=3, quick=quick)
    summary = run_three_way_planning_ablation(encoder, model, probe_params, ctx_len=3, frame_skip=4, quick=quick)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    modes = ["oracle_state", "learned_probe", "learned_feature"]
    labels = ["完美模型+状态代价", "预测器+探针代价", "预测器+特征代价"]
    axes[0].bar(labels, [summary[m]["final_dist"] for m in modes], color=["tab:green", "tab:orange", "tab:blue"])
    axes[0].axhline(summary["oracle_state"]["init_dist"], ls="--", color="gray", label="初始平均距离")
    axes[0].set_title("三组隔离对照：终点到目标平均距离 (px，越低越好)"); axes[0].legend()

    keys = sorted(grid_lat.keys())
    axes[1].bar([f"S={s},H={h}" for s, h in keys], [grid_lat[k] for k in keys], color="tab:purple")
    axes[1].set_title("单次特征空间 CEM 规划耗时 (ms)"); axes[1].set_ylabel("ms")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save_figure(fig, OUT / "07-planning-in-feature-space.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：08-end-to-end-jepa-world-models.py")


if __name__ == "__main__":
    main()
