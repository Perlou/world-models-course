"""Phase 10 实战项目主脚本：训练 PushT DINO-WM 预测器并评测开环特征预测与目标图像条件 CEM 规划。"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(os.path.dirname(_THIS)), "src")
for _p in (_THIS, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jepa_planner import DINOWMPredictor, encode_dinov2_batch, load_offline_dinov2, plan_to_goal_image
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.pusht import load_pusht_demos, make_pusht_env

plt = setup_matplotlib()
OUT = output_dir("phase-10-project")
P10_CACHE_DIR = output_dir("phase-10")


def load_or_build_features(n_episodes=40, frame_skip=4, quick=False):
    cache_path = str(P10_CACHE_DIR / ("pusht_dinov2_112_quick.npz" if quick else "pusht_dinov2_112.npz"))
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        if int(d["n_episodes"]) >= min(n_episodes, 10) and int(d["frame_skip"]) == frame_skip:
            use_n = min(n_episodes, int(d["n_episodes"]))
            ep_lens = d["ep_lens"][:use_n]
            tot = int(ep_lens.sum())
            f_all = d["feats"][:tot].astype(np.float32)
            a_all = d["actions"][:tot].astype(np.float32)
            s_all = d["states"][:tot].astype(np.float32)
            eps, off = [], 0
            for L in ep_lens:
                L = int(L)
                eps.append((f_all[off : off + L], a_all[off : off + L], s_all[off : off + L]))
                off += L
            return eps, float(d["precompute_seconds"]), True

    t0 = time.time()
    encoder = load_offline_dinov2()
    demos = load_pusht_demos()
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    f_list, a_list, s_list, lens = [], [], [], []
    for i in range(min(n_episodes, len(starts))):
        s, e = int(starts[i]), int(ends[i])
        idxs = list(range(s, e - frame_skip, frame_skip))
        if len(idxs) < 12:
            continue
        f_seq = encode_dinov2_batch(encoder, demos["images"][idxs], img_size=112).numpy()
        a_seq = np.stack([
            (demos["actions"][idx : idx + frame_skip].astype(np.float32) / 256.0 - 1.0).reshape(-1)
            for idx in idxs
        ], axis=0)
        st_seq = demos["states"][idxs].astype(np.float32)
        f_list.append(f_seq); a_list.append(a_seq); s_list.append(st_seq); lens.append(len(idxs))
    pre_sec = time.time() - t0
    np.savez(
        cache_path,
        feats=np.concatenate(f_list, axis=0).astype(np.float16),
        actions=np.concatenate(a_list, axis=0).astype(np.float32),
        states=np.concatenate(s_list, axis=0).astype(np.float32),
        ep_lens=np.asarray(lens, dtype=np.int64),
        n_episodes=len(lens),
        frame_skip=frame_skip,
        precompute_seconds=pre_sec,
    )
    return list(zip(f_list, a_list, s_list)), pre_sec, False


def main():
    t0 = time.time()
    seed_everything(0)
    quick = is_quick()

    print("=" * 64)
    print("Phase 10 实战项目：PushT JEPA 规划器（冻结 DINOv2 + 预测器 + CEM）")
    print("=" * 64)
    frame_skip = 4
    ctx_len = 3
    episodes, pre_sec, from_cache = load_or_build_features(n_episodes=10 if quick else 36, frame_skip=frame_skip, quick=quick)
    n_tr = max(2, int(len(episodes) * 0.8))
    tr_eps, te_eps = episodes[:n_tr], episodes[n_tr:]
    print(f"  DINOv2 特征：{len(episodes)} 回合（预计算 {pre_sec:.1f}s，缓存={from_cache}）")

    predictor = DINOWMPredictor(feat_dim=384, act_dim=2 * frame_skip, num_patches=64, d_model=128, n_layers=2, n_heads=4, max_frames=8)
    opt = torch.optim.AdamW(predictor.parameters(), lr=1e-3, weight_decay=1e-4)

    n_iters = 70 if quick else 500
    t_tr = time.time()
    losses = []
    for it in range(1, n_iters + 1):
        zb, ab, zt = [], [], []
        for _ in range(16):
            f, a, _ = tr_eps[np.random.randint(len(tr_eps))]
            t = np.random.randint(0, len(f) - 4)
            zb.append(f[t : t + 4]); ab.append(a[t : t + 4]); zt.append(f[t + 1 : t + 5])
        opt.zero_grad()
        pred = predictor(torch.from_numpy(np.asarray(zb)), torch.from_numpy(np.asarray(ab)))
        loss = F.mse_loss(pred, torch.from_numpy(np.asarray(zt)))
        loss.backward(); nn.utils.clip_grad_norm_(predictor.parameters(), 1.0); opt.step()
        losses.append(float(loss.item()))
    train_sec = time.time() - t_tr
    print(f"  预测器训练 {n_iters} 步完成（耗时 {train_sec:.1f}s，末段 MSE={np.mean(losses[-20:]):.4f}）")

    # 开环特征预测 ρ（1, 5, 10, 15 步）
    horizons = (1, 5, 10) if quick else (1, 5, 10, 15)
    max_h = max(horizons)
    zh_l, as_l, gt_l = [], [], []
    for f, a, _ in te_eps:
        for t in range(0, len(f) - ctx_len - max_h, 3):
            zh_l.append(f[t : t + ctx_len])
            as_l.append(a[t : t + ctx_len + max_h - 1])
            gt_l.append(f[t + ctx_len : t + ctx_len + max_h])
    zh = torch.from_numpy(np.asarray(zh_l[:32]))
    as_t = torch.from_numpy(np.asarray(as_l[:32]))
    gt = torch.from_numpy(np.asarray(gt_l[:32]))
    pr = predictor.rollout(zh, as_t)

    open_loop_rho = {}
    open_loop_cos = {}
    for h in horizons:
        ec = float(F.mse_loss(zh[:, -1], gt[:, h - 1]).item())
        em = float(F.mse_loss(pr[:, h - 1], gt[:, h - 1]).item())
        cos = float(F.cosine_similarity(pr[:, h - 1], gt[:, h - 1], dim=-1).mean().item())
        open_loop_rho[str(h)] = em / max(ec, 1e-8)
        open_loop_cos[str(h)] = cos
        print(f"  开环 H={h:2d} 步：e_model={em:.4f}，e_copy={ec:.4f}，ρ={open_loop_rho[str(h)]:.3f}，余弦相似度={cos:.4f}")

    # 真实 gym-pusht 目标图像条件 CEM 规划评测
    encoder = load_offline_dinov2()
    env = make_pusht_env(obs_type="pixels_agent_pos")
    n_eval = 2 if quick else 6
    rng = np.random.default_rng(2026)
    succ_cnt, plan_times, init_dists, final_dists = 0, [], [], []

    for k in range(n_eval):
        seed_k = 300 + k
        obs0, _ = env.reset(seed=seed_k)
        p0 = obs0["agent_pos"].copy()
        dvec = rng.uniform(-1.0, 1.0, size=2)
        dvec = dvec / max(np.linalg.norm(dvec), 1e-4)
        cp = p0.copy()
        for _ in range(12):
            cp = np.clip(cp + dvec * 18.0, 40.0, 472.0)
            obs_g, _, _, _, _ = env.step(cp)
        goal_pos = obs_g["agent_pos"].copy()
        z_goal = encode_dinov2_batch(encoder, obs_g["pixels"][None], img_size=112)[0]
        d0 = float(np.linalg.norm(goal_pos - p0))

        obs, _ = env.reset(seed=seed_k)
        z0 = encode_dinov2_batch(encoder, obs["pixels"][None], img_size=112)[0]
        z_hist = z0.view(1, 1, 64, 384).repeat(1, ctx_len, 1, 1)
        a_prev = torch.from_numpy((obs["agent_pos"].astype(np.float32) / 256.0 - 1.0)).view(1, 1, 2).repeat(1, ctx_len - 1, frame_skip)

        for _step in range(3 if quick else 5):
            best_seq, dt_ms = plan_to_goal_image(
                predictor, z_hist, a_prev, z_goal,
                horizon=3, n_samples=12 if quick else 24, n_iters=2 if quick else 3, frame_skip=frame_skip,
            )
            plan_times.append(dt_ms)
            act_env = np.clip((best_seq[0] + 1.0) * 256.0, 0.0, 512.0)
            for _ in range(frame_skip):
                obs, _, term, trunc, _ = env.step(act_env)
                if term or trunc:
                    break
            zn = encode_dinov2_batch(encoder, obs["pixels"][None], img_size=112)[0]
            z_hist = torch.cat([z_hist[:, 1:], zn.view(1, 1, 64, 384)], dim=1)
            an = torch.from_numpy(best_seq[0].astype(np.float32)).view(1, 1, 2).repeat(1, 1, frame_skip)
            a_prev = torch.cat([a_prev[:, 1:], an], dim=1)

        df = float(np.linalg.norm(obs["agent_pos"] - goal_pos))
        init_dists.append(d0); final_dists.append(df)
        if df < 45.0:
            succ_cnt += 1
    env.close()

    results = {
        "total_seconds": time.time() - t0,
        "precompute_seconds": pre_sec,
        "train_seconds": train_sec,
        "open_loop_rho": open_loop_rho,
        "open_loop_cos": open_loop_cos,
        "plan_ms_mean": float(np.mean(plan_times)),
        "success_rate": succ_cnt / n_eval,
        "init_dist_mean": float(np.mean(init_dists)),
        "final_dist_mean": float(np.mean(final_dists)),
        "n_eval_tasks": n_eval,
    }
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  PushT 目标图像规划结果（{n_eval} 任务）：成功率(<45px) = {succ_cnt}/{n_eval} ({results['success_rate']*100:.1f}%)")
    print(f"  平均距离：{results['init_dist_mean']:.1f}px → {results['final_dist_mean']:.1f}px | 单次规划耗时：{results['plan_ms_mean']:.1f} ms")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    hs = list(horizons)
    axes[0].plot(hs, [open_loop_rho[str(h)] for h in hs], "o-", label="ρ = e_model / e_copy")
    axes[0].axhline(1.0, ls="--", color="gray", label="原样复制基线")
    axes[0].set_title("DINO-WM 开环特征预测相对误差 ρ"); axes[0].set_xlabel("模型步数 H"); axes[0].legend()

    axes[1].bar(["初始距离", "规划后终点距离"], [results["init_dist_mean"], results["final_dist_mean"]], color=["gray", "tab:blue"])
    axes[1].set_title(f"PushT 目标图像规划（成功率 {results['success_rate']*100:.0f}%，{results['plan_ms_mean']:.0f}ms/步）")
    axes[1].set_ylabel("像素距离 (px)")
    fig.tight_layout()
    save_figure(fig, OUT / "jepa-planner-summary.png")
    print(f"\n📄 结果已保存：{OUT / 'results.json'}（总用时 {results['total_seconds']:.1f}s）")


if __name__ == "__main__":
    main()
