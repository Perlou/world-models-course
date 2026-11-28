"""Phase 12 毕业项目主实验脚本：PushT 零样本 JEPA 目标规划器（多种子 + 2×2 预注册消融）。

遵循 `PLAN.md` 的预注册实验矩阵与 Phase 12 工程规范：
1. 数据校验、按回合切分（严格隔离训练/验证/测试）、数据集内容指纹记录
2. 双编码器表征诊断（离线 DINOv2 ViT-S/14 vs 未预训练冻结 PatchCNN）：线性探针 RMSE / R²
3. 多种子（完整模式 n=5，冒烟模式 n=2）训练：
   - 完整方法：DINOv2 + JEPA + CEM (H=3, patch_l2)
   - 消融格 (1,0)：DINOv2 + JEPA + CEM (H=1, patch_l2)
   - 消融格 (0,1)：DINOv2 + JEPA + CEM (H=3, mean_pool_l2)
   - 消融格 (0,0)：DINOv2 + JEPA + CEM (H=1, mean_pool_l2)
   - 编码器对照：Random-PatchCNN + JEPA + CEM (H=3, patch_l2)
   - 离线基线：Goal-Conditioned BC（相同演示轨迹训练）
   - 参照上下界：Random Policy 与 Oracle-CEM
4. 每格每种子的原始逐任务评估结果写入 `outputs/phase-12-project/runs/<cell>/seed-<k>/eval.jsonl`
5. 自动汇总开环误差曲线、可信视野 k*、95% t 区间、Wilson 区间、2×2 相互作用与 Markdown 结果表
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
_SRC = os.path.join(_ROOT, "src")
_P10_PROJ = os.path.join(_ROOT, "projects", "phase-10-jepa-planner")
for _p in (_THIS, _SRC, _P10_PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from capstone_jepa import (
    CapstoneJEPAPredictor,
    GoalConditionedBC,
    RandomPatchCNNEncoder,
    compute_episodes_fingerprint,
    credible_horizon,
    diff_ci,
    fit_and_eval_state_probe,
    interaction_2x2,
    mean_and_ci,
    plan_jepa_cem,
    split_by_episode,
    wilson_interval,
)
from jepa_planner import encode_dinov2_batch, load_offline_dinov2
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.planning import cem_plan
from utils.pusht import load_pusht_demos, make_pusht_env

plt = setup_matplotlib()
OUT = output_dir("phase-12-project")
P10_CACHE = output_dir("phase-10")


def load_or_build_dinov2_episodes(n_episodes, frame_skip=4, quick=False):
    """复用或构建离线 DINOv2 ViT-S/14 的 PushT 回合特征。"""
    cache_path = P10_CACHE / ("pusht_dinov2_112_quick.npz" if quick else "pusht_dinov2_112.npz")
    if cache_path.exists():
        d = np.load(cache_path)
        if int(d["n_episodes"]) >= min(n_episodes, 10) and int(d["frame_skip"]) == frame_skip:
            use_n = min(n_episodes, int(d["n_episodes"]))
            lens = d["ep_lens"][:use_n]
            tot = int(lens.sum())
            f_all = d["feats"][:tot].astype(np.float32)
            a_all = d["actions"][:tot].astype(np.float32)
            s_all = d["states"][:tot].astype(np.float32)
            eps, off = [], 0
            for L in lens:
                L = int(L)
                eps.append((f_all[off : off + L], a_all[off : off + L], s_all[off : off + L]))
                off += L
            return eps, float(d["precompute_seconds"]), True

    t0 = time.time()
    encoder = load_offline_dinov2()
    demos = load_pusht_demos()
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    f_l, a_l, s_l, lens = [], [], [], []
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
        s_seq = demos["states"][idxs].astype(np.float32)
        f_l.append(f_seq); a_l.append(a_seq); s_l.append(s_seq); lens.append(len(idxs))
    pre_sec = time.time() - t0
    np.savez(
        cache_path,
        feats=np.concatenate(f_l, axis=0).astype(np.float16),
        actions=np.concatenate(a_l, axis=0).astype(np.float32),
        states=np.concatenate(s_l, axis=0).astype(np.float32),
        ep_lens=np.asarray(lens, dtype=np.int64),
        n_episodes=len(lens), frame_skip=frame_skip, precompute_seconds=pre_sec,
    )
    return list(zip(f_l, a_l, s_l)), pre_sec, False


def build_random_cnn_episodes(rand_enc, n_episodes, frame_skip=4):
    """用未预训练冻结 PatchCNN 编码与 DINOv2 相同的 PushT 演示回合。"""
    demos = load_pusht_demos()
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    eps = []
    for i in range(min(n_episodes, len(starts))):
        s, e = int(starts[i]), int(ends[i])
        idxs = list(range(s, e - frame_skip, frame_skip))
        if len(idxs) < 12:
            continue
        f_seq = rand_enc.encode_images(demos["images"][idxs], img_size=112).numpy()
        a_seq = np.stack([
            (demos["actions"][idx : idx + frame_skip].astype(np.float32) / 256.0 - 1.0).reshape(-1)
            for idx in idxs
        ], axis=0)
        s_seq = demos["states"][idxs].astype(np.float32)
        eps.append((f_seq, a_seq, s_seq))
    return eps


def train_predictor(tr_eps, feat_dim, act_dim, seed, n_iters=220, batch_size=16):
    """在训练回合上训练动作条件潜空间预测器。"""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = CapstoneJEPAPredictor(
        feat_dim=feat_dim, act_dim=act_dim, num_patches=64,
        d_model=128, n_layers=2, n_heads=4, max_frames=8,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    losses = []
    for _ in range(n_iters):
        zb, ab, zt = [], [], []
        for _ in range(batch_size):
            f, a, _ = tr_eps[int(rng.integers(0, len(tr_eps)))]
            t = int(rng.integers(0, len(f) - 4))
            zb.append(f[t : t + 4]); ab.append(a[t : t + 4]); zt.append(f[t + 1 : t + 5])
        opt.zero_grad()
        pred = model(torch.from_numpy(np.asarray(zb)), torch.from_numpy(np.asarray(ab)))
        loss = F.mse_loss(pred, torch.from_numpy(np.asarray(zt)))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
    model.eval()
    return model, float(np.mean(losses[-20:]))


def train_goal_bc(tr_eps, feat_dim, seed, n_iters=250, batch_size=32):
    """在相同的 PushT 训练回合上离线训练 Goal-Conditioned BC（用未来 1~4 步作目标帧）。"""
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    policy = GoalConditionedBC(feat_dim=feat_dim, hidden=128)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(n_iters):
        zc, zg, at = [], [], []
        for _ in range(batch_size):
            f, a, _ = tr_eps[int(rng.integers(0, len(tr_eps)))]
            t = int(rng.integers(0, len(f) - 5))
            dt = int(rng.integers(1, 5))
            zc.append(f[t])
            zg.append(f[t + dt])
            # a[t] 含 4 帧跳帧动作 [8]，取第一步对应的目标动作 [2]
            at.append(a[t, :2])
        opt.zero_grad()
        pred = policy(torch.from_numpy(np.asarray(zc)), torch.from_numpy(np.asarray(zg)))
        loss = F.mse_loss(pred, torch.from_numpy(np.asarray(at)))
        loss.backward()
        opt.step()
    policy.eval()
    return policy


def eval_open_loop(predictor, te_eps, ctx_len=3, horizons=(1, 2, 4, 6, 8, 10)):
    """计算测试集上的开环预测误差曲线 e_model(k)、复制基线 e_copy(k) 与可信视野 k*。"""
    max_h = max(horizons)
    zh_l, as_l, gt_l = [], [], []
    for f, a, _ in te_eps:
        for t in range(0, len(f) - ctx_len - max_h, 4):
            zh_l.append(f[t : t + ctx_len])
            as_l.append(a[t : t + ctx_len + max_h - 1])
            gt_l.append(f[t + ctx_len : t + ctx_len + max_h])
    zh = torch.from_numpy(np.asarray(zh_l[:24]))
    as_t = torch.from_numpy(np.asarray(as_l[:24]))
    gt = torch.from_numpy(np.asarray(gt_l[:24]))
    pr = predictor.rollout(zh, as_t)

    e_model, e_copy, rho = [], [], []
    for h in horizons:
        ec = float(F.mse_loss(zh[:, -1], gt[:, h - 1]).item())
        em = float(F.mse_loss(pr[:, h - 1], gt[:, h - 1]).item())
        e_model.append(em)
        e_copy.append(ec)
        rho.append(em / max(ec, 1e-8))
    k_star = credible_horizon(e_model, e_copy, threshold=0.5, ks=horizons)
    return {"horizons": list(horizons), "e_model": e_model, "e_copy": e_copy, "rho": rho, "k_star": k_star}


def build_eval_tasks(dino_enc, rand_enc, n_tasks=8):
    """构造固定的一组 PushT 可达目标到达任务（不与训练数据重叠）。"""
    env = make_pusht_env(obs_type="pixels_agent_pos")
    rng = np.random.default_rng(202612)
    tasks = []
    for k in range(n_tasks):
        seed_k = 700 + k
        obs0, _ = env.reset(seed=seed_k)
        p0 = obs0["agent_pos"].copy()
        dvec = rng.uniform(-1.0, 1.0, size=2)
        dvec = dvec / max(np.linalg.norm(dvec), 1e-4)
        cp = p0.copy()
        for _ in range(12):
            cp = np.clip(cp + dvec * 18.0, 40.0, 472.0)
            obs_g, _, _, _, _ = env.step(cp)
        goal_pos = obs_g["agent_pos"].copy()
        goal_img = obs_g["pixels"].copy()
        z_goal_dino = encode_dinov2_batch(dino_enc, goal_img[None], img_size=112)[0]
        z_goal_rand = rand_enc.encode_images(goal_img[None], img_size=112)[0]
        tasks.append({
            "task_id": k,
            "seed": seed_k,
            "start_pos": p0,
            "goal_pos": goal_pos,
            "init_dist": float(np.linalg.norm(goal_pos - p0)),
            "z_goal_dino": z_goal_dino,
            "z_goal_rand": z_goal_rand,
        })
    env.close()
    return tasks


def eval_tasks_in_env(mode, tasks, dino_enc, rand_enc, model=None, horizon=3, cost_type="patch_l2",
                      n_candidates=16, n_iters=2, n_mpc_steps=4, frame_skip=4, ctx_len=3, eval_seed_offset=0):
    """在真实 gym-pusht 上评测指定策略/规划器，返回逐任务原始记录列表。"""
    env = make_pusht_env(obs_type="pixels_agent_pos")
    records = []
    for task in tasks:
        obs, _ = env.reset(seed=task["seed"])
        gen = torch.Generator().manual_seed(task["seed"] + eval_seed_offset * 100)
        step_ms = []

        if mode in ("dino_jepa", "rand_jepa"):
            use_dino = mode == "dino_jepa"
            z0 = encode_dinov2_batch(dino_enc, obs["pixels"][None], img_size=112)[0] if use_dino else rand_enc.encode_images(obs["pixels"][None], img_size=112)[0]
            z_goal = task["z_goal_dino"] if use_dino else task["z_goal_rand"]
            z_hist = z0.unsqueeze(0).unsqueeze(0).repeat(1, ctx_len, 1, 1)
            a_prev = torch.from_numpy((obs["agent_pos"].astype(np.float32) / 256.0 - 1.0)).view(1, 1, 2).repeat(1, ctx_len - 1, frame_skip)

        for _ in range(n_mpc_steps):
            t_s = time.perf_counter()
            if mode == "random":
                act_m11 = (torch.rand(2, generator=gen).numpy() * 2.0 - 1.0).astype(np.float32)
            elif mode == "oracle_cem":
                cur_pos = obs["agent_pos"].copy()
                g_pos = task["goal_pos"]

                def score_oracle(cand_t):
                    cnp = cand_t.cpu().numpy()
                    pos = np.repeat(cur_pos[None, :], len(cnp), axis=0)
                    for h in range(horizon):
                        tgt = (cnp[:, h, :] + 1.0) * 256.0
                        pos = pos + 0.8 * (tgt - pos)
                    return torch.from_numpy(-np.linalg.norm(pos - g_pos[None, :], axis=-1).astype(np.float32))

                best = cem_plan(
                    score_oracle, torch.zeros(horizon, 2), generator=gen,
                    action_low=-1.0, action_high=1.0,
                    n_candidates=n_candidates, n_elites=max(2, n_candidates // 4), n_iters=n_iters,
                ).cpu().numpy()
                act_m11 = best[0]
            elif mode == "goal_bc":
                z_cur = encode_dinov2_batch(dino_enc, obs["pixels"][None], img_size=112)[0]
                with torch.no_grad():
                    act_m11 = model(z_cur.unsqueeze(0), task["z_goal_dino"].unsqueeze(0))[0].numpy()
            else:
                best, _ = plan_jepa_cem(
                    model, z_hist, a_prev, z_goal,
                    horizon=horizon, n_candidates=n_candidates, n_iters=n_iters,
                    frame_skip=frame_skip, cost_type=cost_type, generator=gen,
                )
                act_m11 = best[0]

            step_ms.append((time.perf_counter() - t_s) * 1000.0)
            act_env = np.clip((act_m11 + 1.0) * 256.0, 0.0, 512.0)
            for _ in range(frame_skip):
                obs, _, term, trunc, _ = env.step(act_env)
                if term or trunc:
                    break

            if mode in ("dino_jepa", "rand_jepa"):
                zn = encode_dinov2_batch(dino_enc, obs["pixels"][None], img_size=112)[0] if use_dino else rand_enc.encode_images(obs["pixels"][None], img_size=112)[0]
                z_hist = torch.cat([z_hist[:, 1:], zn.unsqueeze(0).unsqueeze(0)], dim=1)
                an = torch.from_numpy(act_m11.astype(np.float32)).view(1, 1, 2).repeat(1, 1, frame_skip)
                a_prev = torch.cat([a_prev[:, 1:], an], dim=1)

        final_dist = float(np.linalg.norm(obs["agent_pos"] - task["goal_pos"]))
        records.append({
            "task_id": task["task_id"],
            "init_dist": task["init_dist"],
            "final_dist": final_dist,
            "success": bool(final_dist < 45.0),
            "step_ms": float(np.mean(step_ms)),
        })
    env.close()
    return records


def main():
    t0 = time.time()
    quick = is_quick()
    seed_everything(0)

    print("=" * 68)
    print("Phase 12 毕业项目：PushT 零样本 JEPA 目标规划器（多种子 + 2×2 消融）")
    print("=" * 68)

    frame_skip = 4
    ctx_len = 3
    n_episodes = 12 if quick else 36
    n_seeds = 2 if quick else 5
    n_tasks = 3 if quick else 8
    pred_iters = 60 if quick else 260
    bc_iters = 80 if quick else 260
    n_cand = 10 if quick else 20
    n_iters_cem = 2 if quick else 3
    n_mpc_steps = 3 if quick else 5
    horizons = (1, 3, 6) if quick else (1, 2, 4, 6, 8, 10)

    # 1. 加载与校验离线数据（双编码器 + 内容指纹 + 严格按回合切分）
    dino_enc = load_offline_dinov2()
    rand_enc = RandomPatchCNNEncoder(feat_dim=128, seed=1234)
    dino_eps, pre_sec, from_cache = load_or_build_dinov2_episodes(n_episodes, frame_skip=frame_skip, quick=quick)
    rand_eps = build_random_cnn_episodes(rand_enc, len(dino_eps), frame_skip=frame_skip)

    fp_dino = compute_episodes_fingerprint(dino_eps)
    fp_rand = compute_episodes_fingerprint(rand_eps)
    split = split_by_episode(len(dino_eps), seed=0)
    tr_ids, va_ids, te_ids = split["train"], split["val"], split["test"]
    print(f"  数据规模：{len(dino_eps)} 回合（训练 {len(tr_ids)} / 验证 {len(va_ids)} / 测试 {len(te_ids)}）")
    print(f"  内容指纹：DINOv2={fp_dino}（缓存={from_cache}，预计算 {pre_sec:.1f}s）| Random-PatchCNN={fp_rand}")

    # 2. 表征线性探针诊断（推杆坐标 (x, y)）
    def _concat_eps(eps, ids):
        return (
            np.concatenate([eps[i][0] for i in ids], axis=0),
            np.concatenate([eps[i][2] for i in ids], axis=0),
        )

    dino_tr_f, dino_tr_s = _concat_eps(dino_eps, tr_ids)
    dino_te_f, dino_te_s = _concat_eps(dino_eps, te_ids)
    rand_tr_f, rand_tr_s = _concat_eps(rand_eps, tr_ids)
    rand_te_f, rand_te_s = _concat_eps(rand_eps, te_ids)
    probe_dino = fit_and_eval_state_probe(dino_tr_f, dino_tr_s, dino_te_f, dino_te_s)
    probe_rand = fit_and_eval_state_probe(rand_tr_f, rand_tr_s, rand_te_f, rand_te_s)
    print(f"\n  [线性探针：读取推杆坐标 (x, y)]")
    print(f"    DINOv2 ViT-S/14  ：测试集 R² = {probe_dino['r2']:.4f}，RMSE = {probe_dino['rmse_px']:.1f} px")
    print(f"    Random-PatchCNN  ：测试集 R² = {probe_rand['r2']:.4f}，RMSE = {probe_rand['rmse_px']:.1f} px")

    # 3. 构造固定评估任务集合
    tasks = build_eval_tasks(dino_enc, rand_enc, n_tasks=n_tasks)
    print(f"\n  固定评估任务：{n_tasks} 个可达 PushT 目标（平均起点距离 {np.mean([t['init_dist'] for t in tasks]):.1f} px）")

    # 4. 清理并初始化原始记录目录
    runs_dir = OUT / ("runs-quick" if quick else "runs")
    shutil.rmtree(runs_dir, ignore_errors=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 参照基线（无训练种子：Random Policy & Oracle-CEM）
    rand_recs = eval_tasks_in_env("random", tasks, dino_enc, rand_enc, n_mpc_steps=n_mpc_steps, frame_skip=frame_skip)
    oracle_recs = eval_tasks_in_env("oracle_cem", tasks, dino_enc, rand_enc, horizon=3, n_candidates=n_cand, n_iters=n_iters_cem, n_mpc_steps=n_mpc_steps, frame_skip=frame_skip)

    # 多种子训练与评估（每种子先训 1 个 DINOv2 预测器、1 个 Random-PatchCNN 预测器、1 个 Goal-BC）
    dino_tr_eps = [dino_eps[i] for i in tr_ids]
    dino_te_eps = [dino_eps[i] for i in te_ids]
    rand_tr_eps = [rand_eps[i] for i in tr_ids]
    rand_te_eps = [rand_eps[i] for i in te_ids]

    cell_specs = [
        ("dino_full_H3_patch", "DINOv2 + JEPA + CEM (完整：H=3, 空间补丁代价)", "dino_jepa", 3, "patch_l2"),
        ("dino_H1_patch", "DINOv2 + JEPA + CEM (消融：H=1, 空间补丁代价)", "dino_jepa", 1, "patch_l2"),
        ("dino_H3_mean", "DINOv2 + JEPA + CEM (消融：H=3, 全局均值代价)", "dino_jepa", 3, "mean_pool_l2"),
        ("dino_H1_mean", "DINOv2 + JEPA + CEM (消融：H=1, 全局均值代价)", "dino_jepa", 1, "mean_pool_l2"),
        ("randcnn_H3_patch", "Random-PatchCNN + JEPA + CEM (编码器对照)", "rand_jepa", 3, "patch_l2"),
        ("goal_bc", "Goal-Conditioned BC (离线监督基线)", "goal_bc", 1, "none"),
    ]
    cell_seed_succ = {k: [] for k, *_ in cell_specs}
    cell_seed_dist = {k: [] for k, *_ in cell_specs}
    cell_seed_ms = {k: [] for k, *_ in cell_specs}
    cell_total_succ = {k: 0 for k, *_ in cell_specs}
    cell_total_eval = {k: 0 for k, *_ in cell_specs}

    ol_dino_seeds, ol_rand_seeds = [], []

    print(f"\n  开始运行 {n_seeds} 个独立随机种子（每种子训练 DINOv2/RandCNN 预测器与 Goal-BC 并完成全部 6 格评估）：")
    for s in range(n_seeds):
        ts = time.time()
        dino_pred, dino_loss = train_predictor(dino_tr_eps, 384, 8, seed=100 + s, n_iters=pred_iters)
        rand_pred, rand_loss = train_predictor(rand_tr_eps, 128, 8, seed=200 + s, n_iters=pred_iters)
        bc_model = train_goal_bc(dino_tr_eps, 384, seed=300 + s, n_iters=bc_iters)

        ol_d = eval_open_loop(dino_pred, dino_te_eps, ctx_len=ctx_len, horizons=horizons)
        ol_r = eval_open_loop(rand_pred, rand_te_eps, ctx_len=ctx_len, horizons=horizons)
        ol_dino_seeds.append(ol_d)
        ol_rand_seeds.append(ol_r)

        for key, _label, mode, H, ctype in cell_specs:
            mdl = dino_pred if mode == "dino_jepa" else (rand_pred if mode == "rand_jepa" else bc_model)
            recs = eval_tasks_in_env(
                mode, tasks, dino_enc, rand_enc, model=mdl, horizon=H, cost_type=ctype,
                n_candidates=n_cand, n_iters=n_iters_cem, n_mpc_steps=n_mpc_steps,
                frame_skip=frame_skip, ctx_len=ctx_len, eval_seed_offset=s,
            )
            seed_dir = runs_dir / key / f"seed-{s}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            with open(seed_dir / "eval.jsonl", "w", encoding="utf-8") as f:
                for row in recs:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            sr = float(np.mean([r["success"] for r in recs]))
            fd = float(np.mean([r["final_dist"] for r in recs]))
            ms = float(np.mean([r["step_ms"] for r in recs]))
            cell_seed_succ[key].append(sr)
            cell_seed_dist[key].append(fd)
            cell_seed_ms[key].append(ms)
            cell_total_succ[key] += sum(int(r["success"]) for r in recs)
            cell_total_eval[key] += len(recs)

        print(
            f"    种子 {s}/{n_seeds - 1} 完成（{time.time() - ts:4.1f}s）| "
            f"DINOv2 k*={ol_d['k_star']} (ρ1={ol_d['rho'][0]:.3f}), 完整成功率={cell_seed_succ['dino_full_H3_patch'][-1]*100:5.1f}%, "
            f"RandCNN={cell_seed_succ['randcnn_H3_patch'][-1]*100:5.1f}%, Goal-BC={cell_seed_succ['goal_bc'][-1]*100:5.1f}%"
        )

    # 5. 汇总统计（均值 ± 95% t 区间、Wilson 区间、与基线差值区间、2×2 相互作用）
    summary = {}
    for key, label, *_ in cell_specs:
        sr_ci = mean_and_ci(cell_seed_succ[key])
        fd_ci = mean_and_ci(cell_seed_dist[key])
        w_lo, w_hi = wilson_interval(cell_total_succ[key], cell_total_eval[key])
        summary[key] = {
            "label": label,
            "seed_success_rates": cell_seed_succ[key],
            "seed_final_dists": cell_seed_dist[key],
            "success_mean": sr_ci["mean"],
            "success_ci_half": sr_ci["half"],
            "success_wilson": [w_lo, w_hi],
            "total_success": cell_total_succ[key],
            "total_eval": cell_total_eval[key],
            "final_dist_mean": fd_ci["mean"],
            "final_dist_ci_half": fd_ci["half"],
            "step_ms_mean": float(np.mean(cell_seed_ms[key])),
        }

    # 参照上下界
    ref_random = {
        "success_rate": float(np.mean([r["success"] for r in rand_recs])),
        "final_dist": float(np.mean([r["final_dist"] for r in rand_recs])),
        "wilson": list(wilson_interval(sum(int(r["success"]) for r in rand_recs), len(rand_recs))),
    }
    ref_oracle = {
        "success_rate": float(np.mean([r["success"] for r in oracle_recs])),
        "final_dist": float(np.mean([r["final_dist"] for r in oracle_recs])),
        "wilson": list(wilson_interval(sum(int(r["success"]) for r in oracle_recs), len(oracle_recs))),
    }

    # 关键差值检验
    diff_vs_randcnn = diff_ci(cell_seed_succ["randcnn_H3_patch"], cell_seed_succ["dino_full_H3_patch"])
    diff_vs_bc = diff_ci(cell_seed_succ["goal_bc"], cell_seed_succ["dino_full_H3_patch"])
    # 2×2 相互作用：A = 空间补丁代价 (0=mean_pool_l2, 1=patch_l2)，B = 多步视野 (0=H=1, 1=H=3)
    inter_succ = interaction_2x2(
        cell_seed_succ["dino_H1_mean"],
        cell_seed_succ["dino_H1_patch"],
        cell_seed_succ["dino_H3_mean"],
        cell_seed_succ["dino_full_H3_patch"],
    )
    inter_dist = interaction_2x2(
        cell_seed_dist["dino_H1_mean"],
        cell_seed_dist["dino_H1_patch"],
        cell_seed_dist["dino_H3_mean"],
        cell_seed_dist["dino_full_H3_patch"],
    )

    ol_summary = {
        "horizons": list(horizons),
        "dino_rho_mean": np.mean([x["rho"] for x in ol_dino_seeds], axis=0).tolist(),
        "dino_k_star_median": float(np.median([x["k_star"] for x in ol_dino_seeds])),
        "rand_rho_mean": np.mean([x["rho"] for x in ol_rand_seeds], axis=0).tolist(),
        "rand_k_star_median": float(np.median([x["k_star"] for x in ol_rand_seeds])),
    }

    # 打印最终报告表
    print("\n" + "=" * 68)
    print("毕业项目主结果与预注册消融汇总")
    print("=" * 68)
    md_lines = [
        "| 方法 | 成功率（均值 ± 95% t 半宽） | 汇总 Wilson 95% 区间 | 终点距离 (px) | 单步耗时 (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label, *_ in cell_specs:
        s = summary[key]
        line = (
            f"| {label} | {s['success_mean']*100:.1f}% ± {s['success_ci_half']*100:.1f}% | "
            f"[{s['success_wilson'][0]*100:.1f}%, {s['success_wilson'][1]*100:.1f}%] ({s['total_success']}/{s['total_eval']}) | "
            f"{s['final_dist_mean']:.1f} ± {s['final_dist_ci_half']:.1f} | {s['step_ms_mean']:.1f} |"
        )
        md_lines.append(line)
        print(" ", line)
    md_lines.append(f"| 参照：Oracle-CEM（上界） | {ref_oracle['success_rate']*100:.1f}% | [{ref_oracle['wilson'][0]*100:.1f}%, {ref_oracle['wilson'][1]*100:.1f}%] | {ref_oracle['final_dist']:.1f} | — |")
    md_lines.append(f"| 参照：随机动作（下界） | {ref_random['success_rate']*100:.1f}% | [{ref_random['wilson'][0]*100:.1f}%, {ref_random['wilson'][1]*100:.1f}%] | {ref_random['final_dist']:.1f} | — |")
    print(" ", md_lines[-2])
    print(" ", md_lines[-1])

    print(f"\n  DINOv2 vs Random-PatchCNN 成功率差值：{diff_vs_randcnn['delta']*100:+.1f}%（95% 区间 [{diff_vs_randcnn['lo']*100:+.1f}%, {diff_vs_randcnn['hi']*100:+.1f}%]）")
    print(f"  DINOv2 vs Goal-BC 成功率差值：        {diff_vs_bc['delta']*100:+.1f}%（95% 区间 [{diff_vs_bc['lo']*100:+.1f}%, {diff_vs_bc['hi']*100:+.1f}%]）")
    print(
        f"  2×2 相互作用（空间补丁代价 × 多步视野，终点距离 px）：{inter_dist['interaction']:+.1f} "
        f"（95% 区间 [{inter_dist['lo']:+.1f}, {inter_dist['hi']:+.1f}]）"
    )
    print(f"  开环可信视野中位数 k* (c=0.5)：DINOv2 = {ol_summary['dino_k_star_median']:.0f} 步，Random-PatchCNN = {ol_summary['rand_k_star_median']:.0f} 步")

    # 保存 Markdown 结果表与 results.json
    (OUT / "results-table.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    payload = {
        "total_seconds": time.time() - t0,
        "quick": quick,
        "n_episodes": len(dino_eps),
        "n_seeds": n_seeds,
        "n_tasks": n_tasks,
        "fingerprints": {"dinov2": fp_dino, "random_patch_cnn": fp_rand},
        "probe": {"dinov2": probe_dino, "random_patch_cnn": probe_rand},
        "open_loop": ol_summary,
        "summary": summary,
        "references": {"random": ref_random, "oracle_cem": ref_oracle},
        "comparisons": {"dino_vs_randcnn_succ": diff_vs_randcnn, "dino_vs_bc_succ": diff_vs_bc},
        "interaction_2x2": {"success_rate": inter_succ, "final_dist": inter_dist},
    }
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # 6. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    hs = ol_summary["horizons"]
    axes[0].plot(hs, ol_summary["dino_rho_mean"], "o-", linewidth=2, label=f"DINOv2 ViT-S/14 (k*={ol_summary['dino_k_star_median']:.0f})")
    axes[0].plot(hs, ol_summary["rand_rho_mean"], "s--", linewidth=2, label=f"Random-PatchCNN (k*={ol_summary['rand_k_star_median']:.0f})")
    axes[0].axhline(0.5, ls=":", color="tab:green", label="可信阈值 c = 0.5")
    axes[0].axhline(1.0, ls="--", color="gray", label="原样复制基线 ρ = 1.0")
    axes[0].set_title(f"表征空间开环相对误差 ρ(k)（n={n_seeds} 种子均值）")
    axes[0].set_xlabel("开环步数 k"); axes[0].set_ylabel("ρ = e_model / e_copy"); axes[0].legend(fontsize=8)

    bar_keys = ["dino_full_H3_patch", "dino_H1_patch", "dino_H3_mean", "randcnn_H3_patch", "goal_bc"]
    bar_labels = ["DINOv2\n(H=3, 补丁)", "DINOv2\n(H=1, 补丁)", "DINOv2\n(H=3, 均值)", "RandCNN\n(H=3, 补丁)", "Goal-BC\n(离线基线)"]
    means = [summary[k]["final_dist_mean"] for k in bar_keys]
    errs = [summary[k]["final_dist_ci_half"] for k in bar_keys]
    axes[1].bar(bar_labels, means, yerr=errs, capsize=4, color=["C0", "C0", "C1", "C3", "C2"], alpha=0.85)
    axes[1].axhline(ref_oracle["final_dist"], ls="--", color="black", label=f"Oracle-CEM ({ref_oracle['final_dist']:.1f}px)")
    axes[1].axhline(ref_random["final_dist"], ls=":", color="gray", label=f"随机动作 ({ref_random['final_dist']:.1f}px)")
    axes[1].axhline(45.0, ls="-.", color="tab:green", label="成功阈值 (45px)")
    axes[1].set_title(f"PushT 目标到达终点距离（越低越好，n={n_seeds} 种子 ± 95% t 区间）")
    axes[1].set_ylabel("终点距离 (px)"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, OUT / "capstone-results.png")

    print(f"\n📄 毕业项目结果已保存至 {OUT / 'results.json'}（总用时 {payload['total_seconds']:.1f}s）")


if __name__ == "__main__":
    main()
