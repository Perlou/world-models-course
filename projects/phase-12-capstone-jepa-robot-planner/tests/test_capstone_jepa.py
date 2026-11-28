"""Phase 12 毕业项目单元测试：覆盖数据切分/指纹、统计区间、2×2 相互作用、探针与规划器。"""

import numpy as np
import torch

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


def test_split_by_episode_disjoint_and_complete():
    sp = split_by_episode(20, seed=42)
    all_ids = np.concatenate([sp["train"], sp["val"], sp["test"]])
    assert len(all_ids) == 20 and len(set(all_ids.tolist())) == 20
    assert set(sp["train"]).isdisjoint(set(sp["val"]))
    assert set(sp["train"]).isdisjoint(set(sp["test"]))


def test_fingerprint_sensitive_to_data_changes():
    ep1 = [(np.zeros((5, 4, 8), dtype=np.float32), np.zeros((5, 2), dtype=np.float32), np.zeros((5, 2), dtype=np.float32))]
    ep2 = [(np.zeros((5, 4, 8), dtype=np.float32), np.ones((5, 2), dtype=np.float32) * 1e-3, np.zeros((5, 2), dtype=np.float32))]
    assert compute_episodes_fingerprint(ep1) == compute_episodes_fingerprint(ep1)
    assert compute_episodes_fingerprint(ep1) != compute_episodes_fingerprint(ep2)


def test_statistics_match_hand_calcs():
    a = [0.62, 0.70, 0.58, 0.74, 0.66]
    b = [0.68, 0.80, 0.70, 0.82, 0.70]
    r = mean_and_ci(a)
    assert abs(r["mean"] - 0.66) < 1e-9 and abs(r["lo"] - 0.581) < 3e-3
    d = diff_ci(a, b)
    assert abs(d["delta"] - 0.08) < 1e-9 and d["lo"] < 0.0 < d["hi"]
    w_lo, w_hi = wilson_interval(20, 20)
    assert abs(w_lo - 0.839) < 3e-3 and abs(w_hi - 1.0) < 1e-9
    inter = interaction_2x2([0.27, 0.33], [0.32, 0.38], [0.47, 0.53], [0.72, 0.78])
    assert abs(inter["interaction"] - 0.20) < 1e-9 and abs(inter["se"] - 0.06) < 1e-9


def test_credible_horizon_definition():
    assert credible_horizon([0.01, 0.03, 0.07, 0.18, 0.33], [0.03, 0.12, 0.20, 0.29, 0.34], threshold=0.5, ks=[1, 5, 10, 20, 30]) == 10


def test_random_patch_encoder_probe_and_planner_smoke():
    enc = RandomPatchCNNEncoder(feat_dim=16, seed=7)
    imgs = np.random.randint(0, 255, size=(6, 32, 32, 3), dtype=np.uint8)
    feats = enc.encode_images(imgs, img_size=32)
    assert feats.shape == (6, 64, 16)

    rng = np.random.default_rng(0)
    f_tr = rng.standard_normal((120, 64, 8)).astype(np.float32)
    g_tr = f_tr.reshape(120, 8, 8, 8)[:, :4, :4, :2].mean(axis=(1, 2))
    s_tr = (g_tr * 50.0 + 250.0).astype(np.float32)
    f_te = rng.standard_normal((30, 64, 8)).astype(np.float32)
    g_te = f_te.reshape(30, 8, 8, 8)[:, :4, :4, :2].mean(axis=(1, 2))
    s_te = (g_te * 50.0 + 250.0).astype(np.float32)
    res = fit_and_eval_state_probe(f_tr, s_tr, f_te, s_te)
    assert res["r2"] > 0.9

    pred = CapstoneJEPAPredictor(feat_dim=16, act_dim=8, num_patches=64, d_model=16, n_layers=1, n_heads=2, max_frames=6)
    z_hist = torch.randn(1, 2, 64, 16)
    a_prev = torch.zeros(1, 1, 8)
    z_goal = torch.randn(64, 16)
    best, ms = plan_jepa_cem(pred, z_hist, a_prev, z_goal, horizon=2, n_candidates=6, n_iters=2, cost_type="patch_l2")
    assert best.shape == (2, 2) and ms >= 0.0
    best_m, _ = plan_jepa_cem(pred, z_hist, a_prev, z_goal, horizon=2, n_candidates=6, n_iters=2, cost_type="mean_pool_l2")
    assert best_m.shape == (2, 2)

    bc = GoalConditionedBC(feat_dim=16, hidden=32)
    out = bc(z_hist[0, 0:1], z_goal.unsqueeze(0))
    assert out.shape == (1, 2)
