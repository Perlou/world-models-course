"""Phase 10 JEPA 规划器单元测试。"""

import torch

from jepa_planner import DINOWMPredictor, build_frame_causal_mask, jepa_feature_cost, plan_to_goal_image


def test_frame_causal_mask_block_structure():
    mask = build_frame_causal_mask(T=3, N=4)
    assert mask.shape == (12, 12)
    # 帧 0 内全可见（False），看帧 1、2 全部屏蔽（True）
    assert not mask[:4, :4].any()
    assert mask[:4, 4:].all()
    assert not mask[8:, :].any()


def test_dinowm_predictor_forward_and_rollout():
    pred = DINOWMPredictor(feat_dim=32, act_dim=8, num_patches=4, d_model=32, n_layers=1, n_heads=2, max_frames=6)
    z = torch.randn(2, 3, 4, 32)
    a = torch.randn(2, 3, 8)
    out = pred(z, a)
    assert out.shape == (2, 3, 4, 32)
    # 残差零初始化：首次前向等于输入
    assert torch.allclose(out, z, atol=1e-5)

    a_roll = torch.randn(2, 5, 8)
    r_out = pred.rollout(z, a_roll)
    assert r_out.shape == (2, 3, 4, 32)


def test_jepa_feature_cost_zero_on_identical():
    z = torch.randn(4, 3, 8, 16)
    zg = z[0, -1]
    costs = jepa_feature_cost(z, zg)
    assert costs.shape == (4,) and costs[0].item() < 1e-6 and costs[1].item() > 0.0


def test_plan_to_goal_image_smoke():
    pred = DINOWMPredictor(feat_dim=16, act_dim=8, num_patches=4, d_model=16, n_layers=1, n_heads=2, max_frames=6)
    z_hist = torch.randn(1, 2, 4, 16)
    a_prev = torch.zeros(1, 1, 8)
    z_goal = torch.randn(4, 16)
    best_seq, ms = plan_to_goal_image(pred, z_hist, a_prev, z_goal, horizon=2, n_samples=6, n_iters=2, frame_skip=4)
    assert best_seq.shape == (2, 2) and ms >= 0.0
