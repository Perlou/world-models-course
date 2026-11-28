"""Phase 11 世界模型策略评估项目单元测试。"""

import torch

from wm_evaluator import PushTBC, PushTDP, PushTWMEvaluator, mmrv, pearson_r, spearman_rho


def test_consistency_metrics_match_concept_hand_calc():
    y = [0.20, 0.45, 0.50, 0.70, 0.90]
    yh_a = [0.30, 0.55, 0.50, 0.60, 0.80]
    yh_b = [0.10, 0.225, 0.25, 0.35, 0.45]
    yh_c = [0.20, 0.45, 0.50, 0.85, 0.75]
    yh_d = [0.20, 0.45, 0.50, 0.70, 0.15]

    assert abs(pearson_r(y, yh_a) - 0.970) < 0.005 and abs(mmrv(y, yh_a) - 0.02) < 1e-6
    assert abs(pearson_r(y, yh_b) - 1.000) < 1e-6 and abs(spearman_rho(y, yh_b) - 1.0) < 1e-6 and mmrv(y, yh_b) == 0.0
    assert abs(spearman_rho(y, yh_c) - 0.90) < 1e-6 and abs(mmrv(y, yh_c) - 0.08) < 1e-6
    assert abs(spearman_rho(y, yh_d) - 0.00) < 1e-6 and abs(mmrv(y, yh_d) - 0.49) < 1e-6


def test_bc_and_dp_shapes():
    obs = torch.randn(4, 2, 5)
    act = torch.randn(4, 8, 2)
    bc = PushTBC()
    dp = PushTDP()
    assert bc.predict_chunk(obs).shape == (4, 8, 2)
    loss = dp.loss(act, obs)
    assert torch.isfinite(loss)
    assert dp.predict_chunk(obs, n_steps=3).shape == (4, 8, 2)


def test_wm_evaluator_step_and_coverage_range():
    wm = PushTWMEvaluator()
    s = torch.randn(3, 5)
    a = torch.randn(3, 2)
    ns, cov = wm.step(s, a)
    assert ns.shape == (3, 5) and cov.shape == (3,)
    assert cov.min().item() >= 0.0 and cov.max().item() <= 1.0
