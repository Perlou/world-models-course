"""Phase 9 可玩神经仿真器单元测试。"""

import torch

from neural_simulator import PlayableSimulatorSession, PushTDiffusionWM, detail_errors, edm_coeffs, karras_schedule


def test_edm_coeffs_and_schedule():
    ci, cs, co, _ = edm_coeffs(torch.tensor([0.5]), sigma_data=0.5)
    assert abs(cs.item() - 0.5) < 1e-5
    assert abs(co.item() - (0.25 / (0.5 ** 0.5))) < 1e-4
    sched = karras_schedule(3, sigma_min=0.002, sigma_max=5.0, rho=7.0)
    assert sched.shape == (4,) and sched[-1].item() == 0.0 and sched[0].item() > sched[1].item() > sched[2].item()


def test_pusht_diffusion_wm_forward_and_loss():
    model = PushTDiffusionWM(ctx_len=3, act_dim=2, base_ch=16, cond_dim=32)
    ctx = torch.randn(2, 3, 3, 32, 32)
    act = torch.randn(2, 3, 2)
    tgt = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    loss = model.edm_loss(tgt, ctx, act, max_ctx_noise=0.2)
    assert torch.isfinite(loss) and loss.item() > 0.0
    nxt = model.sample_next_frame(ctx, act, n_steps=2)
    assert nxt.shape == (2, 3, 32, 32) and nxt.min().item() >= -1.0 and nxt.max().item() <= 1.0


def test_playable_session_stepping():
    model = PushTDiffusionWM(ctx_len=3, act_dim=2, base_ch=16, cond_dim=32)
    sess = PlayableSimulatorSession(model, torch.zeros(1, 3, 3, 32, 32), torch.zeros(1, 3, 2), n_steps=2)
    frame, dt_ms = sess.step_key("d")
    assert frame.shape == (3, 32, 32) and dt_ms >= 0.0
    assert abs(sess.cursor[0] - 0.15) < 1e-5


def test_detail_errors_zero_on_identical():
    img = torch.zeros(2, 3, 32, 32)
    img[:, 2, 10:14, 10:14] = 0.8
    pe, fe = detail_errors(img, img)
    assert pe == 0.0 and fe == 0.0
