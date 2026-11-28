import math

import numpy as np
import torch

from benchmark_protocol import get_task_protocol
from muzero_tictactoe import MuZeroTTTNet, TTTBoard, muzero_ttt_mcts
from tdmpc2_core import MiniTDMPC2, run_tdmpc2_on_benchmark, td2_simnorm, td2_two_hot, td2_two_hot_inv


def test_simnorm_and_twohot_bounds():
    x = torch.randn(4, 64) * 10.0
    z = td2_simnorm(x, v_dim=8)
    norms_sq = z.pow(2).sum(dim=-1)
    assert torch.all(norms_sq >= 8.0 / 8.0 - 1e-4) and torch.all(norms_sq <= 8.0 + 1e-4)

    bins = torch.linspace(-10.0, 10.0, 101)
    vals = torch.tensor([-50.0, 0.0, 20.0, 500.0])
    th = td2_two_hot(vals, bins)
    assert torch.allclose(th.sum(dim=-1), torch.ones_like(vals), atol=1e-5)
    logits = torch.log(th.clamp_min(1e-12))
    dec = td2_two_hot_inv(logits, bins)
    assert torch.allclose(dec, vals, rtol=1e-2, atol=1e-2)


def test_tdmpc2_plan_and_benchmark_smoke():
    cfg = get_task_protocol("Pendulum-v1", quick=True)
    cfg["total_steps"] = 240
    cfg["eval_every"] = 240
    st, rt, dt, _ = run_tdmpc2_on_benchmark("Pendulum-v1", cfg, seed=0)
    assert len(st) >= 1 and np.isfinite(rt[-1]) and dt >= 0.0


def test_muzero_ttt_mcts_smoke():
    net = MuZeroTTTNet(latent_dim=16, hidden=32)
    b = TTTBoard()
    pi = muzero_ttt_mcts(net, b, n_sims=8, add_noise=False)
    assert pi.shape == (9,) and abs(pi.sum() - 1.0) < 1e-5
