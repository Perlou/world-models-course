import numpy as np
import torch

from benchmark_algos import BenchProbEnsemble, BenchReplayBuffer, run_mbpo, run_pets, run_ppo, run_random, run_sac
from benchmark_protocol import get_task_protocol, is_terminal_state, steps_to_threshold


def test_hopper_termination_rule():
    healthy = torch.tensor([[1.2, 0.05, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    fallen_z = torch.tensor([[0.5, 0.05, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    fallen_ang = torch.tensor([[1.2, 0.35, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    assert not bool(is_terminal_state("Hopper-v5", healthy)[0])
    assert bool(is_terminal_state("Hopper-v5", fallen_z)[0])
    assert bool(is_terminal_state("Hopper-v5", fallen_ang)[0])
    assert not bool(is_terminal_state("Pendulum-v1", torch.zeros(1, 3))[0])


def test_steps_to_threshold():
    assert steps_to_threshold([500, 1000, 1500], [-800, -240, -150], -250) == 1000
    assert steps_to_threshold([500, 1000], [-800, -400], -250) is None


def test_ensemble_fit_and_sample():
    buf = BenchReplayBuffer(obs_dim=3, act_dim=1, capacity=100)
    for _ in range(40):
        o = np.random.randn(3).astype(np.float32)
        a = np.random.uniform(-1, 1, size=1).astype(np.float32)
        buf.add(o, a, -1.0, o + 0.05, False)
    ens = BenchProbEnsemble(obs_dim=3, act_dim=1, n_ensemble=3, hidden=32)
    ens.fit(buf, epochs=2, batch_size=16)
    nxt, rew, term = ens.sample_step(torch.zeros(5, 3), torch.zeros(5, 1), "Pendulum-v1")
    assert nxt.shape == (5, 3) and rew.shape == (5, 1) and term.shape == (5,)
    assert torch.isfinite(nxt).all() and torch.isfinite(rew).all()


def test_all_five_algos_smoke_pendulum():
    cfg = get_task_protocol("Pendulum-v1", quick=True)
    cfg["total_steps"] = 240
    cfg["eval_every"] = 240
    for fn in (run_random, run_ppo, run_sac, run_mbpo, run_pets):
        st, ret, dt, _ = fn("Pendulum-v1", cfg, seed=0)
        assert len(st) == len(ret) >= 1
        assert np.isfinite(ret[-1]) and dt >= 0.0
