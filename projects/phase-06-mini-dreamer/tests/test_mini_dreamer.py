import numpy as np
import torch

from benchmark_protocol import get_task_protocol
from mini_dreamer import MiniDreamerV3Agent, MiniTwoHotHead, run_dreamer_on_benchmark, symexp, symlog, twohot_encode


def test_symlog_twohot_exactness():
    xs = torch.tensor([-50.0, -1.0, 0.0, 3.5, 1000.0])
    assert torch.allclose(symexp(symlog(xs)), xs, rtol=1e-4, atol=1e-4)
    head = MiniTwoHotHead(in_dim=8)
    bins = head.bins
    th = twohot_encode(xs, bins)
    assert torch.allclose(th.sum(dim=-1), torch.ones_like(xs), atol=1e-5)
    assert torch.allclose((th * bins).sum(dim=-1), xs, rtol=1e-4, atol=1e-4)
    # 零初始化末层确保初始输出为 0
    assert torch.allclose(head.predict(head(torch.randn(4, 8))), torch.zeros(4), atol=1e-5)


def test_mini_dreamer_train_step_and_act():
    agent = MiniDreamerV3Agent(obs_dim=3, act_dim=1, horizon=4, hidden=32)
    batch = (
        torch.randn(4, 6, 3),
        torch.randn(4, 6, 1).clamp(-1, 1),
        torch.randn(4, 6),
        torch.ones(4, 6),
        torch.zeros(4, 6),
    )
    wm_l, a_l, c_l = agent.train_step(batch)
    assert all(np.isfinite(x) for x in (wm_l, a_l, c_l))
    st = agent.init_state()
    act, st2 = agent.act(np.zeros(3, dtype=np.float32), st, deterministic=True)
    assert act.shape == (1,) and np.all(np.abs(act) <= 1.0) and st2[0].shape == (1, 32)


def test_mini_dreamer_benchmark_smoke():
    cfg = get_task_protocol("Pendulum-v1", quick=True)
    cfg["total_steps"] = 240
    cfg["eval_every"] = 240
    st, rt, dt, _ = run_dreamer_on_benchmark("Pendulum-v1", cfg, seed=0)
    assert len(st) >= 1 and np.isfinite(rt[-1]) and dt >= 0.0
