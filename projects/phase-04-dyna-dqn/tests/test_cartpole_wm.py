"""测试动力学与终止世界模型能学会 CartPole 的单步转移与越界终止边界。"""

import numpy as np
import torch

from dyna_env_model import CartPoleWorldModel, ReplayBuffer, world_model_step_loss


def test_replay_buffer_stores_and_samples():
    buf = ReplayBuffer(capacity=10)
    rng = np.random.default_rng(0)
    for i in range(15):
        buf.add(np.full(4, i), i % 2, 1.0, np.full(4, i + 1), terminated=(i % 5 == 0))
    assert len(buf) == 10
    s, a, r, sn, term = buf.sample(8, rng)
    assert s.shape == (8, 4) and a.shape == (8,) and term.shape == (8,)
    assert torch.allclose(sn - s, torch.ones_like(s))


def test_world_model_learns_delta_and_termination():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    wm = CartPoleWorldModel(hidden=64)
    opt = torch.optim.Adam(wm.parameters(), lr=5e-3)

    # 构造合成规则：动作 1 让 x 增加 0.2，动作 0 让 x 减少 0.2；|x'| > 1.0 时终止
    states = torch.from_numpy(rng.uniform(-1.2, 1.2, size=(512, 4)).astype(np.float32))
    actions = torch.from_numpy(rng.integers(0, 2, size=512))
    step_dir = (actions.float() * 2.0 - 1.0) * 0.2
    next_states = states.clone()
    next_states[:, 0] += step_dir
    terminated = (next_states[:, 0].abs() > 1.0).float()
    rewards = torch.ones(512)

    for _ in range(250):
        loss, _ = world_model_step_loss(wm, states, actions, rewards, next_states, terminated)
        opt.zero_grad()
        loss.backward()
        opt.step()

    pred_next, pred_r, pred_term = wm.step(states, actions)
    assert (pred_next - next_states).abs().mean().item() < 0.03
    assert (pred_term == terminated).float().mean().item() > 0.94
