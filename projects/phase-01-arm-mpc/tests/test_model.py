"""集成模型：形状、成员之间互不影响、能学会一个简单的线性系统。"""

import torch

import reacher
from model import ProbabilisticEnsemble, train_ensemble


def make_model(n_models=3):
    torch.manual_seed(0)
    return ProbabilisticEnsemble(reacher.features, feature_dim=8, state_dim=4, n_models=n_models, hidden=32)


def test_output_shapes():
    model = make_model()
    states, actions = torch.randn(3, 7, 4), torch.randn(3, 7, 2)
    mean, std = model.predict(states, actions)
    assert mean.shape == (3, 7, 4) and std.shape == (3, 7, 4)
    assert (std > 0).all()


def test_members_are_independent():
    model = make_model()
    states, actions = torch.randn(3, 5, 4), torch.randn(3, 5, 2)
    before, _ = model.predict(states, actions)
    with torch.no_grad():
        model.layers[0].weight[0] += 1.0  # 只改成员 0
    after, _ = model.predict(states, actions)
    assert not torch.allclose(before[0], after[0])
    assert torch.allclose(before[1:], after[1:])


def test_members_differ_at_init():
    model = make_model()
    states, actions = torch.randn(1, 5, 4).expand(3, -1, -1), torch.randn(1, 5, 2).expand(3, -1, -1)
    mean, _ = model.predict(states, actions)
    assert not torch.allclose(mean[0], mean[1])


def test_learns_simple_dynamics():
    """s' = s + 0.1·(a1, a2, a1, a2)：训练后的预测误差应远小于"预测不变"的误差。"""
    torch.manual_seed(0)
    states = torch.randn(2000, 4)
    actions = torch.rand(2000, 2) * 2 - 1
    next_states = states + 0.1 * torch.cat([actions, actions], dim=-1)
    model = train_ensemble(make_model(), (states, actions, next_states), n_steps=500, device=torch.device("cpu"))
    s, a = states[:200].expand(3, -1, -1), actions[:200].expand(3, -1, -1)
    delta, _ = model.predict(s, a)
    error = (s + delta - next_states[:200]).abs().mean().item()
    baseline = (next_states[:200] - states[:200]).abs().mean().item()
    assert error < 0.2 * baseline
