"""规划器：CEM 能找到简单目标的最优解；轨迹采样的形状和排序正确。"""

import torch

from planner import cem_plan, score_sequences, shift_plan


def test_cem_finds_optimum():
    """最大化 −‖A − 0.5‖²，最优解是所有动作都取 0.5。"""
    generator = torch.Generator().manual_seed(0)
    score = lambda actions: -((actions - 0.5) ** 2).sum(dim=(1, 2))
    mean = cem_plan(score, torch.zeros(6, 2), generator, n_iters=10)
    assert (mean - 0.5).abs().max() < 0.1


def test_cem_respects_action_bounds():
    """最优解在边界之外时，结果应该贴在边界上。"""
    generator = torch.Generator().manual_seed(0)
    score = lambda actions: actions.sum(dim=(1, 2))
    mean = cem_plan(score, torch.zeros(4, 2), generator, n_iters=10)
    assert mean.max() <= 1.0 and mean.min() > 0.8


class FakeEnsemble:
    """假的集成：状态变化量就是动作本身（2 维状态、2 维动作），没有噪声。"""

    n_models = 3

    def predict(self, states, actions):
        return actions, torch.zeros_like(states)


def test_score_sequences_shape_and_order():
    generator = torch.Generator().manual_seed(0)
    reward_fn = lambda states, actions: -states.norm(dim=-1)  # 离原点越近越好
    state = torch.tensor([1.0, 1.0])
    good = torch.full((1, 5, 2), -0.2)   # 朝原点走
    bad = torch.full((1, 5, 2), 0.2)     # 远离原点
    scores = score_sequences(FakeEnsemble(), reward_fn, state, torch.cat([good, bad]), 2, generator)
    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_shift_plan():
    mean = torch.arange(8.0).view(4, 2)
    shifted = shift_plan(mean)
    assert torch.equal(shifted[:3], mean[1:])
    assert torch.equal(shifted[3], torch.zeros(2))
