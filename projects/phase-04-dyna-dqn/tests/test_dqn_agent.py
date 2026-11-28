"""测试 Double DQN 损失停止梯度、终止掩码与软更新。"""

import torch

from dqn_agent import DynaDQNAgent, QNetwork, double_dqn_loss


def test_double_dqn_stop_gradient_and_terminal_mask():
    torch.manual_seed(0)
    q_on = QNetwork(hidden=32)
    q_tg = QNetwork(hidden=32)
    s = torch.randn(6, 4)
    a = torch.tensor([0, 1, 0, 1, 0, 1])
    r = torch.ones(6)
    sn = torch.randn(6, 4)
    term = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])

    loss = double_dqn_loss(q_on, q_tg, s, a, r, sn, term, gamma=0.99)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in q_on.parameters())
    assert all(p.grad is None for p in q_tg.parameters()), "目标网络不应收到任何梯度"


def test_soft_update_interpolates_weights():
    agent = DynaDQNAgent(tau=0.5)
    with torch.no_grad():
        for p in agent.q_online.parameters():
            p.fill_(2.0)
        for p in agent.q_target.parameters():
            p.fill_(0.0)
    agent.soft_update_target()
    for p in agent.q_target.parameters():
        assert torch.allclose(p, torch.ones_like(p))
