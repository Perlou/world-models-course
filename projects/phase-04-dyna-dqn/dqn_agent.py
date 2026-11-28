"""Double DQN 智能体与 Dyna 后台模拟更新。"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dyna_env_model import N_ACTIONS, STATE_DIM


class QNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, states):
        return self.net(states)


def double_dqn_loss(q_online, q_target, s, a, r, s_next, terminated, gamma=0.99):
    """Double DQN 半梯度 Huber 损失：在线网络选动作，目标网络估价值，目标完全停止梯度。"""
    q_sa = q_online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        best_next_a = q_online(s_next).argmax(dim=1, keepdim=True)
        next_v = q_target(s_next).gather(1, best_next_a).squeeze(1)
        target = r + gamma * (1.0 - terminated) * next_v
    return F.smooth_l1_loss(q_sa, target)


class DynaDQNAgent:
    def __init__(self, lr_q=1e-3, lr_wm=2e-3, gamma=0.99, tau=0.02, device=torch.device("cpu")):
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.q_online = QNetwork().to(device)
        self.q_target = QNetwork().to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_opt = torch.optim.Adam(self.q_online.parameters(), lr=lr_q)
        self.lr_wm = lr_wm

    @torch.no_grad()
    def select_action(self, state_np, epsilon, rng):
        if rng.random() < epsilon:
            return int(rng.integers(0, N_ACTIONS))
        s = torch.from_numpy(np.asarray(state_np, dtype=np.float32)).unsqueeze(0).to(self.device)
        return int(self.q_online(s).argmax(dim=1).item())

    def soft_update_target(self):
        with torch.no_grad():
            for p_t, p_o in zip(self.q_target.parameters(), self.q_online.parameters()):
                p_t.data.mul_(1.0 - self.tau).add_(p_o.data, alpha=self.tau)

    def update_q_from_batch(self, batch):
        loss = double_dqn_loss(self.q_online, self.q_target, *batch, gamma=self.gamma)
        self.q_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), 10.0)
        self.q_opt.step()
        self.soft_update_target()
        return loss.item()
