"""经验回放缓冲区与学出来的 CartPole 动力学 + 终止世界模型。"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM = 4
N_ACTIONS = 2


class ReplayBuffer:
    """环形经验回放缓冲区（严格只存 terminated，不把时间截断 truncated 当作终止）。"""

    def __init__(self, capacity=20_000, state_dim=STATE_DIM):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s_next, terminated):
        i = self.ptr
        self.states[i] = s
        self.actions[i] = int(a)
        self.rewards[i] = float(r)
        self.next_states[i] = s_next
        self.terminated[i] = float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size, rng, device=torch.device("cpu")):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_states[idx]).to(device),
            torch.from_numpy(self.terminated[idx]).to(device),
        )

    def sample_states(self, batch_size, rng, device=torch.device("cpu")):
        idx = rng.integers(0, self.size, size=batch_size)
        return torch.from_numpy(self.states[idx]).to(device)


class CartPoleWorldModel(nn.Module):
    """离散动作动力学模型：输入 (s, one_hot(a))，预测状态变化量 Δs、奖励 r̂ 和终止 logit。

    CartPole-v1 的关键信号在"什么时候杆子倒下终止"：如果世界模型只预测 s' 而不预测 terminated，
    模拟出来的越界状态仍会加上 γ max Q(s')，导致 Q 值发散。
    """

    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS, hidden=128):
        super().__init__()
        self.n_actions = n_actions
        self.trunk = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.delta_head = nn.Linear(hidden, state_dim)
        self.reward_head = nn.Linear(hidden, 1)
        self.done_head = nn.Linear(hidden, 1)

    def forward(self, states, actions):
        a_oh = F.one_hot(actions.long(), num_classes=self.n_actions).float()
        h = self.trunk(torch.cat([states, a_oh], dim=-1))
        return self.delta_head(h), self.reward_head(h).squeeze(-1), self.done_head(h).squeeze(-1)

    @torch.no_grad()
    def step(self, states, actions):
        """从真实状态单步想象：(s, a) → (ŝ', r̂, d̂ ∈ {0, 1})。"""
        delta, rew, done_logit = self.forward(states, actions)
        next_states = states + delta
        terminated = (torch.sigmoid(done_logit) > 0.5).float()
        return next_states, rew, terminated


def world_model_step_loss(wm, s, a, r, s_next, terminated):
    """动力学 MSE + 奖励 MSE + 加权终止二分类交叉熵（倒杆终止步占比小，给正样本加权）。"""
    delta_pred, rew_pred, done_logit = wm(s, a)
    delta_loss = F.mse_loss(delta_pred, s_next - s)
    rew_loss = 0.1 * F.mse_loss(rew_pred, r)
    # 终止帧通常只占 ~2–5%，pos_weight=10 防止模型把所有状态都判为未终止
    pos_w = torch.tensor(10.0, device=s.device)
    done_loss = F.binary_cross_entropy_with_logits(done_logit, terminated, pos_weight=pos_w)
    return delta_loss + rew_loss + 0.5 * done_loss, dict(
        delta_rmse=delta_loss.detach().sqrt().item(),
        done_bce=done_loss.item(),
    )
