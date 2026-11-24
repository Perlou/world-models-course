"""
概率集成动力学模型
==================

和讲义 07、08 的模型是同一个东西，只是把状态和动作的维度做成了参数。

    输入   (cos q, sin q, q̇, a)            [B, N, 8]
    输出   状态变化量 Δs 的均值和对数方差      [B, N, 4] × 2

B 是集成的成员数。所有成员的权重堆在一个多一维的张量里，一次前向计算全部算完。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleLinear(nn.Module):
    """B 个线性层并排放：输入 [B, N, in] → 输出 [B, N, out]。"""

    def __init__(self, n_models, in_dim, out_dim):
        super().__init__()
        bound = 1.0 / np.sqrt(in_dim)
        self.weight = nn.Parameter(torch.empty(n_models, in_dim, out_dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(n_models, 1, out_dim).uniform_(-bound, bound))

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class ProbabilisticEnsemble(nn.Module):
    """概率集成。feature_fn 把 (状态, 动作) 变成网络的输入特征。"""

    def __init__(self, feature_fn, feature_dim, state_dim, n_models=5, hidden=200):
        super().__init__()
        self.feature_fn = feature_fn
        self.n_models, self.state_dim = n_models, state_dim
        self.layers = nn.ModuleList([
            EnsembleLinear(n_models, feature_dim, hidden),
            EnsembleLinear(n_models, hidden, hidden),
            EnsembleLinear(n_models, hidden, hidden),
            EnsembleLinear(n_models, hidden, 2 * state_dim),
        ])
        self.max_logvar = nn.Parameter(torch.full((state_dim,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((state_dim,), -10.0))
        self.register_buffer("x_mean", torch.zeros(feature_dim))
        self.register_buffer("x_std", torch.ones(feature_dim))
        self.register_buffer("y_mean", torch.zeros(state_dim))
        self.register_buffer("y_std", torch.ones(state_dim))

    def fit_normalizer(self, states, actions, next_states):
        x, y = self.feature_fn(states, actions), next_states - states
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp_min(1e-6))
        self.y_mean.copy_(y.mean(0))
        self.y_std.copy_(y.std(0).clamp_min(1e-6))

    def forward(self, states, actions):
        """标准化空间里的均值和对数方差。states [B, N, S]，actions [B, N, A] → 两个 [B, N, S]。"""
        h = (self.feature_fn(states, actions) - self.x_mean) / self.x_std
        for layer in self.layers[:-1]:
            h = F.silu(layer(h))
        out = self.layers[-1](h)
        mean, logvar = out[..., :self.state_dim], out[..., self.state_dim:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def nll_loss(self, states, actions, next_states):
        """高斯负对数似然，外加对数方差上下界的轻微正则。"""
        mean, logvar = self(states, actions)
        target = (next_states - states - self.y_mean) / self.y_std
        nll = 0.5 * ((mean - target) ** 2 * torch.exp(-logvar) + logvar).mean()
        return nll + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())

    @torch.no_grad()
    def predict(self, states, actions):
        """物理单位下，状态变化量的均值和标准差。"""
        mean, logvar = self(states, actions)
        return mean * self.y_std + self.y_mean, torch.exp(0.5 * logvar) * self.y_std


def train_ensemble(model, data, n_steps, device, seed=0, batch_size=256, lr=1e-3):
    """在转移数据 (s, a, s') 上训练。每个成员用自助采样得到的数据。

    data 是三个 CPU 张量。训练完模型留在 device 上。
    """
    model.to(device)
    states, actions, next_states = [d.to(device) for d in data]
    model.fit_normalizer(states, actions, next_states)
    n = states.shape[0]
    generator = torch.Generator().manual_seed(seed)
    bootstrap = torch.randint(0, n, (model.n_models, n), generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    for _ in range(n_steps):
        pick = torch.randint(0, n, (model.n_models, batch_size), generator=generator)
        idx = torch.gather(bootstrap, 1, pick).to(device)
        loss = model.nll_loss(states[idx], actions[idx], next_states[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model
