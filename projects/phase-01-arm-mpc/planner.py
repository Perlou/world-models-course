"""
CEM 规划器 + 轨迹采样
=====================

和讲义 06、08 的规划器是同一个，动作从 1 维变成 2 维，其余不变。

    候选动作序列   [N, H, A]
    粒子的状态     [B, N × k, S]     每个成员带 k 个粒子，一条序列共 B × k 个粒子（TS∞）
    候选的分数     [N]               对一条序列的全部粒子的回报取平均
"""

import torch


@torch.no_grad()
def score_sequences(model, reward_fn, state, actions, particles_per_member, generator):
    """用轨迹采样（TS∞）给候选动作序列打分。

    参数：
        model:      概率集成，提供 predict(states, actions) → (变化量均值, 变化量标准差)
        reward_fn:  (下一状态 [..., S], 动作 [..., A]) → 奖励 [...]
        state:      当前的真实状态 [S]
        actions:    候选动作序列 [N, H, A]
    返回：
        scores [N]
    """
    n, horizon, action_dim = actions.shape
    k, n_models = particles_per_member, model.n_models
    particle_actions = actions.repeat_interleave(k, dim=0)                # [N × k, H, A]
    states = state.expand(n_models, n * k, state.shape[-1])               # [B, N × k, S]
    returns = torch.zeros(n_models, n * k, device=actions.device)
    for h in range(horizon):
        a = particle_actions[:, h].expand(n_models, -1, -1)
        delta_mean, delta_std = model.predict(states, a)
        noise = torch.randn(states.shape, generator=generator, device=states.device)
        states = states + delta_mean + delta_std * noise
        returns += reward_fn(states, a)                                   # 奖励用执行动作之后的状态算
    return returns.view(n_models, n, k).mean(dim=(0, 2))


def cem_plan(score_fn, mean, generator, action_low=-1.0, action_high=1.0,
             n_candidates=200, n_elites=20, n_iters=5, init_std=0.5, min_std=0.05, smoothing=0.1):
    """交叉熵方法。mean [H, A] 是采样分布的初始均值，返回优化后的均值。"""
    std = torch.full_like(mean, init_std)
    for _ in range(n_iters):
        noise = torch.randn(n_candidates, *mean.shape, generator=generator, device=mean.device)
        actions = (mean + std * noise).clamp(action_low, action_high)
        elites = actions[score_fn(actions).topk(n_elites).indices]
        mean = smoothing * mean + (1 - smoothing) * elites.mean(dim=0)
        std = elites.std(dim=0, correction=0).clamp_min(min_std)
    return mean


def shift_plan(mean):
    """热启动：把规划结果向前平移一格，末尾补 0。"""
    return torch.cat([mean[1:], torch.zeros_like(mean[:1])])
