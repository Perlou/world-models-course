"""测试隐空间 CEM 规划器能在可控的玩具隐空间模型上找到最优动作。"""

import torch
import torch.nn as nn

from latent_planner import latent_cem_plan


class ToyRSSM:
    """玩具隐空间动力学：h' = 0，z' = z + a（1 维），无噪声，奖励 = −(z − 1.5)²。"""

    def imagine(self, h, z, actions, sample=True, generator=None):
        zs = []
        cur = z
        for k in range(actions.shape[1]):
            cur = cur + actions[:, k]
            zs.append(torch.cat([h, cur], dim=-1))
        return torch.stack(zs, dim=1)


class ToyWorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rssm = ToyRSSM()
        self.reward_head = lambda feats: -((feats[..., -1] - 1.5) ** 2)


def test_latent_cem_reaches_target_and_respects_bounds():
    model = ToyWorldModel()
    h = torch.zeros(1, 2)
    z = torch.zeros(1, 1)
    gen = torch.Generator().manual_seed(0)
    plan = latent_cem_plan(model, h, z, horizon=4, n_candidates=200, n_elites=20, n_iters=6, generator=gen)
    assert plan.shape == (4, 1)
    assert plan.min() >= -1.0 and plan.max() <= 1.0
    # 从 z=0 出发 4 步到达 1.5，第一步应为明显正动作，4 步累积和应接近 1.5
    assert plan[0, 0].item() > 0.5
    assert abs(plan.sum().item() - 1.5) < 0.25
