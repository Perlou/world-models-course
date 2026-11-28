"""测试 RSSM 形状、高斯 KL 闭式解、先验/后验一致性与梯度隔离。"""

import torch
from torch.distributions import Normal, kl_divergence

from rssm_model import H_DIM, WorldModel, Z_DIM, gaussian_kl, world_model_loss


def test_gaussian_kl_matches_pytorch():
    gen = torch.Generator().manual_seed(0)
    qm = torch.randn(5, 8, Z_DIM, generator=gen)
    qs = torch.rand(5, 8, Z_DIM, generator=gen) + 0.1
    pm = torch.randn(5, 8, Z_DIM, generator=gen)
    ps = torch.rand(5, 8, Z_DIM, generator=gen) + 0.1
    ours = gaussian_kl(qm, qs, pm, ps)
    ref = kl_divergence(Normal(qm, qs), Normal(pm, ps)).sum(dim=-1)
    assert torch.allclose(ours, ref, atol=1e-4)
    assert torch.allclose(gaussian_kl(qm, qs, qm, qs), torch.zeros(5, 8), atol=1e-6)


def test_observe_and_imagine_consistency():
    torch.manual_seed(0)
    wm = WorldModel()
    obs = torch.randn(2, 6, 3, 64, 64) * 0.2
    act = torch.randn(2, 6, 1)
    embeds = wm.encode_seq(obs)
    out = wm.rssm.observe(embeds, act)
    assert out["features"].shape == (2, 6, H_DIM + Z_DIM)

    # 从第 2 步的后验状态走一步 imagine_step，算出的先验均值必须与 observe 第 3 步完全相同
    _, _, (pm_imag, ps_imag) = wm.rssm.imagine_step(out["h"][:, 2], out["z"][:, 2], act[:, 3], sample=False)
    assert torch.allclose(pm_imag, out["prior_mean"][:, 3], atol=1e-6)
    assert torch.allclose(ps_imag, out["prior_std"][:, 3], atol=1e-6)


def test_prior_only_receives_kl_gradient():
    torch.manual_seed(0)
    wm = WorldModel()
    obs = torch.randn(2, 4, 3, 64, 64) * 0.2
    act = torch.randn(2, 4, 1)
    rew = torch.randn(2, 4)

    # beta=0 时没有 KL 项，先验网络不应收到任何梯度
    loss_no_kl, _ = world_model_loss(wm, obs, act, rew, beta=0.0, free_nats=0.0)
    wm.zero_grad()
    loss_no_kl.backward()
    assert all(p.grad is None or p.grad.abs().max() == 0 for p in wm.rssm.prior_net.parameters())

    # beta=1 且无免费额度截断时，先验网络收到非零梯度
    loss_with_kl, _ = world_model_loss(wm, obs, act, rew, beta=1.0, free_nats=0.0)
    wm.zero_grad()
    loss_with_kl.backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in wm.rssm.prior_net.parameters())
