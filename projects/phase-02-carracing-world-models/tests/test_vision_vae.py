"""测试 V（ConvVAE）形状、参数量、预处理与损失尺度。"""

import numpy as np
import torch

from vision_vae import ConvVAE, preprocess_frame, to_tensor_obs, vae_loss


def test_preprocess_and_tensor_range():
    raw = np.full((96, 96, 3), 255, dtype=np.uint8)
    raw[84:] = 0  # 底部 12 行仪表盘是黑色，裁剪后应全白
    small = preprocess_frame(raw)
    assert small.shape == (64, 64, 3) and small.dtype == np.uint8
    assert small.min() == 255
    t = to_tensor_obs(small)
    assert t.shape == (3, 64, 64) and torch.allclose(t, torch.ones_like(t))


def test_conv_vae_parameter_count_and_shapes():
    vae = ConvVAE()
    assert sum(p.numel() for p in vae.parameters()) == 4_348_547
    x = torch.rand(3, 3, 64, 64)
    recon, mu, logvar = vae(x)
    assert recon.shape == (3, 3, 64, 64) and mu.shape == (3, 32) and logvar.shape == (3, 32)
    r_loss, k_loss = vae_loss(x, recon, torch.zeros_like(mu), torch.zeros_like(logvar))
    assert abs(k_loss.item()) < 1e-6 and r_loss.item() > 0.0
