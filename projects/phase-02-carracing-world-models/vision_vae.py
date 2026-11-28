"""V（视觉模型）：CarRacing-v3 画面预处理与卷积 VAE。"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import numpy as np
import torch
import torch.nn as nn

Z_DIM = 32
CROP_ROWS = 84
IMAGE_SIZE = 64
ZOOM_FRAMES = 50


def preprocess_frame(frame_u8):
    """[96, 96, 3] uint8 → [64, 64, 3] uint8（裁掉底部 12 行仪表盘并缩放）。"""
    return cv2.resize(frame_u8[:CROP_ROWS], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def to_tensor_obs(frames_u8, device=torch.device("cpu")):
    """[..., 64, 64, 3] uint8 → [..., 3, 64, 64] float32 ∈ [0, 1]。"""
    t = torch.as_tensor(frames_u8, device=device)
    if t.ndim == 3:
        return t.permute(2, 0, 1).float() / 255.0
    return t.permute(0, 3, 1, 2).float() / 255.0


class ConvVAE(nn.Module):
    """Ha & Schmidhuber (2018) 卷积 VAE（4,348,547 参数）。"""

    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(1024, z_dim)
        self.fc_logvar = nn.Linear(1024, z_dim)
        self.fc_decode = nn.Linear(z_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid(),
        )

    def encode(self, obs):
        h = self.encoder(obs)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.decoder(self.fc_decode(z).view(-1, 1024, 1, 1))

    def forward(self, obs):
        mu, logvar = self.encode(obs)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def vae_loss(obs, recon, mu, logvar):
    """对像素与潜变量维分别求和，再对批次求平均。"""
    recon_loss = (obs - recon).pow(2).sum(dim=(1, 2, 3)).mean()
    kl_loss = (0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)).sum(dim=1).mean()
    return recon_loss, kl_loss
