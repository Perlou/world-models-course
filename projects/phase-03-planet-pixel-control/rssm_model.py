"""RSSM 世界模型与序列 ELBO 损失。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixel_env import ACTION_DIM, IMAGE_CHANNELS, IMAGE_SIZE

EMBED_DIM = 1024
H_DIM = 200
Z_DIM = 30
HIDDEN_DIM = 200
MIN_STD = 0.1
FREE_NATS = 3.0


def gaussian_kl(post_mean, post_std, prior_mean, prior_std):
    """对角高斯 KL( N(μ_q, σ_q²) ‖ N(μ_p, σ_p²) )，对最后一维求和。"""
    var_ratio = (post_std / prior_std).pow(2)
    mean_shift = ((post_mean - prior_mean) / prior_std).pow(2)
    return (torch.log(prior_std / post_std) + 0.5 * (var_ratio + mean_shift - 1.0)).sum(dim=-1)


class ConvEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )

    def forward(self, obs):
        return self.conv(obs)


class ConvDecoder(nn.Module):
    def __init__(self, state_dim=H_DIM + Z_DIM):
        super().__init__()
        self.fc = nn.Linear(state_dim, EMBED_DIM)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(EMBED_DIM, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, IMAGE_CHANNELS, 6, stride=2),
        )

    def forward(self, features):
        return self.deconv(self.fc(features).view(-1, EMBED_DIM, 1, 1))


class RewardHead(nn.Module):
    def __init__(self, state_dim=H_DIM + Z_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class RSSM(nn.Module):
    def __init__(self, action_dim=ACTION_DIM, h_dim=H_DIM, z_dim=Z_DIM,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, min_std=MIN_STD):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.min_std = min_std
        self.inp_fc = nn.Linear(z_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, h_dim)
        self.prior_net = nn.Sequential(nn.Linear(h_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * z_dim))
        self.post_net = nn.Sequential(nn.Linear(h_dim + embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2 * z_dim))

    def initial_state(self, batch_size, device):
        return (torch.zeros(batch_size, self.h_dim, device=device),
                torch.zeros(batch_size, self.z_dim, device=device))

    def _split_dist(self, raw):
        mean, raw_std = raw.chunk(2, dim=-1)
        return mean, F.softplus(raw_std) + self.min_std

    def _deterministic_step(self, h_prev, z_prev, a_prev):
        return self.gru(F.relu(self.inp_fc(torch.cat([z_prev, a_prev], dim=-1))), h_prev)

    def prior(self, h_t):
        return self._split_dist(self.prior_net(h_t))

    def posterior(self, h_t, embed_t):
        return self._split_dist(self.post_net(torch.cat([h_t, embed_t], dim=-1)))

    @staticmethod
    def _sample(mean, std, sample, generator=None):
        if not sample:
            return mean
        eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        return mean + std * eps

    def observe_step(self, h_prev, z_prev, a_prev, embed_t, sample=True, generator=None):
        h_t = self._deterministic_step(h_prev, z_prev, a_prev)
        pm, ps = self.prior(h_t)
        qm, qs = self.posterior(h_t, embed_t)
        z_t = self._sample(qm, qs, sample, generator)
        return h_t, z_t, (pm, ps), (qm, qs)

    def imagine_step(self, h_prev, z_prev, a_prev, sample=True, generator=None):
        h_t = self._deterministic_step(h_prev, z_prev, a_prev)
        pm, ps = self.prior(h_t)
        z_t = self._sample(pm, ps, sample, generator)
        return h_t, z_t, (pm, ps)

    def observe(self, embeds, actions, state=None, sample=True):
        b, seq_len, _ = embeds.shape
        h, z = self.initial_state(b, embeds.device) if state is None else state
        keys = ("h", "z", "prior_mean", "prior_std", "post_mean", "post_std")
        lists = {k: [] for k in keys}
        for t in range(seq_len):
            h, z, (pm, ps), (qm, qs) = self.observe_step(h, z, actions[:, t], embeds[:, t], sample=sample)
            for k, val in zip(keys, (h, z, pm, ps, qm, qs)):
                lists[k].append(val)
        out = {k: torch.stack(v, dim=1) for k, v in lists.items()}
        out["features"] = torch.cat([out["h"], out["z"]], dim=-1)
        return out

    def imagine(self, h, z, actions, sample=True, generator=None):
        feats = []
        for k in range(actions.shape[1]):
            h, z, _ = self.imagine_step(h, z, actions[:, k], sample=sample, generator=generator)
            feats.append(torch.cat([h, z], dim=-1))
        return torch.stack(feats, dim=1)


class WorldModel(nn.Module):
    def __init__(self, action_dim=ACTION_DIM, h_dim=H_DIM, z_dim=Z_DIM):
        super().__init__()
        state_dim = h_dim + z_dim
        self.encoder = ConvEncoder()
        self.rssm = RSSM(action_dim=action_dim, h_dim=h_dim, z_dim=z_dim)
        self.decoder = ConvDecoder(state_dim=state_dim)
        self.reward_head = RewardHead(state_dim=state_dim)

    def encode_seq(self, obs_seq):
        b, t = obs_seq.shape[:2]
        return self.encoder(obs_seq.flatten(0, 1)).view(b, t, -1)

    def decode_seq(self, features):
        b, t = features.shape[:2]
        return self.decoder(features.flatten(0, 1)).view(b, t, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


def world_model_loss(model, obs, actions, rewards, beta=1.0, free_nats=FREE_NATS):
    """计算一个序列批次的重建 + 奖励 + free nats KL 损失。"""
    embeds = model.encode_seq(obs)
    out = model.rssm.observe(embeds, actions)
    recon = model.decode_seq(out["features"])
    pred_rew = model.reward_head(out["features"])

    recon_loss = 0.5 * (recon - obs).pow(2).sum(dim=(2, 3, 4))
    reward_loss = 0.5 * (pred_rew - rewards).pow(2)
    raw_kl = gaussian_kl(out["post_mean"], out["post_std"], out["prior_mean"], out["prior_std"])
    kl_loss = raw_kl.clamp_min(free_nats) if free_nats > 0.0 else raw_kl
    total = (recon_loss + reward_loss + beta * kl_loss).mean()
    return total, dict(recon=recon_loss.mean().item(), reward=reward_loss.mean().item(), raw_kl=raw_kl.mean().item())
