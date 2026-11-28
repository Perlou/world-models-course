"""Phase 11 实战项目核心模块：用世界模型评估机器人策略（PushT 策略套件 + 双版本世界模型对比 + Pearson/Spearman/MMRV）。"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pusht import load_pusht_demos, make_pusht_env


def pearson_r(y, y_hat):
    y = np.asarray(y, dtype=np.float64)
    yh = np.asarray(y_hat, dtype=np.float64)
    ey, eyh = y - y.mean(), yh - yh.mean()
    denom = math.sqrt(float((ey ** 2).sum() * (eyh ** 2).sum()))
    return 0.0 if denom < 1e-12 else float((ey * eyh).sum() / denom)


def _avg_ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty_like(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_rho(y, y_hat):
    return pearson_r(_avg_ranks(y), _avg_ranks(y_hat))


def mmrv(y, y_hat):
    y = np.asarray(y, dtype=np.float64)
    yh = np.asarray(y_hat, dtype=np.float64)
    N = len(y)
    viols = []
    for i in range(N):
        vi = 0.0
        for j in range(N):
            if i != j and (y[i] != y[j]) and ((yh[i] < yh[j]) != (y[i] < y[j])):
                vi = max(vi, abs(y[i] - y[j]))
        viols.append(vi)
    return float(np.mean(viols))


def extract_block_pose_batch(imgs):
    """从 PushT 96×96 RGB 画面提取灰色 T 形块的质心与主轴朝向 (cx, cy, theta)。"""
    f = imgs.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    lum = (r + g + b) / 3.0
    mask = ((sat < 25.0) & (lum > 70.0) & (lum < 210.0)).astype(np.float32)
    _, H, W = mask.shape
    xs = np.linspace(0.0, 512.0, W, dtype=np.float32)[None, None, :]
    ys = np.linspace(0.0, 512.0, H, dtype=np.float32)[None, :, None]
    area = mask.sum(axis=(1, 2)) + 1e-6
    cx = (mask * xs).sum(axis=(1, 2)) / area
    cy = (mask * ys).sum(axis=(1, 2)) / area
    dx = xs - cx[:, None, None]
    dy = ys - cy[:, None, None]
    mu20 = (mask * dx * dx).sum(axis=(1, 2)) / area
    mu02 = (mask * dy * dy).sum(axis=(1, 2)) / area
    mu11 = (mask * dx * dy).sum(axis=(1, 2)) / area
    theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    return np.stack([cx, cy, theta], axis=-1).astype(np.float32)


def extract_demo_states_5d(demos, s, e):
    return np.concatenate([demos["states"][s:e, :2].astype(np.float32), extract_block_pose_batch(demos["images"][s:e])], axis=-1)


def normalize_pusht_state(st):
    out = np.asarray(st, dtype=np.float32).copy()
    out[..., :4] = out[..., :4] / 256.0 - 1.0
    if out.shape[-1] >= 5:
        out[..., 4] = out[..., 4] / (math.pi * 0.5)
    return out


def extract_state_5d(obs_dict):
    bp = extract_block_pose_batch(obs_dict["pixels"][None])[0]
    return normalize_pusht_state(np.concatenate([obs_dict["agent_pos"][:2], bp]))


class _SinEmb(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        h = self.dim // 2
        f = torch.exp(torch.linspace(0.0, -math.log(1000.0), h, device=t.device))
        a = t.unsqueeze(-1) * f.unsqueeze(0)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class PushTBC(nn.Module):
    def __init__(self, obs_dim=5, act_dim=2, T_o=2, T_p=8, hidden=160):
        super().__init__()
        self.T_o, self.T_p, self.act_dim = T_o, T_p, act_dim
        self.net = nn.Sequential(
            nn.Linear(T_o * obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, T_p * act_dim),
        )

    def forward(self, obs_win):
        return self.net(obs_win.flatten(1)).reshape(-1, self.T_p, self.act_dim)

    @torch.no_grad()
    def predict_chunk(self, obs_win):
        self.eval()
        return self(obs_win)


class PushTDP(nn.Module):
    def __init__(self, obs_dim=5, act_dim=2, T_o=2, T_p=8, hidden=160):
        super().__init__()
        self.T_o, self.T_p, self.act_dim = T_o, T_p, act_dim
        self.u_emb = _SinEmb(32)
        self.ctx = nn.Sequential(nn.Linear(T_o * obs_dim, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(T_p * act_dim + 64 + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, T_p * act_dim),
        )

    def forward(self, x_u, u, obs_win):
        c = torch.cat([self.ctx(obs_win.flatten(1)), self.u_emb(u)], dim=-1)
        return self.net(torch.cat([x_u.flatten(1), c], dim=-1)).reshape(-1, self.T_p, self.act_dim)

    def loss(self, clean, obs_win):
        B = clean.shape[0]
        eps = torch.randn_like(clean)
        u = torch.rand(B, device=clean.device)
        x_u = (1.0 - u.view(B, 1, 1)) * eps + u.view(B, 1, 1) * clean
        return F.mse_loss(self(x_u, u, obs_win), clean - eps)

    @torch.no_grad()
    def predict_chunk(self, obs_win, n_steps=8):
        self.eval()
        B = obs_win.shape[0]
        x = torch.randn(B, self.T_p, self.act_dim, device=obs_win.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            x = x + dt * self(x, torch.full((B,), i * dt, device=obs_win.device), obs_win)
        return x


class RandomPushTPolicy:
    T_o = 2
    T_p = 8

    def predict_chunk(self, obs_win):
        return torch.rand(obs_win.shape[0], self.T_p, 2) * 2.0 - 1.0


class PushTWMEvaluator(nn.Module):
    """用于策略评估的 PushT 动力学世界模型 + 覆盖度/成功判定器。"""

    def __init__(self, state_dim=5, act_dim=2, hidden=192):
        super().__init__()
        self.dyn = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
        self.cov_head = nn.Sequential(nn.Linear(state_dim, 64), nn.SiLU(), nn.Linear(64, 1), nn.Sigmoid())

    def step(self, s, a):
        ns = (s + self.dyn(torch.cat([s, a], dim=-1))).clamp(-1.2, 1.2)
        return ns, self.cov_head(ns).squeeze(-1)


def evaluate_in_real_env(policy, seeds, max_steps=120, T_a=4, thresh=0.50):
    env = make_pusht_env(obs_type="pixels_agent_pos")
    covs, succ, init_states = [], 0, []
    for sd in seeds:
        obs, _ = env.reset(seed=sd)
        s0 = extract_state_5d(obs)
        init_states.append(s0.copy())
        buf = [s0.copy() for _ in range(policy.T_o)]
        m_cov, step = 0.0, 0
        while step < max_steps:
            chunk = policy.predict_chunk(torch.from_numpy(np.asarray(buf)).unsqueeze(0))[0].numpy()
            for k in range(min(T_a, policy.T_p)):
                if step >= max_steps:
                    break
                act_env = np.clip((chunk[k] + 1.0) * 256.0, 0.0, 512.0)
                obs, r, term, trunc, info = env.step(act_env)
                m_cov = max(m_cov, float(info.get("coverage", r)))
                buf = buf[1:] + [extract_state_5d(obs)]
                step += 1
                if term or trunc:
                    step = max_steps
                    break
        covs.append(m_cov); succ += int(m_cov >= thresh)
    env.close()
    return float(np.mean(covs)), succ / len(seeds), init_states


@torch.no_grad()
def evaluate_in_world_model(policy, wm, init_states, max_steps=120, T_a=4, thresh=0.50):
    wm.eval()
    covs, succ = [], 0
    for s0 in init_states:
        buf = [s0.copy() for _ in range(policy.T_o)]
        cur_s = torch.from_numpy(s0).unsqueeze(0)
        m_cov, step = 0.0, 0
        while step < max_steps:
            chunk = policy.predict_chunk(torch.from_numpy(np.asarray(buf)).unsqueeze(0))[0]
            for k in range(min(T_a, policy.T_p)):
                if step >= max_steps:
                    break
                cur_s, c_hat = wm.step(cur_s, chunk[k].clamp(-1.0, 1.0).unsqueeze(0))
                m_cov = max(m_cov, float(c_hat.item()))
                buf = buf[1:] + [cur_s[0].numpy()]
                step += 1
        covs.append(m_cov); succ += int(m_cov >= thresh)
    return float(np.mean(covs)), succ / len(init_states)
