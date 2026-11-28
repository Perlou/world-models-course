"""Phase 9 实战项目核心模块：PushT 可玩神经仿真器（动作条件 EDM 扩散世界模型 + 噪声增强 + CEM 规划）。"""

import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.planning import cem_plan
from utils.pusht import load_pusht_demos, make_pusht_env


def edm_coeffs(sigma, sigma_data=0.5):
    s2 = sigma ** 2
    sd2 = sigma_data ** 2
    denom = torch.sqrt(s2 + sd2)
    return 1.0 / denom, sd2 / (s2 + sd2), (sigma * sigma_data) / denom, 0.25 * torch.log(sigma.clamp_min(1e-8))


def karras_schedule(n_steps, sigma_min=0.002, sigma_max=5.0, rho=7.0, device="cpu"):
    if n_steps == 1:
        return torch.tensor([sigma_max, 0.0], dtype=torch.float32, device=device)
    ramp = torch.linspace(0.0, 1.0, n_steps, device=device)
    s = (sigma_max ** (1.0 / rho) + ramp * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
    return torch.cat([s, torch.zeros(1, device=device)])


class _SinEmb(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        h = self.dim // 2
        f = torch.exp(torch.linspace(0.0, -math.log(1000.0), h, device=x.device))
        a = x.unsqueeze(-1) * f.unsqueeze(0)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class _FiLMResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=48):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.n2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.c1(F.silu(self.n1(x)))
        g, b = self.film(cond).chunk(2, dim=-1)
        h = self.n2(h) * (1.0 + g[:, :, None, None]) + b[:, :, None, None]
        return self.c2(F.silu(h)) + self.skip(x)


class PushTDiffusionWM(nn.Module):
    """PushT 32×32 RGB 动作条件 EDM 扩散世界模型，支持上下文噪声增强与交互式逐帧推演。"""

    def __init__(self, ctx_len=3, act_dim=2, base_ch=24, cond_dim=48, sigma_data=0.5):
        super().__init__()
        self.ctx_len = ctx_len
        self.act_dim = act_dim
        self.sigma_data = float(sigma_data)

        self.noise_emb = nn.Sequential(_SinEmb(cond_dim), nn.Linear(cond_dim, cond_dim), nn.SiLU())
        self.ctx_noise_emb = nn.Sequential(_SinEmb(cond_dim), nn.Linear(cond_dim, cond_dim), nn.SiLU())
        self.act_emb = nn.Sequential(nn.Linear(ctx_len * act_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        in_ch = (ctx_len + 1) * 3
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.d1 = _FiLMResBlock(base_ch, base_ch, cond_dim)
        self.p1 = nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)
        self.d2 = _FiLMResBlock(base_ch * 2, base_ch * 2, cond_dim)
        self.p2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)
        self.mid = _FiLMResBlock(base_ch * 4, base_ch * 4, cond_dim)
        self.u2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        self.dec2 = _FiLMResBlock(base_ch * 4, base_ch * 2, cond_dim)
        self.u1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.dec1 = _FiLMResBlock(base_ch * 2, base_ch, cond_dim)
        self.out_conv = nn.Conv2d(base_ch, 3, 3, 1, 1)

        # 轻量视觉进度/奖励预测头（用于 CEM 规划打分）
        self.rew_head = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 1),
        )

    def _unet(self, x_cat, cond):
        h1 = self.d1(self.in_conv(x_cat), cond)
        h2 = self.d2(self.p1(h1), cond)
        hm = self.mid(self.p2(h2), cond)
        u2 = self.dec2(torch.cat([self.u2(hm), h2], dim=1), cond)
        u1 = self.dec1(torch.cat([self.u1(u2), h1], dim=1), cond)
        return self.out_conv(u1)

    def forward(self, noisy_next, sigma, ctx_frames, actions, ctx_sigma=None):
        B = noisy_next.shape[0]
        sigma = sigma.reshape(B)
        c_in, c_skip, c_out, c_noise = edm_coeffs(sigma, self.sigma_data)
        ctx_flat = ctx_frames.reshape(B, self.ctx_len * 3, *noisy_next.shape[2:])
        x_cat = torch.cat([ctx_flat, c_in.view(B, 1, 1, 1) * noisy_next], dim=1)
        cond = self.noise_emb(c_noise) + self.act_emb(actions.reshape(B, -1))
        if ctx_sigma is not None:
            cond = cond + self.ctx_noise_emb(ctx_sigma.reshape(B))
        f_out = self._unet(x_cat, cond)
        return c_skip.view(B, 1, 1, 1) * noisy_next + c_out.view(B, 1, 1, 1) * f_out

    def predict_reward(self, frame_m11):
        return self.rew_head(frame_m11).squeeze(-1)

    def edm_loss(self, target_next, ctx_frames, actions, max_ctx_noise=0.25):
        B = target_next.shape[0]
        device = target_next.device
        sigma = torch.exp(torch.randn(B, device=device) * 1.2 - 0.4).clamp(0.002, 10.0)
        eps = torch.randn_like(target_next)
        noisy = target_next + sigma.view(B, 1, 1, 1) * eps
        c_in, c_skip, c_out, c_noise = edm_coeffs(sigma, self.sigma_data)

        if max_ctx_noise > 0.0:
            ctx_sig = torch.rand(B, device=device) * max_ctx_noise
            ctx_in = ctx_frames + ctx_sig.view(B, 1, 1, 1, 1) * torch.randn_like(ctx_frames)
        else:
            ctx_sig = torch.zeros(B, device=device)
            ctx_in = ctx_frames

        ctx_flat = ctx_in.reshape(B, self.ctx_len * 3, *target_next.shape[2:])
        x_cat = torch.cat([ctx_flat, c_in.view(B, 1, 1, 1) * noisy], dim=1)
        cond = self.noise_emb(c_noise) + self.act_emb(actions.reshape(B, -1)) + self.ctx_noise_emb(ctx_sig)
        f_pred = self._unet(x_cat, cond)
        f_tgt = (target_next - c_skip.view(B, 1, 1, 1) * noisy) / c_out.view(B, 1, 1, 1).clamp_min(1e-6)
        return F.mse_loss(f_pred, f_tgt)

    @torch.no_grad()
    def sample_next_frame(self, ctx_frames, actions, n_steps=3, sigma_stab=0.05):
        self.eval()
        B, _, C, H, W = ctx_frames.shape
        device = ctx_frames.device
        sigmas = karras_schedule(n_steps, 0.002, 5.0, 7.0, device=device)
        x = torch.randn(B, C, H, W, device=device) * sigmas[0]
        ctx_sig = torch.full((B,), float(sigma_stab), device=device)
        for i in range(n_steps):
            s_cur, s_nxt = sigmas[i], sigmas[i + 1]
            xh = self(x, torch.full((B,), s_cur.item(), device=device), ctx_frames, actions, ctx_sigma=ctx_sig)
            x = xh + (s_nxt / s_cur) * (x - xh)
        return x.clamp(-1.0, 1.0)


class PlayableSimulatorSession:
    """可交互神经仿真器会话：维护最近 m 帧历史，接收按键或连续 2D 动作实时吐出下一帧。"""

    KEY_DELTAS = {
        "w": np.array([0.0, -0.15], dtype=np.float32),
        "s": np.array([0.0, 0.15], dtype=np.float32),
        "a": np.array([-0.15, 0.0], dtype=np.float32),
        "d": np.array([0.15, 0.0], dtype=np.float32),
    }

    def __init__(self, model, init_frames, init_actions, n_steps=3, sigma_stab=0.05):
        self.model = model
        self.n_steps = n_steps
        self.sigma_stab = sigma_stab
        self.buf_f = init_frames.clone()   # [1, m, 3, 32, 32]
        self.buf_a = init_actions.clone()  # [1, m, 2]
        self.cursor = init_actions[0, -1].numpy().copy()

    def step_action(self, action_m11):
        t0 = time.perf_counter()
        a_t = torch.tensor(action_m11, dtype=torch.float32).view(1, 1, 2)
        a_win = torch.cat([self.buf_a[:, 1:], a_t], dim=1)
        nxt = self.model.sample_next_frame(self.buf_f, a_win, n_steps=self.n_steps, sigma_stab=self.sigma_stab)
        self.buf_f = torch.cat([self.buf_f[:, 1:], nxt.unsqueeze(1)], dim=1)
        self.buf_a = a_win
        self.cursor = np.asarray(action_m11, dtype=np.float32)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return nxt[0], dt_ms

    def step_key(self, key):
        delta = self.KEY_DELTAS.get(key.lower(), np.zeros(2, dtype=np.float32))
        new_act = np.clip(self.cursor + delta, -1.0, 1.0)
        return self.step_action(new_act)


def load_pusht_dataset(n_episodes=40, img_size=32, frame_stride=2):
    demos = load_pusht_demos()
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    episodes = []
    for i in range(min(n_episodes, len(starts))):
        s, e = int(starts[i]), int(ends[i])
        ims_raw = demos["images"][s:e:frame_stride]
        acts_raw = demos["actions"][s:e:frame_stride]
        if len(ims_raw) < 25:
            continue
        ims = np.stack([cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA) for im in ims_raw], axis=0)
        ims = np.transpose((ims.astype(np.float32) / 127.5) - 1.0, (0, 3, 1, 2))
        acts = (acts_raw.astype(np.float32) / 256.0) - 1.0
        # 用每回合推进进度作为奖励代理（PushT 演示向目标持续靠近，末帧完成）
        rews = np.linspace(0.0, 1.0, len(ims), dtype=np.float32)
        episodes.append((ims, acts, rews))
    return episodes


def psnr_m11(pred, tgt):
    p01 = (pred.clamp(-1, 1) + 1.0) * 0.5
    t01 = (tgt.clamp(-1, 1) + 1.0) * 0.5
    mse = F.mse_loss(p01, t01).item()
    return float(-10.0 * math.log10(max(mse, 1e-8)))


def detail_errors(pred, tgt):
    """分别测量 PushT 推杆（蓝色小点）和 T 形块（灰色主体）区域的 L1 重建误差。"""
    p01 = (pred.clamp(-1, 1) + 1.0) * 0.5
    t01 = (tgt.clamp(-1, 1) + 1.0) * 0.5
    blue_mask = ((t01[..., 2, :, :] - t01[..., 0, :, :]) > 0.15).float()
    # 非白背景像素近似为物体区（T 形块 + 目标轮廓）
    fg_mask = (t01.mean(dim=-3) < 0.88).float()
    l1_map = (p01 - t01).abs().mean(dim=-3)
    pusher_err = float(((l1_map * blue_mask).sum() / blue_mask.sum().clamp_min(1.0)).item())
    fg_err = float(((l1_map * fg_mask).sum() / fg_mask.sum().clamp_min(1.0)).item())
    return pusher_err, fg_err


def eval_cem_in_pusht(model, n_episodes=2, max_steps=40, horizon=3, n_samples=16, n_iters=2, n_denoise=1):
    """在真实 gym-pusht 环境中接上 Phase 1 的 cem_plan，测量扩散模型用于决策时规划的得分与耗时。"""
    env = make_pusht_env(obs_type="pixels_agent_pos")
    model.eval()
    returns, step_times = [], []
    for ep in range(n_episodes):
        obs_dict, _ = env.reset(seed=500 + ep)
        img = cv2.resize(obs_dict["pixels"], (32, 32), interpolation=cv2.INTER_AREA)
        f0 = torch.from_numpy(np.transpose((img.astype(np.float32) / 127.5) - 1.0, (2, 0, 1)))
        buf_f = f0.unsqueeze(0).unsqueeze(0).repeat(1, model.ctx_len, 1, 1, 1)
        a0 = (obs_dict["agent_pos"].astype(np.float32) / 256.0) - 1.0
        buf_a = torch.from_numpy(a0).view(1, 1, 2).repeat(1, model.ctx_len, 1)
        ep_rew = 0.0
        for _ in range(max_steps):
            t_s = time.perf_counter()

            def score_fn(cand_t):
                # cand_t: [N, H, 2] in [-1, 1]
                N, H, _ = cand_t.shape
                c_f = buf_f.repeat(N, 1, 1, 1, 1)
                c_a = buf_a.repeat(N, 1, 1)
                total = torch.zeros(N)
                with torch.no_grad():
                    for h in range(H):
                        a_win = torch.cat([c_a[:, 1:], cand_t[:, h : h + 1]], dim=1)
                        nxt = model.sample_next_frame(c_f, a_win, n_steps=n_denoise, sigma_stab=0.05)
                        total = total + model.predict_reward(nxt)
                        c_f = torch.cat([c_f[:, 1:], nxt.unsqueeze(1)], dim=1)
                        c_a = a_win
                return total

            best_seq = cem_plan(
                score_fn, torch.zeros(horizon, 2), action_low=-1.0, action_high=1.0,
                n_candidates=n_samples, n_elites=max(2, n_samples // 4), n_iters=n_iters,
            ).cpu().numpy()
            step_times.append((time.perf_counter() - t_s) * 1000.0)
            act_m11 = np.clip(best_seq[0], -1.0, 1.0)
            act_env = (act_m11 + 1.0) * 256.0
            obs_dict, r, term, trunc, _ = env.step(act_env)
            ep_rew += float(r)
            img = cv2.resize(obs_dict["pixels"], (32, 32), interpolation=cv2.INTER_AREA)
            fn = torch.from_numpy(np.transpose((img.astype(np.float32) / 127.5) - 1.0, (2, 0, 1)))
            buf_f = torch.cat([buf_f[:, 1:], fn.view(1, 1, 3, 32, 32)], dim=1)
            buf_a = torch.cat([buf_a[:, 1:], torch.from_numpy(act_m11).view(1, 1, 2)], dim=1)
            if term or trunc:
                break
        returns.append(ep_rew)
    env.close()
    return float(np.mean(returns)), float(np.mean(step_times))
