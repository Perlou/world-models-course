"""
Phase 9 · 第 05 课：自回归漂移、噪声增强与 Diffusion Forcing
==========================================================

学习目标：
    1. 在真实 PushT 演示序列上测量扩散世界模型开环展开的复合漂移曲线
    2. 实现上下文噪声增强（noise augmentation）与推理期稳定化噪声标记 σ_stab
    3. 实现 Diffusion Forcing：因果序列扩散模型 + 每帧独立噪声水平 σ_t ∈ [B, T]
       + 三种采样日程表（全序列同步 / 自回归 / 金字塔）
    4. 同一测试集上对比「无处理 / 噪声增强 / Diffusion Forcing」三条开环误差曲线及
       不同稳定化噪声水平 σ_stab ∈ {0.0, 0.05, 0.15, 0.30} 的影响

对应概念文档：
    src/phase-09-diffusion-video-world-models/CONCEPT.md 第 5 节

运行方式：
    uv run python src/phase-09-diffusion-video-world-models/05-autoregressive-drift-and-diffusion-forcing.py
    WM_QUICK=1 uv run python src/phase-09-diffusion-video-world-models/05-autoregressive-drift-and-diffusion-forcing.py

预计运行时间：完整约 3 分钟（M1 Max CPU），冒烟约 15 秒
"""

import importlib.util
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec01 = importlib.util.spec_from_file_location("p9_01", os.path.join(_DIR, "01-ddpm-to-edm.py"))
_mod01 = importlib.util.module_from_spec(_spec01)
_spec01.loader.exec_module(_mod01)

_spec03 = importlib.util.spec_from_file_location("p9_03", os.path.join(_DIR, "03-action-conditioned-frame-diffusion.py"))
_mod03 = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(_mod03)

edm_precond_coeffs = _mod01.edm_precond_coeffs
karras_sigmas = _mod01.karras_sigmas
sample_train_sigmas = _mod01.sample_train_sigmas
SinusoidalEmb = _mod01.SinusoidalEmb

FrameDenoiser = _mod03.FrameDenoiser
sample_next_frame = _mod03.sample_next_frame
load_pusht_clips = _mod03.load_pusht_clips
sample_window_batch = _mod03.sample_window_batch
psnr_from_m11 = _mod03.psnr_from_m11

plt = setup_matplotlib()
OUT = output_dir("phase-09")
SEED = 0


# ==================== 第一部分：噪声增强、日程表与因果 Diffusion Forcing ====================


def noise_augment_context(context, max_sigma=0.35):
    """CONCEPT.md 第 5.3 节：给上下文帧注入随机高斯噪声，让模型习惯不完美的历史。"""
    B = context.shape[0]
    sigma_c = torch.rand(B, device=context.device) * max_sigma
    noisy_ctx = context + sigma_c.view(B, 1, 1, 1, 1) * torch.randn_like(context)
    return noisy_ctx, sigma_c


def sample_per_frame_sigmas(batch_size, n_frames, p_mean=-0.4, p_std=1.2, device="cpu"):
    """CONCEPT.md 第 5.4 节：序列中每一帧独立采样噪声水平 σ_t，形状 [B, T]。"""
    return torch.exp(torch.randn(batch_size, n_frames, device=device) * p_std + p_mean).clamp(0.002, 10.0)


def make_schedule_matrix(mode, n_frames, n_levels=4):
    """CONCEPT.md 第 5.5 节：生成三种采样日程矩阵 [R, T]（值越大噪声越高，0 代表已干净）。"""
    if mode == "full_sequence":
        levels = np.linspace(n_levels, 0, n_levels + 1, dtype=int)
        return np.repeat(levels[:, None], n_frames, axis=1)
    if mode == "autoregressive":
        rows = []
        state = np.full(n_frames, n_levels, dtype=int)
        rows.append(state.copy())
        for t in range(n_frames):
            for lv in range(n_levels - 1, -1, -1):
                state[t] = lv
                rows.append(state.copy())
        return np.asarray(rows)
    if mode == "pyramid":
        # 金字塔日程：越靠后的帧滞后一步，流水线并行去噪
        rows = []
        total_r = n_frames + n_levels - 1
        for r in range(total_r + 1):
            row = [max(0, min(n_levels, n_levels - (r - t))) for t in range(n_frames)]
            rows.append(row)
        return np.asarray(rows)
    raise ValueError(f"未知日程模式: {mode}")


class CausalDiffusionForcingModel(nn.Module):
    """每帧独立噪声水平 σ_t ∈ [B, T] 的因果序列扩散模型（空间卷积编码 + 因果时间 Transformer + 空间解码）。"""

    def __init__(self, img_ch=3, act_dim=2, d_model=128, n_heads=4, n_layers=2, sigma_data=0.5, max_len=24):
        super().__init__()
        self.sigma_data = float(sigma_data)
        self.noise_emb = SinusoidalEmb(32)
        self.enc = nn.Sequential(
            nn.Conv2d(img_ch, 24, 4, 2, 1), nn.SiLU(),  # 16
            nn.Conv2d(24, 48, 4, 2, 1), nn.SiLU(),      # 8
            nn.Conv2d(48, 32, 4, 2, 1), nn.SiLU(),      # 4 -> 32*4*4 = 512
        )
        self.in_proj = nn.Linear(32 * 4 * 4 + 32 + act_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, 32 * 4 * 4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 48, 4, 2, 1), nn.SiLU(),  # 8
            nn.ConvTranspose2d(48, 24, 4, 2, 1), nn.SiLU(),  # 16
            nn.ConvTranspose2d(24, img_ch, 4, 2, 1),         # 32
        )
        # 局部跳连保留高频空间细节
        self.local_res = nn.Sequential(nn.Conv2d(img_ch, 16, 3, 1, 1), nn.SiLU(), nn.Conv2d(16, img_ch, 3, 1, 1))

    def _forward_f(self, x_in, c_noise, actions):
        # x_in: [B, T, 3, 32, 32], c_noise: [B, T], actions: [B, T, A]
        B, T, C, H, W = x_in.shape
        feat = self.enc(x_in.reshape(B * T, C, H, W)).reshape(B, T, -1)
        n_e = self.noise_emb(c_noise.reshape(B * T)).reshape(B, T, -1)
        tokens = self.in_proj(torch.cat([feat, n_e, actions], dim=-1))
        pos = torch.arange(T, device=x_in.device)
        mask = torch.triu(torch.ones(T, T, device=x_in.device, dtype=torch.bool), diagonal=1)
        h = self.tf(tokens + self.pos_emb(pos).unsqueeze(0), mask=mask)
        dec_in = self.out_proj(h).reshape(B * T, 32, 4, 4)
        out = self.dec(dec_in) + 0.3 * self.local_res(x_in.reshape(B * T, C, H, W))
        return out.reshape(B, T, C, H, W)

    def forward(self, noisy_frames, sigmas, actions):
        B, T = sigmas.shape
        c_in, c_skip, c_out, c_noise, _ = edm_precond_coeffs(sigmas, self.sigma_data)
        x_in = c_in[:, :, None, None, None] * noisy_frames
        f_out = self._forward_f(x_in, c_noise, actions)
        return c_skip[:, :, None, None, None] * noisy_frames + c_out[:, :, None, None, None] * f_out

    def diffusion_forcing_loss(self, clean_frames, actions):
        """CONCEPT.md 第 5.7 节：每帧独立采 σ_t，在 F 空间上按帧算均方误差。"""
        B, T = clean_frames.shape[:2]
        sigmas = sample_per_frame_sigmas(B, T, device=clean_frames.device)
        # 以 25% 概率把首帧标为极低噪声（模拟推理时干净起始帧）
        clean_mask = torch.rand(B, device=clean_frames.device) < 0.25
        sigmas[clean_mask, 0] = 0.002
        eps = torch.randn_like(clean_frames)
        noisy = clean_frames + sigmas[:, :, None, None, None] * eps
        c_in, c_skip, c_out, c_noise, _ = edm_precond_coeffs(sigmas, self.sigma_data)
        f_pred = self._forward_f(c_in[:, :, None, None, None] * noisy, c_noise, actions)
        f_target = (clean_frames - c_skip[:, :, None, None, None] * noisy) / c_out[:, :, None, None, None].clamp_min(1e-6)
        return F.mse_loss(f_pred, f_target)


@torch.no_grad()
def rollout_with_stabilization(df_model, init_frames, action_seq, n_steps=3, sigma_stab=0.10, window=10):
    """CONCEPT.md 第 5.7 节：带稳定化噪声标记 σ_stab 的 Diffusion Forcing 自回归展开。"""
    df_model.eval()
    B, m, C, H, W = init_frames.shape
    device = init_frames.device
    H_roll = action_seq.shape[1] - m
    sigmas_sched = karras_sigmas(n_steps, 0.002, 5.0, 7.0, device=device)

    hist_f = init_frames.clone()
    hist_s = torch.full((B, m), 0.002, device=device)  # 真实历史帧标为干净
    preds = []

    for t in range(H_roll):
        x = torch.randn(B, C, H, W, device=device) * sigmas_sched[0]
        # 截断滑动窗口防止超出 max_len
        cur_f = hist_f[:, -(window - 1) :]
        cur_s = hist_s[:, -(window - 1) :]
        start_idx = max(0, (m + t) - cur_f.shape[1])
        cur_a = action_seq[:, start_idx : m + t + 1]  # 长度 = cur_f.shape[1] + 1

        for i in range(n_steps):
            s_cur, s_nxt = sigmas_sched[i], sigmas_sched[i + 1]
            seq_f = torch.cat([cur_f, x.unsqueeze(1)], dim=1)
            seq_s = torch.cat([cur_s, torch.full((B, 1), s_cur.item(), device=device)], dim=1)
            x_hat = df_model(seq_f, seq_s, cur_a)[:, -1]
            x = x_hat + (s_nxt / s_cur) * (x - x_hat)

        x = x.clamp(-1.0, 1.0)
        preds.append(x)
        # 关键：将生成帧追加进历史时，标为稳定化噪声 σ_stab（而非 0）
        if sigma_stab > 0.0:
            x_hist = x + sigma_stab * torch.randn_like(x)
            s_tag = sigma_stab
        else:
            x_hist = x
            s_tag = 0.002
        hist_f = torch.cat([hist_f, x_hist.unsqueeze(1)], dim=1)
        hist_s = torch.cat([hist_s, torch.full((B, 1), s_tag, device=device)], dim=1)

    return torch.stack(preds, dim=1)


def sample_clip_batch(episodes, batch_size, seq_len=10):
    fs, acts = [], []
    for _ in range(batch_size):
        imgs, a = episodes[np.random.randint(len(episodes))]
        t = np.random.randint(0, len(imgs) - seq_len)
        fs.append(imgs[t : t + seq_len])
        # 第 t 帧预测时传入导致它的动作 a_{t-1}（第 0 帧补零）
        a_shifted = np.concatenate([np.zeros((1, a.shape[1]), dtype=np.float32), a[t : t + seq_len - 1]], axis=0)
        acts.append(a_shifted)
    return torch.from_numpy(np.asarray(fs)), torch.from_numpy(np.asarray(acts))


@torch.no_grad()
def measure_drift(rollout_fn, te_eps, ctx_len=3, horizon=12, n_clips=8):
    """在同一批测试片段上测量每步开环展开的 MSE 与 PSNR。"""
    mses = np.zeros(horizon, dtype=np.float64)
    psnrs = np.zeros(horizon, dtype=np.float64)
    count = 0
    for i in range(min(n_clips, len(te_eps))):
        imgs, acts = te_eps[i]
        if len(imgs) < ctx_len + horizon + 1:
            continue
        init_f = torch.from_numpy(imgs[:ctx_len]).unsqueeze(0)
        gt = torch.from_numpy(imgs[ctx_len : ctx_len + horizon])
        pred = rollout_fn(init_f, acts, horizon)[0]
        for k in range(horizon):
            p01 = (pred[k].clamp(-1, 1) + 1) * 0.5
            g01 = (gt[k].clamp(-1, 1) + 1) * 0.5
            m = F.mse_loss(p01, g01).item()
            mses[k] += m
            psnrs[k] += -10.0 * math.log10(max(m, 1e-8))
        count += 1
    return mses / max(count, 1), psnrs / max(count, 1)


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    print("=" * 60)
    print("第一部分：验证三种采样日程矩阵（全序列 / 自回归 / 金字塔）")
    print("=" * 60)
    for mode in ["full_sequence", "autoregressive", "pyramid"]:
        mat = make_schedule_matrix(mode, n_frames=4, n_levels=3)
        print(f"  [{mode:14s}] 形状 {mat.shape}，首行 {mat[0].tolist()} → 末行 {mat[-1].tolist()}")

    print("\n" + "=" * 60)
    print("第二部分：PushT 32×32 上对比三种自回归展开方案")
    print("=" * 60)
    ctx_len = 3
    horizon = 8 if quick else 12
    episodes = load_pusht_clips(n_episodes=10 if quick else 32, img_size=32, frame_stride=2)
    n_tr = max(2, int(len(episodes) * 0.75))
    tr_eps, te_eps = episodes[:n_tr], episodes[n_tr:]
    sigma_data = float(np.concatenate([e[0] for e in tr_eps[:4]], axis=0).std())

    # 1. 无处理 AR 扩散
    m_plain = FrameDenoiser(img_ch=3, ctx_len=ctx_len, act_dim=2, base_ch=24, cond_dim=48, sigma_data=sigma_data)
    # 2. 噪声增强 AR 扩散
    m_aug = FrameDenoiser(img_ch=3, ctx_len=ctx_len, act_dim=2, base_ch=24, cond_dim=48, sigma_data=sigma_data)
    # 3. 因果 Diffusion Forcing
    m_df = CausalDiffusionForcingModel(img_ch=3, act_dim=2, d_model=128, n_heads=4, n_layers=2, sigma_data=sigma_data)

    opt_p = torch.optim.Adam(m_plain.parameters(), lr=1e-3)
    opt_a = torch.optim.Adam(m_aug.parameters(), lr=1e-3)
    opt_df = torch.optim.Adam(m_df.parameters(), lr=1e-3)

    n_iters = 70 if quick else 500
    for it in range(1, n_iters + 1):
        ctx, act, tgt = sample_window_batch(tr_eps, 24, ctx_len=ctx_len)
        opt_p.zero_grad(); m_plain.loss(tgt, ctx, act).backward(); nn.utils.clip_grad_norm_(m_plain.parameters(), 1.0); opt_p.step()

        n_ctx, sig_c = noise_augment_context(ctx, max_sigma=0.35)
        opt_a.zero_grad(); m_aug.loss(tgt, n_ctx, act, ctx_sigma=sig_c).backward(); nn.utils.clip_grad_norm_(m_aug.parameters(), 1.0); opt_a.step()

        f_clip, a_clip = sample_clip_batch(tr_eps, 16, seq_len=8)
        opt_df.zero_grad(); m_df.diffusion_forcing_loss(f_clip, a_clip).backward(); nn.utils.clip_grad_norm_(m_df.parameters(), 1.0); opt_df.step()

    print(f"  三模型训练完成（各 {n_iters} 步，{time.time() - t0:.1f}s）")

    def ar_roll(model, use_stab=0.0):
        def _fn(init_f, acts_np, H):
            buf = init_f.clone()
            preds = []
            for k in range(H):
                a_w = torch.from_numpy(acts_np[k : k + ctx_len]).unsqueeze(0)
                if use_stab > 0.0 and k > 0:
                    c_in = buf + use_stab * torch.randn_like(buf)
                    cs = torch.tensor([use_stab])
                else:
                    c_in = buf
                    cs = torch.tensor([0.0]) if use_stab > 0.0 else None
                nxt = sample_next_frame(model, c_in, a_w, n_steps=3, ctx_sigma=cs)
                preds.append(nxt)
                buf = torch.cat([buf[:, 1:], nxt.unsqueeze(1)], dim=1)
            return torch.stack(preds, dim=1)
        return _fn

    def df_roll(sigma_stab):
        def _fn(init_f, acts_np, H):
            a_shift = np.concatenate([np.zeros((1, 2), dtype=np.float32), acts_np[: ctx_len + H - 1]], axis=0)
            a_t = torch.from_numpy(a_shift).unsqueeze(0)
            return rollout_with_stabilization(m_df, init_f, a_t, n_steps=3, sigma_stab=sigma_stab)
        return _fn

    mse_plain, psnr_plain = measure_drift(ar_roll(m_plain, 0.0), te_eps, ctx_len, horizon)
    mse_aug, psnr_aug = measure_drift(ar_roll(m_aug, 0.10), te_eps, ctx_len, horizon)
    df_results = {}
    for st in [0.0, 0.05, 0.15, 0.30]:
        df_results[st] = measure_drift(df_roll(st), te_eps, ctx_len, horizon)

    print(f"\n  开环展开 PSNR (dB) 随步数变化（共 {horizon} 步，越高越好）：")
    check_steps = [0, min(4, horizon - 1), min(7, horizon - 1), horizon - 1]
    hdr = "  " + f"{'方法':26s} | " + " | ".join(f"t+{k+1:2d}" for k in check_steps) + " | 全程均MSE"
    print(hdr)
    for name, (ms, ps) in [
        ("无处理 AR 扩散", (mse_plain, psnr_plain)),
        ("噪声增强 AR (σ_c=0.10)", (mse_aug, psnr_aug)),
        ("Diffusion Forcing (σ_stab=0.00)", df_results[0.0]),
        ("Diffusion Forcing (σ_stab=0.05)", df_results[0.05]),
        ("Diffusion Forcing (σ_stab=0.15)", df_results[0.15]),
        ("Diffusion Forcing (σ_stab=0.30)", df_results[0.30]),
    ]:
        vals = " | ".join(f"{ps[k]:5.2f}" for k in check_steps)
        print(f"  {name:26s} | {vals} | {ms.mean():.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = np.arange(1, horizon + 1)
    axes[0].plot(xs, psnr_plain, "o-", label="无处理 AR")
    axes[0].plot(xs, psnr_aug, "s-", label="噪声增强 AR (σ=0.10)")
    axes[0].plot(xs, df_results[0.15][1], "^-", label="Diffusion Forcing (σ_stab=0.15)")
    axes[0].set_title("三条开环误差曲线（PSNR 越高越好）")
    axes[0].set_xlabel("开环展开步数"); axes[0].set_ylabel("PSNR (dB)"); axes[0].legend()

    for st in [0.0, 0.05, 0.15, 0.30]:
        axes[1].plot(xs, df_results[st][1], "o-", label=f"σ_stab = {st:.2f}")
    axes[1].set_title("Diffusion Forcing 稳定化噪声水平消融")
    axes[1].set_xlabel("开环展开步数"); axes[1].set_ylabel("PSNR (dB)"); axes[1].legend()

    fig.tight_layout()
    save_figure(fig, OUT / "05-drift-and-diffusion-forcing.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：06-video-tokenizers-and-latent-diffusion.py")


if __name__ == "__main__":
    main()
