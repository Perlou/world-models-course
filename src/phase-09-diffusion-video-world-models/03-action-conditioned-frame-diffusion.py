"""
Phase 9 · 第 03 课：以动作为条件的下一帧扩散世界模型
====================================================

学习目标：
    1. 核对 CONCEPT.md 第 3.4 节双峰分布最优去噪器 D*(x_σ; σ) = tanh(x_σ / σ²)，
       直观看到 1 步采样为何发糊、3 步 Euler 如何选定具体模态
    2. 在真实 PushT 演示数据（32×32 RGB）上实现以过去 m 帧与连续 2D 动作为条件的 EDM 去噪器
    3. 对比 N ∈ {1, 2, 3, 5, 10} 步去噪下的下一帧 PSNR、运动区域清晰度（高频梯度幅值）与耗时
    4. 验证动作可控性（固定上下文、替换不同动作时的 ΔPSNR）以及自回归开环展开

对应概念文档：
    src/phase-09-diffusion-video-world-models/CONCEPT.md 第 3 节

运行方式：
    uv run python src/phase-09-diffusion-video-world-models/03-action-conditioned-frame-diffusion.py
    WM_QUICK=1 uv run python src/phase-09-diffusion-video-world-models/03-action-conditioned-frame-diffusion.py

预计运行时间：完整约 2 分钟（M1 Max CPU），冒烟约 12 秒
"""

import importlib.util
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.pusht import load_pusht_demos

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec01 = importlib.util.spec_from_file_location("p9_01", os.path.join(_DIR, "01-ddpm-to-edm.py"))
_mod01 = importlib.util.module_from_spec(_spec01)
_spec01.loader.exec_module(_mod01)

edm_precond_coeffs = _mod01.edm_precond_coeffs
karras_sigmas = _mod01.karras_sigmas
sample_train_sigmas = _mod01.sample_train_sigmas
SinusoidalEmb = _mod01.SinusoidalEmb

plt = setup_matplotlib()
OUT = output_dir("phase-09")
SEED = 0


# ==================== 第一部分：条件去噪 U-Net（供 03/04/05 复用） ====================


class ActionEmbedding(nn.Module):
    """把过去 m 步的连续或离散动作压成条件向量。"""

    def __init__(self, act_dim=2, ctx_len=4, cond_dim=64, discrete_actions=None):
        super().__init__()
        self.discrete_actions = discrete_actions
        if discrete_actions is not None:
            self.emb = nn.Embedding(discrete_actions, 16)
            in_dim = ctx_len * 16
        else:
            in_dim = ctx_len * act_dim
        self.proj = nn.Sequential(nn.Linear(in_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))

    def forward(self, actions):
        if self.discrete_actions is not None:
            h = self.emb(actions.long()).flatten(1)
        else:
            h = actions.float().flatten(1)
        return self.proj(h)


class ResFiLMBlock(nn.Module):
    """带 GroupNorm + FiLM 条件注入的残差卷积块。"""

    def __init__(self, in_ch, out_ch, cond_dim=64):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.film = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = self.norm2(h) * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class FrameDenoiser(nn.Module):
    """CONCEPT.md 第 3.3 节：通道拼接上下文帧 + FiLM 动作/噪声条件 + EDM 预条件的 U-Net。"""

    def __init__(self, img_ch=3, ctx_len=4, act_dim=2, base_ch=32, cond_dim=64,
                 sigma_data=0.5, discrete_actions=None, use_edm=True):
        super().__init__()
        self.img_ch = img_ch
        self.ctx_len = ctx_len
        self.sigma_data = float(sigma_data)
        self.use_edm = use_edm

        self.noise_emb = nn.Sequential(SinusoidalEmb(cond_dim), nn.Linear(cond_dim, cond_dim), nn.SiLU())
        self.act_emb = ActionEmbedding(act_dim, ctx_len, cond_dim, discrete_actions=discrete_actions)
        self.ctx_noise_emb = nn.Sequential(SinusoidalEmb(cond_dim), nn.Linear(cond_dim, cond_dim), nn.SiLU())

        in_ch = (ctx_len + 1) * img_ch
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.down1 = ResFiLMBlock(base_ch, base_ch, cond_dim)
        self.pool1 = nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)         # H/2
        self.down2 = ResFiLMBlock(base_ch * 2, base_ch * 2, cond_dim)
        self.pool2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)     # H/4
        self.mid = ResFiLMBlock(base_ch * 4, base_ch * 4, cond_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1)
        self.dec2 = ResFiLMBlock(base_ch * 4, base_ch * 2, cond_dim)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1)
        self.dec1 = ResFiLMBlock(base_ch * 2, base_ch, cond_dim)
        self.out_conv = nn.Conv2d(base_ch, img_ch, 3, 1, 1)

    def _unet(self, x_cat, cond):
        h0 = self.in_conv(x_cat)
        h1 = self.down1(h0, cond)
        h2 = self.down2(self.pool1(h1), cond)
        hm = self.mid(self.pool2(h2), cond)
        u2 = self.dec2(torch.cat([self.up2(hm), h2], dim=1), cond)
        u1 = self.dec1(torch.cat([self.up1(u2), h1], dim=1), cond)
        return self.out_conv(u1)

    def forward(self, noisy_next, sigma, context_frames, actions, ctx_sigma=None):
        """
        noisy_next:     [B, C, H, W]（范围 [-1, 1] 加噪）
        sigma:          [B]
        context_frames: [B, m, C, H, W]
        actions:        [B, m, A] 或 [B, m]
        """
        B = noisy_next.shape[0]
        sigma = sigma.reshape(B)
        c_in, c_skip, c_out, c_noise, _ = edm_precond_coeffs(sigma, self.sigma_data)
        ctx_flat = context_frames.reshape(B, self.ctx_len * self.img_ch, *noisy_next.shape[2:])
        x_in = c_in.view(B, 1, 1, 1) * noisy_next
        x_cat = torch.cat([ctx_flat, x_in], dim=1)

        cond = self.noise_emb(c_noise) + self.act_emb(actions)
        if ctx_sigma is not None:
            cond = cond + self.ctx_noise_emb(ctx_sigma.reshape(B))
        net_out = self._unet(x_cat, cond)
        if self.use_edm:
            return c_skip.view(B, 1, 1, 1) * noisy_next + c_out.view(B, 1, 1, 1) * net_out
        # 朴素 DDPM ε-预测对照
        return noisy_next - sigma.view(B, 1, 1, 1) * net_out

    def loss(self, target_next, context_frames, actions, sigma=None, ctx_sigma=None):
        B = target_next.shape[0]
        device = target_next.device
        if sigma is None:
            sigma = sample_train_sigmas(B, p_mean=-0.4, p_std=1.2, device=device)
        eps = torch.randn_like(target_next)
        noisy_next = target_next + sigma.view(B, 1, 1, 1) * eps
        c_in, c_skip, c_out, c_noise, _ = edm_precond_coeffs(sigma, self.sigma_data)
        ctx_flat = context_frames.reshape(B, self.ctx_len * self.img_ch, *target_next.shape[2:])
        x_cat = torch.cat([ctx_flat, c_in.view(B, 1, 1, 1) * noisy_next], dim=1)
        cond = self.noise_emb(c_noise) + self.act_emb(actions)
        if ctx_sigma is not None:
            cond = cond + self.ctx_noise_emb(ctx_sigma.reshape(B))
        net_out = self._unet(x_cat, cond)
        if self.use_edm:
            f_target = (target_next - c_skip.view(B, 1, 1, 1) * noisy_next) / c_out.view(B, 1, 1, 1).clamp_min(1e-6)
            return F.mse_loss(net_out, f_target)
        return F.mse_loss(net_out, eps)


@torch.no_grad()
def sample_next_frame(model, context_frames, actions, n_steps=3, sigma_min=0.002, sigma_max=5.0,
                      rho=7.0, ctx_sigma=None, init_eps=None):
    """CONCEPT.md 第 3.7 节：给定上下文与动作，用 N 步 Euler 采样下一帧（范围 [-1, 1]）。"""
    model.eval()
    B, _, C, H, W = context_frames.shape
    device = context_frames.device
    sigmas = karras_sigmas(n_steps, sigma_min, sigma_max, rho, device=device)
    eps = torch.randn(B, C, H, W, device=device) if init_eps is None else init_eps.to(device)
    x = eps * sigmas[0]
    for i in range(n_steps):
        s_cur, s_next = sigmas[i], sigmas[i + 1]
        x_hat = model(x, torch.full((B,), s_cur.item(), device=device), context_frames, actions, ctx_sigma=ctx_sigma)
        x = x_hat + (s_next / s_cur) * (x - x_hat)
    return x.clamp(-1.0, 1.0)


@torch.no_grad()
def rollout(model, init_frames, action_seq, n_steps=3, ctx_sigma_val=None):
    """自回归开环展开：init_frames [B, m, C, H, W], action_seq [B, m + H - 1, ...] -> [B, H, C, H, W]。"""
    model.eval()
    buf = init_frames.clone()
    B, m = buf.shape[:2]
    H_steps = action_seq.shape[1] - m + 1
    preds = []
    ctx_sig = None if ctx_sigma_val is None else torch.full((B,), float(ctx_sigma_val), device=buf.device)
    for t in range(H_steps):
        a_win = action_seq[:, t : t + m]
        nxt = sample_next_frame(model, buf, a_win, n_steps=n_steps, ctx_sigma=ctx_sig)
        preds.append(nxt)
        buf = torch.cat([buf[:, 1:], nxt.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)


# ==================== 第二部分：PushT 序列数据与评估工具 ====================


def load_pusht_clips(n_episodes=30, img_size=32, frame_stride=2):
    """读取 PushT 演示数据，缩放至 img_size×img_size，像素归一化到 [-1, 1]，动作归一化到 [-1, 1]。"""
    demos = load_pusht_demos()
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    episodes = []
    for ep_i in range(min(n_episodes, len(starts))):
        s, e = int(starts[ep_i]), int(ends[ep_i])
        imgs_raw = demos["images"][s:e:frame_stride]
        acts_raw = demos["actions"][s:e:frame_stride]
        if len(imgs_raw) < 12:
            continue
        imgs = np.stack([cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA) for im in imgs_raw], axis=0)
        imgs = (imgs.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
        imgs = np.transpose(imgs, (0, 3, 1, 2))        # [T, 3, H, W]
        acts = (acts_raw.astype(np.float32) / 256.0) - 1.0  # [0, 512] -> [-1, 1]
        episodes.append((imgs, acts))
    return episodes


def psnr_from_m11(pred, target):
    """[-1, 1] 图像张量的 PSNR（dB）。"""
    p01 = (pred.clamp(-1.0, 1.0) + 1.0) * 0.5
    t01 = (target.clamp(-1.0, 1.0) + 1.0) * 0.5
    mse = F.mse_loss(p01, t01).item()
    return float(-10.0 * math.log10(max(mse, 1e-8)))


def sharpness_score(imgs):
    """用空间梯度幅值衡量图像清晰度（越高越锐利，发糊则下降）。"""
    dx = (imgs[..., :, 1:] - imgs[..., :, :-1]).abs().mean().item()
    dy = (imgs[..., 1:, :] - imgs[..., :-1, :]).abs().mean().item()
    return 0.5 * (dx + dy)


def verify_section_3_4_bimodal():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 3.4 节双峰分布最优去噪器（1 步模糊 vs 3 步清晰）")
    print("=" * 60)
    d_star = lambda x, s: math.tanh(x / (s * s))
    # 1 步从 σ=5, x=3.0
    one_step = d_star(3.0, 5.0)
    # 3 步日程 5 -> 1 -> 0.2 -> 0
    x = 3.0
    for s_cur, s_nxt in [(5.0, 1.0), (1.0, 0.2), (0.2, 0.0)]:
        xh = d_star(x, s_cur)
        x = xh + (s_nxt / s_cur) * (x - xh)
    print(f"  双峰 x ∈ {{-1, +1}}，起点 x_0=3.0 (σ=5)：1 步输出 = {one_step:.3f}（鬼影平均），3 步输出 = {x:.3f}（锁定 +1 峰 ✓）")
    assert abs(one_step - 0.119) < 0.01 and x > 0.99


def sample_window_batch(episodes, batch_size, ctx_len=4):
    ctx_list, act_list, tgt_list = [], [], []
    for _ in range(batch_size):
        imgs, acts = episodes[np.random.randint(len(episodes))]
        t = np.random.randint(0, len(imgs) - ctx_len)
        ctx_list.append(imgs[t : t + ctx_len])
        act_list.append(acts[t : t + ctx_len])
        tgt_list.append(imgs[t + ctx_len])
    return (
        torch.from_numpy(np.asarray(ctx_list)),
        torch.from_numpy(np.asarray(act_list)),
        torch.from_numpy(np.asarray(tgt_list)),
    )


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()
    verify_section_3_4_bimodal()

    print("\n" + "=" * 60)
    print("第二部分：PushT 32×32 动作条件下一帧扩散模型训练与评估")
    print("=" * 60)
    ctx_len = 4
    episodes = load_pusht_clips(n_episodes=8 if quick else 32, img_size=32, frame_stride=2)
    n_tr = max(2, int(len(episodes) * 0.8))
    tr_eps, te_eps = episodes[:n_tr], episodes[n_tr:]
    sigma_data = float(np.concatenate([e[0] for e in tr_eps[:4]], axis=0).std())
    print(f"  数据划分：训练 {len(tr_eps)} 回合，测试 {len(te_eps)} 回合（32×32 RGB，σ_data = {sigma_data:.3f}）")

    model = FrameDenoiser(img_ch=3, ctx_len=ctx_len, act_dim=2, base_ch=32, cond_dim=64, sigma_data=sigma_data)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_iters = 80 if quick else 700
    t_tr = time.time()
    for it in range(1, n_iters + 1):
        ctx, act, tgt = sample_window_batch(tr_eps, batch_size=32, ctx_len=ctx_len)
        opt.zero_grad()
        loss = model.loss(tgt, ctx, act)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it in {n_iters // 2, n_iters}:
            print(f"  [Iter {it:4d}/{n_iters}] EDM loss = {loss.item():.4f} ({time.time() - t_tr:.1f}s)")

    # 评估不同去噪步数 N ∈ {1, 2, 3, 5, 10}
    te_ctx, te_act, te_tgt = sample_window_batch(te_eps, batch_size=24, ctx_len=ctx_len)
    init_eps = torch.randn_like(te_tgt)
    real_sharp = sharpness_score(te_tgt)
    print(f"\n  测试集下一帧预测对比（真实帧清晰度 = {real_sharp:.4f}）：")
    print(f"  {'步数 N':>6s} | {'PSNR (dB)':>10s} | {'清晰度梯度':>10s} | {'单帧耗时(ms)':>12s}")
    step_imgs = {}
    for n in [1, 2, 3, 5, 10]:
        ts = time.time()
        pred = sample_next_frame(model, te_ctx, te_act, n_steps=n, init_eps=init_eps)
        ms = (time.time() - ts) * 1000.0 / len(te_ctx)
        ps = psnr_from_m11(pred, te_tgt)
        sh = sharpness_score(pred)
        step_imgs[n] = pred[0]
        print(f"  {n:>6d} | {ps:10.2f} | {sh:10.4f} | {ms:12.2f}")

    # 动作可控性检查（真实动作 vs 随机动作）
    rand_act = torch.rand_like(te_act) * 2.0 - 1.0
    pred_true = sample_next_frame(model, te_ctx, te_act, n_steps=3, init_eps=init_eps)
    pred_rand = sample_next_frame(model, te_ctx, rand_act, n_steps=3, init_eps=init_eps)
    psnr_true = psnr_from_m11(pred_true, te_tgt)
    psnr_rand = psnr_from_m11(pred_rand, te_tgt)
    print(f"\n  动作可控性检查（3 步去噪）：真实动作 PSNR = {psnr_true:.2f} dB，随机动作 PSNR = {psnr_rand:.2f} dB，ΔPSNR = {psnr_true - psnr_rand:+.2f} dB")

    # 8 步开环展开
    imgs0, acts0 = te_eps[0]
    H_roll = min(8, len(imgs0) - ctx_len)
    init_f = torch.from_numpy(imgs0[:ctx_len]).unsqueeze(0)
    act_s = torch.from_numpy(acts0[: ctx_len + H_roll - 1]).unsqueeze(0)
    gt_roll = torch.from_numpy(imgs0[ctx_len : ctx_len + H_roll])
    pred_roll = rollout(model, init_f, act_s, n_steps=3)[0]
    roll_psnrs = [psnr_from_m11(pred_roll[k], gt_roll[k]) for k in range(H_roll)]
    print(f"  开环展开 {H_roll} 步 PSNR 曲线：{[round(v, 2) for v in roll_psnrs]}")

    # 绘图
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    axes[0, 0].imshow(((te_tgt[0].permute(1, 2, 0).numpy() + 1) * 0.5).clip(0, 1))
    axes[0, 0].set_title("真实下一帧"); axes[0, 0].axis("off")
    for idx, n in enumerate([1, 2, 3, 5, 10], start=1):
        axes[0, idx].imshow(((step_imgs[n].permute(1, 2, 0).numpy() + 1) * 0.5).clip(0, 1))
        axes[0, idx].set_title(f"N={n} 步"); axes[0, idx].axis("off")

    for col in range(min(6, H_roll)):
        axes[1, col].imshow(((pred_roll[col].permute(1, 2, 0).numpy() + 1) * 0.5).clip(0, 1))
        axes[1, col].set_title(f"展开 t+{col+1} ({roll_psnrs[col]:.1f}dB)"); axes[1, col].axis("off")
    fig.suptitle("PushT 动作条件下一帧扩散：步数对比（上）与 3 步去噪开环展开（下）")
    fig.tight_layout()
    save_figure(fig, OUT / "03-action-conditioned-diffusion.png")

    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：04-diamond.py")


if __name__ == "__main__":
    main()
