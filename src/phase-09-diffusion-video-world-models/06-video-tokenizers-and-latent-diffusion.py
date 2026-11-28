"""
Phase 9 · 第 06 课：视频 Tokenizer 与隐空间扩散世界模型
======================================================

学习目标：
    1. 实现因果视频自编码器 CausalVideoAutoencoder（连续隐变量 + 轻微 KL/L2 正则）
    2. 核对 CONCEPT.md 第 6.3 节压缩率与隐变量标准差 σ_z 归一化，测试高压缩率下
       PushT 小圆点（推杆）是否仍能被重建保留
    3. 先编码并冻结 tokenizer，在连续隐空间训练 LatentFrameDenoiser（EDM 预条件）
    4. 在同一份 PushT 演示数据上对比「像素空间扩散 (32×32×3)」与「隐空间扩散 (8×8×4)」的
       单步耗时、PSNR 与开环预测质量

对应概念文档：
    src/phase-09-diffusion-video-world-models/CONCEPT.md 第 6 节

运行方式：
    uv run python src/phase-09-diffusion-video-world-models/06-video-tokenizers-and-latent-diffusion.py
    WM_QUICK=1 uv run python src/phase-09-diffusion-video-world-models/06-video-tokenizers-and-latent-diffusion.py

预计运行时间：完整约 2 分钟（M1 Max CPU），冒烟约 12 秒
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
_spec03 = importlib.util.spec_from_file_location("p9_03", os.path.join(_DIR, "03-action-conditioned-frame-diffusion.py"))
_mod03 = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(_mod03)

FrameDenoiser = _mod03.FrameDenoiser
sample_next_frame = _mod03.sample_next_frame
load_pusht_clips = _mod03.load_pusht_clips
sample_window_batch = _mod03.sample_window_batch
psnr_from_m11 = _mod03.psnr_from_m11

plt = setup_matplotlib()
OUT = output_dir("phase-09")
SEED = 0


# ==================== 第一部分：因果视频自编码器与隐空间去噪器 ====================


class CausalVideoAutoencoder(nn.Module):
    """CONCEPT.md 第 6.2 节：带因果 1D 时间卷积的轻量连续视频 tokenizer（空间压缩 f_s ∈ {4, 8}）。"""

    def __init__(self, latent_ch=4, f_s=4):
        super().__init__()
        self.f_s = f_s
        self.latent_ch = latent_ch
        if f_s == 4:
            self.s_enc = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),       # 16×16
                nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),      # 8×8
                nn.Conv2d(64, latent_ch, 3, 1, 1),
            )
            self.s_dec = nn.Sequential(
                nn.Conv2d(latent_ch, 64, 3, 1, 1), nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
            )
        else:
            # 高压缩率 f_s=8：32×32 → 4×4×latent_ch
            self.s_enc = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),       # 16
                nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),      # 8
                nn.Conv2d(64, latent_ch, 4, 2, 1),          # 4
            )
            self.s_dec = nn.Sequential(
                nn.ConvTranspose2d(latent_ch, 64, 4, 2, 1), nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
            )
        # 因果时间混合：只看当前帧与前一帧（padding 放在左侧），确保不泄露未来
        self.t_enc = nn.Conv1d(latent_ch, latent_ch, kernel_size=2, padding=0)
        self.t_dec = nn.Conv1d(latent_ch, latent_ch, kernel_size=2, padding=0)

    def encode(self, video):
        # video: [B, T, 3, H, W] -> [B, T, C_z, h, w]
        B, T, C, H, W = video.shape
        z = self.s_enc(video.reshape(B * T, C, H, W))
        _, Cz, h, w = z.shape
        z_seq = z.reshape(B, T, Cz, h, w).permute(0, 3, 4, 2, 1).reshape(B * h * w, Cz, T)
        z_pad = F.pad(z_seq, (1, 0), mode="replicate")  # 因果左填充
        z_out = z_seq + 0.2 * self.t_enc(z_pad)
        return z_out.reshape(B, h, w, Cz, T).permute(0, 4, 3, 1, 2)

    def decode(self, latents):
        B, T, Cz, h, w = latents.shape
        z_seq = latents.permute(0, 3, 4, 2, 1).reshape(B * h * w, Cz, T)
        z_pad = F.pad(z_seq, (1, 0), mode="replicate")
        z_out = z_seq + 0.2 * self.t_dec(z_pad)
        z_2d = z_out.reshape(B, h, w, Cz, T).permute(0, 4, 3, 1, 2).reshape(B * T, Cz, h, w)
        return self.s_dec(z_2d).reshape(B, T, 3, h * self.f_s, w * self.f_s)

    def forward(self, video):
        z = self.encode(video)
        return self.decode(z), z


def tokenizer_loss(model, video, reg_weight=1e-4):
    recon, z = model(video)
    return F.mse_loss(recon, video) + reg_weight * (z ** 2).mean(), recon, z


@torch.no_grad()
def estimate_latent_std(tokenizer, episodes, n_eps=8):
    tokenizer.eval()
    zs = []
    for imgs, _ in episodes[:n_eps]:
        v = torch.from_numpy(imgs[:16]).unsqueeze(0)
        zs.append(tokenizer.encode(v).reshape(-1))
    return float(torch.cat(zs).std().item())


@torch.no_grad()
def encode_dataset(tokenizer, episodes, sigma_z):
    tokenizer.eval()
    out = []
    for imgs, acts in episodes:
        v = torch.from_numpy(imgs).unsqueeze(0)
        z = (tokenizer.encode(v)[0] / max(sigma_z, 1e-4)).numpy()
        out.append((z, acts))
    return out


def pusher_preservation_score(orig_m11, recon_m11):
    """PushT 推杆是偏蓝色小圆点：用 (B - R) 通道高值区域的误差衡量小物体是否保留。"""
    o01 = (orig_m11.clamp(-1, 1) + 1.0) * 0.5
    r01 = (recon_m11.clamp(-1, 1) + 1.0) * 0.5
    blue_mask = ((o01[..., 2, :, :] - o01[..., 0, :, :]) > 0.15).float()
    if blue_mask.sum() < 1:
        return 0.0
    err = ((o01 - r01).abs().mean(dim=-3) * blue_mask).sum() / blue_mask.sum()
    return float(err.item())


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    print("=" * 60)
    print("第一部分：训练 f_s=4 与 f_s=8 两种因果视频 Tokenizer 并检验因果性")
    print("=" * 60)
    episodes = load_pusht_clips(n_episodes=8 if quick else 32, img_size=32, frame_stride=2)
    n_tr = max(2, int(len(episodes) * 0.8))
    tr_eps, te_eps = episodes[:n_tr], episodes[n_tr:]

    tok4 = CausalVideoAutoencoder(latent_ch=4, f_s=4)
    tok8 = CausalVideoAutoencoder(latent_ch=4, f_s=8)
    opt4 = torch.optim.Adam(tok4.parameters(), lr=2e-3)
    opt8 = torch.optim.Adam(tok8.parameters(), lr=2e-3)

    # 验证因果性：修改未来帧不改变过去帧的编码
    dummy = torch.randn(1, 5, 3, 32, 32)
    dummy2 = dummy.clone(); dummy2[:, 3:] = torch.randn_like(dummy2[:, 3:])
    with torch.no_grad():
        diff_past = (tok4.encode(dummy)[:, :3] - tok4.encode(dummy2)[:, :3]).abs().max().item()
    assert diff_past < 1e-6, f"因果性违背: {diff_past}"
    print(f"  因果性单元测试通过（修改未来帧对过去隐变量最大影响 = {diff_past:.2e} ✓）")

    def sample_vid(eps, B=12, T=6):
        vs = []
        for _ in range(B):
            imgs, _ = eps[np.random.randint(len(eps))]
            s = np.random.randint(0, len(imgs) - T)
            vs.append(imgs[s : s + T])
        return torch.from_numpy(np.asarray(vs))

    tok_iters = 80 if quick else 450
    for _ in range(tok_iters):
        v = sample_vid(tr_eps, B=12, T=6)
        opt4.zero_grad(); l4, _, _ = tokenizer_loss(tok4, v); l4.backward(); opt4.step()
        opt8.zero_grad(); l8, _, _ = tokenizer_loss(tok8, v); l8.backward(); opt8.step()

    te_v = sample_vid(te_eps, B=12, T=6)
    with torch.no_grad():
        rec4, _ = tok4(te_v)
        rec8, _ = tok8(te_v)
    psnr4 = psnr_from_m11(rec4, te_v)
    psnr8 = psnr_from_m11(rec8, te_v)
    dot4 = pusher_preservation_score(te_v, rec4)
    dot8 = pusher_preservation_score(te_v, rec8)
    sigma_z = estimate_latent_std(tok4, tr_eps)
    print(f"  f_s=4 (32×32×3 → 8×8×4，12× 压缩)：重建 PSNR = {psnr4:.2f} dB，推杆蓝点 L1 误差 = {dot4:.4f}，σ_z = {sigma_z:.3f}")
    print(f"  f_s=8 (32×32×3 → 4×4×4，48× 压缩)：重建 PSNR = {psnr8:.2f} dB，推杆蓝点 L1 误差 = {dot8:.4f}")

    print("\n" + "=" * 60)
    print("第二部分：像素空间扩散 vs 隐空间扩散（f_s=4）对比")
    print("=" * 60)
    ctx_len = 3
    tr_lat = encode_dataset(tok4, tr_eps, sigma_z)
    te_lat = encode_dataset(tok4, te_eps, sigma_z)
    sigma_pix = float(np.concatenate([e[0] for e in tr_eps[:4]], axis=0).std())

    pix_wm = FrameDenoiser(img_ch=3, ctx_len=ctx_len, act_dim=2, base_ch=24, cond_dim=48, sigma_data=sigma_pix)
    lat_wm = FrameDenoiser(img_ch=4, ctx_len=ctx_len, act_dim=2, base_ch=24, cond_dim=48, sigma_data=1.0)
    opt_p = torch.optim.Adam(pix_wm.parameters(), lr=1e-3)
    opt_l = torch.optim.Adam(lat_wm.parameters(), lr=1e-3)

    wm_iters = 70 if quick else 450
    t_tr_p, t_tr_l = 0.0, 0.0
    for _ in range(wm_iters):
        t1 = time.time()
        ctx_p, act_p, tgt_p = sample_window_batch(tr_eps, 24, ctx_len=ctx_len)
        opt_p.zero_grad(); pix_wm.loss(tgt_p, ctx_p, act_p).backward(); opt_p.step()
        t_tr_p += time.time() - t1

        t2 = time.time()
        ctx_l, act_l, tgt_l = sample_window_batch(tr_lat, 24, ctx_len=ctx_len)
        opt_l.zero_grad(); lat_wm.loss(tgt_l, ctx_l, act_l).backward(); opt_l.step()
        t_tr_l += time.time() - t2

    # 在相同测试窗口上测速与测 PSNR
    te_ctx_p, te_act, te_tgt_p = sample_window_batch(te_eps, 16, ctx_len=ctx_len)
    with torch.no_grad():
        te_ctx_l = tok4.encode(te_ctx_p) / max(sigma_z, 1e-4)

    ts = time.time()
    pred_p = sample_next_frame(pix_wm, te_ctx_p, te_act, n_steps=3)
    ms_p = (time.time() - ts) * 1000.0 / len(te_ctx_p)

    ts = time.time()
    pred_z = sample_next_frame(lat_wm, te_ctx_l, te_act, n_steps=3)
    with torch.no_grad():
        seq_z = torch.cat([te_ctx_l, pred_z.unsqueeze(1)], dim=1) * sigma_z
        pred_l_img = tok4.decode(seq_z)[:, -1]
    ms_l = (time.time() - ts) * 1000.0 / len(te_ctx_p)

    psnr_pix = psnr_from_m11(pred_p, te_tgt_p)
    psnr_lat = psnr_from_m11(pred_l_img, te_tgt_p)
    print(f"  像素扩散 (32×32×3)：训练耗时 {t_tr_p:.1f}s | 单帧推理 {ms_p:.2f} ms | 下一帧 PSNR = {psnr_pix:.2f} dB")
    print(f"  隐空间扩散 (8×8×4)：训练耗时 {t_tr_l:.1f}s | 单帧推理 {ms_l:.2f} ms | 下一帧 PSNR = {psnr_lat:.2f} dB（上限 {psnr4:.2f} dB）")
    print(f"  速度比：训练加速 {t_tr_p / max(t_tr_l, 1e-3):.2f}×，单帧推理加速 {ms_p / max(ms_l, 1e-3):.2f}×")

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for col, (title, im) in enumerate([
        ("原始帧", te_v[0, 0]),
        (f"Tokenizer f_s=4 ({psnr4:.1f}dB)", rec4[0, 0]),
        (f"Tokenizer f_s=8 ({psnr8:.1f}dB)", rec8[0, 0]),
        ("真实下一帧", te_tgt_p[0]),
    ]):
        axes[0, col].imshow(((im.permute(1, 2, 0).numpy() + 1) * 0.5).clip(0, 1))
        axes[0, col].set_title(title); axes[0, col].axis("off")

    axes[1, 0].imshow(((pred_p[0].permute(1, 2, 0).numpy() + 1) * 0.5).clip(0, 1))
    axes[1, 0].set_title(f"像素扩散 ({psnr_pix:.1f}dB, {ms_p:.1f}ms)"); axes[1, 0].axis("off")
    axes[1, 1].imshow(((pred_l_img[0].permute(1, 2, 0).numpy() + 1) * 0.5).clip(0, 1))
    axes[1, 1].set_title(f"隐空间扩散 ({psnr_lat:.1f}dB, {ms_l:.1f}ms)"); axes[1, 1].axis("off")
    axes[1, 2].bar(["像素扩散", "隐空间扩散"], [ms_p, ms_l], color=["tab:blue", "tab:green"])
    axes[1, 2].set_title("单帧推理延迟 (ms)"); axes[1, 2].set_ylabel("ms")
    axes[1, 3].bar(["f_s=4", "f_s=8"], [dot4, dot8], color=["tab:green", "tab:red"])
    axes[1, 3].set_title("推杆小圆点重建误差 (越低越好)")

    fig.tight_layout()
    save_figure(fig, OUT / "06-video-tokenizer-and-latent-diffusion.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：07-latent-action-models.py")


if __name__ == "__main__":
    main()
