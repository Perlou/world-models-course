"""
FSQ（有限标量量化）与 Tokenizer 质量天花板诊断
==============================================

学习目标：
    1. 核对 CONCEPT.md 第 2.4 节 FSQ 手算（L=(5,3), z=(0.8,-1.5) → ẑ=(1,-1), index=3, 梯度系数 (1.118, 0.181)）
       及第 2.6 节小物体 MSE/PSNR 陷阱（2×2 白球丢掉，PSNR 仍高达 30.1 dB！）
    2. 实现 FSQ、codes_to_indices()、indices_to_codes() 与 reconstruction_error_map()
    3. 在 PushT 画面上同码本规模（N≈125–128）对比 VQ(梯度)、VQ(EMA) 与 FSQ 的重建误差、利用率与困惑度
    4. 对比 PushT 在 4×4 (K=16) 与 8×8 (K=64) 两种 token 粒度下的逐像素误差热力图，
       并训练线性/MLP 探针从 token 里恢复推杆位置与 T 形块相关特征，保存 4×4 FSQ tokenizer 供 03、04 节复用

核心概念：
    - FSQ（Finite Scalar Quantization）：把连续低维向量经 tanh 限幅后逐维取整到固定整数网格，无码本、零坍塌
    - 重建误差热力图与任务变量探针：PSNR 掩盖小物体丢失，必须定位误差落在画面的什么位置

前置知识：
    - 01-vq-vae-from-scratch.py

对应概念文档：
    - CONCEPT.md 第 2 节「FSQ 与 tokenizer 的质量」

预计运行时间：约 2 分钟（CPU）
"""

import importlib.util
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.pusht import load_pusht_demos

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec01 = importlib.util.spec_from_file_location("p8_01", os.path.join(_DIR, "01-vq-vae-from-scratch.py"))
_mod01 = importlib.util.module_from_spec(_spec01)
_spec01.loader.exec_module(_mod01)

SmallVQVAE = _mod01.SmallVQVAE
codebook_usage = _mod01.codebook_usage
vq_loss = _mod01.vq_loss

plt = setup_matplotlib()
OUT = output_dir("phase-08")
SEED = 0


# ==================== 第一部分：FSQ 核心实现 ====================


def codes_to_indices(codes, levels):
    """把整数网格坐标 codes [..., d]（范围 [-half, +half]）经混合进制编码成 0..N-1 整数索引。"""
    lv = torch.as_tensor(levels, device=codes.device, dtype=torch.long)
    half = (lv - 1) // 2
    basis = torch.cumprod(torch.cat([torch.ones(1, device=codes.device, dtype=torch.long), lv[:-1]]), dim=0)
    digits = codes.long() + half
    return (digits * basis).sum(dim=-1)


def indices_to_codes(indices, levels):
    """把整数索引还原为 [-half, +half] 整数网格坐标 [..., d]。"""
    lv = torch.as_tensor(levels, device=indices.device, dtype=torch.long)
    half = (lv - 1) // 2
    rem = indices.long()
    cols = []
    for L_i in lv.tolist():
        cols.append(rem % L_i)
        rem = rem // L_i
    digits = torch.stack(cols, dim=-1)
    return (digits - half).float()


class FSQ(nn.Module):
    """CONCEPT.md 第 2.7–2.8 节：奇数级数有限标量量化器，无任何可训练码本参数。"""

    def __init__(self, levels=(5, 5, 5)):
        super().__init__()
        assert all(L % 2 == 1 for L in levels), "本讲义使用奇数级数以保持零点对称"
        self.levels = tuple(int(L) for L in levels)
        self.dim = len(self.levels)
        self.num_codes = int(math.prod(self.levels))
        lv = torch.tensor(self.levels, dtype=torch.float32)
        self.register_buffer("half_width", (lv - 1.0) / 2.0)
        basis = torch.cumprod(torch.cat([torch.ones(1, dtype=torch.long), torch.tensor(self.levels[:-1], dtype=torch.long)]), dim=0)
        self.register_buffer("basis", basis)

    def forward(self, z):
        # z: [..., d]
        bounded = torch.tanh(z) * self.half_width
        rounded = torch.round(bounded)
        z_q_st = bounded + (rounded - bounded).detach()
        idx = ((rounded.long() + self.half_width.long()) * self.basis).sum(dim=-1)
        # 归一化到 [-1, 1] 送进解码器，让各维数值尺度一致
        z_q_norm = z_q_st / self.half_width
        return z_q_norm, idx

    def decode_indices(self, idx):
        return indices_to_codes(idx, self.levels) / self.half_width


class FSQTokenizer(nn.Module):
    """支持 grid='4x4' (K=16) 或 '8x8' (K=64) 的 PushT FSQ Tokenizer。"""

    def __init__(self, levels=(5, 5, 5), grid="4x4"):
        super().__init__()
        self.grid = grid
        self.fsq = FSQ(levels)
        d = self.fsq.dim
        if grid == "4x4":
            self.enc = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),   # 32
                nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),  # 16
                nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(),  # 8
                nn.Conv2d(64, d, 4, 2, 1),              # 4
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(d, 64, 4, 2, 1), nn.SiLU(),
                nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
            )
        else:
            self.enc = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),   # 32
                nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),  # 16
                nn.Conv2d(64, d, 4, 2, 1),              # 8
            )
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(d, 64, 4, 2, 1), nn.SiLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
            )

    def encode(self, obs):
        z = self.enc(obs).permute(0, 2, 3, 1)
        z_q, idx = self.fsq(z)
        return z_q, idx

    def decode_tokens(self, idx):
        # idx: [B, h, w]
        z_q = self.fsq.decode_indices(idx).permute(0, 3, 1, 2)
        return self.dec(z_q)

    def forward(self, obs):
        z_q, idx = self.encode(obs)
        recon = self.dec(z_q.permute(0, 3, 1, 2))
        return recon, idx


@torch.no_grad()
def reconstruction_error_map(model, obs_batch):
    """CONCEPT.md 第 2.6 节：逐像素平均绝对误差热力图 [64, 64]。"""
    out = model(obs_batch)
    recon = out[0]
    err = (recon - obs_batch).abs().mean(dim=(0, 1)).cpu().numpy()
    return err


# ==================== 第二部分：手算验证 ====================


def verify_section_2_hand_calcs():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 2.4 节 FSQ 手算与第 2.6 节小物体 PSNR 陷阱")
    print("=" * 60)
    fsq = FSQ(levels=(5, 3))
    z = torch.tensor([[0.8, -1.5]], requires_grad=True)
    bounded = torch.tanh(z) * fsq.half_width
    rounded = torch.round(bounded)
    idx = codes_to_indices(rounded, (5, 3))
    back = indices_to_codes(idx, (5, 3))
    quant_err = (bounded - rounded).pow(2).sum().item()
    # 核对 tanh 梯度系数
    z_q_st = bounded + (rounded - bounded).detach()
    z_q_st.sum().backward()
    print(f"  1. FSQ (L=(5,3), z=(0.8, -1.5))：ẑ = {[int(x) for x in rounded[0].tolist()]}，"
          f"index = {idx.item()}（手算 3），还原 = {[int(x) for x in back[0].tolist()]}")
    print(f"     量化误差 = {quant_err:.3f} (≤ d/4 = 0.5) ｜ 梯度系数 = {[round(x, 3) for x in z.grad[0].tolist()]}（手算 [1.118, 0.181] ✓）")
    assert idx.item() == 3 and torch.allclose(back[0], torch.tensor([1.0, -1.0]))
    assert torch.allclose(z.grad[0], torch.tensor([1.118, 0.181]), atol=2e-3)

    mse_ball = 4.0 / (64.0 * 64.0)
    psnr_ball = 10.0 * math.log10(1.0 / mse_ball)
    print(f"  2. 小物体陷阱（64×64 丢掉 2×2 白球）：MSE = {mse_ball:.5f}，PSNR = {psnr_ball:.1f} dB（手算 30.1 dB ✓）")
    assert abs(psnr_ball - 30.1) < 0.1


# ==================== 第三部分：VQ vs FSQ 对比 & PushT 粒度/探针诊断 ====================


def load_pusht_with_states(max_frames=1800):
    demos = load_pusht_demos()
    imgs = demos["images"]
    states = demos["states"]  # [N, 2] 推杆像素坐标 0-512
    step = max(1, len(imgs) // max_frames)
    sub_i = imgs[::step][:max_frames]
    sub_s = states[::step][:max_frames].astype(np.float32) / 512.0
    resized = np.stack([cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA) for im in sub_i], axis=0)
    x = torch.from_numpy(resized).permute(0, 3, 1, 2).float() / 255.0
    y = torch.from_numpy(sub_s)
    return x, y


def probe_position_rmse(model, tr_x, tr_y, val_x, val_y, is_fsq=True):
    """冻结 tokenizer，从展平的量化特征 z_q 训练小型 MLP 探针回归推杆坐标（还原到 512×512 像素误差）。"""
    model.eval()
    with torch.no_grad():
        if is_fsq:
            zq_tr, _ = model.encode(tr_x)
            zq_val, _ = model.encode(val_x)
        else:
            zq_tr, _, _ = model.vq(model.enc(tr_x).permute(0, 2, 3, 1))
            zq_val, _, _ = model.vq(model.enc(val_x).permute(0, 2, 3, 1))
        f_tr = zq_tr.reshape(len(tr_x), -1)
        f_val = zq_val.reshape(len(val_x), -1)

    probe = nn.Sequential(nn.Linear(f_tr.shape[1], 64), nn.ReLU(), nn.Linear(64, 2))
    opt = torch.optim.Adam(probe.parameters(), lr=5e-3)
    for _ in range(120):
        pred = probe(f_tr)
        loss = F.mse_loss(pred, tr_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_val = probe(f_val)
        px_err = ((pred_val - val_y) * 512.0).pow(2).sum(dim=-1).sqrt().mean().item()
    return px_err


def main():
    seed_everything(SEED)
    verify_section_2_hand_calcs()

    quick = is_quick()
    x_all, y_all = load_pusht_with_states(max_frames=360 if quick else 1800)
    n_tr = int(len(x_all) * 0.85)
    tr_x, val_x = x_all[:n_tr], x_all[n_tr:]
    tr_y, val_y = y_all[:n_tr], y_all[n_tr:]
    n_steps = 40 if quick else 240

    print("\n" + "=" * 60)
    print(f"第二部分：同码本规模（N≈125–128, 4×4）对比 VQ(梯度) / VQ(EMA) / FSQ，以及 4×4 vs 8×8 粒度")
    print("=" * 60)
    t0 = time.time()

    models = [
        ("VQ-梯度 (N=128, 4×4)", SmallVQVAE(num_codes=128, dim=32, mode="data_init"), 128, False),
        ("VQ-EMA  (N=128, 4×4)", SmallVQVAE(num_codes=128, dim=32, mode="ema"), 128, False),
        ("FSQ     (N=125, 4×4)", FSQTokenizer(levels=(5, 5, 5), grid="4x4"), 125, True),
        ("FSQ     (N=125, 8×8)", FSQTokenizer(levels=(5, 5, 5), grid="8x8"), 125, True),
    ]

    summary = []
    err_maps = {}
    fsq_4x4 = None

    for name, model, n_codes, is_fsq in models:
        seed_everything(SEED)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        for _ in range(n_steps):
            model.train()
            idx = torch.randint(0, len(tr_x), (32,))
            if is_fsq:
                recon, _ = model(tr_x[idx])
                loss = F.mse_loss(recon, tr_x[idx])
            else:
                recon, _, ql = model(tr_x[idx])
                loss, _ = vq_loss(recon, tr_x[idx], ql)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out = model(val_x)
            rec_v, id_v = out[0], out[1]
            mse_v = F.mse_loss(rec_v, val_x).item()
            psnr_v = 10.0 * math.log10(1.0 / max(mse_v, 1e-8))
            util, ppl = codebook_usage(id_v, n_codes)
        px_err = probe_position_rmse(model, tr_x, tr_y, val_x, val_y, is_fsq=is_fsq)
        summary.append((name, mse_v, psnr_v, util * 100.0, ppl, px_err))
        if is_fsq:
            err_maps[name] = reconstruction_error_map(model, val_x)
        if "4×4" in name and is_fsq:
            fsq_4x4 = model
        print(f"  {name:<22} ｜ MSE {mse_v:.5f} (PSNR {psnr_v:4.1f} dB) ｜ 利用率 {util*100:5.1f}% ｜ "
              f"PPL {ppl:5.1f}/{n_codes} ｜ 推杆位置探针误差 {px_err:4.1f} px")

    if fsq_4x4 is not None and not quick:
        torch.save(fsq_4x4.state_dict(), OUT / "pusht_fsq_4x4.pt")
        print(f"\n  💾 已保存 PushT 4×4 FSQ Tokenizer 至 {OUT / 'pusht_fsq_4x4.pt'}（供 03、04 节复用）")
    print(f"  总耗时 {time.time() - t0:.1f}s")

    # 图 1：VQ vs FSQ 指标柱状图
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    names = [s[0].replace(" (", "\n(") for s in summary]
    axes[0].bar(names, [s[2] for s in summary], color=["C0", "C1", "seagreen", "navy"], width=0.55)
    axes[0].set_ylabel("PSNR (dB)"); axes[0].set_title("重建 PSNR（越高越好）")
    axes[1].bar(names, [s[4] for s in summary], color=["C0", "C1", "seagreen", "navy"], width=0.55)
    axes[1].set_ylabel("困惑度 PPL"); axes[1].set_title("有效使用码数（困惑度）")
    axes[2].bar(names, [s[5] for s in summary], color=["C0", "C1", "seagreen", "navy"], width=0.55)
    axes[2].set_ylabel("位置回归平均像素误差 (px / 512)"); axes[2].set_title("任务变量探针误差（越低越好）")
    save_figure(fig, OUT / "02-vq-vs-fsq.png")

    # 图 2：4×4 vs 8×8 重建图与误差热力图
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
    sample_img = val_x[0:1]
    fsq_models = [(m[0], m[1]) for m in models if m[3]]
    for row, (nm, mod) in enumerate(fsq_models):
        with torch.no_grad():
            rec0 = mod(sample_img)[0][0].permute(1, 2, 0).numpy()
        axes[row, 0].imshow(sample_img[0].permute(1, 2, 0).numpy()); axes[row, 0].set_title("原始 PushT 画面"); axes[row, 0].axis("off")
        axes[row, 1].imshow(np.clip(rec0, 0, 1)); axes[row, 1].set_title(f"{nm} 重建"); axes[row, 1].axis("off")
        im = axes[row, 2].imshow(err_maps[nm], cmap="hot", vmin=0, vmax=max(m.max() for m in err_maps.values()))
        axes[row, 2].set_title(f"{nm} 逐像素误差热力图"); axes[row, 2].axis("off")
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046)
    save_figure(fig, OUT / "02-pusht-error-heatmap.png")

    print("\n" + "=" * 60)
    print("本节完成！下一节：03-gpt-as-world-model.py（把 minGPT 改成帧 token 与动作交错的世界模型）")
    print("=" * 60)


if __name__ == "__main__":
    main()
