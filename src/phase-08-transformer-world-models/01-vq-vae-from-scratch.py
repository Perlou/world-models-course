"""
VQ-VAE 从零实现：码本、三项损失、Straight-Through 与码本坍塌诊断
================================================================

学习目标：
    1. 核对 CONCEPT.md 第 1.4–1.6 节的全部手算例子：
       - 最近邻查表、三项损失（0.16 + 0.10 + 0.025 = 0.285）与三路精确梯度（z_e: (-1.45, -0.85), e_4: (-0.6, 0.2)）
       - 码本利用率与困惑度手算（4.00, 3.36, 1.99, 1.00）
       - EMA 码本更新手算（N_j = 9.2, e_j = (0.504, 0.500)）
    2. 实现 VectorQuantizer（支持随机初始化、数据初始化、死码重置、EMA 四种配置）、vq_loss() 与 codebook_usage()
    3. 在真实的 PushT 推块画面（utils.pusht.load_pusht_demos，缩放到 64×64）上训练 VQ-VAE，
       对比四种配置下码本利用率与困惑度随训练步数的演进曲线，亲手复现并解决码本坍塌

核心概念：
    - 矢量量化（Vector Quantization）：把连续特征图的每个空间位置映射到码本里最近的离散码向量
    - 三项损失梯度分工：重建项经 straight-through 训 E/D；码本项只动 C；commitment 项只把 E 拉向 C
    - 利用率（utilization）与困惑度（perplexity = exp(H)）：分别诊断死码与使用均匀度

前置知识：
    - Phase 6 的 straight-through 梯度估计器

对应概念文档：
    - CONCEPT.md 第 1 节「把画面变成 token：VQ-VAE」

预计运行时间：约 1.5 分钟（CPU）
"""

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

plt = setup_matplotlib()
OUT = output_dir("phase-08")
SEED = 0


# ==================== 第一部分：码本诊断与 VectorQuantizer ====================


def codebook_usage(indices, num_codes):
    """CONCEPT.md 第 1.5 节：统计 token 索引的码本利用率（0–1）与困惑度（1–N）。"""
    flat = indices.reshape(-1).long()
    counts = torch.bincount(flat, minlength=num_codes).float()
    total = counts.sum().clamp_min(1.0)
    probs = counts / total
    utilization = float((counts > 0).float().mean().item())
    nz = probs[probs > 0]
    entropy = float(-(nz * torch.log(nz)).sum().item())
    perplexity = float(math.exp(entropy))
    return utilization, perplexity


class VectorQuantizer(nn.Module):
    """支持标准梯度更新、数据初始化、死码重置与 EMA 更新四种模式的矢量量化器。"""

    def __init__(self, num_codes=128, dim=32, beta=0.25, mode="grad", ema_decay=0.99, reset_every=25):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.mode = mode  # "random", "data_init", "dead_reset", "ema"
        self.ema_decay = ema_decay
        self.reset_every = reset_every
        self.step_count = 0
        self.initialized = (mode == "random")

        self.codebook = nn.Embedding(num_codes, dim)
        if mode == "random":
            # 刻意用较宽的随机初始化（放大经典码本坍塌现象）
            nn.init.uniform_(self.codebook.weight, -2.5, 2.5)
        else:
            nn.init.uniform_(self.codebook.weight, -0.5, 0.5)

        self.register_buffer("ema_count", torch.ones(num_codes))
        self.register_buffer("ema_sum", self.codebook.weight.detach().clone())
        self.register_buffer("usage_acc", torch.zeros(num_codes))

    @torch.no_grad()
    def _data_init(self, flat):
        n = flat.shape[0]
        idx = torch.randint(0, n, (self.num_codes,), device=flat.device)
        chosen = flat[idx] + 1e-3 * torch.randn(self.num_codes, self.dim, device=flat.device)
        self.codebook.weight.copy_(chosen)
        self.ema_count.fill_(1.0)
        self.ema_sum.copy_(chosen)
        self.initialized = True

    def forward(self, z_e):
        # z_e: [B, h, w, d]
        flat = z_e.reshape(-1, self.dim)
        if self.training and not self.initialized:
            self._data_init(flat.detach())

        # 计算平方距离 ||z_e - e_j||^2
        dist = torch.cdist(flat, self.codebook.weight, p=2).pow(2)
        idx = dist.argmin(dim=-1)
        z_q = self.codebook(idx).view_as(z_e)

        if self.mode == "ema":
            commit_loss = F.mse_loss(z_e, z_q.detach())
            q_loss = self.beta * commit_loss
            if self.training:
                with torch.no_grad():
                    oh = F.one_hot(idx, self.num_codes).float()
                    batch_count = oh.sum(dim=0)
                    batch_sum = oh.t() @ flat.detach()
                    self.ema_count.mul_(self.ema_decay).add_(batch_count, alpha=1.0 - self.ema_decay)
                    self.ema_sum.mul_(self.ema_decay).add_(batch_sum, alpha=1.0 - self.ema_decay)
                    # 拉普拉斯平滑防止除零
                    n_tot = self.ema_count.sum()
                    smoothed = (self.ema_count + 1e-5) / (n_tot + self.num_codes * 1e-5) * n_tot
                    self.codebook.weight.copy_(self.ema_sum / smoothed.unsqueeze(-1))
        else:
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            commit_loss = F.mse_loss(z_e, z_q.detach())
            q_loss = codebook_loss + self.beta * commit_loss

        if self.training and self.mode in ("dead_reset", "ema"):
            with torch.no_grad():
                self.usage_acc += torch.bincount(idx, minlength=self.num_codes).float()
                self.step_count += 1
                if self.step_count % self.reset_every == 0:
                    dead = self.usage_acc < 1.0
                    if dead.any():
                        n_dead = int(dead.sum().item())
                        repl = flat.detach()[torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)]
                        repl = repl + 1e-3 * torch.randn_like(repl)
                        self.codebook.weight[dead] = repl
                        self.ema_count[dead] = 1.0
                        self.ema_sum[dead] = repl
                    self.usage_acc.zero_()

        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, idx.view(z_e.shape[:-1]), q_loss


def vq_loss(recon, target, q_loss):
    rec_loss = F.mse_loss(recon, target)
    return rec_loss + q_loss, rec_loss.item()


class SmallVQVAE(nn.Module):
    """64×64 → 4×4 (K=16) 的轻量 VQ-VAE，适合快速在 PushT 图像上对比四种量化配置。"""

    def __init__(self, num_codes=128, dim=32, mode="random"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.SiLU(),    # 32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(),   # 16
            nn.Conv2d(64, 64, 4, stride=2, padding=1), nn.SiLU(),   # 8
            nn.Conv2d(64, dim, 4, stride=2, padding=1),             # 4
        )
        self.vq = VectorQuantizer(num_codes=num_codes, dim=dim, beta=0.25, mode=mode)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, 64, 4, stride=2, padding=1), nn.SiLU(),  # 8
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1), nn.SiLU(),   # 16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SiLU(),   # 32
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(), # 64
        )

    def forward(self, obs):
        z_e = self.enc(obs).permute(0, 2, 3, 1)  # [B, 4, 4, d]
        z_q, idx, q_loss = self.vq(z_e)
        recon = self.dec(z_q.permute(0, 3, 1, 2))
        return recon, idx, q_loss


# ==================== 第二部分：手算验证 ====================


def verify_section_1_hand_calcs():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 1.4–1.6 节全部手算例子")
    print("=" * 60)

    # 1. 第 1.4 节：最近邻、三项损失与三路梯度
    cb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.5, 0.5]], requires_grad=True)
    z_e = torch.tensor([0.8, 0.4], requires_grad=True)
    dists = (z_e.unsqueeze(0) - cb).pow(2).sum(dim=-1)
    k_star = int(dists.argmin().item())
    z_q = cb[k_star]
    z_q_st = z_e + (z_q - z_e).detach()
    o_hat = 2.0 * z_q_st[0] + z_q_st[1]
    l_rec = (1.9 - o_hat).pow(2)
    l_cb = (z_e.detach() - z_q).pow(2).sum()
    l_cm = 0.25 * (z_e - z_q.detach()).pow(2).sum()
    l_tot = l_rec + l_cb + l_cm
    l_tot.backward()
    print(f"  1. 最近邻下标 = {k_star + 1} (e_4) ｜ 总损失 = {l_tot.item():.3f}（手算 0.285 = 0.16 + 0.10 + 0.025）")
    print(f"     z_e 梯度 = {[round(x, 2) for x in z_e.grad.tolist()]}（手算 [-1.45, -0.85]）")
    print(f"     e_4 梯度 = {[round(x, 2) for x in cb.grad[3].tolist()]}（手算 [-0.6, 0.2]），其余码梯度全 0 ✓")
    assert k_star == 3 and abs(l_tot.item() - 0.285) < 1e-5
    assert torch.allclose(z_e.grad, torch.tensor([-1.45, -0.85]), atol=1e-5)
    assert torch.allclose(cb.grad[3], torch.tensor([-0.6, 0.2]), atol=1e-5)
    assert cb.grad[:3].abs().sum().item() == 0.0

    # 2. 第 1.5 节：利用率与困惑度
    cases = [
        ([0]*4 + [1]*4 + [2]*4 + [3]*4, 1.0, 4.00),
        ([0]*8 + [1]*4 + [2]*2 + [3]*2, 1.0, 3.36),
        ([0]*13 + [1]*1 + [2]*1 + [3]*1, 1.0, 1.99),
        ([0]*16, 0.25, 1.00),
    ]
    for arr, exp_u, exp_p in cases:
        u, p = codebook_usage(torch.tensor(arr), 4)
        assert abs(u - exp_u) < 1e-4 and abs(p - exp_p) < 0.02
    print("  2. 利用率与困惑度 4 组手算例全部精确吻合（PPL = 4.00, 3.36, 1.99, 1.00）✓")

    # 3. 第 1.6 节：EMA 单步更新
    N_j = 0.9 * 10.0 + 0.1 * 2.0
    m_j = 0.9 * np.array([5.0, 5.0]) + 0.1 * np.array([1.4, 1.0])
    e_j = m_j / N_j
    print(f"  3. EMA 更新：N_j = {N_j:.1f}，e_j = ({e_j[0]:.3f}, {e_j[1]:.3f})（手算 (0.504, 0.500) ✓）")
    assert abs(N_j - 9.2) < 1e-6 and np.allclose(e_j, [0.5043, 0.500], atol=1e-3)


# ==================== 第三部分：PushT 画面上的四种设置对比 ====================


def load_pusht_64(max_frames=1600):
    demos = load_pusht_demos()
    imgs = demos["images"]
    step = max(1, len(imgs) // max_frames)
    sub = imgs[::step][:max_frames]
    resized = np.stack([cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA) for im in sub], axis=0)
    return torch.from_numpy(resized).permute(0, 3, 1, 2).float() / 255.0  # [N, 3, 64, 64]


def run_collapse_experiment():
    print("\n" + "=" * 60)
    print("第二部分：PushT 画面上对比 4 种 VQ-VAE 配置（码本大小 N = 128，4×4 = 16 token/帧）")
    print("=" * 60)
    quick = is_quick()
    frames = load_pusht_64(max_frames=300 if quick else 1600)
    n_tr = int(len(frames) * 0.85)
    tr_x, val_x = frames[:n_tr], frames[n_tr:]

    n_steps = 40 if quick else 220
    eval_every = 10 if quick else 22
    modes = [
        ("random", "随机初始化（无防坍塌）"),
        ("data_init", "数据初始化"),
        ("dead_reset", "数据初始化 + 死码重置"),
        ("ema", "EMA 码本更新 + 死码重置"),
    ]

    results = {}
    t0 = time.time()
    for mode, label in modes:
        seed_everything(SEED)
        model = SmallVQVAE(num_codes=128, dim=32, mode=mode)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        steps_x, util_h, ppl_h, mse_h = [], [], [], []

        for step in range(1, n_steps + 1):
            model.train()
            idx = torch.randint(0, len(tr_x), (32,))
            recon, _, ql = model(tr_x[idx])
            loss, _ = vq_loss(recon, tr_x[idx], ql)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step == 1 or step % eval_every == 0 or step == n_steps:
                model.eval()
                with torch.no_grad():
                    r_v, id_v, _ = model(val_x)
                    u, p = codebook_usage(id_v, 128)
                    m = F.mse_loss(r_v, val_x).item()
                steps_x.append(step)
                util_h.append(u * 100.0)
                ppl_h.append(p)
                mse_h.append(m)

        results[label] = (steps_x, util_h, ppl_h, mse_h)
        print(f"  {label:<24} → 最终利用率 {util_h[-1]:5.1f}% ｜ 困惑度 {ppl_h[-1]:5.1f}/128 ｜ 验证 MSE {mse_h[-1]:.5f}")
    print(f"  总耗时 {time.time() - t0:.1f}s")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    for label, (sx, uh, ph, mh) in results.items():
        axes[0].plot(sx, uh, "o-", linewidth=2, label=label)
        axes[1].plot(sx, ph, "o-", linewidth=2, label=label)
        axes[2].plot(sx, mh, "o-", linewidth=2, label=label)
    axes[0].set_title("码本利用率 (%)"); axes[0].set_xlabel("训练步数"); axes[0].legend(fontsize=8)
    axes[1].set_title("码本困惑度 PPL（最大 128）"); axes[1].set_xlabel("训练步数"); axes[1].legend(fontsize=8)
    axes[2].set_title("验证集重建 MSE"); axes[2].set_xlabel("训练步数"); axes[2].legend(fontsize=8)
    save_figure(fig, OUT / "01-vq-codebook-collapse.png")


def main():
    seed_everything(SEED)
    verify_section_1_hand_calcs()
    run_collapse_experiment()
    print("\n" + "=" * 60)
    print("本节完成！下一节：02-fsq-and-tokenizer-quality.py（FSQ 与 Tokenizer 质量诊断）")
    print("=" * 60)


if __name__ == "__main__":
    main()
