"""
Phase 9 · 第 09 课：实时推理 —— 少步采样、Shortcut 模型与因果 KV Cache
=====================================================================

学习目标：
    1. 核对 CONCEPT.md 第 9.4 节一维高斯 shortcut 自洽手算，理解步长条件 d 的作用
    2. 实现 ShortcutNet(x_u, u, d) 与两项训练损失（d=0 的 flow matching + 2d 自举自洽项），
       在 8-高斯环上对比 Flow Matching 与 Shortcut 模型在 1、2、4 步下的样本质量
    3. 剖析每帧推理延迟构成（t_帧 = N × t_网络 + t_解码 + t_其他），测量本机 CPU 实测耗时
    4. 实现按帧管理的因果滑动窗口 KV Cache（CausalFrameCache），测量开启前后的单帧推理加速比

对应概念文档：
    src/phase-09-diffusion-video-world-models/CONCEPT.md 第 9 节

运行方式：
    uv run python src/phase-09-diffusion-video-world-models/09-realtime-few-step-inference.py
    WM_QUICK=1 uv run python src/phase-09-diffusion-video-world-models/09-realtime-few-step-inference.py

预计运行时间：完整约 45 秒（M1 Max CPU），冒烟约 6 秒
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
_spec01 = importlib.util.spec_from_file_location("p9_01_rt", os.path.join(_DIR, "01-ddpm-to-edm.py"))
_mod01 = importlib.util.module_from_spec(_spec01)
_spec01.loader.exec_module(_mod01)

_spec02 = importlib.util.spec_from_file_location("p9_02_rt", os.path.join(_DIR, "02-flow-matching.py"))
_mod02 = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(_mod02)

SinusoidalEmb = _mod01.SinusoidalEmb
ToyMLP = _mod01.ToyMLP
sample_8gaussians = _mod01.sample_8gaussians
eval_2d_samples = _mod01.eval_2d_samples
interpolate = _mod02.interpolate
cfm_loss = _mod02.cfm_loss
sample_flow_euler = _mod02.sample_flow_euler

plt = setup_matplotlib()
OUT = output_dir("phase-09")
SEED = 0


# ==================== 第一部分：Shortcut 模型（带步长条件 d 与自举损失） ====================


class ShortcutNet(nn.Module):
    """CONCEPT.md 第 9.4 节：接收 (x_u, u, d) 输出从 u 走到 u+d 的平均速度 v̄_θ。"""

    def __init__(self, data_dim=2, hidden=128):
        super().__init__()
        self.u_emb = SinusoidalEmb(64)
        self.d_emb = SinusoidalEmb(64)
        self.net = nn.Sequential(
            nn.Linear(data_dim + 64, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, x_u, u, d):
        cond = self.u_emb(u) + self.d_emb(d)
        return self.net(torch.cat([x_u, cond], dim=-1))


def shortcut_loss(model, x, d_choices=(0.0625, 0.125, 0.25, 0.5), sc_ratio=0.5):
    """CONCEPT.md 第 9.6 节：批次前一半做 flow matching (d=0)，后一半做两小步自洽自举 (2d)。"""
    B = x.shape[0]
    device = x.device
    n_sc = max(1, int(B * sc_ratio))
    n_fm = B - n_sc

    # 1. Flow Matching 接地项（d = 0）
    x_fm = x[:n_fm]
    eps_fm = torch.randn_like(x_fm)
    u_fm = torch.rand(n_fm, device=device)
    x_u_fm = interpolate(x_fm, eps_fm, u_fm)
    v_pred_fm = model(x_u_fm, u_fm, torch.zeros_like(u_fm))
    loss_fm = F.mse_loss(v_pred_fm, x_fm - eps_fm)

    # 2. 自洽自举项：v̄(x_u, u, 2d) ≈ 0.5 * (v̄(x_u, u, d) + v̄(x_mid, u+d, d))
    x_sc = x[n_fm:]
    eps_sc = torch.randn_like(x_sc)
    d_idx = torch.randint(0, len(d_choices), (n_sc,), device=device)
    d_vals = torch.tensor(d_choices, device=device)[d_idx]
    # 采样离散网格起点 u ∈ {0, 2d, 4d, ...} 确保 u + 2d <= 1
    u_sc = torch.zeros(n_sc, device=device)
    for i in range(n_sc):
        step2 = 2.0 * float(d_vals[i].item())
        n_slots = max(1, int(round(1.0 / step2)))
        u_sc[i] = float(np.random.randint(0, n_slots)) * step2

    x_u_sc = interpolate(x_sc, eps_sc, u_sc)
    # 最小步长用 d_in = 0 接住 flow matching 接地信号，更大步长自举下一档
    d_in = torch.where(d_vals <= d_choices[0] + 1e-6, torch.zeros_like(d_vals), d_vals)
    with torch.no_grad():
        v1 = model(x_u_sc, u_sc, d_in)
        x_mid = x_u_sc + d_vals.unsqueeze(-1) * v1
        v2 = model(x_mid, (u_sc + d_vals).clamp_max(1.0), d_in)
        v_target = 0.5 * (v1 + v2)

    v_pred_2d = model(x_u_sc, u_sc, 2.0 * d_vals)
    loss_sc = F.mse_loss(v_pred_2d, v_target)
    return loss_fm + loss_sc, loss_fm.item(), loss_sc.item()


@torch.no_grad()
def sample_shortcut(model, shape, n_steps, device="cpu", init_eps=None):
    """CONCEPT.md 第 9.4 节：临时指定步数 N，以步长 d = 1/N 调用 N 次。"""
    model.eval()
    x = torch.randn(shape, device=device) if init_eps is None else init_eps.to(device).clone()
    B = shape[0]
    d = 1.0 / n_steps
    d_t = torch.full((B,), d, device=device)
    for i in range(n_steps):
        u_t = torch.full((B,), i * d, device=device)
        x = x + d * model(x, u_t, d_t)
    return x


# ==================== 第二部分：因果帧级 KV Cache 与每帧延迟剖析 ====================


class CausalFrameBlock(nn.Module):
    """支持按帧增量 KV Cache 与滑动窗口截断的轻量因果时间 Transformer 层。"""

    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, kv_cache=None, max_ctx=32):
        # x: [B, L, D]
        B, L, D = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, L, H, Dh]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, H, L, Dh]

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=2)[:, :, -max_ctx:]
            v = torch.cat([v_prev, v], dim=2)[:, :, -max_ctx:]
            new_cache = (k, v)
            is_causal = False
        else:
            new_cache = (k[:, :, -max_ctx:], v[:, :, -max_ctx:])
            is_causal = True

        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).reshape(B, L, D)
        x = x + self.proj(out)
        return x + self.ffn(self.norm2(x)), new_cache


class CausalStreamModel(nn.Module):
    def __init__(self, d_model=128, n_layers=4, n_heads=4, latent_ch=4):
        super().__init__()
        self.blocks = nn.ModuleList([CausalFrameBlock(d_model, n_heads) for _ in range(n_layers)])
        self.out_head = nn.Linear(d_model, d_model)
        # 轻量解码器（8×8×4 → 32×32×3）用于测量 t_解码
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, 32, 3, 1, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
        )

    def forward_full(self, x_seq):
        h = x_seq
        for blk in self.blocks:
            h, _ = blk(h, kv_cache=None)
        return self.out_head(h[:, -1:])

    def forward_cached(self, x_step, caches, max_ctx=32):
        h = x_step
        new_caches = []
        for blk, c in zip(self.blocks, caches):
            h, nc = blk(h, kv_cache=c, max_ctx=max_ctx)
            new_caches.append(nc)
        return self.out_head(h), new_caches


@torch.no_grad()
def profile_kv_cache_and_latency(quick=False):
    print("\n" + "=" * 60)
    print("第三部分：每帧延迟构成剖析 & 因果 KV Cache 加速实测（本机 CPU）")
    print("=" * 60)
    model = CausalStreamModel(d_model=128, n_layers=4, n_heads=4)
    model.eval()

    # 1. 对比无 KV Cache vs 有 KV Cache 在不同上下文长度下的单步网络耗时
    ctx_lens = [8, 16, 32, 64]
    tokens_per_frame = 16
    reps = 5 if quick else 15
    print(f"  {'上下文帧数':>8s} | {'无 KV Cache (ms)':>16s} | {'有 KV Cache (ms)':>16s} | {'加速比':>8s}")
    no_cache_ms, cache_ms = [], []
    for T_ctx in ctx_lens:
        seq = torch.randn(1, T_ctx * tokens_per_frame, 128)
        new_f = torch.randn(1, tokens_per_frame, 128)
        # 预热 cache
        caches = []
        h = seq[:, :-tokens_per_frame]
        for blk in model.blocks:
            h, c = blk(h, kv_cache=None, max_ctx=T_ctx * tokens_per_frame)
            caches.append(c)

        t1 = time.perf_counter()
        for _ in range(reps):
            _ = model.forward_full(seq)
        dt_full = (time.perf_counter() - t1) * 1000.0 / reps

        t2 = time.perf_counter()
        for _ in range(reps):
            _, _ = model.forward_cached(new_f, caches, max_ctx=T_ctx * tokens_per_frame)
        dt_c = (time.perf_counter() - t2) * 1000.0 / reps

        no_cache_ms.append(dt_full)
        cache_ms.append(dt_c)
        print(f"  {T_ctx:>8d} | {dt_full:16.2f} | {dt_c:16.2f} | {dt_full / max(dt_c, 1e-4):7.2f}×")

    # 2. 剖析不同去噪步数 N 下的完整每帧延迟构成 (N × t_网络 + t_解码 + t_其他)
    z_lat = torch.randn(1, 4, 8, 8)
    t_dec_s = time.perf_counter()
    for _ in range(reps * 2):
        _ = model.decoder(z_lat)
    t_dec = (time.perf_counter() - t_dec_s) * 1000.0 / (reps * 2)
    t_net = cache_ms[1]  # T_ctx = 16 帧时的单步去噪网络耗时
    t_other = 0.25       # 张量拼装/噪声采样开销
    print(f"\n  本机 CPU 单帧耗时拆解（T_ctx=16 + KV Cache，t_网络={t_net:.2f}ms，t_解码={t_dec:.2f}ms）：")
    print(f"  {'步数 N':>6s} | {'N×t_网络 (ms)':>13s} | {'t_解码 (ms)':>11s} | {'总延迟 (ms)':>11s} | {'实时帧率 (FPS)':>14s}")
    for N in [1, 2, 4, 20]:
        tot = N * t_net + t_dec + t_other
        print(f"  {N:>6d} | {N*t_net:13.2f} | {t_dec:11.2f} | {tot:11.2f} | {1000.0/tot:14.1f}")
    return ctx_lens, no_cache_ms, cache_ms, t_net, t_dec


def verify_section_9_4_hand_calc():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 9.4 节一维高斯 Shortcut 自洽手算")
    print("=" * 60)
    x0 = 0.8
    x05 = 1.0 + x0 * math.sqrt(0.5)
    x1 = 2.8
    v_0_05 = (x05 - x0) / 0.5
    v_05_1 = (x1 - x05) / 0.5
    v_0_1 = (x1 - x0) / 1.0
    assert abs(0.5 * (v_0_05 + v_05_1) - v_0_1) < 1e-6
    print(f"  v̄(0→0.5)={v_0_05:.3f}，v̄(0.5→1)={v_05_1:.3f}，均值={0.5*(v_0_05+v_05_1):.3f} == v̄(0→1)={v_0_1:.3f} ✓")
    print(f"  一步瞬时速度落点 = {x0 + 1.2:.2f}（塌向均值 2.0），一步 shortcut 落点 = {x0 + v_0_1:.2f}（精确样本 2.8 ✓）")


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()
    verify_section_9_4_hand_calc()

    print("\n" + "=" * 60)
    print("第二部分：8-高斯环上对比 Flow Matching 与 Shortcut 模型（1 / 2 / 4 步）")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    data_np, centers = sample_8gaussians(4000, rng=rng)
    data_t = torch.from_numpy(data_np)

    fm = ToyMLP(2, 128)
    sc = ShortcutNet(2, 128)
    opt_fm = torch.optim.Adam(fm.parameters(), lr=2e-3)
    opt_sc = torch.optim.Adam(sc.parameters(), lr=2e-3)

    n_iters = 500 if quick else 3000
    for _ in range(n_iters):
        idx = torch.randint(0, len(data_t), (256,))
        x = data_t[idx]

        opt_fm.zero_grad()
        cfm_loss(fm, x).backward()
        opt_fm.step()

        opt_sc.zero_grad()
        l_sc, _, _ = shortcut_loss(sc, x)
        l_sc.backward()
        nn.utils.clip_grad_norm_(sc.parameters(), 1.0)
        opt_sc.step()

    init_eps = torch.randn(1000, 2)
    print(f"\n  {'步数 N':>6s} | {'Flow Matching 距中心':>20s} {'命中率':>7s} | {'Shortcut 模型 距中心':>20s} {'命中率':>7s}")
    pts_store = {}
    for N in [1, 2, 4]:
        p_fm = sample_flow_euler(fm, (1000, 2), N, init_eps=init_eps).numpy()
        p_sc = sample_shortcut(sc, (1000, 2), N, init_eps=init_eps).numpy()
        m_fm = eval_2d_samples(p_fm, centers)
        m_sc = eval_2d_samples(p_sc, centers)
        pts_store[N] = (p_fm, p_sc, m_fm, m_sc)
        print(f"  {N:>6d} | {m_fm[0]:20.4f} {m_fm[1]*100:6.1f}% | {m_sc[0]:20.4f} {m_sc[1]*100:6.1f}%")

    ctx_lens, no_cache_ms, cache_ms, t_net, t_dec = profile_kv_cache_and_latency(quick=quick)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    axes[0].scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=45, zorder=5)
    axes[0].scatter(pts_store[1][0][:600, 0], pts_store[1][0][:600, 1], s=5, alpha=0.5)
    axes[0].set_title(f"Flow Matching 1 步 (命中 {pts_store[1][2][1]*100:.0f}%)")
    axes[0].set_xlim(-1.6, 1.6); axes[0].set_ylim(-1.6, 1.6)

    axes[1].scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=45, zorder=5)
    axes[1].scatter(pts_store[1][1][:600, 0], pts_store[1][1][:600, 1], s=5, alpha=0.5)
    axes[1].set_title(f"Shortcut 模型 1 步 (命中 {pts_store[1][3][1]*100:.0f}%)")
    axes[1].set_xlim(-1.6, 1.6); axes[1].set_ylim(-1.6, 1.6)

    axes[2].plot([1, 2, 4], [pts_store[n][2][0] for n in [1, 2, 4]], "o-", label="Flow Matching")
    axes[2].plot([1, 2, 4], [pts_store[n][3][0] for n in [1, 2, 4]], "s-", label="Shortcut 模型")
    axes[2].set_title("1/2/4 步距中心误差"); axes[2].set_xlabel("步数 N"); axes[2].legend()

    axes[3].plot(ctx_lens, no_cache_ms, "o--", label="无 KV Cache")
    axes[3].plot(ctx_lens, cache_ms, "s-", label="有 KV Cache")
    axes[3].set_title("因果时间注意力单步耗时"); axes[3].set_xlabel("上下文帧数"); axes[3].set_ylabel("ms"); axes[3].legend()

    fig.tight_layout()
    save_figure(fig, OUT / "09-realtime-and-shortcut.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：10-dreamer4-and-video-foundation-models.py")


if __name__ == "__main__":
    main()
