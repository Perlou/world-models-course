"""
Dreamer V3（一）：Symlog 变换与 Two-Hot 离散回归
================================================

学习目标：
    1. 实现 symlog(x)、symexp(x)、twohot_encode(y, bins) 与输出层零初始化的 TwoHotHead
    2. 严格核对 CONCEPT.md 第 6.6–6.8 节的全部手算例子：
       - 方式 A（symlog 空间等距桶）：y=3.5 → two-hot [0, 0, 0, 0.4959, 0.5041]，交叉熵 0.9506
       - 方式 B（原始空间指数间距桶）：y=3.5 → 权重 (0.6185, 0.3815)
       - 随机目标（50% 得 0，50% 得 6.389，真期望 3.195）：方式 A 读出 1.718（Jensen 不等式偏低），方式 B 精确读出 3.195
    3. 构造目标跨 6 个数量级（0.01 到 10,000）的回归实验，用同一个学习率对比：
       原始平方误差（MSE） vs symlog 平方误差 vs two-hot 离散回归，观察梯度尺度解耦的决定性优势

核心概念：
    - symlog(x) = sign(x) ln(|x| + 1)：无状态、可逆、奇函数、零附近恒等、远处对数压缩
    - two-hot 编码：在相邻两个桶之间做线性插值，期望精确还原连续目标，无量化误差
    - 梯度尺度解耦：two-hot 交叉熵对 logits 的梯度恒为 p̂ − ω ∈ [−1, 1]，与目标绝对大小无关

前置知识：
    - Phase 4 的 C51 分布式价值（07 节）
    - 05-kl-balancing-and-free-bits.py

对应概念文档：
    - CONCEPT.md 第 6 节「Dreamer V3（一）：symlog 与 two-hot」

预计运行时间：约 15 秒（CPU）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-06")
SEED = 0


# ==================== 第一部分：symlog / symexp / two-hot 实现 ====================


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


def twohot_encode(y, bins):
    """CONCEPT.md 第 6.9 节：把任意形状张量 y [...] 线性插值到递增桶 bins [M] 上，返回 [..., M]。"""
    y_clip = y.clamp(bins[0], bins[-1])
    # 找到满足 bins[i] <= y_clip 的最大下标 i ∈ [0, M-2]
    idx = torch.searchsorted(bins, y_clip, right=True) - 1
    idx = idx.clamp(0, len(bins) - 2)
    b_lo = bins[idx]
    b_hi = bins[idx + 1]
    w_hi = (y_clip - b_lo) / (b_hi - b_lo).clamp_min(1e-8)
    w_lo = 1.0 - w_hi

    out = torch.zeros(*y.shape, len(bins), device=y.device, dtype=y.dtype)
    out.scatter_(-1, idx.unsqueeze(-1), w_lo.unsqueeze(-1))
    out.scatter_(-1, (idx + 1).unsqueeze(-1), w_hi.unsqueeze(-1))
    return out


class TwoHotHead(nn.Module):
    """Dreamer V3 的 two-hot 离散回归头（方式 B：255 个指数间距桶 symexp(linspace(-20, 20, 255))，末层零初始化）。"""

    def __init__(self, in_dim, hidden=128, n_bins=255, low=-20.0, high=20.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.out = nn.Linear(hidden, n_bins)
        # V3 关键细节：输出层权重和偏置初始化为 0 → 均匀分布 → 初始预测严格为 0
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("bins", symexp(torch.linspace(low, high, n_bins)))

    def forward(self, x):
        return self.out(self.net(x))

    def predict(self, logits):
        """对称配对求和（Dreamer V3 官方数值技巧）：避免 float32 下 ±4.85e8 对消产生约 ±1.0 的舍入误差。"""
        probs = F.softmax(logits, dim=-1)
        n = probs.shape[-1]
        half = n // 2
        if n % 2 == 1:
            p_neg = probs[..., :half]
            p_pos = probs[..., half + 1 :].flip(-1)
            b_neg = self.bins[:half]
            b_pos = self.bins[half + 1 :].flip(-1)
            return (p_neg * b_neg + p_pos * b_pos).sum(dim=-1) + probs[..., half] * self.bins[half]
        return (probs * self.bins).sum(dim=-1)

    def loss(self, logits, target):
        twohot = twohot_encode(target.detach(), self.bins)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(twohot * log_probs).sum(dim=-1)


# ==================== 第二部分：手算单元测试 ====================


def verify_hand_calculations():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 6.2、6.6–6.8 节手算例子")
    print("=" * 60)

    # 1. 可逆性验证
    xs = torch.tensor([-1000.0, -3.5, 0.0, 0.1, 3.5, 10000.0])
    assert torch.allclose(symexp(symlog(xs)), xs, rtol=1e-5, atol=1e-5)

    # 2. 方式 A 手算例子（bins = [-2, -1, 0, 1, 2]，y = 3.5）
    bins_a = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = torch.tensor(3.5)
    th_a = twohot_encode(symlog(y), bins_a)
    rec_a = symexp((th_a * bins_a).sum()).item()
    p_cur = torch.tensor([0.05, 0.05, 0.1, 0.5, 0.3])
    ce_a = -(th_a * torch.log(p_cur)).sum().item()
    print(f"  1. 方式 A（symlog 等距桶, y=3.5）：")
    print(f"     two-hot = {[round(v, 4) for v in th_a.tolist()]}（手算 [0, 0, 0, 0.4959, 0.5041]）")
    print(f"     还原 symexp(E[b]) = {rec_a:.4f} ｜ 交叉熵损失 = {ce_a:.4f}（手算 0.9506）")
    assert torch.allclose(th_a, torch.tensor([0.0, 0.0, 0.0, 0.4959, 0.5041]), atol=1e-3)
    assert abs(rec_a - 3.5) < 1e-4 and abs(ce_a - 0.9506) < 1e-3

    # 3. 方式 B 手算例子（原始空间指数间距桶 symexp([-2,-1,0,1,2])，y = 3.5）
    bins_b = symexp(bins_a)
    th_b = twohot_encode(y, bins_b)
    rec_b = (th_b * bins_b).sum().item()
    print(f"\n  2. 方式 B（原始空间指数间距桶, y=3.5）：")
    print(f"     two-hot = {[round(v, 4) for v in th_b.tolist()]}（手算 [0, 0, 0, 0.6185, 0.3815]）")
    print(f"     还原 E[symexp(b)] = {rec_b:.4f}")
    assert torch.allclose(th_b, torch.tensor([0.0, 0.0, 0.0, 0.6185, 0.3815]), atol=1e-3)
    assert abs(rec_b - 3.5) < 1e-4

    # 4. 随机目标：50% 得 0，50% 得 6.389（真实期望 3.1945），对比方式 A 与方式 B 的读出值
    p_bimodal = torch.tensor([0.0, 0.0, 0.5, 0.0, 0.5])
    read_a = symexp((p_bimodal * bins_a).sum()).item()
    read_b = (p_bimodal * bins_b).sum().item()
    true_mean = 0.5 * 0.0 + 0.5 * bins_b[-1].item()
    print(f"\n  3. 随机目标（50% 为 0，50% 为 6.389，真期望 {true_mean:.3f}）：")
    print(f"     方式 A 读出 = {read_a:.3f}（手算 1.718，受 Jensen 不等式偏低！）")
    print(f"     方式 B 读出 = {read_b:.3f}（精确等于真期望 {true_mean:.3f} ✓）")
    assert abs(read_a - 1.7183) < 1e-3 and abs(read_b - true_mean) < 1e-4

    # 5. 校验零初始化输出严格为 0
    head = TwoHotHead(in_dim=8)
    init_pred = head.predict(head(torch.randn(16, 8)))
    assert torch.allclose(init_pred, torch.zeros_like(init_pred), atol=1e-5)
    print("  ✓ 零初始化 TwoHotHead 初始预测严格为 0，全部手算验证通过！")


# ==================== 第三部分：多尺度回归对比实验 ====================


def run_multiscale_experiment():
    print("\n" + "=" * 60)
    print("第二部分：多尺度回归对比 — 原始 MSE vs Symlog MSE vs Two-Hot（同一学习率 lr=1e-2）")
    print("=" * 60)

    scales = [0.01, 1.0, 100.0, 10000.0]
    n_steps = 120 if is_quick() else 350
    lr = 1e-2

    rng = np.random.default_rng(SEED)
    X_np = rng.normal(0, 1, size=(256, 8)).astype(np.float32)
    # 构造非线性目标：随特征平滑变化，有正有负
    base_y_np = (np.sin(X_np[:, 0]) + 0.5 * X_np[:, 1] * X_np[:, 2] - 0.3 * X_np[:, 3]).astype(np.float32)
    X = torch.from_numpy(X_np)

    results = {m: {} for m in ["Raw MSE", "Symlog MSE", "Two-Hot (V3)"]}

    for scale in scales:
        y_true = torch.from_numpy(base_y_np * scale)
        denom = y_true.abs().mean().item() + 1e-8

        for method in results:
            seed_everything(SEED)
            if method == "Two-Hot (V3)":
                model = TwoHotHead(in_dim=8, hidden=64)
            else:
                model = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            rel_errs = []

            for _ in range(n_steps):
                if method == "Raw MSE":
                    pred = model(X).squeeze(-1)
                    loss = 0.5 * (pred - y_true).pow(2).mean()
                    pred_orig = pred.detach()
                elif method == "Symlog MSE":
                    out = model(X).squeeze(-1)
                    loss = 0.5 * (out - symlog(y_true)).pow(2).mean()
                    pred_orig = symexp(out.detach())
                else:
                    logits = model(X)
                    loss = model.loss(logits, y_true).mean()
                    pred_orig = model.predict(logits.detach())

                opt.zero_grad()
                loss.backward()
                opt.step()

                rel = (pred_orig - y_true).abs().mean().item() / denom
                rel_errs.append(min(rel, 1e4) if np.isfinite(rel) else 1e4)
            results[method][scale] = np.array(rel_errs)

    print(f"  {'目标尺度':<12} {'原始 MSE 相对误差':>18} {'Symlog MSE 相对误差':>20} {'Two-Hot 相对误差':>18}")
    for scale in scales:
        e_raw = results["Raw MSE"][scale][-1]
        e_sym = results["Symlog MSE"][scale][-1]
        e_th = results["Two-Hot (V3)"][scale][-1]
        print(f"  {scale:<12g} {e_raw:>17.4f} {e_sym:>20.4f} {e_th:>18.4f}")

    fig, axes = plt.subplots(1, len(scales), figsize=(15.2, 3.8))
    for ax, scale in zip(axes, scales):
        for method in results:
            ax.plot(results[method][scale], linewidth=2, label=method)
        ax.set_yscale("log")
        ax.set_title(f"目标尺度 ×{scale:g}")
        ax.set_xlabel("训练步数")
        if scale == scales[0]:
            ax.set_ylabel("相对平均绝对误差（对数）")
        ax.legend(fontsize=8)
    fig.suptitle("同一学习率（lr=1e-2）跨 6 个数量级：原始 MSE 在大尺度上失稳/慢收敛，Symlog 与 Two-Hot 全尺度稳健", y=1.02)
    save_figure(fig, OUT / "06-symlog-twohot-multiscale.png")


def main():
    seed_everything(SEED)
    verify_hand_calculations()
    run_multiscale_experiment()
    print("\n" + "=" * 60)
    print("本节完成！下一节：07-return-normalization-and-robustness.py（回报归一化、unimix 与 EMA 正则）")
    print("=" * 60)


if __name__ == "__main__":
    main()
