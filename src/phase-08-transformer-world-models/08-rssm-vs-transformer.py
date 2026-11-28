"""
RSSM（GRU）还是 Transformer：长时记忆与窗口截断对照实验
=======================================================

学习目标：
    1. 实现 make_memory_dataset() 与 evaluate_recall_vs_delay()：
       构造受控延迟回忆任务——在时刻 t_0 给出 4 种信号之一，中间插入 τ 步带随机干扰的无关观测，在 t_0 + τ 步要求模型复现该信号
    2. 在同一份数据集、相近参数量下训练两个骨干：
       - GRU（RSSM 骨干）：恒定状态大小，无硬窗口，但每步非线性覆盖逐渐冲淡旧记忆
       - 因果 Transformer（上下文窗口 W = 24）：窗口内直接注意力一步直达，窗口外彻底失明
    3. 绘制"回忆准确率 — 间隔步数 τ"曲线并记录各自训练耗时，亲眼验证：
       Transformer 在 τ < W 内几乎 100% 持平、超出 W 后断崖跌回 25% 随机水平；GRU 随 τ 平滑衰减

核心概念：
    - 循环压缩（O(1) 状态、衰减记忆） vs 注意力直达（O(W) 显存、窗口内无损、窗口外硬截断）
    - 世界模型骨干选型：短记忆/实时控选 RSSM，窗口内多线索检索选 Transformer

前置知识：
    - Phase 3 的 03-deterministic-vs-stochastic.py
    - 03、04 节的因果 Transformer

对应概念文档：
    - CONCEPT.md 第 8 节「RSSM 还是 Transformer」

预计运行时间：约 1 分钟（CPU）
"""

import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec03 = importlib.util.spec_from_file_location("p8_03", os.path.join(_DIR, "03-gpt-as-world-model.py"))
_mod03 = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(_mod03)

TransformerBlock = _mod03.TransformerBlock

plt = setup_matplotlib()
OUT = output_dir("phase-08")
SEED = 0


# ==================== 第一部分：延迟回忆数据集 ====================


def make_memory_dataset(n_samples, seq_len, delay, n_classes=4, obs_dim=12, seed=0):
    """构造延迟回忆序列：
    在 t_cue = seq_len - 1 - delay 处编码 4 选 1 提示（前 4 维 one-hot + 第 4 维标记=1），
    中间全部是高斯噪声干扰，末尾 t = seq_len - 1 发出查询信号（第 5 维=1），要求输出提示类别！
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 0.35, size=(n_samples, seq_len, obs_dim)).astype(np.float32)
    y = rng.integers(0, n_classes, size=(n_samples,), dtype=np.int64)
    t_cue = seq_len - 1 - delay
    assert t_cue >= 0
    X[:, :, 4] = 0.0
    X[:, :, 5] = 0.0
    for i in range(n_samples):
        X[i, t_cue, :n_classes] = 0.0
        X[i, t_cue, y[i]] = 2.5
        X[i, t_cue, 4] = 1.0  # 提示标志位
    X[:, -1, 5] = 1.0         # 查询标志位
    return torch.from_numpy(X), torch.from_numpy(y)


# ==================== 第二部分：参数量匹配的 GRU 与带窗口 Transformer ====================


class GRUBackbone(nn.Module):
    def __init__(self, obs_dim=12, hidden=96, n_classes=4):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class WindowedTransformerBackbone(nn.Module):
    """固定上下文窗口 W：只看最后 W 步（模拟 Transformer 世界模型的硬上下文截断，如 IRIS 的 T_c=20）。"""

    def __init__(self, obs_dim=12, d_model=64, n_layers=2, n_heads=4, window=24, n_classes=4):
        super().__init__()
        self.window = window
        self.in_proj = nn.Linear(obs_dim, d_model)
        self.pos_emb = nn.Embedding(window, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x_w = x[:, -self.window :]  # 硬窗口截断
        L = x_w.shape[1]
        h = self.in_proj(x_w) + self.pos_emb(torch.arange(L, device=x.device)).unsqueeze(0)
        for blk in self.blocks:
            h, _ = blk(h)
        return self.head(self.ln_f(h)[:, -1])


def evaluate_recall_vs_delay(delays, window=24):
    """在混合延迟训练集上统一训练两个骨干，再在各延迟 τ 上分别评测回忆准确率。"""
    quick = is_quick()
    max_seq = max(delays) + 2
    n_train_per_d = 120 if quick else 320
    n_test = 120 if quick else 300
    n_steps = 90 if quick else 260

    # 训练数据覆盖所有延迟（含窗口内外），让两个模型公平竞争
    xs_tr, ys_tr = [], []
    for d in delays:
        xd, yd = make_memory_dataset(n_train_per_d, max_seq, delay=d, seed=SEED + d)
        xs_tr.append(xd); ys_tr.append(yd)
    X_tr = torch.cat(xs_tr, dim=0)
    y_tr = torch.cat(ys_tr, dim=0)

    seed_everything(SEED)
    gru = GRUBackbone(obs_dim=12, hidden=96)
    tf = WindowedTransformerBackbone(obs_dim=12, d_model=64, n_layers=2, n_heads=4, window=window)
    p_gru = sum(p.numel() for p in gru.parameters())
    p_tf = sum(p.numel() for p in tf.parameters())

    times = {}
    for name, model, lr in [("GRU (RSSM骨干)", gru, 3e-3), (f"Transformer (窗口 W={window})", tf, 2e-3)]:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        t0 = time.time()
        model.train()
        for _ in range(n_steps):
            idx = torch.randint(0, len(X_tr), (64,))
            loss = F.cross_entropy(model(X_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        times[name] = time.time() - t0

    gru.eval(); tf.eval()
    gru_accs, tf_accs = [], []
    print(f"  参数量：GRU = {p_gru:,} ｜ Transformer = {p_tf:,}")
    print(f"  训练耗时：GRU = {times['GRU (RSSM骨干)']:.1f}s ｜ Transformer = {times[f'Transformer (窗口 W={window})']:.1f}s\n")
    print(f"  {'延迟步数 τ':<12} {'GRU 准确率':>14} {'Transformer(W=24) 准确率':>26}")
    for d in delays:
        X_te, y_te = make_memory_dataset(n_test, max_seq, delay=d, seed=900 + d)
        with torch.no_grad():
            ag = (gru(X_te).argmax(dim=-1) == y_te).float().mean().item() * 100.0
            at = (tf(X_te).argmax(dim=-1) == y_te).float().mean().item() * 100.0
        gru_accs.append(ag); tf_accs.append(at)
        mark = "（超出窗口）" if d >= window else ""
        print(f"  τ = {d:<8} {ag:>13.1f}% {at:>24.1f}%  {mark}")

    return np.array(gru_accs), np.array(tf_accs), times


def main():
    seed_everything(SEED)
    print("=" * 60)
    print("长时记忆对决：GRU（RSSM 骨干） vs 因果 Transformer（固定窗口 W = 24）")
    print("=" * 60)
    delays = [2, 10, 20, 32] if is_quick() else [2, 6, 12, 18, 22, 26, 36, 48]
    window = 24
    gru_accs, tf_accs, times = evaluate_recall_vs_delay(delays, window=window)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(delays, gru_accs, "o-", linewidth=2.4, label=f"GRU / RSSM（无硬窗口，训练 {times['GRU (RSSM骨干)']:.1f}s）")
    ax.plot(delays, tf_accs, "s-", linewidth=2.4, label=f"Transformer W={window}（训练 {times[f'Transformer (窗口 W={window})']:.1f}s）")
    ax.axvline(window, color="tomato", linestyle="--", label=f"Transformer 上下文窗口边界 W = {window}")
    ax.axhline(25.0, color="gray", linestyle=":", label="4 选 1 瞎猜基线 (25%)")
    ax.set_xlabel("记忆间隔步数 τ")
    ax.set_ylabel("回忆准确率 (%)")
    ax.set_ylim(15, 106)
    ax.set_title("窗口内 Transformer 一步直达，超出窗口断崖跌落；GRU 无硬上限但平滑衰减")
    ax.legend(fontsize=8.5)
    save_figure(fig, OUT / "08-rssm-vs-transformer.png")

    print("\n" + "=" * 60)
    print("🎉 Phase 8 全部 8 节讲义完成！下一步：运行实战项目 projects/phase-08-pusht-token-world-model/！")
    print("=" * 60)


if __name__ == "__main__":
    main()
