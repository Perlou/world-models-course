"""
KL Balancing 与 Free Bits：防止后验坍塌，让先验主动追赶表征
=============================================================

学习目标：
    1. 实现数值稳定的 categorical_kl(logits_q, logits_p) 与 unimix 保护
    2. 核对 CONCEPT.md 第 5.3–5.5 节的手算例子：
       - KL(q‖p) = 0.0851 nat，KL(p‖q) = 0.0920 nat（不对称性）
       - KL balancing（α=0.8）下先验侧梯度 (−0.16, 0.08, 0.08) 与后验侧梯度 (0.0352, −0.0196, −0.0156)
    3. 实现 Dreamer V3 风格的 kl_loss_balanced()：拆分 L_dyn 与 L_rep 并逐元素截断 free bits（1 nat）
    4. 在序列预测自编码玩具任务上做 2×2 消融（有无 KL balancing × 有无 free bits），
       亲眼验证"无 free bits 且 β_rep 较大时 KL 被压向 0、重建变差"的坍塌现象

核心概念：
    - KL balancing：L_dyn = KL(sg(q) ‖ p) 权重大（先验追后验），L_rep = KL(q ‖ sg(p)) 权重小（后验慢让步）
    - free bits（1 nat）：逐时间步 max(1, KL)，低于预算时关闭 KL 梯度，把全部信息留给重建与预测

前置知识：
    - 04-categorical-latents-straight-through.py（类别隐变量与 straight-through）

对应概念文档：
    - CONCEPT.md 第 5 节「KL balancing 与 free bits」

预计运行时间：约 20 秒（CPU）
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec04 = importlib.util.spec_from_file_location("p6_04", os.path.join(_DIR, "04-categorical-latents-straight-through.py"))
_mod04 = importlib.util.module_from_spec(_spec04)
_spec04.loader.exec_module(_mod04)
sample_categorical_st = _mod04.sample_categorical_st

plt = setup_matplotlib()
OUT = output_dir("phase-06")
SEED = 0


# ==================== 第一部分：categorical_kl 与 kl_loss_balanced ====================


def unimix_probs(logits, eps=0.01):
    """CONCEPT.md 第 5.7、7.4 节：与均匀分布按 (1−ε):ε 混合，杜绝概率精确为 0 导致 KL 爆炸。"""
    probs = F.softmax(logits, dim=-1)
    if eps <= 0.0:
        return probs
    return (1.0 - eps) * probs + eps / logits.shape[-1]


def categorical_kl(logits_q, logits_p, unimix=0.01):
    """计算类别分布的 KL(q ‖ p)。

    输入 logits 形状可为 [..., K] 或 [..., N_z, K]：
    - 3 维及以上且有 N_z 维度（如 [B, T, N_z, K]）时，对最后两维 (N_z, K) 求和 → [B, T]；
    - 仅 1 维/2 维（如手算单个变量 [K] 或 [B, K]）时，对最后一维 K 求和。
    """
    if unimix > 0.0:
        q = unimix_probs(logits_q, unimix)
        p = unimix_probs(logits_p, unimix)
        log_q = torch.log(q.clamp_min(1e-8))
        log_p = torch.log(p.clamp_min(1e-8))
    else:
        log_q = F.log_softmax(logits_q, dim=-1)
        log_p = F.log_softmax(logits_p, dim=-1)
        q = log_q.exp()

    kl_per_var = (q * (log_q - log_p)).sum(dim=-1)
    if logits_q.ndim >= 3:
        return kl_per_var.sum(dim=-1)
    return kl_per_var


def kl_loss_balanced(post_logits, prior_logits, beta_dyn=1.0, beta_rep=0.1, free_nats=1.0, unimix=0.01):
    """CONCEPT.md 第 5.8 节：V3 风格的拆分 KL + 逐元素 free bits 截断。

    返回 (loss_scalar, raw_kl_mean, dyn_loss_mean, rep_loss_mean)。
    """
    kl_dyn = categorical_kl(post_logits.detach(), prior_logits, unimix=unimix)
    kl_rep = categorical_kl(post_logits, prior_logits.detach(), unimix=unimix)
    raw_kl = kl_dyn.detach().mean()

    if free_nats > 0.0:
        loss_dyn = kl_dyn.clamp_min(free_nats)
        loss_rep = kl_rep.clamp_min(free_nats)
    else:
        loss_dyn = kl_dyn
        loss_rep = kl_rep

    total = beta_dyn * loss_dyn.mean() + beta_rep * loss_rep.mean()
    return total, raw_kl.item(), loss_dyn.mean().item(), loss_rep.mean().item()


# ==================== 第二部分：手算核对 ====================


def verify_hand_calculations():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 5.3–5.5 节手算例子")
    print("=" * 60)

    q_probs = torch.tensor([0.7, 0.2, 0.1])
    p_probs = torch.tensor([0.5, 0.3, 0.2])
    lq = torch.log(q_probs)
    lp = torch.log(p_probs)

    kl_qp = categorical_kl(lq, lp, unimix=0.0).item()
    kl_pq = categorical_kl(lp, lq, unimix=0.0).item()
    print(f"  1. KL 不对称性核对：")
    print(f"     KL(q ‖ p) = {kl_qp:.4f} nat（手算 0.0851）")
    print(f"     KL(p ‖ q) = {kl_pq:.4f} nat（手算 0.0920）")
    assert abs(kl_qp - 0.0851) < 5e-4 and abs(kl_pq - 0.0920) < 5e-4

    # 核对第 5.5 节 α = 0.8 时先验与后验两侧梯度
    lq_g = lq.clone().requires_grad_(True)
    lp_g = lp.clone().requires_grad_(True)
    loss_bal = 0.8 * categorical_kl(lq_g.detach(), lp_g, unimix=0.0) + 0.2 * categorical_kl(lq_g, lp_g.detach(), unimix=0.0)
    loss_bal.backward()
    gp = lp_g.grad.numpy()
    gq = lq_g.grad.numpy()
    print(f"\n  2. KL balancing（α=0.8）两侧梯度核对（损失数值仍为 {loss_bal.item():.4f}）：")
    print(f"     先验 logits 梯度 = ({gp[0]:+.4f}, {gp[1]:+.4f}, {gp[2]:+.4f})  ｜  手算 (-0.1600, +0.0800, +0.0800)")
    print(f"     后验 logits 梯度 = ({gq[0]:+.4f}, {gq[1]:+.4f}, {gq[2]:+.4f})  ｜  手算 (+0.0352, -0.0196, -0.0156)")
    assert np.allclose(gp, [-0.16, 0.08, 0.08], atol=1e-4)
    assert np.allclose(gq, [0.0352, -0.0196, -0.0156], atol=2e-4)

    # 核对第 5.6 节逐元素截断 vs 先平均再截断
    sample_kls = torch.tensor([0.3, 0.9, 2.5])
    elem_clamp = sample_kls.clamp_min(1.0).mean().item()
    mean_clamp = sample_kls.mean().clamp_min(1.0).item()
    print(f"\n  3. Free bits 两种写法（KL=[0.3, 0.9, 2.5]）：逐元素截断 = {elem_clamp:.3f}（手算 1.500）｜ 先平均再截断 = {mean_clamp:.3f}（手算 1.233）")
    assert abs(elem_clamp - 1.5) < 1e-5 and abs(mean_clamp - 1.2333) < 1e-3
    print("  ✓ 全部手算验证通过！")


# ==================== 第三部分：2×2 消融实验 ====================


class ToySeqCategoricalModel(nn.Module):
    """轻量序列类别隐变量模型：每步 8 个 8 类变量（共 64 维 one-hot），用于清晰观察 KL balancing 与 free bits 作用。"""

    def __init__(self, obs_dim=16, n_z=8, n_c=8, h_dim=48):
        super().__init__()
        self.n_z, self.n_c = n_z, n_c
        self.z_dim = n_z * n_c
        self.gru = nn.GRUCell(self.z_dim, h_dim)
        self.prior = nn.Linear(h_dim, self.z_dim)
        self.post = nn.Sequential(nn.Linear(h_dim + obs_dim, 64), nn.SiLU(), nn.Linear(64, self.z_dim))
        # 解码器故意只依赖 z_t（不给 h_t），直接检验 z_t 是否被 KL 压塌！
        self.dec = nn.Sequential(nn.Linear(self.z_dim, 64), nn.SiLU(), nn.Linear(64, obs_dim))

    def forward(self, obs_seq):
        B, T, _ = obs_seq.shape
        h = torch.zeros(B, 48)
        z = torch.zeros(B, self.z_dim)
        recons, posts, priors = [], [], []
        for t in range(T):
            h = self.gru(z, h)
            pr = self.prior(h).reshape(B, self.n_z, self.n_c)
            po = self.post(torch.cat([h, obs_seq[:, t]], dim=-1)).reshape(B, self.n_z, self.n_c)
            z_oh, _ = sample_categorical_st(po, unimix=0.01)
            z = z_oh.reshape(B, self.z_dim)
            recons.append(self.dec(z))
            posts.append(po)
            priors.append(pr)
        return torch.stack(recons, dim=1), torch.stack(posts, dim=1), torch.stack(priors, dim=1)


def make_synthetic_sequences(n_seq=256, seq_len=12, obs_dim=16):
    """合成带偶发随机跳变的正弦波序列：既需要记忆，又需要每步用后验编码新跳变信息。"""
    rng = np.random.default_rng(SEED)
    data = np.zeros((n_seq, seq_len, obs_dim), dtype=np.float32)
    freqs = np.linspace(0.5, 2.5, obs_dim)
    for i in range(n_seq):
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.6, 1.4)
        for t in range(seq_len):
            if t == 5 and rng.random() < 0.5:
                phase += np.pi * 0.7  # 随机相位跳变，无法仅靠历史猜中
            data[i, t] = amp * np.sin(freqs * (t * 0.4) + phase) + rng.normal(0, 0.05, size=obs_dim)
    return torch.from_numpy(data)


def run_ablation_2x2():
    print("\n" + "=" * 60)
    print("第二部分：2×2 消融实验（有无 KL balancing × 有无 free bits）")
    print("=" * 60)

    seqs = make_synthetic_sequences()
    n_steps = 120 if is_quick() else 450
    # 故意把 KL 整体放大（模拟单帧像素较少或 KL 权重偏大的场景），清晰放大四组差异
    configs = [
        ("无 balancing + 无 free bits (β_dyn=2.0, β_rep=2.0, free=0)", 2.0, 2.0, 0.0),
        ("无 balancing + 有 free bits (β_dyn=2.0, β_rep=2.0, free=1)", 2.0, 2.0, 1.0),
        ("有 balancing + 无 free bits (β_dyn=2.0, β_rep=0.2, free=0)", 2.0, 0.2, 0.0),
        ("有 balancing + 有 free bits (β_dyn=2.0, β_rep=0.2, free=1) [V3]", 2.0, 0.2, 1.0),
    ]

    results = {}
    for label, b_dyn, b_rep, free in configs:
        seed_everything(SEED)
        model = ToySeqCategoricalModel()
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        kl_hist, rec_hist = [], []
        for step in range(n_steps):
            idx = torch.randint(0, len(seqs), (32,))
            batch = seqs[idx]
            recon, po, pr = model(batch)
            rec_loss = 0.5 * (recon - batch).pow(2).sum(dim=-1).mean()
            kl_loss, raw_kl, _, _ = kl_loss_balanced(po, pr, beta_dyn=b_dyn, beta_rep=b_rep, free_nats=free)
            loss = rec_loss + kl_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            kl_hist.append(raw_kl)
            rec_hist.append(rec_loss.item())
        results[label] = (np.array(kl_hist), np.array(rec_hist))
        print(f"  {label:<64} → 末段 KL = {np.mean(kl_hist[-20:]):5.2f} nat ｜ 重建损失 = {np.mean(rec_hist[-20:]):5.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    short_names = ["无 bal + 无 free", "无 bal + 有 free", "有 bal + 无 free", "有 bal + 有 free (V3)"]
    for (label, (kl_h, rec_h)), sn in zip(results.items(), short_names):
        axes[0].plot(kl_h, linewidth=2, label=sn)
        axes[1].plot(rec_h, linewidth=2, label=sn)
    axes[0].axhline(1.0, color="gray", linestyle=":", label="free bits = 1 nat")
    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("未截断的真实 KL (nat)")
    axes[0].set_title("无 free bits 且 β_rep 偏大时，KL 被迅速压塌")
    axes[0].legend(fontsize=8.5)

    axes[1].set_xlabel("训练步数")
    axes[1].set_ylabel("重建损失（越低越好）")
    axes[1].set_title("KL balancing + free bits 保住表征信息，重建误差最低")
    axes[1].legend(fontsize=8.5)
    save_figure(fig, OUT / "05-kl-balancing-ablation.png")


def main():
    seed_everything(SEED)
    verify_hand_calculations()
    run_ablation_2x2()
    print("\n" + "=" * 60)
    print("本节完成！下一节：06-symlog-and-twohot.py（Dreamer V3 的 symlog 与 two-hot 离散回归）")
    print("=" * 60)


if __name__ == "__main__":
    main()
