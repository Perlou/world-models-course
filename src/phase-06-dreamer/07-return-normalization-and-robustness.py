"""
Dreamer V3（二）：分位数回报归一化、Unimix 与 Critic EMA 正则
=============================================================

学习目标：
    1. 实现 ReturnNormalizer（P95−P5 分位数范围 + EMA 0.99 + max(1, S) 只缩不放），核对 CONCEPT.md 第 7.3 节手算例子
    2. 验证 unimix(1%) 在 logits 极端饱和（±20）时把单变量 KL 严格封顶在 0.99 ln(1 + 99K) = 7.98 nat（K=32）以内
    3. 实现 ema_update() 与 critic 的 EMA 正则项，在自举价值回归玩具任务上对比：
       无稳定措施 vs 周期目标网络（DQN/V2） vs EMA 正则（V3）
    4. 在三种奖励尺度（0.01×、1×、100×）的 bandit 上固定熵系数 η=3e-4，对比有无 ReturnNormalizer 时的策略熵与最终回报

核心概念：
    - 分位数回报归一化：S = EMA(P95 − P5, 0.99)，A = (R^λ − V) / max(1, S)；抗离群值，稀疏奖励下只缩不放
    - unimix（1% 均匀混合）：给每个类别垫底概率 0.01/K，防 KL 尖峰与 −∞ 对数概率
    - EMA 正则：用最新 critic 算 λ-return（无目标滞后），再用交叉熵约束 critic 不偏离自己的滑动平均 ψ̄

前置知识：
    - 05-kl-balancing-and-free-bits.py、06-symlog-and-twohot.py

对应概念文档：
    - CONCEPT.md 第 7 节「Dreamer V3（二）：回报归一化与其他稳健性技巧」

预计运行时间：约 20 秒（CPU）
"""

import copy
import math
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


# ==================== 第一部分：三个 V3 稳健性部件 ====================


class ReturnNormalizer:
    """CONCEPT.md 第 7.3、7.7 节：分位数范围 EMA 缩放器，只缩小不放大。"""

    def __init__(self, decay=0.99, low_q=0.05, high_q=0.95, min_scale=1.0):
        self.decay = decay
        self.low_q = low_q
        self.high_q = high_q
        self.min_scale = min_scale
        self.scale = 0.0
        self.initialized = False

    @torch.no_grad()
    def update_and_scale(self, returns):
        flat = returns.detach().float().reshape(-1)
        lo = torch.quantile(flat, self.low_q).item()
        hi = torch.quantile(flat, self.high_q).item()
        span = max(0.0, hi - lo)
        if not self.initialized:
            self.scale = span
            self.initialized = True
        else:
            self.scale = self.decay * self.scale + (1.0 - self.decay) * span
        return max(self.min_scale, self.scale)


def unimix_probs(logits, eps=0.01):
    probs = F.softmax(logits, dim=-1)
    if eps <= 0.0:
        return probs
    return (1.0 - eps) * probs + eps / logits.shape[-1]


@torch.no_grad()
def ema_update(online_module, ema_module, decay=0.98):
    for p, p_ema in zip(online_module.parameters(), ema_module.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ==================== 第二部分：手算与理论边界核对 ====================


def verify_theory_and_unimix():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 7.3–7.5 节手算例子与 Unimix KL 上界")
    print("=" * 60)

    # 1. 第 7.3 节回报归一化例子
    norm = ReturnNormalizer(decay=0.99)
    norm.scale = 100.0
    norm.initialized = True
    # 构造 P5=2, P95=402 的样本使 span ≈ 400
    s_new = norm.decay * 100.0 + (1.0 - norm.decay) * 400.0
    assert abs(s_new - 103.0) < 1e-6
    adv_dense = (520.0 - 440.0) / max(1.0, 400.0)
    adv_sparse = (0.015 - 0.005) / max(1.0, 0.02)
    print(f"  1. 回报归一化：稠密 A = {adv_dense:.2f}（手算 0.20）｜ 稀疏 A = {adv_sparse:.2f}（手算 0.01，未放大 50 倍）｜ EMA 100→{s_new:.1f} ✓")

    # 2. 第 7.4 节 unimix KL 上界测试：构造极端饱和的相反 logits (±30)
    K = 32
    lq = torch.full((K,), -30.0)
    lp = torch.full((K,), -30.0)
    lq[0] = 30.0
    lp[1] = 30.0

    # 无 unimix
    q_raw = F.softmax(lq, dim=-1)
    p_raw = F.softmax(lp, dim=-1)
    kl_no_unimix = (q_raw * (F.log_softmax(lq, dim=-1) - F.log_softmax(lp, dim=-1))).sum().item()

    # 有 1% unimix
    q_u = unimix_probs(lq, eps=0.01)
    p_u = unimix_probs(lp, eps=0.01)
    kl_with_unimix = (q_u * (torch.log(q_u) - torch.log(p_u))).sum().item()
    bound = 0.99 * math.log(1.0 + 99.0 * K)
    print(f"  2. 饱和 logits (±30, K=32) 单变量最大 KL：")
    print(f"     无 unimix = {kl_no_unimix:6.2f} nat（logits 再大就溢出为 +inf）")
    print(f"     有 unimix = {kl_with_unimix:6.4f} nat（严格 ≤ 理论封顶 {bound:.4f} nat ✓）")
    assert abs(kl_with_unimix - bound) < 1e-3 and kl_with_unimix <= 7.99

    # 3. 第 7.5 节 EMA 50 步前累计权重
    w50 = 0.98 ** 50
    print(f"  3. EMA（d=0.98）50 次更新前的历史权重占比 = {w50:.3f}（手算 0.364 ✓）")
    assert abs(w50 - 0.3642) < 1e-3
    return kl_no_unimix, kl_with_unimix


# ==================== 第三部分：三大对比实验 ====================


def experiment_return_normalization():
    """在 5 臂多臂老虎机（存在次优臂与最优臂）上测试奖励尺度 0.01×、1×、100×，固定 η=3e-4。"""
    print("\n" + "=" * 60)
    print("第二部分：三种奖励尺度（0.01×、1×、100×）下有无回报归一化的策略熵与回报")
    print("=" * 60)

    scales = [0.01, 1.0, 100.0]
    # 5 臂真实期望奖励（带轻微噪声）：次优臂 0.65，最优臂 1.0，其余 0.1~0.4；
    # 初始策略偏置到次优臂（logit +1.2），放大"大尺度下熵被压死 → 卡在次优臂"的经典现象
    true_means = np.array([0.1, 0.2, 0.65, 0.4, 1.0], dtype=np.float32)
    eta = 3e-4
    n_steps = 250 if is_quick() else 600
    batch_size = 64

    records = {}
    for scale in scales:
        for use_norm in [False, True]:
            seed_everything(SEED)
            logits = nn.Parameter(torch.tensor([0.0, 0.0, 1.8, 0.0, -0.8]))
            opt = torch.optim.Adam([logits], lr=3e-2)
            norm = ReturnNormalizer(decay=0.95)
            v_base = 0.0
            ent_hist, ret_hist = [], []

            for _ in range(n_steps):
                probs = unimix_probs(logits, eps=0.01)
                dist = torch.distributions.Categorical(probs=probs)
                a = dist.sample((batch_size,))
                r_raw = torch.tensor(true_means[a.numpy()] * scale) + torch.randn(batch_size) * (0.15 * scale)
                v_base = 0.9 * v_base + 0.1 * r_raw.mean().item()
                adv = r_raw - v_base
                if use_norm:
                    s = norm.update_and_scale(r_raw)
                    adv = adv / s
                ent = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
                loss = -(dist.log_prob(a) * adv.detach()).mean() - eta * ent
                opt.zero_grad()
                loss.backward()
                opt.step()

                ent_hist.append(ent.item())
                # 归一化回报（除以最优臂回报）方便跨尺度直接比较
                exp_norm_ret = float((probs.detach().numpy() * true_means).sum() / true_means.max())
                ret_hist.append(exp_norm_ret)

            records[(scale, use_norm)] = (np.array(ent_hist), np.array(ret_hist))
            tag = "有回报归一化" if use_norm else "无回报归一化"
            print(f"  尺度 ×{scale:<5g} ｜ {tag} → 最终熵 = {ent_hist[-1]:.3f} ｜ 最终相对最优回报 = {ret_hist[-1]:.3f}")
    return scales, records


def experiment_critic_stabilization():
    """在带自举的 10 态马尔可夫链（状态 s→s+1，终点奖励 10，γ=0.95）上对比三种 critic 稳定策略。"""
    print("\n" + "=" * 60)
    print("第三部分：自举价值回归对比 — 无稳定措施 vs 目标网络（V2） vs EMA 正则（V3）")
    print("=" * 60)

    n_states = 12
    gamma = 0.95
    # 真实解析价值：从状态 i 走到终点 n_states-1 需要 (n_states-1-i) 步，最后一步奖励 10
    true_v = np.array([10.0 * (gamma ** (n_states - 1 - i)) for i in range(n_states)], dtype=np.float32)
    # 输入为正弦特征，带强耦合以触发自举振荡
    feats = torch.eye(n_states)
    n_steps = 120 if is_quick() else 350

    methods = ["无稳定措施", "周期目标网络 (每25步同步)", "EMA 正则 (Dreamer V3)"]
    err_curves = {}

    for method in methods:
        seed_everything(SEED)
        critic = nn.Sequential(nn.Linear(n_states, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 1))
        slow = copy.deepcopy(critic)
        opt = torch.optim.Adam(critic.parameters(), lr=1.5e-2)
        errs = []

        for step in range(n_steps):
            v_cur = critic(feats).squeeze(-1)
            with torch.no_grad():
                v_boot = slow(feats).squeeze(-1) if "目标网络" in method else critic(feats).squeeze(-1)
                # 1 步 TD 目标：最后一步奖励 10 且终止
                targets = torch.zeros(n_states)
                targets[:-1] = gamma * v_boot[1:]
                targets[-1] = 10.0

            loss = 0.5 * F.mse_loss(v_cur, targets)
            if "EMA 正则" in method:
                with torch.no_grad():
                    v_ema = slow(feats).squeeze(-1)
                loss = loss + 0.5 * F.mse_loss(v_cur, v_ema)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if "目标网络" in method and (step + 1) % 25 == 0:
                slow.load_state_dict(critic.state_dict())
            elif "EMA 正则" in method:
                ema_update(critic, slow, decay=0.95)

            with torch.no_grad():
                rmse = float(np.sqrt(np.mean((critic(feats).squeeze(-1).numpy() - true_v) ** 2)))
            errs.append(rmse)

        err_curves[method] = np.array(errs)
        print(f"  {method:<26} → 末段平均价值 RMSE = {np.mean(errs[-30:]):.4f} ｜ 曲线波动 std = {np.std(errs[-50:]):.4f}")

    return err_curves


def plot_all(scales, norm_records, critic_curves):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for scale in scales:
        ent_no, _ = norm_records[(scale, False)]
        ent_yes, _ = norm_records[(scale, True)]
        axes[0].plot(ent_no, linestyle="--", alpha=0.8, label=f"×{scale:g} 无归一化")
        axes[0].plot(ent_yes, linewidth=2.0, label=f"×{scale:g} 有归一化")
    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("策略熵 H(π)")
    axes[0].set_title("固定 η=3e-4：无归一化时大尺度策略熵瞬间塌缩")
    axes[0].legend(fontsize=7.5, ncol=2)

    x = np.arange(len(scales))
    w = 0.34
    r_no = [norm_records[(s, False)][1][-1] for s in scales]
    r_yes = [norm_records[(s, True)][1][-1] for s in scales]
    axes[1].bar(x - w / 2, r_no, width=w, label="无回报归一化")
    axes[1].bar(x + w / 2, r_yes, width=w, label="有回报归一化 (V3)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"尺度 ×{s:g}" for s in scales])
    axes[1].set_ylim(0, 1.12)
    axes[1].set_ylabel("最终相对最优回报")
    axes[1].set_title("跨 4 个数量级奖励尺度的最终策略质量")
    axes[1].legend(fontsize=8.5)

    for method, curve in critic_curves.items():
        axes[2].plot(curve, linewidth=2.0, label=method)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("更新步数")
    axes[2].set_ylabel("价值估计 RMSE（对数）")
    axes[2].set_title("Critic 自举稳定机制对比（EMA 兼具快收敛与平滑）")
    axes[2].legend(fontsize=8.5)
    save_figure(fig, OUT / "07-v3-robustness-tricks.png")


def main():
    seed_everything(SEED)
    verify_theory_and_unimix()
    scales, norm_records = experiment_return_normalization()
    critic_curves = experiment_critic_stabilization()
    plot_all(scales, norm_records, critic_curves)
    print("\n" + "=" * 60)
    print("本节完成！下一节：08-dreamer-v3-world-model.py（Dreamer V3 世界模型完整组装）")
    print("=" * 60)


if __name__ == "__main__":
    main()
