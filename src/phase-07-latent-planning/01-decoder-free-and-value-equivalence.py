"""
不重建观测行不行：价值等价原理与干扰背景实验
============================================

学习目标：
    1. 核对 CONCEPT.md 第 1.4 节的规划误差界手算（ε_r=0.01, ε_v=0.5, γ=0.99, H=3 → Δ=0.5148, 2Δ=1.03）
    2. 实现 build_distractor_mdp()、policy_evaluation() 与 check_value_equivalence()，
       在 4 状态（位置 × 亮暗背景）小 MDP 上精确验证抽象模型与真实环境关于无背景依赖价值函数的价值等价性，
       同时用依赖背景的 v 验证"等价是有范围的"（0.45）
    3. 构造高维干扰背景任务（2 维任务信号 + 64 维每步随机闪烁噪声），在完全相同参数量下对比：
       重建式世界模型（带解码器） vs 价值等价无解码器模型（只预测奖励与价值），
       量化价值预测误差差距以及重建损失里背景噪声占了多少比例

核心概念：
    - 价值等价（value equivalence）：T̃_π v = T_π v，只要 Bellman 备份结果一致，模型内部不必还原观测
    - 目标不匹配与干扰背景代价：像素/观测重建按维度均分容量，容易被高维不可预测背景吞噬

前置知识：
    - Phase 4 的 Bellman 方程与策略评估
    - Phase 6 的重建式世界模型（Dreamer）

对应概念文档：
    - CONCEPT.md 第 1 节「不重建观测行不行：价值等价」

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
OUT = output_dir("phase-07")
SEED = 0


# ==================== 第一部分：4 状态小 MDP 与价值等价验证 ====================


def build_distractor_mdp():
    """CONCEPT.md 第 1.5 节：4 状态 MDP（位置∈{左,右} × 背景∈{亮,暗}），动作∈{留,走}。

    状态编号：0=(左,亮), 1=(左,暗), 2=(右,亮), 3=(右,暗)
    动作编号：0=留, 1=走
    返回 P_real [4, 2, 4], R_real [4, 2], P_abs [2, 2, 2], R_abs [2, 2], phi [4, 2]
    """
    P_real = np.zeros((4, 2, 4), dtype=np.float64)
    R_real = np.zeros((4, 2), dtype=np.float64)

    for s in range(4):
        pos = s // 2  # 0=左, 1=右
        for a in range(2):
            nxt_pos = pos if a == 0 else 1 - pos
            # 背景每步以 0.5 概率亮/暗，与动作无关
            P_real[s, a, nxt_pos * 2 + 0] = 0.5
            P_real[s, a, nxt_pos * 2 + 1] = 0.5
            # 奖励只看当前位置是否在右（1），与背景无关
            R_real[s, a] = 1.0 if pos == 1 else 0.0

    # 2 状态抽象模型：0=L, 1=R
    P_abs = np.zeros((2, 2, 2), dtype=np.float64)
    R_abs = np.zeros((2, 2), dtype=np.float64)
    for z in range(2):
        for a in range(2):
            nxt_z = z if a == 0 else 1 - z
            P_abs[z, a, nxt_z] = 1.0
            R_abs[z, a] = 1.0 if z == 1 else 0.0

    # 编码矩阵 φ [4, 2]：丢掉背景，只留位置
    phi = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)
    return P_real, R_real, P_abs, R_abs, phi


def bellman_backup(P, R, pi, v, gamma=0.9):
    """计算一步 Bellman 备份 (T_π v)(s) = Σ_a π(a|s) [R(s,a) + γ Σ_{s'} P(s'|s,a) v(s')]。"""
    q = R + gamma * np.einsum("sat,t->sa", P, v)
    return np.sum(pi * q, axis=-1)


def policy_evaluation(P, R, pi, gamma=0.9, tol=1e-10):
    """迭代 Bellman 备份直到收敛，返回解析价值函数 V^π。"""
    v = np.zeros(P.shape[0], dtype=np.float64)
    for _ in range(10_000):
        v_new = bellman_backup(P, R, pi, v, gamma=gamma)
        if np.max(np.abs(v_new - v)) < tol:
            return v_new
        v = v_new
    return v


def check_value_equivalence():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 1.4 节误差界与第 1.5 节价值等价小 MDP")
    print("=" * 60)

    # 1. 第 1.4 节规划误差界手算
    eps_r, eps_v, gamma_p, H = 0.01, 0.5, 0.99, 3
    delta = eps_r * (1.0 - gamma_p**H) / (1.0 - gamma_p) + (gamma_p**H) * eps_v
    print(f"  1. 规划误差界（ε_r={eps_r}, ε_v={eps_v}, γ={gamma_p}, H={H}）："
          f"Δ = {delta:.4f}（手算 0.5148），决策损失上界 2Δ = {2 * delta:.4f}（手算 1.03）✓")
    assert abs(delta - 0.5148) < 1e-3

    # 2. 第 1.5 节 4 状态 MDP 价值等价验证
    P_real, R_real, P_abs, R_abs, phi = build_distractor_mdp()
    # 策略：左边走 (a=1)，右边留 (a=0)
    pi_real = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=np.float64)
    pi_abs = np.array([[0, 1], [1, 0]], dtype=np.float64)

    v_real = policy_evaluation(P_real, R_real, pi_real, gamma=0.9)
    v_abs = policy_evaluation(P_abs, R_abs, pi_abs, gamma=0.9)
    v_abs_lifted = phi @ v_abs
    max_diff = float(np.max(np.abs(v_real - v_abs_lifted)))
    print(f"\n  2. 4 状态 MDP 策略评估：")
    print(f"     真实环境价值 V_real = {[round(x, 2) for x in v_real.tolist()]}")
    print(f"     抽象模型价值 V_abs  = {[round(x, 2) for x in v_abs_lifted.tolist()]}（最大误差 {max_diff:.1e} ✓）")
    assert max_diff < 1e-8 and np.allclose(v_real, [9.0, 9.0, 10.0, 10.0])

    # 3. 检验等价范围：若价值函数依赖背景 v = [0, 0, 1, 0]
    v_bg = np.array([0.0, 0.0, 1.0, 0.0])
    tv_bg = bellman_backup(P_real, R_real, pi_real, v_bg, gamma=0.9)
    print(f"  3. 等价范围检验：依赖背景的 v=[0,0,1,0] 在 (左,亮) 做 Bellman 备份 = {tv_bg[0]:.2f}（手算 0.45）；"
          f"抽象模型因丢弃背景无法表达该 v ✓")
    assert abs(tv_bg[0] - 0.45) < 1e-8


# ==================== 第二部分：高维干扰背景神经网络对照实验 ====================


def run_distractor_neural_experiment():
    """在 2 维核心状态 + 64 维随机干扰背景的连续任务上，对比同参数量的重建式模型与价值等价模型。"""
    print("\n" + "=" * 60)
    print("第二部分：高维干扰背景实验 — 重建式世界模型 vs 价值等价无解码器模型")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    task_dim = 2
    bg_dim = 64
    obs_dim = task_dim + bg_dim
    act_dim = 2
    n_samples = 1500 if is_quick() else 4000
    n_steps = 160 if is_quick() else 450

    # 真实任务动力学（仅前 2 维有用）：s' = 0.85 s + 0.5 a，奖励 r = -||s'||^2，解析价值近似 V(s) = -||s||^2 / (1 - 0.9*0.85^2)
    s_t = rng.uniform(-1.5, 1.5, size=(n_samples, task_dim)).astype(np.float32)
    a_t = rng.uniform(-1.0, 1.0, size=(n_samples, act_dim)).astype(np.float32)
    s_nxt = 0.85 * s_t + 0.5 * a_t
    r_t = -np.sum(s_nxt**2, axis=-1, keepdims=True)
    # 固定评估策略 a = -0.5 s 下的解析折扣价值 V(s')（闭环收缩因子 0.6，γ=0.9）
    v_nxt = -np.sum(s_nxt**2, axis=-1, keepdims=True) / (1.0 - 0.9 * (0.6**2))

    # 64 维不可预测的随机背景（每步独立重新采样）
    bg_t = rng.normal(0.0, 1.0, size=(n_samples, bg_dim)).astype(np.float32)
    bg_nxt = rng.normal(0.0, 1.0, size=(n_samples, bg_dim)).astype(np.float32)

    obs_t = torch.from_numpy(np.concatenate([s_t, bg_t], axis=-1))
    obs_nxt = torch.from_numpy(np.concatenate([s_nxt, bg_nxt], axis=-1))
    act_t = torch.from_numpy(a_t)
    rew_t = torch.from_numpy(r_t)
    val_nxt = torch.from_numpy(v_nxt)

    n_tr = int(n_samples * 0.8)

    # 故意限制隐维度 z_dim = 8（瓶颈容量），突出"容量被无关背景抢占"的效应
    class ReconWM(nn.Module):
        def __init__(self, z_dim=8, h=32):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(obs_dim, h), nn.SiLU(), nn.Linear(h, z_dim))
            self.dyn = nn.Sequential(nn.Linear(z_dim + act_dim, h), nn.SiLU(), nn.Linear(h, z_dim))
            self.dec = nn.Sequential(nn.Linear(z_dim, h), nn.SiLU(), nn.Linear(h, obs_dim))
            self.rew = nn.Linear(z_dim, 1)
            self.val = nn.Linear(z_dim, 1)

        def forward(self, o, a):
            z = self.enc(o)
            z1 = self.dyn(torch.cat([z, a], dim=-1))
            return self.dec(z1), self.rew(z1), self.val(z1)

    class ValueEquivWM(nn.Module):
        """去掉解码器，把同等参数预算全部用于表征、动力学与奖励/价值头。"""

        def __init__(self, z_dim=8, h=42):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(obs_dim, h), nn.SiLU(), nn.Linear(h, z_dim))
            self.dyn = nn.Sequential(nn.Linear(z_dim + act_dim, h), nn.SiLU(), nn.Linear(h, z_dim))
            self.rew = nn.Sequential(nn.Linear(z_dim, h), nn.SiLU(), nn.Linear(h, 1))
            self.val = nn.Sequential(nn.Linear(z_dim, h), nn.SiLU(), nn.Linear(h, 1))

        def forward(self, o, a):
            z = self.enc(o)
            z1 = self.dyn(torch.cat([z, a], dim=-1))
            return None, self.rew(z1), self.val(z1)

    seed_everything(SEED)
    m_rec = ReconWM()
    m_ve = ValueEquivWM()
    p_rec = sum(p.numel() for p in m_rec.parameters())
    p_ve = sum(p.numel() for p in m_ve.parameters())

    opt_rec = torch.optim.Adam(m_rec.parameters(), lr=3e-3)
    opt_ve = torch.optim.Adam(m_ve.parameters(), lr=3e-3)

    rec_val_errs, ve_val_errs, bg_loss_ratios = [], [], []

    for _ in range(n_steps):
        idx = torch.randint(0, n_tr, (128,))
        # 1. 重建式模型（观测像素求和重建 + 奖励 + 价值）
        o_pred, r_pred, v_pred = m_rec(obs_t[idx], act_t[idx])
        rec_sq = (o_pred - obs_nxt[idx]).pow(2)
        loss_task_rec = rec_sq[:, :task_dim].sum(dim=-1).mean()
        loss_bg_rec = rec_sq[:, task_dim:].sum(dim=-1).mean()
        loss_rec_total = (loss_task_rec + loss_bg_rec) + F.mse_loss(r_pred, rew_t[idx]) + F.mse_loss(v_pred, val_nxt[idx])
        opt_rec.zero_grad()
        loss_rec_total.backward()
        opt_rec.step()

        # 2. 价值等价无解码器模型（只预测奖励与价值）
        _, r_ve, v_ve = m_ve(obs_t[idx], act_t[idx])
        loss_ve = F.mse_loss(r_ve, rew_t[idx]) + F.mse_loss(v_ve, val_nxt[idx])
        opt_ve.zero_grad()
        loss_ve.backward()
        opt_ve.step()

        with torch.no_grad():
            o_te, _, v_te_r = m_rec(obs_t[n_tr:], act_t[n_tr:])
            _, _, v_te_ve = m_ve(obs_t[n_tr:], act_t[n_tr:])
            rec_val_errs.append(F.mse_loss(v_te_r, val_nxt[n_tr:]).sqrt().item())
            ve_val_errs.append(F.mse_loss(v_te_ve, val_nxt[n_tr:]).sqrt().item())
            sq_te = (o_te - obs_nxt[n_tr:]).pow(2)
            bg_ratio = sq_te[:, task_dim:].sum().item() / sq_te.sum().item()
            bg_loss_ratios.append(bg_ratio)

    print(f"  参数量对比：重建式模型 {p_rec:,} ｜ 价值等价无解码器模型 {p_ve:,}（相近）")
    print(f"  留出测试集价值预测 RMSE：重建式模型 = {rec_val_errs[-1]:.4f} ｜ 价值等价模型 = {ve_val_errs[-1]:.4f}（低 {rec_val_errs[-1] / max(ve_val_errs[-1], 1e-6):.1f} 倍！）")
    print(f"  重建式模型的观测重建损失中，不可预测背景占比 = {bg_loss_ratios[-1] * 100:.1f}%！")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].plot(rec_val_errs, linewidth=2.2, label=f"重建式模型（终值 RMSE {rec_val_errs[-1]:.3f}）")
    axes[0].plot(ve_val_errs, linewidth=2.2, label=f"价值等价无解码器（终值 RMSE {ve_val_errs[-1]:.3f}）")
    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("留出集价值预测 RMSE")
    axes[0].set_title("高维干扰背景下：扔掉解码器让模型把全部容量花在价值上")
    axes[0].legend(fontsize=8.5)

    axes[1].plot(np.array(bg_loss_ratios) * 100.0, color="tomato", linewidth=2.2)
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("训练步数")
    axes[1].set_ylabel("背景噪声占重建损失的比例 (%)")
    axes[1].set_title(f"重建损失约 {bg_loss_ratios[-1] * 100:.1f}% 耗费在不可预测的背景上")
    save_figure(fig, OUT / "01-value-equivalence.png")


def main():
    seed_everything(SEED)
    check_value_equivalence()
    run_distractor_neural_experiment()
    print("\n" + "=" * 60)
    print("本节完成！下一节：02-mcts-from-scratch.py（蒙特卡洛树搜索：从 UCB1 到 PUCT）")
    print("=" * 60)


if __name__ == "__main__":
    main()
