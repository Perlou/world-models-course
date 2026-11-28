"""
Phase 9 · 第 10 课：Dreamer 4、Shortcut Forcing 与视频基础模型检查表
===================================================================

学习目标：
    1. 实现玩具序列版 Shortcut Forcing（每帧独立路径时间 u_t + 步长条件 d + 自洽自举损失）
    2. 在受动作控制的二维多模态轨迹序列上，对比「普通 Diffusion Forcing」与「Shortcut Forcing」
       在 N ∈ {1, 2, 4} 采样步数下的多步自回归展开误差
    3. 汇总对比本章四类代表性世界模型路线（像素扩散 / 隐空间扩散 / Diffusion Forcing / 精简版 Genie）
       在统一环境指标上的表现与权衡
    4. 输出视频基础模型作为世界模型的五维检查表（CONCEPT.md 第 10.7 节）

对应概念文档：
    src/phase-09-diffusion-video-world-models/CONCEPT.md 第 10 节

运行方式：
    uv run python src/phase-09-diffusion-video-world-models/10-dreamer4-and-video-foundation-models.py
    WM_QUICK=1 uv run python src/phase-09-diffusion-video-world-models/10-dreamer4-and-video-foundation-models.py

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
_spec01 = importlib.util.spec_from_file_location("p9_01_d4", os.path.join(_DIR, "01-ddpm-to-edm.py"))
_mod01 = importlib.util.module_from_spec(_spec01)
_spec01.loader.exec_module(_mod01)
SinusoidalEmb = _mod01.SinusoidalEmb

plt = setup_matplotlib()
OUT = output_dir("phase-09")
SEED = 0


# ==================== 第一部分：玩具序列与 Shortcut Forcing ====================


def generate_controlled_bimodal_sequences(n_seq=1500, seq_len=10, seed=0):
    """生成带动作控制与局部双峰随机扰动的二维隐状态序列 [N, T, 2] 与动作 [N, T, 2]。"""
    rng = np.random.default_rng(seed)
    xs = np.zeros((n_seq, seq_len, 2), dtype=np.float32)
    acts = rng.uniform(-1.0, 1.0, size=(n_seq, seq_len, 2)).astype(np.float32)
    xs[:, 0] = rng.uniform(-0.5, 0.5, size=(n_seq, 2)).astype(np.float32)
    for t in range(1, seq_len):
        # 确定性控制动力学 + 随机二值横向偏转（多模态），让 1 步瞬时速度必然偏糊/偏离
        a = acts[:, t - 1]
        sign = rng.choice([-1.0, 1.0], size=(n_seq, 1)).astype(np.float32)
        perp = np.stack([-a[:, 1], a[:, 0]], axis=-1)
        nxt = 0.85 * xs[:, t - 1] + 0.35 * a + 0.18 * sign * perp
        xs[:, t] = np.clip(nxt, -2.0, 2.0)
    return torch.from_numpy(xs), torch.from_numpy(acts)


class CausalSequenceFlowModel(nn.Module):
    """因果 Transformer 序列模型：接收 (z̃_{1:T}, u_{1:T}, d_{1:T}, a_{1:T})，输出每帧速度 v̄_θ。"""

    def __init__(self, state_dim=2, act_dim=2, d_model=96, n_heads=4, n_layers=2, max_len=16):
        super().__init__()
        self.u_emb = SinusoidalEmb(32)
        self.d_emb = SinusoidalEmb(32)
        self.in_proj = nn.Linear(state_dim + act_dim + 32, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_head = nn.Linear(d_model, state_dim)

    def forward(self, z_noisy, u, d, actions):
        # z_noisy: [B, T, D_s], u/d: [B, T], actions: [B, T, D_a]
        B, T, _ = z_noisy.shape
        cond = (self.u_emb(u.reshape(B * T)) + self.d_emb(d.reshape(B * T))).reshape(B, T, -1)
        h = self.in_proj(torch.cat([z_noisy, actions, cond], dim=-1))
        h = h + self.pos_emb(torch.arange(T, device=z_noisy.device)).unsqueeze(0)
        mask = torch.triu(torch.ones(T, T, device=z_noisy.device, dtype=torch.bool), diagonal=1)
        return self.out_head(self.tf(h, mask=mask))


def shortcut_forcing_loss(model, z_clean, actions, use_shortcut=True, d_choices=(0.125, 0.25, 0.5)):
    """CONCEPT.md 第 10.3 节：每帧独立 u_t + 步长 d 的自洽自举（use_shortcut=False 即普通 Diffusion Forcing）。"""
    B, T, _ = z_clean.shape
    device = z_clean.device

    # 1. 普通每帧独立 Flow Matching 项（u_t ~ U(0,1)，d=0）
    eps = torch.randn_like(z_clean)
    u = torch.rand(B, T, device=device)
    # 以 25% 概率让前 2 帧几乎干净，贴合自回归推理条件
    clean_ctx = torch.rand(B, device=device) < 0.25
    u[clean_ctx, :2] = 0.99
    z_u = (1.0 - u.unsqueeze(-1)) * eps + u.unsqueeze(-1) * z_clean
    v_pred = model(z_u, u, torch.zeros_like(u), actions)
    loss_fm = F.mse_loss(v_pred, z_clean - eps)

    if not use_shortcut:
        return loss_fm

    # 2. Shortcut 自洽项（对序列中每帧或末帧做两小步自举）
    n_sc = B // 2
    z_s = z_clean[:n_sc]
    a_s = actions[:n_sc]
    eps_s = torch.randn_like(z_s)
    # 历史帧给干净或轻微噪声，最后目标帧做 shortcut 自洽
    u_s = torch.full((n_sc, T), 0.95, device=device)
    d_mat = torch.zeros(n_sc, T, device=device)
    d_pick = torch.tensor(d_choices, device=device)[torch.randint(0, len(d_choices), (n_sc,), device=device)]
    u_last = torch.zeros(n_sc, device=device)
    for i in range(n_sc):
        s2 = 2.0 * float(d_pick[i].item())
        u_last[i] = float(np.random.randint(0, max(1, int(round(1.0 / s2))))) * s2
    u_s[:, -1] = u_last
    z_u_s = (1.0 - u_s.unsqueeze(-1)) * eps_s + u_s.unsqueeze(-1) * z_s

    d_in = torch.where(d_pick <= d_choices[0] + 1e-6, torch.zeros_like(d_pick), d_pick)
    d_mat_in = d_mat.clone(); d_mat_in[:, -1] = d_in
    with torch.no_grad():
        v1 = model(z_u_s, u_s, d_mat_in, a_s)[:, -1]
        z_mid = z_u_s.clone()
        z_mid[:, -1] = z_u_s[:, -1] + d_pick.unsqueeze(-1) * v1
        u_mid = u_s.clone()
        u_mid[:, -1] = (u_last + d_pick).clamp_max(1.0)
        v2 = model(z_mid, u_mid, d_mat_in, a_s)[:, -1]
        v_tgt = 0.5 * (v1 + v2)

    d_mat_2d = d_mat.clone(); d_mat_2d[:, -1] = 2.0 * d_pick
    v_2d = model(z_u_s, u_s, d_mat_2d, a_s)[:, -1]
    loss_sc = F.mse_loss(v_2d, v_tgt)
    return loss_fm + loss_sc


@torch.no_grad()
def few_step_rollout(model, z_init, actions, n_steps=1, use_step_cond=True, u_stab=0.95):
    """从前 m 帧出发自回归展开，每帧仅走 n_steps 步 ODE/shortcut。"""
    model.eval()
    B, m, D = z_init.shape
    T_total = actions.shape[1]
    device = z_init.device
    hist_z = z_init.clone()
    hist_u = torch.full((B, m), 0.99, device=device)
    preds = []
    dt = 1.0 / n_steps

    for t in range(m, T_total):
        cur_z = torch.randn(B, D, device=device)
        a_seq = actions[:, : t + 1]
        d_val = dt if use_step_cond else 0.0
        for s in range(n_steps):
            u_cur = s * dt
            seq_z = torch.cat([hist_z, cur_z.unsqueeze(1)], dim=1)
            seq_u = torch.cat([hist_u, torch.full((B, 1), u_cur, device=device)], dim=1)
            seq_d = torch.zeros_like(seq_u); seq_d[:, -1] = d_val
            v = model(seq_z, seq_u, seq_d, a_seq)[:, -1]
            cur_z = cur_z + dt * v
        preds.append(cur_z)
        # 带轻微稳定化噪声追加进历史
        z_hist = u_stab * cur_z + (1.0 - u_stab) * torch.randn_like(cur_z)
        hist_z = torch.cat([hist_z, z_hist.unsqueeze(1)], dim=1)
        hist_u = torch.cat([hist_u, torch.full((B, 1), u_stab, device=device)], dim=1)
    return torch.stack(preds, dim=1)


def bimodal_trajectory_distance(pred, gt_prev, acts):
    """衡量预测状态离两个合法物理分支中较近那一支的距离（避免对随机双峰取平均的误判）。"""
    # pred: [B, H, 2], gt_prev: [B, H, 2], acts: [B, H, 2]
    perp = torch.stack([-acts[..., 1], acts[..., 0]], dim=-1)
    branch_p = (0.85 * gt_prev + 0.35 * acts + 0.18 * perp).clamp(-2.0, 2.0)
    branch_m = (0.85 * gt_prev + 0.35 * acts - 0.18 * perp).clamp(-2.0, 2.0)
    d_p = (pred - branch_p).norm(dim=-1)
    d_m = (pred - branch_m).norm(dim=-1)
    return float(torch.minimum(d_p, d_m).mean().item())


def video_model_checklist():
    print("\n" + "=" * 60)
    print("第三部分：视频基础模型作为世界模型的五维检查表（CONCEPT.md 第 10.7 节）")
    print("=" * 60)
    items = [
        ("1. 动作条件粒度", "能否接收真实控制指令（如末端 6DoF 位姿/关节角），而非仅文字或粗粒度相机运动？"),
        ("2. 物理外推真实性", "分布外组合（新物体质量、碰撞角度）时是遵守物理规律还是套用训练外观模板？"),
        ("3. 长期一致性与记忆", "视角移开再转回时，被遮挡物体的位置与状态是否保持不变？"),
        ("4. 决策导向评估", "是否报告了策略排序相关性（Phase 11）或真机任务成功率，而非仅 FVD/PSNR？"),
        ("5. 实时性与可控预算", "每帧延迟 N × t_网络 + t_解码 能否满足闭环控制频率（如 ≥10–20 Hz）？"),
    ]
    for title, desc in items:
        print(f"  [ ] {title:16s}：{desc}")


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    print("=" * 60)
    print("第一部分：玩具多模态序列上对比 Diffusion Forcing 与 Shortcut Forcing")
    print("=" * 60)
    z_tr, a_tr = generate_controlled_bimodal_sequences(600 if quick else 2000, seq_len=10, seed=SEED)
    z_te, a_te = generate_controlled_bimodal_sequences(128, seq_len=10, seed=SEED + 99)

    m_df = CausalSequenceFlowModel(state_dim=2, act_dim=2, d_model=96, n_heads=4, n_layers=2)
    m_sf = CausalSequenceFlowModel(state_dim=2, act_dim=2, d_model=96, n_heads=4, n_layers=2)
    opt_df = torch.optim.Adam(m_df.parameters(), lr=2e-3)
    opt_sf = torch.optim.Adam(m_sf.parameters(), lr=2e-3)

    n_iters = 250 if quick else 1500
    for _ in range(n_iters):
        idx = torch.randint(0, len(z_tr), (64,))
        zb, ab = z_tr[idx], a_tr[idx]

        opt_df.zero_grad()
        shortcut_forcing_loss(m_df, zb, ab, use_shortcut=False).backward()
        nn.utils.clip_grad_norm_(m_df.parameters(), 1.0)
        opt_df.step()

        opt_sf.zero_grad()
        shortcut_forcing_loss(m_sf, zb, ab, use_shortcut=True).backward()
        nn.utils.clip_grad_norm_(m_sf.parameters(), 1.0)
        opt_sf.step()

    print(f"  训练完成（各 {n_iters} 步，{time.time() - t0:.1f}s）")
    print(f"\n  自回归展开 8 步评估（最近合法分支误差越低越锐利、越少平均发糊）：")
    print(f"  {'步数 N':>6s} | {'Diffusion Forcing 分支误差':>26s} | {'Shortcut Forcing 分支误差':>25s} | {'DF 全程MSE':>10s} | {'SF 全程MSE':>10s}")
    res = {}
    for N in [1, 2, 4]:
        p_df = few_step_rollout(m_df, z_te[:, :2], a_te, n_steps=N, use_step_cond=False)
        p_sf = few_step_rollout(m_sf, z_te[:, :2], a_te, n_steps=N, use_step_cond=True)
        # 单步分支贴合度（第一步从真实 z_te[:, 1] 出发）与 8 步展开状态 MSE
        b_df = bimodal_trajectory_distance(p_df[:, :1], z_te[:, 1:2], a_te[:, 1:2])
        b_sf = bimodal_trajectory_distance(p_sf[:, :1], z_te[:, 1:2], a_te[:, 1:2])
        mse_df = float(F.mse_loss(p_df, z_te[:, 2:]).item())
        mse_sf = float(F.mse_loss(p_sf, z_te[:, 2:]).item())
        res[N] = (b_df, b_sf, mse_df, mse_sf)
        print(f"  {N:>6d} | {b_df:26.4f} | {b_sf:25.4f} | {mse_df:10.4f} | {mse_sf:10.4f}")

    print("\n" + "=" * 60)
    print("第二部分：本章四类世界模型路线结构与工程特性汇总")
    print("=" * 60)
    summary_rows = [
        ("像素空间扩散 (03/04 DIAMOND)", "32×32 像素", "U-Net + EDM", "N=3 步去噪", "保小物体细节最好，高分辨率代价大"),
        ("隐空间扩散 (06)", "8×8×4 连续隐变量", "因果 VAE + EDM", "N=3 步 + 解码", "显存与单帧耗时大幅降低，受重建上限约束"),
        ("Diffusion Forcing (05)", "每帧独立 σ_t", "因果序列扩散", "N=3 步 + σ_stab", "长程开环漂移明显缓于朴素 AR"),
        ("精简版 Genie (07/08)", "4×4 离散 token + 隐动作", "ST-TF + MaskGIT", "R=4 轮并行解码", "无需动作标签，可控性受窄瓶颈码本约束"),
        ("Shortcut Forcing (09/10)", "每帧独立 u_t + 步长 d", "因果序列流模型", "N=1–2 步", "一步直达不塌向均值，面向实时交互"),
    ]
    for name, rep, bb, cost, note in summary_rows:
        print(f"  - {name:26s} | {rep:16s} | {bb:15s} | {cost:14s} | {note}")

    video_model_checklist()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ns = [1, 2, 4]
    axes[0].plot(ns, [res[n][0] for n in ns], "o--", label="Diffusion Forcing")
    axes[0].plot(ns, [res[n][1] for n in ns], "s-", label="Shortcut Forcing")
    axes[0].set_title("多模态分支贴合误差（越低越少发糊）"); axes[0].set_xlabel("采样步数 N"); axes[0].legend()

    axes[1].plot(ns, [res[n][2] for n in ns], "o--", label="Diffusion Forcing")
    axes[1].plot(ns, [res[n][3] for n in ns], "s-", label="Shortcut Forcing")
    axes[1].set_title("8 步自回归展开 MSE"); axes[1].set_xlabel("采样步数 N"); axes[1].legend()

    fig.tight_layout()
    save_figure(fig, OUT / "10-shortcut-forcing-comparison.png")
    print(f"\n本章全部 10 节讲义完成（本节用时 {time.time() - t0:.1f}s）！接下来：实战项目 projects/phase-09-playable-neural-simulator/")


if __name__ == "__main__":
    main()
