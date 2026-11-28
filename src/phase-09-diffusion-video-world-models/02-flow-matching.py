"""
Phase 9 · 第 02 课：Flow Matching 与 Rectified Flow
===================================================

学习目标：
    1. 实现线性插值路径 x_u = (1-u)·ε + u·x、条件 flow matching 损失与 Euler 采样
    2. 实现速度/干净样本换算及 u ↔ σ 坐标互换，核对 CONCEPT.md 第 2.4–2.5 节手算
    3. 在同一份 8-高斯环数据上对比 Flow Matching 与 EDM 在 1、2、4、8 步下的样本质量
    4. 实现一次 Reflow（生成配对再训练），定量测量轨迹弯曲度与少步样本质量的改善

对应概念文档：
    src/phase-09-diffusion-video-world-models/CONCEPT.md 第 2 节

运行方式：
    uv run python src/phase-09-diffusion-video-world-models/02-flow-matching.py
    WM_QUICK=1 uv run python src/phase-09-diffusion-video-world-models/02-flow-matching.py

预计运行时间：完整约 35 秒（M1 Max CPU），冒烟约 5 秒
"""

import importlib.util
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec01 = importlib.util.spec_from_file_location("p9_01", os.path.join(_DIR, "01-ddpm-to-edm.py"))
_mod01 = importlib.util.module_from_spec(_spec01)
_spec01.loader.exec_module(_mod01)

ToyMLP = _mod01.ToyMLP
EDMDenoiser = _mod01.EDMDenoiser
sample_train_sigmas = _mod01.sample_train_sigmas
sample_euler = _mod01.sample_euler
sample_8gaussians = _mod01.sample_8gaussians
eval_2d_samples = _mod01.eval_2d_samples

plt = setup_matplotlib()
OUT = output_dir("phase-09")
SEED = 0


# ==================== 第一部分：Flow Matching 与坐标换算 ====================


def interpolate(x, eps, u):
    """CONCEPT.md 第 2.2 节：直线条件路径 x_u = (1 - u)·ε + u·x（u=0 为纯噪声，u=1 为数据）。"""
    view = [-1] + [1] * (x.ndim - 1)
    u_v = u.view(*view)
    return (1.0 - u_v) * eps + u_v * x


def velocity_to_denoised(x_u, v, u):
    """CONCEPT.md 第 2.5 节：由当前点 x_u 和速度 v 换算干净样本预测 x̂ = x_u + (1 - u)·v。"""
    view = [-1] + [1] * (x_u.ndim - 1)
    return x_u + (1.0 - u.view(*view)) * v


def u_to_sigma(u):
    return (1.0 - u) / u.clamp_min(1e-6)


def sigma_to_u(sigma):
    return 1.0 / (1.0 + sigma)


def cfm_loss(model, x, eps=None):
    """CONCEPT.md 第 2.7 节：条件 flow matching 损失 E[||v_θ(x_u, u) - (x - ε)||²]。"""
    eps = torch.randn_like(x) if eps is None else eps
    u = torch.rand(x.shape[0], device=x.device)
    x_u = interpolate(x, eps, u)
    v_target = x - eps
    v_pred = model(x_u, u)
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample_flow_euler(model, shape, n_steps, device="cpu", init_eps=None, return_traj=False):
    """CONCEPT.md 第 2.7 节：从 u=0 沿速度场走 N 步 Euler 到 u=1。"""
    model.eval()
    x = torch.randn(shape, device=device) if init_eps is None else init_eps.to(device).clone()
    dt = 1.0 / n_steps
    traj = [x.clone()] if return_traj else None
    B = shape[0]
    for i in range(n_steps):
        u = torch.full((B,), i * dt, device=device)
        x = x + dt * model(x, u)
        if return_traj:
            traj.append(x.clone())
    return (x, torch.stack(traj, dim=0)) if return_traj else x


@torch.no_grad()
def reflow_pairs(model, n_pairs, data_dim=2, n_steps=20, device="cpu"):
    """CONCEPT.md 第 2.6 节：用已训练的 1-Flow 沿 ODE 积分生成 (ε, x̂) 配对，消除轨迹交叉。"""
    eps = torch.randn(n_pairs, data_dim, device=device)
    x_hat = sample_flow_euler(model, (n_pairs, data_dim), n_steps=n_steps, device=device, init_eps=eps)
    return eps, x_hat


def measure_straightness(traj):
    """计算轨迹弯曲度：路径总弧长 / 首尾直线弦长（完全直线 = 1.000）以及速度变化方差。"""
    # traj: [T+1, B, D]
    diffs = traj[1:] - traj[:-1]
    path_len = diffs.norm(dim=-1).sum(dim=0)
    chord_len = (traj[-1] - traj[0]).norm(dim=-1).clamp_min(1e-6)
    ratio = float((path_len / chord_len).mean().item())
    # Rectified Flow 论文定义的直度 S = E[∫ ||v_t - (x_1 - x_0)||² dt]
    T = traj.shape[0] - 1
    vels = diffs * T
    chord_vel = (traj[-1] - traj[0]).unsqueeze(0)
    curvature = float(((vels - chord_vel) ** 2).sum(dim=-1).mean().item())
    return ratio, curvature


def verify_section_2_hand_calcs():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 2.4–2.5 节手算与 EDM↔FM 换算")
    print("=" * 60)
    # 1D 高斯 N(2, 1)，精确速度 v*(x_u, u) 在 u=0.5, x_u=1.5 处
    # 边际方差 (1-u)^2 + u^2 = 0.5, 均值 2u = 1.0
    # 精确解析速度：v*(x_u, u) = (2 - x_u) / (1 - u) * ... 直接检查 x_0=0.8 的轨迹点
    x0 = 0.8
    x_05 = 2.0 * 0.5 + x0 * math.sqrt(0.5 ** 2 + 0.5 ** 2)
    x_1 = 2.0 + x0
    print(f"  1D 高斯 N(2,1) 精确轨迹 (x_0=0.8)：x_0.5 = {x_05:.3f}，x_1 = {x_1:.3f}（手算 1.566 / 2.800 ✓）")
    # 坐标互换验证
    u_test = torch.tensor([0.2, 0.5, 0.8])
    s_test = u_to_sigma(u_test)
    u_back = sigma_to_u(s_test)
    assert torch.allclose(u_test, u_back, atol=1e-6)
    print(f"  u↔σ 互换：u={u_test.tolist()} → σ={[round(v, 3) for v in s_test.tolist()]} → u={u_back.tolist()} ✓")


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()
    verify_section_2_hand_calcs()

    print("\n" + "=" * 60)
    print("第二部分：8-高斯环上对比 Flow Matching、EDM 与一次 Reflow")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    data_np, centers = sample_8gaussians(4000, rng=rng)
    sigma_data = float(data_np.std())
    data_t = torch.from_numpy(data_np)

    fm1 = ToyMLP(2, 128)
    edm = EDMDenoiser(ToyMLP(2, 128), sigma_data=sigma_data)
    opt_fm1 = torch.optim.Adam(fm1.parameters(), lr=2e-3)
    opt_edm = torch.optim.Adam(edm.parameters(), lr=2e-3)

    n_iters = 450 if quick else 2500
    for _ in range(n_iters):
        idx = torch.randint(0, len(data_t), (256,))
        x = data_t[idx]

        opt_fm1.zero_grad()
        cfm_loss(fm1, x).backward()
        opt_fm1.step()

        opt_edm.zero_grad()
        edm.f_loss(x, sample_train_sigmas(256, -1.0, 1.2)).backward()
        opt_edm.step()

    # Reflow：从 1-Flow 采样确定性配对，训练 2-Flow（直线化）
    n_pairs = 2000 if quick else 8000
    eps_p, x_p = reflow_pairs(fm1, n_pairs=n_pairs, n_steps=20)
    fm2 = ToyMLP(2, 128)
    fm2.load_state_dict(fm1.state_dict())
    opt_fm2 = torch.optim.Adam(fm2.parameters(), lr=1.5e-3)
    for _ in range(n_iters):
        idx = torch.randint(0, n_pairs, (256,))
        opt_fm2.zero_grad()
        cfm_loss(fm2, x_p[idx], eps=eps_p[idx]).backward()
        opt_fm2.step()

    init_eps = torch.randn(1000, 2)
    step_list = [1, 2, 4, 8]
    print(f"\n  {'步数 N':>6s} | {'EDM 距中心':>10s} {'命中率':>7s} | {'1-Flow 距中心':>13s} {'命中率':>7s} | {'Reflow 距中心':>13s} {'命中率':>7s}")
    samples_store = {}
    for n in step_list:
        s_edm = sample_euler(edm, (1000, 2), n, sigma_max=8.0, init_eps=init_eps).numpy()
        s_fm1 = sample_flow_euler(fm1, (1000, 2), n, init_eps=init_eps).numpy()
        s_fm2 = sample_flow_euler(fm2, (1000, 2), n, init_eps=init_eps).numpy()
        m_edm = eval_2d_samples(s_edm, centers)
        m_fm1 = eval_2d_samples(s_fm1, centers)
        m_fm2 = eval_2d_samples(s_fm2, centers)
        samples_store[n] = (s_edm, s_fm1, s_fm2)
        print(
            f"  {n:>6d} | {m_edm[0]:10.4f} {m_edm[1]*100:6.1f}% | "
            f"{m_fm1[0]:13.4f} {m_fm1[1]*100:6.1f}% | {m_fm2[0]:13.4f} {m_fm2[1]*100:6.1f}%"
        )

    _, traj1 = sample_flow_euler(fm1, (400, 2), 20, init_eps=init_eps[:400], return_traj=True)
    _, traj2 = sample_flow_euler(fm2, (400, 2), 20, init_eps=init_eps[:400], return_traj=True)
    r1, c1 = measure_straightness(traj1)
    r2, c2 = measure_straightness(traj2)
    print(f"\n  轨迹直度对比（20 步 ODE，完全直线弧弦比=1.000、曲率=0.000）：")
    print(f"    1-Flow（Reflow 前）：弧弦比 = {r1:.4f}，速度弯曲度 S = {c1:.4f}")
    print(f"    2-Flow（Reflow 后）：弧弦比 = {r2:.4f}，速度弯曲度 S = {c2:.4f}")

    # 绘图：1 步与 4 步对比 + Reflow 前后轨迹图
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for col, (title, pts) in enumerate([
        ("EDM（1 步）", samples_store[1][0]),
        ("1-Flow（1 步）", samples_store[1][1]),
        ("Reflow（1 步）", samples_store[1][2]),
    ]):
        ax = axes[0, col]
        ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=45, zorder=5)
        ax.scatter(pts[:600, 0], pts[:600, 1], s=5, alpha=0.5)
        ax.set_title(title); ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)

    for col, (title, traj_t) in enumerate([
        (f"1-Flow 轨迹（弯曲度 S={c1:.3f}）", traj1.numpy()),
        (f"Reflow 轨迹（弯曲度 S={c2:.3f}）", traj2.numpy()),
    ]):
        ax = axes[1, col]
        for b in range(40):
            ax.plot(traj_t[:, b, 0], traj_t[:, b, 1], alpha=0.45, lw=0.9)
        ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="x", s=45, zorder=5)
        ax.set_title(title); ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)

    ax = axes[1, 2]
    ax.plot(step_list, [eval_2d_samples(samples_store[n][0], centers)[0] for n in step_list], "o-", label="EDM")
    ax.plot(step_list, [eval_2d_samples(samples_store[n][1], centers)[0] for n in step_list], "s-", label="1-Flow")
    ax.plot(step_list, [eval_2d_samples(samples_store[n][2], centers)[0] for n in step_list], "^-", label="Reflow")
    ax.set_title("各步数下距中心平均误差"); ax.set_xlabel("步数 N"); ax.legend()
    fig.tight_layout()
    save_figure(fig, OUT / "02-flow-matching-and-reflow.png")

    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：03-action-conditioned-frame-diffusion.py")


if __name__ == "__main__":
    main()
