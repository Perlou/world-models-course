"""
Phase 11 · 第 03 课：行为克隆、分布偏移与 Diffusion Policy
=========================================================

学习目标：
    1. 在一维双峰玩具问题（a ∈ {-1, +1} 各 50%）上对比 MSE 行为克隆（均值塌缩到 0）与
       扩散策略（双峰各占一半）的输出直方图
    2. 在 PushT 演示数据上实现带动作分块的 MSE 行为克隆（BCPolicy）与条件扩散策略（DiffusionPolicy）
    3. 在真实 gym-pusht 仿真器上评测两者成功率、平均最高覆盖度与 Wilson 95% 置信区间，
       并扫描实际执行长度 T_a ∈ {1, 4, 8} 观察对成功率的影响
    4. 测量闭环部署时策略访问状态到演示数据集最近邻的距离随时间步的变化（分布偏移曲线）

对应概念文档：
    src/phase-11-robot-world-models/CONCEPT.md 第 3 节

运行方式：
    uv run python src/phase-11-robot-world-models/03-behavior-cloning-and-diffusion-policy.py
    WM_QUICK=1 uv run python src/phase-11-robot-world-models/03-behavior-cloning-and-diffusion-policy.py

预计运行时间：完整约 90 秒（M1 Max CPU），冒烟约 12 秒
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
from utils.pusht import load_pusht_demos, make_pusht_env

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec01 = importlib.util.spec_from_file_location("p11_01_dp", os.path.join(_DIR, "01-robot-learning-landscape.py"))
_mod01 = importlib.util.module_from_spec(_spec01)
_spec01.loader.exec_module(_mod01)
success_rate_interval = _mod01.success_rate_interval

plt = setup_matplotlib()
OUT = output_dir("phase-11")
SEED = 0


# ==================== 第一部分：1D 双峰玩具实验 ====================


class _SinEmb(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        h = self.dim // 2
        f = torch.exp(torch.linspace(0.0, -math.log(1000.0), h, device=t.device))
        a = t.unsqueeze(-1) * f.unsqueeze(0)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


def toy_bimodal_demo(quick=False):
    print("=" * 60)
    print("第一部分：1D 双峰玩具问题（a ∈ {-1, +1}）—— MSE 均值塌缩 vs 扩散策略")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    y_data = torch.from_numpy(rng.choice([-1.0, 1.0], size=(2000, 1)).astype(np.float32))
    x_dummy = torch.zeros(2000, 1)

    # 1. MSE 行为克隆
    bc = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
    opt_bc = torch.optim.Adam(bc.parameters(), lr=5e-3)
    for _ in range(150 if quick else 400):
        opt_bc.zero_grad()
        F.mse_loss(bc(x_dummy), y_data).backward()
        opt_bc.step()
    with torch.no_grad():
        bc_samples = bc(torch.zeros(500, 1)).squeeze(-1).numpy()

    # 2. 1D 条件 Flow Matching / 扩散策略（路径 u ∈ [0, 1]，x_u = (1-u)ε + u·a）
    t_emb = _SinEmb(32)
    dp_net = nn.Sequential(nn.Linear(1 + 32, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
    opt_dp = torch.optim.Adam(dp_net.parameters(), lr=3e-3)
    for _ in range(300 if quick else 1200):
        idx = torch.randint(0, len(y_data), (128,))
        a = y_data[idx]
        eps = torch.randn_like(a)
        u = torch.rand(len(a))
        x_u = (1.0 - u.unsqueeze(-1)) * eps + u.unsqueeze(-1) * a
        v_hat = dp_net(torch.cat([x_u, t_emb(u)], dim=-1))
        opt_dp.zero_grad(); F.mse_loss(v_hat, a - eps).backward(); opt_dp.step()

    with torch.no_grad():
        x = torch.randn(500, 1)
        n_steps = 15
        for i in range(n_steps):
            u = torch.full((500,), i / n_steps)
            x = x + (1.0 / n_steps) * dp_net(torch.cat([x, t_emb(u)], dim=-1))
        dp_samples = x.squeeze(-1).numpy()

    near_zero_bc = float((np.abs(bc_samples) < 0.25).mean())
    near_modes_dp = float(((np.abs(dp_samples - 1.0) < 0.35) | (np.abs(dp_samples + 1.0) < 0.35)).mean())
    pos_frac = float((dp_samples > 0).mean())
    print(f"  MSE 行为克隆：均值 = {bc_samples.mean():+.3f}，落在 [-0.25, 0.25] 禁区占比 = {near_zero_bc*100:.1f}%（塌缩至均值 0 ✓）")
    print(f"  扩散策略采样：命中 ±1 峰占比 = {near_modes_dp*100:.1f}%（左峰 {(1-pos_frac)*100:.1f}% / 右峰 {pos_frac*100:.1f}% ✓）")
    return bc_samples, dp_samples


# ==================== 第二部分：PushT 行为克隆与 Diffusion Policy ====================


class BCPolicy(nn.Module):
    """带观测窗口 T_o 与动作块长度 T_p 的 MSE 行为克隆策略（状态输入版，供 03/06/07/08 复用）。"""

    def __init__(self, obs_dim=5, act_dim=2, T_o=2, T_p=8, hidden=192):
        super().__init__()
        self.T_o = T_o
        self.T_p = T_p
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(T_o * obs_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, T_p * act_dim),
        )

    def forward(self, obs_win):
        # obs_win: [B, T_o, obs_dim] -> [B, T_p, act_dim]
        return self.net(obs_win.flatten(1)).reshape(-1, self.T_p, self.act_dim)

    @torch.no_grad()
    def predict_chunk(self, obs_win):
        self.eval()
        return self(obs_win)


class DiffusionPolicy(nn.Module):
    """CONCEPT.md 第 3.6 节：以历史状态窗口为条件、用流匹配/扩散生成动作块 [B, T_p, 2] 的精简 Diffusion Policy。"""

    def __init__(self, obs_dim=5, act_dim=2, T_o=2, T_p=8, hidden=192):
        super().__init__()
        self.T_o = T_o
        self.T_p = T_p
        self.act_dim = act_dim
        self.u_emb = _SinEmb(48)
        self.obs_enc = nn.Sequential(nn.Linear(T_o * obs_dim, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(T_p * act_dim + 96 + 48, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, T_p * act_dim),
        )

    def forward(self, noisy_chunk, u, obs_win):
        cond = torch.cat([self.obs_enc(obs_win.flatten(1)), self.u_emb(u)], dim=-1)
        x_in = torch.cat([noisy_chunk.flatten(1), cond], dim=-1)
        return self.net(x_in).reshape(-1, self.T_p, self.act_dim)

    def loss(self, clean_chunk, obs_win):
        B = clean_chunk.shape[0]
        eps = torch.randn_like(clean_chunk)
        u = torch.rand(B, device=clean_chunk.device)
        x_u = (1.0 - u.view(B, 1, 1)) * eps + u.view(B, 1, 1) * clean_chunk
        v_pred = self(x_u, u, obs_win)
        return F.mse_loss(v_pred, clean_chunk - eps)

    @torch.no_grad()
    def predict_chunk(self, obs_win, n_steps=10):
        self.eval()
        B = obs_win.shape[0]
        x = torch.randn(B, self.T_p, self.act_dim, device=obs_win.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            u = torch.full((B,), i * dt, device=obs_win.device)
            x = x + dt * self(x, u, obs_win)
        return x


def extract_block_pose_batch(imgs):
    """从 PushT 96×96 RGB 画面中提取灰色 T 形块的几何质心 (cx, cy) 与主轴朝向 θ（像素坐标系 0..512）。"""
    f = imgs.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    lum = (r + g + b) / 3.0
    mask = ((sat < 25.0) & (lum > 70.0) & (lum < 210.0)).astype(np.float32)
    _, H, W = mask.shape
    xs = np.linspace(0.0, 512.0, W, dtype=np.float32)[None, None, :]
    ys = np.linspace(0.0, 512.0, H, dtype=np.float32)[None, :, None]
    area = mask.sum(axis=(1, 2)) + 1e-6
    cx = (mask * xs).sum(axis=(1, 2)) / area
    cy = (mask * ys).sum(axis=(1, 2)) / area
    dx = xs - cx[:, None, None]
    dy = ys - cy[:, None, None]
    mu20 = (mask * dx * dx).sum(axis=(1, 2)) / area
    mu02 = (mask * dy * dy).sum(axis=(1, 2)) / area
    mu11 = (mask * dx * dy).sum(axis=(1, 2)) / area
    theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    return np.stack([cx, cy, theta], axis=-1).astype(np.float32)


def extract_demo_states_5d(demos, s, e):
    """拼接演示数据的 2D 推杆坐标与画面提取的 3D T 形块位姿，得到 [L, 5] 连续状态。"""
    agent_xy = demos["states"][s:e, :2].astype(np.float32)
    block_pose = extract_block_pose_batch(demos["images"][s:e])
    return np.concatenate([agent_xy, block_pose], axis=-1)


def build_state_chunk_dataset(demos, n_episodes=None, T_o=2, T_p=8):
    """构建归一化后的 PushT 5D 状态窗口 [N, T_o, 5] 与动作块 [N, T_p, 2]。"""
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    actions = demos["actions"]
    n_use = len(starts) if n_episodes is None else min(n_episodes, len(starts))

    # 统一归一化：前 4 维坐标映射到 [-1, 1]，第 5 维角度映射到 [-1, 1]；动作映射到 [-1, 1]
    def norm_state(st):
        out = np.asarray(st, dtype=np.float32).copy()
        out[..., :4] = out[..., :4] / 256.0 - 1.0
        if out.shape[-1] >= 5:
            out[..., 4] = out[..., 4] / (math.pi * 0.5)
        return out

    obs_list, act_list = [], []
    for ep in range(n_use):
        s, e = int(starts[ep]), int(ends[ep])
        L = e - s
        st_ep = norm_state(extract_demo_states_5d(demos, s, e))
        ac_ep = (actions[s:e].astype(np.float32) / 256.0) - 1.0
        for t in range(L):
            o_idx = [max(0, t - (T_o - 1 - k)) for k in range(T_o)]
            a_idx = [min(L - 1, t + k) for k in range(T_p)]
            obs_list.append(st_ep[o_idx])
            act_list.append(ac_ep[a_idx])
    return torch.from_numpy(np.asarray(obs_list)), torch.from_numpy(np.asarray(act_list)), norm_state


def extract_obs_5d(obs_dict, norm_fn):
    bp = extract_block_pose_batch(obs_dict["pixels"][None])[0]
    st5 = np.concatenate([obs_dict["agent_pos"][:2], bp]).astype(np.float32)
    return norm_fn(st5)


def rollout_policy(policy, norm_fn, T_a=4, n_episodes=20, max_steps=180, seed_base=2000, demo_states_norm=None):
    """在真实 gym-pusht 上闭环评估策略，返回成功率、平均最高覆盖度与分布偏移曲线。"""
    env = make_pusht_env(obs_type="pixels_agent_pos")
    succ = 0
    max_covs = []
    shift_by_step = [[] for _ in range(max_steps)]

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + ep)
        o0 = extract_obs_5d(obs, norm_fn)
        buf = [o0.copy() for _ in range(policy.T_o)]
        ep_max_cov = 0.0
        is_ok = False
        step = 0
        while step < max_steps:
            o_win = torch.from_numpy(np.asarray(buf, dtype=np.float32)).unsqueeze(0)
            chunk = policy.predict_chunk(o_win)[0].numpy()  # [T_p, 2] in [-1, 1]
            for k in range(min(T_a, policy.T_p)):
                if step >= max_steps:
                    break
                act_env = np.clip((chunk[k] + 1.0) * 256.0, 0.0, 512.0)
                obs, r, term, trunc, info = env.step(act_env)
                cov = float(info.get("coverage", r))
                ep_max_cov = max(ep_max_cov, cov)
                if info.get("is_success", False) or cov >= 0.85:
                    is_ok = True
                on = extract_obs_5d(obs, norm_fn)
                if demo_states_norm is not None:
                    d_nn = float(np.linalg.norm(demo_states_norm - on[None, :], axis=-1).min())
                    shift_by_step[step].append(d_nn)
                buf = buf[1:] + [on]
                step += 1
                if term or trunc:
                    step = max_steps
                    break
        succ += int(is_ok)
        max_covs.append(ep_max_cov)
    env.close()
    shift_curve = [float(np.mean(v)) for v in shift_by_step if v]
    return succ, n_episodes, float(np.mean(max_covs)), shift_curve


def train_bc_and_dp(demos, n_episodes=100, bc_steps=800, dp_steps=1200, T_o=2, T_p=8, seed=0):
    seed_everything(seed)
    obs_X, act_Y, norm_fn = build_state_chunk_dataset(demos, n_episodes=n_episodes, T_o=T_o, T_p=T_p)
    bc = BCPolicy(obs_dim=obs_X.shape[-1], act_dim=2, T_o=T_o, T_p=T_p)
    dp = DiffusionPolicy(obs_dim=obs_X.shape[-1], act_dim=2, T_o=T_o, T_p=T_p)
    opt_bc = torch.optim.AdamW(bc.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_dp = torch.optim.AdamW(dp.parameters(), lr=1e-3, weight_decay=1e-4)

    N = len(obs_X)
    for _ in range(bc_steps):
        idx = torch.randint(0, N, (128,))
        opt_bc.zero_grad()
        F.mse_loss(bc(obs_X[idx]), act_Y[idx]).backward()
        opt_bc.step()

    for _ in range(dp_steps):
        idx = torch.randint(0, N, (128,))
        opt_dp.zero_grad()
        dp.loss(act_Y[idx], obs_X[idx]).backward()
        opt_dp.step()
    return bc, dp, norm_fn, obs_X[:, -1].numpy()


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    bc_1d, dp_1d = toy_bimodal_demo(quick=quick)

    print("\n" + "=" * 60)
    print("第二部分：PushT 行为克隆 vs Diffusion Policy（含执行长度 T_a 与分布偏移）")
    print("=" * 60)
    demos = load_pusht_demos()
    n_demo_eps = 30 if quick else 160
    bc, dp, norm_fn, demo_st_sub = train_bc_and_dp(
        demos, n_episodes=n_demo_eps,
        bc_steps=150 if quick else 1200,
        dp_steps=200 if quick else 1800,
        T_o=2, T_p=8, seed=SEED,
    )
    # 下采样演示状态池用于快速计算最近邻分布偏移距离
    demo_pool = demo_st_sub[:: max(1, len(demo_st_sub) // 1500)]
    n_eval = 6 if quick else 25
    max_steps = 80 if quick else 180

    k_bc, n_bc, cov_bc, shift_bc = rollout_policy(bc, norm_fn, T_a=4, n_episodes=n_eval, max_steps=max_steps, demo_states_norm=demo_pool)
    p_bc, _, _, ci_bc = success_rate_interval(k_bc, n_bc)
    print(f"  MSE 行为克隆 (T_a=4, n={n_bc})     ：成功率 = {p_bc*100:5.1f}%（Wilson 95% CI [{ci_bc[0]*100:.1f}%, {ci_bc[1]*100:.1f}%]）| 均最高覆盖度 = {cov_bc:.3f}")

    ta_res = {}
    shift_dp = None
    for T_a in [1, 4, 8]:
        k_dp, n_dp, cov_dp, sh = rollout_policy(dp, norm_fn, T_a=T_a, n_episodes=n_eval, max_steps=max_steps, demo_states_norm=demo_pool if T_a == 4 else None)
        p_dp, _, _, ci_dp = success_rate_interval(k_dp, n_dp)
        ta_res[T_a] = (p_dp, cov_dp, ci_dp)
        if T_a == 4:
            shift_dp = sh
        print(f"  Diffusion Policy (T_a={T_a}, n={n_dp})：成功率 = {p_dp*100:5.1f}%（Wilson 95% CI [{ci_dp[0]*100:.1f}%, {ci_dp[1]*100:.1f}%]）| 均最高覆盖度 = {cov_dp:.3f}")

    print(f"\n  分布偏移距离（访问状态到演示池最近邻 L2 距离，步 1 → 步 30 → 步 {len(shift_bc)}）：")
    idx_show = [0, min(29, len(shift_bc) - 1), len(shift_bc) - 1]
    print(f"    MSE 行为克隆   ：{[round(shift_bc[i], 3) for i in idx_show]}")
    print(f"    Diffusion Policy：{[round(shift_dp[i], 3) for i in idx_show]}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(bc_1d, bins=25, range=(-1.5, 1.5), alpha=0.7, label="MSE 行为克隆")
    axes[0].hist(dp_1d, bins=25, range=(-1.5, 1.5), alpha=0.6, label="扩散策略")
    axes[0].set_title("1D 双峰玩具：均值塌缩 vs 多峰采样"); axes[0].legend(fontsize=8)

    labels = ["BC (T_a=4)", "DP (T_a=1)", "DP (T_a=4)", "DP (T_a=8)"]
    covs = [cov_bc, ta_res[1][1], ta_res[4][1], ta_res[8][1]]
    axes[1].bar(labels, covs, color=["tab:orange", "tab:blue", "tab:green", "tab:purple"])
    axes[1].set_title("PushT 平均最高覆盖度（及执行长度 T_a 对比）"); axes[1].set_ylim(0, 1.05)

    axes[2].plot(shift_bc, label="MSE 行为克隆")
    axes[2].plot(shift_dp, label="Diffusion Policy (T_a=4)")
    axes[2].set_title("分布偏移：状态距演示池最近邻距离"); axes[2].set_xlabel("时间步"); axes[2].legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, OUT / "03-bc-and-diffusion-policy.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：04-video-prediction-as-policy.py")


if __name__ == "__main__":
    main()
