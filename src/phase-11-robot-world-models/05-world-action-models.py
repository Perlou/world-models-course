"""
Phase 11 · 第 05 课：World-Action Model（双噪声水平联合建模动作与未来观测）
===========================================================================

学习目标：
    1. 复现 CONCEPT.md 第 5.3 节 2×2 联合分布表玩具实验，用一个双独立扩散步网络切换四种查询
       （策略 / 视频预测 / 前向动力学 / 逆动力学），核对采样频率与手算（0.50 / 0.55 / 0.90 / 0.818）
    2. 在 PushT 上实现 WorldActionModel（动作块 A 与未来观测块 O' 各带独立路径时间 u_a, u_o）
       及支持混入「无动作观测序列」的 joint_denoising_loss()
    3. 对比三种训练配置在 PushT 上的策略表现：
       ① 仅动作扩散  ② 动作 + 未来观测联合训练  ③ 联合训练 + 额外无动作观测数据

对应概念文档：
    src/phase-11-robot-world-models/CONCEPT.md 第 5 节

运行方式：
    uv run python src/phase-11-robot-world-models/05-world-action-models.py
    WM_QUICK=1 uv run python src/phase-11-robot-world-models/05-world-action-models.py

预计运行时间：完整约 80 秒（M1 Max CPU），冒烟约 12 秒
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
from utils.pusht import load_pusht_demos

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec03 = importlib.util.spec_from_file_location("p11_03_wam", os.path.join(_DIR, "03-behavior-cloning-and-diffusion-policy.py"))
_mod03 = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(_mod03)

_SinEmb = _mod03._SinEmb
rollout_policy = _mod03.rollout_policy

plt = setup_matplotlib()
OUT = output_dir("phase-11")
SEED = 0


# ==================== 第一部分：2×2 联合分布表四种查询玩具实验 ====================


def joint_table_demo(quick=False):
    """CONCEPT.md 第 5.3 节：2×2 联合表 [[0.45, 0.05], [0.10, 0.40]]，用双独立路径时间 (u_a, u_o) 切换四种查询。"""
    print("=" * 60)
    print("第一部分：2×2 联合分布表 —— 一个模型通过 (u_a, u_o) 切换四种查询")
    print("=" * 60)
    # 动作：左推=-1, 右推=+1；未来观测：块左移=-1, 块右移=+1
    rng = np.random.default_rng(SEED)
    probs = np.array([0.45, 0.05, 0.10, 0.40])
    pairs = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
    idx_data = rng.choice(4, size=4000, p=probs)
    A_data = torch.from_numpy(pairs[idx_data, 0:1])
    O_data = torch.from_numpy(pairs[idx_data, 1:2])

    emb_a, emb_o = _SinEmb(24), _SinEmb(24)
    net = nn.Sequential(
        nn.Linear(2 + 48, 96), nn.SiLU(),
        nn.Linear(96, 96), nn.SiLU(),
        nn.Linear(96, 2),
    )
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    n_iters = 400 if quick else 1600
    for _ in range(n_iters):
        idx = torch.randint(0, len(A_data), (256,))
        a, o = A_data[idx], O_data[idx]
        ea, eo = torch.randn_like(a), torch.randn_like(o)
        # 独立采样 u_a, u_o ∈ [0, 1]，并以一定概率精确命中边界 0（纯噪声边缘化）与 1（干净条件）
        ua = torch.rand(len(a))
        uo = torch.rand(len(a))
        r = torch.rand(len(a))
        ua = torch.where(r < 0.2, torch.zeros_like(ua), torch.where(r < 0.4, torch.ones_like(ua), ua))
        r2 = torch.rand(len(a))
        uo = torch.where(r2 < 0.2, torch.zeros_like(uo), torch.where(r2 < 0.4, torch.ones_like(uo), uo))

        a_u = (1.0 - ua.unsqueeze(-1)) * ea + ua.unsqueeze(-1) * a
        o_u = (1.0 - uo.unsqueeze(-1)) * eo + uo.unsqueeze(-1) * o
        out = net(torch.cat([a_u, o_u, emb_a(ua), emb_o(uo)], dim=-1))
        loss = F.mse_loss(out[:, 0:1], a - ea) + F.mse_loss(out[:, 1:2], o - eo)
        opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def query(mode, cond_val=None, n=2000, steps=20):
        a = torch.randn(n, 1) if mode != "forward" else torch.full((n, 1), float(cond_val))
        o = torch.randn(n, 1) if mode != "inverse" else torch.full((n, 1), float(cond_val))
        dt = 1.0 / steps
        for i in range(steps):
            u = i * dt
            ua = torch.full((n,), 1.0 if mode == "forward" else (0.0 if mode == "video" else u))
            uo = torch.full((n,), 1.0 if mode == "inverse" else (0.0 if mode == "policy" else u))
            v = net(torch.cat([a, o, emb_a(ua), emb_o(uo)], dim=-1))
            if mode in {"policy", "inverse"}:
                a = a + dt * v[:, 0:1]
            if mode in {"video", "forward"}:
                o = o + dt * v[:, 1:2]
        return float((a < 0).float().mean().item()) if mode in {"policy", "inverse"} else float((o < 0).float().mean().item())

    p_pol = query("policy")
    p_vid = query("video")
    p_fwd = query("forward", cond_val=-1.0)
    p_inv = query("inverse", cond_val=-1.0)
    print(f"  策略查询   P(左推)          = {p_pol:.3f}（手算 0.500）")
    print(f"  视频查询   P(左移)          = {p_vid:.3f}（手算 0.550）")
    print(f"  前向动力学 P(左移 | 左推)    = {p_fwd:.3f}（手算 0.900）")
    print(f"  逆动力学   P(左推 | 左移)    = {p_inv:.3f}（手算 0.818）")
    return {"policy": p_pol, "video": p_vid, "forward": p_fwd, "inverse": p_inv}


# ==================== 第二部分：PushT World-Action Model 对比 ====================


class WorldActionModel(nn.Module):
    """UWM 式双噪声水平 World-Action Model：同时对动作块 [K, 2] 与未来状态块 [K, 5] 做流匹配去噪。"""

    def __init__(self, obs_dim=5, act_dim=2, T_o=2, K=8, hidden=192):
        super().__init__()
        self.T_o = T_o
        self.T_p = K
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ua_emb = _SinEmb(32)
        self.uo_emb = _SinEmb(32)
        self.ctx_enc = nn.Sequential(nn.Linear(T_o * obs_dim, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU())
        in_dim = K * act_dim + K * obs_dim + 96 + 64
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.act_head = nn.Linear(hidden, K * act_dim)
        self.obs_head = nn.Linear(hidden, K * obs_dim)

    def forward(self, a_u, o_u, u_a, u_o, obs_win):
        ctx = self.ctx_enc(obs_win.flatten(1))
        h = self.trunk(torch.cat([a_u.flatten(1), o_u.flatten(1), ctx, self.ua_emb(u_a), self.uo_emb(u_o)], dim=-1))
        return self.act_head(h).reshape(-1, self.T_p, self.act_dim), self.obs_head(h).reshape(-1, self.T_p, self.obs_dim)

    @torch.no_grad()
    def predict_chunk(self, obs_win, n_steps=10):
        """策略查询：固定未来观测为纯噪声 (u_o = 0)，只对动作块去噪（推理无需生成观测）。"""
        self.eval()
        B = obs_win.shape[0]
        a = torch.randn(B, self.T_p, self.act_dim, device=obs_win.device)
        o_noise = torch.randn(B, self.T_p, self.obs_dim, device=obs_win.device)
        u_o = torch.zeros(B, device=obs_win.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            u_a = torch.full((B,), i * dt, device=obs_win.device)
            va, _ = self(a, o_noise, u_a, u_o, obs_win)
            a = a + dt * va
        return a


def build_wam_dataset(demos, ep_indices, T_o=2, K=8):
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    actions = demos["actions"]
    extract_demo_states_5d = _mod03.extract_demo_states_5d

    def norm_state(st):
        out = np.asarray(st, dtype=np.float32).copy()
        out[..., :4] = out[..., :4] / 256.0 - 1.0
        if out.shape[-1] >= 5:
            out[..., 4] = out[..., 4] / (math.pi * 0.5)
        return out

    obs_l, act_l, fut_l = [], [], []
    for ep in ep_indices:
        s, e = int(starts[ep]), int(ends[ep])
        L = e - s
        st_ep = norm_state(extract_demo_states_5d(demos, s, e))
        ac_ep = (actions[s:e].astype(np.float32) / 256.0) - 1.0
        for t in range(L):
            o_idx = [max(0, t - (T_o - 1 - k)) for k in range(T_o)]
            a_idx = [min(L - 1, t + k) for k in range(K)]
            f_idx = [min(L - 1, t + 1 + k) for k in range(K)]
            obs_l.append(st_ep[o_idx]); act_l.append(ac_ep[a_idx]); fut_l.append(st_ep[f_idx])
    return torch.from_numpy(np.asarray(obs_l)), torch.from_numpy(np.asarray(act_l)), torch.from_numpy(np.asarray(fut_l)), norm_state


def train_wam_variant(obs_l, act_l, fut_l, unlab_obs=None, unlab_fut=None, lam_o=1.0, n_steps=1000, seed=0):
    seed_everything(seed)
    model = WorldActionModel(obs_dim=5, act_dim=2, T_o=2, K=8, hidden=192)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    N = len(obs_l)
    for _ in range(n_steps):
        idx = torch.randint(0, N, (96,))
        ow, ac, fu = obs_l[idx], act_l[idx], fut_l[idx]
        ea, eo = torch.randn_like(ac), torch.randn_like(fu)
        ua = torch.rand(len(ow))
        uo = torch.zeros(len(ow)) if lam_o == 0.0 else torch.rand(len(ow))
        # 25% 概率让未来观测为纯噪声 u_o=0，专门对齐策略推理分布
        if lam_o > 0.0:
            uo = torch.where(torch.rand(len(ow)) < 0.25, torch.zeros_like(uo), uo)
        a_u = (1.0 - ua.view(-1, 1, 1)) * ea + ua.view(-1, 1, 1) * ac
        o_u = (1.0 - uo.view(-1, 1, 1)) * eo + uo.view(-1, 1, 1) * fu
        va, vo = model(a_u, o_u, ua, uo, ow)
        loss = F.mse_loss(va, ac - ea) + lam_o * F.mse_loss(vo, fu - eo)

        # 混入无动作观测序列（令 u_a = 0、动作填纯噪声，只算观测预测损失）
        if unlab_obs is not None and lam_o > 0.0:
            uidx = torch.randint(0, len(unlab_obs), (48,))
            uow, ufu = unlab_obs[uidx], unlab_fut[uidx]
            ua0 = torch.zeros(len(uow))
            uo2 = torch.rand(len(uow))
            eo2 = torch.randn_like(ufu)
            a_rand = torch.randn(len(uow), model.T_p, 2)
            o_u2 = (1.0 - uo2.view(-1, 1, 1)) * eo2 + uo2.view(-1, 1, 1) * ufu
            _, vo2 = model(a_rand, o_u2, ua0, uo2, uow)
            loss = loss + 0.5 * lam_o * F.mse_loss(vo2, ufu - eo2)

        opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    q_res = joint_table_demo(quick=quick)

    print("\n" + "=" * 60)
    print("第二部分：PushT 少量带动作演示上对比动作扩散 vs 联合训练 vs 混入无动作数据")
    print("=" * 60)
    demos = load_pusht_demos()
    # 故意用较少带动作演示（模拟真实机器人动作数据稀缺），另取大量无动作轨迹模拟视频预训练
    n_labeled = 12 if quick else 35
    n_unlabeled = 20 if quick else 120
    obs_l, act_l, fut_l, norm_fn = build_wam_dataset(demos, list(range(n_labeled)))
    obs_u, _, fut_u, _ = build_wam_dataset(demos, list(range(n_labeled, min(len(demos["episode_starts"]), n_labeled + n_unlabeled))))

    steps = 180 if quick else 1100
    m_act_only = train_wam_variant(obs_l, act_l, fut_l, lam_o=0.0, n_steps=steps, seed=SEED)
    m_joint = train_wam_variant(obs_l, act_l, fut_l, lam_o=0.5, n_steps=steps, seed=SEED)
    m_joint_unlab = train_wam_variant(obs_l, act_l, fut_l, unlab_obs=obs_u, unlab_fut=fut_u, lam_o=0.5, n_steps=steps, seed=SEED)

    n_eval = 6 if quick else 20
    max_steps = 80 if quick else 160
    summary = {}
    for name, mod in [
        ("仅动作扩散 (λ_o=0)", m_act_only),
        ("动作+未来观测联合", m_joint),
        ("联合 + 无动作轨迹", m_joint_unlab),
    ]:
        k, n, cov, _ = rollout_policy(mod, norm_fn, T_a=4, n_episodes=n_eval, max_steps=max_steps)
        summary[name] = (k / n, cov)
        print(f"  {name:20s}：成功率 = {k/n*100:5.1f}% ({k}/{n}) | 平均最高覆盖度 = {cov:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    q_names = ["策略\n(手算0.50)", "视频\n(手算0.55)", "前向\n(手算0.90)", "逆动力\n(手算0.82)"]
    axes[0].bar(q_names, [q_res["policy"], q_res["video"], q_res["forward"], q_res["inverse"]], alpha=0.8, label="实测")
    axes[0].plot(range(4), [0.50, 0.55, 0.90, 0.818], "ro--", label="手算真值")
    axes[0].set_title("2×2 联合表四种查询实测 vs 手算"); axes[0].set_ylim(0, 1.05); axes[0].legend()

    axes[1].bar(list(summary.keys()), [v[1] for v in summary.values()], color=["tab:gray", "tab:blue", "tab:green"])
    axes[1].set_title(f"PushT 少样本 ({n_labeled} 演示) 平均最高覆盖度"); axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=12)
    fig.tight_layout()
    save_figure(fig, OUT / "05-world-action-models.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：06-world-model-as-simulator.py")


if __name__ == "__main__":
    main()
