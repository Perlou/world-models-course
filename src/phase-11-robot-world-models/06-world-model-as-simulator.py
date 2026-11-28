"""
Phase 11 · 第 06 课：把世界模型当仿真器 —— 策略评估与相关性指标
==============================================================

学习目标：
    1. 实现三个一致性指标 pearson_r()、spearman_rho()、mmrv()，逐项核对
       CONCEPT.md 第 6.5 节「五个策略 × 四个世界模型（甲乙丙丁）」的全部手算数字
    2. 在 PushT 上训练学到的动力学世界模型与成功判定器，并在真实仿真器与世界模型里
       闭环评估 6 个不同水平的策略（随机 / 欠拟合 BC / 完整 BC / 不同步数 Diffusion Policy）
    3. 报告真实表现 y_i 与世界模型表现 ŷ_i 的散点对照，计算 Pearson、Spearman、MMRV，
       以及回合长度减半时三个指标的变化
    4. 评估成功判定器在真实轨迹与生成轨迹上的准确率

对应概念文档：
    src/phase-11-robot-world-models/CONCEPT.md 第 6 节

运行方式：
    uv run python src/phase-11-robot-world-models/06-world-model-as-simulator.py
    WM_QUICK=1 uv run python src/phase-11-robot-world-models/06-world-model-as-simulator.py

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
_spec03 = importlib.util.spec_from_file_location("p11_03_sim", os.path.join(_DIR, "03-behavior-cloning-and-diffusion-policy.py"))
_mod03 = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(_mod03)

BCPolicy = _mod03.BCPolicy
DiffusionPolicy = _mod03.DiffusionPolicy
build_state_chunk_dataset = _mod03.build_state_chunk_dataset
extract_obs_5d = _mod03.extract_obs_5d

plt = setup_matplotlib()
OUT = output_dir("phase-11")
SEED = 0


# ==================== 第一部分：三个一致性指标（Pearson / Spearman / MMRV） ====================


def pearson_r(y, y_hat):
    y = np.asarray(y, dtype=np.float64)
    yh = np.asarray(y_hat, dtype=np.float64)
    ey, eyh = y - y.mean(), yh - yh.mean()
    denom = math.sqrt(float((ey ** 2).sum() * (eyh ** 2).sum()))
    return 0.0 if denom < 1e-12 else float((ey * eyh).sum() / denom)


def _avg_ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty_like(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg_r = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_r
        i = j + 1
    return ranks


def spearman_rho(y, y_hat):
    return pearson_r(_avg_ranks(y), _avg_ranks(y_hat))


def mmrv(y, y_hat):
    """CONCEPT.md 第 6.4 节：SIMPLER 平均最大排序违例（越小越好，0 表示零违例）。"""
    y = np.asarray(y, dtype=np.float64)
    yh = np.asarray(y_hat, dtype=np.float64)
    N = len(y)
    max_viols = []
    for i in range(N):
        v_i = 0.0
        for j in range(N):
            if i == j:
                continue
            if (y[i] != y[j]) and ((yh[i] < yh[j]) != (y[i] < y[j])):
                v_i = max(v_i, abs(y[i] - y[j]))
        max_viols.append(v_i)
    return float(np.mean(max_viols))


def verify_section_6_5_hand_calcs():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 6.5 节五策略 × 四世界模型手算")
    print("=" * 60)
    y = [0.20, 0.45, 0.50, 0.70, 0.90]
    models = {
        "甲": ([0.30, 0.55, 0.50, 0.60, 0.80], (0.970, 0.90, 0.02)),
        "乙": ([0.10, 0.225, 0.25, 0.35, 0.45], (1.000, 1.00, 0.00)),
        "丙": ([0.20, 0.45, 0.50, 0.85, 0.75], (0.918, 0.90, 0.08)),
        "丁": ([0.20, 0.45, 0.50, 0.70, 0.15], (0.073, 0.00, 0.49)),
    }
    print(f"  {'模型':4s} | {'MAE':>6s} | {'Pearson':>8s} | {'Spearman':>8s} | {'MMRV':>6s} | {'手算对照'}")
    for nm, (yh, exp) in models.items():
        mae = float(np.mean(np.abs(np.asarray(y) - np.asarray(yh))))
        pr, sr, mv = pearson_r(y, yh), spearman_rho(y, yh), mmrv(y, yh)
        print(f"  {nm:4s} | {mae:6.3f} | {pr:8.3f} | {sr:8.2f} | {mv:6.2f} | ({exp[0]:.3f}, {exp[1]:.2f}, {exp[2]:.2f}) ✓")
        assert abs(pr - exp[0]) < 0.005 and abs(sr - exp[1]) < 0.01 and abs(mv - exp[2]) < 0.005


# ==================== 第二部分：PushT 神经仿真器与策略评估 ====================


class PushTStateWorldModel(nn.Module):
    """学习 PushT 5D 状态转移 (s_t, a_t) → (Δs_t, coverage_hat) 的神经仿真器与覆盖度/成功判定器。"""

    def __init__(self, state_dim=5, act_dim=2, hidden=192):
        super().__init__()
        self.dyn = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )
        self.cov_head = nn.Sequential(
            nn.Linear(state_dim, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def step(self, s_norm, a_norm):
        ns = (s_norm + self.dyn(torch.cat([s_norm, a_norm], dim=-1))).clamp(-1.2, 1.2)
        cov = self.cov_head(ns).squeeze(-1)
        return ns, cov


class RandomPolicy:
    T_o = 2
    T_p = 8

    def predict_chunk(self, obs_win):
        return torch.rand(obs_win.shape[0], self.T_p, 2) * 2.0 - 1.0


def collect_mixed_wm_data(demos, norm_fn, n_demo_eps=80, n_rand_eps=20):
    """按 CONCEPT.md 第 6.6 节：混合专家演示 + 随机/噪声失败回合，缓解世界模型只见过成功的乐观偏差。"""
    extract_demo_states_5d = _mod03.extract_demo_states_5d
    s_l, a_l, ns_l, cov_l = [], [], [], []
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    # 1. 直接提取专家演示的 5D 状态转移与真实覆盖度奖励
    for ep in range(min(n_demo_eps, len(starts))):
        s, e = int(starts[ep]), int(ends[ep])
        st_ep = norm_fn(extract_demo_states_5d(demos, s, e))
        ac_ep = (demos["actions"][s:e].astype(np.float32) / 256.0) - 1.0
        rw_ep = demos["rewards"][s:e].astype(np.float32)
        for t in range(min(len(st_ep) - 1, 120)):
            s_l.append(st_ep[t]); a_l.append(ac_ep[t]); ns_l.append(st_ep[t + 1]); cov_l.append(float(rw_ep[t]))
    # 2. 加入随机/扰动失败轨迹扩大覆盖面
    env = make_pusht_env(obs_type="pixels_agent_pos")
    rng = np.random.default_rng(SEED)
    for ep in range(n_rand_eps):
        obs, _ = env.reset(seed=5000 + ep)
        st_cur = extract_obs_5d(obs, norm_fn)
        cur_a = obs["agent_pos"].copy()
        for _ in range(60):
            cur_a = np.clip(cur_a + rng.normal(0.0, 45.0, size=2), 20.0, 492.0)
            obs, r, term, trunc, info = env.step(cur_a)
            st_nxt = extract_obs_5d(obs, norm_fn)
            s_l.append(st_cur); a_l.append((cur_a / 256.0) - 1.0); ns_l.append(st_nxt); cov_l.append(float(info.get("coverage", r)))
            st_cur = st_nxt
            if term or trunc:
                break
    env.close()
    return (
        torch.from_numpy(np.asarray(s_l, dtype=np.float32)),
        torch.from_numpy(np.asarray(a_l, dtype=np.float32)),
        torch.from_numpy(np.asarray(ns_l, dtype=np.float32)),
        torch.from_numpy(np.asarray(cov_l, dtype=np.float32)),
    )


def evaluate_policy_pair(policy, wm, norm_fn, init_states, max_steps=120, T_a=4, thresh=0.50):
    """在完全相同的初始状态集合上，分别于真实 gym-pusht 和世界模型内闭环评估策略。"""
    env = make_pusht_env(obs_type="pixels_agent_pos")
    wm.eval()
    real_scores, wm_scores = [], []
    real_succ, wm_succ = 0, 0
    cls_correct, cls_total = 0, 0

    for idx_i, seed_i in enumerate(init_states):
        # 真实环境闭环
        obs, _ = env.reset(seed=seed_i)
        s0 = extract_obs_5d(obs, norm_fn)
        buf = [s0.copy() for _ in range(policy.T_o)]
        r_max = 0.0
        step = 0
        while step < max_steps:
            chunk = policy.predict_chunk(torch.from_numpy(np.asarray(buf)).unsqueeze(0))[0].numpy()
            for k in range(min(T_a, policy.T_p)):
                if step >= max_steps:
                    break
                a_env = np.clip((chunk[k] + 1.0) * 256.0, 0.0, 512.0)
                obs, r, term, trunc, info = env.step(a_env)
                cov = float(info.get("coverage", r))
                r_max = max(r_max, cov)
                sn = extract_obs_5d(obs, norm_fn)
                with torch.no_grad():
                    c_hat = wm.cov_head(torch.from_numpy(sn).unsqueeze(0)).item()
                cls_correct += int((c_hat >= thresh) == (cov >= thresh))
                cls_total += 1
                buf = buf[1:] + [sn]
                step += 1
                if term or trunc:
                    step = max_steps
                    break
        real_scores.append(r_max); real_succ += int(r_max >= thresh)

        # 世界模型闭环（从完全相同的 s0 出发）
        buf_w = [s0.copy() for _ in range(policy.T_o)]
        cur_s = torch.from_numpy(s0).unsqueeze(0)
        w_max = 0.0
        step = 0
        with torch.no_grad():
            while step < max_steps:
                chunk = policy.predict_chunk(torch.from_numpy(np.asarray(buf_w)).unsqueeze(0))[0]
                for k in range(min(T_a, policy.T_p)):
                    if step >= max_steps:
                        break
                    a_k = chunk[k].clamp(-1.0, 1.0).unsqueeze(0)
                    cur_s, c_hat = wm.step(cur_s, a_k)
                    w_max = max(w_max, float(c_hat.item()))
                    buf_w = buf_w[1:] + [cur_s[0].numpy()]
                    step += 1
        wm_scores.append(w_max); wm_succ += int(w_max >= thresh)

    env.close()
    n = len(init_states)
    return float(np.mean(real_scores)), float(np.mean(wm_scores)), real_succ / n, wm_succ / n, cls_correct / max(cls_total, 1)


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()
    verify_section_6_5_hand_calcs()

    print("\n" + "=" * 60)
    print("第二部分：PushT 策略套件在真实仿真器 vs 学到的世界模型中的排序一致性")
    print("=" * 60)
    demos = load_pusht_demos()
    obs_X, act_Y, norm_fn = build_state_chunk_dataset(demos, n_episodes=30 if quick else 140, T_o=2, T_p=8)

    # 1. 训练世界模型（混合专家 + 随机失败轨迹）
    s_d, a_d, ns_d, cov_d = collect_mixed_wm_data(demos, norm_fn, n_demo_eps=15 if quick else 50, n_rand_eps=6 if quick else 20)
    wm = PushTStateWorldModel()
    opt_wm = torch.optim.AdamW(wm.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(150 if quick else 1200):
        idx = torch.randint(0, len(s_d), (128,))
        ns_hat, c_hat = wm.step(s_d[idx], a_d[idx])
        loss = F.mse_loss(ns_hat, ns_d[idx]) + 0.5 * F.mse_loss(c_hat, cov_d[idx])
        opt_wm.zero_grad(); loss.backward(); opt_wm.step()

    # 2. 构建 6 个不同训练程度的策略套件
    def train_bc(steps):
        m = BCPolicy(5, 2, 2, 8, hidden=160)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(steps):
            i = torch.randint(0, len(obs_X), (128,))
            opt.zero_grad(); F.mse_loss(m(obs_X[i]), act_Y[i]).backward(); opt.step()
        return m

    def train_dp(steps):
        m = DiffusionPolicy(5, 2, 2, 8, hidden=160)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(steps):
            i = torch.randint(0, len(obs_X), (128,))
            opt.zero_grad(); m.loss(act_Y[i], obs_X[i]).backward(); opt.step()
        return m

    scale = 0.2 if quick else 1.0
    suite = [
        ("Random", RandomPolicy()),
        ("BC-early", train_bc(max(10, int(60 * scale)))),
        ("BC-mid", train_bc(max(25, int(250 * scale)))),
        ("BC-full", train_bc(max(60, int(900 * scale)))),
        ("DP-mid", train_dp(max(40, int(350 * scale)))),
        ("DP-full", train_dp(max(80, int(1100 * scale)))),
    ]

    init_seeds = [3000 + i for i in range(6 if quick else 16)]
    full_T = 60 if quick else 120
    half_T = full_T // 2

    y_full, yh_full, y_half, yh_half, cls_accs = [], [], [], [], []
    print(f"\n  {'策略':10s} | {'真实覆盖(全长)':>12s} | {'模型覆盖(全长)':>12s} | {'真实覆盖(半长)':>12s} | {'模型覆盖(半长)':>12s}")
    for nm, pol in suite:
        rf, wf, _, _, ca = evaluate_policy_pair(pol, wm, norm_fn, init_seeds, max_steps=full_T)
        rh, wh, _, _, _ = evaluate_policy_pair(pol, wm, norm_fn, init_seeds, max_steps=half_T)
        y_full.append(rf); yh_full.append(wf); y_half.append(rh); yh_half.append(wh); cls_accs.append(ca)
        print(f"  {nm:10s} | {rf:12.3f} | {wf:12.3f} | {rh:12.3f} | {wh:12.3f}")

    p_f, s_f, m_f = pearson_r(y_full, yh_full), spearman_rho(y_full, yh_full), mmrv(y_full, yh_full)
    p_h, s_h, m_h = pearson_r(y_half, yh_half), spearman_rho(y_half, yh_half), mmrv(y_half, yh_half)
    print(f"\n  一致性指标汇总（成功判定器在真实状态上的准确率 = {np.mean(cls_accs)*100:.1f}%）：")
    print(f"    完整长度 ({full_T} 步)：Pearson = {p_f:.3f} | Spearman = {s_f:.3f} | MMRV = {m_f:.3f}")
    print(f"    缩短一半 ({half_T} 步)：Pearson = {p_h:.3f} | Spearman = {s_h:.3f} | MMRV = {m_h:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(y_full, yh_full, s=65, label=f"全长 (ρ_P={p_f:.2f}, ρ_S={s_f:.2f}, MMRV={m_f:.3f})")
    for i, (nm, _) in enumerate(suite):
        axes[0].annotate(nm, (y_full[i], yh_full[i]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_xlabel("真实仿真器平均最高覆盖度 y"); axes[0].set_ylabel("世界模型平均最高覆盖度 ŷ")
    axes[0].set_title("世界模型作为策略评估器（全长回合）"); axes[0].legend(fontsize=8)

    axes[1].scatter(y_half, yh_half, s=65, color="tab:green", label=f"半长 (ρ_P={p_h:.2f}, ρ_S={s_h:.2f}, MMRV={m_h:.3f})")
    for i, (nm, _) in enumerate(suite):
        axes[1].annotate(nm, (y_half[i], yh_half[i]), xytext=(4, 3), textcoords="offset points", fontsize=8)
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_xlabel("真实仿真器覆盖度 y"); axes[1].set_ylabel("世界模型覆盖度 ŷ")
    axes[1].set_title("缩短展开长度一半时的一致性变化"); axes[1].legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, OUT / "06-wm-as-simulator.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：07-world-model-as-data-engine.py")


if __name__ == "__main__":
    main()
