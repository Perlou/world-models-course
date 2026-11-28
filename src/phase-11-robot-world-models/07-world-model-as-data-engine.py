"""
Phase 11 · 第 07 课：把世界模型当数据引擎 —— 有偏生成数据的增益边界与一致性过滤
=============================================================================

学习目标：
    1. 数值模拟 CONCEPT.md 第 7.3 节偏差-方差公式（σ=0.5, b=0.1, 生成量 9×）：
       核对数据少 (10+90 → 0.0106 < 0.0250) 与数据多 (100+900 → 0.00835 > 0.0025) 的反转现象
    2. 在 PushT 上实现缩小版 DreamGen 数据引擎流水线：
       真实起点 → 扰动动作在世界模型里生成神经轨迹 → 逆动力学模型打伪动作标签 → 自洽性过滤器筛选
    3. 对比「仅真实数据」「真实 + 未过滤生成」「真实 + 过滤后生成」在低数据量与高数据量两档下的
       真实 PushT 策略表现，并报告过滤器丢弃比例与丢弃样本特征

对应概念文档：
    src/phase-11-robot-world-models/CONCEPT.md 第 7 节

运行方式：
    uv run python src/phase-11-robot-world-models/07-world-model-as-data-engine.py
    WM_QUICK=1 uv run python src/phase-11-robot-world-models/07-world-model-as-data-engine.py

预计运行时间：完整约 85 秒（M1 Max CPU），冒烟约 12 秒
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
_spec03 = importlib.util.spec_from_file_location("p11_03_de", os.path.join(_DIR, "03-behavior-cloning-and-diffusion-policy.py"))
_mod03 = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(_mod03)

_spec06 = importlib.util.spec_from_file_location("p11_06_de", os.path.join(_DIR, "06-world-model-as-simulator.py"))
_mod06 = importlib.util.module_from_spec(_spec06)
_spec06.loader.exec_module(_mod06)

BCPolicy = _mod03.BCPolicy
build_state_chunk_dataset = _mod03.build_state_chunk_dataset
rollout_policy = _mod03.rollout_policy
PushTStateWorldModel = _mod06.PushTStateWorldModel

plt = setup_matplotlib()
OUT = output_dir("phase-11")
SEED = 0


# ==================== 第一部分：第 7.3 节偏差-方差反转数值模拟 ====================


def bias_variance_demo():
    print("=" * 60)
    print("第一部分：数值模拟 CONCEPT.md 第 7.3 节有偏生成数据的增益反转")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    sigma, bias, trials = 0.5, 0.1, 20000
    out = {}
    for label, nr, ng in [("数据少 (10+90)", 10, 90), ("数据多 (100+900)", 100, 900)]:
        real = rng.normal(0.0, sigma, size=(trials, nr))
        gen = rng.normal(bias, sigma, size=(trials, ng))
        mse_real_emp = float((real.mean(axis=1) ** 2).mean())
        mix_mean = (real.sum(axis=1) + gen.sum(axis=1)) / (nr + ng)
        mse_mix_emp = float((mix_mean ** 2).mean())
        mse_real_th = (sigma ** 2) / nr
        mse_mix_th = ((ng / (nr + ng)) * bias) ** 2 + (sigma ** 2) / (nr + ng)
        out[label] = (mse_mix_emp, mse_real_emp, mse_mix_th, mse_real_th)
        print(
            f"  {label:16s} | 混合 MSE = {mse_mix_emp:.5f} (理论 {mse_mix_th:.5f}) | "
            f"纯真实 MSE = {mse_real_emp:.5f} (理论 {mse_real_th:.5f}) | 混合更优? {mse_mix_emp < mse_real_emp}"
        )
    return out


# ==================== 第二部分：神经轨迹生成 + IDM 打伪标 + 一致性过滤 ====================


class StateIDM(nn.Module):
    def __init__(self, state_dim=5, act_dim=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * state_dim, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, act_dim))

    def forward(self, s_t, s_next):
        return self.net(torch.cat([s_t, s_next], dim=-1))


@torch.no_grad()
def generate_and_filter_neural_data(wm, idm, obs_real, act_real, n_gen=400, noise_std=0.12, filter_thresh=0.14):
    """从真实起点出发给动作加小扰动在世界模型里展开，再用 IDM 反推伪动作并做一致性过滤。"""
    wm.eval(); idm.eval()
    B_all, T_o, D_s = obs_real.shape
    T_p = act_real.shape[1]
    idx = torch.randint(0, B_all, (n_gen,))
    o_starts = obs_real[idx].clone()  # [M, T_o, 5]
    a_base = act_real[idx].clone()    # [M, T_p, 2]
    # 轻微扰动动作生成反事实邻域轨迹
    a_pert = (a_base + noise_std * torch.randn_like(a_base)).clamp(-1.0, 1.0)

    gen_obs, gen_pseudo_acts, errs = [], [], []
    cur_s = o_starts[:, -1]
    prev_s = o_starts[:, 0]
    pseudo_chunks = []
    traj_errs = torch.zeros(n_gen)

    s_seq = [cur_s]
    for k in range(T_p):
        ns, _ = wm.step(s_seq[-1], a_pert[:, k])
        a_pseudo = idm(s_seq[-1], ns).clamp(-1.0, 1.0)
        pseudo_chunks.append(a_pseudo)
        traj_errs += (a_pseudo - a_pert[:, k]).norm(dim=-1) / T_p
        s_seq.append(ns)

    pseudo_chunk_t = torch.stack(pseudo_chunks, dim=1)  # [M, T_p, 2]
    keep_mask = traj_errs < filter_thresh
    drop_rate = float((~keep_mask).float().mean().item())
    kept_err = float(traj_errs[keep_mask].mean().item()) if keep_mask.any() else 0.0
    drop_err = float(traj_errs[~keep_mask].mean().item()) if (~keep_mask).any() else 0.0
    return o_starts, pseudo_chunk_t, keep_mask, drop_rate, kept_err, drop_err


def train_with_mixed_data(obs_r, act_r, obs_g=None, act_g=None, n_steps=700, real_ratio=0.5, seed=0):
    seed_everything(seed)
    pol = BCPolicy(5, 2, 2, 8, hidden=160)
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(n_steps):
        if obs_g is not None and len(obs_g) > 0:
            br = max(8, int(96 * real_ratio))
            bg = 96 - br
            ir = torch.randint(0, len(obs_r), (br,))
            ig = torch.randint(0, len(obs_g), (bg,))
            ob = torch.cat([obs_r[ir], obs_g[ig]], dim=0)
            ab = torch.cat([act_r[ir], act_g[ig]], dim=0)
        else:
            ir = torch.randint(0, len(obs_r), (96,))
            ob, ab = obs_r[ir], act_r[ir]
        opt.zero_grad(); F.mse_loss(pol(ob), ab).backward(); opt.step()
    return pol


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    bv = bias_variance_demo()

    print("\n" + "=" * 60)
    print("第二部分：PushT 数据引擎 —— 低数据量 vs 高数据量下的生成数据增益与过滤")
    print("=" * 60)
    demos = load_pusht_demos()
    # 用所有演示的状态转移预训练一个共享的「基础世界模型 + IDM」（模拟预训练物理基座）
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    all_o, all_a, norm_fn = build_state_chunk_dataset(demos, n_episodes=30 if quick else 120, T_o=2, T_p=8)

    s_l, a_l, ns_l = [], [], []
    extract_demo_states_5d = _mod03.extract_demo_states_5d
    for ep in range(min(30 if quick else 120, len(starts))):
        s, e = int(starts[ep]), int(ends[ep])
        st_ep = norm_fn(extract_demo_states_5d(demos, s, e))
        ac_ep = (demos["actions"][s:e].astype(np.float32) / 256.0) - 1.0
        s_l.append(st_ep[:-1]); a_l.append(ac_ep[:-1]); ns_l.append(st_ep[1:])
    s_t = torch.from_numpy(np.concatenate(s_l))
    a_t = torch.from_numpy(np.concatenate(a_l))
    ns_t = torch.from_numpy(np.concatenate(ns_l))

    wm = PushTStateWorldModel()
    idm = StateIDM()
    opt_wm = torch.optim.Adam(wm.parameters(), lr=1e-3)
    opt_idm = torch.optim.Adam(idm.parameters(), lr=1e-3)
    for _ in range(120 if quick else 900):
        idx = torch.randint(0, len(s_t), (128,))
        opt_wm.zero_grad(); F.mse_loss(wm.step(s_t[idx], a_t[idx])[0], ns_t[idx]).backward(); opt_wm.step()
        opt_idm.zero_grad(); F.mse_loss(idm(s_t[idx], ns_t[idx]), a_t[idx]).backward(); opt_idm.step()

    regimes = [("少真实演示 (5 回合)", 5), ("多真实演示 (60 回合)", 20 if quick else 60)]
    n_eval = 6 if quick else 20
    max_steps = 70 if quick else 160
    train_steps = 150 if quick else 900
    res_table = {}

    for label, n_real_ep in regimes:
        obs_r, act_r, _ = build_state_chunk_dataset(demos, n_episodes=n_real_ep, T_o=2, T_p=8)
        o_gen, a_gen, keep_mask, drop_rate, k_err, d_err = generate_and_filter_neural_data(
            wm, idm, obs_r, act_r, n_gen=len(obs_r) * 3, noise_std=0.14, filter_thresh=0.16,
        )
        o_filt, a_filt = o_gen[keep_mask], a_gen[keep_mask]

        p_real = train_with_mixed_data(obs_r, act_r, None, None, n_steps=train_steps, seed=SEED)
        p_raw = train_with_mixed_data(obs_r, act_r, o_gen, a_gen, n_steps=train_steps, real_ratio=0.35, seed=SEED)
        p_flt = train_with_mixed_data(obs_r, act_r, o_filt, a_filt, n_steps=train_steps, real_ratio=0.50, seed=SEED)

        sr_r, _, cov_r, _ = rollout_policy(p_real, norm_fn, T_a=4, n_episodes=n_eval, max_steps=max_steps)
        sr_u, _, cov_u, _ = rollout_policy(p_raw, norm_fn, T_a=4, n_episodes=n_eval, max_steps=max_steps)
        sr_f, _, cov_f, _ = rollout_policy(p_flt, norm_fn, T_a=4, n_episodes=n_eval, max_steps=max_steps)
        res_table[label] = (cov_r, cov_u, cov_f, sr_r / n_eval, sr_u / n_eval, sr_f / n_eval, drop_rate)

        print(f"\n  【{label}】过滤器丢弃比例 = {drop_rate*100:.1f}%（保留自洽误差 {k_err:.3f} vs 丢弃 {d_err:.3f}）")
        print(f"    仅真实数据         ：成功率 = {sr_r/n_eval*100:5.1f}% | 平均最高覆盖度 = {cov_r:.3f}")
        print(f"    真实 + 未过滤生成  ：成功率 = {sr_u/n_eval*100:5.1f}% | 平均最高覆盖度 = {cov_u:.3f} (增益 {cov_u - cov_r:+.3f})")
        print(f"    真实 + 过滤后生成  ：成功率 = {sr_f/n_eval*100:5.1f}% | 平均最高覆盖度 = {cov_f:.3f} (增益 {cov_f - cov_r:+.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    bv_labels = list(bv.keys())
    x = np.arange(len(bv_labels))
    axes[0].bar(x - 0.16, [bv[k][1] for k in bv_labels], 0.32, label="纯真实 MSE")
    axes[0].bar(x + 0.16, [bv[k][0] for k in bv_labels], 0.32, label="混合生成 MSE")
    axes[0].set_xticks(x); axes[0].set_xticklabels(bv_labels)
    axes[0].set_title("第 7.3 节偏差-方差反转数值模拟"); axes[0].legend()

    r_labels = list(res_table.keys())
    x2 = np.arange(len(r_labels))
    axes[1].bar(x2 - 0.22, [res_table[k][0] for k in r_labels], 0.22, label="仅真实")
    axes[1].bar(x2, [res_table[k][1] for k in r_labels], 0.22, label="真实+未过滤")
    axes[1].bar(x2 + 0.22, [res_table[k][2] for k in r_labels], 0.22, label="真实+过滤")
    axes[1].set_xticks(x2); axes[1].set_xticklabels(r_labels)
    axes[1].set_title("PushT 数据引擎：不同真实数据量下的覆盖度增益"); axes[1].set_ylim(0, 1.05); axes[1].legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, OUT / "07-wm-as-data-engine.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：08-real-robot-world-models.py")


if __name__ == "__main__":
    main()
