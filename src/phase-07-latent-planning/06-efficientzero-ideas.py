"""
EfficientZero 的三项省样本改进：一致性损失、Value Prefix 与离策略修正
=====================================================================

学习目标：
    1. 核对 CONCEPT.md 第 6.2 节负余弦相似度与单位向量平方距离的等价性手算（u=(1,0), v=(0.6,0.8) → -cos=-0.6, ||u-v||²=0.8）
    2. 实现 EfficientZero 的三个独立部件：
       - consistency_loss()：SimSiam 式投影 + 预测头 + 停止梯度，把 z_t^k 拉向 sg(enc(o_{t+k}))
       - ValuePrefixHead：用轻量 LSTM 预测从起点到第 k 步的累计奖励前缀，压低单步奖励错位误差
       - corrected_value_target()：离策略修正（旧轨迹越老，越缩短自举步数 l 以减轻旧策略偏差）
    3. 在 CartPole-v1 有限样本预算下逐项叠加三项改进做消融，记录每组达到目标回报所需的真实环境步数

核心概念：
    - SimSiam 一致性损失：不靠解码器也能每步提供 D 维自监督表征信号
    - Value Prefix（奖励前缀）：预测 Σ_{j=1}^k r_j 比预测单步精确落点更抗时间对齐抖动
    - 离策略自举步数自适应（off-policy correction）：陈旧经验缩短 n 步自举长度

前置知识：
    - 05-muzero-agent.py

对应概念文档：
    - CONCEPT.md 第 6 节「EfficientZero：让 MuZero 省样本」

预计运行时间：约 2 分钟（CPU）
"""

import importlib.util
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec04 = importlib.util.spec_from_file_location("p7_04", os.path.join(_DIR, "04-muzero-model.py"))
_mod04 = importlib.util.module_from_spec(_spec04)
_spec04.loader.exec_module(_mod04)
_spec05 = importlib.util.spec_from_file_location("p7_05", os.path.join(_DIR, "05-muzero-agent.py"))
_mod05 = importlib.util.module_from_spec(_spec05)
_spec05.loader.exec_module(_mod05)

MuZeroNet = _mod04.MuZeroNet
measure_latent_alignment = _mod04.measure_latent_alignment
scale_gradient = _mod05.scale_gradient
n_step_value_target = _mod05.n_step_value_target
run_mcts_latent = _mod05.run_mcts_latent

plt = setup_matplotlib()
OUT = output_dir("phase-07")
SEED = 0


# ==================== 第一部分：三项改进部件 ====================


class SimSiamProjectorPredictor(nn.Module):
    """CONCEPT.md 第 6.2 节：投影网络 proj 与非对称预测头 q，用于自监督隐状态一致性。"""

    def __init__(self, latent_dim=32, proj_dim=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.pred = nn.Sequential(
            nn.Linear(proj_dim, proj_dim), nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def consistency_loss(self, z_pred, z_target):
        p = self.pred(self.proj(z_pred))
        with torch.no_grad():
            t = self.proj(z_target).detach()
        return -F.cosine_similarity(p, t, dim=-1).mean()


class ValuePrefixHead(nn.Module):
    """CONCEPT.md 第 6.3 节：用循环单元沿展开步累积并预测奖励前缀 Σ_{j=1}^k r_j。"""

    def __init__(self, latent_dim=32, hidden=32):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim, hidden)
        self.head = nn.Linear(hidden, 1)
        self.hidden = hidden

    def step(self, z, h):
        h = self.gru(z, h)
        return self.head(h).squeeze(-1), h


def corrected_value_target(episode, idx, base_n_step, age_ratio, gamma):
    """CONCEPT.md 第 6.4 节：数据越陈旧（age_ratio ∈ [0, 1] 越大），越缩短 n 步自举步数以压低离策略偏差。"""
    eff_n = max(1, int(round(base_n_step * (1.0 - 0.7 * age_ratio))))
    return n_step_value_target(episode["rewards"], episode["root_vs"], idx, eff_n, gamma, episode["terminated"])


# ==================== 第二部分：手算验证与逐项消融实验 ====================


def verify_cosine_equivalence():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 6.2 节负余弦相似度与平方距离等价手算")
    print("=" * 60)
    u = torch.tensor([1.0, 0.0])
    v = torch.tensor([0.6, 0.8])
    cos_uv = F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item()
    sq_dist = (u - v).pow(2).sum().item()
    print(f"  u=(1, 0), v=(0.6, 0.8)：cos(u, v) = {cos_uv:.2f}，损失 = {-cos_uv:.2f}，"
          f"||u − v||² = {sq_dist:.2f} = 2 − 2 cos(u,v) = {2 - 2 * cos_uv:.2f} ✓")
    assert abs(sq_dist - (2.0 - 2.0 * cos_uv)) < 1e-6


def run_ez_variant(use_consistency=False, use_value_prefix=False, use_offpolicy_corr=False,
                   n_episodes=65, n_sims=14, seed=0):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    net = MuZeroNet(obs_dim=4, act_dim=2, latent_dim=32, hidden=64)
    sim_head = SimSiamProjectorPredictor(32, 32) if use_consistency else None
    vp_head = ValuePrefixHead(32, 32) if use_value_prefix else None

    params = list(net.parameters())
    if sim_head is not None:
        params += list(sim_head.parameters())
    if vp_head is not None:
        params += list(vp_head.parameters())
    opt = torch.optim.Adam(params, lr=3e-3, weight_decay=1e-4)

    replay = deque(maxlen=150)
    unroll_k = 5
    base_n = 10
    gamma = 0.99

    env_steps = 0
    step_axis, ret_hist = [], []
    hit_150_steps = None

    for ep_idx in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed * 1000 + ep_idx)
        ep_obs, ep_act, ep_rew, ep_pi, ep_v = [], [], [], [], []
        terminated = False
        net.eval()

        while True:
            if ep_idx <= 6:
                pi = np.array([0.5, 0.5], dtype=np.float32)
                rv = 0.0
                a = int(rng.integers(0, 2))
            else:
                pi, rv = run_mcts_latent(net, obs, n_sims=n_sims, gamma=gamma, add_noise=True, rng=rng)
                temp = 1.0 if ep_idx < n_episodes * 0.45 else 0.35
                p_s = (pi ** (1.0 / temp))
                p_s /= p_s.sum()
                a = int(rng.choice(2, p=p_s))
            nxt, r, term, trunc, _ = env.step(a)
            ep_obs.append(obs.astype(np.float32))
            ep_act.append(a)
            ep_rew.append(float(r))
            ep_pi.append(pi)
            ep_v.append(rv)
            obs = nxt
            env_steps += 1
            if term or trunc:
                terminated = bool(term)
                break

        replay.append(dict(
            obs=np.asarray(ep_obs), actions=np.asarray(ep_act),
            rewards=np.asarray(ep_rew, dtype=np.float32),
            pis=np.asarray(ep_pi, dtype=np.float32),
            root_vs=np.asarray(ep_v, dtype=np.float32),
            terminated=terminated, ep_born=ep_idx,
        ))
        ep_ret = float(sum(ep_rew))
        step_axis.append(env_steps)
        ret_hist.append(ep_ret)
        if hit_150_steps is None and len(ret_hist) >= 5 and float(np.mean(ret_hist[-5:])) >= 120.0:
            hit_150_steps = env_steps

        if len(replay) >= 5:
            net.train()
            n_upd = min(18, max(6, len(ep_rew) // 4))
            for _ in range(n_upd):
                B = 32
                o_seq = np.zeros((B, unroll_k + 1, 4), dtype=np.float32)
                o_valid = np.zeros((B, unroll_k + 1), dtype=np.float32)
                a_b = np.zeros((B, unroll_k), dtype=np.int64)
                pi_b = np.zeros((B, unroll_k + 1, 2), dtype=np.float32)
                v_b = np.zeros((B, unroll_k + 1), dtype=np.float32)
                r_b = np.zeros((B, unroll_k), dtype=np.float32)
                vp_b = np.zeros((B, unroll_k), dtype=np.float32)
                pm_b = np.zeros((B, unroll_k + 1), dtype=np.float32)

                for bi in range(B):
                    ep = replay[int(rng.integers(0, len(replay)))]
                    L = len(ep["rewards"])
                    s_i = int(rng.integers(0, L))
                    age = min(1.0, (ep_idx - ep["ep_born"]) / max(1.0, n_episodes * 0.5))
                    prefix_sum = 0.0
                    for k in range(unroll_k + 1):
                        t = s_i + k
                        if t < L:
                            o_seq[bi, k] = ep["obs"][t]
                            o_valid[bi, k] = 1.0
                            pi_b[bi, k] = ep["pis"][t]
                            v_b[bi, k] = (
                                corrected_value_target(ep, t, base_n, age, gamma)
                                if use_offpolicy_corr
                                else n_step_value_target(ep["rewards"], ep["root_vs"], t, base_n, gamma, ep["terminated"])
                            )
                            pm_b[bi, k] = 1.0
                        if k > 0 and s_i + k - 1 < L:
                            a_b[bi, k - 1] = ep["actions"][s_i + k - 1]
                            r_b[bi, k - 1] = ep["rewards"][s_i + k - 1]
                            prefix_sum += ep["rewards"][s_i + k - 1]
                            vp_b[bi, k - 1] = prefix_sum
                        elif k > 0:
                            vp_b[bi, k - 1] = prefix_sum

                o_t = torch.from_numpy(o_seq)
                ov_t = torch.from_numpy(o_valid)
                a_t = torch.from_numpy(a_b)
                pi_t = torch.from_numpy(pi_b)
                v_t = torch.from_numpy(v_b)
                r_t = torch.from_numpy(r_b)
                vp_t = torch.from_numpy(vp_b)
                pm_t = torch.from_numpy(pm_b)

                z, p_l, v_p = net.initial_inference(o_t[:, 0])
                loss = (-(pi_t[:, 0] * F.log_softmax(p_l, -1)).sum(-1) * pm_t[:, 0]).mean() + 0.25 * F.mse_loss(v_p, v_t[:, 0])
                vp_h = torch.zeros(B, 32) if vp_head is not None else None

                for k in range(1, unroll_k + 1):
                    z = scale_gradient(z, 0.5)
                    z, r_p, p_l, v_p = net.recurrent_inference(z, a_t[:, k - 1])
                    step_loss = (-(pi_t[:, k] * F.log_softmax(p_l, -1)).sum(-1) * pm_t[:, k]).mean() + 0.25 * F.mse_loss(v_p, v_t[:, k]) + F.mse_loss(r_p, r_t[:, k - 1])
                    if sim_head is not None and ov_t[:, k].sum() > 0:
                        with torch.no_grad():
                            z_tgt, _, _ = net.initial_inference(o_t[:, k])
                        mask = ov_t[:, k] > 0.5
                        step_loss = step_loss + 2.0 * sim_head.consistency_loss(z[mask], z_tgt[mask])
                    if vp_head is not None:
                        vp_pred, vp_h = vp_head.step(z, vp_h)
                        step_loss = step_loss + 0.5 * F.mse_loss(vp_pred, vp_t[:, k - 1])
                    loss = loss + step_loss / unroll_k

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()

    env.close()
    _, cos_c = measure_latent_alignment(net)
    return np.array(step_axis), np.array(ret_hist), hit_150_steps, cos_c


def main():
    seed_everything(SEED)
    verify_cosine_equivalence()

    quick = is_quick()
    n_eps = 16 if quick else 70
    n_sims = 10 if quick else 15

    configs = [
        ("原始 MuZero (K=5)", False, False, False),
        ("+ 一致性损失 (SimSiam)", True, False, False),
        ("+ 一致性 + Value Prefix", True, True, False),
        ("+ 三项全部 (EfficientZero)", True, True, True),
    ]
    if quick:
        configs = [configs[0], configs[-1]]

    print("\n" + "=" * 60)
    print(f"第二部分：CartPole-v1 逐项叠加 EfficientZero 三项改进消融（{n_eps} 回合）")
    print("=" * 60)
    t0 = time.time()
    curves = {}
    for name, c_flag, vp_flag, oc_flag in configs:
        st, rt, hit, cos_c = run_ez_variant(c_flag, vp_flag, oc_flag, n_episodes=n_eps, n_sims=n_sims, seed=SEED)
        tail_m = float(np.mean(rt[-min(15, len(rt)):]))
        hit_s = f"{hit} 步" if hit is not None else "未达标"
        curves[name] = (st, rt, tail_m)
        print(f"  {name:<28} → 末15回合均值 {tail_m:6.1f} ｜ 达标(≥120) {hit_s:<9} ｜ 去中心化隐态余弦 {cos_c:+.3f}")
    print(f"  总用时 {time.time() - t0:.1f}s")

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for name, (st, rt, tail_m) in curves.items():
        w = min(7, len(rt))
        sm = np.convolve(rt, np.ones(w) / w, mode="valid")
        ax.plot(st[len(rt) - len(sm) :], sm, linewidth=2.2, label=f"{name}（末段 {tail_m:.0f}）")
    ax.axhline(120.0, color="gray", linestyle=":", label="达标参照线（120）")
    ax.set_xlabel("真实环境交互步数")
    ax.set_ylabel("回合回报（滑动平均）")
    ax.set_title("CartPole-v1：EfficientZero 逐项改进消融曲线")
    ax.legend(fontsize=8.5)
    save_figure(fig, OUT / "06-efficientzero-ablation.png")

    print("\n" + "=" * 60)
    print("本节完成！下一节：07-tdmpc-latent-model.py（连续控制无解码器隐模型 + MPPI：TD-MPC）")
    print("=" * 60)


if __name__ == "__main__":
    main()
