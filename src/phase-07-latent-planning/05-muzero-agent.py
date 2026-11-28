"""
MuZero 智能体：K 步展开训练与隐空间 MCTS 搜索
=============================================

学习目标：
    1. 核对 CONCEPT.md 第 5.3 节的全部手算例子：
       - n 步搜索自举目标 G_t^(3) = 6.265（γ=0.9, r=[1,0,2], ν_3=5）
       - K=2 展开损失 = 策略 1.751 + 价值 0.30 + 奖励 0.01 = 2.061
       - 梯度缩放 Σ_{k=0}^5 2^{-k} = 1.96875
    2. 实现 n_step_value_target()、make_targets()、scale_gradient()、muzero_loss() 与带 min-max Q 归一化的隐空间搜索 run_mcts_latent()
    3. 在 CartPole-v1 上完整训练 MuZero，记录学习曲线与三项损失曲线，并对比展开步数 K = 1 与 K = 5
    4. 测量纯 MuZero 训练后 z_t^1 与 z_{t+1}^0 的余弦相似度，直接验证第 4.3 节"无一致性损失时二者不必对齐"的论断

核心概念：
    - K 步展开（K-step unroll）：只对起点调用 enc，后续 K 步完全在隐空间靠 dyn 展开并对齐三项目标
    - 吸收状态边界：真终止后奖励与价值目标置 0，策略损失掩码置 0
    - 隐空间 MCTS：节点仅存隐状态 z^k，边存预测奖励 r̂，PUCT 里动态做 min-max Q 归一化

前置知识：
    - 04-muzero-model.py（MuZeroNet、scale_hidden_state）

对应概念文档：
    - CONCEPT.md 第 5 节「MuZero 的训练：K 步展开」（含第 4.5 节隐空间搜索）

预计运行时间：约 1.5 分钟（CPU）
"""

import importlib.util
import math
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

MuZeroNet = _mod04.MuZeroNet
measure_latent_alignment = _mod04.measure_latent_alignment

plt = setup_matplotlib()
OUT = output_dir("phase-07")
SEED = 0


# ==================== 第一部分：梯度缩放、目标构造与损失 ====================


def scale_gradient(x, scale):
    """CONCEPT.md 第 5.3 节：前向恒等，反向梯度乘以 scale。"""
    return x * scale + x.detach() * (1.0 - scale)


def n_step_value_target(rewards, root_values, idx, n_step, gamma, terminated):
    """CONCEPT.md 第 5.3 节递推计算 G_{idx}^{(n)}（处理回合真终止与截断边界）。"""
    L = len(rewards)
    g = 0.0
    boot_idx = idx + n_step
    if boot_idx < L:
        g = float(root_values[boot_idx])
    elif not terminated:
        # 步数上限截断时仍用末步估值近似自举，不把未来清零
        g = float(root_values[-1]) if L > 0 else 0.0
    end_i = min(boot_idx, L)
    for t in reversed(range(idx, end_i)):
        g = float(rewards[t]) + gamma * g
    return g


def make_targets(episode, start_idx, unroll_k, n_step, gamma, act_dim):
    """按 CONCEPT.md 第 5.3 节对齐表，为单条起点构造 K 步展开的全部目标与掩码。"""
    L = len(episode["rewards"])
    terminated = episode["terminated"]

    actions = np.zeros(unroll_k, dtype=np.int64)
    target_pis = np.zeros((unroll_k + 1, act_dim), dtype=np.float32)
    target_vs = np.zeros(unroll_k + 1, dtype=np.float32)
    target_rs = np.zeros(unroll_k, dtype=np.float32)
    pi_masks = np.zeros(unroll_k + 1, dtype=np.float32)

    for k in range(unroll_k + 1):
        t = start_idx + k
        if t < L:
            target_pis[k] = episode["pis"][t]
            target_vs[k] = n_step_value_target(episode["rewards"], episode["root_vs"], t, n_step, gamma, terminated)
            pi_masks[k] = 1.0
        else:
            # 吸收状态：价值目标 0，策略损失不计（掩码 0）
            target_vs[k] = 0.0
            pi_masks[k] = 0.0
        if k > 0:
            t_prev = start_idx + k - 1
            if t_prev < L:
                actions[k - 1] = episode["actions"][t_prev]
                target_rs[k - 1] = episode["rewards"][t_prev]
            else:
                actions[k - 1] = 0
                target_rs[k - 1] = 0.0
    return episode["obs"][start_idx], actions, target_pis, target_vs, target_rs, pi_masks


def muzero_loss(net, obs0, actions, target_pis, target_vs, target_rs, pi_masks):
    """CONCEPT.md 第 5.3、5.5 节：K 步展开计算三项损失（每步乘 1/K，动力学梯度乘 0.5）。"""
    K = actions.shape[1]
    z, p_logits, v_pred = net.initial_inference(obs0)

    log_p = F.log_softmax(p_logits, dim=-1)
    p_loss = (-(target_pis[:, 0] * log_p).sum(dim=-1) * pi_masks[:, 0]).mean()
    v_loss = F.mse_loss(v_pred, target_vs[:, 0])
    r_loss = torch.tensor(0.0, device=obs0.device)

    inv_k = 1.0 / max(1, K)
    for k in range(1, K + 1):
        z = scale_gradient(z, 0.5)
        z, r_pred, p_logits, v_pred = net.recurrent_inference(z, actions[:, k - 1])
        log_p = F.log_softmax(p_logits, dim=-1)
        p_loss = p_loss + inv_k * (-(target_pis[:, k] * log_p).sum(dim=-1) * pi_masks[:, k]).mean()
        v_loss = v_loss + inv_k * F.mse_loss(v_pred, target_vs[:, k])
        r_loss = r_loss + inv_k * F.mse_loss(r_pred, target_rs[:, k - 1])

    total = p_loss + 0.25 * v_loss + r_loss
    return total, p_loss.item(), v_loss.item(), r_loss.item()


# ==================== 第二部分：隐空间 MCTS（单人带奖励 + Min-Max Q 归一化） ====================


class LatentMCTSNode:
    def __init__(self, prior=1.0):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.latent = None
        self.children = {}

    @property
    def q_value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


@torch.no_grad()
def run_mcts_latent(net, obs_np, n_sims=20, gamma=0.99, c_puct=1.25, add_noise=True, rng=None):
    """CONCEPT.md 第 4.5、4.7 节：只在根节点编码观测，树内全部调用 recurrent_inference。"""
    rng = np.random.default_rng(SEED) if rng is None else rng
    act_dim = net.act_dim

    obs_t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)[None])
    z0, p0_logits, v0 = net.initial_inference(obs_t)
    priors0 = F.softmax(p0_logits[0], dim=-1).numpy()
    if add_noise:
        noise = rng.dirichlet([0.5] * act_dim)
        priors0 = 0.75 * priors0 + 0.25 * noise

    root = LatentMCTSNode()
    root.latent = z0
    for a in range(act_dim):
        root.children[a] = LatentMCTSNode(prior=priors0[a])

    q_min, q_max = float("inf"), -float("inf")

    def norm_q(val):
        if q_max > q_min:
            return (val - q_min) / (q_max - q_min)
        return val

    for _ in range(n_sims):
        node = root
        path = [node]
        act_path = []
        while len(node.children) > 0:
            n_parent = max(1, sum(ch.visit_count for ch in node.children.values()))
            best_sc, best_a, best_ch = -float("inf"), 0, None
            for a, ch in node.children.items():
                q_est = norm_q(ch.reward + gamma * ch.q_value) if ch.visit_count > 0 else 0.0
                u = c_puct * ch.prior * math.sqrt(n_parent) / (1 + ch.visit_count)
                sc = q_est + u
                if sc > best_sc:
                    best_sc, best_a, best_ch = sc, a, ch
            act_path.append(best_a)
            node = best_ch
            path.append(node)

        # 扩展叶子节点（调用一次 recurrent_inference）
        parent = path[-2]
        last_a = act_path[-1]
        a_t = torch.tensor([last_a], dtype=torch.long)
        z_new, r_pred, p_logits, v_pred = net.recurrent_inference(parent.latent, a_t)
        node.latent = z_new
        node.reward = float(r_pred.item())
        priors = F.softmax(p_logits[0], dim=-1).numpy()
        for a in range(act_dim):
            node.children[a] = LatentMCTSNode(prior=priors[a])

        # 单人带折扣回传
        g = float(v_pred.item())
        for nd in reversed(path):
            nd.visit_count += 1
            nd.value_sum += g
            q_cur = nd.reward + gamma * nd.q_value
            if q_cur < q_min:
                q_min = q_cur
            if q_cur > q_max:
                q_max = q_cur
            g = nd.reward + gamma * g

    counts = np.array([root.children[a].visit_count for a in range(act_dim)], dtype=np.float64)
    pi = counts / max(1.0, counts.sum())
    return pi.astype(np.float32), float(root.q_value)


# ==================== 第三部分：手算验证与 CartPole-v1 训练 ====================


def verify_section_5_hand_calc():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 5.3 节 n 步目标、K 步损失与梯度缩放手算")
    print("=" * 60)

    # 1. G_t^(3) = 6.265
    g3 = n_step_value_target([1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 5.0], 0, n_step=3, gamma=0.9, terminated=False)
    print(f"  1. n 步自举价值目标 G_0^(3) = {g3:.4f}（手算 6.2650 ✓）")
    assert abs(g3 - 6.265) < 1e-4

    # 2. K=2 展开损失手算
    ps = np.array([[0.5, 0.5], [0.7, 0.3], [0.2, 0.8]])
    pis = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
    p_l = -float(np.sum(pis * np.log(ps)))
    v_l = (1.0 - 1.5) ** 2 + (0.8 - 1.0) ** 2 + (0.5 - 0.4) ** 2
    r_l = (0.9 - 1.0) ** 2 + (0.0 - 0.0) ** 2
    tot = p_l + v_l + r_l
    print(f"  2. K=2 三项损失：策略 {p_l:.3f}（1.751）+ 价值 {v_l:.2f}（0.30）+ 奖励 {r_l:.2f}（0.01）= {tot:.3f}（手算 2.061 ✓）")
    assert abs(tot - 2.061) < 2e-3

    # 3. scale_gradient 与几何级数权重
    x = torch.tensor(3.0, requires_grad=True)
    y = scale_gradient(x, 0.5)
    assert y.item() == 3.0
    (y * 2.0).backward()
    assert abs(x.grad.item() - 1.0) < 1e-6
    geom = sum(0.5**k for k in range(6))
    print(f"  3. scale_gradient(0.5) 验证通过；K=5 总权重 Σ 2^{{-k}} = {geom:.5f}（手算 1.96875 ✓）")


def train_muzero_cartpole(unroll_k=5, n_episodes=80, n_sims=15, n_step=10, gamma=0.99, seed=0):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    net = MuZeroNet(obs_dim=4, act_dim=2, latent_dim=32, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    replay = deque(maxlen=150)

    env_steps = 0
    step_axis, ret_hist = [], []
    pl_hist, vl_hist, rl_hist = [], [], []

    for ep_idx in range(1, n_episodes + 1):
        obs, _ = env.reset(seed=seed * 1000 + ep_idx)
        ep_obs, ep_act, ep_rew, ep_pi, ep_v = [], [], [], [], []
        terminated = False
        net.eval()

        while True:
            if ep_idx <= 8:
                pi = np.array([0.5, 0.5], dtype=np.float32)
                root_v = 0.0
                a = int(rng.integers(0, 2))
            else:
                pi, root_v = run_mcts_latent(net, obs, n_sims=n_sims, gamma=gamma, add_noise=True, rng=rng)
                temp = 1.0 if ep_idx < n_episodes * 0.5 else 0.35
                p_samp = (pi ** (1.0 / temp))
                p_samp /= p_samp.sum()
                a = int(rng.choice(2, p=p_samp))

            nxt, r, term, trunc, _ = env.step(a)
            ep_obs.append(obs.astype(np.float32))
            ep_act.append(a)
            ep_rew.append(float(r))
            ep_pi.append(pi)
            ep_v.append(root_v)
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
            terminated=terminated,
        ))
        step_axis.append(env_steps)
        ret_hist.append(float(sum(ep_rew)))

        # 训练更新
        if len(replay) >= 5:
            net.train()
            p_acc, v_acc, r_acc = [], [], []
            n_upd = min(20, max(6, len(ep_rew) // 4))
            for _ in range(n_upd):
                o0_b, a_b, pi_b, v_b, r_b, m_b = [], [], [], [], [], []
                for _ in range(32):
                    ep = replay[int(rng.integers(0, len(replay)))]
                    s_i = int(rng.integers(0, len(ep["rewards"])))
                    o0, acts, t_pi, t_v, t_r, pm = make_targets(ep, s_i, unroll_k, n_step, gamma, act_dim=2)
                    o0_b.append(o0)
                    a_b.append(acts)
                    pi_b.append(t_pi)
                    v_b.append(t_v)
                    r_b.append(t_r)
                    m_b.append(pm)
                loss, pl, vl, rl = muzero_loss(
                    net,
                    torch.from_numpy(np.stack(o0_b)),
                    torch.from_numpy(np.stack(a_b)),
                    torch.from_numpy(np.stack(pi_b)),
                    torch.from_numpy(np.stack(v_b)),
                    torch.from_numpy(np.stack(r_b)),
                    torch.from_numpy(np.stack(m_b)),
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                opt.step()
                p_acc.append(pl)
                v_acc.append(vl)
                r_acc.append(rl)
            pl_hist.append(float(np.mean(p_acc)))
            vl_hist.append(float(np.mean(v_acc)))
            rl_hist.append(float(np.mean(r_acc)))

    env.close()
    # 独立贪心评估 5 回合
    eval_env = gym.make("CartPole-v1")
    net.eval()
    eval_rets = []
    for ei in range(5):
        eo, _ = eval_env.reset(seed=900 + ei)
        tot = 0.0
        while True:
            pi, _ = run_mcts_latent(net, eo, n_sims=n_sims, gamma=gamma, add_noise=False, rng=rng)
            eo, r, term, trunc, _ = eval_env.step(int(np.argmax(pi)))
            tot += float(r)
            if term or trunc:
                break
        eval_rets.append(tot)
    eval_env.close()

    return net, np.array(step_axis), np.array(ret_hist), np.array(pl_hist), np.array(vl_hist), np.array(rl_hist), float(np.mean(eval_rets))


def main():
    seed_everything(SEED)
    verify_section_5_hand_calc()

    quick = is_quick()
    n_eps = 18 if quick else 90
    n_sims = 10 if quick else 18

    print("\n" + "=" * 60)
    print(f"第二部分：CartPole-v1 上对比 MuZero 展开步数 K = 1 与 K = 5（{n_eps} 回合，每步 {n_sims} 次隐空间 MCTS）")
    print("=" * 60)
    t0 = time.time()
    net_k1, st1, rt1, pl1, vl1, rl1, ev1 = train_muzero_cartpole(unroll_k=1, n_episodes=n_eps, n_sims=n_sims, seed=SEED)
    net_k5, st5, rt5, pl5, vl5, rl5, ev5 = train_muzero_cartpole(unroll_k=5, n_episodes=n_eps, n_sims=n_sims, seed=SEED)
    print(f"  总耗时 {time.time() - t0:.1f}s")
    print(f"  K = 1：末 15 回合均值 {np.mean(rt1[-min(15, len(rt1)):]):6.1f} ｜ 贪心评估均值 {ev1:6.1f}（总交互 {st1[-1]} 步）")
    print(f"  K = 5：末 15 回合均值 {np.mean(rt5[-min(15, len(rt5)):]):6.1f} ｜ 贪心评估均值 {ev5:6.1f}（总交互 {st5[-1]} 步）")

    # 兑现第 4.8 节伏笔：测量训练后 z_t^1 与 z_{t+1}^0 的余弦相似度
    cos_raw, cos_c = measure_latent_alignment(net_k5)
    print(f"\n  训练后隐状态对齐检验（回答第 4.8 节）：")
    print(f"    纯 MuZero (K=5) 训练后：z_t^1 与 z_{{t+1}}^0 原始余弦 = {cos_raw:.4f} ｜ 去中心化余弦 = {cos_c:+.4f}")
    print("    （纯 MuZero 损失不强制隐状态对齐，所以下一节 EfficientZero 加入自监督一致性损失！）")

    def smooth(y, w=7):
        if len(y) < w:
            return y
        return np.convolve(y, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))
    s1, s5 = smooth(rt1), smooth(rt5)
    axes[0].plot(st1[len(rt1) - len(s1) :], s1, linewidth=2.2, label=f"MuZero K=1（评估 {ev1:.0f}）")
    axes[0].plot(st5[len(rt5) - len(s5) :], s5, linewidth=2.2, label=f"MuZero K=5（评估 {ev5:.0f}）")
    axes[0].set_xlabel("真实环境交互步数")
    axes[0].set_ylabel("回合回报（滑动平均）")
    axes[0].set_title("CartPole-v1：MuZero 学习曲线（K=1 vs K=5）")
    axes[0].legend(fontsize=9)

    axes[1].plot(pl5, linewidth=1.8, label="策略交叉熵")
    axes[1].plot(vl5, linewidth=1.8, label="价值 MSE")
    axes[1].plot(rl5, linewidth=1.8, label="奖励 MSE")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("回合数")
    axes[1].set_ylabel("损失（对数尺度）")
    axes[1].set_title("MuZero (K=5) 三项损失演进")
    axes[1].legend(fontsize=9)
    save_figure(fig, OUT / "05-muzero-cartpole.png")

    print("\n" + "=" * 60)
    print("本节完成！下一节：06-efficientzero-ideas.py（EfficientZero 三项省样本改进消融）")
    print("=" * 60)


if __name__ == "__main__":
    main()
