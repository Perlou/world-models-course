"""
Dreamer V1：λ-Return 价值目标与沿可微世界模型反向传播
=====================================================

学习目标：
    1. 实现 lambda_return(rewards, values, conts, gamma, lam)，并严格核对 CONCEPT.md 第 3.3 节的手算例子
       （无终止：[6.2875, 6.75, 11.0, 10.0]；第 2 步终止：R_1 = 0, R_0 = 3.25）
    2. 核对 CONCEPT.md 第 3.4 节的一维玩具系统自动求导梯度（J = -1.25, dJ/dκ = 6.0）
    3. 实现 actor_loss_pathwise()（梯度穿过冻结的 RSSM 与 critic 只更新 actor）与 critic_loss_mse()（输入与目标均 detach）
    4. 在像素版 Pendulum-v1 上完整跑通 Dreamer V1（复用 Phase 3 的高斯 RSSM），
       与 Phase 3 的 PlaNet 在相同环境步数下对比回合回报与单步决策耗时

核心概念：
    - 想象轨迹上的 λ-return：R^λ_k = r̂_k + γ ĉ_{k+1} [(1−λ) v_{k+1} + λ R^λ_{k+1}]，R^λ_H = v_H
    - 穿过但不更新：actor 优化器只含 φ，冻结世界模型 θ 与 critic ψ 的梯度写入，仅借用其计算图
    - 半梯度 TD（semi-gradient TD）：critic 损失对目标 R^λ_k 和输入状态 m̂_k 均加 detach

前置知识：
    - Phase 3 的高斯 RSSM 与像素 Pendulum-v1（04、05 节）
    - 02-imagination-rollouts.py

对应概念文档：
    - CONCEPT.md 第 3 节「Dreamer V1：沿可微的世界模型反向传播」

预计运行时间：约 6 分钟（CPU）
"""

import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_P3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "phase-03-rssm-planet"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_p3_04 = _load(os.path.join(_P3_DIR, "04-rssm-from-scratch.py"), "p3_04")
_p3_05 = _load(os.path.join(_P3_DIR, "05-world-model-training.py"), "p3_05")

GaussianWorldModel = _p3_04.WorldModel
IMAGE_SIZE = _p3_04.IMAGE_SIZE
ACTION_DIM = _p3_04.ACTION_DIM

SequenceReplay = _p3_05.SequenceReplay
render_pendulum = _p3_05.render_pendulum
step_with_repeat = _p3_05.step_with_repeat
preprocess_obs = _p3_05.preprocess_obs
world_model_loss = _p3_05.world_model_loss
EPISODE_DECISIONS = _p3_05.EPISODE_DECISIONS

plt = setup_matplotlib()
OUT = output_dir("phase-06")
SEED = 0
DEVICE = get_device()


# ==================== 第一部分：λ-return 与损失函数 ====================


def lambda_return(rewards, values, conts, gamma=0.99, lam=0.95):
    """CONCEPT.md 第 3.3 节：在想象轨迹上从后往前计算 λ-return。

    rewards: [H, N]，values: [H+1, N]，conts: [H, N]（ĉ_1 … ĉ_H）。
    返回 returns: [H+1, N]，其中 returns[H] = values[H]。
    """
    H = rewards.shape[0]
    outs = [None] * (H + 1)
    outs[H] = values[H]
    for k in reversed(range(H)):
        outs[k] = rewards[k] + gamma * conts[k] * ((1.0 - lam) * values[k + 1] + lam * outs[k + 1])
    return torch.stack(outs, dim=0)


def actor_loss_pathwise(returns, weights):
    """CONCEPT.md 第 3.2、3.8 节：Dreamer V1 的 actor 损失（returns 保留计算图，不 detach）。"""
    return -(weights[:-1].detach() * returns[:-1]).mean()


def critic_loss_mse(critic, imag_states, returns, weights):
    """CONCEPT.md 第 3.5、3.8 节：平方误差 critic 损失（输入状态与 λ-return 目标均 detach）。"""
    v_pred = critic(imag_states[:-1].detach()).squeeze(-1)
    target = returns[:-1].detach()
    return 0.5 * (weights[:-1].detach() * (v_pred - target).pow(2)).mean()


# ==================== 第二部分：手算单元测试 ====================


def verify_hand_calculations():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 3.3 节 λ-return 与第 3.4 节路径梯度手算例子")
    print("=" * 60)

    # 1. 第 3.3 节 λ-return（H=3, γ=0.9, λ=0.5）
    r = torch.tensor([[1.0], [0.0], [2.0]])
    v = torch.tensor([[0.0], [5.0], [4.0], [10.0]])
    c_no_term = torch.ones(3, 1)
    ret1 = lambda_return(r, v, c_no_term, gamma=0.9, lam=0.5).squeeze(-1)
    print(f"  1. 无终止 λ-return：{[round(x, 4) for x in ret1.tolist()]}（手算 [6.2875, 6.75, 11.0, 10.0]）")
    assert torch.allclose(ret1, torch.tensor([6.2875, 6.75, 11.0, 10.0]), atol=1e-4)

    c_term = torch.tensor([[1.0], [0.0], [1.0]])  # ĉ_2 = 0 → 第 2 步终止
    ret2 = lambda_return(r, v, c_term, gamma=0.9, lam=0.5).squeeze(-1)
    print(f"  2. 第 2 步终止 λ-return：R_0 = {ret2[0].item():.4f}（手算 3.25），R_1 = {ret2[1].item():.4f}（手算 0.0）")
    assert abs(ret2[0].item() - 3.25) < 1e-4 and abs(ret2[1].item() - 0.0) < 1e-4

    # 2. 第 3.4 节一维玩具系统（s'=s+a, a=-κs, r=-s'^2, s_0=2, κ=0.5, H=2）
    kappa = torch.tensor(0.5, requires_grad=True)
    s = torch.tensor(2.0)
    J = 0.0
    for _ in range(2):
        a = -kappa * s
        s = s + a
        J = J - s.pow(2)
    J.backward()
    print(f"  3. 一维路径梯度手算：J(κ=0.5) = {J.item():.4f}（手算 −1.25），dJ/dκ = {kappa.grad.item():.4f}（手算 6.0）")
    assert abs(J.item() - (-1.25)) < 1e-6 and abs(kappa.grad.item() - 6.0) < 1e-6
    print("  ✓ 全部手算验证通过！")


# ==================== 第三部分：Dreamer V1 智能体（高斯 RSSM） ====================


class DreamerV1Actor(nn.Module):
    def __init__(self, feat_dim=230, act_dim=1, hidden=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.std_head = nn.Linear(hidden, act_dim)

    def forward(self, feat, deterministic=False):
        h = self.net(feat)
        mean = self.mean_head(h)
        if deterministic:
            return torch.tanh(mean)
        std = F.softplus(self.std_head(h) + 0.5) + 0.1
        return torch.tanh(mean + std * torch.randn_like(mean))


class DreamerV1Critic(nn.Module):
    def __init__(self, feat_dim=230, hidden=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat):
        return self.net(feat)


@torch.no_grad()
def collect_pixel_episode(seed, wm=None, actor=None, explore_std=0.0):
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 123)
    L = EPISODE_DECISIONS + 1
    obs_ep = np.zeros((L, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    act_ep = np.zeros((L, ACTION_DIM), dtype=np.float32)
    rew_ep = np.zeros(L, dtype=np.float32)
    obs_ep[0] = render_pendulum(env)

    act_time = 0.0
    if wm is not None and actor is not None:
        wm.eval()
        actor.eval()
        h, z = wm.rssm.initial_state(1, DEVICE)
        a_prev = torch.zeros(1, ACTION_DIM, device=DEVICE)

    for t in range(1, L):
        if wm is None or actor is None:
            if seed % 2 == 1 and t > 1:
                a = np.clip(0.7 * act_ep[t - 1] + 0.65 * rng.standard_normal(ACTION_DIM), -1.0, 1.0).astype(np.float32)
            else:
                a = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
        else:
            t0 = time.time()
            obs_t = preprocess_obs(torch.from_numpy(obs_ep[t - 1 : t]).to(DEVICE))
            e_t = wm.encoder(obs_t)
            h, z, _, _ = wm.rssm.observe_step(h, z, a_prev, e_t, sample=False)
            feat = torch.cat([h, z], dim=-1)
            a_t = actor(feat, deterministic=(explore_std <= 0.0))
            a = a_t.squeeze(0).cpu().numpy().astype(np.float32)
            if explore_std > 0.0:
                a = np.clip(a + rng.normal(0.0, explore_std, size=ACTION_DIM), -1.0, 1.0).astype(np.float32)
            a_prev = torch.from_numpy(a[None]).to(DEVICE)
            act_time += time.time() - t0

        frame, r, _ = step_with_repeat(env, a)
        obs_ep[t] = frame
        act_ep[t] = a
        rew_ep[t] = r
    env.close()
    return dict(obs=obs_ep, actions=act_ep, rewards=rew_ep, ret=float(rew_ep.sum()), act_ms_per_step=act_time / EPISODE_DECISIONS * 1000.0)


def train_dreamer_v1():
    print("\n" + "=" * 60)
    print("第二部分：在像素版 Pendulum-v1 上训练 Dreamer V1")
    print("=" * 60)
    quick = is_quick()
    n_init = 2 if quick else 5
    n_online = 2 if quick else 10
    upd_init = 12 if quick else 100
    upd_per_ep = 8 if quick else 45
    batch_size = 4 if quick else 10
    chunk_len = 12 if quick else 20
    horizon = 8 if quick else 15
    eval_seeds = [1000] if quick else [1000, 1001, 1002, 1003, 1004]

    replay = SequenceReplay()
    for i in range(n_init):
        ep = collect_pixel_episode(SEED + i)
        replay.add_episode(ep["obs"], ep["actions"], ep["rewards"])

    wm = GaussianWorldModel().to(DEVICE)
    actor = DreamerV1Actor().to(DEVICE)
    critic = DreamerV1Critic().to(DEVICE)

    wm_opt = torch.optim.Adam(wm.parameters(), lr=6e-4)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    gen = torch.Generator().manual_seed(SEED)

    eval_steps, eval_means, eval_stds = [], [], []
    decision_ms_list = []
    t_start = time.time()

    for round_idx in range(1, n_online + 1):
        n_upd = upd_init if round_idx == 1 else upd_per_ep
        wm.train()
        actor.train()
        critic.train()
        for _ in range(n_upd):
            obs_b, act_b, rew_b = replay.sample(batch_size, chunk_len, DEVICE, generator=gen)

            # ① 更新世界模型
            wm_loss, wm_m = world_model_loss(wm, obs_b, act_b, rew_b)
            wm_opt.zero_grad()
            wm_loss.backward()
            nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
            wm_opt.step()

            # 用当前世界模型算出后验起点（不带梯度）
            with torch.no_grad():
                embeds = wm.encode_seq(obs_b)
                obs_out = wm.rssm.observe(embeds, act_b)
                h0 = obs_out["features"][..., :200].reshape(-1, 200)
                z0 = obs_out["features"][..., 200:].reshape(-1, 30)

            # ② 在想象中更新 Actor（冻结世界模型与 critic 参数，梯度只穿过不更新）
            for p in list(wm.parameters()) + list(critic.parameters()):
                p.requires_grad_(False)

            h_cur, z_cur = h0.detach(), z0.detach()
            states = [torch.cat([h_cur, z_cur], dim=-1)]
            for _ in range(horizon):
                a_k = actor(states[-1], deterministic=False)
                h_cur, z_cur, _ = wm.rssm.imagine_step(h_cur, z_cur, a_k, sample=True)
                states.append(torch.cat([h_cur, z_cur], dim=-1))
            imag_states = torch.stack(states, dim=0)  # [H+1, N, 230]
            imag_rewards = wm.reward_head(imag_states[1:])  # [H, N]
            imag_values = critic(imag_states).squeeze(-1)   # [H+1, N]
            imag_conts = torch.ones_like(imag_rewards)
            weights = torch.ones_like(imag_values)

            rets = lambda_return(imag_rewards, imag_values, imag_conts, gamma=0.99, lam=0.95)
            a_loss = actor_loss_pathwise(rets, weights)
            actor_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
            actor_opt.step()

            # 解冻世界模型与 critic，更新 Critic（输入与目标均 detach）
            for p in list(wm.parameters()) + list(critic.parameters()):
                p.requires_grad_(True)
            c_loss = critic_loss_mse(critic, imag_states, rets, weights)
            critic_opt.zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
            critic_opt.step()

        # ③ 在线采集 1 个新回合
        ep = collect_pixel_episode(SEED + 200 + round_idx, wm=wm, actor=actor, explore_std=0.25)
        replay.add_episode(ep["obs"], ep["actions"], ep["rewards"])
        env_steps = len(replay) * 200

        if round_idx == 1 or round_idx % 2 == 0 or round_idx == n_online:
            evs = [collect_pixel_episode(s, wm=wm, actor=actor, explore_std=0.0) for s in eval_seeds]
            ev_m = float(np.mean([e["ret"] for e in evs]))
            ev_s = float(np.std([e["ret"] for e in evs]))
            ms = float(np.mean([e["act_ms_per_step"] for e in evs]))
            eval_steps.append(env_steps)
            eval_means.append(ev_m)
            eval_stds.append(ev_s)
            decision_ms_list.append(ms)
            print(f"  第 {round_idx:>2}/{n_online} 轮（累计 {env_steps:>4} 步）："
                  f"重建 {wm_m['recon']:5.1f} ｜ KL {wm_m['raw_kl']:4.1f} ｜ "
                  f"评估回报 {ev_m:7.1f} ± {ev_s:5.1f} ｜ 单步决策 {ms:.2f} ms")

    total_s = time.time() - t_start
    avg_ms = float(np.mean(decision_ms_list))
    print(f"\n  Dreamer V1 训练完成，总用时 {total_s:.1f}s；最终评估回报 {eval_means[-1]:.1f} ± {eval_stds[-1]:.1f}；"
          f"平均单步决策耗时 {avg_ms:.2f} ms（对比 Phase 3 PlaNet CEM 约 201 ms/步，快约 {201.2 / max(avg_ms, 0.01):.0f} 倍！）")

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.errorbar(eval_steps, eval_means, yerr=eval_stds, fmt="o-", color="C0", linewidth=2.2, capsize=4, label="Dreamer V1（像素输入）")
    ax.axhline(-1150.0, color="tomato", linestyle=":", label="随机策略参考（≈−1150）")
    ax.set_xlabel("累计真实环境交互步数")
    ax.set_ylabel("5 回合平均评估回报")
    ax.set_title(f"像素版 Pendulum-v1：Dreamer V1 学习曲线（单步决策 {avg_ms:.1f} ms）")
    ax.legend(fontsize=9)
    save_figure(fig, OUT / "03-dreamer-v1-learning-curve.png")


def main():
    seed_everything(SEED)
    if DEVICE.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    verify_hand_calculations()
    train_dreamer_v1()
    print("\n" + "=" * 60)
    print("本节完成！下一节：04-categorical-latents-straight-through.py（Dreamer V2 类别隐变量）")
    print("=" * 60)


if __name__ == "__main__":
    main()
