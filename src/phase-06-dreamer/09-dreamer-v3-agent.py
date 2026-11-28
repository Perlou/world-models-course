"""
Dreamer V3 完整智能体：Actor、Two-Hot EMA Critic 与想象行为学习闭环
===================================================================

学习目标：
    1. 核对 CONCEPT.md 第 9.4 节的离散动作 actor 梯度手算（回报项 (0.25, -0.35, 0.10)、熵 1.0297、熵梯度 (5.0,-1.6,-3.5)e-5）
    2. 实现完整 DreamerV3Agent：世界模型 θ + TanhGaussianActor φ + TwoHotHead Critic ψ + EMA Critic ψ̄ + ReturnNormalizer
    3. 同时实现两种 actor 梯度模式：REINFORCE（V3 Nature 版）与沿模型反传 pathwise（V3 arXiv / V1），
       在连续控制任务 Pendulum-v1 上同条件实测对比两者样本效率与稳定性
    4. 在像素版 Pendulum-v1 上跑通完整的 Dreamer V3 在线闭环，验证所有 10 处 stop-gradient 协同工作

核心概念：
    - 10 处 stop-gradient 边界（CONCEPT.md 第 9.5 节）
    - critic_loss_twohot：同时拟合 λ-return 与 EMA critic 的输出（两个 two-hot 交叉熵）
    - actor_loss_reinforce：−sg(w) [ sg(A) log π(â|sg(m̂)) + η H(π) ]，梯度完全不穿过世界模型

前置知识：
    - 03、06、07、08 节

对应概念文档：
    - CONCEPT.md 第 9 节「Dreamer V3 的 actor 与 critic」

预计运行时间：约 4.5 分钟（CPU）
"""

import copy
import importlib.util
import math
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
from torch.distributions import Normal

from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_P3_DIR = os.path.abspath(os.path.join(_DIR, "..", "phase-03-rssm-planet"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod03 = _load(os.path.join(_DIR, "03-dreamer-v1-actor-critic.py"), "p6_03")
_mod06 = _load(os.path.join(_DIR, "06-symlog-and-twohot.py"), "p6_06")
_mod07 = _load(os.path.join(_DIR, "07-return-normalization-and-robustness.py"), "p6_07")
_mod08 = _load(os.path.join(_DIR, "08-dreamer-v3-world-model.py"), "p6_08")
_p3_05 = _load(os.path.join(_P3_DIR, "05-world-model-training.py"), "p3_05")

lambda_return = _mod03.lambda_return
TwoHotHead = _mod06.TwoHotHead
ReturnNormalizer = _mod07.ReturnNormalizer
ema_update = _mod07.ema_update
DreamerV3WorldModel = _mod08.DreamerV3WorldModel

render_pendulum = _p3_05.render_pendulum
step_with_repeat = _p3_05.step_with_repeat
preprocess_obs = _p3_05.preprocess_obs
EPISODE_DECISIONS = _p3_05.EPISODE_DECISIONS

plt = setup_matplotlib()
OUT = output_dir("phase-06")
SEED = 0
DEVICE = get_device()


# ==================== 第一部分：手算单元测试 ====================


def verify_section_9_hand_calc():
    """严格核对 CONCEPT.md 第 9.4 节离散动作 actor 梯度手算例子。"""
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 9.4 节 Actor 梯度手算例子")
    print("=" * 60)

    probs_target = torch.tensor([0.5, 0.3, 0.2])
    logits = torch.log(probs_target).clone().detach().requires_grad_(True)
    A = (12.0 - 10.0) / max(1.0, 4.0)  # 0.5
    eta = 3e-4

    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    ent = -(p * log_p).sum()

    # 1. 回报项梯度
    loss_rew = -A * log_p[1]
    loss_rew.backward(retain_graph=True)
    g_rew = logits.grad.detach().numpy().copy()

    # 2. 熵项梯度
    logits.grad.zero_()
    loss_ent = -eta * ent
    loss_ent.backward()
    g_ent = logits.grad.detach().numpy().copy()

    print(f"  优势 A = {A:.2f} ｜ 策略熵 H(π) = {ent.item():.4f}（手算 1.0297）")
    print(f"  回报项对 logits 梯度 = ({g_rew[0]:+.4f}, {g_rew[1]:+.4f}, {g_rew[2]:+.4f})  ｜  手算 (+0.25, -0.35, +0.10)")
    print(f"  熵项对 logits 梯度   = ({g_ent[0]:+.2e}, {g_ent[1]:+.2e}, {g_ent[2]:+.2e})  ｜  手算 (+5.0e-5, -1.6e-5, -3.5e-5)")
    assert abs(ent.item() - 1.0297) < 5e-4
    assert np.allclose(g_rew, [0.25, -0.35, 0.10], atol=1e-4)
    assert np.allclose(g_ent, [5.05e-5, -1.57e-5, -3.48e-5], atol=2e-6)
    print("  ✓ 与 CONCEPT.md 第 9.4 节完全一致！")


# ==================== 第二部分：Dreamer V3 Actor、Critic 与 Agent ====================


class DreamerV3Actor(nn.Module):
    """连续动作 Tanh-Normal 策略，同时支持 REINFORCE（log_prob + 熵）与 pathwise 重参数化采样。"""

    def __init__(self, feat_dim, act_dim, hidden=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.std_head = nn.Linear(hidden, act_dim)

    def dist_params(self, feat):
        h = self.net(feat)
        mean = torch.tanh(self.mean_head(h)) * 1.0
        # 把 std 约束在 [0.1, 1.0]（V3 论文：min_std=0.1, max_std=1.0）
        std = 0.1 + 0.9 * torch.sigmoid(self.std_head(h))
        return mean, std

    def sample_and_info(self, feat, deterministic=False):
        mean, std = self.dist_params(feat)
        if deterministic:
            return mean.clamp(-1.0, 1.0), None, None
        dist = Normal(mean, std)
        raw_a = dist.rsample()  # 保留路径梯度
        # 裁剪到 [-1, 1]（使用 straight-through 裁剪保持梯度在边界外平滑）
        act = raw_a.clamp(-1.0, 1.0)
        act_st = act + raw_a - raw_a.detach()
        log_prob = dist.log_prob(raw_a.detach()).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return act_st, log_prob, entropy

    def log_prob_and_entropy(self, feat, raw_actions):
        mean, std = self.dist_params(feat)
        dist = Normal(mean, std)
        return dist.log_prob(raw_actions).sum(dim=-1), dist.entropy().sum(dim=-1)


def critic_loss_twohot(critic, ema_critic, imag_states, returns, weights, use_ema_reg=True):
    """CONCEPT.md 第 9.2 节：两个 two-hot 交叉熵（拟合 λ-return + EMA 正则），全部 4 处 sg 到位。"""
    s_in = imag_states[:-1].detach()
    logits = critic(s_in)
    loss_ret = critic.loss(logits, returns[:-1].detach())
    if use_ema_reg:
        with torch.no_grad():
            ema_v = ema_critic.predict(ema_critic(s_in))
        loss_ema = critic.loss(logits, ema_v)
        per_step = loss_ret + loss_ema
    else:
        per_step = loss_ret
    return (weights[:-1].detach() * per_step).mean()


def actor_loss_reinforce(actor, imag_states, imag_actions, advantages, weights, eta=3e-4):
    """CONCEPT.md 第 9.3 节：REINFORCE actor 损失（状态、优势、权重全部 detach，梯度不穿过世界模型）。"""
    log_p, ent = actor.log_prob_and_entropy(imag_states[:-1].detach(), imag_actions.detach())
    per_step = -advantages.detach() * log_p - eta * ent
    return (weights[:-1].detach() * per_step).mean(), ent.mean().item()


class EpisodeReplay:
    """支持像素与向量双模式的按回合序列回放池，保存 (obs, act, rew, cont, is_first)。"""

    def __init__(self, obs_mode="pixel"):
        self.obs_mode = obs_mode
        self.episodes = []

    def add_episode(self, obs, actions, rewards, conts):
        obs_arr = np.asarray(obs, dtype=np.uint8 if self.obs_mode == "pixel" else np.float32)
        self.episodes.append((
            obs_arr,
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(conts, dtype=np.float32),
        ))

    def __len__(self):
        return len(self.episodes)

    def sample(self, batch_size, chunk_len, device, rng):
        obs_b, act_b, rew_b, cont_b, first_b = [], [], [], [], []
        for _ in range(batch_size):
            ep = self.episodes[int(rng.integers(0, len(self.episodes)))]
            o_ep, a_ep, r_ep, c_ep = ep
            L = len(o_ep)
            # 从下标 1 起采样，保证 (o_t, a_{t-1}, r_{t-1}, c_t) 严格对齐
            max_s = max(1, L - chunk_len)
            s = int(rng.integers(1, max_s + 1)) if max_s > 1 else 1
            e = min(L, s + chunk_len)
            o_sl, a_sl, r_sl, c_sl = o_ep[s:e], a_ep[s:e], r_ep[s:e], c_ep[s:e]
            f_sl = np.zeros(len(o_sl), dtype=np.float32)
            if s == 1:
                f_sl[0] = 1.0
            obs_b.append(o_sl)
            act_b.append(a_sl)
            rew_b.append(r_sl)
            cont_b.append(c_sl)
            first_b.append(f_sl)

        o_np = np.stack(obs_b)
        o_t = preprocess_obs(torch.from_numpy(o_np).to(device)) if self.obs_mode == "pixel" else torch.from_numpy(o_np).to(device)
        return (
            o_t,
            torch.from_numpy(np.stack(act_b)).to(device),
            torch.from_numpy(np.stack(rew_b)).to(device),
            torch.from_numpy(np.stack(cont_b)).to(device),
            torch.from_numpy(np.stack(first_b)).to(device),
        )


class DreamerV3Agent:
    """完整 Dreamer V3 智能体，支持像素/向量观测与 REINFORCE / Pathwise 两种 actor 模式。"""

    def __init__(
        self,
        act_dim,
        obs_mode="vector",
        obs_dim=None,
        deter_dim=128,
        n_stoch=8,
        n_classes=8,
        hidden=128,
        horizon=12,
        gamma=0.99,
        lam=0.95,
        eta=3e-4,
        actor_mode="reinforce",
        use_ret_norm=True,
        use_ema_reg=True,
        use_twohot_critic=True,
        beta_dyn=1.0,
        beta_rep=0.1,
        free_nats=1.0,
        detach_start=True,
        device=torch.device("cpu"),
    ):
        self.act_dim = act_dim
        self.obs_mode = obs_mode
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.eta = eta
        self.actor_mode = actor_mode
        self.use_ret_norm = use_ret_norm
        self.use_ema_reg = use_ema_reg
        self.use_twohot_critic = use_twohot_critic
        self.detach_start = detach_start
        self.device = device

        self.wm = DreamerV3WorldModel(
            act_dim=act_dim, obs_mode=obs_mode, obs_dim=obs_dim,
            deter_dim=deter_dim, n_stoch=n_stoch, n_classes=n_classes, hidden=hidden,
            beta_dyn=beta_dyn, beta_rep=beta_rep, free_nats=free_nats,
        ).to(device)
        feat_dim = self.wm.feat_dim
        self.actor = DreamerV3Actor(feat_dim, act_dim, hidden=hidden).to(device)
        if use_twohot_critic:
            self.critic = TwoHotHead(feat_dim, hidden=hidden).to(device)
        else:
            self.critic = nn.Sequential(
                nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                nn.Linear(hidden, 1),
            ).to(device)
        self.ema_critic = copy.deepcopy(self.critic)
        self.ret_norm = ReturnNormalizer(decay=0.99)

        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=1e-3)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def critic_predict(self, feat, use_ema=False):
        net = self.ema_critic if use_ema else self.critic
        if self.use_twohot_critic:
            return net.predict(net(feat))
        return net(feat).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_single, state, deterministic=False):
        """在线单步决策：obs_single 为单帧 [3,64,64] 或 [S]，state=(h, z, a_prev)。"""
        self.wm.eval()
        self.actor.eval()
        h, z, a_prev = state
        if self.obs_mode == "pixel":
            o_t = preprocess_obs(torch.from_numpy(obs_single[None]).to(self.device))
        else:
            o_t = torch.from_numpy(np.asarray(obs_single, dtype=np.float32)[None]).to(self.device)
        e_t = self.wm.encoder(o_t)
        h, z, _, _ = self.wm.rssm.observe_step(h, z, a_prev, e_t, sample=False)
        feat = torch.cat([h, z], dim=-1)
        act_t, _, _ = self.actor.sample_and_info(feat, deterministic=deterministic)
        act_np = act_t.squeeze(0).cpu().numpy().astype(np.float32)
        return act_np, (h, z, act_t), self.critic_predict(feat).item()

    def init_online_state(self):
        h, z = self.wm.rssm.initial_state(1, self.device)
        return (h, z, torch.zeros(1, self.act_dim, device=self.device))

    def train_step(self, batch):
        """CONCEPT.md 第 9.6 节完整单步更新：世界模型 → 批量想象 → Actor → Critic + EMA。"""
        obs, act, rew, cont, is_first = batch
        self.wm.train()
        self.actor.train()
        self.critic.train()

        # 1. 世界模型更新
        wm_loss, obs_out, wm_m = self.wm.loss(obs, act, rew, cont, is_first=is_first)
        if self.detach_start:
            self.wm_opt.zero_grad()
            wm_loss.backward()
            nn.utils.clip_grad_norm_(self.wm.parameters(), 100.0)
            self.wm_opt.step()

        # 2. 批量想象（起点默认 detach；若做"不停止梯度"错误消融，则保留连接以观察自欺症状）
        if self.detach_start:
            h_cur = obs_out["h"].detach().reshape(-1, self.wm.rssm.deter_dim)
            z_cur = obs_out["z"].detach().reshape(-1, self.wm.rssm.stoch_dim)
        else:
            h_cur = obs_out["h"].reshape(-1, self.wm.rssm.deter_dim)
            z_cur = obs_out["z"].reshape(-1, self.wm.rssm.stoch_dim)
        c0 = cont.detach().reshape(-1)

        if self.actor_mode == "pathwise" and self.detach_start:
            for p in list(self.wm.parameters()) + list(self.critic.parameters()):
                p.requires_grad_(False)

        states = [torch.cat([h_cur, z_cur], dim=-1)]
        actions = []
        ents = []
        for _ in range(self.horizon):
            if self.actor_mode == "reinforce":
                with torch.no_grad():
                    a_k, _, _ = self.actor.sample_and_info(states[-1], deterministic=False)
                    h_cur, z_cur, _ = self.wm.rssm.imagine_step(h_cur, z_cur, a_k, sample=True)
                states.append(torch.cat([h_cur, z_cur], dim=-1))
                actions.append(a_k)
            else:
                a_k, _, ent_k = self.actor.sample_and_info(states[-1], deterministic=False)
                h_cur, z_cur, _ = self.wm.rssm.imagine_step(h_cur, z_cur, a_k, sample=True)
                states.append(torch.cat([h_cur, z_cur], dim=-1))
                actions.append(a_k)
                ents.append(ent_k)

        imag_states = torch.stack(states, dim=0)    # [H+1, N, D]
        imag_actions = torch.stack(actions, dim=0)  # [H, N, A]

        if self.actor_mode == "reinforce":
            with torch.no_grad():
                imag_rewards = self.wm.reward_head.predict(self.wm.reward_head(imag_states[1:]))
                imag_conts = torch.sigmoid(self.wm.cont_head(imag_states[1:]).squeeze(-1))
                imag_values = self.critic_predict(imag_states)
                w_list = [c0]
                for k in range(self.horizon):
                    w_list.append(w_list[-1] * imag_conts[k])
                weights = torch.stack(w_list, dim=0)
                rets = lambda_return(imag_rewards, imag_values, imag_conts, gamma=self.gamma, lam=self.lam)
        else:
            imag_rewards = self.wm.reward_head.predict(self.wm.reward_head(imag_states[1:]))
            imag_conts = torch.sigmoid(self.wm.cont_head(imag_states[1:]).squeeze(-1)).detach()
            imag_values = self.critic_predict(imag_states)
            w_list = [c0]
            for k in range(self.horizon):
                w_list.append(w_list[-1] * imag_conts[k])
            weights = torch.stack(w_list, dim=0).detach()
            rets = lambda_return(imag_rewards, imag_values, imag_conts, gamma=self.gamma, lam=self.lam)

        scale = self.ret_norm.update_and_scale(rets[:-1]) if self.use_ret_norm else 1.0

        # 3. 更新 Actor
        if self.actor_mode == "reinforce":
            adv = (rets[:-1] - imag_values[:-1]) / scale
            a_loss, ent_val = actor_loss_reinforce(self.actor, imag_states, imag_actions, adv, weights, eta=self.eta)
            self.actor_opt.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
            self.actor_opt.step()
            if not self.detach_start:
                self.wm_opt.zero_grad()
                wm_loss.backward()
                nn.utils.clip_grad_norm_(self.wm.parameters(), 100.0)
                self.wm_opt.step()
        else:
            ent_tensor = torch.stack(ents, dim=0)
            a_loss = -(weights[:-1].detach() * (rets[:-1] / scale + self.eta * ent_tensor)).mean()
            ent_val = ent_tensor.detach().mean().item()
            self.actor_opt.zero_grad()
            if not self.detach_start:
                self.wm_opt.zero_grad()
                (wm_loss + a_loss).backward()
                nn.utils.clip_grad_norm_(self.wm.parameters(), 100.0)
                nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
                self.wm_opt.step()
                self.actor_opt.step()
            else:
                a_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
                self.actor_opt.step()
                for p in list(self.wm.parameters()) + list(self.critic.parameters()):
                    p.requires_grad_(True)

        # 4. 更新 Critic + EMA
        if self.use_twohot_critic:
            c_loss = critic_loss_twohot(self.critic, self.ema_critic, imag_states, rets, weights, use_ema_reg=self.use_ema_reg)
        else:
            v_p = self.critic_predict(imag_states[:-1].detach())
            c_loss = 0.5 * (weights[:-1].detach() * (v_p - rets[:-1].detach()).pow(2)).mean()
            if self.use_ema_reg:
                with torch.no_grad():
                    v_e = self.critic_predict(imag_states[:-1].detach(), use_ema=True)
                c_loss = c_loss + 0.5 * (weights[:-1].detach() * (v_p - v_e).pow(2)).mean()

        self.critic_opt.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()
        ema_update(self.critic, self.ema_critic, decay=0.98)

        wm_m["actor_loss"] = a_loss.item()
        wm_m["critic_loss"] = c_loss.item()
        wm_m["entropy"] = ent_val
        wm_m["ret_scale"] = scale
        return wm_m


# ==================== 第三部分：向量 REINFORCE vs Pathwise 对比 & 像素 Dreamer V3 ====================


def collect_vector_episode(env_id, seed, agent=None, deterministic=False, reward_scale=1.0, shift_reward_bug=False):
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])
    rng = np.random.default_rng(seed + 77)

    o_list = [obs.astype(np.float32)]
    a_list = [np.zeros(act_dim, dtype=np.float32)]
    r_list = [0.0]
    c_list = [1.0]
    raw_ret = 0.0
    state = agent.init_online_state() if agent is not None else None

    while True:
        if agent is None:
            a = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            a, state, _ = agent.act(obs, state, deterministic=deterministic)
        nxt, r, term, trunc, _ = env.step(a * act_scale)
        raw_ret += float(r)
        o_list.append(nxt.astype(np.float32))
        a_list.append(a)
        r_list.append(float(r) * reward_scale)
        c_list.append(0.0 if term else 1.0)
        obs = nxt
        if term or trunc:
            break
    env.close()

    r_arr = np.asarray(r_list, dtype=np.float32)
    if shift_reward_bug and len(r_arr) > 2:
        # 故意把奖励错开一位（CONCEPT.md 第 10.5 节经典 bug 复现）
        r_arr = np.roll(r_arr, -1)
    return dict(obs=np.asarray(o_list), actions=np.asarray(a_list), rewards=r_arr, conts=np.asarray(c_list), ret=raw_ret)


def run_vector_actor_comparison():
    print("\n" + "=" * 60)
    print("第二部分：连续控制 Pendulum-v1 — REINFORCE vs 沿模型反传（Pathwise）对比")
    print("=" * 60)

    quick = is_quick()
    n_init = 3 if quick else 5
    n_online = 4 if quick else 15
    upd_per_ep = 15 if quick else 40
    eval_seeds = [900, 901] if quick else [900, 901, 902, 903, 904]
    curves = {}

    for mode, label in [("reinforce", "Dreamer V3 (REINFORCE)"), ("pathwise", "Dreamer V3 (沿模型反传 Pathwise)")]:
        seed_everything(SEED)
        rng = np.random.default_rng(SEED)
        replay = EpisodeReplay(obs_mode="vector")
        for i in range(n_init):
            ep = collect_vector_episode("Pendulum-v1", SEED + i)
            replay.add_episode(ep["obs"], ep["actions"], ep["rewards"], ep["conts"])

        agent = DreamerV3Agent(act_dim=1, obs_mode="vector", obs_dim=3, actor_mode=mode, device=DEVICE)
        steps_l, rets_l = [], []
        t0 = time.time()

        for r_idx in range(1, n_online + 1):
            n_u = upd_per_ep * 2 if r_idx == 1 else upd_per_ep
            for _ in range(n_u):
                batch = replay.sample(16, 20, DEVICE, rng)
                agent.train_step(batch)
            ep = collect_vector_episode("Pendulum-v1", SEED + 100 + r_idx, agent=agent, deterministic=False)
            replay.add_episode(ep["obs"], ep["actions"], ep["rewards"], ep["conts"])
            if r_idx == 1 or r_idx % 3 == 0 or r_idx == n_online:
                evs = [collect_vector_episode("Pendulum-v1", s, agent=agent, deterministic=True)["ret"] for s in eval_seeds]
                steps_l.append(len(replay) * 200)
                rets_l.append(float(np.mean(evs)))
        curves[label] = (np.array(steps_l), np.array(rets_l))
        print(f"  {label:<34} → 最终评估回报 {rets_l[-1]:7.1f}（最高 {max(rets_l):7.1f}，用时 {time.time() - t0:.1f}s）")

    return curves


def run_pixel_dreamer_v3():
    print("\n" + "=" * 60)
    print("第三部分：像素版 Pendulum-v1 完整 Dreamer V3 在线闭环")
    print("=" * 60)
    quick = is_quick()
    n_init = 2 if quick else 5
    n_online = 2 if quick else 8
    upd_init = 12 if quick else 80
    upd_per_ep = 8 if quick else 35
    eval_seeds = [1000] if quick else [1000, 1001, 1002, 1003, 1004]

    def collect_pix(seed, agent=None, det=False):
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
        env.reset(seed=seed)
        rng = np.random.default_rng(seed + 55)
        L = EPISODE_DECISIONS + 1
        obs_ep = np.zeros((L, 64, 64, 3), dtype=np.uint8)
        act_ep = np.zeros((L, 1), dtype=np.float32)
        rew_ep = np.zeros(L, dtype=np.float32)
        cont_ep = np.ones(L, dtype=np.float32)
        obs_ep[0] = render_pendulum(env)
        state = agent.init_online_state() if agent is not None else None
        for t in range(1, L):
            if agent is None:
                a = rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
            else:
                a, state, _ = agent.act(obs_ep[t - 1], state, deterministic=det)
            frame, r, _ = step_with_repeat(env, a)
            obs_ep[t], act_ep[t], rew_ep[t] = frame, a, r
        env.close()
        return dict(obs=obs_ep, actions=act_ep, rewards=rew_ep, conts=cont_ep, ret=float(rew_ep.sum()))

    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    replay = EpisodeReplay(obs_mode="pixel")
    for i in range(n_init):
        ep = collect_pix(SEED + i)
        replay.add_episode(ep["obs"], ep["actions"], ep["rewards"], ep["conts"])

    agent = DreamerV3Agent(
        act_dim=1, obs_mode="pixel", deter_dim=192, n_stoch=16, n_classes=16, hidden=192,
        horizon=10 if quick else 15, actor_mode="reinforce", device=DEVICE,
    )
    ev_steps, ev_rets = [], []
    t0 = time.time()
    for r_idx in range(1, n_online + 1):
        n_u = upd_init if r_idx == 1 else upd_per_ep
        for _ in range(n_u):
            batch = replay.sample(4 if quick else 8, 12 if quick else 20, DEVICE, rng)
            m = agent.train_step(batch)
        ep = collect_pix(SEED + 300 + r_idx, agent=agent, det=False)
        replay.add_episode(ep["obs"], ep["actions"], ep["rewards"], ep["conts"])
        if r_idx == 1 or r_idx % 2 == 0 or r_idx == n_online:
            evs = [collect_pix(s, agent=agent, det=True)["ret"] for s in eval_seeds]
            ev_m = float(np.mean(evs))
            ev_steps.append(len(replay) * 200)
            ev_rets.append(ev_m)
            print(f"  第 {r_idx:>2}/{n_online} 轮（累计 {ev_steps[-1]:>4} 步）：重建 {m['recon']:5.1f} ｜ "
                  f"KL {m['raw_kl']:4.1f} ｜ 熵 {m['entropy']:.2f} ｜ 评估回报 {ev_m:7.1f}（{time.time() - t0:.0f}s）")

    return np.array(ev_steps), np.array(ev_rets)


def main():
    seed_everything(SEED)
    if DEVICE.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    verify_section_9_hand_calc()
    vec_curves = run_vector_actor_comparison()
    pix_steps, pix_rets = run_pixel_dreamer_v3()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
    for lbl, (st, rt) in vec_curves.items():
        axes[0].plot(st, rt, "o-", linewidth=2.2, label=f"{lbl} ({rt[-1]:.0f})")
    axes[0].axhline(-200.0, color="gray", linestyle=":", label="优秀参考（−200）")
    axes[0].set_xlabel("真实环境交互步数")
    axes[0].set_ylabel("评估回报")
    axes[0].set_title("向量 Pendulum-v1：REINFORCE vs 沿模型反传")
    axes[0].legend(fontsize=8.5)

    axes[1].plot(pix_steps, pix_rets, "s-", color="C2", linewidth=2.2, label=f"Dreamer V3 像素版 ({pix_rets[-1]:.0f})")
    axes[1].axhline(-1150.0, color="tomato", linestyle=":", label="随机策略参考")
    axes[1].set_xlabel("真实环境交互步数")
    axes[1].set_ylabel("评估回报")
    axes[1].set_title("像素版 Pendulum-v1：Dreamer V3 学习曲线")
    axes[1].legend(fontsize=8.5)
    save_figure(fig, OUT / "09-dreamer-v3-learning-curves.png")

    print("\n" + "=" * 60)
    print("本节完成！下一节：10-diagnostics-and-ablations.py（Dreamer 诊断工具与系统消融实验）")
    print("=" * 60)


if __name__ == "__main__":
    main()
