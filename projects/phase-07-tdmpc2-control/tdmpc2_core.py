"""TD-MPC2 独立可复用核心模块（Phase 7 实战项目，接入 Phase 5–7 统一连续控制排行榜）。"""

import copy
import math
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from utils import seed_everything
from utils.planning import shift_plan


def td2_simnorm(x, v_dim=8):
    s = x.shape
    return F.softmax(x.reshape(*s[:-1], -1, v_dim), dim=-1).reshape(*s)


def td2_symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def td2_symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


def td2_two_hot(x_scalar, bins):
    z = td2_symlog(x_scalar).clamp(bins[0], bins[-1])
    idx = (torch.searchsorted(bins, z, right=True) - 1).clamp(0, len(bins) - 2)
    b0, b1 = bins[idx], bins[idx + 1]
    w1 = (z - b0) / (b1 - b0).clamp_min(1e-8)
    w0 = 1.0 - w1
    out = torch.zeros(*x_scalar.shape, len(bins), device=x_scalar.device, dtype=x_scalar.dtype)
    out.scatter_(-1, idx.unsqueeze(-1), w0.unsqueeze(-1))
    out.scatter_(-1, (idx + 1).unsqueeze(-1), w1.unsqueeze(-1))
    return out


def td2_two_hot_inv(logits, bins):
    return td2_symexp((F.softmax(logits, dim=-1) * bins).sum(dim=-1))


def _block(d_in, hidden, d_out):
    return nn.Sequential(
        nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.Mish(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish(),
        nn.Linear(hidden, d_out),
    )


class MiniTDMPC2(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim=64, hidden=128, n_q=5, n_bins=101, simnorm_dim=8):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.simnorm_dim = simnorm_dim
        self.n_q = n_q

        self.enc = _block(obs_dim, hidden, latent_dim)
        self.dyn = _block(latent_dim + act_dim, hidden, latent_dim)
        self.rew = _block(latent_dim + act_dim, hidden, n_bins)
        nn.init.zeros_(self.rew[-1].weight)
        nn.init.zeros_(self.rew[-1].bias)
        self.qs = nn.ModuleList([_block(latent_dim + act_dim, hidden, n_bins) for _ in range(n_q)])
        for q in self.qs:
            nn.init.zeros_(q[-1].weight)
            nn.init.zeros_(q[-1].bias)
        self.qs_targ = copy.deepcopy(self.qs)

        self.pi_trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish(),
        )
        self.pi_mean = nn.Linear(hidden, act_dim)
        self.pi_log_std = nn.Linear(hidden, act_dim)
        self.register_buffer("bins", torch.linspace(-10.0, 10.0, n_bins))
        self.q_scale = 1.0
        self.q_scale_init = False

    def encode(self, obs):
        return td2_simnorm(self.enc(obs), self.simnorm_dim)

    def next_z(self, z, a):
        return td2_simnorm(self.dyn(torch.cat([z, a], dim=-1)), self.simnorm_dim)

    def pred_r(self, z, a):
        return td2_two_hot_inv(self.rew(torch.cat([z, a], dim=-1)), self.bins)

    def q_values(self, z, a, target=False):
        za = torch.cat([z, a], dim=-1)
        net_list = self.qs_targ if target else self.qs
        logits = torch.stack([q(za) for q in net_list], dim=0)
        return td2_two_hot_inv(logits, self.bins), logits

    def sample_pi(self, z, deterministic=False):
        h = self.pi_trunk(z)
        mu = self.pi_mean(h)
        if deterministic:
            return torch.tanh(mu), None
        std = self.pi_log_std(h).clamp(-4.0, 1.0).exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        lp = (Normal(mu, std).log_prob(u) - 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(dim=-1)
        return a, lp

    @torch.no_grad()
    def plan(self, obs_np, prev_mean=None, horizon=3, n_samples=96, n_pi=12, n_elites=20,
             n_iters=4, lam=2.0, gamma=0.99, deterministic=False):
        z0 = self.encode(torch.from_numpy(np.asarray(obs_np, dtype=np.float32)[None]))
        z_p = z0.expand(n_pi, -1)
        pi_acts = []
        for _ in range(horizon):
            ap, _ = self.sample_pi(z_p, deterministic=False)
            pi_acts.append(ap)
            z_p = self.next_z(z_p, ap)
        pi_cand = torch.stack(pi_acts, dim=1)

        mean = shift_plan(prev_mean) if (prev_mean is not None and prev_mean.shape[0] == horizon) else torch.zeros(horizon, self.act_dim)
        std = torch.full((horizon, self.act_dim), 0.7)

        for _ in range(n_iters):
            gauss = (mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(n_samples, horizon, self.act_dim)).clamp(-1.0, 1.0)
            cand = torch.cat([gauss, pi_cand], dim=0)
            N = cand.shape[0]
            z = z0.expand(N, -1)
            ret = torch.zeros(N)
            disc = 1.0
            for k in range(horizon):
                ak = cand[:, k]
                ret = ret + disc * self.pred_r(z, ak)
                z = self.next_z(z, ak)
                disc *= gamma
            a_end, _ = self.sample_pi(z, deterministic=True)
            qv, _ = self.q_values(z, a_end, target=False)
            ret = ret + disc * qv.mean(dim=0)

            top_v, top_i = ret.topk(min(n_elites, N))
            elites = cand[top_i]
            w = torch.softmax((top_v - top_v.max()) / lam, dim=0).view(-1, 1, 1)
            mean = (w * elites).sum(dim=0)
            std = torch.sqrt((w * (elites - mean.unsqueeze(0)).pow(2)).sum(dim=0)).clamp(0.05, 1.0)

        a0 = mean[0] if deterministic else (mean[0] + std[0] * torch.randn(self.act_dim)).clamp(-1.0, 1.0)
        return a0.numpy(), mean


def run_tdmpc2_on_benchmark(task_name, cfg, seed=0):
    """按 Phase 5 统一排行榜协议运行精简版 TD-MPC2，返回 (eval_steps, eval_rets, wall_time, decision_ms)。"""
    t0 = time.time()
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(task_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    agent = MiniTDMPC2(obs_dim, act_dim, latent_dim=64, hidden=128, n_q=5, n_bins=101)
    wm_params = list(agent.enc.parameters()) + list(agent.dyn.parameters()) + list(agent.rew.parameters()) + list(agent.qs.parameters())
    pi_params = list(agent.pi_trunk.parameters()) + list(agent.pi_mean.parameters()) + list(agent.pi_log_std.parameters())
    wm_opt = torch.optim.Adam(wm_params, lr=2e-3)
    pi_opt = torch.optim.Adam(pi_params, lr=1e-3)

    episodes = []
    horizon = 3
    rho = 0.5
    gamma = 0.99
    init_random = min(500, cfg["total_steps"] // 3)
    train_every = 20
    updates_per_call = 10
    plan_repeat = 1 if task_name == "Pendulum-v1" else 2

    obs, _ = env.reset(seed=seed)
    ep_o, ep_a, ep_r, ep_term = [obs.astype(np.float32)], [], [], []
    prev_mean = None
    cached_a = np.zeros(act_dim, dtype=np.float32)
    eval_steps, eval_rets, dec_ms = [], [], []

    for step in range(1, cfg["total_steps"] + 1):
        if step <= init_random:
            a = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            if (step - 1) % plan_repeat == 0:
                td0 = time.time()
                cached_a, prev_mean = agent.plan(obs, prev_mean=prev_mean, horizon=horizon, deterministic=False)
                dec_ms.append((time.time() - td0) * 1000.0)
            a = cached_a

        nxt, r, term, trunc, _ = env.step(a * act_scale)
        ep_o.append(nxt.astype(np.float32))
        ep_a.append(a)
        ep_r.append(float(r))
        ep_term.append(float(term))
        obs = nxt

        if term or trunc:
            if len(ep_r) >= horizon:
                episodes.append((np.asarray(ep_o), np.asarray(ep_a), np.asarray(ep_r, dtype=np.float32), np.asarray(ep_term, dtype=np.float32)))
            obs, _ = env.reset()
            ep_o, ep_a, ep_r, ep_term = [obs.astype(np.float32)], [], [], []
            prev_mean = None

        if step >= init_random and step % train_every == 0 and len(episodes) > 0:
            for _ in range(updates_per_call):
                ob, ab, rb, tb = [], [], [], []
                for _ in range(64):
                    oe, ae, re, te = episodes[int(rng.integers(0, len(episodes)))]
                    s = int(rng.integers(0, len(re) - horizon + 1))
                    ob.append(oe[s : s + horizon + 1])
                    ab.append(ae[s : s + horizon])
                    rb.append(re[s : s + horizon])
                    tb.append(te[s : s + horizon])
                obs_seq = torch.from_numpy(np.stack(ob))
                act_seq = torch.from_numpy(np.stack(ab))
                rew_seq = torch.from_numpy(np.stack(rb))
                term_seq = torch.from_numpy(np.stack(tb))

                z = agent.encode(obs_seq[:, 0])
                zs_det = []
                wm_loss = 0.0
                for i in range(horizon):
                    zs_det.append(z.detach())
                    ai, ri, ti, o_nxt = act_seq[:, i], rew_seq[:, i], term_seq[:, i], obs_seq[:, i + 1]
                    with torch.no_grad():
                        z_tgt = agent.encode(o_nxt)
                        a_nxt, _ = agent.sample_pi(z_tgt, deterministic=False)
                        qv_t, _ = agent.q_values(z_tgt, a_nxt, target=True)
                        perm = torch.randperm(agent.n_q)[:2]
                        y_i = ri + gamma * (1.0 - ti) * torch.min(qv_t[perm[0]], qv_t[perm[1]])

                    r_logits = agent.rew(torch.cat([z, ai], dim=-1))
                    _, q_logits = agent.q_values(z, ai, target=False)
                    z = agent.next_z(z, ai)

                    l_z = F.mse_loss(z, z_tgt)
                    l_r = -(td2_two_hot(ri, agent.bins) * F.log_softmax(r_logits, dim=-1)).sum(dim=-1).mean()
                    y_th = td2_two_hot(y_i, agent.bins).unsqueeze(0).expand(agent.n_q, -1, -1)
                    l_q = -(y_th * F.log_softmax(q_logits, dim=-1)).sum(dim=-1).mean()
                    wm_loss = wm_loss + (rho**i) * (2.0 * l_z + 0.5 * l_r + 0.5 * l_q)

                wm_opt.zero_grad()
                wm_loss.backward()
                nn.utils.clip_grad_norm_(wm_params, 20.0)
                wm_opt.step()

                for p in agent.qs.parameters():
                    p.requires_grad_(False)
                pi_loss = 0.0
                for i, zi in enumerate(zs_det):
                    a_p, lp = agent.sample_pi(zi, deterministic=False)
                    qv, _ = agent.q_values(zi, a_p, target=False)
                    qm = qv.mean(dim=0)
                    with torch.no_grad():
                        sp = (torch.quantile(qm, 0.95) - torch.quantile(qm, 0.05)).item()
                        agent.q_scale = sp if not agent.q_scale_init else 0.99 * agent.q_scale + 0.01 * sp
                        agent.q_scale_init = True
                    pi_loss = pi_loss - (rho**i) * (qm / max(1.0, agent.q_scale) - 1e-3 * lp).mean()
                pi_opt.zero_grad()
                pi_loss.backward()
                nn.utils.clip_grad_norm_(pi_params, 20.0)
                pi_opt.step()
                for p in agent.qs.parameters():
                    p.requires_grad_(True)

                with torch.no_grad():
                    for p, pt in zip(agent.qs.parameters(), agent.qs_targ.parameters()):
                        pt.data.mul_(0.99).add_(p.data, alpha=0.01)

        if step % cfg["eval_every"] == 0 or step == cfg["total_steps"]:
            ev_env = gym.make(task_name)
            scores = []
            for ei in range(cfg["eval_episodes"]):
                eo, _ = ev_env.reset(seed=900 + ei)
                pm = None
                tot = 0.0
                k_ev = 0
                ev_a = np.zeros(act_dim, dtype=np.float32)
                while True:
                    if k_ev % plan_repeat == 0:
                        ev_a, pm = agent.plan(eo, prev_mean=pm, horizon=horizon, deterministic=True)
                    k_ev += 1
                    eo, er, eterm, etrunc, _ = ev_env.step(ev_a * act_scale)
                    tot += float(er)
                    if eterm or etrunc:
                        break
                scores.append(tot)
            ev_env.close()
            eval_steps.append(step)
            eval_rets.append(float(np.mean(scores)))

    env.close()
    return eval_steps, eval_rets, time.time() - t0, float(np.mean(dec_ms)) if dec_ms else 0.0
