"""Mini-Dreamer（Phase 6 实战项目核心模块）。

将 Phase 6 第 04–09 节的 Dreamer V3 全部部件整理为独立可复用的模块：
- CategoricalRSSM（unimix 1% + straight-through）
- Symlog MLP / CNN 编解码器
- TwoHotHead（255 桶指数间距 + 零初始化输出层）
- ReturnNormalizer（P95−P5 EMA + max(1, S)）
- MiniDreamerV3Agent（支持 REINFORCE 与 Pathwise 两种 actor 模式 + EMA Critic）
"""

import copy
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from benchmark_protocol import evaluate_policy_fn
from utils import seed_everything


def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


def unimix_probs(logits, eps=0.01):
    p = F.softmax(logits, dim=-1)
    return (1.0 - eps) * p + eps / logits.shape[-1] if eps > 0 else p


def sample_categorical_st(logits, unimix=0.01):
    probs = unimix_probs(logits, eps=unimix)
    idx = torch.distributions.Categorical(probs=probs).sample()
    oh = F.one_hot(idx, num_classes=logits.shape[-1]).to(probs.dtype)
    return oh + probs - probs.detach()


def twohot_encode(y, bins):
    y_c = y.clamp(bins[0], bins[-1])
    idx = (torch.searchsorted(bins, y_c, right=True) - 1).clamp(0, len(bins) - 2)
    b0, b1 = bins[idx], bins[idx + 1]
    w1 = (y_c - b0) / (b1 - b0).clamp_min(1e-8)
    w0 = 1.0 - w1
    out = torch.zeros(*y.shape, len(bins), device=y.device, dtype=y.dtype)
    out.scatter_(-1, idx.unsqueeze(-1), w0.unsqueeze(-1))
    out.scatter_(-1, (idx + 1).unsqueeze(-1), w1.unsqueeze(-1))
    return out


class MiniTwoHotHead(nn.Module):
    def __init__(self, in_dim, hidden=128, n_bins=255):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.out = nn.Linear(hidden, n_bins)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("bins", symexp(torch.linspace(-20.0, 20.0, n_bins)))

    def forward(self, x):
        return self.out(self.net(x))

    def predict(self, logits):
        probs = F.softmax(logits, dim=-1)
        n = probs.shape[-1]
        half = n // 2
        if n % 2 == 1:
            return (probs[..., :half] * self.bins[:half] + probs[..., half + 1 :].flip(-1) * self.bins[half + 1 :].flip(-1)).sum(dim=-1) + probs[..., half] * self.bins[half]
        return (probs * self.bins).sum(dim=-1)

    def loss(self, logits, target):
        th = twohot_encode(target.detach(), self.bins)
        return -(th * F.log_softmax(logits, dim=-1)).sum(dim=-1)


class MiniCategoricalRSSM(nn.Module):
    def __init__(self, embed_dim, act_dim, deter_dim=128, n_stoch=8, n_classes=8, hidden=128, unimix=0.01):
        super().__init__()
        self.deter_dim = deter_dim
        self.n_stoch = n_stoch
        self.n_classes = n_classes
        self.stoch_dim = n_stoch * n_classes
        self.feat_dim = deter_dim + self.stoch_dim
        self.unimix = unimix

        self.in_proj = nn.Sequential(nn.Linear(self.stoch_dim + act_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru = nn.GRUCell(hidden, deter_dim)
        self.prior_net = nn.Sequential(nn.Linear(deter_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, self.stoch_dim))
        self.post_net = nn.Sequential(nn.Linear(deter_dim + embed_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, self.stoch_dim))

    def initial_state(self, batch_size, device):
        return torch.zeros(batch_size, self.deter_dim, device=device), torch.zeros(batch_size, self.stoch_dim, device=device)

    def _sample(self, logits, sample=True):
        if sample:
            z = sample_categorical_st(logits, self.unimix)
        else:
            idx = F.softmax(logits, dim=-1).argmax(dim=-1)
            z = F.one_hot(idx, num_classes=self.n_classes).to(logits.dtype)
        return z.reshape(*logits.shape[:-2], self.stoch_dim)

    def imagine_step(self, h, z, act, sample=True):
        h = self.gru(self.in_proj(torch.cat([z, act], dim=-1)), h)
        pr = self.prior_net(h).reshape(-1, self.n_stoch, self.n_classes)
        return h, self._sample(pr, sample=sample), pr

    def observe_step(self, h, z, act, embed, sample=True):
        h = self.gru(self.in_proj(torch.cat([z, act], dim=-1)), h)
        pr = self.prior_net(h).reshape(-1, self.n_stoch, self.n_classes)
        po = self.post_net(torch.cat([h, embed], dim=-1)).reshape(-1, self.n_stoch, self.n_classes)
        return h, self._sample(po, sample=sample), po, pr

    def observe(self, embeds, actions, is_first=None, sample=True):
        B, T, _ = embeds.shape
        h, z = self.initial_state(B, embeds.device)
        hs, zs, pos, prs = [], [], [], []
        for t in range(T):
            if is_first is not None:
                keep = (1.0 - is_first[:, t].float()).unsqueeze(-1)
                h, z, a_t = h * keep, z * keep, actions[:, t] * keep
            else:
                a_t = actions[:, t]
            h, z, po, pr = self.observe_step(h, z, a_t, embeds[:, t], sample=sample)
            hs.append(h)
            zs.append(z)
            pos.append(po)
            prs.append(pr)
        hs, zs = torch.stack(hs, dim=1), torch.stack(zs, dim=1)
        return dict(h=hs, z=zs, feat=torch.cat([hs, zs], dim=-1), post=torch.stack(pos, dim=1), prior=torch.stack(prs, dim=1))


class MiniDreamerWorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, deter_dim=128, n_stoch=8, n_classes=8, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.rssm = MiniCategoricalRSSM(hidden, act_dim, deter_dim=deter_dim, n_stoch=n_stoch, n_classes=n_classes, hidden=hidden)
        self.feat_dim = self.rssm.feat_dim
        self.decoder = nn.Sequential(nn.Linear(self.feat_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, obs_dim))
        self.reward_head = MiniTwoHotHead(self.feat_dim, hidden=hidden)
        self.cont_head = nn.Sequential(nn.Linear(self.feat_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def _cat_kl(self, lq, lp):
        q = unimix_probs(lq, 0.01)
        p = unimix_probs(lp, 0.01)
        return (q * (torch.log(q.clamp_min(1e-8)) - torch.log(p.clamp_min(1e-8)))).sum(dim=(-1, -2))

    def loss(self, obs, act, rew, cont, is_first):
        emb = self.encoder(symlog(obs))
        out = self.rssm.observe(emb, act, is_first=is_first, sample=True)
        feat = out["feat"]
        rec_l = 0.5 * (self.decoder(feat) - symlog(obs)).pow(2).sum(dim=-1).mean()
        rew_l = self.reward_head.loss(self.reward_head(feat), rew).mean()
        cont_l = F.binary_cross_entropy_with_logits(self.cont_head(feat).squeeze(-1), cont).mean()
        kl_dyn = self._cat_kl(out["post"].detach(), out["prior"]).clamp_min(1.0).mean()
        kl_rep = self._cat_kl(out["post"], out["prior"].detach()).clamp_min(1.0).mean()
        total = rec_l + rew_l + cont_l + 1.0 * kl_dyn + 0.1 * kl_rep
        return total, out


class MiniDreamerActor(nn.Module):
    def __init__(self, feat_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.mean_h = nn.Linear(hidden, act_dim)
        self.std_h = nn.Linear(hidden, act_dim)

    def dist(self, feat):
        h = self.net(feat)
        mean = torch.tanh(self.mean_h(h))
        std = 0.1 + 0.9 * torch.sigmoid(self.std_h(h))
        return Normal(mean, std)


class MiniDreamerV3Agent:
    def __init__(self, obs_dim, act_dim, horizon=12, gamma=0.99, lam=0.95, eta=3e-4, hidden=128, device=torch.device("cpu")):
        self.act_dim = act_dim
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.eta = eta
        self.device = device

        self.wm = MiniDreamerWorldModel(obs_dim, act_dim, deter_dim=hidden, n_stoch=8, n_classes=8, hidden=hidden).to(device)
        self.actor = MiniDreamerActor(self.wm.feat_dim, act_dim, hidden=hidden).to(device)
        self.critic = MiniTwoHotHead(self.wm.feat_dim, hidden=hidden).to(device)
        self.ema_critic = copy.deepcopy(self.critic)
        self.wm_opt = torch.optim.Adam(self.wm.parameters(), lr=1e-3)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.ret_scale = 0.0
        self.scale_init = False

    def init_state(self):
        h, z = self.wm.rssm.initial_state(1, self.device)
        return h, z, torch.zeros(1, self.act_dim, device=self.device)

    @torch.no_grad()
    def act(self, obs_np, state, deterministic=False):
        h, z, a_prev = state
        o = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)[None]).to(self.device)
        h, z, _, _ = self.wm.rssm.observe_step(h, z, a_prev, self.wm.encoder(symlog(o)), sample=False)
        d = self.actor.dist(torch.cat([h, z], dim=-1))
        a = (d.mean if deterministic else d.sample()).clamp(-1.0, 1.0)
        return a.squeeze(0).cpu().numpy(), (h, z, a)

    def train_step(self, batch):
        obs, act, rew, cont, is_first = batch
        wm_loss, obs_out = self.wm.loss(obs, act, rew, cont, is_first)
        self.wm_opt.zero_grad()
        wm_loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 100.0)
        self.wm_opt.step()

        h = obs_out["h"].detach().reshape(-1, self.wm.rssm.deter_dim)
        z = obs_out["z"].detach().reshape(-1, self.wm.rssm.stoch_dim)
        c0 = cont.detach().reshape(-1)

        states = [torch.cat([h, z], dim=-1)]
        actions = []
        with torch.no_grad():
            for _ in range(self.horizon):
                a_k = self.actor.dist(states[-1]).sample().clamp(-1.0, 1.0)
                h, z, _ = self.wm.rssm.imagine_step(h, z, a_k, sample=True)
                states.append(torch.cat([h, z], dim=-1))
                actions.append(a_k)
            imag_s = torch.stack(states, dim=0)
            imag_a = torch.stack(actions, dim=0)
            imag_r = self.wm.reward_head.predict(self.wm.reward_head(imag_s[1:]))
            imag_c = torch.sigmoid(self.wm.cont_head(imag_s[1:]).squeeze(-1))
            imag_v = self.critic.predict(self.critic(imag_s))
            ws = [c0]
            for k in range(self.horizon):
                ws.append(ws[-1] * imag_c[k])
            weights = torch.stack(ws, dim=0)
            rets = [None] * (self.horizon + 1)
            rets[self.horizon] = imag_v[self.horizon]
            for k in reversed(range(self.horizon)):
                rets[k] = imag_r[k] + self.gamma * imag_c[k] * ((1.0 - self.lam) * imag_v[k + 1] + self.lam * rets[k + 1])
            rets = torch.stack(rets, dim=0)
            span = (torch.quantile(rets[:-1].reshape(-1), 0.95) - torch.quantile(rets[:-1].reshape(-1), 0.05)).item()
            self.ret_scale = span if not self.scale_init else 0.99 * self.ret_scale + 0.01 * span
            self.scale_init = True
            adv = (rets[:-1] - imag_v[:-1]) / max(1.0, self.ret_scale)

        d = self.actor.dist(imag_s[:-1].detach())
        a_loss = (weights[:-1] * (-adv * d.log_prob(imag_a).sum(dim=-1) - self.eta * d.entropy().sum(dim=-1))).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()

        c_logits = self.critic(imag_s[:-1].detach())
        with torch.no_grad():
            ema_v = self.ema_critic.predict(self.ema_critic(imag_s[:-1].detach()))
        c_loss = (weights[:-1] * (self.critic.loss(c_logits, rets[:-1]) + self.critic.loss(c_logits, ema_v))).mean()
        self.critic_opt.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.critic_opt.step()

        with torch.no_grad():
            for p, pe in zip(self.critic.parameters(), self.ema_critic.parameters()):
                pe.data.mul_(0.98).add_(p.data, alpha=0.02)
        return wm_loss.item(), a_loss.item(), c_loss.item()


def run_dreamer_on_benchmark(task_name, cfg, seed=0):
    """按 Phase 5 统一排行榜协议运行 Mini-Dreamer V3，返回 (eval_steps, eval_rets, wall_time, decision_ms)。"""
    t0 = time.time()
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = gym.make(task_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    agent = MiniDreamerV3Agent(obs_dim, act_dim, horizon=12, hidden=128)
    episodes = []

    def sample_batch(B=16, T=16):
        ob, ab, rb, cb, fb = [], [], [], [], []
        valid_eps = [ep for ep in episodes if len(ep[0]) > 3]
        for _ in range(B):
            o_e, a_e, r_e, c_e = valid_eps[int(rng.integers(0, len(valid_eps)))]
            L = len(o_e)
            eff_T = min(T, L - 1)
            s = int(rng.integers(1, L - eff_T + 1)) if L - eff_T > 1 else 1
            o_sl, a_sl, r_sl, c_sl = o_e[s : s + eff_T], a_e[s : s + eff_T], r_e[s : s + eff_T], c_e[s : s + eff_T]
            if eff_T < T:
                pad = T - eff_T
                o_sl = np.concatenate([o_sl, np.repeat(o_sl[-1:], pad, axis=0)], axis=0)
                a_sl = np.concatenate([a_sl, np.zeros((pad, act_dim), dtype=np.float32)], axis=0)
                r_sl = np.concatenate([r_sl, np.zeros(pad, dtype=np.float32)], axis=0)
                c_sl = np.concatenate([c_sl, np.zeros(pad, dtype=np.float32)], axis=0)
            f_sl = np.zeros(T, dtype=np.float32)
            if s == 1:
                f_sl[0] = 1.0
            ob.append(o_sl)
            ab.append(a_sl)
            rb.append(r_sl)
            cb.append(c_sl)
            fb.append(f_sl)
        return tuple(torch.from_numpy(np.stack(x)) for x in (ob, ab, rb, cb, fb))

    init_random = min(600, cfg["total_steps"] // 3)
    train_every = 40
    updates_per_call = 12
    total_steps = cfg["total_steps"]
    eval_every = cfg["eval_every"]

    obs, _ = env.reset(seed=seed)
    ep_o, ep_a, ep_r, ep_c = [obs.astype(np.float32)], [np.zeros(act_dim, dtype=np.float32)], [0.0], [1.0]
    state = agent.init_state()
    eval_steps, eval_rets = [], []
    decision_times = []

    for step in range(1, total_steps + 1):
        if step <= init_random:
            a = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            ta0 = time.time()
            a, state = agent.act(obs, state, deterministic=False)
            decision_times.append((time.time() - ta0) * 1000.0)

        nxt, r, term, trunc, _ = env.step(a * act_scale)
        ep_o.append(nxt.astype(np.float32))
        ep_a.append(a)
        ep_r.append(float(r))
        ep_c.append(0.0 if term else 1.0)
        obs = nxt

        if term or trunc:
            episodes.append((np.asarray(ep_o), np.asarray(ep_a), np.asarray(ep_r, dtype=np.float32), np.asarray(ep_c, dtype=np.float32)))
            obs, _ = env.reset()
            ep_o, ep_a, ep_r, ep_c = [obs.astype(np.float32)], [np.zeros(act_dim, dtype=np.float32)], [0.0], [1.0]
            state = agent.init_state()

        if step >= init_random and step % train_every == 0 and any(len(e[0]) > 3 for e in episodes):
            for _ in range(updates_per_call):
                agent.train_step(sample_batch())

        if step % eval_every == 0 or step == total_steps:
            # 每回合独立重置 RSSM 隐状态
            ev_env = gym.make(task_name)
            ev_scores = []
            for ei in range(cfg["eval_episodes"]):
                eo, _ = ev_env.reset(seed=900 + ei)
                es = agent.init_state()
                tot = 0.0
                while True:
                    ea, es = agent.act(eo, es, deterministic=True)
                    eo, er, eterm, etrunc, _ = ev_env.step(ea * act_scale)
                    tot += float(er)
                    if eterm or etrunc:
                        break
                ev_scores.append(tot)
            ev_env.close()
            eval_steps.append(step)
            eval_rets.append(float(np.mean(ev_scores)))

    env.close()
    avg_ms = float(np.mean(decision_times)) if decision_times else 0.0
    return eval_steps, eval_rets, time.time() - t0, avg_ms
