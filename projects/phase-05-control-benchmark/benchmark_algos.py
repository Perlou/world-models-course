"""Phase 5 排行榜五类算法实现：Random、PETS、PPO、SAC、MBPO。

全部遵循统一接口：输入 task_name、cfg（来自 benchmark_protocol）和 seed，
返回 (eval_steps, eval_returns, wall_time_sec, utd_ratio)。
动作空间统一在内部归一化到 [-1, 1]，与环境交互时乘以 action_space.high。
"""

import copy
import math
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from benchmark_protocol import evaluate_policy_fn, is_terminal_state
from utils import seed_everything
from utils.planning import cem_plan, shift_plan


# ==================== 共享回放池与网络部件 ====================


class BenchReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=50_000):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.term = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, no, terminated):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr, 0] = r
        self.next_obs[self.ptr] = no
        self.term[self.ptr, 0] = float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.act[idx]),
            torch.from_numpy(self.rew[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.term[idx]),
        )


def _tanh_log_prob(mean, std, u):
    base = Normal(mean, std).log_prob(u)
    corr = 2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))
    return (base - corr).sum(dim=-1, keepdim=True)


class BenchTanhActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs, deterministic=False):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        if deterministic:
            return torch.tanh(mean), None
        log_std = self.log_std_head(h).clamp(-5.0, 2.0)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        return torch.tanh(u), _tanh_log_prob(mean, std, u)


class BenchQCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class BenchSAC:
    def __init__(self, obs_dim, act_dim, hidden=128, lr=1e-3, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -float(act_dim)
        self.actor = BenchTanhActor(obs_dim, act_dim, hidden)
        self.q1 = BenchQCritic(obs_dim, act_dim, hidden)
        self.q2 = BenchQCritic(obs_dim, act_dim, hidden)
        self.q1_t = copy.deepcopy(self.q1)
        self.q2_t = copy.deepcopy(self.q2)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.log_alpha = torch.tensor(math.log(0.2), requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @torch.no_grad()
    def act(self, obs_np, deterministic=False):
        t = torch.from_numpy(np.asarray(obs_np, dtype=np.float32)).unsqueeze(0)
        a, _ = self.actor(t, deterministic=deterministic)
        return a.squeeze(0).numpy()

    def update(self, batch):
        obs, act, rew, nxt, term = batch
        alpha = self.log_alpha.detach().exp()
        with torch.no_grad():
            nxt_a, nxt_lp = self.actor(nxt)
            q_nxt = torch.min(self.q1_t(nxt, nxt_a), self.q2_t(nxt, nxt_a)) - alpha * nxt_lp
            target = rew + self.gamma * (1.0 - term) * q_nxt

        q_loss = F.mse_loss(self.q1(obs, act), target) + F.mse_loss(self.q2(obs, act), target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        new_a, lp = self.actor(obs)
        q_pi = torch.min(self.q1(obs, new_a), self.q2(obs, new_a))
        actor_loss = (alpha * lp - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (lp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)


class BenchProbEnsemble(nn.Module):
    """带奖励预测的概率集成（供 PETS 与 MBPO 共享）。"""

    def __init__(self, obs_dim, act_dim, n_ensemble=5, hidden=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_ensemble = n_ensemble
        self.members = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, 2 * (obs_dim + 1)),
            )
            for _ in range(n_ensemble)
        ])
        self.max_logvar = nn.Parameter(torch.full((obs_dim + 1,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((obs_dim + 1,), -6.0))
        self.register_buffer("in_mean", torch.zeros(obs_dim + act_dim))
        self.register_buffer("in_std", torch.ones(obs_dim + act_dim))
        self.register_buffer("tgt_mean", torch.zeros(obs_dim + 1))
        self.register_buffer("tgt_std", torch.ones(obs_dim + 1))

    def _bound_lv(self, raw_lv):
        lv = self.max_logvar - F.softplus(self.max_logvar - raw_lv)
        return self.min_logvar + F.softplus(lv - self.min_logvar)

    def _member_pred(self, member, x_norm):
        out = member(x_norm)
        mu, raw_lv = out.chunk(2, dim=-1)
        return mu, self._bound_lv(raw_lv)

    def fit(self, replay, epochs=10, batch_size=128, lr=2e-3):
        n = len(replay)
        obs = torch.from_numpy(replay.obs[:n])
        act = torch.from_numpy(replay.act[:n])
        rew = torch.from_numpy(replay.rew[:n])
        nxt = torch.from_numpy(replay.next_obs[:n])
        x_raw = torch.cat([obs, act], dim=-1)
        y_raw = torch.cat([nxt - obs, rew], dim=-1)
        self.in_mean.copy_(x_raw.mean(dim=0))
        self.in_std.copy_(x_raw.std(dim=0).clamp_min(1e-3))
        self.tgt_mean.copy_(y_raw.mean(dim=0))
        self.tgt_std.copy_(y_raw.std(dim=0).clamp_min(1e-3))

        x_n = (x_raw - self.in_mean) / self.in_std
        y_n = (y_raw - self.tgt_mean) / self.tgt_std
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        for _ in range(epochs):
            perms = [torch.randperm(n) for _ in range(self.n_ensemble)]
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                loss = 0.0
                for idx_m, m in enumerate(self.members):
                    idx = perms[idx_m][s:e]
                    mu, lv = self._member_pred(m, x_n[idx])
                    loss = loss + ((mu - y_n[idx]).pow(2) * torch.exp(-lv) + lv).mean()
                loss = loss / self.n_ensemble + 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
                opt.zero_grad()
                loss.backward()
                opt.step()

    @torch.no_grad()
    def sample_step(self, obs, act, task_name, generator=None):
        x_n = (torch.cat([obs, act], dim=-1) - self.in_mean) / self.in_std
        mus, lvs = [], []
        for m in self.members:
            mu, lv = self._member_pred(m, x_n)
            mus.append(mu)
            lvs.append(lv)
        mus = torch.stack(mus, dim=0)
        lvs = torch.stack(lvs, dim=0)
        M = obs.shape[0]
        m_idx = torch.randint(0, self.n_ensemble, (M,), generator=generator)
        b_idx = torch.arange(M)
        mu = mus[m_idx, b_idx]
        std = torch.exp(0.5 * lvs[m_idx, b_idx])
        pred = (mu + std * torch.randn(mu.shape, generator=generator)) * self.tgt_std + self.tgt_mean
        nxt = obs + pred[:, : self.obs_dim]
        rew = pred[:, self.obs_dim :]
        if task_name == "Pendulum-v1":
            norm = torch.sqrt(nxt[:, 0:1].pow(2) + nxt[:, 1:2].pow(2)).clamp_min(1e-5)
            nxt = torch.cat([nxt[:, 0:2] / norm, nxt[:, 2:3].clamp(-8.0, 8.0)], dim=-1)
        else:
            nxt = nxt.clamp(-50.0, 50.0)
        term = is_terminal_state(task_name, nxt)
        return nxt, rew, term


# ==================== 1. 随机策略 ====================


def run_random(task_name, cfg, seed=0):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    env = gym.make(task_name)
    act_dim = env.action_space.shape[0]
    env.close()

    val = evaluate_policy_fn(
        task_name, lambda _o: rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32),
        n_episodes=cfg["eval_episodes"], seed_base=900,
    )
    steps = list(range(cfg["eval_every"], cfg["total_steps"] + 1, cfg["eval_every"]))
    if not steps or steps[-1] != cfg["total_steps"]:
        steps.append(cfg["total_steps"])
    return steps, [val for _ in steps], time.time() - t0, 0.0


# ==================== 2. PPO（连续控制） ====================


class BenchPPOActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.pi_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.v_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def dist(self, obs):
        mean = torch.tanh(self.pi_net(obs))
        std = self.log_std.clamp(-3.0, 0.5).exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, obs):
        return self.v_net(obs).squeeze(-1)


def run_ppo(task_name, cfg, seed=0):
    t0 = time.time()
    seed_everything(seed)
    env = gym.make(task_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    ac = BenchPPOActorCritic(obs_dim, act_dim, hidden=128)
    opt = torch.optim.Adam(ac.parameters(), lr=3e-4)

    rollout_len = min(1000, cfg["eval_every"])
    obs, _ = env.reset(seed=seed)
    total_steps = cfg["total_steps"]
    eval_every = cfg["eval_every"]
    next_eval = eval_every
    eval_steps, eval_rets = [], []
    step = 0
    grad_updates = 0

    while step < total_steps:
        buf_o, buf_a, buf_lp, buf_r, buf_d, buf_v = [], [], [], [], [], []
        n_collect = min(rollout_len, total_steps - step)
        for _ in range(n_collect):
            o_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                d = ac.dist(o_t)
                a_raw = d.sample()
                lp = d.log_prob(a_raw).sum(dim=-1)
                v = ac.value(o_t)
            a_clip = a_raw.squeeze(0).clamp(-1.0, 1.0).numpy()
            nxt, r, term, trunc, _ = env.step(a_clip * act_scale)
            buf_o.append(obs.astype(np.float32))
            buf_a.append(a_raw.squeeze(0).numpy())
            buf_lp.append(lp.item())
            buf_r.append(float(r))
            buf_d.append(float(term or trunc))
            buf_v.append(v.item())
            obs = nxt
            step += 1
            if term or trunc:
                obs, _ = env.reset()

        with torch.no_grad():
            last_v = ac.value(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)).item()
        adv = np.zeros(n_collect, dtype=np.float32)
        gae = 0.0
        for i in reversed(range(n_collect)):
            nxt_v = last_v if i == n_collect - 1 else buf_v[i + 1]
            delta = buf_r[i] + 0.99 * nxt_v * (1.0 - buf_d[i]) - buf_v[i]
            gae = delta + 0.99 * 0.95 * (1.0 - buf_d[i]) * gae
            adv[i] = gae
        ret = adv + np.asarray(buf_v, dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        o_b = torch.from_numpy(np.asarray(buf_o, dtype=np.float32))
        a_b = torch.from_numpy(np.asarray(buf_a, dtype=np.float32))
        lp_old = torch.from_numpy(np.asarray(buf_lp, dtype=np.float32))
        adv_b = torch.from_numpy(adv)
        ret_b = torch.from_numpy(ret)

        bs = min(64, n_collect)
        for _ in range(6):
            perm = torch.randperm(n_collect)
            for s in range(0, n_collect, bs):
                idx = perm[s : s + bs]
                d = ac.dist(o_b[idx])
                lp = d.log_prob(a_b[idx]).sum(dim=-1)
                ratio = torch.exp((lp - lp_old[idx]).clamp(-10.0, 10.0))
                surr1 = ratio * adv_b[idx]
                surr2 = ratio.clamp(0.8, 1.2) * adv_b[idx]
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * F.mse_loss(ac.value(o_b[idx]), ret_b[idx])
                ent = d.entropy().sum(dim=-1).mean()
                loss = pi_loss + 0.5 * v_loss - 0.005 * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 0.5)
                opt.step()
                grad_updates += 1

        while step >= next_eval:
            ev = evaluate_policy_fn(
                task_name,
                lambda o: ac.dist(torch.from_numpy(o.astype(np.float32)).unsqueeze(0)).mean.detach().squeeze(0).numpy(),
                n_episodes=cfg["eval_episodes"], seed_base=900,
            )
            eval_steps.append(next_eval)
            eval_rets.append(ev)
            next_eval += eval_every

    env.close()
    return eval_steps, eval_rets, time.time() - t0, grad_updates / max(1, total_steps)


# ==================== 3. SAC ====================


def run_sac(task_name, cfg, seed=0):
    t0 = time.time()
    seed_everything(seed)
    env = gym.make(task_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    agent = BenchSAC(obs_dim, act_dim, hidden=128, lr=1e-3)
    replay = BenchReplayBuffer(obs_dim, act_dim, capacity=max(30_000, cfg["total_steps"] + 500))
    init_random = min(400, cfg["total_steps"] // 3)

    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    eval_steps, eval_rets = [], []
    grad_updates = 0

    for step in range(1, cfg["total_steps"] + 1):
        if step <= init_random:
            a = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            a = agent.act(obs, deterministic=False)
        nxt, r, term, trunc, _ = env.step(a * act_scale)
        replay.add(obs, a, r, nxt, term)
        obs = nxt
        if term or trunc:
            obs, _ = env.reset()

        if step >= init_random:
            agent.update(replay.sample(128))
            grad_updates += 1

        if step % cfg["eval_every"] == 0 or step == cfg["total_steps"]:
            ev = evaluate_policy_fn(
                task_name, lambda o: agent.act(o, deterministic=True),
                n_episodes=cfg["eval_episodes"], seed_base=900,
            )
            eval_steps.append(step)
            eval_rets.append(ev)

    env.close()
    return eval_steps, eval_rets, time.time() - t0, grad_updates / max(1, cfg["total_steps"])


# ==================== 4. MBPO ====================


def run_mbpo(task_name, cfg, seed=0, rollout_k=1, utd=6, model_ratio=0.85):
    t0 = time.time()
    seed_everything(seed)
    env = gym.make(task_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    agent = BenchSAC(obs_dim, act_dim, hidden=128, lr=1e-3)
    env_replay = BenchReplayBuffer(obs_dim, act_dim, capacity=max(30_000, cfg["total_steps"] + 500))
    rollouts_per_fit = 300
    model_replay = BenchReplayBuffer(obs_dim, act_dim, capacity=max(2000, rollouts_per_fit * rollout_k * 2))
    ensemble = BenchProbEnsemble(obs_dim, act_dim, n_ensemble=5, hidden=128)

    init_random = min(400, cfg["total_steps"] // 3)
    fit_every = max(200, cfg["eval_every"] // 2)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    eval_steps, eval_rets = [], []
    grad_updates = 0

    for step in range(1, cfg["total_steps"] + 1):
        if step <= init_random:
            a = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            a = agent.act(obs, deterministic=False)
        nxt, r, term, trunc, _ = env.step(a * act_scale)
        env_replay.add(obs, a, r, nxt, term)
        obs = nxt
        if term or trunc:
            obs, _ = env.reset()

        if step >= init_random:
            if step == init_random or (step - init_random) % fit_every == 0:
                ensemble.fit(env_replay, epochs=10, batch_size=128)
                n_s = min(rollouts_per_fit, len(env_replay))
                idx = np.random.randint(0, len(env_replay), size=n_s)
                curr = torch.from_numpy(env_replay.obs[idx])
                with torch.no_grad():
                    for _ in range(rollout_k):
                        act_t, _ = agent.actor(curr, deterministic=False)
                        nxt_t, rew_t, term_t = ensemble.sample_step(curr, act_t, task_name, generator=gen)
                        c_np, a_np, r_np, n_np, t_np = (
                            curr.numpy(), act_t.numpy(), rew_t.squeeze(-1).numpy(), nxt_t.numpy(), term_t.numpy()
                        )
                        for i in range(c_np.shape[0]):
                            model_replay.add(c_np[i], a_np[i], float(r_np[i]), n_np[i], bool(t_np[i]))
                        alive = ~term_t
                        if not alive.any():
                            break
                        curr = nxt_t[alive]

            n_mod = int(round(128 * model_ratio))
            n_env = 128 - n_mod
            for _ in range(utd):
                if len(model_replay) >= n_mod and n_mod > 0:
                    bm = model_replay.sample(n_mod)
                    be = env_replay.sample(n_env)
                    batch = tuple(torch.cat([x, y], dim=0) for x, y in zip(bm, be))
                else:
                    batch = env_replay.sample(128)
                agent.update(batch)
                grad_updates += 1

        if step % cfg["eval_every"] == 0 or step == cfg["total_steps"]:
            ev = evaluate_policy_fn(
                task_name, lambda o: agent.act(o, deterministic=True),
                n_episodes=cfg["eval_episodes"], seed_base=900,
            )
            eval_steps.append(step)
            eval_rets.append(ev)

    env.close()
    return eval_steps, eval_rets, time.time() - t0, grad_updates / max(1, cfg["total_steps"])


# ==================== 5. PETS（概率集成 + CEM 决策时规划） ====================


def run_pets(task_name, cfg, seed=0, horizon=12, n_candidates=120, n_iters=4):
    t0 = time.time()
    seed_everything(seed)
    env = gym.make(task_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = float(env.action_space.high[0])

    replay = BenchReplayBuffer(obs_dim, act_dim, capacity=max(30_000, cfg["total_steps"] + 500))
    ensemble = BenchProbEnsemble(obs_dim, act_dim, n_ensemble=5, hidden=128)

    init_random = min(400, cfg["total_steps"] // 3)
    fit_every = max(400, cfg["eval_every"])
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    eval_steps, eval_rets = [], []

    def plan_action(o_np, warm_state):
        o0 = torch.from_numpy(np.asarray(o_np, dtype=np.float32)).unsqueeze(0).expand(n_candidates, -1)

        def score_fn(actions):
            s = o0
            total_r = torch.zeros(n_candidates)
            alive = torch.ones(n_candidates, dtype=torch.bool)
            for h in range(horizon):
                s, r, term = ensemble.sample_step(s, actions[:, h], task_name, generator=gen)
                total_r = total_r + r.squeeze(-1) * alive.float()
                alive = alive & (~term)
            return total_r

        init_mean = warm_state["mean"]
        plan_mean = cem_plan(
            score_fn, init_mean, generator=gen, action_low=-1.0, action_high=1.0,
            n_candidates=n_candidates, n_elites=max(10, n_candidates // 10),
            n_iters=n_iters, init_std=0.6, min_std=0.05, smoothing=0.1,
        )
        warm_state["mean"] = shift_plan(plan_mean)
        return plan_mean[0].numpy()

    warm = {"mean": torch.zeros(horizon, act_dim)}
    # Pendulum-v1 每次完整 CEM；MuJoCo 长回合用动作重复 2 步以保持规划开销合理
    plan_repeat = 1 if task_name == "Pendulum-v1" else 2
    cached_a = np.zeros(act_dim, dtype=np.float32)

    for step in range(1, cfg["total_steps"] + 1):
        if step <= init_random:
            a = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            if (step - 1) % plan_repeat == 0:
                cached_a = np.clip(plan_action(obs, warm) + rng.normal(0.0, 0.1, size=act_dim), -1.0, 1.0).astype(np.float32)
            a = cached_a

        nxt, r, term, trunc, _ = env.step(a * act_scale)
        replay.add(obs, a, r, nxt, term)
        obs = nxt
        if term or trunc:
            obs, _ = env.reset()
            warm["mean"] = torch.zeros(horizon, act_dim)

        if step == init_random or (step > init_random and (step - init_random) % fit_every == 0):
            ensemble.fit(replay, epochs=12, batch_size=128)

        if step % cfg["eval_every"] == 0 or step == cfg["total_steps"]:
            if step < init_random:
                ensemble.fit(replay, epochs=10, batch_size=128)
            eval_warm = {"mean": torch.zeros(horizon, act_dim)}
            eval_counter = {"k": 0, "a": np.zeros(act_dim, dtype=np.float32)}

            def _eval_act(o_in):
                if eval_counter["k"] % plan_repeat == 0:
                    eval_counter["a"] = plan_action(o_in, eval_warm)
                eval_counter["k"] += 1
                return eval_counter["a"]

            # 每回合重置热启动
            ev_rets_tmp = []
            ev_env = gym.make(task_name)
            for ei in range(cfg["eval_episodes"]):
                eo, _ = ev_env.reset(seed=900 + ei)
                eval_warm["mean"] = torch.zeros(horizon, act_dim)
                eval_counter["k"] = 0
                tot = 0.0
                while True:
                    ea = np.clip(_eval_act(eo), -1.0, 1.0)
                    eo, er, eterm, etrunc, _ = ev_env.step(ea * act_scale)
                    tot += float(er)
                    if eterm or etrunc:
                        break
                ev_rets_tmp.append(tot)
            ev_env.close()
            eval_steps.append(step)
            eval_rets.append(float(np.mean(ev_rets_tmp)))

    env.close()
    return eval_steps, eval_rets, time.time() - t0, 0.0
