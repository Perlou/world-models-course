"""隐空间 CEM 规划器与像素回合采集。"""

import time

import numpy as np
import torch

from pixel_env import (
    ACTION_DIM,
    DEFAULT_ACTION_REPEAT,
    ENV_MAX_STEPS,
    IMAGE_SIZE,
    make_env,
    preprocess_obs,
    render_frame,
    step_repeat,
)
from utils.planning import cem_plan, shift_plan


@torch.no_grad()
def latent_cem_plan(model, h, z, horizon=15, n_candidates=300, n_elites=30, n_iters=6,
                    warm_mean=None, generator=None):
    """从信念状态 (h, z) 出发在隐空间运行 CEM，返回最优动作序列均值 [H, A]。"""
    h_batch = h.expand(n_candidates, -1)
    z_batch = z.expand(n_candidates, -1)

    def score_fn(actions):
        feats = model.rssm.imagine(h_batch, z_batch, actions, sample=True, generator=generator)
        return model.reward_head(feats).sum(dim=-1)

    init_mean = torch.zeros(horizon, ACTION_DIM, device=h.device) if warm_mean is None else warm_mean
    smoothing = 0.0 if warm_mean is None else 0.1
    return cem_plan(
        score_fn,
        init_mean,
        generator=generator,
        action_low=-1.0,
        action_high=1.0,
        n_candidates=n_candidates,
        n_elites=n_elites,
        n_iters=n_iters,
        init_std=1.0,
        min_std=0.05,
        smoothing=smoothing,
    )


@torch.no_grad()
def run_episode(seed, model=None, device=torch.device("cpu"), action_repeat=DEFAULT_ACTION_REPEAT,
                explore_std=0.0, horizon=15, n_candidates=300, n_iters=6, warm_start=False):
    """在像素版 Pendulum-v1 上运行一个回合，智能体全程只能看到 64×64 渲染图。"""
    n_decisions = ENV_MAX_STEPS // action_repeat
    L = n_decisions + 1
    env = make_env(seed=seed)
    rng = np.random.default_rng(seed + 777)

    obs_ep = np.zeros((L, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    act_ep = np.zeros((L, ACTION_DIM), dtype=np.float32)
    rew_ep = np.zeros(L, dtype=np.float32)
    obs_ep[0] = render_frame(env)

    plan_time = 0.0
    if model is not None:
        model.eval()
        h, z = model.rssm.initial_state(1, device)
        a_prev = torch.zeros(1, ACTION_DIM, device=device)
        warm_mean = torch.zeros(horizon, ACTION_DIM, device=device) if warm_start else None
        gen = torch.Generator(device=device).manual_seed(seed)

    for t in range(1, L):
        if model is None:
            if seed % 2 == 1 and t > 1:
                a = np.clip(0.7 * act_ep[t - 1] + 0.65 * rng.standard_normal(ACTION_DIM), -1.0, 1.0).astype(np.float32)
            else:
                a = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
        else:
            t0 = time.time()
            obs_t = preprocess_obs(torch.from_numpy(obs_ep[t - 1 : t]).to(device))
            h, z, _, _ = model.rssm.observe_step(h, z, a_prev, model.encoder(obs_t), sample=False)
            plan_mean = latent_cem_plan(
                model, h, z, horizon=horizon, n_candidates=n_candidates,
                n_elites=max(10, n_candidates // 10), n_iters=n_iters,
                warm_mean=warm_mean, generator=gen,
            )
            a = plan_mean[0].cpu().numpy().astype(np.float32)
            if explore_std > 0.0:
                a = np.clip(a + rng.normal(0.0, explore_std, size=ACTION_DIM), -1.0, 1.0).astype(np.float32)
            a_prev = torch.from_numpy(a[None]).to(device)
            if warm_start:
                warm_mean = shift_plan(plan_mean)
            plan_time += time.time() - t0

        frame, reward, _ = step_repeat(env, a, action_repeat=action_repeat)
        obs_ep[t] = frame
        act_ep[t] = a
        rew_ep[t] = reward

    env.close()
    return dict(obs=obs_ep, actions=act_ep, rewards=rew_ep, ret=float(rew_ep.sum()), plan_time=plan_time)
