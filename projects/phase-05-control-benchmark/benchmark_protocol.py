"""Phase 5–7 统一连续控制排行榜协议。

Phase 5（Random / PETS / PPO / SAC / MBPO）、Phase 6（Dreamer）、Phase 7（TD-MPC2）
全部复用这里的任务配置、随机种子与无噪评估函数，确保横向对比完全公平。
"""

import gymnasium as gym
import numpy as np


BENCHMARK_TASKS = {
    "Pendulum-v1": {
        "total_steps": 4000,
        "eval_every": 500,
        "eval_episodes": 5,
        "seeds": [0, 1, 2],
        "threshold": -250.0,
        "max_ep_steps": 200,
    },
    "HalfCheetah-v5": {
        "total_steps": 10000,
        "eval_every": 2000,
        "eval_episodes": 3,
        "seeds": [0, 1],
        "threshold": 200.0,
        "max_ep_steps": 1000,
    },
    "Hopper-v5": {
        "total_steps": 10000,
        "eval_every": 2000,
        "eval_episodes": 3,
        "seeds": [0, 1],
        "threshold": 200.0,
        "max_ep_steps": 1000,
    },
}


def get_task_protocol(task_name, quick=False):
    cfg = dict(BENCHMARK_TASKS[task_name])
    if quick:
        cfg["total_steps"] = 600
        cfg["eval_every"] = 300
        cfg["eval_episodes"] = 1
        cfg["seeds"] = [0]
    return cfg


def is_terminal_state(task_name, obs_batch):
    """基于环境已知物理规则判断预测状态是否终止（供 MBPO / PETS 想象展开使用）。

    obs_batch: torch.Tensor 或 np.ndarray，形状 [..., S]。返回 bool 掩码 [...]。
    """
    if task_name == "Hopper-v5":
        # Gymnasium Hopper-v5 健康范围：高度 z > 0.7，俯仰角 |angle| < 0.2，其余状态绝对值 < 100
        height = obs_batch[..., 0]
        angle = obs_batch[..., 1]
        finite_ok = (obs_batch.abs() < 100.0).all(dim=-1) if hasattr(obs_batch, "abs") else np.all(np.abs(obs_batch) < 100.0, axis=-1)
        healthy = (height > 0.7) & (angle.abs() < 0.2 if hasattr(angle, "abs") else np.abs(angle) < 0.2) & finite_ok
        return ~healthy
    if hasattr(obs_batch, "new_zeros"):
        return obs_batch.new_zeros(obs_batch.shape[:-1], dtype=obs_batch.dtype).bool()
    return np.zeros(obs_batch.shape[:-1], dtype=bool)


def evaluate_policy_fn(task_name, act_fn, n_episodes=5, seed_base=900):
    """在固定评估种子上运行确定性策略（act_fn 接收 obs，返回归一化 [-1, 1] 动作）。"""
    env = gym.make(task_name)
    act_scale = float(env.action_space.high[0])
    rets = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        tot = 0.0
        while True:
            act_norm = np.clip(np.asarray(act_fn(obs), dtype=np.float32), -1.0, 1.0)
            obs, r, term, trunc, _ = env.step(act_norm * act_scale)
            tot += float(r)
            if term or trunc:
                break
        rets.append(tot)
    env.close()
    return float(np.mean(rets))


def steps_to_threshold(steps_arr, mean_curve, threshold):
    """返回平均评估曲线首次达到 threshold 所需的真实环境步数；未达到返回 None。"""
    for s, v in zip(steps_arr, mean_curve):
        if float(v) >= float(threshold):
            return int(s)
    return None
