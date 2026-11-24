"""
Reacher-v5 的状态、运动学与奖励
==============================

这个文件只做一件事：把 Gymnasium 的 Reacher-v5 整理成世界模型好用的形式。

    状态 s = (q1, q2, q̇1, q̇2)      两个关节的角度和角速度
    动作 a = (u1, u2) ∈ [-1, 1]²    两个关节的力矩
    目标 g = (gx, gy)               每个回合固定的目标点，不属于要预测的状态

环境的观测是 10 维的：(cos q, sin q, 目标, q̇, 指尖 − 目标)。其中目标在一个回合里不变，
指尖位置可以由关节角度算出来（正运动学）。所以世界模型只需要预测 4 维的关节状态，
奖励由已知的公式从关节状态算出。这和讲义里"奖励函数已知"的设定一致。
"""

import gymnasium as gym
import numpy as np
import torch

ENV_ID = "Reacher-v5"
STATE_DIM = 4
ACTION_DIM = 2
EPISODE_LENGTH = 50
# 两节连杆的长度（米）。来自 MuJoCo 的 reacher.xml，tests/test_reacher.py 里和仿真器逐点核对过
LINK1 = 0.10
LINK2 = 0.11


def make_env(render_mode=None):
    return gym.make(ENV_ID, render_mode=render_mode)


def get_state(env):
    """从仿真器里读出关节状态 (q1, q2, q̇1, q̇2)。"""
    data = env.unwrapped.data
    return np.concatenate([data.qpos[:2], data.qvel[:2]]).astype(np.float32)


def get_goal(env):
    """目标点 (gx, gy)。在仿真器里它是 qpos 的后两个分量。"""
    return env.unwrapped.data.qpos[2:4].astype(np.float32).copy()


def fingertip(states):
    """正运动学：关节角度 → 指尖位置。states [..., 4] → [..., 2]。

    NumPy 数组和 PyTorch 张量都能用。
    """
    lib = torch if isinstance(states, torch.Tensor) else np
    q1, q2 = states[..., 0], states[..., 1]
    x = LINK1 * lib.cos(q1) + LINK2 * lib.cos(q1 + q2)
    y = LINK1 * lib.sin(q1) + LINK2 * lib.sin(q1 + q2)
    return lib.stack([x, y], -1)


def elbow(states):
    """肘部（第一节连杆末端）的位置，画图用。"""
    q1 = states[..., 0]
    return np.stack([LINK1 * np.cos(q1), LINK1 * np.sin(q1)], -1)


def reward_fn(states, actions, goal):
    """Reacher-v5 的奖励：−(指尖到目标的距离) − (动作的平方和)。

    注意时序：环境是先执行动作、再用"执行之后"的指尖位置算距离。
    所以这里的 states 要传下一个状态 s_{t+1}，actions 传 a_t。
    states [..., 4]，actions [..., 2]，goal [2] → [...]

    和环境返回的奖励有不超过千分之几的出入：MuJoCo 在每个物理子步的开头计算各部件的位置，
    环境读到的指尖位置比关节角度滞后一个子步（0.01 秒）。这点出入对规划没有影响，
    tests/test_reacher.py 里按 5e-3 的容差核对。
    """
    dist = (fingertip(states) - goal).norm(dim=-1)
    return -dist - (actions**2).sum(dim=-1)


def features(states, actions):
    """模型的输入特征：角度换成 (cos, sin)，和讲义 03 的处理一样。[..., 4], [..., 2] → [..., 8]"""
    q, qdot = states[..., :2], states[..., 2:]
    return torch.cat([torch.cos(q), torch.sin(q), qdot, actions], dim=-1)


def run_random_episode(seed):
    """用随机策略跑一个回合。返回 (states [T, 4], actions [T, 2], next_states [T, 4], goal [2], 回报)。"""
    env = make_env()
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    goal = get_goal(env)
    s, a, s_next, total = [], [], [], 0.0
    for _ in range(EPISODE_LENGTH):
        s.append(get_state(env))
        action = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
        _, reward, _, _, _ = env.step(action)
        total += reward
        a.append(action)
        s_next.append(get_state(env))
    env.close()
    return np.array(s), np.array(a), np.array(s_next), goal, total
