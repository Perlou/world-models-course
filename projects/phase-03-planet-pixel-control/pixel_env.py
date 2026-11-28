"""像素版 Pendulum-v1 环境封装与序列回放缓冲区。"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import gymnasium as gym
import numpy as np
import torch

IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
ACTION_DIM = 1
MAX_TORQUE = 2.0
DEFAULT_ACTION_REPEAT = 2
ENV_MAX_STEPS = 200


def make_env(seed=None):
    """创建渲染模式的 Pendulum-v1。"""
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    if seed is not None:
        env.reset(seed=seed)
    return env


def render_frame(env, size=IMAGE_SIZE):
    """渲染当前画面并缩放到 64×64 uint8，隐藏转轴处的力矩箭头以避免动作直接画在图像上。"""
    raw = env.unwrapped
    raw.last_u = None
    frame = raw.render()
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def step_repeat(env, norm_action, action_repeat=DEFAULT_ACTION_REPEAT):
    """输入归一化动作 a ∈ [-1, 1]，连续执行 R 次环境步，累加奖励并返回最后一帧。"""
    torque = np.clip(np.asarray(norm_action, dtype=np.float32), -1.0, 1.0) * MAX_TORQUE
    total_reward = 0.0
    for _ in range(action_repeat):
        _, reward, _, _, _ = env.step(torque)
        total_reward += float(reward)
    return render_frame(env), total_reward, env.unwrapped.state.copy().astype(np.float32)


def preprocess_obs(obs_u8):
    """uint8 [..., H, W, 3] → float32 [..., 3, H, W]，缩放到 [-0.5, 0.5]。"""
    if obs_u8.ndim == 4:
        return obs_u8.permute(0, 3, 1, 2).float() / 255.0 - 0.5
    return obs_u8.permute(0, 1, 4, 2, 3).float() / 255.0 - 0.5


def postprocess_obs(obs_tensor):
    """float32 [..., 3, H, W] → float32 [..., H, W, 3]，裁剪到 [0, 1]。"""
    if obs_tensor.ndim == 4:
        return (obs_tensor + 0.5).clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()
    return (obs_tensor + 0.5).clamp(0.0, 1.0).permute(0, 1, 3, 4, 2).cpu().numpy()


class SequenceReplay:
    """按回合存储 uint8 图像的序列回放缓冲区，采样片段绝不跨越回合边界。"""

    def __init__(self):
        self.episodes = []

    def add_episode(self, obs_u8, actions, rewards):
        obs_arr = np.asarray(obs_u8, dtype=np.uint8)
        act_arr = np.asarray(actions, dtype=np.float32)
        rew_arr = np.asarray(rewards, dtype=np.float32)
        assert len(obs_arr) == len(act_arr) == len(rew_arr) and len(obs_arr) >= 2
        self.episodes.append((obs_arr, act_arr, rew_arr))

    def __len__(self):
        return len(self.episodes)

    def total_decision_steps(self):
        return sum(len(ep[0]) - 1 for ep in self.episodes)

    def sample(self, batch_size, chunk_len, device, generator=None):
        """采样 [B, T, ·]，起点 ≥ 1 以确保每步都有真实执行的动作 a_{t-1} 和奖励 r_{t-1}。"""
        ep_ids = torch.randint(0, len(self.episodes), (batch_size,), generator=generator).tolist()
        obs_b, act_b, rew_b = [], [], []
        for idx in ep_ids:
            obs_ep, act_ep, rew_ep = self.episodes[idx]
            max_start = len(obs_ep) - chunk_len
            assert max_start >= 1, f"回合长度 {len(obs_ep)} 不足以截取 chunk_len={chunk_len}"
            start = int(torch.randint(1, max_start + 1, (1,), generator=generator).item())
            obs_b.append(obs_ep[start : start + chunk_len])
            act_b.append(act_ep[start : start + chunk_len])
            rew_b.append(rew_ep[start : start + chunk_len])

        obs_t = preprocess_obs(torch.from_numpy(np.stack(obs_b)).to(device))
        act_t = torch.from_numpy(np.stack(act_b)).to(device)
        rew_t = torch.from_numpy(np.stack(rew_b)).to(device)
        return obs_t, act_t, rew_t
