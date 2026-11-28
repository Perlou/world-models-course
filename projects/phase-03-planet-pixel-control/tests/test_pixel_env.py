"""测试像素版环境封装、动作重复累加与序列回放边界不跨越。"""

import numpy as np
import torch

from pixel_env import IMAGE_SIZE, MAX_TORQUE, SequenceReplay, make_env, preprocess_obs, render_frame, step_repeat


def test_render_shape_and_no_torque_leak():
    env = make_env(seed=0)
    env.unwrapped.state = np.array([0.8, -2.0])
    env.unwrapped.last_u = -MAX_TORQUE
    f1 = render_frame(env)
    env.unwrapped.last_u = +MAX_TORQUE
    f2 = render_frame(env)
    env.close()
    assert f1.shape == (IMAGE_SIZE, IMAGE_SIZE, 3) and f1.dtype == np.uint8
    assert np.array_equal(f1, f2), "渲染画面不应泄露上一步力矩箭头"


def test_step_repeat_accumulates_reward():
    env_a, env_b = make_env(seed=0), make_env(seed=0)
    state0 = np.array([1.2, 0.5], dtype=np.float32)
    env_a.unwrapped.state = state0.copy()
    env_b.unwrapped.state = state0.copy()

    _, r_rep, s_rep = step_repeat(env_a, np.array([0.5], dtype=np.float32), action_repeat=3)
    r_manual = 0.0
    for _ in range(3):
        _, r, _, _, _ = env_b.step(np.array([0.5 * MAX_TORQUE], dtype=np.float32))
        r_manual += float(r)
    s_manual = env_b.unwrapped.state.copy()
    env_a.close()
    env_b.close()
    assert abs(r_rep - r_manual) < 1e-5
    assert np.allclose(s_rep, s_manual, atol=1e-6)


def test_sequence_replay_never_crosses_episode_boundary():
    replay = SequenceReplay()
    # 造两个特征鲜明的回合：回合 0 全是奖励 +10，回合 1 全是奖励 -10，且第 0 位放哨兵值 999
    for sign in (1.0, -1.0):
        obs = np.zeros((15, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        act = np.full((15, 1), sign * 0.5, dtype=np.float32)
        rew = np.full(15, sign * 10.0, dtype=np.float32)
        rew[0] = 999.0  # 下标 0 是起点占位，不应被采进训练片段
        replay.add_episode(obs, act, rew)

    obs_b, act_b, rew_b = replay.sample(batch_size=32, chunk_len=8, device=torch.device("cpu"))
    assert obs_b.shape == (32, 8, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert obs_b.min() >= -0.5 and obs_b.max() <= 0.5
    assert (rew_b != 999.0).all(), "不应采到下标 0 的起点占位奖励"
    # 每条片段内部所有步必须同号（全部来自同一个回合）
    for row in rew_b:
        assert (row == 10.0).all() or (row == -10.0).all()
