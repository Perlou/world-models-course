"""核对手写的运动学和奖励公式与仿真器一致。模型的全部训练信号都建立在这两个公式上。"""

import numpy as np
import torch

import reacher


def test_fingertip_matches_simulator():
    env = reacher.make_env()
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        qpos = env.unwrapped.data.qpos.copy()
        qvel = env.unwrapped.data.qvel.copy()
        qpos[:2] = rng.uniform(-3.0, 3.0, size=2)
        env.unwrapped.set_state(qpos, qvel)
        expected = env.unwrapped.get_body_com("fingertip")[:2]
        actual = reacher.fingertip(reacher.get_state(env))
        assert np.allclose(actual, expected, atol=1e-5)
    env.close()


def test_reward_matches_env():
    env = reacher.make_env()
    env.reset(seed=1)
    goal = torch.as_tensor(reacher.get_goal(env))
    rng = np.random.default_rng(1)
    for _ in range(30):
        action = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        _, reward, _, _, _ = env.step(action)
        next_state = torch.as_tensor(reacher.get_state(env))
        predicted = reacher.reward_fn(next_state, torch.as_tensor(action), goal).item()
        # 容差放到 5e-3 的原因见 reacher.reward_fn 的说明：仿真器算奖励用的指尖位置滞后半个控制步
        assert abs(predicted - reward) < 5e-3
    env.close()


def test_goal_is_fixed_within_episode():
    env = reacher.make_env()
    env.reset(seed=2)
    goal = reacher.get_goal(env)
    for _ in range(10):
        env.step(env.action_space.sample())
    assert np.allclose(reacher.get_goal(env), goal)
    env.close()


def test_random_episode_shapes():
    s, a, s_next, goal, total = reacher.run_random_episode(seed=3)
    assert s.shape == (reacher.EPISODE_LENGTH, 4)
    assert a.shape == (reacher.EPISODE_LENGTH, 2)
    assert s_next.shape == s.shape
    assert goal.shape == (2,)
    assert np.allclose(s[1:], s_next[:-1])  # 相邻两条转移首尾相接
    assert total < 0
