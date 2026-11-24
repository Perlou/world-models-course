"""
离线数据管线
============

⚠️ 草稿状态：本文件写成时运行环境不允许执行 Python，尚未实际运行过。
   定稿前必须完整跑一遍、核对输出，并把下面的「预计运行时间」换成实测值。

学习目标：
    1. 把一份外部数据（PushT 人类演示，LeRobot 逐帧格式）转成本课程约定的"每个回合一个文件"
    2. 给数据写单元测试：形状、数值、范围、结束标志、重复、时间对齐、可重放
    3. 按回合切分，并用实验看到"按转移切分"造成的泄漏
    4. 给数据集算内容指纹，知道它什么时候会变、什么时候不该变
    5. 建不跨回合的序列窗口索引；分别测数据吞吐和训练吞吐，判断瓶颈在哪

核心概念：
    - 对齐约定：下标 t 的那一行放 o_t、a_t、r_t；T 个动作配 T+1 个观测
    - 数据校验（data validation）：每一项是一条能自动运行的断言
    - 数据泄漏（data leakage）：同一回合的相邻样本分到了训练集和验证集两边
    - 数据集指纹（dataset fingerprint）：由数据内容算出的版本标识
    - 序列窗口（sequence window）：从一个回合里截出的连续 L 步，不跨回合

前置知识：
    - Phase 1 的 02-gymnasium-and-mdp.py（环境交互循环）
    - Phase 11 的 LeRobot 数据格式（这里只用到"每帧一行"这一点）

对应概念文档：
    - CONCEPT.md 第 3 节「数据管线」

预计运行时间：待实测（本稿尚未运行）
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from utils import is_quick, output_dir, save_figure, save_video, seed_everything, setup_matplotlib
from utils.pusht import load_pusht_demos

plt = setup_matplotlib()
OUT = output_dir("phase-12")
SEED = 0
SUFFIX = "-quick" if is_quick() else ""
PUSHT_DIR = OUT / f"01-pusht-episodes{SUFFIX}"
PENDULUM_DIR = OUT / f"01-pendulum-episodes{SUFFIX}"
# 这一节只有一个用来测吞吐的小卷积网络，放在 CPU 上就够了
DEVICE = torch.device("cpu")

PUSHT_ACTION_LOW, PUSHT_ACTION_HIGH = 0.0, 512.0   # 推杆目标位置的像素坐标范围
WINDOW = 16                                        # 序列窗口长度 L


# ==================== 第一部分：管线全景 ====================


def introduction():
    """数据管线的三件事：格式有约定，约定有校验，数据有版本。"""
    print("=" * 60)
    print("第一部分：数据管线要做什么")
    print("=" * 60)

    print("""
数据层的 bug 不会让程序崩溃。动作和观测错位一帧、验证集混进训练集，模型照样训练，
损失照样下降，只是结果差一截，而你会把原因归到模型上。
所以这一节不训练任何"正经"模型，只做一件事：把数据管线的每一站都配上能自动运行的检查。

    外部数据（PushT 演示）            仿真器（Pendulum）
          │ 转换                          │ 采集
          ▼                               ▼
    ┌──────────────────────────────────────────────┐
    │ 每个回合一个文件：save_episode()                │   约定：T 个动作，T+1 个观测
    └──────────────────────────────────────────────┘
          │
          ▼
    ┌──────────────────────────────────────────────┐
    │ 校验：validate_dataset()                       │   形状 / 有限 / 范围 / 长度 / 结束标志 / 重复
    │       check_action_alignment()                │   时间对齐
    │       check_replay()                          │   可重放（确定性仿真器）
    └──────────────────────────────────────────────┘
          │ 全部通过才往下走
          ▼
    ┌──────────────────────────────────────────────┐
    │ 定版：split_by_episode() → 归一化统计量（只用训练集）│
    │       dataset_fingerprint() → 写进 meta.json    │
    └──────────────────────────────────────────────┘
          │
          ▼
    ┌──────────────────────────────────────────────┐
    │ 使用：SequenceDataset                          │   窗口索引 [窗口总数, 2]，不跨回合
    │       一个批次：图像 [B, L+1, C, H, W]            │   uint8 存盘，取出后转 float32
    └──────────────────────────────────────────────┘

对齐约定（全课程通用）：下标 t 的那一行放的是在 o_t 时执行的 a_t 和因此得到的 r_t。

    观测   o_0    o_1    o_2   ...   o_{T-1}   o_T        共 T+1 个
    动作   a_0    a_1    a_2   ...   a_{T-1}              共 T 个
    奖励   r_0    r_1    r_2   ...   r_{T-1}              共 T 个
    """)


# ==================== 第二部分：按回合存盘 ====================


def save_episode(path, episode):
    """按约定把一个回合写成一个 .npz 文件（原子写入）。

    episode 是一个字典：
        observation.image  [T+1, H, W, C] uint8      可选（低维任务没有图像）
        observation.state  [T+1, S] float32
        action             [T, A] float32
        reward             [T] float32
        terminated         [T] bool                  执行 a_t 之后是否进入终止状态
        truncated          [T] bool                  执行 a_t 之后是否被截断
        success            标量 bool
    其余的键（例如仿真器的内部状态）原样保存。
    """
    n_steps = len(episode["action"])
    assert len(episode["observation.state"]) == n_steps + 1, "观测应比动作多一帧"
    assert len(episode["reward"]) == n_steps, "奖励应和动作一样多"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:                       # 传文件对象，NumPy 就不会自作主张地补扩展名
        np.savez_compressed(f, **episode)
    os.replace(tmp, path)                            # 先写临时文件再改名：要么完整，要么不存在


def load_episode(path):
    """读回一个回合，返回 NumPy 数组的字典。"""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def save_dataset(episodes, directory):
    """把一组回合写进一个目录，文件名 episode-0000.npz、episode-0001.npz ……"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("episode-*.npz"):      # 清掉上一次运行留下的文件，免得新旧混在一起
        old.unlink()
    for i, episode in enumerate(episodes):
        save_episode(directory / f"episode-{i:04d}.npz", episode)
    return sorted(directory.glob("episode-*.npz"))


def pusht_to_episodes(demos, n_episodes):
    """把 LeRobot 的逐帧表切成符合本课程约定的回合。

    LeRobot 的 PushT 数据每帧一行：第 i 行是第 i 帧的画面、推杆位置、这一帧发出的动作，
    以及"执行这个动作之后"的奖励（next.reward）。一个有 N 帧的回合有 N 个观测、N 个动作，
    最后一个动作执行之后的画面没有被记录。按本课程的约定（T 个动作配 T+1 个观测），
    取 T = N - 1：保留全部 N 帧观测，丢掉最后一个动作和它的奖励。
    """
    episodes = []
    starts, ends = demos["episode_starts"][:n_episodes], demos["episode_ends"][:n_episodes]
    for start, end in zip(starts, ends):
        start, end = int(start), int(end)
        n_steps = end - start - 1                                   # T
        truncated = np.zeros(n_steps, dtype=bool)
        truncated[-1] = True                                        # 记录到这一步为止
        episodes.append({
            "observation.image": demos["images"][start:end],        # [T+1, 96, 96, 3] uint8
            "observation.state": demos["states"][start:end],        # [T+1, 2] float32
            "action": demos["actions"][start:end - 1],              # [T, 2] float32
            "reward": demos["rewards"][start:end - 1],              # [T] float32
            "terminated": np.zeros(n_steps, dtype=bool),
            "truncated": truncated,
            "success": np.asarray(bool(demos["successes"][start:end].any())),
        })
    return episodes


def collect_pendulum_episodes(n_episodes, seed):
    """在 Pendulum-v1 里用随机策略采集回合，同时记下仿真器的内部状态（重放检查要用）。"""
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        observations, sim_states = [obs], [env.unwrapped.state.copy()]
        actions, rewards, terminated, truncated = [], [], [], []
        done = False
        while not done:
            action = rng.uniform(-2.0, 2.0, size=1).astype(np.float32)
            obs, reward, term, trunc, _ = env.step(action)
            observations.append(obs)
            sim_states.append(env.unwrapped.state.copy())
            actions.append(action)
            rewards.append(reward)
            terminated.append(term)
            truncated.append(trunc)
            done = term or trunc
        episodes.append({
            "observation.state": np.asarray(observations, dtype=np.float32),   # [T+1, 3]：cos θ、sin θ、θ̇
            "action": np.asarray(actions, dtype=np.float32),                   # [T, 1]
            "reward": np.asarray(rewards, dtype=np.float32),                   # [T]
            "terminated": np.asarray(terminated, dtype=bool),
            "truncated": np.asarray(truncated, dtype=bool),
            "success": np.asarray(False),                                      # 摆没有"成功"的定义
            # 仿真器内部的 (θ, θ̇)，只存在文件里、不进网络，所以保留仿真器自己的 float64
            "sim_state": np.asarray(sim_states, dtype=np.float64),             # [T+1, 2]
        })
    env.close()
    return episodes


def build_datasets(n_pusht, n_pendulum):
    """转换 PushT、采集 Pendulum，存盘后再从盘上读回来。"""
    print("\n" + "=" * 60)
    print("第二部分：按回合存盘")
    print("=" * 60)

    print("""
两份数据，两种来源：

    PushT 演示       外部数据。LeRobot 发布的人类遥操作演示，每帧一行。要先弄清它的对齐约定再转换
    Pendulum 随机    自己采集。按"reset(seed) → 循环 step → 记录"的标准流程，顺便存下仿真器内部状态
    """)

    start = time.time()
    demos = load_pusht_demos()
    total_episodes, total_frames = len(demos["episode_starts"]), len(demos["images"])
    print(f"  PushT 原始数据：{total_episodes} 个回合，{total_frames} 帧，"
          f"画面 {demos['images'].shape[1]}×{demos['images'].shape[2]}（读取用时 {time.time() - start:.1f} 秒）")
    pusht = pusht_to_episodes(demos, n_pusht)
    del demos
    start = time.time()
    pusht_paths = save_dataset(pusht, PUSHT_DIR)
    print(f"  已写入 {len(pusht_paths)} 个回合文件：{PUSHT_DIR.name}/（用时 {time.time() - start:.1f} 秒）")

    # 读回来，确认存取一来一回没有改变任何东西
    start = time.time()
    loaded = [load_episode(path) for path in pusht_paths]
    for key in pusht[0]:
        assert np.array_equal(pusht[0][key], loaded[0][key]), f"字段 {key} 存取前后不一致"
        assert pusht[0][key].dtype == loaded[0][key].dtype, f"字段 {key} 的类型变了"
    print(f"  从盘上读回全部回合用时 {time.time() - start:.1f} 秒；第 0 个回合逐字段核对一致")
    pusht = loaded

    pendulum = collect_pendulum_episodes(n_pendulum, seed=SEED)
    pendulum_paths = save_dataset(pendulum, PENDULUM_DIR)
    pendulum = [load_episode(path) for path in pendulum_paths]
    print(f"  Pendulum：采集并写入 {len(pendulum_paths)} 个回合，每个 {len(pendulum[0]['action'])} 步")

    first = pusht[0]
    print("\n  PushT 第 0 个回合的字段：")
    for key, value in first.items():
        print(f"    {key:<20} 形状 {str(value.shape):<18} 类型 {value.dtype}")

    lengths = np.array([len(ep["action"]) for ep in pusht])
    n_frames = int(sum(len(ep["observation.state"]) for ep in pusht))
    image_shape = first["observation.image"].shape[1:]
    bytes_uint8 = n_frames * int(np.prod(image_shape))
    bytes_disk = sum(path.stat().st_size for path in pusht_paths)
    # 手算例子（CONCEPT.md 第 3.3 节）：200 个回合 × 300 帧 × 96 × 96 × 3 × 1 字节
    assert 200 * 300 * 96 * 96 * 3 * 1 == 1_658_880_000
    print(f"""
存储的账（公式：回合数 × 每回合帧数 × H × W × C × 每个数的字节数）：

    回合数                 {len(pusht)}
    总帧数                 {n_frames}（回合长度 T：最短 {lengths.min()}，中位数 {int(np.median(lengths))}，最长 {lengths.max()}）
    分辨率                 {image_shape[0]}×{image_shape[1]}×{image_shape[2]}，uint8
    图像按 uint8 存         {bytes_uint8 / 1e6:8.1f} MB    ← {n_frames} × {int(np.prod(image_shape))} 字节
    图像按 float32 存       {bytes_uint8 * 4 / 1e6:8.1f} MB    ← 每个数 4 字节，正好 4 倍
    盘上实际占用（压缩后）   {bytes_disk / 1e6:8.1f} MB    ← np.savez_compressed，压缩到 uint8 大小的 {bytes_disk / bytes_uint8:.1%}

图像一律以 uint8 存盘，取出一个批次之后再转成 float32。压缩比取决于画面内容：
PushT 的画面是大片纯色，压得很小；真实相机的画面压缩比会差得多，预算要按未压缩的大小留。
    """)
    return pusht, pusht_paths, pendulum, pendulum_paths


# ==================== 第三部分：数据校验 ====================

CHECK_NAMES = ["形状与类型", "数值有限", "取值范围", "回合长度", "结束标志", "重复"]


def _schema_problem(ep):
    """检查一个回合的字段、类型、形状。没问题返回 None，否则返回一句说明。"""
    required = {"observation.state": ("float32", 2), "action": ("float32", 2), "reward": ("float32", 1),
                "terminated": ("bool", 1), "truncated": ("bool", 1)}
    for key, (dtype, ndim) in required.items():
        if key not in ep:
            return f"缺少字段 {key}"
        if ep[key].dtype != np.dtype(dtype):
            return f"{key} 的类型是 {ep[key].dtype}，应为 {dtype}"
        if ep[key].ndim != ndim:
            return f"{key} 是 {ep[key].ndim} 维，应为 {ndim} 维"
    n_steps = len(ep["action"])
    if len(ep["observation.state"]) != n_steps + 1:
        return f"观测 {len(ep['observation.state'])} 帧、动作 {n_steps} 个，观测应比动作多一帧"
    for key in ("reward", "terminated", "truncated"):
        if len(ep[key]) != n_steps:
            return f"{key} 的长度是 {len(ep[key])}，应为 {n_steps}"
    if "observation.image" in ep:
        image = ep["observation.image"]
        if image.dtype != np.uint8:
            return f"图像的类型是 {image.dtype}，应为 uint8"
        if image.ndim != 4 or image.shape[-1] != 3:
            return f"图像的形状是 {image.shape}，应为 [T+1, H, W, 3]"
        if len(image) != n_steps + 1:
            return f"图像 {len(image)} 帧，应为 {n_steps + 1} 帧"
    return None


def validate_dataset(episodes, action_low, action_high, min_length=2):
    """对一组回合运行 CONCEPT.md 第 3.5 节表里的通用断言。

    返回一个列表，每项是 (检查项, 是否通过, 说明)。
    时间对齐和可重放依赖具体任务的物理关系，分别由 check_action_alignment() 和 check_replay() 负责。
    """
    problems = {name: [] for name in CHECK_NAMES}
    seen = {}
    for i, ep in enumerate(episodes):
        schema = _schema_problem(ep)
        if schema:
            problems["形状与类型"].append(f"回合 {i}：{schema}")
            continue                                             # 形状都不对，后面的检查没有意义
        state, action = ep["observation.state"], ep["action"]

        if not (np.isfinite(state).all() and np.isfinite(action).all() and np.isfinite(ep["reward"]).all()):
            problems["数值有限"].append(f"回合 {i}：出现 NaN 或无穷大")

        n_out = int(((action < action_low) | (action > action_high)).sum())
        if n_out:
            problems["取值范围"].append(f"回合 {i}：{n_out} 个动作分量超出 [{action_low:g}, {action_high:g}]")
        if "observation.image" in ep and ep["observation.image"].min() == ep["observation.image"].max():
            problems["取值范围"].append(f"回合 {i}：图像是一片纯色")

        if len(action) < min_length:
            problems["回合长度"].append(f"回合 {i}：只有 {len(action)} 步")

        done = ep["terminated"] | ep["truncated"]
        if done[:-1].any():
            problems["结束标志"].append(f"回合 {i}：第 {int(np.argmax(done))} 步就出现了结束标志（共 {len(done)} 步）")

        digest = hashlib.sha1(action.tobytes() + state.tobytes()).hexdigest()
        if digest in seen:
            problems["重复"].append(f"回合 {i} 与回合 {seen[digest]} 内容完全相同")
        seen.setdefault(digest, i)

    report = []
    for name in CHECK_NAMES:
        found = problems[name]
        detail = f"{len(episodes)} 个回合全部通过" if not found else f"{len(found)} 处问题，例如：{found[0]}"
        report.append((name, not found, detail))
    return report


def print_report(report, indent="    "):
    for name, passed, detail in report:
        print(f"{indent}{'✅' if passed else '❌'} {name:<8} {detail}")


def check_action_alignment(episodes, shifts=(-1, 0, 1)):
    """时间对齐检查：在几种假设的对齐方式下，算"动作指向的方向"与"实际位移"的相关系数。

    PushT 的动作是推杆的目标位置，所以 (a − s) 应该和 (s' − s) 方向一致。
    对每个偏移 k，把 a_t 配给转移 (s_{t+k} → s_{t+k+1})：

        k =  0    a_t 配 (o_t → o_{t+1})          本课程的约定
        k = -1    a_t 配 (o_{t-1} → o_t)
        k = +1    a_t 配 (o_{t+1} → o_{t+2})

    返回 {k: 相关系数}。相关系数最高的那个 k 是数据真实的对齐方式。
    """
    result = {}
    for k in shifts:
        intent, moved = [], []
        for ep in episodes:
            state, action = ep["observation.state"], ep["action"]        # [T+1, 2]、[T, 2]
            n_steps = len(action)
            t = np.arange(max(0, -k), min(n_steps, n_steps - k))         # 保证 0 ≤ t+k 且 t+k+1 ≤ T
            intent.append(action[t] - state[t + k])                      # 目标位置 − 当前位置
            moved.append(state[t + k + 1] - state[t + k])                # 下一帧位置 − 当前位置
        intent, moved = np.concatenate(intent).ravel(), np.concatenate(moved).ravel()
        result[k] = float(np.corrcoef(intent, moved)[0, 1])
    return result


def check_replay(episodes, n_check=3):
    """可重放检查（确定性仿真器）：从记录的初始状态重放动作，看得到的观测和奖励与记录差多少。"""
    env = gym.make("Pendulum-v1")
    worst = 0.0
    for ep in episodes[:n_check]:
        env.reset(seed=0)
        env.unwrapped.state = ep["sim_state"][0].copy()                  # 把仿真器摆到记录的初始状态
        for t in range(len(ep["action"])):
            obs, reward, _, _, _ = env.step(ep["action"][t])
            worst = max(worst, float(np.abs(obs - ep["observation.state"][t + 1]).max()),
                        abs(float(reward) - float(ep["reward"][t])))
    env.close()
    return worst


def shift_actions(ep):
    """故障注入：动作整体提前一格（第 t 行放的其实是 a_{t+1}）。"""
    bad = dict(ep)
    bad["action"] = np.concatenate([ep["action"][1:], ep["action"][-1:]])
    return bad


def validation_experiments(pusht, pendulum):
    """先在真实数据上跑一遍校验，再故意把数据弄坏，看每种问题被哪一项检查抓到。"""
    print("\n" + "=" * 60)
    print("第三部分：数据校验")
    print("=" * 60)

    print("\n  PushT 演示数据：")
    pusht_report = validate_dataset(pusht, PUSHT_ACTION_LOW, PUSHT_ACTION_HIGH)
    print_report(pusht_report)
    print("\n  Pendulum 随机数据：")
    pendulum_report = validate_dataset(pendulum, -2.0, 2.0)
    print_report(pendulum_report)
    failed = [name for name, passed, _ in pusht_report + pendulum_report if not passed]
    if failed:
        print(f"\n  ⚠️ 真实数据上有没通过的检查项：{'、'.join(failed)}。这不是演示用的故障，是这份数据本身的情况。"
              "\n     按流程，这里应该先停下来弄清原因（是数据的问题，还是检查的阈值定得不对），再往下走。")
    else:
        print("\n  两份数据的通用检查全部通过。")

    # ---- 时间对齐 ----
    print("""
时间对齐检查。PushT 的动作是推杆的目标位置，所以"目标位置 − 当前位置"应该和
"下一帧位置 − 当前位置"方向一致。把 a_t 分别配给三段转移，算相关系数：
    """)
    labels = {0: "a_t 配 (o_t → o_{t+1})", -1: "a_t 配 (o_{t-1} → o_t)", 1: "a_t 配 (o_{t+1} → o_{t+2})"}
    corr = check_action_alignment(pusht)
    best = max(corr, key=corr.get)
    for k in (0, -1, 1):
        print(f"    {labels[k]:<28} 相关系数 {corr[k]:+.3f}{'   ← 最高' if k == best else ''}")
    runner_up = max(v for k, v in corr.items() if k != best)
    if best == 0:
        print(f"\n  最高的是本课程约定的对齐方式（{corr[0]:.3f}，第二名 {runner_up:.3f}）：转换后的数据对齐正确。")
    else:
        print(f"\n  ⚠️ 最高的不是 k = 0，而是 k = {best:+d}（{corr[best]:.3f} 对 {corr[0]:.3f}）。"
              "\n     这说明数据的对齐方式和转换时以为的不一样，pusht_to_episodes() 的切法需要修改。")

    shifted = [shift_actions(ep) for ep in pusht]
    corr_shifted = check_action_alignment(shifted)
    best_shifted = max(corr_shifted, key=corr_shifted.get)
    print("\n  故意把动作整体提前一格（模拟转换脚本里的差一错误）之后：")
    for k in (0, -1, 1):
        print(f"    {labels[k]:<28} 相关系数 {corr_shifted[k]:+.3f}{'   ← 最高' if k == best_shifted else ''}")
    if best_shifted != best:
        print(f"  最高的一行从 k = {best:+d} 移到了 k = {best_shifted:+d}：这项检查抓到了错位。"
              f"\n  注意错位之后 k = 0 的相关系数仍有 {corr_shifted[0]:.3f}。演示动作是平滑的，相邻两帧的动作差不多，"
              "\n  所以不能只看 k = 0 这一个数「够不够高」，要和相邻的两种错位比。")
    else:
        print("  ⚠️ 最高的一行没有变化：在这份数据上，这个相关系数分辨不出一格的错位。"
              "\n  这时要用退一步的办法：训练一个小的单步模型，比较几种对齐方式下的验证误差。")

    # ---- 可重放 ----
    replay_error = check_replay(pendulum)
    halved = []
    for ep in pendulum[:3]:
        bad = dict(ep)
        bad["action"] = ep["action"] / 2.0          # 模拟"存盘时把动作归一化到了 [-1, 1]，元数据里却写的是力矩"
        halved.append(bad)
    replay_error_bad = check_replay(halved)
    print(f"""
可重放检查（Pendulum，从记录的初始状态重放动作）：

    原始数据          观测与奖励的最大偏差 {replay_error:.2e}{'   ✅' if replay_error < 1e-5 else '   ❌'}
    动作被缩小一半     观测与奖励的最大偏差 {replay_error_bad:.2e}{'   ❌ 抓到了' if replay_error_bad > 1e-3 else '   没有抓到'}

注意第二行的数据能通过上面全部的通用检查（缩小一半的动作仍在 [-2, 2] 之内）。
只有重放能发现"数值看着合理，但含义不对"这一类问题。PushT 的演示数据做不了这项检查：
它没有记录 T 形块的初始位姿，没法把仿真器摆回去。自己采集数据时，记得把仿真器的内部状态一起存下来。
    """)

    # ---- 故障注入 ----
    base = pusht[:8]
    glued = {key: (np.concatenate([base[0][key], base[1][key][1:]]) if key.startswith("observation")
                   else np.concatenate([base[0][key], base[1][key]]))
             for key in base[0] if key != "success"}
    glued["success"] = base[0]["success"]
    with_nan = dict(base[2])
    with_nan["action"] = base[2]["action"].copy()
    with_nan["action"][3, 0] = np.nan
    float_image = dict(base[3])
    float_image["observation.image"] = base[3]["observation.image"].astype(np.float32) / 255.0
    short_obs = dict(base[4])
    short_obs["observation.state"] = base[4]["observation.state"][:-1]
    short_obs["observation.image"] = base[4]["observation.image"][:-1]
    doubled = dict(base[5])
    doubled["action"] = base[5]["action"] * 2.0
    faults = {
        "两个回合粘在一起": [glued] + base[2:],
        "同一个回合存了两次": base + [base[0]],
        "动作里混进一个 NaN": base[:2] + [with_nan] + base[3:],
        "图像存成了 float32": base[:3] + [float_image] + base[4:],
        "少存了最后一帧观测": base[:4] + [short_obs] + base[5:],
        "动作的单位错了（放大一倍）": base[:5] + [doubled] + base[6:],
        "动作错位一格": [shift_actions(ep) for ep in base],
    }
    print("  故障注入：取 8 个回合，每次只弄坏一处，看通用检查里哪一项报错。\n")
    print(f"    {'注入的故障':<22}{'通用检查抓到它的项'}")
    silent = []
    for name, episodes in faults.items():
        caught = [check for check, passed, _ in validate_dataset(episodes, PUSHT_ACTION_LOW, PUSHT_ACTION_HIGH) if not passed]
        if not caught:
            silent.append(name)
        print(f"    {name:<22}{'、'.join(caught) if caught else '（没有任何一项报错）'}")
    if silent:
        print(f"\n  通用检查漏掉了：{'、'.join(silent)}。"
              "\n  这类问题的数值、形状都合法，只有利用任务本身的物理关系（上面的对齐检查、重放检查）才能发现。")
    else:
        print("\n  这次每种故障都被通用检查抓到了。")
    return corr, corr_shifted, replay_error


def visual_check(pusht):
    """人眼检查：抽几个回合，把画面、推杆位置、动作目标画在一起；另存一段视频。"""
    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(pusht), size=min(3, len(pusht)), replace=False)
    n_cols = 6
    fig, axes = plt.subplots(len(picks), n_cols, figsize=(2.2 * n_cols, 2.4 * len(picks)), squeeze=False)
    for row, e in enumerate(picks):
        ep = pusht[e]
        n_steps = len(ep["action"])
        scale = ep["observation.image"].shape[1] / PUSHT_ACTION_HIGH       # 像素坐标 0–512 → 画面 0–96
        for col, t in enumerate(np.linspace(0, n_steps - 1, n_cols).astype(int)):
            ax = axes[row, col]
            ax.imshow(ep["observation.image"][t])
            ax.plot(*(ep["observation.state"][t] * scale), "o", color="cyan", markersize=5, label="推杆位置 s_t")
            ax.plot(*(ep["action"][t] * scale), "x", color="red", markersize=7, markeredgewidth=2, label="动作目标 a_t")
            ax.set_title(f"回合 {e}，t = {t}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    axes[0, 0].legend(fontsize=6, loc="lower left")
    fig.suptitle("人眼检查：动作目标（红叉）应该落在推杆（青点）即将前往的方向上", fontsize=11)
    save_figure(fig, OUT / "01-pusht-contact-sheet.png")

    ep = pusht[int(picks[0])]
    frames = [np.repeat(np.repeat(frame, 3, axis=0), 3, axis=1) for frame in ep["observation.image"]]   # 放大 3 倍
    save_video(frames, OUT / "01-pusht-episode.mp4", fps=10)
    print("  断言想不到的问题（画面上下颠倒、通道顺序反了、动作坐标系不对）要靠人眼看一遍。上面的图和视频就是为此存的。")


# ==================== 第四部分：切分、泄漏、归一化 ====================


def split_by_episode(n_episodes, fractions=(0.8, 0.1, 0.1), seed=0):
    """固定种子的按回合切分。返回 {"train": 回合编号数组, "val": ..., "test": ...}，编号已排序。"""
    order = np.random.default_rng(seed).permutation(n_episodes)
    n_train = int(round(fractions[0] * n_episodes))
    n_val = int(round(fractions[1] * n_episodes))
    return {"train": np.sort(order[:n_train]),
            "val": np.sort(order[n_train:n_train + n_val]),
            "test": np.sort(order[n_train + n_val:])}


def nearest_neighbor_error(x_train, y_train, x_val, y_val, chunk=2048):
    """最近邻"模型"：验证样本的预测 = 离它最近的训练样本的标签。返回均方根误差和最近邻下标。"""
    x_train_t = torch.from_numpy(x_train)
    nearest = []
    for i in range(0, len(x_val), chunk):
        dist = torch.cdist(torch.from_numpy(x_val[i:i + chunk]), x_train_t)      # [chunk, N_train]
        nearest.append(dist.argmin(dim=1))
    nearest = torch.cat(nearest).numpy()
    rmse = float(np.sqrt(((y_train[nearest] - y_val) ** 2).sum(axis=1).mean()))
    return rmse, nearest


def split_and_leakage(pusht):
    """按回合切分；用最近邻模型演示按转移切分造成的泄漏；归一化统计量只从训练集算。"""
    print("\n" + "=" * 60)
    print("第四部分：切分、泄漏、归一化")
    print("=" * 60)

    split = split_by_episode(len(pusht), seed=SEED)
    all_ids = np.concatenate(list(split.values()))
    assert len(all_ids) == len(pusht) == len(set(all_ids.tolist())), "切分应当不重不漏"
    print(f"\n  按回合切分（种子 {SEED}）：训练 {len(split['train'])} 个回合，"
          f"验证 {len(split['val'])} 个，测试 {len(split['test'])} 个。名单会写进 meta.json。")

    print("""
泄漏实验。任务：由 (s_{t-1}, s_t, a_t) 预测推杆的位移 s_{t+1} − s_t。
"模型"用最近邻：验证样本的预测就是离它最近的那条训练样本的位移。
最近邻是"纯靠记忆"的极端情形，容量大的神经网络训练久了也会往这个方向走。
比较两种切分方式下的验证误差：
    """)
    x, y, episode_id = [], [], []
    for e, ep in enumerate(pusht):
        state, action = ep["observation.state"], ep["action"]
        t = np.arange(1, len(action))
        x.append(np.concatenate([state[t - 1], state[t], action[t]], axis=1) / PUSHT_ACTION_HIGH)   # [T-1, 6]
        y.append(state[t + 1] - state[t])                                                           # [T-1, 2]，像素
        episode_id.append(np.full(len(t), e))
    x, y, episode_id = np.concatenate(x).astype(np.float32), np.concatenate(y), np.concatenate(episode_id)

    # 切分一：把所有转移打乱后按 8:2 分
    order = np.random.default_rng(SEED).permutation(len(x))
    n_train = int(0.8 * len(x))
    tr, va = order[:n_train], order[n_train:]
    rmse_transition, nearest = nearest_neighbor_error(x[tr], y[tr], x[va], y[va])
    same_episode = float((episode_id[tr][nearest] == episode_id[va]).mean())

    # 切分二：按回合分（训练回合 对 验证 + 测试回合）
    is_train = np.isin(episode_id, split["train"])
    rmse_episode, _ = nearest_neighbor_error(x[is_train], y[is_train], x[~is_train], y[~is_train])
    rmse_zero = float(np.sqrt((y[~is_train] ** 2).sum(axis=1).mean()))

    print(f"    {'切分方式':<16}{'验证样本数':>10}{'验证误差（位移的均方根，像素）':>22}")
    print(f"    {'按转移随机切分':<16}{len(va):>10}{rmse_transition:>22.2f}")
    print(f"    {'按回合切分':<18}{int((~is_train).sum()):>10}{rmse_episode:>22.2f}")
    print(f"    {'参照：一律预测不动':<15}{int((~is_train).sum()):>10}{rmse_zero:>22.2f}")
    print(f"\n  按转移切分时，{same_episode:.0%} 的验证样本，它的最近邻来自同一个回合（多半就是前后相邻的那一帧）。")
    if rmse_transition < rmse_episode:
        print(f"  按转移切分的验证误差只有按回合切分的 {rmse_transition / rmse_episode:.0%}。模型没有变，数据没有变，"
              "\n  低出来的那一截全部来自泄漏：验证集里的样本，训练集里几乎都有一个孪生兄弟。"
              "\n  部署时遇到的是全新的回合，对应的是「按回合切分」那一行的数字。")
    else:
        print("  ⚠️ 这次运行里按转移切分的验证误差没有更低。泄漏依然存在（见上面的同回合比例），"
              "\n  只是在这个特征和这个数据量下，别的回合里也能找到同样近的邻居，所以没有体现在误差上。")

    # 归一化统计量：只从训练回合算
    def stats(ids):
        state = np.concatenate([pusht[i]["observation.state"] for i in ids])
        action = np.concatenate([pusht[i]["action"] for i in ids])
        return {"state_mean": state.mean(0), "state_std": state.std(0), "action_mean": action.mean(0), "action_std": action.std(0)}

    train_stats, all_stats = stats(split["train"]), stats(np.arange(len(pusht)))
    gap = max(float(np.abs(train_stats[key] - all_stats[key]).max()) for key in train_stats)
    print(f"""
归一化统计量只从训练回合算：

    推杆位置的均值 {np.round(train_stats['state_mean'], 1)}，标准差 {np.round(train_stats['state_std'], 1)}（像素）
    动作的均值     {np.round(train_stats['action_mean'], 1)}，标准差 {np.round(train_stats['action_std'], 1)}

和"用全部回合算"相比，各项最大只差 {gap:.2f} 像素。数值上的差别很小，但规矩不能破：
统计量一旦用到了验证集和测试集，这两个集合就不再是"模型没见过的数据"。
    """)
    return split, train_stats, (rmse_transition, rmse_episode, same_episode)


# ==================== 第五部分：数据集指纹 ====================


def file_sha256(path):
    """文件字节的哈希。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(path):
    """一个回合文件里全部数组内容的哈希（键名、类型、形状、数据），以及数组的总字节数。"""
    digest, n_bytes = hashlib.sha256(), 0
    with np.load(path) as data:
        for key in sorted(data.files):
            array = np.ascontiguousarray(data[key])
            digest.update(f"{key}|{array.dtype.str}|{array.shape}|".encode())
            digest.update(array.tobytes())
            n_bytes += array.nbytes
    return digest.hexdigest(), n_bytes


def dataset_fingerprint(directory, pattern="episode-*.npz"):
    """数据集指纹：对目录下每个回合文件算内容哈希，再把（相对路径，字节数，哈希）拼起来哈希一次。

    返回前 12 位十六进制字符。哈希的是解码后的数组内容，不是文件的原始字节（原因见 fingerprint_demo）。
    """
    directory = Path(directory)
    lines = []
    for path in sorted(directory.glob(pattern)):
        digest, n_bytes = array_digest(path)
        lines.append(f"{path.relative_to(directory).as_posix()}\t{n_bytes}\t{digest}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]


def fingerprint_demo(pendulum, pendulum_paths):
    """指纹应该在内容不变时不变、内容变了时变。"""
    print("\n" + "=" * 60)
    print("第五部分：数据集指纹")
    print("=" * 60)

    start = time.time()
    pusht_fingerprint = dataset_fingerprint(PUSHT_DIR)
    print(f"\n  PushT 回合数据的指纹：{pusht_fingerprint}（计算用时 {time.time() - start:.1f} 秒）")

    fp_original = dataset_fingerprint(PENDULUM_DIR)
    bytes_original = file_sha256(pendulum_paths[0])

    # 实验一：把同样的内容重新存一遍（隔两秒多，跨过 zip 时间戳的最小刻度）
    copy_dir = OUT / f"01-pendulum-copy{SUFFIX}"
    time.sleep(2.1)
    copy_paths = save_dataset(pendulum, copy_dir)
    fp_copy = dataset_fingerprint(copy_dir)
    bytes_copy = file_sha256(copy_paths[0])

    # 实验二：只改一个回合里的一个动作，改动量 0.001
    tampered = dict(pendulum[1])
    tampered["action"] = pendulum[1]["action"].copy()
    tampered["action"][7, 0] += 1e-3
    save_episode(copy_paths[1], tampered)
    fp_tampered = dataset_fingerprint(copy_dir)

    print(f"""
在 Pendulum 的小数据集上做两个实验：

    原始目录                       指纹 {fp_original}    第 0 个文件的字节哈希 {bytes_original[:12]}
    同样的内容重新存一遍             指纹 {fp_copy}    第 0 个文件的字节哈希 {bytes_copy[:12]}
    再把一个动作改动 0.001           指纹 {fp_tampered}
    """)
    assert fp_copy == fp_original, "内容相同，指纹必须相同"
    assert fp_tampered != fp_original, "内容变了，指纹必须变"
    if bytes_copy != bytes_original:
        print("  重新存一遍之后，内容指纹没变，文件的字节哈希却变了：.npz 是 zip 容器，里面记着写入时间。"
              "\n  所以指纹要对解码后的数组内容算。直接哈希文件字节的话，同一份数据每重新导出一次就换一个指纹，"
              "\n  「两次实验用的是不是同一份数据」这个问题反而回答不了。")
    else:
        print("  这次重新存一遍之后，文件的字节哈希也没有变。即便如此，对解码后的数组内容算指纹仍是更稳妥的做法："
              "\n  容器格式里的时间戳、压缩级别这些与内容无关的东西，不应该影响数据的版本标识。")
    print("  改动一个数的千分之一，指纹就变了。数据被用过之后不再原地修改；要改就生成一份新的，得到新的指纹。")
    return pusht_fingerprint


# ==================== 第六部分：序列窗口 ====================


def count_windows(lengths, window):
    """按回合计算合法窗口数：每个回合 max(T − L + 1, 0) 个。"""
    return int(sum(max(n_steps - window + 1, 0) for n_steps in lengths))


class SequenceDataset:
    """序列窗口数据集：窗口不跨回合；取批次时才把 uint8 转成 float32 并归一化。

    episodes     回合列表（已读进内存的字典）。paths 不为 None 时改为每次取样都从盘上读文件
    window       窗口长度 L。一个样本含 L 个动作、L+1 个观测
    episode_ids  这个数据集可以用哪些回合（训练集、验证集各建一个）
    stats        归一化统计量（只从训练回合算出来的那一份）
    """

    def __init__(self, episodes, window, episode_ids, stats, paths=None):
        self.episodes, self.window, self.stats, self.paths = episodes, window, stats, paths
        index = [(e, t) for e in episode_ids for t in range(max(len(episodes[e]["action"]) - window + 1, 0))]
        self.index = np.asarray(index, dtype=np.int64).reshape(-1, 2)        # [窗口总数, 2]：（回合编号，起始步）

    def __len__(self):
        return len(self.index)

    def _episode(self, e):
        return self.episodes[e] if self.paths is None else load_episode(self.paths[e])

    def sample(self, batch_size, rng):
        """随机取一个批次。返回的张量都是 float32。"""
        picks = self.index[rng.integers(0, len(self.index), size=batch_size)]
        images, states, actions, rewards = [], [], [], []
        for e, t in picks:
            ep = self._episode(int(e))
            images.append(ep["observation.image"][t:t + self.window + 1])    # [L+1, H, W, C] uint8
            states.append(ep["observation.state"][t:t + self.window + 1])
            actions.append(ep["action"][t:t + self.window])
            rewards.append(ep["reward"][t:t + self.window])
        s = self.stats
        image = torch.from_numpy(np.stack(images)).permute(0, 1, 4, 2, 3).float() / 255.0    # [B, L+1, C, H, W]
        state = torch.from_numpy((np.stack(states) - s["state_mean"]) / s["state_std"]).float()      # [B, L+1, S]
        action = torch.from_numpy((np.stack(actions) - s["action_mean"]) / s["action_std"]).float()  # [B, L, A]
        reward = torch.from_numpy(np.stack(rewards)).float()                                         # [B, L]
        return {"image": image, "state": state, "action": action, "reward": reward}


def window_demo(pusht, split, stats):
    """窗口计数：按回合建索引，和"首尾相接再截"对比。"""
    print("\n" + "=" * 60)
    print("第六部分：序列窗口")
    print("=" * 60)

    # 手算例子（CONCEPT.md 第 3.3 节）：三个回合 100、150、80 步，L = 64
    assert count_windows([100, 150, 80], 64) == 141
    assert (100 + 150 + 80) - 64 + 1 == 267

    lengths = [len(pusht[e]["action"]) for e in split["train"]]
    dataset = SequenceDataset(pusht, WINDOW, split["train"], stats)
    correct = count_windows(lengths, WINDOW)
    naive = sum(lengths) - WINDOW + 1
    assert len(dataset) == correct, "窗口索引的条数应等于按回合计算的公式"
    too_short = sum(1 for n_steps in lengths if n_steps < WINDOW)
    batch = dataset.sample(4, np.random.default_rng(SEED))
    print(f"""
训练集 {len(lengths)} 个回合，共 {sum(lengths)} 步，窗口长度 L = {WINDOW}：

    按回合建索引（每个回合 max(T − L + 1, 0) 个）     {correct} 个窗口
    把所有回合首尾相接再截                            {naive} 个窗口
    多出来的                                         {naive - correct} 个，占 {(naive - correct) / naive:.1%}

多出来的每一个窗口都跨过了回合边界：前半段是上一个演示的结尾，后半段是下一个演示的开头，
中间有一次推杆和 T 形块的"瞬移"。模型会花容量去拟合这些不存在的转移，而且不会有任何报错。
（训练集里比窗口短、一个窗口也贡献不了的回合有 {too_short} 个。）

一个批次（B = 4）的张量：
    图像   {tuple(batch['image'].shape)}   [B, L+1, C, H, W]，{batch['image'].dtype}，取值 {batch['image'].min():.2f} – {batch['image'].max():.2f}
    状态   {tuple(batch['state'].shape)}            [B, L+1, S]，已归一化
    动作   {tuple(batch['action'].shape)}            [B, L, A]，已归一化
    奖励   {tuple(batch['reward'].shape)}               [B, L]
    """)
    return dataset, correct, naive


# ==================== 第七部分：加载速度 ====================


class TinyImageDynamics(nn.Module):
    """输入一帧画面、推杆位置和动作，预测推杆位置的变化量。只用来测训练吞吐。"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),                           # [N, 32 × 4 × 4]
        )
        self.head = nn.Sequential(nn.Linear(512 + 4, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, image, state, action):
        """image [N, C, H, W]，state [N, 2]，action [N, 2] → 位置变化量 [N, 2]。"""
        return self.head(torch.cat([self.encoder(image), state, action], dim=-1))


def train_step(model, optimizer, batch):
    image, state, action = batch["image"], batch["state"], batch["action"]
    b, l = action.shape[:2]
    pred = model(image[:, :-1].reshape(b * l, *image.shape[2:]), state[:, :-1].reshape(b * l, -1), action.reshape(b * l, -1))
    loss = ((pred - (state[:, 1:] - state[:, :-1]).reshape(b * l, -1)) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def measure_data_throughput(dataset, batch_size, n_batches):
    """只取批次，不做前向和反向：每秒多少个批次。"""
    rng = np.random.default_rng(SEED)
    dataset.sample(batch_size, rng)                                          # 预热一次
    start = time.perf_counter()
    for _ in range(n_batches):
        dataset.sample(batch_size, rng)
    return n_batches / (time.perf_counter() - start)


def throughput_experiment(pusht, pusht_paths, split, stats, dataset):
    """数据吞吐（从盘上读 / 已读进内存）对训练吞吐。"""
    print("\n" + "=" * 60)
    print("第七部分：瓶颈在数据还是在模型")
    print("=" * 60)

    batch_size = 16
    n_lazy, n_cached, n_train = (3, 20, 6) if is_quick() else (15, 200, 40)
    lazy = SequenceDataset(pusht, WINDOW, split["train"], stats, paths=pusht_paths)
    data_lazy = measure_data_throughput(lazy, batch_size, n_lazy)
    data_cached = measure_data_throughput(dataset, batch_size, n_cached)

    torch.manual_seed(SEED)
    model = TinyImageDynamics().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED)
    fixed = dataset.sample(batch_size, rng)
    for _ in range(3):                                                       # 预热：头几步有初始化开销，不计时
        train_step(model, optimizer, fixed)
    start = time.perf_counter()
    for _ in range(n_train):
        train_step(model, optimizer, fixed)                                  # 固定一个批次：只测模型这一半
    model_only = n_train / (time.perf_counter() - start)
    start = time.perf_counter()
    for _ in range(n_train):
        train_step(model, optimizer, dataset.sample(batch_size, rng))        # 完整的训练循环
    full_loop = n_train / (time.perf_counter() - start)

    print(f"""
批大小 {batch_size}，窗口长度 {WINDOW}（每个批次 {batch_size * (WINDOW + 1)} 帧画面），设备 {DEVICE}：

    数据吞吐：每次取样都从盘上读回合文件     {data_lazy:8.1f} 批 / 秒
    数据吞吐：回合已全部读进内存             {data_cached:8.1f} 批 / 秒     是从盘上读的 {data_cached / data_lazy:.0f} 倍
    训练吞吐：固定一个批次，只算模型         {model_only:8.1f} 步 / 秒
    训练吞吐：完整循环（内存取样 + 模型）     {full_loop:8.1f} 步 / 秒
    """)
    if data_lazy < model_only:
        print(f"  如果每次取样都从盘上读文件，数据（{data_lazy:.1f} 批 / 秒）比模型（{model_only:.1f} 步 / 秒）还慢："
              "\n  这时瓶颈在数据，换再快的显卡也没用。")
    else:
        print(f"  即使每次取样都从盘上读文件，数据（{data_lazy:.1f} 批 / 秒）也比这个模型（{model_only:.1f} 步 / 秒）快，"
              "\n  在本机 CPU 上瓶颈在模型。换到快得多的显卡上，模型这一半会变快，从盘上读就可能反过来成为瓶颈。")
    if data_cached > 5 * model_only:
        print(f"  读进内存之后，数据吞吐是模型的 {data_cached / model_only:.0f} 倍，完整循环（{full_loop:.1f} 步 / 秒）"
              f"和只算模型（{model_only:.1f} 步 / 秒）几乎一样快：瓶颈完全在模型。")
    else:
        print(f"  读进内存之后，数据吞吐是模型的 {data_cached / model_only:.1f} 倍，两者在同一个量级，取样的开销在完整循环里仍然看得见。")
    print("  这个数据集的图像只有几百 MB，整个读进内存是最省事的提速办法。数据放不进内存时，再考虑预先缩放、预计算特征和多进程加载。")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    names = ["数据：从盘上读", "数据：读进内存", "训练：只算模型", "训练：完整循环"]
    values = [data_lazy, data_cached, model_only, full_loop]
    bars = ax.barh(names, values, color=["C3", "C2", "C0", "C0"])
    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.1f}", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("每秒批次数 / 步数（对数坐标）")
    ax.set_title(f"数据吞吐与训练吞吐（批大小 {batch_size}，L = {WINDOW}，{DEVICE}）")
    ax.invert_yaxis()
    save_figure(fig, OUT / "01-throughput.png")
    return data_lazy, data_cached, model_only, full_loop


def write_metadata(pusht, split, stats, fingerprint):
    """把元数据写在数据旁边。"""
    meta = {
        "source": "lerobot/pusht（人类遥操作演示），由 01-offline-data-pipeline.py 转换",
        "environment": "gym_pusht/PushT-v0",
        "control_frequency_hz": 10,                                          # 来自 LeRobot 元信息里的 fps
        "observation": {"image": "96×96×3 uint8", "state": "推杆位置 [2]，像素坐标 0–512"},
        "action": {"meaning": "推杆的目标位置（不是速度增量）", "low": PUSHT_ACTION_LOW, "high": PUSHT_ACTION_HIGH},
        "collection_policy": "人类遥操作",
        "alignment": "下标 t 的行放 o_t、a_t、r_t；T 个动作，T+1 个观测；每个回合丢掉了原始数据的最后一个动作",
        "n_episodes": len(pusht),
        "n_steps": int(sum(len(ep["action"]) for ep in pusht)),
        "split_seed": SEED,
        "split": {name: ids.tolist() for name, ids in split.items()},
        "normalization": {key: np.asarray(value, dtype=np.float64).round(4).tolist() for key, value in stats.items()},
        "fingerprint": fingerprint,
    }
    path = PUSHT_DIR / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  元数据已写入 {PUSHT_DIR.name}/meta.json：环境、控制频率、动作含义、采集方式、切分名单、归一化统计量、指纹 {fingerprint}。")


# ==================== 练习与思考 ====================


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：给 Pendulum 写一个时间对齐检查
    PushT 的检查用的是"动作是目标位置"这条关系。摆的动作是力矩，关系换成：
    角速度的变化量 θ̇_{t+1} − θ̇_t 减去重力项 15 sin θ_t × 0.05，应该正比于力矩 a_t（比例 3 × 0.05）。
    仿照 check_action_alignment()，对 k = -1、0、+1 算相关系数。

练习 1 参考答案：
    用回合里存的 sim_state（θ, θ̇）：
        residual[t] = (θ̇[t+1] − θ̇[t]) − 15 × sin(θ[t]) × 0.05
    再算 corrcoef(a[t−k], residual[t])。随机策略每一步的力矩相互独立，
    所以对齐正确时相关系数接近 1（角速度撞到上限 8 被裁剪的那几步除外），错位一格时接近 0。
    这比 PushT 的情形好分辨得多：人类演示的动作是平滑的，错位一格之后相关系数只掉一点。
    采集数据用的策略越随机，时间对齐越容易验证。

练习 2：把窗口长度改成 64
    把 WINDOW 改成 64 重新运行第六部分。合法窗口数怎么变？有多少回合一个窗口也贡献不了？

练习 2 参考答案：
    每个回合的窗口数是 max(T − L + 1, 0)，L 变大，每个回合少 48 个窗口，
    长度不到 64 的回合整个被丢掉。PushT 的演示长短不一，窗口越长，
    训练数据越偏向"长的演示"（往往是推得不顺、反复调整的那些）。
    这是一种不显眼的采样偏差：用长窗口之前，先看一眼回合长度的分布。

练习 3：让 validate_dataset() 多查一项
    加一项"成功标志与奖励一致"：success 为真的回合，最后几步的奖励应该接近这个回合的最大值。

练习 3 参考答案：
    对每个 success 为真的回合，断言 reward[-5:].max() >= 0.9 × reward.max()。
    这类检查利用的是"字段之间应有的关系"。字段各自合法不代表它们放在一起说得通：
    成功标志和奖励错开了回合、图像和状态来自不同的帧，都属于这一类。

思考题 1：为什么按回合切分，而不是按"任务的初始条件"切分？

思考题 1 参考答案：
    按回合切分防的是"相邻帧泄漏"。如果数据里有多个回合从几乎相同的初始条件出发
    （比如脚本采集时只有 10 种起点），按回合切分之后，验证集的回合在训练集里仍有近亲。
    这时要按更粗的单位切：按初始条件、按物体、按场景。原则是：
    切分的单位要和"部署时会变的东西"一致。

思考题 2：指纹为什么不把 meta.json 也算进去？

思考题 2 参考答案：
    指纹是写在 meta.json 里的，把它算进去就成了"自己包含自己的哈希"。
    更重要的是分工：指纹回答"数据内容是不是同一份"，切分名单和归一化统计量是从数据派生出来的，
    它们由（指纹，切分种子）决定。实验记录里把这两样都记下来即可。

思考题 3：把冻结编码器的特征预先算好存盘，缓存的键应该包含什么？

思考题 3 参考答案：
    原始数据的指纹、编码器的名称与权重版本、预处理参数（分辨率、裁剪、归一化）。
    少了任何一项，都可能出现"换了编码器却读到旧缓存"的情况，而且不会报错：
    特征的形状往往一样，只是数值来自另一个模型。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()

    n_pusht, n_pendulum = (24, 4) if is_quick() else (None, 10)             # None：全部回合
    pusht, pusht_paths, pendulum, pendulum_paths = build_datasets(n_pusht, n_pendulum)
    validation_experiments(pusht, pendulum)
    visual_check(pusht)
    split, stats, _ = split_and_leakage(pusht)
    fingerprint = fingerprint_demo(pendulum, pendulum_paths)
    dataset, _, _ = window_demo(pusht, split, stats)
    throughput_experiment(pusht, pusht_paths, split, stats, dataset)
    write_metadata(pusht, split, stats, fingerprint)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：02-scaling-up-training.py（把训练变成可中断、可重建的作业）")
    print("=" * 60)


if __name__ == "__main__":
    main()
