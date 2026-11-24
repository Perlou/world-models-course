"""
从状态到像素
============

学习目标：
    1. 认识 CarRacing-v3：观测是 96×96 的画面，动作是 3 维连续量，奖励按"压过的新地砖"发放
    2. 用一个能手算的例子看清：像素空间里的距离没有语义
    3. 写出画面预处理：裁掉仪表条 → 缩放到 64×64 → 归一化
    4. 理解随机策略为什么要带时间相关性，并用"压过的地砖比例"量出差别
    5. 采集一份随机策略的数据集，后面训练 V（VAE）和 M（MDN-RNN）都用它

核心概念：
    - 观测（observation）与状态（state）：这一章起环境不再把状态交到我们手上，只给画面
    - 表示（representation）：把一帧画面压成一个小向量 z，这是下一节 VAE 的任务
    - 部分可观测：单帧画面里没有车速，下一帧不由"当前帧 + 动作"唯一决定
    - 覆盖（coverage）：随机策略到不了的地方，数据集里就没有

前置知识：
    - Phase 1 的 02-gymnasium-and-mdp.py（交互循环、采集转移数据）

对应概念文档：
    - CONCEPT.md 第 1 节「从状态到像素：为什么需要学表示」

预计运行时间：约 2.5 分钟（CPU，4 个进程并行采集）
"""

import os
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import gymnasium as gym
import numpy as np

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-02")
SEED = 0
ZOOM_FRAMES = 50      # 每个回合开头约 1 秒是镜头拉近的动画，这段画面不采
CROP_ROWS = 84        # 原始画面 96 行，底部 12 行是仪表条，只保留上面 84 行
IMAGE_SIZE = 64
N_WORKERS = 4
DATASET_PATH = OUT / ("carracing-random-quick.npz" if is_quick() else "carracing-random.npz")


# ==================== 第一部分：环境不再给状态了 ====================


def introduction():
    """从状态到像素，问题变在哪。"""
    print("=" * 60)
    print("第一部分：环境不再给状态了")
    print("=" * 60)

    print("""
Phase 1 里，环境直接把状态交给我们：角度、角速度，两个数，含义明确。
真实的机器人没有这个待遇，它只有摄像头。这一章换成只给画面的环境：

    Phase 1：Pendulum-v1                  本章：CarRacing-v3
    ┌──────────────────────┐              ┌──────────────────────────┐
    │ 观测 = 状态            │              │ 观测 = 96×96×3 的俯视画面   │
    │ [cos θ, sin θ, θ̇]     │              │ 27648 个数，大部分是草地    │
    │ 3 个数，全都有用        │              │ 车速、角速度不在画面里      │
    └──────────────────────┘              └──────────────────────────┘

画面带来三个麻烦：大、冗余、不完整。本章的办法是把问题拆成三块（论文叫 V、M、C）：

    画面 o_t ──▶ ┌─────┐  z_t   ┌─────┐  h_t   ┌─────┐
                 │  V  │──────▶│  M  │──────▶│  C  │──▶ 动作 a_t
                 │ VAE │        │ RNN │        │ 线性 │
                 └─────┘        └─────┘        └─────┘
                 看：压缩画面     记：预测下一个 z   做：极简控制器

这一节只做准备工作：认识环境、写预处理、采一份数据。
    """)


def make_env():
    """创建 CarRacing-v3。不开窗口，画面直接从 step() 返回的观测里拿。"""
    return gym.make("CarRacing-v3", continuous=True)


def reset_env(env, seed):
    """重置环境并跳过开头的镜头动画，返回动画结束后的第一帧。

    回合开始的约 50 帧里镜头从远处拉近，画面和正常行驶时完全不同。
    这里用"不踩油门不打方向"的动作把这段跳过去。后面每一节讲义都沿用这个约定。
    """
    frame, _ = env.reset(seed=seed)
    for _ in range(ZOOM_FRAMES):
        frame, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    return frame


def inspect_env():
    """打印观测、动作、回合长度，核对 CONCEPT.md 第 1.2 节的表。"""
    print("\n" + "=" * 60)
    print("第二部分：认识 CarRacing-v3")
    print("=" * 60)

    env = make_env()
    first, _ = env.reset(seed=SEED)
    print(f"\n观测空间：形状 {env.observation_space.shape}，类型 {env.observation_space.dtype}，"
          f"取值 {int(env.observation_space.low.min())}–{int(env.observation_space.high.max())}")
    print(f"动作空间：形状 {env.action_space.shape}，下界 {env.action_space.low}，上界 {env.action_space.high}")
    print("          三个分量依次是：方向（-1 左，+1 右）、油门、刹车")
    print(f"回合步数上限：{env.spec.max_episode_steps}")
    n_tiles = len(env.unwrapped.track)
    print(f"这条赛道有 {n_tiles} 块地砖，每压过一块新地砖奖励 1000 / {n_tiles} = {1000 / n_tiles:.2f}，"
          "每一帧另扣 0.1")

    after_zoom = first
    for _ in range(ZOOM_FRAMES):
        after_zoom, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    env.close()

    print(f"""
一帧画面是 96 × 96 × 3 = {96 * 96 * 3} 个数。对比 Phase 1 的摆：3 个数。
这 {96 * 96 * 3} 个数里真正决定"接下来会怎样"的量只有十来个：
车在赛道上的横向位置、朝向、速度、前方弯道的形状。其余大部分是草地。
    """)
    return first, after_zoom


def pixel_distance_demo():
    """CONCEPT.md 第 1.3 节的手算例子：像素距离不反映"差多远"。"""
    print("=" * 60)
    print("第三部分：像素空间里的距离没有语义")
    print("=" * 60)

    def line_image(col):
        img = np.zeros((64, 64), dtype=np.float32)
        img[:, col:col + 2] = 1.0  # 2 像素宽的白色竖线
        return img

    base, near, far = line_image(10), line_image(12), line_image(40)
    mse_near = float(((base - near) ** 2).mean())
    mse_far = float(((base - far) ** 2).mean())
    print(f"""
黑底上一条 2 像素宽的白色竖线，位于第 10–11 列。把它右移 2 像素和右移 30 像素，
分别和原图算均方误差：

    原图 vs 右移 2 像素 ：{mse_near:.4f}
    原图 vs 右移 30 像素：{mse_far:.4f}

两个数完全一样。只要两条线不重叠，移多远都是 256 个像素不同，256 / 4096 = 0.0625。
对控制来说这两种情况天差地别：赛道边线偏 2 像素无所谓，偏 30 像素就是冲出赛道。
所以不能把像素直接当状态用：需要一个表示，让"差一点"的距离小、"差很远"的距离大。
    """)
    assert abs(mse_near - 0.0625) < 1e-6 and abs(mse_far - 0.0625) < 1e-6


# ==================== 第四部分：预处理 ====================


def crop_and_resize(frame):
    """裁掉底部仪表条并缩放。[96, 96, 3] uint8 → [64, 64, 3] uint8（存盘用这个格式）。"""
    cropped = frame[:CROP_ROWS]                                            # [84, 96, 3]
    return cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def preprocess_frame(frame):
    """完整的预处理：[96, 96, 3] uint8 → [3, 64, 64] float32，取值 [0, 1]（喂给网络用这个格式）。"""
    small = crop_and_resize(frame)                                         # [64, 64, 3] uint8
    return np.transpose(small, (2, 0, 1)).astype(np.float32) / 255.0       # [3, 64, 64] float32


def show_preprocessing(first, after_zoom):
    print("=" * 60)
    print("第四部分：预处理")
    print("=" * 60)

    small = crop_and_resize(after_zoom)
    tensor = preprocess_frame(after_zoom)
    print(f"""
    步骤                    形状              类型与范围
    环境给出的原始画面        {str(list(after_zoom.shape)):<16}  {after_zoom.dtype}，0–255
    裁掉底部 12 行仪表条      {str(list(after_zoom[:CROP_ROWS].shape)):<16}  uint8
    缩放                    {str(list(small.shape)):<16}  uint8          ← 存盘用这个
    归一化、通道放最前        {str(list(tensor.shape)):<16}  {tensor.dtype}，{tensor.min():.2f}–{tensor.max():.2f}  ← 喂给网络用这个

为什么存盘用 uint8：一帧 64×64×3 的 uint8 是 {small.nbytes} 字节，存成 float32 是 {tensor.nbytes} 字节，差 4 倍。
为什么裁掉仪表条：它是画面里的"作弊通道"（速度条直接画出了车速），而且真实机器人的摄像头里没有这种东西。
    """)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    axes[0].imshow(first)
    axes[0].set_title("回合第 0 帧：镜头还在拉近")
    axes[1].imshow(after_zoom)
    axes[1].axhline(CROP_ROWS - 0.5, color="red", lw=1.5, ls="--")
    axes[1].set_title(f"第 {ZOOM_FRAMES} 帧（红线以下是仪表条）")
    axes[2].imshow(small)
    axes[2].set_title("预处理后 64×64")
    for ax in axes:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    save_figure(fig, OUT / "01-preprocessing.png")


# ==================== 第五部分：随机策略怎么采 ====================


def sample_independent_action(rng):
    """逐帧独立：每一帧都从动作空间里均匀采一个新动作。"""
    return np.array([rng.uniform(-1, 1), rng.uniform(0, 1), rng.uniform(0, 1)], dtype=np.float32)


def sample_random_action(prev_action, rng, steer_noise=0.1, max_gas=0.4):
    """带时间相关性的随机动作：在上一个动作的基础上加小扰动。

    方向：steer ← 0.9 × steer + steer_noise × 噪声。0.9 的衰减让方向盘有回正的趋势，不会一直打死
    油门：gas ← gas + 0.1 × 噪声，限制在 [0, max_gas]
    刹车：每帧有 5% 的概率点一脚，其余时候为 0（油门和刹车同时乱踩，车基本走不动）
    """
    if prev_action is None:
        steer, gas = 0.0, 0.3
    else:
        steer, gas = float(prev_action[0]), float(prev_action[1])
    steer = float(np.clip(0.9 * steer + steer_noise * rng.normal(), -1, 1))
    gas = float(np.clip(gas + 0.1 * rng.normal(), 0, max_gas))
    brake = float(rng.uniform(0, 0.6)) if rng.random() < 0.05 else 0.0
    return np.array([steer, gas, brake], dtype=np.float32)


def _rollout_worker(job):
    """在子进程里跑若干回合。job = (回合种子列表, 每回合步数, 策略名, 是否保存画面)。"""
    seeds, max_steps, policy, keep_frames = job
    env = make_env()
    episodes = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        frame = reset_env(env, seed)
        n_tiles = len(env.unwrapped.track)
        ep = {"frames": [], "actions": [], "rewards": [], "dones": [], "speeds": []}
        action = None
        for _ in range(max_steps):
            action = (sample_independent_action(rng) if policy == "independent"
                      else sample_random_action(action, rng))
            if keep_frames:
                ep["frames"].append(crop_and_resize(frame))
            velocity = env.unwrapped.car.hull.linearVelocity
            ep["speeds"].append(float(np.hypot(velocity[0], velocity[1])))  # 仿真器内部的真实车速
            frame, reward, terminated, truncated, _ = env.step(action)
            ep["actions"].append(action)
            ep["rewards"].append(reward)
            ep["dones"].append(terminated)
            if terminated or truncated:
                break
        ep["coverage"] = env.unwrapped.tile_visited_count / n_tiles
        ep["return"] = float(np.sum(ep["rewards"]))
        episodes.append(ep)
    env.close()
    return episodes


def run_episodes(n_episodes, max_steps, policy, keep_frames, seed_offset=0):
    """用 N_WORKERS 个进程并行跑 n_episodes 个回合，返回回合列表（顺序与种子一致）。"""
    seeds = [SEED + seed_offset + i for i in range(n_episodes)]
    chunks = [seeds[i::N_WORKERS] for i in range(N_WORKERS) if seeds[i::N_WORKERS]]
    jobs = [(chunk, max_steps, policy, keep_frames) for chunk in chunks]
    with get_context("spawn").Pool(len(jobs)) as pool:
        results = pool.map(_rollout_worker, jobs)
    by_seed = {s: ep for chunk, eps in zip(chunks, results) for s, ep in zip(chunk, eps)}
    return [by_seed[s] for s in seeds]


def compare_random_policies(n_episodes, max_steps):
    """逐帧独立 vs 带时间相关性：车能开多远。"""
    print("=" * 60)
    print("第五部分：随机策略也有讲究")
    print("=" * 60)
    print(f"""
两种"随机"：
    逐帧独立    每一帧重新采一个动作。这一帧左打下一帧右打，这一帧油门下一帧刹车，互相抵消
    带时间相关  在上一帧动作的基础上加小扰动，方向和油门都是缓慢变化的

各跑 {n_episodes} 个回合、每回合 {max_steps} 步，用"压过的地砖占整条赛道的比例"衡量车开了多远：
    """)

    results = {}
    for policy, label in [("independent", "逐帧独立"), ("correlated", "带时间相关")]:
        episodes = run_episodes(n_episodes, max_steps, policy, keep_frames=False, seed_offset=10_000)
        coverage = np.array([ep["coverage"] for ep in episodes])
        returns = np.array([ep["return"] for ep in episodes])
        speeds = np.concatenate([ep["speeds"] for ep in episodes])
        results[policy] = (coverage, returns, speeds)
        print(f"    {label:<6}  赛道覆盖 {coverage.mean() * 100:5.1f}%（最少 {coverage.min() * 100:.1f}%，"
              f"最多 {coverage.max() * 100:.1f}%）   平均车速 {speeds.mean():5.1f}   "
              f"{max_steps} 步的回报 {returns.mean():6.1f}")

    cov_ind, cov_cor = results["independent"][0].mean(), results["correlated"][0].mean()
    speed_ind, speed_cor = results["independent"][2].mean(), results["correlated"][2].mean()
    print(f"\n车速：带时间相关是逐帧独立的 {speed_cor / max(speed_ind, 1e-9):.1f} 倍。"
          f"赛道覆盖：{cov_cor / max(cov_ind, 1e-9):.1f} 倍。")
    if cov_cor > 1.5 * cov_ind:
        print("逐帧独立的动作互相抵消，车几乎停在起点；带时间相关的版本能开出去，压到的地砖明显更多。")
    elif cov_cor > cov_ind:
        print("带时间相关的版本车速高得多，但压到的地砖只多了一点：车是开出去了，可很快就开进了草地。")
    else:
        print("这次实验里带时间相关的随机策略没有覆盖更多赛道，和预期不符。")
    print(f"""注意两点：
    1. 即使是带时间相关的版本，{max_steps} 步里平均也只压过约 {cov_cor * 100:.0f}% 的地砖（最好的一个回合 {results["correlated"][0].max() * 100:.0f}%）。
       随机策略到不了的地方，采再多回合也没有。这是覆盖问题，不是数量问题（Phase 1 第 5.6 节）。
    2. 数据集里的画面决定了 V 和 M 见过什么。本节之后的数据集用"带时间相关"的策略采集。
    """)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    labels = ["逐帧独立", "带时间相关"]
    axes[0].boxplot([results["independent"][0] * 100, results["correlated"][0] * 100], tick_labels=labels)
    axes[0].set_ylabel("压过的地砖占赛道的比例（%）")
    axes[0].set_title(f"两种随机策略的赛道覆盖（各 {n_episodes} 个回合）")
    bins = np.linspace(0, max(results["correlated"][2].max(), 1.0), 40)
    axes[1].hist(results["independent"][2], bins=bins, alpha=0.6, label="逐帧独立", density=True)
    axes[1].hist(results["correlated"][2], bins=bins, alpha=0.6, label="带时间相关", density=True)
    axes[1].set_xlabel("车速（仿真器内部单位）")
    axes[1].set_ylabel("密度")
    axes[1].set_title("车速分布")
    axes[1].legend()
    save_figure(fig, OUT / "01-random-policy-coverage.png")
    return results


def random_baseline(n_episodes):
    """完整回合（到步数上限）下随机策略的回报，作为后面控制器的参照。"""
    max_steps = 1000 - ZOOM_FRAMES
    rows = {}
    for policy, label in [("independent", "逐帧独立"), ("correlated", "带时间相关")]:
        episodes = run_episodes(n_episodes, max_steps, policy, keep_frames=False, seed_offset=20_000)
        # 开头跳过的 ZOOM_FRAMES 帧每帧也扣 0.1，补回去才是完整回合的回报
        returns = np.array([ep["return"] for ep in episodes]) - 0.1 * ZOOM_FRAMES
        rows[policy] = returns
        print(f"    {label:<6}  完整回合的平均回报 {returns.mean():7.1f} ± {returns.std():.1f}"
              f"（{n_episodes} 个回合，最低 {returns.min():.1f}，最高 {returns.max():.1f}）")
    print("""
    参照：停在原地不动 1000 帧的回报是 -0.1 × 1000 = -100；论文的"解决"标准是 100 个回合平均 900。
    """)
    return rows


# ==================== 第六部分：采集数据集 ====================


def collect_rollouts(n_episodes, max_steps):
    """用带时间相关性的随机策略采集数据集。

    返回字典：
        frames   [E, T, 64, 64, 3] uint8   frames[e, t] 是执行 actions[e, t] 之前看到的画面 o_t
        actions  [E, T, 3] float32
        rewards  [E, T] float32            执行 actions[e, t] 之后得到的奖励 r_t
        dones    [E, T] bool               执行 actions[e, t] 之后回合是否终止
        speeds   [E, T] float32            仿真器内部的真实车速，只用于 04 节的探针实验，模型训练不用
        lengths  [E] int64                 每个回合的实际长度（提前终止的回合后面补 0）
    """
    episodes = run_episodes(n_episodes, max_steps, "correlated", keep_frames=True)
    data = {
        "frames": np.zeros((n_episodes, max_steps, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "actions": np.zeros((n_episodes, max_steps, 3), dtype=np.float32),
        "rewards": np.zeros((n_episodes, max_steps), dtype=np.float32),
        "dones": np.zeros((n_episodes, max_steps), dtype=bool),
        "speeds": np.zeros((n_episodes, max_steps), dtype=np.float32),
        "lengths": np.zeros(n_episodes, dtype=np.int64),
    }
    for e, ep in enumerate(episodes):
        n = len(ep["actions"])
        data["lengths"][e] = n
        data["frames"][e, :n] = np.stack(ep["frames"])
        data["actions"][e, :n] = np.stack(ep["actions"])
        data["rewards"][e, :n] = ep["rewards"]
        data["dones"][e, :n] = ep["dones"]
        data["speeds"][e, :n] = ep["speeds"]
    return data


def build_dataset(n_episodes, max_steps):
    print("=" * 60)
    print("第六部分：采集数据集")
    print("=" * 60)

    start = time.time()
    data = collect_rollouts(n_episodes, max_steps)
    elapsed = time.time() - start
    np.savez(DATASET_PATH, **data)
    size_mb = DATASET_PATH.stat().st_size / 1e6
    n_frames = int(data["lengths"].sum())

    print(f"""
采集完成：{n_episodes} 个回合 × 最多 {max_steps} 步，共 {n_frames} 帧，用时 {elapsed:.0f} 秒（{N_WORKERS} 个进程）
存盘：{DATASET_PATH.relative_to(OUT.parents[1])}，{size_mb:.0f} MB

    数组      形状                       类型
    frames   {str(list(data['frames'].shape)):<26} {data['frames'].dtype}
    actions  {str(list(data['actions'].shape)):<26} {data['actions'].dtype}
    rewards  {str(list(data['rewards'].shape)):<26} {data['rewards'].dtype}
    dones    {str(list(data['dones'].shape)):<26} {data['dones'].dtype}
    speeds   {str(list(data['speeds'].shape)):<26} {data['speeds'].dtype}
    lengths  {str(list(data['lengths'].shape)):<26} {data['lengths'].dtype}

提前终止（冲出场地边界）的回合：{int(data['dones'].any(axis=1).sum())} 个
每步奖励：均值 {data['rewards'].mean():.3f}，最小 {data['rewards'].min():.2f}，最大 {data['rewards'].max():.2f}
动作各维的均值：方向 {data['actions'][..., 0].mean():+.2f}，油门 {data['actions'][..., 1].mean():.2f}，刹车 {data['actions'][..., 2].mean():.2f}

和论文比：论文采了 10,000 个回合（约 123 GB 的 uint8 画面）。这里是它的 {n_episodes / 10000 * 100:.1f}%。
    """)

    # 看一眼数据：一个回合里等间隔取 8 帧
    episode = int(np.argmax(data["speeds"].mean(axis=1)))  # 挑车速最高的回合，画面变化最明显
    steps = np.linspace(0, data["lengths"][episode] - 1, 8).astype(int)
    fig, axes = plt.subplots(1, 8, figsize=(14, 2.2))
    for ax, t in zip(axes, steps):
        ax.imshow(data["frames"][episode, t])
        ax.set_title(f"t = {t}", fontsize=9)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"数据集里的一个回合（第 {episode} 个回合，等间隔取 8 帧）", fontsize=11)
    save_figure(fig, OUT / "01-dataset-samples.png")
    return data


def single_frame_ambiguity(data):
    """单帧画面里没有车速：找两帧几乎一样、车速差很多的画面。"""
    print("=" * 60)
    print("第七部分：单帧画面里没有车速")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    lengths = data["lengths"]
    pool = [(e, t) for e in range(len(lengths)) for t in range(0, int(lengths[e]), 4)]
    idx = rng.choice(len(pool), size=min(1500, len(pool)), replace=False)
    picks = [pool[i] for i in idx]
    frames = np.stack([data["frames"][e, t] for e, t in picks]).astype(np.float32) / 255.0   # [N, 64, 64, 3]
    speeds = np.array([data["speeds"][e, t] for e, t in picks])
    flat = frames.reshape(len(frames), -1).astype(np.float64)                               # [N, 12288]
    # 两两的均方误差：‖a - b‖² = ‖a‖² + ‖b‖² - 2 a·b。这是纯 NumPy 的 CPU 计算，用 float64 减小舍入误差
    sq = (flat**2).sum(axis=1)
    mse = np.maximum(sq[:, None] + sq[None, :] - 2 * flat @ flat.T, 0.0) / flat.shape[1]
    np.fill_diagonal(mse, np.inf)
    typical = float(np.median(mse[np.isfinite(mse)]))
    # 在"画面很像"的帧对里（均方误差排在最小的 0.5%），找车速差最大的一对
    threshold = np.quantile(mse[np.isfinite(mse)], 0.005)
    speed_gap = np.abs(speeds[:, None] - speeds[None, :])
    speed_gap[mse > threshold] = -1
    i, j = np.unravel_index(np.argmax(speed_gap), speed_gap.shape)
    print(f"""
从数据集里抽 {len(picks)} 帧，两两算像素均方误差（中位数 {typical:.4f}）。
在画面最像的 0.5% 帧对里，车速差最大的一对：

    帧 A：车速 {speeds[i]:5.1f}
    帧 B：车速 {speeds[j]:5.1f}
    两帧的像素均方误差：{mse[i, j]:.5f}（是典型帧对的 {mse[i, j] / typical * 100:.1f}%）

画面几乎一样，车速差了 {abs(speeds[i] - speeds[j]):.1f}。给模型看其中任何一帧，它都没法知道下一帧
车会前进多少。这就是"部分可观测"：下一帧不由"当前帧 + 动作"唯一决定，模型必须能用到历史。
这是 M（04 节的 MDN-RNN）要有记忆的原因。
    """)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    axes[0].imshow(frames[i])
    axes[0].set_title(f"帧 A：车速 {speeds[i]:.1f}")
    axes[1].imshow(frames[j])
    axes[1].set_title(f"帧 B：车速 {speeds[j]:.1f}")
    axes[2].imshow(np.abs(frames[i] - frames[j]).mean(axis=-1), cmap="magma", vmin=0, vmax=0.5)
    axes[2].set_title("逐像素差的绝对值")
    for ax in axes:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    save_figure(fig, OUT / "01-single-frame-ambiguity.png")


# ==================== 第八部分：练习 ====================


def exercises():
    print("=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把 CROP_ROWS 改成 96（不裁仪表条），重新看预处理后的画面。
        仪表条里哪一段对应车速？如果不裁，后面的 V 会把它编码进 z 吗？

练习 2：把 sample_random_action 的 steer_noise 改成 0.02 和 0.4，重新运行第五部分，
        看赛道覆盖和车速怎么变。方向盘晃得越小越好吗？

练习 3：手算。论文采 10,000 个回合，如果每个回合存满 1000 帧 64×64×3 的 uint8 画面，需要多少磁盘空间？
        如果误存成 float32 呢？

思考题 1：把最近 4 帧叠在一起当观测，第七部分的问题就解决了吗？还缺什么？

思考题 2：数据集是随机策略采的。训练出来的控制器开得比随机策略远得多之后，
          V 和 M 会遇到什么问题？（提示：Phase 1 第 5.6 节）

参考答案：
    练习 1：底部仪表条最左边的白色竖条是车速，其余几条是 ABS、方向盘角度和陀螺仪。
            不裁的话 VAE 会尽量重建它，z 里就带上了车速，相当于把"部分可观测"偷偷变成了"接近完全可观测"。
            真实机器人的摄像头里没有仪表条，所以这里裁掉。
    练习 2：steer_noise 太大，方向盘来回猛打，车在原地附近画圈；太小则车几乎一直走直线，
            遇到第一个弯道就冲进草地，数据里也看不到"转弯"这种画面。中间存在一个比较合适的范围。
    练习 3：10,000 × 1000 × 12288 字节 ≈ 123 GB；float32 再乘 4，约 492 GB。
    思考题 1：叠帧能补上车速这类短期信息，观测更接近马尔可夫。但叠帧不提供"预测未来"的能力：
              M 既是记忆也是动力学模型，后者叠帧替代不了。
    思考题 2：控制器开到了随机策略没到过的赛道后半段和高速状态，那里的画面 V 没见过、重建会变差，
              M 的预测也不可靠。这是分布偏移，办法是用新控制器采数据再重新训练 V 和 M（CONCEPT.md 第 8.7 节）。
    """)


def main():
    seed_everything(SEED)
    # 冒烟模式只验证流程能跑通
    n_compare, compare_steps, n_baseline, n_episodes, max_steps = (
        (4, 100, 0, 8, 100) if is_quick() else (20, 300, 8, 200, 300)
    )

    introduction()
    first, after_zoom = inspect_env()
    pixel_distance_demo()
    show_preprocessing(first, after_zoom)
    compare_random_policies(n_compare, compare_steps)
    if n_baseline:
        random_baseline(n_baseline)
    data = build_dataset(n_episodes, max_steps)
    single_frame_ambiguity(data)
    exercises()

    print("=" * 60)
    print("本节小结")
    print("=" * 60)
    print("""
    1. CarRacing-v3 只给画面：96×96×3，27648 个数，其中大部分和开车无关。
    2. 像素空间的距离没有语义：竖线移 2 像素和移 30 像素，均方误差一样。
    3. 预处理：裁掉仪表条 → 缩放到 64×64 → 存 uint8；喂给网络时再转 float32、除以 255。
    4. 随机策略要带时间相关性，车才能开出去；即使这样，随机策略也只覆盖赛道的一小段。
    5. 单帧画面里没有车速，这是后面需要 M（带记忆的模型）的原因。

下一节：训练卷积 VAE（V），把这份数据集里的每一帧压成一个 32 维的隐向量 z。
    """)
    if is_quick():
        print("（当前是冒烟模式 WM_QUICK=1：数据量很小，上面的数字没有参考价值。）")


if __name__ == "__main__":
    main()
