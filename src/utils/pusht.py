"""PushT 演示数据：下载、解码、缓存。

PushT 推块任务是 Phase 8–11 的贯穿任务。这里只负责把 LeRobot 发布的人类演示数据
（Hugging Face 上的 `lerobot/pusht`）取回来，解码成 NumPy 数组并缓存到 `data/pusht/`。
`data/` 已写进 .gitignore，不入库。

数据是 LeRobot v3.0 格式，一共四个文件（约 8 MB）：

    meta/info.json                                   数据集的元信息
    meta/episodes/chunk-000/file-000.parquet         每个回合在总表里的起止位置
    data/chunk-000/file-000.parquet                  逐帧的状态、动作、奖励
    videos/observation.image/chunk-000/file-000.mp4  全部回合首尾相接的 96×96 画面（AV1 编码）

LeRobot 格式本身的讲解和手写解析在 Phase 11 的讲义 `02-lerobot-data-and-action-chunks.py`。

用法：

    from utils.pusht import load_pusht_demos, make_pusht_env
    demos = load_pusht_demos()
    demos["images"]         # [N, 96, 96, 3] uint8，N = 25650
    demos["states"]         # [N, 2] float32，推杆位置，像素坐标 0–512
    demos["actions"]        # [N, 2] float32，推杆的目标位置，像素坐标 0–512
    demos["rewards"]        # [N] float32，执行该帧动作之后的奖励（T 形块与目标区域的重叠程度）
    demos["successes"]      # [N] bool
    demos["episode_starts"] # [E] int64，每个回合第一帧的下标，E = 206
    demos["episode_ends"]   # [E] int64，每个回合最后一帧的下标 + 1

需要闭环交互（跑策略、做规划）时用 `make_pusht_env()` 创建 `gym-pusht` 的仿真环境。
演示数据里没有 T 形块的位姿；需要真实位姿时在仿真环境里重新采集。

下载走 urllib，会使用环境变量 `https_proxy`。访问不了 huggingface.co 时可以设置镜像：

    HF_ENDPOINT=https://hf-mirror.com uv run python <讲义路径>
"""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from .plotting import REPO_ROOT

PUSHT_DIR = REPO_ROOT / "data" / "pusht"
_REPO_FILES = [
    "meta/info.json",
    "meta/episodes/chunk-000/file-000.parquet",
    "data/chunk-000/file-000.parquet",
    "videos/observation.image/chunk-000/file-000.mp4",
]


def download_pusht_raw(root: Path = PUSHT_DIR) -> Path:
    """把 `lerobot/pusht` 的原始文件下载到 root/raw/，已存在的文件不重复下载。返回 raw 目录。"""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    raw = Path(root) / "raw"
    for name in _REPO_FILES:
        target = raw / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{endpoint}/datasets/lerobot/pusht/resolve/main/{name}"
        print(f"⬇️  下载 {url}")
        tmp = target.with_suffix(target.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp)
        except Exception as err:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"下载 PushT 数据失败：{err}\n"
                "可以设置镜像后重试：HF_ENDPOINT=https://hf-mirror.com"
            ) from err
        tmp.rename(target)
    return raw


def _decode_raw(raw: Path) -> dict:
    """把原始的 parquet + mp4 解码成 NumPy 数组。"""
    import imageio.v2 as imageio
    import pyarrow.parquet as pq

    frames = pq.read_table(raw / "data/chunk-000/file-000.parquet").to_pydict()
    episodes = pq.read_table(
        raw / "meta/episodes/chunk-000/file-000.parquet",
        columns=["episode_index", "dataset_from_index", "dataset_to_index"],
    ).to_pydict()
    info = json.loads((raw / "meta/info.json").read_text())

    reader = imageio.get_reader(raw / "videos/observation.image/chunk-000/file-000.mp4")
    images = np.stack([frame for frame in reader])                      # [N, 96, 96, 3]，RGB
    reader.close()

    n = info["total_frames"]
    assert len(images) == n == len(frames["action"]), (
        f"帧数对不上：视频 {len(images)} 帧，表格 {len(frames['action'])} 行，元信息 {n} 帧"
    )
    assert frames["index"] == list(range(n)), "逐帧表没有按全局下标排好序"
    return {
        "images": images.astype(np.uint8),
        "states": np.asarray(frames["observation.state"], dtype=np.float32),
        "actions": np.asarray(frames["action"], dtype=np.float32),
        "rewards": np.asarray(frames["next.reward"], dtype=np.float32),
        "successes": np.asarray(frames["next.success"], dtype=bool),
        "episode_starts": np.asarray(episodes["dataset_from_index"], dtype=np.int64),
        "episode_ends": np.asarray(episodes["dataset_to_index"], dtype=np.int64),
    }


def load_pusht_demos(root: Path = PUSHT_DIR) -> dict:
    """读取 PushT 的人类演示数据，返回一个 NumPy 数组的字典（字段见本文件开头）。

    第一次调用会下载（约 8 MB）并解码，把结果缓存成 root/pusht-demos.npz；之后直接读缓存。
    全部画面放进内存约占 700 MB。
    """
    cache = Path(root) / "pusht-demos.npz"
    if cache.exists():
        with np.load(cache) as data:
            return {key: data[key] for key in data.files}
    demos = _decode_raw(download_pusht_raw(root))
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f"pusht-demos.{os.getpid()}.tmp.npz")  # 先写临时文件再改名，避免留下写了一半的缓存
    np.savez_compressed(tmp, **demos)
    tmp.rename(cache)
    print(f"💾 PushT 演示数据已缓存：{cache.relative_to(REPO_ROOT)}"
          f"（{len(demos['episode_starts'])} 个回合，{len(demos['images'])} 帧）")
    return demos


def make_pusht_env(obs_type: str = "pixels_agent_pos", **kwargs):
    """创建 `gym-pusht` 的仿真环境。

    obs_type:
        "pixels_agent_pos"  观测是字典：pixels [96, 96, 3] uint8，agent_pos [2]
        "state"             观测是 [5]：推杆位置 (2)、T 形块位置 (2)、T 形块朝向 (1)
    动作是推杆的目标位置 [2]，像素坐标 0–512。step() 返回的 info 里有真实的
    block_pose、goal_pose、coverage（重叠比例）、is_success。
    """
    import gym_pusht  # noqa: F401  导入时注册环境
    import gymnasium as gym

    kwargs.setdefault("render_mode", "rgb_array")
    return gym.make("gym_pusht/PushT-v0", obs_type=obs_type, **kwargs)
