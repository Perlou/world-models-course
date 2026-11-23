"""设备选择、随机种子、冒烟模式开关。"""

import os
import random

import numpy as np
import torch


def is_quick() -> bool:
    """是否处于冒烟模式。

    运行讲义时加上环境变量 WM_QUICK=1，训练预算会缩到很小，
    只用来确认代码能跑通，结果没有参考价值：

        WM_QUICK=1 uv run python src/phase-01-dynamics-and-planning/03-learning-dynamics-mlp.py
    """
    return os.environ.get("WM_QUICK") == "1"


def get_device() -> torch.device:
    """按 cuda → mps → cpu 的顺序选设备。

    可以用环境变量 WM_DEVICE 强制指定，例如 WM_DEVICE=cpu。
    在 M1 Max 上实测，哪怕是两层 200 单元的小 MLP，MPS 也比 CPU 快约 2 倍，
    所以本课程默认用 MPS。

    注意：MPS 不支持 float64，课程里的张量一律用 float32。
    """
    forced = os.environ.get("WM_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy、PyTorch 的随机种子。

    环境的随机性要另外用 env.reset(seed=...) 固定。
    MPS 和 CUDA 上的浮点运算不保证逐位可复现，同一种子两次运行的结果
    可能有细微差别，这是正常现象。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
