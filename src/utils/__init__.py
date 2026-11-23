"""
课程共享基础设施
================

这里只放和"世界模型"本身无关的杂事：选设备、固定随机种子、画图、存视频。
讲义里的模型、规划器等核心代码都写在各自的讲义文件里，不藏在这里。

讲义中的用法（每节讲义开头都是这三行）：

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
"""

from .device import get_device, is_quick, seed_everything
from .plotting import output_dir, save_figure, save_video, setup_matplotlib

__all__ = [
    "get_device",
    "is_quick",
    "seed_everything",
    "output_dir",
    "save_figure",
    "save_video",
    "setup_matplotlib",
]
