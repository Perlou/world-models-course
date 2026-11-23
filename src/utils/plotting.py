"""画图与存视频：中文字体、统一的输出目录。"""

from pathlib import Path

import numpy as np

# 仓库根目录：src/utils/plotting.py → 上两级
REPO_ROOT = Path(__file__).resolve().parents[2]

# 按顺序找第一个系统里有的中文字体（macOS → Linux → Windows）
_CJK_FONTS = [
    "Arial Unicode MS",
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
]


def setup_matplotlib():
    """配置 matplotlib 并返回 pyplot。

    - 使用 Agg 后端：只存文件、不弹窗口，脚本不会卡在 plt.show()。
    - 自动选一个可用的中文字体，避免图里的中文变成方块。

    用法：
        plt = setup_matplotlib()
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CJK_FONTS:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    else:
        print("⚠️  没找到中文字体，图里的中文可能显示成方块。"
              "Linux 上可以安装：apt install fonts-noto-cjk")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    return plt


def output_dir(phase: str) -> Path:
    """返回并创建 outputs/<phase>/ 目录，例如 output_dir("phase-01")。

    outputs/ 已写进 .gitignore，运行产物不入库。
    """
    path = REPO_ROOT / "outputs" / phase
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, path) -> Path:
    """保存并关闭一张图，打印保存位置。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"📊 图已保存：{_relative(path)}")
    return path


def save_video(frames, path, fps: int = 30) -> Path:
    """把一串 RGB 帧（H×W×3，uint8）存成 mp4 或 gif。"""
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    if path.suffix == ".gif":
        imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
    else:
        # macro_block_size=1：不强制把画面尺寸补成 16 的倍数
        imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    print(f"🎬 视频已保存：{_relative(path)}")
    return path


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
