"""
把训练搬到云端 GPU
==================

⚠️ 草稿状态：本文件写成时运行环境不允许执行 Python，尚未实际运行过。
   定稿前必须完整跑一遍、核对输出，并把下面的「预计运行时间」换成实测值。

学习目标：
    1. 把一次训练写成"无人值守、可中断、可重建"的作业：配置、运行目录、日志、检查点、续训
    2. 用续训等价测试找出检查点漏存了什么
    3. 走一遍上云之前的调试台阶：初始损失、过拟合一个批次、梯度流向
    4. 实测数据吞吐、训练吞吐和规划器选一个动作的耗时，代进预算公式
    5. 知道混合精度怎么写、哪些运算要留在 32 位（这部分本机没有 NVIDIA 显卡，只讲写法）

核心概念：
    - 一次运行（run）=（代码版本，配置，数据指纹，随机种子，软硬件环境）
    - 检查点（checkpoint）：任何在训练中会变、又会影响后续训练的东西都要存
    - 续训等价：连续跑 2N 步，和"跑 N 步、中断、再续 N 步"结果一致
    - 原子写入（atomic write）：先写临时文件再改名
    - 预算：h = N_step / v + N_eval × E × T × t_act；G = N_run × h ×（1 + η）+ 收尾评估

前置知识：
    - 01-offline-data-pipeline.py（数据指纹）
    - Phase 1 的 06-cem-and-mppi.py（CEM，这里只用来测"选一个动作要多久"）

对应概念文档：
    - CONCEPT.md 第 4 节「把训练搬到云端」

预计运行时间：待实测（本稿尚未运行）
"""

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.planning import cem_plan

plt = setup_matplotlib()
OUT = output_dir("phase-12")
SUFFIX = "-quick" if is_quick() else ""
RUNS = OUT / f"02-runs{SUFFIX}"

# 预算演示用的外部数字。都是示例值，不是任何平台的报价或任何项目的实测，用的时候换成自己的
PRICE_PER_GPU_HOUR = 2.0            # 元 / 小时
EXAMPLE_PLAN = dict(n_step=100_000, n_eval=10, n_eval_episodes=10, episode_len=200,
                    n_run=25, eta=1.0, n_final_episodes=50)


# ==================== 第一部分：开发环境与生产环境 ====================


def introduction():
    """本机负责"对不对"，4090 负责"跑得完"。"""
    print("=" * 60)
    print("第一部分：把训练变成一个批处理作业")
    print("=" * 60)

    print("""
本机（M1 Max）是开发环境：免费、随时可用，但慢。租来的 4090 是生产环境：快，但按小时计费，
而且实例随时可能被关机。所以"把训练搬到云端"的主要工作在搬之前：

    ┌──────────────────────────── 本机 ────────────────────────────┐      ┌────── 4090 ──────┐
    │ 配置（单一事实来源）                                            │      │ 冒烟一遍确认环境    │
    │   └▶ 调试台阶：初始损失 → 过拟合一个批次 → 梯度流向              │ ───▶ │ 在目标机上重测吞吐  │
    │   └▶ 训练作业：运行目录 + 日志 + 检查点 + 续训                   │      │ 全量 × 多个种子    │
    │   └▶ 续训等价测试；"杀掉进程再续训"演练                          │      │ 结果拉回本机       │
    │   └▶ 测吞吐 → 预算公式 → 决定实验矩阵多大                        │      └──────────────────┘
    └──────────────────────────────────────────────────────────────┘

这一节的演示任务故意选得很小：在 Pendulum 的随机数据上训练一个 GRU 序列模型，
读入 (o_t, a_t) 序列，预测每一步观测的变化量。重点是训练作业的骨架，不是模型。

    观测 o_t = (cos θ, sin θ, θ̇)   [B, L+1, 3]
    动作 a_t = 力矩                 [B, L, 1]
    模型：Linear → GRU → Dropout → Linear，输出 [B, L, 3]，目标是 (o_{t+1} − o_t) / 标准差

骨架里该有的东西它都有：学习率调度、EMA 权重、Dropout（用到随机数）、归一化统计量。
这几样正是检查点最容易漏存的。
    """)


# ==================== 第二部分：配置 ====================


@dataclass
class TrainConfig:
    """一次运行的全部超参数。代码里不出现散落的常数。"""
    # 数据
    n_episodes: int = 40            # 采集多少个回合（最后 1/5 留作验证）
    window: int = 32                # 序列窗口长度 L
    # 模型
    hidden: int = 128
    dropout: float = 0.1
    # 优化
    total_steps: int = 2000
    batch_size: int = 64
    lr: float = 1e-3
    warmup_steps: int = 100
    grad_clip: float = 10.0
    ema_decay: float = 0.99
    # 作业
    seed: int = 0
    device: str = "cpu"             # 这个 GRU 很小，在 CPU 上更快，而且 CPU 上能做到逐位可复现
    use_amp: bool = False           # 混合精度，只在 CUDA 上生效
    log_every: int = 50
    eval_every: int = 500
    ckpt_every: int = 250
    data_fingerprint: str = ""      # 启动时填入，写进检查点；续训时核对


# 冒烟模式只是配置的一种覆盖：缩小预算，不走另一条代码路径
QUICK_OVERRIDES = ["n_episodes=10", "total_steps=120", "warmup_steps=20", "hidden=32",
                   "log_every=10", "eval_every=60", "ckpt_every=30"]


def apply_overrides(cfg, overrides):
    """用形如 "lr=3e-4" 的字符串覆盖配置项，返回新的配置。类型按该项默认值的类型转换。"""
    changes = {}
    for item in overrides:
        key, _, value = item.partition("=")
        if not hasattr(cfg, key):
            raise ValueError(f"配置里没有 {key} 这一项。可用的项：{', '.join(asdict(cfg))}")
        kind = type(getattr(cfg, key))
        changes[key] = value.lower() in ("1", "true", "yes") if kind is bool else kind(value)
    return dataclasses.replace(cfg, **changes)


def make_config(cli_overrides=()):
    """默认值 → 冒烟覆盖 → 命令行覆盖。"""
    cfg = TrainConfig()
    if is_quick():
        cfg = apply_overrides(cfg, QUICK_OVERRIDES)
    return apply_overrides(cfg, cli_overrides)


def config_demo(cfg):
    print("\n" + "=" * 60)
    print("第二部分：配置")
    print("=" * 60)
    demo = apply_overrides(TrainConfig(), ["lr=3e-4", "hidden=256", "use_amp=true"])
    assert (demo.lr, demo.hidden, demo.use_amp) == (3e-4, 256, True)
    print(f"""
所有超参数集中在 TrainConfig 这一个数据类里。三层来源，后面的覆盖前面的：

    默认值（写在代码里）  →  冒烟覆盖（WM_QUICK=1 时）  →  命令行覆盖

    例：uv run python 02-scaling-up-training.py lr=3e-4 hidden=256 use_amp=true
        得到 lr = {demo.lr}，hidden = {demo.hidden}，use_amp = {demo.use_amp}，其余保持默认

排实验矩阵时，每个格子就是一组命令行覆盖。启动时把合并后的完整配置写进运行目录的 config.json，
而不是只记命令行参数：默认值以后会改，改了之后只凭命令行就重建不出当时的配置。

本次运行生效的配置{'（冒烟模式）' if is_quick() else ''}：
    {json.dumps(asdict(cfg), ensure_ascii=False)}
    """)


# ==================== 数据与模型 ====================


def collect_dataset(n_episodes, seed):
    """Pendulum-v1 上的随机策略数据。返回 (数据字典, 归一化统计量, 数据指纹)。"""
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)
    all_obs, all_actions = [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        observations, actions = [obs], []
        for _ in range(200):
            action = rng.uniform(-2.0, 2.0, size=1).astype(np.float32)
            obs, _, _, _, _ = env.step(action)
            observations.append(obs)
            actions.append(action)
        all_obs.append(observations)
        all_actions.append(actions)
    env.close()
    obs = np.asarray(all_obs, dtype=np.float32)              # [E, T+1, 3]
    action = np.asarray(all_actions, dtype=np.float32)       # [E, T, 1]
    n_val = max(n_episodes // 5, 1)                          # 按回合切分
    data = {"train_obs": obs[:-n_val], "train_action": action[:-n_val],
            "val_obs": obs[-n_val:], "val_action": action[-n_val:]}
    delta = data["train_obs"][:, 1:] - data["train_obs"][:, :-1]
    norm = {"obs_mean": torch.from_numpy(data["train_obs"].mean(axis=(0, 1))),
            "obs_std": torch.from_numpy(data["train_obs"].std(axis=(0, 1))),
            "delta_std": torch.from_numpy(delta.std(axis=(0, 1)))}                # 只从训练回合算
    fingerprint = hashlib.sha256(obs.tobytes() + action.tobytes()).hexdigest()[:12]
    return data, norm, fingerprint


class SeqDynamics(nn.Module):
    """GRU 序列模型：读入归一化的 (o_t, a_t) 序列，预测每一步观测变化量（以标准差为单位）。"""

    def __init__(self, hidden=128, dropout=0.1, obs_dim=3, action_dim=1):
        super().__init__()
        self.inp = nn.Linear(obs_dim + action_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, obs_dim)

    def forward(self, obs, action, h=None):
        """obs [B, L, 3]，action [B, L, 1] → 预测 [B, L, 3]，以及 GRU 的末状态 [1, B, hidden]。"""
        x = torch.relu(self.inp(torch.cat([obs, action], dim=-1)))
        y, h = self.gru(x, h)
        return self.out(self.drop(y)), h


def prediction_loss(model, obs, action, norm):
    """obs [B, L+1, 3]，action [B, L, 1] → 标量损失。"""
    x = (obs[:, :-1] - norm["obs_mean"]) / norm["obs_std"]
    target = (obs[:, 1:] - obs[:, :-1]) / norm["delta_std"]
    pred, _ = model(x, action)
    return ((pred - target) ** 2).mean()


def sample_batch(data, cfg, rng):
    """随机取一个批次的窗口（不跨回合）。返回 obs [B, L+1, 3]、action [B, L, 1]，已放到设备上。"""
    obs, action = data["train_obs"], data["train_action"]
    e = rng.integers(0, obs.shape[0], size=cfg.batch_size)
    t = rng.integers(0, action.shape[1] - cfg.window + 1, size=cfg.batch_size)
    idx = t[:, None] + np.arange(cfg.window + 1)                              # [B, L+1]
    device = torch.device(cfg.device)
    return (torch.from_numpy(obs[e[:, None], idx]).to(device),
            torch.from_numpy(action[e[:, None], idx[:, :-1]]).to(device))


@torch.no_grad()
def evaluate(model, data, norm, cfg):
    """在验证回合上算损失，每个回合一个数。返回 [验证回合数]。"""
    device = torch.device(cfg.device)
    obs = torch.from_numpy(data["val_obs"]).to(device)
    action = torch.from_numpy(data["val_action"]).to(device)
    x = (obs[:, :-1] - norm["obs_mean"]) / norm["obs_std"]
    target = (obs[:, 1:] - obs[:, :-1]) / norm["delta_std"]
    pred, _ = model(x, action)
    return ((pred - target) ** 2).mean(dim=(1, 2)).cpu().numpy()


# ==================== 训练状态、单步更新、检查点 ====================


def lr_factor(step, cfg):
    """学习率调度：线性预热，然后余弦下降到 10%。"""
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = min((step - cfg.warmup_steps) / max(cfg.total_steps - cfg.warmup_steps, 1), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_state(cfg, norm):
    """从配置新建一次训练的全部状态（相当于一个新进程启动时做的事）。"""
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    model = SeqDynamics(cfg.hidden, cfg.dropout).to(device)
    ema = copy.deepcopy(model).requires_grad_(False).eval()                   # EMA 权重：评估用，不参与反向传播
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, cfg))
    amp = cfg.use_amp and device.type == "cuda"
    return {"model": model, "ema": ema, "optimizer": optimizer, "scheduler": scheduler,
            "scaler": torch.amp.GradScaler("cuda") if amp else None,
            "step": 0, "batch_rng": np.random.default_rng(cfg.seed),
            "norm": {key: value.to(device) for key, value in norm.items()}}


def train_one_step(state, data, cfg, batch=None):
    """一步更新：取批次 → 前向 → 损失 → 反向 → 梯度裁剪 → 更新 → 调度器 → EMA。返回 (损失, 梯度范数)。"""
    model, optimizer = state["model"], state["optimizer"]
    model.train()
    obs, action = batch if batch is not None else sample_batch(data, cfg, state["batch_rng"])
    optimizer.zero_grad()
    if state["scaler"] is not None:
        # 混合精度路径（只在 CUDA 上走到这里）。⚠️ 未在本机验证：本机没有 NVIDIA 显卡
        scaler = state["scaler"]
        x = (obs[:, :-1] - state["norm"]["obs_mean"]) / state["norm"]["obs_std"]
        target = (obs[:, 1:] - obs[:, :-1]) / state["norm"]["delta_std"]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pred, _ = model(x, action)                                        # 前向用 16 位
        loss = ((pred.float() - target) ** 2).mean()                          # 损失转回 32 位再算
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                                            # 裁剪之前先把梯度缩放回来
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss = prediction_loss(model, obs, action, state["norm"])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
    state["scheduler"].step()
    with torch.no_grad():
        for p_ema, p in zip(state["ema"].parameters(), model.parameters()):
            p_ema.lerp_(p, 1.0 - cfg.ema_decay)
    state["step"] += 1
    return loss.item(), float(grad_norm)


def save_checkpoint(path, state, cfg):
    """把恢复这次运行所需的全部状态写进一个文件（原子写入）。"""
    checkpoint = {
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "optimizer": state["optimizer"].state_dict(),                         # Adam 的一阶、二阶动量
        "scheduler": state["scheduler"].state_dict(),
        "scaler": state["scaler"].state_dict() if state["scaler"] is not None else None,
        "step": state["step"],
        "norm": state["norm"],                                                # 归一化统计量
        "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                "batch": state["batch_rng"].bit_generator.state},
        "config": asdict(cfg),
        "data_fingerprint": cfg.data_fingerprint,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)                                                     # 进程在写的中途被杀，last.pt 也不会损坏


def load_checkpoint(path, state, cfg, skip=()):
    """把检查点恢复进一个新建的 state。skip 用来模拟"漏存了某一项"，正常使用时不传。"""
    # weights_only=False：检查点里除了张量还有随机数状态。只加载自己写出的检查点
    checkpoint = torch.load(path, map_location=cfg.device, weights_only=False)
    if checkpoint["data_fingerprint"] != cfg.data_fingerprint:
        raise RuntimeError(f"数据指纹对不上：检查点是在 {checkpoint['data_fingerprint']} 上训练的，"
                           f"当前数据是 {cfg.data_fingerprint}。拒绝续训。")
    state["model"].load_state_dict(checkpoint["model"])
    state["step"] = checkpoint["step"]
    state["norm"] = checkpoint["norm"]
    # 漏存 EMA 时能做的"补救"只有用当前权重重新初始化
    state["ema"].load_state_dict(checkpoint["model" if "ema" in skip else "ema"])
    if "optimizer" not in skip:
        state["optimizer"].load_state_dict(checkpoint["optimizer"])
    if "scheduler" not in skip:
        state["scheduler"].load_state_dict(checkpoint["scheduler"])
    if state["scaler"] is not None and checkpoint["scaler"] is not None:
        state["scaler"].load_state_dict(checkpoint["scaler"])
    if "rng" not in skip:
        random.setstate(checkpoint["rng"]["python"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        state["batch_rng"].bit_generator.state = checkpoint["rng"]["batch"]
    return checkpoint


# ==================== 第三部分：调试台阶 ====================


def overfit_one_batch(cfg, data, norm, n_steps, batch_size=8):
    """台阶 3：固定一个很小的批次反复训练，看损失能不能降到接近 0。返回损失序列 [n_steps]。"""
    # 关掉 Dropout；学习率固定不下降，并取大一些（只有 8 条序列，不怕不稳）
    cfg = dataclasses.replace(cfg, dropout=0.0, batch_size=batch_size, lr=3e-3, warmup_steps=1, total_steps=10**9)
    state = build_state(cfg, norm)
    batch = sample_batch(data, cfg, state["batch_rng"])
    return [train_one_step(state, data, cfg, batch=batch)[0] for _ in range(n_steps)]


def debugging_ladder(cfg, data, norm):
    """上云之前先走的几级台阶：初始损失、过拟合一个批次、梯度流向。"""
    print("\n" + "=" * 60)
    print("第三部分：上云之前的调试台阶")
    print("=" * 60)

    # ---- 台阶 2：初始损失 ----
    n_classes = 255
    torch.manual_seed(0)
    features, labels = torch.randn(512, 64), torch.randint(0, n_classes, (512,))
    head = nn.Linear(64, n_classes)
    ce_default = nn.functional.cross_entropy(head(features), labels).item()
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    ce_zero = nn.functional.cross_entropy(head(features), labels).item()
    assert abs(ce_zero - math.log(n_classes)) < 1e-4, "输出层置零后，交叉熵应当正好是 log K"

    state = build_state(cfg, norm)
    obs, action = sample_batch(data, cfg, state["batch_rng"])
    state["model"].eval()
    with torch.no_grad():
        initial = prediction_loss(state["model"], obs, action, state["norm"]).item()
        target_power = (((obs[:, 1:] - obs[:, :-1]) / state["norm"]["delta_std"]) ** 2).mean().item()
    print(f"""
台阶 2：初始损失应当和理论值相符。

    分类头（{n_classes} 个类别，理论值 log {n_classes} = {math.log(n_classes):.3f}）
        输出层置零（Dreamer V3 对奖励头、价值头的做法）   交叉熵 {ce_zero:.3f}
        PyTorch 默认初始化                              交叉熵 {ce_default:.3f}

    本节的回归模型（目标已按标准差归一化，"什么都不预测"的损失是目标的均方 {target_power:.3f}）
        刚初始化的模型                                  损失 {initial:.3f}

默认初始化的输出不是严格的 0，所以初始损失会比理论值略高一点，这是正常的。
要警惕的是量级不对：打印出来是 60 或 0.003，说明损失没取平均、标签错位，或归一化做了两遍。
    """)

    # ---- 台阶 3：过拟合一个批次 ----
    n_steps = 60 if is_quick() else 500
    losses = overfit_one_batch(cfg, data, norm, n_steps)
    ratio = losses[-1] / losses[0]
    print(f"台阶 3：过拟合一个批次（8 条序列，关掉 Dropout，训练 {n_steps} 步）。\n")
    print(f"    第 1 步的损失 {losses[0]:.4f}  →  第 {n_steps} 步的损失 {losses[-1]:.2e}（是初始的 {ratio:.2e}）")
    if ratio < 0.01:
        print("    损失降到了初始值的 1% 以下：模型的容量和梯度通路是通的。"
              "\n    这只说明代码没有大错，不说明方法有效。")
    elif is_quick():
        print("    冒烟模式只训练了几十步，损失还没降到初始值的 1% 以下，这个结果不作判断。")
    else:
        print("    ⚠️ 损失没有降到初始值的 1% 以下。先别往下走：一个批次都记不住，说明模型、损失或优化器里有地方不通。")

    # ---- 台阶 4：梯度流向 ----
    state = build_state(cfg, norm)
    obs, action = sample_batch(data, cfg, state["batch_rng"])
    prediction_loss(state["model"], obs, action, state["norm"]).backward()
    print("\n台阶 4：梯度流向。反向传播一次之后，各模块参数的梯度范数：\n")
    for name, module in [("inp", state["model"].inp), ("gru", state["model"].gru), ("out", state["model"].out)]:
        grad = math.sqrt(sum(float((p.grad ** 2).sum()) for p in module.parameters()))
        print(f"    model.{name:<5} 梯度范数 {grad:.4f}")
    ema_has_grad = any(p.grad is not None for p in state["ema"].parameters())
    print(f"    ema          {'⚠️ 有梯度' if ema_has_grad else '没有梯度（它只通过滑动平均更新，这是对的）'}")
    print("""
该有梯度的模块梯度不为零，不该有梯度的地方确实没有。世界模型里标了 sg(·) 的地方
（Dreamer 的 KL 平衡、JEPA 的目标编码器）都用这个办法检查一遍。
    """)
    return losses


# ==================== 第四部分：续训等价 ====================


def run_steps(state, data, cfg, n_steps):
    return [train_one_step(state, data, cfg)[0] for _ in range(n_steps)]


def check_resume_equivalence(cfg, data, norm, n, skip=(), reference=None):
    """续训等价测试。

    A：连续跑 2N 步。
    B：跑 N 步 → 存检查点 → 新建全部对象（相当于换了一个进程）→ 读检查点 → 再跑 N 步。
    逐步比较两条训练损失序列 [2N]，以及最后的 EMA 权重在验证集上的损失。
    skip 里列出的项在恢复时被跳过，用来模拟"检查点漏存了这一项"。
    """
    cfg = dataclasses.replace(cfg, total_steps=2 * n)
    if reference is None:
        state = build_state(cfg, norm)
        losses_a = run_steps(state, data, cfg, 2 * n)
        reference = (np.asarray(losses_a), float(evaluate(state["ema"], data, state["norm"], cfg).mean()))
    losses_a, val_a = reference

    state = build_state(cfg, norm)
    first = run_steps(state, data, cfg, n)
    path = OUT / f"02-resume-test{SUFFIX}.pt"
    save_checkpoint(path, state, cfg)
    state = build_state(cfg, norm)                                            # 新进程：一切从头新建
    load_checkpoint(path, state, cfg, skip=skip)
    losses_b = np.asarray(first + run_steps(state, data, cfg, n))
    val_b = float(evaluate(state["ema"], data, state["norm"], cfg).mean())

    diff = np.abs(losses_a - losses_b)                                        # [2N]
    return {"losses_a": losses_a, "losses_b": losses_b, "diff": diff, "max_diff": float(diff.max()),
            "val_a": val_a, "val_b": val_b, "reference": reference}


def resume_experiment(cfg, data, norm):
    """完整的检查点应当续训等价；逐项漏存，看各自造成什么后果。"""
    print("\n" + "=" * 60)
    print("第四部分：续训等价测试")
    print("=" * 60)

    n = 40 if is_quick() else 300
    print(f"""
连续跑 {2 * n} 步（A），和"跑 {n} 步、存检查点、新建全部对象、读检查点、再跑 {n} 步"（B）比较。
先用完整的检查点，再每次故意漏掉一项。设备：{cfg.device}。

    看三样：续训之后两条训练损失序列的最大差值；续训后头 20 步的平均损失（B 对 A）；
           最后 EMA 权重在验证集上的损失之差。
    """)
    variants = [("完整的检查点", ()), ("漏了优化器状态", ("optimizer",)), ("漏了调度器状态", ("scheduler",)),
                ("漏了随机数状态", ("rng",)), ("漏了 EMA 权重", ("ema",))]
    results, reference = {}, None
    for label, skip in variants:
        results[label] = check_resume_equivalence(cfg, data, norm, n, skip=skip, reference=reference)
        reference = results[label]["reference"]

    print(f"    {'检查点':<12}{'训练损失最大差值':>14}{'续训后 20 步的平均损失 B / A':>28}{'EMA 验证损失之差':>18}")
    for label, r in results.items():
        after_b, after_a = r["losses_b"][n:n + 20].mean(), r["losses_a"][n:n + 20].mean()
        print(f"    {label:<12}{r['max_diff']:>18.2e}{after_b:>18.4f} / {after_a:.4f}{abs(r['val_b'] - r['val_a']):>20.2e}")

    full = results["完整的检查点"]
    full_gap = max(full["max_diff"], abs(full["val_b"] - full["val_a"]))
    if full_gap == 0.0:
        print("\n  完整的检查点：两条损失序列逐位相同，差值正好是 0。检查点没有漏存任何会影响训练的状态。")
    elif full_gap < 1e-5:
        print(f"\n  完整的检查点：最大差值 {full_gap:.1e}，只剩浮点舍入的量级。可以认为续训等价。")
    else:
        print(f"\n  ⚠️ 完整的检查点也没有做到续训等价（最大差值 {full_gap:.1e}）。检查点里还漏了东西，或者有不确定的算子。")

    notes = {
        "漏了优化器状态": "Adam 的动量和二阶矩从零开始，头几步的步长和方向都不对",
        "漏了调度器状态": "学习率回到预热的起点重新爬升",
        "漏了随机数状态": "续训后取到的批次和 Dropout 掩码都变了，走上另一条轨迹（统计上等价，逐步对不上）",
        "漏了 EMA 权重": "训练本身不受影响，受影响的是用 EMA 权重做的评估",
    }
    print("\n  逐项漏存的后果（按上表的实测数字归类）：")
    for label, note in notes.items():
        r = results[label]
        train_changed, eval_changed = r["max_diff"] > 1e-6, abs(r["val_b"] - r["val_a"]) > 1e-6
        if train_changed:
            seen = "训练损失序列分叉"
        elif eval_changed:
            seen = "训练损失序列完全一样，只有 EMA 的验证损失变了（只盯着训练曲线发现不了）"
        else:
            seen = "这次运行里没有观察到差别"
        print(f"    {label}：{seen}。原因：{note}。")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    steps = np.arange(1, 2 * n + 1)
    lo, hi = max(n - 30, 0), min(n + 120, 2 * n)
    axes[0].plot(steps[lo:hi], full["losses_a"][lo:hi], color="black", linewidth=2.5, alpha=0.5, label="A：连续运行")
    for (label, r), color in zip(results.items(), ["C2", "C3", "C1", "C0", "C4"]):
        axes[0].plot(steps[lo:hi], r["losses_b"][lo:hi], color=color, linewidth=1, label=f"B：{label}")
        axes[1].plot(steps, r["diff"], color=color, linewidth=1, label=label)
    for ax in axes:
        ax.axvline(n, color="gray", linestyle=":")
        ax.set_xlabel("训练步数（竖线：中断并续训的位置）")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("训练损失（对数坐标）")
    axes[0].set_title("续训前后的训练损失")
    axes[0].legend(fontsize=7)
    axes[1].set_yscale("symlog", linthresh=1e-8)
    axes[1].set_ylabel("|A − B|（对称对数坐标，0 在最下面）")
    axes[1].set_title("两条损失序列逐步的差")
    axes[1].legend(fontsize=7)
    save_figure(fig, OUT / "02-resume-equivalence.png")
    return results


# ==================== 第五部分：完整的训练作业 ====================


def code_version():
    """代码版本：提交号，以及工作区有没有未提交的改动。"""
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10).stdout.strip())
        return {"commit": commit or "unknown", "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


def truncate_jsonl(path, max_step):
    """续训时丢掉检查点之后写下的日志行，否则这一段的步数会出现两遍。"""
    if path.exists():
        kept = [line for line in path.read_text().splitlines() if line and json.loads(line)["step"] <= max_step]
        path.write_text("".join(line + "\n" for line in kept))


def train_job(cfg, data, norm, run_dir, stop_at=None):
    """可中断、可重建的训练作业（CONCEPT.md 第 4.8 节的算法）。

    运行目录里有 last.pt 就从它续训。stop_at 用来模拟"进程在第 stop_at 步被杀"：直接返回，不做任何收尾。
    """
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / "checkpoints" / "last.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    state = build_state(cfg, norm)
    resumed_from = None
    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, state, cfg)                          # 里面会核对数据指纹
        resumed_from = state["step"]
        truncate_jsonl(run_dir / "metrics.jsonl", resumed_from)
        truncate_jsonl(run_dir / "eval.jsonl", resumed_from)

    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    meta = {"code": code_version(), "data_fingerprint": cfg.data_fingerprint, "seed": cfg.seed, "device": cfg.device,
            "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
            "command": " ".join(sys.argv), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "resumed_from_step": resumed_from, "status": "running"}
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    window_start, window_step = time.perf_counter(), state["step"]
    with open(run_dir / "metrics.jsonl", "a") as metrics_file, open(run_dir / "eval.jsonl", "a") as eval_file:
        while state["step"] < cfg.total_steps:
            if stop_at is not None and state["step"] >= stop_at:
                return {"status": "killed", "step": state["step"]}
            loss, grad_norm = train_one_step(state, data, cfg)
            step = state["step"]
            if step % cfg.log_every == 0:
                now = time.perf_counter()
                row = {"step": step, "loss/pred": loss, "grad_norm": grad_norm, "lr": state["scheduler"].get_last_lr()[0],
                       "steps_per_sec": (step - window_step) / (now - window_start)}
                metrics_file.write(json.dumps(row) + "\n")
                window_start, window_step = now, step
            if step % cfg.eval_every == 0 or step == cfg.total_steps:
                for episode, value in enumerate(evaluate(state["ema"], data, state["norm"], cfg)):
                    eval_file.write(json.dumps({"step": step, "episode": episode, "val_loss": float(value)}) + "\n")   # 每个回合一行
            if step % cfg.ckpt_every == 0:
                save_checkpoint(checkpoint_path, state, cfg)

    save_checkpoint(checkpoint_path, state, cfg)
    meta.update(status="finished", finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return {"status": "finished", "step": state["step"], "state": state,
            "val": evaluate(state["ema"], data, state["norm"], cfg)}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def job_experiment(cfg, data, norm):
    """跑一个完整的作业；再做一次"杀掉进程再续训"的演练，确认结果和不中断时一样。"""
    print("\n" + "=" * 60)
    print("第五部分：完整的训练作业，以及一次中断演练")
    print("=" * 60)

    shutil.rmtree(RUNS, ignore_errors=True)                                   # 这一节自己的产物，每次重新生成
    dir_a = RUNS / "gru-pendulum" / "uninterrupted" / f"seed-{cfg.seed}"
    dir_b = RUNS / "gru-pendulum" / "killed-and-resumed" / f"seed-{cfg.seed}"

    start = time.time()
    result_a = train_job(cfg, data, norm, dir_a)
    time_a = time.time() - start

    stop_at = int(0.55 * cfg.total_steps)
    killed = train_job(cfg, data, norm, dir_b, stop_at=stop_at)               # 第一个"进程"：跑到一半被杀
    result_b = train_job(cfg, data, norm, dir_b)                              # 第二个"进程"：同一条命令再执行一次
    meta_b = json.loads((dir_b / "meta.json").read_text())

    param_diff = max(float((pa - pb).abs().max()) for pa, pb in
                     zip(result_a["state"]["model"].parameters(), result_b["state"]["model"].parameters()))
    val_diff = float(np.abs(result_a["val"] - result_b["val"]).max())
    steps_b = [row["step"] for row in read_jsonl(dir_b / "metrics.jsonl")]
    expected_rows = cfg.total_steps // cfg.log_every
    log_ok = steps_b == sorted(set(steps_b)) and len(steps_b) == expected_rows

    files = sorted(str(p.relative_to(dir_a)) for p in dir_a.rglob("*") if p.is_file())
    print(f"""
作业 A（不中断）：{cfg.total_steps} 步，用时 {time_a:.1f} 秒。运行目录 {dir_a.relative_to(OUT)}/ 里有：
    {'、'.join(files)}

    config.json    合并后的完整配置
    meta.json      代码版本（提交 {json.loads((dir_a / 'meta.json').read_text())['code']['commit']}，工作区有未提交改动：{json.loads((dir_a / 'meta.json').read_text())['code']['dirty']}）、数据指纹、设备、依赖版本、启动命令、起止时间
    metrics.jsonl  每 {cfg.log_every} 步一行：损失、梯度范数、学习率、吞吐
    eval.jsonl     每次评估、每个验证回合一行（存原始记录，不存聚合后的均值）
    checkpoints/last.pt   每 {cfg.ckpt_every} 步原子写入一次

作业 B（演练）：第一次启动在第 {killed['step']} 步被"杀掉"（状态：{killed['status']}）；
再执行同一条命令，它发现 last.pt，从第 {meta_b['resumed_from_step']} 步接着跑到结束（状态：{result_b['status']}）。
第 {meta_b['resumed_from_step']} 步到第 {killed['step']} 步之间的工作白做了：检查点间隔决定了一次中断最多损失多少。

    A 与 B 最终模型参数的最大差值        {param_diff:.2e}
    A 与 B 最终验证损失（逐回合）的最大差值 {val_diff:.2e}
    B 的 metrics.jsonl                  {len(steps_b)} 行（应为 {expected_rows} 行），步数{'没有重复、没有缺口' if log_ok else ' ⚠️ 有重复或缺口'}
    """)
    if param_diff == 0.0 and val_diff == 0.0:
        print("  中断过一次的作业，最终结果和不中断的逐位相同。这次演练通过，可以放心把作业交给会被关机的机器。")
    elif max(param_diff, val_diff) < 1e-5:
        print("  两次结果只差浮点舍入的量级，演练通过。")
    else:
        print("  ⚠️ 中断续训后的结果和不中断的不一样。上云之前必须先找到原因。")

    rows = read_jsonl(dir_a / "metrics.jsonl")
    evals = read_jsonl(dir_a / "eval.jsonl")
    eval_steps = sorted({row["step"] for row in evals})
    eval_mean = [np.mean([row["val_loss"] for row in evals if row["step"] == s]) for s in eval_steps]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    axes[0].plot([r["step"] for r in rows], [r["loss/pred"] for r in rows], label="训练损失（当前权重，带 Dropout）")
    axes[0].plot(eval_steps, eval_mean, "o-", label="验证损失（EMA 权重，各验证回合的平均）")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("损失（对数坐标）")
    axes[0].set_title("由 metrics.jsonl 和 eval.jsonl 画出")
    axes[0].legend(fontsize=8)
    axes[1].plot([r["step"] for r in rows], [r["lr"] for r in rows], color="C3")
    axes[1].set_xlabel("训练步数")
    axes[1].set_ylabel("学习率")
    axes[1].set_title("学习率调度：线性预热 + 余弦下降")
    save_figure(fig, OUT / "02-training-job.png")
    print("  图是从运行目录里的 JSON 文件画出来的：图是构建产物，原始记录才是源文件。")
    return result_a


# ==================== 第六部分：混合精度 ====================


def mixed_precision_notes():
    """16 位浮点的数值范围（本机可验证）；自动混合精度的写法（未在本机验证）。"""
    print("\n" + "=" * 60)
    print("第六部分：混合精度")
    print("=" * 60)

    info = np.finfo(np.float16)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        sigma16, sigma32 = np.float16(0.003), np.float32(0.003)
        inv_var16, inv_var32 = np.float16(1.0) / (sigma16 * sigma16), np.float32(1.0) / (sigma32 * sigma32)
        prob16, prob32 = np.float16(1e-8), np.float32(1e-8)
        log16, log32 = np.log(prob16), np.log(prob32)
    print(f"""
混合精度让前向和反向的大部分运算用 16 位浮点，在支持的显卡上更快、更省显存。
代价是 16 位浮点的范围很窄：最大 {float(info.max):.0f}，最小的正规数 {float(info.tiny):.1e}。
世界模型的损失里有两类运算正好踩在这个范围的边上（下面是在本机用 NumPy 的 float16 实算的）：

    运算                               float16          float32
    高斯 NLL 里的 1/σ²（σ = 0.003）     {float(inv_var16):<16.6g} {float(inv_var32):.6g}
    对概率取对数 log p（p = 1e-8）       {float(log16):<16.6g} {float(log32):.6g}
    """)
    if np.isinf(inv_var16) and np.isinf(log16):
        print("  两项在 16 位下都成了无穷大，损失会直接变成 NaN。所以 KL、对数似然这些损失要放在 32 位下计算。")
    else:
        print("  ⚠️ 这两项在本机的 float16 下没有同时溢出，和预期不符；以上面打印的数值为准。")
    print("""
PyTorch 里的写法（train_one_step() 里 scaler 不为 None 的那条分支就是这样写的）：

    scaler = torch.amp.GradScaler("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        pred, _ = model(x, action)                       # 前向：16 位
    loss = ((pred.float() - target) ** 2).mean()         # 数值敏感的损失：转回 32 位再算
    scaler.scale(loss).backward()                        # 放大损失，防止小梯度在 16 位下变成 0
    scaler.unscale_(optimizer)                           # 梯度裁剪之前先缩放回来
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    scaler.step(optimizer)
    scaler.update()

    检查点里要多存一项 scaler.state_dict()（save_checkpoint() 里已经留了位置）。
    """)
    if torch.cuda.is_available():
        print("  检测到 CUDA。下面各跑一小段，对比开和不开混合精度（这段代码未在本机验证过）：")
        cfg = dataclasses.replace(make_config(), device="cuda", total_steps=300)
        data, norm, fingerprint = collect_dataset(cfg.n_episodes, cfg.seed)
        cfg = dataclasses.replace(cfg, data_fingerprint=fingerprint)
        for use_amp in (False, True):
            run_cfg = dataclasses.replace(cfg, use_amp=use_amp)
            state = build_state(run_cfg, norm)
            start = time.perf_counter()
            losses = run_steps(state, data, run_cfg, 300)
            print(f"    use_amp = {use_amp!s:<5}  {300 / (time.perf_counter() - start):7.1f} 步 / 秒   "
                  f"最后 50 步的平均损失 {np.mean(losses[-50:]):.4f}")
    else:
        print("  ⚠️ 未在本机验证：本机没有 NVIDIA 显卡，混合精度的提速幅度、开关前后损失曲线是否重合，"
              "\n     都要等租到 4090 之后再测（在有 CUDA 的机器上运行本讲义，这里会自动各跑一小段做对比）。"
              "\n     本机的 MPS 只负责调通，不依赖混合精度。")


# ==================== 第七部分：吞吐与预算 ====================


def measure_throughput(cfg, data, norm, n_steps, warmup=20):
    """丢掉预热部分后测吞吐。返回 {"train": 训练步 / 秒, "data": 只取批次时的批 / 秒}。"""
    state = build_state(cfg, norm)
    for _ in range(warmup):                                                   # 头几步有初始化和缓存开销，不计时
        train_one_step(state, data, cfg)
    start = time.perf_counter()
    for _ in range(n_steps):
        train_one_step(state, data, cfg)                                      # 里面的 loss.item() 会等设备算完
    train = n_steps / (time.perf_counter() - start)
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    for _ in range(n_steps):
        sample_batch(data, cfg, rng)
    return {"train": train, "data": n_steps / (time.perf_counter() - start)}


def bench_worker(overrides):
    """子进程入口：限制为 1 个线程，测训练吞吐，把结果以一行 JSON 打印出来。"""
    torch.set_num_threads(1)
    cfg = make_config(overrides)
    data, norm, fingerprint = collect_dataset(min(cfg.n_episodes, 10), cfg.seed)
    cfg = dataclasses.replace(cfg, data_fingerprint=fingerprint)
    result = measure_throughput(cfg, data, norm, n_steps=cfg.total_steps, warmup=10)
    print(json.dumps({"steps_per_sec": result["train"]}))


def measure_parallel_throughput(n_procs, n_steps):
    """同时启动 n_procs 个子进程各测一次训练吞吐，返回每个进程的步 / 秒。"""
    command = [sys.executable, os.path.abspath(__file__), "--bench-worker", f"total_steps={n_steps}"]
    procs = [subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(n_procs)]
    speeds = []
    for proc in procs:
        out, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"吞吐测量子进程失败：{err.strip()[-500:]}")
        speeds.append(json.loads(out.strip().splitlines()[-1])["steps_per_sec"])
    return speeds


def measure_action_time(model, norm, cfg, horizon=15, n_repeat=10):
    """规划器选一个动作的耗时 t_act：用学到的 GRU 模型给 CEM 的候选动作序列打分。"""
    device = torch.device(cfg.device)
    start_obs = torch.tensor([-1.0, 0.0, 0.0], device=device)                 # 摆垂在正下方

    def score_fn(actions):                                                    # actions [N, H, 1] → 分数 [N]
        obs, h = start_obs.expand(actions.shape[0], 3), None
        total = torch.zeros(actions.shape[0], device=device)
        for t in range(actions.shape[1]):
            a = actions[:, t]                                                 # [N, 1]
            theta = torch.atan2(obs[:, 1], obs[:, 0])
            total = total - (theta ** 2 + 0.1 * obs[:, 2] ** 2 + 0.001 * a[:, 0] ** 2)
            x = ((obs - norm["obs_mean"]) / norm["obs_std"])[:, None]         # [N, 1, 3]
            pred, h = model(x, a[:, None], h)
            obs = obs + pred[:, 0] * norm["delta_std"]
        return total

    mean = torch.zeros(horizon, 1, device=device)
    generator = torch.Generator(device=device).manual_seed(0)
    cem_plan(score_fn, mean, generator, action_low=-2.0, action_high=2.0, init_std=1.0)   # 预热
    start = time.perf_counter()
    for _ in range(n_repeat):
        cem_plan(score_fn, mean, generator, action_low=-2.0, action_high=2.0, init_std=1.0)
    return (time.perf_counter() - start) / n_repeat


def estimate_budget(n_step, v, n_eval, n_eval_episodes, episode_len, t_act, n_run, eta, n_final_episodes, price_per_hour=None):
    """CONCEPT.md 第 4.6 节的两个公式。时间的单位是小时。

        h = N_step / v + N_eval × E × T × t_act
        G = N_run × h ×（1 + η）+ N_run × E_final × T × t_act
    """
    train_hours = n_step / v / 3600
    eval_hours = n_eval * n_eval_episodes * episode_len * t_act / 3600
    h = train_hours + eval_hours
    matrix_hours = n_run * h * (1 + eta)
    final_eval_hours = n_run * n_final_episodes * episode_len * t_act / 3600
    total = matrix_hours + final_eval_hours
    return {"train_hours": train_hours, "eval_hours": eval_hours, "h": h, "matrix_hours": matrix_hours,
            "final_eval_hours": final_eval_hours, "total_hours": total,
            "cost": None if price_per_hour is None else total * price_per_hour}


def throughput_and_budget(cfg, data, norm, trained_state):
    """实测吞吐和选动作的耗时，代进预算公式。"""
    print("\n" + "=" * 60)
    print("第七部分：吞吐与预算")
    print("=" * 60)

    n_steps = 30 if is_quick() else 200
    cpu = measure_throughput(cfg, data, norm, n_steps)
    print(f"""
训练吞吐 v 和数据吞吐（批大小 {cfg.batch_size}，窗口 {cfg.window}，隐藏层 {cfg.hidden}；丢掉预热后测 {n_steps} 步）：

    设备      训练吞吐（步 / 秒）   只取批次（批 / 秒）
    cpu       {cpu['train']:>12.1f}   {cpu['data']:>14.1f}""")
    mps = None
    if torch.backends.mps.is_available():
        try:
            mps = measure_throughput(dataclasses.replace(cfg, device="mps"), data, norm, n_steps)
            print(f"    mps       {mps['train']:>12.1f}   {mps['data']:>14.1f}")
        except (RuntimeError, NotImplementedError) as err:
            print(f"    mps       运行失败，如实记录：{str(err).splitlines()[0][:80]}")
    print(f"\n  数据吞吐是训练吞吐的 {cpu['data'] / cpu['train']:.0f} 倍：这个作业的瓶颈在模型，不在数据。")
    if mps is not None:
        faster = "MPS" if mps["train"] > cpu["train"] else "CPU"
        ratio = max(mps["train"], cpu["train"]) / min(mps["train"], cpu["train"])
        print(f"  这个小 GRU 在{faster}上更快（{ratio:.1f} 倍）。哪个设备快取决于模型：批量大的卷积网络通常是 MPS 快，"
              "\n  小的循环网络常常是 CPU 快。所以吞吐只能实测，不能按「GPU 一定快」来估。")

    # 单机并行几个进程最划算
    counts = [1, 2] if is_quick() else [1, 2, 4]
    print("\n  同时跑几个进程最划算？每个进程限制为 1 个线程，各测一次训练吞吐（本机 CPU）：\n")
    print(f"    {'并行进程数':<10}{'每个进程（步 / 秒）':>18}{'合计（步 / 秒）':>16}")
    totals = {}
    for n_procs in counts:
        speeds = measure_parallel_throughput(n_procs, n_steps)
        totals[n_procs] = sum(speeds)
        print(f"    {n_procs:<14}{np.mean(speeds):>18.1f}{totals[n_procs]:>18.1f}")
    best = max(totals, key=totals.get)
    print(f"\n  合计吞吐最高的是 {best} 个进程（{totals[best]:.1f} 步 / 秒，是 1 个进程的 {totals[best] / totals[1]:.1f} 倍）。"
          f"\n  对照：不限线程数的单个进程是 {cpu['train']:.1f} 步 / 秒。"
          "\n  这些数字会随机器上同时在跑的其他任务波动，只说明一件事：并行几个最划算要在目标机上实测。"
          "\n  ⚠️ 未在本机验证：4090 上单卡并行 1、2、4 个进程的总吞吐、它和本机吞吐之比，都要租到机器之后再测。")

    # 规划器选一个动作的耗时
    model = trained_state["ema"]
    t_act = measure_action_time(model, trained_state["norm"], cfg, n_repeat=3 if is_quick() else 10)
    print(f"\n  选一个动作的耗时 t_act：CEM（200 条候选 × 5 轮，视野 15）用这个 GRU 模型打分，平均 {t_act * 1000:.1f} 毫秒 / 步。")

    # 公式的校验：CONCEPT.md 第 4.6 节的手算例子
    example = estimate_budget(n_step=60_000, v=12, n_eval=6, n_eval_episodes=10, episode_len=300, t_act=0.02,
                              n_run=25, eta=1.0, n_final_episodes=50, price_per_hour=2.0)
    assert abs(example["train_hours"] - 1.389) < 0.001 and abs(example["eval_hours"] - 0.100) < 0.001
    assert abs(example["final_eval_hours"] - 2.083) < 0.001 and 76 < example["total_hours"] < 78
    slow = estimate_budget(n_step=60_000, v=12, n_eval=6, n_eval_episodes=10, episode_len=300, t_act=0.5,
                           n_run=25, eta=1.0, n_final_episodes=50)
    assert abs(slow["final_eval_hours"] - 52.08) < 0.01
    print(f"""
预算公式先用 CONCEPT.md 第 4.6 节的手算例子校验（v = 12 步 / 秒，t_act = 0.02 秒，25 次训练，η = 1）：
    单次训练 h = {example['train_hours']:.2f} + {example['eval_hours']:.2f} = {example['h']:.2f} 小时；合计 {example['total_hours']:.1f} 小时，与手算的"约 77 小时"一致。
    把 t_act 换成 0.5 秒：收尾评估 {slow['final_eval_hours']:.0f} 小时，与手算的"约 52 小时"一致。
    """)

    plan = EXAMPLE_PLAN
    rows = [("用本机实测的 v 和 t_act", cpu["train"], t_act),
            ("规划器慢 10 倍（t_act × 10）", cpu["train"], t_act * 10),
            ("模型慢 10 倍（v ÷ 10）", cpu["train"] / 10, t_act)]
    print(f"""把本机实测的数字代进去，演示公式怎么用。设想的实验规模（EXAMPLE_PLAN，示例值）：
    每次训练 {plan['n_step']} 步，训练中评估 {plan['n_eval']} 次 × {plan['n_eval_episodes']} 回合，回合长 {plan['episode_len']} 步；
    实验矩阵 {plan['n_run']} 次训练，返工系数 η = {plan['eta']}，收尾评估每次训练 {plan['n_final_episodes']} 回合；
    单价按 {PRICE_PER_GPU_HOUR} 元 / 小时（PRICE_PER_GPU_HOUR，示例值，不是任何平台的报价）。

    {'情形':<24}{'v':>8}{'t_act（毫秒）':>14}{'单次 h（小时）':>14}{'矩阵（小时）':>12}{'收尾评估':>10}{'合计':>8}{'费用（元）':>10}""")
    budgets = []
    for label, v, t in rows:
        b = estimate_budget(v=v, t_act=t, price_per_hour=PRICE_PER_GPU_HOUR, **plan)
        budgets.append(b)
        print(f"    {label:<24}{v:>8.1f}{t * 1000:>14.1f}{b['h']:>16.2f}{b['matrix_hours']:>14.1f}{b['final_eval_hours']:>12.1f}{b['total_hours']:>10.1f}{b['cost']:>12.0f}")
    base = budgets[0]
    eval_share = (base["eval_hours"] * plan["n_run"] * (1 + plan["eta"]) + base["final_eval_hours"]) / base["total_hours"]
    costlier = "规划器" if budgets[1]["total_hours"] > budgets[2]["total_hours"] else "模型"
    print(f"""
  第一行里，评估（训练中的定期评估 + 收尾评估）占总预算的 {eval_share:.0%}。
  规划器慢 10 倍，总预算从 {base['total_hours']:.1f} 小时变成 {budgets[1]['total_hours']:.1f} 小时；模型慢 10 倍，变成 {budgets[2]['total_hours']:.1f} 小时。
  在这组实测数字下，{costlier}变慢对总预算的影响更大。v 和 t_act 两头都要测：决策时规划的方法，评估成本常常被低估。

  ⚠️ 这张表里的 v 是本机 CPU 上一个小 GRU 的吞吐，只用来演示公式。毕业项目的预算要用
     目标机器（4090）上、目标模型的实测吞吐重算一遍，不能拿本机的数字乘一个"大概快多少倍"的系数。
    """)
    return cpu, mps, totals, t_act


# ==================== 练习与思考 ====================


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：给检查点加一项，再让续训等价测试失败
    给模型加一个"输入归一化的滑动统计量"（每一步用当前批次的均值、方差做滑动更新），
    先不把它存进检查点，跑 check_resume_equivalence()，看差值；再补上，看差值回到 0。

练习 1 参考答案：
    把统计量注册成 buffer（register_buffer），它会自动进入 model.state_dict()，也就自动被保存。
    如果把它放在模型之外的普通 Python 变量里，就必须手动加进检查点。
    漏存时，续训后模型看到的输入尺度突然变了，训练损失会跳一下再慢慢恢复。
    经验规则：任何在训练中会变、又会影响后续训练的东西，都要进检查点；
    续训等价测试就是用来替你数"有没有漏"的。

练习 2：把检查点间隔改成每 50 步一次
    修改 ckpt_every=50，重新做第五部分的演练。白做的步数怎么变？保存本身的开销怎么估？

练习 2 参考答案：
    一次中断最多损失 ckpt_every 步的工作，间隔越短损失越小。
    代价是每次保存都要把全部状态写一遍盘：用 time.perf_counter() 包住 save_checkpoint() 测一次，
    乘以保存次数，和总训练时间比。这个小模型的检查点很小，开销可以忽略；
    带回放缓冲区的在线方法检查点可以有几个 GB，间隔就要按"保存耗时不超过训练时间的百分之几"来定。

练习 3：在 MPS 上做续训等价测试
    把 device 改成 mps 重新跑第四部分。完整的检查点还能做到差值为 0 吗？

练习 3 参考答案：
    不一定。检查点里存的 torch.get_rng_state() 是 CPU 随机数发生器的状态；
    在 MPS（和 CUDA）上，Dropout 用的是设备自己的随机数发生器，需要另外保存和恢复
    （CUDA 上是 torch.cuda.get_rng_state_all()）。此外设备上的部分算子本身不保证逐位可复现。
    所以跨设备、甚至同一块 GPU 上，通常只要求"差值在浮点舍入量级"或"统计一致"，不要求逐位相同。
    把 dropout 设为 0 再测一次，可以分清差值来自随机数还是来自算子。

思考题 1：为什么要把"合并后的完整配置"写进运行目录，而不是只记下命令行？

思考题 1 参考答案：
    命令行只记了"和默认值不同的地方"。默认值写在代码里，会随着项目推进被修改。
    三周后想重建第一周的那次运行，只凭命令行得到的是"新的默认值 + 旧的覆盖"，不是当时的配置。
    完整配置加上代码版本，才能唯一确定当时跑的是什么。

思考题 2：续训等价测试通过了，是不是就能保证云端的作业可以放心中断？

思考题 2 参考答案：
    还差两件事。一是测试和实际用的要是同一条代码路径：这一节的 train_job() 被中断后再次调用，
    走的就是正常启动的路径，而不是另写的一个"恢复脚本"。二是检查点要写在实例被释放后还在的位置。
    所以上云后的第一件事，是在那台机器上再做一次"杀掉进程再续训"的演练。

思考题 3：评估很贵时，能不能把训练中的定期评估省掉，只做收尾评估？

思考题 3 参考答案：
    可以减少，不建议完全省掉。定期评估的作用是尽早发现"训练损失在降、控制效果没有变好"
    这类问题（目标不匹配）。折中的办法：训练中用很少的回合（比如 3 个）看趋势，
    收尾时再用完整的回合数出正式结果。两者用的评估种子要事先定好，
    而且正式结果只报告最后一个检查点，不从定期评估里挑最高的。
    """)


# ==================== 主函数 ====================


def main():
    parser = argparse.ArgumentParser(description="把训练变成可中断、可重建的作业")
    parser.add_argument("overrides", nargs="*", help="配置覆盖，例如 lr=3e-4 hidden=256")
    parser.add_argument("--bench-worker", action="store_true", help="内部使用：作为子进程测一次训练吞吐")
    args = parser.parse_args()
    if args.bench_worker:
        bench_worker(args.overrides)
        return

    introduction()
    cfg = make_config(args.overrides)
    data, norm, fingerprint = collect_dataset(cfg.n_episodes, cfg.seed)
    cfg = dataclasses.replace(cfg, data_fingerprint=fingerprint)
    config_demo(cfg)
    print(f"  数据：{data['train_obs'].shape[0]} 个训练回合、{data['val_obs'].shape[0]} 个验证回合，"
          f"每个 {data['train_action'].shape[1]} 步；数据指纹 {fingerprint}")

    debugging_ladder(cfg, data, norm)
    resume_experiment(cfg, data, norm)
    result = job_experiment(cfg, data, norm)
    mixed_precision_notes()
    throughput_and_budget(cfg, data, norm, result["state"])
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：03-evaluation-protocol.py（评估协议：种子、区间、开环与闭环）")
    print("=" * 60)


if __name__ == "__main__":
    main()
