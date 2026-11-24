"""
消融实验与结果的整理
====================

⚠️ 草稿状态：本文件写成时运行环境不允许执行 Python，尚未实际运行过。
   定稿前必须完整跑一遍、核对输出，并把下面的「预计运行时间」换成实测值。
   文件里的「预注册」那张表确实是在任何一次运行之前写下的。

学习目标：
    1. 把方法里的每个部件做成配置里的一个开关，用 build_ablation_matrix() 生成实验矩阵
    2. 在跑实验之前写下每组消融的预期，跑完之后逐条对照：一致、相反，还是分不出来
    3. 算一组 2×2 的相互作用及其区间，看到"某个部件贡献了多少"为什么没有唯一答案
    4. 从运行目录里的原始记录生成带区间的学习曲线、有效数字正确的结果表
    5. 把每一句结论和它的证据对上，按证据的强弱决定措辞

核心概念：
    - 消融（ablation）：去掉或替换一个部件，其余不变，看结果怎么变
    - 去掉一个（leave-one-out）/ 加上一个（add-one-in）/ 全因子
    - 相互作用（interaction）= y11 − y10 − y01 + y00
    - 预注册（preregistration）：先写断言，再跑测试
    - 有效数字跟着不确定性走；图和表是从原始记录生成的构建产物

前置知识：
    - 03-evaluation-protocol.py（t 区间、差的区间；这里把要用的两个函数再写一遍）
    - Phase 1 的 03-learning-dynamics-mlp.py（预测变化量、角度用 cos / sin）、06-cem-and-mppi.py（热启动）

对应概念文档：
    - CONCEPT.md 第 7 节「消融实验」、第 8 节「写作与展示」

预计运行时间：待实测（本稿尚未运行）
"""

import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.planning import cem_plan, shift_plan

plt = setup_matplotlib()
OUT = output_dir("phase-12")
SUFFIX = "-quick" if is_quick() else ""
RUNS = OUT / f"04-runs{SUFFIX}" / "pendulum-ablation"
MAX_TORQUE, MAX_SPEED, DT = 2.0, 8.0, 0.05
# 模型是两层 128 单元的 MLP，规划时的批量只有 100，放在 CPU 上更快


# ==================== 第一部分：消融是什么 ====================


def introduction():
    """消融 = 把每个部件做成开关，在其他条件相同的情况下比较开和关。"""
    print("=" * 60)
    print("第一部分：消融实验要回答什么")
    print("=" * 60)

    print("""
你的方法比基线多了几个部件，结果变好了。是哪个部件的功劳？哪个是白加的？
消融实验逐个去掉部件，看结果掉多少。它和排查线上回退时的二分是同一个思路。

这一节的"方法"是 Phase 1 那套小的基于模型的控制器，在 Pendulum 上：

    3 个随机回合的数据 ──▶ 训练 MLP 动力学模型 ──▶ CEM-MPC 控制真实环境 ──▶ 回合回报

它有三个可以单独开关的部件，都是 Phase 1 里讲过的技巧：

    开关               打开（完整方法）                  关掉
    predict_delta      模型预测状态的变化量 Δs            模型直接预测下一个状态
    angle_features     角度以 (cos θ, sin θ) 输入        角度以折到 [−π, π) 的原始值输入
    warm_start         CEM 用上一步的规划结果热启动        每一步都从零开始规划

流程：

    写下预期（预注册）
         │
         ▼
    build_ablation_matrix()          完整方法 + 每个因素关掉一次 + 一组 2×2 + 基线（全关）
         │                           每格 n 个训练种子，评估协议完全相同
         ▼
    每一格 = 一次普通的训练作业       runs/<实验名>/<格子名>/seed-<k>/{config.json, eval.jsonl, meta.json}
         │
         ▼
    分析（只读运行目录里的原始记录）
         ├─ 去掉一个：与完整方法之差及其区间 ──▶ 对照预期
         ├─ 2×2：interaction_effect()
         └─ 写作：plot_learning_curves()、format_with_uncertainty()、make_results_table()
    """)


# ==================== 统计工具（第一次讲解在 03-evaluation-protocol.py）====================


def t_critical(df, confidence=0.95):
    """t 分布的双侧临界值 κ：对概率密度做梯形积分得到累积分布，再用二分法求分位数。"""
    log_c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)

    def cdf(x):
        grid = np.linspace(0.0, x, 20001)
        pdf = np.exp(log_c - (df + 1) / 2 * np.log1p(grid ** 2 / df))
        return 0.5 + float(((pdf[1:] + pdf[:-1]) / 2).sum() * (grid[1] - grid[0]))

    lo, hi = 0.0, 500.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < 0.5 + confidence / 2 else (lo, mid)
    return (lo + hi) / 2


def mean_and_ci(x, confidence=0.95):
    """均值与 t 区间。x [n]：各个训练种子的得分。"""
    x = np.asarray(x, dtype=np.float64)
    mean, se = x.mean(), x.std(ddof=1) / math.sqrt(len(x))
    half = t_critical(len(x) - 1, confidence) * se
    return {"mean": mean, "se": se, "half": half, "lo": mean - half, "hi": mean + half, "n": len(x)}


def diff_ci(a, b, confidence=0.95):
    """δ = mean(b) − mean(a) 的区间，自由度用 Welch 近似。"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    denom = va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)
    df = (va + vb) ** 2 / denom if denom > 0 else len(a) + len(b) - 2
    half = t_critical(df, confidence) * se
    delta = b.mean() - a.mean()
    return {"delta": delta, "se": se, "df": df, "half": half, "lo": delta - half, "hi": delta + half}


# ==================== 第二部分：配置、开关、实验矩阵 ====================


@dataclass(frozen=True)
class AgentConfig:
    """一次运行的全部配置。前三项是要消融的开关，其余在所有格子里都相同。"""
    predict_delta: bool = True      # 部件 A
    angle_features: bool = True     # 部件 B
    warm_start: bool = True         # 部件 C
    # 数据与训练
    n_data_episodes: int = 3
    train_steps: int = 1200
    hidden: int = 128
    lr: float = 1e-3
    batch_size: int = 128
    # 规划
    plan_horizon: int = 20
    n_candidates: int = 100
    n_elites: int = 10
    cem_iters: int = 3
    # 评估
    episode_len: int = 200
    n_eval_episodes: int = 4
    n_eval_points: int = 1          # 训练过程中评估几次（等间隔，最后一次在训练结束时）


FACTORS = {"predict_delta": "预测变化量", "angle_features": "cos / sin 角度输入", "warm_start": "热启动"}

# 预注册：下面这张表是在跑任何实验之前写的。direction 是"关掉这个部件之后，回报预期怎么变"。
PREREGISTRATION = [
    {"factor": "predict_delta", "direction": "下降", "size": "中",
     "reason": "每一步的状态变化很小。直接预测下一个状态，网络得先学会把输入几乎原样输出，剩下的精度才用在变化量上"},
    {"factor": "angle_features", "direction": "下降", "size": "中到大",
     "reason": "原始角度在 ±π（正下方）处不连续，而摆的起点大多在下半圈，甩起来的过程要反复经过那里"},
    {"factor": "warm_start", "direction": "下降", "size": "小",
     "reason": "不热启动等于每一步都丢掉上一步的优化结果；但每步仍有 3 轮 CEM，摆的动作只有 1 维，可能够用"},
]
PREREGISTERED_INTERACTION = "A（预测变化量）× B（cos / sin 输入）：方向不确定。两者解决的是不同的问题（输出的尺度、输入的连续性），预期接近可加"


def build_ablation_matrix(full, factors, seeds, pairs=(), include_baseline=True):
    """生成实验矩阵。

    输入：完整方法的配置 full、要消融的开关名 factors、训练种子编号 seeds、要做 2×2 的因素对 pairs。
    输出：格子的列表，每格是 {"name", "config", "off"（关掉了哪些开关）, "seeds"}：
        完整方法；每个因素关掉一次（去掉一个）；每个因素对同时关掉（2×2 里缺的那一格）；基线（全部关掉）。
    每个格子的配置和完整方法相比，只在 off 列出的开关上不同。这一点由断言保证。
    """
    cells = [("完整方法", full, ())]
    for factor in factors:
        cells.append((f"去掉 {FACTORS[factor]}", replace(full, **{factor: False}), (factor,)))
    for a, b in pairs:
        cells.append((f"去掉 {FACTORS[a]} 和 {FACTORS[b]}", replace(full, **{a: False, b: False}), (a, b)))
    if include_baseline:
        cells.append(("基线（全部关掉）", replace(full, **{factor: False for factor in factors}), tuple(factors)))
    matrix = []
    for name, config, off in cells:
        changed = sorted(key for key, value in asdict(config).items() if value != getattr(full, key))
        assert changed == sorted(off), f"格子「{name}」和完整方法相比多改了东西：{changed}"
        matrix.append({"name": name, "config": config, "off": off, "seeds": list(seeds)})
    return matrix


def slug(name):
    """格子名 → 目录名。"""
    table = {"完整方法": "full", "基线（全部关掉）": "baseline"}
    if name in table:
        return table[name]
    return "no-" + "-".join(key.replace("_", "") for key, label in FACTORS.items() if label in name)


# ==================== 摆、模型、规划（Phase 1 的部件，紧凑版）====================


def wrap_angle(theta):
    return ((theta + math.pi) % (2 * math.pi)) - math.pi


def true_step(states, actions):
    """摆的真实动力学。states [N, 2] = (θ, θ̇)，actions [N, 1] → [N, 2]。"""
    theta, theta_dot = states[..., 0], states[..., 1]
    u = actions[..., 0].clamp(-MAX_TORQUE, MAX_TORQUE)
    theta_dot_new = (theta_dot + (15.0 * torch.sin(theta) + 3.0 * u) * DT).clamp(-MAX_SPEED, MAX_SPEED)
    return torch.stack([theta + theta_dot_new * DT, theta_dot_new], dim=-1)


def reward_fn(states, actions):
    return -(wrap_angle(states[..., 0]) ** 2 + 0.1 * states[..., 1] ** 2 + 0.001 * actions[..., 0] ** 2)


class DynamicsModel(nn.Module):
    """MLP 动力学模型。predict_delta 和 angle_features 两个开关决定输出和输入的形式。"""

    def __init__(self, cfg):
        super().__init__()
        self.predict_delta, self.angle_features = cfg.predict_delta, cfg.angle_features
        in_dim = 4 if cfg.angle_features else 3          # (cos θ, sin θ, θ̇, u) 或 (θ, θ̇, u)
        out_dim = 2 if cfg.predict_delta else 3          # (Δθ, Δθ̇) 或 (cos θ', sin θ', θ̇')
        self.net = nn.Sequential(nn.Linear(in_dim, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(),
                                 nn.Linear(cfg.hidden, out_dim))
        self.register_buffer("y_mean", torch.zeros(out_dim))
        self.register_buffer("y_std", torch.ones(out_dim))

    def inputs(self, states, actions):
        """states [N, 2]，actions [N, 1] → 网络输入 [N, 4] 或 [N, 3]。"""
        theta, theta_dot = states[..., :1], states[..., 1:]
        angle = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1) if self.angle_features else wrap_angle(theta) / math.pi
        return torch.cat([angle, theta_dot / MAX_SPEED, actions / MAX_TORQUE], dim=-1)

    def targets(self, states, next_states):
        """回归目标（物理单位）：变化量 [N, 2]，或下一个状态的 (cos θ', sin θ', θ̇') [N, 3]。"""
        if self.predict_delta:
            return next_states - states
        theta = next_states[..., :1]
        return torch.cat([torch.cos(theta), torch.sin(theta), next_states[..., 1:]], dim=-1)

    def raw(self, states, actions):
        return self.net(self.inputs(states, actions))

    def forward(self, states, actions):
        """states [N, 2]，actions [N, 1] → 预测的下一状态 [N, 2]。"""
        out = self.raw(states, actions) * self.y_std + self.y_mean
        if self.predict_delta:
            theta, theta_dot = states[..., 0] + out[..., 0], states[..., 1] + out[..., 1]
        else:
            theta, theta_dot = torch.atan2(out[..., 1], out[..., 0]), out[..., 2]
        return torch.stack([theta, theta_dot.clamp(-MAX_SPEED, MAX_SPEED)], dim=-1)


def collect_random_episodes(n_episodes, seed):
    """随机策略采集。返回转移 (s [N, 2], a [N, 1], s' [N, 2])。"""
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)
    s, a, s_next = [], [], []
    for i in range(n_episodes):
        env.reset(seed=seed + i)
        for _ in range(200):
            s.append(env.unwrapped.state.copy())
            action = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=1).astype(np.float32)
            env.step(action)
            a.append(action)
            s_next.append(env.unwrapped.state.copy())
    env.close()
    return [torch.tensor(np.asarray(x), dtype=torch.float32) for x in (s, a, s_next)]


def run_mpc_episode(step_fn, cfg, eval_seed):
    """用 CEM-MPC 在真实环境里跑一个回合，返回真实回报。step_fn 为 None 时用随机策略。"""
    env = gym.make("Pendulum-v1")
    env.reset(seed=eval_seed)
    generator = torch.Generator().manual_seed(eval_seed)
    mean, total = torch.zeros(cfg.plan_horizon, 1), 0.0
    for _ in range(cfg.episode_len):
        if step_fn is None:
            action = (torch.rand(1, generator=generator) * 2 - 1) * MAX_TORQUE
        else:
            state = torch.as_tensor(env.unwrapped.state, dtype=torch.float32)

            def score_fn(actions):                                       # actions [N, H, 1] → [N]
                states, score = state.expand(actions.shape[0], 2), torch.zeros(actions.shape[0])
                for h in range(actions.shape[1]):
                    score = score + reward_fn(states, actions[:, h])
                    states = step_fn(states, actions[:, h])
                return score

            mean = cem_plan(score_fn, mean, generator, action_low=-MAX_TORQUE, action_high=MAX_TORQUE,
                            n_candidates=cfg.n_candidates, n_elites=cfg.n_elites, n_iters=cfg.cem_iters, init_std=1.0, min_std=0.1)
            action = mean[0]
            # 开关 C：热启动（把规划结果平移一格留给下一步），或每一步都从零开始
            mean = shift_plan(mean) if cfg.warm_start else torch.zeros_like(mean)
        _, reward, _, _, _ = env.step(action.numpy().astype(np.float32))
        total += float(reward)
    env.close()
    return total


def run_job(cfg, seed, run_dir, eval_seeds):
    """矩阵里的一格、一个种子：采数据 → 训练 → 按协议评估，逐回合写进 eval.jsonl。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({**asdict(cfg), "seed": seed}, ensure_ascii=False, indent=2))

    # 训练种子同时决定：采到哪些数据、网络的初始化、取批次的顺序
    s, a, s_next = collect_random_episodes(cfg.n_data_episodes, seed=100 * (seed + 1))
    torch.manual_seed(seed)
    model = DynamicsModel(cfg)
    y = model.targets(s, s_next)
    model.y_mean.copy_(y.mean(dim=0))
    model.y_std.copy_(y.std(dim=0).clamp_min(1e-6))
    target = (y - model.y_mean) / model.y_std
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    generator = torch.Generator().manual_seed(seed)
    eval_at = {round(cfg.train_steps * (i + 1) / cfg.n_eval_points) for i in range(cfg.n_eval_points)}

    train_seconds, start = 0.0, time.perf_counter()
    with open(run_dir / "eval.jsonl", "w") as f:
        for step in range(1, cfg.train_steps + 1):
            idx = torch.randint(0, len(s), (cfg.batch_size,), generator=generator)
            loss = ((model.raw(s[idx], a[idx]) - target[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step in eval_at:
                train_seconds += time.perf_counter() - start
                for eval_seed in eval_seeds:                             # 每个评估回合一行，不存聚合后的均值
                    row = {"step": step, "episode_seed": eval_seed, "return": run_mpc_episode(model, cfg, eval_seed)}
                    f.write(json.dumps(row) + "\n")
                start = time.perf_counter()
    meta = {"n_params": sum(p.numel() for p in model.parameters()), "train_seconds": train_seconds,
            "final_train_loss": loss.item(), "torch": torch.__version__}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# ==================== 从运行目录读原始记录 ====================


def load_cell(cell_dir):
    """读一个格子下所有种子的 eval.jsonl。返回评估步数 [P] 和种子层得分 [n, P]（每个点是 E 个回合的平均）。"""
    curves, steps = [], None
    for seed_dir in sorted(Path(cell_dir).glob("seed-*"), key=lambda p: int(p.name.split("-")[1])):
        rows = [json.loads(line) for line in (seed_dir / "eval.jsonl").read_text().splitlines() if line]
        steps = sorted({row["step"] for row in rows})
        curves.append([np.mean([row["return"] for row in rows if row["step"] == step]) for step in steps])
    return np.array(steps), np.array(curves)


def final_scores(cell_dir):
    """最后一个检查点的种子层得分 [n]。协议里固定报告最后一个，不挑。"""
    return load_cell(cell_dir)[1][:, -1]


def cell_meta(cell_dir):
    metas = [json.loads((d / "meta.json").read_text()) for d in sorted(Path(cell_dir).glob("seed-*"))]
    return {"n_params": metas[0]["n_params"], "train_seconds": float(np.mean([m["train_seconds"] for m in metas]))}


def run_matrix(matrix, curve_cells, n_eval_points, eval_seeds):
    """把矩阵里的每一格、每个种子跑一遍。curve_cells 里的格子在训练过程中多评估几次，用来画学习曲线。"""
    print("\n" + "=" * 60)
    print("第三部分：运行实验矩阵")
    print("=" * 60)
    shutil.rmtree(RUNS, ignore_errors=True)                              # 这一节自己的产物，每次重新生成
    n_runs = sum(len(cell["seeds"]) for cell in matrix)
    full = matrix[0]["config"]
    print(f"""
  {len(matrix)} 个格子 × {len(matrix[0]['seeds'])} 个训练种子 = {n_runs} 次训练。每次：{full.n_data_episodes} 个随机回合的数据，训练 {full.train_steps} 步，
  然后在 {len(eval_seeds)} 个固定的评估种子（{eval_seeds[0]}–{eval_seeds[-1]}）上各跑一个 {full.episode_len} 步的回合。
  「完整方法」和「基线」在训练过程中评估 {n_eval_points} 次（画学习曲线用），其余格子只评估最后一个检查点。
    """)
    print(f"    {'格子':<28}{'关掉的开关':<34}{'参数量':>8}{'训练（秒 / 次）':>14}{'整格用时（秒）':>14}")
    dirs = {}
    for cell in matrix:
        cfg = replace(cell["config"], n_eval_points=n_eval_points) if cell["name"] in curve_cells else cell["config"]
        cell_dir = RUNS / slug(cell["name"])
        start = time.time()
        for seed in cell["seeds"]:
            run_job(cfg, seed, cell_dir / f"seed-{seed}", eval_seeds)
        dirs[cell["name"]] = cell_dir
        meta = cell_meta(cell_dir)
        print(f"    {cell['name']:<28}{', '.join(cell['off']) or '—':<38}{meta['n_params']:>8}{meta['train_seconds']:>14.1f}{time.time() - start:>16.0f}")
    params = {name: cell_meta(d)["n_params"] for name, d in dirs.items()}
    if len(set(params.values())) > 1:
        full_params = params[matrix[0]["name"]]
        rel = max(abs(value - full_params) for value in params.values()) / full_params
        size = "可以忽略" if rel < 0.02 else "已经不算小，解读结果时要记着这一点"
        print(f"""
  注意「参数量」一列不完全相同（{min(params.values())} 到 {max(params.values())}）。关掉 predict_delta 时输出从 2 维变成 3 维，
  关掉 angle_features 时输入从 4 维变成 3 维。"只改一个开关"几乎总会带来这样的连带变化。
  这里和完整方法相比最多差 {rel:.2%}，{size}；差别大的时候（比如去掉解码器），要加一组参数量对齐的对照。""")
    return dirs


# ==================== 第四部分：去掉一个，对照预期 ====================


def leave_one_out_analysis(matrix, dirs, eval_seeds):
    """每个因素关掉一次：与完整方法之差及其区间；和预注册的预期逐条对照。"""
    print("\n" + "=" * 60)
    print("第四部分：去掉一个，对照预期")
    print("=" * 60)

    print("\n  预注册（跑实验之前写在代码里的 PREREGISTRATION）：\n")
    for item in PREREGISTRATION:
        print(f"    去掉 {FACTORS[item['factor']]}：预期{item['direction']}，幅度{item['size']}。理由：{item['reason']}。")
    print(f"    2×2 的相互作用：{PREREGISTERED_INTERACTION}。")

    cfg = matrix[0]["config"]
    oracle = np.array([run_mpc_episode(true_step, cfg, s) for s in eval_seeds])
    random_policy = np.array([run_mpc_episode(None, cfg, s) for s in eval_seeds])
    full = final_scores(dirs["完整方法"])
    full_stats = mean_and_ci(full)
    n, n_eval = len(full), len(eval_seeds)
    print(f"""
  参照（没有训练种子，只在同一批评估回合上跑一遍）：随机策略 {random_policy.mean():.1f}；真实模型 + 同一个规划器（上界）{oracle.mean():.1f}。
  完整方法：{format_with_uncertainty(full_stats['mean'], full_stats['half'])}（{n} 个训练种子的均值 ± 95% t 区间半宽，每个种子 {n_eval} 个评估回合）。

  消融得分的形状是 [k+1, n] = [{len(PREREGISTRATION) + 1}, {n}]：第 0 行是完整方法，其余每行去掉一个因素。
    """)
    print(f"    {'消融':<24}{'各训练种子的得分':<{9 * n + 2}}{'均值':>8}{'与完整方法之差':>14}{'差的 95% 区间':>22}   {'预期':<6}{'对照'}")
    print(f"    {'完整方法':<24}{'  '.join(f'{v:7.1f}' for v in full):<{9 * n + 2}}{full.mean():>10.1f}")
    verdicts = {}
    for item in PREREGISTRATION:
        name = f"去掉 {FACTORS[item['factor']]}"
        scores = final_scores(dirs[name])
        d = diff_ci(full, scores)                                        # δ = 去掉之后 − 完整方法
        observed = "下降" if d["hi"] < 0 else ("上升" if d["lo"] > 0 else "分不出来")
        verdict = "分不出来" if observed == "分不出来" else ("一致" if observed == item["direction"] else "相反")
        verdicts[item["factor"]] = {"name": name, "scores": scores, "diff": d, "observed": observed, "verdict": verdict}
        interval = f"[{d['lo']:.1f}, {d['hi']:.1f}]"
        print(f"    {name:<24}{'  '.join(f'{v:7.1f}' for v in scores):<{9 * n + 2}}{scores.mean():>10.1f}{d['delta']:>16.1f}{interval:>24}   {item['direction']:<6}{verdict}")

    print("\n  逐条解读（由上表的区间生成）：")
    for item in PREREGISTRATION:
        v = verdicts[item["factor"]]
        d = v["diff"]
        label = FACTORS[item["factor"]]
        if v["verdict"] == "一致":
            print(f"    - {label}：去掉后回报变化 {d['delta']:.1f}，区间不含 0，方向和预期一致。这个部件在完整方法里是必要的。")
        elif v["verdict"] == "相反":
            print(f"    - {label}：去掉后回报变化 {d['delta']:+.1f}，区间不含 0，方向和预期相反。先查实现有没有 bug，"
                  "\n      再查去掉它时有没有连带改变别的东西；都排除之后，修正对这个部件的理解。")
        else:
            print(f"    - {label}：差的区间 [{d['lo']:.1f}, {d['hi']:.1f}] 包含 0。{n} 个种子下分不出它有没有用。"
                  f"\n      这不等于「没有用」：区间的半宽是 {d['half']:.1f}，比它小的作用这个实验看不见。"
                  "要么加种子，要么如实写「在这个任务上没有观察到作用」。")
    n_undecided = sum(v["verdict"] == "分不出来" for v in verdicts.values())
    if n_undecided == len(verdicts):
        headroom = oracle.mean() - full.mean()
        print(f"\n  ⚠️ 三组消融都分不出来。对照陷阱清单里的那一行：任务太简单，或者种子太少。"
              f"\n  完整方法离上界还有 {headroom:.1f}，各组差值区间的半宽在 "
              f"{min(v['diff']['half'] for v in verdicts.values()):.1f} 到 {max(v['diff']['half'] for v in verdicts.values()):.1f} 之间。"
              "先看这两个数，再决定是换更难的设定还是加种子。")
    return verdicts, full, oracle, random_policy


# ==================== 第五部分：2×2 与相互作用 ====================


def interaction_effect(scores, confidence=0.95):
    """2×2 消融的两种"作用"、相互作用及其区间。

    scores [2, 2, n]：scores[a, b] 是部件 A 取 a（0 关 1 开）、部件 B 取 b 时各个种子的得分，即 y_ab。
    相互作用 = (y11 − y01) − (y10 − y00)。它是四个均值的加减，方差是四个 SE² 之和。
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[-1]
    y = scores.mean(axis=-1)                                             # [2, 2]
    var = scores.var(axis=-1, ddof=1) / n                                # [2, 2]：每个均值的 SE²
    se = math.sqrt(var.sum())
    denom = (var ** 2 / (n - 1)).sum()
    df = var.sum() ** 2 / denom if denom > 0 else 4 * (n - 1)            # Welch–Satterthwaite 近似
    interaction = (y[1, 1] - y[0, 1]) - (y[1, 0] - y[0, 0])
    half = t_critical(df, confidence) * se
    return {
        "y": y,
        "a_when_b_off": y[1, 0] - y[0, 0], "a_when_b_on": y[1, 1] - y[0, 1],
        "b_when_a_off": y[0, 1] - y[0, 0], "b_when_a_on": y[1, 1] - y[1, 0],
        "total": y[1, 1] - y[0, 0], "interaction": interaction, "se": se, "df": df,
        "half": half, "lo": interaction - half, "hi": interaction + half,
    }


def interaction_analysis(dirs):
    """A = 预测变化量，B = cos / sin 输入（热启动保持打开）。"""
    print("\n" + "=" * 60)
    print("第五部分：2×2 与相互作用")
    print("=" * 60)

    # 先用 CONCEPT.md 第 7.3 节的手算例子校验。每格取两个种子 m ± 0.03，这样每个均值的标准误正好是 0.03
    means = np.array([[0.30, 0.50], [0.35, 0.75]])                       # means[a, b] = y_ab
    check = interaction_effect(means[:, :, None] + np.array([-0.03, 0.03]))
    assert abs(check["a_when_b_off"] - 0.05) < 1e-9 and abs(check["a_when_b_on"] - 0.25) < 1e-9
    assert abs(check["interaction"] - 0.20) < 1e-9 and abs(check["se"] - 0.06) < 1e-9 and abs(check["total"] - 0.45) < 1e-9
    print(f"""
  校验（CONCEPT.md 第 7.3 节的手算例子，y00 = 0.30，y10 = 0.35，y01 = 0.50，y11 = 0.75，四个均值的标准误都是 0.03）：
    B 关时 A 的作用 {check['a_when_b_off']:.2f}，B 开时 A 的作用 {check['a_when_b_on']:.2f}，相互作用 {check['interaction']:.2f}，标准误 {check['se']:.2f}。与手算一致。
    """)

    name_a, name_b = FACTORS["predict_delta"], FACTORS["angle_features"]
    scores = np.array([[final_scores(dirs[f"去掉 {name_a} 和 {name_b}"]), final_scores(dirs[f"去掉 {name_a}"])],
                       [final_scores(dirs[f"去掉 {name_b}"]), final_scores(dirs["完整方法"])]])       # [2, 2, n]
    r = interaction_effect(scores)
    y = r["y"]
    print(f"""  实测。A = {name_a}，B = {name_b}；得分的形状 {scores.shape} = [A 开关, B 开关, 训练种子]：

                        B 关            B 开
        A 关         {y[0, 0]:8.1f}        {y[0, 1]:8.1f}
        A 开         {y[1, 0]:8.1f}        {y[1, 1]:8.1f}

    B 关时 A 的作用   {r['a_when_b_off']:+8.1f}          A 关时 B 的作用   {r['b_when_a_off']:+8.1f}
    B 开时 A 的作用   {r['a_when_b_on']:+8.1f}          A 开时 B 的作用   {r['b_when_a_on']:+8.1f}
    相互作用         {r['interaction']:+8.1f}，标准误 {r['se']:.1f}，自由度 {r['df']:.1f}，95% 区间 [{r['lo']:.1f}, {r['hi']:.1f}]

    量法                         说 A 贡献了     说 B 贡献了     两者相加     实际总提升 y11 − y00
    加上一个（从 A、B 都关出发）   {r['a_when_b_off']:>10.1f}{r['b_when_a_off']:>14.1f}{r['a_when_b_off'] + r['b_when_a_off']:>14.1f}{r['total']:>16.1f}
    去掉一个（从完整方法出发）     {r['a_when_b_on']:>10.1f}{r['b_when_a_on']:>14.1f}{r['a_when_b_on'] + r['b_when_a_on']:>14.1f}{r['total']:>16.1f}
    """)
    full_stats = mean_and_ci(scores[1, 1])
    print(f"  相互作用的标准误（{r['se']:.1f}）是完整方法那一格均值标准误（{full_stats['se']:.1f}）的 {r['se'] / max(full_stats['se'], 1e-12):.1f} 倍："
          "它是四个均值的加减，四份噪声叠在一起。")
    if r["lo"] > 0:
        print(f"  区间不含 0，相互作用为正：两个部件相互成就。此时「A 的贡献」可以是 {r['a_when_b_off']:.1f}，也可以是 {r['a_when_b_on']:.1f}，"
              "取决于 B 在不在场，报告里要写明用的是哪种量法。")
    elif r["hi"] < 0:
        print(f"  区间不含 0，相互作用为负：两个部件的作用有重叠，有了其中一个，另一个的作用就变小。"
              f"\n  「A 的贡献」可以是 {r['a_when_b_off']:.1f}（B 不在场），也可以是 {r['a_when_b_on']:.1f}（B 在场），报告里要写明用的是哪种量法。")
    else:
        print(f"  区间包含 0：这份数据下说不清两个部件之间有没有相互作用。预注册时写的是「方向不确定，预期接近可加」，"
              "\n  这个结果和它不矛盾，但也谈不上印证：区间的半宽有 "
              f"{r['half']:.1f}，比这小的相互作用测不出来。"
              "\n  上表里两种量法给出的「贡献」之所以对不上，这时主要是噪声，不能解读成机制。")
    return r


# ==================== 第六部分：学习曲线 ====================


def plot_learning_curves(cells, references, path, confidence=0.95, xlabel="模型训练步数", ylabel="平均回报"):
    """读各种子的 eval.jsonl，画均值曲线加区间带。区间类型和种子数自动写进标题。

    cells：{图例名: 格子目录}。references：{参照线的名字: 数值}。
    返回 {图例名: (步数 [P], 学习曲线 [n, P], 区间带 [2, P])}。
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    result, n_seeds, n_points = {}, None, None
    for (label, cell_dir), color in zip(cells.items(), ["C0", "C3", "C2", "C4"]):
        steps, curves = load_cell(cell_dir)                              # [P]、[n, P]
        n_seeds, n_points = curves.shape
        mean = curves.mean(axis=0)
        half = t_critical(n_seeds - 1, confidence) * curves.std(axis=0, ddof=1) / math.sqrt(n_seeds)
        band = np.stack([mean - half, mean + half])                      # [2, P]
        result[label] = (steps, curves, band)
        ax.plot(steps, mean, "o-", color=color, linewidth=2, label=label)
        ax.fill_between(steps, band[0], band[1], color=color, alpha=0.2)
        for curve in curves:                                             # 每个种子的原始曲线也画出来，不做平滑
            ax.plot(steps, curve, color=color, linewidth=0.6, alpha=0.4)
    for (label, value), style in zip(references.items(), ["--", ":"]):
        ax.axhline(value, color="black", linestyle=style, linewidth=1, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"学习曲线：粗线是均值，阴影是 {confidence:.0%} t 区间（n = {n_seeds} 个训练种子），细线是各个种子；未做平滑")
    ax.legend(fontsize=8)
    save_figure(fig, path)
    return result


def learning_curve_analysis(dirs, oracle, random_policy, n_eval):
    """画学习曲线；用同一份数据看"挑检查点"会虚高多少。"""
    print("\n" + "=" * 60)
    print("第六部分：带区间的学习曲线")
    print("=" * 60)
    print()
    curves = plot_learning_curves(
        {"完整方法": dirs["完整方法"], "基线（全部关掉）": dirs["基线（全部关掉）"]},
        {"真实模型 + 同一个规划器（上界）": oracle.mean(), "随机策略": random_policy.mean()},
        OUT / "04-learning-curves.png",
        xlabel="模型训练步数（离线设定：数据量固定。在线方法的横轴应当是环境交互步数）",
        ylabel=f"平均回报（每个点：每个种子 {n_eval} 个评估回合）")
    print("\n  这张图满足的规范：坐标轴有名称；阴影是什么、几个种子写在标题里；基线和上界画在同一张图上；"
          "\n  没有平滑；每个种子的原始曲线都在；图由脚本从 eval.jsonl 生成。")

    print("\n  同一份数据，两种报告方式（「挑检查点」的最大化偏差）：\n")
    print(f"    {'方法':<20}{'最后一个检查点（协议规定）':>22}{'每个种子取曲线上的最高点':>24}{'虚高':>10}")
    gaps = {}
    for label, (steps, c, _) in curves.items():
        last, best = c[:, -1].mean(), c.max(axis=1).mean()
        gaps[label] = best - last
        print(f"    {label:<20}{last:>26.1f}{best:>26.1f}{best - last:>14.1f}")
    n_points = len(next(iter(curves.values()))[0])
    if n_points < 2:
        print("    只有一个评估点，没有可挑的检查点。")
    elif max(gaps.values()) > 0:
        print(f"\n  曲线上只有 {n_points} 个点，取最高点就已经让数字好看了 {max(gaps.values()):.1f}（最大的那一行）。"
              "\n  这部分「提升」来自挑选：评估点越多、评估回合越少（噪声越大），它越大。"
              "\n  按定义它不可能为负，所以两个方法都会被抬高，但抬高的幅度不同，排名因此可能改变。协议里要固定报告最后一个检查点。")
    else:
        print("\n  这次运行里每个种子的最高点恰好都是最后一个检查点，没有虚高。评估点更多、噪声更大时它会出现。")
    return curves


# ==================== 第七部分：有效数字与结果表 ====================


def format_with_uncertainty(mean, uncertainty, sig=1):
    """按不确定性决定有效数字：先把不确定性保留 sig 位有效数字，再把均值保留到同一个小数位。

    例：format_with_uncertainty(0.6612, 0.0785) → "0.66 ± 0.08"
    """
    if not (math.isfinite(uncertainty) and uncertainty > 0):
        return f"{mean:.3g}"
    exponent = math.floor(math.log10(uncertainty))
    decimals = sig - 1 - exponent                                        # 保留到小数点后几位；负数表示取整到十位、百位
    if round(uncertainty, decimals) >= 10 ** (exponent + 1):             # 例如 0.096 进位成 0.10，要少留一位
        decimals -= 1
    digits = max(decimals, 0)
    return f"{round(mean, decimals):.{digits}f} ± {round(uncertainty, decimals):.{digits}f}"


def make_results_table(rows, full_name, n_eval):
    """生成 Markdown 结果表。

    rows：[{"name", "scores"（种子层得分 [n]，参照行为 None）, "value"（参照行的数值）, "n_params", "train_seconds"}]
    加粗规则：最好的方法加粗；和它的差的区间包含 0 的方法也加粗（分不出来的一视同仁）。
    """
    methods = [row for row in rows if row["scores"] is not None]
    best = max(methods, key=lambda row: np.mean(row["scores"]))
    full = next(row for row in methods if row["name"] == full_name)
    lines = ["| 方法 | 回报（均值 ± 95% 区间半宽） | 与完整方法之差 [95% 区间] | 训练种子 n | 参数量 | 训练用时（秒 / 次） |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        if row["scores"] is None:
            lines.append(f"| {row['name']} | {row['value']:.0f} | — | — | — | — |")
            continue
        stats = mean_and_ci(row["scores"])
        text = format_with_uncertainty(stats["mean"], stats["half"])
        tied = row is best or (lambda d: d["lo"] <= 0 <= d["hi"])(diff_ci(best["scores"], row["scores"]))
        if tied:
            text = f"**{text}**"
        if row is full:
            diff_text = "—"
        else:
            d = diff_ci(full["scores"], row["scores"])
            center, half = format_with_uncertainty(d["delta"], d["half"]).split(" ± ")
            diff_text = f"{center} [{float(center) - float(half):g}, {float(center) + float(half):g}]"
        lines.append(f"| {row['name']} | {text} | {diff_text} | {stats['n']} | {row['n_params']} | {row['train_seconds']:.1f} |")
    lines.append("")
    lines.append(f"表注：± 后面是 95% t 区间的半宽；每个训练种子的得分是 {n_eval} 个固定评估回合的平均，报告的是最后一个检查点。"
                 "加粗：均值最高的方法，以及和它的差的 95% 区间包含 0 的方法。参照行没有训练种子，只有评估回合层的波动，不给区间。")
    return "\n".join(lines)


def writing_section(matrix, dirs, verdicts, interaction, oracle, random_policy, n_eval):
    """有效数字、结果表、结论—证据对照表。"""
    print("\n" + "=" * 60)
    print("第七部分：有效数字、结果表、结论与证据")
    print("=" * 60)

    examples = [((0.6612, 0.0785, 1), "0.66 ± 0.08"), ((812.37, 45.2, 2), "812 ± 45"), ((812.37, 45.2, 1), "810 ± 50"),
                ((0.03127, 0.00094, 1), "0.0313 ± 0.0009"), ((0.5, 0.096, 1), "0.5 ± 0.1")]
    print("\n  format_with_uncertainty() 的校验（前四行是 CONCEPT.md 第 8.3 节的例子）：\n")
    print(f"    {'原始输出':<28}{'保留几位':<10}{'应该写成':<20}{'函数给出'}")
    for (mean, unc, sig), expected in examples:
        got = format_with_uncertainty(mean, unc, sig)
        assert got == expected, f"format_with_uncertainty({mean}, {unc}, {sig}) 给出 {got}，应为 {expected}"
        print(f"    {f'{mean} ± {unc}':<28}{sig:<14}{expected:<24}{got}")
    print("\n  多写的那几位不是更精确，是噪声。")

    rows = []
    for cell in matrix:
        meta = cell_meta(dirs[cell["name"]])
        rows.append({"name": cell["name"], "scores": final_scores(dirs[cell["name"]]), **meta})
    rows.append({"name": "参照：真实模型 + 同一个规划器（上界）", "scores": None, "value": oracle.mean()})
    rows.append({"name": "参照：随机策略", "scores": None, "value": random_policy.mean()})
    table = make_results_table(rows, "完整方法", n_eval)
    path = OUT / "04-results-table.md"
    path.write_text(table + "\n", encoding="utf-8")
    print(f"\n  结果表（由 make_results_table() 从运行目录生成，已存到 outputs/phase-12/{path.name}）：\n")
    print("\n".join("    " + line for line in table.split("\n")))

    # 结论—证据对照表：措辞由证据的强弱决定
    full = final_scores(dirs["完整方法"])
    baseline = final_scores(dirs["基线（全部关掉）"])
    main = diff_ci(baseline, full)
    claims = []
    if main["lo"] > 0:
        claims.append(("完整方法的回报高于基线", f"差 {main['delta']:.0f}，95% 区间 [{main['lo']:.0f}, {main['hi']:.0f}] 不含 0", "强：写「表明」"))
    elif main["hi"] < 0:
        claims.append(("完整方法的回报低于基线（和预期相反）", f"差 {main['delta']:.0f}，区间 [{main['lo']:.0f}, {main['hi']:.0f}] 不含 0", "强：先查实现，再写进报告"))
    else:
        claims.append(("完整方法是否优于基线", f"差 {main['delta']:.0f}，区间 [{main['lo']:.0f}, {main['hi']:.0f}] 含 0", "分不出来：不写成结论"))
    for factor, v in verdicts.items():
        d = v["diff"]
        evidence = f"去掉后变化 {d['delta']:.0f}，区间 [{d['lo']:.0f}, {d['hi']:.0f}]"
        if v["observed"] == "下降":
            claims.append((f"{FACTORS[factor]}是必要的", evidence + " 不含 0", "强：写「表明」"))
        elif v["observed"] == "上升":
            claims.append((f"去掉{FACTORS[factor]}反而更好", evidence + " 不含 0", "强，但要先排除 bug 和连带变化"))
        else:
            claims.append((f"{FACTORS[factor]}有没有用", evidence + " 含 0", "分不出来：写「未观察到作用」，不写「没有作用」"))
    if interaction["lo"] > 0 or interaction["hi"] < 0:
        claims.append(("两个模型部件之间有相互作用", f"相互作用 {interaction['interaction']:.0f}，区间 [{interaction['lo']:.0f}, {interaction['hi']:.0f}] 不含 0", "强：贡献的拆分要写明量法"))
    else:
        claims.append(("两个模型部件之间有没有相互作用", f"相互作用 {interaction['interaction']:.0f}，区间 [{interaction['lo']:.0f}, {interaction['hi']:.0f}] 含 0", "分不出来"))
    claims.append(("这些结论能推广到别的任务", "无：只在 Pendulum 一个任务上做过", "没有证据：不写成结论，放进局限"))
    print("\n  结论—证据对照表（每一行的措辞由区间决定，不由期望决定）：\n")
    print(f"    {'结论':<28}{'证据':<52}{'强度与措辞'}")
    for claim, evidence, strength in claims:
        print(f"    {claim:<28}{evidence:<52}{strength}")
    print(f"""
  局限也是报告的一部分。这次实验的局限至少有：单任务（Pendulum）；{len(full)} 个训练种子；每个种子 {n_eval} 个评估回合；
  离线设定、数据量固定为 {matrix[0]['config'].n_data_episodes} 个随机回合；三个开关之外的超参数没有调过，也没有给基线单独调过。
    """)


# ==================== 练习与思考 ====================


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：加一个因素
    给 AgentConfig 加一个开关 clip_velocity（模型输出的角速度是否按物理上限 ±8 裁剪），
    写下预期，再把它加进 build_ablation_matrix() 的 factors 里跑一遍。

练习 1 参考答案：
    DynamicsModel.forward() 里把 theta_dot.clamp(...) 改成按开关决定是否裁剪。
    写预期时要想清楚机制：不裁剪时，模型在高速区域的外推可以给出物理上不可能的角速度，
    规划器有可能利用它（Phase 1 第 8.5 节的模型被钻空子）。预期方向是下降，幅度不确定。
    写不出机制的消融，多半不值得跑；写出来之后结果相反，才知道自己的理解哪里错了。

练习 2：敏感性扫描
    把 plan_horizon 取 5、10、20、40 各跑一遍（每个值 5 个种子），画"回报 对 规划视野"的曲线，带区间。
    它和消融的区别是什么？

练习 2 参考答案：
    用 replace(full, plan_horizon=h) 生成配置，其余流程不变。
    这是敏感性分析：回答"方法对这个超参数有多敏感"，不回答"某个部件有没有用"。
    注意连带变化：视野翻倍，每步的计算量翻倍，模型误差累积的步数也翻倍。
    预期是先升后降（太短看不到甩上去的收益，太长超过模型的可信视野），峰值的位置和 03 节量出的 k* 有关。

练习 3：把种子数减到 2
    只用种子 0 和 1 重新做第四部分的分析。三组消融的"对照"一栏怎么变？

练习 3 参考答案：
    2 个种子时自由度只有 1 左右，t 分布的临界值是 12.7（5 个种子时是 2.78），区间宽得几乎一定包含 0，
    对照栏会全部变成"分不出来"。差值本身可能没怎么变，变的是你对它的把握。
    这就是"消融只跑 1、2 个种子看趋势"的问题：每一行的正负号都可能是随机的。

思考题 1："去掉一个"和"加上一个"，报告里应该用哪种？

思考题 1 参考答案：
    通常用"去掉一个"：读者关心的是"在你的完整系统里，这个部件能不能省"。
    "加上一个"回答的是"这个部件单独有没有用"，对想把它搬到别的系统里的人更有参考价值。
    有相互作用时两者给出的数字不同，都没有错。报告里写明用的是哪一种；条件允许时把 2×2 的四个数都列出来。

思考题 2：去掉某个部件后分数没变。应该怎么处理这个部件？

思考题 2 参考答案：
    先看区间够不够窄：区间很宽时结论是"分不出来"，需要更多种子；区间很窄时才能说"作用即使有也很小"。
    后一种情况下，把它从最终方法里去掉，并把这一行留在消融表里。
    白加的部件是之后每个使用者的负担，"它可以省"本身是一个有用的结论。

思考题 3：预注册之后，看到结果又想加一组消融，可以吗？

思考题 3 参考答案：
    可以，在报告里标明它是事后加的（探索性的）。预注册约束的不是"能做什么实验"，
    而是"哪些结论有资格被当作事先的预测得到了验证"。事后看出来的规律只能算新的假设，
    要用新的种子再验证一次，不能用发现它的那批数据来确认它。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(0)
    introduction()

    quick = is_quick()
    full = AgentConfig()
    if quick:                                                            # 冒烟模式只是缩小预算的一组覆盖
        full = replace(full, n_data_episodes=1, train_steps=60, plan_horizon=8, n_candidates=32, n_elites=4,
                       cem_iters=2, episode_len=30, n_eval_episodes=1)
    seeds = [0, 1] if quick else [0, 1, 2, 3, 4]
    eval_seeds = [2000 + j for j in range(full.n_eval_episodes)]         # 固定的评估种子，所有格子、所有训练种子共用
    n_eval_points = 2 if quick else 4

    print("\n" + "=" * 60)
    print("第二部分：实验矩阵")
    print("=" * 60)
    matrix = build_ablation_matrix(full, list(FACTORS), seeds, pairs=[("predict_delta", "angle_features")])
    print(f"\n  build_ablation_matrix() 生成了 {len(matrix)} 个格子（每个格子的配置和完整方法相比，只在「关掉的开关」上不同）：\n")
    for cell in matrix:
        print(f"    {cell['name']:<28} 关掉：{', '.join(cell['off']) or '—':<46} 训练种子 {cell['seeds']}")
    k, n = len(FACTORS), len(seeds)
    print(f"\n  成本的账：去掉一个要 (k + 1) × n = {(k + 1) * n} 次训练；全因子要 2^k × n = {2 ** k * n} 次。"
          f"\n  这里做的是：去掉一个（{(k + 1) * n} 次）+ 一组 2×2 里缺的那一格（{n} 次）+ 基线（{n} 次）= {len(matrix) * n} 次。")

    dirs = run_matrix(matrix, curve_cells={"完整方法", "基线（全部关掉）"}, n_eval_points=n_eval_points, eval_seeds=eval_seeds)
    verdicts, _, oracle, random_policy = leave_one_out_analysis(matrix, dirs, eval_seeds)
    interaction = interaction_analysis(dirs)
    learning_curve_analysis(dirs, oracle, random_policy, len(eval_seeds))
    writing_section(matrix, dirs, verdicts, interaction, oracle, random_policy, len(eval_seeds))
    exercises()

    print("\n" + "=" * 60)
    print("Phase 12 的讲义全部完成！接下来：写一页计划（CONCEPT.md 第 2 节），开始毕业项目。")
    print("=" * 60)


if __name__ == "__main__":
    main()
