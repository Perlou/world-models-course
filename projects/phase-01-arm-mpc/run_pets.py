"""
机械臂模型预测控制：在 Reacher-v5 上跑 PETS
==========================================

把 Phase 1 讲义里的方法原样搬到二连杆机械臂上：

    状态从 2 维变成 4 维（两个关节的角度和角速度）
    动作从 1 维变成 2 维（两个关节的力矩）
    每个回合的目标点不同

其余不变：概率集成 + 轨迹采样 + CEM-MPC + 迭代采集数据。

用法（在仓库根目录）：
    uv run python projects/phase-01-arm-mpc/run_pets.py              # 完整运行，约 4 分钟（M1 Max，MPS）
    WM_QUICK=1 uv run python projects/phase-01-arm-mpc/run_pets.py   # 冒烟运行，约 20 秒
    uv run python projects/phase-01-arm-mpc/run_pets.py --init-episodes 1 --collect-episodes 1 --tag init1   # 对比实验：初始只给 1 个随机回合

产物（outputs/phase-01-project/；给了 --tag init1 时文件名变成 learning-curve-init1.png 等）：
    learning-curve.png        学习曲线
    open-loop-error.png       模型的开环误差
    real-vs-imagined.mp4      真实轨迹与模型想象轨迹的对比视频
    results.json              全部数字
"""

import argparse
import json
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import reacher
from model import ProbabilisticEnsemble, train_ensemble
from planner import cem_plan, score_sequences, shift_plan
from utils import get_device, is_quick, output_dir, save_figure, save_video, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-01-project")
DEVICE = get_device()
SEED = 0
HORIZON = 15
N_MODELS = 5
PARTICLES_PER_MEMBER = 2  # 每条候选序列 5 × 2 = 10 个粒子


def new_model():
    return ProbabilisticEnsemble(reacher.features, feature_dim=8, state_dim=reacher.STATE_DIM, n_models=N_MODELS)


def run_mpc_episode(model, seed):
    """用 CEM-MPC 在真实环境里跑一个回合。

    返回一个字典：states [T+1, 4]、actions [T, 2]、goal [2]、回报、最后一步指尖到目标的距离。
    """
    env = reacher.make_env()
    env.reset(seed=seed)
    goal_np = reacher.get_goal(env)
    goal = torch.as_tensor(goal_np, device=DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    reward_fn = lambda states, actions: reacher.reward_fn(states, actions, goal)
    mean = torch.zeros(HORIZON, reacher.ACTION_DIM, device=DEVICE)
    states, actions, total = [reacher.get_state(env)], [], 0.0
    for _ in range(reacher.EPISODE_LENGTH):
        state = torch.as_tensor(states[-1], device=DEVICE)
        score_fn = lambda acts: score_sequences(model, reward_fn, state, acts, PARTICLES_PER_MEMBER, generator)
        mean = cem_plan(score_fn, mean, generator)
        action = mean[0].cpu().numpy().astype(np.float32)
        _, reward, _, _, _ = env.step(action)
        total += reward
        actions.append(action)
        states.append(reacher.get_state(env))
        mean = shift_plan(mean)
    env.close()
    states = np.array(states)
    final_dist = float(np.linalg.norm(reacher.fingertip(states[-1]) - goal_np))
    return dict(states=states, actions=np.array(actions), goal=goal_np, ret=total, final_dist=final_dist)


def to_transitions(episodes):
    """把若干个回合拼成转移 (s, a, s')，CPU 张量。"""
    s = np.concatenate([ep["states"][:-1] for ep in episodes])
    a = np.concatenate([ep["actions"] for ep in episodes])
    s_next = np.concatenate([ep["states"][1:] for ep in episodes])
    return [torch.as_tensor(x, dtype=torch.float32) for x in (s, a, s_next)]


def random_episode(seed):
    s, a, s_next, goal, total = reacher.run_random_episode(seed)
    states = np.concatenate([s, s_next[-1:]])
    final_dist = float(np.linalg.norm(reacher.fingertip(states[-1]) - goal))
    return dict(states=states, actions=a, goal=goal, ret=total, final_dist=final_dist)


@torch.no_grad()
def imagine(model, start_state, actions):
    """从 start_state 出发，按给定的动作序列在模型里开环展开（每个成员各展开一条，取均值预测）。

    返回 [B, T+1, 4]：每个成员想象出来的状态序列。
    """
    state = torch.as_tensor(start_state, device=DEVICE).expand(N_MODELS, 1, -1)
    actions = torch.as_tensor(actions, device=DEVICE)
    trajectory = [state]
    for t in range(actions.shape[0]):
        delta, _ = model.predict(state, actions[t].expand(N_MODELS, 1, -1))
        state = state + delta
        trajectory.append(state)
    return torch.cat(trajectory, dim=1).cpu().numpy()


def open_loop_error(model, episodes):
    """模型的开环误差：从回合起点出发，按真实执行的动作序列展开，指尖位置差多少（米）。

    返回 (各成员平均的误差 [T+1], 成员之间指尖位置的分歧 [T+1])，对回合取平均。
    """
    errors, spreads = [], []
    for ep in episodes:
        imagined = imagine(model, ep["states"][0], ep["actions"])            # [B, T+1, 4]
        tips = reacher.fingertip(imagined)                                    # [B, T+1, 2]
        true_tips = reacher.fingertip(ep["states"])                           # [T+1, 2]
        errors.append(np.linalg.norm(tips - true_tips, axis=-1).mean(axis=0))
        spreads.append(np.linalg.norm(tips - tips.mean(axis=0), axis=-1).mean(axis=0))
    return np.mean(errors, axis=0), np.mean(spreads, axis=0)


def make_video(model, episode, path):
    """左：真实执行的机械臂。右：模型从第 0 步开始开环想象的机械臂（5 个成员各画一条）。"""
    imagined = imagine(model, episode["states"][0], episode["actions"])  # [B, T+1, 4]
    goal = episode["goal"]
    frames = []
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), dpi=100)
    for t in range(len(episode["states"])):
        for ax in axes:
            ax.clear()
            ax.set_xlim(-0.25, 0.25)
            ax.set_ylim(-0.25, 0.25)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.plot(*goal, "r*", markersize=14)
        s = episode["states"][t]
        e, f = reacher.elbow(s), reacher.fingertip(s)
        axes[0].plot([0, e[0], f[0]], [0, e[1], f[1]], "o-", color="black", linewidth=4, markersize=6)
        axes[0].plot(*reacher.fingertip(episode["states"][:t + 1]).T, color="C0", linewidth=1)
        axes[0].set_title(f"真实环境（第 {t} 步）")
        for b in range(N_MODELS):
            sb = imagined[b, t]
            e, f = reacher.elbow(sb), reacher.fingertip(sb)
            axes[1].plot([0, e[0], f[0]], [0, e[1], f[1]], "o-", color=f"C{b}", linewidth=2, markersize=4, alpha=0.6)
        axes[1].plot(*reacher.fingertip(episode["states"][:t + 1]).T, color="gray", linewidth=1, linestyle="--")
        axes[1].set_title("模型的开环想象（5 个成员，同一串动作）")
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    save_video(frames, path, fps=10)


def parse_args():
    parser = argparse.ArgumentParser(description="Reacher-v5 上的 PETS")
    parser.add_argument("--init-episodes", type=int, default=None, help="初始的随机回合数（默认 5，冒烟模式 2）")
    parser.add_argument("--collect-episodes", type=int, default=None, help="每一轮用控制器新采的回合数（默认 2，冒烟模式 1）")
    parser.add_argument("--tag", default="", help="产物文件名的后缀，做对比实验时用，避免覆盖默认结果")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(SEED)
    if is_quick():
        n_init, n_iterations, n_collect, eval_seeds, train_steps = 2, 2, 1, [1000], 100
    else:
        n_init, n_iterations, n_collect, eval_seeds, train_steps = 5, 8, 2, list(range(1000, 1005)), 2000
    n_init = args.init_episodes or n_init
    n_collect = args.collect_episodes or n_collect
    tag = f"-{args.tag}" if args.tag else ""

    print("=" * 60)
    print("机械臂模型预测控制：Reacher-v5 上的 PETS")
    print("=" * 60)
    print(f"""
设置：
    初始数据     {n_init} 个随机回合（{n_init * reacher.EPISODE_LENGTH} 条转移）
    每一轮       训练集成 → 用 CEM-MPC 采 {n_collect} 个新回合加进数据集 → 在 {len(eval_seeds)} 个固定的评估回合上打分
    模型         {N_MODELS} 个成员的概率集成，三层 200 单元
    规划         CEM 每轮 200 条候选 × 5 轮，视野 {HORIZON}，每条序列 {N_MODELS * PARTICLES_PER_MEMBER} 个粒子（TS∞）
    设备         {DEVICE}

评估回合的目标点固定不变，而且不加进训练数据，所以各轮之间可以直接比较。
    """)

    # 参照：随机策略在评估回合上的表现
    random_eval = [random_episode(seed) for seed in eval_seeds]
    random_return = np.mean([ep["ret"] for ep in random_eval])
    random_dist = np.mean([ep["final_dist"] for ep in random_eval])
    print(f"  参照：随机策略   平均回报 {random_return:7.2f}   最后一步指尖到目标的距离 {random_dist * 100:5.1f} 厘米")

    dataset = [random_episode(seed) for seed in range(n_init)]
    history = []
    start_all = time.time()
    for it in range(n_iterations):
        start = time.time()
        torch.manual_seed(SEED + it)
        model = train_ensemble(new_model(), to_transitions(dataset), train_steps, DEVICE, seed=SEED + it)
        n_data = sum(len(ep["actions"]) for ep in dataset)
        evals = [run_mpc_episode(model, seed) for seed in eval_seeds]
        err, spread = open_loop_error(model, evals)
        record = dict(
            n_data=n_data,
            ret=float(np.mean([ep["ret"] for ep in evals])),
            ret_std=float(np.std([ep["ret"] for ep in evals])),
            final_dist=float(np.mean([ep["final_dist"] for ep in evals])),
            open_loop_error=err.tolist(), spread=spread.tolist(),
        )
        history.append(record)
        k = min(HORIZON, len(err) - 1)
        print(f"  第 {it + 1:>2} 轮：{n_data:>4} 条转移   评估回报 {record['ret']:7.2f} ± {record['ret_std']:.2f}   "
              f"终点距离 {record['final_dist'] * 100:5.1f} 厘米   开环 {k} 步的指尖误差 {err[k] * 100:5.2f} 厘米   [{time.time() - start:.0f} 秒]")
        # 用当前的控制器采新数据（新的起点和目标），加进数据集
        dataset += [run_mpc_episode(model, seed=100 + it * n_collect + j) for j in range(n_collect)]
    total_time = time.time() - start_all

    # ---------- 图 ----------
    x = np.arange(1, n_iterations + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].errorbar(x, [h["ret"] for h in history], yerr=[h["ret_std"] for h in history], marker="o", capsize=4, linewidth=2, label="PETS")
    axes[0].axhline(random_return, color="tomato", linestyle=":", label=f"随机策略（{random_return:.1f}）")
    axes[0].set_xlabel("轮次")
    axes[0].set_ylabel(f"评估回报（{len(eval_seeds)} 个回合，误差棒是标准差）")
    axes[0].set_title("学习曲线")
    axes[0].legend()
    axes[1].plot(x, [h["final_dist"] * 100 for h in history], "o-", linewidth=2, color="C2")
    axes[1].axhline(random_dist * 100, color="tomato", linestyle=":", label="随机策略")
    axes[1].set_xlabel("轮次")
    axes[1].set_ylabel("回合结束时指尖到目标的距离（厘米）")
    axes[1].set_title("末端到达精度")
    axes[1].legend()
    save_figure(fig, OUT / f"learning-curve{tag}.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = np.arange(len(history[0]["open_loop_error"]))
    for it in sorted({0, n_iterations // 2, n_iterations - 1}):
        ax.plot(steps[1:], np.array(history[it]["open_loop_error"])[1:] * 100, linewidth=2, label=f"第 {it + 1} 轮：真实误差")
        ax.plot(steps[1:], np.array(history[it]["spread"])[1:] * 100, linewidth=1, linestyle="--", color=ax.lines[-1].get_color(),
                label=f"第 {it + 1} 轮：成员之间的分歧")
    ax.set_yscale("log")
    ax.set_xlabel("开环展开的步数")
    ax.set_ylabel("指尖位置（厘米，对数坐标）")
    ax.set_title("模型的开环误差，以及集成对它的估计")
    ax.legend(fontsize=8)
    save_figure(fig, OUT / f"open-loop-error{tag}.png")

    # ---------- 视频：用最后一轮的模型，在一个没见过的回合上 ----------
    episode = run_mpc_episode(model, seed=2000)
    make_video(model, episode, OUT / f"real-vs-imagined{tag}.{'gif' if is_quick() else 'mp4'}")

    results = dict(random_return=float(random_return), random_final_dist=float(random_dist), history=history,
                   total_time_sec=total_time, horizon=HORIZON, n_models=N_MODELS,
                   particles=N_MODELS * PARTICLES_PER_MEMBER, device=str(DEVICE))
    with open(OUT / f"results{tag}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 数字已保存：outputs/phase-01-project/results{tag}.json")

    first, last = history[0], history[-1]
    k = min(HORIZON, len(first["open_loop_error"]) - 1)
    best = max(range(n_iterations), key=lambda i: history[i]["ret"])
    print(f"""
结果（共用时 {total_time / 60:.1f} 分钟）：

    {'':<14}{'评估回报':>10}{'终点距离（厘米）':>16}{f'开环 {k} 步指尖误差（厘米）':>24}
    {'随机策略':<12}{random_return:>12.2f}{random_dist * 100:>16.1f}{'—':>22}
    {'第 1 轮':<13}{first['ret']:>12.2f}{first['final_dist'] * 100:>16.1f}{first['open_loop_error'][k] * 100:>22.2f}
    {f'第 {n_iterations} 轮':<13}{last['ret']:>12.2f}{last['final_dist'] * 100:>16.1f}{last['open_loop_error'][k] * 100:>22.2f}

    真实交互步数：初始 {first['n_data']} 步，到最后一轮训练时共 {last['n_data']} 步。
    回报最高的是第 {best + 1} 轮（{history[best]['ret']:.2f}）。
    """)


if __name__ == "__main__":
    main()
