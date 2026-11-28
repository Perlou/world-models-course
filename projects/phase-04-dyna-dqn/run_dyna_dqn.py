"""Phase 4 实战项目：CartPole-v1 上对比 DQN 与不同规划步数 n_p 的 Dyna-DQN。"""

import argparse
import json
import os
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PROJECT_DIR, "..", ".."))
for p in (os.path.join(REPO_ROOT, "src"), PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import gymnasium as gym
import numpy as np
import torch

from dqn_agent import DynaDQNAgent
from dyna_env_model import CartPoleWorldModel, N_ACTIONS, ReplayBuffer, world_model_step_loss
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-04-project")
DEVICE = torch.device("cpu")  # 4 维小 MLP 在 CPU 上单步延迟最低


@torch.no_grad()
def evaluate_greedy(agent, eval_seeds):
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(0)
    rets = []
    for s in eval_seeds:
        obs, _ = env.reset(seed=int(s))
        tot = 0.0
        while True:
            a = agent.select_action(obs, epsilon=0.0, rng=rng)
            obs, r, term, trunc, _ = env.step(a)
            tot += float(r)
            if term or trunc:
                break
        rets.append(tot)
    env.close()
    return float(np.mean(rets))


@torch.no_grad()
def evaluate_world_model(wm, replay, rng):
    s, a, _, s_next, term = replay.sample(min(512, len(replay)), rng, device=DEVICE)
    s_pred, _, d_pred = wm.step(s, a)
    state_rmse = (s_pred - s_next).pow(2).mean().sqrt().item()
    done_acc = (d_pred == term).float().mean().item()
    pos_mask = term > 0.5
    done_recall = float(d_pred[pos_mask].mean().item()) if pos_mask.any() else float("nan")
    return dict(state_rmse=state_rmse, done_acc=done_acc, done_recall=done_recall)


def run_single_experiment(planning_steps, seed, total_env_steps, eval_every, warmup=200, batch_size=64):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=seed)

    replay = ReplayBuffer(capacity=20_000)
    agent = DynaDQNAgent(device=DEVICE)
    wm = CartPoleWorldModel().to(DEVICE)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=2e-3)
    eval_seeds = [2000 + i for i in range(10)]

    eval_curve = []
    wm_curve = []
    ep_idx = 0

    for step in range(1, total_env_steps + 1):
        eps = max(0.05, 1.0 - step / (0.35 * total_env_steps))
        action = agent.select_action(obs, epsilon=eps, rng=rng)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # 关键细节：时间上限 truncated=True 时绝不能当作 terminated=True！
        replay.add(obs, action, reward, next_obs, terminated)

        if terminated or truncated:
            ep_idx += 1
            obs, _ = env.reset(seed=seed * 10_000 + ep_idx)
        else:
            obs = next_obs

        if len(replay) >= warmup:
            real_batch = replay.sample(batch_size, rng, device=DEVICE)
            agent.update_q_from_batch(real_batch)

            if planning_steps > 0:
                # 更新世界模型（每步用真实数据更新 2 次，确保模型始终走在策略前面）
                for _ in range(2):
                    wm_b = replay.sample(batch_size, rng, device=DEVICE)
                    wm_loss, _ = world_model_step_loss(wm, *wm_b)
                    wm_opt.zero_grad()
                    wm_loss.backward()
                    wm_opt.step()

                # Dyna 后台规划：从真实访问过的状态出发，由世界模型单步想象生成经验更新 Q 网络
                if step >= warmup + 100:
                    for _ in range(planning_steps):
                        s_start = replay.sample_states(batch_size, rng, device=DEVICE)
                        # 一半按当前 Q 网络贪心选，一半随机探索动作分支
                        with torch.no_grad():
                            greedy_a = agent.q_online(s_start).argmax(dim=1)
                            rand_a = torch.from_numpy(rng.integers(0, N_ACTIONS, size=batch_size)).to(DEVICE)
                            mask = torch.from_numpy(rng.random(batch_size) < 0.5).to(DEVICE)
                            sim_a = torch.where(mask, rand_a, greedy_a)
                            sim_s_next, sim_r, sim_term = wm.step(s_start, sim_a)
                        agent.update_q_from_batch((s_start, sim_a, sim_r, sim_s_next, sim_term))

        if step % eval_every == 0:
            ev = evaluate_greedy(agent, eval_seeds)
            eval_curve.append((step, ev))
            if planning_steps > 0 and len(replay) >= warmup:
                wm_curve.append((step, evaluate_world_model(wm, replay, rng)))

    env.close()
    return eval_curve, wm_curve


def parse_args():
    p = argparse.ArgumentParser(description="CartPole-v1 上的 DQN vs Dyna-DQN 样本效率对比")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--tag", default="")
    return p.parse_args()


def main():
    args = parse_args()
    quick = is_quick()
    total_steps = args.steps or (900 if quick else 6000)
    n_seeds = args.seeds or (1 if quick else 3)
    eval_every = 300 if quick else 400
    ratios = [0, 2, 5]
    tag = f"-{args.tag}" if args.tag else ""

    print("=" * 60)
    print("Phase 4 实战项目：Dyna-DQN（CartPole-v1 样本效率对比）")
    print("=" * 60)
    print(f"  配置：总交互步数 = {total_steps}，随机种子数 = {n_seeds}，对比规划比 n_p ∈ {ratios}\n")

    t0 = time.time()
    all_results = {}
    wm_logs = {}
    for np_steps in ratios:
        t_r = time.time()
        curves = []
        last_wm = []
        for s in range(n_seeds):
            ec, wc = run_single_experiment(np_steps, seed=s, total_env_steps=total_steps, eval_every=eval_every)
            curves.append([v for _, v in ec])
            xs = [k for k, _ in ec]
            if wc:
                last_wm = wc
        arr = np.array(curves)  # [S, K]
        mean_c, std_c = arr.mean(axis=0), arr.std(axis=0)
        # 达到平均回报 >= 450 所需的真实环境步数
        hit_idx = np.where(mean_c >= 450.0)[0]
        steps_to_450 = int(xs[hit_idx[0]]) if len(hit_idx) > 0 else None
        all_results[str(np_steps)] = dict(
            steps=xs, mean=mean_c.tolist(), std=std_c.tolist(),
            final_mean=float(mean_c[-1]), final_std=float(std_c[-1]),
            auc_mean=float(mean_c.mean()), steps_to_450=steps_to_450,
            elapsed=time.time() - t_r,
        )
        if last_wm:
            wm_logs[str(np_steps)] = last_wm
        label = "纯 Double DQN (n_p=0)" if np_steps == 0 else f"Dyna-DQN (n_p={np_steps})"
        s450 = f"{steps_to_450} 步" if steps_to_450 else "未达到"
        print(f"  {label:<24} 达 450 分步数：{s450:<8} ｜ 全程平均分：{mean_c.mean():6.1f} ｜ "
              f"终点：{mean_c[-1]:5.1f} ± {std_c[-1]:4.1f}  [{all_results[str(np_steps)]['elapsed']:.1f}s]")

    # 画学习曲线 + 模型精度曲线
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    for np_steps, color in zip(ratios, ("C0", "C1", "C2")):
        r = all_results[str(np_steps)]
        xs = np.array(r["steps"])
        m, s = np.array(r["mean"]), np.array(r["std"])
        lbl = "DQN (n_p = 0)" if np_steps == 0 else f"Dyna-DQN (n_p = {np_steps})"
        axes[0].plot(xs, m, "o-", markersize=4, linewidth=2, color=color, label=lbl)
        axes[0].fill_between(xs, m - s, m + s, color=color, alpha=0.18)
    axes[0].axhline(475.0, color="gray", linestyle=":", label="解决线 (475)")
    axes[0].set_xlabel("真实环境交互步数")
    axes[0].set_ylabel("贪心评估平均回报（最高 500）")
    axes[0].set_title(f"样本效率对比（{n_seeds} 种子均值 ± 标准差）")
    axes[0].legend(fontsize=8.5)

    if wm_logs:
        key = max(wm_logs.keys(), key=int)
        w_xs = [k for k, _ in wm_logs[key]]
        rmses = [d["state_rmse"] for _, d in wm_logs[key]]
        recalls = [d["done_recall"] * 100.0 for _, d in wm_logs[key]]
        ax2 = axes[1]
        ax2.plot(w_xs, rmses, "o-", color="C3", linewidth=2, label="单步状态预测 RMSE")
        ax2.set_xlabel("真实环境交互步数")
        ax2.set_ylabel("单步状态 RMSE", color="C3")
        ax2_r = ax2.twinx()
        ax2_r.plot(w_xs, recalls, "s--", color="C4", linewidth=1.8, label="倒杆终止召回率 (%)")
        ax2_r.set_ylabel("倒杆终止帧召回率 (%)", color="C4")
        ax2_r.grid(False)
        ax2.set_title(f"学出来的动力学与终止模型精度 (n_p={key})")

    save_figure(fig, OUT / f"learning-curve{tag}.png")

    payload = dict(total_seconds=time.time() - t0, total_steps=total_steps, n_seeds=n_seeds, results=all_results)
    with open(OUT / f"results{tag}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n📄 结果已保存：outputs/phase-04-project/results{tag}.json（总用时 {payload['total_seconds']:.1f}s）")


if __name__ == "__main__":
    main()
