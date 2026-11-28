"""运行 Phase 5 连续控制排行榜，保存 results.json 与学习曲线图。"""

import json
import os
import sys

_PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_PROJ_DIR, "..", "..", "src"))
for p in (_PROJ_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from benchmark_algos import run_mbpo, run_pets, run_ppo, run_random, run_sac
from benchmark_protocol import BENCHMARK_TASKS, get_task_protocol, steps_to_threshold
from utils import is_quick, output_dir, save_figure, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-05-project")

ALGO_REGISTRY = {
    "Random": run_random,
    "PETS": run_pets,
    "PPO": run_ppo,
    "SAC": run_sac,
    "MBPO": run_mbpo,
}


def run_all():
    quick = is_quick()
    tasks = ["Pendulum-v1"] if quick else list(BENCHMARK_TASKS.keys())
    summary = {"quick": quick, "protocol": {}, "tasks": {}}

    for task_name in tasks:
        cfg = get_task_protocol(task_name, quick=quick)
        summary["protocol"][task_name] = cfg
        summary["tasks"][task_name] = {}
        print("\n" + "=" * 68)
        print(f"任务：{task_name}（预算 {cfg['total_steps']} 步，种子 {cfg['seeds']}，阈值 {cfg['threshold']}）")
        print("=" * 68)

        for algo_name, runner in ALGO_REGISTRY.items():
            seed_curves = []
            wall_times = []
            utds = []
            steps_ref = None
            for s in cfg["seeds"]:
                st, ret, dt, utd = runner(task_name, cfg, seed=s)
                steps_ref = st
                seed_curves.append(ret)
                wall_times.append(dt)
                utds.append(utd)

            arr = np.asarray(seed_curves, dtype=np.float64)
            mean_c = arr.mean(axis=0)
            std_c = arr.std(axis=0)
            final_mean = float(mean_c[-1])
            final_std = float(std_c[-1])
            hit = steps_to_threshold(steps_ref, mean_c, cfg["threshold"])
            avg_time = float(np.mean(wall_times))
            avg_utd = float(np.mean(utds))

            summary["tasks"][task_name][algo_name] = {
                "final_mean": final_mean,
                "final_std": final_std,
                "steps_to_threshold": hit,
                "wall_time_sec": avg_time,
                "utd": avg_utd,
                "steps": [int(x) for x in steps_ref],
                "mean_curve": [float(x) for x in mean_c],
                "std_curve": [float(x) for x in std_c],
            }
            hit_str = f"{hit} 步" if hit is not None else "未达标"
            print(f"  {algo_name:<8} ｜ 最终 {final_mean:8.1f} ± {final_std:5.1f} ｜ "
                  f"达标 {hit_str:<9} ｜ 单子耗时 {avg_time:5.1f}s ｜ UTD {avg_utd:.2f}")

    res_path = OUT / "results.json"
    res_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 排行榜结果已保存：{res_path}")

    # 画多任务学习曲线图
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.2))
    if len(tasks) == 1:
        axes = [axes]
    for ax, task_name in zip(axes, tasks):
        for algo_name, info in summary["tasks"][task_name].items():
            st = np.array(info["steps"])
            mu = np.array(info["mean_curve"])
            sd = np.array(info["std_curve"])
            ax.plot(st, mu, "o-", linewidth=2, label=f"{algo_name} ({info['final_mean']:.0f})")
            ax.fill_between(st, mu - sd, mu + sd, alpha=0.14)
        ax.set_title(task_name)
        ax.set_xlabel("真实环境步数")
        ax.set_ylabel("无噪评估回报")
        ax.legend(fontsize=8)
    save_figure(fig, OUT / "benchmark-curves.png")


if __name__ == "__main__":
    run_all()
