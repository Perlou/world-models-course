"""运行 Phase 6 实战项目 Mini-Dreamer，接入 Phase 5 统一连续控制排行榜。"""

import json
import os
import sys

_PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_PROJ_DIR, "..", "..", "src"))
_P5_BENCH = os.path.abspath(os.path.join(_PROJ_DIR, "..", "phase-05-control-benchmark"))
for p in (_PROJ_DIR, _SRC_DIR, _P5_BENCH):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from benchmark_protocol import BENCHMARK_TASKS, get_task_protocol, steps_to_threshold
from mini_dreamer import run_dreamer_on_benchmark
from utils import is_quick, output_dir, save_figure, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-06-project")
P5_OUT = output_dir("phase-05-project")


def main():
    quick = is_quick()
    tasks = ["Pendulum-v1"] if quick else list(BENCHMARK_TASKS.keys())
    summary = {"quick": quick, "tasks": {}}

    for task_name in tasks:
        cfg = get_task_protocol(task_name, quick=quick)
        print(f"\n运行 Mini-Dreamer V3：{task_name}（预算 {cfg['total_steps']} 步，种子 {cfg['seeds']}）")
        curves, wall_ts, dec_ms = [], [], []
        steps_ref = None
        for s in cfg["seeds"]:
            st, rt, dt, ms = run_dreamer_on_benchmark(task_name, cfg, seed=s)
            steps_ref = st
            curves.append(rt)
            wall_ts.append(dt)
            dec_ms.append(ms)
        arr = np.asarray(curves, dtype=np.float64)
        mean_c, std_c = arr.mean(axis=0), arr.std(axis=0)
        hit = steps_to_threshold(steps_ref, mean_c, cfg["threshold"])
        summary["tasks"][task_name] = {
            "final_mean": float(mean_c[-1]),
            "final_std": float(std_c[-1]),
            "steps_to_threshold": hit,
            "wall_time_sec": float(np.mean(wall_ts)),
            "decision_ms": float(np.mean(dec_ms)),
            "steps": [int(x) for x in steps_ref],
            "mean_curve": [float(x) for x in mean_c],
            "std_curve": [float(x) for x in std_c],
        }
        print(f"  Dreamer V3 ｜ 最终 {mean_c[-1]:7.1f} ± {std_c[-1]:5.1f} ｜ "
              f"单步决策 {np.mean(dec_ms):.2f} ms ｜ 单子耗时 {np.mean(wall_ts):.1f}s")

    res_path = OUT / "results.json"
    res_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 Mini-Dreamer 排行榜结果已保存：{res_path}")

    # 若 Phase 5 排行榜结果已存在，自动合画对比图
    p5_json = P5_OUT / "results.json"
    p5_data = json.loads(p5_json.read_text(encoding="utf-8")) if p5_json.exists() else None

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.2))
    if len(tasks) == 1:
        axes = [axes]
    for ax, task_name in zip(axes, tasks):
        if p5_data and task_name in p5_data.get("tasks", {}):
            for algo, info in p5_data["tasks"][task_name].items():
                ax.plot(info["steps"], info["mean_curve"], "--", alpha=0.75, label=f"{algo} ({info['final_mean']:.0f})")
        d_info = summary["tasks"][task_name]
        st, mu, sd = np.array(d_info["steps"]), np.array(d_info["mean_curve"]), np.array(d_info["std_curve"])
        ax.plot(st, mu, "s-", color="crimson", linewidth=2.5, label=f"Dreamer V3 ({d_info['final_mean']:.0f})")
        ax.fill_between(st, mu - sd, mu + sd, color="crimson", alpha=0.15)
        ax.set_title(task_name)
        ax.set_xlabel("真实环境步数")
        ax.set_ylabel("无噪评估回报")
        ax.legend(fontsize=8)
    save_figure(fig, OUT / "leaderboard-with-dreamer.png")


if __name__ == "__main__":
    main()
