"""运行 Phase 7 实战项目：TD-MPC2 接入统一连续控制排行榜 + MuZero 井字棋交付物。"""

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
from muzero_tictactoe import train_muzero_tictactoe
from tdmpc2_core import run_tdmpc2_on_benchmark
from utils import is_quick, output_dir, save_figure, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-07-project")
P5_OUT = output_dir("phase-05-project")
P6_OUT = output_dir("phase-06-project")


def main():
    quick = is_quick()
    tasks = ["Pendulum-v1"] if quick else list(BENCHMARK_TASKS.keys())
    summary = {"quick": quick, "tasks": {}, "muzero_tictactoe": {}}

    # 1. 交付物 A：TD-MPC2 加入 Phase 5–7 连续控制统一排行榜
    for task_name in tasks:
        cfg = get_task_protocol(task_name, quick=quick)
        print(f"\n运行 TD-MPC2：{task_name}（预算 {cfg['total_steps']} 步，种子 {cfg['seeds']}）")
        curves, wall_ts, dec_ms = [], [], []
        steps_ref = None
        for s in cfg["seeds"]:
            st, rt, dt, ms = run_tdmpc2_on_benchmark(task_name, cfg, seed=s)
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
        print(f"  TD-MPC2 ｜ 最终 {mean_c[-1]:7.1f} ± {std_c[-1]:5.1f} ｜ "
              f"单步决策 {np.mean(dec_ms):.2f} ms ｜ 单子耗时 {np.mean(wall_ts):.1f}s")

    # 2. 交付物 B：MuZero 学会井字棋
    print("\n运行附带交付物：MuZero 井字棋（无规则隐空间树搜索）……")
    _, ttt_stats = train_muzero_tictactoe(n_games=20 if quick else 160, n_sims=12 if quick else 28, seed=0)
    summary["muzero_tictactoe"] = ttt_stats
    print(f"  MuZero 井字棋对随机对手（40 盘）：胜 {ttt_stats['win_rate']*100:.1f}% ｜ "
          f"和 {ttt_stats['draw_rate']*100:.1f}% ｜ 负 {ttt_stats['loss_rate']*100:.1f}%")

    res_path = OUT / "results.json"
    res_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 Phase 7 排行榜结果已保存：{res_path}")

    # 合并 Phase 5 + Phase 6 + Phase 7 全家福曲线图
    p5 = json.loads((P5_OUT / "results.json").read_text(encoding="utf-8")) if (P5_OUT / "results.json").exists() else None
    p6 = json.loads((P6_OUT / "results.json").read_text(encoding="utf-8")) if (P6_OUT / "results.json").exists() else None

    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.3))
    if len(tasks) == 1:
        axes = [axes]
    for ax, task_name in zip(axes, tasks):
        if p5 and task_name in p5.get("tasks", {}):
            for algo, info in p5["tasks"][task_name].items():
                ax.plot(info["steps"], info["mean_curve"], "--", alpha=0.65, label=f"{algo} ({info['final_mean']:.0f})")
        if p6 and task_name in p6.get("tasks", {}):
            d6 = p6["tasks"][task_name]
            ax.plot(d6["steps"], d6["mean_curve"], "^-", color="crimson", alpha=0.85, linewidth=2.0, label=f"Dreamer V3 ({d6['final_mean']:.0f})")
        d7 = summary["tasks"][task_name]
        st, mu, sd = np.array(d7["steps"]), np.array(d7["mean_curve"]), np.array(d7["std_curve"])
        ax.plot(st, mu, "o-", color="navy", linewidth=2.5, label=f"TD-MPC2 ({d7['final_mean']:.0f})")
        ax.fill_between(st, mu - sd, mu + sd, color="navy", alpha=0.14)
        ax.set_title(task_name)
        ax.set_xlabel("真实环境步数")
        ax.set_ylabel("无噪评估回报")
        ax.legend(fontsize=7.5)
    save_figure(fig, OUT / "leaderboard-full-p5-p7.png")


if __name__ == "__main__":
    main()
