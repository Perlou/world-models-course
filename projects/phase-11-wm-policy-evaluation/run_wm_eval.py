"""Phase 11 实战项目主脚本：在 PushT 上训练 7 个不同水平策略并验证「专家数据 WM vs 混合失败数据 WM」的评估一致性。"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(os.path.dirname(_THIS)), "src")
for _p in (_THIS, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.pusht import load_pusht_demos, make_pusht_env
from wm_evaluator import (
    PushTBC,
    PushTDP,
    PushTWMEvaluator,
    RandomPushTPolicy,
    evaluate_in_real_env,
    evaluate_in_world_model,
    extract_demo_states_5d,
    extract_state_5d,
    mmrv,
    normalize_pusht_state,
    pearson_r,
    spearman_rho,
)

plt = setup_matplotlib()
OUT = output_dir("phase-11-project")


def build_policy_chunks(demos, n_episodes=140, T_o=2, T_p=8):
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    obs_l, act_l = [], []
    for ep in range(min(n_episodes, len(starts))):
        s, e = int(starts[ep]), int(ends[ep])
        L = e - s
        st = normalize_pusht_state(extract_demo_states_5d(demos, s, e))
        ac = (demos["actions"][s:e].astype(np.float32) / 256.0) - 1.0
        for t in range(L):
            obs_l.append(st[[max(0, t - (T_o - 1 - k)) for k in range(T_o)]])
            act_l.append(ac[[min(L - 1, t + k) for k in range(T_p)]])
    return torch.from_numpy(np.asarray(obs_l)), torch.from_numpy(np.asarray(act_l))


def collect_wm_transitions(demos, n_demo_eps=50, n_fail_eps=25):
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    demo_data = [[], [], [], []]
    for ep in range(min(n_demo_eps, len(starts))):
        s, e = int(starts[ep]), int(ends[ep])
        st_ep = normalize_pusht_state(extract_demo_states_5d(demos, s, e))
        ac_ep = (demos["actions"][s:e].astype(np.float32) / 256.0) - 1.0
        rw_ep = demos["rewards"][s:e].astype(np.float32)
        for t in range(min(len(st_ep) - 1, 120)):
            demo_data[0].append(st_ep[t]); demo_data[1].append(ac_ep[t])
            demo_data[2].append(st_ep[t + 1]); demo_data[3].append(float(rw_ep[t]))

    env = make_pusht_env(obs_type="pixels_agent_pos")

    fail_data = [[], [], [], []]
    rng = np.random.default_rng(99)
    for ep in range(n_fail_eps):
        obs, _ = env.reset(seed=6000 + ep)
        sc = extract_state_5d(obs)
        ca = obs["agent_pos"].copy()
        for _ in range(70):
            ca = np.clip(ca + rng.normal(0.0, 50.0, size=2), 15.0, 497.0)
            obs, r, term, trunc, info = env.step(ca)
            sn = extract_state_5d(obs)
            fail_data[0].append(sc); fail_data[1].append((ca / 256.0) - 1.0)
            fail_data[2].append(sn); fail_data[3].append(float(info.get("coverage", r)))
            sc = sn
            if term or trunc:
                break
    env.close()
    to_t = lambda lst: tuple(torch.from_numpy(np.asarray(x, dtype=np.float32)) for x in lst)
    return to_t(demo_data), to_t(fail_data)


def train_wm(s, a, ns, cov, n_steps=1000, seed=0):
    seed_everything(seed)
    wm = PushTWMEvaluator()
    opt = torch.optim.AdamW(wm.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(n_steps):
        i = torch.randint(0, len(s), (128,))
        nh, ch = wm.step(s[i], a[i])
        loss = F.mse_loss(nh, ns[i]) + 0.5 * F.mse_loss(ch, cov[i])
        opt.zero_grad(); loss.backward(); opt.step()
    return wm


def main():
    t0 = time.time()
    seed_everything(0)
    quick = is_quick()

    print("=" * 68)
    print("Phase 11 实战项目：用世界模型评估 PushT 策略套件（相关性与排序一致性）")
    print("=" * 68)
    demos = load_pusht_demos()
    obs_X, act_Y = build_policy_chunks(demos, n_episodes=30 if quick else 150)

    # 训练两个世界模型对照：仅演示数据 vs 演示 + 失败探索数据（检验第 6.6 节"乐观偏差"）
    demo_t, fail_t = collect_wm_transitions(demos, n_demo_eps=15 if quick else 55, n_fail_eps=6 if quick else 25)
    mix_t = tuple(torch.cat([d, f], dim=0) for d, f in zip(demo_t, fail_t))
    wm_steps = 150 if quick else 1300
    wm_demo = train_wm(*demo_t, n_steps=wm_steps, seed=0)
    wm_mix = train_wm(*mix_t, n_steps=wm_steps, seed=1)

    # 构建 7 个不同水平的策略
    scale = 0.2 if quick else 1.0
    def make_bc(steps):
        m = PushTBC()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(steps):
            i = torch.randint(0, len(obs_X), (128,))
            opt.zero_grad(); F.mse_loss(m(obs_X[i]), act_Y[i]).backward(); opt.step()
        return m

    def make_dp(steps):
        m = PushTDP()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(steps):
            i = torch.randint(0, len(obs_X), (128,))
            opt.zero_grad(); m.loss(act_Y[i], obs_X[i]).backward(); opt.step()
        return m

    suite = [
        ("Random", RandomPushTPolicy()),
        ("BC-50", make_bc(max(10, int(50 * scale)))),
        ("BC-250", make_bc(max(25, int(250 * scale)))),
        ("BC-1000", make_bc(max(60, int(1000 * scale)))),
        ("DP-150", make_dp(max(20, int(150 * scale)))),
        ("DP-500", make_dp(max(45, int(500 * scale)))),
        ("DP-1400", make_dp(max(90, int(1400 * scale)))),
    ]

    eval_seeds = [7000 + i for i in range(6 if quick else 20)]
    max_steps = 60 if quick else 120
    rows = []
    print(f"\n  {'策略':10s} | {'真实覆盖 y':>10s} | {'仅演示WM ŷ':>10s} | {'混合WM ŷ':>10s} | {'真实达标率':>10s} | {'混合WM达标率':>12s}")
    for nm, pol in suite:
        r_cov, r_sr, init_s = evaluate_in_real_env(pol, eval_seeds, max_steps=max_steps)
        d_cov, d_sr = evaluate_in_world_model(pol, wm_demo, init_s, max_steps=max_steps)
        m_cov, m_sr = evaluate_in_world_model(pol, wm_mix, init_s, max_steps=max_steps)
        rows.append({
            "name": nm, "real_cov": r_cov, "wm_demo_cov": d_cov, "wm_mix_cov": m_cov,
            "real_sr": r_sr, "wm_demo_sr": d_sr, "wm_mix_sr": m_sr,
        })
        print(f"  {nm:10s} | {r_cov:10.3f} | {d_cov:10.3f} | {m_cov:10.3f} | {r_sr*100:9.1f}% | {m_sr*100:11.1f}%")

    y_cov = [r["real_cov"] for r in rows]
    yh_demo = [r["wm_demo_cov"] for r in rows]
    yh_mix = [r["wm_mix_cov"] for r in rows]
    metrics_demo = {"pearson": pearson_r(y_cov, yh_demo), "spearman": spearman_rho(y_cov, yh_demo), "mmrv": mmrv(y_cov, yh_demo)}
    metrics_mix = {"pearson": pearson_r(y_cov, yh_mix), "spearman": spearman_rho(y_cov, yh_mix), "mmrv": mmrv(y_cov, yh_mix)}

    print(f"\n  仅演示数据 WM 评估指标：Pearson = {metrics_demo['pearson']:.3f} | Spearman = {metrics_demo['spearman']:.3f} | MMRV = {metrics_demo['mmrv']:.3f}")
    print(f"  混合失败数据 WM 指标  ：Pearson = {metrics_mix['pearson']:.3f} | Spearman = {metrics_mix['spearman']:.3f} | MMRV = {metrics_mix['mmrv']:.3f}")

    results = {
        "total_seconds": time.time() - t0,
        "n_eval_episodes": len(eval_seeds),
        "max_steps": max_steps,
        "policies": rows,
        "metrics_wm_demo_only": metrics_demo,
        "metrics_wm_mixed": metrics_mix,
    }
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for ax, yh, met, title, col in [
        (axes[0], yh_demo, metrics_demo, "仅专家演示训练的 WM（易对差策略乐观）", "tab:orange"),
        (axes[1], yh_mix, metrics_mix, "混合专家+失败轨迹训练的 WM", "tab:blue"),
    ]:
        ax.scatter(y_cov, yh, s=70, color=col)
        for r in rows:
            yv = r["wm_demo_cov"] if col == "tab:orange" else r["wm_mix_cov"]
            ax.annotate(r["name"], (r["real_cov"], yv), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("真实 PushT 覆盖度 y"); ax.set_ylabel("世界模型预测覆盖度 ŷ")
        ax.set_title(f"{title}\nPearson={met['pearson']:.3f}, Spearman={met['spearman']:.3f}, MMRV={met['mmrv']:.3f}")
    fig.tight_layout()
    save_figure(fig, OUT / "scatter-and-ranking.png")
    print(f"\n📄 结果已保存：{OUT / 'results.json'}（总用时 {results['total_seconds']:.1f}s）")


if __name__ == "__main__":
    main()
