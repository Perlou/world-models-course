"""
Phase 11 · 第 08 课：真机上的世界模型 —— 延迟、异步学习与目标图像规划
===================================================================

学习目标：
    1. 核对 CONCEPT.md 第 8.3 节异步执行可行条件 d ≤ T_a ≤ k - d 的手算表格（30 Hz, k=16）
    2. 在 PushT 上给同一个训练好的 Diffusion Policy 注入观测延迟 d ∈ {0, 2, 4, 8} 步，
       对比「直接执行」与「丢弃动作块前 d 步过期动作」的恢复程度
    3. 实现双线程 AsyncActorLearner（DayDreamer 式固定控制节拍 Actor + 持续梯度更新 Learner）
    4. 测量目标图像条件 DINOv2 + CEM 规划的单步耗时，换算可支持的最高真机控制频率（Hz）

对应概念文档：
    src/phase-11-robot-world-models/CONCEPT.md 第 8 节

运行方式：
    uv run python src/phase-11-robot-world-models/08-real-robot-world-models.py
    WM_QUICK=1 uv run python src/phase-11-robot-world-models/08-real-robot-world-models.py

预计运行时间：完整约 75 秒（M1 Max CPU），冒烟约 12 秒
"""

import collections
import importlib.util
import math
import os
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.planning import cem_plan
from utils.pusht import load_pusht_demos, make_pusht_env

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_DIR)

_spec03 = importlib.util.spec_from_file_location("p11_03_rr", os.path.join(_DIR, "03-behavior-cloning-and-diffusion-policy.py"))
_mod03 = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(_mod03)

_spec_p10_06 = importlib.util.spec_from_file_location(
    "p10_06_rr", os.path.join(_SRC, "phase-10-jepa", "06-action-conditioned-latent-dynamics.py")
)
_mod_p10_06 = importlib.util.module_from_spec(_spec_p10_06)
_spec_p10_06.loader.exec_module(_mod_p10_06)

DiffusionPolicy = _mod03.DiffusionPolicy
build_state_chunk_dataset = _mod03.build_state_chunk_dataset
extract_obs_5d = _mod03.extract_obs_5d
LatentDynamics = _mod_p10_06.LatentDynamics
rollout_jepa = _mod_p10_06.rollout

plt = setup_matplotlib()
OUT = output_dir("phase-11")
SEED = 0


# ==================== 第一部分：异步执行可行区间与延迟仿真器 ====================


def feasible_execution_range(k=16, dt_ms=1000.0 / 30.0, delta_inf_ms=100.0):
    """CONCEPT.md 第 8.3 节：返回过期步数 d 与 T_a 可行区间 [d, k - d]（无解返回 None）。"""
    d = int(math.ceil(delta_inf_ms / dt_ms))
    if d > k - d:
        return d, None
    return d, (d, k - d)


def simulate_latency(policy, norm_fn, delay_d=0, drop_expired=False, T_a=4, n_episodes=15, max_steps=160):
    """在真实 gym-pusht 上给策略注入 d 步观测延迟，测试直接执行 vs 丢弃过期动作前 d 步。"""
    env = make_pusht_env(obs_type="pixels_agent_pos")
    succ, covs = 0, []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=4000 + ep)
        s0 = extract_obs_5d(obs, norm_fn)
        # 延迟队列保留最近 d + T_o 帧，策略只能读队首（d 步之前的历史窗口）
        hist_q = collections.deque([s0.copy() for _ in range(delay_d + policy.T_o)], maxlen=delay_d + policy.T_o)
        max_cov, ok = 0.0, False
        step = 0
        while step < max_steps:
            delayed_win = list(hist_q)[: policy.T_o]
            chunk = policy.predict_chunk(torch.from_numpy(np.asarray(delayed_win)).unsqueeze(0))[0].numpy()
            start_k = min(delay_d, policy.T_p - 1) if drop_expired else 0
            exec_len = min(T_a, policy.T_p - start_k)
            for k in range(start_k, start_k + exec_len):
                if step >= max_steps:
                    break
                act_env = np.clip((chunk[k] + 1.0) * 256.0, 0.0, 512.0)
                obs, r, term, trunc, info = env.step(act_env)
                cov = float(info.get("coverage", r))
                max_cov = max(max_cov, cov)
                if info.get("is_success", False) or cov >= 0.85:
                    ok = True
                hist_q.append(extract_obs_5d(obs, norm_fn))
                step += 1
                if term or trunc:
                    step = max_steps
                    break
        succ += int(ok); covs.append(max_cov)
    env.close()
    return succ / n_episodes, float(np.mean(covs))


# ==================== 第二部分：DayDreamer 式双线程异步 Actor-Learner ====================


class AsyncActorLearner:
    """用两个线程模拟 DayDreamer：Actor 按固定 25 Hz 节拍采经验，Learner 后台不停做梯度更新。"""

    def __init__(self, duration_sec=0.8, control_hz=25.0):
        self.duration_sec = duration_sec
        self.dt = 1.0 / control_hz
        self.lock = threading.Lock()
        self.replay = []
        self.model = nn.Sequential(nn.Linear(7, 64), nn.SiLU(), nn.Linear(64, 5))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.actor_steps = 0
        self.learner_updates = 0
        self.stop_flag = threading.Event()

    def _actor_loop(self):
        t_end = time.perf_counter() + self.duration_sec
        while time.perf_counter() < t_end:
            t_tick = time.perf_counter()
            s = np.random.randn(5).astype(np.float32)
            a = np.random.randn(2).astype(np.float32)
            ns = s + 0.1 * np.random.randn(5).astype(np.float32)
            with self.lock:
                self.replay.append((s, a, ns))
                self.actor_steps += 1
            remain = self.dt - (time.perf_counter() - t_tick)
            if remain > 0:
                time.sleep(remain)
        self.stop_flag.set()

    def _learner_loop(self):
        while not self.stop_flag.is_set():
            with self.lock:
                if len(self.replay) < 8:
                    batch = None
                else:
                    idxs = np.random.randint(0, len(self.replay), size=16)
                    batch = [self.replay[i] for i in idxs]
            if batch is None:
                time.sleep(0.002)
                continue
            sa = torch.from_numpy(np.asarray([np.concatenate([b[0], b[1]]) for b in batch]))
            ns = torch.from_numpy(np.asarray([b[2] for b in batch]))
            self.opt.zero_grad(); F.mse_loss(self.model(sa), ns).backward(); self.opt.step()
            self.learner_updates += 1

    def run(self):
        t_a = threading.Thread(target=self._actor_loop)
        t_l = threading.Thread(target=self._learner_loop)
        t_a.start(); t_l.start()
        t_a.join(); t_l.join()
        return self.actor_steps, self.learner_updates


def goal_image_cem_benchmark(quick=False):
    """测量目标图像条件 DINOv2 + CEM 规划在 CPU 上的单步耗时与最高控制频率。"""
    model = LatentDynamics(feat_dim=384, act_dim=8, num_patches=64, d_model=128, n_layers=2, n_heads=4, max_frames=8)
    model.eval()
    z_hist = torch.randn(1, 3, 64, 384)
    a_prev = torch.zeros(1, 2, 8)
    z_goal = torch.randn(64, 384)

    configs = [("轻量 (S=16, H=3, I=2)", 16, 3, 2), ("标准 (S=32, H=4, I=3)", 32, 4, 3)]
    if quick:
        configs = [("轻量 (S=8, H=2, I=2)", 8, 2, 2), ("标准 (S=16, H=3, I=2)", 16, 3, 2)]
    out = []
    for name, S, H, I in configs:
        t0 = time.perf_counter()

        def score_fn(cand_t):
            ct = cand_t.repeat(1, 1, 4)
            af = torch.cat([a_prev.repeat(len(ct), 1, 1), ct], dim=1)
            zo = rollout_jepa(model, z_hist.repeat(len(ct), 1, 1, 1), af, ctx_len=3)
            return -((zo[:, -1] - z_goal.unsqueeze(0)) ** 2).mean(dim=(1, 2))

        cem_plan(score_fn, torch.zeros(H, 2), action_low=-1.0, action_high=1.0, n_candidates=S, n_elites=max(2, S // 4), n_iters=I)
        ms = (time.perf_counter() - t0) * 1000.0
        max_hz = 1000.0 / max(ms, 1e-3)
        out.append((name, ms, max_hz))
    return out


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 8.3 节异步执行可行范围（30 Hz, k=16）")
    print("=" * 60)
    for ms in [100.0, 200.0, 300.0]:
        d, rng = feasible_execution_range(k=16, dt_ms=1000.0 / 30.0, delta_inf_ms=ms)
        print(f"  延迟 {ms:5.0f} ms → 过期步数 d = {d}，可行 T_a = {rng if rng else '无解（不可行）'} ✓")

    print("\n" + "=" * 60)
    print("第二部分：PushT 观测延迟消融（d ∈ {0, 2, 4, 8}，不修正 vs 丢弃过期动作）")
    print("=" * 60)
    demos = load_pusht_demos()
    obs_X, act_Y, norm_fn = build_state_chunk_dataset(demos, n_episodes=30 if quick else 140, T_o=2, T_p=12)
    dp = DiffusionPolicy(obs_dim=5, act_dim=2, T_o=2, T_p=12, hidden=192)
    opt = torch.optim.AdamW(dp.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(180 if quick else 1500):
        idx = torch.randint(0, len(obs_X), (128,))
        opt.zero_grad(); dp.loss(act_Y[idx], obs_X[idx]).backward(); opt.step()

    delays = [0, 2, 4, 8]
    n_eval = 6 if quick else 20
    max_steps = 80 if quick else 160
    raw_res, fix_res = {}, {}
    print(f"  {'延迟步 d':>8s} | {'不修正 成功率 / 覆盖度':>22s} | {'丢弃过期动作 成功率 / 覆盖度':>28s}")
    for d in delays:
        sr_r, cov_r = simulate_latency(dp, norm_fn, delay_d=d, drop_expired=False, T_a=4, n_episodes=n_eval, max_steps=max_steps)
        sr_f, cov_f = simulate_latency(dp, norm_fn, delay_d=d, drop_expired=True, T_a=4, n_episodes=n_eval, max_steps=max_steps)
        raw_res[d] = (sr_r, cov_r); fix_res[d] = (sr_f, cov_f)
        print(f"  {d:>8d} | {sr_r*100:6.1f}% / {cov_r:.3f}          | {sr_f*100:6.1f}% / {cov_f:.3f}")

    print("\n" + "=" * 60)
    print("第三部分：DayDreamer 式双线程异步学习 & V-JEPA 2-AC 式规划频率基准")
    print("=" * 60)
    async_runner = AsyncActorLearner(duration_sec=0.4 if quick else 1.0, control_hz=25.0)
    a_steps, l_updates = async_runner.run()
    print(f"  AsyncActorLearner（25 Hz 节拍）：Actor 采集 {a_steps} 步，Learner 同步完成 {l_updates} 次更新（更新/采集比 = {l_updates/max(a_steps,1):.1f}×）")

    cem_bench = goal_image_cem_benchmark(quick=quick)
    for nm, ms, hz in cem_bench:
        print(f"  目标图像 CEM 规划 [{nm}]：单步耗时 = {ms:.1f} ms → 最高控制频率 = {hz:.2f} Hz")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(delays, [raw_res[d][1] for d in delays], "o--", label="不修正（执行过期动作）")
    axes[0].plot(delays, [fix_res[d][1] for d in delays], "s-", label="丢弃前 d 步过期动作")
    axes[0].set_title("PushT 观测延迟对 Diffusion Policy 覆盖度的影响")
    axes[0].set_xlabel("观测延迟步数 d"); axes[0].set_ylabel("平均最高覆盖度"); axes[0].legend()

    axes[1].bar([c[0] for c in cem_bench], [c[2] for c in cem_bench], color=["tab:blue", "tab:orange"])
    axes[1].axhline(10.0, ls="--", color="red", label="PushT 实时节拍 (10 Hz)")
    axes[1].set_title("特征空间 CEM 规划可支持的最高控制频率 (Hz)"); axes[1].set_ylabel("Hz"); axes[1].legend()
    fig.tight_layout()
    save_figure(fig, OUT / "08-real-robot-constraints.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：09-other-frontiers.py")


if __name__ == "__main__":
    main()
