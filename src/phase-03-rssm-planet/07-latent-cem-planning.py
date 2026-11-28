"""
在隐空间里用 CEM 规划
=====================

学习目标：
    1. 兑现全课"同一个规划器"线索：直接调用 src/utils/planning.py 的 cem_plan，只换打分函数就在隐空间完成规划
    2. 复现 CONCEPT.md 第 9.5 节的 4 候选 × 2 步隐空间 CEM 手算例子
    3. 实测单次规划耗时：看清"只在 230 维隐空间展开 + 奖励头打分"比"每步解码成 64×64 图像"快多少倍
    4. 诊断"想象回报 vs 真实回报"的散点图与相关系数，看规划器选出的高分序列在真实环境里是不是真的高分
    5. 固定训练好的世界模型，扫描规划视野 H、候选数 N，以及冷启动 vs 热启动对像素摆控制回报的影响

核心概念：
    - 隐空间规划（latent-space planning）：起点是后验校准过的信念 (h_t, z_t)，展开用先验 imagine()，打分用奖励头
    - 面向接口编程：cem_plan 只依赖 score_fn(actions [N, H, A]) → scores [N]
    - 单样本轨迹打分：每条候选只采样一条先验轨迹，靠精英集合平均平滑采样噪声
    - 冷启动 vs 热启动：PlaNet 每步从 N(0, I) 重新搜索；热启动平移上一步均值，少迭代轮数时更省算力

前置知识：
    - Phase 1 的 06-cem-and-mppi.py（CEM 规划器）
    - 05-world-model-training.py、06-open-loop-prediction-and-overshooting.py

对应概念文档：
    - CONCEPT.md 第 9 节「在隐空间里用 CEM 规划」

预计运行时间：约 4 分钟（CPU）/ 约 1 分钟（MPS / GPU）
"""

import importlib.util
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import numpy as np
import torch

from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.planning import cem_plan, shift_plan

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lesson04 = _load_module("lesson04", "04-rssm-from-scratch.py")
_lesson05 = _load_module("lesson05", "05-world-model-training.py")

WorldModel = _lesson04.WorldModel
ACTION_DIM = _lesson04.ACTION_DIM
render_pendulum = _lesson05.render_pendulum
step_with_repeat = _lesson05.step_with_repeat
preprocess_obs = _lesson05.preprocess_obs
ACTION_REPEAT = _lesson05.ACTION_REPEAT
EPISODE_DECISIONS = _lesson05.EPISODE_DECISIONS

plt = setup_matplotlib()
OUT = output_dir("phase-03")
SEED = 0
DEVICE = get_device()
SUFFIX = "-quick" if is_quick() else ""
MODEL_PATH = OUT / f"world_model{SUFFIX}.pt"


# ==================== 第一部分：导读与手算核对 ====================


def introduction():
    print("=" * 60)
    print("第一部分：同一个 CEM，换一个打分函数就进了隐空间")
    print("=" * 60)
    print("""
回看 src/utils/planning.py 里的 cem_plan：它只要求调用方提供一个函数
    score_fn(actions [N, H, A]) → scores [N]

Phase 1 里，这个函数在 2 维真实物理状态 (θ, θ̇) 上展开、用已知奖励公式打分；
这一节，我们只写十几行胶水代码把它接到 RSSM 上：

    最新画面 o_t ──▶ 编码器 + observe_step ──▶ 当前信念 (h_t, z_t)   ← 唯一用后验的地方
                                                    │
                                     复制 N 份送进 cem_plan
                                                    ▼
                    候选动作 [N, H, 1] ──▶ RSSM.imagine ──▶ 奖励头求和 ──▶ 分数 [N]
                                            （纯先验，全程不调用解码器！）
    """)


def verify_hand_calculation():
    """复现 CONCEPT.md 第 9.5 节：4 条候选、视野 H=2、精英 K=2 的一轮 CEM 更新。"""
    print("\n" + "=" * 60)
    print("第二部分：核对 CONCEPT.md 第 9.5 节的手算例子")
    print("=" * 60)
    actions = torch.tensor([[-0.5, 0.9], [0.8, 0.3], [-1.2, -0.4], [0.4, -0.1]])
    pred_rewards = torch.tensor([[-1.0, -0.8], [-0.5, -0.2], [-1.2, -1.5], [-0.4, -0.6]])
    scores = pred_rewards.sum(dim=-1)
    elites = actions[scores.topk(2).indices]
    mu = elites.mean(dim=0)
    sigma = elites.std(dim=0, correction=0)
    print(f"  四条候选的想象总分：{scores.tolist()} → 精英下标 {scores.topk(2).indices.tolist()}")
    print(f"  更新后的均值 μ = ({mu[0]:.1f}, {mu[1]:.1f})，标准差 σ = ({sigma[0]:.1f}, {sigma[1]:.1f})，执行第 1 个动作 {mu[0]:.1f}")
    assert torch.allclose(mu, torch.tensor([0.6, 0.1])) and torch.allclose(sigma, torch.tensor([0.2, 0.2]))
    print("  ✓ 与 CONCEPT.md 第 9.5 节完全一致。")


# ==================== 第二部分：隐空间 CEM 规划器 ====================


def ensure_model():
    if MODEL_PATH.exists():
        return
    print("⚠️  未找到 05 节的模型权重，先运行 05-world-model-training.py 生成……")
    subprocess.run([sys.executable, os.path.join(_DIR, "05-world-model-training.py")], check=True)


@torch.no_grad()
def latent_cem_plan(model, h, z, horizon=15, n_candidates=400, n_elites=40, n_iters=6,
                    warm_mean=None, generator=None, return_imagined_score=False):
    """在 RSSM 的隐空间里运行 CEM 规划（CONCEPT.md 第 9.6、9.7 节）。

    h [1, 200]，z [1, 30]：刚用当前帧观测校准过的后验状态。
    动作统一归一化在 [-1, 1] 内搜索；warm_mean 不为 None 时使用热启动，否则每步从 0 均值冷启动（PlaNet 默认）。
    """
    h_batch = h.expand(n_candidates, -1)
    z_batch = z.expand(n_candidates, -1)

    def score_fn(actions):
        feats = model.rssm.imagine(h_batch, z_batch, actions, sample=True, generator=generator)
        return model.reward_head(feats).sum(dim=-1)

    if warm_mean is None:
        init_mean = torch.zeros(horizon, ACTION_DIM, device=h.device)
        smoothing = 0.0
    else:
        init_mean = warm_mean
        smoothing = 0.1

    plan_mean = cem_plan(
        score_fn,
        init_mean,
        generator=generator,
        action_low=-1.0,
        action_high=1.0,
        n_candidates=n_candidates,
        n_elites=n_elites,
        n_iters=n_iters,
        init_std=1.0,
        min_std=0.05,
        smoothing=smoothing,
    )
    if not return_imagined_score:
        return plan_mean
    # 用最终均值序列评估一次预期累计奖励（供"想象回报 vs 真实回报"诊断）
    imag_feats = model.rssm.imagine(h, z, plan_mean.unsqueeze(0), sample=False)
    imag_return = model.reward_head(imag_feats).sum().item()
    return plan_mean, imag_return


# ==================== 第三部分：单次规划耗时对比 ====================


@torch.no_grad()
def benchmark_planning(model, n_candidates, horizon, n_iters):
    print("\n" + "=" * 60)
    print("第三部分：为什么隐空间规划必须跳过解码器——实测耗时对比")
    print("=" * 60)

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    elif torch.cuda.is_available():
        devices.append("cuda")

    timings = {}
    for dev_name in devices:
        dev = torch.device(dev_name)
        m = WorldModel().to(dev)
        m.load_state_dict(model.state_dict())
        m.eval()
        h, z = m.rssm.initial_state(1, dev)
        gen = torch.Generator(device=dev).manual_seed(SEED)

        # 预热一次
        latent_cem_plan(m, h, z, horizon=horizon, n_candidates=n_candidates,
                        n_elites=max(10, n_candidates // 10), n_iters=n_iters, generator=gen)
        if dev_name == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        n_rep = 5
        for _ in range(n_rep):
            latent_cem_plan(m, h, z, horizon=horizon, n_candidates=n_candidates,
                            n_elites=max(10, n_candidates // 10), n_iters=n_iters, generator=gen)
        if dev_name == "mps":
            torch.mps.synchronize()
        ms_latent = (time.time() - t0) / n_rep * 1000.0

        # 对比：如果每一轮每一步都把 [N, 230] 解码成 64×64 图像要多久（只跑 1 轮推算 I 轮，避免白等）
        feats_dummy = torch.randn(n_candidates * horizon, 230, device=dev)
        _ = m.decoder(feats_dummy[:32])
        if dev_name == "mps":
            torch.mps.synchronize()
        t1 = time.time()
        _ = m.decoder(feats_dummy)
        if dev_name == "mps":
            torch.mps.synchronize()
        ms_decode_all = (time.time() - t1) * n_iters * 1000.0 + ms_latent
        timings[dev_name] = (ms_latent, ms_decode_all)
        print(f"  设备 {dev_name.upper():<4}（N={n_candidates}, H={horizon}, I={n_iters}，共 {n_candidates * horizon * n_iters:,} 次想象步）：")
        print(f"      纯隐空间 + 奖励头：          {ms_latent:7.1f} ms / 步")
        print(f"      若每步都调用解码器生成图像：  {ms_decode_all:7.1f} ms / 步（慢 {ms_decode_all / ms_latent:.1f} 倍！）")
    return timings


# ==================== 第四部分：想象回报 vs 真实回报 ====================


@torch.no_grad()
def imagined_vs_real_experiment(model, n_queries, horizon=15):
    """CONCEPT.md 第 9.7 节诊断：在同一批起点上比较想象 H 步回报与真实环境 H 步回报。"""
    print("\n" + "=" * 60)
    print("第四部分：想象回报 vs 真实回报——模型排出的名次准不准？")
    print("=" * 60)

    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    rng = np.random.default_rng(SEED + 10)
    imag_returns, real_returns, is_planned = [], [], []

    n_states = max(4, n_queries // 8)
    per_state = max(2, n_queries // n_states)

    for s_idx in range(n_states):
        env.reset(seed=500 + s_idx)
        h, z = model.rssm.initial_state(1, DEVICE)
        a_prev = torch.zeros(1, ACTION_DIM, device=DEVICE)
        # 先用 8 步随机动作走一段，让后验建立起准确的速度信念
        for k in range(8):
            frame = render_pendulum(env)
            obs_t = preprocess_obs(torch.from_numpy(frame[None]).to(DEVICE))
            embed_t = model.encoder(obs_t)
            h, z, _, _ = model.rssm.observe_step(h, z, a_prev, embed_t, sample=False)
            if k < 7:
                act_np = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
                step_with_repeat(env, act_np)
                a_prev = torch.from_numpy(act_np[None]).to(DEVICE)

        saved_state = env.unwrapped.state.copy()
        gen = torch.Generator(device=DEVICE).manual_seed(SEED + s_idx)
        best_plan, _ = latent_cem_plan(model, h, z, horizon=horizon, n_candidates=300, n_elites=30,
                                       n_iters=6, generator=gen, return_imagined_score=True)

        for j in range(per_state):
            if j == 0:
                seq = best_plan  # 第 0 条放 CEM 精心挑出的序列，看有没有显著高估（钻空子）
                planned = True
            else:
                seq = torch.from_numpy(rng.uniform(-1.0, 1.0, size=(horizon, ACTION_DIM)).astype(np.float32)).to(DEVICE)
                planned = False

            # 模型想象的累计奖励（4 次先验采样取平均）
            feats = model.rssm.imagine(h.expand(4, -1), z.expand(4, -1), seq.unsqueeze(0).expand(4, -1, -1))
            imag_r = model.reward_head(feats).sum(dim=-1).mean().item()

            # 真实环境从完全相同的物理状态出发执行同一串动作
            env.unwrapped.state = saved_state.copy()
            real_r = 0.0
            for step_k in range(horizon):
                _, r, _ = step_with_repeat(env, seq[step_k].cpu().numpy())
                real_r += r

            imag_returns.append(imag_r)
            real_returns.append(real_r)
            is_planned.append(planned)
    env.close()

    imag_arr = np.array(imag_returns)
    real_arr = np.array(real_returns)
    plan_mask = np.array(is_planned)
    corr = float(np.corrcoef(imag_arr, real_arr)[0, 1])
    bias_rand = float((imag_arr[~plan_mask] - real_arr[~plan_mask]).mean())
    bias_plan = float((imag_arr[plan_mask] - real_arr[plan_mask]).mean())

    print(f"""
  共测试 {len(imag_arr)} 条长度为 H = {horizon} 的动作序列（含 {int(plan_mask.sum())} 条 CEM 选出的最优序列）：
      皮尔逊相关系数 r            = {corr:.3f}
      随机候选序列的平均预测偏差  = {bias_rand:+.2f}（想象 − 真实）
      CEM 最优序列的平均预测偏差  = {bias_plan:+.2f}（正值越大，说明规划器越倾向挑模型高估的序列）
    """)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    lo = min(imag_arr.min(), real_arr.min()) - 5
    hi = max(imag_arr.max(), real_arr.max()) + 5
    ax.scatter(real_arr[~plan_mask], imag_arr[~plan_mask], s=18, alpha=0.55, label="随机动作序列")
    ax.scatter(real_arr[plan_mask], imag_arr[plan_mask], s=42, color="C3", marker="^", label="CEM 选出的动作序列")
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="理想 y = x")
    ax.set_xlabel(f"真实环境的 {horizon} 步累计奖励")
    ax.set_ylabel(f"模型想象的 {horizon} 步累计奖励")
    ax.set_title(f"想象回报 vs 真实回报（相关系数 r = {corr:.3f}）")
    ax.legend(fontsize=9)
    save_figure(fig, OUT / "07-imagined-vs-real.png")
    return corr, bias_rand, bias_plan


# ==================== 第五部分：在线闭环控制与超参数扫描 ====================


@torch.no_grad()
def run_planet_episode(model, seed, horizon=15, n_candidates=300, n_iters=6, warm_start=False):
    """只看 64×64 像素画面，用隐空间 CEM 控制一个完整回合（200 个环境步 = 100 个决策步）。"""
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.reset(seed=seed)
    h, z = model.rssm.initial_state(1, DEVICE)
    a_prev = torch.zeros(1, ACTION_DIM, device=DEVICE)
    warm_mean = torch.zeros(horizon, ACTION_DIM, device=DEVICE) if warm_start else None
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    total_reward = 0.0
    for _ in range(EPISODE_DECISIONS):
        frame = render_pendulum(env)
        obs_t = preprocess_obs(torch.from_numpy(frame[None]).to(DEVICE))
        embed_t = model.encoder(obs_t)
        h, z, _, _ = model.rssm.observe_step(h, z, a_prev, embed_t, sample=False)

        plan_mean = latent_cem_plan(
            model, h, z, horizon=horizon, n_candidates=n_candidates,
            n_elites=max(10, n_candidates // 10), n_iters=n_iters,
            warm_mean=warm_mean, generator=gen,
        )
        action = plan_mean[0]
        _, r, _ = step_with_repeat(env, action.cpu().numpy())
        total_reward += r
        a_prev = action.unsqueeze(0)
        if warm_start:
            warm_mean = shift_plan(plan_mean)
    env.close()
    return total_reward


def sweep_planning_budget(model, eval_seeds, n_cand_default, n_iters_default):
    print("\n" + "=" * 60)
    print("第五部分：固定世界模型，扫描规划超参数（H、N、冷启动 vs 热启动）")
    print("=" * 60)

    # 随机策略参照线
    rand_rets = []
    for s in eval_seeds:
        env = gym.make("Pendulum-v1")
        env.reset(seed=s)
        rng = np.random.default_rng(s)
        tot = 0.0
        for _ in range(200):
            _, r, _, _, _ = env.step(rng.uniform(-2.0, 2.0, size=1).astype(np.float32))
            tot += float(r)
        env.close()
        rand_rets.append(tot)
    rand_mean = float(np.mean(rand_rets))
    print(f"  参照：随机策略在同组评估种子上的平均回报 = {rand_mean:.1f}\n")

    horizons = [5, 10, 15] if is_quick() else [5, 10, 15, 25]
    h_returns = []
    print("  1. 扫描规划视野 H（冷启动）：")
    for h_val in horizons:
        rets = [run_planet_episode(model, s, horizon=h_val, n_candidates=n_cand_default, n_iters=n_iters_default)
                for s in eval_seeds]
        h_returns.append((float(np.mean(rets)), float(np.std(rets))))
        print(f"      H = {h_val:<3}（对应 {h_val * ACTION_REPEAT:<2} 个环境步）  回合回报 = {h_returns[-1][0]:7.1f} ± {h_returns[-1][1]:5.1f}")

    # 冷启动 vs 热启动在少迭代轮数下的对比
    iters_compare = [2, 5] if is_quick() else [2, 4, 8]
    cold_rets, warm_rets = [], []
    print("\n  2. 对比 PlaNet 冷启动 vs Phase 1 热启动（固定 H = 15）：")
    for n_it in iters_compare:
        c_r = [run_planet_episode(model, s, horizon=15, n_candidates=n_cand_default, n_iters=n_it, warm_start=False)
               for s in eval_seeds]
        w_r = [run_planet_episode(model, s, horizon=15, n_candidates=n_cand_default, n_iters=n_it, warm_start=True)
               for s in eval_seeds]
        cold_rets.append(float(np.mean(c_r)))
        warm_rets.append(float(np.mean(w_r)))
        print(f"      CEM 轮数 I = {n_it}：冷启动 {cold_rets[-1]:7.1f}  ｜  热启动 {warm_rets[-1]:7.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].errorbar(horizons, [x[0] for x in h_returns], yerr=[x[1] for x in h_returns],
                     marker="o", capsize=4, linewidth=2, label="隐空间 CEM")
    axes[0].axhline(rand_mean, color="tomato", linestyle=":", label=f"随机策略（{rand_mean:.0f}）")
    axes[0].set_xlabel("规划视野 H（决策步，每步 = 2 环境步）")
    axes[0].set_ylabel("真实回合回报（越高越好）")
    axes[0].set_title("视野太短甩不起摆，过长受模型漂移影响")
    axes[0].legend(fontsize=8.5)

    x_idx = np.arange(len(iters_compare))
    w = 0.32
    axes[1].bar(x_idx - w / 2, cold_rets, width=w, label="冷启动（每步重置 N(0,1)）")
    axes[1].bar(x_idx + w / 2, warm_rets, width=w, label="热启动（平移上一步均值）")
    axes[1].axhline(rand_mean, color="tomato", linestyle=":")
    axes[1].set_xticks(x_idx)
    axes[1].set_xticklabels([f"I = {it}" for it in iters_compare])
    axes[1].set_ylabel("真实回合回报")
    axes[1].set_title("冷启动 vs 热启动随 CEM 迭代轮数的表现")
    axes[1].legend(fontsize=8.5)
    save_figure(fig, OUT / "07-planning-sweep.png")


# ==================== 第六部分：练习与思考 ====================


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把 run_planet_episode 里的信念起点从后验 (h, z) 故意改成先验（即不看当前画面 o_t，只喂动作），回报会变多少？

练习 1 参考答案：
    那等于从回合第 1 帧之前起就蒙着眼睛盲开 100 步，连摆一开始在哪都不知道，回报直接退化到随机策略水平。
    CONCEPT.md 第 9.8 节强调：规划的起点必须是刚用最新观测校准过的后验，只有从起点往后的 H 步才走先验。

练习 2：换成 MPPI 规划器需要改几行代码？
    src/utils/planning.py 里同时提供了 mppi_plan。试把 latent_cem_plan 里的 cem_plan 换成 mppi_plan（注意温度与回报量级匹配）。

练习 2 参考答案：
    只需把 cem_plan(...) 一行替换成 mppi_plan(score_fn, init_mean, temperature=2.0, ...)，打分函数一行不用改。
    这就是把"模型展开"和"采样优化器"解耦的好处。

思考题 1：本节用的模型只见过 50 个随机探索回合，从来没见过"把摆稳在正上方"的轨迹，为什么已经能把摆甩上去？
          那还有什么缺陷需要下一节的闭环来补？

思考题 1 参考答案：
    Pendulum-v1 的 reset() 覆盖了 [-π, π] 的各种角度，随机与平滑随机动作让模型在大部分区域都见过单步规律，
    CEM 把这些局部规律串成了甩摆路线。但正上方"微调平衡（小角度、近零角速度）"在随机数据里停留极短，
    模型在那里的奖励和动力学不够精细，需要下一节"边规划控制、边把新数据加回缓冲区重训"来补齐最后一块拼图。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    if DEVICE.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    quick = is_quick()
    if quick:
        print("（冒烟模式：候选数和评估回合数大幅缩小，仅验证流程跑通）\n")

    n_cand, horizon, n_iters = (60, 10, 3) if quick else (300, 15, 6)
    n_queries = 16 if quick else 160
    eval_seeds = [1000] if quick else [1000, 1001, 1002, 1003, 1004]

    introduction()
    verify_hand_calculation()
    ensure_model()

    model = WorldModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    benchmark_planning(model, n_candidates=n_cand, horizon=horizon, n_iters=n_iters)
    imagined_vs_real_experiment(model, n_queries=n_queries, horizon=horizon)
    sweep_planning_budget(model, eval_seeds=eval_seeds, n_cand_default=n_cand, n_iters_default=n_iters)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：08-planet-agent.py（完整的 PlaNet 在线采集—训练—规划闭环）")
    print("=" * 60)


if __name__ == "__main__":
    main()
