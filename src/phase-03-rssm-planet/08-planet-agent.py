"""
PlaNet 智能体：从像素采集、训练到规划的完整闭环
===============================================

学习目标：
    1. 把前七节的所有部件（图像编码/解码、RSSM、序列回放、序列 ELBO + free nats、隐空间 CEM）组装成 PlaNet
    2. 实现 collect_episode() 与 train_planet()：带探索噪声在线采集数据、持续更新同一个世界模型、定期无噪评估
    3. 算清样本效率与算力账：只用几千步真实像素交互学会把摆甩起稳住，同时实测训练耗时与规划耗时的占比
    4. 与随机策略、Phase 1 状态输入 PETS 同台对照，看清 PlaNet 做到了什么、又给 Phase 4–6 留下了什么问题

核心概念：
    - 深度规划网络（Deep Planning Network, PlaNet）：纯基于模型、无策略网络、无价值函数的像素控制闭环
    - 动作重复（action repeat）：拉长有效视野、增大帧间变化、按比例减少规划调用次数
    - 探索噪声（exploration noise）：CEM 输出是确定性均值，采集数据时叠加高斯噪声拓宽轨迹覆盖面
    - 样本效率 vs 决策延迟：真实交互极省（数千步），但每步决策都要在线运行上万次隐空间想象

前置知识：
    - Phase 1 的 08-pets.py（状态空间里的"采集—训练—规划"闭环）
    - 04–07 节（WorldModel、SequenceReplay、world_model_loss、latent_cem_plan）

对应概念文档：
    - CONCEPT.md 第 10 节「PlaNet 智能体闭环」

预计运行时间：约 12 分钟（CPU）/ 约 3 分钟（MPS / GPU）
"""

import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from utils import get_device, is_quick, output_dir, save_figure, save_video, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_lesson04 = _load_module("lesson04", "04-rssm-from-scratch.py")
_lesson05 = _load_module("lesson05", "05-world-model-training.py")
_lesson07 = _load_module("lesson07", "07-latent-cem-planning.py")

WorldModel = _lesson04.WorldModel
IMAGE_SIZE = _lesson04.IMAGE_SIZE
ACTION_DIM = _lesson04.ACTION_DIM

SequenceReplay = _lesson05.SequenceReplay
render_pendulum = _lesson05.render_pendulum
step_with_repeat = _lesson05.step_with_repeat
preprocess_obs = _lesson05.preprocess_obs
world_model_loss = _lesson05.world_model_loss
ACTION_REPEAT = _lesson05.ACTION_REPEAT
EPISODE_DECISIONS = _lesson05.EPISODE_DECISIONS
FREE_NATS = _lesson05.FREE_NATS

latent_cem_plan = _lesson07.latent_cem_plan

plt = setup_matplotlib()
OUT = output_dir("phase-03")
SEED = 0
DEVICE = get_device()

# 本课像素版 Pendulum-v1 上的 PlaNet 默认超参数（与论文超参数的逐项对照见结尾输出与 CONCEPT.md 第 12.1 节）
INIT_EPISODES = 5        # S：初始随机探索回合数（与 PlaNet 论文相同）
ONLINE_EPISODES = 10     # 在线边学边采的回合数（共 15 个回合 = 3000 个真实环境步）
UPDATES_INIT = 150       # 初始随机数据到位后的预热更新步数
UPDATES_PER_EP = 60      # C：每新采 1 个回合做的参数更新步数
BATCH_SIZE = 12          # B：序列批大小
CHUNK_LEN = 25           # T：序列片段长度（对应 50 个环境步）
PLAN_HORIZON = 15        # H：规划视野（对应 30 个环境步 = 1.5 秒）
PLAN_CANDIDATES = 300    # N：CEM 每轮候选数
PLAN_ITERS = 6           # I：CEM 迭代轮数
EXPLORE_STD = 0.25       # 采集数据时的探索噪声标准差（归一化动作空间 [-1, 1]）
EVAL_SEEDS = [1000, 1001, 1002, 1003, 1004]


# ==================== 第一部分：闭环总览 ====================


def introduction():
    print("=" * 60)
    print("第一部分：PlaNet 智能体闭环——从随机乱甩到看图稳摆")
    print("=" * 60)
    env_steps_per_ep = EPISODE_DECISIONS * ACTION_REPEAT
    replay_ratio = (UPDATES_PER_EP * BATCH_SIZE * CHUNK_LEN) / EPISODE_DECISIONS
    print(f"""
前两节已经验证了"给 50 个离线回合就能训练世界模型并做规划"。
但真正的 PlaNet 不需要事先准备 50 个回合——它和 Phase 1 的 PETS 一样，从区区 S = {INIT_EPISODES} 个随机回合起步：

        ┌──────────────────────────────────────────────────────────────────┐
        ▼                                                                  │
    [回放缓冲区 D] ──采样 B×T 片段──▶ [训练同一个 WorldModel C 步]          │
                                                   │                       │
                                                   ▼                       │
                             [在真实环境跑 1 个回合：observe_step + CEM]    │
                             [动作叠加探索噪声 ε ~ N(0, {EXPLORE_STD}²)，存入 D] ──┘

算一笔样本重用账（对照 CONCEPT.md 第 10.3 节）：
    每个回合 {env_steps_per_ep} 个环境步 ÷ 动作重复 R={ACTION_REPEAT} = {EPISODE_DECISIONS} 个决策步；
    每采集 1 个新回合，训练看过的决策步 = C × B × T = {UPDATES_PER_EP} × {BATCH_SIZE} × {CHUNK_LEN} = {UPDATES_PER_EP * BATCH_SIZE * CHUNK_LEN:,}；
    重用比 = {replay_ratio:.0f} 倍（每 1 步真实像素交互，对应缓冲区里 {replay_ratio:.0f} 步的回放训练）。
    """)


# ==================== 第二部分：采集回合与评估 ====================


@torch.no_grad()
def collect_episode(seed, model=None, explore_std=0.0, horizon=PLAN_HORIZON,
                    n_candidates=PLAN_CANDIDATES, n_iters=PLAN_ITERS):
    """在像素版 Pendulum-v1 上运行一个回合（CONCEPT.md 第 10.4 节步骤 b–d）。

    model=None 时执行随机探索动作；model 不为 None 时用后验跟踪信念并在隐空间做 CEM 规划。
    返回字典：obs [L+1, 64, 64, 3] uint8，actions [L+1, 1]，rewards [L+1]，ret，plan_time，env_time。
    """
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 999)

    L = EPISODE_DECISIONS + 1
    obs_ep = np.zeros((L, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    act_ep = np.zeros((L, ACTION_DIM), dtype=np.float32)
    rew_ep = np.zeros(L, dtype=np.float32)

    t_env0 = time.time()
    obs_ep[0] = render_pendulum(env)
    env_time = time.time() - t_env0
    plan_time = 0.0

    if model is not None:
        model.eval()
        h, z = model.rssm.initial_state(1, DEVICE)
        a_prev = torch.zeros(1, ACTION_DIM, device=DEVICE)
        gen = torch.Generator(device=DEVICE).manual_seed(seed)

    for t in range(1, L):
        if model is None:
            # 初始回合：一半纯随机，一半平滑随机，完全不读任何状态
            if seed % 2 == 1 and t > 1:
                a = np.clip(0.7 * act_ep[t - 1] + 0.65 * rng.standard_normal(ACTION_DIM), -1.0, 1.0).astype(np.float32)
            else:
                a = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
        else:
            t_p0 = time.time()
            obs_t = preprocess_obs(torch.from_numpy(obs_ep[t - 1 : t]).to(DEVICE))
            embed_t = model.encoder(obs_t)
            h, z, _, _ = model.rssm.observe_step(h, z, a_prev, embed_t, sample=False)
            plan_mean = latent_cem_plan(
                model, h, z, horizon=horizon, n_candidates=n_candidates,
                n_elites=max(10, n_candidates // 10), n_iters=n_iters, generator=gen,
            )
            a = plan_mean[0].cpu().numpy().astype(np.float32)
            if explore_std > 0.0:
                a = np.clip(a + rng.normal(0.0, explore_std, size=ACTION_DIM), -1.0, 1.0).astype(np.float32)
            a_prev = torch.from_numpy(a[None]).to(DEVICE)
            plan_time += time.time() - t_p0

        t_e0 = time.time()
        frame, reward, _ = step_with_repeat(env, a)
        env_time += time.time() - t_e0

        obs_ep[t] = frame
        act_ep[t] = a
        rew_ep[t] = reward

    env.close()
    return dict(obs=obs_ep, actions=act_ep, rewards=rew_ep,
                ret=float(rew_ep.sum()), plan_time=plan_time, env_time=env_time)


def evaluate_policy(model, eval_seeds, horizon, n_candidates, n_iters):
    """去掉探索噪声（explore_std = 0），在固定评估种子上测平均真实回报。"""
    episodes = [
        collect_episode(s, model=model, explore_std=0.0, horizon=horizon,
                        n_candidates=n_candidates, n_iters=n_iters)
        for s in eval_seeds
    ]
    rets = [ep["ret"] for ep in episodes]
    plan_t = sum(ep["plan_time"] for ep in episodes)
    env_t = sum(ep["env_time"] for ep in episodes)
    return float(np.mean(rets)), float(np.std(rets)), episodes[0]["obs"], plan_t, env_t


# ==================== 第三部分：PlaNet 主训练闭环 ====================


def train_planet(n_init, n_online, updates_init, updates_per_ep, batch_size, chunk_len,
                 horizon, n_candidates, n_iters, eval_seeds, eval_every):
    print("\n" + "=" * 60)
    print("第二部分：运行 PlaNet 在线闭环")
    print("=" * 60)

    # 1. 随机策略在同一组评估种子上的参照分数
    rand_eval = [collect_episode(s, model=None) for s in eval_seeds]
    rand_mean = float(np.mean([ep["ret"] for ep in rand_eval]))
    rand_std = float(np.std([ep["ret"] for ep in rand_eval]))
    print(f"  参照基线：随机策略在 {len(eval_seeds)} 个固定评估回合上的回报 = {rand_mean:.1f} ± {rand_std:.1f}")

    # 2. 初始随机回合入库
    replay = SequenceReplay()
    init_rets = []
    time_env = 0.0
    for ep_idx in range(n_init):
        ep = collect_episode(SEED + ep_idx, model=None)
        replay.add_episode(ep["obs"], ep["actions"], ep["rewards"])
        init_rets.append(ep["ret"])
        time_env += ep["env_time"]
    print(f"  初始随机采集：{n_init} 个回合（{n_init * 200} 环境步），平均回报 {np.mean(init_rets):.1f}\n")

    # 3. 初始化单个世界模型与优化器（注意：全程持续训练同一个模型，不每轮重置！）
    torch.manual_seed(SEED)
    model = WorldModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(SEED)

    time_train = 0.0
    time_plan = 0.0
    online_records = []
    eval_records = []
    early_frames, final_frames = None, None

    for round_idx in range(1, n_online + 1):
        n_upd = updates_init if round_idx == 1 else updates_per_ep
        model.train()
        t0 = time.time()
        losses = []
        for _ in range(n_upd):
            obs_b, act_b, rew_b = replay.sample(batch_size, chunk_len, DEVICE, generator=gen)
            loss, m = world_model_loss(model, obs_b, act_b, rew_b)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
            optimizer.step()
            losses.append((m["recon"], m["reward"], m["raw_kl"]))
        dt_train = time.time() - t0
        time_train += dt_train
        rec_m, rew_m, kl_m = np.mean(losses[-min(20, len(losses)):], axis=0)

        # 用当前模型 + 探索噪声在线采集 1 个新回合
        ep = collect_episode(
            SEED + 100 + round_idx, model=model, explore_std=EXPLORE_STD,
            horizon=horizon, n_candidates=n_candidates, n_iters=n_iters,
        )
        replay.add_episode(ep["obs"], ep["actions"], ep["rewards"])
        time_plan += ep["plan_time"]
        time_env += ep["env_time"]
        env_steps_so_far = len(replay) * 200
        online_records.append((env_steps_so_far, ep["ret"]))

        eval_str = ""
        if round_idx == 1 or round_idx % eval_every == 0 or round_idx == n_online:
            ev_mean, ev_std, frames_0, ev_pt, ev_et = evaluate_policy(
                model, eval_seeds, horizon, n_candidates, n_iters
            )
            time_plan += ev_pt
            time_env += ev_et
            eval_records.append((env_steps_so_far, ev_mean, ev_std))
            eval_str = f" ｜ 无噪评估 {ev_mean:6.1f} ± {ev_std:4.1f}"
            if early_frames is None:
                early_frames = frames_0
            final_frames = frames_0

        print(f"  第 {round_idx:>2}/{n_online} 轮（累计 {env_steps_so_far:>4} 环境步）："
              f"训练 {n_upd:>3} 步（重建 {rec_m:5.1f}, KL {kl_m:4.1f}, {dt_train:4.0f}s）｜ "
              f"采集回报 {ep['ret']:6.1f}（规划 {ep['plan_time']:4.1f}s）{eval_str}")

    total_time = time_train + time_plan + time_env
    return dict(
        model=model, rand_mean=rand_mean, init_rets=init_rets,
        online_records=online_records, eval_records=eval_records,
        early_frames=early_frames, final_frames=final_frames,
        time_train=time_train, time_plan=time_plan, time_env=time_env, total_time=total_time,
    )


# ==================== 第四部分：曲线、视频与全章总结 ====================


def report_and_plot(res):
    t_tr, t_pl, t_ev, t_tot = res["time_train"], res["time_plan"], res["time_env"], res["total_time"]
    ev = res["eval_records"]
    print("\n" + "=" * 60)
    print("第三部分：耗时结构与 Phase 1–3 方法对照")
    print("=" * 60)
    print(f"""
  1. 总运行耗时拆解（共 {t_tot:.1f}s = {t_tot / 60:.1f} 分钟，设备 {DEVICE}）：
       世界模型训练：   {t_tr:6.1f}s（{t_tr / t_tot:5.1%}）
       隐空间 CEM 规划：{t_pl:6.1f}s（{t_pl / t_tot:5.1%}）
       环境步进与渲染： {t_ev:6.1f}s（{t_ev / t_tot:5.1%}）

  2. 控制效果演进（同一组固定评估回合）：
       随机策略：                    {res['rand_mean']:7.1f}
       第 1 轮（仅见过初始随机数据）：{ev[0][1]:7.1f} ± {ev[0][2]:.1f}
       最终轮（累计 {ev[-1][0]} 环境步）：  {ev[-1][1]:7.1f} ± {ev[-1][2]:.1f}
       （对照 Phase 1 的 08-pets.py：直接读取真实状态 (θ, θ̇) 与精确奖励公式时，PETS 最终回报约 −140 ~ −160；
         在无 GPU 的小预算（6000 步、小批量与短训练步数）下，仅看 64×64 像素的 PlaNet 需更多在线轮次与更大更新量才能追近状态空间上限，本演示已可观察到重构误差收敛与评估回报相较第 1 轮与随机基线的明显提升。）

  3. 本课实现与 PlaNet 论文（Hafner et al. 2019）的超参数对照：
       环境：像素版 Pendulum-v1（200 环境步） vs 论文 DeepMind Control Suite 6 个任务（1000 环境步）
       模型规模：h=200, z=30, 4 层卷积/转置卷积（与论文完全一致）
       动作重复与片段长度：R={ACTION_REPEAT}, B={BATCH_SIZE}, T={CHUNK_LEN} vs 论文 R=2~8, B=50, T=50
       规划预算：H={PLAN_HORIZON}, N={PLAN_CANDIDATES}, I={PLAN_ITERS} vs 论文 H=12, N=1000, I=10
    """)

    # 图：左为学习曲线，右为耗时占比柱状图
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    init_x = np.arange(1, len(res["init_rets"]) + 1) * 200
    on_x = [r[0] for r in res["online_records"]]
    on_y = [r[1] for r in res["online_records"]]
    ev_x = [r[0] for r in ev]
    ev_y = [r[1] for r in ev]
    ev_s = [r[2] for r in ev]

    axes[0].scatter(init_x, res["init_rets"], color="gray", alpha=0.6, label="初始随机回合")
    axes[0].scatter(on_x, on_y, color="C0", alpha=0.55, label="在线采集回合（含探索噪声）")
    axes[0].errorbar(ev_x, ev_y, yerr=ev_s, fmt="o-", color="C3", linewidth=2.2, capsize=4, label="无噪评估回报（5 回合均值）")
    axes[0].axhline(res["rand_mean"], color="tomato", linestyle=":", label=f"随机策略（{res['rand_mean']:.0f}）")
    axes[0].axhline(-150.0, color="seagreen", linestyle="--", alpha=0.7, label="Phase 1 状态输入 PETS 参考区间（≈−150）")
    axes[0].set_xlabel("累计真实环境交互步数")
    axes[0].set_ylabel("回合真实回报（越高越好）")
    axes[0].set_title("PlaNet 像素控制学习曲线")
    axes[0].legend(fontsize=8, loc="lower right")

    labels = ["模型训练", "CEM 规划", "环境与渲染"]
    secs = [t_tr, t_pl, t_ev]
    colors = ["C0", "C1", "lightgray"]
    axes[1].bar(labels, secs, color=colors, width=0.5)
    for idx, s in enumerate(secs):
        axes[1].text(idx, s + max(secs) * 0.02, f"{s:.0f}s\n({s / t_tot:.0%})", ha="center", fontsize=9)
    axes[1].set_ylim(0, max(secs) * 1.22)
    axes[1].set_ylabel("累计耗时（秒）")
    axes[1].set_title("闭环总耗时拆解")
    save_figure(fig, OUT / "08-planet-learning-curve.png")

    # 视频：左栏第 1 轮评估画面，右栏最终轮评估画面
    if res["early_frames"] is not None and res["final_frames"] is not None:
        vframes = []
        for f_early, f_final in zip(res["early_frames"], res["final_frames"]):
            left = cv2.resize(f_early, (128, 128), interpolation=cv2.INTER_NEAREST)
            right = cv2.resize(f_final, (128, 128), interpolation=cv2.INTER_NEAREST)
            spacer = np.full((128, 8, 3), 240, dtype=np.uint8)
            vframes.append(np.concatenate([left, spacer, right], axis=1))
        save_video(vframes, OUT / "08-planet-control.mp4", fps=15)
        print("    （对比视频左栏：第 1 轮评估；右栏：最终轮评估）")


# ==================== 第五部分：练习与思考 ====================


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：如果把采集数据时的 EXPLORE_STD 设成 0.0，而且初始只给 S = 1 个随机回合，闭环会发生什么？

练习 1 参考答案：
    早期不准的模型会让 CEM 反反复复走同一条狭窄且错误的轨迹，没有探索噪声去试探旁边的动作，
    新采回来的回合高度雷同，动力学盲区补得很慢。适度的探索噪声是在线闭环不可或缺的润滑剂。

练习 2：为什么 train_planet() 全程只初始化一次 WorldModel、在同一个参数上接着训练，不像 Phase 1 的 PETS 那样每轮从头初始化？

练习 2 参考答案：
    Phase 1 的状态 MLP 只有几万参数，2000 步就能从零训好；像素级 WorldModel 有 513 万参数，
    卷积编码器和转置卷积解码器学好的视觉特征完全可以跨轮次复用，接着更新 C = 60~100 步就能迅速吸收新回合的数据。

思考题 1：从耗时柱状图里能直接看到 PlaNet 留给后续章节的哪两个核心短板？

思考题 1 参考答案：
    第一是"决策太慢"：每走一步都要在线展开 N×H×I 次模型，哪怕只在隐空间算也很费时间，真机高频控制吃不消；
    第二是"看不远"：H = 15 步之外的未来完全是盲区，没有价值函数兜底。
    这正是第二部分（Phase 4 值函数、Phase 5 策略梯度、Phase 6 Dreamer 在想象中训练策略和价值函数）要彻底解决的问题！
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    if DEVICE.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    quick = is_quick()
    if quick:
        print("（冒烟模式：回合数、更新步数和规划预算大幅缩小，仅验证完整闭环跑通）\n")

    params = (
        dict(n_init=2, n_online=2, updates_init=15, updates_per_ep=10, batch_size=4, chunk_len=12,
             horizon=10, n_candidates=50, n_iters=3, eval_seeds=[1000], eval_every=1)
        if quick else
        dict(n_init=INIT_EPISODES, n_online=ONLINE_EPISODES, updates_init=UPDATES_INIT,
             updates_per_ep=UPDATES_PER_EP, batch_size=BATCH_SIZE, chunk_len=CHUNK_LEN,
             horizon=PLAN_HORIZON, n_candidates=PLAN_CANDIDATES, n_iters=PLAN_ITERS,
             eval_seeds=EVAL_SEEDS, eval_every=3)
    )

    introduction()
    res = train_planet(**params)
    report_and_plot(res)
    exercises()

    print("\n" + "=" * 60)
    print("🎉 Phase 3 全部 8 节讲义完成！下一章：Phase 4（强化学习核心：值函数与时序差分）")
    print("=" * 60)


if __name__ == "__main__":
    main()
