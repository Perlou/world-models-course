"""
DQN：把 Q-learning 接上神经网络
===============================

学习目标：
    1. 说清表格 Q-learning 换成神经网络后为什么不稳定：样本高度相关、目标跟着网络一起跑
    2. 从零实现 DQN 的三个部件：回放缓冲区 ReplayBuffer、Q 网络 QNetwork、损失 dqn_loss()
    3. 认出 dqn_loss() 里三个最容易写错的地方：gather、目标不带梯度、形状广播
    4. 在 CartPole-v1 上把 DQN 训练起来，并通过两组消融看经验回放和目标网络各自起什么作用
    5. 正确处理 terminated 与 truncated：回放里只存 terminated

核心概念：
    - 深度 Q 网络（Deep Q-Network，DQN）
    - 经验回放（experience replay）：把转移存起来，训练时随机抽，打破样本之间的相关性
    - 目标网络（target network）：在线网络的一个滞后副本，专门用来算目标
    - 半梯度（semi-gradient）与停止梯度：目标当作常数，不对它求导
    - Huber 损失：误差小时是平方，误差大时是线性

前置知识：
    - 05-q-learning.py：Q-learning 的目标、ε-greedy、离策略
    - deep-learning-course：MLP、训练循环、Adam

对应概念文档：
    - CONCEPT.md 第 8 节「DQN：把 Q-learning 接上神经网络」

预计运行时间：尚未实测（本文件是还没有运行过的草稿，定稿前要完整跑一遍并改写这一行）
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-04")
SEED = 0
# 网络很小（两层 128 单元的 MLP），而且每个环境步都要单独前向一次选动作，CPU 比 MPS 快。
# 只用一个线程：小网络多线程没有收益，也不去抢这台机器上别的训练的 CPU。
DEVICE = torch.device("cpu")
torch.set_num_threads(1)

CONFIG = dict(
    gamma=0.99,
    hidden=128,
    lr=5e-4,
    batch_size=64,
    buffer_size=20000,
    learning_starts=1000,  # 先用随机动作攒这么多步数据再开始训练
    target_sync=500,  # 每隔多少个环境步把在线网络的参数拷给目标网络（硬更新）
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_steps=8000,  # ε 从 eps_start 线性降到 eps_end 用多少步
    grad_clip=10.0,
    eval_interval=2000,  # 每隔多少步，关掉探索评估一次
    eval_episodes=5,
)


# ==================== 第一部分：为什么不能直接把表格换成网络 ====================


def introduction():
    """从表格到网络：好处、麻烦、两个稳定器。"""
    print("=" * 60)
    print("第一部分：把查表换成函数调用")
    print("=" * 60)

    print("""
上一节的 Q 表给每个 (s, a) 存一个数。CartPole-v1 的状态是 4 个连续的实数，表格放不下；
更根本的问题是表格不会举一反三：两个几乎一样的状态，在表里是两个不相干的格子。

办法是用一个神经网络 Q_ψ 近似 Q。输入状态，一次前向输出每个动作的价值：

                    ┌──────────────┐      ┌─▶ Q(s, 向左推)
     状态 s  ──────▶ │  Q 网络 Q_ψ   │ ─────┤
     [B, 4]         └──────────────┘      └─▶ Q(s, 向右推)          输出 [B, 2]

网络带来了泛化，也带来了两个麻烦：

    麻烦一：样本高度相关。智能体一步一步交互，相邻的转移几乎一样；
            连续喂进去，网络会过度拟合最近这一小段经历，把之前学的冲掉。
    麻烦二：目标在跟着跑。目标 y = r + γ·max Q_ψ(s', ·) 是网络自己算的：

            更新 ψ，让 Q_ψ(s, a) 向 y 靠近
                 │
                 ▼
            ψ 变了 ──▶ Q_ψ(s', ·) 也变了（s 和 s' 往往很像）──▶ y 也变了
                                                              刚才追的靶子已经挪了位置

DQN（Mnih et al. 2015）的贡献是两个稳定器：

    经验回放：  转移 (s, a, r, s', d) ──▶ [ 回放缓冲区，环形队列 ] ──均匀随机抽一批──▶ 梯度更新
                小批量里的样本来自不同的时间、不同的回合，近似独立；每个转移还能被反复利用
    目标网络：  另存一份参数 ψ⁻ 专门算目标，每隔 C 步才从 ψ 拷贝一次
                两次同步之间目标是固定的，每一段都是一个普通的监督回归问题

损失函数：
    L(ψ) = E_{(s,a,r,s',d)~D} [ Huber( y − Q_ψ(s, a) ) ]，    y = sg( r + γ·(1 − d)·max_a' Q_ψ⁻(s', a') )
    """)


# ==================== 第二部分：三个部件 ====================


class ReplayBuffer:
    """回放缓冲区：预分配的 NumPy 数组加一个写指针，写满后从头覆盖最旧的数据。"""

    def __init__(self, capacity, obs_dim):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.float32)
        self.capacity, self.pos, self.size = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, terminated):
        """存一个转移。terminated 只在环境真的终止时为 1；被步数上限掐断（truncated）时存 0。"""
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.terminated[i] = next_obs, float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _batch(self, idx):
        return {
            "obs": torch.from_numpy(self.obs[idx]),  # [B, obs_dim] float32
            "actions": torch.from_numpy(self.actions[idx]),  # [B] int64
            "rewards": torch.from_numpy(self.rewards[idx]),  # [B] float32
            "next_obs": torch.from_numpy(self.next_obs[idx]),  # [B, obs_dim]
            "terminated": torch.from_numpy(self.terminated[idx]),  # [B] float32，0 或 1
        }

    def sample(self, batch_size, rng):
        """均匀随机抽一个小批量。"""
        return self._batch(rng.integers(self.size, size=batch_size))

    def latest(self):
        """只取最新存进来的那一个转移（消融"去掉经验回放"时用）。"""
        return self._batch(np.array([(self.pos - 1) % self.capacity]))


class QNetwork(nn.Module):
    """Q 网络：两个隐藏层的 MLP。输入状态 [B, obs_dim]，输出每个动作的价值 [B, A]。"""

    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)


def dqn_loss(q_net, target_net, batch, gamma, huber=True):
    """DQN 的损失（CONCEPT.md 第 8.2、8.7 节）。三个最容易写错的地方标了 ①②③。

    返回：
        loss:   标量
        q:      [B]，实际执行的动作的 Q（不带梯度，诊断用）
        target: [B]，回归目标
    """
    # ① 用 gather 取出"实际执行的那个动作"的 Q。
    #    写成 q_net(obs).max(dim=1) 是错的：那是当前认为最好的动作的 Q，不是数据里做的那个动作的。
    q = q_net(batch["obs"]).gather(1, batch["actions"][:, None]).squeeze(1)  # [B, A] → [B, 1] → [B]

    # ② 目标不带梯度（半梯度：老师是目标，学生是 q，不能让老师向学生靠拢），并且用目标网络算。
    with torch.no_grad():
        next_q = target_net(batch["next_obs"]).max(dim=1).values  # [B]
        target = batch["rewards"] + gamma * (1.0 - batch["terminated"]) * next_q  # [B]，终止之后不自举

    # ③ q 和 target 必须都是 [B]。一个是 [B]、另一个是 [B, 1] 时相减会被广播成 [B, B]，
    #    不报错，损失照样能算，只是完全不对。
    assert q.shape == target.shape, f"形状不一致：{tuple(q.shape)} vs {tuple(target.shape)}"

    loss = F.huber_loss(q, target, delta=1.0) if huber else F.mse_loss(q, target)
    return loss, q.detach(), target


def loss_by_hand():
    """复现 CONCEPT.md 第 8.5 节的手算例子，并演示形状广播的坑。"""
    print("\n" + "=" * 60)
    print("第二部分：先用手算的例子核对 dqn_loss()")
    print("=" * 60)

    gamma = 0.99
    batch = {
        "obs": torch.zeros(2, 4),  # 用假网络，状态的内容无所谓
        "actions": torch.tensor([0, 1]),
        "rewards": torch.tensor([1.0, 1.0]),
        "next_obs": torch.zeros(2, 4),
        "terminated": torch.tensor([0.0, 1.0]),  # 转移 2 是终止的
    }
    fake_q_net = lambda obs: torch.tensor([[2.5, -9.0], [-9.0, 4.0]])  # 实际执行的动作的 Q 是 2.5 和 4.0
    fake_target_net = lambda obs: torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    mse, q, target = dqn_loss(fake_q_net, fake_target_net, batch, gamma, huber=False)
    huber, _, _ = dqn_loss(fake_q_net, fake_target_net, batch, gamma, huber=True)
    assert torch.allclose(q, torch.tensor([2.5, 4.0])) and torch.allclose(target, torch.tensor([3.97, 1.0]))
    assert abs(mse.item() - 5.58045) < 1e-4 and abs(huber.item() - 1.735) < 1e-4

    wrong_target = 1.0 + gamma * 5.0  # 如果忘了乘 (1 − d)
    broadcast = (q[:, None] - target).shape  # 把 q 误写成 [B, 1] 时的结果
    print(f"""
一个大小为 2 的小批量，γ = {gamma}（CONCEPT.md 第 8.5 节）：

              r    d      目标网络 Q(s', ·)     实际动作的 Q(s, a)     目标 y            误差 y − Q
    转移 1    1    0      [2.0, 3.0]           {q[0]:.1f}                  {target[0]:.2f}             {target[0] - q[0]:+.2f}
    转移 2    1    1      [4.0, 5.0]           {q[1]:.1f}                  {target[1]:.2f}             {target[1] - q[1]:+.2f}

    y_1 = 1 + 0.99 × max(2, 3) = 3.97
    y_2 = 1（终止：目标网络说下一状态值 5.0 也不算数）

    均方误差   = (1.47² + 3.0²) / 2 = {mse.item():.3f}
    Huber 损失 = [(1.47 − 0.5) + (3.0 − 0.5)] / 2 = {huber.item():.3f}

  和手算一致。两点提醒：
    - 如果转移 2 忘了乘 (1 − d)，目标会是 1 + 0.99 × 5 = {wrong_target:.2f} 而不是 1。不报错，只是价值系统性地不对。
    - 如果 q 的形状写成了 [2, 1] 而 target 是 [2]，相减得到的形状是 {tuple(broadcast)}，损失照样是个标量。
      dqn_loss() 里的 assert 就是防这个的。
  Huber 损失的梯度大小不超过 1：转移 2 的误差是 -3，平方损失的梯度大小是 3，Huber 是 1。
  一个离谱的目标不会产生一个巨大的梯度。
    """)


# ==================== 第三部分：训练循环 ====================


def evaluate(q_net, env, n_episodes, seed):
    """关掉探索（ε = 0），按贪心策略跑几个回合，返回平均回报。"""
    total = 0.0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            with torch.no_grad():
                action = int(q_net(torch.as_tensor(obs)[None]).argmax(dim=1).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
    return total / n_episodes


def train_dqn(seed, total_steps, use_target_net=True, use_replay=True, cfg=CONFIG):
    """在 CartPole-v1 上训练 DQN（CONCEPT.md 第 8.6 节的算法）。

    两个开关用来做消融：
        use_target_net=False  目标直接用在线网络算
        use_replay=False      不从缓冲区里随机抽，每一步只用最新的那一个转移更新
    返回一个字典：训练期间每个回合的回报和结束时的步数、定期评估的结果、Q 值的量级、用时。
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env, eval_env = gym.make("CartPole-v1"), gym.make("CartPole-v1")
    obs_dim, n_actions = env.observation_space.shape[0], int(env.action_space.n)
    q_net = QNetwork(obs_dim, n_actions, cfg["hidden"]).to(DEVICE)
    target_net = QNetwork(obs_dim, n_actions, cfg["hidden"]).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())  # ψ⁻ ← ψ
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg["lr"])
    buffer = ReplayBuffer(cfg["buffer_size"], obs_dim)

    log = {"episode_returns": [], "episode_ends": [], "eval_steps": [], "eval_returns": [], "q_mean": []}
    recent_q = []
    start = time.time()
    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    for step in range(1, total_steps + 1):
        # ---- 与环境交互：ε-greedy，ε 线性衰减 ----
        frac = min(step / cfg["eps_decay_steps"], 1.0)
        epsilon = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])
        if rng.random() < epsilon:
            action = int(rng.integers(n_actions))
        else:
            with torch.no_grad():
                action = int(q_net(torch.as_tensor(obs)[None]).argmax(dim=1).item())
        next_obs, reward, terminated, truncated, _ = env.step(action)
        # 只存 terminated。被 500 步上限掐断时 next_obs 是最后一个真实观测，照常用来自举
        buffer.add(obs, action, reward, next_obs, terminated)
        ep_return += reward
        obs = next_obs
        if terminated or truncated:
            log["episode_returns"].append(ep_return)
            log["episode_ends"].append(step)
            ep_return = 0.0
            obs, _ = env.reset()

        # ---- 更新网络 ----
        if step >= cfg["learning_starts"]:
            batch = buffer.sample(cfg["batch_size"], rng) if use_replay else buffer.latest()
            loss, q, _ = dqn_loss(q_net, target_net if use_target_net else q_net, batch, cfg["gamma"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), cfg["grad_clip"])
            optimizer.step()
            recent_q.append(q.mean().item())
            if use_target_net and step % cfg["target_sync"] == 0:
                target_net.load_state_dict(q_net.state_dict())  # 硬更新

        # ---- 定期评估 ----
        if step % cfg["eval_interval"] == 0:
            log["eval_steps"].append(step)
            log["eval_returns"].append(evaluate(q_net, eval_env, cfg["eval_episodes"], seed=10000 + seed))
            log["q_mean"].append(float(np.mean(recent_q)) if recent_q else 0.0)  # 这一段里小批量 Q 的平均
            recent_q = []
    env.close()
    eval_env.close()
    log["seconds"] = time.time() - start
    return log


# ==================== 第四部分：实验 ====================


def windowed_training_returns(log, total_steps, window):
    """把训练期间的回合回报按"结束时的环境步数"分段取平均。返回 [total_steps // window]，没有回合结束的段是 NaN。"""
    ends, rets = np.array(log["episode_ends"]), np.array(log["episode_returns"])
    out = np.full(total_steps // window, np.nan)
    for i in range(len(out)):
        mask = (ends > i * window) & (ends <= (i + 1) * window)
        if mask.any():
            out[i] = rets[mask].mean()
    return out


def run_experiments(total_steps, seeds):
    """DQN 本体，加两组消融。"""
    print("\n" + "=" * 60)
    print("第三部分：在 CartPole-v1 上训练")
    print("=" * 60)

    cfg = CONFIG
    print(f"""
CartPole-v1：小车上立着一根杆，向左或向右推小车，让杆不倒。
    观测 4 维（小车位置、小车速度、杆的角度、杆的角速度），2 个离散动作
    杆没倒的每一步奖励 +1；杆倾斜过大或小车出界时 terminated；撑满 500 步时 truncated
    所以一个回合的回报就是撑了多少步，最高 500

本课的设置（括号里是 DQN 论文在 Atari 上的设置，量级差很多，因为任务小得多）：
    网络        两个隐藏层的 MLP，各 {cfg['hidden']} 个单元（论文：3 个卷积层 + 512 单元的全连接层）
    回放容量    {cfg['buffer_size']} 个转移（论文：100 万）
    批大小      {cfg['batch_size']}（论文：32）
    目标网络    每 {cfg['target_sync']} 步硬更新一次（论文：每 1 万次参数更新）
    探索        ε 从 {cfg['eps_start']} 线性降到 {cfg['eps_end']}，用时 {cfg['eps_decay_steps']} 步（论文：1.0 → 0.1，100 万帧）
    预热        先收集 {cfg['learning_starts']} 步数据再训练（论文：5 万帧）
    优化器      Adam，学习率 {cfg['lr']}（论文：RMSProp，学习率 0.00025）
    折扣 γ      {cfg['gamma']}（论文相同）；损失是 Huber（论文叫误差截断，是一回事）

每个设置训练 {total_steps} 个环境步，{len(seeds)} 个随机种子；每 {cfg['eval_interval']} 步关掉探索评估 {cfg['eval_episodes']} 个回合。
    """)

    variants = {
        "DQN": dict(use_target_net=True, use_replay=True),
        "去掉目标网络": dict(use_target_net=False, use_replay=True),
        "去掉经验回放": dict(use_target_net=True, use_replay=False),
    }
    logs = {name: [] for name in variants}
    for name, flags in variants.items():
        for seed in seeds:
            log = train_dqn(seed, total_steps, **flags)
            logs[name].append(log)
            print(f"  [{name}，种子 {seed}] 训练了 {len(log['episode_returns'])} 个回合，用时 {log['seconds']:.0f} 秒，"
                  f"最后一次评估的平均回报 {log['eval_returns'][-1]:.0f}")

    eval_steps = np.array(logs["DQN"][0]["eval_steps"])
    evals = {name: np.array([log["eval_returns"] for log in runs]) for name, runs in logs.items()}  # [种子, 评估次数]
    q_means = {name: np.array([log["q_mean"] for log in runs]) for name, runs in logs.items()}

    # ---- DQN 的学习曲线 ----
    window = max(total_steps // 6, 1)
    train_curve = np.array([windowed_training_returns(log, total_steps, window) for log in logs["DQN"]])
    print(f"\n  DQN 的学习曲线（{len(seeds)} 个种子的平均）：\n")
    print(f"  {'环境步数':<16}{'训练期间的回合回报（带探索）':>24}{'评估回报（不带探索）':>20}")
    for i in range(train_curve.shape[1]):
        lo, hi = i * window, (i + 1) * window
        in_window = (eval_steps > lo) & (eval_steps <= hi)
        eval_text = f"{evals['DQN'][:, in_window].mean():.0f}" if in_window.any() else "—"
        train_text = f"{np.nanmean(train_curve[:, i]):.0f}" if not np.all(np.isnan(train_curve[:, i])) else "—"
        print(f"  {f'{lo + 1}–{hi}':<18}{train_text:>24}{eval_text:>20}")
    dqn_seconds = np.mean([log["seconds"] for log in logs["DQN"]])
    first = evals["DQN"].mean(axis=0)[0]
    best = evals["DQN"].mean(axis=0).max()
    print(f"\n  平均每次训练用时 {dqn_seconds:.0f} 秒（CPU，单线程）。")
    print(f"  评估回报从第一次评估（{eval_steps[0]} 步）的 {first:.0f} 升到最高 {best:.0f}（各种子平均；满分 500）。")
    reached = [int(eval_steps[np.argmax(run >= 475)]) if (run >= 475).any() else None for run in evals["DQN"]]
    if all(r is not None for r in reached):
        print(f"  每个种子都至少有一次评估达到 475 以上，最早出现在第 {'、'.join(str(r) for r in reached)} 步。")
    else:
        n_ok = sum(r is not None for r in reached)
        print(f"  {len(seeds)} 个种子里有 {n_ok} 个至少有一次评估达到 475 以上。")
    drops = (np.diff(evals["DQN"], axis=1) < -100).sum()
    if drops > 0:
        print(f"  学习曲线不是单调的：相邻两次评估之间回报下跌超过 100 的情况出现了 {drops} 次。")
        print("  这是 DQN 的常态：策略变好之后访问到新的状态，那里的 Q 还没学准，表现会突然掉下去再爬回来。")

    # ---- 消融 ----
    tail = max(len(eval_steps) // 3, 1)  # 用最后三分之一的评估来比较
    print(f"\n  消融对比（最后 {tail} 次评估，{len(seeds)} 个种子合在一起统计）：\n")
    print(f"  {'设置':<14}{'平均回报':>10}{'最低':>8}{'最高':>8}{'标准差':>10}{'小批量 Q 的平均（最后一段）':>26}")
    summary = {}
    for name in variants:
        late = evals[name][:, -tail:]
        summary[name] = (late.mean(), late.min(), late.max(), late.std(), q_means[name][:, -1].mean())
        print(f"  {name:<14}{late.mean():>10.0f}{late.min():>8.0f}{late.max():>8.0f}{late.std():>10.0f}{summary[name][4]:>26.1f}")

    q_bound = 1.0 / (1.0 - cfg["gamma"])
    print(f"\n  Q 值的上界是 r_max / (1 − γ) = {q_bound:.0f}（每步奖励 1）。超过它说明价值在发散。")
    base = summary["DQN"][0]
    for name in ("去掉目标网络", "去掉经验回放"):
        mean, low, _, std, q_mean = summary[name]
        if mean < 0.8 * base:
            verdict = f"平均回报 {mean:.0f}，明显低于完整 DQN 的 {base:.0f}"
        elif std > 1.5 * summary["DQN"][3] or low < 0.5 * summary["DQN"][1]:
            verdict = f"平均回报 {mean:.0f} 与完整 DQN 的 {base:.0f} 接近，但波动更大（标准差 {std:.0f} 对 {summary['DQN'][3]:.0f}，最低 {low:.0f}）"
        else:
            verdict = f"平均回报 {mean:.0f}、标准差 {std:.0f}，在这个任务和这组超参数下没有表现出明显劣势"
        if q_mean > q_bound:
            verdict += f"；Q 的平均值 {q_mean:.0f} 超过了上界"
        print(f"  {name}：{verdict}。")
    print("""
  怎么读这张表：CartPole 是一个很小的任务，消融的差别不一定每次都很大；
  但两个稳定器针对的问题是真实的——去掉回放，每一步只用最新的一个转移，样本前后高度相关；
  去掉目标网络，目标每一步都在跟着在线网络变。任务越大、训练越久，这两件事越要命。
    """)

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    colors = {"DQN": "tab:blue", "去掉目标网络": "tab:red", "去掉经验回放": "tab:green"}
    for name in variants:
        mean = evals[name].mean(axis=0)
        axes[0].plot(eval_steps, mean, color=colors[name], linewidth=2, label=name)
        axes[0].fill_between(eval_steps, evals[name].min(axis=0), evals[name].max(axis=0), color=colors[name], alpha=0.15)
    axes[0].axhline(500, color="gray", linestyle=":", label="满分 500")
    axes[0].set_xlabel("环境步数")
    axes[0].set_ylabel("评估回报（不带探索）")
    axes[0].set_title(f"CartPole-v1：DQN 与两组消融（{len(seeds)} 个种子的平均，阴影是最低到最高）")
    axes[0].legend(fontsize=9)
    for i, log in enumerate(logs["DQN"]):
        axes[1].plot(log["episode_ends"], log["episode_returns"], ".", markersize=3, alpha=0.6, label=f"种子 {seeds[i]}")
    axes[1].set_xlabel("环境步数")
    axes[1].set_ylabel("训练期间每个回合的回报（带探索）")
    axes[1].set_title("DQN：训练期间每个回合的回报")
    axes[1].legend(fontsize=9, markerscale=4)
    save_figure(fig, OUT / "06-dqn-cartpole.png")


# ==================== 第五部分：练习与思考 ====================


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    # 练习 1：环形缓冲区
    buffer = ReplayBuffer(capacity=3, obs_dim=1)
    for i in range(5):
        buffer.add(np.array([i], dtype=np.float32), 0, float(i), np.array([i + 1], dtype=np.float32), False)
    stored = sorted(buffer.rewards.tolist())
    latest = buffer.latest()["rewards"].item()
    assert stored == [2.0, 3.0, 4.0] and latest == 4.0
    # 练习 2：Huber 损失的梯度
    errors = torch.tensor([0.5, 1.47, -3.0, 20.0], requires_grad=True)
    F.huber_loss(errors, torch.zeros(4), reduction="sum").backward()
    huber_grads = errors.grad.tolist()

    print(f"""
练习 1：环形缓冲区
    容量为 3 的 ReplayBuffer，依次存入奖励为 0、1、2、3、4 的 5 个转移。缓冲区里现在是哪几个？latest() 返回哪个？

练习 1 参考答案：
    程序算出来的：缓冲区里的奖励是 {stored}，latest() 返回奖励为 {latest:.0f} 的那个。
    写指针走到头就绕回 0，最旧的两个（0 和 1）被覆盖了。
    size 记录有效数据的个数，采样时只在 [0, size) 里抽，缓冲区没写满时不会抽到全 0 的空位。

练习 2：Huber 损失的梯度
    误差分别是 0.5、1.47、-3、20 时，Huber 损失（δ = 1）对误差的梯度各是多少？

练习 2 参考答案：
    程序算出来的：{', '.join(f'{g:+.2f}' for g in huber_grads)}。
    误差的绝对值不超过 1 时梯度等于误差本身（和平方损失一样），超过 1 之后被截到 ±1。
    误差 20 的那个样本如果用平方损失，梯度是 20，一个样本就能把网络带偏。

练习 3：把 truncated 当成 terminated
    把 train_dqn() 里 buffer.add(...) 的最后一个参数改成 terminated or truncated，重新训练，观察评估回报。

练习 3 参考答案：
    （这道题留给你自己跑，这里不给数字。）改了之后，"撑满 500 步"的最后一个转移被标成终止，目标变成 r = 1，
    而同样的状态在别的时刻的目标是 1 + γ·max Q ≈ 几十到 100。状态里没有"现在是第几步"的信息，
    网络对同一类好状态收到了互相矛盾的目标。策略越好、越常撑满 500 步，这种矛盾的样本越多。

思考题 1：经验回放为什么要求算法是离策略的？

思考题 1 参考答案：
    缓冲区里的数据是过去的策略采的，不是当前策略采的。一步的 Q-learning 目标里没有行为策略
    （上一节第三部分），所以用旧数据没问题。SARSA 和 Phase 5 的 PPO 这类在策略算法不能这样用旧数据。

思考题 2：训练时损失在下降，能说明智能体在进步吗？

思考题 2 参考答案：
    不能。DQN 的损失是相对于一个不断变化的目标算的，绝对值几乎不说明问题。
    策略变好之后访问到新的状态，损失常常反而上升。要看的是评估回报，以及 Q 值的量级是否合理
    （CartPole 里不应超过 1/(1−γ) = 100）。

思考题 3：DQN 的目标里有一个对动作的 max。动作是连续的（比如机械臂的关节力矩）时怎么办？

思考题 3 参考答案：
    连续动作没法枚举着取 max，DQN 不能直接用。办法是再学一个网络（策略）专门输出"让 Q 最大的动作"，
    这是 Phase 5 的 DDPG、SAC 的思路；Dreamer 的 actor 也是这样来的。
    另一条路是像 Phase 1 那样用 CEM 之类的规划器在线搜索动作，Phase 7 的 TD-MPC2 走的是这条路。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()
    loss_by_hand()
    if is_quick():
        CONFIG.update(learning_starts=200, eps_decay_steps=1000, eval_interval=1000, eval_episodes=2)
        run_experiments(total_steps=2000, seeds=[0])
    else:
        run_experiments(total_steps=30000, seeds=[0, 1, 2])
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：07-dqn-improvements-and-distributional.py（Double DQN 与分布式价值）")
    print("=" * 60)


if __name__ == "__main__":
    main()
