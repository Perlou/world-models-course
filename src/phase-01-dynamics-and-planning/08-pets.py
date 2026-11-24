"""
PETS：把所有部件接成闭环
========================

学习目标：
    1. 把前七节的部件组装成 PETS：概率集成 + 轨迹采样 + CEM-MPC + 迭代采集数据
    2. 实现轨迹采样：每条候选动作序列用多个粒子展开，每个粒子跟定集成里的一个成员
    3. 跑通"控制 → 采数据 → 重新训练"的闭环，看它怎样自己把模型的空子补上
    4. 用对照实验分清每个部件的贡献：闭环本身、集成、轨迹采样
    5. 知道 PETS 的边界，以及后面的章节各自在解决哪一条

核心概念：
    - PETS = Probabilistic Ensembles with Trajectory Sampling（Chua et al. 2018）
    - 粒子（particle）：一条动作序列的一种可能的未来。P 个粒子的回报取平均作为这条序列的分数
    - TS∞ / TS1：粒子全程跟定一个成员 / 每一步重新随机选成员
    - 基于模型的强化学习闭环：模型在哪里骗了控制器，控制器就在哪里碰壁，碰壁的经历变成训练数据
    - 样本效率：达到同样的控制效果需要多少步真实交互

前置知识：
    - 06-cem-and-mppi.py（CEM）、07-uncertainty-and-ensembles.py（概率集成）

对应概念文档：
    - CONCEPT.md 第 9 节「PETS 与基于模型的强化学习闭环」

预计运行时间：约 6 分钟（MPS）
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

from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-01")
SEED = 0
MAX_TORQUE = 2.0
MAX_SPEED = 8.0
DT = 0.05
HORIZON = 30
N_MODELS = 5
# 这一节每次规划要同时展开几千个粒子，批量大，放在 MPS / GPU 上更快
DEVICE = get_device()


# ==================== 第一部分：闭环 ====================


def introduction():
    """PETS 的三个部件和外层的闭环。"""
    print("=" * 60)
    print("第一部分：把部件接成闭环")
    print("=" * 60)

    print("""
前面几节的部件：从数据里学模型（03）、用模型规划（05、06）、让模型知道自己哪里不准（07）。
上一节的结尾留下一个问题：只有 200 条转移、全在底部附近时，集成能让规划器少受骗，
却变不出模型没学到的知识。缺的知识只能靠数据补。

PETS 的办法是让这几个环节转起来：

        ┌──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          │
    ┌─────────┐    ┌──────────────┐    ┌──────────────────┐    ┌───────────┐
    │ 数据集 D │───▶│ 训练概率集成   │───▶│ 在真实环境跑一个回合 │───▶│ 新的转移    │
    └─────────┘    │（07 课）      │    │ 每一步：CEM 规划    │    │ 加进 D     │
                   └──────────────┘    │ + 轨迹采样（本节）   │    └───────────┘
                                       └──────────────────┘

这个闭环会自己纠错：
    - 模型在某个区域有空子 → 规划器被吸引过去 → 在真实环境里碰壁
    - 碰壁的经历进了数据集 → 下一轮训练后这个空子被补上

新的部件只有一个：轨迹采样。有了集成之后，"在模型里展开一条动作序列"不再只有一种结果。
PETS 对每条候选序列同时展开 P 个粒子，每个粒子是这条序列的一种可能的未来：

    一条动作序列 A = (a_0, ..., a_{H-1})，展开 P 个粒子

    粒子 1（跟成员 2）： s ──▶ ŝ_1 ──▶ ŝ_2 ──▶ ··· ──▶ ŝ_H      累计奖励 J_1
    粒子 2（跟成员 5）： s ──▶ ŝ_1 ──▶ ŝ_2 ──▶ ··· ──▶ ŝ_H      累计奖励 J_2
      ...
    粒子 P（跟成员 3）： s ──▶ ŝ_1 ──▶ ŝ_2 ──▶ ··· ──▶ ŝ_H      累计奖励 J_P

                              这条序列的分数 = J_1 … J_P 的平均

每个粒子每一步做两件事：用自己跟定的成员预测下一状态的分布（认知不确定性），
再从这个分布里采样（偶然不确定性）。
    """)
    print(f"  使用设备：{DEVICE}")


# ==================== 概率集成（与 07 课相同）====================


class EnsembleLinear(nn.Module):
    """B 个线性层并排放：输入 [B, N, in] → 输出 [B, N, out]。"""

    def __init__(self, n_models, in_dim, out_dim):
        super().__init__()
        bound = 1.0 / np.sqrt(in_dim)
        self.weight = nn.Parameter(torch.empty(n_models, in_dim, out_dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(n_models, 1, out_dim).uniform_(-bound, bound))

    def forward(self, x, members=None):
        weight, bias = (self.weight, self.bias) if members is None else (self.weight[members], self.bias[members])
        return torch.baddbmm(bias, x, weight)


class ProbabilisticEnsemble(nn.Module):
    """摆的概率集成：输入 (cos θ, sin θ, θ̇, u)，输出状态变化量 (Δθ, Δθ̇) 的均值和对数方差。"""

    def __init__(self, n_models=N_MODELS, hidden=200):
        super().__init__()
        self.n_models = n_models
        self.layers = nn.ModuleList([
            EnsembleLinear(n_models, 4, hidden),
            EnsembleLinear(n_models, hidden, hidden),
            EnsembleLinear(n_models, hidden, 4),
        ])
        self.max_logvar = nn.Parameter(torch.full((2,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((2,), -10.0))
        self.register_buffer("x_mean", torch.zeros(4))
        self.register_buffer("x_std", torch.ones(4))
        self.register_buffer("y_mean", torch.zeros(2))
        self.register_buffer("y_std", torch.ones(2))

    @staticmethod
    def features(states, actions):
        return torch.cat([torch.cos(states[..., :1]), torch.sin(states[..., :1]), states[..., 1:], actions], dim=-1)

    def forward(self, states, actions, members=None):
        """标准化空间的均值和对数方差，形状都是 [B, N, 2]。"""
        h = (self.features(states, actions) - self.x_mean) / self.x_std
        h = F.silu(self.layers[0](h, members))
        h = F.silu(self.layers[1](h, members))
        out = self.layers[2](h, members)
        mean, logvar = out[..., :2], out[..., 2:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    @torch.no_grad()
    def predict(self, states, actions, members=None):
        """物理单位下，状态变化量的均值和标准差，形状都是 [B, N, 2]。"""
        mean, logvar = self(states, actions, members)
        return mean * self.y_std + self.y_mean, torch.exp(0.5 * logvar) * self.y_std


def train_ensemble(data, n_steps, seed, batch_size=256, lr=1e-3):
    """在转移数据 (s, a, s') 上从头训练一个概率集成（高斯 NLL + 自助采样）。"""
    s, a, s_next = [d.to(DEVICE) for d in data]
    torch.manual_seed(seed)
    model = ProbabilisticEnsemble().to(DEVICE)
    x, y = model.features(s, a), s_next - s
    # 归一化统计量每次都用当前的全部数据重新算（数据分布在闭环里一直在变）
    model.x_mean.copy_(x.mean(0))
    model.x_std.copy_(x.std(0).clamp_min(1e-6))
    model.y_mean.copy_(y.mean(0))
    model.y_std.copy_(y.std(0).clamp_min(1e-6))
    n = s.shape[0]
    generator = torch.Generator().manual_seed(seed)
    bootstrap = torch.randint(0, n, (N_MODELS, n), generator=generator)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_steps)
    for _ in range(n_steps):
        pick = torch.randint(0, n, (N_MODELS, batch_size), generator=generator)
        idx = torch.gather(bootstrap, 1, pick).to(DEVICE)
        mean, logvar = model(s[idx], a[idx])
        target = (y[idx] - model.y_mean) / model.y_std
        nll = 0.5 * ((mean - target) ** 2 * torch.exp(-logvar) + logvar).mean()
        loss = nll + 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model


# ==================== 环境相关 ====================


def wrap_angle(theta):
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def true_step(states, actions):
    theta, theta_dot = states[..., 0], states[..., 1]
    u = actions[..., 0].clamp(-MAX_TORQUE, MAX_TORQUE)
    theta_dot_new = (theta_dot + (15.0 * torch.sin(theta) + 3.0 * u) * DT).clamp(-MAX_SPEED, MAX_SPEED)
    return torch.stack([theta + theta_dot_new * DT, theta_dot_new], dim=-1)


def reward_fn(states, actions):
    theta = wrap_angle(states[..., 0])
    return -(theta**2 + 0.1 * states[..., 1] ** 2 + 0.001 * actions[..., 0] ** 2)


def collect_random_episode(seed):
    """用随机策略跑一个回合，返回 CPU 上的转移 (s, a, s') 和回报。"""
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    s, a, s_next, total = [], [], [], 0.0
    for _ in range(200):
        s.append(env.unwrapped.state.copy())
        action = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=1).astype(np.float32)
        _, reward, _, _, _ = env.step(action)
        total += reward
        a.append(action)
        s_next.append(env.unwrapped.state.copy())
    env.close()
    return [torch.as_tensor(np.array(x, dtype=np.float32)) for x in (s, a, s_next)], total


# ==================== 第二部分：轨迹采样 ====================


@torch.no_grad()
def score_sequences(model, state, actions, mode, particles_per_member, generator):
    """给候选动作序列打分。actions [N, H, 1] → scores [N]。

    mode:
        "perfect"  用真实物理方程展开（参照线）
        "single"   只用成员 0 的均值预测，一条序列一种结果（03–06 课的做法）
        "ts_inf"   轨迹采样 TS∞：每个粒子全程跟定一个成员
        "ts_1"     轨迹采样 TS1：每一步把粒子随机重新分给成员

    粒子的排布：每个成员带 k = particles_per_member 个粒子，一条序列共 P = B × k 个。
    状态张量的形状是 [B, N × k, 2]，第 0 维就是"这个粒子现在归哪个成员管"。
    """
    n, horizon, _ = actions.shape
    if mode == "perfect":
        states, returns = state.expand(1, n, 2), torch.zeros(1, n, device=actions.device)
        for h in range(horizon):
            a = actions[None, :, h]
            returns += reward_fn(states, a)
            states = true_step(states, a)
        return returns[0]
    if mode == "single":
        states, returns = state.expand(1, n, 2), torch.zeros(1, n, device=actions.device)
        for h in range(horizon):
            a = actions[None, :, h]
            returns += reward_fn(states, a)
            # 用切片而不是列表来选成员：切片不复制权重。在 MPS 上，每次调用都用列表索引会慢好几倍
            states = states + model.predict(states, a, members=slice(0, 1))[0]
        return returns[0]

    k, n_models = particles_per_member, model.n_models
    particle_actions = actions.repeat_interleave(k, dim=0)        # [N × k, H, 1]，同一条序列的粒子排在一起
    states = state.expand(n_models, n * k, 2)                     # [B, N × k, 2]
    returns = torch.zeros(n_models, n * k, device=actions.device)  # [B, N × k]
    for h in range(horizon):
        a = particle_actions[:, h].expand(n_models, -1, -1)
        returns += reward_fn(states, a)
        delta_mean, delta_std = model.predict(states, a)          # 每个成员预测自己名下的粒子
        noise = torch.randn(states.shape, generator=generator, device=states.device)
        states = states + delta_mean + delta_std * noise          # 从预测的高斯分布里采样
        if mode == "ts_1":
            # 每一列是同一条序列的 B 个粒子，把它们在成员之间随机换位。累计奖励跟着粒子走。
            perm = torch.rand(n_models, n * k, generator=generator, device=states.device).argsort(dim=0)
            states = torch.gather(states, 0, perm[..., None].expand(-1, -1, 2))
            returns = torch.gather(returns, 0, perm)
    # 把 [B, N × k] 整理成 [B, N, k]，对一条序列的全部 B × k 个粒子取平均
    return returns.view(n_models, n, k).mean(dim=(0, 2))


def cem_plan(score_fn, mean, generator, n_candidates=200, n_elites=20, n_iters=5):
    """06 课的 CEM。"""
    std = torch.ones_like(mean)
    for _ in range(n_iters):
        noise = torch.randn(n_candidates, *mean.shape, generator=generator, device=mean.device)
        actions = (mean + std * noise).clamp(-MAX_TORQUE, MAX_TORQUE)
        elites = actions[score_fn(actions).topk(n_elites).indices]
        mean = 0.1 * mean + 0.9 * elites.mean(dim=0)
        std = elites.std(dim=0, correction=0).clamp_min(0.1)
    return mean


def run_episode(model, mode, seed, particles_per_member=4):
    """用 CEM-MPC 在真实环境里跑一个回合。

    返回：
        transitions  这个回合的转移 (s, a, s')，CPU 张量
        total        真实回报
        heights      每一步的摆尖高度
        model_error  模型对这个回合每条转移的单步预测误差（角速度，取成员 0 的均值）。
                     这些转移模型还没见过，所以这是它在"控制器真正去的地方"的泛化误差
    """
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    mean = torch.zeros(HORIZON, 1, device=DEVICE)
    s, a, s_next, heights, total = [], [], [], [], 0.0
    for _ in range(200):
        state_np = env.unwrapped.state.copy()
        state = torch.as_tensor(state_np, dtype=torch.float32, device=DEVICE)
        mean = cem_plan(lambda acts: score_sequences(model, state, acts, mode, particles_per_member, generator), mean, generator)
        action = mean[0].cpu().numpy().astype(np.float32)
        _, reward, _, _, _ = env.step(action)
        total += reward
        s.append(state_np)
        a.append(action)
        s_next.append(env.unwrapped.state.copy())
        heights.append(np.cos(state_np[0]))
        mean = torch.cat([mean[1:], torch.zeros(1, 1, device=DEVICE)])
    env.close()
    transitions = [torch.as_tensor(np.array(x, dtype=np.float32)) for x in (s, a, s_next)]
    if model is None:
        model_error = float("nan")
    else:
        ts, ta, tn = [t.to(DEVICE)[None] for t in transitions]
        pred = ts + model.predict(ts, ta, members=slice(0, 1))[0]
        model_error = (pred - tn)[0, :, 1].abs().mean().item()
    return transitions, total, np.array(heights), model_error


def particle_demo(model):
    """看一眼粒子：同一条动作序列，在数据少的模型里展开出来的未来有多分散。"""
    print("\n" + "=" * 60)
    print("第二部分：轨迹采样长什么样")
    print("=" * 60)

    print("""
先看粒子本身。模型是用 1 个随机回合（200 条转移，摆只在底部附近晃）训练的集成，
也就是上一节"数据很少"的那个模型。从正下方出发，执行同一条"荡秋千"的动作序列
（顺着摆的运动方向推），用 TS∞ 展开 20 个粒子，再和真实物理方程的结果比。
    """)
    horizon, k = 40, 4
    # 先用真实物理方程生成一条荡秋千的动作序列
    state = torch.tensor([np.pi, 0.0], device=DEVICE)
    true_states, actions, s = [state], [], state
    for _ in range(horizon):
        u = MAX_TORQUE if s[1] >= 0 else -MAX_TORQUE
        act = torch.tensor([u], device=DEVICE)
        s = true_step(s[None], act[None])[0]
        actions.append(act)
        true_states.append(s)
    true_heights = torch.cos(torch.stack(true_states)[:, 0]).cpu().numpy()
    actions = torch.stack(actions)  # [H, 1]

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    states = state.expand(N_MODELS, k, 2)
    particle_heights = [torch.cos(states[..., 0])]
    for h in range(horizon):
        a = actions[h].expand(N_MODELS, k, 1)
        delta_mean, delta_std = model.predict(states, a)
        states = states + delta_mean + delta_std * torch.randn(states.shape, generator=generator, device=DEVICE)
        particle_heights.append(torch.cos(states[..., 0]))
    particle_heights = torch.stack(particle_heights).cpu().numpy()  # [H+1, B, k]

    print(f"  {'展开步数':<8}{'真实的摆尖高度':>14}{'20 个粒子：最低':>16}{'最高':>8}{'成员之间的差距（各成员粒子均值的极差）':>36}")
    for h in (5, 10, 20, 30, 40):
        ph = particle_heights[h]
        member_means = ph.mean(axis=1)
        print(f"  {h:<10}{true_heights[h]:>14.2f}{ph.min():>18.2f}{ph.max():>10.2f}{member_means.max() - member_means.min():>24.2f}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = np.arange(horizon + 1)
    for b in range(N_MODELS):
        for j in range(k):
            ax.plot(t, particle_heights[:, b, j], color=f"C{b}", alpha=0.6, linewidth=1,
                    label=f"成员 {b} 的粒子" if j == 0 else None)
    ax.plot(t, true_heights, "k--", linewidth=2, label="真实物理方程")
    ax.set_xlabel("展开步数")
    ax.set_ylabel("摆尖高度 cos θ")
    ax.set_title("同一条动作序列的 20 个粒子（模型只见过 200 条底部附近的转移）")
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, OUT / "08-particles.png")

    spread_early = particle_heights[5].max() - particle_heights[5].min()
    spread_late = particle_heights[40].max() - particle_heights[40].min()
    print(f"""
前几步摆还在底部附近，那里有数据，20 个粒子挤在一起（第 5 步的高度差 {spread_early:.2f}）。
摆越甩越高，进入模型没见过的区域，成员们开始各说各的，粒子散开
（第 40 步的高度差 {spread_late:.2f}）。图上同一种颜色是同一个成员的粒子：
颜色之间差得远、同色之间挨得近，说明分散来自成员之间的分歧（认知不确定性），
不是来自采样噪声（偶然不确定性）。摆是确定性系统，这符合预期。

规划时，这条序列的分数是 20 个粒子回报的平均。一条序列只在某一个成员里拿高分，
平均之后就不突出了；在所有成员里都拿高分，才会被选中。
    """)


# ==================== 第三部分：完整的 PETS ====================


def pets_loop(mode, n_iterations, train_steps, loop_seed, verbose=True):
    """PETS 的外层循环。mode 决定规划时怎么用模型（见 score_sequences）。

    初始数据是 1 个随机回合。之后每一轮：用全部数据训练集成 → 跑一个回合 → 数据加进数据集。
    每一轮的回合用不同的起点（种子 200 + 轮数），各种方法之间起点相同，可以逐回合比较。
    """
    data, random_return = collect_random_episode(seed=10)  # 上一节那个"数据很少"的回合
    history = []
    for it in range(n_iterations):
        start = time.time()
        model = train_ensemble(data, train_steps, seed=loop_seed + it)
        train_time = time.time() - start
        start = time.time()
        transitions, total, heights, model_error = run_episode(model, mode, seed=200 + it)
        history.append(dict(n_data=data[0].shape[0], ret=total, error=model_error, heights=heights))
        if verbose:
            settled = "稳住了" if heights[-50:].min() > 0.95 else "没稳住"
            print(f"  第 {it + 1:>2} 轮：已有 {data[0].shape[0]:>5} 条转移 → 训练 {train_time:4.1f} 秒 → "
                  f"回合回报 {total:8.1f}（{settled}）  模型在新数据上的误差 {model_error:.1e}   [{time.time() - start:.0f} 秒]")
        data = [torch.cat([d, t]) for d, t in zip(data, transitions)]
    return history, random_return


def run_pets(n_iterations, train_steps):
    """跑完整的 PETS，并和完美模型的 MPC 逐回合比较。"""
    print("\n" + "=" * 60)
    print("第三部分：完整的 PETS")
    print("=" * 60)

    print(f"""
设置：集成 {N_MODELS} 个成员，每条候选序列 20 个粒子（TS∞），CEM 每轮 200 条候选 × 5 轮，视野 {HORIZON}。
初始数据就是上一节那个运气不好的随机回合（200 条转移，摆只在底部附近）。
每一轮用全部数据从头训练集成，然后在真实环境里跑一个回合，把新数据加进去。
    """)

    history, random_return = pets_loop("ts_inf", n_iterations, train_steps, loop_seed=SEED)

    # 参照：完美模型 + 同样的 CEM，在同一组起点上
    perfect = [run_episode(None, "perfect", seed=200 + it)[1] for it in range(n_iterations)]
    returns = np.array([h["ret"] for h in history])
    print(f"\n  {'轮次':<6}{'累计真实交互步数':>14}{'PETS 的回报':>14}{'完美模型 MPC（同一起点）':>24}{'差距':>10}")
    for it, h in enumerate(history):
        print(f"  {it + 1:<8}{h['n_data']:>14}{h['ret']:>16.1f}{perfect[it]:>20.1f}{h['ret'] - perfect[it]:>16.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(1, n_iterations + 1)
    axes[0].plot(x, returns, "o-", linewidth=2, label="PETS")
    axes[0].plot(x, perfect, "s--", color="gray", label="完美模型 + CEM-MPC（同一起点）")
    axes[0].axhline(random_return, color="tomato", linestyle=":", label=f"随机策略（初始的那个回合，{random_return:.0f}）")
    axes[0].set_xlabel("轮次（每轮在真实环境里跑一个回合，200 步）")
    axes[0].set_ylabel("回合的真实回报")
    axes[0].set_title("PETS 的学习曲线")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, [h["error"] for h in history], "o-", linewidth=2, color="C3")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("轮次")
    axes[1].set_ylabel("模型在这一轮新数据上的单步误差（角速度，对数坐标）")
    axes[1].set_title("模型在控制器真正去的地方越来越准")
    save_figure(fig, OUT / "08-pets-learning-curve.png")

    # 第几轮开始和完美模型打平（差距在 30 分以内）并且之后一直保持
    gaps = returns - np.array(perfect)
    solved = next((i for i in range(n_iterations) if (gaps[i:] > -30).all()), None)
    return history, perfect, random_return, solved


def pets_summary(history, perfect, random_return, solved):
    """根据实测数字写结论。"""
    errors = [h["error"] for h in history]
    if solved is None:
        solved_text = "在这次运行的轮数内还没有稳定地追上完美模型。"
    else:
        steps = history[solved]["n_data"]
        solved_text = (f"从第 {solved + 1} 轮起，PETS 的回报和完美模型的差距一直在 30 分以内。\n"
                       f"       那时数据集里只有 {steps} 条转移：1 个随机回合，加上 {solved} 个自己控制的回合。")
    print(f"""
怎么读这些结果：

    1. 第 1 轮的模型只见过 200 条底部附近的转移，回报 {history[0]['ret']:.0f}，没能把摆稳住
       （同一个起点上完美模型是 {perfect[0]:.0f}；初始的那个随机回合是 {random_return:.0f}）。
       这就是上一节结尾的状态。
    2. {solved_text}
    3. 右图是原因：模型在"这一轮控制器实际走过的地方"的单步误差，
       从第 1 轮的 {errors[0]:.3f} 降到第 2 轮的 {errors[1]:.4f}，最后一轮是 {errors[-1]:.1e}。
       第 1 轮控制器想把摆甩上去，走进了模型没见过的区域，误差很大，没成功；
       但这次失败的 200 条转移恰好覆盖了甩上去的那一段。下一轮模型就知道了。

注意数据是怎么来的：没有人去设计"该采哪里的数据"。控制器想去哪，数据就落在哪；
模型在哪里错得离谱，控制器就在哪里多待一会儿（因为没成功）。
数据采集和任务目标在这个闭环里是绑在一起的。
对比一下：03 到 06 课用的是 40 个随机回合、8000 条转移，其中大部分落在和任务关系不大的地方。
    """)


# ==================== 第四部分：每个部件值多少 ====================


def ablation(n_iterations, train_steps, pets_history, perfect):
    """同一个闭环里，换掉规划时用模型的方式。"""
    print("\n" + "=" * 60)
    print("第四部分：每个部件值多少")
    print("=" * 60)

    print(f"""
PETS 有三样东西：闭环、概率集成、轨迹采样。上面的实验三样全开。
现在保持闭环不变，只换规划时用模型的方式，各跑 {n_iterations} 轮（差别如果有，只会出现在数据最少的头几轮）：

    单模型         只用成员 0 的均值预测，一条序列一种结果
    TS1           每一步把粒子随机重新分给成员
    TS∞（PETS）    每个粒子全程跟定一个成员（就是上面那次运行）
    """)

    results = {"TS∞（PETS）": pets_history[:n_iterations]}
    for label, mode in [("单模型", "single"), ("TS1", "ts_1")]:
        start = time.time()
        results[label], _ = pets_loop(mode, n_iterations, train_steps, loop_seed=SEED, verbose=False)
        print(f"  {label} 跑完，用时 {time.time() - start:.0f} 秒")

    order = ["单模型", "TS1", "TS∞（PETS）"]
    print(f"\n  {'轮次':<6}" + "".join(f"{name:>14}" for name in order) + f"{'完美模型':>12}")
    for it in range(n_iterations):
        print(f"  {it + 1:<8}" + "".join(f"{results[name][it]['ret']:>14.1f}" for name in order) + f"{perfect[it]:>14.1f}")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(1, n_iterations + 1)
    for name in order:
        ax.plot(x, [h["ret"] for h in results[name]], "o-", linewidth=2, label=name)
    ax.plot(x, perfect[:n_iterations], "s--", color="gray", label="完美模型 + CEM-MPC")
    ax.set_xlabel("轮次")
    ax.set_ylabel("回合的真实回报")
    ax.set_title("同一个闭环，三种用模型的方式")
    ax.legend(fontsize=9)
    save_figure(fig, OUT / "08-ablation.png")
    return results, order


def ablation_summary(results, order, perfect, n_iterations):
    """根据实测数字写结论。"""
    first = {name: results[name][0]["ret"] for name in order}
    rest_gap = {name: max(abs(results[name][i]["ret"] - perfect[i]) for i in range(1, n_iterations)) if n_iterations > 1 else float("nan")
                for name in order}
    print(f"""
怎么读这张表：

    - 第 1 轮三种方式都没把摆稳住：单模型 {first['单模型']:.0f}，TS1 {first['TS1']:.0f}，TS∞ {first['TS∞（PETS）']:.0f}
      （完美模型是 {perfect[0]:.0f}）。它们之间的差别不用解读：每种方式只跑了一次，
      而上一节看到，同一份数据训练出的 5 个成员，真实回报就能差出几倍。
    - 从第 2 轮起，三种方式和完美模型的差距最大只有 {rest_gap['单模型']:.1f}、{rest_gap['TS1']:.1f}、{rest_gap['TS∞（PETS）']:.1f} 分，分不出彼此。
      在摆上，起决定作用的是闭环本身：只要控制器能把自己的经历喂回给模型，
      单个确定性模型也够用。集成和轨迹采样在这里没有派上用场。

这是一个如实的"没有差别"。原因是摆太简单：状态 2 维、动作 1 维，
失败一个回合就把要紧的区域采到了，模型的空子只存活了一轮。
PETS 论文在 7 自由度机械臂、HalfCheetah 这类任务上做了同类的消融，
那里概率集成相对单个确定性模型的优势是明显的（见论文的消融实验部分）。
本章的实战项目换到 Reacher-v5，Phase 5 的排行榜换到 HalfCheetah，届时再看这些部件的作用。
    """)


# ==================== 第五部分：小结与练习 ====================


def chapter_summary():
    """这一章学了什么，PETS 的边界在哪。"""
    print("\n" + "=" * 60)
    print("第五部分：这一章的终点，也是后面各章的起点")
    print("=" * 60)

    print("""
八节讲义搭起来的东西：

    01 世界模型是什么      一个回答"如果我这样做，接下来会怎样"的函数
    02 MDP 与数据         状态、动作、转移、奖励；随机数据覆盖不到要紧的地方
    03 学动力学模型        预测变化量、归一化、角度用 cos/sin
    04 复合误差           误差在不稳定的区域按指数放大；别让模型开环跑太远
    05 MPC               每一步重新规划，只执行第一个动作
    06 CEM 与 MPPI        让采样分布跟着分数走；热启动
    07 不确定性与集成      成员之间的分歧能发现"没见过"；集成让规划器少受骗
    08 PETS              把它们接成闭环，让控制器自己把模型的空子补上

PETS 把"状态输入 + 决策时规划"这条路走通了，也把这条路的边界露了出来。
每一条边界都是后面某一章的起点：

    ┌────────────────────┬──────────────────────────────────────┬──────────┐
    │ PETS 的局限          │ 后面怎么解决                           │ 在哪      │
    ├────────────────────┼──────────────────────────────────────┼──────────┤
    │ 只能处理低维状态      │ 先把图像压成隐状态，在隐空间里预测和规划   │ P2、P3   │
    │ 决策很慢             │ 提前在想象里训练一个策略                 │ P6       │
    │ 看不到视野之外        │ 用价值函数概括视野之外的回报             │ P4、P7   │
    │ 需要已知的奖励函数    │ 奖励也由模型来预测                      │ P3 起    │
    │ 所有状态维度一视同仁  │ 只预测对决策有用的量                    │ P7、P10  │
    └────────────────────┴──────────────────────────────────────┴──────────┘

本章写的 CEM 和 MPPI 会一直用下去：P3 把 CEM 搬进隐空间（PlaNet），
P7 给 MPPI 加上价值函数（TD-MPC2），P10 用 CEM 在预训练特征上规划（DINO-WM）。
    """)


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：算一下计算量
    PETS 每个真实步的模型调用次数是 N × P × H × I（候选数 × 粒子数 × 视野 × 迭代轮数）。
    本节的设置下是多少？一个 200 步的回合呢？

练习 1 参考答案：
    200 × 20 × 30 × 5 = 600,000 次模型预测 / 步，一个回合是 1.2 亿次。
    靠批量化才算得动：每一步规划是 30 × 5 = 150 次前向计算，
    每次的批量是 200 × 20 = 4000 个粒子。
    06 课单模型规划的批量是 200。粒子让计算量涨了 20 倍，这是集成的直接代价。

练习 2：把分歧当惩罚
    score_sequences() 对粒子的回报取平均，这是风险中性的。
    改成"平均 − β × 粒子回报的标准差"，取 β = 1，重新跑第三部分。

练习 2 参考答案：
    把最后一行换成：
        r = returns.view(n_models, n, k).permute(1, 0, 2).reshape(n, -1)   # [N, P]
        return r.mean(dim=1) - beta * r.std(dim=1)
    规划器会回避粒子意见不一致的动作序列，也就是回避模型没把握的区域。
    在这个闭环里要小心它的副作用：没把握的区域正是需要去采数据的地方，
    回避它们会让模型的空白一直补不上。头几轮的进展可能反而变慢。
    保守的规划适合"不许试错"的场合（比如只有离线数据）；
    能自己采数据时，取平均甚至反过来奖励分歧（鼓励探索）往往更好。

练习 3：每一轮不从头训练
    train_ensemble() 每一轮都重新初始化网络。改成接着上一轮的权重继续训练
    （训练步数减到三分之一），比较学习曲线和总耗时。

练习 3 参考答案：
    接着训练更省时间，PETS 的原始实现就是这样做的。要注意两点：
    一是归一化统计量变了，旧权重对应的是旧的输入尺度，通常做法是统计量也平滑地更新；
    二是网络可能记住了早期数据里的错误模式，新数据的占比在头几轮又很小。
    数据量不大时从头训练更省心，本节选的就是这种做法。

思考题 1：PETS 没有训练任何神经网络策略，它算强化学习吗？

思考题 1 参考答案：
    算。强化学习指的是"通过和环境交互来学会做决策"这一类问题，不限定解法。
    PETS 通过交互改进模型，进而改进决策，属于基于模型的强化学习。
    它和 Phase 4、5 的方法的区别在于"学到的东西存在哪"：
    PETS 把经验存进模型，决策时现场规划；无模型方法把经验存进价值函数或策略。

思考题 2：闭环为什么能自己纠错？有没有它纠不了的错？

思考题 2 参考答案：
    模型高估了某个区域 → 规划器去那里 → 真实结果不好 → 这段经历进数据集 → 高估被纠正。
    高估总会被发现，因为规划器会主动去验证它。
    纠不了的是低估：如果模型错误地认为某条路很糟，规划器永远不会去走，
    也就永远采不到能纠正这个错误的数据。这是"利用"压过"探索"的典型后果。
    摆的任务里这个问题不明显，因为甩上去只有一种办法。
    在有多种解法、奖励稀疏的任务上，需要专门的探索机制（比如奖励分歧）。

思考题 3：本章一直假设奖励函数是已知的。真实的机器人任务里，这个假设成立吗？

思考题 3 参考答案：
    多数时候不成立。"把杯子放到架子上"这样的任务，很难写出一个从关节角度
    算出来的奖励公式；如果输入是图像，更是无从写起。常见的出路有三条：
    让模型把奖励也学出来（P3 起的做法）；给一张目标图像，
    把"和目标有多像"当作奖励（P10 的 DINO-WM）；或者干脆不用奖励，
    从人的演示里学（P11 的模仿学习）。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()

    n_iterations, n_ablation, train_steps = (2, 2, 100) if is_quick() else (6, 4, 2000)

    data, _ = collect_random_episode(seed=10)
    particle_demo(train_ensemble(data, train_steps, seed=SEED))

    history, perfect, random_return, solved = run_pets(n_iterations, train_steps)
    pets_summary(history, perfect, random_return, solved)
    results, order = ablation(n_ablation, train_steps, history, perfect)
    ablation_summary(results, order, perfect, n_ablation)
    chapter_summary()
    exercises()

    print("\n" + "=" * 60)
    print("Phase 1 的讲义全部完成！接下来：projects/phase-01-arm-mpc/（机械臂模型预测控制）")
    print("=" * 60)


if __name__ == "__main__":
    main()
