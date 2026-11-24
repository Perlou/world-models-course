"""
多步展开与复合误差
==================

学习目标：
    1. 区分单步预测和开环展开，知道规划用的是后者
    2. 实测学到的模型在开环展开时误差怎么随步数增长
    3. 理解误差增长的快慢取决于系统在哪个区域：摆的顶部放大误差，底部不放大
    4. 用对照实验看到分布偏移：训练数据没覆盖的区域，模型不可靠
    5. 了解多步损失，并如实评估它在这个任务上有没有用

核心概念：
    - 开环展开（open-loop rollout）：只有起点是真实状态，之后每一步的输入都是模型自己的输出
    - 复合误差（compounding error）：e_{t+1} ≤ L·e_t + ε，旧误差被放大，再加一份新误差
    - 稳定区与不稳定区：线性化后矩阵的最大特征值大于 1，偏差就按指数放大
    - 分布偏移（distribution shift）：使用时遇到的状态不在训练数据的分布里
    - 多步损失：训练时也做开环展开，把多步误差放进损失

前置知识：
    - 03-learning-dynamics-mlp.py（动力学模型）

对应概念文档：
    - CONCEPT.md 第 5 节「多步预测与复合误差」

预计运行时间：约 1 分钟（MPS）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-01")
SEED = 0
MAX_TORQUE = 2.0
MAX_SPEED = 8.0
DT = 0.05
DEVICE = get_device()


# ==================== 第一部分：单步预测与开环展开 ====================


def introduction():
    """两种评估模型的方式。"""
    print("=" * 60)
    print("第一部分：单步预测与开环展开")
    print("=" * 60)

    print("""
上一节的模型单步误差只有万分之几弧度。但规划要往前看几十步，
模型一次只预测一步，所以要把输出再喂回去当输入，一步一步往前推：

    单步预测：每一步的输入都是真实状态

       真实： s_0 ───▶ s_1 ───▶ s_2 ───▶ s_3
               │        │        │
               ▼        ▼        ▼
       预测：  ŝ_1      ŝ_2      ŝ_3            误差不会累积

    开环展开：只有第 0 步是真实状态，之后全靠模型自己

       真实： s_0 ───▶ s_1 ───▶ s_2 ───▶ s_3
               │
               ▼
       预测：  ŝ_1 ───▶ ŝ_2 ───▶ ŝ_3            误差一路累积

这和航位推算是一回事：没有 GPS 的船靠航向和航速估计位置，
每次估计的误差很小，但会一直累加，开得越远偏得越多。

规划时模型一直处于开环状态。所以衡量一个模型能不能用来规划，
要看开环展开的误差，不是上一节报告的单步误差。
    """)
    print(f"  使用设备：{DEVICE}")


# ==================== 数据与模型（与 03 课相同，只保留完整版）====================


def wrap_angle(theta):
    """把角度折到 [-π, π)。两个角度相减后再折一次，得到的是它们在圆周上的真实距离。"""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def true_step(states, actions):
    """真实的物理方程（NumPy 批量版）。states [N, 2]，actions [N, 1] → 下一状态 [N, 2]。"""
    theta, theta_dot = states[:, 0], states[:, 1]
    u = np.clip(actions[:, 0], -MAX_TORQUE, MAX_TORQUE)
    theta_dot_new = np.clip(theta_dot + (15.0 * np.sin(theta) + 3.0 * u) * DT, -MAX_SPEED, MAX_SPEED)
    return np.stack([theta + theta_dot_new * DT, theta_dot_new], axis=-1)


def collect_random_dataset(n_episodes, seed=SEED):
    """用随机策略采集数据。返回按回合排好的序列：states [E, 201, 2]，actions [E, 200, 1]。"""
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)
    all_states, all_actions = [], []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        states, actions = [env.unwrapped.state.copy()], []
        for _ in range(200):
            action = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=1).astype(np.float32)
            env.step(action)
            actions.append(action)
            states.append(env.unwrapped.state.copy())
        all_states.append(states)
        all_actions.append(actions)
    env.close()
    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.float32)


def to_transitions(seq_states, seq_actions):
    """把序列 [E, 201, 2]、[E, 200, 1] 拆成转移 (s, a, s')，放到设备上。"""
    s = seq_states[:, :-1].reshape(-1, 2)
    a = seq_actions.reshape(-1, 1)
    s_next = seq_states[:, 1:].reshape(-1, 2)
    return [torch.as_tensor(x, device=DEVICE) for x in (s, a, s_next)]


class DynamicsModel(nn.Module):
    """03 课的完整模型：输入 (cos θ, sin θ, θ̇, u)，输出标准化后的 (Δθ, Δθ̇)。"""

    def __init__(self, hidden=200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        self.register_buffer("x_mean", torch.zeros(4))
        self.register_buffer("x_std", torch.ones(4))
        self.register_buffer("y_mean", torch.zeros(2))
        self.register_buffer("y_std", torch.ones(2))

    def features(self, states, actions):
        theta, theta_dot = states[:, :1], states[:, 1:]
        return torch.cat([torch.cos(theta), torch.sin(theta), theta_dot, actions], dim=-1)

    def fit_normalizer(self, states, actions, next_states):
        x, y = self.features(states, actions), next_states - states
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp_min(1e-6))
        self.y_mean.copy_(y.mean(0))
        self.y_std.copy_(y.std(0).clamp_min(1e-6))

    def predict(self, states, actions):
        """[B, 2], [B, 1] → 预测的下一状态 [B, 2]。"""
        x = (self.features(states, actions) - self.x_mean) / self.x_std
        return states + self.net(x) * self.y_std + self.y_mean


def train_model(model, transitions, epochs, multistep=None, batch_size=256, lr=1e-3):
    """训练动力学模型。

    multistep=None：单步损失，和 03 课一样。
    multistep=(seq_states, seq_actions, k)：多步损失。从真实状态出发在模型里展开 k 步，
        每一步的预测都和真实状态比，k 步的误差取平均。梯度沿展开的链条反向传播。
    """
    model.to(DEVICE)
    model.fit_normalizer(*transitions)
    if multistep is None:
        data = transitions
    else:
        seq_states, seq_actions, k = multistep
        # 把每个回合切成长度为 k 的滑动窗口：起点状态 + k 个动作 + k 个后续状态
        starts = np.arange(seq_actions.shape[1] - k + 1)
        win_s = np.stack([seq_states[:, i:i + k + 1] for i in starts], axis=1).reshape(-1, k + 1, 2)
        win_a = np.stack([seq_actions[:, i:i + k] for i in starts], axis=1).reshape(-1, k, 1)
        data = [torch.as_tensor(win_s, device=DEVICE), torch.as_tensor(win_a, device=DEVICE)]

    n = data[0].shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    steps = epochs * ((n + batch_size - 1) // batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            if multistep is None:
                s, a, s_next = [t[idx] for t in data]
                loss = (((model.predict(s, a) - s_next) / model.y_std) ** 2).mean()
            else:
                win_s, win_a = data[0][idx], data[1][idx]  # [B, k+1, 2], [B, k, 1]
                state = win_s[:, 0]
                loss = 0.0
                for j in range(k):
                    state = model.predict(state, win_a[:, j])  # 输入是上一步的预测，不是真实状态
                    loss = loss + (((state - win_s[:, j + 1]) / model.y_std) ** 2).mean()
                loss = loss / k
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    return model


# ==================== 第二部分：开环展开 ====================


def rollout_true(start_states, actions):
    """用真实物理方程展开。start_states [N, 2]，actions [N, H, 1] → 状态序列 [N, H+1, 2]。"""
    states = [np.asarray(start_states, dtype=np.float64)]
    for t in range(actions.shape[1]):
        states.append(true_step(states[-1], actions[:, t]))
    return np.stack(states, axis=1)


@torch.no_grad()
def rollout_model(model, start_states, actions):
    """在学到的模型里开环展开。形状同 rollout_true。

    这个函数是后面所有规划器的核心：规划就是对很多条候选动作序列同时做开环展开。
    """
    state = torch.as_tensor(start_states, dtype=torch.float32, device=DEVICE)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=DEVICE)
    states = [state]
    for t in range(actions.shape[1]):
        state = model.predict(state, actions[:, t])  # 把自己的输出当作下一步的输入
        states.append(state)
    return torch.stack(states, dim=1).cpu().numpy()


def angle_error(pred, truth):
    """两条状态序列的角度误差 [N, H+1]，按圆周上的距离算，最大是 π。"""
    return np.abs(wrap_angle(pred[..., 0] - truth[..., 0]))


def open_loop_error(model, n_traj, horizon):
    """随机起点、随机动作，比较模型的开环展开和真实轨迹。"""
    print("\n" + "=" * 60)
    print("第二部分：开环展开的误差随步数怎么长")
    print("=" * 60)

    print(f"""
实验：{n_traj} 条轨迹，起点按 Pendulum-v1 的 reset 分布取（角度任意，角速度在 ±1 内），
每条执行 {horizon} 步随机动作。模型和真实物理方程从同一个起点出发、执行同一串动作，
看两者的角度差多少。
    """)

    rng = np.random.default_rng(SEED + 1)
    starts = np.stack([rng.uniform(-np.pi, np.pi, n_traj), rng.uniform(-1.0, 1.0, n_traj)], axis=-1)
    actions = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=(n_traj, horizon, 1))
    truth = rollout_true(starts, actions)
    pred = rollout_model(model, starts, actions)
    err = angle_error(pred, truth)  # [N, H+1]

    steps = [s for s in (1, 5, 10, 20, 30, 50, 100) if s <= horizon]
    print(f"  {'展开步数':<8}{'平均角度误差':>12}{'中位数':>10}{'误差超过 0.5 弧度的轨迹':>24}")
    for s in steps:
        print(f"  {s:<10}{err[:, s].mean():>14.4f}{np.median(err[:, s]):>12.4f}{(err[:, s] > 0.5).mean():>20.1%}")

    one_step = err[:, 1].mean()
    h_mid = 50 if horizon >= 50 else horizon
    growth = err[:, h_mid].mean() / one_step
    diverged = err[:, h_mid] > 0.5

    # 发散的轨迹有什么共同点？看它们有没有经过顶部附近
    near_top = (np.abs(wrap_angle(truth[:, :h_mid + 1, 0])) < 0.5).any(axis=1)
    p_div_top = diverged[near_top].mean()
    p_div_other = diverged[~near_top].mean()
    print(f"\n  第 1 步的平均误差 {one_step:.5f} 弧度，第 {h_mid} 步是 {err[:, h_mid].mean():.3f} 弧度，放大了约 {growth:.0f} 倍。")
    print(f"\n  前 {h_mid} 步里经过顶部附近（|θ| < 0.5）的轨迹占 {near_top.mean():.0%}：")
    print(f"    经过顶部的轨迹，第 {h_mid} 步误差超过 0.5 弧度的比例：  {p_div_top:.0%}")
    print(f"    没经过顶部的轨迹，第 {h_mid} 步误差超过 0.5 弧度的比例：{p_div_other:.0%}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(1, horizon + 1)
    axes[0].plot(x, err[:, 1:].mean(axis=0), linewidth=2, label="平均值")
    axes[0].plot(x, np.median(err[:, 1:], axis=0), linewidth=2, label="中位数")
    axes[0].fill_between(x, np.percentile(err[:, 1:], 10, axis=0), np.percentile(err[:, 1:], 90, axis=0),
                         alpha=0.2, label="10%–90% 分位")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("展开步数")
    axes[0].set_ylabel("角度误差（弧度，对数坐标）")
    axes[0].set_title("开环展开的误差随步数增长")
    axes[0].legend()

    # 挑一条发散的轨迹画出来
    idx = int(np.argmax(diverged)) if diverged.any() else 0
    t = np.arange(horizon + 1)
    axes[1].plot(t, np.cos(truth[idx, :, 0]), "k--", linewidth=1.5, label="真实环境")
    axes[1].plot(t, np.cos(pred[idx, :, 0]), linewidth=2.5, alpha=0.7, label="模型开环展开")
    axes[1].set_xlabel("展开步数")
    axes[1].set_ylabel("摆尖高度 cos θ")
    axes[1].set_title("一条发散的轨迹：在顶部附近分道扬镳")
    axes[1].legend()
    save_figure(fig, OUT / "04-open-loop-error.png")

    print(f"""
三点观察：

    1. 单步误差小得可以忽略，开环 {h_mid} 步之后平均误差放大了约 {growth:.0f} 倍。
       验证损失再漂亮，也说明不了模型能往前推多远。
    2. 平均值远大于中位数。大多数轨迹的误差一直很小，少数轨迹的误差大到
       "完全不是一回事"，是这少数轨迹把平均值拉了上去。
    3. 这少数轨迹有一个共同点：都经过了顶部附近。
       摆冲到顶部时速度很慢，差一点点能量，结果就是"翻过去"和"掉回来"的区别。
       模型在那里只要有一丝误差，后面的轨迹就完全不同（见右图）。

第 3 点说明误差涨得快不快，和系统在哪个区域有关。下一部分把这件事定量算清楚。
    """)
    return err, growth, p_div_top, p_div_other


# ==================== 第三部分：稳定区与不稳定区 ====================


def stability_regions(model):
    """摆的顶部放大误差，底部不放大。先用理论算，再用实验验证。"""
    print("\n" + "=" * 60)
    print("第三部分：为什么顶部附近误差涨得快")
    print("=" * 60)

    print("""
CONCEPT.md 第 5.3 节推出了误差的递推式：

    e_{t+1} ≤ L · e_t + ε          L：系统把偏差放大多少倍；ε：模型的单步误差

L 大于 1 时误差按指数增长。L 是多少，由系统在当前位置的局部性质决定。
把摆的转移方程在某个角度附近线性化（不施力），得到一个 2×2 的矩阵，
它的特征值的模就是偏差每一步被放大的倍数：
    """)

    for name, theta0 in [("顶部 θ=0", 0.0), ("底部 θ=π", np.pi)]:
        # θ̇' = θ̇ + 15·cos(θ0)·δθ·dt，θ' = θ + θ̇'·dt
        k = 15.0 * np.cos(theta0) * DT
        jac = np.array([[1.0 + k * DT, DT], [k, 1.0]])
        eig = np.abs(np.linalg.eigvals(jac))
        print(f"  {name}：线性化矩阵 [[{jac[0, 0]:.4f}, {jac[0, 1]:.2f}], [{jac[1, 0]:+.2f}, {jac[1, 1]:.0f}]]，"
              f"特征值的模 = {eig.max():.3f}、{eig.min():.3f}")

    print("""
顶部的最大特征值是 1.213：偏差每一步放大约 21%。底部两个特征值的模都是 1：
偏差只会来回振荡，不会变大。

实验一：给真实物理方程加一个固定的单步误差 ε（每一步的角速度多算 ε），
把摆放在正上方、不施力。真实的摆会一直立着不动，带误差的模型会慢慢倒下去。
    """)

    print(f"  {'单步误差 ε':<12}{'第 10 步':>10}{'第 20 步':>10}{'第 30 步':>10}{'每步放大倍数':>14}{'误差超过 0.1 弧度的步数':>24}")
    cross_steps = []
    for eps in (1e-4, 1e-3, 1e-2):
        state = np.zeros((1, 2))
        errs = [0.0]
        for _ in range(80):
            state = true_step(state, np.zeros((1, 1)))
            state[:, 1] += eps  # 模型每一步都多算了一点角速度
            errs.append(abs(state[0, 0]))  # 真实的摆一直在 θ=0，所以 |θ| 就是误差
        errs = np.array(errs)
        cross = int(np.argmax(errs > 0.1))
        cross_steps.append(cross)
        rate = (errs[30] / errs[20]) ** 0.1  # 第 20 到 30 步之间，平均每步放大多少倍
        print(f"  {eps:<14.0e}{errs[10]:>10.5f}{errs[20]:>12.5f}{errs[30]:>12.5f}{rate:>14.3f}{cross:>22}")

    gap = np.diff(cross_steps)
    print(f"""
    - 误差每步放大 1.21 到 1.22 倍，和最大特征值 1.213 吻合。
    - 单步误差每缩小 10 倍，"误差超过 0.1 弧度"只推迟了 {abs(gap[0])} 步和 {abs(gap[1])} 步。
      理论值是 ln 10 / ln 1.213 ≈ 11.9 步。
      在不稳定的区域，把模型精度提高 10 倍，换来的只是多十来步的可用视野。

实验二：换成学到的模型。分别从顶部附近和底部附近出发，不施力，开环展开。
    """)

    n = 500
    rng = np.random.default_rng(SEED + 2)
    zero_actions = np.zeros((n, 60, 1))
    curves = {}
    print(f"  {'起点':<26}{'第 10 步':>10}{'第 20 步':>10}{'第 30 步':>10}{'第 40 步':>10}   （平均角度误差，弧度）")
    for name, center in [("顶部附近（θ ≈ 0 ± 0.3）", 0.0), ("底部附近（θ ≈ π ± 0.3）", np.pi)]:
        starts = np.stack([center + rng.uniform(-0.3, 0.3, n), rng.uniform(-0.5, 0.5, n)], axis=-1)
        err = angle_error(rollout_model(model, starts, zero_actions), rollout_true(starts, zero_actions)).mean(axis=0)
        curves[name] = err
        print(f"  {name:<22}{err[10]:>12.4f}{err[20]:>12.4f}{err[30]:>12.4f}{err[40]:>12.4f}")

    top, bottom = list(curves.values())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, err in curves.items():
        ax.plot(np.arange(1, 61), err[1:], linewidth=2, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("展开步数")
    ax.set_ylabel("平均角度误差（弧度，对数坐标）")
    ax.set_title("同一个模型，从不同区域出发，误差增长的快慢完全不同")
    ax.legend()
    save_figure(fig, OUT / "04-stable-vs-unstable.png")

    ratio = top[30] / bottom[30]
    print(f"""
同一个模型，第 30 步的误差：从顶部出发是从底部出发的 {ratio:.0f} 倍。

这对控制任务是个坏消息：我们最需要模型准的地方（把摆稳在顶部），
恰好是误差放大最快的地方。这不是巧合，需要控制的系统通常都是不稳定的，
稳定的系统放着不管也没事。
    """)
    return cross_steps, ratio


# ==================== 第四部分：分布偏移 ====================


def handcrafted_policy(state):
    """02 课的手写控制器：攒能量甩上去，再用比例—微分控制稳住。"""
    theta, theta_dot = wrap_angle(state[0]), state[1]
    if abs(theta) < 0.4 and abs(theta_dot) < 3.0:
        u = -10.0 * theta - 2.0 * theta_dot
    else:
        energy = 0.5 * theta_dot**2 + 15.0 * np.cos(theta)
        direction = 1.0 if theta_dot >= 0 else -1.0
        u = MAX_TORQUE * direction if energy < 15.0 else -MAX_TORQUE * direction
    return float(np.clip(u, -MAX_TORQUE, MAX_TORQUE))


def collect_controller_states(n_episodes):
    """用手写控制器跑几个回合，返回它经过的转移 (s, a, s')。"""
    env = gym.make("Pendulum-v1")
    s, a, s_next = [], [], []
    for ep in range(n_episodes):
        env.reset(seed=1000 + ep)
        for _ in range(200):
            state = env.unwrapped.state.copy()
            u = handcrafted_policy(state)
            env.step(np.array([u], dtype=np.float32))
            s.append(state)
            a.append([u])
            s_next.append(env.unwrapped.state.copy())
    env.close()
    return [torch.as_tensor(np.array(x, dtype=np.float32), device=DEVICE) for x in (s, a, s_next)]


@torch.no_grad()
def one_step_error(model, transitions):
    """单步预测的平均绝对误差 (角度, 角速度)。"""
    s, a, s_next = transitions
    return (model.predict(s, a) - s_next).abs().mean(0).cpu().numpy()


def distribution_shift(model, train, val, epochs, n_ctrl_episodes):
    """把上半圈的训练数据删掉，看模型在上半圈的表现。"""
    print("\n" + "=" * 60)
    print("第四部分：分布偏移——没见过的地方不可靠")
    print("=" * 60)

    print("""
误差递推式里的 ε 不是处处一样小，它只在训练数据覆盖的区域小。
做一个对照实验：把训练集里上半圈（|θ| < π/2）的数据全部删掉，重新训练一个模型。
然后在三组数据上比较两个模型的单步误差：

    验证集·下半圈    两个模型都见过类似的数据
    验证集·上半圈    "只见过下半圈"的模型没见过
    控制器的轨迹     02 课的手写控制器把摆甩上去并稳住，大部分时间待在顶部
    """)

    theta = wrap_angle(train[0][:, 0].cpu().numpy())
    upper_frac, n_top = (np.abs(theta) < np.pi / 2).mean(), int((np.abs(theta) < 0.5).sum())
    lower_mask = torch.as_tensor(np.abs(theta) >= np.pi / 2, device=DEVICE)
    lower_train = [t[lower_mask] for t in train]
    torch.manual_seed(SEED)
    lower_model = train_model(DynamicsModel(), lower_train, epochs)
    print(f"  完整训练集 {train[0].shape[0]} 条；删掉上半圈后剩 {lower_train[0].shape[0]} 条")

    val_theta = wrap_angle(val[0][:, 0].cpu().numpy())
    val_upper = torch.as_tensor(np.abs(val_theta) < np.pi / 2, device=DEVICE)
    ctrl = collect_controller_states(n_ctrl_episodes)
    ctrl_top = (np.abs(wrap_angle(ctrl[0][:, 0].cpu().numpy())) < 0.5).mean()
    groups = [
        ("验证集·下半圈", [t[~val_upper] for t in val]),
        ("验证集·上半圈", [t[val_upper] for t in val]),
        ("控制器的轨迹", ctrl),
    ]
    print(f"  控制器的轨迹里，{ctrl_top:.0%} 的状态在顶部附近（|θ| < 0.5）\n")
    print(f"  {'测试数据':<14}{'完整数据训练：θ 误差':>18}{'θ̇ 误差':>10}{'只见过下半圈：θ 误差':>22}{'θ̇ 误差':>10}{'θ 误差之比':>12}")
    ratios = []
    full_errs = []
    for name, group in groups:
        e_full, e_lower = one_step_error(model, group), one_step_error(lower_model, group)
        ratios.append(e_lower[0] / e_full[0])
        full_errs.append(e_full)
        print(f"  {name:<14}{e_full[0]:>20.5f}{e_full[1]:>12.5f}{e_lower[0]:>22.5f}{e_lower[1]:>12.5f}{ratios[-1]:>12.1f}")

    print(f"""
两点观察：

    1. 在下半圈，两个模型差不多（{ratios[0]:.1f} 倍）。到了没见过的上半圈，
       "只见过下半圈"的模型角度误差是完整模型的 {ratios[1]:.0f} 倍；
       在控制器的轨迹上是 {ratios[2]:.0f} 倍。神经网络在训练分布之外的预测不可信。
    2. 用完整的随机数据训练的模型，在控制器的轨迹上误差并不比验证集大
       （θ 误差 {full_errs[2][0]:.5f}，验证集下半圈是 {full_errs[0][0]:.5f}）。
       随机策略虽然稳不住摆，但训练集里有 {upper_frac:.0%} 的数据落在上半圈，
       顶部附近（|θ| < 0.5）也有 {n_top} 条，对摆这么简单的系统已经够了。

第 2 点是这个任务的运气：状态只有 2 维，随机乱推就能把整个状态空间扫个大概。
状态维度一高（机械臂、图像），随机数据只能覆盖状态空间里极小的一块，
"规划器想去的地方模型没见过"就是常态。那时要靠第 08 课的办法：
让控制器自己去采数据，把它真正会去的地方补进训练集。

分布偏移和上一部分的误差放大会相互加强：误差把展开的状态带离数据分布，
离开数据分布后单步误差更大，更大的误差带得更远。
    """)
    return ratios, full_errs


# ==================== 第五部分：多步损失 ====================


def multistep_loss(model, train, seq_states, seq_actions, epochs, n_traj, horizon, k=10):
    """训练时也做开环展开，看开环误差有没有改善。"""
    print("\n" + "=" * 60)
    print("第五部分：多步损失有没有用")
    print("=" * 60)

    print(f"""
一个直接的想法：既然使用时要开环展开，训练时也展开。
从真实状态出发在模型里展开 k = {k} 步，每一步的预测都和真实状态比：

    L_multi = (1/k) · Σ_j ‖ŝ_t+j − s_t+j‖²       ŝ_t+j = f(ŝ_t+j-1, a_t+j-1)，ŝ_t = s_t

梯度沿着展开的链条反向传播，和训练 RNN 的 BPTT 是同一个机制。
用同样的数据、同样的训练轮数训练一个多步损失的模型，和单步损失的模型比开环误差。
    """)

    torch.manual_seed(SEED)
    multi_model = train_model(DynamicsModel(), train, epochs, multistep=(seq_states, seq_actions, k))

    rng = np.random.default_rng(SEED + 1)  # 和第二部分用同一组起点和动作
    starts = np.stack([rng.uniform(-np.pi, np.pi, n_traj), rng.uniform(-1.0, 1.0, n_traj)], axis=-1)
    actions = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=(n_traj, horizon, 1))
    truth = rollout_true(starts, actions)
    err_single = angle_error(rollout_model(model, starts, actions), truth)
    err_multi = angle_error(rollout_model(multi_model, starts, actions), truth)

    steps = [s for s in (1, 10, 30, 50, 100) if s <= horizon]
    print(f"  {'展开步数':<8}{'单步损失：平均':>14}{'中位数':>10}{'多步损失：平均':>16}{'中位数':>10}")
    for s in steps:
        print(f"  {s:<10}{err_single[:, s].mean():>14.4f}{np.median(err_single[:, s]):>12.4f}"
              f"{err_multi[:, s].mean():>16.4f}{np.median(err_multi[:, s]):>12.4f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, horizon + 1)
    ax.plot(x, err_single[:, 1:].mean(axis=0), linewidth=2, label="单步损失")
    ax.plot(x, err_multi[:, 1:].mean(axis=0), linewidth=2, linestyle="--", label=f"多步损失（k = {k}）")
    ax.set_yscale("log")
    ax.set_xlabel("展开步数")
    ax.set_ylabel("平均角度误差（弧度，对数坐标）")
    ax.set_title("两种损失训练出的模型，开环误差对比")
    ax.legend()
    save_figure(fig, OUT / "04-multistep-loss.png")

    h = steps[-2] if len(steps) >= 2 else steps[-1]
    change = err_multi[:, h].mean() / err_single[:, h].mean() - 1
    print(f"""
结果：两条曲线几乎重合。第 {h} 步的平均误差，多步损失相对单步损失的变化是 {change:+.0%}。
在这个任务上，多步损失没有带来看得出来的改善。

原因在第二、三部分已经看到了：
    - 这个模型的单步误差本来就很小，开环误差主要来自少数经过顶部的轨迹。
      那里的放大倍数是系统本身的性质，不是损失函数能改的。
    - 在不稳定区域，精度提高 10 倍也只多换来十来步。多步损失对精度的改善远没有 10 倍。

多步损失不是没用。它在这些情况下有帮助：单步模型有系统性的偏差
（比如在学出来的隐空间里展开，Phase 3 的 latent overshooting 就是多步损失）；
或者观测带噪声，单步目标本身就不准。
这里如实记下一个没有改善的结果：一个办法在论文里有效，不代表在你的任务上有效，
要用开环误差这样贴近实际用法的指标去量。
    """)
    return change


# ==================== 第六部分：小结与练习 ====================


def summary(err):
    """怎么和复合误差相处。"""
    print("\n" + "=" * 60)
    print("第六部分：怎么和复合误差相处")
    print("=" * 60)

    e1, e10 = err[:, 1].mean(), err[:, 10].mean()
    e30 = err[:, 30].mean() if err.shape[1] > 30 else float("nan")
    print(f"""
复合误差消除不了，只能绕着走。回头看第二部分的表：

    只信模型 1 步    平均误差 {e1:.4f} 弧度
    只信模型 10 步   平均误差 {e10:.4f} 弧度
    只信模型 30 步   平均误差 {e30:.4f} 弧度

办法就藏在这张表里：不要让模型开环跑太远。每走一步都回到真实状态重新出发，
误差就没有机会滚起来。这正是下一节的模型预测控制（MPC）：

    ┌──────────────────────┬───────────────────────────────────────────┬────────┐
    │ 办法                  │ 原理                                       │ 在哪    │
    ├──────────────────────┼───────────────────────────────────────────┼────────┤
    │ 短视野 + 每步重新规划   │ 只信模型往前推的几十步，每步用真实状态清零误差   │ 05 课   │
    │ 估计不确定性           │ 让模型知道自己哪里不准，规划时打折扣            │ 07 课   │
    │ 迭代采集数据           │ 让控制器去采数据，补上它会去的区域              │ 08 课   │
    │ 多步损失              │ 把多步误差放进训练目标（本节：在摆上没有改善）    │ Phase 3 │
    │ 换预测空间            │ 在更容易预测的隐空间里展开                     │ Phase 3 │
    └──────────────────────┴───────────────────────────────────────────┴────────┘
    """)


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：手算误差上界
    单步误差 ε = 0.001，放大倍数 L = 1.213。按 e_t ≤ ε·(L^t − 1)/(L − 1)，
    第 10 步和第 30 步的误差上界各是多少？和第三部分实验一的 ε = 1e-3 那一行比。

练习 1 参考答案：
    1.213^10 = 6.90，上界 = 0.001 × 5.90 / 0.213 = 0.0277。
    1.213^30 = 328，上界 = 0.001 × 327 / 0.213 = 1.54。
    实验一的实测值是 0.0029 和 0.198，都在上界之内。ε 是加在角速度上的，
    传到角度上还要乘时间步长 0.05，所以实测值比上界小一截。
    增长的倍数是同一个量级：从第 10 步到第 30 步，上界放大 55 倍，实测放大 67 倍。

练习 2：有动作时的不稳定
    第三部分的实验都不施力。把实验二改成施加随机力矩，
    顶部和底部的差距会变大还是变小？

练习 2 参考答案：
    实测差距基本不变。第 30 步的平均误差：从顶部出发 0.079，从底部出发 0.007，
    和不施力时（0.078 和 0.008）几乎一样。
    随机力矩忽正忽负，平均下来给不了摆多少能量。从底部出发的摆在几十步内
    到不了上半圈，一直待在稳定区里；从顶部出发的摆不管推不推都会倒下去。
    决定误差涨多快的是轨迹经过了哪些区域，随机动作改变不了这一点。

练习 3：误差涨到多少算"不能用"
    第二部分的图里，10%–90% 分位的带子在几十步之后变得非常宽。
    如果规划器要比较两条动作序列的好坏，这意味着什么？

练习 3 参考答案：
    规划器比较的是"模型认为的回报"。如果模型对同一条动作序列的预测
    可能偏出 0.5 弧度以上，两条序列的排序就不可信了。
    规划视野要选在误差带还比较窄的范围内。对这个模型，大约是二三十步。
    下一节会实测不同视野下 MPC 的表现。

思考题 1：真实世界里的"顶部"是什么？

思考题 1 参考答案：
    任何"差一点结果就完全不同"的时刻。抓取时手指刚接触物体的瞬间、
    走路时单脚支撑的阶段、推一个快要倒的物体。这些时刻决定了任务成败，
    也是世界模型最容易错的地方。Phase 9 的 DIAMOND 论文强调"视觉细节重要"，
    说的也是这类时刻：画面上很小的差别对应着完全不同的后续。

思考题 2：既然开环展开不可靠，为什么 Dreamer（Phase 6）敢在模型里想象 15 步来训练策略？

思考题 2 参考答案：
    三个原因。一是视野短，15 步在多数任务上还在可用范围内；
    二是想象的起点是真实数据里的状态，每条想象轨迹都从"对的地方"出发；
    三是策略是闭环的，每一步根据模型当前的状态选动作，
    而且价值函数把 15 步之后的事概括成一个数，不需要模型继续往下推。

思考题 3：能不能让模型一次直接预测 30 步之后的状态，绕开误差累积？

思考题 3 参考答案：
    可以，但代价很大。直接预测 s_{t+30} 需要把 30 个动作都当输入，
    输入维度变大，函数变复杂，需要的数据多得多；而且视野一变就要重新训练。
    单步模型的好处是可以任意组合：一个模型，想推几步推几步。
    折中的做法是预测一小段（比如动作分块），Phase 11 会讲。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()

    if is_quick():
        n_episodes, n_val, epochs, n_traj, horizon, n_ctrl = 10, 2, 5, 100, 50, 3
    else:
        n_episodes, n_val, epochs, n_traj, horizon, n_ctrl = 50, 10, 100, 1000, 100, 30
    seq_states, seq_actions = collect_random_dataset(n_episodes)
    n_train = n_episodes - n_val
    train = to_transitions(seq_states[:n_train], seq_actions[:n_train])
    val = to_transitions(seq_states[n_train:], seq_actions[n_train:])

    torch.manual_seed(SEED)
    model = train_model(DynamicsModel(), train, epochs)
    val_err = one_step_error(model, val)
    print(f"\n  模型训练完毕。验证集单步误差：角度 {val_err[0]:.5f} 弧度，角速度 {val_err[1]:.5f} 弧度/秒")

    err, _, _, _ = open_loop_error(model, n_traj, horizon)
    stability_regions(model)
    distribution_shift(model, train, val, epochs, n_ctrl)
    multistep_loss(model, train, seq_states[:n_train], seq_actions[:n_train], epochs, n_traj, horizon)
    summary(err)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：05-random-shooting-mpc.py（随机打靶与模型预测控制）")
    print("=" * 60)


if __name__ == "__main__":
    main()
