"""
用神经网络学动力学模型
======================

学习目标：
    1. 把"学世界模型"写成一个监督学习问题：输入 (状态, 动作)，标签是下一个状态
    2. 掌握三个决定成败的处理：预测变化量、归一化、角度用 (cos, sin) 表示
    3. 用消融实验确认每个处理各自贡献了多少
    4. 学会按轨迹切分训练集 / 验证集，并在物理单位下报告误差
    5. 用真实物理给模型打分，看到误差和数据覆盖的关系：没数据的地方误差大

核心概念：
    - 动力学模型 f_θ：ŝ' = s + f_θ(s, a)，网络输出的是状态的变化量 Δs
    - 标准化：输入和目标各自减均值、除以标准差；统计量是模型的一部分
    - 角度的表示：输入用 (cos θ, sin θ)，输出用 Δθ
    - 单步误差：从真实状态出发只预测一步的误差（多步的事下一节再说）

前置知识：
    - 02-gymnasium-and-mdp.py（转移数据集）
    - deep-learning-course 的 MLP 与训练循环

对应概念文档：
    - CONCEPT.md 第 4 节「学一个动力学模型：把物理变成监督学习」

预计运行时间：约 30 秒（MPS）
"""

import os
import sys
import time

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
DEVICE = get_device()


# ==================== 第一部分：把物理变成监督学习 ====================


def introduction():
    """学动力学模型 = 回归。"""
    print("=" * 60)
    print("第一部分：把物理变成监督学习")
    print("=" * 60)

    print("""
假设我们不知道摆的物理方程，只有上一节采到的交互记录。这是一份带标签的数据集：

    输入 x = (状态, 动作)           标签 y = 下一个状态
    ─────────────────────         ─────────────────
    (θ=3.00, θ̇=0.50, u=1.0)       (θ=3.04, θ̇=0.76)
    (θ=0.20, θ̇=-1.3, u=-2.0)      (θ=0.12, θ̇=-1.45)
    ...                           ...

学世界模型在这里就是回归，和预测房价没有本质区别。
直接拿 MLP 去拟合能跑，但效果很差。本节的模型多做三件事：

    ┌─────────────┐   ┌──────────────────┐   ┌─────────┐   ┌──────────────┐   ┌────────┐
    │ 状态 (θ, θ̇) │──▶│ 特征              │──▶│ 标准化   │──▶│ MLP          │──▶│ 反标准化 │──▶ Δs
    │ 动作 u      │   │ (cos θ, sin θ,    │   │ (x-μ)/σ │   │ 4→200→200→2  │   │ ·σ + μ │
    └─────────────┘   │  θ̇, u)           │   └─────────┘   └──────────────┘   └────────┘
          │           └──────────────────┘                                          │
          │                 ③ 角度用 cos/sin        ② 归一化                          ▼
          └──────────────────────────────────────────────────────────────────▶ ŝ' = s + Δs
                                                                              ① 预测变化量

    ① 预测变化量：一步之内状态变化很小，让网络只学"变了多少"
    ② 归一化：各个量的数值范围差好几倍，不处理的话小的那一维学不到
    ③ 角度用 (cos θ, sin θ)：θ 和 θ+2π 是同一个位置，要让网络看到的输入也相同

下面先训练完整的模型，再把这三件事逐个拿掉，看各自值多少。
    """)
    print(f"  使用设备：{DEVICE}")


# ==================== 第二部分：数据 ====================


def collect_random_dataset(n_episodes, seed=SEED):
    """用随机策略采集转移数据（与 02 课相同）。返回 states [M,2]、actions [M,1]、next_states [M,2]、episode [M]。"""
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)
    states, actions, next_states, episode = [], [], [], []
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        done = False
        while not done:
            states.append(env.unwrapped.state.copy())
            action = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=1).astype(np.float32)
            _, _, terminated, truncated, _ = env.step(action)
            actions.append(action)
            next_states.append(env.unwrapped.state.copy())
            episode.append(ep)
            done = terminated or truncated
    env.close()
    return (np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32),
            np.array(next_states, dtype=np.float32), np.array(episode))


def split_by_episode(arrays, episode, n_val_episodes):
    """按回合切分：最后 n_val_episodes 个回合做验证集。

    不能把所有转移混在一起随机切。同一个回合里相邻的两条转移几乎一样，
    随机切会让验证集里全是训练集的"近似重复"，验证误差偏低，看不出过拟合。
    """
    val_mask = episode >= episode.max() + 1 - n_val_episodes
    train = [torch.as_tensor(a[~val_mask], device=DEVICE) for a in arrays]
    val = [torch.as_tensor(a[val_mask], device=DEVICE) for a in arrays]
    return train, val


# ==================== 第三部分：动力学模型 ====================


class DynamicsModel(nn.Module):
    """状态动力学模型：给定 (θ, θ̇) 和力矩，预测下一个 (θ, θ̇)。

    三个开关对应三个处理，默认全开；消融实验里逐个关掉。
        predict_delta:  True 时网络输出变化量 Δs，False 时直接输出下一状态
        normalize:      True 时对输入和目标做标准化
        angle_features: True 时输入用 (cos θ, sin θ)，False 时直接输入 θ
    """

    def __init__(self, hidden=200, predict_delta=True, normalize=True, angle_features=True):
        super().__init__()
        self.predict_delta = predict_delta
        self.normalize = normalize
        self.angle_features = angle_features
        in_dim = 4 if angle_features else 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        # 归一化统计量是模型的一部分：注册成 buffer，会跟着 state_dict 一起保存
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(2))
        self.register_buffer("y_std", torch.ones(2))

    def features(self, states, actions):
        """[B, 2], [B, 1] → 输入特征 [B, 4]（或 [B, 3]）。"""
        theta, theta_dot = states[:, :1], states[:, 1:]
        if self.angle_features:
            return torch.cat([torch.cos(theta), torch.sin(theta), theta_dot, actions], dim=-1)
        return torch.cat([theta, theta_dot, actions], dim=-1)

    def targets(self, states, next_states):
        """回归目标 [B, 2]：变化量，或者下一状态本身。"""
        return next_states - states if self.predict_delta else next_states

    def fit_normalizer(self, states, actions, next_states):
        """在训练集上统计输入和目标各维的均值、标准差。"""
        if not self.normalize:
            return
        x, y = self.features(states, actions), self.targets(states, next_states)
        self.x_mean.copy_(x.mean(0))
        self.x_std.copy_(x.std(0).clamp_min(1e-6))
        self.y_mean.copy_(y.mean(0))
        self.y_std.copy_(y.std(0).clamp_min(1e-6))

    def loss(self, states, actions, next_states):
        """标准化空间里的均方误差。"""
        x = (self.features(states, actions) - self.x_mean) / self.x_std
        y = (self.targets(states, next_states) - self.y_mean) / self.y_std
        return ((self.net(x) - y) ** 2).mean()

    def predict(self, states, actions):
        """[B, 2], [B, 1] → 预测的下一状态 [B, 2]。"""
        x = (self.features(states, actions) - self.x_mean) / self.x_std
        out = self.net(x) * self.y_std + self.y_mean  # 反标准化
        return states + out if self.predict_delta else out


def train_model(model, train, val, epochs, batch_size=256, lr=1e-3):
    """标准的小批量训练循环。返回每个 epoch 的 (训练损失, 验证损失)。

    学习率按余弦曲线从 lr 降到 0：后期步子小，结果更稳，不同随机种子之间差别也小。
    """
    model.to(DEVICE)
    model.fit_normalizer(*train)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = train[0].shape[0]
    steps_per_epoch = (n + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * steps_per_epoch)
    history = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = model.loss(*[t[idx] for t in train])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(idx)
        with torch.no_grad():
            val_loss = model.loss(*val).item()
        history.append((total / n, val_loss))
    return history


@torch.no_grad()
def one_step_rmse(model, states, actions, next_states):
    """单步预测的均方根误差，物理单位：(角度误差 rad, 角速度误差 rad/s)。"""
    err = model.predict(states, actions) - next_states
    return err.pow(2).mean(0).sqrt().cpu().numpy()


def true_step(states, actions):
    """真实的物理方程（NumPy 批量版，与 01 课的 pendulum_step 相同）。

    只用来给模型打分。训练时模型看不到它，只看得到数据。
    """
    theta, theta_dot = states[:, 0], states[:, 1]
    u = np.clip(actions[:, 0], -MAX_TORQUE, MAX_TORQUE)
    theta_dot_new = np.clip(theta_dot + (15.0 * np.sin(theta) + 3.0 * u) * 0.05, -8.0, 8.0)
    return np.stack([theta + theta_dot_new * 0.05, theta_dot_new], axis=-1)


# ==================== 第四部分：训练完整模型 ====================


def train_full_model(train, val, epochs):
    """训练三个处理全开的模型，报告单步误差。"""
    print("\n" + "=" * 60)
    print("第二部分：训练完整的模型")
    print("=" * 60)

    n_train, n_val = train[0].shape[0], val[0].shape[0]
    print(f"\n  训练集 {n_train} 条转移，验证集 {n_val} 条转移（按回合切分）")

    torch.manual_seed(SEED)
    model = DynamicsModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  网络：4 → 200 → 200 → 2，共 {n_params:,} 个参数")

    start = time.time()
    history = train_model(model, train, val, epochs)
    elapsed = time.time() - start
    print(f"  训练 {epochs} 个 epoch，用时 {elapsed:.1f} 秒")
    for ep in sorted({0, epochs // 4, epochs // 2, epochs - 1}):
        print(f"    epoch {ep + 1:>3}：训练损失 {history[ep][0]:.2e}   验证损失 {history[ep][1]:.2e}")

    # 标准化空间的损失不直观，换算回物理单位
    states, actions, next_states = val
    rmse = one_step_rmse(model, *val)
    # 参照：一个"什么都不学"的模型，预测"下一状态 = 当前状态"
    naive = (states - next_states).pow(2).mean(0).sqrt().cpu().numpy()
    val_mse = history[-1][1]
    print(f"\n  验证集上的单步误差（均方根）：")
    print(f"    {'':<22}{'角度 θ（rad）':>14}{'角速度 θ̇（rad/s）':>20}")
    print(f"    {'预测「状态不变」':<18}{naive[0]:>14.5f}{naive[1]:>18.5f}")
    print(f"    {'学到的模型':<20}{rmse[0]:>14.5f}{rmse[1]:>18.5f}")
    print(f"    标准化空间的验证 MSE：{val_mse:.2e}（1.0 相当于只会预测平均值）")
    print(f"""
怎么读这些数字：

    - 角度的单步误差约 {rmse[0]:.4f} 弧度，也就是 {np.degrees(rmse[0]):.2f} 度。
      一步之内角度本身平均变化 {naive[0]:.3f} 弧度，模型的误差是它的 {rmse[0] / naive[0]:.1%}。
    - 标准化空间的 MSE 是 {val_mse:.1e}：模型解释了变化量里 {1 - val_mse:.2%} 的方差。

单看这一步，模型已经相当准了。但规划要把模型连续用几十步，
这点误差会被放大多少，是下一节的主题。
    """)

    # 拿 CONCEPT.md 第 4.3 节手算过的那个例子核对一下
    s = torch.tensor([[3.0, 0.5]], device=DEVICE)
    a = torch.tensor([[1.0]], device=DEVICE)
    with torch.no_grad():
        pred = model.predict(s, a)[0].cpu().numpy()
    print("  核对 CONCEPT.md 第 4.3 节的手算例子：状态 (θ=3.0, θ̇=0.5)，力矩 1.0")
    print(f"    物理方程算出的下一状态：(3.038, 0.756)")
    print(f"    模型预测的下一状态：    ({pred[0]:.3f}, {pred[1]:.3f})")
    return model, history, rmse, naive, val_mse


# ==================== 第五部分：模型学到了什么 ====================


def inspect_model(model, history, train, val):
    """把学到的函数和真实物理对比，并看误差和数据覆盖的关系。"""
    print("\n" + "=" * 60)
    print("第三部分：模型学到了什么，又在哪里不准")
    print("=" * 60)

    print("""
真实的物理规律是（01 课的 pendulum_step）：

    Δθ̇ = (15·sin θ + 3·u) × 0.05

固定 θ̇ = 0、u = 0，让 θ 扫过一整圈，模型预测的 Δθ̇ 应该是一条正弦曲线 0.75·sin θ。
网络从没见过这个公式，它只见过 (s, a, s') 的样本。
    """)

    theta_grid = torch.linspace(-np.pi, np.pi, 200, device=DEVICE)
    grid_states = torch.stack([theta_grid, torch.zeros_like(theta_grid)], dim=-1)
    grid_actions = torch.zeros(200, 1, device=DEVICE)
    with torch.no_grad():
        pred_delta = (model.predict(grid_states, grid_actions) - grid_states)[:, 1].cpu().numpy()
    true_delta = 0.75 * np.sin(theta_grid.cpu().numpy())
    curve_err = np.abs(pred_delta - true_delta).max()
    print(f"  学到的曲线与 0.75·sin θ 的最大偏差：{curve_err:.4f}（曲线本身的幅度是 0.75）")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    epochs = np.arange(1, len(history) + 1)
    axes[0].plot(epochs, [h[0] for h in history], label="训练损失")
    axes[0].plot(epochs, [h[1] for h in history], label="验证损失")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("标准化空间的 MSE（对数坐标）")
    axes[0].set_title("训练曲线")
    axes[0].legend()
    axes[1].plot(theta_grid.cpu().numpy(), true_delta, "k--", linewidth=1.5, label="物理方程：0.75·sin θ")
    axes[1].plot(theta_grid.cpu().numpy(), pred_delta, linewidth=3, alpha=0.6, label="学到的模型")
    axes[1].set_xlabel("角度 θ（固定 θ̇ = 0，u = 0）")
    axes[1].set_ylabel("一步的角速度变化 Δθ̇")
    axes[1].set_title("模型从数据里学出了重力项")
    axes[1].legend()
    save_figure(fig, OUT / "03-learned-dynamics.png")

    print("""
验证集和训练集来自同一个随机策略，分布相同。要知道模型在"没见过的地方"表现如何，
得换一种考法：在整个状态空间上均匀撒点，用真实物理方程算出正确答案，再和模型比。
同时统计训练数据在状态空间每个格子里有多少条。
    """)

    n_bins = 32
    theta_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    speed_edges = np.linspace(-8.0, 8.0, n_bins + 1)
    train_states = train[0].cpu().numpy()
    train_theta = ((train_states[:, 0] + np.pi) % (2 * np.pi)) - np.pi
    counts, _, _ = np.histogram2d(train_theta, train_states[:, 1], bins=[theta_edges, speed_edges])

    rng = np.random.default_rng(SEED)
    n_test = 200_000
    test_states = np.stack([rng.uniform(-np.pi, np.pi, n_test), rng.uniform(-8.0, 8.0, n_test)], axis=-1).astype(np.float32)
    test_actions = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=(n_test, 1)).astype(np.float32)
    truth = true_step(test_states, test_actions)
    with torch.no_grad():
        pred = model.predict(torch.as_tensor(test_states, device=DEVICE),
                             torch.as_tensor(test_actions, device=DEVICE)).cpu().numpy()
    err = np.abs(pred - truth)

    ti = np.clip(np.digitize(test_states[:, 0], theta_edges) - 1, 0, n_bins - 1)
    vi = np.clip(np.digitize(test_states[:, 1], speed_edges) - 1, 0, n_bins - 1)
    cell_count = counts[ti, vi]  # 每个测试点所在的格子里有多少条训练数据
    clipped = np.abs(truth[:, 1]) >= 7.999  # 下一步的角速度撞到了上限 ±8

    print(f"  状态空间切成 {n_bins}×{n_bins} 个格子，其中 {(counts == 0).mean():.0%} 的格子里一条训练数据都没有。")
    print(f"\n  {'格子里的训练数据':<14}{'测试点占比':>10}{'角度平均误差':>14}{'角速度平均误差':>16}")
    buckets = [("0 条", cell_count == 0), ("1–4 条", (cell_count >= 1) & (cell_count < 5)),
               ("5–19 条", (cell_count >= 5) & (cell_count < 20)), ("20 条以上", cell_count >= 20)]
    bucket_err = []
    for name, mask in buckets:
        m = mask & ~clipped
        # 冒烟模式下数据少，有的档位可能是空的
        bucket_err.append(err[m].mean(axis=0) if m.any() else np.full(2, np.nan))
        print(f"  {name:<18}{mask.mean():>12.1%}{bucket_err[-1][0]:>16.5f}{bucket_err[-1][1]:>18.5f}")
    print(f"  （上表不含角速度撞到上限的测试点。那部分占 {clipped.mean():.1%}，角速度平均误差 {err[clipped, 1].mean():.4f}。）")

    err_map = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        for j in range(n_bins):
            m = (ti == i) & (vi == j)
            err_map[i, j] = err[m, 1].mean()
    extent = [-np.pi, np.pi, -8, 8]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    im0 = axes[0].imshow(np.log10(counts.T + 1), origin="lower", extent=extent, aspect="auto", cmap="viridis")
    fig.colorbar(im0, ax=axes[0], label="log10(训练数据条数 + 1)")
    axes[0].set_title("训练数据在状态空间里的分布")
    im1 = axes[1].imshow(np.log10(err_map.T), origin="lower", extent=extent, aspect="auto", cmap="magma")
    fig.colorbar(im1, ax=axes[1], label="log10(角速度的单步平均误差)")
    axes[1].set_title("模型的单步误差（用真实物理方程打分）")
    for ax in axes:
        ax.set_xlabel("角度 θ（0 是正上方，±π 是正下方）")
        ax.set_ylabel("角速度 θ̇")
        ax.grid(False)
    save_figure(fig, OUT / "03-error-vs-coverage.png")

    n_clip_train = int((train[2][:, 1].abs() >= 7.999).sum())
    n_clip_val = int((val[2][:, 1].abs() >= 7.999).sum())
    ratio_theta = bucket_err[0][0] / bucket_err[3][0]
    ratio_speed = bucket_err[0][1] / bucket_err[3][1]
    print(f"""
三点观察：

    1. 模型从样本里学出了重力项的正弦形状，没有人告诉它 sin 这个函数。
    2. 数据越少的格子误差越大。完全没有数据的格子，角度误差是数据充足的格子的
       {ratio_theta:.1f} 倍，角速度误差是 {ratio_speed:.1f} 倍。对照两张热力图：左图的暗区就是右图的亮区。
       模型只在有数据的地方准，这件事后面会一再出现。
    3. 角速度撞到上限 ±8 的地方误差最大。真实的转移函数在那里有一个折角
       （没到上限时正常加速，到了上限被截住），而神经网络是光滑的，折角附近拟合不好。
       这类转移在训练集里有 {n_clip_train} 条，在验证集里有 {n_clip_val} 条。
       第二部分里训练损失比验证损失还高，就是被这几十条转移拉高的。
    """)
    return curve_err, ratio_theta, ratio_speed


# ==================== 第六部分：三个处理各值多少 ====================


def ablation(train, val, epochs, full_model):
    """从最朴素的做法出发，逐个加上三个处理。"""
    print("\n" + "=" * 60)
    print("第四部分：消融实验——三个处理各值多少")
    print("=" * 60)

    print("""
从最朴素的做法出发：把 (θ, θ̇, u) 原样喂给网络，让它直接输出 (θ', θ̇')。
然后每次加一个处理，其余设置（网络、数据、训练轮数、随机种子）完全相同。

除了验证集上的单步误差，再加一项测试：把验证集里每个状态的角度加上 2π。
物理上这是同一个状态，摆的下一步变化应该完全一样。
    """)

    variants = [
        ("朴素：直接预测下一状态", dict(predict_delta=False, normalize=False, angle_features=False)),
        ("+ ① 预测变化量", dict(predict_delta=True, normalize=False, angle_features=False)),
        ("+ ③ 角度用 cos/sin", dict(predict_delta=True, normalize=False, angle_features=True)),
        ("+ ② 归一化（完整模型）", None),
    ]
    states, actions, next_states = val
    shift = torch.tensor([2 * np.pi, 0.0], device=DEVICE)
    results = []
    print(f"  {'做法':<24}{'θ 误差':>10}{'θ̇ 误差':>10}{'角度 +2π 后的 θ̇ 误差':>24}")
    for name, kwargs in variants:
        if kwargs is None:
            model = full_model  # 就是第二部分训练的那个模型，不重复训练
        else:
            torch.manual_seed(SEED)
            model = DynamicsModel(**kwargs)
            train_model(model, train, val, epochs)
        rmse = one_step_rmse(model, *val)
        shifted = one_step_rmse(model, states + shift, actions, next_states + shift)
        results.append((name, rmse, shifted))
        print(f"  {name:<22}{rmse[0]:>12.5f}{rmse[1]:>12.5f}{shifted[1]:>20.5f}")

    naive, delta, angle, full = [r[1] for r in results]
    print(f"""
怎么读这张表（误差是验证集上的均方根，越小越好）：

    ① 预测变化量：θ 误差从 {naive[0]:.4f} 降到 {delta[0]:.4f}，θ̇ 误差从 {naive[1]:.4f} 降到 {delta[1]:.4f}。
       直接预测下一状态时，网络得先学会把输入原样抄到输出，再叠加一个小修正；
       预测变化量时，输出 0 就已经是不错的起点。
    ③ 角度用 cos/sin：θ 误差从 {delta[0]:.4f} 降到 {angle[0]:.4f}，θ̇ 误差从 {delta[1]:.4f} 降到 {angle[1]:.4f}。
       更要紧的是最后一列：把角度加上 2π 之后，前两个模型的 θ̇ 误差涨到
       {results[0][2][1]:.2f} 和 {results[1][2][1]:.2f}（它们没见过这么大的 θ，只能乱猜），用了 cos/sin 的模型不受影响。
       展开很多步时摆可能转过不止一圈，这个缺陷是会真的碰上的。
    ② 归一化：θ 误差从 {angle[0]:.5f} 降到 {full[0]:.5f}，θ̇ 误差从 {angle[1]:.4f} 变到 {full[1]:.4f}。
       受益的主要是数值小的那一维：Δθ 的标准差约 0.18，Δθ̇ 约 0.55，
       不归一化时损失里 Δθ̇ 的份量是 Δθ 的 9 倍左右，网络优先拟合它。
       摆的各个量差得还不算悬殊。真实机器人的状态里既有弧度又有毫米、牛顿，
       量级能差出几个数量级，不归一化基本没法训练。
    """)
    return results


# ==================== 第七部分：练习与思考 ====================


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：数据量
    把训练数据从 40 个回合减到 5 个回合（1000 条转移），其余不变。
    验证集上的单步误差会变成多少？

练习 1 参考答案：
    实测：θ 误差从 0.0005 涨到 0.0010，θ̇ 误差从 0.0086 涨到 0.0139。
    数据少了 8 倍，误差只涨了不到 2 倍。对摆这样的低维系统，
    一千条转移就能把单步误差压到很低。状态维度一高，需要的数据量会迅速增加：
    要让每个维度都有覆盖，格子数是按维度指数增长的。

练习 2：网络大小
    把隐藏层宽度从 200 改成 32，再改成 512，比较验证误差。

练习 2 参考答案：
    实测（θ 误差 / θ̇ 误差）：宽度 32 是 0.0009 / 0.0079，宽度 200 是 0.0005 / 0.0086，
    宽度 512 是 0.0004 / 0.0088。32 的角度误差大一倍，200 和 512 没有区别。
    摆的动力学只是一个正弦加线性项，很小的网络就能表示。
    这里的瓶颈是数据覆盖，不是网络容量：没数据的区域，再大的网络也只能靠外推。

练习 3：随机切分的陷阱
    把 split_by_episode() 换成"所有转移打乱后随机取 20% 做验证集"，
    验证损失会比按回合切分更高还是更低？

练习 3 参考答案：
    通常的担心是随机切分会让验证损失偏低：相邻两步的状态几乎相同，
    验证集里的每个样本在训练集里都有"近邻"，过拟合会被掩盖。
    但在这份数据上实测，随机切分的验证损失反而更高（2.8e-4 对 1.3e-4）。
    原因是第三部分讲的那几十条撞到速度上限的转移：随机切分把其中 5 条
    分进了验证集，每条的误差是普通样本的几十倍，把平均值拉高了。
    而按回合切分时，验证集的 10 个回合里碰巧一条都没有。
    两点体会：
    - 这个模型没有过拟合（数据是确定性的，网络也不大），所以"近邻泄漏"
      在这里不明显。数据带噪声、模型容量大的时候它才会咬人。
    - 一个平均数可能被极少数样本主导。验证损失对不上预期时，
      先把逐样本的误差排个序看看。

思考题 1：摆的角速度被环境限制在 [-8, 8]。这对模型意味着什么？

思考题 1 参考答案：
    真实的转移函数在 θ̇ = ±8 处有一个"折角"：没到上限时正常加速，
    到了上限就被截住。神经网络是光滑函数，折角附近一定拟合得差一些。
    真实机器人里这类不光滑到处都是：关节限位、碰撞、摩擦从静到动的切换。
    接触和碰撞是世界模型最难学的部分之一。

思考题 2：归一化的统计量是在训练集上算的。如果之后数据分布变了
    （比如控制器开始把摆稳在顶点），会怎样？

思考题 2 参考答案：
    旧的统计量不再匹配新数据：比如稳在顶点时 θ̇ 几乎为 0，
    而旧的标准差是按"到处乱晃"算的。模型仍然能用，但数值尺度不再合适。
    所以第 08 课的闭环里，每次用新数据重新训练时，统计量也要重新算。

思考题 3：本节的模型是确定性的：一个输入对应一个输出。
    什么情况下这不够用？

思考题 3 参考答案：
    当同样的 (s, a) 可能导致不同的 s' 时。比如环境有随机扰动，
    或者状态里缺了信息（02 课的"只知道角度"）。这时确定性模型
    只能输出各种可能结果的平均，而平均值本身可能是一个不会出现的状态。
    第 07 课让模型输出一个分布（均值和方差），Phase 2 的 MDN 进一步输出多峰分布。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()

    n_episodes, n_val, epochs = (10, 2, 5) if is_quick() else (50, 10, 100)
    states, actions, next_states, episode = collect_random_dataset(n_episodes)
    train, val = split_by_episode([states, actions, next_states], episode, n_val)

    model, history, _, _, _ = train_full_model(train, val, epochs)
    inspect_model(model, history, train, val)
    ablation(train, val, epochs, model)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：04-compounding-error.py（多步展开与复合误差）")
    print("=" * 60)


if __name__ == "__main__":
    main()
