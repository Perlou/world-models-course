"""
n 步回报与 λ-return
===================

学习目标：
    1. 写出 n 步回报：前 n 步用真实奖励，之后用价值估计补上；n = 1 是 TD 目标，n 取到回合结束是蒙特卡洛
    2. 写出 λ-return 的定义（所有 n 步回报的几何加权平均）和它的有限视野版本
    3. 实现从后往前的递推 G^λ_t = r_t + γ·c_t·[(1−λ)·V(s_{t+1}) + λ·G^λ_{t+1}]，
       并验证它和按定义算的结果逐位相等
    4. 说清 λ = 0、λ = 1 分别退化成什么，以及 G^λ − V 等于 TD 误差的 (γλ) 折扣和（Phase 5 的 GAE）
    5. 知道这个函数就是 Phase 6 Dreamer 训练 critic 的目标

核心概念：
    - n 步回报（n-step return）G^(n)_t
    - λ-return G^λ_t：一个参数 λ 在 TD(0) 和蒙特卡洛之间连续调节
    - 继续标志（continue flag）c_t = 1 − d_t：回合在这一步之后终止时，把后面的东西全部抹掉
    - 停止梯度（stop-gradient）：λ-return 作为回归目标时当作常数

前置知识：
    - 03-monte-carlo-vs-td.py：TD 目标、蒙特卡洛目标、TD 误差、偏差与方差

对应概念文档：
    - CONCEPT.md 第 6 节「n 步回报与 λ-return」

预计运行时间：尚未实测（本文件是还没有运行过的草稿，定稿前要完整跑一遍并改写这一行）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-04")
SEED = 0
# 这一节的计算量很小，全部在 CPU 上算


# ==================== 第一部分：从两个极端到一族目标 ====================


def introduction():
    """n 步回报和 λ-return 的直觉。"""
    print("=" * 60)
    print("第一部分：TD(0) 和蒙特卡洛之间还有什么")
    print("=" * 60)

    print("""
上一节的两种目标是两个极端：

    TD(0)     只相信一步真实的奖励，之后全靠估计    → 传播慢、有偏、方差小
    蒙特卡洛   完全不相信估计，一路用真实奖励到底    → 传播快、无偏、方差大

折中的办法：先走 n 步真实的，再用估计补上剩下的。这就是 n 步回报（n-step return）：

    TD(0)       s_t ─r_t─▶ ● V(s_{t+1})
    2 步        s_t ─r_t─▶ s_{t+1} ─r_{t+1}─▶ ● V(s_{t+2})
    3 步        s_t ─r_t─▶ s_{t+1} ─r_{t+1}─▶ s_{t+2} ─r_{t+2}─▶ ● V(s_{t+3})
     ⋮
    蒙特卡洛     s_t ─r_t─▶ s_{t+1} ─r_{t+1}─▶  ···  ─▶ 回合结束（不用 V）

    ─r─▶ 是真实拿到的奖励      ● 是用价值估计补上的部分

    G^(n)_t = r_t + γ r_{t+1} + … + γ^(n-1) r_{t+n-1} + γ^n · V(s_{t+n})

n 取多少？取小了传播慢，取大了噪声大。λ-return 的回答是不选，全都要：
把 1 步、2 步、3 步……的回报做加权平均，步数每多一步，权重乘一个 λ。

    G^λ_t = (1−λ) · [ G^(1)_t + λ·G^(2)_t + λ²·G^(3)_t + … ]

    权重
     │ █  (1−λ)
     │ █  █  (1−λ)λ
     │ █  █  █  (1−λ)λ²
     │ █  █  █  █  ▆  ▅  ▄  ▃  ▂  ▁
     └──────────────────────────────▶ n          λ 小：接近 TD(0)；λ 大：接近蒙特卡洛
       1  2  3  4  5  6  7  8  …

这和指数滑动平均是同一个套路：与其纠结窗口取多长，不如用一个衰减系数把所有长度都包进来。
    """)


# ==================== 第二部分：三个函数 ====================
#
# 形状约定（和 CONCEPT.md 第 6.8 节一致）：
#     rewards    [..., H]      H 步的奖励 r_t … r_{t+H-1}
#     values     [..., H+1]    H+1 个状态的价值估计 V(s_t) … V(s_{t+H})，比 rewards 多一个（末端自举要用）
#     continues  [..., H]      c_k：这一步之后回合终止则为 0，否则为 1
# 前面的 ... 可以没有（一条轨迹），也可以是批维度 [B]。


def n_step_return(rewards, values, n, gamma):
    """从第 0 个位置出发的 n 步回报（CONCEPT.md 第 6.2 节）。

    G^(n) = Σ_{k<n} γ^k r_k + γ^n V(s_n)。要求 1 ≤ n ≤ H。返回形状 [...]。
    """
    G = sum(gamma**k * rewards[..., k] for k in range(n))
    return G + gamma**n * values[..., n]


def lambda_weights(H, lam):
    """有限视野下各个 n 步回报的权重：前 H−1 个是 (1−λ)λ^(n−1)，最后一个拿剩下的 λ^(H−1)。返回 [H]。"""
    weights = (1.0 - lam) * lam ** np.arange(H)  # 下标 n−1 处是 (1−λ)λ^(n−1)
    weights[-1] = lam ** (H - 1)
    return weights


def lambda_return_naive(rewards, values, gamma, lam):
    """按定义算 λ-return（CONCEPT.md 第 6.3 节）：每个位置都先算出全部 n 步回报，再加权。

    只用来对照：慢，而且不处理回合终止。返回形状 [..., H]。
    """
    H = rewards.shape[-1]
    out = []
    for t in range(H):
        h = H - t  # 从位置 t 到数据末尾还剩几步
        weights = lambda_weights(h, lam)
        out.append(sum(weights[n - 1] * n_step_return(rewards[..., t:], values[..., t:], n, gamma)
                       for n in range(1, h + 1)))
    return np.stack(out, axis=-1)


def lambda_return(rewards, values, continues, gamma, lam):
    """λ-return 的递推实现（CONCEPT.md 第 6.4、6.7 节）。从后往前扫一遍，O(H)。

        G_H = V(s_H)
        G_k = r_k + γ · c_k · [ (1−λ) · V(s_{k+1}) + λ · G_{k+1} ]

    NumPy 数组和 PyTorch 张量都能用。返回形状 [..., H]，是 values[..., :-1] 的回归目标。
    """
    H = rewards.shape[-1]
    G = values[..., H]  # 末端只能用估计
    out = [None] * H
    for k in reversed(range(H)):
        G = rewards[..., k] + gamma * continues[..., k] * ((1.0 - lam) * values[..., k + 1] + lam * G)
        out[k] = G
    if isinstance(G, torch.Tensor):
        return torch.stack(out, dim=-1)
    return np.stack(out, axis=-1)


def unit_tests():
    """用 CONCEPT.md 第 6.5 节的手算例子核对，再用随机数据验证两种实现相等。"""
    print("\n" + "=" * 60)
    print("第二部分：实现并核对")
    print("=" * 60)

    gamma, lam = 0.9, 0.8
    rewards = np.array([1.0, 2.0, 3.0])
    values = np.array([np.nan, 5.0, 4.0, 6.0])  # V(s_0) 用不到，故意放一个 NaN：结果里没有 NaN 就说明确实没用
    continues = np.ones(3)

    n_steps = [n_step_return(rewards, values, n, gamma) for n in (1, 2, 3)]
    weights = lambda_weights(3, lam)
    recursive = lambda_return(rewards, values, continues, gamma, lam)
    naive = lambda_return_naive(rewards, values, gamma, lam)
    lam0 = lambda_return(rewards, values, continues, gamma, 0.0)
    lam1 = lambda_return(rewards, values, continues, gamma, 1.0)

    # 逐项对上 CONCEPT.md 第 6.5 节的手算
    assert np.allclose(n_steps, [5.5, 6.04, 9.604])
    assert np.allclose(weights, [0.2, 0.16, 0.64]) and np.isclose(weights.sum(), 1.0)
    assert np.allclose(recursive, [1 + 0.9 * 8.0144, 8.768, 8.4])  # G_0 = 8.21296，手算四舍五入成 8.213
    assert np.isclose(np.dot(weights, n_steps), recursive[0])
    assert np.allclose(naive, recursive) and not np.isnan(recursive).any()
    assert np.isclose(lam0[0], 5.5) and np.isclose(lam1[0], 9.604)

    print(f"""
手算例子（CONCEPT.md 第 6.5 节）：一段 3 步的轨迹，γ = {gamma}，λ = {lam}

    下标 k           0       1       2       3
    奖励 r_k         1       2       3
    价值 V(s_k)     (不用)    5       4       6

  方法一：按递推式，从后往前（lambda_return）
      G_3 = V(s_3) = 6
      G_2 = 3 + 0.9 × [0.2 × 6 + 0.8 × 6]       = {recursive[2]:.4f}
      G_1 = 2 + 0.9 × [0.2 × 4 + 0.8 × {recursive[2]:.1f}]     = {recursive[1]:.4f}
      G_0 = 1 + 0.9 × [0.2 × 5 + 0.8 × {recursive[1]:.3f}]   = {recursive[0]:.4f}

  方法二：按定义，先算 n 步回报再加权（lambda_return_naive）
      1 步回报 {n_steps[0]:.3f}，2 步回报 {n_steps[1]:.3f}，3 步回报 {n_steps[2]:.3f}
      权重 {weights[0]:.2f}、{weights[1]:.2f}、{weights[2]:.2f}（最后一个拿剩下的 λ² = 0.64，和为 {weights.sum():.0f}）
      G_0 = {weights[0]:.2f} × {n_steps[0]:.3f} + {weights[1]:.2f} × {n_steps[1]:.3f} + {weights[2]:.2f} × {n_steps[2]:.3f} = {naive[0]:.4f}

  两种方法在三个位置上都相等：{np.array2string(naive, precision=4)}。和文档里的 8.213、8.768、8.4 一致。

  换不同的 λ：
      λ = 0    G_0 = {lam0[0]:.3f}    就是 1 步回报（TD 目标）
      λ = 0.8  G_0 = {recursive[0]:.3f}    三者的加权平均
      λ = 1    G_0 = {lam1[0]:.3f}    就是 3 步回报：一路用真实奖励到数据末尾，再接上末端的 V(s_3)
  这个例子里 V(s_1) = 5 明显偏低。λ = 0 完全相信它；λ 越大，越多地用后面实际的奖励去纠正它。
    """)

    # 随机数据：两种实现应该处处相等
    rng = np.random.default_rng(SEED)
    B, H = 4, 15
    rewards = rng.normal(size=(B, H))
    values = rng.normal(size=(B, H + 1))
    continues = np.ones((B, H))
    max_gap = 0.0
    for g, l in [(0.9, 0.0), (0.9, 0.5), (0.99, 0.8), (0.997, 0.95), (1.0, 1.0)]:
        a = lambda_return(rewards, values, continues, g, l)
        b = lambda_return_naive(rewards, values, g, l)
        assert a.shape == (B, H) and np.allclose(a, b)
        max_gap = max(max_gap, np.abs(a - b).max())
    print(f"  随机数据（rewards {rewards.shape}，values {values.shape}，5 组 γ、λ）：两种实现的最大差距 {max_gap:.1e}。")

    # 两个极端
    g, l = 0.95, 0.9
    td_targets = rewards + g * values[:, 1:]
    h_step = np.stack([n_step_return(rewards[:, t:], values[:, t:], H - t, g) for t in range(H)], axis=-1)
    assert np.allclose(lambda_return(rewards, values, continues, g, 0.0), td_targets)
    assert np.allclose(lambda_return(rewards, values, continues, g, 1.0), h_step)
    print("  λ = 0 时处处等于 TD 目标 r_k + γ V(s_{k+1})；λ = 1 时处处等于「走到数据末尾再自举」的回报。")

    # 和 TD 误差的关系：G^λ_t − V(s_t) = Σ_k (γλ)^k δ_{t+k}
    deltas = rewards + g * values[:, 1:] - values[:, :-1]  # [B, H]
    advantage = np.zeros((B, H))
    acc = np.zeros(B)
    for k in reversed(range(H)):
        acc = deltas[:, k] + g * l * acc
        advantage[:, k] = acc
    assert np.allclose(lambda_return(rewards, values, continues, g, l) - values[:, :-1], advantage)
    print("  G^λ_t − V(s_t) 处处等于 TD 误差的 (γλ) 折扣和 δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + …")
    print("  上一节的版本是 λ = 1（每项只有 γ^k）。这个量在 Phase 5 叫广义优势估计（GAE）：GAE = λ-return − V。")


# ==================== 第三部分：回合终止 ====================


def continues_demo():
    """继续标志 c_k：回合在某一步之后终止时，后面的东西不能漏进来。"""
    print("\n" + "=" * 60)
    print("第三部分：回合在中间终止怎么办")
    print("=" * 60)

    gamma, lam = 0.9, 0.8
    rewards = np.array([1.0, 2.0, 3.0])
    values = np.array([0.0, 5.0, 4.0, 6.0])
    continues = np.array([1.0, 0.0, 1.0])  # 第 1 步之后回合终止；位置 2 起已经是下一个回合的数据
    out = lambda_return(rewards, values, continues, gamma, lam)
    # 把终止之后的数据全部改掉，终止之前的结果不应该变
    changed = lambda_return(np.array([1.0, 2.0, -50.0]), np.array([0.0, 5.0, 99.0, -99.0]), continues, gamma, lam)
    assert np.isclose(out[1], 2.0) and np.isclose(out[0], 1 + 0.9 * (0.2 * 5 + 0.8 * 2))
    assert np.allclose(out[:2], changed[:2])

    print(f"""
回放数据或想象出来的轨迹是一段连续的数组，回合可能在中间某一步终止。
递推式里多乘一个继续标志 c_k = 1 − d_k（这一步之后终止则为 0）：

    G_k = r_k + γ · c_k · [ (1−λ) · V(s_{{k+1}}) + λ · G_{{k+1}} ]

c_k = 0 时方括号整个被抹掉，只剩 r_k。这就是上一节"terminated 之后不自举"的规则搬到 λ-return 上。

还是那段轨迹，但第 1 步之后回合终止（continues = {continues}）：

    G_1 = r_1 = {out[1]:.3f}                                  终止前的最后一步：只剩奖励
    G_0 = 1 + 0.9 × [0.2 × 5 + 0.8 × {out[1]:.0f}] = {out[0]:.3f}          第 0 步照常递推，但它"看到"的未来到第 1 步为止

  把终止之后的奖励和价值全部换成别的数，G_0 和 G_1 不变（{changed[0]:.3f}、{changed[1]:.3f}）：
  终止点之后的数据没有漏到终止点之前。

  注意只有 terminated 才把 c 置 0。被步数上限掐断（truncated）时 c 仍然是 1，照常用 V 自举。
  在 Dreamer 里，c_k 是世界模型的 continue 预测头给出的概率，可以取 0 到 1 之间的值。
    """)


# ==================== 第四部分：Dreamer 用的就是这个形式 ====================


def dreamer_connection():
    """权重分配、停止梯度：把这一节的函数和 Dreamer 对上。"""
    print("\n" + "=" * 60)
    print("第四部分：这就是 Dreamer 训练 critic 的目标")
    print("=" * 60)

    H, lam, gamma = 15, 0.95, 0.997  # Dreamer V3 的默认值（论文的超参数表）
    weights = lambda_weights(H, lam)
    assert np.isclose(weights.sum(), 1.0)
    print(f"""
Phase 6 的 Dreamer 让世界模型从某个隐状态出发想象 H 步，然后用上面的 lambda_return() 算 critic 的目标：

    公式里的量              在 Dreamer 里是什么
    轨迹 s_t … s_(t+H)      世界模型按当前 actor 想象出来的一段隐状态
    r_k                     奖励预测头的输出（不是真实奖励）
    c_k                     continue 预测头的输出
    V(s)                    critic 网络的输出
    H、λ、γ                 Dreamer V3 的默认值是 {H}、{lam}、{gamma}

用这组默认值算一下，想象轨迹第一个位置的 λ-return 把权重分给了谁：

    1 步回报   {weights[0]:.4f}
    2 步回报   {weights[1]:.4f}
    ……        逐项乘 {lam}
    {H - 1} 步回报  {weights[H - 2]:.4f}
    {H} 步回报  {weights[H - 1]:.4f}    ← 用满 {H} 步想象的奖励，再接末端价值

  权重之和是 {weights.sum():.4f}。{weights[H - 1]:.1%} 的权重压在最长的那个回报上，其余摊在更短的回报上。
  想象的奖励来自模型，越往后越不可靠（Phase 1 的复合误差），短回报提供了"不那么依赖长程想象"的那部分信号；
  末端的 V 把 {H} 步之外的未来接了进来（γ = {gamma} 的有效视野约 {1 / (1 - gamma):.0f} 步）。
  想象负责近处，价值函数负责远处。
    """)

    # 停止梯度：目标当常数。用 PyTorch 张量调用同一个函数
    torch.manual_seed(SEED)
    B = 8
    rewards = torch.randn(B, H)
    continues = torch.ones(B, H)
    grads = {}
    for name, stop_gradient in (("加 sg", True), ("不加 sg", False)):
        values = torch.randn(B, H + 1, requires_grad=True)  # 假装是 critic 在 H+1 个状态上的输出
        targets = lambda_return(rewards, values, continues, gamma, lam)  # [B, H]
        if stop_gradient:
            targets = targets.detach()
        loss = ((values[:, :-1] - targets) ** 2).mean()
        loss.backward()
        grads[name] = values.grad[:, -1].abs().mean().item()  # 末端状态 V(s_H) 收到的梯度
    assert targets.shape == (B, H) and grads["加 sg"] == 0.0 and grads["不加 sg"] > 0.0
    print(f"""  同一个函数直接接受 PyTorch 张量：rewards {tuple(rewards.shape)}、values {tuple(values.shape)} → 目标 {tuple(targets.shape)}。
  critic 的损失是让 V(s_k) 去回归 sg(G^λ_k)。sg 是停止梯度（stop-gradient），代码里就是 .detach()。
  末端状态的 V(s_H) 只出现在目标里，看它收到的梯度：

      目标加 sg（.detach()）    |∂loss/∂V(s_H)| 的平均 = {grads['加 sg']:.4f}
      目标不加 sg               |∂loss/∂V(s_H)| 的平均 = {grads['不加 sg']:.4f}

  加了 sg，目标就是常数，梯度只让"预测"去靠近"目标"。
  不加的话，梯度还会反过来把目标往预测上拉（上面第二行不是 0）：两边互相迁就，损失降了但价值没有变准。
  为什么目标要当常数，第 06 课讲 DQN 的损失时细说。
    """)

    # 图：不同 λ 的权重分配
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, l in zip(axes, (0.5, 0.8, 0.95)):
        ax.bar(np.arange(1, H + 1), lambda_weights(H, l), color="tab:blue")
        ax.set_title(f"λ = {l}" + ("（Dreamer V3 默认）" if l == 0.95 else ""))
        ax.set_xlabel("n 步回报的 n")
    axes[0].set_ylabel("权重")
    fig.suptitle(f"有限视野 H = {H} 时，λ-return 分给各个 n 步回报的权重（最后一根柱子拿走剩下的全部）")
    save_figure(fig, OUT / "04-lambda-weights.png")


# ==================== 第五部分：λ 取多少学得最好 ====================


def random_walk_episode(n_states, rng):
    """上一节的随机游走：从正中间出发，左右各一半；走出左端奖励 -1，右端 +1，其余 0。

    返回 states [T+1]（最后一个是终止，编号 n_states）和 rewards [T]。
    """
    s = n_states // 2
    states, rewards = [s], []
    while True:
        s += 1 if rng.random() < 0.5 else -1
        if s < 0 or s >= n_states:
            states.append(n_states)
            rewards.append(-1.0 if s < 0 else 1.0)
            return np.array(states), np.array(rewards)
        states.append(s)
        rewards.append(0.0)


def lambda_return_learning(episodes, n_states, alpha, gamma, lam, v_true):
    """用 λ-return 当目标做策略评估：每个回合结束后，按回合开始时的 V 算出每一步的 G^λ，再逐步更新。

    返回每个回合之后的均方根误差的平均值（越小说明前几个回合学得越快）。
    """
    V = np.zeros(n_states + 1)  # 多留一格给终止状态，它的价值恒为 0
    errors = []
    for states, rewards in episodes:
        continues = np.ones(len(rewards))
        continues[-1] = 0.0  # 最后一步之后回合终止
        targets = lambda_return(rewards, V[states], continues, gamma, lam)  # [T]；V[states] 是拷贝，回合内目标不变
        for t in range(len(rewards)):
            V[states[t]] += alpha * (targets[t] - V[states[t]])
        errors.append(np.sqrt(np.mean((V[:n_states] - v_true) ** 2)))
    return float(np.mean(errors))


def lambda_sweep(n_runs, n_episodes):
    """随机游走上，不同 λ 在各自较好的步长下的误差。"""
    print("\n" + "=" * 60)
    print("第五部分：λ 取多少学得最好")
    print("=" * 60)

    n_states, gamma = 19, 1.0
    v_true = -1.0 + 2.0 * np.arange(1, n_states + 1) / (n_states + 1)  # 这个环境的真值是一条直线
    # 核对真值：它应该满足 Bellman 期望方程 V(s) = 0.5·V(左邻) + 0.5·V(右邻)，两端的"邻居"是出口的奖励
    padded = np.concatenate([[-1.0], v_true, [1.0]])
    assert np.allclose(v_true, 0.5 * (padded[:-2] + padded[2:]))

    lams = [0.0, 0.5, 0.8, 0.95, 1.0]
    alphas = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    print(f"""
把 λ-return 当作目标来学价值，λ 取多少最好？在 {n_states} 个状态的随机游走上实测
（比上一节的 5 个状态长得多：奖励只在两端，中间的状态要靠传播才能学到）。

  做法：每个回合结束后，用回合开始时的 V 算出每一步的 G^λ（lambda_return），再让 V(s_t) 朝它挪一步。
  指标：前 {n_episodes} 个回合里，每个回合之后的均方根误差的平均（{n_runs} 次运行再平均）。越小说明学得越快。
  每个 λ 都扫一遍步长 α，取它最好的那个来比。初始（V 全 0）的误差是 {np.sqrt(np.mean(v_true**2)):.3f}。
    """)

    results = np.zeros((len(lams), len(alphas)))
    for run in range(n_runs):
        rng = np.random.default_rng(SEED + run)
        episodes = [random_walk_episode(n_states, rng) for _ in range(n_episodes)]  # 所有 λ、α 用同一批回合
        for i, lam in enumerate(lams):
            for j, alpha in enumerate(alphas):
                results[i, j] += lambda_return_learning(episodes, n_states, alpha, gamma, lam, v_true) / n_runs

    best_j = results.argmin(axis=1)
    best_err = results.min(axis=1)
    print(f"  {'λ':<8}{'最好的步长 α':>12}{'该步长下的误差':>16}{'  说明'}")
    notes = {0.0: "一步自举（TD 目标）", 1.0: "不自举（蒙特卡洛回报）"}
    for i, lam in enumerate(lams):
        print(f"  {lam:<8}{alphas[best_j[i]]:>12}{best_err[i]:>16.3f}{'  ' + notes.get(lam, '')}")

    i_best = int(best_err.argmin())
    print(f"\n  误差最小的是 λ = {lams[i_best]}（{best_err[i_best]:.3f}）；"
          f"两端 λ = 0 是 {best_err[0]:.3f}，λ = 1 是 {best_err[-1]:.3f}。")
    if 0 < i_best < len(lams) - 1:
        print("  两端都不如中间：λ = 0 每个回合只把奖励往回传一格，太慢；λ = 1 的目标是完整的回报，噪声太大。")
        print("  中间的 λ 两头都占一点：传得够快，又靠自举把方差压了下去。")
    elif i_best == 0:
        print("  这次实验里 λ = 0 最好：自举带来的低方差比传播速度更重要。")
    else:
        print("  这次实验里 λ = 1 最好：回合数很少时，传播速度比方差更重要。")
    if best_j[0] > best_j[-1]:
        print(f"  还能看到：λ 越大，最好的步长越小（λ = 0 时 {alphas[best_j[0]]}，λ = 1 时 {alphas[best_j[-1]]}）。"
              f"目标越吵，越不敢一步迈太大。")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(lams)))
    for i, lam in enumerate(lams):
        axes[0].plot(alphas, results[i], "o-", color=colors[i], linewidth=2, markersize=4, label=f"λ = {lam}")
    axes[0].set_xlabel("步长 α")
    axes[0].set_ylabel(f"前 {n_episodes} 个回合的平均均方根误差")
    axes[0].set_title(f"{n_states} 状态随机游走：不同 λ 的误差随步长的变化")
    axes[0].legend(fontsize=9)
    axes[1].bar([str(lam) for lam in lams], best_err, color=colors)
    axes[1].set_xlabel("λ")
    axes[1].set_ylabel("各自最好步长下的误差")
    axes[1].set_title("每个 λ 在最好步长下的误差")
    save_figure(fig, OUT / "04-lambda-sweep.png")


# ==================== 第六部分：练习与思考 ====================


def lambda_return_off_by_one(rewards, values, continues, gamma, lam):
    """练习 2 用：一个差一位的错误版本，自举时用了 V(s_k) 而不是 V(s_{k+1})。"""
    H = rewards.shape[-1]
    G = values[..., H]
    out = [None] * H
    for k in reversed(range(H)):
        G = rewards[..., k] + gamma * continues[..., k] * ((1.0 - lam) * values[..., k] + lam * G)  # 错在 values[..., k]
        out[k] = G
    return np.stack(out, axis=-1)


def exercises():
    """练习题与参考答案。答案里的数字都是下面的代码现算的。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    rewards = np.array([1.0, 2.0, 3.0])
    values = np.array([0.0, 5.0, 4.0, 6.0])
    ones = np.ones(3)
    ex1 = lambda_return(rewards, values, ones, 0.9, 0.5)
    assert np.allclose(ex1, [6.661, 7.58, 8.4])

    rng = np.random.default_rng(SEED + 1)
    r, v = rng.normal(size=(1000, 15)), 5.0 + rng.normal(size=(1000, 16))
    good = lambda_return(r, v, np.ones((1000, 15)), 0.99, 0.95)
    bad = lambda_return_off_by_one(r, v, np.ones((1000, 15)), 0.99, 0.95)
    naive = lambda_return_naive(r, v, 0.99, 0.95)
    corr = np.corrcoef(good.ravel(), bad.ravel())[0, 1]

    w10 = lambda_weights(15, 0.95)
    w_short = lambda_weights(15, 0.5)

    print(f"""
练习 1：换一个 λ 手算
    第二部分的那段轨迹（奖励 1, 2, 3；价值 5, 4, 6；γ = 0.9），把 λ 改成 0.5，手算 G_2、G_1、G_0。

练习 1 参考答案：
    G_2 = 3 + 0.9 × 6 = 8.4（最后一步和 λ 无关）
    G_1 = 2 + 0.9 × [0.5 × 4 + 0.5 × 8.4] = 2 + 0.9 × 6.2 = 7.58
    G_0 = 1 + 0.9 × [0.5 × 5 + 0.5 × 7.58] = 1 + 0.9 × 6.29 = 6.661
    程序算出来的：({ex1[0]:.3f}, {ex1[1]:.3f}, {ex1[2]:.3f})。介于 λ = 0 的 5.5 和 λ = 1 的 9.604 之间。

练习 2：差一位的 bug
    lambda_return_off_by_one() 把 values[..., k + 1] 错写成了 values[..., k]。
    用随机数据（1000 条长 15 的轨迹）比较它、正确版本、按定义的版本。只看输出，能发现它错了吗？

练习 2 参考答案：
    正确版本与按定义的版本最大相差 {np.abs(good - naive).max():.1e}；错误版本与正确版本最大相差 {np.abs(bad - good).max():.2f}，
    但两者的相关系数是 {corr:.3f}，均值分别是 {good.mean():.2f} 和 {bad.mean():.2f}。
    错误版本的输出形状对、数值范围也合理，训练能照常跑，只是目标是错的。
    这类 bug 靠看输出很难发现，所以要保留一个按定义写的慢版本当对照，并且用手算的例子做单元测试。

练习 3：λ 和"平均看多远"
    H = 15 时，λ = 0.95 和 λ = 0.5 各自把多少权重放在"10 步及以上"的回报上？

练习 3 参考答案：
    程序算出来的：λ = 0.95 是 {w10[9:].sum():.1%}，λ = 0.5 是 {w_short[9:].sum():.2%}。
    λ = 0.5 时权重几乎全在前几步上，长程想象基本没用上；λ = 0.95 时大部分权重在长回报上。
    Dreamer 取 0.95，是因为想象的轨迹整段都在手上，多用几步没有额外代价。

思考题 1：λ 和 γ 都是 0 到 1 之间的"衰减系数"，调一个不就行了？

思考题 1 参考答案：
    不行，它们管的不是一件事。γ 是任务定义的一部分，决定"我们想要的价值"是什么；
    λ 只影响"怎么估计这个价值"，不改变估计的对象。价值估计完全准确时，不管 λ 取多少，
    G^λ 的期望都等于 V^π；而换一个 γ，V^π 本身就变了。

思考题 2：λ = 1 就是蒙特卡洛、没有自举，对吗？

思考题 2 参考答案：
    只有数据一直延伸到回合终止时才对。有限视野下 λ = 1 得到的是 H 步回报，末端仍然用 V(s_H) 自举
    （第二部分的例子里 λ = 1 的结果 9.604 里就含着 0.729 × 6）。Dreamer 里永远有这个末端自举。

思考题 3：在真实环境里，n 取大的风险是采样噪声。在世界模型的想象里，风险是什么？

思考题 3 参考答案：
    想象里的 r_k 是模型的预测，没有采样噪声，但有模型误差，而且越往后越大（复合误差）。
    偏差与方差的权衡变成了"模型误差与价值误差"的权衡：λ 大，更依赖模型；λ 小，更依赖 critic。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()
    unit_tests()
    continues_demo()
    dreamer_connection()

    n_runs = 5 if is_quick() else 100
    lambda_sweep(n_runs, n_episodes=10)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：05-q-learning.py（表格 Q-learning 与 SARSA）")
    print("=" * 60)


if __name__ == "__main__":
    main()
