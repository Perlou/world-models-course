"""
Gymnasium 交互循环与 MDP
========================

学习目标：
    1. 会用 Gymnasium 的 reset() / step() 写出智能体与环境的交互循环
    2. 能把 MDP 的五样东西（状态、动作、转移、奖励、折扣）和代码里的变量一一对上
    3. 理解马尔可夫性：状态里必须包含"预测未来所需的全部信息"
    4. 会算回报，并用回报比较两个策略的好坏
    5. 采集一份转移数据集 (s, a, r, s')，这是后面所有讲义的训练数据

核心概念：
    - MDP：状态空间、动作空间、转移规律、奖励函数、折扣因子
    - 状态 vs 观测：Pendulum 的观测 (cos θ, sin θ, θ̇) 和状态 (θ, θ̇) 包含相同的信息
    - terminated vs truncated：环境自己结束 vs 被步数上限截断
    - 策略：从状态到动作的函数；回报：一个回合里奖励的（折扣）累加
    - 数据覆盖：随机策略采到的数据集中在哪里，哪里是空白

前置知识：
    - 01-what-is-a-world-model.py（摆的物理方程）

对应概念文档：
    - CONCEPT.md 第 3 节「用 MDP 描述智能体与环境」

预计运行时间：约 5 秒
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-01")
SEED = 0
MAX_TORQUE = 2.0


# ==================== 第一部分：MDP 是什么 ====================


def introduction():
    """智能体—环境循环与 MDP 的五样东西。"""
    print("=" * 60)
    print("第一部分：用 MDP 描述智能体与环境")
    print("=" * 60)

    print("""
上一节我们手写了摆的物理方程。这一节退一步，把"环境"当成一个黑盒：
我们只能给它动作，它只告诉我们观测和奖励。这是后面所有方法面对的真实处境。

    ┌──────────┐      动作 a_t        ┌──────────┐
    │          │ ───────────────────▶ │          │
    │  智能体   │                      │   环境    │
    │          │ ◀─────────────────── │          │
    └──────────┘  观测 o_t+1，奖励 r_t  └──────────┘

写成代码就是一个 while 循环（Web 开发里的"请求—响应"循环，只是带状态）：

    obs, info = env.reset()
    while not done:
        action = policy(obs)                                    # 智能体决定做什么
        obs, reward, terminated, truncated, info = env.step(action)   # 环境给出结果
        done = terminated or truncated

描述这个循环的数学语言叫马尔可夫决策过程（Markov Decision Process，MDP）：

    ┌──────────┬─────────────┬──────────────────────────────────────┐
    │ 组成      │ 符号         │ 在 Pendulum-v1 里                     │
    ├──────────┼─────────────┼──────────────────────────────────────┤
    │ 状态空间   │ S           │ 角度和角速度 (θ, θ̇)                   │
    │ 动作空间   │ A           │ 力矩 u ∈ [-2, 2]                      │
    │ 转移规律   │ p(s'|s, a)  │ 上一节手写的物理方程                    │
    │ 奖励函数   │ r(s, a)     │ -(θ² + 0.1·θ̇² + 0.001·u²)            │
    │ 折扣因子   │ γ           │ 本章不用，Phase 4 再讲                  │
    └──────────┴─────────────┴──────────────────────────────────────┘

世界模型要学的，就是中间两行：转移规律和奖励函数。
    """)


# ==================== 第二部分：Gymnasium 的接口 ====================


def explore_env():
    """看一遍 reset() / step() 的输入输出。"""
    print("\n" + "=" * 60)
    print("第二部分：Gymnasium 的接口")
    print("=" * 60)

    env = gym.make("Pendulum-v1")
    print(f"\n  观测空间：{env.observation_space}")
    print(f"  动作空间：{env.action_space}")
    print(f"  回合步数上限：{env.spec.max_episode_steps}")

    obs, info = env.reset(seed=SEED)
    theta, theta_dot = env.unwrapped.state  # 环境内部保存的真实状态
    print(f"\n  reset() 返回的观测 o_0 = {np.round(obs, 4)}")
    print(f"  环境内部的状态   s_0 = (θ={theta:.4f}, θ̇={theta_dot:.4f})")
    print(f"  按公式算 (cos θ, sin θ, θ̇) = {np.round([np.cos(theta), np.sin(theta), theta_dot], 4)}")
    assert np.allclose(obs, [np.cos(theta), np.sin(theta), theta_dot], atol=1e-6)

    # 观测可以无损地还原出状态：θ = atan2(sin θ, cos θ)
    theta_recovered = np.arctan2(obs[1], obs[0])
    print(f"  从观测反推 θ = atan2(sin θ, cos θ) = {theta_recovered:.4f}")
    print("""
观测 (cos θ, sin θ, θ̇) 和状态 (θ, θ̇) 可以互相换算，包含的信息一样多。
这种情况叫"状态完全可观测"，本章都是这样。Phase 2 起观测变成图像，
图像里看不出速度，观测和状态就不再是一回事了。
    """)

    # 走完一个回合，看两个结束标志
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if steps == 1:
            print(f"  第 1 步：action = {np.round(action, 3)}，reward = {reward:.3f}，"
                  f"terminated = {terminated}，truncated = {truncated}")
    print(f"  第 {steps} 步：terminated = {terminated}，truncated = {truncated}")
    env.close()
    assert steps == 200 and truncated and not terminated

    print("""
两个结束标志的区别：

    ┌────────────┬──────────────────────────────┬────────────────────────┐
    │ 标志        │ 含义                          │ Pendulum-v1             │
    ├────────────┼──────────────────────────────┼────────────────────────┤
    │ terminated │ 环境进入终止状态，之后没有未来   │ 永远是 False            │
    │ truncated  │ 到了步数上限被截断，本来还能继续 │ 第 200 步变成 True      │
    └────────────┴──────────────────────────────┴────────────────────────┘

本章把两者一起当作"回合结束"就够了。Phase 4 学价值函数时必须分开：
被截断的状态后面还有奖励可拿，终止状态没有。
    """)


# ==================== 第三部分：马尔可夫性 ====================


def collect_random_dataset(n_episodes, seed=SEED):
    """用随机策略采集转移数据集。

    返回一个字典，M = n_episodes × 200：
        states      [M, 2]  每行是 (θ, θ̇)，θ 不折叠（可能超出 [-π, π]）
        actions     [M, 1]  力矩
        rewards     [M]     奖励
        next_states [M, 2]  下一个状态
        episode     [M]     这条转移属于第几个回合（按回合切分训练集 / 验证集时用）
    """
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)
    data = {k: [] for k in ["states", "actions", "rewards", "next_states", "episode"]}
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        done = False
        while not done:
            state = env.unwrapped.state.copy()
            action = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=1).astype(np.float32)
            _, reward, terminated, truncated, _ = env.step(action)
            data["states"].append(state)
            data["actions"].append(action)
            data["rewards"].append(reward)
            data["next_states"].append(env.unwrapped.state.copy())
            data["episode"].append(ep)
            done = terminated or truncated
    env.close()
    return {
        "states": np.array(data["states"], dtype=np.float32),
        "actions": np.array(data["actions"], dtype=np.float32),
        "rewards": np.array(data["rewards"], dtype=np.float32),
        "next_states": np.array(data["next_states"], dtype=np.float32),
        "episode": np.array(data["episode"], dtype=np.int64),
    }


def wrap_angle(theta):
    """把角度折到 [-π, π)。正上方是 0，正下方是 ±π。"""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def markov_property(data):
    """状态里少了角速度，未来就预测不准。"""
    print("\n" + "=" * 60)
    print("第三部分：马尔可夫性——状态里要有预测未来所需的全部信息")
    print("=" * 60)

    print("""
马尔可夫性说的是：知道了现在的状态和动作，下一个状态的分布就确定了，
不需要再知道更早的历史。它其实是对"状态"的要求：状态必须够完整。

反例：假如我们只看角度 θ（相当于只看一张静止的照片），不知道角速度。
从数据集里挑出"角度几乎相同、力矩也几乎相同"的转移，看它们下一步的角度变化：
    """)

    theta = wrap_angle(data["states"][:, 0])
    theta_dot = data["states"][:, 1]
    u = data["actions"][:, 0]
    delta_theta = data["next_states"][:, 0] - data["states"][:, 0]

    # 条件一：只固定角度和力矩（θ≈3.0 是底部附近，数据最多）
    only_angle = (np.abs(wrap_angle(theta - 3.0)) < 0.15) & (np.abs(u - 1.0) < 1.0)
    # 条件二：再固定角速度
    full_state = only_angle & (np.abs(theta_dot - 1.0) < 0.25)

    print(f"  {'给定的信息':<28}{'样本数':>8}{'下一步 Δθ 的范围':>24}{'标准差':>10}")
    for name, mask in [("只知道 θ≈3.0，u≈1.0", only_angle), ("还知道 θ̇≈1.0", full_state)]:
        d = delta_theta[mask]
        print(f"  {name:<28}{mask.sum():>8}{f'[{d.min():+.3f}, {d.max():+.3f}]':>24}{d.std():>10.4f}")

    ratio = delta_theta[only_angle].std() / delta_theta[full_state].std()
    print(f"\n  补上角速度后，下一步角度变化的不确定范围缩小到原来的约 1/{ratio:.0f}。")
    print("""
只知道角度时，摆下一步可能往左也可能往右：信息不够，谈不上预测。
补上角速度，下一步就基本确定了（剩下的一点差异来自筛选窗口本身的宽度）。

所以 (θ, θ̇) 是一个合格的马尔可夫状态，单独的 θ 不是。
后面的章节会反复遇到这个问题：
    - Phase 2：单帧图像看不出速度，要靠 RNN 记住历史
    - Phase 3：RSSM 的隐状态 (h, z) 就是为了凑出一个马尔可夫状态
    """)


# ==================== 第四部分：策略与回报 ====================


def random_policy(state, rng):
    """随机策略：不看状态，在动作范围内均匀采样。"""
    return rng.uniform(-MAX_TORQUE, MAX_TORQUE)


def handcrafted_policy(state, rng=None):
    """手写策略：远离顶点时荡秋千攒能量，靠近顶点时用比例—微分控制稳住。

    这个策略用到了我们对摆的物理知识，后面学出来的控制器要和它比。
    """
    theta = wrap_angle(state[0])
    theta_dot = state[1]
    if abs(theta) < 0.4 and abs(theta_dot) < 3.0:
        # 顶点附近：偏了就往回推（比例项），转得快就刹车（微分项）
        u = -10.0 * theta - 2.0 * theta_dot
    else:
        # 机械能（差一个常数因子）：动能 θ̇²/2 + 势能 15·cos θ，顶点静止时等于 15
        energy = 0.5 * theta_dot**2 + 15.0 * np.cos(theta)
        direction = 1.0 if theta_dot >= 0 else -1.0
        # 能量不够就顺着运动方向推，能量多了就逆着推
        u = MAX_TORQUE * direction if energy < 15.0 else -MAX_TORQUE * direction
    return float(np.clip(u, -MAX_TORQUE, MAX_TORQUE))


def run_episode(policy, seed, gamma=1.0):
    """跑一个回合，返回回报、折扣回报、摆尖高度序列。"""
    env = gym.make("Pendulum-v1")
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total, discounted, heights = 0.0, 0.0, []
    t, done = 0, False
    while not done:
        state = env.unwrapped.state
        heights.append(np.cos(state[0]))
        u = policy(state, rng)
        _, reward, terminated, truncated, _ = env.step(np.array([u], dtype=np.float32))
        total += reward
        discounted += gamma**t * reward
        t += 1
        done = terminated or truncated
    env.close()
    return total, discounted, np.array(heights)


def policies_and_returns():
    """用回报比较两个策略。"""
    print("\n" + "=" * 60)
    print("第四部分：策略与回报")
    print("=" * 60)

    print("""
策略（policy）是一个函数：输入状态，输出动作。
回报（return）是一个回合里所有奖励的累加，用来衡量策略好不好：

    G = r_0 + r_1 + r_2 + ... + r_{T-1}              （不折扣）
    G = r_0 + γ·r_1 + γ²·r_2 + ...                   （折扣，γ < 1 时越远的奖励越不值钱）

Pendulum 每步的奖励在 -16.3 到 0 之间，所以 200 步的回报最差约 -3250，最好是 0。
比较两个策略：完全随机，和一个利用物理知识手写的控制器。
    """)

    n_episodes = 5 if is_quick() else 20
    gamma = 0.99
    results = {}
    for name, policy in [("随机策略", random_policy), ("手写策略", handcrafted_policy)]:
        runs = [run_episode(policy, seed=SEED + i, gamma=gamma) for i in range(n_episodes)]
        returns = np.array([r[0] for r in runs])
        discounted = np.array([r[1] for r in runs])
        results[name] = (returns, runs[0][2])
        print(f"  {name}：{n_episodes} 个回合的平均回报 = {returns.mean():8.1f} ± {returns.std():.1f}"
              f"   折扣回报（γ={gamma}）= {discounted.mean():7.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, (returns, heights) in results.items():
        axes[0].plot(np.arange(len(heights)) * 0.05, heights, label=name, linewidth=2)
        axes[1].hist(returns, bins=10, alpha=0.7, label=name)
    axes[0].axhline(1.0, color="gray", linestyle=":")
    axes[0].set_xlabel("时间（秒）")
    axes[0].set_ylabel("摆尖高度 cos θ")
    axes[0].set_title("同一个起点，两个策略的一个回合")
    axes[0].legend()
    axes[1].set_xlabel("回合回报")
    axes[1].set_ylabel("回合数")
    axes[1].set_title(f"{n_episodes} 个回合的回报分布")
    axes[1].legend()
    save_figure(fig, OUT / "02-policies-and-returns.png")

    random_mean = results["随机策略"][0].mean()
    hand_mean = results["手写策略"][0].mean()
    print(f"""
记住这两个数字，它们是本章后面所有控制器的参照：

    随机策略  约 {random_mean:.0f}    下限：什么都没学会
    手写策略  约 {hand_mean:.0f}    参考线：用了物理知识的人工控制器

手写策略各回合的回报差别很大，原因是起点不同：
起点离顶点近，几步就稳住了；起点在底部，要先花几秒钟把摆甩上去。
所以比较两个控制器时要用同一组起点（同一组种子），并且多跑几个回合取平均。

折扣回报的绝对值小得多，因为 γ=0.99 时 100 步之后的奖励只剩 0.99¹⁰⁰ ≈ 37% 的权重。
折扣在 Phase 4 才正式登场，本章的规划器直接用视野内奖励的不折扣累加。
    """)
    return random_mean, hand_mean


# ==================== 第五部分：采集转移数据集 ====================


def dataset_and_coverage(data):
    """数据集的形状，以及它覆盖了状态空间的哪些地方。"""
    print("\n" + "=" * 60)
    print("第五部分：转移数据集，以及它覆盖了哪里")
    print("=" * 60)

    print("""
把交互循环里每一步的 (s_t, a_t, r_t, s_t+1) 存下来，就是一份转移数据集。
下一节用它做监督学习：输入 (s, a)，标签是 s'。
    """)
    for key in ["states", "actions", "rewards", "next_states", "episode"]:
        print(f"  {key:<12} 形状 {str(data[key].shape):<12} dtype = {data[key].dtype}")

    states, next_states = data["states"], data["next_states"]
    delta = next_states - states
    print("\n  各个量的数值范围：")
    print(f"    θ̇（角速度）      [{states[:, 1].min():+.2f}, {states[:, 1].max():+.2f}]")
    print(f"    Δθ（一步的变化）  [{delta[:, 0].min():+.3f}, {delta[:, 0].max():+.3f}]   标准差 {delta[:, 0].std():.3f}")
    print(f"    Δθ̇（一步的变化）  [{delta[:, 1].min():+.3f}, {delta[:, 1].max():+.3f}]   标准差 {delta[:, 1].std():.3f}")

    theta = wrap_angle(states[:, 0])
    theta_dot = states[:, 1]
    near_top = (np.abs(theta) < 0.5) & (np.abs(theta_dot) < 1.0)
    near_bottom = (np.abs(theta) > np.pi - 0.5) & (np.abs(theta_dot) < 1.0)
    upper_half = np.abs(theta) < np.pi / 2
    print("\n  数据落在状态空间的哪些地方：")
    print(f"    上半圈（|θ| < π/2）                 {upper_half.mean():6.1%}")
    print(f"    顶点附近且转得慢（|θ|<0.5, |θ̇|<1）  {near_top.mean():6.1%}   ← 任务要求摆停在这里")
    print(f"    底部附近且转得慢（|θ|>π-0.5, |θ̇|<1）{near_bottom.mean():6.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    h = axes[0].hist2d(theta, theta_dot, bins=[40, 40], range=[[-np.pi, np.pi], [-8, 8]], cmap="viridis")
    fig.colorbar(h[3], ax=axes[0], label="转移条数")
    axes[0].add_patch(plt.Rectangle((-0.5, -1.0), 1.0, 2.0, fill=False, edgecolor="red", linewidth=2))
    axes[0].text(0.6, 1.3, "目标区域", color="red", fontsize=10)
    axes[0].set_xlabel("角度 θ（0 是正上方，±π 是正下方）")
    axes[0].set_ylabel("角速度 θ̇")
    axes[0].set_title("随机策略采到的数据在状态空间里的分布")
    axes[0].grid(False)
    axes[1].hist(theta, bins=40, range=(-np.pi, np.pi), color="steelblue")
    axes[1].set_xlabel("角度 θ")
    axes[1].set_ylabel("转移条数")
    axes[1].set_title("角度的分布：大部分时间摆都垂在下半圈")
    save_figure(fig, OUT / "02-dataset-coverage.png")

    path = OUT / "02-pendulum-random.npz"
    np.savez_compressed(path, **data)
    print(f"💾 数据集已保存：outputs/phase-01/{path.name}")

    print(f"""
两点观察：

    1. 数据分布很不均匀。随机乱推没法把摆甩上去，摆大部分时间在下半圈晃，
       顶点附近转得慢的样本只占 {near_top.mean():.1%}，而那里恰恰是任务的目标区域。
    2. Δθ 和 Δθ̇ 的数值范围差好几倍，下一节做归一化时会用到。

第一点是后面几节反复出现的主题：模型只在有数据的地方准。
    - 第 04 课：模型在数据稀少的区域误差明显更大（分布偏移）
    - 第 07 课：用集成让模型知道自己在哪里没把握
    - 第 08 课：让控制器自己去采数据，把目标区域的空白补上

（后面的讲义为了能单独运行，会各自重新采集数据，不依赖上面保存的文件。）
    """)
    return near_top.mean()


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：换一个筛选条件
    markov_property() 里筛的是 θ≈3.0（底部附近）。改成 θ≈0（顶点附近），
    "只知道角度"时的样本数和 Δθ 的标准差会怎么变？为什么？

练习 1 参考答案：
    实测：样本数从 460 条降到 58 条，Δθ 的标准差从 0.193 降到 0.080。
    - 样本少：随机策略很少能把摆送到顶点（第五部分的覆盖图）。
    - 标准差小：Δθ = θ̇_new · dt，它的散布来自角速度的散布。
      随机力矩平均下来不给摆加多少能量，能爬到顶点的摆在那里已经把动能
      换成了势能，转得都很慢；而在底部，摆的速度有快有慢。
    结论不变：补上角速度后标准差只有 0.01 左右，缺了它仍然差 8 倍。

练习 2：只用观测，不碰 env.unwrapped
    真实的机器人拿不到"内部状态"。改写 collect_random_dataset()，
    只用 step() 返回的观测 (cos θ, sin θ, θ̇) 来构造 (θ, θ̇)。
    这样算出来的 Δθ 会出什么问题？

练习 2 参考答案：
    用 θ = atan2(sin θ, cos θ) 还原出的角度在 [-π, π] 里。摆越过正下方时
    θ 会从 +3.14 跳到 -3.14，直接相减得到 Δθ ≈ -6.28，是一个假的大跳变。
    修法是把差值也折回去：Δθ = wrap_angle(θ' - θ)。
    因为一步之内角度最多变 0.4 弧度，折叠后的差值就是真实的变化量。

练习 3：去掉"攒能量"的部分
    把 handcrafted_policy() 改成全程只用比例—微分控制
    u = clip(-10·θ - 2·θ̇, -2, 2)，平均回报会怎样？

练习 3 参考答案：
    实测：20 个回合的平均回报从 -137 掉到 -950，只有 5 个回合最后稳在顶点
    （完整的手写策略是 20 个回合全部稳住）。
    最大力矩只能产生 6 的角加速度，而重力在 θ 处产生 15·sin θ。
    |θ| > 0.41 时重力更大，直接往上推是推不上去的，只有起点碰巧
    离顶点近的那几个回合能成功。从底部出发必须先来回甩，把能量攒够。
    这说明手写控制器的每一段逻辑背后都是物理知识。换一个系统，全部要重新推。
    学一个模型再做规划，就是为了不用为每个系统手工推这些东西。

思考题 1：Pendulum 的奖励里有一项 0.001·u²（惩罚用力）。
    如果去掉这一项，最优策略会有什么不同？

思考题 1 参考答案：
    摆稳在顶点之后，最优的力矩本来就接近 0，这一项几乎没影响。
    影响主要在甩上去的过程：没有这一项，控制器会毫不犹豫地一直用最大力矩。
    在真实机器人上，这类惩罚项对应能耗和电机发热，通常不能省。

思考题 2：真实机器人的"状态"是什么？关节角度和角速度够吗？

思考题 2 参考答案：
    往往不够。电机有延迟（上一步的指令还没执行完）、有摩擦发热、
    被操作的物体有自己的位置和速度。这些量如果影响未来却不在状态里，
    从智能体的角度看，环境就不满足马尔可夫性。
    常见的补救办法是把最近几步的观测和动作拼起来当状态，
    或者像 Phase 3 那样让循环网络自己学一个状态出来。

思考题 3：随机策略采到的数据里，顶点附近的样本很少。
    多采 10 倍数据能解决问题吗？

思考题 3 参考答案：
    比例不会变，绝对数量会涨 10 倍，有一点帮助，但效率很低：
    绝大部分新数据还是落在已经很熟悉的下半圈。
    更有效的办法是换采集策略：用当前的控制器去采数据，
    控制器越好，数据就越集中在任务相关的区域。这就是第 08 课 PETS 的闭环。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()
    explore_env()

    n_episodes = 10 if is_quick() else 50
    data = collect_random_dataset(n_episodes)
    markov_property(data)
    policies_and_returns()
    dataset_and_coverage(data)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：03-learning-dynamics-mlp.py（用神经网络学动力学模型）")
    print("=" * 60)


if __name__ == "__main__":
    main()
