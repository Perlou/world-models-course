"""
世界模型是什么
==============

学习目标：
    1. 说清楚"世界模型"是什么：一个能回答"如果我这样做，接下来会怎样"的函数
    2. 亲手写出一个世界模型（摆的物理方程），并验证它和真实环境逐步一致
    3. 理解世界模型的三种用途：预测、想象、规划
    4. 看到"模型不准"的后果，明白为什么后面要从数据里学模型

核心概念：
    - 世界模型：s' = f(s, a)，给定当前状态和动作，预测下一个状态（和奖励）
    - 展开（rollout）：把模型的输出再喂回模型，一步步向前推演
    - 想象：不碰真实环境，只在模型里试不同的做法
    - 规划：在想象出的多个未来里，挑结果最好的那个

前置知识：
    - NumPy 基础
    - 一点高中物理（力矩让摆加速）

对应概念文档：
    - CONCEPT.md 第 1 节「什么是世界模型」、第 2 节「分类与全景」

预计运行时间：约 5 秒
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np

from utils import output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-01")
SEED = 0

# Pendulum-v1 的物理常数（与 Gymnasium 源码一致）
G = 10.0  # 重力加速度
M = 1.0  # 摆的质量
L = 1.0  # 摆的长度
DT = 0.05  # 每一步的时间间隔（秒）
MAX_SPEED = 8.0  # 角速度上限
MAX_TORQUE = 2.0  # 力矩上限


# ==================== 第一部分：世界模型是什么 ====================


def introduction():
    """世界模型的定义和三种用途。"""
    print("=" * 60)
    print("第一部分：世界模型是什么")
    print("=" * 60)

    print("""
你过马路时不会等车撞上来才知道危险。你脑子里有一个关于世界怎么运转的模型：
车在动，按这个速度两秒后会到我面前，所以现在不能走。

这就是世界模型要做的事。用一句话说：

    世界模型是一个函数，输入"当前状态 + 我要做的动作"，
    输出"接下来会发生什么"。

    ┌───────────┐
    │  状态 s_t  │──┐
    └───────────┘  │    ┌──────────────┐     ┌──────────────┐
                   ├───▶│   世界模型 f   │────▶│ 下一状态 s_t+1 │
    ┌───────────┐  │    └──────────────┘     │ 奖励 r_t      │
    │  动作 a_t  │──┘                         └──────────────┘
    └───────────┘

有了它，可以做三件事：

    ┌──────┬────────────────────────────┬──────────────────────┐
    │ 用途  │ 做什么                      │ 本课程哪里会用到       │
    ├──────┼────────────────────────────┼──────────────────────┤
    │ 预测  │ 给定动作序列，推演未来轨迹    │ 所有阶段              │
    │ 想象  │ 不碰真实环境，在模型里试错    │ Dreamer（第 6 阶段）   │
    │ 规划  │ 比较多个想象出的未来，选最好的 │ MPC、PlaNet、TD-MPC2  │
    └──────┴────────────────────────────┴──────────────────────┘

为什么值得费劲学一个世界模型？因为真实世界里试错很贵：
机器人摔一次可能就坏了，而在模型里摔一万次不花一分钱。

本节课用的环境是 Gymnasium 的 Pendulum-v1（倒立摆）：

            ● ← 目标：把摆甩到正上方并保持住
            │
            │         状态：角度 θ（正上方为 0）、角速度 θ̇
            ○ ← 转轴   动作：施加在转轴上的力矩 u ∈ [-2, 2]
            │
            ● ← 初始时摆大多垂在下面

难点在于力矩太小，没法把摆直接举上去，必须先来回甩几下攒够能量。

这个环境的物理规律很简单，我们可以直接把它写成代码。
这段代码就是一个"完美的世界模型"。
    """)


# ==================== 第二部分：亲手写一个世界模型 ====================


def pendulum_step(state, torque, g=G, m=M, l=L):
    """摆的世界模型：给定状态和力矩，返回下一个状态。

    参数：
        state:  (theta, theta_dot)，角度（弧度，正上方为 0）和角速度
        torque: 力矩，会被裁剪到 [-MAX_TORQUE, MAX_TORQUE]
        g, m, l: 物理参数，默认是真实值；第四部分会故意传错的值进来
    返回：
        下一时刻的 (theta, theta_dot)
    """
    theta, theta_dot = state
    u = float(np.clip(torque, -MAX_TORQUE, MAX_TORQUE))

    # 角加速度 = 重力产生的 + 力矩产生的
    theta_acc = 3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * u
    # 先更新角速度，再用新的角速度更新角度（半隐式欧拉法）
    theta_dot_new = np.clip(theta_dot + theta_acc * DT, -MAX_SPEED, MAX_SPEED)
    theta_new = theta + theta_dot_new * DT
    return np.array([theta_new, theta_dot_new])


def pendulum_reward(state, torque):
    """奖励函数：离正上方越近、转得越慢、用力越小，奖励越高（最高为 0）。"""
    theta, theta_dot = state
    u = float(np.clip(torque, -MAX_TORQUE, MAX_TORQUE))
    theta_wrapped = ((theta + np.pi) % (2 * np.pi)) - np.pi  # 把角度折到 [-π, π)
    return -(theta_wrapped**2 + 0.1 * theta_dot**2 + 0.001 * u**2)


def verify_against_env():
    """把手写的模型和真实环境逐步对比。"""
    print("\n" + "=" * 60)
    print("第二部分：亲手写一个世界模型，并和真实环境对比")
    print("=" * 60)

    print("""
上面的 pendulum_step() 只有 6 行，它就是 Pendulum-v1 的全部物理规律。
现在验证它：让真实环境和我们的模型从同一个状态出发，执行同样的随机动作。

    单步预测：每一步都从真实状态出发，只预测一步
    开环预测：只在第 0 步看一眼真实状态，之后全靠模型自己推 200 步
    """)

    env = gym.make("Pendulum-v1")
    env.reset(seed=SEED)
    true_state = env.unwrapped.state.copy()  # 环境内部的 (theta, theta_dot)
    open_loop_state = true_state.copy()
    rng = np.random.default_rng(SEED)

    max_one_step_err = 0.0
    max_open_loop_err = 0.0
    max_reward_err = 0.0
    for _ in range(200):
        # 环境收到的是 float32 动作，模型也用同一个数，避免精度差异
        action = np.array([rng.uniform(-MAX_TORQUE, MAX_TORQUE)], dtype=np.float32)
        u = float(action[0])

        one_step_pred = pendulum_step(true_state, u)
        reward_pred = pendulum_reward(true_state, u)
        open_loop_state = pendulum_step(open_loop_state, u)

        _, reward, _, _, _ = env.step(action)
        true_state = env.unwrapped.state.copy()

        max_one_step_err = max(max_one_step_err, np.abs(one_step_pred - true_state).max())
        max_open_loop_err = max(max_open_loop_err, np.abs(open_loop_state - true_state).max())
        max_reward_err = max(max_reward_err, abs(reward_pred - reward))
    env.close()

    print(f"  单步预测的最大误差：  {max_one_step_err:.2e}")
    print(f"  开环 200 步的最大误差：{max_open_loop_err:.2e}")
    print(f"  奖励的最大误差：      {max_reward_err:.2e}")
    print("""
误差只有 1e-8 到 1e-7，这是 float32 的舍入精度（环境内部用 float32 存动作，
我们的模型用 float64）。除此之外，这 6 行代码和真实环境完全等价。
这是一个完美的世界模型：推 200 步也不会走样。

记住这个"开环 200 步误差约等于 0"的结果。
第 04 课换成神经网络学出来的模型后，这个数字会大得多。
    """)
    assert max_one_step_err < 1e-6, "手写模型应该和环境逐步一致"
    assert max_open_loop_err < 1e-4, "完美模型开环展开也应该一致"


# ==================== 第三部分：预测、想象、规划 ====================


def rollout(state, policy, horizon, **model_kwargs):
    """在模型里展开：从 state 出发，按 policy 选动作，推演 horizon 步。

    policy 是一个函数：输入 (状态, 步数)，输出力矩。
    返回：状态序列 [horizon+1, 2] 和累计奖励。
    """
    states = [np.asarray(state, dtype=np.float64)]
    total_reward = 0.0
    for t in range(horizon):
        u = policy(states[-1], t)
        total_reward += pendulum_reward(states[-1], u)
        states.append(pendulum_step(states[-1], u, **model_kwargs))
    return np.array(states), total_reward


def three_uses():
    """用完美模型演示想象和规划。"""
    print("\n" + "=" * 60)
    print("第三部分：用世界模型做想象和规划")
    print("=" * 60)

    print("""
设想摆垂在正下方、静止不动（θ = π）。我们有三种做法，不知道哪种好：

    A. 什么都不做（力矩恒为 0）
    B. 一直朝一个方向使劲推（力矩恒为 +2）
    C. 顺着摆的运动方向推（摆往哪边动就往哪边加力，像荡秋千）

不用去真实环境里试。把三种做法分别放进模型里推演 100 步，
看看各自会走到哪里，这就是"想象"。
    """)

    start = np.array([np.pi, 0.0])
    horizon = 100
    policies = {
        "A 不施力": lambda s, t: 0.0,
        "B 恒定力矩 +2": lambda s, t: MAX_TORQUE,
        "C 顺着运动方向推": lambda s, t: MAX_TORQUE if s[1] >= 0 else -MAX_TORQUE,
    }

    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, policy in policies.items():
        states, imagined_return = rollout(start, policy, horizon)
        height = np.cos(states[:, 0])  # 摆尖的高度：+1 在正上方，-1 在正下方
        results[name] = (imagined_return, height.max())
        axes[0].plot(np.arange(horizon + 1) * DT, height, label=name, linewidth=2)
        print(f"  {name:<12}  想象中的累计奖励 = {imagined_return:9.1f}   最高到达高度 = {height.max():+.2f}")
    axes[0].axhline(1.0, color="gray", linestyle=":", label="正上方（目标）")
    axes[0].set_xlabel("时间（秒）")
    axes[0].set_ylabel("摆尖高度 cos θ")
    axes[0].set_title("三种做法在模型里的想象结果")
    axes[0].legend(fontsize=9)

    # 规划 = 想象 + 挑选
    best = max(results, key=lambda k: results[k][0])
    print(f"\n  规划的结果：选累计奖励最高的 → 「{best}」")

    # 把选中的做法拿到真实环境里执行，核对想象是否准确
    env = gym.make("Pendulum-v1")
    env.reset(seed=SEED)
    env.unwrapped.state = start.copy()  # 把真实环境也摆到同一个起点
    real_return = 0.0
    real_height = [np.cos(start[0])]
    policy = policies[best]
    for t in range(horizon):
        state = env.unwrapped.state
        u = policy(state, t)
        _, reward, _, _, _ = env.step(np.array([u], dtype=np.float32))
        real_return += reward
        real_height.append(np.cos(env.unwrapped.state[0]))
    env.close()

    imagined_states, imagined_return = rollout(start, policy, horizon)
    axes[1].plot(np.arange(horizon + 1) * DT, np.cos(imagined_states[:, 0]), linewidth=4, alpha=0.4, label="模型里的想象")
    axes[1].plot(np.arange(horizon + 1) * DT, real_height, "k--", linewidth=1.5, label="真实环境里的执行")
    axes[1].set_xlabel("时间（秒）")
    axes[1].set_ylabel("摆尖高度 cos θ")
    axes[1].set_title(f"「{best}」：想象 vs 真实")
    axes[1].legend(fontsize=9)
    save_figure(fig, OUT / "01-imagination-and-planning.png")

    print(f"  想象中的累计奖励：{imagined_return:.3f}")
    print(f"  真实执行的累计奖励：{real_return:.3f}")
    print("""
三点观察：

    1. 做法 B 失败了。力矩不够大，一直朝一个方向推，摆只是在底部附近来回晃，
       最高只到 cos θ ≈ -0.66。这一点不用试真实环境，在模型里就看出来了。
    2. 做法 C 靠来回甩把能量攒上去，摆尖到达了正上方。
    3. 因为模型是完美的，想象中的奖励和真实执行的奖励一致。

"在模型里试几种做法，选最好的去执行"，这就是规划最朴素的形式。
第 05 课会把"三种做法"换成"一千条随机动作序列"，那就是模型预测控制（MPC）。
    """)


# ==================== 第四部分：模型不准会怎样 ====================


def wrong_model_demo():
    """故意把物理参数写错一点，看展开误差如何增长。"""
    print("\n" + "=" * 60)
    print("第四部分：模型不准会怎样")
    print("=" * 60)

    print("""
现实中我们拿不到完美的物理方程。机械臂的摩擦、电机的延迟、
物体的质量分布，都没法精确写出来。

模拟一下这种情况：假设我们把摆的质量量错了一点（真实值 m = 1.0）。
让"错的模型"和"真实模型"从同一个状态出发，执行同一串随机动作，
看两条轨迹的角度差多少。为了不被个别轨迹误导，重复 300 次取平均。
    """)

    rng = np.random.default_rng(SEED)
    horizon = 100
    n_traj = 300
    mass_errors = [0.01, 0.05, 0.20]
    # errors[质量误差][第几条轨迹, 第几步] = 角度误差
    errors = {me: np.zeros((n_traj, horizon + 1)) for me in mass_errors}
    for i in range(n_traj):
        # 起点的分布和 Pendulum-v1 的 reset 一样：角度任意，角速度在 [-1, 1]
        start = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-1.0, 1.0)])
        actions = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=horizon)
        policy = lambda s, t: actions[t]
        true_states, _ = rollout(start, policy, horizon)
        for me in mass_errors:
            wrong_states, _ = rollout(start, policy, horizon, m=M * (1 + me))
            errors[me][i] = np.abs(wrong_states[:, 0] - true_states[:, 0])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    print(f"  {'质量误差':<8}{'1 步后':>12}{'10 步后':>12}{'30 步后':>12}{'50 步后':>12}{'100 步后':>12}   （平均角度误差，弧度）")
    for me in mass_errors:
        mean_err = errors[me].mean(axis=0)
        print(f"  {me:>6.0%}  {mean_err[1]:>12.5f}{mean_err[10]:>12.5f}{mean_err[30]:>12.5f}{mean_err[50]:>12.5f}{mean_err[100]:>12.5f}")
        ax.plot(np.arange(1, horizon + 1), mean_err[1:], label=f"质量量错 {me:.0%}", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("展开步数")
    ax.set_ylabel("平均角度误差（弧度，对数坐标）")
    ax.set_title(f"模型只错一点点，展开得越远误差越大（{n_traj} 条轨迹的平均）")
    ax.legend()
    save_figure(fig, OUT / "01-wrong-model-divergence.png")

    small = errors[0.01].mean(axis=0)
    print(f"\n  质量只量错 1% 时：1 步后的平均误差是 {small[1]:.5f} 弧度，"
          f"100 步后是 {small[100]:.3f} 弧度，放大了约 {small[100] / small[1]:.0f} 倍。")
    print("""
三点观察：

    1. 只预测一步时，误差小到几乎看不出来。
    2. 误差会累积：这一步的错误输出是下一步的输入，错上加错。
       这叫"复合误差"（compounding error），是世界模型最核心的难题。
       （表里是很多条轨迹的平均。单看某一条轨迹，误差会忽大忽小，
        因为两个摆可能碰巧又摆到相近的位置；但平均来看是越推越大。）
    3. 所以世界模型有一个天然的"有效视野"：只能信它往前推的有限步数。

后面的课程很大一部分都在和这个问题较劲：
    - 第 03 课：不写物理方程，用神经网络从数据里学 f
    - 第 04 课：定量研究复合误差
    - 第 05 课：每走一步就重新规划（MPC），不让误差有机会累积
    - 第 07 课：让模型知道自己哪里不准（不确定性）
    """)


# ==================== 第五部分：课程地图 ====================


def course_map():
    """本课程的全景。"""
    print("\n" + "=" * 60)
    print("第五部分：从这 6 行物理方程到课程全景")
    print("=" * 60)

    print("""
这节课的世界模型有三个"过于理想"的地方，后面逐个去掉：

    ┌──────────────────────┬──────────────────────────────────────┐
    │ 本节课的理想假设       │ 去掉它之后要学什么                      │
    ├──────────────────────┼──────────────────────────────────────┤
    │ 物理方程是已知的       │ 从数据里学动力学模型（第 1 阶段）         │
    │ 状态是 2 个数（θ, θ̇）  │ 从图像里学隐状态：VAE、RSSM（第 2-3 阶段）│
    │ 只比较 3 种做法        │ 规划器、在想象中训练策略（第 1、4-7 阶段） │
    └──────────────────────┴──────────────────────────────────────┘

整门课的路线：

    第一部分  学模型 + 做规划
        P1 状态动力学模型与 MPC ──▶ P2 World Models（VAE+RNN）──▶ P3 RSSM 与 PlaNet

    第二部分  在想象中学习
        P4 RL：值函数 ──▶ P5 RL：策略梯度 ──▶ P6 Dreamer ──▶ P7 MuZero 与 TD-MPC2

    第三部分  生成式与表征式世界模型
        P8 Token/Transformer ──▶ P9 扩散与视频 ──▶ P10 JEPA ──▶ P11 机器人专题 ──▶ P12 毕业项目

不管后面的模型多复杂（Dreamer、Genie、V-JEPA），核心接口都和今天这 6 行一样：
给我现在的状态和动作，我告诉你接下来会怎样。
    """)


# ==================== 第六部分：练习与思考 ====================


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：换一个起点
    把 three_uses() 里的起点改成 (θ=π/2, θ̇=0)，也就是摆水平伸出。
    三种做法的排名会变吗？做法 B 这次能把摆举到正上方吗？

练习 1 参考答案：
    排名变了：A 的累计奖励最高（约 -596），B 和 C 完全相同（约 -821）。
    - 从水平位置松手后，摆朝 θ 增大的方向下落，角速度一直不小于 0，
      所以 C 每一步都选 +2，和 B 是同一串动作。
    - 正力矩和摆的运动方向一致，一直在给它加能量。摆越转越快，
      冲过正上方后停不下来，一直打转（角速度顶到上限 8）。
      转速的惩罚让它的累计奖励比什么都不做还低。
    所以 B 这次能"经过"正上方，但停不住。"经过"和"保持"是两回事：
    要保持住，得在接近顶点前反向减速。这三种简单做法都做不到，
    要靠第 05 课的规划。

练习 2：把长度量错
    wrong_model_demo() 里量错的是质量。改成把长度 l 量错 5%，
    100 步后的误差比质量量错 5% 更大还是更小？为什么？

练习 2 参考答案：
    更大。质量只出现在力矩项 3u/(m·l²) 里，长度同时出现在重力项 3g/(2l)
    和力矩项里（而且力矩项里是平方）。长度的误差影响每一步的重力加速度，
    即使不施力也会累积。
    实测（300 条轨迹的平均）：100 步后的角度误差，质量量错 5% 约 0.47 弧度，
    长度量错 5% 约 0.92 弧度。

练习 3：写一个新的世界模型
    仿照 pendulum_step()，为"一维小车"写世界模型：
    状态是 (位置 x, 速度 v)，动作是推力 F，质量 m = 1，没有摩擦。
    用半隐式欧拉法，dt = 0.05。

练习 3 参考答案：
    def cart_step(state, force, m=1.0, dt=0.05):
        x, v = state
        v_new = v + force / m * dt
        x_new = x + v_new * dt
        return np.array([x_new, v_new])

思考题 1：世界模型一定要预测"完整的下一个状态"吗？
    如果我只关心任务能不能完成，模型可以只预测什么？

思考题 1 参考答案：
    不一定。只预测奖励和价值也可以做规划，MuZero 和 TD-MPC2 就是这样
    （第 7 阶段）。预测什么、预测到多细，是世界模型设计中最重要的选择，
    CONCEPT.md 第 2 节按这个维度给世界模型做了分类。

思考题 2：做法 C（顺着运动方向推）是一个"策略"，不是一串固定的动作。
    在模型里想象一个策略和想象一串固定动作，有什么区别？

思考题 2 参考答案：
    固定动作序列不看状态，模型推到哪算哪（开环）。策略每一步都根据
    模型当前给出的状态选动作（闭环）。Dreamer 在想象中训练的就是策略，
    而本阶段的 MPC 优化的是固定动作序列，靠每步重新规划来弥补开环的缺点。

思考题 3：既然有了完美的物理方程，为什么不直接解出最优控制，还要"试"？

思考题 3 参考答案：
    对摆这种低维、方程已知的系统，确实可以用最优控制理论直接求解。
    但只要方程写不出来（比如输入是摄像头图像），或者状态维度很高，
    解析方法就用不上了。"学一个模型，然后在模型里试"是能推广到
    这些情况的通用办法，这也是本课程选择这条路线的原因。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    introduction()
    verify_against_env()
    three_uses()
    wrong_model_demo()
    course_map()
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：02-gymnasium-and-mdp.py（环境交互循环与 MDP）")
    print("=" * 60)


if __name__ == "__main__":
    main()
