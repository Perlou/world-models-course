"""
滤波与信念状态
==============

学习目标：
    1. 亲眼看到"只给画面"为什么是部分可观测：单帧里有角度，没有角速度
    2. 理解信念状态：看过全部历史之后，对"现在是什么局面"的判断，它是一个分布
    3. 从零实现一维卡尔曼滤波，看清"预测让不确定性变大，修正让它变小"
    4. 看到修正步的"增益"可以不靠公式、只靠"把下一个观测预测准"从数据里学出来，
       这正是后面 RSSM 的做法

核心概念：
    - 部分可观测（partial observability）：智能体拿不到状态 s_t，只能拿到由它产生的观测 o_t
    - 信念状态（belief state）：b_t(s) = p(s_t = s | o_≤t, a_<t)
    - 滤波（filtering）：只用过去和现在的观测估计现在的状态
    - 卡尔曼滤波（Kalman filter）：线性高斯情形下"预测—修正"的闭式解
    - 卡尔曼增益（Kalman gain）g_t：修正时观测所占的权重

前置知识：
    - Phase 1 的 02-gymnasium-and-mdp.py（状态与观测）、04-compounding-error.py（开环展开）
    - 高斯分布、贝叶斯公式

对应概念文档：
    - CONCEPT.md 第 1 节「部分可观测与信念状态」

预计运行时间：约 30 秒（CPU）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # 渲染只取像素数组，不需要弹窗口

import cv2
import gymnasium as gym
import numpy as np
import torch

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-03")
SEED = 0

IMAGE_SIZE = 64  # 本章所有像素观测都是 64×64×3
MAX_TORQUE = 2.0
MAX_SPEED = 8.0

# 一维小车的噪声设定：移动比较准，传感器比较差
Q_VAR = 0.04  # 转移噪声方差 σ_w²
R_VAR = 1.0  # 观测噪声方差 σ_v²
T_STEPS = 80
DROP_START, DROP_END = 30, 50  # 第 30 ≤ t < 50 步传感器掉线


# ==================== 第一部分：问题从哪来 ====================


def introduction():
    """部分可观测、信念状态、预测—修正。"""
    print("=" * 60)
    print("第一部分：只给一张图，凭什么知道现在的局面")
    print("=" * 60)

    print("""
Phase 1 里，环境直接把摆的角度和角速度 (θ, θ̇) 交给我们。从这一章起只给一张渲染出来的图。
图里读得出角度，读不出角速度：同一张图，摆可能正在往上甩，也可能正在往下掉。
这叫部分可观测（partial observability）。

人看一小段视频就知道摆在往哪边走。做的事情是：把过去看到的东西攒起来，
形成一个对"现在是什么局面"的判断，每来一帧新画面就更新它。
这个判断叫信念状态（belief state）。它不是一个点，是一个分布：我认为状态在哪，以及有多大把握。

信念靠两步交替更新，这一节和后面整章都围着这两步转：

        上一步的信念 b_{t-1}
               │
               │  ① 预测：按转移规律和动作 a_{t-1} 往前推        不确定性变大
               ▼
        还没看观测时的信念 b_t⁻           ← RSSM 里叫"先验"
               │
               │  ② 修正：用新来的观测 o_t 校准                  不确定性变小
               ▼
        看了观测之后的信念 b_t            ← RSSM 里叫"后验"

    这一步没有观测？那就跳过 ②，只靠预测往前走。RSSM 里这叫"想象"。

本节分两半：
    前一半  渲染像素版 Pendulum-v1，确认单帧里确实没有角速度
    后一半  在一个能手算的例子（一维小车）上实现"预测—修正"，也就是卡尔曼滤波（Kalman filter）
    """)


# ==================== 第二部分：单帧画面里没有速度 ====================


def make_pixel_pendulum(seed):
    """像素版 Pendulum-v1。智能体只看 render() 的画面，状态向量只用来做分析。"""
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env.reset(seed=seed)
    return env


def render_frame(env, size=IMAGE_SIZE):
    """渲染当前画面并缩到 64×64。返回 [64, 64, 3] 的 uint8 数组。

    Gymnasium 的渲染会在转轴处画一个箭头表示上一步的力矩（大小和方向）。
    动作本来就是模型的输入，不需要从画面里读，所以本章统一把这个箭头关掉，
    画面里只剩一根摆杆：能读出的只有角度。
    """
    raw = env.unwrapped
    raw.last_u = None  # 关掉力矩箭头
    frame = raw.render()  # [500, 500, 3]
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def show_ambiguity():
    """同一张图，两种相反的运动。"""
    print("\n" + "=" * 60)
    print("第二部分：单帧画面里没有速度")
    print("=" * 60)

    print("""
实验一：把摆放在同一个角度（水平伸出，θ = π/2），给两个相反的角速度（±4 rad/s），
不施加力矩，各往后走 3 步，把画面排在一起看。
    """)
    env = make_pixel_pendulum(SEED)
    raw = env.unwrapped
    theta0, speed, n_future = np.pi / 2, 4.0, 3
    frames = {}
    for name, velocity in [("A", +speed), ("B", -speed)]:
        raw.state = np.array([theta0, velocity])
        frames[name] = [render_frame(env)]
        for _ in range(n_future):
            raw.step(np.zeros(1, dtype=np.float32))
            frames[name].append(render_frame(env))
    env.close()

    print(f"  {'时刻':<8}{'两个画面差别最大的像素（0–255）':>22}{'有差别的像素占比':>14}")
    diffs = []
    for k in range(n_future + 1):
        diff = np.abs(frames["A"][k].astype(np.int32) - frames["B"][k].astype(np.int32))
        diffs.append(int(diff.max()))
        label = "t" if k == 0 else f"t+{k}"
        print(f"  {label:<10}{diff.max():>22d}{(diff.max(axis=-1) > 0).mean():>20.1%}")

    fig, axes = plt.subplots(2, n_future + 1, figsize=(2.6 * (n_future + 1), 5.4))
    for row, (name, velocity) in enumerate([("A", +speed), ("B", -speed)]):
        for k in range(n_future + 1):
            axes[row, k].imshow(frames[name][k])
            axes[row, k].set_xticks([])
            axes[row, k].set_yticks([])
            axes[row, k].grid(False)
            axes[row, k].set_title(f"{name}（θ̇ = {velocity:+.0f}）：" + ("t" if k == 0 else f"t+{k}"), fontsize=9)
    fig.suptitle("第一列的两张图完全一样，之后却走向相反的方向")
    save_figure(fig, OUT / "01-single-frame-ambiguity.png")

    if diffs[0] == 0 and diffs[-1] > 0:
        print(f"""
t 时刻两张图逐像素相同（最大差别 0），{n_future} 步之后最大差别是 {diffs[-1]}。
只看 t 时刻的画面，没有任何办法知道接下来是 A 还是 B。
所以"画面"不满足马尔可夫性：下一帧长什么样，不能只由当前这一帧和动作决定。
    """)
    else:
        print(f"""
注意：这次运行里 t 时刻两张图的最大差别是 {diffs[0]}，{n_future} 步之后是 {diffs[-1]}，
和"同一角度的画面应该完全相同"的预期不一致，需要检查渲染里是否还有别的随状态变化的内容。
    """)
    return diffs


def collect_frame_pairs(n_samples, seed):
    """采集 n_samples 组（上一帧, 当前帧, 当前状态）。状态均匀覆盖全部角度和角速度。

    返回：
        prev_frames [N, 64, 64, 3]  uint8
        cur_frames  [N, 64, 64, 3]  uint8
        states      [N, 2]          当前帧对应的 (θ, θ̇)
    """
    env = make_pixel_pendulum(seed)
    raw = env.unwrapped
    rng = np.random.default_rng(seed)
    prev_frames = np.zeros((n_samples, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    cur_frames = np.zeros_like(prev_frames)
    states = np.zeros((n_samples, 2), dtype=np.float32)
    for i in range(n_samples):
        raw.state = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-MAX_SPEED, MAX_SPEED)])
        prev_frames[i] = render_frame(env)
        raw.step(rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=1).astype(np.float32))
        cur_frames[i] = render_frame(env)
        states[i] = raw.state
    env.close()
    return prev_frames, cur_frames, states


def knn_predict(train_x, train_y, test_x, k=5):
    """k 近邻回归：在像素空间里找最像的 k 张训练图，把它们的标签取平均。

    train_x [N, D]，train_y [N, C]，test_x [M, D] → 预测 [M, C]
    """
    dist = torch.cdist(test_x, train_x)  # [M, N]
    idx = dist.topk(k, largest=False).indices  # [M, k]
    return train_y[idx].mean(dim=1)  # [M, C]


def velocity_from_frames(n_samples):
    """定量版本：从一帧 / 两帧里能读出什么。"""
    print(f"""
实验二：把"读不出速度"量化。随机摆放 {n_samples} 个状态（角度、角速度都均匀分布），
每个状态记下相邻的两帧画面。然后用最朴素的办法从画面读状态：k 近邻
（在像素空间里找最像的 5 张训练图，取它们标签的平均）。不训练任何网络，
读得出来就说明信息在画面里，读不出来就说明画面里没有。
    """)
    prev_frames, cur_frames, states = collect_frame_pairs(n_samples, SEED)
    # 这里只是一次几千张图的最近邻查找，放在 CPU 上就够快
    one = torch.as_tensor(cur_frames, dtype=torch.float32).flatten(1) / 255.0  # [N, 12288]
    two = torch.cat([torch.as_tensor(prev_frames, dtype=torch.float32).flatten(1) / 255.0, one], dim=1)  # [N, 24576]
    theta, velocity = torch.as_tensor(states[:, 0]), torch.as_tensor(states[:, 1])
    labels = torch.stack([torch.cos(theta), torch.sin(theta), velocity], dim=1)  # [N, 3]

    n_train = int(0.8 * n_samples)
    results = {}
    for name, x in [("单帧", one), ("相邻两帧", two)]:
        pred = knn_predict(x[:n_train], labels[:n_train], x[n_train:])  # [M, 3]
        true = labels[n_train:]
        angle_err = torch.atan2(pred[:, 1], pred[:, 0]) - torch.atan2(true[:, 1], true[:, 0])
        angle_err = torch.rad2deg(torch.atan2(torch.sin(angle_err), torch.cos(angle_err)).abs())
        results[name] = dict(
            pred_velocity=pred[:, 2].numpy(),
            angle_err=angle_err.median().item(),
            velocity_rmse=(pred[:, 2] - true[:, 2]).pow(2).mean().sqrt().item(),
        )
    true_velocity = velocity[n_train:].numpy()
    baseline = float(np.sqrt(np.mean((true_velocity - velocity[:n_train].mean().item()) ** 2)))

    print(f"  {'输入':<10}{'角度误差的中位数（度）':>16}{'角速度的均方根误差（rad/s）':>22}")
    for name, r in results.items():
        print(f"  {name:<10}{r['angle_err']:>20.1f}{r['velocity_rmse']:>26.2f}")
    print(f"  {'不看图，永远猜平均值':<10}{'—':>14}{baseline:>26.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (name, r) in zip(axes, results.items()):
        ax.scatter(true_velocity, r["pred_velocity"], s=8, alpha=0.5)
        ax.plot([-MAX_SPEED, MAX_SPEED], [-MAX_SPEED, MAX_SPEED], "k--", linewidth=1, label="完全读对")
        ax.set_xlabel("真实角速度（rad/s）")
        ax.set_ylabel("从画面读出的角速度（rad/s）")
        ax.set_title(f"输入：{name}（均方根误差 {r['velocity_rmse']:.2f}）")
        ax.legend(fontsize=9)
    save_figure(fig, OUT / "01-velocity-from-frames.png")

    single, pair = results["单帧"], results["相邻两帧"]
    print("\n怎么读这张表：\n")
    if single["velocity_rmse"] > 0.8 * baseline:
        print(f"    1. 单帧：角度读得出来（误差中位数 {single['angle_err']:.1f} 度），角速度的误差 {single['velocity_rmse']:.2f}，\n"
              f"       不比闭着眼睛猜平均值（{baseline:.2f}）好。单帧里没有速度信息。")
    else:
        print(f"    1. 单帧：角速度的误差 {single['velocity_rmse']:.2f}，明显小于猜平均值的 {baseline:.2f}。\n"
              f"       这和\"单帧里没有速度\"的预期不符，需要检查画面里是否泄露了速度。")
    if pair["velocity_rmse"] < 0.5 * baseline:
        print(f"    2. 相邻两帧：角速度的误差降到 {pair['velocity_rmse']:.2f}（猜平均值是 {baseline:.2f}）。\n"
              f"       两帧之间摆杆转过的角度就是速度，信息在\"历史\"里，不在任何一帧里。")
    else:
        print(f"    2. 相邻两帧：角速度的误差是 {pair['velocity_rmse']:.2f}（猜平均值是 {baseline:.2f}），改善有限。\n"
              f"       k 近邻在两帧拼起来的像素空间里太粗糙；速度信息仍然只可能来自帧与帧之间的差别。")
    print("""
把最近两帧叠起来，对摆这个任务确实能补上速度。但叠帧只能恢复"几帧之内看得出来"的信息，
叠几帧还是一个要调的超参数，而且它没有回答"有多大把握"。
更一般的办法是维护一个随时间递推更新的信念。下面在一个能手算的例子上把它做出来。
    """)
    return results, baseline


# ==================== 第三部分：一维卡尔曼滤波 ====================


def kalman_predict(mean, var, action, q_var):
    """预测步：按转移规律 s_t = s_{t-1} + a_{t-1} + w_t 把信念往前推一步。

    独立高斯变量相加，均值相加，方差相加。不确定性变大。
    返回：还没看观测时的信念 (m_t⁻, P_t⁻)
    """
    return mean + action, var + q_var


def kalman_update(mean_pred, var_pred, obs, r_var):
    """修正步：用观测 o_t = s_t + v_t 校准信念。不确定性变小。

    返回：看了观测之后的信念 (m_t, P_t)，以及卡尔曼增益 g_t
    """
    gain = var_pred / (var_pred + r_var)  # 观测越准（r_var 越小），增益越接近 1
    mean = mean_pred + gain * (obs - mean_pred)  # obs - mean_pred：观测和预测差多少（"意外"）
    var = (1.0 - gain) * var_pred
    return mean, var, gain


def verify_hand_calculation():
    """复现 CONCEPT.md 第 1.5 节的手算例子。"""
    print("\n" + "=" * 60)
    print("第三部分：一维卡尔曼滤波")
    print("=" * 60)

    print("""
设定：一辆一维小车，真实位置 s_t 看不到。每步按指令 a 移动（有误差），传感器读数也有误差：

    转移：s_t = s_{t-1} + a_{t-1} + w_t，   w_t ~ N(0, σ_w²)
    观测：o_t = s_t + v_t，               v_t ~ N(0, σ_v²)

信念始终是一个高斯分布 N(m_t, P_t)，只需要记均值和方差。两步的公式（推导见 CONCEPT.md 第 1.4 节）：

    预测：m⁻ = m + a                P⁻ = P + σ_w²
    修正：g = P⁻ / (P⁻ + σ_v²)      m = m⁻ + g · (o − m⁻)      P = (1 − g) · P⁻

先用 CONCEPT.md 第 1.5 节的手算例子核对代码：m_0 = 0，P_0 = 1，σ_w² = σ_v² = 0.25，指令恒为 1，
第 1 步观测到 1.6，第 2 步传感器掉线，第 3 步观测到 3.2。
    """)
    q, r = 0.25, 0.25
    mean, var = 0.0, 1.0
    print(f"  {'时刻':<6}{'预测后 m⁻':>10}{'预测后 P⁻':>10}{'增益 g':>10}{'修正后 m':>10}{'修正后 P':>10}")
    trace = []
    for t, obs in [(1, 1.6), (2, None), (3, 3.2)]:
        mean_pred, var_pred = kalman_predict(mean, var, 1.0, q)
        if obs is None:
            mean, var, gain = mean_pred, var_pred, float("nan")
        else:
            mean, var, gain = kalman_update(mean_pred, var_pred, obs, r)
        trace.append((mean, var))
        gain_text = "（无观测）" if obs is None else f"{gain:10.3f}"
        print(f"  {t:<8}{mean_pred:>10.3f}{var_pred:>10.3f}{gain_text:>10}{mean:>10.3f}{var:>10.3f}")
    expected = [(1.5, 0.208), (2.5, 0.458), (3.278, 0.185)]
    for (mean, var), (mean_ref, var_ref) in zip(trace, expected):
        assert abs(mean - mean_ref) < 2e-3 and abs(var - var_ref) < 2e-3, "和手算结果不一致"
    print("\n  ✓ 与 CONCEPT.md 的手算结果一致：方差 1.000 → 0.208（修正）→ 0.458（只预测）→ 0.185（修正）")


def simulate_cart(n_runs, n_steps, q_var, r_var, rng):
    """模拟小车。下标 i 对应时刻 t = i + 1。

    返回：
        actions [R, T]  到达第 t 步之前执行的指令 a_{t-1}
        states  [R, T]  真实位置 s_t（滤波器看不到）
        obs     [R, T]  带噪读数 o_t
    """
    actions = np.tile(0.5 * np.sin(np.arange(n_steps) / 5.0), (n_runs, 1))
    start = rng.normal(0.0, 1.0, size=(n_runs, 1))  # s_0 ~ N(0, 1)，和初始信念一致
    process_noise = rng.normal(0.0, np.sqrt(q_var), size=(n_runs, n_steps))
    states = start + np.cumsum(actions + process_noise, axis=1)
    obs = states + rng.normal(0.0, np.sqrt(r_var), size=(n_runs, n_steps))
    return actions, states, obs


def run_filter(obs, actions, has_obs, q_var, r_var, mean0=0.0, var0=1.0):
    """对一批序列运行卡尔曼滤波。

    obs、actions [R, T]；has_obs [T] 布尔数组，False 的时刻只做预测。
    返回：信念均值 [R, T]，信念方差 [T]（方差不依赖具体的观测值，所有序列共用）
    """
    mean, var = np.full(obs.shape[0], mean0), var0
    means, variances = [], []
    for i in range(obs.shape[1]):
        mean, var = kalman_predict(mean, var, actions[:, i], q_var)
        if has_obs[i]:
            mean, var, _ = kalman_update(mean, var, obs[:, i], r_var)
        means.append(mean)
        variances.append(var)
    return np.stack(means, axis=1), np.array(variances)


def kalman_experiment(n_runs):
    """带噪读数、中途掉线：信念怎么变。"""
    print(f"""
现在跑一个长一点的实验：{T_STEPS} 步，σ_w² = {Q_VAR}（移动比较准），σ_v² = {R_VAR}（传感器比较差），
第 {DROP_START} 到 {DROP_END - 1} 步传感器掉线。重复 {n_runs} 次，统计信念的误差。
    """)
    rng = np.random.default_rng(SEED)
    actions, states, obs = simulate_cart(n_runs, T_STEPS, Q_VAR, R_VAR, rng)
    t = np.arange(1, T_STEPS + 1)
    has_obs = ~((t >= DROP_START) & (t < DROP_END))
    means, variances = run_filter(obs, actions, has_obs, Q_VAR, R_VAR)
    filter_mse = ((means - states) ** 2).mean(axis=0)  # [T]，实测的平方误差
    obs_mse = ((obs - states) ** 2).mean(axis=0)  # [T]

    print(f"  {'时刻 t':<8}{'有观测吗':<8}{'信念方差 P_t（滤波器自己算的）':>20}{'信念均值的实测平方误差':>18}{'直接用读数的实测平方误差':>18}")
    for step in (1, 5, DROP_START - 1, DROP_START, DROP_START + 10, DROP_END - 1, DROP_END, DROP_END + 1, DROP_END + 5, T_STEPS):
        i = step - 1
        raw_text = f"{obs_mse[i]:>24.3f}" if has_obs[i] else f"{'（没有读数）':>20}"
        print(f"  {step:<10}{'有' if has_obs[i] else '无':<10}{variances[i]:>24.3f}{filter_mse[i]:>26.3f}{raw_text}")

    i_before, i_peak, i_back = DROP_START - 2, DROP_END - 2, DROP_END - 1
    steady = variances[i_before]
    growth = np.diff(variances[DROP_START - 2:DROP_END - 1])  # 掉线期间每一步的方差增量
    recover = next((k + 1 for k in range(T_STEPS - i_back) if variances[i_back + k] <= 1.05 * steady), None)
    calibration = np.abs(filter_mse / variances - 1.0).max()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    std = np.sqrt(variances)
    axes[0].fill_between(t, means[0] - 2 * std, means[0] + 2 * std, alpha=0.25, label="信念：均值 ± 2 倍标准差")
    axes[0].plot(t, means[0], linewidth=2, label="信念的均值 m_t")
    axes[0].plot(t, states[0], "k--", linewidth=1.5, label="真实位置（滤波器看不到）")
    axes[0].scatter(t[has_obs], obs[0][has_obs], s=14, color="tomato", label="带噪读数 o_t")
    axes[0].axvspan(DROP_START - 0.5, DROP_END - 0.5, color="gray", alpha=0.12, label="传感器掉线")
    axes[0].set_ylabel("位置")
    axes[0].set_title("一维卡尔曼滤波：一条序列上的信念")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].plot(t, variances, linewidth=2, label="信念方差 P_t（滤波器自己算的）")
    axes[1].plot(t, filter_mse, ".", color="C3", label=f"信念均值的实测平方误差（{n_runs} 次平均）")
    axes[1].axhline(R_VAR, color="tomato", linestyle=":", label=f"单次读数的方差 σ_v² = {R_VAR}")
    axes[1].axvspan(DROP_START - 0.5, DROP_END - 0.5, color="gray", alpha=0.12)
    axes[1].set_xlabel("时刻 t")
    axes[1].set_ylabel("方差")
    axes[1].set_title("预测让方差变大，修正让方差变小")
    axes[1].legend(fontsize=8)
    save_figure(fig, OUT / "01-kalman-filter.png")

    obs_rmse = float(np.sqrt(obs_mse[has_obs].mean()))
    filter_rmse = float(np.sqrt(filter_mse[i_before]))
    print("\n怎么读这些结果：\n")
    if filter_rmse < obs_rmse:
        print(f"    1. 滤波比单次读数准。掉线前信念均值的均方根误差是 {filter_rmse:.3f}，直接用读数是 {obs_rmse:.3f}。\n"
              f"       每一次读数都很吵，但信念把很多次读数的信息攒在了一起。")
    else:
        print(f"    1. 这次运行里滤波的误差（{filter_rmse:.3f}）没有小于直接用读数（{obs_rmse:.3f}），与预期不符。")
    if np.allclose(growth, Q_VAR):
        print(f"    2. 掉线期间只有预测、没有修正，方差每一步恰好增加 σ_w² = {Q_VAR}，线性增长：\n"
              f"       从 {steady:.3f} 涨到 {variances[i_peak]:.3f}。这是\"开环展开时不确定性越滚越大\"的最简版本。")
    else:
        print(f"    2. 掉线期间方差每步的增量在 {growth.min():.4f} 到 {growth.max():.4f} 之间，不等于 σ_w² = {Q_VAR}，需要检查。")
    recover_text = f"再过 {recover - 1} 步回到掉线前水平的 5% 以内" if recover else "到序列结束还没有回到掉线前的水平"
    print(f"    3. 读数恢复后的第一次修正就把方差从 {variances[i_peak] + Q_VAR:.3f} 压到 {variances[i_back]:.3f}，{recover_text}。\n"
          f"       刚恢复时信念很没把握，增益大，几乎全听读数的；之后增益逐步回落。")
    print(f"    4. 滤波器自己算的方差 P_t 和实测的平方误差对得上（全程相对偏差最大 {calibration:.1%}，\n"
          f"       来自 {n_runs} 次重复的统计波动）。信念不只给出猜测，还如实给出了这个猜测有多准。\n"
          f"       这有一个前提：转移和观测的规律是已知的、正确的。")
    return dict(filter_rmse=filter_rmse, obs_rmse=obs_rmse, steady=steady, peak=variances[i_peak],
                back=variances[i_back], recover=recover, calibration=calibration)


# ==================== 第四部分：增益是什么，能不能学出来 ====================


def steady_state(q_var, r_var, n_iters=2000):
    """反复做"预测 + 修正"直到方差不再变化，返回稳态的增益和修正后的方差。"""
    var, gain = 1.0, 0.0
    for _ in range(n_iters):
        _, var_pred = kalman_predict(0.0, var, 0.0, q_var)
        _, var, gain = kalman_update(0.0, var_pred, 0.0, r_var)
    return gain, var


def gain_table():
    """增益怎样随传感器的好坏变化。"""
    print("\n" + "=" * 60)
    print("第四部分：增益是什么，能不能从数据里学出来")
    print("=" * 60)

    print(f"""
卡尔曼增益（Kalman gain）g 在 0 和 1 之间，是修正时"听观测的"比例。
固定 σ_w² = {Q_VAR}，换不同质量的传感器，看长期运行后稳定下来的增益：
    """)
    print(f"  {'传感器噪声方差 σ_v²':<18}{'稳态增益 g':>12}{'稳态信念方差 P':>16}{'信念的标准差 / 单次读数的标准差':>28}")
    rows = []
    for r_var in (0.0001, 0.01, 0.04, 1.0, 25.0):
        gain, var = steady_state(Q_VAR, r_var)
        rows.append((r_var, gain, var))
        print(f"  {r_var:<22}{gain:>12.3f}{var:>18.4f}{np.sqrt(var / r_var):>24.2f}")
    print(f"""
传感器很准时（σ_v² = {rows[0][0]}），增益是 {rows[0][1]:.3f}：基本只听读数。
传感器很差时（σ_v² = {rows[-1][0]}），增益是 {rows[-1][1]:.3f}：基本只信自己的预测，每次读数只挪一小步。
最后一列始终小于 1：信念比任何一次单独的读数都准，传感器越差，"攒信息"的好处越大。
    """)
    return rows


def surprise_loss(gain, obs, actions, burn_in=10):
    """用固定增益做滤波，返回"下一个读数的预测误差"的平均平方值。

    obs、actions [B, T]。每一步先预测（m⁻ = m + a），再看这一步的读数和预测差多少（意外），
    最后按增益修正。损失只用到读数，从头到尾没有用真实位置。
    """
    mean = torch.zeros(obs.shape[0])
    total = 0.0
    for i in range(obs.shape[1]):
        mean_pred = mean + actions[:, i]  # 预测步
        surprise = obs[:, i] - mean_pred  # 读数和预测差多少
        if i >= burn_in:  # 开头几步受初始信念影响，不计入
            total = total + (surprise**2).mean()
        mean = mean_pred + gain * surprise  # 修正步
    return total / (obs.shape[1] - burn_in)


def learn_gain_from_data(n_iters):
    """不用卡尔曼公式，只靠"把下一个读数预测准"来学增益。"""
    print("""
卡尔曼增益的公式用到了 σ_w² 和 σ_v²。如果这两个数不知道呢？
换一个办法：把增益当成一个可学习的参数，训练目标只有一条——
"每一步，先预测、再看读数，让读数和预测的差（意外）尽量小"。
训练只用读数和指令，不用真实位置（真实位置本来就看不到）。
    """)
    rng = np.random.default_rng(SEED + 1)
    actions, _, obs = simulate_cart(512, 100, Q_VAR, R_VAR, rng)
    # 只有一个标量参数，放在 CPU 上
    actions, obs = torch.as_tensor(actions, dtype=torch.float32), torch.as_tensor(obs, dtype=torch.float32)

    raw = torch.zeros((), requires_grad=True)  # 增益 = sigmoid(raw)，初始值 0.5
    optimizer = torch.optim.Adam([raw], lr=0.05)
    gains = []
    for _ in range(n_iters):
        loss = surprise_loss(torch.sigmoid(raw), obs, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        gains.append(torch.sigmoid(raw).item())
    learned = gains[-1]

    kalman_gain, kalman_var = steady_state(Q_VAR, R_VAR)
    with torch.no_grad():
        grid = np.linspace(0.02, 0.9, 45)
        grid_loss = [surprise_loss(torch.tensor(float(g)), obs, actions).item() for g in grid]
        learned_loss = surprise_loss(torch.tensor(learned), obs, actions).item()
    theory_loss = kalman_var + Q_VAR + R_VAR  # 稳态时"意外"的方差 = P⁻ + σ_v²

    print(f"  学出来的增益：        {learned:.3f}（从 0.500 出发，训练 {n_iters} 步）")
    print(f"  卡尔曼公式的稳态增益：{kalman_gain:.3f}")
    print(f"  训练后的损失（意外的平均平方）：{learned_loss:.3f}    理论最小值 P⁻ + σ_v²：{theory_loss:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].plot(np.arange(1, n_iters + 1), gains, linewidth=2, label="学出来的增益")
    axes[0].axhline(kalman_gain, color="k", linestyle="--", label=f"卡尔曼公式的稳态增益 {kalman_gain:.3f}")
    axes[0].set_xlabel("训练步数")
    axes[0].set_ylabel("增益 g")
    axes[0].set_title("只靠\"把下一个读数预测准\"学增益")
    axes[0].legend(fontsize=9)
    axes[1].plot(grid, grid_loss, linewidth=2)
    axes[1].axvline(kalman_gain, color="k", linestyle="--", label="卡尔曼增益")
    axes[1].scatter([learned], [learned_loss], color="C3", zorder=3, label="学出来的增益")
    axes[1].set_xlabel("增益 g")
    axes[1].set_ylabel("意外的平均平方（损失）")
    axes[1].set_title("损失随增益的变化")
    axes[1].legend(fontsize=9)
    save_figure(fig, OUT / "01-learned-gain.png")

    if abs(learned - kalman_gain) < 0.03:
        print(f"""
学出来的增益（{learned:.3f}）和卡尔曼公式给出的（{kalman_gain:.3f}）基本一致。
没有人告诉优化器噪声有多大，它只是在让"预测"和"读数"尽量对得上，结果就找到了贝叶斯公式的答案。
    """)
    else:
        print(f"""
学出来的增益（{learned:.3f}）和卡尔曼公式给出的（{kalman_gain:.3f}）还差 {abs(learned - kalman_gain):.3f}。
训练步数不够时会这样（冒烟模式只训练很少几步）；右图的损失曲线在卡尔曼增益处取最小值，方向是对的。
    """)
    return learned, kalman_gain


def bridge_to_rssm():
    """从卡尔曼滤波到 RSSM。"""
    print("""
这个小实验就是本章后面要做的事情的缩影：

    ┌────────────────┬──────────────────────────┬──────────────────────────────────┐
    │                │ 卡尔曼滤波                 │ RSSM（讲义 04 起）                 │
    ├────────────────┼──────────────────────────┼──────────────────────────────────┤
    │ 状态是什么       │ 人指定的物理量（位置）      │ 模型自己发明的隐状态 (h_t, z_t)      │
    │ 转移规律         │ 已知的线性方程             │ 学出来的 GRU 和先验网络             │
    │ 观测规律         │ 已知的线性方程             │ 学出来的解码器                     │
    │ 预测步           │ m⁻ = m + a，P⁻ = P + σ_w² │ 先验 p(z_t | h_t)                 │
    │ 修正步           │ 增益 g 的公式              │ 后验网络 q(z_t | h_t, o_t)，学出来的 │
    │ 没有观测时       │ 只做预测，方差增大          │ 只走先验，叫"想象"                  │
    │ 靠什么训练       │ 不用训练                   │ 让预测和观测对得上（序列 ELBO）       │
    └────────────────┴──────────────────────────┴──────────────────────────────────┘

卡尔曼滤波要求转移和观测的规律已知、线性、噪声是高斯的。像素版的摆三条都不满足：
没人告诉我们状态是什么，更没有从状态到 64×64 画面的线性方程。
RSSM 的办法是三样全部用神经网络学出来。训练它的目标函数（序列 ELBO）是下一节的内容。
    """)


# ==================== 第五部分：练习与思考 ====================


def exercises():
    """练习题与参考答案。数值答案由代码现场算出。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    mean_pred, var_pred = kalman_predict(0.0, 4.0, 2.0, 1.0)
    mean, var, gain = kalman_update(mean_pred, var_pred, 3.0, 1.0)
    steady_gain, steady_var = steady_state(Q_VAR, R_VAR)
    drop = 20
    width = 4 * np.sqrt(steady_var + drop * Q_VAR)
    print(f"""
练习 1：手算一步
    初始信念 m_0 = 0，P_0 = 4。σ_w² = 1，σ_v² = 1。指令 a_0 = 2，观测到 o_1 = 3。
    求预测后的信念、增益、修正后的信念。再用 kalman_predict() / kalman_update() 核对。

练习 1 参考答案：
    预测：m⁻ = 0 + 2 = {mean_pred:.0f}，P⁻ = 4 + 1 = {var_pred:.0f}
    增益：g = 5 / (5 + 1) = {gain:.3f}
    修正：m = 2 + {gain:.3f} × (3 − 2) = {mean:.3f}，P = (1 − {gain:.3f}) × 5 = {var:.3f}
    预测后的方差（5）远大于读数的方差（1），所以修正后的均值几乎就是读数，
    方差也降到了比读数的方差还小一点的 {var:.3f}。

练习 2：掉线多久，信念会宽到没用
    本节的设定（σ_w² = {Q_VAR}，σ_v² = {R_VAR}）下，稳态的信念方差是 {steady_var:.3f}。
    传感器连续掉线 {drop} 步之后，"均值 ± 2 倍标准差"的带子有多宽？

练习 2 参考答案：
    掉线期间方差每步加 σ_w²：{steady_var:.3f} + {drop} × {Q_VAR} = {steady_var + drop * Q_VAR:.3f}，
    标准差 {np.sqrt(steady_var + drop * Q_VAR):.3f}，带子的宽度是 4 倍标准差 = {width:.2f}。
    对照 kalman_experiment() 的图：掉线结束时带子的宽度就是这个数。
    规划时连续十几步都没有观测，模型状态的不确定性也是这样一步步攒起来的。

练习 3：模型写错了会怎样
    把 run_filter() 用的 q_var 故意传成真实值的 1/100（滤波器以为小车走得非常准），
    其余不变，重新跑 kalman_experiment()。信念方差和实测平方误差还对得上吗？

练习 3 参考答案：
    对不上。滤波器以为自己的预测非常可靠，算出的增益和方差都偏小，
    对读数几乎不理会，于是信念的均值跟不上真实位置的漂移，实测误差会明显大于它自称的方差。
    "方差如实反映误差"只在模型正确时成立。学出来的模型同理：
    先验的方差只是模型的自我评价，模型不准的地方，这个评价也不可信（CONCEPT.md 第 14 节 Q7）。

思考题 1：把最近两帧叠起来就能读出速度，为什么还要信念状态？

思考题 1 参考答案：
    叠帧只能恢复固定窗口之内的信息，窗口多长是要调的超参数；被遮挡了几十步的物体它无能为力。
    它也不表达把握：刚看到第一帧时速度完全未知，这件事叠帧表示不出来。
    更重要的是，叠帧只给了"当前局面的表示"，没有给"怎么往前推演"。
    规划需要一个能在里面想象未来的状态，这要靠带预测步的模型。

思考题 2：Pendulum-v1 是确定性环境，为什么还要用概率（分布）来描述状态？

思考题 2 参考答案：
    环境确定不等于观测者确定。只看到第一帧时，角速度可能是任何值，
    智能体对状态的认识是一个很宽的分布，看了几帧才收窄。
    不确定性来自"看不全"，不来自环境的随机性。

思考题 3：learn_gain_from_data() 的损失里没有真实位置，为什么还能学到正确的增益？

思考题 3 参考答案：
    增益太小，信念跟不上真实位置的漂移，下一步的预测会偏；
    增益太大，信念被每一次读数的噪声带着跑，下一步的预测也会偏。
    两种错误都会体现在"下一个读数和预测的差"上，所以只看读数就够了。
    RSSM 的训练是同一个道理：没有人提供真实状态，
    模型只被要求把观测预测准（重建项）并且少被观测"意外"到（KL 项）。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    quick = is_quick()
    if quick:
        print("（冒烟模式：样本数和训练步数都缩得很小，下面的数字没有参考价值）\n")
    n_samples, n_runs, n_iters = (300, 300, 30) if quick else (3000, 5000, 400)

    introduction()
    show_ambiguity()
    velocity_from_frames(n_samples)
    verify_hand_calculation()
    kalman_experiment(n_runs)
    gain_table()
    learn_gain_from_data(n_iters)
    bridge_to_rssm()
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：02-sequential-vae-elbo.py（把 VAE 的 ELBO 推广到序列）")
    print("=" * 60)


if __name__ == "__main__":
    main()
