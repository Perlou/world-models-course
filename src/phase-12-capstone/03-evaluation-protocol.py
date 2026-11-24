"""
评估协议
========

⚠️ 草稿状态：本文件写成时运行环境不允许执行 Python，尚未实际运行过。
   定稿前必须完整跑一遍、核对输出，并把下面的「预计运行时间」换成实测值。

学习目标：
    1. 从零写出评估要用的统计量：t 区间、bootstrap、IQM、Wilson 区间、改进概率，并各用一个手算例子校验
    2. 用已知分布的合成数据检查区间的覆盖率：种子少的时候，哪种区间还靠得住
    3. 给一个世界模型做两层评估：开环误差曲线与可信视野，闭环的回报（带种子、区间和上界）
    4. 用"想象回报与真实回报之差"看出模型被规划器钻了空子；用秩相关判断模型能不能用来做选择
    5. 把这套统计用到真实的评估记录上：两个方法、5 个训练种子、固定的评估种子

核心概念：
    - 标准误 SE = s / √n；独立重复的单位是训练种子，不是评估回合
    - 百分位 bootstrap：对种子有放回重采样，看统计量有多分散；种子少时偏窄
    - 四分位均值（IQM）：去掉最低和最高各 25% 之后的平均
    - 开环误差曲线 e(k)、平凡基线 e_copy(k)、可信视野 k*
    - 想象回报 Ĵ 与真实回报 J；Spearman 秩相关 ρ_S

前置知识：
    - Phase 1 的 04-compounding-error.py（开环展开）、06-cem-and-mppi.py（CEM）
    - 均值、方差、正态分布

对应概念文档：
    - CONCEPT.md 第 5 节「评估协议与统计」、第 6 节「世界模型的评估：开环与闭环」

预计运行时间：待实测（本稿尚未运行）
"""

import itertools
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.planning import cem_plan, shift_plan

plt = setup_matplotlib()
OUT = output_dir("phase-12")
SEED = 0
MAX_TORQUE, MAX_SPEED, DT = 2.0, 8.0, 0.05
PLAN_HORIZON = 20
# 模型是两层 128 单元的 MLP，规划时的批量只有 100，放在 CPU 上更快


# ==================== 第一部分：评估的两个问题 ====================


def introduction():
    """这个数字可信吗？这个模型能信多远？"""
    print("=" * 60)
    print("第一部分：评估要回答的两个问题")
    print("=" * 60)

    print("""
问题一：这个数字可信吗？（第二到四部分）

    训练种子 i ──▶ 一次完整训练 ──▶ 一个策略 ──▶ E 个评估回合 ──▶ 得分 x_i（E 个回合的平均）

    n 个种子的得分 x_1 … x_n  ──▶  均值 ± 区间   （独立重复的单位是种子，通常只有 5 个）

    这一半全部用 NumPy 从零写。每个函数先在一个能手算的例子上校验，
    再在已知真值的合成数据上检查：它声称的 95%，实际是百分之多少。

问题二：这个世界模型能信多远？（第五到七部分）

    开环：只给起点和动作，让模型自己往前推 k 步            ──▶ e(k)、可信视野 k*
    桥接：同一条动作序列，在模型里的回报 Ĵ 对真实回报 J      ──▶ 偏差、秩相关
    闭环：把模型接上规划器去控制真实环境                    ──▶ 回报 ± 区间，外加上界

    演示对象是 Phase 1 的摆。两个"方法"：同一个 MLP 动力学模型，一个只用 1 个随机回合训练
    （故意训练得不充分），一个用 20 个随机回合训练。真实状态和完美模型都现成可用，
    所以每一层指标都有真值可以对照。
    """)


# ==================== 第二部分：统计量，从零实现 ====================


def t_cdf(x, df, n_grid=20001):
    """t 分布的累积分布函数（x ≥ 0）：对概率密度做梯形积分。df 可以不是整数。"""
    log_c = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
    grid = np.linspace(0.0, x, n_grid)
    pdf = np.exp(log_c - (df + 1) / 2 * np.log1p(grid ** 2 / df))
    return 0.5 + float(((pdf[1:] + pdf[:-1]) / 2).sum() * (grid[1] - grid[0]))


def t_critical(df, confidence=0.95):
    """t 分布的双侧临界值 κ：用二分法解 t_cdf(κ) = 0.5 + confidence / 2。"""
    lo, hi = 0.0, 200.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < 0.5 + confidence / 2 else (lo, mid)
    return (lo + hi) / 2


def normal_critical(confidence=0.95):
    """标准正态分布的双侧临界值（95% 时是 1.96）。"""
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < 0.5 + confidence / 2 else (lo, mid)
    return (lo + hi) / 2


def mean_and_ci(x, confidence=0.95):
    """均值、样本标准差 s、标准误 SE、t 区间。x 是各个种子的得分 [n]。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mean, s = x.mean(), x.std(ddof=1)                        # ddof=1：分母是 n − 1
    se = s / math.sqrt(n)
    kappa = t_critical(n - 1, confidence)
    return {"mean": mean, "s": s, "se": se, "kappa": kappa, "lo": mean - kappa * se, "hi": mean + kappa * se, "n": n}


def diff_ci(a, b, confidence=0.95):
    """两个方法均值之差 δ = mean(b) − mean(a) 的区间。自由度用 Welch 近似。"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)  # SE_A²、SE_B²
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    kappa = t_critical(df, confidence)
    delta = b.mean() - a.mean()
    return {"delta": delta, "se": se, "df": df, "kappa": kappa, "lo": delta - kappa * se, "hi": delta + kappa * se}


def iqm(x):
    """四分位均值：沿最后一维排序，两端各去掉 ⌊N/4⌋ 个，对剩下的求平均。支持批量输入 [..., N]。"""
    x = np.sort(np.asarray(x, dtype=np.float64), axis=-1)
    n = x.shape[-1]
    cut = n // 4
    return x[..., cut:n - cut].mean(axis=-1)


def bootstrap_ci(x, stat, n_resamples=10000, confidence=0.95, rng=None):
    """百分位 bootstrap 区间。stat 要能沿最后一维计算（例如 lambda v: v.mean(axis=-1)、iqm）。

    返回 (统计量, 下界, 上界, 重采样得到的 R 个统计量)。
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.asarray(x, dtype=np.float64)
    idx = rng.integers(0, len(x), size=(n_resamples, len(x)))            # [R, n]：有放回地抽下标
    values = stat(x[idx])                                                # [R]
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(values, [tail, 100 - tail])
    return float(stat(x)), float(lo), float(hi), values


def stratified_bootstrap_iqm(scores, n_resamples=5000, confidence=0.95, rng=None):
    """多任务的 IQM 与分层 bootstrap 区间。scores [M, n]：M 个任务，每个任务 n 个种子。

    重采样只在每个任务内部的种子之间做，任务本身不重采样。
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    scores = np.asarray(scores, dtype=np.float64)
    n_tasks, n_seeds = scores.shape
    idx = rng.integers(0, n_seeds, size=(n_resamples, n_tasks, n_seeds))                         # [R, M, n]
    resampled = np.take_along_axis(np.broadcast_to(scores, idx.shape), idx, axis=2)              # [R, M, n]
    values = iqm(resampled.reshape(n_resamples, -1))                                             # [R]
    tail = (1 - confidence) / 2 * 100
    lo, hi = np.percentile(values, [tail, 100 - tail])
    return float(iqm(scores.ravel())), float(lo), float(hi)


def wilson_interval(n_success, n_total, confidence=0.95):
    """成功率的 Wilson 区间。成功率接近 0 或 1、回合数少的时候也不会塌成一个点。"""
    kappa = normal_critical(confidence)
    p = n_success / n_total
    denom = 1 + kappa ** 2 / n_total
    center = (p + kappa ** 2 / (2 * n_total)) / denom
    half = kappa / denom * math.sqrt(p * (1 - p) / n_total + kappa ** 2 / (4 * n_total ** 2))
    return max(center - half, 0.0), min(center + half, 1.0)


def probability_of_improvement(a, b):
    """改进概率 P(B > A)：随机挑 A 的一次运行和 B 的一次运行，B 更高的概率。打平算半个。"""
    a, b = np.asarray(a, dtype=np.float64)[:, None], np.asarray(b, dtype=np.float64)[None, :]
    return float((b > a).mean() + 0.5 * (b == a).mean())


def rankdata(x):
    """名次（从 1 开始）；并列的取平均名次。"""
    x = np.asarray(x, dtype=np.float64)
    ranks = np.empty(len(x))
    ranks[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    for value in np.unique(x):
        tie = x == value
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(a, b):
    """Spearman 秩相关：两列数各自换成名次之后的 Pearson 相关系数。"""
    with np.errstate(invalid="ignore", divide="ignore"):                 # 全部并列时相关系数没有定义，返回 nan
        return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def rank_agreement(j_hat, j, n_resamples=2000, rng=None):
    """想象回报与真实回报的排序一致性：Spearman 秩相关，以及对候选做 bootstrap 得到的 95% 区间。"""
    rng = rng if rng is not None else np.random.default_rng(0)
    j_hat, j = np.asarray(j_hat, dtype=np.float64), np.asarray(j, dtype=np.float64)
    idx = rng.integers(0, len(j), size=(n_resamples, len(j)))
    values = np.array([spearman(j_hat[i], j[i]) for i in idx])
    lo, hi = np.nanpercentile(values, [2.5, 97.5])
    return {"rho": spearman(j_hat, j), "lo": float(lo), "hi": float(hi)}


def verify_statistics():
    """每个函数用 CONCEPT.md 里的手算例子（或闭式解）校验一遍。"""
    print("\n" + "=" * 60)
    print("第二部分：统计量从零实现，逐个校验")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    mean_stat = lambda v: v.mean(axis=-1)

    # t 分布临界值：对照教科书上的表
    table = {2: 4.303, 4: 2.776, 9: 2.262, 19: 2.093, 29: 2.045}
    kappas = {df: t_critical(df) for df in table}
    for df, expected in table.items():
        assert abs(kappas[df] - expected) < 2e-3, f"t 临界值不对：df = {df}"
    assert abs(normal_critical() - 1.960) < 1e-3

    # 均值与 t 区间：5 个种子的成功率
    a = [0.62, 0.70, 0.58, 0.74, 0.66]
    r = mean_and_ci(a)
    assert abs(r["mean"] - 0.66) < 1e-9 and abs(r["s"] - 0.0632) < 1e-4 and abs(r["se"] - 0.0283) < 1e-4
    assert abs(r["lo"] - 0.581) < 2e-3 and abs(r["hi"] - 0.739) < 2e-3

    # bootstrap：三个得分 2、4、9，27 种重采样全部枚举
    x = np.array([2.0, 4.0, 9.0])
    exact = np.sort([np.mean(c) for c in itertools.product(x, repeat=3)])
    exact_se = exact.std()
    formula_se = x.std(ddof=1) / math.sqrt(3)
    assert len(exact) == 27 and abs(exact.mean() - 5.0) < 1e-9 and abs(exact_se - 1.70) < 5e-3
    assert abs(exact[2] - 2.667) < 1e-3 and abs(exact[-3] - 7.333) < 1e-3          # 两端各去掉 2 个之后的范围
    assert abs(exact_se / formula_se - math.sqrt(2 / 3)) < 1e-9
    _, _, _, sampled = bootstrap_ci(x, mean_stat, n_resamples=20000, rng=rng)
    assert abs(sampled.std() - exact_se) < 0.05, "随机重采样的标准误应接近枚举的结果"

    # IQM：8 个得分，其中一个种子训崩
    scores = [0.10, 0.52, 0.55, 0.58, 0.60, 0.63, 0.66, 0.95]
    fixed = [0.50] + scores[1:]
    assert abs(iqm(scores) - 0.590) < 1e-9 and abs(iqm(fixed) - 0.590) < 1e-9
    assert abs(np.mean(scores) - 0.574) < 1e-3 and abs(np.mean(fixed) - 0.624) < 1e-3
    assert abs(np.median(scores) - 0.590) < 1e-9

    # Wilson 区间：20 个回合全部成功
    w_lo, w_hi = wilson_interval(20, 20)
    assert abs(w_lo - 0.839) < 2e-3 and abs(w_hi - 1.0) < 1e-9

    # 差的区间与改进概率
    b = [0.68, 0.80, 0.70, 0.82, 0.70]
    d = diff_ci(a, b)
    assert abs(d["delta"] - 0.08) < 1e-9 and abs(d["se"] - 0.0405) < 1e-4 and abs(d["df"] - 8.0) < 0.05
    assert abs(d["lo"] - (-0.013)) < 2e-3 and abs(d["hi"] - 0.173) < 2e-3
    poi = probability_of_improvement(a, b)
    assert abs(poi - 0.80) < 1e-9

    # Spearman：5 个策略；随机排序碰巧达到 0.8 的概率
    real, model = [0.30, 0.45, 0.60, 0.75, 0.90], [0.45, 0.30, 0.70, 0.65, 0.95]
    rho = spearman(real, model)
    lucky = sum(spearman(range(5), perm) >= 0.8 - 1e-9 for perm in itertools.permutations(range(5)))
    assert abs(rho - 0.80) < 1e-9 and lucky == 8

    # 最大值的期望：n 个标准正态变量里最大的那个
    c_table = {2: 0.564, 3: 0.846, 5: 1.163, 10: 1.539}
    c_sim = {n: float(rng.standard_normal((200_000, n)).max(axis=1).mean()) for n in c_table}
    for n, expected in c_table.items():
        assert abs(c_sim[n] - expected) < 0.01, f"c_{n} 的模拟值与表不符"

    print(f"""
全部断言通过。每个函数对照的是 CONCEPT.md 第 5、6 节里的手算例子：

    函数                              例子                              手算 / 查表         函数算出
    t_critical()                      自由度 4（5 个种子）               2.776              {kappas[4]:.3f}
    mean_and_ci()                     0.62 0.70 0.58 0.74 0.66          0.66，[0.581, 0.739]  {r['mean']:.2f}，[{r['lo']:.3f}, {r['hi']:.3f}]
    bootstrap_ci()                    得分 2、4、9，枚举 27 种重采样      标准误 1.70         枚举 {exact_se:.3f}，随机抽 2 万次 {sampled.std():.3f}
    iqm()                             8 个得分，一个种子训崩             0.590              {iqm(scores):.3f}
    wilson_interval()                 20 战全胜                         [0.84, 1.00]       [{w_lo:.3f}, {w_hi:.3f}]
    diff_ci()                         A 与 B 各 5 个种子                 [−0.013, 0.173]    [{d['lo']:.3f}, {d['hi']:.3f}]，自由度 {d['df']:.2f}
    probability_of_improvement()      同上                              0.80               {poi:.2f}
    spearman()                        5 个策略                          0.80               {rho:.2f}
    （枚举）                           5 个候选随机排序，ρ ≥ 0.8 的排法    8 / 120            {lucky} / 120
    （模拟）                           n 个标准正态里最大值的期望 c_5      1.163              {c_sim[5]:.3f}

三个值得停一下的数：
    - bootstrap 的标准误（{exact_se:.2f}）比公式 s / √n（{formula_se:.2f}）小，比值 {exact_se / formula_se:.3f} = √(2/3)：
      它把样本当总体，方差的分母是 n 而不是 n − 1。种子越少，偏窄得越多。
    - A、B 的均值差 8 个百分点，差的区间仍然包含 0。5 个种子分不开这个差距。
    - 5 个策略时，即使模型里的评估和真实成功率毫无关系，也有 {lucky / 120:.1%} 的概率碰巧得到 ρ ≥ 0.8。
    """)
    return c_sim


# ==================== 第三部分：区间的覆盖率 ====================


def coverage_experiment(n_trials, n_resamples):
    """在已知真值的合成数据上，数一数"95% 区间"实际盖住真值的比例。"""
    print("\n" + "=" * 60)
    print("第三部分：95% 区间，实际盖住真值多少次")
    print("=" * 60)

    print(f"""
合成实验：每次"实验"从已知分布里抽 n 个种子的得分，算均值的 95% 区间，看它有没有盖住真实均值。
重复 {n_trials} 次，盖住的比例就是这种区间的实际覆盖率。名义值是 95%。

    总体一（正态）    得分 ~ N(0.60, 0.10²)
    总体二（会训崩）  每个种子有 20% 的概率训崩：得分 ~ N(0.10, 0.05²)；否则 ~ N(0.70, 0.05²)。真实均值 0.58

    三种区间：t 区间；错用正态临界值 1.96 的区间；百分位 bootstrap（重采样 {n_resamples} 次）
    """)
    rng = np.random.default_rng(SEED)

    def draw_normal(size):
        return rng.normal(0.60, 0.10, size)

    def draw_collapse(size):
        return np.where(rng.random(size) < 0.2, rng.normal(0.10, 0.05, size), rng.normal(0.70, 0.05, size))

    populations = [("正态", draw_normal, 0.60), ("会训崩", draw_collapse, 0.58)]
    n_values = [3, 5, 10, 20]
    z = normal_critical()
    coverage = {}
    print(f"    {'总体':<8}{'种子数 n':>8}{'t 区间':>10}{'正态 1.96':>12}{'bootstrap':>12}{'t 区间平均半宽':>16}{'bootstrap 平均半宽':>18}")
    for name, draw, mu in populations:
        for n in n_values:
            x = draw((n_trials, n))                                          # [试验数, n]
            mean, se = x.mean(axis=1), x.std(axis=1, ddof=1) / math.sqrt(n)
            kappa = t_critical(n - 1)
            cover_t = np.abs(mean - mu) <= kappa * se
            cover_z = np.abs(mean - mu) <= z * se
            lows, highs = [], []
            for start in range(0, n_trials, 500):                            # 分块，免得一次占太多内存
                chunk = x[start:start + 500]
                idx = rng.integers(0, n, size=(len(chunk), n_resamples, n))  # [块, R, n]
                boot = np.take_along_axis(np.broadcast_to(chunk[:, None, :], idx.shape), idx, axis=2).mean(axis=2)
                lo, hi = np.percentile(boot, [2.5, 97.5], axis=1)
                lows.append(lo)
                highs.append(hi)
            lows, highs = np.concatenate(lows), np.concatenate(highs)
            cover_b = (lows <= mu) & (mu <= highs)
            coverage[(name, n)] = (cover_t.mean(), cover_z.mean(), cover_b.mean())
            print(f"    {name:<8}{n:>8}{cover_t.mean():>12.1%}{cover_z.mean():>12.1%}{cover_b.mean():>12.1%}"
                  f"{(kappa * se).mean():>18.3f}{((highs - lows) / 2).mean():>20.3f}")

    margin = 2 * math.sqrt(0.95 * 0.05 / n_trials)                           # 覆盖率自己的统计误差（约两个标准误）
    t5, z5, b5 = coverage[("正态", 5)]
    t5c, _, b5c = coverage[("会训崩", 5)]
    print(f"\n  （{n_trials} 次试验下，覆盖率这个数自己有约 ±{margin:.1%} 的统计误差。）")
    print("\n  怎么读这张表（n = 5 那一行，也就是毕业项目的典型情况）：")
    if abs(t5 - 0.95) <= margin:
        print(f"    - 正态总体上，t 区间的覆盖率 {t5:.1%}，和名义的 95% 在误差范围内一致。")
    else:
        print(f"    - ⚠️ 正态总体上 t 区间的覆盖率是 {t5:.1%}，偏离 95% 超过了统计误差，需要检查实现。")
    if b5 < 0.95 - margin:
        print(f"    - 百分位 bootstrap 只有 {b5:.1%}，错用 1.96 的区间只有 {z5:.1%}：两者都偏窄。"
              "\n      5 个种子的 bootstrap 区间应当读作「至少这么宽」。")
    else:
        print(f"    - 这次运行里百分位 bootstrap 的覆盖率是 {b5:.1%}，没有明显低于 95%（错用 1.96 的区间是 {z5:.1%}）。")
    if t5c < 0.95 - margin:
        print(f"    - 换成会训崩的总体，连 t 区间也只剩 {t5c:.1%}（bootstrap {b5c:.1%}）。t 区间假设得分近似正态，"
              "\n      而「偶尔训崩」的分布有两个峰。5 个种子里恰好一个都没崩时，样本看起来很整齐，区间就窄得过分。"
              "\n      统计方法补不回样本里没有的信息：这种情况下要报告崩了几个，并且加种子。")
    else:
        print(f"    - 换成会训崩的总体，t 区间的覆盖率是 {t5c:.1%}（bootstrap {b5c:.1%}），这次没有明显低于 95%。")
    first_b, last_b = coverage[("正态", n_values[0])][2], coverage[("正态", n_values[-1])][2]
    print(f"    - 正态总体上 bootstrap 的覆盖率：{n_values[0]} 个种子时 {first_b:.1%}，{n_values[-1]} 个种子时 {last_b:.1%}。"
          "\n      种子越多，几种区间越接近；它们之间的差别主要是种子太少造成的。")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (name, _, _) in zip(axes, populations):
        for j, (label, style) in enumerate([("t 区间", "o-"), ("正态临界值 1.96", "s--"), ("百分位 bootstrap", "^-")]):
            ax.plot(n_values, [coverage[(name, n)][j] for n in n_values], style, label=label)
        ax.axhline(0.95, color="gray", linestyle=":", label="名义值 95%")
        ax.set_xscale("log")
        ax.set_xticks(n_values, [str(n) for n in n_values])
        ax.set_xlabel("种子数 n")
        ax.set_title(f"总体：{name}")
    axes[0].set_ylabel(f"实际覆盖率（{n_trials} 次试验）")
    axes[0].legend(fontsize=8)
    save_figure(fig, OUT / "03-interval-coverage.png")
    return coverage


# ==================== 第四部分：两层随机性与聚合 ====================


def aggregation_demos(c_sim):
    """两层方差的公式对不对；挑最好的种子会虚高多少；多任务的 IQM 与分层 bootstrap。"""
    print("\n" + "=" * 60)
    print("第四部分：两层随机性、最大化偏差、多任务聚合（合成数据）")
    print("=" * 60)
    rng = np.random.default_rng(SEED + 1)
    n_trials = 400 if is_quick() else 4000

    # ---- 两层方差：Var[x̄] = (σ_seed² + σ_ep² / E) / n ----
    sigma_seed, sigma_ep = 0.10, 0.45
    print(f"\n  两层方差。种子之间 σ_seed = {sigma_seed}，回合之间 σ_ep = {sigma_ep}，模拟 {n_trials} 次「跑 n 个种子、每个评估 E 个回合」：\n")
    print(f"    {'n':>4}{'E':>6}{'公式给出的均值标准差':>20}{'模拟得到的':>14}")
    formulas = {}
    for n, n_episodes in [(5, 10), (5, 50), (5, 500), (10, 50)]:
        formula = math.sqrt((sigma_seed ** 2 + sigma_ep ** 2 / n_episodes) / n)
        formulas[(n, n_episodes)] = formula
        seed_effect = rng.normal(0, sigma_seed, (n_trials, n, 1))
        episodes = seed_effect + rng.normal(0, sigma_ep, (n_trials, n, n_episodes))      # [试验, n, E]
        simulated = episodes.mean(axis=(1, 2)).std()
        if not is_quick():
            assert abs(simulated / formula - 1) < 0.06, "模拟与两层方差公式不符"
        print(f"    {n:>4}{n_episodes:>6}{formula:>22.4f}{simulated:>18.4f}")
    floor = sigma_seed / math.sqrt(5)
    print(f"\n  回合数从 50 加到 500，评估成本乘 10，公式给出的标准差只从 {formulas[(5, 50)]:.4f} 降到 {formulas[(5, 500)]:.4f}："
          f"5 个种子时它不可能低于 σ_seed / √5 = {floor:.4f}。"
          f"\n  回合数保持 50、把种子从 5 个加到 10 个（最后一行），标准差是 {formulas[(10, 50)]:.4f}，降得更多；代价是多训练 5 次。")

    # ---- 最大化偏差 ----
    scores = rng.normal(0.66, 0.063, (n_trials * 10, 5))
    best = scores.max(axis=1).mean()
    print(f"""
  最大化偏差。假设一个方法的真实水平是 0.66，种子之间的标准差 0.063，每次跑 5 个种子：
    "5 个种子的平均"的期望        {scores.mean():.3f}
    "5 个种子里最好的一个"的期望   {best:.3f}      公式 μ + c_5 σ = 0.66 + {c_sim[5]:.3f} × 0.063 = {0.66 + c_sim[5] * 0.063:.3f}
  只报告最好的种子，平白多出 {best - 0.66:.3f}，和方法没有关系。挑检查点、用测试回合调超参数是同一种偏差。""")

    # ---- 多任务：IQM 与分层 bootstrap ----
    task_means = np.array([0.55, 0.70, 0.40, 0.85])
    multi = np.clip(task_means[:, None] + rng.normal(0, 0.06, (4, 5)), 0, None)          # [M = 4, n = 5]
    multi[2, 1] = 0.02                                                                   # 任务 3 的一个种子训崩了
    point, lo, hi = stratified_bootstrap_iqm(multi, rng=rng)
    healthy = multi.copy()
    healthy[2, 1] = task_means[2]
    print(f"""
  多任务聚合（本节只有一个真实任务，这里用合成的归一化得分演示）：4 个任务 × 5 个种子，其中一个种子训崩。

    聚合量        含训崩的种子       把那个种子换成正常水平      变化
    均值          {multi.mean():.3f}              {healthy.mean():.3f}                    {healthy.mean() - multi.mean():+.3f}
    中位数        {np.median(multi):.3f}              {np.median(healthy):.3f}                    {np.median(healthy) - np.median(multi):+.3f}
    IQM          {iqm(multi.ravel()):.3f}              {iqm(healthy.ravel()):.3f}                    {iqm(healthy.ravel()) - iqm(multi.ravel()):+.3f}

    IQM = {point:.3f}，分层 bootstrap 的 95% 区间 [{lo:.3f}, {hi:.3f}]（只在每个任务内部对种子重采样）
""")
    d_mean = healthy.mean() - multi.mean()
    d_iqm = iqm(healthy.ravel()) - iqm(multi.ravel())
    if abs(d_iqm) < 0.5 * abs(d_mean):
        print(f"  那个训崩的种子让均值动了 {d_mean:+.3f}，IQM 只动了 {d_iqm:+.3f}。IQM 对它不敏感，这是优点，也是代价："
              "\n  光看 IQM 看不出「20 次训练里崩了 1 次」。所以稳健统计量要和每个种子的明细一起报告。")
    else:
        print(f"  ⚠️ 这次运行里 IQM 的变化（{d_iqm:+.3f}）没有明显小于均值的变化（{d_mean:+.3f}）：换回去的那个得分落进了中间一半，"
              "\n  IQM 也跟着动了。以上表的数字为准；稳健统计量仍要和每个种子的明细一起报告。")


# ==================== 摆：真实模型、学到的模型、规划 ====================


def wrap_angle(theta):
    return ((theta + math.pi) % (2 * math.pi)) - math.pi


def true_step(states, actions):
    """摆的真实动力学（与 Pendulum-v1 一致）。states [N, 2] = (θ, θ̇)，actions [N, 1] → 下一状态 [N, 2]。"""
    theta, theta_dot = states[..., 0], states[..., 1]
    u = actions[..., 0].clamp(-MAX_TORQUE, MAX_TORQUE)
    theta_dot_new = (theta_dot + (15.0 * torch.sin(theta) + 3.0 * u) * DT).clamp(-MAX_SPEED, MAX_SPEED)
    return torch.stack([theta + theta_dot_new * DT, theta_dot_new], dim=-1)


def reward_fn(states, actions):
    """已知的奖励函数。states [N, 2]，actions [N, 1] → [N]。"""
    return -(wrap_angle(states[..., 0]) ** 2 + 0.1 * states[..., 1] ** 2 + 0.001 * actions[..., 0] ** 2)


def features(states):
    """把角度展开成 (cos θ, sin θ)，避免 θ 与 θ + 2π 被当成两个状态。[..., 2] → [..., 3]。"""
    return torch.stack([torch.cos(states[..., 0]), torch.sin(states[..., 0]), states[..., 1]], dim=-1)


class DynamicsMLP(nn.Module):
    """学到的动力学模型（第一次讲解在 Phase 1 的 03-learning-dynamics-mlp.py，这里是紧凑版）。"""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.register_buffer("y_mean", torch.zeros(2))
        self.register_buffer("y_std", torch.ones(2))

    def raw(self, states, actions):
        """标准化空间里的状态变化量预测 [N, 2]。"""
        x = torch.cat([features(states)[..., :2], states[..., 1:] / MAX_SPEED, actions / MAX_TORQUE], dim=-1)   # [N, 4]
        return self.net(x)

    def forward(self, states, actions):
        """states [N, 2]，actions [N, 1] → 预测的下一状态 [N, 2]。角速度按已知的物理上限裁剪。"""
        nxt = states + self.raw(states, actions) * self.y_std + self.y_mean
        return torch.stack([nxt[..., 0], nxt[..., 1].clamp(-MAX_SPEED, MAX_SPEED)], dim=-1)


def collect_random_episodes(n_episodes, seed):
    """随机策略采集。返回状态 [E, 201, 2] 和动作 [E, 200, 1]。"""
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)
    all_states, all_actions = [], []
    for i in range(n_episodes):
        env.reset(seed=seed + i)
        states, actions = [env.unwrapped.state.copy()], []
        for _ in range(200):
            action = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=1).astype(np.float32)
            env.step(action)
            actions.append(action)
            states.append(env.unwrapped.state.copy())
        all_states.append(states)
        all_actions.append(actions)
    env.close()
    return (torch.tensor(np.asarray(all_states), dtype=torch.float32),
            torch.tensor(np.asarray(all_actions), dtype=torch.float32))


def train_model(states, actions, n_steps, seed):
    """在转移数据上训练一个 DynamicsMLP。states [E, T+1, 2]，actions [E, T, 1]。"""
    s, a, s_next = states[:, :-1].reshape(-1, 2), actions.reshape(-1, 1), states[:, 1:].reshape(-1, 2)
    torch.manual_seed(seed)
    model = DynamicsMLP()
    delta = s_next - s
    model.y_mean.copy_(delta.mean(dim=0))
    model.y_std.copy_(delta.std(dim=0).clamp_min(1e-6))
    target = (delta - model.y_mean) / model.y_std
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(n_steps):
        idx = torch.randint(0, len(s), (128,), generator=generator)
        loss = ((model.raw(s[idx], a[idx]) - target[idx]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.eval()


def make_score_fn(step_fn, state):
    """规划器用的打分函数：从 state 出发，用 step_fn 展开候选动作序列，返回累计奖励。"""

    @torch.no_grad()
    def score_fn(actions):                                               # actions [N, H, 1] → [N]
        states = state.expand(actions.shape[0], 2)
        total = torch.zeros(actions.shape[0])
        for h in range(actions.shape[1]):
            total = total + reward_fn(states, actions[:, h])
            states = step_fn(states, actions[:, h])
        return total

    return score_fn


def plan(step_fn, state, mean, generator, n_iters=3):
    """一次 CEM 规划（utils.planning.cem_plan，第一次讲解在 Phase 1 的 06 课）。"""
    return cem_plan(make_score_fn(step_fn, state), mean, generator, action_low=-MAX_TORQUE, action_high=MAX_TORQUE,
                    n_candidates=100, n_elites=10, n_iters=n_iters, init_std=1.0, min_std=0.1)


def run_mpc_episode(step_fn, eval_seed, episode_len):
    """用 CEM-MPC 在真实环境里跑一个回合，返回真实回报。step_fn 为 None 时用随机策略。"""
    env = gym.make("Pendulum-v1")
    env.reset(seed=eval_seed)                                            # 评估种子决定初始状态
    generator = torch.Generator().manual_seed(eval_seed)
    mean, total = torch.zeros(PLAN_HORIZON, 1), 0.0
    for _ in range(episode_len):
        if step_fn is None:
            action = (torch.rand(1, generator=generator) * 2 - 1) * MAX_TORQUE
        else:
            state = torch.as_tensor(env.unwrapped.state, dtype=torch.float32)
            mean = plan(step_fn, state, mean, generator)
            action = mean[0]
            mean = shift_plan(mean)
        _, reward, _, _, _ = env.step(action.numpy().astype(np.float32))
        total += float(reward)
    env.close()
    return total


def build_models(n_seeds, quick):
    """两个"方法"各训练 n_seeds 个模型；另采一份留出数据用于开环评估。"""
    print("\n" + "=" * 60)
    print("第五部分：开环评估")
    print("=" * 60)

    # 先确认手写的真实动力学和环境一致（单步）
    env = gym.make("Pendulum-v1")
    env.reset(seed=SEED)
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for _ in range(100):
        state = torch.as_tensor(env.unwrapped.state, dtype=torch.float32)[None]
        action = rng.uniform(-MAX_TORQUE, MAX_TORQUE, size=1).astype(np.float32)
        pred = true_step(state, torch.from_numpy(action)[None])[0].numpy()
        env.step(action)
        worst = max(worst, float(np.abs(pred - env.unwrapped.state).max()))
    env.close()
    assert worst < 1e-4, "手写的真实动力学应当和环境逐步一致"

    specs = {"A：1 个随机回合": (1, 60 if quick else 300), "B：20 个随机回合": (4 if quick else 20, 100 if quick else 3000)}
    models = {name: [] for name in specs}
    start = time.time()
    for name, (n_episodes, n_steps) in specs.items():
        for i in range(n_seeds):
            # 训练种子同时决定：采到哪些数据、网络的初始化、取批次的顺序
            states, actions = collect_random_episodes(n_episodes, seed=100 * (i + 1))
            models[name].append(train_model(states, actions, n_steps, seed=i))
    heldout = collect_random_episodes(3 if quick else 10, seed=9000)     # 留出的回合，任何模型都没见过
    print(f"""
  手写的真实动力学 true_step() 与 Pendulum-v1 的单步最大偏差 {worst:.1e}（float32 的舍入量级），下面把它当作真实环境用。

  两个"方法"，各 {n_seeds} 个训练种子（训练共用时 {time.time() - start:.1f} 秒）：
    A：1 个随机回合     每个种子采 1 个随机回合（200 条转移），训练 {specs['A：1 个随机回合'][1]} 步。故意训练得不充分
    B：20 个随机回合    每个种子采 {specs['B：20 个随机回合'][0]} 个随机回合，训练 {specs['B：20 个随机回合'][1]} 步
  模型结构、规划器、评估协议完全相同。训练种子同时决定采到的数据、网络初始化和取批次的顺序。""")
    return models, heldout


# ==================== 第五部分：开环误差曲线与可信视野 ====================


@torch.no_grad()
def open_loop_error_curve(model, states, actions, feature_std):
    """开环误差曲线。

    states [N, K+1, 2]：N 个起点之后的真实状态序列（第 0 个是起点，相当于上下文，C = 1）
    actions [N, K, 1]：对应的动作
    返回 e [K]、e_copy [K]：模型展开到第 k 步的误差，以及"假设状态不动"的误差。
    距离 d：在 (cos θ, sin θ, θ̇) 上各维按标准差归一化后的均方误差。
    """
    pred, start = states[:, 0], features(states[:, 0])
    e, e_copy = [], []
    for k in range(actions.shape[1]):
        pred = model(pred, actions[:, k])                                # 只喂动作，状态用模型自己上一步的输出
        truth = features(states[:, k + 1])
        e.append((((features(pred) - truth) / feature_std) ** 2).mean().item())
        e_copy.append((((start - truth) / feature_std) ** 2).mean().item())
    return np.array(e), np.array(e_copy)


def credible_horizon(e, e_copy, c=0.5, ks=None):
    """可信视野 k*：误差比 e(k) / e_copy(k) 从第一步起一直低于阈值 c 的最大步数。第一步就不满足时返回 0。

    误差比只穿过阈值一次时，这和定义 k* = max{k : e(k) < c · e_copy(k)} 完全相同；
    取"从头起一直满足"的版本，是为了不被很远处两条曲线都饱和之后的偶然回落骗到。
    """
    ks = np.arange(1, len(e) + 1) if ks is None else np.asarray(ks)
    ok = np.asarray(e) < c * np.asarray(e_copy)
    n_ok = len(ok) if ok.all() else int(np.argmin(ok))
    return int(ks[n_ok - 1]) if n_ok > 0 else 0


def open_loop_experiment(models, heldout, k_max):
    """每个模型一条开环误差曲线，和复制基线画在一起，求可信视野。"""
    # 手算例子（CONCEPT.md 第 6.3 节）
    assert credible_horizon([0.010, 0.030, 0.070, 0.180, 0.330, 0.420], [0.030, 0.120, 0.200, 0.290, 0.340, 0.360],
                            ks=[1, 5, 10, 20, 30, 40]) == 10

    states, actions = heldout
    starts = range(0, actions.shape[1] - k_max + 1, 10)                  # 起点分散在回合的不同阶段
    seq_states = torch.cat([states[:, t:t + k_max + 1] for t in starts])              # [N, K+1, 2]
    seq_actions = torch.cat([actions[:, t:t + k_max] for t in starts])                # [N, K, 1]
    feature_std = features(states.reshape(-1, 2)).std(dim=0)
    print(f"\n  留出数据：{states.shape[0]} 个随机回合，取 {len(seq_states)} 个起点，每个开环展开 {k_max} 步。阈值 c = 0.5。\n")

    ks = np.arange(1, k_max + 1)
    show = [k for k in (1, 5, 10, 15, 20, 30, 50) if k <= k_max]
    curves, horizons = {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    print(f"    {'':<20}" + "".join(f"{'k=' + str(k):>9}" for k in show) + f"{'可信视野 k*（各种子）':>26}")
    for (name, members), color in zip(models.items(), ["C3", "C0"]):
        results = [open_loop_error_curve(m, seq_states, seq_actions, feature_std) for m in members]
        e = np.array([r[0] for r in results])                            # [种子数, K]
        e_copy = results[0][1]
        curves[name] = (e, e_copy)
        horizons[name] = [credible_horizon(row, e_copy) for row in e]
        print(f"    {name + ' 误差比':<20}" + "".join(f"{e.mean(axis=0)[k - 1] / e_copy[k - 1]:>9.2f}" for k in show)
              + f"      {horizons[name]}，中位数 {int(np.median(horizons[name]))}")
        axes[0].plot(ks, e.mean(axis=0), color=color, linewidth=2, label=f"{name}（{len(members)} 个种子的平均）")
        axes[0].fill_between(ks, e.min(axis=0), e.max(axis=0), color=color, alpha=0.15)
        for row in e:
            axes[1].plot(ks, row / e_copy, color=color, linewidth=0.8, alpha=0.7)
    print(f"    {'复制基线 e_copy(k)':<20}" + "".join(f"{e_copy[k - 1]:>9.3f}" for k in show))
    axes[0].plot(ks, e_copy, "k--", label="复制起点状态 e_copy(k)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("开环展开步数 k")
    axes[0].set_ylabel("归一化均方误差（对数坐标）")
    axes[0].set_title("开环误差曲线（阴影：各种子的最小–最大范围）")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.5, color="gray", linestyle=":", label="阈值 c = 0.5")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="和不预测一样")
    axes[1].axvline(PLAN_HORIZON, color="green", linestyle=":", label=f"规划视野 H = {PLAN_HORIZON}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("开环展开步数 k")
    axes[1].set_ylabel("误差比 e(k) / e_copy(k)")
    axes[1].set_title("误差比（每个种子一条线；红 A，蓝 B）")
    axes[1].legend(fontsize=8)
    save_figure(fig, OUT / "03-open-loop-error.png")

    med = {name: float(np.median(h)) for name, h in horizons.items()}
    name_a, name_b = list(models)
    print(f"\n  可信视野的中位数：A 是 {med[name_a]:.0f} 步，B 是 {med[name_b]:.0f} 步；规划器要看 {PLAN_HORIZON} 步。")
    for name in (name_a, name_b):
        verdict = "覆盖了规划视野" if med[name] >= PLAN_HORIZON else "比规划视野短，规划器打分的后半段建立在不可信的预测上"
        print(f"    {name}：{verdict}。")
    print("  注意这是在随机策略的留出数据上量的。规划器会把摆带到随机策略到不了的地方（甩上去、停在顶上），"
          "\n  那里的误差这条曲线量不到。这正是下一部分要查的事。")
    return curves, horizons


# ==================== 第六部分：想象回报与真实回报 ====================


@torch.no_grad()
def imagined_vs_real_return(model, start_states, action_seqs):
    """同一批动作序列，分别在模型里和真实环境里执行。

    start_states [N, 2]，action_seqs [N, H, 1] → (想象回报 Ĵ [N]，真实回报 J [N])
    """
    s_model, s_real = start_states, start_states
    j_hat, j = torch.zeros(len(start_states)), torch.zeros(len(start_states))
    for h in range(action_seqs.shape[1]):
        a = action_seqs[:, h]
        j_hat, j = j_hat + reward_fn(s_model, a), j + reward_fn(s_real, a)
        s_model, s_real = model(s_model, a), true_step(s_real, a)
    return j_hat.numpy(), j.numpy()


def make_policies():
    """六个水平不同的策略：输入状态 [N, 2] 和随机数发生器，输出动作 [N, 1]。"""

    def swing_up(gain):
        def policy(states, generator):
            theta, theta_dot = wrap_angle(states[:, 0]), states[:, 1]
            direction = torch.where(theta_dot >= 0, torch.ones_like(theta_dot), -torch.ones_like(theta_dot))
            pump = gain * direction                                      # 顺着运动方向推，给摆加能量
            hold = -8.0 * theta - 2.0 * theta_dot                        # 靠近顶点时用 PD 控制稳住
            return torch.where(torch.cos(theta) > 0.9, hold, pump).clamp(-MAX_TORQUE, MAX_TORQUE)[:, None]
        return policy

    return {
        "不施力": lambda s, g: torch.zeros(len(s), 1),
        "随机力矩": lambda s, g: (torch.rand(len(s), 1, generator=g) * 2 - 1) * MAX_TORQUE,
        "恒定力矩 +2": lambda s, g: torch.full((len(s), 1), MAX_TORQUE),
        "甩起再稳住（力矩 0.5）": swing_up(0.5),
        "甩起再稳住（力矩 1）": swing_up(1.0),
        "甩起再稳住（力矩 2）": swing_up(2.0),
    }


@torch.no_grad()
def policy_return(step_fn, policy, starts, n_steps, seed):
    """让一个策略在 step_fn（模型或真实动力学）里闭环跑 n_steps 步，返回各起点回报的平均。"""
    generator = torch.Generator().manual_seed(seed)
    states, total = starts.clone(), torch.zeros(len(starts))
    for _ in range(n_steps):
        a = policy(states, generator)
        total = total + reward_fn(states, a)
        states = step_fn(states, a)
    return float(total.mean())


def bridge_experiment(models, quick):
    """桥接指标：随机序列对规划器选出的序列；策略排序的一致性。"""
    print("\n" + "=" * 60)
    print("第六部分：想象回报与真实回报")
    print("=" * 60)

    n_starts, horizon = (8, 20) if quick else (64, 20)
    iteration_list = [1, 2] if quick else [1, 2, 4, 8]
    generator = torch.Generator().manual_seed(SEED)
    # 起点的分布和 Pendulum-v1 的 reset 一样：角度任意，角速度在 [-1, 1]
    starts = torch.stack([(torch.rand(n_starts, generator=generator) * 2 - 1) * math.pi,
                          torch.rand(n_starts, generator=generator) * 2 - 1], dim=1)
    random_seqs = (torch.rand(n_starts, horizon, 1, generator=generator) * 2 - 1) * MAX_TORQUE

    print(f"""
  每个模型取种子 0 的那一个。{n_starts} 个起点，动作序列长 {horizon} 步。两组候选：
    随机序列         每个起点一条均匀随机的动作序列
    CEM 选出的序列   每个起点用 CEM 在这个模型里规划（100 条候选，取不同的迭代轮数），拿最后的均值序列

  偏差 = 想象回报 − 真实回报 的平均。正的偏差说明模型把这些序列想得比实际好。
    """)
    bias_curves = {}
    print(f"    {'模型':<18}{'候选':<14}{'想象回报 Ĵ':>12}{'真实回报 J':>12}{'偏差 Ĵ − J':>12}{'平均绝对误差':>12}")
    for name, members in models.items():
        model = members[0]
        rows = [("随机序列", *imagined_vs_real_return(model, starts, random_seqs))]
        for n_iters in iteration_list:
            plans = torch.stack([plan(model, s, torch.zeros(horizon, 1), generator, n_iters=n_iters) for s in starts])
            rows.append((f"CEM {n_iters} 轮", *imagined_vs_real_return(model, starts, plans)))
        bias_curves[name] = [float((j_hat - j).mean()) for _, j_hat, j in rows]
        for label, j_hat, j in rows:
            print(f"    {name:<18}{label:<14}{j_hat.mean():>12.1f}{j.mean():>12.1f}{(j_hat - j).mean():>12.1f}{np.abs(j_hat - j).mean():>12.1f}")

    print("\n  怎么读（按实测数字）：")
    for name, bias in bias_curves.items():
        random_bias, planned = bias[0], bias[1:]
        if planned[-1] > max(abs(random_bias), 1.0) and planned[-1] > planned[0]:
            print(f"    {name}：随机序列上的偏差是 {random_bias:+.1f}，CEM 选出的序列上是 {planned[0]:+.1f} → {planned[-1]:+.1f}，"
                  "为正，而且优化得越用力越大。\n        这是模型被钻空子的特征：规划器专挑模型高估的动作序列。")
        elif planned[-1] > max(abs(random_bias), 1.0):
            print(f"    {name}：CEM 选出的序列上偏差为正（{planned[0]:+.1f} → {planned[-1]:+.1f}），大于随机序列上的 {random_bias:+.1f}，"
                  "但没有随迭代轮数单调增大。")
        else:
            print(f"    {name}：CEM 选出的序列上的偏差（{planned[0]:+.1f} → {planned[-1]:+.1f}）和随机序列上的（{random_bias:+.1f}）"
                  "在同一个量级，\n        这次运行里没有看到明显的被钻空子。")

    # ---- 排序一致性之一：同一个起点上的随机候选（规划器第一轮面对的局面）----
    n_candidates = 50 if quick else 200
    bottom = torch.tensor([[math.pi, 0.0]]).repeat(n_candidates, 1)                              # 摆垂在正下方
    candidates = (torch.rand(n_candidates, horizon, 1, generator=generator) * 2 - 1) * MAX_TORQUE
    print(f"\n  排序一致性（一）：从正下方出发的 {n_candidates} 条随机动作序列，按想象回报排序和按真实回报排序的 Spearman 秩相关：\n")
    for name, members in models.items():
        j_hat, j = imagined_vs_real_return(members[0], bottom, candidates)
        r = rank_agreement(j_hat, j, n_resamples=300 if quick else 2000)
        print(f"    {name:<18} ρ_S = {r['rho']:.3f}，95% 区间 [{r['lo']:.3f}, {r['hi']:.3f}]；偏差 {np.mean(j_hat - j):+.1f}")

    # ---- 排序一致性之二：六个水平不同的策略，在模型里评估对在真实环境里评估 ----
    policies = make_policies()
    n_steps = 40 if quick else 100
    real = np.array([policy_return(true_step, p, starts, n_steps, SEED) for p in policies.values()])
    print(f"\n  排序一致性（二）：{len(policies)} 个水平不同的策略，各从 {n_starts} 个起点闭环跑 {n_steps} 步。"
          "\n  「真实」一列用真实动力学，其余两列把环境换成学到的模型（Phase 11 的设定：在世界模型里评估策略）：\n")
    in_model = {name: np.array([policy_return(members[0], p, starts, n_steps, SEED) for p in policies.values()])
                for name, members in models.items()}
    print(f"    {'策略':<22}{'真实回报':>10}" + "".join(f"{'在 ' + name[0] + ' 里':>12}" for name in models))
    for i, label in enumerate(policies):
        print(f"    {label:<22}{real[i]:>12.1f}" + "".join(f"{in_model[name][i]:>14.1f}" for name in models))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    agreement = {}
    for (name, values), color in zip(in_model.items(), ["C3", "C0"]):
        agreement[name] = rank_agreement(values, real, n_resamples=300 if quick else 2000)
        r = agreement[name]
        print(f"    {name}：ρ_S = {r['rho']:.2f}，95% 区间 [{r['lo']:.2f}, {r['hi']:.2f}]；"
              f"偏差 {np.mean(values - real):+.1f}，平均绝对误差 {np.mean(np.abs(values - real)):.1f}")
        axes[1].scatter(real, values, color=color, label=f"在 {name} 里评估（ρ_S = {r['rho']:.2f}）")
    widest = max(r["hi"] - r["lo"] for r in agreement.values())
    print(f"\n  只有 {len(policies)} 个候选，秩相关的 bootstrap 区间最宽的有 {widest:.2f}。"
          f"{len(policies)} 个候选一共只有 {math.factorial(len(policies))} 种排序，"
          "\n  这个 ρ 和 5 个种子的均值一样，要配区间，要谨慎下结论。")

    x = np.arange(len(iteration_list) + 1)
    for (name, bias), color in zip(bias_curves.items(), ["C3", "C0"]):
        axes[0].plot(x, bias, "o-", color=color, label=name)
    axes[0].axhline(0, color="gray", linestyle=":")
    axes[0].set_xticks(x, ["随机序列"] + [f"CEM {n} 轮" for n in iteration_list])
    axes[0].set_ylabel("想象回报 − 真实回报（平均）")
    axes[0].set_title(f"优化得越用力，模型越被高估？（{n_starts} 个起点，视野 {horizon}）")
    axes[0].legend(fontsize=8)
    lims = [min(real.min(), min(v.min() for v in in_model.values())), max(real.max(), max(v.max() for v in in_model.values()))]
    axes[1].plot(lims, lims, "k--", linewidth=0.8, label="完全一致")
    axes[1].set_xlabel("真实环境里的平均回报")
    axes[1].set_ylabel("世界模型里的平均回报")
    axes[1].set_title(f"{len(policies)} 个策略：在模型里评估 对 在真实环境里评估")
    axes[1].legend(fontsize=8)
    save_figure(fig, OUT / "03-imagined-vs-real.png")
    return bias_curves, agreement


# ==================== 第七部分：闭环评估与真实评估记录上的统计 ====================


def closed_loop_experiment(models, horizons, n_episodes, episode_len):
    """按协议做闭环评估，然后把第二部分的统计函数用到这些真实记录上。"""
    print("\n" + "=" * 60)
    print("第七部分：闭环评估，以及真实评估记录上的统计")
    print("=" * 60)

    eval_seeds = [1000 + j for j in range(n_episodes)]                   # 固定的评估种子，所有方法、所有训练种子共用
    n_seeds = len(next(iter(models.values())))
    print(f"""
  评估协议（在看到结果之前定下）：
    任务            Pendulum-v1，每个评估回合 {episode_len} 步
    训练种子        每个方法 {n_seeds} 个（编号 0–{n_seeds - 1}）
    评估回合        每个训练种子 {n_episodes} 个，评估种子 {eval_seeds[0]}–{eval_seeds[-1]}（与训练数据的种子不相交）
    规划器          CEM，100 条候选 × 3 轮，视野 {PLAN_HORIZON}，热启动；各方法相同
    聚合            种子层得分 = {n_episodes} 个回合的平均回报；报告均值与 95% t 区间
    主比较          B 对 A，差的 95% 区间（Welch）
    参照            随机策略；真实模型 + 同一个规划器（上界）
    """)
    start = time.time()
    names = list(models)
    records = np.array([[[run_mpc_episode(m, s, episode_len) for s in eval_seeds] for m in models[name]] for name in names])   # [2, n, E]
    oracle = np.array([run_mpc_episode(true_step, s, episode_len) for s in eval_seeds])       # [E]
    random_policy = np.array([run_mpc_episode(None, s, episode_len) for s in eval_seeds])     # [E]
    print(f"  共 {records.size + 2 * n_episodes} 个评估回合，用时 {time.time() - start:.0f} 秒。原始记录的形状 {records.shape} = [方法, 训练种子, 评估回合]。\n")

    seed_scores = records.mean(axis=2)                                   # [2, n]：种子层得分
    stats = [mean_and_ci(x) for x in seed_scores]
    span = oracle.mean() - random_policy.mean()
    print(f"    {'方法':<20}{'各训练种子的得分':<46}{'均值':>8}{'95% t 区间':>22}{'归一化得分':>10}")
    for name, x, st in zip(names, seed_scores, stats):
        print(f"    {name:<20}{'  '.join(f'{v:7.1f}' for v in x):<46}{st['mean']:>10.1f}"
              f"{'[' + format(st['lo'], '.1f') + ', ' + format(st['hi'], '.1f') + ']':>24}{(st['mean'] - random_policy.mean()) / span:>12.2f}")
    print(f"    {'随机策略':<20}{'（不训练，只有回合层的波动）':<46}{random_policy.mean():>10.1f}{'':>24}{0.0:>12.2f}")
    print(f"    {'真实模型 + CEM（上界）':<18}{'（同上）':<50}{oracle.mean():>10.1f}{'':>24}{1.0:>12.2f}")
    print("    归一化得分 =（得分 − 随机策略）/（上界 − 随机策略）。")

    # ---- 主比较 ----
    d = diff_ci(seed_scores[0], seed_scores[1])
    poi = probability_of_improvement(seed_scores[0], seed_scores[1])
    rng = np.random.default_rng(SEED)
    poi_boot = [probability_of_improvement(rng.choice(seed_scores[0], n_seeds), rng.choice(seed_scores[1], n_seeds)) for _ in range(2000)]
    poi_lo, poi_hi = np.percentile(poi_boot, [2.5, 97.5])
    print(f"""
  主比较（B − A）：
    均值之差 δ = {d['delta']:.1f}，标准误 {d['se']:.1f}，Welch 自由度 {d['df']:.1f}，95% 区间 [{d['lo']:.1f}, {d['hi']:.1f}]
    改进概率 P(B > A) = {poi:.2f}，bootstrap 95% 区间 [{poi_lo:.2f}, {poi_hi:.2f}]""")
    if d["lo"] > 0:
        print(f"    差的区间不含 0：在这份协议下，可以说 B 的回报高于 A。{n_seeds} 个种子时改进概率的区间仍然很宽，它只是辅助。")
    elif d["hi"] < 0:
        print("    差的区间不含 0，而且是负的：在这份协议下 A 的回报高于 B，和「数据多的更好」的预期相反，值得追查原因。")
    else:
        print(f"    差的区间包含 0：{n_seeds} 个种子分不出 A 和 B。这不等于两者一样好，只是现有的数据下不了结论。")

    # ---- 两层方差：该加种子还是加回合 ----
    print("\n  两层方差的分解（由原始记录估计；σ_ep 是同一个策略不同回合之间的标准差，σ_seed 是不同训练种子之间的）：\n")
    print(f"    {'方法':<20}{'σ_ep':>10}{'σ_seed':>10}{'σ_ep² / E':>12}{'σ_seed²':>12}   结论")
    for name, rec, x in zip(names, records, seed_scores):
        var_ep = rec.var(axis=1, ddof=1).mean()
        var_seed = max(x.var(ddof=1) - var_ep / n_episodes, 0.0)         # 种子层得分的方差里扣掉回合噪声的那一份
        advice = "种子之间的差别是主要来源，要缩小区间只能加种子" if var_seed > var_ep / n_episodes else "回合噪声还占大头，加评估回合仍然有用"
        print(f"    {name:<20}{math.sqrt(var_ep):>10.1f}{math.sqrt(var_seed):>10.1f}{var_ep / n_episodes:>12.1f}{var_seed:>12.1f}   {advice}")

    # ---- 配对评估 ----
    paired_var = np.mean([np.var(records[1, i] - records[0, i], ddof=1) for i in range(n_seeds)])
    unpaired_var = np.mean([records[1, i].var(ddof=1) + records[0, i].var(ddof=1) for i in range(n_seeds)])
    corr = np.mean([np.corrcoef(records[0, i], records[1, i])[0, 1] for i in range(n_seeds)])
    print(f"""
  配对评估。所有模型在同一批初始状态上评估。取 A、B 同编号的模型两两比较，看"逐回合之差"的方差：
    两个模型在同一批回合上的回报，相关系数平均 {corr:.2f}
    配对之差的方差 {paired_var:.0f}；如果各自用不同的初始状态（方差相加）{unpaired_var:.0f}；比值 {paired_var / unpaired_var:.2f}""")
    if paired_var < unpaired_var:
        print("    难的初始状态对两个模型都难，共用评估种子把这部分波动消掉了。")
    else:
        print("    这次运行里配对没有降低方差：两个模型在各个初始状态上的表现相关性很弱。共用评估种子仍然是应该做的，它没有坏处。")

    # ---- 成功率 ----
    threshold = 0.5 * (oracle.mean() + random_policy.mean())
    success = records > threshold
    print(f"\n  换成成功率看一眼。把「回报高于随机策略与上界的中点（{threshold:.0f}）」记作成功，每个模型 {n_episodes} 个回合：\n")
    for name, rec in zip(names, success):
        k = int(rec[0].sum())
        lo, hi = wilson_interval(k, n_episodes)
        print(f"    {name}，种子 0 的模型：{k}/{n_episodes} 成功，Wilson 95% 区间 [{lo:.2f}, {hi:.2f}]；"
              f"各种子的成功率 {'  '.join(f'{v:.2f}' for v in rec.mean(axis=1))}")
    print(f"    {n_episodes} 个回合的 Wilson 区间有几十个百分点宽，而且它只描述一个模型。对「方法」下结论仍要回到种子层。")

    # ---- 两层指标放在一起 ----
    print("\n  两层指标放在一起（开环：可信视野的中位数是否不短于规划视野；闭环：归一化得分是否不低于 0.8）：\n")
    for name, st in zip(names, stats):
        normalized = (st["mean"] - random_policy.mean()) / span
        accurate, controls = np.median(horizons[name]) >= PLAN_HORIZON, normalized >= 0.8
        cell = {(True, True): "预测得准，控制得好：正常",
                (True, False): "预测得准，控制得差：问题在怎么用模型，或模型准在了规划器不去的地方",
                (False, True): "预测得差，控制得好：先排除泄漏；每步重新规划的 MPC 只依赖前几步的预测，对长时误差不敏感",
                (False, False): "预测得差，控制得差：先修模型"}[(bool(accurate), bool(controls))]
        print(f"    {name}：k* = {np.median(horizons[name]):.0f}，归一化得分 {normalized:.2f} → {cell}")
    gap = 1 - (stats[1]["mean"] - random_policy.mean()) / span
    if gap > 0:
        print(f"  上界这条线的用处：B 离「真实模型 + 同一个规划器」还差归一化得分 {gap:.2f}，这就是学到的模型造成的损失；"
              "\n  上界本身离理想值的距离，则是规划器（视野、预算）的损失，换再好的模型也补不回来。")
    else:
        print(f"  上界这条线的用处：这次运行里 B 的归一化得分没有低于「真实模型 + 同一个规划器」（差 {gap:+.2f}），"
              "\n  学到的模型没有造成看得见的损失；上界本身离理想值的距离是规划器（视野、预算）的损失，换再好的模型也补不回来。")

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, (name, x, st) in enumerate(zip(names, seed_scores, stats)):
        ax.scatter(np.full(len(x), i) + np.linspace(-0.08, 0.08, len(x)), x, color="gray", s=18, zorder=3)
        ax.errorbar(i, st["mean"], yerr=[[st["mean"] - st["lo"]], [st["hi"] - st["mean"]]], fmt="o", color=["C3", "C0"][i],
                    capsize=6, markersize=8, zorder=4)
    ax.axhline(oracle.mean(), color="green", linestyle="--", label="真实模型 + 同一个 CEM（上界）")
    ax.axhline(random_policy.mean(), color="black", linestyle=":", label="随机策略")
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel(f"平均回报（{episode_len} 步的回合）")
    ax.set_title(f"闭环结果：均值与 95% t 区间（{n_seeds} 个训练种子 × {n_episodes} 个评估回合；灰点是各个种子）")
    ax.legend(fontsize=8)
    save_figure(fig, OUT / "03-closed-loop.png")
    return records, oracle, random_policy


# ==================== 练习与思考 ====================


def exercises():
    """练习题与参考答案。"""
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)

    print("""
练习 1：性能剖面
    写一个 performance_profile(scores, taus)：对每个阈值 τ，返回"得分超过 τ 的运行所占的比例"。
    用第四部分的多任务合成数据画出来，再把训崩的那个种子换成正常水平重画一次。

练习 1 参考答案：
    def performance_profile(scores, taus):
        scores = np.asarray(scores).ravel()
        return np.array([(scores > tau).mean() for tau in taus])
    性能剖面是一条从 1 降到 0 的阶梯线。训崩的种子让曲线在很低的 τ 处就先掉一级（20 个里少了 1 个），
    这一级在 IQM 里完全看不见，在剖面上一眼就能看到。所以 IQM 要和剖面或逐种子的明细一起报告。

练习 2：把可信视野的阈值改成 0.2 和 0.8
    重新调用 credible_horizon()。两个模型的 k* 各变成多少？哪个阈值"对"？

练习 2 参考答案：
    阈值越严（c 越小），k* 越短。没有哪个阈值是"对"的，它是一个约定：
    c = 0.5 的意思是"模型的误差不到平凡基线的一半"。要紧的是在看到结果之前把它定下来写进计划，
    而不是事后挑一个让 k* 好看的值。报告里把整条误差比曲线画出来，读者可以自己换阈值。

练习 3：让评估种子和训练数据的种子重叠
    把 closed_loop_experiment() 里的评估种子从 1000 起改成从 100 起（和种子 0 的训练数据相同的初始状态）。
    闭环结果会变吗？为什么在这个例子里影响有限，换一个设定就可能很大？

练习 3 参考答案：
    这里训练数据是随机策略采的，模型只是见过"从这些初始状态出发、乱动一通"的转移，
    评估时规划器走的是完全不同的轨迹，所以重叠带来的好处很小。
    如果训练数据是专家演示（PushT 就是），评估的初始状态又出现在演示里，
    行为克隆几乎可以照着背下来，成功率会虚高。泄漏的严重程度取决于
    "评估时会经过的状态"有多少在训练数据里，所以规矩是一律不相交，不去赌它影响小。

思考题 1：开环误差是在随机策略的留出数据上量的。这有什么问题，怎么补？

思考题 1 参考答案：
    规划器会把系统带到随机策略到不了的状态，模型在那里的误差，随机数据上的曲线反映不出来。
    补的办法有两个：一是在规划器实际走出来的轨迹上再量一次开环误差（闭环评估时顺手把轨迹存下来）；
    二是看桥接指标，也就是第六部分做的事：在规划器选出的动作序列上比较想象回报和真实回报。

思考题 2：想象回报整体偏高 20%，但排序完全正确。这个模型能用来规划吗？能用来做别的什么不能做的事？

思考题 2 参考答案：
    能用来规划：规划器只需要知道哪条动作序列更好。
    不能直接用来估计"这个策略的真实回报是多少"，也不能把想象回报当成价值目标去训练评论家
    （Dreamer 那样的想象训练会把这 20% 学进价值函数）。用途不同，对模型的要求不同：
    做选择看秩相关，做估计看偏差和绝对误差。

思考题 3：两个方法各 5 个种子，差的区间包含 0。接下来应该怎么办？

思考题 3 参考答案：
    先用第七部分的两层方差分解看波动主要来自哪里：种子之间的差别是主要来源就加种子，
    回合噪声占大头就加评估回合（便宜得多）。按当前的标准差估一下要分辨这个差距需要多少个种子；
    如果算下来要几十个，说明这个差距在这个任务上小到不值得追，如实写"分不出来"。
    不要做的是：换指标、换检查点、去掉几个种子，直到区间不含 0。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    quick = is_quick()
    introduction()

    c_sim = verify_statistics()
    coverage_experiment(n_trials=500 if quick else 4000, n_resamples=200 if quick else 1000)
    aggregation_demos(c_sim)

    models, heldout = build_models(n_seeds=2 if quick else 5, quick=quick)
    _, horizons = open_loop_experiment(models, heldout, k_max=20 if quick else 50)
    bridge_experiment(models, quick)
    closed_loop_experiment(models, horizons, n_episodes=2 if quick else 10, episode_len=30 if quick else 200)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：04-ablations-and-writing.py（消融实验；把结果整理成图和表）")
    print("=" * 60)


if __name__ == "__main__":
    main()
