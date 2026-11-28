"""
进化策略与 CMA-ES：不用梯度的优化
================================

学习目标：
    1. 理解黑盒优化与 ask / tell 接口：不用梯度、只看排名
    2. 复核 CONCEPT.md 里的三个手算例子：重组权重、围绕旧均值算协方差、步长自适应
    3. 从零实现最简单的进化策略（SimpleES）和简化版 CMA-ES（SimplifiedCMAES）
    4. 在正圆碗、斜椭圆碗、弯曲峡谷三个二维测试函数上对比两者的评估效率
    5. 画出搜索分布椭圆的演化，观察协方差如何转向并拉长到与山谷对齐

核心概念：
    - 黑盒优化（black-box optimization）：只能给参数拿分数，拿不到梯度
    - ask / tell 接口：ask 采样一代候选，外部评估后 tell 交回分数并更新分布
    - rank-μ / rank-one 更新：用精英相对旧均值的步子和演化路径更新协方差矩阵 C
    - 累积步长自适应（CSA）：按白化后的演化路径长度缩放全局步长 σ

前置知识：
    - Phase 1 的 06-cem-and-mppi.py（CEM：采样、选精英、重新拟合分布）

对应概念文档：
    - CONCEPT.md 第 6 节「进化策略与 CMA-ES：不用梯度的优化」

预计运行时间：约 15 秒（CPU）
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from matplotlib.patches import Ellipse

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-02")
SEED = 0


# ==================== 第一部分：为什么不用梯度 ====================


def introduction():
    print("=" * 60)
    print("第一部分：黑盒优化与 ask / tell")
    print("=" * 60)
    print("""
V 和 M 用梯度下降训练，因为损失对参数可导。控制器 C 不行：它的目标是整回合的总分，
环境是个黑盒，分数要等回合结束才知道，没法求导。进化策略（ES）把它当成调参：

    搜索分布 N(m, σ²C)
          │
          ▼  ask()：采样 N 组候选参数 φ_1 … φ_N
    外部评估：每组参数跑若干回合，拿回分数 F_1 … F_N（只用排序，不怕尺度古怪）
          │
          ▼  tell(candidates, fitness)：按排名更新搜索分布
      均值 m        ← 精英的加权平均（中心往高分区搬）
      协方差 C      ← 围绕旧均值的精英步子 + 均值移动的演化路径 p_c
      步长 σ        ← 白化路径 p_σ 比随机游走长就放大，短就缩小

这一节不碰赛车，在二维测试函数上把 SimpleES 和 SimplifiedCMAES 从零写出来并画图看清。
    """)


# ==================== 第二部分：核对 CONCEPT.md 里的手算例子 ====================


def recombination_weights(pop_size):
    """Hansen (2016) 默认的按排名递减权重：取前一半为精英，返回 (weights [N_e], mu_eff)。"""
    n_elite = pop_size // 2
    raw = np.log((pop_size + 1) / 2) - np.log(np.arange(1, n_elite + 1))
    weights = raw / raw.sum()
    mu_eff = float(1.0 / np.sum(weights ** 2))
    return weights, mu_eff


def check_hand_calculations():
    print("=" * 60)
    print("第二部分：和 CONCEPT.md 的三个手算例子逐项核对")
    print("=" * 60)

    # 1. 重组权重与有效精英数（6.3 节）
    weights, mu_eff = recombination_weights(6)
    print(f"""
[1] 重组权重（N = 6，N_e = 3）：
    权重 w = ({weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f})，有效精英数 μ_eff = {mu_eff:.2f}
    """)
    assert np.allclose(weights, [0.637, 0.285, 0.078], atol=1e-3) and abs(mu_eff - 2.03) < 0.02

    # 2. 围绕旧均值 vs 围绕新均值（6.4 节，一维斜坡 + 二维外积）
    elites_1d = np.array([0.4, 1.0, 1.4])
    m_old, m_new = 0.0, float(elites_1d.mean())
    var_cem = float(((elites_1d - m_new) ** 2).mean())
    var_cma = float(((elites_1d - m_old) ** 2).mean())
    print(f"""[2] 一维斜坡上的方差估计（旧均值 m = 0，精英 0.4、1.0、1.4，新均值 {m_new:.3f}）：
    围绕新均值（CEM）：    {var_cem:.3f}（缩成一团，容易卡在半坡）
    围绕旧均值（CMA-ES）： {var_cma:.3f}（= {var_cem:.3f} + {m_new:.3f}² = {var_cem + m_new ** 2:.3f}）
    """)
    assert abs(var_cem - 0.169) < 1e-3 and abs(var_cma - 1.04) < 0.01

    steps_2d = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    c_mu = (steps_2d.T @ steps_2d) / len(steps_2d)
    eigvals_cma, eigvecs_cma = np.linalg.eigh(c_mu)
    centered = steps_2d - steps_2d.mean(axis=0)
    c_cem = (centered.T @ centered) / len(steps_2d)
    eigvals_cem, _ = np.linalg.eigh(c_cem)
    print(f"""    二维外积例子（精英步子 (1,1)、(2,1)、(1,2)）：
    CMA-ES 协方差特征值： {eigvals_cma[0]:.3f}（(1,-1) 方向），{eigvals_cma[1]:.3f}（(1,1) 前进方向，被拉长）
    CEM 协方差特征值：    {eigvals_cem[0]:.3f}（(1,1) 方向），{eigvals_cem[1]:.3f}（(1,-1) 方向，横在前进方向上）
    """)
    assert np.allclose(np.sort(eigvals_cma), [0.333, 3.667], atol=1e-3)
    assert np.allclose(np.sort(eigvals_cem), [0.111, 0.333], atol=1e-3)

    # 3. 步长自适应 CSA（6.5 节，一维）
    c_sigma, d_sigma, chi_1 = 0.5, 1.0, math.sqrt(2.0 / math.pi)
    coeff = math.sqrt(c_sigma * (2.0 - c_sigma))
    mults = {}
    for name, seq in [("情形 A（连走三步 +1）", [1, 1, 1]), ("情形 B（来回折返 +1,-1,+1）", [1, -1, 1])]:
        p = 0.0
        for s in seq:
            p = (1 - c_sigma) * p + coeff * s
        mult = math.exp((c_sigma / d_sigma) * (abs(p) / chi_1 - 1.0))
        mults[name] = (p, mult)
    print("[3] 步长路径检验（3 代结束时）：")
    for name, (p, mult) in mults.items():
        print(f"    {name}：p_σ = {p:+.3f}，步长乘以 {mult:.2f}")
    print()
    assert abs(list(mults.values())[0][1] - 1.57) < 0.02 and abs(list(mults.values())[1][1] - 0.91) < 0.02


# ==================== 第三部分：二维测试函数 ====================


def sphere(x):
    """正圆碗：F(x) = -(x_1² + x_2²)，最优解 (0, 0)，最大值为 0。"""
    x = np.asarray(x, dtype=np.float64)
    return -np.sum(x ** 2, axis=-1)


def tilted_ellipsoid(x):
    """斜放的细长椭圆碗（条件数 100）：绕 45° 旋转后两轴尺度差 10 倍，最大值 0 在 (0, 0)。"""
    x = np.asarray(x, dtype=np.float64)
    u = (x[..., 0] + x[..., 1]) / math.sqrt(2.0)
    v = (-x[..., 0] + x[..., 1]) / math.sqrt(2.0)
    return -(u ** 2 + 100.0 * v ** 2)


def rosenbrock(x):
    """Rosenbrock 弯曲峡谷：F(x) = -[(1 - x_1)² + 100(x_2 - x_1²)²]，最大值 0 在 (1, 1)。"""
    x = np.asarray(x, dtype=np.float64)
    return -((1.0 - x[..., 0]) ** 2 + 100.0 * (x[..., 1] - x[..., 0] ** 2) ** 2)


# ==================== 第四部分：从零实现 SimpleES 与 SimplifiedCMAES ====================


class SimpleES:
    """最简单的进化策略：固定步长、各向同性高斯 N(m, σ²I)，每代取精英平均更新均值。"""

    def __init__(self, mean, sigma, pop_size=12, elite_frac=0.5, seed=SEED):
        self.mean = np.array(mean, dtype=np.float64)
        self.sigma = float(sigma)
        self.pop_size = int(pop_size)
        self.n_elite = max(1, int(round(self.pop_size * elite_frac)))
        self.dim = len(self.mean)
        self.rng = np.random.default_rng(seed)

    def ask(self):
        """采样一代候选参数 [N, n]。"""
        eps = self.rng.standard_normal((self.pop_size, self.dim))
        return self.mean[None, :] + self.sigma * eps

    def tell(self, candidates, fitness):
        """只用排名：取适应度最高的 N_e 个候选求平均作为新均值。"""
        order = np.argsort(fitness)[::-1]
        self.mean = np.asarray(candidates, dtype=np.float64)[order[:self.n_elite]].mean(axis=0)


class SimplifiedCMAES:
    """简化版 CMA-ES（对应 CONCEPT.md 第 6.6 节伪代码）：均值 + 协方差 rank-μ/rank-one + CSA 步长。"""

    def __init__(self, mean, sigma, pop_size=None, seed=SEED):
        self.mean = np.array(mean, dtype=np.float64)
        self.dim = len(self.mean)
        self.sigma = float(sigma)
        self.pop_size = int(pop_size) if pop_size is not None else 4 + int(3 * math.log(self.dim))
        self.weights, self.mu_eff = recombination_weights(self.pop_size)
        self.n_elite = len(self.weights)

        n, mu_eff = self.dim, self.mu_eff
        self.c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        self.c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        self.c_mu = min(1.0 - self.c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        self.cov = np.eye(n, dtype=np.float64)
        self.path_c = np.zeros(n, dtype=np.float64)
        self.path_sigma = np.zeros(n, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self._update_eigensystem()

    def _update_eigensystem(self):
        self.cov = 0.5 * (self.cov + self.cov.T)
        eigvals, eigvecs = np.linalg.eigh(self.cov)
        eigvals = np.maximum(eigvals, 1e-20)
        self._B = eigvecs
        self._D = np.sqrt(eigvals)
        self._inv_sqrt_C = (eigvecs * (1.0 / self._D)) @ eigvecs.T

    def ask(self):
        """从 N(m, σ²C) 采样一代候选参数 [N, n]。"""
        eps = self.rng.standard_normal((self.pop_size, self.dim))
        steps = (eps * self._D) @ self._B.T
        return self.mean[None, :] + self.sigma * steps

    def tell(self, candidates, fitness):
        """按适应度排序后更新均值 m、演化路径、步长 σ 和协方差 C。"""
        candidates = np.asarray(candidates, dtype=np.float64)
        fitness = np.asarray(fitness, dtype=np.float64)
        order = np.argsort(fitness)[::-1]
        steps = (candidates - self.mean[None, :]) / max(self.sigma, 1e-20)
        elite_steps = steps[order[:self.n_elite]]
        d_w = self.weights @ elite_steps

        # 1. 均值
        self.mean = self.mean + self.sigma * d_w

        # 2. 步长路径与步长（使用更新前的 C^{-1/2} 做白化）
        c_s = self.c_sigma
        self.path_sigma = (1.0 - c_s) * self.path_sigma + math.sqrt(c_s * (2.0 - c_s) * self.mu_eff) * (
            self._inv_sqrt_C @ d_w
        )
        self.sigma *= math.exp((c_s / self.d_sigma) * (np.linalg.norm(self.path_sigma) / self.chi_n - 1.0))

        # 3. 协方差路径与 rank-one / rank-μ 更新
        c_c = self.c_c
        self.path_c = (1.0 - c_c) * self.path_c + math.sqrt(c_c * (2.0 - c_c) * self.mu_eff) * d_w
        c_mu_mat = (elite_steps.T * self.weights) @ elite_steps
        self.cov = (
            (1.0 - self.c_1 - self.c_mu) * self.cov
            + self.c_1 * np.outer(self.path_c, self.path_c)
            + self.c_mu * c_mu_mat
        )
        self._update_eigensystem()


# ==================== 第五部分：在二维测试函数上对比与画图 ====================


def run_optimizer(opt, fn, max_evals, target_fitness=-1e-3):
    """运行 ask / tell 循环，记录每一代均值、协方差、最佳适应度，以及首次达标时的评估次数。"""
    history, snapshots = [], []
    evals, hit_evals = 0, None
    best_mean_f = float(fn(opt.mean))
    gen = 0
    while evals < max_evals:
        candidates = opt.ask()
        fitness = fn(candidates)
        evals += len(candidates)
        opt.tell(candidates, fitness)
        mean_f = float(fn(opt.mean))
        best_mean_f = max(best_mean_f, mean_f)
        cov = opt.cov if hasattr(opt, "cov") else np.eye(opt.dim)
        history.append((evals, mean_f))
        snapshots.append((gen, opt.mean.copy(), cov.copy(), opt.sigma))
        if hit_evals is None and mean_f >= target_fitness:
            hit_evals = evals
        gen += 1
    return history, snapshots, hit_evals, best_mean_f


def plot_search_distribution(mean, cov, sigma, ax, color="tab:red", lw=1.4, label=None):
    """在等高线图上画出搜索分布 N(m, σ²C) 的 2 倍标准差椭圆和中心点。"""
    eigvals, eigvecs = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = np.maximum(eigvals[order], 1e-20), eigvecs[:, order]
    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))
    width = 4.0 * sigma * math.sqrt(eigvals[0])    # 2 倍标准差对应的直径
    height = 4.0 * sigma * math.sqrt(eigvals[1])
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor="none", lw=lw, label=label)
    ax.add_patch(ellipse)
    ax.plot(mean[0], mean[1], marker="o", ms=3.5, color=color)


def compare_on_benchmarks(max_evals):
    print("=" * 60)
    print("第三部分：在三个二维测试函数上对比 SimpleES 与 CMA-ES")
    print("=" * 60)

    tasks = [
        ("正圆碗 sphere", sphere, np.array([3.0, -2.5]), 1.0, (0.0, 0.0), (-4.0, 4.0, -4.0, 4.0)),
        ("斜椭圆碗 tilted_ellipsoid", tilted_ellipsoid, np.array([3.0, -2.5]), 1.0, (0.0, 0.0), (-4.0, 4.0, -4.0, 4.0)),
        ("弯曲峡谷 rosenbrock", rosenbrock, np.array([-1.5, 1.5]), 0.8, (1.0, 1.0), (-2.2, 2.2, -1.0, 3.0)),
    ]
    pop_size = 12
    target = -1e-3
    print(f"""
统一预算：每次跑 {max_evals} 次函数评估（种群 N = {pop_size}），达到目标适应度 F(m) ≥ {target} 算"达到同一精度"。
SimpleES 的步长 σ 固定，不学协方差；CMA-ES 同时更新均值、协方差 C 和步长 σ。
    """)
    print(f"    {'测试函数':<26}  {'SimpleES 达标评估次数':<20}  {'CMA-ES 达标评估次数':<20}  {'最终 F(m) (ES / CMA-ES)'}")

    all_results = []
    for title, fn, init_mean, init_sigma, optimum, bounds in tasks:
        es = SimpleES(init_mean, init_sigma, pop_size=pop_size, seed=SEED)
        cma = SimplifiedCMAES(init_mean, init_sigma, pop_size=pop_size, seed=SEED)
        h_es, s_es, hit_es, best_es = run_optimizer(es, fn, max_evals, target)
        h_cma, s_cma, hit_cma, best_cma = run_optimizer(cma, fn, max_evals, target)
        hit_es_str = str(hit_es) if hit_es is not None else f"未达标({best_es:.2e})"
        hit_cma_str = str(hit_cma) if hit_cma is not None else f"未达标({best_cma:.2e})"
        print(f"    {title:<26}  {hit_es_str:<20}  {hit_cma_str:<20}  {h_es[-1][1]:.2e} / {h_cma[-1][1]:.2e}")
        all_results.append((title, fn, optimum, bounds, h_es, s_es, h_cma, s_cma))

    # 图 1：三个函数上的收敛曲线（用误差 -F(m) 的对数尺度）
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (title, _, _, _, h_es, _, h_cma, _) in zip(axes, all_results):
        ax.semilogy([e for e, _ in h_es], [max(-f, 1e-12) for _, f in h_es], label="SimpleES（固定 σ）")
        ax.semilogy([e for e, _ in h_cma], [max(-f, 1e-12) for _, f in h_cma], label="CMA-ES（自适应 C, σ）")
        ax.axhline(-target, color="gray", ls="--", lw=1, label=f"精度线 {-target:.0e}")
        ax.set_xlabel("函数评估次数")
        ax.set_ylabel("离最优值的距离 -F(m)")
        ax.set_title(title)
        ax.legend(fontsize=8)
    save_figure(fig, OUT / "06-es-vs-cmaes-curves.png")

    # 图 2：搜索分布的椭圆在等高线上的演化（斜椭圆与 Rosenbrock）
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, idx in zip(axes, [1, 2]):
        title, fn, optimum, (xmin, xmax, ymin, ymax), _, s_es, _, s_cma = all_results[idx]
        gx, gy = np.meshgrid(np.linspace(xmin, xmax, 160), np.linspace(ymin, ymax, 160))
        grid = np.stack([gx, gy], axis=-1)
        zz = np.log10(np.maximum(-fn(grid), 1e-6))
        ax.contour(gx, gy, zz, levels=16, cmap="Greys", alpha=0.65)
        ax.plot(optimum[0], optimum[1], marker="*", ms=11, color="gold", mec="black", zorder=5, label="最优解")

        # 画出 SimpleES 的均值轨迹，和 CMA-ES 在若干代时的搜索椭圆
        es_path = np.array([snap[1] for snap in s_es])
        cma_path = np.array([snap[1] for snap in s_cma])
        ax.plot(es_path[:, 0], es_path[:, 1], color="tab:blue", ls="--", lw=1.2, alpha=0.8, label="SimpleES 轨迹")
        ax.plot(cma_path[:, 0], cma_path[:, 1], color="tab:red", lw=1.5, label="CMA-ES 轨迹")

        pick_idx = np.unique(np.linspace(0, min(len(s_cma) - 1, 35), 6, dtype=int))
        colors = plt.cm.autumn(np.linspace(0.85, 0.15, len(pick_idx)))
        for k, (i_snap, col) in enumerate(zip(pick_idx, colors)):
            gen, m, cov, sig = s_cma[i_snap]
            plot_search_distribution(m, cov, sig, ax, color=col, lw=1.5,
                                     label=f"CMA-ES 第 {gen + 1} 代" if k in (0, len(pick_idx) - 1) else None)
        ax.set_xlim(xmin, xmax), ax.set_ylim(ymin, ymax)
        ax.set_xlabel("φ₁"), ax.set_ylabel("φ₂")
        ax.set_title(f"{title}：采样椭圆转向并拉长到与山谷一致")
        ax.legend(fontsize=8, loc="upper right")
    save_figure(fig, OUT / "06-cmaes-ellipse-evolution.png")

    # 读取实测比例并打印结论
    _, _, _, _, _, _, hit_sphere, _ = all_results[0]
    ell_es = [f for _, f in all_results[1][4]][-1]
    ell_cma = [f for _, f in all_results[1][6]][-1]
    print(f"""
怎么读：
    1. 正圆碗（各个方向对称）：SimpleES 方向走得对，但 σ = 1.0 固定不变，走到碗底附近就以 σ 的幅度晃荡，停在 10⁻² 量级；
       CMA-ES 靠自适应步长不断缩小 σ，迅速达标并一直压到机器精度。
    2. 斜椭圆碗（两轴尺度差 10 倍，且斜放）：SimpleES 的固定圆形分布要么在窄方向来回跳过，要么在长方向走不动；
       CMA-ES 的采样椭圆很快转向 45° 并拉长到与等高线一致，之后沿着窄谷直奔最优解（最终 F(m)：SimpleES {ell_es:.2e} vs CMA-ES {ell_cma:.2e}）。
    3. 弯曲峡谷（Rosenbrock）：不仅斜，山谷还拐弯。CMA-ES 跟着弯道边走边旋转协方差，顺着谷底爬向 (1, 1)。
    """)


# ==================== 第六部分：维度上限与实际开销 ====================


def scaling_check():
    print("=" * 60)
    print("第四部分：维度的代价——为什么控制器不能太大")
    print("=" * 60)
    print("""
CMA-ES 每代要存 [n, n] 的协方差矩阵、做一次 O(n³) 的特征分解：
    n = 99  （控制器只用 z_t）        C 有       9,801 个数，特征分解约 0.1 毫秒
    n = 867 （控制器用 [z_t ; h_t]）  C 有     751,689 个数，特征分解约几十毫秒
    n = 100 万（直接换小 CNN）        C 有  10^12 个数，内存直接放不下

所以 CarRacing 的控制器故意做成线性（867 个参数）：CMA-ES 刚好能发挥威力，
同时把复杂的视觉和记忆全部留给用梯度下降训练的 V 和 M。
    """)


# ==================== 第七部分：练习 ====================


def exercises():
    print("=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把 SimplifiedCMAES 的 rank-one 项（c_1）设为 0，只留 rank-μ 更新，在斜椭圆和弯曲峡谷上再跑一遍。
        哪个函数的达标评估次数变化更大？为什么？

练习 2：把 SimplifiedCMAES 里的步长自适应去掉（固定 σ），看看它在正圆碗上还能不能精准收敛到 1e-3 以内。

练习 3：把 SimpleES 的步长改成按指数衰减（比如每代乘以 0.95），它在正圆碗和弯曲峡谷上各自表现如何？

练习 4：手算。N = 10 时，按 Hansen 默认公式 N_e = 5，第一名和第五名的归一化重组权重分别是多少？

思考题 1：CMA-ES 只看适应度的排名，不看具体分数值。这在 CarRacing 上有什么好处，又会带来什么问题？

思考题 2：在真实世界里评估一次 867 维的控制器要几十秒。如果每代用默认种群 4 + ⌊3 ln 867⌋ = 24，
          和用论文里的 64，各有什么利弊？

参考答案：
    练习 1：弯曲峡谷变化更大，斜椭圆次之。种群只有 12（6 个精英），单代提供的几何信息有限；
            rank-one 靠演化路径跨代累积连续前进的方向，在小种群下是拉长协方差的主要力量。
    练习 2：不能。σ 固定时均值会在最优解周围以 σ 为半径晃荡，精度卡在 σ² 的量级。
    练习 3：正圆碗上因为最优解在附近，衰减步长能收敛；弯曲峡谷上如果步长提前衰减完，均值会半路停在谷壁上（过早收敛）。
    练习 4：raw_i = ln(5.5) - ln(i)，1 到 5 依次为 1.705、1.012、0.606、0.318、0.095，和为 3.736。
            归一化后第一名约 0.456，第五名约 0.025（差 18 倍）。
    思考题 1：好处是不怕奖励尺度变化（-100 到 +900 和 0 到 10 的行为完全一样），也不怕个别极端离群回合。
              代价是丢掉了"好多少"的信息：两个候选差 1 分和差 500 分，在排序里是一回事；当回合分数的随机波动
              大于候选之间的真实差距时，排名会被噪声洗掉。
    思考题 2：24 更省每一代的回合数，同样时间能走更多代；但每代只有 12 个精英，在有噪声的适应度下排名抖动大，
              协方差估得不稳。论文用 64（并配合多核并行）换取更稳的排名和协方差估计。
    """)


def main():
    seed_everything(SEED)
    max_evals = 360 if is_quick() else 1800

    introduction()
    check_hand_calculations()
    compare_on_benchmarks(max_evals)
    scaling_check()
    exercises()

    print("=" * 60)
    print("本节小结")
    print("=" * 60)
    print("""
    1. 进化策略通过 ask / tell 接口做黑盒优化：只看排名，不需要梯度。
    2. CMA-ES 的三个关键设计：精英按排名加权、协方差围绕旧均值算（加上演化路径）、步长由白化路径长度控制。
    3. 正圆碗上简单 ES 就能走动；一旦参数之间尺度不同或有耦合（斜椭圆、弯曲峡谷），CMA-ES 的协方差自适应会带来数量级的优势。
    4. 存储 O(n²) 和分解 O(n³) 决定了 CMA-ES 适合几百到几千个参数的小模型，这正是控制器只有 867 个参数的原因。

下一节：把今天写的 CMA-ES 搬到真实 CarRacing-v3 里，训练那个 867 个参数的线性控制器，并对比用不用 M 的隐藏状态 h。
    """)
    if is_quick():
        print("（当前是冒烟模式 WM_QUICK=1：评估预算已缩小，完整结果请去掉 WM_QUICK 运行。）")


if __name__ == "__main__":
    main()
