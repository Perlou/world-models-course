"""
三种梯度估计器：score-function、pathwise 与 straight-through
===========================================================

学习目标：
    1. 理解同一个数学问题 ∇_φ E_{x ~ p_φ}[f(x)] 的三条求解路径及各自的适用前提
    2. 在一维高斯 f(x) = x² 上验证 CONCEPT.md 第 7.3 节的精确方差公式：
       μ=1, σ=1 时无基线 SF、带基线 SF、PW 的样本均值都接近 2，样本方差分别精确落在 30、18、4 附近
    3. 扫描标准差 σ → 0：亲眼看到 PW 方差趋于 0、无基线 SF 方差按 μ⁴/σ² 爆炸、带基线 SF 卡在常数底 8μ²
    4. 在带高频波纹的 f(x) = x² + 0.1 sin(100x) 上复现排名反转：PW 因导数剧烈震荡反而输给带基线 SF
    5. 在离散伯努利变量上对比 SF 与 straight-through（ST）：量化 ST 的偏差与比 SF 低几百倍的方差

核心概念：
    - score-function 估计器（SF / REINFORCE）：E[f(x) ∇ log p_φ(x)]，把 f 当黑盒，无偏但方差大
    - pathwise 估计器（PW / 重参数化）：x = g_φ(ε)，E[f'(x) ∇_φ g_φ(ε)]，利用 f 的导数，光滑时方差极小
    - straight-through 估计器（ST）：前向用离散 one-hot 采样，反向按连续概率 probs 传梯度，有偏但低方差

前置知识：
    - 01-policy-gradient-reinforce.py、02-baselines-and-advantage.py
    - Phase 2 / 3 的重参数化采样

对应概念文档：
    - CONCEPT.md 第 7 节「三种梯度估计器」

预计运行时间：约 5 秒（CPU）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-05")
SEED = 0


# ==================== 第一部分：导读 ====================


def introduction():
    print("=" * 60)
    print("第一部分：采样阻断了计算图——三条绕行路线")
    print("=" * 60)
    print("""
世界模型与强化学习里到处都是同一个问题：求 ∇_φ E_{x ~ p_φ}[f(x)]。

    ① score-function     不穿过 f：f(x) · ∇_φ log p_φ(x)       离散/连续皆可，f 可为黑盒；无偏，方差大
    ② pathwise           重参数化 x = μ + σ ε：f'(x) · ∂x/∂φ   要求 x 连续、f 可微；无偏，光滑时方差极小
    ③ straight-through   x = one_hot + probs - probs.detach()  专门对付离散隐变量/动作；有偏，方差极小

Phase 6 的 Dreamer 会把三者全用上：连续动作 actor 用 ②，离散动作 actor 用 ①，32×32 类别隐变量用 ③！
    """)


# ==================== 第二部分：三个核心函数 ====================


def score_function_grad(mu, sigma, fn, n_samples, baseline=0.0, generator=None):
    """用 score-function 估计 ∂/∂μ E_{x ~ N(μ, σ²)}[fn(x)]，返回 [n_samples] 个单样本梯度估计。"""
    eps = torch.randn(n_samples, generator=generator)
    x = mu + sigma * eps
    score_mu = (x - mu) / (sigma**2)  # ∂ log N(x; μ, σ²) / ∂μ
    return (fn(x) - baseline) * score_mu


def pathwise_grad(mu, sigma, fn, n_samples, generator=None):
    """用重参数化（pathwise）估计 ∂/∂μ E_{x ~ N(μ, σ²)}[fn(x)]，返回 [n_samples] 个单样本梯度估计。"""
    eps = torch.randn(n_samples, generator=generator)
    mu_t = torch.full((n_samples,), float(mu), requires_grad=True)
    x = mu_t + sigma * eps
    y = fn(x).sum()
    (grads,) = torch.autograd.grad(y, mu_t)
    return grads.detach()


def straight_through_sample(logits, generator=None):
    """类别分布的 straight-through 采样（CONCEPT.md 第 7.2、7.5 节）：前向是精确 one-hot，反向梯度等于 ∂probs/∂logits。"""
    probs = logits.softmax(dim=-1)
    idx = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    one_hot = F.one_hot(idx, num_classes=probs.shape[-1]).float()
    return one_hot + probs - probs.detach()


# ==================== 第三部分：高斯分布方差定律与波纹反转 ====================


def experiment_gaussian_variance(n_samples):
    print("\n" + "=" * 60)
    print("第二部分：实验一 & 二——高斯分布上的精确方差对比与高频波纹反转")
    print("=" * 60)

    gen = torch.Generator().manual_seed(SEED)
    mu, sigma = 1.0, 1.0
    fn_quad = lambda x: x**2
    true_grad = 2.0 * mu
    baseline = mu**2 + sigma**2  # E[x²] = μ² + σ²

    g_sf = score_function_grad(mu, sigma, fn_quad, n_samples, baseline=0.0, generator=gen)
    g_sfb = score_function_grad(mu, sigma, fn_quad, n_samples, baseline=baseline, generator=gen)
    g_pw = pathwise_grad(mu, sigma, fn_quad, n_samples, generator=gen)

    print(f"  1. 光滑二次函数 f(x) = x² 在 μ={mu}, σ={sigma}（真实梯度 2μ = {true_grad:.3f}，样本数 = {n_samples:,}）：")
    print(f"  {'估计器':<22}{'样本均值':>12}{'样本方差':>14}{'理论精确方差':>16}")
    for name, g, th_var in [
        ("SF（无基线）", g_sf, mu**4 / sigma**2 + 14 * mu**2 + 15 * sigma**2),
        ("SF + 精确基线 E[f]", g_sfb, 8 * mu**2 + 10 * sigma**2),
        ("Pathwise（重参数化）", g_pw, 4 * sigma**2),
    ]:
        print(f"  {name:<22}{g.mean().item():>12.4f}{g.var(correction=1).item():>14.3f}{th_var:>16.3f}")

    # 扫描 σ 从 0.05 到 2.0
    sigmas = np.geomspace(0.05, 2.0, 20)
    var_sf, var_sfb, var_pw = [], [], []
    for s in sigmas:
        b = mu**2 + s**2
        var_sf.append(score_function_grad(mu, float(s), fn_quad, n_samples // 2, 0.0, gen).var().item())
        var_sfb.append(score_function_grad(mu, float(s), fn_quad, n_samples // 2, float(b), gen).var().item())
        var_pw.append(pathwise_grad(mu, float(s), fn_quad, n_samples // 2, gen).var().item())

    # 实验二：带高频波纹 f(x) = x² + 0.1 sin(100 x)（CONCEPT.md 第 7.4 节）
    fn_wiggly = lambda x: x**2 + 0.1 * torch.sin(100.0 * x)
    w_sfb = score_function_grad(mu, sigma, fn_wiggly, n_samples, baseline=baseline, generator=gen)
    w_pw = pathwise_grad(mu, sigma, fn_wiggly, n_samples, generator=gen)
    print(f"""
  2. 高频波纹函数 f(x) = x² + 0.1 sin(100x)（μ=1, σ=1）：
     SF + 基线：样本均值 = {w_sfb.mean().item():.3f}，样本方差 = {w_sfb.var().item():.2f}（函数值几乎没变，方差依然 ~18）
     Pathwise ：样本均值 = {w_pw.mean().item():.3f}，样本方差 = {w_pw.var().item():.2f}（导数多出 10 cos(100x)，方差反超 SF！）
     ──▶ 印证第 7.4 节：沿混沌或长视野模型反向传播时，雅可比连乘会让 pathwise 导数剧烈震荡。
    """)

    return sigmas, var_sf, var_sfb, var_pw, (g_sf, g_sfb, g_pw), (w_sfb, w_pw)


# ==================== 第四部分：伯努利离散变量的偏差与方差 ====================


def experiment_bernoulli_st(n_samples):
    """CONCEPT.md 第 7.5 节：伯努利 x ~ Bern(p)，f(x) = x² + 10，真实梯度恒为 f(1) - f(0) = 1。"""
    print("=" * 60)
    print("第三部分：实验三——离散伯努利变量上 Straight-Through 的偏差与低方差")
    print("=" * 60)

    gen = torch.Generator().manual_seed(SEED + 1)
    fn_bern = lambda x: x**2 + 10.0
    ps = [0.2, 0.5, 0.8]

    print(f"  设定：x ~ Bernoulli(p)，f(x) = x² + 10，真实梯度 ∂E[f]/∂p = f(1) − f(0) = 1.000\n")
    print(f"  {'p':<6}{'ST 均值':>10}{'ST 偏差':>10}{'ST 方差':>10}{'SF(无基线)均值':>16}{'SF(无基线)方差':>16}")
    rows = []
    for p_val in ps:
        u = torch.rand(n_samples, generator=gen)
        x_hard = (u < p_val).float()
        # ST：前向 x_hard，反向 ∂x/∂p = 1 → 单样本梯度就是 f'(x_hard) = 2 * x_hard
        p_param = torch.full((n_samples,), p_val, requires_grad=True)
        x_st = x_hard + p_param - p_param.detach()
        (st_grads,) = torch.autograd.grad(fn_bern(x_st).sum(), p_param)
        st_grads = st_grads.detach()

        # SF：x=1 时得分 1/p，x=0 时得分 -1/(1-p)
        score_p = torch.where(x_hard > 0.5, 1.0 / p_val, -1.0 / (1.0 - p_val))
        sf_grads = fn_bern(x_hard) * score_p

        st_m, st_v = st_grads.mean().item(), st_grads.var().item()
        sf_m, sf_v = sf_grads.mean().item(), sf_grads.var().item()
        rows.append((p_val, st_m, st_m - 1.0, st_v, sf_m, sf_v))
        print(f"  {p_val:<6.1f}{st_m:>10.3f}{st_m - 1.0:>+10.3f}{st_v:>10.3f}{sf_m:>16.3f}{sf_v:>16.1f}")

    # 验证 straight_through_sample 在 K 类 one-hot 上的前向与反向性质
    logits = torch.tensor([[0.0, 1.0, -0.5]], requires_grad=True)
    st_oh = straight_through_sample(logits.expand(4, -1), generator=gen)
    assert torch.all((st_oh == 0.0) | (st_oh == 1.0)) and torch.all(st_oh.sum(dim=-1) == 1.0)
    (st_oh * torch.tensor([1.0, 3.0, -2.0])).sum().backward()
    assert logits.grad is not None and logits.grad.abs().max().item() > 0

    return rows


def plot_all(sigmas, var_sf, var_sfb, var_pw, bern_rows):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].loglog(sigmas, var_sf, "o-", markersize=4, linewidth=2, label="SF（无基线）：~ μ⁴/σ² 爆炸")
    axes[0].loglog(sigmas, var_sfb, "s-", markersize=4, linewidth=2, label="SF + 基线：卡在常数底 8μ²=8")
    axes[0].loglog(sigmas, var_pw, "^-", markersize=4, linewidth=2, label="Pathwise：4σ² → 0")
    axes[0].set_xlabel("采样标准差 σ（对数坐标）")
    axes[0].set_ylabel("单样本梯度方差（对数坐标）")
    axes[0].set_title("f(x) = x²（μ=1）：σ 越小时重参数化优势越悬殊")
    axes[0].legend(fontsize=8.5)

    p_grid = np.linspace(0.05, 0.95, 50)
    st_bias_curve = 2.0 * p_grid - 1.0  # E[2x] - 1 = 2p - 1
    st_var_curve = 4.0 * p_grid * (1.0 - p_grid)
    sf_var_curve = 121.0 / p_grid + 100.0 / (1.0 - p_grid) - 1.0
    ax2 = axes[1]
    ax2.plot(p_grid, np.abs(st_bias_curve), linewidth=2, color="C2", label="ST 的绝对偏差 |2p − 1|")
    ax2.plot(p_grid, np.sqrt(st_var_curve), linewidth=2, color="C0", label="ST 的标准差 √Var ≤ 1")
    ax2.plot(p_grid, np.sqrt(sf_var_curve) / 20.0, "--", linewidth=1.8, color="C3", label="SF 的标准差 ÷ 20（原值高达 21~50）")
    ax2.set_xlabel("伯努利概率 p")
    ax2.set_ylabel("梯度误差量级")
    ax2.set_title("伯努利 f(x)=x²+10：ST 用小偏差换回几十倍的降噪")
    ax2.legend(fontsize=8)
    save_figure(fig, OUT / "05-gradient-estimators.png")


# ==================== 第五部分：练习与思考 ====================


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把 PyTorch 里的 dist.rsample() 错写成 dist.sample()，程序会报错吗？梯度变成多少？

练习 1 参考答案：
    dist.sample() 返回的张量 requires_grad=False。如果下游还有其他带梯度的参数（比如 critic），
    backward() 甚至不会报错，只是策略参数 φ 的梯度静默变成全 0，actor 完全原地踏步！

思考题 1：为什么在真实机器人或 Gymnasium 环境里做无模型 RL（如 PPO），不能直接对环境用 pathwise 估计器？

思考题 1 参考答案：
    真实物理世界没有 PyTorch 计算图，拿不到 ∂s_{t+1}/∂a_t；但只要我们学出了一个神经网络世界模型（RSSM）或 Q 网络（SAC），
    模型和 Q 网络都是可微的 PyTorch 模块，pathwise 和 straight-through 就立刻派上用场了！
    """)


def main():
    seed_everything(SEED)
    n_samples = 5000 if is_quick() else 200_000
    introduction()
    sigmas, var_sf, var_sfb, var_pw, _, _ = experiment_gaussian_variance(n_samples)
    bern_rows = experiment_bernoulli_st(n_samples)
    plot_all(sigmas, var_sf, var_sfb, var_pw, bern_rows)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：06-ddpg-td3.py（用 critic 的动作导数做连续控制）")
    print("=" * 60)


if __name__ == "__main__":
    main()
