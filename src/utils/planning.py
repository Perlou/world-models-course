"""采样式规划器：CEM 与 MPPI。

这两个规划器是在 Phase 1 的讲义 `06-cem-and-mppi.py` 里从零写出来的（推导见 Phase 1 的
CONCEPT.md 第 7 节）。从 Phase 3 起它们被反复复用（PlaNet 的隐空间规划、TD-MPC2、
DINO-WM 式的特征空间规划），所以搬到这里。

和 Phase 1 讲义里的版本相比，只改了一处：规划器不再自己展开模型，而是接收一个打分函数

    score_fn(actions [N, H, A]) → scores [N]      分数越高越好

"用什么模型展开、在什么空间里展开、怎么算分"全部由调用方决定。
所以同一个 `cem_plan` 既能在真实状态空间里用，也能在隐空间、特征空间里用。

用法（滚动时域控制的一步）：

    mean = torch.zeros(horizon, action_dim, device=device)        # 回合开始时
    ...
    mean = cem_plan(score_fn, mean, action_low=-2.0, action_high=2.0)
    action = mean[0]                                              # 只执行第一个动作
    mean = shift_plan(mean)                                       # 热启动下一步
"""

import torch


def _sample_noise(n_candidates, mean, generator):
    """采样 [N, H, A] 的标准正态噪声，和 mean 在同一个设备上。"""
    return torch.randn(n_candidates, *mean.shape, generator=generator, device=mean.device)


@torch.no_grad()
def cem_plan(score_fn, mean, generator=None, action_low=-1.0, action_high=1.0,
             n_candidates=200, n_elites=20, n_iters=5,
             init_std=0.5, min_std=0.05, smoothing=0.1, return_std=False):
    """交叉熵方法（CEM）。

    每一轮：从高斯分布里采样 N 条动作序列 → 打分 → 取分数最高的 K 条（精英）→
    用精英的均值和标准差更新采样分布。

    参数：
        score_fn:    actions [N, H, A] → scores [N]，分数越高越好
        mean:        采样分布的初始均值 [H, A]（热启动时是上一步的结果平移一格）
        generator:   torch.Generator，要和 mean 在同一个设备上；不需要可复现时传 None
        action_low / action_high: 动作的上下界，标量或可广播到 [H, A] 的张量
        init_std:    初始标准差。应和动作范围同量级（动作范围是 [-2, 2] 时可取 1.0）
        min_std:     标准差下限，防止分布过早收缩成一个点
        smoothing:   均值更新的平滑系数，新均值 = smoothing × 旧均值 + (1 - smoothing) × 精英均值
        return_std:  为 True 时同时返回最后一轮的标准差
    返回：
        优化后的均值 [H, A]；它的第一个元素就是当前要执行的动作
    """
    std = torch.full_like(mean, init_std)
    for _ in range(n_iters):
        noise = _sample_noise(n_candidates, mean, generator)
        actions = (mean + std * noise).clamp(action_low, action_high)      # [N, H, A]
        scores = score_fn(actions)                                         # [N]
        elites = actions[scores.topk(n_elites).indices]                    # [K, H, A]
        mean = smoothing * mean + (1 - smoothing) * elites.mean(dim=0)
        std = elites.std(dim=0, correction=0).clamp_min(min_std)
    return (mean, std) if return_std else mean


@torch.no_grad()
def mppi_plan(score_fn, mean, generator=None, action_low=-1.0, action_high=1.0,
              n_candidates=200, n_iters=5, noise_std=0.5, temperature=1.0, ess_log=None):
    """MPPI（Model Predictive Path Integral）。

    和 CEM 的区别：不做"取前 K 名"的硬截断，而是让全部候选按 softmax(分数 / 温度) 加权平均；
    标准差是固定的超参数，只更新均值。

    参数：
        temperature: 温度。要和分数的量级匹配：太小退化成只取最好的一条，太大接近简单平均
        ess_log:     传入一个列表时，把每一轮的有效样本数 1 / Σ w² 记进去，用来判断温度是否合适
    返回：
        优化后的均值 [H, A]
    """
    for _ in range(n_iters):
        noise = _sample_noise(n_candidates, mean, generator)
        actions = (mean + noise_std * noise).clamp(action_low, action_high)         # [N, H, A]
        scores = score_fn(actions)                                                  # [N]
        weights = torch.softmax((scores - scores.max()) / temperature, dim=0)       # [N]，和为 1
        mean = (weights.view(-1, *[1] * mean.dim()) * actions).sum(dim=0)
        if ess_log is not None:
            ess_log.append(1.0 / (weights**2).sum().item())
    return mean


def shift_plan(mean, fill=0.0):
    """热启动：把规划结果向前平移一格，末尾补 fill。mean [H, A] → [H, A]。"""
    return torch.cat([mean[1:], torch.full_like(mean[:1], fill)])
