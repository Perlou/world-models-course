"""
确定性路径还是随机路径
======================

学习目标：
    1. 实现三种状态转移：纯确定性（RNN）、纯随机（SSM）、两者结合（RSSM 的简化版）
    2. 在"记忆任务"上检验：信息能不能原样传很多步
    3. 在"分叉任务"上检验：能不能把两种不同的未来分开表示，而不是平均成一个哪种都不像的结果
    4. 用实验结果理解 RSSM 为什么同时需要 h_t 和 z_t

核心概念：
    - 确定性路径（deterministic path）：h_t = f(h_{t-1}, ·)，不采样，像寄存器
    - 随机路径（stochastic path）：z_t ~ p(z_t | ·)，每一步都从分布里采样
    - 单点预测 + 均方误差的最优解是条件均值：多峰的未来会被平均掉
    - 每一步注入噪声，长程记忆就要穿过一串有噪声的信道

前置知识：
    - 02-sequential-vae-elbo.py（序列 VAE、序列 ELBO）
    - 旧课的 GRU

对应概念文档：
    - CONCEPT.md 第 5 节「确定性路径还是随机路径」

预计运行时间：约 12 分钟（CPU）
"""

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-03")
SEED = 0
DEVICE = torch.device("cpu")  # 玩具模型，CPU 足够

N_CUES = 4  # 记忆任务：提示有 4 种
STRIP = 16  # 分叉任务：像素条的宽度
O_DIM = STRIP  # 两个任务共用 16 维的观测（记忆任务只用前 4 维）
E_DIM, H_DIM, Z_DIM, HIDDEN = 32, 64, 8, 64
OBS_STD = 0.1  # 观测噪声；解码器的方差也固定为它
MIN_STD = 0.1
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
CELLS = torch.linspace(-1.2, 1.2, STRIP)
FORK_AT, FORK_T = 4, 12  # 分叉任务：前 4 步在中间，之后向左或向右


# ==================== 第一部分：两个需求 ====================


def introduction():
    print("=" * 60)
    print("第一部分：状态传递的两个需求")
    print("=" * 60)
    print("""
世界模型的状态从一步传到下一步，要同时满足两件事：
    记得住            第 1 帧看到的东西，第 50 步还得知道
    容得下多种可能    只看了前几帧时，接下来可能向左也可能向右；模型要能分开表示，不能平均

三种设计（CONCEPT.md 第 5.2 节）：

    (a) 纯确定性 RNN      h_1 ══▶ h_2 ══▶ h_3            ══▶ 函数计算，不采样
    (b) 纯随机 SSM        z_1 ┄┄▶ z_2 ┄┄▶ z_3            ┄┄▶ 每一步都从分布里采样
    (c) 两者结合 RSSM     h_1 ══▶ h_2 ══▶ h_3
                           ┊ ╱     ┊ ╱     ┊             z_t 由 h_t 采样，再输入下一步的 h
                          z_1     z_2     z_3

两个玩具任务分别检验这两个需求：

    记忆任务：  [提示] [空白] [空白] … [空白] [提示]      第 1 步给出 4 种提示之一，L 步空白之后复现
    分叉任务：  ● 在中间 4 步 → 然后随机向左或向右移动    从分叉前的画面看不出会往哪边

三个模型的编码器、解码器结构相同，状态的大小也相近；
(a) 用均方误差（固定方差的高斯 NLL）训练，(b)、(c) 用序列 ELBO 训练。
    """)


# ==================== 第二部分：两个玩具任务 ====================


CUE_POSITIONS = torch.tensor([-0.9, -0.3, 0.3, 0.9])


def bump(x):
    return torch.exp(-0.5 * ((x[..., None] - CELLS) / 0.15) ** 2)


def memory_task(n, delay, gen):
    """记忆任务。返回 obs [N, delay+2, 16]、cues [N]。

    第 1 步：小球出现在 4 个固定位置之一（提示）；中间 delay 步：全空白；最后一步：小球再次出现在同一个位置。
    """
    cues = torch.randint(0, N_CUES, (n,), generator=gen)
    clean = torch.zeros(n, delay + 2, O_DIM)
    cue_bump = bump(CUE_POSITIONS[cues])
    clean[:, 0] = cue_bump
    clean[:, -1] = cue_bump
    return clean + OBS_STD * torch.randn(clean.shape, generator=gen), cues


def fork_positions(direction):
    """分叉任务里小球的位置。direction [N] 取 ±1 → [N, FORK_T]"""
    t = torch.arange(1, FORK_T + 1, dtype=torch.float32)
    offset = (0.3 * (t - FORK_AT)).clamp(0.0, 0.9)  # 分叉之前是 0，之后每步向外 0.3，最远 0.9
    return direction[:, None] * offset


def fork_task(n, gen):
    """分叉任务。返回 obs [N, 12, 16]、方向 [N]（−1 左，+1 右）。"""
    direction = torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1
    clean = bump(fork_positions(direction))
    return clean + OBS_STD * torch.randn(clean.shape, generator=gen), direction


# ==================== 第三部分：三种模型 ====================


def mlp(inp, out):
    return nn.Sequential(nn.Linear(inp, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, out))


def obs_nll(obs, pred):
    """固定方差高斯的负对数似然，对 16 维求和。[B, O] → [B]"""
    return (0.5 * ((obs - pred) / OBS_STD) ** 2 + math.log(OBS_STD) + HALF_LOG_2PI).sum(-1)


def split_std(out):
    mean, raw = out.chunk(2, dim=-1)
    return mean, F.softplus(raw) + MIN_STD


class DeterministicModel(nn.Module):
    """(a) 纯确定性：h_t = GRU(h_{t-1}, enc(o_{t-1}))，用 h_t 预测 o_t。

    没有先验 / 后验之分，观测只能当输入喂进去。没有观测时（开环），只能把自己的预测喂回来。
    """

    def __init__(self):
        super().__init__()
        self.enc = mlp(O_DIM, E_DIM)
        self.gru = nn.GRUCell(E_DIM, H_DIM)
        self.dec = mlp(H_DIM, O_DIM)

    def initial(self, batch):
        return torch.zeros(batch, H_DIM), torch.zeros(batch, O_DIM)  # (h_0, o_0 = 0)

    def loss(self, obs):
        h, prev = self.initial(obs.shape[0])
        total = 0.0
        for t in range(obs.shape[1]):
            h = self.gru(self.enc(prev), h)
            total = total + obs_nll(obs[:, t], self.dec(h))
            prev = obs[:, t]
        return (total / obs.shape[1]).mean()

    def filter(self, obs):
        """读入整段观测，返回读完之后的状态。"""
        h, prev = self.initial(obs.shape[0])
        for t in range(obs.shape[1]):
            h = self.gru(self.enc(prev), h)
            prev = obs[:, t]
        return h, prev

    def rollout(self, state, n_steps):
        """开环预测 n_steps 步。[B, ·] → [B, n_steps, O]"""
        h, prev = state
        preds = []
        for _ in range(n_steps):
            h = self.gru(self.enc(prev), h)
            prev = self.dec(h)  # 没有观测：把自己的预测当输入
            preds.append(prev)
        return torch.stack(preds, dim=1)


class StochasticModel(nn.Module):
    """(b) 纯随机：先验 p(z_t | z_{t-1})，后验 q(z_t | z_{t-1}, o_t)，解码器 p(o_t | z_t)。"""

    def __init__(self):
        super().__init__()
        self.enc = mlp(O_DIM, E_DIM)
        self.prior_net = mlp(Z_DIM, 2 * Z_DIM)
        self.post_net = mlp(Z_DIM + E_DIM, 2 * Z_DIM)
        self.dec = mlp(Z_DIM, O_DIM)

    def initial(self, batch):
        return torch.zeros(batch, Z_DIM)

    def observe_step(self, z, o_t):
        prior = split_std(self.prior_net(z))
        post = split_std(self.post_net(torch.cat([z, self.enc(o_t)], dim=-1)))
        z = post[0] + post[1] * torch.randn_like(post[0])
        return z, prior, post, self.dec(z)

    def imagine_step(self, z):
        mean, std = split_std(self.prior_net(z))
        z = mean + std * torch.randn_like(mean)
        return z, self.dec(z)

    def loss(self, obs):
        z = self.initial(obs.shape[0])
        total = 0.0
        for t in range(obs.shape[1]):
            z, prior, post, pred = self.observe_step(z, obs[:, t])
            kl = kl_divergence(Normal(*post), Normal(*prior)).sum(-1)
            total = total + obs_nll(obs[:, t], pred) + kl  # −ELBO 的第 t 项
        return (total / obs.shape[1]).mean()

    def filter(self, obs):
        z = self.initial(obs.shape[0])
        for t in range(obs.shape[1]):
            z = self.observe_step(z, obs[:, t])[0]
        return z

    def rollout(self, z, n_steps):
        preds = []
        for _ in range(n_steps):
            z, pred = self.imagine_step(z)
            preds.append(pred)
        return torch.stack(preds, dim=1)


class SimpleRSSM(nn.Module):
    """(c) 两者结合：h_t = GRU(h_{t-1}, z_{t-1})，先验 p(z_t | h_t)，后验 q(z_t | h_t, o_t)，解码器 p(o_t | h_t, z_t)。

    这是讲义 04 完整 RSSM 的简化版：没有动作、没有奖励头，编码器和解码器是 MLP。
    """

    def __init__(self):
        super().__init__()
        self.enc = mlp(O_DIM, E_DIM)
        self.inp = nn.Linear(Z_DIM, E_DIM)
        self.gru = nn.GRUCell(E_DIM, H_DIM)
        self.prior_net = mlp(H_DIM, 2 * Z_DIM)
        self.post_net = mlp(H_DIM + E_DIM, 2 * Z_DIM)
        self.dec = mlp(H_DIM + Z_DIM, O_DIM)

    def initial(self, batch):
        return torch.zeros(batch, H_DIM), torch.zeros(batch, Z_DIM)

    def observe_step(self, state, o_t):
        h, z = state
        h = self.gru(F.elu(self.inp(z)), h)  # 确定性路径
        prior = split_std(self.prior_net(h))
        post = split_std(self.post_net(torch.cat([h, self.enc(o_t)], dim=-1)))
        z = post[0] + post[1] * torch.randn_like(post[0])  # 随机路径：从后验采样
        return (h, z), prior, post, self.dec(torch.cat([h, z], dim=-1))

    def imagine_step(self, state):
        h, z = state
        h = self.gru(F.elu(self.inp(z)), h)
        mean, std = split_std(self.prior_net(h))
        z = mean + std * torch.randn_like(mean)  # 随机路径：从先验采样
        return (h, z), self.dec(torch.cat([h, z], dim=-1))

    def loss(self, obs):
        state = self.initial(obs.shape[0])
        total = 0.0
        for t in range(obs.shape[1]):
            state, prior, post, pred = self.observe_step(state, obs[:, t])
            kl = kl_divergence(Normal(*post), Normal(*prior)).sum(-1)
            total = total + obs_nll(obs[:, t], pred) + kl
        return (total / obs.shape[1]).mean()

    def filter(self, obs):
        state = self.initial(obs.shape[0])
        for t in range(obs.shape[1]):
            state = self.observe_step(state, obs[:, t])[0]
        return state

    def rollout(self, state, n_steps):
        preds = []
        for _ in range(n_steps):
            state, pred = self.imagine_step(state)
            preds.append(pred)
        return torch.stack(preds, dim=1)


MODELS = {"(a) 纯确定性": DeterministicModel, "(b) 纯随机": StochasticModel, "(c) 两者结合": SimpleRSSM}


def repeat_state(state, n):
    """把状态在批维度上重复 n 次，用来从同一个起点采样多条未来。"""
    if isinstance(state, tuple):
        return tuple(s.repeat_interleave(n, dim=0) for s in state)
    return state.repeat_interleave(n, dim=0)


def train(model_cls, make_batch, n_iters, batch_size=64, lr=2e-3):
    torch.manual_seed(SEED)
    model = model_cls()
    if hasattr(model, "gru"):
        # GRU 更新门偏置初始化为正值，让循环状态默认倾向保留历史（标准长程记忆初始化技巧）
        model.gru.bias_hh.data[H_DIM : 2 * H_DIM].fill_(1.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(SEED)
    for _ in range(n_iters):
        loss = model.loss(make_batch(batch_size, gen))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        optimizer.step()
    return model, loss.item()


# ==================== 第四部分：记忆任务 ====================


@torch.no_grad()
def memory_accuracy(model, delay, n_test):
    """读入除最后一步之外的全部观测，预测最后一步，看最接近哪一个提示位置。"""
    obs, cues = memory_task(n_test, delay, torch.Generator().manual_seed(SEED + 1))
    torch.manual_seed(SEED)
    state = model.filter(obs[:, :-1])
    pred = model.rollout(state, 1)[:, 0]  # [N, O]
    templates = bump(CUE_POSITIONS)        # [4, O]
    ans = torch.cdist(pred, templates).argmin(-1)
    return (ans == cues).float().mean().item()


def compare_memory(delays, n_iters, n_test):
    print("\n" + "=" * 60)
    print("第二部分：记忆任务")
    print("=" * 60)
    print(f"""
序列：第 1 步小球出现在 4 个位置之一（提示），中间 L 步全空白，最后一步在同一位置复现。
观测都带标准差 {OBS_STD} 的噪声。每个延迟 L 各训练三个模型（{n_iters} 步），在 {n_test} 条新序列上测：
读入前 L + 1 步，预测最后一步，看预测画面最接近哪个提示位置。瞎猜的准确率是 {1 / N_CUES:.0%}。
随机模型（b、c）的预测是从先验里采样一次，不取均值。
    """)
    acc = {name: [] for name in MODELS}
    print(f"  {'延迟 L':<8}" + "".join(f"{name:>14}" for name in MODELS) + f"{'用时':>8}")
    for delay in delays:
        start = time.time()
        for name, cls in MODELS.items():
            model, _ = train(cls, lambda b, g: memory_task(b, delay, g)[0], n_iters)
            acc[name].append(memory_accuracy(model, delay, n_test))
        print(f"  {delay:<10}" + "".join(f"{acc[name][-1]:>15.1%}" for name in MODELS) + f"{time.time() - start:>8.0f}s")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in MODELS:
        ax.plot(delays, acc[name], "o-", linewidth=2, label=name)
    ax.axhline(1 / N_CUES, color="gray", linestyle=":", label="瞎猜")
    ax.set_xlabel("延迟 L（提示和复现之间隔了多少步）")
    ax.set_ylabel("复现提示的准确率")
    ax.set_ylim(0, 1.05)
    ax.set_title("记忆任务：信息能不能原样传很多步")
    ax.legend()
    save_figure(fig, OUT / "03-memory-task.png")
    return acc


# ==================== 第五部分：分叉任务 ====================


@torch.no_grad()
def fork_metrics(model, n_samples, look=3):
    """读入分叉前的 4 步，从同一个起点采样 n_samples 条未来，看分叉后第 look 步的画面。"""
    obs, _ = fork_task(1, torch.Generator().manual_seed(SEED + 2))
    torch.manual_seed(SEED)
    state = repeat_state(model.filter(obs[:, :FORK_AT]), n_samples)
    preds = model.rollout(state, FORK_T - FORK_AT)  # [S, 8, O]
    frame = preds[:, look - 1]  # 分叉后第 look 步，[S, O]
    pos = fork_positions(torch.tensor([-1.0, 1.0]))[:, FORK_AT + look - 1]
    left, right = bump(pos[0]), bump(pos[1])
    average = 0.5 * (left + right)
    to_branch = torch.minimum((frame - left).pow(2).mean(-1), (frame - right).pow(2).mean(-1))
    to_average = (frame - average).pow(2).mean(-1)
    centre = (frame.clamp_min(0) * CELLS).sum(-1) / frame.clamp_min(1e-6).sum(-1)
    return dict(frames=frame.numpy(), left=left.numpy(), right=right.numpy(),
                peak=frame.max(-1).values.mean().item(), left_frac=(centre < 0).float().mean().item(),
                to_branch=to_branch.mean().item(), to_average=to_average.mean().item(), preds=preds.numpy())


def compare_fork(n_iters, n_samples):
    print("\n" + "=" * 60)
    print("第三部分：分叉任务")
    print("=" * 60)
    print(f"""
序列共 {FORK_T} 步：前 {FORK_AT} 步小球停在像素条中间，之后各一半的概率向左或向右，每步移 0.3，最远到 ±0.9。
分叉前的 {FORK_AT} 帧对两种情况完全一样（只差噪声），模型没有任何办法知道会往哪边。

每个模型训练 {n_iters} 步。测试时读入分叉前的 {FORK_AT} 帧，从同一个状态出发开环预测 {n_samples} 次，
看分叉后第 3 步的画面（此时小球在 ±0.9）：
    峰值            画面最亮处的平均亮度。一个清楚的小球是 1；左右各半个小球的"平均画面"约 0.5
    向左的比例      理想是 50%
    到最近分支的误差 / 到平均画面的误差：哪个小，预测就更像哪个
    """)
    results = {}
    print(f"  {'模型':<12}{'峰值':>8}{'向左的比例':>12}{'到最近分支':>12}{'到平均画面':>12}")
    for name, cls in MODELS.items():
        model, _ = train(cls, lambda b, g: fork_task(b, g)[0], n_iters)
        r = fork_metrics(model, n_samples)
        results[name] = r
        print(f"  {name:<12}{r['peak']:>8.2f}{r['left_frac']:>13.0%}{r['to_branch']:>13.4f}{r['to_average']:>13.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    cells = CELLS.numpy()
    for ax, (name, r) in zip(axes, results.items()):
        for f in r["frames"][:12]:
            ax.plot(cells, f, color="C0", alpha=0.35)
        ax.plot(cells, r["left"], "k--", linewidth=1, label="真实可能 1：向左")
        ax.plot(cells, r["right"], "k:", linewidth=1, label="真实可能 2：向右")
        ax.set_title(f"{name}：12 次采样的预测")
        ax.set_xlabel("像素条上的位置")
    axes[0].set_ylabel("亮度")
    axes[0].legend(fontsize=8)
    fig.suptitle("分叉后第 3 步的开环预测（蓝线是预测，黑线是两种真实的可能）")
    save_figure(fig, OUT / "03-fork-task.png")
    return results


def summarize(acc, delays, fork):
    det, sto, rssm = (acc[name] for name in MODELS)
    fdet, fsto, frssm = (fork[name] for name in MODELS)
    mid = min(2, len(delays) - 1)
    print("\n" + "=" * 60)
    print("小结：两个需求，三种设计")
    print("=" * 60)
    print(f"""
  {'':<14}{'记忆（L=' + str(delays[0]) + '）':>12}{'记忆（L=' + str(delays[mid]) + '）':>14}{'记忆（L=' + str(delays[-1]) + '）':>14}{'分叉任务峰值':>14}
  {'(a) 纯确定性':<14}{det[0]:>12.0%}{det[mid]:>14.0%}{det[-1]:>14.0%}{fdet['peak']:>14.2f}
  {'(b) 纯随机':<15}{sto[0]:>12.0%}{sto[mid]:>14.0%}{sto[-1]:>14.0%}{fsto['peak']:>14.2f}
  {'(c) 两者结合':<14}{rssm[0]:>12.0%}{rssm[mid]:>14.0%}{rssm[-1]:>14.0%}{frssm['peak']:>14.2f}
    """)
    lines = []
    if fdet["to_average"] < fdet["to_branch"]:
        lines.append(f"纯确定性模型在分叉任务上预测出的是\"平均画面\"：峰值 {fdet['peak']:.2f}，离平均画面的误差 "
                     f"{fdet['to_average']:.4f} 远小于离任一真实分支的 {fdet['to_branch']:.4f}（CONCEPT.md 第 5.3 节）。")
    else:
        lines.append(f"纯确定性模型在分叉任务上的峰值是 {fdet['peak']:.2f}。")
    for name, r in [("纯随机", fsto), ("两者结合", frssm)]:
        sharp = r["to_branch"] < r["to_average"]
        lines.append(f"{name}模型的采样{'各自清晰地落在一个分支上' if sharp else '没有清楚地分成两个分支'}"
                     f"（峰值 {r['peak']:.2f}，向左 {r['left_frac']:.0%}）。")
    lines.append(f"记忆任务上，纯随机模型在短延迟 L = {delays[0]} 时能达到 {sto[0]:.0%}，"
                 f"到 L = {delays[mid]} 时就掉到 {sto[mid]:.0%}（接近 25% 瞎猜）：每一步采样都在冲刷历史信息（第 5.4 节）。")
    if rssm[mid] > 0.8 and rssm[-1] < 0.5:
        lines.append(f"两者结合（RSSM）在 L = {delays[mid]} 时仍保持 {rssm[mid]:.0%}（靠确定性的 h 保存记忆），"
                     f"但到更长的 L = {delays[-1]} 时也降到 {rssm[-1]:.0%}：\n"
                     f"    原因在于序列 ELBO 训练时，最后一步的后验可以直接看真值 o_{{L+2}}，提前记住 4 选 1 提示最多只省 ln 4 ≈ 1.39 nat 的 KL，\n"
                     f"    摊到整条序列上激励很弱。这也印证了 CONCEPT.md 第 14 节 Q12：RSSM 擅长几十步之内的动力学，超长程记忆以后要靠 Transformer（Phase 8）。")
    else:
        lines.append(f"两者结合（RSSM）在 L = {delays[mid]} 时准确率 {rssm[mid]:.0%}，L = {delays[-1]} 时 {rssm[-1]:.0%}，"
                     f"分叉峰值 {frssm['peak']:.2f}：h 负责保存，z 负责表达每一步的不确定性。")
    for i, line in enumerate(lines, 1):
        print(f"  {i}. {line}")


# ==================== 第六部分：练习与思考 ====================


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    rows = [(k, 0.1 * math.sqrt(k), phi(-1 / (0.1 * math.sqrt(k)))) for k in (10, 50, 200)]
    print(f"""
练习 1：复算 CONCEPT.md 第 5.4 节的表
    z_t = z_{{t-1}} + σ ε_t，σ = 0.1，从 z_0 = +1 出发，k 步之后靠正负号读这个比特，读错的概率是多少？

练习 1 参考答案：
    累积噪声的标准差是 σ√k，读错的概率是 Φ(−1 / (σ√k))：
""" + "\n".join(f"        k = {k:<4} 标准差 {s:.2f}   读错的概率 {p:.2%}" for k, s, p in rows) + f"""

练习 2：给纯确定性模型的分叉预测"补救"
    把 DeterministicModel 的输出从一个均值换成两个高斯的混合（Phase 2 的 MDN），分叉任务的预测会怎样？

练习 2 参考答案：
    每一步的输出分布可以有两个峰，单步采样能得到清楚的左或右。
    但采样只发生在输出端，状态 h 本身不知道"刚才采到了左"，下一步可能又采到右，
    开环预测的轨迹会左右乱跳。把随机性放进状态（z_t 再喂回 h_{{t+1}}），采到的分支才会一直保持。

练习 3：把 MIN_STD 改成 0.01 重跑记忆任务
    纯随机模型的长程记忆会变好还是变差？KL 的数值会怎样？

练习 3 参考答案：
    每步注入的噪声下限变小，记忆更容易保持；但先验可以变得非常窄，
    后验一旦偏离预测，KL 就是几十上百 nat（CONCEPT.md 第 4.5 节例四），训练更容易出现尖峰。
    同一个 σ 既要小（记得住）又要大（容得下意外），这是纯随机模型的两难。

思考题 1：摆是确定性的物理系统，像素版的摆还需要随机路径吗？

思考题 1 参考答案：
    需要。不确定性来自"看不全"：只看到第一帧时速度未知，模型需要表达"可能往左也可能往右"。
    而且在 RSSM 里，观测的信息只能经由后验的 z_t 进入状态，去掉 z 等于换了一种模型。

思考题 2：(c) 里 h_t 是确定性的，为什么两次想象得到的 h_t 不一样？

思考题 2 参考答案：
    "确定性"说的是从输入到 h_t 这一步不注入新的噪声。h_t 的输入里有采样出来的 z_{{t-1}}，
    所以只要前面某一步的 z 不同，之后的 h 就不同。分叉任务里，三个模型中只有 (c) 的 h 会跟着 z 走上不同的分支。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    torch.set_num_threads(4)
    quick = is_quick()
    if quick:
        print("（冒烟模式：训练步数缩得很小，下面的数字没有参考价值）\n")
    delays, mem_iters, fork_iters, n_test, n_samples = (
        ([1, 5], 40, 40, 100, 50) if quick else ([1, 5, 8, 15], 1500, 1500, 1000, 500))

    introduction()
    acc = compare_memory(delays, mem_iters, n_test)
    fork = compare_fork(fork_iters, n_samples)
    summarize(acc, delays, fork)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：04-rssm-from-scratch.py（从零实现完整的 RSSM）")
    print("=" * 60)


if __name__ == "__main__":
    main()
