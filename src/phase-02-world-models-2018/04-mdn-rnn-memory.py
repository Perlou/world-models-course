"""
M：MDN-RNN 记忆模型
===================

学习目标：
    1. 把上一节的 MDN 输出层接到 LSTM 上，得到 M：读入 (z_t, a_t)，预测 z_{t+1} 的分布
    2. 逐项核对形状和参数量；看清下标：h_t 是读入 z_t 之前的状态
    3. 用 teacher forcing 训练 M，损失是序列上的 MDN 负对数似然
    4. 学会给 M 的损失找参照：平凡基线在上，噪声下限在下，M 落在中间的什么位置
    5. 用两个实验检验 M 学到了什么：打乱动作（它用没用动作）、线性探针（h 里有没有车速）

核心概念：
    - MDN-RNN：LSTM 负责记忆，MDN 头负责输出"下一个 z 的分布"
    - teacher forcing：训练时每一步的输入都是真实的 z，不是模型自己的预测
    - 基线（baseline）与下限：一个损失数值只有和参照物比才有意义
    - 线性探针（linear probe）：在冻结的特征上训练一个线性模型，看能读出什么

前置知识：
    - 02-conv-vae-vision.py（数据集已经编码成 μ 和 log σ²）
    - 03-mixture-density-network.py（MDN 的损失）

对应概念文档：
    - CONCEPT.md 第 4 节「M：MDN-RNN 记忆模型」

预计运行时间：约 1 分钟（MPS）
"""

import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn

from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-02")
SEED = 0
Z_DIM, ACTION_DIM, HIDDEN_DIM, N_COMPONENTS = 32, 3, 256, 5
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -5.0, 2.0
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
DEVICE = get_device()
SUFFIX = "-quick" if is_quick() else ""
LATENTS_PATH = OUT / f"latents{SUFFIX}.npz"
MDNRNN_PATH = OUT / f"mdnrnn{SUFFIX}.pt"


# ==================== 第一部分：M 要做什么 ====================


def introduction():
    print("=" * 60)
    print("第一部分：M 要做什么")
    print("=" * 60)
    print("""
V 把每一帧压成了 32 个数。一个回合现在是一串 z，中间夹着动作。M 在这串数上做两件事：
    记：把看过的东西汇总成一个向量 h。单帧里没有的车速，可以从 z 的前后变化里读出来
    预测：给定当前的 z_t 和动作 a_t，给出下一个 z_{t+1} 的分布

            z_0, a_0            z_1, a_1            z_2, a_2
               │                   │                   │
               ▼                   ▼                   ▼
   h_0 ──▶ ┌────────┐ ── h_1 ──▶ ┌────────┐ ── h_2 ──▶ ┌────────┐ ── h_3 ──▶ ···
   (零)    │  LSTM  │            │  LSTM  │            │  LSTM  │
           └────────┘            └────────┘            └────────┘
               │ h_1                │ h_2                │ h_3
               ▼                   ▼                   ▼
            MDN 头               MDN 头               MDN 头
               │                   │                   │
               ▼                   ▼                   ▼
            p(z_1)              p(z_2)              p(z_3)
          和真实的 z_1 比       和真实的 z_2 比       和真实的 z_3 比

这就是一个语言模型：token 换成了连续的 z，softmax 换成了 MDN，每一步多了一个动作当条件。
下标只需要记一条：h_t 是读入 z_t 之前的状态，它只见过 z_0 … z_{t-1} 和 a_0 … a_{t-1}。
    """)


# ==================== 第二部分：数据 ====================


def ensure_artifact(path, lesson):
    """跨讲义共享的产物不存在时，先运行生成它的那一节讲义（继承当前的 WM_QUICK 设置）。"""
    if path.exists():
        return
    print(f"⚠️  没找到 {path.name}，先运行 {lesson} 生成它……")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), lesson)
    subprocess.run([sys.executable, script], check=True, stdout=subprocess.DEVNULL)


def load_latents():
    """读取 02 节编码好的数据集，划分训练 / 验证回合。

    返回字典（张量都在 DEVICE 上）：
        mu, logvar  [E, T, 32]   V 对每一帧的编码
        actions     [E, T, 3]
        speeds      [E, T]       仿真器内部的真实车速，只用于探针实验
        valid       [E, T-1]     布尔。valid[e, t] 为真表示 (z_t, a_t) → z_{t+1} 是一条真实的转移
        train, val  回合编号（最后 10% 的回合当验证集，和 02 节的划分一致）
    """
    ensure_artifact(LATENTS_PATH, "02-conv-vae-vision.py")
    with np.load(LATENTS_PATH) as f:
        raw = {k: f[k] for k in f.files}
    n_episodes, horizon = raw["mu"].shape[:2]
    lengths = torch.from_numpy(raw["lengths"])
    # 回合长度为 n 时，帧是 o_0 … o_{n-1}，转移是 t = 0 … n-2
    valid = torch.arange(horizon - 1)[None, :] < (lengths[:, None] - 1)
    n_val = max(1, n_episodes // 10)
    data = {k: torch.from_numpy(raw[k]).to(DEVICE) for k in ("mu", "logvar", "actions", "speeds")}
    data.update(valid=valid.to(DEVICE), lengths=lengths,
                train=torch.arange(n_episodes - n_val), val=torch.arange(n_episodes - n_val, n_episodes))
    return data


def sample_z(data, episodes, generator):
    """z = μ + σ·ε。每次调用重新采样，训练时相当于数据增强。返回 [B, T, 32]。"""
    mu, logvar = data["mu"][episodes], data["logvar"][episodes]
    eps = torch.randn(mu.shape, generator=generator).to(DEVICE)       # 在 CPU 上采样，结果可复现
    return mu + torch.exp(0.5 * logvar) * eps


def describe_data(data):
    print("=" * 60)
    print("第二部分：数据——M 只看 z，不看画面")
    print("=" * 60)
    mu, logvar, valid = data["mu"], data["logvar"], data["valid"]
    n_train = int(valid[data["train"]].sum())
    n_val = int(valid[data["val"]].sum())
    kl = 0.5 * (mu**2 + logvar.exp() - 1 - logvar)                    # [E, T, 32]
    frame_ok = torch.arange(mu.shape[1], device=DEVICE)[None, :] < data["lengths"].to(DEVICE)[:, None]
    kl_per_dim = kl[frame_ok].mean(dim=0)                             # [32]
    active = kl_per_dim > 0.1
    if int(active.sum()) == 0:
        # V 训练得太少（冒烟模式）时没有哪一维的 KL 超过 0.1，退而取 KL 最大的 4 维，保证后面的统计能算
        print("\n⚠️  V 没有 KL 大于 0.1 nat 的维度（V 训练太少），下面改用 KL 最大的 4 维当作\"在用\"。")
        active = torch.zeros_like(active)
        active[kl_per_dim.topk(4).indices] = True
    sigma = torch.exp(0.5 * logvar)[frame_ok]                         # [N, 32]
    step_change = (mu[:, 1:] - mu[:, :-1])[valid].std(dim=0)          # [32]
    print(f"""
    mu、logvar  {list(mu.shape)}      V 对每一帧的编码（回合 × 时间 × 32）
    actions     {list(data['actions'].shape)}
    训练 {len(data['train'])} 个回合（{n_train} 条转移），验证 {len(data['val'])} 个回合（{n_val} 条转移）

32 维里只有 {int(active.sum())} 维"在用"（KL 大于 0.1 nat，02 节看过）。两类维度对 M 来说完全不同：

                     V 的编码噪声 σ      相邻两帧 μ 的变化（标准差）
    在用的 {int(active.sum()):>2} 维          {sigma[:, active].mean().item():.2f}                 {step_change[active].mean().item():.3f}
    不在用的 {int((~active).sum()):>2} 维        {sigma[:, ~active].mean().item():.2f}                 {step_change[~active].mean().item():.3f}

不在用的维度，z 就是从 N(0, 1) 里采的纯噪声，没有任何东西可预测。
在用的维度，一步之内 μ 只变 {step_change[active].mean().item():.2f} 左右，而 V 自己的编码噪声有 {sigma[:, active].mean().item():.2f}：
M 的预测目标 z_{{t+1}} 里，噪声比信号的变化大得多。读后面的损失数值时要记住这一点。
    """)
    return active


# ==================== 第三部分：模型 ====================


class MDNRNN(nn.Module):
    """LSTM + MDN 头。z_{t+1} 的每一维有自己的一套 K 个成分的一维高斯混合。"""

    def __init__(self, z_dim=Z_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, n_components=N_COMPONENTS):
        super().__init__()
        self.z_dim, self.n_components = z_dim, n_components
        self.lstm = nn.LSTM(z_dim + action_dim, hidden_dim, batch_first=True)
        self.mdn_head = nn.Linear(hidden_dim, z_dim * n_components * 3)

    def mdn_params(self, hidden):
        """LSTM 的输出 [..., 256] → log_pi、mu、log_sigma，形状都是 [..., 32, 5]。"""
        out = self.mdn_head(hidden).view(*hidden.shape[:-1], self.z_dim, self.n_components, 3)
        logits, mu, log_sigma = out.unbind(dim=-1)
        return torch.log_softmax(logits, dim=-1), mu, log_sigma

    def encode_history(self, z, a, state=None):
        """整段序列过 LSTM。z [B, T, 32]，a [B, T, 3] → 输出 [B, T, 256]，其中第 t 个是 h_{t+1}。"""
        return self.lstm(torch.cat([z, a], dim=-1), state)

    def forward(self, z, a, state=None):
        """训练用：整段序列的前向计算。返回 MDN 参数（第 t 个位置预测 z_{t+1}）和最后的 LSTM 状态。"""
        hidden, state = self.encode_history(z, a, state)
        return self.mdn_params(hidden), state

    def step(self, z, a, state=None):
        """做梦和控制时用：只走一步。z [B, 32]，a [B, 3] → MDN 参数 [B, 32, 5] 和新的 LSTM 状态。"""
        (log_pi, mu, log_sigma), state = self.forward(z.unsqueeze(1), a.unsqueeze(1), state)
        return (log_pi.squeeze(1), mu.squeeze(1), log_sigma.squeeze(1)), state


def mdn_rnn_nll(log_pi, mu, log_sigma, target):
    """逐维的负对数似然。参数 [..., 32, 5]，target [..., 32] → [..., 32]。

    和上一节的 mdn_nll 是同一个式子，只是前面多了批量、时间、维度三个轴。
    """
    log_sigma = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    y = target.unsqueeze(-1)                                          # [..., 32, 1]，和成分维广播
    log_normal = -HALF_LOG_2PI - log_sigma - 0.5 * ((y - mu) / log_sigma.exp()) ** 2
    return -torch.logsumexp(log_pi + log_normal, dim=-1)


def mdn_rnn_loss(log_pi, mu, log_sigma, target, valid):
    """训练用的标量损失：每步每维的平均负对数似然，只算真实存在的转移。"""
    nll = mdn_rnn_nll(log_pi, mu, log_sigma, target)                  # [B, T, 32]
    return nll[valid].mean()


def check_model():
    print("=" * 60)
    print("第三部分：模型——形状、参数量、损失")
    print("=" * 60)
    model = MDNRNN()
    z, a = torch.zeros(4, 10, Z_DIM), torch.zeros(4, 10, ACTION_DIM)
    (log_pi, mu, log_sigma), (h, c) = model(z, a)
    n_lstm = sum(p.numel() for p in model.lstm.parameters())
    n_head = sum(p.numel() for p in model.mdn_head.parameters())
    hand_lstm = 4 * HIDDEN_DIM * (Z_DIM + ACTION_DIM + HIDDEN_DIM) + 4 * HIDDEN_DIM
    print(f"""
批大小 4、序列长 10 时各张量的形状：
    LSTM 输入 concat(z, a)      {list(torch.cat([z, a], -1).shape)}
    LSTM 输出（每一步的 h）      {list(model.encode_history(z, a)[0].shape)}
    log_pi、mu、log_sigma       {list(log_pi.shape)}     每一步、每一维、每个成分
    最后的状态 (h, c)           两个 {list(h.shape)}      第一维是 LSTM 的层数

参数量：
    LSTM   手算 4 × 256 × (35 + 256) + 4 × 256 = {hand_lstm:,}；PyTorch 数出来 {n_lstm:,}
           （nn.LSTM 每组门有两个偏置向量，多出 4 × 256 = {n_lstm - hand_lstm:,}）
    MDN 头 256 × 480 + 480 = {n_head:,}
    合计   {n_lstm + n_head:,}（论文参数量表里是 422,368，差的就是那 {n_lstm - hand_lstm:,} 个偏置）
    """)
    assert n_lstm - hand_lstm == 4 * HIDDEN_DIM and hand_lstm + n_head == 422_368

    # 损失的手算核对：只留一个成分时，它就是高斯负对数似然
    log_pi1 = torch.zeros(1, 1, 1)                                    # K = 1：log π = 0
    nll = mdn_rnn_nll(log_pi1, torch.tensor([[[0.3]]]), torch.tensor([[[math.log(0.7)]]]), torch.tensor([[1.0]]))
    gauss = HALF_LOG_2PI + math.log(0.7) + (1.0 - 0.3) ** 2 / (2 * 0.7**2)
    print(f"损失核对：一维、一个成分、μ = 0.3、σ = 0.7、目标 1.0 → {nll.item():.4f}，高斯公式 {gauss:.4f}。")
    assert abs(nll.item() - gauss) < 1e-5

    # step 和 forward 应该给出同样的结果
    torch.manual_seed(SEED)
    z, a = torch.randn(2, 6, Z_DIM), torch.randn(2, 6, ACTION_DIM)
    (_, mu_seq, _), _ = model(z, a)
    state, mu_steps = None, []
    for t in range(6):
        (_, mu_t, _), state = model.step(z[:, t], a[:, t], state)
        mu_steps.append(mu_t)
    gap = (torch.stack(mu_steps, dim=1) - mu_seq).abs().max().item()
    print(f"一步一步调用 step() 和一次性调用 forward() 的输出最大相差 {gap:.1e}：两条路径等价。\n")
    assert gap < 1e-5


# ==================== 第四部分：训练 ====================


@torch.no_grad()
def evaluate_nll(model, data, episodes, n_repeats=4, shuffle_actions=False, seed=1234):
    """逐维的负对数似然 [32]：在 episodes 上对真实转移取平均，z 的采样重复 n_repeats 次取平均。

    shuffle_actions=True 时把每个回合的动作换成另一个回合的，用来检验模型用没用动作。
    """
    generator = torch.Generator().manual_seed(seed)
    idx = episodes.to(DEVICE)
    valid = data["valid"][idx]
    actions = data["actions"][idx]
    if shuffle_actions:
        actions = actions.roll(1, dims=0)
    total = torch.zeros(Z_DIM, device=DEVICE)
    for _ in range(n_repeats):
        z = sample_z(data, idx, generator)                            # [B, T, 32]
        params, _ = model(z[:, :-1], actions[:, :-1])
        total += mdn_rnn_nll(*params, z[:, 1:])[valid].mean(dim=0)
    return (total / n_repeats).cpu()


def train_mdn_rnn(data, epochs, batch_size=32, lr=1e-3):
    """teacher forcing 训练。每个批次是若干个完整回合，LSTM 从零状态开始读。"""
    torch.manual_seed(SEED)
    model = MDNRNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(SEED)
    train = data["train"]
    history, best_val, best_state = [], float("inf"), None
    start = time.time()
    for epoch in range(epochs):
        perm = train[torch.randperm(len(train), generator=generator)]
        losses = []
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size].to(DEVICE)
            z = sample_z(data, idx, generator)                        # [B, T, 32]，每个批次重新采样
            params, _ = model(z[:, :-1], data["actions"][idx][:, :-1])
            loss = mdn_rnn_loss(*params, z[:, 1:], data["valid"][idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)         # BPTT 几百步，不裁剪容易出 NaN
            optimizer.step()
            losses.append(loss.item())
        val = evaluate_nll(model, data, data["val"], n_repeats=2).mean().item()
        history.append((float(np.mean(losses)), val))
        if val < best_val:
            best_val, best_state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0:
            print(f"    第 {epoch + 1:>3} 轮  训练 {history[-1][0]:.4f}   验证 {val:.4f}   已用时 {time.time() - start:4.0f} 秒")
    model.load_state_dict(best_state)
    best_epoch = int(np.argmin([h[1] for h in history])) + 1
    return model, history, best_epoch, time.time() - start


def train_main(data, epochs):
    print("=" * 60)
    print("第四部分：训练 M")
    print("=" * 60)
    print(f"""
设备 {DEVICE}。每个批次 32 个完整回合，z 每次重新采样；梯度范数裁剪到 1.0。
下面的损失是"每步每维的平均负对数似然"（单位 nat）。每一轮结束后在验证回合上评估，最后保留验证损失最低的参数。
    """)
    model, history, best_epoch, elapsed = train_mdn_rnn(data, epochs)
    torch.save(model.state_dict(), MDNRNN_PATH)
    print(f"""
训练 {epochs} 轮用时 {elapsed:.0f} 秒。验证损失最低出现在第 {best_epoch} 轮（{history[best_epoch - 1][1]:.4f}），
最后一轮是 {history[-1][1]:.4f}，训练损失最后是 {history[-1][0]:.4f}。
💾 M 已保存：{MDNRNN_PATH.relative_to(OUT.parents[1])}
    """)
    return model, history


# ==================== 第五部分：这个损失算好还是不好 ====================


@torch.no_grad()
def baselines(data, seed=1234):
    """三个参照物在验证集上的逐维负对数似然 [32]。

    静态基线    z_{t+1} ~ N(m, v)：完全不看输入，m、v 是训练集上 z 每一维的均值和方差
    抄上一帧    z_{t+1} ~ N(z_t, s²)：s² 是训练集上 z_{t+1} - z_t 每一维的方差
    噪声下限    z_{t+1} ~ N(μ_{t+1}, σ²_{t+1})：事先知道下一帧的编码，只剩 V 的采样噪声
    """
    generator = torch.Generator().manual_seed(seed)
    tr, va = data["train"].to(DEVICE), data["val"].to(DEVICE)
    z_tr, z_va = sample_z(data, tr, generator), sample_z(data, va, generator)
    valid_tr, valid_va = data["valid"][tr], data["valid"][va]

    def gaussian_nll(target, mean, var):
        return 0.5 * math.log(2 * math.pi) + 0.5 * var.log() + (target - mean) ** 2 / (2 * var)

    nxt_tr, nxt_va = z_tr[:, 1:][valid_tr], z_va[:, 1:][valid_va]     # [N, 32]
    static = gaussian_nll(nxt_va, nxt_tr.mean(dim=0), nxt_tr.var(dim=0)).mean(dim=0)
    diff_var = (z_tr[:, 1:] - z_tr[:, :-1])[valid_tr].var(dim=0)
    copy_last = gaussian_nll(nxt_va, z_va[:, :-1][valid_va], diff_var).mean(dim=0)
    # 噪声下限是高斯的熵：0.5 log(2πe) + log σ，对验证集的全部目标帧取平均
    floor = (0.5 * math.log(2 * math.pi * math.e) + 0.5 * data["logvar"][va][:, 1:][valid_va]).mean(dim=0)
    return static.cpu(), copy_last.cpu(), floor.cpu()


@torch.no_grad()
def mean_prediction_error(model, data, active, seed=1234):
    """把 M 的预测分布压成均值（Σ π_k μ_k），和真正的 μ_{t+1} 比，只看在用的维度。返回三个均方根误差。"""
    generator = torch.Generator().manual_seed(seed)
    va = data["val"].to(DEVICE)
    valid = data["valid"][va]
    z = sample_z(data, va, generator)
    (log_pi, mu, _), _ = model(z[:, :-1], data["actions"][va][:, :-1])
    pred = (log_pi.exp() * mu).sum(dim=-1)                            # [B, T, 32]
    target = data["mu"][va][:, 1:]
    rmse = lambda x: (x - target)[valid][:, active].pow(2).mean().sqrt().item()
    return rmse(pred), rmse(z[:, :-1]), rmse(data["mu"][va][:, :-1])


def judge_loss(model, data, active, history):
    print("=" * 60)
    print("第五部分：这个损失算好还是不好")
    print("=" * 60)
    nll = evaluate_nll(model, data, data["val"])
    nll_shuffled = evaluate_nll(model, data, data["val"], shuffle_actions=True)
    static, copy_last, floor = baselines(data)
    act, idle = active.cpu(), ~active.cpu()
    rows = [("静态基线：不看输入，只用每一维的总体均值和方差", static),
            ("抄上一帧：N(z_t, s²)", copy_last),
            ("M", nll),
            ("M，但把动作换成别的回合的", nll_shuffled),
            ("噪声下限：事先知道下一帧的 μ 和 σ", floor)]
    print(f"\n验证集上每步每维的平均负对数似然（nat），越低越好：\n")
    width = lambda text: sum(2 if ord(ch) > 127 else 1 for ch in text)
    print(f"    {'全部 32 维':>10}  {'在用的':>8}  {'不在用的':>8}")
    for name, values in rows:
        print(f"    {values.mean().item():10.3f}  {values[act].mean().item():9.3f}  {values[idle].mean().item():10.3f}    {name}")

    gain = (static[act].mean() - nll[act].mean()).item()
    room = (static[act].mean() - floor[act].mean()).item()
    action_cost = (nll_shuffled[act].mean() - nll[act].mean()).item()
    rmse_m, rmse_copy_noisy, rmse_copy_clean = mean_prediction_error(model, data, active)
    print(f"""
怎么读这张表：
    1. 不在用的维度上，除了"抄上一帧"，各行几乎一样，都接近 0.5 ln(2πe) = {0.5 * math.log(2 * math.pi * math.e):.3f}——标准高斯的熵。
       那里没有东西可学，M 做对了唯一能做的事：输出 N(0, 1)。"全部 32 维"的平均被这 {int(idle.sum())} 维冲淡了，要看"在用的"那一列。
    2. "抄上一帧"在全部 32 维上的平均是 {copy_last.mean().item():.3f}，比什么都不看的静态基线（{static.mean().item():.3f}）还{'差' if copy_last.mean() > static.mean() else '好'}。
       z 是采样出来的，z_t 自己带着 V 的编码噪声，拿它当预测，噪声算了两遍：不在用的维度上它是 {copy_last[idle].mean().item():.3f}。
       在用的维度上它是 {copy_last[act].mean().item():.3f}，M 是 {nll[act].mean().item():.3f}。用采样的 z 训练时，更该比的是静态基线和噪声下限。
    3. 在用的维度上，静态基线到噪声下限之间有 {room:.3f} nat 的空间，M 拿下了其中的 {gain:.3f}（{gain / room * 100:.0f}%）。
    4. 把动作换成别的回合的，损失上升 {action_cost:.3f} nat。""")
    if action_cost > 0.02 * gain:
        print(f"       M 用到了动作，但动作贡献的只是它全部收益（{gain:.3f}）的 {action_cost / gain * 100:.0f}%。一步之内，画面主要由惯性决定。")
    else:
        print("       这个差别很小：一步之内画面主要由惯性决定，M 几乎没有用上动作。")
    print(f"""
负对数似然不直观，换一个量再看一遍：把 M 的预测分布压成均值，和下一帧真正的 μ 比（在用的维度，均方根误差）：

    M 的预测均值                 {rmse_m:.3f}
    抄上一帧，用采样的 z_t        {rmse_copy_noisy:.3f}      （M 的输入就是它）
    抄上一帧，用干净的 μ_t        {rmse_copy_clean:.3f}      （M 看不到 μ_t，只看得到带噪声的 z_t）
    """)
    if rmse_m < rmse_copy_clean:
        print("M 的输入带着编码噪声，预测却比干净的 μ_t 还准：它既从历史里平均掉了噪声，也学到了画面一步之内怎么变。\n")
    elif rmse_m < rmse_copy_noisy:
        print(f"""M 比它拿到的输入 z_t 准得多（{rmse_copy_noisy:.3f} → {rmse_m:.3f}）：它把一串带噪声的 z 平均成了对 μ 的估计，这是记忆在起作用。
但它没有超过"干净的上一帧"（{rmse_copy_clean:.3f}）。也就是说，M 的收益主要来自去噪，而不是预测画面下一步怎么变。
单步损失再低，也回答不了"它能不能往前推演很多步"。那要做开环展开才知道——下一节。
        """)
    else:
        print("这次运行里 M 的预测均值没有比输入更准（训练轮数太少时会这样，比如冒烟模式）。\n")

    epochs = np.arange(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(epochs, [h[0] for h in history], label="训练")
    ax.plot(epochs, [h[1] for h in history], label="验证")
    for value, name, color in [(static.mean(), "静态基线", "gray"), (floor.mean(), "噪声下限", "black")]:
        ax.axhline(value.item(), color=color, ls="--", lw=1, label=name)
    ax.set_xlabel("轮数")
    ax.set_ylabel("每步每维的负对数似然（nat，全部 32 维）")
    ax.set_title("M 的训练曲线：夹在静态基线和噪声下限之间")
    ax.legend()
    save_figure(fig, OUT / "04-training-curve.png")
    return nll


@torch.no_grad()
def mixture_usage(model, data, active):
    """M 的 5 个成分实际用了几个；并画出一个验证回合上的单步预测。"""
    va = data["val"].to(DEVICE)
    valid = data["valid"][va]
    generator = torch.Generator().manual_seed(SEED)
    z = sample_z(data, va, generator)
    (log_pi, mu, log_sigma), _ = model(z[:, :-1], data["actions"][va][:, :-1])
    pi = log_pi.exp()[valid]                                          # [N, 32, 5]
    n_effective = torch.exp(-(pi * pi.clamp_min(1e-12).log()).sum(dim=-1))   # [N, 32]，有效成分数 exp(熵)
    dominated = (pi.max(dim=-1).values > 0.9).float()                 # 一个成分占了九成以上
    act = active
    print(f"""混合成分用了几个（验证集，对全部时刻取平均）：

                      有效成分数（1 = 单峰，5 = 五个成分均分）    一个成分权重超过 0.9 的时刻
    在用的 {int(act.sum()):>2} 维              {n_effective[:, act].mean().item():.2f}                              {dominated[:, act].mean().item() * 100:5.1f}%
    不在用的 {int((~act).sum()):>2} 维            {n_effective[:, ~act].mean().item():.2f}                              {dominated[:, ~act].mean().item() * 100:5.1f}%

K = 5 是容量上限，不是"未来有 5 种情况"。有效成分数大于 1 也不等于多峰：几个成分的 μ 很接近时，
它们只是合起来拼一个形状。要判断是不是真的多峰，得看各成分的 μ 是否分得开（这里没有做这个检验）。
    """)

    # 画一个验证回合：KL 最大的那一维，真实的 μ、采样的 z、M 的单步预测
    kl = 0.5 * (data["mu"] ** 2 + data["logvar"].exp() - 1 - data["logvar"]).mean(dim=(0, 1))
    dim = int(kl.argmax())
    e = int(data["lengths"][data["val"]].argmax())                    # 挑验证集里最长的回合
    n = int(valid[e].sum())
    pred_mean = (log_pi.exp() * mu).sum(dim=-1)[e, :n, dim].cpu()
    second = (log_pi.exp() * (mu**2 + log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX).exp() ** 2)).sum(dim=-1)[e, :n, dim].cpu()
    pred_std = (second - pred_mean**2).clamp_min(0).sqrt()
    steps = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.scatter(steps, z[e, 1:n + 1, dim].cpu(), s=5, color="lightgray", label="采样的 z（M 的训练目标）")
    ax.plot(steps, data["mu"][va][e, 1:n + 1, dim].cpu(), color="black", lw=1.5, label="V 编码的 μ")
    ax.plot(steps, pred_mean, color="tab:blue", lw=1.5, label="M 的单步预测（混合的均值）")
    ax.fill_between(steps, pred_mean - pred_std, pred_mean + pred_std, color="tab:blue", alpha=0.2, label="± 1 个标准差")
    ax.set_xlabel("时间步 t")
    ax.set_ylabel(f"z 的第 {dim} 维")
    ax.set_title("teacher forcing 下的单步预测（一个验证回合）")
    ax.legend(fontsize=8, ncol=4, loc="upper center")
    save_figure(fig, OUT / "04-one-step-prediction.png")


# ==================== 第六部分：h 里有没有车速 ====================


def probe_speed_from_state(train_x, train_y, val_x, val_y, ridge=1e-3):
    """线性探针：在训练回合上做岭回归，返回验证回合上的决定系数 R²。

    特征先按训练集标准化。这是纯 NumPy 的 CPU 计算，用 float64。
    """
    mean, std = train_x.mean(axis=0), train_x.std(axis=0) + 1e-6
    add_bias = lambda x: np.concatenate([(x - mean) / std, np.ones((len(x), 1))], axis=1).astype(np.float64)
    a, b = add_bias(train_x), add_bias(val_x)
    weights = np.linalg.solve(a.T @ a + ridge * len(a) * np.eye(a.shape[1]), a.T @ train_y.astype(np.float64))
    residual = val_y - b @ weights
    return 1.0 - residual.var() / val_y.var()


@torch.no_grad()
def speed_probe(model, data):
    print("=" * 60)
    print("第六部分：h 里有没有车速")
    print("=" * 60)
    # 控制器运行时用的是 μ（不采样），这里也用 μ 当输入
    hidden, _ = model.encode_history(data["mu"][:, :-1], data["actions"][:, :-1])     # 第 t 个是 h_{t+1}
    h_before = torch.cat([torch.zeros_like(hidden[:, :1]), hidden[:, :-1]], dim=1)   # 第 t 个是 h_t（没看过 z_t）
    z_now = data["mu"][:, :-1]
    speed = data["speeds"][:, :-1]

    def gather(features, episodes):
        idx = episodes.to(DEVICE)
        return features[idx][data["valid"][idx]].cpu().numpy()

    results = {}
    for name, features in [("z_t", z_now), ("h_t", h_before), ("[z_t ; h_t]", torch.cat([z_now, h_before], dim=-1))]:
        results[name] = probe_speed_from_state(gather(features, data["train"]), gather(speed, data["train"]),
                                               gather(features, data["val"]), gather(speed, data["val"]))
    speed_val = gather(speed, data["val"])
    print(f"""
仿真器内部知道真实车速（01 节采数据时顺手存了，V 和 M 的训练都没用过它）。
冻结 M，用一个线性模型从特征里读车速：在训练回合上拟合，在验证回合上算决定系数 R²
（1 表示完全读得出，0 表示和只猜平均值一样）。验证回合的车速：均值 {speed_val.mean():.1f}，标准差 {speed_val.std():.1f}。

    特征                 维度     R²
    z_t（只有当前帧）       32     {results['z_t']:.3f}
    h_t（M 的记忆）        256     {results['h_t']:.3f}
    [z_t ; h_t]           288     {results['[z_t ; h_t]']:.3f}
    """)
    if results["[z_t ; h_t]"] > 0.3 and results["[z_t ; h_t]"] > results["z_t"] + 0.2:
        print(f"""单帧的 z 读车速的 R² 是 {results['z_t']:.2f}，加上 h 之后是 {results['[z_t ; h_t]']:.2f}。
单帧的 z 并非完全读不出车速：画面里有间接线索（车在赛道的什么位置、有没有开上草地），但只够解释一部分。
训练 M 时没有人告诉它"车速"这个量。它只是为了预测下一个 z，就自己把车速从 z 的前后变化里算出来存进了 h。
注意 h_t 还没看过当前这一帧（下标的约定），光靠它就能读出车速。这就是控制器要同时拿 z 和 h 的原因（07 节）。
        """)
    else:
        print("这次运行里加上 h 之后车速仍然读不出来（训练轮数太少时会这样，比如冒烟模式）。\n")
    return results


# ==================== 第七部分：练习 ====================


def exercises():
    print("=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把 sample_z 改成直接返回 μ（不采样），重新训练。验证损失怎么变？
        "抄上一帧"基线怎么变？这时再比 M 和基线，结论有什么不同？

练习 2：把 N_COMPONENTS 改成 1（单高斯），重新训练，比较"在用的维度"上的负对数似然。
        差距大吗？结合第五部分末尾的"有效成分数"解释。

练习 3：把 train_mdn_rnn 里的梯度裁剪那一行删掉，学习率调到 1e-2，观察训练是否稳定。

练习 4：手算。LSTM 的隐藏维度从 256 改成 512，参数量变成多少？（按论文的算法，每组门一个偏置）

思考题 1：第五部分里，把动作换掉之后损失只上升了一点。这说明动作对开车不重要吗？
          怎样的评估才能看出动作的作用？

思考题 2：M 的输入是采样的 z（带噪声），控制器运行时给 M 的却是 μ（不带噪声）。
          这两种输入的分布不同，会出问题吗？

参考答案：
    练习 1：目标里没有采样噪声了，M 可以把 σ 学得很小，负对数似然会大幅下降（变成负数）。
            "抄上一帧"也跟着变强：N(μ_t, s²) 里的 s 只剩相邻两帧 μ 的变化，很小。
            这时 M 要超过它，就得真的预测出画面怎么变，比较更严格。两种设定下的损失数值不能互相比。
    练习 2：差距通常很小。大多数时刻每一维的预测接近单峰，多出来的成分只是在拼形状。
            MDN 的价值体现在少数真正分叉的时刻，平均损失里看不大出来。
    练习 3：一条序列反向传播几百步，梯度容易爆炸；MDN 损失在 σ 很小时梯度又很大。
            两者叠加，常见的现象是损失突然跳成很大的数，或者直接出现 NaN。
    练习 4：LSTM：4 × 512 × (35 + 512) + 4 × 512 = 1,122,304；MDN 头：512 × 480 + 480 = 246,240；
            合计 1,368,544。论文 Doom 实验用的就是 512 个单元（它的 z 是 64 维，参数量另算）。
    思考题 1：不说明。单步之内画面主要由惯性决定，动作的影响要累积很多步才显现（方向盘打一下，
              车头要过一阵才转过去）。要看动作的作用，得做多步的开环展开：同一个起点、不同的动作序列，
              看梦境是否走向不同的结果。下一节做这件事。
    思考题 2：μ 是 z 的分布的中心，属于 M 见过的范围，一般不会出问题，论文也是这样做的。
              反过来（用干净的 μ 训练，用带噪声的 z 运行）才危险：M 没见过输入的抖动。
              做梦时情况又不同：输入是 M 自己采样的 ẑ，那是另一种分布偏移（下一节）。
    """)


def main():
    seed_everything(SEED)
    epochs = 3 if is_quick() else 150

    introduction()
    data = load_latents()
    active = describe_data(data)
    check_model()
    model, history = train_main(data, epochs)
    judge_loss(model, data, active, history)
    mixture_usage(model, data, active)
    speed_probe(model, data)
    exercises()

    print("=" * 60)
    print("本节小结")
    print("=" * 60)
    print("""
    1. M = LSTM + MDN 头。输入 concat(z_t, a_t)，输出 z_{t+1} 每一维的 5 个成分的混合参数 [B, T, 32, 5]。
    2. 训练用 teacher forcing，损失是序列上的 MDN 负对数似然，只算真实存在的转移，要裁剪梯度。
    3. 损失数值要有参照：上面是静态基线，下面是 V 的采样噪声决定的下限。先分清哪些维度有东西可学。
    4. 单步损失低，主要说明 M 会去噪、会利用惯性；它用没用动作、能不能多步推演，要另外检验。
    5. h 里有单帧没有的信息（车速），而且是 M 为了预测自己学出来的。

下一节：让 M 脱离真实数据，自己一步一步往前生成——在隐空间里做梦。
    """)
    if is_quick():
        print("（当前是冒烟模式 WM_QUICK=1：只训练了几轮，上面的数字没有参考价值。）")


if __name__ == "__main__":
    main()
