"""
在隐空间里做梦
==============

学习目标：
    1. 实现带温度的 MDN 采样，和手算的例子逐位核对
    2. 实现做梦：热身 → 自回归展开 → （可选）解码成画面
    3. 量出梦的走样：同一起点、同一串动作下，梦和真实的距离随步数怎么变，和两个平凡基线比
    4. 检验 M 在多步展开里用没用动作：同一个状态出发左打和右打，真实环境分开多少，梦里分开多少，梦跟的是不是对的那一路
    5. 看温度对梦的影响：多样性和准确度是两回事

核心概念：
    - 梦（dream）：隐空间里的开环展开，每一步从 MDN 采样，再喂回去
    - 温度 τ：logits 除以 τ，标准差乘 √τ。只在采样时起作用
    - 热身（warm-up）：先让 M 读一段真实序列，得到有内容的 h，再开始做梦
    - 反事实检验：同一个起点换一串动作，看模型的预测是否跟着变，变的方向对不对

前置知识：
    - 04-mdn-rnn-memory.py（M 的结构和训练）
    - Phase 1 的 04-compounding-error.py（开环展开与复合误差）

对应概念文档：
    - CONCEPT.md 第 5 节「在隐空间里做梦」

预计运行时间：约 1.5 分钟（MPS；其中第四部分要在真实环境里重放 240 段，4 个进程）
"""

import math
import os
import subprocess
import sys
from multiprocessing import get_context

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from utils import get_device, is_quick, output_dir, save_figure, save_video, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-02")
SEED = 0
Z_DIM, ACTION_DIM, HIDDEN_DIM, N_COMPONENTS = 32, 3, 256, 5
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -5.0, 2.0
ZOOM_FRAMES, CROP_ROWS, IMAGE_SIZE = 50, 84, 64      # 和 01 节的预处理保持一致
N_WORKERS = 4                                        # 第四部分在真实环境里重放用的进程数
DEVICE = get_device()
SUFFIX = "-quick" if is_quick() else ""
DATASET_PATH = OUT / f"carracing-random{SUFFIX}.npz"
LATENTS_PATH = OUT / f"latents{SUFFIX}.npz"
VAE_PATH = OUT / f"vae{SUFFIX}.pt"
MDNRNN_PATH = OUT / f"mdnrnn{SUFFIX}.pt"


# ==================== 第一部分：清醒与做梦 ====================


def introduction():
    print("=" * 60)
    print("第一部分：清醒与做梦")
    print("=" * 60)
    print("""
上一节的 M 每一步都拿到真实的 z（teacher forcing）。现在把真实数据撤掉，让它自己往前走：

   清醒（训练时）                               做梦（本节）

   真实画面 ─▶ V ─▶ z_t                        上一步采样得到的 ẑ_t
                     │                                  │
                     ▼                                  ▼
           a_t ─▶ ┌─────┐                     a_t ─▶ ┌─────┐
                  │  M  │                            │  M  │
                  └─────┘                            └─────┘
                     │ p(z_{t+1})                       │ p(z_{t+1})
                     ▼                                  ▼
                只用来算损失                      按温度 τ 采样得到 ẑ_{t+1}，
              下一个 z 仍来自真实画面              喂回去当下一步的输入

用 Phase 1 的话说：左边是单步预测，右边是开环展开。上一节的结尾留了两个疑问——
M 的单步收益主要来自去噪；动作只贡献了一小部分。这一节用开环展开来回答：它到底能不能推演。
    """)


# ==================== 模型（和 02、04 节相同，这里只为加载权重） ====================


class ConvVAE(nn.Module):
    """02 节的卷积 VAE，结构原样照搬。"""

    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(), nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(), nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(), nn.Flatten())
        self.fc_mu = nn.Linear(1024, z_dim)
        self.fc_logvar = nn.Linear(1024, z_dim)
        self.fc_decode = nn.Linear(z_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(), nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(), nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid())

    def encode(self, obs):
        hidden = self.encoder(obs)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def decode(self, z):
        return self.decoder(self.fc_decode(z).view(-1, 1024, 1, 1))


class MDNRNN(nn.Module):
    """04 节的 MDN-RNN，结构原样照搬。"""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(Z_DIM + ACTION_DIM, HIDDEN_DIM, batch_first=True)
        self.mdn_head = nn.Linear(HIDDEN_DIM, Z_DIM * N_COMPONENTS * 3)

    def forward(self, z, a, state=None):
        hidden, state = self.lstm(torch.cat([z, a], dim=-1), state)
        out = self.mdn_head(hidden).view(*hidden.shape[:-1], Z_DIM, N_COMPONENTS, 3)
        logits, mu, log_sigma = out.unbind(dim=-1)
        return (torch.log_softmax(logits, dim=-1), mu, log_sigma), state

    def step(self, z, a, state=None):
        (log_pi, mu, log_sigma), state = self.forward(z.unsqueeze(1), a.unsqueeze(1), state)
        return (log_pi.squeeze(1), mu.squeeze(1), log_sigma.squeeze(1)), state


def ensure_artifact(path, lesson):
    """跨讲义共享的产物不存在时，先运行生成它的那一节讲义（继承当前的 WM_QUICK 设置）。"""
    if path.exists():
        return
    print(f"⚠️  没找到 {path.name}，先运行 {lesson} 生成它……")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), lesson)
    subprocess.run([sys.executable, script], check=True, stdout=subprocess.DEVNULL)


def load_everything():
    ensure_artifact(LATENTS_PATH, "02-conv-vae-vision.py")
    ensure_artifact(MDNRNN_PATH, "04-mdn-rnn-memory.py")
    vae, mdn_rnn = ConvVAE().to(DEVICE), MDNRNN().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    mdn_rnn.load_state_dict(torch.load(MDNRNN_PATH, map_location=DEVICE))
    vae.eval(), mdn_rnn.eval()
    with np.load(LATENTS_PATH) as f:
        raw = {k: f[k] for k in f.files}
    data = {k: torch.from_numpy(raw[k]).to(DEVICE) for k in ("mu", "logvar", "actions")}
    data["lengths"] = raw["lengths"]
    n_episodes = len(raw["lengths"])
    n_val = max(1, n_episodes // 10)
    data["val"] = np.arange(n_episodes - n_val, n_episodes)          # 和 02、04 节相同的验证回合
    data["train"] = np.arange(n_episodes - n_val)
    kl = 0.5 * (data["mu"] ** 2 + data["logvar"].exp() - 1 - data["logvar"]).mean(dim=(0, 1))
    active = kl > 0.1
    if int(active.sum()) == 0:                                        # 冒烟模式下 V 训练太少
        active[kl.topk(4).indices] = True
    data["active"] = active
    return vae, mdn_rnn, data


# ==================== 第二部分：带温度的采样 ====================


def sample_mdn(log_pi, mu, log_sigma, temperature=1.0, generator=None):
    """带温度的 MDN 采样，逐维独立。参数 [B, 32, 5] → ẑ [B, 32]。

    选成分：k ~ Categorical(softmax(logits / τ))；采值：ẑ = μ_k + √τ · σ_k · ε。
    随机数在 CPU 上用 generator 生成，结果可复现。
    """
    probs = torch.softmax(log_pi / temperature, dim=-1).cpu()                        # [B, 32, 5]
    k = torch.multinomial(probs.view(-1, probs.shape[-1]), 1, generator=generator)   # [B × 32, 1]
    k = k.view(*probs.shape[:-1], 1).to(mu.device)
    mu_k = mu.gather(-1, k).squeeze(-1)
    sigma_k = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX).exp().gather(-1, k).squeeze(-1)
    eps = torch.randn(mu_k.shape, generator=generator).to(mu.device)
    return mu_k + math.sqrt(temperature) * sigma_k * eps


def check_temperature():
    print("=" * 60)
    print("第二部分：带温度的采样——和手算核对")
    print("=" * 60)
    pi = torch.tensor([0.7, 0.2, 0.1])
    print("\n手算例子：某一维的混合权重 π = (0.7, 0.2, 0.1)，三个成分的 σ 都是 0.2，μ = (-10, 0, 10)（离得足够远，每个采样值属于哪个成分一目了然）。")
    print("\n    温度     调整后的权重（公式）        采样 20 万次统计出的权重      采样值在成分内的标准差（公式 √τ × 0.2）")
    generator = torch.Generator().manual_seed(SEED)
    n = 200_000
    log_pi = pi.log().expand(n, 1, 3)
    mu = torch.tensor([-10.0, 0.0, 10.0]).expand(n, 1, 3)
    log_sigma = torch.full((n, 1, 3), math.log(0.2))
    for tau in (0.5, 1.0, 2.0):
        expected = torch.softmax(pi.log() / tau, dim=0)
        samples = sample_mdn(log_pi, mu, log_sigma, tau, generator).squeeze(-1)
        nearest = (samples.unsqueeze(-1) - torch.tensor([-10.0, 0.0, 10.0])).abs().argmin(dim=-1)
        freq = torch.bincount(nearest, minlength=3).float() / n
        std = (samples - torch.tensor([-10.0, 0.0, 10.0])[nearest]).std().item()
        print(f"    τ = {tau:<4}  ({expected[0]:.3f}, {expected[1]:.3f}, {expected[2]:.3f})        "
              f"({freq[0]:.3f}, {freq[1]:.3f}, {freq[2]:.3f})            {std:.3f}（公式 {math.sqrt(tau) * 0.2:.3f}）")
        assert (freq - expected).abs().max() < 0.01 and abs(std - math.sqrt(tau) * 0.2) < 0.01
    print("""
τ < 1：大的权重更大，成分内部更集中，梦更"规矩"。τ > 1：反过来，少见的成分更常被抽到。
温度只在采样时起作用，训练好的 M 不变。
    """)


# ==================== 第三部分：做梦 ====================


@torch.no_grad()
def warm_up_state(mdn_rnn, z_seq, a_seq):
    """热身：让 M 读入一段真实序列 z [B, W, 32]、a [B, W, 3]，返回读完之后的 LSTM 状态。"""
    _, state = mdn_rnn(z_seq, a_seq)
    return state


@torch.no_grad()
def dream_rollout(mdn_rnn, z0, state, actions, temperature=1.0, generator=None):
    """自回归展开。z0 [B, 32]，actions [B, H, 3] → 梦的隐向量 [B, H, 32]（第 i 个是第 i+1 步之后的 ẑ）。"""
    z, dream = z0, []
    for t in range(actions.shape[1]):
        params, state = mdn_rnn.step(z, actions[:, t], state)
        z = sample_mdn(*params, temperature, generator)
        dream.append(z)
    return torch.stack(dream, dim=1)


@torch.no_grad()
def decode_dream(vae, z_seq):
    """把隐向量序列 [H, 32] 解码成画面 [H, 64, 64, 3] uint8。"""
    frames = vae.decode(z_seq).permute(0, 2, 3, 1)
    return (frames.clamp(0, 1) * 255).byte().cpu().numpy()


def pick_starts(data, episodes, warmup, horizon, per_episode):
    """在给定回合里挑起点 t0，保证 t0 - warmup ≥ 0 且 t0 + horizon 不超出回合。返回 [(回合, t0), …]。"""
    starts = []
    for e in episodes:
        last = int(data["lengths"][e]) - 1 - horizon
        if last >= warmup:
            starts += [(int(e), int(t)) for t in np.linspace(warmup, last, per_episode).astype(int)]
    return starts


@torch.no_grad()
def dream_from_data(mdn_rnn, data, starts, warmup, horizon, temperature, n_samples, generator, actions=None):
    """从数据集里的若干起点出发做梦。actions 不给时用录好的动作；给了就用它（[N, H, 3]，反事实检验用）。

    返回：梦 [S, N, H, 32]、录好的动作下真实的 μ [N, H, 32]、起点的 μ [N, 32]。S 是每个起点重复做梦的次数。
    """
    ep = torch.tensor([s[0] for s in starts], device=DEVICE)
    t0 = torch.tensor([s[1] for s in starts], device=DEVICE)
    window = lambda key, lo, hi: torch.stack([data[key][e, t + lo:t + hi] for e, t in starts])
    state = warm_up_state(mdn_rnn, window("mu", -warmup, 0), window("actions", -warmup, 0))
    z0 = data["mu"][ep, t0]                                           # 起点用 μ（控制器运行时也是这样）
    if actions is None:
        actions = window("actions", 0, horizon)
    real = window("mu", 1, horizon + 1)
    tile = lambda x: x.repeat(n_samples, *[1] * (x.dim() - 1))
    state = tuple(s.repeat(1, n_samples, 1) for s in state)
    dream = dream_rollout(mdn_rnn, tile(z0), state, tile(actions), temperature, generator)
    return dream.view(n_samples, len(starts), horizon, Z_DIM), real, z0


def drift_experiment(mdn_rnn, data, warmup, horizon, per_episode, n_samples, temperatures):
    print("=" * 60)
    print("第三部分：梦会走样——和真实轨迹比")
    print("=" * 60)
    starts = pick_starts(data, data["val"], warmup, horizon, per_episode)
    active = data["active"]
    generator = torch.Generator().manual_seed(SEED)
    rmse = lambda a, b: (a - b)[..., active].pow(2).mean(dim=-1).sqrt()    # 在用的维度上的均方根误差
    curves, spreads = {}, {}
    for tau in temperatures:
        dream, real, z0 = dream_from_data(mdn_rnn, data, starts, warmup, horizon, tau, n_samples, generator)
        curves[tau] = rmse(dream, real).mean(dim=(0, 1)).cpu().numpy()            # 单个梦离真实多远 [H]
        spreads[tau] = dream[..., active].std(dim=0).mean(dim=(0, 2)).cpu().numpy()   # 同一起点的梦彼此散多开 [H]
        if tau == 1.0:
            mean_dream = rmse(dream.mean(dim=0), real).mean(dim=0).cpu().numpy()  # 多个梦取平均之后离真实多远
    freeze = rmse(z0.unsqueeze(1).expand_as(real), real).mean(dim=0).cpu().numpy()
    train_mean = data["mu"][torch.as_tensor(data["train"], device=DEVICE)].mean(dim=(0, 1))
    static = rmse(train_mean.expand_as(real), real).mean(dim=0).cpu().numpy()

    marks = [t for t in (1, 5, 10, 20, 50, 100) if t <= horizon]
    print(f"""
{len(starts)} 个起点（全部来自验证回合），每个起点先让 M 读 {warmup} 步真实序列热身，
然后做梦 {horizon} 步，动作用数据集里录好的；每个起点重复 {n_samples} 次。
下表是梦的 ẑ 和 V 对真实画面的编码 μ 之间的距离（在用的 {int(active.sum())} 维上的均方根误差）：
    """)
    print("    步数                         " + "".join(f"{t:>8}" for t in marks))
    row = lambda name, values: print(f"    {name}" + "".join(f"{values[t - 1]:8.3f}" for t in marks))
    for tau in temperatures:
        row(f"梦，τ = {tau:<4}                 ", curves[tau])
    row("多个梦取平均，τ = 1.0        ", mean_dream)
    row("基线：画面定格在起点          ", freeze)
    row("基线：总是猜训练集的平均画面  ", static)
    print("\n同一起点的多个梦彼此之间散得多开（各维标准差的平均）：")
    for tau in temperatures:
        row(f"τ = {tau:<4}                     ", spreads[tau])

    low = min(temperatures)
    cross = next((t for t in range(1, horizon + 1) if curves[low][t - 1] >= static[t - 1]), None)
    beats_freeze = next((t for t in range(1, horizon + 1) if curves[low][t - 1] < freeze[t - 1]), None)
    print(f"""
怎么读：
    1. 误差随步数增大。这是 Phase 1 的复合误差，只是发生在隐空间里。
    2. τ = 1 的单个梦，第 1 步的误差就有 {curves[1.0][0]:.2f}：采样把 V 的编码噪声也采了进来（上一节：在用的维度 σ 平均 0.55）。
       梦是带噪声的 ẑ，真实这边是干净的 μ，这部分差距不是 M 的预测错误。看 τ = {low} 那一行更接近 M 的"预测中心"。
    3. 和"定格在起点"比：τ = {low} 的梦{f"从第 {beats_freeze} 步起误差更小" if beats_freeze else "在整个视野内都没有更准"}；""" +
          (f"到第 {cross} 步，它已经不比'总是猜平均画面'准。" if cross else f"{horizon} 步之内它一直比'总是猜平均画面'准。") + f"""
    4. 温度越高，同一起点的梦散得越开。多样不等于准确：散开的方向是否对应真实环境里的不同可能，这张表回答不了。
    """)

    steps = np.arange(1, horizon + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for tau in temperatures:
        axes[0].plot(steps, curves[tau], label=f"梦，τ = {tau}")
        axes[1].plot(steps, spreads[tau], label=f"τ = {tau}")
    axes[0].plot(steps, freeze, color="gray", ls="--", label="画面定格在起点")
    axes[0].plot(steps, static, color="black", ls=":", label="总是猜平均画面")
    axes[0].set_xlabel("做梦的步数")
    axes[0].set_ylabel("和真实 μ 的均方根误差（在用的维度）")
    axes[0].set_title("梦离真实轨迹多远")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("做梦的步数")
    axes[1].set_ylabel("同一起点的梦之间的标准差")
    axes[1].set_title("同一起点的梦彼此散多开")
    axes[1].legend(fontsize=8)
    save_figure(fig, OUT / "05-dream-drift.png")
    return curves, freeze, static


# ==================== 第四部分：梦里的方向盘管不管用 ====================


def preprocess(frame):
    """和 01 节相同的预处理：[96, 96, 3] uint8 → [3, 64, 64] float32。"""
    small = cv2.resize(frame[:CROP_ROWS], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return np.transpose(small, (2, 0, 1)).astype(np.float32) / 255.0


def _branch_worker(job):
    """子进程：把一个回合重放到分叉点，再从那个状态出发执行另一串动作。

    job = (环境种子, 分叉点之前录好的动作 [t0, 3], 分叉之后要执行的动作 [H, 3])。
    返回：分叉点和之后每一步的画面 [H'+1, 3, 64, 64]；每一步车头相对分叉点转过的角度 [H']（度）。
    回合提前结束时 H' < H。
    """
    seed, recorded_actions, branch_actions = job
    env = gym.make("CarRacing-v3", continuous=True)
    frame, _ = env.reset(seed=seed)
    for _ in range(ZOOM_FRAMES):
        frame, *_ = env.step(np.zeros(3, dtype=np.float32))
    for action in recorded_actions:
        frame, *_ = env.step(action)
    frames, heading = [preprocess(frame)], []
    angle_at_branch = env.unwrapped.car.hull.angle
    for action in branch_actions:
        frame, _, terminated, truncated, _ = env.step(action)
        frames.append(preprocess(frame))
        heading.append(math.degrees(env.unwrapped.car.hull.angle - angle_at_branch))
        if terminated or truncated:
            break
    env.close()
    return np.stack(frames), np.array(heading)


@torch.no_grad()
def branch_in_real_env(vae, data, starts, action_sequences):
    """对每一串动作 [H, 3]、每个分叉点 (回合, t0)，在真实环境里从那个状态出发执行这串动作。

    怎么回到"那个状态"：01 节采第 e 个回合时用的环境种子就是 e，物理仿真是确定性的，
    所以照着录好的动作再走一遍就回去了（调用方会校验）。N_WORKERS 个进程并行。
    返回 results[i][j]：第 i 串动作、第 j 个分叉点的
        (分叉点画面的 μ [32]，之后每一步画面的 μ [H', 32]，车头转过的角度 [H'])。
    """
    recorded = data["actions"].cpu().numpy()
    jobs = [(e, recorded[e, :t0], actions) for actions in action_sequences for e, t0 in starts]
    with get_context("spawn").Pool(min(N_WORKERS, len(jobs))) as pool:
        raw = pool.map(_branch_worker, jobs)
    encoded = []
    for frames, heading in raw:
        mu, _ = vae.encode(torch.from_numpy(frames).to(DEVICE))
        encoded.append((mu[0], mu[1:], heading))
    return [encoded[i * len(starts):(i + 1) * len(starts)] for i in range(len(action_sequences))]


def steering_experiment(vae, mdn_rnn, data, branch_times, warmup, horizon, n_samples, steers=(0.6, 1.0)):
    print("=" * 60)
    print("第四部分：梦里的方向盘管不管用")
    print("=" * 60)
    active = data["active"]
    rmse = lambda a, b: (a - b)[..., active].pow(2).mean(dim=-1).sqrt().cpu().numpy()    # 在用的维度上的均方根
    turn = lambda value: np.tile(np.array([value, 0.2, 0.0], dtype=np.float32), (horizon, 1))
    signed = [sign * s for s in steers for sign in (-1, +1)]                  # 例如 [-0.6, +0.6, -1.0, +1.0]

    # 1. 真实环境：每个分叉点、每一串动作各重放一次
    candidates = [(int(e), t0) for e in data["val"] for t0 in branch_times
                  if t0 >= warmup and t0 + horizon < int(data["lengths"][e])]
    if not candidates:
        print("\n没有可用的分叉点（验证回合太短），跳过这一部分。\n")
        return
    rollouts = dict(zip(signed, branch_in_real_env(vae, data, candidates, [turn(value) for value in signed])))
    # 只留下每一串动作都跑满 horizon 步的分叉点（有的分叉之后会冲出场地边界）
    keep = [i for i in range(len(candidates)) if all(len(rollouts[v][i][1]) == horizon for v in signed)]
    if not keep:
        print("\n所有分叉点在分叉之后都提前结束了，跳过这一部分。\n")
        return
    starts = [candidates[i] for i in keep]
    real = {v: torch.stack([rollouts[v][i][1] for i in keep]) for v in signed}            # 每个 [N, H, 32]
    turned = np.mean([np.abs(rollouts[signed[0]][i][2]) for i in keep], axis=0)           # [H]
    replay_gap = max((rollouts[signed[0]][i][0] - data["mu"][candidates[i]]).abs().max().item() for i in keep)

    # 2. 梦：同样的分叉点、同样的热身、同样的动作，每一串动作做 n_samples 个梦取平均
    generator = torch.Generator().manual_seed(SEED)
    as_batch = lambda value: torch.from_numpy(turn(value)).to(DEVICE).expand(len(starts), horizon, ACTION_DIM)
    dream = {v: dream_from_data(mdn_rnn, data, starts, warmup, horizon, 1.0, n_samples, generator,
                                as_batch(v))[0].mean(dim=0) for v in signed}              # 每个 [N, H, 32]

    # 3. 四个量，形状都是 [N, H]
    def measure(s):
        real_gap = rmse(real[-s], real[+s])                                   # 真实环境里两路分开多少
        dream_gap = rmse(dream[-s], dream[+s])                                # 梦里两路分开多少
        same = (rmse(dream[-s], real[-s]) + rmse(dream[+s], real[+s])) / 2    # 梦离"同一路"的真实多远
        cross = (rmse(dream[-s], real[+s]) + rmse(dream[+s], real[-s])) / 2   # 梦离"另一路"的真实多远
        return real_gap, dream_gap, same, cross

    steer = steers[0]
    real_gap, dream_gap, same, cross = measure(steer)
    right_side = cross > same                                                 # 梦是不是离同一路更近
    marks = [t for t in (1, 5, 10, 20, 30, 40) if t <= horizon]
    cell = lambda values: "".join(f"{values.mean(axis=0)[t - 1]:8.3f}" for t in marks)
    header = "".join(f"{t:>8}" for t in marks)
    print(f"""
反事实检验：同一个状态出发，只换动作，看后果跟不跟着变。
分叉点取自验证回合的第 {"、".join(str(t) for t in branch_times)} 步，共 {len(starts)} 个（车在赛道上的、在草地上的，快的、慢的都有）。
真实环境：把回合重放到分叉点，然后方向盘一直左打（-{steer}）或一直右打（+{steer}），各 {horizon} 步。
          重放校验：分叉点画面的编码和数据集里存的最大相差 {replay_gap:.1e}，确实回到了当时的状态。
梦：先读分叉点之前的 {warmup} 步真实序列热身，再用同样的两串动作各做 {n_samples} 个梦取平均（平均掉采样噪声）。
下面的距离都是画面编码 μ 在用的维度上的均方根，对 {len(starts)} 个分叉点取平均。{horizon} 步之后车头平均转过了 {turned[-1]:.0f}°。

    步数                          {header}
    真实环境：左打 vs 右打        {cell(real_gap)}
    梦里：左打 vs 右打            {cell(dream_gap)}
    梦 vs 同一路的真实            {cell(same)}
    梦 vs 另一路的真实            {cell(cross)}
    梦离同一路更近的分叉点占比    """ + "".join(f"{right_side.mean(axis=0)[t - 1] * 100:7.0f}%" for t in marks) + "\n")

    half = max(horizon // 2, 6)
    early, late = slice(4, half), slice(half, horizon)                        # 第 5 步到一半；后一半
    ratio = lambda window: dream_gap[:, window].mean() / max(real_gap[:, window].mean(), 1e-9)
    side = lambda window: right_side[:, window].mean()
    noise_floor = dream_gap.mean(axis=0)[0]
    print(f"""怎么读：
    1. 第 1 步是校准。动作还没来得及起作用，真实环境里两路只差 {real_gap.mean(axis=0)[0]:.3f}；梦里却差 {noise_floor:.3f}，
       这是 {n_samples} 个梦的平均没消干净的采样噪声。"梦离同一路更近"的占比是 {right_side.mean(axis=0)[0] * 100:.0f}%，和抛硬币差不多。
       所以 {noise_floor:.2f} 以下的差别不要当真。
    2. 第 5–{half} 步：梦里两路分开的幅度是真实的 {ratio(early) * 100:.0f}%，梦离同一路更近的比例是 {side(early) * 100:.0f}%。
       第 {half + 1}–{horizon} 步：幅度是真实的 {ratio(late) * 100:.0f}%，离同一路更近的比例是 {side(late) * 100:.0f}%。""")
    if ratio(early) > 0.75 and side(early) > 0.75:
        print("""       方向盘在梦里是管用的：两路分得开，而且分对了方向。上一节的单步损失里动作只占一小部分，
       不等于 M 没学会动作的后果——每一步的那点差别累积十几步，就是左转和右转的区别。""")
    elif ratio(early) < 0.5 or side(early) < 0.6:
        print("""       梦里两路没有像真实环境那样分开（或者分错了方向）：这个 M 在多步展开里没有用好动作。
       （训练轮数太少时会这样，比如冒烟模式。）""")
    else:
        print("       梦里两路分开了，但幅度或方向和真实环境有明显出入。")

    # 管用不等于准：换一个"真实后果完全相同"的动作数值，看梦的误差变不变
    if len(steers) > 1:
        other = steers[1]
        _, _, same_other, _ = measure(other)
        real_diff = max((real[sign * steer] - real[sign * other]).abs().max().item() for sign in (-1, +1))
        frame_ok = torch.arange(data["actions"].shape[1])[None, :] < torch.as_tensor(data["lengths"])[:, None]
        recorded_steer = data["actions"][..., 0].cpu()[frame_ok]              # 数据集里全部真实帧的方向盘
        beyond = lambda x: (recorded_steer.abs() > x).float().mean().item() * 100
        mid = min(20, horizon)
        err, err_other = same.mean(axis=0)[mid - 1], same_other.mean(axis=0)[mid - 1]
        print(f"""
    3. 管用不等于准。再打得狠一点：±{other}。先看真实环境：{len(starts)} 个分叉点、每一步画面的编码，±{steer} 和 ±{other} 两种打法最大相差 {real_diff:.1e}。
       CarRacing 里方向盘的数值是前轮转角的目标值，而前轮最多只能转 0.4 弧度：{steer} 和 {other} 都是"打到头"，后果一模一样。梦里呢：

    步数                          {header}
    梦 vs 同一路的真实，±{steer:<4}    {cell(same)}
    梦 vs 同一路的真实，±{other:<4}    {cell(same_other)}
""")
        if real_diff < 1e-3 and err_other > 1.1 * err and beyond(0.9) < 1.0:
            print(f"""       真实的后果一模一样，梦的误差在第 {mid} 步却从 {err:.3f} 变成了 {err_other:.3f}（大了 {(err_other / err - 1) * 100:.0f}%）。
       M 不知道方向盘有"打到头"这回事。数据里方向盘的标准差是 {recorded_steer.std().item():.2f}，绝对值超过 0.4 的帧占 {beyond(0.4):.1f}%，
       超过 {steer} 的占 {beyond(steer):.1f}%，超过 0.9 的占 {beyond(0.9):.2f}%：{other} 这个数值 M 几乎没见过，它在这里的预测没有任何保障。
       动作离数据越远，梦走样得越快——这是 Phase 1 第 5.6 节的分布偏移，这次偏的是动作。
       对"在梦里训练控制器"（08 节）来说要记住这一点：控制器的开法和随机策略差得越远，梦越不可信。""")
        elif real_diff >= 1e-3:
            print("       这次运行里两种打法在真实环境里的后果并不相同，上面两行不能直接比。")
        else:
            print(f"""       这次运行里这一步下不了结论：第 {mid} 步的误差是 {err:.3f} 和 {err_other:.3f}，数据里方向盘绝对值超过 0.9 的帧占 {beyond(0.9):.2f}%。
       （冒烟模式下模型只训练了几轮；或者数据里本来就有这样的动作，{other} 算不上"没见过"。）""")
    print()

    steps = np.arange(1, horizon + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(steps, real_gap.mean(axis=0), label="真实环境")
    axes[0].plot(steps, dream_gap.mean(axis=0), label=f"梦里（{n_samples} 个梦的平均）")
    axes[0].set_xlabel("分叉之后的步数")
    axes[0].set_ylabel("左打与右打的 μ 之差（在用的维度，均方根）")
    axes[0].set_title(f"同一个状态，方向盘 ±{steer}：两路分开多少（{len(starts)} 个分叉点的平均）")
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, same.mean(axis=0), label=f"梦 vs 同一路的真实，±{steer}")
    axes[1].plot(steps, cross.mean(axis=0), label=f"梦 vs 另一路的真实，±{steer}")
    if len(steers) > 1:
        axes[1].plot(steps, same_other.mean(axis=0), color="tab:blue", ls="--", label=f"梦 vs 同一路的真实，±{other}")
    axes[1].set_xlabel("分叉之后的步数")
    axes[1].set_ylabel("梦和真实的距离（在用的维度，均方根）")
    axes[1].set_title("梦跟的是不是对的那一路；动作离数据更远时准不准")
    axes[1].legend(fontsize=8)
    save_figure(fig, OUT / "05-steering-counterfactual.png")


# ==================== 第五部分：把梦解码成视频 ====================


@torch.no_grad()
def make_video(vae, mdn_rnn, data, warmup, horizon):
    print("=" * 60)
    print("第五部分：把梦解码成视频")
    print("=" * 60)
    starts = pick_starts(data, data["val"], warmup, horizon, 1)
    if not starts or not DATASET_PATH.exists():
        print("\n没有足够长的验证回合或找不到原始画面，跳过视频。\n")
        return
    e, t0 = max(starts, key=lambda s: data["lengths"][s[0]])
    with np.load(DATASET_PATH) as f:
        real_frames = f["frames"][e, t0 + 1:t0 + horizon + 1]                       # [H, 64, 64, 3]
    recon = decode_dream(vae, data["mu"][e, t0 + 1:t0 + horizon + 1])
    generator = torch.Generator().manual_seed(SEED)
    rows = [real_frames, recon]
    for tau in (0.1, 1.0):
        dream, _, _ = dream_from_data(mdn_rnn, data, [(e, t0)], warmup, horizon, tau, 1, generator)
        rows.append(decode_dream(vae, dream[0, 0]))
    big = lambda img: cv2.resize(img, (192, 192), interpolation=cv2.INTER_NEAREST)
    video = [np.concatenate([big(r[t]) for r in rows], axis=1) for t in range(horizon)]
    save_video(video, OUT / "05-dream.mp4", fps=20)
    picks = np.linspace(0, horizon - 1, 6).astype(int)
    fig, axes = plt.subplots(4, 6, figsize=(11, 7.4))
    for r, name in enumerate(["真实画面", "VAE 重建", "梦，τ = 0.1", "梦，τ = 1.0"]):
        for c, t in enumerate(picks):
            axes[r, c].imshow(rows[r][t])
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([]), axes[r, c].grid(False)
            if r == 0:
                axes[r, c].set_title(f"第 {t + 1} 步", fontsize=9)
        axes[r, 0].set_ylabel(name, fontsize=10)
    fig.suptitle("同一起点、同一串动作：真实、重建、梦", fontsize=12)
    save_figure(fig, OUT / "05-dream-frames.png")
    print(f"""
视频从左到右四栏：真实画面 ｜ VAE 重建 ｜ 梦（τ = 0.1）｜ 梦（τ = 1.0）。起点是验证回合 {e} 的第 {t0} 步，共 {horizon} 步。
前两栏的差别是 V 丢掉的东西；后两栏和第二栏的差别才是 M 的事。
τ = 1.0 的梦画面在抖：每一步都重新采了一次编码噪声。梦的画面"像不像赛车游戏"和它"准不准"是两回事，
准不准要看第三、四部分的数字。
    """)


# ==================== 第六部分：练习 ====================


def exercises():
    print("=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把热身步数改成 0（LSTM 从零状态开始，只给一帧 z_0），重新运行第三部分。
        前 10 步的误差怎么变？为什么？

练习 2：把 sample_mdn 改成"只对选成分用温度，采值时不乘 √τ"，对比 τ = 0.1 时梦的误差和多样性。

练习 3：第四部分把方向盘从 ±0.6 改成 ±0.3（更接近数据里常见的幅度），表里哪些行变化大，哪些几乎不变？

练习 4：手算。π = (0.5, 0.5)，τ = 0.1 时调整后的权重是多少？π = (0.6, 0.4) 呢？

思考题 1：第三部分里，"多个梦取平均"比任何单个梦都更接近真实。那做梦时直接输出平均值不是更好吗？
          （回忆 03 节）

思考题 2：第四部分里，动作序列离数据越远，梦走样得越快。要缓解这一点，可以从数据、损失、模型结构三方面分别做什么？

参考答案：
    练习 1：前几步的误差明显变大。没有热身，h 是空的，M 不知道车速和转向的趋势，
            只能按"平均情况"预测；读了十几步之后 h 才有内容。
    练习 2：τ → 0 时两种写法都趋于"选权重最大的成分并输出它的均值"。τ 较小但不为 0 时，
            不乘 √τ 的版本成分内部的抖动保持原样，梦更散、单个梦离真实更远。
    练习 3："真实环境：左打 vs 右打"那一行变化不大：0.3 离前轮转角的上限 0.4 不远，车转得只慢一点。
            "梦 vs 同一路的真实"明显变小：±0.3 是数据里常见的幅度，M 不用外推。
            这和第四部分的第 3 点是同一件事的两面：动作离数据近，梦就准一些。
    练习 4：(0.5, 0.5) 不管温度多少都是 (0.5, 0.5)。(0.6, 0.4)：取 10 次方，0.6¹⁰ = 0.00605，0.4¹⁰ = 0.000105，
            归一化后约 (0.983, 0.017)。低温把微小的权重差别放大成几乎确定的选择。
    思考题 1：平均值离每一个真实结果的平方距离最小，这正是 MSE 的最优解。但它不是任何一个可能的未来：
              两条岔路的平均是中间的墙。把平均喂回 M，下一步的输入就落在 M 没见过的区域。
    思考题 2：数据：用更多样的动作采集（大幅度、长时间的转向），或者用训练中的控制器迭代采集（CONCEPT.md 第 8.7 节）。
              损失：加多步展开的损失，让动作的累积后果进入目标（Phase 1 第 5.5 节）。
              结构：让表示本身为预测服务，而不是只为重建——这是 Phase 3 的 RSSM。
    """)


def main():
    seed_everything(SEED)
    if is_quick():
        warmup, horizon, per_episode, n_samples, branch_times, turn_horizon = 10, 30, 2, 4, (40,), 10
    else:
        warmup, horizon, per_episode, n_samples, branch_times, turn_horizon = 20, 100, 6, 16, (40, 100, 160), 40

    introduction()
    vae, mdn_rnn, data = load_everything()
    check_temperature()
    drift_experiment(mdn_rnn, data, warmup, horizon, per_episode, n_samples, (0.1, 1.0, 1.5))
    steering_experiment(vae, mdn_rnn, data, branch_times, warmup, turn_horizon, n_samples)
    make_video(vae, mdn_rnn, data, warmup, horizon)
    exercises()

    print("=" * 60)
    print("本节小结")
    print("=" * 60)
    print("""
    1. 做梦 = 隐空间里的开环展开：热身得到 h，之后每一步从 MDN 采样、喂回去。只需要 M，解码只是给人看。
    2. 温度：logits 除以 τ，标准差乘 √τ。它调的是梦的随机程度，不改变模型。
    3. 梦的质量要和真实轨迹比，并且要有基线（定格在起点、总是猜平均画面）。
    4. 梦"多样"和梦"准确"是两回事；梦的画面"像"和梦"准"也是两回事。
    5. 反事实检验（同一个状态出发换动作）能直接量出模型用没用动作。单步损失看不出这一点。
       这里的 M 用了动作，而且方向是对的；但动作离数据越远，梦越不准。

下一节：再离开赛车一次，学控制器的训练工具——进化策略与 CMA-ES。
    """)
    if is_quick():
        print("（当前是冒烟模式 WM_QUICK=1：模型只训练了几轮，上面的数字没有参考价值。）")


if __name__ == "__main__":
    main()
