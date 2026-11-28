"""
训练世界模型：序列回放、三项损失与 free nats
============================================

学习目标：
    1. 实现像素版 Pendulum-v1 的动作重复与时间下标对齐：(o_t, a_{t-1}, r_{t-1})
    2. 实现按回合存储、只在回合内部截取片段的序列回放 SequenceReplay（uint8 存图省 4 倍内存）
    3. 实现 world_model_loss()：对像素求和的重建项 + 奖励项 + 带 free nats 的 KL，并监控截断前的真实 KL
    4. 亲手复现"把重建项错写成对像素求平均"造成的后验坍塌，对比健康模型的重建画面
    5. 用线性探针分别从单帧编码 e_t、随机状态 z_t、确定性状态 h_t、模型状态 [h_t, z_t] 读出角度和角速度，
       验证速度信息确实通过序列训练进入了模型状态

核心概念：
    - 序列回放（sequence replay）：随机抽回合、再随机抽连续 T 步片段，绝不跨回合边界
    - 截断的沿时间反向传播（truncated BPTT）：每个片段从 h_0 = 0, z_0 = 0 冷启动，梯度在片段内回传
    - 重建项尺度：对 3×64×64 = 12288 个像素求和，而不是求平均
    - free nats：max(κ, KL)，给每一步 κ nat 的免费信息额度，防止训练初期后验坍塌
    - 线性探针（linear probe）：冻结模型参数，只拟合一层线性回归来检验表示里包含什么物理量

前置知识：
    - 01-filtering-and-belief-state.py（像素版 Pendulum-v1，单帧没有速度）
    - 04-rssm-from-scratch.py（WorldModel、RSSM、gaussian_kl）

对应概念文档：
    - CONCEPT.md 第 4.7–4.9 节（free nats）、第 6.7 节（线性探针）、第 7 节「训练世界模型」

预计运行时间：约 20 分钟（CPU）/ 约 3 分钟（MPS / GPU）
"""

import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

# 复用 04 节从零写好的 WorldModel 和 gaussian_kl，保持单一实现来源
_LESSON_04_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "04-rssm-from-scratch.py")
_spec04 = importlib.util.spec_from_file_location("lesson04", _LESSON_04_PATH)
_lesson04 = importlib.util.module_from_spec(_spec04)
_spec04.loader.exec_module(_lesson04)

WorldModel = _lesson04.WorldModel
gaussian_kl = _lesson04.gaussian_kl
IMAGE_SIZE = _lesson04.IMAGE_SIZE
ACTION_DIM = _lesson04.ACTION_DIM

plt = setup_matplotlib()
OUT = output_dir("phase-03")
SEED = 0
DEVICE = get_device()
SUFFIX = "-quick" if is_quick() else ""
DATASET_PATH = OUT / f"pendulum-pixel-dataset{SUFFIX}.npz"
MODEL_PATH = OUT / f"world_model{SUFFIX}.pt"

MAX_TORQUE = 2.0
ACTION_REPEAT = 2       # 每个决策动作连续执行 2 个环境步（200 步回合 = 100 个决策步）
EPISODE_DECISIONS = 200 // ACTION_REPEAT
FREE_NATS = 3.0         # PlaNet 论文取 κ = 3 nat
BETA = 1.0              # PlaNet 论文不对 KL 缩放


# ==================== 第一部分：导读与手算核对 ====================


def introduction():
    print("=" * 60)
    print("第一部分：数据怎么对齐，损失怎么定尺度")
    print("=" * 60)
    print("""
上一节搭好了编码器、RSSM、解码器和奖励头。这一节把它们放到像素版 Pendulum-v1 上真正训练起来。
动手前有两处最容易写错的工程细节（CONCEPT.md 第 6.4、7.3 节）：

  1. 时间下标的对齐（按"到达某状态之前执行的动作和拿到的奖励"存放）：

       位置 k：      0（回合起点）     1             2            ···     L
       观测         o_1（reset 帧）   o_2           o_3                  o_{L+1}
       之前的动作    a_0 = 0          a_1           a_2                  a_L
       之前的奖励    r_0 = 0          r_1           r_2                  r_L

     模型读入 (a_{t-1}, o_t) 得到 (h_t, z_t)，用它重建 o_t、预测 r_{t-1}。
     截取训练片段时从 k ≥ 1 开始取连续 T 步，这样片段里每一步都是真实发生过的转移。

  2. 三项损失的尺度：
       L = (1/T) Σ_t [ ½ Σ_像素 (ô_t − o_t)²  +  ½ (r̂_{t-1} − r_{t-1})²  +  β · max(κ, KL_t) ]
                      └── 对 12288 个像素求和 ──┘   └──── 标量奖励 ────┘      └── free nats ──┘

     如果把第一项错写成 F.mse_loss 的默认求平均，重建项会缩小 12288 倍，
     优化器会立刻把 KL 压到 0（后验坍塌），解码器对所有帧都输出同一张模糊圆环。
    """)


def verify_hand_calculations():
    """核对 CONCEPT.md 第 4.7 节（free nats 两种截断）和第 7.6 节（三项损失手算）。"""
    print("\n" + "=" * 60)
    print("第二部分：核对 CONCEPT.md 的手算例子")
    print("=" * 60)

    kls = torch.tensor([0.5, 2.0, 4.0, 7.5], requires_grad=True)
    no_free = kls.mean().item()
    step_clamp = kls.clamp_min(FREE_NATS).mean()
    step_clamp.backward()
    step_grads = kls.grad.tolist()

    kls2 = torch.tensor([0.5, 2.0, 4.0, 7.5], requires_grad=True)
    mean_clamp = kls2.mean().clamp_min(FREE_NATS)
    mean_clamp.backward()
    mean_grads = kls2.grad.tolist()

    print(f"  1. free nats 对比（κ = {FREE_NATS:.0f}，四步 KL = [0.5, 2.0, 4.0, 7.5]）：")
    print(f"     不用 free nats：    损失 = {no_free:.3f}")
    print(f"     逐步截断再平均：    损失 = {step_clamp.item():.3f}，各步对 KL 的梯度 = {step_grads}（前 2 步免费，梯度为 0）")
    print(f"     先平均再截断：      损失 = {mean_clamp.item():.3f}，各步对 KL 的梯度 = {mean_grads}（全部 4 步仍被往下压）")
    assert abs(step_clamp.item() - 4.375) < 1e-6 and abs(mean_clamp.item() - 3.5) < 1e-6

    recon = torch.tensor([60.0, 55.0, 52.0])
    rew = torch.tensor([0.02, 0.05, 0.01])
    raw_kl = torch.tensor([1.2, 3.8, 5.0])
    total = (recon + rew + BETA * raw_kl.clamp_min(FREE_NATS)).mean().item()
    print(f"\n  2. 三步片段总损失（CONCEPT.md 第 7.6 节）：(63.02 + 58.85 + 57.01) / 3 = {total:.2f} ✓")
    assert abs(total - 59.63) < 1e-2


# ==================== 第二部分：像素版环境与序列回放 ====================


def render_pendulum(env):
    """渲染当前画面并缩放到 64×64 uint8，隐藏转轴处的力矩箭头。"""
    raw = env.unwrapped
    raw.last_u = None
    frame = raw.render()
    return cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def step_with_repeat(env, norm_action, action_repeat=ACTION_REPEAT):
    """输入归一化动作 a ∈ [-1, 1]，连续执行 R 个环境步，返回 (最后一帧 uint8, 累加奖励, 真实状态)。"""
    torque = np.clip(norm_action, -1.0, 1.0).astype(np.float32) * MAX_TORQUE
    total_reward = 0.0
    for _ in range(action_repeat):
        _, r, _, _, _ = env.step(torque)
        total_reward += float(r)
    return render_pendulum(env), total_reward, env.unwrapped.state.copy().astype(np.float32)


def preprocess_obs(obs_u8):
    """uint8 [..., 64, 64, 3] → float32 [..., 3, 64, 64]，缩放到 [-0.5, 0.5]。"""
    if obs_u8.ndim == 4:
        return obs_u8.permute(0, 3, 1, 2).float() / 255.0 - 0.5
    return obs_u8.permute(0, 1, 4, 2, 3).float() / 255.0 - 0.5


def postprocess_obs(obs_tensor):
    """float32 [..., 3, 64, 64] → float32 [..., 64, 64, 3]，裁剪到 [0, 1] 供画图。"""
    if obs_tensor.ndim == 4:
        return (obs_tensor + 0.5).clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()
    return (obs_tensor + 0.5).clamp(0.0, 1.0).permute(0, 1, 3, 4, 2).cpu().numpy()


def collect_offline_dataset(n_episodes):
    """采集或加载像素版 Pendulum-v1 的随机探索数据集。

    一半回合用独立均匀随机动作，一半用时间平滑的随机动作（让摆能荡起更高的角速度，但完全不看状态）。
    states 只用来做末尾的线性探针诊断，绝不进入 SequenceReplay。
    """
    if DATASET_PATH.exists():
        with np.load(DATASET_PATH) as f:
            return {k: f[k] for k in f.files}

    print(f"  正在采集 {n_episodes} 个回合的像素版 Pendulum-v1 数据（动作重复 R = {ACTION_REPEAT}）……")
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    rng = np.random.default_rng(SEED)
    L = EPISODE_DECISIONS + 1
    obs_all = np.zeros((n_episodes, L, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    act_all = np.zeros((n_episodes, L, ACTION_DIM), dtype=np.float32)
    rew_all = np.zeros((n_episodes, L), dtype=np.float32)
    state_all = np.zeros((n_episodes, L, 2), dtype=np.float32)

    for ep in range(n_episodes):
        env.reset(seed=SEED + ep)
        obs_all[ep, 0] = render_pendulum(env)
        state_all[ep, 0] = env.unwrapped.state.copy()
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        smooth = ep % 2 == 1
        for t in range(1, L):
            if smooth:
                a = np.clip(0.75 * a + 0.65 * rng.standard_normal(ACTION_DIM), -1.0, 1.0).astype(np.float32)
            else:
                a = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
            frame, reward, state = step_with_repeat(env, a)
            obs_all[ep, t] = frame
            act_all[ep, t] = a
            rew_all[ep, t] = reward
            state_all[ep, t] = state
    env.close()

    data = dict(obs=obs_all, actions=act_all, rewards=rew_all, states=state_all)
    np.savez_compressed(DATASET_PATH, **data)
    return data


class SequenceReplay:
    """按回合存储的序列回放缓冲区（CONCEPT.md 第 7.5、7.9 节）。"""

    def __init__(self):
        self.episodes = []

    def add_episode(self, obs_u8, actions, rewards):
        """存入一个完整回合：obs_u8 [L, 64, 64, 3] uint8，actions [L, A]，rewards [L]。"""
        self.episodes.append((
            np.asarray(obs_u8, dtype=np.uint8),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
        ))

    def __len__(self):
        return len(self.episodes)

    def total_steps(self):
        return sum(len(ep[0]) - 1 for ep in self.episodes)

    def memory_mb(self):
        return sum(ep[0].nbytes + ep[1].nbytes + ep[2].nbytes for ep in self.episodes) / (1024**2)

    def sample(self, batch_size, chunk_len, device, generator=None):
        """随机抽 batch_size 个回合，在每个回合内随机截取连续 chunk_len 步（起点 ≥ 1，不跨回合）。"""
        ep_indices = torch.randint(0, len(self.episodes), (batch_size,), generator=generator).tolist()
        obs_b, act_b, rew_b = [], [], []
        for idx in ep_indices:
            obs_ep, act_ep, rew_ep = self.episodes[idx]
            # 从下标 1 开始截取，确保片段里每一步都有真实执行的动作 a_{t-1} 和奖励 r_{t-1}
            max_start = len(obs_ep) - chunk_len
            start = int(torch.randint(1, max_start + 1, (1,), generator=generator).item())
            end = start + chunk_len
            obs_b.append(obs_ep[start:end])
            act_b.append(act_ep[start:end])
            rew_b.append(rew_ep[start:end])

        obs_t = preprocess_obs(torch.from_numpy(np.stack(obs_b)).to(device))  # [B, T, 3, 64, 64]
        act_t = torch.from_numpy(np.stack(act_b)).to(device)                  # [B, T, A]
        rew_t = torch.from_numpy(np.stack(rew_b)).to(device)                  # [B, T]
        return obs_t, act_t, rew_t


# ==================== 第三部分：损失函数与训练 ====================


def world_model_loss(model, obs, actions, rewards, beta=BETA, free_nats=FREE_NATS, pixel_sum=True):
    """计算一个序列批次的世界模型损失（CONCEPT.md 第 7.2、7.7 节）。

    pixel_sum=True（正确）：重建损失对 3×64×64 像素求和；
    pixel_sum=False（常见错误）：对像素求平均，仅用于对照复现后验坍塌。
    """
    embeds = model.encode_seq(obs)
    out = model.rssm.observe(embeds, actions)
    recon = model.decode_seq(out["features"])
    pred_rew = model.reward_head(out["features"])

    sq_err = (recon - obs).pow(2)
    recon_loss = 0.5 * (sq_err.sum(dim=(2, 3, 4)) if pixel_sum else sq_err.mean(dim=(2, 3, 4)))
    reward_loss = 0.5 * (pred_rew - rewards).pow(2)
    raw_kl = gaussian_kl(out["post_mean"], out["post_std"], out["prior_mean"], out["prior_std"])
    kl_loss = raw_kl.clamp_min(free_nats) if free_nats > 0.0 else raw_kl

    total_loss = (recon_loss + reward_loss + beta * kl_loss).mean()
    metrics = dict(
        total=total_loss.item(),
        recon=recon_loss.mean().item(),
        reward=reward_loss.mean().item(),
        raw_kl=raw_kl.mean().item(),
        kl_loss=kl_loss.mean().item(),
        step_kl=raw_kl.detach().mean(dim=0).cpu().numpy(),
    )
    return total_loss, metrics


def train_world_model(replay, n_iters, batch_size, chunk_len, pixel_sum=True, free_nats=FREE_NATS, label="标准模型"):
    torch.manual_seed(SEED)
    model = WorldModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(SEED)
    history = []
    step_kl_acc = np.zeros(chunk_len, dtype=np.float64)
    n_tail = max(1, n_iters // 5)
    t0 = time.time()

    for it in range(1, n_iters + 1):
        obs, act, rew = replay.sample(batch_size, chunk_len, DEVICE, generator=gen)
        loss, m = world_model_loss(model, obs, act, rew, free_nats=free_nats, pixel_sum=pixel_sum)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
        optimizer.step()
        history.append((m["recon"], m["reward"], m["raw_kl"], m["kl_loss"]))
        if it > n_iters - n_tail:
            step_kl_acc += m["step_kl"]
        if it % max(1, n_iters // 5) == 0 or it == 1:
            print(f"    [{label}] 步 {it:>4}/{n_iters}  重建 {m['recon']:7.2f}  奖励 {m['reward']:6.3f}  "
                  f"真实KL {m['raw_kl']:5.2f}（截断后 {m['kl_loss']:5.2f}）  用时 {time.time() - t0:5.1f}s")

    elapsed = time.time() - t0
    return model, np.array(history), step_kl_acc / n_tail, elapsed


@torch.no_grad()
def evaluate_and_compare(good_model, bad_model, history, step_kl, val_data, elapsed_good):
    obs_val = preprocess_obs(torch.from_numpy(val_data["obs"][:, 1:]).to(DEVICE))
    act_val = torch.from_numpy(val_data["actions"][:, 1:]).to(DEVICE)
    rew_val = torch.from_numpy(val_data["rewards"][:, 1:]).to(DEVICE)

    _, m_good = world_model_loss(good_model, obs_val, act_val, rew_val)
    _, m_bad = world_model_loss(bad_model, obs_val, act_val, rew_val, free_nats=0.0, pixel_sum=True)

    out_good = good_model.rssm.observe(good_model.encode_seq(obs_val), act_val, sample=False)
    recon_good = good_model.decode_seq(out_good["features"])
    pred_rew_good = good_model.reward_head(out_good["features"])
    rew_rmse = (pred_rew_good - rew_val).pow(2).mean().sqrt().item()
    pix_rmse = (recon_good - obs_val).pow(2).mean().sqrt().item()

    out_bad = bad_model.rssm.observe(bad_model.encode_seq(obs_val[:1]), act_val[:1], sample=False)
    recon_bad = bad_model.decode_seq(out_bad["features"])

    print(f"""
验证集评估（{len(obs_val)} 个未见回合）：
    标准模型（像素求和 + κ=3）：重建损失 {m_good['recon']:.2f}（单像素 RMSE {pix_rmse:.4f}），
                                奖励损失 {m_good['reward']:.3f}（奖励 RMSE {rew_rmse:.2f}，真实范围约 [-32, 0]），
                                截断前真实 KL = {m_good['raw_kl']:.2f} nat
    错误模型（像素求平均，κ=0）：按同口径计算的重建损失 {m_bad['recon']:.2f}，真实 KL = {m_bad['raw_kl']:.4f} nat（视觉重建坍塌，仅标量奖励项维持约 1 nat KL！）
    片段内第 1 步 vs 最后一步的平均真实 KL：{step_kl[0]:.2f} nat → {step_kl[-1]:.2f} nat
    （验证 CONCEPT.md 第 7.5 节：片段从零状态冷启动，开头几步先验还没热身，KL 天然更高）
    """)

    # 图 1：训练曲线（4 面板：重建、奖励、截断前后的 KL、片段内各位置的平均 KL）
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    win = max(1, len(history) // 40)
    smooth = lambda x: np.convolve(x, np.ones(win) / win, mode="valid")
    xs = np.arange(win, len(history) + 1)
    axes[0].plot(xs, smooth(history[:, 0]), linewidth=2)
    axes[0].set_title("重建损失（对像素求和）")
    axes[0].set_xlabel("训练步数")

    axes[1].plot(xs, smooth(history[:, 1]), linewidth=2, color="C2")
    axes[1].set_title("奖励损失 ½(r̂ − r)²")
    axes[1].set_xlabel("训练步数")

    axes[2].plot(xs, smooth(history[:, 2]), linewidth=2, color="C1", label="截断前的真实 KL")
    axes[2].plot(xs, smooth(history[:, 3]), linewidth=1.5, linestyle="--", color="C3", label="截断后的 KL 损失")
    axes[2].axhline(FREE_NATS, color="gray", linestyle=":", label=f"free nats κ = {FREE_NATS:.0f}")
    axes[2].set_title("KL 散度（nat / 步）")
    axes[2].set_xlabel("训练步数")
    axes[2].legend(fontsize=8)

    axes[3].plot(np.arange(1, len(step_kl) + 1), step_kl, "o-", linewidth=1.8, markersize=4, color="C4")
    axes[3].set_title("片段内各步的平均真实 KL")
    axes[3].set_xlabel("片段内的步下标")
    axes[3].set_ylabel("KL（nat）")
    fig.suptitle(f"像素版 Pendulum-v1 世界模型训练诊断（训练耗时 {elapsed_good:.0f}s，{DEVICE}）", fontsize=11)
    save_figure(fig, OUT / "05-training-curves.png")

    # 图 2：原图 vs 标准模型重建 vs 后验坍塌模型重建
    pick = np.linspace(0, obs_val.shape[1] - 1, 6, dtype=int)
    orig_imgs = postprocess_obs(obs_val[0, pick])
    good_imgs = postprocess_obs(recon_good[0, pick])
    bad_imgs = postprocess_obs(recon_bad[0, pick])

    fig, axes = plt.subplots(3, 6, figsize=(11.5, 5.6))
    row_titles = ["真实画面", "像素求和 + κ=3\n（健康）", "像素求平均 + κ=0\n（后验坍塌）"]
    for col in range(6):
        for row, imgs in enumerate((orig_imgs, good_imgs, bad_imgs)):
            axes[row, col].imshow(imgs[col])
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            axes[row, col].grid(False)
        axes[0, col].set_title(f"t = {pick[col] + 1}", fontsize=9)
    for row, title in enumerate(row_titles):
        axes[row, 0].set_ylabel(title, fontsize=9.5)
    fig.suptitle("对像素求和 vs 求平均：相差 12288 倍的重建权重直接决定是否坍塌", fontsize=11)
    save_figure(fig, OUT / "05-posterior-collapse.png")

    return dict(val_recon=m_good["recon"], pix_rmse=pix_rmse, rew_rmse=rew_rmse,
                val_kl=m_good["raw_kl"], bad_kl=m_bad["raw_kl"], bad_recon=m_bad["recon"])


# ==================== 第四部分：线性探针读角速度 ====================


def fit_ridge_probe(train_x, train_y, test_x, reg=1e-2):
    """闭式解拟合带 L2 正则的线性探针 W：min ‖X W − Y‖² + reg ‖W‖²。"""
    ones_tr = torch.ones(train_x.shape[0], 1)
    ones_te = torch.ones(test_x.shape[0], 1)
    x_tr = torch.cat([train_x, ones_tr], dim=1)
    x_te = torch.cat([test_x, ones_te], dim=1)
    eye = torch.eye(x_tr.shape[1]) * reg
    eye[-1, -1] = 0.0  # 偏置项不加正则
    weights = torch.linalg.solve(x_tr.T @ x_tr + eye, x_tr.T @ train_y)
    return x_te @ weights


@torch.no_grad()
def linear_probe_experiment(model, val_data):
    """CONCEPT.md 第 6.7 节待回填实验：从冻结的 e_t、z_t、h_t、[h_t, z_t] 线性读取角度与角速度。"""
    print("\n" + "=" * 60)
    print("第四部分：线性探针——速度信息进到哪个向量里了？")
    print("=" * 60)

    obs_val = preprocess_obs(torch.from_numpy(val_data["obs"][:, 1:]).to(DEVICE))
    act_val = torch.from_numpy(val_data["actions"][:, 1:]).to(DEVICE)
    states = torch.from_numpy(val_data["states"][:, 1:])  # [E, L, 2]：(θ, θ̇)

    embeds = model.encode_seq(obs_val)
    out = model.rssm.observe(embeds, act_val, sample=False)

    # 丢掉每个回合开头 3 步冷启动期，按回合划分探针的训练集和测试集（避免相邻帧泄漏）
    burn = 3
    n_ep = obs_val.shape[0]
    n_probe_train = max(1, int(0.7 * n_ep))
    feats_dict = {
        "单帧编码 e_t [1024]": embeds[:, burn:].cpu(),
        "随机状态 z_t [30]": out["post_mean"][:, burn:].cpu(),
        "确定性状态 h_t [200]": out["h"][:, burn:].cpu(),
        "模型状态 [h_t, z_t] [230]": torch.cat([out["h"], out["post_mean"]], dim=-1)[:, burn:].cpu(),
    }
    st = states[:, burn:]
    targets = torch.stack([torch.cos(st[..., 0]), torch.sin(st[..., 0]), st[..., 1]], dim=-1)  # [E, L', 3]
    y_tr = targets[:n_probe_train].flatten(0, 1)
    y_te = targets[n_probe_train:].flatten(0, 1)
    baseline_vel_rmse = (y_te[:, 2] - y_tr[:, 2].mean()).pow(2).mean().sqrt().item()

    print(f"\n  在冻结的世界模型上训练线性探针（{n_probe_train} 个回合拟合，{n_ep - n_probe_train} 个独立回合测试）：")
    print(f"  {'表示向量':<26}{'角度中位误差（度）':>18}{'角速度 RMSE（rad/s）':>22}")
    probe_results = {}
    for name, feat in feats_dict.items():
        pred = fit_ridge_probe(feat[:n_probe_train].flatten(0, 1), y_tr, feat[n_probe_train:].flatten(0, 1))
        ang_diff = torch.atan2(pred[:, 1], pred[:, 0]) - torch.atan2(y_te[:, 1], y_te[:, 0])
        ang_err = torch.rad2deg(torch.atan2(torch.sin(ang_diff), torch.cos(ang_diff)).abs()).median().item()
        vel_rmse = (pred[:, 2] - y_te[:, 2]).pow(2).mean().sqrt().item()
        probe_results[name] = dict(ang_err=ang_err, vel_rmse=vel_rmse, pred_vel=pred[:, 2].numpy())
        print(f"  {name:<26}{ang_err:>18.2f}{vel_rmse:>22.2f}")
    print(f"  {'基线：永远猜平均角速度':<24}{'—':>16}{baseline_vel_rmse:>22.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    true_vel = y_te[:, 2].numpy()
    for ax, key in zip(axes, ["单帧编码 e_t [1024]", "模型状态 [h_t, z_t] [230]"]):
        r = probe_results[key]
        ax.scatter(true_vel, r["pred_vel"], s=10, alpha=0.5)
        ax.plot([-8, 8], [-8, 8], "k--", linewidth=1, label="理想 y = x")
        ax.set_xlabel("真实角速度 θ̇（rad/s）")
        ax.set_ylabel("线性探针读出的角速度（rad/s）")
        ax.set_title(f"{key}（RMSE = {r['vel_rmse']:.2f}）")
        ax.legend(fontsize=9)
    save_figure(fig, OUT / "05-velocity-probe.png")

    e_rmse = probe_results["单帧编码 e_t [1024]"]["vel_rmse"]
    m_rmse = probe_results["模型状态 [h_t, z_t] [230]"]["vel_rmse"]
    print(f"""
怎么读这张表：
    1. 单帧编码 e_t 只看当前一张 64×64 图：读角度很准（中位误差 {probe_results['单帧编码 e_t [1024]']['ang_err']:.1f}°），
       读角速度的 RMSE 是 {e_rmse:.2f} rad/s，和闭着眼猜平均值（{baseline_vel_rmse:.2f}）没有本质区别——与 01 节 k 近邻的结论一致。
    2. 经过 RSSM 滤波之后，模型状态 [h_t, z_t] 的角速度 RMSE 降到 {m_rmse:.2f} rad/s，
       没有人在损失里提供过真实角速度标签，纯粹靠"预测未来画面与奖励"就把速度提炼并存进了状态！
    """)
    return probe_results, baseline_vel_rmse


# ==================== 第五部分：练习与思考 ====================


def exercises():
    print("\n" + "=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：为什么日志里一定要画"截断前"的真实 KL，不能只画截断后的 kl_loss？

练习 1 参考答案：
    kl_loss = max(3.0, raw_kl) 最小就是 3.0。如果模型已经发生后验坍塌（真实 KL = 0.001），
    只看 kl_loss 会看到一条稳稳停在 3.0 的水平线，误以为模型很健康。

练习 2：如果 SequenceReplay.sample() 把所有回合拼成一条长数组再随机切片，会发生什么？

练习 2 参考答案：
    跨过回合边界的片段里，上一帧摆还在正上方，下一帧突然跳到 reset 后的随机位置，
    而且中间对应的动作并不是造成这次跳变的原因。先验会被逼着放大方差来容纳这种"假瞬移"。

练习 3：h_t 能读出一定的角速度，为什么角度误差比 z_t 大？

练习 3 参考答案：
    h_t 是看 o_t 之前的预测状态，它只知道上一步的位置和刚执行的动作；
    z_t 融合了当前帧的编码 e_t，所以当前帧的精确角度主要修正到了 z_t 里，两者拼在一起最完整。

思考题 1：本节训练好的模型重建图已经很清晰，能直接断定它一定适合做 15 步 CEM 规划吗？

思考题 1 参考答案：
    不能！本节的重建图用的是后验 z_t（刚看过真图 o_t）。规划时连续 15 步都不看图、只走先验，
    先验多步展开到底准不准，必须用下一节的"开环预测"来检验。

思考题 2：uint8 存 50 个回合（5050 帧 64×64×3）占多少内存？存 float32 呢？到 1000 个回合时差多少？

思考题 2 参考答案：
    单帧 uint8 占 12,288 字节（12 KB），5050 帧约 59 MB；float32 是 4 倍，约 237 MB。
    到 1000 个回合（约 10 万帧）时，uint8 约 1.2 GB，float32 接近 4.8 GB，在笔记本上很容易撑爆内存。
    """)


# ==================== 主函数 ====================


def main():
    seed_everything(SEED)
    if DEVICE.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    quick = is_quick()
    if quick:
        print("（冒烟模式：回合数和训练步数大幅缩小，仅验证流程跑通）\n")

    n_episodes, n_val = (8, 2) if quick else (60, 10)
    n_iters, n_iters_bad, batch_size, chunk_len = (20, 12, 4, 12) if quick else (500, 150, 12, 25)

    introduction()
    verify_hand_calculations()

    print("\n" + "=" * 60)
    print("第三部分：构建序列回放并训练世界模型")
    print("=" * 60)
    data = collect_offline_dataset(n_episodes)
    n_train = n_episodes - n_val
    replay = SequenceReplay()
    for ep in range(n_train):
        replay.add_episode(data["obs"][ep], data["actions"][ep], data["rewards"][ep])
    val_data = {k: data[k][n_train:] for k in ("obs", "actions", "rewards", "states")}
    print(f"  训练集：{len(replay)} 个回合（{replay.total_steps()} 个决策步），缓冲区占用 {replay.memory_mb():.1f} MB（uint8 存图）")
    print(f"  验证集：{n_val} 个独立回合；训练设备：{DEVICE}，批大小 B = {batch_size}，片段长度 T = {chunk_len}\n")

    good_model, history, step_kl, elapsed = train_world_model(
        replay, n_iters, batch_size, chunk_len, pixel_sum=True, free_nats=FREE_NATS, label="像素求和+κ=3"
    )
    torch.save(good_model.state_dict(), MODEL_PATH)
    print(f"  💾 世界模型已保存：{MODEL_PATH.relative_to(OUT.parents[1])}\n")

    bad_model, _, _, _ = train_world_model(
        replay, n_iters_bad, batch_size, chunk_len, pixel_sum=False, free_nats=0.0, label="像素平均+κ=0"
    )

    evaluate_and_compare(good_model, bad_model, history, step_kl, val_data, elapsed)
    linear_probe_experiment(good_model, val_data)
    exercises()

    print("\n" + "=" * 60)
    print("本节完成！下一节：06-open-loop-prediction-and-overshooting.py（开环预测与多步 KL）")
    print("=" * 60)


if __name__ == "__main__":
    main()
