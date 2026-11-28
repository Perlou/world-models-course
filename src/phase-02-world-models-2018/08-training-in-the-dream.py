"""
在梦里训练控制器
================

学习目标：
    1. 用代码验证"对带误差的估计取最大值"本身就会造成分数虚高（CONCEPT.md 第 8.3 节）
    2. 给 M 加上奖励头（MSE）和终止头（加权 BCE），把 M 包装成批量化的梦境环境 DreamEnv
    3. 把整个种群 N × R 个回合堆成一个张量批次，完全在梦里用 CMA-ES 训练控制器
    4. 对比不同温度 τ ∈ {0.1, 0.5, 1.0, 1.3} 下梦里分数与真实分数：亲眼看到"钻空子"与温度的缓解作用
    5. 把低温下钻空子的梦境解码成视频和关键帧，看清控制器在梦里利用了什么漏洞

核心概念：
    - 梦境环境（DreamEnv）：M 的自回归展开加上奖励头与终止头，接口与真实环境对齐
    - 批量想象（batched rollout）：N × R 个梦境回合放在同一个张量里并行推进，比真实环境快得多
    - 钻空子（exploiting the world model）：优化器把模型的预测漏洞放大成虚高的梦境分数
    - 温度调节：提高 τ 让梦更随机、不可靠，逼迫控制器学不依赖模型细节的稳健策略

前置知识：
    - 05-dreaming-in-latent-space.py（带温度的 MDN 采样与自回归做梦）
    - 06-evolution-strategies-cma-es.py（CMA-ES 优化器）
    - 07-controller-in-real-env.py（线性控制器与真实环境评估）

对应概念文档：
    - CONCEPT.md 第 8 节「在梦里训练：把世界模型当环境」

预计运行时间：约 2–3 分钟（CPU/MPS）
"""

import math
import os
import subprocess
import sys
import time

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
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
ZOOM_FRAMES, CROP_ROWS, IMAGE_SIZE = 50, 84, 64
DEVICE = get_device()
SUFFIX = "-quick" if is_quick() else ""
LATENTS_PATH = OUT / f"latents{SUFFIX}.npz"
VAE_PATH = OUT / f"vae{SUFFIX}.pt"
MDNRNN_PATH = OUT / f"mdnrnn{SUFFIX}.pt"
WORLD_MODEL_PATH = OUT / f"mdnrnn-dream{SUFFIX}.pt"


# ==================== 第一部分：把 M 包装成环境 ====================


def introduction():
    print("=" * 60)
    print("第一部分：把 M 包装成环境，完全在梦里训练")
    print("=" * 60)
    print("""
上一节在真实环境里训练控制器，每评估一个候选就要跑真实的物理仿真和像素渲染。
现在把整件事搬进 M 生成的梦里：

    真实环境训练（07 节）                    梦里训练（本节）
    ┌──────────┐ o_t  ┌───┐                 ┌───────────────────────┐
    │ 真实环境  │ ───▶ │ V │                 │     M（当环境用）       │
    └──────────┘      └───┘                 │  ẑ_{t+1} ~ P_τ(z|…)    │
         ▲              │ z_t               │  r̂_t = reward_head(h) │
         │ a_t          ▼                   │  d̂_t = done_head(h)   │
         │        ┌──────────┐              └───────────────────────┘
         └─────── │    C     │                    ▲            │
                  └──────────┘                a_t │            │ ẑ_t, h_t, r̂_t, d̂_t
                        ▲                         │            ▼
                        │ h_t               ┌───────────────────────┐
                     ┌─────┐                │   C（批量：N × R）     │
                     │  M  │                └───────────────────────┘
                     └─────┘

好处：全程没有画面、没有物理引擎、没有卷积网络，整个种群 N × R 个回合可以拼成一个张量同时推进。
风险：M 只在随机策略的数据上训过，而且误差不是随机噪声——控制器会专挑让 M 虚高打分的漏洞走。
    """)


# ==================== 第二部分：为什么最大值本身就会虚高 ====================


def check_selection_bias():
    print("=" * 60)
    print("第二部分：对带误差的估计取最大值，为什么一定会高估")
    print("=" * 60)
    print("""
手算验证（CONCEPT.md 第 8.3 节）：假设所有 N 个候选的真实水平都是 0，
梦境模型的误差 e_i 独立取 +1 或 -1（各一半，均值 0，无偏）。
此时 E[max_i e_i] = 1 - 2^(1 - N)：

    候选数 N     公式 1 - 2^(1-N)     十万次模拟平均""")
    rng = np.random.default_rng(SEED)
    for n in (2, 4, 10, 64):
        formula = 1.0 - 2.0 ** (1 - n)
        sim = float(rng.choice([-1.0, 1.0], size=(100_000, n)).max(axis=1).mean())
        print(f"       {n:>2}           {formula:.3f}                 {sim:.3f}")
        assert abs(sim - formula) < 0.015
    print("""
即使模型完全无偏，只要在 N 个带误差的估计里挑最大值，挑出来的就系统性偏高。
更糟的是：模型的误差不是独立噪声，而是参数 φ 的确定性函数——CMA-ES 会顺着漏洞越挖越深。
    """)


# ==================== 模型与组件 ====================


class ConvVAE(nn.Module):
    """02 节的卷积 VAE（用于把真实画面编码成 z，以及把漏洞梦境解码出来看）。"""

    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(1024, z_dim)
        self.fc_logvar = nn.Linear(1024, z_dim)
        self.fc_decode = nn.Linear(z_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid(),
        )

    def encode(self, obs):
        hidden = self.encoder(obs)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def decode(self, z):
        return self.decoder(self.fc_decode(z).view(-1, 1024, 1, 1))


class MDNRNNWithHeads(nn.Module):
    """在 04 节 MDNRNN 的基础上增加奖励头和终止头。"""

    def __init__(self, z_dim=Z_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, n_components=N_COMPONENTS):
        super().__init__()
        self.z_dim, self.n_components = z_dim, n_components
        self.lstm = nn.LSTM(z_dim + action_dim, hidden_dim, batch_first=True)
        self.mdn_head = nn.Linear(hidden_dim, z_dim * n_components * 3)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)

    def _heads(self, hidden):
        out = self.mdn_head(hidden).view(*hidden.shape[:-1], self.z_dim, self.n_components, 3)
        logits, mu, log_sigma = out.unbind(dim=-1)
        reward_pred = self.reward_head(hidden).squeeze(-1)
        done_logit = self.done_head(hidden).squeeze(-1)
        return (torch.log_softmax(logits, dim=-1), mu, log_sigma), reward_pred, done_logit

    def forward(self, z, a, state=None):
        """整段序列 [B, T, ...] → (MDN 参数, 奖励预测 [B, T], 终止 logit [B, T]), state。"""
        hidden, state = self.lstm(torch.cat([z, a], dim=-1), state)
        return self._heads(hidden), state

    def step(self, z, a, state=None):
        """单步推进：z [B, 32], a [B, 3] → (MDN 参数 [B, 32, 5], r̂ [B], d̂_logit [B]), state。"""
        (mdn, r_hat, d_logit), state = self.forward(z.unsqueeze(1), a.unsqueeze(1), state)
        return (tuple(x.squeeze(1) for x in mdn), r_hat.squeeze(1), d_logit.squeeze(1)), state


class SimplifiedCMAES:
    """06 节的简化版 CMA-ES。"""

    def __init__(self, dim, sigma, pop_size, seed=SEED):
        self.dim = int(dim)
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.sigma = float(sigma)
        self.pop_size = int(pop_size)
        n_elite = self.pop_size // 2
        raw = np.log((self.pop_size + 1) / 2) - np.log(np.arange(1, n_elite + 1))
        self.weights = raw / raw.sum()
        self.n_elite = n_elite
        self.mu_eff = float(1.0 / np.sum(self.weights ** 2))

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
        eps = self.rng.standard_normal((self.pop_size, self.dim))
        steps = (eps * self._D) @ self._B.T
        return self.mean[None, :] + self.sigma * steps

    def tell(self, candidates, fitness):
        order = np.argsort(fitness)[::-1]
        steps = (candidates - self.mean[None, :]) / max(self.sigma, 1e-20)
        elite_steps = steps[order[:self.n_elite]]
        d_w = self.weights @ elite_steps
        self.mean = self.mean + self.sigma * d_w

        c_s = self.c_sigma
        self.path_sigma = (1.0 - c_s) * self.path_sigma + math.sqrt(c_s * (2.0 - c_s) * self.mu_eff) * (
            self._inv_sqrt_C @ d_w
        )
        self.sigma *= math.exp((c_s / self.d_sigma) * (np.linalg.norm(self.path_sigma) / self.chi_n - 1.0))

        c_c = self.c_c
        self.path_c = (1.0 - c_c) * self.path_c + math.sqrt(c_c * (2.0 - c_c) * self.mu_eff) * d_w
        c_mu_mat = (elite_steps.T * self.weights) @ elite_steps
        self.cov = (
            (1.0 - self.c_1 - self.c_mu) * self.cov
            + self.c_1 * np.outer(self.path_c, self.path_c)
            + self.c_mu * c_mu_mat
        )
        self._update_eigensystem()


def ensure_artifact(path, lesson):
    if path.exists():
        return
    print(f"⚠️  没找到 {path.name}，先运行 {lesson} 生成它……")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), lesson)
    subprocess.run([sys.executable, script], check=True, stdout=subprocess.DEVNULL)


# ==================== 第三部分：训练带奖励头和终止头的 M ====================


def world_model_loss(outputs, z_next, reward, done, valid_trans, valid_step, pos_weight):
    """MDN 负对数似然 + 奖励均方误差 + 加权终止二分类交叉熵。

    注意对齐细节：
        一个长度为 n 的回合（索引 0 … n-1），第 n-1 步（冲出边界）执行后没有下一帧 z_n，
        但有 r_{n-1}（-100）和 done_{n-1}（True）！
        因此 MDN 损失只在 valid_trans（0 … n-2）上算，奖励和终止损失在全部 valid_step（0 … n-1）上算。
    """
    (log_pi, mu, log_sigma), reward_pred, done_logit = outputs

    # 1. MDN 损失（只看有下一帧的转移 0 .. T-2）
    lp = log_pi[:, :-1]
    m = mu[:, :-1]
    ls = log_sigma[:, :-1].clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)
    y = z_next.unsqueeze(-1)
    log_normal = -HALF_LOG_2PI - ls - 0.5 * ((y - m) / ls.exp()) ** 2
    mdn_loss = (-torch.logsumexp(lp + log_normal, dim=-1))[valid_trans].mean()

    # 2. 奖励损失（MSE）
    reward_loss = ((reward_pred[valid_step] - reward[valid_step]) ** 2).mean()

    # 3. 终止损失（正样本极少，给正样本加权避免全猜 0）
    bce = nn.functional.binary_cross_entropy_with_logits(
        done_logit[valid_step], done[valid_step].float(), pos_weight=pos_weight
    )
    return mdn_loss + reward_loss + bce, (mdn_loss.item(), reward_loss.item(), bce.item())


def prepare_world_model(epochs):
    print("=" * 60)
    print("第三部分：训练带奖励头和终止头的 M")
    print("=" * 60)

    ensure_artifact(LATENTS_PATH, "02-conv-vae-vision.py")
    ensure_artifact(MDNRNN_PATH, "04-mdn-rnn-memory.py")
    with np.load(LATENTS_PATH) as f:
        raw = {k: f[k] for k in f.files}

    n_episodes, horizon = raw["mu"].shape[:2]
    lengths = torch.from_numpy(raw["lengths"])
    valid_step = torch.arange(horizon)[None, :] < lengths[:, None]
    valid_trans = torch.arange(horizon - 1)[None, :] < (lengths[:, None] - 1)
    n_val = max(1, n_episodes // 10)
    train_idx = torch.arange(n_episodes - n_val)
    val_idx = torch.arange(n_episodes - n_val, n_episodes)

    data = {
        "mu": torch.from_numpy(raw["mu"]).to(DEVICE),
        "logvar": torch.from_numpy(raw["logvar"]).to(DEVICE),
        "actions": torch.from_numpy(raw["actions"]).to(DEVICE),
        "rewards": torch.from_numpy(raw["rewards"]).to(DEVICE),
        "dones": torch.from_numpy(raw["dones"]).to(DEVICE),
        "valid_step": valid_step.to(DEVICE),
        "valid_trans": valid_trans.to(DEVICE),
        "lengths": lengths,
        "train": train_idx,
        "val": val_idx,
    }

    n_done = int(data["dones"][train_idx][data["valid_step"][train_idx]].sum())
    n_total = int(data["valid_step"][train_idx].sum())
    # pos_weight 设为 100.0：足够防止"永远不结束"，又不会让正常帧的假阳性太高
    pos_weight = torch.tensor(100.0, device=DEVICE)
    print(f"""
训练集共 {n_total} 个时间步，其中回合终止（done = True）只有 {n_done} 步（占 {n_done / n_total * 100:.3f}%）。
为避免终止头学成"永远输出 False"，给正样本加权 pos_weight = {pos_weight.item():.0f}。
基于 04 节已经训好的 MDNRNN 权重继续联合微调奖励头和终止头（{epochs} 轮）。
    """)

    torch.manual_seed(SEED)
    model = MDNRNNWithHeads().to(DEVICE)
    base_state = torch.load(MDNRNN_PATH, map_location=DEVICE)
    model.load_state_dict(base_state, strict=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(SEED)

    start = time.time()
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx), generator=generator)]
        for i in range(0, len(perm), 32):
            idx = perm[i:i + 32].to(DEVICE)
            eps = torch.randn(data["mu"][idx].shape, generator=generator).to(DEVICE)
            z = data["mu"][idx] + torch.exp(0.5 * data["logvar"][idx]) * eps
            outputs, _ = model(z, data["actions"][idx])
            loss, _ = world_model_loss(
                outputs, z[:, 1:], data["rewards"][idx], data["dones"][idx],
                data["valid_trans"][idx], data["valid_step"][idx], pos_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            va = val_idx.to(DEVICE)
            outputs, _ = model(data["mu"][va], data["actions"][va])
            val_loss, (v_mdn, v_rew, v_bce) = world_model_loss(
                outputs, data["mu"][va][:, 1:], data["rewards"][va], data["dones"][va],
                data["valid_trans"][va], data["valid_step"][va], pos_weight,
            )
            if val_loss.item() < best_val:
                best_val = val_loss.item()
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"    第 {epoch + 1:>2} 轮  验证总损失 {val_loss.item():.3f}"
                  f"（MDN {v_mdn:.3f} | 奖励MSE {v_rew:.3f} | 终止BCE {v_bce:.3f}）")

    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), WORLD_MODEL_PATH)

    # 检查奖励头与终止头在验证集上的直观表现
    with torch.no_grad():
        va = val_idx.to(DEVICE)
        (_, _, _), r_pred, d_logit = model(data["mu"][va], data["actions"][va])[0]
        vs = data["valid_step"][va]
        r_true = data["rewards"][va][vs]
        r_hat = r_pred[vs]
        d_true = data["dones"][va][vs]
        d_prob = torch.sigmoid(d_logit[vs])
        r_rmse = float(((r_hat - r_true) ** 2).mean().sqrt())
        d_pos_mean = float(d_prob[d_true].mean()) if d_true.any() else float("nan")
        d_neg_mean = float(d_prob[~d_true].mean())
    print(f"""
训练用时 {time.time() - start:.1f}s。验证集核对：
    奖励预测 RMSE：{r_rmse:.2f}（真实奖励均值 {r_true.mean().item():+.3f}，标准差 {r_true.std().item():.2f}）
    终止概率均值：真实终止步 {d_pos_mean:.3f} vs 非终止步 {d_neg_mean:.4f}
💾 带头世界模型已保存：{WORLD_MODEL_PATH.relative_to(OUT.parents[1])}
    """)
    return model, data


# ==================== 第四部分：批量梦境环境 DreamEnv ====================


def sample_mdn_torch(log_pi, mu, log_sigma, temperature, generator):
    """带温度的 MDN 采样（支持在 CPU / MPS / CUDA 上直接按张量形状批量采样）。"""
    probs = torch.softmax(log_pi / temperature, dim=-1).cpu()
    k = torch.multinomial(probs.view(-1, probs.shape[-1]), 1, generator=generator)
    k = k.view(*probs.shape[:-1], 1).to(mu.device)
    mu_k = mu.gather(-1, k).squeeze(-1)
    sigma_k = log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX).exp().gather(-1, k).squeeze(-1)
    eps = torch.randn(mu_k.shape, generator=generator).to(mu.device)
    return mu_k + math.sqrt(temperature) * sigma_k * eps


class DreamEnv:
    """把带奖励头和终止头的 M 包装成可批量推进的梦境环境。"""

    def __init__(self, model, init_z_pool, temperature=1.0, done_threshold=0.5, seed=SEED):
        self.model = model
        self.init_z_pool = init_z_pool                              # [E, 32] 数据集里各回合的第 0 帧 μ
        self.temperature = float(temperature)
        self.done_threshold = float(done_threshold)
        self.generator = torch.Generator().manual_seed(seed)
        self.z = None
        self.state = None

    @torch.no_grad()
    def reset(self, batch_size):
        idx = torch.randint(0, len(self.init_z_pool), (batch_size,), generator=self.generator).to(DEVICE)
        self.z = self.init_z_pool[idx]
        h0 = torch.zeros(1, batch_size, HIDDEN_DIM, device=DEVICE)
        c0 = torch.zeros(1, batch_size, HIDDEN_DIM, device=DEVICE)
        self.state = (h0, c0)
        return self.z, h0.squeeze(0)

    @torch.no_grad()
    def step(self, actions):
        (mdn, r_hat, d_logit), self.state = self.model.step(self.z, actions, self.state)
        self.z = sample_mdn_torch(*mdn, self.temperature, self.generator)
        h = self.state[0].squeeze(0)
        done = torch.sigmoid(d_logit) > self.done_threshold
        return self.z, h, r_hat, done


@torch.no_grad()
def rollout_dream(dream_env, params_batch, n_rollouts, horizon, record_first=False):
    """整个种群 N 个候选 × R 个回合同时在梦里展开，返回每个候选的平均回报 [N]。"""
    n_cands = len(params_batch)
    in_dim = Z_DIM + HIDDEN_DIM
    w_size = ACTION_DIM * in_dim
    params_t = torch.from_numpy(np.asarray(params_batch, dtype=np.float32)).to(DEVICE)
    w_c = params_t[:, :w_size].view(n_cands, ACTION_DIM, in_dim)    # [N, 3, 288]
    b_c = params_t[:, w_size:]                                      # [N, 3]

    z, h = dream_env.reset(n_cands * n_rollouts)
    total_r = torch.zeros(n_cands * n_rollouts, device=DEVICE)
    alive = torch.ones(n_cands * n_rollouts, dtype=torch.bool, device=DEVICE)
    traj_z, traj_r = ([z[0].clone()] if record_first else None), ([] if record_first else None)

    for _ in range(horizon):
        feat = torch.cat([z, h], dim=-1).view(n_cands, n_rollouts, in_dim)
        raw = torch.einsum("noi,nri->nro", w_c, feat) + b_c[:, None, :]
        steer = torch.tanh(raw[..., 0])
        gas = 0.5 * (torch.tanh(raw[..., 1]) + 1.0)
        brake = torch.tanh(raw[..., 2]).clamp(0.0, 1.0)
        actions = torch.stack([steer, gas, brake], dim=-1).view(n_cands * n_rollouts, ACTION_DIM)

        z, h, r_hat, done = dream_env.step(actions)
        total_r += r_hat * alive.float()
        if record_first:
            traj_z.append(z[0].clone())
            traj_r.append(float(r_hat[0]))
        alive = alive & (~done)
        if not alive.any():
            break

    fitness = total_r.view(n_cands, n_rollouts).mean(dim=1).cpu().numpy()
    if record_first:
        return fitness, torch.stack(traj_z, dim=0), traj_r
    return fitness


# ==================== 第五部分：不同温度下的梦里训练与真实对比 ====================


def preprocess(frame):
    small = cv2.resize(frame[:CROP_ROWS], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    arr = np.transpose(small, (2, 0, 1)).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def eval_in_real_env(vae, model, flat_params, seeds, horizon):
    """把在梦里学到的控制器放回真实 CarRacing-v3，跑一组未见过的赛道（先跳过 50 帧镜头动画）。"""
    in_dim = Z_DIM + HIDDEN_DIM
    w_size = ACTION_DIM * in_dim
    w = flat_params[:w_size].reshape(ACTION_DIM, in_dim).astype(np.float32)
    b = flat_params[w_size:].astype(np.float32)
    env = gym.make("CarRacing-v3", continuous=True)
    returns = []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        total_r = 0.0
        for _ in range(ZOOM_FRAMES):
            obs, r, term, trunc, _ = env.step(np.zeros(3, dtype=np.float32))
            total_r += float(r)
            if term or trunc:
                break
        state = (torch.zeros(1, 1, HIDDEN_DIM, device=DEVICE), torch.zeros(1, 1, HIDDEN_DIM, device=DEVICE))
        for _ in range(horizon):
            mu, _ = vae.encode(preprocess(obs))
            feat = np.concatenate([mu.squeeze(0).cpu().numpy(), state[0].squeeze(0).squeeze(0).cpu().numpy()])
            raw = w @ feat + b
            action = np.array([
                math.tanh(float(raw[0])),
                0.5 * (math.tanh(float(raw[1])) + 1.0),
                min(1.0, max(0.0, math.tanh(float(raw[2])))),
            ], dtype=np.float32)
            obs, r, term, trunc, _ = env.step(action)
            total_r += float(r)
            _, state = model.lstm(torch.cat([mu, torch.from_numpy(action).unsqueeze(0).to(DEVICE)], dim=-1).unsqueeze(1), state)
            if term or trunc:
                break
        returns.append(total_r)
    env.close()
    return np.array(returns, dtype=np.float64)


def compare_dream_and_real(vae, model, init_z_pool, params, tau, dream_horizon, real_horizon, real_seeds):
    """同一个控制器在梦里（给定 τ）和在真实环境里分别评估，返回两者分数。"""
    dream_env = DreamEnv(model, init_z_pool, temperature=tau, seed=SEED + 777)
    dream_scores = rollout_dream(dream_env, params[None, :], n_rollouts=16, horizon=dream_horizon)
    real_scores = eval_in_real_env(vae, model, params, real_seeds, horizon=real_horizon)
    return float(dream_scores[0]), real_scores


def train_controller_in_dream(model, vae, init_z_pool, tau, n_gens, pop_size, n_rollouts, horizon, real_seeds):
    """在指定温度 τ 的梦境里用 CMA-ES 训练控制器，定期回真实环境检验。

    注意：和 07 节不同，这里返回最终在梦里训完的均值参数 es.mean（而不是按真实分数挑出来的早期快照），
    以便直接对比"在该温度下完全在梦里训出来的控制器"在梦里和真实环境里的表现差距。
    """
    in_dim = Z_DIM + HIDDEN_DIM
    n_params = ACTION_DIM * in_dim + ACTION_DIM
    es = SimplifiedCMAES(dim=n_params, sigma=0.15, pop_size=pop_size, seed=SEED)
    dream_env = DreamEnv(model, init_z_pool, temperature=tau, seed=SEED)

    curve = []
    eval_every = max(1, n_gens // 4)
    t0 = time.time()
    for gen in range(1, n_gens + 1):
        candidates = es.ask()
        fitness = rollout_dream(dream_env, candidates, n_rollouts=n_rollouts, horizon=horizon)
        es.tell(candidates, fitness)
        if gen == 1 or gen % eval_every == 0 or gen == n_gens:
            dream_mean = float(rollout_dream(DreamEnv(model, init_z_pool, temperature=tau, seed=SEED + 10),
                                             es.mean[None, :], n_rollouts=8, horizon=horizon)[0])
            real_vals = eval_in_real_env(vae, model, es.mean, real_seeds[:4], horizon=horizon)
            real_mean = float(real_vals.mean())
            curve.append((gen, dream_mean, real_mean))
    return es.mean.copy(), curve, time.time() - t0


def temperature_experiment(vae, model, data, temperatures, n_gens, pop_size, n_rollouts, horizon, real_seeds):
    print("=" * 60)
    print("第四部分：在梦里训练控制器——不同温度下的钻空子现象")
    print("=" * 60)

    init_z_pool = data["mu"][data["train"], 0]                      # 训练集各回合起始帧 z_0
    print(f"""
把 {pop_size} 个候选 × {n_rollouts} 个梦境回合（共 {pop_size * n_rollouts} 条轨迹）拼成一个批次，
在梦里展开 {horizon} 步，跑 {n_gens} 代 CMA-ES。每 {max(1, n_gens // 5)} 代把均值放回真实 CarRacing-v3 评估一次。
下表是各温度下最终选出参数（并在梦里与 {len(real_seeds)} 条真实赛道上重测）的对比：
    """)
    print("    训练温度 τ    训练用时    梦里的分数（该温度下）    真实环境里的分数（均值 ± 标准差）")

    results = {}
    for tau in temperatures:
        params, curve, elapsed = train_controller_in_dream(
            model, vae, init_z_pool, tau, n_gens, pop_size, n_rollouts, horizon, real_seeds
        )
        dream_s, real_vals = compare_dream_and_real(vae, model, init_z_pool, params, tau, horizon, horizon, real_seeds)
        results[tau] = (params, curve, dream_s, real_vals)
        print(f"    τ = {tau:<5.2f}     {elapsed:4.1f}s          {dream_s:7.1f}                "
              f"{real_vals.mean():6.1f} ± {real_vals.std():5.1f}")

    # 选最低温度（钻空子最明显）解码一段梦境视频与关键帧
    low_tau = min(temperatures)
    low_params = results[low_tau][0]
    dream_env = DreamEnv(model, init_z_pool, temperature=low_tau, seed=SEED + 99)
    _, traj_z, traj_r = rollout_dream(dream_env, low_params[None, :], n_rollouts=1, horizon=min(horizon, 120), record_first=True)
    with torch.no_grad():
        frames = (vae.decode(traj_z[:-1]).permute(0, 2, 3, 1).clamp(0, 1) * 255).byte().cpu().numpy()
    big = lambda im: cv2.resize(im, (192, 192), interpolation=cv2.INTER_NEAREST)
    save_video([big(f) for f in frames], OUT / "08-dream-exploit.mp4", fps=20)

    picks = np.linspace(0, len(frames) - 1, 6).astype(int)
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
    for ax, idx in zip(axes, picks):
        ax.imshow(frames[idx])
        ax.set_title(f"t={idx + 1}\nr̂={traj_r[idx]:+.2f}", fontsize=8)
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    fig.suptitle(f"在 τ = {low_tau} 梦里学出的控制器：它看到的梦境与 M 给的每步奖励", fontsize=11)
    save_figure(fig, OUT / "08-dream-exploit-frames.png")

    # 图：各温度下的梦境分数 vs 真实分数对比 + 训练轨迹
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    taus = list(temperatures)
    dream_means = [results[t][2] for t in taus]
    real_means = [float(results[t][3].mean()) for t in taus]
    real_stds = [float(results[t][3].std()) for t in taus]
    x = np.arange(len(taus))
    w = 0.36
    axes[0].bar(x - w / 2, dream_means, width=w, label="梦里的分数")
    axes[0].bar(x + w / 2, real_means, width=w, yerr=real_stds, capsize=4, label="真实环境的分数")
    axes[0].set_xticks(x), axes[0].set_xticklabels([f"τ={t}" for t in taus])
    axes[0].set_ylabel(f"回合回报（{horizon} 步）")
    axes[0].set_title("低温下梦里虚高，提高 τ 后差距缩小")
    axes[0].legend(fontsize=8)

    for t in taus:
        c = results[t][1]
        axes[1].plot([g for g, _, _ in c], [d for _, d, _ in c], ls="--", label=f"τ={t} 梦")
        axes[1].plot([g for g, _, _ in c], [r for _, _, r in c], "o-", label=f"τ={t} 真实")
    axes[1].set_xlabel("代数")
    axes[1].set_ylabel("回合回报")
    axes[1].set_title("训练过程中：梦境分数与真实分数是否同步")
    axes[1].legend(fontsize=7, ncol=2)
    save_figure(fig, OUT / "08-dream-vs-real-temperatures.png")

    best_tau = max(taus, key=lambda t: results[t][3].mean())
    low_dream, low_real = results[low_tau][2], float(results[low_tau][3].mean())
    best_dream, best_real = results[best_tau][2], float(results[best_tau][3].mean())
    if low_dream > low_real + 20.0:
        print(f"""
怎么读：
    1. 低温（τ = {low_tau}）下梦近乎确定性，模型里的正奖励漏洞稳定可复现，控制器在梦里刷到 {low_dream:.1f} 分，
       放回真实环境只有 {low_real:.1f} 分——梦里的分数和真实分数完全脱节。
    2. 提高温度到 τ = {best_tau} 时，每次梦都更随机，靠精确卡模型漏洞的策略站不住脚，
       梦里分数是 {best_dream:.1f}，真实分数是 {best_real:.1f}（本次实验里真实环境最高，与梦境的虚高差距大幅缩小）。
    3. 再看上面保存的 `08-dream-exploit-frames.png`：低温控制器在梦里维持着某种看似给正奖励的画面或转向节奏，
       放回真实赛道却无法正常过弯。
        """)
    else:
        print("\n这次运行里低温梦境分数没有明显高于真实环境（训练预算太少时会这样，比如冒烟模式）。\n")


# ==================== 第六部分：练习 ====================


def exercises():
    print("=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把 world_model_loss 里的 pos_weight 从 100 改成 1（不加权），重新训练并统计验证集上
        真实终止步的预测概率均值。DreamEnv 里一个回合还会提前终止吗？

练习 2：在 DreamEnv.reset 里，先用真实数据集里的一段 20 步前缀给 LSTM 热身（而不是从零状态开始），
        再展开做梦。低温下的钻空子现象会消失吗？

练习 3：把 DreamEnv.step 里喂给控制器的特征改掉：仍然保持 M 内部推进，但只把 ẑ 喂给控制器
        （不把 h 喂给控制器）。对比梦里分数还能不能刷得那么高（回忆第 8.3 节第 3 条原因）。

练习 4：手算。设 N = 8 个候选，真实适应度全为 0，每个候选的估计误差独立等概率取 +2 或 -2。
        梦里看到的最高分的期望是多少？

思考题 1：为什么在真实环境里训练（07 节）不用担心钻空子，而在梦里训练必须每隔几代回真实环境测一次？

思考题 2：按照第 8.7 节的闭环迭代训练（用当前控制器回真实环境采数据 → 重训 M → 再在梦里训 C），
          第一轮低温梦里被钻出来的漏洞，第二轮还会存在吗？

参考答案：
    练习 1：正样本只占 0.056%，不加权时终止头几乎学成常数负 logit，真实终止步的预测概率只有 0.001 量级，
            DreamEnv 里回合永远不会因为撞出界而提前结束，会一直跑满 horizon。
    练习 2：不会消失。热身只让开头几步更贴近真实；只要控制器随后把 M 带出训练分布之外，
            后面一百多步里仍然可以找到并利用 M 的漏洞。
    练习 3：会有所下降，但不会消失。看不到 h 让控制器没法直接"读引擎内存"卡死某个神经元状态，
            但只要 ẑ 本身能引导 M 进入奖励头虚高的区域，漏洞依然存在。
    练习 4：误差缩放到 ±2，套公式 2 × (1 - 2^(1-8)) = 2 × (1 - 1/128) = 1.984。
    思考题 1：07 节的适应度由真实环境给出，没有"模型误差"可利用；梦境训练只靠 M 的输出打分，
              模型分数高不等于真实分数高，唯一可信的尺子是真实环境里的定期评估。
    思考题 2：在上一轮漏洞附近，新采回来的真实数据会告诉 M："开到那里其实没有正奖励（甚至会出界扣 100）"，
              那个漏洞被补上；但新控制器可能会在另一片没见过的数据区挖出新的漏洞，所以闭环要多轮迭代。
    """)


def main():
    seed_everything(SEED)
    if is_quick():
        wm_epochs = 2
        n_gens, pop_size, n_rollouts, horizon, n_real = 3, 6, 2, 60, 2
        temperatures = (0.1, 1.0)
    else:
        wm_epochs = 40
        n_gens, pop_size, n_rollouts, horizon, n_real = 40, 24, 4, 300, 8
        temperatures = (0.1, 0.5, 1.0, 1.3)

    introduction()
    check_selection_bias()
    model, data = prepare_world_model(wm_epochs)

    ensure_artifact(VAE_PATH, "02-conv-vae-vision.py")
    vae = ConvVAE().to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    vae.eval()

    real_seeds = list(range(4000, 4000 + n_real))
    temperature_experiment(vae, model, data, temperatures, n_gens, pop_size, n_rollouts, horizon, real_seeds)
    exercises()

    print("=" * 60)
    print("本节小结")
    print("=" * 60)
    print("""
    1. 对带误差的估计取最大值本身就会高估；在模型里做优化时，模型误差更是会被当成漏洞主动挖掘。
    2. 给 M 加上奖励头和终止头（注意最后一步终止帧也要算入奖励/终止损失，并对极少数的终止帧加权），
       就能把 M 包装成批量推进的 DreamEnv，几百条梦境轨迹一个张量同时展开。
    3. 温度 τ 是双刃剑：太低（0.1）梦境确定、漏洞可复现，梦里分数虚高但真实几乎无效；
       适度提高温度（接近或略高于 1）像"混沌工程"，迫使策略摆脱对模型细节的依赖；太高则会让梦过于混乱。
    4. 梦里训练必须配真实环境定期评估，要彻底缓解分布偏移，需要把采集数据与梦里训练做成闭环（Phase 3）。

Phase 2 全部讲义到此结束。下一章（Phase 3：RSSM 与 PlaNet）将把本章分开训练的 V 和 M 焊成一个端到端的序列世界模型。
    """)
    if is_quick():
        print("（当前是冒烟模式 WM_QUICK=1：训练轮数与代数都很小，上面的数字没有参考价值。）")


if __name__ == "__main__":
    main()
