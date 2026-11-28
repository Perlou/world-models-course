"""
C：极简控制器，在真实环境中训练
===============================

学习目标：
    1. 实现线性控制器 LinearController，核对参数量：只用 z 是 99，用 [z ; h] 是 867
    2. 按正确的时序接通 V → M → C 闭环：先用当前 z_t 和旧的 h_t 选动作，再用 (z_t, a_t) 更新 h_{t+1}
    3. 用上一节的 CMA-ES 在真实 CarRacing-v3 里训练控制器，每代共用随机种子以压低排名噪声
    4. 对比"只用 z"与"同时用 z 和 h"两种输入配置，在未见过的完整赛道上独立评估

核心概念：
    - 信用分配（credit assignment）：整回合只有总分一个数，所以控制器必须极小
    - 冻结特征 + 线性头：V 和 M 负责看和记（共 477 万参数），C 只用几百个参数学怎么做
    - 公共随机数（common random numbers）：同一代的所有候选跑同一组赛道，抵消赛道难易的噪声
    - 独立再评估：训练曲线上的"当代最高分"系统性偏高，最终成绩必须在独立赛道上重测

前置知识：
    - 02-conv-vae-vision.py（冻结的 V）
    - 04-mdn-rnn-memory.py（冻结的 M）
    - 06-evolution-strategies-cma-es.py（CMA-ES 黑盒优化器）

对应概念文档：
    - CONCEPT.md 第 7 节「C：极简控制器，在真实环境中训练」

预计运行时间：约 4–6 分钟（CPU 多进程并行评估）
"""

import math
import os
import subprocess
import sys
import time
from multiprocessing import get_context

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from utils import is_quick, output_dir, save_figure, save_video, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-02")
SEED = 0
Z_DIM, ACTION_DIM, HIDDEN_DIM, N_COMPONENTS = 32, 3, 256, 5
ZOOM_FRAMES, CROP_ROWS, IMAGE_SIZE = 50, 84, 64
N_WORKERS = min(6, os.cpu_count() or 4)
SUFFIX = "-quick" if is_quick() else ""
VAE_PATH = OUT / f"vae{SUFFIX}.pt"
MDNRNN_PATH = OUT / f"mdnrnn{SUFFIX}.pt"
CTRL_PATH = OUT / f"controller-real{SUFFIX}.npz"


# ==================== 第一部分：为什么控制器只有 867 个参数 ====================


def introduction():
    print("=" * 60)
    print("第一部分：极简控制器——把难事留给 V 和 M")
    print("=" * 60)
    print("""
三个部件的参数量对比：
    V（卷积 VAE）        4,348,547   每一帧 12288 个像素的监督信号 → 梯度下降
    M（MDN-RNN）            422,368   每一步 32 维 z 的分布信号     → 梯度下降
    C（线性控制器）             867   整回合结束才有一个总分        → CMA-ES（不用梯度）

控制器的公式只是一次仿射映射再压到合法动作范围：
    a_t = squash( W_c [z_t ; h_t] + b_c )
          方向 = tanh(raw_0) ∈ [-1, 1]
          油门 = 0.5 · (tanh(raw_1) + 1) ∈ [0, 1]
          刹车 = clip(tanh(raw_2), 0, 1) ∈ [0, 1]

当参数全为 0 时，初始动作是"方向 0、油门 0.5、刹车 0"，车会往前开而不是停在原地；
这让第 1 代候选之间立刻拉开分数差距，CMA-ES 才能看到方向。

注意真实环境里每一步的严格顺序（CONCEPT.md 第 7.4 节）：
    1. 拿到当前画面 o_t，用 V 编码出均值 z_t（不采样，减少评估噪声）
    2. 用当前 z_t 和"还没看过这一帧"的 h_t 算出动作 a_t = C(z_t, h_t)
    3. 环境执行 a_t，得到下一帧 o_{t+1} 和奖励 r_t
    4. 用 (z_t, a_t) 推动 M 走一步，更新出 h_{t+1}
    """)


# ==================== 模型与组件（V、M、CMA-ES、C） ====================


class ConvVAE(nn.Module):
    """02 节的卷积 VAE（这里只用编码器，加上解码器是为了和保存的权重匹配）。"""

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


class MDNRNN(nn.Module):
    """04 节的 MDN-RNN（这里主要用它的 LSTM 状态更新）。"""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(Z_DIM + ACTION_DIM, HIDDEN_DIM, batch_first=True)
        self.mdn_head = nn.Linear(HIDDEN_DIM, Z_DIM * N_COMPONENTS * 3)

    def step_hidden(self, z, a, state):
        """只推动 LSTM 走一步，跳过 MDN 头节省时间。z [1, 32], a [1, 3]。"""
        x = torch.cat([z, a], dim=-1).unsqueeze(1)
        _, new_state = self.lstm(x, state)
        return new_state


class LinearController:
    """线性控制器：把 z_t（或 [z_t ; h_t]）映射到 3 维连续动作。"""

    def __init__(self, use_hidden_state=True):
        self.use_hidden_state = use_hidden_state
        self.in_dim = Z_DIM + HIDDEN_DIM if use_hidden_state else Z_DIM
        self.n_params = ACTION_DIM * self.in_dim + ACTION_DIM
        self.w = np.zeros((ACTION_DIM, self.in_dim), dtype=np.float32)
        self.b = np.zeros(ACTION_DIM, dtype=np.float32)

    def set_params(self, flat_params):
        """把一维向量（99 或 867）拆成 W_c [3, in_dim] 和 b_c [3]。"""
        flat = np.asarray(flat_params, dtype=np.float32)
        assert flat.shape == (self.n_params,), f"期望参数长度 {self.n_params}，收到 {flat.shape}"
        w_size = ACTION_DIM * self.in_dim
        self.w = flat[:w_size].reshape(ACTION_DIM, self.in_dim)
        self.b = flat[w_size:]

    def act(self, z, h=None):
        """z [32], h [256] → 动作 [3]（float32）。"""
        feat = np.concatenate([z, h], axis=0) if self.use_hidden_state else z
        raw = self.w @ feat + self.b
        steer = math.tanh(float(raw[0]))
        gas = 0.5 * (math.tanh(float(raw[1])) + 1.0)
        brake = min(1.0, max(0.0, math.tanh(float(raw[2]))))
        return np.array([steer, gas, brake], dtype=np.float32)


class SimplifiedCMAES:
    """06 节的简化版 CMA-ES，搬过来直接用于优化控制器参数。"""

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


def preprocess(frame):
    """[96, 96, 3] uint8 → [1, 3, 64, 64] float32。"""
    small = cv2.resize(frame[:CROP_ROWS], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    arr = np.transpose(small, (2, 0, 1)).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ==================== 第二部分：单回合与多进程评估 ====================


def check_controller_shapes():
    print("=" * 60)
    print("第二部分：核对控制器参数量与动作映射")
    print("=" * 60)
    c_z = LinearController(use_hidden_state=False)
    c_zh = LinearController(use_hidden_state=True)
    c_zh.set_params(np.zeros(c_zh.n_params))
    a0 = c_zh.act(np.zeros(Z_DIM, dtype=np.float32), np.zeros(HIDDEN_DIM, dtype=np.float32))
    print(f"""
    只用 z_t：        输入 {c_z.in_dim:>3} 维 → 参数量 3 × {c_z.in_dim} + 3 = {c_z.n_params}
    同时用 [z_t ; h_t]：输入 {c_zh.in_dim:>3} 维 → 参数量 3 × {c_zh.in_dim} + 3 = {c_zh.n_params}
    全零参数下的输出动作：{np.round(a0, 3)}（方向 0、油门 0.5、刹车 0）
    """)
    assert c_z.n_params == 99 and c_zh.n_params == 867
    assert np.allclose(a0, [0.0, 0.5, 0.0], atol=1e-6)


@torch.no_grad()
def rollout_real(env, vae, mdn_rnn, controller, seed, max_steps=950, record=False):
    """在真实 CarRacing-v3 里跑一个回合，返回总回报（若 record=True 另返回画面帧）。"""
    obs, _ = env.reset(seed=int(seed))
    total_reward = 0.0
    for _ in range(ZOOM_FRAMES):
        obs, r, term, trunc, _ = env.step(np.zeros(3, dtype=np.float32))
        total_reward += float(r)
        if term or trunc:
            return (total_reward, []) if record else total_reward

    state = (torch.zeros(1, 1, HIDDEN_DIM), torch.zeros(1, 1, HIDDEN_DIM))
    frames = []
    for _ in range(max_steps):
        if record:
            frames.append(obs.copy())
        obs_t = preprocess(obs)
        mu, _ = vae.encode(obs_t)
        z_np = mu.squeeze(0).numpy()
        h_np = state[0].squeeze(0).squeeze(0).numpy()
        action = controller.act(z_np, h_np)
        obs, r, term, trunc, _ = env.step(action)
        total_reward += float(r)
        if controller.use_hidden_state:
            a_t = torch.from_numpy(action).unsqueeze(0)
            state = mdn_rnn.step_hidden(mu, a_t, state)
        if term or trunc:
            break
    return (total_reward, frames) if record else total_reward


# 多进程 worker 全局状态：每个子进程只在启动时加载一次权重和环境
_WORKER = {}


def _init_worker(vae_path, rnn_path):
    torch.set_num_threads(1)
    vae = ConvVAE()
    vae.load_state_dict(torch.load(vae_path, map_location="cpu"))
    vae.eval()
    mdn_rnn = MDNRNN()
    mdn_rnn.load_state_dict(torch.load(rnn_path, map_location="cpu"))
    mdn_rnn.eval()
    env = gym.make("CarRacing-v3", continuous=True)
    _WORKER["vae"] = vae
    _WORKER["rnn"] = mdn_rnn
    _WORKER["env"] = env


def _eval_candidate(job):
    flat_params, seeds, use_hidden_state, max_steps = job
    ctrl = LinearController(use_hidden_state=use_hidden_state)
    ctrl.set_params(flat_params)
    scores = [
        rollout_real(_WORKER["env"], _WORKER["vae"], _WORKER["rnn"], ctrl, s, max_steps=max_steps)
        for s in seeds
    ]
    return float(np.mean(scores)), scores


def evaluate_fitness(pool, candidates, seeds, use_hidden_state, max_steps):
    """在一批随机种子上并行评估整个种群，返回每个候选的平均回报 [N]。"""
    jobs = [(cand, seeds, use_hidden_state, max_steps) for cand in candidates]
    results = pool.map(_eval_candidate, jobs)
    return np.array([r[0] for r in results], dtype=np.float64)


# ==================== 第三部分：在真实环境中训练并对比 ====================


def train_controller(pool, use_hidden_state, n_gens, pop_size, n_rollouts, max_steps, eval_seeds):
    """CMA-ES 外循环。每代共用一组随机种子，每隔若干代在独立测试赛道上评估均值参数。"""
    ctrl = LinearController(use_hidden_state=use_hidden_state)
    es = SimplifiedCMAES(dim=ctrl.n_params, sigma=0.15, pop_size=pop_size, seed=SEED)
    rng = np.random.default_rng(SEED + (100 if use_hidden_state else 0))

    label = "[z_t ; h_t] (867 维)" if use_hidden_state else "只有 z_t (99 维)"
    print(f"\n--- 开始训练：{label}，种群 {pop_size}，每候选 {n_rollouts} 回合 × {max_steps} 步，共 {n_gens} 代 ---")

    gen_logs = []
    eval_points = []
    best_eval_score, best_params = -float("inf"), es.mean.copy()
    start = time.time()

    eval_every = max(1, n_gens // 5)
    for gen in range(1, n_gens + 1):
        gen_seeds = rng.integers(10_000, 1_000_000, size=n_rollouts).tolist()
        candidates = es.ask()
        fitness = evaluate_fitness(pool, candidates, gen_seeds, use_hidden_state, max_steps)
        es.tell(candidates, fitness)
        gen_logs.append((gen, float(fitness.mean()), float(fitness.max()), es.sigma))

        if gen == 1 or gen % eval_every == 0 or gen == n_gens:
            # 用独立测试赛道评估当前均值（完整 950 步）
            mean_scores = evaluate_fitness(pool, [es.mean], eval_seeds, use_hidden_state, max_steps=950)
            mean_val = float(mean_scores[0])
            eval_points.append((gen, mean_val))
            if mean_val > best_eval_score:
                best_eval_score = mean_val
                best_params = es.mean.copy()
            print(f"    第 {gen:>2} 代  当代均值 {fitness.mean():6.1f}  当代最高 {fitness.max():6.1f}  "
                  f"独立评估 {mean_val:6.1f}  σ={es.sigma:.3f}  用时 {time.time() - start:5.0f}s")

    elapsed = time.time() - start
    return best_params, gen_logs, eval_points, elapsed


def run_comparison(n_gens, pop_size, n_rollouts, max_steps, n_eval):
    print("=" * 60)
    print("第三部分：在真实环境中训练并对比有无 M 的记忆")
    print("=" * 60)

    ensure_artifact(VAE_PATH, "02-conv-vae-vision.py")
    ensure_artifact(MDNRNN_PATH, "04-mdn-rnn-memory.py")
    vae_state = torch.load(VAE_PATH, map_location="cpu")
    rnn_state = torch.load(MDNRNN_PATH, map_location="cpu")

    # 独立测试赛道种子：从 2000 开始，和 01 节采集数据的 0..199 以及训练时的 10000+ 完全隔离
    eval_seeds = list(range(2000, 2000 + n_eval))
    final_eval_seeds = list(range(3000, 3000 + max(n_eval, 12 if not is_quick() else 4)))

    print(f"""
启动 {N_WORKERS} 个并行工作进程（训练时每回合 {max_steps} 步，独立评估时用完整 950 步）。
先在 0 步外加对比：随机策略在完整回合上是 -43.0（逐帧独立）和 -90.9（时间相关），原地不动是 -100.0。
    """)

    with get_context("spawn").Pool(N_WORKERS, initializer=_init_worker, initargs=(str(VAE_PATH), str(MDNRNN_PATH))) as pool:
        params_z, logs_z, eval_z, time_z = train_controller(
            pool, False, n_gens, pop_size, n_rollouts, max_steps, eval_seeds
        )
        params_zh, logs_zh, eval_zh, time_zh = train_controller(
            pool, True, n_gens, pop_size, n_rollouts, max_steps, eval_seeds
        )

        # 用更充裕的一组全新赛道做最终评估（完整 950 步），避免被训练/中间筛选的方差高估
        print(f"\n在 {len(final_eval_seeds)} 条全新完整赛道（950 步）上做最终独立评估：")
        res_z = pool.map(_eval_candidate, [(params_z, [s], False, 950) for s in final_eval_seeds])
        res_zh = pool.map(_eval_candidate, [(params_zh, [s], True, 950) for s in final_eval_seeds])

    scores_z = np.array([r[0] for r in res_z])
    scores_zh = np.array([r[0] for r in res_zh])
    np.savez(
        CTRL_PATH,
        params_z=params_z,
        params_zh=params_zh,
        scores_z=scores_z,
        scores_zh=scores_zh,
        logs_z=np.array(logs_z),
        logs_zh=np.array(logs_zh),
    )

    print(f"""
    配置                 参数量     训练用时     独立评估回报（均值 ± 标准差）     最低 / 最高
    只有 z_t               99      {time_z:5.0f}s        {scores_z.mean():6.1f} ± {scores_z.std():5.1f}               {scores_z.min():6.1f} / {scores_z.max():6.1f}
    [z_t ; h_t]           867      {time_zh:5.0f}s        {scores_zh.mean():6.1f} ± {scores_zh.std():5.1f}               {scores_zh.min():6.1f} / {scores_zh.max():6.1f}

💾 控制器参数已保存：{CTRL_PATH.relative_to(OUT.parents[1])}
    """)

    # 画学习曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    gens_z = [x[0] for x in logs_z]
    gens_zh = [x[0] for x in logs_zh]
    axes[0].plot(gens_z, [x[1] for x in logs_z], color="tab:orange", label="只有 z_t（当代均值）")
    axes[0].plot(gens_z, [x[2] for x in logs_z], color="tab:orange", ls=":", alpha=0.7, label="只有 z_t（当代最高）")
    axes[0].plot(gens_zh, [x[1] for x in logs_zh], color="tab:blue", label="[z_t ; h_t]（当代均值）")
    axes[0].plot(gens_zh, [x[2] for x in logs_zh], color="tab:blue", ls=":", alpha=0.7, label="[z_t ; h_t]（当代最高）")
    axes[0].set_xlabel("代数")
    axes[0].set_ylabel(f"训练期回报（{max_steps} 步）")
    axes[0].set_title("训练曲线：当代最高分总比均值虚高一截")
    axes[0].legend(fontsize=8)

    axes[1].plot([g for g, _ in eval_z], [v for _, v in eval_z], "o-", color="tab:orange", label="只有 z_t（独立评估）")
    axes[1].plot([g for g, _ in eval_zh], [v for _, v in eval_zh], "s-", color="tab:blue", label="[z_t ; h_t]（独立评估）")
    axes[1].axhline(-43.0, color="gray", ls="--", lw=1, label="随机策略 (-43.0)")
    axes[1].set_xlabel("代数")
    axes[1].set_ylabel("独立测试赛道回报（950 步）")
    axes[1].set_title("独立赛道评估：有无 M 的记忆 h")
    axes[1].legend(fontsize=8)
    save_figure(fig, OUT / "07-real-env-training.png")

    # 保存一条对比驾驶视频
    vae, mdn_rnn = ConvVAE(), MDNRNN()
    vae.load_state_dict(vae_state)
    mdn_rnn.load_state_dict(rnn_state)
    vae.eval(), mdn_rnn.eval()
    env = gym.make("CarRacing-v3", continuous=True)
    ctrl_z = LinearController(use_hidden_state=False)
    ctrl_z.set_params(params_z)
    ctrl_zh = LinearController(use_hidden_state=True)
    ctrl_zh.set_params(params_zh)
    vid_seed = final_eval_seeds[int(np.argmax(scores_zh))]
    ret_z, frames_z = rollout_real(env, vae, mdn_rnn, ctrl_z, vid_seed, max_steps=450, record=True)
    ret_zh, frames_zh = rollout_real(env, vae, mdn_rnn, ctrl_zh, vid_seed, max_steps=450, record=True)
    env.close()
    n_f = min(len(frames_z), len(frames_zh))
    if n_f > 0:
        big = lambda im: cv2.resize(im, (192, 192), interpolation=cv2.INTER_NEAREST)
        video = [np.concatenate([big(frames_z[t]), big(frames_zh[t])], axis=1) for t in range(n_f)]
        save_video(video, OUT / "07-controller-compare.mp4", fps=25)
        print(f"视频对比（赛道种子 {vid_seed}，前 {n_f} 步）：左 = 只有 z ({ret_z:.1f}) ｜ 右 = [z ; h] ({ret_zh:.1f})\n")


# ==================== 第四部分：练习 ====================


def exercises():
    print("=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把 LinearController 的动作映射里刹车那一维改成 sigmoid(raw_2)（全零初始化时刹车默认 0.5），
        训练早期会发生什么？和本节的 clip(tanh, 0, 1) 对比。

练习 2：去掉 train_controller 里"同一代共用一组随机种子"的设置（改成每个候选各自抽种子），
        观察同一代里适应度的波动和学习曲线的稳定性。

练习 3：把控制器改造成只用 z_t 但加一个 16 单元的隐藏层（32→16→3，tanh 激活），参数量是多少？
        在相同预算下再训一遍，它能追上线性 [z_t ; h_t] 吗？

练习 4：手算。如果单回合回报的标准差是 s = 200，要让评估均值的标准差压到 20 以内，
        每个候选至少要跑多少个回合 R？

思考题 1：为什么训练曲线上"当代最高分"不能当作最终成绩？如果在第 15 代选出了分数最高的候选，
          直接拿它和"第 15 代搜索分布的均值"去独立评估，哪个通常更稳？

思考题 2：在真实环境里训完之后，V 和 M 的权重从头到尾都没变。如果控制器因为开得好而进入了
          随机策略从未到过的急弯区域，会出什么问题？

参考答案：
    练习 1：全零参数时油门 0.5、刹车 0.5，车几乎不动或开得极慢，大部分第一代候选分数都堆在 -100 附近，
            排序几乎全是噪声，步长很快缩掉。
    练习 2：赛道难易的随机性直接混进排名里，跑在简单直道赛道上的坏策略可能排在跑在急弯赛道的好策略前面，
            学习曲线抖动加剧、收敛变慢。
    练习 3：参数量 32×16 + 16 + 16×3 + 3 = 579。和论文第 7.5 节的结论一致：加隐藏层能比线性只用 z 好一点，
            但单帧 z 里本来就没有车速，容量再大也变不出缺失的动力学状态。
    练习 4：s / √R ≤ 20 → √R ≥ 10 → R ≥ 100 个回合（这也正是论文最终评估用 100 个回合的原因）。
    思考题 1：当代最高分包含"碰巧抽到好参数/好赛道"的正噪声（CONCEPT.md 第 8.3 节）。
              独立评估时，搜索分布均值 m 聚合了多代精英，通常比单代挑出来的极值更稳。
    思考题 2：V 重建出的 z 和 M 更新出的 h 处于分布外区域，精度下降；但由于控制器是在真实环境里直接交互、
              靠真实奖励筛选的，CMA-ES 仍能适应哪怕有偏差的 [z ; h] 特征——只要这个特征对动作还算有区分度。
              而在梦里训练时（下一节），这会变成更严重的问题。
    """)


def main():
    seed_everything(SEED)
    if is_quick():
        n_gens, pop_size, n_rollouts, max_steps, n_eval = 2, 6, 1, 80, 2
    else:
        n_gens, pop_size, n_rollouts, max_steps, n_eval = 25, 16, 2, 450, 8

    introduction()
    check_controller_shapes()
    run_comparison(n_gens, pop_size, n_rollouts, max_steps, n_eval)
    exercises()

    print("=" * 60)
    print("本节小结")
    print("=" * 60)
    print("""
    1. 控制器只有 867 个参数（线性），把重活留给 V 和 M，才能用每回合只有一个标量总分的 CMA-ES 训得动。
    2. 时序不能错：先用 (z_t, h_t) 选 a_t，再把 (z_t, a_t) 喂进 M 得到 h_{t+1}。
    3. 每代共用一组随机种子减少排名噪声；最终分数必须放到独立赛道上重新评估。
    4. 加上 M 的隐藏状态 h_t，控制器拿到了车速与运动趋势，分数和稳定性通常都好于只用当前帧 z_t。

下一节：干脆不进真实环境——给 M 加上奖励头和终止头，把 M 包装成环境，完全在梦里训练控制器。
    """)
    if is_quick():
        print("（当前是冒烟模式 WM_QUICK=1：训练预算极小，上面的数字没有参考价值。）")


if __name__ == "__main__":
    main()
