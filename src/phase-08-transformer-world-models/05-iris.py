"""
IRIS：Token 级自回归世界模型 + 想象中的 Actor-Critic 闭环
========================================================

学习目标：
    1. 核对 CONCEPT.md 第 5.3 节 λ-return 手算例子（[3.93, 3.51, 3.80, 2.0]，第 0 步优势 +0.43）
    2. 组装完整缩小版 IRIS（IrisWorldModel + imagine_rollout() + actor_critic_loss()）：
       三部件独立训练（FSQ Tokenizer / 因果 WorldModelGPT + KV Cache / 像素输入 Actor-Critic），互不传梯度
    3. 在轻量像素接球环境 PixelCatchEnv（16×16×3，3 离散动作）上跑通完整在线闭环，
       记录学习曲线，并精确统计训练耗时在 Tokenizer / 世界模型 / 想象与行为三部分的占比

核心概念：
    - 离散 token 想象 + REINFORCE：token 采样不可导，世界模型冻结当作黑盒仿真器，用带基线 REINFORCE 训练策略
    - 同分布观测：真实交互时策略看到的也是经 tokenizer 编解码的画面，与想象解码画面保持分布一致
    - 速度瓶颈定位：每想象 1 步需 K+1 次串行 Transformer 调用 + 1 次解码，想象阶段占据绝大部分耗时

前置知识：
    - 02、03、04 节（FSQ、WorldModelGPT、KVCache）

对应概念文档：
    - CONCEPT.md 第 5 节「IRIS：token 级的自回归世界模型」

预计运行时间：约 1.5 分钟（CPU）
"""

import importlib.util
import math
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod02 = _load("p8_02", "02-fsq-and-tokenizer-quality.py")
_mod03 = _load("p8_03", "03-gpt-as-world-model.py")
_mod04 = _load("p8_04", "04-kv-cache-and-imagination.py")

FSQ = _mod02.FSQ
WorldModelGPT = _mod03.WorldModelGPT
world_model_loss = _mod03.world_model_loss
KVCache = _mod04.KVCache

plt = setup_matplotlib()
OUT = output_dir("phase-08")
SEED = 0


# ==================== 第一部分：轻量 16×16 像素接球环境 ====================


class PixelCatchEnv:
    """16×16 RGB 像素接球小环境：绿色小球从顶部下落，蓝色挡板底部左右移动（3 动作：左/停/右），接住 +1、漏接 -1。"""

    def __init__(self, grid_size=8, img_size=16, seed=0):
        self.grid_size = grid_size
        self.img_size = img_size
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.paddle_x = self.grid_size // 2
        self.ball_x = int(self.rng.integers(1, self.grid_size - 1))
        self.ball_y = 0
        self.steps = 0
        return self._render()

    def _render(self):
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        # 蓝色挡板（宽 3 格，底行）
        x0 = max(0, self.paddle_x - 1)
        x1 = min(self.grid_size, self.paddle_x + 2)
        img[self.grid_size - 1, x0:x1, 2] = 1.0
        # 绿色小球
        img[min(self.ball_y, self.grid_size - 1), self.ball_x, 1] = 1.0
        # 上采样到 16×16
        img_up = np.repeat(np.repeat(img, self.img_size // self.grid_size, axis=0), self.img_size // self.grid_size, axis=1)
        return np.transpose(img_up, (2, 0, 1))  # [3, 16, 16]

    def step(self, action):
        # action: 0=左, 1=停, 2=右
        self.paddle_x = int(np.clip(self.paddle_x + (int(action) - 1), 1, self.grid_size - 2))
        self.ball_y += 1
        self.steps += 1
        done = False
        reward = 0.0
        if self.ball_y >= self.grid_size - 1:
            caught = abs(self.ball_x - self.paddle_x) <= 1
            reward = 1.0 if caught else -1.0
            # 重新生成下一颗球（每回合 3 颗球 = 21 步）
            if self.steps >= (self.grid_size - 1) * 3:
                done = True
            else:
                self.ball_y = 0
                self.ball_x = int(self.rng.integers(1, self.grid_size - 1))
        else:
            # 轻微稠密引导：靠近小球水平位置
            reward = -0.02 * abs(self.ball_x - self.paddle_x)
        return self._render(), float(reward), done


# ==================== 第二部分：IRIS 三部件（Tokenizer / WorldModel / ActorCritic） ====================


class MiniPixelTokenizer(nn.Module):
    """16×16×3 → 4×4 (K=16) FSQ Tokenizer（levels=[5,5,3] → 词表 N=75）。"""

    def __init__(self, levels=(5, 5, 3)):
        super().__init__()
        self.fsq = FSQ(levels)
        d = self.fsq.dim
        self.enc = nn.Sequential(
            nn.Conv2d(3, 24, 4, 2, 1), nn.SiLU(),  # 8
            nn.Conv2d(24, d, 4, 2, 1),             # 4
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(d, 24, 4, 2, 1), nn.SiLU(),  # 8
            nn.ConvTranspose2d(24, 3, 4, 2, 1), nn.Sigmoid(),  # 16
        )

    def encode(self, obs):
        z = self.enc(obs).permute(0, 2, 3, 1)
        z_q, idx = self.fsq(z)
        return z_q, idx.reshape(obs.shape[0], -1)  # idx: [B, 16]

    def decode_tokens(self, idx_flat):
        idx = idx_flat.reshape(-1, 4, 4)
        z_q = self.fsq.decode_indices(idx).permute(0, 3, 1, 2)
        return self.dec(z_q)

    def forward(self, obs):
        z_q, idx = self.encode(obs)
        return self.dec(z_q.permute(0, 3, 1, 2)), idx


class IrisWorldModel(nn.Module):
    """包装 WorldModelGPT 与 KV Cache，提供每步想象接口。"""

    def __init__(self, vocab_size=75, tokens_per_frame=16, act_dim=3, d_model=96, n_layers=2, n_heads=4):
        super().__init__()
        self.gpt = WorldModelGPT(
            vocab_size=vocab_size, tokens_per_frame=tokens_per_frame, act_dim=act_dim,
            discrete_actions=True, d_model=d_model, n_layers=n_layers, n_heads=n_heads, max_seq_len=300,
        )
        self.K = tokens_per_frame

    def loss(self, tokens, actions, rewards):
        return world_model_loss(self.gpt, tokens, actions, rewards)

    @torch.no_grad()
    def imagine_step(self, cache, action_idx):
        a_emb = self.gpt.act_emb(action_idx).unsqueeze(1)
        logits, rew_p = cache.append_and_forward(self.gpt, a_emb)
        rew = rew_p[:, -1]
        toks = []
        for _ in range(self.K):
            t_next = torch.distributions.Categorical(logits=logits[:, -1]).sample()
            toks.append(t_next)
            logits, _ = cache.append_and_forward(self.gpt, self.gpt.tok_emb(t_next).unsqueeze(1))
        return torch.stack(toks, dim=1), rew


class PixelActorCritic(nn.Module):
    """看 16×16 解码画面的离散动作 Actor-Critic（CONCEPT.md 第 5.2 节）。"""

    def __init__(self, act_dim=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),  # 8
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), # 4
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64), nn.ReLU(),
        )
        self.pi = nn.Linear(64, act_dim)
        self.v = nn.Linear(64, 1)

    def forward(self, obs):
        h = self.conv(obs)
        return self.pi(h), self.v(h).squeeze(-1)


def compute_lambda_returns_iris(rewards, values, gamma=0.95, lam=0.9):
    """rewards: [B, H], values: [B, H+1] → returns [B, H]。"""
    B, H = rewards.shape
    outs = torch.zeros(B, H, device=rewards.device)
    g = values[:, H]
    for t in reversed(range(H)):
        g = rewards[:, t] + gamma * ((1.0 - lam) * values[:, t + 1] + lam * g)
        outs[:, t] = g
    return outs


@torch.no_grad()
def imagine_rollout(tokenizer, iris_wm, ac, start_obs, horizon=8):
    """CONCEPT.md 第 5.4 节第 ④ 步：冻结 tokenizer 与世界模型，带 KV cache 批量想象 H 步。"""
    tokenizer.eval()
    iris_wm.eval()
    _, tok0 = tokenizer.encode(start_obs)  # [B, 16]
    cache = KVCache()
    cache.append_and_forward(iris_wm.gpt, iris_wm.gpt.tok_emb(tok0))

    obs_list = [tokenizer.decode_tokens(tok0)]
    act_list, rew_list = [], []
    for _ in range(horizon):
        logits_a, _ = ac(obs_list[-1])
        a = torch.distributions.Categorical(logits=logits_a).sample()
        nxt_tok, rew = iris_wm.imagine_step(cache, a)
        nxt_obs = tokenizer.decode_tokens(nxt_tok)
        act_list.append(a)
        rew_list.append(rew)
        obs_list.append(nxt_obs)
    return torch.stack(obs_list, dim=1), torch.stack(act_list, dim=1), torch.stack(rew_list, dim=1)


def actor_critic_loss(ac, imag_obs, imag_actions, imag_rewards, gamma=0.95, lam=0.9, eta=1e-2):
    """CONCEPT.md 第 5.3 节：REINFORCE + 基线 + 熵奖励 + 价值回归。"""
    B, Hp1, C, H_im, W_im = imag_obs.shape
    H = Hp1 - 1
    logits_all, vals_all = ac(imag_obs.reshape(B * Hp1, C, H_im, W_im))
    logits_all = logits_all.reshape(B, Hp1, -1)
    vals_all = vals_all.reshape(B, Hp1)

    with torch.no_grad():
        targets = compute_lambda_returns_iris(imag_rewards, vals_all.detach(), gamma=gamma, lam=lam)
        adv = targets - vals_all[:, :-1].detach()

    dist = torch.distributions.Categorical(logits=logits_all[:, :-1])
    pi_loss = -(dist.log_prob(imag_actions) * adv).mean() - eta * dist.entropy().mean()
    v_loss = 0.5 * F.mse_loss(vals_all[:, :-1], targets)
    return pi_loss + v_loss


# ==================== 第三部分：手算核对与完整闭环训练 ====================


def verify_section_5_hand_calc():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 5.3 节 λ-return 手算例子")
    print("=" * 60)
    r = torch.tensor([[1.0, 0.0, 2.0]])
    v = torch.tensor([[3.5, 3.0, 4.0, 2.0]])
    g = compute_lambda_returns_iris(r, v, gamma=0.9, lam=0.5)[0]
    adv0 = (g[0] - v[0, 0]).item()
    print(f"  G^λ = {[round(x, 3) for x in g.tolist()]}（手算 [3.93, 3.51, 3.80]），第 0 步优势 = {adv0:+.3f}（手算 +0.43 ✓）")
    assert torch.allclose(g, torch.tensor([3.9295, 3.51, 3.8]), atol=1e-3) and abs(adv0 - 0.4295) < 1e-3


def run_iris_experiment():
    quick = is_quick()
    n_rounds = 4 if quick else 14
    horizon = 6 if quick else 8
    tok_upd = 10 if quick else 35
    wm_upd = 10 if quick else 35
    ac_upd = 6 if quick else 18

    print("\n" + "=" * 60)
    print(f"第二部分：PixelCatchEnv 上运行缩小版 IRIS（{n_rounds} 轮，想象视野 H={horizon}，K=16 token/帧）")
    print("=" * 60)

    env = PixelCatchEnv(seed=SEED)
    tokenizer = MiniPixelTokenizer(levels=(5, 5, 3))
    iris_wm = IrisWorldModel(vocab_size=75, tokens_per_frame=16, act_dim=3, d_model=96, n_layers=2, n_heads=4)
    ac = PixelActorCritic(act_dim=3)

    opt_tok = torch.optim.Adam(tokenizer.parameters(), lr=3e-3)
    opt_wm = torch.optim.Adam(iris_wm.parameters(), lr=2e-3)
    opt_ac = torch.optim.Adam(ac.parameters(), lr=2e-3)

    replay_eps = deque(maxlen=120)
    all_frames = deque(maxlen=3000)
    rng = np.random.default_rng(SEED)

    def collect(n_eps, use_policy=False):
        rets = []
        ac.eval(); tokenizer.eval()
        for _ in range(n_eps):
            o = env.reset()
            ep_o, ep_a, ep_r = [], [], []
            done = False
            while not done:
                if not use_policy:
                    a = int(rng.integers(0, 3))
                else:
                    with torch.no_grad():
                        o_t = torch.from_numpy(o[None])
                        # IRIS 同分布细节：策略看 tokenizer 重建后的画面
                        o_rec, _ = tokenizer(o_t)
                        logits, _ = ac(o_rec)
                        a = int(torch.distributions.Categorical(logits=logits).sample().item())
                nxt, r, done = env.step(a)
                ep_o.append(o)
                ep_a.append(a)
                ep_r.append(r)
                all_frames.append(o)
                o = nxt
            replay_eps.append((np.asarray(ep_o, dtype=np.float32), np.asarray(ep_a, dtype=np.int64), np.asarray(ep_r, dtype=np.float32)))
            rets.append(sum(ep_r))
        return float(np.mean(rets))

    def eval_policy(n_eps=10):
        ac.eval(); tokenizer.eval()
        ev_env = PixelCatchEnv(seed=900)
        rets = []
        for i in range(n_eps):
            o = ev_env.reset(seed=900 + i)
            done = False
            tot = 0.0
            while not done:
                with torch.no_grad():
                    o_rec, _ = tokenizer(torch.from_numpy(o[None]))
                    logits, _ = ac(o_rec)
                    a = int(logits.argmax(dim=-1).item())
                o, r, done = ev_env.step(a)
                tot += r
            rets.append(tot)
        return float(np.mean(rets))

    collect(12, use_policy=False)
    rand_score = eval_policy(10)
    t_tok = t_wm = t_imag = 0.0
    imag_steps_done = 0
    eval_x, eval_y = [], []
    env_steps = sum(len(ep[0]) for ep in replay_eps)

    for rd in range(1, n_rounds + 1):
        # ② 更新 Tokenizer
        t0 = time.time()
        tokenizer.train()
        frames_arr = np.asarray(all_frames, dtype=np.float32)
        for _ in range(tok_upd):
            idx = rng.integers(0, len(frames_arr), size=32)
            x_b = torch.from_numpy(frames_arr[idx])
            rec, _ = tokenizer(x_b)
            loss = F.mse_loss(rec, x_b)
            opt_tok.zero_grad(); loss.backward(); opt_tok.step()
        t_tok += time.time() - t0

        # ③ 更新世界模型（Tokenizer 冻结）
        t1 = time.time()
        tokenizer.eval(); iris_wm.train()
        T_c = 8
        for _ in range(wm_upd):
            ob, ab, rb = [], [], []
            for _ in range(16):
                eo, ea, er = replay_eps[int(rng.integers(0, len(replay_eps)))]
                s = int(rng.integers(0, len(eo) - T_c + 1))
                ob.append(eo[s : s + T_c]); ab.append(ea[s : s + T_c]); rb.append(er[s : s + T_c])
            o_t = torch.from_numpy(np.stack(ob))
            with torch.no_grad():
                _, toks = tokenizer.encode(o_t.reshape(-1, 3, 16, 16))
                toks = toks.reshape(16, T_c, 16)
            loss_w, _, _, _ = iris_wm.loss(toks, torch.from_numpy(np.stack(ab)), torch.from_numpy(np.stack(rb)))
            opt_wm.zero_grad(); loss_w.backward(); opt_wm.step()
        t_wm += time.time() - t1

        # ④ 在想象中更新 Actor-Critic（从第 2 轮起世界模型已初步收敛）
        t2 = time.time()
        ac.train()
        for _ in range(ac_upd):
            idx = rng.integers(0, len(frames_arr), size=16)
            start_o = torch.from_numpy(frames_arr[idx])
            im_o, im_a, im_r = imagine_rollout(tokenizer, iris_wm, ac, start_o, horizon=horizon)
            imag_steps_done += 16 * horizon
            loss_ac = actor_critic_loss(ac, im_o, im_a, im_r)
            opt_ac.zero_grad(); loss_ac.backward(); opt_ac.step()
        t_imag += time.time() - t2

        # ① 收集新经验并评估
        collect(4, use_policy=True)
        env_steps += 4 * 21
        ev_r = eval_policy(8 if quick else 16)
        eval_x.append(env_steps)
        eval_y.append(ev_r)
        if rd == 1 or rd % max(1, n_rounds // 4) == 0 or rd == n_rounds:
            print(f"  第 {rd:>2}/{n_rounds} 轮（累计 {env_steps:>4} 步）：评估回报 = {ev_r:+.2f}（满分 +3.0，随机 ≈ {rand_score:+.2f}）")

    t_tot = t_tok + t_wm + t_imag
    ips = imag_steps_done / max(t_imag, 1e-6)
    print(f"\n  训练耗时占比（总 {t_tot:.1f}s）：")
    print(f"    Tokenizer: {t_tok:5.1f}s ({t_tok/t_tot*100:4.1f}%) ｜ 世界模型: {t_wm:5.1f}s ({t_wm/t_tot*100:4.1f}%) ｜ 想象+AC: {t_imag:5.1f}s ({t_imag/t_tot*100:4.1f}%)")
    print(f"    IRIS 想象吞吐率 ≈ {ips:.1f} 想象步/秒（供下一节与 STORM 直接对比！）")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    axes[0].plot(eval_x, eval_y, "o-", color="C0", linewidth=2.2, label=f"缩小版 IRIS（最终 {eval_y[-1]:+.2f}）")
    axes[0].axhline(rand_score, color="tomato", linestyle=":", label=f"随机策略（{rand_score:+.2f}）")
    axes[0].set_xlabel("真实环境交互步数"); axes[0].set_ylabel("平均回合回报（满分 +3.0）")
    axes[0].set_title("PixelCatchEnv：缩小版 IRIS 学习曲线"); axes[0].legend(fontsize=8.5)

    labs = ["Tokenizer", "世界模型", "想象 + AC"]
    vals = [t_tok, t_wm, t_imag]
    axes[1].bar(labs, vals, color=["C0", "C1", "tomato"], width=0.5)
    for i, v in enumerate(vals):
        axes[1].text(i, v + max(vals) * 0.02, f"{v:.1f}s\n({v/t_tot:.0%})", ha="center", fontsize=9)
    axes[1].set_ylim(0, max(vals) * 1.24)
    axes[1].set_ylabel("累计耗时 (s)"); axes[1].set_title("IRIS 耗时拆解（逐 Token 自回归想象最费时）")
    save_figure(fig, OUT / "05-iris-training.png")


def main():
    seed_everything(SEED)
    verify_section_5_hand_calc()
    run_iris_experiment()
    print("\n" + "=" * 60)
    print("本节完成！下一节：06-storm-and-twm.py（STORM 每帧一位置提速一个量级 & TWM 对照）")
    print("=" * 60)


if __name__ == "__main__":
    main()
