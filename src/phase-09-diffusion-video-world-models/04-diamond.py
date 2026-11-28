"""
Phase 9 · 第 04 课：DIAMOND —— 像素扩散世界模型与在扩散的梦里训练智能体
======================================================================

学习目标：
    1. 核对 CONCEPT.md 第 4.3 节（EDM vs DDPM 误差放大 10× 对比）与第 4.4 节（漏球导致 λ-return 暴跌）手算
    2. 组装 DIAMOND 三个部件：EDM 像素去噪器 D_θ、CNN+GRU 奖励/终止模型 R_ψ、CNN+GRU Actor-Critic
    3. 在与 Phase 8（IRIS / STORM）完全相同的 32×32 小球接住环境 PixelCatchEnv 上在线训练 DIAMOND
    4. 对比 1 步 vs 3 步去噪的想象质量、想象回报 vs 真实回报，并对照 Phase 8 的离散 token 路线

对应概念文档：
    src/phase-09-diffusion-video-world-models/CONCEPT.md 第 4 节

运行方式：
    uv run python src/phase-09-diffusion-video-world-models/04-diamond.py
    WM_QUICK=1 uv run python src/phase-09-diffusion-video-world-models/04-diamond.py

预计运行时间：完整约 3 分钟（M1 Max CPU），冒烟约 15 秒
"""

import importlib.util
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_DIR)

_spec03 = importlib.util.spec_from_file_location("p9_03", os.path.join(_DIR, "03-action-conditioned-frame-diffusion.py"))
_mod03 = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(_mod03)

_spec_p8_05 = importlib.util.spec_from_file_location(
    "p8_05_env", os.path.join(_SRC, "phase-08-transformer-world-models", "05-iris.py")
)
_mod_p8_05 = importlib.util.module_from_spec(_spec_p8_05)
_spec_p8_05.loader.exec_module(_mod_p8_05)

FrameDenoiser = _mod03.FrameDenoiser
sample_next_frame = _mod03.sample_next_frame
psnr_from_m11 = _mod03.psnr_from_m11
PixelCatchEnv = _mod_p8_05.PixelCatchEnv

plt = setup_matplotlib()
OUT = output_dir("phase-09")
SEED = 0


# ==================== 第一部分：DIAMOND 部件（奖励/终止模型 + ActorCritic） ====================


class TinyCNNBackbone(nn.Module):
    def __init__(self, in_ch=3, feat_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, feat_dim), nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)


class RewardEndModel(nn.Module):
    """CONCEPT.md 第 4.2 节：CNN + GRU 独立预测每一步的奖励与终止概率。"""

    def __init__(self, n_actions=3, feat_dim=64, hidden_dim=64):
        super().__init__()
        self.cnn = TinyCNNBackbone(3, feat_dim)
        self.act_emb = nn.Embedding(n_actions, 16)
        self.gru = nn.GRU(feat_dim + 16, hidden_dim, batch_first=True)
        self.rew_head = nn.Linear(hidden_dim, 1)
        self.end_head = nn.Linear(hidden_dim, 1)

    def forward(self, frames, actions, h=None):
        # frames: [B, T, 3, 32, 32], actions: [B, T]
        B, T = frames.shape[:2]
        feat = self.cnn(frames.reshape(B * T, *frames.shape[2:])).reshape(B, T, -1)
        a_e = self.act_emb(actions.long())
        out, h_next = self.gru(torch.cat([feat, a_e], dim=-1), h)
        return self.rew_head(out).squeeze(-1), self.end_head(out).squeeze(-1), h_next


class ActorCritic(nn.Module):
    """CONCEPT.md 第 4.2 节：直接看像素（CNN + GRU）输出策略分布与价值。"""

    def __init__(self, n_actions=3, feat_dim=64, hidden_dim=64):
        super().__init__()
        self.cnn = TinyCNNBackbone(3, feat_dim)
        self.gru = nn.GRU(feat_dim, hidden_dim, batch_first=True)
        self.pi_head = nn.Linear(hidden_dim, n_actions)
        self.v_head = nn.Linear(hidden_dim, 1)

    def step(self, obs, h=None):
        feat = self.cnn(obs).unsqueeze(1)
        out, h_next = self.gru(feat, h)
        out = out.squeeze(1)
        return self.pi_head(out), self.v_head(out).squeeze(-1), h_next

    def forward(self, frames, h=None):
        B, T = frames.shape[:2]
        feat = self.cnn(frames.reshape(B * T, *frames.shape[2:])).reshape(B, T, -1)
        out, h_next = self.gru(feat, h)
        return self.pi_head(out), self.v_head(out).squeeze(-1), h_next


def lambda_returns(rewards, ends, values, gamma=0.99, lam=0.95):
    """CONCEPT.md 第 4.4 节：带终止标志的反向 λ-return（rewards/ends [B, H], values [B, H+1]）。"""
    B, H = rewards.shape
    out = torch.zeros_like(rewards)
    g = values[:, -1]
    for t in reversed(range(H)):
        cont = 1.0 - ends[:, t]
        g = rewards[:, t] + gamma * cont * ((1.0 - lam) * values[:, t + 1] + lam * g)
        out[:, t] = g
    return out


def actor_critic_loss(log_probs, entropies, values, rewards, ends, gamma=0.99, lam=0.95, ent_coef=0.02):
    """CONCEPT.md 第 4.4 节：REINFORCE + 价值基线 + 熵正则 + 价值回归。"""
    targets = lambda_returns(rewards, ends, values.detach(), gamma=gamma, lam=lam).detach()
    adv = targets - values[:, :-1].detach()
    adv = (adv - adv.mean()) / adv.std().clamp_min(1e-4)
    pi_loss = -(log_probs * adv).mean() - ent_coef * entropies.mean()
    v_loss = 0.5 * F.mse_loss(values[:, :-1], targets)
    return pi_loss + v_loss, targets.mean().item()


def imagine(wm, rew_end, actor_critic, init_frames, init_actions, horizon=8, n_denoise=3):
    """从真实片段出发在像素扩散世界模型里想象 H 步（世界模型作为黑盒采样器，不反传穿过 U-Net）。"""
    B, m = init_frames.shape[:2]
    buf_f = init_frames.clone()   # [B, m, 3, 32, 32]
    buf_a = init_actions.clone()  # [B, m-1]

    # 用真实上下文帧预热 actor-critic 的 GRU 隐藏状态
    with torch.no_grad():
        _, _, h_ac = actor_critic(buf_f[:, :-1]) if m > 1 else (None, None, None)
        h_re = None

    frames_seq = [buf_f[:, -1]]
    log_probs, entropies, values, rews, ends = [], [], [], [], []

    for _ in range(horizon):
        logits, val, h_ac = actor_critic.step(buf_f[:, -1], h_ac)
        dist = torch.distributions.Categorical(logits=logits)
        act = dist.sample()
        log_probs.append(dist.log_prob(act))
        entropies.append(dist.entropy())
        values.append(val)

        a_win = torch.cat([buf_a, act.unsqueeze(1)], dim=1) if m > 1 else act.unsqueeze(1)
        with torch.no_grad():
            nxt_f = sample_next_frame(wm, buf_f, a_win, n_steps=n_denoise)
            r_hat, end_logit, h_re = rew_end(buf_f[:, -1:].contiguous(), act.unsqueeze(1), h_re)
            d_hat = (torch.sigmoid(end_logit) > 0.5).float()

        rews.append(r_hat.squeeze(1))
        ends.append(d_hat.squeeze(1))
        frames_seq.append(nxt_f)
        buf_f = torch.cat([buf_f[:, 1:], nxt_f.unsqueeze(1)], dim=1)
        if m > 1:
            buf_a = a_win[:, 1:]

    _, last_val, _ = actor_critic.step(buf_f[:, -1], h_ac)
    values.append(last_val)
    return (
        torch.stack(frames_seq, dim=1),
        torch.stack(log_probs, dim=1),
        torch.stack(entropies, dim=1),
        torch.stack(values, dim=1),
        torch.stack(rews, dim=1),
        torch.stack(ends, dim=1),
    )


# ==================== 第二部分：手算验证与 DIAMOND 在线训练 ====================


def verify_section_4_hand_calcs():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 4.3–4.4 节手算")
    print("=" * 60)
    # 4.3 节：σ_0=5, σ_1=0.283, σ_data=0.5, δ=0.05
    s0, s1, sd, delta = 5.0, 0.283, 0.5, 0.05
    c_out = (s0 * sd) / math.sqrt(s0 * s0 + sd * sd)
    e_eps = (1.0 - s1 / s0) * (s0 * delta)
    e_edm = (1.0 - s1 / s0) * (c_out * delta)
    print(f"  σ_0=5 第一步误差对比：预测噪声 = {e_eps:.3f} ({e_eps/sd*100:.1f}% σ_data)，EDM = {e_edm:.3f} ({e_edm/sd*100:.1f}% σ_data)，差 {e_eps/e_edm:.1f}× ✓")

    # 4.4 节：λ-return 与漏球对比
    vals = torch.tensor([[0.0, 0.5, 0.2, 0.4]])
    ends = torch.zeros(1, 3)
    lam_ok = lambda_returns(torch.tensor([[0.0, 1.0, 0.0]]), ends, vals, gamma=0.99, lam=0.95)[0, 0].item()
    lam_miss = lambda_returns(torch.tensor([[0.0, 0.0, 0.0]]), ends, vals, gamma=0.99, lam=0.95)[0, 0].item()
    print(f"  λ-return（γ=0.99, λ=0.95）：保住小球 Λ_0 = {lam_ok:.3f}（手算 1.325），漏掉小球 Λ_0 = {lam_miss:.3f}（手算 0.384）✓")
    assert abs(lam_ok - 1.325) < 0.01 and abs(lam_miss - 0.384) < 0.01


def collect_episodes(env, policy_fn, n_episodes):
    eps = []
    for _ in range(n_episodes):
        obs = env.reset()  # [3, 32, 32] in [0, 1]
        frames = [obs * 2.0 - 1.0]
        acts, rews, dones = [], [], []
        h = None
        done = False
        while not done:
            a, h = policy_fn(frames[-1], h)
            nxt, r, done = env.step(a)
            acts.append(a)
            rews.append(r)
            dones.append(float(done))
            frames.append(nxt * 2.0 - 1.0)
        eps.append({
            "frames": np.asarray(frames, dtype=np.float32),  # [T+1, 3, 32, 32] in [-1, 1]
            "actions": np.asarray(acts, dtype=np.int64),
            "rewards": np.asarray(rews, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
            "return": float(sum(rews)),
        })
    return eps


def sample_segments(episodes, batch_size, ctx_len=2):
    ctxs, acts, tgts, rews, dones = [], [], [], [], []
    for _ in range(batch_size):
        ep = episodes[np.random.randint(len(episodes))]
        T = len(ep["actions"])
        t = np.random.randint(0, max(1, T - ctx_len + 1))
        ctxs.append(ep["frames"][t : t + ctx_len])
        acts.append(ep["actions"][t : t + ctx_len])
        tgts.append(ep["frames"][t + ctx_len])
        rews.append(ep["rewards"][t + ctx_len - 1])
        dones.append(ep["dones"][t + ctx_len - 1])
    return (
        torch.from_numpy(np.asarray(ctxs)),
        torch.from_numpy(np.asarray(acts)),
        torch.from_numpy(np.asarray(tgts)),
        torch.from_numpy(np.asarray(rews)),
        torch.from_numpy(np.asarray(dones)),
    )


def eval_policy(env, actor_critic, n_episodes=15):
    actor_critic.eval()
    def pol(f_m11, h):
        with torch.no_grad():
            logits, _, h2 = actor_critic.step(torch.from_numpy(f_m11).unsqueeze(0), h)
            return int(logits.argmax(dim=-1).item()), h2
    eps = collect_episodes(env, pol, n_episodes)
    return float(np.mean([e["return"] for e in eps]))


def train_diamond(quick=False):
    print("\n" + "=" * 60)
    print("第二部分：在 PixelCatchEnv (32×32) 上运行缩小版 DIAMOND")
    print("=" * 60)
    env = PixelCatchEnv(seed=SEED)
    eval_env = PixelCatchEnv(seed=SEED + 100)
    ctx_len = 2

    def rand_pol(_f, h):
        return int(np.random.randint(0, 3)), h

    replay = collect_episodes(env, rand_pol, n_episodes=20 if quick else 60)
    sigma_data = float(np.concatenate([e["frames"] for e in replay[:10]], axis=0).std())
    wm = FrameDenoiser(img_ch=3, ctx_len=ctx_len, act_dim=1, base_ch=24, cond_dim=48,
                       sigma_data=sigma_data, discrete_actions=3, use_edm=True)
    rew_end = RewardEndModel(n_actions=3)
    ac = ActorCritic(n_actions=3)

    opt_wm = torch.optim.Adam(wm.parameters(), lr=1e-3)
    opt_re = torch.optim.Adam(rew_end.parameters(), lr=1e-3)
    opt_ac = torch.optim.Adam(ac.parameters(), lr=1e-3)

    n_rounds = 3 if quick else 8
    wm_steps = 40 if quick else 160
    re_steps = 25 if quick else 90
    ac_steps = 15 if quick else 55
    horizon = 6 if quick else 10
    collect_per_round = 8 if quick else 20

    history = {"env_steps": [], "real_ret": [], "imag_ret": [], "wm_loss": []}
    total_env_steps = sum(len(e["actions"]) for e in replay)

    for rd in range(1, n_rounds + 1):
        wm.train(); rew_end.train(); ac.train()
        # 1. 更新扩散世界模型
        wl = 0.0
        for _ in range(wm_steps):
            ctx, act, tgt, _, _ = sample_segments(replay, 24, ctx_len=ctx_len)
            opt_wm.zero_grad()
            loss = wm.loss(tgt, ctx, act)
            loss.backward()
            nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            opt_wm.step()
            wl = loss.item()

        # 2. 更新奖励 / 终止模型
        for _ in range(re_steps):
            # 取完整小片段训练 GRU
            ep = replay[np.random.randint(len(replay))]
            f_seq = torch.from_numpy(ep["frames"][:-1]).unsqueeze(0)
            a_seq = torch.from_numpy(ep["actions"]).unsqueeze(0)
            r_seq = torch.from_numpy(ep["rewards"]).unsqueeze(0)
            d_seq = torch.from_numpy(ep["dones"]).unsqueeze(0)
            r_hat, d_logit, _ = rew_end(f_seq, a_seq)
            re_loss = F.mse_loss(r_hat, r_seq) + F.binary_cross_entropy_with_logits(d_logit, d_seq)
            opt_re.zero_grad(); re_loss.backward(); opt_re.step()

        # 3. 在扩散世界模型的想象中训练 Actor-Critic
        imag_rets = []
        for _ in range(ac_steps):
            ctx, act, _, _, _ = sample_segments(replay, 12, ctx_len=ctx_len)
            _, lp, ent, vals, r_im, d_im = imagine(wm, rew_end, ac, ctx, act[:, :-1], horizon=horizon, n_denoise=3)
            loss_ac, _ = actor_critic_loss(lp, ent, vals, r_im, d_im)
            opt_ac.zero_grad(); loss_ac.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 1.0); opt_ac.step()
            imag_rets.append(float(r_im.sum(dim=1).mean().item()))

        real_ret = eval_policy(eval_env, ac, n_episodes=10 if quick else 20)
        mean_im_ret = float(np.mean(imag_rets))
        history["env_steps"].append(total_env_steps)
        history["real_ret"].append(real_ret)
        history["imag_ret"].append(mean_im_ret)
        history["wm_loss"].append(wl)
        print(f"  [Round {rd}/{n_rounds}] 真实步数={total_env_steps:5d} | WM loss={wl:.4f} | 想象回报={mean_im_ret:+.3f} | 真实评估回报={real_ret:+.3f}")

        # 4. 用当前策略采集新数据
        ac.eval()
        def cur_pol(f_m11, h):
            with torch.no_grad():
                logits, _, h2 = ac.step(torch.from_numpy(f_m11).unsqueeze(0), h)
                a = int(torch.distributions.Categorical(logits=logits).sample().item())
                return a, h2
        new_eps = collect_episodes(env, cur_pol, n_episodes=collect_per_round)
        replay.extend(new_eps)
        total_env_steps += sum(len(e["actions"]) for e in new_eps)

    # 对比 1 步 vs 3 步去噪的想象画面与小物体（红色小球）保真度
    ctx, act, tgt, _, _ = sample_segments(replay, 24, ctx_len=ctx_len)
    init_eps = torch.randn_like(tgt)
    pred1 = sample_next_frame(wm, ctx, act, n_steps=1, init_eps=init_eps)
    pred3 = sample_next_frame(wm, ctx, act, n_steps=3, init_eps=init_eps)
    p1, p3 = psnr_from_m11(pred1, tgt), psnr_from_m11(pred3, tgt)
    # 红色小球在 R 通道亮而 G/B 暗，用 (R - G) 衡量小球峰值保真
    def ball_peak(x):
        x01 = (x + 1.0) * 0.5
        return float((x01[:, 0] - x01[:, 1]).amax(dim=(1, 2)).mean().item())
    print(f"\n  1 步 vs 3 步去噪对比（真实帧小球 R-G 峰值 = {ball_peak(tgt):.3f}）：")
    print(f"    1 步去噪：PSNR = {p1:.2f} dB，小球峰值 = {ball_peak(pred1):.3f}")
    print(f"    3 步去噪：PSNR = {p3:.2f} dB，小球峰值 = {ball_peak(pred3):.3f}")
    return history, tgt, pred1, pred3


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()
    verify_section_4_hand_calcs()
    hist, tgt, pred1, pred3 = train_diamond(quick=quick)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    axes[0, 0].plot(hist["env_steps"], hist["real_ret"], "o-", label="真实评估回报")
    axes[0, 0].plot(hist["env_steps"], hist["imag_ret"], "s--", label="想象回报")
    axes[0, 0].set_title("DIAMOND 学习曲线（PixelCatchEnv）")
    axes[0, 0].set_xlabel("真实环境步数"); axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(hist["env_steps"], hist["wm_loss"], "o-", color="tab:orange")
    axes[0, 1].set_title("扩散世界模型损失"); axes[0, 1].set_xlabel("真实环境步数")

    for col, (title, img) in enumerate([
        ("真实下一帧", tgt[0]),
        ("DIAMOND 1 步去噪", pred1[0]),
    ], start=2):
        axes[0, col].imshow(((img.permute(1, 2, 0).numpy() + 1) * 0.5).clip(0, 1))
        axes[0, col].set_title(title); axes[0, col].axis("off")

    for col in range(4):
        if col < 2:
            img = pred3[col]
            title = f"DIAMOND 3 步去噪 #{col+1}"
        else:
            img = tgt[col - 2]
            title = f"对照真实帧 #{col-1}"
        axes[1, col].imshow(((img.permute(1, 2, 0).numpy() + 1) * 0.5).clip(0, 1))
        axes[1, col].set_title(title); axes[1, col].axis("off")

    fig.tight_layout()
    save_figure(fig, OUT / "04-diamond.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：05-autoregressive-drift-and-diffusion-forcing.py")


if __name__ == "__main__":
    main()
