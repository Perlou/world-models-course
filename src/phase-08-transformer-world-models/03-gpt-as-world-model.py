"""
把 GPT 改成世界模型：帧 Token 与动作交错排列
===========================================

学习目标：
    1. 核对 CONCEPT.md 第 3.4 节手算例子（K=2, N=3，NLL = 2.408，平均 0.602 vs 均匀 ln 3 = 1.099；序列长 20×17 = 340）
    2. 实现 interleave_tokens()、支持 KV cache 的因果 WorldModelGPT 以及精确掩码对齐的 world_model_loss()
       （只在帧 token 目标位置算交叉熵，不在"预测动作"位置算损失；在动作位置预测奖励）
    3. 在真实 PushT 演示轨迹（经 02 节 4×4 FSQ 编码成 K=16, N=125 token 序列）上训练 WorldModelGPT，
       记录交叉熵从 ln 125 ≈ 4.83 下降的曲线、teacher forcing 下一 token 准确率，以及开环展开 1–10 步的 token 准确率衰退

核心概念：
    - 扮演环境的聊天模型：[z_0^1 … z_0^K, a_0, z_1^1 … z_1^K, a_1, …]
    - 位置分工：动作位置预测下一帧第 1 个 token 及奖励；每帧最后一个 token 位置不计 token 损失
    - 曝光偏差（exposure bias）：训练用 teacher forcing 并行无漂移，开环想象逐 token 自回归会累积误差

前置知识：
    - 02-fsq-and-tokenizer-quality.py

对应概念文档：
    - CONCEPT.md 第 3 节「把世界模型写成序列模型」

预计运行时间：约 2 分钟（CPU）
"""

import importlib.util
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.pusht import load_pusht_demos

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec02 = importlib.util.spec_from_file_location("p8_02", os.path.join(_DIR, "02-fsq-and-tokenizer-quality.py"))
_mod02 = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(_mod02)

FSQTokenizer = _mod02.FSQTokenizer

plt = setup_matplotlib()
OUT = output_dir("phase-08")
SEED = 0


# ==================== 第一部分：交错拼接与支持 KV Cache 的因果 GPT ====================


def interleave_tokens(token_emb, action_emb):
    """CONCEPT.md 第 3.7 节：把 [B, T, K, D] 与 [B, T, 1, D] 交错拼成 [B, T*(K+1), D]。"""
    return torch.cat([token_emb, action_emb], dim=2).flatten(1, 2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, kv_cache=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if kv_cache is not None:
            k_past, v_past = kv_cache
            past_len = k_past.shape[2]
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)
        new_cache = (k, v)

        total_len = k.shape[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 因果掩码：新查询位置 i（全局下标 past_len + i）只能看 <= past_len + i 的键
        q_pos = torch.arange(past_len, past_len + L, device=x.device).unsqueeze(1)
        k_pos = torch.arange(total_len, device=x.device).unsqueeze(0)
        causal_mask = k_pos > q_pos
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(out), new_cache


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x, kv_cache=None):
        a_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + a_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class WorldModelGPT(nn.Module):
    """支持连续或离散动作、自带逐层 KV Cache 的因果 Transformer 世界模型。"""

    def __init__(self, vocab_size, tokens_per_frame, act_dim, discrete_actions=False,
                 d_model=128, n_layers=4, n_heads=4, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.tokens_per_frame = tokens_per_frame
        self.discrete_actions = discrete_actions
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.act_emb = nn.Embedding(act_dim, d_model) if discrete_actions else nn.Linear(act_dim, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.token_head = nn.Linear(d_model, vocab_size)
        self.reward_head = nn.Linear(d_model, 1)

    def embed_sequence(self, tokens, actions):
        # tokens: [B, T, K], actions: [B, T] (discrete) or [B, T, A] (continuous)
        t_emb = self.tok_emb(tokens)  # [B, T, K, D]
        a_emb = self.act_emb(actions).unsqueeze(2)  # [B, T, 1, D]
        return interleave_tokens(t_emb, a_emb)  # [B, T*(K+1), D]

    def forward_embeddings(self, emb, past_kv=None, pos_offset=0):
        B, L, _ = emb.shape
        pos = torch.arange(pos_offset, pos_offset + L, device=emb.device)
        x = emb + self.pos_emb(pos).unsqueeze(0)
        new_kv = []
        for idx, blk in enumerate(self.blocks):
            kv_i = None if past_kv is None else past_kv[idx]
            x, kv_out = blk(x, kv_cache=kv_i)
            new_kv.append(kv_out)
        h = self.ln_f(x)
        return self.token_head(h), self.reward_head(h).squeeze(-1), new_kv

    def forward(self, tokens, actions):
        emb = self.embed_sequence(tokens, actions)
        return self.forward_embeddings(emb, past_kv=None, pos_offset=0)


def world_model_loss(model, tokens, actions, rewards):
    """CONCEPT.md 第 3.3、3.7 节：只在帧 token 目标位置算交叉熵，在动作位置预测奖励。"""
    B, T, K = tokens.shape
    logits, pred_rew_all, _ = model(tokens, actions)  # [B, T*(K+1), N]

    # 构建对齐目标：位置 i 预测位置 i+1；每帧最后一个 token（模 K+1 == K-1）与序列末尾不计 token 损失
    seq_len = T * (K + 1)
    flat_targets = torch.full((B, seq_len), -100, dtype=torch.long, device=tokens.device)
    for t in range(T):
        base = t * (K + 1)
        # 帧内：位置 base+k (0 <= k < K-1) 预测 tokens[:, t, k+1]
        for k in range(K - 1):
            flat_targets[:, base + k] = tokens[:, t, k + 1]
        # 动作位置 base+K 预测下一帧第 0 个 token tokens[:, t+1, 0]
        if t + 1 < T:
            flat_targets[:, base + K] = tokens[:, t + 1, 0]

    token_loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), flat_targets.reshape(-1), ignore_index=-100)
    with torch.no_grad():
        valid = flat_targets != -100
        preds = logits.argmax(dim=-1)
        acc = (preds[valid] == flat_targets[valid]).float().mean().item()

    act_positions = torch.arange(K, seq_len, K + 1, device=tokens.device)
    rew_loss = F.mse_loss(pred_rew_all[:, act_positions], rewards)
    return token_loss + rew_loss, token_loss.item(), rew_loss.item(), acc


# ==================== 第二部分：手算验证与 PushT 序列世界模型实验 ====================


def verify_section_3_hand_calc():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 3.4 节 NLL 与序列长度手算例子")
    print("=" * 60)
    probs = [0.5, 0.8, 0.9, 0.25]
    nll = sum(-math.log(p) for p in probs)
    avg = nll / len(probs)
    unif = math.log(3.0)
    print(f"  1. K=2, N=3：总 NLL = {nll:.3f}（手算 2.408），平均每 token = {avg:.3f}（手算 0.602；均匀基线 ln 3 = {unif:.3f}）✓")
    print(f"  2. IRIS 序列长度（T_c=20, K=16）：20 × 17 = {20 * 17}；K=64 时 = {20 * 65}（注意力贵 {(1300/340)**2:.1f} 倍 ✓）")
    assert abs(nll - 2.408) < 2e-3 and abs(avg - 0.602) < 2e-3


def build_pusht_token_episodes(n_episodes=24, frame_stride=2):
    """用 02 节的 4×4 FSQ tokenizer 把 PushT 真实演示编码为紧凑 token 轨迹。"""
    demos = load_pusht_demos()
    imgs = demos["images"]
    acts = demos["actions"].astype(np.float32) / 512.0 * 2.0 - 1.0  # 归一化到 [-1, 1]
    rews = demos["rewards"].astype(np.float32)
    starts = demos["episode_starts"]
    ends = demos["episode_ends"]

    tok_model = FSQTokenizer(levels=(5, 5, 5), grid="4x4")
    ckpt = OUT / "pusht_fsq_4x4.pt"
    if ckpt.exists():
        tok_model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    else:
        # 冒烟模式或未运行 02 节时快速拟合一个 4×4 FSQ
        sub = imgs[:: max(1, len(imgs) // 256)][:256]
        x_tr = torch.from_numpy(np.stack([cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA) for im in sub])).permute(0, 3, 1, 2).float() / 255.0
        opt_t = torch.optim.Adam(tok_model.parameters(), lr=3e-3)
        for _ in range(30):
            r_x, _ = tok_model(x_tr[torch.randint(0, len(x_tr), (32,))])
            loss_t = F.mse_loss(r_x, x_tr[:32])
            opt_t.zero_grad()
            loss_t.backward()
            opt_t.step()
    tok_model.eval()

    episodes = []
    with torch.no_grad():
        for ei in range(min(n_episodes, len(starts))):
            s, e = int(starts[ei]), int(ends[ei])
            sl = slice(s, e, frame_stride)
            im_ep = np.stack([cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA) for im in imgs[sl]])
            x_ep = torch.from_numpy(im_ep).permute(0, 3, 1, 2).float() / 255.0
            _, idx_ep = tok_model.encode(x_ep)  # [L, 4, 4]
            episodes.append((idx_ep.reshape(len(x_ep), -1).numpy().astype(np.int64), acts[sl], rews[sl]))
    return tok_model, episodes


@torch.no_grad()
def evaluate_open_loop_token_acc(model, val_eps, context_frames=2, rollout_steps=8):
    """给前 context_frames 帧真实 token，之后只喂真实动作自回归展开 rollout_steps 帧，测逐帧 token 准确率。"""
    model.eval()
    K = model.tokens_per_frame
    hits = np.zeros(rollout_steps, dtype=np.float64)
    total = 0

    for tok_ep, act_ep, _ in val_eps:
        if len(tok_ep) < context_frames + rollout_steps:
            continue
        cur_toks = torch.from_numpy(tok_ep[:context_frames][None])   # [1, C, K]
        cur_acts = torch.from_numpy(act_ep[:context_frames][None])   # [1, C, 2]
        # 逐帧自回归（带 KV cache）
        emb0 = model.embed_sequence(cur_toks, cur_acts)  # [1, C*(K+1), D]
        logits, _, kv = model.forward_embeddings(emb0, past_kv=None, pos_offset=0)
        pos = emb0.shape[1]

        for step_k in range(rollout_steps):
            pred_frame = []
            next_tok = logits[:, -1].argmax(dim=-1)  # [1]
            pred_frame.append(next_tok.item())
            for _ in range(K - 1):
                e_tok = model.tok_emb(next_tok).unsqueeze(1)
                logits, _, kv = model.forward_embeddings(e_tok, past_kv=kv, pos_offset=pos)
                pos += 1
                next_tok = logits[:, -1].argmax(dim=-1)
                pred_frame.append(next_tok.item())
            # 喂入该步的真实动作推进到下一帧
            true_frame = tok_ep[context_frames + step_k]
            hits[step_k] += float(np.mean(np.array(pred_frame) == true_frame))
            if step_k + 1 < rollout_steps:
                # 把刚生成的最后一个 token + 下一步真实动作送入缓存
                e_last_tok = model.tok_emb(next_tok).unsqueeze(1)
                a_next = torch.from_numpy(act_ep[context_frames + step_k : context_frames + step_k + 1][None])
                e_act = model.act_emb(a_next)
                e_pair = torch.cat([e_last_tok, e_act], dim=1)
                logits, _, kv = model.forward_embeddings(e_pair, past_kv=kv, pos_offset=pos)
                pos += 2
        total += 1
    return hits / max(1, total)


def main():
    seed_everything(SEED)
    verify_section_3_hand_calc()

    quick = is_quick()
    _, episodes = build_pusht_token_episodes(n_episodes=6 if quick else 28, frame_stride=2)
    n_tr = max(1, int(len(episodes) * 0.8))
    tr_eps, val_eps = episodes[:n_tr], episodes[n_tr:]

    K = 16
    V = 125
    T_chunk = 8 if quick else 12
    n_steps = 30 if quick else 220

    model = WorldModelGPT(vocab_size=V, tokens_per_frame=K, act_dim=2, discrete_actions=False,
                          d_model=128, n_layers=3, n_heads=4, max_seq_len=400)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    rng = np.random.default_rng(SEED)

    ce_hist, acc_hist = [], []
    print("\n" + "=" * 60)
    print(f"第二部分：PushT 上训练因果 Transformer 世界模型（N={V}, K={K}, T={T_chunk}, 初始理论 CE = ln {V} = {math.log(V):.2f}）")
    print("=" * 60)
    t0 = time.time()

    for step in range(1, n_steps + 1):
        model.train()
        tb, ab, rb = [], [], []
        for _ in range(12 if quick else 24):
            t_e, a_e, r_e = tr_eps[int(rng.integers(0, len(tr_eps)))]
            s = int(rng.integers(0, max(1, len(t_e) - T_chunk)))
            tb.append(t_e[s : s + T_chunk])
            ab.append(a_e[s : s + T_chunk])
            rb.append(r_e[s : s + T_chunk])
        loss, tok_ce, rew_mse, tf_acc = world_model_loss(
            model, torch.from_numpy(np.stack(tb)), torch.from_numpy(np.stack(ab)), torch.from_numpy(np.stack(rb))
        )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ce_hist.append(tok_ce)
        acc_hist.append(tf_acc * 100.0)
        if step == 1 or step % max(1, n_steps // 4) == 0 or step == n_steps:
            print(f"  步 {step:>3}/{n_steps}  token CE = {tok_ce:5.3f}  奖励 MSE = {rew_mse:.4f}  "
                  f"Teacher-Forcing 准确率 = {tf_acc*100:5.1f}%  用时 {time.time() - t0:4.1f}s")

    ol_steps = 5 if quick else 10
    ol_acc = evaluate_open_loop_token_acc(model, val_eps, context_frames=2, rollout_steps=ol_steps)
    print(f"\n  开环自回归展开 token 准确率（1..{ol_steps} 步）：" +
          "  ".join(f"d={i+1}:{a*100:.1f}%" for i, a in enumerate(ol_acc)))

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    axes[0].plot(ce_hist, linewidth=2.0, label="训练 token 交叉熵")
    axes[0].axhline(math.log(V), color="gray", linestyle=":", label=f"均匀随机基线 ln {V} = {math.log(V):.2f}")
    axes[0].set_xlabel("训练步数"); axes[0].set_ylabel("交叉熵 (nat)")
    axes[0].set_title(f"Token 世界模型训练曲线（末段 TF 准确率 {acc_hist[-1]:.1f}%）")
    axes[0].legend(fontsize=8.5)

    axes[1].plot(np.arange(1, ol_steps + 1), ol_acc * 100.0, "o-", color="C3", linewidth=2.2)
    axes[1].set_xlabel("开环展开步数 d"); axes[1].set_ylabel("预测 Token 准确率 (%)")
    axes[1].set_title("曝光偏差：开环自回归准确率随步数逐步下降")
    save_figure(fig, OUT / "03-gpt-world-model.png")

    print("\n" + "=" * 60)
    print("本节完成！下一节：04-kv-cache-and-imagination.py（KV Cache 精确性检验与速度账）")
    print("=" * 60)


if __name__ == "__main__":
    main()
