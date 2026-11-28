"""
MaskGIT：双向掩码建模与置信度迭代并行解码
=========================================

学习目标：
    1. 核对 CONCEPT.md 第 7.3–7.4 节余弦调度手算（K=16, S=4 → 剩余 [15, 12, 7, 0]，每轮揭开 [1, 3, 5, 7]；对比线性 [4, 4, 4, 4]）
    2. 实现 cosine_schedule()、linear_schedule()、双向条件 MaskGITTransformer、maskgit_loss() 与按置信度迭代揭开的 maskgit_decode()
    3. 在真实 PushT 演示数据（4×4 FSQ token）上训练条件下一帧 MaskGIT 世界模型，
       扫描解码迭代次数 S ∈ {1, 2, 4, 8, 16} 并对比余弦调度与线性调度的每帧耗时、token 准确率与解码画面 MSE

核心概念：
    - 双向掩码预测：以前帧 token 和动作为可见条件，随机掩蔽下一帧 γ(r)·K 个 token，只在被掩位置算交叉熵
    - 迭代轮数 S 的连续谱：S=1 退化为条件独立（STORM 式），S=K 退化为全串行链式法则（IRIS 式）
    - 余弦调度先少后多：早期大局未定少揭几个防矛盾，后期上下文充足大批并行揭开

前置知识：
    - 02、03 节（PushT FSQ token）

对应概念文档：
    - CONCEPT.md 第 7 节「MaskGIT：并行解码」

预计运行时间：约 1 分钟（CPU）
"""

import importlib.util
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec03 = importlib.util.spec_from_file_location("p8_03", os.path.join(_DIR, "03-gpt-as-world-model.py"))
_mod03 = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(_mod03)

build_pusht_token_episodes = _mod03.build_pusht_token_episodes

plt = setup_matplotlib()
OUT = output_dir("phase-08")
SEED = 0


# ==================== 第一部分：掩码调度与双向 MaskGIT 模型 ====================


def cosine_schedule(r):
    r = float(r)
    if r >= 1.0:
        return 0.0
    return max(0.0, math.cos(0.5 * math.pi * r))


def linear_schedule(r):
    return 1.0 - float(r)


class MaskGITDynamics(nn.Module):
    """以 (当前帧 K 个 token, 动作 a_t) 为条件，对下一帧被掩蔽的 K 个位置做双向 Transformer 预测。"""

    def __init__(self, vocab_size=125, tokens_per_frame=16, act_dim=2, d_model=128, n_layers=3, n_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size  # N 作为特殊 [MASK] 编号
        self.K = tokens_per_frame

        self.tok_emb = nn.Embedding(vocab_size + 1, d_model)
        self.act_emb = nn.Linear(act_dim, d_model)
        self.pos_emb = nn.Embedding(2 * tokens_per_frame + 1, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, prev_tokens, action_vec, masked_next_tokens):
        # prev_tokens: [B, K], action_vec: [B, 2], masked_next_tokens: [B, K]
        e_prev = self.tok_emb(prev_tokens)
        e_act = self.act_emb(action_vec).unsqueeze(1)
        e_next = self.tok_emb(masked_next_tokens)
        seq = torch.cat([e_prev, e_act, e_next], dim=1)  # [B, 2K+1, D]
        pos = torch.arange(seq.shape[1], device=seq.device)
        h = self.transformer(seq + self.pos_emb(pos).unsqueeze(0))
        return self.head(h[:, self.K + 1 :])  # 只取下一帧的 K 个位置 [B, K, N]


def maskgit_loss(model, prev_tokens, actions, next_tokens, schedule_fn=cosine_schedule):
    """CONCEPT.md 第 7.5 节：对每个样本随机采样 r ~ U(0, 1)，掩蔽 ⌈γ(r)·K⌉ 个位置并只在被掩处算交叉熵。"""
    B, K = next_tokens.shape
    device = next_tokens.device
    masked = next_tokens.clone()
    mask_bool = torch.zeros(B, K, dtype=torch.bool, device=device)

    for b in range(B):
        r = float(torch.rand(1).item())
        n_m = max(1, min(K, int(math.ceil(schedule_fn(r) * K))))
        perm = torch.randperm(K, device=device)[:n_m]
        mask_bool[b, perm] = True
        masked[b, perm] = model.mask_id

    logits = model(prev_tokens, actions, masked)  # [B, K, N]
    return F.cross_entropy(logits[mask_bool], next_tokens[mask_bool])


@torch.no_grad()
def maskgit_decode(model, prev_tokens, actions, steps=4, schedule_fn=cosine_schedule, temperature=0.8):
    """CONCEPT.md 第 7.5 节：从全 [MASK] 开始，按置信度迭代并行揭开 S 轮。"""
    model.eval()
    B, K = prev_tokens.shape
    device = prev_tokens.device
    tokens = torch.full((B, K), model.mask_id, dtype=torch.long, device=device)
    mask = torch.ones(B, K, dtype=torch.bool, device=device)

    for s in range(1, steps + 1):
        logits = model(prev_tokens, actions, tokens)
        probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        sampled = torch.distributions.Categorical(probs=probs).sample() if temperature > 1e-3 else logits.argmax(dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        # 已揭开的位置置信度设为 +∞，永不被重新掩盖
        conf = conf.masked_fill(~mask, float("inf"))

        cur_masked = int(mask[0].sum().item())
        if s == steps:
            n_keep_masked = 0
        else:
            n_target = int(math.ceil(schedule_fn(s / steps) * K))
            # 边界保护：每轮至少揭开 1 个
            n_keep_masked = max(0, min(n_target, cur_masked - 1))

        tokens = torch.where(mask, sampled, tokens)
        if n_keep_masked > 0:
            low_idx = conf.topk(n_keep_masked, dim=-1, largest=False).indices
            new_mask = torch.zeros_like(mask)
            new_mask.scatter_(1, low_idx, True)
            tokens = tokens.masked_fill(new_mask, model.mask_id)
            mask = new_mask
        else:
            mask.fill_(False)
            break
    return tokens


# ==================== 第二部分：手算验证与 PushT 解码对比 ====================


def verify_section_7_hand_calc():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 7.3 节余弦调度 vs 线性调度手算")
    print("=" * 60)
    K, S = 16, 4
    cos_rem = [int(math.ceil(cosine_schedule(s / S) * K)) for s in range(1, S + 1)]
    lin_rem = [int(math.ceil(linear_schedule(s / S) * K)) for s in range(1, S + 1)]
    cos_rev = [K - cos_rem[0]] + [cos_rem[i - 1] - cos_rem[i] for i in range(1, S)]
    lin_rev = [K - lin_rem[0]] + [lin_rem[i - 1] - lin_rem[i] for i in range(1, S)]
    print(f"  K=16, S=4 余弦调度：剩余 {cos_rem}（手算 [15, 12, 7, 0]），每轮揭开 {cos_rev}（先少后多 ✓）")
    print(f"  K=16, S=4 线性调度：剩余 {lin_rem}（手算 [12,  8, 4, 0]），每轮揭开 {lin_rev} ✓")
    assert cos_rem == [15, 12, 7, 0] and lin_rem == [12, 8, 4, 0]


def main():
    seed_everything(SEED)
    verify_section_7_hand_calc()

    quick = is_quick()
    tok_model, episodes = build_pusht_token_episodes(n_episodes=6 if quick else 28, frame_stride=2)
    n_tr = max(1, int(len(episodes) * 0.8))

    # 收集所有 (z_t, a_t, z_{t+1}) 转移对
    def collect_pairs(eps):
        p_t, a_t, n_t = [], [], []
        for t_e, a_e, _ in eps:
            for i in range(len(t_e) - 1):
                p_t.append(t_e[i]); a_t.append(a_e[i]); n_t.append(t_e[i + 1])
        return torch.from_numpy(np.asarray(p_t)), torch.from_numpy(np.asarray(a_t)), torch.from_numpy(np.asarray(n_t))

    tr_p, tr_a, tr_n = collect_pairs(episodes[:n_tr])
    val_p, val_a, val_n = collect_pairs(episodes[n_tr:])
    val_sub = min(len(val_p), 64 if quick else 200)
    vp, va, vn = val_p[:val_sub], val_a[:val_sub], val_n[:val_sub]

    model = MaskGITDynamics(vocab_size=125, tokens_per_frame=16, act_dim=2, d_model=128, n_layers=3, n_heads=4)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    n_train = 40 if quick else 260

    print("\n" + "=" * 60)
    print(f"第二部分：在 PushT (4×4 FSQ token) 上训练 MaskGIT 并对比迭代轮数 S 与两种调度")
    print("=" * 60)
    for step in range(1, n_train + 1):
        model.train()
        idx = torch.randint(0, len(tr_p), (64,))
        loss = maskgit_loss(model, tr_p[idx], tr_a[idx], tr_n[idx])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 1 or step == n_train:
            print(f"  训练步 {step:>3}/{n_train}：掩码交叉熵 = {loss.item():.3f}")

    with torch.no_grad():
        true_imgs = tok_model.decode_tokens(vn.reshape(-1, 4, 4))

    s_list = [1, 2, 4, 8] if quick else [1, 2, 4, 8, 16]
    res = {"余弦调度": [], "线性调度": []}
    print(f"\n  {'调度':<10} {'S':>4} {'耗时(ms/帧)':>12} {'Token准确率':>12} {'解码画面MSE':>12}")
    for sched_name, sched_fn in [("余弦调度", cosine_schedule), ("线性调度", linear_schedule)]:
        for S in s_list:
            seed_everything(SEED)
            t0 = time.time()
            pred_toks = maskgit_decode(model, vp, va, steps=S, schedule_fn=sched_fn, temperature=0.5)
            ms_per_frame = (time.time() - t0) / len(vp) * 1000.0
            acc = (pred_toks == vn).float().mean().item() * 100.0
            with torch.no_grad():
                pred_imgs = tok_model.decode_tokens(pred_toks.reshape(-1, 4, 4))
                img_mse = F.mse_loss(pred_imgs, true_imgs).item()
            res[sched_name].append((S, ms_per_frame, acc, img_mse))
            print(f"  {sched_name:<10} {S:>4} {ms_per_frame:>11.2f} {acc:>11.1f}% {img_mse:>12.5f}")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
    for sn, rows in res.items():
        ss = [r[0] for r in rows]
        axes[0].plot(ss, [r[2] for r in rows], "o-", linewidth=2.2, label=sn)
        axes[1].plot(ss, [r[3] for r in rows], "o-", linewidth=2.2, label=sn)
        axes[2].plot(ss, [r[1] for r in rows], "o-", linewidth=2.2, label=sn)
    axes[0].set_title("下一帧 Token 准确率 (%)"); axes[0].set_xlabel("迭代次数 S"); axes[0].legend(fontsize=8.5)
    axes[1].set_title("解码画面 MSE（越低越好）"); axes[1].set_xlabel("迭代次数 S"); axes[1].legend(fontsize=8.5)
    axes[2].set_title("单帧并行解码耗时 (ms)"); axes[2].set_xlabel("迭代次数 S"); axes[2].legend(fontsize=8.5)
    save_figure(fig, OUT / "07-maskgit-decoding.png")

    print("\n" + "=" * 60)
    print("本节完成！下一节：08-rssm-vs-transformer.py（同一长时记忆任务上对决 GRU 与 Transformer）")
    print("=" * 60)


if __name__ == "__main__":
    main()
