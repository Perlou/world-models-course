"""
Phase 9 · 第 08 课：精简版 Genie —— 视频 Tokenizer + 隐动作模型 + MaskGIT 动力学
=============================================================================

学习目标：
    1. 严格按 Genie（Bruce et al. 2024）的两阶段顺序组装三大部件：
       ① 离散视频 Tokenizer（FSQ 4×4）
       ② 直接看像素的无监督隐动作模型 LAM（码本 M=4，训练后仅保留码本）
       ③ 时空分解 Transformer + MaskGIT 动力学模型（隐动作以加性嵌入注入）
    2. 在 PixelCatchEnv 上全程不使用动作标签完成训练，事后检验隐动作与真实动作的对应表及 ΔPSNR
    3. 测量 MaskGIT 并行解码轮数 R ∈ {1, 2, 4, 8} 对画面质量（PSNR）与每帧耗时（ms）的影响
    4. 提供提示图像 + 键盘序列驱动的 play() 交互推演示例

对应概念文档：
    src/phase-09-diffusion-video-world-models/CONCEPT.md 第 8 节

运行方式：
    uv run python src/phase-09-diffusion-video-world-models/08-mini-genie.py
    WM_QUICK=1 uv run python src/phase-09-diffusion-video-world-models/08-mini-genie.py

预计运行时间：完整约 2 分钟（M1 Max CPU），冒烟约 12 秒
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

_spec_p8_05 = importlib.util.spec_from_file_location(
    "p8_05_genie", os.path.join(_SRC, "phase-08-transformer-world-models", "05-iris.py")
)
_mod_p8_05 = importlib.util.module_from_spec(_spec_p8_05)
_spec_p8_05.loader.exec_module(_mod_p8_05)

_spec07 = importlib.util.spec_from_file_location("p9_07", os.path.join(_DIR, "07-latent-action-models.py"))
_mod07 = importlib.util.module_from_spec(_spec07)
_spec07.loader.exec_module(_mod07)

PixelCatchEnv = _mod_p8_05.PixelCatchEnv
MiniPixelTokenizer = _mod_p8_05.MiniPixelTokenizer
LatentActionModel = _mod07.LatentActionModel
psnr01 = _mod07.psnr01

plt = setup_matplotlib()
OUT = output_dir("phase-09")
SEED = 0


# ==================== 第一部分：时空分解 Transformer 与 MaskGIT 动力学 ====================


class STTransformerBlock(nn.Module):
    """CONCEPT.md 第 8.3 节：空间自注意力（帧内 S 个 token）+ 因果时间自注意力（跨 T 帧同空间位）+ FFN。"""

    def __init__(self, d_model=96, n_heads=4):
        super().__init__()
        self.norm_s = nn.LayerNorm(d_model)
        self.attn_s = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_t = nn.LayerNorm(d_model)
        self.attn_t = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_f = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))

    def forward(self, x):
        # x: [B, T, S, D]
        B, T, S, D = x.shape
        # 1. 空间注意力：每一帧内部 S 个位置双向交互
        xs = self.norm_s(x).reshape(B * T, S, D)
        hs, _ = self.attn_s(xs, xs, xs, need_weights=False)
        x = x + hs.reshape(B, T, S, D)

        # 2. 因果时间注意力：每个空间位置沿时间维只看过去与当前帧
        xt = self.norm_t(x).permute(0, 2, 1, 3).reshape(B * S, T, D)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        ht, _ = self.attn_t(xt, xt, xt, attn_mask=causal_mask, need_weights=False)
        x = x + ht.reshape(B, S, T, D).permute(0, 2, 1, 3)

        # 3. 前馈层
        return x + self.ffn(self.norm_f(x))


class DynamicsModel(nn.Module):
    """CONCEPT.md 第 8.2/8.4 节：Genie 动力学模型（加性隐动作嵌入 + ST-Transformer + MaskGIT）。"""

    def __init__(self, vocab_size=75, num_latent_actions=4, tokens_per_frame=16, max_frames=12,
                 d_model=96, n_layers=2, n_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_id = vocab_size
        self.S = tokens_per_frame
        self.tok_emb = nn.Embedding(vocab_size + 1, d_model)
        self.act_emb = nn.Embedding(num_latent_actions, d_model)
        self.pos_s = nn.Embedding(tokens_per_frame, d_model)
        self.pos_t = nn.Embedding(max_frames, d_model)
        self.blocks = nn.ModuleList([STTransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, latent_actions):
        """
        tokens:         [B, T, S]（最后一帧可能含 mask_id）
        latent_actions: [B, T-1]（以加性嵌入注入到对应下一帧 1..T-1 的全部 S 个位置）
        """
        B, T, S = tokens.shape
        device = tokens.device
        h = self.tok_emb(tokens)
        h = h + self.pos_s(torch.arange(S, device=device)).view(1, 1, S, -1)
        h = h + self.pos_t(torch.arange(T, device=device)).view(1, T, 1, -1)
        if T > 1:
            a_e = self.act_emb(latent_actions.long()).unsqueeze(2)  # [B, T-1, 1, D]
            h[:, 1:] = h[:, 1:] + a_e
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))  # [B, T, S, V]


def dynamics_loss(model, tokens, latent_actions):
    """随机掩蔽目标帧（最后一帧）的一部分 token，只对被掩位置算交叉熵。"""
    B, T, S = tokens.shape
    device = tokens.device
    inp = tokens.clone()
    mask_bool = torch.zeros(B, S, dtype=torch.bool, device=device)
    for b in range(B):
        r = float(torch.rand(1).item())
        n_m = max(1, min(S, int(math.ceil(math.cos(0.5 * math.pi * r) * S))))
        perm = torch.randperm(S, device=device)[:n_m]
        mask_bool[b, perm] = True
        inp[b, -1, perm] = model.mask_id
    logits = model(inp, latent_actions)[:, -1]  # [B, S, V]
    return F.cross_entropy(logits[mask_bool], tokens[:, -1][mask_bool])


@torch.no_grad()
def maskgit_generate_frame(model, history_tokens, latent_actions, n_rounds=4, temperature=0.5):
    """CONCEPT.md 第 8.6 节：给定历史 token [B, T_h, S] 与动作 [B, T_h]，R 轮并行生成下一帧 [B, S]。"""
    model.eval()
    B, T_h, S = history_tokens.shape
    device = history_tokens.device
    new_tok = torch.full((B, S), model.mask_id, dtype=torch.long, device=device)
    mask = torch.ones(B, S, dtype=torch.bool, device=device)

    for r_i in range(1, n_rounds + 1):
        seq = torch.cat([history_tokens, new_tok.unsqueeze(1)], dim=1)
        logits = model(seq, latent_actions)[:, -1]  # [B, S, V]
        probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
        sampled = torch.distributions.Categorical(probs=probs).sample() if temperature > 1e-2 else logits.argmax(-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).masked_fill(~mask, float("inf"))

        cur_m = int(mask[0].sum().item())
        if r_i == n_rounds:
            n_keep = 0
        else:
            frac = max(0.0, math.cos(0.5 * math.pi * (r_i / n_rounds)))
            n_keep = max(0, min(int(math.ceil(frac * S)), cur_m - 1))

        new_tok = torch.where(mask, sampled, new_tok)
        if n_keep > 0:
            low_idx = conf.topk(n_keep, dim=-1, largest=False).indices
            new_mask = torch.zeros_like(mask)
            new_mask.scatter_(1, low_idx, True)
            new_tok = new_tok.masked_fill(new_mask, model.mask_id)
            mask = new_mask
        else:
            break
    return new_tok


@torch.no_grad()
def play(tokenizer, dynamics, prompt_image, key_sequence, n_rounds=4):
    """CONCEPT.md 第 8.7 节：从单张提示图像出发，按按键（隐动作）序列逐帧生成交互画面。"""
    tokenizer.eval(); dynamics.eval()
    _, z0 = tokenizer.encode(prompt_image.unsqueeze(0))  # [1, S]
    hist = z0.unsqueeze(1)  # [1, 1, S]
    acts = []
    frames = [prompt_image.cpu()]
    for a_id in key_sequence:
        acts.append(int(a_id))
        a_t = torch.tensor([acts[-5:]], dtype=torch.long)
        h_in = hist[:, -5:]
        nxt_tok = maskgit_generate_frame(dynamics, h_in, a_t, n_rounds=n_rounds, temperature=0.0)
        hist = torch.cat([hist, nxt_tok.unsqueeze(1)], dim=1)
        img = tokenizer.decode_tokens(nxt_tok)[0].cpu()
        frames.append(img)
    return frames


def collect_videos(n_episodes=60, seed=0):
    env = PixelCatchEnv(seed=seed)
    vids, true_acts = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        fs, acts = [obs], []
        done = False
        while not done:
            a = int(np.random.randint(0, 3))
            nxt, _, done = env.step(a)
            fs.append(nxt); acts.append(a)
        vids.append(np.asarray(fs, dtype=np.float32))
        true_acts.append(np.asarray(acts, dtype=np.int64))
    return vids, true_acts


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    print("=" * 60)
    print("第一部分：验证时空分解注意力复杂度（CONCEPT.md 第 8.3 节）")
    print("=" * 60)
    T_demo, S_demo = 16, 256
    full_cost = (T_demo * S_demo) ** 2
    st_cost = T_demo * (S_demo ** 2) + S_demo * (T_demo ** 2)
    print(f"  T={T_demo}, S={S_demo}：全注意力 = {full_cost:,}，时空分解 = {st_cost:,}（节省 {full_cost/st_cost:.1f}× ✓）")

    print("\n" + "=" * 60)
    print("第二部分：分阶段训练精简版 Genie（无动作标签）")
    print("=" * 60)
    tr_vids, _ = collect_videos(n_episodes=25 if quick else 80, seed=SEED)
    te_vids, te_ta = collect_videos(n_episodes=8 if quick else 25, seed=SEED + 77)

    # 展平帧与相邻帧对用于阶段 1
    all_frames = torch.from_numpy(np.concatenate(tr_vids, axis=0))
    o_t = torch.from_numpy(np.concatenate([v[:-1] for v in tr_vids], axis=0))
    o_n = torch.from_numpy(np.concatenate([v[1:] for v in tr_vids], axis=0))

    # 阶段 1a：离散 Tokenizer (FSQ 4×4, V=75)
    tokenizer = MiniPixelTokenizer(levels=(5, 5, 3))
    opt_tok = torch.optim.Adam(tokenizer.parameters(), lr=3e-3)
    tok_iters = 80 if quick else 350
    for _ in range(tok_iters):
        idx = torch.randint(0, len(all_frames), (64,))
        opt_tok.zero_grad()
        rec, _ = tokenizer(all_frames[idx])
        F.mse_loss(rec, all_frames[idx]).backward()
        opt_tok.step()

    # 阶段 1b：像素级隐动作模型 LAM (M=4)
    lam = LatentActionModel(num_codes=4, act_dim=16)
    opt_lam = torch.optim.Adam(lam.parameters(), lr=2e-3)
    lam_iters = 100 if quick else 450
    for _ in range(lam_iters):
        idx = torch.randint(0, len(o_t), (64,))
        opt_lam.zero_grad()
        _, _, loss, _ = lam(o_t[idx], o_n[idx])
        loss.backward()
        opt_lam.step()

    # 冻结 tokenizer 和 LAM，给全部视频打 token + 隐动作标签
    tokenizer.eval(); lam.eval()
    @torch.no_grad()
    def encode_vids(vids):
        out = []
        for v in vids:
            vt = torch.from_numpy(v)
            _, toks = tokenizer.encode(vt)                       # [T, 16]
            _, la, _ = lam.encode_action(vt[:-1], vt[1:])        # [T-1]
            out.append((toks, la))
        return out

    tr_enc = encode_vids(tr_vids)
    te_enc = encode_vids(te_vids)

    # 阶段 2：MaskGIT 时空动力学模型
    dyn = DynamicsModel(vocab_size=tokenizer.fsq.num_codes, num_latent_actions=4,
                        tokens_per_frame=16, max_frames=8, d_model=96, n_layers=2, n_heads=4)
    opt_dyn = torch.optim.AdamW(dyn.parameters(), lr=2e-3)
    dyn_iters = 120 if quick else 600
    clip_T = 4
    for _ in range(dyn_iters):
        bt, ba = [], []
        for _ in range(32):
            toks, la = tr_enc[np.random.randint(len(tr_enc))]
            s = np.random.randint(0, len(toks) - clip_T)
            bt.append(toks[s : s + clip_T])
            ba.append(la[s : s + clip_T - 1])
        opt_dyn.zero_grad()
        loss = dynamics_loss(dyn, torch.stack(bt), torch.stack(ba))
        loss.backward()
        nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt_dyn.step()
    print(f"  三部件训练完成（Tokenizer {tok_iters} 步 / LAM {lam_iters} 步 / MaskGIT {dyn_iters} 步，共 {time.time() - t0:.1f}s）")

    # 评估 1：隐动作 vs 真实动作对应表
    tbl = np.zeros((3, 4), dtype=np.int64)
    for (_, la), ta in zip(te_enc, te_ta):
        for t_a, l_a in zip(ta, la.numpy()):
            tbl[t_a, l_a] += 1
    purity = float(tbl.max(axis=0).sum() / max(tbl.sum(), 1))
    print(f"\n  测试集隐动作与真实动作对应纯度：{purity*100:.1f}%")
    for i, nm in enumerate(["左 (0)", "停 (1)", "右 (2)"]):
        print(f"    {nm}: {tbl[i].tolist()}")

    # 评估 2：MaskGIT 解码轮数 R ∈ {1, 2, 4, 8} 的 PSNR、ΔPSNR 与耗时
    eval_hist, eval_la, eval_gt = [], [], []
    for v, (toks, la) in zip(te_vids, te_enc):
        for s in range(0, len(toks) - clip_T, 2):
            eval_hist.append(toks[s : s + clip_T - 1])
            eval_la.append(la[s : s + clip_T - 1])
            eval_gt.append(torch.from_numpy(v[s + clip_T - 1]))
    eval_hist = torch.stack(eval_hist[:48])
    eval_la = torch.stack(eval_la[:48])
    eval_gt = torch.stack(eval_gt[:48])

    rand_la = eval_la.clone()
    rand_la[:, -1] = torch.randint(0, 4, (len(eval_la),), generator=torch.Generator().manual_seed(42))

    print(f"\n  MaskGIT 并行解码轮数对比（批量 {len(eval_hist)} 帧）：")
    print(f"  {'轮数 R':>6s} | {'推断动作 PSNR':>13s} | {'随机动作 PSNR':>13s} | {'ΔPSNR':>7s} | {'单帧耗时(ms)':>12s}")
    round_stats = {}
    for R in [1, 2, 4, 8]:
        ts = time.time()
        gen_tok = maskgit_generate_frame(dyn, eval_hist, eval_la, n_rounds=R, temperature=0.0)
        with torch.no_grad():
            gen_img = tokenizer.decode_tokens(gen_tok)
        ms = (time.time() - ts) * 1000.0 / len(eval_hist)

        gen_tok_r = maskgit_generate_frame(dyn, eval_hist, rand_la, n_rounds=R, temperature=0.0)
        with torch.no_grad():
            gen_img_r = tokenizer.decode_tokens(gen_tok_r)

        p_inf = psnr01(gen_img, eval_gt)
        p_rnd = psnr01(gen_img_r, eval_gt)
        round_stats[R] = (p_inf, p_rnd, p_inf - p_rnd, ms)
        print(f"  {R:>6d} | {p_inf:13.2f} | {p_rnd:13.2f} | {p_inf - p_rnd:+7.2f} | {ms:12.2f}")

    # 演示 play()：同一张提示图，按不同隐动作序列推演 5 步
    prompt = torch.from_numpy(te_vids[0][2])
    play_frames_0 = play(tokenizer, dyn, prompt, [0, 0, 0, 0, 0], n_rounds=4)
    play_frames_1 = play(tokenizer, dyn, prompt, [1, 1, 1, 1, 1], n_rounds=4)

    fig, axes = plt.subplots(2, 6, figsize=(13, 4.8))
    for k in range(6):
        axes[0, k].imshow(play_frames_0[k].permute(1, 2, 0).numpy().clip(0, 1))
        axes[0, k].set_title(f"码 0 · t+{k}"); axes[0, k].axis("off")
        axes[1, k].imshow(play_frames_1[k].permute(1, 2, 0).numpy().clip(0, 1))
        axes[1, k].set_title(f"码 1 · t+{k}"); axes[1, k].axis("off")
    fig.suptitle("精简版 Genie 交互推演：同一张提示图施加不同隐动作序列")
    fig.tight_layout()
    save_figure(fig, OUT / "08-mini-genie.png")

    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：09-realtime-few-step-inference.py")


if __name__ == "__main__":
    main()
