"""
自回归想象的工程：KV Cache 精确性证明、实现与速度账
===================================================

学习目标：
    1. 核对 CONCEPT.md 第 4.3 节的手算例子：
       - K=4, H=3 时：无缓存 138 次 token 计算 / 1802 对注意力 vs 有缓存 18 次 / 171 对（省 7.7× / 10.5×）
       - IRIS 规模（10 层, n=340, D=256）单条轨迹 KV cache ≈ 6.96 MB（float32）
    2. 实现清晰的 KVCache 与 imagine_step()，并用 `torch.allclose` 严格验证"整段前向 vs 增量 KV Cache"输出完全一致（仅差浮点舍入误差 ~1e-5）
    3. 实测同一个 Transformer 想象 20 步（K=16，共 340 个 token 位置）在无缓存与有缓存下的墙上耗时与每秒想象步数

核心概念：
    - 前缀不变性（prefix invariance）：因果掩码 + 逐位置操作 + 固定位置索引 → KV Cache 数学上精确无损
    - 算力换显存：把 O(n²) 的 token 计算降到 O(n)，但每帧 K+1 次串行前向次数不变

前置知识：
    - 03-gpt-as-world-model.py

对应概念文档：
    - CONCEPT.md 第 4 节「自回归想象的工程：KV cache 与速度账」

预计运行时间：约 15 秒（CPU）
"""

import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec03 = importlib.util.spec_from_file_location("p8_03", os.path.join(_DIR, "03-gpt-as-world-model.py"))
_mod03 = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(_mod03)

WorldModelGPT = _mod03.WorldModelGPT

plt = setup_matplotlib()
OUT = output_dir("phase-08")
SEED = 0


# ==================== 第一部分：KVCache 包装与 imagine_step ====================


class KVCache:
    """管理多层因果 Transformer 的键值缓存与当前位置偏移。"""

    def __init__(self):
        self.past_kv = None
        self.seq_len = 0

    def reset(self):
        self.past_kv = None
        self.seq_len = 0

    @torch.no_grad()
    def append_and_forward(self, model, new_emb):
        logits, rew, new_kv = model.forward_embeddings(new_emb, past_kv=self.past_kv, pos_offset=self.seq_len)
        self.past_kv = new_kv
        self.seq_len += new_emb.shape[1]
        return logits, rew


@torch.no_grad()
def imagine_step(model, cache, action_vec, greedy=True):
    """CONCEPT.md 第 4.6 节：给定动作，利用 KV Cache 逐 token 自回归生成下一帧的 K 个 token 与奖励。"""
    K = model.tokens_per_frame
    a_emb = model.act_emb(action_vec).unsqueeze(1)  # [B, 1, D]
    logits, rew_pred = cache.append_and_forward(model, a_emb)
    reward = rew_pred[:, -1]

    gen_tokens = []
    for k in range(K):
        next_tok = logits[:, -1].argmax(dim=-1) if greedy else torch.distributions.Categorical(logits=logits[:, -1]).sample()
        gen_tokens.append(next_tok)
        tok_emb = model.tok_emb(next_tok).unsqueeze(1)
        logits, _ = cache.append_and_forward(model, tok_emb)
    return torch.stack(gen_tokens, dim=1), reward  # [B, K], [B]


# ==================== 第二部分：手算验证与精确性单元测试 ====================


def verify_hand_calcs_and_exactness():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 4.3 节手算与 KV Cache 数值精确性单元测试")
    print("=" * 60)

    # 1. K=4, H=3 手算
    lens = [5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18]
    tok_no, attn_no = sum(lens), sum(l * l for l in lens)
    tok_yes, attn_yes = 18, 18 * 19 // 2
    iris_mb = 2 * 10 * 340 * 256 * 4 / (1024**2)
    print(f"  1. K=4, H=3：无缓存 ({tok_no} tokens, {attn_no} pairs) vs 有缓存 ({tok_yes}, {attn_yes}) → 省 {tok_no/tok_yes:.1f}× / {attn_no/attn_yes:.1f}× ✓")
    print(f"  2. IRIS 缓存显存（10层, n=340, D=256, fp32）：单条 {iris_mb:.2f} MB，64 条并行 = {iris_mb * 64:.1f} MB ✓")
    assert tok_no == 138 and attn_no == 1802 and tok_yes == 18 and attn_yes == 171

    # 2. 前缀不变性数值单元测试
    model = WorldModelGPT(vocab_size=128, tokens_per_frame=16, act_dim=2, discrete_actions=False,
                          d_model=128, n_layers=4, n_heads=4, max_seq_len=400).eval()
    tokens = torch.randint(0, 128, (2, 4, 16))
    actions = torch.randn(2, 4, 2)
    emb_full = model.embed_sequence(tokens, actions)  # [2, 68, 128]

    with torch.no_grad():
        logits_full, _, _ = model.forward_embeddings(emb_full, past_kv=None, pos_offset=0)
        cache = KVCache()
        # 先喂入前 30 个位置，再逐 token 喂入剩下 38 个位置
        l_pref, _ = cache.append_and_forward(model, emb_full[:, :30])
        inc_list = [l_pref]
        for pos in range(30, emb_full.shape[1]):
            l_step, _ = cache.append_and_forward(model, emb_full[:, pos : pos + 1])
            inc_list.append(l_step)
        logits_inc = torch.cat(inc_list, dim=1)

    max_diff = (logits_full - logits_inc).abs().max().item()
    print(f"  3. 前缀不变性单元测试（序列长 68）：整段前向 vs 增量 KV Cache 最大绝对差值 = {max_diff:.2e} ✓")
    assert max_diff < 1e-4
    return max_diff


# ==================== 第三部分：实测 20 步自回归想象速度账 ====================


@torch.no_grad()
def benchmark_imagination(batch_size=8, horizon=20, tokens_per_frame=16):
    print("\n" + "=" * 60)
    print(f"第二部分：实测同一个 Transformer 想象 {horizon} 步的耗时对比（B={batch_size}, K={tokens_per_frame}, 4层 D=128）")
    print("=" * 60)

    model = WorldModelGPT(vocab_size=128, tokens_per_frame=tokens_per_frame, act_dim=2, discrete_actions=False,
                          d_model=128, n_layers=4, n_heads=4, max_seq_len=512).eval()
    init_toks = torch.randint(0, 128, (batch_size, 1, tokens_per_frame))
    actions = torch.randn(batch_size, horizon, 2)

    # 方法 A：有 KV Cache
    t0 = time.time()
    cache = KVCache()
    cache.append_and_forward(model, model.tok_emb(init_toks[:, 0]))
    frames_cached = []
    for t in range(horizon):
        f_t, _ = imagine_step(model, cache, actions[:, t], greedy=True)
        frames_cached.append(f_t)
    dt_cached = time.time() - t0
    frames_cached = torch.stack(frames_cached, dim=1)

    # 方法 B：无 KV Cache（每采样一个 token 都把整条前缀重算一遍）
    t1 = time.time()
    seq_emb = model.tok_emb(init_toks[:, 0])  # [B, K, D]
    frames_nocache = []
    for t in range(horizon):
        seq_emb = torch.cat([seq_emb, model.act_emb(actions[:, t]).unsqueeze(1)], dim=1)
        f_toks = []
        for _ in range(tokens_per_frame):
            logits, _, _ = model.forward_embeddings(seq_emb, past_kv=None, pos_offset=0)
            nxt = logits[:, -1].argmax(dim=-1)
            f_toks.append(nxt)
            seq_emb = torch.cat([seq_emb, model.tok_emb(nxt).unsqueeze(1)], dim=1)
        frames_nocache.append(torch.stack(f_toks, dim=1))
    dt_nocache = time.time() - t1
    frames_nocache = torch.stack(frames_nocache, dim=1)

    exact_match = bool(torch.equal(frames_cached, frames_nocache))
    steps_per_sec_cache = (batch_size * horizon) / dt_cached
    steps_per_sec_nocache = (batch_size * horizon) / dt_nocache
    speedup = dt_nocache / dt_cached
    print(f"  生成结果逐 token 完全一致？{exact_match} ✓")
    print(f"  无 KV Cache：{dt_nocache*1000:7.1f} ms（{steps_per_sec_nocache:6.1f} 想象步/秒）")
    print(f"  有 KV Cache：{dt_cached*1000:7.1f} ms（{steps_per_sec_cache:6.1f} 想象步/秒，提速 {speedup:.1f}×！）")
    assert exact_match

    # 扫描想象视野 H 看二次 vs 线性时间增长
    hs = [5, 10, 15, 20] if not is_quick() else [4, 8]
    t_no_list, t_yes_list = [], []
    for h in hs:
        tc0 = time.time()
        c = KVCache()
        c.append_and_forward(model, model.tok_emb(init_toks[:, 0]))
        for step_i in range(h):
            imagine_step(model, c, actions[:, step_i], greedy=True)
        t_yes_list.append((time.time() - tc0) * 1000.0)

        tn0 = time.time()
        se = model.tok_emb(init_toks[:, 0])
        for step_i in range(h):
            se = torch.cat([se, model.act_emb(actions[:, step_i]).unsqueeze(1)], dim=1)
            for _ in range(tokens_per_frame):
                lg, _, _ = model.forward_embeddings(se)
                nt = lg[:, -1].argmax(dim=-1)
                se = torch.cat([se, model.tok_emb(nt).unsqueeze(1)], dim=1)
        t_no_list.append((time.time() - tn0) * 1000.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(hs, t_no_list, "s--", color="tomato", linewidth=2.2, label="无 KV Cache（三次/二次增长）")
    ax.plot(hs, t_yes_list, "o-", color="seagreen", linewidth=2.4, label=f"有 KV Cache（H={horizon} 提速 {speedup:.1f}×）")
    ax.set_xlabel("想象视野 H（帧数，每帧 16+1 个 token）")
    ax.set_ylabel(f"批量想象 {batch_size} 条轨迹耗时 (ms)")
    ax.set_title("KV Cache 将自回归想象复杂度降低一阶且输出零误差")
    ax.legend(fontsize=9)
    save_figure(fig, OUT / "04-kv-cache-speedup.png")


def main():
    seed_everything(SEED)
    verify_hand_calcs_and_exactness()
    benchmark_imagination(batch_size=4 if is_quick() else 8, horizon=8 if is_quick() else 20)
    print("\n" + "=" * 60)
    print("本节完成！下一节：05-iris.py（IRIS 完整闭环：Tokenizer + Transformer + 想象 Actor-Critic）")
    print("=" * 60)


if __name__ == "__main__":
    main()
