"""
STORM 与 TWM：每帧一个类别隐变量 + 因果 Transformer
===================================================

学习目标：
    1. 核对 CONCEPT.md 第 6.3 节帧内条件独立手算例子（自回归 NLL 0.693 vs 条件独立 1.386，50% 不自洽组合）
    2. 实现 StormWorldModel（每帧并行 CNN 编码 16×16 类别隐变量 + 动作混合器 + 因果 Transformer + KV Cache）、
       storm_loss()（重建 + 奖励 + KL balancing + free bits）与 imagine()（每步仅 1 次 Transformer 调用、无需解码画面！）
    3. 在与 05-iris.py 完全相同的 PixelCatchEnv 上同条件对比 STORM、TWM 变体与 IRIS：
       实测每秒想象步数比值（预期 STORM 快约一个量级）与学习曲线

核心概念：
    - 每帧 1 个位置 vs 每帧 K+1 个位置：把串行自回归调用数从 K+1 降到 1，同时策略直接吃隐状态 [z_t, h_t] 省掉解码器
    - 帧内条件独立权衡：p(z_{t+1}|h_t) = ∏_i p(z_{t+1}^i | h_t)，以微弱多峰表达折损换来数量级加速
    - TWM 变体：策略只看 z_t，部署时无需跑 Transformer

前置知识：
    - Phase 6 的类别隐变量与 KL balancing（04、05 节）
    - 05-iris.py（PixelCatchEnv 与 IrisWorldModel）

对应概念文档：
    - CONCEPT.md 第 6 节「STORM 与 TWM：每帧一个隐变量」

预计运行时间：约 1 分钟（CPU）
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
_spec03 = importlib.util.spec_from_file_location("p8_03", os.path.join(_DIR, "03-gpt-as-world-model.py"))
_mod03 = importlib.util.module_from_spec(_spec03)
_spec03.loader.exec_module(_mod03)
_spec05 = importlib.util.spec_from_file_location("p8_05", os.path.join(_DIR, "05-iris.py"))
_mod05 = importlib.util.module_from_spec(_spec05)
_spec05.loader.exec_module(_mod05)

TransformerBlock = _mod03.TransformerBlock
PixelCatchEnv = _mod05.PixelCatchEnv
MiniPixelTokenizer = _mod05.MiniPixelTokenizer
IrisWorldModel = _mod05.IrisWorldModel
PixelActorCritic = _mod05.PixelActorCritic
imagine_rollout_iris = _mod05.imagine_rollout
compute_lambda_returns = _mod05.compute_lambda_returns_iris

plt = setup_matplotlib()
OUT = output_dir("phase-08")
SEED = 0


# ==================== 第一部分：StormWorldModel 与损失 ====================


def sample_cat_st(logits, unimix=0.01):
    probs = F.softmax(logits, dim=-1)
    probs = (1.0 - unimix) * probs + unimix / logits.shape[-1]
    idx = torch.distributions.Categorical(probs=probs).sample()
    oh = F.one_hot(idx, num_classes=logits.shape[-1]).to(probs.dtype)
    return oh + probs - probs.detach()


def cat_kl(lq, lp, unimix=0.01):
    q = (1.0 - unimix) * F.softmax(lq, -1) + unimix / lq.shape[-1]
    p = (1.0 - unimix) * F.softmax(lp, -1) + unimix / lp.shape[-1]
    return (q * (torch.log(q.clamp_min(1e-8)) - torch.log(p.clamp_min(1e-8)))).sum(dim=(-1, -2))


class StormWorldModel(nn.Module):
    """CONCEPT.md 第 6.2 节：每帧独立编码类别隐变量 z_t [N_z, K]，与动作混合后过因果 Transformer。"""

    def __init__(self, act_dim=3, n_z=16, n_c=16, d_model=96, n_layers=2, n_heads=4, max_len=64):
        super().__init__()
        self.act_dim = act_dim
        self.n_z = n_z
        self.n_c = n_c
        self.z_dim = n_z * n_c
        self.d_model = d_model

        self.enc = nn.Sequential(
            nn.Conv2d(3, 24, 4, 2, 1), nn.SiLU(),  # 8
            nn.Conv2d(24, 48, 4, 2, 1), nn.SiLU(), # 4
            nn.Flatten(),
            nn.Linear(48 * 4 * 4, self.z_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(self.z_dim, 48 * 4 * 4), nn.SiLU(),
            nn.Unflatten(1, (48, 4, 4)),
            nn.ConvTranspose2d(48, 24, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(24, 3, 4, 2, 1), nn.Sigmoid(),
        )
        self.action_mixer = nn.Sequential(nn.Linear(self.z_dim + act_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.dyn_head = nn.Linear(d_model, self.z_dim)
        self.rew_head = nn.Linear(d_model, 1)

    def encode_obs(self, obs):
        lead = obs.shape[:-3]
        x = obs.reshape(-1, *obs.shape[-3:])
        logits = self.enc(x).reshape(*lead, self.n_z, self.n_c)
        z = sample_cat_st(logits).reshape(*lead, self.z_dim)
        return z, logits

    def decode_z(self, z):
        lead = z.shape[:-1]
        return self.dec(z.reshape(-1, self.z_dim)).reshape(*lead, 3, 16, 16)

    def forward_transformer(self, e_seq, past_kv=None, pos_offset=0):
        B, L, _ = e_seq.shape
        pos = torch.arange(pos_offset, pos_offset + L, device=e_seq.device)
        x = e_seq + self.pos_emb(pos).unsqueeze(0)
        new_kv = []
        for idx, blk in enumerate(self.blocks):
            kv_i = None if past_kv is None else past_kv[idx]
            x, kv_o = blk(x, kv_cache=kv_i)
            new_kv.append(kv_o)
        h = self.ln_f(x)
        return h, new_kv

    def storm_loss(self, obs, actions, rewards):
        """CONCEPT.md 第 6.5 节：并行编码 + 因果 Transformer + 对齐切片 KL balancing + free bits。"""
        B, T = actions.shape
        z, post_logits = self.encode_obs(obs)  # [B, T, z_dim], [B, T, n_z, n_c]
        recon = self.decode_z(z)
        l_rec = F.mse_loss(recon, obs)

        a_oh = F.one_hot(actions.long(), num_classes=self.act_dim).float()
        e = self.action_mixer(torch.cat([z, a_oh], dim=-1))  # [B, T, D]
        h, _ = self.forward_transformer(e)                    # [B, T, D]

        prior_logits = self.dyn_head(h).reshape(B, T, self.n_z, self.n_c)
        pred_r = self.rew_head(h).squeeze(-1)
        l_rew = F.mse_loss(pred_r, rewards)

        # 关键时间对齐：h_t 预测 z_{t+1} → prior[:, :-1] 对齐 post[:, 1:]
        l_dyn = cat_kl(post_logits[:, 1:].detach(), prior_logits[:, :-1]).clamp_min(1.0).mean()
        l_rep = cat_kl(post_logits[:, 1:], prior_logits[:, :-1].detach()).clamp_min(1.0).mean()
        total = l_rec * 50.0 + l_rew + 0.5 * l_dyn + 0.1 * l_rep
        return total, z.detach(), h.detach()

    @torch.no_grad()
    def imagine(self, start_z, ac, horizon=8, use_h_in_policy=True):
        """CONCEPT.md 第 6.5 节：每步仅 1 次 Transformer 调用、无需解码画面！"""
        B = start_z.shape[0]
        z_cur = start_z
        h_cur = torch.zeros(B, self.d_model, device=start_z.device)
        past_kv = None
        pos = 0

        feats_list, act_list, rew_list = [], [], []
        for _ in range(horizon):
            feat = torch.cat([z_cur, h_cur], dim=-1) if use_h_in_policy else z_cur
            feats_list.append(feat)
            logits_a, _ = ac(feat)
            a = torch.distributions.Categorical(logits=logits_a).sample()
            a_oh = F.one_hot(a, num_classes=self.act_dim).float()
            e = self.action_mixer(torch.cat([z_cur, a_oh], dim=-1)).unsqueeze(1)
            h_out, past_kv = self.forward_transformer(e, past_kv=past_kv, pos_offset=pos)
            pos += 1
            h_cur = h_out[:, 0]
            rew = self.rew_head(h_cur).squeeze(-1)
            pr_l = self.dyn_head(h_cur).reshape(B, self.n_z, self.n_c)
            z_cur = sample_cat_st(pr_l).reshape(B, self.z_dim)
            act_list.append(a)
            rew_list.append(rew)

        feats_list.append(torch.cat([z_cur, h_cur], dim=-1) if use_h_in_policy else z_cur)
        return torch.stack(feats_list, dim=1), torch.stack(act_list, dim=1), torch.stack(rew_list, dim=1)


class LatentActorCritic(nn.Module):
    def __init__(self, in_dim, act_dim=3, hidden=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.pi = nn.Linear(hidden, act_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        return self.pi(h), self.v(h).squeeze(-1)


# ==================== 第二部分：手算核对与速度/学习对比 ====================


def verify_section_6_hand_calc():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 6.3 节帧内条件独立手算例子")
    print("=" * 60)
    nll_ar = -math.log(0.5)
    nll_ind = -math.log(0.25)
    print(f"  双二值同步分布 50%(0,0)+50%(1,1)：自回归 NLL = {nll_ar:.3f} nat ｜ "
          f"条件独立 NLL = {nll_ind:.3f} nat（多付 {nll_ind - nll_ar:.3f} nat，50% 采样落入非法 (0,1)/(1,0)）✓")
    assert abs(nll_ar - 0.6931) < 1e-3 and abs(nll_ind - 1.3863) < 1e-3


def benchmark_speed_iris_vs_storm():
    print("\n" + "=" * 60)
    print("第二部分：同模型宽度（D=96, 2 层）实测想象吞吐率 — IRIS vs STORM")
    print("=" * 60)
    B, H = 16, 15
    tok = MiniPixelTokenizer(levels=(5, 5, 3)).eval()
    iris = IrisWorldModel(vocab_size=75, tokens_per_frame=16, act_dim=3, d_model=96, n_layers=2, n_heads=4).eval()
    ac_iris = PixelActorCritic(act_dim=3).eval()

    storm = StormWorldModel(act_dim=3, n_z=16, n_c=16, d_model=96, n_layers=2, n_heads=4).eval()
    ac_storm = LatentActorCritic(storm.z_dim + storm.d_model, act_dim=3).eval()

    obs0 = torch.rand(B, 3, 16, 16)
    with torch.no_grad():
        z0, _ = storm.encode_obs(obs0)
        # 预热
        imagine_rollout_iris(tok, iris, ac_iris, obs0, horizon=4)
        storm.imagine(z0, ac_storm, horizon=4)

        n_rep = 2 if is_quick() else 4
        t0 = time.time()
        for _ in range(n_rep):
            imagine_rollout_iris(tok, iris, ac_iris, obs0, horizon=H)
        dt_iris = (time.time() - t0) / n_rep

        t1 = time.time()
        for _ in range(n_rep * 5):
            storm.imagine(z0, ac_storm, horizon=H)
        dt_storm = (time.time() - t1) / (n_rep * 5)

    ips_iris = (B * H) / dt_iris
    ips_storm = (B * H) / dt_storm
    ratio = ips_storm / ips_iris
    print(f"  IRIS  （K=16 逐 token 自回归 + 解码）：{ips_iris:7.1f} 想象步/秒（{dt_iris*1000:6.1f} ms / {B}×{H}步）")
    print(f"  STORM （每帧 1 位置 + 隐空间策略） ：{ips_storm:7.1f} 想象步/秒（{dt_storm*1000:6.1f} ms / {B}×{H}步）→ 快 {ratio:.1f} 倍！")
    return ips_iris, ips_storm, ratio


def train_storm_or_twm(use_h_in_policy=True, n_rounds=14, seed=0):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    env = PixelCatchEnv(seed=seed)
    wm = StormWorldModel(act_dim=3, n_z=16, n_c=16, d_model=96, n_layers=2, n_heads=4)
    in_d = (wm.z_dim + wm.d_model) if use_h_in_policy else wm.z_dim
    ac = LatentActorCritic(in_d, act_dim=3)
    opt_wm = torch.optim.Adam(wm.parameters(), lr=2e-3)
    opt_ac = torch.optim.Adam(ac.parameters(), lr=2e-3)

    replay = deque(maxlen=120)

    def collect(n_eps, use_pi=False):
        wm.eval(); ac.eval()
        for _ in range(n_eps):
            o = env.reset()
            eo, ea, er = [], [], []
            done = False
            past_kv = None
            h_cur = torch.zeros(1, wm.d_model)
            pos = 0
            while not done:
                if not use_pi:
                    a = int(rng.integers(0, 3))
                else:
                    with torch.no_grad():
                        z_t, _ = wm.encode_obs(torch.from_numpy(o[None]))
                        feat = torch.cat([z_t, h_cur], dim=-1) if use_h_in_policy else z_t
                        logits, _ = ac(feat)
                        a = int(torch.distributions.Categorical(logits=logits).sample().item())
                        if use_h_in_policy and pos < 60:
                            a_oh = F.one_hot(torch.tensor([a]), 3).float()
                            e_t = wm.action_mixer(torch.cat([z_t, a_oh], dim=-1)).unsqueeze(1)
                            h_o, past_kv = wm.forward_transformer(e_t, past_kv=past_kv, pos_offset=pos)
                            h_cur = h_o[:, 0]
                            pos += 1
                nxt, r, done = env.step(a)
                eo.append(o); ea.append(a); er.append(r)
                o = nxt
            replay.append((np.asarray(eo, dtype=np.float32), np.asarray(ea, dtype=np.int64), np.asarray(er, dtype=np.float32)))

    def eval_pi(n_eps=12):
        wm.eval(); ac.eval()
        ev_env = PixelCatchEnv(seed=900)
        rets = []
        for i in range(n_eps):
            o = ev_env.reset(seed=900 + i)
            done = False
            tot = 0.0
            past_kv, pos = None, 0
            h_cur = torch.zeros(1, wm.d_model)
            while not done:
                with torch.no_grad():
                    z_t, _ = wm.encode_obs(torch.from_numpy(o[None]))
                    feat = torch.cat([z_t, h_cur], dim=-1) if use_h_in_policy else z_t
                    a = int(ac(feat)[0].argmax(dim=-1).item())
                    if use_h_in_policy and pos < 60:
                        a_oh = F.one_hot(torch.tensor([a]), 3).float()
                        e_t = wm.action_mixer(torch.cat([z_t, a_oh], dim=-1)).unsqueeze(1)
                        h_o, past_kv = wm.forward_transformer(e_t, past_kv=past_kv, pos_offset=pos)
                        h_cur = h_o[:, 0]
                        pos += 1
                o, r, done = ev_env.step(a)
                tot += r
            rets.append(tot)
        return float(np.mean(rets))

    collect(12, use_pi=False)
    eval_x, eval_y = [], []
    env_steps = sum(len(e[0]) for e in replay)
    n_upd = 10 if is_quick() else 30

    for rd in range(1, n_rounds + 1):
        wm.train(); ac.train()
        for _ in range(n_upd):
            ob, ab, rb = [], [], []
            for _ in range(16):
                eo, ea, er = replay[int(rng.integers(0, len(replay)))]
                s = int(rng.integers(0, len(eo) - 8 + 1))
                ob.append(eo[s : s + 8]); ab.append(ea[s : s + 8]); rb.append(er[s : s + 8])
            loss_w, z_post, _ = wm.storm_loss(torch.from_numpy(np.stack(ob)), torch.from_numpy(np.stack(ab)), torch.from_numpy(np.stack(rb)))
            opt_wm.zero_grad(); loss_w.backward(); opt_wm.step()

            # 想象并更新 Actor-Critic
            start_z = z_post.reshape(-1, wm.z_dim)[:32]
            im_f, im_a, im_r = wm.imagine(start_z, ac, horizon=8, use_h_in_policy=use_h_in_policy)
            B_i, Hp1, D_f = im_f.shape
            logits_all, vals_all = ac(im_f.reshape(B_i * Hp1, D_f))
            logits_all = logits_all.reshape(B_i, Hp1, -1)
            vals_all = vals_all.reshape(B_i, Hp1)
            with torch.no_grad():
                targs = compute_lambda_returns(im_r, vals_all.detach(), gamma=0.95, lam=0.9)
                adv = targs - vals_all[:, :-1].detach()
            dist = torch.distributions.Categorical(logits=logits_all[:, :-1])
            loss_ac = -(dist.log_prob(im_a) * adv).mean() - 1e-2 * dist.entropy().mean() + 0.5 * F.mse_loss(vals_all[:, :-1], targs)
            opt_ac.zero_grad(); loss_ac.backward(); opt_ac.step()

        collect(4, use_pi=True)
        env_steps += 4 * 21
        eval_x.append(env_steps)
        eval_y.append(eval_pi(8 if is_quick() else 16))
    return np.array(eval_x), np.array(eval_y)


def main():
    seed_everything(SEED)
    verify_section_6_hand_calc()
    ips_iris, ips_storm, ratio = benchmark_speed_iris_vs_storm()

    quick = is_quick()
    n_rounds = 4 if quick else 14
    print("\n" + "=" * 60)
    print(f"第三部分：PixelCatchEnv 上对比 STORM（策略看 [z,h]）与 TWM 变体（策略只看 z）")
    print("=" * 60)
    t0 = time.time()
    xs_s, ys_s = train_storm_or_twm(use_h_in_policy=True, n_rounds=n_rounds, seed=SEED)
    xs_t, ys_t = train_storm_or_twm(use_h_in_policy=False, n_rounds=n_rounds, seed=SEED)
    print(f"  STORM（[z, h] 输入）：最终评估回报 = {ys_s[-1]:+.2f}（最高 {ys_s.max():+.2f}）")
    print(f"  TWM 变体（仅 z 输入）：最终评估回报 = {ys_t[-1]:+.2f}（最高 {ys_t.max():+.2f}）｜ 共用时 {time.time() - t0:.1f}s")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    axes[0].bar(["IRIS\n(K=16 逐token)", "STORM\n(每帧1位置)"], [ips_iris, ips_storm], color=["tomato", "seagreen"], width=0.45)
    axes[0].set_ylabel("每秒想象步数 (steps/s)")
    axes[0].set_title(f"想象吞吐率：STORM 比 IRIS 快 {ratio:.1f} 倍")

    axes[1].plot(xs_s, ys_s, "o-", linewidth=2.2, label=f"STORM ([z,h], 最终 {ys_s[-1]:+.2f})")
    axes[1].plot(xs_t, ys_t, "s--", linewidth=2.2, label=f"TWM 变体 (仅 z, 最终 {ys_t[-1]:+.2f})")
    axes[1].set_xlabel("真实环境交互步数"); axes[1].set_ylabel("平均评估回报（满分 +3.0）")
    axes[1].set_title("PixelCatchEnv：STORM vs TWM 学习曲线"); axes[1].legend(fontsize=8.5)
    save_figure(fig, OUT / "06-storm-vs-iris.png")

    print("\n" + "=" * 60)
    print("本节完成！下一节：07-maskgit-parallel-decoding.py（MaskGIT 双向掩码与余弦迭代并行解码）")
    print("=" * 60)


if __name__ == "__main__":
    main()
