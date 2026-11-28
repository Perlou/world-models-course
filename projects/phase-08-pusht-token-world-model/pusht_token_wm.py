"""Phase 8 实战项目核心模块：PushT 离散 Token 世界模型（FSQ Tokenizer + 因果 Transformer + CEM 规划）。"""

import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.planning import cem_plan
from utils.pusht import load_pusht_demos, make_pusht_env


def inspect_pusht_metadata():
    """读取 PushT 演示数据与 gym-pusht 环境的真实规格参数。"""
    demos = load_pusht_demos()
    env = make_pusht_env(obs_type="pixels_agent_pos")
    act_low = env.action_space.low.tolist()
    act_high = env.action_space.high.tolist()
    max_ep_steps = getattr(env.spec, "max_episode_steps", 300) if env.spec is not None else 300
    env.close()
    return {
        "image_shape": list(demos["images"].shape[1:]),
        "num_episodes": int(len(demos["episode_starts"])),
        "num_frames": int(len(demos["images"])),
        "action_low": [float(x) for x in act_low],
        "action_high": [float(x) for x in act_high],
        "max_episode_steps": int(max_ep_steps),
        "avg_ep_len": float(np.mean(demos["episode_ends"] - demos["episode_starts"])),
    }


class PushTFSQ(nn.Module):
    def __init__(self, levels=(5, 5, 5)):
        super().__init__()
        self.levels = tuple(levels)
        self.num_codes = int(math.prod(levels))
        self.dim = len(levels)
        lv = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("half_width", (lv - 1.0) / 2.0)
        basis = torch.cumprod(torch.cat([torch.ones(1, dtype=torch.long), torch.tensor(levels[:-1], dtype=torch.long)]), dim=0)
        self.register_buffer("basis", basis)

    def forward(self, z):
        b = torch.tanh(z) * self.half_width
        r = torch.round(b)
        z_q = (b + (r - b).detach()) / self.half_width
        idx = ((r.long() + self.half_width.long()) * self.basis).sum(dim=-1)
        return z_q, idx

    def decode_indices(self, idx):
        rem = idx.long()
        cols = []
        for L_i in self.levels:
            cols.append(rem % L_i)
            rem = rem // L_i
        return (torch.stack(cols, dim=-1).float() - self.half_width) / self.half_width


class PushTTokenizer(nn.Module):
    """64×64×3 → 4×4 (K=16) FSQ Tokenizer（N=125）。"""

    def __init__(self, levels=(5, 5, 5)):
        super().__init__()
        self.fsq = PushTFSQ(levels)
        d = self.fsq.dim
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, d, 4, 2, 1),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(d, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, obs):
        z = self.enc(obs).permute(0, 2, 3, 1)
        z_q, idx = self.fsq(z)
        return z_q, idx.reshape(obs.shape[0], -1)  # [B, 16]

    def decode_tokens(self, idx_flat):
        z_q = self.fsq.decode_indices(idx_flat.reshape(-1, 4, 4)).permute(0, 3, 1, 2)
        return self.dec(z_q)

    def forward(self, obs):
        z_q, idx = self.encode(obs)
        return self.dec(z_q.permute(0, 3, 1, 2)), idx


class PushTTokenTransformerWM(nn.Module):
    """每帧一个位置（将 16 个 token 嵌入 + 2D 动作拼接投影）的高效 STORM 式 PushT Token 世界模型：
    既在离散 token 空间预测下一帧全部 16 个 token 与奖励，又支持快速批量 CEM 规划。
    """

    def __init__(self, vocab_size=125, tokens_per_frame=16, act_dim=2, d_model=128, n_layers=3, n_heads=4, max_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.K = tokens_per_frame
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, 16)
        self.in_proj = nn.Sequential(nn.Linear(tokens_per_frame * 16 + act_dim, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.next_tok_head = nn.Linear(d_model, tokens_per_frame * vocab_size)
        self.rew_head = nn.Linear(d_model, 1)

    def forward(self, tokens_seq, act_seq):
        # tokens_seq: [B, T, K], act_seq: [B, T, 2]
        B, T, K = tokens_seq.shape
        te = self.tok_emb(tokens_seq).reshape(B, T, K * 16)
        x = self.in_proj(torch.cat([te, act_seq], dim=-1))
        pos = torch.arange(T, device=tokens_seq.device)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=tokens_seq.device)
        h = self.tf(x + self.pos_emb(pos).unsqueeze(0), mask=mask, is_causal=True)
        tok_logits = self.next_tok_head(h).reshape(B, T, K, self.vocab_size)
        rew_pred = self.rew_head(h).squeeze(-1)
        return tok_logits, rew_pred

    def loss(self, tokens_seq, act_seq, rew_seq):
        tok_logits, rew_pred = self(tokens_seq, act_seq)
        # h_t 预测 tokens_{t+1} 与 rew_t
        ce = F.cross_entropy(tok_logits[:, :-1].reshape(-1, self.vocab_size), tokens_seq[:, 1:].reshape(-1))
        rw = F.mse_loss(rew_pred, rew_seq)
        return ce + 5.0 * rw, ce.item(), rw.item()

    @torch.no_grad()
    def imagine_rollout(self, init_tokens, actions):
        """init_tokens: [N, K], actions: [N, H, 2] → (pred_tokens [N, H, K], pred_rewards [N, H])。"""
        N, H, _ = actions.shape
        tok_hist = init_tokens.unsqueeze(1)  # [N, 1, K]
        pred_toks, pred_rews = [], []
        for t in range(H):
            a_hist = actions[:, : t + 1]
            t_logits, r_p = self(tok_hist, a_hist)
            nxt_tok = t_logits[:, -1].argmax(dim=-1)  # [N, K]
            pred_toks.append(nxt_tok)
            pred_rews.append(r_p[:, -1])
            if t + 1 < H:
                tok_hist = torch.cat([tok_hist, nxt_tok.unsqueeze(1)], dim=1)
        return torch.stack(pred_toks, dim=1), torch.stack(pred_rews, dim=1)


def resize_frame_64(img_u8):
    arr = cv2.resize(img_u8, (64, 64), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


@torch.no_grad()
def plan_pusht_token_cem(tokenizer, wm, obs_img_u8, horizon=5, n_candidates=48, n_iters=3):
    """在离散 Token 空间用 src/utils/planning.py 的 cem_plan 做 PushT 规划，最大化预测奖励。"""
    t0 = time.time()
    x0 = resize_frame_64(obs_img_u8).unsqueeze(0)
    _, tok0 = tokenizer.encode(x0)  # [1, 16]
    tok_batch = tok0.expand(n_candidates, -1)

    def score_fn(actions):
        _, pred_r = wm.imagine_rollout(tok_batch, actions)
        return pred_r.sum(dim=-1)

    init_mean = torch.zeros(horizon, 2)
    plan_mean = cem_plan(
        score_fn, init_mean, action_low=-1.0, action_high=1.0,
        n_candidates=n_candidates, n_elites=max(6, n_candidates // 6),
        n_iters=n_iters, init_std=0.6, min_std=0.05,
    )
    a_norm = plan_mean[0].numpy()
    a_px = (a_norm + 1.0) * 0.5 * 512.0
    return np.clip(a_px, 0.0, 512.0).astype(np.float32), (time.time() - t0) * 1000.0
