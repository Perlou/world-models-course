"""
Phase 10 · 第 06 课：动作条件的隐空间动力学（冻结 DINOv2 特征 + 预测器）
=======================================================================

学习目标：
    1. 核对 CONCEPT.md 第 6.3 节「原样复制基线」与相对误差 ρ = e_model / e_copy 手算
    2. 离线加载本地缓存的冻结 DINOv2 (ViT-S/14) 编码器，预先提取 PushT 演示数据的
       112×112（8×8 = 64 patch，384 维）特征并存盘复用
    3. 实现帧内全连接、帧间因果的 Transformer 预测器 LatentDynamics（动作拼到每个 patch）
    4. 在 PushT 验证集上定量报告 1、5、10 步开环相对误差 ρ、打乱动作诊断，以及预计算与训练各自耗时

对应概念文档：
    src/phase-10-jepa/CONCEPT.md 第 6 节

运行方式：
    uv run python src/phase-10-jepa/06-action-conditioned-latent-dynamics.py
    WM_QUICK=1 uv run python src/phase-10-jepa/06-action-conditioned-latent-dynamics.py

预计运行时间：完整约 2 分钟（含首次 DINOv2 特征缓存，M1 Max CPU），冒烟约 15 秒
"""

import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.pusht import load_pusht_demos

plt = setup_matplotlib()
OUT = output_dir("phase-10")
SEED = 0

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ==================== 第一部分：离线加载冻结 DINOv2 与特征预计算 ====================


def load_frozen_dinov2():
    """纯离线加载仓库 data/torch-hub 下缓存的 DINOv2 ViT-S/14 权重，零网络依赖。"""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hub_dir = os.path.join(repo_root, "data", "torch-hub")
    local_repo = os.path.join(hub_dir, "hub", "facebookresearch_dinov2_main")
    ckpt_path = os.path.join(hub_dir, "hub", "checkpoints", "dinov2_vits14_pretrain.pth")
    torch.hub.set_dir(hub_dir)
    model = torch.hub.load(local_repo, "dinov2_vits14", source="local", pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_images_dinov2(encoder, imgs_uint8, img_size=112, batch_size=32):
    """将 uint8 RGB 图像序列 [N, H, W, 3] 批量编码为 DINOv2 patch 特征 [N, 64, 384]（float32）。"""
    outs = []
    for i in range(0, len(imgs_uint8), batch_size):
        batch_raw = imgs_uint8[i : i + batch_size]
        resized = np.stack([cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA) for im in batch_raw], axis=0)
        x = torch.from_numpy(np.transpose(resized.astype(np.float32) / 255.0, (0, 3, 1, 2)))
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        feats = encoder.forward_features(x)["x_norm_patchtokens"]  # [B, 64, 384]
        outs.append(feats.cpu())
    return torch.cat(outs, dim=0)


def precompute_features(n_episodes=30, frame_skip=4, img_size=112, cache_path=None):
    """CONCEPT.md 第 6.4/6.6 节：按跳帧提取 PushT 的冻结 DINOv2 特征并存成 float16 缓存。"""
    if cache_path is not None and os.path.exists(cache_path):
        d = np.load(cache_path)
        if int(d["n_episodes"]) >= n_episodes and int(d["frame_skip"]) == frame_skip:
            ep_lens = d["ep_lens"][:n_episodes]
            total_f = int(ep_lens.sum())
            feats_all = d["feats"][:total_f].astype(np.float32)
            acts_all = d["actions"][:total_f].astype(np.float32)
            states_all = d["states"][:total_f].astype(np.float32)
            episodes, offset = [], 0
            for L in ep_lens:
                L = int(L)
                episodes.append((feats_all[offset : offset + L], acts_all[offset : offset + L], states_all[offset : offset + L]))
                offset += L
            return episodes, float(d["precompute_seconds"]), True

    t0 = time.time()
    encoder = load_frozen_dinov2()
    demos = load_pusht_demos()
    starts, ends = demos["episode_starts"], demos["episode_ends"]

    feats_list, acts_list, states_list, ep_lens = [], [], [], []
    for ep_i in range(min(n_episodes, len(starts))):
        s, e = int(starts[ep_i]), int(ends[ep_i])
        # 每 frame_skip 步取 1 帧，把期间的动作合并为 2*frame_skip 维（或首尾均动作）
        idxs = list(range(s, e - frame_skip, frame_skip))
        if len(idxs) < 12:
            continue
        im_seq = demos["images"][idxs]
        f_seq = encode_images_dinov2(encoder, im_seq, img_size=img_size).numpy()  # [L, 64, 384]
        # 跳帧动作拼接：将这 frame_skip 步的 2D 动作展平为 [L, 2*frame_skip]
        a_seq = np.stack([
            (demos["actions"][idx : idx + frame_skip].astype(np.float32) / 256.0 - 1.0).reshape(-1)
            for idx in idxs
        ], axis=0)
        st_seq = demos["states"][idxs].astype(np.float32)
        feats_list.append(f_seq)
        acts_list.append(a_seq)
        states_list.append(st_seq)
        ep_lens.append(len(idxs))

    precompute_sec = time.time() - t0
    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(
            cache_path,
            feats=np.concatenate(feats_list, axis=0).astype(np.float16),
            actions=np.concatenate(acts_list, axis=0).astype(np.float32),
            states=np.concatenate(states_list, axis=0).astype(np.float32),
            ep_lens=np.asarray(ep_lens, dtype=np.int64),
            n_episodes=len(ep_lens),
            frame_skip=frame_skip,
            precompute_seconds=precompute_sec,
        )
    episodes = list(zip(feats_list, acts_list, states_list))
    return episodes, precompute_sec, False


# ==================== 第二部分：帧间因果 Patch 预测器与诊断函数 ====================


def frame_causal_mask(T, N, device="cpu"):
    """CONCEPT.md 第 6.2 节：帧内 N×N 全可见，帧间严格因果（下三角块矩阵，True 表示屏蔽）。"""
    t_ids = torch.arange(T, device=device).repeat_interleave(N)
    return t_ids.unsqueeze(0) > t_ids.unsqueeze(1)  # [T*N, T*N]


class LatentDynamics(nn.Module):
    """DINO-WM 风格动作条件 Patch 预测器：降维投影 + 动作拼接至每个 patch + 帧间因果 Transformer + 残差输出。"""

    def __init__(self, feat_dim=384, act_dim=8, num_patches=64, d_model=128, n_layers=2, n_heads=4, max_frames=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_patches = num_patches
        self.act_emb = nn.Sequential(nn.Linear(act_dim, 32), nn.GELU(), nn.Linear(32, 32))
        self.in_proj = nn.Linear(feat_dim + 32, d_model)
        self.pos_s = nn.Parameter(torch.randn(1, 1, num_patches, d_model) * 0.02)
        self.pos_t = nn.Parameter(torch.randn(1, max_frames, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, feat_dim)
        # 残差零初始化：起步即等于“原样复制”基线，加速稳定收敛
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_seq, act_seq):
        # z_seq: [B, T, N, D], act_seq: [B, T, A]
        B, T, N, D = z_seq.shape
        a_e = self.act_emb(act_seq).unsqueeze(2).expand(B, T, N, -1)
        h = self.in_proj(torch.cat([z_seq, a_e], dim=-1)) + self.pos_s + self.pos_t[:, :T]
        mask = frame_causal_mask(T, N, device=z_seq.device)
        out = self.tf(h.reshape(B, T * N, -1), mask=mask).reshape(B, T, N, -1)
        return z_seq + self.out_proj(out)


@torch.no_grad()
def rollout(model, z_hist, actions, ctx_len=3):
    """CONCEPT.md 第 6.6 节：从历史特征 z_hist [B, k, N, D] 按动作序列 actions [B, k+H-1, A] 展开 H 步。"""
    model.eval()
    buf_z = z_hist.clone()
    B, k = buf_z.shape[:2]
    H = actions.shape[1] - k + 1
    preds = []
    for t in range(H):
        a_win = actions[:, t : t + k]
        z_next = model(buf_z, a_win)[:, -1]  # [B, N, D]
        preds.append(z_next)
        buf_z = torch.cat([buf_z[:, 1:], z_next.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1)  # [B, H, N, D]


def copy_baseline_error(z_start, z_target):
    return float(F.mse_loss(z_start, z_target).item())


@torch.no_grad()
def evaluate_horizons_and_shuffle(model, te_eps, ctx_len=3, horizons=(1, 5, 10)):
    model.eval()
    max_h = max(horizons)
    z_hists, act_seqs, z_gts = [], [], []
    for f, a, _ in te_eps:
        for t in range(0, len(f) - ctx_len - max_h, 4):
            z_hists.append(f[t : t + ctx_len])
            act_seqs.append(a[t : t + ctx_len + max_h - 1])
            z_gts.append(f[t + ctx_len : t + ctx_len + max_h])
    z_h = torch.from_numpy(np.asarray(z_hists[:32]))
    a_s = torch.from_numpy(np.asarray(act_seqs[:32]))
    z_gt = torch.from_numpy(np.asarray(z_gts[:32]))

    pred_true = rollout(model, z_h, a_s, ctx_len=ctx_len)
    perm = torch.randperm(len(a_s), generator=torch.Generator().manual_seed(42))
    pred_shuf = rollout(model, z_h, a_s[perm], ctx_len=ctx_len)

    metrics = {}
    for h in horizons:
        e_copy = copy_baseline_error(z_h[:, -1], z_gt[:, h - 1])
        e_mod = float(F.mse_loss(pred_true[:, h - 1], z_gt[:, h - 1]).item())
        e_shuf = float(F.mse_loss(pred_shuf[:, h - 1], z_gt[:, h - 1]).item())
        metrics[h] = {
            "e_copy": e_copy,
            "e_model": e_mod,
            "rho": e_mod / max(e_copy, 1e-8),
            "e_shuf": e_shuf,
            "shuf_ratio": e_shuf / max(e_mod, 1e-8),
        }
    return metrics


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 6.3 节原样复制基线手算")
    print("=" * 60)
    e_copy_hand = (8 / 256) * 2.0
    print(f"  N=256 patch 中 8 个运动 patch (d²=2.0)：e_copy = {e_copy_hand:.4f}，甲 ρ={0.050/e_copy_hand:.2f}，乙 ρ={0.006/e_copy_hand:.2f} ✓")

    print("\n" + "=" * 60)
    print("第二部分：离线提取冻结 DINOv2 (ViT-S/14) PushT 特征并训练动作条件预测器")
    print("=" * 60)
    n_eps = 10 if quick else 36
    cache_file = str(OUT / ("pusht_dinov2_112_quick.npz" if quick else "pusht_dinov2_112.npz"))
    episodes, pre_sec, from_cache = precompute_features(n_episodes=n_eps, frame_skip=4, img_size=112, cache_path=cache_file)
    n_tr = max(2, int(len(episodes) * 0.8))
    tr_eps, te_eps = episodes[:n_tr], episodes[n_tr:]
    n_frames_tot = sum(len(e[0]) for e in episodes)
    print(f"  冻结 DINOv2 特征：共 {len(episodes)} 回合 / {n_frames_tot} 帧（每帧 64×384），预计算耗时 {pre_sec:.1f}s（命中缓存={from_cache}）")

    ctx_len = 3
    seq_len = 4  # T=4 帧教师强制窗口
    model = LatentDynamics(feat_dim=384, act_dim=8, num_patches=64, d_model=128, n_layers=2, n_heads=4, max_frames=8)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_iters = 60 if quick else 450
    t_tr = time.time()
    losses = []
    for it in range(1, n_iters + 1):
        zb, ab, z_tgt = [], [], []
        for _ in range(16):
            f, a, _ = tr_eps[np.random.randint(len(tr_eps))]
            t = np.random.randint(0, len(f) - seq_len)
            zb.append(f[t : t + seq_len])
            ab.append(a[t : t + seq_len])
            z_tgt.append(f[t + 1 : t + seq_len + 1])
        z_in = torch.from_numpy(np.asarray(zb))
        a_in = torch.from_numpy(np.asarray(ab))
        z_t = torch.from_numpy(np.asarray(z_tgt))

        opt.zero_grad()
        z_pred = model(z_in, a_in)
        loss = F.mse_loss(z_pred, z_t)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
    train_sec = time.time() - t_tr
    print(f"  预测器训练完成：{n_iters} 步，训练耗时 {train_sec:.1f}s，末段损失 {np.mean(losses[-20:]):.4f}")

    # 评估 1、5、10 步 ρ 与打乱动作诊断
    metrics = evaluate_horizons_and_shuffle(model, te_eps, ctx_len=ctx_len, horizons=(1, 5, 10))
    print(f"\n  PushT 验证集开环诊断（ρ < 1 表示优于「原样复制」；打乱动作比 > 1 表示模型真的在用动作）：")
    print(f"  {'步数 H':>6s} | {'e_copy':>8s} | {'e_model':>8s} | {'ρ = e/copy':>10s} | {'e_打乱动作':>10s} | {'打乱/真实':>9s}")
    for h in (1, 5, 10):
        m = metrics[h]
        print(f"  {h:>6d} | {m['e_copy']:8.4f} | {m['e_model']:8.4f} | {m['rho']:10.3f} | {m['e_shuf']:10.4f} | {m['shuf_ratio']:8.2f}×")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(losses)
    axes[0].set_title("DINOv2 特征预测器训练 MSE"); axes[0].set_xlabel("步数")

    hs = [1, 5, 10]
    axes[1].plot(hs, [metrics[h]["rho"] for h in hs], "o-", label="真实动作 ρ = e_model / e_copy")
    axes[1].plot(hs, [metrics[h]["e_shuf"] / metrics[h]["e_copy"] for h in hs], "s--", label="打乱动作 e_shuf / e_copy")
    axes[1].axhline(1.0, ls=":", color="gray", label="原样复制基线 (ρ = 1)")
    axes[1].set_title("开环特征预测相对误差 ρ 与动作敏感性"); axes[1].set_xlabel("展开步数 H"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, OUT / "06-latent-dynamics-dinov2.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：07-planning-in-feature-space.py")


if __name__ == "__main__":
    main()
