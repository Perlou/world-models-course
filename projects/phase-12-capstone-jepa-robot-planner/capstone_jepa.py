"""Phase 12 毕业项目核心模块：PushT 零样本 JEPA 目标规划器与可信评估工具。

整合全课程的关键工程与研究方法组件：
- Phase 1: 采样式规划器 `utils.planning.cem_plan`
- Phase 10: 冻结离线 DINOv2 ViT-S/14 编码器 + 帧级因果 Transformer 动作条件潜空间预测器
- Phase 12: 数据集内容指纹、按回合切分、开环误差曲线与可信视野 k*、
            t 区间 / Welch 差值区间 / Wilson 成功率区间 / 2×2 相互作用检验
"""

import hashlib
import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.planning import cem_plan


# ==================== 1. 统计工具（Phase 12 第 5/7 节）====================


def t_cdf(x, df, n_grid=10001):
    """t 分布的累积分布函数（x >= 0），数值梯形积分。"""
    log_c = math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0) - 0.5 * math.log(df * math.pi)
    grid = np.linspace(0.0, float(x), n_grid)
    pdf = np.exp(log_c - (df + 1.0) / 2.0 * np.log1p(grid ** 2 / df))
    return 0.5 + float(((pdf[1:] + pdf[:-1]) / 2.0).sum() * (grid[1] - grid[0]))


def t_critical(df, confidence=0.95):
    """t 分布双侧临界值 κ。"""
    lo, hi = 0.0, 200.0
    target = 0.5 + confidence / 2.0
    for _ in range(55):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if t_cdf(mid, df) < target else (lo, mid)
    return (lo + hi) / 2.0


def mean_and_ci(x, confidence=0.95):
    """样本均值、标准差 s、标准误 SE、95% t 区间。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    mean = float(x.mean())
    if n < 2:
        return {"mean": mean, "s": 0.0, "se": 0.0, "half": 0.0, "lo": mean, "hi": mean, "n": n}
    s = float(x.std(ddof=1))
    se = s / math.sqrt(n)
    half = t_critical(n - 1, confidence) * se
    return {"mean": mean, "s": s, "se": se, "half": half, "lo": mean - half, "hi": mean + half, "n": n}


def diff_ci(a, b, confidence=0.95):
    """两组独立样本均值之差 δ = mean(b) - mean(a) 的 Welch t 区间。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    va = float(a.var(ddof=1)) / len(a) if len(a) > 1 else 0.0
    vb = float(b.var(ddof=1)) / len(b) if len(b) > 1 else 0.0
    se = math.sqrt(va + vb)
    delta = float(b.mean() - a.mean())
    if se <= 1e-12:
        return {"delta": delta, "se": 0.0, "df": float(len(a) + len(b) - 2), "half": 0.0, "lo": delta, "hi": delta}
    denom = (va ** 2 / max(len(a) - 1, 1)) + (vb ** 2 / max(len(b) - 1, 1))
    df = (va + vb) ** 2 / max(denom, 1e-12)
    half = t_critical(df, confidence) * se
    return {"delta": delta, "se": se, "df": df, "half": half, "lo": delta - half, "hi": delta + half}


def wilson_interval(n_success, n_total, confidence=0.95):
    """成功率的 Wilson 置信区间（在 0/1 边界和小样本下稳健）。"""
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-6 else 1.6448536269514722
    p = n_success / max(n_total, 1)
    denom = 1.0 + z * z / max(n_total, 1)
    center = (p + z * z / (2.0 * max(n_total, 1))) / denom
    half = z / denom * math.sqrt(p * (1.0 - p) / max(n_total, 1) + z * z / (4.0 * max(n_total, 1) ** 2))
    return max(0.0, center - half), min(1.0, center + half)


def interaction_2x2(y00, y10, y01, y11, confidence=0.95):
    """2×2 因子消融的相互作用 (y11 - y01) - (y10 - y00) 与 Welch t 区间。"""
    arr = [np.asarray(v, dtype=np.float64) for v in (y00, y10, y01, y11)]
    means = [float(v.mean()) for v in arr]
    vars_se = [float(v.var(ddof=1)) / len(v) if len(v) > 1 else 0.0 for v in arr]
    inter = (means[3] - means[2]) - (means[1] - means[0])
    se = math.sqrt(sum(vars_se))
    if se <= 1e-12:
        half, df = 0.0, float(sum(len(v) - 1 for v in arr))
    else:
        denom = sum((v ** 2) / max(len(a) - 1, 1) for v, a in zip(vars_se, arr))
        df = (sum(vars_se) ** 2) / max(denom, 1e-12)
        half = t_critical(df, confidence) * se
    return {
        "y00": means[0], "y10": means[1], "y01": means[2], "y11": means[3],
        "a_when_b_off": means[1] - means[0],
        "a_when_b_on": means[3] - means[2],
        "b_when_a_off": means[2] - means[0],
        "b_when_a_on": means[3] - means[1],
        "interaction": inter,
        "se": se, "df": df, "half": half, "lo": inter - half, "hi": inter + half,
    }


def credible_horizon(e_model, e_copy, threshold=0.5, ks=None):
    """可信视野 k*：误差比 e_model(k) / e_copy(k) 从第一步起持续低于 threshold 的最大步数。"""
    e_m = np.asarray(e_model, dtype=np.float64)
    e_c = np.asarray(e_copy, dtype=np.float64)
    ks = np.arange(1, len(e_m) + 1) if ks is None else np.asarray(ks)
    ok = e_m < threshold * e_c
    n_ok = len(ok) if ok.all() else int(np.argmin(ok))
    return int(ks[n_ok - 1]) if n_ok > 0 else 0


# ==================== 2. 数据指纹与按回合切分（Phase 12 第 3 节）====================


def split_by_episode(n_episodes, train_frac=0.75, val_frac=0.125, seed=0):
    """固定种子的按回合切分，保证 train/val/test 严格不相交。"""
    order = np.random.default_rng(seed).permutation(n_episodes)
    n_tr = max(1, int(round(train_frac * n_episodes)))
    n_va = max(1, int(round(val_frac * n_episodes)))
    if n_tr + n_va >= n_episodes:
        n_va = max(1, (n_episodes - n_tr) // 2)
    return {
        "train": np.sort(order[:n_tr]),
        "val": np.sort(order[n_tr : n_tr + n_va]),
        "test": np.sort(order[n_tr + n_va :]),
    }


def compute_episodes_fingerprint(episodes):
    """基于数组内容计算数据集 12 位 SHA-256 指纹（不受文件时间戳影响）。"""
    h = hashlib.sha256()
    for i, (feats, acts, states) in enumerate(episodes):
        h.update(f"ep-{i}|{feats.shape}|{acts.shape}|{states.shape}|".encode())
        h.update(np.ascontiguousarray(feats, dtype=np.float32).tobytes()[:4096])
        h.update(np.ascontiguousarray(acts, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(states, dtype=np.float32).tobytes())
    return h.hexdigest()[:12]


# ==================== 3. 对照编码器与线性探针 ====================


class RandomPatchCNNEncoder(nn.Module):
    """未预训练的冻结卷积补丁编码器对照（输出与 DINOv2 同构的 [B, 64, feat_dim] 空间网格）。"""

    def __init__(self, feat_dim=128, seed=1234):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        for p in self.parameters():
            if p.ndim >= 2:
                nn.init.normal_(p, mean=0.0, std=0.08, generator=gen)
            else:
                nn.init.zeros_(p)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_images(self, imgs_uint8, img_size=112):
        """将 [B, H, W, 3] uint8 图像编码为 [B, 64, feat_dim] 空间补丁特征。"""
        xs = []
        for im in imgs_uint8:
            if im.shape[0] != img_size or im.shape[1] != img_size:
                im = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA)
            x = (im.astype(np.float32) / 255.0 - 0.5) * 2.0
            xs.append(np.transpose(x, (2, 0, 1)))
        xt = torch.from_numpy(np.asarray(xs, dtype=np.float32))
        fm = self.net(xt)  # [B, D, 8, 8]
        return fm.flatten(2).transpose(1, 2)  # [B, 64, D]


def fit_and_eval_state_probe(tr_feats, tr_states, te_feats, te_states, ridge=1e-1):
    """在冻结补丁特征上拟合岭回归线性探针，预测 PushT 推杆坐标 (x, y)（像素单位）。

    为了同时保留空间位置信息，将 8×8 补丁先降采样为 2×2 空间均值（4 × D）再线性回归。
    """
    def _spatial_pool(f):
        # f: [N, 64, D] -> [N, 8, 8, D] -> 4 个象限均值 -> [N, 4*D]
        N, P, D = f.shape
        g = f.reshape(N, 8, 8, D)
        q = np.concatenate([
            g[:, :4, :4].mean(axis=(1, 2)),
            g[:, :4, 4:].mean(axis=(1, 2)),
            g[:, 4:, :4].mean(axis=(1, 2)),
            g[:, 4:, 4:].mean(axis=(1, 2)),
        ], axis=-1)
        return q.astype(np.float64)

    Xtr = _spatial_pool(tr_feats)
    Xte = _spatial_pool(te_feats)
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtr_n = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], axis=1)
    Xte_n = np.concatenate([(Xte - mu) / sd, np.ones((len(Xte), 1))], axis=1)

    Ytr = tr_states[:, :2].astype(np.float64)
    Yte = te_states[:, :2].astype(np.float64)

    dim = Xtr_n.shape[1]
    reg = np.eye(dim) * ridge
    reg[-1, -1] = 0.0
    W = np.linalg.solve(Xtr_n.T @ Xtr_n + reg, Xtr_n.T @ Ytr)
    pred_te = Xte_n @ W
    mse = float(np.mean((pred_te - Yte) ** 2))
    rmse_px = float(np.sqrt(np.mean(np.sum((pred_te - Yte) ** 2, axis=-1))))
    var_y = float(np.mean((Yte - Yte.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - mse / max(var_y, 1e-8)
    return {"rmse_px": rmse_px, "r2": r2}


# ==================== 4. 动作条件潜空间预测器与 Goal-BC 基线 ====================


def build_frame_causal_mask(T, N):
    """帧级因果注意力掩码：同帧补丁互见，屏蔽未来帧（True 表示屏蔽）。"""
    f_idx = torch.arange(T).repeat_interleave(N)
    return f_idx[None, :] > f_idx[:, None]


class CapstoneJEPAPredictor(nn.Module):
    """帧级因果 Transformer 潜空间预测器（支持任意补丁维度 feat_dim）。"""

    def __init__(self, feat_dim=384, act_dim=8, num_patches=64, d_model=128, n_layers=2, n_heads=4, max_frames=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_patches = num_patches
        self.in_proj = nn.Linear(feat_dim, d_model)
        self.act_proj = nn.Linear(act_dim, d_model)
        self.patch_pos = nn.Parameter(torch.randn(1, 1, num_patches, d_model) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(1, max_frames, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, feat_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_seq, a_seq):
        B, T, N, _ = z_seq.shape
        x = self.in_proj(z_seq) + self.act_proj(a_seq).unsqueeze(2)
        x = x + self.patch_pos[:, :, :N] + self.time_pos[:, :T]
        mask = build_frame_causal_mask(T, N).to(z_seq.device)
        y = self.tr(x.reshape(B, T * N, -1), mask=mask).reshape(B, T, N, -1)
        return z_seq + self.out_proj(self.norm(y))

    @torch.no_grad()
    def rollout(self, z_hist, actions):
        self.eval()
        buf = z_hist.clone()
        k = buf.shape[1]
        H = actions.shape[1] - k + 1
        preds = []
        for t in range(H):
            zn = self(buf, actions[:, t : t + k])[:, -1]
            preds.append(zn)
            buf = torch.cat([buf[:, 1:], zn.unsqueeze(1)], dim=1)
        return torch.stack(preds, dim=1)


class GoalConditionedBC(nn.Module):
    """目标条件行为克隆（Goal-Conditioned BC）离线基线：输入当前帧与目标帧的池化 DINOv2 特征，输出下一步动作。"""

    def __init__(self, feat_dim=384, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 8, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
            nn.Tanh(),
        )

    @staticmethod
    def _pool_quadrants(z):
        # z: [..., 64, D] -> [..., 4*D]
        shape = z.shape[:-2]
        g = z.reshape(*shape, 8, 8, z.shape[-1])
        return torch.cat([
            g[..., :4, :4, :].mean(dim=(-3, -2)),
            g[..., :4, 4:, :].mean(dim=(-3, -2)),
            g[..., 4:, :4, :].mean(dim=(-3, -2)),
            g[..., 4:, 4:, :].mean(dim=(-3, -2)),
        ], dim=-1)

    def forward(self, z_cur, z_goal):
        x = torch.cat([self._pool_quadrants(z_cur), self._pool_quadrants(z_goal)], dim=-1)
        return self.net(x)


def feature_cost(z_pred_seq, z_goal, cost_type="patch_l2"):
    """特征空间目标代价：支持逐补丁 L2（保留空间结构）与全局平均池化 L2（消融对照）。"""
    z_end = z_pred_seq[:, -1]  # [N, 64, D]
    if cost_type == "patch_l2":
        return ((z_end - z_goal.unsqueeze(0)) ** 2).mean(dim=(1, 2))
    elif cost_type == "mean_pool_l2":
        return ((z_end.mean(dim=1) - z_goal.mean(dim=0, keepdim=True)) ** 2).mean(dim=-1)
    raise ValueError(f"未知代价类型: {cost_type}")


def plan_jepa_cem(predictor, z_hist, a_prev, z_goal, horizon=3, n_candidates=24, n_iters=3,
                  frame_skip=4, cost_type="patch_l2", generator=None):
    """在表征空间里用 CEM 规划目标到达动作序列。"""
    t0 = time.perf_counter()

    def score_fn(cand_t):
        cand_rep = cand_t.repeat(1, 1, frame_skip)
        a_full = torch.cat([a_prev.repeat(len(cand_rep), 1, 1), cand_rep], dim=1)
        z_out = predictor.rollout(z_hist.repeat(len(cand_rep), 1, 1, 1), a_full)
        return -feature_cost(z_out, z_goal, cost_type=cost_type)

    best_seq = cem_plan(
        score_fn, torch.zeros(horizon, 2), generator=generator,
        action_low=-1.0, action_high=1.0,
        n_candidates=n_candidates, n_elites=max(2, n_candidates // 4), n_iters=n_iters,
    ).cpu().numpy()
    return best_seq, (time.perf_counter() - t0) * 1000.0
