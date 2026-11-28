"""Phase 10 实战项目核心模块：PushT 冻结 DINOv2 + 动作条件预测器 + 目标图像条件 CEM 规划器（DINO-WM）。"""

import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.planning import cem_plan
from utils.pusht import load_pusht_demos, make_pusht_env

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_offline_dinov2():
    """纯离线加载 data/torch-hub 下缓存的 DINOv2 ViT-S/14。"""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hub_dir = os.path.join(repo_root, "data", "torch-hub")
    local_repo = os.path.join(hub_dir, "hub", "facebookresearch_dinov2_main")
    ckpt_path = os.path.join(hub_dir, "hub", "checkpoints", "dinov2_vits14_pretrain.pth")
    torch.hub.set_dir(hub_dir)
    model = torch.hub.load(local_repo, "dinov2_vits14", source="local", pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_dinov2_batch(encoder, imgs_uint8, img_size=112, batch_size=32):
    outs = []
    for i in range(0, len(imgs_uint8), batch_size):
        raw = imgs_uint8[i : i + batch_size]
        res = np.stack([cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_AREA) for im in raw], axis=0)
        x = torch.from_numpy(np.transpose(res.astype(np.float32) / 255.0, (0, 3, 1, 2)))
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        outs.append(encoder.forward_features(x)["x_norm_patchtokens"].cpu())
    return torch.cat(outs, dim=0)


def build_frame_causal_mask(T, N, device="cpu"):
    t_ids = torch.arange(T, device=device).repeat_interleave(N)
    return t_ids.unsqueeze(0) > t_ids.unsqueeze(1)


class DINOWMPredictor(nn.Module):
    """DINO-WM 帧间因果 Patch 预测器：接收最近 k 帧 64×384 patch 特征与跳帧动作，残差预测下一帧特征。"""

    def __init__(self, feat_dim=384, act_dim=8, num_patches=64, d_model=128, n_layers=2, n_heads=4, max_frames=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_patches = num_patches
        self.act_mlp = nn.Sequential(nn.Linear(act_dim, 32), nn.GELU(), nn.Linear(32, 32))
        self.in_proj = nn.Linear(feat_dim + 32, d_model)
        self.pos_s = nn.Parameter(torch.randn(1, 1, num_patches, d_model) * 0.02)
        self.pos_t = nn.Parameter(torch.randn(1, max_frames, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, feat_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_seq, a_seq):
        B, T, N, D = z_seq.shape
        ae = self.act_mlp(a_seq).unsqueeze(2).expand(B, T, N, -1)
        h = self.in_proj(torch.cat([z_seq, ae], dim=-1)) + self.pos_s + self.pos_t[:, :T]
        mask = build_frame_causal_mask(T, N, device=z_seq.device)
        out = self.tf(h.reshape(B, T * N, -1), mask=mask).reshape(B, T, N, -1)
        return z_seq + self.out_proj(out)

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


def jepa_feature_cost(z_pred, z_goal):
    return ((z_pred[:, -1] - z_goal.unsqueeze(0)) ** 2).mean(dim=(1, 2))


def plan_to_goal_image(predictor, z_hist, a_prev, z_goal, horizon=3, n_samples=24, n_iters=3, frame_skip=4):
    """用 Phase 1 统一的 cem_plan 在 DINOv2 特征空间里规划目标到达动作序列。"""
    t0 = time.perf_counter()

    def score_fn(cand_t):
        cand_rep = cand_t.repeat(1, 1, frame_skip)
        a_full = torch.cat([a_prev.repeat(len(cand_rep), 1, 1), cand_rep], dim=1)
        z_out = predictor.rollout(z_hist.repeat(len(cand_rep), 1, 1, 1), a_full)
        return -jepa_feature_cost(z_out, z_goal)

    best_seq = cem_plan(
        score_fn, torch.zeros(horizon, 2), action_low=-1.0, action_high=1.0,
        n_candidates=n_samples, n_elites=max(2, n_samples // 4), n_iters=n_iters,
    ).cpu().numpy()
    return best_seq, (time.perf_counter() - t0) * 1000.0
