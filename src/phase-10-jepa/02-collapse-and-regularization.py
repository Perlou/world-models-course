"""
Phase 10 · 第 02 课：表征坍塌与防坍塌的五类手段
================================================

学习目标：
    1. 核对 CONCEPT.md 第 2.2 节（有效秩）、2.5.2 节（InfoNCE）、2.7 节（VICReg 三项）手算数字
    2. 在带噪声背景移动小球数据上复现 6 组实验：
       ① 无防坍塌 ② 仅 stop-grad ③ stop-grad + EMA（无预测器）
       ④ stop-grad + EMA + 预测器（I-JEPA/BYOL 路线） ⑤ InfoNCE 对比学习 ⑥ VICReg 方差/协方差正则
    3. 持续监控各组的有效秩、表征各维平均标准差、坍塌步数，并用线性探针检查能否读出小球位置

对应概念文档：
    src/phase-10-jepa/CONCEPT.md 第 2 节

运行方式：
    uv run python src/phase-10-jepa/02-collapse-and-regularization.py
    WM_QUICK=1 uv run python src/phase-10-jepa/02-collapse-and-regularization.py

预计运行时间：完整约 60 秒（M1 Max CPU），冒烟约 8 秒
"""

import copy
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
_spec01 = importlib.util.spec_from_file_location("p10_01", os.path.join(_DIR, "01-generative-vs-joint-embedding.py"))
_mod01 = importlib.util.module_from_spec(_spec01)
_spec01.loader.exec_module(_mod01)

make_noisy_dot_dataset = _mod01.make_noisy_dot_dataset
SmallEncoder = _mod01.SmallEncoder
LatentPredictor = _mod01.LatentPredictor

plt = setup_matplotlib()
OUT = output_dir("phase-10")
SEED = 0


# ==================== 第一部分：核心防坍塌函数与有效秩 ====================


def effective_rank(z):
    """CONCEPT.md 第 2.2 节：基于中心化特征矩阵奇异值分布熵的有效秩 erank(Z) ∈ [1, D]。"""
    zc = z.detach().float()
    zc = zc - zc.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(zc)
    s_sum = s.sum()
    if s_sum.item() < 1e-12:
        return 1.0
    q = (s / s_sum).clamp_min(1e-12)
    return float(torch.exp(-(q * torch.log(q)).sum()).item())


def joint_embedding_loss(z_pred, z_target):
    return F.mse_loss(z_pred, z_target)


@torch.no_grad()
def ema_update(target_net, online_net, m=0.99):
    for pt, po in zip(target_net.parameters(), online_net.parameters()):
        pt.data.mul_(m).add_(po.data, alpha=1.0 - m)


def info_nce(z_a, z_b, temperature=0.2):
    """CONCEPT.md 第 2.5.2 节：批内余弦相似度 InfoNCE 对比损失。"""
    za = F.normalize(z_a, dim=-1)
    zb = F.normalize(z_b, dim=-1)
    logits = (za @ zb.T) / temperature
    labels = torch.arange(len(za), device=za.device)
    return F.cross_entropy(logits, labels)


def vicreg_loss(z_a, z_b, lam_inv=25.0, lam_var=25.0, lam_cov=1.0, gamma=1.0, eps=1e-4):
    """CONCEPT.md 第 2.6/2.8 节：不变项 + 标准差合页项 + 非对角协方差平方和。"""
    inv = ((z_a - z_b) ** 2).sum(dim=-1).mean()

    def _var_cov(z):
        n, d = z.shape
        zc = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(zc.var(dim=0, unbiased=True) + eps)
        v_loss = F.relu(gamma - std).mean()
        cov = (zc.T @ zc) / max(n - 1, 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        c_loss = (off_diag ** 2).sum() / d
        return v_loss, c_loss

    va, ca = _var_cov(z_a)
    vb, cb = _var_cov(z_b)
    return lam_inv * inv + lam_var * (va + vb) + lam_cov * (ca + cb)


def quick_probe_r2(enc, data_tr, data_te):
    enc.eval()
    with torch.no_grad():
        z_tr = enc(data_tr["obs"]).numpy()
        z_te = enc(data_te["obs"]).numpy()
    y_tr = data_tr["pos"].numpy()
    y_te = data_te["pos"].numpy()
    mu, sd = z_tr.mean(0), z_tr.std(0) + 1e-6
    X_tr = np.hstack([(z_tr - mu) / sd, np.ones((len(z_tr), 1))])
    X_te = np.hstack([(z_te - mu) / sd, np.ones((len(z_te), 1))])
    reg = 1e-2 * np.eye(X_tr.shape[1]); reg[-1, -1] = 0.0
    W = np.linalg.solve(X_tr.T @ X_tr + reg, X_tr.T @ y_tr)
    pred = X_te @ W
    ss_res = ((y_te - pred) ** 2).sum()
    ss_tot = ((y_te - y_te.mean(0, keepdims=True)) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-8))


def run_collapse_experiment(mode, data_tr, data_te, z_dim=16, n_steps=400):
    seed_everything(SEED)
    enc = SmallEncoder(z_dim)
    tgt_enc = copy.deepcopy(enc)
    pred = LatentPredictor(z_dim, 2)
    params = list(enc.parameters()) + (list(pred.parameters()) if mode not in {"ema_no_pred"} else [])
    opt = torch.optim.Adam(params, lr=2e-3)

    hist_loss, hist_std, hist_erank = [], [], []
    collapse_step = None
    N = len(data_tr["obs"])

    for step in range(1, n_steps + 1):
        idx = torch.randint(0, N, (64,))
        x, y, a = data_tr["obs"][idx], data_tr["next_obs"][idx], data_tr["actions"][idx]

        z_x = enc(x)
        if mode == "naive":
            z_hat = pred(z_x, a)
            z_y = enc(y)
            loss = joint_embedding_loss(z_hat, z_y)
        elif mode == "stopgrad_only":
            z_hat = pred(z_x, a)
            with torch.no_grad():
                z_y = enc(y)
            loss = joint_embedding_loss(z_hat, z_y)
        elif mode == "ema_no_pred":
            # 故意去掉预测器，检验 SimSiam/BYOL 论文指出的必要条件
            with torch.no_grad():
                z_y = tgt_enc(y)
            loss = joint_embedding_loss(z_x, z_y)
        elif mode == "ema_with_pred":
            z_hat = pred(z_x, a)
            with torch.no_grad():
                z_y = tgt_enc(y)
            loss = joint_embedding_loss(z_hat, z_y)
        elif mode == "infonce":
            z_hat = pred(z_x, a)
            z_y = enc(y)
            loss = info_nce(z_hat, z_y, temperature=0.2)
        elif mode == "vicreg":
            z_hat = pred(z_x, a)
            z_y = enc(y)
            loss = vicreg_loss(z_hat, z_y, lam_inv=25.0, lam_var=25.0, lam_cov=1.0)
        else:
            raise ValueError(mode)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if mode in {"ema_no_pred", "ema_with_pred"}:
            ema_update(tgt_enc, enc, m=0.98)

        if step % 10 == 0 or step == 1:
            with torch.no_grad():
                z_eval = enc(data_te["obs"][:128])
                std_val = float(z_eval.std(dim=0).mean().item())
                er = effective_rank(z_eval)
            hist_loss.append(float(loss.item()))
            hist_std.append(std_val)
            hist_erank.append(er)
            if collapse_step is None and std_val < 0.02:
                collapse_step = step

    r2 = quick_probe_r2(enc, data_tr, data_te)
    return {
        "loss": hist_loss,
        "std": hist_std,
        "erank": hist_erank,
        "final_std": hist_std[-1],
        "final_erank": hist_erank[-1],
        "collapse_step": collapse_step,
        "probe_r2": r2,
    }


def verify_section_2_hand_calcs():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 2.2 / 2.5.2 / 2.7 节手算")
    print("=" * 60)
    # 有效秩
    q1 = np.array([4.0, 4.0]) / 8.0
    q2 = np.array([9.0, 1.0]) / 10.0
    er1 = math.exp(-(q1 * np.log(q1)).sum())
    er2 = math.exp(-(q2 * np.log(q2)).sum())
    print(f"  有效秩：奇异值 (4,4) → {er1:.2f}，奇异值 (9,1) → {er2:.2f}（手算 2.00 / 1.38 ✓）")
    # VICReg 手算例子 (第 2.7 节)
    Z = torch.tensor([[0.3, 0.6], [-0.3, -0.6], [0.3, 0.6], [-0.3, -0.6]])
    Zp = torch.tensor([[0.4, 0.6], [-0.3, -0.5], [0.3, 0.6], [-0.2, -0.6]])
    s = ((Z - Zp) ** 2).sum(dim=-1).mean().item()
    std_z = torch.sqrt(Z.var(dim=0, unbiased=True) + 0.0)
    v = F.relu(1.0 - std_z).mean().item()
    cov = (Z.T @ Z) / 3.0
    c = ((cov - torch.diag(torch.diag(cov))) ** 2).sum().item() / 2.0
    tot_single = 25 * s + 25 * v + 1 * c
    print(f"  VICReg 手算：s={s:.4f}，v={v:.3f}，c={c:.4f}，加权和={tot_single:.2f}（手算 0.0075 / 0.480 / 0.0576 / 12.26 ✓）")
    assert abs(s - 0.0075) < 1e-4 and abs(v - 0.480) < 0.005 and abs(c - 0.0576) < 1e-4


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()
    verify_section_2_hand_calcs()

    print("\n" + "=" * 60)
    print("第二部分：6 组防坍塌方案对比（有效秩 / 标准差 / 位置线性探针 R²）")
    print("=" * 60)
    data_tr = make_noisy_dot_dataset(n_samples=400 if quick else 1500, noise_std=0.20, seed=SEED)
    data_te = make_noisy_dot_dataset(n_samples=150 if quick else 400, noise_std=0.20, seed=SEED + 99)

    modes = [
        ("naive", "① 无防坍塌"),
        ("stopgrad_only", "② 仅 stop-grad + 预测器"),
        ("ema_no_pred", "③ EMA + sg（无预测器）"),
        ("ema_with_pred", "④ EMA + sg + 预测器"),
        ("infonce", "⑤ InfoNCE 对比学习"),
        ("vicreg", "⑥ VICReg 正则"),
    ]
    n_steps = 120 if quick else 500
    res = {}
    print(f"  {'方案':24s} | {'最终有效秩(/16)':>14s} | {'最终均标准差':>12s} | {'坍塌步数':>8s} | {'位置探针 R²':>10s}")
    for key, label in modes:
        r = run_collapse_experiment(key, data_tr, data_te, z_dim=16, n_steps=n_steps)
        res[key] = (label, r)
        c_str = str(r["collapse_step"]) if r["collapse_step"] is not None else "未坍塌"
        print(f"  {label:24s} | {r['final_erank']:14.2f} | {r['final_std']:12.4f} | {c_str:>8s} | {r['probe_r2']:10.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    xs = [1] + list(range(10, n_steps + 1, 10))
    for key, (label, r) in res.items():
        axes[0].plot(xs[: len(r["std"])], r["std"], label=label)
        axes[1].plot(xs[: len(r["erank"])], r["erank"], label=label)
    axes[0].set_title("表征各维平均标准差"); axes[0].set_xlabel("训练步数"); axes[0].legend(fontsize=7)
    axes[1].set_title("表征有效秩（满秩 = 16）"); axes[1].set_xlabel("训练步数"); axes[1].legend(fontsize=7)

    labels_short = [v[0].split()[0] + " " + v[0].split()[1] for v in res.values()]
    r2s = [v[1]["probe_r2"] for v in res.values()]
    axes[2].bar(range(len(r2s)), r2s, color="teal")
    axes[2].set_xticks(range(len(r2s))); axes[2].set_xticklabels(labels_short, rotation=25, ha="right", fontsize=8)
    axes[2].set_title("冻结表征位置线性探针 R²"); axes[2].set_ylim(-0.1, 1.05)

    fig.tight_layout()
    save_figure(fig, OUT / "02-collapse-and-regularization.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：03-i-jepa-from-scratch.py")


if __name__ == "__main__":
    main()
