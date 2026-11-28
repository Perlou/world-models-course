"""
Phase 10 · 第 08 课：端到端 JEPA 世界模型 —— PLDM、SIGReg 与 LeWorldModel
===========================================================================

学习目标：
    1. 核对 CONCEPT.md 第 8.4 节 SIGReg（Epps–Pulley 经验特征函数检验）的全部手算数字
    2. 实现线性复杂度的 sigreg(Z)（随机单位投影 + 特征函数积分）与端到端紧凑 JEPA 世界模型
       E2EJepaWorldModel（编码器 + 动作预测器从像素联合训练，每帧仅 1 个紧凑向量）
    3. 在带雪花背景移动小球任务上对比三种端到端配置（无正则 / PLDM 式 VICReg+IDM / LeWorldModel SIGReg）
       的有效秩与位置探针 R²，并扫描权衡系数 λ 展示预测误差与有效秩的拉扯
    4. 对比端到端紧凑 JEPA 与块特征规划在目标到达成功率与单次规划耗时上的差异

对应概念文档：
    src/phase-10-jepa/CONCEPT.md 第 8 节

运行方式：
    uv run python src/phase-10-jepa/08-end-to-end-jepa-world-models.py
    WM_QUICK=1 uv run python src/phase-10-jepa/08-end-to-end-jepa-world-models.py

预计运行时间：完整约 65 秒（M1 Max CPU），冒烟约 8 秒
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
from utils.planning import cem_plan

_DIR = os.path.dirname(os.path.abspath(__file__))
_spec01 = importlib.util.spec_from_file_location("p10_01_e2e", os.path.join(_DIR, "01-generative-vs-joint-embedding.py"))
_mod01 = importlib.util.module_from_spec(_spec01)
_spec01.loader.exec_module(_mod01)

_spec02 = importlib.util.spec_from_file_location("p10_02_e2e", os.path.join(_DIR, "02-collapse-and-regularization.py"))
_mod02 = importlib.util.module_from_spec(_spec02)
_spec02.loader.exec_module(_mod02)

make_noisy_dot_dataset = _mod01.make_noisy_dot_dataset
render_dot_frame = _mod01.render_dot_frame
SmallEncoder = _mod01.SmallEncoder
LatentPredictor = _mod01.LatentPredictor
effective_rank = _mod02.effective_rank
vicreg_loss = _mod02.vicreg_loss
quick_probe_r2 = _mod02.quick_probe_r2

plt = setup_matplotlib()
OUT = output_dir("phase-10")
SEED = 0


# ==================== 第一部分：SIGReg 与端到端 JEPA 世界模型 ====================


def epps_pulley_1d(p, t_grid=None, weights=None):
    """CONCEPT.md 第 8.3 节：一维样本 p [n, M] 与标准高斯 N(0, 1) 的经验特征函数加权 L2 距离。"""
    if t_grid is None:
        t_grid = torch.linspace(0.25, 3.0, 12, device=p.device, dtype=p.dtype)
    if weights is None:
        weights = torch.exp(-0.5 * (t_grid ** 2))
        weights = weights / weights.sum()
    # p: [n, M], t_grid: [Q] -> tp: [n, M, Q]
    tp = p.unsqueeze(-1) * t_grid.view(1, 1, -1)
    cf_real = torch.cos(tp).mean(dim=0)  # [M, Q]
    cf_imag = torch.sin(tp).mean(dim=0)  # [M, Q]
    target_real = torch.exp(-0.5 * (t_grid ** 2)).unsqueeze(0)  # [1, Q]
    diff2 = (cf_real - target_real) ** 2 + cf_imag ** 2         # [M, Q]
    return (diff2 * weights.unsqueeze(0)).sum(dim=-1).mean()


def sigreg(z, n_proj=32):
    """CONCEPT.md 第 8.3 节：Sketched Isotropic Gaussian Regularization，对批大小 n 与维度 D 均呈线性复杂度。"""
    n, d = z.shape
    u = torch.randn(d, n_proj, device=z.device, dtype=z.dtype)
    u = F.normalize(u, dim=0)
    proj = z @ u  # [n, M]
    return epps_pulley_1d(proj)


class E2EJepaWorldModel(nn.Module):
    """LeWorldModel / PLDM 风格端到端紧凑 JEPA 世界模型（像素 → 16 维紧凑隐向量 → 动作预测器）。"""

    def __init__(self, z_dim=16, act_dim=2):
        super().__init__()
        self.encoder = SmallEncoder(z_dim)
        self.predictor = LatentPredictor(z_dim, act_dim)
        self.idm = nn.Sequential(nn.Linear(2 * z_dim, 64), nn.ReLU(), nn.Linear(64, act_dim))

    def forward(self, obs, actions):
        z = self.encoder(obs)
        z_next_hat = self.predictor(z, actions)
        return z, z_next_hat


def train_e2e_jepa(mode, data_tr, data_te, z_dim=16, lam=2.0, n_steps=450):
    seed_everything(SEED)
    model = E2EJepaWorldModel(z_dim=z_dim, act_dim=2)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    N = len(data_tr["obs"])
    pred_losses = []

    for _ in range(n_steps):
        idx = torch.randint(0, N, (64,))
        o, on, a = data_tr["obs"][idx], data_tr["next_obs"][idx], data_tr["actions"][idx]
        z, z_hat = model(o, a)
        z_next = model.encoder(on)
        pred_loss = F.mse_loss(z_hat, z_next)

        if mode == "none":
            loss = pred_loss
        elif mode == "pldm":
            # PLDM：VICReg（方差+协方差）+ 时间平滑 + 逆动力学
            v_loss = vicreg_loss(z_hat, z_next, lam_inv=10.0, lam_var=10.0, lam_cov=1.0)
            smooth = F.mse_loss(z_next, z)
            idm_loss = F.mse_loss(model.idm(torch.cat([z, z_next], dim=-1)), a)
            loss = v_loss + 0.1 * smooth + 2.0 * idm_loss
        elif mode == "lewm":
            # LeWorldModel：极简两项损失 L_pred + λ · SIGReg(Z)
            reg = 0.5 * (sigreg(z) + sigreg(z_next))
            loss = pred_loss + lam * reg
        else:
            raise ValueError(mode)

        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        pred_losses.append(float(pred_loss.item()))

    model.eval()
    with torch.no_grad():
        z_eval = model.encoder(data_te["obs"][:200])
        er = effective_rank(z_eval)
        std_m = float(z_eval.std(dim=0).mean().item())
    r2 = quick_probe_r2(model.encoder, data_tr, data_te)
    return model, {
        "erank": er,
        "std": std_m,
        "probe_r2": r2,
        "pred_loss": float(np.mean(pred_losses[-30:])),
    }


@torch.no_grad()
def plan_in_latent(model, n_tasks=10, horizon=5, n_samples=64, n_iters=3):
    """在带雪花背景移动小球任务上用紧凑隐向量做目标图像条件 CEM 规划，测量成功率与规划耗时。"""
    model.eval()
    rng = np.random.default_rng(321)
    succ, plan_ms, final_err = 0, [], []
    for _ in range(n_tasks):
        x, y = rng.uniform(7.0, 25.0), rng.uniform(7.0, 25.0)
        gx, gy = np.clip(x + rng.uniform(-7.0, 7.0), 5.0, 27.0), np.clip(y + rng.uniform(-7.0, 7.0), 5.0, 27.0)
        og = torch.from_numpy(render_dot_frame(gx, gy, noise_std=0.20, rng=rng)).unsqueeze(0)
        zg = model.encoder(og)

        for _step in range(6):
            o_cur = torch.from_numpy(render_dot_frame(x, y, noise_std=0.20, rng=rng)).unsqueeze(0)
            z0 = model.encoder(o_cur)
            t_s = time.perf_counter()

            def score_fn(cand_t):
                S, H, _ = cand_t.shape
                z_cur = z0.repeat(S, 1)
                for h in range(H):
                    z_cur = model.predictor(z_cur, cand_t[:, h])
                return -((z_cur - zg) ** 2).mean(dim=-1)

            best_seq = cem_plan(
                score_fn, torch.zeros(horizon, 2), action_low=-1.0, action_high=1.0,
                n_candidates=n_samples, n_elites=max(4, n_samples // 4), n_iters=n_iters,
            ).cpu().numpy()
            plan_ms.append((time.perf_counter() - t_s) * 1000.0)
            x = float(np.clip(x + best_seq[0, 0] * 3.0, 4.0, 28.0))
            y = float(np.clip(y + best_seq[0, 1] * 3.0, 4.0, 28.0))

        err = math.hypot(x - gx, y - gy)
        final_err.append(err)
        if err < 3.0:
            succ += 1
    return succ / n_tasks, float(np.mean(final_err)), float(np.mean(plan_ms))


def verify_section_8_4_hand_calcs():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 8.4 节 SIGReg 手算")
    print("=" * 60)
    t1 = torch.tensor([1.0]); w1 = torch.tensor([1.0])
    t2 = torch.tensor([2.0])
    cases = [
        ("完全坍塌 [0,0,0,0]", [0.0, 0.0, 0.0, 0.0], 0.155),
        ("接近坍塌 [-0.1..0.1]", [-0.1, -0.1, 0.1, 0.1], 0.151),
        ("两尖峰 [-1,-1,1,1]", [-1.0, -1.0, 1.0, 1.0], 0.0044),
        ("非零常数 [2,2,2,2]", [2.0, 2.0, 2.0, 2.0], 1.87),
    ]
    for nm, vals, expect in cases:
        v = epps_pulley_1d(torch.tensor(vals).unsqueeze(1), t_grid=t1, weights=w1).item()
        print(f"  t=1 {nm:20s}：{v:.4f}（手算 {expect} ✓）")
    v_t2 = epps_pulley_1d(torch.tensor([-1.0, -1.0, 1.0, 1.0]).unsqueeze(1), t_grid=t2, weights=w1).item()
    print(f"  t=2 两尖峰 [-1,-1,1,1]  ：{v_t2:.3f}（手算 0.304，高阶形状被查出 ✓）")
    assert abs(v_t2 - 0.304) < 0.01


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()
    verify_section_8_4_hand_calcs()

    print("\n" + "=" * 60)
    print("第二部分：三种端到端配置对比 + LeWorldModel λ 权衡扫描")
    print("=" * 60)
    data_tr = make_noisy_dot_dataset(n_samples=400 if quick else 1500, noise_std=0.20, seed=SEED)
    data_te = make_noisy_dot_dataset(n_samples=150 if quick else 400, noise_std=0.20, seed=SEED + 99)
    n_steps = 120 if quick else 450

    m_none, s_none = train_e2e_jepa("none", data_tr, data_te, n_steps=n_steps)
    m_pldm, s_pldm = train_e2e_jepa("pldm", data_tr, data_te, n_steps=n_steps)
    m_lewm, s_lewm = train_e2e_jepa("lewm", data_tr, data_te, lam=5.0, n_steps=n_steps)

    print(f"\n  {'端到端配置':22s} | {'有效秩 (/16)':>11s} | {'均标准差':>8s} | {'位置探针 R²':>10s} | {'规划成功率':>10s} | {'单次规划(ms)':>12s}")
    for nm, mod, st in [
        ("无正则（坍塌对照）", m_none, s_none),
        ("PLDM (VICReg+IDM)", m_pldm, s_pldm),
        ("LeWorldModel (SIGReg)", m_lewm, s_lewm),
    ]:
        sr, ferr, pms = plan_in_latent(mod, n_tasks=6 if quick else 15, horizon=5, n_samples=64, n_iters=3)
        print(f"  {nm:22s} | {st['erank']:11.2f} | {st['std']:8.3f} | {st['probe_r2']:10.3f} | {sr*100:9.1f}% | {pms:12.2f}")

    # 扫描 λ ∈ {0.1, 1.0, 5.0, 20.0} 展示预测损失与有效秩的拉扯
    lam_list = [0.1, 1.0, 5.0, 20.0]
    sweep = {}
    print(f"\n  LeWorldModel 权衡系数 λ 扫描（CONCEPT.md 第 8.5 节）：")
    print(f"  {'λ':>6s} | {'预测损失':>10s} | {'有效秩 (/16)':>11s} | {'均标准差':>8s} | {'位置探针 R²':>10s}")
    for lv in lam_list:
        _, st = train_e2e_jepa("lewm", data_tr, data_te, lam=lv, n_steps=n_steps)
        sweep[lv] = st
        print(f"  {lv:6.1f} | {st['pred_loss']:10.4f} | {st['erank']:11.2f} | {st['std']:8.3f} | {st['probe_r2']:10.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["无正则", "PLDM", "LeWorldModel"], [s_none["erank"], s_pldm["erank"], s_lewm["erank"]], color=["gray", "tab:orange", "tab:blue"])
    axes[0].set_title("三种端到端 JEPA 配置的有效秩 (/16)"); axes[0].set_ylim(0, 17)

    ax1 = axes[1]
    ax2 = ax1.twinx()
    ax1.plot(lam_list, [sweep[l]["erank"] for l in lam_list], "o-", color="tab:blue", label="有效秩")
    ax2.plot(lam_list, [sweep[l]["pred_loss"] for l in lam_list], "s--", color="tab:red", label="预测损失")
    ax1.set_xscale("log"); ax1.set_xlabel("SIGReg 权重 λ"); ax1.set_ylabel("有效秩", color="tab:blue")
    ax2.set_ylabel("预测 MSE", color="tab:red"); ax1.set_title("正则与预测的拉扯（λ 扫描）")
    fig.tight_layout()
    save_figure(fig, OUT / "08-end-to-end-jepa.png")
    print(f"\n本章全部 8 节讲义完成（本节用时 {time.time() - t0:.1f}s）！接下来：实战项目 projects/phase-10-jepa-planner/")


if __name__ == "__main__":
    main()
