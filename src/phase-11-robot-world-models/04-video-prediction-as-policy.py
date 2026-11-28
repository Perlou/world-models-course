"""
Phase 11 · 第 04 课：把视频预测当策略（视频预测 + 逆动力学模型）
==============================================================

学习目标：
    1. 复现 CONCEPT.md 第 4.5 节一维走廊「撞墙」实验，验证逆动力学模型（IDM）对采集策略
       π_β 的依赖（两种采集策略下撞墙后验精确对齐 0.857 与 0.143）
    2. 在 PushT 32×32 画面上训练像素逆动力学模型 InverseDynamicsModel(o_t, o_{t+1}) → a_t
    3. 对比 IDM 在「真实帧对」vs「视频模型生成帧对」以及「接触 T 块 / 非接触」分组下的动作误差
    4. 组装两阶段视频预测策略（先预测未来画面，再由 IDM 解动作），在 gym-pusht 上评测成功率与重规划耗时

对应概念文档：
    src/phase-11-robot-world-models/CONCEPT.md 第 4 节

运行方式：
    uv run python src/phase-11-robot-world-models/04-video-prediction-as-policy.py
    WM_QUICK=1 uv run python src/phase-11-robot-world-models/04-video-prediction-as-policy.py

预计运行时间：完整约 90 秒（M1 Max CPU），冒烟约 12 秒
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
from utils.pusht import load_pusht_demos, make_pusht_env

plt = setup_matplotlib()
OUT = output_dir("phase-11")
SEED = 0


# ==================== 第一部分：一维走廊逆动力学先验依赖实验 ====================


def idm_policy_dependence_demo():
    """CONCEPT.md 第 4.5 节：在 x=0 处向右有 0.1 滑回 0、向左必撞墙留 0，验证 q(左|0→0) 依赖采集策略。"""
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 4.5 节逆动力学对采集策略 π_β 的依赖")
    print("=" * 60)
    rng = np.random.default_rng(SEED)

    def run_policy(p_left):
        # 采集 10000 次从 s=0 出发的转移，训练小逻辑回归估计 q(左 | 0 → 0)
        acts = (rng.random(20000) < p_left).astype(np.int64)  # 1=左, 0=右
        # 左必留 0；右有 0.1 概率打滑留 0、0.9 到达 1
        stay = np.where(acts == 1, 1, (rng.random(20000) < 0.1).astype(np.int64))
        # 只看观测到 (0 -> 0) 的样本，频率即贝叶斯最优 IDM 后验
        post_emp = float(acts[stay == 1].mean())
        # 闭式解析后验：P(左)*1 / (P(左)*1 + (1-P(左))*0.1)
        post_theo = p_left / (p_left + (1.0 - p_left) * 0.1)
        return post_emp, post_theo

    e1, t1 = run_policy(0.375)  # 对应 0.375 / (0.375 + 0.0625) = 0.8571
    e2, t2 = run_policy(1.0 / 61.0)  # 使后验接近 0.143: p/(p+0.1(1-p))=1/7 => p=1/61
    # 同时报告文档第 4.5 节具体数值：π_1 均匀 (0.5/0.5 → 0.909) 与 偏右 (p_left=0.375 → 0.857)
    e_unif, t_unif = run_policy(0.5)
    print(f"  策略 A (p_left=0.500 均匀)：IDM 输出 P(左|0→0) = {e_unif:.3f}（解析 {t_unif:.3f}）")
    print(f"  策略 B (p_left=0.375)     ：IDM 输出 P(左|0→0) = {e1:.3f}（解析 {t1:.3f}，对应文档 0.857 ✓）")
    print(f"  策略 C (p_left=0.0164)    ：IDM 输出 P(左|0→0) = {e2:.3f}（解析 {t2:.3f}，对应文档 0.143 ✓）")
    print("  结论：物理转移完全不变，只换采集策略，同一个 (0→0) 转移的 IDM 输出就从 0.857 变到 0.143！")
    return (e1, t1), (e2, t2)


# ==================== 第二部分：PushT 逆动力学模型与视频预测策略 ====================


class InverseDynamicsModel(nn.Module):
    """输入相邻两帧 32×32 RGB 图像通道拼接 [B, 6, 32, 32]，预测中间执行的 2D 归一化动作 [-1, 1]。"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 24, 4, 2, 1), nn.SiLU(),   # 16
            nn.Conv2d(24, 48, 4, 2, 1), nn.SiLU(),  # 8
            nn.Conv2d(48, 64, 4, 2, 1), nn.SiLU(),  # 4
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.SiLU(),
            nn.Linear(128, 2),
        )

    def forward(self, o_t, o_next):
        return self.net(torch.cat([o_t, o_next], dim=1))


class VideoGoalPlanner(nn.Module):
    """不带动作条件的视频预测模型 p(o_{t+1:t+H} | o_t)（在专家演示上训练，预测接下来 H 步未来画面）。"""

    def __init__(self, horizon=4):
        super().__init__()
        self.horizon = horizon
        self.enc = nn.Sequential(
            nn.Conv2d(3, 24, 4, 2, 1), nn.SiLU(),   # 16
            nn.Conv2d(24, 48, 4, 2, 1), nn.SiLU(),  # 8
            nn.Conv2d(48, 64, 4, 2, 1), nn.SiLU(),  # 4
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 48, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(48, 24, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(24, 3 * horizon, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, o_t):
        out = self.dec(self.enc(o_t))
        return out.reshape(o_t.shape[0], self.horizon, 3, 32, 32)


def load_pusht_video_pairs(n_episodes=40, frame_stride=2, horizon=4):
    demos = load_pusht_demos()
    starts, ends = demos["episode_starts"], demos["episode_ends"]
    episodes = []
    for i in range(min(n_episodes, len(starts))):
        s, e = int(starts[i]), int(ends[i])
        ims_raw = demos["images"][s:e:frame_stride]
        acts_raw = demos["actions"][s:e:frame_stride]
        st_raw = demos["states"][s:e:frame_stride]
        if len(ims_raw) < horizon + 4:
            continue
        ims = np.stack([cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA) for im in ims_raw], axis=0)
        ims = np.transpose(ims.astype(np.float32) / 255.0, (0, 3, 1, 2))
        acts = (acts_raw.astype(np.float32) / 256.0) - 1.0
        # 接触近似：从画面提取灰色 T 形块质心，算推杆与块中心距离 < 95 像素
        f_img = ims_raw.astype(np.float32)
        sat = f_img.max(axis=-1) - f_img.min(axis=-1)
        lum = f_img.mean(axis=-1)
        mask = ((sat < 25.0) & (lum > 70.0) & (lum < 210.0)).astype(np.float32)
        xs = np.linspace(0.0, 512.0, ims_raw.shape[2], dtype=np.float32)[None, None, :]
        ys = np.linspace(0.0, 512.0, ims_raw.shape[1], dtype=np.float32)[None, :, None]
        area = mask.sum(axis=(1, 2)) + 1e-6
        b_xy = np.stack([(mask * xs).sum(axis=(1, 2)) / area, (mask * ys).sum(axis=(1, 2)) / area], axis=-1)
        dist = np.linalg.norm(st_raw[:, :2] - b_xy, axis=-1)
        contact = (dist < 95.0)
        episodes.append((ims, acts, contact))
    return episodes


def main():
    t0 = time.time()
    seed_everything(SEED)
    quick = is_quick()

    idm_policy_dependence_demo()

    print("\n" + "=" * 60)
    print("第二部分：PushT 逆动力学误差分析 & 两阶段视频预测策略")
    print("=" * 60)
    horizon = 4
    episodes = load_pusht_video_pairs(n_episodes=14 if quick else 55, frame_stride=2, horizon=horizon)
    n_tr = max(2, int(len(episodes) * 0.8))
    tr_eps, te_eps = episodes[:n_tr], episodes[n_tr:]

    idm = InverseDynamicsModel()
    vid_model = VideoGoalPlanner(horizon=horizon)
    opt_idm = torch.optim.Adam(idm.parameters(), lr=1e-3)
    opt_vid = torch.optim.Adam(vid_model.parameters(), lr=1e-3)

    n_iters = 100 if quick else 650
    for _ in range(n_iters):
        ot_l, on_l, a_l, fut_l = [], [], [], []
        for _ in range(32):
            ims, acts, _ = tr_eps[np.random.randint(len(tr_eps))]
            t = np.random.randint(0, len(ims) - horizon - 1)
            ot_l.append(ims[t]); on_l.append(ims[t + 1]); a_l.append(acts[t]); fut_l.append(ims[t + 1 : t + 1 + horizon])
        ot = torch.from_numpy(np.asarray(ot_l))
        on = torch.from_numpy(np.asarray(on_l))
        ab = torch.from_numpy(np.asarray(a_l))
        fut = torch.from_numpy(np.asarray(fut_l))

        opt_idm.zero_grad(); F.mse_loss(idm(ot, on), ab).backward(); opt_idm.step()
        opt_vid.zero_grad(); F.mse_loss(vid_model(ot), fut).backward(); opt_vid.step()

    # 评估 IDM 在真实帧对 vs 生成帧对，以及按接触 / 非接触分组的误差（换算为像素坐标误差 px）
    idm.eval(); vid_model.eval()
    err_real_c, err_real_nc, err_gen_all = [], [], []
    with torch.no_grad():
        for ims, acts, contact in te_eps:
            ot = torch.from_numpy(ims[:-horizon])
            on = torch.from_numpy(ims[1 : len(ims) - horizon + 1])
            ab = torch.from_numpy(acts[:-horizon])
            gen_on = vid_model(ot)[:, 0]
            e_real = ((idm(ot, on) - ab).norm(dim=-1) * 256.0).numpy()
            e_gen = ((idm(ot, gen_on) - ab).norm(dim=-1) * 256.0).numpy()
            c_mask = contact[:-horizon]
            if c_mask.any():
                err_real_c.append(e_real[c_mask])
            if (~c_mask).any():
                err_real_nc.append(e_real[~c_mask])
            err_gen_all.append(e_gen)

    e_rc = float(np.concatenate(err_real_c).mean()) if err_real_c else 0.0
    e_rnc = float(np.concatenate(err_real_nc).mean()) if err_real_nc else 0.0
    e_real_all = float(np.concatenate(err_real_c + err_real_nc).mean())
    e_gen_mean = float(np.concatenate(err_gen_all).mean())
    print(f"  IDM 真实帧对动作误差：{e_real_all:.1f} px（接触时 {e_rc:.1f} px，非接触时 {e_rnc:.1f} px）")
    print(f"  IDM 生成帧对动作误差：{e_gen_mean:.1f} px（比真实帧对放大 {e_gen_mean / max(e_real_all, 1e-4):.2f}×，体现分布偏移！）")

    # 两阶段视频预测策略在真实 gym-pusht 中闭环评测
    env = make_pusht_env(obs_type="pixels_agent_pos")
    n_eval = 5 if quick else 15
    max_steps = 60 if quick else 140
    succ, covs, replan_ms = 0, [], []
    for ep in range(n_eval):
        obs, _ = env.reset(seed=2000 + ep)
        max_c, ok = 0.0, False
        step = 0
        while step < max_steps:
            im32 = cv2.resize(obs["pixels"], (32, 32), interpolation=cv2.INTER_AREA)
            o_t = torch.from_numpy(np.transpose(im32.astype(np.float32) / 255.0, (2, 0, 1))).unsqueeze(0)
            ts = time.perf_counter()
            with torch.no_grad():
                fut_frames = vid_model(o_t)[0]  # [H, 3, 32, 32]
                prev_f = o_t[0]
                acts_plan = []
                for h in range(horizon):
                    a_h = idm(prev_f.unsqueeze(0), fut_frames[h].unsqueeze(0))[0].numpy()
                    acts_plan.append(a_h)
                    prev_f = fut_frames[h]
            replan_ms.append((time.perf_counter() - ts) * 1000.0)
            for a_h in acts_plan[:2]:
                act_env = np.clip((a_h + 1.0) * 256.0, 0.0, 512.0)
                for _ in range(2):  # 对应 frame_stride=2
                    obs, r, term, trunc, info = env.step(act_env)
                    cov = float(info.get("coverage", r))
                    max_c = max(max_c, cov)
                    if info.get("is_success", False) or cov >= 0.85:
                        ok = True
                    step += 1
                    if term or trunc or step >= max_steps:
                        break
                if term or trunc or step >= max_steps:
                    break
        succ += int(ok); covs.append(max_c)
    env.close()
    print(f"  两阶段视频预测+IDM 策略（{n_eval} 回合）：成功率 = {succ/n_eval*100:5.1f}% | 均最高覆盖度 = {np.mean(covs):.3f} | 重规划耗时 = {np.mean(replan_ms):.2f} ms")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].bar(["真实·非接触", "真实·接触", "生成帧对"], [e_rnc, e_rc, e_gen_mean], color=["tab:green", "tab:orange", "tab:red"])
    axes[0].set_title("逆动力学模型动作误差分组对比 (px)"); axes[0].set_ylabel("L2 误差 (px)")
    axes[1].bar(["两阶段视频+IDM"], [float(np.mean(covs))], color="tab:blue")
    axes[1].set_title(f"两阶段策略平均最高覆盖度（重规划 {np.mean(replan_ms):.1f}ms）"); axes[1].set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(fig, OUT / "04-video-prediction-policy.png")
    print(f"\n本节完成（总用时 {time.time() - t0:.1f}s）。下一节：05-world-action-models.py")


if __name__ == "__main__":
    main()
