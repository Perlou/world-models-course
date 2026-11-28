"""Phase 9 实战项目主脚本：训练可玩的 PushT 扩散神经仿真器，输出漂移曲线、延迟表与规划基准。"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(os.path.dirname(_THIS)), "src")
for _p in (_THIS, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from neural_simulator import (
    PlayableSimulatorSession,
    PushTDiffusionWM,
    detail_errors,
    eval_cem_in_pusht,
    load_pusht_dataset,
    psnr_m11,
)
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-09-project")


def sample_batch(episodes, batch_size, ctx_len=3):
    ctxs, acts, tgts, rews = [], [], [], []
    for _ in range(batch_size):
        ims, a, r = episodes[np.random.randint(len(episodes))]
        t = np.random.randint(0, len(ims) - ctx_len)
        ctxs.append(ims[t : t + ctx_len])
        acts.append(a[t : t + ctx_len])
        tgts.append(ims[t + ctx_len])
        rews.append(r[t + ctx_len])
    return (
        torch.from_numpy(np.asarray(ctxs)),
        torch.from_numpy(np.asarray(acts)),
        torch.from_numpy(np.asarray(tgts)),
        torch.from_numpy(np.asarray(rews)),
    )


@torch.no_grad()
def eval_open_loop(model, te_eps, horizons=(1, 5, 10, 20, 50), n_steps=3, sigma_stab=0.05):
    model.eval()
    max_h = max(horizons)
    psnr_by_step = [[] for _ in range(max_h)]
    pusher_errs, fg_errs = [], []
    sample_strip = None
    for ims, a, _ in te_eps[:6]:
        H = min(max_h, len(ims) - model.ctx_len)
        if H < 1:
            continue
        buf_f = torch.from_numpy(ims[: model.ctx_len]).unsqueeze(0)
        buf_a = torch.from_numpy(a[: model.ctx_len]).unsqueeze(0)
        preds = []
        for k in range(H):
            a_win = torch.from_numpy(a[k : k + model.ctx_len]).unsqueeze(0)
            nxt = model.sample_next_frame(buf_f, a_win, n_steps=n_steps, sigma_stab=sigma_stab)
            gt = torch.from_numpy(ims[model.ctx_len + k]).unsqueeze(0)
            psnr_by_step[k].append(psnr_m11(nxt, gt))
            if k == 0:
                pe, fe = detail_errors(nxt, gt)
                pusher_errs.append(pe); fg_errs.append(fe)
            preds.append(nxt[0])
            buf_f = torch.cat([buf_f[:, 1:], nxt.unsqueeze(1)], dim=1)
        if sample_strip is None:
            sample_strip = (ims[model.ctx_len : model.ctx_len + min(H, 20)], torch.stack(preds[: min(H, 20)]).numpy())

    out_h = {}
    for h in horizons:
        vals = psnr_by_step[h - 1]
        out_h[str(h)] = float(np.mean(vals)) if vals else None
    curve = [float(np.mean(v)) for v in psnr_by_step if v]
    return out_h, curve, float(np.mean(pusher_errs)), float(np.mean(fg_errs)), sample_strip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="训练后进入命令行 WASD 交互模式")
    args = parser.parse_args()

    t0 = time.time()
    seed_everything(0)
    quick = is_quick()

    print("=" * 64)
    print("Phase 9 实战项目：可玩的 PushT 扩散神经仿真器")
    print("=" * 64)
    episodes = load_pusht_dataset(n_episodes=10 if quick else 45, img_size=32, frame_stride=2)
    n_tr = max(2, int(len(episodes) * 0.8))
    tr_eps, te_eps = episodes[:n_tr], episodes[n_tr:]
    sigma_data = float(np.concatenate([e[0] for e in tr_eps[:4]], axis=0).std())
    print(f"  PushT 32×32 数据：训练 {len(tr_eps)} 回合，测试 {len(te_eps)} 回合（σ_data={sigma_data:.3f}）")

    # 训练两个版本：噪声增强 EDM（主模型）vs 无噪声增强对照
    model_aug = PushTDiffusionWM(ctx_len=3, act_dim=2, base_ch=24, cond_dim=48, sigma_data=sigma_data)
    model_plain = PushTDiffusionWM(ctx_len=3, act_dim=2, base_ch=24, cond_dim=48, sigma_data=sigma_data)
    opt_aug = torch.optim.Adam(model_aug.parameters(), lr=1e-3)
    opt_plain = torch.optim.Adam(model_plain.parameters(), lr=1e-3)

    n_iters = 80 if quick else 700
    for it in range(1, n_iters + 1):
        ctx, act, tgt, rew = sample_batch(tr_eps, 32, ctx_len=3)
        opt_aug.zero_grad()
        l_aug = model_aug.edm_loss(tgt, ctx, act, max_ctx_noise=0.25) + 0.2 * F.mse_loss(model_aug.predict_reward(tgt), rew)
        l_aug.backward(); nn.utils.clip_grad_norm_(model_aug.parameters(), 1.0); opt_aug.step()

        opt_plain.zero_grad()
        l_pl = model_plain.edm_loss(tgt, ctx, act, max_ctx_noise=0.0)
        l_pl.backward(); nn.utils.clip_grad_norm_(model_plain.parameters(), 1.0); opt_plain.step()
        if it in {n_iters // 2, n_iters}:
            print(f"  [Iter {it:4d}/{n_iters}] 噪声增强 loss={l_aug.item():.4f} | 无增强 loss={l_pl.item():.4f} ({time.time() - t0:.1f}s)")

    horizons = (1, 5, 10, 20) if quick else (1, 5, 10, 20, 50)
    aug_h, aug_curve, pusher_err, fg_err, strip = eval_open_loop(model_aug, te_eps, horizons=horizons, n_steps=3, sigma_stab=0.05)
    pl_h, pl_curve, _, _, _ = eval_open_loop(model_plain, te_eps, horizons=horizons, n_steps=3, sigma_stab=0.0)

    print("\n  开环展开 PSNR (dB) 随步数变化：")
    print("    步数      :", list(horizons))
    print("    噪声增强  :", [round(aug_h[str(h)], 2) if aug_h[str(h)] is not None else None for h in horizons])
    print("    无增强对照:", [round(pl_h[str(h)], 2) if pl_h[str(h)] is not None else None for h in horizons])
    print(f"  细节误差（1 步预测）：推杆蓝点 L1 = {pusher_err:.4f}，T 形块前景 L1 = {fg_err:.4f}")

    # 每帧推理延迟测量（N ∈ {1, 2, 3, 5}）
    ctx0 = torch.from_numpy(te_eps[0][0][:3]).unsqueeze(0)
    act0 = torch.from_numpy(te_eps[0][1][:3]).unsqueeze(0)
    latency_table = {}
    print("\n  单帧交互延迟（本机 CPU，batch=1）：")
    for N in [1, 2, 3, 5]:
        sess = PlayableSimulatorSession(model_aug, ctx0, act0, n_steps=N, sigma_stab=0.05)
        dts = [sess.step_key(k)[1] for k in ["w", "d", "s", "a", "d", "w"]]
        ms = float(np.mean(dts[1:]))
        latency_table[str(N)] = {"ms": ms, "fps": 1000.0 / ms}
        print(f"    N = {N} 步：{ms:6.2f} ms/帧（{1000.0 / ms:5.1f} FPS）")

    # CEM 闭环规划基准（用于 CONCEPT.md §11.6 与 Phase 8/10 对比）
    cem_ret, cem_ms = eval_cem_in_pusht(
        model_aug, n_episodes=1 if quick else 2, max_steps=10 if quick else 30,
        horizon=2 if quick else 3, n_samples=8 if quick else 16, n_iters=2, n_denoise=1,
    )
    print(f"\n  PushT CEM 规划基准（N=1 步去噪）：平均回报 = {cem_ret:.3f}，每步决策耗时 = {cem_ms:.1f} ms")

    results = {
        "total_seconds": time.time() - t0,
        "train_iters": n_iters,
        "open_loop_psnr_aug": aug_h,
        "open_loop_psnr_plain": pl_h,
        "pusher_l1_error": pusher_err,
        "foreground_l1_error": fg_err,
        "latency_cpu": latency_table,
        "cem_eval_return": cem_ret,
        "cem_step_ms": cem_ms,
    }
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xs = np.arange(1, len(aug_curve) + 1)
    axes[0].plot(xs, aug_curve, label="噪声增强 EDM（主模型）")
    axes[0].plot(xs[: len(pl_curve)], pl_curve, "--", label="无噪声增强对照")
    axes[0].set_title("PushT 扩散神经仿真器开环 PSNR 漂移曲线")
    axes[0].set_xlabel("开环步数"); axes[0].set_ylabel("PSNR (dB)"); axes[0].legend()

    ns = [1, 2, 3, 5]
    axes[1].bar([f"N={n}" for n in ns], [latency_table[str(n)]["ms"] for n in ns], color="tab:blue")
    axes[1].set_title("单帧推理延迟（CPU，ms）"); axes[1].set_ylabel("ms/帧")
    fig.tight_layout()
    save_figure(fig, OUT / "drift-and-latency.png")

    if strip is not None:
        gt_s, pr_s = strip
        show_idx = [i for i in [0, 4, 9, 14, 19] if i < len(pr_s)]
        fig2, axes2 = plt.subplots(2, len(show_idx), figsize=(2.6 * len(show_idx), 5))
        for c, idx in enumerate(show_idx):
            axes2[0, c].imshow(((np.transpose(gt_s[idx], (1, 2, 0)) + 1) * 0.5).clip(0, 1))
            axes2[0, c].set_title(f"真实 t+{idx+1}"); axes2[0, c].axis("off")
            axes2[1, c].imshow(((np.transpose(pr_s[idx], (1, 2, 0)) + 1) * 0.5).clip(0, 1))
            axes2[1, c].set_title(f"仿真 t+{idx+1}"); axes2[1, c].axis("off")
        fig2.tight_layout()
        save_figure(fig2, OUT / "rollout-strip.png")

    print(f"\n📄 结果已保存：{OUT / 'results.json'}（总用时 {results['total_seconds']:.1f}s）")

    if args.interactive:
        sess = PlayableSimulatorSession(model_aug, ctx0, act0, n_steps=3, sigma_stab=0.05)
        print("进入交互模式（输入 w/a/s/d 推进一帧，q 退出）：")
        while True:
            cmd = input("按键 [w/a/s/d/q]> ").strip().lower()
            if cmd == "q":
                break
            frame, dt = sess.step_key(cmd)
            print(f"  生成新帧完成，耗时 {dt:.1f} ms，当前动作坐标 {sess.cursor.round(2).tolist()}")


if __name__ == "__main__":
    main()
