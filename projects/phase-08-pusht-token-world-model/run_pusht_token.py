"""运行 Phase 8 实战项目：PushT Token 世界模型（数据检查 → 训练 → 开环预测 → CEM 规划评估）。"""

import json
import os
import sys
import time

_PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_PROJ_DIR, "..", "..", "src"))
for p in (_PROJ_DIR, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pusht_token_wm import PushTTokenTransformerWM, PushTTokenizer, inspect_pusht_metadata, plan_pusht_token_cem
from utils import is_quick, output_dir, save_figure, seed_everything, setup_matplotlib
from utils.pusht import load_pusht_demos, make_pusht_env

plt = setup_matplotlib()
OUT = output_dir("phase-08-project")
SEED = 0


def main():
    seed_everything(SEED)
    quick = is_quick()

    # 1. 数据集与环境元信息检查
    meta = inspect_pusht_metadata()
    print("=" * 64)
    print("PushT 数据与环境规格检查")
    print("=" * 64)
    print(f"  图像分辨率：{meta['image_shape']} ｜ 动作范围：{meta['action_low']} ~ {meta['action_high']}")
    print(f"  回合步数上限：{meta['max_episode_steps']} ｜ 演示条数：{meta['num_episodes']} ｜ 总帧数：{meta['num_frames']:,}（均值 {meta['avg_ep_len']:.1f} 帧/回合）")

    # 2. 准备数据
    demos = load_pusht_demos()
    imgs = demos["images"]
    acts_norm = (demos["actions"].astype(np.float32) / 512.0) * 2.0 - 1.0
    rews = demos["rewards"].astype(np.float32)
    starts, ends = demos["episode_starts"], demos["episode_ends"]

    max_eps = 10 if quick else 45
    frame_stride = 2
    ep_imgs, ep_acts, ep_rews = [], [], []
    for ei in range(min(max_eps, len(starts))):
        sl = slice(int(starts[ei]), int(ends[ei]), frame_stride)
        res = np.stack([cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA) for im in imgs[sl]])
        ep_imgs.append(torch.from_numpy(res).permute(0, 3, 1, 2).float() / 255.0)
        ep_acts.append(torch.from_numpy(acts_norm[sl]))
        ep_rews.append(torch.from_numpy(rews[sl]))

    n_tr = max(1, int(len(ep_imgs) * 0.8))
    all_tr_imgs = torch.cat(ep_imgs[:n_tr], dim=0)

    # 3. 训练 FSQ Tokenizer
    t0 = time.time()
    tokenizer = PushTTokenizer(levels=(5, 5, 5))
    opt_t = torch.optim.Adam(tokenizer.parameters(), lr=2e-3)
    tok_steps = 40 if quick else 260
    for _ in range(tok_steps):
        idx = torch.randint(0, len(all_tr_imgs), (32,))
        rec, _ = tokenizer(all_tr_imgs[idx])
        loss = F.mse_loss(rec, all_tr_imgs[idx])
        opt_t.zero_grad(); loss.backward(); opt_t.step()
    tokenizer.eval()
    t_tok = time.time() - t0
    with torch.no_grad():
        val_imgs = torch.cat(ep_imgs[n_tr:], dim=0)[:200]
        rec_v, idx_v = tokenizer(val_imgs)
        tok_mse = F.mse_loss(rec_v, val_imgs).item()
        tok_psnr = float(10.0 * np.log10(1.0 / max(tok_mse, 1e-8)))
    print(f"\n  Tokenizer 训练完成（{t_tok:.1f}s）：验证 MSE = {tok_mse:.5f}，PSNR = {tok_psnr:.1f} dB")

    # 4. 编码全部轨迹并训练 Token Transformer 世界模型
    with torch.no_grad():
        ep_toks = [tokenizer.encode(im)[1] for im in ep_imgs]

    wm = PushTTokenTransformerWM(vocab_size=125, tokens_per_frame=16, act_dim=2, d_model=128, n_layers=3, n_heads=4)
    opt_w = torch.optim.AdamW(wm.parameters(), lr=2e-3, weight_decay=1e-2)
    rng = np.random.default_rng(SEED)
    T_chunk = 8 if quick else 12
    wm_steps = 40 if quick else 280
    t1 = time.time()
    for _ in range(wm_steps):
        wm.train()
        tb, ab, rb = [], [], []
        for _ in range(24):
            ei = int(rng.integers(0, n_tr))
            L = len(ep_toks[ei])
            s = int(rng.integers(0, max(1, L - T_chunk)))
            tb.append(ep_toks[ei][s : s + T_chunk])
            ab.append(ep_acts[ei][s : s + T_chunk])
            rb.append(ep_rews[ei][s : s + T_chunk])
        loss_w, ce_v, rw_v = wm.loss(torch.stack(tb), torch.stack(ab), torch.stack(rb))
        opt_w.zero_grad(); loss_w.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0); opt_w.step()
    wm.eval()
    t_wm = time.time() - t1
    print(f"  Token 世界模型训练完成（{t_wm:.1f}s）：末步 CE = {ce_v:.3f}，奖励 MSE = {rw_v:.5f}")

    # 5. 开环预测质量随步数变化（d = 1..10）
    H_eval = 5 if quick else 10
    pix_mses = np.zeros(H_eval)
    tok_accs = np.zeros(H_eval)
    rew_maes = np.zeros(H_eval)
    cnt = 0
    with torch.no_grad():
        for ei in range(n_tr, len(ep_toks)):
            if len(ep_toks[ei]) < H_eval + 2:
                continue
            p_tok, p_rew = wm.imagine_rollout(ep_toks[ei][0:1], ep_acts[ei][:H_eval].unsqueeze(0))
            for d in range(H_eval):
                tok_accs[d] += (p_tok[0, d] == ep_toks[ei][d + 1]).float().mean().item()
                dec_img = tokenizer.decode_tokens(p_tok[0, d : d + 1])[0]
                pix_mses[d] += F.mse_loss(dec_img, ep_imgs[ei][d + 1]).item()
                rew_maes[d] += abs(p_rew[0, d].item() - ep_rews[ei][d].item())
            cnt += 1
    pix_mses /= max(1, cnt)
    tok_accs /= max(1, cnt)
    rew_maes /= max(1, cnt)
    print(f"  开环预测（{cnt} 条留出轨迹）：")
    print("    像素 MSE  ：" + "  ".join(f"d={i+1}:{v:.4f}" for i, v in enumerate(pix_mses)))
    print("    Token 准确：" + "  ".join(f"d={i+1}:{v*100:.1f}%" for i, v in enumerate(tok_accs)))

    # 6. 在真实 PushT 仿真环境里用 Token 世界模型 + CEM 规划评估
    n_eval_eps = 1 if quick else 3
    max_plan_steps = 20 if quick else 60
    env = make_pusht_env(obs_type="pixels_agent_pos")
    max_coverages, successes, plan_times = [], [], []
    for ei in range(n_eval_eps):
        obs, _ = env.reset(seed=900 + ei)
        best_cov = 0.0
        succ = False
        for _ in range(max_plan_steps):
            a_px, ms = plan_pusht_token_cem(tokenizer, wm, obs["pixels"], horizon=4, n_candidates=24 if quick else 48, n_iters=2 if quick else 3)
            plan_times.append(ms)
            obs, r, term, trunc, info = env.step(a_px)
            best_cov = max(best_cov, float(r))
            if info.get("is_success", False) or term:
                succ = True
                break
            if trunc:
                break
        max_coverages.append(best_cov)
        successes.append(float(succ))
    env.close()

    avg_cov = float(np.mean(max_coverages))
    succ_rate = float(np.mean(successes))
    avg_plan_ms = float(np.mean(plan_times))
    print(f"\n  PushT 真实闭环 CEM 规划（{n_eval_eps} 回合 × {max_plan_steps} 步）：")
    print(f"    最高重叠覆盖率均值 = {avg_cov*100:.1f}% ｜ 成功率 = {succ_rate*100:.1f}% ｜ 单步规划耗时 = {avg_plan_ms:.1f} ms")

    summary = {
        "quick": quick,
        "metadata": meta,
        "tokenizer": {"val_mse": tok_mse, "val_psnr": tok_psnr, "train_sec": t_tok},
        "world_model": {"final_ce": ce_v, "final_rew_mse": rw_v, "train_sec": t_wm},
        "open_loop": {
            "steps": list(range(1, H_eval + 1)),
            "pixel_mse": [float(x) for x in pix_mses],
            "token_acc": [float(x) for x in tok_accs],
            "reward_mae": [float(x) for x in rew_maes],
        },
        "planning": {
            "n_episodes": n_eval_eps,
            "max_steps": max_plan_steps,
            "avg_max_coverage": avg_cov,
            "success_rate": succ_rate,
            "plan_ms_per_step": avg_plan_ms,
        },
    }
    res_path = OUT / "results.json"
    res_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📄 Phase 8 项目结果已保存：{res_path}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ds = np.arange(1, H_eval + 1)
    axes[0].plot(ds, pix_mses, "o-", color="C0", linewidth=2.2)
    axes[0].set_xlabel("开环步数 d"); axes[0].set_ylabel("解码画面像素 MSE")
    axes[0].set_title("PushT Token 世界模型：开环像素误差")

    axes[1].plot(ds, tok_accs * 100.0, "s-", color="seagreen", linewidth=2.2)
    axes[1].set_xlabel("开环步数 d"); axes[1].set_ylabel("Token 预测准确率 (%)")
    axes[1].set_title(f"Token 准确率随步数衰减（规划 {avg_plan_ms:.1f} ms/步）")
    save_figure(fig, OUT / "pusht-token-open-loop.png")


if __name__ == "__main__":
    main()
