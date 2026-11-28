"""
Dreamer V3 世界模型：六大部件与完整三项损失组装
===============================================

学习目标：
    1. 组装 Dreamer V3 完整世界模型 DreamerV3WorldModel（同时支持像素 CNN 与向量 symlog MLP 两种观测模式）：
       编码器 + CategoricalRSSM（带 1% unimix）+ 解码器 + TwoHot 奖励头 + Bernoulli continue 头
    2. 实现 WorldModel.observe()、WorldModel.loss() 与 WorldModel.imagine()，严格落地全部 stop-gradient
    3. 核对 CONCEPT.md 第 8.6 节的 2 步损失手算例子（14.156 与 13.6731，平均 13.9146）及第 8.5 节梯度流向表
    4. 在像素版 Pendulum-v1 上训练世界模型，记录四项分量曲线（重建、奖励、continue、未截断 KL），
       单独统计非零奖励步上的奖励预测误差，并生成开环预测对比视频

核心概念：
    - 三项世界模型损失：L(θ) = β_pred L_pred + β_dyn L_dyn + β_rep L_rep（1.0, 1.0, 0.1）
    - 双模式编解码：图像走 CNN 像素求和 MSE，向量观测走 symlog MLP 平方误差
    - 回合边界重置（is_first）：跨回合片段在 is_first=1 处清零 (h, z, a_prev)

前置知识：
    - 04–07 节（CategoricalRSSM、kl_loss_balanced、symlog、TwoHotHead）

对应概念文档：
    - CONCEPT.md 第 8 节「Dreamer V3 世界模型的完整损失」

预计运行时间：约 4 分钟（CPU）
"""

import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_device, is_quick, output_dir, save_figure, save_video, seed_everything, setup_matplotlib

_DIR = os.path.dirname(os.path.abspath(__file__))
_P3_DIR = os.path.abspath(os.path.join(_DIR, "..", "phase-03-rssm-planet"))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod04 = _load(os.path.join(_DIR, "04-categorical-latents-straight-through.py"), "p6_04")
_mod05 = _load(os.path.join(_DIR, "05-kl-balancing-and-free-bits.py"), "p6_05")
_mod06 = _load(os.path.join(_DIR, "06-symlog-and-twohot.py"), "p6_06")
_p3_05 = _load(os.path.join(_P3_DIR, "05-world-model-training.py"), "p3_05")

CategoricalRSSM = _mod04.CategoricalRSSM
kl_loss_balanced = _mod05.kl_loss_balanced
symlog = _mod06.symlog
symexp = _mod06.symexp
TwoHotHead = _mod06.TwoHotHead
preprocess_obs = _p3_05.preprocess_obs
collect_dataset = _p3_05.collect_offline_dataset

plt = setup_matplotlib()
OUT = output_dir("phase-06")
P3_OUT = output_dir("phase-03")
SEED = 0
DEVICE = get_device()


# ==================== 第一部分：编解码器与 DreamerV3WorldModel ====================


class PixelEncoder(nn.Module):
    """64×64×3 图像 → 1024 维特征（轻量化 4 层步长卷积，适合本机 CPU/MPS 训练）。"""

    def __init__(self, base_ch=24):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, base_ch, 4, stride=2, padding=1), nn.SiLU(),            # 32
            nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1), nn.SiLU(),  # 16
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1), nn.SiLU(),  # 8
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=2, padding=1), nn.SiLU(),  # 4
        )
        self.out_dim = base_ch * 8 * 4 * 4

    def forward(self, obs):
        # obs: [..., 3, 64, 64]
        lead = obs.shape[:-3]
        x = obs.reshape(-1, *obs.shape[-3:])
        e = self.conv(x).flatten(1)
        return e.reshape(*lead, self.out_dim)


class PixelDecoder(nn.Module):
    def __init__(self, feat_dim, base_ch=24):
        super().__init__()
        self.base_ch = base_ch
        self.fc = nn.Linear(feat_dim, base_ch * 8 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, stride=2, padding=1), nn.SiLU(),  # 8
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1), nn.SiLU(),  # 16
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1), nn.SiLU(),      # 32
            nn.ConvTranspose2d(base_ch, 3, 4, stride=2, padding=1),                           # 64
        )

    def forward(self, feat):
        lead = feat.shape[:-1]
        x = self.fc(feat.reshape(-1, feat.shape[-1])).reshape(-1, self.base_ch * 8, 4, 4)
        out = self.deconv(x)
        return out.reshape(*lead, 3, 64, 64)


class VectorEncoder(nn.Module):
    """向量观测经 symlog 变换后过 MLP 编码（CONCEPT.md 第 6.4 节）。"""

    def __init__(self, obs_dim, embed_dim=128):
        super().__init__()
        self.out_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU(),
        )

    def forward(self, obs):
        return self.net(symlog(obs))


class VectorDecoder(nn.Module):
    """在 symlog 空间预测向量观测。"""

    def __init__(self, feat_dim, obs_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, feat):
        return self.net(feat)


class DreamerV3WorldModel(nn.Module):
    """完整 Dreamer V3 世界模型（像素 / 向量双模式）。"""

    def __init__(
        self,
        act_dim,
        obs_mode="pixel",
        obs_dim=None,
        deter_dim=192,
        n_stoch=16,
        n_classes=16,
        hidden=192,
        unimix=0.01,
        base_ch=24,
        beta_pred=1.0,
        beta_dyn=1.0,
        beta_rep=0.1,
        free_nats=1.0,
    ):
        super().__init__()
        self.obs_mode = obs_mode
        self.beta_pred = beta_pred
        self.beta_dyn = beta_dyn
        self.beta_rep = beta_rep
        self.free_nats = free_nats
        self.unimix = unimix

        if obs_mode == "pixel":
            self.encoder = PixelEncoder(base_ch=base_ch)
            self.rssm = CategoricalRSSM(
                embed_dim=self.encoder.out_dim, act_dim=act_dim,
                deter_dim=deter_dim, n_stoch=n_stoch, n_classes=n_classes, hidden=hidden, unimix=unimix,
            )
            self.decoder = PixelDecoder(self.rssm.feat_dim, base_ch=base_ch)
        else:
            assert obs_dim is not None
            self.encoder = VectorEncoder(obs_dim, embed_dim=hidden)
            self.rssm = CategoricalRSSM(
                embed_dim=hidden, act_dim=act_dim,
                deter_dim=deter_dim, n_stoch=n_stoch, n_classes=n_classes, hidden=hidden, unimix=unimix,
            )
            self.decoder = VectorDecoder(self.rssm.feat_dim, obs_dim, hidden=hidden)

        self.feat_dim = self.rssm.feat_dim
        self.reward_head = TwoHotHead(self.feat_dim, hidden=hidden)
        self.cont_head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def observe(self, obs, actions, is_first=None, sample=True):
        embeds = self.encoder(obs)
        return self.rssm.observe(embeds, actions, is_first=is_first, sample=sample)

    def loss(self, obs, actions, rewards, conts, is_first=None):
        """CONCEPT.md 第 8.7 节：计算世界模型总损失并返回后验状态 (h, z) 供想象使用。"""
        out = self.observe(obs, actions, is_first=is_first, sample=True)
        feat = out["features"]  # [B, T, D]

        # 1. 预测损失 L_pred = 重建 + two-hot 奖励 + 二元交叉熵 continue
        if self.obs_mode == "pixel":
            recon = self.decoder(feat)
            recon_loss = 0.5 * (recon - obs).pow(2).sum(dim=(-3, -2, -1)).mean()
        else:
            recon_symlog = self.decoder(feat)
            recon_loss = 0.5 * (recon_symlog - symlog(obs)).pow(2).sum(dim=-1).mean()

        rew_logits = self.reward_head(feat)
        rew_loss = self.reward_head.loss(rew_logits, rewards).mean()

        cont_logits = self.cont_head(feat).squeeze(-1)
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, conts).mean()

        pred_loss = recon_loss + rew_loss + cont_loss

        # 2. 拆分 KL + free bits
        kl_loss, raw_kl, dyn_l, rep_l = kl_loss_balanced(
            out["post_logits"], out["prior_logits"],
            beta_dyn=self.beta_dyn, beta_rep=self.beta_rep,
            free_nats=self.free_nats, unimix=self.unimix,
        )

        total = self.beta_pred * pred_loss + kl_loss
        with torch.no_grad():
            pred_r = self.reward_head.predict(rew_logits)
            nz_mask = rewards.abs() > 1e-4
            nz_mae = (pred_r[nz_mask] - rewards[nz_mask]).abs().mean().item() if nz_mask.any() else 0.0

        metrics = dict(
            total=total.item(), recon=recon_loss.item(), reward=rew_loss.item(),
            cont=cont_loss.item(), raw_kl=raw_kl, dyn=dyn_l, rep=rep_l, nz_rew_mae=nz_mae,
        )
        return total, out, metrics

    @torch.no_grad()
    def open_loop_imagine(self, ctx_obs, ctx_act, future_act, sample=False):
        out = self.observe(ctx_obs, ctx_act, sample=False)
        h = out["h"][:, -1]
        z = out["z"][:, -1]
        feats = []
        for t in range(future_act.shape[1]):
            h, z, _ = self.rssm.imagine_step(h, z, future_act[:, t], sample=sample)
            feats.append(torch.cat([h, z], dim=-1))
        feats = torch.stack(feats, dim=1)
        recon = self.decoder(feats)
        if self.obs_mode != "pixel":
            recon = symexp(recon)
        pred_r = self.reward_head.predict(self.reward_head(feats))
        return recon, pred_r


# ==================== 第二部分：手算核对与梯度流向验证 ====================


def verify_section_8():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 8.6 节损失手算与第 8.5 节梯度流向")
    print("=" * 60)
    # 1. 第 8.6 节 2 步手算
    l_pred = torch.tensor([12.0 + 0.9506 + (-np.log(0.9)), 10.0 + 0.70 + (-np.log(0.8))])
    raw_kl = torch.tensor([0.6, 2.5])
    l_dyn = raw_kl.clamp_min(1.0)
    l_rep = raw_kl.clamp_min(1.0)
    total_step = l_pred + 1.0 * l_dyn + 0.1 * l_rep
    print(f"  1. 两步加权总损失 = {[round(x, 4) for x in total_step.tolist()]}（手算 [14.1560, 13.6731]），"
          f"均值 = {total_step.mean().item():.4f}（手算 13.9146）✓")
    assert abs(total_step.mean().item() - 13.9146) < 1e-3

    # 2. 第 8.5 节梯度流向验证：单独对 L_dyn 反传，确认第 t 步 post_net 不收到来自 L_dyn 的直接梯度（T=1 时 post_net 梯度严格为 0）
    wm = DreamerV3WorldModel(act_dim=1, obs_mode="vector", obs_dim=3, deter_dim=32, n_stoch=4, n_classes=4, hidden=32)
    o = torch.randn(2, 1, 3)
    a = torch.randn(2, 1, 1)
    out = wm.observe(o, a)
    kl_dyn = _mod05.categorical_kl(out["post_logits"].detach(), out["prior_logits"]).mean()
    wm.zero_grad()
    kl_dyn.backward()
    post_g = sum(p.grad.abs().sum().item() for p in wm.rssm.post_net.parameters() if p.grad is not None)
    prior_g = sum(p.grad.abs().sum().item() for p in wm.rssm.prior_net.parameters() if p.grad is not None)
    print(f"  2. T=1 单独反传 L_dyn：post_net 梯度 = {post_g:.6f} (严格 0 ✓) ｜ prior_net 梯度 = {prior_g:.4f} (>0 ✓)")
    assert post_g == 0.0 and prior_g > 0.0


# ==================== 第三部分：像素版 Pendulum-v1 训练与开环视频 ====================


def train_v3_world_model():
    print("\n" + "=" * 60)
    print("第二部分：在像素版 Pendulum-v1 上训练 Dreamer V3 世界模型")
    print("=" * 60)

    quick = is_quick()
    ds_path = P3_OUT / "pendulum-pixel-dataset.npz"
    if ds_path.exists():
        data = np.load(ds_path)
        obs_all, act_all, rew_all = data["obs"], data["actions"], data["rewards"]
        print(f"  复用 Phase 3 数据集：{ds_path}（{len(obs_all)} 回合）")
    else:
        data = collect_dataset(n_episodes=4 if quick else 20)
        obs_all, act_all, rew_all = data["obs"], data["actions"], data["rewards"]

    n_eps = len(obs_all)
    n_train = max(1, int(n_eps * 0.85))
    n_steps = 25 if quick else 320
    batch_size = 4 if quick else 10
    chunk_len = 12 if quick else 24

    wm = DreamerV3WorldModel(
        act_dim=1, obs_mode="pixel", deter_dim=192, n_stoch=16, n_classes=16, hidden=192, base_ch=24
    ).to(DEVICE)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    rng = np.random.default_rng(SEED)

    hist = {k: [] for k in ["recon", "reward", "cont", "raw_kl", "nz_rew_mae"]}
    t0 = time.time()

    for step in range(1, n_steps + 1):
        ep_ids = rng.integers(0, n_train, size=batch_size)
        o_b, a_b, r_b, c_b = [], [], [], []
        for eid in ep_ids:
            start = int(rng.integers(1, obs_all.shape[1] - chunk_len + 1))
            o_b.append(obs_all[eid, start : start + chunk_len])
            a_b.append(act_all[eid, start : start + chunk_len])
            r_b.append(rew_all[eid, start : start + chunk_len])
            c_b.append(np.ones(chunk_len, dtype=np.float32))
        o_t = preprocess_obs(torch.from_numpy(np.stack(o_b)).to(DEVICE))
        a_t = torch.from_numpy(np.stack(a_b)).to(DEVICE)
        r_t = torch.from_numpy(np.stack(r_b)).to(DEVICE)
        c_t = torch.from_numpy(np.stack(c_b)).to(DEVICE)

        loss, _, m = wm.loss(o_t, a_t, r_t, c_t)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
        opt.step()

        for k in hist:
            hist[k].append(m[k])
        if step == 1 or step % max(1, n_steps // 4) == 0 or step == n_steps:
            print(f"  步 {step:>3}/{n_steps}  重建 {m['recon']:6.1f}  奖励CE {m['reward']:5.2f}  "
                  f"continue {m['cont']:5.3f}  原始KL {m['raw_kl']:5.2f}  非零奖励MAE {m['nz_rew_mae']:5.2f}  "
                  f"用时 {time.time() - t0:5.1f}s")

    # 保存权重供复用
    if not quick:
        torch.save(wm.state_dict(), OUT / "v3_world_model.pt")

    # 在留出验证回合上做开环预测并生成对比视频
    val_eid = n_train if n_train < n_eps else 0
    C = 10
    H = 15 if quick else 30
    ctx_o = preprocess_obs(torch.from_numpy(obs_all[val_eid, 1 : 1 + C][None]).to(DEVICE))
    ctx_a = torch.from_numpy(act_all[val_eid, 1 : 1 + C][None]).to(DEVICE)
    fut_a = torch.from_numpy(act_all[val_eid, 1 + C : 1 + C + H][None]).to(DEVICE)
    real_o = obs_all[val_eid, 1 + C : 1 + C + H]
    real_r = rew_all[val_eid, 1 + C : 1 + C + H]

    wm.eval()
    pred_o, pred_r = wm.open_loop_imagine(ctx_o, ctx_a, fut_a, sample=False)
    pred_img = (pred_o[0].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    pred_r_np = pred_r[0].cpu().numpy()

    ol_pix_mse = float(np.mean((real_o.astype(np.float32) / 255.0 - pred_img.astype(np.float32) / 255.0) ** 2))
    ol_rew_mae = float(np.mean(np.abs(pred_r_np - real_r)))
    zero_baseline_mae = float(np.mean(np.abs(real_r)))
    print(f"\n  验证回合开环预测（上下文 {C} 帧，纯先验想象 {H} 步）：")
    print(f"    开环像素 MSE = {ol_pix_mse:.5f} ｜ 开环非零奖励 MAE = {ol_rew_mae:.2f}（全猜 0 基线 MAE = {zero_baseline_mae:.2f}）")

    # 拼开环视频（左真图，右 V3 想象图）
    vframes = []
    for k in range(H):
        l = cv2.resize(real_o[k], (128, 128), interpolation=cv2.INTER_NEAREST)
        r = cv2.resize(pred_img[k], (128, 128), interpolation=cv2.INTER_NEAREST)
        sp = np.full((128, 6, 3), 235, dtype=np.uint8)
        vframes.append(np.concatenate([l, sp, r], axis=1))
    save_video(vframes, OUT / "08-v3-open-loop.mp4", fps=10)

    # 画四项训练损失曲线
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    plots = [
        ("recon", "重建损失（像素求和）", "C0"),
        ("reward", "奖励 Two-Hot 交叉熵", "C1"),
        ("cont", "Continue 二元交叉熵", "C2"),
        ("raw_kl", "未截断原始 KL (nat)", "C3"),
    ]
    for ax, (key, title, col) in zip(axes.flat, plots):
        ax.plot(hist[key], color=col, linewidth=1.8)
        if key == "raw_kl":
            ax.axhline(1.0, color="gray", linestyle=":", label="free bits = 1 nat")
            ax.legend(fontsize=8.5)
        ax.set_title(title)
        ax.set_xlabel("训练步数")
    save_figure(fig, OUT / "08-v3-world-model-losses.png")


def main():
    seed_everything(SEED)
    if DEVICE.type == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    verify_section_8()
    train_v3_world_model()
    print("\n" + "=" * 60)
    print("本节完成！下一节：09-dreamer-v3-agent.py（组装完整 Dreamer V3 智能体并训练）")
    print("=" * 60)


if __name__ == "__main__":
    main()
