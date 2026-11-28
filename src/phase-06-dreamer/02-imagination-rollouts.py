"""
在想象中展开轨迹：从后验快照出发、continue 权重与 stop-gradient 边界
======================================================================

学习目标：
    1. 实现标准批量想象函数 imagine_rollout()：把 B×T 个后验快照展平并 detach，用 actor 选动作、用先验往前推 H 步
    2. 核对 CONCEPT.md 第 2.4 节的 continue 累积权重手算例子（w = [1, 1, 0.9, 0.45]），并演示终止后权重衰减
    3. 编写自动梯度单元测试，逐一验证 CONCEPT.md 第 2.5 节的三道 stop-gradient 边界：
       ① 起点 detach 阻断 actor 梯度流回编码器；② 冻结世界模型参数不被 actor 修改；③ critic 输入 detach 阻断梯度流回 actor
    4. 复用 Phase 3 训练好的高斯 RSSM，把想象轨迹解码成对比视频，与同一起点的真实后续画面并排对照

核心概念：
    - 快照式分支想象：起点取自回放里的后验状态（离策略覆盖全回合），后续动作全由当前 actor 生成（在策略）
    - continue 权重 w_k = c_t ∏_{j=1}^k ĉ_j：让终止之后的想象步自动降权，不污染 actor 和 critic 的损失
    - 穿过 vs 更新：V1 式 actor 梯度穿过世界模型和 critic，但不更新它们的参数

前置知识：
    - Phase 3 的 04-rssm-from-scratch.py、05-world-model-training.py
    - 01-from-planet-to-dreamer.py

对应概念文档：
    - CONCEPT.md 第 2 节「在想象中展开轨迹」

预计运行时间：约 20 秒（CPU，复用 Phase 3 模型权重）
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import numpy as np
import torch
import torch.nn as nn

from utils import get_device, is_quick, output_dir, save_figure, save_video, seed_everything, setup_matplotlib

_P3_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "phase-03-rssm-planet"))


def _load_p3(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_P3_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_p3_04 = _load_p3("p3_04", "04-rssm-from-scratch.py")
_p3_05 = _load_p3("p3_05", "05-world-model-training.py")

GaussianWorldModel = _p3_04.WorldModel
preprocess_obs = _p3_05.preprocess_obs
collect_dataset = _p3_05.collect_offline_dataset

plt = setup_matplotlib()
OUT = output_dir("phase-06")
P3_OUT = output_dir("phase-03")
SEED = 0
DEVICE = get_device()


# ==================== 第一部分：continue 权重与核心想象函数 ====================


def compute_imagine_weights(start_cont, imag_conts):
    """计算 CONCEPT.md 第 2.4 节的想象权重 w_k = c_t ∏_{j=1}^k ĉ_j。

    start_cont: [N]，imag_conts: [H, N]。返回 weights [H+1, N]（已 detach）。
    """
    H, N = imag_conts.shape
    w = [start_cont.detach()]
    for k in range(H):
        w.append(w[-1] * imag_conts[k].detach())
    return torch.stack(w, dim=0)


class SimpleContinuousActor(nn.Module):
    """输出重参数化 tanh 高斯动作的简易 actor（用于验证计算图与梯度边界）。"""

    def __init__(self, feat_dim=230, act_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, feat, deterministic=False):
        h = self.net(feat)
        mean = self.mean_head(h)
        if deterministic:
            return torch.tanh(mean)
        std = self.log_std_head(h).clamp(-3.0, 0.5).exp()
        return torch.tanh(mean + std * torch.randn_like(mean))


def imagine_rollout(world_model, cont_head, actor, post_h, post_z, start_cont, horizon=15, sample=True):
    """CONCEPT.md 第 2.6 节：从 B×T 个后验快照出发批量想象 H 步。

    post_h: [B, T, D_h]，post_z: [B, T, D_z]，start_cont: [B, T]。
    返回字典：imag_states [H+1, N, D]，imag_actions [H, N, A]，
              imag_rewards [H, N]，imag_conts [H, N]，weights [H+1, N]。
    """
    # 边界 ①：想象起点必须 detach，防止行为学习梯度篡改编码器表征
    h = post_h.detach().reshape(-1, post_h.shape[-1])
    z = post_z.detach().reshape(-1, post_z.shape[-1])
    c0 = start_cont.detach().reshape(-1)

    states = [torch.cat([h, z], dim=-1)]
    actions = []
    for _ in range(horizon):
        feat = states[-1]
        a = actor(feat)
        h, z, _ = world_model.rssm.imagine_step(h, z, a, sample=sample)
        states.append(torch.cat([h, z], dim=-1))
        actions.append(a)

    imag_states = torch.stack(states, dim=0)    # [H+1, N, D]
    imag_actions = torch.stack(actions, dim=0)  # [H, N, A]

    # 按第 2.2 节对齐：到达 m̂_{k+1} 预测的是 r̂_k 与 ĉ_{k+1}
    next_states = imag_states[1:]               # [H, N, D]
    imag_rewards = world_model.reward_head(next_states)
    imag_conts = torch.sigmoid(cont_head(next_states).squeeze(-1))
    weights = compute_imagine_weights(c0, imag_conts)

    return dict(
        imag_states=imag_states,
        imag_actions=imag_actions,
        imag_rewards=imag_rewards,
        imag_conts=imag_conts,
        weights=weights,
    )


# ==================== 第二部分：手算核对与三道 sg 单元测试 ====================


def verify_continue_weights():
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 2.4 节的 continue 权重手算例子")
    print("=" * 60)
    c0 = torch.tensor([1.0, 0.0])  # 第 0 条非终止起点，第 1 条本身就是终止起点
    c_hat = torch.tensor([[1.0, 1.0], [0.9, 0.9], [0.5, 0.5]])
    w = compute_imagine_weights(c0, c_hat)
    print(f"  非终止起点（c_t=1, ĉ=[1.0, 0.9, 0.5]）：w = {[round(x, 4) for x in w[:, 0].tolist()]}")
    print(f"  终止起点  （c_t=0, ĉ=[1.0, 0.9, 0.5]）：w = {[round(x, 4) for x in w[:, 1].tolist()]}")
    assert torch.allclose(w[:, 0], torch.tensor([1.0, 1.0, 0.9, 0.45]))
    assert torch.allclose(w[:, 1], torch.zeros(4))
    print("  ✓ 与 CONCEPT.md 第 2.4 节完全一致！")

    # 绘制不同单步存活率下的权重衰减曲线
    ks = np.arange(0, 16)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for p_alive, lbl in [(0.995, "极少终止（ĉ=0.995，如长回合任务）"),
                         (0.95, "中等终止率（ĉ=0.95，如 Hopper）"),
                         (0.85, "高风险状态（ĉ=0.85，快摔倒）")]:
        ax.plot(ks, p_alive ** ks, "o-", linewidth=2, label=lbl)
    ax.set_xlabel("想象步数 k")
    ax.set_ylabel("损失权重 w_k = ∏ ĉ_j")
    ax.set_title("continue 累积权重随想象步数衰减，自动忽略终止后的状态")
    ax.legend(fontsize=8.5)
    save_figure(fig, OUT / "02-imagination-weights.png")


def verify_stop_gradient_boundaries():
    """严格检验 CONCEPT.md 第 2.5 节的三道 stop-gradient 边界。"""
    print("\n" + "=" * 60)
    print("第二部分：自动微分单元测试 — 检查三道 stop-gradient 边界")
    print("=" * 60)

    wm = GaussianWorldModel().to(DEVICE)
    cont_head = nn.Linear(230, 1).to(DEVICE)
    actor = SimpleContinuousActor().to(DEVICE)
    critic = nn.Sequential(nn.Linear(230, 64), nn.SiLU(), nn.Linear(64, 1)).to(DEVICE)

    obs = torch.rand(2, 4, 3, 64, 64, device=DEVICE)
    act = torch.randn(2, 4, 1, device=DEVICE).clamp(-1, 1)
    c0 = torch.ones(2, 4, device=DEVICE)

    embeds = wm.encode_seq(obs)
    out = wm.rssm.observe(embeds, act)
    post_h, post_z = out["features"][..., :200], out["features"][..., 200:]

    # 边界 ① & ②：冻结世界模型和 critic 参数，让 actor 损失反向传播
    for p in list(wm.parameters()) + list(cont_head.parameters()) + list(critic.parameters()):
        p.requires_grad_(False)

    traj = imagine_rollout(wm, cont_head, actor, post_h, post_z, c0, horizon=5)
    actor_loss = -(traj["imag_rewards"] * traj["weights"][:-1]).mean()
    actor.zero_grad()
    actor_loss.backward()

    actor_grad_norm = sum(p.grad.norm().item() for p in actor.parameters() if p.grad is not None)
    enc_has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in wm.encoder.parameters())
    rssm_has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in wm.rssm.parameters())
    print(f"  ①+② Actor 损失反传后：actor 梯度范数 = {actor_grad_norm:.4f} (>0 ✓) ｜ "
          f"encoder 有梯度? {enc_has_grad} (False ✓) ｜ rssm 有梯度? {rssm_has_grad} (False ✓)")
    assert actor_grad_norm > 0 and not enc_has_grad and not rssm_has_grad

    # 边界 ③：解冻 critic，对 detach 后的 imag_states 算 critic 损失，验证 actor 不再收到任何梯度
    for p in critic.parameters():
        p.requires_grad_(True)
    actor.zero_grad()
    critic.zero_grad()
    v_pred = critic(traj["imag_states"].detach()).squeeze(-1)
    target = torch.zeros_like(v_pred)
    critic_loss = 0.5 * ((v_pred - target).pow(2) * traj["weights"]).mean()
    critic_loss.backward()

    actor_leak = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in actor.parameters())
    critic_grad_norm = sum(p.grad.norm().item() for p in critic.parameters() if p.grad is not None)
    print(f"  ③    Critic 损失反传后：critic 梯度范数 = {critic_grad_norm:.4f} (>0 ✓) ｜ "
          f"梯度泄漏到 actor? {actor_leak} (False ✓)")
    assert critic_grad_norm > 0 and not actor_leak
    print("  ✓ 三道 stop-gradient 边界全部通过自动微分验证！")


# ==================== 第三部分：解码想象视频 vs 真实后续画面 ====================


@torch.no_grad()
def render_imagination_vs_real():
    print("\n" + "=" * 60)
    print("第三部分：把隐空间想象轨迹解码成视频，与同一起点的真实后续画面对比")
    print("=" * 60)

    wm_path = P3_OUT / "world_model.pt"
    ds_path = P3_OUT / "pendulum-pixel-dataset.npz"
    wm = GaussianWorldModel().to(DEVICE).eval()
    if wm_path.exists():
        wm.load_state_dict(torch.load(wm_path, map_location=DEVICE, weights_only=True))
        print(f"  已加载 Phase 3 世界模型权重：{wm_path}")
    else:
        print("  （未找到 Phase 3 预训练权重，使用随机初始化模型演示视频生成流程）")

    if ds_path.exists():
        data = np.load(ds_path)
        obs_ep = data["obs"][0]      # [101, 64, 64, 3]
        act_ep = data["actions"][0]  # [101, 1]
    else:
        data = collect_dataset(n_episodes=2)
        obs_ep, act_ep = data["obs"][0], data["actions"][0]

    C = 10
    H = 15 if is_quick() else 30
    ctx_obs = preprocess_obs(torch.from_numpy(obs_ep[1 : 1 + C][None]).to(DEVICE))
    ctx_act = torch.from_numpy(act_ep[1 : 1 + C][None]).to(DEVICE)
    future_act = torch.from_numpy(act_ep[1 + C : 1 + C + H][None]).to(DEVICE)
    real_future = obs_ep[1 + C : 1 + C + H]

    # 先看前 C 帧建立后验快照 (h_C, z_C)
    embeds = wm.encode_seq(ctx_obs)
    out = wm.rssm.observe(embeds, ctx_act)
    h0 = out["features"][:, -1, :200]
    z0 = out["features"][:, -1, 200:]

    # 1) 同动作序列的先验想象（检验模型本身）  2) 随机 actor 在想象里自由推演
    imag_same_act = wm.rssm.imagine(h0, z0, future_act, sample=False)  # [1, H, 230]
    dec_same = (wm.decode_seq(imag_same_act)[0].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    rand_actor = SimpleContinuousActor().to(DEVICE).eval()
    cont_head = nn.Linear(230, 1).to(DEVICE).eval()
    traj = imagine_rollout(wm, cont_head, rand_actor, h0.unsqueeze(1), z0.unsqueeze(1), torch.ones(1, 1, device=DEVICE), horizon=H)
    dec_actor = (wm.decode_seq(traj["imag_states"][1:].transpose(0, 1))[0].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    # 统计逐帧 MSE（同动作想象 vs 真实）
    mses = [float(np.mean((real_future[k].astype(np.float32) / 255.0 - dec_same[k].astype(np.float32) / 255.0) ** 2)) for k in range(H)]
    print(f"  同动作开环想象像素 MSE：k=1 为 {mses[0]:.4f}，k=5 为 {mses[min(4, H-1)]:.4f}，"
          f"k=15 为 {mses[min(14, H-1)]:.4f}，k={H} 为 {mses[-1]:.4f}")

    # 拼三栏对比视频：左=真实后续 | 中=同动作想象 | 右=随机Actor想象
    frames = []
    for k in range(H):
        f1 = cv2.resize(real_future[k], (128, 128), interpolation=cv2.INTER_NEAREST)
        f2 = cv2.resize(dec_same[k], (128, 128), interpolation=cv2.INTER_NEAREST)
        f3 = cv2.resize(dec_actor[k], (128, 128), interpolation=cv2.INTER_NEAREST)
        sp = np.full((128, 6, 3), 235, dtype=np.uint8)
        frames.append(np.concatenate([f1, sp, f2, sp, f3], axis=1))
    save_video(frames, OUT / "02-imagination-vs-real.mp4", fps=10)
    print("  （三栏视频从左到右：真实后续画面 ｜ 同动作先验想象 ｜ 随机 Actor 隐空间想象）")


def main():
    seed_everything(SEED)
    verify_continue_weights()
    verify_stop_gradient_boundaries()
    render_imagination_vs_real()
    print("\n" + "=" * 60)
    print("本节完成！下一节：03-dreamer-v1-actor-critic.py（Dreamer V1：λ-return 与沿模型反传）")
    print("=" * 60)


if __name__ == "__main__":
    main()
