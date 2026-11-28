"""
MuZero 的三个网络：表征、动力学、预测与隐状态缩放
=================================================

学习目标：
    1. 实现 scale_hidden_state()（逐样本 min-max 缩放到 [0, 1]），核对 CONCEPT.md 第 4.6 节手算 (2, -1, 5) → (0.5, 0, 1)
    2. 核对 CONCEPT.md 第 4.3 节随机环境抛硬币期望手算（确定性隐状态把两分支平均成 0.5 vs 真观测可得 1.0）
    3. 实现 MuZeroNet 的三个子网络（enc / dyn / pred）及搜索专用的两个推理接口：
       initial_inference(obs) 与 recurrent_inference(z, action)
    4. 在真实 CartPole-v1 转移上测量未训练模型的 z_t^1（推一步）与 z_{t+1}^0（编码下一观测）的余弦相似度，
       为 05 节训练后对照与 06 节 EfficientZero 一致性损失埋下伏笔

核心概念：
    - 三个网络 = 编译器前端(enc) / 中端优化器(dyn) / 后端(pred)，无需解码器
    - 两个推理接口：initial_inference 根节点调用 1 次，recurrent_inference 每次模拟扩展调用 1 次
    - 隐状态自由度：原始 MuZero 不约束 z_t^1 ≈ z_{t+1}^0

前置知识：
    - 03-alphazero-mini.py

对应概念文档：
    - CONCEPT.md 第 4 节「MuZero 的三个网络」

预计运行时间：约 5 秒（CPU）
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import seed_everything

SEED = 0


def scale_hidden_state(z):
    """CONCEPT.md 第 4.6 节：逐样本 min-max 归一化到 [0, 1]，防止反复展开数值爆炸。"""
    z_min = z.min(dim=-1, keepdim=True).values
    z_max = z.max(dim=-1, keepdim=True).values
    return (z - z_min) / (z_max - z_min).clamp_min(1e-5)


class MuZeroNet(nn.Module):
    """精简版 MuZero 三个网络（适用于低维状态离散动作环境如 CartPole-v1 或井字棋）。"""

    def __init__(self, obs_dim, act_dim, latent_dim=32, hidden=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = latent_dim

        # 1. 表征函数 enc: o_t → z^0
        self.enc = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        # 2. 动力学函数 dyn: (z^{k-1}, one_hot(a)) → (z^k, r̂^k)
        self.dyn_trunk = nn.Sequential(
            nn.Linear(latent_dim + act_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.dyn_state = nn.Linear(hidden, latent_dim)
        self.dyn_reward = nn.Linear(hidden, 1)

        # 3. 预测函数 pred: z^k → (policy_logits, value)
        self.pred_trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pred_policy = nn.Linear(hidden, act_dim)
        self.pred_value = nn.Linear(hidden, 1)

    def initial_inference(self, obs):
        z0 = scale_hidden_state(self.enc(obs))
        h = self.pred_trunk(z0)
        return z0, self.pred_policy(h), self.pred_value(h).squeeze(-1)

    def recurrent_inference(self, z_prev, action_idx):
        a_oh = F.one_hot(action_idx.long(), num_classes=self.act_dim).to(z_prev.dtype)
        dh = self.dyn_trunk(torch.cat([z_prev, a_oh], dim=-1))
        z_next = scale_hidden_state(self.dyn_state(dh))
        reward = self.dyn_reward(dh).squeeze(-1)
        ph = self.pred_trunk(z_next)
        return z_next, reward, self.pred_policy(ph), self.pred_value(ph).squeeze(-1)


@torch.no_grad()
def measure_latent_alignment(net, n_transitions=200, seed=0):
    """测量 z_t^1 = dyn(enc(o_t), a_t) 与 z_{t+1}^0 = enc(o_{t+1}) 之间的平均余弦相似度。"""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    obs_list, act_list, nxt_list = [], [], []
    obs, _ = env.reset(seed=seed)
    for _ in range(n_transitions):
        a = int(rng.integers(0, 2))
        nxt, _, term, trunc, _ = env.step(a)
        obs_list.append(obs.astype(np.float32))
        act_list.append(a)
        nxt_list.append(nxt.astype(np.float32))
        obs = nxt
        if term or trunc:
            obs, _ = env.reset()
    env.close()

    o_t = torch.from_numpy(np.asarray(obs_list))
    a_t = torch.tensor(act_list, dtype=torch.long)
    o_nxt = torch.from_numpy(np.asarray(nxt_list))

    z0, _, _ = net.initial_inference(o_t)
    z1, _, _, _ = net.recurrent_inference(z0, a_t)
    z_nxt_0, _, _ = net.initial_inference(o_nxt)

    # 先去均值再算余弦（因为 scale_hidden_state 把向量压进非负象限 [0, 1]，不去均值会虚高）
    cos_raw = F.cosine_similarity(z1, z_nxt_0, dim=-1).mean().item()
    cos_centered = F.cosine_similarity(z1 - z1.mean(dim=-1, keepdim=True), z_nxt_0 - z_nxt_0.mean(dim=-1, keepdim=True), dim=-1).mean().item()
    return cos_raw, cos_centered


def main():
    seed_everything(SEED)
    print("=" * 60)
    print("第一部分：核对 CONCEPT.md 第 4.3、4.6 节手算例子")
    print("=" * 60)
    z_ex = torch.tensor([[2.0, -1.0, 5.0]])
    z_sc = scale_hidden_state(z_ex)
    print(f"  1. 隐状态缩放：(2, -1, 5) → {[round(x, 3) for x in z_sc[0].tolist()]}（手算 [0.5, 0.0, 1.0] ✓）")
    assert torch.allclose(z_sc, torch.tensor([[0.5, 0.0, 1.0]]))

    # 确定性模型在抛硬币随机环境下的条件期望
    e_det = 0.5 * 1.0 + 0.5 * 0.0
    print(f"  2. 随机抛硬币示例：真观测后最优奖励 = 1.0，确定性隐状态预测的条件期望 = {e_det:.2f} ✓")

    print("\n" + "=" * 60)
    print("第二部分：检验 MuZeroNet 张量形状与未训练隐状态对齐度（CartPole-v1）")
    print("=" * 60)
    net = MuZeroNet(obs_dim=4, act_dim=2, latent_dim=32, hidden=64)
    obs = torch.randn(8, 4)
    act = torch.randint(0, 2, (8,))
    z0, p0, v0 = net.initial_inference(obs)
    z1, r1, p1, v1 = net.recurrent_inference(z0, act)
    print(f"  initial_inference  : z0 {list(z0.shape)} ∈ [{z0.min():.2f}, {z0.max():.2f}], p0 {list(p0.shape)}, v0 {list(v0.shape)}")
    print(f"  recurrent_inference: z1 {list(z1.shape)} ∈ [{z1.min():.2f}, {z1.max():.2f}], r1 {list(r1.shape)}, p1 {list(p1.shape)}, v1 {list(v1.shape)}")
    assert z0.shape == (8, 32) and z1.shape == (8, 32) and r1.shape == (8,)

    cos_raw, cos_c = measure_latent_alignment(net)
    print(f"\n  未训练模型（200 条 CartPole-v1 真实转移）：")
    print(f"    z_t^1 与 z_{{t+1}}^0 原始余弦相似度 = {cos_raw:.4f} ｜ 去中心化余弦相似度 = {cos_c:+.4f}（几乎无关！）")
    print("    （下一节 05-muzero-agent.py 会在纯 MuZero 训练后再次测量这个数字！）")

    print("\n" + "=" * 60)
    print("本节完成！下一节：05-muzero-agent.py（K 步展开训练与隐空间 MCTS）")
    print("=" * 60)


if __name__ == "__main__":
    main()
