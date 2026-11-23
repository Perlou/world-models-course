# Phase 6: Dreamer——在想象中学习行为

> **学习目标**：理解并实现 Dreamer V1 → V3：在 RSSM 的隐空间里想象轨迹，用想象出的轨迹训练 actor 和 critic
> **预计时长**：3 周（第 11–13 周）
> **前置**：Phase 3（RSSM）、Phase 4（λ-return）、Phase 5（actor-critic、三种梯度估计器）
> **状态**：README ✅ ｜ CONCEPT.md ⏳ ｜ 讲义 0/10 ⏳ ｜ 项目 ⏳

## 🧭 在课程中的位置

这是全课的核心章。前五章的内容在这里汇合：

- Phase 3 的 **RSSM** 提供世界模型
- Phase 4 的 **λ-return** 提供价值目标
- Phase 5 的 **actor-critic** 和**梯度估计器**提供行为学习的办法

PlaNet 每走一步都要在线规划，慢，而且只看得到规划视野之内。
Dreamer 的做法是：在世界模型里批量想象轨迹，用这些想象出的轨迹训练一个策略和一个价值函数。
决策时只需要一次前向计算，价值函数还能把视野延伸到想象长度之外。

三代 Dreamer 的主线：

- **V1**：连续隐变量，actor 的梯度沿可微的世界模型反传
- **V2**：换成类别隐变量 + straight-through，加入 KL balancing
- **V3**：一组稳健性技巧（symlog、two-hot、回报归一化等），目标是一套超参数通吃多种任务

学完本章后，Phase 7–10 的各种世界模型都可以看成"换掉 Dreamer 的某一个部件"。

## 📚 本阶段内容

### 1. 从规划到学习行为

- `01-from-planet-to-dreamer.py` - PlaNet 的短板；Dreamer 的整体结构
- `02-imagination-rollouts.py` - 从后验状态出发批量想象轨迹；stop-gradient 放在哪

### 2. Dreamer V1 与 V2

- `03-dreamer-v1-actor-critic.py` - λ-return 价值目标 + 沿模型反传的 actor
- `04-categorical-latents-straight-through.py` - 类别隐变量、straight-through 梯度
- `05-kl-balancing-and-free-bits.py` - KL balancing 与 free bits

### 3. Dreamer V3 的稳健性技巧

- `06-symlog-and-twohot.py` - symlog 变换与 two-hot 回归
- `07-return-normalization-and-robustness.py` - 回报的分位数归一化、unimix、critic 的 EMA 正则

### 4. 完整的 Dreamer V3

- `08-dreamer-v3-world-model.py` - 组装世界模型：编码器、RSSM、解码器、奖励与 continue 预测头
- `09-dreamer-v3-agent.py` - 组装完整智能体并训练
- `10-diagnostics-and-ablations.py` - 诊断（想象回报与真实回报是否一致）与消融实验

## 🧠 核心概念

详见 [CONCEPT.md](./CONCEPT.md)《Dreamer：在想象中学习行为》：

1. 从规划到学习行为：PlaNet 的短板
2. 在想象中展开轨迹
3. Dreamer V1：沿可微的世界模型反向传播
4. Dreamer V2：类别隐变量与 straight-through
5. KL balancing 与 free bits
6. Dreamer V3（一）：symlog 与 two-hot
7. Dreamer V3（二）：回报归一化与其他稳健性技巧
8. Dreamer V3 世界模型的完整损失
9. Dreamer V3 的 actor 与 critic
10. 诊断与消融：怎么知道哪里出了问题

## 🎯 实战项目

**Mini-Dreamer**（`projects/phase-06-mini-dreamer/`）：把本章的讲义整理成一个可复用的 Dreamer V3 小型实现，
在像素控制任务上训练，并加入 Phase 5 的排行榜，和 PPO、SAC、MBPO 比样本效率。
本机跑缩小版；完整规模的一次对照实验放在 4090 上。

## 📖 论文与资料

| 论文 | 编号 | 必读 | 说明 |
| --- | --- | --- | --- |
| Hafner et al. 2020, Dreamer V1 | arXiv:1912.01603 | ⭐⭐⭐⭐⭐ | 在隐空间想象中学行为 |
| Hafner et al. 2021, Dreamer V2 | arXiv:2010.02193 | ⭐⭐⭐⭐ | 类别隐变量、KL balancing |
| Hafner et al. 2023, Dreamer V3 | arXiv:2301.04104；Nature 640 (2025) | ⭐⭐⭐⭐⭐ | 稳健性技巧，一套超参数跨领域 |
| Wu et al. 2022, DayDreamer | arXiv:2206.14176 | ⭐⭐⭐ | Dreamer 在真实机器人上在线学习 |
| Sekar et al. 2020, Plan2Explore | arXiv:2005.05960 | ⭐⭐ | 用世界模型的不确定性驱动探索 |
| Hafner et al. 2022, Director | arXiv:2206.04114 | ⭐ | 在世界模型上做分层规划，选读 |

## 💻 算力与环境

- 本机（MPS）：像素版 `Pendulum-v1` 等小任务的缩小版，预计每次训练数小时（未实测的估算）。
- 4090：完整规模的对照实验（如 DeepMind Control Suite 的像素任务）。
  `dm-control` 与 MuJoCo 的版本搭配在写本章讲义时验证。

## ✅ 学习检查

- [ ] 能画出 Dreamer 的三个训练环节：学世界模型、在想象中学行为、与环境交互
- [ ] 能凭记忆写出 Dreamer V3 世界模型的损失，并标出每个 stop-gradient
- [ ] 能解释 V1 → V2 → V3 的每一处改动解决了什么问题
- [ ] 能解释 symlog 和 two-hot 分别应对什么情况
- [ ] 能解释想象轨迹为什么从回放数据的后验状态出发，而不是从初始状态出发
- [ ] 自己的实现在同等环境步数下样本效率优于自己写的 PPO
