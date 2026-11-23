# Phase 8: Token 与 Transformer 世界模型

> **学习目标**：把画面离散成 token，用 Transformer 做世界模型；实现 VQ-VAE、IRIS、STORM 的精简版
> **预计时长**：2 周（第 16–17 周）
> **前置**：Phase 6（Dreamer）；`deep-learning-course` 的 Transformer / GPT
> **状态**：README ✅ ｜ CONCEPT.md ⏳ ｜ 讲义 0/8 ⏳ ｜ 项目 ⏳

## 🧭 在课程中的位置

从本章起进入第三部分：生成式与表征式世界模型。前七章的世界模型骨干都是 RNN（RSSM 里的 GRU）。
语言模型已经证明了 Transformer 在序列建模上的能力，本章把它搬到世界模型里。

关键的一步是把连续的画面变成离散的 token。一旦画面是 token 序列，
"预测下一帧"就和"预测下一个词"是同一个问题，GPT 的全套做法可以直接用。

本章有两条路线：

- **IRIS**：每帧切成多个 token，自回归地逐个预测。保真度高，但想象很慢
- **STORM / TWM**：每帧只用一个隐变量（沿用 Dreamer 的类别隐变量），Transformer 只管时间维度。快得多

这里学的 tokenizer（VQ-VAE、FSQ）和并行解码（MaskGIT）是 Phase 9 Genie 的基础。

从本章起，项目换到 PushT 推块任务上，Phase 8–11 会在同一个任务上对比 Token、扩散、JEPA 三类世界模型。

## 📚 本阶段内容

### 1. 把画面变成 token

- `01-vq-vae-from-scratch.py` - VQ-VAE：码本、commitment loss、straight-through、码本坍塌
- `02-fsq-and-tokenizer-quality.py` - FSQ；tokenizer 的重建质量如何限制世界模型的上限

### 2. Transformer 当世界模型

- `03-gpt-as-world-model.py` - 把 minGPT 改成世界模型：帧 token 与动作 token 交错排列
- `04-kv-cache-and-imagination.py` - 自回归想象的工程：KV cache 与速度账

### 3. 两条路线

- `05-iris.py` - IRIS：离散 tokenizer + 自回归 Transformer + 在想象中训练 actor-critic
- `06-storm-and-twm.py` - STORM 与 TWM：每帧一个类别隐变量 + Transformer

### 4. 并行解码与对比

- `07-maskgit-parallel-decoding.py` - MaskGIT：掩码建模与迭代并行解码
- `08-rssm-vs-transformer.py` - 在同一份数据上对比 RSSM 与 Transformer 的长时记忆

## 🧠 核心概念

详见 [CONCEPT.md](./CONCEPT.md)《Token 与 Transformer 世界模型》：

1. 把画面变成 token：VQ-VAE
2. FSQ 与 tokenizer 的质量
3. 把世界模型写成序列模型
4. 自回归想象的工程：KV cache 与速度账
5. IRIS：token 级的自回归世界模型
6. STORM 与 TWM：每帧一个隐变量
7. MaskGIT：并行解码
8. RSSM 还是 Transformer

## 🎯 实战项目

**PushT Token 世界模型**（`projects/phase-08-pusht-token-world-model/`）：在 PushT 的演示数据上训练
tokenizer + Transformer 世界模型，评估开环预测质量随步数的变化，并用它做基于模型的规划。
这是 Phase 8–11 贯穿任务的第一站。

## 📖 论文与资料

| 论文 | 编号 | 必读 | 说明 |
| --- | --- | --- | --- |
| Micheli et al. 2023, IRIS | arXiv:2209.00588 | ⭐⭐⭐⭐⭐ | token 级自回归世界模型 |
| van den Oord et al. 2017, VQ-VAE | arXiv:1711.00937 | ⭐⭐⭐⭐ | 离散表示学习 |
| Zhang et al. 2023, STORM | arXiv:2310.09615 | ⭐⭐⭐⭐ | 每帧一个隐变量的高效 Transformer 世界模型 |
| Robine et al. 2023, TWM | arXiv:2303.07109 | ⭐⭐⭐ | Transformer 世界模型的另一种设计 |
| Mentzer et al. 2023, FSQ | arXiv:2309.15505 | ⭐⭐⭐ | 不需要码本的量化 |
| Chang et al. 2022, MaskGIT | arXiv:2202.04200 | ⭐⭐⭐ | 掩码生成式 Transformer，Genie 的动力学模型用它 |
| Micheli et al. 2024, Δ-IRIS | arXiv:2406.19320 | ⭐⭐ | 只编码帧间变化，减少 token 数 |
| Kaiser et al. 2020, SimPLe | arXiv:1903.00374 | ⭐⭐ | Atari 100k 基准的出处 |
| Esser et al. 2021, VQGAN | arXiv:2012.09841 | ⭐⭐ | 加入感知损失和对抗损失的 tokenizer |
| Yu et al. 2023, MAGVIT-v2 | arXiv:2310.05737 | ⭐⭐ | 视频 tokenizer 与 lookup-free 量化 |
| Chen et al. 2021, Decision Transformer | arXiv:2106.01345 | ⭐ | 对照阅读：把决策本身当序列建模 |
| Janner et al. 2021, Trajectory Transformer | arXiv:2106.02039 | ⭐ | 对照阅读：轨迹级序列模型 + 束搜索规划 |
| Schwarzer et al. 2023, BBF | arXiv:2305.19452 | ⭐ | Atari 100k 上的无模型强基线，用来校准预期 |

## 💻 算力与环境

- 环境：PushT（`gym-pusht`，数据来自 LeRobot 的 `lerobot/pusht`）。
  `gym-pusht` 依赖 `pygame`，与 Gymnasium 经典控制依赖的 `pygame-ce` 冲突，本章起的 PushT 项目用独立环境。
- 本机跑缩小版；完整规模的训练建议用 4090。

## ✅ 学习检查

- [ ] 能写出 VQ-VAE 的三项损失，并解释 straight-through 在这里的作用
- [ ] 能解释码本坍塌是什么，FSQ 为什么没有这个问题
- [ ] 能说出把一段轨迹排成 token 序列的方式，以及动作 token 放在哪
- [ ] 能算出 IRIS 想象一步需要多少次前向计算，并解释 KV cache 省掉了什么
- [ ] 能对比 IRIS 与 STORM 的设计取舍
- [ ] 能解释 MaskGIT 的迭代解码过程
