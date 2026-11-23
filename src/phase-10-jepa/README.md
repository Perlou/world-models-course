# Phase 10: JEPA——在表征空间里预测与规划

> **学习目标**：理解联合嵌入预测架构（JEPA）：不生成像素，在表征空间里预测；实现 I-JEPA 精简版和基于冻结特征的世界模型规划
> **预计时长**：2 周（第 21–22 周）
> **前置**：Phase 1（CEM）、Phase 7（不重建观测的思路）；`deep-learning-course` 的 ViT / Transformer
> **状态**：README ✅ ｜ CONCEPT.md ⏳ ｜ 讲义 0/8 ⏳ ｜ 项目 ⏳

## 🧭 在课程中的位置

Phase 8、9 的世界模型是生成式的：预测的是 token 或像素。
生成每一个像素很贵，而且很多细节（树叶怎么晃、水面的反光）本来就不可预测，也和决策无关。

JEPA 的主张是：**只在表征空间里预测**。编码器把观测变成表征，预测器预测未来观测的表征，
全程不解码回像素。

这和 Phase 7 的"不重建"一脉相承，但更进一步：MuZero 和 TD-MPC2 靠奖励信号来塑造表征，
JEPA 连奖励都不用，纯自监督。代价是要面对一个新问题：**表征坍塌**。
如果编码器对所有输入都输出同一个向量，预测误差直接为零。本章有相当篇幅在讲怎么防坍塌。

对机器人方向，这一章很实用：拿一个预训练好的视觉编码器（DINOv2、V-JEPA 2）冻住，
只训练一个小的动作条件预测器，再用 Phase 1 的 CEM 在特征空间里规划，
就能做零样本的目标到达。这就是 DINO-WM 和 V-JEPA 2-AC 的做法，也是本章项目要做的事。

## 📚 本阶段内容

### 1. 为什么在表征空间里预测

- `01-generative-vs-joint-embedding.py` - 生成式预测与联合嵌入预测的对比；能量模型的视角
- `02-collapse-and-regularization.py` - 表征坍塌；EMA 目标网络、stop-gradient、VICReg 的玩具实验

### 2. JEPA 家族

- `03-i-jepa-from-scratch.py` - I-JEPA：上下文编码器、目标编码器、预测器、掩码策略
- `04-v-jepa-video.py` - V-JEPA：视频的时空掩码与特征预测
- `05-probing-latent-representations.py` - 线性探针：表征里有没有位置、速度这些物理量

### 3. 用表征做世界模型

- `06-action-conditioned-latent-dynamics.py` - 在冻结的预训练特征上学动作条件的动力学
- `07-planning-in-feature-space.py` - 在特征空间里做目标条件的 CEM 规划
- `08-end-to-end-jepa-world-models.py` - 端到端从像素训练 JEPA 世界模型：PLDM、LeJEPA、LeWorldModel

## 🧠 核心概念

详见 [CONCEPT.md](./CONCEPT.md)《JEPA：在表征空间里预测与规划》：

1. 生成式预测与联合嵌入预测
2. 表征坍塌与防坍塌的办法
3. I-JEPA：图像上的 JEPA
4. V-JEPA 与 V-JEPA 2：视频上的 JEPA
5. 探针：表征里到底有什么
6. 动作条件的隐空间动力学
7. 在特征空间里规划
8. 端到端的 JEPA 世界模型

## 🎯 实战项目

**JEPA 规划器**（`projects/phase-10-jepa-planner/`）：在 PushT 上，用冻结的预训练视觉特征 + 自己训练的
动作条件预测器 + CEM，做目标图像条件的规划。与 Phase 8、9 在同一任务上的结果放在一起对比。

## 📖 论文与资料

| 论文 | 编号 | 必读 | 说明 |
| --- | --- | --- | --- |
| LeCun 2022, A Path Towards Autonomous Machine Intelligence | OpenReview 立场论文 | ⭐⭐⭐⭐ | JEPA 的思想来源，读前半部分 |
| Assran et al. 2023, I-JEPA | arXiv:2301.08243 | ⭐⭐⭐⭐⭐ | 图像上的 JEPA |
| Zhou et al. 2024, DINO-WM | arXiv:2411.04983 | ⭐⭐⭐⭐⭐ | 冻结 DINOv2 特征 + 预测器 + 规划 |
| Assran et al. 2025, V-JEPA 2 | arXiv:2506.09985 | ⭐⭐⭐⭐ | 视频预训练 + 动作条件预测器，用于机器人规划 |
| Bardes et al. 2024, V-JEPA | arXiv:2404.08471 | ⭐⭐⭐ | 视频上的特征预测 |
| Bardes et al. 2022, VICReg | arXiv:2105.04906 | ⭐⭐⭐⭐ | 用方差、协方差正则防坍塌 |
| Grill et al. 2020, BYOL | arXiv:2006.07733 | ⭐⭐⭐ | EMA 目标网络，不用负样本 |
| Sobal et al. 2025, PLDM | arXiv:2502.14819 | ⭐⭐⭐ | 无奖励离线数据上用隐动力学模型规划 |
| Balestriero & LeCun 2025, LeJEPA | arXiv:2511.08544 | ⭐⭐⭐ | 去掉启发式技巧的 JEPA 训练目标（SIGReg） |
| Maes et al. 2026, LeWorldModel | arXiv:2603.19312 | ⭐⭐⭐ | 从像素端到端稳定训练 JEPA 世界模型 |
| Oquab et al. 2023, DINOv2 | arXiv:2304.07193 | ⭐⭐ | 本章项目用的冻结特征 |
| Chen & He 2021, SimSiam | arXiv:2011.10566 | ⭐⭐ | stop-gradient 为什么能防坍塌 |
| Zbontar et al. 2021, Barlow Twins | arXiv:2103.03230 | ⭐⭐ | 冗余削减 |
| He et al. 2022, MAE | arXiv:2111.06377 | ⭐⭐ | 对照：在像素空间做掩码重建 |
| Caron et al. 2021, DINO | arXiv:2104.14294 | ⭐ | 自蒸馏 |
| Chen et al. 2020, SimCLR | arXiv:2002.05709 | ⭐ | 对比学习，了解即可 |
| He et al. 2020, MoCo | arXiv:1911.05722 | ⭐ | 对比学习，了解即可 |

## 💻 算力与环境

- 本机可完成：预先把冻结编码器的特征算好存盘，之后只训练小的预测器。
- 需要下载预训练权重（DINOv2）。环境：PushT（独立环境，原因见 Phase 8）。

## ✅ 学习检查

- [ ] 能解释为什么"预测每个像素"对世界模型来说可能是不必要的
- [ ] 能解释表征坍塌为什么是训练目标的一个合法最小值
- [ ] 能说出至少三种防坍塌的办法，以及各自的原理
- [ ] 能画出 I-JEPA 的结构，并说明目标编码器为什么用 EMA 更新
- [ ] 能解释怎么用线性探针检验表征里有没有某个物理量
- [ ] 能说出 DINO-WM 的规划流程：目标怎么给、代价怎么算、动作怎么优化
- [ ] 能对比 JEPA、Dreamer、TD-MPC2 三者"表征靠什么信号学出来"
