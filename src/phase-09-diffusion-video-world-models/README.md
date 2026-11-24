# Phase 9: 扩散与交互式视频世界模型

> **学习目标**：用扩散模型做世界模型；理解自回归漂移与 Diffusion Forcing；实现隐动作模型和精简版 Genie；了解 Dreamer 4 与视频基础模型
> **预计时长**：3 周（第 18–20 周）
> **前置**：Phase 6（Dreamer）、Phase 8（tokenizer、MaskGIT）；`deep-learning-course` 的 DDPM
> **状态**：README ✅ ｜ CONCEPT.md ✅ ｜ 讲义 0/10 ⏳ ｜ 项目 ⏳

## 🧭 在课程中的位置

Phase 8 把画面离散成 token 再预测。离散化会丢细节，而有些细节恰恰决定了任务成败
（DIAMOND 论文的标题就是 Visual Details Matter）。本章换成扩散模型直接生成连续的画面。

本章分三段：

1. **扩散世界模型**（第 1–5 节）：从旧课的 DDPM 过渡到 EDM 和 flow matching，
   做出以动作为条件的下一帧扩散，学习 DIAMOND，并处理自回归生成中误差越滚越大的问题
2. **从无标注视频里学可控的世界**（第 6–8 节）：视频 tokenizer、隐动作模型、Genie。
   不需要动作标签，只看视频就学出"可以操控"的世界模型
3. **走向实时和大规模**（第 9–10 节）：少步采样与蒸馏；Dreamer 4；视频基础模型

对机器人方向，本章是 Phase 11 的技术基础：那里的视频预测策略、神经仿真器、
数据引擎，用的都是本章的扩散 / 视频世界模型。

## 📚 本阶段内容

### 1. 扩散世界模型

- `01-ddpm-to-edm.py` - 从 DDPM 到 EDM：σ 参数化、预条件、少步采样器
- `02-flow-matching.py` - flow matching 与 rectified flow
- `03-action-conditioned-frame-diffusion.py` - 以过去帧和动作为条件，扩散生成下一帧
- `04-diamond.py` - DIAMOND：扩散世界模型 + 在扩散的梦里训练智能体
- `05-autoregressive-drift-and-diffusion-forcing.py` - 自回归漂移、噪声增强、Diffusion Forcing

### 2. 从视频里学可控的世界

- `06-video-tokenizers-and-latent-diffusion.py` - 视频 tokenizer、时空压缩、在隐空间里做扩散
- `07-latent-action-models.py` - 隐动作模型：从无标注视频里学出离散的"动作"
- `08-mini-genie.py` - 精简版 Genie：视频 tokenizer + 隐动作模型 + 动力学模型

### 3. 实时与大规模

- `09-realtime-few-step-inference.py` - 少步采样、蒸馏、因果化：让视频模型能实时交互
- `10-dreamer4-and-video-foundation-models.py` - Dreamer 4；Cosmos、UniSim 等视频基础模型

## 🧠 核心概念

详见 [CONCEPT.md](./CONCEPT.md)《扩散与交互式视频世界模型》：

1. 从 DDPM 到 EDM
2. Flow matching
3. 以动作为条件的下一帧扩散
4. DIAMOND：扩散世界模型
5. 自回归漂移与 Diffusion Forcing
6. 视频 tokenizer 与隐空间扩散
7. 隐动作模型：从无标注视频里学动作
8. Genie：生成式交互环境
9. 实时推理：少步采样与蒸馏
10. Dreamer 4 与视频基础模型

## 🎯 实战项目

**可玩的神经仿真器**（`projects/phase-09-playable-neural-simulator/`）：训练一个以动作为条件的扩散世界模型
（PushT 或 `CarRacing-v3` 的画面），做成可以用键盘实时操控的"神经仿真器"，
报告预测质量随步数的变化，以及每帧的推理延迟。可选：用浏览器做前端。

## 📖 论文与资料

| 论文 | 编号 | 必读 | 说明 |
| --- | --- | --- | --- |
| Alonso et al. 2024, DIAMOND | arXiv:2405.12399 | ⭐⭐⭐⭐⭐ | 扩散世界模型 + 在其中训练智能体 |
| Bruce et al. 2024, Genie | arXiv:2402.15391 | ⭐⭐⭐⭐⭐ | 从无标注视频学可交互环境 |
| Karras et al. 2022, EDM | arXiv:2206.00364 | ⭐⭐⭐⭐ | 扩散模型设计空间的梳理，DIAMOND 用它 |
| Chen et al. 2024, Diffusion Forcing | arXiv:2407.01392 | ⭐⭐⭐⭐ | 每帧独立噪声水平 |
| Hafner et al. 2025, Dreamer 4 | arXiv:2509.24527 | ⭐⭐⭐⭐ | 在可扩展的世界模型里训练智能体 |
| Lipman et al. 2023, Flow Matching | arXiv:2210.02747 | ⭐⭐⭐ | 另见 Rectified Flow，arXiv:2209.03003 |
| Valevski et al. 2024, GameNGen | arXiv:2408.14837 | ⭐⭐⭐ | 扩散模型做实时游戏引擎 |
| Yin et al. 2024, CausVid | arXiv:2412.07772 | ⭐⭐⭐ | 把双向视频扩散蒸馏成快速自回归模型 |
| Huang et al. 2025, Self Forcing | arXiv:2506.08009 | ⭐⭐⭐ | 弥合自回归视频扩散的训练—推理差距 |
| Yang et al. 2023, UniSim | arXiv:2310.06114 | ⭐⭐⭐ | 学一个通用的真实世界交互仿真器 |
| NVIDIA 2025, Cosmos | arXiv:2501.03575 | ⭐⭐⭐ | 面向物理 AI 的世界基础模型平台 |
| Baker et al. 2022, VPT | arXiv:2206.11795 | ⭐⭐ | 另一条路线：用逆动力学模型给视频打动作标签 |
| Ye et al. 2024, LAPA | arXiv:2410.11758 | ⭐⭐ | 隐动作预训练用于机器人策略 |
| Frans et al. 2024, Shortcut Models | arXiv:2410.12557 | ⭐⭐ | 一步 / 少步生成，Dreamer 4 的相关技术 |
| Song et al. 2025, History-Guided Video Diffusion | arXiv:2502.06764 | ⭐⭐ | Diffusion Forcing 的后续 |
| Kanervisto et al. 2025, WHAM | Nature 638 | ⭐⭐ | 游戏画面与动作的联合序列模型 |
| Ho et al. 2020, DDPM | arXiv:2006.11239 | ⭐⭐ | 旧课学过，按需回看 |
| Song et al. 2021, Score SDE | arXiv:2011.13456 | ⭐⭐ | 扩散的连续时间视角 |
| Rombach et al. 2022, Latent Diffusion | arXiv:2112.10752 | ⭐⭐ | 在隐空间里做扩散 |
| Peebles & Xie 2023, DiT | arXiv:2212.09748 | ⭐⭐ | 用 Transformer 做扩散骨干 |
| Ho et al. 2022, Video Diffusion Models | arXiv:2204.03458 | ⭐ | 视频扩散的早期工作 |

Genie 2、Genie 3 目前只有官方博客，没有论文，细节以官方说明为准。

## 💻 算力与环境

- 本机（MPS）只适合低分辨率的缩小版。完整项目建议用 4090（训练时间未实测，写讲义时实测后填入）。
- 环境：PushT（已在主环境中，依赖冲突的处理见 Phase 8）或 `CarRacing-v3`。

## ✅ 学习检查

- [ ] 能说出 EDM 相比 DDPM 改了哪几件事，预条件的作用是什么
- [ ] 能写出 flow matching 的训练目标，并解释它和扩散的关系
- [ ] 能解释自回归漂移的成因，以及噪声增强和 Diffusion Forcing 各自怎么缓解
- [ ] 能说出 DIAMOND 的世界模型由哪几部分组成，奖励和终止怎么预测
- [ ] 能解释隐动作模型凭什么不用标签就能学出动作，瓶颈为什么必须很窄
- [ ] 能画出 Genie 的三个组件及其训练顺序
- [ ] 能说出让视频扩散模型实时运行的几类办法
