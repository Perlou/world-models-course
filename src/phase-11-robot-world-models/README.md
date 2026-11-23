# Phase 11: 机器人世界模型专题

> **学习目标**：了解世界模型在机器人上的四种用法——当策略、当仿真器、当数据引擎、在真机上在线学习；能读懂这个方向的最新论文
> **预计时长**：2 周（第 23–24 周）
> **前置**：Phase 6（Dreamer）、Phase 9（扩散与视频世界模型）、Phase 10（JEPA）
> **状态**：README ✅ ｜ CONCEPT.md ✅ ｜ 讲义 0/9 ⏳ ｜ 项目 ⏳

## 🧭 在课程中的位置

前十章讲的是通用的世界模型方法。本章回到你选的方向——具身智能，
看这些方法在机器人上到底怎么用。

机器人学习和前面的游戏、仿真控制有几处实质的不同：

- **数据**：主要来自人类遥操作的演示，没有奖励，量很少而且贵
- **策略**：主流是模仿学习（行为克隆、Diffusion Policy、VLA），不是强化学习
- **评估**：每次评估都要在真机上跑，慢、贵、难复现

世界模型在这三个痛点上各有用武之地，本章按用法组织：

1. **当策略**：预测未来画面本身就是在做计划（视频预测策略、World-Action Model）
2. **当仿真器**：在世界模型里评估策略，少跑真机
3. **当数据引擎**：用世界模型生成训练数据
4. **在真机上在线学习**：Dreamer 式的方法直接在机器人上跑

这个方向 2025–2026 年论文很多，变化很快。本章的目标是建立一张能把新论文放进去的地图，
不追求覆盖每一篇。

## 📚 本阶段内容

### 1. 机器人学习基础

- `01-robot-learning-landscape.py` - 机器人学习的全景：数据从哪来、策略怎么学、怎么评估
- `02-lerobot-data-and-action-chunks.py` - LeRobot 数据格式、动作分块（action chunking）
- `03-behavior-cloning-and-diffusion-policy.py` - 行为克隆、分布偏移、Diffusion Policy

### 2. 世界模型当策略

- `04-video-prediction-as-policy.py` - 先预测未来画面，再用逆动力学解出动作
- `05-world-action-models.py` - World-Action Model：把视频预测和动作生成放进同一个模型

### 3. 世界模型当仿真器和数据引擎

- `06-world-model-as-simulator.py` - 在世界模型里评估策略；与真实成功率的相关性
- `07-world-model-as-data-engine.py` - 用世界模型生成训练数据

### 4. 真机与其他方向

- `08-real-robot-world-models.py` - 真机上的世界模型：DayDreamer、V-JEPA 2-AC、仿真到真实的差距
- `09-other-frontiers.py` - 自动驾驶与交互式视频方向速览

## 🧠 核心概念

详见 [CONCEPT.md](./CONCEPT.md)《机器人世界模型专题》：

1. 机器人学习的全景
2. 机器人数据与动作分块
3. 行为克隆与 Diffusion Policy
4. 把视频预测当策略
5. World-Action Model
6. 把世界模型当仿真器：策略评估
7. 把世界模型当数据引擎
8. 真机上的世界模型
9. 其他前沿：自动驾驶与交互式视频

## 🎯 实战项目

**用世界模型评估策略**（`projects/phase-11-wm-policy-evaluation/`）：在 PushT 上训练几个水平不同的策略
（行为克隆、Diffusion Policy，不同训练程度），分别在真实仿真器和学到的世界模型里评估，
检验"世界模型里的成功率"与"真实成功率"的排序是否一致。

## 📖 论文与资料

| 论文 | 编号 | 必读 | 说明 |
| --- | --- | --- | --- |
| Chi et al. 2023, Diffusion Policy | arXiv:2303.04137 | ⭐⭐⭐⭐⭐ | 用扩散模型生成动作序列 |
| Zhao et al. 2023, ACT | arXiv:2304.13705 | ⭐⭐⭐⭐ | 动作分块 + 低成本双臂硬件（ALOHA） |
| Du et al. 2023, UniPi | arXiv:2302.00111 | ⭐⭐⭐⭐ | 文本条件的视频生成当通用策略 |
| Wu et al. 2023, GR-1 | arXiv:2312.13139 | ⭐⭐⭐ | 视频生成预训练用于机器人操作 |
| Cheang et al. 2024, GR-2 | arXiv:2410.06158 | ⭐⭐⭐ | GR-1 的扩展 |
| Hu et al. 2024, Video Prediction Policy | arXiv:2412.14803 | ⭐⭐⭐ | 用视频预测模型的表征做策略 |
| Li et al. 2025, Unified Video Action Model | arXiv:2503.00200 | ⭐⭐⭐⭐ | 视频与动作的统一模型 |
| Zhu et al. 2025, Unified World Models | arXiv:2504.02792 | ⭐⭐⭐⭐ | 视频扩散与动作扩散耦合 |
| Jang et al. 2025, DreamGen | arXiv:2505.12705 | ⭐⭐⭐⭐ | 视频世界模型当数据引擎 |
| Quevedo et al. 2025, WorldGym | arXiv:2506.00613 | ⭐⭐⭐⭐ | 世界模型当策略评估环境 |
| Li et al. 2025, WorldEval | arXiv:2505.19017 | ⭐⭐⭐ | 世界模型评估真实机器人策略 |
| Zhu et al. 2024, IRASim | arXiv:2406.14540 | ⭐⭐⭐ | 面向操作的细粒度世界模型 |
| Wu et al. 2022, DayDreamer | arXiv:2206.14176 | ⭐⭐⭐⭐ | Dreamer 在真机上在线学习 |
| Assran et al. 2025, V-JEPA 2 | arXiv:2506.09985 | ⭐⭐⭐ | 重点看动作条件部分（V-JEPA 2-AC）的真机实验 |
| Shen et al. 2026, World Action Models: A Survey | arXiv:2606.20781 | ⭐⭐⭐⭐ | 综述，用来建立地图 |
| Wang et al. 2026, World Models for Robotic Manipulation: A Survey | arXiv:2606.00113 | ⭐⭐⭐⭐ | 综述 |
| Lu et al. 2026, World-Action Models for Robot Learning and Control: A Survey | arXiv:2609.16074 | ⭐⭐⭐ | 综述 |
| Mei et al. 2026, Robotic Video World Models: A Survey | arXiv:2601.07823 | ⭐⭐⭐ | 综述 |
| Liao et al. 2025, Genie Envisioner | arXiv:2508.05635 | ⭐⭐ | 面向操作的世界基础平台 |
| Brohan et al. 2022 / 2023, RT-1 / RT-2 | arXiv:2212.06817 / 2307.15818 | ⭐⭐ | VLA 路线的背景 |
| Kim et al. 2024, OpenVLA | arXiv:2406.09246 | ⭐⭐ | 开源 VLA |
| Black et al. 2024, π0 | arXiv:2410.24164 | ⭐⭐ | flow matching 的 VLA |
| NVIDIA 2025, GR00T N1 | arXiv:2503.14734 | ⭐⭐ | 人形机器人基础模型 |
| Octo Model Team 2024, Octo | arXiv:2405.12213 | ⭐⭐ | 开源通用机器人策略 |
| Open X-Embodiment 2023 | arXiv:2310.08864 | ⭐⭐ | 跨本体数据集 |
| Ross et al. 2011, DAgger | arXiv:1011.0686 | ⭐⭐ | 模仿学习的分布偏移问题 |
| Bar et al. 2024, Navigation World Models | arXiv:2412.03572 | ⭐ | 导航方向 |
| Li et al. 2024, SIMPLER | arXiv:2405.05941 | ⭐⭐⭐ | 用物理仿真器评估真机策略；MMRV 指标的出处 |
| Hu et al. 2023, GAIA-1；Russell et al. 2025, GAIA-2 | arXiv:2309.17080 / 2503.20523 | ⭐ | 自动驾驶方向 |
| Gao et al. 2024, Vista | arXiv:2405.17398 | ⭐ | 自动驾驶方向 |

写本章讲义和概念文档之前，先用综述刷新一遍论文清单：这个方向每个月都有几十篇新论文。

## 💻 算力与环境

- 数据与工具：LeRobot（`lerobot` 要求 Python ≥ 3.12，与本课一致）、PushT。
  可选：ALOHA 仿真、Meta-World、LIBERO（都依赖 MuJoCo 3.x）。
- 训练 Diffusion Policy 和世界模型：本机跑缩小版，完整规模建议用 4090。

## ✅ 学习检查

- [ ] 能说出机器人学习和游戏 / 仿真控制的三处主要不同
- [ ] 能解释动作分块解决什么问题
- [ ] 能解释行为克隆的分布偏移，以及它和世界模型复合误差的相似之处
- [ ] 能说出"视频预测当策略"的两步流程，以及逆动力学模型的作用
- [ ] 能说出 World-Action Model 相比"先视频后动作"的两阶段方法改了什么
- [ ] 能说出用世界模型评估策略时，应该看什么指标来判断评估可不可信
- [ ] 拿到一篇新的机器人世界模型论文，能说出它属于四种用法中的哪一种
