# Phase 7: MuZero 与 TD-MPC2——不重构观测的世界模型

> **学习目标**：理解"只预测对决策有用的量"这一路线；实现 MCTS、MuZero 和 TD-MPC2
> **预计时长**：2 周（第 14–15 周）
> **前置**：Phase 1（MPPI）、Phase 4（价值函数、TD）、Phase 5（SAC）、Phase 6（Dreamer）
> **状态**：README ✅ ｜ CONCEPT.md ⏳ ｜ 讲义 0/9 ⏳ ｜ 项目 ⏳

## 🧭 在课程中的位置

到 Phase 6 为止，世界模型都要把观测重建出来（解码器 + 重建损失）。
重建有代价：模型的容量会花在和任务无关的细节上，比如背景纹理。

本章走另一条路：**不重建观测**，隐状态只需要能预测奖励、价值和策略。
这条路线的两个代表：

- **MuZero**：离散动作，用蒙特卡洛树搜索（MCTS）在学到的模型里规划
- **TD-MPC2**：连续控制，用 MPPI 在隐空间里规划，外加一个价值函数补上视野之外的部分

TD-MPC2 是具身智能方向很重要的一个基线，本章的项目会把它加进排行榜。

本章也回答一个贯穿前面几章的问题：有了模型之后，是每步规划（PlaNet、MuZero、TD-MPC2），
还是提前学好策略（Dreamer）？两者可以结合。

"不重建"的思路在 Phase 10 的 JEPA 里会再次出现，只是那里连奖励都不需要。

## 📚 本阶段内容

### 1. 为什么可以不重建

- `01-decoder-free-and-value-equivalence.py` - 重建的代价；价值等价：模型只需在"算价值"这件事上和真实环境一致

### 2. 树搜索与 MuZero

- `02-mcts-from-scratch.py` - 蒙特卡洛树搜索：从 UCB1 到 PUCT（井字棋）
- `03-alphazero-mini.py` - 已知规则下的策略 / 价值网络 + MCTS 自我对弈
- `04-muzero-model.py` - MuZero 的三个网络：表示、动力学、预测
- `05-muzero-agent.py` - K 步展开训练、在学到的模型里搜索
- `06-efficientzero-ideas.py` - EfficientZero 的三项改进：一致性损失、value prefix、离策略修正

### 3. 连续控制：TD-MPC

- `07-tdmpc-latent-model.py` - TD-MPC 的无解码器隐模型与联合训练目标
- `08-tdmpc2.py` - TD-MPC2：SimNorm、Q 集成、离散回归；MPPI + 策略先验 + 终端价值
- `09-planning-vs-policy.py` - 规划和策略怎么选、怎么结合

## 🧠 核心概念

详见 [CONCEPT.md](./CONCEPT.md)《MuZero 与 TD-MPC2：不重构观测的世界模型》：

1. 不重建观测行不行：价值等价
2. 蒙特卡洛树搜索（MCTS）
3. AlphaZero：策略 / 价值网络 + 搜索
4. MuZero 的三个网络
5. MuZero 的训练：K 步展开
6. EfficientZero：让 MuZero 省样本
7. TD-MPC：无解码器隐模型 + MPPI
8. TD-MPC2：稳健与可扩展
9. 规划还是策略：怎么选，怎么结合

## 🎯 实战项目

**TD-MPC2 连续控制**（`projects/phase-07-tdmpc2-control/`）：实现精简版 TD-MPC2，
在状态输入的 MuJoCo 任务上训练，加入排行榜，与 SAC、MBPO、Dreamer 对比。
附带一个小交付物：MuZero 学会井字棋。

## 📖 论文与资料

| 论文 | 编号 | 必读 | 说明 |
| --- | --- | --- | --- |
| Schrittwieser et al. 2020, MuZero | arXiv:1911.08265；Nature 588 | ⭐⭐⭐⭐⭐ | 在学到的模型里做树搜索 |
| Hansen et al. 2024, TD-MPC2 | arXiv:2310.16828 | ⭐⭐⭐⭐⭐ | 具身方向的重要基线 |
| Hansen et al. 2022, TD-MPC | arXiv:2203.04955 | ⭐⭐⭐⭐ | 无解码器隐模型 + MPPI + 终端价值 |
| Silver et al. 2018, AlphaZero | arXiv:1712.01815；Science 362 | ⭐⭐⭐ | MuZero 的前身：规则已知时的搜索 + 学习 |
| Ye et al. 2021, EfficientZero | arXiv:2111.00210 | ⭐⭐⭐ | 让 MuZero 在少量数据下可用 |
| Grimm et al. 2020, Value Equivalence | arXiv:2011.03506 | ⭐⭐ | 价值等价原理的形式化 |
| Wang et al. 2024, EfficientZero V2 | arXiv:2403.00564 | ⭐⭐ | 推广到连续控制，选读 |
| Schrittwieser et al. 2021, MuZero Unplugged | arXiv:2104.06294 | ⭐ | 离线设定，选读 |

## 💻 算力与环境

- 环境：自己写的井字棋、`CartPole-v1`、`Pendulum-v1`、`HalfCheetah-v5`、`Hopper-v5`。
- 本机可完成。TD-MPC2 在 MuJoCo 任务上预计需要数小时（未实测的估算），可选在 4090 上运行。

## ✅ 学习检查

- [ ] 能解释价值等价的意思，以及不重建观测的好处和风险
- [ ] 能手写 MCTS 的四个步骤，并写出 PUCT 的公式
- [ ] 能说出 MuZero 三个网络的输入输出，并写出 K 步展开的损失
- [ ] 能说出 EfficientZero 的三项改进各自针对什么问题
- [ ] 能解释 TD-MPC 规划时为什么需要终端价值和策略先验
- [ ] 能对比 Dreamer 与 TD-MPC2：模型学什么、怎么用模型、各自的优势
