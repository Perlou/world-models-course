# Phase 3: RSSM 与 PlaNet——在隐空间里规划

> **学习目标**：推导并实现循环状态空间模型（RSSM），在隐空间里用 CEM 规划，从像素完成连续控制
> **预计时长**：2 周（第 5–6 周）
> **前置**：Phase 1（CEM、MPC）、Phase 2（VAE + RNN 世界模型）
> **状态**：README ✅ ｜ CONCEPT.md ⏳ ｜ 讲义 0/8 ⏳ ｜ 项目 ⏳

## 🧭 在课程中的位置

Phase 2 的 V 和 M 是分开训练的：VAE 不知道哪些信息对预测未来有用，RNN 只能接受 VAE 给它的东西。
本章把两者合成一个端到端训练的隐状态空间模型，这就是 RSSM。

RSSM 是全课最重要的一个结构：Phase 6 的 Dreamer V1 / V2 / V3 用的世界模型都是它，
只是换了隐变量的类型和几个训练技巧。本章要把它的推导（序列 ELBO）和实现都吃透。

决策部分仍然不需要强化学习：直接把 Phase 1 的 CEM 规划器搬到隐空间里用，这就是 PlaNet。
它的短板（每步都要重新规划、看不远）正是 Phase 4–6 要解决的问题。

## 📚 本阶段内容

### 1. 序列隐变量模型的理论

- `01-filtering-and-belief-state.py` - 部分可观测、信念状态；用一维卡尔曼滤波演示"预测—修正"
- `02-sequential-vae-elbo.py` - 把 VAE 的 ELBO 推广到序列：学出来的先验、滤波后验
- `03-deterministic-vs-stochastic.py` - 纯确定性、纯随机、两者结合，三种状态转移的对比实验

### 2. RSSM

- `04-rssm-from-scratch.py` - 从零实现 RSSM：`observe()`（有观测时）与 `imagine()`（无观测时）
- `05-world-model-training.py` - 序列回放、重建 + 奖励 + KL 三项损失、free nats
- `06-open-loop-prediction-and-overshooting.py` - 开环预测视频、多步 KL、latent overshooting

### 3. PlaNet

- `07-latent-cem-planning.py` - 在隐空间里做 CEM 规划
- `08-planet-agent.py` - 完整闭环：采集数据、训练模型、规划、再采集

## 🧠 核心概念

详见 [CONCEPT.md](./CONCEPT.md)《RSSM 与 PlaNet：在隐空间里规划》：

1. 部分可观测与信念状态
2. 隐状态空间模型
3. 序列 ELBO：把 VAE 推广到时间序列
4. 两个高斯之间的 KL 散度与 free nats
5. 确定性路径还是随机路径
6. RSSM：结构与数据流
7. 训练世界模型
8. 开环预测与 latent overshooting
9. 在隐空间里用 CEM 规划
10. PlaNet 智能体闭环

## 🎯 实战项目

**PlaNet 像素控制**（`projects/phase-03-planet-pixel-control/`）：只用 64×64 的渲染画面，
让 PlaNet 学会把摆甩起并保持（像素版 `Pendulum-v1`），输出开环预测视频和学习曲线。
可选：换成 DeepMind Control Suite 的任务。

## 📖 论文与资料

| 论文 | 编号 | 必读 | 说明 |
| --- | --- | --- | --- |
| Hafner et al. 2019, PlaNet | arXiv:1811.04551 | ⭐⭐⭐⭐⭐ | 本章主线：RSSM + 隐空间 CEM 规划 |
| Chung et al. 2015, VRNN | arXiv:1506.02216 | ⭐⭐⭐ | 序列隐变量模型的早期代表，看模型结构和 ELBO |
| Lee et al. 2019, SLAC | arXiv:1907.00953 | ⭐⭐ | 另一条路线：纯随机隐变量 + 无模型 RL |
| Ha & Schmidhuber 2018, World Models | arXiv:1803.10122 | ⭐⭐ | 对照阅读：分开训练 vs 端到端训练 |

## 💻 算力与环境

- 环境：像素版 `Pendulum-v1`（用环境自带的渲染）。可选 DeepMind Control Suite：
  `dm-control` 新版要求 MuJoCo 4.x，与本课锁定的 MuJoCo 3.x 冲突，需要固定到兼容的旧版本（写本章讲义时验证）。
- 本机可完成，预计训练需要数小时，CEM 规划是主要耗时（未实测的估算）。可选在 4090 上运行。

## ✅ 学习检查

- [ ] 能解释为什么从像素控制是部分可观测问题，信念状态是什么
- [ ] 能从单步 VAE 的 ELBO 出发，推出序列 ELBO，并说出每一项的含义
- [ ] 能写出两个对角高斯之间 KL 散度的闭式解
- [ ] 能解释 RSSM 为什么同时需要确定性路径和随机路径
- [ ] 能画出 RSSM 在"有观测"和"无观测"两种情况下的数据流，并标出先验和后验
- [ ] 能说出 PlaNet 相比 Phase 2 的 World Models 改进了什么，自己还剩什么短板
