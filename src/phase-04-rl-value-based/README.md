# Phase 4: 强化学习核心（上）——价值函数、TD 与 Dyna

> **学习目标**：掌握回报、价值函数、Bellman 方程、TD 学习和 λ-return；实现 DQN 和 Dyna-Q
> **预计时长**：2 周（第 7–8 周）
> **前置**：Phase 1（MDP 词汇、规划）。本章从零讲强化学习，不需要其他基础
> **状态**：README ✅ ｜ CONCEPT.md ✅ ｜ 讲义 8/8 ✅ ｜ 项目 ✅

## 🧭 在课程中的位置

Phase 1–3 的决策方式都是规划：每走一步，在模型里试上千条动作序列。
这有两个硬伤：算得慢，而且只能看到规划视野以内的事。

强化学习给出的办法是把"规划的结果"存下来：

- **价值函数**记住"从这个状态出发，未来总共能拿多少奖励"，这样就不用每次展开到很远
- **策略**记住"在这个状态该做什么"，这样就不用每次重新搜索

本章讲价值函数这一半，Phase 5 讲策略那一半。两章都只讲后面的世界模型真正用到的内容：

- **λ-return**（第 6 节）是 Dreamer 训练价值函数的目标，Phase 6 会原样用到
- **Dyna**（第 10 节）是"用模型生成经验来训练"的原型，Dreamer 和 MBPO 都是它的后代
- **分布式价值**（第 9 节）是 Dreamer V3 的 two-hot 回归的前身

## 📚 本阶段内容

### 1. 价值函数与 Bellman 方程

- `01-returns-and-value-functions.py` - 回报、折扣、状态价值与动作价值；在网格世界里手算
- `02-dynamic-programming.py` - Bellman 方程、策略评估、值迭代：模型已知时的精确解法

### 2. 从经验中学价值

- `03-monte-carlo-vs-td.py` - 蒙特卡洛与 TD(0)：自举、偏差与方差
- `04-n-step-and-lambda-returns.py` - n 步回报与 λ-return 的递推实现（Dreamer 的价值目标）
- `05-q-learning.py` - 表格 Q-learning、ε-greedy、在策略与离策略

### 3. 深度 Q 网络

- `06-dqn-from-scratch.py` - DQN：经验回放、目标网络
- `07-dqn-improvements-and-distributional.py` - 过估计与 Double DQN；分布式价值（C51）

### 4. 模型与价值结合

- `08-dyna-q.py` - Dyna-Q：真实经验 + 模型生成的经验一起训练

## 🧠 核心概念

详见 [CONCEPT.md](./CONCEPT.md)《强化学习核心（上）：价值函数、TD 与 Dyna》：

1. 为什么现在要学强化学习
2. 回报、折扣与价值函数
3. Bellman 方程
4. 动态规划：模型已知时的精确解法
5. 蒙特卡洛与时序差分（TD）
6. n 步回报与 λ-return
7. Q-learning 与离策略学习
8. DQN：把 Q-learning 接上神经网络
9. DQN 的改进与分布式价值
10. Dyna：用模型生成经验

## 🎯 实战项目

**Dyna-DQN**（`projects/phase-04-dyna-dqn/`）：给 DQN 加上一个学出来的动力学模型，
用模型生成的转移扩充回放数据。在 `CartPole-v1` 上对比 DQN 与 Dyna-DQN 的样本效率，
并考察"每步真实交互配多少步模型数据"的影响。

## 📖 论文与资料

| 论文 / 书 | 编号 | 必读 | 说明 |
| --- | --- | --- | --- |
| Sutton & Barto, Reinforcement Learning: An Introduction（第 2 版） | 第 3–8 章 | ⭐⭐⭐⭐⭐ | 第 3、4、6 章精读，第 7、8 章读主要小节 |
| Mnih et al. 2015, DQN | Nature 518；预印本 arXiv:1312.5602 | ⭐⭐⭐⭐ | 经验回放与目标网络 |
| van Hasselt et al. 2016, Double DQN | arXiv:1509.06461 | ⭐⭐⭐ | 过估计从哪来、怎么修 |
| Bellemare et al. 2017, C51 | arXiv:1707.06887 | ⭐⭐⭐ | 把价值预测成分布，Dreamer V3 two-hot 的前身 |
| Sutton 1991, Dyna | ACM SIGART Bulletin 2(4) | ⭐⭐⭐ | "学习、规划、反应"一体的原始想法，很短 |
| Hessel et al. 2018, Rainbow | arXiv:1710.02298 | ⭐⭐ | DQN 各项改进的汇总，浏览即可 |

## 💻 算力与环境

- 环境：自己写的网格世界、`CliffWalking-v1`、`CartPole-v1`。全部在本机 CPU 上运行。
- 本机（CPU 单线程）实测耗时：讲义 01–05 与 08 各约 1–10 秒；讲义 06（DQN + 消融，3 种子）约 2.5 分钟；讲义 07（DQN/Double DQN/C51，3 种子）约 3.4 分钟；实战项目 Dyna-DQN（3 组 $n_p \times 3$ 种子）约 4.4 分钟。

## ✅ 学习检查

- [ ] 能写出 $V^\pi$ 和 $Q^\pi$ 的 Bellman 方程，并在一个小例子上手算
- [ ] 能解释蒙特卡洛和 TD 各自的偏差与方差从哪来
- [ ] 能写出 λ-return 的递推式，并说明 λ = 0 和 λ = 1 分别退化成什么
- [ ] 能解释 DQN 的经验回放和目标网络各自解决什么问题
- [ ] 能解释 Q-learning 为什么会高估，Double DQN 怎么修
- [ ] 能说出 Dyna 的三个组成部分，以及它和 Phase 1 的 MPC 在"怎么用模型"上的区别
