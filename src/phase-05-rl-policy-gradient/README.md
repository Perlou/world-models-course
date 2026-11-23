# Phase 5: 强化学习核心（下）——策略梯度、SAC 与梯度估计器

> **学习目标**：掌握策略梯度、Actor-Critic、PPO、SAC；弄清三种梯度估计器；实现 MBPO 并理解"沿模型反向传播"
> **预计时长**：2 周（第 9–10 周）
> **前置**：Phase 4（价值函数、TD、λ-return、经验回放）
> **状态**：README ✅ ｜ CONCEPT.md ⏳ ｜ 讲义 0/9 ⏳ ｜ 项目 ⏳

## 🧭 在课程中的位置

Phase 4 学了价值函数，它记住"这个状态值多少"。本章学策略，它记住"这个状态该做什么"。
有了策略，决策时只需要一次前向计算，不用再像 MPC 那样每步搜索上千条动作序列。

本章有三样东西是 Phase 6 的 Dreamer 直接要用的：

- **三种梯度估计器**（第 7 节）：score-function、重参数化、straight-through。
  Dreamer 对连续动作用重参数化、对离散动作用 score-function、对类别隐变量用 straight-through
- **沿模型反向传播**（第 11 节）：把可微的世界模型当成计算图的一部分，让回报的梯度直接流回策略。
  这就是 Dreamer V1 训练 actor 的办法
- **Actor-Critic 与熵正则**（第 4、9 节）：Dreamer 在想象里跑的就是 actor-critic

另外两样是后面对比用的基线：PPO（无模型方法的代表）和 SAC（MBPO、TD-MPC 的基础）。
本章的项目建立一张连续控制排行榜，Phase 6、7 的 Dreamer、TD-MPC2 会继续往上加。

## 📚 本阶段内容

### 1. 策略梯度

- `01-policy-gradient-reinforce.py` - 策略梯度定理与 REINFORCE
- `02-baselines-and-advantage.py` - 基线、优势函数，用实验看方差降了多少
- `03-actor-critic.py` - Actor-Critic 与熵正则
- `04-gae-and-ppo.py` - GAE 与 PPO-clip

### 2. 梯度估计器

- `05-gradient-estimators.py` - score-function、重参数化、straight-through 三种估计器的方差对比实验

### 3. 连续控制

- `06-ddpg-td3.py` - 确定性策略梯度：DDPG 与 TD3
- `07-sac-from-scratch.py` - SAC：tanh 高斯策略、双 Q、自动调温度

### 4. 模型与策略结合

- `08-mbpo.py` - MBPO：用模型的短展开生成数据喂给 SAC
- `09-backprop-through-model.py` - 沿可微模型反向传播回报的梯度（SVG 思想，Dreamer 的前奏）

## 🧠 核心概念

详见 [CONCEPT.md](./CONCEPT.md)《强化学习核心（下）：策略梯度、SAC 与梯度估计器》：

1. 直接优化策略：策略梯度定理
2. REINFORCE 与方差问题
3. 基线与优势函数
4. Actor-Critic
5. 广义优势估计（GAE）
6. PPO
7. 三种梯度估计器
8. 确定性策略梯度：DDPG 与 TD3
9. SAC：最大熵强化学习
10. MBPO：把模型的短展开喂给 SAC
11. 沿模型反向传播：从 SVG 到 Dreamer

## 🎯 实战项目

**连续控制排行榜**（`projects/phase-05-control-benchmark/`）：在同一套评估协议下（相同任务、相同交互步数、多个随机种子）
对比 PETS（Phase 1）、PPO、SAC、MBPO，输出"回报—环境步数"曲线和汇总表。
这张排行榜在 Phase 6、7 会继续加入 Dreamer 和 TD-MPC2。

## 📖 论文与资料

| 论文 / 书 | 编号 | 必读 | 说明 |
| --- | --- | --- | --- |
| Sutton & Barto（第 2 版） | 第 13 章 | ⭐⭐⭐⭐⭐ | 策略梯度定理、REINFORCE、Actor-Critic |
| Schulman et al. 2016, GAE | arXiv:1506.02438 | ⭐⭐⭐⭐ | 优势估计的偏差—方差权衡 |
| Schulman et al. 2017, PPO | arXiv:1707.06347 | ⭐⭐⭐⭐ | 本课的无模型基线 |
| Haarnoja et al. 2018, SAC | arXiv:1801.01290 | ⭐⭐⭐⭐ | 最大熵 RL；自动调温度见 arXiv:1812.05905 |
| Janner et al. 2019, MBPO | arXiv:1906.08253 | ⭐⭐⭐⭐⭐ | 何时该信模型：短展开的理由 |
| Heess et al. 2015, SVG | arXiv:1510.09142 | ⭐⭐⭐ | 沿学到的模型反传价值梯度 |
| Mohamed et al. 2020, Monte Carlo Gradient Estimation | arXiv:1906.10652 | ⭐⭐⭐ | 只读 score-function 与 pathwise 两节 |
| Lillicrap et al. 2016, DDPG | arXiv:1509.02971 | ⭐⭐ | 确定性策略梯度 |
| Fujimoto et al. 2018, TD3 | arXiv:1802.09477 | ⭐⭐ | 双 Q 截断、延迟更新 |
| Bengio et al. 2013, Straight-Through | arXiv:1308.3432 | ⭐⭐ | straight-through 估计器的出处 |
| Jang et al. 2017, Gumbel-Softmax | arXiv:1611.01144 | ⭐⭐ | 离散变量的另一种可微近似，了解即可 |

## 💻 算力与环境

- 环境：`CartPole-v1`、`Pendulum-v1`、`HalfCheetah-v5`、`Hopper-v5`。
- 本机可完成。SAC、MBPO 在 MuJoCo 任务上预计各需数小时（未实测的估算）。

## ✅ 学习检查

- [ ] 能推导策略梯度定理，并解释 $\nabla \log \pi$ 这一项从哪来
- [ ] 能解释减去基线为什么不改变梯度的期望、却能降低方差
- [ ] 能写出 GAE 的递推式，并说明它和 λ-return 的关系
- [ ] 能说出三种梯度估计器各自的适用条件、偏差和方差特点
- [ ] 能解释 SAC 的熵项起什么作用，tanh 高斯策略的对数概率怎么修正
- [ ] 能解释 MBPO 为什么只做很短的模型展开
- [ ] 能说出"沿模型反向传播"需要模型和策略满足什么条件
