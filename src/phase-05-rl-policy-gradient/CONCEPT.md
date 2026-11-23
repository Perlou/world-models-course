# 强化学习核心（下）：策略梯度、SAC 与梯度估计器

> Phase 5 · 概念精讲 ｜ 对应讲义：`src/phase-05-rl-policy-gradient/` ｜ 预计阅读：6–8 小时
>
> Phase 4 学了价值函数，它记住"这个状态值多少"。这一章学另一半：策略，它记住"这个状态该做什么"。
> 全章围绕一个问题展开：期望回报对策略参数的梯度怎么估计。三种估计办法各有适用条件，
> Phase 6 的 Dreamer 三种都要用。

---

## 目录

- [0. 本章导读](#0-本章导读)
- [1. 直接优化策略：策略梯度定理](#1-直接优化策略策略梯度定理)
- [2. REINFORCE 与方差问题](#2-reinforce-与方差问题)
- [3. 基线与优势函数](#3-基线与优势函数)
- [4. Actor-Critic](#4-actor-critic)
- [5. 广义优势估计（GAE）](#5-广义优势估计gae)
- [6. PPO](#6-ppo)
- [7. 三种梯度估计器](#7-三种梯度估计器)
- [8. 确定性策略梯度：DDPG 与 TD3](#8-确定性策略梯度ddpg-与-td3)
- [9. SAC：最大熵强化学习](#9-sac最大熵强化学习)
- [10. MBPO：把模型的短展开喂给 SAC](#10-mbpo把模型的短展开喂给-sac)
- [11. 沿模型反向传播：从 SVG 到 Dreamer](#11-沿模型反向传播从-svg-到-dreamer)
- [12. 方法对比与选型](#12-方法对比与选型)
- [13. 论文精读卡片](#13-论文精读卡片)
- [14. 实现陷阱与调试清单](#14-实现陷阱与调试清单)
- [15. 常见问题（FAQ）](#15-常见问题faq)
- [16. 自测题](#16-自测题)
- [17. 小结与下章预告](#17-小结与下章预告)
- [附录 A：术语表（中英对照）](#附录-a术语表中英对照)
- [附录 B：延伸阅读](#附录-b延伸阅读)

---

## 0. 本章导读

### 0.1 本章要回答的问题

1. 策略没有"正确动作"的标签，凭什么能用梯度下降训练？$\nabla \log \pi$ 这一项是从哪冒出来的？
2. 策略梯度的方差为什么大得出名？基线、优势函数、GAE 各自砍掉了哪一部分方差，付出了什么代价？
3. 采样这一步不可微。有哪几种办法让梯度"越过"采样？它们各自要求什么、偏差和方差如何？
4. 连续控制里最常用的无模型算法 SAC 是怎么拼出来的？最大熵、重参数化、双 Q、自动调温度各管什么？
5. 手上有一个学到的世界模型时，怎么用它帮策略学习：拿它生成数据（MBPO），还是直接沿着它反向传播（Dreamer 的做法）？

### 0.2 一图总览

全章可以看成对同一个问题的三种回答，再加上"数据从哪来、梯度穿过什么"两个选择。

```
                     目标：让期望回报 J(φ) = E[G_0] 变大
                                     │
                  难点：动作是采样出来的，采样这一步挡住了反向传播
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
  score-function                  pathwise                 straight-through
  只要求 log π 可微               要求被求期望的函数可微         离散变量专用：前向用样本，
  无偏，方差大                    无偏，方差通常小               反向假装可微；有偏
        │                            │                            │
        ▼                            ▼                            ▼
  第 1–6 节                       第 8–11 节                   第 7 节讲原理；
  REINFORCE                       DDPG / TD3                   Phase 6 的 Dreamer V2 / V3
    → 基线与优势                     → SAC（最大熵）              把它用在类别隐变量上
    → Actor-Critic                  → MBPO（数据来自模型的短展开）
    → GAE → PPO                     → 沿模型反向传播
  （在策略，无模型）                  （离策略；后两个用到世界模型）

  第 7 节是全章的枢纽：把前六节用的办法和后四节用的办法放在同一个框架里比较。
  第 12 节：把 PETS、PPO、SAC、MBPO 放进同一张排行榜。
```

### 0.3 前置知识检查

| 需要的知识 | 在哪学过 | 本章哪里用 |
| --- | --- | --- |
| 回报 $G_t$、$V^\pi$、$Q^\pi$、Bellman 方程 | [Phase 4](../phase-04-rl-value-based/CONCEPT.md) 的价值函数与 Bellman 方程 | 第 1、3、4、9 节 |
| TD 误差、自举（bootstrapping）、偏差与方差 | Phase 4 的蒙特卡洛与 TD | 第 4、5 节 |
| n 步回报与 λ-return | Phase 4 的 λ-return | 第 5 节：GAE 就是减掉 $V$ 的 λ-return |
| 经验回放、目标网络、过估计与 Double DQN | Phase 4 的 DQN 及其改进 | 第 8、9 节 |
| Dyna：真实经验加模型生成的经验 | Phase 4 的 Dyna | 第 10 节 |
| 复合误差的界 $e_t \le \varepsilon (L^t - 1)/(L - 1)$ | [Phase 1](../phase-01-dynamics-and-planning/CONCEPT.md) 第 5 节 | 第 10、11 节 |
| 概率集成、模型被钻空子、PETS | Phase 1 第 8、9 节 | 第 8、10 节 |
| MPPI 指数权重的推导 | Phase 1 第 7.5 节 | 第 9 节：最优的最大熵策略是同一个形式 |
| 重参数化技巧 | `deep-learning-course` 的 VAE；本课 Phase 2、3 | 第 7、9、11 节 |
| softmax、交叉熵、高斯对数似然 | `deep-learning-course`；Phase 1 第 8.3 节 | 第 1 节 |
| BPTT 与梯度爆炸 | `deep-learning-course` 的 RNN | 第 11 节 |

如果 λ-return 的递推式已经记不清，先回看 Phase 4 那一节再读本章第 5 节。

### 0.4 概念与讲义对照表

讲义还没写。下表里的函数名是规划中的名字，以最终的讲义为准。

| 概念文档 | 对应讲义 | 讲义里做什么 | 规划中的核心函数 |
| --- | --- | --- | --- |
| 第 1 节 策略梯度定理；第 2 节 REINFORCE | `01-policy-gradient-reinforce.py` | 在两臂老虎机上数值验证策略梯度；在 `CartPole-v1` 上实现 REINFORCE | `check_bandit_gradient()`、`compute_returns_to_go()`、`reinforce_update()` |
| 第 3 节 基线与优势函数 | `02-baselines-and-advantage.py` | 加上学出来的基线，实测梯度方差降了多少 | `ValueNet`、`measure_gradient_variance()` |
| 第 4 节 Actor-Critic | `03-actor-critic.py` | 一步 TD 的 actor-critic，熵正则的作用 | `ActorCritic`、`a2c_update()` |
| 第 5 节 GAE；第 6 节 PPO | `04-gae-and-ppo.py` | 实现 GAE 的反向递推和 PPO-clip | `compute_gae()`、`ppo_clip_loss()`、`ppo_update()` |
| 第 7 节 三种梯度估计器 | `05-gradient-estimators.py` | 在一维高斯和伯努利例子上对比三种估计器的偏差与方差 | `score_function_grad()`、`pathwise_grad()`、`straight_through_sample()` |
| 第 8 节 DDPG 与 TD3 | `06-ddpg-td3.py` | 确定性策略梯度，TD3 的三项改进 | `ddpg_update()`、`td3_update()` |
| 第 9 节 SAC | `07-sac-from-scratch.py` | tanh 高斯策略、双 Q、自动调温度 | `TanhGaussianActor`、`sac_update()` |
| 第 10 节 MBPO | `08-mbpo.py` | 概率集成加短分支展开，生成数据喂给 SAC | `branched_rollout()`、`mbpo_train()` |
| 第 11 节 沿模型反向传播 | `09-backprop-through-model.py` | 在可微模型里展开，把回报的梯度直接传回策略 | `imagine_rollout()`、`actor_loss_through_model()` |
| 第 12.4 节 排行榜 | 项目 `projects/phase-05-control-benchmark/` | 同一评估协议下对比 PETS、PPO、SAC、MBPO | — |

建议的学习顺序：第 1–6 节是一条线，按顺序读；第 7 节读两遍，第一遍在读完第 6 节后，
第二遍在读完第 11 节后。第 8–11 节是另一条线。

### 0.5 符号表

下表在全课程符号约定的基础上增补。论文里的记号和本课不同的地方，在"说明"一列写明。

| 符号 | 含义 | 说明 |
| --- | --- | --- |
| $\pi_\phi(a \mid s)$ | 随机策略，参数 $\phi$ | 多数论文把策略参数记作 $\theta$；本课的 $\theta$ 留给世界模型 |
| $\mu_\phi(s)$，$\sigma_\phi(s)$ | 高斯策略的均值与标准差；确定性策略也记作 $\mu_\phi(s)$ | |
| $V_\psi(s)$，$Q_\psi(s, a)$ | critic，参数 $\psi$；$\bar\psi$ 是目标网络的参数 | 论文里常记作 $\phi$、$w$ 或 $\theta^Q$ |
| $p_\phi(\tau)$ | 策略 $\pi_\phi$ 与环境共同决定的轨迹分布 | |
| $J(\phi)$ | 策略的目标：期望回报 $\mathbb{E}_{\tau \sim p_\phi}[G_0]$ | Phase 1 的 $J(A)$ 是一条动作序列的分数，含义相近，对象不同 |
| $G_t$ | 从第 $t$ 步起的折扣回报 $\sum_{k \ge 0} \gamma^k r_{t+k}$ | 本章也叫 reward-to-go |
| $b(s)$ | 基线 | |
| $A^\pi(s, a)$ | 优势函数 $Q^\pi(s, a) - V^\pi(s)$ | 不要和 Phase 1 的动作序列 $A$ 混淆 |
| $\hat{A}_t$ | 第 $t$ 步的优势估计 | |
| $\delta_t$ | TD 误差 $r_t + \gamma V(s_{t+1}) - V(s_t)$ | 与 Phase 4 相同 |
| $\lambda$ | GAE / λ-return 的混合系数 | Phase 1 的 $\lambda$ 是 MPPI 的温度，与此无关 |
| $G_t^\lambda$ | λ-return | 与 Phase 4 相同 |
| $\rho_t(\phi)$ | 新旧策略的概率比 $\pi_\phi(a_t \mid s_t) / \pi_{\phi_{\text{old}}}(a_t \mid s_t)$ | PPO 论文记作 $r_t(\theta)$，为了不和奖励撞名改用 $\rho$ |
| $\epsilon_{\text{clip}}$ | PPO 的裁剪范围 | PPO 论文记作 $\epsilon$ |
| $\epsilon$ | 重参数化用的噪声，$\epsilon \sim \mathcal{N}(0, 1)$ | Phase 1 的 $\varepsilon$（模型单步误差）写法不同，含义不同 |
| $\mathcal{H}(\pi(\cdot \mid s))$ | 策略在状态 $s$ 的熵 | |
| $\alpha$，$\bar{\mathcal{H}}$ | SAC 的温度；目标熵 | |
| $u$ | tanh 之前的动作，$a = \tanh(u)$ | |
| $\kappa$ | 目标网络的软更新系数 | 论文记作 $\tau$，本课的 $\tau$ 是轨迹 |
| $\eta$ | 学习率 | |
| $c_v$，$c_{\text{ent}}$ | 价值损失、熵奖励在总损失里的系数 | |
| $f(x)$，$p_\phi(x)$ | 第 7 节讨论一般梯度估计问题时的被积函数与采样分布 | 这里的 $f$ 不是环境的转移函数 |
| $\hat{g}_{\text{SF}}$，$\hat{g}_{\text{PW}}$，$\hat{g}_{\text{ST}}$ | 三种估计器给出的单样本梯度估计 | |
| $k$ | MBPO 的分支展开长度 | |
| $\mathcal{D}_{\text{env}}$，$\mathcal{D}_{\text{model}}$ | 真实数据的回放池；模型生成数据的回放池 | |
| $H$ | 在模型里想象的步数 | 与 Phase 1 的规划视野是同一个符号 |
| `[B, T, D]` | 张量形状：批大小、时间长度、特征维度 | 并行环境数写作 `N`，动作维度写作 `A` |

关于奖励的下标：本课把在 $s_t$ 执行 $a_t$ 得到的奖励记作 $r_t$，Sutton & Barto 记作 $R_{t+1}$。
对照教材第 13 章的公式时，所有奖励的下标要减一。

---

## 1. 直接优化策略：策略梯度定理

### 1.1 直觉

到目前为止，课程里出现过两种做决策的办法：

- **规划**（Phase 1、3）：每走一步，在模型里试上千条动作序列，挑最好的。
- **价值函数**（Phase 4）：学一个 $Q(s, a)$，决策时取 $\arg\max_a Q(s, a)$。

第一种慢。第二种在离散动作上很好用，到了连续动作就卡住了：`HalfCheetah-v5` 的动作是 6 维实数向量，
$\arg\max_a$ 本身就是一个优化问题，每做一次决策都要解一遍。

第三种办法是把答案直接存下来：学一个函数 $\pi_\phi$，输入状态，输出动作。这就是**策略**（policy）。
对写代码的人来说，规划像解释执行，每个请求都现场算一遍；策略像提前编译，把搜索的结果固化成一次前向计算。

麻烦在于怎么训练它。监督学习需要标签，而这里没有人告诉你"在这个状态下正确的动作是什么"。
我们手上只有事后的一个分数：这一回合总共拿了多少奖励。

策略梯度的想法和线上的 A/B 实验一样：

1. 让策略带一点随机性，同一个状态下有时这样做、有时那样做
2. 记下每次做了什么、最后拿了多少分
3. 分数高的那些做法，以后多做；分数低的，以后少做

"多做"的意思是提高策略输出那个动作的概率。于是训练信号有了：**用回报给"提高这个动作的概率"这件事加权。**
本节把这个想法变成一个可以算的梯度公式。

### 1.2 定义

**随机策略。** 策略 $\pi_\phi(a \mid s)$ 是给定状态下动作的概率分布，由参数为 $\phi$ 的神经网络输出。
本章用到两种：

| 动作类型 | 网络输出 | 分布 | 采样 | 输出形状 |
| --- | --- | --- | --- | --- |
| 离散（如 `CartPole-v1` 的左、右） | 每个动作一个 logit | 类别分布，$\pi(a \mid s) = \mathrm{softmax}(z)_a$ | 按概率抽一个 | logits `[B, A]`，`A` 是动作个数 |
| 连续（如 `Pendulum-v1` 的力矩） | 均值 $\mu_\phi(s)$ 和对数标准差 $\log \sigma_\phi(s)$ | 各维独立的高斯分布 | $a = \mu + \sigma \epsilon$ | 均值、对数标准差各 `[B, A]`，`A` 是动作维度 |

**轨迹的概率。** 一条长度为 $T$ 的轨迹 $\tau = (s_0, a_0, r_0, s_1, \dots, s_T)$ 出现的概率，
是初始状态的概率、每一步策略选动作的概率、每一步环境转移的概率连乘起来：

$$
p_\phi(\tau) = p(s_0) \prod_{t=0}^{T-1} \pi_\phi(a_t \mid s_t) \, p(s_{t+1} \mid s_t, a_t)
$$

**目标。** 让轨迹的期望回报最大：

$$
J(\phi) = \mathbb{E}_{\tau \sim p_\phi}\big[ G_0(\tau) \big], \qquad G_0(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t
$$

这个目标和监督学习的损失有一个结构上的区别。监督学习里，参数出现在被求期望的函数里，数据分布是固定的：

$$
\mathcal{L}(\theta) = \mathbb{E}_{x \sim \text{数据}}\big[ \ell_\theta(x) \big]
$$

而在 $J(\phi)$ 里，$G_0(\tau)$ 这个函数本身不含 $\phi$（给定一条轨迹，回报是多少就是多少），
**参数藏在"按什么分布抽轨迹"里面。** 改变 $\phi$，改变的是哪些轨迹更容易出现。
对"采样分布的参数"求梯度，是整章的核心困难，第 7 节会把它单独拿出来讨论。

### 1.3 推导：策略梯度定理

目标是把 $\nabla_\phi J(\phi)$ 写成一个期望，这样才能用采样来估计。

**第一步：把期望写成积分。** 用期望的定义。

$$
J(\phi) = \int p_\phi(\tau) \, G_0(\tau) \, d\tau
$$

**第二步：把梯度移进积分号。** $G_0(\tau)$ 不含 $\phi$，所以梯度只作用在 $p_\phi(\tau)$ 上。

$$
\nabla_\phi J(\phi) = \int \nabla_\phi p_\phi(\tau) \, G_0(\tau) \, d\tau
$$

到这里卡住了：$\int \nabla_\phi p_\phi(\tau) (\cdots) d\tau$ 不是"某个量在 $p_\phi$ 下的期望"的形式，
因为被积式里没有 $p_\phi(\tau)$ 这个因子。不是期望，就没法靠采样轨迹来估计。

**第三步：对数导数技巧（log-derivative trick）。** 由对数的求导法则 $\nabla \log p = \nabla p / p$，移项得到一个恒等式：

$$
\nabla_\phi p_\phi(\tau) = p_\phi(\tau) \, \nabla_\phi \log p_\phi(\tau)
$$

代入上一步，$p_\phi(\tau)$ 这个因子回来了，积分重新变成期望：

$$
\nabla_\phi J(\phi) = \int p_\phi(\tau) \, \nabla_\phi \log p_\phi(\tau) \, G_0(\tau) \, d\tau
= \mathbb{E}_{\tau \sim p_\phi}\big[ \nabla_\phi \log p_\phi(\tau) \, G_0(\tau) \big]
$$

$\nabla \log \pi$ 这一项的来历就是这个恒等式：它是为了把"对分布求导"改写成"在这个分布下求期望"而付出的代价。

**第四步：展开 $\log p_\phi(\tau)$。** 对第 1.2 节的连乘式取对数，连乘变成求和：

$$
\log p_\phi(\tau) = \log p(s_0) + \sum_{t=0}^{T-1} \log \pi_\phi(a_t \mid s_t) + \sum_{t=0}^{T-1} \log p(s_{t+1} \mid s_t, a_t)
$$

对 $\phi$ 求梯度。第一项和第三项是环境的性质，不含 $\phi$，梯度为零：

$$
\nabla_\phi \log p_\phi(\tau) = \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t \mid s_t)
$$

这一步有一个重要的后果：**环境的转移规律从公式里消失了。** 算策略梯度不需要知道环境怎么运转，
也不需要环境可微。这就是策略梯度可以"无模型"的原因。

**第五步：代回去。**

$$
\nabla_\phi J(\phi) = \mathbb{E}_{\tau \sim p_\phi}\left[ \Big( \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t \mid s_t) \Big) \, G_0(\tau) \right]
$$

这已经是一个可用的公式。但它让每一步的动作都对整条轨迹的回报负责，包括这个动作发生之前就已经拿到的奖励。
下面把这部分多余的项去掉。

**第六步：因果性，得到 reward-to-go。** 先证明一个后面要反复用的引理。

**引理（得分函数的期望为零）。** 对任意状态 $s$：

$$
\mathbb{E}_{a \sim \pi_\phi(\cdot \mid s)}\big[ \nabla_\phi \log \pi_\phi(a \mid s) \big] = 0
$$

证明分三步，以离散动作为例（连续动作把求和换成积分）：

$$
\begin{aligned}
\mathbb{E}_{a \sim \pi_\phi}\big[ \nabla_\phi \log \pi_\phi(a \mid s) \big]
&= \sum_a \pi_\phi(a \mid s) \, \nabla_\phi \log \pi_\phi(a \mid s) && \text{期望的定义} \\
&= \sum_a \nabla_\phi \pi_\phi(a \mid s) && \text{对数导数技巧，反着用} \\
&= \nabla_\phi \sum_a \pi_\phi(a \mid s) = \nabla_\phi 1 = 0 && \text{概率之和恒为 1}
\end{aligned}
$$

直观含义：提高某些动作的概率，必然要降低另一些动作的概率，总账是平的。

现在把第五步的式子展开。$G_0 = \sum_{t'} \gamma^{t'} r_{t'}$，所以期望里是一个双重求和，
每一项形如 $\nabla_\phi \log \pi_\phi(a_t \mid s_t) \cdot \gamma^{t'} r_{t'}$。
看其中 $t' < t$ 的项，也就是"第 $t$ 步的动作"乘"它之前的奖励"。
记到第 $t$ 步选动作之前的全部历史为 $h_t = (s_0, a_0, \dots, s_{t-1}, a_{t-1}, s_t)$。
$r_{t'}$ 由 $(s_{t'}, a_{t'})$ 决定，而 $t' < t$，所以给定 $h_t$ 时 $r_{t'}$ 已经是一个确定的数。
用全期望公式，先对 $a_t$ 求期望，再对 $h_t$ 求期望：

$$
\begin{aligned}
\mathbb{E}\big[ \nabla_\phi \log \pi_\phi(a_t \mid s_t) \, r_{t'} \big]
&= \mathbb{E}_{h_t}\Big[ \, r_{t'} \cdot \underbrace{\mathbb{E}_{a_t \sim \pi_\phi(\cdot \mid s_t)}\big[ \nabla_\phi \log \pi_\phi(a_t \mid s_t) \big]}_{= \, 0 \text{（引理）}} \Big] = 0
\end{aligned}
$$

所以 $t' < t$ 的项期望全为零，可以删掉。只留下 $t' \ge t$ 的项：

$$
\nabla_\phi J(\phi)
= \mathbb{E}_{\tau \sim p_\phi}\left[ \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t \mid s_t) \sum_{t'=t}^{T-1} \gamma^{t'} r_{t'} \right]
= \mathbb{E}_{\tau \sim p_\phi}\left[ \sum_{t=0}^{T-1} \gamma^t \, \nabla_\phi \log \pi_\phi(a_t \mid s_t) \, G_t \right]
$$

第二个等号用了 $\sum_{t' \ge t} \gamma^{t'} r_{t'} = \gamma^t \sum_{k \ge 0} \gamma^k r_{t+k} = \gamma^t G_t$。
$G_t$ 只包含第 $t$ 步及之后的奖励，叫 **reward-to-go**。一句话概括这一步：动作不能影响过去，所以不该为过去的奖励记功或背锅。

注意删掉的那些项期望为零，方差却不为零。删掉它们不改变梯度的期望，只减少噪声。
"减去期望为零的项来降方差"这个手法，第 3 节还会再用一次。

**第七步：把 $G_t$ 换成 $Q^\pi$。** 再用一次全期望公式，这次先固定 $(s_t, a_t)$，对之后发生的事求期望。
按 Phase 4 的定义，$\mathbb{E}[G_t \mid s_t, a_t] = Q^\pi(s_t, a_t)$，所以：

$$
\nabla_\phi J(\phi) = \mathbb{E}_{\tau \sim p_\phi}\left[ \sum_{t=0}^{T-1} \gamma^t \, \nabla_\phi \log \pi_\phi(a_t \mid s_t) \, Q^{\pi}(s_t, a_t) \right]
$$

这就是**策略梯度定理**（Policy Gradient Theorem），对应 Sutton & Barto 第 13 章的结论。
它说：策略梯度是"得分函数 $\nabla \log \pi$"按"这个动作有多好"加权后的期望。

**关于 $\gamma^t$ 这个因子。** 严格推导出来的公式里，第 $t$ 步的项前面有一个 $\gamma^t$。
几乎所有实现（包括 PPO、A2C 的常见实现）都把它丢掉，只在 $G_t$ 内部保留折扣。
丢掉之后，得到的不再是折扣目标 $J(\phi)$ 的精确梯度，而是一个把每个时间步同等对待的近似。
这样做的理由是实用：否则回合后段的样本几乎没有权重。知道有这回事即可，本章后面的公式都按实现的惯例省略 $\gamma^t$。

### 1.4 两种策略的得分函数

$\nabla_\phi \log \pi_\phi$ 由自动求导来算，不用手推。但知道它长什么样，有助于理解梯度在做什么。

**softmax 策略。** $\pi(a) = e^{z_a} / \sum_b e^{z_b}$，所以 $\log \pi(a) = z_a - \log \sum_b e^{z_b}$。对第 $b$ 个 logit 求导：

$$
\frac{\partial \log \pi(a)}{\partial z_b} = \mathbb{1}[a = b] - \pi(b)
$$

被选中的那个动作，logit 的梯度是 $1 - \pi(a)$（正的）；没被选中的动作，梯度是 $-\pi(b)$（负的）。
这和分类任务里交叉熵对 logits 的梯度形式完全一样，只是差一个符号，并且"标签"是自己采样出来的动作。

**高斯策略**（一维）。$\log \pi(a) = -\dfrac{(a - \mu)^2}{2\sigma^2} - \log \sigma - \dfrac{1}{2} \log(2\pi)$。对均值和对数标准差求导：

$$
\frac{\partial \log \pi(a)}{\partial \mu} = \frac{a - \mu}{\sigma^2}, \qquad
\frac{\partial \log \pi(a)}{\partial \log \sigma} = \frac{(a - \mu)^2}{\sigma^2} - 1
$$

第一个式子的含义：采到的动作在均值右边，就把均值往右拉；偏得越远拉得越用力。
分母上的 $\sigma^2$ 值得留意：$\sigma$ 越小，这个量越大。第 7 节会看到，这正是 score-function 估计器在策略接近确定时方差爆炸的来源。

### 1.5 手算例子：两臂老虎机

用一个只有一步的问题把公式走一遍。没有状态，两个动作，奖励是确定的：

| 动作 | 奖励 |
| --- | --- |
| 动作 1 | 1 |
| 动作 2 | 3 |

策略是两个 logit 上的 softmax，当前 $z = (0, 0)$，所以 $\pi = (0.5, 0.5)$，期望回报 $J = 0.5 \times 1 + 0.5 \times 3 = 2$。

**先直接求导，得到标准答案。** $J = \pi_1 \times 1 + \pi_2 \times 3$。
softmax 的导数是 $\partial \pi_1 / \partial z_1 = \pi_1 (1 - \pi_1) = 0.25$，$\partial \pi_2 / \partial z_1 = -\pi_1 \pi_2 = -0.25$。所以：

$$
\frac{\partial J}{\partial z_1} = 0.25 \times 1 + (-0.25) \times 3 = -0.5, \qquad \frac{\partial J}{\partial z_2} = +0.5 \;\;\text{（同理）}
$$

应该降低动作 1 的 logit、提高动作 2 的，符合直觉。

**再用策略梯度公式算。** 一步问题里 $G_0 = r$。用第 1.4 节的得分函数：

| 采到的动作 | 概率 | 得分函数 $\nabla_z \log \pi$ | 奖励 | 单样本估计 $\nabla_z \log \pi \cdot r$ |
| --- | --- | --- | --- | --- |
| 动作 1 | 0.5 | $(1 - 0.5, \; -0.5) = (0.5, -0.5)$ | 1 | $(0.5, -0.5)$ |
| 动作 2 | 0.5 | $(-0.5, \; 1 - 0.5) = (-0.5, 0.5)$ | 3 | $(-1.5, 1.5)$ |

对两种情况按概率加权平均：

$$
0.5 \times (0.5, -0.5) + 0.5 \times (-1.5, 1.5) = (-0.5, \; 0.5)
$$

和直接求导的结果一致。

**这个例子里藏着方差问题的种子。** 看表的第一行：采到动作 1 时，单样本估计是 $(0.5, -0.5)$，
它在**提高**动作 1 的概率，而动作 1 是较差的那个。原因是它的奖励是正数，公式只看到"做了动作 1，拿了正分"。
只有把两种情况平均起来，方向才是对的。单个样本的方向可以是错的，这就是策略梯度的方差，第 2、3 节处理它。

### 1.6 和监督学习对照着看

把策略梯度公式里的期望换成样本平均，再添一个负号，就得到实现里的"损失"：

$$
\mathcal{L}_{\text{PG}}(\phi) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t} \log \pi_\phi(a_t^i \mid s_t^i) \cdot \mathrm{sg}(G_t^i)
$$

对它求梯度，正好是策略梯度估计的相反数，所以对它做梯度下降等于对 $J$ 做梯度上升。

对照分类任务的交叉熵 $-\log p_\theta(y \mid x)$：

| | 监督学习（分类） | 策略梯度 |
| --- | --- | --- |
| 输入 | $x$ | 状态 $s_t$ |
| "标签" | 人标的 $y$ | 策略自己采样出来的动作 $a_t$ |
| 每个样本的权重 | 1 | 回报 $G_t$（之后换成优势） |
| 数据分布 | 固定 | 随策略变化 |

所以策略梯度可以读成：**模仿自己做过的动作，回报高的多模仿，回报低的少模仿，回报为负的反着来。**

这个"损失"有两点和普通损失不同，初学时容易被它骗：

1. 它的数值没有意义。它不是在度量策略有多差，只是一个梯度恰好正确的辅助量，所以也叫**代理损失**（surrogate loss）。
   它下降不代表策略变好，判断策略好坏只能看回报。
2. 数据用完就得扔。期望是在当前策略的轨迹分布 $p_\phi$ 下取的。参数更新之后，分布变了，
   旧数据不再是从新分布里采的。这叫**在策略**（on-policy），和 Phase 4 里 DQN 能反复使用回放数据的**离策略**（off-policy）相对。

### 1.7 代码对应

对应 `01-policy-gradient-reinforce.py`（讲义待写，函数名以讲义为准）。讲义先在老虎机上验证第 1.5 节的数字，
再搬到 `CartPole-v1`。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `obs` | `[T, 4]` | 一个回合的观测，`CartPole-v1` 是 4 维 |
| `logits` | `[T, 2]` | 策略网络输出，两个离散动作 |
| `actions` | `[T]` | 采样出的动作（整数） |
| `log_probs` | `[T]` | $\log \pi_\phi(a_t \mid s_t)$，要保留计算图 |
| `returns` | `[T]` | $G_t$，不带梯度 |

核心只有三行：

```python
dist = torch.distributions.Categorical(logits=policy(obs))   # [T, 2]
log_probs = dist.log_prob(actions)                           # [T]
loss = -(log_probs * returns).mean()                         # returns 不带梯度
```

> 📌 待回填：老虎机例子里用 1 万个样本估计的策略梯度与解析值 $(-0.5, 0.5)$ 的对比，来自 `01-policy-gradient-reinforce.py` 的 `check_bandit_gradient()`。

### 1.8 常见误区

- **误区：策略梯度需要对环境求导，所以环境必须可微。**
  实际情况：第四步里环境的转移概率对 $\phi$ 的梯度为零，直接消失了。
  策略梯度只要求 $\log \pi_\phi$ 对 $\phi$ 可微。环境可以是黑盒，奖励可以不连续。
- **误区：$-\log \pi \cdot G$ 是损失，它降下去说明训练在变好。**
  实际情况：它是代理损失，数值没有意义。回报全是负数时它甚至是负的。只看回报曲线。
- **误区：策略梯度是在"对回报求导"。**
  实际情况：回报对参数没有可以求导的路径（中间隔着采样和环境）。
  策略梯度求的是"轨迹出现的概率"对参数的导数，再用回报加权。能直接对回报求导的是第 7 节的 pathwise 估计器，它需要更强的条件。
- **误区：随机策略只是为了探索，训练完应该换成确定性策略才对。**
  实际情况：随机性在这里是推导成立的前提。没有随机性就没有 $\log \pi$，也就没有这个公式。
  评估时取均值或概率最大的动作是常见做法，但那是部署时的选择，不是训练时的。

---

## 2. REINFORCE 与方差问题

### 2.1 直觉

策略梯度定理给出的是一个期望。实际训练时只能跑有限几个回合，用样本平均去近似它。
最直接的做法叫 **REINFORCE**（Williams 1992）：跑完一个回合，算出每一步的 $G_t$，按公式更新一次。

这个估计在期望上是对的，但噪声极大。原因可以用一个工程场景说明：
一次上线包含 200 个提交，上线后只有一个总的指标"这周营收涨了还是跌了"。
你要根据这一个数字，判断 200 个提交里每一个是好是坏。
营收受很多因素影响，大部分和任何一个具体的提交无关。只看一周，结论基本是噪声；
要看很多周，把无关因素平均掉，单个提交的影响才显出来。

REINFORCE 面对的是同样的处境：一个回合里有几百个动作，只有一个事后的回报。
把功劳或责任分摊到每个动作上，这个问题叫**信用分配**（credit assignment）。
REINFORCE 的分法最粗：每个动作都按它之后的全部回报来算。

### 2.2 定义

用 $N$ 条轨迹的样本平均替换期望，得到 REINFORCE 的梯度估计：

$$
\hat{g} = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t^i \mid s_t^i) \, G_t^i
$$

评价一个估计器看两个量：

| 量 | 含义 | REINFORCE |
| --- | --- | --- |
| **偏差**（bias） | 估计的期望和真实梯度差多少 | 零。$\mathbb{E}[\hat{g}] = \nabla_\phi J$（省略 $\gamma^t$ 带来的差别除外，见第 1.3 节末尾） |
| **方差**（variance） | 不同批次的样本给出的估计彼此差多少 | 很大 |

无偏不等于好用。梯度下降每一步用的是一个具体的估计值，不是它的期望。
方差大意味着每一步的方向都带着很大的随机成分，学习率必须很小，或者每次更新要用很多条轨迹。
Phase 4 讲蒙特卡洛与 TD 时见过同一对矛盾，那里估计的是价值，这里估计的是梯度。

### 2.3 推导：方差从哪来

先在第 1.5 节的老虎机上把方差算出来，再看它的三个来源。

**算老虎机的方差。** 只看第一个 logit 的梯度分量。单样本估计有两种取值：

| 采到的动作 | 概率 | 单样本估计（第一个分量） |
| --- | --- | --- |
| 动作 1 | 0.5 | $0.5 \times 1 = 0.5$ |
| 动作 2 | 0.5 | $-0.5 \times 3 = -1.5$ |

均值是 $0.5 \times 0.5 + 0.5 \times (-1.5) = -0.5$。用方差的定义 $\mathrm{Var}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$：

$$
\mathrm{Var} = \big( 0.5 \times 0.5^2 + 0.5 \times 1.5^2 \big) - (-0.5)^2 = 1.25 - 0.25 = 1.0
$$

标准差是 1.0，是梯度本身大小（0.5）的两倍。单个样本连方向都未必对。

**来源一：回报的绝对水平。** 把两个动作的奖励都加上 100，变成 101 和 103。
哪个动作更好没有变，真实梯度也没有变：

$$
\frac{\partial J}{\partial z_1} = 0.25 \times 101 - 0.25 \times 103 = -0.5
$$

但单样本估计变成了 $0.5 \times 101 = 50.5$ 和 $-0.5 \times 103 = -51.5$。均值仍然是 $-0.5$，方差是：

$$
\mathrm{Var} = \big( 0.5 \times 50.5^2 + 0.5 \times 51.5^2 \big) - 0.25 = 2601.25 - 0.25 = 2601
$$

| 奖励 | 真实梯度 | 单样本方差 | 标准差 | 让标准误差降到 0.25 需要的样本数 |
| --- | --- | --- | --- | --- |
| 1 和 3 | −0.5 | 1.0 | 1.0 | $1.0 / 0.25^2 = 16$ |
| 101 和 103 | −0.5 | 2601 | 51.0 | $2601 / 0.25^2 = 41616$ |

最后一列用的是：$N$ 个独立样本的平均值，方差是单样本方差的 $1/N$，标准误差是 $\sqrt{\mathrm{Var}/N}$。

一个对决策毫无影响的常数，让需要的样本数涨了 2600 倍。
有用的信息是两个动作之间差 2 分，它被淹没在"两个动作都有 100 多分"这个大数里。
`CartPole-v1` 正是这种情况：每一步奖励都是 +1，所有回报都是正的。第 3 节的基线就是为了去掉这个大数。

**来源二：回报是很多随机量的和。** $G_t$ 是第 $t$ 步之后所有奖励的和，
而之后的每一步都有随机性（策略在采样，环境可能也随机）。
粗略地假设每一步的奖励相互独立、方差都是 $\sigma_r^2$，不考虑折扣，则：

$$
\mathrm{Var}[G_t] = (T - t) \, \sigma_r^2
$$

用了"独立随机变量之和的方差等于方差之和"。回合越长，权重 $G_t$ 的噪声越大。
这部分噪声里绝大多数和 $a_t$ 这一个动作无关：它来自之后几百个动作的随机性。
第 1.3 节的 reward-to-go 已经砍掉了一截（之前的奖励不算），把方差从 $T \sigma_r^2$ 降到 $(T - t) \sigma_r^2$。
要砍掉之后的部分，需要用价值函数来代替"实际发生的未来"，这是第 4、5 节的内容。

**来源三：得分函数本身的大小。** 高斯策略的得分函数是 $(a - \mu)/\sigma^2$。
策略越接近确定（$\sigma$ 越小），这个量的波动越大。第 7 节会算出具体的公式。

三个来源和对应的办法：

| 方差来源 | 办法 | 在哪 |
| --- | --- | --- |
| 回报里包含动作之前的奖励 | reward-to-go | 第 1.3 节 |
| 回报的绝对水平 | 减去基线 | 第 3 节 |
| 回报是很多步随机量的和 | 用 critic 自举，截短求和 | 第 4、5 节 |
| 得分函数随 $\sigma$ 变小而变大 | 换一种估计器（pathwise） | 第 7 节 |

### 2.4 算法

```
算法：REINFORCE

输入：策略网络 π_φ，学习率 η，折扣 γ
重复：
 1. 用当前策略跑完 N 个回合，记录每一步的 (s_t, a_t, r_t)
 2. 对每个回合，从后往前算 reward-to-go：
        G_T = 0
        对 t = T−1 … 0：G_t = r_t + γ · G_{t+1}
 3. 代理损失：L = − mean( log π_φ(a_t | s_t) · G_t )                G_t 不带梯度
 4. 反向传播，φ ← φ − η ∇L
 5. 丢弃这批数据
```

第 2 步的递推用的是 $G_t = r_t + \gamma G_{t+1}$，Phase 4 讲回报时见过。

**手算例子。** 一个三步的回合，奖励是 $(1, 0, 2)$，$\gamma = 0.9$。从后往前：

$$
\begin{aligned}
G_2 &= 2 \\
G_1 &= 0 + 0.9 \times 2 = 1.8 \\
G_0 &= 1 + 0.9 \times 1.8 = 2.62
\end{aligned}
$$

这组数字在第 5 节讲 GAE 时还会用到。

很多实现在第 3 步之前会把一批 $G_t$ 标准化（减去批内均值、除以批内标准差）。
这相当于用了一个很粗糙的基线，再顺便统一了梯度的尺度。它管用的原因见下一节。

### 2.5 代码对应

对应 `01-policy-gradient-reinforce.py`（讲义待写）。

| 函数 | 输入形状 | 输出形状 | 说明 |
| --- | --- | --- | --- |
| `compute_returns_to_go(rewards, gamma)` | `[T]` | `[T]` | 从后往前递推 |
| `reinforce_update(policy, optimizer, batch)` | `obs` `[M, 4]`，`actions` `[M]`，`returns` `[M]` | 标量损失 | `M` 是这批回合的总步数 |

回合长度各不相同，所以不把数据整理成 `[B, T, D]`，而是把所有回合的步拼成一条长度为 `M` 的序列。
`returns` 必须逐回合计算后再拼接，不能跨回合递推。

> 📌 待回填：REINFORCE 在 `CartPole-v1` 上的学习曲线（多个随机种子的均值与波动范围），以及达到满分所需的回合数，来自 `01-policy-gradient-reinforce.py`。定性预期：能学会，但不同种子之间差别很大，曲线抖动明显。

### 2.6 常见误区

- **误区：REINFORCE 是无偏的，所以只是慢一点，最终一定收敛到好策略。**
  实际情况：无偏只保证"无穷多样本的平均"是对的。有限样本下，一次方向错误的大步更新可能把策略推到一个很差的区域，
  之后采到的数据都来自这个差策略，很难再回来。方差不只影响速度，也影响能不能学出来。
- **误区：方差大，把批量开大就行了。**
  实际情况：标准误差按 $1/\sqrt{N}$ 下降。上面的例子里方差大了 2600 倍，批量也得大 2600 倍。
  而这里的每个样本是一次真实的环境交互。先降低单样本方差，再考虑批量。
- **误区：把奖励整体加一个常数不影响学习，因为最优策略不变。**
  实际情况：最优策略确实不变，但梯度估计的方差变了，学习的难度完全不同。
  回合长度不固定时还有一层影响：每步加一个正数等于奖励"活得久"，会改变最优策略本身。

---

## 3. 基线与优势函数

### 3.1 直觉

上一节的例子里，两个动作的奖励是 101 和 103。有用的信息是"动作 2 比平均水平高 1 分，动作 1 低 1 分"。

这就是考试里"看排名不看分数"的道理。一张卷子全班都考了 100 分以上，101 分并不说明学得好；
要看的是比班里的平均分高还是低。把每个人的分数减去平均分，正负号就有了意义：
正的是高于平均，该鼓励；负的是低于平均，该抑制。

对策略梯度来说，"平均分"是这个状态下按当前策略行动能拿到的期望回报，也就是 Phase 4 学过的 $V^\pi(s)$。
从回报里减掉它，剩下的是"这个动作比平均水平好多少"，叫**优势**（advantage）。

### 3.2 定义

**基线**（baseline）是任意一个只依赖状态、不依赖动作的函数 $b(s)$。带基线的策略梯度是：

$$
\nabla_\phi J(\phi) = \mathbb{E}_{\tau \sim p_\phi}\left[ \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t \mid s_t) \, \big( G_t - b(s_t) \big) \right]
$$

**优势函数**是动作价值与状态价值之差：

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

它有一个直接由定义得到的性质：按策略对动作求平均，优势为零。
因为 Phase 4 里 $V^\pi(s) = \mathbb{E}_{a \sim \pi}[Q^\pi(s, a)]$，所以：

$$
\mathbb{E}_{a \sim \pi(\cdot \mid s)}\big[ A^\pi(s, a) \big] = \mathbb{E}_{a \sim \pi}\big[ Q^\pi(s, a) \big] - V^\pi(s) = 0
$$

优势有正有负，平均为零。这正是"相对于平均水平"的数学表达。

### 3.3 推导：基线不引入偏差，却能降低方差

**第一部分：不引入偏差。** 要证明减掉的那一项期望为零：

$$
\mathbb{E}_{\tau \sim p_\phi}\big[ \nabla_\phi \log \pi_\phi(a_t \mid s_t) \, b(s_t) \big] = 0
$$

用全期望公式，先固定 $s_t$ 对 $a_t$ 求期望。$b(s_t)$ 不依赖 $a_t$，可以提到内层期望外面：

$$
\begin{aligned}
\mathbb{E}\big[ \nabla_\phi \log \pi_\phi(a_t \mid s_t) \, b(s_t) \big]
&= \mathbb{E}_{s_t}\Big[ \, \mathbb{E}_{a_t \sim \pi_\phi(\cdot \mid s_t)}\big[ \nabla_\phi \log \pi_\phi(a_t \mid s_t) \, b(s_t) \big] \Big] && \text{全期望公式} \\
&= \mathbb{E}_{s_t}\Big[ \, b(s_t) \cdot \mathbb{E}_{a_t \sim \pi_\phi(\cdot \mid s_t)}\big[ \nabla_\phi \log \pi_\phi(a_t \mid s_t) \big] \Big] && b \text{ 不依赖 } a_t \\
&= \mathbb{E}_{s_t}\big[ \, b(s_t) \cdot 0 \, \big] = 0 && \text{第 1.3 节的引理}
\end{aligned}
$$

整个证明只用到一个条件：**$b$ 不依赖 $a_t$。** 第二个等号靠的就是它。
$b$ 可以是常数，可以是一个训练得很差的网络，甚至可以依赖 $s_t$ 之前的全部历史，梯度的期望都不变。
反过来，如果 $b$ 依赖 $a_t$，它就提不出来，引理用不上，估计就有偏了。

**第二部分：能降低方差。** 看一个状态、一个参数分量。记得分函数的这个分量为 $g = \partial \log \pi_\phi(a \mid s) / \partial \phi_j$，
单样本估计是 $g \, (G - b)$。它的方差是：

$$
\mathrm{Var}\big[ g \, (G - b) \big] = \mathbb{E}\big[ g^2 (G - b)^2 \big] - \big( \mathbb{E}[ g \, (G - b) ] \big)^2
$$

第一部分已经证明第二项和 $b$ 无关。所以要让方差最小，只需让第一项最小。对 $b$ 求导并令其为零：

$$
\frac{\partial}{\partial b} \, \mathbb{E}\big[ g^2 (G - b)^2 \big] = -2 \, \mathbb{E}\big[ g^2 (G - b) \big] = 0
\quad \Longrightarrow \quad
b^\star = \frac{\mathbb{E}[ g^2 \, G ]}{\mathbb{E}[ g^2 ]}
$$

最优基线是回报的加权平均，权重是得分函数的平方。它不好算（每个参数分量都不一样），
实践中用不加权的平均 $\mathbb{E}[G \mid s] = V^\pi(s)$ 来代替。这不是最优的，但通常已经拿到了大部分好处。

**第三部分：从带基线的回报到优势。** 取 $b(s) = V^\pi(s)$，权重是 $G_t - V^\pi(s_t)$。
按第 1.3 节第七步的办法，把 $G_t$ 换成它的条件期望 $Q^\pi(s_t, a_t)$，得到：

$$
\nabla_\phi J(\phi) = \mathbb{E}_{\tau \sim p_\phi}\left[ \sum_{t=0}^{T-1} \nabla_\phi \log \pi_\phi(a_t \mid s_t) \, A^\pi(s_t, a_t) \right]
$$

这是策略梯度最常用的形式。真实的 $A^\pi$ 不知道，要估计。不同的估计办法就是不同的算法：

| 权重 $\Psi_t$ | 名字 | 偏差 | 方差 | 谁在用 |
| --- | --- | --- | --- | --- |
| $G_0$ | 整条轨迹的回报 | 无 | 最大 | 第 1.3 节第五步，只有推导意义 |
| $G_t$ | reward-to-go | 无 | 大 | REINFORCE |
| $G_t - V_\psi(s_t)$ | 蒙特卡洛优势 | 无（不管 $V_\psi$ 准不准） | 中 | 带基线的 REINFORCE |
| $r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$ | TD 误差 $\delta_t$ | 有（$V_\psi$ 不准时） | 小 | 一步 actor-critic，第 4 节 |
| $\sum_l (\gamma\lambda)^l \delta_{t+l}$ | GAE | 介于两者之间，由 $\lambda$ 调节 | 介于两者之间 | PPO，第 5 节 |

这张表改写自 GAE 论文（Schulman et al. 2016，arXiv:1506.02438）第 2 节列出的几种形式。它是第 3–5 节的路线图。

### 3.4 手算例子

**例一：基线把方差降到零。** 回到奖励为 101 和 103 的老虎机，$\pi = (0.5, 0.5)$。
取基线 $b = V = 0.5 \times 101 + 0.5 \times 103 = 102$。两个动作的优势是 $-1$ 和 $+1$。

| 采到的动作 | 得分函数（第一个分量） | $r - b$ | 单样本估计 |
| --- | --- | --- | --- |
| 动作 1 | $0.5$ | $-1$ | $0.5 \times (-1) = -0.5$ |
| 动作 2 | $-0.5$ | $+1$ | $-0.5 \times 1 = -0.5$ |

两种情况给出的估计都是 $-0.5$，恰好等于真实梯度。方差从 2601 降到了 0，一个样本就够。

这个例子也解释了第 1.5 节的那个怪现象。没有基线时，采到较差的动作 1 反而在提高它的概率。
有了基线，动作 1 的权重是负的，它的概率被压低，方向对了。

**例二：$V$ 不是最优基线。** 把策略换成不均匀的 $\pi = (0.8, 0.2)$，奖励还是 1 和 3。
真实梯度（第一个分量）是 $\pi_1(1 - \pi_1) \times 1 - \pi_1 \pi_2 \times 3 = 0.16 - 0.48 = -0.32$。
得分函数的第一个分量：采到动作 1 是 $1 - 0.8 = 0.2$，采到动作 2 是 $-0.8$。

| 基线 | 采到动作 1（概率 0.8） | 采到动作 2（概率 0.2） | 均值 | 方差 |
| --- | --- | --- | --- | --- |
| $b = 0$ | $0.2 \times 1 = 0.2$ | $-0.8 \times 3 = -2.4$ | $-0.32$ | $1.184 - 0.1024 = 1.0816$ |
| $b = V = 1.4$ | $0.2 \times (-0.4) = -0.08$ | $-0.8 \times 1.6 = -1.28$ | $-0.32$ | $0.3328 - 0.1024 = 0.2304$ |
| $b = b^\star = 2.6$ | $0.2 \times (-1.6) = -0.32$ | $-0.8 \times 0.4 = -0.32$ | $-0.32$ | $0$ |

其中 $V = 0.8 \times 1 + 0.2 \times 3 = 1.4$，最优基线按公式算：

$$
b^\star = \frac{0.8 \times 0.2^2 \times 1 + 0.2 \times 0.8^2 \times 3}{0.8 \times 0.2^2 + 0.2 \times 0.8^2} = \frac{0.032 + 0.384}{0.032 + 0.128} = 2.6
$$

三行的均值都是 $-0.32$，验证了基线不改变期望。用 $V$ 做基线，方差降到原来的五分之一左右；
最优基线降得更多。（只有两个动作时最优基线总能把方差降到零，这是个特例，动作多了就不行。）

### 3.5 算法

基线要学。最简单的办法是另起一个网络 $V_\psi$，用回归去拟合观测到的回报：

```
算法：带基线的 REINFORCE

输入：策略网络 π_φ，价值网络 V_ψ
重复：
 1. 用当前策略跑完 N 个回合
 2. 对每个回合从后往前算 G_t
 3. 优势估计：Â_t = G_t − V_ψ(s_t)                         不带梯度
 4. 策略损失：L_π = − mean( log π_φ(a_t | s_t) · Â_t )
 5. 价值损失：L_V = mean( (V_ψ(s_t) − G_t)² )               G_t 不带梯度
 6. 分别反向传播，更新 φ 和 ψ
```

第 3 步里 $V_\psi(s_t)$ 必须从计算图里断开（`detach`）。否则策略损失的梯度会流进价值网络，
让它去学"怎样让优势看起来更有利于策略损失下降"，而不是去拟合回报。

### 3.6 代码对应

对应 `02-baselines-and-advantage.py`（讲义待写）。这一节讲义的重点是把方差**量出来**：

| 函数 | 做什么 |
| --- | --- |
| `ValueNet` | 输入 `[M, 4]`，输出 `[M]` 的状态价值 |
| `measure_gradient_variance(policy, estimator, K)` | 固定策略参数，重复采 `K` 批数据，每批算一次梯度，拼成 `[K, P]`（`P` 是策略参数的总个数），再对第 0 维求方差 |

用同一个策略、同样多的数据，分别用"整条轨迹的回报""reward-to-go""带基线"三种权重各算 `K` 次梯度，
比较 `[K, P]` 沿第 0 维的方差。三者的均值应该一致，方差应该依次变小。

> 📌 待回填：三种权重（$G_0$、$G_t$、$G_t - V_\psi$）下梯度估计的方差之比，以及对应的学习曲线对比，来自 `02-baselines-and-advantage.py`。

### 3.7 常见误区

- **误区：基线不准会让梯度有偏，所以要先把价值网络训练好。**
  实际情况：只要基线不依赖当前动作，它再不准，梯度的期望也不变，只是方差降得少。
  会引入偏差的是下一节的做法：用 $V_\psi$ 去**自举**，那时它才必须准。
- **误区：既然 $V$ 好用，那用 $Q_\psi(s, a)$ 当基线会更好。**
  实际情况：$Q$ 依赖动作，第 3.3 节的证明不成立，直接减掉会有偏。
  依赖动作的基线需要额外加一个修正项才能保持无偏，本课不讲。
- **误区：优势是负的，说明这个动作很差。**
  实际情况：优势是相对量。在一个怎么做都很好的状态里，稍差一点的动作优势也是负的。
  它只说明这个动作低于当前策略在这个状态的平均水平。
- **误区：对优势做批内标准化只是一个数值技巧，不会改变什么。**
  实际情况：减去批内均值相当于又加了一个常数基线，不引入偏差；除以标准差改变了梯度的整体尺度，
  等价于每个批次用不同的学习率。它通常有帮助，但批量很小时会带来明显的噪声。

---

## 4. Actor-Critic

### 4.1 直觉

带基线的 REINFORCE 还有两个毛病：

1. 必须等回合结束才能算 $G_t$。回合很长或者根本不结束时，没法用。
2. $G_t$ 仍然是几百步随机量的和，第 2.3 节的"来源二"没有解决。

价值网络 $V_\psi$ 已经在手上了，上一节只拿它当基线用。Phase 4 讲过它更大的用处：**自举**。
不用等到回合结束看实际拿了多少，走一步之后直接问 $V_\psi$："从下一个状态出发，预计还能拿多少？"

这像代码评审和线上指标的区别。线上指标（蒙特卡洛回报）是最终的事实，但要等很久，而且受很多无关因素干扰。
评审者（critic）看一眼改动就能给出意见，反馈快、波动小；代价是评审者自己可能看走眼。

于是有了两个角色：

- **actor**（演员）：策略 $\pi_\phi$，负责选动作
- **critic**（评论家）：价值函数 $V_\psi$，负责评价 actor 的动作比预期好还是差

```
                  ┌───────────────┐
        s_t ────▶ │  actor  π_φ   │ ────▶ a_t ────▶ 环境 ────▶ r_t, s_{t+1}
                  └───────────────┘                               │
                          ▲                                       │
                          │ 用 δ_t 加权 ∇ log π                    │
                          │                                       ▼
                  ┌───────────────┐         δ_t = r_t + γ V(s_{t+1}) − V(s_t)
                  │  critic  V_ψ  │ ◀──────────────────────────────┘
                  └───────────────┘         critic 自己也用 δ_t 来更新
```

### 4.2 定义

一步 actor-critic 用 TD 误差作为优势的估计：

$$
\hat{A}_t = \delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

两个网络各有一个损失。写成要最小化的形式：

$$
\begin{aligned}
\mathcal{L}_{\text{actor}}(\phi) &= -\,\mathbb{E}_t\big[ \log \pi_\phi(a_t \mid s_t) \cdot \mathrm{sg}(\hat{A}_t) \big] \\
\mathcal{L}_{\text{critic}}(\psi) &= \mathbb{E}_t\Big[ \big( V_\psi(s_t) - \mathrm{sg}\big( r_t + \gamma V_\psi(s_{t+1}) \big) \big)^2 \Big]
\end{aligned}
$$

两个 $\mathrm{sg}$ 的含义不同：

| 位置 | 为什么要停止梯度 |
| --- | --- |
| actor 损失里的 $\hat{A}_t$ | 优势是给策略梯度加权用的数，不该让 actor 的损失去改 critic |
| critic 损失里的目标 $r_t + \gamma V_\psi(s_{t+1})$ | 这是 Phase 4 讲过的半梯度 TD：目标当作常数，只调整 $V_\psi(s_t)$ 去靠近它 |

实际的损失还会加一项熵奖励（第 4.4 节），三项合在一起：

$$
\mathcal{L} = \mathcal{L}_{\text{actor}} + c_v \, \mathcal{L}_{\text{critic}} - c_{\text{ent}} \, \mathbb{E}_t\big[ \mathcal{H}(\pi_\phi(\cdot \mid s_t)) \big]
$$

### 4.3 推导：TD 误差为什么是优势的估计

**第一步：critic 准确时，TD 误差的期望就是优势。** 设 $V_\psi = V^\pi$。固定 $(s_t, a_t)$，对下一状态求期望：

$$
\begin{aligned}
\mathbb{E}\big[ \delta_t \mid s_t, a_t \big]
&= \mathbb{E}\big[ r_t + \gamma V^\pi(s_{t+1}) \mid s_t, a_t \big] - V^\pi(s_t) && V^\pi(s_t) \text{ 在给定 } s_t \text{ 时是常数} \\
&= Q^\pi(s_t, a_t) - V^\pi(s_t) && Q^\pi \text{ 的 Bellman 方程（Phase 4）} \\
&= A^\pi(s_t, a_t) && \text{优势的定义}
\end{aligned}
$$

所以 $\delta_t$ 是优势的无偏估计，而且它只包含一步的随机性（$r_t$ 和 $s_{t+1}$），方差远小于 $G_t - V(s_t)$。

**第二步：critic 不准时，偏差是多少。** 记 critic 的误差为 $e(s) = V_\psi(s) - V^\pi(s)$。把 $V_\psi = V^\pi + e$ 代入：

$$
\begin{aligned}
\mathbb{E}\big[ \delta_t \mid s_t, a_t \big]
&= \mathbb{E}\big[ r_t + \gamma V^\pi(s_{t+1}) + \gamma e(s_{t+1}) \mid s_t, a_t \big] - V^\pi(s_t) - e(s_t) \\
&= A^\pi(s_t, a_t) + \underbrace{\gamma \, \mathbb{E}\big[ e(s_{t+1}) \mid s_t, a_t \big] - e(s_t)}_{\text{偏差}}
\end{aligned}
$$

偏差由 critic 在两个相邻状态上的误差决定。其中 $-e(s_t)$ 不依赖 $a_t$，它的作用相当于一个基线，不影响梯度的期望。
真正造成偏差的是 $\gamma \, \mathbb{E}[e(s_{t+1}) \mid s_t, a_t]$：critic 对"不同动作通向的下一状态"评价有误，
就会系统性地偏袒某些动作。

**第三步：和蒙特卡洛优势对比。**

| | 蒙特卡洛优势 $G_t - V_\psi(s_t)$ | TD 优势 $\delta_t$ |
| --- | --- | --- |
| 用到 critic 的方式 | 只当基线 | 当基线，并且用来自举 |
| 偏差 | 无，critic 再差也无偏 | critic 不准时有偏 |
| 方差 | 大：包含之后所有步的随机性 | 小：只包含一步的随机性 |
| 需要等回合结束吗 | 需要 | 不需要，每一步都能更新 |

这和 Phase 4 里蒙特卡洛与 TD 的对比是同一张表，只是那里估计的是价值，这里估计的是优势。
两个极端之间有一整条连续的折中，就是下一节的 GAE。

**手算例子。** $\gamma = 0.9$。critic 认为当前状态值 5.0，下一个状态值 6.0，这一步的奖励是 1：

$$
\delta_t = 1 + 0.9 \times 6.0 - 5.0 = 1.4
$$

TD 误差为正：这一步之后的局面比 critic 原先预计的好。于是提高这个动作的概率。
同时 critic 把 $V_\psi(s_t)$ 往目标 $1 + 0.9 \times 6.0 = 6.4$ 的方向调。

如果同样的状态下另一个动作通向的下一状态只值 4.0，奖励还是 1：
$\delta_t = 1 + 3.6 - 5.0 = -0.4$，这个动作的概率被降低。

### 4.4 熵正则

策略梯度有一个自我强化的失败模式。某个动作早期碰巧拿了几次高回报，它的概率被提高；
概率高了，被采样得更多，其他动作被采样得更少；其他动作的信息越来越少，更没有机会翻身。
策略很快变成几乎确定的，探索就此停止。一旦某个动作的概率接近 0，它几乎不再被采到，也就不再有关于它的梯度。

对付它的办法是奖励策略保持随机。衡量随机程度的量是**熵**（entropy）：

$$
\mathcal{H}(\pi(\cdot \mid s)) = -\,\mathbb{E}_{a \sim \pi}\big[ \log \pi(a \mid s) \big]
$$

**手算例子。** 两个动作的类别分布，熵是 $-\sum_a \pi(a) \log \pi(a)$（用自然对数）：

| 策略 | 熵 |
| --- | --- |
| $(0.5, 0.5)$ | $-2 \times 0.5 \times \ln 0.5 = 0.693$ |
| $(0.9, 0.1)$ | $-(0.9 \ln 0.9 + 0.1 \ln 0.1) = 0.095 + 0.230 = 0.325$ |
| $(0.99, 0.01)$ | $-(0.99 \ln 0.99 + 0.01 \ln 0.01) = 0.010 + 0.046 = 0.056$ |

越接近确定，熵越小。均匀分布的熵最大，等于 $\ln(\text{动作个数})$。

一维高斯分布的熵有闭式解，只和标准差有关：

$$
\mathcal{H}\big( \mathcal{N}(\mu, \sigma^2) \big) = \frac{1}{2} \log(2\pi e \sigma^2) = \log \sigma + 1.419
$$

$\sigma = 1$ 时熵是 1.419；$\sigma = 0.1$ 时是 $\ln 0.1 + 1.419 = -0.884$。
连续分布的熵（微分熵）可以是负数，这一点在第 9 节设定 SAC 的目标熵时会用到。

在损失里减去 $c_{\text{ent}} \cdot \mathcal{H}$，相当于给"保持随机"发一点奖金。
系数 $c_{\text{ent}}$ 通常很小，它的作用是推迟策略变确定的时间，给探索留出余地。

这里的熵只是加在损失上的一个正则项。第 9 节的 SAC 走得更远：把熵写进目标本身，连 critic 估计的价值里都包含未来的熵。
Dreamer 的 actor 损失里也有熵项（Phase 6）。

### 4.5 算法

```
算法：同步的优势 actor-critic（A2C），N 个并行环境，每次收集 T 步

输入：actor π_φ，critic V_ψ（可以共用前几层），系数 c_v、c_ent
重复：
 1. 每个环境用当前策略走 T 步，记录 s, a, r, terminated, truncated          形状 [T, N, …]
 2. 算出每一步的 V_ψ(s_t)，以及最后一个状态的 V_ψ(s_T)（用于自举）
 3. 算优势 Â_t 和价值目标 R_t = Â_t + V_ψ(s_t)                            一步 TD、n 步或 GAE（第 5 节）
 4. 损失 = − mean( log π_φ(a_t | s_t) · Â_t )
          + c_v · mean( (V_ψ(s_t) − R_t)² )
          − c_ent · mean( H(π_φ(· | s_t)) )                               Â_t、R_t 不带梯度
 5. 反向传播，更新 φ 和 ψ；丢弃这批数据
```

并行环境的作用是让一个批次里的样本来自不同的回合，彼此相关性低。这是在策略方法里代替经验回放的办法。
A2C 是 A3C（Mnih et al. 2016，Asynchronous Methods for Deep Reinforcement Learning）的同步版本。

**回合边界怎么处理。** Phase 1 第 3.6 节提过 `terminated` 和 `truncated` 的区别，这里必须分开：

| 情况 | 自举目标 | 原因 |
| --- | --- | --- |
| 正常的一步 | $r_t + \gamma V_\psi(s_{t+1})$ | |
| `terminated`（真的结束了） | $r_t$ | 终止状态之后没有奖励，价值按定义为 0 |
| `truncated`（到了步数上限） | $r_t + \gamma V_\psi(s_{t+1})$ | 环境本来还能继续，后面还有奖励，只是不让走了 |

把截断当成终止，等于告诉 critic"第 200 步之后什么都没有"。critic 会学出一个随时间步下降的价值，
而状态里并没有时间信息，于是学出来的是矛盾的东西。

### 4.6 代码对应

对应 `03-actor-critic.py`（讲义待写）。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `obs` | `[T, N, D]` | `T` 步，`N` 个并行环境，观测 `D` 维 |
| `actions`，`log_probs`，`rewards` | `[T, N]` | 离散动作 |
| `values` | `[T + 1, N]` | 多出的一行是最后一个状态的价值，用于自举 |
| `terminated` | `[T, N]` | 布尔；自举时乘以 `1 - terminated` |
| `advantages`，`value_targets` | `[T, N]` | 不带梯度 |
| `entropy` | `[T, N]` | `dist.entropy()` |

`ActorCritic` 模块的 `forward` 返回动作分布和价值；`a2c_update()` 实现上面的第 3–5 步。

> 📌 待回填：一步 actor-critic 与带基线的 REINFORCE 在 `CartPole-v1` 上的学习曲线对比；熵系数取 0 与取一个小正数时策略熵随训练的变化，来自 `03-actor-critic.py`。定性预期：不加熵正则时策略熵下降得更快，个别种子会卡在次优策略上。

### 4.7 常见误区

- **误区：actor-critic 用了价值函数，所以它是基于价值的方法。**
  实际情况：动作仍然由策略网络直接输出，更新仍然沿着策略梯度。critic 只是换掉了公式里的权重。
  基于价值的方法（DQN）没有单独的策略，动作由 $\arg\max Q$ 得出。
- **误区：actor 和 critic 共用网络主干一定更好，因为参数共享省计算。**
  实际情况：两个损失的梯度尺度可能差几个数量级，共用主干时大的那个会主导表示的学习。
  这就是需要系数 $c_v$ 的原因。输入是低维状态时，两个独立的网络通常更省心。
- **误区：critic 的损失越低，actor 学得越好。**
  实际情况：critic 追的目标随策略变化，策略一变，旧的价值就过时了。
  critic 损失长期接近零往往说明策略已经不变了，或者回报本身没有区分度。
- **误区：TD 误差里的两个 $V$ 都应该传梯度。**
  实际情况：目标 $r_t + \gamma V_\psi(s_{t+1})$ 要停止梯度。让梯度从两头都流，
  critic 可以通过同时抬高或压低两个状态的价值来让误差变小，而不是去拟合真实的回报。

---

## 5. 广义优势估计（GAE）

### 5.1 直觉

上一节留下一道选择题：

- 一步 TD 优势 $\delta_t$：方差小，但全靠 critic，critic 不准就有偏
- 蒙特卡洛优势 $G_t - V(s_t)$：无偏，但方差大

Phase 4 讲 λ-return 时遇到过完全一样的选择，当时的答案是不选：
把 1 步、2 步、3 步……的回报按指数衰减的权重全部混在一起，用一个旋钮 $\lambda$ 调节更相信哪一头。

广义优势估计（Generalized Advantage Estimation，GAE）就是把这个办法用到优势上。
按评审的类比：$\lambda$ 决定"多大程度相信评审意见，多大程度等实际结果"。
$\lambda = 0$ 完全相信评审（一步 TD），$\lambda = 1$ 完全等结果（蒙特卡洛）。

### 5.2 定义

**n 步优势估计。** 先实际走 $n$ 步，把这 $n$ 步的真实奖励加起来，之后的部分交给 critic：

$$
\hat{A}_t^{(n)} = \underbrace{\sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n V(s_{t+n})}_{\text{n 步回报 } G_t^{(n)}\text{（Phase 4）}} - V(s_t)
$$

$n = 1$ 就是 $\delta_t$；$n \to \infty$ 就是 $G_t - V(s_t)$。

**GAE。** 所有 $n$ 步估计的指数加权平均，权重是 $(1 - \lambda) \lambda^{n-1}$：

$$
\hat{A}_t^{\mathrm{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \, \hat{A}_t^{(n)}
$$

权重之和为 1：$(1 - \lambda)(1 + \lambda + \lambda^2 + \cdots) = 1$。

这个定义没法直接算，要对无穷多项求和。下面把它化成一个可以递推的形式。

### 5.3 推导：从 TD 误差的指数加权和到递推式

**第一步：n 步优势等于 TD 误差的折扣和。** 把 $\delta_{t+l} = r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})$ 代入求和：

$$
\begin{aligned}
\sum_{l=0}^{n-1} \gamma^l \delta_{t+l}
&= \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \sum_{l=0}^{n-1} \big( \gamma^{l+1} V(s_{t+l+1}) - \gamma^l V(s_{t+l}) \big)
\end{aligned}
$$

第二个求和是"裂项相消"的形式。写出前几项：

$$
\big( \gamma V(s_{t+1}) - V(s_t) \big) + \big( \gamma^2 V(s_{t+2}) - \gamma V(s_{t+1}) \big) + \cdots + \big( \gamma^n V(s_{t+n}) - \gamma^{n-1} V(s_{t+n-1}) \big)
$$

相邻两个括号里，前一个的正项和后一个的负项抵消。只剩头尾：$\gamma^n V(s_{t+n}) - V(s_t)$。所以：

$$
\sum_{l=0}^{n-1} \gamma^l \delta_{t+l} = \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n V(s_{t+n}) - V(s_t) = \hat{A}_t^{(n)}
$$

**第二步：代入 GAE 的定义，交换求和顺序。**

$$
\hat{A}_t^{\mathrm{GAE}} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}
$$

这是一个双重求和。换个角度数：某一个固定的 $\delta_{t+l}$ 出现在哪些 $n$ 里？
它出现在所有 $n \ge l + 1$ 的项里（走的步数至少要够到第 $l$ 步）。把它的系数收集起来：

$$
\begin{aligned}
\delta_{t+l} \text{ 的系数}
&= (1 - \lambda) \, \gamma^l \sum_{n=l+1}^{\infty} \lambda^{n-1} \\
&= (1 - \lambda) \, \gamma^l \cdot \frac{\lambda^l}{1 - \lambda} && \text{等比数列求和：首项 } \lambda^l \text{，公比 } \lambda \\
&= (\gamma\lambda)^l
\end{aligned}
$$

**第三步：得到闭式。**

$$
\hat{A}_t^{\mathrm{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \, \delta_{t+l}
$$

GAE 就是 TD 误差的指数加权和，衰减率是 $\gamma\lambda$。

**第四步：写成递推。** 把求和的第一项拆出来，剩下的部分提出一个 $\gamma\lambda$：

$$
\hat{A}_t = \delta_t + \gamma\lambda \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+1+l} = \delta_t + \gamma\lambda \, \hat{A}_{t+1}
$$

从轨迹末尾往前扫一遍就能算出所有时间步的优势，复杂度是 $O(T)$。

**两个极端。**

| $\lambda$ | $\hat{A}_t$ | 等于什么 |
| --- | --- | --- |
| $0$ | $\delta_t$ | 一步 TD 优势（约定 $0^0 = 1$） |
| $1$ | $\sum_l \gamma^l \delta_{t+l}$ | 由第一步的裂项相消，等于 $G_t - V(s_t)$，蒙特卡洛优势 |

**第五步：和 λ-return 的关系。** Phase 4 定义的 λ-return 是 n 步回报的同一种加权平均：

$$
G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}
$$

而 $\hat{A}_t^{(n)} = G_t^{(n)} - V(s_t)$。权重之和为 1，所以减掉的 $V(s_t)$ 可以整个提出来：

$$
\hat{A}_t^{\mathrm{GAE}} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} \big( G_t^{(n)} - V(s_t) \big) = G_t^\lambda - V(s_t)
$$

**GAE 就是 λ-return 减去基线。** 它们是同一个东西的两种写法：

$$
G_t^\lambda = \hat{A}_t^{\mathrm{GAE}} + V(s_t)
$$

递推式也对得上。Phase 4 的 λ-return 递推是 $G_t^\lambda = r_t + \gamma \big( (1 - \lambda) V(s_{t+1}) + \lambda G_{t+1}^\lambda \big)$。
两边减去 $V(s_t)$，并把右边的 $\gamma V(s_{t+1})$ 凑出来：

$$
\begin{aligned}
G_t^\lambda - V(s_t)
&= r_t + \gamma V(s_{t+1}) - V(s_t) + \gamma\lambda \big( G_{t+1}^\lambda - V(s_{t+1}) \big) \\
&= \delta_t + \gamma\lambda \, \hat{A}_{t+1}
\end{aligned}
$$

正是第四步的递推。这个关系在实现里直接用到：算完 GAE，加回 $V(s_t)$ 就是 critic 的回归目标，不用另算一遍。

### 5.4 手算例子

沿用第 2.4 节的三步回合：奖励 $(1, 0, 2)$，$\gamma = 0.9$。
critic 给出的价值是 $V(s_0) = 2.0$，$V(s_1) = 1.5$，$V(s_2) = 1.8$；$s_3$ 是终止状态，$V(s_3) = 0$。

**先算 TD 误差：**

$$
\begin{aligned}
\delta_0 &= 1 + 0.9 \times 1.5 - 2.0 = 0.35 \\
\delta_1 &= 0 + 0.9 \times 1.8 - 1.5 = 0.12 \\
\delta_2 &= 2 + 0.9 \times 0 - 1.8 = 0.20
\end{aligned}
$$

**取 $\lambda = 0.8$，$\gamma\lambda = 0.72$，从后往前递推：**

$$
\begin{aligned}
\hat{A}_2 &= \delta_2 = 0.20 \\
\hat{A}_1 &= 0.12 + 0.72 \times 0.20 = 0.264 \\
\hat{A}_0 &= 0.35 + 0.72 \times 0.264 = 0.540
\end{aligned}
$$

**三种 $\lambda$ 的结果放在一起：**

| | $\hat{A}_0$ | $\hat{A}_1$ | $\hat{A}_2$ |
| --- | --- | --- | --- |
| $\lambda = 0$（一步 TD） | 0.35 | 0.12 | 0.20 |
| $\lambda = 0.8$ | 0.540 | 0.264 | 0.20 |
| $\lambda = 1$（蒙特卡洛） | 0.62 | 0.30 | 0.20 |

**验证 $\lambda = 1$ 那一行。** 第 2.4 节算过 $G_0 = 2.62$，$G_1 = 1.8$，$G_2 = 2$。
减去价值：$2.62 - 2.0 = 0.62$，$1.8 - 1.5 = 0.30$，$2 - 1.8 = 0.20$。和表里一致。

**验证和 λ-return 的关系。** $\lambda = 0.8$ 时，critic 的目标是 $\hat{A}_t + V(s_t)$：$(2.540, 1.764, 2.0)$。
用 Phase 4 的递推直接算 λ-return 来对照：

$$
\begin{aligned}
G_2^\lambda &= 2 + 0.9 \times (0.2 \times 0 + 0.8 \times 0) = 2 \\
G_1^\lambda &= 0 + 0.9 \times (0.2 \times 1.8 + 0.8 \times 2) = 0.9 \times 1.96 = 1.764 \\
G_0^\lambda &= 1 + 0.9 \times (0.2 \times 1.5 + 0.8 \times 1.764) = 1 + 0.9 \times 1.7112 = 2.540
\end{aligned}
$$

两条路算出的结果相同。

### 5.5 $\gamma$ 和 $\lambda$ 各管什么

两个参数在公式里总是以乘积 $\gamma\lambda$ 的形式出现，但它们的角色不同。GAE 论文第 3 节专门讨论了这一点：

| | $\gamma$ | $\lambda$ |
| --- | --- | --- |
| 改变了什么 | 目标本身：多远的奖励还算数 | 估计的方式：多相信 critic |
| critic 完全准确时 | $\gamma < 1$ 仍然让估计偏离不带折扣的真实目标 | 不引入任何偏差 |
| critic 不准时 | 同上 | $\lambda < 1$ 引入偏差，$\lambda$ 越小偏差越大 |
| 调小的效果 | 方差降低，策略变短视 | 方差降低，更依赖 critic |

$\gamma\lambda$ 决定了一个 TD 误差能影响到多远之前的动作。权重 $(\gamma\lambda)^l$ 的总和是 $1/(1 - \gamma\lambda)$，可以把它当作"信用回溯的有效步数"。
取 PPO 论文在 MuJoCo 任务上用的 $\gamma = 0.99$、$\lambda = 0.95$（论文附录的超参数表）：

$$
\gamma\lambda = 0.9405, \qquad \frac{1}{1 - 0.9405} \approx 16.8
$$

一个动作的优势估计主要由它之后十几步的 TD 误差决定，更远的事交给 critic。

### 5.6 算法

```
算法：GAE（一段长度为 T 的轨迹片段，可以在回合中途截断）

输入：rewards[0..T−1]，values[0..T]（values[T] 是片段末尾状态的价值，用于自举），
      terminated[0..T−1]，done[0..T−1]（done = terminated 或 truncated）
 1. adv ← 0
 2. 对 t = T−1 … 0：
        δ ← rewards[t] + γ · values[t+1] · (1 − terminated[t]) − values[t]
        adv ← δ + γλ · (1 − done[t]) · adv
        advantages[t] ← adv
 3. value_targets ← advantages + values[0..T−1]                  就是 λ-return
输出：advantages，value_targets（都不带梯度）
```

第 2 步里有两个不同的掩码，对应第 4.5 节的表：

- `1 − terminated[t]`：真的终止了，下一状态的价值按 0 算
- `1 − done[t]`：只要回合结束了（不管是终止还是截断），递推就不能跨到下一个回合去，`adv` 要清零

被截断时 `values[t+1]` 必须是截断那一刻的下一个观测的价值，不是环境重置后新回合第一个观测的价值。
向量化环境会自动重置，这个观测要从 `info` 里取。很多实现图省事把截断当终止处理，这是一个已知的小偏差。

### 5.7 代码对应

对应 `04-gae-and-ppo.py`（讲义待写）的 `compute_gae()`。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `rewards`，`terminated`，`done` | `[T, N]` | `N` 个并行环境 |
| `values` | `[T + 1, N]` | 最后一行用于自举 |
| `advantages`，`value_targets` | `[T, N]` | 输出 |

讲义里先用第 5.4 节的数字做单元测试：输入 $(1, 0, 2)$ 和 $(2.0, 1.5, 1.8, 0)$，$\lambda = 0.8$ 时应得到 $(0.540, 0.264, 0.20)$。

同一个递推在 Phase 6 会再出现一次：Dreamer 在想象出的轨迹上算 λ-return，当作 critic 的目标。
区别只在于数据来源（想象的轨迹，而不是真实的轨迹），公式一样。

> 📌 待回填：$\lambda$ 取 0、0.8、0.95、1 时优势估计的方差（同一批数据上）和最终回报的对比，来自 `04-gae-and-ppo.py`。

### 5.8 常见误区

- **误区：GAE 是一种新的算法。**
  实际情况：它只是优势的一种估计办法，可以插进任何用到优势的策略梯度算法里。它和 λ-return 只差一个 $V(s_t)$。
- **误区：$\lambda$ 和 $\gamma$ 总是乘在一起，所以调哪个都一样。**
  实际情况：$\gamma$ 还出现在 $\delta_t$ 内部和 critic 的目标里，它改变的是问题的定义。
  $\lambda$ 只出现在加权里，critic 完全准确时调它不会改变优势的期望。
- **误区：$\lambda = 1$ 无偏，所以数据够多时应该用 $\lambda = 1$。**
  实际情况：片段在回合中途截断时，末尾仍然要用 critic 自举，并不是真正的蒙特卡洛。
  而且无偏的代价是方差，实践中 0.9 到 0.99 之间的值几乎总是更好。
- **误区：递推时遇到回合结束，把下一状态的价值清零就够了。**
  实际情况：还要把累计的 `adv` 清零。否则下一个回合的 TD 误差会顺着递推漏进上一个回合。

---

## 6. PPO

### 6.1 直觉

到上一节为止的算法都有一个浪费的地方：辛苦收集一批数据，只用来做一次梯度更新，然后扔掉。
原因在第 1.6 节说过：策略梯度是在当前策略的数据分布下的期望，参数一变，旧数据就不算数了。

能不能在同一批数据上多更新几次？直接这么做会出事。

在监督学习里，一步走大了，损失上升，下一步还能走回来，因为数据集没变。
强化学习里**数据是策略自己产生的**。一步走大了把策略弄坏，坏策略采回来的全是坏数据，
用坏数据算出的梯度没法把它带回去。这是一个不可逆的故障，回报曲线上表现为突然断崖式下跌，再也起不来。

工程上对付"改动可能不可逆地搞坏系统"的办法是限制每次变更的幅度：灰度发布，一次只放一小部分流量，
指标没问题再继续放。近端策略优化（Proximal Policy Optimization，PPO）做的是同一件事：
**允许在一批数据上反复更新，但新策略不能离采集这批数据的旧策略太远。**

它的前身 TRPO（Schulman et al. 2015）用 KL 散度的硬约束来限制步长，推导和实现都比较重，本课不讲。
PPO 用一个简单得多的办法达到类似的效果：直接在目标函数上动手，把"走太远"的收益裁掉。

### 6.2 定义

记采集数据时的策略参数为 $\phi_{\text{old}}$，正在优化的参数为 $\phi$。**概率比**是：

$$
\rho_t(\phi) = \frac{\pi_\phi(a_t \mid s_t)}{\pi_{\phi_{\text{old}}}(a_t \mid s_t)}
$$

$\rho_t > 1$ 表示新策略比旧策略更倾向于选 $a_t$，$\rho_t < 1$ 表示更不倾向。还没更新时 $\rho_t = 1$。

**不加限制的代理目标**（要最大化）：

$$
L^{\text{IS}}(\phi) = \mathbb{E}_t\big[ \rho_t(\phi) \, \hat{A}_t \big]
$$

**PPO 的裁剪目标**（要最大化）：

$$
L^{\text{CLIP}}(\phi) = \mathbb{E}_t\Big[ \min\big( \rho_t(\phi) \, \hat{A}_t, \;\; \mathrm{clip}\big( \rho_t(\phi), \, 1 - \epsilon_{\text{clip}}, \, 1 + \epsilon_{\text{clip}} \big) \, \hat{A}_t \big) \Big]
$$

其中 $\mathrm{clip}(x, lo, hi)$ 把 $x$ 限制在 $[lo, hi]$ 之内，$\hat{A}_t$ 用 GAE 计算。
PPO 论文（Schulman et al. 2017，arXiv:1707.06347）第 3 节给出这个目标，实验里 $\epsilon_{\text{clip}} = 0.2$ 的效果最好。

### 6.3 推导：裁剪目标是怎么来的

**第一步：用重要性采样复用旧数据。** 我们想估计新策略下的期望，但手上只有旧策略采的样本。
对任意函数 $h$，有恒等式：

$$
\begin{aligned}
\mathbb{E}_{a \sim \pi_\phi}\big[ h(a) \big]
&= \sum_a \pi_\phi(a) \, h(a) && \text{期望的定义} \\
&= \sum_a \pi_{\phi_{\text{old}}}(a) \, \frac{\pi_\phi(a)}{\pi_{\phi_{\text{old}}}(a)} \, h(a) && \text{乘以再除以 } \pi_{\phi_{\text{old}}}(a) \\
&= \mathbb{E}_{a \sim \pi_{\phi_{\text{old}}}}\big[ \rho(a) \, h(a) \big] && \text{期望的定义}
\end{aligned}
$$

用旧策略的样本，每个样本乘上概率比，就能估计新策略下的期望。这叫**重要性采样**（importance sampling）。
把它用在"动作的优势"上，就得到 $L^{\text{IS}}$：它估计的是"新策略选的动作，按旧策略的优势函数衡量，平均有多好"。

这里有一个近似：只修正了动作的分布，没有修正状态的分布（状态仍然是旧策略访问到的那些）。
新旧策略接近时这个近似是好的，离得远就不成立了。这是必须限制步长的理论原因。

**第二步：在 $\phi = \phi_{\text{old}}$ 处，$L^{\text{IS}}$ 的梯度就是策略梯度。** 对 $\rho_t$ 求梯度，分母不含 $\phi$：

$$
\nabla_\phi \rho_t(\phi) = \frac{\nabla_\phi \pi_\phi(a_t \mid s_t)}{\pi_{\phi_{\text{old}}}(a_t \mid s_t)}
= \frac{\pi_\phi(a_t \mid s_t)}{\pi_{\phi_{\text{old}}}(a_t \mid s_t)} \, \nabla_\phi \log \pi_\phi(a_t \mid s_t)
= \rho_t(\phi) \, \nabla_\phi \log \pi_\phi(a_t \mid s_t)
$$

第二个等号又是对数导数技巧。在 $\phi = \phi_{\text{old}}$ 处 $\rho_t = 1$，所以：

$$
\nabla_\phi L^{\text{IS}}(\phi) \Big\vert_{\phi = \phi_{\text{old}}} = \mathbb{E}_t\big[ \nabla_\phi \log \pi_\phi(a_t \mid s_t) \, \hat{A}_t \big]
$$

和第 3 节的策略梯度完全一样。所以第一次小批量更新时，PPO 和 A2C 走的是同一个方向。
区别出现在之后：$\phi$ 离开 $\phi_{\text{old}}$ 后，$L^{\text{IS}}$ 仍然给出有意义的梯度，A2C 的损失则不再正确。

**第三步：$L^{\text{IS}}$ 的问题。** 没有任何东西阻止优化器把 $\rho_t$ 推到很大。
某个动作的优势估计是正的，优化器就会一直提高它的概率，$\rho_t$ 从 1 涨到 5、涨到 50。
而 $\hat{A}_t$ 只是一个带噪声的估计，并且只在旧策略附近有效。这又是"对有噪声的估计用力做最大化"，
和 Phase 1 第 8.5 节的模型被钻空子是同一个毛病。

**第四步：裁剪。分情况看 $\min$ 取的是哪一项。**

| 优势 | 概率比 | 未裁剪项 $\rho \hat{A}$ 与裁剪项的大小关系 | $\min$ 取哪个 | 对 $\phi$ 有梯度吗 | 含义 |
| --- | --- | --- | --- | --- | --- |
| $\hat{A} > 0$ | $\rho > 1 + \epsilon_{\text{clip}}$ | 未裁剪项更大 | 裁剪项（常数） | 没有 | 好动作的概率已经涨够了，不再奖励 |
| $\hat{A} > 0$ | $\rho < 1 - \epsilon_{\text{clip}}$ | 未裁剪项更小 | 未裁剪项 | 有 | 好动作的概率反而降了，要纠正 |
| $\hat{A} < 0$ | $\rho < 1 - \epsilon_{\text{clip}}$ | 未裁剪项更大（负得少） | 裁剪项（常数） | 没有 | 坏动作的概率已经降够了，不再奖励 |
| $\hat{A} < 0$ | $\rho > 1 + \epsilon_{\text{clip}}$ | 未裁剪项更小（负得多） | 未裁剪项 | 有 | 坏动作的概率反而涨了，要纠正 |
| 任意 | 在 $[1 - \epsilon_{\text{clip}}, 1 + \epsilon_{\text{clip}}]$ 内 | 相等 | 两者相同 | 有 | 正常的策略梯度 |

规律是：**朝着"让目标变大"的方向走出了范围，梯度归零；朝着"让目标变小"的方向走出了范围，梯度保留。**
取 $\min$ 的效果是让目标成为一个悲观的估计：只把走远的好处裁掉，走远的坏处照单全收。
这样优化器没有动力把策略推离旧策略太远，但如果已经走错了，还能被拉回来。

```
   Â > 0 时，单个样本的目标随 ρ 的变化          Â < 0 时

   目标                                        目标
    │            ┌──────────── 封顶              │ ────────┐
    │           ╱                               │         ╲
    │          ╱                                │          ╲
    │         ╱                                 │           ╲
    │        ╱                                  │            ╲
    └───────┼───┼───┼──────▶ ρ                  └────┼───┼───┼─╲────▶ ρ
           1−ε  1  1+ε                              1−ε  1  1+ε
```

### 6.4 手算例子

取 $\epsilon_{\text{clip}} = 0.2$，裁剪范围是 $[0.8, 1.2]$。五个样本：

| 样本 | $\hat{A}$ | $\rho$ | 未裁剪项 $\rho \hat{A}$ | 裁剪项 $\mathrm{clip}(\rho) \hat{A}$ | $\min$ | 梯度 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | $+2$ | 1.5 | 3.0 | $1.2 \times 2 = 2.4$ | 2.4 | 无 |
| 2 | $+2$ | 0.5 | 1.0 | $0.8 \times 2 = 1.6$ | 1.0 | 有，推高 $\rho$ |
| 3 | $-2$ | 0.5 | $-1.0$ | $0.8 \times (-2) = -1.6$ | $-1.6$ | 无 |
| 4 | $-2$ | 1.5 | $-3.0$ | $1.2 \times (-2) = -2.4$ | $-3.0$ | 有，压低 $\rho$ |
| 5 | $+2$ | 1.1 | 2.2 | $1.1 \times 2 = 2.2$ | 2.2 | 有 |

这批样本的裁剪目标是五个 $\min$ 的平均：$(2.4 + 1.0 - 1.6 - 3.0 + 2.2) / 5 = 0.2$。
不裁剪的目标是 $(3.0 + 1.0 - 1.0 - 3.0 + 2.2) / 5 = 0.44$。裁剪后的值总是不大于不裁剪的值。

样本 1 和样本 3 不再贡献梯度。训练时统计"被裁剪的样本占比"（clip fraction）是一个有用的诊断量：
接近 0 说明更新太保守，占了一大半说明每轮走得太猛。

### 6.5 完整的损失与算法

PPO 把三项合成一个损失，和第 4.2 节的结构相同，只是策略项换成了裁剪目标（论文第 5 节）：

$$
\mathcal{L}(\phi, \psi) = -L^{\text{CLIP}}(\phi) + c_v \, \mathbb{E}_t\big[ (V_\psi(s_t) - \hat{R}_t)^2 \big] - c_{\text{ent}} \, \mathbb{E}_t\big[ \mathcal{H}(\pi_\phi(\cdot \mid s_t)) \big]
$$

其中 $\hat{R}_t = \hat{A}_t + V_{\psi_{\text{old}}}(s_t)$ 是第 5 节的 λ-return。

```
算法：PPO-clip

输入：actor π_φ，critic V_ψ，N 个并行环境，每轮收集 T 步，每批数据训练 K 个 epoch
重复：
 1. 用当前策略收集 T × N 步数据；同时记下 log π_old(a_t | s_t) 和 V_old(s_t)      不带梯度
 2. 用 GAE 算 Â_t，价值目标 R̂_t = Â_t + V_old(s_t)                              只算一次
 3. 重复 K 个 epoch：
        把 T × N 个样本打乱，切成若干个小批量
        对每个小批量：
            （可选）把这个小批量里的 Â 标准化
            重新前向计算 log π_φ(a_t | s_t)、V_ψ(s_t)、熵
            ρ = exp( log π_φ − log π_old )
            L_clip = mean( min( ρ · Â, clip(ρ, 1−ε, 1+ε) · Â ) )
            loss = −L_clip + c_v · mean( (V_ψ − R̂)² ) − c_ent · mean(熵)
            反向传播，梯度裁剪，更新 φ 和 ψ
        （可选）若近似 KL 超过阈值，提前结束这一轮
 4. 丢弃这批数据
```

几个细节：

- **概率比在对数空间里算。** $\rho = \exp(\log \pi_\phi - \log \pi_{\text{old}})$，避免两个很小的概率直接相除。
- **$\log \pi_{\text{old}}$ 和优势在收集数据时算一次，之后不变。** 它们是常数。
  每个 epoch 重新算的只有新策略的 $\log \pi_\phi$。
- **近似 KL。** 用 $\mathbb{E}_t[\log \pi_{\text{old}} - \log \pi_\phi]$ 估计新旧策略的 KL 散度，
  作为"走了多远"的监控量。它不出现在损失里，只用来报警或者提前停止。

PPO 论文附录给出的 MuJoCo 任务超参数：每轮 2048 步，Adam 学习率 $3 \times 10^{-4}$，10 个 epoch，小批量 64，
$\gamma = 0.99$，$\lambda = 0.95$。

PPO 以"实现细节决定成败"出名。除了上面的核心，常见实现还带着一串小技巧：观测标准化、奖励缩放、
正交初始化、学习率线性衰减、梯度裁剪、优势标准化。Engstrom et al. 2020（Implementation Matters in Deep Policy Gradients）
和 Andrychowicz et al. 2020（What Matters in On-Policy Reinforcement Learning）系统地研究过这些细节的影响，
结论是其中一些的作用不亚于算法本身。写自己的 PPO 时对不上别人的分数，多半差在这些地方。

### 6.6 代码对应

对应 `04-gae-and-ppo.py`（讲义待写）。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| 收集阶段的缓冲 | `[T, N, …]` | 观测、动作、`log_probs_old`、`values_old`、奖励、结束标志 |
| 展平后的数据 | `[T × N, …]` | 算完 GAE 之后才能展平打乱，之前要保持时间顺序 |
| 小批量 | `[B, …]` | 从展平的数据里取 |
| `ratio` | `[B]` | $\rho$ |
| `advantages` | `[B]` | 不带梯度 |

`ppo_clip_loss(log_probs, log_probs_old, advantages, clip_eps)` 只有四行：

```python
ratio = torch.exp(log_probs - log_probs_old)                      # [B]
unclipped = ratio * advantages
clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
loss = -torch.min(unclipped, clipped).mean()
```

讲义用第 6.4 节的五个样本做单元测试，期望的输出是 $-0.2$。

PPO 在本课的角色是无模型方法的基线：它稳定、好调，但费样本。
Phase 6 的学习检查里有一条"自己的 Dreamer 在同等环境步数下样本效率优于自己写的 PPO"，比的就是这一节的实现。

> 📌 待回填：PPO 在 `CartPole-v1` 与 `Pendulum-v1` 上的学习曲线；每批数据训练的 epoch 数取 1、4、10 时的对比；去掉裁剪后的表现，来自 `04-gae-and-ppo.py`。定性预期：去掉裁剪并增加 epoch 数时，训练更容易崩溃。

### 6.7 常见误区

- **误区：裁剪保证了新策略和旧策略的概率比不会超出 $[1 - \epsilon_{\text{clip}}, 1 + \epsilon_{\text{clip}}]$。**
  实际情况：裁剪只是让超出范围的样本不再提供"继续往外走"的梯度。网络的参数是共享的，
  其他样本的梯度照样可能把这个样本的概率比推得更远。所以还需要监控近似 KL，限制 epoch 数。
- **误区：PPO 在一批数据上更新很多次，所以它是离策略算法。**
  实际情况：它只能复用**当前这一轮**刚采的数据，而且只在旧策略附近有效。更早的数据仍然要扔掉。
  它是在策略算法，只是比 A2C 多榨了几次。真正的离策略算法见第 8、9 节。
- **误区：代理目标 $L^{\text{CLIP}}$ 的值在上升，说明策略在变好。**
  实际情况：每一轮开始时 $\rho = 1$，优势的批内平均接近零，目标值也接近零；每一轮它都会从零附近升上去一点。
  这个值在不同轮之间没有可比性。看回报。
- **误区：PPO 样本效率低，所以在机器人上没用。**
  实际情况：在仿真里，样本几乎免费，大规模并行仿真加 PPO 是目前训练机器人运动策略的常用路线，
  墙上时间反而短。样本效率只在真实交互很贵时才是主要矛盾，那正是世界模型发挥作用的地方。

---

## 7. 三种梯度估计器

### 7.1 直觉

把前六节做的事抽象出来，其实一直在解同一个问题：

$$
\nabla_\phi \, \mathbb{E}_{x \sim p_\phi}\big[ f(x) \big] = \; ?
$$

一个随机变量 $x$ 从带参数的分布里采出来，送进函数 $f$ 得到一个分数。问：分数的期望对分布的参数的梯度是多少。
在策略梯度里，$x$ 是动作（或整条轨迹），$p_\phi$ 是策略，$f$ 是回报。

困难在于采样这一步。反向传播沿着计算图把梯度一层层传回去，而"从分布里抽一个样本"不是一个可微的运算：
样本的值和参数之间没有确定的函数关系。计算图在这里断开了。

```
      φ ──▶ [ 分布 p_φ ] ══ 采样 ══▶ x ──▶ [ f ] ──▶ 分数
                             ╳
                   梯度从分数传回来，走到这里过不去
```

有三种办法绕过去。它们在后面的章节里都会用到，所以这一节把它们放在一起比。

**办法一：score-function 估计器。** 不让梯度穿过采样。把 $f$ 当成黑盒，只看它的输出值，
然后问"哪些样本得分高，就提高哪些样本的概率"。这是 A/B 实验的思路：不需要知道系统内部怎么运作，
只需要随机尝试、观察结果、做相关性分析。前六节用的就是它。

**办法二：pathwise 估计器**（也叫重参数化）。把随机性从分布里"搬出来"：
先抽一个和参数无关的噪声 $\epsilon$，再用一个确定的、可微的函数把它变成 $x$。
这样从 $\phi$ 到 $x$ 就有了一条可微的路径，梯度可以直接穿过 $f$ 传回来。
这是白盒的思路：手上有 $f$ 的"源码"，可以直接问"$x$ 往哪个方向动一点，分数会变高"。
旧课的 VAE 和本课 Phase 2、3 用的就是它。

**办法三：straight-through 估计器。** 专门对付离散变量。离散的 $x$ 没法"动一点"，重参数化用不上。
straight-through 的做法是：前向计算老老实实用离散的样本，反向传播时假装采样这一步是个恒等映射，
让梯度直接通过。这是一个明知不对但管用的近似。

```
  score-function                 pathwise                       straight-through

  φ ─▶ p_φ ═采样═▶ x ─▶ f        ε ──┐                          φ ─▶ 概率 ═采样═▶ x(one-hot) ─▶ f
        ▲          ╳    │        φ ─▶ x = g_φ(ε) ─▶ f           ▲      ▲                      │
        │               │        ▲        ▲          │           │      └── 反向：当作恒等 ◀────┘
        └─ f 的值 × ∇log p ◀─┘    └────────┴── ∇f ◀──┘           └───────────────
  梯度不穿过 f，f 可以是黑盒       梯度穿过 f，f 必须可微          前向离散，反向连续；f 必须可微
```

### 7.2 定义

**score-function 估计器**（SF，又叫 REINFORCE 估计器、似然比估计器）：

$$
\nabla_\phi \, \mathbb{E}_{x \sim p_\phi}\big[ f(x) \big] = \mathbb{E}_{x \sim p_\phi}\big[ f(x) \, \nabla_\phi \log p_\phi(x) \big]
$$

推导就是第 1.3 节的第一到第三步，把轨迹换成 $x$、回报换成 $f$。

**pathwise 估计器**（PW）。假设 $x$ 可以写成 $x = g_\phi(\epsilon)$，其中 $\epsilon$ 服从一个不含参数的固定分布。
对高斯分布 $\mathcal{N}(\mu, \sigma^2)$，就是 $x = \mu + \sigma \epsilon$，$\epsilon \sim \mathcal{N}(0, 1)$。于是：

$$
\begin{aligned}
\nabla_\phi \, \mathbb{E}_{x \sim p_\phi}\big[ f(x) \big]
&= \nabla_\phi \, \mathbb{E}_{\epsilon}\big[ f(g_\phi(\epsilon)) \big] && \text{换元：期望改成对 } \epsilon \text{ 取} \\
&= \mathbb{E}_{\epsilon}\big[ \nabla_\phi f(g_\phi(\epsilon)) \big] && \epsilon \text{ 的分布不含 } \phi \text{，梯度可以移进期望} \\
&= \mathbb{E}_{\epsilon}\big[ f'(g_\phi(\epsilon)) \, \nabla_\phi g_\phi(\epsilon) \big] && \text{链式法则}
\end{aligned}
$$

对比第 1.3 节：那里梯度移不进期望，是因为分布含参数。重参数化把参数从分布里挪到了被积函数里，问题就变回了监督学习的形式。

**straight-through 估计器**（ST）。$x$ 是离散的，比如从概率向量 $\pi$ 里抽出的 one-hot 向量。
前向用样本，反向把 $\partial x / \partial \pi$ 当成单位矩阵：

$$
\hat{g}_{\text{ST}} = \nabla_x f(x) \cdot \underbrace{\frac{\partial x}{\partial \pi}}_{\text{定义为 } I} \cdot \frac{\partial \pi}{\partial \phi}
$$

实现只要一行：

```python
x = one_hot + probs - probs.detach()
```

前向时后两项的数值抵消，`x` 的值恰好是 `one_hot`；反向时 `one_hot` 和 `probs.detach()` 没有梯度，
只有中间的 `probs` 有，所以梯度就像 `x = probs` 那样传回去。

三者的适用条件：

| | score-function | pathwise | straight-through |
| --- | --- | --- | --- |
| 对 $f$ 的要求 | 能求值就行，可以是黑盒、不连续 | 必须对 $x$ 可微，且能拿到梯度 | 必须对 $x$ 可微（把离散的 $x$ 当成实数向量） |
| 对分布的要求 | $\log p_\phi(x)$ 对 $\phi$ 可微 | 能重参数化（连续分布） | 离散分布 |
| $x$ 的类型 | 离散、连续都行 | 连续 | 离散 |
| 偏差 | 无 | 无 | 有 |
| 方差 | 大 | 通常小 | 小 |

### 7.3 推导：在一维高斯上把两种估计器的方差算出来

取一个能算到底的例子。$x \sim \mathcal{N}(\mu, \sigma^2)$，$f(x) = x^2$，求对均值 $\mu$ 的梯度。

**标准答案。** $\mathbb{E}[x^2] = \mu^2 + \sigma^2$（二阶矩等于均值的平方加方差），所以：

$$
\frac{\partial}{\partial \mu} \mathbb{E}[x^2] = 2\mu
$$

下面统一用 $x = \mu + \sigma \epsilon$ 代换，把所有期望化成标准正态分布的矩。需要的几个矩：

$$
\mathbb{E}[\epsilon] = 0, \quad \mathbb{E}[\epsilon^2] = 1, \quad \mathbb{E}[\epsilon^3] = 0, \quad \mathbb{E}[\epsilon^4] = 3, \quad \mathbb{E}[\epsilon^5] = 0, \quad \mathbb{E}[\epsilon^6] = 15
$$

**score-function 估计器。** 第 1.4 节给过高斯分布的得分函数 $\partial \log p / \partial \mu = (x - \mu)/\sigma^2 = \epsilon / \sigma$。单样本估计是：

$$
\hat{g}_{\text{SF}} = f(x) \cdot \frac{x - \mu}{\sigma^2} = \frac{(\mu + \sigma\epsilon)^2 \, \epsilon}{\sigma}
$$

*均值。* 展开 $(\mu + \sigma\epsilon)^2 \epsilon = \mu^2 \epsilon + 2\mu\sigma \epsilon^2 + \sigma^2 \epsilon^3$，逐项取期望：

$$
\mathbb{E}[\hat{g}_{\text{SF}}] = \frac{\mu^2 \cdot 0 + 2\mu\sigma \cdot 1 + \sigma^2 \cdot 0}{\sigma} = 2\mu
$$

无偏。

*二阶矩。* $\hat{g}_{\text{SF}}^2 = (\mu + \sigma\epsilon)^4 \epsilon^2 / \sigma^2$。用二项式定理展开四次方，再乘上 $\epsilon^2$：

$$
(\mu + \sigma\epsilon)^4 \epsilon^2 = \mu^4 \epsilon^2 + 4\mu^3\sigma \epsilon^3 + 6\mu^2\sigma^2 \epsilon^4 + 4\mu\sigma^3 \epsilon^5 + \sigma^4 \epsilon^6
$$

逐项取期望，奇数次的项为零：

$$
\mathbb{E}\big[ (\mu + \sigma\epsilon)^4 \epsilon^2 \big] = \mu^4 \cdot 1 + 6\mu^2\sigma^2 \cdot 3 + \sigma^4 \cdot 15 = \mu^4 + 18\mu^2\sigma^2 + 15\sigma^4
$$

除以 $\sigma^2$：

$$
\mathbb{E}[\hat{g}_{\text{SF}}^2] = \frac{\mu^4}{\sigma^2} + 18\mu^2 + 15\sigma^2
$$

*方差。* 二阶矩减去均值的平方 $(2\mu)^2 = 4\mu^2$：

$$
\mathrm{Var}[\hat{g}_{\text{SF}}] = \frac{\mu^4}{\sigma^2} + 14\mu^2 + 15\sigma^2
$$

**pathwise 估计器。** $x = \mu + \sigma\epsilon$，所以 $\partial x / \partial \mu = 1$。$f'(x) = 2x$。单样本估计是：

$$
\hat{g}_{\text{PW}} = f'(x) \cdot \frac{\partial x}{\partial \mu} = 2x = 2\mu + 2\sigma\epsilon
$$

*均值。* $\mathbb{E}[2\mu + 2\sigma\epsilon] = 2\mu$。无偏。

*方差。* $2\mu$ 是常数，只有 $2\sigma\epsilon$ 贡献方差：

$$
\mathrm{Var}[\hat{g}_{\text{PW}}] = (2\sigma)^2 \, \mathrm{Var}[\epsilon] = 4\sigma^2
$$

**对比。**

$$
\mathrm{Var}[\hat{g}_{\text{SF}}] = \frac{\mu^4}{\sigma^2} + 14\mu^2 + 15\sigma^2
\qquad\qquad
\mathrm{Var}[\hat{g}_{\text{PW}}] = 4\sigma^2
$$

代入几组数：

| $\mu$ | $\sigma$ | 真实梯度 $2\mu$ | SF 的方差 | PW 的方差 | 比值 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 2 | $1 + 14 + 15 = 30$ | 4 | 7.5 |
| 1 | 0.1 | 2 | $100 + 14 + 0.15 = 114.15$ | 0.04 | 约 2850 |
| 3 | 1 | 6 | $81 + 126 + 15 = 222$ | 4 | 55.5 |
| 3 | 0.1 | 6 | $8100 + 126 + 0.15 = 8226.15$ | 0.04 | 约 20 万 |

读这张表：

1. **$\sigma \to 0$ 时两者走向相反。** 分布越集中，pathwise 的方差越小，趋于零；
   score-function 的方差因为 $\mu^4 / \sigma^2$ 这一项反而爆炸。
   策略训练到后期越来越接近确定，正是 score-function 最吃亏的时候。
2. **score-function 的方差和 $f$ 的绝对大小有关。** $\mu$ 从 1 变到 3，$f$ 的典型值从 1 变到 9，方差涨了几十倍。
   这是第 2.3 节"来源一"的连续版本。pathwise 的方差只和 $f$ 的斜率有关，和 $f$ 的绝对大小无关。

**单个样本长什么样。** 取 $\mu = 1$，$\sigma = 1$，真实梯度是 2。三个不同的噪声：

| $\epsilon$ | $x$ | $\hat{g}_{\text{SF}} = x^2 \epsilon$ | $\hat{g}_{\text{PW}} = 2x$ |
| --- | --- | --- | --- |
| 0.5 | 1.5 | $2.25 \times 0.5 = 1.125$ | 3.0 |
| −1.0 | 0.0 | $0 \times (-1) = 0$ | 0.0 |
| 2.0 | 3.0 | $9 \times 2 = 18$ | 6.0 |

pathwise 的估计在真实值附近摆动；score-function 的估计一个样本就可能偏出去一个数量级。

**方差为什么差这么多。** pathwise 用到了 $f$ 的导数，它直接知道"$x$ 往右一点，$f$ 增加多少"。
score-function 只知道 $f$ 的值，它得从"碰巧往右偏的样本得分高一些"这种相关性里把斜率反推出来。
这相当于用随机扰动做有限差分，扰动的幅度是 $\sigma$，所以结果里要除以 $\sigma$，扰动越小，噪声放大得越厉害。

**加上基线能救回多少。** 给 score-function 估计器减去基线 $b = \mathbb{E}[f] = \mu^2 + \sigma^2$：

$$
\hat{g}_{\text{SF+b}} = \frac{\big( (\mu + \sigma\epsilon)^2 - \mu^2 - \sigma^2 \big) \, \epsilon}{\sigma}
= \frac{(2\mu\sigma\epsilon + \sigma^2\epsilon^2 - \sigma^2) \, \epsilon}{\sigma}
= 2\mu\epsilon^2 + \sigma(\epsilon^3 - \epsilon)
$$

均值是 $2\mu \cdot 1 + \sigma(0 - 0) = 2\mu$，仍然无偏。二阶矩逐项算：

$$
\begin{aligned}
\mathbb{E}\big[ (2\mu\epsilon^2)^2 \big] &= 4\mu^2 \, \mathbb{E}[\epsilon^4] = 12\mu^2 \\
\mathbb{E}\big[ 2 \cdot 2\mu\epsilon^2 \cdot \sigma(\epsilon^3 - \epsilon) \big] &= 4\mu\sigma \big( \mathbb{E}[\epsilon^5] - \mathbb{E}[\epsilon^3] \big) = 0 \\
\mathbb{E}\big[ \sigma^2 (\epsilon^3 - \epsilon)^2 \big] &= \sigma^2 \big( \mathbb{E}[\epsilon^6] - 2\mathbb{E}[\epsilon^4] + \mathbb{E}[\epsilon^2] \big) = \sigma^2 (15 - 6 + 1) = 10\sigma^2
\end{aligned}
$$

三项相加再减去均值的平方：

$$
\mathrm{Var}[\hat{g}_{\text{SF+b}}] = 12\mu^2 + 10\sigma^2 - 4\mu^2 = 8\mu^2 + 10\sigma^2
$$

| $\mu = 1$ | 不带基线的 SF | 带基线的 SF | PW |
| --- | --- | --- | --- |
| $\sigma = 1$ | 30 | 18 | 4 |
| $\sigma = 0.1$ | 114.15 | 8.1 | 0.04 |

基线消掉了 $\mu^4 / \sigma^2$ 这个会爆炸的项，这是很大的改进。但它留下了一个不随 $\sigma$ 减小的底 $8\mu^2$。
pathwise 没有这个底。结论是：**基线能解决"回报的绝对水平"带来的方差，解决不了"只看函数值、不看导数"带来的方差。**

### 7.4 pathwise 不是永远更好

上面的例子里 $f$ 是一条光滑的抛物线，对 pathwise 很有利。它的方差由 $f'$ 的大小决定：

$$
\mathrm{Var}[\hat{g}_{\text{PW}}] = \mathrm{Var}\big[ f'(x) \big] \quad \text{（对均值求梯度时）}
$$

如果 $f$ 的值变化平缓但导数剧烈震荡，情况会反过来。在 $f(x) = x^2$ 上叠加一个幅度很小的高频波纹：

$$
f(x) = x^2 + 0.1 \sin(100x), \qquad f'(x) = 2x + 10\cos(100x)
$$

函数值最多变了 0.1，score-function 估计器几乎不受影响。导数却多出一项幅度为 10 的震荡。
$\sigma = 1$ 时这一项在采样范围内扫过很多个周期，$\cos^2$ 的平均值是 $1/2$，所以它贡献的方差约为 $10^2 / 2 = 50$。
pathwise 的方差从 4 涨到 54 左右，超过了带基线的 score-function（18）。

这不是人造的刁钻例子。把一个模型展开很多步，回报对早期动作的导数就是许多雅可比矩阵的连乘，
在放大扰动的系统里它会变得又大又乱，第 11 节会具体讨论。
Mohamed et al. 2020（arXiv:1906.10652）对两种估计器各自适用的条件有系统的讨论。

不可微的 $f$ 则根本用不了 pathwise：稀疏的 0/1 奖励、真实环境、带接触的物理仿真器都属于这一类。

### 7.5 straight-through 的偏差

用一个伯努利变量看清 straight-through 做了什么。$x \in \lbrace 0, 1 \rbrace$，取 1 的概率是 $p$，参数就是 $p$ 本身。

**标准答案。** $\mathbb{E}[f(x)] = p \, f(1) + (1 - p) \, f(0)$，所以：

$$
\frac{\partial}{\partial p} \mathbb{E}[f(x)] = f(1) - f(0)
$$

真实梯度是 $f$ 在两个离散点之间的**差**。

**straight-through 的估计。** 把 $f$ 当成实数上的可微函数，在采到的 $x$ 处求导，再把 $\partial x / \partial p$ 当作 1：

$$
\hat{g}_{\text{ST}} = f'(x), \qquad \mathbb{E}[\hat{g}_{\text{ST}}] = p \, f'(1) + (1 - p) \, f'(0)
$$

它用 $f$ 在端点处的**导数**去代替两点之间的差。两者一般不相等，所以有偏。

什么时候恰好相等？$f$ 是线性函数 $f(x) = cx + d$ 时，导数处处是 $c$，差也是 $c$，无偏。
$f$ 弯得越厉害，偏差越大。

**手算例子。** 取 $f(x) = x^2 + 10$。真实梯度是 $f(1) - f(0) = 11 - 10 = 1$，和 $p$ 无关。

*straight-through：* $f'(x) = 2x$。采到 1 时估计是 2，采到 0 时估计是 0。
均值是 $2p$，方差是 $4p(1 - p)$（取值为 0 和 2 的两点分布）。

*score-function：* 伯努利分布的得分函数是 $\partial \log p(x) / \partial p$，采到 1 时为 $1/p$，采到 0 时为 $-1/(1 - p)$。
估计值是 $f(1)/p = 11/p$ 或 $-f(0)/(1 - p) = -10/(1 - p)$。
均值是 $p \cdot 11/p - (1 - p) \cdot 10/(1 - p) = 1$，无偏。
二阶矩是 $p \cdot (11/p)^2 + (1 - p) \cdot (10/(1 - p))^2 = 121/p + 100/(1 - p)$。

| $p$ | 真实梯度 | ST 的均值 | ST 的偏差 | ST 的方差 | SF 的均值 | SF 的方差 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 1 | 1.0 | 0 | 1.0 | 1 | $242 + 200 - 1 = 441$ |
| 0.2 | 1 | 0.4 | $-0.6$ | 0.64 | 1 | $605 + 125 - 1 = 729$ |
| 0.8 | 1 | 1.6 | $+0.6$ | 0.64 | 1 | $151.25 + 500 - 1 = 650.25$ |

（$p = 0.5$ 时 straight-through 恰好无偏，是二次函数的巧合：抛物线两端导数的平均等于割线的斜率。）

straight-through 的梯度大小不对，但符号对了，方差比 score-function 小两到三个数量级，而且不受 $f$ 里那个常数 10 的影响。
用一个方向大致正确、噪声很小的梯度，往往比用一个期望正确、噪声极大的梯度学得快。这就是它被广泛使用的原因。

**类别分布的版本。** 实际使用时变量是 $K$ 类的类别分布，参数是 logits：

```
前向：probs = softmax(logits)                      形状 [B, K]
      index ~ Categorical(probs)                   形状 [B]
      one_hot = onehot(index)                      形状 [B, K]，数值是 0 和 1
      x = one_hot + probs − sg(probs)              数值等于 one_hot
反向：∂x/∂logits 按 ∂probs/∂logits 来算
```

Dreamer V2 起，世界模型的随机隐变量从高斯换成了一组类别变量，采样这一步用的就是这一行
（出处是 Bengio et al. 2013，arXiv:1308.3432）。Phase 6 会说明为什么要换，以及梯度穿过它之后流向哪里。

**另一种选择：Gumbel-Softmax**（Jang et al. 2017，arXiv:1611.01144）。它不输出 one-hot，而是输出一个"软"的近似：
给 logits 加上 Gumbel 噪声后过一个带温度的 softmax。温度高时输出接近均匀，温度趋于零时趋近 one-hot。
它把离散采样变成了一个可以重参数化的连续运算，代价是下游拿到的不是真正的离散值，并且同样有偏。本课不用，了解即可。

### 7.6 三种估计器在本课的位置

| 估计器 | 本章谁在用 | 后面谁在用 |
| --- | --- | --- |
| score-function | REINFORCE、A2C、PPO（第 1–6 节） | Dreamer V2、V3 训练离散动作的 actor |
| pathwise | DDPG、TD3、SAC 的 actor 更新（第 8、9 节）；沿模型反向传播（第 11 节） | 所有用重参数化的隐变量模型（VAE、RSSM）；Dreamer V1 的 actor；TD-MPC2 的策略先验 |
| straight-through | 本节 | Dreamer V2、V3 的类别隐变量；离散动作要沿模型反向传播时也靠它让梯度穿过动作采样；Phase 8 的 VQ-VAE |

Dreamer 各个版本、各种动作类型具体选了哪种估计器，Phase 6 会逐一说明。这里只需要记住选择的依据：

```
  变量是连续的，并且下游的函数可微吗？
        │
        ├── 是 ──▶ pathwise（方差小，无偏）
        │
        └── 否 ──▶ 变量是离散的，下游可微，能接受一点偏差吗？
                     │
                     ├── 是 ──▶ straight-through（方差小，有偏）
                     │
                     └── 否 ──▶ score-function + 基线（无偏，方差大，什么情况都能用）
```

### 7.7 算法

三种估计器在 PyTorch 里各是几行。以"对分布的参数求 $\mathbb{E}[f(x)]$ 的梯度"为例：

```python
# score-function：采样不带梯度，用 log_prob 加权
dist = Normal(mu, sigma)
x = dist.sample()                                   # 不带梯度
surrogate = (dist.log_prob(x) * (f(x) - baseline)).mean()    # f(x) 当常数
surrogate.backward()

# pathwise：采样带梯度，直接对 f 求导
x = dist.rsample()                                  # 等价于 mu + sigma * randn_like(mu)
f(x).mean().backward()

# straight-through：前向 one-hot，反向走 probs
probs = logits.softmax(-1)
one_hot = F.one_hot(Categorical(probs=probs).sample(), K).float()
x = one_hot + probs - probs.detach()
f(x).mean().backward()
```

`sample()` 和 `rsample()` 只差一个字母，含义完全不同：前者把样本从计算图里断开，后者保留路径。
该用 `rsample()` 的地方用了 `sample()`，程序不会报错，只是那部分梯度悄悄变成了零。

### 7.8 代码对应

对应 `05-gradient-estimators.py`（讲义待写）。这一节讲义不涉及强化学习，只做数值实验：

| 函数 | 做什么 | 关键形状 |
| --- | --- | --- |
| `score_function_grad(mu, sigma, f, n, baseline)` | 返回 `n` 个单样本梯度估计 | `[n]` |
| `pathwise_grad(mu, sigma, f, n)` | 同上 | `[n]` |
| `straight_through_sample(logits)` | 返回前向为 one-hot、反向走概率的样本 | 输入 `[B, K]`，输出 `[B, K]` |

实验一用 $f(x) = x^2$ 验证第 7.3 节的公式：在 $\mu = 1$、$\sigma = 1$ 处，三种估计（不带基线的 SF、带基线的 SF、PW）
的样本均值都应接近 2，样本方差应分别接近 30、18、4。再扫一遍 $\sigma$，画出方差随 $\sigma$ 变化的曲线。
实验二换成第 7.4 节带高频波纹的 $f$，看两者的排名反转。
实验三在伯努利变量上对比 SF 和 ST 的偏差与方差，对照第 7.5 节的表。

> 📌 待回填：三种估计器的样本均值与样本方差（对照公式给出的 30、18、4），方差随 $\sigma$ 变化的曲线，以及伯努利实验里 ST 的偏差，来自 `05-gradient-estimators.py`。

### 7.9 常见误区

- **误区：重参数化是 VAE 的技巧，和强化学习无关。**
  实际情况：它是一个通用的梯度估计器。SAC 的 actor 更新、Dreamer V1 的 actor 更新用的都是它，
  和 VAE 里是同一个公式，只是 $f$ 从重建损失换成了价值。
- **误区：pathwise 估计器总是比 score-function 方差小。**
  实际情况：取决于 $f$ 的导数。$f$ 光滑时 pathwise 好得多；$f$ 的导数剧烈变化（长时间的模型展开、混沌的动力学）时可能更差；
  $f$ 不可微时根本不能用。
- **误区：straight-through 有偏，所以是个不严谨的权宜之计，能不用就不用。**
  实际情况：估计器好不好要看偏差和方差的总和。在高维的离散隐变量上，
  无偏的 score-function 估计器方差大到没法训练，有偏但低方差的 straight-through 是实际能用的选择。
- **误区：score-function 估计器的方差大，是因为环境有随机性。**
  实际情况：第 7.3 节的例子里 $f$ 是完全确定的，方差全部来自估计器本身：它只用函数值，不用导数。
  环境的随机性会让情况更糟，但不是根源。

---

## 8. 确定性策略梯度：DDPG 与 TD3

### 8.1 直觉

前六节的算法都是在策略的：数据用一轮就扔。Phase 4 的 DQN 不是这样，它把经验存进回放池反复使用，样本效率高得多。
能不能把 DQN 搬到连续动作上？

DQN 卡在一步上：算目标和选动作都要 $\max_a Q(s, a)$。离散动作时把每个动作的 $Q$ 值都算出来挑最大的就行。
连续动作有无穷多个，$\max_a$ 本身成了一个优化问题，每次都解一遍太慢。

解决办法是**把求最大值这件事也交给一个网络**：训练一个 actor $\mu_\phi(s)$，让它直接输出"使 $Q$ 最大的那个动作"。
这是工程里常见的"把一个反复求解的优化问题摊销成一次查询"：不是每个请求都跑一遍优化器，而是训练一个函数去预测优化器的答案。

怎么训练这个 actor？critic $Q_\psi(s, a)$ 是一个神经网络，它对输入的动作 $a$ 是可微的。
所以可以直接问 critic："动作往哪个方向挪一点，$Q$ 会变大？"然后让 actor 的输出往那个方向挪。

```
          ┌────────────┐   a = μ_φ(s)   ┌────────────┐
   s ───▶ │ actor μ_φ  │ ─────────────▶ │ critic Q_ψ │ ───▶ Q(s, a)
          └────────────┘                └────────────┘
                ▲                             │
                └──────── ∂Q/∂a · ∂a/∂φ ◀─────┘
                    梯度穿过 critic 流回 actor
```

用第 7 节的话说：这是 **pathwise 估计器**，被求导的函数 $f$ 就是 critic。
critic 在这里扮演的角色是一个"学出来的、可微的回报函数"。环境不可微，但 critic 可微。

### 8.2 定义

**确定性策略。** $a = \mu_\phi(s)$，同一个状态永远输出同一个动作，没有采样。

**actor 的目标**（要最大化）。状态从回放池 $\mathcal{D}$ 里取：

$$
J(\phi) = \mathbb{E}_{s \sim \mathcal{D}}\big[ Q_\psi(s, \mu_\phi(s)) \big]
$$

**critic 的目标。** 和 DQN 一样的 TD 回归，只是 $\max_a$ 被目标 actor 的输出代替：

$$
y = r + \gamma \, (1 - d) \, Q_{\bar\psi}\big( s', \mu_{\bar\phi}(s') \big), \qquad
\mathcal{L}_{\text{critic}}(\psi) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}}\Big[ \big( Q_\psi(s, a) - \mathrm{sg}(y) \big)^2 \Big]
$$

其中 $d$ 是终止标志（`terminated`，不含截断），$\bar\psi$、$\bar\phi$ 是目标网络的参数。

**目标网络的软更新。** DQN 每隔若干步把在线网络整个复制给目标网络。这里改成每一步挪一点点：

$$
\bar\psi \leftarrow (1 - \kappa) \, \bar\psi + \kappa \, \psi, \qquad \bar\phi \leftarrow (1 - \kappa) \, \bar\phi + \kappa \, \phi
$$

$\kappa$ 很小（DDPG 论文用 0.001，TD3 和 SAC 论文用 0.005；论文里这个符号记作 $\tau$）。
目标网络是在线网络的指数滑动平均，变化平缓。Dreamer V3 的 critic 也用滑动平均的副本来稳定训练（Phase 6）。

这个算法叫深度确定性策略梯度（Deep Deterministic Policy Gradient，DDPG；Lillicrap et al. 2016，arXiv:1509.02971）。

### 8.3 推导：确定性策略梯度

对 $J(\phi)$ 求梯度。$Q_\psi(s, \mu_\phi(s))$ 是两个函数的复合：先由 $\phi$ 算出动作，再由动作算出 $Q$。用链式法则：

$$
\nabla_\phi J(\phi) = \mathbb{E}_{s \sim \mathcal{D}}\Big[ \nabla_a Q_\psi(s, a) \big\vert_{a = \mu_\phi(s)} \cdot \nabla_\phi \mu_\phi(s) \Big]
$$

第一个因子是 critic 对动作的梯度（"动作往哪挪 $Q$ 会变大"），第二个因子是 actor 的输出对参数的梯度（"参数怎么改能让动作往那边挪"）。
这就是**确定性策略梯度**（Deterministic Policy Gradient，DPG；Silver et al. 2014）。

对照第 7.2 节的 pathwise 公式 $\mathbb{E}[f'(x) \nabla_\phi g_\phi]$：$f$ 是 $Q_\psi$，$x$ 是动作，$g_\phi$ 是 $\mu_\phi$，只是这里连噪声 $\epsilon$ 都没有。

**它为什么可以离策略。** 注意期望只对状态取，没有对动作取期望。
公式里的动作是 actor 现算的 $\mu_\phi(s)$，不是回放池里存的那个动作。所以回放池里的数据是谁采的、什么时候采的都无所谓，
只要状态分布还算合理。对比第 1 节的策略梯度：那里的期望对"当前策略采的动作"取，所以必须在策略。

**手算例子。** 一维动作。假设在某个状态 $s$，critic 学到的是 $Q_\psi(s, a) = -(a - 2)^2$，最优动作是 $a = 2$。
actor 在这个状态的输出就是它的参数：$\mu_\phi(s) = \phi$，当前 $\phi = 0.5$。

$$
\nabla_a Q = -2(a - 2) \big\vert_{a = 0.5} = 3, \qquad \nabla_\phi \mu_\phi = 1, \qquad \nabla_\phi J = 3 \times 1 = 3
$$

学习率取 0.1，做一步梯度上升：$\phi \leftarrow 0.5 + 0.1 \times 3 = 0.8$。动作朝最优值 2 挪了一步。
没有采样，没有方差，一个状态算一次就得到准确的梯度方向。
如果换成 score-function 估计器，要在 $a = 0.5$ 周围加噪声采很多个动作，才能从它们的 $Q$ 值的差异里把同一个方向估计出来。

这个例子也暴露了它的弱点：梯度的方向完全由 critic 决定。critic 画出来的那条曲线哪里错了，actor 就往哪里错。

### 8.4 探索

确定性策略自己不会探索。DDPG 的做法是只在采集数据时给动作加噪声，训练时不加：

$$
a = \mathrm{clip}\big( \mu_\phi(s) + \sigma_{\text{explore}} \cdot \epsilon, \; a_{\min}, \; a_{\max} \big), \qquad \epsilon \sim \mathcal{N}(0, 1)
$$

数据由"带噪声的策略"采集，学的是"不带噪声的策略"。行为策略和目标策略不是同一个，这正是离策略的定义。

噪声的标准差是一个要手调的超参数，而且对所有状态都一样。
第 9 节的 SAC 把它变成策略自己的输出，让策略学会在哪些状态多试、哪些状态少试。

### 8.5 DDPG 的问题与 TD3 的三个改动

DDPG 能用，但很脆：对超参数敏感，训练中途经常崩。病根是 **critic 的高估被 actor 放大**。

Phase 4 讲过 Q-learning 的过估计：对带噪声的 $Q$ 值取 $\max$，结果系统性偏高。
DDPG 里没有显式的 $\max$，但 actor 干的就是求 $\max$ 的活：它专门去找 critic 给分最高的动作。
critic 在某个动作上因为误差给了虚高的分，actor 立刻挪过去；这个虚高的值又通过自举进入目标，传给其他状态。
这和 Phase 1 第 8.5 节的"模型被钻空子"是同一个结构，只是被钻的对象从模型换成了 critic。

TD3（Twin Delayed DDPG；Fujimoto et al. 2018，arXiv:1802.09477）加了三个改动：

| 改动 | 做法 | 针对什么 |
| --- | --- | --- |
| **双 Q 取小**（clipped double Q） | 训练两个独立的 critic，算目标时取两者中较小的 | 高估 |
| **延迟更新 actor** | critic 每更新两次，actor 和目标网络才更新一次 | actor 追着一个还没稳定的 critic 跑 |
| **目标策略平滑** | 算目标时给目标动作加一点截断的噪声 | critic 在动作空间里的尖峰被 actor 利用 |

写成公式，TD3 的目标是：

$$
\tilde{a} = \mathrm{clip}\big( \mu_{\bar\phi}(s') + \mathrm{clip}(\tilde\sigma \epsilon, -c, c), \; a_{\min}, \; a_{\max} \big), \qquad
y = r + \gamma \, (1 - d) \, \min_{j = 1, 2} Q_{\bar\psi_j}(s', \tilde{a})
$$

TD3 论文用的数值是 $\tilde\sigma = 0.2$、$c = 0.5$、每两次 critic 更新做一次 actor 更新。

**双 Q 取小为什么管用：一个手算例子。** 某个动作的真实价值是 10。每个 critic 的估计有误差，
简单起见假设误差等概率地取 $+1$ 或 $-1$，两个 critic 的误差相互独立。四种情况等概率：

| critic 1 | critic 2 | 取小 |
| --- | --- | --- |
| 9 | 9 | 9 |
| 9 | 11 | 9 |
| 11 | 9 | 9 |
| 11 | 11 | 11 |

$$
\mathbb{E}\big[ \min(Q_1, Q_2) \big] = \frac{9 + 9 + 9 + 11}{4} = 9.5 < 10
$$

取小之后的估计偏低 0.5。作为对比，如果有两个真实价值都是 10 的动作，用一个 critic 对它们取 $\max$（Q-learning 的做法），
同样的四种情况会得到 $(9 + 11 + 11 + 11)/4 = 10.5$，偏高 0.5。

偏低为什么比偏高好？因为两种偏差的后果不对称：

- **高估会自我强化。** 被高估的动作会被 actor 选中，它的虚高的值通过自举传播出去，越滚越大。
- **低估不会。** 被低估的动作只是不被选中，错误停在原地，不往外传。

宁可保守，不可乐观。这个原则在 SAC（第 9 节）、MBPO（第 10 节）、TD-MPC2（Phase 7）里都会原样出现。

### 8.6 算法

```
算法：TD3

初始化：actor μ_φ，两个 critic Q_ψ1、Q_ψ2，各自的目标网络，回放池 D
每个环境步：
 1. 选动作 a = μ_φ(s) + 探索噪声，执行，把 (s, a, r, s', terminated) 存入 D
 2. 从 D 采一个小批量                                                   形状 [B, …]
 3. 算目标（不带梯度）：
        ã = μ_φ̄(s') + clip(噪声, −c, c)，再裁剪到动作范围
        y = r + γ (1 − terminated) · min( Q_ψ̄1(s', ã), Q_ψ̄2(s', ã) )
 4. 更新两个 critic：最小化 (Q_ψ1(s, a) − y)² + (Q_ψ2(s, a) − y)²
 5. 每隔 2 步：
        更新 actor：最大化 mean( Q_ψ1(s, μ_φ(s)) )                       只更新 φ
        软更新三个目标网络
```

去掉"取小""延迟""目标噪声"三处，就是 DDPG。

第 5 步里 actor 的损失是 $-Q_{\psi_1}(s, \mu_\phi(s))$。反向传播时梯度会流过 critic 的参数，
但优化器里只放 actor 的参数，critic 不被这个损失更新。critic 在这一步只是一条让梯度通过的管道。

### 8.7 代码对应

对应 `06-ddpg-td3.py`（讲义待写），在 `Pendulum-v1` 上运行。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `obs`，`next_obs` | `[B, 3]` | 从回放池采样 |
| `actions` | `[B, 1]` | 回放池里存的动作，只用于 critic 的损失 |
| `rewards`，`terminated` | `[B, 1]` | 和 `Q` 的输出同形状，避免广播出错 |
| `Q(obs, actions)` | `[B, 1]` | critic 把 `obs` 和 `actions` 拼接后输入 |
| `actor(obs)` | `[B, 1]` | 最后一层用 `tanh`，再乘以动作上限 |

`ddpg_update()` 和 `td3_update()` 的差别就是第 8.5 节表里的三行，讲义用开关逐项打开，观察各自的作用。

> 📌 待回填：DDPG 与 TD3 在 `Pendulum-v1` 上的学习曲线；训练过程中 critic 的预测值与真实折扣回报的对比（用来观察高估），来自 `06-ddpg-td3.py`。定性预期：DDPG 的 $Q$ 值会明显高于真实回报，TD3 的略低于真实回报。

### 8.8 常见误区

- **误区：DDPG 也叫"策略梯度"，所以它和 REINFORCE 是同一类方法。**
  实际情况：名字相近，估计器不同。REINFORCE 用 score-function，梯度不穿过任何价值函数；
  DDPG 用 pathwise，梯度完全来自 critic 对动作的导数。前者在策略，后者离策略。
- **误区：确定性策略没有随机性，所以不需要探索。**
  实际情况：不探索，回放池里只有当前策略附近的数据，critic 对其他动作的评价全是外推，actor 得到的梯度毫无根据。
  探索噪声必须有，只是它加在采集数据的时候。
- **误区：两个 critic 取小会让价值偏低，偏低也是偏差，应该取平均才公平。**
  实际情况：取平均仍然会高估，因为 actor 仍然在对它求最大。高估和低估的后果不对称，低估不会自我强化，所以宁可偏低。
- **误区：actor 的损失是 $-Q$，那么 critic 的参数也会被这个损失更新，变得越来越乐观。**
  实际情况：如果把 critic 的参数也放进了 actor 的优化器，确实会这样，这是一个常见的实现错误。
  正确的做法是 actor 的损失只更新 actor 的参数。

---

## 9. SAC：最大熵强化学习

### 9.1 直觉

TD3 修好了高估，但探索仍然是外挂的：一个固定大小的高斯噪声，不管状态。
这种探索很笨。有些状态下动作必须精确（比如摆快到顶部时），有些状态下怎么做都差不多（比如摆在底部晃的时候），
两种情况用同样大小的噪声并不合理。

软演员-评论家（Soft Actor-Critic，SAC）换了一个出发点：**让策略本身是随机的，并且把"保持随机"写进目标。**
智能体要同时追求两样东西：回报高，以及在回报不受损的前提下动作尽量随机。

这像负载均衡的策略。两台服务器的响应时间估计值是 100 毫秒和 102 毫秒，把全部流量都打到第一台并不明智：
估计值有误差，差距又很小，全压在一边既脆弱又让你永远没机会发现第二台其实更快。
合理的做法是按好坏的差距来分配流量：差距大时集中，差距小时分散。

"尽量随机"带来三个好处：

1. **探索是自适应的。** 哪里的动作好坏差别大，策略就在哪里变得确定；差别小的地方保持随机，继续尝试。
2. **不过早下注。** 两个动作看起来差不多时，策略不会因为一点估计误差就彻底倒向其中一个。
3. **对估计误差不敏感。** 策略不追着 critic 的尖峰跑，训练更稳定。

第 4.4 节的熵正则只是在损失上加一项。SAC 的熵在目标里：critic 估计的"价值"本身就包含了未来能保持多少随机性。

### 9.2 定义

**最大熵目标。** 在每一步的奖励上加一份和策略的熵成正比的奖金：

$$
J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[ \sum_{t} \gamma^t \Big( r_t + \alpha \, \mathcal{H}\big( \pi(\cdot \mid s_t) \big) \Big) \right]
$$

$\alpha$ 叫**温度**，决定熵奖金和任务奖励之间的兑换比例。$\alpha = 0$ 就退回普通的强化学习。

**软价值函数。** 目标变了，价值函数的定义也跟着变。用 $\mathcal{H}(\pi) = \mathbb{E}_{a \sim \pi}[-\log \pi(a \mid s)]$ 把熵写进期望里：

$$
\begin{aligned}
V(s) &= \mathbb{E}_{a \sim \pi}\big[ Q(s, a) - \alpha \log \pi(a \mid s) \big] \\
Q(s, a) &= r(s, a) + \gamma \, \mathbb{E}_{s'}\big[ V(s') \big]
\end{aligned}
$$

和 Phase 4 的 Bellman 方程相比，只多了 $-\alpha \log \pi$ 这一项。约定是：$Q(s, a)$ 不包含当前这一步的熵奖金，
但包含之后所有步的；$V(s)$ 包含当前这一步的。

**SAC 的四个部件：**

| 部件 | 是什么 | 从哪来 |
| --- | --- | --- |
| 随机 actor | tanh 高斯策略 $\pi_\phi(a \mid s)$ | 本节 |
| 重参数化的 actor 更新 | 梯度穿过 critic 和采样 | 第 7 节的 pathwise；第 8 节 |
| 两个 critic 取小 | $\min(Q_{\psi_1}, Q_{\psi_2})$ | 第 8.5 节 |
| 自动调节温度 | 把 $\alpha$ 当成一个可学习的量 | 本节 |

出处是 Haarnoja et al. 2018（arXiv:1801.01290）。自动调节温度和去掉单独的 $V$ 网络是后续版本（arXiv:1812.05905）加的，
现在说的 SAC 通常指后者，本节讲的也是后者。

### 9.3 推导（一）：最优的最大熵策略是 softmax

先看一个状态。假设 $Q(s, \cdot)$ 已知，问：什么样的策略让 $\mathbb{E}_{a \sim \pi}[Q(s, a)] + \alpha \mathcal{H}(\pi)$ 最大？

为了推导简单，假设动作只有有限个。加上"概率之和为 1"的约束（拉格朗日乘子记为 $\nu$）：

$$
\mathcal{F}(\pi) = \sum_a \pi(a) \, Q(a) - \alpha \sum_a \pi(a) \log \pi(a) + \nu \Big( \sum_a \pi(a) - 1 \Big)
$$

**第一步：对某一个 $\pi(a)$ 求偏导并令其为零。** 用到 $\frac{d}{dx}(x \log x) = \log x + 1$。

$$
\frac{\partial \mathcal{F}}{\partial \pi(a)} = Q(a) - \alpha \big( \log \pi(a) + 1 \big) + \nu = 0
$$

**第二步：解出 $\pi(a)$。**

$$
\log \pi(a) = \frac{Q(a)}{\alpha} + \frac{\nu}{\alpha} - 1
\quad \Longrightarrow \quad
\pi(a) = \exp\Big( \frac{Q(a)}{\alpha} \Big) \cdot \underbrace{\exp\Big( \frac{\nu}{\alpha} - 1 \Big)}_{\text{与 } a \text{ 无关的常数}}
$$

**第三步：常数由归一化确定。**

$$
\pi^\star(a \mid s) = \frac{\exp\big( Q(s, a) / \alpha \big)}{\sum_{a'} \exp\big( Q(s, a') / \alpha \big)}
$$

最优策略是对 $Q / \alpha$ 做 softmax。

这个推导和 Phase 1 第 7.5 节 MPPI 权重的推导逐行对应。那里的问题是"期望分数要高，同时不能离原分布太远"，
解是 $q^\star \propto p \cdot \exp(J / \lambda)$；这里的问题是"期望 $Q$ 要高，同时熵要大"，解是 $\pi^\star \propto \exp(Q / \alpha)$。
"熵要大"等价于"离均匀分布近"，所以两者是同一个问题，MPPI 的温度 $\lambda$ 对应这里的 $\alpha$。
这不是巧合：Phase 7 的 TD-MPC2 把 MPPI 规划器和一个最大熵策略配在一起用，两边说的是同一种语言。

**第四步：把最优策略代回去，得到软价值。** 记归一化常数为 $Z = \sum_{a'} \exp(Q(a') / \alpha)$，
则 $\log \pi^\star(a) = Q(a) / \alpha - \log Z$。代入 $V$ 的定义：

$$
V(s) = \mathbb{E}_{a \sim \pi^\star}\big[ Q(a) - \alpha \log \pi^\star(a) \big]
= \mathbb{E}_{a \sim \pi^\star}\big[ Q(a) - Q(a) + \alpha \log Z \big]
= \alpha \log \sum_{a} \exp\big( Q(s, a) / \alpha \big)
$$

这是 $Q$ 值的 log-sum-exp，一个"软"的最大值。SAC 名字里的 soft 就是这个意思：
普通 Bellman 方程里的 $\max_a Q$ 被换成了 soft-max。

**手算例子。** 两个动作，$Q = (1, 2)$。

| $\alpha$ | $\exp(Q / \alpha)$ | 最优策略 $\pi^\star$ | 软价值 $V$ | 说明 |
| --- | --- | --- | --- | --- |
| 1 | $(2.718, 7.389)$，和为 10.107 | $(0.269, 0.731)$ | $\ln 10.107 = 2.313$ | 偏向好动作，但差的也常选 |
| 0.1 | $(e^{10}, e^{20})$ | $(0.00005, 0.99995)$ | $2.000$ | 几乎是贪心，$V \approx \max Q$ |
| 10 | $(1.105, 1.221)$，和为 2.327 | $(0.475, 0.525)$ | $10 \times \ln 2.327 = 8.44$ | 几乎均匀，$V$ 里大部分是熵奖金 |

验证 $\alpha = 1$ 那一行：期望 $Q$ 是 $0.269 \times 1 + 0.731 \times 2 = 1.731$，
熵是 $-(0.269 \ln 0.269 + 0.731 \ln 0.731) = 0.353 + 0.229 = 0.582$，两者相加 $1.731 + 0.582 = 2.313$。对上了。

$\alpha = 10$ 那一行说明了一个实际问题：$\alpha$ 相对于奖励的尺度太大时，$V$ 里熵奖金（约 6.9）远大于任务回报（约 1.5），
智能体会专心"保持随机"而不管任务。$\alpha$ 必须和奖励的尺度匹配，这是需要自动调节它的原因。

### 9.4 推导（二）：critic 与 actor 的损失

连续动作下，$\exp(Q / \alpha)$ 的归一化常数是一个算不出来的积分，没法直接从最优策略里采样。
SAC 的办法是用一个参数化的策略 $\pi_\phi$ 去逼近它。

**critic 的损失。** 把第 9.2 节的两个式子合起来，就是软 Bellman 方程的采样版本。
下一步的动作 $a'$ 从**当前策略**采（不是从回放池取），两个目标 critic 取小：

$$
y = r + \gamma \, (1 - d) \Big( \min_{j = 1, 2} Q_{\bar\psi_j}(s', a') - \alpha \log \pi_\phi(a' \mid s') \Big), \qquad a' \sim \pi_\phi(\cdot \mid s')
$$

$$
\mathcal{L}_{\text{critic}}(\psi_j) = \mathbb{E}_{(s, a, r, s', d) \sim \mathcal{D}}\Big[ \big( Q_{\psi_j}(s, a) - \mathrm{sg}(y) \big)^2 \Big], \qquad j = 1, 2
$$

和 TD3 的目标相比有两处不同：多了 $-\alpha \log \pi$（熵奖金）；不需要目标策略平滑，因为策略本来就是随机的，
对 $a'$ 采样已经起到了平滑的作用。SAC 也没有目标 actor，只有目标 critic。

**actor 的损失。** 让 $\pi_\phi(\cdot \mid s)$ 尽量接近 $\pi^\star \propto \exp(Q_\psi(s, \cdot) / \alpha)$，用 KL 散度衡量：

$$
\begin{aligned}
D_{\mathrm{KL}}\Big( \pi_\phi(\cdot \mid s) \,\Big\Vert\, \frac{\exp(Q_\psi(s, \cdot) / \alpha)}{Z(s)} \Big)
&= \mathbb{E}_{a \sim \pi_\phi}\Big[ \log \pi_\phi(a \mid s) - \frac{Q_\psi(s, a)}{\alpha} + \log Z(s) \Big] && \text{KL 的定义}
\end{aligned}
$$

$\log Z(s)$ 不含 $\phi$，可以丢掉。两边乘以 $\alpha$（不改变最小值点），再对回放池里的状态取平均：

$$
\mathcal{L}_{\text{actor}}(\phi) = \mathbb{E}_{s \sim \mathcal{D}, \, a \sim \pi_\phi(\cdot \mid s)}\big[ \alpha \log \pi_\phi(a \mid s) - Q_\psi(s, a) \big]
$$

读法：最小化它，等于让 $Q$ 大（第二项），同时让 $\log \pi$ 小，也就是熵大（第一项）。
它正好是软价值 $V(s)$ 的相反数：actor 在最大化自己在每个状态的软价值。

**actor 的梯度：重参数化。** 期望是对 $a \sim \pi_\phi$ 取的，$\phi$ 在采样分布里，又回到了第 7 节的问题。
动作是连续的，$Q_\psi$ 是可微的网络，所以用 pathwise。把动作写成噪声的确定性函数 $a = a_\phi(s, \epsilon)$：

$$
\mathcal{L}_{\text{actor}}(\phi) = \mathbb{E}_{s \sim \mathcal{D}, \, \epsilon \sim \mathcal{N}(0, I)}\Big[ \alpha \log \pi_\phi\big( a_\phi(s, \epsilon) \mid s \big) - \min_{j = 1, 2} Q_{\psi_j}\big( s, a_\phi(s, \epsilon) \big) \Big]
$$

现在期望对 $\epsilon$ 取，分布不含 $\phi$，直接反向传播就行。梯度有两条路：

- 穿过 critic：$\nabla_a Q \cdot \nabla_\phi a_\phi$，和 DDPG 一样
- 穿过 $\log \pi_\phi$：既通过动作 $a_\phi$，也通过分布的参数 $\mu_\phi$、$\sigma_\phi$

SAC 的 actor 更新可以看成"带噪声输入、带熵项的 DDPG"。把 $\alpha$ 设为 0、把噪声去掉，它就退化成 DDPG 的 actor 更新。

### 9.5 推导（三）：tanh 高斯策略的对数概率

环境的动作有界（比如 $[-1, 1]$），高斯分布的取值却是整个实数轴。SAC 的做法是先从高斯里采一个 $u$，再用 $\tanh$ 压进 $(-1, 1)$：

$$
u = \mu_\phi(s) + \sigma_\phi(s) \, \epsilon, \qquad a = \tanh(u)
$$

损失里要用 $\log \pi_\phi(a \mid s)$。$a$ 是 $u$ 经过非线性变换得到的，它的概率密度不等于 $u$ 的概率密度，要用换元公式修正。

**第一步：概率守恒。** $u$ 落在一个小区间 $[u, u + du]$ 里的概率，等于 $a$ 落在对应的小区间 $[a, a + da]$ 里的概率：

$$
p_a(a) \, \vert da \vert = p_u(u) \, \vert du \vert
\quad \Longrightarrow \quad
p_a(a) = p_u(u) \, \Big\vert \frac{da}{du} \Big\vert^{-1}
$$

**第二步：算 $\tanh$ 的导数。**

$$
\frac{da}{du} = 1 - \tanh^2(u)
$$

**第三步：取对数。** 动作有多维时各维独立，对每一维的修正项求和：

$$
\log \pi_\phi(a \mid s) = \log \mathcal{N}\big( u; \, \mu_\phi(s), \sigma_\phi^2(s) \big) - \sum_{i} \log\big( 1 - \tanh^2(u_i) \big)
$$

直观理解：$\tanh$ 把很宽的一段 $u$ 压进了很窄的一段 $a$（尤其在两端），同样多的概率挤在更小的范围里，密度就变高了。
$1 - \tanh^2(u) < 1$，它的对数是负的，减去一个负数，$\log \pi$ 变大。

**手算例子。** 一维，$\mu = 0$，$\sigma = 1$，采到 $\epsilon = 0.5$，所以 $u = 0.5$。

$$
\begin{aligned}
a &= \tanh(0.5) = 0.4621 \\
\log \mathcal{N}(0.5; 0, 1) &= -\frac{0.5^2}{2} - \frac{1}{2} \ln(2\pi) = -0.125 - 0.919 = -1.044 \\
\log\big( 1 - \tanh^2(0.5) \big) &= \ln(1 - 0.2135) = \ln 0.7865 = -0.240 \\
\log \pi(a) &= -1.044 - (-0.240) = -0.804
\end{aligned}
$$

**数值稳定性。** $u$ 较大时 $\tanh(u)$ 非常接近 1，$1 - \tanh^2(u)$ 在浮点数里会变成 0，取对数得到负无穷。
$u = 10$ 时 $1 - \tanh^2(u) \approx 8 \times 10^{-9}$，float32 下 $\tanh(10)$ 已经四舍五入成 1。有两种处理办法：

| 办法 | 做法 |
| --- | --- |
| 加一个小常数 | $\log(1 - a^2 + 10^{-6})$。简单，但在两端有误差 |
| 换一个等价的表达式 | $\log(1 - \tanh^2 u) = 2 \big( \log 2 - u - \mathrm{softplus}(-2u) \big)$。精确，不会溢出 |

第二个式子的来历：$1 - \tanh^2 u = 4 / (e^u + e^{-u})^2$，取对数得 $2\log 2 - 2\log(e^u + e^{-u})$，
再把 $\log(e^u + e^{-u})$ 写成 $u + \log(1 + e^{-2u}) = u + \mathrm{softplus}(-2u)$。
用上面的例子验证：$2 \times (0.693 - 0.5 - \ln(1 + e^{-1})) = 2 \times (0.693 - 0.5 - 0.313) = -0.240$。和直接算的一致。

还有一个要点：算 $\log \pi$ 要用 $\tanh$ **之前**的 $u$。如果只存了 $a$，再用 $\mathrm{atanh}(a)$ 反算 $u$，在 $a$ 接近 $\pm 1$ 时会数值爆炸。

### 9.6 推导（四）：自动调节温度

第 9.3 节的例子说明 $\alpha$ 必须和奖励的尺度匹配。不同任务的奖励尺度差很多，同一个任务在训练的不同阶段 $Q$ 的尺度也在变。
手调一个固定的 $\alpha$ 很难。

换一个更容易指定的量：不规定"熵值多少钱"，改为规定"策略至少要有多随机"。把问题写成带约束的优化：

$$
\max_\pi \; \mathbb{E}\Big[ \sum_t \gamma^t r_t \Big] \qquad \text{约束：} \;\; \mathbb{E}_{s, \, a \sim \pi}\big[ -\log \pi(a \mid s) \big] \ge \bar{\mathcal{H}}
$$

$\bar{\mathcal{H}}$ 是**目标熵**，一个比 $\alpha$ 好设得多的超参数。用拉格朗日乘子把约束并进目标，乘子正好就是 $\alpha$：
熵低于目标时乘子变大，加重熵的奖励；熵高于目标时乘子变小。对应的 $\alpha$ 的损失是：

$$
\mathcal{L}(\alpha) = \mathbb{E}_{s \sim \mathcal{D}, \, a \sim \pi_\phi}\Big[ -\alpha \, \big( \log \pi_\phi(a \mid s) + \bar{\mathcal{H}} \big) \Big]
$$

**看它的梯度在做什么。** 括号里的东西对 $\alpha$ 来说是常数（停止梯度）。对 $\alpha$ 求导：

$$
\frac{\partial \mathcal{L}}{\partial \alpha} = -\,\mathbb{E}\big[ \log \pi_\phi(a \mid s) \big] - \bar{\mathcal{H}} = \mathcal{H}_{\text{当前}} - \bar{\mathcal{H}}
$$

其中 $\mathcal{H}_{\text{当前}} = -\mathbb{E}[\log \pi_\phi]$ 是当前策略的平均熵。梯度下降的方向是 $-\partial \mathcal{L} / \partial \alpha$：

| 情况 | 梯度的符号 | $\alpha$ 的变化 | 效果 |
| --- | --- | --- | --- |
| 当前熵低于目标（策略太确定） | 负 | 增大 | 熵奖金更值钱，策略被推向更随机 |
| 当前熵高于目标（策略太随机） | 正 | 减小 | 任务奖励占上风，策略收紧 |

这是一个反馈控制器：$\alpha$ 是控制量，策略的熵是被控量，$\bar{\mathcal{H}}$ 是设定值。

**手算例子。** 一维动作，目标熵 $\bar{\mathcal{H}} = -1$。

- 某个批次里 $\log \pi$ 的平均值是 $0.5$，即当前熵是 $-0.5$，高于目标。
  梯度是 $-0.5 - (-1) = 0.5 > 0$，$\alpha$ 减小。
- 另一个批次里 $\log \pi$ 的平均值是 $2.0$，即当前熵是 $-2.0$，低于目标。
  梯度是 $-2.0 - (-1) = -1.0 < 0$，$\alpha$ 增大。

**目标熵怎么设。** 论文（arXiv:1812.05905）给的经验值是动作维度的相反数：$\bar{\mathcal{H}} = -\dim(\mathcal{A})$。
`Pendulum-v1` 是 $-1$，`HalfCheetah-v5` 是 $-6$。

目标熵是负数，初看很奇怪。第 4.4 节算过：连续分布的熵可以是负的，一维高斯的熵是 $\log \sigma + 1.419$。
熵等于 $-1$ 对应 $\log \sigma = -2.419$，即 $\sigma \approx 0.09$（忽略 tanh 的修正）。
所以"每个动作维度的熵不低于 $-1$"大致是说：每一维的动作至少保留标准差 0.09 左右的随机性。

实现时优化的是 $\log \alpha$ 而不是 $\alpha$，这样 $\alpha = \exp(\log \alpha)$ 自动保持为正。

**和 Dreamer 的对照。** "熵系数要和回报的尺度匹配"这个问题 Dreamer 也有，它的解法方向相反：
不调熵系数，而是把回报归一化到一个固定的尺度，然后用一个固定的熵系数（Phase 6 的回报归一化）。
两种办法解决的是同一个问题。

### 9.7 算法

```
算法：SAC

初始化：actor π_φ，两个 critic Q_ψ1、Q_ψ2 及其目标网络，log α，回放池 D，目标熵 H̄ = −动作维度
每个环境步：
 1. 采样动作 a ~ π_φ(· | s)，执行，把 (s, a, r, s', terminated) 存入 D
 2. 从 D 采一个小批量                                                        形状 [B, …]

 3. 更新 critic：
        a' , log π(a'|s') ← 从 π_φ(· | s') 采样                               不带梯度
        y = r + γ (1 − terminated) · ( min(Q_ψ̄1, Q_ψ̄2)(s', a') − α · log π(a'|s') )
        最小化 (Q_ψ1(s, a) − y)² + (Q_ψ2(s, a) − y)²                          a 来自回放池

 4. 更新 actor：
        ã , log π(ã|s) ← 从 π_φ(· | s) 重参数化采样                            带梯度
        最小化 mean( α · log π(ã|s) − min(Q_ψ1, Q_ψ2)(s, ã) )                 只更新 φ；α 当常数

 5. 更新温度：
        最小化 mean( − log α · ( log π(ã|s) + H̄ ) )                          log π 当常数

 6. 软更新两个目标 critic
```

注意第 3 步和第 4 步里两个不同的动作：critic 的损失用回放池里存的 $a$，actor 的损失用当前策略新采的 $\tilde{a}$。
搞混是最常见的实现错误之一。

第 5 步写的是常见实现的形式：直接拿 $\log \alpha$ 去乘括号里的量。
它和第 9.6 节的损失对 $\log \alpha$ 的梯度符号相同，调节的方向一样。

论文（arXiv:1812.05905）附录给出的默认超参数：Adam 学习率 $3 \times 10^{-4}$，$\gamma = 0.99$，回放池容量 $10^6$，
两层各 256 个单元的 MLP，小批量 256，软更新系数 0.005，每个环境步做一次梯度更新。
SAC 的一个优点是这套超参数在很多连续控制任务上不用改。

### 9.8 代码对应

对应 `07-sac-from-scratch.py`（讲义待写）。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `mean`，`log_std` | `[B, A]` | actor 网络的两个输出头；`log_std` 要夹在一个范围内 |
| `u` | `[B, A]` | `mean + log_std.exp() * randn_like(mean)`，tanh 之前 |
| `action` | `[B, A]` | `tanh(u)`，再按环境的动作范围缩放 |
| `log_prob` | `[B, 1]` | 高斯对数密度减去 tanh 修正，**对动作维求和** |
| `q1`，`q2` | `[B, 1]` | 两个 critic 的输出 |
| `log_alpha` | 标量 | 可学习参数 |

`TanhGaussianActor.forward(obs)` 返回 `(action, log_prob)`；`sac_update()` 实现第 9.7 节的第 3–6 步。
讲义用第 9.5 节的数字给 `log_prob` 做单元测试（$u = 0.5$ 时应得到 $-0.804$）。

SAC 是本章排行榜里的无模型主力，也是下一节 MBPO 的底座。到 Phase 7，TD-MPC2 训练策略先验时用的也是这一节的做法：
重参数化的 actor 更新加熵项，critic 用多个 $Q$ 网络来抑制高估。

> 📌 待回填：SAC 在 `Pendulum-v1` 和 `HalfCheetah-v5` 上的学习曲线；训练过程中 $\alpha$ 与策略熵的变化曲线；固定 $\alpha$ 取几个不同值时的对比，来自 `07-sac-from-scratch.py`。定性预期：$\alpha$ 在训练初期较大，随后下降并稳定；策略熵收敛到目标熵附近。

### 9.9 常见误区

- **误区：SAC 的策略是随机的，所以它用 score-function 估计器。**
  实际情况：SAC 用的是 pathwise。随机策略通过重参数化变成"噪声的确定性函数"，梯度穿过 critic 传回来。
  策略是不是随机的，和用哪种估计器是两个独立的选择。
- **误区：熵项只是为了探索，训练后期应该把 $\alpha$ 降到 0。**
  实际情况：熵在目标里，它改变了什么是"最优"。自动调节会让 $\alpha$ 随训练下降，但不会到 0，
  因为约束要求策略保留一定的随机性。评估时取均值动作 $\tanh(\mu_\phi(s))$ 即可。
- **误区：tanh 修正项只是个细节，省掉了影响不大。**
  实际情况：省掉之后 $\log \pi$ 在动作接近边界时严重偏小，熵被高估。策略会被推向边界，
  输出饱和在 $\pm 1$，这是一个典型的失败模式。
- **误区：目标熵是负数，说明公式写错了。**
  实际情况：连续分布的熵是微分熵，可以是负的。负的目标熵对应一个较窄但不为零的分布宽度。

---

## 10. MBPO：把模型的短展开喂给 SAC

### 10.1 直觉

SAC 比 PPO 省样本，但在 MuJoCo 任务上仍然要几十万到上百万步真实交互。
放到真机上，这个数字意味着几天到几周不间断地跑，中间不能坏。

Phase 1 学了怎么从交互数据里学一个动力学模型，Phase 4 的 Dyna 给出了用法：
**真实数据用来学模型，模型用来生成更多的数据，两种数据一起训练智能体。**
每一步真实交互被模型"放大"成很多步训练数据。

把这个想法直接套到 SAC 上，会撞上 Phase 1 第 5 节的复合误差。从回合的起点出发，在模型里展开一整个回合（1000 步），
误差按 $L^t$ 增长，后半段的数据全是模型的幻觉。SAC 在幻觉上训练，学到的是怎么在幻觉里拿高分。

MBPO（Model-Based Policy Optimization；Janner et al. 2019，arXiv:1906.08253）的办法只有一句话：
**不从起点展开，从回放池里随便挑一个真实状态开始；不展开很长，只走很少几步。**

这像从线上数据库的快照 fork 出一个临时环境做实验。快照是真实的，实验只跑很短的时间，
临时环境和线上的差异还来不及积累就结束了。想覆盖不同的时间段，就多取几个不同时刻的快照，而不是让一个临时环境一直跑下去。

```
   真实轨迹（存在回放池 D_env 里）

   s_0 ──▶ s_1 ──▶ s_2 ──▶ s_3 ──▶ s_4 ──▶ ··· ──▶ s_999
            │               │               │
            │ 模型           │ 模型           │ 模型
            ▼               ▼               ▼
           ŝ ─▶ ŝ          ŝ ─▶ ŝ          ŝ ─▶ ŝ           每个分支只走 k 步（这里 k = 2）
                                                             动作由当前策略选

   所有分支上的转移 (ŝ, a, r̂, ŝ') 存进 D_model，用来训练 SAC
```

每个分支都从真实状态分叉出去，所以叫**分支展开**（branched rollout）。

### 10.2 定义

MBPO 由三样已经学过的东西拼成：

| 部件 | 是什么 | 在哪学的 |
| --- | --- | --- |
| 模型 | 概率集成：若干个高斯 MLP，预测下一状态的变化量和奖励 | Phase 1 第 8 节 |
| 数据生成 | 从真实状态出发的 $k$ 步分支展开 | 本节 |
| 策略优化 | SAC，主要在模型生成的数据上训练 | 第 9 节 |

**分支展开。** 从真实数据的回放池 $\mathcal{D}_{\text{env}}$ 里均匀采一个状态 $s$，然后重复 $k$ 次：

$$
a \sim \pi_\phi(\cdot \mid \hat{s}), \qquad (\hat{s}', \hat{r}) \sim p_{\theta_b}(\cdot \mid \hat{s}, a), \quad b \sim \mathrm{Uniform}\lbrace 1, \dots, B \rbrace
$$

每一步随机选集成里的一个模型 $b$ 来预测（对应 Phase 1 第 9.3 节的 TS1），把 $(\hat{s}, a, \hat{r}, \hat{s}')$ 存进模型数据的回放池 $\mathcal{D}_{\text{model}}$。

注意三点：

1. **起点是真实的**，而且来自回合的各个位置。回合第 800 步附近的状态也会被选作起点，
   所以不需要从头展开 800 步就能覆盖到那里。
2. **动作由当前策略选**，不是回放池里存的旧动作。模型回答的是"现在的策略在这个状态会做什么、结果怎样"。
3. **模型只当数据生成器用**。没有梯度穿过模型，模型可微不可微都无所谓。这一点和第 11 节相反。

### 10.3 推导：为什么要短

**从复合误差的界出发。** Phase 1 第 5.3 节推出，开环展开 $k$ 步后的状态误差满足：

$$
e_k \le \varepsilon \, \frac{L^k - 1}{L - 1}
$$

沿用那里的数字 $\varepsilon = 0.01$、$L = 1.2$：

| 展开长度 $k$ | $L^k$ | 误差上界 | 是单步误差的多少倍 |
| --- | --- | --- | --- |
| 1 | 1.2 | 0.010 | 1 |
| 5 | 2.488 | $0.01 \times 1.488 / 0.2 = 0.074$ | 7.4 |
| 10 | 6.19 | 0.26 | 26 |
| 30 | 237.4 | 11.8 | 1180 |

$k = 1$ 时模型数据的误差就是单步误差，这是模型最拿手的事。$k$ 每增加一点，数据质量按指数下降。

**短展开为什么够用。** 两个原因：

*覆盖靠起点，不靠长度。* 长展开的用处是到达远处的状态。分支展开的起点已经分布在真实数据覆盖的所有状态上，
不需要靠展开来"走过去"。

*长期的后果靠 critic，不靠展开。* SAC 的 critic 用自举学习：一条一步的转移 $(\hat{s}, a, \hat{r}, \hat{s}')$
加上 $Q(\hat{s}', \cdot)$，就包含了关于长期回报的信息。价值沿着一条条短的转移一步步往回传，
和 Phase 4 里 TD 学习把价值从终点传回起点是同一个机制。Phase 1 的 MPC 没有价值函数，只能靠展开得够长来看见未来，这里不需要。

**那为什么不干脆永远用 $k = 1$？** 展开长一点有一个好处：能生成"当前策略会到达、但回放池里还没有"的状态。
回放池里的数据是过去的策略采的，当前策略已经变了，它实际会去的地方和回放池覆盖的地方有偏差。
从真实状态出发按当前策略多走几步，能把数据往当前策略的分布上挪一点。

所以 $k$ 上有一个权衡：

```
   k 小                                                        k 大
   ◀──────────────────────────────────────────────────────────▶
   模型误差小                                    模型误差按 L^k 增长
   但数据分布停留在旧策略访问的状态附近            但数据更接近当前策略的分布
```

MBPO 论文对这个权衡做了理论分析，结论的定性版本是：只要模型的泛化足够好，最优的展开长度大于 0，但不大。
论文的做法是让 $k$ 随训练逐步增大：早期模型差，用 $k = 1$；后期数据多了、模型准了，再慢慢加长。
具体的日程因任务而异（论文附录的超参数表；本课不照搬，按自己模型的开环误差曲线来定）。

论文摘要的结论是：这个简单的做法在样本效率上超过了此前的基于模型的方法，最终性能追平最好的无模型方法，
并且能用在那些让其他基于模型的方法完全失效的长时域任务上。

### 10.4 手算例子：一步真实交互被放大多少倍

设每个真实环境步之后做 $M = 400$ 条分支展开，每条 $k = 1$ 步，SAC 做 $G = 20$ 次梯度更新，小批量 256。
（这组数字是 MBPO 论文附录超参数表里的量级，待核对；这里只用来说明记账的方法。）

| 量 | 计算 | 结果 |
| --- | --- | --- |
| 每个真实步生成的模型转移 | $M \times k = 400 \times 1$ | 400 |
| 每个真实步 SAC 消耗的样本 | $G \times 256 = 20 \times 256$ | 5120 |
| 1000 个真实步（一个回合）生成的模型转移 | $1000 \times 400$ | 40 万 |
| 1000 个真实步里的梯度更新次数 | $1000 \times 20$ | 2 万 |

普通 SAC 每个真实步做 1 次梯度更新，MBPO 做 20 次。这个比值叫**更新数据比**（update-to-data ratio，UTD）。

为什么普通 SAC 不能直接把 UTD 调到 20？因为它只有真实数据。在同一小撮数据上反复更新，critic 会过拟合，
高估问题加重。MBPO 敢这么做，是因为每次更新用的大多是新生成的模型数据：同一个真实起点，
当前策略每次采的动作不同，模型给出的后继也不同，等于做了大量的数据增强。

代价也看得出来：每个真实步要做 400 次集成的前向计算和 20 次 SAC 更新。MBPO 省的是真实交互，花的是计算。
对真机来说这笔交易很划算，对仿真来说未必。

### 10.5 MBPO 与 Dyna、PETS、Dreamer 的关系

把四个用模型的方法并排放：

| | Dyna-Q（Phase 4） | PETS（Phase 1） | MBPO（本节） | Dreamer（Phase 6） |
| --- | --- | --- | --- | --- |
| 模型 | 表格，或一个确定性模型 | 概率集成 | 概率集成 | RSSM（隐空间） |
| 模型在什么时候用 | 训练时，生成数据 | 决策时，现场规划 | 训练时，生成数据 | 训练时，生成想象轨迹 |
| 展开的起点 | 见过的真实状态 | 当前的真实状态 | 回放池里的真实状态 | 回放数据推断出的隐状态 |
| 展开的长度 | 1 步 | 整个规划视野（几十步） | 很短（1 到十几步） | 较短（十几步） |
| 展开时的动作 | 回放过的动作 | 规划器采样的候选序列 | 当前策略 | 当前策略 |
| 学什么 | $Q$ 表 | 不学策略 | SAC 的 actor 和 critic | actor 和 critic |
| 模型数据与真实数据 | 混用 | 不适用 | 以模型数据为主，混少量真实数据 | 行为学习只用想象数据 |
| 梯度穿过模型吗 | 不 | 不（采样式规划） | 不 | V1 穿过（第 11 节） |

几条线索：

- **MBPO 是深度版的 Dyna。** 取 $k = 1$ 时两者的结构几乎相同：从见过的状态出发，用模型生成一步转移，喂给一个无模型算法。
  区别是 MBPO 的动作由当前策略现选，模型是概率集成，底下的算法从 Q-learning 换成了 SAC。
- **MBPO 和 PETS 用同一种模型，用法相反。** PETS 在决策时用模型，没有策略，每一步都要规划，决策慢；
  MBPO 在训练时用模型，学出一个策略，决策时一次前向计算。Phase 1 第 10.3 节的那张对比表说的就是这件事。
- **Dreamer 继承了分支展开。** Dreamer 的想象轨迹也不是从回合起点开始，而是从回放数据对应的隐状态出发往前走一小段。
  Phase 6 的学习检查里有一条"想象轨迹为什么从回放数据的后验状态出发"，答案就在本节。
- **集成对抗钻空子的作用没变。** 策略优化同样会利用模型的漏洞（Phase 1 第 8.5 节）。
  每一步随机换一个集成成员，单个模型的漏洞不容易被持续利用。

### 10.6 算法

```
算法：MBPO

初始化：概率集成 p_θ，SAC（actor、两个 critic、α），真实数据池 D_env，模型数据池 D_model
先用随机策略采一些数据放进 D_env
重复若干个 epoch：
 1. 在 D_env 上训练集成（最大似然；留出一部分验证，早停）
 2. 对这个 epoch 里的每个环境步：
      a. 用当前策略在真实环境走一步，存进 D_env
      b. 分支展开：
           从 D_env 采 M 个状态作为起点                                  形状 [M, S]
           重复 k 步：
               a ~ π_φ(· | ŝ)
               随机选集成成员，采样 (ŝ', r̂)
               把 (ŝ, a, r̂, ŝ', 是否终止) 存进 D_model
               已经终止的分支不再继续
      c. 做 G 次 SAC 更新，小批量主要来自 D_model，混入少量 D_env 的数据
 3. 按日程调整 k；丢弃 D_model 里过旧的数据
```

几个容易忽略的地方：

- **模型数据会过期。** $\mathcal{D}_{\text{model}}$ 里的数据是旧模型、旧策略生成的。模型重新训练之后，旧数据的误差更大；
  策略更新之后，旧数据的动作分布也不对了。所以 $\mathcal{D}_{\text{model}}$ 只保留最近几个 epoch 的数据。
- **终止条件要另外给。** 模型预测下一状态和奖励，但"回合是否结束"通常不让模型学，而是用环境已知的终止规则
  （比如 `Hopper-v5` 的躯干高度和角度超出范围）在预测出的状态上判断。
- **奖励由模型预测。** 这一点和 Phase 1 不同，那里的奖励函数是已知的。做法是让模型多输出一维。

### 10.7 代码对应

对应 `08-mbpo.py`（讲义待写）。概率集成直接复用 Phase 1 `07-uncertainty-and-ensembles.py` 的实现，SAC 复用 `07-sac-from-scratch.py`。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| 起点 `start_obs` | `[M, S]` | 从 $\mathcal{D}_{\text{env}}$ 均匀采样 |
| 集成的输出均值、对数方差 | `[E, M, S + 1]` | `E` 个集成成员；最后一维是 `S` 维状态变化量加 1 维奖励 |
| 成员索引 `member_idx` | `[M]` | 每个分支这一步用哪个成员，每步重新随机 |
| 选出的预测 | `[M, S + 1]` | 按 `member_idx` 取出，再从高斯里采样 |
| 存进 $\mathcal{D}_{\text{model}}$ 的转移 | `obs` `[M, S]`，`action` `[M, A]`，`reward` `[M, 1]`，`next_obs` `[M, S]`，`terminated` `[M, 1]` | |

`branched_rollout(ensemble, actor, start_obs, k)` 实现第 2b 步；`mbpo_train()` 是外层循环。

注意这里把集成的成员数写成 `E`。Phase 1 用的字母是 `B`，在本章会和批大小撞名。

> 📌 待回填：MBPO 与 SAC 在相同环境步数下的回报曲线；展开长度 $k$ 取 1、5、20 时的对比；模型数据占比的影响，来自 `08-mbpo.py`。定性预期：MBPO 在训练早期明显领先；$k$ 过大时性能下降，下降的位置应该和模型开环误差开始失控的步数相符。

### 10.8 常见误区

- **误区：MBPO 的展开很短，所以它是一个短视的方法。**
  实际情况：展开长度决定的是数据的质量，不是策略能看多远。策略的视野由 critic 的自举和折扣因子决定，
  $\gamma = 0.99$ 时有效视野是 100 步的量级，和 $k$ 无关。
- **误区：模型数据是假的，应该让真实数据占大头才安全。**
  实际情况：MBPO 的样本效率正是来自大量的模型数据。真实数据只占小批量的很小一部分。
  安全靠的是短展开和集成，不是压低模型数据的比例。
- **误区：MBPO 用的是世界模型，所以它在"想象中做规划"。**
  实际情况：MBPO 不做规划。模型只是一个数据生成器，学习算法是普通的 SAC，决策时完全不碰模型。
- **误区：模型越准，$k$ 就该越大，越大越好。**
  实际情况：$k$ 大了之后，一条分支上后面几步的起点已经是模型自己预测出来的状态。
  策略优化会主动把分支带往模型给分虚高的地方，那里恰好是模型最不准的地方。误差和钻空子会相互加强。

---

## 11. 沿模型反向传播：从 SVG 到 Dreamer

### 11.1 直觉

MBPO 用模型的方式有点浪费。动力学模型是一个神经网络，它对输入的状态和动作都是可微的。
MBPO 把它当成黑盒仿真器来用：喂进去状态和动作，拿出来下一状态，仅此而已。
"下一状态对动作的导数"这个现成的信息完全没用上。

回顾一下第 8 节：DDPG 的 actor 之所以能高效地学，是因为 critic 可微，梯度能穿过 critic 告诉 actor"动作往哪挪"。
现在手上有了另一个可微的东西：世界模型。把策略、模型、奖励函数首尾相接，
"想象出的回报"就成了策略参数的一个可微函数，整条链路可以一次反向传播到底。

```
   s_0 ──▶ π_φ ──▶ a_0 ──▶ f_θ ──▶ ŝ_1 ──▶ π_φ ──▶ a_1 ──▶ f_θ ──▶ ŝ_2 ──▶ ··· ──▶ ŝ_H ──▶ V_ψ
            ▲       │       │       │        ▲       │       │       │                      │
            │       ▼       │       ▼        │       ▼       │       ▼                      ▼
            │      r̂_0 ◀────┘      (r̂_1 …)   │      r̂_1 ◀────┘                          终端价值
            │                                │
            └────────────────────────────────┴──── 梯度从所有的 r̂ 和 V 出发，沿箭头反方向流回每一个 π_φ
```

对工程师来说，这是黑盒压测和白盒分析的区别。黑盒压测是改一个参数、跑一遍、看指标，要跑很多遍才能看出趋势。
白盒分析是手上有整条链路的源码，可以直接算出"这个参数动一点，最终指标动多少"。

用第 7 节的话说：MBPO 把模型生成的数据交给 SAC，SAC 的 actor 梯度只穿过 critic；
这里是把 **pathwise 估计器用在整条想象轨迹上**，梯度穿过模型的每一步。

### 11.2 定义

**可微的想象展开。** 从一个真实状态 $s$ 出发，在模型里走 $H$ 步：

$$
\hat{s}_0 = s, \qquad a_k = \pi_\phi(\hat{s}_k, \epsilon_k), \qquad \hat{s}_{k+1} = f_\theta(\hat{s}_k, a_k), \qquad \hat{r}_k = r_\theta(\hat{s}_k, a_k)
$$

其中 $\pi_\phi(\hat{s}, \epsilon)$ 是重参数化之后的策略：给定噪声，动作是状态的确定性可微函数。

**actor 的目标**（要最大化）。$H$ 步的想象奖励，加上末端的价值：

$$
J(\phi) = \mathbb{E}_{s \sim \mathcal{D}, \, \epsilon}\left[ \sum_{k=0}^{H-1} \gamma^k \hat{r}_k + \gamma^H V_\psi(\hat{s}_H) \right]
$$

末端的 $V_\psi$ 概括了 $H$ 步之外的回报。Phase 1 第 6.6 节说过"视野之外的事规划器完全看不见，要用价值函数来补"，这里补上了。

**要满足的条件。** 整条链路上每一环都必须可微：

| 环节 | 要求 | 不满足时怎么办 |
| --- | --- | --- |
| 策略 | 能重参数化（连续动作的高斯策略可以） | 离散动作：改用 score-function，或用 straight-through |
| 模型 $f_\theta$ | 对状态和动作可微 | 神经网络模型天然满足；真实环境和多数仿真器不满足 |
| 奖励 $r_\theta$ | 对状态和动作可微 | 用学出来的奖励模型；稀疏的 0/1 奖励要靠模型把它"抹平" |
| 状态 | 连续 | 离散隐变量：straight-through |
| 价值 $V_\psi$ | 对状态可微 | 神经网络天然满足 |

### 11.3 推导：把梯度手算一遍

反向传播由自动求导完成，不需要手写。但手算一个小例子能看清梯度沿哪些路径流动。

**设定。** 一维状态，一维动作。模型是 $\hat{s}' = \hat{s} + a$，策略是线性的 $a = \phi \, \hat{s}$（确定性，$\phi$ 是一个标量），
奖励只看到达的状态：到达 $\hat{s}'$ 得到 $-\hat{s}'^2$，目标是把状态拉到 0。
从 $s_0 = 1$ 出发，想象 $H = 2$ 步，不折扣，不加末端价值。当前 $\phi = -0.5$。

**前向。**

$$
\begin{aligned}
a_0 &= \phi \, s_0 = -0.5, & \hat{s}_1 &= s_0 + a_0 = 0.5 \\
a_1 &= \phi \, \hat{s}_1 = -0.25, & \hat{s}_2 &= \hat{s}_1 + a_1 = 0.25 \\
J &= -\hat{s}_1^2 - \hat{s}_2^2 = -0.25 - 0.0625 = -0.3125
\end{aligned}
$$

**反向。** 从后往前，先算 $J$ 对每个状态的全导数，再汇总到 $\phi$。

*对 $\hat{s}_2$：* 只有最后一项奖励依赖它。

$$
\frac{dJ}{d\hat{s}_2} = -2\hat{s}_2 = -0.5
$$

*对 $\hat{s}_1$：* 它有两条路影响 $J$。一条直接通过第一项奖励；一条通过 $\hat{s}_2$。
而 $\hat{s}_2 = \hat{s}_1 + \phi \hat{s}_1$，所以 $\partial \hat{s}_2 / \partial \hat{s}_1 = 1 + \phi = 0.5$，
其中的 1 来自模型，$\phi$ 来自"状态变了，策略给出的动作也跟着变"。

$$
\frac{dJ}{d\hat{s}_1} = -2\hat{s}_1 + \frac{dJ}{d\hat{s}_2} \cdot (1 + \phi) = -1 + (-0.5)(0.5) = -1.25
$$

*对 $\phi$：* $\phi$ 在两个时间步各出现一次。在第 $k$ 步，$\phi$ 通过动作 $a_k = \phi \hat{s}_k$ 直接影响 $\hat{s}_{k+1}$，
这条直接路径的导数是 $\partial \hat{s}_{k+1} / \partial \phi = \hat{s}_k$。把两个时间步的贡献加起来：

$$
\frac{dJ}{d\phi} = \frac{dJ}{d\hat{s}_1} \cdot s_0 + \frac{dJ}{d\hat{s}_2} \cdot \hat{s}_1 = (-1.25)(1) + (-0.5)(0.5) = -1.5
$$

**验证。** 这个例子可以直接写出闭式。$\hat{s}_1 = (1 + \phi) s_0$，$\hat{s}_2 = (1 + \phi)^2 s_0$，所以：

$$
J(\phi) = -(1 + \phi)^2 - (1 + \phi)^4, \qquad
\frac{dJ}{d\phi} = -2(1 + \phi) - 4(1 + \phi)^3 = -2(0.5) - 4(0.125) = -1.5
$$

一致。梯度是负的，说明应该减小 $\phi$。最优值是 $\phi = -1$（一步就把状态拉到 0），方向是对的。

**一般形式。** 把上面的手算写成递推，就是 SVG 论文（Heess et al. 2015，arXiv:1510.09142）里的价值梯度。
记 $f_s$、$f_a$ 为模型对状态和动作的雅可比，$r_s$、$r_a$ 为奖励的梯度，$\pi_s$、$\pi_\phi$ 为策略对状态和参数的雅可比：

$$
\begin{aligned}
\frac{dJ}{d\hat{s}_k} &= r_s + r_a \, \pi_s + \gamma \, \frac{dJ}{d\hat{s}_{k+1}} \big( f_s + f_a \, \pi_s \big) \\
\frac{dJ}{d\phi} &= \sum_{k=0}^{H-1} \gamma^k \Big( r_a + \gamma \, \frac{dJ}{d\hat{s}_{k+1}} \, f_a \Big) \pi_\phi
\end{aligned}
$$

其中每一项都在第 $k$ 步的 $(\hat{s}_k, a_k)$ 处取值，$dJ/d\hat{s}_k$ 指的是从第 $k$ 步往后的回报对 $\hat{s}_k$ 的导数。
第一行从末端的 $dJ/d\hat{s}_H = \nabla V_\psi(\hat{s}_H)$ 开始往回递推，这就是沿时间的反向传播（BPTT）。
（手算例子把奖励记在"到达的状态"上，这里记在"出发的状态和动作"上，记账方式略有不同，算出的 $dJ/d\phi$ 相同。）

**梯度和误差是同一个矩阵连乘出来的。** 看第一行里的 $(f_s + f_a \pi_s)$：它是"这一步的状态变一点，下一步的状态变多少"，
也就是闭环系统的单步雅可比。梯度从第 $H$ 步传回第 0 步，要连乘 $H$ 个这样的矩阵。

Phase 1 第 5.3 节里，模型的误差每一步被放大 $L$ 倍，$L$ 就是这个雅可比的大小的上界。所以：

| 闭环雅可比的大小 | 展开时的误差 | 反向传播时的梯度 |
| --- | --- | --- |
| 小于 1 | 有界 | 指数衰减（梯度消失，看不到远处的奖励） |
| 大于 1 | 指数增长 | 指数增长（梯度爆炸，方差极大） |

**误差会爆炸的系统，梯度也会爆炸，原因是同一个。** 上面的例子里这个因子是 $1 + \phi = 0.5$，很温和。
如果策略是 $\phi = 0.5$，因子变成 1.5，展开 20 步后梯度里就有 $1.5^{20} \approx 3300$ 这样的项。
这也是第 7.4 节预告过的情况：$f$ 的导数又大又乱时，pathwise 估计器的方差可以比 score-function 还大。

所以沿模型反向传播的视野 $H$ 不能长。两头的限制和 Phase 1 第 6.6 节讲 MPC 视野时一样，只是多了一条"梯度不稳定"。
解决办法也是同一个：短视野，加末端价值。

### 11.4 一个家族：从 SAC 到 Dreamer

把"梯度穿过模型多少步"当作一个旋钮，前面几节的方法排成一条线。SVG 论文把这个家族记作 SVG($H$)：

| 方法 | 梯度穿过模型的步数 | 末端 | 数据从哪来 |
| --- | --- | --- | --- |
| DDPG、SAC 的 actor 更新（对应 SVG(0)） | 0 | $Q_\psi(s, a)$ | 回放池的真实状态（MBPO 里是模型生成的状态） |
| SVG(1) | 1 | $V_\psi(\hat{s}_1)$ | 回放池的真实状态 |
| Dreamer V1（Hafner et al. 2020，arXiv:1912.01603） | $H$（论文用 15） | $V_\psi$，并用 λ-return 把各个步数的估计混合 | 回放数据推断出的隐状态 |
| SVG($\infty$) | 整个回合 | 无 | 回合起点 |

$H = 0$ 时模型完全不参与 actor 的梯度，所有关于未来的信息都压在 critic 里；$H$ 越大，越多的信息来自模型的显式展开，
对 critic 的依赖越小，对模型的依赖越大。这和第 5 节 GAE 的 $\lambda$ 是同一种权衡：多相信学出来的价值，还是多相信实际（这里是想象）的展开。

Dreamer V1 做的事就是这一行：在 RSSM 的隐空间里想象 $H$ 步，用 λ-return（第 5 节）混合不同步数的价值估计，
actor 的梯度沿着隐空间里的模型一路反传回来。它和本节的区别只在于"状态"是从图像学出来的隐状态。
Phase 6 从这里接着讲。

### 11.5 和 MBPO 的对比

| | MBPO：模型生成数据 | 沿模型反向传播 |
| --- | --- | --- |
| 模型的角色 | 数据生成器（黑盒） | 计算图的一部分（白盒） |
| actor 的梯度穿过什么 | 只穿过 critic | 穿过模型的 $H$ 步、奖励模型、末端的 critic |
| 用到模型的导数吗 | 不用 | 用 |
| 对模型的要求 | 能采样就行 | 必须可微，而且导数要靠得住 |
| 动作类型 | 连续（SAC）；换个无模型算法也能处理离散 | 连续；离散动作需要 straight-through 或改用 score-function |
| 模型误差的影响 | 影响数据的质量 | 影响数据的质量，还影响梯度的方向 |
| 被钻空子的风险 | 有 | 更高：梯度会精确地指向模型给分虚高的方向 |

最后一行值得展开说。采样式的方法（CEM、MBPO）要靠碰运气才能撞上模型的漏洞。
梯度方法不用碰运气，它直接沿着"模型认为回报上升最快"的方向走，而模型的漏洞往往正是那样的方向。
Phase 1 的 FAQ 里说过"梯度会精确地指向模型有漏洞的方向"，指的就是这件事。

应对的办法和前面是同一套：视野要短；起点取自真实数据（和 MBPO 的分支展开一样）；策略带熵正则，不要走到太极端的地方。

### 11.6 算法

```
算法：沿可微模型反向传播的 actor-critic

输入：可微的模型 f_θ 和奖励 r_θ（已训练好，或与本算法交替训练），actor π_φ，critic V_ψ
重复：
 1. 从回放池采一批真实状态作为起点 ŝ_0                               形状 [B, S]
 2. 想象展开，全程保留计算图：
        对 k = 0 … H−1：
            a_k = π_φ(ŝ_k) 的重参数化采样                           形状 [B, A]
            r̂_k = r_θ(ŝ_k, a_k)                                    形状 [B]
            ŝ_{k+1} = f_θ(ŝ_k, a_k)                                形状 [B, S]
 3. actor 损失 = − mean( Σ_k γ^k r̂_k + γ^H V_ψ(ŝ_H) ) − 熵奖励
    反向传播；只更新 φ（θ 和 ψ 不放进这个优化器）；梯度裁剪
 4. critic 损失 = mean( (V_ψ(sg(ŝ_k)) − sg(目标_k))² )
    目标用想象轨迹上的 λ-return 计算；只更新 ψ
```

三处停止梯度各有分工：

| 位置 | 作用 |
| --- | --- |
| actor 损失只更新 $\phi$ | 梯度**流过**模型和 critic，但不**更新**它们。否则模型会被改成"让策略看起来回报更高"，而不是预测得准 |
| critic 的输入 $\mathrm{sg}(\hat{s}_k)$ | critic 的损失不该反过来影响 actor 或模型 |
| critic 的目标 $\mathrm{sg}(\cdot)$ | 半梯度 TD，和第 4.2 节一样 |

"梯度流过但不更新"是这类方法里最容易搞混的一点，Phase 6 讲 Dreamer V3 的损失时每一个 $\mathrm{sg}$ 都会标出来。

### 11.7 代码对应

对应 `09-backprop-through-model.py`（讲义待写）。讲义在 `Pendulum-v1` 上做，模型用 Phase 1 学到的确定性 MLP，
奖励用已知的解析式（它对状态和动作可微）。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `start_obs` | `[B, S]` | 起点，不带梯度 |
| `imagined_obs` | `[B, H + 1, S]` | 带计算图 |
| `imagined_actions` | `[B, H, A]` | 重参数化采样，带计算图 |
| `imagined_rewards` | `[B, H]` | 带计算图 |
| `terminal_value` | `[B]` | $V_\psi(\hat{s}_H)$，带计算图 |

`imagine_rollout(model, actor, start_obs, horizon)` 做第 2 步；`actor_loss_through_model()` 做第 3 步。

讲义还做两个诊断实验：一是在同一批想象轨迹上，分别用 pathwise 和 score-function 估计 actor 的梯度，比较方差；
二是画出梯度范数随视野 $H$ 的变化。

> 📌 待回填：沿模型反向传播的 actor 在 `Pendulum-v1` 上的学习曲线；pathwise 与 score-function 两种梯度估计的方差之比；梯度范数随视野 $H$ 变化的曲线，来自 `09-backprop-through-model.py`。定性预期：$H$ 较小时 pathwise 的方差明显更小；$H$ 增大到一定程度后梯度范数快速增长。

### 11.8 常见误区

- **误区：沿模型反向传播就是在训练模型。**
  实际情况：模型的参数在这一步是冻结的。梯度穿过模型，最终更新的只有策略的参数。
  模型另有自己的损失（预测误差），在另一步里训练。
- **误区：既然能反向传播，视野越长，策略看得越远，效果越好。**
  实际情况：梯度要连乘 $H$ 个雅可比矩阵，长了会爆炸或消失；模型误差也随 $H$ 累积。
  远处的事交给末端的价值函数，视野保持在十几步的量级。
- **误区：这个办法比 MBPO 高级，所以应该总是用它。**
  实际情况：它要求动作连续、模型和奖励可微且导数可靠。动作离散、奖励稀疏、动力学有不连续的接触时，
  用模型生成数据再交给无模型算法反而更稳。Dreamer V2 在离散动作的任务上就改用了 score-function 估计器（Phase 6 详述）。
- **误区：模型预测得准，它的导数就准。**
  实际情况：两个函数的值处处接近，导数可以差很远（第 7.4 节的高频波纹就是例子）。
  模型的训练目标只约束了预测值，没有约束导数。这是沿模型反传比生成数据更怕模型误差的原因。

---

## 12. 方法对比与选型

### 12.1 本章的算法

| 算法 | 策略 | actor 的梯度估计器 | 权重或 critic | 数据 | 用到模型吗 |
| --- | --- | --- | --- | --- | --- |
| REINFORCE | 随机 | score-function | 蒙特卡洛回报 $G_t$ | 在策略，用一次 | 否 |
| A2C | 随机 | score-function | $V_\psi$，TD 或 n 步优势 | 在策略，用一次 | 否 |
| PPO | 随机 | score-function（带重要性权重和裁剪） | $V_\psi$，GAE | 在策略，每批用若干个 epoch | 否 |
| DDPG | 确定性 | pathwise（穿过 $Q_\psi$） | 一个 $Q_\psi$ | 离策略，回放池 | 否 |
| TD3 | 确定性 | pathwise（穿过 $Q_\psi$） | 两个 $Q_\psi$ 取小 | 离策略，回放池 | 否 |
| SAC | 随机（tanh 高斯） | pathwise（穿过 $Q_\psi$ 和采样） | 两个 $Q_\psi$ 取小，带熵项 | 离策略，回放池 | 否 |
| MBPO | 同 SAC | 同 SAC | 同 SAC | 离策略，主要是模型生成的短分支 | 是，生成数据 |
| 沿模型反向传播 | 随机（可重参数化） | pathwise（穿过模型的 $H$ 步和 $V_\psi$） | $V_\psi$ 作末端价值 | 想象轨迹，起点来自回放池 | 是，梯度穿过模型 |

竖着看这张表，每一列都只有两三种取值。新算法几乎都是在这几列里换一个选项：

- 把 A2C 的"用一次"换成"用若干个 epoch 并限制步长"，得到 PPO
- 把 DDPG 的确定性策略换成带熵项的随机策略，得到 SAC
- 把 SAC 的数据来源换成模型的短分支，得到 MBPO
- 把 SAC 的"梯度只穿过 critic"换成"梯度穿过模型的 $H$ 步"，得到 Dreamer V1 的行为学习

### 12.2 在策略与离策略

| | 在策略（REINFORCE、A2C、PPO） | 离策略（DDPG、TD3、SAC） |
| --- | --- | --- |
| 数据能复用吗 | 不能，或只能复用当前这一轮 | 能，回放池里的数据反复用 |
| 样本效率 | 低 | 高一到两个数量级 |
| 稳定性 | 较好，调参相对容易 | 自举、离策略、函数逼近三者叠加，容易不稳定 |
| 墙上时间 | 仿真很快、能大规模并行时很有优势 | 每个环境步都要做梯度更新，交互便宜时不占优 |
| 适合 | 仿真便宜的场合；离散和连续动作都行 | 交互昂贵的场合；本章讲的三个都只适用于连续动作 |
| 和世界模型的关系 | 可以直接在模型的想象里跑（想象轨迹就是"当前策略的数据"） | 模型生成的数据放进回放池（MBPO） |

最后一行解释了一个现象：Dreamer 在想象里用的是在策略式的 actor-critic，而不是 SAC。
想象轨迹是当前策略在模型里现跑出来的，要多少有多少，在策略方法"费样本"的缺点在想象里不存在。

### 12.3 模型的四种用法

到本章为止，学到的模型一共有四种用法。Phase 1 第 10.3 节的两分法在这里细化成了四类：

| 用法 | 怎么做 | 模型要可微吗 | 代表 | 本课 |
| --- | --- | --- | --- | --- |
| 决策时规划 | 每一步在模型里比较多条候选动作序列 | 不需要 | PETS、PlaNet、TD-MPC2 | P1、P3、P7 |
| 生成数据 | 模型的输出当作经验，交给无模型算法 | 不需要 | Dyna、MBPO | P4、本章第 10 节 |
| 沿模型反向传播 | 想象回报对策略参数直接求导 | 需要 | SVG、Dreamer V1 | 本章第 11 节、P6 |
| 在想象里做在策略学习 | 想象轨迹当作在策略数据，用 score-function 估计器 | 不需要 | Dreamer V2、V3 处理离散动作时 | P6 |

四种用法不互斥。TD-MPC2 在决策时规划，同时学一个策略给规划器当初始猜测；
Dreamer 的不同版本、不同动作类型在后两种之间选择或混合。

### 12.4 同一张排行榜

本章的实战项目（`projects/phase-05-control-benchmark/`）建立一张连续控制排行榜。
它是贯穿全课的三条线索之一：Phase 6 把 Dreamer 加进来，Phase 7 把 TD-MPC2 加进来，在同一套协议下比较。

**评估协议。** 排行榜的意义全在于"同一套"。下面这些必须对所有算法保持一致：

| 项目 | 约定 | 为什么 |
| --- | --- | --- |
| 任务 | `Pendulum-v1`、`HalfCheetah-v5`、`Hopper-v5` | 从简单到中等难度；`Hopper-v5` 有提前终止，能暴露终止处理的问题 |
| 横轴 | 真实环境的交互步数 | 样本效率是世界模型的卖点。模型里想象的步数不计入 |
| 预算 | 每个任务所有算法用相同的环境步数上限 | 否则比的是谁跑得久 |
| 随机种子 | 每个算法在每个任务上跑多个种子，种子列表固定 | 强化学习的种子间方差很大，单个种子的结论不可信 |
| 评估 | 每隔固定的环境步数，暂停训练，用不带探索噪声的策略跑若干个评估回合 | 训练回合的回报混着探索噪声，不同算法的噪声大小不同，不可比 |
| 报告 | 学习曲线（均值加种子间的波动范围）和汇总表 | 只报最终分数会掩盖样本效率的差别 |
| 额外记录 | 墙上时间、每个环境步的梯度更新次数 | 省样本往往是用计算换来的，要把代价也摆出来 |

预算和种子数的具体数值在写项目时根据本机算力确定。

**汇总表。** 每个格子填"该任务在预算用完时的评估回报（多个种子的均值与标准差）"，另附"达到指定回报阈值所需的环境步数"。

| 算法 | 类型 | `Pendulum-v1` | `HalfCheetah-v5` | `Hopper-v5` | 加入排行榜的章节 |
| --- | --- | --- | --- | --- | --- |
| 随机策略 | 参考线 | 待回填 | 待回填 | 待回填 | Phase 5 |
| PETS | 基于模型，决策时规划 | 待回填 | 待回填 | 待回填 | Phase 1 实现，Phase 5 入榜 |
| PPO | 无模型，在策略 | 待回填 | 待回填 | 待回填 | Phase 5 |
| SAC | 无模型，离策略 | 待回填 | 待回填 | 待回填 | Phase 5 |
| MBPO | 基于模型，生成数据 | 待回填 | 待回填 | 待回填 | Phase 5 |
| Dreamer | 基于模型，在想象中学习 | — | — | — | Phase 6 |
| TD-MPC2 | 基于模型，隐空间规划 | — | — | — | Phase 7 |

> 📌 待回填：上表全部数字、各任务的"回报—环境步数"曲线、各算法的墙上时间，来自项目 `projects/phase-05-control-benchmark/`。Dreamer 与 TD-MPC2 两行分别在 Phase 6、Phase 7 的项目里填。

**读结果之前的预期。** 文献里的定性结论是：在相同的环境步数下，基于模型的方法（PETS、MBPO）早期领先，
SAC 居中，PPO 最慢；预算足够大时无模型方法的最终性能会追上来。
PETS 论文摘要里的说法是在 HalfCheetah 上达到相近性能所需的样本约为 SAC 的 1/8、PPO 的 1/125；
MBPO 论文摘要的说法见第 10.3 节。我们自己的实现规模小、调参少，未必复现这些结论，
以实测为准。如果实测的排序和文献不一致，先查实现，再下结论。

**关于怎么比才公平。** 强化学习的实验结论对种子、超参数、实现细节都很敏感。
Henderson et al. 2018（Deep Reinforcement Learning that Matters）展示过同一个算法的不同实现、不同种子之间的差距可以大过算法之间的差距；
Agarwal et al. 2021（Deep Reinforcement Learning at the Edge of the Statistical Precipice）建议在种子数少的时候报告更稳健的汇总统计量和置信区间。
Phase 12 讲研究工程方法时会系统地讲这些。

### 12.5 选型建议

| 场景 | 建议 | 理由 |
| --- | --- | --- |
| 离散动作，仿真便宜 | PPO | 稳定，好调，容易并行 |
| 连续动作，仿真便宜，能大规模并行 | PPO | 墙上时间短 |
| 连续动作，交互昂贵 | SAC | 样本效率高，默认超参数通用 |
| 连续动作，交互非常昂贵，状态是低维的 | MBPO，或 Phase 7 的 TD-MPC2 | 用计算换样本 |
| 输入是图像 | Phase 6 的 Dreamer | 需要先学隐状态，本章的方法都假设状态是低维向量 |
| 需要一个参考基线来验证环境和评估流程 | 先跑 PPO 或 SAC | 先确认无模型基线能学会，再上基于模型的方法 |

---

## 13. 论文精读卡片

### 13.1 GAE

**Schulman, Moritz, Levine, Jordan, Abbeel. High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR 2016. arXiv:1506.02438**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 策略梯度需要大量样本；价值函数能降方差，但带来偏差。怎样在两者之间平滑地调节 |
| 核心方法 | 把各种 n 步优势估计做指数加权平均，得到 $\sum_l (\gamma\lambda)^l \delta_{t+l}$；策略和价值函数都用信赖域方法优化 |
| 关键实验 | 三维仿真机器人的运动任务（双足、四足的行走，双足从地上站起来），策略直接从关节状态输出力矩 |
| 主要结论 | 合适的 $\lambda$（介于 0 和 1 之间）明显优于两个极端；$\gamma$ 和 $\lambda$ 的角色不同 |
| 读的时候注意 | 第 2 节列出的策略梯度的几种等价形式（本章第 3.3 节的表）；第 3 节对 $\gamma$ 和 $\lambda$ 各自引入什么偏差的讨论；把 GAE 解释成奖励塑形的那一节可以跳过 |
| 与本课的联系 | PPO 用它算优势；Dreamer 在想象轨迹上用同一个递推算 λ-return |

### 13.2 PPO

**Schulman, Wolski, Dhariwal, Radford, Klimov. Proximal Policy Optimization Algorithms. 2017. arXiv:1707.06347**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 普通策略梯度每批数据只能用一次，步长难调；TRPO 稳定但实现复杂，不方便和共享参数的网络、辅助损失一起用 |
| 核心方法 | 裁剪的代理目标：对概率比做裁剪并取悲观的下界；在同一批数据上做多个 epoch 的小批量更新 |
| 关键实验 | MuJoCo 连续控制任务上比较几种代理目标（不裁剪、裁剪、KL 惩罚）；Atari 上与 A2C 等对比 |
| 主要结论 | 裁剪目标在比较的几种代理目标里最好；实现简单，表现稳定 |
| 局限 | 在策略，样本效率低；表现很依赖论文正文没有强调的实现细节 |
| 读的时候注意 | 第 3 节的裁剪目标和配图（本章第 6.3 节的两张示意图就是它）；第 5 节的完整损失；附录的超参数表 |
| 与本课实现的差异 | 📌 待回填：写完 `04-gae-and-ppo.py` 后列出（并行环境数、每轮步数、用了哪些实现技巧） |

### 13.3 SAC

**Haarnoja, Zhou, Abbeel, Levine. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. ICML 2018. arXiv:1801.01290**

**Haarnoja et al. Soft Actor-Critic Algorithms and Applications. 2018. arXiv:1812.05905**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 在策略方法费样本；离策略的 DDPG 省样本但很脆，对超参数敏感 |
| 核心方法 | 最大熵目标；随机 actor 用重参数化更新；两个 critic 取小。第二篇加上了温度的自动调节，并去掉了单独的状态价值网络 |
| 关键实验 | MuJoCo 连续控制任务上与 DDPG、TD3、PPO 等对比；第二篇还有真实机器人上的实验（四足机器人行走、灵巧手操作） |
| 主要结论 | 样本效率和最终性能都达到或超过当时的方法，不同随机种子之间的表现更一致 |
| 局限 | 只适用于连续动作（离散版本需要另外推导）；目标熵仍是一个经验设定 |
| 读的时候注意 | 第一篇的软策略迭代部分给出了理论依据，第一遍可以只读结论；第二篇的约束优化视角（本章第 9.6 节）和附录的超参数表 |
| 与本课的联系 | MBPO 的底座；TD-MPC2 的策略先验沿用最大熵的 actor 更新 |
| 与本课实现的差异 | 📌 待回填：写完 `07-sac-from-scratch.py` 后列出 |

### 13.4 MBPO

**Janner, Fu, Zhang, Levine. When to Trust Your Model: Model-Based Policy Optimization. NeurIPS 2019. arXiv:1906.08253**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 用模型生成数据能省样本，但模型误差会随展开长度累积。到底该在多大程度上相信模型 |
| 核心方法 | 从真实状态出发做很短的分支展开；概率集成；用模型数据训练 SAC |
| 理论部分 | 给出"在模型里的回报提升能在多大程度上保证真实回报提升"的界，分析展开长度的影响；结合对模型泛化能力的实测，论证短展开是合理的 |
| 关键实验 | MuJoCo 连续控制任务上与 SAC、PETS 等对比；对展开长度、模型数据的用法做了消融 |
| 主要结论 | 摘要：样本效率超过此前的基于模型的方法，最终性能追平最好的无模型方法，并能用于让其他基于模型的方法失效的长时域任务 |
| 局限 | 只处理低维状态；计算量大（每个真实步要做大量模型前向计算和策略更新）；终止条件需要已知 |
| 读的时候注意 | 标题就是全文的问题。读理论部分时抓住结论的形状即可：哪些项随展开长度增大，哪些项随它减小 |
| 与本课实现的差异 | 📌 待回填：写完 `08-mbpo.py` 后列出（集成大小、展开长度的日程、更新数据比等与论文的差别） |

### 13.5 SVG

**Heess, Wayne, Silver, Lillicrap, Tassa, Erez. Learning Continuous Control Policies by Stochastic Value Gradients. NeurIPS 2015. arXiv:1510.09142**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 确定性的价值梯度方法要求模型和策略都是确定的。怎样把"沿模型反向传播"推广到随机策略和随机的环境 |
| 核心方法 | 用重参数化把随机的策略和随机的模型都写成"噪声的确定性函数"，于是可以沿着真实轨迹或想象轨迹反向传播价值的梯度。给出一个按展开步数区分的算法家族 |
| 主要结论 | 这一族算法能在多个连续控制任务上学会控制；论文还比较了不同展开步数的变体，以及它们对模型精度的敏感程度 |
| 读的时候注意 | 价值梯度的递推式（本章第 11.3 节）；论文写于 VAE 的重参数化提出后不久，可以看到同一个技巧怎样从生成模型搬到控制里 |
| 与本课的联系 | Dreamer V1 的行为学习可以看成这个家族在隐空间里的成员 |

### 13.6 蒙特卡洛梯度估计

**Mohamed, Rosca, Figurnov, Mnih. Monte Carlo Gradient Estimation in Machine Learning. JMLR 2020. arXiv:1906.10652**

| 项 | 内容 |
| --- | --- |
| 是什么 | 一篇综述。系统整理"对期望求梯度"的各种估计器 |
| 涵盖的估计器 | score-function、pathwise，以及第三类 measure-valued（本课不讲）。straight-through 不在它的主线里，出处是 Bengio et al. 2013（arXiv:1308.3432） |
| 读哪些部分 | 只读 score-function 和 pathwise 两节，以及比较两者方差的小例子。那些例子和本章第 7.3、7.4 节是同一类 |
| 读的时候注意 | 它的记号里分布的参数是 $\theta$、被积函数是 $f$。对应到强化学习：分布是策略，$f$ 是回报或 critic |

### 13.7 TD3

**Fujimoto, van Hoof, Meger. Addressing Function Approximation Error in Actor-Critic Methods. ICML 2018. arXiv:1802.09477**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | actor-critic 里价值的高估，以及它怎样被策略更新放大 |
| 核心方法 | 两个 critic 取小；延迟更新 actor；目标策略平滑 |
| 读的时候注意 | 论文前半部分对"高估在 actor-critic 里同样存在"的论证和实验。这部分是 Phase 4 过估计内容在连续动作上的延续 |
| 与本课的联系 | "两个 critic 取小"被 SAC、MBPO 原样采用；TD-MPC2 用多个 $Q$ 网络的集成来做同样的事 |

---

## 14. 实现陷阱与调试清单

| 现象 | 可能的原因 | 怎么排查 | 怎么修 |
| --- | --- | --- | --- |
| 损失在下降，回报不动 | 策略梯度的损失是代理损失，数值没有意义 | 画回报曲线，不看损失 | 以评估回报为准；另外监控策略熵和近似 KL |
| REINFORCE 的回报曲线剧烈抖动，不同种子结果差很多 | 梯度方差太大；回报全为正且没有基线 | 固定策略，重复估计梯度，看分量的方差 | 用 reward-to-go；加基线；标准化优势；加大批量 |
| 加了基线后反而学不动 | 优势没有从计算图里断开，策略损失的梯度流进了价值网络 | 检查优势张量的 `requires_grad` | 对优势调用 `detach()` |
| 策略很快变成几乎确定的，回报卡在一个次优值 | 熵塌缩，探索停止 | 画策略熵随训练的变化 | 加熵正则或调大系数；降低学习率；检查优势的尺度是否过大 |
| 到步数上限的任务上，价值函数学得很差，价值随时间步莫名下降 | 把截断当成了终止，自举时把下一状态的价值清零了 | 打印回合最后一步的自举目标 | 只对 `terminated` 清零；截断时用截断前最后一个观测的价值自举 |
| GAE 算出的优势在回合交界处异常大 | 递推跨过了回合边界，没有把累计量清零 | 用第 5.4 节的数字做单元测试；构造一个含两个回合的片段检查 | 递推里乘以 `1 - done` |
| PPO 训练中回报突然崩溃，之后起不来 | 单轮更新走得太远 | 监控近似 KL 和被裁剪样本的占比 | 减少 epoch 数；降低学习率；近似 KL 超阈值时提前停止；加梯度裁剪 |
| PPO 的概率比出现 `inf` 或 `nan` | 直接用概率相除；或者 `log_probs_old` 带着梯度、数值不一致 | 打印 `ratio` 的最大最小值 | 在对数空间相减再取指数；旧的对数概率在收集时存下来并断开梯度 |
| 高斯策略的损失出现 `nan` | 对数标准差没有限制，标准差变成 0 或极大 | 监控 `log_std` 的范围 | 把 `log_std` 夹在一个范围内 |
| 沿模型或 critic 反传时，actor 的梯度是零 | 用了 `sample()` 而不是 `rsample()`，计算图在采样处断了 | 检查动作张量的 `requires_grad` | 换成 `rsample()`，或手写 `mean + std * noise` |
| DDPG 或 SAC 的 $Q$ 值一路上涨，远高于实际回报 | 高估被自举放大；或者终止标志处理错了 | 把 critic 的预测和真实折扣回报画在一起 | 两个 critic 取小；检查目标里的 `1 - terminated`；降低学习率 |
| SAC 的动作总是饱和在边界上 | 漏了 tanh 的对数概率修正；或者 $\alpha$ 太小 | 画动作的直方图；用第 9.5 节的数字测试 `log_prob` | 补上修正项；检查目标熵的设置 |
| SAC 的 `log_prob` 出现 `-inf` 或 `nan` | 用 $\log(1 - a^2)$ 且 $a$ 四舍五入成了 $\pm 1$；或者用反双曲正切从动作反算 | 打印 tanh 之前的值的范围 | 用 tanh 之前的值算；换成 softplus 形式的稳定表达式 |
| SAC 的 $\alpha$ 一直涨到很大，或者掉到接近 0 后策略不动 | 目标熵的符号或大小设错；温度损失的符号写反 | 画 $\alpha$ 和策略熵的曲线，看熵是否朝目标熵靠近 | 目标熵设为动作维度的相反数；对照第 9.6 节的梯度方向检查符号 |
| critic 的损失因为形状广播而悄悄算错 | `Q` 的输出是 `[B, 1]`，奖励是 `[B]`，相减得到 `[B, B]` | 打印损失里每个张量的形状 | 统一成 `[B, 1]` 或统一成 `[B]` |
| actor 的更新让 critic 也跟着变 | critic 的参数被放进了 actor 的优化器 | 检查每个优化器管理的参数 | 每个网络一个优化器；actor 的损失只对 actor 的参数调用 `step()` |
| MBPO 的回报先升后崩，critic 的值远高于真实回报 | 模型被钻空子；展开太长；模型数据过期 | 比较模型数据上的预测回报和真实回报；画模型的开环误差曲线 | 缩短展开；用集成并每步随机换成员；只保留最近的模型数据 |
| MBPO 在会提前终止的任务上学出奇怪的行为 | 模型展开时没有判断终止，生成了"已经摔倒还在继续"的数据 | 检查模型数据里状态的范围 | 在预测出的状态上套用环境的终止规则 |
| 沿模型反传时梯度范数爆炸 | 视野太长，雅可比连乘 | 画梯度范数随视野的变化 | 缩短视野；加末端价值；梯度裁剪 |
| 沿模型反传的策略在模型里回报很高，在真实环境里很差 | 梯度沿着模型的漏洞走 | 把想象回报和真实回报画在一起 | 缩短视野；起点取自真实数据；加熵正则；补充数据重训模型 |

调试顺序的建议：先在最简单的环境上（`CartPole-v1`、`Pendulum-v1`）确认算法能学会，再换难的任务；
先确认无模型的 SAC 能学会，再接上模型做 MBPO。出问题时先看三样东西：评估回报、策略熵、critic 的预测值与真实回报的差。
能用手算例子做单元测试的函数（reward-to-go、GAE、PPO 损失、tanh 修正）都先测过再用。

---

## 15. 常见问题（FAQ）

### Q1：Phase 4 的值函数方法和本章的策略梯度方法，根本区别是什么？

区别在于"策略存在哪里"。值函数方法（DQN）只学 $Q$，策略是隐含的：每次决策时对 $Q$ 取 $\arg\max$。
策略梯度方法显式地学一个策略网络，决策时直接前向计算。
前者在离散动作上简单高效，但 $\arg\max$ 在连续动作上不好算，而且只能表示确定性策略。
后者天然支持连续动作和随机策略，代价是梯度估计的方差。
actor-critic 两样都学，用价值函数来降低策略梯度的方差。本章第 8 节之后的算法以及 Dreamer、TD-MPC2 都属于这一类。

### Q2：策略梯度定理里为什么没有出现环境的转移概率？

轨迹的对数概率是三部分之和：初始状态、策略、环境转移。只有策略那一部分含参数 $\phi$，
对 $\phi$ 求梯度时另外两部分直接为零（第 1.3 节第四步）。
环境的影响并没有消失，它藏在期望里：轨迹是按真实环境的转移采样出来的。
所以我们不需要知道环境的方程，只需要能在环境里跑。

### Q3：实现里把 $\gamma^t$ 丢掉，算出来的还是梯度吗？

严格说不是折扣目标的梯度。带 $\gamma^t$ 的版本会让回合后段的样本权重趋于零，那些状态下的策略几乎学不到东西，
而回合后段的状态往往同样重要。丢掉之后，每个时间步被同等对待，折扣只留在 $G_t$ 或优势的计算里，
起控制方差和有效视野的作用。这是一个有偏但实用的做法，几乎所有实现都这么干。

### Q4：基线可以是任意只依赖状态的函数，为什么大家都学一个 $V$？

三个原因。$V^\pi(s)$ 接近最优基线，降方差的效果好（第 3.3、3.4 节）。
减去它之后得到的是优势，有清楚的含义。更重要的是，学出来的 $V$ 不只能当基线，还能用来自举，
这才是 actor-critic、GAE 降低方差的主要来源。既然要学，就一物两用。

### Q5：PPO 能用在连续动作上吗？

能。把类别分布换成高斯分布，其余完全不变：$\log \pi_\phi(a \mid s)$ 换成高斯的对数密度，概率比照样算。
连续动作的 PPO 通常让标准差是一个不依赖状态的可学习参数，动作超出范围时在环境那一侧裁剪。
注意 PPO 即使用高斯策略，用的仍然是 score-function 估计器。它的"损失"里有 $\log \pi$ 和概率比，没有任何东西对动作求导。

### Q6：离策略方法为什么更省样本？代价是什么？

省样本是因为每条数据被用很多次：存进回放池之后，直到被挤出去之前会被反复采到。
在策略方法的数据只用一轮。代价是稳定性。离策略、自举、函数逼近这三样同时出现时，
价值学习没有收敛的保证，误差可能自我放大（Phase 4 讲 DQN 时提过）。
目标网络、两个 critic 取小、延迟更新，这些看似零碎的技巧都是为了压住这个不稳定性。

### Q7：SAC 能用于离散动作吗？

能，而且更简单。离散动作下对动作的期望可以精确求和，不需要采样，也就不需要重参数化：
actor 的损失直接写成 $\sum_a \pi_\phi(a \mid s) \big( \alpha \log \pi_\phi(a \mid s) - Q_\psi(s, a) \big)$，critic 一次输出所有动作的 $Q$ 值。
这个变体通常叫离散 SAC。本课的连续控制排行榜用不到它。

### Q8：重参数化只能用于高斯分布吗？

不是。只要采样能写成"固定分布的噪声经过一个对参数可微的变换"就行。
位置—尺度族的分布（高斯、拉普拉斯、逻辑分布）都可以；经过可逆变换的分布也可以，tanh 高斯就是一个例子。
不行的是离散分布：样本取值是跳变的，对参数没有导数。这就是需要 straight-through 或 Gumbel-Softmax 的地方。

### Q9：straight-through 有偏，用它训练出来的模型还能信吗？

估计器有偏不等于训练结果是错的，只是说参数更新的方向和真实梯度有夹角。
只要夹角不大，沿着它走损失仍然下降。验证办法也很直接：看用它训练时损失是否正常下降、最终性能如何。
Dreamer V2 的论文报告类别隐变量配合 straight-through 的效果好于高斯隐变量，Phase 6 会讨论可能的原因。
在不能接受偏差的场合，才需要退回 score-function。

### Q10：MBPO 和 PETS 用的是同一种模型，为什么还要再学一个策略？

两个原因。一是决策速度：PETS 每一步都要做上万次模型调用，MBPO 决策时一次前向计算。
二是视野：PETS 只能看到规划视野之内的奖励，MBPO 的 critic 通过自举能概括很远的未来，所以不需要长的展开。
反过来，PETS 换一个奖励函数就能直接用，MBPO 要重新训练策略。
把两者的优点合起来的办法是"规划加价值函数加策略先验"，这是 Phase 7 的 TD-MPC2。

### Q11：更新数据比（UTD）能不能无限调高？

不能。对普通的 SAC，UTD 调高会让 critic 在有限的数据上过拟合，高估加重，性能反而下降。
MBPO 能用高 UTD 是因为模型在不断供应新数据。后来有一些无模型的工作通过更大的 critic 集成或更强的正则化，
让 SAC 在不用模型的情况下也能把 UTD 提到很高。这说明 MBPO 的收益有一部分来自"更新得多"，
不全是模型的功劳。读基于模型的论文时值得留意这一点：对比的无模型基线用的是多大的 UTD。

### Q12：Dreamer 既然是本课的重点，为什么还要花时间学 PPO、SAC、MBPO？

第一，Dreamer 的行为学习部分就是本章的内容：想象里跑的是 actor-critic，价值目标是 λ-return，
actor 的梯度用的是本章的三种估计器。不先在简单的设定里弄明白，到 Dreamer 里和世界模型搅在一起就很难调试。
第二，它们是排行榜上的参照物。说一个世界模型方法"省样本"，得有无模型的基线来比。
第三，SAC 是 TD-MPC2 的基础，MBPO 的分支展开被 Dreamer 继承。

### Q13：熵正则和 SAC 的最大熵，是一回事吗？

不完全是。A2C、PPO 的熵正则是在每一步的损失上加一项，只鼓励**当前这一步**的动作分布别太尖。
SAC 把熵写进回报，价值函数里包含了**未来所有步**的熵。后者会让策略偏好那些"之后还能保持很多选择"的状态。
在实践中，前者是一个防止过早收敛的小补丁，后者是算法的核心。

### Q14：沿模型反向传播和 Phase 1 里"用梯度优化动作序列"有什么不同？

优化的对象不同。Phase 1 的 FAQ 里说的是对一条具体的动作序列做梯度下降，每到一个新状态都要重新优化一遍，结果只用一次。
本章第 11 节优化的是策略的参数：梯度来自很多个起点的想象轨迹的平均，优化的结果是一个能泛化到新状态的函数。
平均之后梯度的噪声更小，学到的策略决策时不需要再做任何优化。两者共同的困难是梯度要穿过多步模型，视野长了都会不稳定。

### Q15：这一章的内容和机器人有什么关系？

关系很直接。目前足式机器人的运动控制，主流做法是在大规模并行的仿真里用 PPO 训练，再迁移到真机。
SAC 的第二篇论文直接在真实的四足机器人和灵巧手上训练。
MBPO 这类方法针对的是真机数据昂贵的问题。到了后面，DayDreamer（Phase 6）把 Dreamer 直接放在真实机器人上在线学习，
TD-MPC2（Phase 7）是具身方向常用的基线，它们的行为学习部分都建立在本章之上。

---

## 16. 自测题

做题时先不要看答案。概念题试着用自己的话说出来，计算题动手算。

### 第 1 题

写出对数导数技巧，并说明它在策略梯度的推导里解决了什么问题。

<details><summary>参考答案</summary>

对数导数技巧是恒等式 $\nabla_\phi p_\phi(x) = p_\phi(x) \, \nabla_\phi \log p_\phi(x)$，来自 $\nabla \log p = \nabla p / p$。

把梯度移进积分号后得到 $\int \nabla_\phi p_\phi(\tau) \, G_0(\tau) \, d\tau$，这个式子不是"在 $p_\phi$ 下的期望"的形式，
没法用采样估计。用这个恒等式把 $\nabla p_\phi$ 换成 $p_\phi \nabla \log p_\phi$ 之后，被积式里重新出现了 $p_\phi$，
积分变回期望 $\mathbb{E}_{\tau \sim p_\phi}[\nabla_\phi \log p_\phi(\tau) \, G_0(\tau)]$，可以用采样的轨迹来估计。

</details>

### 第 2 题

策略梯度的推导中，环境的转移概率在哪一步消失了？这带来什么好处，又留下什么限制？

<details><summary>参考答案</summary>

在展开 $\log p_\phi(\tau)$ 的那一步。轨迹的对数概率是初始状态、策略、环境转移三部分的对数之和，
后两部分里只有策略含 $\phi$，对 $\phi$ 求梯度时初始状态和环境转移的项为零。

好处：不需要知道环境的模型，环境也不必可微，奖励可以不连续。策略梯度因此可以是无模型的。

限制：期望仍然是在"当前策略与真实环境"产生的轨迹分布下取的，所以必须用当前策略去环境里采样，数据不能复用（在策略）。

</details>

### 第 3 题

两臂老虎机，softmax 策略，两个 logit 都是 0。动作 1 的奖励是 2，动作 2 的奖励是 6。

1. 求期望回报对第一个 logit 的真实梯度。
2. 写出采到每个动作时的单样本估计（第一个分量），验证其期望等于真实梯度，并求方差。
3. 取基线 $b = 4$，重做第 2 问。

<details><summary>参考答案</summary>

$\pi = (0.5, 0.5)$。

1. $\partial J / \partial z_1 = \pi_1(1 - \pi_1) \times 2 - \pi_1 \pi_2 \times 6 = 0.5 - 1.5 = -1$。
2. 得分函数的第一个分量：采到动作 1 是 $1 - 0.5 = 0.5$，采到动作 2 是 $-0.5$。
   单样本估计是 $0.5 \times 2 = 1$ 和 $-0.5 \times 6 = -3$。
   均值 $0.5 \times 1 + 0.5 \times (-3) = -1$，等于真实梯度。
   方差 $0.5 \times 1 + 0.5 \times 9 - (-1)^2 = 5 - 1 = 4$。
3. 减去基线后的权重是 $-2$ 和 $+2$。单样本估计是 $0.5 \times (-2) = -1$ 和 $-0.5 \times 2 = -1$。
   均值是 $-1$，方差是 0。

基线没有改变期望，把方差从 4 降到了 0。

</details>

### 第 4 题

证明减去基线 $b(s_t)$ 不改变策略梯度的期望。如果基线依赖动作 $a_t$，证明在哪一步不成立？

<details><summary>参考答案</summary>

要证 $\mathbb{E}[\nabla_\phi \log \pi_\phi(a_t \mid s_t) \, b(s_t)] = 0$。先固定 $s_t$ 对 $a_t$ 求期望：

$$
\mathbb{E}_{a_t \sim \pi_\phi}\big[ \nabla_\phi \log \pi_\phi(a_t \mid s_t) \, b(s_t) \big]
= b(s_t) \sum_a \pi_\phi(a \mid s_t) \, \nabla_\phi \log \pi_\phi(a \mid s_t)
= b(s_t) \, \nabla_\phi \sum_a \pi_\phi(a \mid s_t) = b(s_t) \, \nabla_\phi 1 = 0
$$

再对 $s_t$ 求期望仍然是 0。

第一个等号把 $b(s_t)$ 提到了对动作的求和外面，这要求 $b$ 不依赖 $a_t$。
如果 $b$ 依赖动作，它留在求和里面，$\sum_a \nabla_\phi \pi_\phi(a \mid s_t) \, b(s_t, a)$ 一般不为零，估计就有偏了。

</details>

### 第 5 题

一个三步的回合，奖励是 $(2, -1, 3)$，$\gamma = 0.5$。求每一步的 reward-to-go。

<details><summary>参考答案</summary>

从后往前用 $G_t = r_t + \gamma G_{t+1}$：

$G_2 = 3$

$G_1 = -1 + 0.5 \times 3 = 0.5$

$G_0 = 2 + 0.5 \times 0.5 = 2.25$

</details>

### 第 6 题

$\gamma = 0.9$，critic 给出 $V(s_t) = 3$，$V(s_{t+1}) = 4$，这一步的奖励是 0.5。分三种情况求 TD 误差：
正常的一步；这一步之后环境终止（`terminated`）；这一步之后因为到达步数上限被截断（`truncated`）。

<details><summary>参考答案</summary>

正常的一步：$\delta = 0.5 + 0.9 \times 4 - 3 = 1.1$。

终止：终止状态的价值为 0，$\delta = 0.5 + 0 - 3 = -2.5$。

截断：环境本来还能继续，照常自举，$\delta = 0.5 + 0.9 \times 4 - 3 = 1.1$。

同一个转移，把截断误当成终止，TD 误差从 $+1.1$ 变成 $-2.5$，符号都反了。

</details>

### 第 7 题

一个两步的轨迹片段（在回合中途截断，没有终止）。奖励是 $(1, 1)$，$V(s_0) = 1.0$，$V(s_1) = 0.5$，$V(s_2) = 2.0$。
取 $\gamma = 1$，$\lambda = 0.5$。求两步的 GAE 优势和 critic 的回归目标。再求 $\lambda = 0$ 和 $\lambda = 1$ 时的 $\hat{A}_0$。

<details><summary>参考答案</summary>

TD 误差：

$\delta_0 = 1 + 0.5 - 1.0 = 0.5$

$\delta_1 = 1 + 2.0 - 0.5 = 2.5$

$\lambda = 0.5$，$\gamma\lambda = 0.5$，从后往前：

$\hat{A}_1 = 2.5$

$\hat{A}_0 = 0.5 + 0.5 \times 2.5 = 1.75$

critic 的目标是优势加上价值：$(1.75 + 1.0, \; 2.5 + 0.5) = (2.75, 3.0)$。

$\lambda = 0$：$\hat{A}_0 = \delta_0 = 0.5$。

$\lambda = 1$：$\hat{A}_0 = 0.5 + 2.5 = 3.0$。验证：两步回报是 $1 + 1 + V(s_2) = 4$，减去 $V(s_0) = 1$ 得 3。

</details>

### 第 8 题

GAE 和 Phase 4 的 λ-return 是什么关系？$\lambda = 0$ 和 $\lambda = 1$ 时 GAE 分别退化成什么，各自的偏差和方差如何？

<details><summary>参考答案</summary>

$\hat{A}_t^{\mathrm{GAE}} = G_t^\lambda - V(s_t)$。两者都是对 n 步估计做权重为 $(1 - \lambda)\lambda^{n-1}$ 的加权平均，
只差一个基线 $V(s_t)$。实现时算出 GAE 再加回 $V(s_t)$，就是 critic 的 λ-return 目标。

$\lambda = 0$：$\hat{A}_t = \delta_t$，一步 TD 优势。方差小；critic 不准时有偏。

$\lambda = 1$：$\hat{A}_t = G_t - V(s_t)$，蒙特卡洛优势。无偏（不管 critic 准不准）；方差大。

</details>

### 第 9 题

PPO 的裁剪范围取 $\epsilon_{\text{clip}} = 0.1$。四个样本的（优势，概率比）分别是
$(1, 1.3)$、$(-1, 1.3)$、$(-1, 0.7)$、$(1, 0.95)$。求每个样本的裁剪目标值，指出哪些样本还有梯度，并求这批样本的裁剪目标。

<details><summary>参考答案</summary>

裁剪范围是 $[0.9, 1.1]$。

| 样本 | 未裁剪项 | 裁剪项 | $\min$ | 有梯度吗 |
| --- | --- | --- | --- | --- |
| $(1, 1.3)$ | 1.3 | 1.1 | 1.1 | 没有 |
| $(-1, 1.3)$ | $-1.3$ | $-1.1$ | $-1.3$ | 有 |
| $(-1, 0.7)$ | $-0.7$ | $-0.9$ | $-0.9$ | 没有 |
| $(1, 0.95)$ | 0.95 | 0.95 | 0.95 | 有 |

裁剪目标是 $(1.1 - 1.3 - 0.9 + 0.95) / 4 = -0.0375$。

第一个和第三个样本已经朝有利的方向走出了范围，梯度被裁掉。
第二个样本是坏动作的概率反而涨了，梯度保留，会把它拉回来。

</details>

### 第 10 题

设 $x \sim \mathcal{N}(\mu, \sigma^2)$，$f(x) = x$。真实梯度 $\partial \mathbb{E}[f] / \partial \mu = 1$。
分别推导 score-function 和 pathwise 估计器的方差。再用正文里 $f(x) = x^2$ 的公式，算出 $\mu = 2$、$\sigma = 0.5$ 时三种估计器
（不带基线的 SF、带基线的 SF、PW）的方差。

<details><summary>参考答案</summary>

**$f(x) = x$。** 令 $x = \mu + \sigma\epsilon$。

score-function：$\hat{g} = x \cdot \epsilon / \sigma = \mu\epsilon/\sigma + \epsilon^2$。
均值是 $0 + 1 = 1$，无偏。
二阶矩是 $\mu^2/\sigma^2 \cdot \mathbb{E}[\epsilon^2] + 2(\mu/\sigma)\,\mathbb{E}[\epsilon^3] + \mathbb{E}[\epsilon^4] = \mu^2/\sigma^2 + 3$。
方差是 $\mu^2/\sigma^2 + 3 - 1 = \mu^2/\sigma^2 + 2$。

pathwise：$\hat{g} = f'(x) \cdot 1 = 1$，是常数。方差为 0。

$f$ 是线性函数时，pathwise 一个样本就得到精确的梯度；score-function 的方差仍然随 $\sigma$ 变小而爆炸。

**$f(x) = x^2$，$\mu = 2$，$\sigma = 0.5$。** 真实梯度是 4。

不带基线的 SF：$\mu^4/\sigma^2 + 14\mu^2 + 15\sigma^2 = 16/0.25 + 56 + 3.75 = 123.75$

带基线的 SF：$8\mu^2 + 10\sigma^2 = 32 + 2.5 = 34.5$

PW：$4\sigma^2 = 1$

</details>

### 第 11 题

$x$ 是伯努利变量，取 1 的概率是 $p$。对下面两个函数，分别求真实梯度 $\partial \mathbb{E}[f(x)] / \partial p$ 和 straight-through 估计的期望，判断是否有偏。

1. $f(x) = 3x + 1$
2. $f(x) = x^3$，取 $p = 0.5$

<details><summary>参考答案</summary>

真实梯度都是 $f(1) - f(0)$。straight-through 的估计是 $f'(x)$，期望是 $p f'(1) + (1 - p) f'(0)$。

1. 真实梯度 $4 - 1 = 3$。$f'(x) = 3$，估计恒为 3。无偏，而且方差为 0。线性函数下 straight-through 是精确的。
2. 真实梯度 $1 - 0 = 1$。$f'(x) = 3x^2$，$f'(1) = 3$，$f'(0) = 0$，期望是 $3p = 1.5$。有偏，偏高 0.5。

函数弯得越厉害，端点处的导数和两点之间的差就越不一致，偏差越大。

</details>

### 第 12 题

三个动作，$Q = (0, 1, 2)$，温度 $\alpha = 1$。求最优的最大熵策略和软价值。
已知 $e = 2.718$，$e^2 = 7.389$，$\ln 11.107 = 2.408$。如果 $\alpha \to 0$，策略和软价值分别趋于什么？

<details><summary>参考答案</summary>

$\exp(Q / \alpha) = (1, 2.718, 7.389)$，和为 11.107。

策略：$(1/11.107, \; 2.718/11.107, \; 7.389/11.107) = (0.090, 0.245, 0.665)$。

软价值：$V = \alpha \log \sum_a \exp(Q(a)/\alpha) = \ln 11.107 = 2.408$。它大于 $\max Q = 2$，多出来的是熵奖金。

$\alpha \to 0$ 时，策略趋于把全部概率放在 $Q$ 最大的动作上（贪心），软价值趋于 $\max_a Q(a) = 2$。
最大熵强化学习退回普通的强化学习。

</details>

### 第 13 题

tanh 高斯策略，一维，$\mu = 0$，$\sigma = 1$，采到 $u = 1.0$。求动作 $a$ 和 $\log \pi(a)$。
已知 $\tanh(1) = 0.7616$，$\ln 0.42 = -0.868$，$\frac{1}{2}\ln(2\pi) = 0.919$。如果漏掉 tanh 的修正项，$\log \pi$ 是偏大还是偏小？

<details><summary>参考答案</summary>

$a = \tanh(1.0) = 0.7616$。

高斯部分：$\log \mathcal{N}(1.0; 0, 1) = -0.5 - 0.919 = -1.419$。

修正项：$1 - \tanh^2(1) = 1 - 0.580 = 0.420$，$\log 0.420 = -0.868$。

$\log \pi(a) = -1.419 - (-0.868) = -0.551$。

漏掉修正项会得到 $-1.419$，偏小。$\log \pi$ 偏小意味着熵被高估，在动作接近边界时尤其严重（那里修正项的绝对值很大）。
策略会误以为"往边界走能增加熵"，输出容易饱和。

</details>

### 第 14 题

SAC 用在一个二维动作的任务上，目标熵按经验规则设置。某个批次里 $\log \pi$ 的平均值是 1.2。
目标熵是多少？当前熵是多少？这一步 $\alpha$ 会增大还是减小？

<details><summary>参考答案</summary>

目标熵是动作维度的相反数：$\bar{\mathcal{H}} = -2$。

当前熵是 $-\mathbb{E}[\log \pi] = -1.2$，高于目标熵。

温度损失对 $\alpha$ 的梯度是 $\mathcal{H}_{\text{当前}} - \bar{\mathcal{H}} = -1.2 - (-2) = 0.8 > 0$，梯度下降让 $\alpha$ 减小。
策略已经比要求的更随机了，熵奖金应该降价。

</details>

### 第 15 题

模型的单步误差 $\varepsilon = 0.02$，$L = 1.1$。用 Phase 1 的复合误差上界，求误差上界不超过 0.1 的最大展开长度 $k$。
已知 $1.1^4 = 1.464$，$1.1^5 = 1.611$。再用一句话说明 MBPO 为什么不需要长展开也能学到长期的行为。

<details><summary>参考答案</summary>

上界是 $e_k \le 0.02 \times (1.1^k - 1) / 0.1 = 0.2 \times (1.1^k - 1)$。

$k = 4$：$0.2 \times 0.464 = 0.093 \le 0.1$。

$k = 5$：$0.2 \times 0.611 = 0.122 > 0.1$。

最大的 $k$ 是 4。

不需要长展开的原因：分支的起点取自回放池里回合各个位置的真实状态，覆盖靠起点；
长期的回报由 critic 的自举沿着一条条短的转移往回传，不靠展开的长度。

</details>

### 第 16 题

沿用第 11.3 节的设定：模型 $\hat{s}' = \hat{s} + a$，策略 $a = \phi \, \hat{s}$，到达 $\hat{s}'$ 的奖励是 $-\hat{s}'^2$，不折扣，想象 2 步。
这次从 $s_0 = 2$ 出发，$\phi = -0.5$。用反向传播的方式手算 $dJ/d\phi$，并用闭式验证。

<details><summary>参考答案</summary>

前向：$\hat{s}_1 = 2 + (-0.5)(2) = 1$，$\hat{s}_2 = 1 + (-0.5)(1) = 0.5$，$J = -1 - 0.25 = -1.25$。

反向：

$dJ/d\hat{s}_2 = -2 \times 0.5 = -1$

$dJ/d\hat{s}_1 = -2 \times 1 + (-1) \times (1 + \phi) = -2 - 0.5 = -2.5$

$dJ/d\phi = (dJ/d\hat{s}_1) \cdot s_0 + (dJ/d\hat{s}_2) \cdot \hat{s}_1 = (-2.5)(2) + (-1)(1) = -6$

闭式：$J = -(1 + \phi)^2 s_0^2 - (1 + \phi)^4 s_0^2$，

$dJ/d\phi = -2(1 + \phi) s_0^2 - 4(1 + \phi)^3 s_0^2 = -2(0.5)(4) - 4(0.125)(4) = -4 - 2 = -6$。

一致。

</details>

### 第 17 题

下面三种情况各应该用哪种梯度估计器？说明理由。

1. 连续动作，critic 是一个神经网络，要更新 actor
2. 离散动作，在真实环境里采的轨迹，要更新策略
3. 世界模型里有一个类别隐变量，采样出的 one-hot 向量送进一个神经网络解码器，要把重建损失的梯度传回编码器

<details><summary>参考答案</summary>

1. pathwise。动作连续，可以重参数化；critic 对动作可微。梯度穿过 critic，方差小。这是 DDPG、SAC 的做法。
2. score-function（配合基线或优势）。动作离散，不能重参数化；真实环境不可微，梯度穿不过去。只能把回报当黑盒。这是 REINFORCE、PPO 的做法。
3. straight-through。变量是离散的，不能重参数化；下游的解码器可微，可以提供梯度；
   隐变量维度高，score-function 的方差太大。接受一点偏差换取低方差。这是 Dreamer V2、V3 的做法。

</details>

### 第 18 题

TD3 和 SAC 都训练两个 critic 并在算目标时取较小的那个。这样做会让价值估计偏低。为什么偏低可以接受，偏高不行？

<details><summary>参考答案</summary>

因为两种偏差的后果不对称。

偏高会自我强化：actor 专门找 critic 给分高的动作，被高估的动作会被选中；
它的虚高的值通过自举进入其他状态的目标，高估在状态之间传播并累积，最终 $Q$ 值发散或策略崩溃。

偏低不会自我强化：被低估的动作只是不被 actor 选中，错误的值不会通过策略的选择传播出去。
代价只是策略偏保守，可能错过一些其实不错的动作。

这和 Phase 1 里"规划器专挑模型预测偏好的地方走"是同一个道理：只要有一个优化器在对带噪声的估计求最大，
就应该让估计偏向悲观。

</details>

---

## 17. 小结与下章预告

### 17.1 本章要点

1. **策略梯度定理。** 期望回报的梯度可以写成 $\mathbb{E}[\sum_t \nabla_\phi \log \pi_\phi(a_t \mid s_t) \, \Psi_t]$。
   $\nabla \log \pi$ 来自对数导数技巧；环境的转移概率在求导时消失，所以不需要模型。
2. **方差是主要矛盾。** 权重 $\Psi_t$ 从整条轨迹的回报，到 reward-to-go，到减去基线，到优势，
   每一步都在不改变（或少量改变）期望的前提下降低方差。减去只依赖状态的基线不引入偏差。
3. **Actor-Critic 与 GAE。** 用 critic 自举来截短对未来的求和，用偏差换方差。
   GAE 是 TD 误差的指数加权和 $\sum_l (\gamma\lambda)^l \delta_{t+l}$，等于 λ-return 减去 $V(s_t)$。
4. **PPO。** 用概率比复用一批数据，用裁剪去掉"走得更远"的收益，保留"走错了"的惩罚。它是本课的无模型在策略基线。
5. **三种梯度估计器。** score-function 只用函数值，无偏、方差大、什么情况都能用；
   pathwise 用函数的导数，无偏、方差通常小，要求连续且可微；straight-through 对付离散变量，有偏、方差小。
   在 $f(x) = x^2$ 的高斯例子里，前两者的方差分别是 $\mu^4/\sigma^2 + 14\mu^2 + 15\sigma^2$ 和 $4\sigma^2$。
6. **DDPG、TD3、SAC。** 让梯度穿过可微的 critic 来更新 actor，可以离策略。
   SAC 在此之上把熵写进目标，用 tanh 高斯策略、两个 critic 取小、自动调节温度。它是本课的无模型离策略基线。
7. **MBPO。** 从真实状态出发做短的分支展开，用模型数据训练 SAC。短是因为复合误差，够用是因为 critic 的自举。
8. **沿模型反向传播。** 模型可微时，想象回报对策略参数可以直接求导。
   梯度和模型误差由同一个雅可比连乘决定，所以视野要短，末端接价值函数。这是 Dreamer V1 的行为学习方式。

### 17.2 一张图回顾

```
      问题                           本章的办法                        留下的问题
  ───────────────────────────────────────────────────────────────────────────────────
   没有动作的标签       ──▶   策略梯度：用回报给 ∇log π 加权    ──▶   方差太大

   方差太大             ──▶   基线、优势、critic 自举、GAE      ──▶   critic 不准时有偏；
                                                                    数据只能用一次

   数据只能用一次       ──▶   PPO：限制步长，复用当前批次       ──▶   仍然是在策略，费样本
                             DDPG / SAC：梯度穿过 critic，离策略

   探索靠外挂的噪声     ──▶   SAC：把熵写进目标                 ──▶   仍需几十万步真实交互

   真实交互太贵         ──▶   MBPO：模型的短分支生成数据        ──▶   没用上模型的导数；
                                                                    只能处理低维状态

   模型是可微的         ──▶   沿模型反向传播                    ──▶   输入是图像时怎么办？
                                                                    离散动作怎么办？
                                                                    （Phase 6）
```

### 17.3 下章预告

到这里，Dreamer 需要的零件都齐了：

| 零件 | 来自 |
| --- | --- |
| 从图像学到的、可以在里面展开的隐状态模型（RSSM） | Phase 3 |
| λ-return 作为价值目标 | Phase 4；本章第 5 节 |
| actor-critic 与熵正则 | 本章第 4、9 节 |
| 从真实数据的状态出发做短展开 | 本章第 10 节 |
| 沿可微的模型反向传播 | 本章第 11 节 |
| 三种梯度估计器 | 本章第 7 节 |

本章的两个基于模型的方法各有一个没解决的问题。
MBPO 和第 11 节的方法都假设环境直接给出低维状态，模型在这个状态空间里预测。
机器人拿到的是图像，在像素空间里做分支展开或反向传播，既算不动也没有必要。
另外，MBPO 把模型当黑盒，浪费了导数；沿模型反传用上了导数，却只适用于连续动作和可微的奖励。

[Phase 6](../phase-06-dreamer/CONCEPT.md) 的 Dreamer 把这些拼到一起：
把 Phase 3 的 RSSM 当作模型，在它的隐空间里从回放数据对应的隐状态出发想象十几步，
在想象出的轨迹上训练 actor 和 critic，全程不解码成图像。三代 Dreamer 的差别很大一部分就落在本章的内容上：
V1 的 actor 沿模型反向传播；V2 把隐变量换成类别变量并用 straight-through，离散动作改用 score-function；
V3 加上一组让同一套超参数适用于各种任务的稳健性技巧。

排行榜也会继续：Phase 6 加入 Dreamer，[Phase 7](../phase-07-latent-planning/CONCEPT.md) 加入 TD-MPC2。
后者把 Phase 1 的 MPPI 规划器、本章的 SAC 式策略和价值函数合在一个隐空间模型里。

---

## 附录 A：术语表（中英对照）

| 中文 | 英文 | 一句话解释 |
| --- | --- | --- |
| 策略梯度 | policy gradient | 期望回报对策略参数的梯度，以及基于它的一类算法 |
| 策略梯度定理 | policy gradient theorem | 把策略梯度写成 $\nabla \log \pi$ 乘以动作价值的期望 |
| 对数导数技巧 | log-derivative trick | $\nabla p = p \, \nabla \log p$，把对分布的求导变成该分布下的期望 |
| 得分函数 | score function | 对数概率对参数的梯度 $\nabla_\phi \log p_\phi(x)$ |
| 随机策略 / 确定性策略 | stochastic / deterministic policy | 输出动作的分布 / 直接输出一个动作 |
| 代理损失 | surrogate loss | 梯度正确但数值本身没有意义的辅助损失 |
| 在策略 / 离策略 | on-policy / off-policy | 训练数据是否必须由当前策略采集 |
| 信用分配 | credit assignment | 判断回报应该归功于哪些动作 |
| 基线 | baseline | 从回报里减去的、只依赖状态的量，用来降方差 |
| 优势函数 | advantage function | $Q^\pi(s, a) - V^\pi(s)$，动作比平均水平好多少 |
| 演员-评论家 | actor-critic | 策略网络（actor）加价值网络（critic）的结构 |
| 自举 | bootstrapping | 用价值函数自己的估计来构造学习目标 |
| TD 误差 | TD error | $r_t + \gamma V(s_{t+1}) - V(s_t)$ |
| 熵 | entropy | 分布随机程度的度量 |
| 熵正则 | entropy regularization | 在损失里奖励策略的熵，防止过早变确定 |
| 广义优势估计 | Generalized Advantage Estimation (GAE) | TD 误差的指数加权和，等于 λ-return 减去基线 |
| 重要性采样 | importance sampling | 用一个分布的样本估计另一个分布下的期望，样本乘以概率比 |
| 概率比 | probability ratio | 新旧策略对同一动作的概率之比 |
| 近端策略优化 | Proximal Policy Optimization (PPO) | 用裁剪的代理目标限制每轮更新幅度的在策略算法 |
| 裁剪目标 | clipped objective | PPO 的目标函数，对概率比裁剪后取较小者 |
| 梯度估计器 | gradient estimator | 用样本估计"期望对分布参数的梯度"的方法 |
| score-function 估计器 | score-function estimator | 用函数值乘以得分函数；又叫 REINFORCE 估计器、似然比估计器 |
| pathwise 估计器 | pathwise estimator | 把样本写成噪声的可微函数，直接对被积函数求导；又叫重参数化梯度 |
| 重参数化 | reparameterization | 把随机性移到一个不含参数的噪声变量上 |
| straight-through 估计器 | straight-through estimator | 前向用离散样本，反向把采样当作恒等映射 |
| 偏差 / 方差 | bias / variance | 估计的期望偏离真值多少 / 估计在不同样本间波动多大 |
| 确定性策略梯度 | Deterministic Policy Gradient (DPG) | 梯度穿过 critic 对动作的导数来更新确定性策略 |
| 深度确定性策略梯度 | Deep Deterministic Policy Gradient (DDPG) | DPG 加回放池和目标网络 |
| 目标网络 | target network | 缓慢更新的网络副本，用来计算稳定的学习目标 |
| 软更新 | soft update / Polyak averaging | 目标网络每步向在线网络靠近一小步 |
| 双 Q 取小 | clipped double Q-learning | 两个 critic 取较小者来计算目标，抑制高估 |
| 目标策略平滑 | target policy smoothing | 算目标时给目标动作加噪声，防止利用 critic 的尖峰 |
| 最大熵强化学习 | maximum entropy RL | 目标里同时包含回报和策略的熵 |
| 软演员-评论家 | Soft Actor-Critic (SAC) | 最大熵目标下的离策略 actor-critic |
| 温度 | temperature | SAC 中熵奖金相对任务奖励的权重 $\alpha$ |
| 目标熵 | target entropy | 自动调节温度时希望策略保持的熵 |
| 软价值 | soft value | 包含未来熵奖金的价值函数 |
| tanh 高斯策略 | tanh-Gaussian / squashed Gaussian policy | 高斯样本经过 tanh 压到有界范围的策略 |
| 分支展开 | branched rollout | 从回放池里的真实状态出发，在模型里展开很少几步 |
| 更新数据比 | update-to-data ratio (UTD) | 每个真实环境步对应多少次梯度更新 |
| 价值梯度 | value gradient | 回报（或价值）对状态、动作的导数，沿模型反向传播得到 |
| 随机价值梯度 | Stochastic Value Gradients (SVG) | 用重参数化沿模型反传价值梯度的一族算法 |
| 沿时间反向传播 | backpropagation through time (BPTT) | 把按时间展开的计算图当作一个深网络来反向传播 |
| 终端价值 | terminal value | 加在想象轨迹末端的价值估计，概括视野之外的回报 |

---

## 附录 B：延伸阅读

**必读**

- Sutton & Barto, Reinforcement Learning: An Introduction（第 2 版）第 13 章。策略梯度定理、REINFORCE、带基线的版本、actor-critic。
  对照时注意教材的奖励下标比本课大 1，策略参数记作 $\theta$。
- Janner et al. 2019, MBPO（arXiv:1906.08253）。先读引言、算法和实验，理论部分抓住结论的形状。
- Haarnoja et al. 2018, SAC（arXiv:1801.01290）及其后续（arXiv:1812.05905）。实现以后一篇为准。

**精读其中的一部分**

- Schulman et al. 2016, GAE（arXiv:1506.02438）。读第 2、3 节。
- Schulman et al. 2017, PPO（arXiv:1707.06347）。全文很短，读第 3、5 节和附录的超参数表。
- Mohamed et al. 2020, Monte Carlo Gradient Estimation in Machine Learning（arXiv:1906.10652）。
  只读 score-function 和 pathwise 两节。
- Heess et al. 2015, SVG（arXiv:1510.09142）。读方法部分，看价值梯度的递推。

**选读**

- Lillicrap et al. 2016, DDPG（arXiv:1509.02971）；Fujimoto et al. 2018, TD3（arXiv:1802.09477）。
  TD3 的前半部分对高估的分析值得读。
- Bengio et al. 2013, Straight-Through（arXiv:1308.3432）；Jang et al. 2017, Gumbel-Softmax（arXiv:1611.01144）。
  离散变量的两种梯度近似，了解即可。

**实现参考**（下面几项不在本章 README 的论文表里，只给标题，不写编号）

- OpenAI Spinning Up in Deep RL。VPG、PPO、DDPG、TD3、SAC 的讲解和参考实现，记号清楚，适合和本章对照着读。
- Huang et al. 2022, The 37 Implementation Details of Proximal Policy Optimization（ICLR Blog Track）。
  写自己的 PPO 时逐条对照。
- Engstrom et al. 2020, Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO。
- Andrychowicz et al. 2020, What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study。
- Henderson et al. 2018, Deep Reinforcement Learning that Matters。做排行榜之前读。

**预习**

- Hafner et al. 2020, Dreamer V1（arXiv:1912.01603）。读引言和行为学习那一节，
  对照本章第 5、11 节，看 λ-return 和沿模型反向传播怎样在隐空间里合到一起。Phase 6 精读。

