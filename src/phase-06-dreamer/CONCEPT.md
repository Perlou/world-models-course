# Dreamer：在想象中学习行为

> Phase 6 · 概念精讲 ｜ 对应讲义：`src/phase-06-dreamer/` ｜ 预计阅读：7–8 小时
>
> 这一章把前五章的东西装到一起：用 Phase 3 的 RSSM 当世界模型，在它的隐空间里批量想象轨迹，
> 用 Phase 4 的 λ-return 当价值目标，用 Phase 5 的 actor-critic 和梯度估计器从想象的轨迹里学出策略。
> 主线是 Dreamer 的三代演进（V1 → V2 → V3），终点是能逐项写出 Dreamer V3 的全部损失，并说清每一个 stop-gradient 为什么放在那里。

---

## 目录

- [0. 本章导读](#0-本章导读)
- [1. 从规划到学习行为：PlaNet 的短板](#1-从规划到学习行为planet-的短板)
- [2. 在想象中展开轨迹](#2-在想象中展开轨迹)
- [3. Dreamer V1：沿可微的世界模型反向传播](#3-dreamer-v1沿可微的世界模型反向传播)
- [4. Dreamer V2：类别隐变量与 straight-through](#4-dreamer-v2类别隐变量与-straight-through)
- [5. KL balancing 与 free bits](#5-kl-balancing-与-free-bits)
- [6. Dreamer V3（一）：symlog 与 two-hot](#6-dreamer-v3一symlog-与-two-hot)
- [7. Dreamer V3（二）：回报归一化与其他稳健性技巧](#7-dreamer-v3二回报归一化与其他稳健性技巧)
- [8. Dreamer V3 世界模型的完整损失](#8-dreamer-v3-世界模型的完整损失)
- [9. Dreamer V3 的 actor 与 critic](#9-dreamer-v3-的-actor-与-critic)
- [10. 诊断与消融：怎么知道哪里出了问题](#10-诊断与消融怎么知道哪里出了问题)
- [11. 方法对比与选型](#11-方法对比与选型)
- [12. 论文精读卡片](#12-论文精读卡片)
- [13. 实现陷阱与调试清单](#13-实现陷阱与调试清单)
- [14. 常见问题（FAQ）](#14-常见问题faq)
- [15. 自测题](#15-自测题)
- [16. 小结与下章预告](#16-小结与下章预告)
- [附录 A：术语表（中英对照）](#附录-a术语表中英对照)
- [附录 B：延伸阅读](#附录-b延伸阅读)

---

## 0. 本章导读

### 0.1 本章要回答的问题

1. PlaNet 已经能从像素做控制了，为什么还要再学一个策略和一个价值函数？
2. 在世界模型里想象出来的轨迹，从哪里开始想、想多远、梯度该流到哪里、不该流到哪里？
3. 训练 actor 有两条路：沿可微的世界模型反向传播（V1），或者用 REINFORCE（V2、V3）。各自什么时候好用？
4. V2 为什么把高斯隐变量换成类别隐变量？KL 项为什么要拆成两半、各带一个 stop-gradient？
5. 奖励的尺度可以从 0.001 到上万。V3 靠哪些办法让同一套超参数在这些任务上都能跑？

### 0.2 一图总览

Dreamer 有三个交替进行的环节。本章每一节讲其中一个环节，或者讲某一代对某个环节的改动。

```
       ┌───────────────────────── ③ 与环境交互 ──────────────────────────┐
       │                                                                 │
       ▼                                                                 │
  ┌──────────┐  (o, a, r, c)   ┌──────────────┐                 ┌─────────────────┐
  │ 真实环境   │ ──────────────▶ │  经验回放 D   │                 │ actor  π_φ(a|m) │
  └──────────┘                 └──────────────┘                 └─────────────────┘
                                      │                                  ▲
                                      │ 采样序列片段 [B, T]                │
                                      ▼                                  │
  ① 学世界模型           ┌──────────────────────────────┐                 │
     （第 4、5、6、8 节）  │ RSSM 世界模型（参数 θ）         │                 │
                        │ 编码器、序列模型、先验           │                 │
                        │ 解码器、奖励头、continue 头     │                 │
                        └──────────────────────────────┘                 │
                                      │                                  │
                                      │ 后验状态 m_t = (h_t, z_t)          │
                                      │ 共 B×T 个，停止梯度                │ 更新 φ
                                      ▼                                  │
  ② 在想象中学行为        ┌──────────────────────────────┐                 │
     （第 2、3、7、9 节）  │ 想象：不看观测，只用先验推 H 步   │                 │
                        │ m̂_0 → m̂_1 → … → m̂_H          │                 │
                        └──────────────────────────────┘                 │
                                      │                                  │
                                      │ 预测的奖励、continue、价值          │
                                      ▼                                  │
                        ┌──────────────────────────────┐                 │
                        │ λ-return  R^λ_0 … R^λ_H      │ ─── actor 损失 ──┘
                        └──────────────────────────────┘
                                      │
                                      └────── critic 损失 ──▶ 更新 critic V_ψ(m)

  第 1 节：为什么要这样做（PlaNet 的短板）       第 10 节：出了问题怎么查
```

三个环节各管一件事：

| 环节 | 输入 | 更新谁 | 承接哪一章 |
| --- | --- | --- | --- |
| ① 学世界模型 | 回放里的真实序列 | 世界模型 $\theta$ | Phase 3 的 RSSM 与序列 ELBO |
| ② 在想象中学行为 | 世界模型想象出的轨迹 | actor $\phi$、critic $\psi$ | Phase 4 的 λ-return；Phase 5 的 actor-critic 与梯度估计器 |
| ③ 与环境交互 | 当前观测 | 经验回放 $\mathcal{D}$ | Phase 1 的"用控制器迭代采集数据" |

### 0.3 前置知识检查

| 需要的知识 | 在哪学过 | 本章哪里用 |
| --- | --- | --- |
| RSSM 的结构：确定性状态 $h_t$、随机状态 $z_t$、先验与后验、`observe()` 与 `imagine()` | [Phase 3](../phase-03-rssm-planet/CONCEPT.md) 的 RSSM | 全章。第 2 节的想象就是 `imagine()` |
| 序列 ELBO；两个高斯之间的 KL；free nats | Phase 3 的序列 ELBO 与 KL | 第 5、8 节：从 ELBO 改写出 V3 的三项损失 |
| 隐空间 CEM 规划（PlaNet） | Phase 3 的 PlaNet；Phase 1 第 7 节 | 第 1 节：拿它当反面对照 |
| 回报、折扣、价值函数、自举、目标网络 | [Phase 4](../phase-04-rl-value-based/CONCEPT.md) 的价值函数与 TD | 第 1、3、7、9 节 |
| λ-return 的递推式 | Phase 4 的 n 步回报与 λ-return | 第 3、9 节：原样搬到想象轨迹上 |
| 分布式价值（C51） | Phase 4 的 DQN 改进 | 第 6 节：two-hot 回归是它的近亲 |
| 策略梯度、基线与优势、熵正则 | [Phase 5](../phase-05-rl-policy-gradient/CONCEPT.md) 的 Actor-Critic | 第 4、9 节 |
| 三种梯度估计器：score-function、pathwise（重参数化）、straight-through | Phase 5 的三种梯度估计器 | 第 3 节（pathwise）、第 4 节（straight-through）、第 9 节（score-function） |
| 沿可微模型反向传播（SVG 思想） | Phase 5 的沿模型反向传播 | 第 3 节：Dreamer V1 的 actor |
| 复合误差、模型被钻空子 | Phase 1 第 5 节、第 8.5 节 | 第 2、10 节 |
| softmax、交叉熵、one-hot | `deep-learning-course` | 第 4、5、6 节 |

如果 λ-return 的递推式已经写不出来，先回看 Phase 4 再读第 3 节。
如果分不清 score-function 和 pathwise 两种估计器各自需要什么条件，先回看 Phase 5 再读第 3、4 节。

### 0.4 概念与讲义对照表

讲义还没写，表里的函数名是规划中的名字，以最终讲义为准。

| 概念文档 | 对应讲义 | 讲义里做什么 | 规划中的核心函数 |
| --- | --- | --- | --- |
| 第 1 节 从规划到学习行为 | `01-from-planet-to-dreamer.py` | 给 PlaNet 的规划算计算量的账；搭出 Dreamer 的三环节骨架 | `planning_cost_per_step()`、`dreamer_training_loop()` |
| 第 2 节 在想象中展开轨迹 | `02-imagination-rollouts.py` | 从后验状态出发批量想象；检查 stop-gradient 的位置 | `imagine_rollout()` |
| 第 3 节 Dreamer V1 | `03-dreamer-v1-actor-critic.py` | λ-return 价值目标；沿模型反传训练 actor | `lambda_return()`、`actor_loss_pathwise()`、`critic_loss_mse()` |
| 第 4 节 Dreamer V2 | `04-categorical-latents-straight-through.py` | 把高斯隐变量换成 32×32 的类别隐变量；straight-through 采样 | `sample_categorical_st()`、`CategoricalRSSM` |
| 第 5 节 KL balancing 与 free bits | `05-kl-balancing-and-free-bits.py` | 类别分布的 KL；两半 KL 各带一个 sg；free bits | `categorical_kl()`、`kl_loss_balanced()` |
| 第 6 节 symlog 与 two-hot | `06-symlog-and-twohot.py` | symlog 变换；two-hot 编码与离散回归头 | `symlog()`、`symexp()`、`twohot_encode()`、`TwoHotHead` |
| 第 7 节 回报归一化与其他技巧 | `07-return-normalization-and-robustness.py` | 分位数回报归一化；unimix；critic 的 EMA 正则 | `ReturnNormalizer`、`unimix_probs()`、`ema_update()` |
| 第 8 节 世界模型的完整损失 | `08-dreamer-v3-world-model.py` | 组装编码器、RSSM、解码器、奖励头、continue 头，写出完整损失 | `WorldModel.observe()`、`WorldModel.loss()` |
| 第 9 节 actor 与 critic | `09-dreamer-v3-agent.py` | 组装完整智能体并训练 | `DreamerV3Agent.train_step()`、`actor_loss_reinforce()`、`critic_loss_twohot()` |
| 第 10 节 诊断与消融 | `10-diagnostics-and-ablations.py` | 想象回报与真实回报是否一致；逐项消融 | `imagined_vs_real_return()`、`run_ablation()` |

建议的学习顺序：第 1–3 节一口气读完（这是 Dreamer 的骨架），跑讲义 01–03；
第 4–7 节每节配一份讲义，都是可以单独验证的小部件；最后读第 8–10 节，跑讲义 08–10 把部件装起来。

### 0.5 符号表

在 CLAUDE.md 的全课程符号约定基础上增补。**参数记号以本课为准**，和各篇 Dreamer 论文的对应关系见后面的对照表。

| 符号 | 含义 |
| --- | --- |
| $o_t$，$a_t$，$r_t$ | 观测、动作、奖励。$r_t$ 是在第 $t$ 步执行 $a_t$ 得到的奖励 |
| $c_t$ | continue 标志。第 $t$ 步的状态是终止状态时为 0，否则为 1 |
| $h_t$，$z_t$ | RSSM 的确定性循环状态；随机隐变量 |
| $m_t$ | 模型状态（model state），$m_t = (h_t, z_t)$。actor、critic 和各个预测头的输入 |
| $\theta$，$\phi$，$\psi$ | 世界模型、actor、critic 的参数 |
| $\bar\psi$ | critic 参数的指数滑动平均（EMA） |
| $f_\theta$ | 序列模型（GRU）：$h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})$ |
| $q_\theta(z_t \mid h_t, o_t)$ | 后验（编码器）：看了当前观测之后对 $z_t$ 的推断 |
| $p_\theta(z_t \mid h_t)$ | 先验（动力学预测器）：不看当前观测，只凭历史预测 $z_t$ |
| $\pi_\phi(a \mid m)$，$V_\psi(m)$ | actor；critic 给出的价值（分布式 critic 时是它的期望） |
| $t$，$T$，$B$ | 回放序列里的时间步；序列片段长度；批大小 |
| $k$，$H$ | 想象轨迹里的步数，$k = 0, 1, \dots, H$；想象视野 |
| $\hat m_k$，$\hat a_k$，$\hat r_k$，$\hat c_k$ | 想象轨迹上的模型状态、动作、预测奖励、预测的 continue 概率 |
| $v_k$ | 想象轨迹上的价值，$v_k = V_\psi(\hat m_k)$ |
| $R^\lambda_k$ | 想象轨迹第 $k$ 步的 λ-return |
| $w_k$ | 想象轨迹第 $k$ 步的权重（到这一步回合还没结束的概率） |
| $\gamma$，$\lambda$ | 折扣因子；λ-return 的混合系数 |
| $N_z$，$K$ | 类别隐变量的个数；每个变量的类别数。Dreamer V2、V3 都是 $32 \times 32$ |
| $\ell$ | logits（softmax 之前的输出） |
| $\beta_{\text{pred}}$，$\beta_{\text{dyn}}$，$\beta_{\text{rep}}$ | V3 世界模型三项损失的权重 |
| $\mathrm{symlog}$，$\mathrm{symexp}$ | 对称对数变换及其逆变换（第 6 节） |
| $b_1 < \dots < b_M$ | two-hot 回归用的 $M$ 个桶（bin）的位置 |
| $S$ | 回报的尺度：想象回报的 95% 分位数减 5% 分位数，再做 EMA（第 7 节） |
| $\eta$ | actor 损失里熵正则的系数 |
| $\mathcal{H}[\pi]$ | 分布 $\pi$ 的熵。用花体是为了和视野 $H$ 区分 |
| $\rho$ | V2 的 actor 损失里 REINFORCE 项所占的比例 |
| $\mathrm{sg}(\cdot)$ | 停止梯度：前向照常计算，反向时当作常数 |
| `[B, T, D]` | 张量形状：批大小、时间长度、特征维度 |

**和论文记号的对应。** 三篇 Dreamer 论文的记号彼此也不一样，读论文时对照这张表：

| 本课 | Dreamer V1（arXiv:1912.01603） | Dreamer V2（arXiv:2010.02193） | Dreamer V3（arXiv:2301.04104） |
| --- | --- | --- | --- |
| 观测 $o_t$ | $o_t$ | $x_t$ | $x_t$ |
| 模型状态 $m_t = (h_t, z_t)$ | $s_t$ | $(h_t, z_t)$ | $s_t = \lbrace h_t, z_t \rbrace$ |
| 世界模型参数 $\theta$ | $\theta$ | $\phi$ | $\phi$ |
| actor 参数 $\phi$ | $\phi$ | $\psi$ | $\theta$ |
| critic 参数 $\psi$ | $\psi$ | $\xi$ | $\psi$ |
| continue 标志 $c_t$ | 主实验不需要（连续控制任务不提前终止） | 折扣预测 $\gamma_t$ | $c_t$ |
| λ-return $R^\lambda_k$ | $\mathrm{V}_\lambda(s_\tau)$ | $V^\lambda_t$ | $R^\lambda_t$ |
| 想象步数 $k$ | $\tau$ | $t$ | $t$ |

两个容易看错的地方：

- **论文里的 $s_t$ 是模型状态，不是环境的真实状态。** 本课的 $s_t$ 留给环境的真实状态，所以模型状态另用 $m_t$。
- **奖励的下标差一位。** Dreamer 论文把"到达第 $t$ 步时拿到的奖励"记作 $r_t$，奖励头写成 $p(r_t \mid h_t, z_t)$。
  按本课的约定，那是执行 $a_{t-1}$ 得到的奖励，记作 $r_{t-1}$。所以本章的奖励头写成 $p_\theta(r_{t-1} \mid h_t, z_t)$。
  这不是吹毛求疵：奖励和状态错开一位是实现 Dreamer 时最常见的 bug 之一（见第 13 节）。

---

## 1. 从规划到学习行为：PlaNet 的短板

### 1.1 直觉

Phase 3 的 PlaNet 每走一步都做一次 CEM 规划：采样上千条动作序列，在 RSSM 里各推十几步，
挑出好的，重复几轮，最后只执行第一个动作。下一步，全部重来。

对写后端的人来说，这相当于**每个请求都做一次全表扫描**。结果是对的，但是：

- 慢。每个请求的代价都是完整的一次搜索。
- 不积累。上一个请求的搜索结果，对下一个请求几乎没有帮助（热启动只复用了一条序列）。
- 看不远。扫描的范围是固定的（规划视野），范围之外的数据根本没看。

工程上的解法是**把在线的计算挪到离线**：提前建好索引，在线只查一次。Dreamer 做的是同一件事：

- **策略**（actor）是索引：给一个状态，直接查出该做什么。一次前向计算。
- **价值函数**（critic）是物化视图：给一个状态，直接查出"从这里往后总共能拿多少回报"，不用真的往后展开。

建索引需要数据。Dreamer 的数据来自世界模型自己的想象：在 RSSM 里批量推演轨迹，用这些轨迹训练 actor 和 critic。
真实环境里的每一步交互只用来训练世界模型。这种"把搜索的结果存进一个网络"的做法叫**摊销**（amortization）。

### 1.2 PlaNet 的三个短板

| 短板 | 具体表现 | Dreamer 的对策 |
| --- | --- | --- |
| 决策慢 | 每个真实时间步要做"候选数 × 视野 × 迭代轮数"次隐空间转移 | 学一个 actor，决策只需一次前向计算 |
| 看不远 | 规划视野之外的奖励完全不计入分数 | 学一个 critic，在想象的末端补上视野之外的回报 |
| 不积累 | 每一步的规划结果用完就丢，下一步从头搜索 | actor 和 critic 的参数就是积累下来的规划结果 |

第二条值得展开。Phase 1 第 6.6 节讲过，规划视野受两头挤压：短了看不到远处的奖励，长了模型误差累积。
PlaNet 只能在两者之间折中。有了价值函数，这个矛盾被拆开了：
**模型只负责短视野之内的推演，视野之外交给价值函数。** 模型不用推很远，所以误差累积有限；
远处的回报由价值函数概括，所以也不短视。

### 1.3 定义：三个部件，三个环节

Dreamer 的智能体由三个网络组成：

| 部件 | 记号 | 输入 → 输出 | 从哪学 |
| --- | --- | --- | --- |
| 世界模型 | 参数 $\theta$ | 观测与动作的序列 → 模型状态 $m_t$；以及下一状态、奖励、是否继续的预测 | 回放里的真实序列 |
| actor | $\pi_\phi(a \mid m)$ | 模型状态 → 动作分布 | 想象出的轨迹 |
| critic | $V_\psi(m)$ | 模型状态 → 从这个状态起的期望回报 | 想象出的轨迹 |

注意 actor 和 critic 的输入是模型状态 $m_t = (h_t, z_t)$，不是图像。
图像只进编码器；编码器之后的一切都发生在隐空间里。这带来两个好处：
想象时不需要生成像素，所以很快；actor 和 critic 是小 MLP，所以好训练。

三个环节交替进行（第 0.2 节的图）：

1. **学世界模型**：从回放里采样真实的序列片段，训练 RSSM 和各个预测头。
2. **在想象中学行为**：从刚才那批序列的后验状态出发，在世界模型里想象 $H$ 步，训练 actor 和 critic。
3. **与环境交互**：用当前的 actor 在真实环境里走若干步，把数据存进回放。

第 1、2 步在同一次梯度更新里完成（共用一批数据），第 3 步和它们按一定比例穿插。

这是 Dyna 结构（Phase 4）的直接后代："真实经验训练模型，模型生成的经验训练策略和价值"。
和 Dyna-Q 相比变了两点：模型换成了从像素学出来的 RSSM；想象发生在隐空间里，而且是成批的多步展开。

### 1.4 推导：只看视野之内，丢掉了多少回报

先把"看不远"量化。从第 $t$ 步起的折扣回报（Phase 4 的定义）是：

$$
G_t = \sum_{j=0}^{\infty} \gamma^j r_{t+j}
$$

**第一步：把求和在第 $H$ 项处切开。**

$$
G_t = \sum_{j=0}^{H-1} \gamma^j r_{t+j} + \sum_{j=H}^{\infty} \gamma^j r_{t+j}
$$

**第二步：第二个求和里换元 $j = H + i$，提出公因子 $\gamma^H$。**

$$
\sum_{j=H}^{\infty} \gamma^j r_{t+j} = \gamma^H \sum_{i=0}^{\infty} \gamma^i r_{t+H+i} = \gamma^H G_{t+H}
$$

最后一个等号用的是回报的定义，只是起点换成了 $t + H$。

**第三步：两边取期望，用价值函数的定义 $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$。**

$$
V^\pi(s_t) = \mathbb{E}_\pi\Big[ \underbrace{\sum_{j=0}^{H-1} \gamma^j r_{t+j}}_{\text{视野之内：靠模型展开}} + \underbrace{\gamma^H \, V^\pi(s_{t+H})}_{\text{视野之外：靠价值函数}} \Big]
$$

这就是 $H$ 步回报（Phase 4）。它是一个恒等式：只要 $V^\pi$ 是准的，展开 $H$ 步再接上价值，和展开无穷步完全等价。

**第四步：PlaNet 丢掉的是哪一部分。** 只做规划、没有价值函数，相当于把第二项当成 0。
设单步奖励的绝对值不超过 $r_{\max}$，丢掉的部分最多是：

$$
\gamma^H \, \vert G_{t+H} \vert \le \gamma^H \sum_{i=0}^{\infty} \gamma^i r_{\max} = \frac{\gamma^H}{1 - \gamma} \, r_{\max}
$$

这里用了等比数列求和 $\sum_{i \ge 0} \gamma^i = 1/(1-\gamma)$。
而整个回报的上界是 $r_{\max} / (1 - \gamma)$。两者相除：

$$
\frac{\text{视野之外的权重}}{\text{全部权重}} = \gamma^H
$$

**数值例子。** 按两组参数各算一次：

| 折扣 $\gamma$ | 视野 $H$ | $\gamma^H$ | 含义 |
| --- | --- | --- | --- |
| 0.99 | 12 | $e^{12 \ln 0.99} = e^{-0.1206} = 0.886$ | 折扣权重的 88.6% 落在视野之外 |
| 0.997 | 15 | $e^{15 \ln 0.997} = e^{-0.0451} = 0.956$ | 折扣权重的 95.6% 落在视野之外 |

第二行是 Dreamer V3 的设置（$\gamma = 0.997$、想象视野 15，出自论文的超参数表）。
也就是说，按折扣权重算，想象的 15 步只覆盖了 4.4%，其余 95.6% 都压在 critic 身上。
**Dreamer 的想象视野之所以可以很短，全靠 critic；critic 学歪了，actor 一定跟着歪。**
这是第 7、9、10 节反复关心 critic 的原因。

### 1.5 推导：计算量的账

**决策时。** PlaNet 每个真实时间步的隐空间转移次数（沿用 Phase 1 第 6.5 节的账）：

$$
\text{PlaNet 每步的模型调用次数} = N \times H \times I
$$

其中 $N$ 是候选数，$H$ 是规划视野，$I$ 是 CEM 的迭代轮数。
PlaNet 论文的设置是 $N = 1000$、$H = 12$、$I = 10$（待核对），合计每步 12 万次隐空间转移。
Dreamer 每步只需要：一次编码器、一次 RSSM 的状态更新、一次 actor 的前向计算。

**训练时。** Dreamer 把计算挪到了训练阶段。每次梯度更新，想象产生的转移数是：

$$
\text{每次更新想象的转移数} = B \times T \times H
$$

以 $B = 16$、$T = 64$、$H = 15$ 为例（V3 论文超参数表里的批大小、序列长度和想象视野）：
起点有 $16 \times 64 = 1024$ 个，每个往前想 15 步，一共 $1024 \times 15 = 15360$ 个想象出的转移。
而这一批数据里真实的转移只有 1024 个。**每一个真实的转移，被放大成了 15 个想象的转移来训练行为。**
这是 Dreamer 样本效率的直接来源。

| | PlaNet | Dreamer |
| --- | --- | --- |
| 决策时的计算 | 重：每步一次完整的 CEM | 轻：一次前向计算 |
| 训练时的计算 | 只训练世界模型 | 训练世界模型，再加想象和 actor-critic 的更新 |
| 规划结果存在哪 | 不存 | 存在 actor 和 critic 的参数里 |

> 📌 待回填：PlaNet 的 CEM 规划与 actor 前向计算在本机上的单步决策耗时对比，来自 `01-from-planet-to-dreamer.py`。

### 1.6 算法

```
算法：Dreamer 的总循环（三代通用的骨架）

初始化：世界模型 θ，actor φ，critic ψ，经验回放 D
 1. 用随机动作采集若干回合，放进 D
 2. 重复：
      ── ① 学世界模型 ──
        从 D 采样 B 条长度为 T 的序列片段 (o, a, r, c)                   形状 [B, T, ...]
        用 RSSM 的 observe() 算出每一步的后验状态 m_t = (h_t, z_t)       形状 [B, T, D]
        计算世界模型损失，更新 θ
      ── ② 在想象中学行为 ──
        把 B×T 个后验状态当作起点（停止梯度）                             形状 [B·T, D]
        用 actor 选动作、用先验往前推 H 步，得到想象轨迹                    形状 [H+1, B·T, D]
        在想象轨迹上预测奖励、continue、价值，算出 λ-return
        用 λ-return 更新 actor φ 和 critic ψ
      ── ③ 与环境交互 ──
        用编码器和 RSSM 维护当前的模型状态 m_t
        从 actor 采样动作，在真实环境里执行，把 (o, a, r, c) 存进 D
```

第 ③ 步里智能体要在线维护模型状态：每来一个新观测，用 `observe()` 把 $h_t$、$z_t$ 往前推一步。
这和 PlaNet 在规划之前做的"状态滤波"是同一个操作，只是之后不再规划，直接把 $m_t$ 交给 actor。

### 1.7 代码对应

对应 `01-from-planet-to-dreamer.py`。这一节讲义不训练任何东西，做两件事：

1. `planning_cost_per_step()`：统计一次 CEM 规划和一次 actor 前向计算各需要多少次网络调用，并实测耗时。
2. `dreamer_training_loop()`：用占位的世界模型、actor、critic 把上面的总循环搭起来，确认数据流和张量形状。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| 回放采样的观测 | `[B, T, 3, 64, 64]` | $B$ 条序列，每条 $T$ 帧 |
| 后验状态 $m_t$ | `[B, T, D_h + N_z·K]` | $h_t$ 与展平的 $z_t$ 拼接 |
| 想象的起点 | `[B·T, D_h + N_z·K]` | 把前两维合并 |
| 想象轨迹 | `[H+1, B·T, D_h + N_z·K]` | 第 0 步是起点 |

### 1.8 三代 Dreamer 一览

后面各节按"部件"组织，不按"版本"组织。先给一张版本的地图，细节在第 11.1 节汇总。

| | Dreamer V1 | Dreamer V2 | Dreamer V3 |
| --- | --- | --- | --- |
| 隐变量 | 高斯 | 类别（$32 \times 32$） | 类别（$32 \times 32$） |
| KL 项 | 一项，带 free nats | 拆成两半（KL balancing） | 拆成两半 + free bits |
| actor 的梯度 | 沿模型反传（pathwise） | 离散动作用 REINFORCE，连续动作沿模型反传 | REINFORCE + 回报归一化（见第 9 节的版本说明） |
| critic | 回归 λ-return（平方误差） | 同左，目标用慢更新的目标网络 | 离散回归（two-hot）+ EMA 正则 |
| 主要测试场 | 连续控制（DMC 像素） | Atari | 一套超参数跨多个领域 |
| 本章哪一节 | 第 3 节 | 第 4、5 节 | 第 6–9 节 |

### 1.9 常见误区

- **误区：Dreamer 比 PlaNet 强，是因为世界模型更好。**
  实际情况：V1 的世界模型和 PlaNet 基本相同（同一个高斯 RSSM）。差别在于怎么用模型：
  PlaNet 在决策时规划，Dreamer 在训练时用模型的想象学出 actor 和 critic。
- **误区：有了 actor 就不需要世界模型了，Dreamer 只是多了个模型的无模型方法。**
  实际情况：actor 和 critic 的训练数据全部来自世界模型的想象，输入也是世界模型算出的状态 $m_t$。
  世界模型是整个系统的地基；真实数据只直接用于训练世界模型。
- **误区：想象视野 15 步，说明 Dreamer 只能解决 15 步之内的问题。**
  实际情况：视野之外的回报由 critic 概括（第 1.4 节）。critic 通过自举一步步把远处的奖励信息传回来，
  有效视野由折扣因子决定，大约是 $1 / (1 - \gamma)$ 步。
- **误区：学了策略，规划就没用了。**
  实际情况：两者可以结合。Phase 7 的 TD-MPC2 既学策略和价值，决策时又做规划，把策略当作规划器的先验。

---

## 2. 在想象中展开轨迹

### 2.1 直觉

做故障演练时，没有人会从一台空机器开始、把线上跑了半年的流量重放一遍才到达想测的场景。
通常的做法是：从生产环境的某个**快照**拉出一个沙箱，从那个时刻开始往后推演"如果我这样操作会怎样"。
快照取自真实发生过的状态，推演只往后做一小段。

Dreamer 的想象就是这么组织的：

- **快照**：回放数据里每一个时间步的后验状态 $m_t$。它是世界模型看了真实观测之后算出来的，贴着真实数据。
- **沙箱推演**：从每个快照出发，不再看任何观测，只用先验往前推 $H$ 步；每一步的动作由当前的 actor 给出。

一批数据有 $B \times T$ 个快照，就同时开 $B \times T$ 个沙箱。全部在隐空间里进行，不生成像素。

### 2.2 定义：想象轨迹

先把回放数据里各个量的对齐方式固定下来。这张图后面每一节都要用：

```
  回放里的时间      t-1                        t                         t+1

  观测             o_{t-1}                    o_t                       o_{t+1}
  动作、奖励              ╲  a_{t-1}, r_{t-1}        ╲  a_t, r_t
                          ╲                          ╲
                           ▼                          ▼
  模型状态         m_{t-1} ─────────────────▶ m_t ─────────────────▶ m_{t+1}
                                              │
                                              ├─▶ 解码器        预测 o_t
                                              ├─▶ 奖励头        预测 r_{t-1}（走到 m_t 这一步拿到的奖励）
                                              └─▶ continue 头   预测 c_t    （到了 m_t，回合还继续吗）
```

奖励头的输入是"到达之后"的状态，预测的是"走这一步拿到的奖励"。这和 Gymnasium 的接口一致：
`env.step(action)` 把下一个观测和这一步的奖励一起返回。

**想象轨迹的定义。** 从回放里第 $t$ 步的后验状态出发，想象步数记作 $k = 0, 1, \dots, H$：

$$
\begin{aligned}
\text{起点：} \quad & \hat m_0 = \mathrm{sg}(m_t), \qquad m_t = (h_t, z_t), \quad z_t \sim q_\theta(z_t \mid h_t, o_t) \\
\text{动作：} \quad & \hat a_k \sim \pi_\phi(\,\cdot \mid \hat m_k) \\
\text{序列模型：} \quad & \hat h_{k+1} = f_\theta(\hat h_k, \hat z_k, \hat a_k) \\
\text{先验：} \quad & \hat z_{k+1} \sim p_\theta(\,\cdot \mid \hat h_{k+1}), \qquad \hat m_{k+1} = (\hat h_{k+1}, \hat z_{k+1}) \\
\text{奖励：} \quad & \hat r_k = \mathbb{E}\big[\, p_\theta(r \mid \hat m_{k+1}) \,\big] \\
\text{continue：} \quad & \hat c_{k+1} = \mathbb{E}\big[\, p_\theta(c \mid \hat m_{k+1}) \,\big] \\
\text{价值：} \quad & v_k = V_\psi(\hat m_k)
\end{aligned}
$$

对照 Phase 3 的 RSSM：起点那一行用的是 `observe()`（有观测，用后验）；之后每一行用的都是 `imagine()`（没有观测，用先验）。

想象时各个部件的出勤情况：

| 部件 | 学世界模型时 | 想象时 |
| --- | --- | --- |
| 编码器（后验） | 用 | 不用。想象里没有观测可以编码 |
| 序列模型 $f_\theta$ | 用 | 用 |
| 先验 $p_\theta(z \mid h)$ | 用（作为 KL 的另一方） | 用。由它采样下一个 $z$ |
| 解码器 | 用 | 不用。想象不需要像素，只在可视化时解码出来看 |
| 奖励头、continue 头 | 用 | 用 |
| actor、critic | 不用 | 用 |

站在 actor 的角度，世界模型就是一个环境：状态是 $\hat m_k$，转移规律是"序列模型 + 先验"，奖励由奖励头给出，
回合是否结束由 continue 头给出。这是一个完全在隐空间里的 MDP。Phase 4、5 的算法可以原样搬进来用。

### 2.3 为什么从回放的后验状态出发

另一种看上去更自然的做法是：每次都从回合的初始状态开始想象。它有三个问题，正好对应"从后验状态出发"的三个好处。

**覆盖。** 想象只有 $H = 15$ 步。从初始状态出发，永远想不到回合第 200 步时的局面。
从回放里的每个状态出发，起点的分布就是智能体真实到过的状态的分布，回合的前、中、后段都有。

**误差。** Phase 1 第 5 节的结论是，开环展开的误差随步数增长，在会放大扰动的系统里是指数增长。
要从初始状态想到第 200 步，误差累积 200 步；从第 200 步的后验状态出发，误差累积从 0 重新算起，最多累积 $H$ 步。
这和 MPC "每一步用真实状态把误差清零"是同一个道理，只是清零发生在训练时。

**接地。** 后验状态是看了真实观测 $o_t$ 之后算出的，和真实数据对得上。
从它出发，想象的前几步离真实数据的分布很近，模型在这里最可信。

Phase 5 的 MBPO 用的是同一个想法：从回放里的真实状态出发，只做很短的模型展开。
区别在于 MBPO 在原始状态空间里展开，Dreamer 在隐空间里展开。

这种组织方式还带来一个不那么显眼的性质：

> 起点来自旧策略采的数据（离策略），但从起点往后的每一步都是**当前的 actor** 在**当前的模型**里走出来的（在策略）。

所以在想象轨迹上可以直接用在策略的算法（λ-return、REINFORCE），不需要重要性采样之类的离策略修正。
"数据不新鲜"的问题被世界模型吸收了：旧数据只用来训练模型和提供起点，行为学习用的永远是新想象出来的数据。

### 2.4 推导：回合会结束时，想象轨迹的权重

有些任务会提前结束（摔倒、通关）。世界模型不知道"结束之后没有东西"，你让它往下推，它照样推。
终止之后的想象状态没有意义，不该拿来训练。

continue 头给出的 $\hat c_j \in [0, 1]$ 是"到了第 $j$ 步，回合还在继续"的概率。
想象轨迹走到第 $k$ 步还没结束，要求第 $1, 2, \dots, k$ 步每一步都没结束。

**第一步：写成联合概率。** 记事件 $E_j$ 为"第 $j$ 步的状态不是终止状态"。

$$
P(\text{第 } k \text{ 步仍然有效}) = P(E_1, E_2, \dots, E_k)
$$

**第二步：用链式法则分解，每一项是 continue 头在该步的输出。**
continue 头预测的是"在之前都没结束的条件下，这一步没结束"的概率，所以：

$$
P(E_1, \dots, E_k) = \prod_{j=1}^{k} P(E_j \mid E_1, \dots, E_{j-1}) = \prod_{j=1}^{k} \hat c_j
$$

**第三步：把起点本身也算进去。** 起点 $m_t$ 来自回放，它是不是终止状态是已知的（真实的 $c_t$）。
于是第 $k$ 步的权重是：

$$
w_k = c_t \prod_{j=1}^{k} \hat c_j, \qquad w_0 = c_t
$$

actor 和 critic 在第 $k$ 步的损失乘上 $\mathrm{sg}(w_k)$。Dreamer V2 的论文明确写了这个加权
（用预测的折扣因子的累积乘积）；V3 的官方实现还会把折扣 $\gamma$ 一并乘进权重（待核对）。

**手算例子。** 起点不是终止状态（$c_t = 1$），想象 3 步，continue 头的输出依次是 $\hat c_1 = 1.0$、$\hat c_2 = 0.9$、$\hat c_3 = 0.5$：

$$
w_0 = 1, \qquad w_1 = 1 \times 1.0 = 1, \qquad w_2 = 1 \times 0.9 = 0.9, \qquad w_3 = 0.9 \times 0.5 = 0.45
$$

第 3 步的状态只有 45% 的概率是一个真实存在的局面，它上面的损失也只算 45%。
如果起点本身就是终止状态（$c_t = 0$），整条想象轨迹的权重全是 0，等于丢弃。

continue 概率在本章有两个用处，不要混：

| 用处 | 在哪 | 作用 |
| --- | --- | --- |
| 损失的权重 $w_k$ | 本节 | 终止之后的状态不参与训练 |
| 回报递推里的系数 | 第 3、9 节 | 终止之后的奖励和价值不计入回报 |

### 2.5 想象环节的 stop-gradient

想象把三个网络串进了同一张计算图：世界模型算出起点、推演轨迹、预测奖励；actor 选动作；critic 估价值。
不加限制的话，任何一个损失的梯度都能流到任何一个网络。必须划清边界：

```
   世界模型的损失 ──▶ 只更新 θ
   actor 的损失   ──▶ 只更新 φ
   critic 的损失  ──▶ 只更新 ψ

                 sg ①                                         sg ③
   回放 ─▶ 编码器 ─╫─▶ m̂_0 ─▶ [actor] ─▶ â_0 ─▶ [模型] ─▶ m̂_1 ─╫─▶ [critic] ─▶ critic 损失
           (θ)                  (φ)             (θ)                  (ψ)
                                                 ②
                                        梯度可以穿过，但 θ 不更新
```

| 编号 | 位置 | 做法 | 不这样做会怎样 |
| --- | --- | --- | --- |
| ① | 想象的起点 | $\hat m_0 = \mathrm{sg}(m_t)$ | actor 的损失会经过起点流进编码器和 RSSM。世界模型会被训练成"让想象的回报变高"，也就是学会自欺：表征不再忠实于观测 |
| ② | 想象过程中的世界模型参数 | 梯度可以**穿过**模型（V1 需要它），但 actor 的优化器里只有 $\phi$，$\theta$ 不更新 | 同上：模型会为了讨好 actor 而改动自己的转移规律和奖励预测 |
| ③ | critic 的输入 | 算 critic 损失时，输入的想象状态停止梯度 | critic 的损失会顺着状态流回 actor（经过可微的动作采样）。actor 会被推向"让 critic 好拟合"的状态，而不是回报高的状态 |

第 ② 条里"穿过"和"更新"的区别是本章的一个关键点，第 3 节会再讲：
**参数不更新，不等于梯度不经过。** V1 的 actor 梯度必须一路穿过世界模型的转移函数和奖励头才能回到 actor，
这时世界模型是计算图里一段可微的函数，只是它的参数不在 actor 的优化器里。

第 ① 条背后是一个设计立场：**世界模型只对真实数据负责。** 它的表征由重建、奖励预测、KL 这些基于真实数据的损失塑造，
actor 和 critic 是它的下游用户，不能反过来改它。Phase 7 的 TD-MPC2 立场相反：
那里价值的梯度会流进表征，表征是为"算准价值"服务的。

### 2.6 算法

```
算法：批量想象 imagine_rollout

输入：后验状态 (h, z)，形状 [B, T, ...]；真实的 continue 标志 c，形状 [B, T]；actor π_φ；视野 H
 1. 展平并停止梯度：ĥ_0, ẑ_0 ← sg(h), sg(z)                         形状 [B·T, ...]
 2. 对 k = 0 … H−1：
        m̂_k ← concat(ĥ_k, ẑ_k)                                     形状 [B·T, D]
        â_k ~ π_φ(· | m̂_k)                                          形状 [B·T, A]
        ĥ_{k+1} ← f_θ(ĥ_k, ẑ_k, â_k)                                序列模型（GRU）走一步
        ẑ_{k+1} ~ p_θ(· | ĥ_{k+1})                                  从先验采样，不看观测
 3. 对想象出的状态 m̂_1 … m̂_H：
        r̂_{k} ← 奖励头(m̂_{k+1}) 的期望，k = 0 … H−1                  形状 [H, B·T]
        ĉ_{k} ← continue 头(m̂_k) 的概率，k = 1 … H                   形状 [H, B·T]
 4. 权重：w_0 ← c（真实标志），w_k ← w_{k−1} · ĉ_k；整体停止梯度       形状 [H+1, B·T]
输出：状态 m̂_{0:H}，动作 â_{0:H−1}，奖励 r̂_{0:H−1}，continue ĉ_{1:H}，权重 w_{0:H}
```

### 2.7 代码对应

对应 `02-imagination-rollouts.py` 的 `imagine_rollout()`。为了把注意力放在想象本身，
这一节讲义直接复用 Phase 3 训练好的高斯 RSSM，actor 先用随机初始化的网络。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `start_h` | `[B·T, D_h]` | 展平后的确定性状态，已停止梯度 |
| `start_z` | `[B·T, D_z]` | 展平后的随机状态。高斯时 $D_z$ 是隐变量维度；类别时 $D_z = N_z \cdot K = 1024$ |
| `imag_states` | `[H+1, B·T, D_h + D_z]` | 第 0 维是想象的步数，第 0 步是起点 |
| `imag_actions` | `[H, B·T, A]` | 每一步 actor 采样的动作 |
| `imag_rewards` | `[H, B·T]` | $\hat r_0, \dots, \hat r_{H-1}$ |
| `imag_conts` | `[H, B·T]` | $\hat c_1, \dots, \hat c_H$ |
| `weights` | `[H+1, B·T]` | $w_0, \dots, w_H$ |

时间维放在第 0 维是为了让后面的 λ-return 递推写起来方便（从后往前对第 0 维做循环）。

讲义里用一个小测试验证 stop-gradient 的位置：对 actor 损失调用 `backward()` 之后，
检查世界模型和 critic 的每个参数的 `.grad` 是不是 `None` 或全零。这种测试几行代码，能挡住一大类 bug。

> 📌 待回填：把想象轨迹解码成视频，与同一起点的真实后续画面对比（预期：前几步接近，之后逐渐偏离）；想象状态的 continue 权重随步数的变化，来自 `02-imagination-rollouts.py`。

### 2.8 常见误区

- **误区：想象就是把世界模型当视频生成器，生成一段画面再喂给策略。**
  实际情况：想象全程在隐空间里进行，不解码。actor 和 critic 吃的是 $\hat m_k$。
  解码成画面只是给人看的调试手段。
- **误区：想象从回合的初始状态开始。**
  实际情况：从回放里每一个时间步的后验状态开始（第 2.3 节）。这是 Dreamer 能用很短的视野覆盖整个回合的原因。
- **误区：起点是旧策略采的，所以想象出的数据是离策略数据，要做离策略修正。**
  实际情况：只有起点的分布来自旧策略。起点之后的动作全部由当前 actor 选出，是在策略数据。
- **误区：给 actor 的损失加上 stop-gradient，是因为反正梯度传不过去。**
  实际情况：恰恰因为梯度传得过去才要拦。隐空间里的一切都是可微的，不拦的话 actor 的损失会改写世界模型。

---

## 3. Dreamer V1：沿可微的世界模型反向传播

### 3.1 直觉

想提高一个线上指标，有两种工作方式。

- **黑盒方式**：系统是别人的，看不到内部。只能做 A/B 测试：改一下配置，看指标涨了还是跌了，再决定往哪边调。
  每次实验只给你一个数字，要做很多次实验才能看出方向。
- **白盒方式**：系统的源码在手上，而且每一步都可导。直接对"指标关于配置"求导，一次就知道该往哪调、调多少。

Phase 5 的 REINFORCE（score-function 估计器）是黑盒方式：环境不可导，只能靠"试了这个动作，回报是多少"来估计梯度，方差大。
而 Dreamer 的"环境"是自己学出来的世界模型，它是神经网络，每一步都可导。于是可以用白盒方式：

> 把想象轨迹的回报写成 actor 参数的可微函数，让自动微分沿着"回报 ← 奖励头 ← 状态 ← 转移 ← 动作 ← actor"这条链一路反传。

这就是 Phase 5 讲的 pathwise 估计器（重参数化）加上"沿模型反向传播"。Dreamer V1 是把它用在从像素学出的隐空间模型上的代表作。

### 3.2 定义：V1 的两个目标

Dreamer V1（Hafner et al. 2020，arXiv:1912.01603）在想象轨迹上定义了两个目标。

**actor**：最大化想象轨迹上各步 λ-return 之和。

$$
\max_\phi \; \mathbb{E}\Big[ \sum_{k=0}^{H} R^\lambda_k \Big]
$$

**critic**：回归 λ-return，目标停止梯度。

$$
\min_\psi \; \mathbb{E}\Big[ \sum_{k=0}^{H} \frac{1}{2} \big( V_\psi(\hat m_k) - \mathrm{sg}(R^\lambda_k) \big)^2 \Big]
$$

期望是对回放里的起点、想象中的动作采样和隐变量采样取的。
两个目标共用同一批想象轨迹和同一组 $R^\lambda_k$。区别只在梯度怎么走：actor 的目标里 $R^\lambda_k$ **不**停止梯度，critic 的目标里停止。

V1 的世界模型是 Phase 3 的高斯 RSSM，损失是"重建 + 奖励 + KL"三项，这里不重复。

### 3.3 推导（一）：想象轨迹上的 λ-return

λ-return 在 Phase 4 推导过。这里把它搬到想象轨迹上，补上两处变化：视野是有限的；回合可能终止。

**第一步：单步目标。** 在想象的第 $k$ 步，走一步拿到 $\hat r_k$，到达 $\hat m_{k+1}$。
如果 $\hat m_{k+1}$ 不是终止状态，后面的回报用价值估计 $v_{k+1}$ 代替；如果是终止状态，后面没有回报。
用 continue 概率把两种情况合起来：

$$
R^{(1)}_k = \hat r_k + \gamma \, \hat c_{k+1} \, v_{k+1}
$$

这是 TD 目标（Phase 4）乘上了一个"还在继续"的系数。

**第二步：往后多走几步再自举，得到 $n$ 步目标。** 例如两步：

$$
R^{(2)}_k = \hat r_k + \gamma \, \hat c_{k+1} \big( \hat r_{k+1} + \gamma \, \hat c_{k+2} \, v_{k+2} \big)
$$

走得越多，用到的想象奖励越多、对 critic 的依赖越少（偏差小），但累积的模型误差和采样噪声越多（方差大）。

**第三步：λ-return 是各个 $n$ 步目标的加权平均，写成递推式。**
Phase 4 的结论是，按几何权重 $(1-\lambda)\lambda^{n-1}$ 混合所有 $n$ 步目标，等价于下面的递推：

$$
R^\lambda_k = \hat r_k + \gamma \, \hat c_{k+1} \Big[ (1 - \lambda) \, v_{k+1} + \lambda \, R^\lambda_{k+1} \Big], \qquad R^\lambda_H = v_H
$$

读法：走一步拿到奖励之后，剩下的回报有两个估计。一个是 critic 现在就给的 $v_{k+1}$（保守，偏差大方差小），
一个是继续往后展开得到的 $R^\lambda_{k+1}$（激进，偏差小方差大）。两者按 $(1-\lambda) : \lambda$ 混合。

末端条件 $R^\lambda_H = v_H$ 是有限视野带来的：想象到第 $H$ 步就停了，后面只能全部交给 critic。
这正是第 1.4 节的"视野之外靠价值函数"。

**第四步：验证递推和"加权平均"是一回事。** 把递推式的右边重新分组：

$$
\begin{aligned}
R^\lambda_k &= \hat r_k + \gamma \hat c_{k+1} (1 - \lambda) v_{k+1} + \gamma \hat c_{k+1} \lambda R^\lambda_{k+1} \\
&= (1 - \lambda) \underbrace{\big( \hat r_k + \gamma \hat c_{k+1} v_{k+1} \big)}_{R^{(1)}_k} + \lambda \underbrace{\big( \hat r_k + \gamma \hat c_{k+1} R^\lambda_{k+1} \big)}_{\text{先走一步，再接下一步的 λ-return}}
\end{aligned}
$$

第二行用的是 $\hat r_k = (1-\lambda)\hat r_k + \lambda \hat r_k$。
所以 λ-return 给单步目标 $(1-\lambda)$ 的权重，剩下的 $\lambda$ 交给"走一步再说"。
对 $R^\lambda_{k+1}$ 重复这个分解，两步目标拿到 $\lambda(1-\lambda)$，三步目标拿到 $\lambda^2(1-\lambda)$，以此类推；
到了末端，剩下的全部权重 $\lambda^{H-k-1}$ 落在最长的那个目标上。这就是 V1 论文里写成求和形式的 $\mathrm{V}_\lambda$。

两个极端：

| $\lambda$ | 递推式变成 | 含义 |
| --- | --- | --- |
| 0 | $R^\lambda_k = \hat r_k + \gamma \hat c_{k+1} v_{k+1}$ | 单步 TD 目标。只信 critic，想象的后几步白想了 |
| 1 | $R^\lambda_k = \hat r_k + \gamma \hat c_{k+1} R^\lambda_{k+1}$ | 把想象的奖励一路加到末端，只在第 $H$ 步用一次 critic |

三代 Dreamer 都用 $\lambda = 0.95$、$H = 15$（各篇论文的超参数表）。

**手算例子。** 取 $H = 3$，$\gamma = 0.9$，$\lambda = 0.5$（数字取得好算，不是论文的设置）。想象轨迹上：

| $k$ | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| 预测奖励 $\hat r_k$ | 1 | 0 | 2 | |
| 价值 $v_k$ | | 5 | 4 | 10 |
| continue $\hat c_k$ | | 1 | 1 | 1 |

从后往前算：

$$
\begin{aligned}
R^\lambda_3 &= v_3 = 10 \\
R^\lambda_2 &= 2 + 0.9 \times 1 \times \big[ 0.5 \times 10 + 0.5 \times 10 \big] = 2 + 9 = 11 \\
R^\lambda_1 &= 0 + 0.9 \times 1 \times \big[ 0.5 \times 4 + 0.5 \times 11 \big] = 0.9 \times 7.5 = 6.75 \\
R^\lambda_0 &= 1 + 0.9 \times 1 \times \big[ 0.5 \times 5 + 0.5 \times 6.75 \big] = 1 + 0.9 \times 5.875 = 6.2875
\end{aligned}
$$

用"加权平均"的定义核对 $R^\lambda_1$。从第 1 步出发只有两个目标可选：

$$
R^{(1)}_1 = 0 + 0.9 \times 4 = 3.6, \qquad R^{(2)}_1 = 0 + 0.9 \times 2 + 0.81 \times 10 = 9.9
$$

单步目标的权重是 $1 - \lambda = 0.5$，最长的目标拿剩下的 $\lambda = 0.5$：$0.5 \times 3.6 + 0.5 \times 9.9 = 6.75$。和递推的结果一致。

**再算一遍有终止的情况。** 其余不变，把 $\hat c_2$ 改成 0（模型认为第 2 步是终止状态）：

$$
\begin{aligned}
R^\lambda_1 &= 0 + 0.9 \times 0 \times [\dots] = 0 \\
R^\lambda_0 &= 1 + 0.9 \times 1 \times \big[ 0.5 \times 5 + 0.5 \times 0 \big] = 1 + 2.25 = 3.25
\end{aligned}
$$

第 2 步之后的奖励和价值全部被挡住，没有漏进前面的回报里。
注意 $R^\lambda_0$ 里还有一项 $0.5 \times 5$：那是 critic 对第 1 步的估计 $v_1 = 5$。
critic 如果还没学会"第 1 步之后马上要终止"，这个 5 就是错的。λ-return 不会替 critic 纠正它自己的错误，只会把它和展开的结果混在一起。

### 3.4 推导（二）：沿模型反传的 actor 梯度

目标是求 $\nabla_\phi \, \mathbb{E}\big[\sum_k R^\lambda_k\big]$。困难在于期望里有两处采样：动作从 actor 采，隐变量从先验采。
采样这个操作本身不可导。

**第一步：重参数化，把随机性挪到参数之外。** 这是 Phase 5 的 pathwise 估计器，也是旧课 VAE 里的重参数化技巧。
V1 的 actor 输出一个高斯分布再过 tanh（和 SAC 的策略一样），先验是高斯：

$$
\hat a_k = \tanh\big( \mu_\phi(\hat m_k) + \sigma_\phi(\hat m_k) \, \epsilon_k \big), \qquad
\hat z_{k+1} = \mu_\theta(\hat h_{k+1}) + \sigma_\theta(\hat h_{k+1}) \, \epsilon'_{k+1}
$$

其中 $\epsilon_k$、$\epsilon'_{k+1}$ 是标准高斯噪声。噪声固定之后，整条想象轨迹是 $\phi$ 的确定性可微函数。

**第二步：交换求导和期望。** 期望现在是对噪声取的，噪声的分布不依赖 $\phi$：

$$
\nabla_\phi \, \mathbb{E}_{\epsilon, \epsilon'}\Big[ \sum_k R^\lambda_k \Big] = \mathbb{E}_{\epsilon, \epsilon'}\Big[ \sum_k \nabla_\phi R^\lambda_k \Big]
$$

右边可以用采样估计：采一组噪声，展开轨迹，对 $\sum_k R^\lambda_k$ 调一次 `backward()`。

**第三步：看清 $\nabla_\phi R^\lambda_k$ 里有什么。** $R^\lambda_k$ 由奖励 $\hat r_j$ 和价值 $v_j$ 组成，它们都是想象状态的函数。
状态又依赖前面所有的动作。记 $\frac{d}{d\phi}$ 为全导数（把所有间接依赖都算上），$\frac{\partial}{\partial}$ 为偏导数（只算直接依赖）。
对状态用链式法则：

$$
\frac{d \hat m_{k+1}}{d \phi} = \underbrace{\frac{\partial \hat m_{k+1}}{\partial \hat m_k}}_{\text{模型对状态的雅可比}} \frac{d \hat m_k}{d \phi} + \underbrace{\frac{\partial \hat m_{k+1}}{\partial \hat a_k}}_{\text{模型对动作的雅可比}} \frac{d \hat a_k}{d \phi}
$$

对动作用链式法则。动作既直接依赖 $\phi$（actor 的权重），又通过输入的状态间接依赖 $\phi$：

$$
\frac{d \hat a_k}{d \phi} = \underbrace{\frac{\partial \hat a_k}{\partial \phi}}_{\text{直接依赖}} + \underbrace{\frac{\partial \hat a_k}{\partial \hat m_k}}_{\text{actor 对状态的雅可比}} \frac{d \hat m_k}{d \phi}
$$

起点停止了梯度，所以 $\frac{d \hat m_0}{d\phi} = 0$。两式交替使用，就能一步步算出每个状态对 $\phi$ 的全导数。
最后：

$$
\frac{d R^\lambda_k}{d \phi} = \sum_{j} \frac{\partial R^\lambda_k}{\partial \hat m_j} \, \frac{d \hat m_j}{d \phi}
$$

其中 $\frac{\partial R^\lambda_k}{\partial \hat m_j}$ 来自奖励头和 critic 对输入状态的梯度。

这三个式子不需要手写，自动微分会做。写出来是为了看清两件事：

- **梯度穿过了世界模型和 critic。** 式子里出现了模型的雅可比、奖励头和 critic 对输入的梯度。
  它们的**参数**不更新，但它们作为可微函数必须在计算图里。
- **雅可比会连乘。** $\frac{d\hat m_{k}}{d\phi}$ 里含有 $k$ 个"模型对状态的雅可比"的乘积。
  这和 BPTT、和 Phase 1 第 5.3 节的误差递推是同一个结构：雅可比的模大于 1 时梯度随步数指数增长。
  这是想象视野不能太长的第二个理由（第一个是模型误差累积）。critic 在第 $H$ 步把链条截断，梯度最多连乘 $H$ 次。

**手算例子。** 用一个一维的玩具系统，把上面的链式法则走一遍。

- 模型：$s' = s + a$
- 奖励：到达 $s'$ 得到 $-s'^2$（离原点越近越好）
- 策略：$a = -\kappa s$，只有一个参数 $\kappa$，没有噪声
- 起点 $s_0 = 2$，当前 $\kappa = 0.5$，想象两步，不折扣，不用 critic

前向：

$$
s_1 = s_0 + a_0 = 2 - 0.5 \times 2 = 1, \qquad s_2 = s_1 + a_1 = 1 - 0.5 \times 1 = 0.5, \qquad J = -s_1^2 - s_2^2 = -1.25
$$

反向，按上面两个链式法则交替计算：

$$
\begin{aligned}
\frac{d a_0}{d\kappa} &= -s_0 = -2 & &\text{（起点不依赖 } \kappa \text{，只有直接依赖）} \\
\frac{d s_1}{d\kappa} &= \frac{d s_0}{d\kappa} + \frac{d a_0}{d\kappa} = 0 - 2 = -2 \\
\frac{d a_1}{d\kappa} &= -s_1 - \kappa \frac{d s_1}{d\kappa} = -1 - 0.5 \times (-2) = 0 & &\text{（直接依赖 + 经过状态的间接依赖）} \\
\frac{d s_2}{d\kappa} &= \frac{d s_1}{d\kappa} + \frac{d a_1}{d\kappa} = -2 + 0 = -2 \\
\frac{d J}{d\kappa} &= -2 s_1 \frac{d s_1}{d\kappa} - 2 s_2 \frac{d s_2}{d\kappa} = -2 \times 1 \times (-2) - 2 \times 0.5 \times (-2) = 4 + 2 = 6
\end{aligned}
$$

用闭式解核对：$s_1 = s_0(1-\kappa)$，$s_2 = s_0(1-\kappa)^2$，所以 $J = -s_0^2\big[(1-\kappa)^2 + (1-\kappa)^4\big]$，

$$
\frac{dJ}{d\kappa} = s_0^2 \big[ 2(1-\kappa) + 4(1-\kappa)^3 \big] = 4 \times \big[ 1 + 0.5 \big] = 6
$$

两种算法一致。梯度为正，说明应该增大 $\kappa$（回原点回得更快），这符合直觉。

注意第三行：$\frac{da_1}{d\kappa}$ 的两项正好抵消。只看直接依赖会得到 $-1$，把经过状态的间接依赖算上才是 0。
**pathwise 梯度的信息量就在这些"经过状态"的项里**：它知道改了 $\kappa$ 之后 $s_1$ 会变，$s_1$ 变了 $a_1$ 也跟着变。
REINFORCE 不利用这层结构，它只知道"这次试出来回报是多少"。

### 3.5 推导（三）：critic 的损失与它的 stop-gradient

critic 的损失是平方误差，目标停止梯度：

$$
\mathcal{L}_{\text{critic}}(\psi) = \mathbb{E}\Big[ \sum_k \frac{1}{2} \big( V_\psi(\mathrm{sg}(\hat m_k)) - \mathrm{sg}(R^\lambda_k) \big)^2 \Big]
$$

这里有两个 sg，各拦一条路。

**目标上的 sg。** $R^\lambda_k$ 里含有 $v_{k+1} = V_\psi(\hat m_{k+1})$，也就是 critic 自己的输出。
不停止梯度的话，降低损失有两个办法：让预测 $V_\psi(\hat m_k)$ 靠近目标，或者让目标靠近预测。
后者是作弊：critic 可以把下一步的价值往回拉来迁就这一步，而不是把这一步的价值修正到位。
把目标当常数，就是 Phase 4 讲的半梯度（semi-gradient）TD 学习。
另外，$R^\lambda_k$ 还依赖 actor 的动作和世界模型，停止梯度也保证 critic 的损失不会去改它们。

**输入上的 sg。** 这就是第 2.5 节的 ③。在 V1 里它尤其要紧：动作是重参数化采样的，状态对 $\phi$ 可导，
不拦的话 critic 的回归误差会顺着状态一路传回 actor。

对一步求梯度看看它在做什么。记 $\delta_k = \mathrm{sg}(R^\lambda_k) - V_\psi(\hat m_k)$：

$$
\nabla_\psi \, \frac{1}{2} \big( V_\psi(\hat m_k) - \mathrm{sg}(R^\lambda_k) \big)^2 = -\delta_k \, \nabla_\psi V_\psi(\hat m_k)
$$

梯度下降把 $V_\psi(\hat m_k)$ 往目标的方向挪，步子正比于误差 $\delta_k$。
**步子正比于误差**这一点记住：回报的尺度大 100 倍，梯度就大 100 倍。这是 V3 要换掉平方误差的原因（第 6 节）。

### 3.6 V1 的梯度流向总表

| 损失 | 更新谁 | 梯度穿过但不更新 | 被 sg 挡住的地方 |
| --- | --- | --- | --- |
| 世界模型损失（重建 + 奖励 + KL） | $\theta$ | 无 | 无（V1 的 KL 是完整的一项） |
| actor 损失 $-\sum_k R^\lambda_k$ | $\phi$ | 序列模型、先验、奖励头（$\theta$）；critic（$\psi$） | 想象的起点 $\hat m_0$ |
| critic 损失 | $\psi$ | 无 | 目标 $R^\lambda_k$；输入状态 $\hat m_k$ |

三个损失各用一个优化器，各管一组参数。V1 论文的超参数表里，世界模型的学习率是 $6 \times 10^{-4}$，
actor 和 critic 是 $8 \times 10^{-5}$；隐变量是 30 维高斯加 200 维确定性状态；KL 的 free nats 是 3。

### 3.7 V1 留下的问题

V1 在 DeepMind Control Suite 的像素任务上效果很好（见第 12.1 节），但它的做法有几处绑死在"连续控制"上
（论文末尾有离散动作和提前终止的初步扩展实验，主体方法和主实验都是连续控制）：

| 问题 | 具体表现 | 后来怎么解决 |
| --- | --- | --- |
| 动作必须可重参数化 | 离散动作（Atari 的按键）采样不可导，pathwise 梯度走不通 | V2：离散动作改用 REINFORCE（第 4 节） |
| 高斯隐变量的表达力 | 画面里有离散的突变（敌人出现、切换房间）时，单峰高斯的先验很难描述"要么这样，要么那样" | V2：类别隐变量（第 4 节） |
| KL 只有一项 | 先验还没学好时，KL 会把表征往一个不靠谱的先验上拉 | V2：KL balancing（第 5 节） |
| critic 的目标追着自己跑 | 目标由 critic 自己算出，critic 一更新目标就变 | V2：目标网络；V3：EMA 正则（第 7 节） |
| 损失的尺度随任务变 | 平方误差的梯度正比于回报的尺度；熵正则和回报的相对大小随任务变 | V3：symlog、two-hot、回报归一化（第 6、7 节） |

### 3.8 算法

```
算法：Dreamer V1 的行为学习（在一批想象轨迹上）

输入：想象轨迹 m̂_{0:H}（动作和隐变量都用重参数化采样，保留计算图）
 1. 预测奖励 r̂_{0:H−1}，价值 v_{0:H}                                  形状 [H, N]、[H+1, N]，N = B·T
 2. 从后往前算 λ-return：
        R_H ← v_H
        对 k = H−1 … 0：R_k ← r̂_k + γ · ĉ_{k+1} · [(1−λ) · v_{k+1} + λ · R_{k+1}]
 3. actor 损失 ← −mean(w · R)            R 不停止梯度；只把 φ 交给优化器
 4. critic 损失 ← mean(w · ½ (V_ψ(sg(m̂)) − sg(R))²)
 5. 分别反向传播，分别更新 φ 和 ψ
```

第 3 步里 `R` 的计算图包含世界模型和 critic。实现时最简单的做法是：三个网络各建一个优化器，
actor 的损失只对 actor 的优化器调 `step()`；并在下一次更新前把其他网络上积累的梯度清零。

### 3.9 代码对应

对应 `03-dreamer-v1-actor-critic.py`。

| 函数 | 做什么 | 关键张量 |
| --- | --- | --- |
| `lambda_return(rewards, values, conts, gamma, lam)` | 第 3.3 节的递推 | 输入 `[H, N]`、`[H+1, N]`、`[H, N]`；输出 `[H+1, N]`（含末端的 $R^\lambda_H = v_H$） |
| `actor_loss_pathwise(returns, weights)` | $-\sum_k w_k R^\lambda_k$ 的均值 | `returns` 带计算图 |
| `critic_loss_mse(critic, states, returns, weights)` | 平方误差，目标和输入都 detach | `states` 为 `[H+1, N, D]` |

讲义先用第 3.4 节的一维玩具系统验证 `actor_loss_pathwise` 的梯度（应当等于手算的 6），
再接到 Phase 3 的高斯 RSSM 上训练像素版 `Pendulum-v1`。

> 📌 待回填：`lambda_return()` 在第 3.3 节手算例子上的输出（应为 6.2875、6.75、11、10）；Dreamer V1 在像素版 `Pendulum-v1` 上的学习曲线，以及与 Phase 3 的 PlaNet 在相同环境步数下的回报和单步决策耗时对比，来自 `03-dreamer-v1-actor-critic.py`。

### 3.10 常见误区

- **误区：actor 的损失里用到了世界模型和 critic，所以这一步也在训练它们。**
  实际情况：梯度穿过它们，但它们的参数不在 actor 的优化器里，不更新。穿过和更新是两回事。
- **误区：critic 的目标停止梯度是一个可有可无的实现细节。**
  实际情况：目标里含有 critic 自己的输出。不停止梯度，critic 会通过挪动目标来降低损失，学到的不再是价值。
- **误区：pathwise 梯度方差小，所以想象视野可以随便加长。**
  实际情况：梯度里有 $H$ 个雅可比的连乘，视野长了会爆炸或消失；模型误差也随步数累积。
  V1 论文的消融显示，有了 critic 之后短视野就够用，这才是正确的用法。
- **误区：λ-return 里的奖励是真实奖励。**
  实际情况：想象轨迹上的奖励全部是奖励头预测出来的。奖励头不准，λ-return 就不准，而且没有任何真实信号在这一步纠正它。
  真实奖励只在训练世界模型时出现。

---

## 4. Dreamer V2：类别隐变量与 straight-through

### 4.1 直觉

要用一条记录描述游戏的一帧画面，有两种表结构：

- **30 个浮点字段**。每个字段可以取任意实数。
- **32 个枚举字段，每个有 32 种取值**。

游戏画面里有大量非此即彼的事实：门是开的还是关的，敌人在不在，现在是哪个房间。
用枚举字段描述这些很自然；用浮点数描述，就得约定"大于 0 算开，小于 0 算关"，而且中间那些值没有意义。

预测未来时差别更明显。假设下一帧有两种可能：敌人从左边出现，或者从右边出现，各一半概率。

- 高斯先验只有一个峰。它要么把峰放在中间（预测了一个不会发生的"敌人在正中"），要么把方差拉得很大去盖住两边。
- 类别先验直接给"左"和"右"各 0.5 的概率。多峰对它来说不需要任何额外机制。

Dreamer V2（Hafner et al. 2021，arXiv:2010.02193）把 RSSM 的随机隐变量从高斯换成了类别变量。
代价是类别采样不能像高斯那样重参数化。V2 用 Phase 5 讲过的 straight-through 估计器绕过去：
**前向用硬的 one-hot，反向假装它是软的概率。**

### 4.2 定义：类别隐变量

$z_t$ 由 $N_z = 32$ 个相互独立的类别变量组成，每个有 $K = 32$ 个类别（V2 论文的设置，V3 沿用）。

```
   编码器 / 先验网络输出 logits               按行 softmax                按行采样，转成 one-hot
        形状 [32, 32]                        形状 [32, 32]                   形状 [32, 32]

   变量 1   [ 1.2  -0.3  ...  0.8 ]         [ 0.11 0.02 ... 0.07 ]         [ 0  0  ...  1  ...  0 ]
   变量 2   [ 0.1   2.5  ... -1.0 ]   ──▶   [ 0.01 0.62 ... 0.00 ]   ──▶   [ 0  1  ...  0  ...  0 ]
     ...                                                                        ...
   变量 32  [-0.7   0.4  ...  1.9 ]         [ 0.01 0.03 ... 0.31 ]         [ 0  0  ...  0  ...  1 ]

                                                                    展平成 1024 维、恰有 32 个 1 的向量
```

形式化地，后验和先验都是 $N_z$ 个类别分布的乘积。以后验为例，记编码器输出的 logits 为 $\ell^{(i)} \in \mathbb{R}^K$：

$$
q_\theta(z_t \mid h_t, o_t) = \prod_{i=1}^{N_z} \mathrm{Cat}\big( z_t^{(i)} ; \, \mathrm{softmax}(\ell^{(i)}) \big)
$$

几个数字，帮助建立感觉：

| 量 | 值 | 怎么来的 |
| --- | --- | --- |
| $z_t$ 展平后的维度 | 1024 | $32 \times 32$ |
| 其中为 1 的位数 | 32 | 每个变量恰好选中一个类别 |
| 可以表示的不同取值 | $32^{32} = 2^{160} \approx 1.46 \times 10^{48}$ | 每个变量 32 种，共 32 个变量 |
| 每一步最多携带的信息 | $32 \times \ln 32 = 110.9$ nat（160 bit） | 每个变量最多 $\ln 32 = 3.466$ nat |

离散不等于容量小。160 bit 足够区分的局面数远超任何游戏里会出现的画面数。

### 4.3 推导：straight-through 梯度

**问题。** 世界模型的预测损失（重建、奖励）是 $z_t$ 的函数，而 $z_t$ 是从 $\mathrm{softmax}(\ell)$ 里采样出的 one-hot 向量。
要训练编码器，需要损失对 logits $\ell$ 的梯度。但"采样并转成 one-hot"是一个阶梯函数：
logits 动一点点，采到的类别要么不变（导数为 0），要么跳变（导数不存在）。梯度传不回去。

**straight-through 的做法。** 记 $p = \mathrm{softmax}(\ell)$，采样得到的 one-hot 向量为 $e$。构造：

$$
z = e + p - \mathrm{sg}(p)
$$

- **前向**：$p - \mathrm{sg}(p)$ 在数值上是 0，所以 $z = e$。下游拿到的是货真价实的 one-hot。
- **反向**：$e$ 是常数，$\mathrm{sg}(p)$ 当常数，只剩 $p$ 有梯度。所以 $\frac{\partial z}{\partial \ell}$ 被替换成了 $\frac{\partial p}{\partial \ell}$。

**第一步：写出链式法则。** 设下游的损失是 $L(z)$，记 $g_k = \frac{\partial L}{\partial z_k}$（在采到的 one-hot 处取值）。

$$
\frac{\partial L}{\partial \ell_j} \;\approx\; \sum_{k=1}^{K} g_k \, \frac{\partial p_k}{\partial \ell_j}
$$

写约等号是因为这一步用 $\frac{\partial p}{\partial \ell}$ 冒充了 $\frac{\partial z}{\partial \ell}$。

**第二步：代入 softmax 的雅可比 $\frac{\partial p_k}{\partial \ell_j} = p_k (\delta_{kj} - p_j)$**（旧课推过，$\delta_{kj}$ 在 $k = j$ 时为 1，否则为 0）。

$$
\frac{\partial L}{\partial \ell_j} \;\approx\; \sum_{k} g_k \, p_k (\delta_{kj} - p_j) = g_j \, p_j - p_j \sum_{k} g_k \, p_k
$$

**第三步：整理。** 记 $\bar g = \sum_k p_k g_k$（下游梯度按概率的加权平均）：

$$
\frac{\partial L}{\partial \ell_j} \;\approx\; p_j \, (g_j - \bar g)
$$

读法：某个类别的下游梯度 $g_j$ 高于平均（选它会让损失变大），它的 logit 的梯度为正，梯度下降会把它压低；反之抬高。
幅度再乘上这个类别当前的概率 $p_j$。

**手算例子（下游是线性函数）。** $K = 3$，$p = (0.2, 0.5, 0.3)$，采到第 2 类，$e = (0, 1, 0)$。
下游损失 $L(z) = w^\top z$，$w = (1, 2, 4)$，所以 $L = 2$，$g = w = (1, 2, 4)$。

$$
\bar g = 0.2 \times 1 + 0.5 \times 2 + 0.3 \times 4 = 2.4
$$

$$
\frac{\partial L}{\partial \ell} \approx \big( 0.2 \times (1 - 2.4), \;\; 0.5 \times (2 - 2.4), \;\; 0.3 \times (4 - 2.4) \big) = (-0.28, \; -0.20, \; 0.48)
$$

第 3 类代价最高（$w_3 = 4$），它的 logit 梯度为正，会被压低。三个分量之和为 0，这是 softmax 的梯度总有的性质。

和真实梯度比较。期望损失是 $\mathbb{E}[L] = \sum_k p_k w_k$，它对 logits 的梯度同样用 softmax 的雅可比算，
结果是 $p_j (w_j - \bar w)$，和上面的式子一模一样。**下游是线性函数时，straight-through 给出的就是精确梯度**，而且和采到哪一类无关。

**手算例子（下游是非线性函数）。** 把损失换成 $L(z) = (w^\top z)^2$，其余不变。

真实梯度：$\mathbb{E}[L] = \sum_k p_k w_k^2 = 0.2 \times 1 + 0.5 \times 4 + 0.3 \times 16 = 7.0$，对 logits 的梯度是 $p_j (w_j^2 - 7.0)$：

$$
\nabla_\ell \, \mathbb{E}[L] = \big( 0.2 \times (1 - 7), \;\; 0.5 \times (4 - 7), \;\; 0.3 \times (16 - 7) \big) = (-1.2, \; -1.5, \; 2.7)
$$

straight-through 的梯度取决于采到哪一类，因为 $g = 2 (w^\top e) \, w$ 随样本变。三种情况分别算出来，再按概率取平均：

| 采到的类别 | 概率 | $g = 2(w^\top e)\,w$ | $\bar g$ | straight-through 梯度 $p_j(g_j - \bar g)$ |
| --- | --- | --- | --- | --- |
| 1 | 0.2 | $(2, 4, 8)$ | 4.8 | $(-0.56, \; -0.40, \; 0.96)$ |
| 2 | 0.5 | $(4, 8, 16)$ | 9.6 | $(-1.12, \; -0.80, \; 1.92)$ |
| 3 | 0.3 | $(8, 16, 32)$ | 19.2 | $(-2.24, \; -1.60, \; 3.84)$ |
| 按概率平均 | | | | $(-1.344, \; -0.96, \; 2.304)$ |

期望是 $(-1.344, -0.96, 2.304)$，真实梯度是 $(-1.2, -1.5, 2.7)$。符号都对，数值不同。
**straight-through 是有偏的估计器**：下游越非线性，偏差越大。它换来的是低方差，以及前向传播里真正的离散样本。

Phase 5 比较过让梯度通过离散采样的三种办法，这里回顾一下它们在 Dreamer 里的位置：

| 估计器 | 有偏吗 | 方差 | 前向是离散的吗 | Dreamer 里谁在用 |
| --- | --- | --- | --- | --- |
| score-function（REINFORCE） | 无偏 | 大 | 是 | V2、V3 的 actor |
| straight-through | 有偏 | 小 | 是 | V2、V3 的类别隐变量 |
| Gumbel-Softmax | 有偏 | 小 | 否（软的），或配合 straight-through 变成硬的 | 不用 |
| pathwise（重参数化） | 无偏 | 小 | 不适用，只能用于连续变量 | V1 的高斯隐变量和 actor |

### 4.4 类别隐变量为什么更好用

V2 的消融实验显示，把高斯换成类别在 Atari 的多数游戏上有明显提升。
至于原因，论文很坦率：没有定论，只给了几个假说（出自论文讨论类别隐变量的部分）。

1. 类别先验能精确拟合"多个后验的混合"，高斯先验不能。
2. 展平后的 $z$ 是 1024 维里只有 32 位为 1 的稀疏向量，这种稀疏性可能有利于泛化。
3. straight-through 忽略了一个会缩放梯度的项，优化上可能反而更稳。
4. 类别变量对"不平滑的变化"（进入新房间、物体消失）是更合适的归纳偏置。

第 1 条可以算出来。先验 $p_\theta(z_t \mid h_t)$ 不看当前观测，它要预测的是"各种可能的下一帧对应的后验，混在一起是什么样"。

**类别的情况。** 两种可能的未来各占一半。未来 A 的后验是 $(1, 0)$，未来 B 的后验是 $(0, 1)$。
两个类别分布的混合还是类别分布：$0.5 \times (1, 0) + 0.5 \times (0, 1) = (0.5, 0.5)$。
先验取 $(0.5, 0.5)$ 就精确等于这个混合。每个后验到它的 KL 是：

$$
D_{\mathrm{KL}}\big( (1, 0) \,\Vert\, (0.5, 0.5) \big) = 1 \times \ln \frac{1}{0.5} = \ln 2 = 0.693 \text{ nat}
$$

正好 1 bit：观测告诉了模型"是 A 还是 B"，这 1 bit 的信息是省不掉的，先验已经做到了最好。

**高斯的情况。** 未来 A 的后验是 $\mathcal{N}(-2, 0.1^2)$，未来 B 的后验是 $\mathcal{N}(+2, 0.1^2)$。
两个高斯的混合是双峰的，不是高斯。单个高斯先验能做到的最好结果是匹配混合分布的均值和方差：
均值 0，方差 $0.1^2 + 2^2 = 4.01$。用 Phase 3 推过的两个高斯之间的 KL：

$$
D_{\mathrm{KL}} = \ln \frac{\sigma_p}{\sigma_q} + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{2 \sigma_p^2} - \frac{1}{2}
= \ln \frac{2.0025}{0.1} + \frac{0.01 + 4}{8.02} - \frac{1}{2} = 2.997 + 0.5 - 0.5 = 2.997 \text{ nat}
$$

同样是"二选一"的不确定性，高斯先验要付 3.0 nat，类别先验只付 0.69 nat。
多出来的 2.3 nat 是高斯先验表达不了双峰的代价。这部分 KL 降不下去，优化器只能去压缩后验里的信息来凑。

### 4.5 V2 的 actor：REINFORCE 回来了

Atari 的动作是离散的按键。离散动作的采样不可导，第 3 节的 pathwise 梯度到 actor 这里就断了。
V2 的 actor 损失把两种估计器写在一起，用系数 $\rho$ 切换（V2 论文的 actor 损失一式，换成本课记号）：

$$
\mathcal{L}_{\text{actor}}(\phi) = \mathbb{E}\Big[ \sum_{k=0}^{H-1} \Big( \underbrace{-\rho \, \ln \pi_\phi(\hat a_k \mid \hat m_k) \; \mathrm{sg}\big( R^\lambda_k - V_\psi(\hat m_k) \big)}_{\text{REINFORCE}} \; \underbrace{- \, (1 - \rho) \, R^\lambda_k}_{\text{沿模型反传}} \; \underbrace{- \, \eta \, \mathcal{H}\big[ \pi_\phi(\,\cdot \mid \hat m_k) \big]}_{\text{熵正则}} \Big) \Big]
$$

- **REINFORCE 项**就是 Phase 5 的策略梯度：$\nabla_\phi \ln \pi_\phi(a \mid m)$ 乘上"这个动作比平均好多少"。
  减去的 $V_\psi(\hat m_k)$ 是基线，不改变梯度的期望，只降方差。括号整体停止梯度，它只是一个权重。
  这一项不要求动作采样可导，也不要求世界模型对动作可导。
- **沿模型反传项**就是第 3 节的 V1 目标。
- **熵正则**鼓励探索，和 Phase 5 的 actor-critic、SAC 里的熵项是同一个东西。

论文的结论是：Atari 用 $\rho = 1$（纯 REINFORCE）最好，连续控制用 $\rho = 0$（纯沿模型反传）最好。
对离散动作也可以用 straight-through 让梯度"穿过"动作，但那样的梯度有偏，论文报告的效果不如 REINFORCE。

为什么同样在可微的世界模型里，离散动作反而选了高方差的 REINFORCE？
一个原因是无偏；另一个原因是在想象里样本很便宜。REINFORCE 的弱点是费样本，
而想象轨迹每次更新就有上万条转移，这个弱点在世界模型里被大幅冲淡了。

### 4.6 V2 的其他改动

| 改动 | 解决什么问题 | 承接 |
| --- | --- | --- |
| 类别隐变量 + straight-through | 高斯先验难以表达多峰的未来（第 4.4 节） | Phase 5 的 straight-through |
| KL balancing | KL 一视同仁地拉近先验和后验，会把表征往没学好的先验上拉 | 第 5 节 |
| actor 用 REINFORCE（离散动作） | 离散动作不可重参数化 | Phase 5 的策略梯度 |
| 目标网络（慢 critic） | λ-return 由 critic 自己算出，目标追着自己跑 | Phase 4 的 DQN 目标网络 |
| 折扣（continue）预测头 | V1 的主实验（连续控制）不会提前终止；Atari 会 | 第 2.4 节 |
| actor 的熵正则 | 离散动作的策略容易过早收敛到确定性 | Phase 5 的熵正则 |

目标网络这一条：V2 用一份慢更新的 critic 副本来算 λ-return 里的 $v_k$，每隔固定的更新步数从在线 critic 复制一次参数
（论文超参数表里是每 100 步，待核对）。V3 把它换成了另一种做法，见第 7 节。

### 4.7 算法

```
算法：类别隐变量的 straight-through 采样

输入：logits，形状 [..., 32, 32]
 1. probs ← softmax(logits, 最后一维)
 2. index ← 按 probs 逐行采样一个类别                    形状 [..., 32]
 3. onehot ← one_hot(index, 32)                         形状 [..., 32, 32]，不带梯度
 4. sample ← onehot + probs − sg(probs)                 前向等于 onehot，反向等于 probs 的梯度
 5. 返回 sample 展平最后两维                              形状 [..., 1024]
```

PyTorch 里第 4 步就是 `onehot + probs - probs.detach()`。

### 4.8 代码对应

对应 `04-categorical-latents-straight-through.py`。

| 名称 | 做什么 | 关键张量 |
| --- | --- | --- |
| `sample_categorical_st(logits)` | 第 4.7 节的算法 | 输入 `[B, T, 32, 32]`，输出 `[B, T, 1024]` |
| `CategoricalRSSM` | 把 Phase 3 的高斯 RSSM 改成类别版：先验头和后验头都输出 `[32, 32]` 的 logits | $h_t$ 为 `[B, D_h]`，$z_t$ 为 `[B, 1024]` |

讲义先在第 4.3 节的两个手算例子上核对 `sample_categorical_st()` 的梯度，
再做一个"两种未来"的玩具实验：让高斯先验和类别先验分别去拟合双峰的后验，比较最终的 KL。

> 📌 待回填：straight-through 梯度在两个手算例子上的数值（线性：应精确等于真实梯度；非线性：多次采样的平均应接近 $(-1.344, -0.96, 2.304)$）；高斯先验与类别先验在双峰玩具问题上收敛后的 KL，来自 `04-categorical-latents-straight-through.py`。

### 4.9 常见误区

- **误区：离散隐变量的表达能力比连续的弱。**
  实际情况：$32 \times 32$ 的类别隐变量有 $2^{160}$ 种取值。关键差别不在容量，在于先验能不能表达多峰。
- **误区：straight-through 就是把采样去掉，直接用 softmax 的概率。**
  实际情况：前向用的是硬的 one-hot 样本，下游网络从不会见到"0.3 个第 2 类"这种输入。只有反向时借用了概率的梯度。
  如果前向也用软概率，训练和想象时下游看到的输入分布就不一致了。
- **误区：straight-through 的梯度是对的，因为训练能收敛。**
  实际情况：它有偏（第 4.3 节的第二个例子）。能用是经验事实，不是理论保证。
- **误区：V2 抛弃了沿模型反传。**
  实际情况：V2 在连续控制上仍然用它（$\rho = 0$）。换成 REINFORCE 的只是离散动作的任务。

---

## 5. KL balancing 与 free bits

### 5.1 直觉

把 RSSM 想成两个要对齐接口的团队：

- **表征团队**（编码器，后验）：看了当前画面，产出 $z_t$。
- **预测团队**（先验）：不看当前画面，只凭历史预测 $z_t$ 会是什么。

KL 项衡量两个团队的输出差多少。想象的时候没有画面可看，只能用预测团队的输出，所以两边必须对齐。
对齐有两种办法：

1. 预测团队把预测做准。这是我们想要的。
2. 表征团队把输出改得更好猜：少放信息，多放废话。极端情况下输出一个常数，预测团队百分之百猜中，KL 为 0。

第 2 种办法在适度时是有益的（表征不该塞满无法预测的噪声），过度了就是灾难：
表征里没有信息了，解码器重建不出画面，奖励头也预测不了奖励。

原始的 KL 项不区分这两种办法，哪个容易走哪个。训练初期预测团队还很弱，容易的恰恰是第 2 种。
本节的两个技巧分别针对这个问题：

- **KL balancing**：规定"主要由预测团队来迁就"。把 KL 拆成两份，预测团队那份权重大，表征团队那份权重小。
- **free bits**：给一个误差预算。差异小于 1 nat 时不再追究，免得表征团队为了最后一点 KL 把信息扔光。

### 5.2 定义：类别分布之间的 KL

单个类别变量，后验 $q = (q_1, \dots, q_K)$，先验 $p = (p_1, \dots, p_K)$：

$$
D_{\mathrm{KL}}(q \,\Vert\, p) = \sum_{k=1}^{K} q_k \ln \frac{q_k}{p_k}
$$

约定 $0 \ln 0 = 0$。$z_t$ 由 $N_z$ 个独立的类别变量组成时，总的 KL 是各变量的 KL 之和（下一小节推导）：

$$
D_{\mathrm{KL}}\big( q_\theta(z_t \mid h_t, o_t) \,\Vert\, p_\theta(z_t \mid h_t) \big) = \sum_{i=1}^{N_z} \sum_{k=1}^{K} q^{(i)}_k \ln \frac{q^{(i)}_k}{p^{(i)}_k}
$$

方向是"后验在前，先验在后"，这是序列 ELBO（Phase 3）里 KL 项的方向。

单位是 nat（用自然对数）。$1 \text{ nat} = 1 / \ln 2 = 1.443 \text{ bit}$。
"free bits"这个名字里虽然有 bit，Dreamer V3 的阈值是 1 nat。

### 5.3 推导：类别 KL 及其手算

**第一步：从 KL 的定义出发。** 对任意两个分布：

$$
D_{\mathrm{KL}}(q \,\Vert\, p) = \mathbb{E}_{z \sim q}\big[ \ln q(z) - \ln p(z) \big]
$$

**第二步：代入独立性。** $z = (z^{(1)}, \dots, z^{(N_z)})$，各分量独立，联合概率是连乘，取对数变成求和：

$$
\ln q(z) - \ln p(z) = \sum_{i=1}^{N_z} \Big[ \ln q^{(i)}(z^{(i)}) - \ln p^{(i)}(z^{(i)}) \Big]
$$

**第三步：期望和求和交换顺序。** 第 $i$ 项只依赖 $z^{(i)}$，对其他分量的期望直接积掉：

$$
D_{\mathrm{KL}}(q \,\Vert\, p) = \sum_{i=1}^{N_z} \mathbb{E}_{z^{(i)} \sim q^{(i)}}\Big[ \ln q^{(i)}(z^{(i)}) - \ln p^{(i)}(z^{(i)}) \Big] = \sum_{i=1}^{N_z} D_{\mathrm{KL}}\big( q^{(i)} \,\Vert\, p^{(i)} \big)
$$

**第四步：单个类别变量的期望是对 $K$ 个类别的加权求和。** $z^{(i)}$ 取第 $k$ 类的概率是 $q^{(i)}_k$：

$$
D_{\mathrm{KL}}\big( q^{(i)} \,\Vert\, p^{(i)} \big) = \sum_{k=1}^{K} q^{(i)}_k \big( \ln q^{(i)}_k - \ln p^{(i)}_k \big)
$$

写代码时两个对数概率都用 `log_softmax(logits)` 算，不要先 softmax 再取 log，后者在概率很小时数值不稳定。

**手算例子。** $K = 3$，后验 $q = (0.7, 0.2, 0.1)$，先验 $p = (0.5, 0.3, 0.2)$：

| 类别 $k$ | $q_k$ | $p_k$ | $\ln(q_k / p_k)$ | $q_k \ln(q_k / p_k)$ |
| --- | --- | --- | --- | --- |
| 1 | 0.7 | 0.5 | $\ln 1.4 = 0.3365$ | 0.2355 |
| 2 | 0.2 | 0.3 | $\ln 0.6667 = -0.4055$ | −0.0811 |
| 3 | 0.1 | 0.2 | $\ln 0.5 = -0.6931$ | −0.0693 |
| 合计 | | | | **0.0851 nat** |

单项可以是负的（第 2、3 类），总和一定非负。

把两个分布对调再算一次：

$$
D_{\mathrm{KL}}(p \,\Vert\, q) = 0.5 \ln \frac{0.5}{0.7} + 0.3 \ln \frac{0.3}{0.2} + 0.2 \ln \frac{0.2}{0.1} = -0.1682 + 0.1216 + 0.1386 = 0.0920 \text{ nat}
$$

$0.0851 \neq 0.0920$：KL 不对称。所以"谁在前谁在后"不能写反。

三条会用到的性质：

| 性质 | 说明 |
| --- | --- |
| 非负，且只在 $q = p$ 时为 0 | KL 为 0 意味着后验等于先验：看不看观测都一样 |
| 不对称 | 见上面的例子 |
| 先验某一类概率为 0、后验在该类概率大于 0 时，KL 为无穷大 | 第 5.7 节的 unimix 处理这个问题 |

### 5.4 推导：KL 对两边 logits 的梯度

KL balancing 要分别控制"先验这边"和"后验这边"的梯度。先把这两个梯度推出来。
只看一个类别变量，记先验的 logits 为 $\ell^p$，后验的 logits 为 $\ell^q$，$p = \mathrm{softmax}(\ell^p)$，$q = \mathrm{softmax}(\ell^q)$。

**对先验 logits 的梯度（后验当常数）。**

第一步，把 KL 里和 $p$ 有关的部分挑出来：

$$
D_{\mathrm{KL}}(q \,\Vert\, p) = \underbrace{\sum_k q_k \ln q_k}_{\text{与 } p \text{ 无关}} - \sum_k q_k \ln p_k
$$

第二步，代入 $\ln p_k = \ell^p_k - \ln \sum_{k'} e^{\ell^p_{k'}}$，并用 $\sum_k q_k = 1$：

$$
-\sum_k q_k \ln p_k = -\sum_k q_k \ell^p_k + \ln \sum_{k'} e^{\ell^p_{k'}}
$$

第三步，对 $\ell^p_j$ 求导。第一项的导数是 $-q_j$；第二项是 log-sum-exp，它的导数是 softmax，即 $p_j$：

$$
\frac{\partial D_{\mathrm{KL}}}{\partial \ell^p_j} = p_j - q_j
$$

和分类问题里交叉熵的梯度"预测减标签"是同一个式子。**对先验来说，后验就是它的软标签。**

**对后验 logits 的梯度（先验当常数）。**

第一步，记 $u_k = \ln q_k - \ln p_k$，则 $D_{\mathrm{KL}} = \sum_k q_k u_k$。$q_k$ 和 $u_k$ 都依赖 $\ell^q$，用乘积法则：

$$
\frac{\partial D_{\mathrm{KL}}}{\partial \ell^q_j} = \sum_k \frac{\partial q_k}{\partial \ell^q_j} \, u_k + \sum_k q_k \, \frac{\partial \ln q_k}{\partial \ell^q_j}
$$

第二步，代入 softmax 的雅可比 $\frac{\partial q_k}{\partial \ell^q_j} = q_k(\delta_{kj} - q_j)$，以及 $\frac{\partial \ln q_k}{\partial \ell^q_j} = \delta_{kj} - q_j$：

$$
\frac{\partial D_{\mathrm{KL}}}{\partial \ell^q_j} = \sum_k q_k (\delta_{kj} - q_j) \, u_k + \sum_k q_k (\delta_{kj} - q_j)
$$

第三步，逐项化简。第一个求和等于 $q_j u_j - q_j \sum_k q_k u_k = q_j u_j - q_j D_{\mathrm{KL}}$；第二个求和等于 $q_j - q_j \sum_k q_k = 0$：

$$
\frac{\partial D_{\mathrm{KL}}}{\partial \ell^q_j} = q_j \Big( \ln \frac{q_j}{p_j} - D_{\mathrm{KL}}(q \,\Vert\, p) \Big)
$$

**手算例子（接上例）。** $q = (0.7, 0.2, 0.1)$，$p = (0.5, 0.3, 0.2)$，$D_{\mathrm{KL}} = 0.0851$。

先验一侧：

$$
\frac{\partial D_{\mathrm{KL}}}{\partial \ell^p} = p - q = (-0.2, \; 0.1, \; 0.1)
$$

梯度下降会抬高第 1 类的 logit、压低第 2、3 类，先验朝后验靠拢。

后验一侧，先算 $\ln(q_j / p_j) - D_{\mathrm{KL}}$，得到 $(0.2513, -0.4906, -0.7783)$，再乘 $q_j$：

$$
\frac{\partial D_{\mathrm{KL}}}{\partial \ell^q} = (0.7 \times 0.2513, \;\; 0.2 \times (-0.4906), \;\; 0.1 \times (-0.7783)) = (0.1759, \; -0.0981, \; -0.0778)
$$

梯度下降会压低第 1 类的 logit、抬高第 2、3 类，后验朝先验靠拢。三个分量之和为 0（在舍入误差内）。

两个梯度做的是同一件事的两面：一个让先验去追后验，一个让后验去迁就先验。

### 5.5 KL balancing：把 KL 拆成两半

**第一步：一个关于梯度的恒等式。** KL 是两个分布的函数，两个分布都依赖 $\theta$。
多元函数的全导数等于对每个自变量的偏导数之和。用 sg 把"只对其中一个自变量求导"写出来就是：

$$
\nabla_\theta \, D_{\mathrm{KL}}(q_\theta \,\Vert\, p_\theta) = \underbrace{\nabla_\theta \, D_{\mathrm{KL}}\big( \mathrm{sg}(q_\theta) \,\Vert\, p_\theta \big)}_{\text{只动先验：先验追后验}} + \underbrace{\nabla_\theta \, D_{\mathrm{KL}}\big( q_\theta \,\Vert\, \mathrm{sg}(p_\theta) \big)}_{\text{只动后验：后验迁就先验}}
$$

右边两项的**数值**相同（sg 不改变前向的值），都等于 $D_{\mathrm{KL}}(q \,\Vert\, p)$；**梯度**不同，分别是上一小节推出的两个。

**第二步：给两项不同的权重。** Dreamer V2 的写法（论文的 KL balancing 算法，$\alpha = 0.8$）：

$$
\mathcal{L}_{\text{KL}} = \alpha \, D_{\mathrm{KL}}\big( \mathrm{sg}(q_\theta) \,\Vert\, p_\theta \big) + (1 - \alpha) \, D_{\mathrm{KL}}\big( q_\theta \,\Vert\, \mathrm{sg}(p_\theta) \big)
$$

- $\alpha = 0.5$ 时，梯度是原始 KL 梯度的一半，两边比例不变，只是整体缩放。
- $\alpha = 0.8$ 时，先验一侧的梯度权重 0.8，后验一侧 0.2，比例是 4 : 1。数值上 $\mathcal{L}_{\text{KL}}$ 仍然等于 $D_{\mathrm{KL}}$。

V2 论文对动机的说明：KL 有两个作用，一是训练先验去靠近表征，二是把表征往先验上正则化。
学转移规律本来就难，不希望表征被拉向一个还没训好的先验。所以让先验学得快，让表征让步让得慢。

**第三步：V3 的写法。** V3 把两项起了名字，各带一个独立的权重，不再要求权重之和为 1（论文的 World Model Learning 一节）：

$$
\mathcal{L}_{\text{dyn}} = D_{\mathrm{KL}}\big( \mathrm{sg}(q_\theta) \,\Vert\, p_\theta \big), \qquad
\mathcal{L}_{\text{rep}} = D_{\mathrm{KL}}\big( q_\theta \,\Vert\, \mathrm{sg}(p_\theta) \big)
$$

（这里先省略 free bits 的裁剪，下一小节加上。）

| 名称 | 式子 | 训练谁 | 权重 |
| --- | --- | --- | --- |
| 动力学损失 $\mathcal{L}_{\text{dyn}}$ | $D_{\mathrm{KL}}(\mathrm{sg}(q) \,\Vert\, p)$ | 先验：学会预测表征 | $\beta_{\text{dyn}}$ |
| 表征损失 $\mathcal{L}_{\text{rep}}$ | $D_{\mathrm{KL}}(q \,\Vert\, \mathrm{sg}(p))$ | 后验：让表征更可预测 | $\beta_{\text{rep}} = 0.1$ |

关于 $\beta_{\text{dyn}}$：论文 2023 年的 arXiv 初版写的是 0.5，之后的版本（含 2025 年的 Nature 版）是 1（待核对各版本的超参数表）。
两种设置下先验与后验两侧的梯度比例分别是 5 : 1 和 10 : 1，结论一致：先验一侧重得多。

**手算例子（接上例）。** 按 V2 的 $\alpha = 0.8$ 给第 5.4 节的两个梯度加权：

$$
\text{先验 logits 的梯度：} 0.8 \times (-0.2, \; 0.1, \; 0.1) = (-0.16, \; 0.08, \; 0.08)
$$

$$
\text{后验 logits 的梯度：} 0.2 \times (0.1759, \; -0.0981, \; -0.0778) = (0.0352, \; -0.0196, \; -0.0156)
$$

损失的数值还是 $0.8 \times 0.0851 + 0.2 \times 0.0851 = 0.0851$。看日志看不出 KL balancing 有没有生效，要看梯度。

### 5.6 free bits：小于 1 nat 的差异不再追究

**定义。** V3 把两项 KL 各自在 1 nat 处从下方截断（论文的 World Model Learning 一节）：

$$
\mathcal{L}_{\text{dyn}} = \max\Big( 1, \; D_{\mathrm{KL}}\big( \mathrm{sg}(q_\theta) \,\Vert\, p_\theta \big) \Big), \qquad
\mathcal{L}_{\text{rep}} = \max\Big( 1, \; D_{\mathrm{KL}}\big( q_\theta \,\Vert\, \mathrm{sg}(p_\theta) \big) \Big)
$$

**它对梯度做了什么。** $\max(1, x)$ 对 $x$ 的导数是：

$$
\frac{d}{dx} \max(1, x) =
\begin{cases}
0 & x < 1 \\
1 & x > 1
\end{cases}
$$

所以 KL 小于 1 nat 时，这一项成了常数 1，没有梯度；大于 1 nat 时和不截断完全一样。
相当于一个开关：**KL 已经够小了就把 KL 损失关掉，让世界模型把全部力气花在预测损失上。**

**为什么需要它。** 光看 KL 项，最优解是后验等于先验、KL 为 0。这时 $z_t$ 不携带任何来自观测的信息，
也就是旧课讲 VAE 时提到的后验坍塌。V3 论文的说法是：为了避免"动力学很好预测、但表征没有包含足够输入信息"的退化解，
用 free bits 把两项损失在 1 nat 以下截断。Phase 3 的 PlaNet 和 Dreamer V1 也用了同样的技巧，那里叫 free nats，阈值是 3。

V3 论文还指出它和小的 $\beta_{\text{rep}}$ 配合解决了一个调参难题：以前的世界模型要按环境的视觉复杂度调 KL 的权重。
3D 环境里有很多和控制无关的细节，需要强一点的正则把表征简化；2D 游戏里个别像素就可能决定成败，需要弱一点的正则保住细节。
free bits 加上很小的 $\beta_{\text{rep}}$ 让同一组超参数在两类环境里都能用。

**手算例子。** 一批数据里三个位置的 KL（已对 32 个类别变量求和）分别是 0.3、0.9、2.5 nat。

逐元素截断，再取平均：

$$
\max(1, 0.3) = 1, \quad \max(1, 0.9) = 1, \quad \max(1, 2.5) = 2.5, \qquad \text{平均} = \frac{1 + 1 + 2.5}{3} = 1.5
$$

只有第三个位置有梯度。前两个位置的 KL 已经在预算之内，不再被往下压。

对比另一种写法，先取平均，再截断：

$$
\text{平均} = \frac{0.3 + 0.9 + 2.5}{3} = 1.233, \qquad \max(1, 1.233) = 1.233
$$

这时三个位置都有梯度，包括已经很小的 0.3。再看一组数：KL 为 0.3、0.9、1.2 时，平均是 0.8，截断后是常数 1，
三个位置都没有梯度，尽管第三个已经超出预算。

两种写法的行为不同。Dreamer V3 的官方实现是逐元素截断；更早的实现是先平均再截断（待核对）。
写代码时选定一种，并在注释里写明。

还有一个容易弄错的地方：阈值 1 nat 针对的是 **32 个类别变量加总之后**的 KL，不是每个变量各 1 nat（按官方实现，待核对）。
对比第 4.2 节算过的容量上限 110.9 nat，1 nat 是一个很小的预算。

### 5.7 KL 会不会变成无穷大：给 unimix 埋个伏笔

第 5.3 节的第三条性质：先验在某一类上的概率为 0，而后验在这一类上有概率，KL 就是无穷大。
神经网络输出的概率不会精确为 0，但 logits 饱和时可以非常接近 0，KL 会出现巨大的尖峰，一次更新就可能把训练带崩。

V3 的对策是 unimix：把网络输出的类别分布和均匀分布按 99% : 1% 混合，

$$
p'_k = 0.99 \, p_k + \frac{0.01}{K}
$$

先验和后验都这样处理。混合之后任何一类的概率至少是 $0.01 / K$，KL 有了上界。

**手算例子。** $K = 4$，取最坏情况：后验把全部概率放在第 1 类，先验把全部概率放在第 2 类。不混合时 KL 为无穷大。混合之后：

$$
q' = (0.9925, \; 0.0025, \; 0.0025, \; 0.0025), \qquad p' = (0.0025, \; 0.9925, \; 0.0025, \; 0.0025)
$$

$$
D_{\mathrm{KL}}(q' \,\Vert\, p') = 0.9925 \ln \frac{0.9925}{0.0025} + 0.0025 \ln \frac{0.0025}{0.9925} + 0 + 0 = (0.9925 - 0.0025) \times \ln 397 = 0.99 \times 5.984 = 5.92 \text{ nat}
$$

从无穷大变成了 5.92 nat。$K = 32$ 时的上界在第 7.4 节算。

### 5.8 算法

```
算法：带 KL balancing 和 free bits 的 KL 损失（V3 的写法）

输入：后验 logits、先验 logits，形状均为 [B, T, 32, 32]
函数 categorical_kl(logits_q, logits_p)：
     log_q ← log(unimix(softmax(logits_q)))；log_p ← log(unimix(softmax(logits_p)))
     返回 sum(exp(log_q) · (log_q − log_p))，对最后两维求和               形状 [B, T]

 1. kl_dyn ← categorical_kl(sg(后验 logits), 先验 logits)                只训练先验
 2. kl_rep ← categorical_kl(后验 logits, sg(先验 logits))                只训练后验
 3. loss_dyn ← max(1, kl_dyn)；loss_rep ← max(1, kl_rep)                 逐元素截断
 4. 返回 β_dyn · mean(loss_dyn) + β_rep · mean(loss_rep)
```

### 5.9 代码对应

对应 `05-kl-balancing-and-free-bits.py`。

| 函数 | 做什么 | 关键张量 |
| --- | --- | --- |
| `categorical_kl(logits_q, logits_p)` | 第 5.3 节的式子，对 $N_z$ 和 $K$ 两维求和 | 输入 `[B, T, 32, 32]`，输出 `[B, T]` |
| `kl_loss_balanced(post_logits, prior_logits, beta_dyn, beta_rep, free)` | 第 5.8 节的算法 | 输出标量，外加未截断的 KL 供日志使用 |

日志里要记**截断之前**的 KL。截断之后的数值在 KL 小于 1 时恒为 1，什么也看不出来。

> 📌 待回填：`categorical_kl()` 在第 5.3 节手算例子上的输出（应为 0.0851 与 0.0920）；有无 KL balancing、有无 free bits 四种组合下，训练过程中 KL 与重建损失的曲线（预期：没有 free bits 且 $\beta_{\text{rep}}$ 较大时 KL 被压到接近 0，重建变差），来自 `05-kl-balancing-and-free-bits.py`。

### 5.10 常见误区

- **误区：拆成两半之后两项数值一样，所以拆不拆没区别。**
  实际情况：数值一样，梯度不一样。一项只更新先验，一项只更新后验，权重不同就改变了"谁迁就谁"。
- **误区：KL 越小，世界模型越好。**
  实际情况：KL 为 0 意味着后验等于先验，观测没有给表征带来任何信息，这是坍塌。
  健康的状态是 KL 稳定在一个不为 0 的值上，同时重建和奖励预测是准的。
- **误区：free bits 是为了让训练更快。**
  实际情况：它是为了防止表征被 KL 压垮。KL 低于阈值时关掉 KL 的梯度，把信息留在表征里。
- **误区：$\mathrm{sg}(q)$ 表示编码器完全不受 $\mathcal{L}_{\text{dyn}}$ 影响。**
  实际情况：sg 包住的是第 $t$ 步后验分布的参数。先验的输入 $h_t$ 依赖上一步采样的 $z_{t-1}$，
  梯度仍然可以经过 $h_t$ 流到更早时间步的编码器。第 8.5 节有完整的梯度流向表。

---

## 6. Dreamer V3（一）：symlog 与 two-hot

### 6.1 直觉

Dreamer V3（Hafner et al. 2023，arXiv:2301.04104）的目标是：**一套超参数，不改一个数，在各种任务上都能跑。**
拦路的头号问题是数值的尺度。不同任务的奖励可以差好几个数量级：有的每步 0.001，有的一次给 1000；
向量观测也一样，有的是归一化的关节角，有的是没处理过的坐标或计数。

做过监控面板的人都遇到过同样的问题：一张图上既有 QPS 是 0.5 的服务，又有 QPS 是 50000 的服务。
线性坐标轴下，小的那条线贴在底上什么也看不见。解决办法是换成对数坐标轴；数据里有负数或 0 时，
用**对称对数轴**（matplotlib 里这种刻度就叫 `symlog`）：零附近是线性的，远处是对数的，正负对称。
V3 的 symlog 变换就是这个东西。

第二个工具来自另一个熟悉的做法：监控系统存延迟分布时，不存每个请求的精确延迟，而是存"落在哪个桶里"的计数（直方图）。
V3 预测奖励和价值时也不直接输出一个数，而是输出"落在各个桶里的概率"，把回归做成分类。
目标值落在两个桶之间时，按距离把它分摊到相邻的两个桶上，这就是 **two-hot 编码**。

### 6.2 定义：symlog 与 symexp

$$
\mathrm{symlog}(x) = \mathrm{sign}(x) \, \ln\big( \vert x \vert + 1 \big), \qquad
\mathrm{symexp}(x) = \mathrm{sign}(x) \, \big( e^{\vert x \vert} - 1 \big)
$$

```
     symlog(x)
        │                                   ·  ·  ·  ·     远处：对数增长，大数值被压缩
      4 ┤                         ·   ·
        │                  ·
      2 ┤            ·
        │        ·
        │     ·           零附近：斜率为 1，约等于恒等映射
      0 ┼───·──────────────────────────────────────────▶ x
        │ ·   1    10                  100
     -2 ┤                 （负半轴与正半轴关于原点对称）
```

几个值：

| $x$ | 0 | 0.1 | 1 | 10 | 100 | 1000 | −1000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| $\mathrm{symlog}(x)$ | 0 | 0.0953 | 0.6931 | 2.3979 | 4.6151 | 6.9088 | −6.9088 |

从 0.1 到 1000 跨了四个数量级，变换之后落在 0.1 到 7 之间。

### 6.3 推导：symlog 的性质

**性质一：symexp 是 symlog 的逆。** 分 $x \ge 0$ 和 $x < 0$ 两种情况验证。

$x \ge 0$ 时，$\mathrm{symlog}(x) = \ln(x + 1) \ge 0$。把它代入 symexp（自变量非负，符号为正，绝对值就是它本身）：

$$
\mathrm{symexp}\big( \ln(x + 1) \big) = e^{\ln(x + 1)} - 1 = (x + 1) - 1 = x
$$

$x < 0$ 时，$\mathrm{symlog}(x) = -\ln(-x + 1) < 0$。代入 symexp（自变量为负，符号为负，绝对值是 $\ln(-x + 1)$）：

$$
\mathrm{symexp}\big( -\ln(-x + 1) \big) = -\big( e^{\ln(-x + 1)} - 1 \big) = -(-x + 1 - 1) = x
$$

反方向 $\mathrm{symlog}(\mathrm{symexp}(x)) = x$ 的验证完全类似。所以两者互为逆函数，变换不丢信息。

**性质二：零附近是恒等映射，远处是对数。** 对 $x > 0$ 求导：

$$
\frac{d}{dx} \, \mathrm{symlog}(x) = \frac{d}{dx} \ln(x + 1) = \frac{1}{x + 1}
$$

由对称性，一般情况是 $\frac{1}{\vert x \vert + 1}$。$x = 0$ 处导数为 1，所以小数值几乎原样通过（$\mathrm{symlog}(0.1) = 0.0953$）；
$\vert x \vert$ 很大时导数趋于 0，大数值被压缩。

**性质三：它是奇函数**，$\mathrm{symlog}(-x) = -\mathrm{symlog}(x)$。正负奖励被同等对待。

和几个替代方案的比较：

| 方案 | 问题 |
| --- | --- |
| 直接取 $\log$ | 0 和负数没有定义 |
| 把奖励截断到 $[-1, 1]$（DQN 的做法） | 不可逆。奖励 1 和奖励 100 变得一样，改变了任务本身 |
| 用滑动均值和标准差归一化 | 统计量随训练变化，目标不平稳；奖励稀疏时标准差接近 0，除以它会放大噪声 |
| symlog | 无状态、可逆、零附近不失真 |

**性质四：用 symlog 做回归，梯度不再随目标的尺度线性增长。**
让网络输出 $f$ 去拟合 $\mathrm{symlog}(y)$，损失是平方误差（V3 论文的 symlog predictions）：

$$
\mathcal{L} = \frac{1}{2} \big( f - \mathrm{sg}(\mathrm{symlog}(y)) \big)^2, \qquad \frac{\partial \mathcal{L}}{\partial f} = f - \mathrm{symlog}(y)
$$

需要原始尺度的预测时取 $\hat y = \mathrm{symexp}(f)$。和直接在原始空间做平方误差比较，设网络初始输出为 0：

| 目标 $y$ | 原始空间平方误差的梯度 $f - y$ | symlog 空间平方误差的梯度 $f - \mathrm{symlog}(y)$ |
| --- | --- | --- |
| 0.1 | −0.1 | −0.095 |
| 1000 | −1000 | −6.91 |
| 两者之比 | 10000 倍 | 约 72 倍 |

同一个学习率，在原始空间里要么对小目标太慢，要么对大目标发散。symlog 之后两类任务的梯度在同一个量级。

**代价：大数值上只保证相对精度。** symlog 空间里 0.1 的误差，换算回原始空间是多少，取决于数值本身的大小：

- 在 $y = 1$ 附近：$\mathrm{symexp}(0.6931 + 0.1) = e^{0.7931} - 1 = 1.21$，误差约 0.21。
- 在 $y = 1000$ 附近：$\mathrm{symexp}(6.9088 + 0.1) = 1001 \times e^{0.1} - 1 = 1105$，误差约 105。

对奖励和价值来说这通常是合理的：区分 1000 和 1105 没有区分 0 和 0.2 那么要紧。

### 6.4 symlog 用在哪

| 位置 | 做法 | 说明 |
| --- | --- | --- |
| 编码器的输入（向量观测） | 先做 symlog 再送进 MLP | 输入的尺度不再需要人工归一化 |
| 解码器的输出（向量观测） | 预测 $\mathrm{symlog}(o)$，用平方误差 | 上一小节的性质四 |
| 奖励头、critic | 配合 two-hot 做离散回归 | 第 6.5–6.8 节 |
| 图像观测 | 不用 symlog | 像素本来就在固定范围内，缩放到 $[0, 1]$ 即可 |

### 6.5 定义：two-hot 编码

取 $M$ 个从小到大排好的桶的位置 $b_1 < b_2 < \dots < b_M$。对一个落在 $[b_i, b_{i+1}]$ 之间的标量 $x$，
它的 two-hot 编码是一个 $M$ 维向量，只有第 $i$ 和第 $i+1$ 两个分量非零：

$$
\mathrm{twohot}(x)_j =
\begin{cases}
\dfrac{b_{i+1} - x}{b_{i+1} - b_i} & j = i \\[2ex]
\dfrac{x - b_i}{b_{i+1} - b_i} & j = i + 1 \\[2ex]
0 & \text{其他}
\end{cases}
$$

```
     桶的位置     b_1     b_2     b_3     b_4     b_5
                  │       │       │       │       │
     目标值 x                             │   ×   │         x 落在 b_4 和 b_5 之间，离 b_5 稍近
                                          │       │
     two-hot      0       0       0      0.496   0.504      离谁近，谁的权重大
```

one-hot 是"把全部权重放在一个桶上"，只能表示桶的位置上的值。
two-hot 是"在相邻两个桶之间做线性插值"，可以精确表示两个桶之间的任何值。

超出范围的 $x$ 截断到最两端的桶。V3 用 $M = 255$ 个桶，覆盖 symlog 空间里的 $[-20, 20]$（论文的 Robust Predictions 部分）。
$\mathrm{symexp}(20) = e^{20} - 1 \approx 4.85 \times 10^8$，所以这个范围对奖励和回报来说足够宽。

### 6.6 推导：two-hot 的期望恰好还原目标值

把 two-hot 向量看成桶上的一个概率分布。要证明两件事：它确实是概率分布；它的期望等于 $x$。
记桶宽 $\Delta = b_{i+1} - b_i > 0$，两个非零权重为 $\omega_i = \frac{b_{i+1} - x}{\Delta}$ 和 $\omega_{i+1} = \frac{x - b_i}{\Delta}$。

**第一步：非负。** 因为 $b_i \le x \le b_{i+1}$，两个分子都非负。

**第二步：和为 1。**

$$
\omega_i + \omega_{i+1} = \frac{(b_{i+1} - x) + (x - b_i)}{\Delta} = \frac{b_{i+1} - b_i}{\Delta} = 1
$$

**第三步：期望等于 $x$。** 期望是"桶的位置乘以权重"之和：

$$
\begin{aligned}
\sum_j \mathrm{twohot}(x)_j \, b_j &= \omega_i \, b_i + \omega_{i+1} \, b_{i+1} \\
&= \frac{(b_{i+1} - x) \, b_i + (x - b_i) \, b_{i+1}}{\Delta} \\
&= \frac{b_i b_{i+1} - x \, b_i + x \, b_{i+1} - b_i b_{i+1}}{\Delta} \\
&= \frac{x \, (b_{i+1} - b_i)}{\Delta} = x
\end{aligned}
$$

第三行把分子展开，第四行消去 $b_i b_{i+1}$ 并提出 $x$。

这个性质是 two-hot 回归能成立的根基：**网络只要把预测的分布拟合到 two-hot 目标上，用"概率加权平均"读出来的值就精确等于目标值。**
离散化没有引入量化误差。

**手算例子。** 为了好算，只用 5 个桶，放在 symlog 空间里：$b = (-2, -1, 0, 1, 2)$。目标奖励 $y = 3.5$。

第一步，变换到 symlog 空间：$x = \mathrm{symlog}(3.5) = \ln 4.5 = 1.5041$。

第二步，找到它所在的区间：$b_4 = 1 \le 1.5041 \le b_5 = 2$，桶宽为 1。

第三步，算两个权重：

$$
\omega_4 = \frac{2 - 1.5041}{1} = 0.4959, \qquad \omega_5 = \frac{1.5041 - 1}{1} = 0.5041
$$

所以 $\mathrm{twohot}(x) = (0, 0, 0, 0.4959, 0.5041)$。

第四步，验证能还原：

$$
0.4959 \times 1 + 0.5041 \times 2 = 1.5041, \qquad \mathrm{symexp}(1.5041) = e^{1.5041} - 1 = 4.5 - 1 = 3.5
$$

### 6.7 推导：交叉熵损失和它有界的梯度

网络输出 $M$ 个 logits，过 softmax 得到预测分布 $\hat p$。损失是预测分布和 two-hot 目标之间的交叉熵：

$$
\mathcal{L} = -\sum_{j=1}^{M} \mathrm{twohot}(x)_j \, \ln \hat p_j, \qquad \hat p = \mathrm{softmax}(\ell)
$$

预测值是分布的期望：

$$
\hat x = \sum_{j=1}^{M} \hat p_j \, b_j
$$

**梯度。** 记目标为 $\omega = \mathrm{twohot}(x)$。和第 5.4 节对先验 logits 求导的过程完全一样（把那里的 $q$ 换成 $\omega$，$p$ 换成 $\hat p$），结果是：

$$
\frac{\partial \mathcal{L}}{\partial \ell_j} = \hat p_j - \omega_j
$$

$\hat p_j$ 和 $\omega_j$ 都在 $[0, 1]$ 之间，所以**每个 logit 的梯度都在 $[-1, 1]$ 之间，和目标值的大小无关。**
目标是 0.01 还是 10000，梯度的量级一样。

对比平方误差：梯度是 $f - y$，目标越大梯度越大，没有上界（第 3.5 节）。
V3 论文把这一点概括为：离散回归把梯度的尺度和预测目标的尺度解耦了。

离散回归还有第二个好处：它预测的是一个分布，能表示多峰。稀疏奖励下，同一个状态之后"多数时候奖励是 0，偶尔是 1"，
平方误差只能学到一个很小的均值，而且被大量的 0 淹没；分类头可以直接在"0"和"1"两个桶上各放一部分概率。

**手算例子（接上例）。** 目标 $\omega = (0, 0, 0, 0.4959, 0.5041)$，网络当前的预测分布是 $\hat p = (0.05, 0.05, 0.1, 0.5, 0.3)$。

损失：

$$
\mathcal{L} = -0.4959 \times \ln 0.5 - 0.5041 \times \ln 0.3 = 0.4959 \times 0.6931 + 0.5041 \times 1.2040 = 0.3437 + 0.6069 = 0.9506
$$

对 logits 的梯度：

$$
\hat p - \omega = (0.05, \; 0.05, \; 0.1, \; 0.0041, \; -0.2041)
$$

梯度下降会抬高第 5 个桶的 logit、压低前三个桶的 logit。

当前的预测值：

$$
\hat x = 0.05 \times (-2) + 0.05 \times (-1) + 0.1 \times 0 + 0.5 \times 1 + 0.3 \times 2 = 0.95, \qquad \mathrm{symexp}(0.95) = e^{0.95} - 1 = 1.586
$$

目标是 3.5，当前预测 1.586，偏低。

**损失的最小值不是 0。** 预测分布恰好等于目标时，交叉熵等于目标分布自身的熵：

$$
\mathcal{L}_{\min} = -0.4959 \ln 0.4959 - 0.5041 \ln 0.5041 = 0.3478 + 0.3453 = 0.6931
$$

目标落在两个桶正中间时这个值最大，为 $\ln 2$；恰好落在某个桶上时为 0。看 two-hot 损失的曲线时要记得这一点。

**和 C51 的关系。** Phase 4 的 C51 也让价值网络输出一组桶上的概率。区别在于：
C51 要把整个回报分布经过 Bellman 更新之后再投影回固定的桶上；V3 的 critic 只是把一个标量目标（λ-return）编码成 two-hot，
用的时候也只取分布的期望。它借用了分布式价值的参数化，目的是得到一个尺度无关的回归损失。

### 6.8 两种读出方式：桶在哪个空间里等间距

"symlog + two-hot"有两种组合方式，V3 论文的不同版本各用了一种（哪个版本用哪种，待核对）。

| | 方式 A | 方式 B |
| --- | --- | --- |
| 桶的位置 | 在 symlog 空间等间距 | 把等间距的点做 symexp，得到原始空间里指数间距的桶 |
| 编码目标 | $\mathrm{twohot}(\mathrm{symlog}(y))$ | $\mathrm{twohot}(y)$ |
| 读出预测 | $\hat y = \mathrm{symexp}\big( \sum_j \hat p_j b_j \big)$ | $\hat y = \sum_j \hat p_j \, \mathrm{symexp}(b_j)$ |
| 期望在哪个空间取 | symlog 空间 | 原始空间 |

第 6.6 节的手算例子是方式 A。同一个例子用方式 B：桶的位置是 $\mathrm{symexp}(b) = (-6.389, -1.718, 0, 1.718, 6.389)$，
目标 $y = 3.5$ 落在 1.718 和 6.389 之间：

$$
\omega_4 = \frac{6.389 - 3.5}{6.389 - 1.718} = \frac{2.889}{4.671} = 0.6185, \qquad \omega_5 = 1 - 0.6185 = 0.3815
$$

验证：$0.6185 \times 1.718 + 0.3815 \times 6.389 = 1.0626 + 2.4374 = 3.5$。

目标是一个确定的数时，两种方式都能精确还原。差别出现在**目标本身有随机性**的时候。
设某个状态之后的奖励一半是 0、一半是 6.389，网络学到了正确的分布：在"0"和"6.389"两个桶上各 0.5。

- 方式 A 的读出：symlog 空间里的期望是 $0.5 \times 0 + 0.5 \times 2 = 1$，再做 symexp 得 $e^1 - 1 = 1.718$。
- 方式 B 的读出：$0.5 \times 0 + 0.5 \times 6.389 = 3.195$。

真实的期望奖励是 3.195。方式 A 给出 1.718，偏低。原因是 symexp 是凸函数，"先取期望再变换"不等于"先变换再取期望"（Jensen 不等式）。
价值函数的定义是回报的**期望**，所以方式 B 在随机环境里才是对的。本课的讲义实现方式 B，并用这个例子做单元测试。

### 6.9 算法

```
算法：two-hot 离散回归头（方式 B）

常量：bins ← symexp(linspace(−20, 20, 255))                              形状 [255]

函数 twohot_encode(y)：                                                   y 形状 [...]
     y ← clip(y, bins[0], bins[−1])
     i ← 满足 bins[i] ≤ y 的最大下标，并限制在 [0, 253]                     形状 [...]
     w_hi ← (y − bins[i]) / (bins[i+1] − bins[i])；w_lo ← 1 − w_hi
     返回在下标 i 处为 w_lo、下标 i+1 处为 w_hi、其余为 0 的向量             形状 [..., 255]

函数 loss(logits, y)：
     返回 −sum(twohot_encode(sg(y)) · log_softmax(logits))                 形状 [...]

函数 predict(logits)：
     返回 sum(softmax(logits) · bins)                                      形状 [...]
```

V3 还把奖励头和 critic 最后一层的权重初始化为 0（论文的 Robust Predictions 部分）。
logits 全为 0 时 softmax 是均匀分布，而桶关于 0 对称，所以初始预测恰好是 0。
随机初始化的 critic 会在训练一开始给出一堆随机的大价值，actor 会去追这些并不存在的回报；零初始化避免了这一点。

### 6.10 代码对应

对应 `06-symlog-and-twohot.py`。

| 名称 | 做什么 | 关键张量 |
| --- | --- | --- |
| `symlog(x)`、`symexp(x)` | 第 6.2 节的两个函数 | 逐元素，形状不变 |
| `twohot_encode(y, bins)` | 第 6.9 节 | 输入 `[H, N]`，输出 `[H, N, 255]` |
| `TwoHotHead` | MLP + 255 维输出；提供 `loss()` 和 `predict()` | logits 为 `[H, N, 255]` |

讲义的实验：造一个目标尺度跨多个数量级的回归任务，比较原始平方误差、symlog 平方误差、two-hot 三种做法在同一学习率下的表现。

> 📌 待回填：`twohot_encode()` 在第 6.6、6.8 节手算例子上的输出；三种回归方式在多尺度目标上的损失曲线和最终相对误差（预期：原始平方误差在大目标上发散或在小目标上学不动，另外两种稳定）；方式 A 与方式 B 在随机目标上的读出值（预期：方式 A 偏低），来自 `06-symlog-and-twohot.py`。

### 6.11 常见误区

- **误区：symlog 就是取对数。**
  实际情况：它在 0 附近是恒等映射，对 0 和负数都有定义，而且是奇函数。只有在数值很大时才表现得像对数。
- **误区：two-hot 把连续值离散化了，预测只能落在桶的位置上，精度有损失。**
  实际情况：预测是按概率对桶的位置做加权平均，可以取到两个桶之间的任何值。第 6.6 节证明了编码再读出是精确的。
- **误区：two-hot 的交叉熵应该能降到 0。**
  实际情况：最小值是目标分布自身的熵，最大可到 $\ln 2 = 0.693$。损失停在零点几不代表没学好，要看读出的预测值和目标的差。
- **误区：critic 用了 symlog 和 two-hot，回报的尺度问题就全解决了。**
  实际情况：这只解决了**预测**回报时的尺度问题。actor 的损失里还直接用到回报的数值，那里的尺度问题要靠下一节的回报归一化。

---

## 7. Dreamer V3（二）：回报归一化与其他稳健性技巧

### 7.1 直觉

万用表有一个"自动量程"功能：测 200 伏时自动切到大量程，把读数缩到表盘里。
但一块好的万用表在表笔悬空、只有几毫伏噪声的时候，不会把量程切到最灵敏那档、让指针满表盘乱晃。
**大信号要缩小，小信号不要放大。** 放大一个本来接近 0 的信号，放大的只是噪声。

V3 对 actor 损失里的回报做的就是这种"只缩不放"的自动量程。
量程怎么估计也有讲究：看监控时我们看 P95 而不是最大值，因为最大值会被一次偶发的毛刺带跑。
V3 用回报的 5% 和 95% 分位数之差当量程，道理相同。

本节还讲另外两个小技巧（unimix、critic 的 EMA 正则）。三者互相独立，放在一起是因为它们都是"防止某个量跑飞"的保险丝。

### 7.2 问题：actor 损失里两项的比例

actor 的损失有两项（第 4.5 节）：一项让回报高的动作概率变大，一项鼓励熵（保持探索）。简写成：

$$
\mathcal{L}_{\text{actor}} = -\underbrace{A \cdot \ln \pi_\phi(a \mid m)}_{\text{回报项，} A \text{ 是优势}} - \underbrace{\eta \, \mathcal{H}[\pi_\phi]}_{\text{熵项}}
$$

熵的数值范围是固定的（$K$ 个离散动作时在 0 到 $\ln K$ 之间），而优势 $A$ 的大小取决于任务的奖励尺度。
V3 想用一个固定的 $\eta$（论文超参数表里是 $3 \times 10^{-4}$）应付所有任务，于是两项的比例完全由 $A$ 的尺度决定：

| 任务 | 优势的典型大小 | 用固定的 $\eta$ 会怎样 |
| --- | --- | --- |
| 奖励稠密、数值大 | 几十到几百 | 回报项压倒熵项，策略很快变成确定性的，停止探索，卡在局部最优 |
| 奖励稀疏、数值小 | 0.01 量级 | 熵项压倒回报项，策略一直接近均匀随机，学不到东西 |

常见的补救是把优势除以它的标准差（PPO 的实现里常这样做）。这在稀疏奖励下会出事：
绝大多数想象轨迹的回报都接近 0，标准差很小，除以它之后，critic 的拟合误差这类噪声被放大成了"显著的优势"。

### 7.3 定义与推导：分位数回报归一化

**定义。** V3 的做法（论文的 Actor Critic Learning 一节）：

$$
S = \mathrm{EMA}\Big( \mathrm{Per}(R^\lambda, 95) - \mathrm{Per}(R^\lambda, 5), \; 0.99 \Big), \qquad
A_k = \mathrm{sg}\left( \frac{R^\lambda_k - V_\psi(\hat m_k)}{\max(1, S)} \right)
$$

其中 $\mathrm{Per}(R^\lambda, 95)$ 是当前这批想象轨迹上所有 λ-return 的 95% 分位数，EMA 的衰减率是 0.99。
这个定义里有四个设计选择，逐个看。

**选择一：用"范围"而不是标准差，用分位数而不是最大最小值。**
范围直接回答"回报高低之间差多少"。分位数是为了抗离群值。举个例子，一批 100 个回报里 99 个在 0 到 10 之间，1 个是 1000：

| 尺度的估计方式 | 结果 | 评价 |
| --- | --- | --- |
| 最大值减最小值 | 约 1000 | 被一个离群值决定。其余 99 个回报之间的差异被压成了 0.01 量级 |
| 95% 分位数减 5% 分位数 | 约 9 | 反映了大多数回报的分布范围 |

**选择二：$\max(1, S)$，只缩小不放大。** 分两种情况：

- $S > 1$（回报的范围大）：除以 $S$，把优势缩到 1 的量级。
- $S \le 1$（回报的范围小）：除以 1，原样不动。

第二种情况就是稀疏奖励。V3 论文的说明是：为了避免在稀疏奖励下放大函数近似带来的噪声，
只缩小大的回报，对低于阈值 1 的小回报不做处理。

**选择三：EMA 平滑。** 单批数据估计的分位数有噪声，而且随策略改进而漂移。更新式是：

$$
S \leftarrow 0.99 \, S + 0.01 \, \big( \mathrm{Per}(R^\lambda, 95) - \mathrm{Per}(R^\lambda, 5) \big)
$$

衰减率 0.99 对应的平均窗口大约是 $1 / (1 - 0.99) = 100$ 次更新。

**选择四：整个归一化后的优势停止梯度。** $S$ 是统计量，不是要优化的东西；$R^\lambda$ 和 $V_\psi$ 在 REINFORCE 里只是权重（第 9 节）。

**这个归一化等价于什么。** 把损失里的 $\frac{1}{\max(1,S)}$ 提出来。设 $S > 1$，记未归一化的优势为 $\tilde A = R^\lambda - V_\psi$：

$$
-\frac{\tilde A}{S} \ln \pi_\phi - \eta \, \mathcal{H}[\pi_\phi] = \frac{1}{S} \Big( -\tilde A \, \ln \pi_\phi - \eta S \, \mathcal{H}[\pi_\phi] \Big)
$$

括号外的 $\frac{1}{S}$ 只是整体缩放（相当于改了学习率）。括号里面是一个没有归一化的 actor 损失，熵系数是 $\eta S$。
所以**回报归一化等价于让熵系数自动正比于回报的范围**：回报的范围大 100 倍，熵系数跟着大 100 倍，两项的比例保持不变。
这就是固定的 $\eta$ 能跨任务使用的原因。SAC 用另一种办法达到类似的目的：直接把熵系数当成可学习的参数自动调（Phase 5）。

**手算例子。**

奖励稠密的任务：这批想象回报的 5% 分位数是 2，95% 分位数是 402，范围是 400。假设 $S$ 已经收敛到 400。
某一步 $R^\lambda = 520$，$V_\psi = 440$：

$$
A = \frac{520 - 440}{\max(1, 400)} = \frac{80}{400} = 0.2
$$

奖励稀疏的任务：5% 分位数是 0，95% 分位数是 0.02，$S = 0.02$。某一步 $R^\lambda = 0.015$，$V_\psi = 0.005$：

$$
A = \frac{0.015 - 0.005}{\max(1, 0.02)} = \frac{0.01}{1} = 0.01
$$

如果没有 $\max(1, \cdot)$ 的保护，直接除以 $S$，会得到 $0.01 / 0.02 = 0.5$，放大了 50 倍。
0.01 的优势很可能只是 critic 的拟合误差，放大之后 actor 会认真地去追它。

EMA 的一步：$S$ 原来是 100，这一批的范围是 400：

$$
S \leftarrow 0.99 \times 100 + 0.01 \times 400 = 103
$$

变化很慢。回报的尺度突然变大时（比如智能体第一次发现了一个大奖励），$S$ 要过上百次更新才跟上，这期间优势会偏大。

### 7.4 unimix：给类别分布留 1% 的均匀成分

**定义。** 把网络输出的类别分布和均匀分布混合（V3 论文的 World Model Learning 一节，混合比例 1%）：

$$
p'_k = (1 - \varepsilon) \, \mathrm{softmax}(\ell)_k + \frac{\varepsilon}{K}, \qquad \varepsilon = 0.01
$$

V3 对编码器（后验）和动力学预测器（先验）的类别分布都这样处理；离散动作的 actor 也用了同样的混合（待核对）。

写过朴素贝叶斯或 n-gram 语言模型的人会认出来，这就是加性平滑：给每个类别垫一个最小概率，避免出现概率为 0 的事件。

**作用一：KL 有上界。** 接第 5.7 节。混合之后每一类的概率至少是 $\varepsilon / K$。KL 是它两个自变量的凸函数，
凸函数在概率单纯形这样的凸多面体上的最大值在顶点处取到，也就是两个分布都退化成 one-hot、且落在不同类别上的时候。
这时混合后的后验在类别 $a$ 上是 $1 - \varepsilon + \frac{\varepsilon}{K}$，其他类上是 $\frac{\varepsilon}{K}$；先验同理，峰在另一个类别 $b$ 上。
只有 $a$、$b$ 两类对 KL 有贡献（其余类上两边概率相等，$\ln 1 = 0$）：

$$
\begin{aligned}
D_{\mathrm{KL}}^{\max} &= \Big( 1 - \varepsilon + \frac{\varepsilon}{K} \Big) \ln \frac{1 - \varepsilon + \varepsilon / K}{\varepsilon / K} + \frac{\varepsilon}{K} \ln \frac{\varepsilon / K}{1 - \varepsilon + \varepsilon / K} \\
&= (1 - \varepsilon) \, \ln \frac{1 - \varepsilon + \varepsilon / K}{\varepsilon / K} \\
&= (1 - \varepsilon) \, \ln \Big( 1 + \frac{(1 - \varepsilon) K}{\varepsilon} \Big)
\end{aligned}
$$

第二行把两项合并（两个对数互为相反数）。代入 $\varepsilon = 0.01$：

$$
D_{\mathrm{KL}}^{\max} = 0.99 \, \ln(1 + 99 K)
$$

| $K$ | $1 + 99K$ | 单个类别变量的 KL 上界 |
| --- | --- | --- |
| 4 | 397 | $0.99 \times 5.984 = 5.92$ nat（和第 5.7 节一致） |
| 32 | 3169 | $0.99 \times 8.061 = 7.98$ nat |

32 个类别变量加起来，KL 的上界是 $32 \times 7.98 \approx 255$ nat。不混合时没有上界。

**作用二：对数概率有下界。** $\ln p'_k \ge \ln(0.01 / 32) = \ln(3.125 \times 10^{-4}) = -8.07$。
任何用到 $\ln p$ 的地方（KL、REINFORCE 里的 $\ln \pi$）都不会出现负无穷。

**作用三（对 actor）：保底的探索。** 每个离散动作至少有 $0.01 / K$ 的概率被选到，策略不会完全变成确定性的。
这和 ε-greedy 的想法一样，只是混合发生在分布层面。

### 7.5 critic 的 EMA 正则

**问题。** critic 的目标 $R^\lambda$ 是用 critic 自己的输出 $v_k$ 算出来的。critic 一更新，目标跟着变，可能越追越偏。
Phase 4 的 DQN 用目标网络解决这个问题：另存一份慢更新的参数，专门用来算目标。Dreamer V2 沿用了这个办法。

目标网络的代价是**信息滞后**：目标是用旧的 critic 算的，critic 新学到的东西要等下一次同步才能进入目标。

**V3 的做法**（论文的 Actor Critic Learning 一节）：不用目标网络算回报，而是

1. λ-return 用**当前的** critic 算，目标里的信息是最新的；
2. 另外维护一份 critic 参数的指数滑动平均 $\bar\psi$，在 critic 的损失里加一项正则，把 critic 的输出往 EMA critic 的输出上拉。

稳定性不再来自"目标算得慢"，而来自"critic 自己不许变得太快"。类比线上的配置变更：
目标网络是"每 100 步才允许发布一次"，EMA 正则是"随时可以发布，但每次的改动幅度有限速"。

**EMA 的定义与推导。** 每次更新 critic 之后：

$$
\bar\psi \leftarrow d \, \bar\psi + (1 - d) \, \psi
$$

把递推展开看它到底在平均什么。记第 $n$ 次更新后的在线参数为 $\psi_n$，EMA 为 $\bar\psi_n$：

$$
\begin{aligned}
\bar\psi_n &= (1 - d) \, \psi_n + d \, \bar\psi_{n-1} \\
&= (1 - d) \, \psi_n + d \, (1 - d) \, \psi_{n-1} + d^2 \, \bar\psi_{n-2} \\
&= (1 - d) \sum_{j=0}^{n-1} d^j \, \psi_{n-j} + d^n \, \bar\psi_0
\end{aligned}
$$

$j$ 步之前的参数的权重是 $(1 - d) d^j$，按几何级数衰减。权重之和 $(1-d) \sum_{j \ge 0} d^j = 1$。
平均窗口的长度大约是 $1 / (1 - d)$。

**手算例子。** 取 $d = 0.98$（V3 官方实现的默认值，待核对）：

- 最新参数的权重是 $1 - d = 0.02$。
- 平均窗口约 $1 / 0.02 = 50$ 次更新。
- 50 次更新之前的所有参数加起来还占 $0.98^{50} = e^{50 \ln 0.98} = e^{-1.010} = 0.364$ 的权重。

**正则项的形式。** critic 是 two-hot 分类头，所以"往 EMA 的输出上拉"也写成一个交叉熵：
把 EMA critic 的预测当作另一个回归目标（停止梯度）。合起来：

$$
\mathcal{L}_{\text{critic}}(\psi) = \underbrace{-\ln p_\psi\big( \mathrm{sg}(R^\lambda_k) \mid \hat m_k \big)}_{\text{拟合 λ-return}} \; \underbrace{- \ln p_\psi\big( \mathrm{sg}(V_{\bar\psi}(\hat m_k)) \mid \hat m_k \big)}_{\text{EMA 正则}}
$$

其中 $-\ln p_\psi(y \mid m)$ 是第 6.7 节的 two-hot 交叉熵的简写。正则项的具体形式和系数以官方实现为准（待核对）。

| | 目标网络（DQN、Dreamer V2） | EMA 正则（Dreamer V3） |
| --- | --- | --- |
| 回报用谁算 | 慢更新的副本 | 当前的 critic |
| 稳定性从哪来 | 目标变得慢 | critic 被拉向自己的滑动平均，变得慢 |
| 目标里的信息 | 滞后 | 最新 |
| 额外的开销 | 存一份参数 | 存一份参数，多一次前向计算 |

### 7.6 V3 的稳健性技巧一览

把第 5–7 节讲的东西汇总起来。每一行都是"某个量在某类任务上会跑飞，于是加一道保险"：

| 技巧 | 作用在哪 | 防的是什么 | 本章位置 |
| --- | --- | --- | --- |
| KL balancing（$\beta_{\text{dyn}}$、$\beta_{\text{rep}}$） | 世界模型 | 表征被拉向没学好的先验 | 第 5.5 节 |
| free bits（1 nat） | 世界模型 | KL 把表征里的信息压光 | 第 5.6 节 |
| unimix（1%） | 先验、后验、离散 actor | KL 尖峰；对数概率为负无穷；策略完全确定 | 第 5.7、7.4 节 |
| symlog | 编码器输入、向量观测的解码 | 观测的尺度随任务变 | 第 6.2–6.4 节 |
| symlog + two-hot | 奖励头、critic | 回归梯度随奖励和回报的尺度变 | 第 6.5–6.8 节 |
| 输出层零初始化 | 奖励头、critic | 训练初期幻觉出大的奖励和价值 | 第 6.9 节 |
| 分位数回报归一化 | actor | 回报项和熵项的比例随任务变；稀疏奖励下放大噪声 | 第 7.3 节 |
| critic 的 EMA 正则 | critic | 自举目标追着自己跑 | 第 7.5 节 |

V3 论文的消融实验的结论是：这些技巧每一个都只在一部分任务上起决定作用，但要让同一套超参数通吃所有任务，它们全都需要（待核对原文表述）。
这也是读 V3 时最容易有的困惑的答案："为什么要这么多小技巧？"因为目标不是在某个基准上拿高分，而是不调参。

### 7.7 算法

```
算法：分位数回报归一化

状态：S（初始为 0 或 1）
函数 update_and_scale(returns)：                           returns 形状 [H, N]，已停止梯度
     lo ← quantile(returns, 0.05)；hi ← quantile(returns, 0.95)       在所有元素上算，标量
     S ← 0.99 · S + 0.01 · (hi − lo)
     返回 max(1, S)

算法：unimix
函数 unimix_probs(logits)：                                 logits 形状 [..., K]
     返回 0.99 · softmax(logits) + 0.01 / K

算法：EMA 更新
函数 ema_update(online, ema, d)：
     对每一对参数 (ψ, ψ̄)：ψ̄ ← d · ψ̄ + (1 − d) · ψ         不记录梯度
```

### 7.8 代码对应

对应 `07-return-normalization-and-robustness.py`。

| 名称 | 做什么 | 关键张量 |
| --- | --- | --- |
| `ReturnNormalizer` | 维护 $S$，提供 `update_and_scale()` | 输入 `[H, N]`，输出标量 |
| `unimix_probs(logits, eps)` | 第 7.4 节的混合 | 输入输出均为 `[..., K]` |
| `ema_update(online, ema, decay)` | 第 7.5 节的参数滑动平均 | 逐参数 |

讲义用三个小实验分别验证：把同一个 bandit 问题的奖励乘上 0.01、1、100，看固定熵系数的 actor 在有无回报归一化时的表现；
构造 logits 饱和的先验和后验，看有无 unimix 时 KL 的最大值；在一个自举的价值回归玩具问题上比较无稳定措施、目标网络、EMA 正则三种做法。

> 📌 待回填：三种奖励尺度下有无回报归一化的策略熵曲线与最终回报；有无 unimix 时 KL 的最大值（预期：有 unimix 时单个变量不超过 7.98 nat）；三种 critic 稳定办法的价值误差曲线，来自 `07-return-normalization-and-robustness.py`。

### 7.9 常见误区

- **误区：回报归一化是为了让 critic 更好学。**
  实际情况：critic 的尺度问题已经由 symlog 和 two-hot 解决。回报归一化只作用在 actor 损失的优势上，
  目的是让回报项和熵项的比例不随任务变。critic 学的仍然是没有归一化的回报。
- **误区：除以 $S$ 就是标准化（减均值、除以标准差）。**
  实际情况：这里不减均值（基线由 critic 提供），除的是分位数范围而不是标准差，而且有 $\max(1, \cdot)$ 的下限，只缩小不放大。
- **误区：unimix 让隐变量变得很随机，会损害表征。**
  实际情况：1% 的概率摊到 32 个类别上，每个类别只多了 0.03%。采样结果几乎不变，变的只是概率不会精确为 0。
- **误区：EMA 正则就是换了个名字的目标网络。**
  实际情况：目标网络用旧参数算目标；EMA 正则用新参数算目标，再限制 critic 偏离自己滑动平均的速度。稳定性的来源不同。

---

## 8. Dreamer V3 世界模型的完整损失

### 8.1 直觉

前面四节各讲了一个零件。这一节把它们装成一个世界模型，写出它的总损失。

可以把 RSSM 类比成**数据压缩里的预测编码**：

- $h_t$ 是到目前为止的上下文。
- 先验 $p_\theta(z_t \mid h_t)$ 是"根据上下文猜下一帧"。
- 后验 $q_\theta(z_t \mid h_t, o_t)$ 是"看了下一帧之后的实际编码"。
- KL 是两者的差距，也就是这一帧里**没被猜到的信息量**。视频编码里它对应残差的码率。

一个好的世界模型要同时做到三件事，正好对应 V3 的三项损失：

| 要求 | 对应的损失 | 不满足会怎样 |
| --- | --- | --- |
| 编码里有足够的信息：从 $(h_t, z_t)$ 能还原画面、奖励、是否结束 | 预测损失 $\mathcal{L}_{\text{pred}}$ | 表征是空的，actor 和 critic 无从学起 |
| 猜得准：先验接近后验 | 动力学损失 $\mathcal{L}_{\text{dyn}}$ | 想象时用的是先验，先验不准，想象出的轨迹就是胡话 |
| 编码要好猜：不往表征里塞预测不了的噪声 | 表征损失 $\mathcal{L}_{\text{rep}}$ | 先验永远追不上后验，想象和现实之间有一道跨不过去的沟 |

### 8.2 定义：六个部件

V3 的世界模型由六个部件组成（论文的 World Model Learning 一节，换成本课记号）：

$$
\begin{aligned}
\text{序列模型：} \quad & h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1}) \\
\text{编码器（后验）：} \quad & z_t \sim q_\theta(z_t \mid h_t, o_t) \\
\text{动力学预测器（先验）：} \quad & \hat z_t \sim p_\theta(\hat z_t \mid h_t) \\
\text{解码器：} \quad & \hat o_t \sim p_\theta(\hat o_t \mid h_t, z_t) \\
\text{奖励预测器：} \quad & \hat r_{t-1} \sim p_\theta(\hat r_{t-1} \mid h_t, z_t) \\
\text{continue 预测器：} \quad & \hat c_t \sim p_\theta(\hat c_t \mid h_t, z_t)
\end{aligned}
$$

一个时间步的数据流：

```
   h_{t-1}, z_{t-1}, a_{t-1}
            │
            ▼
     ┌─────────────┐              ┌───────────┐
     │ 序列模型 GRU  │──▶ h_t ──┬──▶│ 先验头 MLP │──▶ 先验 logits [32, 32] ────┐
     └─────────────┘          │   └───────────┘                            │
                              │                                        两项 KL
                              │   ┌───────────┐                            │
   o_t ─▶ 编码器 CNN ─▶ e_t ───┴──▶│ 后验头 MLP │──▶ 后验 logits [32, 32] ────┘
                                  └───────────┘          │
                                                         │ unimix + straight-through 采样
                                                         ▼
                                                        z_t   [1024]
                                                         │
                              m_t = (h_t, z_t) ──────────┤
                                                         ├──▶ 解码器        ─▶ ô_t            重建
                                                         ├──▶ 奖励头        ─▶ 255 个桶的 logits
                                                         └──▶ continue 头   ─▶ 1 个 logit
```

每个头的输出分布和损失：

| 头 | 输出分布 | 损失 $-\ln p_\theta(\cdot \mid h_t, z_t)$ 的具体形式 | 出处 |
| --- | --- | --- | --- |
| 解码器（图像） | 方差为 1 的高斯 | $\frac{1}{2} \Vert \hat o_t - o_t \Vert^2$，对所有像素求和 | Phase 1 第 8.3 节：方差固定时高斯 NLL 就是平方误差 |
| 解码器（向量） | symlog 空间里方差为 1 的高斯 | $\frac{1}{2} \Vert f - \mathrm{symlog}(o_t) \Vert^2$ | 第 6.3 节 |
| 奖励头 | 255 个桶上的类别分布 | two-hot 交叉熵 | 第 6.7 节 |
| continue 头 | 伯努利分布 | 二元交叉熵 $-c_t \ln \hat c_t - (1 - c_t) \ln (1 - \hat c_t)$ | 伯努利分布的对数似然 |

### 8.3 推导：从序列 ELBO 到 V3 的三项损失

**第一步：从 Phase 3 的序列 ELBO 出发。** 那里推出的结论是：观测序列的对数似然有下界

$$
\ln p_\theta(o_{1:T} \mid a_{1:T}) \;\ge\; \sum_{t=1}^{T} \mathbb{E}_{q_\theta}\Big[ \ln p_\theta(o_t \mid h_t, z_t) - D_{\mathrm{KL}}\big( q_\theta(z_t \mid h_t, o_t) \,\Vert\, p_\theta(z_t \mid h_t) \big) \Big]
$$

第一项是重建，第二项是后验和先验之间的 KL。

**第二步：把奖励和 continue 标志也当成要解释的观测。** 假设给定 $(h_t, z_t)$ 之后，画面、奖励、continue 三者条件独立，联合似然是三者的乘积，取对数是三者之和：

$$
\ln p_\theta(o_t, r_{t-1}, c_t \mid h_t, z_t) = \ln p_\theta(o_t \mid h_t, z_t) + \ln p_\theta(r_{t-1} \mid h_t, z_t) + \ln p_\theta(c_t \mid h_t, z_t)
$$

**第三步：取负号，把"最大化下界"变成"最小化损失"。** 每个时间步的损失是：

$$
\underbrace{-\ln p_\theta(o_t \mid h_t, z_t) - \ln p_\theta(r_{t-1} \mid h_t, z_t) - \ln p_\theta(c_t \mid h_t, z_t)}_{\mathcal{L}_{\text{pred}}} + D_{\mathrm{KL}}\big( q_\theta \,\Vert\, p_\theta \big)
$$

到这里它还是一个严格的负 ELBO。Dreamer V1 的世界模型损失就是这个式子（当时没有 continue 项）。

**第四步：把 KL 拆成两半。** 用第 5.5 节的恒等式。这一步不改变梯度：

$$
\nabla_\theta D_{\mathrm{KL}}(q_\theta \,\Vert\, p_\theta) = \nabla_\theta D_{\mathrm{KL}}\big( \mathrm{sg}(q_\theta) \,\Vert\, p_\theta \big) + \nabla_\theta D_{\mathrm{KL}}\big( q_\theta \,\Vert\, \mathrm{sg}(p_\theta) \big)
$$

**第五步：两半各给一个权重，各加一个 free bits 截断。** 这一步改变了梯度，是工程上的选择：

$$
\mathcal{L}_{\text{dyn}} = \max\Big( 1, \; D_{\mathrm{KL}}\big( \mathrm{sg}(q_\theta(z_t \mid h_t, o_t)) \,\Vert\, p_\theta(z_t \mid h_t) \big) \Big)
$$

$$
\mathcal{L}_{\text{rep}} = \max\Big( 1, \; D_{\mathrm{KL}}\big( q_\theta(z_t \mid h_t, o_t) \,\Vert\, \mathrm{sg}(p_\theta(z_t \mid h_t)) \big) \Big)
$$

**最终的损失**（V3 论文的世界模型损失一式）：

$$
\mathcal{L}(\theta) = \mathbb{E}_{q_\theta}\Big[ \sum_{t=1}^{T} \Big( \beta_{\text{pred}} \, \mathcal{L}_{\text{pred}}(t) + \beta_{\text{dyn}} \, \mathcal{L}_{\text{dyn}}(t) + \beta_{\text{rep}} \, \mathcal{L}_{\text{rep}}(t) \Big) \Big]
$$

权重：$\beta_{\text{pred}} = 1$，$\beta_{\text{rep}} = 0.1$；$\beta_{\text{dyn}}$ 在 2023 年的 arXiv 初版是 0.5，之后的版本是 1（待核对）。

经过第五步，这个损失不再是对数似然的严格下界。它是"从 ELBO 出发、按工程需要调过的目标"。
$\beta_{\text{dyn}} = \beta_{\text{rep}} = 1$ 且不截断时，它退回第三步的负 ELBO。

### 8.4 逐个标出 stop-gradient

世界模型的损失里，**显式的 sg 只有两个**，都在 KL 项里。

| 位置 | 包住了什么 | 为什么放在这里 | 去掉会怎样 |
| --- | --- | --- | --- |
| $\mathcal{L}_{\text{dyn}}$ 里的 $\mathrm{sg}(q_\theta)$ | 第 $t$ 步后验分布的参数（后验 logits） | 这一项的任务是训练先验去预测表征。后验是"标签"，标签不该为了让学生好做题而改动 | 后验会被这一项往先验上拉，而且权重是较大的 $\beta_{\text{dyn}}$：等于没做 KL balancing，训练初期表征被拉向没学好的先验 |
| $\mathcal{L}_{\text{rep}}$ 里的 $\mathrm{sg}(p_\theta)$ | 第 $t$ 步先验分布的参数（先验 logits） | 这一项的任务是让表征变得好预测。先验在这里只是一个参照物 | 先验会多收到一份权重为 $\beta_{\text{rep}}$ 的梯度。影响不大，但两项的分工就不干净了 |

还有三处"看起来该有 sg、其实不需要"或者"藏起来的 sg"，一并说清：

| 位置 | 情况 |
| --- | --- |
| straight-through 采样内部 | $z = e + p - \mathrm{sg}(p)$ 里有一个 sg（第 4.3 节）。它是采样的实现细节，不在损失的式子里出现 |
| 预测损失的目标 | $o_t$、$r_{t-1}$、$c_t$ 来自回放，是常数，本来就没有梯度。对它们做的 symlog 和 two-hot 编码也是常数。论文在 symlog 损失里写的 $\mathrm{sg}(\mathrm{symlog}(y))$ 只是强调这一点 |
| 预测损失对表征的梯度 | **没有 sg。** 重建、奖励、continue 三项的梯度都流回 $z_t$ 和 $h_t$，进而流到编码器和序列模型。表征正是靠这些梯度学出来的 |

第三条是有意的设计。奖励的梯度流进表征，表征里才会保留对任务重要、但在画面上占比很小的东西
（乒乓球在画面上只有几个像素，重建损失几乎不在乎它，奖励却完全取决于它）。
V2 论文做过对应的消融：去掉图像重建的梯度，性能大幅下降；去掉奖励的梯度，影响相对小（待核对）。
也就是说在 Dreamer 里，表征主要是靠重建学出来的。Phase 7 走的是相反的路：不要重建，只靠奖励和价值来塑造表征。

### 8.5 梯度流向总表

每一项损失更新了哪些部件：

| 损失项 | 解码器、奖励头、continue 头 | 后验头与编码器（第 $t$ 步） | 先验头 | 序列模型（GRU） | 更早时间步的编码器 |
| --- | --- | --- | --- | --- | --- |
| $\mathcal{L}_{\text{pred}}$ | 更新 | 更新（经过 $z_t$ 的 straight-through） | 不更新 | 更新（经过 $h_t$） | 更新（经过 $h_t \leftarrow z_{t-1}$） |
| $\mathcal{L}_{\text{dyn}}$ | 不更新 | **不更新（被 sg 挡住）** | 更新 | 更新（先验的输入是 $h_t$） | 更新（经过 $h_t \leftarrow z_{t-1}$） |
| $\mathcal{L}_{\text{rep}}$ | 不更新 | 更新 | **不更新（被 sg 挡住）** | 更新（后验的输入也有 $h_t$） | 更新（经过 $h_t \leftarrow z_{t-1}$） |

最后一列说明：sg 只挡住"第 $t$ 步这个分布的参数"这一条路。$h_t$ 是由上一步的 $z_{t-1}$ 算出来的，
而 $z_{t-1}$ 是上一步后验的 straight-through 样本，所以梯度可以沿着时间往回流（就是 BPTT）。
$\mathcal{L}_{\text{dyn}}$ 因此会训练更早时间步的编码器："请把 $z_{t-1}$ 编码成让下一步更好预测的样子"。

### 8.6 手算例子

取一条序列里相邻的两个时间步，把每一项的数值摆出来（数值是为了演示加法而假设的，不是实验结果）。
用 $\beta_{\text{pred}} = 1$、$\beta_{\text{dyn}} = 1$、$\beta_{\text{rep}} = 0.1$。

| 项 | 第 1 步 | 第 2 步 | 说明 |
| --- | --- | --- | --- |
| 重建 $-\ln p(o)$ | 12.0 | 10.0 | 对所有像素求和 |
| 奖励 $-\ln p(r)$ | 0.9506 | 0.70 | two-hot 交叉熵。第 1 步取第 6.7 节算出的值 |
| continue $-\ln p(c)$ | $-\ln 0.9 = 0.1054$ | $-\ln 0.8 = 0.2231$ | 真实标志都是 1，预测的概率分别是 0.9 和 0.8 |
| $\mathcal{L}_{\text{pred}}$ | 13.056 | 10.9231 | 以上三项之和 |
| 未截断的 KL | 0.6 | 2.5 | 32 个类别变量求和之后 |
| $\mathcal{L}_{\text{dyn}} = \max(1, \text{KL})$ | 1 | 2.5 | 第 1 步被截断，没有梯度 |
| $\mathcal{L}_{\text{rep}} = \max(1, \text{KL})$ | 1 | 2.5 | 数值和 $\mathcal{L}_{\text{dyn}}$ 相同，梯度不同 |
| 加权合计 | $13.056 + 1 + 0.1 = 14.156$ | $10.9231 + 2.5 + 0.25 = 13.6731$ | |

两步平均：$(14.156 + 13.6731) / 2 = 13.91$。

从这张表能读出三件事：

1. 第 1 步的 KL 只有 0.6 nat，低于 free bits，这一步的 KL 损失是常数，世界模型在这一步只优化预测。
2. 第 2 步的 KL 有梯度：先验头收到权重为 1 的"去追后验"的梯度，后验头收到权重为 0.1 的"去迁就先验"的梯度。
3. 重建项在数值上占了绝大部分。这是常态（它是成千上万个像素的误差之和），不代表其他项不重要。
   **比较各项的重要性要看梯度，不看损失的数值。**

### 8.7 算法

```
算法：世界模型的一次损失计算 WorldModel.loss

输入：一批序列 (o, a, r, c, is_first)，形状 [B, T, ...]
 1. 编码：e ← 编码器(o)                                                   形状 [B, T, E]
 2. 初始化 h、z、上一步动作为零
 3. 对 t = 0 … T−1（observe）：
        如果 is_first[t]：把这条序列的 h、z、上一步动作清零                 回合边界处状态要重置
        h_t ← f_θ(h_{t−1}, z_{t−1}, a_{t−1})                              形状 [B, D_h]
        先验 logits_t ← 先验头(h_t)                                        形状 [B, 32, 32]
        后验 logits_t ← 后验头(h_t, e_t)                                   形状 [B, 32, 32]
        z_t ← straight-through 采样(unimix(后验 logits_t))                  形状 [B, 1024]
 4. m ← concat(h, z)                                                       形状 [B, T, D_h + 1024]
 5. L_pred ← 重建损失(解码器(m), o) + two-hot 交叉熵(奖励头(m), r) + 二元交叉熵(continue 头(m), c)
 6. L_dyn ← max(1, KL(sg(后验) ‖ 先验))；L_rep ← max(1, KL(后验 ‖ sg(先验)))     各为 [B, T]
 7. 损失 ← mean(β_pred · L_pred + β_dyn · L_dyn + β_rep · L_rep)
输出：损失；后验状态 (h, z) 交给想象环节当起点
```

第 5 步里 `r` 和 `m` 的对齐遵循第 2.2 节的图：`m[t]` 上的奖励头预测的是到达第 $t$ 步时拿到的奖励。
回放里怎么存，这里就要怎么取，错开一位是常见的 bug。

### 8.8 代码对应

对应 `08-dreamer-v3-world-model.py`。

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `obs` | `[B, T, 3, 64, 64]` | 像素缩放到 $[0, 1]$ |
| `embed` | `[B, T, E]` | 编码器（CNN）的输出 |
| `h` | `[B, T, D_h]` | 确定性状态 |
| `post_logits`、`prior_logits` | `[B, T, 32, 32]` | 后验和先验的 logits |
| `z` | `[B, T, 1024]` | 展平的 one-hot 样本，带 straight-through 梯度 |
| `feat` | `[B, T, D_h + 1024]` | 模型状态 $m_t$，各个头的输入 |
| `recon` | `[B, T, 3, 64, 64]` | 解码器的输出 |
| `reward_logits` | `[B, T, 255]` | 奖励头的输出 |
| `cont_logits` | `[B, T]` | continue 头的输出，过 sigmoid 得到概率 |

`WorldModel.observe()` 返回后验状态和两组 logits，`WorldModel.loss()` 在它之上算第 8.7 节的损失，
`WorldModel.imagine()` 就是第 2.6 节的 `imagine_rollout()`。
网络的宽度和层数按本机能训练的规模取，和论文的差别写在讲义里。
论文还用了层归一化和 SiLU 激活等结构上的细节（待核对各版本），讲义里从简。

> 📌 待回填：世界模型训练过程中各项损失的曲线（重建、奖励、continue、未截断的 KL）；开环预测视频（给若干帧上下文，之后只用先验想象）与真实画面的对比；奖励头在非零奖励时间步上的预测误差，来自 `08-dreamer-v3-world-model.py`。

### 8.9 常见误区

- **误区：$\mathcal{L}_{\text{dyn}}$ 和 $\mathcal{L}_{\text{rep}}$ 数值相同，日志里记一个就行，实现里也可以只算一次。**
  实际情况：数值相同，梯度不同。必须算两次，一次 detach 后验，一次 detach 先验。日志里记未截断的 KL 一个值就够。
- **误区：$\beta_{\text{rep}} = 0.1$ 很小，说明表征损失可有可无。**
  实际情况：去掉它，表征可以无限制地往里塞先验预测不了的细节，先验永远追不上，想象和现实脱节。权重小是为了不压垮表征，不是不需要。
- **误区：解码器只是用来可视化的，训练时可以不要。**
  实际情况：在 Dreamer 里，重建是塑造表征的主要信号。去掉重建又不加别的约束，表征会坍塌。
  不要解码器是可以的，但那是另一类方法，需要别的信号来撑住表征（Phase 7、Phase 10）。
- **误区：世界模型的损失比上次低，就说明模型更好了。**
  实际情况：损失由重建项主导，重建项又由画面里面积大的部分主导。总损失降了，奖励预测可能反而变差。要分项看。

---

## 9. Dreamer V3 的 actor 与 critic

### 9.1 直觉

世界模型就绪之后，行为学习是一个标准的 actor-critic，只是数据全部来自想象。
可以用代码评审来类比两者的分工：

- **critic 是评审人**：看到一个局面，估计"照现在的做法走下去，最后能拿多少"。
- **actor 是提交者**：做了一个动作之后，把实际结果（λ-return）和评审人的预期（价值）比较。
  比预期好，以后多这么做；比预期差，以后少这么做。

V3 在这个标准结构上只改了"数值怎么处理"：critic 用 two-hot 做离散回归（第 6 节），
actor 的优势除以回报的范围（第 7 节）。这样两者的损失都不随奖励的尺度变化。

### 9.2 定义：critic 的损失

critic 输出 255 个桶上的分布 $p_\psi(\cdot \mid m)$，价值是按这个分布对桶的位置取的期望（第 6 节）：

$$
V_\psi(m) = \sum_{j=1}^{M} p_\psi(b_j \mid m) \, b_j
$$

λ-return 用第 3.3 节的递推，在想象轨迹上从后往前算：

$$
R^\lambda_k = \hat r_k + \gamma \, \hat c_{k+1} \Big[ (1 - \lambda) \, v_{k+1} + \lambda \, R^\lambda_{k+1} \Big], \qquad R^\lambda_H = v_H, \qquad v_k = V_\psi(\hat m_k)
$$

critic 的损失（V3 论文的 Critic Learning 部分，加上第 2.4 节的权重和第 7.5 节的 EMA 正则）：

$$
\mathcal{L}_{\text{critic}}(\psi) = \mathbb{E}\Big[ \sum_{k=0}^{H-1} \mathrm{sg}(w_k) \Big( -\ln p_\psi\big( \mathrm{sg}(R^\lambda_k) \mid \mathrm{sg}(\hat m_k) \big) - \ln p_\psi\big( \mathrm{sg}(V_{\bar\psi}(\hat m_k)) \mid \mathrm{sg}(\hat m_k) \big) \Big) \Big]
$$

其中 $-\ln p_\psi(y \mid m)$ 是把标量 $y$ 编码成 two-hot 之后和预测分布算的交叉熵。
求和到 $H - 1$：第 $H$ 步的目标就是 $v_H$ 自己，没有信息。

这个式子里有四个 sg：

| sg | 为什么 |
| --- | --- |
| $\mathrm{sg}(R^\lambda_k)$ | 目标里含有 critic 自己的输出，也含有 actor 和世界模型。当作常数才是半梯度 TD（第 3.5 节） |
| $\mathrm{sg}(\hat m_k)$ | critic 的损失不能顺着输入状态流回 actor 和世界模型（第 2.5 节的 ③）。在 PyTorch 里对输入 detach；也可以只对 $\psi$ 求导，效果相同 |
| $\mathrm{sg}(V_{\bar\psi}(\hat m_k))$ | EMA critic 只是一个参照物，它的参数靠滑动平均更新，不靠梯度 |
| $\mathrm{sg}(w_k)$ | 权重来自 continue 头。critic 的损失不该去改 continue 的预测 |

和 V1 的平方误差损失（第 3.5 节）对比：目标的计算方式一模一样，变的只是损失的形式。
V3 的较新版本还会用回放里的真实轨迹额外训练 critic，把想象的价值和真实奖励对上（待核对），本课的讲义不实现。

### 9.3 推导：actor 的损失

**第一步：想象中的策略梯度。** 先看单步。在状态 $m$ 上，actor 想最大化动作价值的期望。
离散动作时期望是求和：

$$
J(\phi) = \mathbb{E}_{a \sim \pi_\phi(\cdot \mid m)}\big[ Q(m, a) \big] = \sum_a \pi_\phi(a \mid m) \, Q(m, a)
$$

对 $\phi$ 求导，把 $Q$ 先当作与 $\phi$ 无关（它对 $\phi$ 的依赖由后面各时间步自己的项负责，这是 Phase 5 策略梯度定理的内容）：

$$
\nabla_\phi J = \sum_a \nabla_\phi \pi_\phi(a \mid m) \, Q(m, a)
$$

**第二步：对数导数技巧。** 用恒等式 $\nabla_\phi \pi = \pi \, \nabla_\phi \ln \pi$，把求和重新写成期望：

$$
\nabla_\phi J = \sum_a \pi_\phi(a \mid m) \, \nabla_\phi \ln \pi_\phi(a \mid m) \, Q(m, a) = \mathbb{E}_{a \sim \pi_\phi}\big[ \nabla_\phi \ln \pi_\phi(a \mid m) \, Q(m, a) \big]
$$

右边可以用采样估计，而且不要求 $Q$ 对动作可导。这就是 score-function 估计器。

**第三步：减去基线不改变期望。** 对任何只依赖状态的 $b(m)$：

$$
\mathbb{E}_{a \sim \pi_\phi}\big[ \nabla_\phi \ln \pi_\phi(a \mid m) \, b(m) \big] = b(m) \sum_a \nabla_\phi \pi_\phi(a \mid m) = b(m) \, \nabla_\phi \sum_a \pi_\phi(a \mid m) = b(m) \, \nabla_\phi 1 = 0
$$

所以可以把 $Q$ 换成 $Q - V_\psi(m)$，期望不变，方差变小（Phase 5 的基线与优势）。

**第四步：用 λ-return 估计 $Q$，写成可以交给自动微分的损失。**
想象的第 $k$ 步，$R^\lambda_k$ 是"在 $\hat m_k$ 采取 $\hat a_k$ 之后的回报"的一个样本估计。梯度的估计是：

$$
\big( R^\lambda_k - V_\psi(\hat m_k) \big) \, \nabla_\phi \ln \pi_\phi(\hat a_k \mid \hat m_k)
$$

要让自动微分算出这个量，构造一个替身损失（surrogate loss），它的梯度恰好是上式的相反数：

$$
-\mathrm{sg}\big( R^\lambda_k - V_\psi(\hat m_k) \big) \, \ln \pi_\phi(\hat a_k \mid \hat m_k)
$$

**这里的 sg 是 score-function 估计器的组成部分，不是可选项。** 优势在这里只是一个权重。
不停止梯度的话，自动微分还会去对 $R^\lambda_k$ 和 $V_\psi$ 求导：梯度会经过 critic、奖励头和世界模型的转移绕回 actor，
等于混进了一份沿模型反传的梯度（穿过离散采样时还是有偏的）。得到的就不再是 score-function 估计器了。

**第五步：加上回报归一化、熵正则和权重，得到 V3 的 actor 损失**（论文的 Actor Learning 部分，换成本课记号）：

$$
\mathcal{L}_{\text{actor}}(\phi) = \mathbb{E}\Big[ \sum_{k=0}^{H-1} \mathrm{sg}(w_k) \Big( -\mathrm{sg}(A_k) \, \ln \pi_\phi\big( \hat a_k \mid \mathrm{sg}(\hat m_k) \big) - \eta \, \mathcal{H}\big[ \pi_\phi(\,\cdot \mid \mathrm{sg}(\hat m_k)) \big] \Big) \Big], \qquad
A_k = \frac{R^\lambda_k - V_\psi(\hat m_k)}{\max(1, S)}
$$

这个式子里的 sg：

| sg | 为什么 |
| --- | --- |
| $\mathrm{sg}(A_k)$ | 见第四步。连同里面的 $R^\lambda_k$、$V_\psi$、$S$ 一起当常数 |
| $\mathrm{sg}(\hat m_k)$ | REINFORCE 只需要"在这个状态上，这个动作的对数概率"对 $\phi$ 的梯度，不需要状态本身对 $\phi$ 的梯度。挡住之后，梯度不会穿过世界模型 |
| $\mathrm{sg}(w_k)$ | 权重来自 continue 头，是常数 |

**关于版本的说明。** V3 论文 2023 年的 arXiv 初版里，连续动作的 actor 仍然沿模型反传
（把上式的 $-\mathrm{sg}(A_k) \ln \pi_\phi$ 换成 $-R^\lambda_k / \max(1, S)$，不停止梯度，和第 3 节相同）；
离散动作用上面的 REINFORCE。之后的版本（含 Nature 版）对两类动作都用 REINFORCE（待核对）。本课的讲义两种都实现，在连续控制任务上对比。

### 9.4 手算例子

3 个离散动作，actor 当前的分布是 $\pi = (0.5, 0.3, 0.2)$。想象中采到了第 2 个动作。
这一步 $R^\lambda = 12$，$V_\psi = 10$，回报的范围 $S = 4$，熵系数 $\eta = 3 \times 10^{-4}$。

**优势。**

$$
A = \frac{12 - 10}{\max(1, 4)} = 0.5
$$

**回报项对 logits 的梯度。** $-A \ln \pi(a)$ 对 logits 的梯度是 $A \, (\pi - e_a)$（交叉熵的梯度"预测减标签"乘上 $A$，推导同第 5.4 节）：

$$
0.5 \times \big( (0.5, 0.3, 0.2) - (0, 1, 0) \big) = (0.25, \; -0.35, \; 0.10)
$$

梯度下降会抬高第 2 个动作的 logit、压低另外两个。实际结果比 critic 预期的好，所以以后多选它。
如果优势是负的，方向正好反过来。

**熵项对 logits 的梯度。** 先算熵：

$$
\mathcal{H}[\pi] = -(0.5 \ln 0.5 + 0.3 \ln 0.3 + 0.2 \ln 0.2) = 0.3466 + 0.3612 + 0.3219 = 1.0297
$$

熵对 logits 的梯度是 $\frac{\partial \mathcal{H}}{\partial \ell_j} = -\pi_j (\ln \pi_j + \mathcal{H})$（用 softmax 的雅可比推，和第 5.4 节后验一侧的推导同构）：

$$
\frac{\partial \mathcal{H}}{\partial \ell} = \big( -0.5 \times 0.3366, \;\; -0.3 \times (-0.1743), \;\; -0.2 \times (-0.5797) \big) = (-0.1683, \; 0.0523, \; 0.1159)
$$

损失里是 $-\eta \mathcal{H}$，所以损失对 logits 的梯度是：

$$
-\eta \, \frac{\partial \mathcal{H}}{\partial \ell} = 3 \times 10^{-4} \times (0.1683, \; -0.0523, \; -0.1159) = (5.0, \; -1.6, \; -3.5) \times 10^{-5}
$$

梯度下降会压低概率最大的第 1 个动作、抬高另外两个，把分布往均匀的方向推。

**两项的比例。** 回报项的梯度在 0.1 的量级，熵项在 $10^{-5}$ 的量级，差了几千倍。
有明确的优势信号时，熵项几乎不起作用。它起作用的场合是优势接近 0 的时候：
稀疏奖励任务在发现第一个奖励之前，所有优势都在 $10^{-4}$ 量级或更小，熵项是主要的梯度来源，它让策略保持接近均匀的探索。
$\max(1, S)$ 保证了这时候优势不会被放大去压过它（第 7.3 节）。

### 9.5 V3 全部 stop-gradient 的清单

把第 4–9 节出现过的 sg 收在一张表里。能凭记忆写出这张表，就算掌握了 Dreamer V3 的训练。

| 编号 | 在哪个损失里 | sg 包住什么 | 一句话理由 |
| --- | --- | --- | --- |
| 1 | $\mathcal{L}_{\text{dyn}}$ | 后验 $q_\theta(z_t \mid h_t, o_t)$ | 只训练先验去追表征 |
| 2 | $\mathcal{L}_{\text{rep}}$ | 先验 $p_\theta(z_t \mid h_t)$ | 只训练表征去迁就先验，而且权重小 |
| 3 | 类别采样 | straight-through 里的 $\mathrm{sg}(p)$ | 前向是 one-hot，反向借用概率的梯度 |
| 4 | 想象的起点 | 后验状态 $m_t$ | 行为学习的梯度不能流进世界模型 |
| 5 | critic 损失 | 目标 $R^\lambda_k$ | 半梯度 TD；不去改 actor 和世界模型 |
| 6 | critic 损失 | 输入状态 $\hat m_k$ | critic 的误差不能流回 actor |
| 7 | critic 损失 | EMA critic 的输出 | 参照物不靠梯度更新 |
| 8 | actor 损失 | 归一化的优势 $A_k$ | 它是 score-function 估计器里的权重 |
| 9 | actor 损失 | 输入状态 $\hat m_k$ | REINFORCE 不穿过世界模型 |
| 10 | actor 与 critic 损失 | 权重 $w_k$ | continue 的预测不由行为损失来改 |

反过来，下面三处**没有** sg，同样重要：

| 位置 | 为什么不加 |
| --- | --- |
| 预测损失对 $z_t$、$h_t$ | 表征靠重建、奖励、continue 的梯度学出来 |
| λ-return 的递推本身 | 它只是数值计算。sg 加在用它的地方（第 5、8 项） |
| V1 式的 actor 损失里的 $R^\lambda_k$ | 沿模型反传时，梯度必须穿过回报、奖励头、critic 和转移函数 |

### 9.6 算法

```
算法：Dreamer V3 的一次训练步 DreamerV3Agent.train_step

 1. 从回放采样 (o, a, r, c, is_first)                                       形状 [B, T, ...]
 ── 世界模型 ──
 2. 损失_wm, (h, z) ← WorldModel.loss(...)                                  第 8.7 节
 3. 反向传播，裁剪梯度，更新 θ
 ── 想象 ──
 4. 轨迹 ← imagine_rollout(sg(h), sg(z), c, actor, H)                       第 2.6 节，N = B·T 条
 5. v ← critic.predict(m̂_{0:H})；从后往前算 R^λ_{0:H}                         形状 [H+1, N]
 ── actor ──
 6. scale ← ReturnNormalizer.update_and_scale(R^λ)                          max(1, S)
 7. A ← sg((R^λ_{0:H−1} − v_{0:H−1}) / scale)
 8. 损失_actor ← mean(sg(w) · (−A · log π_φ(â | sg(m̂)) − η · 熵))
 9. 反向传播，更新 φ
 ── critic ──
10. 损失_critic ← mean(sg(w) · (two-hot 交叉熵(critic(sg(m̂)), sg(R^λ))
                               + two-hot 交叉熵(critic(sg(m̂)), sg(EMA critic 的预测))))
11. 反向传播，更新 ψ；再做一次 EMA：ψ̄ ← d · ψ̄ + (1 − d) · ψ
```

三个损失各有一个优化器。更新的先后顺序不要紧，要紧的是每个损失只动自己那组参数。

**和环境交互的比例。** 每走一步真实环境，做多少次上面的训练步，是 Dreamer 最影响样本效率和运行时间的旋钮。
这个比例越高，每一份真实数据被用得越充分，训练也越慢，还越容易让 actor 钻进模型的漏洞里。
论文对不同基准用了不同的比例（具体数值待核对），本课讲义里的取值按本机的算力定。

### 9.7 代码对应

对应 `09-dreamer-v3-agent.py`。

| 名称 | 做什么 | 关键张量 |
| --- | --- | --- |
| `critic_loss_twohot(critic, ema_critic, states, returns, weights)` | 第 9.2 节 | `states` 为 `[H, N, D]`，`returns` 为 `[H, N]` |
| `actor_loss_reinforce(actor, states, actions, advantages, weights, eta)` | 第 9.3 节 | `actions` 为 `[H, N, A]`，`advantages` 为 `[H, N]` |
| `DreamerV3Agent.train_step(batch)` | 第 9.6 节 | 输入 `[B, T, ...]` |
| `DreamerV3Agent.act(obs, state)` | 交互时用：`observe()` 走一步，再从 actor 采样 | 输入单帧，状态 `(h, z)` 各为 `[1, ·]` |

**加入排行榜。** Phase 5 的项目建立了一张连续控制排行榜（PETS、PPO、SAC、MBPO）。
本章的项目把 Dreamer 加进去：状态输入的任务上，编码器和解码器换成 MLP，输入和重建目标都过 symlog，其余不变。

> 📌 待回填：Dreamer V3 在像素版 `Pendulum-v1` 上的学习曲线；状态输入版在排行榜任务上、与 Phase 5 相同评估协议下的回报—环境步数曲线和汇总表（与 PETS、PPO、SAC、MBPO 并列）；连续动作下 REINFORCE 与沿模型反传两种 actor 梯度的对比，来自 `09-dreamer-v3-agent.py` 和本章项目。

### 9.8 常见误区

- **误区：actor 损失里的 $\mathrm{sg}(A_k)$ 丢掉了回报对策略的依赖，梯度是不完整的。**
  实际情况：这正是 score-function 估计器的形式。回报对策略的依赖体现在 $\nabla \ln \pi$ 乘以回报这个结构里，不靠对回报求导。
- **误区：V3 的 critic 学的是归一化之后的回报。**
  实际情况：critic 学的是原始尺度的 λ-return（经过 symlog 和 two-hot 编码）。归一化只发生在 actor 损失的优势上。
- **误区：REINFORCE 方差大，所以 V3 的 actor 一定比 V1 的差。**
  实际情况：想象里每次更新有上万条转移，基线又是一个学得不错的 critic，方差被压得很低。
  而沿模型反传要穿过类别隐变量的 straight-through 和很多步的雅可比连乘，偏差和梯度爆炸的问题并不小。两者各有适用的场合。
- **误区：熵系数 $3 \times 10^{-4}$ 太小，等于没有。**
  实际情况：有明确优势信号时它确实可以忽略；没有信号时（稀疏奖励的早期）它是唯一的梯度来源，决定了策略会不会过早停止探索。

---

## 10. 诊断与消融：怎么知道哪里出了问题

### 10.1 直觉

Dreamer 是一条三级流水线：世界模型的输出喂给想象，想象的输出喂给 actor 和 critic。
最终的回报曲线只是最下游的一个指标。它不涨，可能是三级里任何一级的问题，光盯着它查不出来。

这和排查线上故障是一回事：用户说"页面打不开"，你不会去盯着页面看，而是沿着调用链一级一级看指标，
先确认上游是好的，再看下游。本节给 Dreamer 的每一级配上指标，并约定排查的顺序：**从上游到下游**。

第二件事是消融实验（ablation）：把某个部件关掉，看性能掉多少。它回答"这个部件到底有没有用"。
工程上的对应物是二分定位和特性开关：每次只动一个开关，其他条件不变。

### 10.2 定义：三级指标

| 级 | 指标 | 健康的样子 | 不健康的信号 |
| --- | --- | --- | --- |
| 世界模型 | 重建损失 | 持续下降后走平；重建画面里和任务有关的物体清晰 | 画面模糊一片；小物体消失 |
| | 未截断的 KL（每步，nat） | 稳定在 free bits 以上的某个值 | 贴着 0（表征坍塌）；持续增长（先验学不会） |
| | 奖励预测误差，**只在奖励非零的时间步上统计** | 明显低于"全猜 0"的误差 | 奖励头只会输出 0 |
| | 开环预测视频 | 前若干步和真实画面接近，之后逐渐偏离 | 第一步就不对；或者画面冻住不动 |
| 想象 | 价值校准：$V_\psi(m_t)$ 对比真实回合里从第 $t$ 步起的折扣回报 $G_t$ | 两者的平均差接近 0，相关性高 | $V_\psi$ 系统性地高于 $G_t$，而且差距越来越大 |
| | continue 权重 $w_k$ 随 $k$ 的衰减 | 和任务的回合长度相符 | 所有权重都是 1（没学会终止）；或者很快衰减到 0 |
| 行为 | 策略的熵 | 逐渐下降，但不到 0 | 一开始就掉到 0；或者始终贴着最大值 |
| | 回报的范围 $S$ | 随训练缓慢变化 | 突变；长期为 0 |
| | 梯度范数（三个网络分开记） | 稳定 | 尖峰；持续增长 |

**价值校准**是本节最重要的一个指标，它直接回答"想象和现实对不对得上"。做法：
用当前的策略在真实环境里跑几个评估回合，每一步记下 critic 给的 $V_\psi(m_t)$；
回合结束后，从后往前算出每一步真实的折扣回报 $G_t$；比较两者。

$$
\text{偏差} = \frac{1}{T} \sum_{t} \big( V_\psi(m_t) - G_t \big)
$$

一定要用当前策略新跑的回合，不能用回放里的旧回合：旧回合的 $G_t$ 是旧策略的回报，和当前 critic 评估的不是同一个策略。
被步数上限截断的回合，末尾若干步的 $G_t$ 缺了截断之后的部分，统计时要去掉。

### 10.3 推导：为什么想象的回报会系统性地偏高

Phase 1 第 8.5 节讲过模型被钻空子。这里用一个能手算的例子说明：**只要对带噪声的估计做最大化，结果就会偏高**，不需要模型有任何系统性的偏差。

设有 $n$ 个候选动作，真实价值是 $\mu_1, \dots, \mu_n$。模型给出的估计带有零均值的误差：$\hat\mu_i = \mu_i + \epsilon_i$，$\mathbb{E}[\epsilon_i] = 0$。

**第一步：对任意一个固定的 $i$，最大值不小于第 $i$ 个。**

$$
\max_j \hat\mu_j \ge \hat\mu_i
$$

**第二步：两边取期望。**

$$
\mathbb{E}\big[ \max_j \hat\mu_j \big] \ge \mathbb{E}[\hat\mu_i] = \mu_i
$$

**第三步：上式对每个 $i$ 都成立，所以对其中最大的那个也成立。**

$$
\mathbb{E}\big[ \max_j \hat\mu_j \big] \ge \max_i \mu_i
$$

估计值的最大值的期望，不小于真实的最大值。每个估计都是无偏的，挑出来的那个却是偏高的。

**手算例子。** 两个动作，真实价值都是 0。每个估计独立地以一半概率是 $+1$、一半概率是 $-1$。四种等可能的情况：

| $\hat\mu_1$ | $\hat\mu_2$ | 最大值 |
| --- | --- | --- |
| $+1$ | $+1$ | $+1$ |
| $+1$ | $-1$ | $+1$ |
| $-1$ | $+1$ | $+1$ |
| $-1$ | $-1$ | $-1$ |

$$
\mathbb{E}\big[ \max(\hat\mu_1, \hat\mu_2) \big] = \frac{1 + 1 + 1 - 1}{4} = 0.5
$$

真实的最大值是 0，按估计挑出来的那个的期望估计值是 0.5。候选越多、噪声越大，偏高得越多。

actor 在想象里做的正是这件事：它被训练去选"模型和 critic 认为回报高"的动作。
模型和 critic 的误差有正有负，actor 专挑正的那边走。所以想象的回报高于真实的回报是常态，
关键是差距稳不稳定：稳定在一个小值上没有问题；持续扩大说明 actor 在钻漏洞，而新采的真实数据没能把漏洞补上。
Dreamer 的纠错回路和 PETS 相同（Phase 1 第 9.4 节）：actor 被漏洞吸引过去，在真实环境里碰壁，碰壁的数据进了回放，下一轮模型就修正了。

### 10.4 排查顺序与症状表

```
   回报不涨
      │
      ▼
   ① 世界模型准吗？        看重建画面、未截断的 KL、非零奖励的预测误差、开环视频
      │ 是
      ▼
   ② 想象可信吗？          看价值校准的偏差、continue 权重
      │ 是
      ▼
   ③ critic 学对了吗？     看 λ-return 与价值的差、critic 损失、价值的量级是否合理
      │ 是
      ▼
   ④ actor 在学吗？        看策略的熵、优势的量级、回报的范围 S
```

| 症状 | 最可能出问题的那一级 | 先查什么 |
| --- | --- | --- |
| 重建清晰，但回报完全不涨，奖励头的输出恒为 0 | 世界模型（奖励头） | 奖励和状态是否错开了一位；稀疏奖励下奖励头有没有学到非零的值 |
| KL 贴着 0，重建模糊 | 世界模型（表征坍塌） | free bits 是否生效；两个 sg 是否写反 |
| 想象的回报很高，真实回报很低，差距越来越大 | 想象 | 模型是否被钻空子；训练步数与环境步数的比例是否太高；起点是否停止了梯度 |
| 价值一路增长到很大的数 | critic | continue 是否进了 λ-return；EMA 正则是否生效 |
| 策略的熵几步之内掉到 0 | actor | 回报归一化是否生效；优势的量级 |
| 策略的熵始终是最大值 | actor 或更上游 | 优势是否全为 0（奖励头或 critic 没学到东西） |

### 10.5 怎么做消融才算数

三条纪律：

1. **一次只改一处。** 同时改两处，结果变了也不知道是谁的功劳。
2. **其余条件完全相同。** 相同的环境步数、相同的评估协议、相同的随机种子集合。
3. **多个随机种子。** 强化学习的结果在不同种子之间差别很大，一个种子的对比什么也说明不了。

第 3 条可以量化。设不同种子的最终回报的标准差是 $\sigma$，每种配置跑 $n$ 个种子取平均。
单个平均值的标准误（standard error）是 $\sigma / \sqrt{n}$，这是独立同分布样本的均值的标准差公式。
两种配置的平均值之差，方差是两个平均值的方差之和，所以差值的标准误是：

$$
\mathrm{SE}_{\text{差}} = \sqrt{\frac{\sigma^2}{n} + \frac{\sigma^2}{n}} = \sigma \sqrt{\frac{2}{n}}
$$

**数值例子**（数字是假设的，只为说明量级）：$\sigma = 100$，两种配置的平均回报相差 50。

| 每种配置的种子数 $n$ | 差值的标准误 | 相差 50 说明得了问题吗 |
| --- | --- | --- |
| 1 | 141.4 | 不能。差距远小于随机波动 |
| 3 | 81.6 | 不能。差距约 0.6 个标准误 |
| 5 | 63.2 | 不能。差距约 0.8 个标准误 |
| 16 | 35.4 | 勉强。差距约 1.4 个标准误 |
| 32 | 25.0 | 可以。差距是标准误的 2 倍 |

所以小的差距靠几个种子是分辨不出来的。本机算力有限时，宁可少做几组消融，也要保证每组至少 3 个种子，
把每个种子的曲线都画出来而不只报平均值，并且只对"差距远大于种子间波动"的结果下结论。

本章讲义计划做的消融：

| 消融 | 关掉什么 | 预期（定性） |
| --- | --- | --- |
| 无 free bits | 去掉 $\max(1, \cdot)$ | KL 更低，重建和奖励预测变差 |
| 无 KL balancing | $\beta_{\text{dyn}} = \beta_{\text{rep}}$ | 训练初期更不稳定 |
| 高斯隐变量 | 换回 Phase 3 的高斯 RSSM | 在本课的简单连续控制任务上差别可能不大 |
| 平方误差 critic | two-hot 换成平方误差 | 奖励尺度放大后变差 |
| 无回报归一化 | 优势不除以 $\max(1, S)$ | 奖励尺度放大后策略过早变成确定性的 |
| 想象视野 | $H$ 取 5、15、30 | 太短依赖 critic，太长模型误差累积 |
| 起点不停止梯度 | 故意引入一个 bug | 世界模型的重建变差，想象回报虚高 |
| 奖励错开一位 | 故意引入一个 bug | 重建正常，回报不涨 |

最后两行是"故意写错"的实验。亲眼看过这两个 bug 的症状，以后遇到就能认出来。

> 📌 待回填：价值校准的散点图与偏差随训练的变化；上表每一项消融的学习曲线（每组至少 3 个种子）与一句话结论，来自 `10-diagnostics-and-ablations.py`。

### 10.6 算法

```
算法：价值校准 imagined_vs_real_return

输入：智能体，环境，评估回合数 E，折扣 γ
 1. 对每个评估回合：
        用当前 actor 跑完一个回合，每一步记录 r_t 和 V_ψ(m_t)                长度 T
        从后往前算 G_t ← r_t + γ · G_{t+1}（终止时 G_T = 0）
        如果回合是被步数上限截断的，丢掉最后约 1/(1−γ) 步
 2. 把所有 (V_ψ(m_t), G_t) 放在一起，计算：平均偏差、相关系数
 3. 画散点图：横轴 G_t，纵轴 V_ψ(m_t)，对角线是完美校准
```

### 10.7 代码对应

对应 `10-diagnostics-and-ablations.py`。

| 函数 | 做什么 | 关键张量 |
| --- | --- | --- |
| `imagined_vs_real_return(agent, env, episodes)` | 第 10.6 节 | 每个回合的 `values`、`returns` 为 `[T]` |
| `open_loop_video(world_model, batch, context)` | 给前 `context` 帧，之后只用先验，解码成视频 | 输出 `[T, 3, 64, 64]` |
| `run_ablation(config_overrides, seeds)` | 用一组改动过的配置训练多个种子，返回曲线 | 输出 `[种子数, 评估点数]` |

### 10.8 常见误区

- **误区：回报曲线在涨，说明各个部件都没问题。**
  实际情况：有的 bug 只是让学习变慢或上限变低，不会让曲线不涨。分级的指标能在回报曲线上看不出来的时候发现它们。
- **误区：想象的回报高于真实回报，说明实现有 bug。**
  实际情况：对带噪声的估计做最大化，偏高是必然的（第 10.3 节）。要看的是差距是否稳定。
- **误区：重建画面越清晰，世界模型越好。**
  实际情况：画面清晰只说明面积大的部分重建得好。对控制要紧的往往是小物体和奖励预测，要单独看。
- **误区：消融里某个部件去掉后性能没掉，说明它没用。**
  实际情况：只能说明它在这个任务、这个设置下没用。V3 的那些技巧，每一个都只在一部分任务上起作用。

---

## 11. 方法对比与选型

### 11.1 三代 Dreamer：每一处改动解决了什么问题

| 部件 | V1（2019） | V2（2020） | V3（2023） | 这一步改动解决了什么 |
| --- | --- | --- | --- | --- |
| 随机隐变量 | 30 维高斯 | $32 \times 32$ 类别 + straight-through | 同 V2，加 1% unimix | V2：高斯先验表达不了多峰的未来。V3：概率接近 0 时 KL 出现尖峰 |
| KL 项 | 一项，free nats = 3 | KL balancing（$\alpha = 0.8$） | $\mathcal{L}_{\text{dyn}}$ + $\mathcal{L}_{\text{rep}}$，各带 1 nat 的 free bits | V2：表征被拉向没学好的先验。V3：KL 权重要按环境的视觉复杂度调 |
| 观测与奖励的预测 | 高斯（平方误差） | 高斯（平方误差） | 向量观测用 symlog；奖励用 symlog + two-hot | V3：损失和梯度的大小随数值尺度变 |
| 终止 | 主实验不处理 | 预测折扣因子 | continue 头 | V2：Atari 的回合会提前结束 |
| actor 的梯度 | 沿模型反传 | 离散动作 REINFORCE；连续动作沿模型反传 | REINFORCE + 回报归一化（初版对连续动作仍沿模型反传，待核对） | V2：离散动作不可重参数化。V3：回报项和熵项的比例随任务变 |
| actor 的熵正则 | 无 | 有，系数按领域分别设 | 有，固定系数 | V2：离散策略过早变成确定性的。V3：不再按领域调系数 |
| critic 的损失 | 平方误差 | 平方误差 | symlog + two-hot 交叉熵 | V3：回归梯度随回报的尺度变；稀疏奖励下的多峰目标 |
| critic 的稳定 | 无 | 目标网络 | 用当前 critic 算目标 + EMA 正则 | V2：目标追着自己跑。V3：目标网络带来的信息滞后 |
| 超参数 | 针对连续控制 | Atari 和连续控制各一套 | 一套跨所有领域 | V3 的全部改动都是为了这一条 |

### 11.2 三种 actor 梯度

| | 沿模型反传（pathwise） | REINFORCE（score-function） | straight-through 穿过离散动作 |
| --- | --- | --- | --- |
| 需要动作可重参数化吗 | 需要（连续动作） | 不需要 | 不需要 |
| 需要世界模型对状态和动作可导吗 | 需要 | 不需要 | 需要 |
| 偏差 | 无偏（隐变量是高斯时）；穿过类别隐变量时有偏 | 无偏 | 有偏 |
| 方差 | 小 | 大，靠基线和大批量想象压低 | 小 |
| 随视野增长的风险 | 雅可比连乘，梯度爆炸或消失 | 方差增大 | 两者都有 |
| 对模型误差的敏感程度 | 高：梯度精确地指向模型认为好的方向，包括模型的漏洞 | 较低 | 高 |
| 谁在用 | V1；V2 的连续控制 | V2 的 Atari；V3 | V2 论文试过，效果不如 REINFORCE |

选型建议：动作离散，用 REINFORCE。动作连续，两种都可以试，沿模型反传通常在低维、平滑的任务上收敛更快；
如果观察到 actor 的梯度范数出现尖峰，换成 REINFORCE 或缩短视野。

### 11.3 Dreamer 和前几章的方法

| | PlaNet（P3） | MBPO（P5） | SAC、PPO（P5） | Dreamer（本章） |
| --- | --- | --- | --- | --- |
| 世界模型 | RSSM | 状态空间的 MLP 集成 | 无 | RSSM |
| 模型怎么用 | 决策时规划（CEM） | 生成短展开的数据，喂给 SAC 的回放 | 不适用 | 在隐空间里想象，直接在想象轨迹上训练 actor-critic |
| 想象从哪里出发 | 当前状态 | 回放里的真实状态 | 不适用 | 回放里的后验状态 |
| 行为学习用的数据 | 无（不学行为） | 真实数据 + 模型数据 | 真实数据 | 只有想象数据 |
| 决策时的计算 | 重 | 轻 | 轻 | 轻 |
| 输入 | 像素 | 低维状态 | 低维状态（像素需要另加编码器） | 像素或低维状态 |
| 模型不确定性 | 无显式估计 | 集成 | 不适用 | 无显式估计（随机隐变量只刻画偶然不确定性） |

最后一行是 Dreamer 相对 PETS、MBPO 的一个缺口：它没有集成，不估计认知不确定性。
Plan2Explore 给 Dreamer 加上了一个集成，用分歧来驱动探索（第 12.5 节）。

### 11.4 把 Dreamer 当成模板：后面各章换掉了哪个部件

Dreamer 可以拆成五个插槽。Phase 7–10 的方法大多只换其中一两个：

```
  ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ ① 表征    │──▶│ ② 序列模型 │──▶│ ③ 学习信号     │──▶│ ④ 怎么用模型   │──▶│ ⑤ 行为怎么学   │
  │ 类别隐变量 │   │ GRU      │   │ 重建像素       │   │ 在想象中训练   │   │ actor-critic │
  │          │   │          │   │ + 奖励 + KL   │   │              │   │              │
  └──────────┘   └──────────┘   └──────────────┘   └──────────────┘   └──────────────┘
```

| 章 | 方法 | 换掉的插槽 | 换成什么 |
| --- | --- | --- | --- |
| P7 | MuZero | ③、④ | 不重建，只预测奖励、价值、策略；决策时用 MCTS 规划 |
| P7 | TD-MPC2 | ③、④ | 不重建，靠奖励、价值和隐状态自洽来学表征；决策时用 MPPI 规划，策略当先验，价值补视野之外 |
| P8 | IRIS | ①、② | 每帧多个离散 token；自回归 Transformer。仍然在想象中训练 actor-critic |
| P8 | STORM、TWM | ② | 保留每帧一个类别隐变量，GRU 换成 Transformer |
| P9 | DIAMOND | ①、②、③ | 不压缩成隐变量，用扩散模型直接生成下一帧。仍然在想象中训练 actor-critic |
| P9 | Dreamer 4 | ①、②、③ | Dreamer 这条线的后续，世界模型换成可扩展的视频生成模型（细节见 Phase 9） |
| P10 | DINO-WM、V-JEPA 2 | ①、③、④、⑤ | 冻结的预训练特征；预测表征，不重建，也不需要奖励；决策时用 CEM 规划，不学策略 |

读后面的论文时，先问"它相对 Dreamer 换了哪个插槽、为什么换"，通常就能抓住要点。

### 11.5 排行榜：Dreamer 加入

Phase 5 的项目在同一套评估协议下对比了 PETS、PPO、SAC、MBPO。本章的项目把 Dreamer 加进去。

| 方法 | 类别 | 怎么用模型 | 来自 |
| --- | --- | --- | --- |
| PETS | 基于模型 | 决策时规划 | Phase 1 |
| PPO | 无模型，在策略 | 不用 | Phase 5 |
| SAC | 无模型，离策略 | 不用 | Phase 5 |
| MBPO | 基于模型 | 模型数据喂给 SAC | Phase 5 |
| Dreamer V3 | 基于模型 | 在想象中训练 actor-critic | 本章 |

> 📌 待回填：上表各方法在排行榜任务上、相同环境步数下的回报（多个种子的均值与标准差）和单步决策耗时，来自本章项目 `projects/phase-06-mini-dreamer/`。

定性预期：样本效率上，基于模型的方法应当优于 PPO；Dreamer 和 SAC、MBPO 在低维状态任务上谁更好，没有先验的把握，以实测为准。
Dreamer 的优势领域是像素输入，那里 SAC 和 MBPO 需要另外解决表征学习的问题。

---

## 12. 论文精读卡片

### 12.1 Dreamer V1

**Hafner, Lillicrap, Ba, Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. ICLR 2020. arXiv:1912.01603**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | PlaNet 这类"学模型 + 在线规划"的方法决策慢，而且只看得到规划视野之内 |
| 核心方法 | 在 RSSM 的隐空间里想象轨迹；学一个价值函数概括视野之外的回报；actor 的梯度沿可微的世界模型反传 |
| 关键实验 | DeepMind Control Suite 的 20 个像素控制任务；对想象视野、有无价值函数、不同的表征学习目标做了消融 |
| 主要结论 | 摘要中给出：在 20 个视觉控制任务上，数据效率、计算时间和最终性能都超过此前的方法。正文报告的平均分是 Dreamer 823（$5 \times 10^6$ 环境步）、PlaNet 332（同样步数）、无模型的 D4PG 786（$10^8$ 环境步）（待核对） |
| 局限 | 见第 3.7 节：主体方法针对连续动作；高斯隐变量；超参数针对连续控制 |
| 读的时候注意 | 第 3 节（行为学习）是全文的核心：$\mathrm{V}_\lambda$ 的定义和 actor、critic 两个目标。消融里"没有价值函数"的那条曲线，对应本章第 1.4 节 |
| 与本课实现的差异 | 📌 待回填：写完 `03-dreamer-v1-actor-critic.py` 后列出 |

### 12.2 Dreamer V2

**Hafner, Lillicrap, Norouzi, Ba. Mastering Atari with Discrete World Models. ICLR 2021. arXiv:2010.02193**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 世界模型方法在 Atari 上一直不如调得好的无模型方法 |
| 核心方法 | 类别隐变量 + straight-through；KL balancing；离散动作的 actor 用 REINFORCE |
| 关键实验 | Atari 的 55 个游戏，2 亿帧；和单 GPU 的无模型方法（IQN、Rainbow）在相同计算预算下比较；对类别隐变量、KL balancing、图像梯度和奖励梯度做了消融 |
| 主要结论 | 摘要中给出：这是第一个在单独训练的世界模型里学习行为、并在 Atari 的 55 个任务上达到人类水平的智能体；相同的计算预算和运行时间下，超过 IQN 和 Rainbow 的最终性能 |
| 局限 | Atari 和连续控制各用一套超参数；奖励需要按领域做处理 |
| 读的时候注意 | KL balancing 的那段伪代码只有几行，对照本章第 5.5 节读；讨论类别隐变量为什么有效的那一段是几个假说，不是结论 |

### 12.3 Dreamer V3

**Hafner, Pasukonis, Ba, Lillicrap. Mastering Diverse Domains through World Models. arXiv:2301.04104；Nature 640 (2025)**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 强化学习算法换一个领域就要重新调超参数，这限制了它的实际应用 |
| 核心方法 | 在 V2 的基础上加一组稳健性技巧：symlog、two-hot、free bits、unimix、分位数回报归一化、critic 的 EMA 正则 |
| 关键实验 | 连续控制、Atari、程序生成的环境、三维环境、Minecraft 等多个领域；对每一项稳健性技巧做了消融；考察了模型大小的影响 |
| 主要结论 | 摘要中给出：用同一套配置，在 150 多个不同的任务上超过各领域专门的方法；它是第一个不靠人类数据和课程设计、从零开始在 Minecraft 里采到钻石的算法 |
| 局限 | 仍然要重建像素；循环网络的记忆有限；想象在隐空间里，生成不了高保真的画面（Phase 8、9 的出发点） |
| 读的时候注意 | 论文有多个版本，超参数和个别公式有出入（$\beta_{\text{dyn}}$、连续动作的 actor 梯度、two-hot 的桶在哪个空间）。读哪一版，就以那一版的公式和超参数表为准 |
| 与本课实现的差异 | 📌 待回填：写完 `09-dreamer-v3-agent.py` 后列出（网络规模、训练比例、省略的结构细节） |

### 12.4 DayDreamer

**Wu, Escontrela, Hafner, Goldberg, Abbeel. DayDreamer: World Models for Physical Robot Learning. CoRL 2022. arXiv:2206.14176**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | Dreamer 的结果都在仿真里。它的样本效率够不够直接在真实机器人上在线学习 |
| 核心方法 | 把 Dreamer 直接用到真实机器人上，不用仿真器；数据采集和训练异步进行 |
| 主要结论 | 摘要中给出：四足机器人从零开始、不需要重置，1 小时内学会翻身、站起和行走，被推之后 10 分钟内适应；两种机械臂从相机图像和稀疏奖励学会抓取和放置多个物体；轮式机器人只靠相机图像学会导航到目标位置。所有实验用同一套超参数 |
| 局限 | 任务相对简单；奖励需要人工设计或额外的传感器 |
| 读的时候注意 | 这篇是本章和 Phase 11 之间的桥：世界模型省下来的样本，在真机上换算成了小时数 |

### 12.5 Plan2Explore

**Sekar, Rybkin, Daniilidis, Abbeel, Hafner, Pathak. Planning to Explore via Self-Supervised World Models. ICML 2020. arXiv:2005.05960**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | Dreamer 的探索只靠策略的随机性；而且学到的行为绑定在一个任务的奖励上 |
| 核心方法 | 在世界模型的隐空间上训练一个集成，用集成成员之间预测的分歧作为内在奖励（对应 Phase 1 第 8 节的认知不确定性）；在想象中训练一个专门追求这种奖励的探索策略。探索阶段不看任何任务奖励；之后拿到任务奖励，再在同一个世界模型里零样本或少样本地学出任务策略 |
| 主要结论 | 没有任务奖励也能学到对下游任务有用的世界模型 |
| 读的时候注意 | "在想象中"这一点是关键：探索策略追求的是**预期的**新奇，在到达之前就知道哪里值得去。这回应了 Phase 1 小结里留下的问题"怎么主动探索未知区域" |

### 12.6 Director（选读）

**Hafner, Lee, Fischer, Abbeel. Deep Hierarchical Planning from Pixels. NeurIPS 2022. arXiv:2206.04114**

在 Dreamer 的世界模型上做分层：上层策略每隔若干步在隐空间里选一个目标，下层策略负责到达这个目标，两层都在想象中训练。
针对的是奖励非常稀疏、需要长程规划的任务。读摘要和方法概览图即可。

---

## 13. 实现陷阱与调试清单

| 现象 | 可能的原因 | 怎么排查 | 怎么修 |
| --- | --- | --- | --- |
| 重建清晰，回报不涨，奖励预测看起来总慢一拍 | 奖励和状态错开了一位：奖励头在 $m_t$ 上预测了 $r_t$ 而不是 $r_{t-1}$ | 打印一条序列，核对 `reward[t]` 是不是到达 `obs[t]` 时拿到的奖励 | 按第 2.2 节的图统一对齐方式，回放、世界模型、λ-return 三处保持一致 |
| KL 很快降到接近 0，重建模糊成一片平均画面 | 表征坍塌：没有 free bits；或者两个 sg 写反了，后验被大权重拉向先验 | 记录未截断的 KL；检查 $\mathcal{L}_{\text{dyn}}$ 里 detach 的是不是后验 | 加 free bits；核对 sg 的位置 |
| KL 偶尔出现巨大的尖峰，随后损失变成 NaN | 类别概率接近 0；先 softmax 再取 log | 记录 KL 的最大值和 logits 的范围 | 加 unimix；用 `log_softmax` |
| 世界模型的重建在训练中途开始变差，想象的回报虚高 | 想象的起点没有 detach，actor 的损失改写了世界模型 | 对 actor 损失反向传播后，检查世界模型参数的梯度是否全为空 | 起点 detach；每个损失只对自己的优化器调 `step()` |
| actor 的梯度范数周期性地出现尖峰 | 沿模型反传时雅可比连乘 | 分别记录三个网络的梯度范数 | 梯度裁剪；缩短视野；换成 REINFORCE |
| 价值一路增长到不合理的大数 | λ-return 里漏了 continue；终止之后的想象状态参与了训练 | 人工构造一条必然终止的轨迹，检查 λ-return | 递推里乘上 $\hat c_{k+1}$；损失乘上权重 $w_k$ |
| 回合明明会提前结束，智能体却不怕"死" | 把 `truncated` 当成了 `terminated`，或者反过来 | 统计回放里 continue 标志为 0 的比例 | 只有 `terminated` 时 continue 为 0；`truncated` 时仍为 1 |
| 策略的熵几步之内掉到 0，回报卡在很低的水平 | 没做回报归一化；优势的尺度远大于熵项 | 打印优势的均值和分位数、$S$ 的值 | 检查 `ReturnNormalizer`；确认除的是 $\max(1, S)$ |
| 稀疏奖励任务上策略乱抖，学不到东西 | 回报归一化没有 $\max(1, \cdot)$ 的下限，噪声被放大 | 打印 $S$；看是不是远小于 1 | 加下限 |
| KL 损失在数值上不起作用，表征里塞满了细节，先验学不会 | 重建损失对像素取了平均而不是求和，KL 相对被放大了上万倍；或者反过来 | 比较重建项和 KL 项的量级 | 重建对像素求和，再对批和时间取平均 |
| two-hot 的损失降不到 0，以为没学好 | 交叉熵的下限是目标分布的熵，不是 0 | 比较读出的预测值和目标值，而不是看损失 | 不用修。另外记录预测值的误差 |
| critic 的预测在大数值上全部顶在同一个值 | 目标超出了桶的范围被截断 | 检查桶的范围；检查是否忘了 symexp | 桶取 symlog 空间里的 $[-20, 20]$ |
| 随机环境里价值系统性偏低 | two-hot 的读出在 symlog 空间取期望再变换（第 6.8 节的方式 A） | 用"一半 0、一半大奖励"的玩具目标测试 | 换成方式 B：桶的位置先做 symexp，再取期望 |
| 序列片段的前几步重建特别差 | 片段从回合中间截取，初始的 $h$ 是零，前几步的后验状态不可靠 | 按时间步分别统计重建损失 | 可以接受；或者不把前几步用作想象的起点 |
| 跨回合的片段里，新回合的第一帧重建成了上一回合的画面 | 回合边界处没有重置 $h$、$z$ 和上一步动作 | 检查 `is_first` 的处理 | 在 `observe()` 里按 `is_first` 清零状态 |
| 训练时回报不错，评估时明显变差 | 交互时模型状态没有正确维护（忘了更新 $h$，或者回合开始时没重置） | 对比训练里 `observe()` 算出的状态和交互时在线算出的状态 | `act()` 里每步调用一次 `observe()`，回合开始时重置 |
| 在 MPS 上报错或者结果是 NaN | 用了 float64；或者某个算子在 MPS 上不支持 | 检查张量的 dtype | 全部用 float32；分位数等统计量可以挪到 CPU 上算 |

调试顺序沿用 Phase 1 的习惯，从上游到下游，每一步确认没问题再往下走：

1. 回放数据：画面、动作、奖励、continue 的对齐。
2. 世界模型：重建、奖励预测、KL、开环预测视频。
3. 想象：轨迹的形状、权重、stop-gradient 的位置。
4. critic：在一个固定的随机策略下，价值校准是否成立。
5. actor：放开策略更新，看熵和回报。

第 4 步值得单独做：把 actor 冻结成随机策略，只训练世界模型和 critic，然后检查 $V_\psi$ 和真实折扣回报是否对得上。
这一步通过了，说明世界模型、想象、λ-return、critic 这条链是通的，剩下的问题只可能在 actor。

---

## 14. 常见问题（FAQ）

### Q1：Dreamer 算基于模型的方法，还是无模型的方法？

基于模型。它学了一个显式的世界模型，并且行为学习完全依赖这个模型。
容易让人犹豫的地方是：它在想象里跑的 actor-critic 本身是一个无模型算法。
可以这样理解：Dreamer 把无模型算法的"环境"换成了学出来的模型。真实环境的数据只用来训练模型。

### Q2：为什么 actor 和 critic 的输入是 $(h_t, z_t)$，而不是图像或者解码出的画面？

三个原因。第一，快：想象不需要生成像素，一步只是一次 GRU 和几个小 MLP 的前向计算。
第二，$(h_t, z_t)$ 已经把历史压缩进去了，是一个近似满足马尔可夫性的状态；单帧图像不是（Phase 1 第 3.4 节）。
第三，表征是为了预测未来而学的，和任务无关的像素细节已经被部分滤掉。
也有方法让策略直接看生成的画面（Phase 8 的 IRIS、Phase 9 的 DIAMOND），代价是想象慢得多。

### Q3：想象视野为什么是 15 步？能不能更长？

15 是三代 Dreamer 论文都用的值。它受两头的约束：太短，回报几乎全靠 critic 自举，学得慢；
太长，模型的误差累积（Phase 1 第 5 节），沿模型反传时梯度还会爆炸（第 3.4 节）。
V1 论文的消融显示，有了价值函数之后，性能对视野长度不敏感，短视野就够用。
视野之外的事情交给 critic，critic 的有效视野由折扣因子决定。

### Q4：想象的起点为什么要停止梯度？让 actor 的梯度也去改进世界模型，不是更好吗？

听上去合理，实际会坏事。actor 的目标是让想象的回报变高。要是它能改世界模型，最省力的办法是让模型"相信"回报很高：
改奖励头、改转移，都比真的学会好的行为容易。世界模型应该只对真实数据负责。
不过"让任务相关的梯度塑造表征"这个想法本身没有错，Phase 7 的 TD-MPC2 就是这么做的。
区别在于那里流进表征的是在真实数据上算的价值预测误差，不是"让想象的回报变高"。

### Q5：既然世界模型是可微的，V3 为什么反而用 REINFORCE？

几个因素。类别隐变量的 straight-through 让沿模型反传的梯度变成有偏的；雅可比连乘在长视野下不稳定；
离散动作本来就只能用 REINFORCE，统一成一种估计器更简单。而 REINFORCE 的主要弱点（方差大、费样本）
在想象里不是大问题：样本几乎免费，基线又是一个不错的 critic。
哪种更好和任务有关，V3 论文的不同版本在连续动作上的选择也不同（待核对）。

### Q6：Dreamer 靠什么探索？

默认只靠两样：策略自身的随机性（从 actor 的分布里采样），以及熵正则不让这个分布过早收窄。
没有内在奖励，也没有对认知不确定性的估计。这在奖励稠密的任务上够用，在奖励很稀疏的任务上是短板。
Plan2Explore（第 12.5 节）给它加上了用模型分歧驱动的探索。

### Q7：重建损失会不会把模型的容量浪费在和任务无关的背景上？

会。重建损失对每个像素一视同仁，背景面积大，就占了大部分损失。
画面里有和任务无关、又很难预测的内容时（晃动的树叶、电视里播放的视频），这个问题很明显。
Dreamer 的缓解办法是让奖励的梯度也流进表征，但主导的仍然是重建。
这正是 Phase 7（不重建，只预测奖励和价值）和 Phase 10（在表征空间里预测）的出发点。

### Q8：Dreamer 能直接用在低维状态输入的任务上吗？

能。把卷积编码器和解码器换成 MLP，输入和重建目标过 symlog，其余不变。本章的项目就是这样把 Dreamer 加进排行榜的。
不过在低维状态任务上，Dreamer 相对 SAC、MBPO 的优势不一定明显：它最擅长解决的"从像素学表征"在这里不是瓶颈。

### Q9：三个网络应该一起更新，还是轮流更新？

每一个训练步里三个都更新一次，顺序是世界模型、actor、critic（或者后两者对调）。
它们共用同一批回放数据和同一批想象轨迹，但各有各的损失和优化器。
没有"先把世界模型训练好再训练策略"的阶段。模型需要策略采来的新数据才能在相关的区域变准，
这和 Phase 1 第 9 节 PETS 的闭环是同一个道理。

### Q10：训练步数和环境步数的比例怎么定？

这个比例决定每一份真实数据被用多少次。比例高，样本效率高，但训练慢，也更容易让 actor 过度拟合当前的模型（钻漏洞）。
经验上：先用较低的比例把整个流程跑通，看价值校准（第 10.2 节）；校准良好再逐步提高比例，直到想象回报和真实回报的差距开始扩大。

### Q11：论文里的"free bits"阈值是 1 nat，是不是太小了？

看上去小，和容量上限 110.9 nat 比只有不到 1%。但它的作用不是给表征留多少容量，而是当开关：
KL 降到 1 nat 以下，说明先验几乎已经能猜中后验，这时继续压 KL 没有好处，只会挤掉表征里的信息。
实际训练中 KL 通常稳定在阈值以上，表征的信息量由预测损失和 KL 之间的平衡决定。

### Q12：$\gamma = 0.997$ 对应多长的有效视野？

把折扣权重 $\gamma^j$ 对 $j$ 求和，得到 $1 / (1 - \gamma) = 1 / 0.003 \approx 333$ 步。这是"有效视野"的常用估计。
想象只有 15 步，其余的三百多步都靠 critic 的自举一步步传回来。

### Q13：Dreamer 在真实机器人上能用吗？

能，DayDreamer（第 12.4 节）就是直接在真机上在线学习。世界模型的价值在真机上最明显：真实交互很贵，想象几乎免费。
限制也很实际：需要奖励信号（人工设计，或者另外训练一个成功检测器）；想象里学到的策略会把模型的误差带进现实；
在线学习时机器人早期的随机动作可能不安全。Phase 11 会系统地讨论世界模型在机器人上的几种用法。

---

## 15. 自测题

做题时先不要看答案。概念题试着用自己的话说出来，计算题动手算。

### 第 1 题

画出（或用文字描述）Dreamer 的三个训练环节，并说明每个环节更新哪些参数、用的是真实数据还是想象数据。

<details><summary>参考答案</summary>

1. 学世界模型：从回放里采样真实的序列片段，更新世界模型的参数 $\theta$（编码器、序列模型、先验、解码器、奖励头、continue 头）。用真实数据。
2. 在想象中学行为：从这批序列的后验状态出发，用先验往前想象 $H$ 步，在想象轨迹上算 λ-return，更新 actor $\phi$ 和 critic $\psi$。用想象数据。
3. 与环境交互：用当前的 actor 在真实环境里走若干步，把数据存进回放。不更新任何参数。

真实数据只直接用于训练世界模型；actor 和 critic 只见过想象出的轨迹。

</details>

### 第 2 题

折扣因子 $\gamma = 0.99$，想象视野 $H = 15$。按折扣权重算，视野之外的回报占多大比例？
已知 $\ln 0.99 = -0.01005$，$e^{-0.1508} = 0.860$。这个数字说明了什么？

<details><summary>参考答案</summary>

视野之外的权重占比是 $\gamma^H$：

$$
0.99^{15} = e^{15 \times (-0.01005)} = e^{-0.1508} = 0.860
$$

86% 的折扣权重落在想象的 15 步之外，这部分全靠 critic 在末端的估计 $v_H$ 来补。
所以 critic 的质量决定了 actor 的质量；只做规划、没有价值函数的 PlaNet 会把这 86% 直接丢掉。

</details>

### 第 3 题

为什么想象轨迹从回放数据的后验状态出发，而不是从回合的初始状态出发？说出三点理由。

<details><summary>参考答案</summary>

1. 覆盖：想象只有十几步。从初始状态出发，永远想不到回合中后段的局面；从回放里的每个状态出发，起点覆盖了智能体到过的所有阶段。
2. 误差：开环展开的误差随步数累积。从后验状态出发，误差从 0 重新算起，最多累积 $H$ 步。
3. 接地：后验状态是看了真实观测之后算出的，贴着真实数据，模型在它附近最可信。

附带的好处：起点虽然来自旧策略的数据，起点之后的动作都由当前 actor 选出，所以想象数据是在策略的，不需要离策略修正。

</details>

### 第 4 题

想象轨迹 $H = 2$，$\gamma = 1$，$\lambda = 0.5$。预测奖励 $\hat r_0 = 1$、$\hat r_1 = 3$，价值 $v_1 = 2$、$v_2 = 4$，continue 都是 1。
求 $R^\lambda_0$、$R^\lambda_1$、$R^\lambda_2$。如果 $\hat c_2 = 0$，结果变成什么？

<details><summary>参考答案</summary>

递推式：$R^\lambda_k = \hat r_k + \gamma \hat c_{k+1} \big[ (1-\lambda) v_{k+1} + \lambda R^\lambda_{k+1} \big]$，$R^\lambda_2 = v_2$。

continue 都是 1 时：

$$
\begin{aligned}
R^\lambda_2 &= 4 \\
R^\lambda_1 &= 3 + (0.5 \times 4 + 0.5 \times 4) = 7 \\
R^\lambda_0 &= 1 + (0.5 \times 2 + 0.5 \times 7) = 5.5
\end{aligned}
$$

$\hat c_2 = 0$ 时，第 2 步是终止状态，它之后的价值不计入：

$$
\begin{aligned}
R^\lambda_1 &= 3 + 0 = 3 \\
R^\lambda_0 &= 1 + (0.5 \times 2 + 0.5 \times 3) = 3.5
\end{aligned}
$$

</details>

### 第 5 题

起点不是终止状态。想象 3 步，continue 头的输出依次是 0.9、0.9、0.5。求权重 $w_0$ 到 $w_3$，并说明权重的用途。

<details><summary>参考答案</summary>

$w_k = c_t \prod_{j=1}^{k} \hat c_j$，$c_t = 1$：

$$
w_0 = 1, \qquad w_1 = 0.9, \qquad w_2 = 0.9 \times 0.9 = 0.81, \qquad w_3 = 0.81 \times 0.5 = 0.405
$$

$w_k$ 是想象走到第 $k$ 步时回合还没结束的概率。actor 和 critic 在第 $k$ 步的损失乘上 $\mathrm{sg}(w_k)$，
这样终止之后那些没有意义的想象状态不参与训练。

</details>

### 第 6 题

Dreamer V1 的 actor 损失是 $-\sum_k R^\lambda_k$。对它反向传播时，梯度穿过了哪些网络？哪些网络的参数会被更新？
critic 的损失里有哪两个 stop-gradient，各自防止什么？

<details><summary>参考答案</summary>

梯度穿过：critic（$R^\lambda_k$ 里有 $v_k$）、奖励头、序列模型和先验（状态之间的转移）、actor 自己。
被更新的只有 actor 的参数 $\phi$。世界模型和 critic 在这里只是计算图里可微的函数，它们的参数不在 actor 的优化器里。
梯度到想象的起点为止，起点停止了梯度。

critic 损失里的两个 sg：

1. 目标 $\mathrm{sg}(R^\lambda_k)$：目标里含有 critic 自己的输出。不停止梯度，critic 会通过挪动目标来降低损失，而不是把预测修正到位。
2. 输入 $\mathrm{sg}(\hat m_k)$：想象状态对 actor 的参数可导。不停止梯度，critic 的回归误差会顺着状态流回 actor。

</details>

### 第 7 题

类别分布 $p = (0.5, 0.5)$，采样得到第 1 类。下游损失是 $L(z) = w^\top z$，$w = (3, 1)$。
用 straight-through 求损失对 logits 的梯度。它和期望损失的真实梯度相等吗？为什么？

<details><summary>参考答案</summary>

straight-through 的梯度是 $p_j (g_j - \bar g)$，其中 $g = \partial L / \partial z = w = (3, 1)$，$\bar g = 0.5 \times 3 + 0.5 \times 1 = 2$：

$$
\big( 0.5 \times (3 - 2), \;\; 0.5 \times (1 - 2) \big) = (0.5, \; -0.5)
$$

期望损失 $\mathbb{E}[L] = \sum_k p_k w_k$ 对 logits 的真实梯度是 $p_j (w_j - \bar w)$，同样是 $(0.5, -0.5)$。两者相等。

原因：下游是 $z$ 的线性函数时，$g$ 不依赖采到的是哪一类，straight-through 给出的恰好是精确梯度。
下游是非线性函数时，$g$ 随样本变化，straight-through 的期望一般不等于真实梯度，是有偏的。

</details>

### 第 8 题

单个类别变量，后验 $q = (0.5, 0.5)$，先验 $p = (0.9, 0.1)$。已知 $\ln(0.5 / 0.9) = -0.5878$，$\ln 5 = 1.6094$。

1. 求 $D_{\mathrm{KL}}(q \,\Vert\, p)$。
2. 假设整个 $z_t$ 只有这一个变量，加上 1 nat 的 free bits 之后损失是多少？这时 KL 项有梯度吗？
3. 不加 free bits 时，$\mathcal{L}_{\text{dyn}}$ 对先验 logits 的梯度是多少？

<details><summary>参考答案</summary>

1. KL：

$$
D_{\mathrm{KL}}(q \,\Vert\, p) = 0.5 \times (-0.5878) + 0.5 \times 1.6094 = -0.2939 + 0.8047 = 0.5108 \text{ nat}
$$

2. $\max(1, 0.5108) = 1$。损失是常数 1，KL 项没有梯度。世界模型在这一步只优化预测损失。

3. $\mathcal{L}_{\text{dyn}}$ 把后验当常数，对先验 logits 的梯度是 $p - q$：

$$
(0.9 - 0.5, \;\; 0.1 - 0.5) = (0.4, \; -0.4)
$$

梯度下降会压低第 1 类的 logit、抬高第 2 类，先验向后验靠拢。

</details>

### 第 9 题

写出 KL balancing 的两项，标出 stop-gradient，并说明：这两项的数值相同，为什么还要分开写？

<details><summary>参考答案</summary>

$$
\mathcal{L}_{\text{dyn}} = D_{\mathrm{KL}}\big( \mathrm{sg}(q_\theta) \,\Vert\, p_\theta \big), \qquad
\mathcal{L}_{\text{rep}} = D_{\mathrm{KL}}\big( q_\theta \,\Vert\, \mathrm{sg}(p_\theta) \big)
$$

sg 不改变前向的数值，所以两项都等于 $D_{\mathrm{KL}}(q \,\Vert\, p)$。但梯度不同：
$\mathcal{L}_{\text{dyn}}$ 的梯度只流向先验（先验去追后验），$\mathcal{L}_{\text{rep}}$ 的梯度只流向后验（后验迁就先验）。

分开写是为了给两个方向不同的权重。先验一侧权重大（V2 的 0.8，V3 的 $\beta_{\text{dyn}}$），后验一侧权重小（V2 的 0.2，V3 的 $\beta_{\text{rep}} = 0.1$）。
这样主要由先验来适应表征，表征不会被拉向一个还没学好的先验。

</details>

### 第 10 题

$K = 2$。后验饱和了，概率是 $(0, 1)$；先验也饱和了，概率是 $(1, 0)$。不做 unimix 时 KL 是多少？
做 1% 的 unimix 之后呢？已知 $\ln 199 = 5.2933$。

<details><summary>参考答案</summary>

不做 unimix：后验在第 2 类上的概率是 1，先验在第 2 类上的概率是 0，$1 \times \ln(1/0)$ 是无穷大。

做 unimix：$p'_k = 0.99 \, p_k + 0.01 / 2$，所以 $q' = (0.005, 0.995)$，$p' = (0.995, 0.005)$。

$$
D_{\mathrm{KL}}(q' \,\Vert\, p') = 0.005 \ln \frac{0.005}{0.995} + 0.995 \ln \frac{0.995}{0.005} = (0.995 - 0.005) \times \ln 199 = 0.99 \times 5.2933 = 5.24 \text{ nat}
$$

和通用公式 $0.99 \ln(1 + 99K)$ 在 $K = 2$ 时的结果一致。unimix 给 KL 加了一个上界。

</details>

### 第 11 题

计算 $\mathrm{symlog}(-99)$，并验证 $\mathrm{symexp}$ 能把它还原。已知 $\ln 100 = 4.605$。
symlog 相比"直接取对数"和"把奖励截断到 $[-1, 1]$"，各有什么好处？

<details><summary>参考答案</summary>

$$
\mathrm{symlog}(-99) = -\ln(99 + 1) = -4.605
$$

$$
\mathrm{symexp}(-4.605) = -\big( e^{4.605} - 1 \big) = -(100 - 1) = -99
$$

相比直接取对数：symlog 对 0 和负数有定义，而且在 0 附近近似恒等映射，小数值不失真。

相比截断：symlog 可逆，不丢信息。截断之后奖励 1 和奖励 100 无法区分，等于改了任务。

</details>

### 第 12 题

桶的位置是 $b = (0, 1, 2, 3)$（为了好算，不做 symlog）。目标值 $y = 1.3$。

1. 写出 two-hot 编码，并验证它的期望等于 1.3。
2. 网络当前输出均匀分布 $(0.25, 0.25, 0.25, 0.25)$。求交叉熵损失对 logits 的梯度。
3. 如果目标值换成 1300（并且桶的范围足够大），梯度的量级会怎么变？

<details><summary>参考答案</summary>

1. $y$ 落在 $b_2 = 1$ 和 $b_3 = 2$ 之间。权重分别是 $(2 - 1.3) / 1 = 0.7$ 和 $(1.3 - 1) / 1 = 0.3$：

$$
\mathrm{twohot}(1.3) = (0, \; 0.7, \; 0.3, \; 0), \qquad 0.7 \times 1 + 0.3 \times 2 = 1.3
$$

2. 梯度是"预测减目标"：

$$
(0.25, \; 0.25, \; 0.25, \; 0.25) - (0, \; 0.7, \; 0.3, \; 0) = (0.25, \; -0.45, \; -0.05, \; 0.25)
$$

3. 不变。梯度的每个分量都是两个概率之差，在 $[-1, 1]$ 之间，和目标值的大小无关。
   这是 two-hot 回归相对平方误差的主要好处：平方误差的梯度正比于"预测减目标"，目标大 1000 倍，梯度也大 1000 倍。

</details>

### 第 13 题

回报归一化。当前 $S = 10$。这一批想象回报的 5% 分位数是 $-1$，95% 分位数是 19。

1. 更新后的 $S$ 是多少？
2. 某一步 $R^\lambda = 15$，$V_\psi = 10$，归一化之后的优势是多少？
3. 另一个稀疏奖励任务里 $S = 0.05$，某一步 $R^\lambda - V_\psi = 0.02$。归一化之后的优势是多少？如果没有 $\max(1, \cdot)$ 会怎样？

<details><summary>参考答案</summary>

1. 这一批的范围是 $19 - (-1) = 20$：

$$
S \leftarrow 0.99 \times 10 + 0.01 \times 20 = 10.1
$$

2. $A = (15 - 10) / \max(1, 10.1) = 5 / 10.1 = 0.495$

3. $A = 0.02 / \max(1, 0.05) = 0.02 / 1 = 0.02$。

   没有下限的话是 $0.02 / 0.05 = 0.4$，放大了 20 倍。稀疏奖励下这种很小的优势多半来自 critic 的拟合误差，
   放大之后 actor 会去追逐噪声，同时把熵正则压过去，过早停止探索。

</details>

### 第 14 题

凭记忆写出 Dreamer V3 世界模型的损失，标出其中的 stop-gradient，并说明预测损失对表征的梯度为什么不停止。

<details><summary>参考答案</summary>

$$
\mathcal{L}(\theta) = \mathbb{E}_{q_\theta}\Big[ \sum_t \big( \beta_{\text{pred}} \mathcal{L}_{\text{pred}} + \beta_{\text{dyn}} \mathcal{L}_{\text{dyn}} + \beta_{\text{rep}} \mathcal{L}_{\text{rep}} \big) \Big]
$$

$$
\begin{aligned}
\mathcal{L}_{\text{pred}} &= -\ln p_\theta(o_t \mid h_t, z_t) - \ln p_\theta(r_{t-1} \mid h_t, z_t) - \ln p_\theta(c_t \mid h_t, z_t) \\
\mathcal{L}_{\text{dyn}} &= \max\Big( 1, \; D_{\mathrm{KL}}\big( \mathrm{sg}(q_\theta(z_t \mid h_t, o_t)) \,\Vert\, p_\theta(z_t \mid h_t) \big) \Big) \\
\mathcal{L}_{\text{rep}} &= \max\Big( 1, \; D_{\mathrm{KL}}\big( q_\theta(z_t \mid h_t, o_t) \,\Vert\, \mathrm{sg}(p_\theta(z_t \mid h_t)) \big) \Big)
\end{aligned}
$$

显式的 sg 有两个：$\mathcal{L}_{\text{dyn}}$ 里包住后验（只训练先验），$\mathcal{L}_{\text{rep}}$ 里包住先验（只训练后验）。
权重 $\beta_{\text{pred}} = 1$、$\beta_{\text{rep}} = 0.1$，$\beta_{\text{dyn}}$ 是 0.5 或 1（随论文版本）。

预测损失对 $z_t$、$h_t$ 不停止梯度，因为表征正是靠重建、奖励和 continue 的预测误差学出来的。
停掉的话，编码器只剩 $\mathcal{L}_{\text{rep}}$ 的梯度，它的最优解是输出一个和观测无关的、最好猜的分布，也就是坍塌。

</details>

### 第 15 题

写出 V3 的 actor 损失（REINFORCE 形式），指出其中的三个 stop-gradient。
然后回答：V1 的 actor 损失和它相比，梯度的路径有什么根本不同？

<details><summary>参考答案</summary>

$$
\mathcal{L}_{\text{actor}}(\phi) = \mathbb{E}\Big[ \sum_{k=0}^{H-1} \mathrm{sg}(w_k) \Big( -\mathrm{sg}(A_k) \ln \pi_\phi\big( \hat a_k \mid \mathrm{sg}(\hat m_k) \big) - \eta \, \mathcal{H}\big[ \pi_\phi(\,\cdot \mid \mathrm{sg}(\hat m_k)) \big] \Big) \Big],
\qquad A_k = \frac{R^\lambda_k - V_\psi(\hat m_k)}{\max(1, S)}
$$

三个 sg：优势 $A_k$（它只是 score-function 估计器里的权重）；输入状态 $\hat m_k$（梯度不穿过世界模型）；权重 $w_k$。

根本不同：V3 的梯度只经过 $\ln \pi_\phi$，即"这个动作的对数概率对 actor 参数的导数"，回报只作为一个数字乘在上面。
世界模型和 critic 不在梯度的路径上。V1 的梯度从回报出发，穿过 critic、奖励头、世界模型的转移函数和重参数化的动作采样，一路回到 actor。
前者不要求任何东西可导（除了 $\ln \pi$），后者要求整条链都可导。

</details>

### 第 16 题

训练一个 Dreamer，观察到：重建画面清晰，未截断的 KL 稳定在 3 nat 左右；
想象的回报持续上升，评估回合的真实回报却停在低位，两者的差距越来越大。
最可能是哪里出了问题？按什么顺序排查？

<details><summary>参考答案</summary>

世界模型的重建和 KL 看起来健康，问题多半在"想象和现实对不上"。按从上游到下游的顺序：

1. 奖励头：只在奖励非零的时间步上统计预测误差；检查奖励和状态是否错开了一位。重建清晰不代表奖励预测得准。
2. continue 头与 λ-return：终止是否被正确处理，价值有没有因为漏掉 continue 而虚高。
3. stop-gradient：想象的起点是否 detach；actor 的损失有没有改到世界模型（检查梯度）。
4. 价值校准：用当前策略新跑几个回合，比较 $V_\psi(m_t)$ 和真实的折扣回报，看偏差出现在哪些状态上。
5. 模型被钻空子：训练步数与环境步数的比例是否太高，actor 在一个更新不够快的模型里过度优化。可以降低比例，或者缩短想象视野。

想象回报略高于真实回报是正常的（对带噪声的估计做最大化必然偏高）；差距持续扩大才是问题。

</details>

### 第 17 题

用第 11.4 节的"五个插槽"描述 TD-MPC2 和 DIAMOND 各自换掉了 Dreamer 的哪些部件，以及各自这么换的动机。

<details><summary>参考答案</summary>

TD-MPC2 换掉了"学习信号"和"怎么用模型"：不重建观测，表征靠奖励、价值和隐状态自洽的损失学出来；
决策时用 MPPI 在隐空间里规划，策略作为规划的先验，价值函数补上视野之外的回报。
动机是不想把模型容量花在和任务无关的像素细节上，并且让决策时还能做前瞻。

DIAMOND 换掉了"表征""序列模型"和"学习信号"：不把画面压成隐变量，用扩散模型直接在像素空间生成下一帧；
"在想象中训练 actor-critic"这一点保留。动机是压缩成离散或低维的隐变量会丢掉视觉细节，而有些细节对任务是决定性的。

两者方向相反：一个认为重建是多余的，一个认为 Dreamer 重建得还不够细。这个分歧贯穿课程的第三部分。

</details>

---

## 16. 小结与下章预告

### 16.1 本章要点

1. **为什么学行为。** PlaNet 每步规划，慢、看不远、不积累。Dreamer 把规划的结果摊销进 actor 和 critic：
   决策只要一次前向计算，critic 概括视野之外的回报。按 V3 的折扣因子算，想象的 15 步之外的部分占了折扣权重的九成以上。
2. **怎么想象。** 从回放里每一个后验状态出发，只用先验往前推 $H$ 步，全程在隐空间里。
   起点停止梯度；终止之后的状态用 continue 的累积乘积降权。
3. **V1。** λ-return 当价值目标；actor 的梯度沿可微的世界模型反传。低方差，但要求动作可重参数化，而且有雅可比连乘的问题。
4. **V2。** 高斯隐变量换成 $32 \times 32$ 的类别隐变量，用 straight-through 传梯度；KL 拆成两半，让先验多迁就后验；
   离散动作的 actor 用 REINFORCE。
5. **V3。** 一组让数值尺度不再要紧的技巧：symlog 压缩输入和目标；two-hot 把回归变成分类，梯度有界；
   回报按分位数范围归一化，只缩不放；unimix 给概率垫底；critic 用 EMA 正则稳住。目标是一套超参数通吃。
6. **全部损失。** 世界模型：$\mathcal{L}_{\text{pred}} + \mathcal{L}_{\text{dyn}} + \mathcal{L}_{\text{rep}}$，两个 sg 都在 KL 里。
   critic：two-hot 交叉熵拟合 $\mathrm{sg}(R^\lambda)$。actor：$-\mathrm{sg}(A) \ln \pi$ 加熵正则。第 9.5 节的清单列了十处 sg。
7. **怎么调试。** 从上游到下游：世界模型、想象、critic、actor。最有用的单个指标是价值校准。

### 16.2 一张图回顾

```
      问题                           本章的办法                         留下的问题
  ───────────────────────────────────────────────────────────────────────────────────
   每步规划太慢、看不远      ──▶   在想象中训练 actor 和 critic     ──▶   决策时不再前瞻，
                                                                      能不能两者都要？（Phase 7）

   怎么得到行为的梯度        ──▶   沿模型反传；REINFORCE            ──▶   哪种更好取决于任务

   未来是多峰的             ──▶   类别隐变量 + straight-through    ──▶   每帧一个隐变量够不够？
                                                                      （Phase 8 的 token）

   表征被 KL 压垮           ──▶   KL balancing + free bits        ──▶   重建把容量花在无关细节上
                                                                      （Phase 7、Phase 10）

   数值尺度随任务变          ──▶   symlog、two-hot、回报归一化       ──▶   隐空间的想象不够逼真
                                                                      （Phase 9 的扩散模型）

   自举目标不稳定           ──▶   目标网络；EMA 正则               ──▶   GRU 的记忆有限
                                                                      （Phase 8 的 Transformer）

   探索只靠策略的随机性      ──▶   熵正则；选读 Plan2Explore        ──▶   没有显式的认知不确定性
```

### 16.3 下章预告

Dreamer 有两个"理所当然"的设计，Phase 7 把它们都推翻了。

**第一个：世界模型必须重建观测。** 重建损失对每个像素一视同仁，模型的容量大半花在背景上。
对控制要紧的东西如果在画面上很小，就容易被忽略。[Phase 7](../phase-07-latent-planning/CONCEPT.md) 的问题是：
能不能不重建，只要求隐状态能预测奖励和价值？这条路线的两个代表是 MuZero 和 TD-MPC2。

**第二个：学了策略就不再规划。** Dreamer 决策时只做一次前向计算，不看未来。
策略有误差的时候，没有任何机制在决策的那一刻纠正它。MuZero 和 TD-MPC2 的回答是两者都要：
学一个策略和一个价值函数，决策时再用模型往前搜索几步。策略给搜索提供先验，价值给搜索补上视野之外的回报。
Phase 1 的 MPPI 会在 TD-MPC2 里回来，这次它有了一个价值函数做终端。

本章的东西在下一章几乎都用得上：λ-return 换成了多步 TD 目标，two-hot 回归被 TD-MPC2 原样采用，
"梯度该流到哪里、不该流到哪里"仍然是读懂损失函数的钥匙。排行榜也会继续：TD-MPC2 加入之后，
"在想象中学策略"和"决策时规划"可以在同一批任务上直接比较。

---

## 附录 A：术语表（中英对照）

| 中文 | 英文 | 一句话解释 |
| --- | --- | --- |
| 想象 | imagination | 不和真实环境交互，在世界模型里展开轨迹 |
| 想象轨迹 | imagined trajectory / rollout | 从某个模型状态出发，用先验和 actor 推演出的一段状态、动作、奖励序列 |
| 想象视野 | imagination horizon | 一次想象往前推的步数 $H$ |
| 模型状态 | model state | RSSM 的状态 $(h_t, z_t)$，actor、critic 和各预测头的输入 |
| 后验 / 先验 | posterior / prior | 看了当前观测之后对 $z_t$ 的推断 / 只凭历史对 $z_t$ 的预测 |
| 摊销 | amortization | 把每次决策时都要做的搜索，提前学进一个网络里 |
| 行为学习 | behavior learning | Dreamer 里训练 actor 和 critic 的环节 |
| actor / critic | actor / critic | 策略网络 / 价值网络 |
| 自举 | bootstrapping | 用价值估计本身来构造价值的学习目标 |
| λ-return | λ-return | 各种 $n$ 步回报按几何权重的加权平均 |
| 沿模型反向传播 | backpropagation through the model / dynamics backpropagation | 把可微的世界模型当成计算图的一部分，让回报的梯度流回策略 |
| pathwise 估计器（重参数化） | pathwise / reparameterization gradient | 把采样写成"确定性函数 + 外部噪声"，对函数求导 |
| score-function 估计器 | score-function estimator / REINFORCE | 用 $\nabla \ln \pi$ 乘以回报来估计梯度，不要求回报可导 |
| straight-through 估计器 | straight-through estimator | 前向用离散样本，反向借用连续松弛的梯度 |
| 替身损失 | surrogate loss | 一个专门构造的损失，它的梯度等于想要的梯度估计 |
| 类别隐变量 | categorical latent | 由若干个类别分布组成的离散随机隐变量 |
| 停止梯度 | stop-gradient | 前向照常计算，反向时当作常数 |
| KL balancing | KL balancing | 把 KL 拆成只更新先验和只更新后验的两项，给不同的权重 |
| 动力学损失 | dynamics loss | $D_{\mathrm{KL}}(\mathrm{sg}(q) \,\Vert\, p)$：训练先验去预测表征 |
| 表征损失 | representation loss | $D_{\mathrm{KL}}(q \,\Vert\, \mathrm{sg}(p))$：让表征更可预测 |
| 预测损失 | prediction loss | 重建、奖励、continue 三项负对数似然之和 |
| free bits（free nats） | free bits / free nats | KL 低于阈值时不再惩罚 |
| 后验坍塌 | posterior collapse | 后验等于先验，隐变量不携带观测的信息 |
| unimix | uniform mixture | 把类别分布和均匀分布按 99 : 1 混合 |
| 对称对数变换 | symlog / symexp | 零附近线性、远处对数的可逆变换，及其逆变换 |
| two-hot 编码 | two-hot encoding | 把标量按距离分摊到相邻两个桶上 |
| 离散回归 | discrete regression | 把回归做成对桶的分类，用交叉熵训练 |
| 桶 | bin / bucket | 离散回归里预先固定的取值点 |
| 回报归一化 | return normalization | 用回报的分位数范围缩放 actor 损失里的优势 |
| 分位数 | percentile | 把数据从小到大排，位于某个百分比位置上的值 |
| 熵正则 | entropy regularization | 在 actor 损失里奖励策略的熵，保持探索 |
| 指数滑动平均 | exponential moving average (EMA) | 按几何衰减的权重对历史取平均 |
| 目标网络 | target network | 一份慢更新的网络副本，专门用来计算学习目标 |
| continue 预测器 | continue predictor | 预测回合在这一步之后是否继续 |
| 价值校准 | value calibration | critic 的预测和真实折扣回报是否一致 |
| 消融实验 | ablation study | 去掉某个部件，看性能变化，判断它的作用 |
| 标准误 | standard error | 样本均值的标准差 |
| 训练比例 | train ratio / replay ratio | 每一步真实交互对应多少训练量 |

---

## 附录 B：延伸阅读

**必读**

- Hafner et al. 2020, Dreamer V1（arXiv:1912.01603）。先读第 3 节（行为学习）。配合本章第 2、3 节。
- Hafner et al. 2023, Dreamer V3（arXiv:2301.04104；Nature 640, 2025）。读方法部分和超参数表。配合本章第 6–9 节。
  注意自己读的是哪个版本。

**精读其中一部分**

- Hafner et al. 2021, Dreamer V2（arXiv:2010.02193）。读世界模型一节（类别隐变量、KL balancing 的伪代码）和 actor 损失一式。配合本章第 4、5 节。

**选读**

- Wu et al. 2022, DayDreamer（arXiv:2206.14176）。Dreamer 在真实机器人上。Phase 11 之前读。
- Sekar et al. 2020, Plan2Explore（arXiv:2005.05960）。用世界模型的不确定性驱动探索。
- Hafner et al. 2022, Director（arXiv:2206.04114）。在世界模型上做分层规划。

**回看**

- Hafner et al. 2019, PlaNet（arXiv:1811.04551）。Dreamer 的世界模型来自这里，对照着读能看清"换掉的只是怎么用模型"。
- Heess et al. 2015, SVG（arXiv:1510.09142）。沿学到的模型反传价值梯度的出处。
- Bengio et al. 2013（arXiv:1308.3432）。straight-through 估计器的出处。
- Bellemare et al. 2017, C51（arXiv:1707.06887）。把价值预测成桶上的分布，two-hot 回归的近亲。

**代码**

- Dreamer V3 的官方实现（作者 Danijar Hafner 的 GitHub 仓库 `dreamerv3`）。本章标了"待核对"的实现细节，
  以那里的代码和配置文件为准。读的时候先看配置文件里的损失权重，再看世界模型和 actor-critic 的损失函数。

**预习**

- Hansen et al. 2024, TD-MPC2（arXiv:2310.16828）。先读引言和方法概览，带着一个问题：它相对 Dreamer 换掉了哪个部件。Phase 7 精读。
- Schrittwieser et al. 2020, MuZero（arXiv:1911.08265）。读引言，了解"不重建观测"的动机。
