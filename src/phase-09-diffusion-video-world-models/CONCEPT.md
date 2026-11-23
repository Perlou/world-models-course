# 扩散与交互式视频世界模型：从去噪到可以操控的梦境

> Phase 9 · 概念精讲 ｜ 对应讲义：`src/phase-09-diffusion-video-world-models/` ｜ 预计阅读：7–9 小时
>
> Phase 8 把画面压成离散 token 再预测，细节在压缩那一步就丢了。这一章换一条路：用扩散模型直接生成连续的画面。
> 全章回答三件事：怎样让扩散模型快到能当世界模型用；怎样让它一帧接一帧地生成而不跑偏；
> 怎样只看视频、不看动作标签，就学出一个可以操控的世界。

---

## 目录

- [0. 本章导读](#0-本章导读)
- [1. 从 DDPM 到 EDM](#1-从-ddpm-到-edm)
- [2. Flow matching](#2-flow-matching)
- [3. 以动作为条件的下一帧扩散](#3-以动作为条件的下一帧扩散)
- [4. DIAMOND：扩散世界模型](#4-diamond扩散世界模型)
- [5. 自回归漂移与 Diffusion Forcing](#5-自回归漂移与-diffusion-forcing)
- [6. 视频 tokenizer 与隐空间扩散](#6-视频-tokenizer-与隐空间扩散)
- [7. 隐动作模型：从无标注视频里学动作](#7-隐动作模型从无标注视频里学动作)
- [8. Genie：生成式交互环境](#8-genie生成式交互环境)
- [9. 实时推理：少步采样与蒸馏](#9-实时推理少步采样与蒸馏)
- [10. Dreamer 4 与视频基础模型](#10-dreamer-4-与视频基础模型)
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

1. 旧课的 DDPM 生成一张图要走上千步。世界模型每想象一帧就要采样一次，怎样把步数降到个位数还不崩？
2. 网络到底该预测噪声、预测干净画面，还是预测 score 或速度？这几种说法是什么关系？
3. 扩散世界模型一帧接一帧地生成，为什么越往后越糊、越走越偏？有哪些办法治？
4. 一段没有任何动作标签的视频，凭什么能学出"按这个键画面会怎么变"？
5. 能生成逼真视频的大模型，离"世界模型"还差什么？

### 0.2 一图总览

```
   Phase 8：画面 ──▶ 离散 token ──▶ Transformer 预测下一帧的 token
                        │
                        │  离散化丢细节，而有些细节决定任务成败
                        ▼
  ┌──────────────── 第一段：扩散世界模型 ─────────────────────────────┐
  │                                                                  │
  │  第 1 节 DDPM → EDM        第 2 节 flow matching                  │  怎样又快又稳地
  │        └──────────┬───────────────┘                              │  生成一张图
  │                   ▼                                              │
  │  第 3 节 以动作为条件的下一帧扩散   p(o_{t+1} | 过去几帧, 动作)       │  把"一张图"变成
  │                   ▼                                              │  "下一帧"
  │  第 4 节 DIAMOND：在扩散的梦里训练 actor-critic                    │
  │                   ▼                                              │
  │  第 5 节 自回归漂移；噪声增强；Diffusion Forcing                    │  一帧接一帧为什么会坏
  └───────────────────┬──────────────────────────────────────────────┘
                      ▼
  ┌──────────────── 第二段：从无标注视频里学可控的世界 ─────────────────┐
  │  第 6 节 视频 tokenizer 与隐空间扩散      先压缩，再建模             │
  │  第 7 节 隐动作模型                      没有动作标签，自己发明动作   │
  │  第 8 节 Genie = 视频 tokenizer + 隐动作模型 + 动力学模型（MaskGIT） │
  └───────────────────┬──────────────────────────────────────────────┘
                      ▼
  ┌──────────────── 第三段：走向实时和大规模 ──────────────────────────┐
  │  第 9 节 少步采样、蒸馏、因果化            让模型跟得上人的操作        │
  │  第 10 节 Dreamer 4；视频基础模型的现状与争议                        │
  └───────────────────┬──────────────────────────────────────────────┘
                      ▼
   Phase 10：非要生成每个像素吗？（JEPA）      Phase 11：这些模型在机器人上怎么用
```

### 0.3 前置知识检查

| 需要的知识 | 在哪学过 | 本章哪里用 |
| --- | --- | --- |
| DDPM：前向加噪、预测噪声的损失、逐步去噪采样 | `deep-learning-course` 的扩散模型一章 | 第 1 节只回顾并对齐记号，不重讲 |
| U-Net、Transformer、VAE | `deep-learning-course` | 第 3、6 节的网络骨干与压缩 |
| 复合误差的界、分布偏移 | [Phase 1 概念文档](../phase-01-dynamics-and-planning/CONCEPT.md) 第 5 节 | 第 5 节：自回归漂移是同一件事 |
| CEM / MPPI 的计算量账 | Phase 1 概念文档第 6.5、7 节 | 第 3 节：扩散模型配规划器为什么贵 |
| λ-return、TD 自举 | Phase 4 | 第 4 节的价值目标；第 9 节 shortcut 模型的自举目标 |
| score-function（REINFORCE）梯度估计器、actor-critic | Phase 5 | 第 4 节：DIAMOND 的策略学习 |
| 在想象中训练策略 | [Phase 6](../phase-06-dreamer/CONCEPT.md) 的 Dreamer | 第 4、10 节 |
| VQ 损失、straight-through、MaskGIT、KV cache | [Phase 8](../phase-08-transformer-world-models/CONCEPT.md) | 第 6、7、8、9 节 |

如果 DDPM 的前向过程公式已经写不出来，先回看旧课那一章再读第 1 节。
高斯分布的对数密度、条件期望这两样会反复用到。

### 0.4 概念与讲义对照表

| 概念文档 | 对应讲义 | 讲义里做什么 |
| --- | --- | --- |
| 第 1 节 从 DDPM 到 EDM | `01-ddpm-to-edm.py` | 在二维玩具数据和小图像上实现 EDM 的预条件、损失、Euler / Heun 采样，对比不同步数 |
| 第 2 节 Flow matching | `02-flow-matching.py` | 实现条件 flow matching 损失和 Euler 采样；与 EDM 互相换算；做一次 reflow |
| 第 3 节 以动作为条件的下一帧扩散 | `03-action-conditioned-frame-diffusion.py` | 以过去几帧和动作为条件，训练下一帧的去噪器，做开环展开 |
| 第 4 节 DIAMOND | `04-diamond.py` | 扩散世界模型 + 奖励 / 终止预测 + 在想象中训练 actor-critic |
| 第 5 节 自回归漂移与 Diffusion Forcing | `05-autoregressive-drift-and-diffusion-forcing.py` | 测量漂移曲线；实现噪声增强和每帧独立噪声水平的训练 |
| 第 6 节 视频 tokenizer 与隐空间扩散 | `06-video-tokenizers-and-latent-diffusion.py` | 训练小型视频自编码器，在隐空间里做第 3 节的事 |
| 第 7 节 隐动作模型 | `07-latent-action-models.py` | 只用画面训练 VQ 瓶颈的隐动作模型，检查隐动作与真实动作的对应 |
| 第 8 节 Genie | `08-mini-genie.py` | 把 tokenizer、隐动作模型、MaskGIT 动力学模型组装成可以用键盘操控的精简版 Genie |
| 第 9 节 实时推理 | `09-realtime-few-step-inference.py` | 测每帧延迟的构成；实现 shortcut 模型的自洽损失；因果缓存 |
| 第 10 节 Dreamer 4 与视频基础模型 | `10-dreamer4-and-video-foundation-models.py` | 玩具版 shortcut forcing；把本章几类世界模型放在同一份数据上对比 |

讲义都还没写。下文"代码对应"里的函数名是规划中的名字，以最终讲义为准。

### 0.5 符号表

本章沿用全课程的符号约定。扩散模型的论文习惯用 $t$ 表示扩散时间，
而本课的 $t$ 是环境的时间步，$\tau$ 是轨迹。为了不冲突，本章**不用 $t$ 和 $\tau$ 表示扩散时间**，
改用下表的记号，并在第三列写明各篇论文的原始写法。

| 符号 | 含义 | 论文里的写法 |
| --- | --- | --- |
| $x$ | 干净样本（一张图，或下一帧）。维度记为 $n$ | DDPM 记作 $x_0$；flow matching 记作 $x_1$；EDM 记作 $y$ 或 $x_0$ |
| $\epsilon$ | 标准高斯噪声，$\epsilon \sim \mathcal{N}(0, I)$ | flow matching 把它记作 $x_0$ |
| $\sigma$ | 噪声水平：加到样本上的高斯噪声的标准差 | EDM 同；DDPM 里对应 $\sqrt{(1 - \bar\alpha_t)/\bar\alpha_t}$ |
| $x_\sigma$ | 噪声水平为 $\sigma$ 的带噪样本，$x_\sigma = x + \sigma\epsilon$ | EDM 记作 $x$ |
| $k$，$K$ | DDPM 的离散扩散步编号与总步数 | DDPM 记作 $t$，$T$ |
| $\beta_k$，$\bar\alpha_k$ | DDPM 的噪声日程；$\bar\alpha_k = \prod_{j \le k} (1 - \beta_j)$ | 同 |
| $x_k$ | DDPM 第 $k$ 步的带噪样本 | DDPM 记作 $x_t$ |
| $u$ | flow matching 的路径时间，$u \in [0, 1]$。$u = 0$ 是纯噪声，$u = 1$ 是数据 | Lipman et al. 与 shortcut 模型记作 $t$；Dreamer 4 记作 $\tau$（待核对） |
| $x_u$ | 路径上的点，$x_u = (1 - u)\,\epsilon + u\,x$ | 论文写成 $x_t = (1 - t)\,x_0 + t\,x_1$ |
| $d$ | shortcut 模型的步长（一步跨过多长的路径时间） | 同 |
| $\bar{v}_\theta(x_u, u, d)$ | shortcut 网络：从 $u$ 走到 $u + d$ 这一段的平均速度 | Frans et al. 记作 $s_\theta(x_t, t, d)$；本课的 $s$ 已用于状态 |
| $p_\sigma(x_\sigma)$ | 带噪样本的边际分布 | $p(x; \sigma)$ |
| $g_\theta$ | score 网络，近似 $\nabla_{x_\sigma} \log p_\sigma(x_\sigma)$ | 论文多记作 $s_\theta$；本课的 $s$ 已用于状态 |
| $\epsilon_\theta$，$D_\theta$，$v_\theta$ | 预测噪声的网络；去噪器（预测干净样本）；速度网络 | 同 |
| $F_\theta$ | EDM 里真正的神经网络（被预条件系数包起来的那一层） | 同 |
| $c_{\text{in}}, c_{\text{out}}, c_{\text{skip}}, c_{\text{noise}}$ | EDM 的四个预条件系数，都是 $\sigma$ 的函数 | 同 |
| $\sigma_{\text{data}}$ | 数据每个分量的标准差 | 同 |
| $\lambda(\sigma)$ | 损失在不同噪声水平上的权重。注意它不是 λ-return 的 $\lambda$ | 同 |
| $N$ | 采样时的去噪步数。每步调用一次网络时，等于网络调用次数（NFE） | EDM 同；DIAMOND 记作 $n$ |
| $m$ | 世界模型看的过去帧数（上下文长度） | DIAMOND 记作 $L$。本章的 $L$ 留给 Lipschitz 常数 |
| $\tilde{o}_t$ | 带噪的第 $t$ 帧。从第 3 节起，下标让给时间步，噪声水平单独写出 | Diffusion Forcing 记作 $x_t^{k_t}$ |
| $\sigma_t$ | Diffusion Forcing 里第 $t$ 帧自己的噪声水平 | Diffusion Forcing 记作 $k_t$ |
| $z_t$ | 第 $t$ 帧经 tokenizer 得到的表示：连续隐变量或离散 token | 同 |
| $\bar{a}_t$，$M$ | 隐动作（从视频里学出来的离散码）；隐动作码本的大小 | Genie 记作 $\tilde{a}_t$ 与 $\lvert A \rvert$。本章的波浪号已用于带噪样本 |
| $w$ | classifier-free guidance 的引导强度 | 同 |
| $\varepsilon$，$L$，$e_t$ | 单步误差上界、Lipschitz 常数、第 $t$ 步的累计误差 | 沿用 Phase 1 第 5 节 |
| `[B, T, C, H, W]` | 视频张量：批大小、帧数、通道、高、宽 | |

两个容易混的地方：

- $\epsilon$（噪声）和 $\varepsilon$（Phase 1 的单步误差上界）是两个不同的量，字形也不同。
- $H$ 在公式里是想象 / 规划的视野，在张量形状 `[B, T, C, H, W]` 里是图像高度。看上下文就能分清。

---

## 1. 从 DDPM 到 EDM

### 1.1 直觉

把扩散模型想成一位修图师。给他一张盖满雪花点的照片，让他猜原图。
雪花少的时候，他修的是细节；雪花多到几乎看不见原图时，他只能猜个大致的轮廓。
扩散模型就是一位对**任何雪花强度**都会修的修图师。生成一张新图的办法是：
从一张纯雪花开始，让他修一点，再修一点，直到雪花全没了。

旧课的 DDPM 已经讲过这件事。但 DDPM 有三样东西是绑死在一起的：

- 雪花强度用"第几步"来编号（1 到 1000）
- 修图师的任务固定为"把雪花指出来"（预测噪声）
- 生成时必须按训练时的那 1000 步往回走

对图像生成这只是慢一点。对世界模型这是致命的：智能体在想象里每走一步，
世界模型就要完整地生成一帧。Phase 1 第 6.5 节算过规划的账，那里每一步只是一次 MLP 前向计算；
如果每一步变成 1000 次 U-Net 前向计算，任何用法都不现实。

EDM（Karras et al. 2022，arXiv:2206.00364）做的事，用工程师的话说是一次**重构**：
把一个配置、业务逻辑、调度写在一起的单体脚本，拆成几个可以独立替换的模块。

```
   DDPM（绑在一起）                       EDM（拆开）
  ┌──────────────────────┐          ┌───────────────────────────────┐
  │ 步数编号 k = 1…1000   │          │ ① 噪声水平：直接用标准差 σ       │
  │ 预测噪声 ε            │   ──▶    │ ② 网络参数化：预条件（本节重点）  │
  │ 损失对所有 k 等权      │          │ ③ 训练：σ 怎么采样、损失怎么加权 │
  │ 采样按训练的步往回走    │          │ ④ 采样：用哪些 σ、用什么求解器   │
  └──────────────────────┘          └───────────────────────────────┘
                                       四个模块各自独立，训练和采样解耦
```

拆开之后才能分别回答"少步采样时哪个模块在拖后腿"。DIAMOND 选 EDM 的理由就出在模块 ②（第 4 节）。

### 1.2 DDPM 回顾与记号对齐

DDPM（Ho et al. 2020，arXiv:2006.11239）的前向过程，用本章的记号写：

$$
x_k = \sqrt{\bar\alpha_k}\; x + \sqrt{1 - \bar\alpha_k}\; \epsilon,
\qquad \bar\alpha_k = \prod_{j=1}^{k} (1 - \beta_j),
\qquad \epsilon \sim \mathcal{N}(0, I)
$$

训练目标是让网络从 $x_k$ 里把噪声指出来：

$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{x, \epsilon, k} \big\Vert \epsilon_\theta(x_k, k) - \epsilon \big\Vert^2
$$

| 量 | DDPM 论文 | 本章 |
| --- | --- | --- |
| 干净样本 | $x_0$ | $x$ |
| 带噪样本 | $x_t$ | $x_k$ |
| 扩散步、总步数 | $t$，$T$ | $k$，$K$ |
| 噪声 | $\epsilon$ | $\epsilon$ |

**把 DDPM 换成用 $\sigma$ 描述。** 前向过程两边同除以 $\sqrt{\bar\alpha_k}$：

$$
\frac{x_k}{\sqrt{\bar\alpha_k}} = x + \sigma_k\, \epsilon,
\qquad \sigma_k = \sqrt{\frac{1 - \bar\alpha_k}{\bar\alpha_k}}
$$

右边就是"干净样本加上标准差为 $\sigma_k$ 的噪声"。也就是说，DDPM 的第 $k$ 步，
等价于噪声水平为 $\sigma_k$ 的带噪样本 $x_{\sigma_k} = x + \sigma_k \epsilon$，再整体缩小 $\sqrt{\bar\alpha_k}$ 倍。
缩放只是为了让送进网络的数值范围稳定，不含新的信息。

| $\bar\alpha_k$ | 0.99 | 0.9 | 0.5 | 0.1 | 0.01 |
| --- | --- | --- | --- | --- | --- |
| $\sigma_k$ | 0.10 | 0.33 | 1.00 | 3.00 | 9.95 |

比如 $\bar\alpha_k = 0.1$：$\sigma_k = \sqrt{0.9 / 0.1} = 3$。信号只剩 $\sqrt{0.1} = 0.32$ 倍，
换个说法就是"噪声的标准差是信号的 3 倍"。

DDPM 的标准设置是 $K = 1000$、$\beta$ 从 $10^{-4}$ 线性增加到 $0.02$。可以估一下它的最大噪声水平：

$$
\ln \bar\alpha_K = \sum_{j} \ln(1 - \beta_j) \approx -\sum_j \beta_j - \frac{1}{2} \sum_j \beta_j^2
\approx -10.05 - 0.07 = -10.12
$$

所以 $\bar\alpha_K \approx 4.0 \times 10^{-5}$，$\sigma_K \approx \sqrt{1 / \bar\alpha_K} \approx 157$。
其中 $\sum_j \beta_j = 1000 \times (10^{-4} + 0.02)/2 = 10.05$ 是等差数列求和。

从这里起，本章统一用 $x_\sigma = x + \sigma\epsilon$ 这种写法。它有两个好处：

1. $\sigma$ 有直接的物理含义（噪声的标准差），可以和数据自身的标准差 $\sigma_{\text{data}}$ 比大小。
   后面所有"高噪声 / 低噪声"的讨论，说的都是 $\sigma$ 比 $\sigma_{\text{data}}$ 大还是小。
2. $\sigma$ 是连续的。训练时可以取任意值，采样时可以任选一串递减的 $\sigma$，不必和训练时一一对应。

### 1.3 定义：加噪核、边际分布、score、去噪器

设数据服从 $p_{\text{data}}(x)$。四个定义：

- **加噪核**：给定干净样本，带噪样本的分布。

$$
p_\sigma(x_\sigma \mid x) = \mathcal{N}(x_\sigma;\; x,\; \sigma^2 I)
$$

- **边际分布**：把所有干净样本各自加噪后混在一起的分布。

$$
p_\sigma(x_\sigma) = \int p_{\text{data}}(x)\; p_\sigma(x_\sigma \mid x)\; dx
$$

- **score**：对数密度对样本的梯度 $\nabla_{x_\sigma} \log p_\sigma(x_\sigma)$。
  它是带噪样本空间里每一点上的一个箭头，指向密度升高最快的方向，也就是"往哪边走更像数据"。
- **去噪器**：$D_\theta(x_\sigma; \sigma)$，输入带噪样本和噪声水平，输出对干净样本的估计。

$\sigma \to 0$ 时 $p_\sigma$ 就是数据分布；$\sigma$ 很大时 $p_\sigma$ 接近一个很宽的高斯。
只要知道所有 $\sigma$ 下的 score，就能从宽高斯里的一个随机点出发，顺着箭头一路走回数据分布（第 1.9 节）。
所以问题归结为：**怎么学 score。**

### 1.4 推导：去噪 score matching

最直接的想法是让网络 $g_\theta$ 去回归真实的 score。这叫显式 score matching：

$$
J_{\text{ESM}}(\theta) = \frac{1}{2}\, \mathbb{E}_{x_\sigma \sim p_\sigma}
\Big\Vert g_\theta(x_\sigma, \sigma) - \nabla_{x_\sigma} \log p_\sigma(x_\sigma) \Big\Vert^2
$$

它没法直接用：$p_\sigma$ 里含有未知的 $p_{\text{data}}$，回归的目标本身算不出来。
下面分六步把它变成一个能算的目标。为了简洁，把 $g_\theta(x_\sigma, \sigma)$ 简写成 $g_\theta$，
梯度 $\nabla$ 都是对 $x_\sigma$ 求的。

**第一步：展开平方。** 用 $\Vert a - b \Vert^2 = \Vert a \Vert^2 - 2 a^\top b + \Vert b \Vert^2$。

$$
J_{\text{ESM}} =
\underbrace{\frac{1}{2}\, \mathbb{E}_{x_\sigma} \Vert g_\theta \Vert^2}_{A}
\;-\; \underbrace{\mathbb{E}_{x_\sigma} \big[ g_\theta^\top\, \nabla \log p_\sigma(x_\sigma) \big]}_{B}
\;+\; \underbrace{\frac{1}{2}\, \mathbb{E}_{x_\sigma} \Vert \nabla \log p_\sigma(x_\sigma) \Vert^2}_{C_1,\ \text{与 } \theta \text{ 无关}}
$$

麻烦全在交叉项 $B$ 里。

**第二步：把 $B$ 写成积分，用恒等式 $\nabla \log p = \nabla p / p$ 消掉分母。**

$$
B = \int p_\sigma(x_\sigma)\; g_\theta^\top\, \frac{\nabla p_\sigma(x_\sigma)}{p_\sigma(x_\sigma)}\; dx_\sigma
= \int g_\theta^\top\, \nabla p_\sigma(x_\sigma)\; dx_\sigma
$$

**第三步：代入边际分布的定义，把梯度移到积分号里面，再反过来用一次 $\nabla p = p\, \nabla \log p$。**

$$
\begin{aligned}
\nabla p_\sigma(x_\sigma)
&= \int p_{\text{data}}(x)\; \nabla p_\sigma(x_\sigma \mid x)\; dx \\
&= \int p_{\text{data}}(x)\; p_\sigma(x_\sigma \mid x)\; \nabla \log p_\sigma(x_\sigma \mid x)\; dx
\end{aligned}
$$

这一步是整个推导的关键：边际分布的梯度，被写成了**条件分布**的 score 的加权和。
而条件分布只是一个高斯，它的 score 可以直接算。

**第四步：代回 $B$。** 两重积分正好是对 $(x, x_\sigma)$ 联合分布的期望。

$$
B = \iint p_{\text{data}}(x)\; p_\sigma(x_\sigma \mid x)\; g_\theta^\top\, \nabla \log p_\sigma(x_\sigma \mid x)\; dx\, dx_\sigma
= \mathbb{E}_{x,\, x_\sigma} \big[ g_\theta^\top\, \nabla \log p_\sigma(x_\sigma \mid x) \big]
$$

**第五步：配方。** $A$ 只依赖 $x_\sigma$，写成对联合分布的期望不改变它的值。
定义去噪 score matching 目标：

$$
J_{\text{DSM}}(\theta) = \frac{1}{2}\, \mathbb{E}_{x,\, x_\sigma}
\Big\Vert g_\theta(x_\sigma, \sigma) - \nabla \log p_\sigma(x_\sigma \mid x) \Big\Vert^2
= A - B + C_2
$$

其中 $C_2 = \frac{1}{2} \mathbb{E} \Vert \nabla \log p_\sigma(x_\sigma \mid x) \Vert^2$ 也与 $\theta$ 无关。
和第一步对比：

$$
J_{\text{ESM}}(\theta) = J_{\text{DSM}}(\theta) + (C_1 - C_2)
$$

两个目标只差一个常数，对 $\theta$ 的梯度处处相同，最小值点也相同。

**第六步：算出条件 score。** 高斯的对数密度是
$\log p_\sigma(x_\sigma \mid x) = -\Vert x_\sigma - x \Vert^2 / (2\sigma^2) + \text{常数}$，求梯度：

$$
\nabla_{x_\sigma} \log p_\sigma(x_\sigma \mid x) = -\frac{x_\sigma - x}{\sigma^2} = -\frac{\epsilon}{\sigma}
$$

最后一个等号用了 $x_\sigma = x + \sigma\epsilon$。代入得到最终形式：

$$
J_{\text{DSM}}(\theta) = \frac{1}{2}\, \mathbb{E}_{x \sim p_{\text{data}},\ \epsilon \sim \mathcal{N}(0, I)}
\Big\Vert g_\theta(x + \sigma\epsilon,\ \sigma) + \frac{\epsilon}{\sigma} \Big\Vert^2
$$

这个式子里的每一样东西都能算：取一个数据样本，加噪，让网络回归 $-\epsilon/\sigma$。
这个结论出自 Vincent 2011（*A Connection Between Score Matching and Denoising Autoencoders*）。

**一个容易被忽略的细节。** 单个训练样本的回归目标 $-\epsilon/\sigma$ 并不是真实的 score。
同一个 $x_\sigma$ 可以由很多对不同的 $(x, \epsilon)$ 得到，每一对给出的目标都不一样。
平方损失的最优解是这些目标的条件平均，而这个平均恰好等于真实的 score：

$$
g^\star(x_\sigma, \sigma) = \mathbb{E}\Big[ -\frac{\epsilon}{\sigma} \,\Big|\, x_\sigma \Big] = \nabla_{x_\sigma} \log p_\sigma(x_\sigma)
$$

"用一堆带噪声的目标做回归，学到的是它们的条件期望"这个机制，
会在 flow matching（第 2 节）和一步采样为什么发糊（第 3、4 节）里反复出现。

### 1.5 三种预测目标的等价换算

网络可以输出三种东西中的任何一种。它们之间只差一个由 $x_\sigma$ 和 $\sigma$ 决定的线性变换。

**预测噪声。** 令 $g_\theta = -\epsilon_\theta / \sigma$，代入 $J_{\text{DSM}}$ 的被积项：

$$
\Big\Vert -\frac{\epsilon_\theta}{\sigma} + \frac{\epsilon}{\sigma} \Big\Vert^2 = \frac{1}{\sigma^2}\, \Vert \epsilon_\theta - \epsilon \Vert^2
$$

这就是 DDPM 的损失，差一个权重 $1/\sigma^2$。

**预测干净样本。** 由 $x = x_\sigma - \sigma\epsilon$，定义 $D_\theta = x_\sigma - \sigma\epsilon_\theta$，
于是 $g_\theta = -\epsilon_\theta/\sigma = (D_\theta - x_\sigma)/\sigma^2$。代入：

$$
\Big\Vert \frac{D_\theta - x_\sigma}{\sigma^2} + \frac{\epsilon}{\sigma} \Big\Vert^2
= \frac{1}{\sigma^4}\, \Vert D_\theta - (x_\sigma - \sigma\epsilon) \Vert^2
= \frac{1}{\sigma^4}\, \Vert D_\theta - x \Vert^2
$$

汇总成一张换算表：

| 网络输出 | 单个样本的回归目标 | 换算成 score | 换算成干净样本 |
| --- | --- | --- | --- |
| score $g_\theta$ | $-\epsilon/\sigma$ | $g_\theta$ | $x_\sigma + \sigma^2 g_\theta$ |
| 噪声 $\epsilon_\theta$ | $\epsilon$ | $-\epsilon_\theta/\sigma$ | $x_\sigma - \sigma\epsilon_\theta$ |
| 干净样本 $D_\theta$ | $x$ | $(D_\theta - x_\sigma)/\sigma^2$ | $D_\theta$ |

三种损失的关系是：

$$
\Big\Vert g_\theta + \frac{\epsilon}{\sigma} \Big\Vert^2
= \frac{1}{\sigma^2}\, \Vert \epsilon_\theta - \epsilon \Vert^2
= \frac{1}{\sigma^4}\, \Vert D_\theta - x \Vert^2
$$

**"等价"的准确含义。** 在任何一个固定的 $\sigma$ 上，三种写法的最优解可以互相换算，是同一个函数。
不等价的地方有两处：不同 $\sigma$ 之间的相对权重差了 $\sigma^2$、$\sigma^4$ 倍；
网络输出的误差换算到别的量上时，会被 $\sigma$ 的某次幂放大或缩小。
这两处恰好是 EDM 预条件要处理的问题（第 1.7 节）。

**最优去噪器与 Tweedie 公式。** 对 $\Vert D_\theta - x \Vert^2$ 做回归，最优解是条件期望
$D^\star(x_\sigma; \sigma) = \mathbb{E}[x \mid x_\sigma]$。再用上表第一行的换算：

$$
\mathbb{E}[x \mid x_\sigma] = x_\sigma + \sigma^2\, \nabla_{x_\sigma} \log p_\sigma(x_\sigma)
$$

读法：**最优的去噪结果 = 带噪样本 + 沿 score 方向走 $\sigma^2$ 那么远。**
去噪器给出的不是"原图"，而是"所有可能的原图的平均"。噪声越大，可能的原图越多，这个平均越模糊。

### 1.6 手算例子：一维高斯

数据是一维的，服从 $\mathcal{N}(2, 1)$（均值 2，标准差 $\sigma_{\text{data}} = 1$）。取噪声水平 $\sigma = 2$。

**边际分布。** 两个独立高斯相加，方差相加：$p_\sigma = \mathcal{N}(2,\ 1 + 4) = \mathcal{N}(2, 5)$。

**真实的 score。** 高斯 $\mathcal{N}(\mu, \nu^2)$ 的 score 是 $-(x - \mu)/\nu^2$。在 $x_\sigma = 4.5$ 处：

$$
\nabla \log p_\sigma(4.5) = -\frac{4.5 - 2}{5} = -0.5
$$

负号表示箭头指向左边，也就是指向数据均值 2 的方向。

**最优去噪器。** 两个高斯的后验均值公式（先验 $\mathcal{N}(2, 1)$，观测噪声方差 4）：

$$
\mathbb{E}[x \mid x_\sigma = 4.5] = 2 + \frac{1}{1 + 4} \times (4.5 - 2) = 2.5
$$

**用 Tweedie 公式验证。** $x_\sigma + \sigma^2 \times \text{score} = 4.5 + 4 \times (-0.5) = 2.5$。一致。

**最优的噪声预测。** $\epsilon^\star = (x_\sigma - D^\star)/\sigma = (4.5 - 2.5)/2 = 1.0$。
再换算回 score：$-\epsilon^\star/\sigma = -0.5$。一致。

**单个样本的目标是带噪声的。** 下面两对 $(x, \epsilon)$ 都能得到 $x_\sigma = 4.5$：

| 干净样本 $x$ | 噪声 $\epsilon$ | $x_\sigma = x + 2\epsilon$ | 这一对给出的回归目标 $-\epsilon/\sigma$ |
| --- | --- | --- | --- |
| 3.0 | 0.75 | 4.5 | −0.375 |
| 1.5 | 1.5 | 4.5 | −0.75 |

两个目标都不等于 −0.5。网络见过足够多这样的样本对之后，学到的是它们的条件平均，才是 −0.5。

这个例子里噪声的标准差是数据的 2 倍。带噪样本 4.5 离均值有 2.5 那么远，
去噪器只敢把它当成"原图大概在 2.5"，把偏离量缩到了原来的五分之一。
噪声越大，去噪器越保守，越往数据的平均靠。

### 1.7 推导：EDM 的预条件

**问题在哪。** 直接让网络输出 $D_\theta$ 或 $\epsilon_\theta$ 都有毛病，毛病出在不同的 $\sigma$ 上：

| | 输入 $x_\sigma$ 的尺度 | 低噪声（$\sigma \ll \sigma_{\text{data}}$） | 高噪声（$\sigma \gg \sigma_{\text{data}}$） |
| --- | --- | --- | --- |
| 预测干净样本 | 标准差从 $\sigma_{\text{data}}$ 变到上百 | 目标几乎等于输入，网络要在"抄输入"上叠加一个极小的修正 | 合理 |
| 预测噪声 | 同上 | 合理 | 目标几乎等于输入除以 $\sigma$，网络在"抄输入"；换算回干净样本时误差被放大 $\sigma$ 倍 |

Phase 1 第 4.3 节讲过"让网络学变化量、不要学恒等映射"。这里是同一个道理，只是哪种写法在抄输入取决于 $\sigma$。

**EDM 的做法。** 不让神经网络 $F_\theta$ 直接当去噪器，而是用四个只依赖 $\sigma$ 的系数把它包起来：

$$
D_\theta(x_\sigma; \sigma) = c_{\text{skip}}(\sigma)\; x_\sigma + c_{\text{out}}(\sigma)\; F_\theta\big( c_{\text{in}}(\sigma)\; x_\sigma;\ c_{\text{noise}}(\sigma) \big)
$$

```
                    ┌──────────────── c_skip(σ) ─────────────────┐
                    │                                            ▼
   x_σ ─────────────┼─▶ × c_in(σ) ─▶ ┌────────┐ ─▶ × c_out(σ) ─▶ (+) ─▶ D_θ(x_σ; σ)
                    │                │  F_θ   │
   σ ──▶ c_noise(σ) ─────────────────▶└────────┘
```

四个系数各自由一条要求定出来。假设数据每个分量的方差是 $\sigma_{\text{data}}^2$，噪声与数据独立。

**要求一：网络的输入是单位方差。** 输入是 $c_{\text{in}}\, x_\sigma = c_{\text{in}} (x + \sigma\epsilon)$。
独立变量之和的方差等于方差之和：

$$
\mathrm{Var}\big[ c_{\text{in}} (x + \sigma\epsilon) \big] = c_{\text{in}}^2\, (\sigma_{\text{data}}^2 + \sigma^2) = 1
\quad \Longrightarrow \quad
c_{\text{in}}(\sigma) = \frac{1}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}
$$

这和 DDPM 乘上 $\sqrt{\bar\alpha_k}$ 做的是同一件事：$\sigma_{\text{data}} = 1$ 时，
$c_{\text{in}} = 1/\sqrt{1 + \sigma^2} = \sqrt{\bar\alpha_k}$（由第 1.2 节的 $\sigma_k$ 反解可得）。

**要求二：网络的回归目标是单位方差。** 先看 $F_\theta$ 实际在回归什么。把 $D_\theta$ 的定义代进去噪损失：

$$
\begin{aligned}
\lambda(\sigma)\, \big\Vert D_\theta(x_\sigma; \sigma) - x \big\Vert^2
&= \lambda(\sigma)\, \big\Vert c_{\text{skip}}\, x_\sigma + c_{\text{out}}\, F_\theta - x \big\Vert^2 \\
&= \lambda(\sigma)\, c_{\text{out}}^2\, \Big\Vert F_\theta - \underbrace{\frac{1}{c_{\text{out}}} \big( x - c_{\text{skip}}\, x_\sigma \big)}_{F_{\text{target}}} \Big\Vert^2
\end{aligned}
$$

第二行是把 $c_{\text{out}}$ 提到范数外面。所以网络真正的回归目标是 $F_{\text{target}}$。代入 $x_\sigma = x + \sigma\epsilon$：

$$
F_{\text{target}} = \frac{1}{c_{\text{out}}} \Big( (1 - c_{\text{skip}})\, x - c_{\text{skip}}\, \sigma\, \epsilon \Big)
$$

它是干净样本和噪声的一个混合。要求它的方差为 1：

$$
\mathrm{Var}[F_{\text{target}}] = \frac{(1 - c_{\text{skip}})^2\, \sigma_{\text{data}}^2 + c_{\text{skip}}^2\, \sigma^2}{c_{\text{out}}^2} = 1
\quad \Longrightarrow \quad
c_{\text{out}}^2 = (1 - c_{\text{skip}})^2\, \sigma_{\text{data}}^2 + c_{\text{skip}}^2\, \sigma^2
$$

这条要求只给出了 $c_{\text{out}}$ 和 $c_{\text{skip}}$ 的一个关系，还差一个条件。

**要求三：$c_{\text{out}}$ 尽可能小。** 理由：$D_\theta = c_{\text{skip}} x_\sigma + c_{\text{out}} F_\theta$，
网络输出 $F_\theta$ 上的任何误差，到了去噪结果上都被乘以 $c_{\text{out}}$。$c_{\text{out}}$ 越小，网络的错被放大得越少。
把上式右边看成 $c_{\text{skip}}$ 的函数，求导并令其为零：

$$
\frac{\partial\, c_{\text{out}}^2}{\partial\, c_{\text{skip}}} = -2\, (1 - c_{\text{skip}})\, \sigma_{\text{data}}^2 + 2\, c_{\text{skip}}\, \sigma^2 = 0
\quad \Longrightarrow \quad
c_{\text{skip}}(\sigma) = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}
$$

代回去求 $c_{\text{out}}$。先有 $1 - c_{\text{skip}} = \sigma^2 / (\sigma^2 + \sigma_{\text{data}}^2)$，于是：

$$
c_{\text{out}}^2
= \frac{\sigma^4\, \sigma_{\text{data}}^2 + \sigma_{\text{data}}^4\, \sigma^2}{(\sigma^2 + \sigma_{\text{data}}^2)^2}
= \frac{\sigma^2\, \sigma_{\text{data}}^2\, (\sigma^2 + \sigma_{\text{data}}^2)}{(\sigma^2 + \sigma_{\text{data}}^2)^2}
\quad \Longrightarrow \quad
c_{\text{out}}(\sigma) = \frac{\sigma\, \sigma_{\text{data}}}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}
$$

**要求四：各个噪声水平对损失的贡献均等。** 损失里 $F_\theta$ 那一项前面的系数是 $\lambda(\sigma)\, c_{\text{out}}^2$，令它为 1：

$$
\lambda(\sigma) = \frac{1}{c_{\text{out}}^2} = \frac{\sigma^2 + \sigma_{\text{data}}^2}{(\sigma\, \sigma_{\text{data}})^2}
$$

**第四个系数。** $c_{\text{noise}}(\sigma) = \frac{1}{4} \ln \sigma$。它只是把 $\sigma$ 变成一个适合喂给网络的数
（$\sigma$ 跨好几个数量级，取对数后范围就小了）。这一项是经验选择，没有推导。

**把结果代回 $F_{\text{target}}$，看网络到底在学什么。**

$$
F_{\text{target}} = \frac{(\sigma / \sigma_{\text{data}})\; x \;-\; \sigma_{\text{data}}\; \epsilon}{\sqrt{\sigma^2 + \sigma_{\text{data}}^2}}
$$

| 噪声水平 | $c_{\text{skip}}$ | $c_{\text{out}}$ | $F_{\text{target}}$ 约等于 | 网络在做什么 |
| --- | --- | --- | --- | --- |
| $\sigma \ll \sigma_{\text{data}}$ | 趋于 1 | 趋于 $\sigma$ | $-\epsilon$ | 预测噪声。输入几乎就是答案，网络只管那一点点修正 |
| $\sigma = \sigma_{\text{data}}$ | 1/2 | $\sigma_{\text{data}}/\sqrt{2}$ | $(x/\sigma_{\text{data}} - \epsilon)/\sqrt{2}$ | 两者各一半 |
| $\sigma \gg \sigma_{\text{data}}$ | 趋于 0 | 趋于 $\sigma_{\text{data}}$ | $x/\sigma_{\text{data}}$ | 预测干净样本。输入几乎全是噪声，抄它没用 |

预条件的效果是：**网络的任务随噪声水平自动切换，在任何 $\sigma$ 上都不抄输入，
并且输出误差被放大的倍数 $c_{\text{out}}$ 始终不超过 $\min(\sigma, \sigma_{\text{data}})$。**

还有一个值得记住的巧合。第 1.6 节的后验均值公式里，零均值高斯数据的最优去噪器是
$\frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}\, x_\sigma$，系数正好是 $c_{\text{skip}}$。
也就是说，跳连那一路已经是"把数据当成高斯时的最优去噪"，网络 $F_\theta$ 只需要学数据不是高斯的那部分。

### 1.8 手算例子：三个噪声水平下的系数

取 EDM 论文的默认值 $\sigma_{\text{data}} = 0.5$（像素缩放到 $[-1, 1]$ 的图像，标准差大致是这个量级）。

**中等噪声 $\sigma = 0.5$。** $\sigma^2 + \sigma_{\text{data}}^2 = 0.5$，开方 0.7071。

$$
c_{\text{in}} = \frac{1}{0.7071} = 1.414, \quad
c_{\text{skip}} = \frac{0.25}{0.5} = 0.5, \quad
c_{\text{out}} = \frac{0.5 \times 0.5}{0.7071} = 0.354, \quad
\lambda = \frac{1}{0.354^2} = 8
$$

$$
F_{\text{target}} = \frac{0.5\, x - 0.5 \times 0.5\, \epsilon}{0.354} = 1.414\, x - 0.707\, \epsilon
$$

验证方差：$1.414^2 \times 0.25 + 0.707^2 \times 1 = 0.5 + 0.5 = 1$。

**低噪声 $\sigma = 0.02$。** $\sigma^2 + \sigma_{\text{data}}^2 = 0.2504$，开方 0.5004。

$$
c_{\text{in}} = 1.998, \quad
c_{\text{skip}} = \frac{0.25}{0.2504} = 0.9984, \quad
c_{\text{out}} = \frac{0.02 \times 0.5}{0.5004} = 0.0200
$$

$$
F_{\text{target}} = \frac{0.0016\, x - 0.9984 \times 0.02\, \epsilon}{0.0200} = 0.080\, x - 0.999\, \epsilon
$$

几乎就是 $-\epsilon$。方差：$0.080^2 \times 0.25 + 0.999^2 = 0.002 + 0.998 = 1$。

**高噪声 $\sigma = 20$。** $\sigma^2 + \sigma_{\text{data}}^2 = 400.25$，开方 20.006。

$$
c_{\text{in}} = 0.0500, \quad
c_{\text{skip}} = \frac{0.25}{400.25} = 0.000625, \quad
c_{\text{out}} = \frac{20 \times 0.5}{20.006} = 0.4998
$$

$$
F_{\text{target}} = \frac{0.999375\, x - 0.000625 \times 20\, \epsilon}{0.4998} = 1.999\, x - 0.025\, \epsilon
$$

几乎就是 $x / \sigma_{\text{data}} = 2x$。方差：$1.999^2 \times 0.25 + 0.025^2 = 0.999 + 0.001 = 1$。

**对比误差放大倍数。** 如果坚持预测噪声，去噪器是 $D_\theta = x_\sigma - \sigma\epsilon_\theta$，
相当于 $c_{\text{skip}} = 1$、$c_{\text{out}} = -\sigma$。在 $\sigma = 20$ 处，网络的误差被放大 20 倍；
EDM 的预条件只放大 0.5 倍。差了 40 倍。这个数会在第 4.3 节解释 DIAMOND 的选择时再用到。

### 1.9 训练时 σ 怎么采样，采样时怎么走

**训练：噪声水平的分布。** EDM 让 $\ln\sigma$ 服从正态分布 $\mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$，
论文默认 $P_{\text{mean}} = -1.2$、$P_{\text{std}} = 1.2$（论文 Table 1）。
中位数是 $e^{-1.2} = 0.30$，正负一个标准差的范围是 $e^{-2.4} = 0.09$ 到 $e^{0} = 1$。
意思是把训练的大部分力气花在中等噪声上：噪声太小时去噪没什么可学的，
噪声太大时除了输出数据的平均也做不了什么。

**采样：一个常微分方程。** 带噪样本随噪声水平变化的轨迹满足下面的方程
（概率流 ODE，出自 Song et al. 2021，arXiv:2011.13456；这里取 EDM 的写法，直接把 $\sigma$ 当作时间）：

$$
\frac{dx}{d\sigma} = -\sigma\, \nabla_x \log p_\sigma(x) = \frac{x - D_\theta(x; \sigma)}{\sigma}
$$

第二个等号用了第 1.5 节的换算 $\nabla \log p_\sigma = (D_\theta - x)/\sigma^2$。
右边就是对噪声的估计 $\hat\epsilon = (x - D_\theta)/\sigma$：
$\sigma$ 每减小一点，就从样本里减掉相应分量的"估计出来的噪声"。

这个方程本章不推导，但可以用第 1.6 节的例子验证它是对的。那里 $D^\star(x; \sigma) = 2 + (x - 2)/(1 + \sigma^2)$，代入：

$$
\frac{dx}{d\sigma} = \frac{(x - 2)\big(1 - \frac{1}{1 + \sigma^2}\big)}{\sigma} = \frac{(x - 2)\, \sigma}{1 + \sigma^2}
\quad \Longrightarrow \quad
x(\sigma) - 2 \;\propto\; \sqrt{1 + \sigma^2}
$$

样本到均值的距离正比于 $\sqrt{1 + \sigma^2}$，而 $p_\sigma = \mathcal{N}(2, 1 + \sigma^2)$ 的标准差正是 $\sqrt{1 + \sigma^2}$。
所以沿着这个方程把 $\sigma$ 从大走到 0，样本的分布恰好从 $p_\sigma$ 变回数据分布 $\mathcal{N}(2, 1)$。

**Euler 采样器。** 选一串递减的噪声水平 $\sigma_0 > \sigma_1 > \dots > \sigma_N = 0$，
从 $x_0 \sim \mathcal{N}(0, \sigma_0^2 I)$ 出发（这里的下标是采样步的编号）：

$$
x_{i+1} = x_i + (\sigma_{i+1} - \sigma_i)\, \frac{x_i - D_\theta(x_i; \sigma_i)}{\sigma_i}
$$

整理一下可以写成更好记的形式：

$$
x_{i+1} = D_\theta(x_i; \sigma_i) + \frac{\sigma_{i+1}}{\sigma_i}\, \big( x_i - D_\theta(x_i; \sigma_i) \big)
$$

读法：**先让去噪器猜出干净样本，再把"当前样本减去猜测"这部分噪声保留 $\sigma_{i+1}/\sigma_i$ 的比例。**
最后一步 $\sigma_{i+1} = 0$，输出就是去噪器的猜测本身。

**Heun 采样器。** EDM 推荐的二阶方法：先按 Euler 试走一步，在终点再算一次斜率，
用两个斜率的平均重新走。每一步调用两次网络，同样的调用次数下通常比 Euler 准。
世界模型里步数压到个位数时，两者差别不大，DIAMOND 用的是 Euler（待核对）。

**噪声日程。** EDM 用下面的公式在 $\sigma_{\max}$ 和 $\sigma_{\min}$ 之间取 $N$ 个点，再补一个 0：

$$
\sigma_i = \Big( \sigma_{\max}^{1/\rho} + \frac{i}{N - 1}\, \big( \sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho} \big) \Big)^{\rho},
\qquad i = 0, \dots, N - 1, \qquad \sigma_N = 0
$$

论文默认 $\rho = 7$、$\sigma_{\min} = 0.002$、$\sigma_{\max} = 80$。$\rho$ 越大，点越往低噪声那一头挤。

**手算日程。** 取 $N = 3$、$\sigma_{\max} = 5$、$\sigma_{\min} = 0.002$、$\rho = 7$：

- $5^{1/7} = e^{1.609/7} = 1.2585$，$0.002^{1/7} = e^{-6.215/7} = 0.4116$
- $i = 0$：$\sigma_0 = 5$
- $i = 1$：中点是 $(1.2585 + 0.4116)/2 = 0.8350$，$\sigma_1 = 0.8350^7 = 0.283$
- $i = 2$：$\sigma_2 = 0.002$；最后 $\sigma_3 = 0$

三步走的是 $5 \to 0.283 \to 0.002 \to 0$。第一步就把噪声水平降了 17 倍多，后面两步在收拾细节。
如果把 $\sigma_{\max}$ 换成 80，同样三步是 $80 \to 2.51 \to 0.002 \to 0$。

**手算采样。** 接着用第 1.6 节的数据 $\mathcal{N}(2, 1)$ 和它的最优去噪器 $D^\star(x; \sigma) = 2 + (x - 2)/(1 + \sigma^2)$。
从 $\sigma = 2$ 处的样本 $x = 4.5$ 出发，比较三种走法：

| 走法 | 过程 | 结果 |
| --- | --- | --- |
| 一步 Euler：$2 \to 0$ | 输出就是 $D^\star(4.5; 2) = 2.5$ | 2.50 |
| 两步 Euler：$2 \to 1 \to 0$ | 第一步：$2.5 + \frac{1}{2}(4.5 - 2.5) = 3.5$；第二步：$D^\star(3.5; 1) = 2 + 1.5/2 = 2.75$ | 2.75 |
| 精确解 | $2 + (4.5 - 2) \times \sqrt{1/5}$ | 3.12 |

精确解把"离均值 2.5"按标准差的比例 $\sqrt{1/5} = 0.447$ 缩小，得到一个分布正确的样本。
一步 Euler 直接输出后验均值，把偏离量缩到了五分之一。如果所有样本都这样生成，
得到的分布是 $\mathcal{N}(2, 1/5)$，比真实的数据分布窄得多。

这就是"步数太少，样本往平均值塌"的机制。数据只有一个峰时，表现为多样性不足；
数据有多个峰时（下一帧有好几种可能），平均值落在几个峰之间，表现为画面发糊。第 3.4 节接着讲。

### 1.10 算法

```
算法：EDM 训练

输入：数据集 D，数据标准差 σ_data，网络 F_θ
重复：
 1. 取一个批次的干净样本 x                                     形状 [B, C, H, W]
 2. 每个样本独立采样噪声水平：ln σ ~ N(P_mean, P_std²)           形状 [B]
 3. 加噪：x_σ = x + σ · ε，ε ~ N(0, I)
 4. 算四个系数 c_in, c_skip, c_out, c_noise（都是 σ 的函数）      形状 [B, 1, 1, 1]
 5. 网络的回归目标：F_target = (x − c_skip · x_σ) / c_out
 6. 损失 = mean((F_θ(c_in · x_σ, c_noise) − F_target)²)
 7. 反向传播，更新 θ
```

第 5、6 步和"先算 $D_\theta$ 再乘权重 $\lambda(\sigma)$"在数学上相同（见第 1.7 节的要求二和要求四），
直接在 $F$ 的空间里算数值更稳。

```
算法：Euler 采样（N 步）

输入：去噪器 D_θ，噪声日程 σ_0 > σ_1 > … > σ_N = 0
 1. x ← σ_0 · ε，ε ~ N(0, I)
 2. 对 i = 0 … N−1：
        x̂ ← D_θ(x; σ_i)                         去噪器猜干净样本
        x ← x̂ + (σ_{i+1} / σ_i) · (x − x̂)        保留一部分噪声
 3. 返回 x
```

### 1.11 代码对应

对应 `01-ddpm-to-edm.py`（讲义待写）。

| 函数（规划中） | 做什么 |
| --- | --- |
| `ddpm_alpha_bar_to_sigma()` | 第 1.2 节的换算，把 DDPM 的日程画成 $\sigma$ 随步数的曲线 |
| `edm_precond_coeffs(sigma)` | 返回 $c_{\text{in}}, c_{\text{skip}}, c_{\text{out}}, c_{\text{noise}}$ |
| `EDMDenoiser.forward(x_noisy, sigma)` | 用四个系数把网络 $F_\theta$ 包成去噪器 $D_\theta$ |
| `edm_loss()` | 第 1.10 节的训练步 |
| `karras_sigmas(n_steps, sigma_min, sigma_max, rho)` | 噪声日程 |
| `sample_euler()`，`sample_heun()` | 两种采样器 |

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `x`，`x_noisy` | `[B, C, H, W]` | 干净样本、带噪样本 |
| `sigma` | `[B]`，计算时扩成 `[B, 1, 1, 1]` | 每个样本自己的噪声水平 |
| `c_noise` | `[B]` | 送进网络的噪声水平编码，之后变成 `[B, D]` 的嵌入 |
| `sigmas` | `[N + 1]` | 采样日程，最后一个是 0 |

> 📌 待回填：同一个网络用 1、2、3、5、10、50 步采样时的样本质量（二维玩具数据上看样本分布，小图像上看 FID 或直接看图），
> 以及"预测噪声"与"EDM 预条件"两种参数化在少步下的对比，来自 `01-ddpm-to-edm.py`。
> 定性预期：步数减少时，预测噪声的版本先坏。

### 1.12 常见误区

- **误区：EDM 是一种不同于扩散模型的新模型。**
  实际情况：它和 DDPM 是同一类模型，学的是同一个东西（各个噪声水平下的 score）。
  EDM 的贡献是把设计空间拆开，并给每个模块找了更好的选择。DDPM 是这个设计空间里的一个点。
- **误区：预测噪声、预测干净样本在数学上等价，所以选哪个无所谓。**
  实际情况：最优解可以互相换算，但网络不会恰好学到最优解。
  同样大小的网络误差，换算到去噪结果上，在不同参数化下可以差几十倍（第 1.8 节）。
- **误区：采样的步数和噪声水平必须和训练时一致。**
  实际情况：网络学的是"任意 $\sigma$ 下怎么去噪"。采样时用几步、每步取哪个 $\sigma$，可以在训练之后再决定。
- **误区：一步采样得到的是"质量差一点的样本"。**
  实际情况：一步 Euler 的输出是后验均值，是所有可能答案的平均，性质上就不是一个样本。
  要一步得到真正的样本，需要专门的训练办法（第 9 节的蒸馏和 shortcut 模型）。

---

## 2. Flow matching

### 2.1 直觉

扩散模型的故事是"先把数据一步步弄脏，再学着擦干净"。Flow matching 换了一个更直白的说法：
**在噪声和数据之间拉一条直线，让网络学沿着这条线该往哪走、走多快。**

工程师熟悉的类比是动画里的补间（tween）。给定起点和终点两个关键帧，
中间帧用线性插值算出来，每一帧的"速度"就是终点减起点。
Flow matching 的训练数据就是这样造出来的：随机配一对（噪声，数据），在两者的连线上随机取一点，
告诉网络"站在这一点时，速度应该是数据减噪声"。

```
   噪声 ε                                                数据 x
   u = 0                      u = 0.5                     u = 1
     ●───────────────────────────●───────────────────────────●
                                 x_u = (1 − u)·ε + u·x
                                 速度 = x − ε（整条线上都一样）
```

麻烦在于连线会交叉。空间里同一个点，可能同时落在很多条不同的连线上，每条线给出的速度都不一样。
网络在这一点上只能输出一个速度。它该听谁的？本节的推导回答的就是这个问题：
听它们的平均，而且这个平均恰好是对的。

这条路线出自 Lipman et al. 2023（arXiv:2210.02747）；
同期的 Rectified Flow（Liu et al. 2022，arXiv:2209.03003）从"把轨迹拉直"的角度得到了同样的训练目标。

### 2.2 定义

路径时间 $u \in [0, 1]$，$u = 0$ 是纯噪声，$u = 1$ 是数据。注意方向和扩散的习惯相反：
扩散里"时间"越大噪声越多，这里 $u$ 越大越干净。

- **线性插值路径**：取一对独立的 $\epsilon \sim \mathcal{N}(0, I)$ 和 $x \sim p_{\text{data}}$，

$$
x_u = (1 - u)\, \epsilon + u\, x
$$

- **条件速度**：固定这一对 $(\epsilon, x)$，路径上的点对 $u$ 求导。

$$
\frac{d x_u}{d u} = x - \epsilon
$$

- **边际分布** $p_u$：所有连线在时刻 $u$ 的点合在一起的分布。$p_0 = \mathcal{N}(0, I)$，$p_1 = p_{\text{data}}$。
- **边际速度场**：经过同一个点 $x_u$ 的所有连线的速度的平均。

$$
v^\star(x_u, u) = \mathbb{E}\big[ x - \epsilon \mid x_u \big]
$$

- **生成**：从 $x_0 \sim \mathcal{N}(0, I)$ 出发，解常微分方程 $\dfrac{d x_u}{d u} = v_\theta(x_u, u)$ 到 $u = 1$。

论文里的记号是 $x_t = (1 - t)\, x_0 + t\, x_1$，其中 $x_0$ 是噪声、$x_1$ 是数据、$t$ 是路径时间。
对应到本章：$t \to u$，$x_0 \to \epsilon$，$x_1 \to x$。

### 2.3 推导：条件 flow matching 损失

**想要的目标**是让网络回归边际速度场。这叫（边际）flow matching 损失：

$$
\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{u,\ x_u \sim p_u} \big\Vert v_\theta(x_u, u) - v^\star(x_u, u) \big\Vert^2
$$

和显式 score matching 一样，它没法直接用：$v^\star$ 是一个条件期望，要对所有经过 $x_u$ 的连线求平均，算不出来。

**能算的目标**是让网络回归单条连线的速度。这叫条件 flow matching 损失：

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{u,\ \epsilon,\ x} \big\Vert v_\theta(x_u, u) - (x - \epsilon) \big\Vert^2,
\qquad x_u = (1 - u)\, \epsilon + u\, x
$$

下面证明两者对 $\theta$ 的梯度相同。固定一个 $u$，把 $v_\theta(x_u, u)$ 简写成 $v_\theta$。

**第一步：展开两个损失的平方。**

$$
\begin{aligned}
\mathcal{L}_{\text{FM}} &= \mathbb{E}_{x_u} \Vert v_\theta \Vert^2 \;-\; 2\, \mathbb{E}_{x_u} \big[ v_\theta^\top\, v^\star(x_u, u) \big] \;+\; \mathbb{E}_{x_u} \Vert v^\star \Vert^2 \\
\mathcal{L}_{\text{CFM}} &= \mathbb{E}_{\epsilon, x} \Vert v_\theta \Vert^2 \;-\; 2\, \mathbb{E}_{\epsilon, x} \big[ v_\theta^\top\, (x - \epsilon) \big] \;+\; \mathbb{E}_{\epsilon, x} \Vert x - \epsilon \Vert^2
\end{aligned}
$$

**第二步：第一项相等。** $v_\theta$ 只依赖 $x_u$。对 $(\epsilon, x)$ 求期望，和先算出 $x_u$ 再对 $x_u$ 的分布求期望，是一回事。

**第三步：第三项都与 $\theta$ 无关。** 它们的数值不同，但求梯度时都消失。

**第四步：交叉项相等。** 用全期望公式 $\mathbb{E}[Y] = \mathbb{E}\big[ \mathbb{E}[Y \mid X] \big]$，
并注意给定 $x_u$ 之后 $v_\theta$ 是个定值，可以提到内层期望外面：

$$
\begin{aligned}
\mathbb{E}_{\epsilon, x} \big[ v_\theta^\top (x - \epsilon) \big]
&= \mathbb{E}_{x_u} \Big[ \mathbb{E}\big[ v_\theta^\top (x - \epsilon) \mid x_u \big] \Big] \\
&= \mathbb{E}_{x_u} \Big[ v_\theta^\top\, \mathbb{E}\big[ x - \epsilon \mid x_u \big] \Big] \\
&= \mathbb{E}_{x_u} \big[ v_\theta^\top\, v^\star(x_u, u) \big]
\end{aligned}
$$

最后一步用了 $v^\star$ 的定义。

**结论。**

$$
\mathcal{L}_{\text{CFM}}(\theta) = \mathcal{L}_{\text{FM}}(\theta) + \text{与 } \theta \text{ 无关的常数},
\qquad
\nabla_\theta \mathcal{L}_{\text{CFM}} = \nabla_\theta \mathcal{L}_{\text{FM}}
$$

这和第 1.4 节是同一个套路：**想回归的目标算不出来，就换成一个算得出来的、带噪声的目标，
只要后者的条件期望等于前者，梯度就完全一样。** 第 1.4 节的条件 score 对应这里的条件速度。

**还差一环：为什么沿着 $v^\star$ 走，分布就恰好是 $p_u$？** 直觉上，$p_u$ 是无数条连线上的点叠出来的。
在每个位置，按经过那里的连线的平均速度推着概率质量走，流进流出的总量和"各条线各走各的"是一样的。
严格的证明要用连续性方程，见 Lipman et al. 2023 的定理 1，本章不展开，下面用手算例子验证。

### 2.4 手算例子：一维高斯

数据还是 $\mathcal{N}(2, 1)$，噪声是 $\mathcal{N}(0, 1)$，两者独立。

**边际分布。** $x_u = (1 - u)\epsilon + u x$ 是两个独立高斯的线性组合：

$$
p_u = \mathcal{N}\big( 2u,\ \nu_u^2 \big), \qquad \nu_u^2 = (1 - u)^2 + u^2
$$

**边际速度场。** $(x - \epsilon)$ 和 $x_u$ 联合高斯，用条件期望公式
$\mathbb{E}[Y \mid X] = \mathbb{E}[Y] + \dfrac{\mathrm{Cov}(Y, X)}{\mathrm{Var}(X)} (X - \mathbb{E}[X])$。其中：

$$
\mathbb{E}[x - \epsilon] = 2, \qquad
\mathrm{Cov}(x - \epsilon,\ x_u) = u \cdot 1 - (1 - u) \cdot 1 = 2u - 1
$$

所以：

$$
v^\star(x_u, u) = 2 + \frac{2u - 1}{(1 - u)^2 + u^2}\, (x_u - 2u)
$$

| $u$ | $\nu_u^2$ | $v^\star(x_u, u)$ | 读法 |
| --- | --- | --- | --- |
| 0 | 1 | $2 - x_u$ | 往数据的均值 2 走，离得越远走得越快 |
| 0.25 | 0.625 | $2 - 0.8\,(x_u - 0.5)$ | |
| 0.5 | 0.5 | $2$ | 所有位置的速度都一样 |
| 0.75 | 0.625 | $2 + 0.8\,(x_u - 1.5)$ | |
| 1 | 1 | $x_u$ | |

**单个样本的目标是带噪声的。** 取 $u = 0.5$，下面三对 $(x, \epsilon)$ 都落在 $x_u = 1.5$：

| 数据 $x$ | 噪声 $\epsilon$ | $x_u = 0.5\,\epsilon + 0.5\,x$ | 这一对的回归目标 $x - \epsilon$ |
| --- | --- | --- | --- |
| 2.5 | 0.5 | 1.5 | 2.0 |
| 3.5 | −0.5 | 1.5 | 4.0 |
| 1.0 | 2.0 | 1.5 | −1.0 |

三条连线在这一点交叉，速度分别是 2、4、−1。网络学到的是所有经过这里的连线的条件平均，等于 2。

**精确的轨迹。** 令 $m = x_u - 2u$（到当前均值的距离）。由速度场，
$\dfrac{dm}{du} = \dfrac{2u - 1}{\nu_u^2}\, m$；而 $\dfrac{d \nu_u^2}{du} = 2(2u - 1)$，所以
$\dfrac{dm}{du} = \dfrac{1}{2 \nu_u^2} \dfrac{d \nu_u^2}{du}\, m$，解得 $m \propto \nu_u$。
轨迹上的点到均值的距离始终正比于当前分布的标准差，这正说明分布一直是 $p_u$。
又因为 $\nu_0 = \nu_1 = 1$，从 $u = 0$ 走到 $u = 1$：

$$
x_1 = 2 + (x_0 - 0) \times \frac{\nu_1}{\nu_0} = 2 + x_0
$$

从噪声 $x_0 = 0.8$ 出发，终点是 2.8。$x_0 \sim \mathcal{N}(0, 1)$，所以 $x_1 \sim \mathcal{N}(2, 1)$，正是数据分布。

**Euler 采样。** 从 $x_0 = 0.8$ 出发，用上表的速度场：

| 步数 | 过程 | 终点 |
| --- | --- | --- |
| 1 步 | $0.8 + 1 \times (2 - 0.8)$ | 2.00 |
| 2 步 | $0.8 + 0.5 \times 1.2 = 1.4$；$1.4 + 0.5 \times 2$ | 2.40 |
| 4 步 | $0.8 \to 1.1 \to 1.48 \to 1.98 \to 2.576$ | 2.58 |
| 精确解 | | 2.80 |

4 步的中间过程：$u = 0.25$ 处速度是 $2 - 0.8 \times (1.1 - 0.5) = 1.52$，走到 $1.1 + 0.25 \times 1.52 = 1.48$；
$u = 0.5$ 处速度是 2，走到 1.98；$u = 0.75$ 处速度是 $2 + 0.8 \times (1.98 - 1.5) = 2.384$，走到 $1.98 + 0.596 = 2.576$。

两个观察：

1. 一步 Euler 不管从哪个噪声出发，结果都是 $x_0 + (2 - x_0) = 2$，也就是数据的均值。
   这和第 1.9 节的一步采样是同一个现象。
2. 每条条件路径都是直线，但边际速度场画出的轨迹是弯的：
   精确轨迹在 $u = 0.5$ 时位于 $1 + 0.8 \times \sqrt{0.5} = 1.566$，而直线上的中点是 $(0.8 + 2.8)/2 = 1.8$。
   **"用了直线路径"不等于"采样轨迹是直的"**，所以 flow matching 模型同样需要多步采样。

### 2.5 与扩散、v-prediction 的关系

**和扩散是同一族带噪分布。** 把路径两边同除以 $u$：

$$
\frac{x_u}{u} = x + \frac{1 - u}{u}\, \epsilon
\quad \Longrightarrow \quad
x_u = u \cdot x_\sigma, \qquad \sigma = \frac{1 - u}{u}, \qquad u = \frac{1}{1 + \sigma}
$$

Flow matching 路径上 $u$ 处的点，就是噪声水平 $\sigma = (1 - u)/u$ 的带噪样本缩小 $u$ 倍。
比如 $u = 0.25$ 对应 $\sigma = 3$；$u = 0.5$ 对应 $\sigma = 1$；$u = 0.8$ 对应 $\sigma = 0.25$。
这和第 1.2 节把 DDPM 换算成 $\sigma$ 是同一个操作。

**速度和其他预测目标的换算。** 由 $x_u = (1 - u)\epsilon + u x$ 和 $v = x - \epsilon$ 两个方程，解出：

$$
x = x_u + (1 - u)\, v, \qquad \epsilon = x_u - u\, v
$$

验证第一个：$x_u + (1 - u)(x - \epsilon) = (1 - u)\epsilon + u x + (1 - u) x - (1 - u)\epsilon = x$。
所以速度网络、去噪器、噪声预测器之间仍然只差一个线性变换：

$$
v_\theta = \frac{D_\theta - x_u}{1 - u} = \frac{x_u - \epsilon_\theta}{u}
$$

用第 2.4 节验证：$u = 0.5$、$x_u = 1.5$ 时 $v^\star = 2$，于是最优去噪结果是 $1.5 + 0.5 \times 2 = 2.5$。
换到 EDM 的坐标：$\sigma = 1$，$x_\sigma = x_u / u = 3$，后验均值 $2 + (3 - 2)/(1 + 1) = 2.5$。一致。

**v-prediction 是什么。** Salimans & Ho 2022（*Progressive Distillation for Fast Sampling of Diffusion Models*）
在 DDPM 那种"信号和噪声的系数平方和为 1"的路径上，定义了一个新的预测目标。
把那条路径用角度 $\varphi$ 写出来：

$$
x_\varphi = \cos\varphi \cdot x + \sin\varphi \cdot \epsilon,
\qquad
v_{\text{pred}} = \frac{d x_\varphi}{d \varphi} = \cos\varphi \cdot \epsilon - \sin\varphi \cdot x
$$

这是一条圆弧路径，$v_{\text{pred}}$ 是沿圆弧的速度。对比 flow matching：

| | 路径 | 速度（对路径参数求导） |
| --- | --- | --- |
| v-prediction | 圆弧：$\cos\varphi \cdot x + \sin\varphi \cdot \epsilon$ | $\cos\varphi \cdot \epsilon - \sin\varphi \cdot x$ |
| flow matching | 直线：$u \cdot x + (1 - u) \cdot \epsilon$ | $x - \epsilon$ |

两者是同一个想法（预测"沿路径的速度"，它是数据和噪声的线性组合），只是路径的形状不同，
时间的方向也相反（$\varphi$ 增大是加噪，$u$ 增大是去噪），所以符号相反。

**和 EDM 预条件的关系。** 第 1.7 节得到的网络目标，取 $\sigma_{\text{data}} = 1$：

$$
F_{\text{target}} = \frac{\sigma\, x - \epsilon}{\sqrt{1 + \sigma^2}} = \sin\varphi \cdot x - \cos\varphi \cdot \epsilon = -v_{\text{pred}},
\qquad \tan\varphi = \sigma
$$

所以 EDM 从"输入输出单位方差、误差放大最小"推出来的目标，正好是 v-prediction（差一个符号）。
三条看起来不同的路线，在这里汇合到同一个结论：**网络应该预测数据和噪声的一个混合，
混合比例随噪声水平变化：高噪声时偏向数据，低噪声时偏向噪声。**

| 噪声水平 | 预测噪声 | 预测干净样本 | 预测速度 / EDM 的 $F$ |
| --- | --- | --- | --- |
| 低噪声 | 好 | 在抄输入 | 好（接近预测噪声） |
| 高噪声 | 在抄输入，误差被放大 | 好 | 好（接近预测干净样本） |

### 2.6 Rectified flow：把轨迹拉直

第 2.4 节看到，独立配对 $(\epsilon, x)$ 时连线到处交叉，边际轨迹是弯的，少步采样有误差。
Rectified Flow 的办法叫 **reflow**：

1. 用独立配对训练出第一个模型
2. 用它从很多个噪声 $\epsilon$ 出发，各自解 ODE 得到样本 $\hat{x}$，把 $(\epsilon, \hat{x})$ 存下来
3. 用这些**配好对**的数据重新训练：路径和损失不变，只是 $\epsilon$ 和 $x$ 不再独立

**手算。** 第 2.4 节的精确轨迹把 $\epsilon$ 送到 $2 + \epsilon$。用这个配对重新造路径：

$$
x_u = (1 - u)\, \epsilon + u\, (2 + \epsilon) = \epsilon + 2u, \qquad \text{条件速度} = (2 + \epsilon) - \epsilon = 2
$$

所有连线互相平行，不再交叉，每条线的速度都是 2。边际速度场就是常数 2，
一步 Euler：$0.8 + 1 \times 2 = 2.8$，精确。

这个例子里一次 reflow 就把轨迹完全拉直了。真实数据上做不到这么干净，但轨迹会明显变直，需要的步数随之减少。
"让一步也能走对"还有别的办法，第 9 节的 shortcut 模型是其中一种，也是 Dreamer 4 用的那一种。

### 2.7 算法

```
算法：条件 flow matching 训练

重复：
 1. 取一个批次的数据 x                              形状 [B, C, H, W]
 2. 采样噪声 ε ~ N(0, I)；采样路径时间 u ~ U(0, 1)   形状 [B]
 3. 插值：x_u = (1 − u) · ε + u · x
 4. 目标速度：v_target = x − ε
 5. 损失 = mean((v_θ(x_u, u) − v_target)²)
 6. 反向传播，更新 θ
```

```
算法：Euler 采样（N 步）

 1. x ← ε，ε ~ N(0, I)；步长 d = 1 / N
 2. 对 i = 0 … N−1：
        u ← i · d
        x ← x + d · v_θ(x, u)
 3. 返回 x
```

和第 1.10 节对比：训练少了四个系数，采样少了噪声日程，代码短得多。
第 2 步里 $u$ 不一定均匀采样。和 EDM 的 $\ln\sigma$ 取正态分布一个道理，
很多实现会让 $u$ 更多地落在中间（具体分布因实现而异）。

### 2.8 代码对应

对应 `02-flow-matching.py`（讲义待写）。

| 函数（规划中） | 做什么 |
| --- | --- |
| `interpolate(x, eps, u)` | 线性插值得到 $x_u$ |
| `cfm_loss(model, x)` | 第 2.7 节的训练步 |
| `sample_flow_euler(model, shape, n_steps)` | Euler 采样 |
| `velocity_to_denoised(x_u, v, u)` | 第 2.5 节的换算 $x = x_u + (1 - u) v$ |
| `u_to_sigma(u)`，`sigma_to_u(sigma)` | flow matching 与 EDM 坐标互换 |
| `reflow_pairs(model, n_pairs)` | 生成 $(\epsilon, \hat{x})$ 配对，用于第二轮训练 |

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `x`，`eps`，`x_u`，`v_target` | `[B, C, H, W]` | 数据、噪声、插值点、目标速度 |
| `u` | `[B]`，计算时扩成 `[B, 1, 1, 1]` | 路径时间 |

> 📌 待回填：同一份二维玩具数据上，flow matching 与 EDM 在 1、2、4、8 步下的样本质量对比；
> 一次 reflow 前后轨迹的弯曲程度和少步样本质量，来自 `02-flow-matching.py`。
> 定性预期：reflow 之后少步采样明显变好。

### 2.9 常见误区

- **误区：flow matching 用的是直线路径，所以一步就能采样。**
  实际情况：直的是每一对 $(\epsilon, x)$ 之间的条件路径。网络学到的边际速度场画出的轨迹是弯的（第 2.4 节）。
  一步 Euler 得到的还是数据的平均。
- **误区：flow matching 和扩散是两种不同的生成模型，要二选一。**
  实际情况：线性路径下，两者描述的是同一族带噪分布，速度、噪声、干净样本之间可以互相换算（第 2.5 节）。
  差别在工程选择上：路径怎么参数化、网络预测哪个量、噪声水平怎么采样和加权。
- **误区：回归目标 $x - \epsilon$ 是随机配对出来的，网络学到的是一个随机的东西。**
  实际情况：单个样本的目标确实是随机的，但平方损失的最优解是条件期望，是一个确定的速度场。
- **误区：$u$ 和 DDPM 的步数 $k$ 是同一个方向。**
  实际情况：方向相反。$u = 0$ 是纯噪声，$u = 1$ 是数据。读论文和代码时先确认这一点，
  不同论文的约定不一样，弄反了模型照样能训练，只是采样出来全是噪声。

---

## 3. 以动作为条件的下一帧扩散

### 3.1 直觉

前两节的模型生成的是"一张图"。世界模型要生成的是"**在这个画面下做了这个动作之后的下一张图**"。

类比：同一位修图师，以前只给他一张盖满雪花的照片。现在额外给他两样参考资料：
前几帧的画面，和这段时间里按下的按键。前几帧告诉他场景长什么样、东西在往哪动；
按键告诉他这一帧里角色会怎么动。有了这些，他要猜的东西少了很多：
画面里绝大部分内容直接从上一帧搬过来就行，真正不确定的只有一小部分。

从接口上看，这仍然是 Phase 1 第 1.3 节的世界模型，只是三样东西换了：

| | Phase 1 | 本节 |
| --- | --- | --- |
| 预测什么 | 低维状态 $s_{t+1}$ | 整帧画面 $o_{t+1}$ |
| 输入 | 当前状态 $s_t$ 和动作 $a_t$ | 最近 $m$ 帧 $o_{t-m+1:t}$ 和对应的动作 |
| 模型输出 | 一个点，或一个高斯 | 一个可以采样的任意分布 |

第三行是换成扩散模型的主要理由。下一帧常常有不止一种可能（敌人往左还是往右），
高斯分布和平方损失只能给出这些可能的平均，扩散模型可以从中挑一种生成出来。

### 3.2 定义

要建模的条件分布是：

$$
p_\theta\big( o_{t+1} \mid o_{t-m+1:t},\ a_{t-m+1:t} \big)
$$

其中 $m$ 是上下文长度。单帧画面里没有速度（Phase 1 第 3.3 节的马尔可夫性），所以至少要看两帧；实际取几帧。

去噪器在第 1 节的基础上多两个输入。记带噪的下一帧为 $\tilde{o}_{t+1} = o_{t+1} + \sigma\epsilon$：

$$
D_\theta\big( \tilde{o}_{t+1};\ \sigma,\ o_{t-m+1:t},\ a_{t-m+1:t} \big) \approx o_{t+1}
$$

训练损失就是第 1.7 节的去噪损失，期望多了一层"从回放数据里取一段"：

$$
\mathcal{L}(\theta) = \mathbb{E}_{(o, a) \sim \mathcal{D},\ \sigma,\ \epsilon}\;
\lambda(\sigma)\, \Big\Vert D_\theta\big( o_{t+1} + \sigma\epsilon;\ \sigma,\ o_{t-m+1:t},\ a_{t-m+1:t} \big) - o_{t+1} \Big\Vert^2
$$

预条件的四个系数照旧，只作用在带噪的那一帧上。条件帧是干净的，不乘 $c_{\text{in}}$。

### 3.3 条件怎么喂进网络

```
   过去 m 帧（干净）  [B, m, C, H, W]
        │  沿通道维摊平成 [B, m·C, H, W]
        ▼
   ┌─────────┐      带噪的下一帧 [B, C, H, W]
   │ 通道拼接 │◀──── （先乘 c_in）
   └────┬────┘
        ▼  [B, (m+1)·C, H, W]
   ┌───────────────────────────────┐
   │            U-Net              │◀── 条件向量 = 噪声水平嵌入 + 动作嵌入   [B, D]
   │   每个残差块里：归一化层的      │     （通过归一化层的缩放和平移注入）
   │   缩放 γ 和平移 β 由条件向量给出 │
   └───────────────┬───────────────┘
                   ▼  F_θ 的输出 [B, C, H, W]
            × c_out，再加上 c_skip × 带噪帧  ──▶  去噪结果 [B, C, H, W]
```

| 条件 | 常见的注入方式 | 说明 |
| --- | --- | --- |
| 过去的画面 | 通道拼接（frame stacking） | 最简单。U-Net 的卷积天然能把上一帧的内容"搬"到下一帧。上下文长度固定 |
| 过去的画面 | 时间注意力 / 把多帧当序列 | 上下文可以更长，第 5、6 节的视频模型用这种 |
| 动作 | 嵌入后与噪声水平嵌入相加，经归一化层的缩放和平移注入 | 离散动作查表，连续动作过一个小 MLP |
| 动作 | 交叉注意力，或当作额外的 token | Transformer 骨干常用 |

"经归一化层的缩放和平移注入"在不同论文里叫 FiLM、条件归一化、自适应归一化，是同一类做法：
条件向量经过一个线性层，得到每个通道的缩放 $\gamma$ 和平移 $\beta$，作用在归一化之后的特征上。

**动作要对齐。** 生成 $o_{t+1}$ 最直接依赖的是 $a_t$。通常把上下文窗口内的每个动作都喂进去
（$a_{t-m+1}, \dots, a_t$）。如果数据集里动作和帧错开一位，模型学到的就是"上一步的动作导致这一帧"，
表现为操控有一帧延迟，或者干脆不跟手。这是本章最常见的实现错误之一（第 13 节）。

**让动作更"管用"：classifier-free guidance。** 训练时以一定概率把动作条件换成一个空标记；
采样时同时算有条件和无条件两个去噪结果，再沿两者之差外推：

$$
D_{\text{cfg}} = D_\theta(\cdot \mid \varnothing) + w\, \big( D_\theta(\cdot \mid a) - D_\theta(\cdot \mid \varnothing) \big)
$$

$w = 1$ 就是普通的有条件采样，$w > 1$ 放大"动作带来的差别"。代价是每步要算两次网络。
它是 Ho & Salimans 2022 提出的通用技巧，本章的各篇论文是否使用、用在哪个条件上各不相同。

### 3.4 推导：步数与模糊

第 1.9 节用单峰的数据说明了"一步采样会往平均值塌"。下一帧预测里更典型的情况是**多峰**。

**设定。** 把下一帧简化成一个数：小球下一帧在左边（$x = -1$）或右边（$x = +1$），各一半的概率。
这是下一帧预测里"敌人往左还是往右"的最小模型。

**最优去噪器。** 给定带噪观测 $x_\sigma$，用贝叶斯公式算小球在右边的后验概率。
两种情况的似然分别是 $\exp\big( -(x_\sigma \mp 1)^2 / (2\sigma^2) \big)$，先验相等，所以：

$$
P(x = +1 \mid x_\sigma)
= \frac{e^{-(x_\sigma - 1)^2 / 2\sigma^2}}{e^{-(x_\sigma - 1)^2 / 2\sigma^2} + e^{-(x_\sigma + 1)^2 / 2\sigma^2}}
= \frac{1}{1 + e^{-2 x_\sigma / \sigma^2}}
$$

第二个等号是分子分母同除以分子，并用 $(x_\sigma + 1)^2 - (x_\sigma - 1)^2 = 4 x_\sigma$。后验均值是：

$$
D^\star(x_\sigma; \sigma) = (+1) \cdot P + (-1) \cdot (1 - P) = 2P - 1 = \tanh\Big( \frac{x_\sigma}{\sigma^2} \Big)
$$

**代入数字。**

| 噪声水平 $\sigma$ | 带噪观测 $x_\sigma$ | $x_\sigma / \sigma^2$ | $D^\star$ | 含义 |
| --- | --- | --- | --- | --- |
| 5 | 3.0 | 0.12 | 0.12 | 几乎分不出左右，输出接近 0：两个位置上各有半个"鬼影" |
| 1 | 0.6 | 0.6 | 0.54 | 开始偏向右边 |
| 0.5 | 0.8 | 3.2 | 0.997 | 基本确定在右边 |

**一步采样。** 从 $\sigma = 5$ 一步走到 0，输出就是第一行的 0.12。
对所有可能的起点，输出都挤在 0 附近。小球既不在左也不在右，画面上是两个半透明的影子。这就是模糊。

**多步采样。** 用第 1.9 节的 Euler 更新，日程取 $5 \to 1 \to 0.2 \to 0$，从 $x = 3.0$ 出发：

- 第 1 步：$D^\star(3.0; 5) = 0.12$，$x \leftarrow 0.12 + \frac{1}{5} (3.0 - 0.12) = 0.70$
- 第 2 步：$D^\star(0.70; 1) = \tanh(0.70) = 0.60$，$x \leftarrow 0.60 + \frac{0.2}{1} (0.70 - 0.60) = 0.62$
- 第 3 步：$D^\star(0.62; 0.2) = \tanh(15.5) \approx 1.00$，$x \leftarrow 1.00$

三步之后落在 $+1$：小球在右边，画面是清晰的。如果起点是负的，同样的过程会落在 $-1$。
**多步采样的作用，是让样本在逐步降噪的过程中"选定"一个峰，而不是停在几个峰的平均上。**

由此得到选步数的原则：

| 下一帧的不确定性 | 需要的步数 | 原因 |
| --- | --- | --- |
| 几乎确定（给定过去帧和动作，只有一种结果） | 1 步也能用 | 后验只有一个峰，平均值就是答案 |
| 有几种可能（对手的动作、随机事件） | 至少几步 | 需要若干步来选定一个峰 |

世界模型里的下一帧，大部分像素属于第一种（背景不动），少部分属于第二种。
所以世界模型需要的步数远少于从零生成一张图（那里每个像素都高度不确定），
这是扩散世界模型能把步数压到个位数的根本原因。DIAMOND 对步数的选择就基于这一点（第 4.3 节）。

### 3.5 计算量的账

Phase 1 第 6.5 节算过决策时规划的账：每个真实步的模型调用次数是候选数乘以视野。
换成扩散世界模型，每次"模型调用"变成 $N$ 次去噪：

$$
\text{每个真实步的网络调用次数} = (\text{候选数}) \times H \times (\text{CEM 轮数}) \times N
$$

取 500 条候选、视野 10、CEM 迭代 5 轮、去噪 3 步：$500 \times 10 \times 5 \times 3 = 75{,}000$ 次 U-Net 前向计算。
批量化之后是 $10 \times 5 \times 3 = 150$ 次前向计算，每次的批大小是 500 帧图像。
Phase 1 里同样的账，网络是两层 MLP，输入是 2 个数。

这笔账说明了扩散世界模型的主流用法为什么是**在想象中训练策略**（Phase 6 的 Dreamer 路线），
而不是决策时规划：训练可以离线慢慢做，训练好的策略决策时只需要一次前向计算。
第 4 节的 DIAMOND 就是这么用的。想在特征空间里便宜地规划，是 Phase 10 的路线。

### 3.6 怎么评估

| 指标 | 衡量什么 | 备注 |
| --- | --- | --- |
| PSNR | 逐像素误差。$\text{PSNR} = 10 \log_{10}(\text{MAX}^2 / \text{MSE})$ | 像素取 $[0, 1]$ 时 MSE 为 0.001 对应 30 dB |
| SSIM、LPIPS | 结构相似度；深度特征上的感知距离 | 比 PSNR 更接近人的观感 |
| FVD | 一批生成视频与一批真实视频在特征分布上的距离 | 衡量"像不像真视频"，不要求逐帧对上 |
| 开环误差曲线 | 上面的指标随展开步数的变化 | 对世界模型最重要，见第 5 节 |
| 可控性 | 换一个动作，画面是否相应地变 | 第 7 节的 $\Delta$PSNR |
| 下游表现 | 在模型里训练出的策略，拿到真实环境里的回报 | 最终的检验；Phase 11 会系统讨论 |

逐像素指标对随机环境不公平：模型生成了一个合理但和真实录像不同的未来，PSNR 会很低。
所以随机性强的环境要看分布层面的指标和下游表现。

一个参照点：GameNGen（Valevski et al. 2024，arXiv:2408.14837）的摘要报告，
它在 DOOM 上下一帧预测的 PSNR 是 29.4，与有损 JPEG 压缩相当。

### 3.7 算法

```
算法：训练以动作为条件的下一帧去噪器

重复：
 1. 从回放数据取一段：帧 o_{t−m+1 : t+1}，动作 a_{t−m+1 : t}        [B, m+1, C, H, W]，[B, m]
 2. 上下文 ctx ← 前 m 帧；目标 x ← 最后一帧
 3. 采样噪声水平 σ（每个样本一个），加噪：x_σ = x + σ · ε
 4. 去噪：x̂ = D_θ(x_σ; σ, ctx, a)
 5. 按第 1.10 节在 F 空间算损失，更新 θ
```

```
算法：自回归展开（想象 H 步）

输入：真实的最近 m 帧，动作从哪来（策略、键盘、或给定的序列）
 1. buffer ← 真实的最近 m 帧
 2. 对 j = 1 … H：
        取动作 a（由策略根据 buffer 给出，或由人输入）
        x ← σ_0 · ε                                         从纯噪声开始
        对 i = 0 … N−1：                                     N 步去噪
            x̂ ← D_θ(x; σ_i, buffer, 动作)
            x ← x̂ + (σ_{i+1} / σ_i) · (x − x̂)
        把 x 追加进 buffer，丢掉最旧的一帧                     生成的帧成了下一步的条件
 3. 返回生成的 H 帧
```

展开算法的最后一行是第 5 节全部问题的来源：**模型生成的帧被当作下一步的条件。**

### 3.8 代码对应

对应 `03-action-conditioned-frame-diffusion.py`（讲义待写）。

| 函数（规划中） | 做什么 |
| --- | --- |
| `FrameDenoiser.forward(noisy_next, sigma, context_frames, actions)` | 带预条件的条件去噪器 |
| `ActionEmbedding` | 把动作变成条件向量 |
| `next_frame_loss()` | 第 3.7 节的训练步 |
| `sample_next_frame(context_frames, actions, n_steps)` | 生成一帧 |
| `rollout(init_frames, action_seq, n_steps)` | 自回归展开，返回整段生成的视频 |

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `context_frames` | `[B, m, C, H, W]` | 过去 $m$ 帧，进网络前摊平成 `[B, m*C, H, W]` |
| `actions` | `[B, m]`（离散）或 `[B, m, A]`（连续） | 与上下文帧一一对应 |
| `noisy_next` | `[B, C, H, W]` | 带噪的下一帧 |
| `sigma` | `[B]` | 噪声水平 |
| `cond` | `[B, D]` | 噪声水平嵌入加动作嵌入 |
| `rollout` 的输出 | `[B, H, C, H_img, W_img]` | 第二维是想象的步数 |

> 📌 待回填：下一帧预测的 PSNR，以及 1、2、3、5、10 步去噪下的画面对比（预期：步数太少时，运动的物体先变糊），
> 来自 `03-action-conditioned-frame-diffusion.py`。

### 3.9 常见误区

- **误区：世界模型和图像生成一样，需要几十步去噪。**
  实际情况：给定过去帧和动作后，下一帧的不确定性很小，几步就够（第 3.4 节）。
- **误区：生成的画面清晰，说明模型学会了动作的作用。**
  实际情况：模型可以完全忽略动作，只靠过去帧外推出一段很清晰的视频。
  要单独检查可控性：固定上下文，换不同的动作，看结果是否不同、是否符合动作的含义。
- **误区：上下文帧越多越好。**
  实际情况：帧多了能缓解部分可观测，但通道拼接的输入随之变大，
  模型也更容易学成"照抄最近几帧的运动趋势"而不看动作。上下文之外的东西模型照样记不住。
- **误区：PSNR 高就是好的世界模型。**
  实际情况：背景占了画面的绝大部分，背景对了 PSNR 就高。
  一个把小球弄丢的模型，PSNR 可能几乎不变，但对决策毫无用处。这正是下一节的主题。

---

## 4. DIAMOND：扩散世界模型

### 4.1 直觉：为什么"视觉细节重要"

Phase 8 的 IRIS 把每帧画面压成少量离散 token。压缩是有损的，丢掉的是什么，由重建损失说了算：
占面积大的东西（背景、大块的物体）重建错了损失就大，会被优先保住；
占面积小的东西重建错了损失几乎不变，最先被丢。

问题是，对决策重要的东西常常很小。打砖块里的球只有几个像素，它在哪，决定了挡板该往哪移；
远处的交通灯只有一个小色块，它是红是绿，决定了车该不该停。

类比：把服务日志做有损压缩，压缩器按"出现频率"决定保留什么。
海量的 INFO 行保住了，偶尔出现的一行 ERROR 被当成噪声丢了。
压缩率和还原度的数字都很好看，但排查故障时最需要的那一行没了。

DIAMOND（Alonso et al. 2024，arXiv:2405.12399）的标题就是这个论点：
*Diffusion for World Modeling: Visual Details Matter in Atari*。
它的做法是不离散化，用扩散模型直接在像素空间生成下一帧，然后像 Dreamer 一样在生成的画面里训练智能体。
名字是 DIffusion As a Model Of eNvironment Dreams 的缩写。

### 4.2 定义：三个部件

```
  ┌──────────────────────────── 世界模型 ────────────────────────────┐
  │                                                                 │
  │   过去 m 帧 + 动作 ──▶ ┌─────────────────────┐                   │
  │                       │ ① 扩散模型 D_θ        │ ──▶ 下一帧 ô_{t+1} │
  │   噪声 ──────────────▶ │   EDM，U-Net，少步去噪 │         │         │
  │                       └─────────────────────┘         │         │
  │                                                       ▼         │
  │   帧与动作的序列 ────▶ ┌─────────────────────┐                   │
  │                       │ ② 奖励 / 终止模型 R_ψ │ ──▶ r̂_t，是否结束  │
  │                       │   CNN + LSTM         │                   │
  │                       └─────────────────────┘                   │
  └─────────────────────────────────┬───────────────────────────────┘
                                    │ 想象出的 (画面, 奖励, 终止)
                                    ▼
                       ┌─────────────────────────┐
                       │ ③ actor-critic           │ ──▶ 动作 a_t ──▶ 回到 ①
                       │   直接看生成的像素         │
                       │   CNN + LSTM，策略头 + 价值头 │
                       └─────────────────────────┘
```

| 部件 | 输入 | 输出 | 怎么训练 |
| --- | --- | --- | --- |
| ① 扩散模型 | 过去 $m$ 帧、对应动作、带噪的下一帧、$\sigma$ | 去噪后的下一帧 | 第 3 节的去噪损失，EDM 预条件 |
| ② 奖励 / 终止模型 | 帧和动作的序列 | 奖励、是否终止 | 监督学习（分类的交叉熵） |
| ③ actor-critic | 当前帧（内部用 LSTM 记历史） | 动作分布 $\pi_\phi$、价值 $V_\psi$ | 完全在想象里训练：REINFORCE 加价值基线，价值目标用 λ-return |

和前面几章的世界模型对比：

| | Dreamer V3（Phase 6） | IRIS（Phase 8） | DIAMOND |
| --- | --- | --- | --- |
| 想象发生在哪 | RSSM 的隐状态里 | 离散 token 里，再解码成像素 | 直接在像素里 |
| 动力学骨干 | GRU | 自回归 Transformer | U-Net 扩散模型 |
| 奖励 / 终止 | 隐状态上的预测头 | Transformer 上的预测头 | 单独的一个模型 |
| 策略看什么 | 隐状态 | 解码出的画面 | 生成的画面 |
| 想象一步的代价 | 一次 GRU 前向 | 每个 token 一次 Transformer 前向 | $N$ 次 U-Net 前向 |

关于具体数字。按论文的方法与实验设置部分：上下文 4 帧，去噪 3 步，想象视野 15 步，
动作经自适应归一化层注入 U-Net（这几个数值待核对，写讲义时对照论文附录的超参数表）。
摘要给出的结果是：在 Atari 100k 基准上平均人类归一化分数 1.46，
是当时"完全在世界模型里训练"的智能体中的最好成绩。
Atari 100k 的含义是每个游戏只允许 10 万步交互，基准出自 SimPLe（Kaiser et al. 2020，arXiv:1903.00374）。

### 4.3 推导：为什么选 EDM，不选 DDPM

DIAMOND 论文花了专门的篇幅回答这个问题。结论是：**步数很少时，DDPM 式的噪声预测不稳定，
自回归展开很快跑偏；EDM 的预条件即使只用 1 步去噪也能保持稳定。** 下面把原因算出来。

**第一步：少步采样时，第一步去噪决定了大局。** 世界模型用 $N = 3$ 这样的步数。
按第 1.9 节的 Euler 更新，第一步之后：

$$
x_1 = D_\theta(x_0; \sigma_0) + \frac{\sigma_1}{\sigma_0}\, \big( x_0 - D_\theta(x_0; \sigma_0) \big)
$$

第一步是在最高的噪声水平 $\sigma_0$ 上调用网络。噪声日程里 $\sigma_1 \ll \sigma_0$，
所以 $x_1$ 基本上就是第一次去噪的结果。画面的整体结构（东西在哪）在这一步就定了，后面两步只是修细节。

**第二步：网络的误差传到去噪结果上，被放大多少。** 设网络在自己的输出空间里有大小为 $\delta$ 的误差。

- 预测噪声：$D_\theta = x_\sigma - \sigma\,\epsilon_\theta$，去噪结果的误差是 $\sigma\,\delta$
- EDM 预条件：$D_\theta = c_{\text{skip}}\, x_\sigma + c_{\text{out}}\, F_\theta$，去噪结果的误差是 $c_{\text{out}}\,\delta$

**第三步：代入数字。** 取 $\sigma_{\text{data}} = 0.5$，第一步的噪声水平 $\sigma_0 = 5$，下一个 $\sigma_1 = 0.283$
（第 1.9 节手算的日程），网络误差 $\delta = 0.05$。

$$
c_{\text{out}}(5) = \frac{5 \times 0.5}{\sqrt{25 + 0.25}} = \frac{2.5}{5.025} = 0.4975
$$

第一步之后，$x_1$ 里来自网络的误差是 $(1 - \sigma_1/\sigma_0) = 0.943$ 倍的去噪误差：

| 参数化 | 去噪结果的误差 | $x_1$ 的误差 | 相对数据标准差 0.5 |
| --- | --- | --- | --- |
| 预测噪声 | $5 \times 0.05 = 0.25$ | $0.943 \times 0.25 = 0.236$ | 47% |
| EDM 预条件 | $0.4975 \times 0.05 = 0.025$ | $0.943 \times 0.025 = 0.023$ | 4.7% |

同样的网络误差，预测噪声的版本在第一步之后就带着相当于数据标准差一半的偏差，后面只剩两步来补救。

**第四步：高噪声下，预测噪声的网络到底学了什么。** $\sigma \gg \sigma_{\text{data}}$ 时，
$x_\sigma = x + \sigma\epsilon \approx \sigma\epsilon$，所以回归目标 $\epsilon \approx x_\sigma / \sigma$。
网络只要把输入除以 $\sigma$ 抄到输出，损失就已经很低。它几乎不需要去理解画面里有什么，
训练信号也就没有逼着它在高噪声下学会"根据过去帧和动作猜出下一帧的结构"。
EDM 的目标在高噪声下接近干净画面本身（第 1.7 节的表），网络必须真的去猜。

**第五步：为什么 DDPM 在图像生成里没出问题，在世界模型里出问题。**

- 图像生成用几百上千步。高噪声阶段的误差，在后面大量的步数里被逐渐纠正掉。
- 世界模型只用几步，没有纠正的余地。更要命的是，这一帧留下的偏差会作为下一帧的条件输入模型，
  偏差喂偏差，就是第 5 节的自回归漂移。

所以这是两个因素叠加的结果：**少步放大了参数化的差别，自回归把每一帧的小偏差累积起来。**

**为什么是 3 步而不是 1 步。** EDM 用 1 步也稳定，但 1 步的输出是后验均值（第 3.4 节）。
环境确定时没问题；环境里有不可预测的成分时（论文举的例子是拳击游戏里对手的动作），
1 步会生成模糊的平均，多几步才能落到某一个具体的结果上。3 步是画面质量和想象速度之间的折中。

### 4.4 在扩散的梦里训练智能体

整体流程和 Phase 6 的 Dreamer 相同，三个环节轮流进行：

1. **收集**：用当前策略在真实环境里玩，把 (画面, 动作, 奖励, 终止) 存进回放数据
2. **学世界模型**：用回放数据更新扩散模型和奖励 / 终止模型
3. **学行为**：从回放数据里取真实的片段作为起点，在世界模型里想象 $H$ 步，用想象出的轨迹更新 actor-critic

**价值目标**用 Phase 4 的 λ-return。记想象出的奖励为 $\hat{r}_t$、终止标志为 $\hat{d}_t$（终止为 1），从后往前递推：

$$
\Lambda_t = \hat{r}_t + \gamma\, (1 - \hat{d}_t)\, \Big[ (1 - \lambda)\, V_\psi(\hat{o}_{t+1}) + \lambda\, \Lambda_{t+1} \Big],
\qquad \Lambda_{H} = V_\psi(\hat{o}_{H})
$$

**价值损失**是让 $V_\psi$ 回归 $\mathrm{sg}(\Lambda_t)$。

**策略损失**用 Phase 5 的 score-function 估计器（REINFORCE）加价值基线，再加熵正则：

$$
\mathcal{L}_\pi(\phi) = -\, \mathbb{E}_{\pi_\phi} \sum_{t=0}^{H-1} \Big[ \log \pi_\phi(a_t \mid \hat{o}_{\le t})\; \mathrm{sg}\big( \Lambda_t - V_\psi(\hat{o}_t) \big) + \eta\, \mathcal{H}\big[ \pi_\phi(\cdot \mid \hat{o}_{\le t}) \big] \Big]
$$

这里有一个和 Dreamer V1 不同的选择，值得停一下。Phase 5 讲过两类梯度估计器：

| 估计器 | 需要什么 | 谁在用 |
| --- | --- | --- |
| pathwise（沿模型反传） | 动作连续，且整条想象链路可微 | Dreamer V1 的连续控制 |
| score-function（REINFORCE） | 只需要能采样，不需要对世界模型求导 | Dreamer V2 / V3 的离散动作；IRIS；DIAMOND |

DIAMOND 用后者有两个原因：Atari 的动作是离散的；沿扩散模型反传意味着梯度要穿过
$H \times N$ 次 U-Net，代价很高。用 score-function 估计器时，世界模型只当一个黑盒采样器用。

**手算 λ-return。** 想象 3 步，奖励 $\hat{r}_0 = 0$、$\hat{r}_1 = 1$、$\hat{r}_2 = 0$，全程没有终止。
价值估计 $V(\hat{o}_1) = 0.5$、$V(\hat{o}_2) = 0.2$、$V(\hat{o}_3) = 0.4$。取 $\gamma = 0.99$、$\lambda = 0.95$。

- $\Lambda_3 = V(\hat{o}_3) = 0.4$
- $\Lambda_2 = 0 + 0.99 \times (0.05 \times 0.4 + 0.95 \times 0.4) = 0.99 \times 0.4 = 0.396$
- $\Lambda_1 = 1 + 0.99 \times (0.05 \times 0.2 + 0.95 \times 0.396) = 1 + 0.99 \times 0.3862 = 1.382$
- $\Lambda_0 = 0 + 0.99 \times (0.05 \times 0.5 + 0.95 \times 1.382) = 0.99 \times 1.338 = 1.325$

如果奖励模型把 $\hat{r}_1 = 1$ 这一次得分漏掉了（比如世界模型没把球画出来，得分事件自然也不会发生），
$\Lambda_1$ 变成 0.382，$\Lambda_0$ 变成 $0.99 \times (0.025 + 0.95 \times 0.382) = 0.384$。
一个小物体的缺失，让这段想象的回报估计从 1.325 掉到 0.384。"视觉细节重要"说的就是这条因果链：
**小物体丢了 → 奖励和后续画面错了 → 价值目标错了 → 策略学歪了。**

### 4.5 DIAMOND 的另一半：当游戏引擎

论文摘要的最后一句说，DIAMOND 的扩散世界模型可以单独拿出来当交互式的神经游戏引擎：
用一份固定的 Counter-Strike: Global Offensive 人类游戏录像（带键鼠动作）训练，
之后人可以用键盘鼠标在模型里"玩"。这部分没有智能体，也没有奖励模型，只有第 3 节的那个条件扩散模型。

这个演示（以及同期的 GameNGen）把扩散世界模型带出了强化学习的圈子：
世界模型不只是训练智能体的工具，它本身就可以是产品。第 9、10 节顺着这条线往下讲。

论文和项目主页也展示了失败模式：模型只看得到有限的几帧，视野之外的东西它不记得；
走到训练数据很少覆盖的地方，画面会逐渐崩坏。这两个问题分别是"记忆"和"漂移"，第 5 节讨论。

### 4.6 算法

```
算法：DIAMOND（一个训练周期）

 1. 收集：用策略 π_φ 在真实环境里交互若干步，存入回放数据 D
 2. 更新扩散模型 D_θ：
        从 D 取片段，按第 3.7 节的训练步做去噪损失
 3. 更新奖励 / 终止模型 R_ψ：
        从 D 取片段，用真实的奖励和终止标志做监督学习
 4. 更新 actor-critic（在想象里）：
        从 D 取真实的一小段作为起点（给 LSTM 预热，并填满扩散模型的上下文）
        对 t = 0 … H−1：
            a_t ~ π_φ(· | ô_≤t)
            ô_{t+1} ← 用 D_θ 做 N 步去噪生成（条件：最近 m 帧和动作）
            r̂_t，d̂_t ← R_ψ
        从后往前算 λ-return，更新价值头和策略头
```

第 4 步里"从真实片段出发"和 Dreamer 一致（Phase 6 讲过原因）：
起点覆盖了回放数据里的各种局面，而且每次想象只往前推 $H$ 步，模型的误差来不及累积太多。
这其实是 Phase 1 第 5.7 节"短视野 + 用真实状态清零误差"在训练阶段的用法。

### 4.7 代码对应

对应 `04-diamond.py`（讲义待写）。扩散模型部分复用第 3 节讲义的去噪器。

| 函数 / 类（规划中） | 做什么 |
| --- | --- |
| `RewardEndModel` | CNN + LSTM，输出奖励类别和终止概率 |
| `ActorCritic` | CNN + LSTM，策略头和价值头 |
| `imagine(wm, rew_end, actor_critic, init_frames, horizon)` | 在世界模型里展开，返回想象的轨迹 |
| `lambda_returns(rewards, ends, values, gamma, lam)` | λ-return 的反向递推 |
| `actor_critic_loss()` | 第 4.4 节的两项损失 |
| `train_diamond()` | 第 4.6 节的外层循环 |

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| 想象的画面 | `[B, H+1, C, H_img, W_img]` | 包含起点；第二维是想象步数 |
| 动作、对数概率、熵 | `[B, H]` | |
| 想象的奖励、终止 | `[B, H]` | |
| 价值 | `[B, H+1]` | 最后一个用于自举 |
| λ-return | `[B, H]` | |

想象一个批次的代价是 $B \times H \times N$ 次单帧去噪。取 $B = 32$、$H = 15$、$N = 3$ 是 1440 帧次，
批量化后是 $15 \times 3 = 45$ 次 U-Net 前向计算，每次的批大小是 32。

> 📌 待回填：缩小版 DIAMOND 在讲义所用环境上的学习曲线；想象回报与真实回报的对比；
> 1 步与 3 步去噪下想象画面的差别；与 Phase 6、Phase 8 同等交互步数下的对比，来自 `04-diamond.py`。

### 4.8 常见误区

- **误区：DIAMOND 比 IRIS 好，是因为扩散模型生成的画面更好看。**
  实际情况：论文的论点是画面里对决策重要的细节被保住了，策略因此拿到更准的奖励和更一致的画面。
  好看只是副产品。一个画面精美但把小球弄丢的模型，对智能体毫无帮助。
- **误区：在像素空间想象，所以策略必须在像素上学，效率一定很低。**
  实际情况：策略确实直接看生成的画面，但想象是批量并行的，而且每帧只要几步去噪。
  代价比隐空间想象高，但没有高到不可用。
- **误区：选 EDM 是因为它是更新的方法，换成 DDPM 多训练一会也一样。**
  实际情况：差别来自参数化在少步采样下的误差放大（第 4.3 节），不是训练时长能补上的。
  给 DDPM 足够多的去噪步数可以达到同样的画面质量，但想象的速度就没法接受了。
- **误区：DIAMOND 用了扩散模型，所以不再有复合误差。**
  实际情况：它和 Dreamer 一样靠"短视野、从真实片段出发"来控制误差。
  把想象的步数拉长，画面一样会跑偏。

---

## 5. 自回归漂移与 Diffusion Forcing

### 5.1 直觉

把一份文件复印一遍，再拿复印件去复印，如此反复。每一次复印只损失一点点清晰度，
但第 20 代复印件上的小字已经认不出来了。每一代的输入是上一代的输出，损失只会累积，不会恢复。

自回归的视频世界模型就是这台复印机。第 3.7 节展开算法的最后一行，把生成的帧放回了上下文：

```
  真实帧 ──▶ [模型] ──▶ 生成帧 1 ──▶ [模型] ──▶ 生成帧 2 ──▶ [模型] ──▶ 生成帧 3 ──▶ …
                          │                      │                      │
                      有一点瑕疵          在瑕疵上再加瑕疵        瑕疵继续累积
```

这叫**自回归漂移**（autoregressive drift）。它不是新问题，就是 Phase 1 第 5 节的复合误差，
只是换到了像素空间。那里的表现是"摆的角度越推越不准"，这里的表现是：

| 症状 | 样子 |
| --- | --- |
| 模糊累积 | 每一帧都比上一帧糊一点，最后变成一团色块 |
| 色彩和对比度漂移 | 画面整体越来越亮 / 越来越饱和 / 偏向某种颜色 |
| 结构崩坏 | 物体变形、融化、凭空出现或消失 |
| 语义漂移 | 场景慢慢变成另一个场景 |

另有一种常被混为一谈的问题叫**遗忘**：转过身再转回来，刚才的东西不见了或者变了样。
它的原因是上下文窗口有限，模型根本看不到那么久之前的画面，不是误差累积。两者的解法不同，要分开诊断。

### 5.2 定义：两种误差来源在像素空间的样子

Phase 1 第 5 节把问题拆成两个来源，这里一一对应：

| Phase 1 的说法 | 在视频世界模型里 |
| --- | --- |
| 误差累积：$e_{t+1} \le L\, e_t + \varepsilon$ | 上下文帧里的瑕疵被模型当真，带进下一帧，再叠加新的瑕疵 |
| 分布偏移：单步误差 $\varepsilon$ 只在训练数据覆盖的区域小 | 训练时上下文永远是真实帧；展开时上下文是生成帧，模型没见过这种输入 |

第二行在序列模型里有个专门的名字：训练时每一步的输入都是真实数据，叫 **teacher forcing**；
训练和推理时输入分布不一致，叫**曝光偏差**（exposure bias）。语言模型也有这个问题，
但文字是离散的，生成的 token 一定是词表里合法的 token。像素是连续的，
生成的画面可以落在"任何真实画面都不长这样"的地方，所以视频模型里这个问题重得多。

两个量决定漂移的快慢：

- $\varepsilon$：模型单步生成的质量
- $L$：模型对上下文扰动的敏感度。上下文里有一点瑕疵，输出里的瑕疵是它的多少倍

把单步质量做好是在降 $\varepsilon$。本节的几种办法主要是在降 $L$，以及消除分布偏移。

### 5.3 推导：噪声增强为什么有效

**做法。** 训练时故意把上下文帧弄脏：给每个上下文帧加上强度为 $\sigma_c$ 的高斯噪声
（$\sigma_c$ 每次随机取），并把 $\sigma_c$ 作为条件告诉模型。推理时，上下文是模型自己生成的帧，
告诉模型一个较小的 $\sigma_c$。这个技巧叫**噪声增强**（noise augmentation），
早先在级联扩散模型里叫 conditioning augmentation；GameNGen 用它实现了长时间稳定的自回归生成，
其摘要里专门提到这一点。

**为什么有效（一）：它把生成帧拉回了训练分布。** 设真实的上下文帧是 $c$，模型生成的是 $c + e$，$e$ 是瑕疵。
不加噪声时，模型在训练中只见过 $c$，从没见过 $c + e$。
加了噪声之后，训练时模型见到的是 $c + \sigma_c \eta$（$\eta$ 是标准高斯噪声），
推理时相当于见到 $c + e + \sigma_c \eta$。比较这两个分布，它们是方差相同、均值相差 $e$ 的两个高斯。
用 Phase 3 推过的两个高斯之间的 KL 公式（方差相同时只剩均值项）：

$$
D_{\mathrm{KL}}\big( \mathcal{N}(c + e,\ \sigma_c^2 I) \,\Vert\, \mathcal{N}(c,\ \sigma_c^2 I) \big) = \frac{\Vert e \Vert^2}{2\, \sigma_c^2}
$$

- $\sigma_c \to 0$（不加噪声）：只要 $e \ne 0$，KL 趋于无穷。生成帧完全在训练分布之外
- $\sigma_c$ 明显大于瑕疵的幅度：KL 很小。瑕疵被噪声"淹没"了，模型分不出这是真实帧还是生成帧

**手算。** 看一个像素，瑕疵 $e = 0.1$：

| 噪声增强的强度 $\sigma_c$ | KL $= 0.01 / (2\sigma_c^2)$ |
| --- | --- |
| 0.02 | 12.5 |
| 0.1 | 0.5 |
| 0.3 | 0.056 |

**为什么有效（二）：它降低了 $L$。** 上下文帧带着噪声时，模型没法照抄上下文里的高频细节
（那些细节被噪声盖住了），只能依赖结构性的信息：物体在哪、往哪动。
于是上下文里的小瑕疵对输出的影响变小了，也就是 $L$ 变小。

**用 Phase 1 的界看效果。** 下面的数字是为了说明界的形态而取的，不是实测值。
取单步误差 $\varepsilon = 0.01$，展开 50 步：

| 情形 | $L$ | $\varepsilon$ | 50 步后的误差上界 |
| --- | --- | --- | --- |
| 不做噪声增强 | 1.1 | 0.01 | $0.01 \times \dfrac{1.1^{50} - 1}{0.1} = 0.01 \times \dfrac{117.4 - 1}{0.1} = 11.6$ |
| 做噪声增强 | 0.95 | 0.02 | 不超过 $\dfrac{\varepsilon}{1 - L} = \dfrac{0.02}{0.05} = 0.4$ |

第二行特意把 $\varepsilon$ 取大了一倍：噪声增强是有代价的，上下文被弄脏之后可用的信息变少，单步预测会差一些。
但只要 $L$ 从大于 1 降到小于 1，误差就从指数增长变成有上限。**牺牲一点单步质量，换来长时间的稳定。**

推理时 $\sigma_c$ 取多大是一个要调的量：太小起不到作用，太大画面细节跟着丢。

### 5.4 Diffusion Forcing：每一帧有自己的噪声水平

噪声增强给上下文加了一个"脏的程度"的旋钮。Diffusion Forcing
（Chen et al. 2024，arXiv:2407.01392）把这个想法推到底：
**序列里的每一帧都有自己独立的噪声水平，模型要学会在任意的噪声水平组合下去噪。**

先看它统一了哪两种做法。把一段视频的各帧噪声水平排成一行：

```
   时间 ──────────────────────────────▶

   下一帧预测（teacher forcing）
   帧：    o_1    o_2    o_3    o_4
   噪声：   0      0      0      σ        历史全干净，只有最后一帧带噪

   全序列扩散
   帧：    o_1    o_2    o_3    o_4
   噪声：   σ      σ      σ      σ        所有帧同一个噪声水平，一起去噪

   Diffusion Forcing
   帧：    o_1    o_2    o_3    o_4
   噪声：  σ_1    σ_2    σ_3    σ_4       每帧独立随机取
```

| | 下一帧预测 | 全序列扩散 | Diffusion Forcing |
| --- | --- | --- | --- |
| 训练时的噪声水平 | 历史为 0，目标帧随机 | 所有帧相同 | 每帧独立随机 |
| 能生成任意长度吗 | 能，一帧一帧往后接 | 不能，长度固定 | 能 |
| 长时间展开稳定吗 | 不稳定（曝光偏差） | 不涉及 | 稳定 |
| 能对整段序列做引导吗 | 难 | 能 | 能 |

**把噪声看成软掩码。** Phase 8 的 MaskGIT 给每个 token 一个二值状态：可见，或被遮住。
噪声水平是它的连续版本：

- $\sigma_t = 0$：这一帧完全可见
- $\sigma_t = \sigma_{\max}$：这一帧完全被遮住（纯噪声，不含信息）
- 中间的值：这一帧"部分可见"，能看出大致结构，看不清细节

于是上面三种做法只是三种不同的遮挡图案，Diffusion Forcing 训练的是所有图案。

**定义。** 记第 $t$ 帧的噪声水平为 $\sigma_t$，带噪帧为 $\tilde{o}_t = o_t + \sigma_t \epsilon_t$。
模型一次接收整段带噪序列和每帧的噪声水平，输出每一帧的去噪结果。训练损失是各帧去噪损失之和：

$$
\mathcal{L}_{\text{DF}}(\theta) = \mathbb{E}_{o_{1:T},\ \sigma_{1:T},\ \epsilon_{1:T}}\;
\sum_{t=1}^{T} \lambda(\sigma_t)\, \Big\Vert D_\theta\big( \tilde{o}_{1:T};\ \sigma_{1:T},\ a_{1:T} \big)_t - o_t \Big\Vert^2,
\qquad \sigma_t \ \text{各帧独立采样}
$$

模型是因果的：第 $t$ 帧的输出只依赖 $\tilde{o}_{1:t}$。原论文用循环网络实现，后续工作多用带因果掩码的 Transformer。
论文证明了这个目标是在优化一个下界，这个下界覆盖了序列的所有子序列的似然（见论文摘要与理论部分）。本章不推导。

**和噪声增强的关系。** 噪声增强是"上下文帧取小噪声、目标帧取任意噪声"这一类图案。
Diffusion Forcing 的训练包含了它，所以天然具备噪声增强的稳定效果。
论文里稳定长展开的做法正是这样：把已经生成的帧当成"带一点噪声的帧"喂回去，
也就是告诉模型一个略大于 0 的噪声水平，让它不要完全相信这些帧的细节。

**图案的数量。** 假设噪声水平离散成 10 档，序列长 4 帧：

| 训练方式 | 训练时见到的噪声图案数 |
| --- | --- |
| 下一帧预测 | 10（只有最后一帧可变） |
| 全序列扩散 | 10（所有帧同步） |
| Diffusion Forcing | $10^4 = 10{,}000$ |

图案多了，任务更难学，但推理时的自由度也大得多。

### 5.5 采样：噪声水平的日程表是一张二维表

普通的扩散采样只需要一串递减的 $\sigma$。Diffusion Forcing 的采样需要一张表：
每一行是一轮去噪，每一列是一帧，格子里是这一轮结束后这一帧的噪声水平。
用 3 帧、3 个档位（2 = 纯噪声，1 = 半干净，0 = 干净）举例：

```
   ① 逐帧自回归              ② 全序列一起                ③ 金字塔
      帧1 帧2 帧3               帧1 帧2 帧3                 帧1 帧2 帧3
   起点  2   2   2            起点  2   2   2             起点  2   2   2
        1   2   2                  1   1   1                  1   2   2
        0   2   2                  0   0   0                  0   1   2
        0   1   2                                              0   0   1
        0   0   2                                              0   0   0
        0   0   1
        0   0   0
      6 轮                       2 轮                        4 轮
```

- **① 逐帧自回归**：一帧完全去噪后再开始下一帧。就是第 3.7 节的做法
- **② 全序列一起**：所有帧同步去噪。就是普通的视频扩散
- **③ 金字塔**：近处的帧先变干净，远处的帧保持更高的噪声水平。
  含义是"近的未来已经确定，远的未来先留着不确定"

第三种是新的能力。它符合规划的直觉：眼前这一步要执行，必须定下来；几步之后的事还可以改。
论文在此基础上做了对整段未来的引导采样，用于规划类任务。
Phase 11 的 Diffusion Policy 和"把视频预测当策略"会再碰到"对一段未来做扩散"这个想法。

### 5.6 其他缓解办法

| 办法 | 原理 | 代表 | 对应 Phase 1 第 5.7 节的哪一条 |
| --- | --- | --- | --- |
| 噪声增强 | 上下文加噪，降低敏感度，消除分布偏移 | GameNGen | 新办法 |
| 每帧独立噪声 | 把噪声增强一般化 | Diffusion Forcing | 新办法 |
| 训练时也展开 | 训练时就让模型以自己生成的帧为条件 | Self Forcing（Huang et al. 2025，arXiv:2506.08009） | 多步损失 |
| 短视野 + 真实帧重新出发 | 每次只想象十几步，起点是真实数据 | DIAMOND、Dreamer | 短视野 + 重新规划 |
| 用历史做引导 | 把"看历史"和"不看历史"两种预测的差放大 | History-Guided Video Diffusion（Song et al. 2025，arXiv:2502.06764） | 新办法 |
| 换预测空间 | 在压缩后的隐空间里展开，高频细节不参与累积 | 第 6 节；Phase 10 | 换预测空间 |
| 更长的上下文 / 外部记忆 | 治的是遗忘，不是漂移 | 各类长上下文视频模型 | 不适用 |

"训练时也展开"和 Phase 1 的多步损失是同一个思路：既然使用时要自回归，训练时就把自回归的后果算进损失。
Self Forcing 的摘要把它表述为弥合训练与推理之间的差距：训练时每一帧以模型此前自己生成的帧为条件，
而不是以真实帧为条件。Phase 11 讲模仿学习的分布偏移时，DAgger 也是同一类办法。

"用历史做引导"建立在 Diffusion Forcing 之上：因为训练时历史帧可以被噪声完全遮住，
同一个模型既会"看历史"也会"不看历史"，不用额外训练就能做第 3.3 节那种引导，只是这回被引导的条件是历史帧。

### 5.7 算法

```
算法：Diffusion Forcing 训练

重复：
 1. 取一段序列：帧 o_{1:T}，动作 a_{1:T}                       [B, T, C, H, W]
 2. 每个样本的每一帧独立采样噪声水平 σ_t                        [B, T]
 3. 加噪：õ_t = o_t + σ_t · ε_t
 4. 因果模型一次前向，得到每一帧的去噪结果                       [B, T, C, H, W]
 5. 损失 = 各帧去噪损失之和（每帧用自己的 σ_t 算预条件系数和权重）
 6. 更新 θ
```

```
算法：带稳定化的自回归展开

输入：真实的起始帧；稳定化噪声水平 σ_stab（一个小的正数）
 1. history ← 真实的起始帧，对应的噪声水平标为 0
 2. 对 j = 1 … H：
        x ← σ_0 · ε
        对 i = 0 … N−1：
            x̂ ← D_θ(history 与 x 拼成的序列；噪声水平 = history 各帧的标记 与 σ_i)
            x ← x̂ + (σ_{i+1} / σ_i) · (x − x̂)
        把 x 追加进 history，噪声水平标为 σ_stab            关键：生成帧不标 0
 3. 返回生成的帧
```

### 5.8 代码对应

对应 `05-autoregressive-drift-and-diffusion-forcing.py`（讲义待写）。

| 函数（规划中） | 做什么 |
| --- | --- |
| `measure_drift(model, real_clips, n_steps)` | 开环展开，记录误差随步数的曲线 |
| `noise_augment_context(context, sigma_c)` | 给上下文帧加噪，返回带噪上下文和 $\sigma_c$ |
| `sample_per_frame_sigmas(batch, n_frames)` | 每帧独立采样噪声水平 |
| `diffusion_forcing_loss()` | 第 5.7 节的训练步 |
| `make_schedule_matrix(mode, n_frames, n_levels)` | 生成第 5.5 节的三种日程表 |
| `rollout_with_stabilization()` | 带稳定化的展开 |

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `frames`，`noisy_frames` | `[B, T, C, H, W]` | 干净序列、带噪序列 |
| `sigmas` | `[B, T]` | 每帧一个噪声水平。和第 3 节的 `[B]` 对比 |
| `schedule` | `[R, T]` | 采样日程表，$R$ 是去噪轮数 |
| 漂移曲线 | `[H]` | 每个展开步数上的平均误差 |

> 📌 待回填：三条开环误差曲线（不做处理、噪声增强、Diffusion Forcing）随展开步数的变化，
> 以及推理时不同稳定化噪声水平的影响，来自 `05-autoregressive-drift-and-diffusion-forcing.py`。
> 定性预期：不做处理的曲线增长最快；噪声增强的单步误差略高但长期更低。

### 5.9 常见误区

- **误区：漂移是扩散模型特有的问题。**
  实际情况：任何自回归的世界模型都有，Phase 1 的 MLP、Phase 3 的 RSSM、Phase 8 的 Transformer 都有。
  扩散模型在像素空间里展开，生成的画面可以偏到任何地方，所以表现得最直观。
- **误区：把单步生成质量做到足够高，漂移就没有了。**
  实际情况：这是在降 $\varepsilon$。只要 $L > 1$，误差仍然指数增长，降 $\varepsilon$ 只能把崩坏推迟若干步
  （Phase 1 第 5.9 节算过：单步误差降 10 倍，可用的步数只多出 $\log 10 / \log L$ 步）。
- **误区：噪声增强是在推理时往画面上加噪点，所以画质会变差。**
  实际情况：噪声主要是训练时加的。推理时可以只把"噪声水平"这个条件设成一个小的正数，不必真的往画面上加噪。
  代价是模型对上下文细节的利用变弱，不是画面上出现噪点。
- **误区：Diffusion Forcing 是一种新的网络结构。**
  实际情况：它是一种训练方式（每帧独立噪声水平）加一套采样日程。骨干可以是 RNN、U-Net 或 Transformer。

---

## 6. 视频 tokenizer 与隐空间扩散

### 6.1 直觉

没有人直接传输原始视频。摄像头拍下的画面先经过编码器压成码流，传输和存储的都是码流，播放时再解码。
压缩之所以可行，是因为视频里绝大部分内容是冗余的：相邻像素差不多，相邻帧也差不多。

世界模型同理。DIAMOND 在像素空间里做扩散，对 64×64 的 Atari 画面没问题；
分辨率一高，每一步去噪都要处理几十万个数，算不动。
办法是把问题拆成两层，像视频系统里"编解码器"和"业务逻辑"分开一样：

```
   像素视频 ──▶ ┌──────────┐ ──▶ 隐变量序列 ──▶ ┌──────────────┐ ──▶ 未来的隐变量 ──▶ ┌──────────┐ ──▶ 像素视频
               │ 编码器 E  │                  │ 世界模型       │                    │ 解码器 G  │
               └──────────┘                  │ （扩散 / token）│                    └──────────┘
                                             └──────────────┘
   第一阶段：只训练 E 和 G（重建）       第二阶段：冻结 E 和 G，只训练中间的世界模型
```

负责压缩的这一对编码器、解码器，就是**视频 tokenizer**。Phase 8 已经学过图像的 tokenizer，这里有两处不同：

- 压缩的对象从单帧变成视频，可以在**时间**上也做压缩
- 输出不一定是离散的 token。给扩散模型用的 tokenizer 输出**连续**的隐变量

"tokenizer"这个词在这里用得比较宽：不管输出是离散还是连续，把视频压成紧凑表示的模块都这么叫。

### 6.2 定义

- **编码器** $E$：把视频 `[B, T, 3, H, W]` 压成隐变量 `[B, T', C_z, h, w]`
- **解码器** $G$：把隐变量还原成视频
- **空间压缩率** $f_s = H / h$；**时间压缩率** $f_t \approx T / T'$
- **因果的 tokenizer**：第 $t$ 个隐变量只依赖第 $t$ 帧及之前的画面，不看未来

两种输出形式：

| | 离散 token | 连续隐变量 |
| --- | --- | --- |
| 怎么得到 | VQ、FSQ 等量化（Phase 8） | 带轻微正则的自编码器（KL 正则的 VAE 等） |
| 配什么世界模型 | 自回归 Transformer、MaskGIT | 扩散模型、flow matching |
| 世界模型的损失 | 交叉熵 | 去噪损失 / 速度损失 |
| 信息损失 | 较大：每个位置只能取码本里的一个值 | 较小：每个位置是若干个实数 |
| 生成结果一定"合法"吗 | 是。输出一定是码本里的 token | 否。隐变量可以漂到解码器没见过的地方 |
| 代表 | Genie、IRIS（Phase 8）、MAGVIT-v2（arXiv:2310.05737） | Latent Diffusion（Rombach et al. 2022，arXiv:2112.10752）、Cosmos 的连续 tokenizer |

**隐空间扩散**（latent diffusion）就是把第 1–5 节的所有做法原封不动地搬到隐变量 $z$ 上：
把公式里的 $x$ 换成 $z = E(o)$，采样结束后用 $G$ 解码成画面。

### 6.3 推导：几笔账

**第一笔：压缩了多少。** 一段 17 帧、256×256 的 RGB 视频：

$$
17 \times 256 \times 256 \times 3 = 3{,}342{,}336 \ \text{个数}
$$

用一个空间压缩 8 倍、时间压缩 4 倍、4 个通道的因果 tokenizer。因果的时间压缩通常把第 1 帧单独编码，
之后每 4 帧合成一个隐帧，所以 17 帧变成 $1 + 16/4 = 5$ 个隐帧：

$$
5 \times 32 \times 32 \times 4 = 20{,}480 \ \text{个数}, \qquad \text{压缩比} = \frac{3{,}342{,}336}{20{,}480} \approx 163
$$

世界模型每一步去噪处理的数据量降了两个数量级。Phase 1 第 6.5 节和本章第 3.5 节的账，每一项都跟着降。

**第二笔：隐变量的尺度必须对上。** 第 1.7 节的预条件系数里处处是 $\sigma_{\text{data}}$。
在像素上它大约是 0.5；隐变量的标准差是 tokenizer 训练出来的，可能是任何值。

假设隐变量的标准差实际是 4，而代码里还沿用 $\sigma_{\text{data}} = 0.5$ 和 EDM 默认的噪声分布（$\ln\sigma$ 的中位数对应 $\sigma = 0.30$）：

| | 设计意图 | 实际发生的 |
| --- | --- | --- |
| 训练时典型的噪声与数据之比 | $0.30 / 0.5 = 0.6$ | $0.30 / 4 = 0.075$ |
| $c_{\text{in}}$ 之后的输入标准差（$\sigma = 0.30$） | 1 | $\sqrt{16 + 0.09} / \sqrt{0.25 + 0.09} = 6.9$ |

训练几乎全落在"噪声远小于数据"的区间，模型没怎么练过高噪声下的去噪；网络的输入也不再是单位方差。
修法有两个，效果相同：把隐变量除以它的标准差再做扩散；或者把 $\sigma_{\text{data}}$ 设成隐变量的实际标准差。

**第三笔：重建质量是上限。** 世界模型预测得再准，画面也要经过解码器 $G$ 才能看到。所以：

$$
\text{世界模型的画面质量} \ \le\ \text{tokenizer 的重建质量}
$$

这和 Phase 8 的结论相同。第 4 节"小物体被丢掉"的问题，在这里换了个位置重新出现：
连续隐变量丢的信息比离散 token 少，但压缩率开得太高，小球一样会消失。
检查办法也相同：先只跑 $G(E(o))$，看任务相关的细节还在不在。
GameNGen 就为此单独微调过解码器，以保住画面里的小细节和界面元素（见该论文摘要）。

**第四笔：时间压缩与动作对齐。** 时间压缩 4 倍意味着一个隐帧对应 4 个原始帧，也就对应 4 个动作。两个后果：

- 动作条件要按隐帧分组：每个隐帧的条件是一小段动作（4 个），不是一个
- 交互的粒度变粗：模型一次生成 4 帧，用户的操作最快也要到下一个隐帧才起作用。
  10 Hz 的画面配 4 倍时间压缩，响应的粒度就是 0.4 秒

所以追求"跟手"的交互式世界模型，时间压缩通常开得很低或者不开，主要靠空间压缩；
以离线生成为主的视频基础模型，才会用较高的时间压缩。

**第五笔：为什么交互式模型需要因果的 tokenizer。** 非因果的 tokenizer 编码第 $t$ 帧时会看后面的帧。
交互时后面的帧还不存在（它们取决于用户接下来按什么键），所以编码和解码都必须只依赖过去。

### 6.4 算法

```
算法：两阶段训练隐空间扩散世界模型

第一阶段：训练 tokenizer
 重复：
   取视频片段 o                                      [B, T, 3, H, W]
   z ← E(o)；ô ← G(z)
   损失 = 重建损失（像素 + 感知）+ 轻微的正则（让 z 的分布规整）
 结束后：冻结 E、G；统计 z 的标准差 σ_z

第二阶段：训练世界模型
 重复：
   取片段和动作；z ← E(o) / σ_z（不回传梯度；可以预先算好存盘）   [B, T', C_z, h, w]
   在 z 上做第 3 节或第 5 节的训练步（动作按隐帧分组）

推理：
   把起始帧编码成 z → 在隐空间里自回归展开 → 每生成一个隐帧就解码显示
```

第二阶段可以把 $z$ 预先算好存到硬盘。训练世界模型时不再碰像素，速度快很多。
Phase 10 的项目对冻结的预训练特征也是这么做的。

### 6.5 代码对应

对应 `06-video-tokenizers-and-latent-diffusion.py`（讲义待写）。

| 函数 / 类（规划中） | 做什么 |
| --- | --- |
| `CausalVideoAutoencoder` | 小型因果视频自编码器，`encode()` 和 `decode()` |
| `tokenizer_loss()` | 第一阶段的重建损失 |
| `estimate_latent_std()` | 统计隐变量的标准差 |
| `encode_dataset()` | 把整个数据集预先编码存盘 |
| `LatentFrameDenoiser` | 第 3 节的去噪器，输入换成隐变量 |
| `latent_rollout()` | 在隐空间展开并解码 |

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `video` | `[B, T, 3, H, W]` | 原始视频 |
| `latents` | `[B, T', C_z, h, w]` | 隐变量。$T' = 1 + (T - 1)/f_t$，$h = H / f_s$ |
| `actions_grouped` | `[B, T', f_t, A]` | 每个隐帧对应的一小段动作 |

> 📌 待回填：tokenizer 的重建 PSNR；同一个任务上像素空间扩散与隐空间扩散的单步耗时和预测质量对比；
> 压缩率调高时小物体是否还在，来自 `06-video-tokenizers-and-latent-diffusion.py`。

### 6.6 常见误区

- **误区：tokenizer 的输出一定是离散的 token。**
  实际情况：给扩散模型用的 tokenizer 输出连续隐变量。名字沿用了，含义变宽了。
- **误区：压缩率越高越好，世界模型越快。**
  实际情况：压缩率是速度和上限之间的交易。丢掉的信息世界模型永远补不回来。
- **误区：隐空间扩散和像素空间扩散的代码一样，换个输入就行。**
  实际情况：算法一样，但隐变量的尺度、时间压缩后的动作对齐、因果性这三处都要改，哪一处漏了都会出问题。
- **误区：在隐空间里展开就不会漂移。**
  实际情况：漂移照样发生，只是发生在隐变量上。连续隐变量还可能漂到解码器没见过的区域，
  解码出来的画面会出现成片的伪影。第 5 节的办法在隐空间里同样需要。

---

## 7. 隐动作模型：从无标注视频里学动作

### 7.1 直觉

到这里为止，训练数据都是 (画面, 动作) 成对出现的。这种数据很少：
要么自己跑环境采，要么找人戴着设备录。而没有动作标签的视频要多得多，
网上的游戏录像、第一人称视频、机器人演示的旁观录像，数量差了好几个数量级。

没有动作标签，能不能学出"可以操控"的世界模型？

看别人打游戏的录像时，你看不到手柄，但能看出他按了什么：角色往右走了，是按了右；角色跳起来了，是按了跳。
依据是**相邻两帧之间的变化**。这个从"前后两帧"反推"中间做了什么"的过程，就是隐动作模型要学的东西。

类比：两个版本的代码之间有一个 diff。现在规定 commit message 只能从 8 个固定的词里选一个。
要让别人凭"旧版本 + 这一个词"就尽量还原出新版本，你会怎么定这 8 个词？
一定是让它们对应 8 种最常见、最不能从旧版本推断出来的改动类型。
隐动作模型做的就是这件事：旧版本是过去的帧，新版本是下一帧，那个词就是**隐动作**（latent action）。

### 7.2 定义

隐动作模型（Latent Action Model，LAM）由两个网络和一个码本组成：

```
   o_1 … o_t ──┐
               ├──▶ ┌────────────┐      连续向量      ┌──────────┐   隐动作 ā_t
   o_{t+1} ────┘    │ 编码器      │ ───────────────▶ │ VQ 码本   │ ──（M 个码之一）──┐
                    │ （看得到未来）│                  │ M 很小    │                 │
                    └────────────┘                  └──────────┘                 │
                                                                                 ▼
   o_1 … o_t ─────────────────────────────────────────────────────────▶ ┌────────────┐
                                                                        │ 解码器      │ ──▶ ô_{t+1}
                                                                        │（看不到未来）│
                                                                        └────────────┘
```

| 部件 | 输入 | 输出 | 相当于 |
| --- | --- | --- | --- |
| 编码器 | 过去的帧 $o_{1:t}$ **和下一帧** $o_{t+1}$ | 一个连续向量 | 逆动力学模型：看前后两帧，推断中间的动作 |
| VQ 码本 | 连续向量 | 最近的码的编号 $\bar{a}_t \in \lbrace 1, \dots, M \rbrace$ | 一个极窄的信息瓶颈 |
| 解码器 | 过去的帧 $o_{1:t}$ 和隐动作 $\bar{a}_t$ | 预测的下一帧 $\hat{o}_{t+1}$ | 正向动力学模型：给画面和动作，预测结果 |

训练损失是 Phase 8 的 VQ-VAE 损失，只是重建的对象换成了下一帧。
记编码器输出为 $e_t$，量化后的码向量为 $c_t$：

$$
\mathcal{L}_{\text{LAM}} = \underbrace{\big\Vert \hat{o}_{t+1} - o_{t+1} \big\Vert^2}_{\text{重建下一帧}}
+ \underbrace{\big\Vert \mathrm{sg}(e_t) - c_t \big\Vert^2}_{\text{码本向编码器靠拢}}
+ \beta\, \underbrace{\big\Vert e_t - \mathrm{sg}(c_t) \big\Vert^2}_{\text{编码器向码本靠拢}}
$$

量化那一步不可导，梯度用 straight-through 直接从 $c_t$ 传回 $e_t$（Phase 8 推过）。

这里有一个不对称的设计，是整个方法的关键：**编码器看得到未来，解码器看不到。**
解码器想把下一帧预测准，唯一的额外信息来源就是隐动作。而隐动作只有 $M$ 种取值。

### 7.3 推导：瓶颈为什么必须很窄

**第一步：瓶颈能装多少信息。** 一个有 $M$ 种取值的离散码，最多携带 $\log_2 M$ 比特。
Genie 的隐动作码本大小是 8（论文方法部分），也就是每次转移 3 比特。
对比一帧 64×64 的 RGB 画面，每个数 8 比特：

$$
64 \times 64 \times 3 \times 8 = 98{,}304 \ \text{比特}
$$

编码器明明看到了完整的下一帧，却只被允许传 3 比特给解码器。它不可能把下一帧"抄"过去。

**第二步：这 3 比特会被花在哪。** 解码器已经有了过去的帧，下一帧的大部分内容它自己就能预测
（背景不变，运动的物体继续运动）。剩下它预测不了的部分，记作残差。
损失是残差的平方，所以码本的最优用法是：**把所有可能的残差分成 $M$ 组，让每组内部尽量相似。**
这就是对"过去帧解释不了的变化"做 k-means 聚类。

**第三步：过去帧解释不了的变化，主要是什么。** 在有一个被操控的角色的视频里，
最大的、最系统的不可预测成分，就是操控者这一步选了什么动作。于是聚出来的类对应动作。

**手算例子。** 一个一维世界：方块的位置每一步变化 $\Delta p$，由动作决定。
动作有三种，各占三分之一：左（$\Delta p = -1$）、不动（$\Delta p = 0$）、右（$\Delta p = +1$）。
解码器预测 $\Delta p$，看不同的码本大小 $M$ 下能达到的最小平方误差：

| $M$ | 最优分组 | 每组的预测值 | 平方误差的期望 |
| --- | --- | --- | --- |
| 1（没有隐动作） | 全部一组 | 0 | $(1 + 0 + 1)/3 = 0.667$ |
| 2 | {左}，{不动、右} | −1，0.5 | $(0 + 0.25 + 0.25)/3 = 0.167$ |
| 3 | 每种动作一组 | −1，0，+1 | 0 |
| 更大 | 多出来的码没有可解释的东西 | | 0 |

$M = 3$ 时，三个码恰好对应三种真实动作，尽管训练中从没出现过动作标签。

**第四步：瓶颈太宽会怎样。** 如果世界里还有随机的成分（比如背景里闪烁的星星），$M$ 开大之后，
多出来的码会被用来编码这些东西，因为这样能进一步降低重建误差。$M$ 再大下去，
隐动作就成了下一帧的一份压缩摘要。此时解码器不再需要理解"动作导致了什么"，照着摘要画就行。
用户随便选一个码，得到的不是"向右走"，而是"某一帧画面的内容"，完全不可控。

所以瓶颈的宽度是一个硬约束：

| 瓶颈 | 后果 |
| --- | --- |
| 太窄 | 不同的真实动作被合并成一个码（上表 $M = 2$），可控的自由度不够 |
| 合适 | 码与真实动作大致对应 |
| 太宽 | 码里混进了画面内容，编码器在"泄露未来"，可控性崩溃 |

### 7.4 隐动作学到的不一定是"动作"

损失函数只关心"哪些变化最难从过去预测、又最影响重建误差"，它不知道什么是智能体。所以：

- 如果画面里**有比角色动作更显眼的不可预测变化**（镜头晃动、场景切换、别的角色、闪烁的背景），
  码本会优先分给它们。这是隐动作模型已知的主要弱点，干扰物多的真实视频上尤其明显
- 隐动作的含义可能随场景变化：同一个码在这个场景里是"向右"，换个场景未必
- 码的编号没有任何先验含义。哪个码是"跳"，要试了才知道

Genie 论文报告，它学到的隐动作在不同的提示画面之间含义基本一致（比如某个码始终是"向右"）。
这是在特定数据（二维平台游戏，镜头跟随角色）上观察到的经验性质，不是目标函数保证的。

**怎么检查。** 如果手头有一小份带真实动作的数据，可以统计隐动作和真实动作的对应关系：

| 真实动作 \ 隐动作 | 码 0 | 码 1 | 码 2 |
| --- | --- | --- | --- |
| 左 | 90 | 5 | 5 |
| 不动 | 10 | 80 | 10 |
| 右 | 0 | 15 | 85 |

每一行取最大的格子：$(90 + 80 + 85) / 300 = 85\%$ 的转移被分到了"正确"的码上（上表是示意数字，不是实测）。
没有任何真实动作时，就用第 7.6 节的可控性指标。

### 7.5 隐动作怎么用

学出隐动作之后有三种用法：

| 用法 | 做法 | 代表 |
| --- | --- | --- |
| 直接当手柄 | 人从 $M$ 个码里选，世界模型生成下一帧。选几次就能摸清每个码的含义 | Genie（第 8 节） |
| 映射到真实动作 | 用一小份带标签的数据，学"隐动作 → 真实动作"的映射 | Genie 论文里的模仿学习实验 |
| 当作预训练目标 | 先让策略学着预测隐动作（数据是海量无标注视频），再用少量机器人数据微调到真实动作 | LAPA（Ye et al. 2024，arXiv:2410.11758） |

**另一条路线：逆动力学模型打伪标签。** VPT（Baker et al. 2022，arXiv:2206.11795）不发明隐动作。
它先用少量带标签的数据训练一个逆动力学模型（Inverse Dynamics Model，IDM）
$p(a_t \mid o_t, o_{t+1})$，再用它给海量无标注视频打上动作标签，然后在这些伪标签上做模仿学习。

| | 隐动作模型（LAM） | 逆动力学模型（IDM） |
| --- | --- | --- |
| 需要带标签的数据吗 | 不需要（要对应到真实动作时才需要一点） | 需要一小份 |
| 学出的动作空间 | 自己发明的离散码 | 真实的动作空间 |
| 跨本体吗 | 能。不同机器人、人手的视频可以共用一套隐动作 | 不能。动作空间绑定在采集标签的那个本体上 |
| 主要风险 | 码里混进无关变化；含义不稳定 | 无标注视频和带标签数据分布不同，伪标签不准 |

两者其实是同一个结构的两种训练方式：LAM 的编码器就是一个没有标签监督的 IDM。
**逆动力学**这条线到 Phase 11 会成为主角：那里"把视频预测当策略"的方法，
先让视频模型生成未来的画面，再用逆动力学模型从画面里把动作解出来。

### 7.6 怎么衡量可控性

Genie 论文提出了一个不需要真实动作标签的指标 $\Delta$PSNR。对一段真实视频，生成两个版本：

- $\hat{o}$：用 LAM 编码器从这段真实视频里推断出的隐动作来生成
- $\hat{o}'$：用随机抽的隐动作来生成

$$
\Delta\text{PSNR} = \text{PSNR}(o, \hat{o}) - \text{PSNR}(o, \hat{o}')
$$

含义：如果隐动作真的在控制画面，用对的隐动作应该比用随机的隐动作更接近真实视频，差值为正且大。
如果世界模型忽略隐动作，两个版本差不多，差值接近 0。

**手算。** 像素取 $[0, 1]$，$\text{PSNR} = -10 \log_{10}(\text{MSE})$。
推断隐动作的版本 MSE 为 0.002，随机隐动作的版本 MSE 为 0.004：

$$
\Delta\text{PSNR} = -10 \log_{10}(0.002) + 10 \log_{10}(0.004) = 27.0 - 24.0 = 3.0 \ \text{dB}
$$

误差每差一倍，$\Delta$PSNR 差 3 dB。

### 7.7 算法

```
算法：训练隐动作模型

重复：
 1. 取一段无标注视频 o_{1:T}                              [B, T, C, H, W]
 2. 编码器看整段视频，对每个相邻帧对输出一个连续向量 e_t      [B, T−1, D]
 3. 量化：ā_t ← 最近的码的编号；c_t ← 该码的向量             [B, T−1]，[B, T−1, D]
 4. 解码器对每个 t：输入 o_{1:t} 和 c_t（straight-through），输出 ô_{t+1}
 5. 损失 = 重建损失 + 码本损失 + β · commitment 损失
 6. 更新编码器、解码器、码本

训练结束后：
   给视频打标签时：只用编码器 + 码本
   交互时：编码器不用了，只保留码本（用户直接给出码的编号）
```

最后两行容易被忽略：LAM 的解码器只是训练时的辅助，生成画面的活最后由另一个更强的动力学模型来干（第 8 节）。

### 7.8 代码对应

对应 `07-latent-action-models.py`（讲义待写）。

| 函数 / 类（规划中） | 做什么 |
| --- | --- |
| `LatentActionEncoder` | 输入相邻帧，输出连续向量 |
| `VectorQuantizer` | 复用 Phase 8 的实现，码本很小 |
| `LatentActionDecoder` | 输入过去帧和隐动作，输出下一帧 |
| `lam_loss()` | 第 7.2 节的三项损失 |
| `label_video(frames)` | 给一段视频打上隐动作标签 |
| `latent_vs_true_action_table()` | 第 7.4 节的对应关系表 |
| `delta_psnr()` | 第 7.6 节的可控性指标 |

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `frames` | `[B, T, C, H, W]` | 无标注视频 |
| `e` | `[B, T-1, D]` | 编码器输出 |
| `latent_actions` | `[B, T-1]` | 每个转移一个码的编号，取值 $0 \dots M-1$ |
| `codebook` | `[M, D]` | $M$ 取个位数 |

> 📌 待回填：在带真实动作的环境上只用画面训练 LAM，得到的隐动作与真实动作的对应关系表；
> 码本大小取 2、4、8、64 时可控性的变化，来自 `07-latent-action-models.py`。
> 定性预期：码本过大时，随机选码生成的画面不再对应一致的运动。

### 7.9 常见误区

- **误区：隐动作模型学到的就是真实的动作。**
  实际情况：它学到的是"对最难预测的帧间变化的一种离散分类"。在合适的数据上这个分类和动作大致重合，
  不保证一一对应，也可能把镜头运动、别的物体的运动编进去。
- **误区：码本大一点更保险，能表达更丰富的动作。**
  实际情况：码本的大小就是瓶颈的宽度。开大了，编码器会把未来画面的内容塞进去，可控性反而崩溃（第 7.3 节）。
- **误区：既然编码器能看到下一帧，这是信息泄露，训练是作弊。**
  实际情况：编码器看未来是设计的一部分，瓶颈保证了它只能传很少的信息。
  交互时编码器根本不参与，隐动作由用户给出。
- **误区：有了隐动作，就不再需要任何带动作标签的数据了。**
  实际情况：只是让人"玩"的话确实不需要。要让机器人执行，隐动作必须对应到真实的电机指令，
  这一步仍然需要带标签的数据，只是需要得很少。

---

## 8. Genie：生成式交互环境

### 8.1 直觉

一台游戏机可以拆成三块：负责画面编解码的显示部分，定义"有哪几个按键"的手柄，
以及根据当前画面和按键算出下一帧的游戏引擎。

Genie（Bruce et al. 2024，arXiv:2402.15391）把这三块全部从视频里学出来，
训练数据里没有一行游戏代码，也没有一条按键记录：

| 游戏机的部件 | Genie 的部件 | 本章哪里讲过 |
| --- | --- | --- |
| 显示：画面与紧凑表示互转 | 视频 tokenizer | 第 6 节；Phase 8 的 VQ-VAE |
| 手柄：定义有哪几个按键 | 隐动作模型 | 第 7 节 |
| 引擎：(画面, 按键) → 下一帧 | 动力学模型 | Phase 8 的 MaskGIT |

训练完成后，给它一张从没见过的图（一张照片、一幅手绘草图、一张生成的图），
它把这张图当作游戏的第一帧，人每选一个隐动作，它生成下一帧。
论文摘要把它称为第一个以无监督方式从无标注的互联网视频中训练出来的**生成式交互环境**。

### 8.2 定义：三个部件

```
   训练（只有视频，没有动作标签）

   视频 o_{1:T} ──▶ ┌───────────────┐ ──▶ 视频 token z_{1:T} ─────────────┐
     │             │ ① 视频 tokenizer │                                  ▼
     │             └───────────────┘                         ┌──────────────────┐
     │                                                       │ ③ 动力学模型       │ ──▶ 预测 z_{2:T}
     │             ┌───────────────┐                         │   MaskGIT         │     （交叉熵）
     └───────────▶ │ ② 隐动作模型    │ ──▶ 隐动作 ā_{1:T−1} ──▶ └──────────────────┘
       （直接看像素） └───────────────┘


   推理（人来操控）

   提示图像 ──▶ tokenizer 编码 ──▶ z_1 ──┐
                                        ├──▶ 动力学模型 ──▶ z_2 ──▶ tokenizer 解码 ──▶ 画面
   人选的隐动作 ā_1（M 个之一）───────────┘        │
                                                └──▶ 把 z_2 放回上下文，等下一个隐动作 …
```

| 部件 | 结构 | 训练目标 | 推理时 |
| --- | --- | --- | --- |
| ① 视频 tokenizer | VQ-VAE，编码器和解码器是时空 Transformer，时间上因果 | 重建视频（Phase 8 的 VQ 损失） | 编码提示图像；解码生成的 token |
| ② 隐动作模型 | 第 7 节的结构，码本很小 | 重建下一帧 | 只保留码本，编码器和解码器都丢掉 |
| ③ 动力学模型 | 只有解码器的时空 Transformer，用 MaskGIT 的方式训练和采样 | 被遮住的 token 的交叉熵 | 每一帧做若干轮 MaskGIT 并行解码 |

**训练顺序。** ① 和 ② 互不依赖，可以分别训练（都只需要视频）。
它们训练好并冻结之后，才训练 ③：用 ① 把视频变成 token，用 ② 给每个转移打上隐动作标签，
③ 学的是"给定过去的 token 和隐动作，预测下一帧的 token"。

两个设计细节（出自论文的方法与消融部分）：

- **隐动作模型直接看像素，不看 token。** 论文的消融发现这样可控性更好。
  原因不难理解：tokenizer 的压缩可能已经丢掉了一部分和运动有关的细节
- **隐动作以加性嵌入的方式进入动力学模型**（加到对应帧的 token 嵌入上），而不是作为额外的 token 拼进序列。
  论文报告这样可控性更好

### 8.3 推导：时空 Transformer 的注意力账

视频的 token 数是帧数乘以每帧的 token 数。设一段视频有 $T$ 帧，每帧 $S$ 个 token。

**全注意力。** 所有 $T \cdot S$ 个 token 两两做注意力：

$$
\text{代价} \propto (T \cdot S)^2
$$

**时空分解的注意力**（Genie 的 ST-Transformer）。每个块里做两次注意力：

- 空间注意力：每一帧内部的 $S$ 个 token 之间。共 $T$ 帧，代价 $T \cdot S^2$
- 时间注意力：同一个空间位置在不同帧之间，带因果掩码。共 $S$ 个位置，代价 $S \cdot T^2$

$$
\text{代价} \propto T \cdot S^2 + S \cdot T^2 = T \cdot S \cdot (S + T)
$$

```
   空间注意力：帧内                       时间注意力：同一位置跨帧（只看过去）

   帧 t：  ■ ■ ■ ■                        帧 1    帧 2    帧 3
          ■ ■ ■ ■   ← 这 S 个 token         ■ ────▶ ■ ────▶ ■   ← 位置 (i, j)
          ■ ■ ■ ■     互相做注意力           
          ■ ■ ■ ■                          每个位置各自沿时间做注意力
```

**手算。** $T = 16$ 帧，每帧 $S = 64$ 个 token：

| | 计算 | 注意力对数 |
| --- | --- | --- |
| 全注意力 | $(16 \times 64)^2 = 1024^2$ | 1,048,576 |
| 时空分解 | $16 \times 64^2 + 64 \times 16^2 = 65{,}536 + 16{,}384$ | 81,920 |

省了约 12.8 倍。更重要的是增长方式：帧数翻倍时，全注意力的代价变成 4 倍；
时空分解里占大头的空间项只变成 2 倍。这对视频这种"帧数可以一直加"的数据很关键。

代价是每个 token 不能在一层之内直接看到"另一帧的另一个位置"，需要叠两层（先空间后时间）才能间接看到。

### 8.4 动力学模型：MaskGIT 加隐动作

Phase 8 讲过 MaskGIT：训练时随机遮住一部分 token 让模型补全；
推理时从全部遮住开始，每一轮并行预测所有位置，保留最有把握的一部分，其余的重新遮住，迭代若干轮。

Genie 把它用在"下一帧"上：

- **训练**：输入是各帧的 token（其中一部分被随机遮住）和隐动作，目标是被遮住位置的真实 token。
  遮挡比例每次随机取（论文取 0.5 到 1 之间，待核对）
- **推理**：过去帧的 token 全部可见，新的一帧全部遮住，做若干轮 MaskGIT 解码
  （论文每帧 25 轮，待核对）

和第 5 节对比一下会发现，这其实是同一个想法的离散版本：

| | Genie（离散 token） | Diffusion Forcing（连续） |
| --- | --- | --- |
| "不确定"怎么表示 | 遮住（二值） | 加噪声（连续的噪声水平） |
| 训练 | 随机遮一部分 token，预测被遮住的 | 每帧随机噪声水平，预测干净的 |
| 推理 | 新的一帧从全遮住开始，逐轮揭开 | 新的一帧从纯噪声开始，逐步去噪 |
| 每帧的网络调用次数 | MaskGIT 的轮数 | 去噪步数 |

为什么不用 Phase 8 的 IRIS 那样逐个 token 自回归？每帧几百上千个 token，逐个生成太慢。
MaskGIT 每轮并行预测所有位置，调用次数由轮数决定，和 token 数无关。

### 8.5 规模、结果与后续

以下数字出自 Genie 论文：

- 模型规模：摘要称 Genie 有 110 亿参数，可以看作一个基础世界模型
- 训练数据：二维平台游戏的网络视频，从约 20 万小时的原始视频中筛出约 3 万小时（论文数据集部分，待核对）
- 隐动作码本：8 个码
- 论文还在机器人视频（无动作标签）上训练了一个较小的模型，说明方法不限于游戏；
  并演示了用学到的隐动作，从没见过的视频里做模仿学习

Genie 生成的是低分辨率、低帧率的短片段，离"实时可玩"有距离：每一帧要做几十轮 MaskGIT 解码。

**后续。** 这个系列此后只有官方博客，没有论文，以下内容以官方说明为准（待核对）：

- **Genie 2**（2024 年 12 月）：从二维平台游戏扩展到三维环境，可以用键盘鼠标操控。
  官方说明称它是一个自回归的隐空间扩散模型，也就是从第 8.4 节的离散 MaskGIT 路线换到了第 6 节的连续隐变量加扩散的路线
- **Genie 3**（2025 年 8 月）：官方说明称可以实时交互（每秒 24 帧、720p），一致性能保持数分钟，
  并支持用文字在交互过程中改变世界里的事件

从 Genie 到 Genie 3 的走向，正好对应本章后半部分的两个主题：从离散 token 走向隐空间扩散（第 6 节），
从离线生成走向实时交互（第 9 节）。

### 8.6 算法

```
算法：训练 Genie（三步）

 1. 训练视频 tokenizer：重建视频，得到编码器、解码器、token 码本。冻结
 2. 训练隐动作模型（输入像素）：按第 7.7 节。冻结，只留编码器和小码本
 3. 训练动力学模型：
      重复：
        取视频片段 o_{1:T}
        z_{1:T} ← tokenizer 编码                              [B, T, S]，整数
        ā_{1:T−1} ← 隐动作模型打标签                           [B, T−1]，整数
        随机遮住 z 里的一部分 token
        logits ← 动力学模型(遮挡后的 z, ā)                     [B, T, S, V]
        损失 = 被遮住位置上的交叉熵
```

```
算法：与 Genie 交互

输入：一张提示图像
 1. z_1 ← tokenizer 编码提示图像；history ← [z_1]；actions ← []
 2. 循环：
        人选择一个隐动作 ā ∈ {0, …, M−1}，追加进 actions
        新的一帧 z_new ← 全部遮住
        重复 R 轮（MaskGIT）：
            logits ← 动力学模型(history + [z_new], actions)
            对被遮住的位置采样，保留置信度最高的一部分，其余重新遮住
        history 追加 z_new；解码 z_new 并显示
```

### 8.7 代码对应

对应 `08-mini-genie.py`（讲义待写）。三个部件分别复用前面的实现：
tokenizer 来自 Phase 8（或第 6 节讲义的离散版本），隐动作模型来自 `07-latent-action-models.py`，
MaskGIT 的遮挡与解码逻辑来自 Phase 8。

| 函数 / 类（规划中） | 做什么 |
| --- | --- |
| `STTransformerBlock` | 空间注意力 + 因果的时间注意力 + 前馈层 |
| `DynamicsModel` | 输入 token 和隐动作，输出每个位置的 logits |
| `dynamics_loss()` | 遮挡 + 交叉熵 |
| `maskgit_generate_frame(history_tokens, latent_actions, n_rounds)` | 生成一帧的 token |
| `play(prompt_image)` | 键盘按键映射到隐动作编号，循环生成并显示 |

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `tokens` | `[B, T, S]` | 每帧 $S$ 个 token 的编号 |
| `latent_actions` | `[B, T-1]` | 隐动作编号 |
| token 嵌入 | `[B, T, S, D]` | 隐动作嵌入 `[B, T-1, 1, D]` 广播后加到下一帧的每个位置上 |
| `logits` | `[B, T, S, V]` | $V$ 是 token 码本大小 |

> 📌 待回填：精简版 Genie 在讲义所用环境上的可控性（$\Delta$PSNR，以及隐动作与真实动作的对应表）；
> MaskGIT 轮数对画面质量和每帧耗时的影响；讲义版本与论文在规模上的差别，来自 `08-mini-genie.py`。

### 8.8 常见误区

- **误区：Genie 是一个扩散模型。**
  实际情况：2024 年论文里的 Genie 没有扩散。它用离散的视频 token，动力学模型是 MaskGIT。
  后续的 Genie 2 才换成隐空间扩散（按官方说明）。
- **误区：Genie 的三个部件是端到端一起训练的。**
  实际情况：分阶段训练。tokenizer 和隐动作模型先各自训练并冻结，再训练动力学模型。
- **误区：隐动作模型的解码器就是 Genie 生成画面用的模型。**
  实际情况：隐动作模型只负责"发明按键并给视频打标签"。它的解码器在训练结束后就丢掉了，
  生成画面的是另一个大得多的动力学模型。
- **误区：Genie 可以直接当强化学习的训练环境。**
  实际情况：它没有奖励，没有任务的定义，动作也是隐动作。要在里面训练智能体，
  得另外解决"奖励从哪来"和"隐动作怎么对应到真实动作"。这是第 10 节和 Phase 11 的话题。

---

## 9. 实时推理：少步采样与蒸馏

### 9.1 直觉

做过前端的人都知道帧预算：屏幕 60 Hz 刷新，一帧只有 16.7 毫秒，超时就掉帧。
交互式世界模型面对的是同一种约束，而且更硬：用户按下按键，模型必须在下一帧的截止时间之前把画面算出来。
算得再好，晚了就是卡顿。

离线的视频生成没有这个约束。等一分钟生成五秒视频，没人在意。
所以大部分视频生成模型在设计时根本没考虑实时，有两处尤其不适合交互：

- **步数多**：一段视频要去噪几十步
- **双向**：生成第 1 帧时要用到第 50 帧的信息。整段视频必须一起生成，没法"生成一帧、显示一帧、等用户输入"

把一个视频模型变成能实时交互的世界模型，要解决的就是这两件事：**少步**和**因果**。

### 9.2 定义：每帧延迟的账

$$
t_{\text{帧}} = N \times t_{\text{网络}} + t_{\text{解码}} + t_{\text{其他}},
\qquad \text{帧率} = \frac{1}{t_{\text{帧}}}
$$

- $N$：每帧的去噪步数（用了 classifier-free guidance 的话，网络调用次数再乘 2）
- $t_{\text{网络}}$：一次去噪网络前向计算的时间，取决于网络大小、token 数、上下文长度
- $t_{\text{解码}}$：tokenizer 解码器把隐变量变回画面的时间

**手算。** 假设（示意数字，不是实测）一次网络前向 12 毫秒，解码 6 毫秒，其他开销 2 毫秒。目标 20 帧每秒，预算 50 毫秒：

$$
N \le \frac{50 - 6 - 2}{12} = 3.5 \quad \Longrightarrow \quad N = 3
$$

如果沿用图像生成常见的 50 步：$50 \times 12 + 8 = 608$ 毫秒一帧，1.6 帧每秒。差了一个数量级还多。

公式里的每一项对应一类优化办法：

| 要降低的项 | 办法 | 本章哪里 |
| --- | --- | --- |
| $N$ | 更好的参数化和采样器 | 第 1 节（EDM）；第 3.4 节（下一帧本来就不需要很多步） |
| $N$ | 蒸馏：把多步的老师压成少步的学生 | 第 9.3 节 |
| $N$ | 直接训练一个支持大步长的模型 | 第 9.4 节（shortcut 模型） |
| $t_{\text{网络}}$ | 在隐空间里做，token 少 | 第 6 节 |
| $t_{\text{网络}}$ | 因果注意力 + KV cache，过去帧不重复计算 | 第 9.5 节；Phase 8 |
| $t_{\text{网络}}$ | 限制上下文长度；模型量化、编译 | 工程手段 |
| $t_{\text{解码}}$ | 轻量的解码器 | 第 6 节 |

### 9.3 蒸馏：三种思路

蒸馏的共同设定：有一个训练好的多步模型（老师），想得到一个只用很少步数、结果相近的模型（学生）。

| 思路 | 学生学什么 | 特点 |
| --- | --- | --- |
| 逐级减半 | 学生的 1 步，模仿老师的 2 步。学成之后学生当老师，再减半 | 简单；每减半一次都要重新训练一轮 |
| 一致性 | 同一条采样轨迹上的任意一点，都应该映射到同一个终点 | 可以一步生成；训练较难调 |
| 分布匹配 | 不要求学生的每个样本和老师逐一对上，只要求学生生成的**分布**和老师一致 | 质量高；需要额外训练一个辅助网络来估计学生的分布 |

第一种出自 Salimans & Ho 2022（就是提出 v-prediction 的那篇），第二种是 consistency models（Song et al. 2023），
第三种是 distribution matching distillation。CausVid 用的是第三种。这三篇不在本章的论文表里，这里只需要知道思路。

蒸馏的两个通病：要先有一个好老师，训练是两阶段的；学生的质量上限通常就是老师。

### 9.4 推导：shortcut 模型

Shortcut 模型（Frans et al. 2024，arXiv:2410.12557）不需要老师，一次训练就得到一个"几步都能用"的模型。
Dreamer 4 的 shortcut forcing 建立在它之上。

**问题回顾。** 第 2.4 节看到，flow matching 学到的速度 $v^\star(x_u, u)$ 是轨迹在这一点的**瞬时**方向。
轨迹是弯的，沿瞬时方向迈一大步会偏离轨迹。一步到底的结果是数据的平均。

**想法。** 让网络多知道一件事：**这一步打算迈多大。** 记步长为 $d$，网络变成 $\bar{v}_\theta(x_u, u, d)$，
它输出的不再是瞬时速度，而是"从 $u$ 走到 $u + d$ 这一段的平均速度"：

$$
x_{u+d} = x_u + d \cdot \bar{v}_\theta(x_u, u, d)
$$

类比导航：瞬时速度是"此刻车头朝哪"，shortcut 是"从这里到 $d$ 公里外的那个点，直线方向是哪"。
路是弯的，这两个方向不一样；步子越大，差得越多。

**第一步：小步长的极限。** $d \to 0$ 时，一段的平均速度就是瞬时速度：

$$
\bar{v}(x_u, u, 0) = v^\star(x_u, u)
$$

这一档可以直接用第 2.3 节的条件 flow matching 损失训练，目标是 $x - \epsilon$。

**第二步：大步长的自洽条件。** 一大步（步长 $2d$）应该等于连续两小步（各 $d$）。写出两小步：

$$
\begin{aligned}
x_{u+d} &= x_u + d \cdot \bar{v}(x_u, u, d) \\
x_{u+2d} &= x_{u+d} + d \cdot \bar{v}(x_{u+d}, u + d, d) \\
         &= x_u + d \cdot \bar{v}(x_u, u, d) + d \cdot \bar{v}(x_{u+d}, u + d, d)
\end{aligned}
$$

一大步是 $x_{u+2d} = x_u + 2d \cdot \bar{v}(x_u, u, 2d)$。两式的右边相等，消去 $x_u$，两边除以 $2d$：

$$
\bar{v}(x_u, u, 2d) = \frac{1}{2}\, \bar{v}(x_u, u, d) + \frac{1}{2}\, \bar{v}(x_{u+d}, u + d, d)
$$

**大步的平均速度，等于两段小步的平均速度的平均。**

**第三步：把自洽条件变成损失。** 右边用网络自己当前的输出来算，并且停止梯度：

$$
\mathcal{L}_{\text{shortcut}}(\theta) = \mathbb{E} \Big[
\underbrace{\big\Vert \bar{v}_\theta(x_u, u, 0) - (x - \epsilon) \big\Vert^2}_{\text{flow matching 项：最小步长，目标来自数据}}
+ \underbrace{\big\Vert \bar{v}_\theta(x_u, u, 2d) - \mathrm{sg}(\bar{v}_{\text{target}}) \big\Vert^2}_{\text{自洽项：大步长，目标来自自己}}
\Big]
$$

$$
\bar{v}_{\text{target}} = \frac{1}{2}\, \bar{v}_\theta(x_u, u, d) + \frac{1}{2}\, \bar{v}_\theta(x'_{u+d}, u + d, d),
\qquad x'_{u+d} = x_u + d \cdot \bar{v}_\theta(x_u, u, d)
$$

步长 $d$ 从一组成倍的值里取（$1/2$、$1/4$、$1/8$ 等）。每一档大步长的目标由比它小一档的输出拼出来，
最小的一档由数据直接监督。

**这是自举。** 学过 Phase 4 的读者应该觉得眼熟：

| | TD 学习（Phase 4） | shortcut 模型 |
| --- | --- | --- |
| 接地的信号 | 真实的奖励 $r_t$ | 最小步长上的 flow matching 目标 $x - \epsilon$ |
| 自举的目标 | $r_t + \gamma V(s_{t+1})$，用自己对下一状态的估计 | 两小步的平均，用自己对小步长的估计 |
| 目标要停止梯度吗 | 要 | 要 |
| 信息怎么传播 | 从回合末尾往前，一步步传 | 从小步长往大步长，一档档传 |

**手算例子。** 沿用第 2.4 节的一维高斯：数据 $\mathcal{N}(2, 1)$，精确轨迹是 $x_u = 2u + x_0 \sqrt{(1 - u)^2 + u^2}$。
从 $x_0 = 0.8$ 出发，轨迹上的三个点：

$$
x_0 = 0.8, \qquad x_{0.5} = 1 + 0.8 \times 0.7071 = 1.566, \qquad x_1 = 2.8
$$

理想的 shortcut 就是这些点之间的平均速度：

| 从哪到哪 | 步长 $d$ | 理想的 $\bar{v}$ | 一步之后 |
| --- | --- | --- | --- |
| $u = 0 \to 1$ | 1 | $(2.8 - 0.8)/1 = 2.0$ | $0.8 + 1 \times 2.0 = 2.8$ |
| $u = 0 \to 0.5$ | 0.5 | $(1.566 - 0.8)/0.5 = 1.531$ | 1.566 |
| $u = 0.5 \to 1$ | 0.5 | $(2.8 - 1.566)/0.5 = 2.469$ | 2.8 |
| $u = 0$ 处的瞬时速度 | 0 | $2 - 0.8 = 1.2$ | |

验证自洽条件：$\frac{1}{2}(1.531 + 2.469) = 2.0$，正好等于步长为 1 的 shortcut。

对比：同一个起点，瞬时速度是 1.2，步长为 1 的 shortcut 是 2.0。
用瞬时速度一步走到底得到 2.0（数据的均值，第 2.4 节），用 shortcut 一步走到底得到 2.8（精确的样本）。
**网络知道了步长，就能给出不同的答案。**

推理时步数可以临时决定：想要 4 步，就用 $d = 1/4$ 调用 4 次；想要 1 步，就用 $d = 1$ 调用 1 次。同一个模型。

### 9.5 因果化：从双向到流式

**双向模型为什么不能交互。** 视频扩散模型常用双向的时间注意力：每一帧都能看到所有其他帧。
这对质量有好处，但意味着第 1 帧的去噪结果依赖第 50 帧的带噪内容。两个后果：

- 必须等整段视频的所有去噪步都做完，第 1 帧才能显示
- 未来的帧取决于用户接下来的操作，而那些操作还没发生

**因果模型。** 把时间注意力加上因果掩码（Phase 8 的 GPT 式掩码），每一帧只看过去。
于是可以生成一帧、显示一帧。还可以用 KV cache：过去帧的键和值算一次就存起来，
生成新帧时不用重新计算（Phase 8 详细讲过这笔账）。

```
   双向（离线）                              因果 + KV cache（流式）

   帧1 ◀──▶ 帧2 ◀──▶ 帧3 ◀──▶ 帧4            帧1 ──▶ 帧2 ──▶ 帧3 ──▶ 帧4
   全部一起去噪，全部做完才能显示               │       │       │
                                           显示    显示    显示      每帧生成完立刻显示
                                           过去帧的 K、V 缓存起来，不重算
```

把因果和少步合在一起做的两个代表：

- **CausVid**（Yin et al. 2024，arXiv:2412.07772）。标题就是它做的事：从慢的双向模型到快的自回归模型。
  摘要给出的做法：用一个双向的老师去监督一个因果的学生（非对称蒸馏），
  同时用分布匹配蒸馏把 50 步压成 4 步，借助 KV cache 在单个 GPU 上达到约 9.4 帧每秒（见该论文摘要）。
  摘要还提到，这种蒸馏方式有效缓解了自回归生成的误差累积
- **Self Forcing**（Huang et al. 2025，arXiv:2506.08009）。针对的是第 5.2 节的曝光偏差。
  摘要给出的做法：训练时就带着 KV cache 做自回归展开，让每一帧以模型此前自己生成的帧为条件，
  再用一个作用在整段视频上的损失来监督。摘要称可以在单个 GPU 上做到亚秒级延迟的实时流式生成

注意这两个工作里因果化和第 5 节的联系：模型一旦变成因果自回归的，漂移问题就跟着来了。
所以"变快"和"变稳"在实时视频模型里总是一起解决的。

GameNGen 是另一个参照点：它的摘要报告，在单个 TPU 上以每秒 20 帧以上实时模拟 DOOM。
它的做法是把一个预训练的图像隐空间扩散模型改造成以过去帧和动作为条件的下一帧模型，用很少的采样步数。

### 9.6 算法

```
算法：shortcut 模型训练

重复：
 1. 取数据 x，噪声 ε，路径时间 u                                [B, …]，[B]
 2. x_u = (1 − u) · ε + u · x
 3. 批次的一部分：flow matching 项
        损失_FM = mean((v̄_θ(x_u, u, d=0) − (x − ε))²)
 4. 批次的另一部分：自洽项
        随机取步长 d（成倍的一组值之一，且 u + 2d ≤ 1）
        不回传梯度地计算：
            v1 ← v̄_θ(x_u, u, d)
            x_mid ← x_u + d · v1
            v2 ← v̄_θ(x_mid, u + d, d)
            v̄_target ← (v1 + v2) / 2
        损失_SC = mean((v̄_θ(x_u, u, 2d) − v̄_target)²)
 5. 更新 θ：损失 = 损失_FM + 损失_SC
```

```
算法：流式交互循环（因果模型 + KV cache + 少步）

 1. 编码起始帧，写入 KV cache
 2. 循环（每一帧）：
        读取用户的动作 a
        x ← 噪声
        对 i = 0 … N−1（N 取 1 到 4）：
            x ← 去噪一步（只对新的一帧做计算；过去帧的 K、V 从 cache 读）
        把新帧写入 KV cache；cache 超过上下文长度时丢掉最旧的帧
        解码并显示
        记录这一帧的耗时
```

### 9.7 代码对应

对应 `09-realtime-few-step-inference.py`（讲义待写）。

| 函数 / 类（规划中） | 做什么 |
| --- | --- |
| `profile_frame_latency(model, n_steps)` | 把一帧的耗时拆成网络、解码、其他三部分 |
| `ShortcutNet.forward(x_u, u, d)` | 多一个步长输入的速度网络 |
| `shortcut_loss()` | 第 9.6 节的两项损失 |
| `sample_shortcut(model, shape, n_steps)` | 用 $d = 1/N$ 采样 |
| `CausalFrameCache` | 按帧管理 KV cache，支持滑动窗口 |
| `stream_loop()` | 流式交互循环 |

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `u`，`d` | `[B]` | 路径时间与步长，各自过嵌入后相加，得到 `[B, D]` |
| `shortcut_target` | 与 `x_u` 相同 | 自洽目标，不带梯度 |
| KV cache | 每层 `[B, T_ctx, S, D]` | $T_{\text{ctx}}$ 是保留的上下文帧数 |

> 📌 待回填：本机（MPS）和 4090 上每帧耗时的构成；flow matching 模型与 shortcut 模型在 1、2、4 步下的样本质量；
> 开启 KV cache 前后的每帧耗时，来自 `09-realtime-few-step-inference.py`。
> 定性预期：1 步时 shortcut 模型明显好于普通的 flow matching 模型。

### 9.8 常见误区

- **误区：实时就是把模型做小。**
  实际情况：模型大小只影响 $t_{\text{网络}}$。步数 $N$ 是乘在外面的，从 50 步降到 4 步的收益比把模型砍一半大得多。
- **误区：少步模型一定比多步模型差。**
  实际情况：直接把多步模型的步数砍掉会差；专门为少步训练的模型（蒸馏、shortcut）可以在几步内接近多步的质量。
  世界模型里下一帧本来就不太不确定，少步的损失更小。
- **误区：shortcut 模型需要先训练一个老师。**
  实际情况：不需要。它是单阶段训练，大步长的目标由自己的小步长输出自举得到。
- **误区：把注意力掩码改成因果的，双向模型就能流式生成了。**
  实际情况：模型是在双向的条件下训练的，直接换掩码等于换了输入分布，质量会崩。
  需要重新训练或蒸馏，并且要同时处理因果化带来的漂移。

---

## 10. Dreamer 4 与视频基础模型

### 10.1 直觉

到这里，本章的零件都摆在桌上了。Dreamer 4（Hafner et al. 2025，arXiv:2509.24527）
是把它们装成一台机器的一次尝试，而且装回了 Phase 6 的框架里：**在世界模型里用强化学习训练智能体。**

| 零件 | 本章哪里 | 在 Dreamer 4 里的角色 |
| --- | --- | --- |
| 因果的视频 tokenizer | 第 6 节 | 把画面压成连续隐变量 |
| flow matching | 第 2 节 | 动力学模型的训练目标的基础 |
| 每帧独立的噪声水平 | 第 5 节 | 让动力学模型可以稳定地自回归展开 |
| shortcut 模型 | 第 9 节 | 让每帧只需要很少的采样步数 |
| 从无动作标签的视频里学 | 第 7、8 节的主题 | 大部分知识来自无标注视频，只需少量带动作的数据 |
| 在想象中训练策略 | 第 4 节；Phase 6 | 智能体完全在世界模型里学 |

第 5 节和第 9 节的两个零件合起来，就是论文所说的 **shortcut forcing**。

### 10.2 Dreamer 4 做了什么

以下按论文摘要整理：

- **目标**：一个可扩展的智能体，在一个又快又准的世界模型里用强化学习学会解决控制任务
- **环境**：Minecraft。摘要称世界模型能准确预测物体交互和游戏机制，大幅超过此前的世界模型
- **实时**：通过 shortcut forcing 目标和一个高效的 Transformer 架构，在单个 GPU 上实现实时的交互式推理
- **数据**：世界模型只需要少量带动作的数据就能学会一般的动作条件，因此可以从大量无标注视频里获得大部分知识
- **任务**：只用离线数据，在 Minecraft 里获得钻石。这个任务需要从原始像素出发，
  选出超过 2 万个鼠标键盘动作组成的序列
- **结果**：通过在想象中学习行为，它是第一个完全不与环境交互、仅凭离线数据就获得钻石的智能体

"只用离线数据"是整件事的重点。Phase 6 的 Dreamer V3 也挖到过钻石，但那是边玩边学。
机器人上边玩边学又慢又危险，摘要明确把这个设定和机器人应用联系起来：
先从已有的数据里学出世界模型，再在世界模型里练，全程不碰真实环境。

### 10.3 shortcut forcing

把第 5.4 节和第 9.4 节的记号合起来。序列里第 $t$ 帧的隐变量是 $z_t$，
它有自己的路径时间 $u_t$（相当于自己的噪声水平），带噪版本是

$$
\tilde{z}_t = (1 - u_t)\, \epsilon_t + u_t\, z_t
$$

动力学模型是一个因果的序列模型。它接收所有帧的带噪隐变量、各自的路径时间、步长 $d$ 和动作，
对每一帧输出预测。训练目标是把 shortcut 模型的两项损失（flow matching 项和自洽项）用在每一帧上，
各帧的 $u_t$ 独立采样：

$$
\mathcal{L}_{\text{SF}} \;\approx\; \mathbb{E} \sum_{t=1}^{T} \Big[ \text{第 } t \text{ 帧的 shortcut 损失}\big( \tilde{z}_{1:t},\ u_{1:t},\ d,\ a_{1:t} \big) \Big]
$$

这是示意性的写法：它只表达"每帧独立噪声水平 + 步长条件 + 自洽目标"这个组合。
Dreamer 4 的精确形式，包括网络预测的是速度还是干净隐变量、损失如何随噪声水平加权、
步长如何采样，这里不展开（待核对，写讲义时对照论文的方法部分）。

有一点值得提前知道。论文让网络预测干净的隐变量而不是速度（待核对）。用第 2.5 节的换算
$z = \tilde{z} + (1 - u)\, v$ 可以互相转换，所以这不改变模型的表达能力，改变的是误差落在哪里。
回忆第 4.3 节的教训：**长序列自回归生成时，网络误差经过参数化换算之后被放大多少，决定了稳不稳。**
这和 DIAMOND 选 EDM 是同一类考虑。

推理时，生成每一帧只需要几次网络调用（步长取大），已经生成的帧按第 5.4 节的做法带一点噪声水平喂回去。

### 10.4 三个阶段

```
   阶段一：学世界模型                    阶段二：接上智能体                 阶段三：在想象中训练
   ─────────────────                   ─────────────────                 ─────────────────
   数据：大量视频                        数据：带动作、带任务的离线数据        数据：不需要新数据
        （少量带动作）                                                    
                                                                        从数据集里的真实片段出发，
   训练：tokenizer                       训练：策略头（行为克隆）             在世界模型里展开，
        动力学模型                             奖励头                      用强化学习更新策略头和价值头
        （shortcut forcing）                                              
                                                                        世界模型不动
```

上图按论文的总体描述整理，各阶段具体训练哪些参数、用什么强化学习算法（待核对）。

第三阶段和 Phase 6、本章第 4 节是同一个算法骨架：想象轨迹、算 λ-return 一类的价值目标、更新 actor 和 critic。
不同之处是整件事发生在离线数据上，智能体从头到尾没有在真实的 Minecraft 里走过一步。

### 10.5 几代"在想象中训练"的世界模型

| | Dreamer V3（Phase 6） | IRIS（Phase 8） | DIAMOND（第 4 节） | Dreamer 4 |
| --- | --- | --- | --- | --- |
| 表示 | RSSM 的类别隐变量 | 离散 token | 像素 | tokenizer 的连续隐变量 |
| 动力学骨干 | GRU | 自回归 Transformer | U-Net | 因果 Transformer |
| 生成式目标 | 重建 + KL | token 的交叉熵 | 去噪（EDM） | shortcut forcing |
| 每步想象的代价 | 很低 | 高（逐 token） | 中（几步去噪） | 中（几步采样） |
| 数据来源 | 在线交互 | 在线交互 | 在线交互 | 离线数据 + 无标注视频 |
| 能用无动作标签的视频吗 | 不能 | 不能 | 不能 | 能 |

和 Genie 对比：两者都想利用无标注视频。Genie 的办法是发明隐动作；
Dreamer 4 的办法是保留真实的动作空间，靠少量带动作的数据把动作条件学出来，
其余的视频只用来学"世界怎么变化"。

### 10.6 视频基础模型作为世界模型

另一条线不从强化学习出发，而是从视频生成出发：先在海量视频上训练一个大的视频模型，再想办法让它变成世界模型。

| 工作 | 是什么 | 动作条件 | 论文怎么定位它 |
| --- | --- | --- | --- |
| UniSim（Yang et al. 2023，arXiv:2310.06114） | 通用的真实世界交互仿真器 | 文字指令、机器人控制、相机运动等多种"动作" | 把多种来源的数据合在一起，学一个仿真器；在里面训练的策略可以迁移到真实世界 |
| Cosmos（NVIDIA 2025，arXiv:2501.03575） | 面向物理 AI 的世界基础模型平台 | 通过后训练加入（机器人动作、驾驶轨迹等） | 一套平台：数据处理流程、tokenizer、预训练的世界基础模型、后训练的样例 |
| GameNGen（Valevski et al. 2024，arXiv:2408.14837） | 用扩散模型当实时游戏引擎 | 游戏按键 | 证明神经网络可以实时模拟一个复杂游戏 |
| WHAM（Kanervisto et al. 2025，Nature 638） | 游戏画面与手柄动作的联合序列模型 | 手柄动作（和画面 token 排在同一个序列里） | 面向游戏创意设计；强调一致性、多样性、对用户修改的保持 |
| Genie 2 / Genie 3 | 可交互的生成式世界 | 键鼠操作；文字事件 | 只有官方博客（见第 8.5 节） |

这些工作的共同点是规模大、数据杂、面向"通用"。它们和本章前半部分的关系是：技术零件相同
（隐空间扩散或 token 自回归、动作条件、因果化、少步），区别在数据规模和工程量。

对机器人方向，这些模型的用处主要是 Phase 11 的三种：当仿真器来评估策略，
当数据引擎来生成训练数据，以及作为策略的预训练骨干。

### 10.7 现状与争议

"视频生成模型是不是世界模型"目前没有共识。几个主要的分歧点：

**一、能不能以动作为条件。** 这是 Phase 1 第 1.2 节定的线。只会按文字或首帧续写的视频模型，
回答不了"如果我这样做会怎样"。上一小节的工作都在补这一点，但补的程度不同：
"相机往左转"这种粗粒度的控制，和"机械臂末端向下 2 厘米并闭合夹爪"这种精细控制，难度差得很远。

**二、看起来对，和真的对。** 视频模型的训练目标是让生成的视频和真实视频在分布上相似。
"相似"由画面的主要成分决定，物理上是否正确并没有被单独要求。
已经有专门检验这一点的研究，结论大体是：模型在训练分布内表现不错，
换到分布之外的组合时，更像是在套用见过的相似案例，而不是在用学到的物理规律
（例如 Kang et al. 2024 的 *How Far is Video Generation from World Model: A Physical Law Perspective*）。
对决策来说，"看起来合理但错了"比"明显错了"更危险，因为不容易被发现。

**三、长时间的一致性和记忆。** 转身再回来，房间还是原来的房间吗？上下文窗口有限的模型做不到。
各家都在延长一致的时长（Genie 3 的官方说明是数分钟），但离"持久的世界"还有距离。

**四、怎么评估。** FVD 这类指标衡量的是"像不像真视频"，和"对决策有没有用"是两回事。
Phase 1 第 1.6 节讲的目标不匹配在这里被放大了：模型把算力花在草叶怎么晃上，
而任务只关心杯子有没有被拿起来。更对口的评估是 Phase 11 要讲的：
在世界模型里评估一组策略，看得到的排序和真实环境里的排序是否一致。

**五、非要生成像素吗。** 生成每一个像素很贵，而画面里大部分细节既不可预测，也和决策无关。
这是 LeCun 一派的核心批评，也是下一章 JEPA 的出发点：只在表征空间里预测，不解码回像素。
反方的理由是：像素是唯一不需要人为定义的通用接口，能直接给人看，
能直接接上任何以图像为输入的策略，还能利用互联网上所有的视频。

这五点里，前四点是程度问题，会随着模型进步逐渐改善；第五点是路线之争，目前两边都有有力的结果。
关于"世界模型"这个概念本身的讨论，见 Phase 1 附录 B 列出的两篇综述性文章
（Ding et al. 2024，arXiv:2411.14499；Xing et al. 2025，arXiv:2507.05169）。

### 10.8 算法

```
算法：Dreamer 4 的训练流程（按论文的总体描述；细节待核对）

阶段一：世界模型预训练
   在视频上训练因果 tokenizer
   在 (隐变量, 动作) 序列上用 shortcut forcing 训练动力学模型
       —— 没有动作标签的片段，动作条件留空

阶段二：智能体微调
   在带动作和任务信息的离线数据上：
       训练策略头模仿数据里的动作（行为克隆）
       训练奖励头预测任务奖励

阶段三：想象训练
   重复：
       从离线数据取真实片段作为起点
       用策略头选动作，用动力学模型少步采样生成下一帧的隐变量，用奖励头给奖励
       展开 H 步，计算价值目标
       只更新策略头和价值头
```

### 10.9 代码对应

对应 `10-dreamer4-and-video-foundation-models.py`（讲义待写）。
这一节不复现 Dreamer 4（规模差太远），做两件小事：

| 函数（规划中） | 做什么 |
| --- | --- |
| `shortcut_forcing_loss()` | 在玩具序列数据上，把第 5 节的每帧独立噪声水平和第 9 节的 shortcut 损失合在一起 |
| `few_step_rollout()` | 用 1、2、4 步采样做自回归展开，对比稳定性 |
| `compare_world_models()` | 把本章几类模型（像素扩散、隐空间扩散、Diffusion Forcing、精简版 Genie）在同一份数据上的开环误差、每帧耗时、可控性汇成一张表 |
| `video_model_checklist()` | 第 10.7 节五个分歧点的检查表，用来读新论文 |

| 张量 | 形状 | 说明 |
| --- | --- | --- |
| `latents` | `[B, T, S, D]` | 每帧 $S$ 个隐变量 token |
| `u` | `[B, T]` | 每帧独立的路径时间 |
| `d` | `[B]` 或 `[B, T]` | 步长 |
| 输出 | `[B, T, S, D]` | 每帧的预测 |

> 📌 待回填：玩具序列上 shortcut forcing 与普通 Diffusion Forcing 在 1、2、4 步下的展开误差；
> 本章几类世界模型的汇总对比表，来自 `10-dreamer4-and-video-foundation-models.py`。

### 10.10 常见误区

- **误区：Dreamer 4 是 Dreamer V3 的小改款。**
  实际情况：名字是延续的，"在想象中训练"的框架也是延续的，但世界模型从 RSSM 整个换成了
  tokenizer 加因果 Transformer 加 shortcut forcing，设定也从在线交互换成了离线。
- **误区：视频基础模型的画面这么逼真，一定学到了物理。**
  实际情况：逼真说明它学到了视频的分布。分布之内的物理现象它能复现，
  分布之外会不会按物理规律外推，是另一个问题，现有的证据并不乐观（第 10.7 节）。
- **误区：世界模型越大越通用，对机器人就越有用。**
  实际情况：对机器人有用，要看它能不能以机器人的真实动作为条件、在任务相关的细节上够不够准、
  推理够不够快。通用的大模型往往要经过针对具体机器人的后训练才能用。
- **误区：这些模型没有论文、只有演示的那部分，可以按演示的效果来理解。**
  实际情况：演示是挑过的。没有论文就没有消融、失败案例和可复现的数字。
  本章凡是只有官方博客的内容都标了待核对。

---

## 11. 方法对比与选型

### 11.1 网络该预测什么

| 预测目标 | 单个样本的回归目标 | 低噪声下 | 高噪声下 | 谁在用 |
| --- | --- | --- | --- | --- |
| 噪声 $\epsilon_\theta$ | $\epsilon$ | 好 | 在抄输入；换算成画面时误差放大 $\sigma$ 倍 | DDPM 及其直接后继 |
| 干净样本 $D_\theta$ | $x$ | 在抄输入 | 好 | 部分少步 / 长序列模型 |
| 速度（v-prediction / flow matching） | 数据与噪声的线性组合 | 好 | 好 | flow matching 系的模型 |
| EDM 的 $F_\theta$ | $(x - c_{\text{skip}} x_\sigma)/c_{\text{out}}$ | 好（接近预测噪声） | 好（接近预测干净样本） | EDM、DIAMOND |

四种写法的最优解可以互相换算（第 1.5、2.5 节）。选型看两点：
步数越少、自回归链条越长，越要避开第一行；后两行在所有噪声水平上都稳，是默认选择。

### 11.2 三套扩散框架

| | DDPM | EDM | Flow matching |
| --- | --- | --- | --- |
| 噪声怎么描述 | 离散步 $k$，日程 $\bar\alpha_k$ | 连续的噪声标准差 $\sigma$ | 路径时间 $u \in [0, 1]$ |
| 带噪样本 | $\sqrt{\bar\alpha_k}\, x + \sqrt{1 - \bar\alpha_k}\, \epsilon$ | $x + \sigma\epsilon$ | $(1 - u)\,\epsilon + u\,x$ |
| 互相换算 | $\sigma = \sqrt{(1 - \bar\alpha_k)/\bar\alpha_k}$ | 基准 | $\sigma = (1 - u)/u$ |
| 网络预测 | 噪声 | 预条件后的混合目标 | 速度 $x - \epsilon$ |
| 训练与采样 | 绑定：按训练的步往回走 | 解耦：任选日程和求解器 | 解耦：任选步数 |
| 实现复杂度 | 低 | 中（四个系数、噪声分布、日程） | 低 |
| 少步表现 | 差 | 好 | 中；配合 reflow 或 shortcut 后好 |
| 本章的例子 | 回顾用 | DIAMOND | Dreamer 4 的基础；Phase 11 的 π0 |

### 11.3 几类世界模型

| | RSSM（Phase 3、6） | Token 自回归（Phase 8） | 像素扩散 | 隐空间扩散 | Token + MaskGIT | shortcut forcing |
| --- | --- | --- | --- | --- | --- | --- |
| 代表 | Dreamer V3 | IRIS | DIAMOND | GameNGen、Cosmos | Genie | Dreamer 4 |
| 表示 | 小的隐状态 | 离散 token | 像素 | 连续隐变量 | 离散 token | 连续隐变量 |
| 细节保真 | 低 | 受码本限制 | 高 | 受 tokenizer 限制，通常较高 | 受码本限制 | 受 tokenizer 限制 |
| 每帧的网络调用 | 1 | 每帧的 token 数 | 几步 | 几步到几十步 | 几十轮 | 几步 |
| 适用的分辨率 | 低 | 低到中 | 低 | 中到高 | 低到中 | 中 |
| 画面能直接给人看吗 | 要解码，较糊 | 能 | 能 | 能 | 能 | 能 |
| 主要用途 | 训练智能体 | 训练智能体 | 训练智能体；游戏引擎 | 游戏引擎；仿真器；数据引擎 | 可交互环境 | 训练智能体 |

### 11.4 治漂移：先用哪个

| 情况 | 建议 |
| --- | --- |
| 只需要十几步的想象，用来训练策略 | 短视野 + 从真实片段出发就够了（DIAMOND、Dreamer 的做法） |
| 要长时间交互，模型是"过去帧 → 下一帧"的结构 | 加噪声增强。改动最小 |
| 从头设计一个序列扩散模型 | 直接用每帧独立噪声水平训练（Diffusion Forcing），噪声增强自动包含在内 |
| 已经做了上面这些，长展开仍然慢慢变坏 | 训练时也做自回归展开（Self Forcing 一类） |
| 问题是"转身回来东西变了" | 这是记忆问题。加长上下文或加外部记忆，上面的办法不管用 |

### 11.5 动作从哪来

| | 真实动作标签 | 隐动作模型 | 逆动力学模型打伪标签 |
| --- | --- | --- | --- |
| 需要的数据 | 全部带动作 | 只要视频 | 少量带动作 + 大量视频 |
| 动作的含义 | 明确 | 要事后摸索或对齐 | 明确 |
| 能直接驱动机器人吗 | 能 | 不能，要再学一个映射 | 能 |
| 数据规模的上限 | 低 | 高 | 高 |
| 代表 | DIAMOND、GameNGen | Genie、LAPA | VPT |
| 折中的做法 | Dreamer 4：大部分视频不带动作，少量带真实动作 | | |

### 11.6 同一个任务：PushT 上的 Token 路线与扩散路线

Phase 8 的项目在 PushT 上训练了 tokenizer 加 Transformer 的世界模型。
本章的项目在同一份演示数据上训练以动作为条件的扩散世界模型。两者放在一起比四件事：

| 对比项 | 怎么量 | 定性预期 |
| --- | --- | --- |
| 细节 | 生成画面里 T 形块的位置和朝向与真实值的误差；推杆（小圆点）是否还在 | 扩散路线在块的边缘和朝向上更准；Token 路线受 tokenizer 重建质量限制 |
| 开环误差随步数的变化 | 同一批起点和动作序列，展开 1、5、10、20、50 步的误差 | 两条曲线都上升；扩散路线不加噪声增强时，后段可能上升更快 |
| 每帧耗时 | 同一台机器上生成一帧的时间 | 取决于 token 数与去噪步数，没有先验的赢家 |
| 用于规划 | 接上 Phase 1 的 CEM，看任务成功率和每步决策耗时 | 两者都偏慢，为 Phase 10 的特征空间规划留下对照 |

PushT 适合做这个对比的原因：任务成败取决于 T 形块的精确位姿，而推杆只占很少的像素。
"视觉细节重要"在这里不是一句口号，是可以量出来的。

> 📌 待回填：上表四项在 PushT 上的实测结果（Token 路线的数字来自 Phase 8 的项目，
> 扩散路线的数字来自本章项目 `projects/phase-09-playable-neural-simulator/`）。
> Phase 10 的 JEPA 路线完成后，在这张表里加第三列。

### 11.7 选型建议

- **要在世界模型里训练智能体，画面不大**：像素空间的扩散世界模型（DIAMOND 的路线），EDM 预条件，3 步左右去噪。
- **画面大，或者要长时间交互**：先训练 tokenizer，在隐空间里做；每帧独立噪声水平；从一开始就按因果结构设计。
- **只有视频，没有动作**：隐动作模型。码本先取个位数，用可控性指标验证之后再考虑放大。
- **有少量带动作的数据和大量视频**：优先考虑保留真实动作空间（逆动力学打标签，或 Dreamer 4 式的混合训练），
  省掉隐动作到真实动作的映射。
- **要实时**：先算第 9.2 节的账，确认瓶颈是步数、网络还是解码器，再决定用蒸馏、shortcut 还是换小模型。
- **只是要做规划，不需要给人看画面**：先别急着生成像素，看完 Phase 10 再决定。

---

## 12. 论文精读卡片

### 12.1 DIAMOND

**Alonso, Jelley, Micheli, Kanervisto, Storkey, Pearce, Fleuret. Diffusion for World Modeling: Visual Details Matter in Atari. NeurIPS 2024. arXiv:2405.12399**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 当时的世界模型多在离散隐变量序列上建模，压缩成离散表示会丢掉对强化学习重要的视觉细节 |
| 核心方法 | 用扩散模型在像素空间生成下一帧（EDM 预条件，少步采样，以过去帧和动作为条件）；单独的奖励 / 终止模型；在想象中用 actor-critic 训练策略 |
| 关键分析 | 让扩散适用于世界模型的几个设计选择：为什么用 EDM 而不是 DDPM；去噪步数怎么选；与离散 token 世界模型的画面对比 |
| 主要结论 | 摘要给出：Atari 100k 上平均人类归一化分数 1.46，是完全在世界模型里训练的智能体中的最好成绩；扩散世界模型还可以单独当作交互式的游戏引擎（CS:GO） |
| 局限 | 像素空间的代价随分辨率上升；上下文只有几帧，没有长期记忆；长展开会漂移 |
| 读的时候注意 | 第 4.3 节对应的那部分分析（EDM 与 DDPM 在不同步数下的展开对比）；1 步与多步去噪的对比图；附录的超参数表 |
| 与本课实现的差异 | 📌 待回填：写完 `04-diamond.py` 后列出（网络规模、环境、训练预算等与论文的差别） |

### 12.2 Genie

**Bruce, Dennis, Edwards, Parker-Holder, Shi, Hughes, Lai, Mavalankar, Steigerwald, Apps 等. Genie: Generative Interactive Environments. ICML 2024. arXiv:2402.15391**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 可交互的环境要么靠人写，要么需要带动作标签的数据。能不能只从无标注的视频里学出可以操控的环境 |
| 核心方法 | 三个部件：时空的视频 tokenizer、隐动作模型（很小的 VQ 码本）、MaskGIT 式的自回归动力学模型。骨干都是时空 Transformer |
| 关键实验 | 用各种提示图像（照片、草图、生成的图）启动并操控；隐动作含义的一致性；可控性指标 $\Delta$PSNR；消融（隐动作模型的输入用像素还是 token 等）；在机器人视频上训练；用隐动作做模仿学习 |
| 主要结论 | 摘要给出：110 亿参数，可视为基础世界模型；无需任何动作标签即可逐帧操控；学到的隐动作空间可以帮助智能体从没见过的视频里模仿行为 |
| 局限 | 低分辨率、低帧率、短片段；每帧要做多轮 MaskGIT 解码，不实时；隐动作要靠人摸索含义 |
| 读的时候注意 | 三个部件各自的输入输出和训练顺序；隐动作为什么用加性嵌入；$\Delta$PSNR 的定义 |
| 与本课实现的差异 | 📌 待回填：写完 `08-mini-genie.py` 后列出 |

### 12.3 EDM

**Karras, Aittala, Aila, Laine. Elucidating the Design Space of Diffusion-Based Generative Models. NeurIPS 2022. arXiv:2206.00364**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 扩散模型的理论和实践被各种记号和约定绑在一起，看不清哪些设计是必需的，哪些可以独立改进 |
| 核心方法 | 把设计空间拆成独立的几块：噪声水平的参数化、网络的预条件、训练时的噪声分布与损失权重、采样的日程与求解器；对每一块给出改进 |
| 主要结论 | 这些改进合起来，在当时的图像生成基准上用少得多的网络调用次数取得了更好的结果；改进的采样器可以直接用在已有的预训练模型上 |
| 读的时候注意 | Table 1（把几种已有方法写进同一套记号）；预条件的推导在附录；确定性采样器（Heun）一节。随机采样器一节可以略读 |
| 与本章的关系 | 第 1.7–1.9 节的全部内容。DIAMOND 直接采用了它的预条件 |

### 12.4 Diffusion Forcing

**Chen, Martí Monsó, Du, Simchowitz, Tedrake, Sitzmann. Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion. NeurIPS 2024. arXiv:2407.01392**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 下一 token 预测能生成变长序列，但连续数据上长展开不稳定，也难以引导；全序列扩散能引导，但长度固定 |
| 核心方法 | 训练时每个 token 有独立的噪声水平；因果的序列模型；采样时用二维的噪声日程表 |
| 主要结论 | 摘要给出：可以稳定地把连续 token 的序列（如视频）展开到超过训练长度；新的采样和引导方式在决策与规划任务上带来明显提升；该目标被证明是在优化所有子序列似然的一个变分下界 |
| 局限 | 原论文的模型规模较小；每帧独立噪声使训练任务更难 |
| 读的时候注意 | 噪声当作掩码的观点；稳定化的做法；金字塔式的采样日程 |
| 后续 | History-Guided Video Diffusion（Song et al. 2025，arXiv:2502.06764）换成 Transformer 骨干，并提出用历史做引导 |

### 12.5 Dreamer 4

**Hafner, Yan, Lillicrap. Training Agents Inside of Scalable World Models. 2025. arXiv:2509.24527**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 此前在想象中训练的智能体，世界模型要么不够准，要么不够快，撑不起复杂环境；并且依赖在线交互 |
| 核心方法 | shortcut forcing 目标 + 高效的 Transformer 架构，得到能在单个 GPU 上实时推理的世界模型；动作条件只需少量带动作的数据；在想象中用强化学习训练策略 |
| 主要结论 | 摘要给出：在 Minecraft 上准确预测物体交互和游戏机制；首次仅凭离线数据、不与环境交互就获得钻石（需要超过 2 万个鼠标键盘动作的序列） |
| 局限 | 结果集中在一个环境上；长期记忆有限（待核对） |
| 读的时候注意 | shortcut forcing 的精确损失；网络预测的量和损失权重；三个训练阶段各自更新哪些参数；带动作数据的比例对结果的影响 |
| 本章中待核对的细节 | 第 10.3、10.4、10.8 节凡是摘要之外的内容 |

### 12.6 Flow Matching 与 Rectified Flow

**Lipman, Chen, Ben-Hamu, Nickel, Le. Flow Matching for Generative Modeling. ICLR 2023. arXiv:2210.02747**
**Liu, Gong, Liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. ICLR 2023. arXiv:2209.03003**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 连续归一化流的训练要反复解常微分方程，太贵；扩散模型的路径是被前向过程定死的 |
| 核心方法 | 直接回归一个速度场。条件 flow matching：对每一对（噪声，数据）定义简单的条件路径，回归条件速度，梯度与边际目标相同。Rectified Flow：线性插值路径，并用 reflow 把轨迹拉直 |
| 主要结论 | 不需要模拟就能训练；线性路径下训练和采样都更简单；轨迹越直，需要的步数越少 |
| 读的时候注意 | Lipman et al. 的定理 1、2（边际速度场为什么生成正确的分布；两个损失的梯度为什么相同）；Liu et al. 的 reflow 一节 |
| 与本章的关系 | 第 2 节。读原文时注意时间方向和记号与本章的对应（第 0.5 节） |

### 12.7 Shortcut Models

**Frans, Hafner, Levine, Abbeel. One Step Diffusion via Shortcut Models. 2024. arXiv:2410.12557**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 少步生成通常要两阶段的蒸馏，流程复杂 |
| 核心方法 | 网络除了噪声水平还以步长为条件；小步长用 flow matching 目标，大步长用"两小步等于一大步"的自洽目标；单个网络、单阶段训练 |
| 主要结论 | 同一个模型可以用 1 步到多步采样，少步质量明显好于直接砍步数的扩散 / flow matching 模型 |
| 读的时候注意 | 自洽目标的构造；步长怎么离散化、批次里两种目标各占多少 |
| 与本章的关系 | 第 9.4 节；Dreamer 4 的 shortcut forcing 的两个来源之一 |

### 12.8 GameNGen

**Valevski, Leviathan, Arar, Fruchter. Diffusion Models Are Real-Time Game Engines. 2024. arXiv:2408.14837**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 神经网络能不能完全取代传统游戏引擎，实时地模拟一个复杂的游戏 |
| 核心方法 | 两阶段：先训练强化学习智能体玩游戏并录下数据；再把一个预训练的图像扩散模型改造成以过去帧和动作为条件的下一帧模型。用条件增强（给上下文帧加噪）来稳定长时间的自回归生成 |
| 主要结论 | 摘要给出：单个 TPU 上每秒 20 帧以上模拟 DOOM；下一帧预测 PSNR 29.4，与有损 JPEG 相当；人类评估者区分真实片段与模拟片段的准确率只略高于随机 |
| 局限 | 只有几秒的上下文记忆；只在一个游戏上验证 |
| 读的时候注意 | 噪声增强的消融；数据为什么要用智能体采而不是随机策略；解码器微调 |

### 12.9 CausVid 与 Self Forcing

**Yin, Zhang, Zhang, Freeman, Durand, Shechtman, Huang. From Slow Bidirectional to Fast Autoregressive Video Diffusion Models. CVPR 2025. arXiv:2412.07772**
**Huang, Li, He, Zhou, Shechtman. Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion. 2025. arXiv:2506.08009**

| 项 | 内容 |
| --- | --- |
| 要解决的问题 | 高质量的视频扩散模型是双向、多步的，不能流式生成；直接训练的自回归模型有曝光偏差 |
| CausVid 的方法 | 把预训练的双向模型改造成因果的自回归模型；用分布匹配蒸馏减少步数；用双向老师监督因果学生 |
| Self Forcing 的方法 | 训练时带 KV cache 做自回归展开，每帧以自己此前生成的帧为条件；用视频级的损失监督整段生成结果 |
| 主要结论 | 两者都在单个 GPU 上实现了流式的视频生成，质量接近或超过更慢的双向模型（具体数字见各自摘要，待核对） |
| 读的时候注意 | 这两篇做的是文本到视频，不带动作条件。读的是"怎样把视频模型变成因果、少步、稳定"，这些做法可以原样搬到带动作的世界模型上 |

### 12.10 UniSim 与 Cosmos

**Yang, Du, Ghasemipour, Tompson, Kaelbling, Schuurmans, Abbeel. Learning Interactive Real-World Simulators. ICLR 2024. arXiv:2310.06114**
**NVIDIA. Cosmos World Foundation Model Platform for Physical AI. 2025. arXiv:2501.03575**

| 项 | 内容 |
| --- | --- |
| UniSim 的问题与方法 | 想要一个通用的真实世界仿真器。把多种数据（各有侧重：有的物体丰富，有的动作密集，有的运动多样）统一到"给定画面和动作，生成后续视频"的接口上训练 |
| UniSim 的结论 | 既能模拟高层指令，也能模拟低层控制；在仿真器里训练的高层规划器和低层策略可以零样本迁移到真实世界 |
| Cosmos 的定位 | 一个平台而不是单个模型：视频数据处理流程、预训练的世界基础模型、后训练的样例、视频 tokenizer；以开放的方式发布 |
| 读的时候注意 | UniSim：不同来源的数据怎么统一动作接口。Cosmos：tokenizer 的设计（连续与离散、时空压缩率）；后训练成动作条件模型的例子 |
| 与本课的关系 | Phase 11 的"世界模型当仿真器"和"当数据引擎"以这两篇为背景 |

### 12.11 VPT 与 LAPA

**Baker, Akkaya, Zhokhov, Huizinga, Tang, Ecoffet, Houghton, Sampedro, Clune. Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos. NeurIPS 2022. arXiv:2206.11795**
**Ye, Jang, Jeon, Joo, Yang, Peng, Mandlekar, Tan, Chao, Lin 等. Latent Action Pretraining from Videos. ICLR 2025. arXiv:2410.11758**

| 项 | 内容 |
| --- | --- |
| 共同的问题 | 无标注的视频很多，带动作的数据很少。怎么用前者训练会行动的模型 |
| VPT 的方法 | 用少量带标签的数据训练逆动力学模型，给大量网络视频打动作伪标签，再做行为克隆 |
| LAPA 的方法 | 用 VQ 瓶颈从相邻帧里学离散隐动作；先让视觉语言模型学着预测隐动作；再用少量机器人数据微调到真实动作 |
| 读的时候注意 | 两篇的对照正好是第 7.5 节那张表。逆动力学模型为什么比行为克隆需要的标签少（它能看到未来帧，任务更简单） |
| 与本课的关系 | Phase 11 的逆动力学与机器人预训练 |

### 12.12 前沿刷新记录

课程规范要求写本章前刷新 2025–2026 年的论文清单。**撰写本文档时联网检索工具不可用，这一步没有完成。**
本章因此只引用了各章 README 论文表里已核对编号的论文；新提到的工作一律不写 arXiv 编号。
下面是写讲义前需要补做的核对：

| 待核对的事项 | 为什么重要 |
| --- | --- |
| Genie 2、Genie 3 是否已有论文或技术报告 | 第 8.5 节目前只能引用官方博客 |
| Dreamer 4 是否有修订版、官方代码、后续工作 | 第 10 节的细节要对照正文；玩具版讲义最好参考官方实现 |
| Dreamer 4 正文里的预测目标、损失权重、三个阶段的细节 | 第 10.3、10.4 节标了待核对 |
| Cosmos 是否有新版本及对应论文 | 第 10.6 节按 2025 年初的版本写 |
| Diffusion Forcing、Self Forcing 一线的后续（长时记忆、更长的稳定展开） | 第 5.6、9.5 节 |
| 实时交互式视频世界模型的新工作（开源的可玩模型） | 本章项目可能有更合适的参考实现 |
| 视频模型物理一致性的评测工作 | 第 10.7 节的第二个分歧点只引了一篇，且未写编号 |
| DIAMOND、Genie、GameNGen 正文里的超参数 | 第 4.2、8.4、8.5 节标了待核对 |

---

## 13. 实现陷阱与调试清单

| 现象 | 可能的原因 | 怎么排查 | 怎么修 |
| --- | --- | --- | --- |
| 损失正常下降，采样出来全是噪声或一片灰 | 采样的时间方向反了（flow matching 的 $u$ 与扩散的 $\sigma$ 方向相反）；或日程最后没有走到 0 | 打印采样每一步的噪声水平和样本的标准差 | 核对方向；确认日程末尾是 $\sigma = 0$ 或 $u = 1$ |
| 少步采样的画面明显比多步差，且偏色、过曝 | 用了预测噪声的参数化，高噪声步的误差被放大 | 单独看第一步去噪的输出 | 换 EDM 预条件或速度预测 |
| 生成的画面清晰但整体发糊、运动的物体有重影 | 步数太少，输出接近后验均值；或下一帧本身有多种可能 | 把步数加到 10 步对比 | 增加步数；用为少步训练的模型 |
| 训练不稳定，不同噪声水平的损失差几个数量级 | 没做预条件；$\sigma$ 没有按样本广播；损失权重不对 | 按噪声水平分桶打印损失 | 用第 1.7 节的系数；在 $F$ 空间算损失 |
| 隐空间扩散训练正常，采样结果很差 | 隐变量的标准差和 $\sigma_{\text{data}}$ 对不上 | 打印隐变量的均值和标准差 | 归一化隐变量，或改 $\sigma_{\text{data}}$（第 6.3 节） |
| 模型不跟手：换动作，画面几乎不变 | 动作和帧错位；动作嵌入的初始尺度太小；上下文帧太多，模型只靠外推 | 固定上下文换动作，算输出的差异 | 核对对齐；训练时随机丢弃动作并在采样时做引导；减少上下文帧 |
| 操控有固定的一帧延迟 | 数据集里动作的下标约定和模型假设的不一致 | 用一个确定性的小环境核对 $(o_t, a_t, o_{t+1})$ | 统一约定：$a_t$ 是在 $o_t$ 之后执行、导致 $o_{t+1}$ 的动作 |
| 开头几帧很好，几十帧后模糊、变色、物体融化 | 自回归漂移 | 画开环误差随步数的曲线 | 噪声增强；每帧独立噪声水平；缩短想象视野 |
| 加了噪声增强后单步质量下降很多 | 推理时的 $\sigma_c$ 太大；训练时 $\sigma_c$ 的范围太大 | 扫一遍推理时的 $\sigma_c$ | 调小推理时的 $\sigma_c$；缩小训练范围 |
| Diffusion Forcing 训练后，逐帧生成质量差 | 训练时各帧的噪声水平实际上是同步的（采样写成了每个序列一个值） | 打印 `sigmas` 的形状，应为 `[B, T]` | 改成每帧独立采样 |
| 转身回来场景变了 | 上下文窗口不够长，属于遗忘而非漂移 | 把被遗忘的内容放回上下文，看是否恢复 | 加长上下文；这是结构限制，噪声增强不管用 |
| 隐动作模型训练后，所有转移都落在同一个码上 | 码本坍塌；解码器太强，靠过去帧就够了，不需要隐动作 | 统计码的使用频率 | Phase 8 的防坍塌办法；削弱解码器的上下文（比如只给一帧） |
| 隐动作可控性差，随机选码得到的是不同的画面内容而不是不同的运动 | 码本太大，或连续向量的维度太高，瓶颈漏了 | 减小码本重训，看可控性是否上升 | 码本取个位数；检查是否有绕过量化的通路 |
| 隐动作对应的是镜头晃动或背景变化 | 数据里最不可预测的变化不是智能体的动作 | 逐个码可视化它导致的画面变化 | 清洗数据；裁剪到智能体附近；用少量真实动作做辅助监督 |
| 想象的回报很高，真实环境里很低 | 世界模型或奖励模型被策略钻了空子（Phase 1 第 8.5 节） | 把想象的轨迹渲染出来看 | 缩短想象视野；增加真实数据的更新频率 |
| 换成因果注意力后质量崩了 | 模型是双向训练的，直接换掩码改变了输入分布 | 对比双向与因果两种推理 | 按因果结构重新训练或蒸馏 |
| 每帧耗时远超预算 | 没分清瓶颈在步数、网络还是解码 | 用第 9.2 节的公式分项计时 | 针对最大的一项优化 |

调试顺序的建议：先在二维玩具数据上验证训练和采样的代码（几秒钟就能看出对错）；
再做无条件的单帧生成；再加过去帧和动作条件，只看单步；然后看开环展开；最后才接智能体或做实时交互。
这和 Phase 1 第 12 节的顺序是同一个原则：每一层都确认没问题，再往上加一层。

---

## 14. 常见问题（FAQ）

### Q1：扩散模型生成一张图都要很多步，为什么还能当世界模型？

因为世界模型要解决的生成问题比"从零生成一张图"容易得多。给定过去几帧和动作，
下一帧的绝大部分内容已经确定，剩下的不确定性很小（第 3.4 节）。不确定性小，需要的去噪步数就少。
再加上少步下稳定的参数化（EDM、速度预测）和专门为少步设计的训练办法（蒸馏、shortcut），
每帧几步就够。DIAMOND 用个位数的步数，Dreamer 4 和实时的视频模型也在这个量级。

### Q2：DDPM、EDM、flow matching，实际做项目选哪个？

它们学的是同一个东西，换算关系见第 11.2 节。实用的建议：新项目用 flow matching 或 EDM，不用原始的 DDPM。
Flow matching 的代码最短，概念最少，现在大多数新的视频模型和机器人策略（Phase 11 的 π0）用它；
EDM 多几个系数，但少步下的行为被分析得最清楚，DIAMOND 用它。
两者之间不必纠结，更重要的是别用"预测噪声 + 很少的步数"这个组合。

### Q3：为什么不直接用一个网络回归下一帧，非要用扩散？

环境完全确定时，回归就够了，而且更快。问题出在下一帧有多种可能的时候：
平方损失的最优解是所有可能结果的平均，画面上就是模糊和重影（第 3.4 节）。
模糊的帧喂回去当条件，下一帧更模糊，几步之后就没法看了。
扩散模型的一步去噪其实就是这个回归器；它多出来的是"再走几步，从平均走到某一个具体结果"的能力。

### Q4：$\sigma_{\text{data}}$ 怎么定？

它是数据每个分量的标准差。像素归一化到 $[-1, 1]$ 的自然图像，EDM 取 0.5。
换了数据就要重新量：在隐空间里做扩散时，用隐变量的实际标准差，或者先把隐变量归一化到标准差为 1
再取 $\sigma_{\text{data}} = 1$。量错了的后果见第 6.3 节。

### Q5：DIAMOND 在像素里想象，Dreamer 在隐空间里想象，哪个更好？

各有适用范围。像素空间的好处是不丢细节，策略看到的就是和真实环境同样形式的画面；
代价是每步想象要几次 U-Net 前向计算，分辨率一高就撑不住。
隐空间想象便宜得多，但表示学得不好时，对任务重要的细节可能根本没进隐状态。
Dreamer 4 走的是中间路线：先用 tokenizer 压缩，再在压缩后的连续隐变量上做生成式的动力学模型。

### Q6：噪声增强和 Diffusion Forcing 是两种办法吗？要都用吗？

噪声增强是 Diffusion Forcing 的一个特例（第 5.4 节）。如果模型是"过去帧拼接进来、只对下一帧去噪"的结构，
加噪声增强是最小的改动。如果模型本来就是对一段序列做去噪的序列模型，
直接用每帧独立噪声水平训练，噪声增强的效果已经包含在里面，不用再单独加。

### Q7：隐动作的编号和真实动作对不上，怎么办？

对不上是正常的，编号本来就是任意的。三种处理：
人来玩的话，试几次就知道哪个码是什么；
要接到程序上，用一小份带真实动作的数据统计对应关系（第 7.4 节的表），或者训练一个小网络做映射；
如果发现一个真实动作分散在好几个码上，或者一个码混了好几种动作，说明瓶颈的宽度或数据有问题，要回到第 7.3、7.4 节排查。

### Q8：Genie 这样的模型能直接拿来训练机器人吗？

不能直接用。至少缺三样：奖励或任务的定义；隐动作到机器人真实动作的映射；
以及对"生成的物理是否可信"的验证。现在更现实的用法是 Phase 11 讲的那几种：
用视频世界模型评估已有的策略、生成额外的训练数据，或者把隐动作当作预训练目标（LAPA）。

### Q9：MaskGIT 和扩散是什么关系？

可以把 MaskGIT 看成离散 token 上的扩散。"加噪"对应"把 token 换成掩码"，"噪声水平"对应"被遮住的比例"，
"逐步去噪"对应"逐轮揭开"。两者都是从完全不确定的状态出发，分几轮并行地细化所有位置。
区别是一个在离散空间里用交叉熵，一个在连续空间里用平方损失。第 8.4 节的对照表列出了对应关系。

### Q10：采样步数压得很少，世界模型会不会失去随机性？

会有这个风险，要看是怎么压的。直接把多步模型砍到 1 步，输出趋向后验均值，随机性确实没了（第 1.9、3.4 节）。
专门为少步训练的模型（蒸馏、shortcut）是把"从噪声到样本"的整个映射学下来，
不同的初始噪声仍然对应不同的样本，随机性保留在初始噪声里。
检查办法：固定上下文和动作，换不同的随机种子生成多次，看结果在该有分歧的地方有没有分歧。

### Q11：本机（M1 Max）能做本章的哪些内容？

讲义都设计成低分辨率的缩小版，在本机上能跑通。第 1、2 节用二维玩具数据，很快。
第 3–8 节用小画面的环境。完整规模的实战项目建议用 4090。
各节的实际耗时要等讲义写完实测，README 的算力一节会更新。
注意 MPS 不支持 float64，噪声日程这类计算在别的设备上常用 float64，移植代码时要改成 float32。

### Q12：视频基础模型的规模个人根本训练不动，学这一章有什么用？

三个用处。第一，读得懂：新论文的方法部分基本是本章零件的重新组合，学完能很快判断它改了哪个零件。
第二，用得上：这些大模型越来越多地以开放权重的形式发布，实际工作是在它们上面做后训练、蒸馏、接动作条件，
做这些事需要的正是本章的知识。第三，小规模同样有价值：在一个具体任务（比如 PushT）上，
一个小的扩散世界模型就足以回答"这类模型对我的任务有没有用"。

### Q13：本章的扩散世界模型和 Phase 11 的 Diffusion Policy 是什么关系？

用的是同一套扩散技术，生成的东西不同。本章生成的是**未来的画面**，条件是动作；
Diffusion Policy 生成的是**未来的动作序列**，条件是画面。把两者放进同一个模型，
同时对画面和动作做扩散，就是 Phase 11 要讲的 World-Action Model。
本章第 5.5 节那张二维的噪声日程表在那里会再次出现：画面和动作可以各有各的噪声水平。

---

## 15. 自测题

做题时先不要看答案。概念题试着用自己的话说出来，计算题动手算。

### 第 1 题

DDPM 某一步的 $\bar\alpha_k = 0.2$。这一步对应的噪声水平 $\sigma$ 是多少？用一句话说出 $\sigma$ 的含义。

<details><summary>参考答案</summary>

$$
\sigma = \sqrt{\frac{1 - \bar\alpha_k}{\bar\alpha_k}} = \sqrt{\frac{0.8}{0.2}} = 2
$$

含义：把这一步的带噪样本除以 $\sqrt{\bar\alpha_k}$ 之后，它等于干净样本加上标准差为 2 的高斯噪声。

</details>

### 第 2 题

显式 score matching 的目标为什么不能直接用？去噪 score matching 用什么替换了回归目标？
推导中最关键的一步是什么？

<details><summary>参考答案</summary>

不能直接用，是因为回归目标 $\nabla \log p_\sigma(x_\sigma)$ 里含有未知的数据分布，算不出来。

去噪 score matching 把目标换成条件分布的 score：
$\nabla_{x_\sigma} \log p_\sigma(x_\sigma \mid x) = -(x_\sigma - x)/\sigma^2 = -\epsilon/\sigma$。
它只涉及一个高斯，可以直接算。

关键的一步是把边际分布的梯度写成条件 score 的加权积分：

$$
\nabla p_\sigma(x_\sigma) = \int p_{\text{data}}(x)\, p_\sigma(x_\sigma \mid x)\, \nabla \log p_\sigma(x_\sigma \mid x)\, dx
$$

于是交叉项可以写成对 $(x, x_\sigma)$ 联合分布的期望。两个目标只差一个与 $\theta$ 无关的常数。

</details>

### 第 3 题

带噪样本 $x_\sigma = 1.0$，噪声水平 $\sigma = 0.5$，网络预测的噪声是 $\epsilon_\theta = 0.4$。
换算成对干净样本的预测和对 score 的预测。

<details><summary>参考答案</summary>

干净样本：$D_\theta = x_\sigma - \sigma\,\epsilon_\theta = 1.0 - 0.5 \times 0.4 = 0.8$

score：$-\epsilon_\theta / \sigma = -0.4 / 0.5 = -0.8$

用另一条换算验证：$(D_\theta - x_\sigma)/\sigma^2 = (0.8 - 1.0)/0.25 = -0.8$。一致。

</details>

### 第 4 题

数据服从 $\mathcal{N}(0, 1)$，噪声水平 $\sigma = 1$，观测到 $x_\sigma = 3$。
求最优去噪器的输出和真实的 score，并用 Tweedie 公式验证。

<details><summary>参考答案</summary>

边际分布是 $\mathcal{N}(0, 1 + 1) = \mathcal{N}(0, 2)$。

score：$-x_\sigma / 2 = -1.5$

最优去噪器（后验均值）：$\dfrac{1}{1 + 1} \times 3 = 1.5$

Tweedie 公式：$x_\sigma + \sigma^2 \times \text{score} = 3 + 1 \times (-1.5) = 1.5$。一致。

去噪器把观测值往数据的均值 0 拉回了一半，因为噪声和数据的方差相等。

</details>

### 第 5 题

取 $\sigma_{\text{data}} = 1$、$\sigma = 1$，算出 EDM 的 $c_{\text{in}}$、$c_{\text{skip}}$、$c_{\text{out}}$，
写出网络的回归目标 $F_{\text{target}}$，并验证它的方差是 1。

<details><summary>参考答案</summary>

$\sigma^2 + \sigma_{\text{data}}^2 = 2$，开方 1.414。

$$
c_{\text{in}} = \frac{1}{1.414} = 0.707, \qquad
c_{\text{skip}} = \frac{1}{2} = 0.5, \qquad
c_{\text{out}} = \frac{1 \times 1}{1.414} = 0.707
$$

$$
F_{\text{target}} = \frac{x - c_{\text{skip}}\, x_\sigma}{c_{\text{out}}} = \frac{0.5\,x - 0.5\,\epsilon}{0.707} = 0.707\,(x - \epsilon)
$$

方差：$0.707^2 \times (1 + 1) = 0.5 \times 2 = 1$。

顺带注意，这个目标正比于 $x - \epsilon$，就是 flow matching 的速度目标。

</details>

### 第 6 题

EDM 的四个预条件系数分别由什么要求定出来？为什么要让 $c_{\text{out}}$ 尽可能小？

<details><summary>参考答案</summary>

- $c_{\text{in}}$：让网络的输入是单位方差
- $c_{\text{out}}$ 和 $c_{\text{skip}}$：让网络的回归目标是单位方差（一个方程）；在此约束下让 $c_{\text{out}}$ 最小（第二个方程）
- $c_{\text{noise}}$：经验选择，把 $\sigma$ 取对数后喂给网络

让 $c_{\text{out}}$ 小的原因：去噪器的输出是 $c_{\text{skip}}\, x_\sigma + c_{\text{out}}\, F_\theta$，
网络输出上的误差被乘以 $c_{\text{out}}$ 后进入去噪结果。$c_{\text{out}}$ 越小，网络的错被放大得越少。

结果是：低噪声时 $c_{\text{skip}} \to 1$，网络只预测一个小的修正（接近预测噪声）；
高噪声时 $c_{\text{skip}} \to 0$，网络直接预测干净样本。

</details>

### 第 7 题

Euler 采样的某一步：当前样本 $x = 4.0$，当前噪声水平 2，下一个噪声水平 1，去噪器输出 1.0。求下一个样本。

<details><summary>参考答案</summary>

$$
x' = D + \frac{\sigma_{i+1}}{\sigma_i}\,(x - D) = 1.0 + \frac{1}{2} \times (4.0 - 1.0) = 2.5
$$

读法：去噪器认为干净样本是 1.0，当前样本里有 3.0 的"噪声"，保留一半。

</details>

### 第 8 题

写出条件 flow matching 的损失。它和边际 flow matching 损失的梯度为什么相同？
另外，$u = 0.8$ 对应的噪声水平 $\sigma$ 是多少？

<details><summary>参考答案</summary>

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{u, \epsilon, x} \big\Vert v_\theta(x_u, u) - (x - \epsilon) \big\Vert^2, \qquad x_u = (1 - u)\,\epsilon + u\,x
$$

梯度相同的原因：把两个损失的平方展开，$\Vert v_\theta \Vert^2$ 项相同；与 $\theta$ 无关的项不影响梯度；
交叉项用全期望公式，$\mathbb{E}[v_\theta^\top (x - \epsilon)] = \mathbb{E}_{x_u}\big[ v_\theta^\top\, \mathbb{E}[x - \epsilon \mid x_u] \big]$，
而 $\mathbb{E}[x - \epsilon \mid x_u]$ 正是边际速度场。

$\sigma = (1 - u)/u = 0.2 / 0.8 = 0.25$。

</details>

### 第 9 题

噪声 $\epsilon = -1$，数据 $x = 3$，路径时间 $u = 0.25$。求 $x_u$ 和这个样本的回归目标。
如果网络在这一点恰好输出了这个目标，换算出它对干净样本和噪声的预测。

<details><summary>参考答案</summary>

$x_u = 0.75 \times (-1) + 0.25 \times 3 = 0$

回归目标：$x - \epsilon = 3 - (-1) = 4$

对干净样本的预测：$x_u + (1 - u)\,v = 0 + 0.75 \times 4 = 3$

对噪声的预测：$x_u - u\,v = 0 - 0.25 \times 4 = -1$

两者都还原了原来的 $x$ 和 $\epsilon$，说明换算公式是对的。

</details>

### 第 10 题

下一帧里小球在 $-1$ 或 $+1$，各一半概率。噪声水平 $\sigma = 2$ 时观测到 $x_\sigma = 1$。
最优去噪器输出多少？这说明一步采样会有什么问题？已知 $\tanh(0.25) = 0.245$。

<details><summary>参考答案</summary>

$$
D^\star = \tanh\Big( \frac{x_\sigma}{\sigma^2} \Big) = \tanh(0.25) = 0.245
$$

输出既不是 $-1$ 也不是 $+1$，而是靠近 0 的一个中间值：两种可能的加权平均。
一步采样直接把这个值当作结果，画面上就是小球在两个位置各有一个淡淡的影子。
多步采样会在噪声水平逐步降低的过程中落到其中一个位置上。

</details>

### 第 11 题

说出 DIAMOND 的三个部件。批大小 64、想象视野 15、每帧去噪 3 步时，
想象一个批次需要多少次单帧去噪？批量化之后是多少次 U-Net 前向计算？

<details><summary>参考答案</summary>

三个部件：以过去帧和动作为条件的扩散模型（生成下一帧）；奖励 / 终止模型；在想象里训练的 actor-critic。

单帧去噪次数：$64 \times 15 \times 3 = 2880$

批量化：每个想象步的每个去噪步是一次批大小为 64 的前向计算，共 $15 \times 3 = 45$ 次。

</details>

### 第 12 题

用 Phase 1 的复合误差界回答：

1. 单步误差 0.02，$L = 1.05$，展开 40 步的误差上界是多少？已知 $1.05^{40} = 7.04$。
2. 噪声增强把 $L$ 降到 0.9，单步误差不变，误差上界是多少？
3. 噪声增强是通过哪两个机制起作用的？

<details><summary>参考答案</summary>

1. $0.02 \times \dfrac{7.04 - 1}{0.05} = 0.02 \times 120.8 = 2.42$
2. $L < 1$ 时上界是 $\varepsilon / (1 - L) = 0.02 / 0.1 = 0.2$，而且不随步数增长
3. 一是消除分布偏移：上下文带噪之后，带瑕疵的生成帧和带噪的真实帧在分布上很接近，模型见过这种输入。
   二是降低敏感度：模型没法依赖上下文里的高频细节，上下文的小瑕疵对输出的影响变小，也就是 $L$ 变小。

</details>

### 第 13 题

用"每帧的噪声水平"描述下一帧预测、全序列扩散、Diffusion Forcing 三种训练方式。
Diffusion Forcing 在自回归展开时怎样防止漂移？

<details><summary>参考答案</summary>

- 下一帧预测：历史帧噪声水平为 0，只有最后一帧带噪
- 全序列扩散：所有帧的噪声水平相同
- Diffusion Forcing：每一帧的噪声水平独立随机采样

防漂移：把已经生成的帧喂回去时，不把它们标成"完全干净"，而是标一个略大于 0 的噪声水平。
模型在训练中见过"历史帧带一点噪声"的情况，知道此时不该完全相信历史帧的细节。
这相当于噪声增强，而且是训练分布内的操作。

</details>

### 第 14 题

一个一维世界里，位置变化 $\Delta p$ 取 $0$ 的概率是 1/2，取 $-2$ 和 $+2$ 的概率各 1/4。
隐动作码本大小分别为 1、2、3 时，解码器预测 $\Delta p$ 的最小平方误差各是多少？
码本大小为 16 时，瓶颈能传多少比特？

<details><summary>参考答案</summary>

**$M = 1$**：只能预测均值 0。误差 $= \frac{1}{4} \times 4 + \frac{1}{4} \times 4 = 2$。

**$M = 2$**：最优分组是把 $-2$ 单独一组，$0$ 和 $+2$ 一组（或对称的另一种）。
第二组的均值是 $\dfrac{0 \times 0.5 + 2 \times 0.25}{0.75} = 0.667$。

误差 $= 0.5 \times 0.667^2 + 0.25 \times (2 - 0.667)^2 = 0.222 + 0.444 = 0.667$。

（如果分成 $\lbrace 0 \rbrace$ 和 $\lbrace -2, +2 \rbrace$，第二组均值是 0，误差仍是 2，没有改善。）

**$M = 3$**：每种变化一个码，误差为 0。

**$M = 16$**：$\log_2 16 = 4$ 比特。

</details>

### 第 15 题

Genie 的三个部件是什么？按什么顺序训练？交互时哪些部分被丢掉了？
另外，$T = 8$ 帧、每帧 $S = 100$ 个 token 时，全注意力与时空分解注意力的代价之比是多少？

<details><summary>参考答案</summary>

三个部件：视频 tokenizer、隐动作模型、动力学模型（MaskGIT）。

训练顺序：tokenizer 和隐动作模型先各自训练（互不依赖），冻结；再训练动力学模型。

交互时：隐动作模型的编码器和解码器都不用了，只保留它的码本（用户直接选码）。
tokenizer 的编码器只用来编码提示图像，之后主要用解码器。

注意力代价：全注意力 $(8 \times 100)^2 = 640{,}000$；
时空分解 $8 \times 100^2 + 100 \times 8^2 = 80{,}000 + 6{,}400 = 86{,}400$。比值约 7.4。

</details>

### 第 16 题

推导 shortcut 模型的自洽条件。然后计算：在某一点，步长 0.25 的平均速度是 1.0；
走完这一步后，下一段步长 0.25 的平均速度是 3.0。步长 0.5 的平均速度应该是多少？
从 $x = 0.5$ 出发，用两种走法验证终点一致。

<details><summary>参考答案</summary>

记步长为 $d$ 的平均速度为 $\bar{v}(x_u, u, d)$，定义是 $x_{u+d} = x_u + d \cdot \bar{v}(x_u, u, d)$。

两小步：$x_{u+2d} = x_u + d \cdot \bar{v}(x_u, u, d) + d \cdot \bar{v}(x_{u+d}, u + d, d)$

一大步：$x_{u+2d} = x_u + 2d \cdot \bar{v}(x_u, u, 2d)$

令两者相等：

$$
\bar{v}(x_u, u, 2d) = \frac{1}{2}\, \bar{v}(x_u, u, d) + \frac{1}{2}\, \bar{v}(x_{u+d}, u + d, d)
$$

计算：$(1.0 + 3.0)/2 = 2.0$。

验证：两小步 $0.5 + 0.25 \times 1.0 = 0.75$，再 $0.75 + 0.25 \times 3.0 = 1.5$；
一大步 $0.5 + 0.5 \times 2.0 = 1.5$。一致。

</details>

### 第 17 题

一个交互式世界模型：网络前向一次 8 毫秒，解码 5 毫秒，其他开销 2 毫秒。目标每秒 30 帧。
最多能用几步去噪？如果用了 classifier-free guidance（每步两次网络调用）呢？

<details><summary>参考答案</summary>

每帧预算 $1000 / 30 = 33.3$ 毫秒。扣掉解码和其他开销，剩 26.3 毫秒给网络。

不用引导：$26.3 / 8 = 3.3$，最多 3 步。

用引导：每步 16 毫秒，$26.3 / 16 = 1.6$，只能 1 步。

这说明在实时场景里，引导的代价是直接把可用的步数砍半。

</details>

### 第 18 题

有人给你看一个文本生成视频的大模型，画面非常逼真，并说"这就是世界模型"。
按本课程的定义，你会追问哪几件事？

<details><summary>参考答案</summary>

1. **能以动作为条件吗？** 动作的粒度有多细？文字描述不算机器人能执行的动作。
2. **换一个动作，结果会相应地变吗？** 即可控性，而不只是画面质量。
3. **物理上对吗？** 尤其是训练分布之外的情况。逼真不等于正确。
4. **能自回归地展开多久？** 是因果的吗？长展开会不会漂移？转身回来场景还在吗？
5. **推理多快？** 能不能支撑交互，或者支撑在里面训练策略、做规划。
6. **对决策有用吗？** 有没有证据表明在它里面评估或训练出的策略，和真实环境里的表现一致。

本课程的定义要求两点：以动作为条件，预测能用来帮助选择动作。只满足"生成逼真视频"还不够。

</details>

---

## 16. 小结与下章预告

### 16.1 本章要点

1. **去噪就是学 score。** 显式 score matching 的目标算不出来，换成"回归条件 score"之后梯度不变。
   预测噪声、预测干净样本、预测 score 三者的最优解可以互相换算：
   $\nabla \log p_\sigma = -\epsilon/\sigma = (D - x_\sigma)/\sigma^2$。
2. **参数化决定少步时稳不稳。** EDM 的预条件让网络的输入和回归目标都是单位方差，
   并把网络误差被放大的倍数压到最小。结果是低噪声时预测噪声、高噪声时预测干净样本。
3. **Flow matching 是同一件事的更简洁的写法。** 线性路径 $x_u = (1 - u)\epsilon + u x$，回归速度 $x - \epsilon$。
   条件损失与边际损失梯度相同。条件路径是直的，采样轨迹不一定直。
4. **世界模型里的扩散只需要几步。** 给定过去帧和动作，下一帧的不确定性很小。
   步数太少时输出趋向后验均值，表现为模糊。
5. **DIAMOND**：像素空间的扩散世界模型加上在想象中训练的 actor-critic。
   选 EDM 是因为少步加自回归会放大参数化的差别。视觉细节之所以重要，是因为小物体的缺失会一路传到价值目标。
6. **自回归漂移就是像素空间的复合误差。** 噪声增强同时消除分布偏移并降低敏感度。
   Diffusion Forcing 让每帧有独立的噪声水平，把下一帧预测和全序列扩散统一起来。
7. **先压缩再建模。** 视频 tokenizer 决定了世界模型的速度和上限。隐变量的尺度、动作对齐、因果性是三个要点。
8. **隐动作模型**靠一个极窄的 VQ 瓶颈，从相邻帧的变化里聚类出离散的"动作"。瓶颈宽了就会泄露未来。
9. **Genie** 等于视频 tokenizer 加隐动作模型加 MaskGIT 动力学模型，分阶段训练，只需要视频。
10. **实时**靠少步（蒸馏、shortcut）和因果（因果注意力、KV cache）。shortcut 模型用"两小步等于一大步"自举出大步长的目标。
11. **Dreamer 4** 把这些零件装回"在想象中训练智能体"的框架，并且只用离线数据。
    视频基础模型能不能算世界模型，取决于动作条件、物理正确性、长时一致性和对决策的用处。

### 16.2 一张图回顾

```
      问题                          本章的办法                       留下的问题
  ──────────────────────────────────────────────────────────────────────────────────
   离散 token 丢细节       ──▶   用扩散模型生成连续的画面          ──▶   生成每个像素很贵

   扩散采样太慢            ──▶   EDM 预条件；flow matching；      ──▶   仍然是每帧几次大网络前向，
                                少步；蒸馏；shortcut                   配规划器用不起

   一帧接一帧会跑偏         ──▶   噪声增强；Diffusion Forcing；     ──▶   长时间的记忆和一致性
                                训练时也展开

   没有动作标签            ──▶   隐动作模型；逆动力学打伪标签        ──▶   隐动作怎么对应到机器人的动作
                                                                    （Phase 11）

   分辨率高了算不动         ──▶   视频 tokenizer + 隐空间扩散       ──▶   压缩丢掉的细节补不回来

   画面逼真 ≠ 对决策有用    ──▶   可控性指标；在模型里训练策略        ──▶   怎么系统地评估世界模型可不可信
                                再回真实环境检验                      （Phase 11）
```

### 16.3 下章预告

本章的每一种方法，最后都要生成画面：像素，或者能解码成像素的隐变量。
这带来三个始终没有解决的问题：

- **贵。** 每想象一步都要跑几次大网络。第 3.5 节算过，配上 Phase 1 的规划器，每个真实步要几万次网络调用。
  所以本章的世界模型基本只用来训练策略，没法在决策时规划。
- **很多细节不可预测，也不需要预测。** 树叶怎么晃、水面怎么反光，模型花了大量容量去生成，
  而这些对"杯子拿起来了没有"毫无影响。
- **训练目标和用途不一致。** 损失衡量的是画面像不像，我们要的是决策对不对。

[Phase 10](../phase-10-jepa/CONCEPT.md) 换一个思路：**不生成像素。** 编码器把画面变成表征，
预测器直接预测未来画面的表征，全程不解码。这就是联合嵌入预测架构（JEPA）。
预测变便宜了，Phase 1 的 CEM 又能用上了：在特征空间里展开几百条候选序列，代价是可以接受的。
PushT 上的三方对比（Token、扩散、JEPA）在那里补齐。

它要面对一个本章不存在的问题：如果编码器对所有画面都输出同一个向量，预测误差直接为零。
本章的模型有像素重建这个"锚"，不会出这种事；去掉了锚，就必须另想办法防止表征坍塌。

本章的另一半去向是 [Phase 11](../phase-11-robot-world-models/CONCEPT.md)。
那里的视频预测策略、世界模型当仿真器、当数据引擎，用的都是本章的扩散与视频世界模型；
第 7 节的逆动力学会成为"从预测的画面里解出动作"的关键部件；
第 10.7 节留下的"怎么判断一个世界模型可不可信"，会用策略评估的相关性指标来回答。

---

## 附录 A：术语表（中英对照）

| 中文 | 英文 | 一句话解释 |
| --- | --- | --- |
| 扩散模型 | diffusion model | 学会在各种噪声水平下去噪，再从纯噪声逐步去噪来生成样本 |
| 噪声水平 | noise level | 加到样本上的高斯噪声的标准差 $\sigma$ |
| 噪声日程 | noise schedule | 训练或采样时使用的一串噪声水平 |
| score | score function | 对数密度对样本的梯度，指向密度升高的方向 |
| 显式 score matching | explicit score matching | 直接回归真实 score 的目标，算不出来 |
| 去噪 score matching | denoising score matching | 回归条件 score 的目标，可计算，梯度与前者相同 |
| 去噪器 | denoiser | 输入带噪样本和噪声水平，输出对干净样本的估计 |
| Tweedie 公式 | Tweedie's formula | 后验均值 = 带噪样本 + $\sigma^2$ 乘以 score |
| 预条件 | preconditioning | 用只依赖噪声水平的系数包住网络，使输入输出尺度稳定 |
| 概率流常微分方程 | probability flow ODE | 一个确定性的方程，沿它演化的样本分布与扩散过程一致 |
| 网络调用次数 | number of function evaluations (NFE) | 生成一个样本要调用多少次网络 |
| 速度预测 | v-prediction | 让网络预测沿加噪路径的速度，是数据和噪声的线性组合 |
| 流匹配 | flow matching | 直接回归把噪声搬运到数据的速度场 |
| 条件流匹配 | conditional flow matching | 回归单条（噪声，数据）连线的速度，梯度与边际目标相同 |
| 整流流 | rectified flow | 线性插值路径的 flow matching，并用 reflow 把轨迹拉直 |
| 无分类器引导 | classifier-free guidance | 沿"有条件减无条件"的方向外推，放大条件的作用 |
| 自回归漂移 | autoregressive drift | 生成的帧被当作条件，误差逐帧累积。即像素空间的复合误差 |
| 曝光偏差 | exposure bias | 训练时条件是真实数据，推理时条件是自己的输出 |
| 教师强制 | teacher forcing | 训练时每一步的输入都用真实数据 |
| 噪声增强 | noise augmentation | 训练时给条件帧加噪，并把噪声水平告诉模型 |
| 扩散强制 | Diffusion Forcing | 序列里每个元素有独立的噪声水平的训练方式 |
| 用历史做引导 | history guidance | 把历史帧当作条件做无分类器引导 |
| 视频 tokenizer | video tokenizer | 把视频压缩成紧凑表示（离散或连续）的编码器与解码器 |
| 隐空间扩散 | latent diffusion | 在自编码器的隐变量上做扩散，而不是在像素上 |
| 因果 tokenizer | causal tokenizer | 编码每一帧时只看当前和过去的帧 |
| 隐动作 | latent action | 从无标注视频里学出来的离散码，起到动作的作用 |
| 隐动作模型 | latent action model (LAM) | 用 VQ 瓶颈从相邻帧里学隐动作的模型 |
| 逆动力学模型 | inverse dynamics model (IDM) | 给前后两帧，推断中间的动作 |
| 正向动力学模型 | forward dynamics model | 给当前帧和动作，预测下一帧 |
| 信息瓶颈 | information bottleneck | 故意限制能通过的信息量，逼模型只传最关键的部分 |
| 时空 Transformer | spatiotemporal (ST) transformer | 帧内的空间注意力和跨帧的时间注意力交替 |
| 生成式交互环境 | generative interactive environment | 由生成模型构成、可以逐帧操控的环境 |
| 蒸馏 | distillation | 让少步的学生模型复现多步老师模型的结果 |
| 一致性模型 | consistency model | 把采样轨迹上任一点直接映射到终点的模型 |
| 分布匹配蒸馏 | distribution matching distillation | 让学生的输出分布与老师一致的蒸馏方法 |
| shortcut 模型 | shortcut model | 以步长为条件、支持任意步数采样的模型 |
| 自洽条件 | self-consistency | 一大步等于连续两小步 |
| 因果化 | causalization | 把双向的视频模型改成只依赖过去的自回归模型 |
| 流式生成 | streaming generation | 生成一帧、输出一帧，不必等整段完成 |
| 神经游戏引擎 | neural game engine | 用神经网络取代传统引擎，根据输入实时生成画面 |
| 世界基础模型 | world foundation model | 在大规模视频上预训练、可后训练到具体用途的世界模型 |
| 峰值信噪比 | PSNR | 逐像素误差的对数度量，越高越好 |
| Fréchet 视频距离 | Fréchet Video Distance (FVD) | 生成视频与真实视频在特征分布上的距离 |

---

## 附录 B：延伸阅读

**必读**

- Alonso et al. 2024, DIAMOND（arXiv:2405.12399）。先读引言、方法、关于 EDM 与步数的分析，再看附录的超参数。
- Bruce et al. 2024, Genie（arXiv:2402.15391）。方法部分的三个部件和消融实验。
- Karras et al. 2022, EDM（arXiv:2206.00364）。第 1–3 节和 Table 1；预条件的推导在附录。

**重点读**

- Chen et al. 2024, Diffusion Forcing（arXiv:2407.01392）。方法部分和稳定化的讨论；理论部分可以略读。
- Hafner et al. 2025, Dreamer 4（arXiv:2509.24527）。读的时候把本章标了待核对的地方逐条核实。
- Frans et al. 2024, Shortcut Models（arXiv:2410.12557）。短，推导清楚，先于 Dreamer 4 读。
- Lipman et al. 2023, Flow Matching（arXiv:2210.02747）与 Liu et al. 2022, Rectified Flow（arXiv:2209.03003）。
  两篇读一篇即可。Rectified Flow 更直观。

**选读：实时与因果**

- Valevski et al. 2024, GameNGen（arXiv:2408.14837）。噪声增强的消融。
- Yin et al. 2024, CausVid（arXiv:2412.07772）。
- Huang et al. 2025, Self Forcing（arXiv:2506.08009）。
- Song et al. 2025, History-Guided Video Diffusion（arXiv:2502.06764）。

**选读：从视频里学动作**

- Baker et al. 2022, VPT（arXiv:2206.11795）。
- Ye et al. 2024, LAPA（arXiv:2410.11758）。Phase 11 会再用到。

**选读：视频基础模型**

- Yang et al. 2023, UniSim（arXiv:2310.06114）。
- NVIDIA 2025, Cosmos（arXiv:2501.03575）。篇幅很长，挑 tokenizer 和后训练两部分读。
- Kanervisto et al. 2025, WHAM（Nature 638）。
- Genie 2、Genie 3 的官方博客。

**背景：扩散模型的基础**

- Ho et al. 2020, DDPM（arXiv:2006.11239）。旧课学过，按需回看。
- Song et al. 2021, Score SDE（arXiv:2011.13456）。概率流 ODE 的出处。本章第 1.9 节没有推导的那个方程在这里。
- Rombach et al. 2022, Latent Diffusion（arXiv:2112.10752）。
- Peebles & Xie 2023, DiT（arXiv:2212.09748）。用 Transformer 当扩散骨干，后来的视频模型多沿用。
- Ho et al. 2022, Video Diffusion Models（arXiv:2204.03458）。

**本章提到但未列入论文表的工作**（编号未经核对，故不写；写讲义前核对后可补进 README 的论文表）

- Vincent 2011, A Connection Between Score Matching and Denoising Autoencoders（Neural Computation）。去噪 score matching 的出处。
- Salimans & Ho 2022, Progressive Distillation for Fast Sampling of Diffusion Models。v-prediction 与逐级减半蒸馏。
- Ho & Salimans 2022, Classifier-Free Diffusion Guidance。
- Song et al. 2023, Consistency Models。
- Kang et al. 2024, How Far is Video Generation from World Model: A Physical Law Perspective。

**预习**

- Assran et al. 2023, I-JEPA（arXiv:2301.08243）。先读引言，体会"不生成像素"的动机。Phase 10 精读。
- Zhou et al. 2024, DINO-WM（arXiv:2411.04983）。它的实验里就有 PushT，可以提前看一眼它怎么和生成式世界模型比较。

