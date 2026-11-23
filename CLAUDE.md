# CLAUDE.md

给 Claude Code 的工作说明：这个仓库是什么、内容怎么写、现在写到哪一步。

## 这是什么

一门自学课程：**世界模型（World Models）从 0 到前沿**，应用方向侧重具身智能 / 机器人。

- **学员**：资深程序员（全栈背景），已学完 `deep-learning-course`（PyTorch、CNN、RNN、Transformer、VAE、扩散模型）和 `llm-course`。**没学过强化学习、规划与控制**，这部分要从 0 讲。
- **语言**：讲解、注释、程序输出一律用中文；代码标识符用英文；术语第一次出现时写成「中文（English）」。
- **形式**：12 个阶段（phase）。每个阶段 = 一份 `CONCEPT.md`（重点）+ 一份 `README.md` + 8–10 节可运行讲义（`.py`）+ 一个实战项目。

## 当前工作顺序：概念先行

1. 先写完全部 12 章的 `README.md` 和 `CONCEPT.md`。
2. 再逐章写讲义并跑通，用实测结果回填 `CONCEPT.md` 里标了「📌 待回填」的位置。
3. 最后写该章的实战项目。

进度以各章 `README.md` 顶部的状态行为准。完成一项就更新状态行。

## 目录结构

```
world-models-course/
├── CLAUDE.md
├── pyproject.toml / uv.lock / .python-version     # Python 3.12，uv 管理
├── scripts/check_course.py                        # 文档检查 + 讲义冒烟运行
├── src/
│   ├── utils/                                     # 只放基础设施：设备、种子、画图、存视频
│   └── phase-NN-<slug>/
│       ├── README.md                              # 章节大纲与进度
│       ├── CONCEPT.md                             # 本章核心概念精讲（重点）
│       └── NN-<slug>.py                           # 讲义，单文件可运行
├── projects/phase-NN-<slug>/                      # 阶段实战项目
└── outputs/                                       # 运行产物（图、视频），不入库
```

## 常用命令

```bash
uv sync                                            # 安装依赖
uv run python src/phase-01-dynamics-and-planning/01-what-is-a-world-model.py   # 运行一节讲义
WM_QUICK=1 uv run python <讲义路径>                  # 冒烟模式：缩小训练预算，只验证能跑通
uv run python scripts/check_course.py              # 检查全部章节文档
uv run python scripts/check_course.py phase-03     # 只检查某一章
uv run python scripts/check_course.py --run phase-01   # 冒烟运行某一章的全部讲义
uv run pytest                                      # 项目测试
```

## 课程大纲

| 阶段 | 目录 | 周次 | 主题 |
| --- | --- | --- | --- |
| **第一部分：学模型 + 做规划**（不需要 RL 理论） | | | |
| P1 | `phase-01-dynamics-and-planning` | 1–2 | 世界模型全景；状态动力学模型；MPC / CEM / PETS |
| P2 | `phase-02-world-models-2018` | 3–4 | Ha & Schmidhuber：VAE + MDN-RNN + CMA-ES，梦境训练 |
| P3 | `phase-03-rssm-planet` | 5–6 | 序列 VAE、RSSM、隐空间规划（PlaNet） |
| **第二部分：在想象中学习** | | | |
| P4 | `phase-04-rl-value-based` | 7–8 | RL 核心（上）：值函数、TD、λ-return、DQN、Dyna |
| P5 | `phase-05-rl-policy-gradient` | 9–10 | RL 核心（下）：策略梯度、PPO、SAC、MBPO、梯度估计器 |
| P6 | `phase-06-dreamer` | 11–13 | Dreamer V1 → V3 |
| P7 | `phase-07-latent-planning` | 14–15 | MuZero 与 TD-MPC2（不重构观测的世界模型） |
| **第三部分：生成式与表征式世界模型** | | | |
| P8 | `phase-08-transformer-world-models` | 16–17 | VQ-VAE / FSQ、IRIS、STORM、MaskGIT |
| P9 | `phase-09-diffusion-video-world-models` | 18–20 | 扩散世界模型、Diffusion Forcing、Genie、Dreamer 4 |
| P10 | `phase-10-jepa` | 21–22 | I-JEPA / V-JEPA 2、DINO-WM、端到端 JEPA |
| P11 | `phase-11-robot-world-models` | 23–24 | 机器人世界模型专题 |
| P12 | `phase-12-capstone` | 25–27 | 研究工程方法 + 毕业项目 |

三条贯穿全课的线索，写任何一章都要接上：

1. **同一个规划器**：P1 写的 CEM / MPPI 一直用到 PlaNet（P3）、TD-MPC2（P7）、DINO-WM（P10）。
2. **同一个任务**：PushT 推块任务贯穿 P8–P11，在同一任务上对比 Token、扩散、JEPA 三类世界模型。
3. **同一张排行榜**：P5 建立连续控制对比基准（PETS / SAC / MBPO），P6、P7 的 Dreamer、TD-MPC2 继续加入。

编排原则是**世界模型优先**：P1–P3 只用"学模型 + 规划"，不依赖 RL 理论；RL 放在 P4–P5，只讲 Dreamer 等后续章节真正用到的内容（λ-return、三种梯度估计器、SAC、Dyna）。TRPO 推导、资格迹、探索奖励、离线 RL、多智能体不讲。

## CONCEPT.md 规范（最重要）

每章一份，位于 `src/phase-NN-<slug>/CONCEPT.md`。它是这门课的主教材：学员先读它，再跑讲义。
**风格与深度以 `src/phase-01-dynamics-and-planning/CONCEPT.md` 为准**，写新章节前先读它。

### 固定结构

```markdown
# {章标题}：{副标题}

> Phase N · 概念精讲 ｜ 对应讲义：`src/phase-NN-<slug>/` ｜ 预计阅读：N 小时
>
> 一两句话说清这一章解决什么问题。

## 目录

## 0. 本章导读
### 0.1 本章要回答的问题        （3–5 个）
### 0.2 一图总览                （ASCII 图）
### 0.3 前置知识检查            （指向旧课或本课前面的章节）
### 0.4 概念与讲义对照表        （本章每一节讲义都必须出现在表里）
### 0.5 符号表

## 1. … N. 核心概念              （每节一个概念，见下面的"六步"）

## N+1. 方法对比与选型
## N+2. 论文精读卡片
## N+3. 实现陷阱与调试清单      （表格：现象 → 原因 → 排查 → 修复，不少于 8 行）
## N+4. 常见问题（FAQ）          （不少于 10 条）
## N+5. 自测题                  （不少于 12 题，答案用 <details> 折叠）
## N+6. 小结与下章预告

## 附录 A：术语表（中英对照）
## 附录 B：延伸阅读
```

### 每个核心概念按六步写

1. **直觉**：先用一个具体例子或工程师熟悉的类比把问题讲明白，不出现公式。
2. **定义**：给出形式化定义，所有符号都能在符号表里查到。
3. **推导**：核心公式逐步推导，每一步写明用了什么（哪个定义、哪个恒等式）。不跳步。推导后给一个能手算的数值例子。
4. **算法**：伪代码。
5. **代码对应**：对应哪节讲义的哪个函数；关键张量的形状写成 `[B, T, D]`。
6. **常见误区**：2–4 条，写成"误区 → 实际情况"。

顺序固定。某一步确实不适用可以省略（例如纯概念的小节没有算法），但"直觉"和"常见误区"不能省。

### 质量线

- **篇幅**：核心章不少于 1200 行；P1 和 P12 不少于 800 行。不为凑行数灌水。
- **推导**：各章必须包含的推导见下表。
- **公式**：用 LaTeX。行内 `$...$`；独立公式 `$$` 单独占行，多行用 `\begin{aligned}`。
  表格单元格里的公式不能出现竖线字符：条件概率写 `\mid`，范数和 KL 写 `\Vert`。
- **图**：架构图和数据流图用代码块里的 ASCII 图，沿用旧课风格。
- **自测题答案**：`<details><summary>参考答案</summary>` 之后空一行再写内容。
- **衔接**：开头说明承接前面哪一章，结尾的"下章预告"说明本章留下了什么问题。

| 章 | 必须包含的推导 |
| --- | --- |
| P1 | 复合误差随步数增长的界；CEM 更新；MPPI 权重；高斯 NLL |
| P2 | MDN 负对数似然；ES / CMA-ES 更新（直觉级） |
| P3 | 序列 ELBO；两个高斯之间的 KL；RSSM 的先验 / 后验分解 |
| P4 | Bellman 方程；TD 误差；λ-return 递推 |
| P5 | 策略梯度定理；GAE；score-function 与 pathwise 估计器的方差 |
| P6 | Dreamer V3 全部损失（标出每个 stop-gradient）；类别分布的 KL；symlog / two-hot |
| P7 | PUCT；MuZero K 步展开损失；TD-MPC 目标 |
| P8 | VQ 损失与 straight-through；自回归分解 |
| P9 | 去噪 score matching；EDM 预条件；flow matching 目标 |
| P10 | 坍塌为何是合法最小值；VICReg / SIGReg 项 |
| P11 | 逆动力学；策略评估的相关性指标 |

### 事实与数字

- **实验数字只写真实跑出来的**。讲义还没写的章节，凡是需要实测结果的地方（曲线解读、运行时间、分数）写一行
  `> 📌 待回填：<要填什么，来自哪节讲义>`，不编数字。可以写定性预期（"误差应随步数增大"）。
- **论文里的数字**（超参数、基准分数）：有把握的写出，并注明出自哪篇论文的哪一部分；没把握的不写，或标「（待核对）」。不编造基准分数。
- **论文引用**写成「作者 年份（arXiv:编号）」。各章 `README.md` 的论文表里的 arXiv 编号都已核对过，优先引用表里的论文；新增论文要先核对编号。
- 2025–2026 年的前沿工作变化快。写 P9–P11 前先刷新论文清单，对拿不准的说法标「（待核对）」。

## 章节 README.md 规范

```markdown
# Phase N: 标题

> **学习目标**：一句话
> **预计时长**：N 周（第 a–b 周）
> **前置**：需要先学完哪些章
> **状态**：README ✅ ｜ CONCEPT.md ✅ ｜ 讲义 0/8 ⏳ ｜ 项目 ⏳

## 🧭 在课程中的位置        （承接什么、为后面铺垫什么）
## 📚 本阶段内容            （分组列出讲义文件，每个一句话）
## 🧠 核心概念              （CONCEPT.md 的章节清单）
## 🎯 实战项目
## 📖 论文与资料            （表格：论文、arXiv 编号、必读程度）
## 💻 算力与环境
## ✅ 学习检查
```

## 讲义规范

- 文件名 `NN-english-slug.py`，单文件可直接运行，约 300–600 行。
- 文件头 docstring：学习目标 / 核心概念 / 前置知识 / 对应概念文档（CONCEPT.md 的小节）/ 预计运行时间。
- 正文按函数分节：`introduction()`（原理 + ASCII 图）→ 从零实现 → 实验与可视化 → `exercises()`（练习 + 思考题，附参考答案）→ `main()`。
- 核心代码（模型、规划器、损失）写在讲义文件里，不藏进 `src/utils/`。某个组件在后续阶段第一次被复用时，再把它移进 `src/utils/`。
- 支持 `WM_QUICK=1` 冒烟模式（`from utils import is_quick`）。
- 图和视频存到 `outputs/phase-NN/`（`from utils import output_dir, save_figure, save_video`）。
- 讲义里打印的结论必须和它实际跑出来的结果一致。写完要完整跑一遍，看输出再定稿。
- 参考样例：`src/phase-01-dynamics-and-planning/01-what-is-a-world-model.py`。

## 全课程符号约定

各章 `CONCEPT.md` 的符号表在此基础上增补；和论文记号不同的地方，在该章符号表里写明对应关系。

| 符号 | 含义 |
| --- | --- |
| $t$，$H$，$T$ | 时间步；规划 / 想象的视野；回合或序列长度 |
| $s_t$ | 环境的真实状态（低维，如关节角度与速度） |
| $o_t$ | 观测（如图像）。状态完全可观测时 $o_t = s_t$ |
| $a_t$ | 动作 |
| $r_t$ | 在 $s_t$ 执行 $a_t$ 得到的奖励（Sutton & Barto 记作 $R_{t+1}$） |
| $\gamma$，$G_t$ | 折扣因子；回报 $G_t = \sum_{k \ge 0} \gamma^k r_{t+k}$ |
| $\tau$ | 轨迹 $(s_0, a_0, r_0, s_1, \dots)$ |
| $p(s_{t+1} \mid s_t, a_t)$ | 真实环境的转移分布 |
| $f_\theta$，$p_\theta$ | 学到的世界模型（确定性 / 概率形式），参数 $\theta$ |
| $z_t$，$h_t$ | 随机隐变量；确定性循环状态（P2 起） |
| $q_\theta(z_t \mid \cdot)$ | 推断模型（后验 / 编码器） |
| $\pi_\phi(a \mid s)$ | 策略，参数 $\phi$ |
| $V_\psi(s)$，$Q_\psi(s, a)$ | 状态价值、动作价值，参数 $\psi$ |
| $D_{\mathrm{KL}}(q \,\Vert\, p)$ | KL 散度 |
| $\mathrm{sg}(\cdot)$ | 停止梯度（stop-gradient） |
| $\mathcal{D}$ | 数据集 / 经验回放 |
| `[B, T, D]` | 张量形状：批大小、时间长度、特征维度 |

## 环境与依赖

- **Python 3.12**，用 `uv` 管理。不要换成系统默认的 3.14：Box2D、pygame、Meta-World 在 3.14 上没有可用的安装包。
- **MuJoCo 锁在 3.x**（`mujoco>=3.3,<4`）。4.0 在 2026-11 发布，包结构变了，机器人生态（Meta-World、ALOHA 仿真、LIBERO）还依赖 3.x。
- **设备**：`utils.get_device()` 按 cuda → mps → cpu 选择。本机是 M1 Max 32GB，没有 NVIDIA GPU；MPS 不支持 float64，张量一律用 float32。重训练在 AutoDL 的 4090 上跑。
- **依赖按阶段添加**：`pyproject.toml` 里只放已经写了讲义的阶段要用的包。
- **已知冲突**：`gym-pusht` 依赖 `pygame`，Gymnasium 的 classic-control / box2d 扩展依赖 `pygame-ce`，两者不能装在同一个环境。P8 起的 PushT 项目要用独立环境。
- 已核对的环境名：`Pendulum-v1`、`Reacher-v5`、`CarRacing-v3`、`HalfCheetah-v5`、`Hopper-v5`。

## 不要做的事

- 不要自动 `git commit` 或 `git push`，除非用户明确要求。
- 不要在 `CONCEPT.md` 或讲义里写没有跑过的实验数字。
- 不要为了"将来可能复用"提前往 `src/utils/` 里加抽象。
- 不要改动 `llm-course`、`deep-learning-course` 两个旧课仓库，它们只作为格式参考。
