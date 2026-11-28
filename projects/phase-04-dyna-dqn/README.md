# Phase 4 实战项目：Dyna-DQN

> 对应章节：[Phase 4 强化学习核心（上）](../../src/phase-04-rl-value-based/README.md) ｜
> 概念文档：[CONCEPT.md](../../src/phase-04-rl-value-based/CONCEPT.md) 第 8–10 节 ｜
> 运行时间：完整约 4.4 分钟（M1 Max CPU，3 种子 × 3 组 $n_p$），冒烟约 15 秒

## 📋 项目概述

将讲义 08 的表格 Dyna-Q 架构迁移到连续状态任务 `CartPole-v1` 上：

- **价值函数**：Double DQN（在线网络选动作、目标网络评估、Huber 损失）
- **世界模型**：MLP 动力学 + 奖励 + 终止模型 `(s, a) → (Δs, r, done_logit)`，在真实经验回放上训练
- **后台规划（background planning）**：每执行 1 步真实交互，从真实回放采样起点状态并按当前策略用模型展开 1 步，将生成的模拟转移写入模型回放，混合真实与模拟数据做 $n_p$ 次额外 Double DQN 更新

在统一的 6000 真实交互步预算下，直接对比纯无模型 Double DQN（$n_p = 0$）与 Dyna-DQN（$n_p = 2, 5$）的样本效率与模型偏差权衡。

## 🏗️ 结构

```
projects/phase-04-dyna-dqn/
├── conftest.py          pytest 路径配置
├── dyna_dqn.py          WorldModelMLP、DynaReplay、DoubleDQN 与 Dyna-DQN 训练循环
├── run_dyna_dqn.py      主实验脚本：多规划比 n_p ∈ {0, 2, 5} × 3 种子对比、画图、落盘 results.json
└── tests/
    └── test_dyna_dqn.py 4 项单元测试（模型前向与拟合、缓冲区覆盖、一步模拟更新、评估流程）
```

## 🚀 运行方式

在仓库根目录：

```bash
uv run python projects/phase-04-dyna-dqn/run_dyna_dqn.py              # 完整运行（6000 步 × 3 种子）
WM_QUICK=1 uv run python projects/phase-04-dyna-dqn/run_dyna_dqn.py   # 冒烟运行
uv run pytest projects/phase-04-dyna-dqn                              # 单元测试（4 passed）
```

## 📁 产物

位于 `outputs/phase-04-project/`（不入库）：

| 文件 | 说明 |
| --- | --- |
| `learning-curve.png` | $n_p \in \{0, 2, 5\}$ 在 6000 真实步内的评估回报曲线（均值 ± 标准差阴影） |
| `results.json` | 每个评估检查点（400…6000 步）的均值、标准差、AUC 与耗时 |

## 📈 实测结果

以下结果来自本机完整运行（M1 Max CPU，总步数 6000，3 种子，总耗时 265.4 秒）：

| 配置 | 400 步回报 | 800 步回报 | 全程平均分（AUC） | 6000 步终点回报 | 用时 |
| --- | --- | --- | --- | --- | --- |
| 纯 Double DQN（$n_p = 0$） | 12.2 ± 2.2 | 154.8 ± 34.1 | 217.1 | 228.1 ± 192.3 | 20.4s |
| **Dyna-DQN（$n_p = 2$）** | 59.4 ± 20.2 | 184.6 ± 4.0 | **251.1** | **358.4 ± 175.2** | 84.3s |
| Dyna-DQN（$n_p = 5$） | **139.2 ± 48.8** | **213.1 ± 91.6** | 211.4 | 287.5 ± 151.1 | 160.4s |

**怎么读这张表：**

1. **极低数据区间的样本效率提升非常明显**：仅 400 步真实交互后，$n_p=2$ 回报达纯 Double DQN 的 4.9 倍（59.4 vs 12.2），$n_p=5$ 达 11.4 倍（139.2 vs 12.2）；800 步时排序仍严格单调（154.8 < 184.6 < 213.1）。
2. **适度规划比（$n_p=2$）全程综合最优**：AUC（251.1）与终点回报（358.4，较纯 Double DQN 高出 +130.3）均居第一。
3. **过大规划比（$n_p=5$）暴露模型偏差**：虽然起步最快，但在 1200–2400 步中段一度回落至 84.1–123.8（低于 $n_p=0$），说明当单个确定性 MLP 动力学模型尚未完全覆盖新访问状态时，过度放大模拟更新会让 Q 网络拟合模型误差；后段随真实数据增加才回升至 287.5。这正是 Phase 5 MBPO 引入**概率集成 + 短分支展开 + 模型数据配比控制**的直接动机。

## ✅ 测试

```bash
uv run pytest projects/phase-04-dyna-dqn -q    # 4 passed
```
