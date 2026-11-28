# Phase 3 实战项目：PlaNet 像素控制（Pendulum-v1，64×64 RGB）

本目录是 Phase 3 的实战项目，完整封装了 Hafner et al. (2019) *PlaNet* 的端到端管线：仅从 64×64 RGB 渲染画面（不读取角度、角速度或奖励解析公式）学习循环状态空间模型（RSSM），并在隐空间中用交叉熵方法（CEM）做滚动时域规划（MPC）。

## 文件结构

```
projects/phase-03-planet-pixel-control/
├── README.md                  # 本说明与实测结果
├── conftest.py                # pytest 导入路径配置
├── pixel_env.py               # 像素版 Pendulum-v1 包装与序列回放缓冲区
├── rssm_model.py              # 卷积编码器/解码器、RSSM（observe/imagine）与序列 ELBO
├── latent_planner.py          # 隐空间 CEM 规划与闭环回合采集
├── run_planet.py              # 完整训练、评估、视频与学习曲线生成脚本
└── tests/
    └── test_planet.py         # 单元测试（RSSM 形状、序列 ELBO、隐空间规划与回放采样）
```

## 运行方式

```bash
# 单元测试
MPLCONFIGDIR=/tmp/mpl uv run pytest projects/phase-03-planet-pixel-control -q

# 冒烟运行（快速验证全流程）
MPLCONFIGDIR=/tmp/mpl WM_QUICK=1 uv run python projects/phase-03-planet-pixel-control/run_planet.py

# 完整运行（默认 5 初始随机回合 + 10 在线规划回合，CPU 约 48 分钟）
MPLCONFIGDIR=/tmp/mpl uv run python projects/phase-03-planet-pixel-control/run_planet.py
```

## 实测结果

在单机 CPU（`batch_size=12`, `chunk_len=25`, `R=2`, 初始 5 随机回合 + 10 轮在线规划，共 3,000 环境步、690 次梯度更新，规划预算 `H=15, N=300, I=6`）上实测（`outputs/phase-03-project/results.json`，总耗时 `2865.0s`，其中模型训练 `1790.7s` 占 `62.5%`，CEM 规划 `1060.3s` 占 `37.0%`）：

| 训练轮次 | 累计环境步 | 无噪评估回报（5 回合均值 ± 标准差） | 15 步开环奖励 RMSE | 图像重建损失 | 原始 KL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 参照：随机策略 | — | `-1232.3` | — | — | — |
| 第 1/10 轮 | 1,200 | `-1233.1 ± 275.7` | `9.50` | `20.1` | `0.5` |
| 第 3/10 轮 | 1,600 | `-1151.6 ± 416.8` | `13.14` | `15.5` | `2.8` |
| 第 6/10 轮 | 2,200 | **`-1133.1 ± 350.8`** | **`0.66`** | `10.7` | `2.9` |
| 第 9/10 轮 | 2,800 | `-1191.2 ± 308.4` | `3.22` | `7.2` | `2.8` |
| 第 10/10 轮 | 3,000 | `-1246.3 ± 289.3` | `2.83` | `7.5` | `3.0` |

## 关键发现

1. **端到端 RSSM 视觉与奖励建模收敛迅速**：在 3,000 环境步（仅 690 次更新）内，图像重建损失从 `20.1` 稳定下降至 `7.2–7.5`，原始 KL 稳定维持在 `2.8–3.0`（避免后验坍塌）；短期开环预测精度良好（最终模型前 6 步开环奖励 RMSE 仅 `0.80–1.28`，1 步像素 MSE 仅 `0.00136`）。
2. **CPU 小预算下的规划视野与训练预算瓶颈**：第 6 轮在 15 步奖励预测最准（RMSE `0.66`）时达到最佳评估回报 `-1133.1`（高出随机基线约 `+99` 分，与讲义 `08-planet-agent.py` 在 6,000 环境步达到的 `-1115.4` 一致）；但随着展开步数超过 ~10 步，开环奖励 RMSE 迅速增大，且 CPU 小预算（3,000 步 / 690 更新）远小于原论文（数十万环境步、每轮 100 次更新、GPU 上 `N=1000, I=10`），不足以将摆完全甩起至状态空间 PETS 的 `-140` 区间。
3. **耗时结构凸显 PlaNet 的核心短板**：即使缩小到 `N=300, I=6`，隐空间 CEM 规划仍占去全流程 `37.0%`（`1060s`）的算力，每步都要重新搜索且受限于短视野 `H=15`——这正是 Phase 6 Dreamer 引入「在想象中训练摊销 Actor-Critic」的直接动机。

## 生成产物

- `outputs/phase-03-project/results.json`：完整训练曲线、耗时拆解与 90 步开环像素/奖励误差数组
- `outputs/phase-03-project/learning-curve.png`：学习曲线与开环奖励预测 RMSE 随步数变化图
- `outputs/phase-03-project/open-loop-prediction.mp4`：「真实画面 / 采样想象 / 均值想象」开环预测对比视频
- `outputs/phase-03-project/control-progress.mp4`：第 1 轮与最终轮闭环控制过程对比视频
