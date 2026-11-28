# Phase 2 实战项目：CarRacing World Models（V + M + C）

本目录是 Phase 2 的实战项目，将 Ha & Schmidhuber (2018) *World Models* 的三阶段管线——视觉模型 **V**（卷积 VAE）、记忆模型 **M**（MDN-RNN）与线性控制器 **C**（CMA-ES）——组装为一个模块化、可测试、可一键复现的完整项目，并输出「真实画面 / VAE 重建 / MDN-RNN 开环梦境」三联对比视频。

## 文件结构

```
projects/phase-02-carracing-world-models/
├── README.md                  # 本说明与实测结果
├── conftest.py                # pytest 导入路径配置
├── world_models_pipeline.py   # V/M/C 封装、开环梦境与三联对比视频生成
├── run_world_models.py        # 完整评估、画图与视频导出主脚本
└── tests/
    └── test_world_models.py   # 6 个单元测试（VAE 编解码、MDN-RNN 单步、控制器映射、梦境形状）
```

## 运行方式

```bash
# 单元测试
MPLCONFIGDIR=/tmp/mpl uv run pytest projects/phase-02-carracing-world-models -q

# 冒烟运行（快速验证全流程）
MPLCONFIGDIR=/tmp/mpl WM_QUICK=1 uv run python projects/phase-02-carracing-world-models/run_world_models.py

# 完整评估与三联对比视频生成（复用 outputs/phase-02/ 训好的 V/M/C 权重）
MPLCONFIGDIR=/tmp/mpl uv run python projects/phase-02-carracing-world-models/run_world_models.py
```

## 实测结果

在 12 条与训练完全隔离的全新 CarRacing-v3 完整赛道（950 步，种子 `3000..3011`）上实测（`outputs/phase-02-project/results.json`，训练预算：种群 16 × 25 代 × 2 回合）：

| 控制器配置 | 参数量 | 12 条全新完整赛道回报（均值 ± 标准差） | 最低分 | 最高分 |
| --- | ---: | ---: | ---: | ---: |
| 随机策略（逐帧独立 / 时间相关） | — | `-43.0` / `-90.9` | — | — |
| 只有 $z_t$（V + C） | 99 | **`508.9 ± 213.3`** | `146.5` | `905.0` |
| $[z_t; h_t]$（V + M + C） | 867 | **`433.3 ± 238.4`** | **`237.3`** | **`911.2`** |

三联对比视频（赛道种子 `3007`，前 240 步回报 `183.1`）：

| 画面对比项 | 实测像素 MSE | 说明 |
| --- | ---: | --- |
| 真实画面 vs VAE 单帧重建 | `0.00110` | 64×64 赛道边缘、车身与仪表盘清晰保留 |
| 真实画面 vs MDN-RNN 开环梦境 | `0.01017` | 随步数累积出现复合误差，约为单帧重建误差的 `9.3×` |

## 关键发现

1. **小参数控制器在真实环境中可被 CMA-ES 有效优化**：两种线性控制器均在 25 代内将回报从 `-43` 的随机水平提升至 `430~510` 区间，并在单条赛道上分别达到 `905.0` 与 `911.2` 分（超过论文 900 分"解决"线）。
2. **维数与预算的权衡**：`[z_t; h_t]` 的参数量（867）几乎是只用 `z_t`（99）的 9 倍。在 25 代 × 16 候选的小预算下，867 维策略在第 25 代的在线评估分（`456.3`）才刚反超 99 维策略（`434.8`），因此 12 赛道均值略低，但其最差赛道下限（`237.3` vs `146.5`）与最高单赛道得分（`911.2`）均显示出时间记忆 `h_t` 的价值。
3. **三联视频直观呈现表征与动力学误差的分离**：单帧重建 MSE 仅 `0.00110`，而自回归展开的开环梦境 MSE 放大至 `0.01017`，直观揭示了 Phase 2 独立训练 V 与 M 的局限，自然引出 Phase 3 的端到端 RSSM。

## 生成产物

- `outputs/phase-02-project/results.json`：12 条赛道的完整逐赛道得分与 MSE 指标
- `outputs/phase-02-project/learning-curve.png`：12 条新赛道的回报分布对比图
- `outputs/phase-02-project/real-recon-dream.mp4`：「真实 / VAE 重建 / MDN-RNN 梦境」三联对比视频
