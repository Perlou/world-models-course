# Phase 2: World Models (2018)——看、记、做

> **学习目标**：复现 Ha & Schmidhuber 的 World Models：从像素学隐空间，在隐空间里预测未来，在"梦"里训练控制器
> **预计时长**：2 周（第 3–4 周）
> **前置**：Phase 1；`deep-learning-course` 的 VAE（ELBO、重参数化）和 LSTM
> **状态**：README ✅ ｜ CONCEPT.md ✅ ｜ 讲义 8/8 ✅ ｜ 项目 ✅

## 🧭 在课程中的位置

Phase 1 的世界模型直接拿到了状态（角度、角速度）。真实的机器人只有摄像头：
输入是几千维的像素，而且单帧画面里看不出速度。

本章用 2018 年那篇让"世界模型"这个词流行起来的论文，解决"从像素出发"的问题。
它把智能体拆成三块：

- **V（视觉）**：VAE 把每一帧压成一个小向量 $z$
- **M（记忆）**：MDN-RNN 在隐空间里预测下一个 $z$
- **C（控制器）**：只有几百个参数的线性策略，用进化算法训练

这套方法三块是分开训练的，简单、好懂，也暴露了问题（V 不知道什么信息对任务有用、
M 只是被动预测）。这些问题引出 Phase 3 的 RSSM。
本章的"在梦里训练"也是 Phase 6 Dreamer 的雏形。

## 📚 本阶段内容

### 1. V：视觉模型

- `01-from-states-to-pixels.py` - 从状态到像素：`CarRacing-v3` 的观测、预处理、数据采集
- `02-conv-vae-vision.py` - 卷积 VAE：把 64×64 的画面压成隐向量 $z$，隐空间插值

### 2. M：记忆模型

- `03-mixture-density-network.py` - 混合密度网络（MDN）：为什么要预测分布，MDN 的损失
- `04-mdn-rnn-memory.py` - MDN-RNN：LSTM + MDN 预测下一个 $z$
- `05-dreaming-in-latent-space.py` - 在隐空间里自回归展开，把"梦"解码成视频，温度参数

### 3. C：控制器

- `06-evolution-strategies-cma-es.py` - 进化策略与 CMA-ES：不用梯度的黑盒优化
- `07-controller-in-real-env.py` - 线性控制器在真实环境里训练，对比有无 M 的隐状态
- `08-training-in-the-dream.py` - 完全在梦里训练控制器，再放回真实环境；模型被钻空子的现象

## 🧠 核心概念

详见 [CONCEPT.md](./CONCEPT.md)《World Models (2018)：看、记、做》：

1. 从状态到像素：为什么需要学表示
2. V：用卷积 VAE 把画面压成隐向量
3. 混合密度网络（MDN）：预测一个分布而不是一个值
4. M：MDN-RNN 记忆模型
5. 在隐空间里做梦
6. 进化策略与 CMA-ES：不用梯度的优化
7. C：极简控制器，在真实环境中训练
8. 在梦里训练：把世界模型当环境
9. 这套方法的局限：为什么需要 RSSM

## 🎯 实战项目

**CarRacing World Models**（`projects/phase-02-carracing-world-models/`）：完整复现 V-M-C 管线——
采集数据、训练 VAE、训练 MDN-RNN、用 CMA-ES 训练控制器，输出"真实画面 / VAE 重建 / 梦境"三联对比视频。

## 📖 论文与资料

| 论文 | 编号 | 必读 | 说明 |
| --- | --- | --- | --- |
| Ha & Schmidhuber 2018, World Models | arXiv:1803.10122 | ⭐⭐⭐⭐⭐ | 本章主线。交互版 worldmodels.github.io 更直观 |
| Ha & Schmidhuber 2018, Recurrent World Models Facilitate Policy Evolution | arXiv:1809.01999 | ⭐⭐⭐⭐ | NeurIPS 版本，实验细节更全 |
| Graves 2013, Generating Sequences With RNNs | arXiv:1308.0850 | ⭐⭐⭐ | MDN-RNN 的出处（手写生成） |
| Hansen 2016, The CMA Evolution Strategy: A Tutorial | arXiv:1604.00772 | ⭐⭐ | 读前几节，建立直觉即可 |
| Kingma & Welling 2013, VAE | arXiv:1312.6114 | ⭐⭐ | 旧课学过，按需回看 |
| Bishop 1994, Mixture Density Networks | 技术报告 | ⭐⭐ | MDN 的原始出处 |

## 💻 算力与环境

- 环境：`CarRacing-v3`，需要给 Gymnasium 加 `box2d` 扩展（写本章讲义时加入依赖）。
- 本机可完成。预计数据采集和 VAE / MDN-RNN 训练各需几十分钟，CMA-ES 在真实环境中训练需要数小时（未实测的估算）。

## ✅ 学习检查

- [ ] 能解释为什么从像素出发时需要先学表示，单帧为什么不够
- [ ] 能写出 VAE 的损失，并说明这里为什么用 VAE 而不是普通自编码器
- [ ] 能推导 MDN 的负对数似然，并解释均方误差在多峰情况下为什么失败
- [ ] 能说出 CMA-ES 每一代更新了哪三样东西
- [ ] 能解释控制器为什么故意做得很小
- [ ] 能解释"在梦里训练"时温度参数的作用，以及模型被钻空子是怎么回事
