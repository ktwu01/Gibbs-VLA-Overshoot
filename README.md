# Gibbs-VLA-Overshoot

> **Hypothesis:** The "jitter" in Vision-Language-Action (VLA) models is not mere noise, but a fundamental mathematical artifact—the **Gibbs Phenomenon**—arising from the mapping of discrete semantic tokens onto continuous physical manifolds.

## ⭐ Overview

Current VLA architectures (e.g., RT-2, OpenVLA) use discrete text tokens to condition continuous robotic actions. In signal processing, approximating a step function (the semantic jump) with a finite sum of continuous basis functions (the neural network) leads to a predictable ~8.9% overshoot.

This repository provides the mathematical framework and empirical evidence to quantify this "Semantic-to-Physical Impedance Mismatch."

## Core Research Questions

1. Does the "overshoot" in robot end-effector trajectories correlate with the theoretical Gibbs constant ($G \approx 0.0895$)?
2. Can we observe "Ringing Artifacts" in the frequency domain at task-switching boundaries?
3. Does a JEPA-style (Joint-Embedding Predictive Architecture) approach mitigate this overshoot by removing the discrete discontinuity?

## Framework

- **Domain:** Signal Processing meets Embodied AI.
- **Data:** Analysis performed on `Language-Table` and `CALVIN`.
- **Metrics:** Spectral leakage, L2-norm of the 3rd-order derivative (Jerk), and Overshoot Ratio.

## Repository layout

| Path | Role |
|------|------|
| `data/` | Symlinks to CALVIN or Language-Table datasets (see `data/README.md`) |
| `models/` | Simplified VLA wrapper for trajectory extraction |
| `analysis/` | FFT, wavelets, Gibbs-ratio metrics |
| `experiments/` | Jupyter notebooks for visualization |
| `paper/` | Manuscript outline and LaTeX source |

## Experiment Design

既然你已经搭好了工程目录，而且拥有极强的数学物理底底气，我们直接进入**实验设计（Experiment Design）**阶段。

目前的任务是把“直觉”转化为“观测”。我们需要验证的是：**当指令（离散信号）发生跳变时，VLA 输出的动作（连续信号）是否存在理论上的 Gibbs 震荡。**

以下行动路线图直接对应当前仓库中的 `analysis/` 和 `experiments/`：

### 第一步：数据采集与预处理 (`analysis/`)

你需要先找到一个“纯净”的阶跃信号。在具身数据中，这通常对应于**任务切换点**。

1. **目标数据集：** 建议先用 **Language-Table**。它的动作空间是二维的 $(x, y)$，信号结构更干净。
2. **提取轨迹：** 编写 `analysis/extract_trajectories.py`，专门提取在 `instruction` 发生改变前后 $N$ 个帧的轨迹片段。
   - 例如：指令从 `"push red block to top"` 变为 `"push green block to left"`。
3. **时间对齐：** 将不同轨迹的时间轴 $t$ 以指令切换点 $t_0$ 为原点对齐。

### 第二步：数学建模与谱分析 (`analysis/`)

这是核心分析部分，建议在 `analysis/spectral_engine.py` 中实现。

1. **时域分析：** 计算动作序列 $a(t)$ 的一阶导（速度）和二阶导（加速度）。
   - 核验点：观察 $t_0$ 附近的加速度是否出现明显、且无法归因于物理惯性的“脉冲”或“反向补偿”。
2. **频域分析（FFT / 小波变换）：**
   - 对比“平稳期”和“切换期”的频谱。
   - **Gibbs 指纹：** 寻找频率分布中的 $1/f$ 衰减特征。如果模型在试图用连续基函数拟合阶跃，高频部分通常会表现出特定幂律分布。
3. **计算 Overshoot ($P_{ov}$)：**

   $$
   P_{ov} = \frac{\max|a(t)| - a_{steady}}{a_{steady} - a_{initial}}
   $$

   检查这个值是否在不同任务间表现出统计意义上的 **~8.9%** 聚类。

### 第三步：对照组实验 (`experiments/`)

为了证明这更像是 VLA 架构问题，而不是数据本身的惯性或噪声，需要做对照组。

1. **Dataset Baseline：** 测量人类演示数据（Ground Truth）在切换点时的 $P_{ov}$。人类动作通常更接近过阻尼（critically damped），理论上不会出现标准 Gibbs 震荡。
2. **Model Inference：** 在 `models/` 下加载一个开源 VLA，例如 **OpenVLA** 或 **RT-1**。
3. **人工合成指令阶跃：** 在 `experiments/synthetic_step.py` 中，给模型提供一个物理世界保持稳定的视觉背景，但在 embedding 层强行施加一个瞬时的指令向量跳变。
   - 观察模型在没有物理阻力干扰的“脑补”状态下，输出的第一个动作包是否自带接近 **8.9%** 的过冲。

## Hello World

你的首要任务是在 `analysis/` 中写一个简单脚本，读入一组轨迹数据，并绘出其**速度（Velocity）的分布图**。

> **专家建议：** 重点观察那些“指令改变但视觉反馈尚未完全变化”的瞬间（约 100-200ms）。这对应了那个失聪失明的“哲学家”刚接到新命令、正急于向“运动员”发号施令的时刻，也是 Gibbs 现象最纯粹的爆发窗口。

## Suggested Starting Point

你可以从以下两条路径任选其一开始：

1. 从读取 **Language-Table** 的 RLDS 数据格式开始，先把切换点附近的轨迹提取出来。
2. 直接针对手头已有的时间序列数据做一次 FFT / 速度分布验证，快速确认分析链路是否闭环。

## License

Apache License 2.0 — see `LICENSE`.
