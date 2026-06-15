# PriFold-SymFlow: RNA 二级结构预测实验

本仓库 fork 自 [BEAM-Labs/PriFold](https://github.com/BEAM-Labs/PriFold)，在其基础上开展 RNA 二级结构预测的改进实验。

## 项目概述

**PriFold** 是一个基于深度学习的 RNA 二级结构预测方法，输入 RNA 序列，输出 L×L contact map。原始论文通过引入生物先验（配对概率注意力 + 协变异数据增强）在 bpRNA、RNAStrAlign、ArchiveII 等数据集上达到 SOTA。

本仓库在 PriFold 之上，探索**轻量高效的判别式/生成式架构**来缩小与 baseline 的差距，实验代码位于 `symfold/` 目录。

## 实验线：PriFold-SymFlow / DensityNet

经历了从生成式到判别式的完整演进：

```
v1-v3: 生成式 Flow Matching 初探（F1 ~0.40）
v4:    + ControlInject + Direct Head + Density Budget（F1=0.49）
v5:    + Dice/Ratio Penalty + 大模型 26M（F1=0.62）
v6:    + 模块化 Loss + 消融框架（F1=0.61）
v7:    ★ 转向纯判别式 DensityNet 3.56M（F1=0.654）✅
```

### 当前最优：v7 DensityNet

| 版本 | 架构 | 参数量 | Test F1 | vs Baseline |
|------|------|--------|---------|-------------|
| v7 (当前) | Axial Transformer | 3.56M | **0.6538** | -15% |
| v6 | Discrete Flow | 26M | 0.6083 | -21% |
| v5 | Discrete Flow | 26M | 0.6188 | -20% |
| **PriFold baseline** | RNAformer | — | **0.7700** | — |

v7 核心设计：
- **MARS-LX**（160M，冻结）作为 RNA 语言模型 encoder
- **Axial Transformer**（8 层，hidden=160，4 heads）—— 仅 3.56M 可训练参数
- **Density-Stratified Tversky Loss** —— 对低密度样本偏向召回
- **BF16 混合精度** —— 单次前向传播，推理高效

## 目录结构

```
PriFold/
├── train.py / inference.py       # 官方 PriFold 主线
├── prifold/                      # MARS/LLaMA2 语言模型代码
├── utils/                        # 主线工具 + RNAformer
├── symfold/                      # ★ 实验主目录
│   ├── train_v7.py               # v7 训练入口（DensityNet）
│   ├── v7/                       # v7 模型代码
│   ├── v6/ v5/ v4/              # 历史版本模型
│   ├── config/                   # 所有配置（含消融）
│   ├── data.py                   # 数据加载
│   ├── metrics.py                # 评估指标
│   └── outputs/                  # 训练输出与曲线
└── docs/                         # 实验文档与分析报告
```

## 环境

```bash
conda create -n RNADiffFold_torch260 python=3.10
conda activate RNADiffFold_torch260
pip install -r requirements.txt
# PyTorch 2.6.0+cu124, GPU: NVIDIA H20 97GB
```

## 快速开始

### 训练 v7

```bash
bash symfold/train/run_train.sh symfold/config/v7/v7_full.json
```

### 运行消融实验

```bash
bash symfold/train/run_train.sh symfold/config/v7/ablations/v7_no_dst.json
```

### 运行官方 PriFold baseline

```bash
./train.sh      # 训练
./inference.sh  # 推理
```

## 数据集

使用 bpRNA、RNAStrAlign、ArchiveII 数据集，详见 [HuggingFace](https://huggingface.co/yfish/PriFold)。

## 原始论文

```bibtex
@inproceedings{yang2025prifold,
  title={PriFold: Biological Priors Improve RNA Secondary Structure Predictions},
  author={Yang, Chenchen and Wu, Hao and Shen, Tao and Zou, Kai and Sun, Siqi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={1},
  pages={950--958},
  year={2025}
}
```

## License

MIT
