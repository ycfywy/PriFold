# PriFold-SymFlow v4 — Hands-on 自包含版本

> 本目录是 SymFlow v4 的**完整自包含副本**，所有代码集中于此，无需跳转到 `v3/`、`v4/`、`prifold/` 等目录即可理解完整模型逻辑。

## 文件结构

```text
handson/
├── README.md               ← 本文件，整体说明
├── ARCHITECTURE.md         ← 架构详解（数据流、模块交互、设计决策）
├── config.json             ← 训练配置（RNAStrAlign 示例）
├── train.py                ← 训练脚本（一键运行）
├── eval.py                 ← 评估脚本
├── model.py                ← 模型 wrapper（MARS 特征提取 + 训练 forward + 采样）
├── backbone.py             ← DiT 主干（DA-SE-DiT-MARS v4，含所有子模块）
├── discrete_flow.py        ← Bernoulli DFM（loss、采样率、投影）
├── data.py                 ← 数据加载（Dataset、Bucket Sampler、Collate）
├── metrics.py              ← P/R/F1/MCC 计算
└── mars_forward.py         ← MARS 语言模型 forward（attention 提取）
```

## 快速开始

```bash
# 1. 激活环境
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
source activate RNADiffFold_torch260
export PYTHONPATH=/root/aigame/dannyyan/PriFold

# 2. 训练（单 GPU）
python symfold/handson/train.py symfold/handson/config.json

# 3. 评估
python symfold/handson/eval.py \
  --ckpt symfold/outputs/v4_rnastralign/model/best.pt \
  --config symfold/handson/config.json
```

## 代码阅读顺序（推荐）

1. **`config.json`** — 理解所有超参数
2. **`data.py`** — 数据如何从 RNA 序列变成 tensor batch
3. **`mars_forward.py`** — MARS 语言模型如何提取 hidden + attention
4. **`backbone.py`** — DiT 主干的完整实现（从 patch 嵌入到 logit 输出）
5. **`discrete_flow.py`** — Bernoulli DFM 的 loss 和 CTMC 采样
6. **`model.py`** — 如何把上面的部件串起来（训练 + 推理）
7. **`train.py`** — 训练循环

## 与原始代码的对应关系

| handson 文件 | 原始文件 |
|---|---|
| `backbone.py` | `symfold/v4/da_se_dit.py` + `symfold/v3/da_se_dit.py`（基础模块） |
| `discrete_flow.py` | `symfold/v4/discrete_flow.py` + `symfold/v3/discrete_flow.py`（基础函数） |
| `model.py` | `symfold/v4/model.py` |
| `data.py` | `symfold/data.py` |
| `metrics.py` | `symfold/metrics.py` |
| `mars_forward.py` | `prifold/llama2_with_attn.py` |
| `train.py` | `symfold/train_v3.py` + `symfold/train_v4.py` |

## 修改指南

- 想改模型结构 → 编辑 `backbone.py`
- 想改 loss 函数 → 编辑 `discrete_flow.py`
- 想改采样/投影策略 → 编辑 `model.py` 的 `sample()` 方法
- 想改数据增强/预处理 → 编辑 `data.py`
- 想改训练策略（LR、early stop 等）→ 编辑 `train.py` 或 `config.json`
