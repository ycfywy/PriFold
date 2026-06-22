# v10: v9 + MARS Unfreeze

## 概述

v10 和 v9 使用**完全相同的模型代码**（`symfold/v9/model.py` 的 `DensityNetProPlus`）。

唯一区别：**MARS 不再冻结**，允许 160M 预训练语言模型参数随训练更新。

## v9 vs v10

| | v9 | v10 |
|---|---|---|
| 模型代码 | `symfold/v9/model.py` | `symfold/v9/model.py`（同一份） |
| `freeze_mars` | `true` | **`false`** |
| 可训参数 | 5.09M | **165.7M** |
| MARS LR | — | 5e-6（分层，远低于 head） |
| Head LR | 5e-4 | 5e-4 |
| 初始化 | 从头训 | 从 v9 best.pt warm-start |

## 核心思路

v9 的分析表明：冻结 MARS 导致表示无法适配 RFAM 长尾结构，F1=0 的样本中模型把高分给了错误位置。解冻 MARS 让语言模型表征能够针对结构预测任务做微调。

## 文件

```
symfold/v10/
└── README.md          # 本文档

symfold/train/
└── train_v10.py       # 单卡训练脚本（分层 LR + warm-start）

symfold/config/v10/
└── v10_ddp.json       # 配置（freeze_mars=false）
```

## 训练

```bash
CUDA_VISIBLE_DEVICES=0 python symfold/train/train_v10.py symfold/config/v10/v10_ddp.json
```

## 输出

```
symfold/outputs/v10_ddp/
├── model/best.pt
├── model/last.pt
├── history.json
└── training_curves.png
```
