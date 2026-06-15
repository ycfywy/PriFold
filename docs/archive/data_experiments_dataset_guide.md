# PriFold 实验数据集说明文档

## 概述

PriFold 项目是一个 **RNA 二级结构预测** 系统，输入 RNA 序列，输出 L×L 的 contact map（碱基配对矩阵）。项目开展了两个核心实验，分别使用不同的数据集进行训练和测试。

---

## 实验一：bpRNA 数据集实验

### 实验说明

这是项目的**主要实验**，涵盖了 PriFold 主线 baseline（RNAformer 架构）以及 SymFlow 实验线的 v4~v7 全部版本。

### 训练集

| 项目 | 信息 |
|------|------|
| **文件路径** | `/root/aigame/dannyyan/PriFold/data/bprna/bpRNA.csv` |
| **Contact Map 目录** | `/root/aigame/dannyyan/PriFold/data/bprna/ct/TR0/{file_name}.npy` |
| **划分标识** | CSV 中 `data_name == 'TR0'` 的记录 |
| **过滤条件** | 序列长度 < 490（`max_len_filter=490`） |
| **样本数量** | **10,807 条**（原始 ~10,817 条，过滤掉约 10 条长度 ≥ 490 的序列） |

### 验证集

| 项目 | 信息 |
|------|------|
| **文件路径** | `/root/aigame/dannyyan/PriFold/data/bprna/bpRNA.csv` |
| **Contact Map 目录** | `/root/aigame/dannyyan/PriFold/data/bprna/ct/VL0/{file_name}.npy` |
| **划分标识** | CSV 中 `data_name == 'VL0'` 的记录 |
| **过滤条件** | 序列长度 < 490 |
| **样本数量** | **1,299 条** |

### 测试集

| 项目 | 信息 |
|------|------|
| **文件路径** | `/root/aigame/dannyyan/PriFold/data/bprna/bpRNA.csv` |
| **Contact Map 目录** | `/root/aigame/dannyyan/PriFold/data/bprna/ct/TS0/{file_name}.npy` |
| **划分标识** | CSV 中 `data_name == 'TS0'` 的记录 |
| **过滤条件** | 序列长度 < 490 |
| **样本数量** | **1,303 条** |

### 数据统计汇总

| 划分 | 样本数 | 占比 | 平均序列长度 | 中位长度 | 最短 | 最长 |
|------|--------|------|------------|---------|------|------|
| Train (TR0) | 10,807 | 80.6% | 133.5 | 105 | 33 | 487 |
| Val (VL0) | 1,299 | 9.7% | 131.6 | 106 | 33 | 454 |
| Test (TS0) | 1,303 | 9.7% | 135.5 | 108 | 22 | 481 |
| **合计** | **13,409** | 100% | — | — | — | — |

### 实验结果

| 模型版本 | 架构 | Test F1 |
|---------|------|---------|
| PriFold baseline | RNAformer (判别式) | **0.7700** |
| v7_full | Axial Transformer (判别式) | 0.6538 |
| v6_full | DA-SE-DiT + Discrete Flow (生成式) | 0.6083 |
| v5_bprna | DA-SE-DiT + BernoulliFlow (生成式) | 0.6188 |
| v4_bprna | DA-SE-DiT + Flow (生成式) | 0.4869 |

---

## 实验二：RNAStrAlign 数据集实验

### 实验说明

此实验使用 RNAStrAlign 作为训练集，ArchiveII 作为独立测试集，主要在 v4_rnastralign 和 PriFold baseline 中开展。

### 训练集

| 项目 | 信息 |
|------|------|
| **文件路径** | `/root/aigame/dannyyan/PriFold/data/RNAStrAlign/rnastralign.csv` |
| **Contact Map 目录** | `/root/aigame/dannyyan/PriFold/data/RNAStrAlign/{file_name}.npy` |
| **划分标识** | CSV 中 `data_name == 'tr'` 的记录 |
| **过滤条件** | 序列长度 < 490（`max_len_filter=490`） |
| **样本数量** | **20,234 条**（原始约 20,700 条，过滤掉长度 ≥ 490 的序列） |

### 验证集

| 项目 | 信息 |
|------|------|
| **文件路径** | `/root/aigame/dannyyan/PriFold/data/RNAStrAlign/rnastralign.csv` |
| **划分标识** | CSV 中 `data_name == 'ts'` 的记录（注意：此处用 test split 做验证） |
| **过滤条件** | 序列长度 < 490 |
| **样本数量** | **2,492 条** |

### 测试集

| 项目 | 信息 |
|------|------|
| **文件路径** | `/root/aigame/dannyyan/PriFold/data/archiveII/archiveII.csv` |
| **Contact Map 目录** | `/root/aigame/dannyyan/PriFold/data/archiveII/ct/{file_name}.npy` |
| **划分标识** | 全部数据（ArchiveII 作为独立测试集） |
| **过滤条件** | 序列长度 < 490 |
| **样本数量** | **3,845 条**（原始 3,966 条，过滤掉 121 条长度 ≥ 490 的序列） |

### 数据统计汇总

| 划分 | 数据来源 | 样本数 | 平均序列长度 | 中位长度 | 最短 | 最长 |
|------|---------|--------|------------|---------|------|------|
| Train | RNAStrAlign (tr) | 20,234 | 142.6 | 116 | 30 | 489 |
| Val | RNAStrAlign (ts) | 2,492 | 144.2 | 116 | 30 | 488 |
| Test | ArchiveII (全部) | 3,845 | 192.4 | 120 | 28 | 489 |
| **合计** | — | **26,571** | — | — | — | — |

### 实验结果

| 模型版本 | 架构 | Test F1 (ArchiveII) |
|---------|------|---------------------|
| PriFold baseline | RNAformer (判别式) | **0.9043** |
| v4_rnastralign | DA-SE-DiT + Flow (生成式) | 0.9459* |

> *注：v4_rnastralign 的 test F1 0.9459 是在 RNAStrAlign ts split 上的验证结果，并非 ArchiveII 独立测试集。

---

## 两个实验对比总结

| 对比维度 | 实验一 (bpRNA) | 实验二 (RNAStrAlign) |
|---------|---------------|---------------------|
| **训练集文件** | `data/bprna/bpRNA.csv` (TR0) | `data/RNAStrAlign/rnastralign.csv` (tr) |
| **测试集文件** | `data/bprna/bpRNA.csv` (TS0) | `data/archiveII/archiveII.csv` (全部) |
| **训练集样本数** | **10,807** | **20,234** |
| **测试集样本数** | **1,303** | **3,845** |
| **验证集样本数** | 1,299 | 2,492 |
| **过滤条件** | 序列长度 < 490 | 序列长度 < 490 |
| **数据特点** | 同源数据集内划分 | 跨数据集独立测试 |
| **Baseline F1** | 0.7700 | 0.9043 |

---

## 数据加载代码位置

- **主线 PriFold baseline**：`/root/aigame/dannyyan/PriFold/utils/tools.py` 中的 `load_data()` 函数
- **SymFlow 实验线**：`/root/aigame/dannyyan/PriFold/symfold/data.py` 中的 `build_records()` + `build_loader()` 函数

### 数据加载模式（mode 参数）

```python
# 训练/验证/测试 模式
--mode bprna          # 实验一：bpRNA 数据集
--mode rnastralign    # 实验二：RNAStrAlign + ArchiveII

# 推理模式
--mode bprna-test          # bpRNA TS0 测试集
--mode rnastralign-test    # RNAStrAlign ts 测试集
--mode archiveii-test      # ArchiveII 独立测试集
```

---

## 相关配置文件

| 实验 | 配置文件路径 |
|------|-------------|
| v7 bpRNA | `symfold/config/v7_full.json` |
| v6 bpRNA | `symfold/config/v6_full.json` |
| v5 bpRNA | `symfold/config/v5_bprna.json` |
| v4 bpRNA | `symfold/config/v4_bprna.json` |
| v4 RNAStrAlign | `symfold/config/v4_rnastralign.json` |
| 主线 baseline | `config/` 目录下的 JSON 配置 |
