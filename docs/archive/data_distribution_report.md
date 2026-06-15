# PriFold 数据分布统计

本文档由 `examples/analyze_data_distribution.py` 生成，统计 `data/` 下被当前 `train.py` / `inference.py` 使用的数据。

说明：下方“数据集划分统计”中的 `n` 是经过 `seq` 长度 `< 490` 过滤后的数量。

## 论文对原始数据的处理

这里的“原始数据”指 PriFold 论文/项目发布的数据包中已经整理好的 CSV 与 contact map 文件，而不是 bpRNA、RNAStrAlign、ArchiveII 外部数据库的最初始下载文件。

处理流程如下：

1. 读取论文发布的数据表：`data/bprna/bpRNA.csv`、`data/RNAStrAlign/rnastralign.csv`、`data/archiveII/archiveII.csv`。
2. 每条样本保留 RNA 序列 `seq`，并通过 `file_name` 关联对应的二级结构 contact map：`*.npy`。
3. 按模型最大长度限制过滤样本：只保留 `seq` 长度 `< 490` 的序列，长度 `>= 490` 的样本不进入当前训练/推理统计。
4. 按论文数据包中的划分字段使用数据：
   - bpRNA：`TR0` 作为训练集，`VL0` 作为验证集，`TS0` 作为测试集。
   - RNAStrAlign：`tr` 作为训练集，`ts` 作为验证/测试集；`vl` 存在于 CSV 中，但当前 `train.py` / `inference.py` 主流程未使用。
   - ArchiveII：作为独立测试集使用。
5. 运行时进一步把序列中的 `U` 替换为 `T`，加载对应 contact map，并在 batch 中 padding 到当前 batch 的最大长度；训练时可选使用 RNA covariation 数据增强和 label smoothing。

## 原始与过滤后数据量

| dataset | raw_n | filtered_n | removed_n |
| --- | ---: | ---: | ---: |
| bpRNA | 13419 | 13409 | 10 |
| RNAStrAlign | 26078 | 25219 | 859 |
| ArchiveII | 3966 | 3845 | 121 |

## 数据集划分统计

| dataset | split | n | len_min | len_p25 | len_mean | len_median | len_p75 | len_max | pair_mean | pair_median | density_mean | density_median | ct_missing | ct_shape_bad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ArchiveII | test | 3845 | 28 | 106.000 | 192.398 | 120.000 | 306.000 | 489 | 54.033 | 35.000 | 0.279 | 0.283 | 0 | 0 |
| RNAStrAlign | train | 20234 | 30 | 76.000 | 142.621 | 116.000 | 122.000 | 489 | 39.728 | 32.000 | 0.279 | 0.282 | 0 | 0 |
| RNAStrAlign | unused(vl) | 2493 | 36 | 76.000 | 143.654 | 117.000 | 122.000 | 489 | 40.257 | 32.000 | 0.280 | 0.283 | 0 | 0 |
| RNAStrAlign | val/test | 2492 | 30 | 77.000 | 144.177 | 116.000 | 122.000 | 488 | 40.251 | 32.000 | 0.279 | 0.282 | 0 | 0 |
| bpRNA | test | 1303 | 22 | 80.000 | 135.497 | 108.000 | 151.500 | 481 | 31.111 | 24.000 | 0.228 | 0.239 | 0 | 0 |
| bpRNA | train | 10807 | 33 | 80.000 | 133.543 | 105.000 | 151.000 | 487 | 30.679 | 25.000 | 0.227 | 0.242 | 0 | 0 |
| bpRNA | val | 1299 | 33 | 80.000 | 131.587 | 106.000 | 150.500 | 454 | 30.235 | 25.000 | 0.227 | 0.237 | 0 | 0 |

## 碱基组成比例

| dataset | split | A | C | G | N | T |
| --- | --- | --- | --- | --- | --- | --- |
| ArchiveII | test | 0.237 | 0.252 | 0.299 | 0.000 | 0.213 |
| RNAStrAlign | train | 0.236 | 0.250 | 0.292 | 0.008 | 0.214 |
| RNAStrAlign | unused(vl) | 0.238 | 0.249 | 0.292 | 0.005 | 0.215 |
| RNAStrAlign | val/test | 0.238 | 0.250 | 0.292 | 0.005 | 0.215 |
| bpRNA | test | 0.252 | 0.232 | 0.264 | 0.003 | 0.249 |
| bpRNA | train | 0.252 | 0.232 | 0.264 | 0.003 | 0.249 |
| bpRNA | val | 0.251 | 0.235 | 0.263 | 0.003 | 0.248 |

## 可视化

![Length Histogram](../outputs/20260525_1851_data_distribution/length_hist_by_split.svg)

![Length Boxplot](../outputs/20260525_1851_data_distribution/length_boxplot.svg)

![Base Composition](../outputs/20260525_1851_data_distribution/base_composition.svg)

![Pair Count vs Length](../outputs/20260525_1851_data_distribution/pair_count_vs_length.svg)

![Contact Density](../outputs/20260525_1851_data_distribution/contact_density_hist.svg)

## 当前项目如何使用这些数据

### 训练 `--mode bprna`

- CSV: `data/bprna/bpRNA.csv`
- 过滤: `seq` 长度 `< 490`
- 训练: `data_name == TR0`，contact map 在 `data/bprna/ct/TR0/{file_name}.npy`
- 验证: `data_name == VL0`，contact map 在 `data/bprna/ct/VL0/{file_name}.npy`
- 测试: `data_name == TS0`，contact map 在 `data/bprna/ct/TS0/{file_name}.npy`

### 训练 `--mode rnastralign`

- CSV: `data/RNAStrAlign/rnastralign.csv` 与 `data/archiveII/archiveII.csv`
- 过滤: `seq` 长度 `< 490`
- 训练: RNAStrAlign 中 `data_name == tr`，contact map 在 `data/RNAStrAlign/{file_name}.npy`
- 验证: RNAStrAlign 中 `data_name == ts`，contact map 在 `data/RNAStrAlign/{file_name}.npy`
- 测试: ArchiveII 全部样本，contact map 在 `data/archiveII/ct/{file_name}.npy`

### 推理测试

- `--mode bprna-test`: 使用 bpRNA 的 `TS0`
- `--mode rnastralign-test`: 使用 RNAStrAlign 的 `ts`
- `--mode archiveii-test`: 使用 ArchiveII 全部样本

### Batch 中的数据结构

`train.py` 和 `inference.py` 的 `collate_fn` 会把单样本 `(seq, ct, _)` 组装为:

```python
{
  'input_ids': Tensor[B, max_len],
  'attention_mask': Tensor[B, max_len],
  'pos_bias': Tensor[B, max_len, max_len],
  'ct': FloatTensor[B, max_len, max_len],
  'ct_mask': FloatTensor[B, max_len, max_len],
  'seq_len': Tensor[B],
}
```

其中 `max_len = max(len(seq) + 2)`，`+2` 是 tokenizer 特殊 token 位置；`ct` 是二级结构 contact map 标签。