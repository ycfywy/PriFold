# PriFold 项目总结

本文综合 `README.md`、`CLAUDE.md`、`docs/` 文档以及当前代码实现，说明 PriFold 的代码结构、RNA 序列处理流程、关键技术、数据集处理与实验结果。

---

## 1. PriFold 是什么

PriFold: **Biological Priors Improve RNA Secondary Structure Predictions** 是一个 RNA 二级结构预测模型。任务是：给定一条 RNA 序列，预测任意两个碱基位置是否配对，最终输出一个 `L × L` 的 contact map。

核心思想：

1. 使用预训练 RNA 语言模型 **MARS** 提取序列上下文特征。
2. 将一维序列特征扩展成二维碱基对特征。
3. 使用 RNAformer / RiboFormer 对 `L × L` 配对矩阵建模。
4. 引入两类生物先验：
   - **碱基配对位置偏置 `pos_bias`**：根据 A-U / G-C / G-U 配对规则引导注意力。
   - **RNA covariation 数据增强**：训练时对已配对碱基做协同替换，模拟进化协变。

整体流程：

```text
RNA 序列
  → U 替换为 T
  → EsmTokenizer
  → MARS 语言模型
  → PairwiseOnly: 1D 序列特征扩展为 2D 配对特征
  → RNAformerStack: 行/列轴注意力 + pos_bias
  → Linear 输出 logits
  → Sigmoid + 阈值
  → L×L 二级结构 contact map
```

---

## 2. 代码结构

### 2.1 顶层文件

| 路径 | 作用 |
| --- | --- |
| `README.md` | 官方项目说明，包含项目简介、模型和数据下载地址、引用信息。 |
| `CLAUDE.md` | 当前项目的中文维护说明，概括模型结构、环境、数据集、实验结果。 |
| `train.py` | 训练主入口，负责加载数据、模型、loss、optimizer、scheduler、评估和保存。 |
| `inference.py` | 推理/评估入口，加载训练好的结构预测模型，在测试集上计算 Precision / Recall / F1。 |
| `train.sh` | 训练启动脚本，默认使用 `accelerate`、4 GPU、bf16。 |
| `inference.sh` | 推理启动脚本，依次评估 `bprna-test`、`rnastralign-test`、`archiveii-test`。 |
| `config_bf16.yaml` | Accelerate 多 GPU 配置，`num_processes=4`，`mixed_precision=bf16`。 |
| `requirements.txt` | Python 依赖。 |
| `vocab_esm_mars.txt` | MARS / EsmTokenizer 使用的 RNA 词表。 |

### 2.2 `utils/` 目录

| 路径 | 作用 |
| --- | --- |
| `utils/tools.py` | 数据集读取、划分、长度过滤；实现 `get_posbias()`。 |
| `utils/predictor.py` | `SSDataset`、contact map 加载、`U -> T`、label smoothing、RNA covariation 数据增强。 |
| `utils/lm.py` | 加载 MARS 预训练语言模型与 `EsmTokenizer`。 |
| `utils/configuration.py` | 读取 RNAformer YAML 配置。 |
| `utils/RNAformer/model/Riboformer_outfirst.py` | 当前主模型实现：`RiboFormer`、`PairwiseOnly`、`RNAformerStack`、`RNAformerBlock`、`TriangleAttention`、`Attention2d`、`ConvFeedForward`。 |
| `utils/RNAformer/models/RNAformer_32M_config_bprna_slow.yml` | 当前默认 RNAformer 配置，`model_dim=256`、`n_layers=4`、`num_head=4`、`max_len=490`。 |

### 2.3 `prifold/` 目录

| 路径 | 作用 |
| --- | --- |
| `prifold/llama2.py` | MARS 语言模型主体，LLaMA2 风格 Transformer，输出 token hidden states。 |
| `prifold/esm2.py` | ESM2 风格模型实现。 |
| `prifold/gpt2.py` | GPT-2 风格模型实现。 |
| `prifold/t5_model.py` | T5 模型实现。 |
| `prifold/llama2_t5.py` | LLaMA2-T5 Encoder-Decoder 变体。 |
| `prifold/modules/` | 语言模型相关模块。 |

### 2.4 文档、日志、输出

| 路径 | 作用 |
| --- | --- |
| `docs/pipeline_walkthrough.md` | 已有的推理流水线说明。 |
| `docs/data_distribution_report.md` | 数据分布统计，包括过滤前后数量、长度、contact density、碱基组成。 |
| `docs/project_convention.md` | 项目目录、日志、输出规范。 |
| `logs/` | 推理/训练日志。当前可见 ArchiveII 推理日志。 |
| `outputs/20260525_1851_data_distribution/` | 数据分布统计输出，包括 CSV 和 SVG 图。 |

---

## 3. 一条 RNA 序列是怎么处理的，以及 shape 如何变化

下面按当前 `train.py` / `inference.py` 实际代码说明。设原始 RNA 长度为 `L`，batch size 为 `B`。因为 tokenizer 会加入特殊 token，代码中 batch 内最大 token 长度记为：

```text
T = max(len(seq) + 2)
```

### 3.1 Dataset 阶段

实现位置：`utils/predictor.py` 中的 `SSDataset.__getitem__()`。

输入来自 CSV：

```text
seq: RNA 序列字符串，长度 L
file_name: contact map 文件名
```

处理：

1. 读取 `seq`。
2. 执行 `seq = seq.replace('U', 'T')`，模型内部用 `T` 表示 RNA 中的 `U`。
3. 根据 `file_name` 加载对应 `*.npy` contact map。
4. 如果是训练集且开启数据增强，则调用 `Augmentation` 修改序列。
5. 如果设置 `smooth`，则额外生成 smoothed contact map；但当前 `collate_fn` 里第三个返回值被 `_` 丢弃，所以主训练路径实际 loss 仍使用原始 `ct`。

单样本输出：

| 名称 | shape / 类型 | 含义 |
| --- | --- | --- |
| `seq` | `str`，长度 `L` | `U` 已替换为 `T` 的序列。 |
| `ct` | `np.ndarray[L, L]` | 二级结构 contact map 标签。 |
| `_` | `None` 或 `np.ndarray[L, L]` | 可选 label smoothing 结果，当前主流程未使用。 |

### 3.2 Collate 阶段

实现位置：`train.py` 和 `inference.py` 的 `collate_fn()`。

对一个 batch 的样本：

```python
seqs, cts, _ = zip(*batch)
max_len = max([len(seq) + 2 for seq in seqs])
```

然后：

1. `EsmTokenizer.batch_encode_plus()` 编码序列，并 padding 到 `max_len`。
2. `get_posbias(seqs, max_len, scale)` 生成生物先验偏置。
3. 将 `ct` 和 `ct_mask` 从 `[L, L]` padding 到 `[T, T]`。
4. 记录原始序列长度 `seq_len = len(seq)`。

Batch 输出结构：

| key | shape | 含义 |
| --- | --- | --- |
| `input_ids` | `Tensor[B, T]` | tokenizer 后的 token id。 |
| `attention_mask` | `Tensor[B, T]` | token padding mask。 |
| `pos_bias` | `Tensor[B, T, T]` | 碱基配对先验矩阵，已在两侧 padding 特殊 token 位置。 |
| `ct` | `FloatTensor[B, T, T]` | contact map 标签，原始 `[L, L]` padding 到 `[T, T]`。 |
| `ct_mask` | `FloatTensor[B, T, T]` | contact map 有效区域 mask。 |
| `seq_len` | `Tensor[B]` | 原始序列长度 `L`，不含 tokenizer 特殊 token。 |

### 3.3 模型前向传播 shape

实现位置：`utils/RNAformer/model/Riboformer_outfirst.py` 中的 `RiboFormer.forward()`。

#### Step 1：MARS 语言模型

输入：

```text
input_ids:      [B, T]
attention_mask: [B, T]
```

调用：

```python
output = self.extractor(tokens=input_ids, attn_mask=attention_mask)
hidden_states = output[1]
```

输出：

```text
hidden_states: [B, T, 1056]
```

其中 `1056` 来自当前 MARS-LX / 160M 规模语言模型。

#### Step 2：PairwiseOnly，一维特征转二维配对特征

实现位置：`PairwiseOnly.forward()`。

流程：

```text
hidden_states: [B, T, 1056]
  → Linear(1056 → 128)
  → 每个位置 i 的特征与每个位置 j 的特征拼接
  → pairwise_concat_embedding: [B, 256, T, T]
  → permute
  → pair_latent: [B, T, T, 256]
```

含义：`pair_latent[b, i, j, :]` 表示第 `i` 个 token 和第 `j` 个 token 的候选配对特征。

#### Step 3：构造 pair mask

```python
pair_mask = self.make_pair_mask(input_ids, data_dict['seq_len'])
```

输出：

```text
pair_mask: [B, T, T]
```

它根据 `seq_len` 生成二维 mask，用于遮蔽 padding 区域。

#### Step 4：RNAformerStack

输入：

```text
pair_latent: [B, T, T, 256]
pair_mask:   [B, T, T]
pos_bias:    [B, T, T]
```

`RNAformerStack` 有 4 层 `RNAformerBlock`。每层结构：

```text
row TriangleAttention
  → column TriangleAttention
  → ConvFeedForward
```

输出：

```text
latent: [B, T, T, 256]
```

#### Step 5：输出层

```python
logits = self.output_mat(latent)
```

输出：

```text
logits: [B, T, T, 1]
```

训练和推理时，会根据 `attention_mask.sum()` 裁剪有效 token 区域，然后与 `ct` 计算 loss 或指标。

### 3.4 训练与推理后处理

训练：

```text
logits → BCEWithLogitsLoss(logits, ct)
```

验证 / 测试：

```text
logits
  → sigmoid
  → threshold
  → binary contact map
  → Precision / Recall / F1
```

当前代码中：

| 场景 | 阈值 |
| --- | --- |
| `train.py` 内每 5 epoch 的验证 / 测试 | `0.5` |
| `inference.py` 独立推理 | `0.45` |

---

## 4. 代码中的关键技术与实现

### 4.1 MARS 预训练语言模型

实现位置：

- 加载：`utils/lm.py`
- 模型：`prifold/llama2.py`

当前默认使用 `model_scale=lx`：

```text
model/mars_run-encoder-mars-lx-train-val-d0.15-2023_10_05_22_03_21/ckpt_175000.pt
```

日志中显示参数量：

```text
160,627,104 parameters
```

MARS 的作用是把 RNA token 序列编码成上下文表示：

```text
input_ids: [B, T]
→ hidden_states: [B, T, 1056]
```

### 4.2 `pos_bias` 碱基配对先验

实现位置：`utils/tools.py:get_posbias()`。

配对分数：

| 碱基对 | 分数 |
| --- | ---: |
| `A-T` / `T-A` | 3 |
| `G-C` / `C-G` | 6 |
| `G-T` / `T-G` | 1 |

计算方式：

```text
posbias 初始为 1
如果 seq[i] 与 seq[j] 是可配对碱基，则：
posbias[i, j] += score * scale
```

默认推理脚本里 `scale=0.01`，所以先验是一个较弱的乘法偏置，不会硬性决定结果。

shape：

```text
原始: [B, T-2, T-2]
两侧 padding 后: [B, T, T]
```

注入方式：

- 在 `RNAformerStack.forward()` 中，`pos_bias` 只传给第 1 层；第 2-4 层使用 `zeros_like(bias)`。
- 在 `Attention2d.forward()` 中，先计算 attention weights 并 softmax，然后执行：

```python
modified_attn_weights = attn_weights * bias
```

即 `pos_bias` 是乘法注意力调制。

### 4.3 PairwiseOnly：1D 到 2D

实现位置：`Riboformer_outfirst.py:PairwiseOnly`。

核心功能：把每个 token 的语言模型 embedding 两两拼接：

```text
h_i: [128]
h_j: [128]
[h_i; h_j]: [256]
```

整体 shape：

```text
[B, T, 1056]
→ Linear(1056, 128)
→ [B, T, T, 256]
```

这是从“单个碱基表示”转向“碱基对表示”的关键步骤。

### 4.4 RNAformer / RiboFormer 轴注意力

实现位置：`Riboformer_outfirst.py`。

当前配置：

| 参数 | 值 |
| --- | ---: |
| `model_dim` | 256 |
| `n_layers` | 4 |
| `num_head` | 4 |
| `max_len` | 490 |
| `ff_kernel` | 3 |
| `precision` | bf16 |
| `posbias` | true |

每个 `RNAformerBlock`：

```text
pair_act
  → TriangleAttention(per_row)
  → TriangleAttention(per_column)
  → ConvFeedForward
```

意义：

- `per_row`：沿 contact map 的行方向建模。
- `per_column`：沿 contact map 的列方向建模。
- 轴注意力避免直接在 `L×L` 矩阵上做全局四维注意力，降低计算复杂度。
- `ConvFeedForward` 使用 `3×3 Conv2d`，可以捕捉 contact map 上相邻配对的局部结构模式。

### 4.5 RNA covariation 数据增强

实现位置：`utils/predictor.py:Augmentation`。

训练时通过参数控制：

```bash
--select 0.1 --replace 0.3
```

含义：

- `select`：一条序列是否被增强的概率。
- `replace`：对该序列中每个已配对碱基对进行替换的概率。

增强流程：

1. 从 contact map 的上三角区域找出真实配对碱基对。
2. 对每个配对位置 `(x, y)`，以 `replace` 概率替换。
3. 根据原配对类型做协同替换，例如：
   - `A-T` 可替换为 `G-C` 或 `G-T`
   - `G-C` 可替换为 `A-T` 或 `G-T`
   - `G-T` 可替换为 `A-T` 或 `G-C`
4. 替换概率使用代码中的自然频率常数，如 `7.24`、`25.77`、`46.3`。

这保证了增强后的序列仍尽量维持合理的碱基配对关系。

### 4.6 Label smoothing

实现位置：`utils/predictor.py:label_smoothing()`。

方法：

- 对 contact map 使用 `3×3` kernel 做卷积。
- 原始为 1 的配对位置保持 1。
- 周围位置赋予较小的平滑值。

注意：当前 `train.py` / `inference.py` 的 `collate_fn` 使用：

```python
seqs, cts, _ = zip(*batch)
```

因此第三个返回值没有进入当前主 loss。除非后续改 `collate_fn` 和 loss，否则 `smooth_ct` 实际不会影响训练。

### 4.7 Loss、优化和评估

训练实现位置：`train.py`。

| 项 | 当前实现 |
| --- | --- |
| Loss | `nn.BCEWithLogitsLoss()` |
| Optimizer | `torch.optim.Adam` |
| Scheduler | `get_cosine_schedule_with_warmup` |
| 分布式 | `Accelerate` |
| 混合精度 | `bf16` |
| 验证频率 | 每 5 个 epoch 评估一次 |
| 训练验证阈值 | `0.5` |
| 独立推理阈值 | `0.45` |
| 指标 | Precision、Recall、F1 |

---

## 5. 数据集、原始数据、分布与处理

### 5.1 使用了哪些数据

当前项目使用 PriFold 论文/项目发布的数据包，数据来自：

1. `bpRNA`
2. `RNAStrAlign`
3. `ArchiveII`

数据包路径：

```text
data/bprna/bpRNA.csv
data/RNAStrAlign/rnastralign.csv
data/archiveII/archiveII.csv
```

对应 contact map：

```text
data/bprna/ct/TR0/{file_name}.npy
data/bprna/ct/VL0/{file_name}.npy
data/bprna/ct/TS0/{file_name}.npy
data/RNAStrAlign/{file_name}.npy
data/archiveII/ct/{file_name}.npy
```

这里的“原始数据”指项目发布数据包中已经整理好的 CSV 与 `*.npy` contact map，而不是外部数据库最初始下载文件。

### 5.2 原始 CSV 长什么样

#### bpRNA

字段：

```text
Unnamed: 0, data_name, file_name, seq, dot_string, seq_len
```

含义：

| 字段 | 含义 |
| --- | --- |
| `data_name` | 数据划分，`TR0` / `VL0` / `TS0`。 |
| `file_name` | contact map 文件名。 |
| `seq` | RNA 序列。 |
| `dot_string` | dot-bracket 二级结构字符串。 |
| `seq_len` | 序列长度。 |

示例样本：

```text
file_name = bpRNA_CRW_2852
seq_len = 400
ct_shape = (400, 400)
pairs = 85
```

#### RNAStrAlign

字段：

```text
file_name, seq, data_name
```

含义：

| 字段 | 含义 |
| --- | --- |
| `file_name` | contact map 路径/文件名，如 `5S_rRNA_database/Bacteria/B01868`。 |
| `seq` | RNA 序列。 |
| `data_name` | 数据划分，`tr` / `ts` / `vl`。 |

示例样本：

```text
file_name = 5S_rRNA_database/Bacteria/B01868
seq_len = 117
ct_shape = (117, 117)
pairs = 33
```

#### ArchiveII

字段：

```text
file_name, seq
```

含义：

| 字段 | 含义 |
| --- | --- |
| `file_name` | contact map 文件名。 |
| `seq` | RNA 序列。 |

示例样本：

```text
file_name = RNaseP_CPB80
seq_len = 301
ct_shape = (301, 301)
pairs = 86
```

### 5.3 论文/项目对原始数据做了哪些处理

当前代码中的处理流程：

1. 从 CSV 读取样本表。
2. 通过 `file_name` 找到对应的 `*.npy` contact map。
3. 过滤长度：只保留 `seq` 长度 `< 490` 的样本。
4. 按数据集字段划分训练/验证/测试：
   - bpRNA：`TR0` 训练，`VL0` 验证，`TS0` 测试。
   - RNAStrAlign：`tr` 训练，`ts` 验证/测试；`vl` 在 CSV 中存在，但当前 `train.py` / `inference.py` 主流程未使用。
   - ArchiveII：作为独立测试集。
5. Dataset 读取时执行 `U -> T`。
6. 训练时可选 RNA covariation 数据增强。
7. Batch 阶段对 token、contact map、mask 做 padding。
8. 生成 `pos_bias` 生物先验。

### 5.4 原始数量与过滤后数量

`docs/data_distribution_report.md` 中的 `n` 是过滤后的数量。

| dataset | 原始数量 | 过滤后数量 | 被过滤掉 |
| --- | ---: | ---: | ---: |
| bpRNA | 13419 | 13409 | 10 |
| RNAStrAlign | 26078 | 25219 | 859 |
| ArchiveII | 3966 | 3845 | 121 |

### 5.5 过滤后的划分分布

| dataset | split | n | len_min | len_mean | len_median | len_max | pair_mean | density_mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ArchiveII | test | 3845 | 28 | 192.398 | 120 | 489 | 54.033 | 0.279 |
| RNAStrAlign | train | 20234 | 30 | 142.621 | 116 | 489 | 39.728 | 0.279 |
| RNAStrAlign | unused(vl) | 2493 | 36 | 143.654 | 117 | 489 | 40.257 | 0.280 |
| RNAStrAlign | val/test | 2492 | 30 | 144.177 | 116 | 488 | 40.251 | 0.279 |
| bpRNA | test | 1303 | 22 | 135.497 | 108 | 481 | 31.111 | 0.228 |
| bpRNA | train | 10807 | 33 | 133.543 | 105 | 487 | 30.679 | 0.227 |
| bpRNA | val | 1299 | 33 | 131.587 | 106 | 454 | 30.235 | 0.227 |

其中：

- `pair_mean`：平均碱基配对数量。
- `density_mean`：contact map 配对密度。
- `ct_missing=0`、`ct_shape_bad=0`，说明当前统计范围内 contact map 文件完整且 shape 与序列长度匹配。

### 5.6 碱基组成

过滤后数据的碱基比例大致如下：

| dataset | split | A | C | G | N | T |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ArchiveII | test | 0.237 | 0.252 | 0.299 | 0.000 | 0.213 |
| RNAStrAlign | train | 0.236 | 0.250 | 0.292 | 0.008 | 0.214 |
| RNAStrAlign | val/test | 0.238 | 0.250 | 0.292 | 0.005 | 0.215 |
| bpRNA | train | 0.252 | 0.232 | 0.264 | 0.003 | 0.249 |
| bpRNA | val | 0.251 | 0.235 | 0.263 | 0.003 | 0.248 |
| bpRNA | test | 0.252 | 0.232 | 0.264 | 0.003 | 0.249 |

注意：这里的 `T` 是代码内部对 RNA `U` 的表示。

---

## 6. 做了哪些实验，实验结果如何

### 6.1 训练实验配置

`train.sh` 当前默认训练 bpRNA：

```bash
accelerate launch --config_file config_bf16.yaml ./train.py \
    --mode bprna \
    --gradient_accumulation_steps 1 \
    --batch_size 1 \
    --lr 1e-4 \
    --select 0.1 --replace 0.3 \
    --pretrained_lm_dir ./model \
    --data_dir ./data \
    --save True
```

训练特点：

| 项 | 配置 |
| --- | --- |
| 分布式框架 | Accelerate |
| GPU 数 | 4 |
| 混合精度 | bf16 |
| 语言模型 | MARS-LX / 160M |
| 下游结构模型 | RNAformer 4 层，`model_dim=256` |
| batch size | 1 |
| learning rate | `1e-4` |
| 数据增强 | `select=0.1`, `replace=0.3` |
| loss | BCEWithLogitsLoss |
| scheduler | cosine schedule with warmup |

### 6.2 推理实验配置

`inference.sh` 中包含三个测试：

```bash
# bpRNA 测试集
python inference.py --mode bprna-test \
    --model_scale lx \
    --batch_size 1 \
    --scale 0.01 \
    --model_path ./model/ss_model_bprna.pth

# RNAStrAlign 测试集
python inference.py --mode rnastralign-test \
    --model_scale lx \
    --batch_size 1 \
    --scale 0.01 \
    --model_path ./model/ss_model_rnastralign.pth

# ArchiveII 测试集
python inference.py --mode archiveii-test \
    --model_scale lx \
    --batch_size 1 \
    --scale 0.01 \
    --model_path ./model/ss_model_rnastralign.pth
```

说明：

- `bpRNA` 使用 `ss_model_bprna.pth`。
- `RNAStrAlign` 与 `ArchiveII` 使用 `ss_model_rnastralign.pth`。
- 推理阈值为 `0.45`。
- 指标为 Precision、Recall、F1。

### 6.3 当前记录的实验结果

`CLAUDE.md` 记录了 2025-05-25 在 NVIDIA H20 上的推理结果：

| 测试集 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| bprna-test | 0.7938 | 0.7623 | 0.7700 |
| rnastralign-test | 0.9742 | 0.9744 | 0.9738 |
| archiveii-test | 0.9102 | 0.9037 | 0.9043 |

当前日志中可确认 ArchiveII 优化推理结果：

```text
Final results: precision: 0.910163, recall: 0.903659, F1: 0.904339
```

对应测试集：

```text
archiveii-test, len of dataset: 3845
model: ss_model_rnastralign.pth
loaded epoch: 5
MARS parameters: 160,627,104
```

### 6.4 结果解读

1. `RNAStrAlign` 测试结果最高，F1 约 `0.9738`，说明模型在与训练集分布相近的数据上表现很强。
2. `ArchiveII` 是独立测试集，F1 约 `0.9043`，说明模型具备较好的跨数据集泛化能力。
3. `bpRNA` 测试 F1 约 `0.7700`，明显低于 RNAStrAlign 和 ArchiveII，可能与 bpRNA 数据分布、结构复杂度或训练 checkpoint 有关。
4. 从数据分布看，bpRNA 的 contact density 约 `0.227`，低于 RNAStrAlign / ArchiveII 的约 `0.279`，类别不平衡和结构分布差异可能影响 Precision / Recall。

---

## 7. 关键代码入口速查

| 问题 | 查看文件 |
| --- | --- |
| 数据怎么加载和划分？ | `utils/tools.py:load_data()` |
| 长度过滤在哪里？ | `utils/tools.py` 中 `df[df['seq'].str.len() < 490]` |
| contact map 怎么加载？ | `utils/predictor.py:SSDataset.__getitem__()` |
| `U -> T` 在哪里做？ | `utils/predictor.py:SSDataset.__getitem__()` |
| batch shape 在哪里形成？ | `train.py:collate_fn()`、`inference.py:collate_fn()` |
| `pos_bias` 怎么算？ | `utils/tools.py:get_posbias()` |
| MARS 怎么加载？ | `utils/lm.py:get_extractor()` |
| 语言模型主体在哪里？ | `prifold/llama2.py:Transformer` |
| 一维特征怎么转二维？ | `utils/RNAformer/model/Riboformer_outfirst.py:PairwiseOnly` |
| RNAformer 主体在哪里？ | `utils/RNAformer/model/Riboformer_outfirst.py:RNAformerStack` |
| row/column attention 在哪里？ | `TriangleAttention` 和 `RNAformerBlock` |
| 数据增强在哪里？ | `utils/predictor.py:Augmentation` |
| loss 和指标在哪里？ | `train.py`、`inference.py` |
| 数据分布报告在哪里？ | `docs/data_distribution_report.md` |

---

## 8. 总结

PriFold 的核心是“语言模型序列表示 + 二维配对建模 + 生物先验”。它先用 MARS 把 RNA 序列编码成上下文 embedding，再通过 `PairwiseOnly` 将 token 特征两两拼接成 `T × T` 配对矩阵，随后用 RNAformer 的行/列轴注意力建模配对关系。代码中显式引入了碱基配对规则 `pos_bias`，并在训练时通过 RNA covariation augmentation 增强模型对进化协变模式的鲁棒性。

数据方面，项目使用 bpRNA、RNAStrAlign 和 ArchiveII 三个数据集，并统一过滤掉长度 `>= 490` 的序列。过滤后共使用：bpRNA 13409 条、RNAStrAlign 25219 条、ArchiveII 3845 条。实验结果显示，模型在 RNAStrAlign 上 F1 约 `0.9738`，在独立 ArchiveII 上 F1 约 `0.9043`，说明该结构在 RNA 二级结构预测任务上具有较强效果和一定泛化能力。
