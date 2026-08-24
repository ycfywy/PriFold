# PriFold 结构预测复现实验说明

本文完整记录当前工作区中 bpRNA、RNAStrAlign 和 ArchiveII 复现实验的模型组成、权重使用方式、数据处理、推理参数、评测方法、实验结果，以及后续 `<512` 长度对比实验的具体含义。

本文对应的项目目录为：

```text
/efs/dannyyan/PriFold
```

## 1. 实验目标

这些实验不是重新训练模型，而是使用已经保存好的 PriFold 权重，在三个 RNA 二级结构数据集上执行测试集推理，复现 precision、recall 和 F1 指标。

复现涉及两个层次的模型：

1. MARS-160M RNA 语言模型：提供 RNA 序列特征。
2. PriFold/RNAformer 结构预测模型：根据序列特征预测碱基配对矩阵。

因此，`ss_model_bprna.pth` 和 `ss_model_rnastralign.pth` 是结构预测模型权重；它们不是 MARS 语言模型权重。

## 2. 使用的模型文件

当前 `model` 目录下相关文件为：

```text
/efs/dannyyan/PriFold/model/ss_model_bprna.pth
/efs/dannyyan/PriFold/model/ss_model_rnastralign.pth
/efs/dannyyan/PriFold/model/mars-160m/ckpt_175000.pt
```

### 2.1 结构预测模型权重

| 测试集 | 使用的结构模型权重 | 含义 |
|---|---|---|
| bpRNA TS0 | `model/ss_model_bprna.pth` | 在 bpRNA 训练设置下得到的结构模型 |
| RNAStrAlign test | `model/ss_model_rnastralign.pth` | 在 RNAStrAlign 训练设置下得到的结构模型 |
| ArchiveII | `model/ss_model_rnastralign.pth` | 使用 RNAStrAlign 模型进行跨数据集测试 |

ArchiveII 没有使用单独的 `ss_model_archiveii.pth`。当时的配置是使用 RNAStrAlign 结构模型测试 ArchiveII。

### 2.2 MARS 语言模型

三个测试都使用同一个 MARS-160M 语言模型：

```text
model/mars-160m/ckpt_175000.pt
```

`utils/lm.py` 根据 `--pretrained_lm_dir` 和模型规模拼接语言模型路径：

```text
<pretrained_lm_dir>/mars-160m/ckpt_175000.pt
```

语言模型通过 `load_model(..., device='cpu')` 首先加载到 CPU，随后作为 `RiboFormer` 的序列特征提取器传入结构预测模型。

## 3. 整体推理流程

每次运行 `inference.py` 的流程如下：

1. 解析命令行参数。
2. 设置 NumPy 和 PyTorch 随机种子，默认是 `3407`。
3. 加载 MARS-160M 语言模型和 tokenizer。
4. 读取 RNAformer 配置文件。
5. 构造 `RiboFormer`。
6. 使用 `--model_path` 读取结构模型 checkpoint。
7. 从 checkpoint 中取出 `model_state_dict`。
8. 如果参数名带有 `module.` 前缀，则去掉该前缀。
9. 将结构模型权重加载到 `RiboFormer`。
10. 根据 `--mode` 读取相应测试集。
11. 将 RNA 序列编码、补齐，并构造 position bias。
12. 使用模型输出配对 logits。
13. 经过 sigmoid 后以 `0.45` 为阈值生成二值配对矩阵。
14. 与真实的 CT 矩阵逐样本计算 precision、recall 和 F1。
15. 对所有测试样本的指标取算术平均。

结构模型 checkpoint 的核心加载逻辑位于 `inference.py`：

```text
checkpoint = torch.load(args.model_path)
original_state_dict = checkpoint['model_state_dict']
model.load_state_dict(new_state_dict)
```

## 4. 原始推理入口

项目中的入口脚本是：

```text
/efs/dannyyan/PriFold/inference.sh
```

其三段任务分别对应：

```text
bprna-test
rnastralign-test
archiveii-test
```

原始脚本中的结构模型映射如下：

```text
bprna-test       -> ./model/ss_model_bprna.pth
rnastralign-test -> ./model/ss_model_rnastralign.pth
archiveii-test   -> ./model/ss_model_rnastralign.pth
```

脚本使用的主要参数是：

```text
--batch_size 1
--scale 0.01
--pretrained_lm_dir ./model
--data_dir ./data
```

## 5. 每个测试模式的数据处理

数据加载逻辑位于：

```text
utils/tools.py
```

### 5.1 bpRNA

`bprna-test` 的数据处理步骤：

1. 读取：

   ```text
   data/bprna/bpRNA.csv
   ```

2. 只保留序列长度满足：

   ```python
   len(seq) < 490
   ```

3. 只保留 `data_name == 'TS0'` 的样本。
4. CT 矩阵从以下目录读取：

   ```text
   data/bprna/ct/TS0
   ```

5. 结构模型使用：

   ```text
   model/ss_model_bprna.pth
   ```

原始 `<490` 条件下，bpRNA TS0 实际测试样本数为 `1303`。

### 5.2 RNAStrAlign

`rnastralign-test` 的数据处理步骤：

1. 读取：

   ```text
   data/RNAStrAlign/rnastralign.csv
   ```

2. 只保留序列长度满足：

   ```python
   len(seq) < 490
   ```

3. 只保留 `data_name == 'ts'` 的样本。
4. 从以下目录读取对应 CT 矩阵：

   ```text
   data/RNAStrAlign
   ```

5. 结构模型使用：

   ```text
   model/ss_model_rnastralign.pth
   ```

原始 `<490` 条件下，RNAStrAlign 测试样本数为 `2492`。

### 5.3 ArchiveII

`archiveii-test` 的数据处理步骤：

1. 读取：

   ```text
   data/archiveII/archiveII.csv
   ```

2. 只保留序列长度满足：

   ```python
   len(seq) < 490
   ```

3. 使用过滤后的全部样本，不再按训练集、验证集或测试集字段二次划分。
4. CT 矩阵从以下目录读取：

   ```text
   data/archiveII/ct
   ```

5. 结构模型使用：

   ```text
   model/ss_model_rnastralign.pth
   ```

这表示 ArchiveII 实验是 RNAStrAlign 模型的跨数据集泛化测试。

原始 `<490` 条件下，ArchiveII 测试样本数为 `3845`。

## 6. 序列和标签的具体处理

数据集类位于：

```text
utils/predictor.py
```

对每个样本：

1. 从 CSV 读取 `seq`。
2. 将 RNA 字母 `U` 替换成 `T`：

   ```python
   seq = seq.replace('U', 'T')
   ```

3. 根据 CSV 中的 `file_name` 拼接 CT 文件路径。
4. 使用 `numpy.load` 读取 CT 矩阵。
5. 推理阶段不启用数据增强。
6. 推理阶段不启用 label smoothing。

因此，这些复现实验使用的是原始序列和原始 CT 标签，不包含 covariation augmentation。

## 7. Position bias

`inference.py` 的 `collate_fn` 会调用：

```text
utils/tools.py::get_posbias
```

本次复现使用：

```text
--scale 0.01
```

position bias 中使用的碱基配对分数为：

| 碱基组合 | 分数 |
|---|---:|
| A-T / T-A | 3 |
| G-C / C-G | 6 |
| G-T / T-G | 1 |

具体 bias 值为：

```text
1.0 + pair_score * scale
```

所以本次实验中对应的附加值为：

- A-T：`1.03`
- G-C：`1.06`
- G-T：`1.01`

position bias 矩阵随后在边界处 padding，并与 tokenizer 产生的输入一起送入结构模型。

## 8. 模型配置

推理默认使用：

```text
utils/RNAformer/models/RNAformer_32M_config_bprna_slow.yml
```

关键配置如下：

| 配置项 | 值 |
|---|---:|
| `max_len` | `490` |
| `model_dim` | `256` |
| `n_layers` | `4` |
| `num_head` | `4` |
| `precision` | `bf16` |
| `seq_vocab_size` | `5` |
| `rel_pos_enc` | `true` |
| `posbias` | `true` |
| `cycling` | `false` |
| `flash_attn` | `false` |

这里的 `max_len=490` 与数据加载阶段的 `len(seq) < 490` 相对应。

需要注意，`len(seq) < 490` 的含义是：

- 长度 `489`：保留
- 长度 `490`：过滤
- 长度大于 `490`：过滤

## 9. 推理参数

复现实验使用的主要推理设置如下：

| 参数 | 值 | 说明 |
|---|---:|---|
| `batch_size` | `1` | 每次推理一条序列 |
| `num_workers` | `8` | DataLoader 工作进程数，来自默认值 |
| `scale` | `0.01` | position bias 缩放系数 |
| `alpha` | `None` | 不启用 label smoothing |
| `select` | `0` | 不启用数据增强 |
| `replace` | `0` | 不启用数据增强 |
| `seed` | `3407` | NumPy/PyTorch 随机种子 |
| threshold | `0.45` | sigmoid 后的二值化阈值 |

虽然参数解析器中有 `--device`，但当前 `inference.py` 实际评测时使用的是：

```python
torch.device('cuda')
```

因此实际使用哪一张卡取决于运行环境中的 CUDA 可见设备设置。此前复现实验使用的是 CUDA 0。

## 10. 评测方式

模型输出的是配对 logits 矩阵：

```text
logits
```

对 logits 做 sigmoid：

```text
probs = sigmoid(logits)
```

然后使用固定阈值 `0.45`：

```text
pred = probs > 0.45
```

真实标签是数据集提供的 CT 矩阵。

每条序列分别计算：

```text
precision_score(label_np, pred)
recall_score(label_np, pred)
f1_score(label_np, pred)
```

最后分别对所有样本的 precision、recall 和 F1 求平均：

```text
final_precision = mean(per_sample_precision)
final_recall    = mean(per_sample_recall)
final_f1        = mean(per_sample_f1)
```

因此日志中的结果是“逐样本指标的宏平均”，不是将所有样本的矩阵拼接后计算的全局 micro 指标。

## 11. 原始 `<490` 复现结果

结果日志文件：

```text
reproduce_cuda0_bprna.log
reproduce_cuda0_rnastralign.log
reproduce_cuda0_archiveii.log
```

### 11.1 bpRNA

使用：

```text
model/ss_model_bprna.pth
```

日志显示：

```text
Loaded model from epoch 99
len of dataset: 1303
Final results: precision: 0.793806, recall: 0.762325, F1: 0.770013
```

结果：

| 指标 | 数值 |
|---|---:|
| Precision | `0.793806` |
| Recall | `0.762325` |
| F1 | `0.770013` |

### 11.2 RNAStrAlign

使用：

```text
model/ss_model_rnastralign.pth
```

日志显示：

```text
Loaded model from epoch 5
len of dataset: 2492
Final results: precision: 0.974228, recall: 0.974376, F1: 0.973758
```

结果：

| 指标 | 数值 |
|---|---:|
| Precision | `0.974228` |
| Recall | `0.974376` |
| F1 | `0.973758` |

### 11.3 ArchiveII

使用：

```text
model/ss_model_rnastralign.pth
```

日志显示：

```text
Loaded model from epoch 5
len of dataset: 3845
Final results: precision: 0.910162, recall: 0.903659, F1: 0.904338
```

结果：

| 指标 | 数值 |
|---|---:|
| Precision | `0.910162` |
| Recall | `0.903659` |
| F1 | `0.904338` |

### 11.4 汇总

| 数据集 | 结构模型 | 测试样本数 | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| bpRNA TS0 | `ss_model_bprna.pth` | 1303 | 0.793806 | 0.762325 | 0.770013 |
| RNAStrAlign | `ss_model_rnastralign.pth` | 2492 | 0.974228 | 0.974376 | 0.973758 |
| ArchiveII | `ss_model_rnastralign.pth` | 3845 | 0.910162 | 0.903659 | 0.904338 |

## 12. `<512` 长度对比实验

之后进行了一个长度过滤敏感性实验。这个实验没有重新训练模型，也没有更换结构模型权重，只将临时实验代码中的过滤条件由：

```python
len(seq) < 490
```

改为：

```python
len(seq) < 512
```

模型配置仍然是：

```text
max_len: 490
```

因此 `<512` 实验的含义是：

> 放宽测试数据筛选条件后，观察原模型对长度 490--511 序列的实际运行结果。

它不是一个经过 `max_len=512` 训练的位置编码模型实验。

### 12.1 测试集规模变化

| 数据集 | 原始样本数 | `<490` | `<512` | 新增样本 |
|---|---:|---:|---:|---:|
| bpRNA TS0 | 1305 | 1303 | 1305 | 2 |
| RNAStrAlign | 2574 | 2492 | 2513 | 21 |
| ArchiveII | 3966 | 3845 | 3864 | 19 |

### 12.2 `<512` 结果

对应日志文件：

```text
reproduce_cuda0_bprna_lt512.log
reproduce_cuda0_rnastralign_lt512.log
reproduce_cuda0_archiveii_lt512.log
```

结果如下：

| 数据集 | Precision | Recall | F1 |
|---|---:|---:|---:|
| bpRNA TS0 | 0.793419 | 0.761405 | 0.769208 |
| RNAStrAlign | 0.973142 | 0.973283 | 0.972667 |
| ArchiveII | 0.908591 | 0.901900 | 0.902594 |

与 `<490` 的 F1 差异：

| 数据集 | F1(`<490`) | F1(`<512`) | 差值 |
|---|---:|---:|---:|
| bpRNA TS0 | 0.770013 | 0.769208 | -0.000805 |
| RNAStrAlign | 0.973758 | 0.972667 | -0.001091 |
| ArchiveII | 0.904338 | 0.902594 | -0.001744 |

### 12.3 `<512` 实验的限制

虽然 `<512` 实验可以实际运行，但需要注意：

1. 结构模型配置中的 `max_len` 仍为 `490`。
2. 相对位置编码的索引会被限制到 `max_len - 1`。
3. 因此长度 490--511 的样本并不是在专门支持长度 512 的模型配置下推理。
4. 该实验只能说明“放宽数据过滤后原模型的实际表现”，不能等价于“512 长度模型的正式 benchmark”。

## 13. 运行命令

### 13.1 推荐的单独运行方式

以下命令假定当前工作目录是项目根目录：

```bash
cd /efs/dannyyan/PriFold
```

由于当前 `utils/lm.py` 中实际识别的模型规模是 `160m`，推荐使用 `--model_scale 160m`：

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --mode bprna-test \
  --model_scale 160m \
  --batch_size 1 \
  --scale 0.01 \
  --model_path /efs/dannyyan/PriFold/model/ss_model_bprna.pth \
  --pretrained_lm_dir /efs/dannyyan/PriFold/model \
  --data_dir /efs/dannyyan/PriFold/data
```

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --mode rnastralign-test \
  --model_scale 160m \
  --batch_size 1 \
  --scale 0.01 \
  --model_path /efs/dannyyan/PriFold/model/ss_model_rnastralign.pth \
  --pretrained_lm_dir /efs/dannyyan/PriFold/model \
  --data_dir /efs/dannyyan/PriFold/data
```

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
  --mode archiveii-test \
  --model_scale 160m \
  --batch_size 1 \
  --scale 0.01 \
  --model_path /efs/dannyyan/PriFold/model/ss_model_rnastralign.pth \
  --pretrained_lm_dir /efs/dannyyan/PriFold/model \
  --data_dir /efs/dannyyan/PriFold/data
```

### 13.2 批量运行

理论上可以运行：

```bash
bash /efs/dannyyan/PriFold/inference.sh
```

但当前 `inference.sh` 中写的是：

```text
--model_scale lx
```

而当前 `utils/lm.py` 中的分支只识别：

```text
6m
25m
85m
160m
```

当前代码直接使用 `lx` 会进入 `NotImplementedError`。因此，如果要使用当前工作区代码，建议先将 `inference.sh` 中三处 `--model_scale lx` 改为：

```text
--model_scale 160m
```

这与已有日志中显示的：

```text
loading model from ./model/mars-160m/ckpt_175000.pt
```

是一致的。

已有日志说明当时的实际运行环境或代码版本能够解析该模型规模；但根据当前工作区的 `utils/lm.py`，`160m` 是明确有效的参数值。

## 14. 运行时间和日志

此前日志中：

- bpRNA：约 1303 条样本，整体约 1--2 分钟。
- RNAStrAlign：约 2492 条样本，整体约 3--4 分钟。
- ArchiveII：约 3845 条样本，整体约 8 分钟。

实际时间会受到 GPU、CPU、DataLoader worker、系统负载和首次 CUDA 初始化影响。

日志中开头的模型信息包括：

```text
loading model from ./model/mars-160m/ckpt_175000.pt
number of parameters: 160627104
```

这里的 `160627104` 是 MARS-160M 语言模型参数量，不是结构模型 checkpoint 的参数量。

## 15. 复现时的检查清单

运行前建议确认：

- [ ] 当前目录为 `/efs/dannyyan/PriFold`。
- [ ] 当前 Python 环境已安装 `requirements.txt` 中依赖。
- [ ] `model/ss_model_bprna.pth` 存在。
- [ ] `model/ss_model_rnastralign.pth` 存在。
- [ ] `model/mars-160m/ckpt_175000.pt` 存在。
- [ ] `vocab_esm_mars.txt` 存在。
- [ ] `data/bprna/bpRNA.csv` 存在。
- [ ] `data/RNAStrAlign/rnastralign.csv` 存在。
- [ ] `data/archiveII/archiveII.csv` 存在。
- [ ] 三个数据集对应的 CT 或 `.npy` 文件存在。
- [ ] 使用 `--model_scale 160m`，不要直接使用当前代码不识别的 `lx`。
- [ ] 如果指定 CUDA 0，使用 `CUDA_VISIBLE_DEVICES=0`，避免误用其他 GPU。

## 16. 结论

之前的 reproduce 确实使用了用户提到的两个结构模型：

```text
/efs/dannyyan/PriFold/model/ss_model_bprna.pth
/efs/dannyyan/PriFold/model/ss_model_rnastralign.pth
```

具体映射是：

```text
bpRNA       -> ss_model_bprna.pth
RNAStrAlign -> ss_model_rnastralign.pth
ArchiveII   -> ss_model_rnastralign.pth
```

同时，三个实验都加载了：

```text
/efs/dannyyan/PriFold/model/mars-160m/ckpt_175000.pt
```

原始 `<490` 复现的最终 F1 为：

```text
bpRNA       0.770013
RNAStrAlign 0.973758
ArchiveII   0.904338
```

这些结果对应的模型、数据过滤、推理参数和评测流程均已在本文中列出。