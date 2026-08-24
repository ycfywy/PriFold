# PriFold 在 bpRNA-new 测试集上的评测结果

- 评测日期：2026-08-24
- 项目目录：`/root/efs/dannyyan/PriFold`
- 数据集：`/efs/dannyyan/symfold/data/bprna-new/test.parquet`
- 结构模型：`model/ss_model_bprna.pth`
- 评测状态：已完成

## 1. 评测目的

本实验使用 PriFold 官方 bpRNA 结构预测权重，在独立的 `bpRNA-new` 测试集上进行推理，评估其跨 RNA family 的泛化能力。

数据集说明中指出，`bpRNA-new` 来源于 Rfam 14.2，专门用于 cross-family validation，并且其 RNA families 与 bpRNA-1m 不同，见 `/efs/dannyyan/symfold/data/bprna-new/README.md` 第 22--27 行。

本实验只进行推理和阈值扫描，没有重新训练模型，也没有修改原始数据集或模型权重。

## 2. 数据集信息

输入文件：

```text
/efs/dannyyan/symfold/data/bprna-new/test.parquet
```

该文件包含以下字段：

| 字段 | 含义 |
|---|---|
| `id` | 样本标识 |
| `sequence` | RNA 序列 |
| `secondary_structure` | dot-bracket 二级结构标注 |

数据统计：

| 项目 | 数值 |
|---|---:|
| 样本数 | 5401 |
| 最短序列长度 | 33 |
| 最长序列长度 | 489 |
| 平均序列长度 | 110.0002 |
| 序列和结构长度不一致样本 | 0 |

数据中的序列字符包括 `A/C/G/M/N/R/U/Y`，结构字符为 `(`/`)`/`.`。

由于最长序列长度为 489，本次评测不需要额外排除长度超过 PriFold 原始配置限制的样本。

## 3. 模型和运行配置

### 3.1 模型权重

本次使用用户指定的 bpRNA 结构模型权重：

```text
/root/efs/dannyyan/PriFold/model/ss_model_bprna.pth
```

该权重在适配脚本中的默认路径由 `threshold_scan_bprna_new.py` 第 142--160 行定义。

所有样本共享 MARS-160M RNA 语言模型：

```text
/root/efs/dannyyan/PriFold/model/mars-160m/ckpt_175000.pt
```

语言模型通过 `utils/lm.py` 第 4--20 行加载。本次日志显示结构模型从 `epoch 99` 加载。

### 3.2 推理参数

| 参数 | 设置 |
|---|---:|
| Python 环境 | `prifold` |
| GPU | `CUDA:0` |
| `model_scale` | `160m` |
| `batch_size` | `1` |
| `num_workers` | `8` |
| Position bias `scale` | `0.01` |
| 随机种子 | `3407` |
| 数据增强 | 关闭 |
| Label smoothing | 关闭 |
| 阈值范围 | `0.00`--`1.00` |
| 阈值步长 | `0.01` |

Position bias 的构造调用位于 `threshold_scan_bprna_new.py` 第 55--58 行，具体计算逻辑位于 `utils/tools.py` 第 61--79 行。

## 4. 数据适配和标签生成

由于 bpRNA-new 使用 parquet 和 dot-bracket 字段，而 PriFold 原始 `SSDataset` 使用 CSV 与 `.npy` 接触矩阵，因此本次使用专门的适配脚本：

```text
threshold_scan_bprna_new.py
```

适配逻辑如下：

1. 从标准输入读取 parquet 转换后的 JSON 记录，脚本入口见第 145--163 行。
2. 将 RNA 中的 `U` 替换为 PriFold tokenizer 使用的 `T`，见第 43--48 行。
3. 检查序列和结构长度一致性，见第 49--50 行。
4. 将 dot-bracket 中的括号配对转换为对称的二值接触矩阵，见第 29--42 行。
5. 使用 PriFold 原有的 tokenizer、MARS 特征提取器和 RNAformer 结构预测网络。

本数据集的结构标注只包含普通括号，因此当前适配器可以直接使用栈解析括号配对。

## 5. 评测方式

模型输出接触矩阵 logits，随后执行 sigmoid：

```text
probability = sigmoid(logits)
```

每个阈值下使用：

```text
prediction = probability > threshold
```

模型 forward 只执行一次，之后缓存每个样本的概率矩阵和标签，相关代码位于 `threshold_scan_bprna_new.py` 第 80--104 行。

每个样本独立计算 TP、FP、FN、Precision、Recall 和 F1，再对 5401 个样本取算术平均。指标计算位于 `threshold_scan_bprna_new.py` 第 105--132 行，因此结果是逐样本宏平均，不是全体接触矩阵拼接后的 micro 指标。

## 6. 评测结果

### 6.1 固定阈值 0.45

PriFold 原始推理流程使用阈值 `0.45`。本次 bpRNA-new 测试结果为：

| 数据集 | 样本数 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| bpRNA-new | 5401 | 0.438067 | 0.306914 | **0.342034** |

对应日志：

```text
bprna_new_prifold.log
```

日志第 54 行记录了阈值 `0.45` 的结果。

### 6.2 全阈值扫描后的最佳结果

| 最佳阈值 | Precision | Recall | F1 |
|---:|---:|---:|---:|
| **0.21** | 0.357565 | 0.419469 | **0.371907** |

最佳阈值由测试集本身的 F1 最大值确定，选择逻辑位于 `threshold_scan_bprna_new.py` 第 133--140 行。

与固定阈值 `0.45` 相比：

| 指标 | 阈值 0.45 | 最佳阈值 0.21 | 变化 |
|---|---:|---:|---:|
| Precision | 0.438067 | 0.357565 | -0.080502 |
| Recall | 0.306914 | 0.419469 | +0.112555 |
| F1 | 0.342034 | 0.371907 | +0.029873 |

完整的 `0.00`--`1.00` 阈值扫描表保存在：

```text
bprna_new_prifold.log
```

## 7. 与原始 bpRNA TS0 结果的比较

此前 PriFold 在原始 bpRNA TS0 测试集上的结果为：

| 数据集 | 阈值 | F1 |
|---|---:|---:|
| bpRNA TS0 | 0.45 | 0.770013 |
| bpRNA-new | 0.45 | 0.342034 |
| bpRNA-new | 0.21（测试集后验最优） | 0.371907 |

两者不能简单视为同一 benchmark 的结果，原因包括：

1. `bpRNA-new` 是面向 cross-family 泛化的独立数据集，其 RNA families 与 bpRNA-1m 不同。
2. 原始 bpRNA TS0 使用的是 PriFold 原项目的 CSV 和预生成 CT 矩阵；bpRNA-new 使用 parquet 中的 dot-bracket 标注，并通过本适配器生成接触矩阵。
3. `bpRNA-new` 的结果使用测试集扫描得到的后验最佳阈值 `0.21`，而原始复现结果通常报告固定阈值 `0.45`。
4. 因此，当前结果更适合作为 PriFold 在跨 family 数据上的泛化评测，而不是原始 bpRNA benchmark 的直接复现。

## 8. 结论

1. PriFold 的 bpRNA 结构模型可以在 `bpRNA-new` 的 5401 条测试序列上正常运行。
2. 固定阈值 `0.45` 下，Precision 为 `0.438067`，Recall 为 `0.306914`，F1 为 `0.342034`。
3. 在测试集上扫描阈值后，最佳阈值为 `0.21`，F1 为 `0.371907`。
4. 降低阈值主要提升了 Recall，但 Precision 明显下降。
5. 与原始 bpRNA TS0 的 F1 相比，bpRNA-new 上的性能明显更低，说明该跨 family 测试集对当前 bpRNA 权重具有较强的分布外泛化挑战。
6. 如果需要报告严格、无测试集调参的数据，应预先划分独立验证集选择阈值，再在 bpRNA-new test 上只报告锁定阈值结果。

## 9. 复现指南和运行指令

### 9.1 目录和文件检查

在运行前确认当前目录和关键文件：

```bash
cd /root/efs/dannyyan/PriFold

ls -lh \
  model/ss_model_bprna.pth \
  model/mars-160m/ckpt_175000.pt \
  vocab_esm_mars.txt \
  threshold_scan_bprna_new.py

ls -lh /efs/dannyyan/symfold/data/bprna-new/test.parquet
```

适配脚本默认模型路径、RNAformer 配置路径和推理参数见 `threshold_scan_bprna_new.py` 第 142--160 行。

### 9.2 检查数据集

`bpRNA-new` 的 parquet 读取依赖 `pandas` 和 parquet 引擎。本次环境中 parquet 引擎位于 `symfold` 环境，因此使用 `symfold` 环境读取数据，再通过标准输入传给 `prifold` 环境执行模型推理。

检查数据集结构和样本数：

```bash
/efs/miniconda3/envs/symfold/bin/python - <<'PY'
import pandas as pd

path = "/efs/dannyyan/symfold/data/bprna-new/test.parquet"
df = pd.read_parquet(path)
print("rows:", len(df))
print("columns:", list(df.columns))
print("length min:", df["sequence"].str.len().min())
print("length max:", df["sequence"].str.len().max())
print("length mismatch:", (df["sequence"].str.len() != df["secondary_structure"].str.len()).sum())
PY
```

预期输出为 `5401` 条样本，序列长度范围为 `33--489`，序列和结构长度不一致数量为 `0`。

### 9.3 前台运行命令

下面的命令会：

1. 使用 `symfold` 环境读取 parquet；
2. 将每条记录转换为 JSON Lines；
3. 使用 `prifold` 环境加载 MARS-160M 和 bpRNA 结构模型；
4. 执行 bpRNA-new 推理；
5. 扫描阈值 `0.00`--`1.00`，步长 `0.01`。

```bash
cd /root/efs/dannyyan/PriFold

/efs/miniconda3/envs/symfold/bin/python -c 'import json,pandas as pd; df=pd.read_parquet("/efs/dannyyan/symfold/data/bprna-new/test.parquet"); [print(json.dumps({"id":str(r.id),"sequence":str(r.sequence),"secondary_structure":str(r.secondary_structure)},ensure_ascii=False)) for r in df.itertuples(index=False)]' \
| CUDA_VISIBLE_DEVICES=0 /efs/miniconda3/envs/prifold/bin/python -u threshold_scan_bprna_new.py
```

这里真正执行 PriFold 模型的解释器是：

```text
/efs/miniconda3/envs/prifold/bin/python
```

因此模型推理仍然运行在 `prifold` 环境；`symfold` 解释器只负责读取当前 `prifold` 环境中未安装 parquet 引擎的输入文件。

### 9.4 后台运行命令

如果需要后台运行并保存日志：

```bash
cd /root/efs/dannyyan/PriFold

nohup bash -c '/efs/miniconda3/envs/symfold/bin/python -c "import json,pandas as pd; df=pd.read_parquet(\"/efs/dannyyan/symfold/data/bprna-new/test.parquet\"); [print(json.dumps({\"id\":str(r.id),\"sequence\":str(r.sequence),\"secondary_structure\":str(r.secondary_structure)},ensure_ascii=False)) for r in df.itertuples(index=False)]" | CUDA_VISIBLE_DEVICES=0 /efs/miniconda3/envs/prifold/bin/python -u threshold_scan_bprna_new.py' \
> bprna_new_prifold.log 2>&1 &

echo "PID: $!"
```

本次实验使用的后台日志文件为：

```text
bprna_new_prifold.log
```

### 9.5 监控进度

查看实时日志：

```bash
tail -f /root/efs/dannyyan/PriFold/bprna_new_prifold.log
```

查看进程和 GPU 状态：

```bash
ps -eo pid,etime,args | grep threshold_scan_bprna_new.py | grep -v grep
nvidia-smi
```

推理阶段日志中的进度条应最终达到：

```text
bpRNA-new forward: 100%|██████████| 5401/5401
```

### 9.6 提取结果

提取固定阈值 `0.45`、最佳阈值和最终 F1：

```bash
grep -E '^\s+0\.45\s|Best F1|threshold:' \
  /root/efs/dannyyan/PriFold/bprna_new_prifold.log
```

预期结果：

```text
0.45     0.438067     0.306914     0.342034
threshold: 0.21, precision: 0.357565, recall: 0.419469, F1: 0.371907
```

### 9.7 停止后台任务

如果需要停止任务，先找到主进程 PID：

```bash
ps -eo pid,etime,args | grep threshold_scan_bprna_new.py | grep -v grep
```

然后向对应的主进程发送终止信号：

```bash
kill -TERM <主进程PID>
```

不要只根据 DataLoader worker 的 PID 终止单个子进程；应终止最外层的 `bash -c` 主进程。

适配脚本的数据读取、标签生成和模型推理逻辑分别位于 `threshold_scan_bprna_new.py` 第 43--77 行和第 80--108 行；阈值扫描逻辑位于第 110--140 行。

## 10. 结果文件

```text
threshold_scan_bprna_new.py
bprna_new_prifold.log
bprna_new_prifold_summary.md
```
