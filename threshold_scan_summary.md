# PriFold 阈值扫描与长度过滤实验结果总结

- 实验日期：2026-08-24
- 项目目录：`/root/efs/dannyyan/PriFold`
- 实验性质：使用已有 PriFold 权重进行推理和阈值扫描，未重新训练模型。
- 完成状态：6/6 组实验完成，批量日志显示 `ALL DONE`。

## 1. 实验目的

本实验重新执行 PriFold 在三个测试集上的推理，并分别比较两种序列长度过滤条件：

- `<490`：保持 reproduction guide 中的原始测试条件；
- `<512`：放宽筛选条件，纳入长度为 490--511 的序列。

对于每组实验，使用 0.01 为步长扫描阈值 `0.00, 0.01, ..., 1.00`，记录每个阈值下的 Precision、Recall 和 F1，并选取宏平均 F1 最高的阈值。

## 2. 模型和数据设置

### 2.1 结构模型映射

| 测试集 | 结构模型 |
|---|---|
| bpRNA | `model/ss_model_bprna.pth` |
| RNAStrAlign | `model/ss_model_rnastralign.pth` |
| ArchiveII | `model/ss_model_rnastralign.pth` |

模型映射由 `threshold_scan_fast.py` 第 21--25 行定义。三个实验均使用 MARS-160M 语言模型，通过 `utils/lm.py` 第 4--20 行加载：

```text
model/mars-160m/ckpt_175000.pt
```

RNAformer 配置为：

```text
utils/RNAformer/models/RNAformer_32M_config_bprna_slow.yml
```

### 2.2 推理参数

| 参数 | 设置 |
|---|---:|
| Conda 环境 | `prifold` |
| GPU | `CUDA_VISIBLE_DEVICES=1` |
| `model_scale` | `160m` |
| `batch_size` | `1` |
| `num_workers` | `8` |
| Position bias `scale` | `0.01` |
| 随机种子 | `3407` |
| 数据增强 | 关闭 |
| Label smoothing | 关闭 |
| 阈值扫描 | `0.00`--`1.00`，步长 `0.01` |

命令行默认参数见 `threshold_scan_fast.py` 第 143--160 行；Position bias 在 `threshold_scan_fast.py` 第 28--32 行构造，具体计算逻辑位于 `utils/tools.py` 第 61--79 行。

### 2.3 数据过滤和样本数

测试集构造逻辑见 `threshold_scan_fast.py` 第 43--60 行。各长度条件下的样本数如下：

| 数据集 | `<490` | `<512` | 新增样本 |
|---|---:|---:|---:|
| bpRNA TS0 | 1303 | 1305 | 2 |
| RNAStrAlign test | 2492 | 2513 | 21 |
| ArchiveII | 3845 | 3864 | 19 |

这里的 `<490` 和 `<512` 是严格小于过滤条件，因此长度 490 不包含在 `<490` 实验中，长度 490--511 包含在 `<512` 实验中。

## 3. 评测方法

模型只执行一次 forward，并缓存每个样本的 sigmoid 概率和标签，相关代码位于 `threshold_scan_fast.py` 第 90--108 行。之后对每个阈值分别计算：

```text
pred = probability > threshold
```

阈值序列由 `threshold_scan_fast.py` 第 114 行生成。每个样本先独立计算 TP、FP、FN、Precision、Recall 和 F1，再对所有样本取算术平均，计算逻辑见 `threshold_scan_fast.py` 第 117--133 行。因此，本文结果是逐样本宏平均，不是将所有矩阵拼接后的 micro 指标。

最佳阈值通过 F1 最大值确定，代码见 `threshold_scan_fast.py` 第 134--140 行。

## 4. 阈值扫描结果

### 4.1 最佳 F1 阈值

| 数据集 | 长度条件 | 最佳阈值 | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| bpRNA | `<490` | 0.38 | 0.783041 | 0.771909 | **0.770671** |
| bpRNA | `<512` | 0.38 | 0.782588 | 0.770983 | **0.769866** |
| RNAStrAlign | `<490` | 0.54 | 0.977925 | 0.971517 | **0.974030** |
| RNAStrAlign | `<512` | 0.54 | 0.976993 | 0.970302 | **0.972949** |
| ArchiveII | `<490` | 0.53 | 0.920692 | 0.896355 | **0.905254** |
| ArchiveII | `<512` | 0.53 | 0.919140 | 0.894543 | **0.903484** |

完整的 101 个阈值结果分别保存在：

```text
scan_fast_bprna_lt490.log
scan_fast_bprna_lt512.log
scan_fast_rnastralign_lt490.log
scan_fast_rnastralign_lt512.log
scan_fast_archiveii_lt490.log
scan_fast_archiveii_lt512.log
```

### 4.2 原始阈值 0.45 的结果

原始复现流程使用阈值 `0.45`。本次扫描在该阈值下得到：

| 数据集 | 长度条件 | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| bpRNA | `<490` | 0.793806 | 0.762325 | **0.770013** |
| bpRNA | `<512` | 0.793419 | 0.761405 | **0.769208** |
| RNAStrAlign | `<490` | 0.974228 | 0.974376 | **0.973758** |
| RNAStrAlign | `<512` | 0.973142 | 0.973283 | **0.972667** |
| ArchiveII | `<490` | 0.910162 | 0.903659 | **0.904338** |
| ArchiveII | `<512` | 0.908591 | 0.901900 | **0.902594** |

这些数值与 `reproduction_guide.md` 中记录的原始 `<490` 和 `<512` 结果一致，说明本次模型加载、数据处理和指标计算流程复现正确。

## 5. `<490` 与 `<512` 对比

### 5.1 在原始阈值 0.45 下

| 数据集 | F1 `<490` | F1 `<512` | 变化 |
|---|---:|---:|---:|
| bpRNA | 0.770013 | 0.769208 | -0.000805 |
| RNAStrAlign | 0.973758 | 0.972667 | -0.001091 |
| ArchiveII | 0.904338 | 0.902594 | -0.001744 |

### 5.2 各自使用最佳阈值时

| 数据集 | 最佳阈值 | F1 `<490` | F1 `<512` | 变化 |
|---|---:|---:|---:|---:|
| bpRNA | 0.38 | 0.770671 | 0.769866 | -0.000805 |
| RNAStrAlign | 0.54 | 0.974030 | 0.972949 | -0.001081 |
| ArchiveII | 0.53 | 0.905254 | 0.903484 | -0.001770 |

### 5.3 结论

1. 放宽长度筛选到 `<512` 后，三个数据集的 F1 均出现轻微下降。
2. 下降幅度最大的是 ArchiveII，约为 `0.0017`；其次是 RNAStrAlign，约为 `0.0011`；bpRNA 约为 `0.0008`。
3. 最优阈值在两种长度条件下保持稳定：bpRNA 为 `0.38`，RNAStrAlign 为 `0.54`，ArchiveII 为 `0.53`。
4. `<512` 结果只能表示“将长度 490--511 的样本纳入现有模型推理后的表现”，不能等价于使用专门按 512 长度训练的位置编码模型。
5. 本次扫描结果显示，原始固定阈值 `0.45` 已经接近 bpRNA 的最佳 F1，但在 RNAStrAlign 和 ArchiveII 上分别可通过阈值 `0.54` 和 `0.53` 获得更高的宏平均 F1。

## 6. 结果文件和运行记录

批量运行脚本：

```text
run_remaining.sh
```

该脚本依次运行六组实验，具体命令见 `run_remaining.sh` 第 1--24 行。批量调度日志：

```text
run_remaining_master.log
```

日志中记录了：

```text
done bprna lt490
done bprna lt512
done rnastralign lt490
done rnastralign lt512
done archiveii lt490
done archiveii lt512
ALL DONE
```

本实验没有修改已有模型权重、数据集或原始推理结果日志。
