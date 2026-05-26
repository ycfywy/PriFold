# PriFold-SymFlow 实现记录

本文记录本次在 `PriFold/symfold/` 下实现的 SymFold 风格实验分支：使用 PriFold 的数据处理与 MARS 预训练模型，结合 SymFold 的 DiT + Bernoulli Flow Matching 思路，训练一版 RNA 二级结构 contact map 生成模型。

---

## 1. 实现目标

原始目标：

1. 在 `/root/aigame/dannyyan/PriFold/symfold` 下实现一版模型。
2. 数据处理沿用 PriFold：CSV + `*.npy` contact map，长度 `<490`，`U -> T`。
3. 预训练模型沿用 PriFold 的 MARS-LX。
4. 模型主体参考 SymFold：DiT 主干 + Flow Matching 训练。
5. 条件输入使用 `MARS embedding + outer concat map`。
6. 日志、结果、checkpoint 记录参考 SymFold。
7. 启动一次训练验证链路。

本次实现的是首版 smoke 版本，命名为：

```text
PriFold-SymFlow v0
```

---

## 2. 当前代码结构

新增目录：

```text
symfold/
├── __init__.py
├── data.py
├── discrete_flow.py
├── dit.py
├── eval.py
├── metrics.py
├── model.py
├── train.py
├── run_train.sh
├── config/
│   └── prifold_symflow_v0.json
├── logs/
│   └── 20260526_1508_prifold_symflow_v0/
└── outputs/
    └── 20260526_1508_prifold_symflow_v0/
```

### 2.1 `symfold/data.py`

职责：使用 PriFold 数据格式构建训练/验证/测试 batch。

主要功能：

- 读取 PriFold 数据：
  - `data/bprna/bpRNA.csv`
  - `data/RNAStrAlign/rnastralign.csv`
  - `data/archiveII/archiveII.csv`
- 使用对应 `*.npy` contact map。
- 过滤 `seq` 长度 `< 490`。
- Dataset 中执行 `U -> T`。
- 支持训练集可选 PriFold `Augmentation`。
- batch 内 padding 到 DiT 需要的 `patch_size` 整数倍。
- 生成：
  - tokenizer 输入：`input_ids`、`attention_mask`
  - flow 标签：`contact: [B,1,S,S]`
  - mask：`contact_mask: [B,1,S,S]`
  - 序列 one-hot：`seq_oh: [B,S,4]`
  - PriFold 配对先验：`pos_bias: [B,S,S]`

当前 split：

| stage | 数据 |
| --- | --- |
| `train` | bpRNA `TR0` + RNAStrAlign `tr` |
| `val` | bpRNA `VL0` + RNAStrAlign `ts` |
| `bprna-test` | bpRNA `TS0` |
| `rnastralign-test` | RNAStrAlign `ts` |
| `archiveii-test` | ArchiveII 全部 |
| `rnastralign-vl` | RNAStrAlign `vl`，可选 |

### 2.2 `symfold/discrete_flow.py`

职责：实现 SymFold 风格的 Bernoulli Flow Matching。

包含：

- `sample_x_t_given_x_1()`：前向 noising。
- `symmetrize_binary()`：二值矩阵对称化。
- `symmetrize_logit()`：logit 对称化。
- `compute_ctmc_rates()`：CTMC 采样 rate。
- `valid_pair_mask()`：过滤 padding 和 `|i-j| < 3` 的短程位置。
- `BernoulliFlowLoss()`：adaptive BCE + focal + density loss，可扩展 stack / non-crossing loss。
- `project_to_valid_contact_map()`：推理后贪心投影，约束每个碱基最多配对一次。

当前 v0 loss：

```text
Loss = adaptive BCE + density_weight * density loss
```

配置里：

```text
stack_weight = 0.0
nc_weight = 0.0
density_weight = 0.2
```

也就是说 stack / non-crossing 代码已预留，但首版没打开。

### 2.3 `symfold/dit.py`

职责：实现简化版 SymFold 风格 Axial DiT 主干。

主要组件：

- `SinusoidalTimeEmbedding`：flow time `t` 的时间嵌入。
- `PatchEmbed2D`：把 `L×L` 特征 patchify。
- `AxialDiTBlock`：行注意力 + 列注意力 + FFN。
- `OutputRefineConv`：输出后在全分辨率 contact map 上做轻量卷积精修。
- `AxialDiT`：完整 DiT backbone。

当前 v0 不是完整 SymFold v5 的 DA-SE-DiT，而是简化版：

| 项 | 当前实现 |
| --- | --- |
| backbone | Axial DiT |
| row attention | 有 |
| column attention | 有 |
| time condition | AdaLN 风格调制 |
| patch embedding | 有 |
| output refine conv | 有 |
| density head | 有 |
| triangle update | 暂无 |
| UFold conditioner | 不使用 |
| RNA-FM attention map | 不使用 |

### 2.4 `symfold/model.py`

职责：把 PriFold MARS、embedding concat map、DiT 和 flow loss 连接起来。

核心流程：

```text
input_ids / attention_mask
  → MARS-LX extractor
  → hidden_states: [B,T,1056]
  → 去掉 <cls>/<eos>，得到 base hidden
  → Linear(1056 -> d_pair)
  → outer concat: [B,2*d_pair,S,S]
  → 拼接 seq_2d 和 pos_bias
  → 拼接 x_t embedding
  → AxialDiT
  → logit: [B,1,S,S]
  → BernoulliFlowLoss
```

当前默认：

```text
d_pair = 64
MARS frozen = true
DiT hidden_dim = 256
num_layers = 6
num_heads = 4
patch_size = 4
```

训练时：

```text
x_1 = GT contact map
随机 t ~ Uniform(0,1)
x_t ~ Bernoulli(t*x_1 + (1-t)*rho_0)
model(x_t, t, MARS condition) -> logit
loss(logit, x_1, t)
```

推理时：

```text
x_0 ~ Bernoulli(rho_0)
循环 num_steps 次 CTMC 采样
projection
输出 pred contact map
```

### 2.5 `symfold/train.py`

职责：训练入口。

功能：

- 读取 JSON config。
- 创建 log / output / model 目录。
- 加载 MARS-LX + tokenizer。
- 构建 train / val loader。
- 构建 `PriFoldSymFlowModel`。
- AdamW 训练。
- 每个 epoch：
  - 训练一个 epoch
  - val 采样评估
  - 保存 `history.json`
  - 保存 `last.pt`
  - 如果 val F1 提升，保存 `best.pt`
  - 保存 `epoch_XXX.pt`
- 写 heartbeat。

### 2.6 `symfold/eval.py`

职责：独立评估入口。

示例：

```bash
python symfold/eval.py \
  --ckpt symfold/outputs/20260526_1508_prifold_symflow_v0/model/best.pt \
  --test_sets bprna-test,rnastralign-test,archiveii-test \
  --num_steps 10 \
  --out_json symfold/outputs/20260526_1508_prifold_symflow_v0/eval_best.json
```

### 2.7 `symfold/run_train.sh`

职责：后台启动训练。

它会：

- 切到 PriFold 根目录。
- 读取 config 里的 `task_name`。
- 创建 `symfold/logs/<task>/`。
- 使用 `setsid` 后台运行。
- 分离 stdout / stderr。
- 写 PID 文件。

---

## 3. 整体训练流程

### 3.1 启动命令

本次启动命令：

```bash
cd /root/aigame/dannyyan/PriFold
bash symfold/run_train.sh /root/aigame/dannyyan/PriFold/symfold/config/prifold_symflow_v0.json
```

启动后日志：

```text
symfold/logs/20260526_1508_prifold_symflow_v0/
├── 20260526_1508_prifold_symflow_v0.log
├── 20260526_1508_prifold_symflow_v0.stdout.log
├── 20260526_1508_prifold_symflow_v0.stderr.log
├── 20260526_1508_prifold_symflow_v0.heartbeat
└── 20260526_1508_prifold_symflow_v0.pid
```

输出：

```text
symfold/outputs/20260526_1508_prifold_symflow_v0/
├── history.json
└── model/
    ├── best.pt
    ├── last.pt
    ├── epoch_001.pt
    ├── epoch_002.pt
    ├── epoch_003.pt
    ├── epoch_004.pt
    └── epoch_005.pt
```

### 3.2 当前训练配置

配置文件：`symfold/config/prifold_symflow_v0.json`

关键参数：

| 参数 | 当前值 |
| --- | ---: |
| `mars_scale` | `lx` |
| `freeze_mars` | `true` |
| `d_pair` | 64 |
| `hidden_dim` | 256 |
| `num_layers` | 6 |
| `patch_size` | 4 |
| `rho_0` | 0.005 |
| `use_pos_bias` | `true` |
| `density_weight` | 0.2 |
| `epochs` | 5 |
| `batch_size` | 1 |
| `lr` | `8e-5` |
| `max_train_samples` | 512 |
| `max_val_samples` | 128 |
| `num_steps` | 10 |

这是 smoke run，不是完整训练。为了快速验证链路，训练集和验证集都被截断了。

### 3.3 当前训练结果

本次训练已完成 5 个 epoch。

| epoch | train loss | val F1 | val Precision | val Recall | val MCC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0780 | 0.0228 | 0.0189 | 0.0291 | 0.0155 |
| 1 | 0.1541 | 0.0388 | 0.0332 | 0.0470 | 0.0318 |
| 2 | 0.0993 | 0.1090 | 0.0953 | 0.1279 | 0.1036 |
| 3 | 0.0524 | 0.1895 | 0.1681 | 0.2180 | 0.1853 |
| 4 | 0.0444 | 0.2464 | 0.2186 | 0.2837 | 0.2434 |

最终 best：

```text
best val F1 = 0.2464
checkpoint = symfold/outputs/20260526_1508_prifold_symflow_v0/model/best.pt
```

这个结果只说明链路能收敛，不能代表最终效果，因为当前只用了：

```text
train 512 samples
val 128 samples
5 epochs
num_steps 10
```

完整实验需要去掉 sample limit，并增加 epoch。

---

## 4. 本次遇到的问题与处理

### 4.1 MARS checkpoint 软链接损坏

第一次启动训练失败，错误：

```text
FileNotFoundError:
/root/aigame/dannyyan/PriFold/model/mars_run-encoder-mars-lx-train-val-d0.15-2023_10_05_22_03_21/ckpt_175000.pt
```

原因：

`model/mars.../ckpt_175000.pt` 是指向 HuggingFace cache blob 的软链接，但目标文件不存在，因此 `torch.load()` 找不到真实 checkpoint。

检查发现：

```text
model/mars.../ckpt_175000.pt -> /root/.cache/huggingface/.../blobs/...
```

但该 blob 不存在。

处理：

使用 `huggingface_hub.snapshot_download()` 下载 `yfish/PriFold` 中的模型文件，然后把 `model/` 下的软链接替换为真实文件：

```text
/root/aigame/dannyyan/PriFold/model/mars_run-encoder-mars-lx-train-val-d0.15-2023_10_05_22_03_21/ckpt_175000.pt
/root/aigame/dannyyan/PriFold/model/ss_model_bprna.pth
/root/aigame/dannyyan/PriFold/model/ss_model_rnastralign.pth
```

当前配置已改回：

```json
"pretrained_lm_dir": "/root/aigame/dannyyan/PriFold/model"
```

### 4.2 `chmod +x` 被拒绝

曾尝试对 `symfold/run_train.sh` 执行：

```bash
chmod +x symfold/run_train.sh
```

该命令被用户拒绝。因此后续没有依赖可执行权限，而是直接用：

```bash
bash symfold/run_train.sh ...
```

训练可以正常启动。

### 4.3 tokenizer / checkpoint warning

stderr 中有 warnings：

```text
EsmTokenizer.from_pretrained() with path to a single file is deprecated
```

以及：

```text
torch.utils.checkpoint: please pass in use_reentrant explicitly
None of the inputs have requires_grad=True. Gradients will be None
```

这些 warning 来自 PriFold 现有 MARS / tokenizer 实现，不影响本次 smoke training。因为当前配置冻结 MARS，`None of the inputs have requires_grad=True` 属于可接受现象。

### 4.4 当前模型是简化版，不是完整 SymFold v5

本次实现重点是先打通：

```text
PriFold data + MARS embedding + concat map + DiT + Flow Matching
```

暂未实现：

- SymFold v5 的 triangle update。
- UFold conditioner。
- RNA-FM attention map。
- full eval report / 可视化图。
- density-guided rate damping 的完整版本。
- multi-sample voting。

这些可以作为下一阶段增强。

---

## 5. 后续建议

### 5.1 跑完整训练

当前 smoke 配置里有：

```json
"max_train_samples": 512,
"max_val_samples": 128
```

完整训练时建议删除或设为 `null`，并把 epochs 提高，例如：

```json
"epochs": 100,
"max_train_samples": null,
"max_val_samples": null,
"num_steps": 20
```

### 5.2 添加 full eval

建议参考 SymFold 的 `full_eval` 机制，增加：

```text
bpRNA TS0
RNAStrAlign ts
ArchiveII
```

每隔若干 epoch 生成：

```text
full_eval_eXXX.json
FULL_EVAL_REPORT_eXXX.md
full_eval_f1_bar_eXXX.png
```

### 5.3 加强模型能力

建议按顺序尝试：

1. 打开 `stack_weight` / `nc_weight`。
2. 增加 `num_layers` 到 9。
3. 实现 triangle update。
4. 加入 density-guided sampling。
5. 开启 PriFold covariation augmentation。
6. 尝试部分解冻 MARS 或 LoRA。

### 5.4 做对照实验

至少比较：

| 实验 | 说明 |
| --- | --- |
| PriFold 原模型 | 当前 baseline。 |
| SymFlow v0 smoke | 当前实现。 |
| SymFlow full-data | 去掉 sample limit 的完整训练。 |
| SymFlow + physics loss | 打开 stack / nc。 |
| SymFlow + augmentation | 打开 PriFold covariation augmentation。 |

---

## 6. 当前可用命令

### 6.1 启动训练

```bash
cd /root/aigame/dannyyan/PriFold
bash symfold/run_train.sh symfold/config/prifold_symflow_v0.json
```

### 6.2 查看日志

```bash
tail -f symfold/logs/20260526_1508_prifold_symflow_v0/20260526_1508_prifold_symflow_v0.log
```

### 6.3 查看 history

```bash
cat symfold/outputs/20260526_1508_prifold_symflow_v0/history.json
```

### 6.4 独立评估

```bash
python symfold/eval.py \
  --ckpt symfold/outputs/20260526_1508_prifold_symflow_v0/model/best.pt \
  --test_sets bprna-test,rnastralign-test,archiveii-test \
  --num_steps 10 \
  --out_json symfold/outputs/20260526_1508_prifold_symflow_v0/eval_best.json
```
