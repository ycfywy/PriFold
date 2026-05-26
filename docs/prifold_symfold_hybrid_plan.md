# PriFold × SymFold 混合模型方案

目标：在 `PriFold` 项目中实现一版新模型，**数据处理和预训练语言模型来自 PriFold**，**主干思路来自 SymFold 的 DiT + Flow Matching**，用于验证“PriFold MARS embedding + concat map + SymFold flow backbone”在 RNA 二级结构预测上的效果。

---

## 1. 背景与目标

### 1.1 当前 PriFold

PriFold 当前流程：

```text
RNA seq
  → U 替换为 T
  → EsmTokenizer
  → MARS 预训练语言模型
  → hidden_states: [B, T, 1056]
  → PairwiseOnly 外积拼接: [B, T, T, 256]
  → RNAformerStack
  → logits/contact map
```

当前优点：

- 已经有稳定的 MARS 预训练模型加载逻辑。
- 已经有 bpRNA / RNAStrAlign / ArchiveII 的 CSV + `*.npy` 数据处理方式。
- 已有明确的训练、推理和评估指标。

当前限制：

- 预测是直接二分类，即 `BCEWithLogitsLoss(logits, ct)`。
- 没有采用 SymFold 那种从随机稀疏图逐步生成 contact map 的 flow matching 思路。

### 1.2 SymFold 可借鉴点

SymFold 将 RNA contact map 视为 **对称二值矩阵生成问题**：

```text
RNA sequence
  → conditioners
  → DA-SE-DiT predicts P(x1 | xt, t)
  → CTMC / τ-leap sampling
  → strict projection
  → valid contact map
```

其关键点：

1. **Bernoulli Discrete Flow Matching**：从稀疏 Bernoulli 初始图 `x_t` 逐步学习到真实 contact map `x_1`。
2. **DiT 主干**：使用带时间条件的 Transformer 主干处理 `L×L` 图。
3. **embedding + outer concat map**：把一维序列 embedding 扩展成二维配对特征。
4. **对称约束和 projection**：输出对称化，并在采样后投影到更合法的 RNA contact map。
5. **训练/日志/results 规范**：有 `history.json`、`best.pt`、`last.pt`、full eval JSON/MD/PNG 报告。

### 1.3 本方案目标

实现一个新分支，暂命名为：

```text
PriFold-SymFlow
```

核心要求：

1. 使用 PriFold 的 MARS 预训练模型，不使用 SymFold 的 RNA-FM。
2. 使用 MARS hidden states 生成 `embedding + concat map`。
3. 主干采用 SymFold 风格的 DiT / DA-SE-DiT。
4. 训练目标采用 Bernoulli Flow Matching。
5. 数据处理以 PriFold 的 CSV + `*.npy` 为基础，同时训练/验证/测试组织、日志和结果记录参考 SymFold。

---

## 2. 总体设计

### 2.1 新模型整体流程

```text
PriFold CSV + npy contact map
  → PriFold Dataset
  → seq: str, ct: [L, L]
  → U -> T
  → EsmTokenizer + MARS
  → hidden_states: [B, T, 1056]
  → 去掉 tokenizer 特殊 token，得到 base embedding: [B, L, 1056]
  → pad 到 S，S = ceil(L / patch_size) * patch_size
  → Linear(1056 -> D)
  → outer concat map: [B, 2D, S, S]
  → Flow noising: x1 -> xt
  → DiT input = concat(x_t embedding, MARS concat map, seq_2d, pos_bias optional)
  → DA-SE-DiT
  → logits: [B, 1, S, S]
  → Bernoulli Flow Matching loss
  → CTMC sampling + projection
  → pred contact map: [B, 1, S, S]
  → crop 回 [L, L] 计算指标
```

### 2.2 和 PriFold / SymFold 的关系

| 模块 | 采用来源 | 说明 |
| --- | --- | --- |
| 数据 CSV 读取 | PriFold | 使用 `data/bprna/bpRNA.csv`、`data/RNAStrAlign/rnastralign.csv`、`data/archiveII/archiveII.csv`。 |
| contact map 标签 | PriFold | 使用现有 `*.npy` contact map。 |
| 长度过滤 | PriFold | 默认保留 `seq` 长度 `< 490`。 |
| `U -> T` | PriFold | 保持和 MARS tokenizer 一致。 |
| 预训练模型 | PriFold | 使用 `utils/lm.py:get_extractor()` 加载 MARS-LX。 |
| embedding + concat map | PriFold/SymFold 结合 | 用 MARS hidden states 做 outer concat，替代 SymFold RNA-FM/UFold 条件。 |
| 主干网络 | SymFold | 采用 DA-SE-DiT / DiT blocks。 |
| 训练目标 | SymFold | Bernoulli Discrete Flow Matching。 |
| 采样 | SymFold | CTMC τ-leap sampling + projection。 |
| 日志和结果 | SymFold | `logs/<task>/`、`outputs/<task>/history.json`、`full_eval`、`best.pt`、`last.pt`。 |

---

## 3. 数据处理方案

### 3.1 使用的数据

第一版只使用 PriFold 当前项目已有数据：

| 数据集 | CSV | contact map |
| --- | --- | --- |
| bpRNA | `data/bprna/bpRNA.csv` | `data/bprna/ct/{TR0,VL0,TS0}/{file_name}.npy` |
| RNAStrAlign | `data/RNAStrAlign/rnastralign.csv` | `data/RNAStrAlign/{file_name}.npy` |
| ArchiveII | `data/archiveII/archiveII.csv` | `data/archiveII/ct/{file_name}.npy` |

当前过滤规则：

```text
seq 长度 < 490
```

过滤后数量：

| dataset | raw_n | filtered_n | removed_n |
| --- | ---: | ---: | ---: |
| bpRNA | 13419 | 13409 | 10 |
| RNAStrAlign | 26078 | 25219 | 859 |
| ArchiveII | 3966 | 3845 | 121 |

### 3.2 推荐训练/验证/测试划分

参考 SymFold 的“standard”思路：训练集同时使用 bpRNA 和 RNAStrAlign，验证集也同时使用两者，测试集覆盖 ID/OOD。

建议第一版划分：

| split | 数据 |
| --- | --- |
| train | bpRNA `TR0` + RNAStrAlign `tr` |
| val | bpRNA `VL0` + RNAStrAlign `ts` 或 `vl` |
| test-id | bpRNA `TS0` + RNAStrAlign `ts` |
| test-ood | ArchiveII 全部 |

注意：PriFold 当前 `utils/tools.py` 中，RNAStrAlign 训练模式把 `ts` 当验证集，`vl` 没有进入主流程。为了与现有 PriFold 结果对齐，第一版建议继续使用：

```text
RNAStrAlign tr → train
RNAStrAlign ts → val/test
RNAStrAlign vl → 暂不使用，或作为额外 val_ablation
```

后续如果想更贴近 SymFold，可改成：

```text
RNAStrAlign tr → train
RNAStrAlign vl → val
RNAStrAlign ts → test
```

但这会和当前 PriFold baseline 的划分不完全一致，需要单独标注。

### 3.3 Dataset 输出格式

建议新增 dataset，例如：

```text
utils/symflow_data.py
```

单样本输出：

```python
{
  "seq": str,              # U 已替换为 T
  "ct": FloatTensor[L,L],  # 原始 contact map
  "length": int,           # L
  "name": str,
  "dataset": str,
  "split": str,
}
```

### 3.4 Collate 与 padding

SymFold 的 DiT patch embedding 要求二维矩阵尺寸能被 `patch_size` 整除。PriFold 当前 `max_len = len(seq)+2`，不一定能被 4 整除。因此新模型建议：

```text
L = 原始 RNA 长度
S = ceil(L / patch_size) * patch_size
```

注意：这里建议 **DiT 只处理碱基本身，不处理 `<cls>/<eos>` 特殊 token**。原因：contact map 是 `[L,L]`，特殊 token 没有结构标签。

Batch 输出：

| key | shape | 说明 |
| --- | --- | --- |
| `input_ids` | `[B, T]` | MARS tokenizer 输入，含特殊 token。 |
| `attention_mask` | `[B, T]` | MARS attention mask。 |
| `seq` | `list[str]` | U->T 后的序列。 |
| `seq_oh` | `[B, S, 4]` | A/T/G/C one-hot，padding 到 S。 |
| `contact` | `[B, 1, S, S]` | contact map，padding 到 S。 |
| `contact_mask` | `[B, 1, S, S]` | 有效区域 mask。 |
| `length` | `[B]` | 原始长度 L。 |
| `set_max_len` | `int` | 当前 batch 的 S。 |
| `name` | `list[str]` | 样本名。 |

### 3.5 是否保留 PriFold 数据增强

建议分两阶段：

- **v0**：不启用 covariation augmentation，只验证 MARS + DiT + FM 是否能收敛。
- **v1**：启用 PriFold `Augmentation(select, replace)`，与 PriFold 默认训练配置对齐，例如 `select=0.1, replace=0.3`。

原因：Flow Matching 本身有随机 `x_t` noising，如果同时启用强数据增强，初期不利于定位问题。

---

## 4. 模型结构方案

### 4.1 新模型命名

建议新增：

```text
utils/symflow/
  __init__.py
  model.py
  dit.py
  discrete_flow.py
  data.py
  train.py
  eval.py
  config/prifold_symflow_v0.json
```

或者如果希望更贴近 PriFold 顶层风格：

```text
symflow/
  model.py
  dit.py
  discrete_flow.py
  data.py
train_symflow.py
eval_symflow.py
```

推荐使用第二种，避免塞进 `utils/` 太多业务代码。

### 4.2 MARS encoder

复用 PriFold：

```python
extractor, tokenizer = get_extractor(args)
output = extractor(tokens=input_ids, attn_mask=attention_mask)
hidden = output[1]  # [B, T, 1056]
```

然后去掉特殊 token：

```python
base_hidden = hidden[:, 1:1+L, :]  # [B, L, 1056]
```

再 padding 到 `S`：

```text
base_hidden_pad: [B, S, 1056]
```

冻结策略：

| 版本 | MARS 是否训练 | 说明 |
| --- | --- | --- |
| v0 | frozen | 显存低、训练稳定，先验证 DiT + Flow。 |
| v1 | LoRA 或部分 finetune | 后续再做。 |
| v2 | full finetune | 显存压力大，不建议第一版。 |

### 4.3 Embedding + concat map

这是用户明确要求的核心点。

设计：

```text
base_hidden_pad: [B, S, 1056]
  → Linear(1056 -> d_pair)
  → h: [B, S, d_pair]
  → h_i expand: [B, S, S, d_pair]
  → h_j expand: [B, S, S, d_pair]
  → concat([h_i, h_j]): [B, S, S, 2*d_pair]
  → permute: [B, 2*d_pair, S, S]
```

推荐参数：

```text
d_pair = 128
concat map channels = 256
```

如果显存压力较大，可以先用：

```text
d_pair = 64
concat map channels = 128
```

### 4.4 DiT 输入特征

借鉴 SymFold v5 的 `_build_features()`，但替换 conditioner：

SymFold v5：

```text
x_t embedding + RNA-FM 2D + RNA-FM attention + seq_2d + UFold condition
```

PriFold-SymFlow v0：

```text
x_t embedding + MARS concat map + seq_2d + pos_bias(optional)
```

建议输入通道：

| 分支 | shape | 说明 |
| --- | --- | --- |
| `x_t_embedding` | `[B, 8, S, S]` | 当前 noised contact map 的 0/1 embedding。 |
| `mars_concat_map` | `[B, 128 or 256, S, S]` | MARS embedding outer concat。 |
| `seq_2d` | `[B, 8, S, S]` | one-hot 的 outer concat，4+4。 |
| `pos_bias` | `[B, 1, S, S]` | 可选，PriFold 生物配对先验。 |
| `contact_mask` | `[B, 1, S, S]` | 不作为输入通道，但用于 mask loss/logit。 |

v0 推荐输入：

```text
in_channels = 8 + 128 + 8 + 1 = 145
```

如果用 `d_pair=128`：

```text
in_channels = 8 + 256 + 8 + 1 = 273
```

为降低首版显存和调试难度，建议 v0 用 `d_pair=64`。

### 4.5 DiT 主干

参考 SymFold v5，但做简化：

| 参数 | SymFold v5 | PriFold-SymFlow v0 建议 |
| --- | ---: | ---: |
| `hidden_dim` | 256 | 256 |
| `num_heads` | 4 | 4 |
| `dim_head` | 64 | 64 |
| `num_layers` | 9 | 6 或 9 |
| `patch_size` | 4 | 4 |
| `max_len` | 640 | 492 或 512 |
| `dilation_pattern` | `[1,1,1,2,2,2,4,4,4]` | 6 层用 `[1,1,2,2,4,4]`；9 层沿用 SymFold |
| `triangle update` | layer >= 6 | v0 可先关闭，v1 打开 |
| `density head` | 有 | v0 可保留 |
| `output refine conv` | 有 | 建议保留 |

建议第一版：

```text
hidden_dim=256
num_layers=6
patch_size=4
d_pair=64
triangle_update=False
output_refine=True
density_head=True
```

原因：先让训练跑通，再逐步加 SymFold v5 的高级组件。

### 4.6 Flow Matching 训练目标

复用 SymFold v4/v5 逻辑：

#### Forward noising

```text
x_1 = ground truth contact map
rho_0 = 0.005
t ~ Uniform(0, 1)
p_t = t * x_1 + (1 - t) * rho_0
x_t ~ Bernoulli(p_t)
x_t = symmetrize(x_t) * contact_mask
```

模型预测：

```text
logit = model(x_t, t, condition)
P(x_1=1 | x_t, t, condition) = sigmoid(logit)
```

#### Loss

建议 v0：

```text
Loss = adaptive BCE + λ_density * density loss
```

建议 v1 加：

```text
Loss = adaptive BCE + λ_stack * stacking loss + λ_nc * non-crossing loss + λ_density * density loss
```

默认参数参考 SymFold v5：

```text
rho_0 = 0.005
pos_weight_base = 199.0
pos_weight_min = 20.0
focal_gamma = 1.5
stack_weight = 0.05
nc_weight = 0.02
density_weight = 0.2
```

但由于 PriFold 数据长度 `<490`、density 与 SymFold 数据分布不完全一致，建议初始实验记录两套：

| 实验 | loss 配置 |
| --- | --- |
| v0-basic | BCE + time weight |
| v0-density | adaptive BCE + density loss |
| v1-physics | adaptive BCE + density + stack + nc |

### 4.7 采样和 projection

推理时参考 SymFold：

```text
x_0 ~ Bernoulli(rho_0)
for k in num_steps:
  logit = model(x_t, t_k, condition)
  p_x1 = sigmoid(logit)
  compute CTMC rates
  sample flips 0->1 / 1->0
  symmetrize
project_to_valid_contact_map
crop to [L,L]
calculate metrics
```

默认：

```text
num_steps = 20
num_samples_per_input = 1
```

后续可加 multi-sample voting：

```text
num_samples_per_input = 5 or 8
```

Projection 约束：

1. 对称。
2. `|i-j| >= 3`。
3. 每个碱基最多配对一次。
4. 按预测概率贪心选择配对。

---

## 5. 训练与测试方案

### 5.1 训练任务设计

建议新增配置：

```text
config/prifold_symflow_v0.json
```

示例：

```json
{
  "project_name": "prifold_symflow",
  "task_name": "YYYYMMDD_HHMM_prifold_symflow_v0",
  "seed": 3407,
  "device": "cuda:0",
  "model": {
    "version": "v0",
    "mars_scale": "lx",
    "freeze_mars": true,
    "d_pair": 64,
    "hidden_dim": 256,
    "num_heads": 4,
    "dim_head": 64,
    "num_layers": 6,
    "patch_size": 4,
    "rho_0": 0.005,
    "pos_weight_base": 199.0,
    "pos_weight_min": 20.0,
    "focal_gamma": 1.5,
    "density_weight": 0.2,
    "use_pos_bias": true,
    "use_output_refine": true
  },
  "training": {
    "epochs": 100,
    "lr": 8e-5,
    "warmup_epochs": 5,
    "eval_every": 2,
    "full_eval_every": 10,
    "save_every": 5,
    "patience": 20,
    "num_workers": 4,
    "max_len_filter": 490,
    "bucket_by_length": true,
    "grad_clip": 1.0,
    "augmentation": {
      "enabled": false,
      "select": 0.1,
      "replace": 0.3
    }
  },
  "sampling": {
    "num_steps": 20,
    "num_samples_per_input": 1
  },
  "paths": {
    "data_dir": "./data",
    "pretrained_lm_dir": "./model",
    "model_save_dir": "outputs/YYYYMMDD_HHMM_prifold_symflow_v0/model",
    "log_dir": "logs/YYYYMMDD_HHMM_prifold_symflow_v0",
    "output_dir": "outputs/YYYYMMDD_HHMM_prifold_symflow_v0"
  }
}
```

### 5.2 训练脚本

建议新增：

```text
train_symflow.py
```

流程参考 SymFold `train_v5.py`：

1. 读取 JSON config。
2. 创建 `log_dir`、`output_dir`、`model_dir`。
3. 初始化 logging、heartbeat、signal handler。
4. 加载 MARS + tokenizer。
5. 构建 PriFold-SymFlow 模型。
6. 构建 train/val dataloader。
7. 每个 epoch：
   - train one epoch
   - 写 heartbeat
   - 保存 `last.pt`
   - 每 `eval_every` 在 val 上采样评估
   - 如果 val F1 提升，保存 `best.pt`
   - 写 `history.json`
   - 画 `curves.png`
   - 每 `full_eval_every` 跑完整测试集并生成 JSON/MD/PNG 报告。

### 5.3 测试脚本

建议新增：

```text
eval_symflow.py
```

参考 SymFold `eval/eval.py`，支持：

```bash
python eval_symflow.py \
  --ckpt outputs/<task>/model/best.pt \
  --test_sets bpRNA,RNAStrAlign,ArchiveII \
  --num_steps 20 \
  --num_samples 1 \
  --out_json outputs/<task>/eval_best.json \
  --detailed
```

测试集映射建议：

| name | 数据 |
| --- | --- |
| `bpRNA` | bpRNA `TS0` |
| `RNAStrAlign` | RNAStrAlign `ts` |
| `ArchiveII` | ArchiveII 全部 |
| `bpRNA_VL0` | bpRNA `VL0`，验证集检查 |
| `RNAStrAlign_vl` | RNAStrAlign `vl`，可选额外验证 |

### 5.4 指标

沿用 SymFold / PriFold：

| 指标 | 说明 |
| --- | --- |
| Precision | 预测配对中正确比例。 |
| Recall | 真实配对中被预测出来的比例。 |
| F1 | 主要指标。 |
| MCC | 建议加入，SymFold 使用。 |
| pair count | GT / Pred 配对数量，用于分析过预测/欠预测。 |

---

## 6. 日志与 results 记录方案

参考 SymFold，建议每次实验按 task 建目录。

### 6.1 目录结构

```text
logs/<task>/
  <task>.log
  <task>.heartbeat
  <task>.stderr.log      # 如果使用后台脚本启动

outputs/<task>/
  history.json
  curves.png
  model/
    last.pt
    best.pt
    epoch_005.pt
    epoch_010.pt
  val_vis/
    vis_e*.png
  full_eval/
    full_eval_history.json
    full_eval_f1_trend.png
    e010/
      full_eval_e010.json
      FULL_EVAL_REPORT_e010.md
      full_eval_f1_bar_e010.png
      vis/*.png
  eval_best.json
```

### 6.2 `history.json` 内容

每个 epoch 一条：

```json
{
  "epoch": 0,
  "loss": 0.123,
  "bce": 0.100,
  "density": 0.010,
  "stack": 0.005,
  "nc": 0.002,
  "time_s": 1234.5,
  "lr": 0.00008,
  "val_f1": 0.75,
  "val_precision": 0.76,
  "val_recall": 0.74,
  "val_mcc": 0.73,
  "full_eval_avg_f1": 0.70,
  "full_eval_bpRNA_f1": 0.65,
  "full_eval_RNAStrAlign_f1": 0.90,
  "full_eval_ArchiveII_f1": 0.82
}
```

### 6.3 完整评估报告

每次 full eval 生成：

```text
FULL_EVAL_REPORT_eXXX.md
```

内容包括：

1. checkpoint 信息。
2. sampling steps / num samples。
3. 各数据集 F1 / Precision / Recall / MCC。
4. Top worst / Top best 样本。
5. GT pairs / Pred pairs 对比。
6. 可视化图片路径。

### 6.4 启动脚本

建议新增：

```text
scripts/run_symflow_train.sh
scripts/run_symflow_eval.sh
```

训练脚本参考 SymFold 的 `setsid + stdout/stderr 分离`：

```bash
setsid bash -c "
  source /root/aigame/dannyyan/miniconda3/bin/activate prifold
  cd /root/aigame/dannyyan/PriFold
  export PYTHONPATH=.
  exec python -u train_symflow.py config/prifold_symflow_v0.json
" < /dev/null \
  > logs/<task>/<task>.stdout.log \
  2> logs/<task>/<task>.stderr.log &
```

---

## 7. 实现步骤

### Phase 0：最小可跑版本

目标：确认 MARS embedding + concat map + DiT + flow loss 可以跑通。

任务：

1. 新增 `symflow/discrete_flow.py`：复制/改造 SymFold v4 的 `sample_x_t_given_x_1`、`symmetrize_binary`、`compute_ctmc_rates`、`project_to_valid_contact_map`、loss。
2. 新增 `symflow/dit.py`：实现简化版 DiT。
3. 新增 `symflow/model.py`：包装 MARS encoder + concat map + DiT + flow loss + sample。
4. 新增 `symflow/data.py`：读取 PriFold CSV/NPY，输出 flow 需要的 batch。
5. 新增 `train_symflow.py`：支持训练 1 epoch、保存 `last.pt`、跑 val。
6. 新增 `eval_symflow.py`：支持 bpRNA/RNAStrAlign/ArchiveII 单次评估。

验收：

```text
能在小 subset 上完成 1 epoch
loss 不是 NaN
eval 能输出 F1/P/R/MCC
能保存 best.pt、last.pt、history.json
```

### Phase 1：完整训练版本

目标：跑完整 bpRNA + RNAStrAlign 训练。

任务：

1. 加入 length bucket batch，降低 padding 浪费。
2. 加入 heartbeat 和 signal handler。
3. 加入 `curves.png`。
4. 加入 full eval：bpRNA / RNAStrAlign / ArchiveII。
5. 加入可视化：GT / Pred / overlay。
6. 加入 auto resume。

验收：

```text
能稳定训练 100 epoch
每 2 epoch 有 val 指标
每 10 epoch 有 full_eval report
中断后可从 last.pt 恢复
```

### Phase 2：增强版本

目标：加入更接近 SymFold v5 的能力。

可选增强：

1. 打开 triangle update。
2. 加入 density head + density-guided sampling。
3. 加入 stack / non-crossing physics loss。
4. 加入 PriFold covariation augmentation。
5. 尝试 MARS partial finetune 或 LoRA。
6. 尝试 multi-sample voting。

---

## 8. 对照实验设计

为了看清每个组件的作用，建议实验表如下：

| 实验名 | MARS | concat map | DiT | Flow | pos_bias | augmentation | physics loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-prifold | yes | PairwiseOnly | RNAformer | no | yes | yes | no |
| symflow-v0 | frozen | yes | 6L DiT | yes | no | no | no |
| symflow-v0-posbias | frozen | yes | 6L DiT | yes | yes | no | no |
| symflow-v1-density | frozen | yes | 6L DiT | yes | yes | no | density |
| symflow-v1-aug | frozen | yes | 6L DiT | yes | yes | yes | density |
| symflow-v2-full | frozen/partial | yes | 9L DA-SE-DiT | yes | yes | yes | density+stack+nc |

主要对比指标：

| 测试集 | 当前 PriFold 参考 F1 | 新模型目标 |
| --- | ---: | ---: |
| bpRNA TS0 | 0.7700 | 先接近，再尝试超过 |
| RNAStrAlign ts | 0.9738 | 尽量接近，防止 flow 退化 |
| ArchiveII | 0.9043 | 重点看 OOD 泛化 |

---

## 9. 风险与注意事项

### 9.1 特殊 token 与 contact map 对齐

PriFold 当前模型把 tokenizer 后的特殊 token 也放进 `T×T` 矩阵中，而原始 contact map 是 `L×L`。新 DiT 建议只在 base positions 上做 flow，即：

```text
MARS hidden: [B, T, 1056]
base hidden: hidden[:, 1:1+L, :]
contact: [B, L, L]
```

这样最清晰，避免 `<cls>/<eos>` 对 flow label 造成干扰。

### 9.2 patch size padding

DiT patch size 通常为 4，因此 `S` 必须能被 4 整除。需要保证：

```text
contact / mask / seq_oh / embedding 全部 pad 到同一个 S
最终评估 crop 回 L
```

### 9.3 显存

`MARS-LX` hidden dim 是 1056，outer concat 会产生较大 `S×S` 特征。

建议：

1. v0 冻结 MARS。
2. v0 使用 `d_pair=64`。
3. batch size 先从 1 或按长度 bucket 开始。
4. 长序列 bucket 单独小 batch。

### 9.4 Flow 训练不一定比直接 BCE 更快收敛

Flow Matching 是生成式目标，初期可能比 PriFold 直接监督更慢。要用 `history.json` 记录：

- train loss
- val F1
- pred pairs / gt pairs
- density error

避免只看 loss。

### 9.5 数据划分要标清楚

PriFold 和 SymFold 对 RNAStrAlign 的 `val/test` 定义可能不同。每个实验必须在 config 和报告中写明：

```text
RNAStrAlign tr / ts / vl 分别怎么用
```

---

## 10. 推荐首版最小方案

我建议第一版不要一次性复刻 SymFold v5 全部功能，而是做一个可控的 v0：

```text
PriFold MARS-LX frozen
+ MARS base embedding outer concat map
+ x_t embedding
+ seq_2d
+ optional pos_bias channel
+ 6-layer DiT
+ Bernoulli Flow Matching
+ CTMC sampling
+ greedy projection
+ SymFold-style logs/results
```

第一版训练：

```text
train = bpRNA TR0 + RNAStrAlign tr
val   = bpRNA VL0 + RNAStrAlign ts
test  = bpRNA TS0 + RNAStrAlign ts + ArchiveII
```

第一版目标：

1. 确认训练稳定。
2. 确认采样后 contact map 合法。
3. 在 ArchiveII 上看是否有 OOD 泛化提升。
4. 与当前 PriFold 三个结果做横向对比。

---

## 11. 预期新增文件清单

```text
symflow/
  __init__.py
  data.py                 # PriFold CSV/NPY dataset + bucket collate
  model.py                # MARS encoder + concat map + DiT + flow wrapper
  dit.py                  # DiT/DA-SE-DiT backbone
  discrete_flow.py        # Bernoulli FM, CTMC sampling, projection, losses
  metrics.py              # Precision/Recall/F1/MCC + pair count
  visualize.py            # GT/Pred/overlay visualization

config/
  prifold_symflow_v0.json

scripts/
  run_symflow_train.sh
  run_symflow_eval.sh

train_symflow.py

eval_symflow.py
```

---

## 12. 最终产出

完成后，每个实验应至少产出：

```text
logs/<task>/<task>.log
logs/<task>/<task>.heartbeat
outputs/<task>/history.json
outputs/<task>/curves.png
outputs/<task>/model/last.pt
outputs/<task>/model/best.pt
outputs/<task>/full_eval/full_eval_history.json
outputs/<task>/full_eval/eXXX/full_eval_eXXX.json
outputs/<task>/full_eval/eXXX/FULL_EVAL_REPORT_eXXX.md
outputs/<task>/eval_best.json
```

这样既能复用 PriFold 的数据和预训练能力，也能系统评估 SymFold 风格 flow matching 主干是否带来收益。
