# MARS 模型与 PriFold v9 架构 Shape Walkthrough

> 目的：从代码实现出发，说明 PriFold 中 MARS 模型的作用、v9 版本模型架构，以及一条 RNA sequence 在 v9 中如何按张量 shape 流动到最终 contact map。

---

## 1. MARS 模型在项目里的定位

MARS 是 PriFold 使用的 RNA 基础语言模型。它负责把 RNA 序列编码成：

1. **1D token hidden states**：每个碱基一个上下文表征；
2. **2D self-attention maps**：不同层、不同 head 的 token-token attention，可直接作为 pairwise 特征来源。

v9 使用的是 `mars_scale="lx"`，配置在 `symfold/config/v9/v9_ddp.json`：

```json
"model": {
  "mars_scale": "lx",
  "freeze_mars": true,
  "mars_dim": 1056,
  "mars_n_attn_layers": 6,
  "mars_n_heads": 12,
  "mars_hidden_layer_indices": [3, 6, 9, 12]
}
```

关键含义：

| 项 | v9 设置 | 说明 |
|---|---:|---|
| MARS 版本 | `lx` | 映射到 `mars_run-encoder-mars-lx-train-val-d0.15-2023_10_05_22_03_21` |
| hidden dim | `1056` | MARS-LX / 160M 级别 hidden size |
| Transformer 层数 | `12` | MARS-LX 级别配置 |
| attention heads | `12` | 每层 12 个 self-attention head |
| 导出 attention 层数 | `6` | v9 取最后 6 层 attention map |
| MARS 是否训练 | `freeze_mars=true` | v9 冻结 MARS，只训练下游结构预测头 |

---

## 2. MARS 是如何加载的

加载入口是 `utils/lm.py:get_extractor()`：

```text
utils/lm.py
get_extractor(args)
  ├─ 根据 args.model_scale 选择模型目录
  ├─ lx -> mars_run-encoder-mars-lx-train-val-d0.15-2023_10_05_22_03_21
  ├─ ckpt_path = <pretrained_lm_dir>/<model>/ckpt_175000.pt
  ├─ extractor = prifold.llama2.load_model(ckpt_path, device='cpu')
  └─ tokenizer = EsmTokenizer.from_pretrained("vocab_esm_mars.txt")
```

对应 v9 配置中：

```json
"paths": {
  "pretrained_lm_dir": "/root/aigame/dannyyan/PriFold/model"
}
```

所以 v9 实际加载路径为：

```text
/root/aigame/dannyyan/PriFold/model/mars_run-encoder-mars-lx-train-val-d0.15-2023_10_05_22_03_21/ckpt_175000.pt
```

MARS 主体实现是 `prifold/llama2.py`：

- `ModelArgs` 定义基础配置：`dim`、`n_layers`、`n_heads`、`vocab_size`、`max_seq_len` 等；
- `Transformer` 是 LLaMA2 风格 Transformer encoder/decoder；
- `Attention` 使用 Q/K/V projection、RoPE、scaled dot-product attention；
- `load_model()` 从 checkpoint 读取 `model_args` 并加载权重。

MARS-LX 对应 160M 级别配置：

```text
dim=1056, n_layers=12, n_heads=12, n_kv_heads=12
head_dim = 1056 / 12 = 88
```

---

## 3. 为什么需要 `llama2_with_attn.py`

原始 `prifold/llama2.py:Attention.forward()` 默认走 PyTorch 的 `scaled_dot_product_attention`，这种路径高效，但不会返回 attention weights。

v9 需要把 MARS 的 attention map 当作 RNA 二级结构的 2D pair feature，所以项目增加了 wrapper：

```text
prifold/llama2_with_attn.py
mars_forward_with_attn(model, tokens, attn_mask, n_attn_layers=6, ...)
```

核心逻辑：

1. 前面的层仍走原始 flash / SDPA 路径，保证速度；
2. 最后 `n_attn_layers=6` 层改用手写 attention：
   - `scores = q @ k^T / sqrt(head_dim)`
   - `attn = softmax(scores)`
   - `output = attn @ v`
3. 返回：

```text
hidden:     (B, T, 1056)
attn_stack: (B, 6, 12, T, T)
```

其中：

- `B`：batch size；
- `T`：tokenized length，包含 special tokens，通常为 `max_raw_len + 2`；
- `6`：最后 6 层；
- `12`：每层 12 个 heads。

注意：v9 传入了 `mars_hidden_layer_indices=[3,6,9,12]`，wrapper 会收集这些中间层 hidden，但 v9 当前下游只使用最终 `hidden` 和 `attn_stack`，没有进一步使用 `hidden_layers` 做多层融合。

---

## 4. v9 模型总体架构

v9 主模型类是：

```text
symfold/v9/model.py
DensityNetProPlus
```

代码顶部给出的架构是：

```text
RNA seq
  → MARS-LX (frozen, 160M)
  → 1D hidden + 2D attention
  → Pair Feature Construction (outer prod + attn + seq_pair)
  → 2D RoPE Positional Encoding
  → Axial Transformer Stack (8 layers, DropPath=0.15, Dropout=0.2)
  → Contact Logit + Density Prediction
```

v9 的关键设计：

| 模块 | 代码位置 | 作用 |
|---|---|---|
| MARS 提取 | `DensityNetProPlus._extract_mars()` | 从 MARS 得到 1D hidden 与 2D attention |
| Pair 特征构造 | `DensityNetProPlus._build_pair_features()` | 把 1D/2D/碱基组合特征拼成 `(L,L,D)` |
| 2D RoPE | `RotaryPositionEmbedding2D` | 给 row/col axial attention 注入相对位置信息 |
| Axial Block | `AxialAttentionBlock` | row attention + column attention + FFN |
| Contact head | `contact_head` | 输出每个 `(i,j)` 是否配对的 logit |
| Density head | `density_head` | 预测每条 RNA 的配对密度，用于推理预算 |

v9 配置：

```json
"v9": {
  "hidden_dim": 192,
  "num_layers": 8,
  "num_heads": 6,
  "dim_head": 32,
  "ff_mult": 4,
  "dropout": 0.2,
  "drop_path": 0.15,
  "use_rope": true
}
```

---

## 5. 从 sequence 到 batch 的 shape

数据流水线在 `symfold/data.py`。

假设一个 batch 中最长原始序列长度为：

```text
max_l = max(len(seq_i))
```

`make_collate_fn()` 会计算：

```python
set_len = ceil(max_l / patch_size) * patch_size
```

v9 配置里没有显式设置 `patch_size`，因此训练脚本调用 `make_collate_fn(tokenizer)` 时使用默认 `patch_size=4`。这里的 `patch_size=4` 只影响 padding 到 4 的倍数，v9 主干本身不是 patch 模型。

collate 输出的核心字段：

| 字段 | shape | 说明 |
|---|---|---|
| `input_ids` | `(B, max_l + 2)` | tokenizer 输出，包含 special tokens |
| `attention_mask` | `(B, max_l + 2)` | MARS padding mask |
| `seq_oh` | `(B, L, 4)` | A/T/G/C one-hot，`L=set_len` |
| `contact` | `(B, 1, L, L)` | GT contact map，padding 后 |
| `contact_mask` | `(B, 1, L, L)` | 有效区域 mask |
| `pos_bias` | `(B, L, L)` | v9 当前不使用；位置由 RoPE 提供 |
| `length` | `(B,)` | 原始序列长度 |

---

## 6. Shape walkthrough：单条 seq 如何跑完整个 v9

下面用符号说明：

```text
B = batch size
S = 原始最长序列长度 max_l
T = token length = S + 2
L = set_len = ceil(S / 4) * 4
D_mars = 1056
D = hidden_dim = 192
H_mars = 12
N_attn = 6
H = v9 axial heads = 6
D_head = 32
```

### 6.1 Tokenize + padding

输入 RNA：

```text
seqs: List[str], 每条长度 <= S
```

经过 tokenizer 和 collate：

```text
input_ids      : (B, T)
attention_mask : (B, T)
seq_oh         : (B, L, 4)
contact        : (B, 1, L, L)
contact_mask   : (B, 1, L, L)
```

其中 `T=S+2` 是因为 tokenizer 加 special tokens；`L` 是 contact map / pair feature 的 padding 长度。

### 6.2 MARS forward

v9 调用：

```python
hidden, attn_stack, hidden_layers = mars_forward_with_attn(
    extractor,
    input_ids,
    attention_mask,
    n_attn_layers=6,
    hidden_layer_indices=[3, 6, 9, 12],
    return_hidden_layers=True,
)
```

MARS 输出：

```text
hidden     : (B, T, 1056)
attn_stack : (B, 6, 12, T, T)
```

然后 `_extract_mars()` 去掉 special tokens：

```python
base_len = input_ids.shape[1] - 2
h = hidden[:, 1:1+base_len, :]
a = attn_stack[:, :, :, 1:1+base_len, 1:1+base_len]
```

得到：

```text
h : (B, S, 1056)
a : (B, 6, 12, S, S)
```

再 pad 到 `set_len=L`：

```text
mars_hidden : (B, L, 1056)
mars_attn   : (B, 6, 12, L, L)
```

### 6.3 MARS 1D hidden → pair_1d

`mars_1d_proj`：

```python
Linear(1056 -> 192)
GELU
Linear(192 -> 96)
```

shape：

```text
mars_hidden : (B, L, 1056)
proj_1d     : (B, L, 96)
```

构造 residue pair `(i,j)` 的 1D 拼接特征：

```python
pair_1d = cat([proj_i, proj_j], dim=-1)
```

shape：

```text
proj_i  : (B, L, L, 96)
proj_j  : (B, L, L, 96)
pair_1d : (B, L, L, 192)
```

这一步表达的是：一个候选配对 `(i,j)` 同时看到第 `i` 个碱基和第 `j` 个碱基的 MARS hidden。

### 6.4 MARS attention → pair_2d

MARS attention：

```text
mars_attn : (B, 6, 12, L, L)
```

先把层和 head 展平：

```python
attn_flat = mars_attn.reshape(B, -1, L, L)
```

shape：

```text
attn_flat : (B, 72, L, L)
```

`mars_2d_proj`：

```python
Conv2d(72 -> 48, kernel_size=1)
GELU
Conv2d(48 -> 48, kernel_size=1)
```

shape：

```text
pair_2d before permute : (B, 48, L, L)
pair_2d after permute  : (B, L, L, 48)
```

这一步把 MARS 最后 6 层 × 12 heads 的 attention map 压缩成 48 维 pair 特征。

### 6.5 one-hot 序列 → seq_pair

`seq_oh`：

```text
seq_oh : (B, L, 4)
```

构造 `(i,j)` 碱基组合：

```python
seq_i = seq_oh.unsqueeze(2).expand(-1, -1, L, -1)
seq_j = seq_oh.unsqueeze(1).expand(-1, L, -1, -1)
seq_pair = (seq_i.unsqueeze(-1) * seq_j.unsqueeze(-2)).reshape(B, L, L, 16)
```

shape：

```text
seq_i    : (B, L, L, 4)
seq_j    : (B, L, L, 4)
outer    : (B, L, L, 4, 4)
seq_pair : (B, L, L, 16)
```

再投影：

```python
seq_proj = Linear(16 -> hidden_dim // 8)
```

v9 中 `hidden_dim=192`，所以：

```text
seq_pair : (B, L, L, 24)
```

### 6.6 拼接 pair feature + input projection

拼接三类特征：

```python
pair = cat([pair_1d, pair_2d, seq_pair], dim=-1)
```

shape：

```text
pair_1d  : (B, L, L, 192)
pair_2d  : (B, L, L, 48)
seq_pair : (B, L, L, 24)
cat      : (B, L, L, 264)
```

然后 `input_proj`：

```python
Linear(264 -> 192)
GELU
Dropout(0.2)
Linear(192 -> 192)
```

得到 Axial Transformer 的输入：

```text
pair : (B, L, L, 192)
```

注意：旧文档中可能写过 `pos_bias` 被拼接进 pair feature；但 v9 当前代码已经去掉 `pos_bias`，位置信息由 2D RoPE 在 attention 中注入。

### 6.7 8 层 Axial Transformer + 2D RoPE

v9 有 8 个 `AxialAttentionBlock`。每层结构：

```text
row attention
  → column attention
  → FFN
```

#### Row attention

输入：

```text
pair : (B, L, L, 192)
```

在代码里记作：

```text
x: (B, N, S, D)
N = L  # row 维
S = L  # 每一行内部 attention 的序列长度
D = 192
```

QKV projection：

```text
inner_dim = num_heads * dim_head = 6 * 32 = 192
row_qkv: Linear(192 -> 3 * 192 = 576)
```

shape：

```text
qkv : (B, L, L, 576)
q   : (B, L, L, 192)
k   : (B, L, L, 192)
v   : (B, L, L, 192)
```

拆 head：

```text
q, k, v : (B*L, 6, L, 32)
```

对 `q,k` 施加 RoPE：

```text
q_rot, k_rot : (B*L, 6, L, 32)
```

SDPA 后再合并：

```text
attn_out : (B*L, 6, L, 32)
out      : (B, L, L, 192)
```

再经过 `row_out: Linear(192 -> 192)`，残差加回 pair。

#### Column attention

column attention 先 transpose：

```python
x = pair.transpose(1, 2)
```

shape 仍是：

```text
x : (B, L, L, 192)
```

只是 attention 方向从 row 变为 column。处理流程与 row attention 相同，最后再 transpose 回来。

#### FFN

```text
LayerNorm(192)
Linear(192 -> 768)
GELU
Dropout(0.2)
Linear(768 -> 192)
Dropout(0.2)
```

每个 block 输入输出 shape 不变：

```text
(B, L, L, 192) -> (B, L, L, 192)
```

8 层堆叠后仍是：

```text
pair : (B, L, L, 192)
```

### 6.8 Contact head

输出前先归一化：

```python
pair = out_norm(pair)
```

shape：

```text
pair : (B, L, L, 192)
```

`contact_head`：

```python
Linear(192 -> 96)
GELU
Dropout(0.2)
Linear(96 -> 1)
```

shape：

```text
contact_head(pair) : (B, L, L, 1)
permute            : (B, 1, L, L)
```

然后强制对称：

```python
logit = (logit + logit.transpose(2, 3)) / 2
logit = logit * contact_mask
```

最终训练 logit：

```text
logit : (B, 1, L, L)
```

### 6.9 Density head

v9 还预测每条 RNA 的配对密度，用于推理时决定最多选多少 pair。

先按有效 mask pooling：

```python
pair_pooled = (pair * valid.unsqueeze(-1)).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).unsqueeze(-1)
```

shape：

```text
valid       : (B, L, L)
pair_pooled : (B, 192)
```

`density_head`：

```python
Linear(192 -> 64)
GELU
Linear(64 -> 1)
Sigmoid
```

shape：

```text
density_pred : (B, 1)
```

---

## 7. 训练时的 loss

v9 的 `forward()` 返回：

```text
loss, loss_dict
```

主要 loss 组件在 `_compute_loss()`：

| Loss | 作用 |
|---|---|
| Focal BCE + OHEM | 处理正负样本极不平衡，只重点学习 hard negatives |
| Dice loss | 提升 contact map 重叠质量 |
| DST / Tversky | 对低密度 RNA 更激进惩罚 false positive |
| pair_count loss | 约束预测配对数量接近 GT |
| ratio penalty | 防止预测数量过多 |
| density MSE | 训练 density head |
| FP penalty | 显式惩罚 false positive |
| Shift-aware loss | 对距离 GT 1/2 格以内的 FP 减轻惩罚，降低位置轻微偏移的损失 |

v9 的关键目标不是只优化逐点 BCE，而是同时约束：

```text
位置是否正确 + 配对数量是否合理 + 低密度样本是否过预测 + 轻微错位是否区别对待
```

---

## 8. 推理时如何从 score 到二值 contact map

`DensityNetProPlus.predict()` 和训练前向基本一致，区别是最后多了 projection。

先得到：

```text
logit : (B, 1, L, L)
score = sigmoid(logit) : (B, 1, L, L)
density_pred : (B, 1)
```

然后计算有效长度：

```text
l_eff : (B,)
```

如果使用 density budget：

```python
length_factor = (100.0 / l_eff.clamp(min=50)) ** length_decay
length_factor = length_factor.clamp(min=budget_floor)
max_pairs = round(density_pred * l_eff * length_factor * 1.05)
```

v9 默认 sampling 配置：

```json
"sampling": {
  "use_density_budget": true,
  "default_budget_fraction": 0.30,
  "score_threshold": 0.43,
  "length_decay": 0.15,
  "budget_floor": 0.6
}
```

候选位置：

```python
upper = triu(ones(L,L), diagonal=3)
candidates = score * contact_mask * upper
candidates[candidates < score_threshold] = 0
```

含义：

- 只取上三角，避免重复；
- `diagonal=3` 排除距离太近的局部位置；
- 低于 `score_threshold=0.43` 的候选丢弃；
- 选 top-k，k 由 `max_pairs` 控制；
- 最后镜像成对称 contact map。

最终输出：

```text
pred  : (B, 1, L, L)  # 二值 contact map
score : (B, 1, L, L)  # sigmoid probability
```

---

## 9. 一个具体 shape 示例

假设 batch 中最长 RNA 长度：

```text
S = 127
B = 2
```

由于 v9 collate 默认 padding 到 4 的倍数：

```text
L = ceil(127 / 4) * 4 = 128
T = 127 + 2 = 129
```

完整 shape 流：

```text
seqs
  -> input_ids                         (2, 129)
  -> attention_mask                    (2, 129)
  -> seq_oh                            (2, 128, 4)
  -> contact/contact_mask              (2, 1, 128, 128)

MARS-LX
  -> hidden                            (2, 129, 1056)
  -> attn_stack                        (2, 6, 12, 129, 129)
  -> remove special + pad
  -> mars_hidden                       (2, 128, 1056)
  -> mars_attn                         (2, 6, 12, 128, 128)

Pair construction
  -> mars_1d_proj                      (2, 128, 96)
  -> pair_1d                           (2, 128, 128, 192)
  -> mars_attn flatten                 (2, 72, 128, 128)
  -> mars_2d_proj                      (2, 128, 128, 48)
  -> seq_pair                          (2, 128, 128, 24)
  -> concat                            (2, 128, 128, 264)
  -> input_proj                        (2, 128, 128, 192)

8 × AxialAttentionBlock
  -> row q/k/v                         (2*128, 6, 128, 32)
  -> col q/k/v                         (2*128, 6, 128, 32)
  -> pair                              (2, 128, 128, 192)

Heads
  -> contact_head                      (2, 128, 128, 1)
  -> permute + symmetrize + mask        (2, 1, 128, 128)
  -> density_head                      (2, 1)

Inference projection
  -> score                             (2, 1, 128, 128)
  -> top-k symmetric pred contact map   (2, 1, 128, 128)
```

---

## 10. v9 的实验结论概况

根据 `docs/v9/v9_version_summary.md`：

| 实验 | RoPE | 正则化 | Val F1 | Test F1 |
|---|---|---|---:|---:|
| `v9_full` | ON | dropout=0.2, drop_path=0.15 | 0.6814 | 0.6961 |
| `v9_low_reg` | ON | dropout=0.1, drop_path=0.05 | 0.6722 | 0.6804 |
| `v9_no_rope` | OFF | dropout=0.2, drop_path=0.15 | 0.5930 | 0.5770 |

关键结论：

1. **2D RoPE 是 v9 最关键增益点**：关闭 RoPE 后 Test F1 从 0.6961 降到 0.5770；
2. **增强正则化有帮助但不是主因**：低正则仍有 0.6804 Test F1；
3. **冻结 MARS 是 v9 上限瓶颈之一**：v9 只训练下游约数百万参数，MARS 160M 基座未针对结构预测微调；
4. 后续 v10 的核心方向就是放开 MARS 权重进行端到端微调。

---

## 11. 一句话总结

v9 可以理解为：

```text
冻结的 MARS-LX 负责提供 RNA 语言先验；
MARS hidden 通过 outer concat 变成 pair 语义；
MARS attention 直接提供天然的 L×L pair map；
one-hot 碱基组合补充局部配对类型；
8 层带 2D RoPE 的 Axial Transformer 在 L×L 空间细化 pair 表征；
最后 contact head 输出配对概率，density head 控制推理时的配对预算。
```
