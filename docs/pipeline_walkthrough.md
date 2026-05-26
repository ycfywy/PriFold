# PriFold 推理流水线 Walkthrough

> 本文档详细描述 PriFold 如何将一条输入的 RNA 序列转换为二级结构预测结果。

## 总览

PriFold 的核心任务是：给定一条 RNA 序列（如 `AUGCGUUAC...`），预测其中哪些碱基彼此配对，输出一个 **L×L 的接触图（Contact Map）**。矩阵中 `[i][j] = 1` 表示位置 i 和位置 j 的碱基形成氢键配对，完整描述了 RNA 的二级结构（茎环、假结等）。

---

## 流水线架构图

```
RNA 序列 (str)
    │
    ▼
┌──────────────────────────────────┐
│  1. 预处理 & Tokenization        │
│     U→T 替换 + EsmTokenizer      │
└──────────────┬───────────────────┘
               │ token_ids (B, L)
               ▼
┌──────────────────────────────────┐
│  2. MARS 语言模型 (LLaMA2)       │
│     12层 Transformer Encoder     │
│     输出: hidden_states (B,L,1056)│
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  3. PairwiseOnly                 │
│     Linear(1056→128)             │
│     外积拼接 → (B, 256, L, L)    │
└──────────────┬───────────────────┘
               │
               │    ┌──────────────────────────────┐
               │    │  生物先验: get_posbias()      │
               │    │  碱基配对规则 → L×L 偏置矩阵  │
               │    └──────────────┬───────────────┘
               │                   │
               ▼                   ▼
┌──────────────────────────────────┐
│  4. RNAformerStack (4层轴注意力)  │
│     行注意力 → 列注意力 → ConvFFN │
│     pos_bias 注入第1层注意力权重   │
│     输出: (B, L, L, 256)         │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  5. 输出层                        │
│     Linear(256→1) + Sigmoid      │
│     阈值判定 (>0.45)             │
└──────────────┬───────────────────┘
               │
               ▼
      L×L 二值接触图 (二级结构)
```

---

## Step-by-Step Walkthrough

### Step 1: 预处理与 Tokenization

**文件**: `utils/tools.py` → `get_posbias()`, `inference.py`

输入的 RNA 序列首先经过预处理：

```python
# RNA序列中的 U 替换为 T（内部使用 DNA 编码）
sequence = rna_seq.replace('U', 'T')

# 使用 EsmTokenizer（词表: vocab_esm_mars.txt, 20个碱基/特殊token）
tokenizer = EsmTokenizer(vocab_file="vocab_esm_mars.txt")
token_ids = tokenizer.encode(sequence)  # → (L,) 的整数数组
```

**关键点**：模型内部统一使用 DNA 编码（A/T/G/C），因此 U 被映射为 T。

---

### Step 2: MARS 语言模型提取序列特征

**文件**: `prifold/llama2.py` → `Transformer` 类

MARS 是一个预训练的 RNA 语言模型，采用 LLaMA2 架构但工作在 Encoder 模式：

```python
class Transformer(nn.Module):
    """
    MARS 语言模型 (LLaMA2 架构)
    - dim=1056, n_layers=12, n_heads=12, vocab_size=20
    - 使用 RoPE 旋转位置编码
    - 支持 MLM/GLM 预训练
    """
    def forward(self, tokens, attention_mask):
        h = self.tok_embeddings(tokens)  # (B, L, 1056)
        for layer in self.layers:        # 12层 TransformerBlock
            h = layer(h, attention_mask)
        h = self.norm(h)                 # RMSNorm
        return logits, h                 # 推理时使用 h
```

输出 `hidden_states` 的形状为 `(B, L, 1056)`，包含每个碱基位置的上下文表示。

---

### Step 3: PairwiseOnly — 1D 到 2D 的升维

**文件**: `utils/RNAformer/model/Riboformer_outfirst.py` → `PairwiseOnly` 类

这一步将一维的序列特征转换为二维的配对表示：

```python
class PairwiseOnly(nn.Module):
    """
    将 (B, L, 1056) 的序列嵌入转换为 (B, 256, L, L) 的配对表示
    """
    def __init__(self, embed_dim=1056, latent_dim=128):
        self.proj = nn.Linear(embed_dim, latent_dim)  # 1056 → 128

    def forward(self, seq_embed):
        x = self.proj(seq_embed)         # (B, L, 128)
        # 外积拼接: 对每对 (i,j) 拼接 [x_i; x_j]
        pair = torch.cat([
            x.unsqueeze(2).expand(-1, -1, L, -1),  # (B, L, L, 128)
            x.unsqueeze(1).expand(-1, L, -1, -1),  # (B, L, L, 128)
        ], dim=-1)                                   # (B, L, L, 256)
        return pair.permute(0, 3, 1, 2)              # (B, 256, L, L)
```

**设计思想**：对于二级结构预测，我们需要知道每对碱基 (i, j) 是否配对，因此需要 L×L 的二维表示。通过拼接 `[特征_i; 特征_j]` 来表征每个碱基对。

---

### Step 4: RNAformerStack — 轴注意力 + 生物先验注入

**文件**: `utils/RNAformer/model/Riboformer_outfirst.py` → `RNAformerStack`, `RNAformerBlock`, `TriangleAttention`

这是模型的核心结构预测模块，使用 4 层轴注意力处理 L×L 矩阵。

#### 4.1 生物先验计算

**文件**: `utils/tools.py` → `get_posbias()`

```python
def get_posbias(seq, scale=0.01):
    """
    根据碱基配对规则生成 L×L 位置偏置矩阵
    
    配对得分:
      - A-T (Watson-Crick): 3
      - G-C (Watson-Crick): 6
      - G-T (Wobble pair):  1
    
    返回: (L, L) 矩阵，元素为 1 + score * scale
    """
    bias = torch.ones(L, L)
    for i in range(L):
        for j in range(L):
            pair = seq[i] + seq[j]
            if pair in {'AT','TA'}: bias[i,j] += 3 * scale
            if pair in {'GC','CG'}: bias[i,j] += 6 * scale
            if pair in {'GT','TG'}: bias[i,j] += 1 * scale
    return bias
```

#### 4.2 RNAformerStack 结构

```python
class RNAformerStack(nn.Module):
    """
    4 层 RNAformerBlock 堆叠
    pos_bias 仅注入第 1 层，后续层 bias = 0
    """
    def forward(self, pair_latent, pos_bias, pair_mask):
        for i, block in enumerate(self.blocks):
            if i == 0:
                pair_latent = block(pair_latent, pos_bias, pair_mask)
            else:
                pair_latent = block(pair_latent, zeros_like(pos_bias), pair_mask)
        return pair_latent
```

#### 4.3 单层 RNAformerBlock

```python
class RNAformerBlock(nn.Module):
    """
    单层轴注意力块:
      1. 行方向 TriangleAttention (per_row)
      2. 列方向 TriangleAttention (per_column)
      3. ConvFeedForward (3×3 卷积 FFN)
    """
    def forward(self, x, bias, mask):
        x = x + self.row_attn(x, bias, mask)   # 沿行方向做注意力
        x = x + self.col_attn(x, bias, mask)   # 沿列方向做注意力
        x = x + self.conv_ffn(x)               # 3×3 卷积特征变换
        return x
```

#### 4.4 TriangleAttention（三角注意力）

```python
class TriangleAttention(nn.Module):
    """
    对 L×L 矩阵沿某一轴做注意力
    - per_row: 将矩阵视为 L 个行序列，每行内部做注意力
    - per_column: 将矩阵视为 L 个列序列，每列内部做注意力
    
    关键: pos_bias 作为乘法偏置注入注意力权重
    """
```

在 `Attention2d` 中，pos_bias 的注入方式为：
```python
attn_weights = query @ key.transpose(-2, -1) / sqrt(d)
attn_weights = attn_weights * bias  # 乘法偏置！非加法
attn_weights = softmax(attn_weights)
output = attn_weights @ value
```

**设计思想**：轴注意力将 O(L⁴) 复杂度降为 O(L³)；pos_bias 将热力学碱基配对规则编码为注意力偏置，引导模型关注更可能配对的位置。

---

### Step 5: 输出层与后处理

**文件**: `utils/RNAformer/model/Riboformer_outfirst.py` → `RiboFormer` 类, `inference.py`

```python
class RiboFormer(nn.Module):
    """主模型类，组合所有子模块"""
    
    def __init__(self):
        self.lm = Transformer(...)           # MARS 语言模型
        self.pair_embedding = PairwiseOnly()  # 1D→2D
        self.rna_former = RNAformerStack()    # 轴注意力
        self.output_proj = nn.Linear(256, 1)  # 输出投影

    def forward(self, data_dict):
        # 1. 语言模型
        _, h = self.lm(tokens, attention_mask)
        # 2. 配对嵌入
        pair = self.pair_embedding(h)
        # 3. 轴注意力 + 生物先验
        pair = self.rna_former(pair, pos_bias, mask)
        # 4. 输出
        logits = self.output_proj(pair).squeeze(-1)  # (B, L, L)
        return logits
```

**后处理**（在 `inference.py` 中）：

```python
# 裁剪到实际序列长度（去除 padding）
seq_length = attention_mask[idx].sum().item()
logit = logits[idx, :seq_length, :seq_length]

# Sigmoid → 概率值 [0, 1]
probs = torch.sigmoid(logit).detach().cpu().numpy()

# 阈值判定 → 二值接触图
threshold = 0.45
contact_map = (probs > threshold).astype(np.float32)
# contact_map[i][j] = 1 → 第i个碱基与第j个碱基配对
```

---

## 关键类速查表

| 类名 | 文件 | 职责 |
|------|------|------|
| `RiboFormer` | `utils/RNAformer/model/Riboformer_outfirst.py` | **主模型类**，组合所有子模块，定义完整前向传播 |
| `Transformer` | `prifold/llama2.py` | 预训练 RNA 语言模型 (LLaMA2 架构, 160M 参数)，提取序列上下文特征 |
| `TransformerBlock` | `prifold/llama2.py` | Transformer 单层：Self-Attention + SwiGLU FFN |
| `PairwiseOnly` | `Riboformer_outfirst.py` | 外积拼接，将 1D 序列特征升维为 2D 配对表示 |
| `RNAformerStack` | `Riboformer_outfirst.py` | 4 层轴注意力堆叠，管理 pos_bias 的注入策略 |
| `RNAformerBlock` | `Riboformer_outfirst.py` | 单层：行注意力 + 列注意力 + 卷积 FFN |
| `TriangleAttention` | `Riboformer_outfirst.py` | 三角注意力，支持行/列方向，注入生物先验偏置 |
| `Attention2d` | `Riboformer_outfirst.py` | 2D 注意力核心，实现 bias 的乘法注入 |
| `ConvFeedForward` | `Riboformer_outfirst.py` | 3×3 卷积前馈网络 (GroupNorm + SiLU + Conv2d) |
| `SSDataset` | `utils/predictor.py` | 数据集类，加载 RNA 序列 + 接触图标注 |
| `Augmentation` | `utils/predictor.py` | RNA 协变数据增强（模拟碱基对协同突变） |
| `EsmTokenizer` | `prifold/tokenization_esm.py` | RNA 序列分词器，20 碱基词表 |

---

## 运行示例

```bash
# 推理脚本
python inference.py \
    --mode bprna-test \
    --model_scale lx \
    --batch_size 1 \
    --scale 0.01 \
    --model_path ./model/ss_model_bprna.pth \
    --pretrained_lm_dir ./model \
    --data_dir ./data
```

**输入**: `data/` 目录下的 RNA 序列文件  
**输出**: 每条序列的 L×L 接触图预测，并计算与 ground truth 的 F1/Precision/Recall

---

## 设计亮点

1. **预训练语言模型 + 结构预测的两阶段架构**：MARS 提供丰富的序列语义，下游轴注意力模块专注于配对关系建模。
2. **生物先验作为乘法注意力偏置**：将 Watson-Crick/Wobble 配对规则以极小系数 (×0.01) 注入，既提供先验又不过度约束模型。
3. **轴注意力 (Axial Attention)**：将 L×L 问题分解为行和列两个方向，复杂度从 O(L⁴) 降至 O(L³)。
4. **卷积 FFN**：3×3 卷积捕捉配对矩阵的局部模式（如相邻碱基对的堆叠效应）。
5. **pos_bias 仅注入第 1 层**：让模型在浅层利用先验快速定位候选配对，深层自由学习复杂模式。
