# PriFold-SymFlow 架构与 Shape 流转讲解

> 本文以**一条 RNA 序列从输入到输出**为主线，逐步追踪张量 shape 的变化，讲清当前 v1 版 SymFold 分支的端到端架构。
> 代码对应：`symfold/data.py`、`symfold/model.py`、`symfold/dit.py`、`symfold/discrete_flow.py`。

更新时间：2026-06-01

---

## 0. 先约定符号与默认超参

| 符号 | 含义 | 默认值 |
| --- | --- | ---: |
| `L` | RNA 序列真实长度（碱基数） | 例：116 |
| `S` | padding 后的长度（`patch_size` 整数倍） | 例：116（已是 4 的倍数） |
| `B` | batch size | 8 |
| `D_mars` | MARS-LX 隐藏维 | 1056 |
| `d_pair` | MARS 投影后的 pair 维 | 64 |
| `H` | DiT hidden_dim | 256 |
| `P` | patch_size | 4 |
| `rho_0` | flow 初始噪声密度 | 0.005 |

为方便追踪，下文用一条 **L=116** 的序列做示例（batch 里只看它，`B` 维省略时用 `1`）。

---

## 1. 总览：一张图看懂数据流

```text
RNA 序列 "GGCUA..."  (字符串, 长度 L)
        │
        │  data.py: U→T, tokenize, one-hot, pos_bias, padding 到 S
        ▼
┌───────────────────────────────────────────────────────────┐
│  一个 batch 的字典 (B 维)                                    │
│   input_ids      (B, L+2)        ← 带 <cls>/<eos>           │
│   attention_mask (B, L+2)                                   │
│   seq_oh         (B, S, 4)                                  │
│   pos_bias       (B, S, S)                                  │
│   contact        (B, 1, S, S)    ← 训练标签 x_1             │
│   contact_mask   (B, 1, S, S)                               │
└───────────────────────────────────────────────────────────┘
        │
        │  model.py: build_conditions()  —— 构造 2D 条件特征
        ▼
   cond  (B, 137, S, S)
        │
        │  flow 加噪 x_t + x_t embedding，拼接成 DiT 输入
        ▼
   features (B, 145, S, S)
        │
        │  dit.py: AxialDiT  (patchify → 6×轴注意力 → unpatch → 精修)
        ▼
   logit (B, 1, S, S)   +   density (B, 1)
        │
        │  discrete_flow.py: loss(训练) / CTMC采样+投影(推理)
        ▼
   pred  (B, 1, S, S)   ← 预测的 0/1 contact map（二级结构）
```

---

## 2. 阶段一：数据预处理（`data.py`）

输入是 CSV 里的一条记录：序列字符串 + 对应 `.npy` 的 `L×L` contact map。

### 2.1 序列侧

```text
"GGCUAGC..."  (str, len=L=116)
   │ U→T
"GGCTAGC..."  (str, len=116)
   │ tokenizer.batch_encode_plus  (加 <cls>/<eos>)
input_ids       (1, 118)      # L+2
attention_mask  (1, 118)
   │ _one_hot
seq_oh          (1, 116, 4)   # A/T/G/C，再 padding 到 (1, S, 4)
```

### 2.2 结构侧 / 先验侧

```text
contact map .npy   (116, 116)  0/1
   │ padding 到 S，加 channel 维
contact        (1, 1, S, S)
contact_mask   (1, 1, S, S)    # 有效区域=1

pos_bias       (1, S, S)       # 配对先验：A-T=3 / G-C=6 / G-T=1，乘 scale
```

> **padding 规则**：`S = ceil(L/P)*P`，保证能被 DiT 的 patch_size=4 整除。L=116 已是 4 的倍数，故 S=116。

---

## 3. 阶段二：构造 DiT 条件特征（`model.py: build_conditions`）

这一步把 1D 的序列信息「外积」成 2D 的 pair 特征图。

### 3.1 MARS-LX 提取序列表征

```text
input_ids (1, 118) ─► MARS-LX (frozen) ─► hidden (1, 118, 1056)
   │ 去掉 <cls>/<eos>，对齐到 S
base_hidden (1, S, 1056)
   │ mars_proj: Linear(1056→128)→GELU→Linear(128→64)
mars_1d (1, 64, S)        # permute 成 (B, C, L) 形式
```

### 3.2 outer concat：1D → 2D（核心操作）

`_outer_concat` 把位置 i、j 的特征拼到格子 `[i,j]`，通道翻倍：

```text
mars_1d (1, 64, S) ─► outer concat ─► mars_2d (1, 128, S, S)
seq_oh  (1, S, 4) → (1,4,S) ─► outer concat ─► seq_2d (1, 8, S, S)
```

原理：
```python
xi = x.unsqueeze(-1).expand(B, C, S, S)   # 行广播
xj = x.unsqueeze(-2).expand(B, C, S, S)   # 列广播
cat([xi, xj], dim=1)                       # 通道 C → 2C
```

### 3.3 拼接条件 + 对称化

```text
parts = [ mars_2d (128) , seq_2d (8) , pos_bias (1) ]
cond = cat(parts, dim=1)        ─►  cond (1, 137, S, S)
cond = 0.5 * (cond + condᵀ)     # 沿最后两维对称化
```

通道账：`2*d_pair(128) + 8 + 1 = 137`。

---

## 4. 阶段三：Flow 加噪与 DiT 输入拼接（`model.py: forward`）

### 4.1 前向加噪（训练）

```text
x_1 = contact (1,1,S,S)            # 真值结构
t ~ Uniform(0,1)   shape (B,)
x_t ~ Bernoulli( t*x_1 + (1-t)*rho_0 )   ─► x_t (1,1,S,S)
x_t = symmetrize(x_t) * contact_mask
```

直观理解：`t→1` 时 `x_t` 接近真值，`t→0` 时接近全噪声（密度 rho_0）。

### 4.2 x_t 嵌入并拼成 DiT 输入

```text
x_t (1,1,S,S) ─► Embedding(2→8) ─► x_emb (1, 8, S, S)
features = cat([ x_emb(8) , cond(137) ], dim=1)
        ─►  features (1, 145, S, S)
```

**这就是 DiT 的最终输入：145 通道的 L×L 特征图**
（145 = x_t嵌入 8 + MARS 128 + 序列 one-hot 8 + pos_bias 1）。

> 时间 `t` **不**进通道，而是后面在 DiT 内部以 AdaLN 调制方式注入。

---

## 5. 阶段四：AxialDiT 主干（`dit.py`）

### 5.1 Patchify：降分辨率进 token 空间

```text
features (1, 145, S, S)
   │ PatchEmbed2D: Conv2d(145→256, kernel=4, stride=4)
tokens (1, S/4, S/4, 256)        # S=116 → 29×29 个 patch token
```

### 5.2 时间嵌入（条件信号）

```text
t (1,) ─► SinusoidalTimeEmbedding(256) ─► MLP ─► cond_t (1, 256)
```

### 5.3 6 × AxialDiTBlock（行注意力 + 列注意力 + FFN）

每个 block 内 token shape 不变 `(1, 29, 29, 256)`，但做三段残差，每段都被 `cond_t` 以 AdaLN（scale/shift）调制：

```text
① 行注意力: reshape (1*29, 29, 256) → MHSA → 还原
② 列注意力: permute后 (1*29, 29, 256) → MHSA → 还原
③ FFN:      Linear 256→2048→256
```

> 轴向注意力把 `O(S²·S²)` 的全注意力降为行/列各 `O(S²·S)`，使长序列可算。

### 5.4 Unpatchify + 输出精修 + 约束

```text
tokens (1, 29, 29, 256)
   │ final AdaLN
   │ density_head: 全局平均 → MLP → Sigmoid ─► density (1, 1)
   │ UnPatchify2D: ConvTranspose2d(256→1, k=4,s=4)
logit (1, 1, S, S)
   │ OutputRefineConv: 残差小卷积精修
   │ 对称化 0.5*(logit+logitᵀ)
   │ mask: |i-j|<3 → -10 ; padding 区 → -10
logit (1, 1, S, S)   +   density (1, 1)
```

---

## 6. 阶段五：训练损失 vs 推理采样（`discrete_flow.py`）

### 6.1 训练：BernoulliFlowLoss

```text
logit (1,1,S,S) , x_1 (1,1,S,S) , t , mask
   → adaptive pos_weight BCE + focal 调制 + 时间加权
   → (+ density MSE，权重 0.2)
loss (标量)
```

### 6.2 推理：CTMC 多步采样 + 贪心投影

```text
x_0 ~ Bernoulli(rho_0)            (B,1,S,S)
for step in range(num_steps=20):
    logit, _ = DiT(x_t, t, cond)
    p_x1 = sigmoid(symmetrize(logit))
    rate_01, rate_10 = compute_ctmc_rates(...)
    按 rate*dt 概率翻转 x_t 的 0/1
    x_t = symmetrize(x_t) * mask
pred = project_to_valid_contact_map(x_t, p_x1, mask)   # 贪心：每碱基最多配对一次
─► pred (B, 1, S, S)   0/1 contact map
```

---

## 7. Shape 流转速查表（单样本，B 维省略）

| 阶段 | 张量 | shape |
| --- | --- | --- |
| 输入 | 序列字符串 | `len=L` |
| token | `input_ids` | `(L+2,)` |
| one-hot | `seq_oh` | `(S, 4)` |
| 先验 | `pos_bias` | `(S, S)` |
| 标签 | `contact (x_1)` | `(1, S, S)` |
| MARS 隐藏 | `hidden` | `(L+2, 1056)` |
| MARS 投影 | `mars_1d` | `(64, S)` |
| MARS 2D | `mars_2d` | `(128, S, S)` |
| 序列 2D | `seq_2d` | `(8, S, S)` |
| 条件 | `cond` | `(137, S, S)` |
| 加噪态嵌入 | `x_emb` | `(8, S, S)` |
| **DiT 输入** | `features` | `(145, S, S)` |
| patch token | `tokens` | `(S/4, S/4, 256)` |
| DiT 输出 | `logit` | `(1, S, S)` |
| 密度 | `density` | `(1,)` |
| **最终预测** | `pred` | `(1, S, S)` |

---

## 8. 一句话总结

一条 RNA 序列经 `数据预处理 → MARS-LX 编码 → outer concat 升 2D → 拼 pos_bias/序列/加噪态(145通道) → 轴向 DiT → L×L logit → Flow 采样+投影`，最终输出一张 `L×L` 的 0/1 contact map，即预测的 RNA 二级结构。整条链路的空间维始终围绕 `L×L`（padding 后 `S×S`，DiT 内部降到 `S/4 × S/4`），通道维是信息融合的载体。
