# PriFold-SymFlow v4 架构详解

## 一句话概括

```
RNA 序列 → MARS-LX(160M) 提取 hidden/attention
         → DA-SE-DiT-v4 (Bernoulli DFM 主干)
         → flow logit + direct score + density 预测
         → score-first greedy 投影
         → L×L contact map
```

## 数据流图

```
输入：RNA 序列 "AUGCGC..." (长度 L)

1. 数据预处理 (data.py)
   ├── 序列 → tokenize (MARS vocab)        → input_ids (L+2)
   ├── 序列 → one-hot (ATGC)               → seq_oh (L, 4)
   ├── 序列 → 碱基配对先验                  → pos_bias (L, L)
   ├── 接触图 .npy                          → contact (L, L)
   └── Padding 到 patch_size 倍数           → set_len = ceil(L/4)*4
       └── contact_mask (set_len, set_len)  标记有效区域

2. MARS 特征提取 (mars_forward.py → model.py._extract_mars)
   ├── input_ids → MARS encoder (12 layers, frozen)
   ├── 最后 6 层 attention   → attn_stack (6, 12, L, L)
   ├── 第 3/6/9/12 层 hidden → hidden_layers [4 × (L, 1056)]
   └── 最终层 hidden         → mars_hidden (L, 1056)

3. DiT 主干 (backbone.py: DASEDiT_MARS_v4)
   ├── 特征构建:
   │   ├── hidden_layers → MultiLayerMarsFusion → mars_emb_1d (L, 32)
   │   │                 → outer_concat         → mars_2d (64, L, L)
   │   ├── attn_stack    → MarsAttentionProj    → mars_attn_2d (16, L, L)
   │   ├── x_t (当前flow状态) → Embedding       → x_emb (8, L, L)
   │   ├── pos_bias                             → (1, L, L)
   │   └── seq_oh        → outer_concat         → seq_2d (8, L, L)
   │   合计: 8+64+16+1+8 = 97 channels → 对称化 → (97, L, L)
   │
   ├── Patch 嵌入:
   │   Conv2d(97, 256, k=4, s=4) → tokens (L/4, L/4, 256)
   │
   ├── 条件注入:
   │   ├── CondAttentionBias: mars_attn_2d + pos_bias → avg_pool → 每层 attention bias
   │   ├── ControlInjectMLP: 每 2 层注入条件到 tokens (zero-init)
   │   └── Global cond: time_emb + mars_global_mean + density_hint → AdaLN modulation
   │
   ├── 9 层 DASEDiTBlockV4:
   │   ├── AdaLN-Zero modulation
   │   ├── DilatedAxialAttentionV4 (row + col, with attn_bias)
   │   │   ├── AxialRoPE (相对位置)
   │   │   ├── QK-Norm (RMSNorm)
   │   │   └── Dilation pattern: [1,1,1, 2,2,2, 4,4,4]
   │   ├── Triangle Multiplicative Update (层 6-8)
   │   └── GatedFFN (SwiGLU)
   │
   └── 输出:
       ├── UnPatchify → (1, L, L) → RefineConv → flow_logit
       ├── UnPatchify → (1, L, L) → RefineConv → direct_logit
       └── DensityHead(global_cond) → density_pred (scalar)

4. Loss (discrete_flow.py: BernoulliFlowLoss_v5)
   ├── Adaptive BCE (pos_weight 随 density 调整 + focal + time weighting)
   ├── Direct BCE (对 direct_logit 单独监督)
   ├── Pair count calibration loss
   ├── Stacking loss (物理约束: 堆叠连续性)
   ├── Non-crossing loss (物理约束: 每碱基最多配对一次)
   └── Density regression loss (Huber)

5. 采样 (model.py: sample)
   ├── 初始化: x_0 ~ Bernoulli(rho_0=0.005) (稀疏随机)
   ├── τ-leap CTMC (20 steps, sine schedule):
   │   每步: flow_logit + direct_logit → p_flow, p_direct
   │         score = (1-w)*p_flow + w*p_direct
   │         rate_01, rate_10 = CTMC_rates(x_t, p_flow, t)
   │         flip bits stochastically
   └── 投影 (score-first greedy matching):
       score → 排序 → 贪心选最高分 pair → 保证 |i-j|≥3 + 每碱基最多1配对
       budget = length * default_budget_fraction (0.35)
       threshold = 0.5

6. 输出: contact_map (L, L) 二值矩阵
```

## 关键设计决策

### 为什么用 Bernoulli Discrete Flow 而不是连续 Diffusion？

- Contact map 是二值 0/1 矩阵，离散 flow 更自然
- 不需要处理连续噪声的方差问题
- CTMC 采样天然保持 0/1 状态

### 为什么用 Axial Attention 而不是 Full 2D Attention？

- Contact map 是 L×L，full attention 是 O(L^4) — 不可行
- Axial (row + col) 是 O(L^3)，且对称结构天然适合 contact map
- Dilation 扩大感受野不增加计算量

### 为什么有 Direct Score Head？

- Flow logit 是条件于时间 t 的"velocity field"，不是最终预测
- Direct score head 直接预测 contact 概率，不依赖 flow 时间
- 推理时混合两者：score = (1-w)*flow + w*direct

### 为什么要 Score-first Projection 而不是直接取 x_T？

- 离散采样的最终 x_T 是随机的，可能漏掉高置信度的 pair
- Score-first 从所有合法位置中选最高分的 pair，更稳定
- Budget 控制防止过预测

### 变长序列处理

- Bucket Sampler: 同 batch 内序列长度相近，减少 padding 浪费
- Dynamic padding: 每个 batch 的 set_len 不同，取 max(lengths) 上取整到 patch_size 倍数
- 隐式 masking: padding 区域输入全 0，输出通过 contact_mask 屏蔽
- 无显式 attention mask（依赖零向量贡献极小）

## 参数量

```
MARS-LX (frozen):      ~160M (不参与训练)
DA-SE-DiT-v4 主干:     ~12.9M (可训练)
总计:                   ~173M
```
