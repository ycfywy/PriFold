# PriFold-SymFlow 改进方案（DiT 路线）

> 配套：架构讲解 `prifold_symflow_architecture.md`、状态评审 `prifold_symflow_status_review.md`。
> 起草时间：2026-06-01。

---

## 0. 路线定位

我们的故事是 **"DiT-based discrete flow matching for RNA secondary structure"**——把 SymFold 风格的离散 flow matching 落到 RNA 上，**坚持 DiT 框架**（patch 化 + AdaLN-Zero 调制 + 时间条件每层注入），**不退回主线 RNAformer 全分辨率轴向 transformer 的判别式骨架**。

因此本方案不抄主线骨架，所有改进都在 DiT 框架内进行。主线 PriFold 给我们的启示仅有两条：
1. **MARS 输出做 2D 是核心**——但我们用比 outer concat 更强的方式（attention map）；
2. **pos_bias 是 attention bias 而非 channel**——我们也按这个用。

---

## 1. 当前架构的真正缺陷

| # | 缺陷 | 严重度 |
| --- | --- | --- |
| **A** | **pair feature 来源单薄**：仅用 MARS 最后一层 hidden 经 outer concat 升 2D，丢掉 12 层×12 head = 144 个天然 `(L,L)` 注意力图 | **高** |
| **B** | **DiT 块缺乏 pair-aware 操作**：当前 `AxialDiTBlock` 只有行/列 1D 注意力，没有 pair triangle 关系建模——而 contact map 的核心约束（嵌套配对、pseudoknot）需要三角传递 | **高** |
| **C** | **条件信号注入失衡**：DiT 的精髓是 AdaLN-Zero 让条件**每层调制**，但我们只让时间 t 走 AdaLN，结构条件（MARS、pos_bias）只在第 0 层 patch_embed 一次性吃掉 | **中** |
| **D** | **冗余通道**：`seq_oh` 是 MARS embedding 真子集；`pos_bias` 当通道吃浪费了它作 attention bias 的天然形态 | **中** |

---

## 2. v2 设计：DiT 框架内三个升级

### 2.1 改动总览

| 改动 | 解决问题 | 做法 |
| --- | --- | --- |
| **(1) MARS 多层 attention map → 主条件** | A | 抽 MARS 后 6 层 × 12 head = 72 个 `(L,L)` attn → 对称化 + APC → Conv 投影到 64 ch；pool 到 patch 网格 `(S/4, S/4, 64)` 后既作 channel cond 又作 attention bias |
| **(2) Patch-grid Triangle Update 模块** | B | 在 DiT block 的行/列 attention 之后插入 **patch 化的 triangle update**——只在 `S/4 × S/4` 网格上做，O((L/4)³) 显存可控 |
| **(3) ControlNet-style 跨层条件注入 + pos_bias 作 attention bias** | C | DiT 每隔 2 层从 zero-init MLP 把 cond 加回 token；pos_bias 池化到 patch 网格后作行/列 attention 的 additive bias |
| (顺手) 删 `seq_oh` | D | MARS embedding 真子集冗余，直接删 |

### 2.2 v2 的 DiT block 结构（核心创新）

我们升级的是 **DiTBlock 本身**，不是骨架。每个 block 内部从原来的 3 段（行 attn / 列 attn / FFN）升级到 4 段，全部走 AdaLN-Zero：

```text
AxialDiTBlock_v2:
  ┌── AdaLN(t, cond_global) ── Row Attention (with attn_bias = pos_bias_proj + mars_attn_proj) ──► +
  ├── AdaLN(t, cond_global) ── Col Attention (with attn_bias) ──► +
  ├── AdaLN(t, cond_global) ── Triangle Update (outgoing + incoming, on patch grid) ──► +
  └── AdaLN(t, cond_global) ── FFN ──► +
```

要点：
- **AdaLN-Zero**：每段调制都 zero-init，保证 v1 → v2 改造**不破坏已学到的能力**，新增模块"渐进开门"；
- **attention bias**：把 `pos_bias` 和 `mars_attn` 作为 row/col attention 的可加 bias（学习一个 `bias_proj: (n_layers×n_heads + 1) → n_heads_block` 投影），而不仅仅是 channel；
- **Triangle Update**：只在 patch grid 上做（`S/4=123` 时单 sample ~7MB，batch=8 ~56MB，OK），但同时用 zero-init 残差，初期为 0，慢慢学。

### 2.3 输入通道账

| 来源 | v1 | **v2** |
| --- | ---: | ---: |
| x_t embedding | 8 | 8 |
| MARS outer concat | 128 | — |
| **MARS attention map (新)** | — | **64** |
| seq_oh (删) | 8 | — |
| pos_bias (移走) | 1 | — |
| **总输入** | **145** | **72** |

**pos_bias 与 MARS attention map 不再走通道**，而是 pool 到 patch 网格后**作 attention bias**注入每个 DiT block 的行/列注意力——这是 DiT 框架内的"条件每层调制"路径。

### 2.4 v2 整体前向

```text
RNA 序列
  └─► MARS-LX (frozen)
        └─► attn_stack (B, 6, 12, L, L)         # 后 6 层 manual softmax
              │ 对称化 + APC + Conv2d(72→64)
              ▼
            mars_attn_2d (B, 64, L, L)
                │
                │ AvgPool(4) → mars_attn_patch (B, 64, S/4, S/4)
                │                                       ┐
RNA 序列 ──► one-hot (4, L, L 不再使用) ✗            │ 作为：
            pos_bias (1, L, L)                        │  (1) channel cond（concat 进 patch_embed）
                │ AvgPool(4) → pos_bias_patch         │  (2) attention bias（作 row/col attn 的 +bias）
                ▼                                      ┘
            
flow x_t (1, L, L) ──► Embedding(2→8) ──► x_emb (8, L, L)
                                          │
                                          │ concat: [x_emb(8), mars_attn_2d(64)] = (72, L, L)
                                          │ PatchEmbed2D (Conv 4×4) → tokens (S/4, S/4, 256)
                                          ▼
        ┌──── 6× AxialDiTBlock_v2 (AdaLN-Zero, 行/列 attn 带 bias, triangle update, FFN) ─┐
        │                                                                                  │
        │      cond_global = pool(mars_attn_patch + pos_bias_patch) → MLP                 │
        │      AdaLN modulation 输入：t_emb + cond_global                                  │
        │                                                                                  │
        │      每 2 层一次 ControlNet-style 注入：cond_proj → zero-init add 到 tokens     │
        └──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                              UnPatchify → logit (1, L, L)
                              对称化 + 短程/padding 屏蔽
                                          ▼
                       Bernoulli Flow Loss (训练) / CTMC sample (推理)
```

### 2.5 预期收益

| 改动 | 预期 ΔF1 | 依据 |
| --- | --- | --- |
| MARS 多层 attention map | +5~10 | RNA-FM/ESM-2 contact head 验证幅度 |
| Patch triangle update | +2~4 | AlphaFold/RNAformer 结论，但 patch 化会折损一部分 |
| pos_bias → attn bias + ControlNet 跨层注入 | +1~2 | DiT-Zero / AlphaFold pair-bias 思路 |

> v1 60-epoch 收敛预估 ≈ 0.55~0.60；**v2 目标 best val F1 ≥ 0.68**。

> 与"路线 1 RNAformer 全分辨率"相比，patch 化预计**少 3~5 分 F1**——但这是路线选择的代价，换来的是 SymFold-style DiT 的故事完整性、O((L/4)²) 的可扩展性，以及未来支持长 RNA（>1000 nt）的能力。

---

## 3. 实施细节

### 3.1 MARS 改造（最关键，1 天）

**派生文件 `prifold/llama2_with_attn.py`**，保留原文件干净：
1. `Attention.forward` 增加 `return_attn` 开关，仅在该开关 True 时走 manual softmax 路径（默认走 flash-attn 不变）；
2. `Transformer.forward` 增加 `output_attentions: int` 参数，表示**只对最后 N 层**返回 attention（默认 N=6）；
3. 返回 `attn_stack: (B, N, n_heads, L, L)`。

**APC 校正**：
```python
def apc(A):  # (B, K, L, L), 已对称化
    a_i = A.sum(-1, keepdim=True); a_j = A.sum(-2, keepdim=True)
    a   = A.sum((-1, -2), keepdim=True)
    return A - a_i * a_j / (a + 1e-9)
```

**显存**：6 × 12 × 490² × bf16 ≈ 17MB / sample，batch=8 ~140MB，无压力。

### 3.2 DiT block 升级（关键创新，1.5 天）

新增 / 改写：
- `AxialDiTBlock_v2`：4 段 AdaLN-Zero，行/列 attention 接受 `attn_bias` 参数；
- `PatchTriangleUpdate`：仿 AlphaFold 的 outgoing / incoming 三角更新，但只在 patch grid 上做；
- `CondAttentionBias`：将 `(B, K, S/4, S/4)` 的 cond 投影到 `(B, n_heads, S/4, S/4)` 作为行/列 attn 的 additive bias；
- `ControlInjectMLP`：zero-init MLP，在每 2 层 block 之间把 pooled cond 加回 token。

### 3.3 训练配置

新建 `symfold/config/prifold_symflow_v2.json`：
- 数据/优化器：继承 v1（全量、batch=8、lr=3e-4、aug=on）；
- 模型：`hidden_dim=256`, `num_heads=4`, `num_layers=6`, `patch_size=4`；
- 新增：`mars_attn_layers=6`, `mars_attn_proj_dim=64`, `triangle_dim=128`, `cond_inject_every=2`, `attn_bias_zero_init=true`。

---

## 4. 实施顺序

| 步骤 | 内容 | 预估 |
| --- | --- | --- |
| 1 | 派生 `prifold/llama2_with_attn.py` + 单测 attn shape | 半天 |
| 2 | 写 `MarsAttentionToPair`（对称化 + APC + Conv）+ 单测 | 半天 |
| 3 | 写 `AxialDiTBlock_v2`（AdaLN-Zero × 4 段、attn_bias 接口）+ 单测前向 shape | 半天 |
| 4 | 写 `PatchTriangleUpdate` + 单测 | 半天 |
| 5 | 写 `CondAttentionBias` + `ControlInjectMLP` | 半天 |
| 6 | 拼装 v2 model + config + 启动 | 半天 |
| 7 | 60 epoch 训练 + 监控 | 数小时 |
| 8 | 评估 + 与 v1 对比 + 写报告 | 半天 |

合计开发约 **3 天**，加训练等约 **4-5 天**出结果。

---

## 5. 路线 2 的故事归一

**v2 是一个完整的 SymFold-style DiT 升级版**，而非主线 RNAformer 的复刻：

| 维度 | v1 SymFlow | **v2 SymFlow** | 主线 PriFold |
| --- | --- | --- | --- |
| 骨架 | DiT (patch + AdaLN) | **DiT (patch + AdaLN-Zero × 4 段)** | RNAformer (全分辨率轴向 transformer) |
| pair feature | outer concat | **多层 attention map** | outer concat |
| pair-pair 关系 | 无 | **Patch Triangle Update** | TriangleAttention 框架 |
| pos_bias 用法 | channel | **attention bias** | attention bias |
| 生成 / 判别 | 生成式 (flow) | **生成式 (flow)** | 判别式 |
| 计算复杂度 | O((L/4)²) | **O((L/4)²)** | O(L²) |

故事一句话：**"我们用 SymFold 的 DiT + flow matching 框架做 RNA 二级结构，在 DiT block 内引入 RNA 特有的多层语言模型 attention map、patch triangle update 和 pair-bias attention，使得生成式模型在 contact map 任务上达到与 SOTA 判别式方法可比的精度"。**

---

## 6. 参考

- DiT (Peebles & Xie, ICCV 2023): AdaLN-Zero modulation
- AlphaFold 2 (Jumper et al., Nature 2021): triangle update / pair attention with bias
- RNA-FM (Chen et al., 2022): contact prediction from multi-layer attention maps + APC
- Discrete Flow Matching (NeurIPS 2024): CTMC-based discrete generation
- ControlNet (Zhang et al., ICCV 2023): zero-init conditioning injection
