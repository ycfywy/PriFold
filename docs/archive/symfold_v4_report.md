# Symfold 实验汇报



---

## 1. 模型架构与特征设计

### 1.1 整体流水线

```
RNA 序列 "AUGCGC..." (长度 L)
    │
    ├─→ MARS-LX 语言模型 (160M, frozen)
    │     ├── 第 3/6/9/12 层 hidden states → [4 × (L, 1056)]
    │     └── 最后 6 层 attention maps    → (6, 12, L, L)
    │
    ├─→ 序列 one-hot 编码              → (L, 4)
    │
    └─→ 碱基配对先验 pos_bias           → (L, L)

         ↓ 特征融合

    DA-SE-DiT-v4 (9层, 12.9M 可训练参数)
         ↓

    flow_logit + direct_logit + density_pred
         ↓ score-first projection

    contact map (L, L)
```

### 1.2 使用的特征

| 特征 | 维度 | 来源 | 作用 |
|---|---|---|---|
| MARS multi-layer hidden | 4×(L, 1056) | MARS 第 3/6/9/12 层 | 捕获不同抽象层级的序列语义 |
| MARS attention maps | (6, 12, L, L) | MARS 最后 6 层 | 提供 co-evolution / co-attention pair 先验 |
| 序列 one-hot | (L, 4) | 输入序列 A/T/G/C | 碱基类型的显式编码 |
| pos_bias | (L, L) | Watson-Crick 配对打分 | 物理先验（A-U=3, G-C=6, G-U=1） |
| x_t (flow state) | (1, L, L) | DFM 当前状态 | 离散 flow 的迭代输入 |
| timestep t | scalar | [0, 1] | flow 进度信息 |
| density hint | scalar | GT 或 predicted | pair 密度先验（训练时 dropout） |

### 1.3 核心模块详解

#### (A) MultiLayerMarsFusion — 多层语言模型特征融合

```
MARS layer 3/6/9/12 hidden → learnable softmax layer weights 加权
                           → 每层独立投影 → concat → MLP 融合
                           → mars_emb_1d (L, 32)
```

**作用**：不同 MARS 层编码不同粒度的信息（浅层=局部模式，深层=长程依赖），学习最优融合权重。

#### (B) MarsAttentionProj — Attention Map 投影

```
MARS attention (6层×12头=72通道) → 对称化 → APC 校正 → 1×1 Conv 投影
                                → mars_attn_2d (16, L, L)
```

**作用**：MARS 的 attention 权重隐含了共进化信息（类似 ESM/RNA-FM 的做法），APC 去除 background bias 后保留真正的配对信号。



---

## 2.  训练效果

### RNAStrAlign 

![v4_rnastralign training curves](../symfold/outputs/v4_rnastralign/training_curves.png)

- **左上 Training Loss**：Phase 1（红）快速下降并收敛；Phase 2 续训（橙）在更低 loss 水平继续优化
- **右上 Val F1/MCC**：Phase 1 达到 0.9459 后被中断；Phase 2（青色）从 best 续训，稳步上升到 **0.9616**
- **右下 LR**：Phase 1 为正常 cosine (8e-5→2e-5)；Phase 2 为独立 cosine (2e-5→4e-6)，红色虚线标记续训分界
- **下方 Test F1**：rnastralign-test 达到 **0.9635**，archiveii-test 达到 **0.8656**，均仍在上升

### bpRNA 

![v4_bprna training curves](../symfold/outputs/v4_bprna/training_curves.png)

**曲线解读**：
- Phase 1 稳步上升到 best F1=0.4609 @ epoch 105
- 在 epoch 107 处 LR 因 config 改动导致跳变（LR 图右上角可见跳升），模型被破坏
- 之后的 epoch 再也没恢复到 best（该问题已在 v4_rnastralign 续训中修复）
- bpRNA 的 val P/R/F1 波动较大，反映任务本身的高难度

### 结果分析

#### RNAStrAlign + ArchiveII

| 测试集 | PriFold Baseline | SymFlow v4 | 差距 |
|---|---:|---:|---|
| rnastralign-test F1 | 0.9738 | **0.9592** | -1.5% |
| rnastralign-test Precision | 0.9742 | 0.9497 | -2.5% |
| rnastralign-test Recall | 0.9744 | 0.9706 | -0.4% |
| archiveii-test F1 | 0.9043 | **0.8656** | -3.9% |
| archiveii-test Precision | 0.9102 | 0.8454 | -6.5% |
| archiveii-test Recall | 0.9037 | 0.8927 | -1.1% |

RNAStrAlign 上 SymFlow v4 与判别式 baseline 的差距仅 **1.5%**，已经非常接近。从 P/R 分布看，v4 的 recall 几乎追平 baseline（差 0.4%），precision 略低（差 2.5%），说明模型倾向于稍微多预测一些 pair，但绝大多数预测是正确的。整体效果令人满意。

ArchiveII 作为 out-of-distribution 测试集，差距稍大（3.9%），主要体现在 precision 不足（-6.5%），即在未见过的 RNA 家族上存在一定的过预测倾向。后续需要专门分析 archiveII 上哪些 RNA 家族/长度区间表现差。

#### bpRNA

| 测试集 | PriFold Baseline | SymFlow v4 | 差距 |
|---|---:|---:|---|
| bprna-test F1 | 0.7700 | **0.4602** | -31% |
| bprna-test Precision | 0.7938 | 0.4065 | -49% |
| bprna-test Recall | 0.7623 | 0.5701 | -25% |

bpRNA 上差距较大。但需要注意：**当前 bpRNA 的训练因中间续训时学习率异常跳变（LR 从 3e-6 突然升至 7.8e-5）导致模型崩溃，训练被迫提前终止**。epoch 105 的 best checkpoint 是在 LR 跳变之前保存的，模型实际上还远未训练充分（从训练曲线可以看到 test F1 在 epoch 90-105 仍在快速上升）。该问题已修复，bpRNA 的续训实验有待后续 GPU 空闲时重新启动，届时有望进一步缩小与 baseline 的差距。


## 3. 下一步工作

1. **分析 ArchiveII 数据集效果不佳的原因**：按 RNA 家族、序列长度分桶进行 bad-case 分析，定位是长序列问题还是特定家族的泛化问题，针对性改进。

2. **bpRNA 实验继续进行**：从 best checkpoint（epoch 105）出发，使用已修复的续训机制（独立小 LR cosine schedule）重新启动训练，观察是否能继续缩小与 baseline 的差距。
