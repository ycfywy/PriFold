# PriFold v7 DensityNet 深度分析报告

> 日期：2026-06-12。模型：v7_full best.pt（200 epochs 完成，best val F1=0.6408）。
> 分析了 train(3009)/val(1299)/test(1303) 共 5611 个 bpRNA 样本。

[TOC]

---

## 1. 总体表现

### 1.1 三阶段对比

| 指标 | Train | Val | Test |
|------|-------|-----|------|
| N | 3009 | 1299 | 1303 |
| **F1** | **0.900** | 0.641 | **0.653** |
| Precision | 0.871 | 0.616 | 0.626 |
| Recall | 0.942 | 0.689 | 0.701 |
| pred/gt ratio | 1.107 | 1.229 | 1.261 |
| F1=0 | 9 (0.3%) | 43 (3.3%) | 40 (3.1%) |
| F1<0.3 | 19 (0.6%) | 190 (14.6%) | 163 (12.5%) |
| F1≥0.9 | 1936 (64%) | 269 (21%) | 270 (21%) |

**关键观察**：
1. **泛化 gap 显著**：Train F1=0.900 vs Test F1=0.653（-24.7%）
2. Train 上 64% 样本 F1≥0.9，说明模型对见过的 RNA 可以很好预测
3. Val/Test 一致（0.641 vs 0.653），无 val overfitting
4. **F1=0 集中在 val/test**：train 仅 9 个(0.3%)，val/test 各 40+(3%)——泛化问题
5. 过预测在 test 上更严重：pred/gt=1.261（比 train 的 1.107 高 14%）

### 1.2 与 Baseline 对比

| 指标 | PriFold Baseline | v7 (test) | 差距 |
|------|-----------------|-----------|------|
| F1 | **0.7700** | 0.6532 | -15.2% |
| Precision | **0.7938** | 0.6258 | -16.8% |
| Recall | **0.7623** | 0.7009 | -6.1% |
| F1=0 | ~0 | 40 (3.1%) | |

- **Recall 差距仅 6%**（模型能找到大部分真配对）
- **Precision 差距 17%**（模型产生太多 FP）
- → 核心瓶颈是 **Precision 不足 / 过预测**

---

## 2. 各 RNA 家族表现

![Per-Family Performance](../symfold/outputs/v7_full/deep_analysis/family_performance_overview.png)

### 2.1 家族指标汇总

| 家族 | N | F1 | Precision | Recall | F1=0 | F1≥0.9 | 特点 |
|------|---|-----|-----------|--------|------|--------|------|
| **RFAM** | 4671 | 0.751 | 0.720 | 0.803 | 86 (1.8%) | 1696 (36%) | 多样性极高，主要困难来源 |
| **CRW** | 715 | **0.969** | 0.965 | 0.974 | 5 (0.7%) | 662 (93%) | rRNA，结构规范 |
| **SPR** | 131 | **0.936** | 0.903 | 0.973 | 0 | 110 (84%) | 小非编码 RNA |
| **RNP** | 26 | 0.794 | 0.788 | 0.807 | 0 | 4 (15%) | 核糖核蛋白 |
| **tmRNA** | 38 | 0.719 | 0.693 | 0.764 | 0 | 2 (5%) | 转移-信使 RNA |
| **SRP** | 30 | 0.629 | 0.610 | 0.654 | 1 (3%) | 1 (3%) | 信号识别粒子 |

**核心发现**：
- **CRW/SPR 接近完美**（F1 > 0.93）：经典结构，训练充分
- **RFAM 拖后腿**（F1=0.751）：包含数百个不同子家族
- **SRP 最差**（F1=0.629）：样本极少（30个）且结构独特（双域）
- **86 个 F1=0 case 中 100% 来自 RFAM**

### 2.2 RFAM 内部分析

RFAM 占总样本 83%（4671/5611），内部表现两极分化：
- 36% 的 RFAM 样本 F1≥0.9（简单结构，如 tRNA, miRNA 前体）
- 1.8% 的 RFAM 样本 F1=0（复杂/罕见家族）

→ 问题不是"RFAM 全差"，而是"RFAM 中的长尾罕见家族完全学不会"

---

## 3. 失败模式深度可视化

### 3.1 Mode A: F1=0 — 完全预测错误

![F1=0 Cases](../symfold/outputs/v7_full/deep_analysis/failure_f1_zero.png)

**特征**（10 个 case）：
- 模型给出高置信度 score（Score heatmap 有明显热区）
- 但热区**与 GT 完全不重合**——模型"自信地错在了别的位置"
- GT 有清晰的反平行对角线带（stem），但 Pred 的带偏移或方向错
- 模型学会了"应该有配对带"，但不知道"带应该在哪里"

**典型 pattern**：
- GT: 位于 (20-40, 60-80) 的 stem
- Pred: 位于 (10-30, 70-90) 的 stem——偏移了 10 nt

### 3.2 Mode B: 数量对但位置错

![Wrong Position](../symfold/outputs/v7_full/deep_analysis/failure_wrong_position.png)

**特征**（10 个 case）：
- pred/gt ∈ [0.7, 1.3]（预测了正确数量的配对）
- 但 F1 < 0.3（位置几乎全错）
- Score heatmap 显示弥散的高分区域，没有集中的配对带
- **这是最大的失败模式**：模型学会了密度但没学会位置

**根因**：pair_count_loss 和 ratio_penalty 只约束数量，不约束位置。对于这些 case，密度相关的 loss 已经很低（≈0），模型没有进一步优化位置的梯度信号。

### 3.3 Mode C: 过预测

![Over-prediction](../symfold/outputs/v7_full/deep_analysis/failure_overpredict.png)

**特征**（10 个 case）：
- pred/gt > 2（预测配对数是 GT 的 2 倍以上）
- GT density 极低（<0.10）
- Score heatmap 几乎全图高分——模型对低密度 RNA 的 density head 校准不准
- Budget 估计过高 → 选了太多配对

**根因**：density head MSE loss 在低密度区域梯度小（0.05² = 0.0025），学习不充分。

### 3.4 Mode D: 欠预测

![Under-prediction](../symfold/outputs/v7_full/deep_analysis/failure_underpredict.png)

**特征**（10 个 case）：
- pred/gt < 0.5（预测太少）
- 模型的 Score heatmap 在正确位置有中等置信度（0.3-0.5），但 threshold=0.4 过滤掉了
- Budget 太小 → 很多正确位置被丢弃

**根因**：score_threshold=0.4 对这些 case 太高。模型对这些 RNA 不够自信，但位置方向是对的。

### 3.5 部分成功（F1 ∈ [0.3, 0.5]）

![Partial Success](../symfold/outputs/v7_full/deep_analysis/partial_success_medium.png)

**特征**：
- Diff 图中有绿色（TP）也有红/蓝（FP/FN）
- 模型预测了部分 stem 正确，但另一些 stem 完全错位或缺失
- 典型 pattern：短的 stem（<5bp）预测正确，长的 stem（>10bp）偏移

---

## 4. Score (概率) 分析

![Score Analysis](../symfold/outputs/v7_full/deep_analysis/score_analysis_detailed.png)

### 4.1 好的 case（F1 > 0.95）

- GT=1 位置的 score 分布：**集中在 0.8-1.0**（高置信度）
- GT=0 位置的 score 分布：**集中在 0.0-0.1**（低置信度）
- → 模型做了很好的区分，阈值 0.4 轻松分开

### 4.2 坏的 case（F1=0）

- GT=1 位置的 score 分布：**分散在 0.1-0.5**（模型不确定）
- GT=0 位置的 score 分布：也有部分在 0.3-0.7（FP 位置有中等 score）
- → 两者严重重叠，阈值 0.4 无法有效分开
- **模型在错误位置给出了和正确位置相似的置信度**

### 4.3 关键洞察

| Case 类型 | GT=1 位置 avg score | GT=0 位置 max score | 可区分性 |
|-----------|--------------------|--------------------|---------|
| F1>0.95 | 0.85+ | <0.2 | 优秀 |
| F1=0 | 0.2-0.4 | 0.3-0.7 | **无法区分** |
| 0.3<F1<0.5 | 0.5-0.7 | 0.2-0.5 | 部分重叠 |

→ 模型的 failure 不是"完全没有信号"，而是"在错误位置有同样强度的信号"。

---

## 5. Per-Stage 可视化

### 5.1 Train 集

- **Best 10**：全部 F1=1.0，短序列（40-70nt），简单 stem 结构

![Train Best 10](../symfold/outputs/v7_full/deep_analysis/train/best_10.png)

- **Worst 10**：F1=0，全是 RFAM 长尾家族

![Train Worst 10](../symfold/outputs/v7_full/deep_analysis/train/worst_10.png)

- **Middle 10**：F1 ∈ [0.5, 0.7]，部分 stem 正确、部分错

![Train Middle 10](../symfold/outputs/v7_full/deep_analysis/train/middle_10.png)

### 5.2 Val 集

![Val Best 10](../symfold/outputs/v7_full/deep_analysis/val/best_10.png)
![Val Worst 10](../symfold/outputs/v7_full/deep_analysis/val/worst_10.png)
![Val Middle 10](../symfold/outputs/v7_full/deep_analysis/val/middle_10.png)

### 5.3 Test 集

![Test Best 10](../symfold/outputs/v7_full/deep_analysis/test/best_10.png)
![Test Worst 10](../symfold/outputs/v7_full/deep_analysis/test/worst_10.png)
![Test Middle 10](../symfold/outputs/v7_full/deep_analysis/test/middle_10.png)

---

## 6. Per-Family 可视化

### 6.1 RFAM

![RFAM Worst 5](../symfold/outputs/v7_full/deep_analysis/family_RFAM_worst5.png)
![RFAM Best 5](../symfold/outputs/v7_full/deep_analysis/family_RFAM_best5.png)

### 6.2 CRW

![CRW Worst 5](../symfold/outputs/v7_full/deep_analysis/family_CRW_worst5.png)
![CRW Best 5](../symfold/outputs/v7_full/deep_analysis/family_CRW_best5.png)

### 6.3 SRP

![SRP Worst 5](../symfold/outputs/v7_full/deep_analysis/family_SRP_worst5.png)
![SRP Best 5](../symfold/outputs/v7_full/deep_analysis/family_SRP_best5.png)

### 6.4 tmRNA

![tmRNA Worst 5](../symfold/outputs/v7_full/deep_analysis/family_tmRNA_worst5.png)
![tmRNA Best 5](../symfold/outputs/v7_full/deep_analysis/family_tmRNA_best5.png)

### 6.5 SPR

![SPR Worst 5](../symfold/outputs/v7_full/deep_analysis/family_SPR_worst5.png)
![SPR Best 5](../symfold/outputs/v7_full/deep_analysis/family_SPR_best5.png)

### 6.6 RNP

![RNP Worst 5](../symfold/outputs/v7_full/deep_analysis/family_RNP_worst5.png)
![RNP Best 5](../symfold/outputs/v7_full/deep_analysis/family_RNP_best5.png)

---

## 7. 问题诊断总结

### 7.1 核心瓶颈排序

| 优先级 | 问题 | 影响范围 | 严重程度 |
|--------|------|---------|---------|
| **P0** | Precision 不足（FP 太多） | 所有 val/test | 与 baseline 差距的主要来源 |
| **P0** | RFAM 长尾家族泛化差 | 86 个 F1=0 + 大量 F1<0.5 | 占 bad case 的 ~100% |
| **P1** | 低密度 RNA 过预测 | density < 0.15 的样本 | pred/gt 严重偏高 |
| **P1** | "位置对数量错"vs"数量对位置错" | F1<0.3 的主要模式 | pair_count loss 无位置信号 |
| **P2** | Score 阈值对中等置信度不友好 | 欠预测案例 | threshold=0.4 可能过高 |

### 7.2 根因链路

```
模型表现差
├── Precision 低（FP 多）
│   ├── pos_weight 偏向 Recall → 模型倾向多预测
│   ├── neg_bce 被 TN 稀释 → FP 不受惩罚 ← [OHEM 解决]
│   └── 不兼容位置也预测 → 生物学不可能的配对 ← [BP compat 解决]
├── RFAM 长尾泛化差
│   ├── 训练数据按家族不均匀 ← [Family balanced 解决]
│   ├── MARS 特征对罕见家族无区分力 → 需要更强的预训练
│   └── 模型容量不够学所有家族 → 需要更大模型或 family-specific adapter
├── 低密度过预测
│   ├── density head MSE loss 梯度小 ← [DST loss 部分解决]
│   └── budget = density × L × 1.1 放大误差 → 需要 conservative budget
└── 位置错误无梯度
    ├── pair_count_loss ≈ 0 当数量匹配 → 无位置信号
    ├── dice loss 的位置敏感性不够 → 需要 position-aware loss
    └── Axial attention 8 层对长程 pair 建模不足 → 需要更深/更强注意力
```

### 7.3 已准备的解决方案

| 方案 | 针对问题 | 配置 | 预期效果 |
|------|---------|------|---------|
| OHEM | FP 被 TN 稀释 | `v7_ohem_only.json` | Precision ↑, F1=0 ↓ |
| FP penalty | FP 惩罚不够 | `v7_fp_penalty_only.json` | Precision ↑ |
| 碱基配对约束 | 不兼容位置预测 | `v7_bp_compat_only.json` | 减少生物学不可能的 FP |
| RFAM 过采样 | 长尾家族训练不足 | `v7_family_balanced_only.json` | RFAM F1 ↑ |
| 全部开启 | 综合 | `v7_all_new.json` | 最大化改善 |

---

## 8. 结论

1. **v7 在 CRW/SPR 上已接近 baseline**（F1>0.93），说明架构能力足够
2. **差距全部来自 RFAM**：86 个 F1=0 + 大量中等 F1，拉低整体
3. **根本问题是 Precision**：Recall 只差 baseline 6%，但 Precision 差 17%
4. **Score 分析揭示**：失败 case 的问题不是"没有信号"而是"在错误位置有同等强度的信号"
5. **过预测控制是关键**：pred/gt=1.26（test），理想值 1.0，需要减少 14% 的 FP

**下一步**：运行消融实验，验证 OHEM + FP penalty + BP compat + Family balanced 各自的贡献。

---

## 9. 数据质量问题：零配对样本

在分析过程中发现，bpRNA 数据集中存在 **7 个 GT 配对数为 0 的样本**（占 0.05%）：

| 样本名 | Split | 长度 | 家族 | 序列 |
|--------|-------|------|------|------|
| bpRNA_CRW_16610 | train | 60 | CRW | AAAGUUUGUAUUGCUAGCUUGGUGGUUAUAGCAUGAGUGAAACACACGAUCCCAUCCCGA |
| bpRNA_CRW_17850 | train | 41 | CRW | UGGGAAUACCAGGUGCUGUAAGCCUUUUCACAGAAUUUUUC |
| bpRNA_RFAM_6490 | train | 52 | RFAM | GAUGAUGAGCCUUCCCCUCACCUGAGUGGUGAUGAGCACACCGGUAGGCUGA |
| bpRNA_RFAM_11464 | train | 59 | RFAM | GAACGAACUUGGCCUGACCUUCAGAAAUGGAGGCAAUACAACUGAUUUAAUGAGCCUGA |
| bpRNA_RFAM_24032 | train | 71 | RFAM | AUCUUUGAUGACCAUUUUUUAAAAUACAAACUAGAGUUUCUGAUUAAUUUAUGAUUUCAAAUUCUUGCUGA |
| bpRNA_CRW_19530 | val | 36 | CRW | UUUGGGAAGUCCUUGUGUUGCAUUCCCUUUUUUGUU |
| bpRNA_SRP_192 | test | 89 | SRP | AAAGUGGUUGGACUUUGUCUUGGANCAGNUGGUUGGGUNCGCCCGCGCAGCACCCGGCCCGNNCAUUNCAAGCCGAGAGGCCGGNNANG |

### 问题分析

这些样本的 contact matrix **全为零**——GT 标注中没有任何碱基配对。可能的原因：

1. **非结构化 RNA 片段**：某些 RNA 在实验条件下确实不形成二级结构（如 linker RNA、降解片段）
2. **标注缺失/错误**：数据来源中该 RNA 的结构信息缺失，被标为全零
3. **N 碱基过多**（如 SRP_192 含多个 N）：结构无法确定

### 对模型的影响

- 训练时：这些样本的 `gt_density=0`，导致 `pos_weight=1/0.01=99`（clamp 到最大值），但所有位置 GT=0，所以 pos_bce=0；density head 学到密度=0；pair_count loss=0。**模型从这些样本学到的是"不要预测任何配对"**
- 如果模型对这些样本预测了配对 → F1=0（因为 tp 永远是 0）
- 这 7 个样本中有 5 个在 train，会给模型"有时候完全不配对"的信号

### 建议

1. **训练时过滤**：在 `build_records` 中过滤掉 `gt_pairs=0` 的样本
2. **或降低权重**：给这些样本极低的 loss weight
3. 数量很少（7/13409 = 0.05%），对整体影响不大，但对 val/test 的 F1=0 统计有影响（val 的 bpRNA_CRW_19530 和 test 的 bpRNA_SRP_192 模型预测了配对就会 F1=0）
