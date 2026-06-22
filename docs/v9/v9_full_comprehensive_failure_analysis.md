# v9_full 全面分析：为什么成功、为什么到不了更高水平

> 生成时间：2026-06-22  
> 对象：`v9_full` / `v9_ddp`  
> 配置：`symfold/config/v9/v9_ddp.json`  
> 模型：`symfold/v9/model.py::DensityNetProPlus`  
> 输出：`symfold/outputs/v9_ddp/`  
> 核心结果：best Val F1 = **0.6814 @ epoch 160**，Test F1 = **0.6961**。

---

## 1. 一句话结论

`v9_full` 是目前最成功的一版，因为它把 RNA contact prediction 从“会严重过预测/采样不稳”的问题，推进到了一个 **数量基本可控、长序列不崩、P/R 相对均衡** 的判别式模型。

但它上不去更高水平的根因不是单一超参，而是四个层面的共同瓶颈：

1. **表示瓶颈**：MARS 完全冻结，下游只有约 5M 可训参数，无法适配 RFAM 长尾结构。
2. **数据长尾瓶颈**：RFAM 占测试主体且内部高度多样，F1=0 几乎都来自 RFAM。
3. **结构解码瓶颈**：推理是 score threshold + density budget + greedy top-k，没有显式保证一碱基一配对、非交叉、stem 连续性等结构约束。
4. **训练目标与最终 F1 不完全一致**：训练 loss 后期继续下降，但 val/test F1 平台化，说明 BCE/OHEM/FP/shift 等局部目标已经不能继续转化为离散结构质量提升。

所以 v9 的失败不是“没训练好”，而是 **当前范式已经接近上限**：冻结 LM 特征 + 小型 axial pair head + 手工 loss + 贪心 top-k 解码，能到 0.69 左右，但很难自然冲到 0.75+。

---

## 2. v9_full 到底成功在哪里

### 2.1 总体指标明显优于 v7/v8

| 模型 | Test F1 | 说明 |
|---|---:|---|
| v7 | 0.6538 | 纯判别式 DensityNet |
| v8 | 0.6105 | OHEM/FP/shift 等组合不理想 |
| **v9_full** | **0.6961** | 当前最佳 |
| 官方 baseline | 0.7700 | RNAformer 量级模型 |

v9 相比 v7 提升约 **+4.2 pp**，相比 v8 修复了明显退化。

### 2.2 过预测问题基本被控制

| 指标 | 值 |
|---|---:|
| Precision | 0.6917 |
| Recall | 0.7186 |
| Pred/GT ratio | 1.133 |
| 平均 GT pairs | 31.09 |
| 平均 Pred pairs | 30.75 |

v7 时代最大问题是过预测，Pred/GT ratio 曾接近 1.9；v9 已经压到 1.13 左右，说明 `pair_count`、`ratio_penalty`、`density_head`、`FP penalty` 和 top-k budget 的组合是有效的。

### 2.3 RoPE 是最关键成功因素

消融结果已经说明：

| 模型 | RoPE | 当前 Test F1 |
|---|---:|---:|
| `v9_full` | ON | 0.6961 |
| `v9_no_rope` | OFF | 0.5770 @ e59 |
| `v9_low_reg` | ON | 0.6804 @ e59 |

关闭 RoPE 后掉点非常大，说明 v9 的主要能力来自 pair matrix 上的 2D 相对位置建模，而不是单纯靠正则或 loss 堆出来。

### 2.4 对经典结构学得很好

按 bpRNA 来源前缀拆分：

| 来源 | N | F1 | Bad rate | 特点 |
|---|---:|---:|---:|---|
| RFAM | 1129 | 0.6691 | 10.5% | 主体，长尾多样，最难 |
| CRW | 99 | **0.9346** | 1.0% | 结构规范，学得很好 |
| SPR | 24 | **0.9560** | 0.0% | 短小规范结构 |
| tmRNA | 23 | 0.7354 | 0.0% | 长序列但模式相对稳定 |
| RNP | 17 | 0.7546 | 0.0% | 长序列，recall 偏低 |
| SRP | 11 | 0.5811 | 18.2% | 特殊结构，样本少 |

这说明模型不是整体无效。它对规范、常见、模式稳定的 RNA 已经接近解决；真正拖后腿的是 RFAM 内部长尾和特殊结构。

---

## 3. 训练曲线说明了什么

### 3.1 Train loss 持续下降，但 Val F1 后期平台化

| Epoch | Train loss | BCE | FP penalty | Shift | Val F1 | Val P | Val R |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 21.2586 | 13.1904 | 0.0196 | -0.0001 | 0.0008 | 0.0008 | 0.0008 |
| 19 | 6.1020 | 3.4203 | 1.1769 | -0.2591 | 0.5676 | 0.5538 | 0.6173 |
| 59 | 4.0092 | 2.0729 | 1.1937 | -0.3516 | 0.6317 | 0.6325 | 0.6585 |
| 99 | 3.0276 | 1.4840 | 1.1775 | -0.4133 | 0.6556 | 0.6536 | 0.6801 |
| 139 | 2.4837 | 1.1864 | 1.1685 | -0.4383 | 0.6725 | 0.6771 | 0.6880 |
| 160 | 2.3484 | 1.0318 | 1.1615 | -0.4578 | **0.6814** | 0.6779 | 0.7054 |
| 182 | **2.2729** | 1.0385 | 1.1487 | -0.4534 | 0.6780 | 0.6777 | 0.6975 |

训练分阶段看：

| 区间 | Val F1 变化 | Loss 变化 | 判断 |
|---|---:|---:|---|
| e0-e19 | 0.0008 → 0.5676 | 21.26 → 6.10 | 快速学会基本结构 |
| e20-e59 | 0.5665 → 0.6317 | 6.03 → 4.01 | 主要收益期 |
| e60-e99 | 0.6327 → 0.6556 | 3.95 → 3.03 | 继续提升但变慢 |
| e100-e139 | 0.6617 → 0.6725 | 2.96 → 2.48 | 接近平台 |
| e140-e182 | 0.6746 → 0.6780 | 2.46 → 2.27 | loss 继续降，F1 基本不涨 |

关键现象：**epoch 160 后 train loss 继续下降，但 Val F1 从 0.6814 降到 0.6780。**

这不是严重过拟合，因为 val 没有崩；更像是：

- 模型继续优化连续概率/局部 BCE；
- 但最终评估依赖 `threshold + budget + top-k` 产生的离散 contact map；
- 概率微调不能继续转化为更多正确配对；
- 后期进入“loss-F1 mismatch”的平台期。

### 3.2 当前没有 train F1，无法判断纯 train ceiling

`history.json` 记录了 train loss 和 val F1，但没有 train F1。严格说，我们不能仅凭 loss 判断 train set 是否也已经到上限。

建议补一个诊断：用 `best.pt` 对 train subset/完整 train 跑同样的 `predict`，统计 train F1、train bad rate、train RFAM bad rate。如果 train F1 也不高，说明是容量/表示不足；如果 train F1 很高而 val/test 低，才是明显泛化问题。

---

## 4. Test 具体失败表现

### 4.1 分布不是均匀差，而是长尾 bad case 拉低均值

| 指标 | 值 |
|---|---:|
| Mean F1 | 0.6961 |
| Median F1 | **0.7692** |
| Q25 | 0.5623 |
| Q75 | 0.8936 |
| F1 std | 0.2572 |
| Bad rate F1<0.3 | **9.4%** / 122 samples |
| F1=0 | **3.5%** / 46 samples |

这说明大多数样本不是很差：58.6% 的样本 F1 ≥ 0.7，25% 的样本 F1 ≥ 0.8936。真正的问题是左尾：少数样本完全失败或半失败。

按 F1 档位拆分：

| 档位 | N | 占比 | 平均 F1 | total FP | total FN | Pred/GT |
|---|---:|---:|---:|---:|---:|---:|
| Bad <0.3 | 122 | 9.4% | 0.1165 | 2977 | 2736 | 1.076 |
| Mid 0.3-0.7 | 418 | 32.1% | 0.5455 | 6078 | 6176 | 0.993 |
| Good ≥0.7 | 763 | 58.6% | 0.8712 | 3106 | 3697 | 0.975 |

注意：bad case 的 Pred/GT 约 1.08，很多不是数量错，而是 **位置错**。

### 4.2 最危险失败模式：数量接近但位置完全错

F1=0 样本有 46 个，其中 45 个来自 RFAM。典型最差样本：

| Name | L | GT | Pred | TP | FP | FN | 现象 |
|---|---:|---:|---:|---:|---:|---:|---|
| `bpRNA_RFAM_41659` | 192 | 56 | 39 | 0 | 39 | 56 | 数量不离谱，位置全错 |
| `bpRNA_RFAM_3116` | 179 | 47 | 45 | 0 | 45 | 47 | 数量几乎正确，位置全错 |
| `bpRNA_RFAM_36748` | 133 | 36 | 34 | 0 | 34 | 36 | 数量接近，位置全错 |
| `bpRNA_RFAM_21346` | 98 | 30 | 30 | 0 | 30 | 30 | 数量完全正确，位置全错 |

这类 case 是最本质的失败：density head/top-k 估对了“有多少对”，但 pairwise score map 把结构 pattern 放在了错误位置。

这说明 v9 已经学会了“RNA 通常应该有多少配对”，但没有学会某些 RFAM 家族的“具体怎么折”。

### 4.3 长度维度：中等长度是主瓶颈，最长序列 recall 不足

| 长度 | N | F1 | Precision | Recall | Bad rate | Pred/GT |
|---|---:|---:|---:|---:|---:|---:|
| 0-100 | 571 | **0.7481** | 0.7282 | 0.7906 | 8.6% | 1.087 |
| 100-200 | 532 | 0.6536 | 0.6477 | 0.6746 | 10.9% | 1.034 |
| 200-300 | 97 | 0.6427 | 0.6677 | 0.6271 | 8.2% | 0.937 |
| 300-400 | 76 | 0.7000 | **0.7566** | 0.6604 | 5.3% | 0.861 |
| 400-500 | 27 | 0.6124 | 0.6934 | **0.5524** | 11.1% | 0.795 |

表现很具体：

- 0-100 最好，短结构容易。
- 100-300 是最大样本区间，也是主要瓶颈。
- 300-400 因 RoPE 表现不错，但 recall 仍不高。
- 400-500 明显欠预测，Pred/GT 只有 0.795，Recall 只有 0.5524。

所以 v9 的长序列问题不是完全没解决，而是：RoPE 让长距离建模可用，但 budget/top-k 和模型容量仍让最长序列漏配对。

### 4.4 稀疏/低 GT pairs 样本仍然不稳

按 GT pairs：

| GT pairs | N | F1 | Precision | Recall | Bad rate | Pred/GT |
|---|---:|---:|---:|---:|---:|---:|
| 0-5 | 52 | 0.6066 | 0.5489 | 0.7532 | **26.9%** | **2.333** |
| 5-10 | 54 | 0.6373 | 0.6411 | 0.6899 | 16.7% | 1.216 |
| 10-20 | 255 | 0.6614 | 0.6287 | 0.7242 | 12.9% | 1.220 |
| 20-40 | 671 | **0.7287** | 0.7190 | 0.7468 | 6.9% | 1.041 |
| 40-80 | 186 | 0.6620 | 0.6915 | 0.6396 | 8.1% | 0.925 |
| 80+ | 85 | 0.7088 | 0.7861 | 0.6483 | 5.9% | 0.818 |

极少配对样本会过预测；高配对样本会欠预测。也就是说，density/budget 的平均效果不错，但在两端失真。

按预测数量误差拆分：

| 类别 | N | F1 | Precision | Recall | Bad rate |
|---|---:|---:|---:|---:|---:|
| 欠预测 `<0.7×GT` | 46 | 0.4745 | 0.6439 | 0.3826 | 23.9% |
| 数量合理 `0.7-1.3×GT` | 1075 | **0.7355** | 0.7384 | 0.7386 | 6.6% |
| 过预测 `>1.3×GT` | 182 | 0.5195 | 0.4281 | 0.6847 | 22.0% |

这直接说明：**只要数量控制失衡，F1 会明显掉。** 当前 budget 是平均意义上有效，但没有做到 per-sample 精准。

---

## 5. 追根溯源：为什么上不去

### 原因 A：MARS 完全冻结，表示无法适配结构预测

v9 使用 MARS-LX 提供 hidden states 和 attention map，但 `freeze_mars=true`。训练只更新下游 pair head。

这带来一个上限：

- 如果 MARS attention 对某些 RNA family 已经含有结构线索，v9 能学得很好，例如 CRW/SPR。
- 如果 MARS 表示对某些 RFAM/SRP 家族没有足够结构信息，下游 5M 参数只能“猜常见模式”。
- F1=0 但 Pred/GT 接近 1 的样本说明模型不是不会估数量，而是特征不足以定位正确配对。

这也是为什么 v10 的方向理论上合理：让 MARS 后层适配结构任务。但 v10 不能直接粗暴 unfreeze，因为显存、batch 多样性和 warm-start 都会影响稳定性。

### 原因 B：RFAM 长尾是主要失败来源

测试集中 RFAM 占 1129/1303，F1 只有 0.6691；CRW/SPR 已经 0.93+。F1=0 的 46 个样本中，45 个来自 RFAM。

这说明提升平均 F1 的关键不是继续优化 CRW/tRNA，而是解决 RFAM 内部长尾。

根因包括：

1. RFAM 家族多，结构模式差异大。
2. 很多 family 在训练集中样本少，梯度被高频 family 淹没。
3. 当前 sampler 主要按长度分桶，不按 family 平衡。
4. loss 权重按 density 调整，但不按 family/难例调整。

### 原因 C：推理解码过于贪心，结构约束不足

v9 推理流程是：

```text
score map -> score_threshold=0.43 -> density budget -> top-k -> 对称化
```

代码层面并没有显式保证：

- 一个碱基最多配一个；
- stem 应该连续；
- hairpin loop 最小长度之外的更多生物约束；
- 非交叉或可控 pseudoknot；
- canonical / wobble / non-canonical 的可调规则。

当前 top-k 只选择高分 pair。它能控制总数，但不能保证全局结构最优。对于局部有多个相近高分候选的区域，top-k 很容易选出冲突配对或把 stem 整体偏移。

这解释了两类现象：

- 数量对但位置错；
- 长序列高 precision、低 recall，budget 下只取了局部最自信的 pair，漏掉远端/弱信号 pair。

### 原因 D：loss 与最终离散 F1 存在错位

训练 loss 包含 BCE/OHEM、Dice、DST、pair count、density、FP penalty、shift relief。它们分别有效，但最终评估是离散 contact map F1。

后期现象是：loss 继续下降，Val F1 不涨。这说明模型在优化：

- 概率校准；
- hard negative BCE；
- FP 的连续惩罚；
- density MSE；

但这些不一定改变 top-k 排名，也不一定让错误样本从 0 TP 变成有 TP。

特别是 v9 的关键错误不是“所有预测都差一点”，而是“某些样本整体结构位置错”。这种错误靠局部 BCE 很难修。

### 原因 E：模型结构缺少更强的几何/三角关系建模

v9 是 8 层 axial transformer，pair 特征来自：

- MARS 1D hidden outer concat；
- MARS attention map；
- sequence pair one-hot；
- RoPE 注入相对位置。

这比 v7/v8 强，但仍然不像 RNAformer/AlphaFold/Evoformer 那样显式建模 pair-pair 之间的三角关系、stem 连续性、局部 motif 和全局结构一致性。

RNA 二级结构不是独立 pair 分类。一个 pair 是否成立依赖：

- 相邻 pair 是否支持一个 stem；
- 同一碱基是否已被占用；
- 是否与其他 pair 冲突；
- loop/junction 的整体结构是否合理。

v9 的 axial attention 能间接学一部分，但没有强 inductive bias，所以在罕见结构上泛化不足。

### 原因 F：正则有效但不是主瓶颈

`v9_low_reg` 当前 Test F1 已到 0.6804，说明低正则也能学得不错。强正则相对低正则约 +1.6 pp，但不是决定性因素。

因此继续盲目加 dropout/drop_path 不太可能突破 0.72。更重要的是表示、结构解码和长尾训练。

---

## 6. 为什么说 v9 不是简单失败

v9 的失败很“有层次”：

1. **它成功解决了平均数量控制**：Pred/GT ratio 接近 1。
2. **它成功解决了部分长序列问题**：300-400 F1 达到 0.7000。
3. **它成功解决了经典结构**：CRW/SPR 接近 0.95。
4. **它失败在 RFAM 长尾和特殊结构**：F1=0 几乎全来自 RFAM。
5. **它失败在结构定位而非纯数量估计**：很多 bad case 数量合理但位置全错。

所以 v9 的上限不是“训练坏了”，而是“模型已经学到了通用 RNA fold prior，但对罕见 family 的条件化不够”。

---

## 7. 改进思路：按优先级

### 7.1 先做无需重训的诊断和推理优化

#### A. 补 train/val/test 同口径分析

当前缺 train F1。建议对 train subset/full train 跑：

- overall F1；
- RFAM vs non-RFAM；
- length bins；
- gt_pairs/density bins；
- F1=0 cases；
- density_pred vs gt_pairs 误差。

目的：判断上限来自 train 也学不会，还是泛化差。

#### B. 做 inference sweep

对 `score_threshold`、`length_decay`、`budget_floor`、budget multiplier 做 sweep：

```text
score_threshold: 0.35, 0.38, 0.40, 0.43, 0.45, 0.48
length_decay:    0.00, 0.10, 0.15, 0.20
budget_floor:    0.6, 0.7, 0.8, 1.0
budget_scale:    0.9, 1.0, 1.1, 1.2
```

重点看：

- 400-500 recall 是否能提高；
- 0-5 GT pairs 是否过预测更严重；
- RFAM bad rate 是否下降。

#### C. 加结构约束解码

比 top-k 更合理的解码方式：

1. **matching decoder**：最大权匹配，保证一个碱基最多配一个；
2. **Nussinov-style DP**：适合非 pseudoknot 二级结构；
3. **ILP / beam search**：允许有限 pseudoknot，但加冲突约束；
4. **stem-aware decoder**：奖励连续 stem，而不是孤立 pair。

这是最值得优先尝试的低成本方向，因为不需要重训模型，只替换 `predict` 后处理。

---

### 7.2 训练策略改进

#### A. v10b：从 v9 warm-start，轻量适配 MARS

不建议再次粗暴 unfreeze 2 层从头训。更稳的是：

```text
base: v9 best.pt warm-start
MARS: unfreeze last 1 layer 或 LoRA/Adapter
LR: downstream 5e-4, MARS/LoRA 1e-5 ~ 3e-5
warmup: 10-15 epochs
max_sq_tokens: 尽量恢复到 400K-600K
```

目标不是让 MARS 大幅漂移，而是让最后表示对结构任务做小幅适配。

#### B. Family-aware / hard-case training

当前 RFAM 长尾是主瓶颈，应该显式处理：

- 按 `bpRNA_RFAM_xxx` 的 family/source 做 balanced sampler；
- 对 bad case family replay；
- 对 F1<0.3 或 RFAM 稀有家族加权；
- 先训 common family，再 curriculum 加 hard RFAM；
- 或反过来 fine-tune 阶段只采 hard RFAM/SRP。

对应长尾学习领域方法：

- Class-Balanced Loss；
- LDAM / DRW；
- Focal loss for long-tail；
- hard example replay；
- group DRO。

#### C. 加 ranking/listwise loss，让训练目标更接近 top-k

现在训练主要是逐点 BCE，但推理是 top-k 排名。可以借鉴推荐系统/检索排序：

- pairwise ranking loss：GT pair score 应高于 negative pair；
- listwise top-k loss：优化整条 RNA 内候选 pair 排名；
- LambdaRank / soft top-k；
- AUC / AP surrogate loss；
- 对每个 GT pair 的 hardest negative 做 margin。

这可能比继续调 BCE 权重更有效。

#### D. 加结构级辅助任务

建议增加辅助 head：

1. 每个 nucleotide 是否 paired；
2. 每个 nucleotide 的 pairing partner distribution；
3. stem start/end prediction；
4. pair distance bin；
5. loop/junction/motif 分类；
6. confidence head，预测 per-sample F1 或 per-pair correctness。

这些任务能让模型学到“结构组织”，不只是 pair 点分类。

---

### 7.3 模型结构改进

#### A. 引入 RNAformer / Evoformer 风格 pair block

从蛋白结构预测借鉴：

- triangle attention；
- triangle multiplicative update；
- pair transition；
- recycling；
- distogram/contact multi-task。

v9 目前只有 axial row/col attention，缺少三角一致性。RNA pair map 也需要 pair-pair 几何关系。

#### B. 多尺度结构建模

从语义分割借鉴：

- U-Net / FPN 多尺度 pair map；
- dilated convolution 捕捉 stem 和 long-range pattern；
- local window + global attention 混合；
- boundary/stem continuity loss。

这对 100-300 主瓶颈区间可能尤其有用。

#### C. 结构约束内嵌模型

从结构化预测借鉴：

- CRF over pairs；
- differentiable matching / Sinkhorn；
- differentiable Nussinov；
- energy-based reranker；
- neural proposal + symbolic decoder。

目标是把“合法 RNA 结构”的先验放进训练或解码，而不是只靠 top-k 后处理。

---

## 8. 可以借鉴的其他领域训练方法

### 8.1 目标检测

RNA contact map 类似极度稀疏目标检测。

可借鉴：

- Focal Loss / Quality Focal Loss；
- hard negative mining；
- positive assignment 策略；
- NMS/Soft-NMS 类后处理；
- teacher-student pseudo label；
- box/point ranking loss。

对应到 RNA：GT pair 是 sparse object，负样本巨大，不能只靠 BCE。

### 8.2 语义分割 / 医学图像分割

contact map 是稀疏二值 mask。

可借鉴：

- Dice / Tversky / Focal Tversky；
- Lovasz loss，直接优化 IoU/F1 surrogate；
- boundary loss；
- deep supervision；
- U-Net/FPN 多尺度特征；
- hard region mining。

对应到 RNA：stem 边界、稀疏区域和难样本都类似医学小目标分割。

### 8.3 推荐系统 / 信息检索排序

推理本质是从候选 pair 中选 top-k。

可借鉴：

- pairwise ranking；
- listwise ranking；
- LambdaRank；
- differentiable top-k；
- calibration-aware ranking。

对应到 RNA：让 GT pair 排在所有 false pair 前面，比让每个点单独 BCE 更贴近最终目标。

### 8.4 蛋白结构预测

可借鉴 AlphaFold/RoseTTAFold/RNAformer：

- triangle update；
- recycling；
- pair representation refinement；
- confidence head；
- distogram/contact multi-task；
- template/structure prior integration。

对应到 RNA：pair map 不是独立像素，pair 之间有强几何关系。

### 8.5 长尾分类 / domain generalization

RFAM 长尾是典型 long-tail/domain generalization 问题。

可借鉴：

- class-balanced sampler；
- group DRO；
- reweight by effective number；
- deferred reweighting；
- hard example replay；
- meta reweighting。

对应到 RNA：不要让高频结构模式淹没稀有 family。

### 8.6 大模型微调

MARS 是冻结 LM，当前适配不足。

可借鉴：

- LoRA；
- Adapter；
- BitFit；
- discriminative layer-wise LR；
- gradual unfreezing；
- warm-start fine-tune；
- gradient checkpointing 换显存。

对应到 RNA：让 MARS 后层轻量适配结构任务，而不是从头训练巨大下游头。

---

## 9. 推荐下一步实验路线

### 第一阶段：诊断 + 无重训提升

1. 跑 train subset eval，补 train F1。
2. 对 v9 best 做 inference sweep。
3. 实现 matching/Nussinov/ILP decoder，对比原 top-k。
4. 输出 RFAM bad case score map，可视化“数量对但位置错”的具体模式。

预期：如果 decoder 有效，可能无重训提升 0.5-2 pp，并明显改善冲突/过预测。

### 第二阶段：v9.5 训练微调

1. 保持 v9 架构，加入 family-aware sampler。
2. 加 ranking loss 或 soft top-k loss。
3. hard RFAM fine-tune 20-40 epoch。
4. 保留 RoPE 和强正则。

目标：降低 RFAM bad rate，从 10.5% 压到 7-8%。

### 第三阶段：v10b/v11 结构升级

1. 从 v9 best warm-start。
2. MARS last-1-layer unfreeze 或 LoRA。
3. 加 triangle pair block / multi-scale pair block。
4. 使用结构约束 decoder。

目标：突破 0.71，再冲 0.73+。

---

## 10. 最终判断

`v9_full` 的成功来自：

- 判别式路线正确；
- RoPE 对 pair matrix 很关键；
- density budget 修复了过预测；
- shift/DST/正则改善了稳定性；
- 对规范结构已经学得很好。

`v9_full` 的失败来自：

- 冻结 MARS 表示对 RFAM 长尾不够；
- 下游 pair head 缺少更强结构归纳偏置；
- top-k 解码不是真正的 RNA 结构解码；
- loss 后期优化不能继续转化为离散 F1；
- bad case 主要是“数量对但位置错”，不是简单阈值问题。

如果要继续提升，优先级应该是：

```text
结构解码约束 > RFAM/hard-case 训练 > MARS 轻量适配 > 更强 pair block > 细调正则
```

不要把主要精力放在继续微调 dropout 或单个 loss 权重上；它们能带来小收益，但无法解决 v9 的根本上限。

---

## 11. 补充深入分析（2026-06-22 第二轮）

### 11.1 训练效率递减的量化

| 阶段 | Epochs | Val F1 增量/epoch | 总增量 | Loss 下降 | 判断 |
|---|---:|---:|---:|---:|---|
| e0-19 | 20 | **0.0298** | +0.567 | -15.16 | 爆发期 |
| e20-59 | 40 | 0.0017 | +0.065 | -2.02 | 主收益期 |
| e60-99 | 40 | 0.0006 | +0.023 | -0.92 | 衰减期 |
| e100-139 | 40 | 0.0003 | +0.011 | -0.48 | 平台前沿 |
| e140-182 | 43 | **0.00008** | +0.004 | -0.19 | 完全平台 |

最后 43 个 epoch 总共只带来 **+0.004 Val F1**，接近统计噪声。说明当前学习目标在 e100 后就已经接近饱和——再训 100 epoch 也不会跳到 0.72。

### 11.2 后期 loss-F1 脱钩的本质

post-160 stats:

```text
loss:   2.348 → 2.273  （持续下降）
val_f1: 0.6814 → 0.6780 （微降/震荡）
bce:    ~1.03-1.10 震荡
fp_penalty: 1.15 → 1.15 稳定
```

这说明后期 loss 的下降主要来自：

1. BCE 的概率校准变好（预测更接近 0/1，但排名不变）
2. shift loss 更负（接近 shift 的 FP 被减免更多，但 top-k 不直接用 shift）
3. density MSE 微降（budget 估计已经够好）

**没有一项能改变 top-k 排名**。训练目标和评估指标之间存在不可忽视的 gap。

### 11.3 密度预测精度与 F1=0 的关系

整体 density budget 精度：

| 统计 | 值 |
|---|---|
| pred_pairs 在 GT 的 0.7-1.3x 内 | **82.6%** (1075/1302) |
| 0-100 长度 | 80.4% |
| 100-200 | 82.1% |
| 200-300 | 91.8% |
| 300-500 | 88.3% |

这很好——超过 80% 的样本数量估计合理。

但 F1=0 的 45 个样本中：

| 指标 | 值 |
|---|---|
| 数量在 0.7-1.3x 内 | **46.7%** (21/45) |
| 全部 TP=0 | ✅ 100% |
| avg pred_pairs | 18.9 |
| avg gt_pairs | 16.4 |

**接近一半 F1=0 样本数量估计是对的，但一个 GT pair 都没命中。**

这是最关键的发现：**模型对这些样本的 score map 中，GT pair 位置的分数低于错误位置。** 也就是说模型对这些 RFAM 家族的折叠规则有"误解"——它用在其他家族学到的 pattern 去预测，结果完全偏离。

更强的 decoder 在这种情况下**帮不了**——因为问题在于 score map 本身就是错的，不是后处理不够。

### 11.4 改进空间的上限估算

如果我们能做到：

| 场景 | Bad(122) 平均 F1 | Mid(418) 平均 F1 | Good(763) | 整体 F1 |
|---|---:|---:|---:|---:|
| 当前 | 0.1165 | 0.5455 | 0.8712 | **0.6961** |
| 场景A: bad→0.3, mid→0.65 | 0.3 | 0.65 | 0.8712 | **0.7468** |
| 场景B: bad→0.5, mid→0.65 | 0.5 | 0.65 | 0.8712 | **0.7655** |

**解读**：

- 要达到 baseline 0.77，需要同时让 bad case 平均到 0.5、mid case 平均到 0.65。
- 只优化 mid case（recover 25% FN + remove 33% FP → F1 从 0.545→0.653），不改 bad case，整体约到 0.72。
- 把 122 个 bad case 从 0.12 拉到 0.5 意味着对 RFAM 长尾有质变提升，这靠当前范式很难。

**结论**：最大的提升杠杆在于 mid 档（418 样本、0.3-0.7），它们是"有希望的样本"，改进空间最大。bad 档需要根本性的表示/结构先验升级。

### 11.5 P/R 差距的训练动态

| Epoch | Precision | Recall | Gap(R-P) | pred_pairs |
|---:|---:|---:|---:|---:|
| 20 | 0.5757 | 0.5907 | +0.015 | 28.8 |
| 60 | 0.6418 | 0.6525 | +0.011 | 29.3 |
| 100 | 0.6604 | 0.6862 | +0.026 | 30.0 |
| 140 | 0.6840 | 0.6861 | +0.002 | 28.9 |
| 160 | 0.6779 | 0.7054 | **+0.028** | 30.1 |
| 182 | 0.6777 | 0.6975 | +0.020 | 29.9 |

- 训练中期(e100)和 best(e160) R 显著高于 P，模型偏 recall。
- P/R gap 在 best epoch 达到最大(+2.8 pp)。
- pred_pairs 波动在 28.9-30.1，说明 budget 相对稳定，gap 来自真正的配对准确度。
- **P 一直上不去**：这说明模型的 score map 排名靠前的位置有稳定比例的 FP，top-k 出来始终有一些假阳。

### 11.6 不重训 decoder 验证（2026-06-22）

为验证"更好的解码能否带来增益"，对 `v9_ddp/model/best.pt` 在 bprna-test 上做了 topk vs matching decoder 对比（不重训，n=1303）。脚本：`symfold/eval/decoder_ablation_v9.py`。

| Decoder | F1 | Precision | Recall | MCC |
|---|---:|---:|---:|---:|
| topk（原） | 0.6961 | 0.6917 | 0.7186 | 0.6990 |
| matching（贪心最大权匹配，一碱基一配对） | 0.6954 | 0.6923 | 0.7160 | 0.6982 |
| **Δ** | **-0.0007** | +0.0006 | -0.0026 | -0.0008 |

**结论：matching decoder 几乎无增益（-0.07pp，噪声级别）。**

- matching 只解决"一碱基多配"冲突，确实小幅提升 precision（+0.06pp），但损失 recall（-0.26pp），净效应略负。
- 说明 v9 的 top-k 输出本身**冲突极少**，瓶颈不在解码层。
- 与 11.3 节一致：F1=0 样本近半数量估计正确但 TP=0 —— 问题是 **score map 本身把高分给了错误位置**，换贪心解码救不了"score 本身错"的样本。

**对行动方案的修正**：

1. **decoder 不是杠杆** → matching/Nussinov 等纯后处理优先级下调，不应作为主攻方向。
2. **真正瓶颈是 score map 质量** → 必须从能改变 score 的项入手：ranking loss（改排名）、warm-start + MARS 轻量适配（改表示）、family curriculum（改 RFAM 长尾）。
3. matching decoder 可作为无害的默认解码（略升 precision），但不指望增益。

---

## 12. 汇总行动方案

基于以上所有分析，按 **投入产出比** 排序，以下是完整行动计划：

---

### 阶段一：诊断与推理优化（1-3 天，不重训）

| # | 行动 | 预期效果 | 工作量 |
|---|---|---|---|
| 1 | **补 train F1 eval** — 用 best.pt 对 train subset 跑完整 predict，区分容量瓶颈还是泛化瓶颈 | 诊断方向性 | 0.5 天 |
| 2 | **inference sweep** — 对 threshold/budget/decay 做网格搜索 | +0.3-1.0 pp | 0.5 天 |
| 3 | **实现 matching decoder** — 用最大权匹配替代 greedy top-k，保证一碱基一配对 | +0.5-1.5 pp | 1 天 |
| 4 | **可视化 bad case score map** — 对 F1=0 样本输出 raw score heatmap，确认是 score 彻底错还是 top-k 误选 | 诊断性 | 0.5 天 |

**阶段一目标**：Test F1 0.700→0.710，并明确 v10 主方向。

---

### 阶段二：v10 全面升级（统一版本）

v10 = 从 v9 best warm-start，一次性包含训练策略 + 架构 + 解码全面改进。

| # | 行动 | 预期效果 | 优先级 |
|---|---|---|---|
| 5 | **MARS LoRA/Adapter** — 对 MARS 最后 1-2 层加 LoRA(rank=16-32)，从 v9 best.pt warm-start | +2-3 pp | 核心 |
| 6 | **Family-balanced sampler** — 按 RFAM family 做 over-sample/reweight | 降 bad rate 2-3 pp | 核心 |
| 7 | **Ranking loss** — 在 BCE 基础上加 pairwise margin：`score(GT_pair) > score(hard_neg) + margin` | 改善 mid 档 F1 | 核心 |
| 8 | **Triangle pair update** — 在 axial attention 上加入 triangle multiplicative/attention block | +1-2 pp | 重要 |
| 9 | **Matching / structure decoder** — 用最大权匹配或 differentiable Nussinov 替换 greedy top-k | +1-2 pp | 重要 |
| 10 | **Multi-scale pair** — 2x/4x 下采样 pair map + FPN upsample | 改善 100-300 区间 | 可选 |
| 11 | **Confidence head** — 预测 per-sample 预期 F1，推理时低置信切保守策略 | +0.3-0.5 pp | 可选 |
| 12 | **Hard-case replay** — fine-tune 阶段对 RFAM bad family 加权 | 针对性修复 | 可选 |

**v10 配置核心设计**：

```text
base checkpoint: v9 best.pt (warm-start)
MARS: LoRA rank=16-32, lr=1e-5~3e-5
downstream: lr=5e-4 (继承 v9)
pair block: axial attention + triangle update (8 layers)
decoder: matching decoder (推理) / soft top-k loss (训练)
sampler: family-balanced + hard-case replay
loss: v9 loss + pairwise ranking margin
正则: 保留 v9 强正则 + RoPE
epochs: 100-150 (warm-start 不需要 200)
```

**v10 目标**：Test F1 突破 **0.73**，追近 baseline 0.77。

---

### 不推荐做的事

| 行动 | 为什么不做 |
|---|---|
| 继续调 dropout/drop_path | ablation 已证明正则只贡献约 1.6 pp，不是瓶颈 |
| 继续堆更多 BCE 权重/loss 组件 | 后期 loss-F1 已脱钩 |
| 从头训 MARS unfreeze 2 层 | 之前 v10 已证明粗暴 unfreeze 不如 warm-start |
| 延长 v9 训练到 300+ epoch | 最后 43 epoch 只带来 +0.004，再训无意义 |
| 大幅改变数据增强策略 | v9_low_reg 表明增强对泛化贡献有限 |
| 拆分多个小版本 v9.5/v11 | 统一为 v10，一次到位 |

---

### 核心判断一句话

> **v9 的天花板不是训练不够或超参不对，而是"冻结表示 + 贪心解码 + 逐点 loss"这个范式本身的上限。v10 需要三管齐下一次到位：LoRA 适配 MARS（突破表示瓶颈）+ RFAM 长尾特化训练 + ranking loss（突破训练瓶颈）+ matching/triangle decoder（突破解码瓶颈）。**
