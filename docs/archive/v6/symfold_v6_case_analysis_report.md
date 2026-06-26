# Symfold 实验报告



总结： 之前提到，在训练的后期模型的Loss比较低，导致模型表现不佳。说明对于简单的case，模型已经学会；但是对于复杂的case，模型虽然还在学习，但是监督信号不够强。改进后的版本，对Loss Function做了处理，给复杂案例case的惩罚，以让模型能够随着训练的进行，逐渐适应复杂的case。

经过loss的调整后，模型在bprna上的表现由 F1 = 0.4602 提升至 F1 = 0.621。虽然距离 baseline 0.7700还有距离，但是表明了我们调整 Loss Function做法的有效性。

此外，对 bad case的分析表明，我们在 bprna 数据集中RFAM类的RNA识别效果不佳，RFAM类RNA是bpRNA中多样性较大的RNA。经过分析，我们发现模型往往会预测出一条与Groud Truth有偏移的配对。这表明：模型很自信地预测，导致配对完全偏移。当前的想法是：

1. 对RFAM家族的数据进行增强，以让模型能够更好的识别RFAM类RNA，有效性仍待验证。
2. 生成式模型的噪声太大，考虑转向判别式模型。（该改进较大 需要谨慎考虑）


[TOC]


---

## 1. 总体表现



![](../symfold/outputs/v6_full/training_curves.png)

| 指标 | Val | Test | 合计 |
|------|-----|------|------|
| N | 1299 | 1303 | 2602 |
| F1 | 0.6127 | 0.6206 | 0.6166 |
| Precision | 0.6035 | 0.6072 | 0.6054 |
| Recall | 0.6404 | 0.6507 | 0.6455 |
| pred/gt ratio | 1.171 | 1.198 | 1.185 |
| F1=0 cases | 104 (8.0%) | 97 (7.4%) | 201 (7.7%) |
| F1<0.3 cases | 241 (18.5%) | 252 (19.3%) | 493 (18.9%) |
| F1≥0.9 cases | 306 (23.6%) | 303 (23.3%) | 609 (23.4%) |

**关键发现**：val 和 test 表现一致，模型没有明显过拟合到 val set。

---

### 与 Baseline 的对比

| 指标 | PriFold Baseline | v6 | 差距 |
|------|-----------------|-----|------|
| F1 | **0.7700** | 0.621 | -14.9% |
| Precision | **0.7938** | 0.607 | -18.7% |
| Recall | **0.7623** | 0.651 | -11.1% |
| pred/gt | ~1.0 | 1.198 | 过预测 20% |
| F1=0 cases | ~0 | 97 (7.4%) | 严重 |
| F1<0.3 cases | ~0 | 252 (19.3%) | 严重 |



### 关键观察

1. **F1 差距 15 个百分点**：生成式 flow 方法与判别式方法差距显著
2. **Precision 差距最大（-18.7%）**：flow 采样引入大量 false positive
3. **Recall 差距较小（-11.1%）**：说明 flow 能发现部分正确配对，但位置噪声导致 precision 崩塌
4. **Baseline 几乎无 F1=0 case**：判别式模型不存在"完全预测错位置"的问题
5. **根因**：flow 需要 20 步采样，每步引入位置噪声；而判别式模型单次 forward 直接预测，无噪声累积


---

## 2. bpRNA 数据集分析

bpRNA 的样本按 **RNA 数据来源** 分为 6 个类别：

| 类别 | 全称 | 描述 | Train | Val | Test |
|------|------|------|-------|-----|------|
| **RFAM** | Rfam database | RNA 家族数据库，结构多样 | 9594 (88.7%) | 1146 (88.2%) | 1130 (86.6%) |
| **CRW** | Comparative RNA Web | 比较 RNA 数据库，结构规范 | 669 (6.2%) | 101 (7.8%) | 99 (7.6%) |
| **SRP** | Signal Recognition Particle | 信号识别粒子 RNA | 154 (1.4%) | 17 (1.3%) | 12 (0.9%) |
| **tmRNA** | Transfer-Messenger RNA | 转移-信使 RNA | 145 (1.3%) | 15 (1.2%) | 23 (1.8%) |
| **SPR** | Small non-coding RNA | 小非编码 RNA | 140 (1.3%) | 12 (0.9%) | 24 (1.8%) |
| **RNP** | Ribonucleoprotein | 核糖核蛋白相关 RNA | 112 (1.0%) | 9 (0.7%) | 17 (1.3%) |

**关键观察**：
- RFAM 占据压倒性多数（~88%），是数据集的绝对主体
- 其他 5 个类别加起来仅约 12%
- 各类别在 train/val/test 中比例一致（无分布偏移）

![类别分析总览](../symfold/outputs/v6_full/visualizations/category_analysis.png)

---

## 3. RFAM vs non-RFAM 性能差距

### 各类别表现 

| 类别 | N | F1 | Precision | Recall | pred/gt | F1=0 | F1≥0.9 |
|------|---|-----|-----------|--------|---------|------|--------|
| **RFAM** | 1129 | **0.588** | 0.573 | 0.621 | 1.208 | **96** | 216 |
| **CRW** | 99 | **0.911** | 0.912 | 0.915 | 1.025 | 0 | 76 |
| **SPR** | 24 | **0.938** | 0.929 | 0.950 | 1.042 | 0 | 22 |
| **tmRNA** | 23 | 0.665 | 0.641 | 0.702 | 1.244 | 0 | 0 |
| **RNP** | 17 | 0.686 | 0.679 | 0.703 | 1.038 | 0 | 2 |
| **SRP** | 11 | 0.474 | 0.399 | 0.634 | 2.216 | 1 | 0 |


| | RFAM | 非 RFAM |
|---|------|---------|
| F1 | 0.588 | **0.833** |
| F1=0 cases | 96 (8.5%) | 1 (0.6%) |
| F1≥0.9 | 216 (19%) | 100 (57%) |

**RFAM 和非 RFAM 之间有 24.5 个百分点的 F1 差距。**

**为什么 CRW/SPR 表现好？**

| 特征 | CRW/SPR | RFAM |
|------|---------|------|
| 结构规范性 | 高（rRNA/tRNA 等经典结构） | 低（涵盖上百个 RNA 家族） |
| 结构多样性 | 有限（同类 RNA 结构相似） | 极高（不同家族完全不同） |
| 训练样本充分性 | 训练中有足够同类样本学习 pattern | 很多罕见家族样本不足 |
| 典型配对模式 | 近对角线 stem + 简单环 | 远程 pseudoknot、多域结构 |
| 平均密度 | 0.27-0.29 | 0.22 |


**为什么 RFAM 这么难？**

1. **家族多样性**：Rfam 包含 3000+ RNA 家族（riboswitch、ribozyme、snoRNA、miRNA 等），每个家族有独特的折叠规则
2. **训练数据不均匀**：9594 个 RFAM 训练样本分散在数百个家族中，每个家族可能只有几十个样本
3. **结构复杂性**：包含 pseudoknot、multi-way junction、long-range interaction 等复杂结构
4. **序列-结构映射不唯一**：不同家族的相似序列可能有完全不同的结构


99% 的 F1=0 case 来自 RFAM：
- 模型给出了"合理数量"的配对预测（pred/gt≈1.0）
- 但**没有一个配对位置是正确的**（tp=0）
- 说明模型只学会了"大约该有这么多配对"的统计规律，对这些 RNA 的折叠规则完全无知


---

## 4. 四种失败模式分析

![失败模式分析](../symfold/outputs/v6_full/case_analysis/failure_mode_analysis.png)

通过对 252 个 bad case (test, F1<0.3) 的详细分析，识别出四种主要失败模式：

### Mode 1: 完全位置错误 (38.1%, N=96)

**特征**：F1=0, tp=0（模型预测了配对但没有一个位置正确）

- 平均长度：110，平均密度：0.170（低密度）
- 平均 gt_pairs：18.7，平均 pred_pairs：22.1（数量接近但位置全错）
- **97% 来自 RFAM**

**根因**：模型从未在训练中见过这类 RNA 的结构，预测出"看似合理但完全错误"的结构。

### Mode 2: 正确数量但错误位置 (42.5%, N=107)

**特征**：0.8 < pred/gt < 1.2, 但 F1 < 0.3

- 平均长度：162，平均密度：0.239
- 平均 gt_pairs：38.8，平均 pred_pairs：38.9（几乎完美匹配数量）

**根因**：这是**最重要的失败模式**——模型学会了"正确的密度/数量"但没学会"正确的位置"。较长序列中配对位置的组合空间爆炸，positional reasoning 能力不足。

### Mode 3: 严重过预测 (12.7%, N=32)

**特征**：pred/gt > 2, F1 < 0.5

- 平均长度：150，平均密度：**0.073**（极低密度）
- 平均 gt_pairs：12.1，平均 pred_pairs：34.4（预测了 3x 的配对）

**根因**：低密度 RNA 时模型的 budget 估计偏高，density head 在低密度区域 calibration 不准。

### Mode 4: 其他混合模式 (6.7%)

剩余 bad case 为各种边界情况的混合。

---



## 5. 可视化分析

### Contact Map 可视化

#### Worst 20 Cases

![最差 20 个 case 的 contact map](../symfold/outputs/v6_full/visualizations/worst_20_contact_maps.png)

- **所有 20 个最差 case 都是 RFAM，全部 F1=0**
- GT 的配对模式（蓝色对角线带）在 pred 中完全找不到对应
- 模型预测了弥散配对（浅橙色），但没有命中任何 GT 配对
- diff 图几乎全是红色（FP）和蓝色（FN），无绿色（TP）

**关键观察**：GT 结构往往有**多条平行的对角线带**（多个 stem），但模型预测的配对位置完全偏移。

#### Best 20 Cases

![最好 20 个 case 的 contact map](../symfold/outputs/v6_full/visualizations/best_20_contact_maps.png)

Best 20 全部 F1=1.000：
- 也全是 RFAM（说明 RFAM 中有很多简单 case 模型能完美预测）
- 序列较短（37-70 nt），结构简单：主要是单 stem
- GT 和 Pred 完美重合

#### Score Heatmap 分析

![Score Heatmap 对比](../symfold/outputs/v6_full/visualizations/score_heatmaps.png)

**F1=0 的 case**：模型确实给出了高 score 区域，但**与 GT 完全不重合**——模型是**自信地预测错了位置**。

**F1>0.9 的 case**：Score map 精确对应 GT 配对位置，配对带清晰集中。

### Case Analysis 总览

![Case Analysis 总览](../symfold/outputs/v6_full/case_analysis/case_analysis_overview.png)

---

## 6. 问题根因深度分析（结合 v6 代码）

### 问题 A：为什么 Precision 差距最大（-18.7%）？

**现象**：v6 Precision=0.607 vs Baseline=0.794，差距远大于 Recall（-11.1%）。

**代码层面根因**：

1. **Flow 采样引入系统性位置噪声**

   v6 的推理通过 20 步 tau-leap 采样（`model.py` 的 `sample()` 方法）：
   ```
   x_init ~ Bernoulli(0.005)
   → 20步迭代: rate_01/rate_10 → 以概率 clamp(rate×dt, max=1) 翻转 0→1 或 1→0
   → 最终 x_T 投影为 contact map
   ```
   
   每一步都有随机翻转，20 步累积后 position drift 不可避免。这就是为什么模型预测出的 stem 结构"形状对但位置偏移几个 nt"。

2. **score-first projection 只看分数不看位置连贯性**

   投影步骤（`_project_score`）贪心地取 top-k 高分位置，但不检查选出的配对是否形成连贯的 stem 结构。一个 stem 中偏移 1-2 位的配对都有较高的 score，但 projection 可能选到"偏移版"而不是"正确版"，因为两者 score 差异很小。

3. **pos_weight 极度偏向 Recall**

   ```python
   # discrete_flow.py
   pos_w = pos_weight_min + (pos_weight_base - pos_weight_min) * (1 - pair_per_base/0.5).clamp(0,1)
   # pos_weight_min=10, pos_weight_base=99
   ```
   
   正样本权重 10-99 倍，模型被强烈鼓励"不要漏掉任何配对"。代价是大量 FP——为了不漏一个真配对，模型宁可多预测几个假配对。这直接拉低 Precision。

**原理分析**：

离散 flow matching 的核心假设是：从噪声 `x_0 ~ Bernoulli(ρ₀)` 到目标 `x_1 = GT` 的转移路径是可学习的。但 contact map 是极度稀疏的（density ~0.22，即 99.78% 的位置是 0），这意味着：
- 大部分时间 flow 在学习"不要翻转这个 0→1"
- 极少部分时间在学习"把这个 0 翻转成 1"
- 每步的翻转决策是 per-position 独立的，**不建模 position 之间的相关性**

所以 flow 能学会"这个区域大致应该有配对"，但精确到哪个 (i,j) 位置，信号太弱。

---

### 问题 B：为什么 F1=0 case 有 97 个（7.4%）？

**现象**：97 个 test 样本完全预测错误（tp=0），但 pred_pairs ≈ gt_pairs（数量接近）。

**代码层面根因**：

1. **density head 学会了密度但没学会位置**

   ```python
   # da_se_dit.py — DensityHead
   # 全局平均池化 → MLP → sigmoid → 标量密度
   density_pred = self.density_head(pair_features.mean(dim=(1,2)))
   ```
   
   density head 只看全局统计特征，学会了"这条 RNA 大约有多少配对"，所以 pair_count_loss 和 ratio_penalty 能快速降低。但这不传递任何位置信息。

2. **训练时 density_hint 被 100% dropout**

   ```python
   # model.py forward()
   keep_mask = torch.rand(...) > self.density_hint_dropout  # dropout=1.0 → 全 drop
   density_hint = density_hint * keep_mask
   ```
   
   GT density 完全不参与训练（dropout=100%），density head 必须自己学预测。但推理时 density hint 也不可用（没有 GT），所以这是一致的。

3. **flow 采样对"从未见过的结构"完全无力**

   对于训练集中从未出现过的 RFAM 家族，MARS 提取的特征没有针对该家族的结构先验。flow 头只能从这些无信息特征中"猜"一个结构，而猜的策略是：用在其他家族上学到的 pattern（如"近对角线应该有 stem"），结果放到一个结构完全不同的 RNA 上就全错了。

**原理分析**：

这暴露了 flow matching 的一个根本局限：**生成式模型需要对目标分布的充分采样**。对于只有几个训练样本的稀有 RFAM 家族（如某些 riboswitch），模型从未见过类似的 contact pattern，flow field 在这些区域是"未定义的"——它只能用在高频家族上学到的 prior 去推断，自然全错。

而判别式模型（如 baseline PriFold）不需要学完整分布——它只需要学 `P(contact[i,j]=1 | features)`，对于从未见过的结构，至少可以输出低置信度的预测（不会"自信地全错"）。

---

### 问题 C：为什么 RFAM 和 non-RFAM 差 24.5 个百分点？

**代码层面根因**：

1. **数据分布极度不均**

   RFAM 占 88%，但内部有数百个子家族。CRW/SRP 只有 5 个类别但结构高度一致——模型用 669 个 CRW 训练样本学会了 rRNA 的经典 cloverleaf 结构，F1=0.911。
   
   RFAM 的 9594 个样本分散在数百个家族中，很多家族不到 20 个样本，不够 flow 模型学会对应的转移 field。

2. **LengthBucketBatchSampler 按长度分桶不按家族**

   ```python
   # data.py
   sampler = LengthBucketBatchSampler(lengths, batch_size, shuffle=True)
   ```
   
   一个 batch 里 88% 是 RFAM，但可能来自 50 个不同家族。模型在单个 batch 中看到的每个家族可能只有 1-2 个样本，梯度信号被平均后，稀有家族的 pattern 几乎学不到。

3. **loss 中没有 family-aware 的权重**

   所有样本的 loss 权重相同（除了 pos_weight 根据 density 调节）。一个 RFAM 稀有家族样本的 loss 被 9000 多个其他样本的梯度淹没。

---

### 问题 D：为什么 SRP 表现最差（F1=0.474, pred/gt=2.216）？

**现象**：SRP 只有 11 个 test 样本但 pred/gt=2.216（过预测 2.2 倍），且有 1 个 F1=0。

**代码层面根因**：

1. **训练样本极少**（154 个）且 SRP RNA 结构特殊

   Signal Recognition Particle RNA 有独特的 Alu domain + S domain 两域结构，配对模式与 RFAM 主流模式很不同。154 个训练样本不够模型学会这种特殊结构。

2. **density head 对 SRP 的密度估计严重偏高**

   SRP 的 GT density 偏低（因为两域之间有大段非配对区域），但 density head 根据序列长度和一般统计倾向预测更高的密度 → budget 过大 → 过预测。

3. **Adaptive budget 放大了 density 估计误差**

   ```python
   # model.py sample()
   max_pairs = (density_pred * l_eff * budget_scale).round().long()
   ```
   
   当 density_pred 偏高 2 倍时，max_pairs 也偏高 2 倍，导致 projection 选了太多配对。

---

## 7. 解决方案（已部分在 v7 实施）

### 已实施（v7 DensityNet）

| 方案 | 解决的问题 | 效果 |
|------|-----------|------|
| **判别式架构** | 消除 flow 采样噪声 | F1=0 从 97→52（-46%）|
| **Axial Transformer** | 更好的 pair-level reasoning | F1 +1.7% |
| **DST loss** | 低密度样本过预测 | pred/gt 改善 |
| **BF16 + 更大 batch** | 训练效率 | 速度 ~20x |

### 新增待验证（v7 config 开关）

| 方案 | 解决的问题 | 状态 |
|------|-----------|------|
| **OHEM** | FP 被海量 TN 稀释（问题 A 根因 3） | 消融配置已就绪 |
| **FP penalty** | 正负样本 loss 不对称（问题 A 根因 3） | 消融配置已就绪 |
| **碱基配对约束** | 在不兼容位置预测配对（问题 B） | 消融配置已就绪 |
| **RFAM 家族过采样** | 数据分布不均（问题 C） | 消融配置已就绪 |

### 未实施（未来方向）

1. **Structure-aware projection**：投影时不仅看 score 排序，还检查选出的配对是否形成连贯的 stem（连续 (i,j), (i+1,j-1) 模式）
2. **Multi-scale feature fusion**：MARS 的多层 hidden states 目前只用了 [3,6,9,12] 四层，可以尝试更密的融合
3. **Coevolution features**：引入 MSA-based coevolution 信号（类似 AlphaFold2）
4. **Test-time augmentation**：序列反转/互补后预测取平均，减少位置偏差

