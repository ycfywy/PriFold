# PriFold v7 DensityNet: 全面推理分析报告

> 模型：v7_full best.pt (200 epochs, best val F1=0.6408, test F1=0.6538)

> 数据集：bpRNA train(10807)/val(1299)/test(1303)


---




## 总结


为了更深入地了解 v7 DensityNet 的性能瓶颈，我们对模型在 bpRNA train/val/test 上的表现进行了全面分析，试图回答以下问题：

1. Bad case 与序列长度是否有关？集中在哪个长度区间？
2. Bad case 与配对密度是否有关？低密度样本表现如何？
3. 伪结对模型表现的影响有多大？
4. Bad case 的失败模式有哪些？哪种模式最主要？
5. 同长度/同密度下，成功与失败案例的 Contact Map 有何区别？
6. 数据集本身的非标准配对对模型学习的影响？

### 主要结论

| # | 结论 | 数据支撑 |
|---|------|----------|
| 1 | **泛化 gap 显著且系统性存在** | Train F1=0.909 vs Test F1=0.655（gap 25.4%），所有长度/密度区间均有 gap |
| 2 | **长序列泛化能力严重下降** | 长度 200-400 区间 Test F1 从 ~0.70 跌至 ~0.40，bad case rate 达 20-30% |
| 3 | **低密度样本过预测严重** | density<0.15 时 pred/gt>2.0，F1 仅 ~0.40-0.50；高密度（>0.30）F1 可达 0.75-0.80 |
| 4 | **最差区域：长序列 + 低密度** | length 200-400 + density ~0.10 构成泛化最差的区域（gap ~35%） |
| 5 | **shifted_prediction 是最主要的失败模式** | Val 33.3%、Test 29.4% 的 bad case 为预测偏移（GT ±3 范围内 FP>30%） |
| 6 | **伪结不是主要失败原因** | 伪结对 Val 有轻微负面影响（-5.3%），对 Test 几乎无影响 |
| 7 | **非标准配对是隐藏的性能杀手** | Bad case 中 GT 非标准配对占 23%（全局仅 10%），但模型几乎不预测非标准配对（仅 5.9%），导致大量 FN |



### 待改进 / 待验证

| # | 方向 | 具体措施 | 对应问题 |
|---|------|----------|----------|
| 1 | **降低 DST threshold** | 对更低密度区间做更激进的惩罚。 | P1 |
| 2 | **引入 Shift-aware Loss** | margin loss 或 soft-matching partial credit，使偏移 1 位的惩罚小于完全错 （对目前的 BCE Loss来说，预测的只要是错的，惩罚是一样的，但直觉上发生 shift 的时候，模型应该受到较低的Loss。） | P2 |
| 3 | **增强正则化** | 增大 Dropout / DropPath、更强数据增强、Family balanced sampling 以提高模型的泛化性 | P3 |
| 4 | **非标准配对处理** | 要允许非标准配对 | P4 |
| 5 | **长距离配对建模** | 比如使用ROPE这种基于相对位置的编码，可能可以提高模型处理长序列时的能力。当前我们的模型没有加上位置编码 | P5 |





## 总体表现

| 指标 | Train | Val | Test |
|------|-------|-----|------|
| F1 | 0.9093 | 0.6413 | 0.6546 |
| PRECISION | 0.8778 | 0.6163 | 0.6267 |
| RECALL | 0.9527 | 0.6907 | 0.7035 |
| MCC | 0.9117 | 0.6454 | 0.6579 |
| N | 10807 | 1299 | 1303 |
| F1=0 | 1 (0.0%) | 42 (3.2%) | 39 (3.0%) |
| F1<0.3 | 2 (0.0%) | 186 (14.3%) | 163 (12.5%) |
| pred/gt ratio | 1.106 | 1.231 | 1.261 |






## F1 与序列长度的关系


我们首先统计了在所有数据（train/val/test）上，F1与RNA序列长度之间的关系，并绘制图表。
![F1 vs Length Trend](../../symfold/outputs/v7_full/comprehensive_analysis/f1_vs_length_trend.png)



**关键观察**：
- **Train（蓝）**：全长度范围 F1 > 0.85，长度 200+ 仍保持 ~0.90
- **Val/Test（橙/红）**：长度 > 150 后 F1 明显下降（从 ~0.70 降到 ~0.55）
- **泛化 gap 随长度增大**：短序列（<100）gap ~15%，长序列（300+）gap ~35%

**结论**：在训练集上，模型表现与RNA序列长度关系不大。但是在测试集和验证集上， Length = 300附近｜Length = 400 + 附近，模型有比较明显的泛化性能下降。（从0.6 f1 score 跌至 0.4 f1 score）。序列长度确实对模型泛化性能有影响。



随后，我们想要分析一下，模型的Bad case与序列长度是否有关（Bad case是否集中在某个长度区间），绘制了Bad case数量与RNA序列长度的图像。
![Bad Case Rate](../../symfold/outputs/v7_full/comprehensive_analysis/bad_case_rate.png)

**关键观察**：
- Train 上 bad case rate 几乎为 0（模型完全能学会train set）
- Val/Test 上：长度 > 200 后 bad case rate 快速升到 20-30%

**结论**：长度在200-400的RNA序列构成了主要的Bad case。



---

## F1 与配对密度的关系

下图是F1在 train/val/test 上随样本配对密度的变化图。
![F1 vs Density Trend](../../symfold/outputs/v7_full/comprehensive_analysis/f1_vs_density_trend.png)

**关键观察**：
- **低密度（<0.15）是灾难区**：Val/Test F1 仅 ~0.40-0.50
- **中等密度（0.20-0.30）**：Val/Test F1 在 ~0.65-0.70
- **高密度（>0.30）**：Val/Test F1 可达 ~0.75-0.80
- Train 在所有密度上都 > 0.85

**结论**：配对密度在0.1左右的样本比较难学习，配对密度较高的样本反而F1更高。




此外，我们又分析了多项指标与配对密度之间的关系( precision / recall / (pred/gt))。 如下图所示。
![Metrics vs Density](../../symfold/outputs/v7_full/comprehensive_analysis/metrics_vs_density.png)

**关键观察**：
- 低密度时 pred/gt ratio 极高（>2.0）→ 严重过预测
- 高密度时 pred/gt ≈ 1.0 → 预测数量准确

**结论**：模型在低配对密度的样本上，很容易过预测。


最后，我们按照长度/密度绘制模型在 Train 和 Test 上的泛化差距图，分析在什么样的样本下，模型的泛化能力最差。
![Generalization Gap](../../symfold/outputs/v7_full/comprehensive_analysis/generalization_gap.png)

**关键观察**：
- **长度维度**：gap 在 200-400 区间最大（~30%），短序列 gap 较小
- **密度维度**：低密度 gap ~35%，高密度 gap ~20%


**结论**：长序列( length 200 - 400 ) + 低密度 ( 密度在0.10 左右) = 泛化最差的区域

---



## F1 与伪结的关系

![F1 vs Complexity Trends](../../symfold/outputs/v7_full/comprehensive_analysis/f1_vs_complexity_trends.png)



| 指标 | Train | Val | Test |
|------|-------|-----|------|
| 有伪结样本占比 | 9.2% | 8.2% | 9.9% |
| 有伪结样本平均 F1 | 0.9452 | 0.6898 | 0.6443 |
| 无伪结样本平均 F1 | 0.9056 | 0.6370 | 0.6557 |
| 伪结导致的 F1 下降 | -0.0396 | -0.0528 | +0.0114 |

**结论**：伪结数=15时，模型在测试集和验证集上表现最差。伪结对 Val 有负面影响（-5.3%），但 Test 上几乎无影响。伪结不是主要失败原因。

---


## 对Bad Case 更全面分析

### 失败模式分类

![Failure Mode Summary](../../symfold/outputs/v7_full/comprehensive_analysis/failure_mode_summary.png)

**VAL** (N=186):

| 模式 | 数量 | 占比 | 说明 |
|------|------|------|------|
| shifted_prediction | 62 | 33.3% | 预测偏移（>30% FP 在 GT ±3 范围内） |
| mixed | 45 | 24.2% | 混合/其他 |
| wrong_position | 32 | 17.2% | 数量对位置错（pred/gt≈1 但 F1<0.3） |
| complete_miss | 22 | 11.8% | 完全预测错误（TP=0，无近似对） |
| severe_overpredict | 19 | 10.2% | 严重过预测（pred/gt > 2） |
| pseudoknot_failure | 5 | 2.7% | 伪结导致失败 |

**TEST** (N=163):

| 模式 | 数量 | 占比 | 说明 |
|------|------|------|------|
| mixed | 48 | 29.4% | 混合/其他 |
| shifted_prediction | 48 | 29.4% | 预测偏移 |
| complete_miss | 27 | 16.6% | 完全预测错误 |
| wrong_position | 26 | 16.0% | 数量对位置错 |
| severe_overpredict | 11 | 6.7% | 严重过预测 |
| pseudoknot_failure | 3 | 1.8% | 伪结导致失败 |

**结论**：无论是 VAL 还是 TEST，预测的偏移都是Bad Case出现的主要模式。


---

### 同长度下 Contact Map 对比

**Test 集，长度 100-200：**

![Paired Comparison Test 100-200](../../symfold/outputs/v7_full/comprehensive_analysis/paired_comparison_test_100-200.png)

**Test 集，长度 200-300：**

![Paired Comparison Test 200-300](../../symfold/outputs/v7_full/comprehensive_analysis/paired_comparison_test_200-300.png)

**Test 集，长度 300+：**

![Paired Comparison Test 300+](../../symfold/outputs/v7_full/comprehensive_analysis/paired_comparison_test_300plus.png)


**结论** ： 我尝试寻找一些规律，看了大概100个case，没找出来规律。（）

---

### 同密度下 Contact Map 对比

**Test 集，低密度（density < 0.15）：**

![Density Comparison Test Low](../../symfold/outputs/v7_full/comprehensive_analysis/density_comparison_test_low_lt0.15.png)

**Test 集，中密度（density 0.15-0.30）：**

![Density Comparison Test Mid](../../symfold/outputs/v7_full/comprehensive_analysis/density_comparison_test_mid_0.15-0.30.png)

**Test 集，高密度（density 0.30-0.45）：**

![Density Comparison Test High](../../symfold/outputs/v7_full/comprehensive_analysis/density_comparison_test_high_0.30-0.45.png)

> **注**：超高密度（density ≥ 0.45）区间在 Test 集上不存在同时包含成功（F1≥0.7）和失败（F1<0.3）案例的情况，因此未生成对比图。

**结论**：
- **低密度（<0.15）**：失败案例的 GT 配对数极少（仅几对），模型倾向于过预测——Diff 图中 FP（红）远多于 TP（绿）。成功案例的 GT 结构相对集中在对角线附近，模型能较好匹配。
- **中密度（0.15-0.30）**：失败案例呈现明显的"位置偏移"模式——GT 配对沿对角线分布较规整，但预测散乱，FP 散布整个矩阵。成功案例的配对结构更紧凑、stems 更明确。
- **高密度（0.30-0.45）**：失败案例多为长序列，GT 结构复杂（多 stem、长距离配对），模型预测虽然在近对角线区域有 TP，但远离对角线的配对几乎全部 miss（FN 蓝色显著）。成功案例的配对多集中在近对角线。
- **总体规律**：同密度下，失败案例往往具有更复杂的拓扑结构（多分支、长距离配对、伪结），而成功案例的配对更"局部"、更规整。密度本身不是决定因素，配对的空间分布（局部 vs 全局）才是关键。

---


## 数据分析中的发现




会有很多badcase是这种情况 

![](/symfold/outputs/v7_full/comprehensive_analysis/bad_cases/097_val_F1=0.049_L=107_sample_542_0.png)

![](/symfold/outputs/v7_full/comprehensive_analysis/bad_cases/092_test_F1=0.038_L=70_sample_51_0.png)

![](/symfold/outputs/v7_full/comprehensive_analysis/bad_cases/089_val_F1=0.036_L=86_sample_306_0.png)



还有一些发现就是有的样本本身的Ground Truth就是没有配对，配对密度为0，但是我们预测的是有配对。

![](/symfold/outputs/v7_full/comprehensive_analysis/bad_cases/081_val_F1=0.000_L=36_sample_0_2.png)





### bpRNA数据集非标准配对比例很高

理论上来说，AU GC GU的配对才是合法的，其他配对都是非法的。但是实际上，模型在预测时，会预测出一些非canonical的配对。所以，我们需要分析模型预测的配对是否是canonical的。我们对数据集中的配对做了一个统计，结果如下：

| 指标 | Train (N=10807) | Val (N=1299) | Test (N=1303) |
|------|-----------------|--------------|---------------|
| Total pairs | 331,361 | 39,253 | 40,515 |
| **Canonical (AU/GC/GU)** | 297,167 (**89.7%**) | 35,166 (**89.6%**) | 36,441 (**89.9%**) |
| **Non-canonical** | 34,194 (**10.3%**) | 4,087 (**10.4%**) | 4,074 (**10.1%**) |
| 含 NC 的样本比例 | 7,826/10,807 (**72.4%**) | 938/1,299 (**72.2%**) | 946/1,303 (**72.6%**) |

**Non-canonical 类型 TOP-8（Train）**:

| 类型 | 数量 | 占 NC 比例 |
|------|------|-----------|
| U-U | 4,492 | 13.1% |
| G-G | 3,741 | 10.9% |
| A-C | 3,712 | 10.9% |
| C-A | 3,608 | 10.6% |
| G-A | 3,582 | 10.5% |
| A-G | 3,364 | 9.8% |
| C-U | 3,194 | 9.3% |
| U-C | 2,934 | 8.6% |

**Bad Cases (F1<0.3) 中的配对合法性**：

| 指标 | GT | Predicted | FP (预测错) | FN (漏掉) |
|------|-----|-----------|------------|-----------|
| Total | 9,759 | 12,333 | 10,548 | 7,974 |
| Canonical | 7,490 (76.7%) | 11,605 (94.1%) | 9,877 (93.6%) | 5,762 (72.3%) |
| Non-canonical | 2,269 (23.3%) | 728 (5.9%) | 671 (6.4%) | 2,212 (27.7%) |

**结论**：

1. **bpRNA 数据集中约 10% 的配对是非标准的**（U-U, G-G, A-C 等），72% 的样本至少含有一个非标准配对
2. **Bad cases 中 GT 的非标准配对高达 23%**（远高于全局 10%）→ 非标准配对多的样本更容易失败
3. **模型几乎不预测非标准配对**（仅 5.9%）→ 因此漏掉了大量 GT 中的非标准配对（FN 中 27.7% 是 NC）
4. **FP 中 93.6% 是 canonical**→ 模型预测错的位置大多是合法碱基组合，问题是位置不对而非配对规则不对



---




