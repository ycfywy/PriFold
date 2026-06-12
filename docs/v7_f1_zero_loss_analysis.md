# v7 DensityNet: F1=0 案例的 Loss 机制分析

> 核心问题：F1=0 意味着预测完全错误，为什么 loss 没有阻止模型做出这样的预测？

[TOC]

---

## 1. 问题描述

在 v6 case analysis 中，约 7-8% 的测试样本 F1=0（tp=0），但 pred_pairs > 0——模型**自信地预测了配对，但没有一个位置正确**。

直觉上：如果 F1=0，所有预测位置都是 FP，loss 应该巨大，模型应该迅速学会避免这种情况。但实际并非如此。

**用户的关键假设**：是否因为我们只加大了"1预测为0"的 loss（即 FN 惩罚，通过高 pos_weight），而没有充分惩罚"0预测为1"（即 FP），导致模型在错误位置自信预测而不受足够惩罚？

---

## 2. 当前 v7 Loss 各组件分析

### 2.1 BCE + Focal + pos_weight

```python
# v7 model.py _compute_loss()
pos_w = (1.0 / gt_density.clamp(min=0.01)).clamp(max=pos_weight_base)  # 最高99
pos_bce = (bce_raw * focal * y * valid * pos_w).sum() / (y * valid).sum()  # 正样本 loss
neg_bce = (bce_raw * focal * (1 - y) * valid).sum() / ((1 - y) * valid).sum()  # 负样本 loss
bce_loss = pos_bce + neg_bce
```

**不对称性分析**：

| 样本类型 | 权重 | 归一化分母 | 说明 |
|----------|------|-----------|------|
| 正样本（GT=1, FN） | **pos_w ≈ 4-100** | Σ(y * valid) = 很少 | 巨大权重，惩罚漏配对 |
| 负样本（GT=0, FP） | **1.0** | Σ((1-y) * valid) = 很多 | 标准权重 |

**关键洞察**：

- 对于 density=0.22 的典型 RFAM 样本（L=100）：
  - 正样本数（GT=1 的位置）：约 22 对 = 44 个位置
  - 负样本数（GT=0 的位置）：L²/2 - 44 ≈ 4956 个位置
  - pos_weight ≈ 1/0.22 ≈ 4.5
  - **正样本总权重贡献**：44 × 4.5 = 198
  - **负样本总权重贡献**：4956 × 1.0 = 4956

看似负样本总权重远大于正样本。但因为分母各自归一化：
- `pos_bce` = average loss on positives × pos_w
- `neg_bce` = average loss on negatives × 1

实际上 **pos_bce** 是被放大了 4.5-100 倍的正样本平均 loss，而 **neg_bce** 只是负样本平均 loss。

---

### 2.2 "预测平移"情况下的 Loss 表现

考虑一个 F1=0 的典型 case：

```
GT:   位置 (10,90), (11,89), (12,88), ...（一个 stem）
Pred: 位置 (15,85), (16,84), (17,83), ...（同样一个 stem，但偏移了 5 个位置）
```

此时的 loss 信号：

| Loss 组件 | 在偏移预测上的值 | 说明 |
|-----------|----------------|------|
| **BCE (pos)** | 高 | GT=1 的位置被预测为 0（FN），有 pos_w 加权 |
| **BCE (neg)** | **中等偏低** | GT=0 但被预测为 1 的位置（FP），仅权重 1.0 |
| **Dice** | 高（接近 1.0） | intersection=0, dice ≈ 1 - 0/(pred+gt) ≈ 1 |
| **DST** | 高 | tp=0, tversky ≈ 0 → loss ≈ 1 |
| **Pair count** | **低** ✨ | pred_pairs ≈ gt_pairs，密度几乎匹配！ |
| **Ratio penalty** | **低** ✨ | pred/gt ≈ 1.0，不触发惩罚 |
| **Density head** | **低** ✨ | density_pred ≈ gt_density |

**问题来了**：

- Pair count loss 和 Ratio penalty 只看"数量对不对"，不看"位置对不对"
- 对于"数量对但位置全错"的 case，这两个 loss 给的信号近乎 0
- BCE (neg) 的信号被稀释了：FP 只有 ~22 个位置错了（在 4956 个负样本中只占 0.4%），平均 neg loss 极低

---

## 3. 核心问题：为什么 Loss 没能阻止 F1=0

### 3.1 根因 1：负样本 loss 被海量 TN 稀释

```
neg_bce = Σ(loss_per_neg * focal) / N_neg

其中 N_neg ≈ 5000，而 FP 只有 ~22 个
→ 22 个 FP 的 loss 被 4978 个正确预测的 TN 平均掉了
→ neg_bce ≈ 0（因为绝大多数负样本预测正确）
```

**这就是问题的本质**：neg_bce 是所有负样本的平均 loss。即使有 22 个位置严重错误（预测为 1 但 GT=0），它们在 5000 个负样本中的平均贡献微乎其微。

### 3.2 根因 2：pos_weight 只加重了 FN，没加重 FP

`pos_weight` 的设计目的：

```
"GT=1 但你预测为 0，惩罚 pos_w 倍"
```

但对于 F1=0 的偏移预测：
- 在 GT=1 的位置，模型预测为 0 → pos_bce 高 ✓（有惩罚）
- 在 GT=0 的位置，模型预测为 1 → neg_bce 低 ✗（惩罚不够）

**不对称**：模型学到"预测 stem 结构"可以降低 pos_bce（因为某些位置确实有配对），但位置偏移时增加的 neg_bce 几乎可以忽略。

### 3.3 根因 3：Focal modulation 进一步降低了 FP 的梯度

```python
focal = (1 - pt) ** gamma  # gamma=1.0
```

对于一个高置信度的 FP（模型输出 p=0.9 但 GT=0）：
- pt = 1 - 0.9 = 0.1（因为 GT=0 时 pt = 1-pred）
- focal = (1 - 0.1)^1.0 = 0.9

Focal 不会很降低高置信度 FP 的 loss（这是对的）。但对于一个"中等置信度"的 FP（p=0.6）：
- pt = 1 - 0.6 = 0.4
- focal = 0.6

这意味着中等置信度的 FP 梯度被降低了 40%，进一步削弱了对"不太确定但错误"预测的纠正力度。

### 3.4 根因 4：Dice/DST 在训练中被"简单样本"主导

Dice loss 是 batch 级别的：
```python
dice = 1 - 2*inter / (pred + gt)  # 这里是 per-sample 然后 .mean()
```

对于 batch_size=12：
- 10 个简单样本 F1>0.8：dice ≈ 0.1-0.2
- 2 个困难样本 F1=0：dice = 1.0

Batch mean dice ≈ (10×0.15 + 2×1.0) / 12 ≈ 0.29

**困难样本的梯度贡献只有 2/12 = 17%**。随着训练进行，简单样本比例越来越高，困难样本的梯度信号被进一步稀释。

### 3.5 根因 5："平移预测"的 loss 景观是平坦的

这是**最关键的洞察**：

假设模型学会了一个 RNA 有 stem 结构（一条反平行对角线带），但位置偏了 k 个 nucleotide。

从 loss 的角度看：
- 偏移 0（正确）：pos_bce=0, neg_bce=0, dice=0 → total ≈ 0
- 偏移 1：pos_bce=高, neg_bce≈0, dice=高 → total 高
- 偏移 5：pos_bce=高, neg_bce≈0, dice=高 → total 高
- 偏移 50（完全错位）：pos_bce=高, neg_bce≈0, dice=高 → total 高

**从偏移 1 到偏移 50，loss 几乎不变！** 因为：
1. pos_bce 在偏移 >0 时就已经是最大值（GT 位置全部 miss）
2. neg_bce 始终约 0（因为 FP 被 TN 稀释）
3. dice 在 tp=0 时就是 1.0，不会再增加

**loss 景观在"偏移量"方向上是一个台阶函数**：只要偏移了，loss 就跳到最大值，但偏多少无所谓。这意味着梯度无法引导模型"从偏移 5 调到偏移 0"——因为在这个方向上 loss 是平坦的。

---

## 4. 为什么判别式模型仍有此问题（vs v6 生成式的区别）

| 问题 | v6 (Flow) | v7 (判别式) |
|------|-----------|------------|
| 噪声引入偏移 | ✓ 20步采样每步加噪 | ✗ 单次前向无噪声 |
| 位置错误的来源 | 采样过程中偏移 | **MARS 特征本身无法区分相似结构** |
| Loss 对偏移的敏感性 | 同样不够 | 同样不够 |

v7 的 F1=0 更可能因为：
1. **MARS 特征对某些 RNA 家族无区分力**：模型从特征中"看到"某种结构模式，但实际该 RNA 的结构与训练集中学到的模式不同
2. **训练样本不足的稀有家族**：模型用从多数家族学到的模式去预测少数家族，导致结构性错误

---

## 5. 解决方案

### 方案 1：加大 FP 的惩罚（直接回应用户的猜想）

**思路**：给 FP 位置也加一个类似 pos_weight 的权重。

```python
# 当前：neg_bce 无加权
neg_bce = (bce_raw * focal * (1-y) * valid).sum() / ((1-y) * valid).sum()

# 改进：对 FP 区域加权
fp_mask = (pred > 0.5) * (1-y) * valid  # 预测为1但GT为0的位置
fp_weight = 1.0 + fp_penalty * fp_mask   # 给 FP 额外惩罚
neg_bce_new = (bce_raw * focal * (1-y) * valid * fp_weight).sum() / ((1-y)*valid).sum()
```

**优点**：直接增大 FP 的 loss 贡献，不被 TN 稀释
**缺点**：需要小心调参，过大会导致模型不敢预测（recall 崩塌）

### 方案 2：Position-Aware Contrastive Loss

**思路**：不仅惩罚"你预测错了"，还惩罚"你预测到了错误的位置"。

```python
# 对于每个 FP 位置 (i,j)，找最近的 GT 位置 (i',j')
# loss += distance(i-i', j-j') * confidence(i,j)
# 这让"偏移 1 位"的惩罚 < "偏移 10 位"的惩罚
```

**优点**：打破 loss 景观在偏移方向的平坦性
**缺点**：计算开销大（需要找最近 GT pair），实现复杂

### 方案 3：Hard Example Mining / OHEM

**思路**：每个 batch 中，只取 loss 最高的 top-k 负样本计算 neg_bce。

```python
# 不按全部 neg 平均，而是只取 top-k hardest negatives
neg_losses = bce_raw * (1-y) * valid  # per-position neg loss
topk_neg_losses = neg_losses.flatten().topk(k=num_pos * 3)[0]
neg_bce = topk_neg_losses.mean()
```

**优点**：
- 直接解决"FP 被 TN 稀释"的问题
- FP 位置的 loss 自然在 top-k 中被选中
- 实现简单，不需要新超参

**缺点**：如果 k 太小可能不稳定

### 方案 4：Per-Sample Loss Reweighting

**思路**：对 batch 中 loss 高的样本（即困难样本）给更大的权重。

```python
# 当前：batch 中所有样本等权平均
total = mean(per_sample_loss)

# 改进：困难样本加权
sample_weights = per_sample_loss.detach()  # loss 越高权重越大
sample_weights = sample_weights / sample_weights.sum() * batch_size
total = (per_sample_loss * sample_weights).sum()
```

**优点**：让 F1=0 的困难样本获得更大的梯度贡献
**缺点**：可能导致训练不稳定

### 方案 5：Structure-Aware Loss（最彻底）

**思路**：引入碱基配对兼容性约束，让模型不能随意在不兼容的位置预测配对。

```python
# RNA 配对规则：AU, GC, GU 是合法配对
compat_mask = compute_base_pair_compatibility(sequence)  # (L, L)
# 在不兼容位置的预测直接给大 penalty
incompatible_fp = pred * (1 - compat_mask) * (1 - y) * valid
struct_loss = incompatible_fp.sum() / valid.sum() * struct_weight
```

**优点**：利用生物学先验，从根本上减少"不可能配对"的预测
**缺点**：需要序列信息参与 loss 计算

---

## 6. 推荐实施优先级

| 优先级 | 方案 | 预期效果 | 实施难度 |
|--------|------|---------|---------|
| **P0** | OHEM (方案 3) | 解决 FP 被 TN 稀释 | ★☆☆ 低 |
| **P0** | FP 加权 (方案 1) | 直接加大 FP 惩罚 | ★☆☆ 低 |
| P1 | Per-sample reweight (方案 4) | 困难样本更多梯度 | ★★☆ 中 |
| P1 | Base-pair compat (方案 5) | 生物学先验约束 | ★★☆ 中 |
| P2 | Position-aware loss (方案 2) | 打破偏移方向平坦 | ★★★ 高 |

**建议**：先实施方案 1 + 3 组合：

```python
# 组合方案：OHEM + FP penalty
# 1. 对负样本取 top-k hardest（k = 3 * num_positives）
# 2. 对这些 hard negatives 额外加权 fp_penalty_weight=2.0
```

---

## 7. 数学验证：为什么"平移"不增加 loss

设一个样本有 N_gt 个 GT 配对，模型预测了 N_pred 个配对，全部偏移（tp=0）。

### 当前 loss 在这个样本上的值：

**BCE (pos)**: 
- GT=1 的 N_gt 个位置，模型输出 sigmoid(logit) ≈ 0（因为预测在别处）
- per-position loss ≈ -log(1-0) ≈ 0... 不对
- 实际：GT=1, pred≈0.1 → loss = -log(0.1) × pos_w ≈ 2.3 × pos_w
- 总计：N_gt × 2.3 × pos_w

**BCE (neg)**:
- GT=0 的位置中，有 N_pred 个被预测为 1（FP），其余 ~L² - N_gt - N_pred 个正确预测为 0
- FP 位置 loss：-log(1-0.9) ≈ 2.3（假设置信度 0.9）
- TN 位置 loss：-log(1-0.05) ≈ 0.05
- 平均 neg_bce = (N_pred × 2.3 + (L²/2 - N_gt - N_pred) × 0.05) / (L²/2 - N_gt)

代入典型值 (L=100, N_gt=22, N_pred=22):
- 分子 = 22×2.3 + 4934×0.05 = 50.6 + 246.7 = 297.3
- 分母 = 4956
- **neg_bce ≈ 0.06**

**vs pos_bce** (假设 pos_w=4.5):
- pos_bce = 2.3 × 4.5 = **10.35**

所以 **pos_bce : neg_bce ≈ 170 : 1**

这证实了：**模型从 FP 得到的惩罚信号（neg_bce≈0.06）相比从 FN 得到的信号（pos_bce≈10.35）微乎其微。**

模型的"最优策略"是：**尽可能覆盖 GT 位置（降低 pos_bce）**，即使产生一些 FP 也无所谓（neg_bce 几乎不变）。

但对于从未见过的 RNA 家族，模型无法覆盖正确位置，只能猜测——猜测产生的 FP 几乎不受惩罚，所以模型仍然"自信地"预测（因为预测的 loss 代价很低）。

---

## 8. 结论

**用户的直觉完全正确**：

1. ✅ 当前 loss 系统**严重偏向惩罚 FN（漏检），而不充分惩罚 FP（误检）**
2. ✅ pos_weight 只加大了"应该预测 1 但预测了 0"的 loss
3. ✅ 没有对应机制加大"应该预测 0 但预测了 1"的 loss
4. ✅ 这导致模型在不确定时倾向于**大胆预测**（因为预测错了位置的代价极低）
5. ✅ 特别是"平移预测"——数量对但位置错——pair_count loss ≈ 0, ratio_penalty ≈ 0, neg_bce ≈ 0

**下一步**：在 v7 中实现 OHEM + FP penalty，预期可以显著减少 F1=0 的案例比例。
