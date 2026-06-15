# PriFold v8: DensityNet-Pro

## 概述

v8 是基于 v7 全面推理分析结果的**精度优化版本**。核心目标：**提升 Precision，减少 False Positive，缩小泛化 gap**。

### v7 → v8 改进动机

| v7 问题 | v8 解决方案 |
|---------|-------------|
| FP 被海量 TN 稀释（neg_bce≈0） | OHEM: 只取 top-k hardest negatives |
| pred/gt=1.26（过预测） | FP Penalty: 对 FP 位置加 3x 权重 |
| 33% bad cases 是偏移预测 | Shift-aware loss: 近似匹配给 partial credit |
| 非法碱基配对（非 AU/GC/GU） | BP Compatibility mask: 训练+推理过滤 |
| 长序列 Precision 崩塌 | Length-aware budget: 长序列收紧预测数量 |
| 泛化 gap 25%+ | Dropout 0.2 + DropPath 0.1 |

---

## 架构

```
RNA seq → MARS-LX (frozen, 160M)
        → 1D hidden + 2D attention maps
        → Pair Feature Construction (outer product + attn + seq_oh + pos_bias)
        → Axial Transformer Stack (8 layers, DropPath)
        → Contact Logit Head + Density Head
        → Loss (9 components) / Predict (length-aware budget + BP filter)
```

- **Trainable params**: ~3.6M (same backbone as v7)
- **Inference**: single forward pass, no sampling

---

## Loss 系统（9 个组件，全部可通过 config 开关控制）

| # | Loss 名称 | 权重 | 作用 | config 开关 |
|---|-----------|------|------|-------------|
| 1 | **Focal BCE** | 1.0 (base) | 主分类 loss，正样本加权 | `focal_gamma`, `pos_weight_base` |
| 2 | **Dice** | 0.5 | Set-level overlap，缓解类不平衡 | `dice_weight` (=0 关闭) |
| 3 | **DST** (Density-Stratified Tversky) | 0.4 | 低密度样本额外 FN 惩罚 | `dst_weight` (=0 关闭) |
| 4 | **Pair Count** | 0.3 | 约束预测总配对数量 | `pair_count_weight` |
| 5 | **Ratio Penalty** | 0.2 | 惩罚 pred/gt > threshold | `ratio_penalty_weight`, `ratio_penalty_threshold` |
| 6 | **Density Head** | 0.3 | 辅助监督密度预测 | `density_loss_weight` |
| 7 | **OHEM** ⭐new | — | 只取 top-k hardest negatives，让 FP 被有效惩罚 | `ohem_enabled` |
| 8 | **FP Penalty** ⭐new | 3.0 | 对 FP 位置额外加权惩罚 | `fp_penalty_enabled`, `fp_penalty_weight` |
| 9 | **BP Compatibility** ⭐new | 0.5 | 惩罚非 AU/GC/GU 位置的预测 | `bp_compat_enabled`, `bp_compat_weight` |
| 10 | **Shift-aware** ⭐new | 0.3 | 对 GT±1 范围内的预测给 partial credit | `shift_loss_enabled`, `shift_loss_weight` |

### Loss 公式

```
Total = Focal_BCE(OHEM) + Dice + DST + PairCount + RatioPenalty 
      + DensityHead + FP_Penalty + BP_Compat + Shift_Reward
```

---

## 推理策略

| 策略 | 说明 | config |
|------|------|--------|
| **Score Threshold** | 低于阈值的预测直接过滤 | `score_threshold=0.45` (v7: 0.4) |
| **Density Budget** | 用密度预测头估计最大配对数 | `use_density_budget=true` |
| **Length Decay** | 长序列收紧 budget: `factor=(100/L)^0.3` | `length_decay=0.3` |
| **BP Filter** | 推理时过滤非法碱基配对 | `bp_compat_in_inference=true` |

---

## 正则化

| 技术 | v7 | v8 | 说明 |
|------|----|----|------|
| Dropout | 0.1 | **0.2** | FFN + attention output |
| DropPath | 无 | **0.1** | 渐进式（浅层 0→深层 0.1） |
| 数据增强 | select=0.1, replace=0.3 | **select=0.15, replace=0.35** | 更强扰动 |

---

## 配置文件

主配置: `symfold/config/v8/v8_full.json`

### 消融实验设计

关闭单个组件即可消融：

```json
// 关闭 OHEM
"ohem_enabled": false

// 关闭 FP penalty  
"fp_penalty_enabled": false, "fp_penalty_weight": 0

// 关闭 BP compatibility
"bp_compat_enabled": false

// 关闭 Shift loss
"shift_loss_enabled": false

// 关闭 DST
"dst_weight": 0

// 关闭 Dice
"dice_weight": 0
```

---

## 启动训练

```bash
cd /root/aigame/dannyyan/PriFold
bash symfold/train/run_train.sh symfold/config/v8/v8_full.json
```

## 查看日志

```bash
tail -f symfold/logs/v8_full/v8_full.stdout.log
```

---

## 预期改善

| 指标 | v7 | v8 预期 | 改善来源 |
|------|-----|---------|----------|
| Test F1 | 0.654 | 0.68-0.70 | OHEM + FP penalty |
| Test Precision | 0.627 | 0.68-0.72 | FP penalty + BP compat + length budget |
| pred/gt | 1.26 | 1.05-1.15 | Ratio threshold↓ + FP penalty |
| F1=0 count | 39 | <25 | BP compat + shift reward |
| 泛化 gap | 25% | 18-20% | Dropout↑ + DropPath + 增强↑ |
