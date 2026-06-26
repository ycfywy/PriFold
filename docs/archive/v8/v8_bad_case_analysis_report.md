# PriFold v8 DensityNet-Pro: Bad Case 全面分析报告

**生成时间**: 2026-06-16  
**模型**: v8 DensityNet-Pro (best.pt, epoch 137, best_val_f1=0.6083)  
**数据集**: bprna-test (N=1303)  
**分析设备**: cuda:1  
**输出目录**: `symfold/outputs/v8_full/comprehensive_analysis/`

---

## 执行摘要

v8 模型（epoch 137/200，仍在训练中）在 bprna-test 上的 **Bad Case Rate = 15.3%**（199/1303），与 v7 持平。当前 Test F1 = 0.6159（v7 final = 0.6538），差距主要来自 Recall 的下降。v8 的 FP Penalty + OHEM + Length Decay 策略成功改善了 Precision/Recall 平衡，但引入了长序列 Recall 不足的新问题。

**核心发现**:
1. **偏移预测**是最主要的失败模式（32.7%），shift_loss 有帮助但仍不够
2. **长序列 Recall 下降严重**（length_decay=0.3 对 350+ 序列过于激进）
3. **52.4% 的 FP 是 near-miss**（距 GT ±1~±3），说明模型"方向正确但定位不精确"
4. **F1=0 的 case 有 48 个**，多为短序列的完全错位（complete_miss）

---

## 1. 总体性能

| 指标 | Mean | Median | Std |
|------|------|--------|-----|
| **F1** | 0.6159 | 0.6667 | 0.2768 |
| **Precision** | 0.6126 | 0.6522 | 0.2867 |
| **Recall** | 0.6451 | 0.6818 | 0.2909 |
| **MCC** | 0.6203 | 0.6753 | 0.2768 |
| **Pred/GT Ratio** | 1.175 | 1.028 | 0.834 |

### 性能分层

| 等级 | 样本数 | 占比 |
|------|--------|------|
| Excellent (F1≥0.8) | 408 | 31.3% |
| Good (0.6≤F1<0.8) | 343 | 26.3% |
| Fair (0.3≤F1<0.6) | 353 | 27.1% |
| **Bad (F1<0.3)** | **199** | **15.3%** |
| Zero (F1=0) | 48 | 3.7% |

---

## 2. 失败模式分类

| 失败模式 | 数量 | 占比 | 含义 |
|----------|------|------|------|
| **shifted_prediction** | 65 | 32.7% | 预测在 GT ±3 范围内偏移 |
| **wrong_position** | 43 | 21.6% | 预测数量合理但位置完全错误 |
| **mixed** | 36 | 18.1% | 混合问题（部分偏移+部分错位） |
| **complete_miss** | 35 | 17.6% | 有预测但零 TP（全部 FP） |
| **severe_overpredict** | 13 | 6.5% | 过度预测（pred/gt > 2.0） |
| **pseudoknot_failure** | 7 | 3.5% | 伪结导致预测困难 |

**关键观察**: 
- shifted_prediction (32.7%) 是第一大问题 → shift_loss 需要增强
- complete_miss (17.6%) 中很多是短序列（<80bp），GT 配对数少但模型完全预测错

---

## 3. 关键发现

### 3.1 长序列 Recall 下降（Length Decay 过于激进）

| 长度区间 | N | F1 | Precision | Recall | Pred/GT | Bad Rate |
|----------|---|----|-----------|----|---------|----------|
| <100 | 571 | 0.680 | 0.645 | **0.747** | 1.28 | 13.1% |
| 100-200 | 532 | 0.567 | 0.566 | 0.586 | 1.19 | 17.5% |
| 200-350 | 129 | 0.542 | **0.595** | **0.509** | 0.89 | 15.5% |
| **350+** | 71 | 0.604 | **0.740** | **0.516** | **0.71** | 15.5% |

**发现**:
- 350+ 序列: Precision=0.740 >> Recall=0.516，pred/gt=0.71 → **预测数量严重不足**
- `length_factor = (100/L)^0.3` 在 L=400 时 = 0.70，在 L=490 时 = 0.64
- Budget 被压缩得太厉害，很多真实配对被截断了

**建议**: 将 `length_decay` 从 0.3 降到 0.15~0.2，或设置 minimum budget floor

### 3.2 Shift 偏移分析

| 类别 | FP 数量 | 占比 |
|------|---------|------|
| **Shift ±1** | 4970 | **33.4%** |
| Shift ±2 | 1801 | 12.1% |
| Shift ±3 | 1042 | 7.0% |
| **Far FP（完全错误）** | 7086 | **47.6%** |
| **Total FP** | 14899 | 100% |

**发现**:
- **超过一半的 FP (52.4%) 是 near-miss**，即预测"几乎正确"但有轻微偏移
- 其中 Shift ±1 占了 33.4%，这正是 shift_loss (radius=1) 应该奖励的范围
- shift_loss 的 weight=0.3 可能过小，需要增加到 0.5~0.8
- 考虑增大 `shift_radius` 到 2

### 3.3 Score Threshold 分析

- **185/199 bad cases** 的 missed GT score < 0.45（threshold）
  - → 模型对这些 GT 位置 confidence 不足，不是 threshold 太高的问题
  - → 这是模型能力问题，需要更多训练或架构改进
- 仅 **9 个 bad cases** missed GT score ≥ 0.45（被 budget 截断）
  - → Budget 截断不是主要问题（除长序列外）

### 3.4 伪结影响

| 分组 | N | Mean F1 | Bad Rate |
|------|---|---------|----------|
| 无伪结 | 1174 | 0.621 | — |
| 有伪结 | 129 | 0.571 | 17.1% |

- 伪结导致 F1 下降约 **5 个百分点**
- 影响不算极端但确实存在

### 3.5 BP Compatibility

- GT 和 Pred 中非标准配对数 = 0
- 由于序列映射未完全对接（sample name 格式），BP 统计为 0
- 从配置看 `bp_compat_enabled=false`，推理时也未启用 BP filter
- **建议**: 在 v9 中考虑启用 `bp_compat_in_inference=true`

---

## 4. v7 vs v8 对比

| 指标 | v7 (200 epochs) | v8 (epoch 137) | 差值 |
|------|-----------------|----------------|------|
| Test F1 | 0.6538 | 0.6159 | **-0.038** |
| Test Precision | 0.6267 | 0.6126 | -0.014 |
| Test Recall | 0.7122 | 0.6451 | **-0.067** |
| Bad Case Rate | ~15% | 15.3% | 持平 |
| F1=0 Cases | 39 | 48 | +9 |
| Pred/GT Ratio | 1.26 (v7) | 1.175 | -0.085 ✓ |

**分析**:
- v8 成功降低了 Pred/GT Ratio（1.26→1.175），过预测问题改善
- 但代价是 **Recall 大幅下降** (-6.7%)，FP Penalty 和 Length Decay 太激进
- F1=0 的 case 反而增加了 9 个（48 vs 39），可能是 threshold 提高和 budget 收紧的副作用
- **注意**: v8 还在训练中 (137/200 epochs)，最终可能收敛到更好

---

## 5. 改进建议（按优先级排序）

| 优先级 | 建议 | 预期效果 | 具体操作 |
|--------|------|----------|----------|
| **P0** | 降低 length_decay | 恢复长序列 Recall | `length_decay: 0.3 → 0.15` |
| **P0** | 增大 shift_loss_weight | 减少偏移预测 | `shift_loss_weight: 0.3 → 0.6` |
| **P1** | 扩大 shift_radius | 覆盖更多 near-miss | `shift_radius: 1 → 2` |
| **P1** | 降低 FP penalty weight | 平衡 Precision/Recall | `fp_penalty_weight: 3.0 → 2.0` |
| **P2** | 启用 BP compat in inference | 过滤非法预测 | `bp_compat_in_inference: true` |
| **P2** | 降低 score_threshold | 减少 no_prediction | `score_threshold: 0.45 → 0.42` |
| **P3** | 伪结专用处理 | 改善 PK 结构预测 | 添加多阶段预测或 PK detector |

---

## 6. Bad Case 可视化

详见 `symfold/outputs/v8_full/comprehensive_analysis/bad_cases/` 目录，包含 top 30 个最差 case 的可视化卡片，每张卡片包含：
- Ground Truth contact map
- Prediction contact map  
- Diff map (TP=green, FP=red, FN=blue)
- Score heatmap
- 详细指标注解

---

## 7. 文件清单

| 文件 | 说明 |
|------|------|
| `overall_performance.png` | F1/Precision/Recall/MCC 分布 |
| `failure_mode_summary.png` | 失败模式分类饼图+柱状图 |
| `length_decay_analysis.png` | Length decay 对 F1/Ratio 的影响 |
| `shift_analysis.png` | Shift 偏移分析 |
| `bp_compat_analysis.png` | BP 兼容性分析 |
| `score_confidence_analysis.png` | 预测 confidence 分析 |
| `pseudoknot_analysis.png` | 伪结影响分析 |
| `complexity_vs_f1.png` | 结构复杂度 vs F1 |
| `bad_cases/` | Top 30 bad case 可视化卡片 |
| `test_metrics.json` | 所有样本的详细指标 (JSON) |
| `bad_cases_summary.csv` | Bad case 汇总表 (CSV) |

---

## 8. 结论

v8 DensityNet-Pro 的精度优化策略（OHEM + FP Penalty + Length Decay）方向正确但力度过大：
1. **Pred/GT Ratio 从 1.26 降到 1.175** ✓ 过预测改善
2. **但 Recall 下降 6.7%** ✗ 长序列预测不足
3. **偏移预测仍是第一问题** (32.7%)，shift_loss 需要增强
4. **模型仍在训练中**，epoch 137→200 期间预计还有 1-3% F1 提升空间

下一步应在 v8 训练完成后，基于上述发现调整超参数（尤其是 length_decay 和 shift_loss），或启动 v9 实验。
