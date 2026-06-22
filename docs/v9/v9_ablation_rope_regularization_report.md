# v9 RoPE 与增强正则化消融阶段性报告

> 生成时间：2026-06-22  
> 参考：`CLAUDE.md`、`docs/v9/v9_ablation_rope_regularization_plan.md`、`docs/v9/v9_test_evaluation_report.md`  
> 当前状态：`v9_no_rope` 跑到 epoch 76，`v9_low_reg` 跑到 epoch 59；两组已有明确趋势，但还不是完整 200 epoch 终局结论。

---

## 1. 背景与问题

`CLAUDE.md` 中当前判断是：v9 的强项可能来自 **RoPE + 正则 + shift margin** 的组合。本轮消融主要验证其中两个因素：

1. **2D RoPE 是否重要**：`v9_no_rope` 仅关闭 `v9.use_rope`，其它保持 v9 full 强正则。
2. **增强正则化是否重要**：`v9_low_reg` 保留 RoPE，但降低 dropout、drop path、augmentation 和 weight decay。

主对照组是已完成的 `v9_full`：

| 模型 | RoPE | 正则 | best Val F1 | Test F1 | 备注 |
|---|---:|---:|---:|---:|---|
| `v9_full` | ON | Enhanced | **0.6814 @ e160** | **0.6961** | 当前最佳 |
| `v9_no_rope` | OFF | Enhanced | 0.5930 @ e73 | 0.5770 @ e59 test-eval | 明显掉点 |
| `v9_low_reg` | ON | Low | 0.6722 @ e51 | 0.6804 @ e59 test-eval | 接近 v9，但仍低于最终 v9 |

说明：ablation 的 Test F1 来自训练中每 20 epoch 的 `test_eval_history.json`，还不是 best checkpoint 的最终完整 eval。

---

## 2. 配置差异

### 2.1 `v9_no_rope`

只关闭 pairwise axial attention 中的 2D RoPE：

```json
"v9": {
  "dropout": 0.2,
  "drop_path": 0.15,
  "use_rope": false
},
"training": {
  "weight_decay": 0.02,
  "augmentation": {"enabled": true, "select": 0.20, "replace": 0.40}
}
```

### 2.2 `v9_low_reg`

保留 RoPE，但显著降低正则强度：

```json
"v9": {
  "dropout": 0.05,
  "drop_path": 0.03,
  "use_rope": true
},
"training": {
  "weight_decay": 0.005,
  "augmentation": {"enabled": true, "select": 0.05, "replace": 0.15}
}
```

---

## 3. 训练与测试趋势

### 3.1 Test-eval 走势

| Epoch | `v9_no_rope` Test F1 | `v9_low_reg` Test F1 | 差距 |
|---:|---:|---:|---:|
| 19 | 0.4889 | 0.6439 | +0.1550 |
| 39 | 0.5446 | 0.6645 | +0.1199 |
| 59 | 0.5770 | **0.6804** | **+0.1033** |

`v9_no_rope` 一直显著落后，而且到 epoch 59 仍只有 0.5770；`v9_low_reg` 在 epoch 59 已经达到 0.6804，接近 `v9_full` 最终 Test F1 0.6961。

### 3.2 Val F1 当前状态

| 模型 | 当前 epoch | 最新 Val F1 | best Val F1 | 训练 loss |
|---|---:|---:|---:|---:|
| `v9_no_rope` | 76 | 0.5872 | 0.5930 @ e73 | 5.6353 |
| `v9_low_reg` | 59 | 0.6699 | 0.6722 @ e51 | 1.7253 |
| `v9_full` | 182 | 0.6780 | **0.6814 @ e160** | 2.2729 |

关键观察：

- `v9_no_rope` 的验证集上限目前只有约 0.59，远低于 v9 full 的 0.68。
- `v9_low_reg` 的训练 loss 明显更低，说明低正则更容易拟合、收敛更快。
- 但 `v9_low_reg` 的 best Val F1 目前仍略低于 `v9_full`：0.6722 vs 0.6814，差约 **0.92 pp**。

---

## 4. RoPE 消融结论

### 4.1 整体效果

按当前可用 Test F1 粗略估算：

```text
RoPE effect ≈ Test F1(v9_full) - Test F1(v9_no_rope)
            ≈ 0.6961 - 0.5770
            ≈ +0.1191
```

即关闭 RoPE 后，Test F1 下降约 **11.9 个百分点**。即使考虑 `v9_no_rope` 尚未跑满，这个差距也已经远超方案中定义的 `>0.030 = 非常关键` 阈值。

### 4.2 结论

**RoPE 是当前 v9 中最重要的单项因素。**

原因：

1. `v9_no_rope` 在同样强正则、同样 loss 设计下严重掉点，说明问题不只是正则或 loss。
2. `v9_low_reg` 保留 RoPE 后，即使正则大幅降低，仍能达到 0.6804 Test F1，远高于 no-RoPE。
3. 这说明 v9 的核心泛化能力首先来自 pairwise 2D 相对位置建模，而不是单纯靠更强正则撑起来。

对后续版本的直接启发：

- v10/v11 不应移除 RoPE。
- 如果扩展模型容量或 unfreeze MARS，RoPE 应作为默认保留组件。
- 后续提升应围绕更强的位置/距离建模做，而不是回到无显式相对位置建模的结构。

---

## 5. 正则化消融结论

### 5.1 整体效果

按当前可用 Test F1 粗略估算：

```text
Reg effect ≈ Test F1(v9_full) - Test F1(v9_low_reg)
           ≈ 0.6961 - 0.6804
           ≈ +0.0157
```

增强正则当前带来的最终收益约 **1.6 个百分点**，落在方案中 `0.015 - 0.030 = 明确有效` 的下沿。

### 5.2 训练行为

`v9_low_reg` 的训练 loss 明显低：

| 模型 | 对比 epoch | Train loss | Val F1 |
|---|---:|---:|---:|
| `v9_full` | e59 | 4.0092 | 0.6317 |
| `v9_low_reg` | e59 | **1.7253** | **0.6699** |

这说明低正则版本前期拟合更快，短期验证效果也更好。但从最终 best Val 看，`v9_full` 仍然更高：

```text
best Val F1(v9_full) = 0.6814
best Val F1(v9_low_reg) = 0.6722
差距 = +0.0092
```

### 5.3 结论

**增强正则化有效，但重要性明显低于 RoPE。**

更准确地说：

- 低正则不是灾难性设置，模型仍然能学到很强结果。
- 强正则可能主要贡献在后期泛化、稳定性和 bad case 控制，而不是早期收敛速度。
- v9 的正则强度可能不是唯一最优点；后续可以在 `dropout=0.10~0.20`、`drop_path=0.05~0.15`、`weight_decay=0.005~0.02` 之间做更细 sweep。

---

## 6. 什么最重要

按当前证据排序：

### 第一优先级：2D RoPE

这是本轮消融最明确的结论。去掉 RoPE 后性能大幅下降，说明 v9 真正关键的能力是对 pair matrix 中相对位置/距离关系的建模。

### 第二优先级：保留足够正则，但不必盲目加重

强正则相对 low-reg 有约 1 pp Val F1、约 1.6 pp Test F1 的优势，属于明确但中等的收益。它应该保留，尤其是在更大模型或 MARS unfreeze 时，但可以继续调轻一些寻找更好折中。

### 第三优先级：shift margin / DST / 允许 NC 仍应保留

本轮没有直接消融这些项，但 `v9_full` 相比 v7/v8 的主报告已经显示：过预测问题、bad rate 和长序列表现都有改善。因此在下一轮实验前，不建议同时改动这些稳定组件。

---

## 7. 对 v10/v11 的建议

1. **RoPE 必须保留**：这是当前最强信号，不建议再做无 RoPE 的主线模型。
2. **低正则可以作为调参方向，不是主方向**：`v9_low_reg` 很强，但还没超过 v9 full；建议跑满或做 best checkpoint 完整 eval 后再判断是否降低正则。
3. **如果继续做 MARS unfreeze，需要 warm-start + 强正则**：`CLAUDE.md` 已指出 v10 问题不是单纯训练不够，而是没有从 v9 warm-start，且 partial unfreeze 实现不理想。结合本轮结果，unfreeze 时更应该保留 RoPE，并用足够正则防止过拟合。
4. **优先补一个完整 eval**：对 `v9_no_rope/model/best.pt` 和 `v9_low_reg/model/best.pt` 跑完整 `bprna-test`，补齐 length-bin、bad rate、per-sample 分布。
5. **交互实验可以延后**：由于 RoPE 主效应已经非常大，`v9_low_reg_no_rope` 的优先级不高；只有在需要论文式完整 2×2 消融时再跑。

---

## 8. 当前结论一句话版

**v9 的最关键改动是 2D RoPE；增强正则化也有收益，但属于第二层级。后续模型设计应默认保留 RoPE，在此基础上再调正则和 MARS unfreeze 策略。**
