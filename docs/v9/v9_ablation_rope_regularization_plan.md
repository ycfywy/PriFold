# v9 RoPE 与增强正则化消融实验方案

> 目标：验证 v9 中两个关键改动是否真正有效：
> 1. 增强正则化是否有用
> 2. 2D RoPE 位置编码是否有用

---

## 1. 背景

v9 是当前最强模型：

| 模型 | Val F1 | Test F1 | 说明 |
|------|--------|---------|------|
| v7 | — | 0.6538 | 纯判别式 DensityNet |
| v8 | 0.6083 | 0.6105 | +OHEM +FP penalty +shift +length decay |
| **v9** | **0.6814** | **0.6961** | +RoPE +shift margin +DST↓ +正则化↑ +允许NC |

v9 相比 v7/v8 有显著提升，但它一次性引入了多个改动：

- P1: 降低 DST threshold
- P2: Shift-aware margin loss
- P3: 增强正则化
- P4: 允许非标准配对
- P5: 2D RoPE 位置编码

本次消融只验证两个问题：

1. **P3 增强正则化是否有效**
2. **P5 2D RoPE 位置编码是否有效**

这两个因素对后续 v10/v11 设计很关键：

- 如果 RoPE 很有效，应继续保留，并考虑更强的位置/距离建模。
- 如果增强正则化有效，说明后续大模型或 MARS unfreeze 更需要正则化。
- 如果增强正则化收益不明显，可以放松正则、提高拟合能力。

---

## 2. 实验变量

### 2.1 因素 A：2D RoPE 位置编码

| 设置 | 配置 |
|------|------|
| RoPE ON | `"use_rope": true` |
| RoPE OFF | `"use_rope": false` |

注意：这里验证的是 `symfold/v9/model.py` 中 pairwise axial attention 的 **2D RoPE**，不是 MARS 语言模型内部的 RoPE。

### 2.2 因素 B：增强正则化

v9 当前使用增强正则化：

```json
{
  "dropout": 0.20,
  "drop_path": 0.15,
  "augmentation": {"select": 0.20, "replace": 0.40},
  "weight_decay": 0.02
}
```

消融中的低正则化版本建议进一步拉开差距：

```json
{
  "dropout": 0.05,
  "drop_path": 0.03,
  "augmentation": {"select": 0.05, "replace": 0.15},
  "weight_decay": 0.005
}
```

这里保留极少量正则，避免完全关闭后训练过于不稳定；但相比 v9 full 已经足够低，可以更清楚地观察增强正则化是否带来泛化收益。

可选极端对照（如果资源允许）可以再跑一组 `v9_no_reg`：

```json
{
  "dropout": 0.0,
  "drop_path": 0.0,
  "augmentation": {"enabled": false},
  "weight_decay": 0.0
}
```

但主实验不建议优先跑 `v9_no_reg`，因为完全无正则可能导致训练分布偏离太大，结论不如低正则对照稳定。

---

## 3. 2×2 实验矩阵

| 实验名 | RoPE | 正则化 | 目的 |
|--------|------|--------|------|
| `v9_full` | ON | Enhanced | 已完成，主对照组 |
| `v9_no_rope` | OFF | Enhanced | 验证 RoPE 是否有效 |
| `v9_low_reg` | ON | Low | 验证增强正则化是否有效 |
| `v9_low_reg_no_rope` | OFF | Low | 验证 RoPE 与正则化是否有交互 |

`v9_full` 已完成：

```text
best val F1 = 0.6814 @ epoch 160
test F1 = 0.6961
```

因此本次需要新跑 3 个实验：

```text
v9_no_rope
v9_low_reg
v9_low_reg_no_rope
```

---

## 4. 配置文件规划

建议新建目录：

```text
symfold/config/v9_ablation/
```

新建配置：

```text
symfold/config/v9_ablation/v9_no_rope.json
symfold/config/v9_ablation/v9_low_reg.json
symfold/config/v9_ablation/v9_low_reg_no_rope.json
```

输出目录：

```text
symfold/outputs/v9_ablation_no_rope/
symfold/outputs/v9_ablation_low_reg/
symfold/outputs/v9_ablation_low_reg_no_rope/
```

日志目录：

```text
symfold/logs/v9_ablation_no_rope/
symfold/logs/v9_ablation_low_reg/
symfold/logs/v9_ablation_low_reg_no_rope/
```

---

## 5. 训练设置

### 5.1 是否 warm-start

本消融建议 **从头训练，不从 v9 best checkpoint warm-start**。

原因：

1. RoPE ON 的权重不能公平迁移到 RoPE OFF 模型。
2. 正则化强弱影响整个优化轨迹，warm-start 会掩盖差异。
3. 消融实验需要从相同 seed、相同初始化条件下比较最终效果。

统一从 MARS 预训练 backbone 开始训练，下游 head 随机初始化。

### 5.2 统一训练参数

所有消融实验应保持一致：

```json
{
  "seed": 3407,
  "epochs": 200,
  "warmup_epochs": 8,
  "dataset_mode": "bprna",
  "max_len_filter": 490,
  "batch_size": 12,
  "max_sq_tokens": 600000,
  "gradient_accumulation_steps": 2,
  "lr": 5e-4,
  "test_eval_every": 20,
  "amp_dtype": "bf16"
}
```

### 5.3 训练规范

每个实验必须满足：

1. 每个 epoch 保存 `history.json`
2. 每个 epoch 输出 `training_curves.png`
3. 每 20 epochs 做一次 `bprna-test` 评估，并写入 `history.json`
4. 保存 `best.pt` 与 `last.pt`
5. 训练结束后用 best checkpoint 做完整 test eval

---

## 6. 评估方式

### 6.1 训练中评估

每个 epoch 记录：

- train loss
- BCE loss
- density loss
- FP penalty
- shift loss
- val precision
- val recall
- val F1
- val MCC
- val pred pairs
- val GT pairs

每 20 epoch 额外记录：

- test precision
- test recall
- test F1

### 6.2 训练结束评估

用 best-val checkpoint 做完整 test：

```bash
python symfold/eval/eval_v9.py \
  --config <config> \
  --ckpt <output_dir>/model/best.pt \
  --device cuda:0 \
  --output_dir <output_dir>/test_eval \
  --stages bprna-test
```

输出：

```text
<output_dir>/test_eval/bprna_test_report.md
<output_dir>/test_eval/bprna_test_per_sample.json
<output_dir>/test_eval/eval_results.json
```

---

## 7. 分析指标

### 7.1 总体指标

每个实验记录：

| 指标 | 说明 |
|------|------|
| best val F1 | 选择 checkpoint 的依据 |
| test F1 | 最终主要指标 |
| test precision | 预测准确性 |
| test recall | 覆盖 GT 配对能力 |
| MCC | 更稳定的二分类相关指标 |
| Pred/GT ratio | 是否过预测/欠预测 |
| bad rate | F1 < 0.3 的样本比例 |

### 7.2 长度分组

验证 RoPE 时必须看长度分组：

| 区间 | 目的 |
|------|------|
| 0-100 | 短序列，结构简单 |
| 100-200 | 主体样本区间，当前瓶颈 |
| 200-300 | 中长序列 |
| 300-400 | v9 表现出乎意料好的区间 |
| 400-500 | 长序列，样本少但难度高 |

判断：

```text
如果 v9_no_rope 在 300-500 区间明显下降，说明 2D RoPE 对长序列有效。
```

### 7.3 配对距离分组

建议额外统计配对距离：

| 距离区间 | 说明 |
|----------|------|
| <25 | 短距离配对 |
| 25-50 | 中短距离 |
| 50-100 | 中长距离 |
| >=100 | 长距离配对 |

判断：

```text
如果 no_rope 的 long-range recall 明显下降，说明 RoPE 对长距离配对有效。
```

### 7.4 过拟合分析

验证增强正则化时看：

1. train loss 是否更低
2. val/test F1 是否更高
3. val-test gap 是否更小
4. bad rate 是否降低
5. F1 分布是否更稳定

判断：

```text
如果 low_reg 训练 loss 更低，但 val/test 更差，说明增强正则化有效。
```

---

## 8. 结论判定标准

### 8.1 F1 差异阈值

| 差异范围 | 解释 |
|----------|------|
| < 0.005 | 基本无效，可能是噪声 |
| 0.005 - 0.015 | 有弱收益，需要多 seed 验证 |
| 0.015 - 0.030 | 明确有效 |
| > 0.030 | 非常关键 |

### 8.2 RoPE 是否有效

比较：

```text
RoPE effect = test_F1(v9_full) - test_F1(v9_no_rope)
```

辅助观察：

```text
long_seq_effect = F1_300_500(v9_full) - F1_300_500(v9_no_rope)
long_range_effect = Recall_long_range(v9_full) - Recall_long_range(v9_no_rope)
```

结论规则：

- 如果 overall F1 提升 > 0.015，RoPE 明确有效。
- 如果 overall 提升不大，但长序列/长距离明显提升，RoPE 仍然有效。
- 如果所有区间差异 < 0.005，RoPE 可认为不是主要贡献。

### 8.3 增强正则化是否有效

比较：

```text
Reg effect = test_F1(v9_full) - test_F1(v9_low_reg)
```

辅助观察：

```text
bad_rate_effect = bad_rate(v9_low_reg) - bad_rate(v9_full)
val_test_gap_effect = gap(v9_low_reg) - gap(v9_full)
```

结论规则：

- 如果 low_reg train loss 更低但 test F1 更差，增强正则化有效。
- 如果 low_reg test F1 更高，说明 v9 正则可能过强。
- 如果差异很小，可以后续选择更低正则以提高拟合能力。

### 8.4 交互作用

比较：

```text
Expected additive drop = drop(no_rope) + drop(low_reg)
Actual drop = drop(low_reg_no_rope)
Interaction = Actual drop - Expected additive drop
```

如果 `Interaction > 0.01`，说明 RoPE 与增强正则化存在明显交互。

---

## 9. 推荐执行顺序

### 第一批：单因素消融

先跑：

```text
1. v9_no_rope
2. v9_low_reg
```

理由：

- 这两个实验直接回答两个核心问题。
- 如果 100 epoch 时差异已经很明显，可以提前判断趋势。

### 第二批：交互消融

再跑：

```text
3. v9_low_reg_no_rope
```

理由：

- 这个实验主要用于判断二者是否有交互。
- 如果前两个单因素差异很小，可以考虑不跑或延后。

---

## 10. 预期结果

基于目前 v9/v10 分析，先验判断如下：

### 10.1 RoPE 大概率有效

理由：

- v9 在 `300-400` 长度区间 test F1 达到 `0.7000`。
- 这一区间在 RNA contact prediction 中通常较难。
- RoPE 提供相对位置建模，理论上应帮助长距离配对。

预期：

```text
v9_full > v9_no_rope
RoPE effect: +1.5 ~ +3.0 pp
长序列/长距离配对收益更明显
```

### 10.2 增强正则化可能有效，但收益有限

理由：

- v9 bad rate 为 9.4%，明显好于 v8 的 15.3%。
- 但 v9 同时改了多个组件，不能确认正则化是主因。
- 当前模型仍是 5.09M 参数，不一定特别需要强正则。

预期：

```text
v9_full >= v9_low_reg
Reg effect: +0.5 ~ +1.5 pp
如果 low_reg 更好，说明 v9 正则偏强
```

---

## 11. 最终报告规划

实验完成后生成：

```text
docs/v9/v9_ablation_rope_regularization_report.md
```

报告结构：

1. 实验设计
2. 四组配置对照
3. 训练曲线对比
4. best val F1 对比
5. test F1 对比
6. length-bin 分析
7. distance-bin 分析
8. bad case rate 对比
9. RoPE 是否有效
10. 增强正则化是否有效
11. 对 v10/v11 的启发

---

## 12. 执行命令模板

### v9_no_rope

```bash
torchrun --nproc_per_node=2 --standalone --nnodes=1 \
  symfold/train/train_v9_ddp.py symfold/config/v9_ablation/v9_no_rope.json
```

### v9_low_reg

```bash
torchrun --nproc_per_node=2 --standalone --nnodes=1 \
  symfold/train/train_v9_ddp.py symfold/config/v9_ablation/v9_low_reg.json
```

### v9_low_reg_no_rope

```bash
torchrun --nproc_per_node=2 --standalone --nnodes=1 \
  symfold/train/train_v9_ddp.py symfold/config/v9_ablation/v9_low_reg_no_rope.json
```

---

## 13. 实验成功标准

一次完整消融应包含：

- [ ] 三个 ablation 实验均训练完成或至少训练到 100 epoch 有明确趋势
- [ ] 每个实验有 `history.json`
- [ ] 每个实验有 `training_curves.png`
- [ ] 每 20 epoch 有 test eval 记录
- [ ] 每个实验最终有 best checkpoint 的完整 test report
- [ ] 生成最终对比报告 `v9_ablation_rope_regularization_report.md`

---

## 14. 注意事项

1. 消融实验不要从 v9 checkpoint warm-start。
2. 所有实验必须固定 seed=3407。
3. 尽量保持除消融变量外其他参数完全一致。
4. 如果训练资源紧张，优先跑 `v9_no_rope` 和 `v9_low_reg`。
5. 如果两个单因素消融已经能明确回答问题，交互实验可以延后。
6. 最终结论以 best-val checkpoint 的 `bprna-test` 结果为准。
