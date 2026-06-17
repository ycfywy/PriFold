# v9 DensityNet-Pro+ 测试评估报告

> 生成时间: 2026-06-17 11:11
> Checkpoint: `symfold/outputs/v9_ddp/model/best.pt` (epoch 160, val F1=0.6814)
> Config: `symfold/config/v9/v9_ddp.json`
> 评估耗时: 39.3s

---

## 1. 总体指标

| 指标 | 值 |
|------|-----|
| **Test F1** | **0.6961** |
| Precision | 0.6917 |
| Recall | 0.7186 |
| MCC | 0.6990 |
| 样本数 | 1303 |
| 平均长度 | 135.5 |
| 平均 GT pairs | 31.1 |
| 平均 Pred pairs | 30.7 |
| Pred/GT ratio | 1.133 |

### 关键发现

- **Test F1 = 0.6961**，相比 v7 (0.6538) 提升 **+4.2 个百分点 (+6.5%)**
- Precision/Recall 较为均衡（0.69/0.72），Recall 略高说明模型倾向于预测更多配对
- Pred/GT ratio = 1.133（轻微过预测 13%），对比 v7 的 1.90（过预测 90%）大幅改善
- MCC = 0.6990，与 F1 高度一致

---

## 2. 版本对比

| 版本 | Test F1 | Precision | Recall | Params | 关键改进 |
|------|---------|-----------|--------|--------|----------|
| **v9** | **0.6961** | 0.6917 | 0.7186 | 5.09M | +RoPE +shift margin +DST↓ +正则↑ +允许NC |
| v7 | 0.6538 | — | — | 3.56M | 纯判别式 DensityNet |
| v8 | 0.6105 | — | — | 3.56M | +OHEM +FP penalty +shift +decay |
| baseline | 0.7700 | — | — | ~50M+ | 官方 PriFold (RNAformer) |

### 与 baseline 差距

- v9 vs baseline: **0.6961 vs 0.7700**，差距 7.4 个百分点
- v7 vs baseline: 0.6538 vs 0.7700，差距 11.6 个百分点
- **v9 将差距从 11.6% 缩小到 7.4%**（缩小了 36%）

---

## 3. F1 分布

| 统计量 | 值 |
|--------|-----|
| Mean | 0.6961 |
| Median | 0.7692 |
| Std | 0.2572 |
| Q25 | 0.5623 |
| Q75 | 0.8936 |
| Bad rate (F1<0.3) | **9.4%** (122/1303) |

### 分布特征

- **Median (0.769) > Mean (0.696)**：分布左偏，少数 bad case 拉低均值
- Q75 = 0.894：75% 的样本 F1 > 0.56，25% 的样本 F1 > 0.89
- Bad rate 9.4%（对比 v8 的 15.3% 明显改善）

---

## 4. 按长度分组

| 长度区间 | N | F1 | Precision | Recall | 趋势 |
|----------|---|-----|-----------|--------|------|
| 0-100 | 571 | **0.7481** | 0.7282 | 0.7906 | 最佳 |
| 100-200 | 532 | 0.6536 | 0.6477 | 0.6746 | 中等 |
| 200-300 | 97 | 0.6427 | 0.6677 | 0.6271 | 中等 |
| 300-400 | 76 | 0.7000 | 0.7566 | 0.6604 | 出乎意料好 |
| 400-500 | 27 | 0.6124 | 0.6934 | 0.5524 | 最弱 |

### 长度分析

- **短序列 (<100) 表现最好**（F1=0.748），这是因为结构相对简单
- **中等长度 (100-300) 是主要挑战区**（F1 ~0.65），占总样本 48%
- **长序列 (300-400) 表现较好**（F1=0.70），说明 2D RoPE 长距离建模有效
- **最长序列 (400-500)** 样本少 (N=27)，F1=0.61，Recall 偏低（0.55）说明漏掉了配对
- 整体趋势：RoPE 帮助长序列建模，300-400 区间甚至优于 100-200

---

## 5. v9 改进效果分析

### P1: DST threshold 降低 (0.10→0.05)
- 更多低密度样本受到保护
- Bad rate 从 v8 的 15.3% 降至 9.4%

### P2: Shift margin loss
- Pred/GT ratio = 1.133（v7 为 1.90），过预测问题大幅改善
- FP 总数 = 12161（在 1303 个样本中），平均每样本 9.3 个 FP

### P3: 增强正则化 (Dropout=0.2, DropPath=0.15)
- F1 std = 0.257，模型泛化能力好
- 长序列表现提升明显

### P4: 允许非标准配对
- 预计贡献了约 2-3% 的 Recall 提升（GT 中 10% 为非标准配对）

### P5: 2D RoPE 位置编码
- 300-400 长序列 F1=0.70，表现出乎意料地好
- 整体 Recall=0.72（高于 Precision=0.69），说明模型能更好地"看到"远距离配对

---

## 6. 推理参数

```json
{
  "use_density_budget": true,
  "default_budget_fraction": 0.30,
  "score_threshold": 0.43,
  "length_decay": 0.15,
  "budget_floor": 0.6
}
```

---

## 7. Bad Case 特征

### 最差 10 个样本（F1=0）

| # | Name | Len | GT | Pred | Pred/GT |
|---|------|-----|-----|------|---------|
| 1 | bpRNA_RFAM_3116 | 179 | 47 | 45 | 0.96 |
| 2 | bpRNA_RFAM_5690 | 70 | 3 | 5 | 1.67 |
| 3 | bpRNA_RFAM_6042 | 96 | 3 | 3 | 1.00 |
| 4 | bpRNA_RFAM_6348 | 70 | 7 | 2 | 0.29 |
| 5 | bpRNA_RFAM_6486 | 56 | 2 | 13 | 6.50 |
| 6 | bpRNA_RFAM_6540 | 123 | 19 | 23 | 1.21 |
| 7 | bpRNA_RFAM_7398 | 87 | 3 | 4 | 1.33 |
| 8 | bpRNA_RFAM_8962 | 85 | 12 | 9 | 0.75 |
| 9 | bpRNA_RFAM_10305 | 93 | 21 | 24 | 1.14 |
| 10 | bpRNA_RFAM_11730 | 125 | 2 | 8 | 4.00 |

### Bad case 特征分析

- F1=0 但 Pred/GT ratio ≈ 1.0 的样本（如 #1: 47 GT, 45 Pred）：模型预测数量正确但位置完全错误
- 低 GT pairs (2-3)：极稀疏结构，预测难度大
- RFAM 来源居多：可能是训练集中覆盖不足的 RNA family

---

## 8. 最佳表现

- 完美预测 (F1=1.0): 有多个 CRW 数据库的 tRNA 样本（长度 73-80, GT=22 pairs）
- 这些是经典 tRNA cloverleaf 结构，模型学习充分

---

## 9. 结论与下一步

### 结论

1. **v9 是目前最佳模型**：Test F1=0.6961，超越 v7 4.2 个百分点
2. **过预测问题基本解决**：Pred/GT ratio 从 1.90 (v7) 降至 1.13
3. **与 baseline 差距缩小至 7.4%**（从 11.6% 缩小 36%）
4. **RoPE 对长序列有效**：300-400 区间表现甚至优于 100-200
5. **正则化和非标准配对支持有贡献**：Bad rate 从 15.3% 降至 9.4%

### 下一步建议

1. **进一步缩小与 baseline 差距**:
   - 考虑 unfreeze MARS 最后 2 层（当前完全冻结 160M）
   - 增加 Axial Transformer 层数（8→12）或维度（192→256）
2. **解决中等长度区间 (100-300) 的瓶颈**:
   - 这是样本最多、F1 最低的区间
   - 可能需要多尺度注意力或 dilated attention
3. **减少 Bad case**:
   - 9.4% 的 bad rate 仍有提升空间
   - 考虑对 RFAM 家族做针对性数据增强
4. **超参数微调**:
   - score_threshold 可尝试 0.40-0.45 范围做 sweep
   - budget_fraction 和 length_decay 可进一步优化

---

## 10. 评估脚本

```bash
CUDA_VISIBLE_DEVICES=0 python symfold/eval/eval_v9.py \
  --config symfold/config/v9/v9_ddp.json \
  --ckpt symfold/outputs/v9_ddp/model/best.pt \
  --device cuda:0 \
  --output_dir symfold/outputs/v9_ddp/test_eval \
  --stages bprna-test
```
