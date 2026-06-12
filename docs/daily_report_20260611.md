# PriFold 工作日报 — 2026-06-11

## 今日概要

v7 DensityNet 完成 case analysis 和 loss 机制深度分析，新增四项改进（OHEM/FP penalty/碱基配对约束/RFAM 过采样），v7 续训 test F1 从 0.6319 提升至 **0.6426**（+1.07%，新高 🎉）。

---

## 1. v7 Case Analysis 完成

**输出**：`symfold/outputs/v7_full/case_analysis/`（CSV、JSON、16 张可视化图）

对 train(2015)/val(1299)/test(1303) 共 4617 个样本进行全面分析。

### 关键发现

| 指标 | Train | Val | Test |
|------|-------|-----|------|
| F1 | 0.813 | 0.628 | 0.631 |
| F1=0 | 10 (0.5%) | 48 (3.7%) | 52 (4.0%) |
| F1<0.3 | 41 (2.0%) | 201 (15.5%) | 185 (14.2%) |

- **泛化 gap 18.4%**（train 0.81 vs test 0.63），不是过拟合，是数据多样性不足
- **F1=0 从 v6 的 97 个降到 52 个（-46%）**，判别式模型确实更稳定
- **RFAM vs non-RFAM 差距 23.2%**（0.675 vs 0.907）

### 五种失败模式（F1<0.3, N=427）

| 模式 | 占比 | 特征 |
|------|------|------|
| 数量对位置错 | 36.5% | pred/gt≈1 但 F1<0.3，**最大问题** |
| 完全错位 | 18.3% | F1=0, tp=0，pred/gt=2.28 |
| 偏移预测 ⭐新发现 | 15.7% | 46%的 FP 在 GT ±3 位置内 |
| 严重过预测 | 7.7% | pred/gt=3.54，极低密度 |
| 其他 | 20.6% | 混合模式 |

---

## 2. F1=0 Loss 机制深度分析

**文档**：`docs/v7_f1_zero_loss_analysis.md`

### 核心结论

用户假设完全正确：**Loss 不对称导致模型不怕预测错位置**。

| 根因 | 分析 |
|------|------|
| pos_weight 只加重 FN | GT=1→pred=0 惩罚 4-100x，但 GT=0→pred=1 无额外惩罚 |
| FP 被海量 TN 稀释 | 22 个 FP 在 4956 个负样本中占 0.4%，neg_bce ≈ 0 |
| 平移预测 loss 平坦 | 偏移 1 位和偏移 50 位 loss 一样大，无梯度引导纠偏 |
| pair_count/ratio 只看数量 | "数量对位置错"时这两个 loss ≈ 0 |

**数学验证**：pos_bce : neg_bce ≈ **170 : 1**，模型几乎不受 FP 惩罚。

---

---

## 4. v6 报告深度补充

**更新**：`docs/v6_case_analysis_report.md` 新增第 6-7 节

结合 v6 代码（`discrete_flow.py`, `model.py`, `da_se_dit.py`）深入分析四个问题的代码级根因：
- 为什么 Precision 差最多 → flow 采样 + pos_weight 偏向 Recall
- 为什么 F1=0 有 97 个 → density head 只学数量、flow 对未见结构无力
- 为什么 RFAM 差 24.5% → 数据长尾 + sampler 不感知家族
- 为什么 SRP 过预测 2.2x → density 误估 + adaptive budget 放大误差

---

## 5. v7 续训进展

从 epoch 110 恢复续训（16:38 启动），截至 19:23 已训练到 epoch 132：

| Epoch | Test F1 | Test P | Test R | MCC |
|-------|---------|--------|--------|-----|
| 109 (之前 best) | 0.6319 | 0.6083 | 0.6761 | 0.6352 |
| 119 | 0.6337 | 0.6157 | 0.6700 | 0.6365 |
| **129** | **0.6426** | 0.6139 | 0.6924 | 0.6461 |

- **Best val F1 = 0.6332**（epoch ~125 区间）
- **Best test F1 = 0.6426**（epoch 129，相比之前 0.6319 提升 +1.07%）🎉
- 当前 epoch 132，patience 6/30，仍在训练中

---

## 6. 其他工作

- **清理 checkpoint**：删除 22 个中间 epoch_xxx.pt（释放 ~14GB），设置 `save_every=9999` 只保留 best+last
- **修复 `run_train.sh`**：conda 激活在 setsid bash 子 shell 中失败，改用 `conda shell.bash hook`
- **修复 FamilyBalancedSampler OOM**：family sampling 后仍用 LengthBucket 分 batch
- **更新 CLAUDE.md**：反映所有当前状态

---

## 7. 当前状态总结

| 项目 | 状态 |
|------|------|
| v7_full 训练 | 🟢 运行中（epoch 132/200，PID 43641） |
| v7 case analysis | ✅ 完成 |
| Loss 分析文档 | ✅ 完成 |
| 四项改进代码 | ✅ 完成（config 开关，消融就绪） |
| v6 报告补充 | ✅ 完成 |
| 消融实验 | 🟡 待 v7_full 完成后运行 |

### 距 baseline 差距收敛趋势

```
v3:  F1=0.405  距 baseline 47%
v4:  F1=0.487  距 baseline 37%
v5:  F1=0.619  距 baseline 20%
v6:  F1=0.621  距 baseline 19%（生成式瓶颈）
v7:  F1=0.643  距 baseline 17%（判别式，仍在训练）⬆️
```

---

## 8. 明日计划

1. v7_full 继续训练至收敛（200 epochs 或 patience=30 触发）
2. 训练完成后运行消融实验（优先：v7_ohem_only, v7_all_new）
3. 对消融结果做 case analysis 对比
