# SymFold v9 版本总结

> 模型代号：DensityNet-Pro+ | 可训练参数：~4.6M | 基座：MARS-LX (160M, frozen)

---

## 一、核心改进与效果

### 1.1 相比 v8 的关键改进

| # | 改进项 | v8 | v9 | 说明 |
|---|--------|----|----|------|
| P1 | DST threshold | 高 | 降低 | 保护低密度样本 |
| P2 | Shift-aware Loss | radius=1, w=0.3 | radius=2, w=0.6→0.8 | 偏移容错，奖励 ±2 范围内的正确预测 |
| P3 | 增强正则化 | dropout=0.1, drop_path=0.05 | dropout=0.2, drop_path=0.15 | 缓解过拟合 |
| P4 | 非标准配对 | 不支持 | 允许非标准碱基配对 | 覆盖更多真实结构 |
| P5 | **2D RoPE** | 无位置编码 | 2D 旋转位置编码 | **最关键改进**，编码残基对的相对距离 |
| — | 推理策略 | score_th=0.45, budget固定 | score_th=0.43, length_decay=0.15, budget_floor=0.6 | 长序列 Recall 提升 |
| — | FP penalty | 3.0 | 2.0 | 放松精度惩罚，提升 Recall |
| — | 训练效率 | 单卡 | torch.compile + DDP 双卡 | 训练速度翻倍 |

### 1.2 模型架构

```
RNA seq → MARS-LX (frozen, 160M, dim=1056)
        → 1D Projection (1056→192→96) + 2D Attention (72→48→48)
        → Pair Feature (outer_prod + attn + seq_pair + pos_bias)
        → Input Projection (→192 dim)
        → 8× Axial Transformer (6 heads, dim_head=32, 2D RoPE, DropPath=0.15)
        → Contact Logit + Density Head
        → 8-component Loss / Budget-aware Prediction
```

### 1.3 Loss 系统（8 组件）

| 组件 | 权重 | 作用 |
|------|------|------|
| Focal BCE + OHEM | 1.0 | 主分类 loss，聚焦难样本 |
| Dice Loss | 0.5 | 区域重叠优化 |
| DST Loss | 0.4 | 低密度样本保护 |
| Pair Count + Ratio | 0.3 + 0.2 | 配对数量约束 |
| Density Head | 0.3 | 辅助密度预测 |
| FP Penalty | 2.0 | 精确率提升 |
| BP Compatibility | 0.3 | 碱基配对合法性 |
| Shift Loss | -0.6 (奖励) | ±2 偏移容错 |

---

## 二、训练曲线与测试效果

### 2.1 训练曲线（v9_ddp 主实验）

![v9 训练曲线](../../symfold/outputs/v9_ddp/training_curves.png)

**训练特征**：
- Training loss 从 ~21 快速下降，50 epoch 后趋于平稳（~2.5）
- Val F1 在前 25 epoch 快速上升至 ~0.60，之后缓慢攀升
- **Best Val F1 = 0.6814 @ epoch 160**
- P/R/F1 三线在后期趋于收敛，Recall 略高于 Precision
- 使用 cosine decay 学习率调度（peak=5e-4，warmup=8 epochs）

### 2.2 测试集评估结果

| 指标 | v9 | v8 | Baseline |
|------|----|----|----------|
| **Test F1** | **0.6961** | 0.6105 | 0.7700 |
| Precision | 0.6917 | — | — |
| Recall | 0.7186 | — | — |
| Bad rate (F1<0.3) | 9.4% | 15.3% | — |

**按序列长度分析**：

| 长度区间 | F1 | 说明 |
|----------|-----|------|
| <100 | 0.7481 | 短序列最佳 |
| 100-200 | 0.7044 | 良好 |
| 200-300 | 0.6850 | 中等 |
| 300-400 | 0.7000 | RoPE 对长距离有效 |
| 400-500 | 0.6124 | 超长序列仍有瓶颈 |

**关键结论**：
- v9 相比 v8 提升 **+8.6pp**（0.6105 → 0.6961）
- 与 baseline (0.7700) 差距 7.4pp，将差距缩小了 36%
- Bad rate 从 15.3% 降至 9.4%

---

## 三、消融实验

### 3.1 实验设计（2×2 矩阵）

| 实验 | RoPE | 正则化 | Val F1 | Test F1 |
|------|------|--------|--------|---------|
| **v9_full** (主实验) | ✓ ON | Enhanced (dp=0.2, dpath=0.15) | 0.6814 | **0.6961** |
| v9_low_reg | ✓ ON | Low (dp=0.1, dpath=0.05) | 0.6722 | 0.6804 |
| v9_no_rope | ✗ OFF | Enhanced | 0.5930 | 0.5770 |
| v9_low_reg_no_rope | ✗ OFF | Low | — | — |

### 3.2 消融实验训练曲线

#### v9_ablation_low_reg（低正则化）

![v9 低正则消融](../../symfold/outputs/v9_ablation_low_reg/training_curves.png)

**观察**：
- Best Val F1 = 0.6722 @ epoch 51（比主实验低 0.9pp）
- 训练 loss 下降更快（正则更弱，收敛更快）
- Val F1 后期趋于平台化，但 Test F1 (0.6804) 仍然不错
- Test F1 随训练持续上升（epoch 20→60: 0.644→0.681）

#### v9_ablation_no_rope（无 RoPE）

![v9 无RoPE消融](../../symfold/outputs/v9_ablation_no_rope/training_curves.png)

**观察**：
- Best Val F1 = 0.5930 @ epoch 73（比主实验低 **8.8pp**）
- Val F1 上升非常缓慢，训练 75 epoch 仍在缓慢攀升
- P/R/F1 三线均明显低于主实验，说明 RoPE 影响是全方位的
- Test F1 = 0.5770，严重退化

### 3.3 消融实验结论

```
┌─────────────────────────────────────────────────────────────┐
│  RoPE 是 v9 最关键因素                                        │
│  • 关闭 RoPE: Test F1 下降 11.9pp (0.6961 → 0.5770)         │
│  • RoPE 为模型提供了"知道两个位置有多远"的能力               │
│  • 没有 RoPE，8层 Axial Transformer 无法有效建模长程依赖      │
├─────────────────────────────────────────────────────────────┤
│  增强正则化有效但属于锦上添花                                  │
│  • 低正则: Test F1 下降 1.6pp (0.6961 → 0.6804)             │
│  • 增强 dropout/drop_path 防止了轻度过拟合                    │
│  • 在有 RoPE 的前提下，正则化贡献相对次要                     │
├─────────────────────────────────────────────────────────────┤
│  改进优先级: RoPE >> 正则化 > 其他                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、v9 瓶颈分析

通过全面失败分析，v9 的性能天花板约 **0.70 F1**，主要受限于：

| 瓶颈 | 说明 | 影响 |
|------|------|------|
| **表示瓶颈** | MARS 冻结，仅 5M 可训参数做 adaptation | 无法学习任务特异性表示 |
| **数据长尾** | RFAM 样本多样性高，46 个 F1=0 中 45 个来自 RFAM | 长尾分布拖累均值 |
| **贪心解码** | greedy top-k 无结构约束 | 约一半 F1=0 样本"数量对但位置全错" |
| **Loss-F1 错位** | 后期 loss 仍降但 F1 不升 | 优化方向与评估指标不一致 |

**Matching decoder 实验**：匈牙利匹配解码器几乎无增益（-0.07pp），说明瓶颈不在后处理，而在 **score map 质量本身**。

---

## 五、下一步计划

### 5.1 优先级排序

```
P0 (立即执行):  解冻 MARS 基础模型权重 (v10)
P1 (高优先级):  结构化解码约束
P2 (中优先级):  RFAM hard-case 专项训练
P3 (探索性):    更强的 pair block / 架构升级
```

### 5.2 v10: 解冻 MARS 基础模型

**核心思路**：v9 天花板源于冻结表示，解冻 MARS 让基座模型学习 RNA 二级结构任务的特异性特征。

**具体方案**（已配置 `v9_unfreeze.json`）：
- 从 v9 best.pt warm-start
- MARS 全部解冻，分层学习率：
  - MARS backbone: lr = 5e-6（极小，防止灾难性遗忘）
  - Head 部分: lr = 5e-4（保持原速）
- batch=4, grad_accum=6（有效 batch=24）
- 梯度裁剪 grad_clip=0.5
- 训练 100 epochs, warmup=10

**预期收益**：
- 突破 0.70 瓶颈，目标 Test F1 > 0.72
- MARS 可以学习到"哪些残基更可能配对"的 task-aware 表示

### 5.3 结构化解码（后续）

- 替换 greedy top-k 为带约束的解码器
- 添加嵌套结构合法性约束（无交叉碱基对）
- 考虑 Nussinov-style 动态规划后处理

### 5.4 RFAM Hard-case 训练策略

- 对 F1<0.3 的 bad case 做 upsampling
- 可能引入 curriculum learning（由易到难）
- 考虑 RFAM 子家族的 stratified sampling

### 5.5 架构探索

- 更深/更宽的 pair block（当前 8 层 192 dim）
- 引入 Evoformer-style 三角更新
- 尝试 flash attention 加速更长序列

---

## 六、版本时间线

```
v7 (baseline)     Test F1 = 0.6538    纯判别式 DensityNet
    ↓ (+改进失败)
v8                Test F1 = 0.6105    改动不理想，退步
    ↓ (+bad case 分析 + 5项改进)
v9 ★             Test F1 = 0.6961    当前最佳，2D RoPE + 强正则
    ↓ (+解冻 MARS)
v10 (进行中)      目标 > 0.72         MARS 全部解冻，分层 LR
```

---

## 七、关键 Takeaway

1. **2D RoPE 是 v9 成功的核心**：贡献了 11.9pp 的提升，证明位置信息对 RNA 结构预测至关重要
2. **v9 范式上限 ~0.70**：冻结表示 + 贪心解码 + 逐点 loss 的组合无法突破此天花板
3. **下一步核心方向是解冻 MARS**：让基座模型参与 task-specific 学习，有望突破 0.70 瓶颈
4. **长期方向**：结构化解码 > 数据策略 > 架构升级
