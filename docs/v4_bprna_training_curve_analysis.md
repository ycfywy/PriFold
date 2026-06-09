# v4_bprna 训练曲线分析

> 分析时间：2026-06-05。基于 249 epoch 完整训练（epoch 0-248，无事故）。

## 训练快照

| 指标 | 值 |
|---|---|
| Best val F1 | **0.4946** @ epoch 245 |
| Best test F1 | **0.4869** @ epoch 219 |
| Final train loss | 0.0094 |
| LR (final) | 6.88e-05 (peak 8e-05) |
| pred/gt ratio (test) | 1.47（全程稳定） |
| Precision / Recall | 0.43 / 0.61 |

---

## 1. 曲线是否正常？

**整体正常，但有几个异常信号值得关注。**

### 正常的方面

- Training loss 呈标准的指数衰减：从 0.85 → 0.01，前 50 epoch 快速下降，之后缓慢收敛
- Val F1 持续上升，没有 overfitting 回落
- Test F1 和 Val F1 同步上涨，泛化 gap 稳定
- 没有 LR 事故或训练崩溃

### 异常/值得关注

1. **Val F1 波动极大**（σ≈0.02，相邻 epoch 可差 0.04），说明评估对 sampling 和 projection 很敏感
2. **pred/gt ratio 全程锁定在 1.47-1.52**，从 epoch 9 到 epoch 239 几乎没有变化——说明模型从未学会正确控制预测数量
3. **train loss 在 epoch 150 后已基本见底**（0.010-0.013），但 val F1 仍在涨——loss 和指标脱钩
4. **上升速度逐渐放缓**：epoch 0→50 涨了 +0.19（F1），epoch 150→245 只涨了 +0.05

---

## 2. 为什么 val F1 虽在上升但绝对值很低？

Best val F1 = 0.4946，对比主线 PriFold bprna-test F1=0.77，差距 28%。原因分析：

### 2.1 极端类别不平衡：输出空间 99.6% 是负样本

bpRNA 数据统计：
- 平均序列长度 134nt
- 平均每条序列只有 30 个 contact pair
- 输出 L×L map 有 ~18,000 个 pixel，其中只有 ~60 个是正样本
- **正负比 ≈ 1:301**

这意味着：
- BCE loss 的 99.6% 梯度来自 negative pixel（模型已经预测对的那些 0）
- loss 可以很低（0.01）但模型对真正困难的 positive 边的预测仍然不精确
- 即使模型对每个 positive edge 的预测从 0.6 提升到 0.7，loss 几乎不变，但 F1 会显著提升

### 2.2 F1 是 hard-threshold 后的离散指标

模型输出的是连续 score，必须经过 projection（threshold=0.5 + budget_fraction=0.35）才变成二值 contact map。这个非线性映射导致：
- 模型对一个边预测 0.49 vs 0.51，loss 几乎无差别，但 F1 差别是 "miss vs hit"
- loss 空间里很小的改善可以对应 F1 上不小的变化（反之亦然）

### 2.3 生成式模型 vs 判别式模型的固有差距

v4 是 Discrete Flow Matching 生成式模型，本质是学习从噪声到 contact map 的采样过程。判别式模型（PriFold 主线）直接输出 logit，不需要走 flow → sample → projection 这条路。生成式模型的 F1 受限于：
- sampling 步数（20步 vs 理论上无穷步）
- flow 学习的精度（前向 + 后向过程都有误差）
- projection 后处理的信息损失

### 2.4 Precision 是瓶颈，不是 Recall

| Epoch | Precision | Recall | F1 |
|---|---|---|---|
| 25 | 0.29 | 0.43 | 0.34 |
| 145 | 0.39 | 0.55 | 0.44 |
| 245 | 0.44 | 0.61 | 0.49 |

Recall 已经到 0.61（模型能找到 61% 的真实 contact），但 Precision 只有 0.44（预测的 contact 里只有 44% 是对的）。**模型倾向于"宁多勿少"，导致大量 false positive 拖低 F1。**

---

## 3. 模型能否从如此微弱的 loss 信号中学到东西？

### 证据表明：能，但越来越低效

| 阶段 | train loss | Val F1 变化 | 效率 |
|---|---|---|---|
| epoch 0-50 | 0.85 → 0.03 | 0.18 → 0.37 (+0.19) | 高 |
| epoch 50-100 | 0.03 → 0.019 | 0.37 → 0.42 (+0.05) | 中 |
| epoch 100-200 | 0.019 → 0.011 | 0.42 → 0.47 (+0.05) | 低 |
| epoch 200-248 | 0.011 → 0.009 | 0.47 → 0.49 (+0.02) | 很低 |

从 epoch 200 开始，train loss 只降了 0.002（从 0.011 到 0.009），但 val F1 仍然涨了 0.02。**这说明模型确实还在从微弱信号中学习，只是效率极低——每个 epoch 的有效信息量已经非常小了。**

### 为什么还能学？

虽然 aggregate loss 很低，但 loss 的组成中仍有部分 **informative gradient**：
- 少量困难的 positive edge（模型预测在 0.3-0.7 区间的边）贡献关键梯度
- `direct_weight=0.3` 的 direct head loss 比 flow loss 更直接，可能是后期学习的主要驱动力
- `pair_count_loss`（虽然权重只有 0.05）在每个 batch 中持续提供"你预测太多了"的信号

### 核心问题：有效梯度被 easy negative 淹没

对于一个 134×134 的 map，模型每个 forward pass 产出 ~18,000 个 loss term。其中：
- ~17,940 个 negative pixel：模型已经预测接近 0，gradient ≈ 0
- ~60 个 positive pixel：其中可能只有 10-20 个是"困难的"（预测 0.3-0.7）

**真正有用的梯度可能只占 0.1% 的 loss terms。** 这就是为什么 loss=0.01 看起来很低但模型还在缓慢学习。

---

## 4. 和数据集的关系：Train 很好学，Test 差别大？

### 4.1 不是经典的 overfitting

如果是 "train 过拟合、test 不行"，我们应该看到：
- train loss 极低 ✓（确实是）
- val/test F1 下降或停滞 ✗（val F1 仍在涨）

val F1 和 test F1 同步上涨，说明**不是泛化问题**。模型在 train 和 val/test 上的行为是一致的。

### 4.2 是 loss function 和评估指标的脱节

真正的问题是：train loss（BCE 为主）和最终指标（F1）衡量的不是同一个东西。

- **BCE loss 衡量**：每个 pixel 的概率预测有多准
- **F1 衡量**：threshold 后的二值结构和 ground truth 的 overlap

模型可以在 train set 上把每个 pixel 的 BCE 压到极低（因为 99.6% 是 easy negative），但这不代表 positive edge 的预测质量很高。

### 4.3 数据集特点加剧了这个问题

bpRNA 训练集：
- 21,628 条序列，平均长度 134
- 大量短序列（<100: 45%）→ 短序列的 contact map 比较简单，容易学
- 长序列（300+: 7%）→ 难度大，但样本少，对 loss 贡献小

**短序列的 "easy learning" 拉低了整体 loss，掩盖了长序列上的困难。** 但测试时按样本平均 F1，长序列的低 F1 会显著拖后腿。

### 4.4 pred/gt ratio 固定在 1.47 的含义

从 epoch 9 到 239，测试集 pred/gt 始终在 1.47-1.52，这非常稳定。说明：
- 过预测不是模型"还没学好"的暂态，而是**结构性偏差**
- `pair_count_weight=0.05` 的惩罚太弱，没有有效约束预测数量
- `default_budget_fraction=0.35` 的 hard cap 可能本身就偏高

---

## 5. 其他训练过程分析

### 5.1 学习率是否有问题？

**当前 LR schedule 极度平坦，几乎等于恒定 LR 训练：**

```
epoch   0: lr = 1.6e-5 (warmup start)
epoch   5: lr = 8.0e-5 (warmup done, peak)
epoch  50: lr = 7.96e-5 (-0.5%)
epoch 100: lr = 7.82e-5 (-2.3%)
epoch 200: lr = 7.26e-5 (-9.3%)
epoch 248: lr = 6.88e-5 (-14.0%)
```

cosine schedule 设了 total=999，但只跑了 248 epoch（25%）。在 cosine 的前 25%，衰减极慢。**实际上这 249 个 epoch 就是在用 ~7.5e-5 的恒定 LR 跑了一遍。**

### 5.2 LR 是太低还是太高？

**两方面证据：**

**偏低的证据：**
- 学习速度确实慢：250 epoch 只到 F1=0.49
- train loss 早就见底，说明当前 LR 下的优化已经到了 local minimum 附近
- val F1 波动大但 envelope 上升慢，说明每步更新幅度太小，随机波动占主导

**偏高的证据：**
- val F1 波动 σ≈0.02，如果 LR 更高波动会更大
- train loss 仍在缓慢下降（还没完全 plateau）

**我的判断：对模型的可训参数规模来说，8e-5 可能不低。**

v4 DiT 部分约 8-10M 参数。对 8M 参数的 Transformer，8e-5 是比较标准的 LR。问题不在于 LR 绝对值，而在于：

1. **LR 没有经历衰减周期**：缺少 "高 LR 探索 → 低 LR 精调" 的过程
2. **loss landscape 可能比较 flat**：在 0.01 附近，梯度方向对 F1 的提升效率很低

### 5.3 加大 LR 的风险与收益

| LR 方案 | 预期效果 | 风险 |
|---|---|---|
| 维持 8e-5（当前） | 继续缓慢上升 | 无风险，但 ROI 低 |
| 提高到 2e-4 | 可能跳出当前 basin | 可能打乱已学到的结构 |
| 提高到 5e-4 | 快速探索新区域 | 大概率崩溃 |
| cosine restart 8e-5→1e-6 | 在当前 basin 内精调 | 收益有限但稳 |

**推荐：cosine restart with higher peak。** 从 best.pt 出发，LR 从 1.5-2e-4 开始（比原来高 2x），短周期（100-150 epoch）衰减到 1e-6。这样既有"冲一把"的探索阶段，又有低 LR 精调阶段。

### 5.4 为什么 precision 始终上不去？

回顾整个训练过程的 precision 变化：

```
epoch   1: P=0.16
epoch  25: P=0.29
epoch 100: P=0.35
epoch 200: P=0.41
epoch 245: P=0.44
```

Precision 确实在涨，但涨速比 recall 慢。核心原因：

- **loss function 不直接惩罚 false positive 的"数量"**：BCE 惩罚的是每个 pixel 的概率误差，不是"你总共预测多了多少条边"
- **`pair_count_weight=0.05` 太弱**：相对于主 loss 的 0.01，pair_count_loss × 0.05 可能只有 0.001 级别的贡献
- **`default_budget_fraction=0.35` 是 hard cap 但可能设偏高了**：如果实际密度 < 0.35，模型会被允许填满到 0.35

### 5.5 每 epoch 训练时间与 GPU 利用

- 训练集 21,628 条，batch_size=8 → 每 epoch ~2,703 步
- H20 GPU 97GB，bf16 训练
- 249 epoch 无 OOM，训练稳定

---

## 6. 总结与后续建议

### 核心诊断

| 问题 | 严重程度 | 说明 |
|---|---|---|
| loss 与 F1 脱钩 | ⚠️ 高 | BCE 被 easy negative 主导，有效梯度极弱 |
| 过预测固化 | ⚠️ 高 | pred/gt=1.47 全程不变，pair_count 约束失效 |
| LR schedule 太平 | ⚠️ 中 | 等效恒定 LR，缺少探索/精调切换 |
| 学习速度慢 | ⚠️ 中 | 250 epoch 只到 F1=0.49，ROI 偏低 |
| 模型容量限制 | ❓ 待验证 | 256 dim × 9 layers 可能不够 |

### 建议优先级

1. **解决过预测**：`pair_count_weight` 从 0.05 提到 0.2-0.3，或直接加 hard ratio penalty
2. **LR cosine restart**：从 best.pt 出发，peak LR 1.5e-4，短周期 100-150 epoch
3. **加强 positive 边的梯度信号**：提高 `focal_gamma`（2→3）或加 Dice/soft-F1 loss
4. **降低 budget_fraction**：从 0.35 降到 0.28-0.30，直接限制过预测上限
5. **如果以上都 plateau**：考虑加大模型（hidden_dim 256→384）
