# v12 首轮完整训练失败分析

> 时间：2026-06-30  
> 对象：`symfold/outputs/v12/history.json`、`symfold/logs/v12/run_v12_patch_20260629_193410.out`、`symfold/config/v12/v12_flow_dit.json`  
> 结论基于已完成的 100 epoch 全量训练记录；未包含 best checkpoint 的额外独立 test report。

## 1. 关键结果

| 指标 | 数值 |
|---|---:|
| Best Val F1 | **0.5879 @ epoch 93** |
| Best Val Precision / Recall | 0.5703 / 0.6297 |
| Best Val MCC | 0.5917 |
| epoch 99 periodic full Test F1 | **0.5934** |
| epoch 99 Test Precision / Recall | 0.5718 / 0.6401 |
| epoch 99 Test MCC | 0.5974 |
| Val GT pairs / Pred pairs @ best | 30.22 / 33.17 |
| Trainable params | 16.2M / 176.8M total |

对比：v10/v11 判别式路线 Test F1 ≈ 0.728~0.729；v6 早期生成式路线 Test F1 ≈ 0.608。v12 首轮 patch-space FlowDiT 目前低于 v6，也显著低于 v10/v11。

## 2. 训练现象

1. **不是完全学不动**：Val F1 从 epoch 0 的 0.3338 上升到 0.5879，Test F1 从 epoch 19 的 0.4365 上升到 epoch 99 的 0.5934。
2. **后期进入平台期**：epoch 80~99 的 Val F1 基本在 0.57~0.59 摆动，继续降 loss 已难以带来 F1 提升。
3. **不是单纯配对数量错**：best epoch 的 `val_pred_pairs=33.17`、`val_gt_pairs=30.22`，预测数量只轻微偏多，核心问题是配对位置/身份不准确。
4. **Recall 高于 Precision**：best Val P/R=0.5703/0.6297，Test P/R=0.5718/0.6401，说明模型能找到不少真实配对附近的信号，但 FP 仍多，精确定位不足。

## 3. 当前最可能原因

### 3.1 `patch_size=4` 是首要嫌疑

v12 主干在 `(L/4)×(L/4)` patch space 运行，再 unpatch 回 full contact map。这个设计节省显存，但对 RNA contact map 这类稀疏、单格精确的任务有天然风险：

- 一个 patch 覆盖 4×4 个 full-resolution cell；
- F1 评估对位置完全严格，偏 1~2 格也会算 FP+FN；
- patch token 学到“这一块可能有配对”不等价于能恢复“哪一个精确格子配对”。

这可以解释当前现象：pair count 接近 GT，但 F1 上不去。

### 3.2 MARS frozen 限制了上限

v12 配置 `freeze_mars=true`，总参数 176.8M 但可训只有 16.2M。历史结果显示，v9→v10 的关键跃升来自 MARS 解冻：Test F1 从 0.6961 到约 0.728。v12 仍冻结 MARS，又引入生成式和 patch 压缩，条件特征难以针对“配对身份”端到端重塑。

### 3.3 生成式训练目标与最终离散 F1 不完全一致

训练时模型学习 `p(x_1=1 | x_t,t,RNA)`，推理时经过 CTMC tau-leap，再用 score-based greedy projection。最终 F1 取决于：

1. score map 排序；
2. threshold；
3. greedy 一对一 projection；
4. 采样步数和随机轨迹。

这些组件的误差会叠加。当前 loss 继续下降但 F1 平台，说明 BCE/Dice/pair-count 已经能做密度校准，却不足以优化最终 matching 后的严格 F1。

### 3.4 `eval_num_steps=8` 可能低估或扰动评估

配置中训练采样步数 `num_steps=20`，但验证使用 `eval_num_steps=8`。虽然最终 projection 主要依赖 `p_last`，少步数仍会改变最后一次前向的 `x_t/t` 分布，可能影响 score map。需要固定 best checkpoint 扫描 `eval_num_steps=8/20/50`、`threshold=0.3~0.7` 后才能判断真实上限。

### 3.5 当前 loss 更会校准数量，不够惩罚“错位高分”

`pair_count` 和 `ratio_penalty` 能控制预测配对数；但对 ±1/±2 位移、同一 stem 附近错配，当前 hard F1 没有容忍，loss 也没有显式的局部结构对齐或 shifted-label 机制。v11 提案里已指出 shifted FP 占比较高，v12 更依赖 patch unpatch，错位问题会更突出。

## 4. 是否说明模型架构有问题？

结论：**不能简单说整个 FlowDiT 架构错了，但当前 v12 配置确实存在架构/训练范式上的瓶颈。**

更准确的判断：

1. **生成式路线本身未被证明不可行**：v6 已到 0.608，v12 也能稳定到 0.59，说明离散 flow 能学到结构信号。
2. **当前 `patch_size=4 + frozen MARS + 单 flow head` 的组合上限偏低**：它牺牲了 full-resolution 精度，也没有 MARS 端到端适配来补偿。
3. **v12 失败更像“精确定位能力不足”而不是“完全不会预测 RNA 结构”**：配对数量合理、Recall 不低，但 Precision/F1 卡住。

因此下一步不建议直接推翻全部架构；应先做小规模、单变量诊断。

## 5. 建议优先实验

按性价比排序：

1. **推理扫描（不重训）**：用 `best.pt` 扫 `eval_num_steps=[8,20,50]`、`threshold=[0.3,0.4,0.5,0.6,0.7]`，确认是不是采样/阈值低估。
2. **解码消融（不重训或少改代码）**：比较 raw score threshold、greedy matching、top-k by GT density oracle、不同 projection mode，判断瓶颈在 score map 还是 projection。
3. **patch 消融**：训练 `patch_size=2`（必要时开 gradient checkpointing / 降 batch），验证 full-resolution 恢复是否是主瓶颈。
4. **MARS unfreeze 小 LR**：参考 v10，尝试 `freeze_mars=false`，MARS LR 用 `1e-6~5e-6`，先短训看 Val F1 是否突破 0.60。
5. **加入 shifted/soft label 或局部容忍 loss**：对 GT 周围 ±1 cell 给 soft target，缓解 patch unpatch 引起的错位惩罚。
6. **辅助 direct head / density head**：恢复 v6/v10 风格的直接监督头，让模型在生成式目标外同时优化判别式 contact score。

## 6. 当前结论一句话

v12 F1 上不去的直接表现是：**配对数量基本对了，但精确配对位置错得多**。最可能根因是 `patch_size=4` 带来的 full-resolution 信息瓶颈，叠加 MARS frozen 和生成式采样/投影目标不完全对齐；不是单纯训练没跑够，也不能直接归咎为整个架构彻底错误。
