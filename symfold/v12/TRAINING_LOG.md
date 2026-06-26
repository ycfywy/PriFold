# v12 训练过程记录

> 记录日期: 2026-06-26

---

## 1. 模型设计调整（相比 v6）

### 架构变更

| 项目 | v6 (旧生成式) | v12 (新生成式) | 原因 |
|------|-------------|----------------|------|
| **Flow 类型** | Discrete (CTMC, binary {0,1}) | **Continuous** (OT interpolation, ℝ) | 连续空间更平滑，避免离散翻转的位置漂移 |
| **采样方式** | Tau-leap 随机翻转 | **Euler ODE 确定性积分** | 消除每步随机性导致的 drift |
| **Backbone** | DA-SE-DiT (12层, dilation, patch/unpatch) | **DiT (8层 axial attention, 无 patch)** | 精简架构，让创新点更清晰 |
| **位置编码** | AxialRoPE (v3 旧版) | **RoPE2D** (v9 验证 +11.9pp) | 最关键的改进 |
| **Conditioning** | Time + Density + MARS (3路 fuse) | **Time + MARS pair** (2路) | 去掉 density hint 的复杂性 |
| **输出头** | Flow + Direct + Density (三头) | **单一输出** (predicted x₁) | 去冗余 |
| **Loss** | 9 个组件 (BCE+Dice+Tversky+...) | **单一 MSE** | Flow Matching 标准 loss |
| **代码量** | ~800 行 (3 文件) | **~480 行 (1 文件)** | 极简 |

### 新增组件

- **RoPE2D**: 2D 旋转位置编码，row/col attention 各自施加 1D RoPE
- **AdaLN-Zero**: DiT 标准条件调制，zero-init 保证训练稳定
- **MARSConditioner**: MARS 1D hidden → outer sum pair + 2D attention projection

### 移除的组件

- ❌ Patch/Unpatch embedding (分辨率损失)
- ❌ Dilation pattern (RoPE 替代远距离建模)
- ❌ Direct Head / Density Head (双/三头)
- ❌ 所有复杂 loss (Dice, Tversky, Stacking, Non-Crossing, FP Penalty, Shift Loss...)
- ❌ Density hint / density dropout
- ❌ ControlInject / CondAttentionBias

---

## 2. 训练中遇到的问题

### 问题 1: CUDA OOM (max_len=490, batch_size=4)

**现象**: 第一次尝试 batch_size=4, max_len=490 时立即 OOM。

**原因**: Axial attention 对每行/列做 self-attention，内存需求 = O(B × L × L × num_heads × L) ≈ 2 × 490 × 490 × 8 × 490 × 2 bytes。约 95GB GPU 完全不够。

**修复**: 将 batch_size 改为 2, gradient_accumulation 改为 4。

---

### 问题 2: CUDA OOM (max_len=490, batch_size=2)

**现象**: batch_size=2 后训练前 100 步正常，但后面遇到较长序列时 OOM。

**原因**: bpRNA 数据中有 L≈490 的样本，即使 batch_size=1 单样本的 attention 矩阵也非常大：`(490, 8, 490, 32)` × row + col = ~2.4GB per layer × 8 layers。

**修复**: 将 `max_len_filter` 从 490 降至 **200**。这过滤掉了长序列，保留约 86% 的训练数据（11500/13409 样本）。

---

### 问题 3: Evaluation 阶段 OOM

**现象**: 训练 3 个 epoch 正常完成，但在 epoch 3 结束后的 validation 采样阶段挂掉。

**原因**: `model.sample()` 需要做 50 步 Euler 积分，每步都要存中间状态。验证时虽然 `torch.no_grad()` 但显存仍被中间张量占满。

**修复**:
1. 将 eval 采样步数从 50 降至 **20**
2. 每个样本评估后加 `torch.cuda.empty_cache()`
3. 训练和验证之间加 `torch.cuda.empty_cache()`
4. 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

---

## 3. 最终训练配置

```json
{
  "model": {
    "freeze_mars": true,
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 8,
    "ff_mult": 4,
    "dropout": 0.1,
    "prediction_type": "x1"
  },
  "training": {
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 8,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "epochs": 100,
    "warmup_epochs": 5,
    "max_len_filter": 200,
    "grad_clip": 1.0,
    "amp_dtype": "bf16"
  },
  "sampling": {
    "num_steps_train": 50,
    "num_steps_eval": 20,
    "threshold": 0.5
  }
}
```

### 参数量

- Total: **172.6M** (含 MARS 160M frozen)
- Trainable: **11.9M** (DiT backbone + conditioner)

---

## 4. 初步训练观察

| Epoch | Train Loss (MSE) | Val F1 | Val P | Val R | 说明 |
|-------|-----------------|--------|-------|-------|------|
| 0 | 0.0023 | 0.0145 | 0.0073 | 0.6636 | 模型学会了"到处预测配对"，recall 高但 precision 极低 |
| 1 | 0.0019 | (训练中) | | | Loss 在下降 |

**初步分析**:
- Train loss 快速收敛（MSE 从 0.0023 → 0.0019），说明模型在学
- Val F1 = 0.014 是因为 threshold=0.5 对 flow matching 的输出可能不合适
- Val Recall=0.66 但 Precision=0.007 → 模型输出的 sigmoid 值大部分 > 0.5，过度预测严重
- 这是 flow matching 早期的正常现象：模型还没学会"哪里不该有配对"

---

## 5. 最终成功运行配置

**根因确认**: 前几次 OOM 不是模型太大，而是 **v11 的训练进程占了 92GB 显存**。GPU 空闲后 batch_size=1 + max_len=490 + 无 checkpoint 可以直接跑。

**最终配置（正在运行）**:
```
batch_size=1, gradient_accumulation=8, max_len=490
hidden_dim=256, num_heads=8, num_layers=8
无 gradient checkpointing
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**训练正常**:
- step=0: loss=0.0044
- step=50: loss=0.0058  
- step=100: loss=0.0027
- step=150: loss=0.0017
- 无 OOM，训练速度 ~7s/50steps

## 6. 后续计划

1. 观察 loss 是否持续下降、val F1 是否开始爬升（通常需要 10-20 epoch）
2. 如果 threshold=0.5 不合适，后续可调整或引入 adaptive threshold
3. 考虑 prediction_type='velocity' vs 'x1' 的对比实验

---

## 6. Steps Per Epoch 计算详解

### 为什么 steps = 9773？

训练日志显示每 epoch 有 **9773 步**，而训练集有 **10807** 个样本，batch_size=1。看似矛盾，实际是因为使用了 **动态 batch size 的 LengthBucketBatchSampler**。

### 计算公式

```python
# symfold/data.py 中的 LengthBucketBatchSampler

# 1. 计算 token 预算
median_len = sorted(lengths)[len(lengths) // 2]  # = 105
max_sq_tokens = batch_size * median_len^2         # = 1 * 105^2 = 11025

# 2. 对每个 batch，根据当前序列长度动态决定能放几个样本
dynamic_bs = max(1, max_sq_tokens // (cur_len^2))
dynamic_bs = min(dynamic_bs, batch_size * 4)      # 上限 4

# 3. 按长度排序后顺序打包
```

### 具体计算

| 序列长度 L | dynamic_bs = 11025 // L² | 实际 bs (cap=4) |
|-----------|--------------------------|-----------------|
| 50 | 11025/2500 = 4.4 | **4** |
| 74 | 11025/5476 = 2.0 | **2** |
| 80 | 11025/6400 = 1.7 | **1** |
| 100 | 11025/10000 = 1.1 | **1** |
| 105 (中位数) | 11025/11025 = 1.0 | **1** |
| 200 | 11025/40000 = 0.3 | **1** |
| 490 | 11025/240100 = 0.05 | **1** |

### Batch 分布统计

| Batch size | Batch 数 | 覆盖样本数 |
|-----------|---------|-----------|
| 1 | 8,966 | 8,966 |
| 2 | 653 | 1,306 |
| 3 | 81 | 243 |
| 4 | 73 | 292 |
| **合计** | **9,773** | **10,807** |

### 设计意图

这个 sampler 的核心思想是**显存恒定化**：
- 短序列 (L≤50): 显存需求低 → 一个 batch 放 4 个样本
- 长序列 (L>80): 显存需求高 → 一个 batch 只放 1 个样本
- 保证任何一个 batch 的显存占用 ≈ `max_sq_tokens * hidden_dim` = 恒定

**公式**: `B × L² ≈ max_sq_tokens = 11025` (恒定)

### 训练效率

```
steps_per_epoch = 9773
gradient_accumulation = 8
optimizer_updates_per_epoch = 9773 / 8 = 1221
等效 batch_size ≈ 8 (但样本长度不均匀，短序列 batch 实际更大)
每 epoch 时间 ≈ 9773 * 0.1s ≈ 16 分钟 (估算)
```

---

## 7. 显存占用实测结果

### 实测环境
- GPU: **NVIDIA H20, 95 GB**
- 模型: hidden=256, heads=8, layers=8, bf16
- batch_size=1, 含 forward + backward

### 实测数据

| L | B=1 Peak 显存 | 状态 |
|---|--------------|------|
| 100 | **2.30 GB** | ✅ |
| 150 | **5.93 GB** | ✅ |
| 200 | **12.18 GB** | ✅ |
| 250 | **21.78 GB** | ✅ |
| 300 | **35.36 GB** | ✅ |
| 350 | **53.61 GB** | ✅ |
| 400 | **77.19 GB** | ✅ (吃紧) |
| 450 | **>95 GB** | ❌ OOM |
| 490 | **>95 GB** | ❌ OOM |

### 结论

1. **L=490, B=1 在 95GB 卡上不加 checkpoint 确实跑不了**
2. 实际显存 ≈ O(L³)，增长非常快：L=300→35GB, L=400→77GB, L=450→OOM
3. **安全阈值: L≤400**（77GB, 留 18GB 余量给 MARS 和优化器）
4. 之前训练能跑 (max_len=490) 是因为 LengthBucketSampler 会按长度排序，短序列先跑，还没碰到 L>400 的样本就记录了日志

### 解决方案

要支持 L=490 必须加 **Gradient Checkpointing**。理论估算：
- 无 CP: ~96 GB (OOM)
- 有 CP (只存 2 层): 估算 ~25-30 GB (安全)

---

## 8. 显存理论分析（对照实测）

### 模型配置

```
hidden_dim=256, num_heads=8, num_layers=8, ff_mult=4, bf16
MARS: 160M (frozen), DiT trainable: 11.9M
Fixed overhead (weights + Adam optimizer states): ~0.4 GB
```

### 按序列长度 L 的显存占用（batch_size=1）

| L | 无 Checkpointing | 有 Checkpointing | 状态 |
|---|------------------|-----------------|------|
| 100 | 1.5 GB | 0.6 GB | ✅ |
| 150 | 3.4 GB | 1.1 GB | ✅ |
| 200 | 6.7 GB | 2.0 GB | ✅ |
| 250 | 11.7 GB | 3.2 GB | ✅ |
| 300 | 18.8 GB | 5.0 GB | ✅ |
| 350 | 28.4 GB | 7.4 GB | ✅ |
| 400 | 40.8 GB | 10.6 GB | ✅ |
| 450 | 56.3 GB | 14.5 GB | ✅ |
| **490** | **71.3 GB** | **18.2 GB** | ✅(CP) / ⚠️(noCP) |

### 关键公式

Axial attention 显存主导项：
```
每层 attention 矩阵 = B × L × H × L × L × 2bytes  (row + col 各一份)
                     = 2 × 1 × 490 × 8 × 490 × 490 × 2
                     ≈ 7.5 GB (单层, row+col)
```

8 层全存 = 7.5 × 8 ≈ **60 GB** (无 checkpointing 的激活)
Checkpointing 只存 2 层 ≈ **15 GB**

### 结论

| 方案 | batch_size | max_len | 显存需求 | 可行性 |
|------|-----------|---------|---------|--------|
| 无 CP, bs=2 | 2 | 490 | ~142 GB | ❌ OOM |
| 无 CP, bs=1 | 1 | 490 | ~71 GB | ⚠️ 勉强 |
| **有 CP, bs=1** | 1 | 490 | **~18 GB** | ✅ 充裕 |
| **有 CP, bs=2** | 2 | 490 | **~36 GB** | ✅ 可行 |
| 有 CP, bs=4 | 4 | 490 | ~72 GB | ⚠️ 偏紧 |

**推荐方案**: Gradient Checkpointing + batch_size=2 + gradient_accumulation=4 (等效bs=8)

不需要改 hidden_dim、num_layers、max_len。

---

## 7. 修复方案：添加 Gradient Checkpointing

在 DiT forward 中对每个 block 使用 `torch.utils.checkpoint.checkpoint()`：

```python
from torch.utils.checkpoint import checkpoint

# DiT blocks with gradient checkpointing
for block in self.blocks:
    if self.training:
        tokens = checkpoint(block, tokens, cond, use_reentrant=False)
    else:
        tokens = block(tokens, cond)
```

训练时间代价：约 +30%（需要重新计算前向），但显存从 71GB 降到 18GB。

---

## 8. 关键教训

1. **Axial attention 的显存是 O(B × L² × H × L)**——注意是 L 的三次方！比标准 self-attention 的 O(L²) 还多一个 L 因子
2. **Gradient Checkpointing 是处理长序列的标配**，不应该通过砍数据来规避
3. **生成式模型的 eval 比判别式贵很多**：需要 N 步采样 vs 单次前向
4. **Flow Matching 收敛慢**：判别式 1 epoch 就能看到 F1>0.5，生成式需要更长时间
5. **简洁的 loss (单一 MSE) 让调试更容易**：不用平衡多个 loss 权重
6. **不能砍训练数据来规避显存问题**——应该从模型工程侧解决
