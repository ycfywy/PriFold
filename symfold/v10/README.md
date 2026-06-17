# v10 DensityNet-Ultra

## 概述

基于 v9 (test F1=0.6961) 的瓶颈分析，v10 聚焦两个核心改进：

1. **[U1] Partial MARS Unfreeze** — 解冻最后 2 层 Transformer，让 160M 预训练 LM 的表征适配结构预测任务
2. **[U2] Family-aware Curriculum** — 对难样本（RFAM 家族）进行 oversampling，课程学习渐进增加难度

## v9 → v10 改进依据

| 瓶颈 | 分析 | v10 方案 |
|------|------|---------|
| 模型容量饱和 | 5.09M 参数已收敛，后 22 epochs 无提升 | 解冻 MARS 后 2 层 (+27M trainable) |
| Bad case "位置全错" | 74% bad case 预测数量对但位置全错 | LM 适配让特征更具结构识别力 |
| RFAM 家族表现差 | Bad case 多来自训练集覆盖不足的家族 | Curriculum oversample 2x |
| 训练稳定性 | MARS 层参数大，直接大 LR 会崩 | 分层 LR: MARS 1e-5, head 5e-4 |

## 架构

```text
RNA seq → MARS-LX (160M, 最后2层可训练, LR=1e-5)
        → 1D hidden + 2D attention (with gradient)
        → Pair Feature (outer prod + attn_proj + seq_pair)
        → 2D RoPE + Axial Transformer (8层, 192dim, LR=5e-4)
        → Contact Logit + Density Head
        → Score Threshold (0.43) + Density Budget
        → Contact Map
```

## 参数量

| 组件 | 参数量 | 可训练 | LR |
|------|--------|--------|-----|
| MARS 前 10 层 | ~133M | ❌ 冻结 | — |
| MARS 后 2 层 + norm | ~27M | ✅ | 1e-5 |
| 下游头 (proj+axial+head) | ~5M | ✅ | 5e-4 |
| **总训练参数** | **~32M** | | |

## 关键配置

```json
{
  "v10": {
    "freeze_mars": "partial",
    "unfreeze_last_n": 2,
    "mars_lr": 1e-5
  },
  "training": {
    "batch_size": 8,
    "max_sq_tokens": 400000,
    "gradient_accumulation_steps": 4,
    "grad_clip": 0.5,
    "warmup_epochs": 15,
    "curriculum": {"hard_oversample": 2.0, "start_epoch": 20}
  }
}
```

### 为什么降低 batch/max_sq_tokens

- MARS 后 2 层的梯度需要存储激活值 → 显存增加 ~20-30GB
- 从 max_sq_tokens=600000 降至 400000，batch_size 12→8
- 用 grad_accum=4 补偿有效 batch size

### 为什么 grad_clip=0.5

- MARS 预训练参数很大，梯度范数可能不稳定
- 0.5 比 v9 的 1.0 更保守，防止前几 epoch 参数剧变

### 为什么 warmup=15

- v9 用 warmup=8，但 v10 有 MARS 层参与
- 更长 warmup 让 MARS 层慢慢适应，避免破坏预训练知识

## 训练命令

```bash
bash symfold/train/run_train_v10_ddp.sh
# 或手动:
torchrun --nproc_per_node=2 --standalone \
  symfold/train/train_v10_ddp.py symfold/config/v10/v10_ddp.json
```

## 预期结果

- 目标: test F1 > 0.73 (vs v9 的 0.6961)
- MARS unfreeze 预期贡献: +2-5pp
- Curriculum 预期贡献: +1-2pp（主要减少 bad rate）
- 与 baseline (0.77) 差距: <4%
