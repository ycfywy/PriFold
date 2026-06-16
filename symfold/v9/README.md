# PriFold v9: DensityNet-Pro+ (Efficiency & Accuracy)

## 概述

v9 基于 v8 bad case 分析结果的**效率+精度双重优化版本**。

### 两个版本

| 版本 | 配置文件 | GPU | 启动方式 |
|------|----------|-----|----------|
| **单卡** | `symfold/config/v9/v9_single.json` | 1×H20 | `bash symfold/train/run_train.sh symfold/config/v9/v9_single.json` |
| **双卡 DDP** | `symfold/config/v9/v9_ddp.json` | 2×H20 | `bash symfold/train/run_train_v9_ddp.sh` |

---

## v8 → v9 改进

### 基于 v8 Bad Case 分析的改进

| v8 问题 | v9 解决方案 |
|---------|-------------|
| 偏移预测是第一失败模式 (32.7%) | **shift_radius 1→2, weight 0.3→0.6** |
| 长序列 Recall 崩塌 (350+: R=0.516) | **length_decay 0.3→0.15 + budget_floor=0.6** |
| FP penalty 过强导致欠预测 | **fp_penalty_weight 3.0→2.0** |
| score_threshold 偏高 | **threshold 0.45→0.43** |
| ratio_penalty 过紧 | **threshold 1.15→1.20** |
| BP compat 未启用 | **bp_compat_enabled=true, weight=0.3** |

### 训练效率提升（拉满 H20）

| 优化项 | v8 | v9 | 加速预估 |
|--------|----|----|----------|
| **torch.compile** | 无 | `reduce-overhead` 模式 | +20~40% |
| **梯度累积** | 无 | 2 步 (单卡) / 1 步 (双卡) | 2x effective batch |
| **max_sq_tokens** | `12×median²` | **4,000,000** | 大 batch 更稳定 |
| **DataLoader workers** | 默认(0) | **8 workers** | 数据加载不卡 |
| **pin_memory** | 无 | **启用** | CPU→GPU 拷贝加速 |
| **prefetch_factor** | 默认(2) | **4** | 预加载更多 batch |
| **向量化 OHEM** | Python for循环 | **批量 topk** | Loss 计算 ~3x |
| **cudnn.benchmark** | False | **True** | 固定长度桶内更快 |
| **DDP (双卡)** | 单卡 | **2 GPU 并行** | 吞吐 ~1.8x |

---

## 架构

```
RNA seq → MARS-LX (frozen, 160M)
        → 1D hidden (1056→192→96) + 2D attn (72→48→48)
        → Pair Feature (outer prod + attn + seq_pair + pos_bias)
        → Input Projection (→ 192 dim)
        → 8× Axial Transformer (6 heads, dim_head=32, DropPath)
        → Contact Logit + Density Head
        → Loss (8 components) / Predict (budget_floor + BP filter)
```

**Trainable params**: ~4.6M (v8: 3.56M, 略增但更高效利用)

---

## Loss 系统

与 v8 相同的 8 组件结构，关键调整：

| Loss 组件 | v8 | v9 | 说明 |
|-----------|----|----|------|
| FP Penalty | 3.0 | **2.0** | 降低，避免过度抑制 |
| BP Compat | 0.0 (关闭) | **0.3 (启用)** | 过滤非标准配对 |
| Shift Loss | -0.3, radius=1 | **-0.6, radius=2** | 更强偏移奖励 |
| Ratio Threshold | 1.15 | **1.20** | 放宽过预测惩罚 |

---

## 推理策略

| 参数 | v8 | v9 | 说明 |
|------|----|----|------|
| score_threshold | 0.45 | **0.43** | 略降，减少 no_prediction |
| length_decay | 0.3 | **0.15** | 大幅降低，恢复长序列 recall |
| **budget_floor** | 无 | **0.6** | 最低 budget 系数不低于 0.6 |
| BP filter | 关闭 | **启用** | 过滤非 AU/GC/GU 预测 |

L=400 时:
- v8: factor = (100/400)^0.3 = 0.55 → 预测量被压缩 45%
- v9: factor = max((100/400)^0.15, 0.6) = max(0.76, 0.6) = 0.76 → 仅压缩 24%

---

## 模型配置对比

| 参数 | v8 | v9 |
|------|----|----|
| hidden_dim | 160 | **192** |
| num_heads | 4 | **6** |
| dim_head | 40 | **32** |
| ff_mult | 4 | 4 |
| dropout | 0.2 | **0.15** |
| drop_path | 0.1 | 0.1 |
| 2D proj channels | 32 | **48** |
| Total trainable | 3.56M | ~4.6M |

---

## 启动

### 单卡 (1×H20)

```bash
cd /root/aigame/dannyyan/PriFold
bash symfold/train/run_train.sh symfold/config/v9/v9_single.json
```

### 双卡 DDP (2×H20)

```bash
cd /root/aigame/dannyyan/PriFold
bash symfold/train/run_train_v9_ddp.sh
# 或手动:
torchrun --nproc_per_node=2 --standalone \
  symfold/train/train_v9_ddp.py symfold/config/v9/v9_ddp.json
```

### 查看日志

```bash
# 单卡
tail -f symfold/logs/v9_single/v9_single.stdout.log

# 双卡
tail -f symfold/logs/v9_ddp/v9_ddp.stdout.log
```

---

## 预期改善

| 指标 | v8 (epoch 137) | v9 预期 | 改善来源 |
|------|----------------|---------|----------|
| Test F1 | 0.616 | **0.67-0.70** | 更好的 P/R 平衡 + 更多训练容量 |
| Test Precision | 0.613 | 0.65-0.68 | BP compat filter |
| Test Recall | 0.645 | **0.68-0.72** | 降低 length_decay + 降低 FP penalty |
| pred/gt | 1.175 | 1.05-1.15 | Ratio threshold 放宽 |
| F1=0 count | 48 | **<30** | 降低 threshold + budget_floor |
| 训练速度 | 1x | **1.5-2x** (单卡) / **3-3.5x** (双卡) | torch.compile + DDP |
| Epoch 时间 | ~300s | **~150-200s** (单卡) / **~100s** (双卡) | 全方位效率优化 |

---

## 文件清单

```
symfold/
├── v9/
│   ├── __init__.py
│   ├── model.py              # DensityNetProPlus 模型
│   └── README.md             # 本文档
├── config/v9/
│   ├── v9_single.json        # 单卡配置
│   └── v9_ddp.json           # 双卡配置
└── train/
    ├── train_v9.py           # 单卡训练入口
    ├── train_v9_ddp.py       # DDP 双卡训练入口
    └── run_train_v9_ddp.sh   # DDP 启动脚本
```
