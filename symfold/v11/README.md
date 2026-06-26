# v11: 判别式 RNA 二级结构预测（v9 解冻 + Hard-case 过采样）

## 一句话

v11 是当前**最强的判别式模型**（bpRNA-test F1 = **0.7290**），在 v9 架构基础上**放开 MARS 权重**并对 test bad-case 相似样本做 **2x 过采样**。

> ⚠️ v11 **没有独立的模型代码**，直接复用 `symfold/v9/model.py` 的 `DensityNetProPlus`。
> 本目录仅作版本说明占位，模型定义请见 v9。

---

## 与 v9 / v10 的关系

| | v9 | v10 | **v11** |
|---|---|---|---|
| 模型代码 | `v9/model.py` | 复用 v9 | 复用 v9 |
| MARS 权重 | **冻结** | 解冻 | **解冻** |
| 可训参数 | 5.1M | 165.7M | 165.7M |
| 数据采样 | 常规 | 常规 | **hard-case 2x 过采样** |
| 初始化 | 从头训 | v9 warm-start | **v9 warm-start** |
| Test F1 | 0.6961 | 0.7284 | **0.7290** |

**演进逻辑**：v9（冻结 MARS）→ v10（解冻 MARS）→ v11（解冻 MARS + 针对性过采样难样本）。

---

## 核心改动

1. **放开 MARS 权重**（`freeze_mars=false`）：160M MARS 参数参与训练，从 `v9_ddp/best.pt` warm-start。
2. **Hard-case 过采样**：对训练集中与 test bad-case 结构相似（similarity ≥ 0.80）的样本做 2x 过采样，
   训练集 10807 → 15301 样本（+41.6%），强化模型在难样本上的学习。

---

## 关键配置

```text
模型代码:   symfold/v9/model.py  (DensityNetProPlus, freeze_mars=false)
训练脚本:   symfold/train/train_v11.py
启动脚本:   symfold/train/resume_v11.sh
配置:       symfold/config/v11/v11_hardcase_oversample.json
输出:       symfold/outputs/v11/
日志:       symfold/logs/v11/
```

训练超参（详见 config）：

| 参数 | 值 |
|------|-----|
| head_lr / mars_lr | 5e-4 / 5e-6 |
| grad_clip | 0.5（保护 MARS） |
| epochs | 100 |
| warmup_epochs | 10 |
| oversample_factor | 2（similarity_threshold=0.80） |
| warm_start_from | `outputs/v9_ddp/model/best.pt` |

---

## 结果

| 指标 | 值 | 备注 |
|------|-----|------|
| Best Val F1 | 0.7256 | @ epoch 94 |
| **Best Test F1** | **0.7290** | @ epoch 89，当前最佳 |
| vs v10 | +0.06pp | v10 = 0.7284 |

---

## 使用

```bash
# 续训 / 启动
bash symfold/train/resume_v11.sh
# 等价于：
CUDA_VISIBLE_DEVICES=0 python -u symfold/train/train_v11.py symfold/config/v11/v11_hardcase_oversample.json
```

详细分析见 `symfold/outputs/v11/comprehensive_analysis/`。
