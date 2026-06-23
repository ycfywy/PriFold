# CLAUDE.md — PriFold 当前工作指南

> 最近更新：2026-06-23 13:06。v10 续训中（从 best.pt epoch 91 开始，小 LR 精调 60 epoch）on cuda0。

## 1. 项目状态

PriFold/SymFold 实验线位于 `symfold/`。当前主结论：

| 版本 | Test F1 | 状态 | 说明 |
|------|---------|------|------|
| v10 | **0.7207** | 🏃 续训中 | MARS 全部解冻，epoch 91→151，小 LR 精调 |
| v9 | 0.6961 | ✅ 完成 | MARS frozen + RoPE + shift margin + 强正则 |
| v7 | 0.6538 | ✅ 完成 | 纯判别式 DensityNet |
| v8 | 0.6105 | ✅ 完成 | v8 改动不理想 |

重要报告：

```text
docs/v9/v9_test_evaluation_report.md                       # v9 测试报告
docs/v9/v9_ablation_rope_regularization_report.md          # 消融结论
docs/v9/v9_full_comprehensive_failure_analysis.md          # v9 全面分析 + v10 行动方案
```

## 2. 当前正在跑的实验

### v10 续训 — cuda0

```text
目的: MARS 解冻已有效（+2.5pp），小 LR 精调看能否进一步提升
代码: symfold/v9/model.py (DensityNetProPlus, freeze_mars=false)
训练脚本: symfold/train/train_v10.py
配置: symfold/config/v10/v10_ddp.json
日志: symfold/logs/v10_ddp/v10.log
输出: symfold/outputs/v10_ddp/
续训参数: mars_lr=1e-6, head_lr=1e-4, 余弦退火 60 epoch, warmup 3
当前状态: 从 best.pt (epoch 90) 续训，epoch 91→151
```

**注意**：之前续训（epoch 96-105）因 LR 跳变退化已回滚，history 截断到 epoch 90。

**v10 阶段性成果（epoch 0-95）：**

| 指标 | 值 | 备注 |
|------|-----|------|
| Best Val F1 | **0.7214** | @ epoch 90 |
| Best Test F1 | **0.7207** | @ epoch 79 |
| vs v9 | **+2.5pp** | v9 test F1 = 0.6961 |
| patience | 5/30 | 仍在爬升中 |

Test F1 进展：e19=0.6759 → e39=0.6975 → e59=0.7125 → e79=0.7207

查看日志：

```bash
tail -f symfold/logs/v10_ddp/v10.stdout.log
```

## 3. 训练规范

以后所有训练必须满足：

1. 每个 epoch 保存 `history.json`
2. 每个 epoch 绘制 `training_curves.png`
3. 每 20 epoch 做一次 `bprna-test` eval，并写入 history
4. 保存 `best.pt` 和 `last.pt`
5. 训练结束后用 best checkpoint 跑完整 test report

## 4. 可视化规范

所有 `training_curves.png` 必须参考 v8 的格式，包含 **6 个子图**（3×2 布局）：

1. **Training Loss** — train loss + bce（双线）
2. **Validation F1 / MCC** — val F1 + val MCC + best F1 标记点
3. **Validation P / R / F1** — precision + recall + F1 三线
4. **Learning Rate** — LR 曲线（如有分层 LR，画多条）
5. **Test F1 (periodic eval)** — 每 20 epoch 的 test F1
6. **Test MCC (periodic eval)** — 每 20 epoch 的 test MCC

越多指标越好。参考文件：`symfold/outputs/v8_full/training_curves.png`

## 5. v10 的设计

v10 和 v9 使用**完全相同的模型代码**（`symfold/v9/model.py`）。

唯一区别：`freeze_mars=false`，MARS 160M 参数全部可训练。

| | v9 | v10 |
|---|---|---|
| freeze_mars | true | **false** |
| 可训参数 | 5.09M | **165.7M** |
| MARS LR | — | 5e-6 |
| Head LR | 5e-4 | 5e-4 |
| 初始化 | 从头训 | 从 v9 best.pt warm-start |
| grad_clip | 1.0 | 0.5（保护 MARS） |

## 6. 环境

```bash
cd /root/aigame/dannyyan/PriFold
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
source activate RNADiffFold_torch260
export PYTHONPATH=/root/aigame/dannyyan/PriFold
```

## 7. 消融结论（已完成）

- **RoPE 是 v9 最关键因素**：关闭后 Test F1 从 0.6961 降到 0.5770（-11.9pp）
- **增强正则化有效但次要**：低正则 Test F1 0.6804（-1.6pp）
- **Matching decoder 无增益**：验证证明瓶颈在 score map 质量，不在解码层
- **v9 天花板**：冻结 MARS + 贪心解码 + 逐点 loss 的范式上限约 0.70

## 8. 后续计划

1. ✅ v10 MARS unfreeze 已验证有效：test F1 0.6961 → 0.7207（+2.5pp）
2. 🏃 续训到 150 epoch，看 F1 能否突破 0.73
3. 如果趋势放缓，考虑：降 LR 精调 / 增大数据 / 调 sampling 阈值
4. 续训完后用 best checkpoint 做完整 test report
