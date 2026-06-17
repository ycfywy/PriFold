# CLAUDE.md — PriFold 当前工作指南

> 最近更新：2026-06-17 20:25。当前已停止 v10，正在跑 v9 消融：`v9_no_rope` on cuda0，`v9_low_reg` on cuda1。

## 1. 项目状态

PriFold/SymFold 实验线位于 `symfold/`。当前主结论：

| 版本 | Test F1 | 状态 | 说明 |
|------|---------|------|------|
| v9 | **0.6961** | ✅ 完成 | 当前最佳，MARS frozen + RoPE + shift margin + 强正则 |
| v10 | 0.6698 | ⏸ 已停 | MARS 后2层 unfreeze，效果不如 v9，训练方法需重做 |
| v7 | 0.6538 | ✅ 完成 | 纯判别式 DensityNet |
| v8 | 0.6105 | ✅ 完成 | v8 改动不理想 |

重要报告：

```text
docs/v9/v9_test_evaluation_report.md                 # v9 测试报告
docs/v9/v9_ablation_rope_regularization_plan.md      # 当前消融计划
docs/v10/v10_analysis_and_next_steps.md              # v10 问题分析
```

## 2. 当前正在跑的实验

### A. v9_no_rope — cuda0

```text
目的: 验证 2D RoPE 是否有效
配置: symfold/config/v9_ablation/v9_no_rope.json
日志: symfold/logs/v9_ablation_no_rope/
输出: symfold/outputs/v9_ablation_no_rope/
关键变量: v9.use_rope=false，其余保持 v9 full 强正则
```

### B. v9_low_reg — cuda1

```text
目的: 验证增强正则化是否有效
配置: symfold/config/v9_ablation/v9_low_reg.json
日志: symfold/logs/v9_ablation_low_reg/
输出: symfold/outputs/v9_ablation_low_reg/
关键变量: dropout=0.05, drop_path=0.03, augmentation=0.05/0.15, weight_decay=0.005
```

查看日志：

```bash
tail -f symfold/logs/v9_ablation_no_rope/v9_ablation_no_rope.log
tail -f symfold/logs/v9_ablation_low_reg/v9_ablation_low_reg.log
```

查看 GPU：

```bash
nvidia-smi
```

## 3. 训练规范

以后所有训练必须满足：

1. 每个 epoch 保存 `history.json`
2. 每个 epoch 绘制 `training_curves.png`
3. 每 20 epoch 做一次 `bprna-test` eval，并写入 history
4. 保存 `best.pt` 和 `last.pt`
5. 训练结束后用 best checkpoint 跑完整 test report

## 4. 环境

```bash
cd /root/aigame/dannyyan/PriFold
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
source activate RNADiffFold_torch260
export PYTHONPATH=/root/aigame/dannyyan/PriFold
```

## 5. 当前判断

- v9 的强项可能来自 RoPE + 正则 + shift margin 的组合。
- v10 最大问题不是单纯训练不够，而是没有从 v9 warm-start，且 partial unfreeze 实现不理想。
- 当前先通过 v9 消融确认：
  1. RoPE 对长序列/长距离配对是否真的有效；
  2. 强正则是否真的提升泛化。

## 6. 后续计划

1. 等 `v9_no_rope` 和 `v9_low_reg` 至少跑到 epoch 100，看趋势。
2. 如果差异明显，继续跑满 200。
3. 若需要验证交互，再跑 `v9_low_reg_no_rope`。
4. 最终生成 `docs/v9/v9_ablation_rope_regularization_report.md`。
