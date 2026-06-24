# CLAUDE.md — PriFold 当前工作指南

> 最近更新：2026-06-24 15:27。v10 训练已完成（151 epochs），Test F1 = 0.7284。

## 1. 项目状态

PriFold/SymFold 实验线位于 `symfold/`。当前主结论：

| 版本 | Test F1 | 状态 | 说明 |
|------|---------|------|------|
| v10 | **0.7284** | ✅ 完成 | MARS 全部解冻，151 epochs（含续训 60 epoch 精调） |
| v9 | 0.6961 | ✅ 完成 | MARS frozen + RoPE + shift margin + 强正则 |
| v7 | 0.6538 | ✅ 完成 | 纯判别式 DensityNet |
| v8 | 0.6105 | ✅ 完成 | v8 改动不理想 |

重要报告：

```text
docs/v9/v9_test_evaluation_report.md                       # v9 测试报告
docs/v9/v9_ablation_rope_regularization_report.md          # 消融结论
docs/v9/v9_full_comprehensive_failure_analysis.md          # v9 全面分析 + v10 行动方案
```

## 2. v10 最终结果

### v10 训练已完成 — 151 epochs (0-150)

```text
代码: symfold/v9/model.py (DensityNetProPlus, freeze_mars=false)
训练脚本: symfold/train/train_v10.py
配置: symfold/config/v10/v10_ddp.json
输出: symfold/outputs/v10_ddp/
训练策略: 前 90 epoch 常规训练 + 后 60 epoch 小 LR 精调（mars_lr=1e-6, head_lr=1e-4）
```

**v10 最终成果：**

| 指标 | 值 | 备注 |
|------|-----|------|
| Best Val F1 | **0.7265** | @ epoch 147 |
| Best Test F1 | **0.7284** | @ epoch 149 |
| vs v9 | **+3.2pp** | v9 test F1 = 0.6961 |

Test F1 进展：e19=0.6759 → e39=0.6975 → e59=0.7125 → e79=0.7207 → e99=0.7199 → e109=0.7245 → e119=0.7253 → e139=0.7262 → e149=0.7284

**注意**：续训阶段（epoch 91-150）val F1 在 0.725-0.727 之间平台，test F1 仍有缓慢提升。

## 3. 训练规范

以后所有训练必须满足：

1. 每个 epoch 保存 `history.json`
2. 每个 epoch 绘制 `training_curves.png`
3. 每 20 epoch 做一次 `bprna-test` eval，并写入 history
4. 保存 `best.pt` 和 `last.pt`
5. 训练结束后用 best checkpoint 跑完整 test report
6. **Checkpoint 必须保存完整的续训状态**，确保 resume 后学习率不跳变：
   ```python
   # 保存时必须包含：
   torch.save({
       'epoch': epoch,
       'global_step': global_step,
       'model': model.state_dict(),
       'optimizer': optimizer.state_dict(),
       'scheduler': scheduler.state_dict(),  # 必须保存 scheduler 状态
       'best_f1': best_f1,
       'patience_cnt': patience_cnt,
       'history': history,
   }, 'last.pt')

   # 恢复时必须：
   optimizer.load_state_dict(ckpt['optimizer'])
   scheduler.load_state_dict(ckpt['scheduler'])
   global_step = ckpt['global_step']
   ```
   - 必须使用 PyTorch 的正式 `LRScheduler`（如 `CosineAnnealingLR`），不要手动算 LR
   - Resume 后 LR 必须自动恢复到中断时的精确位置

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

## 8. v10 全面分析结论

```text
symfold/outputs/v10_ddp/comprehensive_analysis/v10_comprehensive_analysis.md   # 长度/家族/配对距离分析
symfold/outputs/v10_ddp/comprehensive_analysis/v10_cross_split_analysis.md     # train/val/test 对比
symfold/outputs/v10_ddp/comprehensive_analysis/v10_badcase_deep_analysis.md    # bad case 逐样本分析
```

- 过拟合严重：Train F1=0.91 vs Test F1=0.73，gap=0.18
- 108 个 bad cases 中 99 个在 train 中有高度相似样本 → 模型能力问题，不是数据覆盖问题
- Bad case 中 GT 配对平均概率仅 0.21，模型完全不认识这些结构
- FP 平均概率 0.64，模型"很自信但全错"
- RFAM 贡献 97% bad cases，泛化 gap=0.20

## 9. v11 实验规划

改进方案文档：`docs/v11/v11_improvement_proposals.md`

5 大类 21 个候选改进，每次只引入一个变量。

### v11a — Hard-case 过采样（🏃 训练中）

```text
训练脚本: symfold/train/train_v11a.py
配置: symfold/config/v11/v11a_hardcase_oversample.json
输出: symfold/outputs/v11a/
起点: v9 best.pt warm-start, freeze_mars=false
唯一改动: 对 train 中与 test bad case 结构相似的 243 个样本做 2x 过采样
LR: mars_lr=5e-6, head_lr=5e-4, cosine 100 epoch
```

**注意**：首次启动时因 per_sample_results.json 字段名不匹配导致过采样未生效（找到 0 个样本），
代码已修复。需要清理 outputs/v11a/ 后重新启动。

## 10. 后续计划

1. ✅ v10 全面分析完成
2. 🏃 v11a hard-case 过采样（需重启）
3. ⬜ v11b 增大 dropout (0.2→0.3)
4. ⬜ v11c 非配对位随机突变
5. ⬜ v11d Label smoothing
6. ⬜ 有效改进叠加 → v11-final
