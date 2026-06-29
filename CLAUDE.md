# CLAUDE.md — PriFold 工作指南

> 本文件是 AI 协作的长期工作规范，只记录**稳定的规则与约定**，不记录时效性的实验进度。
> 具体实验结论、报告、分析一律写入 `docs/`。

## 0. 科研严谨性铁律（最高优先级，AI 必须遵守）

**AI 不允许图省事、偷懒、走捷径。** 以下行为严格禁止：

1. **禁止擅自采样/缩减数据集**：评估和分析必须用**全量数据**。
   - 例如对比 train/val/test 时，train 集所有样本必须全部评估，不允许采样子集"加速"。
   - 若确实因算力受限需要采样，必须**先明确征求用户同意**，并在报告显著位置标注。

2. **禁止擅自修改训练数据**：
   - 不允许通过缩小 `max_len_filter` 来规避 OOM（这会丢弃长序列样本）。
   - 显存问题必须从工程侧解决（gradient checkpointing、动态 batch、减小 batch_size 等），不能砍数据。

3. **禁止伪造或估算代替实测**：
   - 显存占用、性能指标等必须实测，不能用理论估算冒充实测结果。
   - 理论估算与实测必须分别标注清楚。

4. **禁止用部分结果代替完整结果**：
   - 跑实验要跑完整流程，不能只跑几个 epoch 就下结论。
   - 报告必须基于真实完整的运行数据。

5. **诚实标注局限**：任何因客观限制而做的简化，必须在产出物中显式说明，不得隐瞒。

违反以上铁律等同于科研不端。宁可慢，不可假。

## 1. 目录组织规范（AI 产出文件时必须遵守）

所有产出物按**类型**归位，并在各自目录下**按版本分开**（如 `v10`、`v11`、`v12`）：

| 类型 | 位置 | 说明 |
|------|------|------|
| **文档 / 报告 / 分析** | `docs/<version>/` | 所有 `.md` 文档、实验报告、分析结论、改进方案 |
| **模型 / history / 可视化** | `symfold/outputs/<version>/` | checkpoint（`best.pt` / `last.pt`）、`history.json`、`training_curves.png` 及其他可视化图 |
| **日志** | `symfold/logs/<version>/` | 训练/评估的运行日志（`.log`） |

规则：

1. **写文档一律放到 `docs/` 下**，并放进对应版本子目录（如 `docs/v11/`）；禁止把报告散落在 `outputs/` 或项目根目录。
2. **`outputs/<version>/` 只放模型产物与可视化**：checkpoint、history、图片。checkpoint 只保留 `best.pt` 和 `last.pt`，不保留中间 `epoch_*.pt`。
3. **`logs/<version>/` 只放日志**。
4. 三类目录都必须**按版本命名分开**，不同版本互不混放。

## 2. 训练规范

以后所有训练必须满足：

1. 每个 epoch 保存 `history.json`
2. 每个 epoch 绘制 `training_curves.png`
3. 默认每 20 epoch 做一次 `bprna-test` eval，并写入 history；若版本配置显式设定不同频率（如 v12 `test_eval_every=10`），以配置为准并在文档说明
4. 保存 `best.pt` 和 `last.pt`（不保留中间 epoch checkpoint）
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

7. **运行脚本默认必须安全续训**：不得每次运行自动删除 `last.pt` / `best.pt` / `history.json` / 日志；若需要从头开始，必须使用显式开关（如 `FRESH_RUN=1`）并先备份旧输出。
8. **训练语义必须由 config 驱动**：数据增强、`max_len_filter`、batch/accumulation、eval 频率等不得在训练脚本中硬编码导致配置失效。

## 3. 可视化规范

所有 `training_curves.png` 必须参考 v8 的格式，包含 **6 个子图**（3×2 布局）：

1. **Training Loss** — train loss + bce（双线）
2. **Validation F1 / MCC** — val F1 + val MCC + best F1 标记点
3. **Validation P / R / F1** — precision + recall + F1 三线
4. **Learning Rate** — LR 曲线（如有分层 LR，画多条）
5. **Test F1 (periodic eval)** — 按配置周期评估的 test F1
6. **Test MCC (periodic eval)** — 按配置周期评估的 test MCC

越多指标越好。参考文件：`symfold/outputs/archive/v8_full/training_curves.png`

## 4. 环境

```bash
cd /root/aigame/dannyyan/PriFold
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
source activate RNADiffFold_torch260
export PYTHONPATH=/root/aigame/dannyyan/PriFold
```

## 5. 主要训练版本概况

实验线位于 `symfold/`。当前主要版本（已完成指标均为 bpRNA-test 全量实测，F1 越高越好；未完成版本标注为待评估）：

| 版本 | 训练范式 | Best Val F1 | Test F1 | epochs | 关键设计 |
|------|---------|-------------|---------|--------|---------|
| **v12** | 生成式离散 Flow Matching（FlowDiT） | 待重训 | 待全量评估 | 100（config） | Bernoulli flow / CTMC tau-leap，v6-style `patch_size=4` 压缩空间，双轨 `single+pair`，MARS frozen |
| **v11** | 监督判别式（v10 基础） | 0.7256 @94 | **0.7290** @89 | 100 | 在 v9 基础上放开 MARS 权重 + 对 test bad-case 相似样本 2x 过采样 |
| **v10** | 监督判别式（Focal BCE + Dice） | 0.7265 @147 | 0.7284 @149 | 151 | MARS 160M **全部解冻**微调，全分辨率 Axial Attention |
| **v9** | 监督判别式 | 0.6814 @160 | 0.6961 | 183 | MARS **冻结** + RoPE + 强正则（v10 的起点） |
| **v6** | 离散 Flow Matching（CTMC） | 0.6059 @213 | 0.6083 @189 | 217 | Patch 下采样 4× + Dilated Axial Attention，生成式 |

要点：

- **v10/v11 仍是当前有完整实测的最强模型**，判别式范式，Test F1 ≈ 0.73；v11 凭 hard-case 过采样小幅超越 v10。
- **v12 是当前生成式新主线**，代码在 `symfold/v12/`：Discrete Bernoulli Flow Matching + `FlowDiT`，不是 continuous OT flow。
- **v12 当前实现要点**：`x_t ~ Bernoulli(t·x_1 + (1-t)·rho_0)`，模型预测 `p(x_1=1|x_t,t,RNA)`；主干参考 v6 在 `patch_size=4` 的压缩空间运行，输出再 unpatch 回 full contact map；推理使用 CTMC tau-leap + score-based projection。
- **v12 旧 continuous-flow checkpoint / 旧说明与当前语义不兼容**；从头训练必须显式 `FRESH_RUN=1 bash symfold/train/run_v12.sh`，默认运行应走安全 resume。
- **v9 → v10 的关键变化**：解冻 MARS（可训参数 5M → 165.7M），从 v9 best.pt warm-start，Test F1 +3.2pp。
- **v6 是早期生成式（Flow Matching）路线**，效率高（patch 下采样使 attention 显存降 ~64×）但精度落后判别式约 12pp。
- v12 细节以 `symfold/v12/README.md`、`symfold/v12/TRAINING_LOG.md` 和 `symfold/config/v12/v12_flow_dit.json` 为准；`docs/training_comparison_v6_v10_v12.md` 中关于 v12 continuous OT / Euler 的内容属于历史说明，阅读时需注意过期。

## 6. 实验结论入口

各版本的详细结论、报告与分析见 `docs/<version>/`，例如：

```text
docs/v9/                      # v9 测试报告、消融结论、失败分析
docs/v10/                     # v10 报告与全面分析
docs/v11/                     # v11 改进方案与各子实验
symfold/v12/README.md         # v12 当前架构与实现说明
symfold/v12/TRAINING_LOG.md   # v12 修复记录与待验证事项
docs/archive/                 # 历史版本 (v6/v7/v8 等) 已归档于此
```

> 各版本的指标进展、逐 epoch 数字等以 `docs/` 中的最新报告为准，本文件只保留概况。
> v12 在未形成正式实验报告前，以 `symfold/v12/` 下的实现说明为准；正式报告仍应写入 `docs/v12/`。
> 历史版本的代码 / 输出 / 日志 / 文档分别归档在 `symfold/archive/`、`symfold/outputs/archive/`、`symfold/logs/archive/`、`docs/archive/`。
