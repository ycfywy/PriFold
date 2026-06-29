# v12 训练与修复记录

> 最近更新：2026-06-26

## 1. 当前 v12 定位

v12 当前是 **Discrete Bernoulli Flow Matching + FlowDiT**，不是 continuous OT flow。

核心流程：

```text
x_t ~ Bernoulli(t · x_1 + (1-t) · rho_0)
FlowDiT(x_t, t, MARS features) → logit p(x_1=1)
loss = Focal BCE + Dice + PairCount + RatioPenalty + Stacking + NonCrossing
inference = CTMC tau-leap sampling + score projection
```

## 2. 已修复问题

### P0: double zero-init 导致 attention 分支可能冻死

原问题：AdaLN gate zero-init 且 `row_out/col_out` 也 zero-init，导致 `pair + g*h` 中 `g=0` 且 `h=0`，attention 分支可能拿不到梯度。

修复：保留 AdaLN-Zero，取消 `row_out/col_out` zero-init。

### P1: `threshold` 参数无效，投影依赖随机 `x_t`

原问题：`sample(..., threshold=0.5)` 没有实际使用 threshold；`project_to_valid_contact_map` 用 `x_t * score` 作为候选，最终结果过度依赖 tau-leap 随机路径。

修复：projection 支持 `min_score` 和 `use_sample_mask`；默认使用 score-based projection，并应用 threshold。

### P1: 非法位置 mask 不统一

原问题：BCE 用 valid mask，但 logit 输出和 NonCrossingLoss 没有统一排除 `|i-j|<3`。

修复：`_dit_forward()` 输出时 mask 掉短距离和 padding；NonCrossingLoss 使用合法位置 mask。

### P2: 生成式输入 `x_t` 初始不起作用

原问题：`xt_proj` zero-init，训练初期模型退化为纯判别式。

修复：`xt_proj.weight` 使用 `normal_(std=0.02)`。

### P2: run 脚本破坏 resume

原问题：`run_v12.sh` 每次运行都会删除 `last.pt/best.pt/history.json/log`。

修复：默认 resume；只有显式 `FRESH_RUN=1` 才备份并清理。

### P2: 数据增强缺失

原问题：训练脚本硬编码 `augment=False`，与 v9/v10 不公平。

修复：从 config 读取 augmentation，默认 `select=0.20, replace=0.40`。

### P2: checkpoint 状态不完整

修复：`last.pt` 保存 `global_step`、optimizer、scheduler、history、best_f1、patience_cnt。

## 3. 当前推荐配置

```json
{
  "model": {
    "patch_size": 4,
    "use_gradient_checkpoint": false,
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 8,
    "dropout": 0.2,
    "drop_path": 0.15
  },
  "training": {
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "max_len_filter": 490,
    "augmentation": {"enabled": true, "select": 0.20, "replace": 0.40},
    "test_eval_every": 20
  },
  "sampling": {
    "num_steps": 20,
    "eval_num_steps": 8
  }
}
```

## 4. v6-style 加速改造

v12 已参考 v6 改为 **patch-space FlowDiT**：

```text
full L×L x_t / MARS attention
    → learned patch embedding (patch_size=4)
    → FlowDiT backbone on (L/4)×(L/4)
    → unpatch + refine
    → full L×L logit + full-resolution loss
```

这样保持 loss / projection 仍在 full-resolution contact map 上，但最重的 pair axial attention 与 `OuterProductMean` 不再跑完整 `L×L` 网格。

## 5. 后续必须验证

1. 跑一个单步 backward 梯度检查，确认 `row_qkv/row_out/col_qkv/col_out/xt_patch/pair_patch` grad norm 非零。
2. 重新从头训练 v12，旧 checkpoint 与 patch-space 结构不兼容，不建议 resume。
3. 评估 `patch_size=4`、`eval_num_steps=8/20` 对速度与 F1 的影响。
4. 评估 threshold / projection mode 对 F1、pred/GT ratio 的影响。
5. 对比 v6/v10/v12 的错误模式，确认生成式是否提供互补性。
