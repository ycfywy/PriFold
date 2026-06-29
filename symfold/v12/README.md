# v12: Discrete Flow Matching + FlowDiT for RNA Secondary Structure

## 一句话

v12 是一个**生成式 RNA 二级结构预测模型**：用 **离散 Bernoulli Flow Matching / CTMC tau-leap** 生成 contact map，用 **FlowDiT（DiT + axial attention + RoPE2D）** 作为 backbone。

---

## 1. 当前真实范式

v12 当前实现是 **Discrete Flow Matching**，不是 continuous OT flow。

训练时：

```text
x_1 = GT contact map ∈ {0,1}^{L×L}
t ~ Uniform(0,1)
x_t ~ Bernoulli(t · x_1 + (1-t) · ρ_0)
model predicts logit p(x_1=1 | x_t, t, RNA)
loss = Focal BCE + Dice + PairCount + RatioPenalty + Stacking + NonCrossing
```

推理时：

```text
x_0 ~ Bernoulli(ρ_0)
for step in tau-leap schedule:
    p_x1 = model(x_t, t, RNA)
    rate_01, rate_10 = CTMC rates
    x_t = stochastic flip(x_t, rates)
final = score-based greedy projection(p_x1, threshold)
```

---

## 2. 架构（双轨 single + pair，类 Evoformer）

```text
RNA sequence
    → MARS-LX frozen extractor
        → 1D hidden states + 2D attention maps
    → 双表示构造 + v6-style patch compression（默认 patch_size=4）
        - single s (B,L/4,Ds)     ← MARS hidden 投影后 1D patchify
        - pair   z (B,L/4,L/4,Dp) ← MARS attention 72→128→64 后 2D patchify
        - noisy x_t 同样 patchify 后注入 pair
    → DualFlowDiT blocks × 8（在 patch space 上运行）
        - single self-attention (1D) + RoPE + SDPA + AdaLN-Zero(t)
        - OuterProductMean(single) → pair        （1D→2D 通信）
        - pair row/col axial attention + RoPE2D + SDPA + AdaLN-Zero(t)
        - FFN + DropPath
    → unpatch + refine → full-resolution contact logit
    → tau-leap sampling + score projection
```

设计要点：
- **v6 式 patch-space flow backbone**：`patch_size=4` 时，主干 pair attention / OPM 从 `L×L` 降到 `(L/4)×(L/4)`，attention 主体计算量约降 16×，OPM 中间张量约降 16×。
- **删除 seq_oh**：碱基配对兼容性交给模型从 MARS 表示中自行学习。
- **删除 pair_1d outer-concat**：single 改走独立 1D 轨道，通过 `OuterProductMean` 注入 pair，替代低效的 expand+concat。
- **pair_2d 少压缩后再 patchify**：`Conv2d 72→128→64` 后用 learned patch embedding 压到 `hidden_dim`。
- **显存提示**：`OuterProductMean` 中间张量为 `(B,L/patch,L/patch,opm_hidden²)`，默认 `patch_size=4`、`opm_hidden=16`。

---

## 3. 关键修复记录

### 3.1 修复 double zero-init 梯度冻结

原实现同时：

- AdaLN gate zero-init (`g1=0`)
- `row_out/col_out` zero-init (`h=0`)

导致 `pair + g1*h` 中 `g1` 与 `h` 都为 0，attention 分支可能拿不到梯度。当前已保留 AdaLN-Zero，但取消 `row_out/col_out` 的 zero-init。

### 3.2 修复 x_t 输入初始无效

`xt_proj` 原来 zero-init，使生成式输入 `x_t` 初始完全不起作用。当前改为 `normal_(std=0.02)`。

### 3.3 修复 sample threshold 无效

原 `sample(..., threshold=0.5)` 中 threshold 没被使用，且投影依赖 `x_t * score`。当前改为：

- `threshold` 传入 projection
- 默认 `score-based projection`，不再强依赖最终随机 `x_t`
- 可选 `use_sample_mask=True` 才使用 `x_t` 作为候选 mask

### 3.4 统一非法位置 mask

`_dit_forward()` 输出 logit 时会统一 mask：

- `|i-j| < 3`
- padding 区域

`NonCrossingLoss` 也改为使用合法位置 mask，避免非法短距离位置干扰 loss。

### 3.5 训练工程修复

- 默认开启 `use_gradient_checkpoint=true`
- `run_v12.sh` 默认 resume，不再删除 checkpoint/history
- 只有 `FRESH_RUN=1` 才会备份并清理旧结果
- checkpoint 保存 `global_step`、optimizer、scheduler、history、patience_cnt
- 训练数据增强按配置开启（默认与 v9/v10 一致：select=0.20, replace=0.40）

---

## 4. 当前配置摘要

```json
{
  "model": {
    "freeze_mars": true,
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 8,
    "dropout": 0.2,
    "drop_path": 0.15,
    "use_rope": true,
    "patch_size": 4,
    "use_gradient_checkpoint": false,
    "rho_0": 0.005
  },
  "training": {
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
    "max_len_filter": 490,
    "augmentation": {"enabled": true, "select": 0.20, "replace": 0.40}
  },
  "sampling": {
    "num_steps": 20,
    "eval_num_steps": 20,
    "threshold": 0.5
  }
}
```

---

## 5. 与 v6/v10 的关系

| 版本 | 范式 | 核心特点 |
|------|------|----------|
| v6 | Discrete Flow | patch_size=4, DA-SE-DiT, 多头 loss |
| v10 | Discriminative | MARS 解冻，DensityNetProPlus，直接预测 contact |
| v12 | Discrete Flow + DiT | v6-style patch-space FlowDiT + RoPE2D + SDPA + MARS frozen |

v12 的目标不是直接替代 v10，而是探索生成式 RNA folding 是否能产生不同错误模式，并与判别式模型互补。
