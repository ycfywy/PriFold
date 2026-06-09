# PriFold-SymFlow v4 改进说明

更新时间：2026-06-03 11:29

本文记录本次新增的 `symfold/v4` 版本。注意：**本次只写代码与文档，没有启动训练**。

---

## 1. v4 目标

v3 的 bad-case 分析显示，主要失败并不是 backbone 完全不会预测 pair，而是：

1. 低 density RNA 过预测严重；
2. 中长 RNA 的采样候选边错位；
3. 默认 `density_guided=True` 反而压掉正确候选；
4. projection 只从 `x_t==1` 候选中选，正确边若未采样到就无法恢复；
5. `pos_bias` 与 MARS attention 只作为输入 channel，在第 0 层 patch embed 后可能被洗掉。

v4 的核心目标：

> 不再先盲目加大模型，而是把结构先验每层注入、把 score 解码路径做稳，并用 pair-count 校准约束低 density 过预测。

---

## 2. 新增/修改文件

```text
symfold/
├── v4/
│   ├── __init__.py
│   ├── da_se_dit.py        # v4 主干：condition attention bias + ControlInject + direct score head
│   ├── model.py            # v4 模型封装：MARS extract / train forward / score-first sample
│   └── discrete_flow.py    # v5 loss：direct BCE + pair-count loss + 新 projection
├── train_v4.py             # v4 训练入口，复用 v3 训练循环但替换 build_model
├── eval_v4.py              # v4 评估入口，支持 projection_mode / density budget
├── analyze_cases.py        # 已支持 v4 模型选择与 v4 sampling 参数
├── run_train.sh            # 已支持 version=v4 自动选择 train_v4.py
└── config/
    ├── v4_bprna.json
    └── v4_rnastralign.json
```

文档：

- `docs/prifold_symflow_v4_improvements.md`：本文。
- `docs/prifold_symflow_v3_architecture_case_analysis.md`：v3 架构与失败 case 依据。

---

## 3. 架构改进 1：`pos_bias` + MARS attention 每层注入

### v3 问题

v3 的结构条件是：

```text
MARS hidden pair + MARS attention + pos_bias + seq_oh
  → concat channel
  → PatchEmbed2D
  → 9-layer DA-SE-DiT
```

也就是说，`MARS attention` 和 `pos_bias` 只在输入层出现一次。对长 RNA 来说，深层 DiT block 可能已经很难直接利用这些 pair prior。

### v4 改法

新增 `CondAttentionBias`：

```text
cond_pair = concat(mars_attn_2d, pos_bias)
cond_patch = AvgPool2d(patch_size=4)(cond_pair)
attn_bias = Conv1x1(cond_patch) → (B, num_heads, S/4, S/4)
```

每个 `DASEDiTBlockV4` 的 row/col axial attention 都接收 `attn_bias`。

具体逻辑：

- row attention：对固定 `i` 的所有 `(i,j)` token，加 `bias[i,j]` 作为 key-position bias；
- col attention：对固定 `j` 的所有 `(i,j)` token，加 `bias[i,j]` 作为 key-position bias；
- `CondAttentionBias` 最后一层默认 zero-init，避免新分支初始时破坏训练稳定性。

代码位置：

- `symfold/v4/da_se_dit.py::CondAttentionBias`
- `symfold/v4/da_se_dit.py::DilatedAxialAttentionV4`
- `symfold/v4/da_se_dit.py::DASEDiTBlockV4`

---

## 4. 架构改进 2：ControlNet-style 条件刷新

### v3 问题

v3 的条件信号除全局 AdaLN 外，局部 pair condition 只在 patch embed 时注入一次。

### v4 改法

新增 `ControlInjectMLP`：

```text
cond_patch → zero-init Conv MLP → (B, S/4, S/4, hidden_dim)
```

每隔 `control_every=2` 层加回 tokens：

```text
tokens = tokens + control_inject(cond_patch)
```

这个分支也是 zero-init，初始不影响原路径，训练时逐步学会“刷新”MARS attention / pos_bias 条件。

代码位置：

- `symfold/v4/da_se_dit.py::ControlInjectMLP`
- `symfold/v4/da_se_dit.py::DASEDiT_MARS_v4.forward`

---

## 5. 架构改进 3：direct contact score head

### v3 问题

v3 只有 flow logits。最终 projection 用的是：

```text
candidate = final sampled x_t
score = p_x1_last
project(candidate, score)
```

如果正确边没被采样进 `x_t`，即使 score 高也无法恢复。

### v4 改法

新增 direct score head：

```text
final_tokens
  ├─ flow unpatch/refine   → flow_logit，用于 CTMC rates / flow loss
  └─ direct unpatch/refine → direct_logit，用于 score-first projection / direct BCE
```

训练 loss 增加：

```text
loss = flow_loss + direct_weight * BCE(direct_logit, contact) + pair_count_loss
```

推理默认融合：

```text
score = (1 - direct_score_weight) * sigmoid(flow_logit)
      + direct_score_weight * sigmoid(direct_logit)
```

默认 `direct_score_weight=0.5`。

代码位置：

- `symfold/v4/da_se_dit.py::DASEDiT_MARS_v4`
- `symfold/v4/model.py::PriFoldSymFlow_v4.sample`
- `symfold/v4/discrete_flow.py::BernoulliFlowLoss_v5`

---

## 6. Loss 改进：pair-count 校准

v3 的低 density 样本过预测非常严重，`density < 0.1` 时 `pred/gt≈4.71×`。

v4 新增 `pair_count_loss`：

```text
pred_pairs = sigmoid(direct_logit).sum(valid_edges) / 2
pred_density = pred_pairs / L_eff
gt_density = gt_pairs / L_eff
pair_count_loss = SmoothL1(pred_density, gt_density)
```

同时调整：

- `pos_weight_min`: 20 → 10，降低低 density 样本正类过强激励；
- `focal_gamma`: 1.5 → 2.0，加强 hard negative 抑制；
- `density_loss`: MSE → SmoothL1，减少极端样本对 density head 的拉扯；
- 新增 `direct_weight=0.3`、`pair_count_weight=0.05`。

代码位置：

- `symfold/v4/discrete_flow.py::BernoulliFlowLoss_v5`

---

## 7. Sampling / Projection 改进

### 7.1 默认关闭 density-guided rate damping

v3 实测：

| setting | F1 | P | R | pred_pairs |
|---|---:|---:|---:|---:|
| `density_guided=True` | 0.4107 | 0.3732 | 0.4807 | 44.95 |
| `density_guided=False` | **0.4530** | **0.3853** | **0.5960** | 52.10 |

因此 v4 配置默认：

```json
"density_guided": false
```

保留该参数，但不再默认让 density 乘 `rate_01`。

### 7.2 默认 score-only projection

v4 默认：

```json
"projection_mode": "score"
```

即最终不再只从 `x_t==1` 候选边里选，而是直接对所有合法边按 score 做 greedy max matching：

```text
score * valid_mask → greedy max matching → contact map
```

可选模式：

| mode | 说明 |
|---|---|
| `score` | 默认；忽略 `x_t` 候选，直接按 score 解码 |
| `hybrid` | `score + candidate_weight * x_t`，采样候选只作为 bonus |
| `sample` | v3 兼容路径，只从 sampled candidates 中选 |

代码位置：

- `symfold/v4/discrete_flow.py::project_score_to_valid_contact_map`
- `symfold/v4/discrete_flow.py::project_hybrid_contact_map`
- `symfold/v4/model.py::PriFoldSymFlow_v4.sample`

### 7.3 多样本后重投影

v3 的 multi-sample 是先投影、平均、阈值，可能破坏“一碱基一配对”。

v4 改为：

```text
多条 trajectory → 平均 score / 平均 candidate
→ 最后统一 greedy projection 一次
```

这样多样本投票后仍保证合法 matching。

### 7.4 density 只作为可选 projection budget

v4 支持：

```json
"use_density_budget": true
```

此时：

```text
max_pairs = round(density_pred * L_eff * budget_scale)
```

它只限制最终最多选多少 pair，不干扰 CTMC candidate 生成。默认先关闭，等训练后再消融。

---

## 8. 配置说明

新增：

- `symfold/config/v4_bprna.json`
- `symfold/config/v4_rnastralign.json`

关键差异：

```json
"model": {
  "version": "v4",
  "pos_weight_min": 10.0,
  "focal_gamma": 2.0,
  "direct_weight": 0.3,
  "pair_count_weight": 0.05,
  "cond_bias_zero_init": true,
  "control_every": 2,
  "direct_score_weight": 0.5
},
"sampling": {
  "density_guided": false,
  "projection_mode": "score",
  "use_density_budget": false
}
```

---

## 9. 使用方式（暂未训练）

### 9.1 启动训练（之后需要时再运行）

```bash
cd /root/aigame/dannyyan/PriFold
bash symfold/run_train.sh symfold/config/v4_bprna.json
bash symfold/run_train.sh symfold/config/v4_rnastralign.json
```

`run_train.sh` 已支持自动识别 `model.version == "v4"` 或 `task_name` 以 `v4_` 开头，并调用：

```text
symfold/train_v4.py
```

### 9.2 评估（训练后使用）

```bash
python symfold/eval_v4.py \
  --ckpt symfold/outputs/v4_bprna/model/best.pt \
  --out_json symfold/outputs/v4_bprna/eval_best.json
```

消融示例：

```bash
# 回退到 v3 candidate-only projection
python symfold/eval_v4.py --ckpt <ckpt> --projection_mode sample

# hybrid projection
python symfold/eval_v4.py --ckpt <ckpt> --projection_mode hybrid --candidate_weight 0.35

# 启用 density budget，但仍不做 rate damping
python symfold/eval_v4.py --ckpt <ckpt> --use_density_budget 1 --budget_scale 1.0

# 多样本 + 最终重投影
python symfold/eval_v4.py --ckpt <ckpt> --num_samples_per_input 5 --projection_mode score
```

### 9.3 bad-case 分析

`analyze_cases.py` 已支持 v4：

```bash
python symfold/analyze_cases.py \
  --ckpt symfold/outputs/v4_bprna/model/best.pt \
  --test_sets bprna-test \
  --out_dir symfold/outputs/v4_bprna/case_analysis
```

---

## 10. 已完成的非训练验证

本次只做代码层验证，没有启动训练。

已执行：

```bash
python3 -m py_compile \
  symfold/v4/da_se_dit.py \
  symfold/v4/discrete_flow.py \
  symfold/v4/model.py \
  symfold/train_v4.py \
  symfold/eval_v4.py \
  symfold/analyze_cases.py
```

已执行 v4 主干小尺寸 forward self-test：

```bash
python -m symfold.v4.da_se_dit
```

输出：

```text
in_channels = 97
logit=(2, 1, 32, 32) density=(2, 1) direct=(2, 1, 32, 32)
params=963,749
```

已验证 `symfold/config/v4_bprna.json` 能构建 `PriFoldSymFlow_v4 / DASEDiT_MARS_v4` 模型对象。

---

## 11. 预期观察指标

训练 v4 后，不要只看 aggregate F1，必须比较：

1. `density < 0.1` / `0.1-0.18` 的 F1 与 pred/gt；
2. `L>=160` 的 F1；
3. `F1=0` case 数量；
4. `pred_pairs / gt_pairs` 分桶；
5. `projection_mode=score/hybrid/sample` 消融；
6. `use_density_budget` 消融；
7. `num_samples_per_input=1/3/5` 消融。

v4 是否成功，第一判断标准不是整体 F1 小幅上涨，而是：

> low-density overprediction 是否明显下降，F1=0 case 是否明显减少，长 RNA 是否不再系统性错位。
