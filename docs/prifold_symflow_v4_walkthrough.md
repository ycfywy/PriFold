# PriFold-SymFlow v4 代码与架构 Walkthrough

更新时间：2026-06-03 12:54

本文从“一条 RNA 进来之后发生什么”出发，梳理 `symfold/v4` 的代码路径、关键模型模块、训练/推理逻辑和 v4 相比 v3 的核心特点。

> 范围：本文只解释 v4 代码与架构，不记录训练结果。v4 当前代码已完成语法、JSON、shape、loss/backward、小步 overfit、projection sanity 检查，但尚未启动正式训练。

---

## 1. v4 一句话定位

`PriFold-SymFlow v4` 是一个 **MARS-only 的 RNA contact map 生成式模型**：

```text
RNA sequence
  → MARS-LX hidden/attention + PriFold pos_bias/seq_oh
  → DA-SE-DiT-v4 条件主干
  → Bernoulli Discrete Flow Matching 训练
  → score-first / hybrid / sample projection 解码
  → RNA secondary structure contact map
```

相比 v3，v4 的重点不是单纯加大模型，而是解决 v3 bad-case 中暴露的三个问题：

1. `pos_bias` / MARS attention 只作为输入 channel，深层利用不足；
2. sampling 生成的 `x_t` 候选边不稳定，正确高分边没采到就无法被 projection 选中；
3. 低 density RNA 过预测严重，pair count 校准不足。

---

## 2. 相关文件地图

```text
symfold/
├── data.py                         # 数据加载、tokenize、pos_bias、seq_oh、padding、bucket sampler
├── metrics.py                      # contact P/R/F1/MCC
├── train_v4.py                     # v4 训练入口，复用 v3 loop + v4 build/evaluate
├── eval_v4.py                      # v4 独立评估入口，支持 v4 projection/sampling 参数
├── analyze_cases.py                # 已支持 v4 bad-case 分析
├── run_train.sh                    # 已支持 version=v4 自动选 train_v4.py
├── config/
│   ├── v4_bprna.json               # v4 bpRNA 配置
│   └── v4_rnastralign.json         # v4 RNAStrAlign 配置
└── v4/
    ├── __init__.py
    ├── model.py                    # PriFoldSymFlow_v4：MARS extract、train forward、sample
    ├── da_se_dit.py                # DASEDiT_MARS_v4 主干
    └── discrete_flow.py            # v5 loss、projection、density budget

prifold/
└── llama2_with_attn.py              # MARS wrapper：返回 hidden_layers + attention maps
```

---

## 3. 一条 RNA 进来后的完整流程

下面以训练/评估中的一条 RNA 为例说明。

### Step 0：原始样本

数据来自 PriFold 数据集：

- bpRNA：`data/bprna/bpRNA.csv` + 对应 `.npy` contact map；
- RNAStrAlign：`data/RNAStrAlign/rnastralign.csv`；
- ArchiveII：`data/archiveII/archiveII.csv`。

每条样本包含：

```text
name / dataset / RNA sequence / contact map
```

contact map 是 `L×L` 二值矩阵，表示第 `i` 个碱基和第 `j` 个碱基是否配对。

---

### Step 1：`symfold/data.py` 预处理

`build_loader(stage, config, tokenizer, shuffle=...)` 会构建 dataset 和 dataloader。

每条 RNA 会被处理成：

| 字段 | shape | 说明 |
|---|---|---|
| `input_ids` | `(L+2,)` | MARS tokenizer 后的 token，包含 `<cls>/<eos>` |
| `attention_mask` | `(L+2,)` | MARS attention mask |
| `seq_oh` | `(S,4)` | A/T/G/C one-hot；`U→T` 后编码 |
| `contact` | `(1,S,S)` | padding 后 GT contact map |
| `contact_mask` | `(1,S,S)` | 有效区域 mask，padding 为 0 |
| `pos_bias` | `(S,S)` | PriFold 原生碱基互补先验 |
| `length` | scalar | 原始长度 `L` |
| `names/datasets` | string | case analysis 用 |

其中：

```text
S = ceil(L / patch_size) * patch_size
patch_size = 4
```

`pos_bias` 规则：

```text
A-T = 3
G-C = 6
G-T = 1
其他 = 0
```

v4 保留 `seq_oh`，因为当前路线不接 RNA-FM / UFold，显式碱基身份对 MARS-only 条件有帮助。

---

### Step 2：进入 `PriFoldSymFlow_v4.forward()`

训练时入口是：

```python
symfold/v4/model.py::PriFoldSymFlow_v4.forward(batch)
```

核心流程：

```text
batch
  ├─ contact → symmetrize + mask → x_1
  ├─ input_ids / attention_mask → MARS extraction
  ├─ t ~ Uniform(0,1)
  ├─ x_t ~ q_t(x_t | x_1)
  └─ backbone(x_t, t, conditions) → logit / density_pred / direct_logit
```

训练中的 noising：

```text
x_1 = GT contact map
 t  ~ Uniform(0,1)
x_t ~ Bernoulli(t * x_1 + (1-t) * rho_0)
rho_0 = 0.005
```

这样模型学的是：给定 noisy contact `x_t`、时间 `t` 和 RNA 条件，恢复最终 clean contact `x_1` 的概率。

---

### Step 3：MARS-LX 条件提取

代码：

```python
PriFoldSymFlow_v4._extract_mars(input_ids, attention_mask, set_len=S)
```

调用：

```python
prifold/llama2_with_attn.py::mars_forward_with_attn(...)
```

返回三类 MARS 条件：

| 输出 | shape | 用途 |
|---|---|---|
| `mars_hidden` | `(B,S,1056)` | final hidden，兼容/兜底 |
| `mars_hidden_layers` | 4 × `(B,S,1056)` | layer `[3,6,9,12]`，多层 hidden fusion |
| `mars_attn` | `(B,6,12,S,S)` | last 6 layers × 12 heads attention maps |

注意点：

1. frozen MARS 每次提取前强制 `eval()`，避免 dropout；
2. 去掉 `<cls>/<eos>`，只保留 base tokens；
3. 按 `S` padding/truncate，与 contact map 对齐。

---

### Step 4：进入 v4 backbone

代码：

```python
symfold/v4/da_se_dit.py::DASEDiT_MARS_v4.forward(...)
```

输入：

```text
x_t                  当前 noisy contact state, (B,1,S,S)
t                    flow time, (B,)
mars_hidden_layers   MARS layer [3,6,9,12]
mars_attn            MARS last 6-layer attention stack
pos_bias             PriFold base-pair prior
seq_oh               one-hot bases
contact_masks        valid region mask
density_hint         optional density condition
```

输出：

| 输出 | shape | 说明 |
|---|---|---|
| `logit` | `(B,1,S,S)` | flow logit，用于 DFM loss / CTMC rates |
| `density_pred` | `(B,1)` | pair-per-base density 预测 |
| `direct_logit` | `(B,1,S,S)` | direct contact score head |

---

## 4. v4 backbone 内部结构

### 4.1 MARS multi-layer hidden fusion

代码：

```python
MultiLayerMarsFusion
```

流程：

```text
MARS hidden layers [3,6,9,12]
  ├─ learnable softmax layer weights 加权平均
  ├─ 每层单独 MLP 投影
  ├─ concat 后再 fuse MLP
  └─ residual avg_proj
  → mars_fused: (B,S,64)
  → mars_emb_proj: (B,S,32)
```

随后做 outer concat：

```text
mars_emb_1d: (B,S,32)
  → outer_concat(i,j)
  → mars_2d: (B,64,S,S)
```

目的：把 MARS 的 1D token representation 转成 pair-level feature。

---

### 4.2 MARS attention projection

代码：

```python
MarsAttentionProj
```

流程：

```text
mars_attn: (B,6,12,S,S)
  → reshape: (B,72,S,S)
  → symmetrize
  → APC correction
  → 1×1 Conv projection
  → mars_attn_2d: (B,16,S,S)
```

MARS attention map 本身就是天然的 `(i,j)` pair feature，是 v4 条件系统的核心。

---

### 4.3 输入 channel 账

v4 与 v3 一样，输入 patch embed 的 channel 是 97：

| 来源 | 通道数 | 说明 |
|---|---:|---|
| `x_t` embedding | 8 | 当前 flow state 0/1 embedding |
| MARS hidden outer concat | 64 | 32-dim hidden pair 的 i/j concat |
| MARS attention projection | 16 | MARS attention pair feature |
| `pos_bias` | 1 | PriFold 碱基配对先验 |
| `seq_oh` outer concat | 8 | A/T/G/C identity pair feature |
| **总计** | **97** | `(B,97,S,S)` |

```text
features: (B,97,S,S)
  → PatchEmbed2D(kernel=4,stride=4)
  → tokens: (B,S/4,S/4,256)
```

---

## 5. v4 的三个关键模型特点

### 特点 1：`pos_bias + MARS attention` 每层作为 attention bias

v3 只把 `pos_bias` / MARS attention 放进输入 channel。

v4 新增：

```python
CondAttentionBias
```

流程：

```text
cond_pair = concat(mars_attn_2d, pos_bias)
cond_patch = AvgPool2d(patch_size=4)(cond_pair)
attn_bias = Conv1x1(cond_patch) → (B,num_heads,S/4,S/4)
```

然后每个 `DASEDiTBlockV4` 的 row/col axial attention 都接收这个 bias。

直观理解：

- `pos_bias` 告诉模型哪些 base pair 从碱基互补角度更合理；
- MARS attention 告诉模型语言模型认为哪些位置有联系；
- v4 不再只让模型在第一层看一次，而是每层 attention 都能参考这些 pair prior。

实现位置：

```text
symfold/v4/da_se_dit.py
├── CondAttentionBias
├── DilatedAxialAttentionV4
└── DASEDiTBlockV4
```

---

### 特点 2：ControlNet-style 条件刷新

v4 新增：

```python
ControlInjectMLP
```

每隔 `control_every=2` 层：

```text
tokens = tokens + control_inject(cond_patch)
```

`control_inject` 最后一层 zero-init，初始不破坏原模型路径，训练时逐步学会把 MARS/pos_bias 条件重新注入深层 token。

作用：避免 pair condition 在 patch embed 后被深层网络逐渐洗掉。

---

### 特点 3：direct contact score head

v3 只有 flow head：

```text
tokens → flow_logit
```

v4 有两个 head：

```text
final_tokens
  ├─ flow head   → logit        # 用于 flow loss 和 CTMC rates
  └─ direct head → direct_logit # 用于 direct BCE 和 score-first projection
```

训练时：

```text
loss = flow_loss
     + direct_weight * direct_BCE
     + pair_count_weight * pair_count_loss
     + density / stack / nc losses
```

推理时：

```text
score = (1 - direct_score_weight) * sigmoid(flow_logit)
      + direct_score_weight * sigmoid(direct_logit)
```

默认：

```json
"direct_score_weight": 0.5
```

这解决 v3 的一个核心问题：如果最终 sampled `x_t` 没包含正确边，v3 projection 无法恢复；v4 默认 score-first projection 可以直接从所有合法边中按 score 解码。

---

## 6. v4 的训练 loss

代码：

```python
symfold/v4/discrete_flow.py::BernoulliFlowLoss_v5
```

组成：

| 项 | 默认权重/设置 | 作用 |
|---|---:|---|
| adaptive BCE | `pos_weight_base=199`, `pos_weight_min=10` | flow logit 监督，低 density 样本降低正类权重 |
| focal | `gamma=2.0` | 更强调 hard negative，抑制 FP |
| stacking loss | `0.05` | 鼓励局部 stacking 连续性 |
| nc loss | `0.02` | 当前是 row-sum≤1 软约束 |
| density SmoothL1 | `0.2` | 预测 pair-per-base density |
| direct BCE | `direct_weight=0.3` | 直接监督 `direct_logit` |
| pair count SmoothL1 | `pair_count_weight=0.05` | 约束 `sigmoid(direct_logit).sum()/2` 接近 GT pair 数 |

### 为什么加 pair-count loss

v3 bad-case 显示：低 density 样本严重过预测。例如 `density<0.1` 时，预测 pair 数约为 GT 的 4.7 倍。

v4 对 direct head 加 pair-count 校准：

```text
pred_pairs = sigmoid(direct_logit).sum(valid_edges) / 2
pred_density = pred_pairs / L_eff
gt_density = gt_pairs / L_eff
pair_count_loss = SmoothL1(pred_density, gt_density)
```

目标是让模型不仅知道“哪条边可能对”，还知道“总共应该选多少边”。

---

## 7. v4 推理 / sampling / projection

入口：

```python
PriFoldSymFlow_v4.sample(...)
```

### 7.1 默认 sampling 配置

`v4_bprna.json` / `v4_rnastralign.json`：

```json
"sampling": {
  "num_steps": 20,
  "num_samples_per_input": 1,
  "density_guided": false,
  "projection_mode": "score",
  "use_density_budget": false,
  "budget_scale": 1.0,
  "candidate_weight": 0.35,
  "direct_score_weight": 0.5,
  "score_threshold": 0.5,
  "default_budget_fraction": 0.35
}
```

### 7.2 为什么默认关闭 density-guided

v3 实测：

```text
density_guided=True  → bprna-test F1≈0.41
density_guided=False → bprna-test F1≈0.45
```

所以 v4 默认不再用 density 乘 `rate_01`，避免在采样阶段过度压掉正确候选边。

---

### 7.3 score-first projection

v4 默认：

```text
projection_mode = score
```

解码逻辑：

```text
score over all legal edges
  → score_threshold 过滤
  → default_budget_fraction 控制最大 pair 数
  → greedy max matching
  → final contact map
```

默认：

```text
score_threshold = 0.5
default_budget_fraction = 0.35
```

如果开启 `use_density_budget=true`，则：

```text
max_pairs = round(density_pred * L_eff * budget_scale)
```

否则使用：

```text
max_pairs = round(0.35 * L_eff)
```

这样避免 score-only projection 因为所有 sigmoid score 都 > 0 而无脑选满 `L/2`。

---

### 7.4 三种 projection mode

| mode | 候选来源 | 说明 |
|---|---|---|
| `score` | 全部合法边 | 默认；不依赖 sampled `x_t` 是否包含正确边 |
| `hybrid` | 全部合法边 + sampled candidate bonus | `score + candidate_weight*x_t` |
| `sample` | 只从 sampled `x_t==1` 中选 | v3 兼容路径 |

代码：

```python
project_score_to_valid_contact_map(...)
project_hybrid_contact_map(...)
project_to_valid_contact_map(...)
```

---

### 7.5 多样本投票

v4 支持：

```json
"num_samples_per_input": 3 或 5
```

逻辑：

```text
多条 trajectory
  → 平均 score / candidate
  → 最终统一 projection 一次
```

这比 v3 “先 projection 再平均阈值”更稳，因为最终仍保证一碱基最多一个 pairing。

---

## 8. 训练入口 walkthrough

启动命令：

```bash
bash symfold/run_train.sh symfold/config/v4_bprna.json
```

流程：

```text
run_train.sh
  ├─ 读取 config.model.version == v4
  ├─ 选择 ENTRY=symfold/train_v4.py
  ├─ 后台启动训练进程
  └─ 同时启动 gpu_monitor daemon
```

`train_v4.py` 做两件事：

1. 复用 `train_v3.py` 的训练 loop、日志、checkpoint、曲线绘制；
2. 替换：
   - `build_model()` → 构建 `PriFoldSymFlow_v4`
   - `evaluate()` → 使用 v4 sampling/projection 参数

因此 v4 训练产物仍然与 v3 一致：

```text
symfold/outputs/v4_bprna/
├── history.json
├── test_eval_history.json
├── training_curves.png
├── gpu_stats.jsonl
└── model/
    ├── last.pt
    ├── best.pt
    └── epoch_*.pt
```

---

## 9. 评估与 bad-case 分析

### 9.1 独立评估

```bash
python symfold/eval_v4.py \
  --ckpt symfold/outputs/v4_bprna/model/best.pt \
  --out_json symfold/outputs/v4_bprna/eval_best.json
```

常用消融：

```bash
# v3 candidate-only projection 路径
python symfold/eval_v4.py --ckpt <ckpt> --projection_mode sample

# hybrid projection
python symfold/eval_v4.py --ckpt <ckpt> --projection_mode hybrid --candidate_weight 0.35

# density budget
python symfold/eval_v4.py --ckpt <ckpt> --use_density_budget 1 --budget_scale 1.0

# 调 projection 阈值 / 默认 budget
python symfold/eval_v4.py --ckpt <ckpt> --score_threshold 0.55 --default_budget_fraction 0.30

# 多样本
python symfold/eval_v4.py --ckpt <ckpt> --num_samples_per_input 5
```

### 9.2 bad-case 分析

```bash
python symfold/analyze_cases.py \
  --ckpt symfold/outputs/v4_bprna/model/best.pt \
  --test_sets bprna-test \
  --out_dir symfold/outputs/v4_bprna/case_analysis
```

`analyze_cases.py` 已支持 v4：

- 自动识别 `model.version == "v4"`；
- 使用 `train_v4.build_model()`；
- 传入 v4 sampling 参数：
  - `projection_mode`
  - `use_density_budget`
  - `budget_scale`
  - `candidate_weight`
  - `direct_score_weight`
  - `score_threshold`
  - `default_budget_fraction`

---

## 10. v4 当前 sanity 验证

已运行小尺寸 sanity test，不启动训练。

输出：

```text
logit (2, 1, 32, 32) sym? True
direct (2, 1, 32, 32) sym? True
density (2, 1) [...]
losses {'bce': 6.3915, 'stack': -0.0096, 'nc': 0.2482,
        'density': 0.0317, 'direct': 0.7947, 'pair_count': 0.284}
total 7.7405
params_with_grad 91 / 241
overfit losses [7.7405, 3.9765, 3.5792, 3.3703, 3.2219]
proj_pairs score/hybrid/sample: 22 22 14
budget [3, 10]
none vs zero hint identical? True
```

解释：

- logit / direct logit shape 正确且对称；
- loss 能反向，5 步小 overfit 明显下降；
- `params_with_grad 91/241` 是 AdaLN-Zero 初始 gate=0 的预期现象；
- score/hybrid/sample 三种 projection 都可运行；
- `density_hint=None` 和 zero hint 路径一致。

---

## 11. v4 相比 v3 的核心差异表

| 模块 | v3 | v4 |
|---|---|---|
| `pos_bias` 用法 | 输入 channel | 输入 channel + 每层 attention bias |
| MARS attention 用法 | 输入 channel | 输入 channel + 每层 attention bias |
| 条件刷新 | 无 | ControlInjectMLP 每 2 层刷新 |
| 输出 head | flow head | flow head + direct score head |
| projection 默认 | sampled candidate | score-first |
| density-guided | 默认 true | 默认 false |
| pair count 约束 | density head 间接约束 | direct head pair-count loss |
| projection budget | 无 / density-guided rate damping | score threshold + default budget / optional density budget |
| train eval | v3 sampling 参数 | v4 完整 sampling/projection 参数 |

---

## 12. 训练后必须看的指标

v4 的目标不是只让 aggregate F1 涨一点，而是修复 v3 的系统性 bad cases。

训练后必须比较：

1. **low-density bins**：
   - `density < 0.10`
   - `0.10 ≤ density < 0.18`
   - 看 `F1 / P / R / pred_pairs/gt_pairs`
2. **length bins**：
   - `<80`
   - `80-159`
   - `160-239`
   - `240+`
3. **F1=0 case 数量**；
4. **pred/gt ratio bins**；
5. projection 消融：
   - `score`
   - `hybrid`
   - `sample`
6. budget 消融：
   - `default_budget_fraction=0.30/0.35/0.40`
   - `use_density_budget=0/1`
7. `num_samples_per_input=1/3/5`。

判断 v4 是否成功的优先标准：

```text
低 density 过预测下降
F1=0 case 减少
160+ 长度样本不再系统性错位
pred/gt ratio 更接近 1
```

---

## 13. 当前建议的实验顺序

正式训练前建议先保持默认配置：

```json
"projection_mode": "score",
"density_guided": false,
"use_density_budget": false,
"score_threshold": 0.5,
"default_budget_fraction": 0.35
```

训练后第一轮评估：

```bash
python symfold/eval_v4.py --ckpt <best.pt>
python symfold/analyze_cases.py --ckpt <best.pt> --test_sets bprna-test --out_dir <case_dir>
```

然后做最小消融：

```bash
# projection 消融
--projection_mode score
--projection_mode hybrid
--projection_mode sample

# budget 消融
--default_budget_fraction 0.30
--default_budget_fraction 0.35
--default_budget_fraction 0.40
--use_density_budget 1

# 多样本
--num_samples_per_input 3
--num_samples_per_input 5
```

---

## 14. 小结

v4 可以理解为：

> 在 v3 的 MARS-only DA-SE-DiT 基础上，把 `pos_bias/MARS attention` 从“一次性输入特征”升级为“每层 attention bias + 深层条件刷新”，同时新增 direct score head 和 pair-count 校准，让解码不再被 sampled `x_t` 候选池绑死。

它重点针对 v3 的三个失败模式：

1. 低 density 过预测；
2. 长 RNA 采样错位；
3. projection 候选边受 `x_t` 限制。

因此 v4 训练后的关键观察，不只是整体 F1，而是 bad-case 分布是否真正被修正。