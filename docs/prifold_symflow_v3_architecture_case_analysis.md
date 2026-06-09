# PriFold-SymFlow v3 架构说明、bad-case 诊断与改进思路

更新时间：2026-06-03 11:13

本文记录 `symfold/v3` 当前版本的架构、`v3_bprna` 已训练模型在 `bprna-test` 上的逐样本失败模式，以及下一版改进方向。

相关产物：

- checkpoint：`symfold/outputs/v3_bprna/model/best.pt`
- 训练历史：`symfold/outputs/v3_bprna/history.json`
- 周期性测试：`symfold/outputs/v3_bprna/test_eval_history.json`
- 本次 bad-case 分析：
  - `symfold/outputs/v3_bprna/case_analysis/bprna-test_cases.csv`
  - `symfold/outputs/v3_bprna/case_analysis/bprna-test_worst_100.json`
  - `symfold/outputs/v3_bprna/case_analysis/summary.json`
- 对照实验：关闭 density guidance 后的分析结果：
  - `symfold/outputs/v3_bprna/case_analysis_no_density/`

---

## 1. v3 版本定位

`PriFold-SymFlow v3` 是一个 **MARS-only 的 SymFold-style 生成式 RNA contact map 模型**。

一句话：

> 用 PriFold 数据与 MARS-LX 作为条件，把 RNA 二级结构 contact map 当成 Bernoulli 离散状态，通过 Discrete Flow Matching 训练，再用 CTMC τ-leap sampling 生成 contact map，最后 greedy projection 到合法配对集合。

与主线 PriFold 的区别：

| 维度 | 主线 PriFold | SymFlow v3 |
|---|---|---|
| 任务形式 | 判别式 contact logits | 生成式 Bernoulli flow |
| 主干 | RNAformer/RiboFormer | DA-SE-DiT-MARS |
| LM 条件 | MARS final hidden → 2D pair | MARS 多层 hidden + last 6 layers attention map |
| 先验 | `pos_bias` 作为 attention bias | `pos_bias` 作为输入 channel |
| 推理 | sigmoid + 阈值 | CTMC sampling + projection |
| 当前 bpRNA-test F1 | 0.7700 | 0.405~0.453（取决于 sampling 设置） |

当前 v3 仍坚持：**不接 RNA-FM，不接 UFold**。所有条件来自 PriFold 数据、MARS-LX、`pos_bias` 和显式 `seq_oh`。

---

## 2. 代码结构

```text
symfold/
├── data.py                  # PriFold CSV/NPY 数据集、padding、pos_bias、seq_oh、长度 bucket sampler
├── metrics.py               # contact P/R/F1/MCC，按上三角且 |i-j|>=3 统计
├── train_v3.py              # v3 训练入口，周期性 val/test eval，曲线绘制
├── eval_v3.py               # v3 独立评估入口
├── analyze_cases.py         # per-RNA bad-case 分析
├── v3/
│   ├── model.py             # PriFoldSymFlow_v2 API，封装 MARS extract / forward / sample
│   ├── da_se_dit.py         # DA-SE-DiT-MARS 主干
│   └── discrete_flow.py     # Bernoulli flow loss、CTMC rates、projection
└── config/
    ├── v3_bprna.json        # bpRNA 模型配置
    └── v3_rnastralign.json  # RNAStrAlign 模型配置

prifold/
└── llama2_with_attn.py      # MARS wrapper：暴露多层 hidden 与最后 N 层 attention map
```

---

## 3. v3 架构拆解

### 3.1 数据输入

每条样本由 `symfold/data.py` 产出：

- RNA sequence：统一 `U→T`；
- `input_ids / attention_mask`：给 MARS-LX；
- `seq_oh`：A/T/G/C one-hot；
- `contact`：`(1, S, S)`，真实二级结构 contact map；
- `contact_mask`：padding 区域 mask；
- `pos_bias`：PriFold 原生碱基互补先验，A-T=3、G-C=6、G-T=1；
- `length` / `name` / `dataset`：评估与 case 分析使用。

其中 `S` 是原始长度按 `patch_size=4` 向上补齐后的长度；训练/评估过滤 `len < 490`。

### 3.2 MARS-LX 条件提取

`PriFoldSymFlow_v2._extract_mars()` 调用 `prifold/llama2_with_attn.py`，返回：

| 输出 | shape | 用途 |
|---|---|---|
| `hidden` | `(B, S, 1056)` | final hidden，兼容/全局条件 |
| `hidden_layers` | 4 × `(B, S, 1056)` | layer `[3,6,9,12]` 多层融合 |
| `attn_stack` | `(B, 6, 12, S, S)` | last 6 layers × 12 heads attention map |

关键实现点：

- frozen MARS 每次 extract 前强制 `eval()`，避免 `model.train()` 递归打开 dropout；
- 原 MARS 的 `<cls>/<eos>` 会被切掉，只保留 base tokens；
- 不足 `S` 的部分补零，保证与 contact map 对齐。

### 3.3 输入 feature 拼接

`DASEDiT_MARS_v2._build_features()` 构造 97 通道输入：

| 来源 | 通道数 | 说明 |
|---|---:|---|
| `x_t` embedding | 8 | 当前 flow 状态 0/1 的 embedding |
| MARS multi-layer hidden outer concat | 64 | hidden 融合后投影到 32，再做 `i/j` outer concat |
| MARS attention projection | 16 | 6×12 attention → 对称化 + APC + 1×1 Conv |
| `pos_bias` | 1 | 作为普通输入 channel |
| `seq_oh` outer concat | 8 | A/T/G/C 的 `i/j` one-hot outer concat |
| **合计** | **97** | `(B,97,S,S)` |

端到端特征流：

```text
x_t + MARS hidden pair + MARS attention pair + pos_bias + seq_oh
  → concat 97ch
  → 对称化
  → PatchEmbed2D(kernel=4,stride=4)
  → tokens: (B, S/4, S/4, 256)
```

### 3.4 DA-SE-DiT-MARS 主干

主干在 `symfold/v3/da_se_dit.py`：

- hidden dim：256；
- heads：4；
- layers：9；
- dilation pattern：`[1,1,1,2,2,2,4,4,4]`；
- triangle update：从 layer 6 开始启用；
- FFN：SwiGLU；
- 条件：`time + MARS global + density_hint` 经 `cond_fuse` 后喂给 AdaLN-Zero。

每个 block：

```text
tokens
  ├─ AdaLN-Zero(time, mars_global, density)
  ├─ Dilated Axial Attention(row + col, shared QKV, RoPE, QK-Norm)
  ├─ Triangle Multiplicative Update(layer >= 6)
  └─ SwiGLU FFN
```

注意：当前 v3 虽然引入了 MARS attention map，但它仍然只是 **输入 channel**，并没有作为每层 row/col attention 的 additive bias 注入。

### 3.5 输出头

```text
tokens
  → final AdaLN-Zero
  → UnPatchify2D
  → OutputRefineConv(3-layer residual conv)
  → symmetric logit
  → mask short-range |i-j|<3 and padding
```

输出：

- `logit`: `(B,1,S,S)`，供 Bernoulli flow loss / sampling 使用；
- `density_pred`: `(B,1)`，预测 pair-per-base density。

### 3.6 训练 loss

`BernoulliFlowLoss_v4` 包含：

| loss | 当前配置 | 作用 |
|---|---:|---|
| adaptive BCE | `pos_weight_base=199`, `pos_weight_min=20` | 根据 GT density 调整正样本权重 |
| focal | `gamma=1.5` | 强调 hard examples |
| stacking | `0.05` | 鼓励相邻 stacking 连续性 |
| nc | `0.02` | 当前实际是 row-sum ≤ 1 软约束，不是真正 non-crossing |
| density MSE | `0.2` | 监督 density head |

训练 forward：

```text
x_1 = GT contact map
 t ~ Uniform(0,1)
x_t ~ Bernoulli(t*x_1 + (1-t)*rho_0)
logit, density_pred = backbone(x_t, t, conditions)
loss = BCE + focal + stacking + row-count + density MSE
```

`v3_bprna.json` 中 `density_hint_dropout=1.0`，即训练时不把 GT density 注入 backbone，只把 density 当辅助监督，避免推理时 density OOD。

### 3.7 推理 sampling

`PriFoldSymFlow_v2.sample()`：

```text
1. x_init ~ Bernoulli(rho_0)
2. 如果 density_guided=True：
   用 t=0.5 前向一次预测 density_pred
3. cosine τ-leap schedule，默认 20 steps
4. 每一步：
   logit → p_x1 → CTMC rates
   如果 density_guided=True：rate_01 *= clamp(2*density_pred, max=1)
   根据 rate flip 0↔1
5. 最后 project_to_valid_contact_map(x_t, p_x1_last, mask)
```

非常关键的一点：

> 当前 projection 只在最终 `x_t==1` 的候选边里按 `score=p_x1_last` 做 greedy matching。也就是说，如果正确边没有被 sampling 采到，即便 `p_x1_last` 分数很高，最终 projection 也无法选它。

这是后面 bad-case 的核心根因之一。

---

## 4. v3_bprna 当前训练结果

`v3_bprna` 已跑满 120 epoch。

周期性 `bprna-test` 结果：

| epoch | F1 | P | R | pred_pairs | gt_pairs |
|---:|---:|---:|---:|---:|---:|
| 9 | 0.2547 | 0.2143 | 0.3355 | 54.23 | 31.09 |
| 49 | 0.3138 | 0.2766 | 0.3830 | 49.17 | 31.09 |
| 79 | 0.3761 | 0.3389 | 0.4430 | 46.25 | 31.09 |
| 89 | 0.3938 | 0.3558 | 0.4638 | 45.68 | 31.09 |
| 109 | 0.4061 | 0.3692 | 0.4718 | 44.58 | 31.09 |
| 119 | 0.4053 | 0.3679 | 0.4717 | 44.78 | 31.09 |

观察：

1. 训练确实在稳步改善，但 90 epoch 后基本平台；
2. 即便后期，平均预测 pair 数仍明显高于 GT：44.8 vs 31.1；
3. 主要差距不是 recall 不够，而是 **false positive 太多 + 候选边错位**。

本次重新运行 `analyze_cases.py` 后，默认 `density_guided=True` 的逐样本汇总为：

| setting | N | F1 | P | R | MCC | gt_pairs | pred_pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| `density_guided=True` | 1303 | 0.4107 | 0.3732 | 0.4807 | 0.4155 | 31.09 | 44.95 |
| `density_guided=False` | 1303 | **0.4530** | **0.3853** | **0.5960** | **0.4674** | 31.09 | 52.10 |

这说明：**当前 v3 的 density-guided sampling 不是增益项，反而拖累了 bpRNA-test。**

---

## 5. bad-case 到底是怎么回事

### 5.1 F1 分布：有一批样本完全错位

默认 `density_guided=True` 时，逐样本 F1 分位数：

| quantile | F1 |
|---:|---:|
| min | 0.000 |
| 5% | 0.000 |
| 10% | 0.036 |
| 25% | 0.179 |
| 50% | 0.385 |
| 75% | 0.600 |
| 90% | 0.833 |
| 95% | 0.952 |
| max | 1.000 |

结论：模型不是所有样本都差；短小/常规结构可以做得很好。但 bottom 25% 非常差，且有 **109/1303 = 8.4%** 的样本 F1=0。

F1=0 的平均特征：

| group | N | F1 | P | R | length | density | gt_pairs | pred_pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F1=0 | 109 | 0.000 | 0.000 | 0.000 | 115.0 | 0.136 | 16.3 | 34.7 |

这类 case 不是没预测边，而是 **预测了一堆边，但没有一个和 GT 对上**。

### 5.2 最差样本示例

`bprna-test_cases.csv` 已按 F1 升序排列。前 20 个 worst cases：

| name | length | density | gt_pairs | pred_pairs | TP | FP | FN | F1 | bin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `bpRNA_RFAM_16638` | 332 | 0.289 | 96 | 119 | 0 | 119 | 96 | 0.000 | 320-399 |
| `bpRNA_CRW_54597` | 328 | 0.238 | 78 | 125 | 0 | 125 | 78 | 0.000 | 320-399 |
| `bpRNA_RFAM_25043` | 313 | 0.048 | 15 | 119 | 0 | 119 | 15 | 0.000 | 240-319 |
| `bpRNA_RFAM_35736` | 239 | 0.071 | 17 | 90 | 0 | 90 | 17 | 0.000 | 160-239 |
| `bpRNA_RFAM_11250` | 229 | 0.328 | 75 | 89 | 0 | 89 | 75 | 0.000 | 160-239 |
| `bpRNA_RFAM_7117` | 222 | 0.144 | 32 | 81 | 0 | 81 | 32 | 0.000 | 160-239 |
| `bpRNA_RFAM_41284` | 204 | 0.211 | 43 | 67 | 0 | 67 | 43 | 0.000 | 160-239 |
| `bpRNA_RFAM_16197` | 202 | 0.203 | 41 | 73 | 0 | 73 | 41 | 0.000 | 160-239 |
| `bpRNA_RFAM_35726` | 199 | 0.085 | 17 | 64 | 0 | 64 | 17 | 0.000 | 160-239 |
| `bpRNA_RFAM_33294` | 198 | 0.253 | 50 | 70 | 0 | 70 | 50 | 0.000 | 160-239 |

直接现象：

1. worst case 大量来自 `RFAM`；
2. 长度集中在 160+，有些达到 300+；
3. 很多是低 density 或中低 density；
4. 模型不是少预测，而是 **错位过预测**：如 `bpRNA_RFAM_25043` 只有 15 条 GT pairs，却预测了 119 条，0 个命中。

### 5.3 按长度分桶：短序列能做，长序列明显崩

默认 `density_guided=True`：

| length bin | N | F1 | P | R | gt_pairs | pred_pairs |
|---|---:|---:|---:|---:|---:|---:|
| `<80` | 316 | **0.598** | 0.575 | 0.647 | 16.6 | 19.5 |
| `80-159` | 682 | 0.385 | 0.340 | 0.468 | 24.2 | 35.2 |
| `160-239` | 174 | 0.251 | 0.212 | 0.323 | 41.9 | 65.7 |
| `240-319` | 38 | 0.238 | 0.208 | 0.289 | 70.8 | 103.6 |
| `320-399` | 66 | 0.352 | 0.317 | 0.410 | 87.6 | 129.8 |
| `400+` | 27 | 0.284 | 0.252 | 0.330 | 112.3 | 165.6 |

结论：**长度是第一类强失败因子**。

短 RNA `L<80` 平均 F1 接近 0.60，说明网络学到了一部分局部 pairing pattern。但一旦进入 `160+`，F1 掉到 0.24~0.35。主要原因包括：

- patch 化后 `S/4 × S/4` token 网格降低了分辨率，长程细粒度配对更难恢复；
- MARS attention 只作为输入 channel，一次性进入 patch embed，后续每层 attention 没有显式 pair bias；
- projection 只从 sampled candidate 中选边，长序列 candidate 空间更大，错位概率更高；
- 当前 row-sum 约束只能保证每个 base 最多一个配对，无法约束真实 RNA 二级结构的嵌套拓扑。

### 5.4 按 density 分桶：低密度样本严重过预测

默认 `density_guided=True`：

| density bin | N | F1 | P | R | gt_pairs | pred_pairs | pred/gt |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0.00-0.10` | 105 | **0.090** | 0.058 | 0.250 | 6.2 | 29.0 | **4.71×** |
| `0.10-0.18` | 187 | 0.236 | 0.182 | 0.350 | 20.9 | 44.7 | **2.14×** |
| `0.18-0.25` | 411 | 0.375 | 0.322 | 0.459 | 29.3 | 45.6 | 1.55× |
| `0.25-0.32` | 515 | 0.546 | 0.520 | 0.580 | 38.8 | 47.2 | 1.21× |
| `0.32+` | 85 | 0.545 | 0.540 | 0.555 | 45.9 | 48.4 | 1.05× |

结论：**density 是第二类强失败因子**。

低 density case 的根因不是 recall 低，而是模型不会控制 pair 数：

- GT 只有 6 对左右时，模型平均预测 29 对；
- GT 20 对左右时，模型平均预测 45 对；
- 只有到 density ≥ 0.25 时，pred/gt 才接近合理范围。

这说明当前 adaptive BCE / focal / density head 没有把 “pair budget” 真正约束进最终结构。

### 5.5 pred/gt 比例：匹配 pair 数时模型其实还不错

按 `pred_pairs / gt_pairs` 分桶：

| pred/gt | N | F1 | P | R | length |
|---|---:|---:|---:|---:|---:|
| `0.75-1.1` | 267 | **0.681** | 0.685 | 0.679 | 101.4 |
| `1.1-1.5` | 517 | 0.439 | 0.395 | 0.496 | 140.8 |
| `1.5-2.0` | 293 | 0.303 | 0.243 | 0.405 | 153.1 |
| `2.0+` | 220 | **0.159** | 0.111 | 0.310 | 143.0 |

这张表非常关键：

> 当预测 pair 数接近 GT pair 数时，v3 的平均 F1 可以到 0.68；一旦 pred/gt 超过 2，F1 立刻掉到 0.16。

所以 v3 的首要问题不是 backbone 完全没学会 pairing，而是 **pair count / candidate selection / projection 校准失败**。

### 5.6 density-guided sampling 为什么反而更差

对比默认 `density_guided=True` 与关闭 `density_guided=False`：

| setting | F1 | P | R | pred_pairs |
|---|---:|---:|---:|---:|
| `density_guided=True` | 0.4107 | 0.3732 | 0.4807 | 44.95 |
| `density_guided=False` | **0.4530** | **0.3853** | **0.5960** | 52.10 |

关闭 density guidance 后，虽然预测 pair 更多，但 P/R/F1 都更好。这说明当前 density guidance 不只是“减少边”，而是在 sampling 过程中把正确候选边也抑制掉了。

进一步检查 density head 校准：

| GT density bin | N | GT density | pred density | bias |
|---|---:|---:|---:|---:|
| `0.00-0.10` | 105 | 0.052 | 0.105 | +0.052 |
| `0.10-0.18` | 186 | 0.149 | 0.198 | +0.050 |
| `0.18-0.25` | 412 | 0.215 | 0.232 | +0.017 |
| `0.25-0.32` | 516 | 0.283 | 0.264 | -0.019 |
| `0.32+` | 84 | 0.347 | 0.277 | -0.071 |

整体平均 density 预测还可以，但它把极端样本压向中间：

- 低 density 被高估；
- 高 density 被低估；
- 作为 `rate_01 *= 2*density_pred` 的硬 damping 时，容易在高 density / 难样本上过度抑制 0→1 flip；
- 最终导致正确边进不了 `x_t` candidate pool，projection 无法选中。

### 5.7 当前 projection 是第三个关键问题

`project_to_valid_contact_map(x, score, mask)` 当前逻辑：

```text
s = x * score * valid_mask
while can_select:
  select max s_ij
  remove row i and row j
```

也就是说，它只能在 `x_t==1` 的边里选。

这会带来两个后果：

1. 如果 sampling 采到了很多错边，projection 会把错边按 matching 约束整理得很“合法”，但仍然不对；
2. 如果某条正确边没有被 sampling 采到，即使 `score=p_x1_last` 对它很高，最后也不会被选。

这正好解释 F1=0 样本：模型预测了几十条合法边，但这些边与 GT 完全错位。

### 5.8 小结：v3 bad-case 的本质

v3 当前失败不是单一 bug，而是四个问题叠加：

1. **pair count calibration 不稳**：低 density 样本严重过预测；
2. **sampling candidate pool 不稳**：单条 trajectory + density damping 容易让正确边进不了候选集；
3. **projection 过度依赖 `x_t`**：高分正确边若未采样到，最终无法恢复；
4. **结构条件注入太浅**：MARS attention / pos_bias 只是第 0 层 channel，没有作为每层 attention bias 反复参与长程配对决策。

---

## 6. v3 改进思路

建议分三层推进：先做不需要重训的 sampling/projection 修复，再做小改重训，最后做 v4 架构升级。

### 6.1 第一优先级：不重训，先修 sampling/projection

#### 改动 A：默认关闭 density-guided sampling

基于本次实测：

- `density_guided=True`: F1=0.4107；
- `density_guided=False`: F1=0.4530。

建议：

1. `v3_bprna` 评估默认先设 `--density_guided 0`；
2. 在报告中把 0.4530 作为当前 v3 更合理的 eval baseline；
3. 保留 density head，但暂时不要让它控制 CTMC rates。

#### 改动 B：增加 score-only projection

新增一种 projection：不依赖最终 `x_t`，直接用 `p_x1_last` 做 greedy matching：

```text
s = p_x1_last * valid_mask
select top scoring non-conflicting pairs
```

对比三种投影：

| projection | 候选边 | 预期 |
|---|---|---|
| 当前 | `x_t==1` 且 score 高 | 易受 sampling 错位影响 |
| score-only | 所有合法边按 score | 检验 backbone logits 是否其实会排正确边 |
| hybrid | `alpha*x_t + beta*p_x1_last` | 兼顾 sampling 与 score |

这是最重要的诊断实验：

- 如果 score-only 大幅提升，说明 backbone 会打分，sampling/projection 是瓶颈；
- 如果 score-only 也不行，说明 pair feature / backbone 本身不够。

#### 改动 C：多样本 sampling 后必须重投影

当前 multi-sample 逻辑是：每条 sample 先 projection，再平均阈值，再 symmetrize。这个结果可能破坏“一碱基一配对”约束。

建议改成：

```text
for each trajectory:
  get p_x1_last or projected map
average scores/maps
run final greedy projection once
```

推荐对照：`num_samples_per_input = 1 / 3 / 5`。

#### 改动 D：按 predicted density 控制 projection budget，而不是控制 CTMC rate

不要用 density 去乘 `rate_01`，改成只控制最终最多选多少 pair：

```text
K = round(density_pred * L)
greedy projection 最多选 K 条 pair
```

这样 density 只影响 pair 数，不干扰候选边生成过程，风险更小。

### 6.2 第二优先级：v3.1 小改重训

#### 改动 E：重做 density / pair-count 校准

当前 density head 平均还可以，但极端样本被压向中间。建议：

1. density loss 从 MSE 改成 Huber 或分桶交叉熵；
2. 对低 density / 高 density 样本加权；
3. 训练时增加 `pred_pair_count` 辅助项：
   ```text
   sigmoid(logit).sum()/2 ≈ gt_pairs
   ```
4. 对低 density 样本降低 `pos_weight_min` 或使用更强的 negative focal；
5. 单独记录 density bin 的 val metrics，不只看整体 F1。

#### 改动 F：强化 low-density hard negatives

低 density 样本的主要错误是 FP 爆炸。建议：

- 对 density < 0.1 / 0.18 的样本做 oversampling；
- 对高置信 FP 加 hard-negative loss；
- 在 loss 中加入 row/column entropy 或 pair budget regularization；
- 训练曲线中单独画 `low_density_f1 / low_density_pred_gt_ratio`。

#### 改动 G：长度分桶不只用于 batch，还用于评估与采样调参

当前 `L<80` 与 `L>=160` 差距很大。建议：

- 按 length bin 分别调 `num_steps` / `num_samples`；
- 对长 RNA 增加 sample 数，短 RNA 保持单 sample；
- 训练时对 `160+` / `240+` 样本加权，避免模型只优化短序列。

### 6.3 第三优先级：v4 架构升级

如果第一、二层修复后仍低于预期，需要改架构。

#### 改动 H：把 `pos_bias` 和 MARS attention 变成每层 attention bias

当前问题：

```text
pos_bias / mars_attn → input channel → patch_embed 一次性吃掉
```

建议改成：

```text
pos_bias_patch + mars_attn_patch
  → CondAttentionBias
  → 每层 row/col attention additive bias
```

这样 PriFold 的生物先验和 MARS pair 先验不会只在第 0 层出现，而是每层参与 attention logits。

#### 改动 I：ControlNet-style 条件注入

每隔 2 层把 pooled condition 用 zero-init MLP 加回 tokens：

```text
tokens = tokens + zero_mlp(cond_patch)
```

目的：防止 patch embed 后条件信号在深层被洗掉。

#### 改动 J：增强结构约束

当前 `NonCrossingLoss` 实际是 row-sum≤1，不是真正 non-crossing。建议：

1. 加真实 crossing penalty：对 `(i,j)` 与 `(k,l)` 且 `i<k<j<l` 的 crossing pairs 施加惩罚；
2. projection 端也支持 non-crossing dynamic programming / Nussinov-style decoding；
3. 把 stacking loss 从“全局鼓励变密”改成条件式：只鼓励高置信 pair 的邻近 stacking，不鼓励所有位置变大。

#### 改动 K：保留 DiT，但补全 score head / direct decoder

当前生成式链路很依赖 sampling。建议让模型同时输出：

- flow logits：用于生成式训练；
- direct contact logits：用于 score-only projection 与辅助监督。

训练：

```text
loss = flow_loss + λ_direct * BCE(direct_logits, contact)
```

推理：

```text
sampling map + direct score → hybrid projection
```

这样可以保留 SymFlow 故事，同时把主线判别式的稳定性引入解码端。

---

## 7. 推荐下一步实验顺序

| 优先级 | 实验 | 是否重训 | 目标 |
|---:|---|---|---|
| P0 | `eval_v3.py --density_guided 0` 固化对照 | 否 | 确认 0.45 F1 baseline |
| P0 | score-only projection | 否 | 判断瓶颈在 logits 还是 sampling |
| P0 | multi-sample + final reproject | 否 | 降低单 trajectory 随机性 |
| P1 | projection budget = `round(density_pred*L)` | 否/小改 | 控制过预测而不破坏 candidate 生成 |
| P1 | density-bin / length-bin dashboard | 否 | 每次实验都定位失败群体 |
| P2 | density/pair-count loss 重训 v3.1 | 是 | 修 low-density overprediction |
| P2 | length/density reweight sampler | 是 | 修长序列与稀疏结构 |
| P3 | pos_bias/MARS attention as per-layer attention bias | 是 | 强化结构条件注入 |
| P3 | true non-crossing decoding/loss | 是 | 提升结构合法性 |

最推荐先做的一个最小闭环：

```text
1. 禁用 density_guided，得到稳定 baseline；
2. 实现 score-only projection；
3. 跑 bprna-test case_analysis；
4. 比较：
   - overall F1
   - low-density F1
   - 160+ length F1
   - pred/gt ratio
   - F1=0 case 数量
```

如果 score-only projection 能把 F1 从 0.45 拉到 0.55+，说明 v3 backbone 仍有价值，重点修 sampling/projection；如果拉不动，就进入 v4 架构改造。

---

## 8. 结论

v3 已经比 v1/v2 初版更稳定，说明 MARS attention + DA-SE-DiT 方向是有效的。但它当前“不行”的 case 主要集中在：

1. **低 density RNA**：预测 pair 数严重超标，FP 爆炸；
2. **中长 RNA**：长程配对错位，projection 后合法但不正确；
3. **F1=0 的 RFAM/CRW case**：不是没预测，而是预测边与 GT 完全无交集；
4. **density-guided sampling**：当前反而降低 F1，应先禁用或改成 projection budget；
5. **projection 候选依赖 `x_t`**：正确边没被采到就无法恢复。

因此下一版不要先盲目堆更大的模型，而应先解决：

> pair count 校准 + score-only/hybrid projection + density/length 分桶评估。

这是最小成本、最可能立刻提升 v3 的方向。