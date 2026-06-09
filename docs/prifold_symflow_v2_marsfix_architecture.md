# PriFold-SymFlow v2_marsfix 当前版本说明

更新时间：2026-06-02 13:17

> 本文记录当前 `PriFold/symfold` 的 **v2_marsfix** 版本：用 PriFold 数据 + MARS-LX 语言模型，参考 RNADiffFold/SymFold 的 DA-SE-DiT + Bernoulli discrete flow matching 做 RNA 二级结构 contact map 生成。当前路线**不接入 RNA-FM，不接入 UFold**。

---

## 1. 当前版本定位

当前推荐训练配置：

| 任务 | 配置 | 训练集 | 验证集 | 测试集 |
|---|---|---|---|---|
| bpRNA 模型 | `symfold/config/v2_bprna_marsfix.json` | bpRNA TR0 | bpRNA VL0 | bpRNA TS0 |
| RNAStrAlign 模型 | `symfold/config/v2_rnastralign_marsfix.json` | RNAStrAlign tr | RNAStrAlign vl | RNAStrAlign ts + ArchiveII |

首轮 `v2_bprna` 在 20 epoch 左右出现 `val F1≈0.28` 平台期、`test F1` 下滑，因此已停止并切换到当前 `v2_*_marsfix` 分支。当前 `v2_bprna_marsfix` 正在训练中。

---

## 2. 代码文件结构

```text
symfold/
├── data.py                         # PriFold CSV/NPY 数据集、padding、pos_bias、seq_oh、augmentation
├── metrics.py                      # 上三角 contact P/R/F1/MCC，|i-j|>=3
├── train_v2.py                     # v2/v2_marsfix 训练入口，val/test 周期评估，AMP
├── eval_v2.py                      # 独立评估入口，按 dataset_mode 自动选择 test 集
├── gpu_monitor.py                  # 独立 GPU monitor daemon，写 gpu_stats.jsonl
├── show_gpu_stats.py               # 查看 gpu_stats.jsonl 的 tail/summary 工具
├── run_train.sh                    # 后台启动训练 + GPU monitor daemon
├── run_gpu_monitor.sh              # 给已有训练补挂 GPU monitor
├── v1/                             # v1 归档
├── v2/
│   ├── da_se_dit.py                # DA-SE-DiT-MARS 主干
│   ├── model.py                    # PriFoldSymFlow_v2：MARS extract + backbone + loss/sample
│   ├── discrete_flow.py            # Bernoulli DFM loss、CTMC rates、projection
│   └── README.md
└── config/
    ├── v2_bprna_marsfix.json       # 当前推荐：bpRNA
    ├── v2_rnastralign_marsfix.json # 当前推荐：RNAStrAlign
    ├── v2_bprna.json               # 首轮 v2 对照配置
    └── v2_rnastralign.json         # 首轮 v2 对照配置

prifold/
└── llama2_with_attn.py             # MARS wrapper：返回 hidden + 多层 hidden + attention map
```

---

## 3. 端到端数据流

```text
PriFold CSV/NPY 样本
  ├─ RNA sequence：U→T，tokenize，seq_oh
  ├─ contact map：L×L → padding 到 S×S，S 为 patch_size=4 的倍数
  └─ pos_bias：A-T=3 / G-C=6 / G-T=1，再乘 pos_bias_scale

MARS-LX frozen encoder
  ├─ layer hidden: [3,6,9,12] → MultiLayerMarsFusion → mars_fused_1d
  ├─ final hidden: 用于 fallback/global
  └─ last 6 layers × 12 heads attention → mars_attn_stack

Feature build
  ├─ x_t embedding: 8 ch
  ├─ MARS hidden outer concat: 64 ch
  ├─ MARS attention map projection: 16 ch
  ├─ pos_bias: 1 ch
  └─ seq_oh outer concat: 8 ch
       ↓
  concat = 97 ch, shape (B,97,S,S)
       ↓
PatchEmbed2D(kernel=4,stride=4)
       ↓
9-layer DA-SE-DiT-MARS
       ↓
UnPatchify → OutputRefineConv → symmetric logit
       ↓
Bernoulli Flow Loss / CTMC Sampling + Greedy Projection
```

---

## 4. MARS 条件提取

文件：`prifold/llama2_with_attn.py`

### 4.1 为什么要 wrapper

原始 `prifold/llama2.py` 默认走 `scaled_dot_product_attention`/flash attention 路径，不返回 attention weights。但 SymFold/RNA-FM 路线中，语言模型 attention map 是天然的 `L×L` pair feature，因此当前版本用 wrapper 派生 MARS forward。

### 4.2 当前返回内容

`mars_forward_with_attn(..., hidden_layer_indices=[3,6,9,12], return_hidden_layers=True)` 返回：

| 输出 | shape | 用途 |
|---|---|---|
| `hidden` | `(B,L+2,1056)` | final hidden，兼容旧接口 |
| `attn_stack` | `(B,6,12,L+2,L+2)` | last 6 layers × 12 heads attention map |
| `hidden_layers` | list of 4 × `(B,L+2,1056)` | MARS layer 3/6/9/12，多层融合 |

`PriFoldSymFlow_v2._extract_mars()` 会去掉 `<cls>/<eos>`，并 padding/truncate 到 `set_len=S`。

### 4.3 Frozen MARS eval 修复

`model.train()` 会递归把 frozen MARS 打回 train mode。当前已在 `_extract_mars()` 内强制：

```python
if self.freeze_mars:
    self.extractor.eval()
    with torch.no_grad():
        ...
```

这样避免 MARS dropout 导致 train/eval conditioning 不一致。

---

## 5. DA-SE-DiT-MARS 主干

文件：`symfold/v2/da_se_dit.py`

### 5.1 输入通道账

| 来源 | 通道数 | 说明 |
|---|---:|---|
| `x_t` embedding | 8 | 当前 flow 状态 0/1 embedding |
| MARS hidden outer concat | 64 | layer `[3,6,9,12]` 融合后投影到 32，再 outer concat |
| MARS attention map | 16 | last 6 × 12 attention，经 symmetrize + APC + Conv |
| `pos_bias` | 1 | PriFold 碱基配对先验 |
| `seq_oh` outer concat | 8 | 显式碱基身份先验 |
| **总计** | **97** | `(B,97,S,S)` |

### 5.2 MultiLayerMarsFusion

参考 SymFold v4/v5 的 `MultiLayerFMFusion`，但把 RNA-FM 换成 MARS：

1. learnable softmax 权重融合 layer `[3,6,9,12]`；
2. 每层独立 MLP 投影；
3. concat 后再 MLP 融合；
4. 加权平均投影作为 residual。

目标是避免只用 MARS final hidden 导致 pair feature 单薄。

### 5.3 MARS attention projection

`MarsAttentionProj`：

```text
(B, 6, 12, S, S)
→ reshape (B,72,S,S)
→ symmetrize
→ APC correction
→ 1×1 Conv → (B,16,S,S)
```

APC 用于去除 attention map 的背景偏置，是 RNA-FM/ESM contact head 常用做法。

### 5.4 DiT block

每层：

```text
AdaLN-Zero(time + mars_global + density)
  ├─ Dilated Axial Attention (row + col, QK-Norm + AxialRoPE)
  ├─ Triangle Multiplicative Update (only layer >= 6)
  └─ SwiGLU FFN
```

默认 9 层，dilation pattern：

```json
[1,1,1,2,2,2,4,4,4]
```

### 5.5 Triangle residual 修复

首轮 v2 的 triangle 模块返回 `z + gate * tri`，外层 block 又做 `x = x + tri_update(norm(x))`，zero-init 时变成 `x + norm(x)`，不是 identity。当前修复为：

```python
TriangleMultiplicativeUpdate.forward(...) -> gate * tri
# 外层 block 负责 residual
x = x + self.tri_update(self.tri_norm(x))
```

这样 zero-init 下 triangle 分支真正是 0 delta。

---

## 6. Flow loss 与 sampling

文件：`symfold/v2/discrete_flow.py`、`symfold/v2/model.py`

### 6.1 训练 forward

```text
x_1 = GT contact map
 t ~ Uniform(0,1)
x_t ~ Bernoulli(t*x_1 + (1-t)*rho_0)
logit, density_pred = backbone(x_t, t, conditions)
loss = adaptive BCE + focal + stack + row-count constraint + density MSE
```

### 6.2 Loss 组成

| 项 | 当前权重/设置 | 作用 |
|---|---:|---|
| adaptive BCE | `pos_weight_base=199`, `pos_weight_min=20` | 按 GT pair density 调整正样本权重 |
| focal | `gamma=1.5` | 抑制 easy examples，强调 hard examples |
| stacking loss | `0.05` | 鼓励相邻 stacking 连续性 |
| nc loss | `0.02` | 实际是 row-sum ≤ 1 软约束 |
| density MSE | `0.2` | 辅助预测配对密度 |

### 6.3 Density hint 策略

首轮 v2 部分训练样本看到 GT density，推理却依赖预测 density，存在“密度抄答案”风险。当前 `v2_marsfix`：

```json
"density_hint_dropout": 1.0
```

含义：训练时**不把 GT density 注入 backbone**，只保留 density head 的 MSE 监督；推理时再用预测 density 做 density-guided sampling。这样训练/推理路径更一致。

### 6.4 Sampling

```text
x_init ~ Bernoulli(rho_0)
如果 density_guided=True：
  先用 t=0.5, density_hint=None 前向一次，预测 density_pred
for k in 1..num_steps:
  logit = backbone(x_t, t_k, density_hint=density_pred)
  p_x1 = sigmoid(logit)
  rate_01, rate_10 = CTMC rates
  rate_01 *= min(1, 2*density_pred)   # density-guided damping
  τ-leap flip x_t
最终：project_to_valid_contact_map(x_t, p_x1_last)
```

注意：projection 只从 `x_t==1` 的候选边中选边，因此 sampling 候选池质量会强烈影响最终 F1。如果 `v2_marsfix` epoch 10 仍低，应优先排查 `density_guided=False` 与 score-only projection 的消融。

---

## 7. 训练脚本与监控

### 7.1 训练入口

```bash
cd /root/aigame/dannyyan/PriFold
bash symfold/run_train.sh symfold/config/v2_bprna_marsfix.json
```

`run_train.sh` 会：

1. 激活 `RNADiffFold_torch260`；
2. 后台启动 `symfold/train_v2.py`；
3. 后台启动独立 GPU monitor daemon；
4. 记录 pid/log/stdout/stderr/gpu_stats。

### 7.2 AMP

当前默认：

```json
"amp_dtype": "bf16"
```

在 `RNADiffFold_torch260`（torch 2.6.0+cu124）+ H20 上稳定，且比 fp32 显存低约 25%-30%。

### 7.3 周期性评估

| 配置 | 默认值 | 说明 |
|---|---:|---|
| `eval_every` | 2 | 每 2 epoch 跑 val |
| `test_eval_every` | 10 | 每 10 epoch 跑 test |
| `save_every` | 10 | 每 10 epoch 保存 epoch checkpoint |

输出：

```text
symfold/outputs/v2_bprna_marsfix/
├── history.json
├── test_eval_history.json
├── training_curves.png
├── gpu_stats.jsonl
└── model/
    ├── best.pt
    ├── last.pt
    └── epoch_XXX.pt
```

### 7.4 GPU monitor

实时：

```bash
tail -f symfold/outputs/v2_bprna_marsfix/gpu_stats.jsonl
```

表格查看：

```bash
python -m symfold.show_gpu_stats symfold/outputs/v2_bprna_marsfix/gpu_stats.jsonl --tail 20
python -m symfold.show_gpu_stats symfold/outputs/v2_bprna_marsfix/gpu_stats.jsonl --summary
```

当前 `v2_bprna_marsfix` 运行中监控示例：

```text
nvml_used≈61.6GB, target≈61.0GB, util peak≈99%, temp≈38C, power≈299W
```

---

## 8. 当前训练状态（写文档时刻）

`v2_bprna_marsfix` 已启动：

| 项 | 值 |
|---|---|
| task | `v2_bprna_marsfix` |
| train pid | `119044` |
| monitor pid | `119045` |
| 当前进度 | epoch 9 附近训练中 |
| 当前观察 | loss 正常下降，待 epoch 10 test eval 验证 |

查看：

```bash
tail -f symfold/logs/v2_bprna_marsfix/v2_bprna_marsfix.log
python -m symfold.show_gpu_stats symfold/outputs/v2_bprna_marsfix/gpu_stats.jsonl --tail 20
```

---

## 9. 与 RNADiffFold/SymFold 的关系

参考项目：`/root/aigame/dannyyan/RNADiffFold/symfold`

当前只参考 SymFold 的工程/结构思想，不复制外部条件器：

| SymFold v4/v5 做法 | 当前 v2_marsfix 对应做法 |
|---|---|
| RNA-FM multi-layer hidden fusion | MARS layer `[3,6,9,12]` hidden fusion |
| RNA-FM attention maps | MARS last 6 layers × 12 heads attention maps |
| UFold conditioner | 不使用；由 PriFold `pos_bias` + `seq_oh` 替代显式先验 |
| DA-SE-DiT 9 层 + dilation 1/2/4 | 保留 |
| Triangle Update | 保留，并修正 delta residual |
| Density head / density-guided sampling | 保留，但训练端去掉 GT density 注入以避免泄漏 |
| Cosine τ-leap sampling | 保留 |

---

## 10. 后续观察点

1. **epoch 10 的 `bprna-test F1`**：这是第一关键观察点；旧 v2 epoch 9 test F1 约 `0.28`。
2. 若 epoch 10 未超过旧版，优先做 sampling 消融：
   - `density_guided=false`
   - `num_steps=50`
   - projection 从 `score` 全图选边，而不是只从 sampled `x_t` 候选中选
3. 若 val F1 提升而 test 仍低，继续检查数据分布与 augmentation。
4. 若 loss 下降但 val/test 不动，重点排查 flow sampling 与 projection，而不是 backbone。
