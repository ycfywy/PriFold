# CLAUDE.md — PriFold 当前工作指南

> 最近更新：2026-06-15 12:42。当前主线任务是 `symfold` 的 PriFold-SymFlow/DensityNet；**v8 DensityNet-Pro 训练已启动**（基于 v7 分析结果，新增 OHEM/FP penalty/BP compat/Shift loss/DropPath）。v7 full 已完成（test F1=0.6538），全面分析已完成。

## 1. 项目定位

PriFold 是 RNA 二级结构预测项目：输入 RNA 序列，输出 `L×L` contact map。

当前有两条线：

1. **官方 PriFold 主线**：判别式 RNAformer/RiboFormer baseline，入口在顶层 `train.py` / `inference.py`。
2. **PriFold-SymFlow/DensityNet 实验线**：`symfold/`，经历了生成式（v1-v6 Bernoulli Discrete Flow Matching）→ **纯判别式（v7 DensityNet）** 的转型。

当前主要开发/训练对象：**v7 DensityNet**（200 epochs 完成，深度分析完成，消融实验待运行）。v6 已完成分析，v5/v4 已完成。

---

## 2. 当前状态

### 训练状态

```text
A. v7_full — 已完成 ✅（2026-06-12 06:18）
config: symfold/config/v7/v7_full.json
device: cuda:0
entry: symfold/train/train_v7.py
架构: MARS-LX (160M 冻结) + Axial Transformer (8层)
参数量: 3.56M trainable / 164M total（MARS 冻结）
LR: 3e-4 peak, cosine over 200 epochs, warmup=5
最终: 200 epochs 完成, best val F1=0.6408 @ epoch ~178
best test F1 = 0.6538 @ epoch 199 🎉
pred/gt ratio = 1.122
特点: 单次前向传播，无 flow sampling
新增特性（config 开关，当前关闭，消融用）: OHEM, FP penalty, BP compat, Family balanced

B. v6_full — 已完成（case analysis 已做，结论：转判别式）
config: symfold/config/v6/v6_full.json
参数量: 26M trainable
最终: epoch 217 (中断), best test F1=0.6083 @ epoch 189
pred/gt ratio = 1.07
结论: 生成式噪声大，低密度样本过预测严重，转 v7 判别式

C. v5_bprna — 已完成（2026-06-05 启动，2026-06-06 结束）
config: symfold/config/v5/v5_bprna.json
参数量: 26.1M trainable
best val F1 = 0.6138 @ epoch 215
best test F1 = 0.6188 @ epoch 209

D. v4_bprna — 已完成
config: symfold/config/v4/v4_bprna.json
best val F1=0.4946 @ epoch 245

E. v4_rnastralign — 已结束
config: symfold/config/v4/v4_rnastralign.json
best val F1=0.9459 @ epoch 41
```

**v7 训练曲线（判别式，200 epochs 完成）**：

| Epoch | Test F1 | Test Precision | Test Recall | Test MCC | pred/gt |
|-------|---------|----------------|-------------|----------|---------|
| 9 | 0.4888 | 0.4738 | 0.5335 | 0.4937 | — |
| 49 | 0.6048 | 0.5817 | 0.6511 | 0.6086 | — |
| 109 | 0.6319 | 0.6083 | 0.6761 | 0.6352 | 1.114 |
| 139 | 0.6484 | 0.6210 | 0.6954 | 0.6516 | 1.132 |
| 179 | 0.6533 | 0.6273 | 0.6990 | 0.6565 | 1.127 |
| **199** | **0.6538** | **0.6293** | **0.6982** | **0.6570** | **1.122** |

**v7 深度分析关键结论**（200 epoch 完成后全量分析，5611 样本）：

| 指标 | Train | Val | Test |
|------|-------|-----|------|
| F1 | 0.900 | 0.641 | 0.653 |
| F1=0 | 9 (0.3%) | 43 (3.3%) | 40 (3.1%) |
| F1≥0.9 | 1936 (64%) | 269 (21%) | 270 (21%) |
| pred/gt | 1.107 | 1.229 | 1.261 |

- **核心瓶颈：Precision 不足**（vs baseline: P差17%, R差仅6%）→ FP 过多
- **泛化 gap 显著**：Train F1=0.900 vs Test F1=0.653（-24.7%）
- **家族表现两极化**：CRW/SPR F1>0.93，RFAM F1=0.751，SRP F1=0.629
- **F1=0 的 86/92 来自 RFAM 长尾罕见子家族**
- **零配对样本**：发现 7 个 GT 配对数为 0 的数据异常

### 最新结果对照

| 版本 | best val F1 | best epoch | test F1 | pred/gt | 状态 |
|---|---:|---:|---:|---:|---|
| `v7_full` | **0.6408** | ~178 | **0.6538** (@e199) | 1.12 | ✅ 已完成 |
| `v6_full` | 0.6059 | 213 | 0.6083 (@e189) | 1.07 | 已完成(分析) |
| `v5_bprna` | 0.6138 | 215 | 0.6188 (@e209) | 1.17 | 已完成 |
| `v4_bprna` | 0.4946 | 245 | 0.4869 (@e219) | 1.47 | 已完成 |
| `v4_rnastralign` | 0.9459 | 41 | 0.9459 | — | 已完成 |
| `v3_bprna` | 0.4003 | 105 | 0.4053 | ~1.5 | 历史 |

vs 主线 PriFold bprna-test: F1=0.7700

**关键观察**：
- v7 test F1=0.6538 > v5 的 0.6188 > v6 的 0.6083，**历史最佳** 🎉
- v7 仅 3.56M 参数，训练效率远超 v5/v6 的 26M flow model
- v7 距 baseline 差距：**~15.2%**（vs v5 的 ~20%，v6 的 ~21%）
- 200 epochs 完成，patience 21/30（没有 early stop，schedule 跑完）
- 判别式单次前向传播 vs 生成式多步 flow：推理速度也大幅提升
- pred/gt=1.12（train）→1.26（test）：泛化时过预测加剧
- **核心瓶颈已明确**：Precision 不足（vs baseline P差17%, R差仅6%）→ 减少 FP 是最大改进方向

---

## 3. 环境

训练 SymFlow/DensityNet 只用：

```bash
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
source activate RNADiffFold_torch260
export PYTHONPATH=/root/aigame/dannyyan/PriFold
```

环境信息：

```text
conda: /root/aigame/dannyyan/miniconda3
python: 3.10.20
pytorch: 2.6.0+cu124
GPU: NVIDIA H20 97GB
transformers: 4.46.2
accelerate: 1.1.0
```

不要用旧 `prifold` 环境训练 SymFlow：`torch 2.2.1+cu121` 在 H20 上曾触发 `libcublasLt.so.12` SIGFPE。

---

## 4. 关键目录

```text
PriFold/
├── train.py / inference.py          # 官方 PriFold 主线
├── prifold/                         # MARS/LLaMA2 等语言模型代码
├── utils/                           # 主线 PriFold 工具与 RNAformer
├── symfold/                         # 当前实验主目录
│   ├── data.py                      # 数据加载、pos_bias、seq_oh、padding、bucket sampler
│   ├── metrics.py                   # P/R/F1/MCC
│   ├── train/                       # ★ 训练相关
│   │   ├── train_v7.py             # v7 训练入口（DensityNet，纯判别式）
│   │   ├── train_v6.py             # v6 训练入口
│   │   ├── train_v5.py             # v5 训练入口
│   │   ├── train_v4.py             # v4 训练入口
│   │   ├── train_v3.py             # v3 训练入口（v4-v7 base）
│   │   ├── train_v2.py             # v2 训练入口
│   │   ├── run_train.sh            # 后台训练 + GPU monitor（支持 v1-v7）
│   │   ├── run_gpu_monitor.sh      # GPU 监控补挂脚本
│   │   └── gpu_monitor.py          # GPU 监控 daemon
│   ├── eval/                        # ★ 评估相关
│   │   ├── eval_v6_improved.py     # v6 推理优化评估
│   │   ├── eval_v4.py              # v4 评估入口
│   │   ├── eval_v3.py              # v3 评估入口
│   │   └── eval_v2.py              # v2 评估入口
│   ├── analysis/                    # ★ 分析与可视化
│   │   ├── deep_analysis_v7.py     # v7 深度分析（5611 样本全覆盖）
│   │   ├── analyze_v7_cases.py     # v7 case 分析
│   │   ├── analyze_v6_cases.py     # v6 case 分析
│   │   ├── analyze_v5_cases.py     # v5 case 分析
│   │   ├── analyze_cases.py        # 通用 bad-case 分析
│   │   ├── visualize_predictions.py    # 预测可视化
│   │   ├── visualize_case_analysis.py  # case 分析可视化
│   │   └── show_gpu_stats.py       # GPU 统计可视化
│   ├── v7/                          # v7 模型代码（DensityNet）
│   ├── v6/                          # v6 模型代码（模块化 loss + DASO）
│   ├── v5/                          # v5 模型代码
│   ├── v4/                          # v4 模型代码
│   ├── v3/                          # v3 模型代码
│   ├── v2/                          # v2 模型代码
│   ├── v1/                          # v1 模型代码
│   ├── config/                      # 配置（按版本分目录）
│   │   ├── v7/                     # v7_full.json + ablations/
│   │   ├── v6/                     # v6_full.json + ablations/
│   │   ├── v5/                     # v5_bprna.json
│   │   ├── v4/                     # v4_bprna.json, v4_rnastralign.json
│   │   ├── v3/                     # v3_bprna.json, v3_rnastralign.json
│   │   ├── v2/                     # v2_*.json (7个)
│   │   ├── v1/                     # v0/v1 配置
│   │   └── gen_ablation_configs.py # 消融配置生成工具
│   ├── outputs/                     # 训练输出（不动）
│   │   ├── v7_full/                # model/, history.json, case_analysis/, deep_analysis/
│   │   ├── v6_full/                # model/, history.json
│   │   └── v5_bprna/ ...          # 各版本输出
│   └── logs/                        # 训练日志（不动）
└── docs/                            # 项目文档
```

v7 代码（当前）：

```text
symfold/v7/
├── model.py                         # DensityNet: Axial Transformer + MARS-LX
└── __init__.py
```

架构流程：
```text
RNA → MARS-LX (冻结, 160M)
    → 1D hidden + 2D attention maps
    → Pair Feature Construction (outer product + attn proj)
    → Axial Transformer Stack (8 layers, row-attn + col-attn + FFN)
    → Contact Logit Head + Density Prediction Head
    → Score-based Projection (budget_fraction=0.30)
    → Contact Map
```

v6 代码（消融用）：

```text
symfold/v6/
├── da_se_dit.py                     # v6 主干（同 v5 架构）
├── discrete_flow.py                 # ModularFlowLoss
├── model.py                         # PriFoldSymFlow_v6 wrapper
├── __init__.py
└── README.md
```

v5 代码：

```text
symfold/v5/
├── da_se_dit.py                     # v5 主干：320dim × 12layers + dilation[1..8]
├── discrete_flow.py                 # BernoulliFlowLoss_v6
├── model.py                         # PriFoldSymFlow_v5 wrapper
└── __init__.py
```

---

## 5. v7 模型要点（当前）

v7 是**架构转型**：从生成式 flow model（v4-v6）转为纯判别式 DensityNet。

### 转型动机（来自 v6 case analysis）

1. 生成式 flow sampling 噪声大，低密度样本系统性过预测
2. Flow model 需要多步采样，推理慢
3. 26M 参数的 flow model 实际表现 ≈ v5 水平（0.608 vs 0.619）
4. 判别式单次前向传播更高效、更稳定

### v7 核心设计

1. **Axial Transformer**（8层，hidden_dim=160，4头）
   - Row attention + Column attention（避免 L² 全注意力）
   - 使用 PyTorch SDPA (Flash Attention) 加速
   - 仅 3.56M trainable params

2. **Density-Stratified Tversky Loss (DST)**
   - 对低密度样本（density < 0.18）使用更高的 FN 惩罚
   - alpha=0.7, beta=0.3：偏向召回率
   - 权重 0.4

3. **Focal + Dice + Pair Count + Ratio Penalty**（继承 v5/v6）
   - focal_gamma=1.0, dice_weight=0.5
   - pair_count_weight=0.3, ratio_penalty_weight=0.2

4. **BF16 混合精度训练**
   - batch_size=12（vs v5/v6 的 6）
   - max_len=490

5. **快速训练**
   - 无 flow sampling 开销
   - 单次前向传播 per batch
   - 200 epoch schedule（vs v5/v6 的 300）

### v7 配置结构

```json
{
  "model": {
    "version": "v7",
    "hidden_dim": 160,
    "num_layers": 8,
    "num_heads": 4,
    "dim_head": 40,
    "focal_gamma": 1.0,
    "pos_weight_base": 99.0,
    "direct_weight": 0.4,
    "pair_count_weight": 0.3,
    "dice_weight": 0.5,
    "ratio_penalty_weight": 0.2,
    "ratio_penalty_threshold": 1.2,
    "density_loss_weight": 0.3,
    "dst_low_threshold": 0.18,
    "dst_tversky_alpha": 0.7,
    "dst_tversky_beta": 0.3,
    "dst_weight": 0.4
  },
  "training": {
    "lr": 3e-4,
    "epochs": 200,
    "batch_size": 12,
    "warmup_epochs": 5,
    "patience": 30,
    "bf16": true,
    "max_len": 490
  },
  "sampling": {
    "default_budget_fraction": 0.30,
    "score_threshold": 0.4
  }
}
```

### v7 新增改进（config 开关控制，可消融）

基于 v7 case analysis 发现的问题（F1=0 cases、FP 不受惩罚、平移预测），新增四项改进：

1. **OHEM** (Online Hard Example Mining)
   - 只取 top-k hardest negatives 计算 neg_bce，FP 不再被海量 TN 稀释
   - `ohem_enabled`, `ohem_neg_ratio=3`（k = 3 × num_positives）

2. **FP Penalty**
   - 对 false positive 位置（pred>0.5 & GT=0）额外加权
   - `fp_penalty_enabled`, `fp_penalty_weight=3.0`

3. **碱基配对约束** (Base-Pair Compatibility)
   - 训练: 惩罚在不兼容位置（非 AU/GC/GU）的预测
   - 推理: 在 projection 时过滤不兼容配对
   - `bp_compat_enabled`, `bp_compat_weight=0.5`, `bp_compat_in_inference=true`

4. **RFAM 家族过采样** (Family Balanced Sampling)
   - 训练时按家族逆频率加权采样，稀有家族被更多地采样
   - alpha=0: 均匀, alpha=1: 完全均衡, alpha=0.5: sqrt-balanced
   - `family_balanced.enabled`, `family_balanced.alpha=0.5`

### v7 消融配置（待运行）

位于 `symfold/config/v7/ablations/`：

```text
# 已创建（6个，loss 组件消融）
v7_dst_only.json          # 仅 DST 损失
v7_fcr_only.json          # 仅 FCR 组件
v7_no_dst.json            # 移除 DST 损失
v7_no_fcr.json            # 移除 FCR 组件
v7_no_scp.json            # 移除 SCP 组件
v7_scp_only.json          # 仅 SCP 组件

# 计划中（尚未创建 JSON，代码 config 开关已就位，需用 gen_ablation_configs.py 生成）
v7_all_new.json           # 全部新特性启用（OHEM+FP+BP+Family）
v7_ohem_only.json         # 仅 OHEM
v7_fp_penalty_only.json   # 仅 FP penalty
v7_bp_compat_only.json    # 仅碱基配对约束
v7_family_balanced_only.json # 仅 RFAM 过采样
```

**注意**：新消融配置文件尚未实际创建到磁盘。运行消融前需先生成这些 JSON。

---

## 5b. v6 模型要点（历史——已分析完成）

v6 架构与 v5 **完全相同**，核心改进是 **loss 系统模块化**，专为论文消融设计。

### v6 case analysis 结论（详见 `docs/symfold_v6_case_analysis_report.md`）

- F1=0 的案例占 7.7%，F1<0.3 占 18.9%
- RFAM 家族 RNA 识别效果差
- 模型预测出与 Ground Truth 有偏移的配对
- **结论：生成式模型噪声太大 → 转向 v7 判别式**

### v6 推理优化尝试（详见 `docs/symfold_v6_inference_optimization.md`）

三种无需重训的策略：
1. Density-Conditional Budget Scaling
2. Multi-Sample Voting
3. Adaptive Score Threshold

部分策略被 v7 的 DST loss 在训练端吸收。

---

## 5c. v5 模型要点（历史）

v5 在 v4 基础上做了三个方向的改进：**更强 loss 信号 + 抗过预测 + 更大模型**。

架构流程（生成式 flow model）：
```text
RNA → MARS-LX hidden/attention + pos_bias/seq_oh
    → DA-SE-DiT-v5 (320dim × 12layers)
    → flow head + direct score head + density head
    → score-first projection
    → contact map
```

核心改进：Dice Loss, pair_count 6x, Ratio Penalty, 降低 Focal Gamma, 更大模型(26M), 更优 LR schedule。

---

## 6. v7 配置

当前 v7 配置：`symfold/config/v7/v7_full.json`

关键对比：

| 参数 | v7 | v5/v6 | 说明 |
|------|-----|-------|------|
| 架构 | Axial Transformer | DA-SE-DiT | 判别式 vs 生成式 |
| trainable params | 3.56M | 26M | 7x 更轻 |
| hidden_dim | 160 | 320 | |
| num_layers | 8 | 12 | |
| batch_size | 12 | 6 | BF16 允许更大 batch |
| lr | 3e-4 | 1.5e-4 | 更高 LR |
| epochs | 200 | 300 | 更短 schedule |
| bf16 | true | false | 混合精度 |
| max_len | 490 | 490 | |
| 推理 | 单次前向 | 多步 flow | 快 N 倍 |

MARS 使用最大模型：`mars_scale=lx`，约 160M 参数（冻结），hidden dim 1056，12 layers，12 heads。

---

## 7. 常用命令

### 恢复 v7 full 训练

```bash
cd /root/aigame/dannyyan/PriFold

# 恢复 v7 full（从 last checkpoint 续训）
bash symfold/train/run_train.sh symfold/config/v7/v7_full.json
```

### 启动 v7 消融实验

```bash
cd /root/aigame/dannyyan/PriFold

# v7 消融：移除 DST
bash symfold/train/run_train.sh symfold/config/v7/ablations/v7_no_dst.json

# v7 消融：仅 DST
bash symfold/train/run_train.sh symfold/config/v7/ablations/v7_dst_only.json

# v7 消融：移除 FCR
bash symfold/train/run_train.sh symfold/config/v7/ablations/v7_no_fcr.json
```

### 旧版 v6 消融实验

```bash
cd /root/aigame/dannyyan/PriFold

# 消融：关闭 Dice
bash symfold/train/run_train.sh symfold/config/v6/ablations/abl_no_dice.json
```

### 评估 v4

```bash
python symfold/eval/eval_v4.py \
  --ckpt symfold/outputs/v4_bprna/model/best.pt \
  --out_json symfold/outputs/v4_bprna/eval_best.json
```

`run_train.sh`（位于 `symfold/train/run_train.sh`）会自动：

- 激活 `RNADiffFold_torch260`
- 设置 `PYTHONPATH`
- 后台启动训练（自动检测 v7 → `train/train_v7.py`）
- 后台启动 GPU monitor
- 写日志、checkpoint、曲线、GPU JSONL

### bad-case 分析

```bash
# v7 deep analysis
python symfold/analysis/deep_analysis_v7.py

# v6 case analysis
python symfold/analysis/analyze_v6_cases.py

# v4 case analysis
python symfold/analysis/analyze_cases.py \
  --ckpt symfold/outputs/v4_bprna/model/best.pt \
  --test_sets bprna-test \
  --out_dir symfold/outputs/v4_bprna/case_analysis
```

---

## 8. 训练后必须看什么

不要只看 aggregate F1。

训练后必须检查：

1. low-density bins：
   - `density < 0.10`
   - `0.10 ≤ density < 0.18`
   - 看 `F1 / P / R / pred_pairs/gt_pairs`
2. length bins：
   - `<80`
   - `80-159`
   - `160-239`
   - `240+`
3. `F1=0` case 数量
4. `pred_pairs / gt_pairs` 分桶
5. projection 消融：`score / hybrid / sample`（仅生成式 v4-v6）
6. budget 消融：`default_budget_fraction=0.30/0.35/0.40`
7. v7 特有：DST loss 对低密度样本的改善程度

判断改进是否成功的优先标准：

```text
低 density 过预测下降
F1=0 case 减少
160+ 长度样本不再系统性错位
pred/gt ratio 更接近 1
整体 F1 提升
```

---

## 9. 文档命名规范

`docs/` 目录下的文档统一按以下前缀命名：

| 前缀 | 用途 | 示例 |
|------|------|------|
| `symfold_v{N}_` | symfold 特定版本相关文档 | `symfold_v7_deep_analysis_report.md` |
| `data_` | 数据集分析、数据分布相关 | `data_bprna_analysis.md` |
| `project_` | 项目级别文档（概述、规范、流程） | `project_convention.md` |
| `report_` | 日报/周报/阶段性报告 | `report_20260611.md` |

**命名格式**：`{前缀}{描述}.md`，描述用下划线分隔，全小写。

新写文档时**必须遵循此规范**。

---

## 10. 重要历史文档

优先看：

```text
docs/v7/v7_comprehensive_analysis_report.md          # ★★ v7 全面推理分析（13409样本，bad case可视化，原因分析，改进方案）
docs/symfold_v7_deep_analysis_report.md             # ★ v7 深度分析（200ep完成后，5611样本全覆盖，家族/失败模式/数据异常）
docs/symfold_v7_case_analysis_report.md             # v7 case 分析（F1=0、失败模式、RFAM 分析，@epoch 109）
docs/symfold_v7_f1_zero_loss_analysis.md            # F1=0 的 loss 机制深度分析 → OHEM/FP penalty 动机
docs/report_20260611.md                             # 2026-06-11 日报（case analysis + loss 分析 + 四项改进 + 续训）
docs/symfold_v6_case_analysis_report.md             # v6 case 分析 → v7 转型动机
docs/symfold_v6_inference_optimization.md           # v6 推理优化 → DST 设计灵感
symfold/v6/README.md                                # v6 DASO 使用指南 + 消融方案
docs/symfold_v5_improvements.md                     # v5 改进报告
docs/symfold_v4_walkthrough.md                      # v4 完整 walkthrough
docs/symfold_v4_improvements.md                     # v4 改进点
docs/symfold_v3_architecture_case_analysis.md       # v3 架构与 bad-case 诊断
docs/data_distribution_report.md                    # 数据集统计
docs/data_experiments_dataset_guide.md              # 两个实验的数据集详细说明
```

旧文档仅参考：

```text
docs/symfold_v1_postmortem.md
docs/symfold_v2_marsfix_architecture.md
docs/symfold_v1_improvement_plan.md
docs/symfold_v1_architecture.md
```

---

## 11. 数据和模型路径

```text
./data/
├── bprna/
├── RNAStrAlign/
└── archiveII/

./model/
├── mars_run-encoder-mars-lx-train-val-d0.15-2023_10_05_22_03_21/
│   └── ckpt_175000.pt
├── ss_model_bprna.pth
└── ss_model_rnastralign.pth
```

训练输出：

```text
# v7 full（已完成 200 epochs）
symfold/outputs/v7_full/history.json
symfold/outputs/v7_full/test_eval_history.json
symfold/outputs/v7_full/training_curves.png
symfold/outputs/v7_full/gpu_stats.jsonl
symfold/outputs/v7_full/model/                      # best.pt + last.pt
symfold/outputs/v7_full/case_analysis/              # 16 文件（CSV/JSON/PNG）
symfold/outputs/v7_full/deep_analysis/              # 37 文件（全量深度分析，per-family/stage/failure mode）

# v6 full（已完成分析）
symfold/outputs/v6_full/model/best.pt
symfold/outputs/v6_full/history.json

# v5 bpRNA（已完成）
symfold/outputs/v5_bprna/model/best.pt
symfold/outputs/v5_bprna/history.json
```

查看日志：

```bash
cd /root/aigame/dannyyan/PriFold

# v7 full（当前）
tail -30 symfold/logs/v7_full/v7_full.stdout.log

# v6 full
tail -30 symfold/logs/v6_full/v6_full.stdout.log
```

MARS-LX：

```text
参数量约 160M
hidden_dim=1056
n_layers=12
n_heads=12
vocab_size=20
```

---

## 12. 已知注意事项

1. `projection` 是把模型 score 后处理成合法 contact map 的步骤。
2. v7 使用 score-based projection（budget_fraction=0.30, score_threshold=0.4）。
3. v4-v6 默认 `projection_mode=score`，不依赖 sampled `x_t`。
4. `density_guided` 默认关闭（v3 实测降低 F1），v7 改用 DST loss 在训练端解决。
5. `run_train.sh` 支持 v1-v7 自动入口选择。
6. **v7 训练已完成**：200 epochs（2026-06-12 06:18），best val F1=0.6408，best test F1=0.6538。
7. v7 BF16 混合精度：确保 MARS-LX 输出和 Axial Transformer 数值稳定。
8. v7 消融配置（`symfold/config/v7/ablations/`）：目前磁盘上只有 6 个（loss 组件消融），新增 5 个（OHEM/FP/BP/Family）尚未创建 JSON 文件。
9. `save_every=9999`：只保留 best.pt + last.pt，不再存中间 epoch checkpoint。
10. **FamilyBalancedSampler 需配合 LengthBucketBatchSampler**：否则长短序列混合会 OOM。
11. **LengthBucketBatchSampler.__len__ 已修复**：现在返回实际 batch 数（之前日志 `step=2160/901` 分母不准）。
12. **深度分析已完成**（2026-06-12）：`symfold/analysis/deep_analysis_v7.py` 输出到 `symfold/outputs/v7_full/deep_analysis/`，文档见 `docs/v7_deep_analysis_report.md`。
13. **数据异常发现**：7 个零配对样本（GT pairs=0），可能需要在训练中过滤。
14. **全面推理分析已完成**（2026-06-15）：`symfold/analysis/comprehensive_v7_analysis.py` 输出到 `symfold/outputs/v7_full/comprehensive_analysis/`（100 张 bad case 卡片 + 9 类可视化 + CSV + report）。
15. **最大失败模式是 shifted_prediction（33%）**：预测位置偏移 ±1-3 位，说明需要 position-aware loss 或 margin loss。

---

## 13. 官方 PriFold baseline

主线 PriFold checkpoint 指标（2025-05-25, H20）：

| 测试集 | Precision | Recall | F1 |
|---|---:|---:|---:|
| bprna-test | 0.7938 | 0.7623 | 0.7700 |
| rnastralign-test | 0.9742 | 0.9744 | 0.9738 |
| archiveii-test | 0.9102 | 0.9037 | 0.9043 |

主线仅作为 baseline；当前实验目标是用轻量判别式 DensityNet 缩小差距。

---

## 14. 版本演进总结

```text
v1-v3: 生成式 Flow Matching 初探（F1 ~0.40）
v4:    + ControlInject + Direct Head + Density Budget（F1=0.49）
v5:    + Dice/Ratio Penalty + 大模型 26M（F1=0.62）
v6:    + 模块化 Loss + 消融框架（F1=0.61，过预测改善但 F1 略降）
v7:    ★ 转向纯判别式 DensityNet 3.56M（F1=0.654，已完成 200 epochs）
       距 baseline 0.77 差距: ~15.2%，核心瓶颈 = Precision（FP 过多）
       深度分析完成: 5611 样本全覆盖，家族/密度/长度/失败模式均分析
       下一步: 创建并运行消融实验（OHEM + FP penalty + 碱基配对约束 + RFAM 过采样）
```

---

## 15. 下一步行动（优先级排序）

```text
1. 生成新消融配置 JSON（5个）
   → 用 config/gen_ablation_configs.py 或手动创建到 symfold/config/v7/ablations/
   → v7_ohem_only, v7_fp_penalty_only, v7_bp_compat_only, v7_family_balanced_only, v7_all_new

2. 运行消融实验（推荐顺序）
   → 优先: v7_ohem_only（直接解决 FP 稀释问题）
   → 次优: v7_all_new（全部新特性开启，看整体天花板）
   → 然后: v7_fp_penalty_only, v7_bp_compat_only, v7_family_balanced_only

3. 消融结果分析
   → 每个消融训练完成后运行 analysis/deep_analysis_v7.py 做对照分析
   → 重点看: F1=0 数量变化、Precision 改善、低密度 bin 表现

4. 后续方向（根据消融结果决定）
   → 如果 OHEM 有效：考虑更激进的 neg_ratio 或结合 FP penalty
   → 如果 BP compat 有效：考虑在 inference 时加入更多物理约束
   → 如果 Family balanced 有效：考虑 curriculum learning
   → 如果仍有 gap：考虑更大模型（增加 layers/hidden_dim）或 MARS 微调
```

---

## 16. symfold 代码组织约定

**目录结构原则**（2026-06-15 重组后确立，后续新增代码须遵循）：

```text
symfold/
├── data.py, metrics.py              # 共享模块（数据加载、指标）保留在根
├── train/                           # 所有训练相关（入口脚本、训练工具、shell 脚本）
│   ├── train_v{N}.py              # 各版本训练入口
│   ├── run_train.sh               # 统一启动脚本
│   ├── run_gpu_monitor.sh         # GPU 监控补挂
│   └── gpu_monitor.py             # GPU daemon
├── eval/                            # 所有评估相关
│   └── eval_v{N}*.py             # 各版本评估脚本
├── analysis/                        # 所有数据分析与可视化
│   ├── analyze_v{N}_cases.py      # case 分析
│   ├── deep_analysis_v{N}.py      # 深度分析
│   ├── visualize_*.py             # 可视化
│   └── show_gpu_stats.py          # GPU 统计查看
├── config/                          # 配置，按版本分目录
│   ├── v{N}/                      # 各版本主配置 + ablations/ 子目录
│   └── gen_ablation_configs.py    # 配置生成工具
├── v{N}/                            # 模型代码（model.py, backbone, loss 等）
├── outputs/                         # 训练输出（不整理，按 task_name 自动创建）
└── logs/                            # 训练日志（不整理，按 task_name 自动创建）
```

**新增代码规则**：
1. 新建训练入口 → 放 `train/`
2. 新建评估脚本 → 放 `eval/`
3. 新建分析/可视化脚本 → 放 `analysis/`
4. 新建模型版本 → 创建 `v{N}/` 目录
5. 新建配置 → 放 `config/v{N}/`
6. Python import 使用完整模块路径：`from symfold.train.train_v7 import ...`
7. `outputs/` 和 `logs/` 按 task_name 自动管理，不手动整理
