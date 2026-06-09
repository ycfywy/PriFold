# CLAUDE.md — PriFold 当前工作指南

> 最近更新：2026-06-08 10:45。当前主线任务是 `symfold` 的 PriFold-SymFlow；v5 训练已完成（best test F1=0.6188），**v6 已就绪**——模块化 loss + 消融实验支持，准备跑 ablation study。

## 1. 项目定位

PriFold 是 RNA 二级结构预测项目：输入 RNA 序列，输出 `L×L` contact map。

当前有两条线：

1. **官方 PriFold 主线**：判别式 RNAformer/RiboFormer baseline，入口在顶层 `train.py` / `inference.py`。
2. **PriFold-SymFlow 实验线**：`symfold/`，用 PriFold 数据 + MARS-LX 条件 + Bernoulli Discrete Flow Matching 生成 contact map。

当前主要开发/训练对象：**SymFlow v6**（已就绪，待跑消融实验）。v5 已完成训练，v4 已完成。

---

## 2. 当前状态

### 训练状态

**v5_bprna 训练已完成**（early stop @ epoch 220，patience=40 触发）。

```text
A. v5_bprna — 已完成（2026-06-05 启动，2026-06-06 结束）
config: symfold/config/v5_bprna.json
device: cuda:0
entry: symfold/train_v5.py
参数量: 26.1M trainable（vs v4 的 ~8M，3x 提升）
LR: 1.5e-4 peak, cosine over 300 epochs, warmup=8
最终: epoch 220, lr=2.67e-05, train_loss=0.0054
best val F1 = 0.6138 @ epoch 215 🎉
best test F1 = 0.6188 @ epoch 209 🎉

B. v4_bprna — 已完成
config: symfold/config/v4_bprna.json
best val F1=0.4946 @ epoch 245

C. v4_rnastralign — 已结束
config: symfold/config/v4_rnastralign.json
best val F1=0.9459 @ epoch 41
```

**v5 最终训练结果**：

| 指标 | v5 (final) | v4 (final) | 提升 |
|------|------------|------------|------|
| best val F1 | **0.6138** (e215) | 0.4946 (e245) | **+11.9%** |
| best test F1 | **0.6188** (e209) | 0.4869 (e219) | **+13.2%** |
| pred/gt ratio | **1.17** | 1.47 | **显著改善** |
| Precision | **0.59** | 0.43 | **+16%** |
| Recall | 0.66 | 0.60 | +6% |
| 训练 epoch | 220 (early stop) | 249 | 更快收敛 |

**关键观察**：
- v5 最终 F1=0.6138，大幅超越 v4 的 0.4946（+24% 相对提升）
- 过预测问题基本解决：pred/gt 从 1.47 降到 1.17（接近理想值 1.0）
- Precision 大幅提升（0.43→0.59），Recall 同时提升（0.60→0.66）
- 与主线 PriFold 差距从 28%（v3）→ 24%（v4）→ **16%**（v5）持续缩小
- train_loss 从初期 0.039 降到 0.0054，模型充分收敛

**v5 设计改进总结**（已验证有效）：
1. **Dice loss**（权重 0.5）：直接优化 F1 proxy，解决 BCE 和 F1 脱钩
2. **pair_count_weight 6x**（0.05→0.3）：强力约束配对数量
3. **ratio_penalty**（权重 0.2）：显式惩罚 pred/gt > 1.2
4. **focal_gamma 降低**（2.0→1.0）：保留中等难度梯度
5. **更大模型**（320 dim × 12 layers = 26M params）
6. **更高 LR + 短 schedule**（1.5e-4, 300 epoch）：真正的 cosine 衰减
7. **更低 budget_fraction**（0.35→0.30）：更紧的投影预算

训练记录位置：

```text
# v5 bpRNA（当前运行）
symfold/logs/v5_bprna/v5_bprna.stdout.log
symfold/logs/v5_bprna/v5_bprna.stderr.log
symfold/outputs/v5_bprna/gpu_stats.jsonl
symfold/outputs/v5_bprna/history.json
symfold/outputs/v5_bprna/test_eval_history.json
symfold/outputs/v5_bprna/training_curves.png
symfold/outputs/v5_bprna/model/

# v4 bpRNA（已完成）
symfold/outputs/v4_bprna/history.json
symfold/outputs/v4_bprna/training_curves.png
symfold/outputs/v4_bprna/model/best.pt
```

查看日志：

```bash
cd /root/aigame/dannyyan/PriFold

# v5 bpRNA（当前运行中）
tail -30 symfold/logs/v5_bprna/v5_bprna.stdout.log
python -m symfold.show_gpu_stats symfold/outputs/v5_bprna/gpu_stats.jsonl --summary

# v4 bpRNA（已结束）
tail -30 symfold/logs/v4_bprna/v4_bprna.stdout.log
```

v5 训练已完成。checkpoint 保留 `best.pt` 和 `last.pt`。

### 最新结果对照

| 版本 | best val F1 | best epoch | test F1 | pred/gt | 状态 |
|---|---:|---:|---:|---:|---|
| `v5_bprna` | **0.6138** | 215 | **0.6188** (@e209) | **1.17** | 已完成 |
| `v4_bprna` | 0.4946 | 245 | 0.4869 (@e219) | 1.47 | 已完成 |
| `v4_rnastralign` | 0.9459 | 41 | 0.9459 | — | 已完成 |
| `v3_bprna` | 0.4003 | 105 | 0.4053 | ~1.5 | 历史 |

vs 主线 PriFold bprna-test: F1=0.7700（差距 ~16%，从 28%→24%→**16%** 持续缩小）

v5 test eval 细节（epoch 209，最佳 test）：
- Precision: 0.5887, Recall: 0.6763, F1: 0.6188
- pred/gt ratio: 1.19（对比 v4 的 1.47，改善 19%）

### v3 历史结果（对照）

- best val F1 = 0.4003 @ epoch 105
- `bprna-test` F1 ≈ 0.4053（density_guided=True），≈ 0.4530（density_guided=False）
- 主要失败模式：低 density 过预测、中长 RNA 错位、projection 依赖 sampled `x_t` 候选。

---

## 3. 环境

训练 SymFlow 只用：

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
│   ├── run_train.sh                 # 后台训练 + GPU monitor
│   ├── gpu_monitor.py               # GPU 监控 daemon
│   ├── show_gpu_stats.py            # 查看 gpu_stats.jsonl
│   ├── analyze_cases.py             # bad-case 分析，已支持 v4
│   ├── train_v6.py                  # v6 训练入口（当前，消融实验用）
│   ├── train_v5.py                  # v5 训练入口
│   ├── train_v4.py                  # v4 训练入口
│   ├── eval_v4.py                   # v4 评估入口
│   ├── v6/                          # v6 模型代码（当前开发，模块化 loss）
│   ├── v5/                          # v5 模型代码
│   ├── v4/                          # v4 模型代码
│   └── config/                      # v1/v2/v3/v4/v5/v6 配置
└── docs/                            # 项目文档
```

v6 代码（当前，消融实验用）：

```text
symfold/v6/
├── da_se_dit.py                     # v6 主干（同 v5 架构，增加 head 开关）
├── discrete_flow.py                 # ModularFlowLoss：每个 loss 组件独立开关
├── model.py                         # PriFoldSymFlow_v6 wrapper
├── __init__.py
└── README.md                        # 详细使用文档和消融方案
```

v5 代码：

```text
symfold/v5/
├── da_se_dit.py                     # v5 主干：320dim × 12layers + dilation[1..8]
├── discrete_flow.py                 # BernoulliFlowLoss_v6: Dice + ratio penalty + 强 pair_count
├── model.py                         # PriFoldSymFlow_v5 wrapper
└── __init__.py
```

v4 代码：

```text
symfold/v4/
├── da_se_dit.py                     # v4 主干：condition bias + ControlInject + direct head
├── discrete_flow.py                 # BernoulliFlowLoss_v5 + projection + density budget
├── model.py                         # PriFoldSymFlow_v4 wrapper
└── __init__.py
```

---

## 5. v5 模型要点（当前）

v5 在 v4 基础上做了三个方向的改进：**更强 loss 信号 + 抗过预测 + 更大模型**。

架构流程（和 v4 相同）：
```text
RNA → MARS-LX hidden/attention + pos_bias/seq_oh
    → DA-SE-DiT-v5 (320dim × 12layers)
    → flow head + direct score head + density head
    → score-first projection
    → contact map
```

### v5 相对 v4 的核心改进：

1. **Dice Loss（F1 proxy）** — 权重 0.5
   - 直接优化 F1-like 目标，解决 BCE 和 F1 脱钩问题
   - 可微 Dice = 2×intersection / (pred + gt)

2. **更强的 pair_count 约束** — 权重 0.05 → 0.3（6x）
   - 直接校准 predicted density vs GT density

3. **Ratio Penalty** — 权重 0.2，阈值 1.2
   - 当 pred/gt > 1.2 时施加显式惩罚
   - 效果：pred/gt 从 v4 的 1.47 降到 1.25

4. **降低 Focal Gamma** — 2.0 → 1.0
   - 保留中等难度样本的梯度信号
   - train_loss 维持在 0.04（vs v4 的 0.01），更多有效梯度

5. **降低 pos_weight_base** — 199 → 99
   - 减少对 positive 的过度鼓励，降低 false positive

6. **更大模型** — 320 dim × 12 layers = 26M params（vs v4 的 8M）
   - dilation pattern: [1,1,1,2,2,2,4,4,4,8,8,8]
   - triangle update 从 layer 4 开始（vs v4 的 layer 6）

7. **更优 LR schedule** — 1.5e-4 peak, 300 epoch cosine
   - 真正的衰减周期（vs v4 的 999 epoch 导致 LR 几乎恒定）

8. **更紧 budget** — default_budget_fraction 0.35 → 0.30

---

## 5b. v6 模型要点（当前——消融实验）

v6 架构与 v5 **完全相同**，核心改进是 **loss 系统模块化**，专为论文消融设计。

### v6 相对 v5 的改动：

1. **模块化 loss**：每个组件通过 config `"loss"` key 独立 `enabled/weight` 控制
2. **新增 Tversky loss**：广义 Dice，可独立控制 FP/FN 权重（alpha/beta）
3. **新增 Label Smoothing**：可选标签平滑
4. **架构消融开关**：`use_direct_head`、`use_density_head`、`control_every=0`
5. **`describe()` 方法**：启动时打印所有 loss 组件状态

### v6 配置结构：

```json
{
  "model": { "version": "v6", ... },
  "loss": {
    "bce": {"enabled": true, "focal_gamma": 1.0, "pos_weight_base": 99, ...},
    "dice": {"enabled": true, "weight": 0.5},
    "tversky": {"enabled": false, "alpha": 0.3, "beta": 0.7},
    "pair_count": {"enabled": true, "weight": 0.3},
    "ratio_penalty": {"enabled": true, "weight": 0.2, "threshold": 1.2},
    "density": {"enabled": true, "weight": 0.2},
    "direct": {"enabled": true, "weight": 0.4},
    "stacking": {"enabled": true, "weight": 0.05},
    "non_crossing": {"enabled": true, "weight": 0.03},
    "label_smoothing": {"enabled": false, "epsilon": 0.01}
  },
  "training": { ... },
  "sampling": { ... }
}
```

消融实验只需改 JSON 的 `enabled` 字段，无需改代码。详见 `symfold/v6/README.md`。

### v4 模型要点（旧，参考用）

详见：`docs/prifold_symflow_v4_walkthrough.md`。

---

## 6. v6 / v5 配置

当前 v5 配置：`symfold/config/v5_bprna.json`

关键字段（vs v4 的变化用 ← 标注）：

```json
"model": {
  "version": "v5",
  "hidden_dim": 320,              // ← v4: 256
  "num_layers": 12,               // ← v4: 9
  "dim_head": 80,                 // ← v4: 64
  "focal_gamma": 1.0,             // ← v4: 2.0
  "pos_weight_base": 99.0,        // ← v4: 199.0
  "direct_weight": 0.4,           // ← v4: 0.3
  "pair_count_weight": 0.3,       // ← v4: 0.05
  "dice_weight": 0.5,             // ← v4: 无
  "ratio_penalty_weight": 0.2,    // ← v4: 无
  "ratio_penalty_threshold": 1.2, // ← v4: 无
  "control_every": 3,             // ← v4: 2
  "tri_start_layer": 4,           // ← v4: 6
  "dilation_pattern": [1,1,1,2,2,2,4,4,4,8,8,8]  // ← v4: 9层
},
"training": {
  "lr": 1.5e-4,                   // ← v4: 8e-5
  "epochs": 300,                  // ← v4: 999
  "batch_size": 6,                // ← v4: 8
  "warmup_epochs": 8,             // ← v4: 5
  "patience": 40                  // ← v4: 30
},
"sampling": {
  "default_budget_fraction": 0.30 // ← v4: 0.35
}
```

MARS 使用最大模型：`mars_scale=lx`，约 160M 参数（冻结），hidden dim 1056，12 layers，12 heads。

---

## 7. 常用命令

### 启动 v6 训练 / 消融

```bash
cd /root/aigame/dannyyan/PriFold

# 全量（复现 v5）
bash symfold/run_train.sh symfold/config/v6_bprna.json

# 消融：关闭 Dice
bash symfold/run_train.sh symfold/config/v6_ablation_no_dice.json

# 消融：关闭 Ratio Penalty
bash symfold/run_train.sh symfold/config/v6_ablation_no_ratio_penalty.json
```

### 启动 v4 训练

```bash
cd /root/aigame/dannyyan/PriFold
bash symfold/run_train.sh symfold/config/v4_bprna.json
```

`run_train.sh` 会自动：

- 激活 `RNADiffFold_torch260`
- 设置 `PYTHONPATH`
- 后台启动训练
- 后台启动 GPU monitor
- 写日志、checkpoint、曲线、GPU JSONL

### 评估 v4

```bash
python symfold/eval_v4.py \
  --ckpt symfold/outputs/v4_bprna/model/best.pt \
  --out_json symfold/outputs/v4_bprna/eval_best.json
```

常用消融：

```bash
# projection 消融
python symfold/eval_v4.py --ckpt <ckpt> --projection_mode score
python symfold/eval_v4.py --ckpt <ckpt> --projection_mode hybrid
python symfold/eval_v4.py --ckpt <ckpt> --projection_mode sample

# budget 消融
python symfold/eval_v4.py --ckpt <ckpt> --default_budget_fraction 0.30
python symfold/eval_v4.py --ckpt <ckpt> --default_budget_fraction 0.35
python symfold/eval_v4.py --ckpt <ckpt> --default_budget_fraction 0.40
python symfold/eval_v4.py --ckpt <ckpt> --use_density_budget 1

# 多样本
python symfold/eval_v4.py --ckpt <ckpt> --num_samples_per_input 3
python symfold/eval_v4.py --ckpt <ckpt> --num_samples_per_input 5
```

### bad-case 分析

```bash
python symfold/analyze_cases.py \
  --ckpt symfold/outputs/v4_bprna/model/best.pt \
  --test_sets bprna-test \
  --out_dir symfold/outputs/v4_bprna/case_analysis
```

输出：

```text
case_analysis/<stage>_cases.csv
case_analysis/<stage>_worst_100.json
case_analysis/summary.json
```

---

## 8. 训练后必须看什么

不要只看 aggregate F1。v4 的目标是修 v3 的 bad cases。

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
5. projection 消融：`score / hybrid / sample`
6. budget 消融：`default_budget_fraction=0.30/0.35/0.40`，`use_density_budget=0/1`
7. `num_samples_per_input=1/3/5`

判断 v4 是否成功的优先标准：

```text
低 density 过预测下降
F1=0 case 减少
160+ 长度样本不再系统性错位
pred/gt ratio 更接近 1
```

---

## 9. 重要历史文档

优先看：

```text
symfold/v6/README.md                                # v6 使用指南 + 消融方案
docs/prifold_symflow_v5_improvements.md             # v5 改进报告（含具体计算示例）
docs/prifold_symflow_v4_walkthrough.md              # v4 完整 walkthrough
docs/prifold_symflow_v4_improvements.md             # v4 改进点
docs/prifold_symflow_v3_architecture_case_analysis.md # v3 架构与 bad-case 诊断
docs/data_distribution_report.md                    # 数据集统计
```

旧文档仅参考：

```text
docs/prifold_symflow_v1_postmortem.md
docs/prifold_symflow_v2_marsfix_architecture.md
docs/prifold_symflow_improvement_plan.md
docs/prifold_symflow_architecture.md
```

---

## 10. 数据和模型路径

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

MARS-LX：

```text
参数量约 160M
hidden_dim=1056
n_layers=12
n_heads=12
vocab_size=20
```

---

## 11. 已知注意事项

1. `projection` 是把模型 score / sampled candidate 后处理成合法 contact map 的步骤。
2. v4 默认 `projection_mode=score`，不依赖 sampled `x_t` 是否包含正确边。
3. score projection 必须配合 `score_threshold` 和 `default_budget_fraction`，否则会选太多 pair。
4. `density_guided` 默认关闭，因为 v3 实测它会降低 F1。
5. `run_train.sh` 支持 v1/v2/v3/v4/v5/v6 自动入口选择。
6. `train_v4.py` 复用 v3 loop，但有 v4 专用 `build_model()` 和 `evaluate()`。
7. v4 已通过 sanity test：shape、对称、loss backward、小步 overfit、projection、density hint 路径一致性。
8. **⚠️ 续训 Bug（已确认）**：`train_v4.py` 从 checkpoint 续训时，LR scheduler state 没有被恢复，导致 cosine schedule 从 step 0 重新 warmup。这在 v4_bprna（epoch 107）和 v4_rnastralign（epoch 43）均导致了训练崩溃。**下次续训前必须修复**：保存/加载 `scheduler.state_dict()`、`epoch`、`global_step`。

---

## 12. 官方 PriFold baseline

主线 PriFold checkpoint 指标（2025-05-25, H20）：

| 测试集 | Precision | Recall | F1 |
|---|---:|---:|---:|
| bprna-test | 0.7938 | 0.7623 | 0.7700 |
| rnastralign-test | 0.9742 | 0.9744 | 0.9738 |
| archiveii-test | 0.9102 | 0.9037 | 0.9043 |

主线仅作为 baseline；当前实验目标是改进生成式 SymFlow 路线。