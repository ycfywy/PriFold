# PriFold-SymFlow v2

> 主线 PriFold 数据 + MARS-LX 语言模型 + SymFold v5 风格的 DA-SE-DiT-v4 主干 + Bernoulli Discrete Flow Matching。
> **不接入 UFold**：所有空间先验来自 PriFold 的 `pos_bias`（碱基互补规则）+ MARS attention map（进化协变信号）。

## 训练环境（重要）

**必须使用 `RNADiffFold_torch260` 环境，不要用 `prifold`**。原因：

- `prifold` 环境 = torch 2.2.1+cu121，在 H20 上 `libcublasLt.so.12` 有除零硬件异常 bug（SIGFPE），训练几百到几千 step 后进程**静默消失**（无 traceback、无 OOM、bash 子壳同时消失），`dmesg` 里只留下 `traps: ... trap divide error in libcublasLt.so.12`。
- `RNADiffFold_torch260` 环境 = torch 2.6.0+cu124，已稳定跑过 SymFold v3-v5 全量训练。

```bash
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
source activate RNADiffFold_torch260
# run_train.sh 内部已自动激活该环境
```

`train_v2.py` / `eval_v2.py` 顶部已加：
- `faulthandler.enable()`：万一仍崩溃，能输出 Python traceback 到 stderr
- `allow_bf16/fp16_reduced_precision_reduction = False`：双重保险，禁掉 cuBLAS 的 bf16 reduce 路径

## AMP / 精度

- 默认 **bf16 autocast**（config 里 `"amp_dtype": "bf16"`），**显存比 fp32 节省 ~25%、速度持平**（H20 上 GEMM 已经被 TF32 加速过，bf16 收益主要在 activation 显存）；
- 在 `RNADiffFold_torch260` 环境下经 300 step 全 L 区间压力测试稳定，无 cuBLAS SIGFPE。
- 切换：将 config 的 `"amp_dtype"` 改为 `"fp32"`（关闭 autocast）或 `"fp16"`（自动启用 GradScaler）。
- 实测显存（B=8、各种 L 混合）：

| 模式 | step 0 (L=428) peak_alloc | 300 step final peak_alloc | 300 step final reserved |
|---|---:|---:|---:|
| fp32 | 39.5 GB | 47.5 GB | 83.3 GB |
| **bf16** | **31.4 GB** | **37.8 GB** | **58.2 GB** |

## v2_marsfix 修复点（2026-06-02）

首轮 `v2_bprna` 训练出现 `val F1≈0.28` 后平台期、test F1 下滑。对照 RNADiffFold/SymFold v4/v5 后，定位到 MARS-only 版本的几个实现问题并修复为 `v2_*_marsfix` 配置：

1. **Frozen MARS train/eval 不一致**：`model.train()` 会递归把 frozen MARS 打回 train mode，导致 dropout 参与 MARS hidden/attention；现已在 `_extract_mars()` 中强制 `self.extractor.eval()`。
2. **MARS hidden 只用最后一层太单薄**：参考 SymFold 的 multi-layer FM fusion，现提取 MARS layer `[3,6,9,12]` hidden，做 learnable weighted fusion + per-layer projection，再 outer concat 成 pair feature。
3. **补回显式序列条件**：参考 SymFold v5 保留 `seq_oh outer concat`（8ch），MARS-only / no-UFold 路线需要这个低层碱基身份先验。
4. **Triangle residual 修正**：`TriangleMultiplicativeUpdate` 现在只返回 delta，避免 `x = x + tri_update(norm(x))` 在 zero-init 时变成 `x + norm(x)`。
5. **Density 条件去泄漏**：`density_hint_dropout=1.0`，训练不再把 GT density 喂给 backbone，只保留 density head MSE 和推理时 density-guided sampling，避免 density head “抄答案”导致推理 mismatch。

## 关键改进 vs v1

| 项 | v1 | **v2_marsfix** |
|---|---|---|
| 主干 | 6 层标准 Axial DiT (`nn.MultiheadAttention` + GELU FFN) | **9 层 DA-SE-DiT** (Dilated Axial 1/2/4 + RoPE + QK-Norm + SwiGLU + AdaLN-Zero) |
| pair feature 来源 | MARS 最后一层 hidden 经 outer concat (128 ch) | **MARS 后 6 层 × 12 head = 72 个 attention map**（对称化 + APC + Conv 投影到 16 ch）+ MARS hidden 投影到 1D 后 outer concat (64 ch) |
| Triangle Update | ✗ | **✓ L6-L8 三层 AF2 风格三角乘法更新** |
| 输出精修 | 单层卷积 | **3 层 Conv 残差（zero-init）全分辨率精修** |
| 条件信号 | AdaLN(t) 单条件 | **AdaLN-Zero(time + MARS global + density)** |
| Density 闭环 | density 仅作 loss | **训练注入 GT density、推理用预测值 + density-guided rate damping** |
| 物理约束 loss | 权重为 0 | **stack=0.05, nc=0.02 已打开** |
| 采样 schedule | 均匀 Euler-CTMC | **Cosine τ-leap schedule** |
| 输入通道 | 145 | **89**（更精炼） |
| 训练设置 | bpRNA + RNAStrAlign 合并训 | **两个独立模型**（对齐 PriFold 主线 train.sh） |

## 文件结构

```
symfold/
├── v1/                       # 旧版（归档）
├── v2/
│   ├── __init__.py
│   ├── da_se_dit.py         # DA-SE-DiT-MARS 主干 (RMSNorm + RoPE + Triangle + SwiGLU)
│   ├── model.py             # PriFoldSymFlow_v2 (MARS extract + backbone + flow loss + sample)
│   └── discrete_flow.py     # Bernoulli flow + adaptive loss + cosine sampling + greedy projection
├── data.py                  # 共享：单数据集 + combined 模式
├── metrics.py               # 共享：上三角 contact P/R/F1/MCC
├── train_v2.py              # v2 训练入口（按 dataset_mode 选数据）
├── eval_v2.py               # v2 评估入口（按 dataset_mode 自动选 test 集）
├── run_train.sh             # 后台启动（自动识别 v1/v2 入口）
└── config/
    ├── v1/                  # 旧配置归档
    ├── v2_bprna_marsfix.json        # 模型 A：bpRNA TR0 训（当前推荐）
    ├── v2_rnastralign_marsfix.json  # 模型 B：RNAStrAlign tr 训（当前推荐）
    ├── v2_bprna.json                # 首轮 v2 对照配置（已暴露问题）
    └── v2_rnastralign.json          # 首轮 v2 对照配置
```

## 两个独立训练任务

| 任务 | 训练 | 验证 | 测试 | epochs | 配置 |
|---|---|---|---|---:|---|
| **A. v2_bprna_marsfix** | bpRNA TR0 (10807) | bpRNA VL0 (1299) | bpRNA TS0 (1303) | 120 | `config/v2_bprna_marsfix.json` |
| **B. v2_rnastralign_marsfix** | RNAStrAlign tr (20234) | RNAStrAlign vl (2493) | RNAStrAlign ts (2492) + ArchiveII (3845) | 60 | `config/v2_rnastralign_marsfix.json` |

bpRNA 训练集只有 RNAStrAlign 一半样本，所以 epoch 数翻倍以保 step 数相近。

## 启动训练（串行，避免 GPU 冲突）

```bash
cd /root/aigame/dannyyan/PriFold
# 任务 A：先训 bpRNA 模型
bash symfold/run_train.sh symfold/config/v2_bprna_marsfix.json
# 等任务 A 跑完后：
bash symfold/run_train.sh symfold/config/v2_rnastralign_marsfix.json
```

监控：

```bash
tail -f symfold/logs/v2_bprna_marsfix/v2_bprna.log
ls   symfold/outputs/v2_bprna_marsfix/training_curves.png       # 每 epoch 自动重绘
cat  symfold/outputs/v2_bprna_marsfix/test_eval_history.json    # 每 10 epoch 一条 test 评估
```

## 训练中周期性 test 评估

`v2_bprna_marsfix.json` / `v2_rnastralign_marsfix.json` 默认 `"test_eval_every": 10`，含义：每 10 epoch 训练完 + val 评估完后，**自动跑一次 test 集 sample + 投影 + 指标**，结果同时写入：

- `outputs/<task>/history.json`：每个 epoch entry 增加 `test_<stage>_{f1,precision,recall,mcc,n}` 字段
- `outputs/<task>/test_eval_history.json`：独立的时间序列文件，按 epoch 排列
- `outputs/<task>/training_curves.png`：自动新增第三排 panel（test F1 / test MCC）

测试集按 `dataset_mode` 自动选择：
- `bprna` → `bprna-test`
- `rnastralign` → `rnastralign-test, archiveii-test`

可选自定义：
- 显式覆盖：在 config `training` 段加 `"test_stages": "bprna-test,archiveii-test"`
- 关闭：设 `"test_eval_every": 0`

## GPU 实时监控（独立 daemon）

`run_train.sh` 启动训练时会**自动 fork 一个独立的 GPU monitor daemon**，每 5 秒通过 NVML 采集一次 GPU 状态，写到 `outputs/<task>/gpu_stats.jsonl`：
- 独立进程，训练崩了也不影响监控；监控崩了也不影响训练
- 训练 PID 退出后 daemon 自动退出（`--stop-on-pid-death`）
- 字段：`nvml_used_mb / util / temp / power / target_pid_used_mb / target_alive / gpu_procs`

```bash
# 实时跟随
tail -f symfold/outputs/v2_bprna_marsfix/gpu_stats.jsonl

# 表格形式
python -m symfold.show_gpu_stats symfold/outputs/v2_bprna_marsfix/gpu_stats.jsonl --tail 20
python -m symfold.show_gpu_stats symfold/outputs/v2_bprna_marsfix/gpu_stats.jsonl --summary

# 不依赖训练，独立看一次：
python -m symfold.gpu_monitor once --device 0

# 给已在跑的训练补挂 monitor
bash symfold/run_gpu_monitor.sh v2_bprna_marsfix <train_pid> 5

# 自定义采样间隔（秒）
GPU_INTERVAL=2 bash symfold/run_train.sh symfold/config/v2_bprna_marsfix.json
```

## 评估

```bash
# A 模型 → bpRNA-test
python symfold/eval_v2.py \
    --ckpt symfold/outputs/v2_bprna_marsfix/model/best.pt \
    --out_json symfold/outputs/v2_bprna_marsfix/eval_best.json

# B 模型 → rnastralign-test + archiveii-test
python symfold/eval_v2.py \
    --ckpt symfold/outputs/v2_rnastralign_marsfix/model/best.pt \
    --out_json symfold/outputs/v2_rnastralign_marsfix/eval_best.json
```

`eval_v2.py` 会按 ckpt 内 `dataset_mode` 自动选对应测试集。也可以用 `--test_sets bprna-test,rnastralign-test,archiveii-test` 显式覆盖。

## 设计依据

详见 `docs/symfold_v5_intro_and_migration.md` 与 `docs/prifold_symflow_v1_postmortem.md`。
