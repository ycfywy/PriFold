# CLAUDE.md — PriFold 项目指导

## 项目概述

**PriFold: Biological Priors Improve RNA Secondary Structure Predictions** (AAAI 2025)

基于深度学习的 RNA 二级结构预测方法。利用生物学先验知识（全局配对特征 + 进化协变信息）提升预测精度。

## 核心创新

1. **配对概率位置偏置 (pos_bias)**: 根据碱基配对规则 (A-U:3, G-C:6, G-U:1) 生成先验配对得分矩阵，作为注意力偏置输入模型
2. **RNA 协变数据增强 (Augmentation)**: 根据配对碱基的自然频率分布，随机替换已配对碱基对为等价配对（模拟进化协变）

## 架构流程

```
RNA序列 → EsmTokenizer → MARS语言模型(LLaMA2架构) → 序列特征
                                                       ↓
                                         EmbedSequence2Matrix (1D→2D)
                                                       ↓
                                         + 碱基配对位置偏置 (pos_bias)
                                                       ↓
                                         RNAformer Stack (Axial Attention)
                                                       ↓
                                         配对概率矩阵 (L×L) → Sigmoid → 二级结构
```

## 项目结构

```
PriFold/
├── train.py              # 训练主脚本（Accelerate 多GPU分布式训练）
├── inference.py          # 推理评估脚本
├── train.sh              # 训练启动命令
├── inference.sh          # 推理启动命令
├── config_bf16.yaml      # Accelerate 分布式配置（4 GPU, BF16）
├── requirements.txt      # Python 依赖
├── vocab_esm_mars.txt    # RNA 词表（20 tokens: A/T/G/C/N + IUPAC模糊编码）
├── config/               # T5模型配置（不同规模: s/m/l/demo）
├── symfold/              # ★ PriFold-SymFlow 实验分支（MARS + concat map + DiT + Flow Matching）
├── prifold/              # RNA 语言模型实现
│   ├── llama2.py         # ★ MARS 模型核心（LLaMA2架构，支持MLM/GLM预训练）
│   ├── esm2.py           # ESM2 蛋白质/RNA语言模型
│   ├── gpt2.py           # GPT-2 自回归模型
│   ├── llama2_t5.py      # LLaMA2-T5 Encoder-Decoder 变体
│   ├── t5_model.py       # T5 Encoder-Decoder
│   ├── modules/resnet.py # ResNet 2D卷积模块
│   └── utils/            # 注意力mask、化学工具函数
└── utils/                # 训练工具 & RNAformer 模型
    ├── lm.py             # ★ 加载预训练 MARS 模型
    ├── tools.py          # ★ 数据加载 & get_posbias() 位置偏置计算
    ├── predictor.py      # ★ SSDataset + Augmentation 数据增强
    ├── configuration.py  # YAML 配置解析
    ├── boundaries.py     # Span 边界解码器
    ├── embedding.py      # 序列到矩阵嵌入
    └── RNAformer/        # ★ RNA 二级结构预测核心模型
        ├── model/
        │   ├── Riboformer_outfirst.py  # 主模型 RiboFormer
        │   ├── RNAformer_block.py      # 轴注意力块
        │   └── RNAformer_stack.py      # 多层堆叠
        ├── module/
        │   ├── axial_attention.py      # FlashAttention2D + TriangleAttention
        │   ├── axial_dropout.py        # 轴Dropout
        │   ├── embedding.py            # 嵌入层
        │   └── feed_forward.py         # FFN（含卷积FFN）
        └── models/*.yml                # 模型配置文件
```

## 环境配置

conda 路径: `/root/aigame/dannyyan/miniconda3`

```bash
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
conda create -n prifold python=3.10 -y
conda activate prifold

# 先装 PyTorch（需指定 CUDA 12.1 源）
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121

# 再装其余依赖
pip install accelerate==1.1.0 einops==0.8.1 rotary-embedding-torch==0.5.3 \
    transformers==4.46.2 scikit-learn==1.5.2 wandb==0.18.5 pandas==2.2.3 \
    scipy==1.15.2 numpy==1.24.3 PyYAML==6.0.2 tqdm==4.67.1 ninja==1.11.1.3

# flash-attn 需要编译（耗时约5分钟）
pip install flash-attn --no-build-isolation
```

### 当前已配置环境信息

- **conda 环境名**: `prifold`
- **环境路径**: `/root/aigame/dannyyan/miniconda3/envs/prifold`
- **Python**: 3.10.20
- **PyTorch**: 2.2.1+cu121
- **CUDA**: 可用
- **GPU**: NVIDIA H20
- **Flash-Attn**: 2.8.3（编译安装，兼容 2.5.8 接口）
- **Transformers**: 4.46.2
- **Accelerate**: 1.1.0

### 激活环境命令

```bash
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
source activate prifold
```

## 目录规范（重要）

详见 `docs/project_convention.md`

- **`docs/`**: 所有文档统一存放
- **`examples/`**: 所有代码示例统一存放
- **`logs/`**: 每次训练/测试的日志，按 `{YYYYMMDD}_{HHMM}_{任务名}/` 分文件夹
- **`outputs/`**: 每次训练/测试的输出（checkpoint、预测结果），同样按时间+任务分文件夹

### 命名格式示例

```
logs/20250525_1330_bprna_train/train.log
logs/20250525_1400_archiveii_inference/inference.log
outputs/20250525_1330_bprna_train/best_model.pth
```

### 每次启动任务时必须

```bash
TIMESTAMP=$(date +%Y%m%d_%H%M)
TASK="bprna_train"  # 格式: {dataset}_{train|inference|eval}
LOG_DIR="logs/${TIMESTAMP}_${TASK}"
OUT_DIR="outputs/${TIMESTAMP}_${TASK}"
mkdir -p $LOG_DIR $OUT_DIR
# 然后用 tee 将输出同时写入日志
... 2>&1 | tee ${LOG_DIR}/train.log
```

## 训练

```bash
# 需要先下载预训练模型和数据到 ./model 和 ./data 目录
# 从 https://huggingface.co/yfish/PriFold 下载
./train.sh
```

关键参数（train.py）:
- `--mode`: 数据集 (bprna/rnastralign)
- `--scale`: 位置偏置缩放因子
- `--select/--replace`: 数据增强参数（选择率/替换率）
- `--model_scale`: 语言模型规模 (6m/25m/85m/160m/lx)
- `--pretrained_lm_dir`: MARS 预训练模型路径
- `--data_dir`: 数据集路径

## 推理

```bash
./inference.sh
```

测试集: bprna-test, rnastralign-test, archiveii-test
阈值: 0.45
指标: Precision, Recall, F1

## PriFold-SymFlow 实验分支

已在 `symfold/` 下实现首版 PriFold × SymFold 混合模型：

```text
PriFold 数据 CSV/NPY
→ MARS-LX frozen encoder
→ hidden_states 去掉特殊 token
→ Linear + outer concat map
→ x_t embedding + seq_2d + pos_bias
→ Axial DiT
→ Bernoulli Flow Matching
→ CTMC sampling + greedy projection
```

关键文件：
- `symfold/data.py`: PriFold CSV/NPY 数据集、padding、`seq_oh`、`pos_bias`。
- `symfold/model.py`: MARS encoder + embedding concat map + DiT + flow loss/sample。
- `symfold/dit.py`: 简化版 SymFold-style Axial DiT。
- `symfold/discrete_flow.py`: Bernoulli Flow Matching、CTMC rates、projection、loss。
- `symfold/train.py`: 训练入口，记录 log、heartbeat、checkpoint、history。
- `symfold/eval.py`: 独立评估入口。
- `symfold/config/prifold_symflow_v0.json`: 首版 smoke 配置。

当前 smoke run：`20260526_1508_prifold_symflow_v0`
- 训练数据限制：`max_train_samples=512`，`max_val_samples=128`。
- 训练 5 epoch 已完成。
- best val F1: `0.2464`。
- checkpoint: `symfold/outputs/20260526_1508_prifold_symflow_v0/model/best.pt`。
- 详细记录见 `docs/prifold_symflow_implementation_report.md`。

启动命令：

```bash
cd /root/aigame/dannyyan/PriFold
bash symfold/run_train.sh symfold/config/prifold_symflow_v0.json
```

注意：这是链路验证版本，不是完整训练结果；完整训练需去掉 sample limit 并增加 epoch / full eval。

## 模型规模

MARS 语言模型支持多种规模:
- 6M, 25M, 85M, 160M, LX

RNAformer 默认配置:
- model_dim: 256, n_layers: 4, num_head: 4, max_len: 490

## 数据集

- **bpRNA**: RNA 二级结构基准数据集
- **RNAStrAlign**: RNA 结构比对数据集
- **ArchiveII**: 经典 RNA 结构测试集

数据格式: 每条样本包含 RNA 序列 + 接触图(contact map)

## 推理验证结果 (2025-05-25, NVIDIA H20)

| 测试集 | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| bprna-test | 0.7938 | 0.7623 | 0.7700 |
| rnastralign-test | 0.9742 | 0.9744 | 0.9738 |
| archiveii-test | 0.9102 | 0.9037 | 0.9043 |

## 已修复的问题

1. **`utils/lm.py` 缺少 `lx` model_scale**：原代码只支持 6m/25m/85m/160m，但 inference.sh 使用 `lx`。已添加 `lx` → `mars_run-encoder-mars-lx-train-val-d0.15-2023_10_05_22_03_21` 映射。
2. **数据目录不完整**：初次解压 `data.tar.gz` 可能丢失部分子目录（如 `RNAStrAlign/5S_rRNA_database`）。需要用 `tar -xzf data.tar.gz` 完整解压覆盖。
3. **MARS checkpoint 软链接损坏**：`model/mars_run-encoder-mars-lx-train-val-d0.15-2023_10_05_22_03_21/ckpt_175000.pt` 指向的 HuggingFace cache blob 不存在，导致 SymFlow 首次训练 `FileNotFoundError`。已将 MARS-LX 下载到工作区 `.hf_prifold/model/`，并在 `symfold/config/prifold_symflow_v0.json` 中把 `pretrained_lm_dir` 指向该目录。
4. **`chmod +x symfold/run_train.sh` 曾被拒绝**：后续统一用 `bash symfold/run_train.sh ...` 启动，不依赖脚本可执行权限。

## 数据与模型路径

```
./model/
├── mars_run-encoder-mars-lx-train-val-d0.15-2023_10_05_22_03_21/
│   └── ckpt_175000.pt          # MARS-LX 预训练语言模型 (160M params, dim=1056, 12层)
├── ss_model_bprna.pth           # bpRNA 微调的结构预测模型
└── ss_model_rnastralign.pth     # RNAStrAlign 微调的结构预测模型

./data/
├── bprna/       (13426 files)   # bpRNA 数据集
├── RNAStrAlign/ (26079 files)   # RNAStrAlign 数据集
└── archiveII/   (7935 files)    # ArchiveII 数据集
```

数据来源: `data.tar.gz`（36MB，解压后含全部 npy 文件）

补充：当前 `./model/` 下的 MARS-LX 与两个结构预测 checkpoint 已替换为真实文件，不再依赖 HuggingFace cache 软链接。

## 开发注意事项

1. 训练使用 Accelerate 框架，默认4 GPU + BF16 混合精度
2. 需要 Flash Attention 2 支持（需 Ampere 及以上 GPU）
3. 模型推理阈值默认 0.45，可调
4. 数据增强是可选的（通过 --select/--replace 控制）
5. 预训练模型和数据需要从 HuggingFace 手动下载
6. 碱基配对规则硬编码在 `utils/tools.py` 的 `get_posbias()` 函数中
7. MARS-LX 模型: dim=1056, n_layers=12, n_heads=12, vocab_size=20, dropout=0.15
