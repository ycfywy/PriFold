# PriFold 项目规范

## 目录结构约定

```
PriFold/
├── docs/       # 所有文档统一存放
├── logs/       # 训练/测试日志
├── outputs/    # 训练/测试输出（模型checkpoint、预测结果等）
├── model/      # 预训练模型（不可修改）
├── data/       # 数据集（不可修改）
└── ...
```

## Logs 规范

每次启动训练或测试，日志存放在 `logs/` 下，以 **时间+任务名** 作为子文件夹：

```
logs/
├── 20250525_1330_bprna_train/
│   ├── train.log
│   └── eval.log
├── 20250525_1400_bprna_inference/
│   └── inference.log
├── 20250525_1430_rnastralign_inference/
│   └── inference.log
└── ...
```

### 命名格式

```
{YYYYMMDD}_{HHMM}_{任务名}/
```

- 时间：启动时的本地时间
- 任务名：`{dataset}_{train|inference|eval}`

示例：
- `20250525_1330_bprna_train`
- `20250525_1400_archiveii_inference`
- `20250526_0900_rnastralign_train`

### 日志内容

- `train.log`: 训练过程输出（loss、lr、epoch 等）
- `inference.log`: 推理结果（precision、recall、F1）
- `eval.log`: 验证集评估结果

## Outputs 规范

模型输出、预测结果等存放在 `outputs/`，同样以 **时间+任务名** 作为子文件夹：

```
outputs/
├── 20250525_1330_bprna_train/
│   ├── best_model.pth
│   ├── checkpoint_epoch_50.pth
│   └── config.yaml
├── 20250525_1400_bprna_inference/
│   └── predictions.csv
└── ...
```

### 命名格式

与 logs 保持一致：`{YYYYMMDD}_{HHMM}_{任务名}/`

### 输出内容

- 训练任务：模型 checkpoint、训练配置副本
- 推理任务：预测结果文件

## Docs 规范

所有文档统一存放在 `docs/` 目录：

```
docs/
├── project_convention.md   # 本文档（项目规范）
├── experiment_notes.md     # 实验记录
└── ...
```

## 启动脚本模板

### 训练

```bash
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
source activate prifold
cd /root/aigame/dannyyan/PriFold

TIMESTAMP=$(date +%Y%m%d_%H%M)
TASK="bprna_train"
LOG_DIR="logs/${TIMESTAMP}_${TASK}"
OUT_DIR="outputs/${TIMESTAMP}_${TASK}"
mkdir -p $LOG_DIR $OUT_DIR

accelerate launch --config_file config_bf16.yaml ./train.py \
    --mode bprna \
    --batch_size 1 \
    --lr 1e-4 \
    --select 0.1 --replace 0.3 \
    --pretrained_lm_dir ./model \
    --data_dir ./data \
    --save True \
    2>&1 | tee ${LOG_DIR}/train.log
```

### 推理

```bash
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
source activate prifold
cd /root/aigame/dannyyan/PriFold

TIMESTAMP=$(date +%Y%m%d_%H%M)
TASK="bprna_inference"
LOG_DIR="logs/${TIMESTAMP}_${TASK}"
OUT_DIR="outputs/${TIMESTAMP}_${TASK}"
mkdir -p $LOG_DIR $OUT_DIR

python inference.py --mode bprna-test --model_scale lx \
    --batch_size 1 --scale 0.01 \
    --model_path ./model/ss_model_bprna.pth \
    --pretrained_lm_dir ./model \
    --data_dir ./data \
    2>&1 | tee ${LOG_DIR}/inference.log
```
