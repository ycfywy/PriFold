#!/bin/bash
# Train v12: Discrete (Bernoulli) Flow Matching + DiT (RoPE) + modular loss
# 说明: v12 已从连续 flow matching 改造为离散 flow matching。
#       旧的连续 flow checkpoint 与新结构语义不兼容，启动前会自动备份并清理。
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
source activate RNADiffFold_torch260
export PYTHONPATH=/root/aigame/dannyyan/PriFold

cd /root/aigame/dannyyan/PriFold

OUT_DIR=symfold/outputs/v12
BACKUP_DIR=symfold/outputs/v12_continuous_backup
LOG=symfold/logs/v12/v12.log

# 备份旧的连续 flow 结果 (仅首次)
if [ -d "$OUT_DIR" ] && [ ! -d "$BACKUP_DIR" ]; then
    echo "[run_v12] backing up continuous-flow results -> $BACKUP_DIR"
    cp -r "$OUT_DIR" "$BACKUP_DIR"
fi

# 清理不兼容的 checkpoint / history / 旧日志，重新开始离散 flow 训练
echo "[run_v12] cleaning incompatible checkpoints & history"
rm -f "$OUT_DIR/model/last.pt" "$OUT_DIR/model/best.pt" "$OUT_DIR/history.json"
rm -f "$LOG"

CUDA_VISIBLE_DEVICES=0 python symfold/train/train_v12.py symfold/config/v12/v12_flow_dit.json
