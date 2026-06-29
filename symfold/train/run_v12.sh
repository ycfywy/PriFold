#!/bin/bash
# Train v12: Discrete (Bernoulli) Flow Matching + DiT (RoPE) + modular loss
# 说明: v12 已从连续 flow matching 改造为离散 flow matching，并采用 v6-style patch-space backbone。
#       旧 checkpoint 与当前结构语义不兼容；需要从头训练时显式 FRESH_RUN=1。
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

# 默认不删除任何 checkpoint/history；只有显式 FRESH_RUN=1 才重新开始。
if [ "${FRESH_RUN:-0}" = "1" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    FRESH_BACKUP="${OUT_DIR}_fresh_backup_${TS}"
    if [ -d "$OUT_DIR" ]; then
        echo "[run_v12] FRESH_RUN=1, backing up current results -> $FRESH_BACKUP"
        cp -r "$OUT_DIR" "$FRESH_BACKUP"
    fi
    echo "[run_v12] cleaning checkpoints/history/log for fresh run"
    rm -f "$OUT_DIR/model/last.pt" "$OUT_DIR/model/best.pt" "$OUT_DIR/history.json"
    rm -f "$LOG"
else
    echo "[run_v12] resume mode: keeping checkpoint/history/log"
fi

CUDA_VISIBLE_DEVICES=0 python symfold/train/train_v12.py symfold/config/v12/v12_flow_dit.json
