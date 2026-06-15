#!/usr/bin/env bash
# 给一个已经在跑的训练补挂 GPU monitor daemon。
#
# 用法：
#   bash symfold/train/run_gpu_monitor.sh <task_name> [pid] [interval]
#
# 例：
#   bash symfold/train/run_gpu_monitor.sh v2_bprna 70242 5
#
# 参数：
#   task_name  → outputs/<task>/gpu_stats.jsonl
#   pid        → 训练进程 PID（不传则只监控全卡，不绑定 stop-on-pid-death）
#   interval   → 采样间隔秒数（默认 5）

set -e
cd /root/aigame/dannyyan/PriFold

TASK_NAME="${1:?usage: bash symfold/train/run_gpu_monitor.sh <task_name> [pid] [interval]}"
TRAIN_PID="${2:-}"
GPU_INTERVAL="${3:-5}"
DEVICE_IDX="${DEVICE_IDX:-0}"

LOG_DIR="/root/aigame/dannyyan/PriFold/symfold/logs/${TASK_NAME}"
OUTPUT_DIR="/root/aigame/dannyyan/PriFold/symfold/outputs/${TASK_NAME}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
GPU_STATS_PATH="${OUTPUT_DIR}/gpu_stats.jsonl"

EXTRA_ARGS=""
if [ -n "$TRAIN_PID" ]; then
  EXTRA_ARGS="--pid $TRAIN_PID --stop-on-pid-death"
fi

setsid bash -c "
  export PATH=\"/root/aigame/dannyyan/miniconda3/bin:\$PATH\"
  source activate RNADiffFold_torch260 || true
  export PYTHONPATH=/root/aigame/dannyyan/PriFold
  exec python3 -u -m symfold.train.gpu_monitor daemon \\
    --out '$GPU_STATS_PATH' \\
    --device $DEVICE_IDX \\
    --interval $GPU_INTERVAL \\
    $EXTRA_ARGS
" < /dev/null > "${LOG_DIR}/${TASK_NAME}.gpu_monitor.log" 2>&1 &
MONITOR_PID=$!
echo $MONITOR_PID > "${LOG_DIR}/${TASK_NAME}.gpu_monitor.pid"

echo "GPU monitor launched"
echo "  task:        ${TASK_NAME}"
echo "  monitor pid: $MONITOR_PID"
echo "  target pid:  ${TRAIN_PID:-<not bound>}"
echo "  interval:    ${GPU_INTERVAL}s"
echo "  stats file:  ${GPU_STATS_PATH}"
echo "  log:         ${LOG_DIR}/${TASK_NAME}.gpu_monitor.log"
echo
echo "Tail:    tail -f ${GPU_STATS_PATH}"
echo "Summary: python -m symfold.analysis.show_gpu_stats ${GPU_STATS_PATH} --summary"
