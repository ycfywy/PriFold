#!/bin/bash
# PriFold v9 DDP Training Launcher (2x H20 GPUs)
#
# Usage:
#   bash symfold/train/run_train_v9_ddp.sh [config_path]
#   Default config: symfold/config/v9/v9_ddp.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-symfold/config/v9/v9_ddp.json}"

# Parse task_name from config
TASK_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG'))['task_name'])")
LOG_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG'))['paths']['log_dir'])")

echo "============================================"
echo "PriFold v9 DDP Training"
echo "Config: $CONFIG"
echo "Task: $TASK_NAME"
echo "GPUs: 2x H20"
echo "============================================"

# Create log directory
mkdir -p "$LOG_DIR"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate RNADiffFold_torch260

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# Launch DDP with torchrun
echo "[$(date)] Starting DDP training on 2 GPUs..."

setsid torchrun \
    --nproc_per_node=2 \
    --standalone \
    --nnodes=1 \
    symfold/train/train_v9_ddp.py "$CONFIG" \
    > "$LOG_DIR/${TASK_NAME}.stdout.log" 2> "$LOG_DIR/${TASK_NAME}.stderr.log" &

TRAIN_PID=$!
echo "$TRAIN_PID" > "$LOG_DIR/${TASK_NAME}.pid"
echo "[$(date)] Training started PID=$TRAIN_PID"
echo "Logs: $LOG_DIR/${TASK_NAME}.stdout.log"
echo "Monitor: tail -f $LOG_DIR/${TASK_NAME}.stdout.log"

# Start GPU monitor
python3 -c "
import subprocess, sys, os, json, time, signal

log_path = '$LOG_DIR/${TASK_NAME}.gpu_monitor.log'
out_path = json.load(open('$CONFIG'))['paths']['output_dir'] + '/gpu_stats.jsonl'
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(log_path, 'w') as logf:
    logf.write(f'[gpu_monitor] daemon started: out={out_path} devices=0,1 interval=5.0s target_pid=$TRAIN_PID\n')

target_pid = $TRAIN_PID
interval = 5.0

def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except:
        return False

with open(out_path, 'a') as fout:
    while pid_alive(target_pid):
        try:
            import subprocess as sp
            result = sp.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                ts = time.time()
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) == 5:
                        entry = {'ts': ts, 'gpu': int(parts[0]), 'util': float(parts[1]),
                                'mem_used_mb': float(parts[2]), 'mem_total_mb': float(parts[3]),
                                'temp_c': float(parts[4])}
                        fout.write(json.dumps(entry) + '\n')
                fout.flush()
        except:
            pass
        time.sleep(interval)

with open(log_path, 'a') as logf:
    logf.write(f'[gpu_monitor] target process {target_pid} ended, stopping monitor.\n')
" &

GPU_MON_PID=$!
echo "$GPU_MON_PID" > "$LOG_DIR/${TASK_NAME}.gpu_monitor.pid"
echo "[$(date)] GPU monitor started PID=$GPU_MON_PID"
echo "============================================"
echo "Done. Training running in background."
