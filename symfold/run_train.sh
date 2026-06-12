#!/usr/bin/env bash
# 启动 PriFold-SymFlow v1/v2/v3/v4 训练 + GPU 监控 daemon（后台 setsid 运行）。
#
# 用法：
#   bash symfold/run_train.sh symfold/config/v4_bprna.json
#   bash symfold/run_train.sh symfold/config/v3_bprna.json
#   bash symfold/run_train.sh symfold/config/v2_bprna_marsfix.json
#   bash symfold/run_train.sh symfold/config/v1/prifold_symflow_v1_full.json   # 也兼容 v1
#
# 行为：
#   1) 启动训练进程（独立进程组）
#   2) 启动 GPU monitor daemon（每 5s 采样一次，写到 outputs/<task>/gpu_stats.jsonl）
#   3) daemon 设置了 --stop-on-pid-death，训练结束自动退出
#
# 监控查看：
#   tail -f symfold/outputs/<task>/gpu_stats.jsonl
#   python -m symfold.show_gpu_stats symfold/outputs/<task>/gpu_stats.jsonl --tail 20
#   python -m symfold.show_gpu_stats symfold/outputs/<task>/gpu_stats.jsonl --summary

set -e
cd /root/aigame/dannyyan/PriFold
CONFIG="${1:-/root/aigame/dannyyan/PriFold/symfold/config/v2_rnastralign.json}"

# 自动判断 v1/v2/v3/v4 入口；同时拿 output_dir 用于放 gpu_stats.jsonl
read -r TASK_NAME ENTRY OUTPUT_DIR DEVICE_IDX <<< "$(python3 - <<PY
import json
cfg = json.load(open("$CONFIG"))
task = cfg['task_name']
out  = cfg.get('paths', {}).get('output_dir', f'/root/aigame/dannyyan/PriFold/symfold/outputs/{task}')
dev  = cfg.get('device', 'cuda:0')
dev_idx = 0
if isinstance(dev, str) and dev.startswith('cuda:'):
    try:
        dev_idx = int(dev.split(':')[1])
    except Exception:
        dev_idx = 0
version = str(cfg.get('model', {}).get('version', '')).lower()
if 'dataset_mode' not in cfg.get('training', {}):
    entry = 'symfold/v1/train.py'
elif version == 'v7' or task.startswith('v7_'):
    entry = 'symfold/train_v7.py'
elif version == 'v6' or task.startswith('v6_'):
    entry = 'symfold/train_v6.py'
elif version == 'v5' or task.startswith('v5_'):
    entry = 'symfold/train_v5.py'
elif version == 'v4' or task.startswith('v4_'):
    entry = 'symfold/train_v4.py'
elif version == 'v3' or task.startswith('v3_'):
    entry = 'symfold/train_v3.py'
else:
    entry = 'symfold/train_v2.py'
print(task, entry, out, dev_idx)
PY
)"

LOG_DIR="/root/aigame/dannyyan/PriFold/symfold/logs/${TASK_NAME}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# ---- 1) 启动训练 ----
setsid bash -c "
  export PATH=\"/root/aigame/dannyyan/miniconda3/bin:\$PATH\"
  source activate RNADiffFold_torch260 || true
  export PYTHONPATH=/root/aigame/dannyyan/PriFold
  exec python3 -u /root/aigame/dannyyan/PriFold/${ENTRY} \"$CONFIG\"
" < /dev/null > "${LOG_DIR}/${TASK_NAME}.stdout.log" 2> "${LOG_DIR}/${TASK_NAME}.stderr.log" &
TRAIN_PID=$!
echo $TRAIN_PID > "${LOG_DIR}/${TASK_NAME}.pid"

# ---- 2) 启动 GPU monitor daemon（独立进程，跟随训练进程退出）----
# 等 1.5s 让训练 PID 稳定再启动 monitor，避免误判 dead
GPU_STATS_PATH="${OUTPUT_DIR}/gpu_stats.jsonl"
GPU_INTERVAL="${GPU_INTERVAL:-5}"  # 可通过环境变量覆盖
setsid bash -c "
  sleep 1.5
  export PATH=\"/root/aigame/dannyyan/miniconda3/bin:\$PATH\"
  source activate RNADiffFold_torch260 || true
  export PYTHONPATH=/root/aigame/dannyyan/PriFold
  exec python3 -u -m symfold.gpu_monitor daemon \\
    --out '$GPU_STATS_PATH' \\
    --device $DEVICE_IDX \\
    --interval $GPU_INTERVAL \\
    --pid $TRAIN_PID \\
    --stop-on-pid-death
" < /dev/null > "${LOG_DIR}/${TASK_NAME}.gpu_monitor.log" 2>&1 &
MONITOR_PID=$!
echo $MONITOR_PID > "${LOG_DIR}/${TASK_NAME}.gpu_monitor.pid"

echo "Launched: $ENTRY $CONFIG"
echo "  task:           ${TASK_NAME}"
echo "  train pid:      $TRAIN_PID"
echo "  monitor pid:    $MONITOR_PID  (interval=${GPU_INTERVAL}s)"
echo "  stdout:         ${LOG_DIR}/${TASK_NAME}.stdout.log"
echo "  stderr:         ${LOG_DIR}/${TASK_NAME}.stderr.log"
echo "  gpu stats:      ${GPU_STATS_PATH}"
echo "  monitor log:    ${LOG_DIR}/${TASK_NAME}.gpu_monitor.log"
echo
echo "Tail GPU stats:   tail -f ${GPU_STATS_PATH}"
echo "Summary:          python -m symfold.show_gpu_stats ${GPU_STATS_PATH} --summary"
