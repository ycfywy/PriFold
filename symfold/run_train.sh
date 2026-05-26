#!/usr/bin/env bash
set -e
cd /root/aigame/dannyyan/PriFold
CONFIG="${1:-/root/aigame/dannyyan/PriFold/symfold/config/prifold_symflow_v0.json}"
TASK_NAME=$(python3 - <<PY
import json
print(json.load(open('$CONFIG'))['task_name'])
PY
)
LOG_DIR="/root/aigame/dannyyan/PriFold/symfold/logs/${TASK_NAME}"
mkdir -p "$LOG_DIR"
setsid bash -c "
  export PATH=\"/root/aigame/dannyyan/miniconda3/bin:\$PATH\"
  source activate prifold || true
  export PYTHONPATH=/root/aigame/dannyyan/PriFold
  exec python3 -u /root/aigame/dannyyan/PriFold/symfold/train.py \"$CONFIG\"
" < /dev/null > "${LOG_DIR}/${TASK_NAME}.stdout.log" 2> "${LOG_DIR}/${TASK_NAME}.stderr.log" &
echo $! > "${LOG_DIR}/${TASK_NAME}.pid"
echo "Launched PriFold-SymFlow training"
echo "  task:   ${TASK_NAME}"
echo "  pid:    $(cat ${LOG_DIR}/${TASK_NAME}.pid)"
echo "  stdout: ${LOG_DIR}/${TASK_NAME}.stdout.log"
echo "  stderr: ${LOG_DIR}/${TASK_NAME}.stderr.log"
