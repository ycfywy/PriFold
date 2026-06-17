#!/bin/bash
# v10 DensityNet-Ultra DDP Training (2x H20)
# Key: Partial MARS unfreeze (last 2 layers) + curriculum sampling

set -e

cd /root/aigame/dannyyan/PriFold
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
source activate RNADiffFold_torch260
export PYTHONPATH=/root/aigame/dannyyan/PriFold
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG=${1:-symfold/config/v10/v10_ddp.json}

echo "========================================"
echo "v10 DensityNet-Ultra DDP Training"
echo "Config: $CONFIG"
echo "GPUs: 2x H20 (DDP)"
echo "Key: Partial MARS unfreeze + Curriculum"
echo "========================================"

# Create output dirs
mkdir -p symfold/logs/v10_ddp
mkdir -p symfold/outputs/v10_ddp/model

# Launch DDP
torchrun --nproc_per_node=2 --standalone --nnodes=1 \
  symfold/train/train_v10_ddp.py "$CONFIG" \
  2>&1 | tee symfold/logs/v10_ddp/v10_ddp.stdout.log
