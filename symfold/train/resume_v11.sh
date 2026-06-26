#!/bin/bash
export PATH="/root/aigame/dannyyan/miniconda3/bin:$PATH"
source activate RNADiffFold_torch260
export PYTHONPATH=/root/aigame/dannyyan/PriFold
cd /root/aigame/dannyyan/PriFold

CUDA_VISIBLE_DEVICES=0 python -u symfold/train/train_v11.py symfold/config/v11/v11_hardcase_oversample.json
