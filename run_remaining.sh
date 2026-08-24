#!/bin/bash
cd /root/efs/dannyyan/PriFold
export CUDA_VISIBLE_DEVICES=1
source activate prifold 2>/dev/null

python threshold_scan_fast.py --mode bprna --max_len 490 --model_scale 160m --pretrained_lm_dir ./model --data_dir ./data > scan_fast_bprna_lt490.log 2>&1
echo "done bprna lt490"

python threshold_scan_fast.py --mode bprna --max_len 512 --model_scale 160m --pretrained_lm_dir ./model --data_dir ./data > scan_fast_bprna_lt512.log 2>&1
echo "done bprna lt512"

python threshold_scan_fast.py --mode rnastralign --max_len 490 --model_scale 160m --pretrained_lm_dir ./model --data_dir ./data > scan_fast_rnastralign_lt490.log 2>&1
echo "done rnastralign lt490"

python threshold_scan_fast.py --mode rnastralign --max_len 512 --model_scale 160m --pretrained_lm_dir ./model --data_dir ./data > scan_fast_rnastralign_lt512.log 2>&1
echo "done rnastralign lt512"

python threshold_scan_fast.py --mode archiveii --max_len 490 --model_scale 160m --pretrained_lm_dir ./model --data_dir ./data > scan_fast_archiveii_lt490.log 2>&1
echo "done archiveii lt490"

python threshold_scan_fast.py --mode archiveii --max_len 512 --model_scale 160m --pretrained_lm_dir ./model --data_dir ./data > scan_fast_archiveii_lt512.log 2>&1
echo "done archiveii lt512"

echo "ALL DONE"
