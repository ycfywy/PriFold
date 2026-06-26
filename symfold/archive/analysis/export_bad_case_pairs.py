# -*- coding: utf-8 -*-
"""导出 v7 bad cases 的详细配对信息。

对每个 bad case (F1 < 0.3) 输出：
- RNA 名称、长度、序列
- Ground Truth 的所有配对位置 (i, j) 以及对应碱基 (A-U, G-C 等)
- 模型预测错误的 FP pairs 和漏掉的 FN pairs

输出 JSON 文件，每条记录包含：
{
  "name": "bpRNA_xxx",
  "length": 150,
  "seq": "AUGC...",
  "f1": 0.12,
  "stage": "test",
  "gt_pairs": [{"i": 3, "j": 45, "base_i": "A", "base_j": "U", "type": "AU"}, ...],
  "pred_pairs": [...],
  "fp_pairs": [...],   # 预测了但 GT 没有的
  "fn_pairs": [...],   # GT 有但模型没预测的
  "tp_pairs": [...]    # 预测正确的
}

Usage:
  python symfold/analysis/export_bad_case_pairs.py \
    --ckpt symfold/outputs/v7_full/model/best.pt \
    --config symfold/config/v7/v7_full.json \
    --out symfold/outputs/v7_full/comprehensive_analysis/bad_case_pairs.json \
    --threshold 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_loader
from symfold.train.train_v7 import build_model, load_config, move_to_device


BASE_MAP = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}


def pair_type(base_i, base_j):
    """Return pair type string like 'AU', 'GC', etc."""
    # Convert T→U for display
    bi = base_i.replace('T', 'U')
    bj = base_j.replace('T', 'U')
    return f"{bi}{bj}"


def is_canonical(base_i, base_j):
    """Check if pair is canonical (AU/UA/GC/CG/GU/UG)."""
    bi = base_i.replace('T', 'U')
    bj = base_j.replace('T', 'U')
    canonical = {'AU', 'UA', 'GC', 'CG', 'GU', 'UG'}
    return f"{bi}{bj}" in canonical


def extract_pairs(contact_map, seq, length):
    """Extract all pairs from contact map with base info."""
    pairs = []
    for i in range(length):
        for j in range(i + 3, length):
            if contact_map[i, j] > 0.5:
                base_i = seq[i] if i < len(seq) else '?'
                base_j = seq[j] if j < len(seq) else '?'
                pairs.append({
                    "i": int(i),
                    "j": int(j),
                    "base_i": base_i.replace('T', 'U'),
                    "base_j": base_j.replace('T', 'U'),
                    "type": pair_type(base_i, base_j),
                    "canonical": is_canonical(base_i, base_j),
                    "distance": int(j - i),
                })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='symfold/outputs/v7_full/model/best.pt')
    parser.add_argument('--config', default='symfold/config/v7/v7_full.json')
    parser.add_argument('--out', default='symfold/outputs/v7_full/comprehensive_analysis/bad_case_pairs.json')
    parser.add_argument('--threshold', type=float, default=0.3, help='F1 threshold for bad cases')
    args = parser.parse_args()

    print('Loading config and model...')
    cfg = load_config(args.config)
    device = torch.device(cfg.get('device', 'cuda:0'))

    class Args:
        pass
    lm_args = Args()
    lm_args.pretrained_lm_dir = cfg['paths'].get('pretrained_lm_dir', str(ROOT / 'model'))
    lm_args.model_scale = cfg['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)

    model = build_model(cfg, extractor)
    ckpt = torch.load(args.ckpt, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    print('Model loaded.')

    # Load data
    import os
    import pandas as pd
    data_dir = cfg['paths']['data_dir']
    bprna_csv = os.path.join(data_dir, 'bprna', 'bpRNA.csv')
    df = pd.read_csv(bprna_csv)
    seq_map = dict(zip(df['file_name'].astype(str), df['seq'].astype(str)))

    # Use larger batch_size for GPU utilization, extract per-sample after
    analysis_cfg = json.loads(json.dumps(cfg))
    analysis_cfg['training']['batch_size'] = 12
    analysis_cfg['training']['augmentation'] = {'enabled': False}

    scfg = cfg.get('sampling', {})
    amp_name = str(cfg.get('training', {}).get('amp_dtype', 'fp32')).lower()
    amp_on = amp_name in ('bf16', 'bfloat16')
    amp_dtype = torch.bfloat16

    stages = {'train': 'bprna-train', 'val': 'bprna-val', 'test': 'bprna-test'}
    all_bad_cases = []

    for stage_name, dataset_name in stages.items():
        print(f'\nProcessing {stage_name}...')
        loader = build_loader(dataset_name, analysis_cfg, tokenizer, shuffle=False)
        print(f'  Samples: {len(loader.dataset)}')

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch = move_to_device(batch, device)
                bs = batch['length'].shape[0]

                if amp_on:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        pred, score = model.predict(
                            batch,
                            budget_fraction=scfg.get('default_budget_fraction', 0.30),
                            use_density_budget=scfg.get('use_density_budget', True),
                            score_threshold=scfg.get('score_threshold', 0.4),
                        )
                else:
                    pred, score = model.predict(
                        batch,
                        budget_fraction=scfg.get('default_budget_fraction', 0.30),
                        use_density_budget=scfg.get('use_density_budget', True),
                        score_threshold=scfg.get('score_threshold', 0.4),
                    )

                names = batch.get('names', [f'sample_{batch_idx}_{i}' for i in range(bs)])
                seqs = batch.get('seqs', [''] * bs)

                for i in range(bs):
                    length = int(batch['length'][i])
                    name = names[i]
                    raw_seq = seqs[i]

                    # Compute F1
                    p = pred[i, 0, :length, :length].cpu().numpy() > 0.5
                    g = batch['contact'][i, 0, :length, :length].cpu().numpy() > 0.5
                    mask = np.triu(np.ones((length, length), dtype=bool), k=3)
                    p_m = p & mask
                    g_m = g & mask

                    tp = int((p_m & g_m).sum())
                    fp = int((p_m & ~g_m).sum())
                    fn = int((~p_m & g_m).sum())
                    gt_n = tp + fn
                    pred_n = tp + fp

                    if gt_n == 0 and pred_n == 0:
                        f1 = 1.0
                    elif gt_n == 0:
                        f1 = 0.0
                    else:
                        prec = tp / max(tp + fp, 1)
                        rec = tp / max(tp + fn, 1)
                        f1 = 2 * prec * rec / max(prec + rec, 1e-12)

                    if f1 >= args.threshold:
                        continue

                    # Bad case — extract detailed pair info
                    seq = raw_seq if raw_seq else seq_map.get(name, '')
                    seq_display = seq.upper().replace('T', 'U')

                    gt_map = batch['contact'][i, 0, :length, :length].cpu().numpy()
                    pred_map = pred[i, 0, :length, :length].cpu().numpy()

                    gt_pairs_list = extract_pairs(gt_map, seq, length)
                    pred_pairs_list = extract_pairs(pred_map, seq, length)

                    gt_set = set((pp['i'], pp['j']) for pp in gt_pairs_list)
                    pred_set = set((pp['i'], pp['j']) for pp in pred_pairs_list)

                    tp_set = gt_set & pred_set
                    fp_set = pred_set - gt_set
                    fn_set = gt_set - pred_set

                    tp_pairs = [pp for pp in pred_pairs_list if (pp['i'], pp['j']) in tp_set]
                    fp_pairs = [pp for pp in pred_pairs_list if (pp['i'], pp['j']) in fp_set]
                    fn_pairs = [pp for pp in gt_pairs_list if (pp['i'], pp['j']) in fn_set]

                    record = {
                        "name": name,
                        "stage": stage_name,
                        "length": length,
                        "seq": seq_display[:length],
                        "f1": round(f1, 4),
                        "precision": round(tp / max(tp + fp, 1), 4),
                        "recall": round(tp / max(tp + fn, 1), 4),
                        "gt_pair_count": len(gt_pairs_list),
                        "pred_pair_count": len(pred_pairs_list),
                        "tp_count": len(tp_pairs),
                        "fp_count": len(fp_pairs),
                        "fn_count": len(fn_pairs),
                        "gt_pairs": gt_pairs_list,
                        "pred_pairs": pred_pairs_list,
                        "tp_pairs": tp_pairs,
                        "fp_pairs": fp_pairs,
                        "fn_pairs": fn_pairs,
                    }
                    all_bad_cases.append(record)

                if (batch_idx + 1) % 100 == 0:
                    print(f'    Batch {batch_idx+1}/{len(loader)}, collected {len(all_bad_cases)} bad cases')

    # Sort by F1
    all_bad_cases.sort(key=lambda x: (x['f1'], -x['length']))

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_bad_cases, f, indent=2, ensure_ascii=False)

    print(f'\n{"="*60}')
    print(f'Exported {len(all_bad_cases)} bad cases to: {out_path}')
    print(f'  Train: {sum(1 for r in all_bad_cases if r["stage"]=="train")}')
    print(f'  Val:   {sum(1 for r in all_bad_cases if r["stage"]=="val")}')
    print(f'  Test:  {sum(1 for r in all_bad_cases if r["stage"]=="test")}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
