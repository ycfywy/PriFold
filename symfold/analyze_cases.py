# -*- coding: utf-8 -*-
"""Per-RNA bad case analysis for PriFold-SymFlow experiments.

输出每条 RNA 的 length / gt_pairs / pred_pairs / density / P/R/F1/MCC，
并按 F1 排序保存 worst cases，供下一阶段改进使用。

Example:
  python symfold/analyze_cases.py \
    --ckpt symfold/outputs/v3_bprna/model/best.pt \
    --test_sets bprna-test \
    --out_dir symfold/outputs/v3_bprna/case_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor                              # noqa: E402
from symfold.data import build_loader                           # noqa: E402
from symfold.train_v2 import load_config, move_to_device         # noqa: E402
from symfold.train_v4 import build_model as build_model_v4       # noqa: E402
from symfold.train_v3 import build_model as build_model_v3       # noqa: E402
from symfold.train_v2 import build_model as build_model_v2       # noqa: E402


def per_sample_metrics(pred: torch.Tensor, target: torch.Tensor, length: int) -> dict:
    p = pred.detach().cpu().float().squeeze()[:length, :length] > 0.5
    y = target.detach().cpu().float().squeeze()[:length, :length] > 0.5
    idx = torch.arange(length)
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    mask &= (idx.view(length, 1) - idx.view(1, length)).abs() >= 3
    p = p[mask]
    y = y[mask]
    tp = int((p & y).sum())
    fp = int((p & ~y).sum())
    fn = int((~p & y).sum())
    tn = int((~p & ~y).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = ((tp * tn) - (fp * fn)) / denom
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'gt_pairs': tp + fn,
        'pred_pairs': tp + fp,
    }


def length_bin(length: int) -> str:
    if length < 80:
        return '<80'
    if length < 160:
        return '80-159'
    if length < 240:
        return '160-239'
    if length < 320:
        return '240-319'
    if length < 400:
        return '320-399'
    return '400+'


def summarize(rows: list[dict]) -> dict:
    def avg(items, key):
        return sum(float(x[key]) for x in items) / max(len(items), 1)

    out = {'n': len(rows)}
    for key in ['precision', 'recall', 'f1', 'mcc', 'gt_pairs', 'pred_pairs', 'length', 'density']:
        out[key] = avg(rows, key)
    by_len = {}
    for b in sorted(set(r['length_bin'] for r in rows)):
        part = [r for r in rows if r['length_bin'] == b]
        by_len[b] = {
            'n': len(part),
            'f1': avg(part, 'f1'),
            'precision': avg(part, 'precision'),
            'recall': avg(part, 'recall'),
            'length': avg(part, 'length'),
            'density': avg(part, 'density'),
            'gt_pairs': avg(part, 'gt_pairs'),
            'pred_pairs': avg(part, 'pred_pairs'),
        }
    out['by_length_bin'] = by_len
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--test_sets', default=None)
    ap.add_argument('--num_steps', type=int, default=None)
    ap.add_argument('--density_guided', type=int, default=None)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--worst_k', type=int, default=100)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = load_config(args.config) if args.config else ckpt['config']
    scfg = cfg.setdefault('sampling', {})
    if args.num_steps is not None:
        scfg['num_steps'] = args.num_steps
    if args.density_guided is not None:
        scfg['density_guided'] = bool(args.density_guided)

    if args.test_sets:
        stages = [x.strip() for x in args.test_sets.split(',') if x.strip()]
    else:
        mode = cfg['training'].get('dataset_mode', 'bprna')
        stages = ['bprna-test'] if mode == 'bprna' else ['rnastralign-test', 'archiveii-test']

    class A:
        pass
    lm_args = A()
    lm_args.pretrained_lm_dir = cfg['paths']['pretrained_lm_dir']
    lm_args.model_scale = cfg['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)

    version = str(cfg.get('model', {}).get('version', '')).lower()
    task_name = cfg.get('task_name', '')
    if version == 'v4' or 'v4_' in task_name:
        build_model = build_model_v4
    elif version == 'v3' or 'v3_' in task_name:
        build_model = build_model_v3
    else:
        build_model = build_model_v2
    model = build_model(cfg, extractor)
    model.load_state_dict(ckpt['model'])
    device = torch.device(cfg.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    amp_name = str(cfg.get('training', {}).get('amp_dtype', 'fp32')).lower()
    amp_on = amp_name in ('bf16', 'bfloat16', 'fp16', 'float16', 'half')
    amp_dtype = torch.bfloat16 if amp_name in ('bf16', 'bfloat16') else torch.float16

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    with torch.no_grad():
        for stage in stages:
            loader = build_loader(stage, cfg, tokenizer, shuffle=False)
            rows = []
            for batch in loader:
                batch = move_to_device(batch, device)
                # NOTE: v4 默认 score-first projection 表现更好，所以 density_guided 默认 False；
                # 旧 v2/v3 case-only projection 模式仍可通过 config 显式开启。
                sample_kwargs = dict(
                    num_steps=scfg.get('num_steps', 20),
                    density_guided=scfg.get('density_guided', False),
                    num_samples_per_input=scfg.get('num_samples_per_input', 1),
                )
                if version == 'v4' or 'v4_' in task_name:
                    sample_kwargs.update(
                        projection_mode=scfg.get('projection_mode', 'score'),
                        use_density_budget=scfg.get('use_density_budget', False),
                        budget_scale=scfg.get('budget_scale', 1.0),
                        candidate_weight=scfg.get('candidate_weight', 0.35),
                        direct_score_weight=scfg.get('direct_score_weight', None),
                        score_threshold=scfg.get('score_threshold', 0.5),
                        default_budget_fraction=scfg.get('default_budget_fraction', 0.35),
                    )
                if amp_on:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        pred, score = model.sample(batch, **sample_kwargs)
                else:
                    pred, score = model.sample(batch, **sample_kwargs)
                for i in range(pred.shape[0]):
                    length = int(batch['length'][i].item())
                    m = per_sample_metrics(pred[i], batch['contact'][i], length)
                    row = {
                        'stage': stage,
                        'name': batch['names'][i],
                        'dataset': batch['datasets'][i],
                        'length': length,
                        'length_bin': length_bin(length),
                        'density': m['gt_pairs'] / max(length, 1),
                        **m,
                    }
                    rows.append(row)
                    all_rows.append(row)
            rows_sorted = sorted(rows, key=lambda x: (x['f1'], -x['length']))
            # CSV 全量
            csv_path = out_dir / f'{stage}_cases.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
                writer.writeheader()
                writer.writerows(rows_sorted)
            with open(out_dir / f'{stage}_worst_{args.worst_k}.json', 'w') as f:
                json.dump(rows_sorted[:args.worst_k], f, indent=2)
            print(f'[{stage}] saved cases -> {csv_path}, worst={len(rows_sorted[:args.worst_k])}')

    summary = summarize(all_rows)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
