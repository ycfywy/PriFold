# -*- coding: utf-8 -*-
"""Per-RNA detailed case analysis for PriFold-SymFlow v6.

Analyzes every sample in bprna-test (and optionally train/val),
outputs per-sample metrics, and produces detailed breakdowns.

Usage:
  python symfold/analyze_v6_cases.py \
    --ckpt symfold/outputs/v6_full/model/best.pt \
    --out_dir symfold/outputs/v6_full/case_analysis \
    --test_sets bprna-train,bprna-val,bprna-test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_loader
from symfold.train.train_v6 import build_model, load_config, move_to_device


def per_sample_metrics(pred: torch.Tensor, target: torch.Tensor, length: int) -> dict:
    """Compute detailed per-sample metrics."""
    p = pred.detach().cpu().float().squeeze()[:length, :length] > 0.5
    y = target.detach().cpu().float().squeeze()[:length, :length] > 0.5
    idx = torch.arange(length)
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    mask &= (idx.view(length, 1) - idx.view(1, length)).abs() >= 3
    p_masked = p[mask]
    y_masked = y[mask]
    tp = int((p_masked & y_masked).sum())
    fp = int((p_masked & ~y_masked).sum())
    fn = int((~p_masked & y_masked).sum())
    tn = int((~p_masked & ~y_masked).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc = ((tp * tn) - (fp * fn)) / denom

    # Pairing pattern analysis
    gt_pairs_count = tp + fn
    pred_pairs_count = tp + fp
    density = gt_pairs_count / max(length, 1)

    # Contact distance distribution
    gt_dists = []
    pred_dists = []
    y_full = target.detach().cpu().float().squeeze()[:length, :length] > 0.5
    p_full = pred.detach().cpu().float().squeeze()[:length, :length] > 0.5
    for i in range(length):
        for j in range(i + 3, length):
            if y_full[i, j]:
                gt_dists.append(j - i)
            if p_full[i, j]:
                pred_dists.append(j - i)

    # Stem analysis: count consecutive pairs (i,j), (i+1,j-1)
    gt_stems = 0
    pred_stems = 0
    for i in range(length - 1):
        for j in range(i + 4, length):
            if y_full[i, j] and y_full[i + 1, j - 1]:
                gt_stems += 1
            if p_full[i, j] and p_full[i + 1, j - 1]:
                pred_stems += 1

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'gt_pairs': gt_pairs_count,
        'pred_pairs': pred_pairs_count,
        'density': density,
        'pred_gt_ratio': pred_pairs_count / max(gt_pairs_count, 1),
        'mean_gt_dist': float(np.mean(gt_dists)) if gt_dists else 0,
        'max_gt_dist': max(gt_dists) if gt_dists else 0,
        'mean_pred_dist': float(np.mean(pred_dists)) if pred_dists else 0,
        'gt_stems': gt_stems,
        'pred_stems': pred_stems,
        'stem_ratio': pred_stems / max(gt_stems, 1),
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


def density_bin(density: float) -> str:
    if density < 0.10:
        return '<0.10'
    if density < 0.18:
        return '0.10-0.18'
    if density < 0.25:
        return '0.18-0.25'
    if density < 0.35:
        return '0.25-0.35'
    return '>=0.35'


def f1_bin(f1: float) -> str:
    if f1 == 0:
        return 'F1=0'
    if f1 < 0.3:
        return '0<F1<0.3'
    if f1 < 0.5:
        return '0.3<=F1<0.5'
    if f1 < 0.7:
        return '0.5<=F1<0.7'
    if f1 < 0.9:
        return '0.7<=F1<0.9'
    return 'F1>=0.9'


def analyze_groups(rows: list[dict], group_key: str) -> dict:
    """Group rows by a key and compute aggregate stats."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[group_key]].append(r)

    result = {}
    for gname, items in sorted(groups.items()):
        n = len(items)
        result[gname] = {
            'n': n,
            'pct': f"{100*n/len(rows):.1f}%",
            'f1': np.mean([x['f1'] for x in items]),
            'precision': np.mean([x['precision'] for x in items]),
            'recall': np.mean([x['recall'] for x in items]),
            'pred_gt_ratio': np.mean([x['pred_gt_ratio'] for x in items]),
            'density': np.mean([x['density'] for x in items]),
            'length': np.mean([x['length'] for x in items]),
            'gt_pairs': np.mean([x['gt_pairs'] for x in items]),
            'pred_pairs': np.mean([x['pred_pairs'] for x in items]),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--test_sets', default='bprna-test')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--worst_k', type=int, default=100)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = load_config(args.config) if args.config else ckpt['config']
    scfg = cfg.setdefault('sampling', {})

    stages = [x.strip() for x in args.test_sets.split(',') if x.strip()]

    class A:
        pass
    lm_args = A()
    lm_args.pretrained_lm_dir = cfg['paths']['pretrained_lm_dir']
    lm_args.model_scale = cfg['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)

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
    print(f"Evaluating on: {stages}")
    print(f"Sampling config: {scfg}")

    with torch.no_grad():
        for stage in stages:
            loader = build_loader(stage, cfg, tokenizer, shuffle=False)
            rows = []
            for step, batch in enumerate(loader):
                batch = move_to_device(batch, device)
                sample_kwargs = dict(
                    num_steps=scfg.get('num_steps', 20),
                    num_samples_per_input=scfg.get('num_samples_per_input', 1),
                    density_guided=scfg.get('density_guided', False),
                    projection_mode=scfg.get('projection_mode', 'score'),
                    use_density_budget=scfg.get('use_density_budget', False),
                    budget_scale=scfg.get('budget_scale', 1.0),
                    candidate_weight=scfg.get('candidate_weight', 0.35),
                    direct_score_weight=scfg.get('direct_score_weight', None),
                    score_threshold=scfg.get('score_threshold', 0.5),
                    default_budget_fraction=scfg.get('default_budget_fraction', 0.30),
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
                        'density_bin': density_bin(m['density']),
                        'f1_bin': f1_bin(m['f1']),
                        **m,
                    }
                    rows.append(row)
                    all_rows.append(row)

                if step % 20 == 0:
                    print(f"  [{stage}] step={step}/{len(loader)}, samples={len(rows)}")

            # Save per-stage results
            rows_sorted = sorted(rows, key=lambda x: (x['f1'], -x['length']))
            csv_path = out_dir / f'{stage.replace("-", "_")}_cases.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
                writer.writeheader()
                writer.writerows(rows_sorted)
            with open(out_dir / f'{stage.replace("-", "_")}_worst_{args.worst_k}.json', 'w') as f:
                json.dump(rows_sorted[:args.worst_k], f, indent=2)
            print(f'  [{stage}] N={len(rows)}, saved -> {csv_path}')

    # ---- Comprehensive analysis ----
    print(f"\n{'='*60}")
    print(f"TOTAL SAMPLES: {len(all_rows)}")
    print(f"{'='*60}")

    # Per-stage summary
    stage_summary = {}
    for stage in stages:
        stage_rows = [r for r in all_rows if r['stage'] == stage]
        stage_summary[stage] = {
            'n': len(stage_rows),
            'f1': float(np.mean([r['f1'] for r in stage_rows])),
            'precision': float(np.mean([r['precision'] for r in stage_rows])),
            'recall': float(np.mean([r['recall'] for r in stage_rows])),
            'pred_gt_ratio': float(np.mean([r['pred_gt_ratio'] for r in stage_rows])),
            'f1_zero_count': sum(1 for r in stage_rows if r['f1'] == 0),
            'f1_below_03_count': sum(1 for r in stage_rows if r['f1'] < 0.3),
            'f1_above_09_count': sum(1 for r in stage_rows if r['f1'] >= 0.9),
        }

    # Overall stats
    overall = {
        'n': len(all_rows),
        'f1': float(np.mean([r['f1'] for r in all_rows])),
        'precision': float(np.mean([r['precision'] for r in all_rows])),
        'recall': float(np.mean([r['recall'] for r in all_rows])),
        'pred_gt_ratio': float(np.mean([r['pred_gt_ratio'] for r in all_rows])),
        'density': float(np.mean([r['density'] for r in all_rows])),
        'length': float(np.mean([r['length'] for r in all_rows])),
    }

    # Group analyses
    by_length = analyze_groups(all_rows, 'length_bin')
    by_density = analyze_groups(all_rows, 'density_bin')
    by_f1 = analyze_groups(all_rows, 'f1_bin')

    # Bad cases analysis (F1 < 0.3)
    bad_cases = [r for r in all_rows if r['f1'] < 0.3]
    good_cases = [r for r in all_rows if r['f1'] >= 0.7]

    bad_analysis = {
        'n': len(bad_cases),
        'pct': f"{100*len(bad_cases)/len(all_rows):.1f}%",
        'avg_f1': float(np.mean([r['f1'] for r in bad_cases])) if bad_cases else 0,
        'avg_length': float(np.mean([r['length'] for r in bad_cases])) if bad_cases else 0,
        'avg_density': float(np.mean([r['density'] for r in bad_cases])) if bad_cases else 0,
        'avg_pred_gt_ratio': float(np.mean([r['pred_gt_ratio'] for r in bad_cases])) if bad_cases else 0,
        'avg_gt_pairs': float(np.mean([r['gt_pairs'] for r in bad_cases])) if bad_cases else 0,
        'avg_pred_pairs': float(np.mean([r['pred_pairs'] for r in bad_cases])) if bad_cases else 0,
        'avg_mean_gt_dist': float(np.mean([r['mean_gt_dist'] for r in bad_cases])) if bad_cases else 0,
        'avg_max_gt_dist': float(np.mean([r['max_gt_dist'] for r in bad_cases])) if bad_cases else 0,
        'avg_stem_ratio': float(np.mean([r['stem_ratio'] for r in bad_cases])) if bad_cases else 0,
        'length_distribution': {},
        'density_distribution': {},
    }
    if bad_cases:
        for lb in sorted(set(r['length_bin'] for r in bad_cases)):
            cnt = sum(1 for r in bad_cases if r['length_bin'] == lb)
            bad_analysis['length_distribution'][lb] = {
                'n': cnt, 'pct': f"{100*cnt/len(bad_cases):.1f}%"}
        for db in sorted(set(r['density_bin'] for r in bad_cases)):
            cnt = sum(1 for r in bad_cases if r['density_bin'] == db)
            bad_analysis['density_distribution'][db] = {
                'n': cnt, 'pct': f"{100*cnt/len(bad_cases):.1f}%"}

    good_analysis = {
        'n': len(good_cases),
        'pct': f"{100*len(good_cases)/len(all_rows):.1f}%",
        'avg_f1': float(np.mean([r['f1'] for r in good_cases])) if good_cases else 0,
        'avg_length': float(np.mean([r['length'] for r in good_cases])) if good_cases else 0,
        'avg_density': float(np.mean([r['density'] for r in good_cases])) if good_cases else 0,
        'avg_pred_gt_ratio': float(np.mean([r['pred_gt_ratio'] for r in good_cases])) if good_cases else 0,
        'avg_mean_gt_dist': float(np.mean([r['mean_gt_dist'] for r in good_cases])) if good_cases else 0,
        'avg_stem_ratio': float(np.mean([r['stem_ratio'] for r in good_cases])) if good_cases else 0,
    }

    # Compile full report
    report = {
        'overall': overall,
        'by_stage': stage_summary,
        'by_length_bin': by_length,
        'by_density_bin': by_density,
        'by_f1_bin': by_f1,
        'bad_cases_analysis': bad_analysis,
        'good_cases_analysis': good_analysis,
    }

    # Convert numpy floats for JSON serialization
    def to_native(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(x) for x in obj]
        return obj

    report = to_native(report)

    with open(out_dir / 'detailed_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\nOverall: F1={overall['f1']:.4f} P={overall['precision']:.4f} R={overall['recall']:.4f}")
    print(f"  pred/gt ratio={overall['pred_gt_ratio']:.3f}")
    print(f"\nBy Stage:")
    for stage, s in stage_summary.items():
        print(f"  {stage}: N={s['n']} F1={s['f1']:.4f} P={s['precision']:.4f} R={s['recall']:.4f} "
              f"pred/gt={s['pred_gt_ratio']:.3f} F1=0:{s['f1_zero_count']} F1<0.3:{s['f1_below_03_count']}")
    print(f"\nBy Length:")
    for k, v in by_length.items():
        print(f"  {k:10s}: n={v['n']:4d} F1={v['f1']:.4f} P={v['precision']:.4f} R={v['recall']:.4f} "
              f"pred/gt={v['pred_gt_ratio']:.3f} density={v['density']:.3f}")
    print(f"\nBy Density:")
    for k, v in by_density.items():
        print(f"  {k:10s}: n={v['n']:4d} F1={v['f1']:.4f} P={v['precision']:.4f} R={v['recall']:.4f} "
              f"pred/gt={v['pred_gt_ratio']:.3f}")
    print(f"\nBy F1 Bin:")
    for k, v in by_f1.items():
        print(f"  {k:15s}: n={v['n']:4d} ({v['pct']}) len={v['length']:.0f} "
              f"density={v['density']:.3f} pred/gt={v['pred_gt_ratio']:.3f}")
    print(f"\nBad cases (F1<0.3): {bad_analysis['n']} ({bad_analysis['pct']})")
    if bad_cases:
        print(f"  avg length={bad_analysis['avg_length']:.0f}, density={bad_analysis['avg_density']:.3f}")
        print(f"  avg pred/gt={bad_analysis['avg_pred_gt_ratio']:.3f}")
        print(f"  avg gt_pairs={bad_analysis['avg_gt_pairs']:.1f}, pred_pairs={bad_analysis['avg_pred_pairs']:.1f}")
        print(f"  length dist: {bad_analysis['length_distribution']}")
        print(f"  density dist: {bad_analysis['density_distribution']}")
    print(f"\nGood cases (F1>=0.7): {good_analysis['n']} ({good_analysis['pct']})")

    print(f"\nReport saved to: {out_dir / 'detailed_analysis.json'}")


if __name__ == '__main__':
    main()
