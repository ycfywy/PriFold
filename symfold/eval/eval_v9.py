# -*- coding: utf-8 -*-
"""PriFold v9 DensityNet-Pro+ — Test Evaluation Script.

Evaluates the best checkpoint on bprna-test (and optionally archiveii-test).
Outputs per-sample metrics and generates a markdown report.

Usage:
  CUDA_VISIBLE_DEVICES=1 python symfold/eval/eval_v9.py \
    --config symfold/config/v9/v9_ddp.json \
    --ckpt symfold/outputs/v9_ddp/model/best.pt \
    --device cuda:0 \
    --output_dir symfold/outputs/v9_ddp/test_eval
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_records, PriFoldSymFlowDataset, make_collate_fn
from symfold.metrics import contact_metrics
from symfold.v9.model import DensityNetProPlus


def build_model(cfg: dict, extractor) -> DensityNetProPlus:
    """Build v9 DensityNet-Pro+ from config."""
    mcfg = cfg['model']
    v9cfg = cfg.get('v9', {})
    lcfg = cfg.get('loss', {})

    model = DensityNetProPlus(
        extractor=extractor,
        freeze_mars=mcfg.get('freeze_mars', True),
        mars_dim=mcfg.get('mars_dim', 1056),
        mars_n_attn_layers=mcfg.get('mars_n_attn_layers', 6),
        mars_n_heads=mcfg.get('mars_n_heads', 12),
        mars_hidden_layer_indices=mcfg.get('mars_hidden_layer_indices', [3, 6, 9, 12]),
        hidden_dim=v9cfg.get('hidden_dim', 192),
        num_layers=v9cfg.get('num_layers', 8),
        num_heads=v9cfg.get('num_heads', 6),
        dim_head=v9cfg.get('dim_head', 32),
        ff_mult=v9cfg.get('ff_mult', 4),
        dropout=v9cfg.get('dropout', 0.2),
        drop_path=v9cfg.get('drop_path', 0.15),
        use_rope=v9cfg.get('use_rope', True),
        focal_gamma=lcfg.get('focal_gamma', 1.0),
        pos_weight_base=lcfg.get('pos_weight_base', 99.0),
        dice_weight=lcfg.get('dice_weight', 0.5),
        dst_weight=lcfg.get('dst_weight', 0.5),
        dst_low_threshold=lcfg.get('dst_low_threshold', 0.05),
        dst_tversky_alpha=lcfg.get('dst_tversky_alpha', 0.7),
        dst_tversky_beta=lcfg.get('dst_tversky_beta', 0.3),
        pair_count_weight=lcfg.get('pair_count_weight', 0.3),
        ratio_penalty_weight=lcfg.get('ratio_penalty_weight', 0.2),
        ratio_penalty_threshold=lcfg.get('ratio_penalty_threshold', 1.20),
        density_loss_weight=lcfg.get('density_loss_weight', 0.3),
        ohem_enabled=lcfg.get('ohem_enabled', True),
        ohem_neg_ratio=lcfg.get('ohem_neg_ratio', 3),
        fp_penalty_enabled=lcfg.get('fp_penalty_enabled', True),
        fp_penalty_weight=lcfg.get('fp_penalty_weight', 2.0),
        bp_compat_enabled=lcfg.get('bp_compat_enabled', False),
        bp_compat_weight=lcfg.get('bp_compat_weight', 0.0),
        bp_compat_in_inference=lcfg.get('bp_compat_in_inference', False),
        shift_loss_enabled=lcfg.get('shift_loss_enabled', True),
        shift_loss_weight=lcfg.get('shift_loss_weight', 0.8),
        shift_radius=lcfg.get('shift_radius', 2),
    )
    return model


def evaluate_detailed(model, loader, device, config):
    """Evaluate model and return per-sample metrics."""
    model.eval()
    scfg = config.get('sampling', {})
    amp_dtype = torch.bfloat16

    per_sample = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                pred, extras = model.predict(
                    batch,
                    budget_fraction=scfg.get('default_budget_fraction', 0.30),
                    use_density_budget=scfg.get('use_density_budget', True),
                    score_threshold=scfg.get('score_threshold', 0.43),
                    length_decay=scfg.get('length_decay', 0.15),
                    budget_floor=scfg.get('budget_floor', 0.6),
                )

            # Per-sample metrics
            pred_cpu = pred.detach().cpu().float()
            target_cpu = batch['contact'].detach().cpu().float()
            lengths_cpu = batch['length'].detach().cpu().long()
            names = batch.get('names', [f'sample_{i}' for i in range(pred.shape[0])])
            datasets = batch.get('datasets', ['unknown'] * pred.shape[0])

            for i in range(pred_cpu.shape[0]):
                length = int(lengths_cpu[i])
                p = pred_cpu[i].squeeze()[:length, :length] > 0.5
                y = target_cpu[i].squeeze()[:length, :length] > 0.5
                idx = torch.arange(length)
                mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
                mask &= (idx.view(length, 1) - idx.view(1, length)).abs() >= 3
                p_m = p[mask]
                y_m = y[mask]
                tp = int((p_m & y_m).sum())
                fp = int((p_m & ~y_m).sum())
                fn = int((~p_m & y_m).sum())
                tn = int((~p_m & ~y_m).sum())

                gt_pairs = tp + fn
                pred_pairs = tp + fp

                if gt_pairs == 0 and pred_pairs == 0:
                    precision, recall, f1, mcc = 1.0, 1.0, 1.0, 1.0
                elif gt_pairs == 0 and pred_pairs > 0:
                    precision, recall, f1, mcc = 0.0, 1.0, 0.0, 0.0
                else:
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
                    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
                    mcc = ((tp * tn) - (fp * fn)) / denom

                # Compute density
                density = gt_pairs / max(length * (length - 1) / 2, 1)

                # Compute shifted predictions (FP within ±1 of any GT pair)
                shifted_fp = 0
                if fp > 0 and gt_pairs > 0:
                    pred_set = set(zip(*torch.where(p & mask)))
                    gt_set = set(zip(*torch.where(y & mask)))
                    for pi, pj in pred_set:
                        if (int(pi), int(pj)) not in gt_set:
                            # Check if within ±1 of any GT
                            is_shifted = False
                            for di in range(-1, 2):
                                for dj in range(-1, 2):
                                    if (di, dj) == (0, 0):
                                        continue
                                    if (int(pi) + di, int(pj) + dj) in gt_set:
                                        is_shifted = True
                                        break
                                if is_shifted:
                                    break
                            if is_shifted:
                                shifted_fp += 1

                per_sample.append({
                    'name': names[i],
                    'dataset': datasets[i],
                    'length': length,
                    'density': density,
                    'gt_pairs': gt_pairs,
                    'pred_pairs': pred_pairs,
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'mcc': mcc,
                    'shifted_fp': shifted_fp,
                    'pred_gt_ratio': pred_pairs / max(gt_pairs, 1),
                })

    return per_sample


def compute_summary(per_sample):
    """Compute summary statistics from per-sample results."""
    n = len(per_sample)
    if n == 0:
        return {}

    keys = ['precision', 'recall', 'f1', 'mcc']
    summary = {k: np.mean([s[k] for s in per_sample]) for k in keys}
    summary['n'] = n
    summary['avg_length'] = np.mean([s['length'] for s in per_sample])
    summary['avg_gt_pairs'] = np.mean([s['gt_pairs'] for s in per_sample])
    summary['avg_pred_pairs'] = np.mean([s['pred_pairs'] for s in per_sample])
    summary['avg_pred_gt_ratio'] = np.mean([s['pred_gt_ratio'] for s in per_sample])
    summary['avg_density'] = np.mean([s['density'] for s in per_sample])

    # F1 distribution
    f1s = [s['f1'] for s in per_sample]
    summary['f1_std'] = np.std(f1s)
    summary['f1_median'] = np.median(f1s)
    summary['f1_q25'] = np.percentile(f1s, 25)
    summary['f1_q75'] = np.percentile(f1s, 75)

    # Bad cases (F1 < 0.3)
    bad = [s for s in per_sample if s['f1'] < 0.3]
    summary['bad_rate'] = len(bad) / n
    summary['bad_count'] = len(bad)

    # Shifted FP analysis
    total_fp = sum(s['fp'] for s in per_sample)
    total_shifted_fp = sum(s['shifted_fp'] for s in per_sample)
    summary['total_fp'] = total_fp
    summary['total_shifted_fp'] = total_shifted_fp
    summary['shifted_fp_rate'] = total_shifted_fp / max(total_fp, 1)

    # Length-bin analysis
    length_bins = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]
    summary['length_bins'] = {}
    for lo, hi in length_bins:
        bin_samples = [s for s in per_sample if lo <= s['length'] < hi]
        if bin_samples:
            summary['length_bins'][f'{lo}-{hi}'] = {
                'n': len(bin_samples),
                'f1': np.mean([s['f1'] for s in bin_samples]),
                'precision': np.mean([s['precision'] for s in bin_samples]),
                'recall': np.mean([s['recall'] for s in bin_samples]),
            }

    return summary


def generate_report(summary, per_sample, config, args, elapsed):
    """Generate markdown evaluation report."""
    scfg = config.get('sampling', {})
    lines = []
    lines.append('# v9 DensityNet-Pro+ 测试评估报告')
    lines.append('')
    lines.append(f'> 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'> Checkpoint: `{args.ckpt}`')
    lines.append(f'> Config: `{args.config}`')
    lines.append(f'> 评估耗时: {elapsed:.1f}s')
    lines.append('')

    # Overall results
    lines.append('## 1. 总体指标')
    lines.append('')
    lines.append(f'| 指标 | 值 |')
    lines.append(f'|------|-----|')
    lines.append(f'| **Test F1** | **{summary["f1"]:.4f}** |')
    lines.append(f'| Precision | {summary["precision"]:.4f} |')
    lines.append(f'| Recall | {summary["recall"]:.4f} |')
    lines.append(f'| MCC | {summary["mcc"]:.4f} |')
    lines.append(f'| 样本数 | {summary["n"]} |')
    lines.append(f'| 平均长度 | {summary["avg_length"]:.1f} |')
    lines.append(f'| 平均 GT pairs | {summary["avg_gt_pairs"]:.1f} |')
    lines.append(f'| 平均 Pred pairs | {summary["avg_pred_pairs"]:.1f} |')
    lines.append(f'| Pred/GT ratio | {summary["avg_pred_gt_ratio"]:.3f} |')
    lines.append('')

    # Version comparison
    lines.append('## 2. 版本对比')
    lines.append('')
    lines.append(f'| 版本 | Test F1 | Precision | Recall | 特性 |')
    lines.append(f'|------|---------|-----------|--------|------|')
    lines.append(f'| **v9** | **{summary["f1"]:.4f}** | {summary["precision"]:.4f} | {summary["recall"]:.4f} | +RoPE +shift margin +DST↓ +正则化↑ +允许NC |')
    lines.append(f'| v7 | 0.6538 | — | — | 纯判别式 DensityNet |')
    lines.append(f'| v8 | 0.6105 | — | — | +OHEM +FP penalty +shift +decay |')
    lines.append(f'| baseline | 0.7700 | — | — | 官方 PriFold |')
    lines.append('')

    # F1 distribution
    lines.append('## 3. F1 分布')
    lines.append('')
    lines.append(f'| 统计量 | 值 |')
    lines.append(f'|--------|-----|')
    lines.append(f'| Mean | {summary["f1"]:.4f} |')
    lines.append(f'| Median | {summary["f1_median"]:.4f} |')
    lines.append(f'| Std | {summary["f1_std"]:.4f} |')
    lines.append(f'| Q25 | {summary["f1_q25"]:.4f} |')
    lines.append(f'| Q75 | {summary["f1_q75"]:.4f} |')
    lines.append(f'| Bad rate (F1<0.3) | {summary["bad_rate"]:.1%} ({summary["bad_count"]}/{summary["n"]}) |')
    lines.append('')

    # Length-bin analysis
    lines.append('## 4. 按长度分组')
    lines.append('')
    lines.append(f'| 长度区间 | N | F1 | Precision | Recall |')
    lines.append(f'|----------|---|-----|-----------|--------|')
    for bin_name, stats in summary.get('length_bins', {}).items():
        lines.append(f'| {bin_name} | {stats["n"]} | {stats["f1"]:.4f} | {stats["precision"]:.4f} | {stats["recall"]:.4f} |')
    lines.append('')

    # Shifted FP analysis
    lines.append('## 5. Shifted Prediction 分析')
    lines.append('')
    lines.append(f'| 指标 | 值 |')
    lines.append(f'|------|-----|')
    lines.append(f'| 总 FP | {summary["total_fp"]} |')
    lines.append(f'| Shifted FP (±1) | {summary["total_shifted_fp"]} |')
    lines.append(f'| Shifted FP 占比 | {summary["shifted_fp_rate"]:.1%} |')
    lines.append('')
    lines.append('> Shifted FP = FP 预测位于 GT 配对的 ±1 位置，说明模型"看到了"配对但位置偏移。')
    lines.append('')

    # Sampling params
    lines.append('## 6. 推理参数')
    lines.append('')
    lines.append(f'```json')
    lines.append(json.dumps(scfg, indent=2))
    lines.append(f'```')
    lines.append('')

    # Top-10 worst cases
    lines.append('## 7. 最差 10 个样本')
    lines.append('')
    sorted_samples = sorted(per_sample, key=lambda x: x['f1'])
    lines.append(f'| # | Name | Len | GT | Pred | F1 | Prec | Rec | Pred/GT |')
    lines.append(f'|---|------|-----|-----|------|-----|------|------|---------|')
    for i, s in enumerate(sorted_samples[:10]):
        lines.append(f'| {i+1} | {s["name"][:20]} | {s["length"]} | {s["gt_pairs"]} | '
                     f'{s["pred_pairs"]} | {s["f1"]:.3f} | {s["precision"]:.3f} | '
                     f'{s["recall"]:.3f} | {s["pred_gt_ratio"]:.2f} |')
    lines.append('')

    # Top-10 best cases
    lines.append('## 8. 最佳 10 个样本')
    lines.append('')
    sorted_best = sorted(per_sample, key=lambda x: x['f1'], reverse=True)
    lines.append(f'| # | Name | Len | GT | Pred | F1 | Prec | Rec |')
    lines.append(f'|---|------|-----|-----|------|-----|------|------|')
    for i, s in enumerate(sorted_best[:10]):
        lines.append(f'| {i+1} | {s["name"][:20]} | {s["length"]} | {s["gt_pairs"]} | '
                     f'{s["pred_pairs"]} | {s["f1"]:.3f} | {s["precision"]:.3f} | {s["recall"]:.3f} |')
    lines.append('')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='v9 Test Evaluation')
    parser.add_argument('--config', type=str, default='symfold/config/v9/v9_ddp.json')
    parser.add_argument('--ckpt', type=str, default='symfold/outputs/v9_ddp/model/best.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='symfold/outputs/v9_ddp/test_eval')
    parser.add_argument('--stages', type=str, nargs='+', default=['bprna-test'],
                        help='Test stages to evaluate')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f'[Eval] Device: {device}')
    print(f'[Eval] Checkpoint: {args.ckpt}')
    print(f'[Eval] Stages: {args.stages}')

    # Load LM extractor
    class Args:
        pass
    lm_args = Args()
    lm_args.pretrained_lm_dir = config['paths'].get('pretrained_lm_dir', str(ROOT / 'model'))
    lm_args.model_scale = config['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)

    # Build model
    model = build_model(config, extractor).to(device)
    
    # Load checkpoint (handle various formats)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    # Try different state dict keys
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'model' in ckpt and isinstance(ckpt['model'], dict):
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    # Strip 'module.' prefix if present (DDP checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f'[Eval] Model loaded from {args.ckpt}')
    if missing:
        print(f'[Eval] WARNING: {len(missing)} missing keys')
    if unexpected:
        print(f'[Eval] WARNING: {len(unexpected)} unexpected keys')
    
    if 'epoch' in ckpt:
        print(f'[Eval] Checkpoint epoch: {ckpt["epoch"]}')
    if 'best_f1' in ckpt:
        print(f'[Eval] Checkpoint best val F1: {ckpt["best_f1"]:.4f}')

    model.eval()

    # Evaluate each stage
    all_results = {}
    for stage in args.stages:
        print(f'\n{"="*60}')
        print(f'[Eval] Stage: {stage}')
        print(f'{"="*60}')

        # Build test loader
        tcfg = config['training']
        data_dir = config['paths']['data_dir']
        max_len = tcfg.get('max_len_filter', 490)
        records = build_records(data_dir, stage, max_len=max_len)
        dataset = PriFoldSymFlowDataset(records, augment=False)
        collate_fn = make_collate_fn(tokenizer)

        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        print(f'[Eval] Samples: {len(dataset)}')
        t0 = time.time()

        # Run evaluation
        per_sample = evaluate_detailed(model, loader, device, config)
        elapsed = time.time() - t0

        # Summary
        summary = compute_summary(per_sample)
        print(f'\n[Result] {stage}: F1={summary["f1"]:.4f} | '
              f'Prec={summary["precision"]:.4f} | Rec={summary["recall"]:.4f} | '
              f'MCC={summary["mcc"]:.4f} | N={summary["n"]} | Time={elapsed:.1f}s')

        # Save per-sample results
        per_sample_path = output_dir / f'{stage.replace("-", "_")}_per_sample.json'
        with open(per_sample_path, 'w') as f:
            json.dump(per_sample, f, indent=2)
        print(f'[Eval] Per-sample saved: {per_sample_path}')

        # Generate report
        report = generate_report(summary, per_sample, config, args, elapsed)
        report_path = output_dir / f'{stage.replace("-", "_")}_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f'[Eval] Report saved: {report_path}')

        all_results[stage] = {
            'summary': summary,
            'elapsed': elapsed,
        }

    # Save combined results
    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    combined_path = output_dir / 'eval_results.json'
    with open(combined_path, 'w') as f:
        json.dump(to_serializable(all_results), f, indent=2)

    print(f'\n[Done] All results saved to {output_dir}')


if __name__ == '__main__':
    main()
