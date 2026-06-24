# -*- coding: utf-8 -*-
"""PriFold v10 — 全面测试分析脚本.

从长度、RNA家族、配对距离三个维度分析模型表现，并找出拉低性能的 bad cases。

Usage:
  CUDA_VISIBLE_DEVICES=0 python symfold/eval/comprehensive_analysis_v10.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_records, PriFoldSymFlowDataset, make_collate_fn
from symfold.v9.model import DensityNetProPlus


# ============================================================
# Config
# ============================================================
CONFIG_PATH = ROOT / 'symfold/config/v10/v10_ddp.json'
CKPT_PATH = ROOT / 'symfold/outputs/v10_ddp/model/best.pt'
OUTPUT_DIR = ROOT / 'symfold/outputs/v10_ddp/comprehensive_analysis'
DEVICE = 'cuda:0'


def build_model(cfg, extractor):
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


def get_family(name: str) -> str:
    """Extract RNA family/source from sample name like bpRNA_CRW_1234."""
    parts = name.split('_')
    if len(parts) >= 3:
        return parts[1]
    return 'unknown'


def compute_pair_distances(contact_matrix, length):
    """Compute distances (|i-j|) of all true pairs in the contact matrix."""
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    pairs = torch.where(contact_matrix[:length, :length] & mask)
    if len(pairs[0]) == 0:
        return []
    distances = (pairs[1] - pairs[0]).tolist()
    return distances


def evaluate_comprehensive(model, loader, device, config):
    """Run evaluation and collect detailed per-sample + per-pair metrics."""
    model.eval()
    scfg = config.get('sampling', {})
    amp_dtype = torch.bfloat16

    per_sample = []
    total = len(loader)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if (idx + 1) % 100 == 0:
                print(f'  [{idx+1}/{total}]')
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

            pred_cpu = pred.detach().cpu().float()
            target_cpu = batch['contact'].detach().cpu().float()
            lengths_cpu = batch['length'].detach().cpu().long()
            names = batch.get('names', [f'sample_{idx}'])

            for i in range(pred_cpu.shape[0]):
                length = int(lengths_cpu[i])
                p = pred_cpu[i].squeeze()[:length, :length] > 0.5
                y = target_cpu[i].squeeze()[:length, :length] > 0.5

                mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
                idx_arr = torch.arange(length)
                mask &= (idx_arr.view(length, 1) - idx_arr.view(1, length)).abs() >= 3

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

                # Pair distance analysis
                gt_distances = compute_pair_distances(y, length)
                pred_distances = compute_pair_distances(p, length)

                # TP/FP/FN distance breakdown
                tp_mask = p & y & mask
                fp_mask = p & ~y & mask
                fn_mask = ~p & y & mask
                tp_dists = (torch.where(tp_mask)[1] - torch.where(tp_mask)[0]).tolist() if tp_mask.any() else []
                fp_dists = (torch.where(fp_mask)[1] - torch.where(fp_mask)[0]).tolist() if fp_mask.any() else []
                fn_dists = (torch.where(fn_mask)[1] - torch.where(fn_mask)[0]).tolist() if fn_mask.any() else []

                # Shifted FP
                shifted_fp = 0
                if fp > 0 and gt_pairs > 0:
                    pred_set = set(zip(*[t.tolist() for t in torch.where(p & mask)]))
                    gt_set = set(zip(*[t.tolist() for t in torch.where(y & mask)]))
                    for pi, pj in pred_set:
                        if (pi, pj) not in gt_set:
                            is_shifted = False
                            for di in range(-1, 2):
                                for dj in range(-1, 2):
                                    if (di, dj) == (0, 0):
                                        continue
                                    if (pi + di, pj + dj) in gt_set:
                                        is_shifted = True
                                        break
                                if is_shifted:
                                    break
                            if is_shifted:
                                shifted_fp += 1

                name = names[i]
                family = get_family(name)

                per_sample.append({
                    'name': name,
                    'family': family,
                    'length': length,
                    'gt_pairs': gt_pairs,
                    'pred_pairs': pred_pairs,
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'mcc': mcc,
                    'shifted_fp': shifted_fp,
                    'pred_gt_ratio': pred_pairs / max(gt_pairs, 1),
                    'gt_distances': gt_distances,
                    'tp_distances': tp_dists,
                    'fp_distances': fp_dists,
                    'fn_distances': fn_dists,
                    'mean_gt_distance': np.mean(gt_distances) if gt_distances else 0,
                    'max_gt_distance': max(gt_distances) if gt_distances else 0,
                })

    return per_sample


def analyze_by_length(per_sample):
    """Analyze metrics by sequence length bins."""
    bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 400), (400, 500)]
    results = {}
    for lo, hi in bins:
        samples = [s for s in per_sample if lo <= s['length'] < hi]
        if not samples:
            continue
        results[f'{lo}-{hi}'] = {
            'n': len(samples),
            'f1_mean': np.mean([s['f1'] for s in samples]),
            'f1_median': np.median([s['f1'] for s in samples]),
            'precision': np.mean([s['precision'] for s in samples]),
            'recall': np.mean([s['recall'] for s in samples]),
            'mcc': np.mean([s['mcc'] for s in samples]),
            'bad_rate': sum(1 for s in samples if s['f1'] < 0.3) / len(samples),
        }
    return results


def analyze_by_family(per_sample):
    """Analyze metrics by RNA family/source."""
    family_groups = defaultdict(list)
    for s in per_sample:
        family_groups[s['family']].append(s)

    results = {}
    for family, samples in sorted(family_groups.items(), key=lambda x: -len(x[1])):
        results[family] = {
            'n': len(samples),
            'f1_mean': np.mean([s['f1'] for s in samples]),
            'f1_median': np.median([s['f1'] for s in samples]),
            'precision': np.mean([s['precision'] for s in samples]),
            'recall': np.mean([s['recall'] for s in samples]),
            'mcc': np.mean([s['mcc'] for s in samples]),
            'avg_length': np.mean([s['length'] for s in samples]),
            'bad_rate': sum(1 for s in samples if s['f1'] < 0.3) / len(samples),
        }
    return results


def analyze_by_pair_distance(per_sample):
    """Analyze performance by base pair distance (|i-j|)."""
    dist_bins = [(3, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500)]

    # Aggregate all TP/FP/FN by distance bin
    results = {}
    for lo, hi in dist_bins:
        tp_total = 0
        fp_total = 0
        fn_total = 0
        for s in per_sample:
            tp_total += sum(1 for d in s['tp_distances'] if lo <= d < hi)
            fp_total += sum(1 for d in s['fp_distances'] if lo <= d < hi)
            fn_total += sum(1 for d in s['fn_distances'] if lo <= d < hi)

        gt_total = tp_total + fn_total
        pred_total = tp_total + fp_total
        precision = tp_total / max(pred_total, 1)
        recall = tp_total / max(gt_total, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        results[f'{lo}-{hi}'] = {
            'gt_pairs': gt_total,
            'pred_pairs': pred_total,
            'tp': tp_total,
            'fp': fp_total,
            'fn': fn_total,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    return results


def generate_report(per_sample, length_analysis, family_analysis, distance_analysis, elapsed):
    """Generate comprehensive markdown report."""
    lines = []
    lines.append('# v10 DensityNet-Pro+ 全面测试分析报告')
    lines.append('')
    lines.append(f'> 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'> Checkpoint: `{CKPT_PATH}`')
    lines.append(f'> 评估耗时: {elapsed:.1f}s')
    lines.append('')

    # ========== 1. Overall ==========
    n = len(per_sample)
    f1s = [s['f1'] for s in per_sample]
    lines.append('## 1. 总体指标')
    lines.append('')
    lines.append('| 指标 | 值 |')
    lines.append('|------|-----|')
    lines.append(f'| **Test F1 (mean)** | **{np.mean(f1s):.4f}** |')
    lines.append(f'| Test F1 (median) | {np.median(f1s):.4f} |')
    lines.append(f'| Precision | {np.mean([s["precision"] for s in per_sample]):.4f} |')
    lines.append(f'| Recall | {np.mean([s["recall"] for s in per_sample]):.4f} |')
    lines.append(f'| MCC | {np.mean([s["mcc"] for s in per_sample]):.4f} |')
    lines.append(f'| 样本数 | {n} |')
    lines.append(f'| 平均长度 | {np.mean([s["length"] for s in per_sample]):.1f} |')
    lines.append(f'| F1 Std | {np.std(f1s):.4f} |')
    lines.append(f'| F1 < 0.3 (bad cases) | {sum(1 for f in f1s if f < 0.3)} ({sum(1 for f in f1s if f < 0.3)/n:.1%}) |')
    lines.append(f'| F1 < 0.5 | {sum(1 for f in f1s if f < 0.5)} ({sum(1 for f in f1s if f < 0.5)/n:.1%}) |')
    lines.append(f'| F1 > 0.9 | {sum(1 for f in f1s if f > 0.9)} ({sum(1 for f in f1s if f > 0.9)/n:.1%}) |')
    lines.append('')

    # ========== 2. By Length ==========
    lines.append('## 2. 按序列长度分析')
    lines.append('')
    lines.append('| 长度区间 | N | F1 Mean | F1 Median | Precision | Recall | MCC | Bad Rate |')
    lines.append('|----------|---|---------|-----------|-----------|--------|-----|----------|')
    for bin_name, stats in length_analysis.items():
        lines.append(f'| {bin_name} | {stats["n"]} | {stats["f1_mean"]:.4f} | '
                     f'{stats["f1_median"]:.4f} | {stats["precision"]:.4f} | '
                     f'{stats["recall"]:.4f} | {stats["mcc"]:.4f} | {stats["bad_rate"]:.1%} |')
    lines.append('')
    lines.append('**观察**：')
    # Find worst length bin
    worst_bin = min(length_analysis.items(), key=lambda x: x[1]['f1_mean'])
    best_bin = max(length_analysis.items(), key=lambda x: x[1]['f1_mean'])
    lines.append(f'- 最佳长度区间：{best_bin[0]}（F1={best_bin[1]["f1_mean"]:.4f}）')
    lines.append(f'- 最差长度区间：{worst_bin[0]}（F1={worst_bin[1]["f1_mean"]:.4f}）')
    lines.append('')

    # ========== 3. By Family ==========
    lines.append('## 3. 按 RNA 家族/来源分析')
    lines.append('')
    lines.append('| 家族 | N | F1 Mean | F1 Median | Precision | Recall | Avg Len | Bad Rate |')
    lines.append('|------|---|---------|-----------|-----------|--------|---------|----------|')
    for family, stats in family_analysis.items():
        lines.append(f'| {family} | {stats["n"]} | {stats["f1_mean"]:.4f} | '
                     f'{stats["f1_median"]:.4f} | {stats["precision"]:.4f} | '
                     f'{stats["recall"]:.4f} | {stats["avg_length"]:.0f} | {stats["bad_rate"]:.1%} |')
    lines.append('')
    lines.append('**观察**：')
    worst_family = min(family_analysis.items(), key=lambda x: x[1]['f1_mean'])
    best_family = max(family_analysis.items(), key=lambda x: x[1]['f1_mean'])
    lines.append(f'- 最佳家族：{best_family[0]}（F1={best_family[1]["f1_mean"]:.4f}, N={best_family[1]["n"]}）')
    lines.append(f'- 最差家族：{worst_family[0]}（F1={worst_family[1]["f1_mean"]:.4f}, N={worst_family[1]["n"]}）')
    lines.append('')

    # ========== 4. By Pair Distance ==========
    lines.append('## 4. 按配对距离 |i-j| 分析')
    lines.append('')
    lines.append('配对距离 = 两个配对碱基在序列上的距离。短距离配对（茎环）通常较容易预测，长距离配对（假结、远程相互作用）较难。')
    lines.append('')
    lines.append('| 距离区间 | GT Pairs | Pred Pairs | TP | FP | FN | Precision | Recall | F1 |')
    lines.append('|----------|----------|------------|----|----|------|-----------|--------|-----|')
    for bin_name, stats in distance_analysis.items():
        lines.append(f'| {bin_name} | {stats["gt_pairs"]} | {stats["pred_pairs"]} | '
                     f'{stats["tp"]} | {stats["fp"]} | {stats["fn"]} | '
                     f'{stats["precision"]:.4f} | {stats["recall"]:.4f} | {stats["f1"]:.4f} |')
    lines.append('')
    lines.append('**观察**：')
    worst_dist = min(distance_analysis.items(), key=lambda x: x[1]['f1'])
    best_dist = max(distance_analysis.items(), key=lambda x: x[1]['f1'])
    lines.append(f'- 最佳距离区间：{best_dist[0]}（F1={best_dist[1]["f1"]:.4f}）')
    lines.append(f'- 最差距离区间：{worst_dist[0]}（F1={worst_dist[1]["f1"]:.4f}）')
    lines.append('')

    # ========== 5. Bad Cases Analysis ==========
    lines.append('## 5. Bad Cases 分析 (F1 < 0.3)')
    lines.append('')
    bad_cases = sorted([s for s in per_sample if s['f1'] < 0.3], key=lambda x: x['f1'])
    lines.append(f'共 {len(bad_cases)} 个 bad cases：')
    lines.append('')

    # Bad case breakdown by family
    bad_family = defaultdict(int)
    for s in bad_cases:
        bad_family[s['family']] += 1
    lines.append('**Bad cases 家族分布**：')
    lines.append('')
    for fam, cnt in sorted(bad_family.items(), key=lambda x: -x[1]):
        total_in_fam = sum(1 for s in per_sample if s['family'] == fam)
        lines.append(f'- {fam}: {cnt}/{total_in_fam} ({cnt/total_in_fam:.1%})')
    lines.append('')

    # Bad case breakdown by length
    lines.append('**Bad cases 长度分布**：')
    lines.append('')
    bad_lens = [s['length'] for s in bad_cases]
    lines.append(f'- 平均长度: {np.mean(bad_lens):.0f} (全体平均: {np.mean([s["length"] for s in per_sample]):.0f})')
    lines.append(f'- 长度范围: {min(bad_lens)} ~ {max(bad_lens)}')
    lines.append('')

    # Bad case characteristics
    lines.append('**Bad cases 特征分析**：')
    lines.append('')
    bad_over_pred = sum(1 for s in bad_cases if s['pred_gt_ratio'] > 1.5)
    bad_under_pred = sum(1 for s in bad_cases if s['pred_gt_ratio'] < 0.5)
    bad_normal_pred = len(bad_cases) - bad_over_pred - bad_under_pred
    lines.append(f'- 过度预测 (pred/gt > 1.5): {bad_over_pred} ({bad_over_pred/max(len(bad_cases),1):.1%})')
    lines.append(f'- 预测不足 (pred/gt < 0.5): {bad_under_pred} ({bad_under_pred/max(len(bad_cases),1):.1%})')
    lines.append(f'- 数量接近但位置错误: {bad_normal_pred} ({bad_normal_pred/max(len(bad_cases),1):.1%})')
    lines.append('')

    # Top 20 worst
    lines.append('**最差 20 个样本**：')
    lines.append('')
    lines.append('| # | Name | Family | Len | GT | Pred | F1 | Prec | Rec | Pred/GT | Avg Dist |')
    lines.append('|---|------|--------|-----|-----|------|-----|------|------|---------|----------|')
    for i, s in enumerate(bad_cases[:20]):
        lines.append(f'| {i+1} | {s["name"][:25]} | {s["family"]} | {s["length"]} | '
                     f'{s["gt_pairs"]} | {s["pred_pairs"]} | {s["f1"]:.3f} | '
                     f'{s["precision"]:.3f} | {s["recall"]:.3f} | '
                     f'{s["pred_gt_ratio"]:.2f} | {s["mean_gt_distance"]:.0f} |')
    lines.append('')

    # ========== 6. Shifted FP Analysis ==========
    lines.append('## 6. Shifted Prediction 分析')
    lines.append('')
    total_fp = sum(s['fp'] for s in per_sample)
    total_shifted = sum(s['shifted_fp'] for s in per_sample)
    lines.append(f'| 指标 | 值 |')
    lines.append(f'|------|-----|')
    lines.append(f'| 总 FP | {total_fp} |')
    lines.append(f'| Shifted FP (±1) | {total_shifted} |')
    lines.append(f'| Shifted FP 占比 | {total_shifted/max(total_fp,1):.1%} |')
    lines.append('')

    # ========== 7. Top performers ==========
    lines.append('## 7. 最佳 10 个样本')
    lines.append('')
    best_samples = sorted(per_sample, key=lambda x: -x['f1'])[:10]
    lines.append('| # | Name | Family | Len | GT | Pred | F1 | Prec | Rec |')
    lines.append('|---|------|--------|-----|-----|------|-----|------|------|')
    for i, s in enumerate(best_samples):
        lines.append(f'| {i+1} | {s["name"][:25]} | {s["family"]} | {s["length"]} | '
                     f'{s["gt_pairs"]} | {s["pred_pairs"]} | {s["f1"]:.3f} | '
                     f'{s["precision"]:.3f} | {s["recall"]:.3f} |')
    lines.append('')

    # ========== 8. Key Insights ==========
    lines.append('## 8. 关键发现与改进方向')
    lines.append('')

    # Auto-generate insights
    # Length insight
    if length_analysis:
        long_bins = {k: v for k, v in length_analysis.items() if int(k.split('-')[0]) >= 200}
        short_bins = {k: v for k, v in length_analysis.items() if int(k.split('-')[1]) <= 100}
        if long_bins and short_bins:
            long_f1 = np.mean([v['f1_mean'] for v in long_bins.values()])
            short_f1 = np.mean([v['f1_mean'] for v in short_bins.values()])
            lines.append(f'1. **长度效应**: 短序列(<100) F1={short_f1:.4f} vs 长序列(≥200) F1={long_f1:.4f}，'
                         f'差距 {short_f1 - long_f1:+.4f}')

    # Distance insight
    if distance_analysis:
        short_dist = [v for k, v in distance_analysis.items() if int(k.split('-')[0]) < 50]
        long_dist = [v for k, v in distance_analysis.items() if int(k.split('-')[0]) >= 100]
        if short_dist and long_dist:
            short_d_f1 = np.mean([v['f1'] for v in short_dist])
            long_d_f1 = np.mean([v['f1'] for v in long_dist])
            lines.append(f'2. **配对距离效应**: 近距离配对(<50) F1={short_d_f1:.4f} vs 远距离配对(≥100) F1={long_d_f1:.4f}，'
                         f'差距 {short_d_f1 - long_d_f1:+.4f}')

    # Family insight
    if family_analysis:
        lines.append(f'3. **家族差异**: 最强 {best_family[0]}(F1={best_family[1]["f1_mean"]:.4f}) vs '
                     f'最弱 {worst_family[0]}(F1={worst_family[1]["f1_mean"]:.4f})')

    # Shifted FP insight
    if total_fp > 0:
        lines.append(f'4. **位置偏移**: {total_shifted/total_fp:.1%} 的 FP 是 ±1 偏移，说明模型感知到配对但定位不精确')

    lines.append('')
    return '\n'.join(lines)


def main():
    print('=' * 60)
    print('  v10 Comprehensive Test Analysis')
    print('=' * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    device = torch.device(DEVICE)
    print(f'Device: {device}')
    print(f'Checkpoint: {CKPT_PATH}')

    # Load LM extractor
    class Args:
        pass
    lm_args = Args()
    lm_args.pretrained_lm_dir = config['paths'].get('pretrained_lm_dir', str(ROOT / 'model'))
    lm_args.model_scale = config['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)

    # Build model
    model = build_model(config, extractor).to(device)

    # Load checkpoint
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    if 'model' in ckpt and isinstance(ckpt['model'], dict):
        state_dict = ckpt['model']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
    new_state_dict = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f'Model loaded. Epoch: {ckpt.get("epoch", "?")}')
    if missing:
        print(f'  WARNING: {len(missing)} missing keys')
    if unexpected:
        print(f'  WARNING: {len(unexpected)} unexpected keys')

    model.eval()

    # Build test loader
    tcfg = config['training']
    data_dir = config['paths']['data_dir']
    max_len = tcfg.get('max_len_filter', 490)
    records = build_records(data_dir, 'bprna-test', max_len=max_len)
    dataset = PriFoldSymFlowDataset(records, augment=False)
    collate_fn = make_collate_fn(tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                        num_workers=4, pin_memory=True)
    print(f'Test samples: {len(dataset)}')
    print()

    # Run evaluation
    t0 = time.time()
    per_sample = evaluate_comprehensive(model, loader, device, config)
    elapsed = time.time() - t0
    print(f'\nEvaluation done in {elapsed:.1f}s')

    # Analysis
    print('\nAnalyzing by length...')
    length_analysis = analyze_by_length(per_sample)

    print('Analyzing by family...')
    family_analysis = analyze_by_family(per_sample)

    print('Analyzing by pair distance...')
    distance_analysis = analyze_by_pair_distance(per_sample)

    # Generate report
    print('Generating report...')
    report = generate_report(per_sample, length_analysis, family_analysis, distance_analysis, elapsed)

    report_path = OUTPUT_DIR / 'v10_comprehensive_analysis.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'Report saved: {report_path}')

    # Save raw per-sample results (without distance lists for size)
    per_sample_slim = []
    for s in per_sample:
        slim = {k: v for k, v in s.items()
                if k not in ('gt_distances', 'tp_distances', 'fp_distances', 'fn_distances')}
        per_sample_slim.append(slim)

    per_sample_path = OUTPUT_DIR / 'per_sample_results.json'
    with open(per_sample_path, 'w') as f:
        json.dump(per_sample_slim, f, indent=2)
    print(f'Per-sample results saved: {per_sample_path}')

    # Print quick summary
    f1s = [s['f1'] for s in per_sample]
    print(f'\n{"="*60}')
    print(f'  SUMMARY')
    print(f'{"="*60}')
    print(f'  Test F1: {np.mean(f1s):.4f} (median {np.median(f1s):.4f})')
    print(f'  Precision: {np.mean([s["precision"] for s in per_sample]):.4f}')
    print(f'  Recall: {np.mean([s["recall"] for s in per_sample]):.4f}')
    print(f'  Bad cases (F1<0.3): {sum(1 for f in f1s if f < 0.3)}/{len(f1s)}')
    print(f'  Report: {report_path}')
    print()


if __name__ == '__main__':
    main()
