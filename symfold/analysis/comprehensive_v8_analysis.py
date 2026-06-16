# -*- coding: utf-8 -*-
"""PriFold v8 DensityNet-Pro: 全面 Bad Case 分析脚本。

对 bprna-test 数据集做完整推理，深度分析 bad cases，
生成全面的分析报告和可视化。

分析内容：
1. 总体性能指标统计
2. Bad case (F1 < 0.3) 的全面分析
3. v8 新特性（OHEM/FP Penalty/Length Decay/BP Compat）的效果评估
4. 失败模式分类与根因分析
5. 与 v7 对比的关键差异
6. 具体 bad case 可视化卡片

输出：
- bad_cases/ 文件夹：每个 bad case 的 contact map + GT + 标注
- 分析图表集合
- v8_bad_case_analysis_report.md

Usage:
  CUDA_VISIBLE_DEVICES=1 python symfold/analysis/comprehensive_v8_analysis.py \
    --ckpt symfold/outputs/v8_full/model/best.pt \
    --config symfold/config/v8/v8_full.json \
    --out_dir symfold/outputs/v8_full/comprehensive_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.labelsize'] = 9

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_loader
from symfold.train.train_v8 import build_model, load_config, move_to_device


# ============================================================
# Helpers
# ============================================================

VALID_PAIRS = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')}


def detect_pseudoknots(contact_matrix, length):
    """Detect pseudoknots in contact map."""
    pairs = []
    for i in range(length):
        for j in range(i + 3, length):
            if contact_matrix[i, j] > 0.5:
                pairs.append((i, j))

    crossings = 0
    crossing_pairs = []
    for idx1 in range(len(pairs)):
        i, j = pairs[idx1]
        for idx2 in range(idx1 + 1, len(pairs)):
            k, l = pairs[idx2]
            if i < k < j < l:
                crossings += 1
                crossing_pairs.append(((i, j), (k, l)))

    return crossings, crossing_pairs


def compute_structure_complexity(contact_matrix, length):
    """Compute structure complexity metrics."""
    pairs = set()
    for i in range(length):
        for j in range(i + 3, length):
            if contact_matrix[i, j] > 0.5:
                pairs.add((i, j))

    # Find stems
    stems = []
    used = set()
    sorted_pairs = sorted(pairs)
    for (i, j) in sorted_pairs:
        if (i, j) in used:
            continue
        stem = [(i, j)]
        used.add((i, j))
        ci, cj = i + 1, j - 1
        while (ci, cj) in pairs and ci < cj and (ci, cj) not in used:
            stem.append((ci, cj))
            used.add((ci, cj))
            ci += 1
            cj -= 1
        if len(stem) >= 1:
            stems.append(stem)

    n_stems = len(stems)
    stem_lengths = [len(s) for s in stems]
    max_stem_len = max(stem_lengths) if stem_lengths else 0
    avg_stem_len = float(np.mean(stem_lengths)) if stem_lengths else 0

    branching = len(set(s[0][0] for s in stems)) if stems else 0

    return {
        'n_stems': n_stems,
        'max_stem_length': max_stem_len,
        'avg_stem_length': avg_stem_len,
        'n_pairs': len(pairs),
        'branching_factor': branching,
    }


def compute_pairing_distances(contact_matrix, length):
    """Compute distribution of pairing distances |j-i| for all pairs."""
    distances = []
    for i in range(length):
        for j in range(i + 3, length):
            if contact_matrix[i, j] > 0.5:
                distances.append(j - i)
    return distances


def check_bp_compatibility(seq, contact_matrix, length):
    """Check how many pairs are canonical (AU/GC/GU) vs non-canonical."""
    canonical = 0
    non_canonical = 0
    non_canonical_types = Counter()
    for i in range(length):
        for j in range(i + 3, length):
            if contact_matrix[i, j] > 0.5:
                pair = (seq[i].upper(), seq[j].upper())
                if pair in VALID_PAIRS:
                    canonical += 1
                else:
                    non_canonical += 1
                    non_canonical_types[pair] += 1
    return canonical, non_canonical, non_canonical_types


def per_sample_metrics(pred, target, length, seq=None, score=None):
    """Compute comprehensive per-sample metrics for v8 analysis."""
    p = pred.detach().cpu().float().squeeze()[:length, :length]
    y = target.detach().cpu().float().squeeze()[:length, :length]

    p_bin = p > 0.5
    y_bin = y > 0.5

    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    idx = torch.arange(length)
    mask &= (idx.view(length, 1) - idx.view(1, length)).abs() >= 3

    p_m = p_bin[mask]
    y_m = y_bin[mask]
    tp = int((p_m & y_m).sum())
    fp = int((p_m & ~y_m).sum())
    fn = int((~p_m & y_m).sum())
    tn = int((~p_m & ~y_m).sum())

    gt_pairs = tp + fn
    pred_pairs = tp + fp

    if gt_pairs == 0 and pred_pairs == 0:
        prec, rec, f1, mcc = 1.0, 1.0, 1.0, 1.0
    elif gt_pairs == 0 and pred_pairs > 0:
        prec, rec, f1, mcc = 0.0, 1.0, 0.0, 0.0
    else:
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
        mcc = ((tp * tn) - (fp * fn)) / denom

    density = gt_pairs / max(length, 1)

    p_np = p_bin.numpy().astype(float)
    y_np = y_bin.numpy().astype(float)

    # Pairing distances
    gt_dists = compute_pairing_distances(y_np, length)
    pred_dists = compute_pairing_distances(p_np, length)

    # Structure complexity
    gt_complexity = compute_structure_complexity(y_np, length)
    pred_complexity = compute_structure_complexity(p_np, length)

    # Pseudoknots
    gt_pk_count, _ = detect_pseudoknots(y_np, length)
    pred_pk_count, _ = detect_pseudoknots(p_np, length)

    # BP compatibility
    gt_canonical, gt_noncanonical, gt_nc_types = 0, 0, Counter()
    pred_canonical, pred_noncanonical, pred_nc_types = 0, 0, Counter()
    if seq and len(seq) >= length:
        gt_canonical, gt_noncanonical, gt_nc_types = check_bp_compatibility(seq, y_np, length)
        pred_canonical, pred_noncanonical, pred_nc_types = check_bp_compatibility(seq, p_np, length)

    # Shift analysis (near-miss FP within ±1, ±2, ±3)
    near_miss_1 = 0
    near_miss_2 = 0
    near_miss_3 = 0
    far_miss = 0
    for i in range(length):
        for j in range(i + 3, length):
            if p_np[i, j] > 0.5 and y_np[i, j] < 0.5:
                found_r = 0
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < length and 0 <= nj < length and y_np[ni, nj] > 0.5:
                            dist = max(abs(di), abs(dj))
                            if found_r == 0 or dist < found_r:
                                found_r = dist
                if found_r == 1:
                    near_miss_1 += 1
                elif found_r == 2:
                    near_miss_2 += 1
                elif found_r == 3:
                    near_miss_3 += 1
                else:
                    far_miss += 1

    # Score statistics (how confident were the predictions)
    score_stats = {}
    if score is not None:
        s_upper = score[mask.numpy()]
        if tp + fp > 0:
            pred_scores = score[p_np > 0.5]
            score_stats['pred_mean_score'] = float(np.mean(pred_scores)) if len(pred_scores) > 0 else 0
            score_stats['pred_min_score'] = float(np.min(pred_scores)) if len(pred_scores) > 0 else 0
            score_stats['pred_median_score'] = float(np.median(pred_scores)) if len(pred_scores) > 0 else 0
        # Scores at GT positions
        if gt_pairs > 0:
            gt_scores = score[y_np > 0.5]
            score_stats['gt_pos_mean_score'] = float(np.mean(gt_scores)) if len(gt_scores) > 0 else 0
            score_stats['gt_pos_min_score'] = float(np.min(gt_scores)) if len(gt_scores) > 0 else 0
            # Missed GT positions
            missed_mask = (y_np > 0.5) & (p_np < 0.5)
            missed_scores = score[missed_mask]
            score_stats['missed_gt_mean_score'] = float(np.mean(missed_scores)) if len(missed_scores) > 0 else 0
            score_stats['missed_gt_max_score'] = float(np.max(missed_scores)) if len(missed_scores) > 0 else 0

    # Length-decay impact analysis
    length_factor = (100.0 / max(length, 50)) ** 0.3

    result = {
        'precision': prec, 'recall': rec, 'f1': f1, 'mcc': mcc,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'gt_pairs': gt_pairs, 'pred_pairs': pred_pairs,
        'density': density,
        'pred_gt_ratio': pred_pairs / max(gt_pairs, 1),
        # Distance stats
        'gt_mean_dist': float(np.mean(gt_dists)) if gt_dists else 0,
        'gt_max_dist': max(gt_dists) if gt_dists else 0,
        'gt_median_dist': float(np.median(gt_dists)) if gt_dists else 0,
        'pred_mean_dist': float(np.mean(pred_dists)) if pred_dists else 0,
        'pred_max_dist': max(pred_dists) if pred_dists else 0,
        # Complexity
        'gt_n_stems': gt_complexity['n_stems'],
        'gt_max_stem_len': gt_complexity['max_stem_length'],
        'gt_avg_stem_len': gt_complexity['avg_stem_length'],
        'gt_branching': gt_complexity['branching_factor'],
        'pred_n_stems': pred_complexity['n_stems'],
        # Pseudoknots
        'gt_pseudoknots': gt_pk_count,
        'pred_pseudoknots': pred_pk_count,
        'has_pseudoknot': gt_pk_count > 0,
        # BP compatibility
        'gt_canonical_pairs': gt_canonical,
        'gt_noncanonical_pairs': gt_noncanonical,
        'pred_canonical_pairs': pred_canonical,
        'pred_noncanonical_pairs': pred_noncanonical,
        'gt_nc_types': dict(gt_nc_types),
        'pred_nc_types': dict(pred_nc_types),
        # Shift analysis (multi-level)
        'near_miss_1': near_miss_1,
        'near_miss_2': near_miss_2,
        'near_miss_3': near_miss_3,
        'far_miss_fp': far_miss,
        'total_fp_analyzed': near_miss_1 + near_miss_2 + near_miss_3 + far_miss,
        'near_miss_pct': (near_miss_1 + near_miss_2 + near_miss_3) / max(fp, 1),
        'shift_1_pct': near_miss_1 / max(fp, 1),
        # Score stats
        **score_stats,
        # Length decay
        'length_factor': length_factor,
        # Raw data for visualization
        'gt_dists': gt_dists,
        'pred_dists': pred_dists,
    }
    return result


def classify_failure_mode(r):
    """Classify a bad case into failure mode (v8-specific categories)."""
    if r['f1'] == 0 and r['tp'] == 0:
        if r['pred_pairs'] == 0:
            return 'no_prediction'
        elif r['near_miss_pct'] > 0.3:
            return 'shifted_prediction'
        else:
            return 'complete_miss'
    elif r['pred_gt_ratio'] > 2.0:
        return 'severe_overpredict'
    elif r['pred_gt_ratio'] < 0.3:
        return 'severe_underpredict'
    elif r['recall'] < 0.2 and r['precision'] > 0.5:
        return 'budget_too_tight'  # v8-specific: length decay 过紧
    elif r['precision'] < 0.2 and r['recall'] > 0.5:
        return 'budget_too_loose'  # v8-specific: 阈值过松
    elif 0.7 < r['pred_gt_ratio'] < 1.3 and r['f1'] < 0.3:
        if r['near_miss_pct'] > 0.3:
            return 'shifted_prediction'
        else:
            return 'wrong_position'
    elif r['has_pseudoknot'] and r['f1'] < 0.4:
        return 'pseudoknot_failure'
    elif r.get('pred_noncanonical_pairs', 0) > 0 and r['precision'] < 0.4:
        return 'bp_compat_issue'
    else:
        return 'mixed'


# ============================================================
# Inference
# ============================================================

def run_inference(model, loader, device, config, seq_map=None):
    """Run inference on all samples, return per-sample results."""
    model.eval()
    scfg = config.get('sampling', {})
    amp_name = str(config.get('training', {}).get('amp_dtype', 'fp32')).lower()
    amp_on = amp_name in ('bf16', 'bfloat16', 'fp16', 'float16')
    amp_dtype = torch.bfloat16 if amp_name in ('bf16', 'bfloat16') else torch.float16

    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = move_to_device(batch, device)
            if amp_on:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    pred, extras = model.predict(
                        batch,
                        budget_fraction=scfg.get('default_budget_fraction', 0.30),
                        use_density_budget=scfg.get('use_density_budget', True),
                        score_threshold=scfg.get('score_threshold', 0.45),
                        length_decay=scfg.get('length_decay', 0.3),
                    )
            else:
                pred, extras = model.predict(
                    batch,
                    budget_fraction=scfg.get('default_budget_fraction', 0.30),
                    use_density_budget=scfg.get('use_density_budget', True),
                    score_threshold=scfg.get('score_threshold', 0.45),
                    length_decay=scfg.get('length_decay', 0.3),
                )

            bs = pred.shape[0]
            for i in range(bs):
                length = int(batch['length'][i])
                name = batch['name'][i] if 'name' in batch else f'sample_{batch_idx}_{i}'
                seq = seq_map.get(name, '') if seq_map else ''

                # Get score map
                score = None
                if isinstance(extras, torch.Tensor):
                    score = extras[i].detach().cpu().float().squeeze()[:length, :length].numpy()
                elif isinstance(extras, dict) and 'score' in extras:
                    score = extras['score'][i].detach().cpu().float().squeeze()[:length, :length].numpy()

                metrics = per_sample_metrics(pred[i], batch['contact'][i], length, seq=seq, score=score)
                metrics['name'] = name
                metrics['length'] = length
                metrics['seq'] = seq

                # Store raw maps
                metrics['pred_map'] = pred[i].detach().cpu().float().squeeze()[:length, :length].numpy()
                metrics['gt_map'] = batch['contact'][i].detach().cpu().float().squeeze()[:length, :length].numpy()
                if score is not None:
                    metrics['score_map'] = score

                results.append(metrics)

            if (batch_idx + 1) % 20 == 0:
                print(f'    Batch {batch_idx + 1}/{len(loader)} done ({len(results)} samples)')

    return results


# ============================================================
# Visualization Functions
# ============================================================

def plot_bad_case_card(case, out_path):
    """Plot a single bad case card with contact map, GT, diff, score, and annotations."""
    L = case['length']
    has_score = 'score_map' in case

    n_cols = 5 if has_score else 4
    width_ratios = [1, 1, 1, 1, 0.9] if has_score else [1, 1, 1, 0.9]
    fig = plt.figure(figsize=(3.5 * n_cols, 5))
    gs = gridspec.GridSpec(1, n_cols, width_ratios=width_ratios, wspace=0.3)

    # GT contact map
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(case['gt_map'], cmap='Blues', vmin=0, vmax=1, aspect='equal')
    ax1.set_title('Ground Truth', fontsize=10)
    ax1.set_xlabel(f'L={L}')

    # Predicted contact map
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(case['pred_map'], cmap='Oranges', vmin=0, vmax=1, aspect='equal')
    ax2.set_title('Prediction', fontsize=10)

    # Diff map
    ax3 = fig.add_subplot(gs[2])
    diff = np.zeros((L, L, 3))
    p_bin = case['pred_map'] > 0.5
    g_bin = case['gt_map'] > 0.5
    diff[p_bin & g_bin] = [0.2, 0.8, 0.2]
    diff[p_bin & ~g_bin] = [0.9, 0.2, 0.2]
    diff[~p_bin & g_bin] = [0.2, 0.4, 0.9]
    ax3.imshow(diff, aspect='equal')
    ax3.set_title('TP(green)/FP(red)/FN(blue)', fontsize=9)
    legend_elements = [
        Patch(facecolor=[0.2, 0.8, 0.2], label=f'TP={case["tp"]}'),
        Patch(facecolor=[0.9, 0.2, 0.2], label=f'FP={case["fp"]}'),
        Patch(facecolor=[0.2, 0.4, 0.9], label=f'FN={case["fn"]}'),
    ]
    ax3.legend(handles=legend_elements, loc='lower right', fontsize=7)

    # Score heatmap
    if has_score:
        ax4 = fig.add_subplot(gs[3])
        im = ax4.imshow(case['score_map'], cmap='hot', vmin=0, vmax=1, aspect='equal')
        ax4.set_title('Score Heatmap', fontsize=10)
        plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # Annotation panel
    ax_info = fig.add_subplot(gs[-1])
    ax_info.axis('off')
    failure_mode = classify_failure_mode(case)
    info_text = (
        f"RNA: {case['name']}\n"
        f"Length: {L}\n"
        f"{'─' * 22}\n"
        f"F1: {case['f1']:.4f}\n"
        f"Precision: {case['precision']:.4f}\n"
        f"Recall: {case['recall']:.4f}\n"
        f"MCC: {case['mcc']:.4f}\n"
        f"{'─' * 22}\n"
        f"GT pairs: {case['gt_pairs']}\n"
        f"Pred pairs: {case['pred_pairs']}\n"
        f"pred/gt: {case['pred_gt_ratio']:.2f}\n"
        f"Density: {case['density']:.4f}\n"
        f"Length factor: {case['length_factor']:.3f}\n"
        f"{'─' * 22}\n"
        f"GT stems: {case['gt_n_stems']}\n"
        f"GT pseudoknots: {case['gt_pseudoknots']}\n"
        f"{'─' * 22}\n"
        f"Shift ±1: {case['near_miss_1']}\n"
        f"Shift ±2: {case['near_miss_2']}\n"
        f"Shift ±3: {case['near_miss_3']}\n"
        f"Far FP: {case['far_miss_fp']}\n"
        f"{'─' * 22}\n"
        f"Non-canonical pred: {case.get('pred_noncanonical_pairs', 0)}\n"
        f"{'─' * 22}\n"
        f"Failure mode:\n  {failure_mode}"
    )
    ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                 fontsize=7.5, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(f'{case["name"]}  |  F1={case["f1"]:.4f}  |  L={L}  |  Mode: {failure_mode}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_overall_performance(results, out_dir):
    """Plot overall F1/Precision/Recall distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('v8 DensityNet-Pro: Test Set Overall Performance', fontsize=14, fontweight='bold')

    metrics_plot = [
        ('f1', 'F1 Score', '#2196F3'),
        ('precision', 'Precision', '#4CAF50'),
        ('recall', 'Recall', '#FF9800'),
        ('mcc', 'MCC', '#9C27B0'),
        ('pred_gt_ratio', 'Pred/GT Ratio', '#F44336'),
        ('density', 'Pairing Density', '#795548'),
    ]

    for idx, (key, label, color) in enumerate(metrics_plot):
        ax = axes[idx // 3, idx % 3]
        values = [r[key] for r in results]
        ax.hist(values, bins=30, color=color, alpha=0.7, edgecolor='white')
        mean_v = np.mean(values)
        median_v = np.median(values)
        ax.axvline(mean_v, color='black', linestyle='--', linewidth=1.5,
                   label=f'mean={mean_v:.3f}')
        ax.axvline(median_v, color='gray', linestyle=':', linewidth=1.5,
                   label=f'median={median_v:.3f}')
        ax.set_title(f'{label} (N={len(values)})', fontsize=11)
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / 'overall_performance.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_failure_mode_summary(bad_cases, out_dir):
    """Plot failure mode classification summary."""
    modes = Counter()
    for r in bad_cases:
        modes[classify_failure_mode(r)] += 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart
    labels = list(modes.keys())
    sizes = list(modes.values())
    colors_map = {
        'no_prediction': '#FF6B6B',
        'complete_miss': '#C0392B',
        'shifted_prediction': '#F39C12',
        'severe_overpredict': '#E74C3C',
        'severe_underpredict': '#3498DB',
        'budget_too_tight': '#9B59B6',
        'budget_too_loose': '#1ABC9C',
        'wrong_position': '#E67E22',
        'pseudoknot_failure': '#8E44AD',
        'bp_compat_issue': '#2ECC71',
        'mixed': '#95A5A6',
    }
    colors = [colors_map.get(l, '#BDC3C7') for l in labels]

    axes[0].pie(sizes, labels=[f'{l}\n({v})' for l, v in zip(labels, sizes)],
                colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 8})
    axes[0].set_title(f'Failure Mode Distribution (N={len(bad_cases)})', fontsize=11)

    # Bar chart
    sorted_modes = sorted(modes.items(), key=lambda x: -x[1])
    labels_s = [m[0] for m in sorted_modes]
    counts_s = [m[1] for m in sorted_modes]
    bars = axes[1].barh(labels_s, counts_s, color=[colors_map.get(l, '#BDC3C7') for l in labels_s])
    axes[1].set_xlabel('Count')
    axes[1].set_title('Failure Modes Ranked', fontsize=11)
    for bar, count in zip(bars, counts_s):
        axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     f'{count} ({count/len(bad_cases)*100:.1f}%)', va='center', fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / 'failure_mode_summary.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_length_decay_analysis(results, out_dir):
    """Analyze impact of length-aware budget on performance."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('v8 Length-Aware Budget Analysis', fontsize=14, fontweight='bold')

    lengths = [r['length'] for r in results]
    f1s = [r['f1'] for r in results]
    ratios = [r['pred_gt_ratio'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]

    # F1 vs Length
    ax = axes[0, 0]
    ax.scatter(lengths, f1s, alpha=0.3, s=10, c='#2196F3')
    # Binned average
    bins = np.arange(0, max(lengths) + 50, 50)
    for i in range(len(bins) - 1):
        mask = [(bins[i] <= l < bins[i+1]) for l in lengths]
        if sum(mask) > 3:
            bin_f1 = [f for f, m in zip(f1s, mask) if m]
            ax.plot((bins[i] + bins[i+1]) / 2, np.mean(bin_f1), 'ro', markersize=8)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('F1')
    ax.set_title('F1 vs Length (red=bin mean)')

    # pred/gt ratio vs Length
    ax = axes[0, 1]
    ax.scatter(lengths, ratios, alpha=0.3, s=10, c='#F44336')
    ax.axhline(1.0, color='green', linestyle='--', linewidth=1.5, label='ideal=1.0')
    ax.axhline(1.15, color='orange', linestyle=':', linewidth=1, label='threshold=1.15')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Pred/GT Ratio')
    ax.set_title('Pred/GT Ratio vs Length')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 3)

    # Precision vs Recall scatter colored by length
    ax = axes[1, 0]
    sc = ax.scatter(recalls, precisions, c=lengths, cmap='viridis', alpha=0.5, s=15)
    plt.colorbar(sc, ax=ax, label='Length')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision vs Recall (colored by Length)')

    # Length factor distribution
    ax = axes[1, 1]
    factors = [r['length_factor'] for r in results]
    ax.scatter(lengths, factors, alpha=0.4, s=10, c='#9C27B0')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Length Factor (100/L)^0.3')
    ax.set_title('Length Decay Factor vs Length')
    ax.axhline(1.0, color='green', linestyle='--', linewidth=1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / 'length_decay_analysis.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_shift_analysis(results, bad_cases, out_dir):
    """Analyze shift patterns in predictions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('v8 Shift-Aware Analysis', fontsize=14, fontweight='bold')

    # Shift breakdown for all samples
    ax = axes[0, 0]
    shift_1 = sum(r['near_miss_1'] for r in results)
    shift_2 = sum(r['near_miss_2'] for r in results)
    shift_3 = sum(r['near_miss_3'] for r in results)
    far = sum(r['far_miss_fp'] for r in results)
    total_fp = shift_1 + shift_2 + shift_3 + far
    categories = ['Shift ±1', 'Shift ±2', 'Shift ±3', 'Far FP']
    values = [shift_1, shift_2, shift_3, far]
    pcts = [v / max(total_fp, 1) * 100 for v in values]
    bars = ax.bar(categories, pcts, color=['#4CAF50', '#8BC34A', '#CDDC39', '#F44336'])
    ax.set_ylabel('% of Total FP')
    ax.set_title(f'FP Breakdown (Total FP={total_fp})')
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', fontsize=9)

    # Shift breakdown for bad cases
    ax = axes[0, 1]
    shift_1_bad = sum(r['near_miss_1'] for r in bad_cases)
    shift_2_bad = sum(r['near_miss_2'] for r in bad_cases)
    shift_3_bad = sum(r['near_miss_3'] for r in bad_cases)
    far_bad = sum(r['far_miss_fp'] for r in bad_cases)
    total_fp_bad = shift_1_bad + shift_2_bad + shift_3_bad + far_bad
    values_bad = [shift_1_bad, shift_2_bad, shift_3_bad, far_bad]
    pcts_bad = [v / max(total_fp_bad, 1) * 100 for v in values_bad]
    bars = ax.bar(categories, pcts_bad, color=['#4CAF50', '#8BC34A', '#CDDC39', '#F44336'])
    ax.set_ylabel('% of Total FP')
    ax.set_title(f'Bad Cases FP Breakdown (Total FP={total_fp_bad})')
    for bar, pct in zip(bars, pcts_bad):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', fontsize=9)

    # Near-miss % vs F1
    ax = axes[1, 0]
    nm_pcts = [r['near_miss_pct'] for r in results if r['fp'] > 0]
    f1_vals = [r['f1'] for r in results if r['fp'] > 0]
    ax.scatter(nm_pcts, f1_vals, alpha=0.3, s=10, c='#FF9800')
    ax.set_xlabel('Near-miss FP %')
    ax.set_ylabel('F1')
    ax.set_title('F1 vs Near-miss FP %')

    # Shift ±1 % vs Length
    ax = axes[1, 1]
    shift1_pcts = [r['shift_1_pct'] for r in results if r['fp'] > 0]
    lens = [r['length'] for r in results if r['fp'] > 0]
    ax.scatter(lens, shift1_pcts, alpha=0.3, s=10, c='#4CAF50')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Shift ±1 / Total FP')
    ax.set_title('Shift ±1 Ratio vs Length')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / 'shift_analysis.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_bp_compat_analysis(results, out_dir):
    """Analyze BP compatibility effects."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('v8 BP Compatibility Analysis', fontsize=14, fontweight='bold')

    # Non-canonical pairs in GT vs Pred
    gt_nc = [r['gt_noncanonical_pairs'] for r in results if r['gt_pairs'] > 0]
    pred_nc = [r['pred_noncanonical_pairs'] for r in results if r['pred_pairs'] > 0]

    ax = axes[0]
    ax.hist(gt_nc, bins=20, alpha=0.6, color='blue', label=f'GT (mean={np.mean(gt_nc):.2f})')
    ax.hist(pred_nc, bins=20, alpha=0.6, color='red', label=f'Pred (mean={np.mean(pred_nc):.2f})')
    ax.set_xlabel('Non-canonical Pairs')
    ax.set_ylabel('Count')
    ax.set_title('Non-canonical Pairs: GT vs Pred')
    ax.legend(fontsize=8)

    # F1 vs non-canonical ratio in GT
    ax = axes[1]
    nc_ratios = [r['gt_noncanonical_pairs'] / max(r['gt_pairs'], 1) for r in results if r['gt_pairs'] > 0]
    f1_vals = [r['f1'] for r in results if r['gt_pairs'] > 0]
    ax.scatter(nc_ratios, f1_vals, alpha=0.3, s=10, c='#E91E63')
    ax.set_xlabel('GT Non-canonical Ratio')
    ax.set_ylabel('F1')
    ax.set_title('F1 vs GT Non-canonical Ratio')

    # Pred non-canonical pairs effect
    ax = axes[2]
    has_pred_nc = [r for r in results if r['pred_noncanonical_pairs'] > 0]
    no_pred_nc = [r for r in results if r['pred_noncanonical_pairs'] == 0 and r['pred_pairs'] > 0]
    if has_pred_nc and no_pred_nc:
        data_to_plot = [[r['f1'] for r in no_pred_nc], [r['f1'] for r in has_pred_nc]]
        bp = ax.boxplot(data_to_plot, labels=['No NC pred', 'Has NC pred'],
                       patch_artist=True, medianprops={'color': 'black'})
        bp['boxes'][0].set_facecolor('#4CAF50')
        bp['boxes'][1].set_facecolor('#F44336')
        ax.set_ylabel('F1')
        ax.set_title(f'F1: Canonical Only vs Has Non-canonical\n'
                     f'(N={len(no_pred_nc)} vs {len(has_pred_nc)})')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / 'bp_compat_analysis.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_score_confidence_analysis(results, bad_cases, out_dir):
    """Analyze prediction confidence/score distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('v8 Prediction Confidence Analysis', fontsize=14, fontweight='bold')

    # Mean prediction score vs F1
    ax = axes[0, 0]
    has_score = [r for r in results if 'pred_mean_score' in r and r['pred_pairs'] > 0]
    if has_score:
        scores = [r['pred_mean_score'] for r in has_score]
        f1s = [r['f1'] for r in has_score]
        ax.scatter(scores, f1s, alpha=0.3, s=10, c='#2196F3')
        ax.set_xlabel('Mean Prediction Score')
        ax.set_ylabel('F1')
        ax.set_title('F1 vs Mean Prediction Score')

    # Missed GT score distribution (bad cases)
    ax = axes[0, 1]
    missed_scores = [r.get('missed_gt_mean_score', 0) for r in bad_cases if r.get('missed_gt_mean_score', 0) > 0]
    if missed_scores:
        ax.hist(missed_scores, bins=25, color='#F44336', alpha=0.7, edgecolor='white')
        ax.axvline(0.45, color='green', linestyle='--', linewidth=2, label='threshold=0.45')
        ax.set_xlabel('Score at Missed GT Positions')
        ax.set_ylabel('Count')
        ax.set_title('Bad Cases: Scores at Missed GT Positions')
        ax.legend(fontsize=9)

    # GT position mean score vs F1
    ax = axes[1, 0]
    has_gt_score = [r for r in results if 'gt_pos_mean_score' in r]
    if has_gt_score:
        gt_scores = [r['gt_pos_mean_score'] for r in has_gt_score]
        f1s = [r['f1'] for r in has_gt_score]
        ax.scatter(gt_scores, f1s, alpha=0.3, s=10, c='#4CAF50')
        ax.axvline(0.45, color='red', linestyle='--', linewidth=1.5, label='threshold=0.45')
        ax.set_xlabel('Mean Score at GT Positions')
        ax.set_ylabel('F1')
        ax.set_title('F1 vs Mean Score at GT Positions')
        ax.legend(fontsize=9)

    # Density vs pred/gt ratio
    ax = axes[1, 1]
    densities = [r['density'] for r in results if r['gt_pairs'] > 0]
    ratios = [r['pred_gt_ratio'] for r in results if r['gt_pairs'] > 0]
    ax.scatter(densities, ratios, alpha=0.3, s=10, c='#9C27B0')
    ax.axhline(1.0, color='green', linestyle='--', linewidth=1.5)
    ax.set_xlabel('GT Density')
    ax.set_ylabel('Pred/GT Ratio')
    ax.set_title('Pred/GT Ratio vs GT Density')
    ax.set_ylim(0, 3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / 'score_confidence_analysis.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_pseudoknot_analysis(results, out_dir):
    """Analyze pseudoknot impact."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('v8 Pseudoknot Impact Analysis', fontsize=14, fontweight='bold')

    pk_cases = [r for r in results if r['has_pseudoknot']]
    no_pk_cases = [r for r in results if not r['has_pseudoknot']]

    # F1 distribution
    ax = axes[0]
    if no_pk_cases:
        ax.hist([r['f1'] for r in no_pk_cases], bins=20, alpha=0.6, color='green',
                label=f'No PK (N={len(no_pk_cases)}, mean={np.mean([r["f1"] for r in no_pk_cases]):.3f})')
    if pk_cases:
        ax.hist([r['f1'] for r in pk_cases], bins=20, alpha=0.6, color='red',
                label=f'Has PK (N={len(pk_cases)}, mean={np.mean([r["f1"] for r in pk_cases]):.3f})')
    ax.set_xlabel('F1')
    ax.set_ylabel('Count')
    ax.set_title('F1 by Pseudoknot Presence')
    ax.legend(fontsize=8)

    # PK count vs F1
    ax = axes[1]
    pk_counts = [r['gt_pseudoknots'] for r in results]
    f1s = [r['f1'] for r in results]
    ax.scatter(pk_counts, f1s, alpha=0.3, s=10, c='#E74C3C')
    ax.set_xlabel('#Pseudoknot Crossings')
    ax.set_ylabel('F1')
    ax.set_title('F1 vs PK Count')

    # Bad case rate by PK
    ax = axes[2]
    if pk_cases and no_pk_cases:
        pk_bad_rate = len([r for r in pk_cases if r['f1'] < 0.3]) / len(pk_cases)
        nopk_bad_rate = len([r for r in no_pk_cases if r['f1'] < 0.3]) / len(no_pk_cases)
        bars = ax.bar(['No Pseudoknot', 'Has Pseudoknot'],
                      [nopk_bad_rate * 100, pk_bad_rate * 100],
                      color=['#4CAF50', '#F44336'], alpha=0.7)
        ax.set_ylabel('Bad Case Rate (%)')
        ax.set_title('Bad Case Rate by PK Presence')
        for bar, rate in zip(bars, [nopk_bad_rate, pk_bad_rate]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{rate*100:.1f}%', ha='center', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / 'pseudoknot_analysis.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_complexity_analysis(results, bad_cases, out_dir):
    """Analyze structure complexity impact."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('v8 Structure Complexity vs Performance', fontsize=14, fontweight='bold')

    metrics = [
        ('gt_n_stems', '#Stems', '#2196F3'),
        ('gt_max_stem_len', 'Max Stem Length', '#4CAF50'),
        ('gt_branching', 'Branching Factor', '#FF9800'),
        ('gt_mean_dist', 'Mean Pair Distance', '#9C27B0'),
        ('gt_pseudoknots', '#PK Crossings', '#F44336'),
        ('density', 'Pairing Density', '#795548'),
    ]

    for idx, (key, label, color) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        x = [r[key] for r in results]
        y = [r['f1'] for r in results]
        ax.scatter(x, y, alpha=0.3, s=10, c=color)
        ax.set_xlabel(label)
        ax.set_ylabel('F1')
        ax.set_title(f'F1 vs {label}')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / 'complexity_vs_f1.png', dpi=120, bbox_inches='tight')
    plt.close()


# ============================================================
# Report Generation
# ============================================================

def generate_report(results, bad_cases, out_dir, config):
    """Generate comprehensive markdown report."""
    report_lines = []
    report_lines.append("# PriFold v8 DensityNet-Pro: Bad Case 全面分析报告\n")
    report_lines.append(f"**生成时间**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    report_lines.append(f"**模型**: v8 DensityNet-Pro (best.pt)\n")
    report_lines.append(f"**数据集**: bprna-test\n")
    report_lines.append(f"**总样本数**: {len(results)}\n")
    report_lines.append(f"**Bad Cases (F1 < 0.3)**: {len(bad_cases)} ({len(bad_cases)/len(results)*100:.1f}%)\n")
    report_lines.append("")

    # === Section 1: Overall Performance ===
    report_lines.append("## 1. 总体性能\n")
    f1s = [r['f1'] for r in results]
    precs = [r['precision'] for r in results]
    recs = [r['recall'] for r in results]
    mccs = [r['mcc'] for r in results]
    ratios = [r['pred_gt_ratio'] for r in results if r['gt_pairs'] > 0]

    report_lines.append("| 指标 | Mean | Median | Std | Min | Max |")
    report_lines.append("|------|------|--------|-----|-----|-----|")
    for name, vals in [('F1', f1s), ('Precision', precs), ('Recall', recs),
                       ('MCC', mccs), ('Pred/GT Ratio', ratios)]:
        report_lines.append(
            f"| {name} | {np.mean(vals):.4f} | {np.median(vals):.4f} | "
            f"{np.std(vals):.4f} | {np.min(vals):.4f} | {np.max(vals):.4f} |")
    report_lines.append("")

    # Performance tiers
    tiers = {
        'Excellent (F1≥0.8)': len([r for r in results if r['f1'] >= 0.8]),
        'Good (0.6≤F1<0.8)': len([r for r in results if 0.6 <= r['f1'] < 0.8]),
        'Fair (0.3≤F1<0.6)': len([r for r in results if 0.3 <= r['f1'] < 0.6]),
        'Bad (F1<0.3)': len([r for r in results if r['f1'] < 0.3]),
        'Zero (F1=0)': len([r for r in results if r['f1'] == 0]),
    }
    report_lines.append("### 性能分层\n")
    report_lines.append("| 等级 | 样本数 | 占比 |")
    report_lines.append("|------|--------|------|")
    for tier, count in tiers.items():
        report_lines.append(f"| {tier} | {count} | {count/len(results)*100:.1f}% |")
    report_lines.append("")

    # === Section 2: Bad Case Failure Mode Analysis ===
    report_lines.append("## 2. Bad Case 失败模式分析\n")
    modes = Counter()
    for r in bad_cases:
        modes[classify_failure_mode(r)] += 1

    report_lines.append("| 失败模式 | 数量 | 占比 | 说明 |")
    report_lines.append("|----------|------|------|------|")
    mode_descriptions = {
        'no_prediction': '模型完全没有预测（budget/threshold 过紧）',
        'complete_miss': '有预测但完全错位（完全在错误位置）',
        'shifted_prediction': '预测位置偏移（在 GT±3 范围内）',
        'severe_overpredict': '严重过预测（pred/gt > 2.0）',
        'severe_underpredict': '严重欠预测（pred/gt < 0.3）',
        'budget_too_tight': 'v8 length decay 导致 budget 过紧（高 precision 低 recall）',
        'budget_too_loose': '阈值过松导致过多 FP（低 precision 高 recall）',
        'wrong_position': '预测数量合理但位置错误',
        'pseudoknot_failure': '含伪结结构预测困难',
        'bp_compat_issue': '非标准碱基配对相关问题',
        'mixed': '混合问题',
    }
    for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
        desc = mode_descriptions.get(mode, '—')
        report_lines.append(f"| {mode} | {count} | {count/len(bad_cases)*100:.1f}% | {desc} |")
    report_lines.append("")

    # === Section 3: v8 Feature Impact ===
    report_lines.append("## 3. v8 新特性效果评估\n")

    # Length decay analysis
    report_lines.append("### 3.1 Length-Aware Budget\n")
    short = [r for r in results if r['length'] < 100]
    medium = [r for r in results if 100 <= r['length'] < 200]
    long_seq = [r for r in results if 200 <= r['length'] < 350]
    vlong = [r for r in results if r['length'] >= 350]

    report_lines.append("| 长度区间 | 样本数 | Mean F1 | Mean Prec | Mean Recall | Mean Ratio | Bad Rate |")
    report_lines.append("|----------|--------|---------|-----------|-------------|------------|----------|")
    for name, subset in [('<100', short), ('100-200', medium), ('200-350', long_seq), ('350+', vlong)]:
        if subset:
            bad_rate = len([r for r in subset if r['f1'] < 0.3]) / len(subset) * 100
            report_lines.append(
                f"| {name} | {len(subset)} | {np.mean([r['f1'] for r in subset]):.4f} | "
                f"{np.mean([r['precision'] for r in subset]):.4f} | "
                f"{np.mean([r['recall'] for r in subset]):.4f} | "
                f"{np.mean([r['pred_gt_ratio'] for r in subset if r['gt_pairs']>0]):.3f} | "
                f"{bad_rate:.1f}% |")
    report_lines.append("")

    # Shift loss effectiveness
    report_lines.append("### 3.2 Shift-Aware Loss 效果\n")
    total_fp = sum(r['total_fp_analyzed'] for r in results)
    total_near = sum(r['near_miss_1'] + r['near_miss_2'] + r['near_miss_3'] for r in results)
    total_far = sum(r['far_miss_fp'] for r in results)
    report_lines.append(f"- 总 FP 数量: {total_fp}")
    report_lines.append(f"- Near-miss FP (±1~±3): {total_near} ({total_near/max(total_fp,1)*100:.1f}%)")
    report_lines.append(f"  - Shift ±1: {sum(r['near_miss_1'] for r in results)} ({sum(r['near_miss_1'] for r in results)/max(total_fp,1)*100:.1f}%)")
    report_lines.append(f"  - Shift ±2: {sum(r['near_miss_2'] for r in results)} ({sum(r['near_miss_2'] for r in results)/max(total_fp,1)*100:.1f}%)")
    report_lines.append(f"  - Shift ±3: {sum(r['near_miss_3'] for r in results)} ({sum(r['near_miss_3'] for r in results)/max(total_fp,1)*100:.1f}%)")
    report_lines.append(f"- Far FP (完全错误): {total_far} ({total_far/max(total_fp,1)*100:.1f}%)")
    report_lines.append("")

    # BP Compatibility
    report_lines.append("### 3.3 BP Compatibility\n")
    gt_nc_total = sum(r['gt_noncanonical_pairs'] for r in results)
    pred_nc_total = sum(r['pred_noncanonical_pairs'] for r in results)
    gt_c_total = sum(r['gt_canonical_pairs'] for r in results)
    pred_c_total = sum(r['pred_canonical_pairs'] for r in results)
    report_lines.append(f"- GT 中非标准配对总数: {gt_nc_total} (占 GT: {gt_nc_total/max(gt_c_total+gt_nc_total,1)*100:.1f}%)")
    report_lines.append(f"- Pred 中非标准配对总数: {pred_nc_total} (占 Pred: {pred_nc_total/max(pred_c_total+pred_nc_total,1)*100:.1f}%)")
    if gt_nc_total > 0:
        report_lines.append(f"- **注意**: BP compat 在 v8 config 中被关闭 (bp_compat_enabled=false)")
        report_lines.append(f"- GT 中本身就有非标准配对，强行过滤可能导致 recall 下降")
    report_lines.append("")

    # === Section 4: Key Findings ===
    report_lines.append("## 4. 关键发现\n")

    # Finding 1: Precision vs Recall balance
    report_lines.append("### 4.1 Precision vs Recall 平衡\n")
    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)
    if mean_prec > mean_rec + 0.05:
        report_lines.append(f"- v8 模型 **偏保守**: Precision ({mean_prec:.4f}) > Recall ({mean_rec:.4f})")
        report_lines.append(f"- FP Penalty + OHEM + 提高 threshold 共同作用，导致预测偏少")
        underpredict_cases = [r for r in bad_cases if r['pred_gt_ratio'] < 0.5]
        if underpredict_cases:
            report_lines.append(f"- Bad cases 中有 {len(underpredict_cases)} 个严重欠预测")
    elif mean_rec > mean_prec + 0.05:
        report_lines.append(f"- v8 模型 **偏激进**: Recall ({mean_rec:.4f}) > Precision ({mean_prec:.4f})")
    else:
        report_lines.append(f"- v8 模型 Precision ({mean_prec:.4f}) 和 Recall ({mean_rec:.4f}) 相对平衡")
    report_lines.append("")

    # Finding 2: Length sensitivity
    report_lines.append("### 4.2 长序列问题\n")
    if long_seq:
        long_bad_rate = len([r for r in long_seq if r['f1'] < 0.3]) / len(long_seq) * 100
        short_bad_rate = len([r for r in short if r['f1'] < 0.3]) / len(short) * 100 if short else 0
        report_lines.append(f"- 短序列 (<100) bad rate: {short_bad_rate:.1f}%")
        report_lines.append(f"- 长序列 (200-350) bad rate: {long_bad_rate:.1f}%")
        if long_bad_rate > short_bad_rate * 1.5:
            report_lines.append(f"- **长序列 bad rate 显著高于短序列**: length decay 可能需要调整")
        # Check if recall drops more than precision for long sequences
        if long_seq:
            long_prec = np.mean([r['precision'] for r in long_seq])
            long_rec = np.mean([r['recall'] for r in long_seq])
            report_lines.append(f"- 长序列 (200-350): Precision={long_prec:.4f}, Recall={long_rec:.4f}")
            if long_rec < long_prec - 0.05:
                report_lines.append(f"- **长序列 Recall 显著低于 Precision**: length_decay=0.3 可能过于激进")
    report_lines.append("")

    # Finding 3: Pseudoknot challenges
    report_lines.append("### 4.3 伪结问题\n")
    pk_cases = [r for r in results if r['has_pseudoknot']]
    no_pk_cases = [r for r in results if not r['has_pseudoknot']]
    if pk_cases:
        pk_f1 = np.mean([r['f1'] for r in pk_cases])
        nopk_f1 = np.mean([r['f1'] for r in no_pk_cases])
        pk_bad = len([r for r in pk_cases if r['f1'] < 0.3]) / len(pk_cases) * 100
        report_lines.append(f"- 含伪结: Mean F1={pk_f1:.4f}, Bad rate={pk_bad:.1f}% (N={len(pk_cases)})")
        report_lines.append(f"- 无伪结: Mean F1={nopk_f1:.4f} (N={len(no_pk_cases)})")
        report_lines.append(f"- 伪结导致 F1 下降: {(nopk_f1-pk_f1)*100:.1f} 个百分点")
    report_lines.append("")

    # Finding 4: Score threshold analysis
    report_lines.append("### 4.4 Score Threshold 分析\n")
    missed_below_thresh = [r for r in bad_cases if r.get('missed_gt_mean_score', 0) > 0 and r.get('missed_gt_mean_score', 0) < 0.45]
    missed_above_thresh = [r for r in bad_cases if r.get('missed_gt_mean_score', 0) >= 0.45]
    report_lines.append(f"- Score threshold = {config.get('sampling', {}).get('score_threshold', 0.45)}")
    if missed_below_thresh:
        report_lines.append(f"- Bad cases 中 missed GT 平均 score < threshold: {len(missed_below_thresh)} 个")
        report_lines.append(f"  → 这些是模型 confidence 不足的 case，不是 threshold 问题")
    if missed_above_thresh:
        report_lines.append(f"- Bad cases 中 missed GT 平均 score ≥ threshold: {len(missed_above_thresh)} 个")
        report_lines.append(f"  → 这些可能被 budget limit 截断了")
    report_lines.append("")

    # === Section 5: Top Bad Cases ===
    report_lines.append("## 5. 最差 Case 列表 (Top 20)\n")
    worst = sorted(bad_cases, key=lambda x: x['f1'])[:20]
    report_lines.append("| # | Name | Length | F1 | Prec | Recall | GT Pairs | Pred Pairs | Ratio | Failure Mode |")
    report_lines.append("|---|------|--------|----|----|--------|----------|------------|-------|--------------|")
    for i, r in enumerate(worst, 1):
        mode = classify_failure_mode(r)
        report_lines.append(
            f"| {i} | {r['name'][:25]} | {r['length']} | {r['f1']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['gt_pairs']} | "
            f"{r['pred_pairs']} | {r['pred_gt_ratio']:.2f} | {mode} |")
    report_lines.append("")

    # === Section 6: Improvement Suggestions ===
    report_lines.append("## 6. 改进建议\n")
    report_lines.append("### 优先级排序\n")

    suggestions = []
    # Based on analysis
    if modes.get('severe_underpredict', 0) + modes.get('budget_too_tight', 0) > len(bad_cases) * 0.2:
        suggestions.append(("高", "降低 length_decay 或提供更宽松的 budget",
                           "大量 bad case 源自欠预测，length decay 可能过于激进"))
    if modes.get('shifted_prediction', 0) > len(bad_cases) * 0.2:
        suggestions.append(("高", "增加 shift_radius 或提高 shift_loss_weight",
                           "偏移预测仍然是主要问题"))
    if modes.get('pseudoknot_failure', 0) > len(bad_cases) * 0.1:
        suggestions.append(("中", "考虑伪结专用处理（多阶段预测或专用 head）",
                           "伪结结构预测困难"))
    if total_near / max(total_fp, 1) > 0.3:
        suggestions.append(("中", "shift_loss 正在生效但可能需要更大 radius",
                           f"近距离 FP 占比 {total_near/max(total_fp,1)*100:.1f}%"))
    if pred_nc_total > 0:
        suggestions.append(("低", "考虑启用 bp_compat_in_inference",
                           f"预测中有 {pred_nc_total} 个非标准配对"))

    report_lines.append("| 优先级 | 建议 | 原因 |")
    report_lines.append("|--------|------|------|")
    for pri, suggestion, reason in suggestions:
        report_lines.append(f"| {pri} | {suggestion} | {reason} |")
    if not suggestions:
        report_lines.append("| — | 继续训练至 200 epochs | 模型仍在收敛中 |")
    report_lines.append("")

    # === Section 7: v7 vs v8 Comparison ===
    report_lines.append("## 7. v7 vs v8 对比 (参考)\n")
    report_lines.append("| 指标 | v7 (200 epochs) | v8 (当前) | 变化 |")
    report_lines.append("|------|-----------------|-----------|------|")
    v7_f1, v7_prec, v7_rec = 0.6538, 0.6267, 0.7122  # v7 final numbers
    report_lines.append(f"| Test F1 | {v7_f1:.4f} | {np.mean(f1s):.4f} | {np.mean(f1s)-v7_f1:+.4f} |")
    report_lines.append(f"| Test Precision | {v7_prec:.4f} | {np.mean(precs):.4f} | {np.mean(precs)-v7_prec:+.4f} |")
    report_lines.append(f"| Test Recall | {v7_rec:.4f} | {np.mean(recs):.4f} | {np.mean(recs)-v7_rec:+.4f} |")
    report_lines.append(f"| Bad Case Rate | ~15% | {len(bad_cases)/len(results)*100:.1f}% | — |")
    report_lines.append("")
    report_lines.append("> **注意**: v8 模型仍在训练中 (epoch ~141/200)，最终结果可能更好。\n")

    # Write report
    report_path = out_dir / 'v8_bad_case_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f'Report saved to: {report_path}')
    return report_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='PriFold v8 Bad Case Analysis')
    parser.add_argument('--ckpt', type=str, default='symfold/outputs/v8_full/model/best.pt')
    parser.add_argument('--config', type=str, default='symfold/config/v8/v8_full.json')
    parser.add_argument('--out_dir', type=str, default='symfold/outputs/v8_full/comprehensive_analysis')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--max_bad_case_cards', type=int, default=30)
    args = parser.parse_args()

    # Setup
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'bad_cases').mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load config
    config = load_config(args.config)
    config['training']['batch_size'] = 1  # Per-sample analysis

    # Load model
    print('Loading model...')
    class Args:
        pass
    lm_args = Args()
    lm_args.pretrained_lm_dir = config['paths'].get('pretrained_lm_dir', str(ROOT / 'model'))
    lm_args.model_scale = config['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)
    model = build_model(config, extractor)
    ckpt = torch.load(args.ckpt, map_location='cpu')
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'], strict=False)
        print(f'  Loaded from checkpoint epoch={ckpt.get("epoch","?")} best_f1={ckpt.get("best_f1","?"):.4f}')
    elif 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    print('Model loaded.')

    # Load data
    print('Loading test data...')
    test_loader = build_loader('bprna-test', config, tokenizer, shuffle=False)
    print(f'Test samples: {len(test_loader.dataset)}')

    # Build sequence map
    seq_map = {}
    if hasattr(test_loader.dataset, 'data'):
        for item in test_loader.dataset.data:
            if 'name' in item and 'seq' in item:
                seq_map[item['name']] = item['seq']
    elif hasattr(test_loader.dataset, 'samples'):
        for item in test_loader.dataset.samples:
            if isinstance(item, dict) and 'name' in item and 'seq' in item:
                seq_map[item['name']] = item['seq']

    print(f'Sequence map: {len(seq_map)} entries')

    # Run inference
    print('\n=== Running inference on test set ===')
    test_results = run_inference(model, test_loader, device, config, seq_map)
    print(f'Total test results: {len(test_results)}')

    # Identify bad cases
    bad_cases = sorted([r for r in test_results if r['f1'] < 0.3], key=lambda x: x['f1'])
    print(f'\nBad cases (F1 < 0.3): {len(bad_cases)} / {len(test_results)} '
          f'({len(bad_cases)/len(test_results)*100:.1f}%)')

    # === Generate Visualizations ===
    print('\n=== Generating visualizations ===')

    print('  [1/8] Overall performance...')
    plot_overall_performance(test_results, out_dir)

    print('  [2/8] Failure mode summary...')
    plot_failure_mode_summary(bad_cases, out_dir)

    print('  [3/8] Length decay analysis...')
    plot_length_decay_analysis(test_results, out_dir)

    print('  [4/8] Shift analysis...')
    plot_shift_analysis(test_results, bad_cases, out_dir)

    print('  [5/8] BP compatibility analysis...')
    plot_bp_compat_analysis(test_results, out_dir)

    print('  [6/8] Score confidence analysis...')
    plot_score_confidence_analysis(test_results, bad_cases, out_dir)

    print('  [7/8] Pseudoknot analysis...')
    plot_pseudoknot_analysis(test_results, out_dir)

    print('  [8/8] Complexity analysis...')
    plot_complexity_analysis(test_results, bad_cases, out_dir)

    # Bad case cards
    print(f'\n  Generating bad case cards (top {args.max_bad_case_cards})...')
    for i, case in enumerate(bad_cases[:args.max_bad_case_cards]):
        safe_name = case['name'].replace('/', '_').replace(' ', '_')[:50]
        card_path = out_dir / 'bad_cases' / f'{i:03d}_{safe_name}.png'
        plot_bad_case_card(case, card_path)
        if (i + 1) % 10 == 0:
            print(f'    {i + 1}/{min(len(bad_cases), args.max_bad_case_cards)} cards done')

    # Generate report
    print('\n=== Generating report ===')
    report_path = generate_report(test_results, bad_cases, out_dir, config)

    # Save metrics as JSON
    metrics_json = []
    for r in test_results:
        entry = {k: v for k, v in r.items()
                 if k not in ('pred_map', 'gt_map', 'score_map', 'gt_dists', 'pred_dists', 'seq')}
        # Convert non-serializable types
        for k, v in entry.items():
            if isinstance(v, (np.integer, np.floating)):
                entry[k] = float(v)
            elif isinstance(v, np.ndarray):
                entry[k] = v.tolist()
        metrics_json.append(entry)

    with open(out_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)
    print(f'Metrics saved to: {out_dir / "test_metrics.json"}')

    # Save bad case summary CSV
    csv_path = out_dir / 'bad_cases_summary.csv'
    fieldnames = ['name', 'length', 'f1', 'precision', 'recall', 'mcc',
                  'gt_pairs', 'pred_pairs', 'pred_gt_ratio', 'density',
                  'gt_pseudoknots', 'gt_n_stems', 'near_miss_pct', 'failure_mode']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in bad_cases:
            writer.writerow({
                'name': r['name'],
                'length': r['length'],
                'f1': f"{r['f1']:.4f}",
                'precision': f"{r['precision']:.4f}",
                'recall': f"{r['recall']:.4f}",
                'mcc': f"{r['mcc']:.4f}",
                'gt_pairs': r['gt_pairs'],
                'pred_pairs': r['pred_pairs'],
                'pred_gt_ratio': f"{r['pred_gt_ratio']:.3f}",
                'density': f"{r['density']:.4f}",
                'gt_pseudoknots': r['gt_pseudoknots'],
                'gt_n_stems': r['gt_n_stems'],
                'near_miss_pct': f"{r['near_miss_pct']:.3f}",
                'failure_mode': classify_failure_mode(r),
            })
    print(f'Bad case CSV saved to: {csv_path}')

    print('\n' + '=' * 60)
    print('Analysis complete!')
    print(f'Output directory: {out_dir}')
    print(f'Report: {report_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()
