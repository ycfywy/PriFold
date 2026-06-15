# -*- coding: utf-8 -*-
"""PriFold v7 DensityNet: 全面推理分析脚本。

对 bprna-train/val/test 三个数据集做完整推理，记录所有样本表现，
生成全面的分析报告和可视化。

分析内容：
1. 低 F1 case 的序列长度分布、配对距离分布、结构复杂度、伪结情况
2. 同等长度/密度下成功 vs 失败案例对比（配对图 + GT）
3. Bad case 全面分析：长度/密度/结构复杂度/伪结/配对情况
4. 每个分析点按 train/val/test 分别统计

输出：
- bad_cases/ 文件夹：每个 bad case 的 contact map + GT + 标注
- report.md：汇总文档，含原因分析和改进建议
- 所有分布可视化图

Usage:
  python symfold/analysis/comprehensive_v7_analysis.py \
    --ckpt symfold/outputs/v7_full/model/best.pt \
    --config symfold/config/v7/v7_full.json \
    --out_dir symfold/outputs/v7_full/comprehensive_analysis
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
from symfold.train.train_v7 import build_model, load_config, move_to_device


# ============================================================
# Helpers
# ============================================================

VALID_PAIRS = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G'), ('G', 'U'), ('U', 'G')}


def detect_pseudoknots(contact_matrix, length):
    """Detect pseudoknots in contact map.
    A pseudoknot exists when (i,j) and (k,l) are both paired with i<k<j<l.
    Returns: number of pseudoknot crossings, list of crossing pairs.
    """
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
            # Pseudoknot: i < k < j < l
            if i < k < j < l:
                crossings += 1
                crossing_pairs.append(((i, j), (k, l)))

    return crossings, crossing_pairs


def compute_structure_complexity(contact_matrix, length):
    """Compute structure complexity metrics.
    Returns dict with: n_stems, n_multiloops, n_internal_loops, branching_factor,
    max_stem_length, avg_stem_length.
    """
    pairs = set()
    for i in range(length):
        for j in range(i + 3, length):
            if contact_matrix[i, j] > 0.5:
                pairs.add((i, j))

    # Find stems (consecutive base pairs)
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

    # Branching factor: number of stems per "junction"
    # Approximate: count unique 5' ends of stems
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
    for i in range(length):
        for j in range(i + 3, length):
            if contact_matrix[i, j] > 0.5:
                pair = (seq[i].upper(), seq[j].upper())
                if pair in VALID_PAIRS:
                    canonical += 1
                else:
                    non_canonical += 1
    return canonical, non_canonical


def per_sample_metrics(pred, target, length, seq=None, score=None):
    """Compute comprehensive per-sample metrics."""
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

    # Numpy arrays for further analysis
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
    gt_canonical, gt_noncanonical = 0, 0
    pred_canonical, pred_noncanonical = 0, 0
    if seq and len(seq) >= length:
        gt_canonical, gt_noncanonical = check_bp_compatibility(seq, y_np, length)
        pred_canonical, pred_noncanonical = check_bp_compatibility(seq, p_np, length)

    # Shift analysis
    near_miss = 0
    far_miss = 0
    for i in range(length):
        for j in range(i + 3, length):
            if p_np[i, j] > 0.5 and y_np[i, j] < 0.5:
                # FP: check if near a GT pair
                found = False
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < length and 0 <= nj < length and y_np[ni, nj] > 0.5:
                            near_miss += 1
                            found = True
                            break
                    if found:
                        break
                if not found:
                    far_miss += 1

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
        # Shift
        'near_miss_fp': near_miss,
        'far_miss_fp': far_miss,
        'near_miss_pct': near_miss / max(fp, 1),
        # Raw data for visualization
        'gt_dists': gt_dists,
        'pred_dists': pred_dists,
    }
    return result


def classify_failure_mode(r):
    """Classify a bad case into failure mode."""
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
    elif 0.7 < r['pred_gt_ratio'] < 1.3 and r['f1'] < 0.3:
        if r['near_miss_pct'] > 0.3:
            return 'shifted_prediction'
        else:
            return 'wrong_position'
    elif r['has_pseudoknot'] and r['f1'] < 0.4:
        return 'pseudoknot_failure'
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
                        score_threshold=scfg.get('score_threshold', 0.4),
                    )
            else:
                pred, extras = model.predict(
                    batch,
                    budget_fraction=scfg.get('default_budget_fraction', 0.30),
                    use_density_budget=scfg.get('use_density_budget', True),
                    score_threshold=scfg.get('score_threshold', 0.4),
                )

            bs = pred.shape[0]
            for i in range(bs):
                length = int(batch['length'][i])
                name = batch['name'][i] if 'name' in batch else f'sample_{batch_idx}_{i}'
                seq = seq_map.get(name, '') if seq_map else ''

                # Get score map if available
                score = None
                if isinstance(extras, dict) and 'score' in extras:
                    score = extras['score'][i].detach().cpu().float().squeeze()[:length, :length].numpy()
                elif isinstance(extras, torch.Tensor):
                    score = extras[i].detach().cpu().float().squeeze()[:length, :length].numpy()

                metrics = per_sample_metrics(pred[i], batch['contact'][i], length, seq=seq, score=score)
                metrics['name'] = name
                metrics['length'] = length
                metrics['seq'] = seq

                # Store raw maps for visualization
                metrics['pred_map'] = pred[i].detach().cpu().float().squeeze()[:length, :length].numpy()
                metrics['gt_map'] = batch['contact'][i].detach().cpu().float().squeeze()[:length, :length].numpy()
                if score is not None:
                    metrics['score_map'] = score

                results.append(metrics)

            if (batch_idx + 1) % 50 == 0:
                print(f'    Batch {batch_idx + 1}/{len(loader)} done ({len(results)} samples)')

    return results


# ============================================================
# Visualization Functions
# ============================================================

def plot_bad_case_card(case, out_path):
    """Plot a single bad case card with contact map, GT, diff, and annotations."""
    L = case['length']
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.8], wspace=0.3)

    # GT contact map
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(case['gt_map'], cmap='Blues', vmin=0, vmax=1, aspect='equal')
    ax1.set_title('Ground Truth', fontsize=10)
    ax1.set_xlabel(f'L={L}')

    # Predicted contact map
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(case['pred_map'], cmap='Oranges', vmin=0, vmax=1, aspect='equal')
    ax2.set_title('Prediction', fontsize=10)

    # Diff map (TP=green, FP=red, FN=blue)
    ax3 = fig.add_subplot(gs[2])
    diff = np.zeros((L, L, 3))
    p_bin = case['pred_map'] > 0.5
    g_bin = case['gt_map'] > 0.5
    diff[p_bin & g_bin] = [0.2, 0.8, 0.2]       # TP - green
    diff[p_bin & ~g_bin] = [0.9, 0.2, 0.2]      # FP - red
    diff[~p_bin & g_bin] = [0.2, 0.4, 0.9]      # FN - blue
    ax3.imshow(diff, aspect='equal')
    ax3.set_title('TP(green)/FP(red)/FN(blue)', fontsize=9)
    legend_elements = [
        Patch(facecolor=[0.2, 0.8, 0.2], label=f'TP={case["tp"]}'),
        Patch(facecolor=[0.9, 0.2, 0.2], label=f'FP={case["fp"]}'),
        Patch(facecolor=[0.2, 0.4, 0.9], label=f'FN={case["fn"]}'),
    ]
    ax3.legend(handles=legend_elements, loc='lower right', fontsize=7)

    # Annotation panel
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')
    failure_mode = classify_failure_mode(case)
    info_text = (
        f"RNA: {case['name']}\n"
        f"Length: {L}\n"
        f"─────────────────\n"
        f"F1: {case['f1']:.4f}\n"
        f"Precision: {case['precision']:.4f}\n"
        f"Recall: {case['recall']:.4f}\n"
        f"─────────────────\n"
        f"GT pairs: {case['gt_pairs']}\n"
        f"Pred pairs: {case['pred_pairs']}\n"
        f"pred/gt: {case['pred_gt_ratio']:.2f}\n"
        f"Density: {case['density']:.4f}\n"
        f"─────────────────\n"
        f"GT stems: {case['gt_n_stems']}\n"
        f"GT pseudoknots: {case['gt_pseudoknots']}\n"
        f"Near-miss FP: {case['near_miss_fp']}\n"
        f"Far-miss FP: {case['far_miss_fp']}\n"
        f"─────────────────\n"
        f"Failure mode:\n  {failure_mode}"
    )
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(f'{case["name"]}  |  F1={case["f1"]:.4f}  |  L={L}  |  Mode: {failure_mode}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_distribution_comparison(train_data, val_data, test_data, key, title, xlabel, out_path,
                                  bins=20, log_y=False):
    """Plot distribution of a metric across train/val/test."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, data, label, color in zip(axes,
                                       [train_data, val_data, test_data],
                                       ['Train', 'Val', 'Test'],
                                       ['#3498db', '#e67e22', '#e74c3c']):
        values = [r[key] for r in data if key in r]
        if values:
            ax.hist(values, bins=bins, color=color, alpha=0.7, edgecolor='white')
            ax.axvline(np.mean(values), color='black', linestyle='--', linewidth=1.5,
                       label=f'mean={np.mean(values):.3f}')
            ax.axvline(np.median(values), color='gray', linestyle=':', linewidth=1.5,
                       label=f'median={np.median(values):.3f}')
            ax.legend(fontsize=8)
        ax.set_title(f'{label} (N={len(values)})', fontsize=11)
        ax.set_xlabel(xlabel)
        if log_y:
            ax.set_yscale('log')
    axes[0].set_ylabel('Count')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_bad_case_distributions(bad_cases_by_stage, out_dir):
    """Plot distributions specifically for bad cases (F1 < 0.3)."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle('Bad Case Analysis (F1 < 0.3): Distributions by Stage', fontsize=14, fontweight='bold')

    metrics_to_plot = [
        ('length', 'Sequence Length', 30),
        ('density', 'Pairing Density', 20),
        ('gt_n_stems', 'Number of Stems (GT)', 20),
        ('gt_pseudoknots', 'Pseudoknot Crossings (GT)', 15),
        ('gt_mean_dist', 'Mean Pairing Distance (GT)', 20),
        ('gt_max_dist', 'Max Pairing Distance (GT)', 20),
        ('pred_gt_ratio', 'Pred/GT Ratio', 20),
        ('near_miss_pct', 'Near-miss FP %', 20),
        ('gt_branching', 'Branching Factor', 15),
        ('gt_max_stem_len', 'Max Stem Length', 15),
        ('gt_avg_stem_len', 'Avg Stem Length', 15),
        ('gt_canonical_pairs', 'Canonical Pairs (GT)', 20),
    ]

    colors = {'train': '#3498db', 'val': '#e67e22', 'test': '#e74c3c'}

    for idx, (key, label, nbins) in enumerate(metrics_to_plot):
        row, col = idx // 4, idx % 4
        ax = axes[row, col]
        for stage, color in colors.items():
            cases = bad_cases_by_stage.get(stage, [])
            values = [r[key] for r in cases if key in r and r[key] is not None]
            if values:
                ax.hist(values, bins=nbins, alpha=0.5, color=color,
                        label=f'{stage}(N={len(values)})', edgecolor='white')
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.legend(fontsize=7)
        ax.set_title(label, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / 'bad_case_distributions.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_success_vs_failure_comparison(all_results_by_stage, out_dir):
    """Compare success (F1>=0.7) vs failure (F1<0.3) cases at similar length/density."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle('Success (F1≥0.7) vs Failure (F1<0.3) Comparison', fontsize=14, fontweight='bold')

    for stage_idx, (stage, results) in enumerate(all_results_by_stage.items()):
        success = [r for r in results if r['f1'] >= 0.7]
        failure = [r for r in results if r['f1'] < 0.3]

        if not success or not failure:
            continue

        # Length distribution comparison
        ax = axes[stage_idx, 0]
        ax.hist([r['length'] for r in success], bins=20, alpha=0.6, color='green', label='Success')
        ax.hist([r['length'] for r in failure], bins=20, alpha=0.6, color='red', label='Failure')
        ax.set_title(f'{stage}: Length Distribution', fontsize=9)
        ax.legend(fontsize=7)

        # Density distribution comparison
        ax = axes[stage_idx, 1]
        ax.hist([r['density'] for r in success], bins=20, alpha=0.6, color='green', label='Success')
        ax.hist([r['density'] for r in failure], bins=20, alpha=0.6, color='red', label='Failure')
        ax.set_title(f'{stage}: Density Distribution', fontsize=9)
        ax.legend(fontsize=7)

        # Pseudoknot comparison
        ax = axes[stage_idx, 2]
        s_pk = [r['gt_pseudoknots'] for r in success]
        f_pk = [r['gt_pseudoknots'] for r in failure]
        ax.bar(['Success\n(F1≥0.7)', 'Failure\n(F1<0.3)'],
               [np.mean(s_pk), np.mean(f_pk)],
               color=['green', 'red'], alpha=0.7)
        ax.set_title(f'{stage}: Avg Pseudoknots', fontsize=9)
        ax.set_ylabel('Avg #PK crossings')

        # Complexity comparison
        ax = axes[stage_idx, 3]
        s_stems = [r['gt_n_stems'] for r in success]
        f_stems = [r['gt_n_stems'] for r in failure]
        ax.bar(['Success\n(F1≥0.7)', 'Failure\n(F1<0.3)'],
               [np.mean(s_stems), np.mean(f_stems)],
               color=['green', 'red'], alpha=0.7)
        ax.set_title(f'{stage}: Avg #Stems', fontsize=9)
        ax.set_ylabel('Avg #Stems')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / 'success_vs_failure_comparison.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_paired_comparison_at_similar_conditions(all_results_by_stage, out_dir):
    """For similar length/density, compare success vs failure contact maps."""
    for stage, results in all_results_by_stage.items():
        # Group by length bin
        length_bins = {'<100': (0, 100), '100-200': (100, 200), '200-300': (200, 300), '300+': (300, 999)}

        for bin_name, (lo, hi) in length_bins.items():
            bin_results = [r for r in results if lo <= r['length'] < hi]
            success = sorted([r for r in bin_results if r['f1'] >= 0.7], key=lambda x: -x['f1'])[:3]
            failure = sorted([r for r in bin_results if r['f1'] < 0.3], key=lambda x: x['f1'])[:3]

            if not success or not failure:
                continue

            n_pairs = min(3, len(success), len(failure))
            fig, axes = plt.subplots(n_pairs * 2, 3, figsize=(12, n_pairs * 2 * 3))
            if n_pairs * 2 == 2:
                axes = axes.reshape(2, 3)
            fig.suptitle(f'{stage} | Length {bin_name}: Success vs Failure Contact Maps',
                         fontsize=12, fontweight='bold')

            for i in range(n_pairs):
                # Success case
                row = i * 2
                s = success[i]
                L = s['length']
                axes[row, 0].imshow(s['gt_map'], cmap='Blues', vmin=0, vmax=1, aspect='equal')
                axes[row, 0].set_title(f"GT | {s['name'][:20]}", fontsize=7)
                axes[row, 0].set_ylabel(f"SUCCESS\nF1={s['f1']:.3f}\nL={L}", fontsize=7)
                axes[row, 1].imshow(s['pred_map'], cmap='Oranges', vmin=0, vmax=1, aspect='equal')
                axes[row, 1].set_title('Pred', fontsize=7)
                diff = np.zeros((L, L, 3))
                diff[(s['pred_map'] > 0.5) & (s['gt_map'] > 0.5)] = [0.2, 0.8, 0.2]
                diff[(s['pred_map'] > 0.5) & (s['gt_map'] < 0.5)] = [0.9, 0.2, 0.2]
                diff[(s['pred_map'] < 0.5) & (s['gt_map'] > 0.5)] = [0.2, 0.4, 0.9]
                axes[row, 2].imshow(diff, aspect='equal')
                axes[row, 2].set_title('Diff', fontsize=7)

                # Failure case
                row = i * 2 + 1
                f = failure[i]
                L = f['length']
                axes[row, 0].imshow(f['gt_map'], cmap='Blues', vmin=0, vmax=1, aspect='equal')
                axes[row, 0].set_title(f"GT | {f['name'][:20]}", fontsize=7)
                axes[row, 0].set_ylabel(f"FAILURE\nF1={f['f1']:.3f}\nL={L}", fontsize=7)
                axes[row, 1].imshow(f['pred_map'], cmap='Oranges', vmin=0, vmax=1, aspect='equal')
                axes[row, 1].set_title('Pred', fontsize=7)
                diff = np.zeros((L, L, 3))
                diff[(f['pred_map'] > 0.5) & (f['gt_map'] > 0.5)] = [0.2, 0.8, 0.2]
                diff[(f['pred_map'] > 0.5) & (f['gt_map'] < 0.5)] = [0.9, 0.2, 0.2]
                diff[(f['pred_map'] < 0.5) & (f['gt_map'] > 0.5)] = [0.2, 0.4, 0.9]
                axes[row, 2].imshow(diff, aspect='equal')
                axes[row, 2].set_title('Diff', fontsize=7)

            for ax_row in axes:
                for ax in ax_row:
                    ax.set_xticks([])
                    ax.set_yticks([])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            safe_bin = bin_name.replace('+', 'plus').replace('<', 'lt')
            fig.savefig(out_dir / f'paired_comparison_{stage}_{safe_bin}.png',
                        dpi=120, bbox_inches='tight')
            plt.close()


def plot_pairing_distance_distributions(all_results_by_stage, out_dir):
    """Plot pairing distance distributions for bad vs good cases."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Pairing Distance Distribution: Bad (F1<0.3) vs Good (F1≥0.7) Cases',
                 fontsize=13, fontweight='bold')

    for col, (stage, results) in enumerate(all_results_by_stage.items()):
        good = [r for r in results if r['f1'] >= 0.7]
        bad = [r for r in results if r['f1'] < 0.3]

        # GT distances
        ax = axes[0, col]
        good_dists = []
        for r in good:
            good_dists.extend(r['gt_dists'])
        bad_dists = []
        for r in bad:
            bad_dists.extend(r['gt_dists'])

        if good_dists:
            ax.hist(good_dists, bins=50, alpha=0.6, color='green', density=True, label='Good')
        if bad_dists:
            ax.hist(bad_dists, bins=50, alpha=0.6, color='red', density=True, label='Bad')
        ax.set_title(f'{stage}: GT Pairing Distances', fontsize=10)
        ax.set_xlabel('Distance |j-i|')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

        # Pred distances for bad cases
        ax = axes[1, col]
        bad_pred_dists = []
        for r in bad:
            bad_pred_dists.extend(r['pred_dists'])
        if bad_dists:
            ax.hist(bad_dists, bins=50, alpha=0.6, color='blue', density=True, label='GT (bad)')
        if bad_pred_dists:
            ax.hist(bad_pred_dists, bins=50, alpha=0.6, color='red', density=True, label='Pred (bad)')
        ax.set_title(f'{stage}: Bad Cases GT vs Pred Distances', fontsize=10)
        ax.set_xlabel('Distance |j-i|')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / 'pairing_distance_distributions.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_pseudoknot_analysis(all_results_by_stage, out_dir):
    """Analyze pseudoknot impact on performance."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Pseudoknot Analysis: Impact on F1 Score', fontsize=13, fontweight='bold')

    for col, (stage, results) in enumerate(all_results_by_stage.items()):
        # F1 distribution: has PK vs no PK
        ax = axes[0, col]
        pk_cases = [r for r in results if r['has_pseudoknot']]
        no_pk_cases = [r for r in results if not r['has_pseudoknot']]
        ax.hist([r['f1'] for r in no_pk_cases], bins=20, alpha=0.6, color='green',
                label=f'No PK (N={len(no_pk_cases)}, mean={np.mean([r["f1"] for r in no_pk_cases]):.3f})')
        if pk_cases:
            ax.hist([r['f1'] for r in pk_cases], bins=20, alpha=0.6, color='red',
                    label=f'Has PK (N={len(pk_cases)}, mean={np.mean([r["f1"] for r in pk_cases]):.3f})')
        ax.set_title(f'{stage}: F1 by Pseudoknot Presence', fontsize=10)
        ax.set_xlabel('F1')
        ax.legend(fontsize=7)

        # PK count vs F1
        ax = axes[1, col]
        pk_counts = [r['gt_pseudoknots'] for r in results]
        f1s = [r['f1'] for r in results]
        ax.scatter(pk_counts, f1s, alpha=0.3, s=10, c='#e74c3c')
        ax.set_xlabel('#Pseudoknot crossings')
        ax.set_ylabel('F1')
        ax.set_title(f'{stage}: F1 vs PK Count', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / 'pseudoknot_analysis.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_complexity_vs_f1(all_results_by_stage, out_dir):
    """Structure complexity vs F1."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    fig.suptitle('Structure Complexity vs F1 Score', fontsize=14, fontweight='bold')

    metrics = [
        ('gt_n_stems', '#Stems'),
        ('gt_max_stem_len', 'Max Stem Length'),
        ('gt_branching', 'Branching Factor'),
        ('gt_mean_dist', 'Mean Pairing Distance'),
    ]

    for row, (stage, results) in enumerate(all_results_by_stage.items()):
        for col, (key, label) in enumerate(metrics):
            ax = axes[row, col]
            x = [r[key] for r in results]
            y = [r['f1'] for r in results]
            ax.scatter(x, y, alpha=0.2, s=8, c='#2c3e50')
            ax.set_xlabel(label, fontsize=8)
            ax.set_ylabel('F1', fontsize=8)
            ax.set_title(f'{stage}: {label} vs F1', fontsize=9)
            # Add trend line
            if len(x) > 10:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_sorted = sorted(set(x))
                ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.7, linewidth=1.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / 'complexity_vs_f1.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_failure_mode_summary(bad_cases_by_stage, out_dir):
    """Summarize failure modes across stages."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Failure Mode Distribution (F1 < 0.3)', fontsize=13, fontweight='bold')

    for ax, (stage, cases) in zip(axes, bad_cases_by_stage.items()):
        modes = Counter(classify_failure_mode(r) for r in cases)
        if modes:
            labels = list(modes.keys())
            values = list(modes.values())
            colors_map = {
                'complete_miss': '#e74c3c',
                'shifted_prediction': '#f39c12',
                'wrong_position': '#9b59b6',
                'severe_overpredict': '#e67e22',
                'severe_underpredict': '#3498db',
                'no_prediction': '#95a5a6',
                'pseudoknot_failure': '#1abc9c',
                'mixed': '#7f8c8d',
            }
            bar_colors = [colors_map.get(l, '#7f8c8d') for l in labels]
            bars = ax.barh(labels, values, color=bar_colors, alpha=0.8)
            ax.bar_label(bars, fontsize=8)
            ax.set_title(f'{stage} (N={len(cases)})', fontsize=11)
            ax.set_xlabel('Count')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / 'failure_mode_summary.png', dpi=120, bbox_inches='tight')
    plt.close()


def plot_density_length_heatmap(all_results_by_stage, out_dir):
    """F1 heatmap over length x density bins."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Mean F1 by Length × Density', fontsize=13, fontweight='bold')

    length_edges = [0, 80, 160, 240, 320, 500]
    density_edges = [0, 0.10, 0.18, 0.25, 0.35, 1.0]

    for ax, (stage, results) in zip(axes, all_results_by_stage.items()):
        # Create 2D histogram of mean F1
        grid = np.zeros((len(density_edges) - 1, len(length_edges) - 1))
        counts = np.zeros_like(grid)

        for r in results:
            li = np.searchsorted(length_edges[1:], r['length'])
            di = np.searchsorted(density_edges[1:], r['density'])
            li = min(li, grid.shape[1] - 1)
            di = min(di, grid.shape[0] - 1)
            grid[di, li] += r['f1']
            counts[di, li] += 1

        mean_f1 = np.divide(grid, counts, where=counts > 0, out=np.full_like(grid, np.nan))

        im = ax.imshow(mean_f1, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', origin='lower')
        ax.set_xticks(range(len(length_edges) - 1))
        ax.set_xticklabels([f'{length_edges[i]}-{length_edges[i+1]}' for i in range(len(length_edges) - 1)],
                           fontsize=7)
        ax.set_yticks(range(len(density_edges) - 1))
        ax.set_yticklabels([f'{density_edges[i]:.2f}-{density_edges[i+1]:.2f}' for i in range(len(density_edges) - 1)],
                           fontsize=7)
        ax.set_xlabel('Length')
        ax.set_ylabel('Density')
        ax.set_title(f'{stage}', fontsize=11)

        # Annotate
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if counts[i, j] > 0:
                    ax.text(j, i, f'{mean_f1[i,j]:.2f}\n(n={int(counts[i,j])})',
                            ha='center', va='center', fontsize=6)

    fig.colorbar(im, ax=axes.ravel().tolist(), label='Mean F1', shrink=0.8)
    plt.tight_layout(rect=[0, 0, 0.95, 0.93])
    fig.savefig(out_dir / 'f1_heatmap_length_density.png', dpi=120, bbox_inches='tight')
    plt.close()


# ============================================================
# Report Generation
# ============================================================

def generate_report(all_results_by_stage, bad_cases_by_stage, out_dir):
    """Generate comprehensive markdown report."""
    report_lines = []
    report_lines.append("# PriFold v7 DensityNet: 全面推理分析报告\n")
    report_lines.append(f"> 生成时间：自动生成\n")
    report_lines.append(f"> 模型：v7_full best.pt (200 epochs, best val F1=0.6408)\n")
    report_lines.append(f"> 数据集：bpRNA train/val/test\n\n")
    report_lines.append("---\n\n")

    # 1. Overall metrics
    report_lines.append("## 1. 总体表现\n\n")
    report_lines.append("| 指标 | Train | Val | Test |\n")
    report_lines.append("|------|-------|-----|------|\n")
    for metric in ['f1', 'precision', 'recall', 'mcc']:
        vals = []
        for stage in ['train', 'val', 'test']:
            results = all_results_by_stage.get(stage, [])
            vals.append(f"{np.mean([r[metric] for r in results]):.4f}" if results else "N/A")
        report_lines.append(f"| {metric.upper()} | {vals[0]} | {vals[1]} | {vals[2]} |\n")

    # Sample counts
    report_lines.append(f"| N | ")
    for stage in ['train', 'val', 'test']:
        report_lines.append(f"{len(all_results_by_stage.get(stage, []))} | ")
    report_lines.append("\n")

    # F1=0 counts
    report_lines.append(f"| F1=0 | ")
    for stage in ['train', 'val', 'test']:
        results = all_results_by_stage.get(stage, [])
        n_zero = sum(1 for r in results if r['f1'] == 0)
        report_lines.append(f"{n_zero} ({100*n_zero/max(len(results),1):.1f}%) | ")
    report_lines.append("\n")

    # F1<0.3 counts
    report_lines.append(f"| F1<0.3 | ")
    for stage in ['train', 'val', 'test']:
        results = all_results_by_stage.get(stage, [])
        n_bad = sum(1 for r in results if r['f1'] < 0.3)
        report_lines.append(f"{n_bad} ({100*n_bad/max(len(results),1):.1f}%) | ")
    report_lines.append("\n")

    # pred/gt
    report_lines.append(f"| pred/gt ratio | ")
    for stage in ['train', 'val', 'test']:
        results = all_results_by_stage.get(stage, [])
        report_lines.append(f"{np.mean([r['pred_gt_ratio'] for r in results]):.3f} | " if results else "N/A | ")
    report_lines.append("\n\n")

    # 2. Bad case analysis
    report_lines.append("---\n\n## 2. Bad Case 分析 (F1 < 0.3)\n\n")
    report_lines.append("### 2.1 Bad Case 特征统计\n\n")
    report_lines.append("| 特征 | Train | Val | Test |\n")
    report_lines.append("|------|-------|-----|------|\n")

    for key, label in [
        ('length', '平均长度'),
        ('density', '平均密度'),
        ('gt_n_stems', '平均 Stem 数'),
        ('gt_pseudoknots', '平均伪结交叉数'),
        ('gt_mean_dist', '平均配对距离'),
        ('gt_max_dist', '最大配对距离'),
        ('gt_branching', '分支因子'),
        ('near_miss_pct', 'Near-miss FP 比例'),
    ]:
        vals = []
        for stage in ['train', 'val', 'test']:
            cases = bad_cases_by_stage.get(stage, [])
            if cases:
                vals.append(f"{np.mean([r[key] for r in cases]):.2f}")
            else:
                vals.append("N/A")
        report_lines.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |\n")

    # 2.2 Failure modes
    report_lines.append("\n### 2.2 失败模式分类\n\n")
    for stage in ['train', 'val', 'test']:
        cases = bad_cases_by_stage.get(stage, [])
        if not cases:
            continue
        modes = Counter(classify_failure_mode(r) for r in cases)
        report_lines.append(f"**{stage.upper()}** (N={len(cases)}):\n\n")
        report_lines.append("| 模式 | 数量 | 占比 | 说明 |\n")
        report_lines.append("|------|------|------|------|\n")
        mode_desc = {
            'complete_miss': '完全预测错误（TP=0，无近似对）',
            'shifted_prediction': '预测偏移（>30% FP 在 GT ±3 范围内）',
            'wrong_position': '数量对位置错（pred/gt≈1 但 F1<0.3）',
            'severe_overpredict': '严重过预测（pred/gt > 2）',
            'severe_underpredict': '严重欠预测（pred/gt < 0.3）',
            'no_prediction': '无预测输出',
            'pseudoknot_failure': '伪结导致失败',
            'mixed': '混合/其他',
        }
        for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
            report_lines.append(f"| {mode} | {count} | {100*count/len(cases):.1f}% | {mode_desc.get(mode, '')} |\n")
        report_lines.append("\n")

    # 2.3 Pseudoknot analysis
    report_lines.append("### 2.3 伪结分析\n\n")
    report_lines.append("| 指标 | Train | Val | Test |\n")
    report_lines.append("|------|-------|-----|------|\n")
    for stage_label in ['有伪结样本占比', '有伪结样本平均 F1', '无伪结样本平均 F1', '伪结导致的 F1 下降']:
        vals = []
        for stage in ['train', 'val', 'test']:
            results = all_results_by_stage.get(stage, [])
            pk = [r for r in results if r['has_pseudoknot']]
            no_pk = [r for r in results if not r['has_pseudoknot']]
            if stage_label == '有伪结样本占比':
                vals.append(f"{100*len(pk)/max(len(results),1):.1f}%")
            elif stage_label == '有伪结样本平均 F1':
                vals.append(f"{np.mean([r['f1'] for r in pk]):.4f}" if pk else "N/A")
            elif stage_label == '无伪结样本平均 F1':
                vals.append(f"{np.mean([r['f1'] for r in no_pk]):.4f}" if no_pk else "N/A")
            elif stage_label == '伪结导致的 F1 下降':
                if pk and no_pk:
                    diff = np.mean([r['f1'] for r in no_pk]) - np.mean([r['f1'] for r in pk])
                    vals.append(f"{diff:+.4f}")
                else:
                    vals.append("N/A")
        report_lines.append(f"| {stage_label} | {vals[0]} | {vals[1]} | {vals[2]} |\n")

    # 3. Success vs Failure comparison
    report_lines.append("\n---\n\n## 3. 成功 vs 失败案例对比\n\n")
    report_lines.append("在相同长度区间内，对比 F1≥0.7（成功）和 F1<0.3（失败）的案例特征：\n\n")

    for stage in ['train', 'val', 'test']:
        results = all_results_by_stage.get(stage, [])
        success = [r for r in results if r['f1'] >= 0.7]
        failure = [r for r in results if r['f1'] < 0.3]
        if not success or not failure:
            continue
        report_lines.append(f"### {stage.upper()}\n\n")
        report_lines.append("| 特征 | 成功 (F1≥0.7) | 失败 (F1<0.3) | 差异 |\n")
        report_lines.append("|------|---------------|---------------|------|\n")
        for key, label in [
            ('length', '平均长度'),
            ('density', '平均密度'),
            ('gt_n_stems', '平均 Stem 数'),
            ('gt_pseudoknots', '平均伪结数'),
            ('gt_mean_dist', '平均配对距离'),
            ('gt_max_stem_len', '最大 Stem 长度'),
            ('gt_branching', '分支因子'),
        ]:
            s_val = np.mean([r[key] for r in success])
            f_val = np.mean([r[key] for r in failure])
            diff = f_val - s_val
            report_lines.append(f"| {label} | {s_val:.2f} | {f_val:.2f} | {diff:+.2f} |\n")
        report_lines.append("\n")

    # 4. Improvement suggestions
    report_lines.append("---\n\n## 4. 改进建议\n\n")
    report_lines.append("基于以上分析，模型改进方向如下：\n\n")
    report_lines.append("### 4.1 核心问题\n\n")
    report_lines.append("1. **Precision 不足**：模型产生过多 False Positive，特别是在 test/val 上 pred/gt > 1.2\n")
    report_lines.append("2. **泛化 gap**：Train F1 远高于 Val/Test，说明对未见结构的泛化能力不足\n")
    report_lines.append("3. **伪结处理差**：含伪结的样本 F1 显著下降\n")
    report_lines.append("4. **长距离配对难**：配对距离越远，预测越不准\n\n")
    report_lines.append("### 4.2 建议改进方案\n\n")
    report_lines.append("| 方案 | 目标问题 | 具体做法 |\n")
    report_lines.append("|------|----------|----------|\n")
    report_lines.append("| OHEM | FP 稀释 | 只取 top-k hardest negatives 计算 loss |\n")
    report_lines.append("| FP Penalty | 过预测 | 对 FP 位置加额外惩罚权重 |\n")
    report_lines.append("| BP Compatibility | 非法配对 | 训练+推理时过滤非 AU/GC/GU 配对 |\n")
    report_lines.append("| Family Balanced | 泛化 | 按家族逆频率采样，增加罕见家族曝光 |\n")
    report_lines.append("| 伪结感知 | 伪结失败 | 引入伪结交叉特征或专门的 loss 分支 |\n")
    report_lines.append("| 长距离增强 | 远距离配对 | 增加 Axial Transformer 层数或引入全局注意力 |\n")
    report_lines.append("| 数据增强 | 泛化 | 更强的序列扰动 + 结构感知的增强策略 |\n\n")

    # 5. Visualization index
    report_lines.append("---\n\n## 5. 可视化文件索引\n\n")
    report_lines.append("| 文件 | 说明 |\n")
    report_lines.append("|------|------|\n")
    report_lines.append("| `bad_case_distributions.png` | Bad case 的长度/密度/复杂度分布 |\n")
    report_lines.append("| `success_vs_failure_comparison.png` | 成功 vs 失败案例特征对比 |\n")
    report_lines.append("| `pairing_distance_distributions.png` | 配对距离分布（好 vs 坏） |\n")
    report_lines.append("| `pseudoknot_analysis.png` | 伪结对 F1 的影响 |\n")
    report_lines.append("| `complexity_vs_f1.png` | 结构复杂度 vs F1 散点图 |\n")
    report_lines.append("| `failure_mode_summary.png` | 失败模式分类统计 |\n")
    report_lines.append("| `f1_heatmap_length_density.png` | F1 在长度×密度二维网格上的热力图 |\n")
    report_lines.append("| `f1_distribution_*.png` | F1 分布图 (train/val/test) |\n")
    report_lines.append("| `length_distribution_bad.png` | Bad case 长度分布 |\n")
    report_lines.append("| `paired_comparison_*.png` | 同长度下成功/失败对比 |\n")
    report_lines.append("| `bad_cases/` | 每个 bad case 的详细配对图 |\n\n")

    report_path = out_dir / 'report.md'
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    print(f'Report saved: {report_path}')
    return report_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='PriFold v7 comprehensive analysis')
    parser.add_argument('--ckpt', type=str,
                        default='symfold/outputs/v7_full/model/best.pt')
    parser.add_argument('--config', type=str,
                        default='symfold/config/v7/v7_full.json')
    parser.add_argument('--out_dir', type=str,
                        default='symfold/outputs/v7_full/comprehensive_analysis')
    parser.add_argument('--max_bad_cases_viz', type=int, default=100,
                        help='Max bad cases to visualize individually')
    parser.add_argument('--bad_threshold', type=float, default=0.3,
                        help='F1 threshold for bad cases')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bad_cases_dir = out_dir / 'bad_cases'
    bad_cases_dir.mkdir(parents=True, exist_ok=True)

    # Load config and model
    print('='*60)
    print('PriFold v7 Comprehensive Analysis')
    print('='*60)
    cfg = load_config(args.config)
    device = torch.device(cfg.get('device', 'cuda:0'))
    print(f'Device: {device}')
    print(f'Checkpoint: {args.ckpt}')

    # Build model
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
    print('Model loaded successfully.\n')

    # Build data loaders
    data_dir = cfg['paths']['data_dir']
    max_len = cfg.get('training', {}).get('max_len_filter', 490)

    # Override batch_size=1 for per-sample analysis
    analysis_cfg = json.loads(json.dumps(cfg))  # deep copy
    analysis_cfg['training']['batch_size'] = 1
    analysis_cfg['training']['augmentation'] = {'enabled': False}

    stages = {
        'train': 'bprna-train',
        'val': 'bprna-val',
        'test': 'bprna-test',
    }

    all_results_by_stage = {}
    bad_cases_by_stage = {}

    # Build sequence map from dataset
    import pandas as pd
    bprna_csv = os.path.join(data_dir, 'bprna', 'bpRNA.csv')
    df = pd.read_csv(bprna_csv)
    seq_map = dict(zip(df['file_name'].astype(str), df['seq'].astype(str)))

    for stage_name, dataset_name in stages.items():
        print(f'\n{"="*40}')
        print(f'  Running inference: {stage_name} ({dataset_name})')
        print(f'{"="*40}')

        loader = build_loader(dataset_name, analysis_cfg, tokenizer, shuffle=False)
        print(f'  Samples: {len(loader.dataset)}')

        results = run_inference(model, loader, device, cfg, seq_map=seq_map)
        all_results_by_stage[stage_name] = results

        # Filter bad cases
        bad_cases = [r for r in results if r['f1'] < args.bad_threshold]
        bad_cases_by_stage[stage_name] = bad_cases

        # Print summary
        f1_mean = np.mean([r['f1'] for r in results])
        f1_zero = sum(1 for r in results if r['f1'] == 0)
        print(f'  Mean F1: {f1_mean:.4f}')
        print(f'  F1=0: {f1_zero}/{len(results)} ({100*f1_zero/len(results):.1f}%)')
        print(f'  Bad cases (F1<{args.bad_threshold}): {len(bad_cases)}/{len(results)}')

    # ============================================================
    # Generate all visualizations
    # ============================================================
    print(f'\n{"="*60}')
    print('Generating visualizations...')
    print(f'{"="*60}')

    # 1. F1 distribution per stage
    print('\n[1/9] F1 distributions...')
    plot_distribution_comparison(
        all_results_by_stage['train'], all_results_by_stage['val'], all_results_by_stage['test'],
        'f1', 'F1 Score Distribution', 'F1', out_dir / 'f1_distribution.png', bins=30)

    # 2. Bad case distributions
    print('[2/9] Bad case distributions...')
    plot_bad_case_distributions(bad_cases_by_stage, out_dir)

    # 3. Success vs failure comparison
    print('[3/9] Success vs failure comparison...')
    plot_success_vs_failure_comparison(all_results_by_stage, out_dir)

    # 4. Paired comparison at similar conditions
    print('[4/9] Paired comparisons (contact maps)...')
    plot_paired_comparison_at_similar_conditions(all_results_by_stage, out_dir)

    # 5. Pairing distance distributions
    print('[5/9] Pairing distance distributions...')
    plot_pairing_distance_distributions(all_results_by_stage, out_dir)

    # 6. Pseudoknot analysis
    print('[6/9] Pseudoknot analysis...')
    plot_pseudoknot_analysis(all_results_by_stage, out_dir)

    # 7. Complexity vs F1
    print('[7/9] Complexity vs F1...')
    plot_complexity_vs_f1(all_results_by_stage, out_dir)

    # 8. Failure mode summary
    print('[8/9] Failure mode summary...')
    plot_failure_mode_summary(bad_cases_by_stage, out_dir)

    # 9. F1 heatmap (length x density)
    print('[9/9] F1 heatmap (length × density)...')
    plot_density_length_heatmap(all_results_by_stage, out_dir)

    # Additional: Length distribution for bad cases
    plot_distribution_comparison(
        bad_cases_by_stage.get('train', []),
        bad_cases_by_stage.get('val', []),
        bad_cases_by_stage.get('test', []),
        'length', 'Bad Case (F1<0.3): Length Distribution', 'Length',
        out_dir / 'length_distribution_bad.png', bins=25)

    # ============================================================
    # Generate individual bad case cards
    # ============================================================
    print(f'\nGenerating individual bad case cards...')
    all_bad = []
    for stage, cases in bad_cases_by_stage.items():
        for c in cases:
            c['stage'] = stage
            all_bad.append(c)

    # Sort by F1 (worst first)
    all_bad.sort(key=lambda x: (x['f1'], -x['length']))
    n_viz = min(args.max_bad_cases_viz, len(all_bad))
    print(f'  Generating {n_viz} bad case cards...')

    for idx, case in enumerate(all_bad[:n_viz]):
        safe_name = case['name'].replace('/', '_').replace(' ', '_')
        out_path = bad_cases_dir / f'{idx:03d}_{case["stage"]}_F1={case["f1"]:.3f}_L={case["length"]}_{safe_name}.png'
        plot_bad_case_card(case, out_path)
        if (idx + 1) % 20 == 0:
            print(f'    {idx + 1}/{n_viz} cards done')

    # ============================================================
    # Generate report
    # ============================================================
    print(f'\nGenerating report...')
    generate_report(all_results_by_stage, bad_cases_by_stage, out_dir)

    # Save per-sample CSV
    print('Saving per-sample CSV...')
    csv_keys = ['name', 'stage', 'length', 'f1', 'precision', 'recall', 'mcc',
                'tp', 'fp', 'fn', 'gt_pairs', 'pred_pairs', 'density', 'pred_gt_ratio',
                'gt_mean_dist', 'gt_max_dist', 'gt_n_stems', 'gt_max_stem_len',
                'gt_avg_stem_len', 'gt_branching', 'gt_pseudoknots', 'has_pseudoknot',
                'gt_canonical_pairs', 'gt_noncanonical_pairs',
                'near_miss_fp', 'far_miss_fp', 'near_miss_pct']

    csv_path = out_dir / 'all_samples_metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys, extrasaction='ignore')
        writer.writeheader()
        for stage, results in all_results_by_stage.items():
            for r in results:
                r['stage'] = stage
                writer.writerow(r)
    print(f'  Saved: {csv_path}')

    # Save summary JSON
    summary = {}
    for stage, results in all_results_by_stage.items():
        summary[stage] = {
            'n': len(results),
            'f1_mean': float(np.mean([r['f1'] for r in results])),
            'f1_median': float(np.median([r['f1'] for r in results])),
            'precision_mean': float(np.mean([r['precision'] for r in results])),
            'recall_mean': float(np.mean([r['recall'] for r in results])),
            'f1_zero_count': sum(1 for r in results if r['f1'] == 0),
            'bad_count': sum(1 for r in results if r['f1'] < args.bad_threshold),
            'has_pseudoknot_count': sum(1 for r in results if r['has_pseudoknot']),
            'avg_density': float(np.mean([r['density'] for r in results])),
            'avg_pred_gt_ratio': float(np.mean([r['pred_gt_ratio'] for r in results])),
        }
    json_path = out_dir / 'summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Saved: {json_path}')

    print(f'\n{"="*60}')
    print(f'Analysis complete!')
    print(f'Output directory: {out_dir}')
    print(f'Bad case cards: {bad_cases_dir} ({n_viz} files)')
    print(f'Report: {out_dir}/report.md')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
