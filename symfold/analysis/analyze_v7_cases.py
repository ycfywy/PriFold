# -*- coding: utf-8 -*-
"""Per-RNA detailed case analysis for PriFold v7 DensityNet.

Analyzes every sample in bprna-train/val/test,
outputs per-sample metrics, visualizations, and detailed breakdowns.

Usage:
  python symfold/analyze_v7_cases.py \
    --ckpt symfold/outputs/v7_full/model/best.pt \
    --config symfold/config/v7_full.json \
    --out_dir symfold/outputs/v7_full/case_analysis \
    --test_sets bprna-train,bprna-val,bprna-test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['figure.dpi'] = 150

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_loader
from symfold.train.train_v7 import build_model, load_config, move_to_device


# ============================================================
# Per-sample metrics
# ============================================================

def per_sample_metrics(pred: torch.Tensor, target: torch.Tensor, length: int) -> dict:
    """Compute detailed per-sample metrics."""
    p = pred.detach().cpu().float().squeeze()[:length, :length] > 0.5
    y = target.detach().cpu().float().squeeze()[:length, :length] > 0.5
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    idx = torch.arange(length)
    mask &= (idx.view(length, 1) - idx.view(1, length)).abs() >= 3
    p_masked = p[mask]
    y_masked = y[mask]
    tp = int((p_masked & y_masked).sum())
    fp = int((p_masked & ~y_masked).sum())
    fn = int((~p_masked & y_masked).sum())
    tn = int((~p_masked & ~y_masked).sum())
    gt_pairs_count = tp + fn
    pred_pairs_count = tp + fp
    # Handle edge case: gt=0 and pred=0 → perfect
    if gt_pairs_count == 0 and pred_pairs_count == 0:
        precision, recall, f1, mcc = 1.0, 1.0, 1.0, 1.0
    elif gt_pairs_count == 0 and pred_pairs_count > 0:
        precision, recall, f1, mcc = 0.0, 1.0, 0.0, 0.0
    else:
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
        mcc = ((tp * tn) - (fp * fn)) / denom

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

    # Stem analysis
    gt_stems = 0
    pred_stems = 0
    for i in range(length - 1):
        for j in range(i + 4, length):
            if y_full[i, j] and y_full[i + 1, j - 1]:
                gt_stems += 1
            if p_full[i, j] and p_full[i + 1, j - 1]:
                pred_stems += 1

    # Shift analysis: check if pred pairs are shifted from GT
    shift_counts = []
    for i in range(length):
        for j in range(i + 3, length):
            if p_full[i, j]:
                # Check if there's a GT pair within ±k offset
                found_near = False
                for di in range(-3, 4):
                    for dj in range(-3, 4):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < length and 0 <= nj < length and y_full[ni, nj]:
                            shift_counts.append(abs(di) + abs(dj))
                            found_near = True
                            break
                    if found_near:
                        break
                if not found_near:
                    shift_counts.append(-1)  # no nearby GT pair

    near_miss_count = sum(1 for s in shift_counts if 1 <= s <= 3)
    far_miss_count = sum(1 for s in shift_counts if s == -1)

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
        'near_miss_count': near_miss_count,
        'far_miss_count': far_miss_count,
        'near_miss_pct': near_miss_count / max(pred_pairs_count, 1),
    }


def length_bin(length: int) -> str:
    if length < 80: return '<80'
    if length < 160: return '80-159'
    if length < 240: return '160-239'
    if length < 320: return '240-319'
    if length < 400: return '320-399'
    return '400+'


def density_bin(density: float) -> str:
    if density < 0.10: return '<0.10'
    if density < 0.18: return '0.10-0.18'
    if density < 0.25: return '0.18-0.25'
    if density < 0.35: return '0.25-0.35'
    return '>=0.35'


def f1_bin(f1: float) -> str:
    if f1 == 0: return 'F1=0'
    if f1 < 0.3: return '0<F1<0.3'
    if f1 < 0.5: return '0.3<=F1<0.5'
    if f1 < 0.7: return '0.5<=F1<0.7'
    if f1 < 0.9: return '0.7<=F1<0.9'
    return 'F1>=0.9'


# ============================================================
# Analysis
# ============================================================

def analyze_groups(rows: list[dict], group_key: str) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r[group_key]].append(r)
    result = {}
    for gname, items in sorted(groups.items()):
        n = len(items)
        result[gname] = {
            'n': n,
            'pct': f"{100*n/len(rows):.1f}%",
            'f1': float(np.mean([x['f1'] for x in items])),
            'precision': float(np.mean([x['precision'] for x in items])),
            'recall': float(np.mean([x['recall'] for x in items])),
            'pred_gt_ratio': float(np.mean([x['pred_gt_ratio'] for x in items])),
            'density': float(np.mean([x['density'] for x in items])),
            'length': float(np.mean([x['length'] for x in items])),
            'gt_pairs': float(np.mean([x['gt_pairs'] for x in items])),
            'pred_pairs': float(np.mean([x['pred_pairs'] for x in items])),
            'f1_zero_count': sum(1 for x in items if x['f1'] == 0),
            'near_miss_pct': float(np.mean([x['near_miss_pct'] for x in items])),
        }
    return result


def failure_mode_analysis(bad_cases: list[dict]) -> dict:
    """Classify bad cases into failure modes."""
    modes = {
        'complete_miss': [],       # F1=0, tp=0, 但有预测
        'shifted_prediction': [],  # near_miss_pct > 0.3, 预测位置偏移
        'wrong_position': [],      # pred/gt ~ 1 但 F1 < 0.3
        'overpredict': [],         # pred/gt > 2
        'underpredict': [],        # pred/gt < 0.5
        'other': [],
    }
    for r in bad_cases:
        if r['f1'] == 0 and r['tp'] == 0 and r['pred_pairs'] > 0:
            if r['near_miss_pct'] > 0.3:
                modes['shifted_prediction'].append(r)
            else:
                modes['complete_miss'].append(r)
        elif r['pred_gt_ratio'] > 2:
            modes['overpredict'].append(r)
        elif r['pred_gt_ratio'] < 0.5:
            modes['underpredict'].append(r)
        elif 0.7 < r['pred_gt_ratio'] < 1.3 and r['f1'] < 0.3:
            if r['near_miss_pct'] > 0.3:
                modes['shifted_prediction'].append(r)
            else:
                modes['wrong_position'].append(r)
        else:
            modes['other'].append(r)

    result = {}
    total = len(bad_cases)
    for mode_name, cases in modes.items():
        if cases:
            result[mode_name] = {
                'n': len(cases),
                'pct': f"{100*len(cases)/max(total,1):.1f}%",
                'avg_f1': float(np.mean([r['f1'] for r in cases])),
                'avg_length': float(np.mean([r['length'] for r in cases])),
                'avg_density': float(np.mean([r['density'] for r in cases])),
                'avg_pred_gt_ratio': float(np.mean([r['pred_gt_ratio'] for r in cases])),
                'avg_near_miss_pct': float(np.mean([r['near_miss_pct'] for r in cases])),
            }
    return result


# ============================================================
# Visualization
# ============================================================

def plot_contact_map(pred, gt, length, name, f1, precision, recall, pred_gt_ratio,
                     ax_pred, ax_gt, ax_diff):
    p = pred[:length, :length]
    g = gt[:length, :length]
    ax_gt.imshow(g, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    ax_gt.set_title('GT', fontsize=7)
    ax_gt.set_xticks([]); ax_gt.set_yticks([])
    ax_pred.imshow(p, cmap='Oranges', vmin=0, vmax=1, aspect='equal')
    ax_pred.set_title('Pred', fontsize=7)
    ax_pred.set_xticks([]); ax_pred.set_yticks([])
    # Diff
    diff = np.zeros((length, length, 3))
    tp_mask = (p > 0.5) & (g > 0.5)
    fp_mask = (p > 0.5) & (g < 0.5)
    fn_mask = (p < 0.5) & (g > 0.5)
    diff[tp_mask] = [0.2, 0.8, 0.2]
    diff[fp_mask] = [0.9, 0.2, 0.2]
    diff[fn_mask] = [0.2, 0.4, 0.9]
    ax_diff.imshow(diff, aspect='equal')
    ax_diff.set_title(f'F1={f1:.3f} P={precision:.2f} R={recall:.2f}\np/g={pred_gt_ratio:.2f}', fontsize=6)
    ax_diff.set_xticks([]); ax_diff.set_yticks([])


def visualize_cases(results, out_dir, n_cases=20, stage_label='test'):
    """Generate contact map visualizations for best and worst cases."""
    sorted_by_f1 = sorted(results, key=lambda x: (x['f1'], -x['length']))
    worst = [r for r in sorted_by_f1 if r['length'] >= 40][:n_cases]
    best = sorted(results, key=lambda x: -x['f1'])[:n_cases]

    for label, cases in [('worst', worst), ('best', best)]:
        n_rows = 5
        n_cols = 4
        fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
        fig.suptitle(f'v7 DensityNet: {label.upper()} {n_cases} Cases ({stage_label})',
                     fontsize=13, fontweight='bold')

        for idx, case in enumerate(cases[:n_rows * n_cols]):
            row_block = idx // n_cols
            col = idx % n_cols
            ax_gt = axes[row_block * 3, col]
            ax_pred = axes[row_block * 3 + 1, col]
            ax_diff = axes[row_block * 3 + 2, col]

            plot_contact_map(
                case['pred'], case['gt'], case['length'],
                case['name'], case['f1'], case['precision'], case['recall'],
                case['pred_gt_ratio'], ax_pred, ax_gt, ax_diff)
            short_name = case['name'].replace('bpRNA_', '')
            ax_gt.set_ylabel(f'{short_name}\nL={case["length"]}', fontsize=5,
                             rotation=0, labelpad=50, va='center')

        for i in range(len(cases), n_rows * n_cols):
            row_block = i // n_cols
            col = i % n_cols
            for j in range(3):
                axes[row_block * 3 + j, col].axis('off')

        legend_elements = [
            Patch(facecolor=[0.2, 0.8, 0.2], label='TP'),
            Patch(facecolor=[0.9, 0.2, 0.2], label='FP'),
            Patch(facecolor=[0.2, 0.4, 0.9], label='FN'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9)
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        path = out_dir / f'{label}_{n_cases}_contact_maps_{stage_label}.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        print(f'  Saved: {path}')
        plt.close()


def visualize_score_heatmaps(results, out_dir, stage_label='test'):
    """Score heatmaps for F1=0 and high F1 cases."""
    f1_zero = sorted([r for r in results if r['f1'] == 0 and r['length'] >= 50],
                     key=lambda x: -x['length'])[:4]
    f1_low = sorted([r for r in results if 0.05 < r['f1'] < 0.3 and r['length'] >= 60],
                    key=lambda x: x['f1'])[:3]
    f1_high = sorted([r for r in results if r['f1'] >= 0.95],
                     key=lambda x: -x['length'])[:3]
    cases = f1_zero + f1_low + f1_high
    if not cases:
        return

    fig, axes = plt.subplots(len(cases), 3, figsize=(12, len(cases) * 2.5))
    fig.suptitle(f'v7 Score Heatmaps ({stage_label})', fontsize=13, fontweight='bold')
    if len(cases) == 1:
        axes = axes.reshape(1, -1)

    for idx, case in enumerate(cases):
        L = case['length']
        ax = axes[idx, 0]
        ax.imshow(case['score'][:L, :L], cmap='hot', vmin=0, vmax=1, aspect='equal')
        ax.set_title('Score', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        short_name = case['name'].replace('bpRNA_', '')
        ax.set_ylabel(f'{short_name}\nL={L} F1={case["f1"]:.2f}', fontsize=6,
                      rotation=0, labelpad=60, va='center')

        ax = axes[idx, 1]
        ax.imshow(case['gt'][:L, :L], cmap='Blues', vmin=0, vmax=1, aspect='equal')
        ax.set_title('GT', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[idx, 2]
        diff = np.zeros((L, L, 3))
        p_bin = case['pred'][:L, :L] > 0.5
        g_bin = case['gt'][:L, :L] > 0.5
        diff[(p_bin & g_bin)] = [0.2, 0.8, 0.2]
        diff[(p_bin & ~g_bin)] = [0.9, 0.2, 0.2]
        diff[(~p_bin & g_bin)] = [0.2, 0.4, 0.9]
        ax.imshow(diff, aspect='equal')
        ax.set_title(f'TP/FP/FN  p/g={case["pred_gt_ratio"]:.2f}', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    legend_elements = [
        Patch(facecolor=[0.2, 0.8, 0.2], label='TP'),
        Patch(facecolor=[0.9, 0.2, 0.2], label='FP'),
        Patch(facecolor=[0.2, 0.4, 0.9], label='FN'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    path = out_dir / f'score_heatmaps_{stage_label}.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close()


def visualize_overview(all_rows, out_dir):
    """Main overview plot: distributions by length, density, category."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('PriFold v7 DensityNet: Case Analysis Overview', fontsize=14, fontweight='bold')

    # 1. F1 distribution RFAM vs non-RFAM
    ax = axes[0, 0]
    rfam = [r['f1'] for r in all_rows if 'RFAM' in r['name']]
    non_rfam = [r['f1'] for r in all_rows if 'RFAM' not in r['name']]
    bins = np.linspace(0, 1, 21)
    ax.hist(rfam, bins=bins, alpha=0.7, label=f'RFAM (N={len(rfam)})', color='#e74c3c')
    ax.hist(non_rfam, bins=bins, alpha=0.7, label=f'non-RFAM (N={len(non_rfam)})', color='#2ecc71')
    if rfam:
        ax.axvline(np.mean(rfam), color='#c0392b', linestyle='--',
                   label=f'RFAM mean={np.mean(rfam):.3f}')
    if non_rfam:
        ax.axvline(np.mean(non_rfam), color='#27ae60', linestyle='--',
                   label=f'non-RFAM mean={np.mean(non_rfam):.3f}')
    ax.set_xlabel('F1 Score')
    ax.set_ylabel('Count')
    ax.set_title('F1 Distribution: RFAM vs non-RFAM')
    ax.legend(fontsize=7)

    # 2. F1 vs length
    ax = axes[0, 1]
    is_rfam = ['RFAM' in r['name'] for r in all_rows]
    lengths = [r['length'] for r in all_rows]
    f1s = [r['f1'] for r in all_rows]
    colors = ['#e74c3c' if ir else '#2ecc71' for ir in is_rfam]
    ax.scatter(lengths, f1s, c=colors, s=8, alpha=0.3)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('F1')
    ax.set_title('F1 vs Length')
    ax.legend(handles=[Patch(facecolor='#e74c3c', label='RFAM'),
                       Patch(facecolor='#2ecc71', label='non-RFAM')], fontsize=8)

    # 3. F1 vs density
    ax = axes[0, 2]
    densities = [r['density'] for r in all_rows]
    ax.scatter(densities, f1s, c=colors, s=8, alpha=0.3)
    ax.set_xlabel('Density')
    ax.set_ylabel('F1')
    ax.set_title('F1 vs Density')

    # 4. pred/gt vs F1
    ax = axes[1, 0]
    ratios = [min(r['pred_gt_ratio'], 5.0) for r in all_rows]
    sc = ax.scatter(ratios, f1s, c=densities, cmap='RdYlGn', s=8, alpha=0.4, vmin=0, vmax=0.4)
    plt.colorbar(sc, ax=ax, label='density')
    ax.axvline(1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('pred/gt ratio')
    ax.set_ylabel('F1')
    ax.set_title('F1 vs pred/gt (colored by density)')

    # 5. F1 by length bin
    ax = axes[1, 1]
    bins_order = ['<80', '80-159', '160-239', '240-319', '320-399', '400+']
    bin_f1s = []
    bin_counts = []
    for b in bins_order:
        subset = [r for r in all_rows if r['length_bin'] == b]
        bin_f1s.append(np.mean([r['f1'] for r in subset]) if subset else 0)
        bin_counts.append(len(subset))
    bars = ax.bar(range(len(bins_order)), bin_f1s, color='#3498db', alpha=0.7)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'N={bin_counts[i]}', ha='center', fontsize=7)
    ax.set_xticks(range(len(bins_order)))
    ax.set_xticklabels(bins_order)
    ax.set_xlabel('Length Bin')
    ax.set_ylabel('Mean F1')
    ax.set_title('F1 by Length Bin')
    ax.set_ylim(0, 1.0)

    # 6. F1 by density bin
    ax = axes[1, 2]
    dbins_order = ['<0.10', '0.10-0.18', '0.18-0.25', '0.25-0.35', '>=0.35']
    dbin_f1s = []
    dbin_counts = []
    dbin_pg = []
    for b in dbins_order:
        subset = [r for r in all_rows if r['density_bin'] == b]
        dbin_f1s.append(np.mean([r['f1'] for r in subset]) if subset else 0)
        dbin_counts.append(len(subset))
        dbin_pg.append(np.mean([r['pred_gt_ratio'] for r in subset]) if subset else 1)
    bars = ax.bar(range(len(dbins_order)), dbin_f1s, color='#e67e22', alpha=0.7)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'N={dbin_counts[i]}\np/g={dbin_pg[i]:.2f}', ha='center', fontsize=6)
    ax.set_xticks(range(len(dbins_order)))
    ax.set_xticklabels(dbins_order)
    ax.set_xlabel('Density Bin')
    ax.set_ylabel('Mean F1')
    ax.set_title('F1 by Density Bin')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    path = out_dir / 'case_analysis_overview.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close()


def visualize_failure_modes(all_rows, out_dir):
    """Failure mode breakdown."""
    bad = [r for r in all_rows if r['f1'] < 0.3]
    if not bad:
        print("  No bad cases (F1<0.3)!")
        return

    modes = failure_mode_analysis(bad)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'v7 Failure Mode Analysis (F1<0.3, N={len(bad)})', fontsize=13, fontweight='bold')

    # Pie chart
    ax = axes[0]
    labels = []
    sizes = []
    colors_pie = ['#e74c3c', '#f39c12', '#9b59b6', '#3498db', '#1abc9c', '#95a5a6']
    for i, (mode, info) in enumerate(modes.items()):
        labels.append(f'{mode}\nn={info["n"]} ({info["pct"]})')
        sizes.append(info['n'])
    ax.pie(sizes, labels=labels, colors=colors_pie[:len(sizes)], autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 7})
    ax.set_title('Failure Mode Distribution')

    # Shift analysis for F1=0 cases
    ax = axes[1]
    f1_zero = [r for r in all_rows if r['f1'] == 0 and r['pred_pairs'] > 0]
    if f1_zero:
        near_pcts = [r['near_miss_pct'] for r in f1_zero]
        ax.hist(near_pcts, bins=20, color='#e74c3c', alpha=0.7)
        ax.axvline(np.mean(near_pcts), color='black', linestyle='--',
                   label=f'mean={np.mean(near_pcts):.3f}')
        ax.set_xlabel('Near-miss fraction (pairs within ±3 of GT)')
        ax.set_ylabel('Count')
        ax.set_title(f'F1=0 Cases: Are predictions shifted? (N={len(f1_zero)})')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No F1=0 cases', ha='center', va='center')

    # Bad cases: density vs length scatter
    ax = axes[2]
    for r in bad:
        color = '#e74c3c' if r['f1'] == 0 else '#f39c12' if r['f1'] < 0.15 else '#3498db'
        ax.scatter(r['length'], r['density'], c=color, s=15, alpha=0.5)
    ax.set_xlabel('Length')
    ax.set_ylabel('Density')
    ax.set_title('Bad Cases: Length vs Density')
    ax.legend(handles=[
        Patch(facecolor='#e74c3c', label='F1=0'),
        Patch(facecolor='#f39c12', label='0<F1<0.15'),
        Patch(facecolor='#3498db', label='0.15≤F1<0.3'),
    ], fontsize=8)

    plt.tight_layout()
    path = out_dir / 'failure_mode_analysis.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close()


def visualize_per_stage_comparison(stage_rows_dict, out_dir):
    """Compare train/val/test performance."""
    stages = list(stage_rows_dict.keys())
    if len(stages) < 2:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('v7 DensityNet: Train/Val/Test Comparison', fontsize=13, fontweight='bold')

    colors_stage = {'bprna-train': '#3498db', 'bprna-val': '#e67e22', 'bprna-test': '#2ecc71'}

    # F1 distribution per stage
    ax = axes[0]
    bins = np.linspace(0, 1, 21)
    for stage in stages:
        rows = stage_rows_dict[stage]
        f1s = [r['f1'] for r in rows]
        label = f'{stage} (N={len(rows)}, μ={np.mean(f1s):.3f})'
        ax.hist(f1s, bins=bins, alpha=0.5, label=label,
                color=colors_stage.get(stage, '#95a5a6'))
    ax.set_xlabel('F1')
    ax.set_ylabel('Count')
    ax.set_title('F1 Distribution by Stage')
    ax.legend(fontsize=7)

    # F1=0 rate per stage
    ax = axes[1]
    stage_names = []
    f1_zero_rates = []
    f1_below03_rates = []
    f1_above09_rates = []
    for stage in stages:
        rows = stage_rows_dict[stage]
        n = len(rows)
        stage_names.append(stage.replace('bprna-', ''))
        f1_zero_rates.append(100 * sum(1 for r in rows if r['f1'] == 0) / n)
        f1_below03_rates.append(100 * sum(1 for r in rows if r['f1'] < 0.3) / n)
        f1_above09_rates.append(100 * sum(1 for r in rows if r['f1'] >= 0.9) / n)

    x = np.arange(len(stage_names))
    width = 0.25
    ax.bar(x - width, f1_zero_rates, width, label='F1=0 %', color='#e74c3c', alpha=0.7)
    ax.bar(x, f1_below03_rates, width, label='F1<0.3 %', color='#f39c12', alpha=0.7)
    ax.bar(x + width, f1_above09_rates, width, label='F1≥0.9 %', color='#2ecc71', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Quality Distribution by Stage')
    ax.legend(fontsize=8)

    # Mean metrics per stage
    ax = axes[2]
    metrics_labels = ['F1', 'Precision', 'Recall', 'pred/gt']
    for i, stage in enumerate(stages):
        rows = stage_rows_dict[stage]
        vals = [
            np.mean([r['f1'] for r in rows]),
            np.mean([r['precision'] for r in rows]),
            np.mean([r['recall'] for r in rows]),
            np.mean([r['pred_gt_ratio'] for r in rows]),
        ]
        x = np.arange(len(metrics_labels))
        ax.bar(x + i * 0.25, vals, 0.25, label=stage.replace('bprna-', ''),
               color=colors_stage.get(stage, '#95a5a6'), alpha=0.7)
    ax.set_xticks(np.arange(len(metrics_labels)) + 0.25)
    ax.set_xticklabels(metrics_labels)
    ax.set_ylabel('Value')
    ax.set_title('Mean Metrics by Stage')
    ax.legend(fontsize=8)
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    path = out_dir / 'stage_comparison.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close()


# ============================================================
# Loss analysis for F1=0 cases
# ============================================================

def compute_loss_for_cases(model, results_with_tensors, device, out_dir):
    """Compute per-sample loss breakdown for F1=0 cases vs good cases.
    This helps understand WHY the model produces F1=0 predictions."""
    f1_zero = [r for r in results_with_tensors if r['f1'] == 0 and r['pred_pairs'] > 0][:20]
    f1_good = [r for r in results_with_tensors if r['f1'] >= 0.9][:20]

    if not f1_zero:
        print("  No F1=0 cases to analyze loss for.")
        return

    print(f"\n  === Loss Analysis: F1=0 cases vs F1≥0.9 cases ===")
    print(f"  F1=0 cases (N={len(f1_zero)}):")
    print(f"    avg length: {np.mean([r['length'] for r in f1_zero]):.0f}")
    print(f"    avg pred_pairs: {np.mean([r['pred_pairs'] for r in f1_zero]):.1f}")
    print(f"    avg gt_pairs: {np.mean([r['gt_pairs'] for r in f1_zero]):.1f}")
    print(f"    avg pred/gt: {np.mean([r['pred_gt_ratio'] for r in f1_zero]):.2f}")
    print(f"    avg near_miss_pct: {np.mean([r['near_miss_pct'] for r in f1_zero]):.3f}")
    print(f"  F1≥0.9 cases (N={len(f1_good)}):")
    print(f"    avg length: {np.mean([r['length'] for r in f1_good]):.0f}")
    print(f"    avg pred_pairs: {np.mean([r['pred_pairs'] for r in f1_good]):.1f}")
    print(f"    avg gt_pairs: {np.mean([r['gt_pairs'] for r in f1_good]):.1f}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--test_sets', default='bprna-val,bprna-test')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--worst_k', type=int, default=100)
    ap.add_argument('--max_train_samples', type=int, default=2000,
                    help='Max train samples to analyze (train is large)')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = load_config(args.config) if args.config else ckpt.get('config', None)
    if cfg is None:
        cfg = load_config('symfold/config/v7_full.json')
    scfg = cfg.get('sampling', {})

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
    stage_rows_dict = {}
    vis_results = {}  # store numpy arrays for visualization

    print(f"Evaluating v7 DensityNet on: {stages}")
    print(f"Sampling config: {scfg}")

    with torch.no_grad():
        for stage in stages:
            loader = build_loader(stage, cfg, tokenizer, shuffle=False)
            rows = []
            vis_stage = []
            max_samples = args.max_train_samples if 'train' in stage else None

            for step, batch in enumerate(loader):
                batch = move_to_device(batch, device)
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

                for i in range(pred.shape[0]):
                    length = int(batch['length'][i].item())
                    m = per_sample_metrics(pred[i], batch['contact'][i], length)
                    row = {
                        'stage': stage,
                        'name': batch['names'][i],
                        'dataset': batch['datasets'][i] if 'datasets' in batch else 'bprna',
                        'length': length,
                        'length_bin': length_bin(length),
                        'density_bin': density_bin(m['density']),
                        'f1_bin': f1_bin(m['f1']),
                        **m,
                    }
                    rows.append(row)
                    all_rows.append(row)

                    # Store arrays for visualization (test/val only, limited)
                    if 'train' not in stage and len(vis_stage) < 2000:
                        vis_stage.append({
                            **row,
                            'pred': pred[i].cpu().float().squeeze().numpy(),
                            'gt': batch['contact'][i].cpu().float().squeeze().numpy(),
                            'score': score[i].cpu().float().squeeze().numpy(),
                        })

                if step % 20 == 0:
                    print(f"  [{stage}] step={step}/{len(loader)}, samples={len(rows)}")

                if max_samples and len(rows) >= max_samples:
                    print(f"  [{stage}] reached max_samples={max_samples}, stopping.")
                    break

            stage_rows_dict[stage] = rows

            # Save per-stage CSV
            rows_sorted = sorted(rows, key=lambda x: (x['f1'], -x['length']))
            csv_path = out_dir / f'{stage.replace("-", "_")}_cases.csv'
            csv_keys = [k for k in rows_sorted[0].keys() if k not in ('pred', 'gt', 'score')]
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_keys,
                                        extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows_sorted)
            with open(out_dir / f'{stage.replace("-", "_")}_worst_{args.worst_k}.json', 'w') as f:
                worst_k = [{k: v for k, v in r.items() if k not in ('pred', 'gt', 'score')}
                           for r in rows_sorted[:args.worst_k]]
                json.dump(worst_k, f, indent=2)
            print(f'  [{stage}] N={len(rows)}, saved -> {csv_path}')

            if vis_stage:
                vis_results[stage] = vis_stage

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

    # Overall
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

    # RFAM vs non-RFAM
    rfam_rows = [r for r in all_rows if 'RFAM' in r['name']]
    non_rfam_rows = [r for r in all_rows if 'RFAM' not in r['name']]
    rfam_analysis = {
        'n': len(rfam_rows),
        'f1': float(np.mean([r['f1'] for r in rfam_rows])) if rfam_rows else 0,
        'f1_zero_count': sum(1 for r in rfam_rows if r['f1'] == 0),
        'f1_below_03_count': sum(1 for r in rfam_rows if r['f1'] < 0.3),
    }
    non_rfam_analysis = {
        'n': len(non_rfam_rows),
        'f1': float(np.mean([r['f1'] for r in non_rfam_rows])) if non_rfam_rows else 0,
        'f1_zero_count': sum(1 for r in non_rfam_rows if r['f1'] == 0),
        'f1_below_03_count': sum(1 for r in non_rfam_rows if r['f1'] < 0.3),
    }

    # Failure mode analysis
    bad_cases = [r for r in all_rows if r['f1'] < 0.3]
    modes = failure_mode_analysis(bad_cases) if bad_cases else {}

    # Compile report
    report = {
        'overall': overall,
        'by_stage': stage_summary,
        'by_length_bin': by_length,
        'by_density_bin': by_density,
        'by_f1_bin': by_f1,
        'rfam_analysis': rfam_analysis,
        'non_rfam_analysis': non_rfam_analysis,
        'failure_modes': modes,
        'n_bad_cases': len(bad_cases),
    }

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
    print(f"\nRFAM: N={rfam_analysis['n']} F1={rfam_analysis['f1']:.4f} "
          f"F1=0:{rfam_analysis['f1_zero_count']}")
    print(f"non-RFAM: N={non_rfam_analysis['n']} F1={non_rfam_analysis['f1']:.4f} "
          f"F1=0:{non_rfam_analysis['f1_zero_count']}")
    print(f"\nBy Length:")
    for k, v in by_length.items():
        print(f"  {k:10s}: n={v['n']:4d} F1={v['f1']:.4f} P={v['precision']:.4f} R={v['recall']:.4f} "
              f"pred/gt={v['pred_gt_ratio']:.3f} F1=0:{v['f1_zero_count']}")
    print(f"\nBy Density:")
    for k, v in by_density.items():
        print(f"  {k:10s}: n={v['n']:4d} F1={v['f1']:.4f} P={v['precision']:.4f} R={v['recall']:.4f} "
              f"pred/gt={v['pred_gt_ratio']:.3f} F1=0:{v['f1_zero_count']}")
    print(f"\nBy F1 Bin:")
    for k, v in by_f1.items():
        print(f"  {k:15s}: n={v['n']:4d} ({v['pct']}) len={v['length']:.0f} "
              f"density={v['density']:.3f} pred/gt={v['pred_gt_ratio']:.3f}")
    print(f"\nFailure Modes (F1<0.3, N={len(bad_cases)}):")
    for mode_name, info in modes.items():
        print(f"  {mode_name}: n={info['n']} ({info['pct']}) "
              f"avg_len={info['avg_length']:.0f} avg_density={info['avg_density']:.3f} "
              f"avg_pred/gt={info['avg_pred_gt_ratio']:.2f} near_miss={info['avg_near_miss_pct']:.3f}")

    # ---- Visualizations ----
    print(f"\n{'='*60}")
    print("Generating visualizations...")
    print(f"{'='*60}")

    visualize_overview(all_rows, out_dir)
    visualize_failure_modes(all_rows, out_dir)
    visualize_per_stage_comparison(stage_rows_dict, out_dir)

    # Per-stage contact map visualizations
    for stage, vis_data in vis_results.items():
        stage_short = stage.replace('bprna-', '')
        visualize_cases(vis_data, out_dir, n_cases=20, stage_label=stage_short)
        visualize_score_heatmaps(vis_data, out_dir, stage_label=stage_short)

    # Loss analysis for F1=0 cases
    f1_zero_cases = [r for r in all_rows if r['f1'] == 0 and r['pred_pairs'] > 0]
    compute_loss_for_cases(model, f1_zero_cases, device, out_dir)

    print(f"\nAll outputs saved to: {out_dir}")
    print(f"Report: {out_dir / 'detailed_analysis.json'}")


if __name__ == '__main__':
    main()
