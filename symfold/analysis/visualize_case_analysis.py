# -*- coding: utf-8 -*-
"""Visualize case analysis for PriFold-SymFlow v6.

Generates comprehensive visualization plots for the case analysis report.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150


def load_cases(csv_path):
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in ['f1', 'precision', 'recall', 'density', 'pred_gt_ratio', 'length',
                      'gt_pairs', 'pred_pairs', 'mean_gt_dist', 'max_gt_dist',
                      'gt_stems', 'pred_stems', 'stem_ratio', 'tp', 'fp', 'fn', 'tn']:
                r[k] = float(r[k])
            r['length'] = int(r['length'])
            r['is_rfam'] = 'RFAM' in r['name']
            rows.append(r)
    return rows


def plot_f1_distribution(rows, ax, title='F1 Distribution'):
    """F1 score histogram with RFAM/non-RFAM breakdown."""
    rfam = [r['f1'] for r in rows if r['is_rfam']]
    non_rfam = [r['f1'] for r in rows if not r['is_rfam']]
    bins = np.linspace(0, 1, 21)
    ax.hist(rfam, bins=bins, alpha=0.7, label=f'RFAM (N={len(rfam)})', color='#e74c3c')
    ax.hist(non_rfam, bins=bins, alpha=0.7, label=f'non-RFAM (N={len(non_rfam)})', color='#2ecc71')
    ax.axvline(np.mean(rfam), color='#c0392b', linestyle='--', label=f'RFAM mean={np.mean(rfam):.3f}')
    ax.axvline(np.mean(non_rfam), color='#27ae60', linestyle='--', label=f'non-RFAM mean={np.mean(non_rfam):.3f}')
    ax.set_xlabel('F1 Score')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(fontsize=8)


def plot_f1_vs_length(rows, ax):
    """Scatter: F1 vs sequence length, colored by RFAM/non-RFAM."""
    rfam = [(r['length'], r['f1']) for r in rows if r['is_rfam']]
    non_rfam = [(r['length'], r['f1']) for r in rows if not r['is_rfam']]
    if rfam:
        ax.scatter([x[0] for x in rfam], [x[1] for x in rfam],
                   alpha=0.3, s=10, c='#e74c3c', label='RFAM')
    if non_rfam:
        ax.scatter([x[0] for x in non_rfam], [x[1] for x in non_rfam],
                   alpha=0.5, s=15, c='#2ecc71', label='non-RFAM')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 vs Length')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)


def plot_f1_vs_density(rows, ax):
    """Scatter: F1 vs density."""
    rfam = [(r['density'], r['f1']) for r in rows if r['is_rfam']]
    non_rfam = [(r['density'], r['f1']) for r in rows if not r['is_rfam']]
    if rfam:
        ax.scatter([x[0] for x in rfam], [x[1] for x in rfam],
                   alpha=0.3, s=10, c='#e74c3c', label='RFAM')
    if non_rfam:
        ax.scatter([x[0] for x in non_rfam], [x[1] for x in non_rfam],
                   alpha=0.5, s=15, c='#2ecc71', label='non-RFAM')
    ax.set_xlabel('Density (gt_pairs / length)')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 vs Density')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)


def plot_pred_gt_ratio(rows, ax):
    """pred/gt ratio vs F1."""
    ratios = [min(r['pred_gt_ratio'], 5.0) for r in rows]  # cap at 5 for vis
    f1s = [r['f1'] for r in rows]
    colors = ['#e74c3c' if r['is_rfam'] else '#2ecc71' for r in rows]
    ax.scatter(ratios, f1s, alpha=0.3, s=10, c=colors)
    ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='ideal ratio=1')
    ax.set_xlabel('pred/gt ratio (capped at 5)')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 vs pred/gt Ratio')
    ax.legend(fontsize=8, handles=[
        Patch(facecolor='#e74c3c', alpha=0.5, label='RFAM'),
        Patch(facecolor='#2ecc71', alpha=0.5, label='non-RFAM'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='ideal=1')
    ])
    ax.set_ylim(-0.05, 1.05)


def plot_length_bin_comparison(rows, ax):
    """Bar chart: F1 by length bin."""
    bins_order = ['<80', '80-159', '160-239', '240-319', '320-399', '400+']
    bin_map = {}
    for r in rows:
        lb = r.get('length_bin', '')
        if not lb:
            L = r['length']
            if L < 80: lb = '<80'
            elif L < 160: lb = '80-159'
            elif L < 240: lb = '160-239'
            elif L < 320: lb = '240-319'
            elif L < 400: lb = '320-399'
            else: lb = '400+'
        if lb not in bin_map:
            bin_map[lb] = {'rfam': [], 'non_rfam': []}
        if r['is_rfam']:
            bin_map[lb]['rfam'].append(r['f1'])
        else:
            bin_map[lb]['non_rfam'].append(r['f1'])

    x = np.arange(len(bins_order))
    width = 0.35
    rfam_means = [np.mean(bin_map.get(b, {}).get('rfam', [0])) for b in bins_order]
    non_rfam_means = [np.mean(bin_map.get(b, {}).get('non_rfam', [0])) if bin_map.get(b, {}).get('non_rfam', []) else 0 for b in bins_order]
    rfam_counts = [len(bin_map.get(b, {}).get('rfam', [])) for b in bins_order]
    non_rfam_counts = [len(bin_map.get(b, {}).get('non_rfam', [])) for b in bins_order]

    bars1 = ax.bar(x - width/2, rfam_means, width, label='RFAM', color='#e74c3c', alpha=0.7)
    bars2 = ax.bar(x + width/2, non_rfam_means, width, label='non-RFAM', color='#2ecc71', alpha=0.7)

    # Add count labels
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.02,
                f'n={rfam_counts[i]}', ha='center', va='bottom', fontsize=6)
        if non_rfam_counts[i] > 0:
            ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.02,
                    f'n={non_rfam_counts[i]}', ha='center', va='bottom', fontsize=6)

    ax.set_xlabel('Length Bin')
    ax.set_ylabel('Mean F1')
    ax.set_title('F1 by Length Bin (RFAM vs non-RFAM)')
    ax.set_xticks(x)
    ax.set_xticklabels(bins_order)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)


def plot_density_bin_comparison(rows, ax):
    """Bar chart: F1 by density bin."""
    bins_order = ['<0.10', '0.10-0.18', '0.18-0.25', '0.25-0.35', '>=0.35']
    bin_map = {}
    for r in rows:
        db = r.get('density_bin', '')
        if not db:
            d = r['density']
            if d < 0.10: db = '<0.10'
            elif d < 0.18: db = '0.10-0.18'
            elif d < 0.25: db = '0.18-0.25'
            elif d < 0.35: db = '0.25-0.35'
            else: db = '>=0.35'
        if db not in bin_map:
            bin_map[db] = []
        bin_map[db].append(r)

    x = np.arange(len(bins_order))
    means = [np.mean([r['f1'] for r in bin_map.get(b, [{'f1': 0}])]) for b in bins_order]
    pred_gt = [np.mean([r['pred_gt_ratio'] for r in bin_map.get(b, [{'pred_gt_ratio': 1}])]) for b in bins_order]
    counts = [len(bin_map.get(b, [])) for b in bins_order]

    bars = ax.bar(x, means, color='#3498db', alpha=0.7)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={counts[i]}\np/g={pred_gt[i]:.2f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Density Bin')
    ax.set_ylabel('Mean F1')
    ax.set_title('F1 by Density Bin')
    ax.set_xticks(x)
    ax.set_xticklabels(bins_order)
    ax.set_ylim(0, 1.1)


def plot_failure_modes(rows, ax):
    """Pie chart of failure modes."""
    bad = [r for r in rows if r['f1'] < 0.3]
    if not bad:
        ax.text(0.5, 0.5, 'No bad cases', ha='center', va='center')
        return

    mode1 = len([r for r in bad if r['f1'] == 0 and r['tp'] == 0 and r['gt_pairs'] > 0 and r['pred_pairs'] > 0])
    mode4 = len([r for r in bad if 0.8 < r['pred_gt_ratio'] < 1.2 and r['f1'] < 0.3 and r['f1'] > 0])
    mode3 = len([r for r in bad if r['pred_gt_ratio'] > 2 and r['f1'] < 0.5])
    mode_other = len(bad) - mode1 - mode4 - mode3

    labels = [
        f'Complete Miss\n(F1=0, tp=0)\nn={mode1}',
        f'Right Count\nWrong Position\nn={mode4}',
        f'Over-predict\n(pred/gt>2)\nn={mode3}',
        f'Other\nn={mode_other}'
    ]
    sizes = [mode1, mode4, mode3, mode_other]
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#95a5a6']
    explode = (0.05, 0.05, 0.05, 0)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=False, startangle=90, textprops={'fontsize': 8})
    ax.set_title(f'Failure Modes (F1<0.3, N={len(bad)})')


def plot_good_vs_bad_radar(rows, ax):
    """Compare characteristics of good vs bad cases."""
    bad = [r for r in rows if r['f1'] < 0.3]
    good = [r for r in rows if r['f1'] >= 0.7]
    if not bad or not good:
        return

    categories = ['Length\n(norm)', 'Density', 'pred/gt\n(norm)', 'Mean Dist\n(norm)', 'Stem Ratio\n(norm)']
    N = len(categories)

    # Normalize values for radar plot
    bad_vals = [
        np.mean([r['length'] for r in bad]) / 500,
        np.mean([r['density'] for r in bad]),
        min(np.mean([r['pred_gt_ratio'] for r in bad]) / 3, 1),
        np.mean([r['mean_gt_dist'] for r in bad]) / 100,
        min(np.mean([r['stem_ratio'] for r in bad]) / 3, 1),
    ]
    good_vals = [
        np.mean([r['length'] for r in good]) / 500,
        np.mean([r['density'] for r in good]),
        min(np.mean([r['pred_gt_ratio'] for r in good]) / 3, 1),
        np.mean([r['mean_gt_dist'] for r in good]) / 100,
        min(np.mean([r['stem_ratio'] for r in good]) / 3, 1),
    ]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    bad_vals += bad_vals[:1]
    good_vals += good_vals[:1]
    angles += angles[:1]

    ax.plot(angles, bad_vals, 'o-', linewidth=2, label=f'Bad (F1<0.3, N={len(bad)})', color='#e74c3c')
    ax.fill(angles, bad_vals, alpha=0.15, color='#e74c3c')
    ax.plot(angles, good_vals, 'o-', linewidth=2, label=f'Good (F1≥0.7, N={len(good)})', color='#2ecc71')
    ax.fill(angles, good_vals, alpha=0.15, color='#2ecc71')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_title('Good vs Bad Cases Profile', pad=20)
    ax.legend(loc='upper right', fontsize=7)


def main():
    out_dir = Path('/root/aigame/dannyyan/PriFold/symfold/outputs/v6_full/case_analysis')

    # Load test data
    test_rows = load_cases(out_dir / 'bprna_test_cases.csv')
    val_rows = load_cases(out_dir / 'bprna_val_cases.csv')
    all_rows = val_rows + test_rows

    # === Figure 1: Main overview (2x3 grid) ===
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('PriFold-SymFlow v6 Case Analysis (bpRNA val+test)', fontsize=14, fontweight='bold')

    plot_f1_distribution(test_rows, axes[0, 0], title='F1 Distribution (test)')
    plot_f1_vs_length(test_rows, axes[0, 1])
    plot_f1_vs_density(test_rows, axes[0, 2])
    plot_pred_gt_ratio(test_rows, axes[1, 0])
    plot_length_bin_comparison(test_rows, axes[1, 1])
    plot_density_bin_comparison(test_rows, axes[1, 2])

    plt.tight_layout()
    fig.savefig(out_dir / 'case_analysis_overview.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {out_dir / "case_analysis_overview.png"}')
    plt.close()

    # === Figure 2: Failure mode analysis ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Failure Mode Analysis (bpRNA test)', fontsize=14, fontweight='bold')

    plot_failure_modes(test_rows, axes[0])

    # Good vs Bad comparison as bar chart instead of radar
    bad = [r for r in test_rows if r['f1'] < 0.3]
    good = [r for r in test_rows if r['f1'] >= 0.7]
    metrics = ['Length', 'Density', 'pred/gt', 'gt_pairs', 'stem_ratio']
    bad_vals_raw = [
        np.mean([r['length'] for r in bad]),
        np.mean([r['density'] for r in bad]),
        np.mean([r['pred_gt_ratio'] for r in bad]),
        np.mean([r['gt_pairs'] for r in bad]),
        np.mean([r['stem_ratio'] for r in bad]),
    ]
    good_vals_raw = [
        np.mean([r['length'] for r in good]),
        np.mean([r['density'] for r in good]),
        np.mean([r['pred_gt_ratio'] for r in good]),
        np.mean([r['gt_pairs'] for r in good]),
        np.mean([r['stem_ratio'] for r in good]),
    ]

    x = np.arange(len(metrics))
    width = 0.35
    # Normalize for visualization
    maxvals = [max(b, g) for b, g in zip(bad_vals_raw, good_vals_raw)]
    bad_norm = [b / m if m > 0 else 0 for b, m in zip(bad_vals_raw, maxvals)]
    good_norm = [g / m if m > 0 else 0 for g, m in zip(good_vals_raw, maxvals)]

    bars1 = axes[1].bar(x - width/2, bad_norm, width, label=f'Bad (F1<0.3, N={len(bad)})', color='#e74c3c', alpha=0.7)
    bars2 = axes[1].bar(x + width/2, good_norm, width, label=f'Good (F1≥0.7, N={len(good)})', color='#2ecc71', alpha=0.7)

    # Add actual values
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        axes[1].text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.02,
                     f'{bad_vals_raw[i]:.2f}', ha='center', va='bottom', fontsize=8)
        axes[1].text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.02,
                     f'{good_vals_raw[i]:.2f}', ha='center', va='bottom', fontsize=8)

    axes[1].set_xlabel('Metric')
    axes[1].set_ylabel('Normalized Value')
    axes[1].set_title('Good vs Bad Cases Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 1.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'failure_mode_analysis.png', dpi=150, bbox_inches='tight')
    print(f'Saved: {out_dir / "failure_mode_analysis.png"}')
    plt.close()

    # === Figure 3: v5 vs v6 comparison ===
    v5_path = Path('/root/aigame/dannyyan/PriFold/symfold/outputs/v5_bprna/case_analysis/detailed_analysis.json')
    if v5_path.exists():
        with open(v5_path) as f:
            v5_data = json.load(f)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('v6 vs v5 Comparison (bpRNA test)', fontsize=14, fontweight='bold')

        # By length bin
        bins_order = ['<80', '80-159', '160-239', '240-319', '320-399', '400+']
        v5_by_len = v5_data['by_length_bin']
        v6_by_len = {}
        for lb in bins_order:
            subset = [r for r in test_rows if (
                (lb == '<80' and r['length'] < 80) or
                (lb == '80-159' and 80 <= r['length'] < 160) or
                (lb == '160-239' and 160 <= r['length'] < 240) or
                (lb == '240-319' and 240 <= r['length'] < 320) or
                (lb == '320-399' and 320 <= r['length'] < 400) or
                (lb == '400+' and r['length'] >= 400)
            )]
            v6_by_len[lb] = np.mean([r['f1'] for r in subset]) if subset else 0

        x = np.arange(len(bins_order))
        width = 0.35
        v5_vals = [v5_by_len.get(b, {}).get('f1', 0) for b in bins_order]
        v6_vals = [v6_by_len.get(b, 0) for b in bins_order]

        axes[0].bar(x - width/2, v5_vals, width, label='v5', color='#3498db', alpha=0.7)
        axes[0].bar(x + width/2, v6_vals, width, label='v6', color='#e67e22', alpha=0.7)
        axes[0].set_xlabel('Length Bin')
        axes[0].set_ylabel('Mean F1')
        axes[0].set_title('F1 by Length Bin')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(bins_order, rotation=30)
        axes[0].legend()
        axes[0].set_ylim(0, 1.0)

        # By density bin
        density_bins = ['<0.10', '0.10-0.18', '0.18-0.25', '0.25-0.35', '>=0.35']
        v5_by_den = v5_data['by_density_bin']
        v6_by_den = {}
        for db in density_bins:
            subset = [r for r in test_rows if (
                (db == '<0.10' and r['density'] < 0.10) or
                (db == '0.10-0.18' and 0.10 <= r['density'] < 0.18) or
                (db == '0.18-0.25' and 0.18 <= r['density'] < 0.25) or
                (db == '0.25-0.35' and 0.25 <= r['density'] < 0.35) or
                (db == '>=0.35' and r['density'] >= 0.35)
            )]
            v6_by_den[db] = np.mean([r['f1'] for r in subset]) if subset else 0

        x = np.arange(len(density_bins))
        v5_dvals = [v5_by_den.get(b, {}).get('f1', 0) for b in density_bins]
        v6_dvals = [v6_by_den.get(b, 0) for b in density_bins]

        axes[1].bar(x - width/2, v5_dvals, width, label='v5', color='#3498db', alpha=0.7)
        axes[1].bar(x + width/2, v6_dvals, width, label='v6', color='#e67e22', alpha=0.7)
        axes[1].set_xlabel('Density Bin')
        axes[1].set_ylabel('Mean F1')
        axes[1].set_title('F1 by Density Bin')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(density_bins, rotation=30)
        axes[1].legend()
        axes[1].set_ylim(0, 1.0)

        # pred/gt comparison
        v5_pred_gt_by_den = [v5_by_den.get(b, {}).get('pred_gt_ratio', 1) for b in density_bins]
        v6_pred_gt_by_den = []
        for db in density_bins:
            subset = [r for r in test_rows if (
                (db == '<0.10' and r['density'] < 0.10) or
                (db == '0.10-0.18' and 0.10 <= r['density'] < 0.18) or
                (db == '0.18-0.25' and 0.18 <= r['density'] < 0.25) or
                (db == '0.25-0.35' and 0.25 <= r['density'] < 0.35) or
                (db == '>=0.35' and r['density'] >= 0.35)
            )]
            v6_pred_gt_by_den.append(np.mean([r['pred_gt_ratio'] for r in subset]) if subset else 1)

        x = np.arange(len(density_bins))
        axes[2].bar(x - width/2, v5_pred_gt_by_den, width, label='v5', color='#3498db', alpha=0.7)
        axes[2].bar(x + width/2, v6_pred_gt_by_den, width, label='v6', color='#e67e22', alpha=0.7)
        axes[2].axhline(1.0, color='black', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Density Bin')
        axes[2].set_ylabel('pred/gt Ratio')
        axes[2].set_title('pred/gt Ratio by Density (lower=less over-predict)')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(density_bins, rotation=30)
        axes[2].legend()

        plt.tight_layout()
        fig.savefig(out_dir / 'v5_vs_v6_comparison.png', dpi=150, bbox_inches='tight')
        print(f'Saved: {out_dir / "v5_vs_v6_comparison.png"}')
        plt.close()

    print('Done!')


if __name__ == '__main__':
    main()
