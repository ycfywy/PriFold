# -*- coding: utf-8 -*-
"""补充可视化：生成 F1 vs 长度/密度/复杂度 的趋势曲线图。

从 all_samples_metrics.csv 读取数据，生成：
1. F1 vs Length 趋势曲线（train/val/test 三线）
2. F1 vs Density 趋势曲线
3. F1 vs #Stems 趋势曲线
4. F1 vs Mean Pairing Distance 趋势曲线
5. F1 vs #Pseudoknots 趋势曲线
6. Precision/Recall vs Length 趋势曲线
7. pred/gt ratio vs Length 趋势曲线
8. 综合 dashboard

Usage:
  python symfold/analysis/supplementary_plots.py \
    --csv symfold/outputs/v7_full/comprehensive_analysis/all_samples_metrics.csv \
    --out_dir symfold/outputs/v7_full/comprehensive_analysis
"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.labelsize'] = 10


def binned_mean(df, x_col, y_col, bins):
    """Compute mean of y_col in bins of x_col."""
    df = df.copy()
    df['_bin'] = pd.cut(df[x_col], bins=bins)
    grouped = df.groupby('_bin', observed=True)[y_col].agg(['mean', 'std', 'count'])
    # Use bin midpoints as x
    midpoints = [(b.left + b.right) / 2 for b in grouped.index]
    return midpoints, grouped['mean'].values, grouped['std'].values, grouped['count'].values


def plot_trend_by_stage(df, x_col, y_col, xlabel, ylabel, title, out_path,
                         n_bins=15, show_scatter=True, ylim=None):
    """Plot y vs x trend curve with train/val/test as three lines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'train': '#3498db', 'val': '#e67e22', 'test': '#e74c3c'}
    
    x_min = df[x_col].min()
    x_max = df[x_col].max()
    bins = np.linspace(x_min, x_max, n_bins + 1)
    
    for stage, color in colors.items():
        sub = df[df['stage'] == stage]
        if sub.empty:
            continue
        
        if show_scatter:
            ax.scatter(sub[x_col], sub[y_col], alpha=0.05, s=5, color=color)
        
        midpoints, means, stds, counts = binned_mean(sub, x_col, y_col, bins)
        # Only show bins with enough samples
        valid = [i for i, c in enumerate(counts) if c >= 5]
        mid_valid = [midpoints[i] for i in valid]
        mean_valid = [means[i] for i in valid]
        std_valid = [stds[i] for i in valid]
        
        ax.plot(mid_valid, mean_valid, '-o', color=color, linewidth=2, markersize=5,
                label=f'{stage} (N={len(sub)}, mean={sub[y_col].mean():.3f})')
        ax.fill_between(mid_valid,
                        [m - s/2 for m, s in zip(mean_valid, std_valid)],
                        [m + s/2 for m, s in zip(mean_valid, std_valid)],
                        alpha=0.15, color=color)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_multi_metric_vs_length(df, out_path):
    """Plot F1, Precision, Recall, pred/gt vs Length in subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Metrics vs Sequence Length (binned mean ± std/2)',
                 fontsize=13, fontweight='bold')
    
    colors = {'train': '#3498db', 'val': '#e67e22', 'test': '#e74c3c'}
    metrics = [
        ('f1', 'F1 Score', (0, 1)),
        ('precision', 'Precision', (0, 1)),
        ('recall', 'Recall', (0, 1)),
        ('pred_gt_ratio', 'Pred/GT Ratio', (0, 3)),
    ]
    
    bins = np.linspace(0, 490, 20)
    
    for ax, (metric, label, ylim) in zip(axes.flat, metrics):
        for stage, color in colors.items():
            sub = df[df['stage'] == stage]
            midpoints, means, stds, counts = binned_mean(sub, 'length', metric, bins)
            valid = [i for i, c in enumerate(counts) if c >= 5]
            mid_v = [midpoints[i] for i in valid]
            mean_v = [means[i] for i in valid]
            std_v = [stds[i] for i in valid]
            
            ax.plot(mid_v, mean_v, '-o', color=color, linewidth=2, markersize=4,
                    label=f'{stage}')
            ax.fill_between(mid_v,
                            [m - s/2 for m, s in zip(mean_v, std_v)],
                            [m + s/2 for m, s in zip(mean_v, std_v)],
                            alpha=0.12, color=color)
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Length')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ylim)
        if metric == 'pred_gt_ratio':
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_multi_metric_vs_density(df, out_path):
    """Plot F1, Precision, Recall, pred/gt vs Density in subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Metrics vs Pairing Density (binned mean ± std/2)',
                 fontsize=13, fontweight='bold')
    
    colors = {'train': '#3498db', 'val': '#e67e22', 'test': '#e74c3c'}
    metrics = [
        ('f1', 'F1 Score', (0, 1)),
        ('precision', 'Precision', (0, 1)),
        ('recall', 'Recall', (0, 1)),
        ('pred_gt_ratio', 'Pred/GT Ratio', (0, 3)),
    ]
    
    bins = np.linspace(0, 0.5, 18)
    
    for ax, (metric, label, ylim) in zip(axes.flat, metrics):
        for stage, color in colors.items():
            sub = df[df['stage'] == stage]
            midpoints, means, stds, counts = binned_mean(sub, 'density', metric, bins)
            valid = [i for i, c in enumerate(counts) if c >= 5]
            mid_v = [midpoints[i] for i in valid]
            mean_v = [means[i] for i in valid]
            std_v = [stds[i] for i in valid]
            
            ax.plot(mid_v, mean_v, '-o', color=color, linewidth=2, markersize=4,
                    label=f'{stage}')
            ax.fill_between(mid_v,
                            [m - s/2 for m, s in zip(mean_v, std_v)],
                            [m + s/2 for m, s in zip(mean_v, std_v)],
                            alpha=0.12, color=color)
        
        ax.set_xlabel('Pairing Density (gt_pairs / length)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(ylim)
        if metric == 'pred_gt_ratio':
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_f1_vs_complexity_trends(df, out_path):
    """F1 vs various complexity metrics as trend lines."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('F1 vs Structure Complexity (binned mean, train/val/test)',
                 fontsize=13, fontweight='bold')
    
    colors = {'train': '#3498db', 'val': '#e67e22', 'test': '#e74c3c'}
    metrics = [
        ('gt_n_stems', '# Stems (GT)', 15),
        ('gt_max_stem_len', 'Max Stem Length', 12),
        ('gt_branching', 'Branching Factor', 12),
        ('gt_mean_dist', 'Mean Pairing Distance', 15),
        ('gt_pseudoknots', '# Pseudoknot Crossings', 12),
        ('gt_max_dist', 'Max Pairing Distance', 15),
    ]
    
    for ax, (col, label, n_bins) in zip(axes.flat, metrics):
        for stage, color in colors.items():
            sub = df[df['stage'] == stage]
            if sub[col].max() == sub[col].min():
                continue
            bins = np.linspace(sub[col].min(), sub[col].max(), n_bins + 1)
            midpoints, means, stds, counts = binned_mean(sub, col, 'f1', bins)
            valid = [i for i, c in enumerate(counts) if c >= 5]
            mid_v = [midpoints[i] for i in valid]
            mean_v = [means[i] for i in valid]
            
            ax.plot(mid_v, mean_v, '-o', color=color, linewidth=2, markersize=4,
                    label=f'{stage}')
        
        ax.set_xlabel(label)
        ax.set_ylabel('Mean F1')
        ax.set_title(f'F1 vs {label}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_density_length_interaction(df, out_path):
    """Show how density and length jointly affect F1."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('F1 Score: Length × Density Interaction', fontsize=13, fontweight='bold')
    
    colors_density = ['#e74c3c', '#f39c12', '#27ae60', '#2980b9']
    density_bins = [(0, 0.15, '<0.15'), (0.15, 0.22, '0.15-0.22'),
                    (0.22, 0.30, '0.22-0.30'), (0.30, 1.0, '≥0.30')]
    
    length_bins_arr = np.linspace(20, 490, 16)
    
    for ax, stage in zip(axes, ['train', 'val', 'test']):
        sub = df[df['stage'] == stage]
        
        for (d_lo, d_hi, d_label), color in zip(density_bins, colors_density):
            d_sub = sub[(sub['density'] >= d_lo) & (sub['density'] < d_hi)]
            if len(d_sub) < 10:
                continue
            midpoints, means, stds, counts = binned_mean(d_sub, 'length', 'f1', length_bins_arr)
            valid = [i for i, c in enumerate(counts) if c >= 3]
            mid_v = [midpoints[i] for i in valid]
            mean_v = [means[i] for i in valid]
            
            ax.plot(mid_v, mean_v, '-o', color=color, linewidth=2, markersize=4,
                    label=f'density {d_label} (N={len(d_sub)})')
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Mean F1')
        ax.set_title(f'{stage}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_generalization_gap(df, out_path):
    """Show train-test gap by length and density."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Generalization Gap: Train F1 - Test F1', fontsize=13, fontweight='bold')
    
    # By length
    ax = axes[0]
    bins = np.linspace(20, 490, 16)
    
    train_sub = df[df['stage'] == 'train']
    test_sub = df[df['stage'] == 'test']
    
    _, train_means, _, train_counts = binned_mean(train_sub, 'length', 'f1', bins)
    midpoints, test_means, _, test_counts = binned_mean(test_sub, 'length', 'f1', bins)
    
    valid = [i for i, (tc, trc) in enumerate(zip(test_counts, train_counts)) if tc >= 5 and trc >= 5]
    mid_v = [midpoints[i] for i in valid]
    gap_v = [train_means[i] - test_means[i] for i in valid]
    
    ax.bar(mid_v, gap_v, width=(bins[1]-bins[0])*0.7, color='#e74c3c', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('F1 Gap (Train - Test)')
    ax.set_title('Generalization Gap by Length')
    ax.grid(True, alpha=0.3, axis='y')
    
    # By density
    ax = axes[1]
    bins = np.linspace(0, 0.45, 14)
    
    _, train_means, _, train_counts = binned_mean(train_sub, 'density', 'f1', bins)
    midpoints, test_means, _, test_counts = binned_mean(test_sub, 'density', 'f1', bins)
    
    valid = [i for i, (tc, trc) in enumerate(zip(test_counts, train_counts)) if tc >= 5 and trc >= 5]
    mid_v = [midpoints[i] for i in valid]
    gap_v = [train_means[i] - test_means[i] for i in valid]
    
    ax.bar(mid_v, gap_v, width=(bins[1]-bins[0])*0.7, color='#9b59b6', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Pairing Density')
    ax.set_ylabel('F1 Gap (Train - Test)')
    ax.set_title('Generalization Gap by Density')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_bad_rate_by_length_density(df, out_path):
    """Show proportion of bad cases (F1<0.3) by length and density."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Bad Case Rate (F1<0.3) by Length and Density',
                 fontsize=13, fontweight='bold')
    
    colors = {'train': '#3498db', 'val': '#e67e22', 'test': '#e74c3c'}
    
    # By length
    ax = axes[0]
    bins = np.linspace(20, 490, 14)
    for stage, color in colors.items():
        sub = df[df['stage'] == stage]
        sub_copy = sub.copy()
        sub_copy['is_bad'] = (sub_copy['f1'] < 0.3).astype(float)
        midpoints, means, _, counts = binned_mean(sub_copy, 'length', 'is_bad', bins)
        valid = [i for i, c in enumerate(counts) if c >= 5]
        mid_v = [midpoints[i] for i in valid]
        mean_v = [means[i] * 100 for i in valid]  # percentage
        ax.plot(mid_v, mean_v, '-o', color=color, linewidth=2, markersize=5,
                label=f'{stage}')
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Bad Case Rate (%)')
    ax.set_title('Bad Case Rate vs Length')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # By density
    ax = axes[1]
    bins = np.linspace(0, 0.45, 14)
    for stage, color in colors.items():
        sub = df[df['stage'] == stage]
        sub_copy = sub.copy()
        sub_copy['is_bad'] = (sub_copy['f1'] < 0.3).astype(float)
        midpoints, means, _, counts = binned_mean(sub_copy, 'density', 'is_bad', bins)
        valid = [i for i, c in enumerate(counts) if c >= 5]
        mid_v = [midpoints[i] for i in valid]
        mean_v = [means[i] * 100 for i in valid]
        ax.plot(mid_v, mean_v, '-o', color=color, linewidth=2, markersize=5,
                label=f'{stage}')
    
    ax.set_xlabel('Pairing Density')
    ax.set_ylabel('Bad Case Rate (%)')
    ax.set_title('Bad Case Rate vs Density')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str,
                        default='symfold/outputs/v7_full/comprehensive_analysis/all_samples_metrics.csv')
    parser.add_argument('--out_dir', type=str,
                        default='symfold/outputs/v7_full/comprehensive_analysis')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    print('Loading CSV...')
    df = pd.read_csv(args.csv)
    print(f'  Loaded {len(df)} samples: {df["stage"].value_counts().to_dict()}')
    
    print('\nGenerating supplementary trend plots...\n')
    
    # 1. F1 vs Length trend
    print('[1/8] F1 vs Length trend...')
    plot_trend_by_stage(df, 'length', 'f1', 'Sequence Length', 'F1 Score',
                        'F1 Score vs Sequence Length (binned mean ± std/2)',
                        out_dir / 'f1_vs_length_trend.png', n_bins=18, ylim=(0, 1))
    
    # 2. F1 vs Density trend
    print('[2/8] F1 vs Density trend...')
    plot_trend_by_stage(df, 'density', 'f1', 'Pairing Density', 'F1 Score',
                        'F1 Score vs Pairing Density (binned mean ± std/2)',
                        out_dir / 'f1_vs_density_trend.png', n_bins=15, ylim=(0, 1))
    
    # 3. Multi-metric vs Length
    print('[3/8] Multi-metric vs Length...')
    plot_multi_metric_vs_length(df, out_dir / 'metrics_vs_length.png')
    
    # 4. Multi-metric vs Density
    print('[4/8] Multi-metric vs Density...')
    plot_multi_metric_vs_density(df, out_dir / 'metrics_vs_density.png')
    
    # 5. F1 vs Complexity trends
    print('[5/8] F1 vs Complexity trends...')
    plot_f1_vs_complexity_trends(df, out_dir / 'f1_vs_complexity_trends.png')
    
    # 6. Density × Length interaction
    print('[6/8] Density × Length interaction...')
    plot_density_length_interaction(df, out_dir / 'density_length_interaction.png')
    
    # 7. Generalization gap
    print('[7/8] Generalization gap...')
    plot_generalization_gap(df, out_dir / 'generalization_gap.png')
    
    # 8. Bad case rate
    print('[8/8] Bad case rate by length/density...')
    plot_bad_rate_by_length_density(df, out_dir / 'bad_case_rate.png')
    
    print('\nAll supplementary plots generated!')


if __name__ == '__main__':
    main()
