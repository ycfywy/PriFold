# -*- coding: utf-8 -*-
"""PriFold v7 DensityNet: 深度可视化分析。

系统性分析 v7 在 bprna train/val/test 上的表现：
1. 每个数据集的 worst/best cases 可视化（GT / Pred / Score / Diff）
2. 各 RNA 家族的表现对比与可视化
3. 失败模式可视化：预测错位 / 没有预测 / 过预测
4. Probability heatmap 分析

Usage:
  python symfold/deep_analysis_v7.py \
    --ckpt symfold/outputs/v7_full/model/best.pt \
    --config symfold/config/v7_full.json \
    --out_dir symfold/outputs/v7_full/deep_analysis
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
import matplotlib.gridspec as gridspec

plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.labelsize'] = 8

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_loader
from symfold.train.train_v7 import build_model, load_config, move_to_device


# ============================================================
# Helpers
# ============================================================

def per_sample_metrics(pred, target, length):
    """Compute per-sample metrics."""
    p = pred.detach().cpu().float().squeeze()[:length, :length] > 0.5
    y = target.detach().cpu().float().squeeze()[:length, :length] > 0.5
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    idx = torch.arange(length)
    mask &= (idx.view(length, 1) - idx.view(1, length)).abs() >= 3
    p_m = p[mask]
    y_m = y[mask]
    tp = int((p_m & y_m).sum())
    fp = int((p_m & ~y_m).sum())
    fn = int((~p_m & y_m).sum())
    gt_pairs = tp + fn
    pred_pairs = tp + fp
    # Handle edge case: gt=0 and pred=0 → perfect
    if gt_pairs == 0 and pred_pairs == 0:
        prec, rec, f1 = 1.0, 1.0, 1.0
    elif gt_pairs == 0 and pred_pairs > 0:
        prec, rec, f1 = 0.0, 1.0, 0.0
    else:
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        'precision': prec, 'recall': rec, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'gt_pairs': gt_pairs, 'pred_pairs': pred_pairs,
        'density': gt_pairs / max(length, 1),
        'pred_gt_ratio': pred_pairs / max(gt_pairs, 1),
    }


def get_category(name: str) -> str:
    """Extract RNA category from bpRNA name."""
    parts = name.split('_')
    return parts[1] if len(parts) >= 3 else 'unknown'


# ============================================================
# Single case visualization (4-panel: GT, Pred, Score, Diff)
# ============================================================

def plot_single_case(case, axes_row, show_ylabel=True):
    """Plot a single case across 4 axes: GT, Score, Pred, Diff."""
    L = case['length']
    gt = case['gt'][:L, :L]
    pred_map = case['pred'][:L, :L]
    score = case['score'][:L, :L]

    # GT
    ax = axes_row[0]
    ax.imshow(gt, cmap='Blues', vmin=0, vmax=1, aspect='equal', interpolation='nearest')
    ax.set_title('Ground Truth', fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    if show_ylabel:
        short = case['name'].replace('bpRNA_', '')
        ax.set_ylabel(f'{short}\nL={L} d={case["density"]:.2f}', fontsize=6,
                      rotation=0, labelpad=55, va='center')

    # Score (probability heatmap)
    ax = axes_row[1]
    im = ax.imshow(score, cmap='hot', vmin=0, vmax=1, aspect='equal', interpolation='nearest')
    ax.set_title('Prob Score', fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])

    # Pred
    ax = axes_row[2]
    ax.imshow(pred_map, cmap='Oranges', vmin=0, vmax=1, aspect='equal', interpolation='nearest')
    ax.set_title('Prediction', fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])

    # Diff (TP=green, FP=red, FN=blue)
    ax = axes_row[3]
    diff = np.zeros((L, L, 3))
    p_bin = pred_map > 0.5
    g_bin = gt > 0.5
    diff[(p_bin & g_bin)] = [0.2, 0.8, 0.2]   # TP green
    diff[(p_bin & ~g_bin)] = [0.9, 0.2, 0.2]  # FP red
    diff[(~p_bin & g_bin)] = [0.2, 0.4, 0.9]  # FN blue
    ax.imshow(diff, aspect='equal', interpolation='nearest')
    f1 = case['f1']
    pg = case['pred_gt_ratio']
    ax.set_title(f'F1={f1:.3f} P/R={case["precision"]:.2f}/{case["recall"]:.2f}\np/g={pg:.2f}', fontsize=6)
    ax.set_xticks([]); ax.set_yticks([])


def visualize_cases_grid(cases, title, out_path, n_cases=10):
    """Generate a grid of case visualizations (4 columns × n_cases rows)."""
    n = min(n_cases, len(cases))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 4, figsize=(14, n * 2.5))
    fig.suptitle(title, fontsize=12, fontweight='bold', y=0.995)
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        plot_single_case(cases[i], axes[i], show_ylabel=True)
    # Legend
    legend_elements = [
        Patch(facecolor=[0.2, 0.8, 0.2], label='TP (correct)'),
        Patch(facecolor=[0.9, 0.2, 0.2], label='FP (wrong predict)'),
        Patch(facecolor=[0.2, 0.4, 0.9], label='FN (missed)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 0.001))
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    print(f'  Saved: {out_path}')
    plt.close()


# ============================================================
# Per-family analysis
# ============================================================

def visualize_family_analysis(all_results, out_dir):
    """Comprehensive per-family breakdown with examples."""
    by_cat = defaultdict(list)
    for r in all_results:
        by_cat[r['category']].append(r)

    cat_order = sorted(by_cat.keys(), key=lambda c: -len(by_cat[c]))

    # --- Figure 1: Family performance overview ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('v7 DensityNet: Per-Family Performance', fontsize=13, fontweight='bold')

    # 1.1 F1 boxplot per family
    ax = axes[0, 0]
    cat_f1s = [np.array([r['f1'] for r in by_cat[c]]) for c in cat_order]
    bp = ax.boxplot(cat_f1s, labels=cat_order, patch_artist=True, showmeans=True,
                    medianprops=dict(color='black'))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    for patch, color in zip(bp['boxes'], colors[:len(cat_order)]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Distribution by Family')
    for i, c in enumerate(cat_order):
        ax.text(i + 1, -0.08, f'N={len(by_cat[c])}', ha='center', fontsize=6)

    # 1.2 Mean F1 + pred/gt bar
    ax = axes[0, 1]
    x = np.arange(len(cat_order))
    f1_means = [np.mean([r['f1'] for r in by_cat[c]]) for c in cat_order]
    pg_means = [np.mean([r['pred_gt_ratio'] for r in by_cat[c]]) for c in cat_order]
    width = 0.35
    ax.bar(x - width/2, f1_means, width, label='Mean F1', color='#3498db', alpha=0.7)
    ax2 = ax.twinx()
    ax2.bar(x + width/2, pg_means, width, label='pred/gt', color='#e67e22', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(cat_order)
    ax.set_ylabel('F1', color='#3498db'); ax2.set_ylabel('pred/gt', color='#e67e22')
    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.3)
    ax.set_title('Mean F1 & pred/gt by Family')
    ax.set_ylim(0, 1.0); ax2.set_ylim(0, 3.0)

    # 1.3 F1=0 and F1≥0.9 counts per family
    ax = axes[0, 2]
    f1_zero = [sum(1 for r in by_cat[c] if r['f1'] == 0) for c in cat_order]
    f1_high = [sum(1 for r in by_cat[c] if r['f1'] >= 0.9) for c in cat_order]
    ax.bar(x - width/2, f1_zero, width, label='F1=0', color='#e74c3c', alpha=0.7)
    ax.bar(x + width/2, f1_high, width, label='F1≥0.9', color='#2ecc71', alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(cat_order)
    ax.set_ylabel('Count')
    ax.set_title('F1=0 vs F1≥0.9 by Family')
    ax.legend(fontsize=8)

    # 1.4 RFAM F1 by length bin
    ax = axes[1, 0]
    rfam = by_cat.get('RFAM', [])
    len_bins = ['<80', '80-159', '160-239', '240-319', '320+']
    rfam_by_len = defaultdict(list)
    for r in rfam:
        L = r['length']
        if L < 80: b = '<80'
        elif L < 160: b = '80-159'
        elif L < 240: b = '160-239'
        elif L < 320: b = '240-319'
        else: b = '320+'
        rfam_by_len[b].append(r)
    f1_vals = [np.mean([r['f1'] for r in rfam_by_len[b]]) if rfam_by_len[b] else 0 for b in len_bins]
    counts = [len(rfam_by_len[b]) for b in len_bins]
    bars = ax.bar(range(len(len_bins)), f1_vals, color='#e74c3c', alpha=0.7)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'N={counts[i]}', ha='center', fontsize=7)
    ax.set_xticks(range(len(len_bins))); ax.set_xticklabels(len_bins)
    ax.set_xlabel('Length'); ax.set_ylabel('Mean F1')
    ax.set_title('RFAM: F1 by Length Bin')
    ax.set_ylim(0, 1.0)

    # 1.5 F1 vs density scatter (RFAM vs others)
    ax = axes[1, 1]
    for r in all_results:
        color = '#e74c3c' if r['category'] == 'RFAM' else '#2ecc71'
        ax.scatter(r['density'], r['f1'], c=color, s=5, alpha=0.2)
    ax.set_xlabel('Density'); ax.set_ylabel('F1')
    ax.set_title('F1 vs Density')
    ax.legend(handles=[Patch(facecolor='#e74c3c', alpha=0.5, label='RFAM'),
                       Patch(facecolor='#2ecc71', alpha=0.5, label='non-RFAM')], fontsize=7)

    # 1.6 Precision vs Recall by family
    ax = axes[1, 2]
    for i, c in enumerate(cat_order):
        p_mean = np.mean([r['precision'] for r in by_cat[c]])
        r_mean = np.mean([r['recall'] for r in by_cat[c]])
        ax.scatter(r_mean, p_mean, s=100, c=colors[i % len(colors)],
                   label=f'{c} (N={len(by_cat[c])})', zorder=3)
        ax.annotate(c, (r_mean, p_mean), textcoords="offset points",
                    xytext=(5, 5), fontsize=7)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Mean Recall'); ax.set_ylabel('Mean Precision')
    ax.set_title('Precision vs Recall by Family')
    ax.set_xlim(0.3, 1.0); ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=6, loc='lower left')

    plt.tight_layout()
    fig.savefig(out_dir / 'family_performance_overview.png', dpi=150, bbox_inches='tight')
    print(f'  Saved: family_performance_overview.png')
    plt.close()

    # --- Per-family best/worst case visualizations ---
    for cat in cat_order:
        cat_cases = by_cat[cat]
        if len(cat_cases) < 5:
            continue
        sorted_cases = sorted(cat_cases, key=lambda x: x['f1'])
        worst = [c for c in sorted_cases if c['length'] >= 30][:5]
        best = sorted(cat_cases, key=lambda x: -x['f1'])[:5]

        if worst:
            visualize_cases_grid(worst, f'{cat}: Worst 5 Cases',
                                 out_dir / f'family_{cat}_worst5.png', n_cases=5)
        if best:
            visualize_cases_grid(best, f'{cat}: Best 5 Cases',
                                 out_dir / f'family_{cat}_best5.png', n_cases=5)


# ============================================================
# Failure mode deep-dive
# ============================================================

def visualize_failure_modes(all_results, out_dir):
    """Deep-dive into specific failure modes with example cases."""

    # Mode A: F1=0, predicted but all wrong
    f1_zero = [r for r in all_results if r['f1'] == 0 and r['pred_pairs'] > 0 and r['length'] >= 40]
    f1_zero = sorted(f1_zero, key=lambda x: -x['pred_pairs'])[:10]
    if f1_zero:
        visualize_cases_grid(f1_zero,
            'Failure Mode: F1=0 (predicted but ALL wrong positions)',
            out_dir / 'failure_f1_zero.png', n_cases=10)

    # Mode B: Good density but wrong position (pred/gt ~ 1 but F1 < 0.3)
    wrong_pos = [r for r in all_results
                 if 0.7 < r['pred_gt_ratio'] < 1.3 and r['f1'] < 0.3 and r['length'] >= 50]
    wrong_pos = sorted(wrong_pos, key=lambda x: x['f1'])[:10]
    if wrong_pos:
        visualize_cases_grid(wrong_pos,
            'Failure Mode: Right Count, Wrong Position (0.7<p/g<1.3, F1<0.3)',
            out_dir / 'failure_wrong_position.png', n_cases=10)

    # Mode C: Severe over-prediction (pred/gt > 2)
    overpredict = [r for r in all_results if r['pred_gt_ratio'] > 2 and r['length'] >= 40]
    overpredict = sorted(overpredict, key=lambda x: -x['pred_gt_ratio'])[:10]
    if overpredict:
        visualize_cases_grid(overpredict,
            'Failure Mode: Over-prediction (pred/gt > 2)',
            out_dir / 'failure_overpredict.png', n_cases=10)

    # Mode D: Under-prediction (pred/gt < 0.5, recall is low)
    underpredict = [r for r in all_results
                    if r['pred_gt_ratio'] < 0.5 and r['gt_pairs'] > 5 and r['length'] >= 40]
    underpredict = sorted(underpredict, key=lambda x: x['pred_gt_ratio'])[:10]
    if underpredict:
        visualize_cases_grid(underpredict,
            'Failure Mode: Under-prediction (pred/gt < 0.5)',
            out_dir / 'failure_underpredict.png', n_cases=10)

    # Mode E: Medium F1 (0.3-0.5) — what's partially right?
    medium = [r for r in all_results if 0.3 <= r['f1'] <= 0.5 and r['length'] >= 60]
    medium = sorted(medium, key=lambda x: x['f1'])[:10]
    if medium:
        visualize_cases_grid(medium,
            'Partial Success: F1 ∈ [0.3, 0.5] (partially correct)',
            out_dir / 'partial_success_medium.png', n_cases=10)


# ============================================================
# Per-stage analysis
# ============================================================

def visualize_per_stage(stage_results, out_dir):
    """Generate per-stage (train/val/test) visualizations."""
    for stage, results in stage_results.items():
        stage_short = stage.replace('bprna-', '')
        stage_dir = out_dir / stage_short
        stage_dir.mkdir(parents=True, exist_ok=True)

        sorted_by_f1 = sorted(results, key=lambda x: x['f1'])

        # Worst 10
        worst = [r for r in sorted_by_f1 if r['length'] >= 40][:10]
        visualize_cases_grid(worst, f'{stage}: Worst 10 Cases (lowest F1)',
                             stage_dir / 'worst_10.png', n_cases=10)

        # Best 10
        best = sorted(results, key=lambda x: -x['f1'])[:10]
        visualize_cases_grid(best, f'{stage}: Best 10 Cases (highest F1)',
                             stage_dir / 'best_10.png', n_cases=10)

        # Random 10 from middle (F1 ∈ [0.5, 0.7])
        mid = [r for r in results if 0.5 <= r['f1'] <= 0.7]
        if len(mid) > 10:
            rng = np.random.default_rng(42)
            mid = [mid[i] for i in rng.choice(len(mid), 10, replace=False)]
        if mid:
            visualize_cases_grid(mid, f'{stage}: Middle 10 Cases (F1 ∈ [0.5, 0.7])',
                                 stage_dir / 'middle_10.png', n_cases=10)

        # Stage summary stats
        n = len(results)
        summary = {
            'stage': stage,
            'n': n,
            'f1': float(np.mean([r['f1'] for r in results])),
            'precision': float(np.mean([r['precision'] for r in results])),
            'recall': float(np.mean([r['recall'] for r in results])),
            'pred_gt_ratio': float(np.mean([r['pred_gt_ratio'] for r in results])),
            'f1_zero': sum(1 for r in results if r['f1'] == 0),
            'f1_below_03': sum(1 for r in results if r['f1'] < 0.3),
            'f1_above_09': sum(1 for r in results if r['f1'] >= 0.9),
        }
        with open(stage_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'  [{stage}] N={n} F1={summary["f1"]:.4f} F1=0:{summary["f1_zero"]} '
              f'F1<0.3:{summary["f1_below_03"]} F1≥0.9:{summary["f1_above_09"]}')


# ============================================================
# Score distribution analysis
# ============================================================

def visualize_score_analysis(all_results, out_dir):
    """Analyze score (probability) distributions for good vs bad cases."""
    # Select representative cases
    f1_zero = sorted([r for r in all_results if r['f1'] == 0 and r['pred_pairs'] > 0
                      and r['length'] >= 50], key=lambda x: -x['length'])[:4]
    f1_low = sorted([r for r in all_results if 0.1 < r['f1'] < 0.3
                     and r['length'] >= 60], key=lambda x: x['f1'])[:3]
    f1_high = sorted([r for r in all_results if r['f1'] >= 0.95
                      and r['length'] >= 50], key=lambda x: -x['length'])[:3]

    cases = f1_zero + f1_low + f1_high
    if not cases:
        return

    # Detailed score analysis per case
    fig, axes = plt.subplots(len(cases), 5, figsize=(18, len(cases) * 2.8))
    fig.suptitle('Score Analysis: Bad vs Good Cases\n(GT | Score | Pred | Diff | Score Histogram)',
                 fontsize=11, fontweight='bold')
    if len(cases) == 1:
        axes = axes.reshape(1, -1)

    for idx, case in enumerate(cases):
        L = case['length']
        gt = case['gt'][:L, :L]
        score = case['score'][:L, :L]
        pred_map = case['pred'][:L, :L]

        # GT
        axes[idx, 0].imshow(gt, cmap='Blues', vmin=0, vmax=1, aspect='equal')
        axes[idx, 0].set_title('GT', fontsize=7)
        axes[idx, 0].set_xticks([]); axes[idx, 0].set_yticks([])
        short = case['name'].replace('bpRNA_', '')
        axes[idx, 0].set_ylabel(f'{short}\nL={L} F1={case["f1"]:.2f}',
                                fontsize=6, rotation=0, labelpad=55, va='center')

        # Score
        axes[idx, 1].imshow(score, cmap='hot', vmin=0, vmax=1, aspect='equal')
        axes[idx, 1].set_title('Score (prob)', fontsize=7)
        axes[idx, 1].set_xticks([]); axes[idx, 1].set_yticks([])

        # Pred
        axes[idx, 2].imshow(pred_map, cmap='Oranges', vmin=0, vmax=1, aspect='equal')
        axes[idx, 2].set_title('Pred', fontsize=7)
        axes[idx, 2].set_xticks([]); axes[idx, 2].set_yticks([])

        # Diff
        diff = np.zeros((L, L, 3))
        p_bin = pred_map > 0.5
        g_bin = gt > 0.5
        diff[(p_bin & g_bin)] = [0.2, 0.8, 0.2]
        diff[(p_bin & ~g_bin)] = [0.9, 0.2, 0.2]
        diff[(~p_bin & g_bin)] = [0.2, 0.4, 0.9]
        axes[idx, 3].imshow(diff, aspect='equal')
        axes[idx, 3].set_title(f'Diff p/g={case["pred_gt_ratio"]:.2f}', fontsize=7)
        axes[idx, 3].set_xticks([]); axes[idx, 3].set_yticks([])

        # Score histogram: at GT=1 positions vs GT=0 positions
        ax = axes[idx, 4]
        mask_upper = np.triu(np.ones((L, L), dtype=bool), k=3)
        gt_pos_scores = score[mask_upper & (gt > 0.5)]
        gt_neg_scores = score[mask_upper & (gt < 0.5)]
        if len(gt_pos_scores) > 0:
            ax.hist(gt_pos_scores.flatten(), bins=30, alpha=0.7, color='blue',
                    label=f'GT=1 (N={len(gt_pos_scores)})', density=True)
        if len(gt_neg_scores) > 0:
            # subsample for speed
            sub = gt_neg_scores.flatten()
            if len(sub) > 5000:
                sub = np.random.choice(sub, 5000, replace=False)
            ax.hist(sub, bins=30, alpha=0.5, color='gray',
                    label=f'GT=0 (sub)', density=True)
        ax.axvline(0.4, color='red', linestyle='--', alpha=0.5, label='threshold=0.4')
        ax.set_xlabel('Score', fontsize=6)
        ax.set_title('Score @ GT positions', fontsize=7)
        ax.legend(fontsize=5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / 'score_analysis_detailed.png', dpi=130, bbox_inches='tight')
    print(f'  Saved: score_analysis_detailed.png')
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--max_train_samples', type=int, default=3000)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = load_config(args.config) if args.config else ckpt.get('config')
    if cfg is None:
        cfg = load_config('symfold/config/v7_full.json')
    scfg = cfg.get('sampling', {})

    # Build model
    class A: pass
    lm_args = A()
    lm_args.pretrained_lm_dir = cfg['paths']['pretrained_lm_dir']
    lm_args.model_scale = cfg['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)
    model = build_model(cfg, extractor)
    model.load_state_dict(ckpt['model'])
    device = torch.device(cfg.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    amp_name = str(cfg.get('training', {}).get('amp_dtype', 'fp32')).lower()
    amp_on = amp_name in ('bf16', 'bfloat16', 'fp16', 'float16')
    amp_dtype = torch.bfloat16 if amp_name in ('bf16', 'bfloat16') else torch.float16

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages = ['bprna-train', 'bprna-val', 'bprna-test']
    stage_results = {}
    all_results = []

    print(f"=== Deep Analysis: v7 DensityNet ===")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Output: {out_dir}")
    print()

    with torch.no_grad():
        for stage in stages:
            loader = build_loader(stage, cfg, tokenizer, shuffle=False)
            results = []
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
                    result = {
                        'stage': stage,
                        'name': batch['names'][i],
                        'category': get_category(batch['names'][i]),
                        'length': length,
                        **m,
                        # Store numpy arrays for visualization
                        'pred': pred[i].cpu().float().squeeze().numpy(),
                        'gt': batch['contact'][i].cpu().float().squeeze().numpy(),
                        'score': score[i].cpu().float().squeeze().numpy(),
                    }
                    results.append(result)
                    all_results.append(result)

                if step % 30 == 0:
                    print(f"  [{stage}] step={step}, samples={len(results)}")
                if max_samples and len(results) >= max_samples:
                    print(f"  [{stage}] max_samples={max_samples} reached")
                    break

            stage_results[stage] = results
            print(f"  [{stage}] done: {len(results)} samples")

    print(f"\nTotal: {len(all_results)} samples")
    print()

    # ============ Generate all visualizations ============

    print("=" * 60)
    print("1. Per-stage analysis (worst/best/middle 10 per stage)...")
    print("=" * 60)
    visualize_per_stage(stage_results, out_dir)

    print()
    print("=" * 60)
    print("2. Failure mode deep-dive...")
    print("=" * 60)
    visualize_failure_modes(all_results, out_dir)

    print()
    print("=" * 60)
    print("3. Per-family analysis...")
    print("=" * 60)
    visualize_family_analysis(all_results, out_dir)

    print()
    print("=" * 60)
    print("4. Score (probability) analysis...")
    print("=" * 60)
    visualize_score_analysis(all_results, out_dir)

    # ============ Summary report ============
    print()
    print("=" * 60)
    print("5. Summary statistics...")
    print("=" * 60)

    summary = {
        'total_samples': len(all_results),
        'by_stage': {},
        'by_family': {},
    }
    for stage, results in stage_results.items():
        n = len(results)
        summary['by_stage'][stage] = {
            'n': n,
            'f1': float(np.mean([r['f1'] for r in results])),
            'precision': float(np.mean([r['precision'] for r in results])),
            'recall': float(np.mean([r['recall'] for r in results])),
            'pred_gt_ratio': float(np.mean([r['pred_gt_ratio'] for r in results])),
            'f1_zero': sum(1 for r in results if r['f1'] == 0),
            'f1_below_03': sum(1 for r in results if r['f1'] < 0.3),
            'f1_above_09': sum(1 for r in results if r['f1'] >= 0.9),
        }
    by_cat = defaultdict(list)
    for r in all_results:
        by_cat[r['category']].append(r)
    for cat, rows in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        summary['by_family'][cat] = {
            'n': len(rows),
            'f1': float(np.mean([r['f1'] for r in rows])),
            'precision': float(np.mean([r['precision'] for r in rows])),
            'recall': float(np.mean([r['recall'] for r in rows])),
            'f1_zero': sum(1 for r in rows if r['f1'] == 0),
            'f1_above_09': sum(1 for r in rows if r['f1'] >= 0.9),
        }

    with open(out_dir / 'deep_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print
    print("\nPer-stage:")
    for stage, s in summary['by_stage'].items():
        print(f"  {stage}: N={s['n']} F1={s['f1']:.4f} P={s['precision']:.4f} "
              f"R={s['recall']:.4f} F1=0:{s['f1_zero']} F1≥0.9:{s['f1_above_09']}")
    print("\nPer-family:")
    for cat, s in summary['by_family'].items():
        print(f"  {cat:8s}: N={s['n']:4d} F1={s['f1']:.4f} P={s['precision']:.4f} "
              f"R={s['recall']:.4f} F1=0:{s['f1_zero']} F1≥0.9:{s['f1_above_09']}")

    print(f"\nAll outputs saved to: {out_dir}")
    print("Files:")
    for p in sorted(out_dir.rglob('*.png')):
        print(f"  {p.relative_to(out_dir)}")


if __name__ == '__main__':
    main()
