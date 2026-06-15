# -*- coding: utf-8 -*-
"""Visualize v6 predictions: good/bad cases, contact maps, and category analysis.

Generates:
1. Contact map visualizations for 20 best + 20 worst cases
2. Category-level performance breakdown
3. RFAM failure pattern analysis

Usage:
  python symfold/visualize_predictions.py \
    --ckpt symfold/outputs/v6_full/model/best.pt \
    --config symfold/config/v6_full.json \
    --out_dir symfold/outputs/v6_full/visualizations
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
import matplotlib.gridspec as gridspec
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
from symfold.train.train_v6 import build_model, load_config, move_to_device


def plot_contact_map_comparison(pred, gt, length, name, f1, precision, recall, 
                                 pred_gt_ratio, ax_pred, ax_gt, ax_diff):
    """Plot pred vs GT contact maps side by side with diff."""
    p = pred[:length, :length]
    g = gt[:length, :length]
    
    # GT
    ax_gt.imshow(g, cmap='Blues', vmin=0, vmax=1, aspect='equal')
    ax_gt.set_title(f'GT', fontsize=8)
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])
    
    # Pred
    ax_pred.imshow(p, cmap='Oranges', vmin=0, vmax=1, aspect='equal')
    ax_pred.set_title(f'Pred', fontsize=8)
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])
    
    # Diff: TP=green, FP=red, FN=blue
    diff = np.zeros((length, length, 3))
    tp_mask = (p > 0.5) & (g > 0.5)
    fp_mask = (p > 0.5) & (g < 0.5)
    fn_mask = (p < 0.5) & (g > 0.5)
    diff[tp_mask] = [0.2, 0.8, 0.2]  # green = TP
    diff[fp_mask] = [0.9, 0.2, 0.2]  # red = FP
    diff[fn_mask] = [0.2, 0.4, 0.9]  # blue = FN
    ax_diff.imshow(diff, aspect='equal')
    ax_diff.set_title(f'F1={f1:.3f} P={precision:.2f} R={recall:.2f}\np/g={pred_gt_ratio:.2f}', fontsize=7)
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])


def generate_predictions(model, loader, device, amp_on, amp_dtype, scfg, max_samples=None):
    """Run model and collect per-sample predictions."""
    results = []
    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch = move_to_device(batch, device)
            sample_kwargs = dict(
                num_steps=scfg.get('num_steps', 20),
                num_samples_per_input=1,
                density_guided=False,
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
                p_np = pred[i].cpu().float().squeeze().numpy()[:length, :length]
                g_np = batch['contact'][i].cpu().float().squeeze().numpy()[:length, :length]
                s_np = score[i].cpu().float().squeeze().numpy()[:length, :length]
                
                # Compute metrics
                p_bin = p_np > 0.5
                g_bin = g_np > 0.5
                idx = np.arange(length)
                mask = np.triu(np.ones((length, length), dtype=bool), k=1)
                mask &= np.abs(idx[:, None] - idx[None, :]) >= 3
                tp = int((p_bin[mask] & g_bin[mask]).sum())
                fp = int((p_bin[mask] & ~g_bin[mask]).sum())
                fn = int((~p_bin[mask] & g_bin[mask]).sum())
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-12)
                gt_pairs = tp + fn
                pred_pairs = tp + fp
                
                name = batch['names'][i]
                parts = name.split('_')
                category = parts[1] if len(parts) >= 3 else 'unknown'
                
                results.append({
                    'name': name,
                    'category': category,
                    'length': length,
                    'density': gt_pairs / max(length, 1),
                    'f1': f1,
                    'precision': prec,
                    'recall': rec,
                    'pred_gt_ratio': pred_pairs / max(gt_pairs, 1),
                    'gt_pairs': gt_pairs,
                    'pred_pairs': pred_pairs,
                    'tp': tp, 'fp': fp, 'fn': fn,
                    'pred': p_np,
                    'gt': g_np,
                    'score': s_np,
                })
            
            if max_samples and len(results) >= max_samples:
                break
            if step % 30 == 0:
                print(f"  step={step}/{len(loader)}, collected={len(results)}")
    
    return results


def visualize_cases(results, out_dir, n_cases=20):
    """Generate contact map visualizations for best and worst cases."""
    sorted_by_f1 = sorted(results, key=lambda x: x['f1'])
    
    # Worst N (excluding very short ones for readability)
    worst = [r for r in sorted_by_f1 if r['length'] >= 40][:n_cases]
    # Best N
    best = sorted(results, key=lambda x: -x['f1'])[:n_cases]
    
    for label, cases in [('worst', worst), ('best', best)]:
        # 4 cases per row, 5 rows = 20 cases
        n_rows = 5
        n_cols = 4
        fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
        fig.suptitle(f'{label.upper()} {n_cases} Cases (bpRNA-test)', fontsize=14, fontweight='bold')
        
        for idx, case in enumerate(cases[:n_rows * n_cols]):
            row_block = idx // n_cols
            col = idx % n_cols
            ax_gt = axes[row_block * 3, col]
            ax_pred = axes[row_block * 3 + 1, col]
            ax_diff = axes[row_block * 3 + 2, col]
            
            plot_contact_map_comparison(
                case['pred'], case['gt'], case['length'],
                case['name'], case['f1'], case['precision'], case['recall'],
                case['pred_gt_ratio'], ax_pred, ax_gt, ax_diff)
            
            # Add name above GT
            short_name = case['name'].replace('bpRNA_', '')
            ax_gt.set_ylabel(f'{short_name}\nL={case["length"]}', fontsize=6, rotation=0, labelpad=50, va='center')
        
        # Hide empty axes
        for i in range(len(cases), n_rows * n_cols):
            row_block = i // n_cols
            col = i % n_cols
            for j in range(3):
                axes[row_block * 3 + j, col].axis('off')
        
        # Add legend
        legend_elements = [
            Patch(facecolor=[0.2, 0.8, 0.2], label='TP (correct)'),
            Patch(facecolor=[0.9, 0.2, 0.2], label='FP (over-predict)'),
            Patch(facecolor=[0.2, 0.4, 0.9], label='FN (missed)'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9)
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        path = out_dir / f'{label}_{n_cases}_contact_maps.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        print(f'  Saved: {path}')
        plt.close()


def visualize_category_analysis(results, out_dir):
    """Generate category-level analysis visualizations."""
    # Group by category
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r['category']].append(r)
    
    cat_order = sorted(by_cat.keys(), key=lambda c: -len(by_cat[c]))
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('bpRNA-test: Performance by RNA Category', fontsize=14, fontweight='bold')
    
    # 1. F1 distribution per category (violin/box)
    ax = axes[0, 0]
    cat_f1s = [np.array([r['f1'] for r in by_cat[c]]) for c in cat_order]
    bp = ax.boxplot(cat_f1s, labels=cat_order, patch_artist=True, showmeans=True)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Distribution by Category')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    # Add N labels
    for i, c in enumerate(cat_order):
        ax.text(i + 1, -0.08, f'N={len(by_cat[c])}', ha='center', fontsize=7)
    
    # 2. Category proportion pie
    ax = axes[0, 1]
    sizes = [len(by_cat[c]) for c in cat_order]
    ax.pie(sizes, labels=cat_order, colors=colors, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 9})
    ax.set_title('Category Distribution')
    
    # 3. Mean F1 + pred/gt bar
    ax = axes[0, 2]
    x = np.arange(len(cat_order))
    f1_means = [np.mean([r['f1'] for r in by_cat[c]]) for c in cat_order]
    pg_means = [np.mean([r['pred_gt_ratio'] for r in by_cat[c]]) for c in cat_order]
    width = 0.35
    bars1 = ax.bar(x - width/2, f1_means, width, label='Mean F1', color='#3498db', alpha=0.7)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, pg_means, width, label='pred/gt', color='#e67e22', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_order)
    ax.set_ylabel('F1', color='#3498db')
    ax2.set_ylabel('pred/gt ratio', color='#e67e22')
    ax2.axhline(1.0, color='black', linestyle='--', alpha=0.3)
    ax.set_title('Mean F1 & pred/gt by Category')
    ax.set_ylim(0, 1.0)
    ax2.set_ylim(0, 3.0)
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    
    # 4. RFAM F1 histogram vs others
    ax = axes[1, 0]
    rfam_f1 = [r['f1'] for r in by_cat.get('RFAM', [])]
    other_f1 = [r['f1'] for r in results if r['category'] != 'RFAM']
    bins = np.linspace(0, 1, 21)
    ax.hist(rfam_f1, bins=bins, alpha=0.6, label=f'RFAM (N={len(rfam_f1)})', color='#e74c3c', density=True)
    ax.hist(other_f1, bins=bins, alpha=0.6, label=f'Others (N={len(other_f1)})', color='#2ecc71', density=True)
    ax.set_xlabel('F1 Score')
    ax.set_ylabel('Density')
    ax.set_title('F1 Distribution: RFAM vs Others')
    ax.legend()
    
    # 5. RFAM by length
    ax = axes[1, 1]
    rfam_rows = by_cat.get('RFAM', [])
    len_bins = ['<80', '80-159', '160-239', '240-319', '320+']
    rfam_by_len = defaultdict(list)
    for r in rfam_rows:
        L = r['length']
        if L < 80: b = '<80'
        elif L < 160: b = '80-159'
        elif L < 240: b = '160-239'
        elif L < 320: b = '240-319'
        else: b = '320+'
        rfam_by_len[b].append(r)
    
    x = np.arange(len(len_bins))
    f1_vals = [np.mean([r['f1'] for r in rfam_by_len[b]]) if rfam_by_len[b] else 0 for b in len_bins]
    counts = [len(rfam_by_len[b]) for b in len_bins]
    f1_zero_pct = [100 * sum(1 for r in rfam_by_len[b] if r['f1'] == 0) / max(len(rfam_by_len[b]), 1) for b in len_bins]
    
    bars = ax.bar(x, f1_vals, color='#e74c3c', alpha=0.7)
    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'N={counts[i]}\n{f1_zero_pct[i]:.0f}%=0', ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(len_bins)
    ax.set_xlabel('Length Bin')
    ax.set_ylabel('Mean F1')
    ax.set_title('RFAM: F1 by Length (% shows F1=0 rate)')
    ax.set_ylim(0, 1.0)
    
    # 6. F1=0 cases analysis
    ax = axes[1, 2]
    f1_zero = [r for r in results if r['f1'] == 0]
    f1_zero_cats = Counter(r['category'] for r in f1_zero)
    if f1_zero:
        labels = [f'{cat}\n(N={cnt})' for cat, cnt in f1_zero_cats.most_common()]
        sizes = [cnt for _, cnt in f1_zero_cats.most_common()]
        ax.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 9})
        ax.set_title(f'F1=0 Cases by Category (total={len(f1_zero)})')
    else:
        ax.text(0.5, 0.5, 'No F1=0 cases', ha='center', va='center')
    
    plt.tight_layout()
    path = out_dir / 'category_analysis.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close()


def visualize_rfam_failure_patterns(results, out_dir):
    """Detailed RFAM failure analysis."""
    rfam = [r for r in results if r['category'] == 'RFAM']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RFAM Failure Pattern Analysis', fontsize=14, fontweight='bold')
    
    # 1. Scatter: F1 vs length, colored by density
    ax = axes[0, 0]
    lengths = [r['length'] for r in rfam]
    f1s = [r['f1'] for r in rfam]
    densities = [r['density'] for r in rfam]
    sc = ax.scatter(lengths, f1s, c=densities, cmap='RdYlGn', s=10, alpha=0.5, vmin=0, vmax=0.4)
    plt.colorbar(sc, ax=ax, label='Density')
    ax.set_xlabel('Length')
    ax.set_ylabel('F1')
    ax.set_title('RFAM: F1 vs Length (colored by density)')
    ax.axhline(0.3, color='red', linestyle='--', alpha=0.5, label='F1=0.3 threshold')
    ax.legend(fontsize=8)
    
    # 2. Failure modes breakdown
    ax = axes[0, 1]
    bad_rfam = [r for r in rfam if r['f1'] < 0.3]
    mode_complete_miss = len([r for r in bad_rfam if r['f1'] == 0 and r['tp'] == 0])
    mode_wrong_pos = len([r for r in bad_rfam if r['f1'] > 0 and 0.8 < r['pred_gt_ratio'] < 1.2])
    mode_overpredict = len([r for r in bad_rfam if r['pred_gt_ratio'] > 2])
    mode_other = len(bad_rfam) - mode_complete_miss - mode_wrong_pos - mode_overpredict
    
    labels = [f'Complete Miss\n(F1=0, tp=0)\nn={mode_complete_miss}',
              f'Right Count\nWrong Position\nn={mode_wrong_pos}',
              f'Over-predict\n(p/g>2)\nn={mode_overpredict}',
              f'Other\nn={mode_other}']
    sizes = [mode_complete_miss, mode_wrong_pos, mode_overpredict, mode_other]
    colors_pie = ['#e74c3c', '#f39c12', '#9b59b6', '#95a5a6']
    ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
           startangle=90, textprops={'fontsize': 8})
    ax.set_title(f'RFAM Failure Modes (F1<0.3, N={len(bad_rfam)})')
    
    # 3. pred/gt ratio vs density
    ax = axes[1, 0]
    pg_ratios = [min(r['pred_gt_ratio'], 5) for r in rfam]
    f1_colors = [r['f1'] for r in rfam]
    sc = ax.scatter(densities, pg_ratios, c=f1_colors, cmap='RdYlGn', s=10, alpha=0.5, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='F1')
    ax.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Density')
    ax.set_ylabel('pred/gt ratio (capped at 5)')
    ax.set_title('RFAM: pred/gt vs Density (colored by F1)')
    
    # 4. F1=0 pattern: do they have specific length/density profiles?
    ax = axes[1, 1]
    f1_zero = [r for r in rfam if r['f1'] == 0]
    f1_good = [r for r in rfam if r['f1'] >= 0.7]
    f1_mid = [r for r in rfam if 0.3 <= r['f1'] < 0.7]
    
    # Plot density distributions
    bins = np.linspace(0, 0.4, 16)
    if f1_zero:
        ax.hist([r['density'] for r in f1_zero], bins=bins, alpha=0.5, 
                label=f'F1=0 (N={len(f1_zero)})', color='#e74c3c', density=True)
    if f1_mid:
        ax.hist([r['density'] for r in f1_mid], bins=bins, alpha=0.5,
                label=f'0.3≤F1<0.7 (N={len(f1_mid)})', color='#f39c12', density=True)
    if f1_good:
        ax.hist([r['density'] for r in f1_good], bins=bins, alpha=0.5,
                label=f'F1≥0.7 (N={len(f1_good)})', color='#2ecc71', density=True)
    ax.set_xlabel('Density')
    ax.set_ylabel('Probability Density')
    ax.set_title('RFAM: Density Distribution by F1 Quality')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    path = out_dir / 'rfam_failure_patterns.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close()


def visualize_score_heatmaps(results, out_dir, n=10):
    """Show score heatmaps for interesting cases."""
    # Select diverse cases: some F1=0, some medium, some good
    f1_zero = sorted([r for r in results if r['f1'] == 0 and r['length'] >= 50],
                     key=lambda x: -x['length'])[:4]
    f1_low = sorted([r for r in results if 0.1 < r['f1'] < 0.3 and r['length'] >= 60],
                    key=lambda x: x['f1'])[:3]
    f1_high = sorted([r for r in results if r['f1'] >= 0.95],
                     key=lambda x: -x['length'])[:3]
    
    cases = f1_zero + f1_low + f1_high
    
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, len(cases) * 2.5))
    fig.suptitle('Score Heatmaps: Failure vs Success Patterns', fontsize=13, fontweight='bold')
    
    for idx, case in enumerate(cases):
        L = case['length']
        # Score heatmap
        ax = axes[idx, 0]
        im = ax.imshow(case['score'][:L, :L], cmap='hot', vmin=0, vmax=1, aspect='equal')
        ax.set_title(f'Score', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        short_name = case['name'].replace('bpRNA_', '')
        ax.set_ylabel(f'{short_name}\nL={L} F1={case["f1"]:.2f}', fontsize=7, rotation=0, labelpad=60, va='center')
        
        # GT
        ax = axes[idx, 1]
        ax.imshow(case['gt'][:L, :L], cmap='Blues', vmin=0, vmax=1, aspect='equal')
        ax.set_title('GT', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Diff
        ax = axes[idx, 2]
        diff = np.zeros((L, L, 3))
        p_bin = case['pred'][:L, :L] > 0.5
        g_bin = case['gt'][:L, :L] > 0.5
        diff[(p_bin & g_bin)] = [0.2, 0.8, 0.2]
        diff[(p_bin & ~g_bin)] = [0.9, 0.2, 0.2]
        diff[(~p_bin & g_bin)] = [0.2, 0.4, 0.9]
        ax.imshow(diff, aspect='equal')
        ax.set_title(f'TP/FP/FN  p/g={case["pred_gt_ratio"]:.2f}', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    
    legend_elements = [
        Patch(facecolor=[0.2, 0.8, 0.2], label='TP'),
        Patch(facecolor=[0.9, 0.2, 0.2], label='FP'),
        Patch(facecolor=[0.2, 0.4, 0.9], label='FN'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    path = out_dir / 'score_heatmaps.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--out_dir', default='symfold/outputs/v6_full/visualizations')
    ap.add_argument('--n_cases', type=int, default=20)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = load_config(args.config) if args.config else ckpt['config']
    scfg = cfg.get('sampling', {})

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

    # Generate predictions for test set
    print("Generating predictions for bprna-test...")
    loader = build_loader('bprna-test', cfg, tokenizer, shuffle=False)
    results = generate_predictions(model, loader, device, amp_on, amp_dtype, scfg)
    print(f"  Total samples: {len(results)}")

    # 1. Contact map visualizations (best/worst 20)
    print("\nGenerating contact map visualizations...")
    visualize_cases(results, out_dir, n_cases=args.n_cases)

    # 2. Category analysis
    print("\nGenerating category analysis...")
    visualize_category_analysis(results, out_dir)

    # 3. RFAM failure patterns
    print("\nGenerating RFAM failure pattern analysis...")
    visualize_rfam_failure_patterns(results, out_dir)

    # 4. Score heatmaps
    print("\nGenerating score heatmaps...")
    visualize_score_heatmaps(results, out_dir)

    # 5. Save summary stats
    summary = {
        'total': len(results),
        'by_category': {},
    }
    for cat in set(r['category'] for r in results):
        subset = [r for r in results if r['category'] == cat]
        summary['by_category'][cat] = {
            'n': len(subset),
            'f1_mean': float(np.mean([r['f1'] for r in subset])),
            'f1_median': float(np.median([r['f1'] for r in subset])),
            'precision': float(np.mean([r['precision'] for r in subset])),
            'recall': float(np.mean([r['recall'] for r in subset])),
            'pred_gt_ratio': float(np.mean([r['pred_gt_ratio'] for r in subset])),
            'f1_zero_count': sum(1 for r in subset if r['f1'] == 0),
            'f1_below_03': sum(1 for r in subset if r['f1'] < 0.3),
            'f1_above_09': sum(1 for r in subset if r['f1'] >= 0.9),
            'avg_length': float(np.mean([r['length'] for r in subset])),
            'avg_density': float(np.mean([r['density'] for r in subset])),
        }
    
    with open(out_dir / 'visualization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll visualizations saved to: {out_dir}")
    print(f"Files generated:")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == '__main__':
    main()
