# -*- coding: utf-8 -*-
"""PriFold v10 — 跨数据集分析 + Bad Case 溯源.

分析 train/val/test 三个集合的数据分布差异，
并为 test 中的 bad cases 在 train 中寻找相似样本（长度、配对密度、配对模式）。

Usage:
  CUDA_VISIBLE_DEVICES=0 python symfold/eval/cross_split_analysis_v10.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_records, PriFoldSymFlowDataset, make_collate_fn
from symfold.v9.model import DensityNetProPlus

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
    parts = name.split('_')
    return parts[1] if len(parts) >= 3 else 'unknown'


def compute_structure_features(contact_matrix, length):
    """Compute structural features for a sample: density, distance distribution, stem count etc."""
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    idx = torch.arange(length)
    mask &= (idx.view(length, 1) - idx.view(1, length)).abs() >= 3
    
    pairs = contact_matrix[:length, :length] & mask
    n_pairs = int(pairs.sum())
    total_possible = int(mask.sum())
    density = n_pairs / max(total_possible, 1)
    
    if n_pairs == 0:
        return {
            'n_pairs': 0,
            'density': 0,
            'mean_distance': 0,
            'max_distance': 0,
            'short_range_frac': 0,  # |i-j| < 20
            'mid_range_frac': 0,    # 20 <= |i-j| < 100
            'long_range_frac': 0,   # |i-j| >= 100
            'distance_hist': [0] * 5,
        }
    
    pair_positions = torch.where(pairs)
    distances = (pair_positions[1] - pair_positions[0]).float()
    
    mean_dist = float(distances.mean())
    max_dist = int(distances.max())
    
    short_range = int((distances < 20).sum()) / n_pairs
    mid_range = int(((distances >= 20) & (distances < 100)).sum()) / n_pairs
    long_range = int((distances >= 100).sum()) / n_pairs
    
    # Distance histogram: [3-10, 10-20, 20-50, 50-100, 100+]
    bins = [3, 10, 20, 50, 100, 10000]
    hist = []
    for i in range(len(bins) - 1):
        count = int(((distances >= bins[i]) & (distances < bins[i+1])).sum())
        hist.append(count / n_pairs)
    
    return {
        'n_pairs': n_pairs,
        'density': density,
        'mean_distance': mean_dist,
        'max_distance': max_dist,
        'short_range_frac': short_range,
        'mid_range_frac': mid_range,
        'long_range_frac': long_range,
        'distance_hist': hist,
    }


def collect_split_stats(records, dataset, collate_fn, split_name):
    """Collect per-sample structural statistics for a split (no model inference needed)."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    stats = []
    total = len(loader)
    for idx, batch in enumerate(loader):
        if (idx + 1) % 500 == 0:
            print(f'  [{split_name}] {idx+1}/{total}')
        
        length = int(batch['length'][0])
        contact = batch['contact'][0].squeeze() > 0.5
        name = batch['names'][0] if 'names' in batch else records[idx].file_name
        family = get_family(name)
        
        features = compute_structure_features(contact, length)
        features['name'] = name
        features['family'] = family
        features['length'] = length
        features['split'] = split_name
        stats.append(features)
    
    return stats


def evaluate_split(model, loader, device, config, split_name):
    """Run model inference on a split and return per-sample metrics."""
    model.eval()
    scfg = config.get('sampling', {})
    amp_dtype = torch.bfloat16
    per_sample = []
    total = len(loader)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if (idx + 1) % 200 == 0:
                print(f'  [{split_name}] {idx+1}/{total}')
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
                idx_arr = torch.arange(length)
                mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
                mask &= (idx_arr.view(length, 1) - idx_arr.view(1, length)).abs() >= 3
                p_m = p[mask]
                y_m = y[mask]
                tp = int((p_m & y_m).sum())
                fp = int((p_m & ~y_m).sum())
                fn = int((~p_m & y_m).sum())

                gt_pairs = tp + fn
                pred_pairs = tp + fp

                if gt_pairs == 0 and pred_pairs == 0:
                    precision, recall, f1 = 1.0, 1.0, 1.0
                elif gt_pairs == 0 and pred_pairs > 0:
                    precision, recall, f1 = 0.0, 1.0, 0.0
                else:
                    precision = tp / max(tp + fp, 1)
                    recall = tp / max(tp + fn, 1)
                    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

                name = names[i]
                family = get_family(name)

                # Compute structural features from GT
                features = compute_structure_features(y, length)

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
                    'pred_gt_ratio': pred_pairs / max(gt_pairs, 1),
                    'density': features['density'],
                    'mean_distance': features['mean_distance'],
                    'max_distance': features['max_distance'],
                    'short_range_frac': features['short_range_frac'],
                    'mid_range_frac': features['mid_range_frac'],
                    'long_range_frac': features['long_range_frac'],
                    'distance_hist': features['distance_hist'],
                })

    return per_sample


def find_similar_in_train(bad_case, train_stats, top_k=5):
    """Find top-k most similar training samples to a bad case.
    
    Similarity based on: length, density, distance distribution.
    """
    bc_len = bad_case['length']
    bc_density = bad_case['density']
    bc_hist = np.array(bad_case['distance_hist'])
    bc_mean_dist = bad_case['mean_distance']
    
    scores = []
    for ts in train_stats:
        # Length similarity (normalized)
        len_sim = 1.0 - abs(ts['length'] - bc_len) / max(bc_len, ts['length'], 1)
        
        # Density similarity
        den_sim = 1.0 - abs(ts['density'] - bc_density) / max(bc_density, ts['density'], 1e-6)
        
        # Distance histogram similarity (cosine)
        ts_hist = np.array(ts['distance_hist'])
        if np.linalg.norm(bc_hist) > 0 and np.linalg.norm(ts_hist) > 0:
            hist_sim = float(np.dot(bc_hist, ts_hist) / (np.linalg.norm(bc_hist) * np.linalg.norm(ts_hist)))
        else:
            hist_sim = 0.0
        
        # Mean distance similarity
        dist_sim = 1.0 - abs(ts['mean_distance'] - bc_mean_dist) / max(bc_mean_dist, ts['mean_distance'], 1)
        
        # Family match bonus
        family_bonus = 0.1 if ts['family'] == bad_case['family'] else 0.0
        
        # Weighted score
        score = 0.25 * len_sim + 0.25 * den_sim + 0.25 * hist_sim + 0.15 * dist_sim + 0.10 * family_bonus
        scores.append((score, ts))
    
    scores.sort(key=lambda x: -x[0])
    return scores[:top_k]


def generate_cross_split_report(train_stats, val_stats, test_results, train_results, val_results, 
                                  bad_cases_with_matches, elapsed):
    """Generate comprehensive cross-split analysis report."""
    lines = []
    lines.append('# v10 跨数据集分析 + Bad Case 溯源报告')
    lines.append('')
    lines.append(f'> 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'> 评估耗时: {elapsed:.1f}s')
    lines.append('')

    # ========== 1. Split Statistics Comparison ==========
    lines.append('## 1. Train / Val / Test 数据分布对比')
    lines.append('')
    
    all_splits = {'train': train_stats, 'val': val_stats, 'test': [s for s in test_results]}
    
    lines.append('### 1.1 基本统计')
    lines.append('')
    lines.append('| 指标 | Train | Val | Test |')
    lines.append('|------|-------|-----|------|')
    
    for split_name, stats in [('train', train_stats), ('val', val_stats), ('test', test_results)]:
        n = len(stats)
        avg_len = np.mean([s['length'] for s in stats])
        avg_pairs = np.mean([s.get('gt_pairs', s.get('n_pairs', 0)) for s in stats])
        avg_density = np.mean([s['density'] for s in stats])
        lines.append(f'| N={split_name} | {n} | — | — |'.replace('— | — |', ''))
    
    # Proper table
    lines = lines[:-3]  # remove wrong lines
    
    def get_stat(stats, key, default=0):
        return np.mean([s.get(key, default) for s in stats])
    
    lines.append(f'| 样本数 | {len(train_stats)} | {len(val_stats)} | {len(test_results)} |')
    lines.append(f'| 平均长度 | {get_stat(train_stats, "length"):.1f} | {get_stat(val_stats, "length"):.1f} | {get_stat(test_results, "length"):.1f} |')
    
    tr_pairs = get_stat(train_stats, 'n_pairs')
    vl_pairs = get_stat(val_stats, 'n_pairs')
    ts_pairs = np.mean([s['gt_pairs'] for s in test_results])
    lines.append(f'| 平均配对数 | {tr_pairs:.1f} | {vl_pairs:.1f} | {ts_pairs:.1f} |')
    
    lines.append(f'| 平均密度 | {get_stat(train_stats, "density"):.6f} | {get_stat(val_stats, "density"):.6f} | {get_stat(test_results, "density"):.6f} |')
    lines.append(f'| 平均配对距离 | {get_stat(train_stats, "mean_distance"):.1f} | {get_stat(val_stats, "mean_distance"):.1f} | {get_stat(test_results, "mean_distance"):.1f} |')
    lines.append(f'| 短距离比例(<20) | {get_stat(train_stats, "short_range_frac"):.3f} | {get_stat(val_stats, "short_range_frac"):.3f} | {get_stat(test_results, "short_range_frac"):.3f} |')
    lines.append(f'| 中距离比例(20-100) | {get_stat(train_stats, "mid_range_frac"):.3f} | {get_stat(val_stats, "mid_range_frac"):.3f} | {get_stat(test_results, "mid_range_frac"):.3f} |')
    lines.append(f'| 长距离比例(≥100) | {get_stat(train_stats, "long_range_frac"):.3f} | {get_stat(val_stats, "long_range_frac"):.3f} | {get_stat(test_results, "long_range_frac"):.3f} |')
    lines.append('')

    # ========== 1.2 Family Distribution ==========
    lines.append('### 1.2 家族分布对比')
    lines.append('')
    
    all_families = set()
    for stats in [train_stats, val_stats, test_results]:
        for s in stats:
            all_families.add(s['family'])
    
    lines.append('| 家族 | Train N (%) | Val N (%) | Test N (%) |')
    lines.append('|------|-------------|-----------|------------|')
    
    for family in sorted(all_families):
        tr_n = sum(1 for s in train_stats if s['family'] == family)
        vl_n = sum(1 for s in val_stats if s['family'] == family)
        ts_n = sum(1 for s in test_results if s['family'] == family)
        tr_pct = tr_n / len(train_stats) * 100
        vl_pct = vl_n / len(val_stats) * 100
        ts_pct = ts_n / len(test_results) * 100
        lines.append(f'| {family} | {tr_n} ({tr_pct:.1f}%) | {vl_n} ({vl_pct:.1f}%) | {ts_n} ({ts_pct:.1f}%) |')
    lines.append('')

    # ========== 1.3 Length Distribution ==========
    lines.append('### 1.3 长度分布对比')
    lines.append('')
    length_bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 500)]
    lines.append('| 长度区间 | Train N (%) | Val N (%) | Test N (%) |')
    lines.append('|----------|-------------|-----------|------------|')
    for lo, hi in length_bins:
        tr_n = sum(1 for s in train_stats if lo <= s['length'] < hi)
        vl_n = sum(1 for s in val_stats if lo <= s['length'] < hi)
        ts_n = sum(1 for s in test_results if lo <= s['length'] < hi)
        lines.append(f'| {lo}-{hi} | {tr_n} ({tr_n/len(train_stats)*100:.1f}%) | '
                     f'{vl_n} ({vl_n/len(val_stats)*100:.1f}%) | '
                     f'{ts_n} ({ts_n/len(test_results)*100:.1f}%) |')
    lines.append('')

    # ========== 2. Model Performance on Train/Val/Test ==========
    lines.append('## 2. 模型在 Train / Val / Test 上的表现')
    lines.append('')
    lines.append('| 指标 | Train | Val | Test |')
    lines.append('|------|-------|-----|------|')
    
    for metric in ['f1', 'precision', 'recall']:
        tr_val = np.mean([s[metric] for s in train_results])
        vl_val = np.mean([s[metric] for s in val_results])
        ts_val = np.mean([s[metric] for s in test_results])
        lines.append(f'| {metric.capitalize()} | {tr_val:.4f} | {vl_val:.4f} | {ts_val:.4f} |')
    
    tr_bad = sum(1 for s in train_results if s['f1'] < 0.3)
    vl_bad = sum(1 for s in val_results if s['f1'] < 0.3)
    ts_bad = sum(1 for s in test_results if s['f1'] < 0.3)
    lines.append(f'| Bad Rate (F1<0.3) | {tr_bad/len(train_results):.1%} ({tr_bad}) | '
                 f'{vl_bad/len(val_results):.1%} ({vl_bad}) | '
                 f'{ts_bad/len(test_results):.1%} ({ts_bad}) |')
    lines.append('')

    # Performance by family across splits
    lines.append('### 2.1 各家族在不同 split 上的 F1')
    lines.append('')
    lines.append('| 家族 | Train F1 | Val F1 | Test F1 | 泛化差距 (Train-Test) |')
    lines.append('|------|----------|--------|---------|----------------------|')
    for family in sorted(all_families):
        tr_f1s = [s['f1'] for s in train_results if s['family'] == family]
        vl_f1s = [s['f1'] for s in val_results if s['family'] == family]
        ts_f1s = [s['f1'] for s in test_results if s['family'] == family]
        if tr_f1s and ts_f1s:
            tr_f1 = np.mean(tr_f1s)
            vl_f1 = np.mean(vl_f1s) if vl_f1s else 0
            ts_f1 = np.mean(ts_f1s)
            gap = tr_f1 - ts_f1
            lines.append(f'| {family} | {tr_f1:.4f} | {vl_f1:.4f} | {ts_f1:.4f} | {gap:+.4f} |')
    lines.append('')

    # ========== 3. Bad Case Tracing ==========
    lines.append('## 3. Test Bad Cases 溯源分析')
    lines.append('')
    lines.append('对每个 test bad case (F1 < 0.3)，在 train 中寻找结构最相似的样本，')
    lines.append('分析模型是否有足够的类似训练数据。')
    lines.append('')

    # Summary of bad cases vs training coverage
    bad_with_good_match = 0
    bad_with_poor_match = 0
    for bc, matches in bad_cases_with_matches:
        best_score = matches[0][0] if matches else 0
        if best_score > 0.85:
            bad_with_good_match += 1
        else:
            bad_with_poor_match += 1
    
    lines.append(f'| 分类 | 数量 | 说明 |')
    lines.append(f'|------|------|------|')
    lines.append(f'| 有高度相似训练样本 (sim>0.85) | {bad_with_good_match} | 模型见过类似数据但没学好 |')
    lines.append(f'| 缺乏相似训练样本 (sim≤0.85) | {bad_with_poor_match} | 训练数据覆盖不足 |')
    lines.append('')

    # Detail for top-20 worst bad cases
    lines.append('### 3.1 最差 20 个 Bad Cases 的训练集匹配')
    lines.append('')
    
    for i, (bc, matches) in enumerate(bad_cases_with_matches[:20]):
        lines.append(f'#### Bad Case #{i+1}: `{bc["name"]}` (F1={bc["f1"]:.3f})')
        lines.append('')
        lines.append(f'- 长度: {bc["length"]}, GT pairs: {bc["gt_pairs"]}, Pred pairs: {bc["pred_pairs"]}')
        lines.append(f'- 密度: {bc["density"]:.6f}, 平均配对距离: {bc["mean_distance"]:.1f}')
        lines.append(f'- 家族: {bc["family"]}, Precision: {bc["precision"]:.3f}, Recall: {bc["recall"]:.3f}')
        lines.append(f'- 距离分布: 短{bc["short_range_frac"]:.2f} / 中{bc["mid_range_frac"]:.2f} / 长{bc["long_range_frac"]:.2f}')
        lines.append('')
        lines.append(f'  最相似的 train 样本：')
        lines.append('')
        lines.append(f'  | # | Name | Sim | Len | Pairs | Density | Family | Mean Dist |')
        lines.append(f'  |---|------|-----|-----|-------|---------|--------|-----------|')
        for j, (score, ts) in enumerate(matches[:3]):
            lines.append(f'  | {j+1} | {ts["name"][:22]} | {score:.3f} | {ts["length"]} | '
                         f'{ts["n_pairs"]} | {ts["density"]:.6f} | {ts["family"]} | {ts["mean_distance"]:.1f} |')
        lines.append('')

    # ========== 4. Distribution Gap Analysis ==========
    lines.append('## 4. 分布差距总结')
    lines.append('')
    
    # Find test samples that are "out of distribution" vs train
    # Density range in train
    train_densities = [s['density'] for s in train_stats]
    train_den_q5 = np.percentile(train_densities, 5)
    train_den_q95 = np.percentile(train_densities, 95)
    
    train_lengths = [s['length'] for s in train_stats]
    train_len_q5 = np.percentile(train_lengths, 5)
    train_len_q95 = np.percentile(train_lengths, 95)
    
    train_mean_dists = [s['mean_distance'] for s in train_stats if s['mean_distance'] > 0]
    train_dist_q5 = np.percentile(train_mean_dists, 5)
    train_dist_q95 = np.percentile(train_mean_dists, 95)
    
    ood_density = [s for s in test_results if s['density'] < train_den_q5 or s['density'] > train_den_q95]
    ood_length = [s for s in test_results if s['length'] < train_len_q5 or s['length'] > train_len_q95]
    ood_dist = [s for s in test_results if s['mean_distance'] > 0 and 
                (s['mean_distance'] < train_dist_q5 or s['mean_distance'] > train_dist_q95)]
    
    lines.append('Test 中超出 Train 分布 (5%-95% 范围) 的样本：')
    lines.append('')
    lines.append(f'| 维度 | Train 范围 (5%-95%) | OOD 样本数 | OOD 样本平均 F1 | 全体平均 F1 |')
    lines.append(f'|------|---------------------|------------|----------------|-------------|')
    
    ood_den_f1 = np.mean([s['f1'] for s in ood_density]) if ood_density else 0
    ood_len_f1 = np.mean([s['f1'] for s in ood_length]) if ood_length else 0
    ood_dist_f1 = np.mean([s['f1'] for s in ood_dist]) if ood_dist else 0
    all_f1 = np.mean([s['f1'] for s in test_results])
    
    lines.append(f'| 密度 | [{train_den_q5:.6f}, {train_den_q95:.6f}] | {len(ood_density)} | {ood_den_f1:.4f} | {all_f1:.4f} |')
    lines.append(f'| 长度 | [{train_len_q5:.0f}, {train_len_q95:.0f}] | {len(ood_length)} | {ood_len_f1:.4f} | {all_f1:.4f} |')
    lines.append(f'| 配对距离 | [{train_dist_q5:.1f}, {train_dist_q95:.1f}] | {len(ood_dist)} | {ood_dist_f1:.4f} | {all_f1:.4f} |')
    lines.append('')

    # ========== 5. Conclusions ==========
    lines.append('## 5. 核心结论')
    lines.append('')
    lines.append('（自动生成的结论，需人工确认）')
    lines.append('')
    
    # Generalization gap
    tr_f1 = np.mean([s['f1'] for s in train_results])
    ts_f1 = np.mean([s['f1'] for s in test_results])
    lines.append(f'1. **泛化差距**: Train F1={tr_f1:.4f} vs Test F1={ts_f1:.4f}，差距={tr_f1-ts_f1:.4f}')
    
    if bad_with_poor_match > bad_with_good_match:
        lines.append(f'2. **数据覆盖不足**: {bad_with_poor_match}/{len(bad_cases_with_matches)} bad cases '
                     f'在训练集中缺乏相似样本，说明训练数据多样性不足')
    else:
        lines.append(f'2. **模型能力不足**: {bad_with_good_match}/{len(bad_cases_with_matches)} bad cases '
                     f'在训练集中有相似样本但模型仍然预测失败，说明模型能力/训练不足')
    
    lines.append('')
    return '\n'.join(lines)


def main():
    print('=' * 60)
    print('  v10 Cross-Split Analysis + Bad Case Tracing')
    print('=' * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    device = torch.device(DEVICE)
    data_dir = config['paths']['data_dir']
    max_len = config['training'].get('max_len_filter', 490)
    
    # Load LM
    class Args: pass
    lm_args = Args()
    lm_args.pretrained_lm_dir = config['paths'].get('pretrained_lm_dir', str(ROOT / 'model'))
    lm_args.model_scale = config['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)
    
    # Build model
    model = build_model(config, extractor).to(device)
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    new_sd = {k.replace('module.', ''): v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    print(f'Model loaded (epoch {ckpt.get("epoch", "?")})')
    
    collate_fn = make_collate_fn(tokenizer)
    t0 = time.time()
    
    # ============================================================
    # Step 1: Collect structural stats for train (no inference needed, just GT analysis)
    # ============================================================
    print('\n[Step 1] Collecting train structural stats...')
    train_records = build_records(data_dir, 'bprna-train', max_len=max_len)
    train_dataset = PriFoldSymFlowDataset(train_records, augment=False)
    train_stats = collect_split_stats(train_records, train_dataset, collate_fn, 'train')
    print(f'  Train: {len(train_stats)} samples')
    
    # ============================================================
    # Step 2: Collect val stats
    # ============================================================
    print('\n[Step 2] Collecting val structural stats...')
    val_records = build_records(data_dir, 'bprna-val', max_len=max_len)
    val_dataset = PriFoldSymFlowDataset(val_records, augment=False)
    val_stats = collect_split_stats(val_records, val_dataset, collate_fn, 'val')
    print(f'  Val: {len(val_stats)} samples')
    
    # ============================================================
    # Step 3: Run model on train (sample 500 for speed)
    # ============================================================
    print('\n[Step 3] Evaluating model on train (sampled 500)...')
    np.random.seed(42)
    train_sample_idx = np.random.choice(len(train_dataset), min(500, len(train_dataset)), replace=False)
    train_sample_dataset = torch.utils.data.Subset(train_dataset, train_sample_idx)
    train_loader = DataLoader(train_sample_dataset, batch_size=1, shuffle=False, 
                              collate_fn=collate_fn, num_workers=4, pin_memory=True)
    train_results = evaluate_split(model, train_loader, device, config, 'train')
    print(f'  Train eval: F1={np.mean([s["f1"] for s in train_results]):.4f}')
    
    # ============================================================
    # Step 4: Run model on val
    # ============================================================
    print('\n[Step 4] Evaluating model on val...')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                            collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_results = evaluate_split(model, val_loader, device, config, 'val')
    print(f'  Val eval: F1={np.mean([s["f1"] for s in val_results]):.4f}')
    
    # ============================================================
    # Step 5: Load test results (from previous analysis)
    # ============================================================
    print('\n[Step 5] Evaluating model on test...')
    test_records = build_records(data_dir, 'bprna-test', max_len=max_len)
    test_dataset = PriFoldSymFlowDataset(test_records, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)
    test_results = evaluate_split(model, test_loader, device, config, 'test')
    print(f'  Test eval: F1={np.mean([s["f1"] for s in test_results]):.4f}')
    
    # ============================================================
    # Step 6: Bad case tracing
    # ============================================================
    print('\n[Step 6] Tracing bad cases in training set...')
    bad_cases = sorted([s for s in test_results if s['f1'] < 0.3], key=lambda x: x['f1'])
    print(f'  {len(bad_cases)} bad cases to trace')
    
    bad_cases_with_matches = []
    for i, bc in enumerate(bad_cases):
        if (i + 1) % 20 == 0:
            print(f'  Tracing {i+1}/{len(bad_cases)}...')
        matches = find_similar_in_train(bc, train_stats, top_k=5)
        bad_cases_with_matches.append((bc, matches))
    
    elapsed = time.time() - t0
    
    # ============================================================
    # Step 7: Generate report
    # ============================================================
    print('\n[Step 7] Generating report...')
    report = generate_cross_split_report(
        train_stats, val_stats, test_results, train_results, val_results,
        bad_cases_with_matches, elapsed
    )
    
    report_path = OUTPUT_DIR / 'v10_cross_split_analysis.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'Report saved: {report_path}')
    
    # Save stats
    stats_path = OUTPUT_DIR / 'split_stats_summary.json'
    summary = {
        'train': {
            'n': len(train_stats),
            'avg_length': float(np.mean([s['length'] for s in train_stats])),
            'avg_pairs': float(np.mean([s['n_pairs'] for s in train_stats])),
            'avg_density': float(np.mean([s['density'] for s in train_stats])),
            'model_f1': float(np.mean([s['f1'] for s in train_results])),
        },
        'val': {
            'n': len(val_stats),
            'avg_length': float(np.mean([s['length'] for s in val_stats])),
            'avg_pairs': float(np.mean([s['n_pairs'] for s in val_stats])),
            'avg_density': float(np.mean([s['density'] for s in val_stats])),
            'model_f1': float(np.mean([s['f1'] for s in val_results])),
        },
        'test': {
            'n': len(test_results),
            'avg_length': float(np.mean([s['length'] for s in test_results])),
            'avg_pairs': float(np.mean([s['gt_pairs'] for s in test_results])),
            'avg_density': float(np.mean([s['density'] for s in test_results])),
            'model_f1': float(np.mean([s['f1'] for s in test_results])),
        },
    }
    with open(stats_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f'\n{"="*60}')
    print(f'  DONE in {elapsed:.1f}s')
    print(f'  Report: {report_path}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
