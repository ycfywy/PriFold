# -*- coding: utf-8 -*-
"""PriFold v10 — Bad Case 逐样本深度分析.

对每个 bad case (F1 < 0.3)：
1. 提取 GT 配对列表 + 每对的预测概率
2. 提取 Pred 配对列表 + 每对的预测概率
3. 分析 GT 中为什么被漏掉（概率低）
4. 分析 Pred 中为什么是 FP（概率高但 GT 中没有）
5. 统计概率分布

Usage:
  CUDA_VISIBLE_DEVICES=0 python symfold/eval/badcase_deep_analysis_v10.py
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


def analyze_single_sample(model, batch, device, config):
    """Analyze a single sample in detail, returning GT/Pred pairs with probabilities."""
    model.eval()
    scfg = config.get('sampling', {})
    amp_dtype = torch.bfloat16

    batch_dev = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            pred, score = model.predict(
                batch_dev,
                budget_fraction=scfg.get('default_budget_fraction', 0.30),
                use_density_budget=scfg.get('use_density_budget', True),
                score_threshold=scfg.get('score_threshold', 0.43),
                length_decay=scfg.get('length_decay', 0.15),
                budget_floor=scfg.get('budget_floor', 0.6),
            )

    # Get data
    length = int(batch['length'][0])
    pred_map = pred[0, 0, :length, :length].cpu() > 0.5
    score_map = score[0, 0, :length, :length].cpu().float()
    gt_map = batch['contact'][0].squeeze()[:length, :length].cpu() > 0.5
    name = batch['names'][0] if 'names' in batch else 'unknown'
    seq = batch.get('seq', [''])[0] if 'seq' in batch else ''

    # Mask: upper triangle, |i-j| >= 3
    idx = torch.arange(length)
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    mask &= (idx.view(length, 1) - idx.view(1, length)).abs() >= 3

    # Extract GT pairs with their predicted probabilities
    gt_pairs_mask = gt_map & mask
    gt_positions = torch.where(gt_pairs_mask)
    gt_pairs = []
    for k in range(len(gt_positions[0])):
        i, j = int(gt_positions[0][k]), int(gt_positions[1][k])
        prob = float(score_map[i, j])
        gt_pairs.append({'i': i, 'j': j, 'distance': j - i, 'pred_prob': prob})

    # Extract Pred pairs with their probabilities
    pred_pairs_mask = pred_map & mask
    pred_positions = torch.where(pred_pairs_mask)
    pred_pairs = []
    for k in range(len(pred_positions[0])):
        i, j = int(pred_positions[0][k]), int(pred_positions[1][k])
        prob = float(score_map[i, j])
        is_tp = bool(gt_map[i, j])
        pred_pairs.append({'i': i, 'j': j, 'distance': j - i, 'pred_prob': prob, 'is_tp': is_tp})

    # Classify GT pairs
    tp_pairs = [p for p in gt_pairs if pred_map[p['i'], p['j']]]
    fn_pairs = [p for p in gt_pairs if not pred_map[p['i'], p['j']]]

    # FP pairs
    fp_pairs = [p for p in pred_pairs if not p['is_tp']]

    # For each FN pair, find the nearest predicted pair (shifted analysis)
    for fn in fn_pairs:
        best_shift = None
        best_shift_prob = 0
        for di in range(-2, 3):
            for dj in range(-2, 3):
                if di == 0 and dj == 0:
                    continue
                ni, nj = fn['i'] + di, fn['j'] + dj
                if 0 <= ni < length and 0 <= nj < length:
                    if pred_map[ni, nj]:
                        prob = float(score_map[ni, nj])
                        if prob > best_shift_prob:
                            best_shift = (ni, nj, di, dj)
                            best_shift_prob = prob
        fn['nearest_pred_shift'] = best_shift
        fn['nearest_pred_prob'] = best_shift_prob

    # Score map statistics
    # Top-k scores in upper triangle
    upper_scores = score_map[mask].numpy()
    gt_probs = np.array([p['pred_prob'] for p in gt_pairs])
    fp_probs = np.array([p['pred_prob'] for p in fp_pairs]) if fp_pairs else np.array([])
    fn_probs = np.array([p['pred_prob'] for p in fn_pairs]) if fn_pairs else np.array([])

    # Overall score distribution
    score_stats = {
        'mean_all': float(upper_scores.mean()),
        'max_all': float(upper_scores.max()),
        'gt_prob_mean': float(gt_probs.mean()) if len(gt_probs) > 0 else 0,
        'gt_prob_min': float(gt_probs.min()) if len(gt_probs) > 0 else 0,
        'gt_prob_max': float(gt_probs.max()) if len(gt_probs) > 0 else 0,
        'fp_prob_mean': float(fp_probs.mean()) if len(fp_probs) > 0 else 0,
        'fp_prob_min': float(fp_probs.min()) if len(fp_probs) > 0 else 0,
        'fn_prob_mean': float(fn_probs.mean()) if len(fn_probs) > 0 else 0,
        'fn_prob_max': float(fn_probs.max()) if len(fn_probs) > 0 else 0,
        'above_threshold_count': int((upper_scores > 0.43).sum()),
    }

    # Metrics
    tp = len(tp_pairs)
    fp = len(fp_pairs)
    fn = len(fn_pairs)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        'name': name,
        'family': get_family(name),
        'length': length,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'n_gt_pairs': len(gt_pairs),
        'n_pred_pairs': len(pred_pairs),
        'tp': tp, 'fp': fp, 'fn': fn,
        'gt_pairs': gt_pairs,
        'tp_pairs': tp_pairs,
        'fn_pairs': fn_pairs,
        'fp_pairs': fp_pairs,
        'score_stats': score_stats,
    }


def generate_badcase_report(analyses):
    """Generate detailed markdown report for bad cases."""
    lines = []
    lines.append('# v10 Bad Case 逐样本深度分析报告')
    lines.append('')
    lines.append(f'> 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'> 分析样本数: {len(analyses)} (Test F1 < 0.3)')
    lines.append('')

    # ========== Overview ==========
    lines.append('## 1. Bad Cases 概率分布总结')
    lines.append('')
    lines.append('### 1.1 GT 配对在模型中的概率分布')
    lines.append('')
    lines.append('模型对 Ground Truth 配对位置给出的概率（应该高但实际低）：')
    lines.append('')

    all_gt_probs = []
    all_fn_probs = []
    all_fp_probs = []
    for a in analyses:
        all_gt_probs.extend([p['pred_prob'] for p in a['gt_pairs']])
        all_fn_probs.extend([p['pred_prob'] for p in a['fn_pairs']])
        all_fp_probs.extend([p['pred_prob'] for p in a['fp_pairs']])

    all_gt_probs = np.array(all_gt_probs)
    all_fn_probs = np.array(all_fn_probs)
    all_fp_probs = np.array(all_fp_probs)

    lines.append('| 类别 | N | 概率 Mean | Median | Min | Max | >0.43 比例 | >0.5 比例 |')
    lines.append('|------|---|-----------|--------|-----|-----|------------|-----------|')
    for name, probs in [('GT 配对 (全部)', all_gt_probs),
                         ('GT 中被漏掉 (FN)', all_fn_probs),
                         ('FP (错误预测)', all_fp_probs)]:
        if len(probs) > 0:
            lines.append(f'| {name} | {len(probs)} | {probs.mean():.4f} | {np.median(probs):.4f} | '
                         f'{probs.min():.4f} | {probs.max():.4f} | '
                         f'{(probs > 0.43).mean():.1%} | {(probs > 0.5).mean():.1%} |')
    lines.append('')

    # GT probability distribution
    lines.append('### 1.2 GT 配对概率分箱统计')
    lines.append('')
    lines.append('| 概率区间 | GT Pairs | FN (漏掉) | 说明 |')
    lines.append('|----------|----------|-----------|------|')
    prob_bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.43), (0.43, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
    for lo, hi in prob_bins:
        gt_in_bin = int(((all_gt_probs >= lo) & (all_gt_probs < hi)).sum())
        fn_in_bin = int(((all_fn_probs >= lo) & (all_fn_probs < hi)).sum())
        note = ''
        if hi <= 0.43:
            note = '低于阈值，不会被选中'
        elif lo >= 0.43 and hi <= 0.5:
            note = '在阈值附近'
        lines.append(f'| [{lo:.2f}, {hi:.2f}) | {gt_in_bin} | {fn_in_bin} | {note} |')
    lines.append('')

    # ========== Shifted FN Analysis ==========
    lines.append('### 1.3 FN 配对的 Shifted 分析')
    lines.append('')
    lines.append('对于被漏掉的 GT 配对，检查 ±2 范围内是否有预测命中（位置偏移）：')
    lines.append('')

    fn_with_shift = 0
    fn_no_shift = 0
    for a in analyses:
        for fn in a['fn_pairs']:
            if fn['nearest_pred_shift'] is not None:
                fn_with_shift += 1
            else:
                fn_no_shift += 1

    total_fn = fn_with_shift + fn_no_shift
    lines.append(f'- FN 总数: {total_fn}')
    lines.append(f'- 有 ±2 偏移预测: {fn_with_shift} ({fn_with_shift/max(total_fn,1):.1%}) — 模型看到了但位置不精确')
    lines.append(f'- 完全没有预测: {fn_no_shift} ({fn_no_shift/max(total_fn,1):.1%}) — 模型完全没感知到')
    lines.append('')

    # ========== Per-sample detailed analysis ==========
    lines.append('## 2. 逐样本详细分析')
    lines.append('')

    # Sort by F1 (worst first)
    sorted_analyses = sorted(analyses, key=lambda x: x['f1'])

    for idx, a in enumerate(sorted_analyses):
        lines.append(f'### Sample #{idx+1}: `{a["name"]}` (F1={a["f1"]:.3f})')
        lines.append('')
        lines.append(f'| 属性 | 值 |')
        lines.append(f'|------|-----|')
        lines.append(f'| 家族 | {a["family"]} |')
        lines.append(f'| 长度 | {a["length"]} |')
        lines.append(f'| GT pairs | {a["n_gt_pairs"]} |')
        lines.append(f'| Pred pairs | {a["n_pred_pairs"]} |')
        lines.append(f'| TP / FP / FN | {a["tp"]} / {a["fp"]} / {a["fn"]} |')
        lines.append(f'| Precision / Recall | {a["precision"]:.3f} / {a["recall"]:.3f} |')
        lines.append(f'| GT 平均概率 | {a["score_stats"]["gt_prob_mean"]:.4f} |')
        lines.append(f'| GT 最低概率 | {a["score_stats"]["gt_prob_min"]:.4f} |')
        lines.append(f'| FP 平均概率 | {a["score_stats"]["fp_prob_mean"]:.4f} |')
        lines.append(f'| FN 平均概率 | {a["score_stats"]["fn_prob_mean"]:.4f} |')
        lines.append(f'| Score>0.43 位置数 | {a["score_stats"]["above_threshold_count"]} |')
        lines.append('')

        # GT pairs table
        lines.append(f'**Ground Truth 配对 (共 {a["n_gt_pairs"]} 对) + 模型给出的概率：**')
        lines.append('')
        lines.append(f'| # | 位置 (i,j) | 距离 |i-j| | 预测概率 | 状态 | 说明 |')
        lines.append(f'|---|------------|---------|---------|------|------|')

        for k, p in enumerate(sorted(a['gt_pairs'], key=lambda x: -x['pred_prob'])):
            status = '✅ TP' if any(tp['i'] == p['i'] and tp['j'] == p['j'] for tp in a['tp_pairs']) else '❌ FN'
            note = ''
            if p['pred_prob'] < 0.43:
                note = '⚠️ 低于阈值'
            elif p['pred_prob'] < 0.5:
                note = '边缘'
            lines.append(f'| {k+1} | ({p["i"]},{p["j"]}) | {p["distance"]} | '
                         f'{p["pred_prob"]:.4f} | {status} | {note} |')
        lines.append('')

        # Pred (FP) pairs
        if a['fp_pairs']:
            lines.append(f'**错误预测 (FP, 共 {a["fp"]} 对)：**')
            lines.append('')
            lines.append(f'| # | 位置 (i,j) | 距离 |i-j| | 预测概率 | 最近GT距离 |')
            lines.append(f'|---|------------|---------|---------|-----------|')

            for k, p in enumerate(sorted(a['fp_pairs'], key=lambda x: -x['pred_prob'])[:20]):
                # Find nearest GT pair
                nearest_gt_dist = float('inf')
                for gt in a['gt_pairs']:
                    d = abs(gt['i'] - p['i']) + abs(gt['j'] - p['j'])
                    nearest_gt_dist = min(nearest_gt_dist, d)
                nearest_str = f'{nearest_gt_dist}' if nearest_gt_dist < 100 else '远'
                lines.append(f'| {k+1} | ({p["i"]},{p["j"]}) | {p["distance"]} | '
                             f'{p["pred_prob"]:.4f} | {nearest_str} |')
            lines.append('')

        # FN pairs with shift analysis
        if a['fn_pairs']:
            lines.append(f'**被漏掉的 GT 配对 (FN, 共 {a["fn"]} 对) — 偏移分析：**')
            lines.append('')
            lines.append(f'| # | GT位置 (i,j) | 距离 | GT概率 | ±2内有pred? | 偏移位置 | 偏移概率 |')
            lines.append(f'|---|-------------|------|--------|------------|---------|---------|')

            for k, fn in enumerate(sorted(a['fn_pairs'], key=lambda x: -x['pred_prob'])[:20]):
                has_shift = fn['nearest_pred_shift'] is not None
                if has_shift:
                    ni, nj, di, dj = fn['nearest_pred_shift']
                    shift_str = f'({ni},{nj}) Δ({di:+d},{dj:+d})'
                    shift_prob = f'{fn["nearest_pred_prob"]:.4f}'
                else:
                    shift_str = '—'
                    shift_prob = '—'
                lines.append(f'| {k+1} | ({fn["i"]},{fn["j"]}) | {fn["distance"]} | '
                             f'{fn["pred_prob"]:.4f} | {"是" if has_shift else "否"} | '
                             f'{shift_str} | {shift_prob} |')
            lines.append('')

        lines.append('---')
        lines.append('')

    # ========== 3. Pattern Summary ==========
    lines.append('## 3. Bad Case 失败模式总结')
    lines.append('')

    # Classify bad cases by failure mode
    mode_completely_wrong = []  # GT概率全低，pred位置全错
    mode_threshold_issue = []   # GT概率在阈值附近
    mode_shifted = []           # 位置偏移导致
    mode_overpredict = []       # 过度预测

    for a in sorted_analyses:
        gt_prob_mean = a['score_stats']['gt_prob_mean']
        fn_shifted = sum(1 for fn in a['fn_pairs'] if fn['nearest_pred_shift'] is not None)

        if gt_prob_mean < 0.2:
            mode_completely_wrong.append(a)
        elif gt_prob_mean < 0.43:
            mode_threshold_issue.append(a)
        elif fn_shifted > len(a['fn_pairs']) * 0.5:
            mode_shifted.append(a)
        elif a['n_pred_pairs'] > a['n_gt_pairs'] * 1.5:
            mode_overpredict.append(a)
        else:
            mode_completely_wrong.append(a)

    lines.append('| 失败模式 | 数量 | 特征 |')
    lines.append('|----------|------|------|')
    lines.append(f'| 模型完全不认识结构 (GT概率<0.2) | {len([a for a in sorted_analyses if a["score_stats"]["gt_prob_mean"] < 0.2])} | '
                 f'模型对 GT 位置给出极低概率，说明 MARS 表征没有捕获这些结构 |')
    lines.append(f'| GT 概率在阈值附近 (0.2-0.43) | {len([a for a in sorted_analyses if 0.2 <= a["score_stats"]["gt_prob_mean"] < 0.43])} | '
                 f'模型有感知但不够自信 |')
    lines.append(f'| GT 概率≥阈值但位置偏移 | {len([a for a in sorted_analyses if a["score_stats"]["gt_prob_mean"] >= 0.43 and a["f1"] < 0.3])} | '
                 f'概率足够但选错了位置 |')
    lines.append('')

    # Probability histogram summary across all bad cases
    lines.append('### 3.1 Bad Cases vs 全体样本的概率对比')
    lines.append('')
    lines.append('（Bad cases 中 GT 配对的模型概率 vs 正常样本中 GT 配对的模型概率）')
    lines.append('')
    lines.append(f'- Bad cases GT 概率均值: {all_gt_probs.mean():.4f}')
    lines.append(f'- Bad cases GT 概率中位数: {np.median(all_gt_probs):.4f}')
    lines.append(f'- Bad cases 中 GT 概率 > 0.43 的比例: {(all_gt_probs > 0.43).mean():.1%}')
    lines.append(f'- Bad cases 中 GT 概率 > 0.5 的比例: {(all_gt_probs > 0.5).mean():.1%}')
    lines.append('')

    return '\n'.join(lines)


def main():
    print('=' * 60)
    print('  v10 Bad Case Deep Analysis')
    print('=' * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    device = torch.device(DEVICE)

    # Load model
    class Args: pass
    lm_args = Args()
    lm_args.pretrained_lm_dir = config['paths'].get('pretrained_lm_dir', str(ROOT / 'model'))
    lm_args.model_scale = config['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)

    model = build_model(config, extractor).to(device)
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    state_dict = ckpt.get('model', ckpt.get('model_state_dict', ckpt))
    new_sd = {k.replace('module.', ''): v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    print(f'Model loaded (epoch {ckpt.get("epoch", "?")})')

    # Build test loader
    data_dir = config['paths']['data_dir']
    max_len = config['training'].get('max_len_filter', 490)
    records = build_records(data_dir, 'bprna-test', max_len=max_len)
    dataset = PriFoldSymFlowDataset(records, augment=False)
    collate_fn = make_collate_fn(tokenizer)

    # First pass: identify bad cases
    print('\n[Pass 1] Identifying bad cases...')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                        num_workers=4, pin_memory=True)

    bad_indices = []
    scfg = config.get('sampling', {})
    amp_dtype = torch.bfloat16

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch_dev = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                         for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                pred, score = model.predict(
                    batch_dev,
                    budget_fraction=scfg.get('default_budget_fraction', 0.30),
                    use_density_budget=scfg.get('use_density_budget', True),
                    score_threshold=scfg.get('score_threshold', 0.43),
                    length_decay=scfg.get('length_decay', 0.15),
                    budget_floor=scfg.get('budget_floor', 0.6),
                )
            length = int(batch['length'][0])
            p = pred[0, 0, :length, :length].cpu() > 0.5
            y = batch['contact'][0].squeeze()[:length, :length].cpu() > 0.5
            idx_arr = torch.arange(length)
            mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
            mask &= (idx_arr.view(length, 1) - idx_arr.view(1, length)).abs() >= 3
            tp = int((p[mask] & y[mask]).sum())
            fp = int((p[mask] & ~y[mask]).sum())
            fn = int((~p[mask] & y[mask]).sum())
            gt_pairs = tp + fn
            pred_pairs = tp + fp
            if gt_pairs == 0 and pred_pairs == 0:
                f1 = 1.0
            elif gt_pairs == 0:
                f1 = 0.0
            else:
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-12)

            if f1 < 0.3:
                bad_indices.append(idx)

            if (idx + 1) % 200 == 0:
                print(f'  [{idx+1}/{len(loader)}] bad so far: {len(bad_indices)}')

    print(f'\n  Found {len(bad_indices)} bad cases (F1 < 0.3)')

    # Second pass: detailed analysis of bad cases
    print(f'\n[Pass 2] Deep analysis of {len(bad_indices)} bad cases...')
    analyses = []

    for count, bad_idx in enumerate(bad_indices):
        if (count + 1) % 20 == 0:
            print(f'  Analyzing {count+1}/{len(bad_indices)}...')

        # Get the specific sample
        batch = collate_fn([dataset[bad_idx]])
        result = analyze_single_sample(model, batch, device, config)
        analyses.append(result)

    # Generate report
    print('\n[Step 3] Generating report...')
    report = generate_badcase_report(analyses)

    report_path = OUTPUT_DIR / 'v10_badcase_deep_analysis.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'Report saved: {report_path}')

    # Save raw analysis data (without huge lists for serialization)
    data_path = OUTPUT_DIR / 'badcase_analysis_data.json'
    slim_analyses = []
    for a in analyses:
        slim = {
            'name': a['name'],
            'family': a['family'],
            'length': a['length'],
            'f1': a['f1'],
            'precision': a['precision'],
            'recall': a['recall'],
            'n_gt_pairs': a['n_gt_pairs'],
            'n_pred_pairs': a['n_pred_pairs'],
            'tp': a['tp'], 'fp': a['fp'], 'fn': a['fn'],
            'score_stats': a['score_stats'],
            'gt_pairs': a['gt_pairs'][:50],  # limit for file size
            'fp_pairs': a['fp_pairs'][:50],
            'fn_pairs': [{'i': fn['i'], 'j': fn['j'], 'distance': fn['distance'],
                          'pred_prob': fn['pred_prob'],
                          'has_shift': fn['nearest_pred_shift'] is not None}
                         for fn in a['fn_pairs'][:50]],
        }
        slim_analyses.append(slim)

    with open(data_path, 'w') as f:
        json.dump(slim_analyses, f, indent=2)
    print(f'Data saved: {data_path}')

    # Quick summary
    all_gt_probs = np.array([p['pred_prob'] for a in analyses for p in a['gt_pairs']])
    print(f'\n{"="*60}')
    print(f'  SUMMARY')
    print(f'{"="*60}')
    print(f'  Bad cases: {len(analyses)}')
    print(f'  GT pair avg prob: {all_gt_probs.mean():.4f} (should be >0.5)')
    print(f'  GT pairs above threshold (0.43): {(all_gt_probs > 0.43).mean():.1%}')
    print(f'  Report: {report_path}')


if __name__ == '__main__':
    main()
