# -*- coding: utf-8 -*-
"""Improved inference-time evaluation for PriFold-SymFlow v6.

Implements three inference-time optimizations that don't require retraining:
  1. Density-conditional budget scaling
  2. Multi-sample voting (N=3 or N=5)
  3. Adaptive score threshold based on density

Usage:
  python symfold/eval_v6_improved.py \
    --ckpt symfold/outputs/v6_full/model/best.pt \
    --config symfold/config/v6_full.json \
    --test_sets bprna-test \
    --strategy all

Strategies:
  baseline     - Original v6 settings
  density_cond - Density-conditional budget scaling
  multisample  - Multi-sample voting (N=3)
  adaptive_thr - Adaptive score threshold
  combined     - All three combined
  all          - Run all strategies and compare
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_loader
from symfold.train.train_v6 import build_model, load_config, move_to_device
from symfold.v6.discrete_flow import (
    symmetrize_binary,
    symmetrize_logit,
    compute_ctmc_rates,
    project_score_to_valid_contact_map,
    density_to_max_pairs,
)
from symfold.metrics import contact_metrics


def density_conditional_budget_scale(density_pred: torch.Tensor) -> torch.Tensor:
    """Compute per-sample budget scale based on predicted density.
    
    Logic (v2 - less aggressive than v1):
    - density < 0.10: scale = 0.80 (conservative, avoid over-predict)
    - density 0.10-0.18: scale = 0.95
    - density 0.18-0.25: scale = 1.05
    - density 0.25-0.35: scale = 1.12 (allow slightly more)
    - density >= 0.35: scale = 1.15
    
    This addresses the core finding: low-density RNAs are over-predicted by 2x.
    """
    # density_pred: [B, 1]
    d = density_pred.squeeze(-1)  # [B]
    scale = torch.ones_like(d)
    scale = torch.where(d < 0.10, torch.full_like(d, 0.80), scale)
    scale = torch.where((d >= 0.10) & (d < 0.18), torch.full_like(d, 0.95), scale)
    scale = torch.where((d >= 0.18) & (d < 0.25), torch.full_like(d, 1.05), scale)
    scale = torch.where((d >= 0.25) & (d < 0.35), torch.full_like(d, 1.12), scale)
    scale = torch.where(d >= 0.35, torch.full_like(d, 1.15), scale)
    return scale  # [B]


def adaptive_score_threshold(density_pred: torch.Tensor) -> torch.Tensor:
    """Compute per-sample score threshold based on predicted density.
    
    Logic:
    - Low density: higher threshold (be more strict, reduce FP)
    - High density: lower threshold (allow more predictions)
    """
    d = density_pred.squeeze(-1)  # [B]
    # Linear interpolation: threshold = 0.65 - 0.5 * density
    # density=0.05 -> thr=0.625, density=0.20 -> thr=0.55, density=0.30 -> thr=0.50
    threshold = (0.65 - 0.5 * d).clamp(min=0.40, max=0.70)
    return threshold  # [B]


@torch.no_grad()
def sample_improved(model, batch: dict,
                    num_steps: int = 20,
                    num_samples_per_input: int = 1,
                    use_density_conditional_budget: bool = False,
                    use_adaptive_threshold: bool = False,
                    base_budget_scale: float = 1.1,
                    base_score_threshold: float = 0.5,
                    default_budget_fraction: float = 0.30,
                    direct_score_weight: float | None = None):
    """Improved sampling with density-conditional strategies."""
    contact_mask = batch['contact_mask']
    pos_bias = batch['pos_bias']
    device = contact_mask.device
    b_real, _, l, _ = contact_mask.shape
    direct_w = model.direct_score_weight if direct_score_weight is None else float(direct_score_weight)

    mars_hidden, mars_attn, mars_hidden_layers = model._extract_mars(
        batch['input_ids'], batch['attention_mask'], l)
    seq_oh = batch.get('seq_oh')

    if num_samples_per_input > 1:
        mars_hidden = mars_hidden.repeat(num_samples_per_input, 1, 1)
        mars_hidden_layers = [h.repeat(num_samples_per_input, 1, 1) for h in mars_hidden_layers]
        mars_attn = mars_attn.repeat(num_samples_per_input, 1, 1, 1, 1)
        pos_bias = pos_bias.repeat(num_samples_per_input, 1, 1)
        contact_mask = contact_mask.repeat(num_samples_per_input, 1, 1, 1)
        if seq_oh is not None:
            seq_oh = seq_oh.repeat(num_samples_per_input, 1, 1)
    b = b_real * num_samples_per_input

    # Initial state
    x_init = (torch.rand(b, 1, l, l, device=device) < model.rho_0).float()
    x_init = symmetrize_binary(x_init) * contact_mask

    # Get density prediction
    density_pred = None
    if model.use_density_head:
        t_half = torch.full((b,), 0.5, device=device)
        _, density_pred, _ = model.backbone(
            x_init, t_half,
            mars_hidden=mars_hidden, mars_hidden_layers=mars_hidden_layers,
            mars_attn=mars_attn, pos_bias=pos_bias, seq_oh=seq_oh,
            contact_masks=contact_mask, density_hint=None,
            return_density=True, return_direct=True)

    # Tau-leap sampling (same as original)
    raw = [math.sin(math.pi * (k + 0.5) / (2 * num_steps)) for k in range(num_steps)]
    total_raw = sum(raw)
    dt_list = [r / total_raw for r in raw]
    x_t = x_init
    score_last = torch.zeros_like(x_t)
    t_cum = 0.0

    for dt in dt_list:
        t_tensor = torch.full((b,), t_cum, device=device)
        backbone_out = model.backbone(
            x_t, t_tensor,
            mars_hidden=mars_hidden, mars_hidden_layers=mars_hidden_layers,
            mars_attn=mars_attn, pos_bias=pos_bias, seq_oh=seq_oh,
            contact_masks=contact_mask,
            density_hint=None,
            return_direct=True)

        if model.use_direct_head:
            flow_logit, direct_logit = backbone_out
        else:
            flow_logit = backbone_out if not isinstance(backbone_out, tuple) else backbone_out[0]
            direct_logit = flow_logit

        flow_logit = symmetrize_logit(flow_logit)
        direct_logit = symmetrize_logit(direct_logit)
        p_flow = torch.sigmoid(flow_logit)
        p_direct = torch.sigmoid(direct_logit)
        score_last = (1.0 - direct_w) * p_flow + direct_w * p_direct

        rate_01, rate_10 = compute_ctmc_rates(x_t, p_flow, t_tensor, rho_0=model.rho_0)
        f01 = torch.clamp(rate_01 * dt, max=1.0)
        f10 = torch.clamp(rate_10 * dt, max=1.0)
        flip01 = (torch.rand_like(f01) < f01) & (x_t < 0.5)
        flip10 = (torch.rand_like(f10) < f10) & (x_t > 0.5)
        x_t = torch.where(flip01, torch.ones_like(x_t), x_t)
        x_t = torch.where(flip10, torch.zeros_like(x_t), x_t)
        x_t = symmetrize_binary(x_t) * contact_mask
        t_cum += dt

    # Aggregate multi-sample
    if num_samples_per_input > 1:
        score_final = score_last.view(num_samples_per_input, b_real, 1, l, l).mean(0)
        mask_final = contact_mask.view(num_samples_per_input, b_real, 1, l, l)[0]
        if density_pred is not None:
            density_final = density_pred.view(num_samples_per_input, b_real, 1).mean(0)
        else:
            density_final = None
    else:
        score_final = score_last
        mask_final = contact_mask
        density_final = density_pred

    # === Improved budget computation ===
    if density_final is not None and (use_density_conditional_budget or use_adaptive_threshold):
        if use_density_conditional_budget:
            # Per-sample density-conditional scale
            per_sample_scale = density_conditional_budget_scale(density_final)  # [B]
            # Apply to density-based budget
            valid = mask_final.squeeze(1)
            length_eff = valid[:, 0, :].sum(dim=-1)  # [B]
            # budget = density * length * scale
            d_val = density_final.squeeze(-1)  # [B]
            max_pairs = torch.round(d_val * length_eff * per_sample_scale).long()
            # Clamp to reasonable range
            max_budget = torch.round(length_eff * 0.40).long()
            min_budget = torch.ones_like(max_pairs) * 2
            max_pairs = torch.clamp(max_pairs, min=min_budget, max=max_budget)
        else:
            # Standard density budget
            max_pairs = density_to_max_pairs(density_final, mask_final, scale=base_budget_scale)

        if use_adaptive_threshold:
            # Per-sample threshold
            per_sample_thr = adaptive_score_threshold(density_final)  # [B]
        else:
            per_sample_thr = torch.full((b_real,), base_score_threshold, device=device)
    else:
        # Fallback: fixed budget fraction
        valid = mask_final.squeeze(1)
        length_eff = valid[:, 0, :].sum(dim=-1)
        max_pairs = torch.round(length_eff * float(default_budget_fraction)).long()
        per_sample_thr = torch.full((b_real,), base_score_threshold, device=device)

    # === Per-sample projection with adaptive threshold ===
    # We need per-sample projection because thresholds differ
    x_final_list = []
    for i in range(b_real):
        x_i = project_score_to_valid_contact_map(
            score_final[i:i+1], mask_final[i:i+1],
            max_pairs=max_pairs[i:i+1],
            min_score=float(per_sample_thr[i]))
        x_final_list.append(x_i)
    x_final = torch.cat(x_final_list, dim=0)

    return x_final, score_final


def evaluate_strategy(model, loader, device, amp_on, amp_dtype,
                      strategy: str, cfg: dict) -> dict:
    """Evaluate a specific strategy."""
    scfg = cfg.get('sampling', {})
    
    # Strategy configs
    strategies = {
        'baseline': {
            'num_samples_per_input': 1,
            'use_density_conditional_budget': False,
            'use_adaptive_threshold': False,
            'base_budget_scale': scfg.get('budget_scale', 1.1),
            'base_score_threshold': scfg.get('score_threshold', 0.5),
        },
        'density_cond': {
            'num_samples_per_input': 1,
            'use_density_conditional_budget': True,
            'use_adaptive_threshold': False,
            'base_budget_scale': 1.0,
            'base_score_threshold': 0.5,
        },
        'multisample': {
            'num_samples_per_input': 3,
            'use_density_conditional_budget': False,
            'use_adaptive_threshold': False,
            'base_budget_scale': scfg.get('budget_scale', 1.1),
            'base_score_threshold': 0.5,
        },
        'adaptive_thr': {
            'num_samples_per_input': 1,
            'use_density_conditional_budget': False,
            'use_adaptive_threshold': True,
            'base_budget_scale': scfg.get('budget_scale', 1.1),
            'base_score_threshold': 0.5,
        },
        'combined': {
            'num_samples_per_input': 3,
            'use_density_conditional_budget': True,
            'use_adaptive_threshold': True,
            'base_budget_scale': 1.0,
            'base_score_threshold': 0.5,
        },
        'multisample5': {
            'num_samples_per_input': 5,
            'use_density_conditional_budget': True,
            'use_adaptive_threshold': True,
            'base_budget_scale': 1.0,
            'base_score_threshold': 0.5,
        },
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
    
    s = strategies[strategy]
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}")
    print(f"  num_samples={s['num_samples_per_input']}, "
          f"density_cond_budget={s['use_density_conditional_budget']}, "
          f"adaptive_thr={s['use_adaptive_threshold']}")
    print(f"{'='*60}")

    all_tp, all_fp, all_fn = 0, 0, 0
    all_gt_pairs, all_pred_pairs = 0.0, 0.0
    n_samples = 0
    t0 = time.time()

    # Per-sample tracking for breakdown
    per_sample = []

    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        
        kwargs = dict(
            num_steps=scfg.get('num_steps', 20),
            num_samples_per_input=s['num_samples_per_input'],
            use_density_conditional_budget=s['use_density_conditional_budget'],
            use_adaptive_threshold=s['use_adaptive_threshold'],
            base_budget_scale=s['base_budget_scale'],
            base_score_threshold=s['base_score_threshold'],
            default_budget_fraction=scfg.get('default_budget_fraction', 0.30),
            direct_score_weight=scfg.get('direct_score_weight', None),
        )
        
        if amp_on:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                pred, score = sample_improved(model, batch, **kwargs)
        else:
            pred, score = sample_improved(model, batch, **kwargs)

        # Compute metrics per sample
        for i in range(pred.shape[0]):
            length = int(batch['length'][i].item())
            p = pred[i].cpu().float().squeeze()[:length, :length] > 0.5
            y = batch['contact'][i].cpu().float().squeeze()[:length, :length] > 0.5
            idx = torch.arange(length)
            mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
            mask &= (idx.view(length, 1) - idx.view(1, length)).abs() >= 3
            p_m = p[mask]
            y_m = y[mask]
            tp = int((p_m & y_m).sum())
            fp = int((p_m & ~y_m).sum())
            fn = int((~p_m & y_m).sum())
            all_tp += tp
            all_fp += fp
            all_fn += fn
            gt_p = tp + fn
            pred_p = tp + fp
            all_gt_pairs += gt_p
            all_pred_pairs += pred_p
            n_samples += 1

            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            density = gt_p / max(length, 1)
            per_sample.append({
                'name': batch['names'][i],
                'length': length,
                'density': density,
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'pred_gt_ratio': pred_p / max(gt_p, 1),
            })

        if step % 30 == 0:
            elapsed = time.time() - t0
            print(f"  step={step}/{len(loader)}, samples={n_samples}, time={elapsed:.0f}s")

    elapsed = time.time() - t0
    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    pred_gt_ratio = all_pred_pairs / max(all_gt_pairs, 1e-12)

    # Breakdown by density bin
    density_bins = {
        '<0.10': [], '0.10-0.18': [], '0.18-0.25': [], '0.25-0.35': [], '>=0.35': []
    }
    for s_item in per_sample:
        d = s_item['density']
        if d < 0.10:
            density_bins['<0.10'].append(s_item)
        elif d < 0.18:
            density_bins['0.10-0.18'].append(s_item)
        elif d < 0.25:
            density_bins['0.18-0.25'].append(s_item)
        elif d < 0.35:
            density_bins['0.25-0.35'].append(s_item)
        else:
            density_bins['>=0.35'].append(s_item)

    f1_zero_count = sum(1 for s_item in per_sample if s_item['f1'] == 0)
    f1_below_03 = sum(1 for s_item in per_sample if s_item['f1'] < 0.3)

    result = {
        'strategy': strategy,
        'n': n_samples,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pred_gt_ratio': pred_gt_ratio,
        'f1_zero': f1_zero_count,
        'f1_below_03': f1_below_03,
        'time_s': elapsed,
        'by_density': {},
    }

    print(f"\n  Results: F1={f1:.4f} P={precision:.4f} R={recall:.4f} pred/gt={pred_gt_ratio:.3f}")
    print(f"  F1=0: {f1_zero_count}, F1<0.3: {f1_below_03}, time={elapsed:.0f}s")
    print(f"  By density:")
    for db, items in density_bins.items():
        if items:
            db_f1 = np.mean([x['f1'] for x in items])
            db_pg = np.mean([x['pred_gt_ratio'] for x in items])
            result['by_density'][db] = {
                'n': len(items),
                'f1': float(db_f1),
                'pred_gt_ratio': float(db_pg),
            }
            print(f"    {db:10s}: n={len(items):3d} F1={db_f1:.4f} pred/gt={db_pg:.3f}")

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--test_sets', default='bprna-test')
    ap.add_argument('--strategy', default='all',
                    help='baseline|density_cond|multisample|adaptive_thr|combined|multisample5|all')
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = load_config(args.config) if args.config else ckpt['config']

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

    if args.strategy == 'all':
        strategy_list = ['baseline', 'density_cond', 'adaptive_thr', 'multisample', 'combined']
    else:
        strategy_list = [args.strategy]

    all_results = {}
    for stage in stages:
        print(f"\n{'#'*60}")
        print(f"# Evaluating: {stage}")
        print(f"{'#'*60}")
        
        stage_results = {}
        for strat in strategy_list:
            loader = build_loader(stage, cfg, tokenizer, shuffle=False)
            result = evaluate_strategy(model, loader, device, amp_on, amp_dtype, strat, cfg)
            stage_results[strat] = result
        
        all_results[stage] = stage_results

    # Print comparison table
    print(f"\n\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}")
    for stage, stage_results in all_results.items():
        print(f"\n{stage}:")
        print(f"  {'Strategy':<15s} {'F1':>6s} {'P':>6s} {'R':>6s} {'p/g':>5s} {'F1=0':>5s} {'F1<.3':>5s} {'Time':>5s}")
        print(f"  {'-'*55}")
        for strat, r in stage_results.items():
            print(f"  {strat:<15s} {r['f1']:.4f} {r['precision']:.4f} {r['recall']:.4f} "
                  f"{r['pred_gt_ratio']:.3f} {r['f1_zero']:>5d} {r['f1_below_03']:>5d} {r['time_s']:>4.0f}s")

    # Save results
    out_path = args.out_json or str(Path(args.ckpt).parent.parent / 'eval_improved_results.json')
    
    # Convert for JSON
    def to_native(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(x) for x in obj]
        return obj

    with open(out_path, 'w') as f:
        json.dump(to_native(all_results), f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
