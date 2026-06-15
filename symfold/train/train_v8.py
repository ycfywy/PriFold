# -*- coding: utf-8 -*-
"""PriFold v8 training: DensityNet-Pro (precision-focused discriminative).

Usage:
  bash symfold/train/run_train.sh symfold/config/v8/v8_full.json
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symfold.train import train_v3 as _base
from symfold.v8.model import DensityNetPro

load_config = _base.load_config
move_to_device = _base.move_to_device


def build_model(cfg: dict, extractor) -> DensityNetPro:
    """Build v8 DensityNet-Pro from config."""
    mcfg = cfg['model']
    v8cfg = cfg.get('v8', {})
    lcfg = cfg.get('loss', {})

    model = DensityNetPro(
        extractor=extractor,
        freeze_mars=mcfg.get('freeze_mars', True),
        mars_dim=mcfg.get('mars_dim', 1056),
        mars_n_attn_layers=mcfg.get('mars_n_attn_layers', 6),
        mars_n_heads=mcfg.get('mars_n_heads', 12),
        mars_hidden_layer_indices=mcfg.get('mars_hidden_layer_indices', [3, 6, 9, 12]),
        hidden_dim=v8cfg.get('hidden_dim', 160),
        num_layers=v8cfg.get('num_layers', 8),
        num_heads=v8cfg.get('num_heads', 4),
        dim_head=v8cfg.get('dim_head', 40),
        ff_mult=v8cfg.get('ff_mult', 4),
        dropout=v8cfg.get('dropout', 0.2),
        drop_path=v8cfg.get('drop_path', 0.1),
        # Loss
        focal_gamma=lcfg.get('focal_gamma', 1.0),
        pos_weight_base=lcfg.get('pos_weight_base', 99.0),
        dice_weight=lcfg.get('dice_weight', 0.5),
        dst_weight=lcfg.get('dst_weight', 0.4),
        dst_low_threshold=lcfg.get('dst_low_threshold', 0.10),
        dst_tversky_alpha=lcfg.get('dst_tversky_alpha', 0.7),
        dst_tversky_beta=lcfg.get('dst_tversky_beta', 0.3),
        pair_count_weight=lcfg.get('pair_count_weight', 0.3),
        ratio_penalty_weight=lcfg.get('ratio_penalty_weight', 0.2),
        ratio_penalty_threshold=lcfg.get('ratio_penalty_threshold', 1.15),
        density_loss_weight=lcfg.get('density_loss_weight', 0.3),
        ohem_enabled=lcfg.get('ohem_enabled', True),
        ohem_neg_ratio=lcfg.get('ohem_neg_ratio', 3),
        fp_penalty_enabled=lcfg.get('fp_penalty_enabled', True),
        fp_penalty_weight=lcfg.get('fp_penalty_weight', 3.0),
        bp_compat_enabled=lcfg.get('bp_compat_enabled', True),
        bp_compat_weight=lcfg.get('bp_compat_weight', 0.5),
        bp_compat_in_inference=lcfg.get('bp_compat_in_inference', True),
        shift_loss_enabled=lcfg.get('shift_loss_enabled', True),
        shift_loss_weight=lcfg.get('shift_loss_weight', 0.3),
        shift_radius=lcfg.get('shift_radius', 1),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[v8 DensityNet-Pro] Trainable params: {n_params/1e6:.2f}M')
    print(f'[v8] {v8cfg.get("num_layers", 8)} layers, dim={v8cfg.get("hidden_dim", 160)}, '
          f'dropout={v8cfg.get("dropout", 0.2)}, drop_path={v8cfg.get("drop_path", 0.1)}')
    print(f'[v8] OHEM={lcfg.get("ohem_enabled", True)}, FP_penalty={lcfg.get("fp_penalty_enabled", True)}, '
          f'BP_compat={lcfg.get("bp_compat_enabled", True)}, Shift_loss={lcfg.get("shift_loss_enabled", True)}')
    return model


def evaluate(model, loader, device, config, logger=None, stage='val'):
    """Evaluate v8 DensityNet-Pro."""
    import torch
    from symfold.metrics import contact_metrics

    model.eval()
    scfg = config.get('sampling', {})
    amp_name = str(config.get('training', {}).get('amp_dtype', 'fp32')).lower()
    amp_on = amp_name in ('bf16', 'bfloat16', 'fp16', 'float16')
    amp_dtype = torch.bfloat16 if amp_name in ('bf16', 'bfloat16') else torch.float16

    results = {'precision': 0, 'recall': 0, 'f1': 0, 'mcc': 0,
               'gt_pairs': 0, 'pred_pairs': 0, 'n': 0}

    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            if amp_on:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    pred, _ = model.predict(
                        batch,
                        budget_fraction=scfg.get('default_budget_fraction', 0.30),
                        use_density_budget=scfg.get('use_density_budget', True),
                        score_threshold=scfg.get('score_threshold', 0.45),
                        length_decay=scfg.get('length_decay', 0.3),
                    )
            else:
                pred, _ = model.predict(
                    batch,
                    budget_fraction=scfg.get('default_budget_fraction', 0.30),
                    use_density_budget=scfg.get('use_density_budget', True),
                    score_threshold=scfg.get('score_threshold', 0.45),
                    length_decay=scfg.get('length_decay', 0.3),
                )

            m = contact_metrics(pred, batch['contact'], batch['length'])
            bs = pred.shape[0]
            results['precision'] += m['precision'] * bs
            results['recall'] += m['recall'] * bs
            results['f1'] += m['f1'] * bs
            results['mcc'] += m['mcc'] * bs
            results['gt_pairs'] += m['gt_pairs'] * bs
            results['pred_pairs'] += m['pred_pairs'] * bs
            results['n'] += bs

    n = results['n']
    if n > 0:
        for k in ['precision', 'recall', 'f1', 'mcc', 'gt_pairs', 'pred_pairs']:
            results[k] /= n
    model.train()
    return results


_base.build_model = build_model
_base.evaluate = evaluate

if __name__ == '__main__':
    _base.main()
