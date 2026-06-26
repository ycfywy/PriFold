# -*- coding: utf-8 -*-
"""PriFold v7 training: DensityNet (pure discriminative).

Fast training — no flow sampling, single forward pass per batch.

Usage:
  bash symfold/train/run_train.sh symfold/config/v7/v7_full.json
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symfold.train import train_v3 as _base
from symfold.v7.model import DensityNet

load_config = _base.load_config
move_to_device = _base.move_to_device


def build_model(cfg: dict, extractor) -> DensityNet:
    """Build v7 DensityNet from config."""
    mcfg = cfg['model']
    v7cfg = cfg.get('v7', {})
    lcfg = cfg.get('loss', {})

    model = DensityNet(
        extractor=extractor,
        freeze_mars=mcfg.get('freeze_mars', True),
        mars_dim=mcfg.get('mars_dim', 1056),
        mars_n_attn_layers=mcfg.get('mars_n_attn_layers', 6),
        mars_n_heads=mcfg.get('mars_n_heads', 12),
        mars_hidden_layer_indices=mcfg.get('mars_hidden_layer_indices', [3, 6, 9, 12]),
        hidden_dim=v7cfg.get('hidden_dim', 128),
        num_layers=v7cfg.get('num_layers', 8),
        num_heads=v7cfg.get('num_heads', 4),
        dim_head=v7cfg.get('dim_head', 32),
        ff_mult=v7cfg.get('ff_mult', 4),
        dropout=mcfg.get('dropout', 0.1),
        # Density-aware loss config
        density_loss_weight=v7cfg.get('density_loss_weight', 0.3),
        dst_low_threshold=v7cfg.get('dst_low_threshold', 0.18),
        dst_tversky_alpha=v7cfg.get('dst_tversky_alpha', 0.7),
        dst_tversky_beta=v7cfg.get('dst_tversky_beta', 0.3),
        dst_weight=v7cfg.get('dst_weight', 0.4),
        # Standard loss config
        focal_gamma=lcfg.get('focal_gamma', 1.0),
        pos_weight_base=lcfg.get('pos_weight_base', 99.0),
        dice_weight=lcfg.get('dice_weight', 0.5),
        pair_count_weight=lcfg.get('pair_count_weight', 0.3),
        ratio_penalty_weight=lcfg.get('ratio_penalty_weight', 0.2),
        ratio_penalty_threshold=lcfg.get('ratio_penalty_threshold', 1.2),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[v7 DensityNet] Trainable params: {n_params/1e6:.2f}M')
    print(f'[v7 DensityNet] {v7cfg.get("num_layers", 8)} axial layers, '
          f'dim={v7cfg.get("hidden_dim", 128)}, heads={v7cfg.get("num_heads", 4)}')
    print(f'[v7 DensityNet] DST: alpha={v7cfg.get("dst_tversky_alpha", 0.7)}, '
          f'threshold={v7cfg.get("dst_low_threshold", 0.18)}')
    return model


def evaluate(model, loader, device, config, logger=None, stage='val'):
    """Evaluate v7 DensityNet — single forward pass, fast."""
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
                        score_threshold=scfg.get('score_threshold', 0.4),
                    )
            else:
                pred, _ = model.predict(
                    batch,
                    budget_fraction=scfg.get('default_budget_fraction', 0.30),
                    use_density_budget=scfg.get('use_density_budget', True),
                    score_threshold=scfg.get('score_threshold', 0.4),
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
