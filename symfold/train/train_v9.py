# -*- coding: utf-8 -*-
"""PriFold v9 training: DensityNet-Pro+ (single GPU, H20 optimized).

Efficiency improvements over v8:
  - torch.compile for ~20-40% speedup on forward/backward
  - Gradient accumulation for effective larger batch
  - Increased max_sq_tokens to fill H20 97GB VRAM
  - Optimized DataLoader (pin_memory, prefetch, multiple workers)
  - Vectorized OHEM in loss (no Python for-loop)

Usage:
  bash symfold/train/run_train.sh symfold/config/v9/v9_single.json
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
import torch
import torch.nn.utils

from symfold.train import train_v3 as _base
from symfold.v9.model import DensityNetProPlus

load_config = _base.load_config
move_to_device = _base.move_to_device


def build_model(cfg: dict, extractor) -> DensityNetProPlus:
    """Build v9 DensityNet-Pro+ from config."""
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
        # Loss
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[v9 DensityNet-Pro+] Trainable params: {n_params/1e6:.2f}M')
    print(f'[v9] {v9cfg.get("num_layers", 8)} layers, dim={v9cfg.get("hidden_dim", 192)}, '
          f'heads={v9cfg.get("num_heads", 6)}, '
          f'dropout={v9cfg.get("dropout", 0.15)}, drop_path={v9cfg.get("drop_path", 0.1)}')
    print(f'[v9] OHEM={lcfg.get("ohem_enabled", True)}, '
          f'FP_penalty={lcfg.get("fp_penalty_weight", 2.0)}, '
          f'BP_compat={lcfg.get("bp_compat_enabled", True)}, '
          f'Shift(r={lcfg.get("shift_radius", 2)},w={lcfg.get("shift_loss_weight", 0.6)})')
    return model


def evaluate(model, loader, device, config, logger=None, stage='val'):
    """Evaluate v9 DensityNet-Pro+."""
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
                        score_threshold=scfg.get('score_threshold', 0.43),
                        length_decay=scfg.get('length_decay', 0.15),
                        budget_floor=scfg.get('budget_floor', 0.6),
                    )
            else:
                pred, _ = model.predict(
                    batch,
                    budget_fraction=scfg.get('default_budget_fraction', 0.30),
                    use_density_budget=scfg.get('use_density_budget', True),
                    score_threshold=scfg.get('score_threshold', 0.43),
                    length_decay=scfg.get('length_decay', 0.15),
                    budget_floor=scfg.get('budget_floor', 0.6),
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


def train_one_epoch_v9(model, loader, optimizer, device, config, logger, epoch,
                       heartbeat_path, compiled_model=None):
    """v9 training loop with gradient accumulation and torch.compile support."""
    import os
    model.train()
    totals = {'loss': 0.0, 'bce': 0.0, 'density': 0.0, 'fp_penalty': 0.0, 'shift': 0.0}
    n = 0
    t0 = time.time()
    
    amp_on = str(config.get('training', {}).get('amp_dtype', 'fp32')).lower() in ('bf16', 'bfloat16', 'fp16', 'float16')
    amp_dtype = torch.bfloat16
    
    grad_accum = config['training'].get('gradient_accumulation_steps', 1)
    grad_clip = config['training'].get('grad_clip', 1.0)
    log_every = config['training'].get('log_every', 20)
    heartbeat_every = config['training'].get('heartbeat_every', 20)
    
    forward_model = compiled_model if compiled_model is not None else model
    
    optimizer.zero_grad(set_to_none=True)
    
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        
        if amp_on:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss, loss_dict = forward_model(batch)
        else:
            loss, loss_dict = forward_model(batch)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f'[Train] e{epoch} step={step} got NaN/Inf, skip')
            optimizer.zero_grad(set_to_none=True)
            continue
        
        # Scale loss by accumulation steps
        scaled_loss = loss / grad_accum
        scaled_loss.backward()
        
        # Step optimizer every grad_accum steps
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        n += 1
        totals['loss'] += float(loss.item())
        for k in ('bce', 'density', 'fp_penalty', 'shift'):
            totals[k] += float(loss_dict.get(k, torch.tensor(0.0)).item())
        
        if step % log_every == 0:
            logger.info(
                f"[Train] e{epoch} step={step}/{len(loader)} L={batch['set_max_len']} "
                f"loss={loss.item():.5f} bce={float(loss_dict['bce']):.4f} "
                f"fp={float(loss_dict.get('fp_penalty', 0)):.4f} "
                f"shift={float(loss_dict.get('shift', 0)):.4f}")
        
        if step % heartbeat_every == 0:
            _base.write_heartbeat(heartbeat_path, {
                'time': time.asctime(),
                'epoch': epoch, 'step': step,
                'loss': float(loss.item()),
                'gpu_mb': torch.cuda.memory_allocated(device) / 1024 / 1024 if device.type == 'cuda' else 0,
                'gpu_max_mb': torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == 'cuda' else 0,
                'pid': os.getpid(),
            })

    avg = {k: v / max(n, 1) for k, v in totals.items()}
    avg['time_s'] = time.time() - t0
    avg['steps_per_sec'] = n / avg['time_s']
    logger.info(f'[Train] e{epoch} done {avg}')
    return avg


# Monkey-patch to override base train_one_epoch
_original_train_one_epoch = _base.train_one_epoch
_compiled_model_cache = [None]


def _patched_train_one_epoch(model, loader, optimizer, device, config, logger, epoch, heartbeat_path):
    """Wrapper that uses v9 training loop with gradient accumulation."""
    compiled = _compiled_model_cache[0]
    return train_one_epoch_v9(model, loader, optimizer, device, config, logger, epoch,
                              heartbeat_path, compiled_model=compiled)


# Override base functions
_base.build_model = build_model
_base.evaluate = evaluate
_base.train_one_epoch = _patched_train_one_epoch

# Custom main to add torch.compile
_original_main = _base.main


def main():
    """v9 main with torch.compile and efficiency optimizations."""
    import argparse
    import json
    import logging
    import numpy as np
    from pathlib import Path
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    config = _base.load_config(args.config)
    
    # Set num_workers and pin_memory in data loading
    tcfg = config['training']
    
    # Enable cudnn benchmark for fixed-size inputs within length buckets
    torch.backends.cudnn.benchmark = True
    
    # Pre-set environment for efficiency
    import os
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    
    # Call original main (which uses our patched build_model/evaluate/train_one_epoch)
    # But first we need to hook into model creation for torch.compile
    original_build = build_model
    
    def build_and_compile(cfg, extractor):
        model = original_build(cfg, extractor)
        if cfg['training'].get('compile_model', False):
            compile_mode = cfg['training'].get('compile_mode', 'reduce-overhead')
            print(f'[v9] Applying torch.compile(mode="{compile_mode}")...')
            try:
                compiled = torch.compile(model, mode=compile_mode)
                _compiled_model_cache[0] = compiled
                print('[v9] torch.compile successful!')
            except Exception as e:
                print(f'[v9] torch.compile failed: {e}, using eager mode')
                _compiled_model_cache[0] = None
        return model
    
    _base.build_model = build_and_compile
    _base.main()


if __name__ == '__main__':
    main()
