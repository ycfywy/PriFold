# -*- coding: utf-8 -*-
"""PriFold-SymFlow v6 training entry.

Reuses the v3 training loop with v6 modular-loss model builder.
Supports ablation studies via config['loss'] dict.
"""
from __future__ import annotations

import time

import torch

from symfold import train_v3 as _base
from symfold.metrics import contact_metrics
from symfold.v6.model import PriFoldSymFlow_v6

load_config = _base.load_config
move_to_device = _base.move_to_device


def build_model(config: dict, extractor) -> PriFoldSymFlow_v6:
    mc = config['model']
    loss_config = config.get('loss', None)  # v6: modular loss from top-level "loss" key
    return PriFoldSymFlow_v6(
        extractor=extractor,
        freeze_mars=mc.get('freeze_mars', True),
        mars_dim=mc.get('mars_dim', 1056),
        mars_n_attn_layers=mc.get('mars_n_attn_layers', 6),
        mars_n_heads=mc.get('mars_n_heads', 12),
        mars_hidden_layer_indices=mc.get('mars_hidden_layer_indices', [3, 6, 9, 12]),
        mars_hidden_fusion_dim=mc.get('mars_hidden_fusion_dim', 64),
        use_seq_oh=mc.get('use_seq_oh', True),
        hidden_dim=mc.get('hidden_dim', 320),
        num_heads=mc.get('num_heads', 4),
        dim_head=mc.get('dim_head', 80),
        num_layers=mc.get('num_layers', 12),
        patch_size=mc.get('patch_size', 4),
        mars_emb_proj_dim=mc.get('mars_emb_proj_dim', 32),
        mars_attn_proj_dim=mc.get('mars_attn_proj_dim', 16),
        xt_emb_dim=mc.get('xt_emb_dim', 8),
        mlp_ratio=mc.get('mlp_ratio', 4),
        dropout=mc.get('dropout', 0.1),
        dilation_pattern=mc.get('dilation_pattern', None),
        tri_start_layer=mc.get('tri_start_layer', 4),
        tri_dim=mc.get('tri_dim', 64),
        refine_mid_ch=mc.get('refine_mid_ch', 16),
        cond_bias_zero_init=mc.get('cond_bias_zero_init', True),
        control_every=mc.get('control_every', 3),
        use_direct_head=mc.get('use_direct_head', True),
        use_density_head=mc.get('use_density_head', True),
        rho_0=mc.get('rho_0', 0.005),
        density_hint_dropout=mc.get('density_hint_dropout', 1.0),
        direct_score_weight=mc.get('direct_score_weight', 0.5),
        loss_config=loss_config,
    )


@torch.no_grad()
def evaluate(model, loader, device, config, logger, split_name: str):
    model.eval()
    scfg = config.get('sampling', {})
    amp_on, amp_dtype = _base._resolve_amp_dtype(config)
    merged = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mcc': 0.0,
              'gt_pairs': 0.0, 'pred_pairs': 0.0}
    n_samples = 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        kwargs = dict(
            num_steps=scfg.get('num_steps', 20),
            num_samples_per_input=scfg.get('num_samples_per_input', 1),
            density_guided=scfg.get('density_guided', False),
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
                pred, _ = model.sample(batch, **kwargs)
        else:
            pred, _ = model.sample(batch, **kwargs)
        m = contact_metrics(pred, batch['contact'], batch['length'])
        bs = m['n']
        n_samples += bs
        for k in merged:
            merged[k] += m[k] * bs
        if step % 20 == 0:
            logger.info(f"[Eval:{split_name}] step={step}/{len(loader)} "
                        f"L={batch['set_max_len']} F1={m['f1']:.4f}")
    out = {k: v / max(n_samples, 1) for k, v in merged.items()}
    out['n'] = n_samples
    out['time_s'] = time.time() - t0
    logger.info(
        f"[Eval:{split_name}] N={out['n']} F1={out['f1']:.4f} "
        f"P={out['precision']:.4f} R={out['recall']:.4f} MCC={out['mcc']:.4f} "
        f"time={out['time_s']:.1f}s")
    return out


def main():
    _base.build_model = build_model
    _base.evaluate = evaluate
    _base.main()


if __name__ == '__main__':
    main()
