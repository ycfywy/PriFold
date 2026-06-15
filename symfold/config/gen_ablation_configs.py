#!/usr/bin/env python3
"""Generate all ablation experiment configs from v6_full.json baseline.

Usage:
    python symfold/gen_ablation_configs.py

Generates configs in symfold/config/ablations/
"""
import json, copy, os

BASE_PATH = 'symfold/config/v6_full.json'
OUT_DIR = 'symfold/config/ablations'
os.makedirs(OUT_DIR, exist_ok=True)

with open(BASE_PATH) as f:
    base = json.load(f)


def make_config(name: str, changes: dict, desc: str):
    """Create ablation config with specified changes."""
    cfg = copy.deepcopy(base)
    cfg['task_name'] = name
    cfg['_comment'] = desc

    for key_path, value in changes.items():
        parts = key_path.split('.')
        obj = cfg
        for p in parts[:-1]:
            obj = obj[p]
        obj[parts[-1]] = value

    # Update paths
    for k in cfg['paths']:
        if k in ('data_dir', 'pretrained_lm_dir'):
            continue
        cfg['paths'][k] = cfg['paths'][k].replace('v6_full', name)

    out_path = os.path.join(OUT_DIR, f'{name}.json')
    with open(out_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'  {out_path:60s} — {desc}')


print("Generating ablation configs...\n")

# ============================================================
# Ablation A: Loss components (one removed at a time)
# ============================================================
print("=== A. Loss Component Ablation ===")

make_config('abl_no_dice', {
    'loss.dice.enabled': False,
}, 'w/o Dice loss (set-level F1 proxy)')

make_config('abl_no_ratio_pen', {
    'loss.ratio_penalty.enabled': False,
}, 'w/o Ratio penalty (anti-overprediction)')

make_config('abl_no_pair_count', {
    'loss.pair_count.enabled': False,
}, 'w/o Pair count calibration')

make_config('abl_no_calibration', {
    'loss.pair_count.enabled': False,
    'loss.ratio_penalty.enabled': False,
}, 'w/o all calibration (pair_count + ratio_penalty)')

make_config('abl_no_setlevel', {
    'loss.dice.enabled': False,
    'loss.pair_count.enabled': False,
    'loss.ratio_penalty.enabled': False,
}, 'w/o all set-level losses (BCE + structural only)')

# ============================================================
# Ablation B: Adaptive decoding
# ============================================================
print("\n=== B. Decoding Strategy Ablation ===")

make_config('abl_fixed_budget', {
    'sampling.use_density_budget': False,
}, 'Fixed budget (0.30) instead of adaptive')

make_config('abl_budget_scale_10', {
    'sampling.budget_scale': 1.0,
}, 'Adaptive budget scale=1.0 (no margin)')

make_config('abl_budget_scale_13', {
    'sampling.budget_scale': 1.3,
}, 'Adaptive budget scale=1.3 (more margin)')

# ============================================================
# Ablation C: Set-level loss variant
# ============================================================
print("\n=== C. Set-Level Loss Variant ===")

make_config('abl_tversky_03_07', {
    'loss.dice.enabled': False,
    'loss.tversky.enabled': True,
    'loss.tversky.alpha': 0.3,
    'loss.tversky.beta': 0.7,
}, 'Tversky(α=0.3,β=0.7) — push recall')

make_config('abl_tversky_07_03', {
    'loss.dice.enabled': False,
    'loss.tversky.enabled': True,
    'loss.tversky.alpha': 0.7,
    'loss.tversky.beta': 0.3,
}, 'Tversky(α=0.7,β=0.3) — push precision')

make_config('abl_tversky_05_05', {
    'loss.dice.enabled': False,
    'loss.tversky.enabled': True,
    'loss.tversky.alpha': 0.5,
    'loss.tversky.beta': 0.5,
}, 'Tversky(α=0.5,β=0.5) — symmetric (=Dice)')

# ============================================================
# Ablation D: Focal gamma
# ============================================================
print("\n=== D. Focal Gamma ===")

make_config('abl_focal_0', {
    'loss.bce.focal_gamma': 0.0,
}, 'No focal (standard BCE)')

make_config('abl_focal_2', {
    'loss.bce.focal_gamma': 2.0,
}, 'focal_gamma=2.0 (v4 setting)')

print(f"\nDone! Total configs in {OUT_DIR}/")
