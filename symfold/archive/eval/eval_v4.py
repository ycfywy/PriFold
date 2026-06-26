# -*- coding: utf-8 -*-
"""PriFold-SymFlow v4 evaluation script."""
from __future__ import annotations

import argparse
import faulthandler
import json
import sys
from pathlib import Path

faulthandler.enable()

import torch

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor                              # noqa: E402
from symfold.data import build_loader                           # noqa: E402
from symfold.metrics import contact_metrics                     # noqa: E402
from symfold.train.train_v4 import build_model, load_config, move_to_device  # noqa: E402


DEFAULT_TEST_SETS = {
    'bprna': 'bprna-test',
    'rnastralign': 'rnastralign-test,archiveii-test',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--test_sets', default=None)
    parser.add_argument('--config', default=None)
    parser.add_argument('--num_steps', type=int, default=None)
    parser.add_argument('--num_samples_per_input', type=int, default=None)
    parser.add_argument('--density_guided', type=int, default=None)
    parser.add_argument('--projection_mode', default=None,
                        choices=['score', 'hybrid', 'sample'])
    parser.add_argument('--use_density_budget', type=int, default=None)
    parser.add_argument('--budget_scale', type=float, default=None)
    parser.add_argument('--candidate_weight', type=float, default=None)
    parser.add_argument('--direct_score_weight', type=float, default=None)
    parser.add_argument('--score_threshold', type=float, default=None)
    parser.add_argument('--default_budget_fraction', type=float, default=None)
    parser.add_argument('--out_json', default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    config = load_config(args.config) if args.config else ckpt['config']

    if args.test_sets:
        test_stages = [s.strip() for s in args.test_sets.split(',') if s.strip()]
    else:
        mode = config['training'].get('dataset_mode', 'rnastralign')
        test_stages = [s.strip() for s in DEFAULT_TEST_SETS[mode].split(',')]

    scfg = config.setdefault('sampling', {})
    if args.num_steps is not None:
        scfg['num_steps'] = args.num_steps
    if args.num_samples_per_input is not None:
        scfg['num_samples_per_input'] = args.num_samples_per_input
    if args.density_guided is not None:
        scfg['density_guided'] = bool(args.density_guided)
    if args.projection_mode is not None:
        scfg['projection_mode'] = args.projection_mode
    if args.use_density_budget is not None:
        scfg['use_density_budget'] = bool(args.use_density_budget)
    if args.budget_scale is not None:
        scfg['budget_scale'] = args.budget_scale
    if args.candidate_weight is not None:
        scfg['candidate_weight'] = args.candidate_weight
    if args.direct_score_weight is not None:
        scfg['direct_score_weight'] = args.direct_score_weight
    if args.score_threshold is not None:
        scfg['score_threshold'] = args.score_threshold
    if args.default_budget_fraction is not None:
        scfg['default_budget_fraction'] = args.default_budget_fraction

    class Args:
        pass
    lm_args = Args()
    lm_args.pretrained_lm_dir = config['paths'].get('pretrained_lm_dir', str(ROOT / 'model'))
    lm_args.model_scale = config['model'].get('mars_scale', 'lx')
    extractor, tokenizer = get_extractor(lm_args)

    device = torch.device(config.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    model = build_model(config, extractor)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    amp_name = str(config.get('training', {}).get('amp_dtype', 'fp32')).lower()
    if amp_name in ('bf16', 'bfloat16'):
        amp_on, amp_dtype = True, torch.bfloat16
    elif amp_name in ('fp16', 'half', 'float16'):
        amp_on, amp_dtype = True, torch.float16
    else:
        amp_on, amp_dtype = False, torch.float32

    print(f'== Eval {args.ckpt} ==')
    print(f'   dataset_mode={config["training"].get("dataset_mode")} '
          f'num_steps={scfg.get("num_steps", 20)} '
          f'samples_per_input={scfg.get("num_samples_per_input", 1)} '
          f'density_guided={scfg.get("density_guided", False)} '
          f'projection_mode={scfg.get("projection_mode", "score")} '
          f'use_density_budget={scfg.get("use_density_budget", False)} '
          f'amp={amp_dtype}')
    print(f'   test_stages = {test_stages}')

    results = {}
    with torch.no_grad():
        for stage in test_stages:
            loader = build_loader(stage, config, tokenizer, shuffle=False)
            merged = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mcc': 0.0,
                      'gt_pairs': 0.0, 'pred_pairs': 0.0}
            n = 0
            import time as _t
            t0 = _t.time()
            for batch in loader:
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
                    default_budget_fraction=scfg.get('default_budget_fraction', 0.35),
                )
                if amp_on:
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        pred, _ = model.sample(batch, **kwargs)
                else:
                    pred, _ = model.sample(batch, **kwargs)
                m = contact_metrics(pred, batch['contact'], batch['length'])
                bs = m['n']
                n += bs
                for k in merged:
                    merged[k] += m[k] * bs
            res = {k: v / max(n, 1) for k, v in merged.items()}
            res['N'] = n
            res['time_s'] = _t.time() - t0
            results[stage] = res
            print(f'[{stage}] N={n} F1={res["f1"]:.4f} '
                  f'P={res["precision"]:.4f} R={res["recall"]:.4f} MCC={res["mcc"]:.4f} '
                  f'time={res["time_s"]:.1f}s')

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump({'ckpt': args.ckpt, 'sampling': scfg, 'results': results}, f, indent=2)
        print(f'saved -> {out}')


if __name__ == '__main__':
    main()
