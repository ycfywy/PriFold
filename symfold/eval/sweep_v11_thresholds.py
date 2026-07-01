# -*- coding: utf-8 -*-
"""Sweep v11 inference score_threshold on bpRNA-test and plot metrics.

This script runs the expensive v11 forward pass once per sample, then applies
multiple score thresholds to the same score map for efficient threshold sweep.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.lm import get_extractor
from symfold.data import build_records, PriFoldSymFlowDataset, make_collate_fn
from symfold.metrics import contact_metrics
from symfold.train.train_v11 import build_model


DEFAULT_THRESHOLDS = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]


def parse_thresholds(text: str | None) -> list[float]:
    if not text:
        return DEFAULT_THRESHOLDS
    vals = sorted({round(float(x.strip()), 4) for x in text.split(',') if x.strip()})
    if not vals:
        raise ValueError('Empty threshold list')
    return vals


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def forward_score_and_budget(model, batch: dict, scfg: dict):
    contact_mask = batch['contact_mask']
    seq_oh = batch.get('seq_oh')
    set_len = contact_mask.shape[-1]

    mars_hidden, mars_attn = model._extract_mars(batch['input_ids'], batch['attention_mask'], set_len)
    pair = model._build_pair_features(mars_hidden, mars_attn, seq_oh)
    for layer in model.layers:
        pair = layer(pair)

    pair = model.out_norm(pair)
    logit = model.contact_head(pair).permute(0, 3, 1, 2)
    logit = (logit + logit.transpose(2, 3)) / 2
    logit = logit * contact_mask
    score = torch.sigmoid(logit)

    valid = contact_mask.squeeze(1)
    denom = valid.sum(dim=(1, 2)).unsqueeze(-1).clamp(min=1)
    pair_pooled = (pair * valid.unsqueeze(-1)).sum(dim=(1, 2)) / denom
    density_pred = model.density_head(pair_pooled).squeeze(-1)

    l_eff = valid[:, 0, :].sum(dim=-1)
    if scfg.get('use_density_budget', True):
        length_decay = scfg.get('length_decay', 0.15)
        budget_floor = scfg.get('budget_floor', 0.6)
        length_factor = (100.0 / l_eff.clamp(min=50)) ** length_decay
        length_factor = length_factor.clamp(min=budget_floor)
        max_pairs = torch.round(density_pred * l_eff * length_factor * 1.05).long()
    else:
        budget_fraction = scfg.get('default_budget_fraction', 0.30)
        max_pairs = torch.round(l_eff * budget_fraction).long()
    max_pairs = max_pairs.clamp(min=2)
    return score, max_pairs


def project_threshold(score: torch.Tensor, contact_mask: torch.Tensor, max_pairs: torch.Tensor, threshold: float) -> torch.Tensor:
    bsz, _, length, _ = score.shape
    upper = torch.triu(torch.ones(length, length, device=score.device, dtype=torch.bool), diagonal=3)
    pred_maps = []
    for b in range(bsz):
        s = score[b, 0]
        m = contact_mask[b, 0]
        candidates = s * m * upper.float()
        candidates = candidates.clone()
        candidates[candidates < threshold] = 0
        flat = candidates.reshape(-1)
        n_valid = int((flat > 0).sum().item())
        contact_map = torch.zeros(length, length, device=score.device, dtype=score.dtype)
        if n_valid > 0:
            k = min(int(max_pairs[b].item()), n_valid)
            vals, idx = flat.topk(k)
            keep = vals > 0
            idx = idx[keep]
            rows = idx // length
            cols = idx % length
            contact_map[rows, cols] = 1.0
            contact_map[cols, rows] = 1.0
        pred_maps.append(contact_map)
    return torch.stack(pred_maps).unsqueeze(1)


def normalize_results(acc: dict[float, dict]) -> list[dict]:
    rows = []
    for thr in sorted(acc):
        item = acc[thr]
        n = item['n']
        row = {'threshold': thr, 'n': n}
        for key in ['precision', 'recall', 'f1', 'mcc', 'gt_pairs', 'pred_pairs']:
            row[key] = item[key] / max(n, 1)
        row['pred_gt_ratio'] = row['pred_pairs'] / max(row['gt_pairs'], 1e-12)
        rows.append(row)
    return rows


def save_outputs(rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / 'v11_threshold_sweep_results.json'
    csv_path = out_dir / 'v11_threshold_sweep_results.csv'
    png_path = out_dir / 'v11_threshold_sweep_f1.png'
    md_path = out_dir / 'v11_threshold_sweep_summary.md'

    json_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    xs = [r['threshold'] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, [r['f1'] for r in rows], marker='o', linewidth=2, label='F1')
    ax.plot(xs, [r['precision'] for r in rows], marker='s', linewidth=1.5, label='Precision')
    ax.plot(xs, [r['recall'] for r in rows], marker='^', linewidth=1.5, label='Recall')
    best = max(rows, key=lambda r: r['f1'])
    ax.axvline(best['threshold'], color='red', linestyle='--', alpha=0.6,
               label=f"Best F1={best['f1']:.4f} @ {best['threshold']:.2f}")
    ax.set_xlabel('score_threshold')
    ax.set_ylabel('Metric')
    ax.set_title('v11 bpRNA-test threshold sweep')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    lines = [
        '# v11 score_threshold sweep on bpRNA-test',
        '',
        f'- Best threshold: `{best["threshold"]:.2f}`',
        f'- Best F1: `{best["f1"]:.6f}`',
        f'- Precision / Recall / MCC: `{best["precision"]:.6f}` / `{best["recall"]:.6f}` / `{best["mcc"]:.6f}`',
        f'- Avg pred / GT pairs: `{best["pred_pairs"]:.3f}` / `{best["gt_pairs"]:.3f}` (ratio `{best["pred_gt_ratio"]:.3f}`)',
        '',
        f'![F1 curve]({png_path.name})',
        '',
        '| threshold | F1 | Precision | Recall | MCC | pred_pairs | gt_pairs | pred/gt |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(
            f'| {r["threshold"]:.2f} | {r["f1"]:.6f} | {r["precision"]:.6f} | '
            f'{r["recall"]:.6f} | {r["mcc"]:.6f} | {r["pred_pairs"]:.3f} | '
            f'{r["gt_pairs"]:.3f} | {r["pred_gt_ratio"]:.3f} |'
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return json_path, csv_path, png_path, md_path, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'symfold/config/v11/v11_hardcase_oversample.json'))
    parser.add_argument('--ckpt', default=str(ROOT / 'symfold/outputs/v11/model/best.pt'))
    parser.add_argument('--stage', default='bprna-test')
    parser.add_argument('--thresholds', default=None, help='Comma-separated thresholds, e.g. 0.4,0.5,0.6')
    parser.add_argument('--out-dir', default=str(ROOT / 'symfold/outputs/v11/threshold_sweep'))
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    thresholds = parse_thresholds(args.thresholds)
    device = torch.device(cfg.get('device', 'cuda:0') if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.get('seed', 3407))

    class A:
        pass

    a = A()
    a.pretrained_lm_dir = cfg['paths']['pretrained_lm_dir']
    a.model_scale = cfg['model']['mars_scale']
    extractor, tokenizer = get_extractor(a)
    model = build_model(cfg, extractor).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt)
    model.load_state_dict(state)
    model.eval()

    tcfg = cfg['training']
    records = build_records(cfg['paths']['data_dir'], args.stage, max_len=tcfg.get('max_len_filter', 490))
    ds = PriFoldSymFlowDataset(records, augment=False)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
        num_workers=args.num_workers,
        pin_memory=tcfg.get('pin_memory', True),
    )

    acc = {
        thr: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mcc': 0.0, 'gt_pairs': 0.0, 'pred_pairs': 0.0, 'n': 0}
        for thr in thresholds
    }
    scfg = cfg.get('sampling', {})
    amp_dtype = torch.bfloat16

    print(f'[Sweep] stage={args.stage} n={len(ds)} thresholds={thresholds}', flush=True)
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = move_batch(batch, device)
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
                score, max_pairs = forward_score_and_budget(model, batch, scfg)
            for thr in thresholds:
                pred = project_threshold(score, batch['contact_mask'], max_pairs, thr)
                metrics = contact_metrics(pred, batch['contact'], batch['length'])
                bs = pred.shape[0]
                for key in ['precision', 'recall', 'f1', 'mcc', 'gt_pairs', 'pred_pairs']:
                    acc[thr][key] += float(metrics[key]) * bs
                acc[thr]['n'] += bs
            if step % 100 == 0 or step == len(loader):
                print(f'[Sweep] {step}/{len(loader)} done', flush=True)
            del batch, score, max_pairs
            if device.type == 'cuda' and step % 50 == 0:
                torch.cuda.empty_cache()

    rows = normalize_results(acc)
    json_path, csv_path, png_path, md_path, best = save_outputs(rows, Path(args.out_dir))
    print('[Sweep] results:')
    for r in rows:
        print(f"  thr={r['threshold']:.2f} F1={r['f1']:.6f} P={r['precision']:.6f} R={r['recall']:.6f} MCC={r['mcc']:.6f} pred/gt={r['pred_gt_ratio']:.3f}")
    print(f"[Sweep] best threshold={best['threshold']:.2f} F1={best['f1']:.6f}")
    print(f'[Sweep] saved json={json_path}')
    print(f'[Sweep] saved csv={csv_path}')
    print(f'[Sweep] saved png={png_path}')
    print(f'[Sweep] saved md={md_path}')


if __name__ == '__main__':
    main()
