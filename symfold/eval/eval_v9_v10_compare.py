# -*- coding: utf-8 -*-
"""v9 vs v10 全面对比评估：在 train/val/test 三个 split 上做 per-sample 评估。

为 v9 和 v10 各自生成 per-sample 结果，覆盖 train/val/test，然后对比分析。

Usage:
  CUDA_VISIBLE_DEVICES=0 python symfold/eval/eval_v9_v10_compare.py
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

DEVICE = 'cuda:0'

VERSIONS = {
    'v9': {
        'config': ROOT / 'symfold/config/v9/v9_ddp.json',
        'ckpt': ROOT / 'symfold/outputs/v9_ddp/model/best.pt',
    },
    'v10': {
        'config': ROOT / 'symfold/config/v10/v10_ddp.json',
        'ckpt': ROOT / 'symfold/outputs/v10_ddp/model/best.pt',
    },
}
OUTPUT_DIR = ROOT / 'symfold/outputs/v9_v10_compare'
# train 集全量评估（10807 样本），不偷懒采样
TRAIN_SUBSAMPLE = None


def build_model(cfg, extractor):
    mcfg = cfg['model']
    v9cfg = cfg.get('v9', {})
    lcfg = cfg.get('loss', {})
    return DensityNetProPlus(
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


def get_family(name: str) -> str:
    parts = name.split('_')
    return parts[1] if len(parts) >= 3 else 'unknown'


@torch.no_grad()
def eval_split(model, records, tokenizer, scfg, max_len=490):
    """Per-sample evaluation on a list of records."""
    ds = PriFoldSymFlowDataset(records, augment=False)
    collate = make_collate_fn(tokenizer)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=4)

    results = []
    for batch in loader:
        batch = {k: v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred, _ = model.predict(
                batch,
                budget_fraction=scfg.get('default_budget_fraction', 0.30),
                use_density_budget=scfg.get('use_density_budget', True),
                score_threshold=scfg.get('score_threshold', 0.43),
                length_decay=scfg.get('length_decay', 0.15),
                budget_floor=scfg.get('budget_floor', 0.6),
            )
        length = int(batch['length'][0])
        p = pred[0].squeeze().cpu()[:length, :length] > 0.5
        y = batch['contact'][0].squeeze().cpu()[:length, :length] > 0.5
        idx = torch.arange(length)
        m = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        m &= (idx.view(length, 1) - idx.view(1, length)).abs() >= 3
        pm, ym = p[m], y[m]
        tp = int((pm & ym).sum()); fp = int((pm & ~ym).sum())
        fn = int((~pm & ym).sum()); tn = int((~pm & ~ym).sum())
        if (tp + fn) == 0 and (tp + fp) == 0:
            prec, rec, f1, mcc = 1.0, 1.0, 1.0, 1.0
        elif (tp + fn) == 0:
            prec, rec, f1, mcc = 0.0, 1.0, 0.0, 0.0
        else:
            prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            denom = math.sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 1))
            mcc = ((tp*tn) - (fp*fn)) / denom
        name = batch['names'][0] if 'names' in batch else f'sample_{len(results)}'
        results.append({
            'name': name, 'family': get_family(name), 'length': length,
            'gt_pairs': tp + fn, 'pred_pairs': tp + fp,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': prec, 'recall': rec, 'f1': f1, 'mcc': mcc,
        })
        del pred, batch
    return results


def run_version(name, vcfg):
    print(f'\n{"="*60}\nEvaluating {name}\n{"="*60}')
    config = json.loads(Path(vcfg['config']).read_text())

    class A: pass
    a = A()
    a.pretrained_lm_dir = config['paths']['pretrained_lm_dir']
    a.model_scale = config['model']['mars_scale']
    extractor, tokenizer = get_extractor(a)

    model = build_model(config, extractor).to(DEVICE)
    ckpt = torch.load(vcfg['ckpt'], map_location=DEVICE, weights_only=False)
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'Loaded {name} from epoch {ckpt.get("epoch", "?")}')

    scfg = config.get('sampling', {})
    data_dir = config['paths']['data_dir']

    out = {}
    for split in ['train', 'val', 'test']:
        recs = build_records(data_dir, f'bprna-{split}', max_len=490)
        if split == 'train' and TRAIN_SUBSAMPLE:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(recs), min(TRAIN_SUBSAMPLE, len(recs)), replace=False)
            recs = [recs[i] for i in sorted(idx)]
        t0 = time.time()
        res = eval_split(model, recs, tokenizer, scfg)
        out[split] = res
        mean_f1 = np.mean([r['f1'] for r in res])
        print(f'  {split}: {len(res)} samples, mean F1={mean_f1:.4f}, time={time.time()-t0:.0f}s')

    del model, extractor
    torch.cuda.empty_cache()
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for name, vcfg in VERSIONS.items():
        all_results[name] = run_version(name, vcfg)
        # Save raw
        with open(OUTPUT_DIR / f'{name}_per_sample.json', 'w') as f:
            json.dump(all_results[name], f, indent=2)

    print('\nSaved per-sample results. Generating comparison report...')
    generate_report(all_results)


def generate_report(all_results):
    """Generate v9 vs v10 comparison report."""
    v9, v10 = all_results['v9'], all_results['v10']
    lines = ['# v9 vs v10 全面对比分析报告\n']
    lines.append(f'> 评估日期: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'> v9 = MARS frozen, v10 = MARS unfrozen')
    lines.append(f'> Train/Val/Test 全部全量评估\n')

    # ===== 1. Train/Val/Test 总体 =====
    lines.append('## 1. Train / Val / Test 总体表现对比\n')
    lines.append('| Split | v9 F1 | v10 F1 | ΔF1 | v9 Prec | v10 Prec | v9 Rec | v10 Rec |')
    lines.append('|-------|-------|--------|-----|---------|----------|--------|---------|')
    for split in ['train', 'val', 'test']:
        v9s, v10s = v9[split], v10[split]
        v9f1 = np.mean([r['f1'] for r in v9s]); v10f1 = np.mean([r['f1'] for r in v10s])
        v9p = np.mean([r['precision'] for r in v9s]); v10p = np.mean([r['precision'] for r in v10s])
        v9r = np.mean([r['recall'] for r in v9s]); v10r = np.mean([r['recall'] for r in v10s])
        lines.append(f'| {split} | {v9f1:.4f} | {v10f1:.4f} | {v10f1-v9f1:+.4f} | '
                     f'{v9p:.4f} | {v10p:.4f} | {v9r:.4f} | {v10r:.4f} |')

    # 泛化 gap
    lines.append('\n### 泛化差距 (Train F1 - Test F1)\n')
    v9_gap = np.mean([r['f1'] for r in v9['train']]) - np.mean([r['f1'] for r in v9['test']])
    v10_gap = np.mean([r['f1'] for r in v10['train']]) - np.mean([r['f1'] for r in v10['test']])
    lines.append(f'| 版本 | Train F1 | Test F1 | Gap |')
    lines.append(f'|------|----------|---------|-----|')
    lines.append(f'| v9 | {np.mean([r["f1"] for r in v9["train"]]):.4f} | {np.mean([r["f1"] for r in v9["test"]]):.4f} | {v9_gap:.4f} |')
    lines.append(f'| v10 | {np.mean([r["f1"] for r in v10["train"]]):.4f} | {np.mean([r["f1"] for r in v10["test"]]):.4f} | {v10_gap:.4f} |')

    # ===== 2. 按家族 =====
    lines.append('\n## 2. 各家族在 Train/Val/Test 上的表现对比\n')
    for split in ['train', 'val', 'test']:
        lines.append(f'\n### {split.upper()} split\n')
        lines.append('| 家族 | N | v9 F1 | v10 F1 | ΔF1 |')
        lines.append('|------|---|-------|--------|-----|')
        v9_fam = defaultdict(list); v10_fam = defaultdict(list)
        for r in v9[split]: v9_fam[r['family']].append(r['f1'])
        for r in v10[split]: v10_fam[r['family']].append(r['f1'])
        fams = sorted(set(v9_fam) | set(v10_fam), key=lambda f: -len(v10_fam.get(f, [])))
        for fam in fams:
            v9l = v9_fam.get(fam, []); v10l = v10_fam.get(fam, [])
            if not v10l: continue
            v9m = np.mean(v9l) if v9l else 0
            v10m = np.mean(v10l)
            lines.append(f'| {fam} | {len(v10l)} | {v9m:.4f} | {v10m:.4f} | {v10m-v9m:+.4f} |')

    # ===== 3. 按长度 (test) =====
    lines.append('\n## 3. 不同长度区间表现对比 (Test)\n')
    lines.append('| 长度区间 | N | v9 F1 | v10 F1 | ΔF1 |')
    lines.append('|----------|---|-------|--------|-----|')
    buckets = [(0,50),(50,100),(100,150),(150,200),(200,300),(300,400),(400,500)]
    v9_test = {r['name']: r for r in v9['test']}
    v10_test = {r['name']: r for r in v10['test']}
    common = set(v9_test) & set(v10_test)
    for lo, hi in buckets:
        names = [n for n in common if lo <= v10_test[n]['length'] < hi]
        if not names: continue
        v9m = np.mean([v9_test[n]['f1'] for n in names])
        v10m = np.mean([v10_test[n]['f1'] for n in names])
        lines.append(f'| {lo}-{hi} | {len(names)} | {v9m:.4f} | {v10m:.4f} | {v10m-v9m:+.4f} |')

    # ===== 4. 改善最明显的样本 (test) =====
    lines.append('\n## 4. v10 相比 v9 改善最明显的样本 (Test)\n')
    deltas = []
    for n in common:
        d = v10_test[n]['f1'] - v9_test[n]['f1']
        deltas.append((n, d, v9_test[n], v10_test[n]))
    deltas.sort(key=lambda x: x[1], reverse=True)
    lines.append('\n### Top 25 改善样本\n')
    lines.append('| 样本 | 家族 | 长度 | v9 F1 | v10 F1 | ΔF1 |')
    lines.append('|------|------|------|-------|--------|-----|')
    for n, d, v9r, v10r in deltas[:25]:
        lines.append(f'| {n} | {v10r["family"]} | {v10r["length"]} | {v9r["f1"]:.4f} | {v10r["f1"]:.4f} | {d:+.4f} |')

    lines.append('\n### Top 15 退化样本\n')
    lines.append('| 样本 | 家族 | 长度 | v9 F1 | v10 F1 | ΔF1 |')
    lines.append('|------|------|------|-------|--------|-----|')
    for n, d, v9r, v10r in deltas[-15:][::-1]:
        lines.append(f'| {n} | {v10r["family"]} | {v10r["length"]} | {v9r["f1"]:.4f} | {v10r["f1"]:.4f} | {d:+.4f} |')

    # 改善样本的家族/长度分布
    improved = [(n,d,a,b) for n,d,a,b in deltas if d > 0.1]
    lines.append(f'\n### 显著改善样本 (ΔF1>0.1) 的分布 — 共 {len(improved)} 个\n')
    imp_fam = defaultdict(int)
    imp_len = defaultdict(int)
    for n, d, _, v10r in improved:
        imp_fam[v10r['family']] += 1
        L = v10r['length']
        bucket = '0-100' if L < 100 else '100-200' if L < 200 else '200-300' if L < 300 else '300+'
        imp_len[bucket] += 1
    lines.append('**家族分布**: ' + ', '.join(f'{k}={v}' for k,v in sorted(imp_fam.items(), key=lambda x:-x[1])))
    lines.append('\n**长度分布**: ' + ', '.join(f'{k}={v}' for k,v in sorted(imp_len.items())))

    # ===== 5. Bad case 对比 =====
    lines.append('\n## 5. Bad Cases 对比 (F1 < 0.3, Test)\n')
    v9_bad = {n for n in common if v9_test[n]['f1'] < 0.3}
    v10_bad = {n for n in common if v10_test[n]['f1'] < 0.3}
    fixed = v9_bad - v10_bad
    new_bad = v10_bad - v9_bad
    still_bad = v9_bad & v10_bad
    lines.append(f'| 类别 | 数量 |')
    lines.append(f'|------|------|')
    lines.append(f'| v9 bad cases | {len(v9_bad)} |')
    lines.append(f'| v10 bad cases | {len(v10_bad)} |')
    lines.append(f'| ✅ v9 bad → v10 修复 | {len(fixed)} |')
    lines.append(f'| ❌ v9 好 → v10 变 bad | {len(new_bad)} |')
    lines.append(f'| ⚠️ 两版本都 bad | {len(still_bad)} |')
    lines.append(f'| Bad case 重合率 | {len(still_bad)/max(len(v9_bad|v10_bad),1)*100:.1f}% |')

    if fixed:
        lines.append(f'\n### v10 修复的 v9 bad cases (Top 20)\n')
        lines.append('| 样本 | 家族 | 长度 | v9 F1 | v10 F1 |')
        lines.append('|------|------|------|-------|--------|')
        fixed_sorted = sorted(fixed, key=lambda n: v10_test[n]['f1'] - v9_test[n]['f1'], reverse=True)
        for n in fixed_sorted[:20]:
            lines.append(f'| {n} | {v10_test[n]["family"]} | {v10_test[n]["length"]} | {v9_test[n]["f1"]:.4f} | {v10_test[n]["f1"]:.4f} |')

    if new_bad:
        lines.append(f'\n### v10 新产生的 bad cases\n')
        lines.append('| 样本 | 家族 | 长度 | v9 F1 | v10 F1 |')
        lines.append('|------|------|------|-------|--------|')
        for n in sorted(new_bad, key=lambda n: v10_test[n]['f1']):
            lines.append(f'| {n} | {v10_test[n]["family"]} | {v10_test[n]["length"]} | {v9_test[n]["f1"]:.4f} | {v10_test[n]["f1"]:.4f} |')

    report = '\n'.join(lines)
    (OUTPUT_DIR / 'v9_vs_v10_comparison.md').write_text(report)
    print(f'Report saved: {OUTPUT_DIR}/v9_vs_v10_comparison.md')
    print(f'\nSummary: v9 test F1={np.mean([r["f1"] for r in v9["test"]]):.4f}, '
          f'v10 test F1={np.mean([r["f1"] for r in v10["test"]]):.4f}')
    print(f'Bad cases: v9={len(v9_bad)}, v10={len(v10_bad)}, fixed={len(fixed)}, new={len(new_bad)}')


if __name__ == '__main__':
    main()
