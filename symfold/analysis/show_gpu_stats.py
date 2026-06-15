"""Tail / 汇总查看 v2 训练 GPU 监控 JSONL。

用法：
  python symfold/show_gpu_stats.py symfold/outputs/v2_bprna/gpu_stats.jsonl
  python symfold/show_gpu_stats.py symfold/outputs/v2_bprna/gpu_stats.jsonl --tail 30
  python symfold/show_gpu_stats.py symfold/outputs/v2_bprna/gpu_stats.jsonl --summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('jsonl', type=str, help='路径，例如 outputs/<task>/gpu_stats.jsonl')
    p.add_argument('--tail', type=int, default=0, help='只显示最后 N 条')
    p.add_argument('--summary', action='store_true', help='按 phase 汇总峰值')
    p.add_argument('--phase', type=str, default=None, help='过滤 phase（substring 匹配）')
    return p.parse_args()


def read_records(path: Path):
    if not path.exists():
        print(f'ERROR: {path} not found', file=sys.stderr)
        sys.exit(1)
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def fmt_row(r: dict) -> str:
    parts = [
        r.get('time', ''),
        f"phase={r.get('phase', '')}",
    ]
    if 'epoch' in r:
        parts.append(f"e{r['epoch']}")
    if 'step' in r:
        parts.append(f"s{r['step']}")
    if 'set_max_len' in r:
        parts.append(f"L={r['set_max_len']}")
    if 'nvml_used_mb' in r:
        parts.append(f"nvml_used={r['nvml_used_mb']:.0f}MB")
        parts.append(f"util={r.get('nvml_util_gpu', 0)}%")
    if 'target_pid_used_mb' in r:
        parts.append(f"target={r['target_pid_used_mb']:.0f}MB")
    if 'target_alive' in r:
        parts.append(f"alive={'Y' if r['target_alive'] else 'N'}")
    if 'nvml_temp_c' in r:
        parts.append(f"T={r['nvml_temp_c']}C")
    if 'nvml_power_w' in r:
        parts.append(f"P={r['nvml_power_w']:.0f}W")
    if 'torch_max_alloc_mb' in r and r['torch_max_alloc_mb'] > 0:
        parts.append(f"torch_peak_alloc={r['torch_max_alloc_mb']:.0f}MB")
    if 'loss' in r:
        parts.append(f"loss={r['loss']:.4f}")
    if 'f1' in r:
        parts.append(f"f1={r['f1']:.4f}")
    return ' | '.join(parts)


def summarize(records: list[dict]):
    """按 phase 汇总最大 nvml_used / target_used / util / 最小 util。"""
    bucket: dict = {}
    for r in records:
        p = r.get('phase', '?')
        b = bucket.setdefault(p, {
            'count': 0,
            'max_nvml_used': 0,
            'max_target_used': 0,
            'max_util': 0,
            'min_util': 100,
            'max_temp': 0,
            'max_power': 0,
        })
        b['count'] += 1
        b['max_nvml_used'] = max(b['max_nvml_used'], r.get('nvml_used_mb', 0))
        b['max_target_used'] = max(b['max_target_used'], r.get('target_pid_used_mb', 0))
        if 'nvml_util_gpu' in r:
            b['max_util'] = max(b['max_util'], r['nvml_util_gpu'])
            b['min_util'] = min(b['min_util'], r['nvml_util_gpu'])
        if 'nvml_temp_c' in r:
            b['max_temp'] = max(b['max_temp'], r['nvml_temp_c'])
        if 'nvml_power_w' in r:
            b['max_power'] = max(b['max_power'], r['nvml_power_w'])
    print(f"{'phase':<22} {'n':>6} {'nvml_used':>12} {'target_used':>13} "
          f"{'util_min/max':>14} {'temp':>6} {'pwr':>7}")
    print('-' * 90)
    for p, b in sorted(bucket.items()):
        print(f"{p:<22} {b['count']:>6} "
              f"{b['max_nvml_used']:>9.0f}MB "
              f"{b['max_target_used']:>10.0f}MB "
              f"{b['min_util']:>5}/{b['max_util']:<5}% "
              f"{b['max_temp']:>4}C "
              f"{b['max_power']:>5.0f}W")


def main():
    args = parse_args()
    records = read_records(Path(args.jsonl))
    if args.phase:
        records = [r for r in records if args.phase in r.get('phase', '')]
    if args.summary:
        summarize(records)
        return
    if args.tail:
        records = records[-args.tail:]
    for r in records:
        print(fmt_row(r))


if __name__ == '__main__':
    main()
