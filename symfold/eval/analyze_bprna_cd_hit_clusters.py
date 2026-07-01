# -*- coding: utf-8 -*-
"""Analyze bpRNA CD-HIT clusters and v11 bad-case cluster coverage."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FASTA_DIR = ROOT / 'data/bprna_cd_hit'
DEFAULT_CDHIT_DIR = ROOT / 'cd-hit-v4.8.1-2019-0228'
DEFAULT_OUT_DIR = ROOT / 'symfold/outputs/bprna_cd_hit_analysis'
DEFAULT_V11_RESULTS = ROOT / 'symfold/outputs/v11/comprehensive_analysis/per_sample_results.json'
DEFAULT_V11_DELTAS = ROOT / 'symfold/outputs/v11/comprehensive_analysis/per_sample_deltas.json'

SPLIT_MAP = {'TR0': 'train', 'VL0': 'val', 'TS0': 'test'}
SPLIT_ORDER = ['train', 'val', 'test']


def identity_tag(identity: float) -> str:
    return f'c{int(round(identity * 1000)):03d}'


def run_cd_hit(args, out_prefix: Path) -> Path:
    clstr_path = Path(str(out_prefix) + '.clstr')
    if clstr_path.exists() and not args.force_cdhit:
        print(f'[cd-hit] reuse existing {clstr_path}')
        return clstr_path

    binary = args.cd_hit_dir / 'cd-hit-est'
    if not binary.exists():
        raise FileNotFoundError(f'cd-hit-est not found: {binary}. Run `make` in {args.cd_hit_dir}` first.')

    cmd = [
        str(binary),
        '-i', str(args.all_fasta),
        '-o', str(out_prefix),
        '-c', str(args.identity),
        '-n', str(args.word_size),
        '-d', '0',
        '-M', str(args.memory_mb),
        '-T', str(args.threads),
        '-g', str(args.accurate_mode),
        '-r', str(args.same_strand_only),
        '-aS', str(args.coverage_short),
        '-aL', str(args.coverage_long),
    ]
    print('[cd-hit] ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return clstr_path


def parse_fasta(path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    header = None
    chunks: list[str] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    seqs[header] = ''.join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        seqs[header] = ''.join(chunks)
    return seqs


def header_parts(header: str) -> dict:
    parts = header.split('|')
    data_name = parts[0] if len(parts) > 0 else ''
    file_name = parts[1] if len(parts) > 1 else header
    split = SPLIT_MAP.get(data_name, data_name)
    length = None
    for p in parts[2:]:
        if p.startswith('len='):
            try:
                length = int(p.split('=', 1)[1])
            except ValueError:
                length = None
    return {
        'header': header,
        'data_name': data_name,
        'split': split,
        'file_name': file_name,
        'length': length,
    }


def parse_clstr(clstr_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    member_rows = []
    cluster_id = -1
    item_re = re.compile(r'^(\d+)\s+(\d+)nt,\s+>(.*?)\.\.\.\s*(.*)$')
    identity_re = re.compile(r'(\d+(?:\.\d+)?)%')

    for raw in clstr_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('>Cluster'):
            cluster_id = int(line.split()[-1])
            continue
        m = item_re.match(line)
        if not m:
            continue
        member_idx = int(m.group(1))
        length = int(m.group(2))
        header = m.group(3)
        suffix = m.group(4)
        hp = header_parts(header)
        is_rep = suffix == '*'
        ident = 100.0 if is_rep else None
        im = identity_re.search(suffix)
        if im:
            ident = float(im.group(1))
        member_rows.append({
            'cluster_id': cluster_id,
            'member_idx': member_idx,
            'is_representative': is_rep,
            'identity_to_representative_pct': ident,
            'cdhit_length': length,
            **hp,
        })

    members = pd.DataFrame(member_rows)
    if members.empty:
        raise RuntimeError(f'No members parsed from {clstr_path}')

    agg_rows = []
    for cid, group in members.groupby('cluster_id', sort=True):
        counts = Counter(group['split'])
        rep_rows = group[group['is_representative']]
        rep = rep_rows.iloc[0] if len(rep_rows) else group.iloc[0]
        present = [s for s in SPLIT_ORDER if counts.get(s, 0) > 0]
        if not present:
            composition = 'unknown'
        elif len(present) == 1:
            composition = f'{present[0]}_only'
        else:
            composition = '+'.join(present)
        agg_rows.append({
            'cluster_id': cid,
            'cluster_size': len(group),
            'n_train': counts.get('train', 0),
            'n_val': counts.get('val', 0),
            'n_test': counts.get('test', 0),
            'composition': composition,
            'representative_split': rep['split'],
            'representative_file_name': rep['file_name'],
            'representative_header': rep['header'],
        })
    clusters = pd.DataFrame(agg_rows).sort_values('cluster_id').reset_index(drop=True)
    return members, clusters


def load_v11_badcases(v11_results: Path, v11_deltas: Path, threshold: float) -> pd.DataFrame:
    rows = json.loads(v11_results.read_text())
    df = pd.DataFrame(rows)
    bad = df[df['f1'] < threshold].copy().sort_values(['f1', 'name'])
    bad = bad.rename(columns={'name': 'badcase_name'})

    if v11_deltas.exists():
        deltas = pd.DataFrame(json.loads(v11_deltas.read_text()))
        keep = [c for c in ['name', 'v10_f1', 'v11a_f1', 'delta_f1'] if c in deltas.columns]
        if keep:
            deltas = deltas[keep].rename(columns={'name': 'badcase_name'})
            bad = bad.merge(deltas, on='badcase_name', how='left')
    return bad


def build_badcase_tables(bad: pd.DataFrame, members: pd.DataFrame, clusters: pd.DataFrame):
    test_members = members[members['split'] == 'test'][['cluster_id', 'file_name']].rename(columns={'file_name': 'badcase_name'})
    cluster_cols = ['cluster_id', 'cluster_size', 'n_train', 'n_val', 'n_test', 'composition', 'representative_file_name']
    bad_table = bad.merge(test_members, on='badcase_name', how='left')
    bad_table = bad_table.merge(clusters[cluster_cols], on='cluster_id', how='left')
    bad_table['train_covered_cluster'] = bad_table['n_train'].fillna(0).astype(int) > 0

    cluster_train = members[members['split'] == 'train'].groupby('cluster_id')['file_name'].apply(lambda x: ';'.join(sorted(x))).reset_index()
    cluster_train = cluster_train.rename(columns={'file_name': 'train_members_in_cluster'})
    bad_table = bad_table.merge(cluster_train, on='cluster_id', how='left')
    bad_table['train_members_in_cluster'] = bad_table['train_members_in_cluster'].fillna('')

    covered_clusters = set(bad_table.loc[bad_table['train_covered_cluster'], 'cluster_id'].dropna().astype(int))
    train_rows = members[(members['split'] == 'train') & (members['cluster_id'].isin(covered_clusters))].copy()
    link_rows = []
    train_by_cluster = {cid: g for cid, g in train_rows.groupby('cluster_id')}
    for _, b in bad_table[bad_table['train_covered_cluster']].iterrows():
        cid = int(b['cluster_id'])
        for _, t in train_by_cluster.get(cid, pd.DataFrame()).iterrows():
            link_rows.append({
                'badcase_name': b['badcase_name'],
                'badcase_f1': b['f1'],
                'badcase_length': b.get('length', np.nan),
                'cluster_id': cid,
                'cluster_size': b['cluster_size'],
                'cluster_n_train': b['n_train'],
                'cluster_n_val': b['n_val'],
                'cluster_n_test': b['n_test'],
                'train_file_name': t['file_name'],
                'train_header': t['header'],
                'train_length': t['cdhit_length'],
                'train_identity_to_representative_pct': t['identity_to_representative_pct'],
            })
    bad_train_links = pd.DataFrame(link_rows)
    return bad_table, train_rows, bad_train_links


def write_fasta_from_members(rows: pd.DataFrame, seqs: dict[str, str], path: Path):
    with path.open('w') as f:
        for _, row in rows.sort_values(['cluster_id', 'file_name']).iterrows():
            header = row['header']
            seq = seqs.get(header)
            if not seq:
                continue
            f.write(f'>{header}\n')
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + '\n')


def plot_outputs(clusters: pd.DataFrame, members: pd.DataFrame, bad_table: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    bins = np.arange(1, min(clusters['cluster_size'].max(), 50) + 2) - 0.5
    plt.hist(clusters['cluster_size'], bins=bins, color='#4C78A8', edgecolor='white')
    plt.yscale('log')
    plt.xlabel('Cluster size')
    plt.ylabel('Cluster count (log scale)')
    plt.title('bpRNA CD-HIT cluster size distribution')
    plt.tight_layout()
    plt.savefig(fig_dir / 'cluster_size_histogram.png', dpi=160)
    plt.close()

    comp = clusters['composition'].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    bars = plt.bar(comp.index, comp.values, color='#59A14F')
    plt.xticks(rotation=35, ha='right')
    plt.ylabel('Cluster count')
    plt.title('Cluster composition by split')
    for b, v in zip(bars, comp.values):
        plt.text(b.get_x() + b.get_width() / 2, v, str(v), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / 'cluster_split_composition.png', dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(clusters['n_train'], clusters['n_test'], s=np.clip(clusters['cluster_size'] * 5, 10, 200),
                c=clusters['n_val'], cmap='viridis', alpha=0.65, edgecolors='none')
    plt.colorbar(label='Val samples in cluster')
    plt.xlabel('Train samples in cluster')
    plt.ylabel('Test samples in cluster')
    plt.title('Train/Test cluster overlap')
    plt.tight_layout()
    plt.savefig(fig_dir / 'cluster_train_test_scatter.png', dpi=160)
    plt.close()

    if not bad_table.empty:
        cov_counts = bad_table['train_covered_cluster'].map({True: 'covered_by_train_cluster', False: 'not_covered'}).value_counts()
        plt.figure(figsize=(6, 4))
        colors = ['#4C78A8' if x == 'covered_by_train_cluster' else '#E15759' for x in cov_counts.index]
        bars = plt.bar(cov_counts.index, cov_counts.values, color=colors)
        plt.ylabel('Bad case count')
        plt.title('v11 bad cases: cluster covered by train?')
        for b, v in zip(bars, cov_counts.values):
            plt.text(b.get_x() + b.get_width() / 2, v, str(v), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(fig_dir / 'badcase_train_coverage.png', dpi=160)
        plt.close()

        groups = [
            bad_table.loc[bad_table['train_covered_cluster'], 'f1'].dropna().values,
            bad_table.loc[~bad_table['train_covered_cluster'], 'f1'].dropna().values,
        ]
        plt.figure(figsize=(6, 4))
        plt.boxplot(groups, tick_labels=['covered', 'not covered'], showmeans=True)
        plt.ylabel('v11 F1')
        plt.title('Bad-case F1 by train cluster coverage')
        plt.tight_layout()
        plt.savefig(fig_dir / 'badcase_f1_by_coverage.png', dpi=160)
        plt.close()

        plt.figure(figsize=(7, 5))
        colors = bad_table['train_covered_cluster'].map({True: '#4C78A8', False: '#E15759'})
        plt.scatter(bad_table['cluster_size'], bad_table['f1'], c=colors, alpha=0.75)
        plt.xlabel('Cluster size')
        plt.ylabel('v11 F1')
        plt.title('v11 bad cases: cluster size vs F1')
        plt.tight_layout()
        plt.savefig(fig_dir / 'badcase_cluster_size_vs_f1.png', dpi=160)
        plt.close()

        top = bad_table.sort_values(['n_train', 'cluster_size'], ascending=False).head(20)
        if len(top):
            x = np.arange(len(top))
            plt.figure(figsize=(12, 5))
            plt.bar(x, top['n_train'], label='train', color='#4C78A8')
            plt.bar(x, top['n_val'], bottom=top['n_train'], label='val', color='#F28E2B')
            plt.bar(x, top['n_test'], bottom=top['n_train'] + top['n_val'], label='test', color='#E15759')
            plt.xticks(x, top['badcase_name'], rotation=75, ha='right', fontsize=8)
            plt.ylabel('Samples in bad-case cluster')
            plt.title('Top bad-case clusters by train coverage')
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / 'top_badcase_clusters_split_stack.png', dpi=160)
            plt.close()


def write_summary_report(args, clusters, members, bad_table, bad_links, out_dir: Path):
    split_counts = members['split'].value_counts().to_dict()
    comp_counts = clusters['composition'].value_counts().to_dict()
    bad_n = len(bad_table)
    bad_covered = int(bad_table['train_covered_cluster'].sum()) if bad_n else 0
    bad_uncovered = bad_n - bad_covered

    lines = [
        '# bpRNA CD-HIT 聚类与 v11 bad case 覆盖分析报告',
        '',
        '## 1. 参数',
        '',
        f'- 输入 FASTA 目录：`{args.fasta_dir}`',
        f'- 全量 FASTA：`{args.all_fasta}`',
        f'- CD-HIT identity：`{args.identity}`',
        f'- word size：`{args.word_size}`',
        f'- coverage：`-aS {args.coverage_short}`，`-aL {args.coverage_long}`',
        f'- strand：`-r {args.same_strand_only}`',
        f'- v11 bad case 阈值：`F1 < {args.badcase_threshold}`',
        '',
        '## 2. 全量聚类概览',
        '',
        f'- 样本总数：`{len(members)}`',
        f'- 聚类总数：`{len(clusters)}`',
        f'- 平均 cluster size：`{clusters["cluster_size"].mean():.3f}`',
        f'- 最大 cluster size：`{int(clusters["cluster_size"].max())}`',
        f'- singleton cluster 数：`{int((clusters["cluster_size"] == 1).sum())}`',
        f'- train 样本数：`{split_counts.get("train", 0)}`',
        f'- val 样本数：`{split_counts.get("val", 0)}`',
        f'- test 样本数：`{split_counts.get("test", 0)}`',
        '',
        '### 2.1 cluster composition',
        '',
        '| composition | clusters |',
        '|---|---:|',
    ]
    for k, v in sorted(comp_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f'| `{k}` | {v} |')

    lines.extend([
        '',
        '## 3. v11 bad case 覆盖概览',
        '',
        f'- bad case 数：`{bad_n}`',
        f'- bad case 所属 cluster 中有 train 样本：`{bad_covered}`',
        f'- bad case 所属 cluster 中没有 train 样本：`{bad_uncovered}`',
        f'- 导出的 bad-case 相关 train 样本行数：`{len(bad_links)}`',
        '',
        '## 4. 主要输出文件',
        '',
        f'- `csv/cluster_summary.tsv`：每个 cluster 的 `train/val/test` 数量。',
        f'- `csv/cluster_members.tsv`：每个样本所属 cluster。',
        f'- `csv/v11_badcase_cluster_summary.tsv`：每个 v11 bad case 所属 cluster 及 train 覆盖情况。',
        f'- `csv/v11_badcase_train_cluster_members.tsv`：与 bad case 同 cluster 的 train 样本明细。',
        f'- `fasta/v11_badcase_train_cluster_members.fa`：与 bad case 同 cluster 的 train 序列 FASTA。',
        f'- `figures/`：可视化图。',
        '',
        '## 5. 图表',
        '',
        '![cluster_size_histogram](figures/cluster_size_histogram.png)',
        '',
        '![cluster_split_composition](figures/cluster_split_composition.png)',
        '',
        '![cluster_train_test_scatter](figures/cluster_train_test_scatter.png)',
        '',
        '![badcase_train_coverage](figures/badcase_train_coverage.png)',
        '',
        '![badcase_f1_by_coverage](figures/badcase_f1_by_coverage.png)',
        '',
        '![badcase_cluster_size_vs_f1](figures/badcase_cluster_size_vs_f1.png)',
        '',
        '![top_badcase_clusters_split_stack](figures/top_badcase_clusters_split_stack.png)',
        '',
        '## 6. 复现命令',
        '',
        '```bash',
        f'python symfold/eval/analyze_bprna_cd_hit_clusters.py --identity {args.identity} --word-size {args.word_size}',
        '```',
        '',
    ])
    (out_dir / 'analysis_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta-dir', type=Path, default=DEFAULT_FASTA_DIR)
    parser.add_argument('--all-fasta', type=Path, default=DEFAULT_FASTA_DIR / 'bprna_all.fa')
    parser.add_argument('--cd-hit-dir', type=Path, default=DEFAULT_CDHIT_DIR)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--v11-results', type=Path, default=DEFAULT_V11_RESULTS)
    parser.add_argument('--v11-deltas', type=Path, default=DEFAULT_V11_DELTAS)
    parser.add_argument('--identity', type=float, default=0.95)
    parser.add_argument('--word-size', type=int, default=10)
    parser.add_argument('--coverage-short', type=float, default=0.9)
    parser.add_argument('--coverage-long', type=float, default=0.9)
    parser.add_argument('--same-strand-only', type=int, default=0)
    parser.add_argument('--accurate-mode', type=int, default=1)
    parser.add_argument('--memory-mb', type=int, default=16000)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--badcase-threshold', type=float, default=0.3)
    parser.add_argument('--force-cdhit', action='store_true')
    args = parser.parse_args()

    tag = identity_tag(args.identity)
    cdhit_dir = args.out_dir / 'cdhit'
    csv_dir = args.out_dir / 'csv'
    fig_dir = args.out_dir / 'figures'
    fasta_dir = args.out_dir / 'fasta'
    for d in [cdhit_dir, csv_dir, fig_dir, fasta_dir]:
        d.mkdir(parents=True, exist_ok=True)

    out_prefix = cdhit_dir / f'bprna_all_{tag}'
    clstr_path = run_cd_hit(args, out_prefix)

    members, clusters = parse_clstr(clstr_path)
    bad = load_v11_badcases(args.v11_results, args.v11_deltas, args.badcase_threshold)
    bad_table, train_rows, bad_links = build_badcase_tables(bad, members, clusters)

    seqs = parse_fasta(args.all_fasta)
    clusters.to_csv(csv_dir / 'cluster_summary.tsv', sep='\t', index=False)
    members.to_csv(csv_dir / 'cluster_members.tsv', sep='\t', index=False)
    bad_table.to_csv(csv_dir / 'v11_badcase_cluster_summary.tsv', sep='\t', index=False)
    bad_links.to_csv(csv_dir / 'v11_badcase_train_cluster_members.tsv', sep='\t', index=False)
    write_fasta_from_members(train_rows, seqs, fasta_dir / 'v11_badcase_train_cluster_members.fa')

    bad_headers = members[(members['split'] == 'test') & (members['file_name'].isin(set(bad['badcase_name'])))]
    write_fasta_from_members(bad_headers, seqs, fasta_dir / 'v11_badcases.fa')

    comp_counts = clusters['composition'].value_counts().rename_axis('composition').reset_index(name='cluster_count')
    comp_counts.to_csv(csv_dir / 'cluster_composition_counts.tsv', sep='\t', index=False)

    plot_outputs(clusters, members, bad_table, fig_dir)
    write_summary_report(args, clusters, members, bad_table, bad_links, args.out_dir)

    print('\nDone.')
    print(f'clusters: {len(clusters)}')
    print(f'members: {len(members)}')
    print(f'v11 bad cases: {len(bad_table)}')
    print(f'bad cases with train-covered cluster: {int(bad_table["train_covered_cluster"].sum())}')
    print(f'output: {args.out_dir}')


if __name__ == '__main__':
    main()
