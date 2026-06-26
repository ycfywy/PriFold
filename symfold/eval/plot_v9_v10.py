# -*- coding: utf-8 -*-
"""绘制 v9 vs v10 对比图表，用于 v10_report.md"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

ROOT = Path('/root/aigame/dannyyan/PriFold')
CMP = ROOT / 'symfold/outputs/v9_v10_compare'
FIG_DIR = ROOT / 'docs/v10/figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

v9 = json.load(open(CMP / 'v9_per_sample.json'))
v10 = json.load(open(CMP / 'v10_per_sample.json'))

plt.rcParams['font.size'] = 11
COLOR_V9 = '#5B9BD5'
COLOR_V10 = '#ED7D31'

# ============================================================
# Fig 1: Train/Val/Test 总体对比 (柱状图)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

splits = ['train', 'val', 'test']
v9_f1 = [np.mean([r['f1'] for r in v9[s]]) for s in splits]
v10_f1 = [np.mean([r['f1'] for r in v10[s]]) for s in splits]

x = np.arange(len(splits))
w = 0.35
ax = axes[0]
b1 = ax.bar(x - w/2, v9_f1, w, label='v9 (MARS frozen)', color=COLOR_V9)
b2 = ax.bar(x + w/2, v10_f1, w, label='v10 (MARS unfrozen)', color=COLOR_V10)
ax.set_ylabel('F1 Score')
ax.set_title('(a) Overall F1 across Train/Val/Test')
ax.set_xticks(x); ax.set_xticklabels([s.upper() for s in splits])
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1.0)
for b in [b1, b2]:
    for rect in b:
        h = rect.get_height()
        ax.annotate(f'{h:.3f}', xy=(rect.get_x()+rect.get_width()/2, h),
                    xytext=(0,3), textcoords='offset points', ha='center', fontsize=9)

# 泛化 gap
ax = axes[1]
v9_gap = v9_f1[0] - v9_f1[2]
v10_gap = v10_f1[0] - v10_f1[2]
gaps = [v9_gap, v10_gap]
bars = ax.bar(['v9', 'v10'], gaps, color=[COLOR_V9, COLOR_V10], width=0.5)
ax.set_ylabel('Train F1 - Test F1 (Generalization Gap)')
ax.set_title('(b) Overfitting: Generalization Gap')
ax.grid(True, alpha=0.3, axis='y')
for rect, g in zip(bars, gaps):
    ax.annotate(f'{g:.3f}', xy=(rect.get_x()+rect.get_width()/2, g),
                xytext=(0,3), textcoords='offset points', ha='center')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_overall.png', dpi=130, bbox_inches='tight')
plt.close()
print('fig1 saved')

# ============================================================
# Fig 2: 各家族对比 (test)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
fam_v9 = defaultdict(list); fam_v10 = defaultdict(list)
for r in v9['test']: fam_v9[r['family']].append(r['f1'])
for r in v10['test']: fam_v10[r['family']].append(r['f1'])
fams = sorted(fam_v10.keys(), key=lambda f: -len(fam_v10[f]))
v9m = [np.mean(fam_v9[f]) for f in fams]
v10m = [np.mean(fam_v10[f]) for f in fams]
counts = [len(fam_v10[f]) for f in fams]

x = np.arange(len(fams))
b1 = ax.bar(x - w/2, v9m, w, label='v9', color=COLOR_V9)
b2 = ax.bar(x + w/2, v10m, w, label='v10', color=COLOR_V10)
ax.set_ylabel('Test F1')
ax.set_title('Per-Family Test F1: v9 vs v10')
ax.set_xticks(x)
ax.set_xticklabels([f'{f}\n(n={c})' for f, c in zip(fams, counts)])
ax.legend(); ax.grid(True, alpha=0.3, axis='y'); ax.set_ylim(0, 1.05)
for b in [b1, b2]:
    for rect in b:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}', xy=(rect.get_x()+rect.get_width()/2, h),
                    xytext=(0,2), textcoords='offset points', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_family.png', dpi=130, bbox_inches='tight')
plt.close()
print('fig2 saved')

# ============================================================
# Fig 3: 不同长度区间对比 (test)
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
buckets = [(0,50),(50,100),(100,150),(150,200),(200,300),(300,400),(400,500)]
v9_test = {r['name']: r for r in v9['test']}
v10_test = {r['name']: r for r in v10['test']}
common = set(v9_test) & set(v10_test)
labels, v9b, v10b, deltas = [], [], [], []
for lo, hi in buckets:
    names = [n for n in common if lo <= v10_test[n]['length'] < hi]
    if not names: continue
    labels.append(f'{lo}-{hi}')
    a = np.mean([v9_test[n]['f1'] for n in names])
    b = np.mean([v10_test[n]['f1'] for n in names])
    v9b.append(a); v10b.append(b); deltas.append(b-a)

x = np.arange(len(labels))
ax.plot(x, v9b, 'o-', color=COLOR_V9, label='v9', linewidth=2, markersize=7)
ax.plot(x, v10b, 's-', color=COLOR_V10, label='v10', linewidth=2, markersize=7)
for i, d in enumerate(deltas):
    color = 'green' if d > 0 else 'red'
    ax.annotate(f'{d:+.3f}', xy=(i, max(v9b[i], v10b[i])+0.02),
                ha='center', fontsize=9, color=color, fontweight='bold')
ax.set_ylabel('Test F1'); ax.set_xlabel('Sequence Length')
ax.set_title('Test F1 by Sequence Length: v9 vs v10 (Δ annotated)')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0.5, 0.85)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_length.png', dpi=130, bbox_inches='tight')
plt.close()
print('fig3 saved')

# ============================================================
# Fig 4: per-sample ΔF1 散点图 (test)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 散点: v9 F1 vs v10 F1
ax = axes[0]
v9_vals = [v9_test[n]['f1'] for n in common]
v10_vals = [v10_test[n]['f1'] for n in common]
ax.scatter(v9_vals, v10_vals, alpha=0.3, s=15, color='purple')
ax.plot([0,1],[0,1],'k--',alpha=0.5, label='y=x (no change)')
ax.set_xlabel('v9 F1'); ax.set_ylabel('v10 F1')
ax.set_title('(a) Per-sample F1: v9 vs v10\n(above line = improved)')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlim(0,1); ax.set_ylim(0,1)
# 统计上下方点数
improved = sum(1 for a,b in zip(v9_vals,v10_vals) if b>a+0.01)
worsened = sum(1 for a,b in zip(v9_vals,v10_vals) if b<a-0.01)
same = len(common)-improved-worsened
ax.text(0.05,0.95,f'Improved: {improved}\nWorsened: {worsened}\nSame: {same}',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# F1 分布直方图
ax = axes[1]
bins = np.linspace(0, 1, 21)
ax.hist(v9_vals, bins=bins, alpha=0.5, label='v9', color=COLOR_V9)
ax.hist(v10_vals, bins=bins, alpha=0.5, label='v10', color=COLOR_V10)
ax.axvline(0.3, color='red', linestyle='--', alpha=0.6, label='bad case threshold')
ax.set_xlabel('F1'); ax.set_ylabel('Sample Count')
ax.set_title('(b) Test F1 Distribution')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_scatter.png', dpi=130, bbox_inches='tight')
plt.close()
print('fig4 saved')

# ============================================================
# Fig 5: Bad case 流转 (Sankey-like / 柱状)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

v9_bad = {n for n in common if v9_test[n]['f1'] < 0.3}
v10_bad = {n for n in common if v10_test[n]['f1'] < 0.3}
fixed = len(v9_bad - v10_bad)
new_bad = len(v10_bad - v9_bad)
still_bad = len(v9_bad & v10_bad)

ax = axes[0]
cats = ['v9 bad\ntotal', 'v10 bad\ntotal']
vals = [len(v9_bad), len(v10_bad)]
bars = ax.bar(cats, vals, color=[COLOR_V9, COLOR_V10], width=0.5)
ax.set_ylabel('Bad Case Count (F1<0.3)')
ax.set_title('(a) Bad Case Count')
for rect, v in zip(bars, vals):
    ax.annotate(f'{v}', xy=(rect.get_x()+rect.get_width()/2, v),
                xytext=(0,3), textcoords='offset points', ha='center')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
trans = ['Fixed\n(v9 bad→v10 ok)', 'New bad\n(v9 ok→v10 bad)', 'Still bad\n(both)']
tvals = [fixed, new_bad, still_bad]
tcolors = ['green', 'red', 'gray']
bars = ax.bar(trans, tvals, color=tcolors, width=0.5, alpha=0.7)
ax.set_ylabel('Count')
ax.set_title(f'(b) Bad Case Transitions (overlap={still_bad/len(v9_bad|v10_bad)*100:.0f}%)')
for rect, v in zip(bars, tvals):
    ax.annotate(f'{v}', xy=(rect.get_x()+rect.get_width()/2, v),
                xytext=(0,3), textcoords='offset points', ha='center')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_badcase.png', dpi=130, bbox_inches='tight')
plt.close()
print('fig5 saved')

# ============================================================
# Fig 6: 改善样本的家族/长度分布
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
improved_samples = [(n, v10_test[n]['f1']-v9_test[n]['f1']) for n in common if v10_test[n]['f1']-v9_test[n]['f1'] > 0.1]
imp_fam = defaultdict(int); imp_len = defaultdict(int)
for n, d in improved_samples:
    imp_fam[v10_test[n]['family']] += 1
    L = v10_test[n]['length']
    bucket = '0-100' if L<100 else '100-200' if L<200 else '200-300' if L<300 else '300+'
    imp_len[bucket] += 1

ax = axes[0]
fams_i = sorted(imp_fam.keys(), key=lambda f:-imp_fam[f])
ax.bar(fams_i, [imp_fam[f] for f in fams_i], color='green', alpha=0.7)
ax.set_ylabel('Count'); ax.set_title(f'(a) Improved Samples by Family (ΔF1>0.1, total={len(improved_samples)})')
ax.grid(True, alpha=0.3, axis='y')
for i, f in enumerate(fams_i):
    ax.annotate(f'{imp_fam[f]}', xy=(i, imp_fam[f]), xytext=(0,3), textcoords='offset points', ha='center')

ax = axes[1]
len_order = ['0-100','100-200','200-300','300+']
len_order = [l for l in len_order if l in imp_len]
ax.bar(len_order, [imp_len[l] for l in len_order], color='teal', alpha=0.7)
ax.set_ylabel('Count'); ax.set_title('(b) Improved Samples by Length')
ax.grid(True, alpha=0.3, axis='y')
for i, l in enumerate(len_order):
    ax.annotate(f'{imp_len[l]}', xy=(i, imp_len[l]), xytext=(0,3), textcoords='offset points', ha='center')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig6_improved_dist.png', dpi=130, bbox_inches='tight')
plt.close()
print('fig6 saved')

print('\nAll figures saved to', FIG_DIR)
print(f'Stats: improved={improved}, worsened={worsened}, fixed={fixed}, new_bad={new_bad}, still_bad={still_bad}')
