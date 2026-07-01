# bpRNA 数据集使用 CD-HIT 聚类与相似样本检索指南

本文档说明如何使用仓库内的 `cd-hit-v4.8.1-2019-0228` 对 `bpRNA` 训练数据做序列聚类、检查 `train` 是否覆盖 `test`，以及给定某条序列时如何在 `train` 中查找相似样本。

## 1. 当前路径与数据说明

仓库路径：

```bash
/root/aigame/dannyyan/PriFold
```

CD-HIT 源码路径：

```bash
/root/aigame/dannyyan/PriFold/cd-hit-v4.8.1-2019-0228
```

你的 bpRNA CSV：

```bash
/root/aigame/dannyyan/PriFold/data/bprna/bpRNA_.csv
```

`bpRNA_.csv` 字段：

| 字段 | 说明 |
|---|---|
| `Unnamed: 0` | CSV 保存时带出的行号/index |
| `data_name` | 数据划分：`TR0` / `VL0` / `TS0` |
| `file_name` | 样本 ID，也对应 contact map 文件名 |
| `seq` | RNA 序列 |
| `dot_string` | dot-bracket 二级结构 |
| `seq_len` | 序列长度 |

split 含义：

| `data_name` | 含义 |
|---|---|
| `TR0` | train |
| `VL0` | validation |
| `TS0` | test |

注意：项目主训练代码当前读取的是 `data/bprna/bpRNA.csv`，不是 `bpRNA_.csv`。如果你要严格检查训练实际使用的数据，建议确认 `bpRNA.csv` 和 `bpRNA_.csv` 是否一致；本文命令默认按你指定的 `bpRNA_.csv` 处理。

## 2. CD-HIT 应该用哪个程序

bpRNA 是 RNA/nucleotide 序列，所以不要用蛋白版 `cd-hit`，应使用：

| 目标 | 程序 |
|---|---|
| 对一个 FASTA 做聚类/去冗余 | `cd-hit-est` |
| 比较两个 FASTA，例如 `train` vs `test` | `cd-hit-est-2d` |
| 给定 query，查 train 中相似样本 | `cd-hit-est-2d`，`train` 作为 db1，query 作为 db2 |

CD-HIT 当前目录里主要是源码，如果还没有二进制，需要先编译。

## 3. 编译 CD-HIT

```bash
cd /root/aigame/dannyyan/PriFold/cd-hit-v4.8.1-2019-0228
make
```

编译后应出现这些可执行文件：

```bash
ls -lh cd-hit-est cd-hit-est-2d cd-hit
```

如果系统缺少 zlib 头文件，可安装 `zlib1g-dev`，或临时无 zlib 编译：

```bash
make zlib=no
```

## 4. 从 bpRNA CSV 导出 FASTA

CD-HIT 输入需要 FASTA。建议把 `U` 统一转成 `T`，与项目数据加载逻辑保持一致，也避免部分 nucleotide 工具对 `U` 支持不稳定。

```bash
cd /root/aigame/dannyyan/PriFold
mkdir -p /root/aigame/dannyyan/PriFold/data/bprna_cd_hit

python - <<'PY'
import pandas as pd
from pathlib import Path

csv_path = Path('/root/aigame/dannyyan/PriFold/data/bprna/bpRNA_.csv')
out_dir = Path('/root/aigame/dannyyan/PriFold/data/bprna_cd_hit')
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
df['seq'] = df['seq'].astype(str).str.upper().str.replace('U', 'T', regex=False)

# 如果想和项目训练逻辑一致，保留长度 < 490 的样本。
df = df[df['seq'].str.len() < 490].copy()

split_map = {
    'TR0': 'train',
    'VL0': 'val',
    'TS0': 'test',
}

def write_fasta(sub_df, path):
    with open(path, 'w') as f:
        for _, row in sub_df.iterrows():
            header = f"{row['data_name']}|{row['file_name']}|len={len(row['seq'])}"
            f.write(f">{header}\n")
            seq = row['seq']
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')

for data_name, split in split_map.items():
    sub = df[df['data_name'] == data_name].copy()
    write_fasta(sub, out_dir / f'bprna_{split}.fa')
    print(split, len(sub), out_dir / f'bprna_{split}.fa')

write_fasta(df, out_dir / 'bprna_all.fa')
print('all', len(df), out_dir / 'bprna_all.fa')
PY
```

输出：

```bash
symfold/outputs/bprna_cd_hit/bprna_train.fa
symfold/outputs/bprna_cd_hit/bprna_val.fa
symfold/outputs/bprna_cd_hit/bprna_test.fa
symfold/outputs/bprna_cd_hit/bprna_all.fa
```

## 5. 关键参数怎么选

常用参数：

| 参数 | 建议 | 说明 |
|---|---|---|
| `-c` | `0.95` / `0.90` / `1.00` | identity 阈值 |
| `-n` | 随 `-c` 改 | word size，必须匹配 identity 阈值 |
| `-d 0` | 推荐 | `.clstr` 中保留完整 FASTA header |
| `-M 16000` | 按机器内存改 | 内存 MB；`0` 表示不限制 |
| `-T 8` | 按 CPU 改 | 线程数；`0` 表示全部线程 |
| `-g 1` | 推荐用于分析 | 更精确但更慢 |
| `-r 0` | 推荐用于 bpRNA | 只比较同方向；RNA 结构任务中方向通常有意义 |
| `-aS 0.9` | 推荐用于泄漏检查 | 短序列覆盖率，避免局部片段误判 |
| `-aL 0.9` | 推荐用于泄漏检查 | 长序列覆盖率，避免局部片段误判 |

`-c` 与 `-n` 推荐关系：

| identity `-c` | 推荐 `-n` |
|---|---|
| `0.95 ~ 1.00` | `10` 或 `11` |
| `0.90 ~ 0.95` | `8` 或 `9` |
| `0.88 ~ 0.90` | `7` |
| `0.85 ~ 0.88` | `6` |
| `0.80 ~ 0.85` | `5` |
| `0.75 ~ 0.80` | `4` |

建议先跑三个阈值：

| 目的 | 参数 |
|---|---|
| 精确重复/近似完全重复 | `-c 1.00 -n 10` |
| 高相似泄漏检查 | `-c 0.95 -n 10` |
| 更宽松的家族级相似检查 | `-c 0.90 -n 8` |

## 6. 对全量 bpRNA 做聚类

用途：看整个 bpRNA 数据中有多少冗余簇、代表序列和簇大小。

```bash
CDHIT=/root/aigame/dannyyan/PriFold/cd-hit-v4.8.1-2019-0228
OUT=/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit

$CDHIT/cd-hit-est \
  -i $OUT/bprna_all.fa \
  -o $OUT/bprna_all_nr95.fa \
  -c 0.95 \
  -n 10 \
  -d 0 \
  -M 16000 \
  -T 8 \
  -g 1 \
  -r 0 \
  -aS 0.9 \
  -aL 0.9
```

主要输出：

```bash
symfold/outputs/bprna_cd_hit/bprna_all_nr95.fa
symfold/outputs/bprna_cd_hit/bprna_all_nr95.fa.clstr
```

其中：

- `bprna_all_nr95.fa`：每个 cluster 的代表序列。
- `bprna_all_nr95.fa.clstr`：每个 cluster 包含哪些样本。

## 7. 检查 train 是否覆盖 test

这是你最关心的问题。使用 `cd-hit-est-2d`：

- db1：`train.fa`
- db2：`test.fa`
- 输出 FASTA：`test` 中“不相似于 train”的样本
- `.clstr`：记录 `test` 中哪些样本被 `train` 命中

### 7.1 95% identity 检查

```bash
CDHIT=/root/aigame/dannyyan/PriFold/cd-hit-v4.8.1-2019-0228
OUT=/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit

$CDHIT/cd-hit-est-2d \
  -i $OUT/bprna_train.fa \
  -i2 $OUT/bprna_test.fa \
  -o $OUT/bprna_test_novel_vs_train_c095.fa \
  -c 0.95 \
  -n 10 \
  -d 0 \
  -M 16000 \
  -T 8 \
  -g 1 \
  -r 0 \
  -aS 0.9 \
  -aL 0.9
```

输出：

```bash
symfold/outputs/bprna_cd_hit/bprna_test_novel_vs_train_c095.fa
symfold/outputs/bprna_cd_hit/bprna_test_novel_vs_train_c095.fa.clstr
```

解释：

- `bprna_test_novel_vs_train_c095.fa`：没有被 train 以 95% identity 覆盖的 test 样本。
- `bprna_test_novel_vs_train_c095.fa.clstr`：被 train 覆盖/命中的 test 样本及对应 cluster。

如果你要更严格检查重复：

```bash
$CDHIT/cd-hit-est-2d \
  -i $OUT/bprna_train.fa \
  -i2 $OUT/bprna_test.fa \
  -o $OUT/bprna_test_novel_vs_train_c100.fa \
  -c 1.00 \
  -n 10 \
  -d 0 \
  -M 16000 \
  -T 8 \
  -g 1 \
  -r 0 \
  -aS 1.0 \
  -aL 1.0
```

如果你要更宽松检查潜在同源/家族相似：

```bash
$CDHIT/cd-hit-est-2d \
  -i $OUT/bprna_train.fa \
  -i2 $OUT/bprna_test.fa \
  -o $OUT/bprna_test_novel_vs_train_c090.fa \
  -c 0.90 \
  -n 8 \
  -d 0 \
  -M 16000 \
  -T 8 \
  -g 1 \
  -r 0 \
  -aS 0.9 \
  -aL 0.9
```

### 7.2 快速统计覆盖数量

CD-HIT 的 `.clstr` 是文本格式。最直接的判断方式：打开 `.clstr`，查看每个 cluster 里是否同时有 `TR0|...` 和 `TS0|...`。

下面脚本会解析 `.clstr`，输出每个被覆盖的 test 样本对应哪些 train 样本：

```bash
python - <<'PY'
from pathlib import Path
import re

clstr = Path('/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit/bprna_test_novel_vs_train_c095.fa.clstr')
out_tsv = clstr.with_suffix(clstr.suffix + '.covered_test.tsv')

clusters = []
cur = []
for line in clstr.read_text().splitlines():
    if line.startswith('>Cluster'):
        if cur:
            clusters.append(cur)
        cur = []
    else:
        m = re.search(r'>([^\.\s]+)', line)
        if m:
            cur.append(m.group(1))
if cur:
    clusters.append(cur)

rows = []
for ci, items in enumerate(clusters):
    train = [x for x in items if x.startswith('TR0|')]
    test = [x for x in items if x.startswith('TS0|')]
    for t in test:
        rows.append((ci, t, ';'.join(train)))

with open(out_tsv, 'w') as f:
    f.write('cluster_id\ttest_sample\ttrain_hits\n')
    for row in rows:
        f.write('\t'.join(map(str, row)) + '\n')

print('covered test samples:', len(rows))
print('written:', out_tsv)
PY
```

输出：

```bash
symfold/outputs/bprna_cd_hit/bprna_test_novel_vs_train_c095.fa.clstr.covered_test.tsv
```

这个 TSV 就是“哪些 test 样本在 train 中有相似样本”的清单。

## 8. 给定某条序列，查 train 中相似样本

思路：把 query 序列写成一个小 FASTA，然后用 `cd-hit-est-2d` 比较：

- db1：`bprna_train.fa`
- db2：`query.fa`
- 如果 query 被 train 命中，`.clstr` 里会出现该 query 和对应 train cluster。
- 如果 query 没有被命中，它会保留在输出 `query_novel.fa` 中。

### 8.1 写 query FASTA

把下面 `QUERY_SEQ` 换成你的序列即可，支持输入 RNA 的 `U`，脚本会转成 `T`。

```bash
OUT=/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit
QUERY_SEQ='AUGCUUAGCUAGCUAGCUAGCUAGCUAGC'

python - <<PY
from pathlib import Path
seq = '${QUERY_SEQ}'.upper().replace('U', 'T')
out = Path('${OUT}') / 'query.fa'
with open(out, 'w') as f:
    f.write('>QUERY|user_input|len=%d\n' % len(seq))
    for i in range(0, len(seq), 80):
        f.write(seq[i:i+80] + '\n')
print(out)
PY
```

### 8.2 查找 train 中相似样本

以 95% identity 为例：

```bash
CDHIT=/root/aigame/dannyyan/PriFold/cd-hit-v4.8.1-2019-0228
OUT=/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit

$CDHIT/cd-hit-est-2d \
  -i $OUT/bprna_train.fa \
  -i2 $OUT/query.fa \
  -o $OUT/query_novel_vs_train_c095.fa \
  -c 0.95 \
  -n 10 \
  -d 0 \
  -M 16000 \
  -T 8 \
  -g 1 \
  -r 0 \
  -aS 0.9 \
  -aL 0.9
```

查看结果：

```bash
cat $OUT/query_novel_vs_train_c095.fa.clstr
```

判断：

- `.clstr` 中出现 `QUERY|user_input` 且同 cluster 有 `TR0|...`：说明 train 里找到相似样本。
- `query_novel_vs_train_c095.fa` 中仍包含 query：说明在该阈值下没有找到相似 train 样本。

如果你想查更宽松相似样本，把阈值改成 `0.90`，同时 `-n 8`：

```bash
$CDHIT/cd-hit-est-2d \
  -i $OUT/bprna_train.fa \
  -i2 $OUT/query.fa \
  -o $OUT/query_novel_vs_train_c090.fa \
  -c 0.90 \
  -n 8 \
  -d 0 \
  -M 16000 \
  -T 8 \
  -g 1 \
  -r 0 \
  -aS 0.9 \
  -aL 0.9
```

## 9. 其他实用功能

### 9.1 只对 train 做聚类，得到 train 代表集

```bash
CDHIT=/root/aigame/dannyyan/PriFold/cd-hit-v4.8.1-2019-0228
OUT=/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit

$CDHIT/cd-hit-est \
  -i $OUT/bprna_train.fa \
  -o $OUT/bprna_train_nr95.fa \
  -c 0.95 \
  -n 10 \
  -d 0 \
  -M 16000 \
  -T 8 \
  -g 1 \
  -r 0 \
  -aS 0.9 \
  -aL 0.9
```

用途：

- 得到去冗余后的 train 代表序列。
- 统计 train 中有多少重复/近重复簇。
- 如果要训练去冗余版本，可以只保留代表序列对应的 `file_name`。

### 9.2 train vs val 泄漏检查

```bash
$CDHIT/cd-hit-est-2d \
  -i $OUT/bprna_train.fa \
  -i2 $OUT/bprna_val.fa \
  -o $OUT/bprna_val_novel_vs_train_c095.fa \
  -c 0.95 \
  -n 10 \
  -d 0 \
  -M 16000 \
  -T 8 \
  -g 1 \
  -r 0 \
  -aS 0.9 \
  -aL 0.9
```

### 9.3 train+val vs test 检查

如果最终训练使用了 train+val，建议合并后检查：

```bash
OUT=/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit
cat $OUT/bprna_train.fa $OUT/bprna_val.fa > $OUT/bprna_train_val.fa

$CDHIT/cd-hit-est-2d \
  -i $OUT/bprna_train_val.fa \
  -i2 $OUT/bprna_test.fa \
  -o $OUT/bprna_test_novel_vs_train_val_c095.fa \
  -c 0.95 \
  -n 10 \
  -d 0 \
  -M 16000 \
  -T 8 \
  -g 1 \
  -r 0 \
  -aS 0.9 \
  -aL 0.9
```

### 9.4 生成 cluster 大小统计

```bash
python - <<'PY'
from pathlib import Path

clstr = Path('/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit/bprna_all_nr95.fa.clstr')
sizes = []
cur = 0
for line in clstr.read_text().splitlines():
    if line.startswith('>Cluster'):
        if cur:
            sizes.append(cur)
        cur = 0
    else:
        cur += 1
if cur:
    sizes.append(cur)

print('clusters:', len(sizes))
print('sequences:', sum(sizes))
print('max_cluster_size:', max(sizes) if sizes else 0)
print('singleton_clusters:', sum(1 for x in sizes if x == 1))
print('non_singleton_clusters:', sum(1 for x in sizes if x > 1))
PY
```

## 10. 推荐实验顺序

建议按下面顺序做：

1. 编译 CD-HIT。
2. 从 `bpRNA_.csv` 导出 `train/val/test/all` FASTA。
3. 先跑 `train vs test` 的 `-c 1.00`，检查精确重复。
4. 再跑 `train vs test` 的 `-c 0.95`，检查高相似泄漏。
5. 可选跑 `-c 0.90`，检查更宽松的潜在同源/家族相似。
6. 对 `train` 或 `all` 做 `cd-hit-est` 聚类，观察冗余簇大小。
7. 对你关心的单条 query，用 `cd-hit-est-2d` 查 train 中相似样本。

## 11. 注意事项

1. `cd-hit-est` / `cd-hit-est-2d` 主要基于序列 identity，不考虑 RNA 二级结构相似性。
2. `-r 0` 表示只比较同方向；如果你希望反向互补也算相似，可以改成 `-r 1`。
3. `-aS` / `-aL` 很重要。没有 coverage 限制时，短局部片段相似可能导致误判。
4. `-c < 0.75` 时不建议继续用普通 `cd-hit-est` 做可靠聚类；低相似度检索可考虑 BLAST/MMseqs2 等工具。
5. 本项目训练代码会过滤 `seq` 长度 `< 490`，如果你要检查训练实际数据，也应使用同样过滤条件。
6. 如果要和项目现有训练完全一致，请优先核对 `data/bprna/bpRNA.csv` 与 `data/bprna/bpRNA_.csv` 的差异。

## 12. 本次实际分析脚本与结果

本次已新增可复现脚本：

```bash
/root/aigame/dannyyan/PriFold/symfold/eval/analyze_bprna_cd_hit_clusters.py
```

脚本功能：

1. 读取 `/root/aigame/dannyyan/PriFold/data/bprna_cd_hit/bprna_all.fa`。
2. 调用 `cd-hit-est` 对全量 bpRNA 聚类。
3. 解析 `.clstr`，输出每个 cluster 中来自 `train/val/test` 的样本数。
4. 读取 v11 per-sample 结果 `/root/aigame/dannyyan/PriFold/symfold/outputs/v11/comprehensive_analysis/per_sample_results.json`。
5. 筛选 v11 bad case：`F1 < 0.3`。
6. 将 bad case 关联到 CD-HIT cluster。
7. 判断 bad case 所属 cluster 中是否存在 train 样本。
8. 导出同 cluster 的 train 样本 TSV 与 FASTA。
9. 生成可视化图和 Markdown 报告。

### 12.1 运行环境

本机 `conda` 路径：

```bash
/root/aigame/dannyyan/miniconda3/bin/conda
```

使用环境：

```bash
RNADiffFold_torch260
```

### 12.2 运行命令

95% identity：

```bash
cd /root/aigame/dannyyan/PriFold
/root/aigame/dannyyan/miniconda3/bin/conda run -n RNADiffFold_torch260 \
  python symfold/eval/analyze_bprna_cd_hit_clusters.py \
  --identity 0.95 \
  --word-size 10
```

90% identity：

```bash
cd /root/aigame/dannyyan/PriFold
/root/aigame/dannyyan/miniconda3/bin/conda run -n RNADiffFold_torch260 \
  python symfold/eval/analyze_bprna_cd_hit_clusters.py \
  --identity 0.90 \
  --word-size 8 \
  --out-dir symfold/outputs/bprna_cd_hit_analysis_c090
```

80% identity：

```bash
cd /root/aigame/dannyyan/PriFold
/root/aigame/dannyyan/miniconda3/bin/conda run -n RNADiffFold_torch260 \
  python symfold/eval/analyze_bprna_cd_hit_clusters.py \
  --identity 0.80 \
  --word-size 5 \
  --out-dir symfold/outputs/bprna_cd_hit_analysis_c080
```

默认使用：

```bash
-aS 0.9 -aL 0.9 -r 0 -g 1 -M 16000 -T 8
```

### 12.3 输出目录

95% identity：

```bash
/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit_analysis
```

90% identity：

```bash
/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit_analysis_c090
```

80% identity：

```bash
/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit_analysis_c080
```

每个输出目录中包含：

| 文件/目录 | 说明 |
|---|---|
| `analysis_report.md` | 自动生成的分析报告 |
| `csv/cluster_summary.tsv` | 每个 cluster 有多少 train/val/test 样本 |
| `csv/cluster_members.tsv` | 每个样本所属 cluster |
| `csv/v11_badcase_cluster_summary.tsv` | 每个 v11 bad case 所属 cluster 及 train 覆盖情况 |
| `csv/v11_badcase_train_cluster_members.tsv` | 与 bad case 同 cluster 的 train 样本明细 |
| `fasta/v11_badcase_train_cluster_members.fa` | 与 bad case 同 cluster 的 train 序列 |
| `fasta/v11_badcases.fa` | v11 bad case 序列 |
| `figures/` | 可视化图片 |
| `cdhit/` | CD-HIT 原始输出和 `.clstr` |

### 12.4 本次核心结果

| identity | clusters | singleton clusters | max cluster size | train-covered v11 bad cases |
|---:|---:|---:|---:|---:|
| 0.95 | 13409 | 13409 | 1 | 0 / 105 |
| 0.90 | 13409 | 13409 | 1 | 0 / 105 |
| 0.80 | 13347 | 13291 | 3 | 1 / 105 |

95% / 90% 下，全量 bpRNA 每条序列都是独立 cluster，因此 v11 bad case 没有被 train 高相似序列覆盖。

80% 下，有 1 个 v11 bad case 所属 cluster 中包含 train 样本：

| bad case | bad case F1 | cluster_id | train sample |
|---|---:|---:|---|
| `bpRNA_RFAM_23463` | 0.0 | 13098 | `bpRNA_RFAM_23488` |

对应导出文件：

```bash
/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit_analysis_c080/csv/v11_badcase_train_cluster_members.tsv
/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit_analysis_c080/fasta/v11_badcase_train_cluster_members.fa
```

### 12.5 可视化

每个分析目录下的 `figures/` 包含：

```bash
cluster_size_histogram.png
cluster_split_composition.png
cluster_train_test_scatter.png
badcase_train_coverage.png
badcase_f1_by_coverage.png
badcase_cluster_size_vs_f1.png
top_badcase_clusters_split_stack.png
```

推荐优先看：

```bash
/root/aigame/dannyyan/PriFold/symfold/outputs/bprna_cd_hit_analysis_c080/analysis_report.md
```

因为 80% identity 下才出现少量跨 split 聚类，图表更有信息量。

## 13. 一句话总结

- 对 bpRNA 单库聚类：用 `cd-hit-est`。
- 检查 `train` 是否覆盖 `test`：用 `cd-hit-est-2d -i train.fa -i2 test.fa`，或像本次脚本一样对 `all.fa` 聚类后统计 cluster split composition。
- 给定 query 查 train 相似样本：把 query 写成 FASTA，然后用 `cd-hit-est-2d -i train.fa -i2 query.fa`。
- 本次在 `95%` 和 `90%` identity 下没有发现 train 覆盖 v11 bad-case cluster；在 `80%` identity 下仅发现 `1/105` 个 v11 bad case 的 cluster 中有 train 样本。
