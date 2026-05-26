"""
统计 PriFold 数据集分布并绘图。

不依赖 matplotlib，直接生成 SVG 图片。

运行:
    python examples/analyze_data_distribution.py \
        --data_dir ./data \
        --output_dir outputs/20260525_1851_data_distribution \
        --docs_path docs/data_distribution_report.md
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


MAX_LEN = 490
COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1"]


def normalize_seq(seq: str) -> str:
    return str(seq).upper().replace("U", "T")


def md_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    cols = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.{float_digits}f}")
            else:
                values.append(str(value))
        rows.append(values)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(values) + " |" for values in rows]
    return "\n".join([header, sep, *body])


def load_tables(data_dir: Path) -> pd.DataFrame:
    records: list[pd.DataFrame] = []

    bprna = pd.read_csv(data_dir / "bprna" / "bpRNA.csv")
    bprna = bprna[bprna["seq"].str.len() < MAX_LEN].copy()
    bprna["dataset"] = "bpRNA"
    bprna["split"] = bprna["data_name"].map({"TR0": "train", "VL0": "val", "TS0": "test"})
    bprna["ct_path"] = bprna.apply(
        lambda row: data_dir / "bprna" / "ct" / str(row["data_name"]) / f"{row['file_name']}.npy",
        axis=1,
    )
    records.append(bprna[["dataset", "split", "file_name", "seq", "ct_path"]])

    rnastr = pd.read_csv(data_dir / "RNAStrAlign" / "rnastralign.csv")
    rnastr = rnastr[rnastr["seq"].str.len() < MAX_LEN].copy()
    rnastr["dataset"] = "RNAStrAlign"
    # 当前 utils/tools.py 的 rnastralign 模式只使用 tr 和 ts；vl 存在于 CSV 中，但主训练/推理入口没有使用。
    rnastr["split"] = rnastr["data_name"].map({"tr": "train", "ts": "val/test", "vl": "unused(vl)"})
    rnastr["ct_path"] = rnastr.apply(
        lambda row: data_dir / "RNAStrAlign" / f"{row['file_name']}.npy",
        axis=1,
    )
    records.append(rnastr[["dataset", "split", "file_name", "seq", "ct_path"]])

    archive = pd.read_csv(data_dir / "archiveII" / "archiveII.csv")
    archive = archive[archive["seq"].str.len() < MAX_LEN].copy()
    archive["dataset"] = "ArchiveII"
    archive["split"] = "test"
    archive["ct_path"] = archive.apply(
        lambda row: data_dir / "archiveII" / "ct" / f"{row['file_name']}.npy",
        axis=1,
    )
    records.append(archive[["dataset", "split", "file_name", "seq", "ct_path"]])

    df = pd.concat(records, ignore_index=True)
    df["seq"] = df["seq"].astype(str)
    df["length"] = df["seq"].str.len()
    df["normalized_seq"] = df["seq"].map(normalize_seq)
    return df


def add_contact_map_stats(df: pd.DataFrame) -> pd.DataFrame:
    pair_counts: list[float] = []
    densities: list[float] = []
    shape_ok: list[bool] = []
    missing: list[bool] = []

    for _, row in df.iterrows():
        ct_path = Path(row["ct_path"])
        if not ct_path.exists():
            pair_counts.append(np.nan)
            densities.append(np.nan)
            shape_ok.append(False)
            missing.append(True)
            continue

        ct = np.load(ct_path)
        length = int(row["length"])
        shape_ok.append(ct.shape == (length, length))
        missing.append(False)

        pair_count = float(np.sum(ct) / 2.0)
        pair_counts.append(pair_count)
        densities.append(pair_count / max(length, 1))

    df = df.copy()
    df["pair_count"] = pair_counts
    df["pair_density_per_base"] = densities
    df["ct_shape_ok"] = shape_ok
    df["ct_missing"] = missing
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["dataset", "split"], dropna=False)
    summary = grouped.agg(
        n=("length", "size"),
        len_min=("length", "min"),
        len_p25=("length", lambda x: np.percentile(x, 25)),
        len_mean=("length", "mean"),
        len_median=("length", "median"),
        len_p75=("length", lambda x: np.percentile(x, 75)),
        len_max=("length", "max"),
        pair_mean=("pair_count", "mean"),
        pair_median=("pair_count", "median"),
        density_mean=("pair_density_per_base", "mean"),
        density_median=("pair_density_per_base", "median"),
        ct_missing=("ct_missing", "sum"),
        ct_shape_bad=("ct_shape_ok", lambda x: int((~x).sum())),
    )
    return summary.reset_index()


def base_composition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, split), part in df.groupby(["dataset", "split"]):
        seq = "".join(part["normalized_seq"].tolist())
        total = max(len(seq), 1)
        for base in ["A", "T", "G", "C", "N"]:
            rows.append({"dataset": dataset, "split": split, "base": base, "count": seq.count(base), "fraction": seq.count(base) / total})
    return pd.DataFrame(rows)


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="#333">{html.escape(text)}</text>'


def save_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.extend(body)
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def scale_fn(vmin: float, vmax: float, out_min: float, out_max: float):
    if vmax == vmin:
        return lambda _: (out_min + out_max) / 2
    return lambda value: out_min + (value - vmin) / (vmax - vmin) * (out_max - out_min)


def group_items(df: pd.DataFrame):
    return list(df.groupby(["dataset", "split"]))


def plot_length_hist(df: pd.DataFrame, output_dir: Path) -> None:
    width, height = 1000, 560
    left, right, top, bottom = 70, 220, 50, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    bins = np.linspace(0, MAX_LEN, 50)
    groups = group_items(df)
    max_count = 1
    hists = []
    for key, part in groups:
        counts, edges = np.histogram(part["length"], bins=bins)
        max_count = max(max_count, int(counts.max()))
        hists.append((key, counts, edges))

    xscale = scale_fn(0, MAX_LEN, left, left + plot_w)
    yscale = scale_fn(0, max_count, top + plot_h, top)
    body = [svg_text(width / 2, 28, "Sequence Length Distribution", 18)]
    body.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>')
    body.append(svg_text(left + plot_w / 2, height - 22, "RNA sequence length", 13))
    body.append(svg_text(18, top + plot_h / 2, "Count", 13, anchor="middle"))

    for idx, (key, counts, edges) in enumerate(hists):
        color = COLORS[idx % len(COLORS)]
        points = []
        centers = (edges[:-1] + edges[1:]) / 2
        for x, y in zip(centers, counts):
            points.append(f"{xscale(float(x)):.1f},{yscale(float(y)):.1f}")
        body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" opacity="0.9"/>')
        ly = top + 20 + idx * 22
        body.append(f'<rect x="{left+plot_w+25}" y="{ly-10}" width="14" height="14" fill="{color}"/>')
        body.append(svg_text(left + plot_w + 45, ly + 2, f"{key[0]}:{key[1]}", 12, anchor="start"))
    save_svg(output_dir / "length_hist_by_split.svg", width, height, body)


def plot_length_box(df: pd.DataFrame, output_dir: Path) -> None:
    width, height = 1000, 560
    left, right, top, bottom = 70, 40, 50, 120
    plot_w, plot_h = width - left - right, height - top - bottom
    groups = group_items(df)
    yscale = scale_fn(0, MAX_LEN, top + plot_h, top)
    x_gap = plot_w / max(len(groups), 1)
    body = [svg_text(width / 2, 28, "Length Distribution Boxplot", 18)]
    body.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>')
    for idx, (key, part) in enumerate(groups):
        vals = part["length"].to_numpy()
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        vmin, vmax = np.min(vals), np.max(vals)
        x = left + x_gap * (idx + 0.5)
        box_w = min(70, x_gap * 0.55)
        color = COLORS[idx % len(COLORS)]
        body.append(f'<line x1="{x}" y1="{yscale(vmin)}" x2="{x}" y2="{yscale(vmax)}" stroke="{color}" stroke-width="2"/>')
        body.append(f'<rect x="{x-box_w/2}" y="{yscale(q3)}" width="{box_w}" height="{yscale(q1)-yscale(q3)}" fill="{color}" opacity="0.35" stroke="{color}"/>')
        body.append(f'<line x1="{x-box_w/2}" y1="{yscale(med)}" x2="{x+box_w/2}" y2="{yscale(med)}" stroke="{color}" stroke-width="3"/>')
        label = f"{key[0]}\n{key[1]}"
        body.append(svg_text(x, top + plot_h + 24, label.replace("\n", " "), 11))
    save_svg(output_dir / "length_boxplot.svg", width, height, body)


def plot_base_composition(comp: pd.DataFrame, output_dir: Path) -> None:
    width, height = 1000, 560
    left, right, top, bottom = 70, 150, 50, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    bases = ["A", "T", "G", "C", "N"]
    base_colors = dict(zip(bases, COLORS))
    groups = list(comp.groupby(["dataset", "split"]))
    x_gap = plot_w / max(len(groups), 1)
    body = [svg_text(width / 2, 28, "Base Composition (U normalized to T)", 18)]
    body.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>')
    for idx, (key, part) in enumerate(groups):
        x = left + x_gap * idx + x_gap * 0.2
        bar_w = x_gap * 0.6
        y = top + plot_h
        part_map = dict(zip(part["base"], part["fraction"]))
        for base in bases:
            frac = float(part_map.get(base, 0.0))
            h = frac * plot_h
            y -= h
            body.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{h}" fill="{base_colors[base]}"/>')
        body.append(svg_text(x + bar_w / 2, top + plot_h + 24, f"{key[0]}:{key[1]}", 11))
    for i, base in enumerate(bases):
        ly = top + 20 + i * 24
        body.append(f'<rect x="{left+plot_w+30}" y="{ly-12}" width="14" height="14" fill="{base_colors[base]}"/>')
        body.append(svg_text(left + plot_w + 52, ly, base, 12, anchor="start"))
    save_svg(output_dir / "base_composition.svg", width, height, body)


def plot_pair_stats(df: pd.DataFrame, output_dir: Path) -> None:
    width, height = 1000, 620
    left, right, top, bottom = 70, 220, 50, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    max_pair = float(np.nanmax(df["pair_count"]))
    xscale = scale_fn(0, MAX_LEN, left, left + plot_w)
    yscale = scale_fn(0, max_pair, top + plot_h, top)
    body = [svg_text(width / 2, 28, "Base Pair Count vs Sequence Length", 18)]
    body.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>')
    rng = np.random.default_rng(3407)
    for idx, (key, part) in enumerate(group_items(df)):
        color = COLORS[idx % len(COLORS)]
        sample = part.dropna(subset=["pair_count"])
        if len(sample) > 2500:
            sample = sample.iloc[rng.choice(len(sample), 2500, replace=False)]
        for _, row in sample.iterrows():
            body.append(f'<circle cx="{xscale(float(row["length"])):.1f}" cy="{yscale(float(row["pair_count"])):.1f}" r="1.5" fill="{color}" opacity="0.35"/>')
        ly = top + 20 + idx * 22
        body.append(f'<rect x="{left+plot_w+25}" y="{ly-10}" width="14" height="14" fill="{color}"/>')
        body.append(svg_text(left + plot_w + 45, ly + 2, f"{key[0]}:{key[1]}", 12, anchor="start"))
    save_svg(output_dir / "pair_count_vs_length.svg", width, height, body)

    width, height = 1000, 560
    left, right, top, bottom = 70, 220, 50, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    max_density = float(np.nanmax(df["pair_density_per_base"]))
    bins = np.linspace(0, max_density, 45)
    groups = group_items(df)
    hists = []
    max_count = 1
    for key, part in groups:
        counts, edges = np.histogram(part["pair_density_per_base"].dropna(), bins=bins)
        max_count = max(max_count, int(counts.max()))
        hists.append((key, counts, edges))
    xscale = scale_fn(0, max_density, left, left + plot_w)
    yscale = scale_fn(0, max_count, top + plot_h, top)
    body = [svg_text(width / 2, 28, "Contact Density Distribution", 18)]
    body.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>')
    for idx, (key, counts, edges) in enumerate(hists):
        color = COLORS[idx % len(COLORS)]
        centers = (edges[:-1] + edges[1:]) / 2
        points = [f"{xscale(float(x)):.1f},{yscale(float(y)):.1f}" for x, y in zip(centers, counts)]
        body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
        ly = top + 20 + idx * 22
        body.append(f'<rect x="{left+plot_w+25}" y="{ly-10}" width="14" height="14" fill="{color}"/>')
        body.append(svg_text(left + plot_w + 45, ly + 2, f"{key[0]}:{key[1]}", 12, anchor="start"))
    save_svg(output_dir / "contact_density_hist.svg", width, height, body)


def write_markdown_report(summary: pd.DataFrame, comp: pd.DataFrame, output_dir: Path, docs_path: Path) -> None:
    def rel(name: str) -> str:
        return os.path.relpath(output_dir / name, docs_path.parent)

    comp_table = comp.pivot_table(index=["dataset", "split"], columns="base", values="fraction").fillna(0).reset_index()
    lines = [
        "# PriFold 数据分布统计",
        "",
        "本文档由 `examples/analyze_data_distribution.py` 生成，统计 `data/` 下被当前 `train.py` / `inference.py` 使用的数据。",
        "",
        "## 数据集划分统计",
        "",
        md_table(summary),
        "",
        "## 碱基组成比例",
        "",
        md_table(comp_table),
        "",
        "## 可视化",
        "",
        f"![Length Histogram]({rel('length_hist_by_split.svg')})",
        "",
        f"![Length Boxplot]({rel('length_boxplot.svg')})",
        "",
        f"![Base Composition]({rel('base_composition.svg')})",
        "",
        f"![Pair Count vs Length]({rel('pair_count_vs_length.svg')})",
        "",
        f"![Contact Density]({rel('contact_density_hist.svg')})",
        "",
        "## 当前项目如何使用这些数据",
        "",
        "### 训练 `--mode bprna`",
        "",
        "- CSV: `data/bprna/bpRNA.csv`",
        "- 过滤: `seq` 长度 `< 490`",
        "- 训练: `data_name == TR0`，contact map 在 `data/bprna/ct/TR0/{file_name}.npy`",
        "- 验证: `data_name == VL0`，contact map 在 `data/bprna/ct/VL0/{file_name}.npy`",
        "- 测试: `data_name == TS0`，contact map 在 `data/bprna/ct/TS0/{file_name}.npy`",
        "",
        "### 训练 `--mode rnastralign`",
        "",
        "- CSV: `data/RNAStrAlign/rnastralign.csv` 与 `data/archiveII/archiveII.csv`",
        "- 过滤: `seq` 长度 `< 490`",
        "- 训练: RNAStrAlign 中 `data_name == tr`，contact map 在 `data/RNAStrAlign/{file_name}.npy`",
        "- 验证: RNAStrAlign 中 `data_name == ts`，contact map 在 `data/RNAStrAlign/{file_name}.npy`",
        "- 测试: ArchiveII 全部样本，contact map 在 `data/archiveII/ct/{file_name}.npy`",
        "",
        "### 推理测试",
        "",
        "- `--mode bprna-test`: 使用 bpRNA 的 `TS0`",
        "- `--mode rnastralign-test`: 使用 RNAStrAlign 的 `ts`",
        "- `--mode archiveii-test`: 使用 ArchiveII 全部样本",
        "",
        "### Batch 中的数据结构",
        "",
        "`train.py` 和 `inference.py` 的 `collate_fn` 会把单样本 `(seq, ct, _)` 组装为:",
        "",
        "```python",
        "{",
        "  'input_ids': Tensor[B, max_len],",
        "  'attention_mask': Tensor[B, max_len],",
        "  'pos_bias': Tensor[B, max_len, max_len],",
        "  'ct': FloatTensor[B, max_len, max_len],",
        "  'ct_mask': FloatTensor[B, max_len, max_len],",
        "  'seq_len': Tensor[B],",
        "}",
        "```",
        "",
        "其中 `max_len = max(len(seq) + 2)`，`+2` 是 tokenizer 特殊 token 位置；`ct` 是二级结构 contact map 标签。",
    ]
    docs_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("./data"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/data_distribution"))
    parser.add_argument("--docs_path", type=Path, default=Path("docs/data_distribution_report.md"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_path.parent.mkdir(parents=True, exist_ok=True)

    df = add_contact_map_stats(load_tables(args.data_dir))
    summary = summarize(df)
    comp = base_composition(df)

    df.drop(columns=["normalized_seq"]).to_csv(args.output_dir / "sample_level_stats.csv", index=False)
    summary.to_csv(args.output_dir / "summary_by_split.csv", index=False)
    comp.to_csv(args.output_dir / "base_composition.csv", index=False)

    plot_length_hist(df, args.output_dir)
    plot_length_box(df, args.output_dir)
    plot_base_composition(comp, args.output_dir)
    plot_pair_stats(df, args.output_dir)

    metadata = {
        "total_samples": int(len(df)),
        "max_len_filter": MAX_LEN,
        "missing_contact_maps": int(df["ct_missing"].sum()),
        "bad_contact_map_shapes": int((~df["ct_shape_ok"]).sum()),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_markdown_report(summary, comp, args.output_dir, args.docs_path)

    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")
    print(f"Saved report to:  {args.docs_path}")


if __name__ == "__main__":
    main()
