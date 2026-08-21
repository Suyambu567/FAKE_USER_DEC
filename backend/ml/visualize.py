#!/usr/bin/env python3
"""The paper's two figures, rendered as standalone SVG.

Paper Figure 4 is a correlation heatmap over the profile features; Figure 5 is a
bar chart comparing algorithm accuracy. Both are reproduced here from *this*
project's numbers, not the paper's.

**No plotting dependency.** matplotlib is not installed and is not added: it
would be ~50 MB in the tooling environment to draw two static charts. SVG is
written directly, so the figures are plain text, diff well in git, scale without
resampling, and open in any browser. If a raster copy is ever needed, any
browser or `rsvg-convert` will produce one.

Usage:
    python -m ml.visualize --dataset ../data/paper_signal.csv \
        --benchmark ../docs/benchmark_signal.json --out ../docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from ml import features
from ml.preprocess import clean

# Diverging scale: blue for negative correlation, red for positive, near-white
# at zero. Chosen so the sign is readable without a legend lookup.
_NEG = (37, 99, 235)
_POS = (220, 38, 38)


def _correlation_colour(value: float) -> str:
    """-1..1 -> a hex colour on the blue-white-red scale."""
    v = max(-1.0, min(1.0, float(value)))
    end = _POS if v >= 0 else _NEG
    t = abs(v)
    rgb = tuple(round(255 + (c - 255) * t) for c in end)
    return "#%02x%02x%02x" % rgb


def correlation_heatmap(frame: pd.DataFrame, target: pd.Series, out: Path,
                        *, max_features: int = 20) -> Path:
    """Paper Figure 4 — correlation between features and with the label.

    The label is included as the last row/column, because "which feature moves
    with fake?" is the question the figure exists to answer; a features-only
    matrix looks impressive and says nothing about detection.

    Features are ranked by absolute correlation with the label and the top
    `max_features` are kept: a 24x24 grid of 3-pixel cells is unreadable.
    """
    numeric = frame.select_dtypes(include=[np.number]).copy()
    numeric["IS_FAKE"] = (target.astype(str) == features.POSITIVE_CLASS).astype(int)

    corr = numeric.corr(numeric_only=True).fillna(0.0)
    ranked = corr["IS_FAKE"].drop("IS_FAKE").abs().sort_values(ascending=False)
    keep = list(ranked.head(max_features).index) + ["IS_FAKE"]
    corr = corr.loc[keep, keep]

    cell, left, top = 30, 190, 190
    right, bottom = 90, 30
    n = len(keep)
    width, height = left + n * cell + right, top + n * cell + bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-size="17" font-weight="600" fill="#111827">'
        'Correlation heatmap — engineered features vs the label</text>',
        f'<text x="{left}" y="56" font-size="12" fill="#6b7280">'
        f'Pearson r. Top {max_features} features by |r| with IS_FAKE. '
        'Blue = negative, red = positive.</text>',
    ]

    for i, row in enumerate(keep):
        y = top + i * cell
        parts.append(
            f'<text x="{left - 8}" y="{y + cell / 2 + 4}" font-size="11" text-anchor="end" '
            f'fill="#374151">{escape(row)}</text>'
        )
        for j, col in enumerate(keep):
            x = left + j * cell
            v = float(corr.loc[row, col])
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{_correlation_colour(v)}" stroke="#ffffff" stroke-width="1"/>'
            )
            if abs(v) >= 0.35:  # only label cells a reader would actually squint at
                ink = "#ffffff" if abs(v) > 0.65 else "#111827"
                parts.append(
                    f'<text x="{x + cell / 2}" y="{y + cell / 2 + 3.5}" font-size="9" '
                    f'text-anchor="middle" fill="{ink}">{v:+.2f}</text>'
                )

    for j, col in enumerate(keep):
        x = left + j * cell + cell / 2
        parts.append(
            f'<text x="{x}" y="{top - 8}" font-size="11" fill="#374151" '
            f'transform="rotate(-60 {x} {top - 8})">{escape(col)}</text>'
        )

    # Colour key.
    key_x, key_y = left + n * cell + 24, top
    parts.append(f'<text x="{key_x}" y="{key_y - 8}" font-size="11" fill="#374151">r</text>')
    for step in range(41):
        v = 1.0 - step / 20.0
        parts.append(
            f'<rect x="{key_x}" y="{key_y + step * 6}" width="18" height="6" '
            f'fill="{_correlation_colour(v)}"/>'
        )
    for label, offset in (("+1", 0), ("0", 120), ("-1", 240)):
        parts.append(
            f'<text x="{key_x + 24}" y="{key_y + offset + 5}" font-size="10" '
            f'fill="#6b7280">{label}</text>'
        )

    parts.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def algorithm_comparison(results: list[dict], baseline: float, out: Path,
                         *, dataset: str = "", highlight: str = "XGBoost") -> Path:
    """Paper Figure 5 — measured accuracy per algorithm.

    The paper's own Table 2 gives accuracy *ranges* per algorithm (XGBoost
    90-95%). Those are not reproduced here: this chart plots what these models
    scored on this data, with the majority-class baseline drawn in, because an
    accuracy bar without its baseline is unreadable on a skewed dataset.
    """
    rows = sorted(
        [r for r in results if r.get("model", "").lower() != "baseline (majority class)"],
        key=lambda r: r.get("accuracy", 0.0), reverse=True,
    )
    bar_h, gap = 26, 8
    left, top, right, bottom = 210, 96, 90, 60
    plot_w = 560
    height = top + len(rows) * (bar_h + gap) + bottom
    width = left + plot_w + right

    lo = min([baseline] + [r["accuracy"] for r in rows]) - 0.04
    lo = max(0.0, round(lo, 2))
    hi = 1.0

    def x_of(value: float) -> float:
        return left + (value - lo) / (hi - lo) * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="34" font-size="17" font-weight="600" fill="#111827">'
        'Algorithm comparison — measured accuracy</text>',
        f'<text x="24" y="56" font-size="12" fill="#6b7280">'
        f'{escape(dataset)} · cross-validated · dashed line = majority-class '
        f'baseline ({baseline:.3f})</text>',
    ]

    for tick in np.arange(lo, hi + 1e-9, 0.05):
        x = x_of(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" '
                     f'y2="{height - bottom + 6}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - bottom + 22}" font-size="10" '
                     f'text-anchor="middle" fill="#6b7280">{tick:.2f}</text>')

    for i, row in enumerate(rows):
        y = top + i * (bar_h + gap)
        acc = float(row["accuracy"])
        is_focus = row["model"] == highlight
        fill = "#1d4ed8" if is_focus else "#93c5fd"
        ink = "#111827" if is_focus else "#374151"
        weight = ' font-weight="600"' if is_focus else ""
        parts.append(
            f'<text x="{left - 10}" y="{y + bar_h / 2 + 4}" font-size="12" '
            f'text-anchor="end" fill="{ink}"{weight}>{escape(row["model"])}</text>'
        )
        parts.append(
            f'<rect x="{left}" y="{y}" width="{x_of(acc) - left:.1f}" height="{bar_h}" '
            f'fill="{fill}" rx="3"/>'
        )
        std = row.get("accuracy_std")
        label = f'{acc:.4f}' + (f' ±{std:.4f}' if std else "")
        parts.append(
            f'<text x="{x_of(acc) + 8:.1f}" y="{y + bar_h / 2 + 4}" font-size="11" '
            f'fill="#374151">{label}</text>'
        )

    bx = x_of(baseline)
    parts.append(
        f'<line x1="{bx:.1f}" y1="{top - 14}" x2="{bx:.1f}" y2="{height - bottom + 6}" '
        f'stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4"/>'
    )
    parts.append(
        f'<text x="{bx + 6:.1f}" y="{top - 20}" font-size="11" fill="#dc2626">'
        f'baseline {baseline:.3f}</text>'
    )
    parts.append("</svg>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, help="CSV for the correlation heatmap.")
    ap.add_argument("--benchmark", type=Path,
                    help="ml.benchmark --json output for the comparison chart.")
    ap.add_argument("--out", type=Path, default=Path("../docs/figures"))
    ap.add_argument("--max-features", type=int, default=20)
    args = ap.parse_args()

    if not args.dataset and not args.benchmark:
        ap.error("give --dataset, --benchmark, or both")

    if args.dataset:
        df, _ = clean(pd.read_csv(args.dataset))
        frame = features.build_feature_frame(df)
        path = correlation_heatmap(frame, df[features.TARGET],
                                   args.out / "correlation_heatmap.svg",
                                   max_features=args.max_features)
        print(f"wrote {path}")

    if args.benchmark:
        data = json.loads(args.benchmark.read_text(encoding="utf-8"))
        path = algorithm_comparison(
            data["results"], float(data["baseline_accuracy"]),
            args.out / "algorithm_comparison.svg",
            dataset=str(data.get("dataset", "")),
        )
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
