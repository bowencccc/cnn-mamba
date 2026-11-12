#!/usr/bin/env python3
"""Plot the Human chr1 strict locus-level UniAnn/AltAnn comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results" / "altann_human_chr1_w10_epoch10_t9_k10"
INPUT = RESULT_DIR / "locus_metrics.tsv"
OUTPUT = RESULT_DIR / "locus_comparison.png"


def main() -> None:
    with INPUT.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    metric_keys = ("sensitivity", "precision", "f1")
    metric_labels = ("Sensitivity", "Precision", "F1")
    colors = ("#7f7f7f", "#4c78a8", "#f58518", "#54a24b")
    x = np.arange(len(metric_keys))
    width = 0.19

    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    for index, (row, color) in enumerate(zip(rows, colors)):
        values = [float(row[key]) for key in metric_keys]
        offset = (index - (len(rows) - 1) / 2) * width
        bars = axis.bar(x + offset, values, width, color=color, label=row["method"])
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )

    axis.set_title("Human chr1 — strict locus level", fontsize=14, weight="bold")
    axis.set_xticks(x, metric_labels)
    axis.set_ylim(0, 104)
    axis.set_ylabel("Percent (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.29), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
