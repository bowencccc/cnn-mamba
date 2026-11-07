#!/usr/bin/env python3
"""Plot chrX+ EviAnn start/stop metrics for transcript-selection strategies."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/coleoptera_eviann_complete_class_v1/label_selection_audit"
INPUT = RESULT / "chrx_plus_eviann_site_prsn.tsv"
OUTPUT_STEM = RESULT / "chrx_plus_eviann_start_stop_selection_prsn_comparison"

TASKS = ("start", "stop")
TASK_LABELS = ("Start codon", "Stop codon")
STRATEGIES = (
    "all_transcripts_union",
    "longest_cds_per_locus",
    "human_lightgbm_p90",
    "human_lightgbm_p90_locus_fallback",
    "complete_equal_else_k_with_fallback",
    "longest_complete_equal_else_k",
)
STRATEGY_LABELS = {
    "all_transcripts_union": "All isoforms (union)",
    "longest_cds_per_locus": "Longest CDS per locus",
    "human_lightgbm_p90": "Human LightGBM (P90 threshold)",
    "human_lightgbm_p90_locus_fallback": "LightGBM P90 + locus fallback",
    "complete_equal_else_k_with_fallback": "Complete '='/'k' + fallback",
    "longest_complete_equal_else_k": "Complete '='; else 'k'",
}
COLORS = {
    "all_transcripts_union": "#7A7A7A",
    "longest_cds_per_locus": "#2676B8",
    "human_lightgbm_p90": "#8C6BB1",
    "human_lightgbm_p90_locus_fallback": "#DD8452",
    "complete_equal_else_k_with_fallback": "#55A868",
    "longest_complete_equal_else_k": "#D65F5F",
}


def style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    axis.set_ylim(0, 108)
    axis.set_yticks(np.arange(0, 101, 20))


def main():
    table = pd.read_csv(INPUT, sep="\t")
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 4.8), sharey=True)
    x = np.arange(len(TASKS))
    width = 0.125

    for axis, metric, title in zip(
        axes,
        ("precision", "sensitivity", "f1"),
        ("Precision", "Sensitivity", "F1"),
    ):
        for index, strategy in enumerate(STRATEGIES):
            subset = table[table.selection == strategy].set_index("task")
            values = np.array([subset.loc[task, metric] for task in TASKS]) * 100
            offset = (index - 2.5) * width
            bars = axis.bar(
                x + offset,
                values,
                width,
                color=COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
            )
            axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.4, rotation=90)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xticks(x, TASK_LABELS)
        axis.set_ylabel("Percent (%)")
        style(axis)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUTPUT_STEM.with_suffix(f".{suffix}"),
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
