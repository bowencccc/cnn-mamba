#!/usr/bin/env python3
"""Plot strict locus metrics for early-stopping UniAnn evaluations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "work" / "uniann" / "early_stopping_v1"
OUT = ROOT / "results" / "early_stopping_v1" / "plots" / "uniann_locus"
WINDOWS = (5, 10, 20, 30, 40, 80)
COLORS = {"UniAnn": "#4C78A8", "EviAnn + UniAnn": "#E45756"}


def load_metrics():
    frames = []
    for window in WINDOWS:
        path = SOURCE / f"w{window}" / "locus_metrics.tsv"
        frames.append(pd.read_csv(path, sep="\t").assign(window_kb=window))
    return pd.concat(frames, ignore_index=True)


def main():
    frame = load_metrics()
    wide = frame.pivot(index="window_kb", columns="method")
    x = np.arange(len(WINDOWS))
    width = 0.36
    panels = (
        ("Locus sensitivity", "sn"),
        ("Locus precision", "pr"),
        ("Locus F1", "f1"),
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    for ax, (title, metric) in zip(axes, panels):
        uniann = wide[metric]["UniAnn"].reindex(WINDOWS).to_numpy()
        combined = wide[metric]["EviAnn+UniAnn"].reindex(WINDOWS).to_numpy()
        bars1 = ax.bar(
            x - width / 2, uniann, width,
            color=COLORS["UniAnn"], label="UniAnn",
        )
        bars2 = ax.bar(
            x + width / 2, combined, width,
            color=COLORS["EviAnn + UniAnn"], label="EviAnn + UniAnn",
        )
        ax.bar_label(bars1, fmt="%.1f", padding=3, fontsize=8.5)
        ax.bar_label(bars2, fmt="%.1f", padding=3, fontsize=8.5)
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xlabel("Window size (kb)")
        ax.set_xticks(x, WINDOWS)
        ax.set_ylim(60, 96)
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.94)
    axes[0].set_ylabel("Percent (%)")

    fig.tight_layout(w_pad=1.5)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "early_stopping_uniann_eviann_combined_locus_metrics.png"
    pdf = OUT / "early_stopping_uniann_eviann_combined_locus_metrics.pdf"
    table = OUT / "early_stopping_uniann_eviann_combined_locus_metrics.tsv"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    frame.sort_values(["window_kb", "method"]).to_csv(
        table, sep="\t", index=False, float_format="%.9f"
    )
    plt.close(fig)
    print(png)


if __name__ == "__main__":
    main()
