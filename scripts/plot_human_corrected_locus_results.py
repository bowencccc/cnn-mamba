#!/usr/bin/env python3
"""Plot the corrected Human chr1 strict locus-level comparison."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT
    / "results/human_joint_frame_beta010_es5_v1/chr1train/"
    "epoch10_joint_softmax_t9_uniann_latest/strict_evaluation/locus_summary.tsv"
)
OUTDIR = INPUT.parents[1] / "report"

ORDER = ["eviann_only", "uniann_only", "eviann_plus_uniann"]
LABELS = {
    "eviann_only": "EviAnn",
    "uniann_only": "UniAnn",
    "eviann_plus_uniann": "EviAnn +\nUniAnn",
}
COLORS = ["#7A7A7A", "#3E78B2", "#3A9D5D"]
METRICS = [
    ("locus_pr_percent", "Precision"),
    ("locus_sn_percent", "Sensitivity"),
    ("locus_f1_percent", "F1"),
]


def main() -> None:
    data = pd.read_csv(INPUT, sep="\t").set_index("query_mode").loc[ORDER]
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 4.2), sharey=True)
    for ax, (column, title) in zip(axes, METRICS):
        values = data[column].astype(float).to_numpy()
        bars = ax.bar(range(3), values, color=COLORS, width=0.68)
        ax.set_title(title, fontsize=12, pad=8)
        ax.set_xticks(range(3), [LABELS[x] for x in ORDER])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=9)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=10)

    axes[0].set_ylabel("Locus-level score (%)")
    fig.suptitle("Human chr1 strict locus evaluation", fontsize=14, y=0.99)
    fig.text(
        0.5,
        0.015,
        "Epoch 10 · chr1 included in training · joint-softmax T=9 · corrected union · strict RefSeq",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), w_pad=1.6)

    png = OUTDIR / "human_chr1_corrected_strict_locus_comparison.png"
    pdf = OUTDIR / "human_chr1_corrected_strict_locus_comparison.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
