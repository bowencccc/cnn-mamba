#!/usr/bin/env python3
"""Create a compact four-panel summary of completed Drosophila experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "report_ready"


def annotate_best(axis, x, y, fmt="{:.1f}"):
    index = int(np.argmax(y))
    axis.annotate(
        fmt.format(y[index]), (x[index], y[index]),
        xytext=(0, 7), textcoords="offset points", ha="center",
        fontsize=8.5, weight="bold",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))

    # A: architectural additions on truly held-out chrX.
    frame = pd.read_csv(
        ROOT / "results/frame_ablation_v1/exact_chrx_plus_auprc.tsv",
        sep="\t",
    )
    tasks = ["donor", "acceptor", "start", "stop"]
    x = np.arange(len(tasks))
    width = 0.19
    colors = ["#9D9D9D", "#72B7B2", "#F58518", "#4C78A8"]
    for index, row in frame.iterrows():
        axes[0, 0].bar(
            x + (index - 1.5) * width,
            [100 * row[task] for task in tasks],
            width=width, label=row["model"], color=colors[index],
        )
    axes[0, 0].set_xticks(x, ["Donor", "Acceptor", "Start", "Stop"])
    axes[0, 0].set_ylim(45, 100)
    axes[0, 0].set_ylabel("Held-out chrX AUPRC (%)")
    axes[0, 0].set_title("A  Frame-aware auxiliary learning")
    axes[0, 0].legend(fontsize=8, ncol=2, frameon=True)

    # B: phase auxiliary weight and strict locus reconstruction.
    phase = pd.read_csv(
        ROOT / "results/phase_weight_grid_v1/all_metrics_wide.tsv", sep="\t"
    ).sort_values("phase_weight")
    phase_x = phase["phase_weight"].to_numpy()
    uniann = phase["uniann_f1"].to_numpy()
    combined = phase["eviann_uniann_f1"].to_numpy()
    axes[0, 1].plot(
        phase_x, uniann, "o-", linewidth=2, color="#F58518", label="UniAnn"
    )
    axes[0, 1].plot(
        phase_x, combined, "s-", linewidth=2, color="#4C78A8",
        label="EviAnn + UniAnn",
    )
    annotate_best(axes[0, 1], phase_x, uniann)
    annotate_best(axes[0, 1], phase_x, combined)
    axes[0, 1].set_xticks(phase_x)
    axes[0, 1].set_xlabel("Phase auxiliary weight")
    axes[0, 1].set_ylabel("Strict locus F1 (%)")
    axes[0, 1].set_title("B  Phase-loss weight controls recall–precision trade-off")
    axes[0, 1].legend(fontsize=8.5, frameon=True)

    # C: dense CDS auxiliary weight.
    cds = pd.read_csv(
        ROOT / "results/cds_weight_grid_v1/all_metrics_wide.tsv", sep="\t"
    ).sort_values("cds_weight")
    cds_x = cds["cds_weight"].to_numpy()
    uniann = cds["uniann_f1"].to_numpy()
    combined = cds["combined_f1"].to_numpy()
    axes[1, 0].plot(
        cds_x, uniann, "o-", linewidth=2, color="#F58518", label="UniAnn"
    )
    axes[1, 0].plot(
        cds_x, combined, "s-", linewidth=2, color="#4C78A8",
        label="EviAnn + UniAnn",
    )
    annotate_best(axes[1, 0], cds_x, uniann)
    annotate_best(axes[1, 0], cds_x, combined)
    axes[1, 0].set_xticks(cds_x)
    axes[1, 0].set_xlabel("Dense CDS auxiliary weight")
    axes[1, 0].set_ylabel("Strict locus F1 (%)")
    axes[1, 0].set_title("C  Dense CDS supervision has limited downstream benefit")
    axes[1, 0].legend(fontsize=8.5, frameon=True)

    # D: window-size comparison, excluding the separate frame model.
    exact = pd.read_csv(
        ROOT / "results/early_stopping_v1/plots/chrx_heldout_auprc/exact_auprc.tsv",
        sep="\t",
    )
    mean_ap = exact.groupby("window_kb", sort=True)["AUPRC"].mean() * 100
    locus = pd.read_csv(
        ROOT / "results/early_stopping_v1/plots/uniann_locus/"
        "early_stopping_uniann_eviann_combined_locus_metrics_with_frame.tsv",
        sep="\t",
    )
    locus = locus[
        (locus["method"] == "EviAnn+UniAnn")
        & (~locus["frame_auxiliary"].astype(bool))
    ].set_index("window_kb").sort_index()
    windows = mean_ap.index.to_numpy()
    axes[1, 1].plot(
        windows, mean_ap.to_numpy(), "o-", linewidth=2,
        color="#54A24B", label="Mean held-out AUPRC",
    )
    axes[1, 1].plot(
        windows, locus.loc[windows, "f1"].to_numpy(), "s-", linewidth=2,
        color="#4C78A8", label="EviAnn + UniAnn locus F1",
    )
    axes[1, 1].set_xticks(windows)
    axes[1, 1].set_xlabel("Window size (kb)")
    axes[1, 1].set_ylabel("Score (%)")
    axes[1, 1].set_title("D  Larger windows do not consistently improve performance")
    axes[1, 1].legend(fontsize=8.5, frameon=True)

    for axis in axes.flat:
        axis.grid(alpha=0.22)
    figure.tight_layout(h_pad=2.1, w_pad=1.5)
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUT / f"cnn_mamba_droso_summary.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
