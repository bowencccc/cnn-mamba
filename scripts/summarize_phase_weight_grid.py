#!/usr/bin/env python3
"""Collect AUPRC, PR--Sn curves and strict locus metrics for phase weights."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from cnn_mamba.evaluate_candidates import reference_sites


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase_weight_grid_v1"
WEIGHTS = (0.03, 0.05, 0.10, 0.20)
TASKS = ("donor", "acceptor", "start", "stop")
TITLES = {
    "donor": "Donor", "acceptor": "Acceptor",
    "start": "Start codon", "stop": "Stop codon",
}
COLORS = {
    0.03: "#4C78A8", 0.05: "#F58518",
    0.10: "#54A24B", 0.20: "#E45756",
}
MAX_DRAW_POINTS = 8000


def score_dir(weight, regime):
    if weight == 0.10:
        if regime == "heldout":
            return (
                ROOT / "artifacts" / "scores" / "frame_ablation_v1"
                / "combined" / "heldout" / "w10"
            )
        return (
            ROOT / "artifacts" / "scores" / "frame_ablation_chrxtrain_v1"
            / "combined" / "chrxtrain" / "w10"
        )
    label = {0.03: "p003", 0.05: "p005", 0.20: "p020"}[weight]
    return (
        ROOT / "artifacts" / "scores" / "phase_weight_grid_v1"
        / label / regime / "w10"
    )


def locus_path(weight):
    if weight == 0.10:
        return (
            ROOT / "work" / "uniann" / "frame_ablation_chrxtrain_v1"
            / "combined" / "w10" / "locus_metrics.tsv"
        )
    label = {0.03: "p003", 0.05: "p005", 0.20: "p020"}[weight]
    return (
        ROOT / "work" / "uniann" / "phase_weight_grid_v1"
        / label / "w10" / "locus_metrics.tsv"
    )


def thin_curve(recall, precision):
    if len(recall) <= MAX_DRAW_POINTS:
        return recall, precision
    indices = np.linspace(0, len(recall) - 1, MAX_DRAW_POINTS, dtype=np.int64)
    return recall[indices], precision[indices]


def plot_prsn(regime, truth, metrics):
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))
    for axis, task in zip(axes.flat, TASKS):
        expected_positions = None
        labels = None
        for weight in WEIGHTS:
            directory = score_dir(weight, regime)
            positions = np.load(directory / f"{task}_positions0.npy", mmap_mode="r")
            scores = np.load(directory / f"{task}_scores.npy", mmap_mode="r")
            if expected_positions is None:
                expected_positions = np.asarray(positions)
                labels = np.isin(positions, truth[task], assume_unique=True)
            elif not np.array_equal(positions, expected_positions):
                raise RuntimeError(
                    f"{regime} {task}: candidate positions differ at weight {weight}"
                )
            precision, recall, _ = precision_recall_curve(labels, scores)
            draw_recall, draw_precision = thin_curve(recall, precision)
            axis.plot(
                draw_recall, draw_precision, color=COLORS[weight],
                linewidth=1.8, alpha=0.95,
                label=f"phase={weight:.2f} (AP={metrics[(regime, weight, task)]:.4f})",
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.set_title(TITLES[task], fontsize=14, weight="bold")
        axis.set_xlabel("Sensitivity (recall)")
        axis.set_ylabel("Precision")
        axis.grid(alpha=0.23)
        axis.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.94)
    figure.tight_layout(h_pad=2.0, w_pad=1.5)
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUT / f"{regime}_four_task_prsn_curves.{suffix}",
            dpi=220 if suffix == "png" else None, bbox_inches="tight",
        )
    plt.close(figure)


def plot_locus(frame):
    methods = ("UniAnn", "EviAnn+UniAnn")
    colors = {"UniAnn": "#4C78A8", "EviAnn+UniAnn": "#E45756"}
    x = np.arange(len(WEIGHTS))
    width = 0.36
    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    for axis, (metric, title) in zip(
        axes, (("sn", "Locus sensitivity"), ("pr", "Locus precision"),
               ("f1", "Locus F1")),
    ):
        for offset, method in ((-width / 2, methods[0]), (width / 2, methods[1])):
            values = (
                frame[frame.method == method].set_index("phase_weight")
                .reindex(WEIGHTS)[metric].to_numpy()
            )
            bars = axis.bar(
                x + offset, values, width, color=colors[method],
                label=method.replace("+", " + "),
            )
            axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8.5)
        axis.set_title(title, fontsize=14, weight="bold")
        axis.set_xlabel("Phase-loss weight")
        axis.set_xticks(x, [f"{weight:.2f}" for weight in WEIGHTS])
        axis.set_ylim(65, 96)
        axis.grid(axis="y", alpha=0.25)
        axis.grid(axis="x", visible=False)
        axis.legend(loc="lower right", fontsize=9, frameon=True)
    axes[0].set_ylabel("Percent (%)")
    figure.tight_layout(w_pad=1.5)
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUT / f"strict_locus_metrics.{suffix}",
            dpi=220 if suffix == "png" else None, bbox_inches="tight",
        )
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    truth, _, _ = reference_sites(
        "drosophila", ROOT / "data" / "raw" / "dmel_reference.gtf",
        ROOT / "data" / "raw" / "dmel_genome.fa", "NC_004354.4",
    )

    auprc_frames = []
    metrics = {}
    for regime in ("heldout", "chrxtrain"):
        for weight in WEIGHTS:
            frame = pd.read_csv(
                score_dir(weight, regime) / "candidate_auprc.tsv", sep="\t"
            ).assign(regime=regime, phase_weight=weight)
            auprc_frames.append(frame)
            for row in frame.itertuples():
                metrics[(regime, weight, row.task)] = float(row.AUPRC)
        plot_prsn(regime, truth, metrics)
    auprc = pd.concat(auprc_frames, ignore_index=True)
    auprc.to_csv(
        OUT / "exact_candidate_auprc.tsv", sep="\t", index=False,
        float_format="%.9f",
    )

    locus_frames = []
    for weight in WEIGHTS:
        locus_frames.append(
            pd.read_csv(locus_path(weight), sep="\t").assign(phase_weight=weight)
        )
    locus = pd.concat(locus_frames, ignore_index=True)
    locus.to_csv(
        OUT / "strict_locus_metrics.tsv", sep="\t", index=False,
        float_format="%.9f",
    )
    plot_locus(locus)

    wide_rows = []
    for weight in WEIGHTS:
        row = {"phase_weight": weight}
        for regime in ("heldout", "chrxtrain"):
            for task in TASKS:
                row[f"{regime}_{task}_AUPRC"] = metrics[(regime, weight, task)]
        for method, prefix in (("UniAnn", "uniann"),
                               ("EviAnn+UniAnn", "eviann_uniann")):
            match = locus[(locus.phase_weight == weight) & (locus.method == method)].iloc[0]
            for metric in ("sn", "pr", "f1"):
                row[f"{prefix}_{metric}"] = float(match[metric])
        wide_rows.append(row)
    pd.DataFrame(wide_rows).to_csv(
        OUT / "all_metrics_wide.tsv", sep="\t", index=False,
        float_format="%.9f",
    )
    print(OUT / "all_metrics_wide.tsv")


if __name__ == "__main__":
    main()
