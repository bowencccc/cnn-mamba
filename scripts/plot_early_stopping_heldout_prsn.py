#!/usr/bin/env python3
"""Plot exact chrX+ PR--Sn curves for an early-stopping window grid regime."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from cnn_mamba.evaluate_candidates import reference_sites


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (5, 10, 20, 30, 40, 80)
TASKS = ("donor", "acceptor", "start", "stop")
TITLES = {
    "donor": "Donor",
    "acceptor": "Acceptor",
    "start": "Start codon",
    "stop": "Stop codon",
}
COLORS = {
    5: "#4C78A8",
    10: "#E45756",
    20: "#59A14F",
    30: "#F28E2B",
    40: "#9C6ADE",
    80: "#76B7B2",
}
MAX_DRAW_POINTS = 8000


def thin_curve(recall: np.ndarray, precision: np.ndarray):
    if len(recall) <= MAX_DRAW_POINTS:
        return recall, precision
    indices = np.linspace(0, len(recall) - 1, MAX_DRAW_POINTS, dtype=np.int64)
    return recall[indices], precision[indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=("heldout", "chrxtrain"), default="heldout")
    args = parser.parse_args()
    scores_root = ROOT / "artifacts" / "scores" / "early_stopping_v1" / args.regime
    output_name = "chrx_heldout_auprc" if args.regime == "heldout" else "chrx_in_training_auprc"
    output_root = ROOT / "results" / "early_stopping_v1" / "plots" / output_name

    truth, _, _ = reference_sites(
        "drosophila",
        ROOT / "data" / "raw" / "dmel_reference.gtf",
        ROOT / "data" / "raw" / "dmel_genome.fa",
        "NC_004354.4",
    )
    metrics = {}
    frames = []
    for window in WINDOWS:
        path = scores_root / f"w{window}" / "candidate_auprc.tsv"
        frame = pd.read_csv(path, sep="\t").assign(window_kb=window)
        frames.append(frame)
        metrics[window] = dict(zip(frame.task, frame.AUPRC))

    output_root.mkdir(parents=True, exist_ok=True)
    values = pd.concat(frames, ignore_index=True)
    values.to_csv(output_root / "exact_auprc.tsv", sep="\t", index=False,
                  float_format="%.9f")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))
    for ax, task in zip(axes.flat, TASKS):
        labels = None
        expected_positions = None
        for window in WINDOWS:
            score_dir = scores_root / f"w{window}"
            positions = np.load(score_dir / f"{task}_positions0.npy", mmap_mode="r")
            scores = np.load(score_dir / f"{task}_scores.npy", mmap_mode="r")
            if labels is None:
                labels = np.isin(positions, truth[task], assume_unique=True)
                expected_positions = np.asarray(positions)
            elif not np.array_equal(positions, expected_positions):
                raise RuntimeError(f"{task}: candidate positions differ for w{window}")
            precision, recall, _ = precision_recall_curve(labels, scores)
            draw_recall, draw_precision = thin_curve(recall, precision)
            ax.plot(
                draw_recall,
                draw_precision,
                color=COLORS[window],
                linewidth=1.7,
                alpha=0.94,
                label=f"{window} kb (AP={metrics[window][task]:.4f})",
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.set_title(TITLES[task], fontsize=14, weight="bold")
        ax.set_xlabel("Sensitivity (recall)")
        ax.set_ylabel("Precision")
        ax.grid(alpha=0.23, linewidth=0.8)
        ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.94)

    fig.tight_layout(h_pad=2.0, w_pad=1.5)
    scope = "heldout" if args.regime == "heldout" else "in_training"
    png = output_root / f"early_stopping_chrX_{scope}_four_task_prsn_curves.png"
    pdf = output_root / f"early_stopping_chrX_{scope}_four_task_prsn_curves.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)


if __name__ == "__main__":
    main()
