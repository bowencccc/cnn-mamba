#!/usr/bin/env python3
"""Compare the joint CDS auxiliary model with the phase=0.10 control."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from cnn_mamba.evaluate_candidates import reference_sites


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "joint_cds_aux_v1" / "lambda_010"
TASKS = ("donor", "acceptor", "start", "stop")
TITLES = {
    "donor": "Donor", "acceptor": "Acceptor",
    "start": "Start codon", "stop": "Stop codon",
}
COLORS = {"Phase auxiliary": "#4C78A8", "+ CDS auxiliary": "#E45756"}
MAX_DRAW_POINTS = 8000


def score_dir(model, regime):
    if model == "Phase auxiliary":
        experiment = "frame_ablation_v1" if regime == "heldout" else "frame_ablation_chrxtrain_v1"
        return ROOT / "artifacts" / "scores" / experiment / "combined" / regime / "w10"
    return ROOT / "artifacts" / "scores" / "joint_cds_aux_v1" / "lambda_010" / regime / "w10"


def thin_curve(recall, precision):
    if len(recall) <= MAX_DRAW_POINTS:
        return recall, precision
    indices = np.linspace(0, len(recall) - 1, MAX_DRAW_POINTS, dtype=np.int64)
    return recall[indices], precision[indices]


def plot_curves(regime, truth, metrics):
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))
    for axis, task in zip(axes.flat, TASKS):
        expected = None
        for model in COLORS:
            directory = score_dir(model, regime)
            positions = np.load(directory / f"{task}_positions0.npy", mmap_mode="r")
            scores = np.load(directory / f"{task}_scores.npy", mmap_mode="r")
            if expected is None:
                expected = np.asarray(positions)
                labels = np.isin(positions, truth[task], assume_unique=True)
            elif not np.array_equal(expected, positions):
                raise RuntimeError(f"candidate positions differ: {regime} {task}")
            precision, recall, _ = precision_recall_curve(labels, scores)
            recall, precision = thin_curve(recall, precision)
            axis.plot(
                recall, precision, color=COLORS[model], linewidth=1.9,
                label=f"{model} (AP={metrics[(model, regime, task)]:.4f})",
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.set_title(TITLES[task], fontsize=14, weight="bold")
        axis.set_xlabel("Sensitivity (recall)")
        axis.set_ylabel("Precision")
        axis.grid(alpha=0.23)
        axis.legend(loc="lower left", fontsize=9, frameon=True)
    figure.tight_layout(h_pad=2.0, w_pad=1.5)
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUT / f"{regime}_raw_vs_joint_cds_prsn.{suffix}",
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
    metrics = {}
    frames = []
    for model in COLORS:
        for regime in ("heldout", "chrxtrain"):
            frame = pd.read_csv(
                score_dir(model, regime) / "candidate_auprc.tsv", sep="\t"
            ).assign(model=model, regime=regime)
            frames.append(frame)
            for row in frame.itertuples():
                metrics[(model, regime, row.task)] = float(row.AUPRC)
    candidate = pd.concat(frames, ignore_index=True)
    candidate.to_csv(
        OUT / "candidate_auprc_comparison.tsv", sep="\t", index=False,
        float_format="%.9f",
    )
    for regime in ("heldout", "chrxtrain"):
        plot_curves(regime, truth, metrics)

    baseline_locus = pd.read_csv(
        ROOT / "work" / "uniann" / "frame_ablation_chrxtrain_v1"
        / "combined" / "w10" / "locus_metrics.tsv", sep="\t",
    ).assign(model="Phase auxiliary")
    joint_locus = pd.read_csv(
        ROOT / "work" / "uniann" / "joint_cds_aux_v1" / "lambda_010"
        / "w10" / "locus_metrics.tsv", sep="\t",
    ).assign(model="+ CDS auxiliary")
    locus = pd.concat((baseline_locus, joint_locus), ignore_index=True)
    locus.to_csv(
        OUT / "strict_locus_comparison.tsv", sep="\t", index=False,
        float_format="%.9f",
    )

    phase_rows = []
    for regime in ("heldout", "chrxtrain"):
        directory = OUT / regime / "w10" / "phase"
        frame = pd.read_csv(directory / "raw_vs_gated_phase_auprc.tsv", sep="\t")
        frame.insert(0, "regime", regime)
        phase_rows.append(frame)
    pd.concat(phase_rows, ignore_index=True).to_csv(
        OUT / "cds_gated_phase_metrics.tsv", sep="\t", index=False,
        float_format="%.9f",
    )

    wide = []
    for model in COLORS:
        row = {"model": model}
        for regime in ("heldout", "chrxtrain"):
            for task in TASKS:
                row[f"{regime}_{task}_AUPRC"] = metrics[(model, regime, task)]
        match = locus[locus.model == model].set_index("method")
        for method, prefix in (("UniAnn", "uniann"), ("EviAnn+UniAnn", "combined")):
            for metric in ("sn", "pr", "f1"):
                row[f"{prefix}_{metric}"] = float(match.loc[method, metric])
        wide.append(row)
    pd.DataFrame(wide).to_csv(
        OUT / "all_metrics_wide.tsv", sep="\t", index=False,
        float_format="%.9f",
    )
    print(OUT / "all_metrics_wide.tsv", flush=True)


if __name__ == "__main__":
    main()
