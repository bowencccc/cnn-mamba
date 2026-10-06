#!/usr/bin/env python3
"""Summarize the four-state joint-frame beta grid and completed baselines."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve

from cnn_mamba.evaluate_candidates import reference_sites


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "joint_frame_grid_v1" / "summary"
TASKS = ("donor", "acceptor", "start", "stop")
TITLES = {
    "donor": "Donor", "acceptor": "Acceptor",
    "start": "Start codon", "stop": "Stop codon",
}
MODELS = (
    ("phase only", "phase", "#9D9D9D"),
    ("dense CDS=0.03", "dense", "#F58518"),
    ("4-state beta=0.03", "b003", "#4C78A8"),
    ("4-state beta=0.10", "b010", "#54A24B"),
    ("4-state beta=0.30", "b030", "#B279A2"),
)


def score_dir(key, regime):
    if key == "phase":
        experiment = (
            "frame_ablation_v1" if regime == "heldout"
            else "frame_ablation_chrxtrain_v1"
        )
        return ROOT / "artifacts" / "scores" / experiment / "combined" / regime / "w10"
    if key == "dense":
        return ROOT / "artifacts" / "scores" / "cds_weight_grid_v1" / "p003" / regime / "w10"
    return ROOT / "artifacts" / "scores" / "joint_frame_grid_v1" / key / regime / "w10"


def locus_path(key):
    if key == "phase":
        return ROOT / "work" / "uniann" / "frame_ablation_chrxtrain_v1" / "combined" / "w10" / "locus_metrics.tsv"
    if key == "dense":
        return ROOT / "work" / "uniann" / "cds_weight_grid_v1" / "p003" / "w10" / "locus_metrics.tsv"
    return ROOT / "work" / "uniann" / "joint_frame_grid_v1" / key / "w10" / "locus_metrics.tsv"


def run_dir(key, regime):
    return ROOT / "runs" / "joint_frame_grid_v1" / key / regime / "w10"


def joint_summary(key, regime):
    return ROOT / "results" / "joint_frame_grid_v1" / key / regime / "w10" / "joint_frame" / "summary.json"


def thin(recall, precision, maximum=8000):
    if len(recall) <= maximum:
        return recall, precision
    indices = np.linspace(0, len(recall) - 1, maximum, dtype=np.int64)
    return recall[indices], precision[indices]


def plot_prsn(regime, truth, lookup):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))
    for axis, task in zip(axes.flat, TASKS):
        expected = None
        for label, key, color in MODELS:
            directory = score_dir(key, regime)
            positions = np.load(directory / f"{task}_positions0.npy", mmap_mode="r")
            scores = np.load(directory / f"{task}_scores.npy", mmap_mode="r")
            if expected is None:
                expected = np.asarray(positions)
                labels = np.isin(positions, truth[task], assume_unique=True)
            elif not np.array_equal(expected, positions):
                raise RuntimeError(f"candidate positions differ: {regime} {task}")
            precision, recall, _ = precision_recall_curve(labels, scores)
            recall, precision = thin(recall, precision)
            axis.plot(
                recall, precision, linewidth=1.7, color=color,
                label=f"{label} (AP={lookup[(key, regime, task)]:.4f})",
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.set_title(TITLES[task], weight="bold")
        axis.set_xlabel("Sensitivity (recall)")
        axis.set_ylabel("Precision")
        axis.grid(alpha=0.23)
        axis.legend(loc="lower left", fontsize=7.8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT / f"{regime}_joint_frame_prsn.{suffix}",
            dpi=220 if suffix == "png" else None, bbox_inches="tight",
        )
    plt.close(fig)


def plot_locus(frame):
    configurations = [label for label, _key, _color in MODELS]
    colors = {"UniAnn": "#4C78A8", "EviAnn+UniAnn": "#E45756"}
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4), sharey=True)
    x = np.arange(len(configurations))
    width = 0.36
    for axis, (title, metric) in zip(
        axes, (("Locus sensitivity", "sn"), ("Locus precision", "pr"), ("Locus F1", "f1"))
    ):
        for offset, method in ((-width / 2, "UniAnn"), (width / 2, "EviAnn+UniAnn")):
            values = [
                float(frame[(frame.model == model) & (frame.method == method)][metric].iloc[0])
                for model in configurations
            ]
            bars = axis.bar(x + offset, values, width, color=colors[method], label=method)
            axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
        axis.set_title(title, weight="bold")
        axis.set_xticks(x, ("phase", "dense\nCDS", "beta\n0.03", "beta\n0.10", "beta\n0.30"))
        axis.set_ylim(60, 96)
        axis.grid(axis="y", alpha=0.25)
        axis.grid(axis="x", visible=False)
        axis.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("Percent (%)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT / f"strict_locus_metrics.{suffix}",
            dpi=220 if suffix == "png" else None, bbox_inches="tight",
        )
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    truth, _, _ = reference_sites(
        "drosophila", ROOT / "data" / "raw" / "dmel_reference.gtf",
        ROOT / "data" / "raw" / "dmel_genome.fa", "NC_004354.4",
    )

    candidate_frames, lookup = [], {}
    for label, key, _color in MODELS:
        for regime in ("heldout", "chrxtrain"):
            frame = pd.read_csv(score_dir(key, regime) / "candidate_auprc.tsv", sep="\t")
            frame = frame.assign(model=label, key=key, regime=regime)
            candidate_frames.append(frame)
            for row in frame.itertuples():
                lookup[(key, regime, row.task)] = float(row.AUPRC)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates.to_csv(OUT / "candidate_auprc.tsv", sep="\t", index=False, float_format="%.9f")
    for regime in ("heldout", "chrxtrain"):
        plot_prsn(regime, truth, lookup)

    locus_frames = []
    for label, key, _color in MODELS:
        locus_frames.append(pd.read_csv(locus_path(key), sep="\t").assign(model=label, key=key))
    locus = pd.concat(locus_frames, ignore_index=True)
    locus.to_csv(OUT / "strict_locus_metrics.tsv", sep="\t", index=False, float_format="%.9f")
    plot_locus(locus)

    joint_rows = []
    for label, key, _color in MODELS[2:]:
        for regime in ("heldout", "chrxtrain"):
            payload = json.loads(joint_summary(key, regime).read_text())
            checkpoint = torch.load(run_dir(key, regime) / "best_model.pt", map_location="cpu", weights_only=False)
            joint_rows.append({
                "model": label, "key": key, "regime": regime,
                "best_epoch": int(checkpoint["epoch"]),
                "cds_AP": payload["cds_metrics"]["AP"],
                "cds_best_F1": payload["cds_metrics"]["best_F1"],
                "frame_macro_AP": payload["frame_macro_AP"],
                "joint_macro_F1": payload["joint_macro_F1"],
            })
    pd.DataFrame(joint_rows).to_csv(
        OUT / "joint_frame_metrics.tsv", sep="\t", index=False,
        float_format="%.9f",
    )

    wide = []
    for label, key, _color in MODELS:
        row = {"model": label, "key": key}
        for regime in ("heldout", "chrxtrain"):
            for task in TASKS:
                row[f"{regime}_{task}_AUPRC"] = lookup[(key, regime, task)]
        match = locus[locus.key == key].set_index("method")
        for method, prefix in (("UniAnn", "uniann"), ("EviAnn+UniAnn", "combined")):
            for metric in ("sn", "pr", "f1"):
                row[f"{prefix}_{metric}"] = float(match.loc[method, metric])
        wide.append(row)
    output = OUT / "all_metrics_wide.tsv"
    pd.DataFrame(wide).to_csv(output, sep="\t", index=False, float_format="%.9f")
    print(output, flush=True)


if __name__ == "__main__":
    main()
