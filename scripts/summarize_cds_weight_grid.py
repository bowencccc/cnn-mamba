#!/usr/bin/env python3
"""Summarize lambda_CDS 0/0.03/0.05/0.10 and gradient diagnostics."""

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
OUT = ROOT / "results" / "cds_weight_grid_v1"
TASKS = ("donor", "acceptor", "start", "stop")
TITLES = {
    "donor": "Donor", "acceptor": "Acceptor",
    "start": "Start codon", "stop": "Stop codon",
}
MODELS = (
    ("phase only", 0.00, "#4C78A8"),
    ("CDS=0.03", 0.03, "#F58518"),
    ("CDS=0.05", 0.05, "#54A24B"),
    ("CDS=0.10", 0.10, "#E45756"),
)
MAX_DRAW_POINTS = 8000


def score_dir(weight, regime):
    if weight == 0:
        experiment = (
            "frame_ablation_v1" if regime == "heldout"
            else "frame_ablation_chrxtrain_v1"
        )
        return (
            ROOT / "artifacts" / "scores" / experiment
            / "combined" / regime / "w10"
        )
    if weight == 0.10:
        return (
            ROOT / "artifacts" / "scores" / "joint_cds_aux_v1"
            / "lambda_010" / regime / "w10"
        )
    label = {0.03: "p003", 0.05: "p005"}[weight]
    return (
        ROOT / "artifacts" / "scores" / "cds_weight_grid_v1"
        / label / regime / "w10"
    )


def run_dir(weight, regime):
    if weight == 0.10:
        return (
            ROOT / "runs" / "joint_cds_aux_v1" / "lambda_010"
            / regime / "w10"
        )
    label = {0.03: "p003", 0.05: "p005"}[weight]
    return ROOT / "runs" / "cds_weight_grid_v1" / label / regime / "w10"


def locus_path(weight):
    if weight == 0:
        return (
            ROOT / "work" / "uniann" / "frame_ablation_chrxtrain_v1"
            / "combined" / "w10" / "locus_metrics.tsv"
        )
    if weight == 0.10:
        return (
            ROOT / "work" / "uniann" / "joint_cds_aux_v1"
            / "lambda_010" / "w10" / "locus_metrics.tsv"
        )
    label = {0.03: "p003", 0.05: "p005"}[weight]
    return (
        ROOT / "work" / "uniann" / "cds_weight_grid_v1"
        / label / "w10" / "locus_metrics.tsv"
    )


def phase_path(weight, regime):
    if weight == 0.10:
        return (
            ROOT / "results" / "joint_cds_aux_v1" / "lambda_010"
            / regime / "w10" / "phase" / "summary.json"
        )
    label = {0.03: "p003", 0.05: "p005"}[weight]
    return (
        ROOT / "results" / "cds_weight_grid_v1" / label
        / regime / "w10" / "phase" / "summary.json"
    )


def thin_curve(recall, precision):
    if len(recall) <= MAX_DRAW_POINTS:
        return recall, precision
    indices = np.linspace(0, len(recall) - 1, MAX_DRAW_POINTS, dtype=np.int64)
    return recall[indices], precision[indices]


def plot_prsn(regime, truth, metric_lookup):
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))
    for axis, task in zip(axes.flat, TASKS):
        expected = None
        for label, weight, color in MODELS:
            directory = score_dir(weight, regime)
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
                recall, precision, color=color, linewidth=1.8,
                label=f"{label} (AP={metric_lookup[(weight, regime, task)]:.4f})",
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.set_title(TITLES[task], fontsize=14, weight="bold")
        axis.set_xlabel("Sensitivity (recall)")
        axis.set_ylabel("Precision")
        axis.grid(alpha=0.23)
        axis.legend(loc="lower left", fontsize=8.5, frameon=True)
    figure.tight_layout(h_pad=2.0, w_pad=1.5)
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUT / f"{regime}_cds_weight_prsn.{suffix}",
            dpi=220 if suffix == "png" else None, bbox_inches="tight",
        )
    plt.close(figure)


def collect_gradient_and_loss_rows():
    gradients, contributions = [], []
    for weight in (0.03, 0.05, 0.10):
        for regime in ("heldout", "chrxtrain"):
            directory = run_dir(weight, regime)
            result = json.loads((directory / "results.json").read_text())
            checkpoint = torch.load(
                directory / "best_model.pt", map_location="cpu",
                weights_only=False,
            )
            best_epoch = int(checkpoint["epoch"])
            for row in result["history"]:
                norms = row.get("train_shared_backbone_gradient_norms")
                if norms:
                    for component, values in norms.items():
                        gradients.append({
                            "cds_weight": weight, "regime": regime,
                            "epoch": row["epoch"], "component": component,
                            **values,
                            "probe_file": row.get("gradient_probe_file"),
                        })
                if int(row["epoch"]) != best_epoch:
                    continue
                raw = {
                    "splice": row["train_splice_loss"],
                    "start_stop": row["train_start_stop_loss"],
                    "phase": row["train_phase_loss"],
                    "cds": row.get("train_cds_loss", 0.0),
                }
                val = {
                    "splice": row["validation_splice_loss"],
                    "start_stop": row["validation_start_stop_loss"],
                    "phase": row["validation_phase_loss"],
                    "cds": row.get("validation_cds_loss", 0.0),
                }
                weights = {
                    "splice": 0.25, "start_stop": 1.0,
                    "phase": 0.10, "cds": weight,
                }
                for split, values in (("train", raw), ("validation", val)):
                    weighted = {
                        name: float(values[name]) * weights[name]
                        for name in weights
                    }
                    total = sum(weighted.values())
                    for component in weights:
                        contributions.append({
                            "cds_weight": weight, "regime": regime,
                            "best_epoch": best_epoch, "split": split,
                            "component": component,
                            "raw_loss": values[component],
                            "weight": weights[component],
                            "weighted_loss": weighted[component],
                            "fraction": weighted[component] / total,
                        })
    return pd.DataFrame(gradients), pd.DataFrame(contributions)


def plot_gradients(frame):
    colors = {
        "splice": "#4C78A8", "start_stop": "#F58518",
        "phase": "#54A24B", "cds": "#E45756",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharey=True)
    for row_index, regime in enumerate(("heldout", "chrxtrain")):
        for column_index, weight in enumerate((0.03, 0.05)):
            axis = axes[row_index, column_index]
            subset = frame[
                (frame.regime == regime) & (frame.cds_weight == weight)
            ]
            for component in colors:
                values = subset[subset.component == component]
                axis.plot(
                    values.epoch, values.weighted, marker="o", markersize=3.5,
                    linewidth=1.7, color=colors[component], label=component,
                )
            axis.set_yscale("log")
            axis.set_title(f"{regime}; CDS weight={weight:.2f}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Weighted shared-backbone gradient L2 norm")
            axis.grid(alpha=0.22)
            axis.legend(fontsize=8.5, frameon=True)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUT / f"shared_backbone_gradient_norms.{suffix}",
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
    candidate_frames, metric_lookup = [], {}
    for label, weight, _color in MODELS:
        for regime in ("heldout", "chrxtrain"):
            frame = pd.read_csv(
                score_dir(weight, regime) / "candidate_auprc.tsv", sep="\t"
            ).assign(model=label, cds_weight=weight, regime=regime)
            candidate_frames.append(frame)
            for row in frame.itertuples():
                metric_lookup[(weight, regime, row.task)] = float(row.AUPRC)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates.to_csv(
        OUT / "candidate_auprc.tsv", sep="\t", index=False,
        float_format="%.9f",
    )
    for regime in ("heldout", "chrxtrain"):
        plot_prsn(regime, truth, metric_lookup)

    locus_frames = []
    for label, weight, _color in MODELS:
        locus_frames.append(
            pd.read_csv(locus_path(weight), sep="\t")
            .assign(model=label, cds_weight=weight)
        )
    locus = pd.concat(locus_frames, ignore_index=True)
    locus.to_csv(
        OUT / "strict_locus_metrics.tsv", sep="\t", index=False,
        float_format="%.9f",
    )

    phase_rows = []
    for label, weight, _color in MODELS[1:]:
        for regime in ("heldout", "chrxtrain"):
            payload = json.loads(phase_path(weight, regime).read_text())
            phase_rows.append({
                "model": label, "cds_weight": weight, "regime": regime,
                "raw_phase_macro_AP": payload["raw_phase_macro_AP"],
                "gated_phase_macro_AP": payload["gated_phase_macro_AP"],
                "cds_AP": payload["cds_metrics"]["AP"],
                "joint_macro_F1": payload["joint_macro_F1"],
            })
    pd.DataFrame(phase_rows).to_csv(
        OUT / "phase_and_cds_metrics.tsv", sep="\t", index=False,
        float_format="%.9f",
    )

    gradients, contributions = collect_gradient_and_loss_rows()
    gradients.to_csv(
        OUT / "shared_backbone_gradient_norms.tsv", sep="\t", index=False,
        float_format="%.9g",
    )
    contributions.to_csv(
        OUT / "best_epoch_loss_contributions.tsv", sep="\t", index=False,
        float_format="%.9g",
    )
    plot_gradients(gradients)

    wide = []
    for label, weight, _color in MODELS:
        row = {"model": label, "cds_weight": weight}
        for regime in ("heldout", "chrxtrain"):
            for task in TASKS:
                row[f"{regime}_{task}_AUPRC"] = metric_lookup[
                    (weight, regime, task)
                ]
        match = locus[locus.cds_weight == weight].set_index("method")
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
