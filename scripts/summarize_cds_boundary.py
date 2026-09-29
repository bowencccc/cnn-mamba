#!/usr/bin/env python3
"""Compare phase-only, dense-CDS=0.03, and boundary-CDS=0.03 models."""

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
OUT = ROOT / "results" / "cds_boundary_r128_v1" / "summary"
TASKS = ("donor", "acceptor", "start", "stop")
TITLES = {
    "donor": "Donor", "acceptor": "Acceptor",
    "start": "Start codon", "stop": "Stop codon",
}
MODELS = (
    ("phase only", "phase", "#4C78A8"),
    ("dense CDS=0.03", "dense", "#F58518"),
    ("boundary CDS=0.03", "boundary", "#54A24B"),
)
MAX_DRAW_POINTS = 8000


def score_dir(key, regime):
    if key == "phase":
        experiment = (
            "frame_ablation_v1" if regime == "heldout"
            else "frame_ablation_chrxtrain_v1"
        )
        return (
            ROOT / "artifacts" / "scores" / experiment
            / "combined" / regime / "w10"
        )
    if key == "dense":
        return (
            ROOT / "artifacts" / "scores" / "cds_weight_grid_v1"
            / "p003" / regime / "w10"
        )
    return (
        ROOT / "artifacts" / "scores" / "cds_boundary_r128_v1"
        / "lambda_003" / regime / "w10"
    )


def run_dir(key, regime):
    if key == "dense":
        return ROOT / "runs" / "cds_weight_grid_v1" / "p003" / regime / "w10"
    if key == "boundary":
        return (
            ROOT / "runs" / "cds_boundary_r128_v1"
            / "lambda_003" / regime / "w10"
        )
    experiment = (
        "frame_ablation_v1" if regime == "heldout"
        else "frame_ablation_chrxtrain_v1"
    )
    return ROOT / "runs" / experiment / "combined" / regime / "w10"


def locus_path(key):
    if key == "phase":
        return (
            ROOT / "work" / "uniann" / "frame_ablation_chrxtrain_v1"
            / "combined" / "w10" / "locus_metrics.tsv"
        )
    if key == "dense":
        return (
            ROOT / "work" / "uniann" / "cds_weight_grid_v1"
            / "p003" / "w10" / "locus_metrics.tsv"
        )
    return (
        ROOT / "work" / "uniann" / "cds_boundary_r128_v1"
        / "lambda_003" / "w10" / "locus_metrics.tsv"
    )


def phase_path(key, regime):
    if key == "dense":
        return (
            ROOT / "results" / "cds_weight_grid_v1" / "p003"
            / regime / "w10" / "phase" / "summary.json"
        )
    return (
        ROOT / "results" / "cds_boundary_r128_v1" / "lambda_003"
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
            recall, precision = thin_curve(recall, precision)
            axis.plot(
                recall, precision, color=color, linewidth=1.8,
                label=f"{label} (AP={metric_lookup[(key, regime, task)]:.4f})",
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
            OUT / f"{regime}_cds_boundary_prsn.{suffix}",
            dpi=220 if suffix == "png" else None, bbox_inches="tight",
        )
    plt.close(figure)


def collect_training_diagnostics():
    gradient_rows, contribution_rows = [], []
    for regime in ("heldout", "chrxtrain"):
        directory = run_dir("boundary", regime)
        result = json.loads((directory / "results.json").read_text())
        checkpoint = torch.load(
            directory / "best_model.pt", map_location="cpu", weights_only=False
        )
        best_epoch = int(checkpoint["epoch"])
        for row in result["history"]:
            norms = row.get("train_shared_backbone_gradient_norms")
            if norms:
                for component, values in norms.items():
                    gradient_rows.append({
                        "model": "boundary CDS=0.03", "regime": regime,
                        "epoch": row["epoch"], "component": component,
                        **values, "probe_file": row.get("gradient_probe_file"),
                    })
            if int(row["epoch"]) != best_epoch:
                continue
            weights = {
                "splice": 0.25, "start_stop": 1.0,
                "phase": 0.1, "cds": 0.03,
            }
            for split in ("train", "validation"):
                values = {
                    "splice": row[f"{split}_splice_loss"],
                    "start_stop": row[f"{split}_start_stop_loss"],
                    "phase": row[f"{split}_phase_loss"],
                    "cds": row[f"{split}_cds_loss"],
                }
                weighted = {
                    name: float(values[name]) * weights[name]
                    for name in weights
                }
                total = sum(weighted.values())
                for component in weights:
                    contribution_rows.append({
                        "model": "boundary CDS=0.03", "regime": regime,
                        "best_epoch": best_epoch, "split": split,
                        "component": component,
                        "raw_loss": values[component],
                        "weight": weights[component],
                        "weighted_loss": weighted[component],
                        "fraction": weighted[component] / total,
                    })
    return pd.DataFrame(gradient_rows), pd.DataFrame(contribution_rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    truth, _, _ = reference_sites(
        "drosophila", ROOT / "data" / "raw" / "dmel_reference.gtf",
        ROOT / "data" / "raw" / "dmel_genome.fa", "NC_004354.4",
    )

    candidate_frames, metric_lookup = [], {}
    for label, key, _color in MODELS:
        for regime in ("heldout", "chrxtrain"):
            frame = pd.read_csv(
                score_dir(key, regime) / "candidate_auprc.tsv", sep="\t"
            ).assign(model=label, regime=regime)
            candidate_frames.append(frame)
            for row in frame.itertuples():
                metric_lookup[(key, regime, row.task)] = float(row.AUPRC)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates.to_csv(
        OUT / "candidate_auprc.tsv", sep="\t", index=False,
        float_format="%.9f",
    )
    for regime in ("heldout", "chrxtrain"):
        plot_prsn(regime, truth, metric_lookup)

    locus_frames = []
    for label, key, _color in MODELS:
        locus_frames.append(
            pd.read_csv(locus_path(key), sep="\t").assign(model=label)
        )
    locus = pd.concat(locus_frames, ignore_index=True)
    locus.to_csv(
        OUT / "strict_locus_metrics.tsv", sep="\t", index=False,
        float_format="%.9f",
    )

    phase_rows = []
    for label, key, _color in MODELS[1:]:
        for regime in ("heldout", "chrxtrain"):
            payload = json.loads(phase_path(key, regime).read_text())
            phase_rows.append({
                "model": label, "regime": regime,
                "raw_phase_macro_AP": payload["raw_phase_macro_AP"],
                "gated_phase_macro_AP": payload["gated_phase_macro_AP"],
                "cds_AP": payload["cds_metrics"]["AP"],
                "joint_macro_F1": payload["joint_macro_F1"],
            })
    pd.DataFrame(phase_rows).to_csv(
        OUT / "phase_and_cds_metrics.tsv", sep="\t", index=False,
        float_format="%.9f",
    )

    gradients, contributions = collect_training_diagnostics()
    gradients.to_csv(
        OUT / "shared_backbone_gradient_norms.tsv", sep="\t", index=False,
        float_format="%.9g",
    )
    contributions.to_csv(
        OUT / "best_epoch_loss_contributions.tsv", sep="\t", index=False,
        float_format="%.9g",
    )

    wide = []
    for label, key, _color in MODELS:
        row = {"model": label}
        for regime in ("heldout", "chrxtrain"):
            for task in TASKS:
                row[f"{regime}_{task}_AUPRC"] = metric_lookup[
                    (key, regime, task)
                ]
        match = locus[locus.model == label].set_index("method")
        for method, prefix in (("UniAnn", "uniann"), ("EviAnn+UniAnn", "combined")):
            for metric in ("sn", "pr", "f1"):
                row[f"{prefix}_{metric}"] = float(match.loc[method, metric])
        wide.append(row)
    output = OUT / "all_metrics_wide.tsv"
    pd.DataFrame(wide).to_csv(
        output, sep="\t", index=False, float_format="%.9f"
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
