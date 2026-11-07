#!/usr/bin/env python3
"""Compare three Drosophila models by candidate PR curves and strict locus metrics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from cnn_mamba.evaluate_candidates import reference_sites


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "droso_pretraining_comparison_v1"
TASKS = ("donor", "acceptor", "start", "stop")
TASK_LABELS = {
    "donor": "Donor",
    "acceptor": "Acceptor",
    "start": "Start codon",
    "stop": "Stop codon",
}
MODELS = {
    "Random initialization": {
        "color": "#4C78A8",
        "scores": ROOT / "artifacts/scores/joint_frame_grid_v1/b010/chrxtrain/w10",
        "locus": ROOT / "work/uniann/joint_frame_grid_v1/b010/w10/locus_metrics.tsv",
    },
    "Human MLM pretraining": {
        "color": "#E45756",
        "scores": ROOT / "artifacts/scores/mlm_transfer_v1/human_grch38_w10/chrxtrain/w10",
        "locus": ROOT / (
            "results/mlm_transfer_v1/human_grch38_w10/chrxtrain/w10/"
            "uniann_joint_t9/strict_locus_metrics.tsv"
        ),
    },
    "Human supervised pretraining": {
        "color": "#59A14F",
        "scores": ROOT / "artifacts/scores/droso_from_human_pretrain_v1/chrxtrain/w10",
        "locus": ROOT / "work/uniann/droso_from_human_pretrain_v1/w10/locus_metrics.tsv",
    },
}


def thin(x: np.ndarray, y: np.ndarray, maximum: int = 8000):
    if len(x) <= maximum:
        return x, y
    indices = np.linspace(0, len(x) - 1, maximum, dtype=np.int64)
    return x[indices], y[indices]


def style_pr_axis(axis, title: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("Sensitivity (recall)")
    axis.set_ylabel("Precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.01)
    axis.grid(color="#D9D9D9", linewidth=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def candidate_plots() -> None:
    truth, _, _ = reference_sites(
        "drosophila",
        ROOT / "data/raw/dmel_reference.gtf",
        ROOT / "data/raw/dmel_genome.fa",
        "NC_004354.4",
    )
    curve_rows = []
    combined, combined_axes = plt.subplots(2, 2, figsize=(11.0, 9.5), sharex=True, sharey=True)

    for task, combined_axis in zip(TASKS, combined_axes.flat):
        figure, axis = plt.subplots(figsize=(6.5, 6.1))
        expected_positions = None
        labels = None
        for model, spec in MODELS.items():
            score_dir = spec["scores"]
            positions = np.load(score_dir / f"{task}_positions0.npy", mmap_mode="r")
            scores = np.load(score_dir / f"{task}_scores.npy", mmap_mode="r")
            if expected_positions is None:
                expected_positions = np.asarray(positions)
                labels = np.isin(positions, truth[task], assume_unique=True)
            elif not np.array_equal(expected_positions, positions):
                raise RuntimeError(f"Candidate positions differ for {task}: {model}")

            metrics = pd.read_csv(score_dir / "candidate_auprc.tsv", sep="\t")
            ap = float(metrics.loc[metrics.task == task, "AUPRC"].iloc[0])
            precision, recall, _ = precision_recall_curve(labels, scores)
            recall_plot, precision_plot = thin(recall, precision)
            for target_axis in (axis, combined_axis):
                target_axis.plot(
                    recall_plot,
                    precision_plot,
                    color=spec["color"],
                    linewidth=2.1,
                    label=f"{model} (AUPRC={ap:.4f})",
                )
            for recall_value, precision_value in zip(recall_plot, precision_plot):
                curve_rows.append({
                    "task": task,
                    "model": model,
                    "sensitivity": float(recall_value),
                    "precision": float(precision_value),
                    "AUPRC": ap,
                })

        style_pr_axis(axis, TASK_LABELS[task])
        axis.legend(frameon=False, fontsize=9.0, loc="lower left")
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(
                OUT / f"{task}_candidate_pr_curve.{suffix}",
                dpi=320 if suffix == "png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(figure)

        style_pr_axis(combined_axis, TASK_LABELS[task])
        combined_axis.legend(frameon=False, fontsize=7.8, loc="lower left")

    combined.tight_layout()
    for suffix in ("png", "pdf"):
        combined.savefig(
            OUT / f"four_task_candidate_pr_curves.{suffix}",
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(combined)
    pd.DataFrame(curve_rows).to_csv(
        OUT / "four_task_candidate_pr_curve_source_data.tsv",
        sep="\t", index=False, float_format="%.9g",
    )


def locus_plots() -> None:
    rows = []
    for model, spec in MODELS.items():
        frame = pd.read_csv(spec["locus"], sep="\t")
        for row in frame.itertuples(index=False):
            rows.append({
                "model": model,
                "method": row.method,
                "sensitivity": float(row.sn),
                "precision": float(row.pr),
                "F1": float(row.f1),
            })
    table = pd.DataFrame(rows)
    table.to_csv(
        OUT / "strict_locus_metrics.tsv", sep="\t", index=False,
        float_format="%.9g",
    )

    methods = ("UniAnn", "EviAnn+UniAnn")
    combined, combined_axes = plt.subplots(1, 2, figsize=(13.4, 5.5), sharey=True)
    metric_names = ("sensitivity", "precision", "F1")
    x = np.arange(len(metric_names))
    width = 0.23
    offsets = (-width, 0.0, width)

    for method, combined_axis in zip(methods, combined_axes):
        figure, axis = plt.subplots(figsize=(7.2, 5.8))
        subset = table[table.method == method].set_index("model")
        for offset, (model, spec) in zip(offsets, MODELS.items()):
            values = [float(subset.loc[model, metric]) for metric in metric_names]
            for target_axis in (axis, combined_axis):
                bars = target_axis.bar(
                    x + offset, values, width, color=spec["color"], label=model,
                )
                target_axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8.2)

        for target_axis in (axis, combined_axis):
            target_axis.set_title(method)
            target_axis.set_ylabel("Percent")
            target_axis.set_xticks(x, ("Sensitivity", "Precision", "F1"))
            target_axis.set_ylim(0, 100)
            target_axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
            target_axis.set_axisbelow(True)
            target_axis.spines["top"].set_visible(False)
            target_axis.spines["right"].set_visible(False)
            target_axis.legend(frameon=False, fontsize=8.5, loc="lower right")

        figure.tight_layout()
        stem = "eviann_plus_uniann" if method == "EviAnn+UniAnn" else "uniann"
        for suffix in ("png", "pdf"):
            figure.savefig(
                OUT / f"{stem}_strict_locus_metrics.{suffix}",
                dpi=320 if suffix == "png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
        plt.close(figure)

    combined.tight_layout()
    for suffix in ("png", "pdf"):
        combined.savefig(
            OUT / f"uniann_and_eviann_plus_uniann_strict_locus_metrics.{suffix}",
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(combined)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
    })
    candidate_plots()
    locus_plots()
    print(OUT)


if __name__ == "__main__":
    main()
