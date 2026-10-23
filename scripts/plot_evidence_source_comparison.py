#!/usr/bin/env python3
"""Compare Drosophila- and Coleoptera-evidence EviAnn training runs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from cnn_mamba.evaluate_candidates import reference_sites


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "evidence_source_comparison_v1"
TASKS = ("donor", "acceptor", "start", "stop")
TASK_LABELS = {
    "donor": "Donor",
    "acceptor": "Acceptor",
    "start": "Start codon",
    "stop": "Stop codon",
}
MODELS = {
    "Drosophila protein evidence": {
        "color": "#3569A8",
        "scores": ROOT / "artifacts/scores/joint_frame_grid_v1/b010/chrxtrain/w10",
        "annotation_gap": ROOT / "results/joint_frame_grid_v1/b010/chrxtrain/w10/annotation_gap_metrics.tsv",
        "joint": ROOT / "results/joint_frame_grid_v1/b010/chrxtrain/w10/joint_frame/joint_class_metrics.tsv",
    },
    "Coleoptera protein evidence": {
        "color": "#D45A45",
        "scores": ROOT / "artifacts/scores/coleoptera_eviann_v1/retrained/chrxtrain/w10",
        "annotation_gap": ROOT / "results/coleoptera_eviann_v1/retrained/chrxtrain/w10/annotation_gap_metrics.tsv",
        "joint": ROOT / "results/coleoptera_eviann_v1/retrained/chrxtrain/w10/joint_frame/joint_class_metrics.tsv",
    },
}
STATE_ORDER = ("non_CDS", "frame_0", "frame_1", "frame_2")
STATE_LABELS = ("Non-CDS", "Frame 0", "Frame 1", "Frame 2")


def thin(x: np.ndarray, y: np.ndarray, maximum: int = 7000):
    if len(x) <= maximum:
        return x, y
    selected = np.linspace(0, len(x) - 1, maximum, dtype=np.int64)
    return x[selected], y[selected]


def style(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.22)
    axis.set_axisbelow(True)


def candidate_comparison(truth):
    auprc_rows = []
    operating_point_rows = []
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), sharex=True, sharey=True)
    for axis, task in zip(axes.flat, TASKS):
        expected_positions = None
        for label, spec in MODELS.items():
            score_dir = spec["scores"]
            positions = np.load(score_dir / f"{task}_positions0.npy", mmap_mode="r")
            scores = np.load(score_dir / f"{task}_scores.npy", mmap_mode="r")
            if expected_positions is None:
                expected_positions = np.asarray(positions)
                targets = np.isin(positions, truth[task], assume_unique=True)
            elif not np.array_equal(expected_positions, positions):
                raise RuntimeError(f"candidate positions differ for {task}")
            precision, sensitivity, _ = precision_recall_curve(targets, scores)
            sensitivity, precision = thin(sensitivity, precision)
            metric = pd.read_csv(score_dir / "candidate_auprc.tsv", sep="\t")
            ap = float(metric.loc[metric.task == task, "AUPRC"].iloc[0])
            auprc_rows.append({"evidence": label, "task": task, "AUPRC": ap})
            axis.plot(
                sensitivity,
                precision,
                color=spec["color"],
                linewidth=2.0,
                label=f"{label.replace(' protein evidence', '')} (AP={ap:.4f})",
            )
            gap = pd.read_csv(spec["annotation_gap"], sep="\t")
            gap_row = gap[gap.task == task].iloc[0]
            annotation_sn = float(gap_row.supported_sites / gap_row.reference_sites)
            annotation_pr = float(gap_row.supported_sites / gap_row.eviann_sites)
            operating_point_rows.append({
                "evidence": label,
                "task": task,
                "sensitivity": annotation_sn,
                "precision": annotation_pr,
                "matched_sites": int(gap_row.supported_sites),
                "reference_sites": int(gap_row.reference_sites),
                "eviann_sites": int(gap_row.eviann_sites),
            })
            axis.scatter(
                [annotation_sn], [annotation_pr], marker="X", s=95,
                color=spec["color"], edgecolor="white", linewidth=0.9,
                zorder=5,
                label=(
                    f"{label.replace(' protein evidence', '')} EviAnn "
                    f"(Sn={annotation_sn:.3f}, Pr={annotation_pr:.3f})"
                ),
            )
        axis.set_title(TASK_LABELS[task], fontweight="bold")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.set_xlabel("Sensitivity (recall)")
        axis.set_ylabel("Precision")
        axis.legend(frameon=False, fontsize=8.5, loc="lower left")
        style(axis)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUT / f"candidate_four_task_prsn_comparison.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)
    frame = pd.DataFrame(auprc_rows)
    frame.to_csv(OUT / "candidate_four_task_auprc.tsv", sep="\t", index=False, float_format="%.9f")
    pd.DataFrame(operating_point_rows).to_csv(
        OUT / "eviann_site_operating_points.tsv",
        sep="\t", index=False, float_format="%.9f",
    )
    return frame


def four_state_comparison():
    rows = []
    for label, spec in MODELS.items():
        frame = pd.read_csv(spec["joint"], sep="\t")
        for row in frame.itertuples():
            rows.append({
                "evidence": label,
                "state": row._1 if hasattr(row, "_1") else row[1],
                "precision": float(row.precision),
                "sensitivity": float(row.recall),
                "F1": float(row.F1),
                "support": int(row.support),
                "predicted": int(row.predicted),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "four_state_prsn.tsv", sep="\t", index=False, float_format="%.9f")

    figure, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), sharey=True)
    x = np.arange(len(STATE_ORDER))
    width = 0.36
    for axis, (metric, title) in zip(
        axes,
        (("precision", "Precision"), ("sensitivity", "Sensitivity"), ("F1", "F1")),
    ):
        for offset, (label, spec) in zip((-width / 2, width / 2), MODELS.items()):
            subset = table[table.evidence == label].set_index("state")
            values = [100 * float(subset.loc[state, metric]) for state in STATE_ORDER]
            bars = axis.bar(x + offset, values, width, color=spec["color"], label=label)
            axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.8)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x, STATE_LABELS, rotation=18, ha="right")
        axis.set_ylim(78, 101.5)
        axis.set_ylabel("Percent (%)")
        style(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, frameon=False, fontsize=8.5, ncol=2,
        loc="lower center", bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUT / f"four_state_prsn_comparison.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)
    return table


def locus_comparison():
    original_aug = pd.read_csv(
        ROOT / "results/coleoptera_eviann_v1/existing_model_augmentation/original/strict_locus_metrics.tsv",
        sep="\t",
    )
    original_scaling = pd.read_csv(
        ROOT / "results/frame_scaling_extended_grid_v1/strict_locus_metrics.tsv",
        sep="\t",
    )
    original_uniann = original_scaling[
        (original_scaling.condition == "joint_s9p000_b0p000")
        & (original_scaling.method == "UniAnn")
    ].iloc[0]
    cole_aug = pd.read_csv(
        ROOT / "results/coleoptera_eviann_v1/existing_model_augmentation/coleoptera/strict_locus_metrics.tsv",
        sep="\t",
    )
    cole_model = pd.read_csv(
        ROOT / "results/coleoptera_eviann_v1/retrained/chrxtrain/w10/uniann_joint_t9/strict_locus_metrics.tsv",
        sep="\t",
    )

    rows = []
    for evidence, source, uniann in (
        ("Drosophila protein evidence", original_aug, original_uniann),
        ("Coleoptera protein evidence", cole_aug, cole_model[cole_model.method == "UniAnn"].iloc[0]),
    ):
        alone = source[source.method == "EviAnn"].iloc[0]
        combined = (
            source[source.method == "EviAnn+UniAnn"].iloc[0]
            if evidence.startswith("Drosophila")
            else cole_model[cole_model.method == "EviAnn+UniAnn"].iloc[0]
        )
        for method, record in (("EviAnn", alone), ("UniAnn", uniann), ("EviAnn+UniAnn", combined)):
            rows.append({
                "evidence": evidence,
                "method": method,
                **{name: float(record[name]) for name in ("sn", "pr", "f1")},
                **{name: int(record[name]) for name in (
                    "predicted_loci", "matched_loci", "missed_loci", "novel_loci"
                )},
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "strict_locus_metrics.tsv", sep="\t", index=False, float_format="%.9f")

    methods = ("EviAnn", "UniAnn", "EviAnn+UniAnn")
    figure, axes = plt.subplots(1, 3, figsize=(14.3, 4.7), sharey=True)
    x = np.arange(len(methods))
    width = 0.36
    for axis, (metric, title) in zip(
        axes, (("sn", "Sensitivity"), ("pr", "Precision"), ("f1", "F1"))
    ):
        for offset, (label, spec) in zip((-width / 2, width / 2), MODELS.items()):
            subset = table[table.evidence == label].set_index("method")
            values = [float(subset.loc[method, metric]) for method in methods]
            bars = axis.bar(x + offset, values, width, color=spec["color"], label=label)
            axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8.0)
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(x, methods, rotation=12, ha="right")
        axis.set_ylim(58, 95)
        axis.set_ylabel("Percent (%)")
        style(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, frameon=False, fontsize=8.5, ncol=2,
        loc="lower center", bbox_to_anchor=(0.5, -0.01),
    )
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUT / f"strict_locus_comparison.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)
    return table


def write_readme(candidate, states, locus):
    mean_ap = candidate.groupby("evidence").AUPRC.mean()
    lines = [
        "# Protein-evidence source comparison",
        "",
        "Both models use chrX-in-training, 10-kb windows, 5-kb stride, the four-state beta=0.1 head, and the same chrX+ RefSeq evaluation.",
        "UniAnn locus results use joint-softmax T=9 with unscaled donor/acceptor/start/stop scores.",
        "",
        "## Mean candidate AUPRC",
        "",
    ]
    for label, value in mean_ap.items():
        lines.append(f"- {label}: {value:.4f}")
    lines.extend(["", "## Strict locus F1", ""])
    for row in locus.itertuples():
        lines.append(f"- {row.evidence}; {row.method}: {row.f1:.2f}")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })
    truth, _, _ = reference_sites(
        "drosophila",
        ROOT / "data/raw/dmel_reference.gtf",
        ROOT / "data/raw/dmel_genome.fa",
        "NC_004354.4",
    )
    candidate = candidate_comparison(truth)
    states = four_state_comparison()
    locus = locus_comparison()
    write_readme(candidate, states, locus)
    print(OUT)


if __name__ == "__main__":
    main()
