#!/usr/bin/env python3
"""Summarize random-init, MLM baseline, and regularized MLM transfer runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_tsv(path, key):
    with path.open() as stream:
        return {row[key]: row for row in csv.DictReader(stream, delimiter="\t")}


def collect(root, name, tag, locus_override=None):
    run = root / "runs" / tag / "chrxtrain/w10/results.json"
    score = root / "artifacts/scores" / tag / "chrxtrain/w10/candidate_auprc.tsv"
    result = root / "results" / tag / "chrxtrain/w10"
    gap = result / "annotation_gap_metrics.tsv"
    joint = result / "joint_frame/joint_frame_auprc.tsv"
    locus = (
        root / locus_override if locus_override is not None
        else result / "uniann_joint_t9/strict_locus_metrics.tsv"
    )
    if not all(path.is_file() for path in (run, score, gap, joint, locus)):
        return None
    payload = json.loads(run.read_text())
    best = max(payload["history"], key=lambda row: row["mean_AP"])
    candidate = read_tsv(score, "task")
    missed = read_tsv(gap, "task")
    frames = read_tsv(joint, "target")
    if locus_override is None:
        loci = read_tsv(locus, "method")
    else:
        with locus.open() as stream:
            loci = {
                row["method"]: row
                for row in csv.DictReader(stream, delimiter="\t")
                if row["condition"] == "joint_s9p000_b0p000"
            }
    row = {
        "model": name,
        "tag": tag,
        "best_epoch": int(best["epoch"]),
        "stopped_epoch": int(payload["stopped_epoch"]),
        "training_hours": float(payload["elapsed_seconds"]) / 3600,
        "validation_mean_AP": float(best["mean_AP"]),
    }
    for task in ("donor", "acceptor", "start", "stop"):
        row[f"{task}_AP"] = float(candidate[task]["AUPRC"])
        row[f"missed_{task}_AP"] = float(missed[task]["missed_auprc"])
    for target, column in (
        ("CDS", "CDS_AP"), ("frame_0", "frame0_AP"),
        ("frame_1", "frame1_AP"), ("frame_2", "frame2_AP"),
    ):
        row[column] = float(frames[target]["AP"])
    for method, prefix in (("UniAnn", "uniann"), ("EviAnn+UniAnn", "combined")):
        for metric in ("sn", "pr", "f1"):
            row[f"{prefix}_{metric}"] = float(loci[method][metric])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    definitions = [
        ("Random init", "joint_frame_grid_v1/b010",
         "results/frame_scaling_extended_grid_v1/strict_locus_metrics.tsv"),
        ("Human MLM", "mlm_transfer_v1/human_grch38_w10", None),
        ("MLM + backbone LR 0.1x", "mlm_transfer_regularization_v1/backbone_lr010", None),
        ("MLM + freeze 4 (2 ep)", "mlm_transfer_regularization_v1/freeze4_e2_lr010", None),
        ("MLM + freeze 6 (2 ep)", "mlm_transfer_regularization_v1/freeze6_e2_lr010", None),
        ("MLM refinement", "mlm_transfer_regularization_v1/refinement", None),
    ]
    rows = [
        row for name, tag, locus_override in definitions
        if (row := collect(root, name, tag, locus_override))
    ]
    if not rows:
        raise RuntimeError("no complete runs to summarize")
    frame = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        args.output_dir / "all_metrics.tsv", sep="\t", index=False,
        float_format="%.9f",
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(frame)))
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    x = np.arange(len(frame))
    labels = frame["model"].tolist()
    axes[0, 0].bar(x, frame["validation_mean_AP"], color=colors)
    axes[0, 0].set_ylabel("Validation mean AP")
    axes[0, 0].set_ylim(max(0, frame["validation_mean_AP"].min() - 0.015), 1)

    width = 0.19
    for index, task in enumerate(("donor", "acceptor", "start", "stop")):
        axes[0, 1].bar(
            x + (index - 1.5) * width, frame[f"{task}_AP"], width,
            label=task.capitalize(),
        )
    axes[0, 1].set_ylabel("chrX+ candidate AUPRC")
    axes[0, 1].legend(frameon=False, ncol=2, fontsize=8)

    for index, task in enumerate(("donor", "acceptor", "start", "stop")):
        axes[1, 0].bar(
            x + (index - 1.5) * width, frame[f"missed_{task}_AP"], width,
            label=task.capitalize(),
        )
    axes[1, 0].set_ylabel("EviAnn-missed-site AUPRC")

    axes[1, 1].bar(x - 0.18, frame["uniann_f1"], 0.36, label="UniAnn")
    axes[1, 1].bar(
        x + 0.18, frame["combined_f1"], 0.36, label="EviAnn+UniAnn"
    )
    axes[1, 1].set_ylabel("Strict locus F1")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.22)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"mlm_transfer_regularization_comparison.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(frame[["model", "validation_mean_AP", "uniann_f1", "combined_f1"]])


if __name__ == "__main__":
    main()
