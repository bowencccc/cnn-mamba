#!/usr/bin/env python3
"""Evaluate paired NetStart predictions for spliced versus genomic inputs."""

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
from sklearn.metrics import average_precision_score, precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/human_spliced_start_pilot_v1/input"
DEFAULT_OUTPUT = ROOT / "results/human_spliced_start_pilot_v1/evaluation"


def metrics(frame: pd.DataFrame, mode: str) -> dict[str, float | int | str]:
    subset = frame[frame["mode"] == mode].copy()
    labels = subset.is_true_start.to_numpy(dtype=np.int8)
    scores = subset.preds.to_numpy(dtype=float)
    ap = float(average_precision_score(labels, scores))
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.nanargmax(f1))
    threshold = float(thresholds[best]) if best < len(thresholds) else 1.0

    winners = subset.loc[subset.groupby("entry_line").preds.idxmax()]
    ranks = []
    for _, group in subset.groupby("entry_line"):
        truth_score = float(group.loc[group.is_true_start, "preds"].iloc[0])
        ranks.append(1 + int((group.preds > truth_score).sum()))
    return {
        "mode": mode,
        "transcripts": int(subset.entry_line.nunique()),
        "candidates": len(subset),
        "candidate_positive_rate": float(labels.mean()),
        "candidate_auprc": ap,
        "top1_accuracy": float(winners.is_true_start.mean()),
        "mean_reciprocal_rank": float(np.mean([1 / rank for rank in ranks])),
        "median_true_rank": float(np.median(ranks)),
        "best_f1": float(f1[best]),
        "best_precision": float(precision[best]),
        "best_recall": float(recall[best]),
        "best_threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_INPUT / "metadata.tsv")
    parser.add_argument("--spliced", type=Path, required=True)
    parser.add_argument("--genomic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(args.metadata, sep="\t")
    predictions = []
    for mode, path in (("spliced", args.spliced), ("genomic", args.genomic)):
        frame = pd.read_csv(path)
        frame["mode"] = mode
        predictions.append(frame)
    frame = pd.concat(predictions, ignore_index=True)
    frame = frame.merge(
        metadata[["entry_line", "pair_id", "mode", "transcript_id", "true_atg_position_1based"]],
        on=["entry_line", "mode"], how="inner", validate="many_to_one",
    )
    frame["is_true_start"] = frame.atg_position == frame.true_atg_position_1based
    truth_counts = frame.groupby("entry_line").is_true_start.sum()
    if not (truth_counts == 1).all():
        raise RuntimeError(f"Expected exactly one true ATG per entry; counts={truth_counts.value_counts().to_dict()}")

    rows = [metrics(frame, mode) for mode in ("genomic", "spliced")]
    pd.DataFrame(rows).to_csv(args.output_dir / "summary.tsv", sep="\t", index=False)
    frame.to_csv(args.output_dir / "candidate_predictions.tsv.gz", sep="\t", index=False, compression="gzip")

    paired = []
    for (pair_id, mode), group in frame.groupby(["pair_id", "mode"]):
        top = group.loc[group.preds.idxmax()]
        true = group[group.is_true_start].iloc[0]
        paired.append({
            "pair_id": pair_id,
            "mode": mode,
            "transcript_id": true.transcript_id,
            "candidate_count": len(group),
            "top1_correct": int(top.is_true_start),
            "true_score": float(true.preds),
            "true_rank": 1 + int((group.preds > true.preds).sum()),
        })
    paired_frame = pd.DataFrame(paired)
    paired_frame.to_csv(args.output_dir / "per_transcript.tsv", sep="\t", index=False)

    plot_values = pd.DataFrame(rows).set_index("mode")
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    fields = ["top1_accuracy", "candidate_auprc", "mean_reciprocal_rank"]
    titles = ["Top-1 accuracy", "Candidate AUPRC", "Mean reciprocal rank"]
    colors = ["#A0A0A0", "#4C78A8"]
    for axis, field, title in zip(axes, fields, titles):
        values = [plot_values.loc[mode, field] for mode in ("genomic", "spliced")]
        axis.bar(["With introns", "Spliced"], values, color=colors)
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
        for index, value in enumerate(values):
            axis.text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=10)
    fig.suptitle("Human chr1 paired start-codon pilot (RefSeq exon-chain oracle)")
    fig.tight_layout()
    fig.savefig(args.output_dir / "spliced_vs_genomic.png", dpi=220)
    fig.savefig(args.output_dir / "spliced_vs_genomic.pdf")
    plt.close(fig)

    report = {
        "design": "same RefSeq transcripts and starts; only intron removal differs",
        "scope": "diagnostic oracle using RefSeq exon chains, not a deployable EviAnn result",
        "metrics": rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
