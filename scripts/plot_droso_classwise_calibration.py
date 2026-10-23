#!/usr/bin/env python3
"""Plot classwise reliability for the Drosophila four-state frame head."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = ("non_CDS", "frame_0", "frame_1", "frame_2")
COLORS = {"raw": "#C44E52", "temperature_scaled": "#4C72B0"}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def bin_class(probability: np.ndarray, truth: np.ndarray, bins: int):
    edges = np.linspace(0.0, 1.0, bins + 1)
    # A probability of exactly one belongs to the final bin.
    indices = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    rows = []
    for index in range(bins):
        selected = indices == index
        count = int(selected.sum())
        positives = int(truth[selected].sum()) if count else 0
        rows.append({
            "bin_index": index,
            "bin_lower": float(edges[index]),
            "bin_upper": float(edges[index + 1]),
            "bin_center": float((edges[index] + edges[index + 1]) / 2),
            "count": count,
            "true_positives": positives,
            "mean_predicted_probability": (
                float(probability[selected].mean()) if count else np.nan
            ),
            "observed_positive_fraction": positives / count if count else np.nan,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--temperature-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()

    if args.bins < 2:
        parser.error("--bins must be at least two")

    with np.load(args.sample) as sample:
        logits = sample["logits"].astype(np.float64)
        targets = sample["targets"].astype(np.int64)
    metadata = json.loads(args.temperature_json.read_text())
    temperature = float(metadata["temperature"])

    conditions = {
        "raw": softmax(logits),
        "temperature_scaled": softmax(logits / temperature),
    }
    all_rows = []
    class_rows = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        truth = targets == class_index
        class_rows[class_name] = {}
        for condition, probabilities in conditions.items():
            rows = bin_class(probabilities[:, class_index], truth, args.bins)
            class_rows[class_name][condition] = rows
            for row in rows:
                all_rows.append({
                    "state": class_name,
                    "state_index": class_index,
                    "condition": condition,
                    **row,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tsv = args.output_dir / f"classwise_calibration_{args.bins}bins.tsv"
    with tsv.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(all_rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(all_rows)

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 9.2), sharex=True, sharey=True)
    for class_index, (axis, class_name) in enumerate(zip(axes.flat, CLASS_NAMES)):
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1,
                  label="Perfect calibration")
        for condition, label in (("raw", "Before (T=1)"),
                                 ("temperature_scaled", f"After (T={temperature:.3f})")):
            rows = [row for row in class_rows[class_name][condition] if row["count"]]
            x = np.asarray([row["mean_predicted_probability"] for row in rows])
            y = np.asarray([row["observed_positive_fraction"] for row in rows])
            counts = np.asarray([row["count"] for row in rows])
            sizes = 10 + 25 * np.log1p(counts) / np.log1p(counts.max())
            order = np.argsort(x)
            axis.plot(x[order], y[order], color=COLORS[condition], linewidth=1.15,
                      alpha=0.9, label=label)
            axis.scatter(x, y, s=sizes, color=COLORS[condition], alpha=0.72,
                         edgecolors="white", linewidths=0.35)
        positives = int((targets == class_index).sum())
        axis.set_title(f"{class_name}  (true bases: {positives:,})")
        axis.grid(alpha=0.22)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_aspect("equal", adjustable="box")

    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    for axis in axes[-1, :]:
        axis.set_xlabel("Predicted probability")
    for axis in axes[:, 0]:
        axis.set_ylabel("Observed fraction of the state")
    figure.suptitle(
        f"Drosophila chr3L validation — classwise calibration ({args.bins} bins)",
        fontsize=14,
    )
    figure.text(
        0.5, 0.015,
        "Each point groups bases by that state's predicted probability; marker size reflects bin count.",
        ha="center", fontsize=9, color="0.3",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))
    stem = args.output_dir / f"classwise_calibration_{args.bins}bins"
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    print(json.dumps({
        "temperature": temperature,
        "bins": args.bins,
        "sampled_bases": int(targets.size),
        "outputs": [str(tsv), str(stem.with_suffix('.png')), str(stem.with_suffix('.pdf'))],
    }, indent=2))


if __name__ == "__main__":
    main()
