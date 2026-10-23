#!/usr/bin/env python3
"""Plot 10-bin classwise calibration for the three retained scaling rules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit_droso_vector_scaling import (
    CLASS_NAMES,
    binned_rows,
    replay_sample,
    softmax,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    ROOT / "results/joint_frame_grid_v1/b010/chrxtrain/w10"
)
SAMPLE = RESULT_ROOT / "temperature_scaling/calibration_sample.npz"
TEMPERATURE_JSON = RESULT_ROOT / "temperature_scaling/temperature.json"
VECTOR_ROOT = RESULT_ROOT / "vector_scaling_block_holdout"
VECTOR_JSON = VECTOR_ROOT / "vector_scaling.json"
DATA_ROOT = ROOT / "data/processed/cds_gating/w10_frame"

BLOCK_SIZE = 2_000_000
BATCH_SIZE = 8
UNIANN_TEMPERATURE = 7.0

CONDITIONS = (
    ("raw", "Raw", "#777777"),
    ("vector", "Vector scaling", "#4C72B0"),
    ("uniann", "UniAnn built-in", "#DD8452"),
)


def binary_odds_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply UniAnn's independent per-score odds-temperature transform."""
    return 1.0 / (
        1.0
        + np.power(
            (1.0 - probabilities) / (probabilities + 1e-10),
            1.0 / temperature,
        )
    )


def compact_count(count: int) -> str:
    """Return a compact nucleotide count that remains legible on the plot."""
    if count >= 1_000_000:
        return f"n={count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"n={count / 1_000:.1f}k"
    return f"n={count}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    bins = args.bins

    sampling = json.loads(TEMPERATURE_JSON.read_text())
    fitted = json.loads(VECTOR_JSON.read_text())
    with np.load(SAMPLE) as sample:
        logits = sample["logits"].astype(np.float64)
        targets = sample["targets"].astype(np.uint8)

    files = sorted((DATA_ROOT / "val").glob("*.npz"))
    block_ids, _file_indices, _dropped = replay_sample(
        files,
        targets,
        int(sampling["seed"]),
        int(sampling["samples_per_window"]),
        BATCH_SIZE,
        int(sampling["owned_center"][0]),
        int(sampling["owned_center"][1]),
        int(sampling["window_size"]),
        BLOCK_SIZE,
    )
    evaluation = (block_ids >= 0) & (block_ids % 2 == 1)
    eval_logits = logits[evaluation]
    eval_targets = targets[evaluation]

    vector_scale = np.asarray(
        [fitted["vector_scale"][name] for name in CLASS_NAMES], dtype=np.float64
    )
    vector_bias = np.asarray(
        [fitted["vector_bias"][name] for name in CLASS_NAMES], dtype=np.float64
    )
    raw = softmax(eval_logits)
    probabilities = {
        "raw": raw,
        "vector": softmax(eval_logits * vector_scale + vector_bias),
        "uniann": binary_odds_temperature(raw, UNIANN_TEMPERATURE),
    }

    class_index = 0
    class_name = CLASS_NAMES[class_index]
    rows_for_tsv = []
    condition_rows = {}
    figure_width = max(13.8, 3.0 + 0.75 * bins)
    figure = plt.figure(figsize=(figure_width, 11.2))
    layout = figure.add_gridspec(2, 1, height_ratios=(4.8, 1.35), hspace=0.30)
    axis = figure.add_subplot(layout[0])
    table_axis = figure.add_subplot(layout[1])
    axis.plot(
        [0, 1], [0, 1], "--", color="0.72", linewidth=1.2,
        label="Perfect calibration",
    )
    for condition, label, color in CONDITIONS:
        rows = binned_rows(
            probabilities[condition][:, class_index],
            eval_targets == class_index,
            bins,
        )
        condition_rows[condition] = rows
        for row in rows:
            rows_for_tsv.append({
                "state": class_name,
                "state_index": class_index,
                "condition": label,
                **row,
            })
        occupied = [row for row in rows if row["count"]]
        x = np.asarray([row["mean_predicted_probability"] for row in occupied])
        y = np.asarray([row["observed_positive_fraction"] for row in occupied])
        order = np.argsort(x)
        axis.plot(
            x[order], y[order], color=color, linewidth=2.0,
            alpha=0.95, label=label,
        )
        axis.scatter(
            x, y, s=52, color=color, alpha=0.9,
            edgecolors="white", linewidths=0.6,
        )
    axis.set_title("non_CDS", fontsize=15)
    axis.set_xlabel("Mean predicted probability in bin", fontsize=12)
    axis.set_ylabel("Observed non_CDS fraction", fontsize=12)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks(np.linspace(0, 1, 11))
    axis.set_yticks(np.linspace(0, 1, 11))
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=10, loc="lower right")

    bin_labels = [
        f"{index / bins:.2f}–{(index + 1) / bins:.2f}"
        for index in range(bins)
    ]
    row_labels = [label for _condition, label, _color in CONDITIONS]
    cell_text = [
        [f'{row["count"]:,}' for row in condition_rows[condition]]
        for condition, _label, _color in CONDITIONS
    ]
    row_colors = [color for _condition, _label, color in CONDITIONS]
    cell_colors = [
        [(*plt.matplotlib.colors.to_rgb(color), 0.13)] * bins
        for color in row_colors
    ]
    table_axis.axis("off")
    table = table_axis.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=bin_labels,
        cellColours=cell_colors,
        rowColours=[(*plt.matplotlib.colors.to_rgb(color), 0.22) for color in row_colors],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2 if bins <= 10 else 7.8)
    table.scale(1.0, 1.65)
    figure.subplots_adjust(left=0.14, right=0.98, bottom=0.04, top=0.95)

    stem = VECTOR_ROOT / f"non_cds_calibration_{bins}bins_counts"
    figure.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    tsv_path = VECTOR_ROOT / f"non_cds_calibration_{bins}bins_counts.tsv"
    with tsv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows_for_tsv[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows_for_tsv)

    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(tsv_path)


if __name__ == "__main__":
    main()
