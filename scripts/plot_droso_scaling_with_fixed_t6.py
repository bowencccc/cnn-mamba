#!/usr/bin/env python3
"""Compare multiclass scaling with an independent binary-odds T=7 rule."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit_droso_vector_scaling import (
    CLASS_NAMES, binned_rows, metrics, replay_sample, softmax,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = (
    ROOT / "results/joint_frame_grid_v1/b010/chrxtrain/w10/"
    "temperature_scaling/calibration_sample.npz"
)
TEMPERATURE_JSON = (
    ROOT / "results/joint_frame_grid_v1/b010/chrxtrain/w10/"
    "temperature_scaling/temperature.json"
)
VECTOR_JSON = (
    ROOT / "results/joint_frame_grid_v1/b010/chrxtrain/w10/"
    "vector_scaling_block_holdout/vector_scaling.json"
)
DATA_ROOT = ROOT / "data/processed/cds_gating/w10_frame"
OUTPUT = VECTOR_JSON.parent
FIXED_TEMPERATURES = {
    "fixed_t6": 6.0,
    "fixed_t7": 7.0,
}
BINS = 50
BLOCK_SIZE = 2_000_000
BATCH_SIZE = 8

COLORS = {
    "raw": "#777777",
    "scalar": "#C44E52",
    "fixed_t6": "#55A868",
    "fixed_t7": "#8172B3",
    "binary_odds_t7": "#DD8452",
    "vector": "#4C72B0",
}


def binary_odds_temperature(probabilities, temperature):
    """Apply the user's independent p/(1-p) temperature formula."""
    return 1.0 / (
        1.0
        + np.power(
            (1.0 - probabilities) / (probabilities + 1e-10),
            1.0 / temperature,
        )
    )


def main():
    sampling = json.loads(TEMPERATURE_JSON.read_text())
    fitted = json.loads(VECTOR_JSON.read_text())
    with np.load(SAMPLE) as sample:
        logits = sample["logits"].astype(np.float64)
        targets = sample["targets"].astype(np.uint8)

    files = sorted((DATA_ROOT / "val").glob("*.npz"))
    block_ids, _file_indices, _dropped = replay_sample(
        files, targets, int(sampling["seed"]),
        int(sampling["samples_per_window"]), BATCH_SIZE,
        int(sampling["owned_center"][0]), int(sampling["owned_center"][1]),
        int(sampling["window_size"]), BLOCK_SIZE,
    )
    evaluation = (block_ids >= 0) & (block_ids % 2 == 1)
    eval_logits = logits[evaluation]
    eval_targets = targets[evaluation]

    learned_temperature = float(
        fitted["scalar_temperature_fit_on_calibration_blocks"]
    )
    vector_scale = np.asarray(
        [fitted["vector_scale"][name] for name in CLASS_NAMES],
        dtype=np.float64,
    )
    vector_bias = np.asarray(
        [fitted["vector_bias"][name] for name in CLASS_NAMES],
        dtype=np.float64,
    )
    raw_probabilities = softmax(eval_logits)
    probabilities = {
        "raw": raw_probabilities,
        "scalar": softmax(eval_logits / learned_temperature),
        **{
            name: softmax(eval_logits / temperature)
            for name, temperature in FIXED_TEMPERATURES.items()
        },
        "binary_odds_t7": binary_odds_temperature(raw_probabilities, 7.0),
        "vector": softmax(eval_logits * vector_scale + vector_bias),
    }
    labels = {
        "raw": "Raw",
        "scalar": f"Learned scalar (T={learned_temperature:.3f})",
        "fixed_t6": "Fixed scalar (T=6)",
        "fixed_t7": "Fixed scalar (T=7)",
        "binary_odds_t7": "Independent binary-odds (T=7)",
        "vector": "Vector scaling",
    }
    metric_table = {
        condition: metrics(values, eval_targets, BINS)
        for condition, values in probabilities.items()
        if condition != "binary_odds_t7"
    }
    raw_argmax = probabilities["raw"].argmax(1)

    metric_path = OUTPUT / "evaluation_metrics_with_fixed_T6_T7_binary_odds_T7.tsv"
    with metric_path.open("w", newline="") as stream:
        fields = [
            "condition", "temperature", "samples", "accuracy", "nll",
            "brier", "top_label_ece", "classwise_ece_mean",
            "mean_max_probability", "argmax_changes_vs_raw",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for condition, row in metric_table.items():
            temperature = (
                learned_temperature if condition == "scalar"
                else FIXED_TEMPERATURES[condition]
                if condition in FIXED_TEMPERATURES
                else 1.0 if condition == "raw" else ""
            )
            writer.writerow({
                "condition": condition,
                "temperature": temperature,
                **{name: row[name] for name in fields if name in row},
                "argmax_changes_vs_raw": int(
                    (probabilities[condition].argmax(1) != raw_argmax).sum()
                ),
            })

    rows_for_tsv = []
    figure, axes = plt.subplots(
        2, 2, figsize=(10.8, 9.2), sharex=True, sharey=True
    )
    for class_index, (axis, class_name) in enumerate(zip(axes.flat, CLASS_NAMES)):
        axis.plot(
            [0, 1], [0, 1], "--", color="0.7", linewidth=1,
            label="Perfect calibration",
        )
        for condition, values in probabilities.items():
            rows = binned_rows(
                values[:, class_index], eval_targets == class_index, BINS
            )
            for row in rows:
                rows_for_tsv.append({
                    "state": class_name,
                    "state_index": class_index,
                    "condition": condition,
                    **row,
                })
            occupied = [row for row in rows if row["count"]]
            x = np.asarray([
                row["mean_predicted_probability"] for row in occupied
            ])
            y = np.asarray([
                row["observed_positive_fraction"] for row in occupied
            ])
            counts = np.asarray([row["count"] for row in occupied])
            sizes = 8 + 20 * np.log1p(counts) / np.log1p(counts.max())
            order = np.argsort(x)
            axis.plot(
                x[order], y[order], color=COLORS[condition], linewidth=1.15,
                alpha=0.9, label=labels[condition],
            )
            axis.scatter(
                x, y, s=sizes, color=COLORS[condition], alpha=0.6,
                edgecolors="white", linewidths=0.3,
            )
        axis.set_title(
            f"{class_name}  "
            f"(evaluation positives: {(eval_targets == class_index).sum():,})"
        )
        axis.grid(alpha=0.2)
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.set_aspect("equal", adjustable="box")
    axes[0, 0].legend(frameon=False, fontsize=7.6, loc="lower right")
    for axis in axes[-1, :]:
        axis.set_xlabel("Predicted probability")
    for axis in axes[:, 0]:
        axis.set_ylabel("Observed fraction of the state")
    figure.tight_layout()
    stem = (
        OUTPUT /
        "held_block_classwise_calibration_50bins_with_fixed_T6_T7_binary_odds_T7"
    )
    figure.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    bin_path = (
        OUTPUT /
        "evaluation_classwise_calibration_50bins_with_fixed_T6_T7_binary_odds_T7.tsv"
    )
    with bin_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows_for_tsv[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows_for_tsv)

    print(stem.with_suffix(".png"))
    print(metric_path)
    print(
        "Independent binary-odds T=7 is excluded from multiclass metrics "
        "because its four outputs do not sum to one."
    )
    print(json.dumps(metric_table, indent=2))


if __name__ == "__main__":
    main()
