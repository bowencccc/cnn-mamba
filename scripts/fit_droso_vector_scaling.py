#!/usr/bin/env python3
"""Fit and independently evaluate scalar and vector calibration on chr3L.

The stored calibration sample contains logits/targets but no genomic metadata.
This script deterministically replays the original sampling indices against the
validation NPZ files, verifies every reconstructed target, and assigns samples
to alternating non-overlapping genomic blocks.  Even blocks fit calibration;
odd blocks are evaluation-only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


CLASS_NAMES = ("non_CDS", "frame_0", "frame_1", "frame_2")
COLORS = {"raw": "#777777", "scalar": "#C44E52", "vector": "#4C72B0"}


def targets_from_npz(path: Path, left: int, right: int):
    with np.load(path) as data:
        phase = torch.from_numpy(data["reference_phase_labels"][left:right].astype(np.int64))
        phase_mask = torch.from_numpy(data["reference_phase_mask"][left:right].astype(bool))
        cds = torch.from_numpy(data["reference_cds_labels"][left:right].astype(bool))
    valid = (~cds) | phase_mask
    targets = torch.zeros_like(phase, dtype=torch.long)
    coding = cds & phase_mask
    targets[coding] = phase[coding] + 1
    return targets, valid


def replay_sample(files, stored_targets, seed, samples_per_window, batch_size,
                  center_left, center_right, window_size, block_size):
    generator = torch.Generator().manual_seed(seed)
    replayed, block_ids, file_indices = [], [], []
    width = center_right - center_left
    dropped_windows = 0
    for batch_start in range(0, len(files), batch_size):
        batch_files = files[batch_start:batch_start + batch_size]
        target_rows, valid_rows = zip(*[
            targets_from_npz(path, center_left, center_right) for path in batch_files
        ])
        targets = torch.stack(target_rows)
        valid = torch.stack(valid_rows)
        indices = torch.randint(
            width, (len(batch_files), samples_per_window), generator=generator
        )
        rows = torch.arange(len(batch_files)).unsqueeze(1)
        chosen_valid = valid[rows, indices]
        replayed.append(targets[rows, indices][chosen_valid])

        batch_blocks = []
        batch_file_indices = []
        for local_index, path in enumerate(batch_files):
            _chrom, _strand, start_text = path.stem.rsplit("_", 2)
            start = int(start_text)
            left_block = start // block_size
            right_block = (start + window_size - 1) // block_size
            block_id = left_block if left_block == right_block else -1
            if block_id < 0:
                dropped_windows += 1
            batch_blocks.append(block_id)
            batch_file_indices.append(batch_start + local_index)
        expanded_blocks = torch.tensor(batch_blocks).unsqueeze(1).expand_as(indices)
        expanded_files = torch.tensor(batch_file_indices).unsqueeze(1).expand_as(indices)
        block_ids.append(expanded_blocks[chosen_valid])
        file_indices.append(expanded_files[chosen_valid])

    replayed = torch.cat(replayed).numpy().astype(np.uint8)
    block_ids = torch.cat(block_ids).numpy().astype(np.int32)
    file_indices = torch.cat(file_indices).numpy().astype(np.int32)
    if replayed.shape != stored_targets.shape or not np.array_equal(replayed, stored_targets):
        mismatch = int(np.sum(replayed != stored_targets)) if replayed.shape == stored_targets.shape else None
        raise RuntimeError(
            f"sampling replay did not match stored targets: replay={replayed.shape}, "
            f"stored={stored_targets.shape}, mismatches={mismatch}"
        )
    return block_ids, file_indices, dropped_windows


def fit_scalar(logits, targets):
    logits = torch.from_numpy(logits).double()
    targets = torch.from_numpy(targets).long()
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.25, max_iter=120, tolerance_grad=1e-11,
        tolerance_change=1e-13, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp(), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp())


def fit_vector(logits, targets, regularization):
    logits = torch.from_numpy(logits).double()
    targets = torch.from_numpy(targets).long()
    log_scale = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_scale, bias], lr=0.25, max_iter=200, tolerance_grad=1e-11,
        tolerance_change=1e-13, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        centered_bias = bias - bias.mean()
        transformed = logits * log_scale.exp() + centered_bias
        loss = F.cross_entropy(transformed, targets)
        loss = loss + regularization * (
            log_scale.square().mean() + centered_bias.square().mean()
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_scale.detach().exp().numpy(), (bias.detach() - bias.detach().mean()).numpy()


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def binned_rows(probability, truth, bins):
    edges = np.linspace(0, 1, bins + 1)
    indices = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    rows = []
    for index in range(bins):
        chosen = indices == index
        count = int(chosen.sum())
        positives = int(truth[chosen].sum()) if count else 0
        rows.append({
            "bin_index": index,
            "bin_lower": float(edges[index]),
            "bin_upper": float(edges[index + 1]),
            "count": count,
            "true_positives": positives,
            "mean_predicted_probability": float(probability[chosen].mean()) if count else None,
            "observed_positive_fraction": positives / count if count else None,
        })
    return rows


def metrics(probabilities, targets, bins):
    rows = np.arange(targets.size)
    clipped = np.clip(probabilities[rows, targets], 1e-30, 1)
    prediction = probabilities.argmax(1)
    one_hot = np.eye(4)[targets]
    confidence = probabilities.max(1)
    correctness = prediction == targets
    top_rows = binned_rows(confidence, correctness, bins)
    top_ece = sum(
        row["count"] / targets.size
        * abs(row["mean_predicted_probability"] - row["observed_positive_fraction"])
        for row in top_rows if row["count"]
    )
    per_class_ece = {}
    for index, name in enumerate(CLASS_NAMES):
        class_rows = binned_rows(probabilities[:, index], targets == index, bins)
        per_class_ece[name] = sum(
            row["count"] / targets.size
            * abs(row["mean_predicted_probability"] - row["observed_positive_fraction"])
            for row in class_rows if row["count"]
        )
    return {
        "samples": int(targets.size),
        "accuracy": float(correctness.mean()),
        "nll": float(-np.log(clipped).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(1).mean()),
        "top_label_ece": float(top_ece),
        "classwise_ece_mean": float(np.mean(list(per_class_ece.values()))),
        "classwise_ece": per_class_ece,
        "mean_max_probability": float(confidence.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--temperature-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--block-size", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--regularization", type=float, default=1e-4)
    args = parser.parse_args()

    metadata = json.loads(args.temperature_json.read_text())
    with np.load(args.sample) as sample:
        logits = sample["logits"].astype(np.float64)
        targets = sample["targets"].astype(np.uint8)
    files = sorted((args.data_root / "val").glob("*.npz"))
    block_ids, file_indices, dropped_windows = replay_sample(
        files, targets, int(metadata["seed"]), int(metadata["samples_per_window"]),
        args.batch_size, int(metadata["owned_center"][0]),
        int(metadata["owned_center"][1]), int(metadata["window_size"]),
        args.block_size,
    )
    retained = block_ids >= 0
    calibration = retained & (block_ids % 2 == 0)
    evaluation = retained & (block_ids % 2 == 1)
    if not calibration.any() or not evaluation.any():
        raise RuntimeError("genomic block split produced an empty subset")

    scalar_temperature = fit_scalar(logits[calibration], targets[calibration])
    vector_scale, vector_bias = fit_vector(
        logits[calibration], targets[calibration], args.regularization
    )
    eval_logits, eval_targets = logits[evaluation], targets[evaluation]
    probabilities = {
        "raw": softmax(eval_logits),
        "scalar": softmax(eval_logits / scalar_temperature),
        "vector": softmax(eval_logits * vector_scale + vector_bias),
    }
    metric_table = {
        condition: metrics(values, eval_targets, args.bins)
        for condition, values in probabilities.items()
    }
    raw_argmax = probabilities["raw"].argmax(1)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    calibration_json = output / "vector_scaling.json"
    payload = {
        "schema": "droso_joint_frame_vector_scaling_block_holdout_v1",
        "checkpoint": metadata["checkpoint"],
        "source_sample": str(args.sample.resolve()),
        "validation_chromosome": "3L",
        "block_size_bp": args.block_size,
        "split_rule": "even non-crossing blocks fit; odd non-crossing blocks evaluate",
        "dropped_boundary_crossing_windows": dropped_windows,
        "calibration_samples": int(calibration.sum()),
        "evaluation_samples": int(evaluation.sum()),
        "calibration_class_counts": {
            name: int((targets[calibration] == index).sum())
            for index, name in enumerate(CLASS_NAMES)
        },
        "evaluation_class_counts": {
            name: int((eval_targets == index).sum())
            for index, name in enumerate(CLASS_NAMES)
        },
        "scalar_temperature_fit_on_calibration_blocks": scalar_temperature,
        "vector_scale": dict(zip(CLASS_NAMES, map(float, vector_scale))),
        "vector_bias": dict(zip(CLASS_NAMES, map(float, vector_bias))),
        "vector_regularization": args.regularization,
        "evaluation_metrics": metric_table,
        "evaluation_argmax_changes_vs_raw": {
            condition: int((values.argmax(1) != raw_argmax).sum())
            for condition, values in probabilities.items()
        },
    }
    calibration_json.write_text(json.dumps(payload, indent=2) + "\n")

    with (output / "evaluation_metrics.tsv").open("w", newline="") as stream:
        fields = [
            "condition", "samples", "accuracy", "nll", "brier",
            "top_label_ece", "classwise_ece_mean", "mean_max_probability",
            *[f"ece_{name}" for name in CLASS_NAMES], "argmax_changes_vs_raw",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for condition, row in metric_table.items():
            writer.writerow({
                "condition": condition,
                **{key: row[key] for key in fields if key in row},
                **{f"ece_{name}": row["classwise_ece"][name] for name in CLASS_NAMES},
                "argmax_changes_vs_raw": payload["evaluation_argmax_changes_vs_raw"][condition],
            })

    reliability_rows = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        for condition, values in probabilities.items():
            for row in binned_rows(values[:, class_index], eval_targets == class_index, args.bins):
                reliability_rows.append({
                    "state": class_name, "state_index": class_index,
                    "condition": condition, **row,
                })
    with (output / f"evaluation_classwise_calibration_{args.bins}bins.tsv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(reliability_rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(reliability_rows)

    figure, axes = plt.subplots(2, 2, figsize=(10.8, 9.2), sharex=True, sharey=True)
    for class_index, (axis, class_name) in enumerate(zip(axes.flat, CLASS_NAMES)):
        axis.plot([0, 1], [0, 1], "--", color="0.65", linewidth=1,
                  label="Perfect calibration")
        for condition, label in (
            ("raw", "Raw"),
            ("scalar", f"Scalar (T={scalar_temperature:.3f})"),
            ("vector", "Vector scaling"),
        ):
            rows = binned_rows(
                probabilities[condition][:, class_index], eval_targets == class_index,
                args.bins,
            )
            rows = [row for row in rows if row["count"]]
            x = np.asarray([row["mean_predicted_probability"] for row in rows])
            y = np.asarray([row["observed_positive_fraction"] for row in rows])
            counts = np.asarray([row["count"] for row in rows])
            sizes = 9 + 22 * np.log1p(counts) / np.log1p(counts.max())
            order = np.argsort(x)
            axis.plot(x[order], y[order], color=COLORS[condition], linewidth=1.1,
                      alpha=0.85, label=label)
            axis.scatter(x, y, s=sizes, color=COLORS[condition], alpha=0.65,
                         edgecolors="white", linewidths=0.3)
        axis.set_title(
            f"{class_name}  (evaluation positives: {(eval_targets == class_index).sum():,})"
        )
        axis.grid(alpha=0.2)
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.set_aspect("equal", adjustable="box")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    for axis in axes[-1, :]:
        axis.set_xlabel("Predicted probability")
    for axis in axes[:, 0]:
        axis.set_ylabel("Observed fraction of the state")
    figure.suptitle(
        f"Drosophila chr3L — held-block classwise calibration ({args.bins} bins)",
        fontsize=14,
    )
    figure.text(
        0.5, 0.014,
        "Calibration parameters fit on even 2-Mb blocks; curves use odd 2-Mb blocks only.",
        ha="center", fontsize=9, color="0.3",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.97))
    stem = output / f"held_block_classwise_calibration_{args.bins}bins"
    figure.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    print(json.dumps({
        "scalar_temperature": scalar_temperature,
        "vector_scale": payload["vector_scale"],
        "vector_bias": payload["vector_bias"],
        "calibration_samples": payload["calibration_samples"],
        "evaluation_samples": payload["evaluation_samples"],
        "evaluation_metrics": metric_table,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
