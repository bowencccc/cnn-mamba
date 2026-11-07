#!/usr/bin/env python3
"""Fit diagonal vector scaling from a saved joint-frame calibration sample.

The four-state class order is non-CDS, frame 0, frame 1, frame 2.  The
transformation is ``z'_c = scale_c * z_c + bias_c`` followed by softmax.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CLASS_NAMES = ("non_CDS", "frame_0", "frame_1", "frame_2")


def fit_vector(logits: np.ndarray, targets: np.ndarray, regularization: float):
    logits_tensor = torch.from_numpy(logits).double()
    targets_tensor = torch.from_numpy(targets).long()
    log_scale = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    bias = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_scale, bias], lr=0.25, max_iter=200,
        tolerance_grad=1e-11, tolerance_change=1e-13,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        centered_bias = bias - bias.mean()
        calibrated = logits_tensor * log_scale.exp() + centered_bias
        loss = F.cross_entropy(calibrated, targets_tensor)
        loss = loss + regularization * (
            log_scale.square().mean() + centered_bias.square().mean()
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        scale = log_scale.exp()
        centered_bias = bias - bias.mean()
    return scale.numpy(), centered_bias.numpy()


def metrics(logits: np.ndarray, targets: np.ndarray):
    tensor = torch.from_numpy(logits).double()
    target = torch.from_numpy(targets).long()
    with torch.no_grad():
        probabilities = tensor.softmax(-1)
        prediction = probabilities.argmax(-1)
        selected = probabilities[torch.arange(target.numel()), target].clamp_min(1e-30)
        one_hot = F.one_hot(target, num_classes=4)
        return {
            "samples": int(target.numel()),
            "accuracy": float((prediction == target).double().mean()),
            "nll": float(-selected.log().mean()),
            "brier": float((probabilities - one_hot).square().sum(-1).mean()),
            "mean_max_probability": float(probabilities.max(-1).values.mean()),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--regularization", type=float, default=1e-4)
    args = parser.parse_args()

    metadata = json.loads(args.metadata_json.read_text())
    with np.load(args.sample) as sample:
        logits = sample["logits"].astype(np.float64)
        targets = sample["targets"].astype(np.int64)
    if logits.ndim != 2 or logits.shape[1] != 4:
        raise RuntimeError(f"expected [N, 4] logits, got {logits.shape}")
    if targets.shape != (logits.shape[0],):
        raise RuntimeError(f"target shape {targets.shape} does not match {logits.shape}")

    scale, bias = fit_vector(logits, targets, args.regularization)
    calibrated_logits = logits * scale + bias
    before = metrics(logits, targets)
    after = metrics(calibrated_logits, targets)
    raw_argmax = logits.argmax(1)
    calibrated_argmax = calibrated_logits.argmax(1)

    payload = {
        "schema": "joint_frame_vector_scaling_v1",
        "method": "classwise positive scale and centered bias fitted by empirical multiclass NLL",
        "checkpoint": metadata["checkpoint"],
        "checkpoint_epoch": metadata["checkpoint_epoch"],
        "data_root": metadata["data_root"],
        "split": metadata["split"],
        "class_order": list(CLASS_NAMES),
        "source_sample": str(args.sample.resolve()),
        "sampled_bases": int(targets.size),
        "class_counts": {
            name: int((targets == index).sum())
            for index, name in enumerate(CLASS_NAMES)
        },
        "objective": "empirical_multiclass_nll",
        "vector_scale": dict(zip(CLASS_NAMES, map(float, scale))),
        "vector_bias": dict(zip(CLASS_NAMES, map(float, bias))),
        "vector_regularization": args.regularization,
        "before": before,
        "after": after,
        "argmax_changes": int((raw_argmax != calibrated_argmax).sum()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "vector_scaling.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output_dir / "vector_scaling_metrics.tsv").open("w", newline="") as stream:
        fields = ("condition", "samples", "accuracy", "nll", "brier", "mean_max_probability")
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow({"condition": "raw", **before})
        writer.writerow({"condition": "vector_scaled", **after})
    print(json.dumps({
        "vector_scale": payload["vector_scale"],
        "vector_bias": payload["vector_bias"],
        "before": before,
        "after": after,
        "argmax_changes": payload["argmax_changes"],
        "output": str(output_json),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
