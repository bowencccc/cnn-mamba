#!/usr/bin/env python3
"""Fit scalar temperature scaling for a four-state joint-frame head."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from .model import model_from_checkpoint
from .temperature_scaling import (
    CLASS_ORDER, calibration_metrics, fit_temperature, joint_frame_targets,
)
from .train import PUDataset


def collect_sample(model, loader, device, center_left, center_right,
                   samples_per_window, seed, amp):
    """Uniformly sample owned center bases while performing one validation pass."""
    generator = torch.Generator().manual_seed(seed)
    sampled_logits, sampled_targets = [], []
    seen_valid = seen_coding = 0
    started = time.time()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, 1):
            sequence = batch[0].to(device, non_blocking=True)
            with autocast("cuda", enabled=amp):
                logits = model(sequence, return_joint_frame=True)[2]
            targets, valid = joint_frame_targets(batch[9], batch[10], batch[12])
            logits = logits[:, center_left:center_right]
            targets = targets[:, center_left:center_right]
            valid = valid[:, center_left:center_right]
            seen_valid += int(valid.sum())
            seen_coding += int(((targets != 0) & valid).sum())

            batch_size, width = targets.shape
            indices = torch.randint(
                width, (batch_size, samples_per_window), generator=generator
            )
            rows = torch.arange(batch_size).unsqueeze(1)
            chosen_valid = valid[rows, indices]
            chosen_targets = targets[rows, indices][chosen_valid]
            chosen_logits = logits.float().cpu()[rows, indices][chosen_valid]
            sampled_logits.append(chosen_logits)
            sampled_targets.append(chosen_targets)
            if batch_index % 500 == 0 or batch_index == len(loader):
                print(
                    f"batches={batch_index:,}/{len(loader):,} "
                    f"sampled={sum(item.shape[0] for item in sampled_targets):,}",
                    flush=True,
                )
    return (
        torch.cat(sampled_logits), torch.cat(sampled_targets),
        {"valid_owned_bases_seen": seen_valid,
         "coding_owned_bases_seen": seen_coding,
         "inference_seconds": time.time() - started},
    )


def plot_calibration(before, after, output):
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.7))
    axes[0].plot([0, 1], [0, 1], color="0.65", linestyle="--", linewidth=1)
    for metrics, label, color in (
        (before, "Before (T=1)", "#C44E52"),
        (after, f"After (T={after['temperature']:.3f})", "#4C72B0"),
    ):
        points = [row for row in metrics["reliability_bins"] if row["count"]]
        axes[0].plot(
            [row["mean_confidence"] for row in points],
            [row["accuracy"] for row in points],
            marker="o", markersize=3, linewidth=1.4, label=label, color=color,
        )
    axes[0].set(xlabel="Mean confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(frameon=False, fontsize=8)

    centers = np.asarray([
        (row["lower"] + row["upper"]) / 2
        for row in before["reliability_bins"]
    ])
    width = 0.8 / len(centers)
    for metrics, label, color, offset in (
        (before, "Before", "#C44E52", -width / 2),
        (after, "After", "#4C72B0", width / 2),
    ):
        counts = np.asarray([row["count"] for row in metrics["reliability_bins"]])
        axes[1].bar(
            centers + offset, counts / counts.sum(), width=width,
            label=label, color=color,
        )
    axes[1].set(xlabel="Maximum class probability", ylabel="Fraction of bases",
                xlim=(0.2, 1.0))
    axes[1].legend(frameon=False, fontsize=8)

    names = ["NLL", "Brier", "ECE", "Mean max P"]
    before_values = [before["nll"], before["brier"], before["ece"],
                     before["mean_max_probability"]]
    after_values = [after["nll"], after["brier"], after["ece"],
                    after["mean_max_probability"]]
    positions = np.arange(len(names))
    axes[2].bar(positions - 0.18, before_values, 0.36, label="Before", color="#C44E52")
    axes[2].bar(positions + 0.18, after_values, 0.36, label="After", color="#4C72B0")
    axes[2].set_xticks(positions, names, rotation=18, ha="right")
    axes[2].set_ylabel("Metric value")
    axes[2].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-window", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--save-sample", action="store_true")
    parser.add_argument("--minimum-temperature", type=float, default=0.05)
    parser.add_argument("--maximum-temperature", type=float, default=100.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to collect calibration logits")
    if args.samples_per_window < 1:
        parser.error("--samples-per-window must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if not any(
        name.startswith("joint_frame_head.")
        for name in checkpoint["model_state_dict"]
    ):
        raise RuntimeError("checkpoint does not contain a four-state joint-frame head")
    window, stride = int(config["window_size"]), int(config["stride"])
    if stride > window:
        raise RuntimeError("stride cannot exceed window size")
    center_left = (window - stride) // 2
    center_right = center_left + stride
    dataset = PUDataset(
        args.data_root / args.split, require_phase=True, require_cds=True
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    device = torch.device("cuda")
    model = model_from_checkpoint(checkpoint, device).eval()
    logits, targets, collection = collect_sample(
        model, loader, device, center_left, center_right,
        args.samples_per_window, args.seed, args.amp,
    )
    noncoding_weight = float(config.get("joint_noncoding_weight", 0.1))
    empirical_temperature = fit_temperature(
        logits, targets, "empirical", noncoding_weight,
        args.minimum_temperature, args.maximum_temperature,
    )
    balanced_temperature = fit_temperature(
        logits, targets, "group_balanced", noncoding_weight,
        args.minimum_temperature, args.maximum_temperature,
    )
    before = calibration_metrics(
        logits, targets, 1.0, args.bins, noncoding_weight
    )
    after = calibration_metrics(
        logits, targets, empirical_temperature, args.bins, noncoding_weight
    )
    balanced = calibration_metrics(
        logits, targets, balanced_temperature, args.bins, noncoding_weight
    )
    argmax_before = logits.argmax(-1)
    argmax_after = (logits / empirical_temperature).argmax(-1)
    payload = {
        "schema": "joint_frame_temperature_scaling_v1",
        "method": "single scalar temperature fitted on held-out validation logits",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "data_root": str(args.data_root.resolve()),
        "split": args.split,
        "class_order": list(CLASS_ORDER),
        "window_size": window,
        "stride": stride,
        "owned_center": [center_left, center_right],
        "seed": args.seed,
        "samples_per_window": args.samples_per_window,
        "sampled_bases": int(targets.numel()),
        "sampled_coding_bases": int((targets != 0).sum()),
        "sampled_noncoding_bases": int((targets == 0).sum()),
        "temperature": empirical_temperature,
        "applied_objective": "empirical_nll",
        "diagnostic_group_balanced_temperature": balanced_temperature,
        "joint_noncoding_weight": noncoding_weight,
        "temperature_bounds": [args.minimum_temperature, args.maximum_temperature],
        "argmax_changes": int((argmax_before != argmax_after).sum()),
        "collection": collection,
        "before": before,
        "after": after,
        "group_balanced_diagnostic": balanced,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output_dir / "temperature.json"
    calibration_path.write_text(json.dumps(payload, indent=2) + "\n")
    metric_names = (
        "accuracy", "nll", "group_balanced_nll", "brier", "ece",
        "mean_max_probability", "median_max_probability", "mean_entropy",
        "fraction_max_ge_0.90", "fraction_max_ge_0.99",
        "fraction_max_ge_0.999",
    )
    with (args.output_dir / "calibration_metrics.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("condition", "temperature", *metric_names),
            delimiter="\t",
        )
        writer.writeheader()
        for condition, metrics in (
            ("before", before),
            ("empirical_nll_calibrated", after),
            ("group_balanced_diagnostic", balanced),
        ):
            writer.writerow({
                "condition": condition,
                "temperature": f"{metrics['temperature']:.9g}",
                **{name: f"{metrics[name]:.9g}" for name in metric_names},
            })
    if args.save_sample:
        np.savez_compressed(
            args.output_dir / "calibration_sample.npz",
            logits=logits.numpy().astype(np.float32),
            targets=targets.numpy().astype(np.uint8),
        )
    plot_calibration(before, after, args.output_dir / "calibration.png")
    print(json.dumps({
        "temperature": empirical_temperature,
        "diagnostic_group_balanced_temperature": balanced_temperature,
        "before": {key: before[key] for key in (
            "nll", "brier", "ece", "mean_max_probability",
            "fraction_max_ge_0.99")},
        "after": {key: after[key] for key in (
            "nll", "brier", "ece", "mean_max_probability",
            "fraction_max_ge_0.99")},
        "argmax_changes": payload["argmax_changes"],
        "output": str(calibration_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
