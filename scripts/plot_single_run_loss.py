#!/usr/bin/env python3
"""Plot recorded train and validation loss for one training run."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="train_validation_loss")
    args = parser.parse_args()

    payload = json.loads(args.results_json.read_text())
    frame = pd.DataFrame(payload["history"])
    best = int(frame.loc[frame["mean_AP"].idxmax(), "epoch"])
    stopped = int(payload["stopped_epoch"] or frame["epoch"].max())

    columns = [
        "epoch", "train_loss", "validation_loss",
        "train_splice_loss", "validation_splice_loss",
        "train_start_stop_loss", "validation_start_stop_loss",
        "train_phase_loss", "validation_phase_loss", "mean_AP",
    ]
    for column in ("train_cds_loss", "validation_cds_loss"):
        if column in frame:
            columns.insert(-1, column)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame[columns].to_csv(
        args.output_dir / f"{args.prefix}.tsv", sep="\t", index=False,
        float_format="%.9f",
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    axis.plot(
        frame["epoch"], frame["train_loss"], marker="o", linewidth=2,
        markersize=4.5, color="#4C78A8", label="Train loss",
    )
    axis.plot(
        frame["epoch"], frame["validation_loss"], marker="s", linewidth=2,
        markersize=4.2, color="#F58518", label="Validation loss",
    )
    axis.axvline(best, color="#54A24B", linestyle="--", linewidth=1.6,
                 label=f"Best epoch ({best})")
    axis.axvline(stopped, color="#E45756", linestyle=":", linewidth=1.8,
                 label=f"Early stop ({stopped})")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted cross-entropy loss")
    axis.set_xticks(frame["epoch"])
    axis.legend(frameon=True)
    axis.grid(alpha=0.24)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"{args.prefix}.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)
    print(args.output_dir / f"{args.prefix}.png")


if __name__ == "__main__":
    main()
