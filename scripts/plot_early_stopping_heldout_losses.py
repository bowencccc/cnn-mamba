#!/usr/bin/env python3
"""Plot per-epoch train and validation loss for an early-stopping regime."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (5, 10, 20, 30, 40, 80)
COLORS = {
    5: "#4C78A8",
    10: "#E45756",
    20: "#59A14F",
    30: "#F28E2B",
    40: "#9C6ADE",
    80: "#76B7B2",
}


def load_histories(runs):
    rows = []
    for window in WINDOWS:
        payload = json.loads((runs / f"w{window}" / "results.json").read_text())
        for record in payload["history"]:
            rows.append({
                "window_kb": window,
                "epoch": int(record["epoch"]),
                "train_loss": float(record["train_loss"]),
                "validation_loss": float(record["validation_loss"]),
                "train_splice_loss": float(record["train_splice_loss"]),
                "train_start_stop_loss": float(record["train_start_stop_loss"]),
                "validation_splice_loss": float(record["validation_splice_loss"]),
                "validation_start_stop_loss": float(record["validation_start_stop_loss"]),
                "validation_mean_AP": float(record["mean_AP"]),
                "early_stopped": bool(payload["early_stopped"]),
                "stopped_epoch": int(payload["stopped_epoch"]),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=("heldout", "chrxtrain"), default="heldout")
    parser.add_argument("--layout", choices=("split", "overlay"), default="split")
    args = parser.parse_args()
    runs = ROOT / "runs" / "early_stopping_v1" / args.regime
    output_name = "heldout_losses" if args.regime == "heldout" else "chrxtrain_losses"
    output_root = ROOT / "results" / "early_stopping_v1" / "plots" / output_name
    prefix = "heldout" if args.regime == "heldout" else "chrxtrain"

    output_root.mkdir(parents=True, exist_ok=True)
    frame = load_histories(runs)
    frame.to_csv(output_root / f"{prefix}_train_validation_loss.tsv", sep="\t", index=False,
                 float_format="%.9f")

    plt.style.use("seaborn-v0_8-whitegrid")
    if args.layout == "overlay":
        fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0))
        for ax, window in zip(axes.flat, WINDOWS):
            subset = frame[frame.window_kb == window].sort_values("epoch")
            ax.plot(
                subset.epoch,
                subset.train_loss,
                marker="o",
                markersize=4.5,
                linewidth=1.8,
                color=COLORS[window],
                label="Train",
            )
            ax.plot(
                subset.epoch,
                subset.validation_loss,
                marker="s",
                markersize=4.2,
                linewidth=1.8,
                linestyle="--",
                color="#333333",
                label="Validation",
            )
            ax.set_title(
                f"{window} kb (stop={int(subset.stopped_epoch.iloc[0])})",
                fontsize=12,
                weight="bold",
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weighted candidate cross-entropy")
            ax.set_xticks(subset.epoch)
            ax.grid(alpha=0.23, linewidth=0.8)
            ax.legend(fontsize=8.5, frameon=True, framealpha=0.94)
        fig.tight_layout(h_pad=1.8, w_pad=1.4)
        stem = f"{prefix}_train_validation_loss_overlay"
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.3))
        for ax, column, title in zip(
            axes,
            ("train_loss", "validation_loss"),
            ("Training loss", "Validation loss"),
        ):
            for window in WINDOWS:
                subset = frame[frame.window_kb == window].sort_values("epoch")
                ax.plot(
                    subset.epoch,
                    subset[column],
                    marker="o",
                    markersize=4.5,
                    linewidth=1.8,
                    color=COLORS[window],
                    label=f"{window} kb (stop={int(subset.stopped_epoch.iloc[0])})",
                )
            ax.set_title(title, fontsize=14, weight="bold")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Weighted candidate cross-entropy")
            ax.set_xticks(range(1, int(frame.epoch.max()) + 1))
            ax.grid(alpha=0.23, linewidth=0.8)
            ax.legend(fontsize=8.5, frameon=True, framealpha=0.94)
        fig.tight_layout(w_pad=2.0)
        stem = f"{prefix}_train_validation_loss"

    png = output_root / f"{stem}.png"
    pdf = output_root / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)


if __name__ == "__main__":
    main()
