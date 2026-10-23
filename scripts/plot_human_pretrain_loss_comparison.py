#!/usr/bin/env python3
"""Plot train/validation loss for random and human-pretrained runs."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "droso_human_pretrain_comparison_v1"
RUNS = [
    (
        "Random initialization",
        ROOT / "runs/joint_frame_grid_v1/b010/chrxtrain/w10/results.json",
    ),
    (
        "Human-pretrained initialization",
        ROOT / "runs/droso_from_human_pretrain_v1/chrxtrain/w10/results.json",
    ),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = []
    source_rows = []
    for model, path in RUNS:
        payload = json.loads(path.read_text())
        frame = pd.DataFrame(payload["history"])
        best_epoch = int(frame.loc[frame["mean_AP"].idxmax(), "epoch"])
        stop_epoch = int(payload["stopped_epoch"] or frame["epoch"].max())
        loaded.append((model, frame, best_epoch, stop_epoch))
        for row in frame.itertuples(index=False):
            source_rows.append({
                "model": model,
                "epoch": row.epoch,
                "train_loss": row.train_loss,
                "validation_loss": row.validation_loss,
                "validation_mean_auprc": row.mean_AP,
                "is_best_epoch": row.epoch == best_epoch,
                "is_early_stop_epoch": row.epoch == stop_epoch,
            })

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
    })
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
    for panel, (axis, (model, frame, best_epoch, stop_epoch)) in enumerate(
        zip(axes, loaded), start=1
    ):
        axis.plot(
            frame["epoch"], frame["train_loss"], color="#4C78A8",
            marker="o", markersize=4, linewidth=2.2, label="Train loss",
        )
        axis.plot(
            frame["epoch"], frame["validation_loss"], color="#F58518",
            marker="s", markersize=4, linewidth=2.2, label="Validation loss",
        )
        axis.axvline(
            best_epoch, color="#54A24B", linestyle="--", linewidth=1.6,
            label=f"Best mean-AUPRC epoch ({best_epoch})",
        )
        axis.axvline(
            stop_epoch, color="#E45756", linestyle=":", linewidth=1.8,
            label=f"Early stop ({stop_epoch})",
        )
        axis.set_title(f"{'AB'[panel - 1]}   {model}", loc="left")
        axis.set_xlabel("Epoch")
        axis.set_yscale("log")
        axis.set_ylim(0.0045, 0.085)
        axis.yaxis.set_major_locator(FixedLocator([0.005, 0.01, 0.02, 0.05]))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
        axis.grid(axis="y", which="both", color="#D9D9D9", alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, fontsize=9.2, loc="upper right")
    axes[0].set_ylabel("Weighted cross-entropy loss (log scale)")
    figure.tight_layout(w_pad=2.6)

    stem = OUT / "random_init_vs_human_pretrained_train_validation_loss"
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(source_rows).to_csv(
        stem.with_suffix(".tsv"), sep="\t", index=False, float_format="%.9g"
    )
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
