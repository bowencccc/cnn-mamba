#!/usr/bin/env python3
"""Plot validation mean AUPRC for three Drosophila initialization strategies."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "droso_pretraining_comparison_v1"
RUNS = {
    "Random initialization": (
        ROOT / "runs/joint_frame_grid_v1/b010/chrxtrain/w10/results.json"
    ),
    "Human MLM pretraining": (
        ROOT / "runs/mlm_transfer_v1/human_grch38_w10/chrxtrain/w10/results.json"
    ),
    "Human supervised pretraining": (
        ROOT / "runs/droso_from_human_pretrain_v1/chrxtrain/w10/results.json"
    ),
}
COLORS = ["#4C78A8", "#E45756", "#59A14F"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_rows = []

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    for (label, path), color in zip(RUNS.items(), COLORS):
        history = pd.DataFrame(json.loads(path.read_text())["history"])
        ax.plot(
            history["epoch"], history["mean_AP"],
            color=color, linewidth=2.4, marker="o", markersize=4.5,
            label=label,
        )
        source_rows.extend(
            {
                "model": label,
                "epoch": int(row.epoch),
                "validation_mean_auprc": float(row.mean_AP),
            }
            for row in history.itertuples(index=False)
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation mean AUPRC")
    ax.set_xlim(1, 26)
    ax.set_ylim(0.78, 0.93)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    stem = OUT / "validation_mean_auprc"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(source_rows).to_csv(
        stem.with_name(stem.name + "_source_data").with_suffix(".tsv"),
        sep="\t", index=False, float_format="%.9g",
    )
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
