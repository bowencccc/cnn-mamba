#!/usr/bin/env python3
"""Plot Drosophila fine-tuning convergence with and without Human MLM pretraining."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mlm_transfer_v1" / "human_grch38_w10" / "comparison"
RUNS = {
    "Random initialization": (
        ROOT / "runs" / "joint_frame_grid_v1" / "b010" / "chrxtrain" / "w10"
        / "results.json"
    ),
    "Human MLM pretraining": (
        ROOT / "runs" / "mlm_transfer_v1" / "human_grch38_w10" / "chrxtrain"
        / "w10" / "results.json"
    ),
}
COLORS = {
    "Random initialization": "#4C78A8",
    "Human MLM pretraining": "#E45756",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    histories = {}
    summary_rows = []
    curve_rows = []

    for label, path in RUNS.items():
        payload = json.loads(path.read_text())
        history = pd.DataFrame(payload["history"])
        histories[label] = history
        best = history.loc[history["mean_AP"].idxmax()]
        crossing = history.loc[history["mean_AP"] >= 0.90].iloc[0]
        stopped = int(payload.get("stopped_epoch") or history["epoch"].max())
        summary_rows.append({
            "model": label,
            "first_epoch_validation_mean_auprc_ge_0.90": int(crossing["epoch"]),
            "best_epoch": int(best["epoch"]),
            "best_validation_mean_auprc": float(best["mean_AP"]),
            "early_stop_epoch": stopped,
        })
        for row in history.itertuples(index=False):
            curve_rows.append({
                "model": label,
                "epoch": int(row.epoch),
                "validation_mean_auprc": float(row.mean_AP),
            })

    summary = pd.DataFrame(summary_rows).set_index("model")
    random_crossing = int(summary.loc[
        "Random initialization", "first_epoch_validation_mean_auprc_ge_0.90"
    ])
    mlm_crossing = int(summary.loc[
        "Human MLM pretraining", "first_epoch_validation_mean_auprc_ge_0.90"
    ])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "legend.fontsize": 11,
    })
    fig, ax = plt.subplots(figsize=(10.0, 6.4))

    for label, history in histories.items():
        color = COLORS[label]
        ax.plot(
            history["epoch"], history["mean_AP"],
            color=color, linewidth=2.8, marker="o", markersize=5.0,
            markeredgecolor="white", markeredgewidth=0.65, label=label,
        )

        crossing_epoch = int(summary.loc[
            label, "first_epoch_validation_mean_auprc_ge_0.90"
        ])
        crossing_value = float(
            history.loc[history["epoch"] == crossing_epoch, "mean_AP"].iloc[0]
        )
        ax.scatter(
            crossing_epoch, crossing_value, marker="D", s=105, color=color,
            edgecolor="white", linewidth=1.2, zorder=5,
        )

        best_epoch = int(summary.loc[label, "best_epoch"])
        best_value = float(summary.loc[label, "best_validation_mean_auprc"])
        ax.scatter(
            best_epoch, best_value, marker="*", s=230, color=color,
            edgecolor="white", linewidth=1.0, zorder=5,
        )
        if label == "Random initialization":
            best_offset, best_align = (-7, 14), "center"
        else:
            best_offset, best_align = (38, -31), "left"
        ax.annotate(
            f"Best: epoch {best_epoch}\nAUPRC {best_value:.3f}",
            (best_epoch, best_value), xytext=best_offset,
            textcoords="offset points", ha=best_align, fontsize=10, color=color,
        )

    ax.axhline(0.90, color="#666666", linestyle="--", linewidth=1.25)
    ax.text(
        25.7, 0.902, "0.90 threshold", ha="right", va="bottom",
        color="#555555", fontsize=10,
    )

    # Show the central convergence result directly on the plot.
    arrow_y = 0.884
    ax.annotate(
        "", xy=(random_crossing, arrow_y), xytext=(mlm_crossing, arrow_y),
        arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 1.5},
    )
    ax.text(
        (random_crossing + mlm_crossing) / 2, arrow_y - 0.006,
        f"{random_crossing - mlm_crossing} epochs earlier",
        ha="center", va="top", fontsize=11, fontweight="bold", color="#333333",
    )
    ax.annotate(
        f"First ≥0.90: epoch {mlm_crossing}",
        (mlm_crossing, float(histories["Human MLM pretraining"].loc[
            histories["Human MLM pretraining"]["epoch"] == mlm_crossing,
            "mean_AP",
        ].iloc[0])),
        xytext=(-18, 34), textcoords="offset points", ha="left",
        color=COLORS["Human MLM pretraining"], fontsize=10,
        arrowprops={"arrowstyle": "-", "color": COLORS["Human MLM pretraining"]},
    )
    ax.annotate(
        f"First ≥0.90: epoch {random_crossing}",
        (random_crossing, float(histories["Random initialization"].loc[
            histories["Random initialization"]["epoch"] == random_crossing,
            "mean_AP",
        ].iloc[0])),
        xytext=(10, -35), textcoords="offset points", color=COLORS["Random initialization"],
        fontsize=10, arrowprops={"arrowstyle": "-", "color": COLORS["Random initialization"]},
    )

    ax.set_title("Human MLM Pretraining Accelerates Drosophila Fine-tuning", loc="left")
    ax.set_xlabel("Fine-tuning epoch")
    ax.set_ylabel("Validation mean AUPRC")
    ax.set_xlim(0.5, 26.5)
    ax.set_ylim(0.78, 0.93)
    ax.set_xticks([1, 3, 5, 8, 10, 12, 15, 20, 23, 26])
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    stem = OUT / "human_mlm_vs_random_convergence"
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    pd.DataFrame(curve_rows).to_csv(
        stem.with_name(stem.name + "_source_data").with_suffix(".tsv"),
        sep="\t", index=False, float_format="%.9g",
    )
    summary.reset_index().to_csv(
        stem.with_name(stem.name + "_summary").with_suffix(".tsv"),
        sep="\t", index=False, float_format="%.9g",
    )
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(summary.to_string())


if __name__ == "__main__":
    main()
