#!/usr/bin/env python3
"""Summarize exact chrX-plus AUPRC from the controlled frame ablation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "frame_ablation_v1"
SOURCES = {
    "Baseline": (
        ROOT / "artifacts" / "scores" / "early_stopping_v1"
        / "heldout" / "w10" / "candidate_auprc.tsv"
    ),
    "Frame dilation 3": (
        ROOT / "artifacts" / "scores" / "frame_ablation_v1"
        / "frame_d3" / "heldout" / "w10" / "candidate_auprc.tsv"
    ),
    "Phase auxiliary": (
        ROOT / "artifacts" / "scores" / "frame_ablation_v1"
        / "phase_aux" / "heldout" / "w10" / "candidate_auprc.tsv"
    ),
    "Combined": (
        ROOT / "artifacts" / "scores" / "frame_ablation_v1"
        / "combined" / "heldout" / "w10" / "candidate_auprc.tsv"
    ),
}
TASKS = ("donor", "acceptor", "start", "stop")


def main():
    rows = []
    for model, path in SOURCES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, sep="\t").set_index("task")
        values = {task: float(frame.loc[task, "AUPRC"]) for task in TASKS}
        rows.append({"model": model, **values, "mean": np.mean(list(values.values()))})

    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "exact_chrx_plus_auprc.tsv", sep="\t", index=False)

    figure, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    colors = ("#777777", "#4C78A8", "#F58518", "#54A24B")
    for axis, task in zip(axes, TASKS):
        axis.bar(np.arange(len(summary)), summary[task], color=colors, width=0.72)
        axis.set_title(task.capitalize())
        axis.set_xticks(np.arange(len(summary)))
        axis.set_xticklabels(("Base", "D3", "Phase", "Both"), rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("AUPRC")
    figure.tight_layout()
    figure.savefig(OUTPUT / "exact_chrx_plus_auprc.png", dpi=220)
    figure.savefig(OUTPUT / "exact_chrx_plus_auprc.pdf")
    plt.close(figure)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
