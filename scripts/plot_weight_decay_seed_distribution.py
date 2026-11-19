#!/usr/bin/env python3
"""Plot per-seed validation AP distributions for the weight-decay sweep."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = [
    ("val_mean_AP", "Mean AP"),
    ("val_donor_AP", "Donor"),
    ("val_acceptor_AP", "Acceptor"),
    ("val_start_AP", "Start"),
    ("val_stop_AP", "Stop"),
]
COLORS = {42: "#0072B2", 43: "#D55E00", 44: "#009E73"}


def read_rows(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["weight_decay"] = float(row["weight_decay"])
        row["seed"] = int(row["seed"])
        for key, _ in TASKS:
            row[key] = float(row[key])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.input)
    weight_decays = sorted({row["weight_decay"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    x = np.arange(len(weight_decays))

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 7.2), sharey=True)
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, TASKS):
        for seed in seeds:
            values = [
                next(row[metric] for row in rows
                     if row["weight_decay"] == wd and row["seed"] == seed)
                for wd in weight_decays
            ]
            ax.plot(x, values, marker="o", markersize=6, linewidth=1.4,
                    color=COLORS.get(seed), alpha=0.9, label=f"Seed {seed}")
        means, sds = [], []
        for wd in weight_decays:
            values = [row[metric] for row in rows if row["weight_decay"] == wd]
            means.append(np.mean(values))
            sds.append(np.std(values, ddof=1))
        ax.errorbar(x, means, yerr=sds, color="black", marker="D",
                    markersize=5, linewidth=1.6, capsize=4, label="Mean ± SD",
                    zorder=5)
        ax.set_title(label, fontsize=12, weight="bold")
        ax.set_xticks(x, [r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])
        ax.set_xlabel("Weight decay")
        ax.set_ylim(0.74, 0.98)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation AP")
    axes[3].set_ylabel("Validation AP")
    axes[5].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].legend(handles, labels, loc="center", frameon=False, fontsize=11)
    fig.tight_layout()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix(".png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    fig.savefig(args.output_prefix.with_suffix(".pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
