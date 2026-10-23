#!/usr/bin/env python3
"""Compare random initialization with human-pretrained Drosophila training."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "droso_human_pretrain_comparison_v1"

RUNS = {
    "Random initialization": (
        ROOT / "runs/joint_frame_grid_v1/b010/chrxtrain/w10/results.json"
    ),
    "Human-pretrained initialization": (
        ROOT / "runs/droso_from_human_pretrain_v1/chrxtrain/w10/results.json"
    ),
}
SCORES = {
    "Random initialization": (
        ROOT / "artifacts/scores/joint_frame_grid_v1/b010/chrxtrain/w10/"
        "candidate_auprc.tsv"
    ),
    "Human-pretrained initialization": (
        ROOT / "artifacts/scores/droso_from_human_pretrain_v1/chrxtrain/w10/"
        "candidate_auprc.tsv"
    ),
}
LOCUS = {
    "Random initialization": (
        ROOT / "work/uniann/joint_frame_grid_v1/b010/w10/locus_metrics.tsv"
    ),
    "Human-pretrained initialization": (
        ROOT / "work/uniann/droso_from_human_pretrain_v1/w10/locus_metrics.tsv"
    ),
}

COLORS = {
    "Random initialization": "#4C78A8",
    "Human-pretrained initialization": "#E45756",
}
SHORT = {
    "Random initialization": "Random init",
    "Human-pretrained initialization": "Human pretrained",
}


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)


def annotate_bars(axis, containers, digits=3):
    for container in containers:
        axis.bar_label(
            container, fmt=f"%.{digits}f", padding=3, fontsize=8.5,
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    histories = {}
    result_payloads = {}
    candidates = {}
    loci = {}
    for name, path in RUNS.items():
        result_payloads[name] = json.loads(path.read_text())
        histories[name] = pd.DataFrame(result_payloads[name]["history"])
        candidates[name] = pd.read_csv(SCORES[name], sep="\t").set_index("task")
        loci[name] = pd.read_csv(LOCUS[name], sep="\t").set_index("method")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10.5,
        "legend.fontsize": 9.5,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.0))
    ax_curve, ax_candidate, ax_uniann, ax_combined = axes.flat

    # A: convergence measured by the same reference validation objective used
    # for model selection.
    for name, history in histories.items():
        color = COLORS[name]
        ax_curve.plot(
            history["epoch"], history["mean_AP"], marker="o", markersize=3.8,
            linewidth=2.2, color=color, label=SHORT[name],
        )
        best_index = history["mean_AP"].idxmax()
        best = history.loc[best_index]
        ax_curve.scatter(
            [best["epoch"]], [best["mean_AP"]], s=85, facecolors="white",
            edgecolors=color, linewidths=2.2, zorder=4,
        )
        crossing = history.loc[history["mean_AP"] >= 0.90]
        if not crossing.empty:
            first = crossing.iloc[0]
            ax_curve.scatter(
                [first["epoch"]], [first["mean_AP"]], marker="D", s=42,
                color=color, edgecolor="white", linewidth=0.7, zorder=5,
            )
            ax_curve.annotate(
                f"epoch {int(first['epoch'])}",
                (first["epoch"], first["mean_AP"]),
                xytext=(4, -15), textcoords="offset points", fontsize=8.5,
                color=color,
            )
    ax_curve.axhline(0.90, color="#777777", linestyle="--", linewidth=1.0)
    ax_curve.set_title("A   Convergence on reference validation data", loc="left")
    ax_curve.set_xlabel("Epoch")
    ax_curve.set_ylabel("Validation mean AUPRC")
    ax_curve.set_xlim(0.5, max(len(x) for x in histories.values()) + 0.5)
    ax_curve.set_ylim(0.78, 0.925)
    ax_curve.legend(frameon=False, loc="lower right")
    style_axis(ax_curve)

    # B: exact full-chromosome positive-strand candidate AUPRC.
    tasks = ["donor", "acceptor", "start", "stop"]
    labels = ["Donor", "Acceptor", "Start", "Stop"]
    x = np.arange(len(tasks))
    width = 0.36
    containers = []
    for offset, name in zip((-width / 2, width / 2), RUNS):
        values = candidates[name].loc[tasks, "AUPRC"].to_numpy()
        containers.append(ax_candidate.bar(
            x + offset, values, width, color=COLORS[name], label=SHORT[name],
        ))
    annotate_bars(ax_candidate, containers, digits=3)
    ax_candidate.set_title("B   Exact chrX+ candidate AUPRC", loc="left")
    ax_candidate.set_ylabel("AUPRC")
    ax_candidate.set_xticks(x, labels)
    ax_candidate.set_ylim(0.84, 0.99)
    style_axis(ax_candidate)

    # C/D: downstream strict-match locus metrics.
    metric_names = ["sn", "pr", "f1"]
    metric_labels = ["Sensitivity", "Precision", "F1"]
    for axis, method, panel, ylim in (
        (ax_uniann, "UniAnn", "C", (74, 82)),
        (ax_combined, "EviAnn+UniAnn", "D", (87, 93)),
    ):
        containers = []
        for offset, name in zip((-width / 2, width / 2), RUNS):
            values = loci[name].loc[method, metric_names].to_numpy(dtype=float)
            containers.append(axis.bar(
                np.arange(3) + offset, values, width,
                color=COLORS[name], label=SHORT[name],
            ))
        annotate_bars(axis, containers, digits=1)
        axis.set_title(f"{panel}   {method}: strict locus-level metrics", loc="left")
        axis.set_ylabel("Percent")
        axis.set_xticks(np.arange(3), metric_labels)
        axis.set_ylim(*ylim)
        style_axis(axis)

    fig.tight_layout(h_pad=2.6, w_pad=2.0)
    png = OUT / "random_init_vs_human_pretrained_droso_comparison.png"
    pdf = OUT / "random_init_vs_human_pretrained_droso_comparison.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    rows = []
    for name, history in histories.items():
        best = history.loc[history["mean_AP"].idxmax()]
        crossing = history.loc[history["mean_AP"] >= 0.90]
        rows.extend([
            {"section": "convergence", "model": SHORT[name],
             "metric": "best_epoch", "value": int(best["epoch"])},
            {"section": "convergence", "model": SHORT[name],
             "metric": "best_validation_mean_auprc", "value": best["mean_AP"]},
            {"section": "convergence", "model": SHORT[name],
             "metric": "first_epoch_validation_mean_auprc_ge_0.90",
             "value": None if crossing.empty else int(crossing.iloc[0]["epoch"])},
            {"section": "convergence", "model": SHORT[name],
             "metric": "early_stop_epoch",
             "value": result_payloads[name]["stopped_epoch"]},
        ])
        for task in tasks:
            rows.append({
                "section": "chrX_candidate_auprc", "model": SHORT[name],
                "metric": task, "value": candidates[name].loc[task, "AUPRC"],
            })
        for method in ("UniAnn", "EviAnn+UniAnn"):
            for metric in metric_names:
                rows.append({
                    "section": f"{method}_strict_locus", "model": SHORT[name],
                    "metric": metric,
                    "value": loci[name].loc[method, metric],
                })
    pd.DataFrame(rows).to_csv(
        OUT / "random_init_vs_human_pretrained_droso_source_data.tsv",
        sep="\t", index=False, float_format="%.9g",
    )
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
