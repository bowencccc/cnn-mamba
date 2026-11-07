#!/usr/bin/env python3
"""Create a compact report figure for the Human spliced-start pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/human_spliced_start_pilot_v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, default=BASE / "evaluation/summary.json")
    parser.add_argument("--rerank", type=Path, default=BASE / "uniann_rerank_evaluation/summary.json")
    parser.add_argument("--output", type=Path, default=BASE / "human_spliced_start_pilot_summary.png")
    args = parser.parse_args()
    oracle = json.loads(args.oracle.read_text())
    rerank = json.loads(args.rerank.read_text())
    by_mode = {row["mode"]: row for row in oracle["metrics"]}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    x = np.arange(3)
    width = 0.34
    fields = ("top1_accuracy", "candidate_auprc", "mean_reciprocal_rank")
    labels = ("Top-1", "Candidate\nAUPRC", "MRR")
    for offset, mode, label, color in (
        (-width / 2, "genomic", "With introns", "#A0A0A0"),
        (width / 2, "spliced", "RefSeq-spliced", "#4C78A8"),
    ):
        values = [by_mode[mode][field] for field in fields]
        bars = axes[0].bar(x + offset, values, width, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.18)
    axes[0].set_ylabel("Score")
    axes[0].set_title("A  RefSeq exon-chain oracle (n=50)", loc="left", weight="bold")
    axes[0].legend(frameon=False, loc="upper center", ncol=2)
    axes[0].grid(axis="y", alpha=0.2)

    values = [rerank["current_uniann_top1_accuracy"], rerank["netstart_top1_accuracy"]]
    bars = axes[1].bar(
        ["Current UniAnn\nstart", "NetStart top-1\non UniAnn chain"],
        values, color=["#54A24B", "#E45756"], width=0.58,
    )
    axes[1].bar_label(bars, labels=[f"{100*v:.0f}/100" for v in values], fontsize=10, padding=3)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Exact RefSeq start fraction")
    axes[1].set_title("B  Predicted-chain reranking (conditional n=100)", loc="left", weight="bold")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].text(
        0.5, 0.04,
        f"Fixed {rerank['fixed_by_netstart']}  |  Broken {rerank['broken_by_netstart']}",
        transform=axes[1].transAxes, ha="center", fontsize=10,
    )
    fig.suptitle("Removing introns helps start recognition, but direct replacement is unsafe", weight="bold")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250)
    fig.savefig(args.output.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
