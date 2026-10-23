#!/usr/bin/env python3
"""Plot strict locus-level results from the extended UniAnn scaling grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/frame_scaling_extended_grid_v1"
INPUT = RESULT / "strict_locus_metrics.tsv"
VECTOR_INPUT = RESULT / "vector_scaling_strict_locus_metrics.tsv"
PSAURON_F1 = 90.3420033


def load_combined():
    with INPUT.open() as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t")
                if r["method"] == "EviAnn+UniAnn"]
    if VECTOR_INPUT.is_file():
        with VECTOR_INPUT.open() as handle:
            rows.extend(
                r for r in csv.DictReader(handle, delimiter="\t")
                if r["method"] == "EviAnn+UniAnn"
            )
    for row in rows:
        for key in ("scale", "coding_bias", "sn", "pr", "f1"):
            row[key] = float(row[key])
    return rows


def matrix(rows, family, scales, biases):
    import numpy as np

    lookup = {(r["scale"], r["coding_bias"]): r["f1"] for r in rows
              if r["family"] == family}
    return np.array([[lookup.get((scale, bias), np.nan) for bias in biases]
                     for scale in scales])


def annotate_heatmap(axis, values):
    import numpy as np

    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            if np.isfinite(values[y, x]):
                axis.text(x, y, f"{values[y, x]:.2f}", ha="center",
                          va="center", fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table-only", action="store_true",
        help="Update the ranked TSV without requiring matplotlib",
    )
    args = parser.parse_args()
    rows = load_combined()
    ranked = sorted(rows, key=lambda r: (-r["f1"], -r["sn"], -r["pr"]))
    with (RESULT / "eviann_uniann_ranked.tsv").open("w", newline="") as handle:
        fields = ["rank", "condition", "family", "scale", "coding_bias",
                  "sn", "pr", "f1", "description"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            writer.writerow({"rank": rank, **{key: row[key] for key in fields[1:]}})
    if args.table_only:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    joint_no_gate = sorted(
        (r for r in rows if r["family"] == "joint_softmax_no_gate"),
        key=lambda r: r["scale"],
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)

    ax = axes[0]
    temperatures = [r["scale"] for r in joint_no_gate]
    ax.plot(temperatures, [r["sn"] for r in joint_no_gate], "o-", label="Sn")
    ax.plot(temperatures, [r["pr"] for r in joint_no_gate], "o-", label="Pr")
    ax.plot(temperatures, [r["f1"] for r in joint_no_gate], "o-", label="F1")
    ax.axhline(PSAURON_F1, color="0.35", linestyle="--", linewidth=1,
               label="PSAURON F1")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Strict locus metric (%)")
    ax.set_title("Joint softmax, no extra gate")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)

    joint_scales = [7.0, 8.0, 9.0]
    joint_biases = [0.01, 0.02, 0.03]
    joint = matrix(rows, "joint_softmax_coding_bias", joint_scales, joint_biases)
    axes[1].imshow(joint, cmap="viridis", vmin=np.nanmin(joint), vmax=np.nanmax(joint))
    annotate_heatmap(axes[1], joint)
    axes[1].set_xticks(range(len(joint_biases)), [f"{x:.2f}" for x in joint_biases])
    axes[1].set_yticks(range(len(joint_scales)), [f"{x:g}" for x in joint_scales])
    axes[1].set_xlabel("Coding bias")
    axes[1].set_ylabel("Temperature")
    axes[1].set_title("Joint softmax F1 (%)")

    emission_scales = [0.10, 0.125, 0.15, 0.175, 0.20]
    emission_biases = [0.0, 0.01, 0.02]
    emission = matrix(rows, "direct_emission_affine", emission_scales, emission_biases)
    axes[2].imshow(
        emission, cmap="viridis", vmin=np.nanmin(emission), vmax=np.nanmax(emission),
        aspect="auto",
    )
    annotate_heatmap(axes[2], emission)
    axes[2].set_xticks(range(len(emission_biases)), [f"{x:.2f}" for x in emission_biases])
    axes[2].set_yticks(range(len(emission_scales)), [f"{x:g}" for x in emission_scales])
    axes[2].set_xlabel("Coding bias")
    axes[2].set_ylabel("Emission scale")
    axes[2].set_title("Direct emission F1 (%)")

    RESULT.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULT / "eviann_uniann_scaling_grid.png", dpi=220)
    fig.savefig(RESULT / "eviann_uniann_scaling_grid.pdf")
    plt.close(fig)

if __name__ == "__main__":
    main()
