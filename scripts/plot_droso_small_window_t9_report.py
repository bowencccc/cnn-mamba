#!/usr/bin/env python3
"""Plot Drosophila small-window candidate curves and T=9 strict locus results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from cnn_mamba.evaluate_candidates import reference_sites


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/droso_small_window_v1_four_state_t9/report"
TASKS = ("donor", "acceptor", "start", "stop")
TASK_TITLES = {
    "donor": "Donor",
    "acceptor": "Acceptor",
    "start": "Start codon",
    "stop": "Stop codon",
}
WINDOWS = (
    ("500 bp", 500, "w0p5", "#4C78A8"),
    ("1 kb", 1000, "w1", "#E45756"),
    ("2 kb", 2000, "w2", "#59A14F"),
    ("5 kb", 5000, "w5", "#F28E2B"),
    ("10 kb", 10000, "w10", "#9C6ADE"),
)
T9_SERIES = (
    ("500 bp", 500, "w0p5", 42, "#4C78A8", "-"),
    ("1 kb", 1000, "w1", 42, "#E45756", "-"),
    ("2 kb", 2000, "w2", 42, "#59A14F", "-"),
    ("5 kb", 5000, "w5", 42, "#F28E2B", "-"),
    ("5 kb (new seed)", 5000, "w5_seed43", 43, "#8C564B", "--"),
    ("10 kb", 10000, "w10", 42, "#9C6ADE", "-"),
)
MAX_CURVE_POINTS = 8000


def score_dir(key: str) -> Path:
    if key == "w10":
        return ROOT / "artifacts/scores/joint_frame_grid_v1/b010/chrxtrain/w10"
    if key == "w5_seed43":
        return ROOT / "artifacts/scores/droso_small_window_seed43/chrxtrain/w5"
    return ROOT / "artifacts/scores/droso_small_window_v1/chrxtrain" / key


def thin(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= MAX_CURVE_POINTS:
        return x, y
    indices = np.linspace(0, len(x) - 1, MAX_CURVE_POINTS, dtype=np.int64)
    return x[indices], y[indices]


def plot_candidate_curves() -> pd.DataFrame:
    truth, _, _ = reference_sites(
        "drosophila",
        ROOT / "data/raw/dmel_reference.gtf",
        ROOT / "data/raw/dmel_genome.fa",
        "NC_004354.4",
    )
    metric_frames = []
    metrics = {}
    for label, window_bp, key, seed, _color, _linestyle in T9_SERIES:
        frame = pd.read_csv(score_dir(key) / "candidate_auprc.tsv", sep="\t")
        frame.insert(0, "window", label)
        frame.insert(1, "window_bp", window_bp)
        frame.insert(2, "seed", seed)
        metric_frames.append(frame)
        metrics.update({(key, row.task): float(row.AUPRC) for row in frame.itertuples()})
    metric_table = pd.concat(metric_frames, ignore_index=True)
    metric_table.to_csv(
        OUT / "candidate_auprc.tsv", sep="\t", index=False, float_format="%.9f"
    )

    curve_rows = []
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 10.6), sharex=True, sharey=True)
    for axis, task in zip(axes.flat, TASKS):
        expected_positions = None
        labels = None
        for window_label, window_bp, key, seed, color, linestyle in T9_SERIES:
            directory = score_dir(key)
            positions = np.load(directory / f"{task}_positions0.npy", mmap_mode="r")
            scores = np.load(directory / f"{task}_scores.npy", mmap_mode="r")
            if expected_positions is None:
                expected_positions = np.asarray(positions)
                labels = np.isin(positions, truth[task], assume_unique=True)
            elif not np.array_equal(expected_positions, positions):
                raise RuntimeError(f"candidate positions differ for {task}: {window_label}")
            precision, sensitivity, _ = precision_recall_curve(labels, scores)
            draw_sensitivity, draw_precision = thin(sensitivity, precision)
            axis.plot(
                draw_sensitivity,
                draw_precision,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=f"{window_label} (AP={metrics[(key, task)]:.4f})",
            )
            curve_rows.extend(
                {
                    "task": task,
                    "window": window_label,
                    "window_bp": window_bp,
                    "seed": seed,
                    "sensitivity": float(x),
                    "precision": float(y),
                    "AUPRC": metrics[(key, task)],
                }
                for x, y in zip(draw_sensitivity, draw_precision)
            )
        axis.set_title(TASK_TITLES[task], fontsize=14, weight="bold")
        axis.set_xlabel("Sensitivity (recall)")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.set_box_aspect(1)
        axis.grid(color="#D9D9D9", linewidth=0.7, alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.94)
    figure.tight_layout(h_pad=1.8, w_pad=1.4)
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUT / f"drosophila_window_test_four_task_prsn_curves.{suffix}",
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)
    pd.DataFrame(curve_rows).to_csv(
        OUT / "four_task_prsn_curve_source_data.tsv.gz",
        sep="\t",
        index=False,
        float_format="%.9g",
        compression="gzip",
    )
    return metric_table


def load_locus_metrics() -> pd.DataFrame:
    small = pd.read_csv(
        ROOT / "results/droso_small_window_v1_four_state_t9/locus_summary.tsv",
        sep="\t",
    )
    old = pd.read_csv(
        ROOT / "results/frame_scaling_extended_grid_v1/strict_locus_metrics.tsv",
        sep="\t",
    )
    old = old[
        (old["condition"] == "joint_s9p000_b0p000")
        & old["method"].isin(("UniAnn", "EviAnn+UniAnn"))
    ].copy()
    old.insert(0, "window_bp", 10000)
    columns = (
        "window_bp", "method", "sn", "pr", "f1", "predicted_loci",
        "matched_loci", "missed_loci", "novel_loci",
    )
    small = small[list(columns)].copy()
    old = old[list(columns)].copy()
    small["seed"] = 42
    old["seed"] = 42
    seed43 = pd.read_csv(
        ROOT
        / "work/uniann/droso_small_window_seed43_four_state_t9/w5/locus_metrics.tsv",
        sep="\t",
    )
    seed43.insert(0, "window_bp", 5000)
    seed43["seed"] = 43
    table = pd.concat((small, seed43, old), ignore_index=True)
    table["window"] = table.window_bp.map(
        {500: "500 bp", 1000: "1 kb", 2000: "2 kb", 5000: "5 kb", 10000: "10 kb"}
    )
    table.loc[table.seed == 43, "window"] = "5 kb (new seed)"
    table.to_csv(
        OUT / "joint_softmax_t9_strict_locus_metrics.tsv",
        sep="\t",
        index=False,
        float_format="%.9f",
    )
    return table


def load_raw_locus_metrics() -> pd.DataFrame:
    frames = []
    for window_label, window_bp, key, _color in WINDOWS:
        frame = pd.read_csv(
            ROOT
            / "work/uniann/droso_small_window_v1_four_state_raw_t1"
            / key
            / "locus_metrics.tsv",
            sep="\t",
        )
        frame.insert(0, "window_bp", window_bp)
        frame.insert(1, "window", window_label)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(
        OUT / "raw_t1_strict_locus_metrics.tsv",
        sep="\t",
        index=False,
        float_format="%.9f",
    )
    return table


def load_vector_locus_metrics() -> pd.DataFrame:
    frames = []
    for window_label, window_bp, key, _color in WINDOWS:
        if key == "w10":
            source = (
                ROOT
                / "work/uniann/frame_scaling_extended_grid_v1/vector_scaling_block_holdout/provenance.json"
            )
            metrics = json.loads(source.read_text())["metrics"]
            frame = pd.DataFrame(
                [{"method": method, **values} for method, values in metrics.items()]
            )
        else:
            source = (
                ROOT
                / "work/uniann/droso_small_window_v1_four_state_vector"
                / key
                / "locus_metrics.tsv"
            )
            frame = pd.read_csv(source, sep="\t")
        frame.insert(0, "window_bp", window_bp)
        frame.insert(1, "window", window_label)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(
        OUT / "vector_scaling_strict_locus_metrics.tsv",
        sep="\t",
        index=False,
        float_format="%.9f",
    )
    return table


def load_vector_builtin_locus_metrics() -> pd.DataFrame:
    frames = []
    for window_label, window_bp, key, _color in WINDOWS:
        if key == "w10":
            source = (
                ROOT
                / "work/uniann/vector_builtin_t3_comparison_v1/w10/locus_metrics.tsv"
            )
        else:
            source = (
                ROOT
                / "work/uniann/droso_small_window_v1_four_state_vector_builtin_t3"
                / key
                / "locus_metrics.tsv"
            )
        frame = pd.read_csv(source, sep="\t")
        frame.insert(0, "window_bp", window_bp)
        frame.insert(1, "window", window_label)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(
        OUT / "vector_scaling_plus_uniann_builtin_strict_locus_metrics.tsv",
        sep="\t",
        index=False,
        float_format="%.9f",
    )
    return table


def plot_locus_metrics(
    table: pd.DataFrame,
    output_stem: str,
    ylim: tuple[float, float],
    window_labels: list[str] | None = None,
) -> None:
    if window_labels is None:
        window_labels = [item[0] for item in WINDOWS]
    display_labels = [label.replace(" (new seed)", "\n(new seed)") for label in window_labels]
    methods = ("UniAnn", "EviAnn+UniAnn")
    method_labels = {"UniAnn": "UniAnn", "EviAnn+UniAnn": "EviAnn + UniAnn"}
    colors = {"UniAnn": "#4C78A8", "EviAnn+UniAnn": "#E45756"}
    x = np.arange(len(window_labels), dtype=float)
    width = 0.36

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(18.4, 5.8), sharey=True)
    for axis, (metric, title) in zip(
        axes,
        (("sn", "Locus sensitivity"), ("pr", "Locus precision"), ("f1", "Locus F1")),
    ):
        for offset, method in ((-width / 2, methods[0]), (width / 2, methods[1])):
            subset = table[table.method == method].set_index("window")
            values = [float(subset.loc[label, metric]) for label in window_labels]
            bars = axis.bar(
                x + offset,
                values,
                width,
                color=colors[method],
                label=method_labels[method],
            )
            axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8.5)
        axis.set_title(title, fontsize=14, weight="bold")
        axis.set_xticks(x, display_labels, fontsize=8.5)
        axis.set_xlabel("Window size")
        axis.set_ylim(*ylim)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.grid(axis="x", visible=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.94)
    axes[0].set_ylabel("Percent (%)")
    figure.tight_layout(w_pad=1.2)
    for suffix in ("png", "pdf"):
        figure.savefig(
            OUT / f"{output_stem}.{suffix}",
            dpi=320 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_candidate_curves()
    plot_locus_metrics(
        load_locus_metrics(),
        "drosophila_window_test_t9_uniann_locus_metrics",
        (65, 95),
        [item[0] for item in T9_SERIES],
    )
    plot_locus_metrics(
        load_raw_locus_metrics(),
        "drosophila_window_test_before_scaling_uniann_locus_metrics",
        (0, 100),
    )
    plot_locus_metrics(
        load_vector_locus_metrics(),
        "drosophila_window_test_vector_scaling_uniann_locus_metrics",
        (0, 100),
    )
    plot_locus_metrics(
        load_vector_builtin_locus_metrics(),
        "drosophila_window_test_vector_scaling_plus_uniann_builtin_locus_metrics",
        (30, 95),
    )
    print(OUT)


if __name__ == "__main__":
    main()
