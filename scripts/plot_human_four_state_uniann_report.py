#!/usr/bin/env python3
"""Plot Human chr1 candidate PR--SN curves and strict locus results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


TASKS = ("donor", "acceptor", "start", "stop")
COLORS = {
    "donor": "#2166ac",
    "acceptor": "#1b9e77",
    "start": "#d95f02",
    "stop": "#7570b3",
}


def parse_attrs(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for key, quoted, plain in re.findall(
        r'(\S+)\s+(?:"([^"]*)"|([^;\s]+))\s*;?', text
    ):
        attrs[key] = quoted or plain
    for field in text.split(";"):
        field = field.strip()
        if "=" in field:
            key, value = field.split("=", 1)
            attrs.setdefault(key.strip(), value.strip())
    return attrs


def read_cds_models(path: Path, seqid: str) -> dict[str, list[tuple[int, int]]]:
    cds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    seen_target = False
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9:
                continue
            if fields[0] != seqid:
                if seen_target:
                    break
                continue
            seen_target = True
            if fields[6] != "+" or fields[2] != "CDS":
                continue
            attrs = parse_attrs(fields[8])
            tid = attrs.get("Parent") or attrs.get("transcript_id")
            if tid:
                cds[tid].append((int(fields[3]), int(fields[4])))
    return {tid: sorted(parts) for tid, parts in cds.items() if parts}


def model_sites(models: dict[str, list[tuple[int, int]]]) -> dict[str, np.ndarray]:
    sites: dict[str, set[int]] = {task: set() for task in TASKS}
    for parts in models.values():
        sites["start"].add(parts[0][0] - 1)
        sites["stop"].add(parts[-1][1] - 3)
        for (_, left_end), (right_start, _) in zip(parts, parts[1:]):
            if right_start > left_end + 1:
                sites["donor"].add(left_end)
                sites["acceptor"].add(right_start - 3)
    return {
        task: np.asarray(sorted(positions), dtype=np.uint32)
        for task, positions in sites.items()
    }


def canonical_subset(raw_sites: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(candidates, raw_sites)
    valid = indices < len(candidates)
    matched = np.zeros(len(raw_sites), dtype=bool)
    matched[valid] = candidates[indices[valid]] == raw_sites[valid]
    return raw_sites[matched]


def labels_for_candidates(candidates: np.ndarray, truth: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(candidates), dtype=np.uint8)
    indices = np.searchsorted(candidates, truth)
    valid = (indices < len(candidates)) & (candidates[indices] == truth)
    labels[indices[valid]] = 1
    return labels


def thin_curve(recall: np.ndarray, precision: np.ndarray, max_points: int = 5000):
    if len(recall) <= max_points:
        return recall, precision
    chosen = np.unique(np.linspace(0, len(recall) - 1, max_points).astype(np.int64))
    return recall[chosen], precision[chosen]


def candidate_metrics(args, output_dir: Path):
    reference_raw = model_sites(read_cds_models(args.reference_gff, args.seqid))
    eviann_raw = model_sites(read_cds_models(args.eviann_gff, args.seqid))
    metric_rows = []
    point_rows = []
    curve_rows = []
    curve_data = {}

    for task in TASKS:
        candidates = np.load(args.score_dir / f"{task}_positions0.npy", mmap_mode="r")
        scores = np.load(args.score_dir / f"{task}_scores.npy", mmap_mode="r")
        truth = canonical_subset(reference_raw[task], candidates)
        eviann = canonical_subset(eviann_raw[task], candidates)
        labels = labels_for_candidates(candidates, truth)
        eviann_labels = labels_for_candidates(candidates, eviann)
        ap = float(average_precision_score(labels, scores))
        eviann_label_ap = float(average_precision_score(eviann_labels, scores))
        precision, recall, thresholds = precision_recall_curve(
            labels, scores, drop_intermediate=True
        )
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
        best = int(np.nanargmax(f1))
        threshold = float(thresholds[best]) if best < len(thresholds) else 1.0
        overlap = int(np.intersect1d(truth, eviann, assume_unique=True).size)
        eviann_pr = overlap / len(eviann) if len(eviann) else 0.0
        eviann_sn = overlap / len(truth) if len(truth) else 0.0
        eviann_f1 = (
            2 * eviann_pr * eviann_sn / (eviann_pr + eviann_sn)
            if eviann_pr + eviann_sn else 0.0
        )
        metric_rows.append({
            "task": task,
            "candidates": len(candidates),
            "reference_raw_sites": len(reference_raw[task]),
            "reference_canonical_sites": len(truth),
            "AUPRC": ap,
            "EviAnn_label_AUPRC": eviann_label_ap,
            "best_threshold": threshold,
            "best_precision": float(precision[best]),
            "best_sensitivity": float(recall[best]),
            "best_F1": float(f1[best]),
        })
        point_rows.append({
            "task": task,
            "eviann_raw_sites": len(eviann_raw[task]),
            "eviann_canonical_sites": len(eviann),
            "true_positive_sites": overlap,
            "precision": eviann_pr,
            "sensitivity": eviann_sn,
            "F1": eviann_f1,
        })
        recall_plot, precision_plot = thin_curve(recall, precision)
        curve_data[task] = (recall_plot, precision_plot)
        curve_rows.extend(
            {"task": task, "sensitivity": float(sn), "precision": float(pr)}
            for sn, pr in zip(recall_plot, precision_plot)
        )
        print(
            f"{task}: AP={ap:.6f}; best P/SN/F1="
            f"{precision[best]:.4f}/{recall[best]:.4f}/{f1[best]:.4f}; "
            f"EviAnn P/SN={eviann_pr:.4f}/{eviann_sn:.4f}",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    points = pd.DataFrame(point_rows)
    curves = pd.DataFrame(curve_rows)
    metrics.to_csv(output_dir / "candidate_metrics.tsv", sep="\t", index=False)
    points.to_csv(output_dir / "eviann_site_operating_points.tsv", sep="\t", index=False)
    with gzip.open(output_dir / "candidate_prsn_curve_points.tsv.gz", "wt") as handle:
        curves.to_csv(handle, sep="\t", index=False)
    return metrics, points, curve_data


def load_locus_metrics(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path, sep="\t")
    if {
        "query_mode", "locus_pr_percent", "locus_sn_percent", "locus_f1_percent"
    } <= set(frame.columns):
        labels = {
            "eviann_only": "EviAnn",
            "uniann_only": "UniAnn",
            "eviann_plus_uniann": "EviAnn + UniAnn",
        }
        frame = pd.DataFrame({
            "method": frame["query_mode"].map(labels),
            "precision": frame["locus_pr_percent"] / 100.0,
            "sensitivity": frame["locus_sn_percent"] / 100.0,
            "F1": frame["locus_f1_percent"] / 100.0,
        })
        order = {name: index for index, name in enumerate(labels.values())}
        frame = frame.sort_values(
            "method", key=lambda values: values.map(order)
        ).reset_index(drop=True)
    required = {"method", "precision", "sensitivity", "F1"}
    if not required <= set(frame.columns):
        raise ValueError(f"{path} lacks columns {sorted(required - set(frame.columns))}")
    return frame


def plot_report(metrics, points, curves, locus, output_dir: Path):
    fig = plt.figure(figsize=(13.2, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 1.05))
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    axes.append(fig.add_subplot(grid[1, 0]))
    locus_ax = fig.add_subplot(grid[1, 1:])

    for task, ax in zip(TASKS, axes):
        row = metrics.loc[metrics.task == task].iloc[0]
        point = points.loc[points.task == task].iloc[0]
        recall, precision = curves[task]
        ax.plot(recall, precision, color=COLORS[task], linewidth=2.2,
                label=f"4-state model (AUPRC {row.AUPRC:.3f})")
        ax.scatter(row.best_sensitivity, row.best_precision, s=58, marker="o",
                   facecolor="white", edgecolor=COLORS[task], linewidth=1.8,
                   label=f"Model best F1 {row.best_F1:.3f}", zorder=4)
        ax.scatter(point.sensitivity, point.precision, s=95, marker="D",
                   color="#c51b7d", edgecolor="white", linewidth=0.8,
                   label=f"EviAnn (F1 {point.F1:.3f})", zorder=5)
        ax.set_title(task.capitalize())
        ax.set_xlabel("Sensitivity")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.22)
        ax.legend(loc="lower left", fontsize=8, frameon=False)

    if locus is None:
        locus_ax.axis("off")
        locus_ax.text(
            0.5, 0.5, "Strict locus evaluation pending",
            ha="center", va="center", fontsize=16, color="#666666",
            transform=locus_ax.transAxes,
        )
    else:
        methods = locus["method"].tolist()
        x = np.arange(len(methods))
        width = 0.24
        series = (
            ("precision", "Precision", "#2166ac"),
            ("sensitivity", "Sensitivity", "#b2182b"),
            ("F1", "F1", "#4d9221"),
        )
        for offset, (column, label, color) in enumerate(series):
            values = locus[column].to_numpy(float)
            bars = locus_ax.bar(x + (offset - 1) * width, values, width,
                                label=label, color=color, alpha=0.88)
            for bar, value in zip(bars, values):
                locus_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012,
                              f"{100 * value:.1f}", ha="center", va="bottom", fontsize=9)
        locus_ax.set_xticks(x, methods)
        locus_ax.set_ylabel("Strict locus-level score")
        locus_ax.set_ylim(0, 1.08)
        locus_ax.set_title("Annotation reconstruction (strict match)")
        locus_ax.grid(axis="y", alpha=0.22)
        locus_ax.legend(frameon=False, ncol=3, loc="upper center")

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"human_chr1_four_state_uniann_report.{suffix}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--reference-gff", type=Path, required=True)
    parser.add_argument("--eviann-gff", type=Path, required=True)
    parser.add_argument("--seqid", default="NC_000001.11")
    parser.add_argument("--locus-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-candidate-tables", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.reuse_candidate_tables:
        metrics = pd.read_csv(args.output_dir / "candidate_metrics.tsv", sep="\t")
        points = pd.read_csv(args.output_dir / "eviann_site_operating_points.tsv", sep="\t")
        curve_frame = pd.read_csv(
            args.output_dir / "candidate_prsn_curve_points.tsv.gz", sep="\t"
        )
        curves = {
            task: (
                curve_frame.loc[curve_frame.task == task, "sensitivity"].to_numpy(),
                curve_frame.loc[curve_frame.task == task, "precision"].to_numpy(),
            )
            for task in TASKS
        }
    else:
        metrics, points, curves = candidate_metrics(args, args.output_dir)
    locus = load_locus_metrics(args.locus_metrics)
    plot_report(metrics, points, curves, locus, args.output_dir)


if __name__ == "__main__":
    main()
