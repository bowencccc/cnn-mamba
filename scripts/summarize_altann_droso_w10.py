#!/usr/bin/env python3
"""Summarize the Dmel chrX+ UniAnn versus AltAnn strict comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


LEVEL_PATTERN = re.compile(
    r"^\s*(Base|Exon|Intron|Intron chain|Transcript|Locus) level:"
    r"\s*([0-9.]+)\s*\|\s*([0-9.]+)",
    re.MULTILINE,
)
QUERY_PATTERN = re.compile(r"Query mRNAs\s*:\s*(\d+) in\s*(\d+) loci")


def parse_stats(path: Path) -> dict[str, object]:
    text = path.read_text()
    query = QUERY_PATTERN.search(text)
    if query is None:
        raise RuntimeError(f"Could not parse query counts from {path}")
    levels: dict[str, dict[str, float]] = {}
    for level, sensitivity, precision in LEVEL_PATTERN.findall(text):
        sn = float(sensitivity)
        pr = float(precision)
        f1 = 0.0 if sn + pr == 0 else 2.0 * sn * pr / (sn + pr)
        levels[level] = {"sensitivity": sn, "precision": pr, "f1": f1}
    expected = {"Base", "Exon", "Intron", "Intron chain", "Transcript", "Locus"}
    if set(levels) != expected:
        raise RuntimeError(f"Missing levels in {path}: {expected - set(levels)}")
    return {
        "query_transcripts": int(query.group(1)),
        "query_loci": int(query.group(2)),
        "levels": levels,
    }


def write_table(records: list[dict[str, object]], output: Path) -> None:
    fields = [
        "method", "dmel_short_cds_filter", "query_transcripts", "query_loci",
        "level", "sensitivity", "precision", "f1",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for record in records:
            parsed = record["parsed"]
            for level, metrics in parsed["levels"].items():
                writer.writerow({
                    "method": record["method"],
                    "dmel_short_cds_filter": record["filter"],
                    "query_transcripts": parsed["query_transcripts"],
                    "query_loci": parsed["query_loci"],
                    "level": level,
                    **metrics,
                })


def plot_main(records: list[dict[str, object]], output: Path) -> None:
    main = [record for record in records if record["main"]]
    methods = [str(record["method"]) for record in main]
    colors = ["#7f7f7f", "#4c78a8", "#f58518", "#54a24b"]
    metrics = ("sensitivity", "precision", "f1")
    labels = ("Sensitivity", "Precision", "F1")
    x = np.arange(len(metrics))
    width = 0.19

    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    for index, (record, color) in enumerate(zip(main, colors)):
        level_metrics = record["parsed"]["levels"]["Locus"]
        values = [level_metrics[name] for name in metrics]
        offset = (index - (len(main) - 1) / 2) * width
        bars = axis.bar(x + offset, values, width, color=color, label=methods[index])
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
            )
    axis.set_title("Locus level", fontsize=14, weight="bold")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 104)
    axis.set_ylabel("Percent (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.29), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specifications = [
        ("UniAnn", "uniann.stats", True, True),
        ("AltAnn K=10", "altann_k10_dmel_filter.stats", True, True),
        ("EviAnn + UniAnn", "eviann_uniann.stats", True, True),
        ("EviAnn + AltAnn K=10", "eviann_altann_k10_dmel_filter.stats", True, True),
        ("AltAnn K=10 (unfiltered)", "altann_k10.stats", False, False),
        ("EviAnn + AltAnn K=10 (unfiltered)", "eviann_altann_k10.stats", False, False),
    ]
    records = []
    for method, filename, short_filter, is_main in specifications:
        path = args.evaluation_dir / filename
        records.append({
            "method": method,
            "filter": short_filter,
            "main": is_main,
            "stats": str(path.resolve()),
            "parsed": parse_stats(path),
        })

    write_table(records, args.output_dir / "strict_metrics.tsv")
    plot_main(records, args.output_dir / "locus_comparison.png")
    (args.output_dir / "parsed_metrics.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
