#!/usr/bin/env python3
"""Create per-window start/stop loss-weight sidecars from LightGBM scores."""

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import re

import numpy as np

from cnn_mamba.prepare_droso import CHROMS


NAME = re.compile(r"^(.+)_(plus|minus)_(\d+)\.npz$")


def read_scores(path):
    result = {}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            result[
                row["chrom"], row["strand"], int(row["position_0based"])
            ] = float(row["human_lgbm_score"])
    return result


def tier_weight(score, medium_threshold, high_threshold, medium_weight):
    if score >= high_threshold:
        return 1.0, "high"
    if score >= medium_threshold:
        return medium_weight, "medium"
    return 0.0, "low"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--medium-weight", type=float, default=0.3)
    parser.add_argument("--start-medium-threshold", type=float, default=0.424418726716758)
    parser.add_argument("--start-high-threshold", type=float, default=0.6603122665221867)
    parser.add_argument("--stop-medium-threshold", type=float, default=0.3089827357956622)
    parser.add_argument("--stop-high-threshold", type=float, default=0.4712660934290894)
    args = parser.parse_args()
    if not 0 <= args.medium_weight <= 1:
        parser.error("--medium-weight must be in [0, 1]")

    score_lookup = {
        1: read_scores(args.score_dir / "start_scores.tsv.gz"),
        2: read_scores(args.score_dir / "stop_scores.tsv.gz"),
    }
    thresholds = {
        1: (args.start_medium_threshold, args.start_high_threshold),
        2: (args.stop_medium_threshold, args.stop_high_threshold),
    }
    counts = Counter()
    for split in ("train", "val", "test"):
        source_dir = args.data_root / split
        target_dir = args.output_root / split
        target_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(source_dir.glob("*.npz")):
            match = NAME.match(source.name)
            if not match:
                raise ValueError(f"unexpected window name: {source}")
            chrom_name, strand_name, start_text = match.groups()
            chrom = CHROMS[chrom_name]
            strand = "+" if strand_name == "plus" else "-"
            window_start = int(start_text)
            with np.load(source) as data:
                labels = data["start_stop_labels"]
            weights = np.ones(labels.shape, dtype=np.float32)
            for offset in np.flatnonzero(labels):
                label = int(labels[offset])
                position = (
                    window_start + int(offset)
                    if strand == "+"
                    else window_start + len(labels) - 3 - int(offset)
                )
                score = score_lookup[label].get((chrom, strand, position))
                if score is None:
                    counts[split, "unmapped", label] += 1
                    continue
                medium, high = thresholds[label]
                weight, tier = tier_weight(score, medium, high, args.medium_weight)
                weights[offset] = weight
                counts[split, tier, label] += 1
            np.savez_compressed(target_dir / source.name, start_stop_weights=weights)

    summary = {
        "schema": "dmel_human_lightgbm_start_stop_weights_v1",
        "data_root": str(args.data_root.resolve()),
        "score_dir": str(args.score_dir.resolve()),
        "medium_weight": args.medium_weight,
        "thresholds": {
            "start": {
                "medium": args.start_medium_threshold,
                "high": args.start_high_threshold,
            },
            "stop": {
                "medium": args.stop_medium_threshold,
                "high": args.stop_high_threshold,
            },
        },
        "window_positive_occurrences": [
            {"split": split, "tier": tier, "class": "start" if label == 1 else "stop", "count": count}
            for (split, tier, label), count in sorted(counts.items())
        ],
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
