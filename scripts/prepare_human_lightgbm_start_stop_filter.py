#!/usr/bin/env python3
"""Create Human per-window start/stop loss masks from cross-fitted LightGBM P90."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import re

import numpy as np

from cnn_mamba.window_storage import WindowStore, WindowWriter


NAME = re.compile(r"^(1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|X|Y)_(plus|minus)_(\d+)\.npz$")


def thresholds_by_fold(summary, task):
    operating_point = next(
        row
        for row in summary["tasks"][task]["cross_fitted_precision_operating_points"]
        if row["target_precision"] == 0.9
    )
    return {
        int(row["fold"]): float(row["threshold_fitted_without_held_fold"])
        for row in operating_point["folds"]
    }


def read_passed_sites(path, thresholds):
    passed = set()
    available = set()
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["chrom"], row["strand"], int(row["position_0based"]))
            available.add(key)
            fold = int(row["fold"])
            if float(row["oof_score"]) >= thresholds[fold]:
                passed.add(key)
    return passed, available


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--lightgbm-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(
        (args.lightgbm_dir / "summary_corrected.json").read_text()
    )
    thresholds = {
        1: thresholds_by_fold(summary, "start"),
        2: thresholds_by_fold(summary, "stop"),
    }
    passed, available = {}, {}
    for label, task in ((1, "start"), (2, "stop")):
        passed[label], available[label] = read_passed_sites(
            args.lightgbm_dir / f"{task}_site_scores.tsv.gz",
            thresholds[label],
        )

    counts = Counter()
    for split in ("train", "val", "test"):
        store = WindowStore(args.data_root / split)
        with WindowWriter(args.output_root / f"{split}.h5") as writer:
            for index, path in enumerate(store.files):
                source = Path(path)
                match = NAME.match(source.name)
                if not match:
                    raise ValueError(f"unexpected Human window name: {source.name}")
                chrom_text, strand_text, start_text = match.groups()
                chrom = f"chr{chrom_text}"
                strand = "+" if strand_text == "plus" else "-"
                window_start = int(start_text)
                with store.load(index) as data:
                    labels = data["start_stop_labels"]
                weights = np.ones(labels.shape, dtype=np.float32)
                for offset in np.flatnonzero(labels):
                    label = int(labels[offset])
                    position = (
                        window_start + int(offset)
                        if strand == "+"
                        else window_start + len(labels) - 3 - int(offset)
                    )
                    key = (chrom, strand, position)
                    if key in passed[label]:
                        counts[split, "retained", label] += 1
                    else:
                        weights[offset] = 0.0
                        state = "filtered" if key in available[label] else "unmapped_filtered"
                        counts[split, state, label] += 1
                writer.append(source.name, {"start_stop_weights": weights})
                if (index + 1) % 5000 == 0:
                    print(f"[{split}] {index + 1:,}/{len(store):,}", flush=True)

        store.close()

    output = {
        "schema": "human_lightgbm_crossfit_p90_start_stop_filter_v1",
        "data_root": str(args.data_root.resolve()),
        "lightgbm_dir": str(args.lightgbm_dir.resolve()),
        "semantics": (
            "EviAnn start/stop positives passing chromosome-held-out OOF P90 "
            "retain loss weight 1; rejected or unmapped positives receive 0 "
            "and are ignored, not relabeled negative; all negatives retain 1"
        ),
        "thresholds_by_fold": {
            "start": thresholds[1],
            "stop": thresholds[2],
        },
        "window_positive_occurrences": [
            {
                "split": split,
                "state": state,
                "task": "start" if label == 1 else "stop",
                "count": count,
            }
            for (split, state, label), count in sorted(counts.items())
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
