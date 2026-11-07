#!/usr/bin/env python3
"""Recompute tie-aware and cross-fitted metrics from saved OOF score files."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from cnn_mamba.human_site_quality import (
    average_precision, cross_fitted_threshold_metrics, threshold_for_precision,
)


def read_scores(path):
    labels, scores, folds, baseline = [], [], [], []
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            label = int(row["label"])
            if label < 0:
                continue
            labels.append(label)
            scores.append(float(row["oof_score"]))
            folds.append(int(row["fold"]))
            baseline.append(float(row["any_complete_non_low_quality"]))
    return tuple(np.asarray(values) for values in (labels, scores, folds, baseline))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.result_dir / "summary_corrected.json"
    original = json.loads((args.result_dir / "summary.json").read_text())
    corrected = json.loads(json.dumps(original))
    corrected["evaluation_revision"] = {
        "average_precision": "tie-aware; equal scores form one threshold group",
        "threshold_evaluation": (
            "each chromosome fold uses a threshold fitted only on the other folds"
        ),
        "models_retrained": False,
    }
    for task in ("start", "stop"):
        labels, scores, folds, baseline = read_scores(
            args.result_dir / f"{task}_site_scores.tsv.gz"
        )
        summary = corrected["tasks"][task]
        summary["oof_average_precision"] = average_precision(labels, scores)
        summary["baseline_complete_non_low_quality_AP"] = average_precision(
            labels, baseline
        )
        summary["precision_operating_points"] = [
            threshold_for_precision(labels, scores, target)
            for target in (0.80, 0.85, 0.90, 0.95)
        ]
        summary["cross_fitted_precision_operating_points"] = [
            cross_fitted_threshold_metrics(labels, scores, folds, target)
            for target in (0.80, 0.85, 0.90, 0.95)
        ]
    output.write_text(json.dumps(corrected, indent=2) + "\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
