#!/usr/bin/env python3
"""Evaluate chromosome-wide four-state non-CDS/frame probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from .evaluate_gated_phase import BinnedBinaryMetrics
from .prepare_droso import (
    build_cds_track, build_phase_track, parse_cds_phase_records,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--reference-gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()

    probabilities = np.load(
        args.prediction_dir / "joint_frame_probabilities.npy", mmap_mode="r"
    )
    argmax = np.load(
        args.prediction_dir / "joint_frame_argmax.npy", mmap_mode="r"
    )
    fasta = pysam.FastaFile(str(args.fasta))
    length = fasta.get_reference_length(args.chrom)
    fasta.close()
    if probabilities.shape != (length, 4) or argmax.shape != (length,):
        raise RuntimeError("joint-frame output shape mismatch")

    records = parse_cds_phase_records(args.reference_gtf)[args.chrom]["+"]
    reference_phase = build_phase_track(records, length, "+")
    reference_cds = build_cds_track(records, length)
    cds_tracker = BinnedBinaryMetrics()
    phase_trackers = [BinnedBinaryMetrics() for _ in range(3)]
    confusion = np.zeros((4, 4), dtype=np.int64)
    max_probability_sum_error = 0.0
    argmax_mismatches = 0

    for start in range(0, length, args.chunk_size):
        end = min(length, start + args.chunk_size)
        values = np.asarray(probabilities[start:end], dtype=np.float32)
        predicted = np.asarray(argmax[start:end], dtype=np.uint8)
        phase_truth = reference_phase[start:end]
        cds_truth = reference_cds[start:end]
        cds_tracker.add(1.0 - values[:, 0], cds_truth)
        valid_phase = phase_truth >= 0
        for phase in range(3):
            phase_trackers[phase].add(
                values[valid_phase, phase + 1],
                phase_truth[valid_phase] == phase,
            )
        valid_joint = phase_truth != -2
        reference_joint = np.zeros(end - start, dtype=np.uint8)
        for phase in range(3):
            reference_joint[phase_truth == phase] = phase + 1
        np.add.at(
            confusion,
            (reference_joint[valid_joint], predicted[valid_joint]),
            1,
        )
        max_probability_sum_error = max(
            max_probability_sum_error,
            float(np.max(np.abs(values.sum(1) - 1.0))),
        )
        argmax_mismatches += int(
            np.count_nonzero(values.argmax(1) != predicted)
        )

    cds_metrics = cds_tracker.metrics()
    phase_metrics = [tracker.metrics() for tracker in phase_trackers]
    class_rows = []
    for cls, name in enumerate(("non_CDS", "frame_0", "frame_1", "frame_2")):
        tp = int(confusion[cls, cls])
        actual = int(confusion[cls].sum())
        predicted = int(confusion[:, cls].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        class_rows.append({
            "class": name, "precision": precision, "recall": recall,
            "F1": f1, "support": actual, "predicted": predicted,
        })
    class_frame = pd.DataFrame(class_rows)
    metric_rows = [{"target": "CDS", "scores": "1-P(non_CDS)", **cds_metrics}]
    metric_rows.extend(
        {"target": f"frame_{phase}", "scores": "joint probability", **metrics}
        for phase, metrics in enumerate(phase_metrics)
    )
    metric_frame = pd.DataFrame(metric_rows)
    payload = {
        "schema": "joint_frame_evaluation_v1",
        "prediction_directory": str(args.prediction_dir.resolve()),
        "reference": str(args.reference_gtf.resolve()),
        "chromosome": args.chrom,
        "strand": "+",
        "metric": "4096-bin bounded-memory average precision",
        "cds_metrics": cds_metrics,
        "frame_macro_AP": float(np.mean([row["AP"] for row in phase_metrics])),
        "joint_confusion_matrix_rows_reference_columns_prediction": confusion.tolist(),
        "joint_macro_F1": float(class_frame.F1.mean()),
        "maximum_probability_sum_error": max_probability_sum_error,
        "argmax_mismatches": argmax_mismatches,
        "excluded_reference_phase_conflict_bases": int(
            (reference_phase == -2).sum()
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(
        args.output_dir / "joint_frame_auprc.tsv", sep="\t", index=False,
        float_format="%.9g",
    )
    class_frame.to_csv(
        args.output_dir / "joint_class_metrics.tsv", sep="\t", index=False,
        float_format="%.9g",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
