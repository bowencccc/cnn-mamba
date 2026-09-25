#!/usr/bin/env python3
"""Evaluate raw and CDS-gated phase scores against chromosome CDS phase labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from .prepare_droso import (
    build_cds_track,
    build_phase_track,
    parse_cds_phase_records,
)


class BinnedBinaryMetrics:
    def __init__(self, bins=4096):
        self.bins = bins
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros(bins, dtype=np.int64)

    def add(self, scores, truth):
        indices = np.clip((scores * (self.bins - 1)).astype(np.int64), 0, self.bins - 1)
        self.positive += np.bincount(indices[truth], minlength=self.bins)
        self.negative += np.bincount(indices[~truth], minlength=self.bins)

    def metrics(self):
        positives = int(self.positive.sum())
        tp = np.cumsum(self.positive[::-1])
        fp = np.cumsum(self.negative[::-1])
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / max(positives, 1)
        ap = float(np.sum(precision * np.diff(np.r_[0.0, recall])))
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
        best = int(np.argmax(f1))
        return {
            "AP": ap,
            "best_F1": float(f1[best]),
            "precision": float(precision[best]),
            "recall": float(recall[best]),
            "threshold": float((self.bins - 1 - best) / (self.bins - 1)),
            "positives": positives,
            "evaluated_bases": positives + int(self.negative.sum()),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-phase", type=Path, required=True)
    parser.add_argument("--gated-dir", type=Path, required=True)
    parser.add_argument("--reference-gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()

    raw = np.load(args.raw_phase, mmap_mode="r")
    gated = np.load(
        args.gated_dir / "gated_phase_probabilities.npy", mmap_mode="r"
    )
    cds_probability = np.load(
        args.gated_dir / "cds_probability.npy", mmap_mode="r"
    )
    joint_argmax = np.load(
        args.gated_dir / "joint_class_argmax.npy", mmap_mode="r"
    )
    fasta = pysam.FastaFile(str(args.fasta))
    length = fasta.get_reference_length(args.chrom)
    fasta.close()
    if raw.shape != (length, 3) or gated.shape != (length, 3):
        raise RuntimeError(f"phase array shape mismatch for chromosome length {length}")
    if cds_probability.shape != (length,) or joint_argmax.shape != (length,):
        raise RuntimeError("CDS output shape mismatch")

    records = parse_cds_phase_records(args.reference_gtf)[args.chrom]["+"]
    reference_phase = build_phase_track(records, length, "+")
    reference_cds = build_cds_track(records, length)
    trackers = {"cds": BinnedBinaryMetrics()}
    for phase in range(3):
        trackers[f"raw_phase_{phase}"] = BinnedBinaryMetrics()
        trackers[f"gated_phase_{phase}"] = BinnedBinaryMetrics()
    confusion = np.zeros((4, 4), dtype=np.int64)
    max_gate_sum_difference = 0.0
    argmax_mismatches = 0

    for start in range(0, length, args.chunk_size):
        end = min(length, start + args.chunk_size)
        phase_truth = reference_phase[start:end]
        cds_truth = reference_cds[start:end]
        raw_chunk = np.asarray(raw[start:end], dtype=np.float32)
        gated_chunk = np.asarray(gated[start:end], dtype=np.float32)
        cds_chunk = np.asarray(cds_probability[start:end], dtype=np.float32)
        trackers["cds"].add(cds_chunk, cds_truth)
        valid = phase_truth != -2
        for phase in range(3):
            truth = phase_truth[valid] == phase
            trackers[f"raw_phase_{phase}"].add(raw_chunk[valid, phase], truth)
            trackers[f"gated_phase_{phase}"].add(gated_chunk[valid, phase], truth)
        max_gate_sum_difference = max(
            max_gate_sum_difference,
            float(np.max(np.abs(gated_chunk.sum(1) - cds_chunk))),
        )
        recomputed = np.column_stack((1.0 - cds_chunk, gated_chunk)).argmax(1)
        argmax_mismatches += int(np.sum(recomputed != joint_argmax[start:end]))
        reference_joint = np.zeros(end - start, dtype=np.uint8)
        for phase in range(3):
            reference_joint[phase_truth == phase] = phase + 1
        predicted = np.asarray(joint_argmax[start:end], dtype=np.uint8)
        np.add.at(confusion, (reference_joint[valid], predicted[valid]), 1)

    rows = []
    cds_metrics = trackers["cds"].metrics()
    rows.append({"target": "CDS", "scores": "CDS head", **cds_metrics})
    for phase in range(3):
        rows.append({
            "target": f"phase_{phase}", "scores": "raw phase",
            **trackers[f"raw_phase_{phase}"].metrics(),
        })
        rows.append({
            "target": f"phase_{phase}", "scores": "CDS-gated phase",
            **trackers[f"gated_phase_{phase}"].metrics(),
        })
    frame = pd.DataFrame(rows)
    raw_macro_ap = float(frame[frame.scores == "raw phase"].AP.mean())
    gated_macro_ap = float(frame[frame.scores == "CDS-gated phase"].AP.mean())
    class_rows = []
    for cls, name in enumerate(("non_CDS", "phase_0", "phase_1", "phase_2")):
        tp = int(confusion[cls, cls])
        actual = int(confusion[cls].sum())
        predicted = int(confusion[:, cls].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_rows.append({
            "class": name, "precision": precision, "recall": recall,
            "F1": f1, "support": actual, "predicted": predicted,
        })
    class_frame = pd.DataFrame(class_rows)
    payload = {
        "schema": "cds_gated_phase_evaluation_v1",
        "raw_phase": str(args.raw_phase.resolve()),
        "gated_directory": str(args.gated_dir.resolve()),
        "reference": str(args.reference_gtf.resolve()),
        "chromosome": args.chrom,
        "strand": "+",
        "metric": "4096-bin bounded-memory average precision",
        "raw_phase_macro_AP": raw_macro_ap,
        "gated_phase_macro_AP": gated_macro_ap,
        "macro_AP_difference": gated_macro_ap - raw_macro_ap,
        "cds_metrics": cds_metrics,
        "joint_confusion_matrix_rows_reference_columns_prediction": confusion.tolist(),
        "joint_macro_F1": float(class_frame.F1.mean()),
        "maximum_gated_sum_minus_cds_probability": max_gate_sum_difference,
        "joint_argmax_mismatches": argmax_mismatches,
        "excluded_reference_phase_conflict_bases": int((reference_phase == -2).sum()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        args.output_dir / "raw_vs_gated_phase_auprc.tsv", sep="\t",
        index=False, float_format="%.9g",
    )
    class_frame.to_csv(
        args.output_dir / "joint_class_metrics.tsv", sep="\t",
        index=False, float_format="%.9g",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
