#!/usr/bin/env python3
"""Export four-state chromosome probabilities as a one-row-per-base TSV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd


INTERNAL_CLASSES = ("noncoding", "frame0", "frame1", "frame2")
OUTPUT_ORDER = (1, 2, 3, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--argmax", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrom", default="chrX")
    parser.add_argument("--strand", choices=("+", "-"), default="+")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()

    if args.chunk_size < 1:
        parser.error("chunk size must be positive")
    sentinel = Path(str(args.output) + ".complete.json")
    if args.output.is_file() and sentinel.is_file():
        print(f"already complete: {args.output}", flush=True)
        return

    probabilities = np.load(args.probabilities, mmap_mode="r")
    argmax = np.load(args.argmax, mmap_mode="r")
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise RuntimeError(
            f"expected probability shape [length, 4], got {probabilities.shape}"
        )
    if argmax.shape != (probabilities.shape[0],):
        raise RuntimeError(
            f"argmax shape {argmax.shape} does not match {probabilities.shape[0]}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    started = time.time()
    length = probabilities.shape[0]
    class_counts = {name: 0 for name in INTERNAL_CLASSES}
    maximum_probability_sum_error = 0.0
    argmax_mismatches = 0

    with partial.open("w") as handle:
        for start in range(0, length, args.chunk_size):
            end = min(length, start + args.chunk_size)
            values = np.asarray(probabilities[start:end], dtype=np.float32)
            saved_argmax = np.asarray(argmax[start:end], dtype=np.uint8)
            calculated_argmax = values.argmax(axis=1).astype(np.uint8)
            maximum_probability_sum_error = max(
                maximum_probability_sum_error,
                float(np.max(np.abs(values.sum(axis=1) - 1.0))),
            )
            argmax_mismatches += int(np.sum(saved_argmax != calculated_argmax))
            for class_index, name in enumerate(INTERNAL_CLASSES):
                class_counts[name] += int(np.sum(saved_argmax == class_index))

            frame = pd.DataFrame({
                "chrom": args.chrom,
                "position_1based": np.arange(start + 1, end + 1, dtype=np.int64),
                "strand": args.strand,
                "frame0_probability": values[:, OUTPUT_ORDER[0]],
                "frame1_probability": values[:, OUTPUT_ORDER[1]],
                "frame2_probability": values[:, OUTPUT_ORDER[2]],
                "noncoding_probability": values[:, OUTPUT_ORDER[3]],
                "predicted_state": np.asarray(INTERNAL_CLASSES)[saved_argmax],
            })
            frame.to_csv(
                handle, sep="\t", index=False, header=start == 0,
                float_format="%.8g", lineterminator="\n",
            )
            if end % 2_500_000 < args.chunk_size or end == length:
                print(f"written={end:,}/{length:,}", flush=True)
        handle.flush()
        os.fsync(handle.fileno())

    if argmax_mismatches:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"saved argmax disagrees with probabilities at {argmax_mismatches} bases"
        )
    os.replace(partial, args.output)
    payload = {
        "schema": "per_base_joint_frame_probability_tsv_v1",
        "output": str(args.output.resolve()),
        "source_probabilities": str(args.probabilities.resolve()),
        "source_argmax": str(args.argmax.resolve()),
        "source_internal_class_order": list(INTERNAL_CLASSES),
        "chromosome": args.chrom,
        "strand": args.strand,
        "rows": length,
        "position_convention": "one-based genomic coordinate",
        "columns": [
            "chrom", "position_1based", "strand",
            "frame0_probability", "frame1_probability",
            "frame2_probability", "noncoding_probability",
            "predicted_state",
        ],
        "predicted_state_counts": class_counts,
        "maximum_probability_sum_error": maximum_probability_sum_error,
        "argmax_mismatches": argmax_mismatches,
        "bytes": args.output.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }
    sentinel.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
