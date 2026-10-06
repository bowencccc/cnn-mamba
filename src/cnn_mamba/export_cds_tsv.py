#!/usr/bin/env python3
"""Export a chromosome-wide CDS probability array as a per-base TSV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrom", default="chrX")
    parser.add_argument("--strand", choices=("+", "-"), default="+")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("threshold must be between zero and one")
    if args.chunk_size < 1:
        parser.error("chunk size must be positive")

    sentinel = Path(str(args.output) + ".complete.json")
    if args.output.is_file() and sentinel.is_file():
        print(f"already complete: {args.output}", flush=True)
        return

    probabilities = np.load(args.probabilities, mmap_mode="r")
    if probabilities.ndim != 1:
        raise RuntimeError(
            f"expected one probability per genomic position, got {probabilities.shape}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    started = time.time()
    length = probabilities.shape[0]
    coding_bases = 0
    with partial.open("w") as handle:
        for start in range(0, length, args.chunk_size):
            end = min(length, start + args.chunk_size)
            coding = np.asarray(probabilities[start:end], dtype=np.float32)
            predicted = coding >= args.threshold
            coding_bases += int(predicted.sum())
            frame = pd.DataFrame({
                "chrom": args.chrom,
                "position_1based": np.arange(start + 1, end + 1, dtype=np.int64),
                "strand": args.strand,
                "noncoding_probability": 1.0 - coding,
                "coding_probability": coding,
                "predicted_class": np.where(predicted, "coding", "noncoding"),
            })
            frame.to_csv(
                handle, sep="\t", index=False, header=start == 0,
                float_format="%.8g", lineterminator="\n",
            )
            if end % 2_500_000 < args.chunk_size or end == length:
                print(f"written={end:,}/{length:,}", flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, args.output)

    payload = {
        "schema": "per_base_cds_probability_tsv_v1",
        "output": str(args.output.resolve()),
        "source_probabilities": str(args.probabilities.resolve()),
        "chromosome": args.chrom,
        "strand": args.strand,
        "rows": length,
        "position_convention": "one-based genomic coordinate",
        "coding_threshold": args.threshold,
        "coding_bases": coding_bases,
        "noncoding_bases": length - coding_bases,
        "columns": [
            "chrom", "position_1based", "strand",
            "noncoding_probability", "coding_probability", "predicted_class",
        ],
        "bytes": args.output.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }
    sentinel.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
