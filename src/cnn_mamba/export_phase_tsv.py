#!/usr/bin/env python3
"""Export chromosome-wide phase probability arrays as a tabular TSV file."""

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
    parser.add_argument("--argmax", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrom", default="chrX")
    parser.add_argument("--strand", choices=("+", "-"), default="+")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    args = parser.parse_args()

    sentinel = Path(str(args.output) + ".complete.json")
    if args.output.is_file() and sentinel.is_file():
        print(f"already complete: {args.output}", flush=True)
        return
    probabilities = np.load(args.probabilities, mmap_mode="r")
    argmax = np.load(args.argmax, mmap_mode="r")
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise RuntimeError(
            f"expected probability shape [length, 3], got {probabilities.shape}"
        )
    if argmax.shape != (probabilities.shape[0],):
        raise RuntimeError(
            f"argmax shape {argmax.shape} does not match {probabilities.shape[0]}"
        )
    if args.chunk_size < 1:
        parser.error("chunk size must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    started = time.time()
    length = probabilities.shape[0]
    with partial.open("w") as handle:
        for start in range(0, length, args.chunk_size):
            end = min(length, start + args.chunk_size)
            values = np.asarray(probabilities[start:end], dtype=np.float32)
            frame = pd.DataFrame({
                "chrom": args.chrom,
                "position_1based": np.arange(start + 1, end + 1, dtype=np.int64),
                "strand": args.strand,
                "phase0_probability": values[:, 0],
                "phase1_probability": values[:, 1],
                "phase2_probability": values[:, 2],
                "predicted_phase": np.asarray(argmax[start:end], dtype=np.uint8),
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
        "schema": "per_base_phase_probability_tsv_v1",
        "output": str(args.output.resolve()),
        "source_probabilities": str(args.probabilities.resolve()),
        "source_argmax": str(args.argmax.resolve()),
        "chromosome": args.chrom,
        "strand": args.strand,
        "rows": length,
        "position_convention": "one-based genomic coordinate",
        "columns": [
            "chrom", "position_1based", "strand",
            "phase0_probability", "phase1_probability", "phase2_probability",
            "predicted_phase",
        ],
        "bytes": args.output.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }
    sentinel.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
