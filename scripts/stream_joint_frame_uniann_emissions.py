#!/usr/bin/env python3
"""Historical headerless joint-frame-to-UniAnn emission converter.

The transformation intentionally matches UniAnn commit 2f5c0ab's
``convert_coding_frame.pl`` first six columns: independent binary-odds
temperature 3 followed by UniAnn's frame/noncoding emission construction.
The input probabilities may already have joint-softmax temperature scaling;
that calibration is recorded in their adjacent metadata file.

Do not use this script for new collaborator-facing CNN--Mamba runs.  Those runs
must preserve the exported TSV header and let unchanged UniAnn convert it; see
docs/UNIANN_FRAME_TSV_CONTRACT.md.  This historical converter uses UniAnn's
headerless row convention and is not equivalent to that interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np


FRAME_INDEX_OFFSET = -1
CONVERTER_SCHEMA = "joint_frame_uniann_emissions_v1"


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_fasta(path: Path) -> str:
    sequence = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith(">"):
                sequence.append(line.strip().upper())
    return "".join(sequence)


def validate_out_ps_file(path: Path, expected_length: int, expected_columns: int | None = None):
    rows = 0
    first = last = None
    final_newline = False
    with path.open("rb") as handle:
        for raw in handle:
            final_newline = raw.endswith(b"\n")
            fields = raw.rstrip(b"\n").split(b"\t")
            if expected_columns is None:
                expected_columns = len(fields)
            if len(fields) != expected_columns:
                raise ValueError(f"{path}: row {rows + 1} has {len(fields)} columns")
            position = int(fields[0])
            if position != rows:
                raise ValueError(f"{path}: expected position {rows}, observed {position}")
            for value in fields[1:6]:
                parsed = float(value)
                if not math.isfinite(parsed):
                    raise ValueError(f"{path}: nonfinite emission at row {rows + 1}")
            if first is None:
                first = position
            last = position
            rows += 1
    if rows != expected_length:
        raise ValueError(f"{path}: expected {expected_length} rows, observed {rows}")
    if rows and not final_newline:
        raise ValueError(f"{path}: missing final newline")
    return {
        "rows": rows,
        "first_position0": first,
        "last_position0": last,
        "columns": expected_columns,
        "final_newline": final_newline,
    }


def binary_odds_temperature(values: np.ndarray, temperature: float) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    return 1.0 / (
        1.0 + np.power((1.0 - values) / (values + 1e-10), 1.0 / temperature)
    )


def convert(args):
    probabilities = np.load(args.index, mmap_mode="r")
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError(f"expected [chromosome_length,4], got {probabilities.shape}")
    if not 0 <= args.start0 < args.end0 <= len(probabilities):
        raise ValueError("requested interval is outside probability array")
    sequence = read_fasta(args.out_fasta)
    length = args.end0 - args.start0
    if len(sequence) != length:
        raise ValueError(f"FASTA length {len(sequence)} != interval length {length}")
    stops = {match.start() for match in re.finditer(r"TAA|TAG|TGA", sequence)}

    args.out_ps.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{args.out_ps.name}.tmp.", dir=args.out_ps.parent)
    coding_frame = [0.0, 0.0, 0.0]
    try:
        with os.fdopen(fd, "w") as output:
            for block_start in range(args.start0, args.end0, args.block_size):
                block_end = min(args.end0, block_start + args.block_size)
                # Stored order: noncoding, frame0, frame1, frame2.
                raw = np.asarray(probabilities[block_start:block_end], dtype=np.float64)
                scaled = binary_odds_temperature(raw, 3.0)
                for offset, row in enumerate(scaled):
                    n = block_start - args.start0 + offset
                    noncoding = row[0]
                    frames = row[1:4]
                    for frame_index in range(3):
                        emission = math.log(
                            math.e * frames[frame_index] * (1.0 - noncoding) + 1e-10
                        )
                        coding_frame[(n - frame_index + FRAME_INDEX_OFFSET) % 3] = emission
                    nc_score = math.log(math.e * noncoding + 1e-10)
                    if n - 2 in stops:
                        coding_frame[(n - 2) % 3] = -1_000_000.0
                        intron = -2.0
                    else:
                        intron = nc_score
                    output.write(
                        f"{n}\t{nc_score:.2f}\t{coding_frame[0]:.2f}\t"
                        f"{coding_frame[1]:.2f}\t{coding_frame[2]:.2f}\t{intron:.2f}\n"
                    )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, args.out_ps)
    finally:
        Path(temporary_name).unlink(missing_ok=True)

    validation = validate_out_ps_file(args.out_ps, length, args.output_columns)
    probability_metadata = args.index.parent / "metadata.json"
    manifest = {
        "schema": CONVERTER_SCHEMA,
        "interval": {"start0": args.start0, "end0": args.end0, "length": length},
        "seqid": args.seqid,
        "joint_frame_probabilities": fingerprint(args.index),
        "joint_frame_metadata": (
            json.loads(probability_metadata.read_text()) if probability_metadata.is_file() else None
        ),
        "transformation": {
            "preexisting_calibration": "read from joint_frame_metadata",
            "uniann_builtin_equivalent": "independent_binary_odds_temperature",
            "uniann_builtin_temperature": 3.0,
            "frame_gate": "scaled_frame_probability * (1 - scaled_noncoding_probability)",
            "source_uniann_commit": "2f5c0ab",
            "frame_index_mapping": (
                f"(local_position0 - model_phase {FRAME_INDEX_OFFSET:+d}) mod 3"
            ),
        },
        "output": fingerprint(args.out_ps),
        "validation": validation,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(validation), flush=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--index", type=Path, required=True)
    convert_parser.add_argument("--start0", type=int, required=True)
    convert_parser.add_argument("--end0", type=int, required=True)
    convert_parser.add_argument("--out-ps", type=Path, required=True)
    convert_parser.add_argument("--out-fasta", type=Path, required=True)
    convert_parser.add_argument("--seqid", required=True)
    convert_parser.add_argument("--manifest", type=Path, required=True)
    convert_parser.add_argument("--output-columns", type=int, choices=(6,), default=6)
    convert_parser.add_argument("--block-size", type=int, default=100_000)
    args = parser.parse_args()
    if args.command == "convert":
        convert(args)


if __name__ == "__main__":
    main()
