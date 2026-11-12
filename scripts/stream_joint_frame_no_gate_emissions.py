#!/usr/bin/env python3
"""Convert pre-calibrated joint-frame probabilities to UniAnn emissions.

The input array can contain scalar-temperature or vector-scaled softmax
probabilities.  This converter applies neither the historical
``frame * (1 - noncoding)`` gate nor an additional binary-odds transform.

This is a historical headerless converter, not the interface for new
collaborator-facing runs. See docs/UNIANN_FRAME_TSV_CONTRACT.md. New runs
preserve the exported TSV header and pass that file directly to unchanged
UniAnn.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np


def fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


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
                if not math.isfinite(float(value)):
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


def convert(args):
    probabilities = np.load(args.index, mmap_mode="r")
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        raise ValueError(f"expected [chromosome_length,4], got {probabilities.shape}")
    if not 0 <= args.start0 < args.end0 <= len(probabilities):
        raise ValueError("requested interval is outside probability array")

    metadata_path = args.index.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if metadata is not None:
        method = metadata.get("calibration_method")
        if args.expected_calibration_method == "raw":
            valid_method = method in (None, "raw", "none")
        else:
            valid_method = method == args.expected_calibration_method
        if not valid_method:
            raise ValueError(
                f"expected calibration method {args.expected_calibration_method}, "
                f"observed {method}"
            )
        if args.expected_temperature is not None:
            observed = float(metadata.get("temperature", float("nan")))
            if observed != args.expected_temperature:
                raise ValueError(
                    f"expected temperature {args.expected_temperature}, "
                    f"observed {observed}"
                )

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
                # Stored order is noncoding, frame0, frame1, frame2.
                block = np.asarray(probabilities[block_start:block_end], dtype=np.float64)
                if args.output_temperature is not None:
                    if args.expected_temperature is None:
                        if args.expected_calibration_method != "raw":
                            raise ValueError(
                                "--output-temperature requires --expected-temperature "
                                "unless the input is raw"
                            )
                        input_temperature = 1.0
                    else:
                        input_temperature = args.expected_temperature
                    if args.output_temperature <= 0:
                        raise ValueError("--output-temperature must be positive")
                    # If p_in = softmax(z / T_in), then probabilities at T_out
                    # are softmax(log(p_in) * T_in / T_out).  This allows a
                    # T=9 probability array to be returned to raw T=1 without
                    # materializing another chromosome-sized array.
                    transformed = np.log(np.clip(block, np.finfo(np.float64).tiny, 1.0))
                    transformed *= input_temperature / args.output_temperature
                    transformed -= transformed.max(axis=1, keepdims=True)
                    np.exp(transformed, out=transformed)
                    block = transformed / transformed.sum(axis=1, keepdims=True)
                for offset, row in enumerate(block):
                    n = block_start - args.start0 + offset
                    noncoding = row[0]
                    frames = row[1:4]
                    for frame_index in range(3):
                        emission = math.log(math.e * frames[frame_index] + 1e-10)
                        coding_frame[(n - frame_index - 1) % 3] = emission
                    noncoding_score = math.log(math.e * noncoding + 1e-10)
                    if n - 2 in stops:
                        coding_frame[(n - 2) % 3] = -1_000_000.0
                        intron_score = -2.0
                    else:
                        intron_score = noncoding_score
                    output.write(
                        f"{n}\t{noncoding_score:.2f}\t{coding_frame[0]:.2f}\t"
                        f"{coding_frame[1]:.2f}\t{coding_frame[2]:.2f}\t{intron_score:.2f}\n"
                    )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, args.out_ps)
    finally:
        Path(temporary_name).unlink(missing_ok=True)

    validation = validate_out_ps_file(args.out_ps, length, args.output_columns)
    manifest = {
        "schema": "precalibrated_joint_frame_no_gate_uniann_emissions_v1",
        "interval": {"start0": args.start0, "end0": args.end0, "length": length},
        "seqid": args.seqid,
        "joint_frame_probabilities": fingerprint(args.index),
        "joint_frame_metadata": metadata,
        "transformation": {
            "preexisting_calibration_method": args.expected_calibration_method,
            "expected_temperature": args.expected_temperature,
            "output_temperature": args.output_temperature,
            "probability_rescaling": (
                "softmax(log(P_input) * T_input / T_output)"
                if args.output_temperature is not None else None
            ),
            "frame_emission": "log(e * P(frame) + 1e-10)",
            "noncoding_emission": "log(e * P(noncoding) + 1e-10)",
            "extra_frame_noncoding_gate": False,
            "additional_binary_odds_temperature": None,
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
    convert_parser.add_argument(
        "--expected-calibration-method",
        choices=("raw", "scalar_temperature", "vector_scaling"),
        default="scalar_temperature",
    )
    convert_parser.add_argument("--expected-temperature", type=float, default=None)
    convert_parser.add_argument(
        "--output-temperature",
        type=float,
        default=None,
        help=(
            "Rescale softmax probabilities from --expected-temperature to this "
            "temperature before constructing emissions. Raw inputs imply T=1."
        ),
    )
    args = parser.parse_args()
    if args.command == "convert":
        convert(args)


if __name__ == "__main__":
    main()
