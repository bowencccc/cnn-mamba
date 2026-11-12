"""Build UniAnn coding-frame emissions with an explicit calibration recipe."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np


def read_fasta(path: Path) -> str:
    sequence = []
    records = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                records += 1
            else:
                sequence.append(line.strip().upper())
    if records != 1:
        raise ValueError(f"expected one FASTA record, observed {records}")
    return "".join(sequence)


def binary_odds(values: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(values, 1e-10, 1.0)
    return 1.0 / (1.0 + ((1.0 - clipped) / clipped) ** (1.0 / temperature))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert four-state non-CDS/frame probabilities into UniAnn "
            "out.ps.txt emissions without hidden scaling. This low-level "
            "headerless path does not reproduce the collaborator-facing "
            "headered TSV contract; use export-joint-frame plus uniann-eval "
            "--require-cnn-mamba-tsv-header for canonical runs."
        )
    )
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seqid", required=True)
    parser.add_argument("--block-size", type=int, default=100_000)
    parser.add_argument(
        "--expected-calibration-method",
        choices=("raw", "scalar_temperature", "vector_scaling"),
    )
    parser.add_argument("--expected-temperature", type=float)
    parser.add_argument(
        "--binary-odds-temperature", type=float,
        help="Optional UniAnn-style independent binary-odds temperature.",
    )
    parser.add_argument(
        "--frame-noncoding-gate", action="store_true",
        help="Multiply each frame probability by 1-P(non_CDS).",
    )
    args = parser.parse_args()
    if args.block_size < 1:
        parser.error("--block-size must be positive")
    if args.binary_odds_temperature is not None and args.binary_odds_temperature <= 0:
        parser.error("--binary-odds-temperature must be positive")

    probabilities = np.load(args.probabilities, mmap_mode="r")
    if probabilities.ndim != 2 or probabilities.shape[1] != 4:
        parser.error(f"expected [chromosome_length,4], got {probabilities.shape}")
    sequence = read_fasta(args.fasta)
    if len(sequence) != len(probabilities):
        parser.error(
            f"FASTA length {len(sequence)} != probability rows {len(probabilities)}"
        )

    metadata_path = args.probabilities.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    if args.expected_calibration_method is not None:
        if metadata is None:
            parser.error("calibration validation requested but metadata.json is absent")
        observed = metadata.get("calibration_method")
        valid = (
            observed in (None, "raw", "none")
            if args.expected_calibration_method == "raw"
            else observed == args.expected_calibration_method
        )
        if not valid:
            parser.error(
                f"expected calibration {args.expected_calibration_method}, observed {observed}"
            )
    if args.expected_temperature is not None:
        if metadata is None or float(metadata.get("temperature", float("nan"))) != args.expected_temperature:
            parser.error(
                f"expected temperature {args.expected_temperature}, observed "
                f"{None if metadata is None else metadata.get('temperature')}"
            )

    stops = {match.start() for match in re.finditer(r"TAA|TAG|TGA", sequence)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or Path(str(args.output) + ".manifest.json")
    started = time.time()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", dir=args.output.parent
    )
    frames_by_genomic_phase = [0.0, 0.0, 0.0]
    try:
        with os.fdopen(descriptor, "w") as output:
            for block_start in range(0, len(probabilities), args.block_size):
                block_end = min(len(probabilities), block_start + args.block_size)
                block = np.asarray(probabilities[block_start:block_end], dtype=np.float64)
                if args.binary_odds_temperature is not None:
                    block = binary_odds(block, args.binary_odds_temperature)
                for offset, row in enumerate(block):
                    position = block_start + offset
                    noncoding = float(row[0])
                    frames = row[1:4]
                    for frame_index, frame_probability in enumerate(frames):
                        if args.frame_noncoding_gate:
                            frame_probability *= 1.0 - noncoding
                        emission = math.log(math.e * frame_probability + 1e-10)
                        frames_by_genomic_phase[(position - frame_index - 1) % 3] = emission
                    noncoding_score = math.log(math.e * noncoding + 1e-10)
                    if position - 2 in stops:
                        frames_by_genomic_phase[(position - 2) % 3] = -1_000_000.0
                        intron_score = -2.0
                    else:
                        intron_score = noncoding_score
                    output.write(
                        f"{position}\t{noncoding_score:.2f}\t"
                        f"{frames_by_genomic_phase[0]:.2f}\t"
                        f"{frames_by_genomic_phase[1]:.2f}\t"
                        f"{frames_by_genomic_phase[2]:.2f}\t{intron_score:.2f}\n"
                    )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, args.output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)

    payload = {
        "schema": "cnn_mamba_uniann_emissions_v1",
        "seqid": args.seqid,
        "rows": len(probabilities),
        "probabilities": str(args.probabilities.resolve()),
        "fasta": str(args.fasta.resolve()),
        "source_metadata": metadata,
        "transformation": {
            "frame_index_mapping": "(position - frame_index - 1) mod 3",
            "cnn_mamba_headered_tsv_compatible": False,
            "binary_odds_temperature": args.binary_odds_temperature,
            "frame_noncoding_gate": args.frame_noncoding_gate,
            "frame_emission": "log(e * P(frame) + 1e-10)",
            "noncoding_emission": "log(e * P(non_CDS) + 1e-10)",
        },
        "output": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "elapsed_seconds": time.time() - started,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
