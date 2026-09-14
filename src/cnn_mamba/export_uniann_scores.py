#!/usr/bin/env python3
"""Export NPY candidate scores in the historical chrX_plus_scores.tsv format."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
import pysam


SITE_TYPES = ("donor", "acceptor", "start", "stop")
TYPE_NAMES = np.asarray(SITE_TYPES, dtype=object)
FIXED_MOTIFS = np.asarray(["GT", "AG", "ATG", ""], dtype=object)
STOP_MOTIFS = np.asarray(["TAA", "TAG", "TGA"], dtype=object)
ENCODE = np.full(256, 4, dtype=np.uint8)
for base, value in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    ENCODE[ord(base)] = value
A, G, T = 0, 2, 3
COMPLETION_VERSION = 1


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sidecar_path(output):
    output = Path(output)
    return output.with_suffix(output.suffix + ".json")


def completion_path(output):
    return Path(str(output) + ".complete.json")


def validate_completion(output, score_dir):
    output = Path(output).resolve()
    score_dir = Path(score_dir).resolve()
    sidecar = sidecar_path(output)
    sentinel = completion_path(output)
    try:
        complete = json.loads(sentinel.read_text())
        if complete.get("format") != "legacy_score_tsv_completion":
            return False, "unrecognized completion format"
        if complete.get("version") != COMPLETION_VERSION:
            return False, "unsupported completion version"
        if file_sha256(score_dir / "metadata.json") != complete.get(
            "source_metadata_sha256"
        ):
            return False, "source metadata digest mismatch"
        for path, prefix in ((output, "tsv"), (sidecar, "sidecar")):
            if not path.is_file():
                return False, f"missing {prefix}"
            if path.stat().st_size != int(complete[f"{prefix}_bytes"]):
                return False, f"{prefix} size mismatch"
            if file_sha256(path) != complete[f"{prefix}_sha256"]:
                return False, f"{prefix} digest mismatch"
        provenance = json.loads(sidecar.read_text())
        if int(provenance.get("rows", -1)) != int(complete.get("rows", -2)):
            return False, "row-count provenance mismatch"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def quarantine_stale_sentinel(path):
    path = Path(path)
    if not path.exists():
        return
    target = path.with_name(
        f"{path.name}.stale.{time.strftime('%Y%m%dT%H%M%S')}.{os.getpid()}"
    )
    os.replace(path, target)
    print(f"moved stale completion marker aside: {path} -> {target}", flush=True)


def audit_motifs(score_dir, fasta_path, accession):
    fasta = pysam.FastaFile(str(fasta_path))
    sequence = fasta.fetch(accession).upper()
    fasta.close()
    encoded = ENCODE[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    results = {}
    for name in SITE_TYPES:
        positions = np.load(score_dir / f"{name}_positions0.npy", mmap_mode="r")
        if len(positions) > 1 and not np.all(positions[1:] > positions[:-1]):
            raise RuntimeError(f"{name} positions are not strictly increasing")
        if name == "donor":
            valid = (encoded[positions] == G) & (encoded[positions + 1] == T)
        elif name == "acceptor":
            valid = (encoded[positions] == A) & (encoded[positions + 1] == G)
        elif name == "start":
            valid = (
                (encoded[positions] == A) & (encoded[positions + 1] == T)
                & (encoded[positions + 2] == G)
            )
        else:
            codes = np.load(score_dir / "stop_motif_codes.npy", mmap_mode="r")
            expected0 = np.full(len(codes), T, dtype=np.uint8)
            expected1 = np.where(codes == 2, G, A)
            expected2 = np.where(codes == 0, A, np.where(codes == 1, G, A))
            valid = (
                (encoded[positions] == expected0)
                & (encoded[positions + 1] == expected1)
                & (encoded[positions + 2] == expected2)
            )
        matches = int(valid.sum())
        if matches != len(positions):
            raise RuntimeError(f"{name}: motif audit {matches:,}/{len(positions):,}")
        results[name] = {"matches": matches, "total": len(positions), "fraction": 1.0}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrom", required=True, help="Historical output name, e.g. X or 1")
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    complete, reason = validate_completion(args.output, args.score_dir)
    if args.validate_only:
        if not complete:
            raise RuntimeError(f"legacy TSV is incomplete: {reason}")
        print(f"valid complete legacy TSV: {args.output.resolve()}", flush=True)
        return
    if complete:
        print(f"legacy TSV already complete: {args.output.resolve()}", flush=True)
        return
    args.output = args.output.resolve()
    quarantine_stale_sentinel(completion_path(args.output))

    started = time.time()
    metadata = json.loads((args.score_dir / "metadata.json").read_text())
    audit = audit_motifs(args.score_dir, args.fasta, args.accession)
    print(f"motif audit passed: {audit}", flush=True)

    position_parts, score_parts, type_parts, stop_code_parts = [], [], [], []
    for type_code, name in enumerate(SITE_TYPES):
        positions = np.asarray(
            np.load(args.score_dir / f"{name}_positions0.npy", mmap_mode="r")
        )
        scores = np.asarray(
            np.load(args.score_dir / f"{name}_scores.npy", mmap_mode="r")
        )
        position_parts.append(positions)
        score_parts.append(scores)
        type_parts.append(np.full(len(positions), type_code, dtype=np.uint8))
        if name == "stop":
            stop_code_parts.append(np.asarray(
                np.load(args.score_dir / "stop_motif_codes.npy", mmap_mode="r")
            ))
        else:
            stop_code_parts.append(np.full(len(positions), 255, dtype=np.uint8))

    positions = np.concatenate(position_parts)
    scores = np.concatenate(score_parts)
    type_codes = np.concatenate(type_parts)
    stop_codes = np.concatenate(stop_code_parts)
    order = np.argsort(positions, kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = len(order)

    sidecar = {
        "tsv": str(args.output.resolve()),
        "source_score_directory": str(args.score_dir.resolve()),
        "checkpoint": metadata["checkpoint"],
        "chrom": args.chrom,
        "fasta_accession": args.accession,
        "strand": "+",
        "columns": ["chrom", "pos", "strand", "type", "motif", "prob"],
        "position_convention": "one-based genomic motif start (pos = internal positions0 + 1)",
        "row_order": "globally ascending genomic pos",
        "motif_audit": audit,
        "rows": rows,
        "elapsed_seconds": None,
    }
    sidecar_file = sidecar_path(args.output)
    output_descriptor, output_temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.tmp.", dir=args.output.parent
    )
    sidecar_descriptor, sidecar_temporary_name = tempfile.mkstemp(
        prefix=f".{sidecar_file.name}.tmp.", dir=sidecar_file.parent
    )
    output_temporary = Path(output_temporary_name)
    sidecar_temporary = Path(sidecar_temporary_name)
    try:
        with os.fdopen(output_descriptor, "w", newline="") as output:
            header = True
            for left in range(0, rows, args.chunk_size):
                selected = order[left:min(rows, left + args.chunk_size)]
                selected_types = type_codes[selected]
                motifs = FIXED_MOTIFS[selected_types].copy()
                is_stop = selected_types == 3
                motifs[is_stop] = STOP_MOTIFS[stop_codes[selected][is_stop]]
                frame = pd.DataFrame({
                    "chrom": args.chrom,
                    # Historical file convention: one-based genomic motif start.
                    "pos": positions[selected].astype(np.uint64) + 1,
                    "strand": "+",
                    "type": TYPE_NAMES[selected_types],
                    "motif": motifs,
                    "prob": scores[selected],
                })
                frame.to_csv(
                    output, sep="\t", index=False, header=header,
                    float_format="%.17g", lineterminator="\n",
                )
                header = False
                done = min(rows, left + args.chunk_size)
                if done % 5_000_000 < args.chunk_size or done == rows:
                    elapsed = time.time() - started
                    print(
                        f"{args.output.name}: {done:,}/{rows:,} rows, "
                        f"{done/max(elapsed, 1e-9):,.0f} rows/s", flush=True,
                    )
            output.flush()
            os.fsync(output.fileno())
        sidecar["elapsed_seconds"] = time.time() - started
        with os.fdopen(sidecar_descriptor, "w") as output:
            json.dump(sidecar, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(output_temporary, args.output)
        os.replace(sidecar_temporary, sidecar_file)
    except Exception:
        print(
            f"legacy TSV was not completely published; temporary files retained: "
            f"{output_temporary}, {sidecar_temporary}",
            flush=True,
        )
        raise

    complete_payload = {
        "format": "legacy_score_tsv_completion",
        "version": COMPLETION_VERSION,
        "tsv": str(args.output),
        "sidecar": str(sidecar_file),
        "source_score_directory": str(args.score_dir.resolve()),
        "source_metadata_sha256": file_sha256(args.score_dir / "metadata.json"),
        "rows": rows,
        "tsv_bytes": args.output.stat().st_size,
        "tsv_sha256": file_sha256(args.output),
        "sidecar_bytes": sidecar_file.stat().st_size,
        "sidecar_sha256": file_sha256(sidecar_file),
        "created_unix_time": time.time(),
    }
    atomic_write_json(completion_path(args.output), complete_payload)
    print(f"finished in {sidecar['elapsed_seconds']/60:.1f} min: {args.output}", flush=True)


if __name__ == "__main__":
    main()
