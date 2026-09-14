#!/usr/bin/env python3
"""Compute exact plus-strand AUPRC for saved canonical-candidate scores."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


SITE_TYPES = ("donor", "acceptor", "start", "stop")
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
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return Path(output).with_suffix(".json")


def completion_path(output):
    return Path(str(output) + ".complete.json")


def validate_completion(output, score_dir):
    output = Path(output).resolve()
    score_dir = Path(score_dir).resolve()
    sidecar = sidecar_path(output)
    sentinel = completion_path(output)
    try:
        complete = json.loads(sentinel.read_text())
        if complete.get("format") != "candidate_auprc_completion":
            return False, "unrecognized completion format"
        if complete.get("version") != COMPLETION_VERSION:
            return False, "unsupported completion version"
        metadata_path = score_dir / "metadata.json"
        if file_sha256(metadata_path) != complete.get("source_metadata_sha256"):
            return False, "source metadata digest mismatch"
        for path, prefix in ((output, "tsv"), (sidecar, "sidecar")):
            if not path.is_file():
                return False, f"missing {prefix}"
            if path.stat().st_size != int(complete[f"{prefix}_bytes"]):
                return False, f"{prefix} size mismatch"
            if file_sha256(path) != complete[f"{prefix}_sha256"]:
                return False, f"{prefix} digest mismatch"
        summary = json.loads(sidecar.read_text())
        if len(summary.get("tasks", [])) != len(SITE_TYPES):
            return False, "sidecar lacks four tasks"
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


def reference_sites(reference_gtf, fasta_path, chrom_key):
    from . import prepare_droso

    prepare_droso.FASTA = Path(fasta_path).resolve()
    meta, exons, cds, explicit = prepare_droso.parse_reference(Path(reference_gtf))
    sites, rejected = prepare_droso.extract_sites(meta, exons, cds, explicit)
    reference = str(Path(reference_gtf).resolve())

    truth = {
        name: np.asarray(
            sorted(position for chrom, strand, position in sites[name]
                   if chrom == chrom_key and strand == "+"),
            dtype=np.uint32,
        )
        for name in SITE_TYPES
    }
    return truth, reference, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--reference-gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--chrom", required=True, help="FASTA accession used in the reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--logical-score-dir", type=Path,
        help="directory to record in provenance when reading from an atomic staging dir",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    complete, reason = validate_completion(args.output, args.score_dir)
    if args.validate_only:
        if not complete:
            raise RuntimeError(f"candidate AUPRC output is incomplete: {reason}")
        print(f"valid complete candidate AUPRC: {args.output.resolve()}", flush=True)
        return
    if complete:
        print(f"candidate AUPRC already complete: {args.output.resolve()}", flush=True)
        return
    quarantine_stale_sentinel(completion_path(args.output.resolve()))

    started = time.time()
    metadata = json.loads((args.score_dir / "metadata.json").read_text())
    truth, reference, rejected = reference_sites(args.reference_gtf, args.fasta, args.chrom)
    rows = []
    for name in SITE_TYPES:
        positions = np.load(
            args.score_dir / f"{name}_positions0.npy", mmap_mode="r"
        )
        scores = np.load(args.score_dir / f"{name}_scores.npy", mmap_mode="r")
        if len(positions) != len(scores):
            raise RuntimeError(f"{name}: position/score length mismatch")
        labels = np.isin(positions, truth[name], assume_unique=True)
        positives = int(labels.sum())
        if positives == 0:
            raise RuntimeError(f"{name}: no reference positives among candidates")
        auprc = float(average_precision_score(labels, scores))
        rows.append({
            "task": name,
            "candidates": len(positions),
            "reference_sites": len(truth[name]),
            "matched_positives": positives,
            "AUPRC": auprc,
        })
        print(
            f"{name}: AUPRC={auprc:.6f} positives={positives:,}/"
            f"{len(positions):,}",
            flush=True,
        )

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    sidecar = sidecar_path(args.output)
    summary = {
        "score_directory": str(
            (args.logical_score_dir or args.score_dir).resolve()
        ),
        "checkpoint": metadata["checkpoint"],
        "checkpoint_epoch": metadata.get("checkpoint_epoch"),
        "species": "drosophila",
        "chromosome": metadata["chromosome"],
        "strand": "+",
        "reference": reference,
        "candidate_scope": "all canonical candidates on the complete plus strand",
        "metric": "exact sklearn average_precision_score",
        "mean_AUPRC": float(frame["AUPRC"].mean()),
        "reference_rejected": rejected,
        "elapsed_seconds": time.time() - started,
        "tasks": rows,
    }
    tsv_descriptor, tsv_temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.tmp.", dir=args.output.parent
    )
    json_descriptor, json_temporary_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.tmp.", dir=sidecar.parent
    )
    tsv_temporary = Path(tsv_temporary_name)
    json_temporary = Path(json_temporary_name)
    try:
        with os.fdopen(tsv_descriptor, "w") as handle:
            frame.to_csv(handle, sep="\t", index=False, float_format="%.9g")
            handle.flush()
            os.fsync(handle.fileno())
        with os.fdopen(json_descriptor, "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tsv_temporary, args.output)
        os.replace(json_temporary, sidecar)
    except Exception:
        print(
            f"candidate AUPRC output not published; temporary files retained: "
            f"{tsv_temporary}, {json_temporary}",
            flush=True,
        )
        raise

    complete_payload = {
        "format": "candidate_auprc_completion",
        "version": COMPLETION_VERSION,
        "output": str(args.output),
        "sidecar": str(sidecar),
        "source_score_directory": str(
            (args.logical_score_dir or args.score_dir).resolve()
        ),
        "source_metadata_sha256": file_sha256(args.score_dir / "metadata.json"),
        "tsv_bytes": args.output.stat().st_size,
        "tsv_sha256": file_sha256(args.output),
        "sidecar_bytes": sidecar.stat().st_size,
        "sidecar_sha256": file_sha256(sidecar),
        "task_count": len(rows),
        "created_unix_time": time.time(),
    }
    atomic_write_json(completion_path(args.output), complete_payload)
    print(f"mean AUPRC={summary['mean_AUPRC']:.6f}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
