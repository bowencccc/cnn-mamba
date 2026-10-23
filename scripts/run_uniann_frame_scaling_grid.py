#!/usr/bin/env python3
"""Run the latest UniAnn on a leakage-controlled joint-frame scaling grid."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
UNIANN = Path("/home/bowencccc/UniAnn")
FASTA = ROOT / "data/raw/dmel_chrX.fa"
REFERENCE = ROOT / "data/raw/dmel_chrX_plus_CDS_reference.gtf"
EVIANN = ROOT / "data/raw/dmel_chrX_plus_EviAnn_CDS.gff"
SITE_SCORES = (
    ROOT / "artifacts/scores/joint_frame_grid_v1/b010/chrxtrain/w10/"
    "chrX_plus_scores.tsv"
)
FRAME_TSV = (
    ROOT / "artifacts/joint_frame_predictions/joint_frame_grid_v1/b010/"
    "chrxtrain/w10/chrX_plus/chrX_plus_joint_frame_probabilities.tsv"
)
CONVERTER = ROOT / "scripts/convert_joint_frame_for_uniann.pl"
WORK_ROOT = ROOT / "work/uniann/frame_scaling_grid_latest_v1"
RESULT_ROOT = ROOT / "results/frame_scaling_grid_latest_v1"

METHODS = (
    ("raw_double_gate", "Raw + frame*(1-nonCDS)"),
    ("binary_t6", "Independent binary-odds T=6"),
    ("binary_t7", "Independent binary-odds T=7"),
    ("softmax_t6", "Multiclass softmax T=6"),
    ("softmax_t7", "Multiclass softmax T=7"),
    ("raw_joint", "Raw joint P(state), no extra gate"),
)


def log_run(command, cwd, log_path, stdin=None, stdout=None):
    command = [str(value) for value in command]
    with log_path.open("a") as log:
        log.write("+ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(
            command, cwd=cwd, stdin=stdin, stdout=stdout,
            stderr=log, check=True,
        )


def ensure_link(source, destination):
    source = source.resolve()
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to replace {destination}")
    destination.symlink_to(source)


def parse_stats(path):
    text = path.read_text()
    locus = re.search(r"Locus level:\s*([0-9.]+)\s*\|\s*([0-9.]+)", text)
    query = re.search(r"Query mRNAs\s*:\s*\d+ in\s*(\d+) loci", text)
    matched = re.search(r"Matching loci:\s*(\d+)", text)
    missed = re.search(r"Missed loci:\s*(\d+)/", text)
    novel = re.search(r"Novel loci:\s*(\d+)/", text)
    if not all((locus, query, matched, missed, novel)):
        raise RuntimeError(f"cannot parse {path}")
    sensitivity, precision = map(float, locus.groups())
    return {
        "sn": sensitivity,
        "pr": precision,
        "f1": 2 * sensitivity * precision / (sensitivity + precision),
        "predicted_loci": int(query.group(1)),
        "matched_loci": int(matched.group(1)),
        "missed_loci": int(missed.group(1)),
        "novel_loci": int(novel.group(1)),
    }


def strict_compare(work, query, prefix):
    merged = work / f"{prefix}.merged.gff"
    stats = work / f"{prefix}.stats"
    if stats.is_file():
        return stats
    log_run(
        [UNIANN / "bin/gffread", "-M", query, "-o", merged],
        work, work / f"{prefix}.gffread.log",
    )
    log_run(
        [UNIANN / "bin/gffcompare", "--strict-match", "-e", "0",
         "-r", REFERENCE.resolve(), merged, "-o", work / prefix],
        work, work / f"{prefix}.gffcompare.log",
    )
    return stats


def write_summary(rows):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output = RESULT_ROOT / "strict_locus_metrics.tsv"
    fields = [
        "condition", "description", "method", "sn", "pr", "f1",
        "predicted_loci", "matched_loci", "missed_loci", "novel_loci",
    ]
    temporary = output.with_suffix(".tsv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def main():
    for required in (FASTA, REFERENCE, EVIANN, SITE_SCORES, FRAME_TSV, CONVERTER):
        if not required.is_file():
            raise FileNotFoundError(required)
    for required in (UNIANN / "bin/uniann", UNIANN / "bin/uniann.sh",
                     UNIANN / "bin/gffread", UNIANN / "bin/gffcompare"):
        if not required.is_file():
            raise FileNotFoundError(required)

    commit = subprocess.check_output(
        ["git", "-C", str(UNIANN), "rev-parse", "HEAD"], text=True
    ).strip()
    all_rows = []
    for condition, description in METHODS:
        started = time.time()
        work = WORK_ROOT / condition
        work.mkdir(parents=True, exist_ok=True)
        ensure_link(FASTA, work / "genome.fa")
        ensure_link(SITE_SCORES, work / "scores.txt")
        ensure_link(FRAME_TSV, work / "coding.txt")

        emission = work / "out.ps.txt"
        emission_marker = work / "out.ps.txt.complete.json"
        if not emission_marker.is_file():
            temporary = work / "out.ps.txt.tmp"
            with FRAME_TSV.open("rb") as source, temporary.open("wb") as output:
                log_run(
                    ["perl", CONVERTER, FASTA.resolve(), condition],
                    work, work / "emission_conversion.log",
                    stdin=source, stdout=output,
                )
            os.replace(temporary, emission)
            emission_marker.write_text(json.dumps({
                "condition": condition,
                "source": str(FRAME_TSV.resolve()),
                "bytes": emission.stat().st_size,
            }, indent=2) + "\n")

        prediction = work / "genome.fa.uniann.gff"
        if not prediction.is_file():
            log_run(
                [UNIANN / "bin/uniann.sh", "-f", "genome.fa", "-c", "coding.txt",
                 "-s", "scores.txt", "-n"],
                work, work / "uniann.log",
            )

        alone = parse_stats(strict_compare(work, prediction, "uniann"))
        all_rows.append({
            "condition": condition, "description": description,
            "method": "UniAnn", **alone,
        })

        combined = work / "eviann_uniann.concat.gff"
        if not combined.is_file():
            with combined.open("w") as destination:
                with EVIANN.open() as source:
                    destination.writelines(source)
                with prediction.open() as source:
                    destination.writelines(
                        line for line in source if not line.startswith("#")
                    )
        merged = parse_stats(strict_compare(work, combined, "eviann_uniann"))
        all_rows.append({
            "condition": condition, "description": description,
            "method": "EviAnn+UniAnn", **merged,
        })
        write_summary(all_rows)
        (work / "provenance.json").write_text(json.dumps({
            "condition": condition,
            "description": description,
            "uniann_commit": commit,
            "checkpoint": str((
                ROOT / "runs/joint_frame_grid_v1/b010/chrxtrain/w10/best_model.pt"
            ).resolve()),
            "frame_probabilities": str(FRAME_TSV.resolve()),
            "site_scores": str(SITE_SCORES.resolve()),
            "fasta": str(FASTA.resolve()),
            "reference": str(REFERENCE.resolve()),
            "elapsed_seconds": time.time() - started,
            "metrics": [all_rows[-2], all_rows[-1]],
        }, indent=2) + "\n")
        print(condition, all_rows[-2], all_rows[-1], flush=True)

    print(RESULT_ROOT / "strict_locus_metrics.tsv", flush=True)


if __name__ == "__main__":
    main()
