#!/usr/bin/env python3
"""Evaluate learned vector-scaled joint-frame probabilities with UniAnn."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import time

import run_uniann_frame_scaling_grid as base


CONDITION = "vector_scaling_block_holdout"
DESCRIPTION = "Vector scaling fit on held chr3L 2-Mb blocks"
FRAME_TSV = (
    base.ROOT
    / "artifacts/joint_frame_predictions/joint_frame_grid_v1/b010/chrxtrain/w10"
    / "chrX_plus_vector_scaled/chrX_plus_joint_frame_probabilities.tsv"
)
CALIBRATION_JSON = (
    base.ROOT
    / "results/joint_frame_grid_v1/b010/chrxtrain/w10"
    / "vector_scaling_block_holdout/vector_scaling.json"
)
WORK = base.ROOT / "work/uniann/frame_scaling_extended_grid_v1" / CONDITION
OUTPUT = (
    base.ROOT / "results/frame_scaling_extended_grid_v1"
    / "vector_scaling_strict_locus_metrics.tsv"
)


def write_summary(rows):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "condition", "family", "scale", "coding_bias", "description",
        "method", "sn", "pr", "f1", "predicted_loci", "matched_loci",
        "missed_loci", "novel_loci",
    ]
    temporary = OUTPUT.with_suffix(".tsv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, OUTPUT)


def main():
    for required in (
        base.FASTA, base.REFERENCE, base.EVIANN, base.SITE_SCORES,
        FRAME_TSV, CALIBRATION_JSON, base.CONVERTER,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    started = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    base.ensure_link(base.FASTA, WORK / "genome.fa")
    base.ensure_link(base.SITE_SCORES, WORK / "scores.txt")
    base.ensure_link(FRAME_TSV, WORK / "coding.txt")

    emission = WORK / "out.ps.txt"
    marker = WORK / "out.ps.txt.complete.json"
    if not marker.is_file():
        temporary = WORK / "out.ps.txt.tmp"
        with FRAME_TSV.open("rb") as source, temporary.open("wb") as output:
            base.log_run(
                ["perl", base.CONVERTER, base.FASTA.resolve(), "raw_joint"],
                WORK, WORK / "emission_conversion.log",
                stdin=source, stdout=output,
            )
        os.replace(temporary, emission)
        marker.write_text(json.dumps({
            "condition": CONDITION,
            "source": str(FRAME_TSV.resolve()),
            "conversion": "raw_joint_no_extra_gate",
            "bytes": emission.stat().st_size,
        }, indent=2) + "\n")

    prediction = WORK / "genome.fa.uniann.gff"
    if not prediction.is_file():
        base.log_run(
            [base.UNIANN / "bin/uniann.sh", "-f", "genome.fa", "-c",
             "coding.txt", "-s", "scores.txt", "-n"],
            WORK, WORK / "uniann.log",
        )

    alone = base.parse_stats(base.strict_compare(WORK, prediction, "uniann"))
    combined = WORK / "eviann_uniann.concat.gff"
    if not combined.is_file():
        with combined.open("w") as destination:
            with base.EVIANN.open() as source:
                destination.writelines(source)
            with prediction.open() as source:
                destination.writelines(
                    line for line in source if not line.startswith("#")
                )
    merged = base.parse_stats(
        base.strict_compare(WORK, combined, "eviann_uniann")
    )

    common = {
        "condition": CONDITION,
        "family": "learned_vector_scaling",
        "scale": "nan",
        "coding_bias": "nan",
        "description": DESCRIPTION,
    }
    rows = [
        {**common, "method": "UniAnn", **alone},
        {**common, "method": "EviAnn+UniAnn", **merged},
    ]
    write_summary(rows)

    commit = subprocess.check_output(
        ["git", "-C", str(base.UNIANN), "rev-parse", "HEAD"], text=True
    ).strip()
    (WORK / "provenance.json").write_text(json.dumps({
        **common,
        "uniann_commit": commit,
        "calibration": json.loads(CALIBRATION_JSON.read_text()),
        "frame_probabilities": str(FRAME_TSV.resolve()),
        "site_scores": str(base.SITE_SCORES.resolve()),
        "reference": str(base.REFERENCE.resolve()),
        "elapsed_seconds": time.time() - started,
        "metrics": {"UniAnn": alone, "EviAnn+UniAnn": merged},
    }, indent=2) + "\n")
    print(json.dumps({"UniAnn": alone, "EviAnn+UniAnn": merged}, indent=2))
    print(OUTPUT)


if __name__ == "__main__":
    main()
