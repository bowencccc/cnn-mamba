#!/usr/bin/env python3
"""Run latest UniAnn for one chrX-included incomplete-label checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import run_uniann_frame_scaling_grid as base


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--label", help="negative-weight label, e.g. n050")
    group.add_argument(
        "--tag", help="experiment tag below runs/artifacts/results"
    )
    parser.add_argument("--temperature", type=float, default=9.0)
    parser.add_argument(
        "--eviann-gff", type=Path, default=None,
        help="Override the chrX-plus EviAnn CDS GFF used for the combined run.",
    )
    args = parser.parse_args()

    tag = (
        Path(args.tag) if args.tag is not None
        else Path("incomplete_label_v2/negative_weight") / args.label
    )
    checkpoint = base.ROOT / "runs" / tag / "chrxtrain/w10/best_model.pt"
    scores = base.ROOT / "artifacts/scores" / tag / "chrxtrain/w10/chrX_plus_scores.tsv"
    frame_tsv = (
        base.ROOT / "artifacts/joint_frame_predictions" / tag
        / "chrxtrain/w10/chrX_plus/chrX_plus_joint_frame_probabilities.tsv"
    )
    result_root = base.ROOT / "results" / tag / "chrxtrain/w10/uniann_joint_t9"
    work = base.ROOT / "work/uniann" / tag / "w10/joint_t9"
    work.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    eviann = base.EVIANN if args.eviann_gff is None else args.eviann_gff.resolve()

    for required in (checkpoint, scores, frame_tsv, base.FASTA, base.REFERENCE,
                     eviann, base.CONVERTER):
        if not required.is_file():
            raise FileNotFoundError(required)

    base.ensure_link(base.FASTA, work / "genome.fa")
    base.ensure_link(scores, work / "scores.txt")
    base.ensure_link(frame_tsv, work / "coding.txt")

    started = time.time()
    emission = work / "out.ps.txt"
    if not emission.is_file():
        temporary = work / "out.ps.txt.tmp"
        with frame_tsv.open("rb") as source, temporary.open("wb") as output:
            base.log_run(
                ["perl", base.CONVERTER, base.FASTA.resolve(), "joint_softmax",
                 args.temperature, 0.0],
                work, work / "emission_conversion.log", stdin=source, stdout=output,
            )
        os.replace(temporary, emission)

    prediction = work / "genome.fa.uniann.gff"
    if not prediction.is_file():
        base.log_run(
            [base.UNIANN / "bin/uniann.sh", "-f", "genome.fa", "-c", "coding.txt",
             "-s", "scores.txt", "-n"],
            work, work / "uniann.log",
        )

    alone = base.parse_stats(base.strict_compare(work, prediction, "uniann"))
    combined = work / "eviann_uniann.concat.gff"
    if not combined.is_file():
        with combined.open("w") as destination:
            destination.write(eviann.read_text())
            with prediction.open() as source:
                destination.writelines(
                    line for line in source if not line.startswith("#")
                )
    merged = base.parse_stats(base.strict_compare(work, combined, "eviann_uniann"))

    rows = [
        {"method": "UniAnn", **alone},
        {"method": "EviAnn+UniAnn", **merged},
    ]
    fields = ["method", "sn", "pr", "f1", "predicted_loci", "matched_loci",
              "missed_loci", "novel_loci"]
    with (result_root / "strict_locus_metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    commit = subprocess.check_output(
        ["git", "-C", str(base.UNIANN), "rev-parse", "HEAD"], text=True
    ).strip()
    (result_root / "provenance.json").write_text(json.dumps({
        "label": args.label,
        "tag": str(tag),
        "temperature": args.temperature,
        "coding_transform": "joint_softmax_no_extra_gate",
        "checkpoint": str(checkpoint.resolve()),
        "site_scores": str(scores.resolve()),
        "frame_probabilities": str(frame_tsv.resolve()),
        "eviann_gff": str(eviann),
        "uniann_commit": commit,
        "elapsed_seconds": time.time() - started,
        "metrics": rows,
    }, indent=2) + "\n")
    print(result_root / "strict_locus_metrics.tsv")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
