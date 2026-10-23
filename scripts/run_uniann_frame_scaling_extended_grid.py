#!/usr/bin/env python3
"""Tune UniAnn frame-emission strength for EviAnn+UniAnn locus accuracy."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import time

import run_uniann_frame_scaling_grid as base


WORK_ROOT = base.ROOT / "work/uniann/frame_scaling_extended_grid_v1"
RESULT_ROOT = base.ROOT / "results/frame_scaling_extended_grid_v1"


def condition_name(prefix, scale, bias):
    scale_text = f"{scale:.3f}".replace(".", "p")
    bias_text = f"{bias:.3f}".replace(".", "p")
    return f"{prefix}_s{scale_text}_b{bias_text}"


CONDITIONS = []
for temperature in (6.0, 7.0, 8.0, 9.0, 10.0):
    CONDITIONS.append({
        "condition": condition_name("joint", temperature, 0.0),
        "family": "joint_softmax_no_gate",
        "converter_method": "joint_softmax",
        "scale": temperature,
        "coding_bias": 0.0,
        "description": f"Joint softmax T={temperature:g}, no extra gate",
    })
for temperature in (7.0, 8.0, 9.0):
    for bias in (0.01, 0.02, 0.03):
        CONDITIONS.append({
            "condition": condition_name("joint", temperature, bias),
            "family": "joint_softmax_coding_bias",
            "converter_method": "joint_softmax",
            "scale": temperature,
            "coding_bias": bias,
            "description": (
                f"Joint softmax T={temperature:g}, coding bias={bias:.2f}"
            ),
        })
for emission_scale in (0.10, 0.125, 0.15, 0.175, 0.20):
    for bias in (0.0, 0.01, 0.02):
        CONDITIONS.append({
            "condition": condition_name("emission", emission_scale, bias),
            "family": "direct_emission_affine",
            "converter_method": "emission_affine",
            "scale": emission_scale,
            "coding_bias": bias,
            "description": (
                f"Direct emission scale={emission_scale:g}, "
                f"coding bias={bias:.2f}"
            ),
        })


def write_summary(rows, filename="strict_locus_metrics.tsv"):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output = RESULT_ROOT / filename
    fields = [
        "condition", "family", "scale", "coding_bias", "description",
        "method", "sn", "pr", "f1", "predicted_loci", "matched_loci",
        "missed_loci", "novel_loci",
    ]
    temporary = output.with_suffix(".tsv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def cleanup_derived_inputs(work):
    for name in (
        "out.ps.txt", "out.ps.txt.complete.json", "out.gt.txt", "out.ag.txt",
        "out.atg.txt", "out.stop.txt",
    ):
        path = work / name
        if path.is_file():
            path.unlink()


def main():
    for required in (
        base.FASTA, base.REFERENCE, base.EVIANN, base.SITE_SCORES,
        base.FRAME_TSV, base.CONVERTER,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    commit = subprocess.check_output(
        ["git", "-C", str(base.UNIANN), "rev-parse", "HEAD"], text=True
    ).strip()

    grid_start = int(os.environ.get("UNIANN_GRID_START", "1"))
    grid_end = int(os.environ.get("UNIANN_GRID_END", str(len(CONDITIONS))))
    result_name = os.environ.get(
        "UNIANN_GRID_RESULT_NAME", "strict_locus_metrics.tsv"
    )
    if not (1 <= grid_start <= grid_end <= len(CONDITIONS)):
        raise ValueError(
            f"invalid grid slice {grid_start}:{grid_end} for {len(CONDITIONS)}"
        )

    rows = []
    selected = list(enumerate(CONDITIONS[grid_start - 1:grid_end], grid_start))
    for index, spec in selected:
        started = time.time()
        work = WORK_ROOT / spec["condition"]
        work.mkdir(parents=True, exist_ok=True)
        base.ensure_link(base.FASTA, work / "genome.fa")
        base.ensure_link(base.SITE_SCORES, work / "scores.txt")
        base.ensure_link(base.FRAME_TSV, work / "coding.txt")
        prediction = work / "genome.fa.uniann.gff"

        if not prediction.is_file():
            emission = work / "out.ps.txt"
            temporary = work / "out.ps.txt.tmp"
            if not emission.is_file():
                with base.FRAME_TSV.open("rb") as source, temporary.open("wb") as output:
                    base.log_run(
                        ["perl", base.CONVERTER, base.FASTA.resolve(),
                         spec["converter_method"], spec["scale"],
                         spec["coding_bias"]],
                        work, work / "emission_conversion.log",
                        stdin=source, stdout=output,
                    )
                os.replace(temporary, emission)
            base.log_run(
                [base.UNIANN / "bin/uniann.sh", "-f", "genome.fa", "-c",
                 "coding.txt", "-s", "scores.txt", "-n"],
                work, work / "uniann.log",
            )

        alone = base.parse_stats(base.strict_compare(work, prediction, "uniann"))
        combined = work / "eviann_uniann.concat.gff"
        if not combined.is_file():
            with combined.open("w") as destination:
                with base.EVIANN.open() as source:
                    destination.writelines(source)
                with prediction.open() as source:
                    destination.writelines(
                        line for line in source if not line.startswith("#")
                    )
        merged = base.parse_stats(
            base.strict_compare(work, combined, "eviann_uniann")
        )

        for method, metrics in (("UniAnn", alone), ("EviAnn+UniAnn", merged)):
            rows.append({
                **{key: spec[key] for key in (
                    "condition", "family", "scale", "coding_bias",
                    "description",
                )},
                "method": method,
                **metrics,
            })
        write_summary(rows, result_name)
        (work / "provenance.json").write_text(json.dumps({
            **spec,
            "grid_index": index,
            "grid_size": len(CONDITIONS),
            "uniann_commit": commit,
            "frame_probabilities": str(base.FRAME_TSV.resolve()),
            "site_scores": str(base.SITE_SCORES.resolve()),
            "reference": str(base.REFERENCE.resolve()),
            "elapsed_seconds": time.time() - started,
            "metrics": {"UniAnn": alone, "EviAnn+UniAnn": merged},
        }, indent=2) + "\n")
        cleanup_derived_inputs(work)
        print(
            f"[{index}/{len(CONDITIONS)}] {spec['condition']} "
            f"combined Sn={merged['sn']:.1f} Pr={merged['pr']:.1f} "
            f"F1={merged['f1']:.4f}", flush=True,
        )

    print(RESULT_ROOT / result_name, flush=True)


if __name__ == "__main__":
    main()
