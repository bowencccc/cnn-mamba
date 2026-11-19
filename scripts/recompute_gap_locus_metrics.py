#!/usr/bin/env python3
"""Correct Gap-locus metrics from existing strict gffcompare outputs."""

import argparse
import json
from pathlib import Path

import pandas as pd

from cnn_mamba.uniann_evaluate import exact_reference_loci, parse_stats


def calculate(work: Path) -> dict:
    uniann = parse_stats(work / "uniann.stats")
    baseline = parse_stats(work / "eviann.stats")
    union = parse_stats(work / "eviann_uniann.stats")
    baseline_matches = exact_reference_loci(
        work / "eviann.eviann.merged.gff.refmap"
    )
    union_matches = exact_reference_loci(
        work / "eviann_uniann.eviann_uniann.merged.gff.refmap"
    )
    if len(baseline_matches) != baseline["matched_loci"]:
        raise RuntimeError(f"baseline exact-match mismatch in {work}")
    if len(union_matches) != union["matched_loci"]:
        raise RuntimeError(f"union exact-match mismatch in {work}")

    recovered = len(union_matches - baseline_matches)
    available = baseline["reference_loci"] - len(baseline_matches)
    predicted = uniann["predicted_loci"]
    precision = recovered / predicted if predicted else 0.0
    sensitivity = recovered / available if available else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if precision + sensitivity else 0.0
    )
    if recovered > predicted:
        raise RuntimeError(
            f"recovered loci ({recovered}) exceed UniAnn predictions ({predicted}) in {work}"
        )
    return {
        "baseline_reference_loci": baseline["reference_loci"],
        "baseline_predicted_loci": baseline["predicted_loci"],
        "baseline_matched_loci": len(baseline_matches),
        "union_predicted_loci": union["predicted_loci"],
        "union_matched_loci": len(union_matches),
        "retained_baseline_matches": len(union_matches & baseline_matches),
        "available_gap_loci": available,
        "uniann_predicted_loci": predicted,
        "net_union_loci_change": (
            union["predicted_loci"] - baseline["predicted_loci"]
        ),
        "recovered_gap_loci": recovered,
        "gap_precision": precision,
        "gap_sensitivity": sensitivity,
        "gap_f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots", type=Path, nargs="+",
        help="Work directories or roots recursively containing gap_locus_metrics.tsv",
    )
    args = parser.parse_args()
    work_dirs = set()
    for root in args.roots:
        if (root / "gap_locus_metrics.tsv").is_file():
            work_dirs.add(root.resolve())
        elif root.is_dir():
            work_dirs.update(
                path.parent.resolve()
                for path in root.rglob("gap_locus_metrics.tsv")
            )
    if not work_dirs:
        parser.error("no existing gap_locus_metrics.tsv files found")

    for work in sorted(work_dirs):
        metrics = calculate(work)
        pd.DataFrame([metrics]).to_csv(
            work / "gap_locus_metrics.tsv", sep="\t", index=False,
            float_format="%.9g",
        )
        provenance_path = work / "provenance.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text())
            provenance["gap_locus_metrics"] = metrics
            provenance["gap_precision_denominator"] = (
                "all loci predicted by UniAnn alone after gffread -M; "
                "net union locus-count change is diagnostic only"
            )
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        print(work)


if __name__ == "__main__":
    main()
