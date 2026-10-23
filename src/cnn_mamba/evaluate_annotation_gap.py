#!/usr/bin/env python3
"""Evaluate reference sites supported by versus missing from EviAnn."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from . import prepare_droso


SITE_TYPES = ("donor", "acceptor", "start", "stop")


def recall_at_precision(labels, scores, target_precision):
    precision, recall, _ = precision_recall_curve(labels, scores)
    eligible = recall[precision >= target_precision]
    return float(eligible.max()) if len(eligible) else 0.0


def site_sets(path, kind, fasta):
    prepare_droso.FASTA = Path(fasta).resolve()
    if kind == "eviann":
        meta, exons, cds, explicit = prepare_droso.parse_eviann(Path(path))
    else:
        meta, exons, cds, explicit = prepare_droso.parse_reference(Path(path))
    sites, rejected = prepare_droso.extract_sites(meta, exons, cds, explicit)
    return sites, rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--eviann-gff", type=Path, required=True)
    parser.add_argument("--reference-gtf", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--strand", choices=("+", "-"), default="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    eviann, eviann_rejected = site_sets(args.eviann_gff, "eviann", args.fasta)
    reference, reference_rejected = site_sets(
        args.reference_gtf, "reference", args.fasta
    )

    rows = []
    for task in SITE_TYPES:
        positions = np.load(
            args.score_dir / f"{task}_positions0.npy", mmap_mode="r"
        )
        scores = np.load(args.score_dir / f"{task}_scores.npy", mmap_mode="r")
        if len(positions) != len(scores):
            raise RuntimeError(f"{task}: position/score length mismatch")

        ref = {
            position for chrom, strand, position in reference[task]
            if chrom == args.chrom and strand == args.strand
        }
        evi = {
            position for chrom, strand, position in eviann[task]
            if chrom == args.chrom and strand == args.strand
        }
        supported = ref & evi
        missed = ref - evi
        eviann_only = evi - ref

        ref_labels = np.isin(positions, np.fromiter(ref, dtype=np.uint32))
        supported_labels = np.isin(
            positions, np.fromiter(supported, dtype=np.uint32)
        )
        missed_labels = np.isin(
            positions, np.fromiter(missed, dtype=np.uint32)
        )
        negatives = ~ref_labels

        row = {
            "task": task,
            "candidates": int(len(positions)),
            "reference_sites": int(len(ref)),
            "eviann_sites": int(len(evi)),
            "supported_sites": int(len(supported)),
            "missed_sites": int(len(missed)),
            "eviann_only_sites": int(len(eviann_only)),
            "overall_auprc": float(average_precision_score(ref_labels, scores)),
        }
        for label, positives in (
            ("supported", supported_labels), ("missed", missed_labels)
        ):
            subset = positives | negatives
            if not positives.any():
                row[f"{label}_auprc"] = None
                row[f"{label}_recall_at_p90"] = None
                row[f"{label}_recall_at_p95"] = None
                continue
            subset_labels = positives[subset]
            subset_scores = scores[subset]
            row[f"{label}_auprc"] = float(
                average_precision_score(subset_labels, subset_scores)
            )
            row[f"{label}_recall_at_p90"] = recall_at_precision(
                subset_labels, subset_scores, 0.90
            )
            row[f"{label}_recall_at_p95"] = recall_at_precision(
                subset_labels, subset_scores, 0.95
            )
        rows.append(row)
        print(
            f"{task}: overall={row['overall_auprc']:.6f} "
            f"supported={row['supported_auprc']:.6f} "
            f"missed={row['missed_auprc']:.6f} "
            f"missed_sites={len(missed):,}",
            flush=True,
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, sep="\t", index=False, float_format="%.9g")
    summary = {
        "score_dir": str(args.score_dir.resolve()),
        "eviann_gff": str(args.eviann_gff.resolve()),
        "reference_gtf": str(args.reference_gtf.resolve()),
        "fasta": str(args.fasta.resolve()),
        "chromosome": args.chrom,
        "strand": args.strand,
        "negative_definition": "canonical candidates absent from reference",
        "supported_definition": "Reference intersection EviAnn",
        "missed_definition": "Reference minus EviAnn",
        "eviann_rejected": eviann_rejected,
        "reference_rejected": reference_rejected,
        "tasks": rows,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
