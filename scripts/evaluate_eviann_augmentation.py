#!/usr/bin/env python3
"""Strictly compare EviAnn alone and EviAnn augmented with a UniAnn GFF."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import run_uniann_frame_scaling_grid as base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eviann-gff", type=Path, required=True)
    parser.add_argument("--uniann-gff", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    eviann = args.eviann_gff.resolve()
    uniann = args.uniann_gff.resolve()
    for path in (eviann, uniann, base.REFERENCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    eviann_stats = base.parse_stats(base.strict_compare(work, eviann, "eviann"))
    combined = work / "eviann_uniann.concat.gff"
    if not combined.is_file():
        with combined.open("w") as destination:
            destination.write(eviann.read_text())
            if not eviann.read_text().endswith("\n"):
                destination.write("\n")
            with uniann.open() as source:
                destination.writelines(
                    line for line in source if not line.startswith("#")
                )
    combined_stats = base.parse_stats(
        base.strict_compare(work, combined, "eviann_uniann")
    )
    rows = [
        {"annotation": args.label, "method": "EviAnn", **eviann_stats},
        {"annotation": args.label, "method": "EviAnn+UniAnn", **combined_stats},
    ]
    fields = (
        "annotation", "method", "sn", "pr", "f1", "predicted_loci",
        "matched_loci", "missed_loci", "novel_loci",
    )
    with (work / "strict_locus_metrics.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (work / "provenance.json").write_text(json.dumps({
        "label": args.label,
        "eviann_gff": str(eviann),
        "uniann_gff": str(uniann),
        "reference": str(base.REFERENCE.resolve()),
        "metrics": rows,
    }, indent=2) + "\n")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
