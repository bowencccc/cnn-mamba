#!/usr/bin/env python3
"""Write per-transcript Human EviAnn ORF-QC calls and aggregate counts."""

import argparse
import csv
import json
from pathlib import Path

from cnn_mamba.human_annotations import EVIANN_GFF, FASTA
from cnn_mamba.human_orf_qc import evaluate_eviann_orfs, result_dict, summarize_orf_qc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gff", type=Path, default=EVIANN_GFF)
    parser.add_argument("--fasta", type=Path, default=FASTA)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, results = evaluate_eviann_orfs(args.gff, args.fasta)
    rows = [result_dict(result) for result in results.values()]
    with (args.output_dir / "transcript_orf_qc.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_orf_qc(results)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
