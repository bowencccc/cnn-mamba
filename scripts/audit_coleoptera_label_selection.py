#!/usr/bin/env python3
"""Audit loci excluded by the Coleoptera EviAnn label-selection policy."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

from cnn_mamba.prepare_droso import CHROMS, cds_length, gff_attrs, open_text


def parse(path: Path):
    accessions = set(CHROMS.values())
    transcripts = {}
    cds = defaultdict(list)
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 9 or fields[0] not in accessions:
                continue
            attrs = gff_attrs(fields[8])
            if fields[2] == "mRNA" and attrs.get("ID"):
                tid = attrs["ID"]
                transcripts[tid] = {
                    "locus": attrs.get("Parent") or attrs.get("geneID") or tid,
                    "chrom": fields[0],
                    "strand": fields[6],
                    "evidence": attrs.get("evidence", attrs.get("Evidence", "")),
                    "class": attrs.get("class", attrs.get("class_code", "")),
                }
            elif fields[2] == "CDS":
                interval = (int(fields[3]) - 1, int(fields[4]))
                for tid in attrs.get("Parent", "").split(","):
                    if tid:
                        cds[tid].append(interval)
    return transcripts, cds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transcripts, cds = parse(args.input)
    per_locus = defaultdict(list)
    for tid, record in transcripts.items():
        per_locus[record["locus"]].append(tid)

    rows = []
    reasons = Counter()
    complete_class_sets = Counter()
    evidence_sets = Counter()
    chromosome_counts = Counter()
    for locus, tids in sorted(per_locus.items()):
        complete = [
            tid for tid in tids if transcripts[tid]["evidence"].lower() == "complete"
        ]
        exact = [tid for tid in complete if transcripts[tid]["class"] == "="]
        fallback = [tid for tid in complete if transcripts[tid]["class"] == "k"]
        eligible = [tid for tid in exact + fallback if cds.get(tid)]
        if eligible:
            continue

        if not complete:
            reason = "no_complete_transcript"
        elif not (exact or fallback):
            reason = "complete_present_but_class_not_equal_or_k"
        else:
            reason = "complete_equal_or_k_missing_CDS"
        reasons[reason] += 1

        record0 = transcripts[tids[0]]
        evidence_values = sorted({transcripts[tid]["evidence"] or "<missing>" for tid in tids})
        class_values = sorted({transcripts[tid]["class"] or "<missing>" for tid in tids})
        complete_classes = sorted({transcripts[tid]["class"] or "<missing>" for tid in complete})
        evidence_sets[",".join(evidence_values)] += 1
        chromosome_counts[record0["chrom"]] += 1
        if complete:
            complete_class_sets[",".join(complete_classes)] += 1

        pair_counts = Counter(
            (transcripts[tid]["evidence"] or "<missing>",
             transcripts[tid]["class"] or "<missing>")
            for tid in tids
        )
        pair_text = ",".join(
            f"{evidence}|{class_code}:{count}"
            for (evidence, class_code), count in sorted(pair_counts.items())
        )
        rows.append({
            "locus": locus,
            "chrom": record0["chrom"],
            "strand": record0["strand"],
            "exclusion_reason": reason,
            "transcript_count": len(tids),
            "complete_transcript_count": len(complete),
            "evidence_values": ",".join(evidence_values),
            "class_values": ",".join(class_values),
            "complete_class_values": ",".join(complete_classes),
            "evidence_class_counts": pair_text,
            "longest_CDS_bp_any": max((cds_length(cds[tid]) for tid in tids), default=0),
            "longest_CDS_bp_complete": max(
                (cds_length(cds[tid]) for tid in complete), default=0
            ),
            "transcript_ids": ",".join(sorted(tids)),
        })

    output = args.output_dir / "excluded_loci.tsv.gz"
    with gzip.open(output, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": str(args.input.resolve()),
        "scope": "Drosophila primary chromosomes used by the training pipeline",
        "input_transcripts": len(transcripts),
        "input_loci": len(per_locus),
        "excluded_loci": len(rows),
        "exclusion_reason_counts": dict(reasons),
        "excluded_evidence_set_counts": dict(evidence_sets),
        "excluded_complete_class_set_counts": dict(complete_class_sets),
        "excluded_chromosome_counts": dict(chromosome_counts),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (args.output_dir / "exclusion_reason_counts.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("exclusion_reason", "loci"))
        writer.writerows(reasons.most_common())
    print(json.dumps(summary, indent=2))
    print(output)


if __name__ == "__main__":
    main()
