#!/usr/bin/env python3
"""Measure Human EviAnn evidence-stratum site agreement with RefSeq.

Agreement is exact membership of a canonical CDS donor/acceptor/start/stop site
in the RefSeq set using (chromosome, strand, 0-based motif-start position).
Validation chromosomes are reported separately so training-weight choices do
not use chr1, the final evaluation chromosome.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pysam

from cnn_mamba.human_annotations import (
    CHR_TO_NC,
    EVIANN_GFF,
    FASTA,
    REFSEQ_GFF,
    SITE_TYPES,
    extract_sites,
    open_text,
    parse_gff_attrs,
    parse_reference,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/human_eviann_evidence_refseq"
VALIDATION_CHROMS = {"chr2", "chr5", "chr9", "chr10", "chr22"}


def parse_eviann_evidence(path):
    tx_meta = {}
    exons = defaultdict(list)
    cds = defaultdict(list)
    evidence = {}
    low_quality = {}
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] not in CHR_TO_NC.values():
                continue
            feature = fields[2]
            if feature not in {"mRNA", "exon", "CDS"}:
                continue
            attrs = parse_gff_attrs(fields[8])
            if feature == "mRNA":
                tid = attrs.get("ID")
                if tid:
                    chrom = next(k for k, v in CHR_TO_NC.items() if v == fields[0])
                    tx_meta[tid] = (chrom, fields[6])
                    evidence[tid] = attrs.get("Evidence", "missing")
                    low_quality[tid] = "LOW_QUALITY_PROTEIN" in attrs.get(
                        "EvidenceProteinID", ""
                    )
                continue
            target = exons if feature == "exon" else cds
            start0, end0 = int(fields[3]) - 1, int(fields[4])
            for tid in attrs.get("Parent", "").split(","):
                if tid:
                    target[tid].append((start0, end0))
    return tx_meta, exons, cds, evidence, low_quality


def subset_annotation(tx_ids, tx_meta, exons, cds):
    tx_ids = set(tx_ids)
    return (
        {tid: tx_meta[tid] for tid in tx_ids if tid in tx_meta},
        {tid: exons[tid] for tid in tx_ids if tid in exons},
        {tid: cds[tid] for tid in tx_ids if tid in cds},
    )


def transcript_stratum(tid, evidence, low_quality):
    value = evidence[tid]
    if value == "complete":
        return "complete_low_quality" if low_quality[tid] else "complete_high_quality"
    if value == "protein_only":
        return (
            "protein_only_low_quality"
            if low_quality[tid] else "protein_only_high_quality"
        )
    return "transcript_only"


def exclusive_site_support(tx_meta, exons, cds, evidence, low_quality, fasta_path):
    """Assign each site to its strongest supporting transcript stratum."""
    hierarchy = [
        "complete_high_quality",
        "complete_low_quality",
        "protein_only_high_quality",
        "protein_only_low_quality",
        "transcript_only",
    ]
    rank = {name: index for index, name in enumerate(hierarchy)}
    support = {task: defaultdict(set) for task in SITE_TYPES}
    fasta = pysam.FastaFile(str(fasta_path))
    for tid, cds_segments in cds.items():
        if not cds_segments or tid not in tx_meta or tid not in exons:
            continue
        chrom, strand = tx_meta[tid]
        accession = CHR_TO_NC[chrom]
        chrom_len = fasta.get_reference_length(accession)
        stratum = transcript_stratum(tid, evidence, low_quality)
        cds_lo = min(start for start, _ in cds_segments)
        cds_hi = max(end for _, end in cds_segments)
        exon_segments = sorted(set(exons[tid]))
        for (_, left_end), (right_start, _) in zip(
            exon_segments, exon_segments[1:]
        ):
            if right_start <= left_end:
                continue
            low_site, high_site = left_end, right_start - 2
            if not (
                cds_lo <= low_site < cds_hi and cds_lo <= high_site < cds_hi
            ):
                continue
            low_motif = fasta.fetch(accession, low_site, low_site + 2).upper()
            high_motif = fasta.fetch(accession, high_site, high_site + 2).upper()
            canonical = (
                (low_motif, high_motif) == ("GT", "AG")
                if strand == "+" else
                (low_motif, high_motif) == ("CT", "AC")
            )
            if not canonical:
                continue
            donor, acceptor = (
                (low_site, high_site) if strand == "+" else
                (high_site, low_site)
            )
            support["donor"][(chrom, strand, donor)].add(stratum)
            support["acceptor"][(chrom, strand, acceptor)].add(stratum)

        segments = sorted(cds_segments)
        start = segments[0][0] if strand == "+" else segments[-1][1] - 3
        start_motif = (
            fasta.fetch(accession, start, start + 3).upper()
            if 0 <= start <= chrom_len - 3 else ""
        )
        if (strand == "+" and start_motif == "ATG") or (
            strand == "-" and start_motif == "CAT"
        ):
            support["start"][(chrom, strand, start)].add(stratum)

        stop = segments[-1][1] - 3 if strand == "+" else segments[0][0]
        stop_motif = (
            fasta.fetch(accession, stop, stop + 3).upper()
            if 0 <= stop <= chrom_len - 3 else ""
        )
        if (strand == "+" and stop_motif in {"TAA", "TAG", "TGA"}) or (
            strand == "-" and stop_motif in {"TTA", "CTA", "TCA"}
        ):
            support["stop"][(chrom, strand, stop)].add(stratum)
    fasta.close()

    exclusive = {
        name: {task: set() for task in SITE_TYPES} for name in hierarchy
    }
    for task, sites in support.items():
        for site, supporting_strata in sites.items():
            strongest = min(supporting_strata, key=rank.get)
            exclusive[strongest][task].add(site)
    return exclusive, hierarchy


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return None, None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def summarize(group, sites, reference, scopes):
    rows = []
    for scope, chromosomes in scopes.items():
        for task in SITE_TYPES:
            predicted = {site for site in sites[task] if site[0] in chromosomes}
            truth = {site for site in reference[task] if site[0] in chromosomes}
            matched = predicted & truth
            lo, hi = wilson_interval(len(matched), len(predicted))
            precision = len(matched) / len(predicted) if predicted else None
            sensitivity = len(matched) / len(truth) if truth else None
            rows.append({
                "scope": scope,
                "evidence_group": group,
                "task": task,
                "eviann_unique_sites": len(predicted),
                "refseq_unique_sites": len(truth),
                "matched_sites": len(matched),
                "agreement_precision": precision,
                "agreement_wilson95_low": lo,
                "agreement_wilson95_high": hi,
                "refseq_coverage": sensitivity,
            })
    return rows


def write_tsv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot_validation(rows, output):
    requested = [
        "complete", "protein_only", "transcript_only", "LOW_QUALITY_PROTEIN"
    ]
    tasks = list(SITE_TYPES)
    selected = {
        (row["evidence_group"], row["task"]): row
        for row in rows
        if row["scope"] == "validation" and row["evidence_group"] in requested
    }
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 9), sharex=True)
    width = 0.19
    x = list(range(len(tasks)))
    for index, (group, color) in enumerate(zip(requested, colors)):
        offset = (index - 1.5) * width
        agreement = [100 * selected[group, task]["agreement_precision"] for task in tasks]
        counts = [selected[group, task]["eviann_unique_sites"] for task in tasks]
        axes[0].bar([value + offset for value in x], agreement, width, label=group, color=color)
        axes[1].bar([value + offset for value in x], counts, width, label=group, color=color)
    axes[0].set_ylabel("Exact RefSeq agreement (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=2, frameon=False)
    axes[1].set_ylabel("Unique EviAnn sites")
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x, [task.capitalize() for task in tasks])
    axes[1].set_xlabel("Validation chromosomes: chr2, chr5, chr9, chr10, chr22")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def recommended_weights(rows):
    """Normalize validation agreement to complete/high-quality within task.

    These are relative positive-label reliability weights, not loss-task
    coefficients. Values are capped at one and should be aggregated with max
    when multiple transcripts support the same genomic site.
    """
    strata = [
        "complete_high_quality",
        "complete_low_quality",
        "protein_only_high_quality",
        "protein_only_low_quality",
        "transcript_only",
    ]
    lookup = {
        (row["evidence_group"], row["task"]): row
        for row in rows if row["scope"] == "validation"
    }
    output = []
    for stratum in strata:
        for task in SITE_TYPES:
            row = lookup[stratum, task]
            anchor = lookup["complete_high_quality", task]["agreement_precision"]
            relative = min(1.0, row["agreement_precision"] / anchor)
            output.append({
                "evidence_stratum": stratum,
                "task": task,
                "validation_unique_sites": row["eviann_unique_sites"],
                "validation_refseq_matches": row["matched_sites"],
                "validation_agreement_precision": row["agreement_precision"],
                "relative_positive_weight": relative,
            })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eviann-gff", type=Path, default=EVIANN_GFF)
    parser.add_argument("--refseq", type=Path, default=REFSEQ_GFF)
    parser.add_argument("--fasta", type=Path, default=FASTA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    print("Parsing EviAnn evidence strata...", flush=True)
    tx_meta, exons, cds, evidence, low_quality = parse_eviann_evidence(
        args.eviann_gff
    )
    print("Parsing and extracting RefSeq truth...", flush=True)
    r_meta, r_exons, r_cds = parse_reference(args.refseq)
    reference, reference_rejected = extract_sites(
        r_meta, r_exons, r_cds, args.fasta
    )

    evidence_counts = Counter(evidence.values())
    groups = {
        "all_eviann": set(tx_meta),
        "complete": {tid for tid, value in evidence.items() if value == "complete"},
        "protein_only": {
            tid for tid, value in evidence.items() if value == "protein_only"
        },
        "transcript_only": {
            tid for tid, value in evidence.items() if value == "transcript_only"
        },
        "LOW_QUALITY_PROTEIN": {tid for tid, value in low_quality.items() if value},
        "complete_high_quality": {
            tid for tid, value in evidence.items()
            if value == "complete" and not low_quality[tid]
        },
        "complete_low_quality": {
            tid for tid, value in evidence.items()
            if value == "complete" and low_quality[tid]
        },
        "protein_only_high_quality": {
            tid for tid, value in evidence.items()
            if value == "protein_only" and not low_quality[tid]
        },
        "protein_only_low_quality": {
            tid for tid, value in evidence.items()
            if value == "protein_only" and low_quality[tid]
        },
    }
    scopes = {
        "validation": VALIDATION_CHROMS,
        "chr1": {"chr1"},
        "all_primary": set(CHR_TO_NC),
    }
    rows = []
    group_metadata = {}
    for group, tids in groups.items():
        print(f"Extracting {group}: {len(tids):,} transcripts...", flush=True)
        meta, group_exons, group_cds = subset_annotation(
            tids, tx_meta, exons, cds
        )
        sites, rejected = extract_sites(meta, group_exons, group_cds, args.fasta)
        rows.extend(summarize(group, sites, reference, scopes))
        group_metadata[group] = {
            "transcripts": len(tids),
            "transcripts_with_cds": sum(tid in cds for tid in tids),
            "rejected_noncanonical_or_missing_motifs": rejected,
        }

    write_tsv(output / "site_agreement.tsv", rows)
    print("Assigning mutually exclusive site-level evidence strata...", flush=True)
    exclusive, exclusive_hierarchy = exclusive_site_support(
        tx_meta, exons, cds, evidence, low_quality, args.fasta
    )
    exclusive_rows = []
    for stratum in exclusive_hierarchy:
        exclusive_rows.extend(
            summarize(stratum, exclusive[stratum], reference, scopes)
        )
    write_tsv(output / "exclusive_site_support_agreement.tsv", exclusive_rows)
    weight_rows = recommended_weights(exclusive_rows)
    write_tsv(output / "recommended_positive_weights.tsv", weight_rows)
    plot_validation(rows, output / "validation_site_agreement.png")
    metadata = {
        "definition": (
            "Exact canonical CDS-site match by chromosome, strand, and 0-based "
            "motif-start position. Precision is EviAnn sites found in RefSeq."
        ),
        "weight_selection_scope": "validation chromosomes only",
        "validation_chromosomes": sorted(VALIDATION_CHROMS),
        "low_quality_note": (
            "LOW_QUALITY_PROTEIN is a flag-defined subset that overlaps complete "
            "and protein_only; mutually exclusive high/low strata are also reported."
        ),
        "recommended_weight_definition": (
            "Validation agreement divided by complete_high_quality agreement "
            "within each task, capped at 1. These are positive-label reliability "
            "weights derived after assigning each site to its strongest supporting "
            "transcript stratum. They do not address unlabeled false negatives."
        ),
        "exclusive_site_hierarchy": exclusive_hierarchy,
        "eviann_evidence_transcript_counts": dict(evidence_counts),
        "groups": group_metadata,
        "refseq_rejected_noncanonical_or_missing_motifs": reference_rejected,
        "inputs": {
            "eviann": str(args.eviann_gff.resolve()),
            "refseq": str(args.refseq.resolve()),
            "fasta": str(args.fasta.resolve()),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
