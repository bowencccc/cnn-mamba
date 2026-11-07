#!/usr/bin/env python3
"""Audit Human EviAnn start/stop disagreements with RefSeq and MANE Select.

Coordinates are 0-based genomic motif starts.  A same-locus comparison maps
each EviAnn site through its supporting transcript(s) to the same-strand RefSeq
gene with the largest exon overlap (span overlap breaks ties).  In addition to
genomic distance, a spliced distance is reported when the EviAnn codon lies in
an exon of a RefSeq transcript supporting the selected reference boundary.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pysam

from cnn_mamba.human_annotations import (
    CHR_TO_NC,
    EVIANN_GFF,
    FASTA,
    NC_TO_CHR,
    REFSEQ_GFF,
    open_text,
    parse_gff_attrs,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/human_eviann_start_stop_mismatch_v1"
VALIDATION_CHROMS = {"chr2", "chr5", "chr9", "chr10", "chr22"}
PRIMARY_CHROMS = set(CHR_TO_NC)
TASKS = ("start", "stop")
EVIDENCE_HIERARCHY = (
    "complete_high_quality",
    "complete_low_quality",
    "protein_only_high_quality",
    "protein_only_low_quality",
    "transcript_only",
)
EVIDENCE_RANK = {name: rank for rank, name in enumerate(EVIDENCE_HIERARCHY)}
DISTANCE_BUCKETS = (
    (1, 3, "1-3"),
    (4, 10, "4-10"),
    (11, 30, "11-30"),
    (31, 100, "31-100"),
    (101, 300, "101-300"),
    (301, 1000, "301-1k"),
    (1001, 10000, "1k-10k"),
    (10001, math.inf, ">10k"),
)


@dataclass
class Transcript:
    tid: str
    chrom: str
    strand: str
    start: int
    end: int
    gene: str
    evidence: str = ""
    low_quality: bool = False
    mane_select: bool = False
    exons: list[tuple[int, int]] = field(default_factory=list)
    cds: list[tuple[int, int]] = field(default_factory=list)
    sites: dict[str, int] = field(default_factory=dict)


@dataclass
class GeneLocus:
    gid: str
    chrom: str
    strand: str
    start: int
    end: int
    exons: list[tuple[int, int]] = field(default_factory=list)


def stratum(tx: Transcript) -> str:
    if tx.evidence == "complete":
        return "complete_low_quality" if tx.low_quality else "complete_high_quality"
    if tx.evidence == "protein_only":
        return "protein_only_low_quality" if tx.low_quality else "protein_only_high_quality"
    return "transcript_only"


def merge_intervals(values: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(set(values)):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    total = 0
    i = j = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def parse_eviann(path: Path) -> tuple[dict[str, Transcript], dict[str, GeneLocus]]:
    transcripts: dict[str, Transcript] = {}
    genes: dict[str, GeneLocus] = {}
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] not in NC_TO_CHR:
                continue
            chrom = NC_TO_CHR[fields[0]]
            feature = fields[2]
            strand = fields[6]
            if strand not in {"+", "-"}:
                continue
            start, end = int(fields[3]) - 1, int(fields[4])
            attrs = parse_gff_attrs(fields[8])
            if feature == "gene":
                gid = attrs.get("ID")
                if gid:
                    genes[gid] = GeneLocus(gid, chrom, strand, start, end)
            elif feature == "mRNA":
                tid = attrs.get("ID")
                if not tid:
                    continue
                gene = attrs.get("Parent") or attrs.get("geneID") or tid
                transcripts[tid] = Transcript(
                    tid=tid,
                    chrom=chrom,
                    strand=strand,
                    start=start,
                    end=end,
                    gene=gene,
                    evidence=attrs.get("Evidence", "missing"),
                    low_quality="LOW_QUALITY_PROTEIN" in attrs.get("EvidenceProteinID", ""),
                )
            elif feature in {"exon", "CDS"}:
                for tid in attrs.get("Parent", "").split(","):
                    tx = transcripts.get(tid)
                    if tx is not None:
                        (tx.exons if feature == "exon" else tx.cds).append((start, end))
    return transcripts, genes


def parse_refseq(path: Path) -> tuple[dict[str, Transcript], dict[str, GeneLocus]]:
    transcripts: dict[str, Transcript] = {}
    genes: dict[str, GeneLocus] = {}
    pending_exons: dict[str, list[tuple[int, int]]] = defaultdict(list)
    pending_cds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] not in NC_TO_CHR:
                continue
            chrom = NC_TO_CHR[fields[0]]
            feature = fields[2]
            strand = fields[6]
            if strand not in {"+", "-"}:
                continue
            start, end = int(fields[3]) - 1, int(fields[4])
            attrs = parse_gff_attrs(fields[8])
            if feature in {"gene", "pseudogene"}:
                gid = attrs.get("ID")
                if gid:
                    genes[gid] = GeneLocus(gid, chrom, strand, start, end)
            elif feature in {
                "mRNA", "transcript", "V_gene_segment", "D_gene_segment",
                "J_gene_segment", "C_gene_segment",
            }:
                tid = attrs.get("ID")
                if not tid:
                    continue
                gene = attrs.get("Parent") or tid
                tx = Transcript(
                    tid=tid,
                    chrom=chrom,
                    strand=strand,
                    start=start,
                    end=end,
                    gene=gene,
                    mane_select="MANE Select" in attrs.get("tag", ""),
                )
                tx.exons.extend(pending_exons.pop(tid, []))
                tx.cds.extend(pending_cds.pop(tid, []))
                transcripts[tid] = tx
            elif feature in {"exon", "CDS"}:
                for tid in attrs.get("Parent", "").split(","):
                    if not tid:
                        continue
                    tx = transcripts.get(tid)
                    target = pending_exons if feature == "exon" else pending_cds
                    if tx is None:
                        target[tid].append((start, end))
                    else:
                        (tx.exons if feature == "exon" else tx.cds).append((start, end))
                        if "MANE Select" in attrs.get("tag", ""):
                            tx.mane_select = True
    return transcripts, genes


def assign_sites(transcripts: dict[str, Transcript], fasta_path: Path) -> Counter:
    rejected: Counter = Counter()
    fasta = pysam.FastaFile(str(fasta_path))
    for tx in transcripts.values():
        if not tx.cds:
            continue
        segments = sorted(set(tx.cds))
        start = segments[0][0] if tx.strand == "+" else segments[-1][1] - 3
        stop = segments[-1][1] - 3 if tx.strand == "+" else segments[0][0]
        accession = CHR_TO_NC[tx.chrom]
        for task, position in (("start", start), ("stop", stop)):
            motif = fasta.fetch(accession, position, position + 3).upper()
            valid = (
                motif == "ATG" if task == "start" and tx.strand == "+" else
                motif == "CAT" if task == "start" else
                motif in {"TAA", "TAG", "TGA"} if tx.strand == "+" else
                motif in {"TTA", "CTA", "TCA"}
            )
            if valid:
                tx.sites[task] = position
            else:
                rejected[task] += 1
    fasta.close()
    return rejected


def finish_refseq_loci(
    transcripts: dict[str, Transcript], genes: dict[str, GeneLocus]
) -> dict[str, GeneLocus]:
    coding_genes: dict[str, GeneLocus] = {}
    grouped_exons: dict[str, list[tuple[int, int]]] = defaultdict(list)
    grouped_txs: dict[str, list[Transcript]] = defaultdict(list)
    for tx in transcripts.values():
        if tx.cds and tx.sites:
            grouped_txs[tx.gene].append(tx)
            grouped_exons[tx.gene].extend(tx.exons or tx.cds)
    for gid, txs in grouped_txs.items():
        locus = genes.get(gid)
        if locus is None:
            locus = GeneLocus(
                gid, txs[0].chrom, txs[0].strand,
                min(tx.start for tx in txs), max(tx.end for tx in txs),
            )
        locus.exons = merge_intervals(grouped_exons[gid])
        coding_genes[gid] = locus
    return coding_genes


def build_bin_index(genes: dict[str, GeneLocus], bin_size: int = 1_000_000):
    index: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for gid, locus in genes.items():
        for bin_id in range(locus.start // bin_size, (max(locus.start, locus.end - 1) // bin_size) + 1):
            index[(locus.chrom, locus.strand, bin_id)].append(gid)
    return index


def overlapping_genes(tx: Transcript, genes, index, bin_size=1_000_000) -> set[str]:
    found: set[str] = set()
    for bin_id in range(tx.start // bin_size, (max(tx.start, tx.end - 1) // bin_size) + 1):
        for gid in index.get((tx.chrom, tx.strand, bin_id), []):
            locus = genes[gid]
            if locus.start < tx.end and tx.start < locus.end:
                found.add(gid)
    return found


def transcript_coordinate(tx: Transcript, motif_start: int) -> int | None:
    anchor = motif_start if tx.strand == "+" else motif_start + 2
    ordered = sorted(set(tx.exons), reverse=(tx.strand == "-"))
    offset = 0
    for start, end in ordered:
        if start <= motif_start and motif_start + 3 <= end:
            return offset + (anchor - start if tx.strand == "+" else end - 1 - anchor)
        offset += end - start
    return None


def distance_bucket(value: int | None) -> str:
    if value is None:
        return "no_same_locus"
    for low, high, label in DISTANCE_BUCKETS:
        if low <= value <= high:
            return label
    raise AssertionError(value)


def nearest(sorted_values: list[int], value: int) -> int | None:
    if not sorted_values:
        return None
    index = bisect.bisect_left(sorted_values, value)
    candidates = []
    if index < len(sorted_values):
        candidates.append(sorted_values[index])
    if index:
        candidates.append(sorted_values[index - 1])
    return min(candidates, key=lambda candidate: (abs(candidate - value), candidate))


def percentile(values: list[int], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def write_tsv(path: Path, rows: list[dict], gzip_output: bool = False):
    opener = gzip.open if gzip_output else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eviann", type=Path, default=EVIANN_GFF)
    parser.add_argument("--refseq", type=Path, default=REFSEQ_GFF)
    parser.add_argument("--fasta", type=Path, default=FASTA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("Parsing EviAnn...", flush=True)
    eviann_tx, _ = parse_eviann(args.eviann)
    print("Parsing RefSeq/MANE...", flush=True)
    ref_tx, ref_genes_raw = parse_refseq(args.refseq)
    print("Extracting canonical CDS boundaries...", flush=True)
    eviann_rejected = assign_sites(eviann_tx, args.fasta)
    ref_rejected = assign_sites(ref_tx, args.fasta)
    ref_genes = finish_refseq_loci(ref_tx, ref_genes_raw)
    locus_index = build_bin_index(ref_genes)

    eviann_support: dict[str, dict[tuple[str, str, int], set[str]]] = {
        task: defaultdict(set) for task in TASKS
    }
    ref_support: dict[str, dict[tuple[str, str, int], set[str]]] = {
        task: defaultdict(set) for task in TASKS
    }
    mane_support: dict[str, dict[tuple[str, str, int], set[str]]] = {
        task: defaultdict(set) for task in TASKS
    }
    gene_sites: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    gene_mane_sites: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))

    for tid, tx in eviann_tx.items():
        for task, position in tx.sites.items():
            eviann_support[task][(tx.chrom, tx.strand, position)].add(tid)
    for tid, tx in ref_tx.items():
        for task, position in tx.sites.items():
            key = (tx.chrom, tx.strand, position)
            ref_support[task][key].add(tid)
            gene_sites[tx.gene][task].add(position)
            if tx.mane_select:
                mane_support[task][key].add(tid)
                gene_mane_sites[tx.gene][task].add(position)

    ref_sets = {task: set(ref_support[task]) for task in TASKS}
    mane_sets = {task: set(mane_support[task]) for task in TASKS}
    genome_ref_positions: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for task in TASKS:
        for chrom, strand, position in ref_sets[task]:
            genome_ref_positions[(task, chrom, strand)].append(position)
    for values in genome_ref_positions.values():
        values.sort()

    print("Mapping EviAnn sites to same-strand RefSeq loci...", flush=True)
    all_rows: list[dict] = []
    mismatch_rows: list[dict] = []
    for task in TASKS:
        for site in sorted(eviann_support[task]):
            chrom, strand, position = site
            tids = sorted(eviann_support[task][site])
            strata = sorted({stratum(eviann_tx[tid]) for tid in tids}, key=EVIDENCE_RANK.get)
            strongest = strata[0]
            exact_refseq = site in ref_sets[task]
            exact_mane = site in mane_sets[task]

            gene_scores: dict[str, tuple[int, int]] = {}
            for tid in tids:
                tx = eviann_tx[tid]
                tx_exons = merge_intervals(tx.exons or tx.cds)
                for gid in overlapping_genes(tx, ref_genes, locus_index):
                    locus = ref_genes[gid]
                    exon_bp = interval_overlap(tx_exons, locus.exons)
                    span_bp = min(tx.end, locus.end) - max(tx.start, locus.start)
                    gene_scores[gid] = max(gene_scores.get(gid, (-1, -1)), (exon_bp, span_bp))
            best_gene = max(gene_scores, key=lambda gid: (*gene_scores[gid], gid)) if gene_scores else None

            nearest_ref = None
            nearest_mane = None
            if best_gene is not None:
                nearest_ref = nearest(sorted(gene_sites[best_gene][task]), position)
                nearest_mane = nearest(sorted(gene_mane_sites[best_gene][task]), position)

            abs_ref = abs(nearest_ref - position) if nearest_ref is not None else None
            signed_ref = (
                (nearest_ref - position) * (1 if strand == "+" else -1)
                if nearest_ref is not None else None
            )
            abs_mane = abs(nearest_mane - position) if nearest_mane is not None else None
            signed_mane = (
                (nearest_mane - position) * (1 if strand == "+" else -1)
                if nearest_mane is not None else None
            )
            genome_nearest = nearest(genome_ref_positions[(task, chrom, strand)], position)
            genome_abs = abs(genome_nearest - position) if genome_nearest is not None else None

            spliced_distances: list[int] = []
            same_ref_exon = False
            if best_gene is not None and nearest_ref is not None:
                boundary_key = (chrom, strand, nearest_ref)
                for ref_tid in ref_support[task].get(boundary_key, set()):
                    rtx = ref_tx[ref_tid]
                    if rtx.gene != best_gene:
                        continue
                    ref_coord = transcript_coordinate(rtx, nearest_ref)
                    eviann_coord = transcript_coordinate(rtx, position)
                    if ref_coord is not None and eviann_coord is not None:
                        spliced_distances.append(ref_coord - eviann_coord)
                    for exon_start, exon_end in rtx.exons:
                        if (
                            exon_start <= position and position + 3 <= exon_end
                            and exon_start <= nearest_ref and nearest_ref + 3 <= exon_end
                        ):
                            same_ref_exon = True
                            break
            spliced = (
                min(spliced_distances, key=lambda value: (abs(value), value))
                if spliced_distances else None
            )

            if exact_mane:
                classification = "exact_MANE"
            elif exact_refseq:
                classification = "exact_nonMANE_RefSeq"
            elif abs_ref is None:
                classification = "mismatch_no_same_locus"
            elif abs_ref <= 30:
                classification = "mismatch_near_1_30nt"
            elif abs_ref <= 300:
                classification = "mismatch_intermediate_31_300nt"
            else:
                classification = "mismatch_distant_gt300nt"

            row = {
                "task": task,
                "chrom": chrom,
                "strand": strand,
                "eviann_position0": position,
                "eviann_position1": position + 1,
                "eviann_supporting_transcripts": len(tids),
                "eviann_transcript_ids": ",".join(tids),
                "evidence_strata": ",".join(strata),
                "strongest_evidence_stratum": strongest,
                "exact_refseq": int(exact_refseq),
                "exact_mane_select": int(exact_mane),
                "classification": classification,
                "mapped_refseq_gene": best_gene or "",
                "locus_exon_overlap_bp": gene_scores[best_gene][0] if best_gene else "",
                "locus_span_overlap_bp": gene_scores[best_gene][1] if best_gene else "",
                "nearest_refseq_position0": nearest_ref if nearest_ref is not None else "",
                "signed_genomic_distance_nt": signed_ref if signed_ref is not None else "",
                "absolute_genomic_distance_nt": abs_ref if abs_ref is not None else "",
                "genomic_distance_multiple_of_3": int(abs_ref % 3 == 0) if abs_ref else "",
                "same_refseq_exon": int(same_ref_exon),
                "signed_spliced_distance_nt": spliced if spliced is not None else "",
                "absolute_spliced_distance_nt": abs(spliced) if spliced is not None else "",
                "spliced_distance_multiple_of_3": int(abs(spliced) % 3 == 0) if spliced else "",
                "nearest_mane_position0": nearest_mane if nearest_mane is not None else "",
                "signed_mane_genomic_distance_nt": signed_mane if signed_mane is not None else "",
                "absolute_mane_genomic_distance_nt": abs_mane if abs_mane is not None else "",
                "nearest_same_strand_genome_refseq_position0": genome_nearest if genome_nearest is not None else "",
                "nearest_same_strand_genome_refseq_distance_nt": genome_abs if genome_abs is not None else "",
            }
            all_rows.append(row)
            if not exact_refseq:
                mismatch_rows.append(row)

    write_tsv(args.output / "all_eviann_start_stop_sites.tsv.gz", all_rows, gzip_output=True)
    write_tsv(args.output / "mismatched_start_stop_sites.tsv.gz", mismatch_rows, gzip_output=True)

    scopes = {
        "all_primary": PRIMARY_CHROMS,
        "validation_chr2_5_9_10_22": VALIDATION_CHROMS,
        "chr1": {"chr1"},
    }
    summary_rows: list[dict] = []
    for scope, chromosomes in scopes.items():
        for task in TASKS:
            selected = [row for row in all_rows if row["task"] == task and row["chrom"] in chromosomes]
            mismatches = [row for row in selected if not row["exact_refseq"]]
            mapped = [row for row in mismatches if row["absolute_genomic_distance_nt"] != ""]
            genomic = [int(row["absolute_genomic_distance_nt"]) for row in mapped]
            spliced = [int(row["absolute_spliced_distance_nt"]) for row in mapped if row["absolute_spliced_distance_nt"] != ""]
            summary_rows.append({
                "scope": scope,
                "task": task,
                "eviann_sites": len(selected),
                "exact_refseq_sites": sum(int(row["exact_refseq"]) for row in selected),
                "exact_refseq_fraction": safe_ratio(sum(int(row["exact_refseq"]) for row in selected), len(selected)),
                "exact_mane_sites": sum(int(row["exact_mane_select"]) for row in selected),
                "exact_mane_fraction": safe_ratio(sum(int(row["exact_mane_select"]) for row in selected), len(selected)),
                "exact_nonmane_refseq_sites": sum(row["classification"] == "exact_nonMANE_RefSeq" for row in selected),
                "mismatched_sites": len(mismatches),
                "mismatch_same_locus_sites": len(mapped),
                "mismatch_same_locus_fraction": safe_ratio(len(mapped), len(mismatches)),
                "mismatch_no_same_locus_sites": len(mismatches) - len(mapped),
                "same_locus_within_30nt": sum(value <= 30 for value in genomic),
                "same_locus_within_30nt_fraction": safe_ratio(sum(value <= 30 for value in genomic), len(mapped)),
                "same_locus_within_100nt": sum(value <= 100 for value in genomic),
                "same_locus_within_100nt_fraction": safe_ratio(sum(value <= 100 for value in genomic), len(mapped)),
                "same_locus_within_300nt": sum(value <= 300 for value in genomic),
                "same_locus_within_300nt_fraction": safe_ratio(sum(value <= 300 for value in genomic), len(mapped)),
                "same_locus_genomic_multiple3": sum(value % 3 == 0 for value in genomic),
                "same_locus_genomic_multiple3_fraction": safe_ratio(sum(value % 3 == 0 for value in genomic), len(mapped)),
                "spliced_evaluable_sites": len(spliced),
                "spliced_multiple3_sites": sum(value % 3 == 0 for value in spliced),
                "spliced_multiple3_fraction": safe_ratio(sum(value % 3 == 0 for value in spliced), len(spliced)),
                "same_locus_median_distance_nt": percentile(genomic, 50),
                "same_locus_p90_distance_nt": percentile(genomic, 90),
            })
    write_tsv(args.output / "scope_summary.tsv", summary_rows)

    bucket_rows: list[dict] = []
    for task in TASKS:
        selected = [row for row in mismatch_rows if row["task"] == task]
        counts = Counter(
            distance_bucket(
                int(row["absolute_genomic_distance_nt"])
                if row["absolute_genomic_distance_nt"] != "" else None
            )
            for row in selected
        )
        order = [label for _, _, label in DISTANCE_BUCKETS] + ["no_same_locus"]
        for bucket in order:
            bucket_rows.append({
                "task": task,
                "distance_bucket": bucket,
                "sites": counts[bucket],
                "fraction_of_mismatches": safe_ratio(counts[bucket], len(selected)),
            })
    write_tsv(args.output / "distance_bucket_summary.tsv", bucket_rows)

    evidence_rows: list[dict] = []
    for evidence in EVIDENCE_HIERARCHY:
        for task in TASKS:
            selected = [row for row in all_rows if row["task"] == task and row["strongest_evidence_stratum"] == evidence]
            mismatch = [row for row in selected if not row["exact_refseq"]]
            mapped = [row for row in mismatch if row["absolute_genomic_distance_nt"] != ""]
            evidence_rows.append({
                "evidence_stratum": evidence,
                "task": task,
                "eviann_sites": len(selected),
                "exact_refseq_sites": sum(int(row["exact_refseq"]) for row in selected),
                "exact_refseq_fraction": safe_ratio(sum(int(row["exact_refseq"]) for row in selected), len(selected)),
                "exact_mane_sites": sum(int(row["exact_mane_select"]) for row in selected),
                "exact_mane_fraction": safe_ratio(sum(int(row["exact_mane_select"]) for row in selected), len(selected)),
                "mismatched_sites": len(mismatch),
                "same_locus_mismatches": len(mapped),
                "same_locus_within_300nt": sum(int(row["absolute_genomic_distance_nt"]) <= 300 for row in mapped),
            })
    write_tsv(args.output / "evidence_stratum_summary.tsv", evidence_rows)

    plot_results(all_rows, mismatch_rows, bucket_rows, args.output)

    metadata = {
        "coordinate_definition": "0-based left genomic coordinate of canonical start/stop motif",
        "signed_distance_definition": "(RefSeq - EviAnn genomic position) * strand; positive is downstream in transcription direction",
        "same_locus_definition": "same strand and overlapping transcript/gene span; choose RefSeq coding gene with maximum exon overlap, breaking ties by span overlap",
        "spliced_distance_definition": "distance on a RefSeq transcript supporting the nearest boundary, only when the EviAnn codon is fully inside one of that transcript's exons",
        "modulo3_warning": "genomic modulo 3 is not a frame test across introns; use spliced_distance_multiple_of_3 when available",
        "mane_definition": "RefSeq transcript carrying tag=MANE Select in the same GRCh38.p14 GFF",
        "inputs": {"eviann": str(args.eviann), "refseq": str(args.refseq), "fasta": str(args.fasta)},
        "rejected_noncanonical": {"eviann": dict(eviann_rejected), "refseq": dict(ref_rejected)},
        "counts": {"eviann_transcripts": len(eviann_tx), "refseq_transcripts": len(ref_tx), "refseq_coding_loci": len(ref_genes)},
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    make_readme(summary_rows, evidence_rows, args.output)
    print(f"Wrote {args.output}", flush=True)


def plot_results(all_rows, mismatch_rows, bucket_rows, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2))
    colors = {"start": "#4C78A8", "stop": "#E45756"}
    for axis, task in zip(axes[0], TASKS):
        values = np.array([
            int(row["absolute_genomic_distance_nt"])
            for row in mismatch_rows
            if row["task"] == task and row["absolute_genomic_distance_nt"] != ""
        ])
        if len(values):
            ordered = np.sort(values)
            y = np.arange(1, len(ordered) + 1) / len(ordered)
            axis.step(np.maximum(ordered, 1), y, where="post", color=colors[task], linewidth=2)
        axis.set_xscale("log")
        axis.set_xlim(1, 10_000_000)
        axis.set_ylim(0, 1.01)
        axis.set_title(f"{task.capitalize()} mismatches: same-locus distance CDF")
        axis.set_xlabel("Absolute genomic distance to nearest RefSeq boundary (nt)")
        axis.set_ylabel("Cumulative fraction")
        axis.grid(alpha=0.25)

    axis = axes[1, 0]
    order = [label for _, _, label in DISTANCE_BUCKETS] + ["no_same_locus"]
    x = np.arange(len(order))
    width = 0.38
    for offset, task in ((-width / 2, "start"), (width / 2, "stop")):
        lookup = {row["distance_bucket"]: row for row in bucket_rows if row["task"] == task}
        values = [100 * float(lookup[label]["fraction_of_mismatches"] or 0) for label in order]
        axis.bar(x + offset, values, width, label=task.capitalize(), color=colors[task])
    axis.set_xticks(x, order, rotation=35, ha="right")
    axis.set_ylabel("Mismatched EviAnn sites (%)")
    axis.set_title("Distance categories")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)

    axis = axes[1, 1]
    classes = (
        "exact_MANE", "exact_nonMANE_RefSeq", "mismatch_near_1_30nt",
        "mismatch_intermediate_31_300nt", "mismatch_distant_gt300nt",
        "mismatch_no_same_locus",
    )
    labels = ("Exact MANE", "Exact other RefSeq", "Mismatch 1-30", "Mismatch 31-300", "Mismatch >300", "No same locus")
    x = np.arange(len(classes))
    for offset, task in ((-width / 2, "start"), (width / 2, "stop")):
        selected = [row for row in all_rows if row["task"] == task]
        counts = Counter(row["classification"] for row in selected)
        values = [100 * counts[name] / len(selected) for name in classes]
        axis.bar(x + offset, values, width, label=task.capitalize(), color=colors[task])
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylabel("All EviAnn sites (%)")
    axis.set_title("RefSeq/MANE agreement classes")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)

    for axis in axes.flat:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "human_eviann_start_stop_mismatch_distances.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "human_eviann_start_stop_mismatch_distances.pdf", bbox_inches="tight")
    plt.close(fig)


def make_readme(summary_rows, evidence_rows, output: Path) -> None:
    all_rows = {row["task"]: row for row in summary_rows if row["scope"] == "all_primary"}
    lines = [
        "# Human EviAnn start/stop mismatch audit",
        "",
        "All comparisons use exact canonical motif-start coordinates on GRCh38.p14. Same-locus mapping is strand-aware and based on maximum exon overlap with a RefSeq coding gene.",
        "",
        "| Task | EviAnn sites | Exact any RefSeq | Exact MANE Select | Mismatches mapped to same locus | Within 30 nt (mapped) | Within 300 nt (mapped) | Spliced distance divisible by 3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        row = all_rows[task]
        lines.append(
            f"| {task} | {row['eviann_sites']:,} | {100 * row['exact_refseq_fraction']:.1f}% | "
            f"{100 * row['exact_mane_fraction']:.1f}% | {100 * row['mismatch_same_locus_fraction']:.1f}% | "
            f"{100 * row['same_locus_within_30nt_fraction']:.1f}% | {100 * row['same_locus_within_300nt_fraction']:.1f}% | "
            f"{100 * row['spliced_multiple3_fraction']:.1f}% ({row['spliced_evaluable_sites']:,} evaluable) |"
        )
    lines.extend([
        "",
        "`Exact MANE Select` is a deliberately narrower target than exact agreement with any RefSeq isoform. A site matching another RefSeq isoform but not MANE is evidence of isoform choice rather than an incorrect coordinate.",
        "",
        "Genomic distance modulo 3 must not be treated as a reading-frame test when an intron lies between sites. The report therefore also supplies spliced-transcript distance where it can be computed on a supporting RefSeq transcript.",
        "",
        "Files:",
        "",
        "- `all_eviann_start_stop_sites.tsv.gz`: every unique EviAnn start/stop site and its RefSeq/MANE classification.",
        "- `mismatched_start_stop_sites.tsv.gz`: exact-RefSeq mismatches with nearest-boundary distances.",
        "- `scope_summary.tsv`: all-primary, validation-chromosome and chr1 summaries.",
        "- `distance_bucket_summary.tsv`: mismatch distance bins.",
        "- `evidence_stratum_summary.tsv`: complete/protein-only/high/low-quality stratification.",
        "- `human_eviann_start_stop_mismatch_distances.png`: report figure.",
        "- `metadata.json`: definitions, inputs and parsing counts.",
    ])
    (output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
