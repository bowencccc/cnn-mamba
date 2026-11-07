#!/usr/bin/env python3
"""Build D. melanogaster sites with the Human LightGBM feature schema."""

from collections import Counter, defaultdict
import math

import numpy as np
import pysam

from .human_annotations import open_text, parse_gtf_attrs
from .human_orf_qc import evaluate_eviann_orfs, reverse_complement
from .human_site_quality import CodingSpanIndex, feature_names


DMEL_PRIMARY = (
    "NT_033779.5",  # 2L
    "NT_033778.4",  # 2R
    "NT_037436.4",  # 3L
    "NT_033777.3",  # 3R
    "NC_004353.4",  # 4
    "NC_004354.4",  # X
    "NC_024512.1",  # Y
)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _fraction(results, name):
    return _mean([float(bool(getattr(result, name))) for result in results])


def _context(fasta, chrom, strand, position, left=20, right=23):
    chrom_length = fasta.get_reference_length(chrom)
    start = max(0, position - left)
    end = min(chrom_length, position + right)
    sequence = fasta.fetch(chrom, start, end).upper()
    sequence = (
        "N" * max(0, left - position) + sequence
        + "N" * max(0, position + right - chrom_length)
    )
    if strand == "-":
        sequence = reverse_complement(sequence)
    if len(sequence) != left + right:
        raise ValueError(f"bad context length at {chrom}:{position}:{strand}")
    return sequence


def _gc_fraction(fasta, chrom, position, radius):
    chrom_length = fasta.get_reference_length(chrom)
    sequence = fasta.fetch(
        chrom, max(0, position - radius), min(chrom_length, position + radius + 3)
    ).upper()
    informative = sum(sequence.count(base) for base in "ACGT")
    return (sequence.count("G") + sequence.count("C")) / max(informative, 1)


def parse_dmel_reference(path, chromosomes=DMEL_PRIMARY):
    """Read coding spans and explicit GTF start/stop codon records."""
    chromosomes = set(chromosomes)
    tx_meta = {}
    cds = defaultdict(list)
    sites = {"start": set(), "stop": set()}
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] not in chromosomes:
                continue
            feature = fields[2]
            if feature not in {"transcript", "mRNA", "CDS", "start_codon", "stop_codon"}:
                continue
            attrs = parse_gtf_attrs(fields[8])
            tid = attrs.get("transcript_id")
            if not tid or fields[6] not in {"+", "-"}:
                continue
            chrom, strand = fields[0], fields[6]
            tx_meta[tid] = (chrom, strand)
            start0, end0 = int(fields[3]) - 1, int(fields[4])
            if feature == "CDS":
                cds[tid].append((start0, end0))
            elif feature in {"start_codon", "stop_codon"}:
                sites["start" if feature == "start_codon" else "stop"].add(
                    (chrom, strand, start0)
                )
    return tx_meta, cds, sites


def build_dmel_site_quality_examples(eviann_gff, reference_gtf, fasta_path):
    identity = {chrom: chrom for chrom in DMEL_PRIMARY}
    records, qc = evaluate_eviann_orfs(
        eviann_gff, fasta_path,
        accession_to_chrom=identity, chrom_to_accession=identity,
    )
    reference_meta, reference_cds, reference_sites = parse_dmel_reference(reference_gtf)
    span_index = CodingSpanIndex(reference_meta, reference_cds)
    gene_isoforms = Counter(
        record.gene_id for record in records.values() if record.gene_id
    )
    support = {"start": defaultdict(list), "stop": defaultdict(list)}
    for tid, result in qc.items():
        if result.start_position >= 0 and result.start_atg:
            support["start"][(result.chrom, result.strand, result.start_position)].append(tid)
        if result.stop_position >= 0 and result.stop_valid:
            support["stop"][(result.chrom, result.strand, result.stop_position)].append(tid)

    names = feature_names()
    fasta = pysam.FastaFile(str(fasta_path))
    examples = {"start": [], "stop": []}
    chrom_rank = {chrom: index for index, chrom in enumerate(DMEL_PRIMARY)}
    for task in ("start", "stop"):
        for site, tids in support[task].items():
            chrom, strand, position = site
            site_results = [qc[tid] for tid in tids]
            site_records = [records[tid] for tid in tids]
            comparable = any(
                record.cds and span_index.comparable(
                    chrom, strand,
                    min(segment[0] for segment in record.cds),
                    max(segment[1] for segment in record.cds),
                )
                for record in site_records
            )
            exact = site in reference_sites[task]
            label = 1 if exact else 0 if comparable else -1
            context = _context(fasta, chrom, strand, position)
            partner_positions = [
                result.stop_position if task == "start" else result.start_position
                for result in site_results
            ]
            partner_counts = Counter(partner_positions)
            genes = {record.gene_id for record in site_records if record.gene_id}
            total_gene_isoforms = sum(gene_isoforms[gene] for gene in genes)
            codon = context[20:23]
            declared_match = []
            for result in site_results:
                declared = result.declared_start if task == "start" else result.declared_stop
                observed = result.start_codon if task == "start" else result.stop_codon
                declared_match.append(float(bool(declared) and declared == observed))
            lengths = [result.cds_length for result in site_results]
            values = [
                float(strand == "-"), float(len(tids)), float(len(genes)),
                _mean([result.evidence == "complete" for result in site_results]),
                _mean([result.evidence == "protein_only" for result in site_results]),
                _mean([result.evidence == "transcript_only" for result in site_results]),
                _mean([result.low_quality_protein for result in site_results]),
                float(any(result.evidence == "complete" and not result.low_quality_protein for result in site_results)),
                float(any(result.evidence == "protein_only" and not result.low_quality_protein for result in site_results)),
                _fraction(site_results, "orf_valid"),
                _fraction(site_results, "sequence_complete"),
                _fraction(site_results, "start_atg"),
                _fraction(site_results, "stop_valid"),
                _fraction(site_results, "length_multiple_of_three"),
                _fraction(site_results, "no_internal_inframe_stop"),
                _fraction(site_results, "phase_consistent"),
                _fraction(site_results, "splice_recognized"),
                _fraction(site_results, "all_splice_gt_ag"),
                _mean([math.log1p(value) for value in lengths]),
                math.log1p(max(lengths)), math.log1p(min(lengths)),
                _mean([result.exon_count for result in site_results]),
                max(result.exon_count for result in site_results),
                _mean([result.cds_segment_count for result in site_results]),
                _mean([result.coding_intron_count for result in site_results]),
                _mean([result.gt_ag_count for result in site_results]),
                _mean([result.gc_ag_count for result in site_results]),
                _mean([result.at_ac_count for result in site_results]),
                _mean([result.other_splice_count for result in site_results]),
                _mean([gene_isoforms[gene] for gene in genes]),
                max((gene_isoforms[gene] for gene in genes), default=0),
                len(tids) / max(total_gene_isoforms, 1),
                float(len(partner_counts)), max(partner_counts.values()) / len(tids),
                _mean(declared_match),
                sum(context.count(base) for base in "GC")
                / max(sum(context.count(base) for base in "ACGT"), 1),
                _gc_fraction(fasta, chrom, position, 49),
                _gc_fraction(fasta, chrom, position, 499),
                float(context[17] in {"A", "G"}), float(context[23] == "G"),
                float(codon == "TAA"), float(codon == "TAG"), float(codon == "TGA"),
            ]
            for base in context:
                values.extend(float(base == requested) for requested in "ACGT")
            if len(values) != len(names):
                raise RuntimeError(f"feature count {len(values)} != {len(names)}")
            examples[task].append({
                "task": task, "chrom": chrom, "strand": strand,
                "position_0based": position, "label": label,
                "exact_refseq_match": int(exact),
                "comparable_refseq_locus": int(comparable),
                "support_transcript_ids": ",".join(sorted(tids)),
                "features": np.asarray(values, dtype=np.float32),
            })
        examples[task].sort(
            key=lambda row: (chrom_rank[row["chrom"]], row["strand"], row["position_0based"])
        )
    fasta.close()
    return examples, names
