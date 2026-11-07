#!/usr/bin/env python3
"""Transcript-oriented ORF quality control for Human EviAnn annotations."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import pysam

from .human_annotations import CHR_TO_NC, NC_TO_CHR, open_text, parse_gff_attrs


STOP_CODONS = {"TAA", "TAG", "TGA"}
RECOGNIZED_SPLICE_PAIRS = {("GT", "AG"), ("GC", "AG"), ("AT", "AC")}
REVERSE_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(sequence):
    return sequence.translate(REVERSE_COMPLEMENT)[::-1]


@dataclass
class EviAnnTranscript:
    transcript_id: str
    chrom: str
    strand: str
    evidence: str
    low_quality_protein: bool
    gene_id: str
    declared_start: str
    declared_stop: str
    exons: list
    cds: list


@dataclass
class ORFQCResult:
    transcript_id: str
    chrom: str
    strand: str
    evidence: str
    low_quality_protein: bool
    gene_id: str
    exon_count: int
    cds_segment_count: int
    cds_length: int
    start_codon: str
    stop_codon: str
    declared_start: str
    declared_stop: str
    sequence_complete: bool
    start_atg: bool
    stop_valid: bool
    length_multiple_of_three: bool
    no_internal_inframe_stop: bool
    phase_consistent: bool
    splice_recognized: bool
    all_splice_gt_ag: bool
    coding_intron_count: int
    gt_ag_count: int
    gc_ag_count: int
    at_ac_count: int
    other_splice_count: int
    start_position: int
    stop_position: int
    start_support: int = 0
    stop_support: int = 0
    orf_valid: bool = False
    tier: str = "uncertain"


def parse_eviann_orf_records(path, accession_to_chrom=None):
    """Parse primary-chromosome mRNA, exon, and phased CDS records."""
    accession_to_chrom = NC_TO_CHR if accession_to_chrom is None else accession_to_chrom
    records = {}
    exons = defaultdict(list)
    cds = defaultdict(list)
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] not in accession_to_chrom:
                continue
            feature = fields[2]
            if feature not in {"mRNA", "exon", "CDS"}:
                continue
            attrs = parse_gff_attrs(fields[8])
            if feature == "mRNA":
                tid = attrs.get("ID")
                if not tid or fields[6] not in {"+", "-"}:
                    continue
                records[tid] = EviAnnTranscript(
                    transcript_id=tid,
                    chrom=accession_to_chrom[fields[0]],
                    strand=fields[6],
                    evidence=attrs.get("Evidence", "missing"),
                    low_quality_protein=(
                        "LOW_QUALITY_PROTEIN"
                        in attrs.get("EvidenceProteinID", "")
                    ),
                    gene_id=attrs.get("geneID", attrs.get("Parent", "")),
                    declared_start=attrs.get("StartCodon", "").upper(),
                    declared_stop=attrs.get("StopCodon", "").upper(),
                    exons=[],
                    cds=[],
                )
                continue
            start0, end0 = int(fields[3]) - 1, int(fields[4])
            parents = [value for value in attrs.get("Parent", "").split(",") if value]
            if feature == "exon":
                for tid in parents:
                    exons[tid].append((start0, end0))
            else:
                phase = int(fields[7]) if fields[7] in {"0", "1", "2"} else None
                for tid in parents:
                    cds[tid].append((start0, end0, phase))
    for tid, record in records.items():
        record.exons = sorted(set(exons.get(tid, ())))
        record.cds = sorted(set(cds.get(tid, ())))
    return records


def expected_gff_phases(segment_lengths):
    """Return expected GFF3 phases in transcript order.

    GFF3 phase is the number of bases from the transcript-oriented start of a
    CDS feature to the first base of the next codon. It is determined by the
    cumulative length of all preceding CDS segments.
    """
    phases = []
    cumulative = 0
    for length in segment_lengths:
        phases.append((3 - cumulative % 3) % 3)
        cumulative += length
    return phases


def _coding_splice_pairs(fasta, accession, ordered_cds, strand):
    pairs = []
    for left, right in zip(ordered_cds, ordered_cds[1:]):
        if strand == "+":
            donor = fasta.fetch(accession, left[1], left[1] + 2).upper()
            acceptor = fasta.fetch(accession, right[0] - 2, right[0]).upper()
        else:
            # ordered_cds runs from high to low genomic coordinates.
            donor = reverse_complement(
                fasta.fetch(accession, left[0] - 2, left[0]).upper()
            )
            acceptor = reverse_complement(
                fasta.fetch(accession, right[1], right[1] + 2).upper()
            )
        pairs.append((donor, acceptor))
    return pairs


def evaluate_transcript_orf(record, fasta, chrom_to_accession=None):
    chrom_to_accession = CHR_TO_NC if chrom_to_accession is None else chrom_to_accession
    accession = chrom_to_accession[record.chrom]
    ordered = sorted(record.cds, reverse=record.strand == "-")
    sequences = []
    for start, end, _ in ordered:
        sequence = fasta.fetch(accession, start, end).upper()
        sequences.append(
            sequence if record.strand == "+" else reverse_complement(sequence)
        )
    spliced = "".join(sequences)
    start_codon = spliced[:3] if len(spliced) >= 3 else ""
    stop_codon = spliced[-3:] if len(spliced) >= 3 else ""
    internal_stops = {
        spliced[offset:offset + 3]
        for offset in range(3, max(len(spliced) - 3, 3), 3)
        if len(spliced[offset:offset + 3]) == 3
        and spliced[offset:offset + 3] in STOP_CODONS
    }
    lengths = [end - start for start, end, _ in ordered]
    observed_phases = [phase for _, _, phase in ordered]
    phase_consistent = bool(ordered) and observed_phases == expected_gff_phases(lengths)
    splice_pairs = _coding_splice_pairs(fasta, accession, ordered, record.strand)
    pair_counts = Counter(splice_pairs)
    recognized = all(pair in RECOGNIZED_SPLICE_PAIRS for pair in splice_pairs)
    sequence_complete = bool(spliced) and set(spliced) <= set("ACGT")
    start_position = -1
    stop_position = -1
    if ordered:
        if record.strand == "+":
            start_position = ordered[0][0]
            stop_position = ordered[-1][1] - 3
        else:
            start_position = ordered[0][1] - 3
            stop_position = ordered[-1][0]
    checks = {
        "sequence_complete": sequence_complete,
        "start_atg": start_codon == "ATG",
        "stop_valid": stop_codon in STOP_CODONS,
        "length_multiple_of_three": bool(spliced) and len(spliced) % 3 == 0,
        "no_internal_inframe_stop": not internal_stops,
        "phase_consistent": phase_consistent,
        "splice_recognized": recognized,
    }
    orf_valid = all(checks.values())
    eligible = (
        not record.low_quality_protein
        and record.evidence in {"complete", "protein_only"}
    )
    tier = (
        "high" if eligible and record.evidence == "complete" and orf_valid
        else "medium" if eligible and record.evidence == "protein_only" and orf_valid
        else "uncertain"
    )
    return ORFQCResult(
        transcript_id=record.transcript_id,
        chrom=record.chrom,
        strand=record.strand,
        evidence=record.evidence,
        low_quality_protein=record.low_quality_protein,
        gene_id=record.gene_id,
        exon_count=len(record.exons),
        cds_segment_count=len(ordered),
        cds_length=len(spliced),
        start_codon=start_codon,
        stop_codon=stop_codon,
        declared_start=record.declared_start,
        declared_stop=record.declared_stop,
        coding_intron_count=len(splice_pairs),
        gt_ag_count=pair_counts[("GT", "AG")],
        gc_ag_count=pair_counts[("GC", "AG")],
        at_ac_count=pair_counts[("AT", "AC")],
        other_splice_count=sum(
            count for pair, count in pair_counts.items()
            if pair not in RECOGNIZED_SPLICE_PAIRS
        ),
        all_splice_gt_ag=all(pair == ("GT", "AG") for pair in splice_pairs),
        start_position=start_position,
        stop_position=stop_position,
        orf_valid=orf_valid,
        tier=tier,
        **checks,
    )


def evaluate_eviann_orfs(
    gff_path, fasta_path, accession_to_chrom=None, chrom_to_accession=None
):
    records = parse_eviann_orf_records(gff_path, accession_to_chrom)
    fasta = pysam.FastaFile(str(fasta_path))
    results = {
        tid: evaluate_transcript_orf(record, fasta, chrom_to_accession)
        for tid, record in records.items()
    }
    fasta.close()

    accepted = [result for result in results.values() if result.tier in {"high", "medium"}]
    start_support = Counter(
        (result.chrom, result.strand, result.start_position) for result in accepted
    )
    stop_support = Counter(
        (result.chrom, result.strand, result.stop_position) for result in accepted
    )
    for result in results.values():
        result.start_support = start_support[
            (result.chrom, result.strand, result.start_position)
        ]
        result.stop_support = stop_support[
            (result.chrom, result.strand, result.stop_position)
        ]
    return records, results


def summarize_orf_qc(results):
    values = list(results.values())
    checks = (
        "sequence_complete", "start_atg", "stop_valid",
        "length_multiple_of_three", "no_internal_inframe_stop",
        "phase_consistent", "splice_recognized", "orf_valid",
    )
    eligible = [
        result for result in values
        if result.evidence in {"complete", "protein_only"}
        and not result.low_quality_protein
    ]
    remaining = eligible
    funnel = {"metadata_eligible": len(remaining)}
    for check in checks[:-1]:
        remaining = [result for result in remaining if getattr(result, check)]
        funnel[check] = len(remaining)
    return {
        "transcripts": len(values),
        "evidence": dict(Counter(result.evidence for result in values)),
        "low_quality_protein": sum(result.low_quality_protein for result in values),
        "tiers": dict(Counter(result.tier for result in values)),
        "checks_pass_nonexclusive_all": {
            check: sum(bool(getattr(result, check)) for result in values)
            for check in checks
        },
        "checks_pass_nonexclusive_metadata_eligible": {
            check: sum(bool(getattr(result, check)) for result in eligible)
            for check in checks
        },
        "sequential_funnel_metadata_eligible": funnel,
        "recognized_coding_introns": {
            "GT-AG": sum(result.gt_ag_count for result in values),
            "GC-AG": sum(result.gc_ag_count for result in values),
            "AT-AC": sum(result.at_ac_count for result in values),
            "other": sum(result.other_splice_count for result in values),
        },
        "accepted_sites_with_multiple_isoform_support": {
            "start_transcripts": sum(
                result.tier in {"high", "medium"} and result.start_support >= 2
                for result in values
            ),
            "stop_transcripts": sum(
                result.tier in {"high", "medium"} and result.stop_support >= 2
                for result in values
            ),
            "unique_start_sites": len({
                (result.chrom, result.strand, result.start_position)
                for result in values
                if result.tier in {"high", "medium"} and result.start_support >= 2
            }),
            "unique_stop_sites": len({
                (result.chrom, result.strand, result.stop_position)
                for result in values
                if result.tier in {"high", "medium"} and result.stop_support >= 2
            }),
        },
    }


def result_dict(result):
    return asdict(result)
