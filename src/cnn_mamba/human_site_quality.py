#!/usr/bin/env python3
"""Leakage-safe features and labels for EviAnn boundary-quality models."""

import bisect
from collections import Counter, defaultdict
import math

import numpy as np
import pysam

from .human_annotations import CHR_TO_NC, extract_sites, parse_reference
from .human_orf_qc import evaluate_eviann_orfs, reverse_complement


PRIMARY_CHROMS = tuple(
    [f"chr{index}" for index in range(1, 23)] + ["chrX", "chrY"]
)
CHROM_FOLDS = (
    ("chr1", "chr6", "chr11", "chr16", "chr21"),
    ("chr2", "chr7", "chr12", "chr17", "chr22"),
    ("chr3", "chr8", "chr13", "chr18", "chrX"),
    ("chr4", "chr9", "chr14", "chr19", "chrY"),
    ("chr5", "chr10", "chr15", "chr20"),
)
CHROM_TO_FOLD = {
    chrom: fold for fold, chroms in enumerate(CHROM_FOLDS) for chrom in chroms
}
BASES = "ACGT"


class CodingSpanIndex:
    """Binned same-strand index for conservative coding-locus matching."""

    def __init__(self, tx_meta, cds, bin_size=100_000):
        self.bin_size = bin_size
        self.bins = defaultdict(list)
        for tid, segments in cds.items():
            if tid not in tx_meta or not segments:
                continue
            chrom, strand = tx_meta[tid]
            start = min(value[0] for value in segments)
            end = max(value[1] for value in segments)
            for index in range(start // bin_size, (end - 1) // bin_size + 1):
                self.bins[chrom, strand, index].append((start, end))

    def comparable(self, chrom, strand, start, end, min_shorter_overlap=0.5):
        if end <= start:
            return False
        candidates = set()
        for index in range(
            start // self.bin_size, (end - 1) // self.bin_size + 1
        ):
            candidates.update(self.bins.get((chrom, strand, index), ()))
        length = end - start
        for other_start, other_end in candidates:
            overlap = min(end, other_end) - max(start, other_start)
            if overlap <= 0:
                continue
            if overlap / min(length, other_end - other_start) >= min_shorter_overlap:
                return True
        return False


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _fraction(results, name):
    return _mean([float(bool(getattr(result, name))) for result in results])


def _context(fasta, chrom, strand, position, left=20, right=23):
    accession = CHR_TO_NC[chrom]
    chrom_length = fasta.get_reference_length(accession)
    start = max(0, position - left)
    end = min(chrom_length, position + right)
    sequence = fasta.fetch(accession, start, end).upper()
    prefix = "N" * max(0, left - position)
    suffix = "N" * max(0, position + right - chrom_length)
    sequence = prefix + sequence + suffix
    if strand == "-":
        sequence = reverse_complement(sequence)
    if len(sequence) != left + right:
        raise ValueError(
            f"context length {len(sequence)} != {left + right}: "
            f"{chrom}:{position}:{strand}"
        )
    return sequence


def _gc_fraction(fasta, chrom, position, radius):
    accession = CHR_TO_NC[chrom]
    chrom_length = fasta.get_reference_length(accession)
    sequence = fasta.fetch(
        accession, max(0, position - radius),
        min(chrom_length, position + radius + 3),
    ).upper()
    informative = sum(sequence.count(base) for base in BASES)
    return (sequence.count("G") + sequence.count("C")) / max(informative, 1)


def feature_names(context_left=20, context_right=23):
    names = [
        "strand_minus", "support_transcripts", "support_genes",
        "fraction_complete", "fraction_protein_only", "fraction_transcript_only",
        "fraction_low_quality", "any_complete_non_low_quality",
        "any_protein_only_non_low_quality", "fraction_orf_valid",
        "fraction_sequence_complete", "fraction_start_atg",
        "fraction_stop_valid", "fraction_length_multiple_of_three",
        "fraction_no_internal_inframe_stop", "fraction_phase_consistent",
        "fraction_splice_recognized", "fraction_all_splice_gt_ag",
        "mean_cds_length_log1p", "max_cds_length_log1p",
        "min_cds_length_log1p", "mean_exon_count", "max_exon_count",
        "mean_cds_segments", "mean_coding_introns", "mean_gt_ag",
        "mean_gc_ag", "mean_at_ac", "mean_other_splice",
        "mean_gene_isoforms", "max_gene_isoforms", "site_gene_support_ratio",
        "partner_unique_sites", "partner_dominance", "declared_codon_match_fraction",
        "gc_context_43", "gc_context_101", "gc_context_1001",
        "kozak_minus3_purine", "kozak_plus4_g",
        "codon_TAA", "codon_TAG", "codon_TGA",
    ]
    for offset in range(-context_left, context_right):
        for base in BASES:
            names.append(f"base_{offset:+d}_{base}")
    return names


def build_site_quality_examples(eviann_gff, refseq_path, fasta_path):
    """Return one example per unique EviAnn start or stop coordinate.

    Labels are 1 for exact RefSeq matches, 0 for non-matches at a comparable
    same-strand coding locus, and -1 when no reliable RefSeq locus match exists.
    RefSeq-derived values are metadata only and never enter the feature vector.
    """
    records, qc = evaluate_eviann_orfs(eviann_gff, fasta_path)
    reference_meta, reference_exons, reference_cds = parse_reference(refseq_path)
    reference_sites, _ = extract_sites(
        reference_meta, reference_exons, reference_cds, fasta_path
    )
    span_index = CodingSpanIndex(reference_meta, reference_cds)
    gene_isoforms = Counter(
        record.gene_id for record in records.values() if record.gene_id
    )
    support = {
        "start": defaultdict(list),
        "stop": defaultdict(list),
    }
    for tid, result in qc.items():
        if result.start_position >= 0 and result.start_atg:
            support["start"][
                (result.chrom, result.strand, result.start_position)
            ].append(tid)
        if result.stop_position >= 0 and result.stop_valid:
            support["stop"][
                (result.chrom, result.strand, result.stop_position)
            ].append(tid)

    names = feature_names()
    fasta = pysam.FastaFile(str(fasta_path))
    examples = {"start": [], "stop": []}
    for task in ("start", "stop"):
        for site, tids in support[task].items():
            chrom, strand, position = site
            site_results = [qc[tid] for tid in tids]
            site_records = [records[tid] for tid in tids]
            comparable = False
            for record in site_records:
                if not record.cds:
                    continue
                start = min(segment[0] for segment in record.cds)
                end = max(segment[1] for segment in record.cds)
                if span_index.comparable(chrom, strand, start, end):
                    comparable = True
                    break
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
                declared = (
                    result.declared_start if task == "start"
                    else result.declared_stop
                )
                observed = result.start_codon if task == "start" else result.stop_codon
                declared_match.append(float(bool(declared) and declared == observed))
            lengths = [result.cds_length for result in site_results]
            values = [
                float(strand == "-"), float(len(tids)), float(len(genes)),
                _mean([result.evidence == "complete" for result in site_results]),
                _mean([result.evidence == "protein_only" for result in site_results]),
                _mean([result.evidence == "transcript_only" for result in site_results]),
                _mean([result.low_quality_protein for result in site_results]),
                float(any(
                    result.evidence == "complete" and not result.low_quality_protein
                    for result in site_results
                )),
                float(any(
                    result.evidence == "protein_only" and not result.low_quality_protein
                    for result in site_results
                )),
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
                float(len(partner_counts)),
                max(partner_counts.values()) / len(tids),
                _mean(declared_match),
                sum(context.count(base) for base in "GC")
                / max(sum(context.count(base) for base in BASES), 1),
                _gc_fraction(fasta, chrom, position, 49),
                _gc_fraction(fasta, chrom, position, 499),
                float(context[17] in {"A", "G"}),
                float(context[23] == "G"),
                float(codon == "TAA"), float(codon == "TAG"),
                float(codon == "TGA"),
            ]
            for base in context:
                values.extend(float(base == requested) for requested in BASES)
            if len(values) != len(names):
                raise RuntimeError(f"feature count {len(values)} != {len(names)}")
            examples[task].append({
                "task": task,
                "chrom": chrom,
                "strand": strand,
                "position_0based": position,
                "fold": CHROM_TO_FOLD[chrom],
                "label": label,
                "exact_refseq_match": int(exact),
                "comparable_refseq_locus": int(comparable),
                "support_transcript_ids": ",".join(sorted(tids)),
                "features": np.asarray(values, dtype=np.float32),
            })
        examples[task].sort(
            key=lambda row: (
                PRIMARY_CHROMS.index(row["chrom"]), row["strand"],
                row["position_0based"],
            )
        )
    fasta.close()
    return examples, names


def average_precision(labels, scores):
    """Average precision with all equal-score examples handled as one group."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    positives = int(ordered.sum())
    if not positives:
        return 0.0
    ordered_scores = scores[order]
    group_ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    true_positives = np.cumsum(ordered)[group_ends]
    selected = group_ends + 1
    precision = true_positives / selected
    recall = true_positives / positives
    recall_increase = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increase * precision))


def threshold_for_precision(labels, scores, target):
    """Lowest whole-score-group threshold satisfying a precision target."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    group_ends = np.flatnonzero(np.r_[ordered_scores[1:] != ordered_scores[:-1], True])
    tp = np.cumsum(ordered_labels)[group_ends]
    selected = group_ends + 1
    precision = tp / selected
    eligible = np.flatnonzero(precision >= target)
    if not len(eligible):
        return None
    group_index = int(eligible[-1])
    index = int(group_ends[group_index])
    return {
        "target_precision": target,
        "threshold": float(ordered_scores[index]),
        "precision": float(precision[group_index]),
        "recall": float(tp[group_index] / max(labels.sum(), 1)),
        "selected_labeled_sites": index + 1,
        "true_positive_sites": int(tp[group_index]),
    }


def cross_fitted_threshold_metrics(labels, scores, folds, target):
    """Choose a threshold off-fold, then evaluate it on each held-out fold."""
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    folds = np.asarray(folds, dtype=np.int16)
    if not (labels.shape == scores.shape == folds.shape):
        raise ValueError("labels, scores, and folds must have identical shapes")

    selected = np.zeros(len(labels), dtype=bool)
    fold_rows = []
    for held_fold in sorted(np.unique(folds).tolist()):
        train = folds != held_fold
        held = ~train
        fitted = threshold_for_precision(labels[train], scores[train], target)
        if fitted is None:
            held_selected = np.zeros(int(held.sum()), dtype=bool)
            threshold = None
        else:
            threshold = fitted["threshold"]
            held_selected = scores[held] >= threshold
        selected[held] = held_selected
        held_labels = labels[held]
        tp = int(held_labels[held_selected].sum())
        count = int(held_selected.sum())
        positives = int(held_labels.sum())
        fold_rows.append({
            "fold": int(held_fold),
            "threshold_fitted_without_held_fold": threshold,
            "threshold_fit": fitted,
            "held_labeled_sites": int(held.sum()),
            "held_positive_sites": positives,
            "selected_labeled_sites": count,
            "true_positive_sites": tp,
            "precision": float(tp / count) if count else None,
            "recall": float(tp / positives) if positives else None,
        })

    tp = int(labels[selected].sum())
    count = int(selected.sum())
    positives = int(labels.sum())
    return {
        "target_precision": target,
        "method": "threshold fitted on the other chromosome folds",
        "precision": float(tp / count) if count else None,
        "recall": float(tp / positives) if positives else None,
        "selected_labeled_sites": count,
        "true_positive_sites": tp,
        "folds": fold_rows,
    }


def calibration_bins(labels, scores, bins=10):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    output = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (scores >= lower) & (
            scores <= upper if index == bins - 1 else scores < upper
        )
        output.append({
            "bin": index,
            "lower": float(lower),
            "upper": float(upper),
            "count": int(mask.sum()),
            "mean_score": float(scores[mask].mean()) if mask.any() else None,
            "observed_match_fraction": (
                float(labels[mask].mean()) if mask.any() else None
            ),
        })
    return output
