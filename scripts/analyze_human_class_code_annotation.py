#!/usr/bin/env python3
"""Audit class-code-aware Human EviAnn labels against RefSeq.

The primary deployment comparison mirrors the earlier Drosophila/Coleoptera
audit: unique canonical sites on chr1 positive strand, with transcript models
selected per EviAnn locus under several label-selection policies.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cnn_mamba.human_annotations import (
    CHR_TO_NC,
    FASTA,
    REFSEQ_GFF,
    SITE_TYPES,
    extract_sites,
    open_text,
    parse_gff_attrs,
    parse_reference,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/GCF_000001405.40_GRCh38.p14_genomic.noalpsnopatches.fna.pseudo_label.gff.gz"
DEFAULT_OUTPUT = ROOT / "results/human_eviann_class_code_v1"
DEFAULT_LIGHTGBM = ROOT / "results/human_site_quality_lightgbm_v1"
VALIDATION_CHROMS = {"chr2", "chr5", "chr9", "chr10", "chr22"}

STRATEGIES = (
    "all_transcripts_union",
    "longest_cds_per_locus",
    "longest_complete_equal_else_k",
    "complete_equal_else_k_with_fallback",
)
PLOT_METHODS = STRATEGIES + ("lightgbm_oof_p90_crossfit",)
STRATEGY_LABELS = {
    "all_transcripts_union": "All isoforms (union)",
    "longest_cds_per_locus": "Longest CDS per locus",
    "longest_complete_equal_else_k": "Complete '='; else 'k'",
    "complete_equal_else_k_with_fallback": "Complete '='/'k' + locus fallback",
    "lightgbm_oof_p90_crossfit": "LightGBM OOF P90",
}
COLORS = {
    "all_transcripts_union": "#7A7A7A",
    "longest_cds_per_locus": "#2676B8",
    "longest_complete_equal_else_k": "#D65F5F",
    "complete_equal_else_k_with_fallback": "#55A868",
    "lightgbm_oof_p90_crossfit": "#8C6BB1",
}


def attr(attrs: dict[str, str], lower: str, legacy: str, default: str = "") -> str:
    return attrs.get(lower, attrs.get(legacy, default))


def parse_eviann(path: Path):
    nc_to_chr = {value: key for key, value in CHR_TO_NC.items()}
    meta: dict[str, dict[str, object]] = {}
    exons = defaultdict(list)
    cds = defaultdict(list)
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[0] not in nc_to_chr:
                continue
            feature = fields[2]
            if feature not in {"mRNA", "exon", "CDS"}:
                continue
            attrs = parse_gff_attrs(fields[8])
            if feature == "mRNA":
                tid = attrs.get("ID")
                if not tid:
                    continue
                protein = attr(attrs, "evidence_protein_id", "EvidenceProteinID")
                meta[tid] = {
                    "locus": attrs.get("Parent", attrs.get("geneID", tid)),
                    "chrom": nc_to_chr[fields[0]],
                    "strand": fields[6],
                    "evidence": attr(attrs, "evidence", "Evidence", "missing"),
                    "class_code": attr(attrs, "class", "class_code", "missing"),
                    "low_quality": "LOW_QUALITY_PROTEIN" in protein,
                }
                continue
            start0, end0 = int(fields[3]) - 1, int(fields[4])
            target = exons if feature == "exon" else cds
            for tid in attrs.get("Parent", "").split(","):
                if tid:
                    target[tid].append((start0, end0))
    return meta, exons, cds


def cds_length(parts) -> int:
    return sum(end - start for start, end in parts)


def choose_longest(tids, cds):
    available = [tid for tid in tids if cds.get(tid)]
    if not available:
        return None
    return max(available, key=lambda tid: (cds_length(cds[tid]), tid))


def select_transcripts(meta, cds):
    loci = defaultdict(list)
    for tid, record in meta.items():
        if cds.get(tid):
            loci[record["locus"]].append(tid)

    selected = {name: set() for name in STRATEGIES}
    selected["all_transcripts_union"] = {
        tid for tids in loci.values() for tid in tids
    }
    exclusion_rows = []
    exclusion_reasons = Counter()

    for locus, tids in loci.items():
        longest = choose_longest(tids, cds)
        if longest:
            selected["longest_cds_per_locus"].add(longest)

        complete = [
            tid for tid in tids if str(meta[tid]["evidence"]).lower() == "complete"
        ]
        exact = [tid for tid in complete if meta[tid]["class_code"] == "="]
        k_class = [tid for tid in complete if meta[tid]["class_code"] == "k"]
        preferred = choose_longest(exact, cds) or choose_longest(k_class, cds)
        if preferred:
            selected["longest_complete_equal_else_k"].add(preferred)
            selected["complete_equal_else_k_with_fallback"].add(preferred)
            continue

        if longest:
            selected["complete_equal_else_k_with_fallback"].add(longest)
        if not complete:
            reason = "no_complete_transcript"
        else:
            reason = "complete_present_but_class_not_equal_or_k"
        exclusion_reasons[reason] += 1
        evidence_counts = Counter(str(meta[tid]["evidence"]) for tid in tids)
        class_counts = Counter(str(meta[tid]["class_code"]) for tid in tids)
        exclusion_rows.append({
            "locus": locus,
            "chrom": meta[tids[0]]["chrom"],
            "strand": meta[tids[0]]["strand"],
            "reason": reason,
            "transcripts": len(tids),
            "complete_transcripts": len(complete),
            "evidence_counts": ",".join(f"{k}:{v}" for k, v in sorted(evidence_counts.items())),
            "class_counts": ",".join(f"{k}:{v}" for k, v in sorted(class_counts.items())),
            "longest_cds_bp": max(cds_length(cds[tid]) for tid in tids),
        })
    return loci, selected, exclusion_rows, exclusion_reasons


def subset_for_extract(tids, meta, exons, cds):
    tids = set(tids)
    tx_meta = {
        tid: (str(meta[tid]["chrom"]), str(meta[tid]["strand"]))
        for tid in tids if tid in meta
    }
    return (
        tx_meta,
        {tid: exons[tid] for tid in tids if exons.get(tid)},
        {tid: cds[tid] for tid in tids if cds.get(tid)},
    )


def scope_sites(sites, scope):
    if scope == "chr1_plus":
        return {site for site in sites if site[0] == "chr1" and site[1] == "+"}
    if scope == "chr1_both":
        return {site for site in sites if site[0] == "chr1"}
    if scope == "validation":
        return {site for site in sites if site[0] in VALIDATION_CHROMS}
    if scope == "all_primary":
        return sites
    raise ValueError(scope)


def metric_rows(label, sites, reference, rejected, group_type):
    rows = []
    for scope in ("chr1_plus", "chr1_both", "validation", "all_primary"):
        for task in SITE_TYPES:
            prediction = scope_sites(sites[task], scope)
            truth = scope_sites(reference[task], scope)
            overlap = prediction & truth
            precision = len(overlap) / len(prediction) if prediction else 0.0
            sensitivity = len(overlap) / len(truth) if truth else 0.0
            f1 = (
                2 * precision * sensitivity / (precision + sensitivity)
                if precision + sensitivity else 0.0
            )
            rows.append({
                "group_type": group_type,
                "selection": label,
                "scope": scope,
                "task": task,
                "precision": precision,
                "sensitivity": sensitivity,
                "f1": f1,
                "matched_sites": len(overlap),
                "eviann_sites": len(prediction),
                "reference_sites": len(truth),
                "eviann_only_sites": len(prediction - truth),
                "missed_sites": len(truth - prediction),
                "rejected_noncanonical_total": rejected.get(task, 0),
            })
    return rows


def lightgbm_chr1_plus_rows(model_dir, current_sites, reference):
    """Evaluate held-fold chr1 OOF scores at cross-fitted P90 thresholds."""
    summary = json.loads((model_dir / "summary_corrected.json").read_text())
    rows = []
    thresholds = {}
    for task in ("start", "stop"):
        operating_point = next(
            row
            for row in summary["tasks"][task]["cross_fitted_precision_operating_points"]
            if row["target_precision"] == 0.9
        )
        fold = next(row for row in operating_point["folds"] if row["fold"] == 0)
        threshold = float(fold["threshold_fitted_without_held_fold"])
        thresholds[task] = threshold
        selected = set()
        score_path = model_dir / f"{task}_site_scores.tsv.gz"
        with gzip.open(score_path, "rt") as handle:
            for record in csv.DictReader(handle, delimiter="\t"):
                if (
                    record["chrom"] == "chr1"
                    and record["strand"] == "+"
                    and float(record["oof_score"]) >= threshold
                ):
                    selected.add(("chr1", "+", int(record["position_0based"])))
        # The LightGBM was fitted before class codes were added. Restrict its
        # score set to candidates still present in the new annotation.
        selected &= scope_sites(current_sites[task], "chr1_plus")
        truth = scope_sites(reference[task], "chr1_plus")
        overlap = selected & truth
        precision = len(overlap) / len(selected) if selected else 0.0
        sensitivity = len(overlap) / len(truth) if truth else 0.0
        f1 = (
            2 * precision * sensitivity / (precision + sensitivity)
            if precision + sensitivity else 0.0
        )
        rows.append({
            "group_type": "lightgbm",
            "selection": "lightgbm_oof_p90_crossfit",
            "scope": "chr1_plus",
            "task": task,
            "precision": precision,
            "sensitivity": sensitivity,
            "f1": f1,
            "matched_sites": len(overlap),
            "eviann_sites": len(selected),
            "reference_sites": len(truth),
            "eviann_only_sites": len(selected - truth),
            "missed_sites": len(truth - selected),
            "rejected_noncanonical_total": 0,
        })
    return rows, thresholds


def write_tsv(path: Path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot_start_stop(rows, output_stem: Path):
    selected = {
        (row["selection"], row["task"]): row
        for row in rows
        if row["group_type"] in {"selection", "lightgbm"}
        and row["scope"] == "chr1_plus"
        and row["task"] in {"start", "stop"}
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.8), sharey=True)
    x = np.arange(2)
    width = 0.15
    for ax, metric, title in zip(
        axes, ("precision", "sensitivity", "f1"), ("Precision", "Sensitivity", "F1")
    ):
        for index, strategy in enumerate(PLOT_METHODS):
            values = np.array([
                100 * selected[strategy, task][metric] for task in ("start", "stop")
            ])
            bars = ax.bar(
                x + (index - (len(PLOT_METHODS) - 1) / 2) * width,
                values,
                width,
                color=COLORS[strategy],
                label=STRATEGY_LABELS[strategy],
            )
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.8, rotation=90)
        ax.set_title(title)
        ax.set_xticks(x, ("Start codon", "Stop codon"))
        ax.set_ylim(0, 108)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Percent (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=False)
    fig.suptitle("Human chr1+ EviAnn start/stop label selection")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--refseq", type=Path, default=REFSEQ_GFF)
    parser.add_argument("--fasta", type=Path, default=FASTA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lightgbm-dir", type=Path, default=DEFAULT_LIGHTGBM)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing class-code-aware Human EviAnn annotation...", flush=True)
    meta, exons, cds = parse_eviann(args.input)
    loci, selections, excluded, exclusion_reasons = select_transcripts(meta, cds)
    print(f"Parsed {len(meta):,} mRNAs and {len(loci):,} coding loci", flush=True)

    evidence_counts = Counter(str(record["evidence"]) for record in meta.values())
    class_counts = Counter(str(record["class_code"]) for record in meta.values())
    pair_counts = Counter(
        (str(record["evidence"]), str(record["class_code"]))
        for record in meta.values()
    )
    distribution_rows = []
    for (evidence, class_code), count in sorted(pair_counts.items()):
        tids = {
            tid for tid, record in meta.items()
            if record["evidence"] == evidence and record["class_code"] == class_code
        }
        distribution_rows.append({
            "evidence": evidence,
            "class_code": class_code,
            "transcripts": count,
            "transcripts_with_cds": sum(bool(cds.get(tid)) for tid in tids),
            "loci": len({meta[tid]["locus"] for tid in tids}),
            "low_quality_transcripts": sum(bool(meta[tid]["low_quality"]) for tid in tids),
        })
    write_tsv(args.output_dir / "evidence_class_distribution.tsv", distribution_rows)

    print("Extracting RefSeq canonical sites...", flush=True)
    ref_meta, ref_exons, ref_cds = parse_reference(args.refseq)
    reference, reference_rejected = extract_sites(
        ref_meta, ref_exons, ref_cds, args.fasta
    )

    rows = []
    strategy_sites = {}
    for strategy in STRATEGIES:
        print(f"Evaluating {strategy}: {len(selections[strategy]):,} transcripts", flush=True)
        subset = subset_for_extract(selections[strategy], meta, exons, cds)
        sites, rejected = extract_sites(*subset, args.fasta)
        strategy_sites[strategy] = sites
        rows.extend(metric_rows(strategy, sites, reference, rejected, "selection"))

    lightgbm_rows, lightgbm_thresholds = lightgbm_chr1_plus_rows(
        args.lightgbm_dir,
        strategy_sites["all_transcripts_union"],
        reference,
    )
    rows.extend(lightgbm_rows)

    # Class-code-specific site quality, without per-locus transcript selection.
    for class_code in sorted(class_counts, key=lambda x: (-class_counts[x], x)):
        tids = {tid for tid, record in meta.items() if record["class_code"] == class_code}
        subset = subset_for_extract(tids, meta, exons, cds)
        sites, rejected = extract_sites(*subset, args.fasta)
        rows.extend(metric_rows(class_code, sites, reference, rejected, "class_code"))

    write_tsv(args.output_dir / "site_prsn.tsv", rows)
    with gzip.open(args.output_dir / "excluded_loci.tsv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(excluded[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(excluded)

    summary = {
        "source": str(args.input.resolve()),
        "reference": str(args.refseq.resolve()),
        "transcripts": len(meta),
        "coding_transcripts": sum(bool(cds.get(tid)) for tid in meta),
        "coding_loci": len(loci),
        "low_quality_transcripts": sum(bool(record["low_quality"]) for record in meta.values()),
        "evidence_counts": dict(evidence_counts),
        "class_code_counts": dict(class_counts),
        "selection_transcript_counts": {key: len(value) for key, value in selections.items()},
        "lightgbm_evaluation": {
            "model_directory": str(args.lightgbm_dir.resolve()),
            "score_type": "chr1 held-fold OOF",
            "threshold_selection": "P90 threshold fitted without chr1 fold",
            "thresholds": lightgbm_thresholds,
            "restricted_to_new_annotation_candidates": True,
        },
        "strict_complete_equal_else_k_excluded_loci": len(excluded),
        "strict_exclusion_reasons": dict(exclusion_reasons),
        "reference_rejected_noncanonical": reference_rejected,
        "metric_definition": (
            "Exact unique canonical CDS-site match by chromosome, strand, and "
            "0-based motif-start position; primary report scope is chr1 positive strand."
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_start_stop(rows, args.output_dir / "chr1_plus_start_stop_selection_prsn")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
