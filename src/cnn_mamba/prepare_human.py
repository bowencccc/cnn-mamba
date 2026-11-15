#!/usr/bin/env python3
"""Prepare a controlled EviAnn positive-unlabeled experiment.

Windows and positive labels come from EviAnn coding-gene regions. RefSeq is
stored separately as reference truth and as an ``ignore`` mask containing only
RefSeq sites absent from EviAnn. Thus training remains EviAnn-supervised while
validation AP, early stopping, and best-checkpoint selection use RefSeq.
"""

import argparse
import bisect
import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pysam

from .window_storage import WindowWriter

from .human_annotations import (
    CHR_TO_NC,
    EVIANN_GFF,
    FASTA,
    NC_TO_CHR,
    REFSEQ_GFF,
    SITE_TYPES,
    extract_sites,
    open_text,
    parse_eviann,
    parse_gff_attrs,
    parse_reference,
)
from .human_window_helpers import (
    COMPLEMENT,
    ENCODE,
    PRIMARY_CHROMS,
    STRIDE,
    WINDOW_SIZE,
    chromosome_split,
    merge_intervals,
    window_starts_for_regions,
)
from .human_orf_qc import (
    evaluate_eviann_orfs, result_dict, summarize_orf_qc,
)
from .prepare_droso import build_cds_track, build_phase_track


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/processed/human_refseq_w10_chr1held"


def parse_eviann_evidence(path):
    """Return transcript evidence class and low-quality-protein flag."""
    evidence = {}
    low_quality = {}
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "mRNA":
                continue
            attrs = parse_gff_attrs(fields[8])
            tid = attrs.get("ID")
            if not tid:
                continue
            evidence[tid] = attrs.get(
                "evidence", attrs.get("Evidence", "missing")
            )
            low_quality[tid] = "LOW_QUALITY_PROTEIN" in attrs.get(
                "evidence_protein_id", attrs.get("EvidenceProteinID", "")
            )
    return evidence, low_quality


def subset_annotation(tx_ids, tx_meta, exons, cds):
    tx_ids = set(tx_ids)
    return (
        {tid: tx_meta[tid] for tid in tx_ids if tid in tx_meta},
        {tid: exons[tid] for tid in tx_ids if tid in exons},
        {tid: cds[tid] for tid in tx_ids if tid in cds},
    )


def subtract_sites(all_sites, retained_sites):
    return {
        task: set(all_sites[task]) - set(retained_sites[task])
        for task in SITE_TYPES
    }


def build_coding_regions(tx_meta, exons, cds):
    """Merged transcript spans for EviAnn transcripts having CDS."""
    regions = defaultdict(lambda: {"+": [], "-": []})
    for tid, segments in cds.items():
        if not segments or tid not in tx_meta:
            continue
        chrom, strand = tx_meta[tid]
        span_segments = exons.get(tid) or segments
        regions[chrom][strand].append(
            (min(start for start, _ in span_segments), max(end for _, end in span_segments))
        )
    for chrom in regions:
        for strand in ("+", "-"):
            regions[chrom][strand] = merge_intervals(regions[chrom][strand])
    return regions


def index_sites(sites):
    indexed = {
        name: defaultdict(lambda: {"+": set(), "-": set()}) for name in SITE_TYPES
    }
    for name in SITE_TYPES:
        for chrom, strand, position in sites[name]:
            indexed[name][chrom][strand].add(position)
    for name in SITE_TYPES:
        for chrom in indexed[name]:
            for strand in ("+", "-"):
                indexed[name][chrom][strand] = sorted(indexed[name][chrom][strand])
    return indexed


def parse_cds_phase_records(path, annotation_kind, allowed_transcripts=None):
    """Collect primary-chromosome CDS intervals and GFF/GTF phase values."""
    if annotation_kind not in {"eviann", "reference", "chess"}:
        raise ValueError(f"unsupported annotation kind: {annotation_kind}")
    records = defaultdict(lambda: {"+": [], "-": []})
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if (
                len(fields) < 8 or fields[2] != "CDS"
                or fields[6] not in {"+", "-"}
                or fields[7] not in {"0", "1", "2"}
            ):
                continue
            chrom = NC_TO_CHR.get(fields[0], fields[0])
            if chrom not in CHR_TO_NC:
                chrom = None
            if chrom is None:
                continue
            interval = (int(fields[3]) - 1, int(fields[4]), int(fields[7]))
            if allowed_transcripts is None:
                records[chrom][fields[6]].append(interval)
                continue
            attrs = parse_gff_attrs(fields[8])
            parents = {value for value in attrs.get("Parent", "").split(",") if value}
            # An interval shared by multiple transcripts is retained when at
            # least one allowed transcript supports it. Duplicate intervals
            # are harmless and are removed before dense-track construction.
            if parents & allowed_transcripts:
                records[chrom][fields[6]].append(interval)
    return records


def oriented_offset(position, win_start, strand, motif_length):
    if strand == "+":
        return position - win_start
    return WINDOW_SIZE - motif_length - (position - win_start)


def make_two_head_labels(win_start, strand, indexed):
    splice = np.zeros(WINDOW_SIZE, dtype=np.int8)
    start_stop = np.zeros(WINDOW_SIZE, dtype=np.int8)
    win_end = win_start + WINDOW_SIZE
    for name, target, cls, motif_length in (
        ("donor", splice, 1, 2),
        ("acceptor", splice, 2, 2),
        ("start", start_stop, 1, 3),
        ("stop", start_stop, 2, 3),
    ):
        # Motifs must lie fully inside the genomic window.
        positions = indexed[name]
        # Empty/default buckets from ``index_sites`` are sets, whereas
        # populated buckets are finalized as sorted lists. Normalize here so
        # all_eviann (which has no ignored sites) follows the same path.
        if isinstance(positions, set):
            positions = sorted(positions)
        left = bisect.bisect_left(positions, win_start)
        right = bisect.bisect_right(positions, win_end - motif_length)
        for position in positions[left:right]:
            offset = oriented_offset(position, win_start, strand, motif_length)
            if 0 <= offset < WINDOW_SIZE:
                target[offset] = cls
    return splice, start_stop


def evenly_limit(values, limit):
    if not limit or len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, limit, dtype=np.int64)
    return [values[int(index)] for index in indices]


def window_starts_for_region_overlap(regions, chrom_len, overlap_bp):
    if overlap_bp == STRIDE:
        return window_starts_for_regions(regions, chrom_len, WINDOW_SIZE, STRIDE)
    if overlap_bp != WINDOW_SIZE:
        raise ValueError(f"region overlap must be 5000 or {WINDOW_SIZE}, got {overlap_bp}")
    max_start = chrom_len - WINDOW_SIZE
    selected = set()
    for region_start, region_end in regions:
        # Full-window half-open intersection: start < region_end and
        # start + WINDOW_SIZE > region_start.
        low = max(0, region_start - WINDOW_SIZE + 1)
        high = min(max_start, region_end - 1)
        first = ((low + STRIDE - 1) // STRIDE) * STRIDE
        selected.update(range(first, high + 1, STRIDE))
    return sorted(selected)


def generate_split(
    output_dir, split, chroms, regions, eviann, eviann_ignored, reference, fasta_path,
    max_per_chrom_strand, region_overlap_bp,
    eviann_phase=None, eviann_excluded_phase=None, reference_phase=None,
    medium_start_stop=None, medium_start_stop_weight=0.3,
):
    """Generate one complete split; existing split files are never appended to."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta = pysam.FastaFile(str(fasta_path))
    summary = {
        "windows": 0,
        "eviann_positive_positions": {name: 0 for name in SITE_TYPES},
        "reference_positive_positions": {name: 0 for name in SITE_TYPES},
        "masked_excluded_eviann_positions": {name: 0 for name in SITE_TYPES},
        "medium_weight_start_stop_positions": {"start": 0, "stop": 0},
        "joint_frame_masked_bases": 0,
        "by_chrom": {},
    }
    with WindowWriter(output_dir / f"{split}.h5") as writer:
        for chrom in chroms:
            accession = CHR_TO_NC[chrom]
            chrom_len = fasta.get_reference_length(accession)
            chrom_windows = 0
            for strand in ("+", "-"):
                phase_tracks = None
                if eviann_phase is not None:
                    e_phase_track = build_phase_track(
                        eviann_phase[chrom][strand], chrom_len, strand
                    )
                    c_phase_track = build_phase_track(
                        reference_phase[chrom][strand], chrom_len, strand
                    )
                    e_cds_track = build_cds_track(
                        eviann_phase[chrom][strand], chrom_len
                    )
                    excluded_cds_track = (
                        build_cds_track(
                            eviann_excluded_phase[chrom][strand], chrom_len
                        )
                        if eviann_excluded_phase is not None
                        else np.zeros(chrom_len, dtype=np.bool_)
                    )
                    phase_tracks = {
                        "phase": e_phase_track,
                        "phase_mask": e_phase_track >= 0,
                        "reference_phase": c_phase_track,
                        "reference_phase_mask": c_phase_track >= 0,
                        "cds": e_cds_track,
                        "reference_cds": build_cds_track(
                            reference_phase[chrom][strand], chrom_len
                        ),
                        # Excluded EviAnn CDS bases are known not to be noncoding.
                        # Retained complete-HQ CDS wins in overlapping isoforms.
                        "joint_frame_mask": (~excluded_cds_track) | e_cds_track,
                    }
                starts = window_starts_for_region_overlap(
                    regions[chrom][strand], chrom_len, region_overlap_bp
                )
                starts = evenly_limit(starts, max_per_chrom_strand)
                eviann_local = {
                    name: eviann[name][chrom][strand] for name in SITE_TYPES
                }
                ignored_local = {
                    name: eviann_ignored[name][chrom][strand] for name in SITE_TYPES
                }
                reference_local = {
                    name: reference[name][chrom][strand] for name in SITE_TYPES
                }
                medium_local = (
                    {
                        name: medium_start_stop[name][chrom][strand]
                        for name in SITE_TYPES
                    }
                    if medium_start_stop is not None else None
                )
                for win_start in starts:
                    sequence = fasta.fetch(accession, win_start, win_start + WINDOW_SIZE).upper()
                    if len(sequence) != WINDOW_SIZE or sequence.count("N") > WINDOW_SIZE // 2:
                        continue
                    e_splice, e_ss = make_two_head_labels(win_start, strand, eviann_local)
                    i_splice, i_ss = make_two_head_labels(
                        win_start, strand, ignored_local
                    )
                    c_splice, c_ss = make_two_head_labels(
                        win_start, strand, reference_local
                    )
                    if medium_local is not None:
                        _, medium_ss = make_two_head_labels(
                            win_start, strand, medium_local
                        )
                        start_stop_weights = np.ones(WINDOW_SIZE, dtype=np.float32)
                        # medium_start_stop excludes any site also supported by a
                        # complete transcript, so high-confidence support wins.
                        start_stop_weights[(medium_ss != 0) & (e_ss != 0)] = (
                            medium_start_stop_weight
                        )

                    # Excluded EviAnn evidence is unlabeled, not negative. RefSeq
                    # truth is never consulted by the training ignore mask.
                    splice_ignore = (i_splice != 0) & (e_splice == 0)
                    ss_ignore = (i_ss != 0) & (e_ss == 0)

                    if strand == "-":
                        sequence = sequence.translate(COMPLEMENT)[::-1]
                    encoded = ENCODE[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
                    filename = f"{chrom[3:]}_{'plus' if strand == '+' else 'minus'}_{win_start}.npz"
                    arrays = dict(
                        sequence=encoded,
                        labels=e_splice,
                        start_stop_labels=e_ss,
                        splice_ignore=splice_ignore,
                        start_stop_ignore=ss_ignore,
                        reference_labels=c_splice,
                        reference_start_stop_labels=c_ss,
                    )
                    if medium_local is not None:
                        arrays["start_stop_weights"] = start_stop_weights
                    if phase_tracks is not None:
                        end = win_start + WINDOW_SIZE
                        e_phase = phase_tracks["phase"][win_start:end]
                        e_phase_mask = phase_tracks["phase_mask"][win_start:end]
                        c_phase = phase_tracks["reference_phase"][win_start:end]
                        c_phase_mask = phase_tracks[
                            "reference_phase_mask"
                        ][win_start:end]
                        e_cds = phase_tracks["cds"][win_start:end]
                        c_cds = phase_tracks["reference_cds"][win_start:end]
                        joint_frame_mask = phase_tracks[
                            "joint_frame_mask"
                        ][win_start:end]
                        if strand == "-":
                            e_phase = e_phase[::-1]
                            e_phase_mask = e_phase_mask[::-1]
                            c_phase = c_phase[::-1]
                            c_phase_mask = c_phase_mask[::-1]
                            e_cds = e_cds[::-1]
                            c_cds = c_cds[::-1]
                            joint_frame_mask = joint_frame_mask[::-1]
                        arrays.update(
                            phase_labels=np.maximum(e_phase, 0),
                            phase_mask=e_phase_mask,
                            reference_phase_labels=np.maximum(c_phase, 0),
                            reference_phase_mask=c_phase_mask,
                            cds_labels=e_cds,
                            reference_cds_labels=c_cds,
                            joint_frame_mask=joint_frame_mask,
                            reference_joint_frame_mask=np.ones_like(
                                c_cds, dtype=np.bool_
                            ),
                        )
                    writer.append(filename, arrays)
                    chrom_windows += 1
                    summary["windows"] += 1
                    for name, e_arr, i_arr, c_arr, ign_arr, cls in (
                        ("donor", e_splice, i_splice, c_splice, splice_ignore, 1),
                        ("acceptor", e_splice, i_splice, c_splice, splice_ignore, 2),
                        ("start", e_ss, i_ss, c_ss, ss_ignore, 1),
                        ("stop", e_ss, i_ss, c_ss, ss_ignore, 2),
                    ):
                        summary["eviann_positive_positions"][name] += int(np.sum(e_arr == cls))
                        summary["reference_positive_positions"][name] += int(np.sum(c_arr == cls))
                        summary["masked_excluded_eviann_positions"][name] += int(
                            np.sum(ign_arr & (i_arr == cls))
                        )
                        if name in {"start", "stop"} and medium_local is not None:
                            summary["medium_weight_start_stop_positions"][name] += int(
                                np.sum((medium_ss == cls) & (e_arr == cls))
                            )
                    if phase_tracks is not None:
                        summary["joint_frame_masked_bases"] += int(
                            np.sum(~joint_frame_mask)
                        )
                if phase_tracks is not None:
                    # Release chromosome-sized dense tracks before allocating the
                    # next strand/chromosome (chr1 alone is about 249 Mb/base track).
                    del (
                        phase_tracks, e_phase_track, c_phase_track,
                        e_cds_track, excluded_cds_track,
                    )
            summary["by_chrom"][chrom] = chrom_windows
            print(f"[{split}] {chrom}: {chrom_windows:,} windows", flush=True)
    fasta.close()
    return summary


def main():
    global WINDOW_SIZE, STRIDE
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fasta", type=Path, default=FASTA)
    parser.add_argument("--eviann-gff", type=Path, default=EVIANN_GFF)
    parser.add_argument(
        "--reference-annotation", "--chess-gtf", dest="reference_annotation",
        type=Path, default=REFSEQ_GFF,
        help=(
            "RefSeq GFF3/GTF used only for validation, model selection, and "
            "reference arrays. --chess-gtf remains as a deprecated alias."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument(
        "--max-windows-per-chrom-strand",
        type=int,
        default=500,
        help="0 makes the full dataset; 500 is the default controlled pilot.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--label-policy",
        choices=("all_eviann", "task_hq", "task_soft", "task_orf_qc"),
        default="all_eviann",
        help=(
            "all_eviann retains the historical labels. task_hq keeps "
            "complete/protein-only high-quality splice labels, complete "
            "high-quality start/stop and four-state labels, and masks "
            "excluded EviAnn evidence rather than treating it as negative. "
            "task_soft additionally includes non-low-quality protein_only "
            "start/stop positives with a configurable training weight while "
            "keeping four-state labels identical to task_hq. task_orf_qc "
            "requires a strand-aware valid spliced ORF and recognized coding "
            "splice motifs before assigning high/medium start/stop labels."
        ),
    )
    parser.add_argument(
        "--medium-start-stop-weight", type=float, default=0.3,
        help=(
            "Training weight for task_soft protein_only start/stop positives. "
            "Complete non-low-quality positives retain weight 1.0."
        ),
    )
    parser.add_argument(
        "--region-overlap-bp", type=int,
        help=(
            "Retain a window when this owned-center/full-window span overlaps "
            "an EviAnn coding-transcript span. Defaults to the full window."
        ),
    )
    parser.add_argument(
        "--include-phase-labels", action="store_true",
        help=(
            "Store EviAnn training and reference CDS phase/CDS tracks "
            "for auxiliary-head training."
        ),
    )
    args = parser.parse_args()
    if args.window_size <= 0 or args.stride <= 0:
        parser.error("--window-size and --stride must be positive")
    if args.stride != args.window_size // 2:
        parser.error("this experiment requires --stride = --window-size / 2")
    WINDOW_SIZE = args.window_size
    STRIDE = args.stride
    if args.region_overlap_bp is None:
        args.region_overlap_bp = WINDOW_SIZE
    if args.region_overlap_bp not in {STRIDE, WINDOW_SIZE}:
        parser.error("--region-overlap-bp must equal stride or window size")
    if not 0 < args.medium_start_stop_weight <= 1:
        parser.error("--medium-start-stop-weight must be in (0, 1]")
    args.fasta = args.fasta.resolve()
    args.eviann_gff = args.eviann_gff.resolve()
    args.reference_annotation = args.reference_annotation.resolve()
    for path in (args.fasta, args.eviann_gff, args.reference_annotation):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f"Output exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_chroms, val_chroms = chromosome_split(args.seed)
    split_chroms = {"train": train_chroms, "val": val_chroms, "test": ["chr1"]}
    started = time.time()
    print("Parsing annotations...", flush=True)
    e_meta, e_exons, e_cds = parse_eviann(args.eviann_gff)
    c_meta, c_exons, c_cds = parse_reference(args.reference_annotation)
    regions = build_coding_regions(e_meta, e_exons, e_cds)
    print("Extracting canonical CDS sites...", flush=True)
    all_e_sites, e_rejected = extract_sites(
        e_meta, e_exons, e_cds, args.fasta
    )
    c_sites, c_rejected = extract_sites(c_meta, c_exons, c_cds, args.fasta)
    selected_transcripts = None
    medium_sites = None
    orf_qc_summary = None
    orf_qc_results = None
    if args.label_policy in {"task_hq", "task_soft", "task_orf_qc"}:
        evidence, low_quality = parse_eviann_evidence(args.eviann_gff)
        splice_tids = {
            tid for tid in e_meta
            if evidence.get(tid) in {"complete", "protein_only"}
            and not low_quality.get(tid, False)
        }
        if args.label_policy == "task_orf_qc":
            print("Running transcript-oriented EviAnn ORF QC...", flush=True)
            _, orf_qc_results = evaluate_eviann_orfs(
                args.eviann_gff, args.fasta
            )
            orf_qc_summary = summarize_orf_qc(orf_qc_results)
            start_stop_tids = {
                tid for tid, result in orf_qc_results.items()
                if result.tier == "high"
            }
            medium_start_stop_tids = {
                tid for tid, result in orf_qc_results.items()
                if result.tier == "medium"
            }
        else:
            start_stop_tids = {
                tid for tid in e_meta
                if evidence.get(tid) == "complete"
                and not low_quality.get(tid, False)
            }
            medium_start_stop_tids = {
                tid for tid in e_meta
                if evidence.get(tid) == "protein_only"
                and not low_quality.get(tid, False)
            }
        splice_meta, splice_exons, splice_cds = subset_annotation(
            splice_tids, e_meta, e_exons, e_cds
        )
        ss_meta, ss_exons, ss_cds = subset_annotation(
            start_stop_tids, e_meta, e_exons, e_cds
        )
        splice_sites, splice_rejected = extract_sites(
            splice_meta, splice_exons, splice_cds, args.fasta
        )
        ss_sites, ss_rejected = extract_sites(
            ss_meta, ss_exons, ss_cds, args.fasta
        )
        if args.label_policy in {"task_soft", "task_orf_qc"}:
            medium_meta, medium_exons, medium_cds = subset_annotation(
                medium_start_stop_tids, e_meta, e_exons, e_cds
            )
            medium_raw_sites, medium_rejected = extract_sites(
                medium_meta, medium_exons, medium_cds, args.fasta
            )
            # A coordinate supported by both evidence classes is high quality.
            medium_sites = {
                task: set(medium_raw_sites[task]) - set(ss_sites[task])
                for task in SITE_TYPES
            }
            retained_ss_sites = {
                task: set(ss_sites[task]) | set(medium_sites[task])
                for task in SITE_TYPES
            }
        else:
            medium_rejected = None
            retained_ss_sites = ss_sites
        e_sites = {
            "donor": splice_sites["donor"],
            "acceptor": splice_sites["acceptor"],
            "start": retained_ss_sites["start"],
            "stop": retained_ss_sites["stop"],
        }
        ignored_e_sites = subtract_sites(all_e_sites, e_sites)
        selected_transcripts = {
            "splice": len(splice_tids),
            "start_stop_high_and_joint_frame": len(start_stop_tids),
            "start_stop_medium": (
                len(medium_start_stop_tids)
                if args.label_policy in {"task_soft", "task_orf_qc"} else 0
            ),
        }
        selected_rejected = {
            "splice": splice_rejected,
            "start_stop": ss_rejected,
            "start_stop_medium": medium_rejected,
        }
    else:
        e_sites = all_e_sites
        ignored_e_sites = {task: set() for task in SITE_TYPES}
        start_stop_tids = None
        selected_rejected = None
    e_indexed = index_sites(e_sites)
    medium_indexed = index_sites(medium_sites) if medium_sites is not None else None
    ignored_e_indexed = index_sites(ignored_e_sites)
    c_indexed = index_sites(c_sites)
    if args.include_phase_labels:
        print("Parsing EviAnn and RefSeq CDS phase records...", flush=True)
        e_phase = parse_cds_phase_records(
            args.eviann_gff, "eviann",
            allowed_transcripts=start_stop_tids,
        )
        e_excluded_phase = (
            parse_cds_phase_records(
                args.eviann_gff, "eviann",
                allowed_transcripts=set(e_meta) - start_stop_tids,
            )
            if args.label_policy in {"task_hq", "task_soft", "task_orf_qc"}
            else None
        )
        c_phase = parse_cds_phase_records(args.reference_annotation, "reference")
    else:
        e_phase = e_excluded_phase = c_phase = None

    if orf_qc_results is not None:
        (output_dir / "orf_qc_summary.json").write_text(
            json.dumps(orf_qc_summary, indent=2) + "\n"
        )
        rows = [result_dict(result) for result in orf_qc_results.values()]
        with (output_dir / "transcript_orf_qc.tsv").open(
            "w", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summaries = {}
    for split in ("train", "val", "test"):
        summaries[split] = generate_split(
            output_dir,
            split,
            split_chroms[split],
            regions,
            e_indexed,
            ignored_e_indexed,
            c_indexed,
            args.fasta,
            args.max_windows_per_chrom_strand,
            args.region_overlap_bp,
            e_phase,
            e_excluded_phase,
            c_phase,
            medium_indexed,
            args.medium_start_stop_weight,
        )
    manifest = {
        "eviann_annotation": str(args.eviann_gff),
        "reference_annotation": str(args.reference_annotation),
        "reference_name": "RefSeq",
        "fasta": str(args.fasta),
        "seed": args.seed,
        "label_policy": args.label_policy,
        "training_ignore_policy": (
            "excluded_EviAnn_evidence_only; RefSeq is never used to mask training candidates"
            if args.label_policy in {"task_hq", "task_soft", "task_orf_qc"}
            else "none"
        ),
        "start_stop_positive_weighting": (
            {
                "complete_non_low_quality": 1.0,
                "protein_only_non_low_quality": args.medium_start_stop_weight,
                "other_eviann_sites": "masked",
            }
            if args.label_policy in {"task_soft", "task_orf_qc"} else None
        ),
        "orf_qc": orf_qc_summary,
        "orf_qc_policy": (
            {
                "sequence": "strand-aware concatenated CDS",
                "required": [
                    "ACGT-only CDS", "ATG start", "TAA/TAG/TGA stop",
                    "CDS length divisible by 3", "no internal in-frame stop",
                    "GFF phase consistent across CDS segments",
                    "all coding introns in GT-AG/GC-AG/AT-AC",
                ],
                "isoform_support": (
                    "recorded in transcript_orf_qc.tsv; not used to change "
                    "the fixed 0.3 medium weight in this controlled run"
                ),
            }
            if args.label_policy == "task_orf_qc" else None
        ),
        "selected_transcripts": selected_transcripts,
        "selected_unique_sites": {
            task: len(e_sites[task]) for task in SITE_TYPES
        },
        "masked_excluded_unique_sites": {
            task: len(ignored_e_sites[task]) for task in SITE_TYPES
        },
        "medium_weight_unique_sites": (
            {task: len(medium_sites[task]) for task in ("start", "stop")}
            if medium_sites is not None else None
        ),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "region_policy": (
            "owned_center_intersects_EviAnn_transcript_span_with_CDS"
            if args.region_overlap_bp == STRIDE
            else "full_window_intersects_EviAnn_transcript_span_with_CDS"
        ),
        "region_overlap_bp": args.region_overlap_bp,
        "max_windows_per_chrom_strand": args.max_windows_per_chrom_strand,
        "phase_labels": (
            "EviAnn/RefSeq CDS phase converted to transcript-oriented per-base "
            "states 0/1/2; non-CDS and isoform conflicts are masked"
            if args.include_phase_labels else None
        ),
        "cds_labels": (
            "complete high-confidence EviAnn CDS; task_orf_qc additionally requires a valid spliced ORF; RefSeq CDS is stored separately"
            if args.include_phase_labels and args.label_policy in {"task_hq", "task_soft", "task_orf_qc"} else
            "union of EviAnn/RefSeq CDS intervals independent of phase conflicts"
            if args.include_phase_labels else None
        ),
        "joint_frame_mask": (
            "excluded EviAnn CDS is masked unless overlapped by retained complete-HQ CDS; remaining non-CDS and retained phased CDS are valid"
            if args.include_phase_labels and args.label_policy in {"task_hq", "task_soft", "task_orf_qc"} else
            "all bases valid"
            if args.include_phase_labels else None
        ),
        "splits": split_chroms,
        "summary": summaries,
        "eviann_rejected": e_rejected,
        "selected_eviann_rejected": selected_rejected,
        "reference_rejected": c_rejected,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Finished in {(time.time() - started) / 60:.1f} min: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
