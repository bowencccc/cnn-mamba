#!/usr/bin/env python3
"""Prepare a controlled EviAnn positive-unlabeled experiment.

Windows and positive labels come from EviAnn coding-gene regions.  CHESS is
stored separately as reference truth and as an ``ignore`` mask containing only
CHESS sites absent from EviAnn.  Thus baseline and masked training can use the
same NPZ files and differ only in whether the ignore mask participates in loss.
"""

import argparse
import bisect
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pysam

from .human_annotations import (
    CHESS_GTF,
    CHR_TO_NC,
    EVIANN_GFF,
    FASTA,
    NC_TO_CHR,
    SITE_TYPES,
    extract_sites,
    parse_chess,
    parse_eviann,
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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/processed/human_w10_chr1held"


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
    if overlap_bp == 5_000:
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
    output_dir, split, chroms, regions, eviann, chess, fasta_path,
    max_per_chrom_strand, region_overlap_bp,
):
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    fasta = pysam.FastaFile(str(fasta_path))
    summary = {
        "windows": 0,
        "eviann_positive_positions": {name: 0 for name in SITE_TYPES},
        "chess_positive_positions": {name: 0 for name in SITE_TYPES},
        "ignored_chess_only_positions": {name: 0 for name in SITE_TYPES},
        "by_chrom": {},
    }
    for chrom in chroms:
        accession = CHR_TO_NC[chrom]
        chrom_len = fasta.get_reference_length(accession)
        chrom_windows = 0
        for strand in ("+", "-"):
            starts = window_starts_for_region_overlap(
                regions[chrom][strand], chrom_len, region_overlap_bp
            )
            starts = evenly_limit(starts, max_per_chrom_strand)
            eviann_local = {
                name: eviann[name][chrom][strand] for name in SITE_TYPES
            }
            chess_local = {name: chess[name][chrom][strand] for name in SITE_TYPES}
            for win_start in starts:
                sequence = fasta.fetch(accession, win_start, win_start + WINDOW_SIZE).upper()
                if len(sequence) != WINDOW_SIZE or sequence.count("N") > WINDOW_SIZE // 2:
                    continue
                e_splice, e_ss = make_two_head_labels(win_start, strand, eviann_local)
                c_splice, c_ss = make_two_head_labels(win_start, strand, chess_local)

                # A class-specific CHESS site is unknown only when the same class is
                # absent from EviAnn. Any EviAnn positive always wins over ignore.
                splice_ignore = (c_splice != 0) & (c_splice != e_splice) & (e_splice == 0)
                ss_ignore = (c_ss != 0) & (c_ss != e_ss) & (e_ss == 0)

                if strand == "-":
                    sequence = sequence.translate(COMPLEMENT)[::-1]
                encoded = ENCODE[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
                filename = f"{chrom[3:]}_{'plus' if strand == '+' else 'minus'}_{win_start}.npz"
                np.savez_compressed(
                    split_dir / filename,
                    sequence=encoded,
                    labels=e_splice,
                    start_stop_labels=e_ss,
                    splice_ignore=splice_ignore,
                    start_stop_ignore=ss_ignore,
                    chess_labels=c_splice,
                    chess_start_stop_labels=c_ss,
                )
                chrom_windows += 1
                summary["windows"] += 1
                for name, e_arr, c_arr, ign_arr, cls in (
                    ("donor", e_splice, c_splice, splice_ignore, 1),
                    ("acceptor", e_splice, c_splice, splice_ignore, 2),
                    ("start", e_ss, c_ss, ss_ignore, 1),
                    ("stop", e_ss, c_ss, ss_ignore, 2),
                ):
                    summary["eviann_positive_positions"][name] += int(np.sum(e_arr == cls))
                    summary["chess_positive_positions"][name] += int(np.sum(c_arr == cls))
                    summary["ignored_chess_only_positions"][name] += int(
                        np.sum(ign_arr & (c_arr == cls))
                    )
        summary["by_chrom"][chrom] = chrom_windows
        print(f"[{split}] {chrom}: {chrom_windows:,} windows", flush=True)
    fasta.close()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fasta", type=Path, default=FASTA)
    parser.add_argument("--eviann-gff", type=Path, default=EVIANN_GFF)
    parser.add_argument("--chess-gtf", type=Path, default=CHESS_GTF)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-windows-per-chrom-strand",
        type=int,
        default=500,
        help="0 makes the full dataset; 500 is the default controlled pilot.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--region-overlap-bp", type=int, choices=(5000, 10000), default=10000,
        help="Retain a window when its central 5 kb or full 10 kb overlaps an EviAnn coding-transcript span.",
    )
    args = parser.parse_args()
    args.fasta = args.fasta.resolve()
    args.eviann_gff = args.eviann_gff.resolve()
    args.chess_gtf = args.chess_gtf.resolve()
    for path in (args.fasta, args.eviann_gff, args.chess_gtf):
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
    c_meta, c_exons, c_cds = parse_chess(args.chess_gtf)
    regions = build_coding_regions(e_meta, e_exons, e_cds)
    print("Extracting canonical CDS sites...", flush=True)
    e_sites, e_rejected = extract_sites(e_meta, e_exons, e_cds, args.fasta)
    c_sites, c_rejected = extract_sites(c_meta, c_exons, c_cds, args.fasta)
    e_indexed, c_indexed = index_sites(e_sites), index_sites(c_sites)

    summaries = {}
    for split in ("train", "val", "test"):
        summaries[split] = generate_split(
            output_dir,
            split,
            split_chroms[split],
            regions,
            e_indexed,
            c_indexed,
            args.fasta,
            args.max_windows_per_chrom_strand,
            args.region_overlap_bp,
        )
    manifest = {
        "eviann_annotation": str(args.eviann_gff),
        "chess_annotation": str(args.chess_gtf),
        "fasta": str(args.fasta),
        "seed": args.seed,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "region_policy": (
            "central_5kb_intersects_EviAnn_transcript_span_with_CDS"
            if args.region_overlap_bp == 5000
            else "full_10kb_intersects_EviAnn_transcript_span_with_CDS"
        ),
        "region_overlap_bp": args.region_overlap_bp,
        "max_windows_per_chrom_strand": args.max_windows_per_chrom_strand,
        "splits": split_chroms,
        "summary": summaries,
        "eviann_rejected": e_rejected,
        "chess_rejected": c_rejected,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Finished in {(time.time() - started) / 60:.1f} min: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
