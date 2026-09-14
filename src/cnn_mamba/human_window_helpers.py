#!/usr/bin/env python3
"""Prepare CHESS-reference human SSM data with chromosome-level 80/20 split.

Split policy (fixed and recorded in ``split_manifest.json``):
  * chr1: independent test
  * remaining primary chromosomes: deterministic 80/20 train/validation split
  * train/validation/test NPZ windows: only coding-gene regions

Each saved sample is a 10 kb strand-oriented sequence with the same label schema
as the existing human EviAnn SSM run.
"""

import argparse
import bisect
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pysam


ROOT = Path(__file__).resolve().parents[2]
GTF_PATH = ROOT / "data/raw/human_chess3.1.3.gtf"
FASTA_PATH = ROOT / "data/raw/human_grch38.fa"
DEFAULT_OUTPUT = ROOT / "data/processed/human_chess_w10_chr1held"
WINDOW_SIZE = 10_000
STRIDE = 5_000
SEED = 42

CHR_TO_NC = {
    "chr1": "NC_000001.11", "chr2": "NC_000002.12",
    "chr3": "NC_000003.12", "chr4": "NC_000004.12",
    "chr5": "NC_000005.10", "chr6": "NC_000006.12",
    "chr7": "NC_000007.14", "chr8": "NC_000008.11",
    "chr9": "NC_000009.12", "chr10": "NC_000010.11",
    "chr11": "NC_000011.10", "chr12": "NC_000012.12",
    "chr13": "NC_000013.11", "chr14": "NC_000014.9",
    "chr15": "NC_000015.10", "chr16": "NC_000016.10",
    "chr17": "NC_000017.11", "chr18": "NC_000018.10",
    "chr19": "NC_000019.10", "chr20": "NC_000020.11",
    "chr21": "NC_000021.9", "chr22": "NC_000022.11",
    "chrX": "NC_000023.11", "chrY": "NC_000024.10",
}
PRIMARY_CHROMS = list(CHR_TO_NC)
ENCODE = np.full(256, 4, dtype=np.uint8)
for base, value in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    ENCODE[ord(base)] = value
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def parse_attrs(text):
    attrs = {}
    for item in text.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        key, _, value = item.partition(" ")
        if value:
            attrs[key] = value.strip().strip('"')
    return attrs


def chromosome_split(seed):
    remaining = [chrom for chrom in PRIMARY_CHROMS if chrom != "chr1"]
    rng = random.Random(seed)
    rng.shuffle(remaining)
    n_train = round(0.8 * len(remaining))
    return sorted(remaining[:n_train], key=PRIMARY_CHROMS.index), sorted(
        remaining[n_train:], key=PRIMARY_CHROMS.index
    )


def parse_chess(gtf_path):
    """Parse primary-chromosome transcript spans, exons and CDS in one pass."""
    tx_meta = {}
    exons = defaultdict(list)
    cds = defaultdict(list)
    counts = defaultdict(int)
    primary = set(PRIMARY_CHROMS)
    with open(gtf_path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] not in primary:
                continue
            chrom, feature, strand = fields[0], fields[2], fields[6]
            if feature not in ("transcript", "exon", "CDS"):
                continue
            start0, end0 = int(fields[3]) - 1, int(fields[4])
            attrs = parse_attrs(fields[8])
            tid = attrs.get("transcript_id")
            if not tid:
                continue
            gid = attrs.get("gene_id", tid)
            previous = tx_meta.get(tid)
            if previous is None or feature == "transcript":
                tx_meta[tid] = (chrom, strand, gid, start0, end0)
            if feature == "exon":
                exons[tid].append((start0, end0))
            elif feature == "CDS":
                cds[tid].append((start0, end0))
            counts[feature] += 1
    print(
        f"Parsed CHESS: transcripts={counts['transcript']:,}, "
        f"exons={counts['exon']:,}, CDS={counts['CDS']:,}",
        flush=True,
    )
    return tx_meta, exons, cds


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def build_gene_regions(tx_meta, exons, cds):
    """Full transcript spans for genes having at least one CDS transcript."""
    gene_spans = {}
    for tid, segments in cds.items():
        if not segments or tid not in tx_meta:
            continue
        chrom, strand, gid, tx_start, tx_end = tx_meta[tid]
        if tid in exons and exons[tid]:
            tx_start = min(start for start, _ in exons[tid])
            tx_end = max(end for _, end in exons[tid])
        key = (chrom, strand, gid)
        if key not in gene_spans:
            gene_spans[key] = [tx_start, tx_end]
        else:
            gene_spans[key][0] = min(gene_spans[key][0], tx_start)
            gene_spans[key][1] = max(gene_spans[key][1], tx_end)
    regions = defaultdict(lambda: {"+": [], "-": []})
    for (chrom, strand, _), span in gene_spans.items():
        regions[chrom][strand].append(tuple(span))
    for chrom in regions:
        for strand in ("+", "-"):
            regions[chrom][strand] = merge_intervals(regions[chrom][strand])
    return regions, len(gene_spans)


def extract_sites(tx_meta, exons, cds, fasta_path):
    splice = defaultdict(lambda: {"+": set(), "-": set()})
    starts = defaultdict(lambda: {"+": set(), "-": set()})
    stops = defaultdict(lambda: {"+": set(), "-": set()})
    fasta = pysam.FastaFile(str(fasta_path))
    rejected_splice = rejected_start = rejected_stop = 0

    for tid, cds_segments in cds.items():
        if not cds_segments or tid not in tx_meta or tid not in exons:
            continue
        chrom_name, strand, _, _, _ = tx_meta[tid]
        nc = CHR_TO_NC[chrom_name]
        chrom_len = fasta.get_reference_length(nc)
        cds_lo = min(start for start, _ in cds_segments)
        cds_hi = max(end for _, end in cds_segments)

        exon_segments = sorted(set(exons[tid]))
        for (_, left_end), (right_start, _) in zip(exon_segments, exon_segments[1:]):
            if right_start <= left_end:
                continue
            donor_gpos = left_end
            acceptor_gpos = right_start - 2
            if not (cds_lo <= donor_gpos < cds_hi and cds_lo <= acceptor_gpos < cds_hi):
                continue
            donor2 = fasta.fetch(nc, donor_gpos, donor_gpos + 2).upper()
            acceptor2 = fasta.fetch(nc, acceptor_gpos, acceptor_gpos + 2).upper()
            canonical = (
                donor2 == "GT" and acceptor2 == "AG"
                if strand == "+"
                else donor2 == "CT" and acceptor2 == "AC"
            )
            if canonical:
                splice[chrom_name][strand].add((donor_gpos, acceptor_gpos))
            else:
                rejected_splice += 1

        segments = sorted(cds_segments)
        start_gpos = segments[0][0] if strand == "+" else segments[-1][1] - 3
        if 0 <= start_gpos <= chrom_len - 3:
            motif = fasta.fetch(nc, start_gpos, start_gpos + 3).upper()
            if (strand == "+" and motif == "ATG") or (strand == "-" and motif == "CAT"):
                starts[chrom_name][strand].add(start_gpos)
            else:
                rejected_start += 1

        # CHESS 3.1.3 CDS boundaries empirically use the RefSeq convention and
        # include the stop codon, matching the existing chr1 truth pipeline.
        stop_gpos = segments[-1][1] - 3 if strand == "+" else segments[0][0]
        if 0 <= stop_gpos <= chrom_len - 3:
            motif = fasta.fetch(nc, stop_gpos, stop_gpos + 3).upper()
            if (strand == "+" and motif in {"TAA", "TAG", "TGA"}) or (
                strand == "-" and motif in {"TTA", "CTA", "TCA"}
            ):
                stops[chrom_name][strand].add(stop_gpos)
            else:
                rejected_stop += 1
    fasta.close()
    print(
        f"Unique sites: splice_pairs={sum(len(v[s]) for v in splice.values() for s in ('+','-')):,}, "
        f"start={sum(len(v[s]) for v in starts.values() for s in ('+','-')):,}, "
        f"stop={sum(len(v[s]) for v in stops.values() for s in ('+','-')):,}",
        flush=True,
    )
    print(
        f"Rejected noncanonical/motif mismatch: splice={rejected_splice:,}, "
        f"start={rejected_start:,}, stop={rejected_stop:,}",
        flush=True,
    )
    return splice, starts, stops


def window_starts_for_regions(regions, chrom_len, window_size, stride):
    """Regular-grid windows whose central 5 kb intersects a coding-gene region."""
    center_margin = window_size // 4
    max_start = chrom_len - window_size
    selected = set()
    for region_start, region_end in regions:
        low = max(0, region_start - (window_size - center_margin))
        high = min(max_start, region_end - center_margin - 1)
        if high < low:
            midpoint = max(0, min(max_start, (region_start + region_end - window_size) // 2))
            selected.add((midpoint // stride) * stride)
            continue
        first = math.ceil(low / stride) * stride
        for start in range(first, high + 1, stride):
            if start + window_size - center_margin > region_start and start + center_margin < region_end:
                selected.add(start)
    return sorted(selected)


def _slice_positions(positions, low, high):
    left = bisect.bisect_left(positions, low)
    right = bisect.bisect_left(positions, high)
    return positions[left:right]


def fill_labels(win_start, donors, acceptors, starts, stops, strand, window_size):
    splice_labels = np.zeros(window_size, dtype=np.int64)
    ss_labels = np.zeros(window_size, dtype=np.int64)
    win_end = win_start + window_size

    if strand == "+":
        for donor in _slice_positions(donors, win_start, win_end):
            offset = donor - win_start
            if offset <= window_size - 2:
                splice_labels[offset] = 1
        for acceptor in _slice_positions(acceptors, win_start, win_end):
            offset = acceptor - win_start
            if offset <= window_size - 2:
                splice_labels[offset] = 2
    else:
        for donor in _slice_positions(donors, win_start - 1, win_end - 1):
            donor_rc = window_size - 1 - ((donor + 1) - win_start)
            if 0 <= donor_rc <= window_size - 2:
                splice_labels[donor_rc] = 2
        for acceptor in _slice_positions(acceptors, win_start - 1, win_end - 1):
            acceptor_rc = window_size - 1 - ((acceptor + 1) - win_start)
            if 0 <= acceptor_rc <= window_size - 2:
                splice_labels[acceptor_rc] = 1

    ss_low, ss_high = (win_start, win_end) if strand == "+" else (win_start - 2, win_end - 2)
    for gpos in _slice_positions(starts, ss_low, ss_high):
        offset = gpos - win_start if strand == "+" else window_size - 1 - ((gpos + 2) - win_start)
        if 0 <= offset <= window_size - 3:
            ss_labels[offset] = 1
    for gpos in _slice_positions(stops, ss_low, ss_high):
        offset = gpos - win_start if strand == "+" else window_size - 1 - ((gpos + 2) - win_start)
        if 0 <= offset <= window_size - 3:
            ss_labels[offset] = 2
    return splice_labels, ss_labels


def generate_split(output_dir, split, chroms, regions, splice, starts, stops, fasta_path):
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    fasta = pysam.FastaFile(str(fasta_path))
    summary = {"windows": 0, "with_any_label": 0, "with_start_stop": 0, "by_chrom": {}}
    for chrom in chroms:
        nc = CHR_TO_NC[chrom]
        chrom_len = fasta.get_reference_length(nc)
        chrom_row = {"windows": 0, "with_any_label": 0, "with_start_stop": 0}
        for strand in ("+", "-"):
            starts_grid = window_starts_for_regions(
                regions[chrom][strand], chrom_len, WINDOW_SIZE, STRIDE
            )
            splice_pairs = splice[chrom][strand]
            donor_sites = sorted(donor for donor, _ in splice_pairs)
            acceptor_sites = sorted(acceptor for _, acceptor in splice_pairs)
            start_sites = sorted(starts[chrom][strand])
            stop_sites = sorted(stops[chrom][strand])
            for win_start in starts_grid:
                seq = fasta.fetch(nc, win_start, win_start + WINDOW_SIZE).upper()
                if len(seq) != WINDOW_SIZE or seq.count("N") > WINDOW_SIZE // 2:
                    continue
                splice_labels, ss_labels = fill_labels(
                    win_start,
                    donor_sites,
                    acceptor_sites,
                    start_sites,
                    stop_sites,
                    strand,
                    WINDOW_SIZE,
                )
                if strand == "-":
                    seq = seq.translate(COMPLEMENT)[::-1]
                encoded = ENCODE[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
                has_any = bool(np.any(splice_labels) or np.any(ss_labels))
                has_ss = bool(np.any(ss_labels))
                filename = f"{chrom[3:]}_{'plus' if strand == '+' else 'minus'}_{win_start}.npz"
                np.savez_compressed(
                    split_dir / filename,
                    sequence=encoded,
                    labels=splice_labels,
                    start_stop_labels=ss_labels,
                    has_any_label=np.array(has_any, dtype=np.bool_),
                )
                chrom_row["windows"] += 1
                chrom_row["with_any_label"] += int(has_any)
                chrom_row["with_start_stop"] += int(has_ss)
        summary["by_chrom"][chrom] = chrom_row
        for key in ("windows", "with_any_label", "with_start_stop"):
            summary[key] += chrom_row[key]
        print(f"[{split}] {chrom}: {chrom_row}", flush=True)
    fasta.close()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise RuntimeError(f"Output exists and is non-empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_chroms, val_chroms = chromosome_split(args.seed)
    split_chroms = {"train": train_chroms, "val": val_chroms, "test": ["chr1"]}
    print(f"train ({len(train_chroms)}): {train_chroms}", flush=True)
    print(f"val ({len(val_chroms)}): {val_chroms}", flush=True)
    print("test: ['chr1']", flush=True)

    start_time = time.time()
    tx_meta, exons, cds = parse_chess(GTF_PATH)
    regions, gene_count = build_gene_regions(tx_meta, exons, cds)
    splice, starts, stops = extract_sites(tx_meta, exons, cds, FASTA_PATH)
    summaries = {}
    for split in ("train", "val", "test"):
        summaries[split] = generate_split(
            output_dir,
            split,
            split_chroms[split],
            regions,
            splice,
            starts,
            stops,
            FASTA_PATH,
        )

    manifest = {
        "annotation": str(GTF_PATH),
        "fasta": str(FASTA_PATH),
        "seed": args.seed,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "region_policy": "central_5kb_intersects_full_span_of_gene_with_CDS",
        "coding_gene_count": gene_count,
        "splits": split_chroms,
        "summary": summaries,
        "elapsed_seconds": time.time() - start_time,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Finished in {(time.time() - start_time) / 60:.1f} min: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
