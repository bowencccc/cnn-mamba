#!/usr/bin/env python3
"""Prepare paired spliced and intron-retaining Human transcripts for NetStart.

The two FASTA entries for a transcript contain exactly the same annotated
RefSeq start codon.  One entry concatenates exons in transcription order; the
other contains the full genomic transcript span in transcription orientation.
This makes the effect of removing introns directly measurable.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pysam


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FASTA = ROOT / "data/raw/human_grch38.fa"
DEFAULT_GFF = ROOT / "data/raw/human_refseq_grch38_p14.gff.gz"
DEFAULT_OUTPUT = ROOT / "results/human_spliced_start_pilot_v1/input"


@dataclass
class Transcript:
    tid: str
    gene: str
    seqid: str
    strand: str
    start: int
    end: int
    mane: bool = False
    exons: list[tuple[int, int]] = field(default_factory=list)
    cds: list[tuple[int, int]] = field(default_factory=list)


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def attrs(text: str) -> dict[str, str]:
    result = {}
    for item in text.rstrip().split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
    return result


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def parse_refseq(path: Path, seqid: str) -> dict[str, Transcript]:
    transcripts: dict[str, Transcript] = {}
    pending_exons: dict[str, list[tuple[int, int]]] = defaultdict(list)
    pending_cds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] != seqid:
                continue
            feature = fields[2]
            if feature not in {"mRNA", "transcript", "exon", "CDS"}:
                continue
            values = attrs(fields[8])
            start, end = int(fields[3]) - 1, int(fields[4])
            if feature in {"mRNA", "transcript"}:
                tid = values.get("ID")
                if not tid:
                    continue
                tx = Transcript(
                    tid=tid,
                    gene=values.get("Parent", tid),
                    seqid=seqid,
                    strand=fields[6],
                    start=start,
                    end=end,
                    mane="MANE Select" in values.get("tag", ""),
                )
                tx.exons.extend(pending_exons.pop(tid, []))
                tx.cds.extend(pending_cds.pop(tid, []))
                transcripts[tid] = tx
            else:
                target = pending_exons if feature == "exon" else pending_cds
                for tid in values.get("Parent", "").split(","):
                    if not tid:
                        continue
                    tx = transcripts.get(tid)
                    if tx is None:
                        target[tid].append((start, end))
                    else:
                        (tx.exons if feature == "exon" else tx.cds).append((start, end))
                        if "MANE Select" in values.get("tag", ""):
                            tx.mane = True
    return transcripts


def start_site(tx: Transcript) -> int:
    segments = sorted(set(tx.cds))
    return segments[0][0] if tx.strand == "+" else segments[-1][1] - 3


def spliced_sequence(fasta: pysam.FastaFile, tx: Transcript) -> str:
    ordered = sorted(set(tx.exons), reverse=tx.strand == "-")
    pieces = [fasta.fetch(tx.seqid, start, end).upper() for start, end in ordered]
    if tx.strand == "-":
        pieces = [reverse_complement(piece) for piece in pieces]
    return "".join(pieces)


def spliced_coordinate(tx: Transcript, motif_start: int) -> int | None:
    """Return zero-based transcript coordinate of the A in the oriented ATG."""
    offset = 0
    for start, end in sorted(set(tx.exons), reverse=tx.strand == "-"):
        anchor = motif_start if tx.strand == "+" else motif_start + 2
        if start <= anchor < end:
            within = anchor - start if tx.strand == "+" else end - 1 - anchor
            return offset + within
        offset += end - start
    return None


def genomic_sequence(fasta: pysam.FastaFile, tx: Transcript) -> str:
    sequence = fasta.fetch(tx.seqid, tx.start, tx.end).upper()
    return sequence if tx.strand == "+" else reverse_complement(sequence)


def genomic_coordinate(tx: Transcript, motif_start: int) -> int:
    return motif_start - tx.start if tx.strand == "+" else tx.end - motif_start - 3


def wrap(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[i:i + width] for i in range(0, len(sequence), width))


def choose_transcripts(
    transcripts: dict[str, Transcript], fasta: pysam.FastaFile,
    count: int, max_span: int,
) -> tuple[list[Transcript], dict[str, int]]:
    rejected: dict[str, int] = defaultdict(int)
    eligible: list[Transcript] = []
    for tx in transcripts.values():
        if tx.strand not in {"+", "-"} or not tx.cds or len(set(tx.exons)) < 2:
            rejected["not_multiexon_coding"] += 1
            continue
        if tx.end - tx.start > max_span:
            rejected["span_too_long"] += 1
            continue
        site = start_site(tx)
        genomic_motif = fasta.fetch(tx.seqid, site, site + 3).upper()
        expected = "ATG" if tx.strand == "+" else "CAT"
        if genomic_motif != expected:
            rejected["noncanonical_start"] += 1
            continue
        sequence = spliced_sequence(fasta, tx)
        coordinate = spliced_coordinate(tx, site)
        if coordinate is None or sequence[coordinate:coordinate + 3] != "ATG":
            rejected["mapping_failure"] += 1
            continue
        eligible.append(tx)

    # One representative per gene: MANE Select first, then curated NM before XM,
    # then the shortest span to keep the intron-retaining control tractable.
    per_gene: dict[str, Transcript] = {}
    def rank(tx: Transcript):
        return (not tx.mane, "NM_" not in tx.tid, tx.end - tx.start, tx.tid)
    for tx in eligible:
        current = per_gene.get(tx.gene)
        if current is None or rank(tx) < rank(current):
            per_gene[tx.gene] = tx
    representatives = sorted(per_gene.values(), key=lambda tx: (tx.start, tx.end, tx.tid))
    if len(representatives) <= count:
        return representatives, dict(rejected)
    # Deterministic approximately uniform sampling across chr1 avoids selecting
    # only its p arm while preserving reproducibility.
    indices = [round(i * (len(representatives) - 1) / (count - 1)) for i in range(count)]
    return [representatives[i] for i in indices], dict(rejected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--gff", type=Path, default=DEFAULT_GFF)
    parser.add_argument("--seqid", default="NC_000001.11")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--max-span", type=int, default=30_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.count < 2:
        parser.error("--count must be at least 2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta = pysam.FastaFile(str(args.fasta))
    transcripts = parse_refseq(args.gff, args.seqid)
    selected, rejected = choose_transcripts(transcripts, fasta, args.count, args.max_span)

    metadata = []
    handles = {
        "spliced": (args.output_dir / "refseq_spliced.fa").open("w"),
        "genomic": (args.output_dir / "refseq_genomic.fa").open("w"),
    }
    try:
        for index, tx in enumerate(selected, 1):
            site = start_site(tx)
            sequences = {
                "spliced": spliced_sequence(fasta, tx),
                "genomic": genomic_sequence(fasta, tx),
            }
            positions = {
                "spliced": spliced_coordinate(tx, site),
                "genomic": genomic_coordinate(tx, site),
            }
            for mode in ("spliced", "genomic"):
                entry = f"pair{index:04d}__{mode}"
                sequence = sequences[mode]
                position = positions[mode]
                assert position is not None and sequence[position:position + 3] == "ATG"
                handles[mode].write(f">{entry}\n{wrap(sequence)}\n")
                metadata.append({
                    "entry_line": entry,
                    "pair_id": f"pair{index:04d}",
                    "mode": mode,
                    "transcript_id": tx.tid,
                    "gene_id": tx.gene,
                    "seqid": tx.seqid,
                    "strand": tx.strand,
                    "transcript_start_0based": tx.start,
                    "transcript_end_0based_exclusive": tx.end,
                    "true_start_genomic_0based": site,
                    "true_atg_position_1based": position + 1,
                    "sequence_length": len(sequence),
                    "atg_candidates": sequence.count("ATG"),
                    "exon_count": len(set(tx.exons)),
                    "mane_select": int(tx.mane),
                    "exons_0based_half_open": json.dumps(sorted(set(tx.exons))),
                })
    finally:
        for handle in handles.values():
            handle.close()
        fasta.close()

    with (args.output_dir / "metadata.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(metadata)
    manifest = {
        "reference": str(args.gff.resolve()),
        "genome": str(args.fasta.resolve()),
        "seqid": args.seqid,
        "requested_pairs": args.count,
        "selected_pairs": len(selected),
        "max_genomic_span": args.max_span,
        "selection": "one transcript per gene; MANE then NM then shortest span; uniform across chromosome",
        "rejected": rejected,
        "spliced_candidates": sum(r["atg_candidates"] for r in metadata if r["mode"] == "spliced"),
        "genomic_candidates": sum(r["atg_candidates"] for r in metadata if r["mode"] == "genomic"),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
