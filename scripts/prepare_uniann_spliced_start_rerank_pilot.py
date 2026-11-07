#!/usr/bin/env python3
"""Prepare transcript-aware start reranking inputs from UniAnn exon chains.

UniAnn currently emits coding exons with its selected start at the terminal CDS
boundary.  To allow an alternative upstream ATG to win, the terminal 5' exon is
extended by a configurable genomic flank before exons are concatenated.
RefSeq starts are used only for evaluation and sample bookkeeping.
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
DEFAULT_REFSEQ = ROOT / "data/raw/human_refseq_grch38_p14.gff.gz"
DEFAULT_OUTPUT = ROOT / "results/human_spliced_start_pilot_v1/uniann_rerank_input"


@dataclass
class Tx:
    tid: str
    seqid: str
    strand: str
    start: int
    end: int
    exons: list[tuple[int, int]] = field(default_factory=list)


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def parse_attrs(text: str) -> dict[str, str]:
    result = {}
    for item in text.rstrip().split(";"):
        item = item.strip()
        if "=" in item:
            key, value = item.split("=", 1)
            result[key] = value
        elif " " in item:
            key, value = item.split(" ", 1)
            result[key] = value.strip('"')
    return result


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def parse_uniann(path: Path, seqid: str, strand: str) -> dict[str, Tx]:
    txs = {}
    pending = defaultdict(list)
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] != seqid or fields[6] != strand:
                continue
            feature = fields[2]
            values = parse_attrs(fields[8])
            start, end = int(fields[3]) - 1, int(fields[4])
            if feature in {"transcript", "mRNA"}:
                tid = values.get("ID") or values.get("transcript_id")
                if tid:
                    txs[tid] = Tx(tid, seqid, strand, start, end, pending.pop(tid, []))
            elif feature == "exon":
                for tid in (values.get("Parent") or values.get("transcript_id") or "").split(","):
                    if tid in txs:
                        txs[tid].exons.append((start, end))
                    elif tid:
                        pending[tid].append((start, end))
    return txs


def refseq_starts(path: Path, seqid: str, strand: str, fasta: pysam.FastaFile) -> set[int]:
    tx_meta = {}
    cds = defaultdict(list)
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[0] != seqid or fields[6] != strand:
                continue
            feature = fields[2]
            values = parse_attrs(fields[8])
            start, end = int(fields[3]) - 1, int(fields[4])
            if feature in {"mRNA", "transcript"}:
                tid = values.get("ID") or values.get("transcript_id")
                if tid:
                    tx_meta[tid] = True
            elif feature == "CDS":
                for tid in (values.get("Parent") or values.get("transcript_id") or "").split(","):
                    if tid:
                        cds[tid].append((start, end))
    result = set()
    for tid, segments in cds.items():
        if tid not in tx_meta or not segments:
            continue
        ordered = sorted(set(segments))
        site = ordered[0][0] if strand == "+" else ordered[-1][1] - 3
        motif = fasta.fetch(seqid, site, site + 3).upper()
        if (strand == "+" and motif == "ATG") or (strand == "-" and motif == "CAT"):
            result.add(site)
    return result


def extended_exons(tx: Tx, flank: int, chrom_length: int) -> list[tuple[int, int]]:
    ordered = sorted(set(tx.exons))
    if not ordered:
        return []
    if tx.strand == "+":
        ordered[0] = (max(0, ordered[0][0] - flank), ordered[0][1])
    else:
        ordered[-1] = (ordered[-1][0], min(chrom_length, ordered[-1][1] + flank))
    return ordered


def oriented_sequence(fasta: pysam.FastaFile, tx: Tx, exons: list[tuple[int, int]]) -> str:
    ordered = sorted(exons, reverse=tx.strand == "-")
    pieces = [fasta.fetch(tx.seqid, start, end).upper() for start, end in ordered]
    if tx.strand == "-":
        pieces = [reverse_complement(piece) for piece in pieces]
    return "".join(pieces)


def tx_coordinate(exons: list[tuple[int, int]], strand: str, site: int) -> int | None:
    offset = 0
    for start, end in sorted(exons, reverse=strand == "-"):
        anchor = site if strand == "+" else site + 2
        if start <= anchor < end:
            return offset + (anchor - start if strand == "+" else end - 1 - anchor)
        offset += end - start
    return None


def wrap(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[i:i + width] for i in range(0, len(sequence), width))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniann-gff", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--refseq", type=Path, default=DEFAULT_REFSEQ)
    parser.add_argument("--seqid", default="NC_000001.11")
    parser.add_argument("--strand", choices=["+", "-"], default="+")
    parser.add_argument("--flank", type=int, default=1000)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fasta = pysam.FastaFile(str(args.fasta))
    chrom_length = fasta.get_reference_length(args.seqid)
    txs = parse_uniann(args.uniann_gff, args.seqid, args.strand)
    truth = refseq_starts(args.refseq, args.seqid, args.strand, fasta)
    candidates = []
    with_truth = []
    for tx in txs.values():
        if len(set(tx.exons)) < 2:
            continue
        exons = extended_exons(tx, args.flank, chrom_length)
        sequence = oriented_sequence(fasta, tx, exons)
        supported = []
        for site in truth:
            coordinate = tx_coordinate(exons, tx.strand, site)
            if coordinate is not None and sequence[coordinate:coordinate + 3] == "ATG":
                supported.append((site, coordinate))
        row = (tx, exons, sequence, supported)
        candidates.append(row)
        if len(supported) == 1:
            with_truth.append(row)

    # Deduplicate by truth site, keeping the chain whose original UniAnn start
    # is closest. This prevents isoform multiplicity from inflating accuracy.
    per_truth = {}
    for row in with_truth:
        tx, _, _, supported = row
        truth_site = supported[0][0]
        uniann_start = tx.start if tx.strand == "+" else tx.end - 3
        key = (abs(uniann_start - truth_site), tx.tid)
        if truth_site not in per_truth or key < per_truth[truth_site][0]:
            per_truth[truth_site] = (key, row)
    rows = [value[1] for _, value in sorted(per_truth.items())]
    if len(rows) > args.count:
        indices = [round(i * (len(rows) - 1) / (args.count - 1)) for i in range(args.count)]
        rows = [rows[i] for i in indices]

    metadata = []
    fasta_path = args.output_dir / "uniann_spliced_extended.fa"
    with fasta_path.open("w") as output:
        for index, (tx, exons, sequence, supported) in enumerate(rows, 1):
            truth_site, truth_coordinate = supported[0]
            current_site = tx.start if tx.strand == "+" else tx.end - 3
            current_coordinate = tx_coordinate(exons, tx.strand, current_site)
            entry = f"uniann{index:04d}"
            output.write(f">{entry}\n{wrap(sequence)}\n")
            metadata.append({
                "entry_line": entry,
                "transcript_id": tx.tid,
                "seqid": tx.seqid,
                "strand": tx.strand,
                "true_start_genomic_0based": truth_site,
                "true_atg_position_1based": truth_coordinate + 1,
                "uniann_start_genomic_0based": current_site,
                "uniann_atg_position_1based": (current_coordinate + 1) if current_coordinate is not None else "",
                "uniann_start_exact": int(current_site == truth_site),
                "sequence_length": len(sequence),
                "atg_candidates": sequence.count("ATG"),
                "exon_count": len(exons),
                "extended_exons_0based_half_open": json.dumps(exons),
            })
    fasta.close()
    with (args.output_dir / "metadata.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(metadata)
    manifest = {
        "uniann_gff": str(args.uniann_gff.resolve()),
        "seqid": args.seqid,
        "strand": args.strand,
        "five_prime_flank": args.flank,
        "uniann_transcripts": len(txs),
        "multi_exon_transcripts": len(candidates),
        "chains_with_exactly_one_embedded_refseq_start": len(with_truth),
        "unique_refseq_starts_covered": len(per_truth),
        "selected": len(rows),
        "selected_current_uniann_exact": sum(row[0].start == row[3][0][0] for row in rows) if args.strand == "+" else sum(row[0].end - 3 == row[3][0][0] for row in rows),
        "selected_atg_candidates": sum(row[2].count("ATG") for row in rows),
        "note": "RefSeq is used only to define evaluation positives and select evaluable chains.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
