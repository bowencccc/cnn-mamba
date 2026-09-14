#!/usr/bin/env python3
"""Generate Dmel coding-region windows for CNN--Mamba training."""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pysam

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "droso_window_grid.json"
EVIANN_GFF = ROOT / "data" / "raw" / "dmel_eviann_pseudo_label.gff"
REFERENCE_GTF = ROOT / "data" / "raw" / "dmel_reference.gtf"
FASTA = ROOT / "data" / "raw" / "dmel_genome.fa"
SPLITS = {
    "train": ["2L", "2R", "3R", "4", "Y"],
    "val": ["3L"],
    "test": ["X"],
}
CHROMS = {
    "2L": "NT_033779.5", "2R": "NT_033778.4", "3L": "NT_037436.4",
    "3R": "NT_033777.3", "4": "NC_004353.4", "X": "NC_004354.4",
    "Y": "NC_024512.1",
}
SITE_TYPES = ("donor", "acceptor", "start", "stop")
ENCODE = np.full(256, 4, dtype=np.uint8)
for _base, _value in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    ENCODE[ord(_base)] = _value
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def gff_attrs(text: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in text.strip().split(";") if "=" in item)


def gtf_attrs(text: str) -> dict[str, str]:
    result = {}
    for item in text.strip().split(";"):
        key, _, value = item.strip().partition(" ")
        if value:
            result[key] = value.strip().strip('"')
    return result


def parse_eviann(path: Path):
    accessions = set(CHROMS.values())
    meta, exons, cds = {}, defaultdict(list), defaultdict(list)
    with path.open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 9 or fields[0] not in accessions:
                continue
            attrs = gff_attrs(fields[8])
            start0, end0 = int(fields[3]) - 1, int(fields[4])
            if fields[2] == "mRNA" and attrs.get("ID"):
                meta[attrs["ID"]] = (fields[0], fields[6])
            elif fields[2] in ("exon", "CDS"):
                target = exons if fields[2] == "exon" else cds
                for tid in attrs.get("Parent", "").split(","):
                    if tid:
                        target[tid].append((start0, end0))
    return meta, exons, cds, None


def parse_reference(path: Path):
    accessions = set(CHROMS.values())
    meta, exons, cds = {}, defaultdict(list), defaultdict(list)
    explicit = {"start": set(), "stop": set()}
    with path.open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 9 or fields[0] not in accessions:
                continue
            feature = fields[2]
            if feature not in ("transcript", "exon", "CDS", "start_codon", "stop_codon"):
                continue
            attrs = gtf_attrs(fields[8])
            tid = attrs.get("transcript_id")
            if not tid:
                continue
            chrom, strand = fields[0], fields[6]
            meta[tid] = (chrom, strand)
            start0, end0 = int(fields[3]) - 1, int(fields[4])
            if feature == "exon":
                exons[tid].append((start0, end0))
            elif feature == "CDS":
                cds[tid].append((start0, end0))
            elif feature in ("start_codon", "stop_codon") and end0 - start0 == 3:
                explicit["start" if feature == "start_codon" else "stop"].add(
                    (chrom, strand, start0)
                )
    return meta, exons, cds, explicit


def extract_sites(meta, exons, cds, explicit=None):
    sites = {name: set() for name in SITE_TYPES}
    rejected = defaultdict(int)
    fasta = pysam.FastaFile(str(FASTA))
    for tid, segments in cds.items():
        if not segments or tid not in meta or tid not in exons:
            continue
        chrom, strand = meta[tid]
        cds_lo = min(x[0] for x in segments)
        cds_hi = max(x[1] for x in segments)
        ordered_exons = sorted(set(exons[tid]))
        for (_, left_end), (right_start, _) in zip(ordered_exons, ordered_exons[1:]):
            if right_start <= left_end:
                continue
            low, high = left_end, right_start - 2
            if not (cds_lo <= low < cds_hi and cds_lo <= high < cds_hi):
                continue
            low2 = fasta.fetch(chrom, low, low + 2).upper()
            high2 = fasta.fetch(chrom, high, high + 2).upper()
            canonical = ((low2, high2) == ("GT", "AG")) if strand == "+" else ((low2, high2) == ("CT", "AC"))
            if not canonical:
                rejected["splice"] += 1
                continue
            donor, acceptor = (low, high) if strand == "+" else (high, low)
            sites["donor"].add((chrom, strand, donor))
            sites["acceptor"].add((chrom, strand, acceptor))
        if explicit is None:
            ordered = sorted(segments)
            start = ordered[0][0] if strand == "+" else ordered[-1][1] - 3
            stop = ordered[-1][1] - 3 if strand == "+" else ordered[0][0]
            for name, position in (("start", start), ("stop", stop)):
                motif = fasta.fetch(chrom, position, position + 3).upper()
                valid = (
                    motif == "ATG" if name == "start" and strand == "+" else
                    motif == "CAT" if name == "start" else
                    motif in {"TAA", "TAG", "TGA"} if strand == "+" else
                    motif in {"TTA", "CTA", "TCA"}
                )
                if valid:
                    sites[name].add((chrom, strand, position))
                else:
                    rejected[name] += 1
    if explicit is not None:
        for name in ("start", "stop"):
            for chrom, strand, position in explicit[name]:
                motif = fasta.fetch(chrom, position, position + 3).upper()
                valid = (
                    motif == "ATG" if name == "start" and strand == "+" else
                    motif == "CAT" if name == "start" else
                    motif in {"TAA", "TAG", "TGA"} if strand == "+" else
                    motif in {"TTA", "CTA", "TCA"}
                )
                if valid:
                    sites[name].add((chrom, strand, position))
                else:
                    rejected[name] += 1
    fasta.close()
    return sites, dict(rejected)


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [tuple(row) for row in merged]


def coding_regions(meta, exons, cds):
    regions = defaultdict(lambda: {"+": [], "-": []})
    for tid, segments in cds.items():
        if not segments or tid not in meta:
            continue
        chrom, strand = meta[tid]
        span = exons.get(tid) or segments
        regions[chrom][strand].append((min(x[0] for x in span), max(x[1] for x in span)))
    for chrom in regions:
        for strand in ("+", "-"):
            regions[chrom][strand] = merge_intervals(regions[chrom][strand])
    return regions


def window_starts(regions, chrom_len: int, window: int, stride: int):
    selected = set()
    max_start = chrom_len - window
    if max_start < 0:
        return []
    for region_start, region_end in regions:
        low = max(0, region_start - window + 1)
        high = min(max_start, region_end - 1)
        if high >= low:
            first = math.ceil(low / stride) * stride
            selected.update(range(first, high + 1, stride))
    return sorted(selected)


def index_sites(sites):
    result = {name: defaultdict(lambda: {"+": [], "-": []}) for name in SITE_TYPES}
    temporary = {name: defaultdict(lambda: {"+": set(), "-": set()}) for name in SITE_TYPES}
    for name in SITE_TYPES:
        for chrom, strand, position in sites[name]:
            temporary[name][chrom][strand].add(position)
        for chrom in temporary[name]:
            for strand in ("+", "-"):
                result[name][chrom][strand] = sorted(temporary[name][chrom][strand])
    return result


def labels_for_window(start: int, strand: str, indexed, window: int):
    splice = np.zeros(window, dtype=np.int8)
    start_stop = np.zeros(window, dtype=np.int8)
    for name, target, cls, length in (
        ("donor", splice, 1, 2), ("acceptor", splice, 2, 2),
        ("start", start_stop, 1, 3), ("stop", start_stop, 2, 3),
    ):
        positions = indexed[name]
        left = bisect.bisect_left(positions, start)
        right = bisect.bisect_right(positions, start + window - length)
        for position in positions[left:right]:
            offset = position - start if strand == "+" else window - length - (position - start)
            target[offset] = cls
    return splice, start_stop


def generate_cell(output: Path, cell: dict, regions, eviann, reference) -> dict:
    window, stride = int(cell["window_bp"]), int(cell["stride_bp"])
    summary = {}
    fasta = pysam.FastaFile(str(FASTA))
    for split, chrom_names in SPLITS.items():
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_summary = {"windows": 0, "by_chrom": {}, "eviann": defaultdict(int), "reference": defaultdict(int)}
        for name in chrom_names:
            chrom = CHROMS[name]
            count = 0
            for strand in ("+", "-"):
                starts = window_starts(regions[chrom][strand], fasta.get_reference_length(chrom), window, stride)
                e_local = {site: eviann[site][chrom][strand] for site in SITE_TYPES}
                r_local = {site: reference[site][chrom][strand] for site in SITE_TYPES}
                for start in starts:
                    text = fasta.fetch(chrom, start, start + window).upper()
                    if len(text) != window or text.count("N") > window // 2:
                        continue
                    e_splice, e_ss = labels_for_window(start, strand, e_local, window)
                    r_splice, r_ss = labels_for_window(start, strand, r_local, window)
                    if strand == "-":
                        text = text.translate(COMPLEMENT)[::-1]
                    sequence = ENCODE[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]
                    filename = f"{name}_{'plus' if strand == '+' else 'minus'}_{start}.npz"
                    np.savez_compressed(
                        split_dir / filename, sequence=sequence, labels=e_splice,
                        start_stop_labels=e_ss, chess_labels=r_splice,
                        chess_start_stop_labels=r_ss,
                    )
                    count += 1
                    split_summary["windows"] += 1
                    owned_left = (window - stride) // 2
                    owned_right = owned_left + stride
                    for site, ea, ra, cls in (
                        ("donor", e_splice, r_splice, 1), ("acceptor", e_splice, r_splice, 2),
                        ("start", e_ss, r_ss, 1), ("stop", e_ss, r_ss, 2),
                    ):
                        split_summary["eviann"][site] += int(np.sum(ea[owned_left:owned_right] == cls))
                        split_summary["reference"][site] += int(np.sum(ra[owned_left:owned_right] == cls))
            split_summary["by_chrom"][name] = count
            print(f"[{cell['id']}:{split}] {name}: {count:,}", flush=True)
        summary[split] = {
            **split_summary, "eviann": dict(split_summary["eviann"]),
            "reference": dict(split_summary["reference"]),
        }
    fasta.close()
    manifest = {
        "schema": "droso_window_cell_v1", "cell": cell["id"],
        "window_size": window, "stride": stride, "adjacent_overlap_bp": window - stride,
        "region_overlap_bp": window,
        "region_policy": "full_window_intersects_EviAnn_transcript_span_with_CDS",
        "owned_local_interval": [(window - stride) // 2, (window - stride) // 2 + stride],
        "splits": SPLITS, "accessions": CHROMS, "summary": summary,
        "sources": {
            "fasta": {"path": str(FASTA), "sha256": sha256(FASTA)},
            "eviann": {"path": str(EVIANN_GFF), "sha256": sha256(EVIANN_GFF)},
            "reference": {"path": str(REFERENCE_GTF), "sha256": sha256(REFERENCE_GTF)},
        },
    }
    atomic_json(output / "split_manifest.json", manifest)
    return manifest


def main() -> None:
    global CONFIG, EVIANN_GFF, REFERENCE_GTF, FASTA
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--fasta", type=Path, default=FASTA)
    parser.add_argument("--eviann-gff", type=Path, default=EVIANN_GFF)
    parser.add_argument("--reference-gtf", type=Path, default=REFERENCE_GTF)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    CONFIG = args.config.resolve()
    FASTA = args.fasta.resolve()
    EVIANN_GFF = args.eviann_gff.resolve()
    REFERENCE_GTF = args.reference_gtf.resolve()
    for path in (CONFIG, FASTA, EVIANN_GFF, REFERENCE_GTF):
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")
    config = json.loads(CONFIG.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("Parsing annotations once...", flush=True)
    e_meta, e_exons, e_cds, _ = parse_eviann(EVIANN_GFF)
    r_meta, r_exons, r_cds, explicit = parse_reference(REFERENCE_GTF)
    regions = coding_regions(e_meta, e_exons, e_cds)
    e_sites, e_rejected = extract_sites(e_meta, e_exons, e_cds)
    r_sites, r_rejected = extract_sites(r_meta, r_exons, r_cds, explicit)
    e_index, r_index = index_sites(e_sites), index_sites(r_sites)
    completed = []
    for cell in config["cells"]:
        cell_dir = args.output_root / cell["id"]
        manifest_path = cell_dir / "split_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            expected = sum(int(manifest["summary"][s]["windows"]) for s in SPLITS)
            actual = sum(1 for s in SPLITS for _ in (cell_dir / s).glob("*.npz"))
            if actual != expected:
                raise RuntimeError(f"partial cached cell {cell['id']}: {actual}/{expected}")
            print(f"validated cached {cell['id']}: {actual:,} windows", flush=True)
        else:
            if cell_dir.exists() and any(cell_dir.iterdir()):
                raise RuntimeError(f"refusing partial non-empty cell: {cell_dir}")
            manifest = generate_cell(cell_dir, cell, regions, e_index, r_index)
        completed.append({"cell": cell["id"], "manifest": str(manifest_path.resolve()), "windows": sum(manifest["summary"][s]["windows"] for s in SPLITS)})
    atomic_json(args.output_root / ".complete.json", {
        "schema": "droso_window_generation_complete_v1", "cells": completed,
        "eviann_rejected": e_rejected, "reference_rejected": r_rejected,
        "elapsed_seconds_current_invocation": time.time() - started,
    })
    print(f"complete: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
