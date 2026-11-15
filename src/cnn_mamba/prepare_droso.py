#!/usr/bin/env python3
"""Generate Dmel coding-region windows for CNN--Mamba training."""

from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pysam

from .window_storage import WindowStore, WindowWriter

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


def open_text(path: Path):
    """Open plain-text or gzip-compressed annotation input."""
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


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


def cds_length(segments) -> int:
    """Return transcript CDS length after removing duplicate intervals."""
    return sum(end - start for start, end in set(segments) if end > start)


def select_eviann_transcripts(meta, exons, cds, annotations):
    """Select one high-confidence coding transcript per EviAnn locus.

    Selection is intentionally lexicographic rather than score based:

    1. longest CDS among ``evidence=complete`` and ``class=`` ``=``;
    2. if absent, longest CDS among ``evidence=complete`` and ``class=k``;
    3. otherwise the locus is excluded from label generation.

    Ties are resolved by transcript ID so repeated preprocessing is stable.
    """
    per_locus = defaultdict(list)
    for tid in meta:
        record = annotations.get(tid, {})
        locus = record.get("locus") or tid
        per_locus[locus].append(tid)

    selected = set()
    selected_by_class = {"=": 0, "k": 0}
    skipped_no_eligible = 0
    for locus, tids in per_locus.items():
        complete = [
            tid for tid in tids
            if annotations.get(tid, {}).get("evidence", "").lower() == "complete"
        ]
        exact = [
            tid for tid in complete
            if annotations.get(tid, {}).get("class") == "=" and cds.get(tid)
        ]
        fallback = [
            tid for tid in complete
            if annotations.get(tid, {}).get("class") == "k" and cds.get(tid)
        ]
        candidates = exact or fallback
        if not candidates:
            skipped_no_eligible += 1
            continue
        # A lexical tie break is deterministic and does not change the stated
        # biological priority, which is CDS length only within a class tier.
        chosen = sorted(candidates, key=lambda tid: (-cds_length(cds[tid]), tid))[0]
        selected.add(chosen)
        selected_by_class[annotations[chosen]["class"]] += 1

    filtered_meta = {tid: meta[tid] for tid in selected}
    filtered_exons = defaultdict(
        list, {tid: exons[tid] for tid in selected if tid in exons}
    )
    filtered_cds = defaultdict(
        list, {tid: cds[tid] for tid in selected if tid in cds}
    )
    summary = {
        "mode": "longest_complete_class_equal_else_k_per_locus",
        "input_transcripts": len(meta),
        "input_loci": len(per_locus),
        "selected_transcripts": len(selected),
        "selected_loci": len(selected),
        "selected_complete_class_equal": selected_by_class["="],
        "selected_complete_class_k": selected_by_class["k"],
        "skipped_loci_without_complete_equal_or_k": skipped_no_eligible,
        "cds_length_definition": "sum of unique CDS interval lengths",
        "tie_break": "lexicographically smallest transcript ID",
    }
    return filtered_meta, filtered_exons, filtered_cds, summary


def select_complete_equal_else_k_with_fallback(meta, exons, cds, annotations):
    """Prefer complete =/k transcripts while retaining every coding locus."""
    per_locus = defaultdict(list)
    for tid in meta:
        record = annotations.get(tid, {})
        locus = record.get("locus") or tid
        per_locus[locus].append(tid)

    selected = set()
    selected_by_tier = {"complete_equal": 0, "complete_k": 0, "fallback": 0}
    skipped_without_cds = 0
    for tids in per_locus.values():
        coding = [tid for tid in tids if cds.get(tid)]
        if not coding:
            skipped_without_cds += 1
            continue
        complete = [
            tid for tid in coding
            if annotations.get(tid, {}).get("evidence", "").lower() == "complete"
        ]
        exact = [
            tid for tid in complete
            if annotations.get(tid, {}).get("class") == "="
        ]
        contained = [
            tid for tid in complete
            if annotations.get(tid, {}).get("class") == "k"
        ]
        if exact:
            candidates, tier = exact, "complete_equal"
        elif contained:
            candidates, tier = contained, "complete_k"
        else:
            candidates, tier = coding, "fallback"
        chosen = sorted(candidates, key=lambda tid: (-cds_length(cds[tid]), tid))[0]
        selected.add(chosen)
        selected_by_tier[tier] += 1

    filtered_meta = {tid: meta[tid] for tid in selected}
    filtered_exons = defaultdict(
        list, {tid: exons[tid] for tid in selected if tid in exons}
    )
    filtered_cds = defaultdict(
        list, {tid: cds[tid] for tid in selected if tid in cds}
    )
    summary = {
        "mode": "complete_equal_else_k_with_longest_cds_fallback_per_locus",
        "input_transcripts": len(meta),
        "input_loci": len(per_locus),
        "selected_transcripts": len(selected),
        "selected_loci": len(selected),
        "selected_complete_class_equal": selected_by_tier["complete_equal"],
        "selected_complete_class_k": selected_by_tier["complete_k"],
        "selected_longest_cds_fallback": selected_by_tier["fallback"],
        "skipped_loci_without_cds": skipped_without_cds,
        "cds_length_definition": "sum of unique CDS interval lengths",
        "tie_break": "lexicographically smallest transcript ID",
    }
    return filtered_meta, filtered_exons, filtered_cds, summary


def select_longest_cds_per_locus(meta, exons, cds, annotations):
    """Select the transcript with the longest spliced CDS in every locus."""
    per_locus = defaultdict(list)
    for tid in meta:
        record = annotations.get(tid, {})
        locus = record.get("locus") or tid
        per_locus[locus].append(tid)

    selected = set()
    skipped_without_cds = 0
    for tids in per_locus.values():
        candidates = [tid for tid in tids if cds.get(tid)]
        if not candidates:
            skipped_without_cds += 1
            continue
        chosen = sorted(candidates, key=lambda tid: (-cds_length(cds[tid]), tid))[0]
        selected.add(chosen)

    filtered_meta = {tid: meta[tid] for tid in selected}
    filtered_exons = defaultdict(
        list, {tid: exons[tid] for tid in selected if tid in exons}
    )
    filtered_cds = defaultdict(
        list, {tid: cds[tid] for tid in selected if tid in cds}
    )
    summary = {
        "mode": "longest_cds_per_locus",
        "input_transcripts": len(meta),
        "input_loci": len(per_locus),
        "selected_transcripts": len(selected),
        "selected_loci": len(selected),
        "skipped_loci_without_cds": skipped_without_cds,
        "cds_length_definition": "sum of unique CDS interval lengths",
        "tie_break": "lexicographically smallest transcript ID",
    }
    return filtered_meta, filtered_exons, filtered_cds, summary


def parse_eviann(path: Path, transcript_selection: str = "all"):
    accessions = set(CHROMS.values())
    meta, exons, cds = {}, defaultdict(list), defaultdict(list)
    annotations = {}
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 9 or fields[0] not in accessions:
                continue
            attrs = gff_attrs(fields[8])
            start0, end0 = int(fields[3]) - 1, int(fields[4])
            if fields[2] == "mRNA" and attrs.get("ID"):
                tid = attrs["ID"]
                meta[tid] = (fields[0], fields[6])
                annotations[tid] = {
                    "locus": attrs.get("Parent") or attrs.get("geneID") or tid,
                    "evidence": attrs.get("evidence", attrs.get("Evidence", "")),
                    "class": attrs.get(
                        "class", attrs.get("class_code", attrs.get("Class", ""))
                    ),
                }
            elif fields[2] in ("exon", "CDS"):
                target = exons if fields[2] == "exon" else cds
                for tid in attrs.get("Parent", "").split(","):
                    if tid:
                        target[tid].append((start0, end0))
    if transcript_selection == "all":
        return meta, exons, cds, {
            "mode": "all_transcripts",
            "input_transcripts": len(meta),
            "selected_transcripts": len(meta),
        }
    if transcript_selection == "longest-cds-per-locus":
        return select_longest_cds_per_locus(meta, exons, cds, annotations)
    if transcript_selection == "complete-equal-else-k-with-fallback":
        return select_complete_equal_else_k_with_fallback(
            meta, exons, cds, annotations
        )
    if transcript_selection == "longest-complete-equal-else-k":
        return select_eviann_transcripts(meta, exons, cds, annotations)
    raise ValueError(f"unknown EviAnn transcript selection: {transcript_selection}")


def parse_reference(path: Path):
    accessions = set(CHROMS.values())
    meta, exons, cds = {}, defaultdict(list), defaultdict(list)
    explicit = {"start": set(), "stop": set()}
    with open_text(path) as handle:
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


def parse_cds_phase_records(path: Path, allowed_transcripts=None):
    """Collect CDS intervals and their GFF/GTF phase without changing site parsing."""
    accessions = set(CHROMS.values())
    records = defaultdict(lambda: {"+": [], "-": []})
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip().split("\t")
            if (
                len(fields) < 9 or fields[0] not in accessions
                or fields[2] != "CDS" or fields[6] not in {"+", "-"}
                or fields[7] not in {"0", "1", "2"}
            ):
                continue
            if allowed_transcripts is not None:
                attrs = (
                    gff_attrs(fields[8]) if "=" in fields[8]
                    else gtf_attrs(fields[8])
                )
                parents = attrs.get("Parent", attrs.get("transcript_id", "")).split(",")
                if not any(tid in allowed_transcripts for tid in parents):
                    continue
            records[fields[0]][fields[6]].append(
                (int(fields[3]) - 1, int(fields[4]), int(fields[7]))
            )
    return records


def build_phase_track(records, chrom_len: int, strand: str):
    """Return -1 unknown, -2 conflicting, or codon offsets 0/1/2."""
    track = np.full(chrom_len, -1, dtype=np.int8)
    for start, end, phase in records:
        length = end - start
        initial = (3 - phase) % 3
        offsets = np.arange(length, dtype=np.int64)
        if strand == "-":
            offsets = offsets[::-1]
        values = ((initial + offsets) % 3).astype(np.int8)
        view = track[start:end]
        unknown = view == -1
        conflict = (view >= 0) & (view != values)
        view[unknown] = values[unknown]
        view[conflict] = -2
    return track


def build_cds_track(records, chrom_len: int):
    """Return the union of all annotated CDS intervals, independent of phase."""
    track = np.zeros(chrom_len, dtype=np.bool_)
    for start, end, _phase in records:
        track[start:end] = True
    return track


def build_cds_boundary_mask(cds_track, radius: int):
    """Mark bases within ``radius`` bp of chromosome-level CDS transitions."""
    if radius < 1:
        raise ValueError("CDS boundary radius must be positive")
    cds_track = np.asarray(cds_track, dtype=np.bool_)
    mask = np.zeros_like(cds_track)
    transitions = np.flatnonzero(cds_track[1:] != cds_track[:-1]) + 1
    for boundary in transitions:
        mask[max(0, boundary - radius):min(len(mask), boundary + radius)] = True
    return mask


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


def generate_cell(
    output: Path, cell: dict, regions, eviann, reference,
    eviann_phase=None, reference_phase=None, cds_boundary_radius=0,
    eviann_transcript_selection=None,
) -> dict:
    window, stride = int(cell["window_bp"]), int(cell["stride_bp"])
    summary = {}
    output.mkdir(parents=True, exist_ok=True)
    fasta = pysam.FastaFile(str(FASTA))
    for split, chrom_names in SPLITS.items():
        split_summary = {"windows": 0, "by_chrom": {}, "eviann": defaultdict(int), "reference": defaultdict(int)}
        with WindowWriter(output / f"{split}.h5") as writer:
            for name in chrom_names:
                chrom = CHROMS[name]
                chrom_len = fasta.get_reference_length(chrom)
                phase_tracks = None
                if eviann_phase is not None:
                    eviann_cds = {
                        strand: build_cds_track(
                            eviann_phase[chrom][strand], chrom_len
                        )
                        for strand in ("+", "-")
                    }
                    reference_cds = {
                        strand: build_cds_track(
                            reference_phase[chrom][strand], chrom_len
                        )
                        for strand in ("+", "-")
                    }
                    phase_tracks = {
                        "eviann": {
                            strand: build_phase_track(
                                eviann_phase[chrom][strand], chrom_len, strand
                            )
                            for strand in ("+", "-")
                        },
                        "eviann_cds": eviann_cds,
                        "reference": {
                            strand: build_phase_track(
                                reference_phase[chrom][strand], chrom_len, strand
                            )
                            for strand in ("+", "-")
                        },
                        "reference_cds": reference_cds,
                    }
                    if cds_boundary_radius:
                        phase_tracks["eviann_cds_boundary"] = {
                            strand: build_cds_boundary_mask(
                                eviann_cds[strand], cds_boundary_radius
                            )
                            for strand in ("+", "-")
                        }
                        phase_tracks["reference_cds_boundary"] = {
                            strand: build_cds_boundary_mask(
                                reference_cds[strand], cds_boundary_radius
                            )
                            for strand in ("+", "-")
                        }
                count = 0
                for strand in ("+", "-"):
                    starts = window_starts(regions[chrom][strand], chrom_len, window, stride)
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
                        arrays = {
                            "sequence": sequence, "labels": e_splice,
                            "start_stop_labels": e_ss, "chess_labels": r_splice,
                            "chess_start_stop_labels": r_ss,
                        }
                        if phase_tracks is not None:
                            e_phase = phase_tracks["eviann"][strand][start:start + window]
                            r_phase = phase_tracks["reference"][strand][start:start + window]
                            e_cds = phase_tracks["eviann_cds"][strand][start:start + window]
                            r_cds = phase_tracks["reference_cds"][strand][start:start + window]
                            if cds_boundary_radius:
                                e_cds_boundary = phase_tracks[
                                    "eviann_cds_boundary"
                                ][strand][start:start + window]
                                r_cds_boundary = phase_tracks[
                                    "reference_cds_boundary"
                                ][strand][start:start + window]
                            if strand == "-":
                                # Tracks are indexed in genomic order; model input is
                                # reverse-complemented into transcript order.
                                e_phase = e_phase[::-1]
                                r_phase = r_phase[::-1]
                                e_cds = e_cds[::-1]
                                r_cds = r_cds[::-1]
                                if cds_boundary_radius:
                                    e_cds_boundary = e_cds_boundary[::-1]
                                    r_cds_boundary = r_cds_boundary[::-1]
                            arrays.update(
                                phase_labels=np.maximum(e_phase, 0),
                                phase_mask=e_phase >= 0,
                                reference_phase_labels=np.maximum(r_phase, 0),
                                reference_phase_mask=r_phase >= 0,
                                cds_labels=e_cds,
                                reference_cds_labels=r_cds,
                            )
                            if cds_boundary_radius:
                                arrays.update(
                                    cds_boundary_mask=e_cds_boundary,
                                    reference_cds_boundary_mask=r_cds_boundary,
                                )
                        writer.append(filename, arrays)
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
        "schema": "droso_window_cell_v2", "cell": cell["id"],
        "window_size": window, "stride": stride, "adjacent_overlap_bp": window - stride,
        "region_overlap_bp": window,
        "region_policy": "full_window_intersects_EviAnn_transcript_span_with_CDS",
        "phase_labels": (
            "GFF/GTF CDS codon offsets 0/1/2; unlabeled and transcript-phase "
            "conflicts masked" if eviann_phase is not None else None
        ),
        "cds_labels": (
            "union of all GFF/GTF CDS intervals, including positions whose "
            "isoform phases conflict" if eviann_phase is not None else None
        ),
        "cds_boundary_labels": (
            {
                "definition": "bases within radius of chromosome-level CDS 0/1 transitions",
                "radius_bp": cds_boundary_radius,
                "window_edges_are_boundaries": False,
            }
            if cds_boundary_radius else None
        ),
        "eviann_transcript_selection": eviann_transcript_selection,
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
    parser.add_argument(
        "--eviann-transcript-selection",
        choices=(
            "all", "longest-cds-per-locus",
            "longest-complete-equal-else-k",
            "complete-equal-else-k-with-fallback",
        ),
        default="all",
        help=(
            "Select representative EviAnn transcripts for start/stop labels "
            "only. Donor/acceptor, CDS, phase, and window-region labels always "
            "retain all EviAnn transcripts."
        ),
    )
    parser.add_argument("--reference-gtf", type=Path, default=REFERENCE_GTF)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument(
        "--include-phase-labels", action="store_true",
        help="Store masked EviAnn and reference CDS phase targets in every window.",
    )
    parser.add_argument(
        "--cds-boundary-radius", type=int, default=0,
        help=(
            "Also store chromosome-aware CDS boundary masks covering this many "
            "bases on each side of every CDS 0/1 transition; 0 disables."
        ),
    )
    args = parser.parse_args()
    if args.cds_boundary_radius < 0:
        parser.error("--cds-boundary-radius must be non-negative")
    if args.cds_boundary_radius and not args.include_phase_labels:
        parser.error("--cds-boundary-radius requires --include-phase-labels")
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
    e_meta, e_exons, e_cds, _all_selection = parse_eviann(EVIANN_GFF, "all")
    if args.eviann_transcript_selection == "all":
        ss_meta, ss_exons, ss_cds = e_meta, e_exons, e_cds
        eviann_selection = _all_selection
    else:
        ss_meta, ss_exons, ss_cds, eviann_selection = parse_eviann(
            EVIANN_GFF, args.eviann_transcript_selection
        )
    eviann_selection = {
        **eviann_selection,
        "applies_to": ["start", "stop"],
        "donor_acceptor_source": "all EviAnn transcripts (union)",
        "cds_phase_source": "all EviAnn transcripts (union; conflicts masked)",
        "window_region_source": "all EviAnn coding transcripts",
    }
    r_meta, r_exons, r_cds, explicit = parse_reference(REFERENCE_GTF)
    regions = coding_regions(e_meta, e_exons, e_cds)
    e_sites, e_rejected = extract_sites(e_meta, e_exons, e_cds)
    if args.eviann_transcript_selection != "all":
        ss_sites, ss_rejected = extract_sites(ss_meta, ss_exons, ss_cds)
        for site in ("start", "stop"):
            e_sites[site] = ss_sites[site]
            e_rejected[site] = ss_rejected.get(site, 0)
    r_sites, r_rejected = extract_sites(r_meta, r_exons, r_cds, explicit)
    e_index, r_index = index_sites(e_sites), index_sites(r_sites)
    e_phase = (
        parse_cds_phase_records(EVIANN_GFF)
        if args.include_phase_labels else None
    )
    r_phase = parse_cds_phase_records(REFERENCE_GTF) if args.include_phase_labels else None
    completed = []
    for cell in config["cells"]:
        cell_dir = args.output_root / cell["id"]
        manifest_path = cell_dir / "split_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            expected = sum(int(manifest["summary"][s]["windows"]) for s in SPLITS)
            actual = 0
            for split in SPLITS:
                store = WindowStore(cell_dir / split)
                try:
                    actual += len(store)
                finally:
                    store.close()
            if actual != expected:
                raise RuntimeError(f"partial cached cell {cell['id']}: {actual}/{expected}")
            print(f"validated cached {cell['id']}: {actual:,} windows", flush=True)
        else:
            if cell_dir.exists() and any(cell_dir.iterdir()):
                raise RuntimeError(f"refusing partial non-empty cell: {cell_dir}")
            manifest = generate_cell(
                cell_dir, cell, regions, e_index, r_index, e_phase, r_phase,
                args.cds_boundary_radius,
                eviann_selection,
            )
        completed.append({"cell": cell["id"], "manifest": str(manifest_path.resolve()), "windows": sum(manifest["summary"][s]["windows"] for s in SPLITS)})
    atomic_json(args.output_root / ".complete.json", {
        "schema": "droso_window_generation_complete_v1", "cells": completed,
        "eviann_transcript_selection": eviann_selection,
        "eviann_rejected": e_rejected, "reference_rejected": r_rejected,
        "elapsed_seconds_current_invocation": time.time() - started,
    })
    print(f"complete: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
