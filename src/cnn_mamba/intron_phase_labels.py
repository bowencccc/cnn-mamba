"""Build transcript-oriented coding-intron phase labels aligned to NPZ windows.

For every transcript, CDS segments are ordered in transcription direction.
Only genomic gaps between two consecutive CDS segments are labeled, excluding
UTR-only introns.  The phase assigned to the gap is the codon offset of the
first transcript-oriented base in the downstream CDS segment under
``build_phase_track``: ``(3 - gtf_phase) % 3``.  On ``-``, that builder reverses
its offset vector before assigning it in genomic order; after the saved window
is reverse-complemented, the first transcript-oriented CDS base therefore
still has this same value.  Using the exact existing convention is important
because these labels are aligned to already-generated windows.

Overlapping transcripts that assign different phases produce ``-2``; bases in
no coding intron are ``-1``.  Sidecars retain these signed labels plus a validity
mask, transcript-oriented distance to donor/acceptor, and a flag restricting
analysis to introns fully contained in the corresponding 10-kb window.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import time

import numpy as np


PHASE_SCHEMA = "coding_intron_phase_downstream_gtf_v2"


@dataclass(frozen=True)
class CodingIntron:
    chrom: str
    strand: str
    start: int
    end: int
    phase: int
    transcript_id: str

    @property
    def length(self):
        return self.end - self.start


class CodingIntronIndex:
    """Coarse genomic-bin index for fast 10-kb window overlap queries."""

    def __init__(self, indexed_introns, bin_size=100_000):
        self.bin_size = int(bin_size)
        self.introns = {}
        self.bins = {}
        for key, records in indexed_introns.items():
            # Transcript duplicates do not add information to a per-base track.
            unique = {}
            for record in records:
                unique[(record.start, record.end, record.phase)] = record
            records = sorted(unique.values(), key=lambda x: (x.start, x.end, x.phase))
            self.introns[key] = records
            bins = defaultdict(list)
            for index, record in enumerate(records):
                first = record.start // self.bin_size
                last = (record.end - 1) // self.bin_size
                for genomic_bin in range(first, last + 1):
                    bins[genomic_bin].append(index)
            self.bins[key] = dict(bins)

    def overlapping(self, chrom, strand, start, end):
        key = (chrom, strand)
        records = self.introns.get(key, ())
        bins = self.bins.get(key, {})
        indices = set()
        for genomic_bin in range(start // self.bin_size, (end - 1) // self.bin_size + 1):
            indices.update(bins.get(genomic_bin, ()))
        return [
            records[index] for index in sorted(indices)
            if records[index].end > start and records[index].start < end
        ]


def open_text(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else Path(path).open()


def gtf_attributes(text):
    result = {}
    for field in text.strip().strip(";").split(";"):
        field = field.strip()
        if not field:
            continue
        key, _, value = field.partition(" ")
        result[key] = value.strip().strip('"')
    return result


def coding_introns_from_gtf(path: Path):
    """Return coding introns indexed by ``(chromosome, strand)``."""
    transcripts = defaultdict(list)
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip().split("\t")
            if (
                len(fields) < 9 or fields[2] != "CDS"
                or fields[6] not in {"+", "-"} or fields[7] not in {"0", "1", "2"}
            ):
                continue
            attrs = gtf_attributes(fields[8])
            transcript = attrs.get("transcript_id")
            if not transcript:
                continue
            transcripts[(fields[0], fields[6], transcript)].append(
                (int(fields[3]) - 1, int(fields[4]), int(fields[7]))
            )

    indexed = defaultdict(list)
    for (chrom, strand, transcript), segments in transcripts.items():
        ordered = sorted(set(segments), reverse=strand == "-")
        for upstream, downstream in zip(ordered, ordered[1:]):
            if strand == "+":
                start, end = upstream[1], downstream[0]
            else:
                start, end = downstream[1], upstream[0]
            if end <= start:
                continue
            downstream_first_phase = (3 - downstream[2]) % 3
            indexed[(chrom, strand)].append(CodingIntron(
                chrom=chrom, strand=strand, start=start, end=end,
                phase=downstream_first_phase, transcript_id=transcript,
            ))
    for key in indexed:
        indexed[key].sort(key=lambda x: (x.start, x.end, x.phase, x.transcript_id))
    return indexed


def window_intron_tracks(introns, win_start: int, window_size: int, strand: str):
    """Create genomic-window tracks, then orient them along the transcript."""
    win_end = win_start + window_size
    phase = np.full(window_size, -1, dtype=np.int8)
    donor_distance = np.full(window_size, -1, dtype=np.int32)
    acceptor_distance = np.full(window_size, -1, dtype=np.int32)
    fits = np.zeros(window_size, dtype=np.bool_)
    intron_start = np.full(window_size, -1, dtype=np.int32)
    intron_end = np.full(window_size, -1, dtype=np.int32)
    for intron in introns:
        if intron.end <= win_start or intron.start >= win_end:
            continue
        lo = max(intron.start, win_start)
        hi = min(intron.end, win_end)
        sl = slice(lo - win_start, hi - win_start)
        view = phase[sl]
        unknown = view == -1
        conflict = (view >= 0) & (view != intron.phase)
        same_phase = view == intron.phase
        start_view = intron_start[sl]
        end_view = intron_end[sl]
        ambiguous_boundary = same_phase & (
            ((start_view >= 0) & (start_view != intron.start))
            | ((end_view >= 0) & (end_view != intron.end))
        )
        view[unknown] = intron.phase
        view[conflict] = -2
        start_view[unknown] = intron.start
        end_view[unknown] = intron.end
        start_view[conflict | ambiguous_boundary] = -2
        end_view[conflict | ambiguous_boundary] = -2
        positions = np.arange(lo, hi, dtype=np.int64)
        if strand == "+":
            donor = positions - intron.start
            acceptor = intron.end - 1 - positions
        else:
            donor = intron.end - 1 - positions
            acceptor = positions - intron.start
        dd = donor_distance[sl]
        ad = acceptor_distance[sl]
        dd[:] = np.where((dd < 0) | (donor < dd), donor, dd)
        ad[:] = np.where((ad < 0) | (acceptor < ad), acceptor, ad)
        if intron.start >= win_start and intron.end <= win_end:
            fits[sl] = True
    invalid = phase < 0
    donor_distance[invalid] = -1
    acceptor_distance[invalid] = -1
    fits[invalid] = False
    # Same-phase overlapping isoforms can make the state valid but the
    # donor/acceptor identity ambiguous.  Keep them for the 7-state probe and
    # exclude them only from boundary-distance analyses.
    fits[(intron_start < 0) | (intron_end < 0)] = False
    if strand == "-":
        phase = phase[::-1].copy()
        donor_distance = donor_distance[::-1].copy()
        acceptor_distance = acceptor_distance[::-1].copy()
        fits = fits[::-1].copy()
        intron_start = intron_start[::-1].copy()
        intron_end = intron_end[::-1].copy()
    return {
        "intron_phase_schema": np.asarray(PHASE_SCHEMA),
        "intron_phase_labels": phase,
        "intron_phase_mask": phase >= 0,
        "distance_to_donor": donor_distance,
        "distance_to_acceptor": acceptor_distance,
        "intron_fits_window": fits,
        "intron_start": intron_start,
        "intron_end": intron_end,
    }


def parse_window_name(path: Path):
    chrom, strand_name, start = path.stem.split("_")
    return f"chr{chrom}", "+" if strand_name == "plus" else "-", int(start)


def write_sidecars(
    gtf, windows_root, output_root, splits=("train", "val", "test"),
    window_size=10000,
):
    indexed = coding_introns_from_gtf(gtf)
    interval_index = CodingIntronIndex(indexed)
    counts = {"transcript_introns": sum(map(len, indexed.values())), "splits": {}}
    started = time.time()
    for split in splits:
        files = sorted((Path(windows_root) / split).glob("*.npz"))
        target = Path(output_root) / split
        target.mkdir(parents=True, exist_ok=True)
        written = valid = fitting = 0
        for path in files:
            chrom, strand, win_start = parse_window_name(path)
            introns = interval_index.overlapping(
                chrom, strand, win_start, win_start + window_size,
            )
            tracks = window_intron_tracks(introns, win_start, window_size, strand)
            np.savez_compressed(target / path.name, **tracks)
            written += 1
            valid += int(tracks["intron_phase_mask"].sum())
            fitting += int(tracks["intron_fits_window"].sum())
        counts["splits"][split] = {
            "windows": written, "valid_phase_positions": valid,
            "fitting_intron_positions": fitting,
        }
    counts["elapsed_seconds"] = time.time() - started
    (Path(output_root) / "manifest.json").write_text(json.dumps(counts, indent=2) + "\n")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--windows-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--window-size", type=int, default=10000)
    args = parser.parse_args()
    result = write_sidecars(
        args.gtf, args.windows_root, args.output_root,
        tuple(x.strip() for x in args.splits.split(",") if x.strip()),
        args.window_size,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
