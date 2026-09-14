#!/usr/bin/env python3
"""Score every plus-strand canonical candidate on Dmel chrX at checkpoint geometry."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pysam
import torch

from .model import model_from_checkpoint

ENCODE = np.full(256, 4, dtype=np.uint8)
for _base, _value in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    ENCODE[ord(_base)] = _value
A, C, G, T = 0, 1, 2, 3
TASKS = ("donor", "acceptor", "start", "stop")
LENGTHS = {"donor": 2, "acceptor": 2, "start": 3, "stop": 3}


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


def encode(text: str):
    return ENCODE[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]


def masks(sequence):
    donor = np.zeros(len(sequence), dtype=bool)
    acceptor = np.zeros(len(sequence), dtype=bool)
    start = np.zeros(len(sequence), dtype=bool)
    stop = np.zeros(len(sequence), dtype=bool)
    donor[:-1] = (sequence[:-1] == G) & (sequence[1:] == T)
    acceptor[:-1] = (sequence[:-1] == A) & (sequence[1:] == G)
    start[:-2] = (sequence[:-2] == A) & (sequence[1:-1] == T) & (sequence[2:] == G)
    x0, x1, x2 = sequence[:-2], sequence[1:-1], sequence[2:]
    stop[:-2] = ((x0 == T) & (x1 == A) & ((x2 == A) | (x2 == G))) | ((x0 == T) & (x1 == G) & (x2 == A))
    return {"donor": donor, "acceptor": acceptor, "start": start, "stop": stop}


def count_candidates(fasta, chrom, length, chunk=5_000_000):
    counts = {name: 0 for name in TASKS}
    for start in range(0, length, chunk):
        end = min(length, start + chunk)
        sequence = encode(fasta.fetch(chrom, start, min(length, end + 2)).upper())
        local = masks(sequence)
        for name in TASKS:
            counts[name] += int(local[name][:end-start].sum())
    return counts


def keep_bounds(index, number, window, stride):
    flank = (window - stride) // 2
    if number == 1:
        return 0, window
    if index == 0:
        return 0, window - flank
    if index == number - 1:
        return flank, window
    return flank, window - flank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--chrom", required=True, help="FASTA accession/sequence name")
    parser.add_argument("--display-chrom", default="chrX")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    if (args.output_dir / ".complete.json").is_file():
        print(f"already complete: {args.output_dir}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    window = int(checkpoint["config"]["window_size"])
    stride = int(checkpoint["config"]["stride"])
    batch_size = args.batch_size or ({5000: 8, 10000: 4, 20000: 2}.get(window, 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    model = model_from_checkpoint(checkpoint, device).eval()
    fasta = pysam.FastaFile(str(args.fasta))
    chrom_len = fasta.get_reference_length(args.chrom)
    counts = count_candidates(fasta, args.chrom, chrom_len)
    arrays = {}
    for name, count in counts.items():
        arrays[name] = {
            "positions": np.lib.format.open_memmap(args.output_dir / f"{name}_positions0.npy", mode="w+", dtype=np.uint32, shape=(count,)),
            "scores": np.lib.format.open_memmap(args.output_dir / f"{name}_scores.npy", mode="w+", dtype=np.float32, shape=(count,)),
        }
    arrays["stop"]["motif_code"] = np.lib.format.open_memmap(args.output_dir / "stop_motif_codes.npy", mode="w+", dtype=np.uint8, shape=(counts["stop"],))
    offsets = {name: 0 for name in TASKS}
    starts = list(range(0, chrom_len, stride))
    started = time.time()
    for batch_start in range(0, len(starts), batch_size):
        batch_starts = starts[batch_start:batch_start + batch_size]
        encoded = []
        for start in batch_starts:
            text = fasta.fetch(args.chrom, start, min(chrom_len, start + window)).upper()
            encoded.append(encode(text + "N" * (window - len(text))))
        encoded = np.stack(encoded)
        tensor = torch.from_numpy(encoded).long().to(device)
        with torch.inference_mode():
            splice, start_stop = model(tensor)
            splice, start_stop = splice.softmax(-1).float().cpu().numpy(), start_stop.softmax(-1).float().cpu().numpy()
        for local_index, (start, sequence) in enumerate(zip(batch_starts, encoded)):
            index = batch_start + local_index
            left, right = keep_bounds(index, len(starts), window, stride)
            local_masks = masks(sequence)
            for name in TASKS:
                valid_right = min(right, chrom_len - start - LENGTHS[name] + 1)
                if valid_right <= left:
                    continue
                positions = np.flatnonzero(local_masks[name][left:valid_right]) + left
                count = len(positions)
                if not count:
                    continue
                destination = slice(offsets[name], offsets[name] + count)
                arrays[name]["positions"][destination] = start + positions
                probs = splice if name in ("donor", "acceptor") else start_stop
                cls = 1 if name in ("donor", "start") else 2
                arrays[name]["scores"][destination] = probs[local_index, positions, cls]
                if name == "stop":
                    middle, last = sequence[positions + 1], sequence[positions + 2]
                    arrays[name]["motif_code"][destination] = np.where(middle == G, 2, np.where(last == G, 1, 0)).astype(np.uint8)
                offsets[name] += count
        done = min(batch_start + batch_size, len(starts))
        if done % 500 < batch_size or done == len(starts):
            print(f"processed={done:,}/{len(starts):,}", flush=True)
    fasta.close()
    for name in TASKS:
        if offsets[name] != counts[name]:
            raise RuntimeError(f"{name}: wrote {offsets[name]}, expected {counts[name]}")
        for array in arrays[name].values():
            array.flush()
    metadata = {
        "schema": "droso_chrx_all_candidate_scores_v1", "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": checkpoint["optimizer_step"], "checkpoint_epoch": None,
        "fasta": str(args.fasta.resolve()), "chromosome": args.display_chrom,
        "fasta_sequence_name": args.chrom, "chromosome_length": chrom_len,
        "strand": "+", "window_size": window, "stride": stride,
        "ownership": "midpoint partition of 50%-overlap windows; chromosome edges retained",
        "position_convention": "zero-based genomic start of motif",
        "candidate_counts": counts, "written_counts": offsets,
        "regime": checkpoint["config"].get("regime", "unspecified"),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / "metadata.json", metadata)
    atomic_json(args.output_dir / ".complete.json", {"schema": "droso_score_complete_v1", "counts": offsets})
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
