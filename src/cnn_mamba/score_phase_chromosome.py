#!/usr/bin/env python3
"""Score per-base phase-0/1/2 probabilities across one chromosome strand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pysam
import torch

from .model import model_from_checkpoint
from .score_chromosome import encode, keep_bounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--display-chrom", default="chrX")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--dtype", choices=("float16", "float32"), default="float32",
        help="Storage dtype for the [chromosome_length, 3] probability matrix.",
    )
    args = parser.parse_args()

    sentinel = args.output_dir / ".complete.json"
    if sentinel.is_file():
        print(f"already complete: {args.output_dir}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    if float(config.get("phase_aux_weight", 0.0)) <= 0:
        raise RuntimeError("checkpoint does not contain a trained phase auxiliary head")
    window = int(config["window_size"])
    stride = int(config["stride"])
    if stride > window:
        raise RuntimeError("stride cannot exceed window size")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta = pysam.FastaFile(str(args.fasta))
    chrom_len = fasta.get_reference_length(args.chrom)
    starts = list(range(0, chrom_len, stride))
    storage_dtype = np.float16 if args.dtype == "float16" else np.float32
    probabilities = np.lib.format.open_memmap(
        args.output_dir / "phase_probabilities.npy", mode="w+",
        dtype=storage_dtype, shape=(chrom_len, 3),
    )
    argmax = np.lib.format.open_memmap(
        args.output_dir / "phase_argmax.npy", mode="w+",
        dtype=np.uint8, shape=(chrom_len,),
    )

    device = torch.device("cuda")
    model = model_from_checkpoint(checkpoint, device).eval()
    started = time.time()
    expected_position = 0
    with torch.inference_mode():
        for batch_start in range(0, len(starts), args.batch_size):
            batch_starts = starts[batch_start:batch_start + args.batch_size]
            encoded = []
            for start in batch_starts:
                text = fasta.fetch(
                    args.chrom, start, min(chrom_len, start + window)
                ).upper()
                encoded.append(encode(text + "N" * (window - len(text))))
            sequence = torch.from_numpy(np.stack(encoded)).long().to(device)
            _, _, phase_logits = model(sequence, return_phase=True)
            phase_probs = phase_logits.softmax(-1).float().cpu().numpy()

            for local_index, start in enumerate(batch_starts):
                window_index = batch_start + local_index
                left, right = keep_bounds(
                    window_index, len(starts), window, stride
                )
                valid_right = min(right, chrom_len - start)
                if valid_right <= left:
                    continue
                destination_start = start + left
                destination_end = start + valid_right
                if destination_start != expected_position:
                    raise RuntimeError(
                        f"non-contiguous ownership: expected {expected_position}, "
                        f"got {destination_start}"
                    )
                values = phase_probs[local_index, left:valid_right]
                probabilities[destination_start:destination_end] = values
                argmax[destination_start:destination_end] = values.argmax(axis=1)
                expected_position = destination_end

            done = min(batch_start + args.batch_size, len(starts))
            if done % 500 < args.batch_size or done == len(starts):
                print(f"processed={done:,}/{len(starts):,}", flush=True)

    fasta.close()
    if expected_position != chrom_len:
        raise RuntimeError(
            f"incomplete chromosome coverage: {expected_position}/{chrom_len}"
        )
    probabilities.flush()
    argmax.flush()
    metadata = {
        "schema": "per_base_phase_probabilities_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "phase_aux_weight": float(config["phase_aux_weight"]),
        "frame_dilation": int(config.get("frame_dilation", 0)),
        "fasta": str(args.fasta.resolve()),
        "fasta_sequence_name": args.chrom,
        "chromosome": args.display_chrom,
        "strand": "+",
        "chromosome_length": chrom_len,
        "window_size": window,
        "stride": stride,
        "probability_file": "phase_probabilities.npy",
        "probability_shape": [chrom_len, 3],
        "probability_dtype": args.dtype,
        "class_order": ["phase_0", "phase_1", "phase_2"],
        "position_convention": (
            "row i corresponds to zero-based genomic position i on the plus strand"
        ),
        "overlap_resolution": (
            "midpoint ownership of overlapping windows; chromosome edges retained"
        ),
        "interpretation_warning": (
            "The phase head was supervised only at non-conflicting annotated CDS "
            "positions. Probabilities outside coding sequence are emitted but are "
            "not calibrated reading-frame probabilities. Predictions are per-base "
            "and are not constrained to follow a global 0-1-2 path."
        ),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    sentinel.write_text(
        json.dumps({"schema": metadata["schema"], "complete": True}, indent=2)
        + "\n"
    )
    print(f"complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
