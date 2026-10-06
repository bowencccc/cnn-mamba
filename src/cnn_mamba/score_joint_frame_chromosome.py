#!/usr/bin/env python3
"""Export four-state non-CDS/frame probabilities chromosome-wide."""

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
    args = parser.parse_args()

    sentinel = args.output_dir / ".complete.json"
    if sentinel.is_file():
        print(f"already complete: {args.output_dir}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    has_joint_head = (
        float(config.get("joint_frame_aux_weight", 0.0)) > 0
        or any(
            name.startswith("joint_frame_head.")
            for name in checkpoint["model_state_dict"]
        )
    )
    if not has_joint_head:
        raise RuntimeError("checkpoint does not contain a four-state joint-frame head")

    window, stride = int(config["window_size"]), int(config["stride"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta = pysam.FastaFile(str(args.fasta))
    chrom_len = fasta.get_reference_length(args.chrom)
    starts = list(range(0, chrom_len, stride))
    probabilities = np.lib.format.open_memmap(
        args.output_dir / "joint_frame_probabilities.npy", mode="w+",
        dtype=np.float32, shape=(chrom_len, 4),
    )
    argmax = np.lib.format.open_memmap(
        args.output_dir / "joint_frame_argmax.npy", mode="w+",
        dtype=np.uint8, shape=(chrom_len,),
    )

    device = torch.device("cuda")
    model = model_from_checkpoint(checkpoint, device).eval()
    expected_position = 0
    started = time.time()
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
            outputs = model(sequence, return_joint_frame=True)
            values = outputs[2].softmax(-1).float().cpu().numpy()

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
                local = values[local_index, left:valid_right]
                probabilities[destination_start:destination_end] = local
                argmax[destination_start:destination_end] = local.argmax(1)
                expected_position = destination_end
            done = min(batch_start + args.batch_size, len(starts))
            if done % 500 < args.batch_size or done == len(starts):
                print(f"processed={done:,}/{len(starts):,}", flush=True)

    fasta.close()
    if expected_position != chrom_len:
        raise RuntimeError(f"incomplete coverage: {expected_position}/{chrom_len}")
    probabilities.flush()
    argmax.flush()
    metadata = {
        "schema": "joint_frame_probabilities_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "fasta": str(args.fasta.resolve()),
        "fasta_sequence_name": args.chrom,
        "chromosome": args.display_chrom,
        "strand": "+",
        "chromosome_length": chrom_len,
        "window_size": window,
        "stride": stride,
        "class_order": ["non_CDS", "frame_0", "frame_1", "frame_2"],
        "coding_probability": "1 - P(non_CDS)",
        "position_convention": "row i is zero-based genomic position i",
        "overlap_resolution": "midpoint ownership of overlapping windows",
        "files": {
            "probabilities": "joint_frame_probabilities.npy",
            "argmax": "joint_frame_argmax.npy",
        },
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
