#!/usr/bin/env python3
"""Export CDS probabilities and CDS-gated phase probabilities chromosome-wide."""

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
        "--write-raw-phase", action="store_true",
        help="Also persist the ungated phase softmax for direct comparison.",
    )
    parser.add_argument(
        "--reference-phase-probabilities", type=Path,
        help="Optional original raw phase array used to prove the frozen model is unchanged.",
    )
    args = parser.parse_args()
    sentinel = args.output_dir / ".complete.json"
    if sentinel.is_file():
        print(f"already complete: {args.output_dir}", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if not config.get("cds_auxiliary"):
        raise RuntimeError("checkpoint does not contain a trained CDS head")
    if float(config.get("phase_aux_weight", 0.0)) <= 0:
        raise RuntimeError("checkpoint does not contain a trained phase head")
    window, stride = int(config["window_size"]), int(config["stride"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta = pysam.FastaFile(str(args.fasta))
    chrom_len = fasta.get_reference_length(args.chrom)
    starts = list(range(0, chrom_len, stride))
    cds_probability = np.lib.format.open_memmap(
        args.output_dir / "cds_probability.npy", mode="w+",
        dtype=np.float32, shape=(chrom_len,),
    )
    gated_phase = np.lib.format.open_memmap(
        args.output_dir / "gated_phase_probabilities.npy", mode="w+",
        dtype=np.float32, shape=(chrom_len, 3),
    )
    raw_phase = (
        np.lib.format.open_memmap(
            args.output_dir / "raw_phase_probabilities.npy", mode="w+",
            dtype=np.float32, shape=(chrom_len, 3),
        )
        if args.write_raw_phase else None
    )
    joint_argmax = np.lib.format.open_memmap(
        args.output_dir / "joint_class_argmax.npy", mode="w+",
        dtype=np.uint8, shape=(chrom_len,),
    )
    reference = None
    if args.reference_phase_probabilities:
        reference = np.load(args.reference_phase_probabilities, mmap_mode="r")
        if reference.shape != (chrom_len, 3):
            raise RuntimeError(
                f"reference phase shape {reference.shape} != {(chrom_len, 3)}"
            )

    device = torch.device("cuda")
    model = model_from_checkpoint(checkpoint, device).eval()
    expected_position = 0
    max_phase_difference = 0.0
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
            _, _, phase_logits, cds_logits = model(
                sequence, return_phase=True, return_cds=True
            )
            phase_values = phase_logits.softmax(-1).float().cpu().numpy()
            cds_values = cds_logits.softmax(-1)[..., 1].float().cpu().numpy()

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
                raw = phase_values[local_index, left:valid_right]
                cds = cds_values[local_index, left:valid_right]
                gated = raw * cds[:, None]
                cds_probability[destination_start:destination_end] = cds
                gated_phase[destination_start:destination_end] = gated
                if raw_phase is not None:
                    raw_phase[destination_start:destination_end] = raw
                joint = np.column_stack((1.0 - cds, gated))
                joint_argmax[destination_start:destination_end] = joint.argmax(1)
                if reference is not None:
                    difference = np.max(
                        np.abs(reference[destination_start:destination_end] - raw)
                    )
                    max_phase_difference = max(max_phase_difference, float(difference))
                expected_position = destination_end
            done = min(batch_start + args.batch_size, len(starts))
            if done % 500 < args.batch_size or done == len(starts):
                print(f"processed={done:,}/{len(starts):,}", flush=True)

    fasta.close()
    if expected_position != chrom_len:
        raise RuntimeError(f"incomplete coverage: {expected_position}/{chrom_len}")
    cds_probability.flush()
    gated_phase.flush()
    if raw_phase is not None:
        raw_phase.flush()
    joint_argmax.flush()
    metadata = {
        "schema": "cds_gated_phase_probabilities_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "base_checkpoint": checkpoint.get("base_checkpoint"),
        "fasta": str(args.fasta.resolve()),
        "fasta_sequence_name": args.chrom,
        "chromosome": args.display_chrom,
        "strand": "+",
        "chromosome_length": chrom_len,
        "window_size": window,
        "stride": stride,
        "gating": "P(CDS) * P(phase_k | CDS)",
        "joint_class_order": ["non_CDS", "phase_0", "phase_1", "phase_2"],
        "files": {
            "cds_probability": "cds_probability.npy",
            "gated_phase_probabilities": "gated_phase_probabilities.npy",
            "joint_class_argmax": "joint_class_argmax.npy",
            "raw_phase_probabilities": (
                "raw_phase_probabilities.npy" if raw_phase is not None else None
            ),
        },
        "reference_phase_probabilities": (
            None if args.reference_phase_probabilities is None
            else str(args.reference_phase_probabilities.resolve())
        ),
        "maximum_raw_phase_probability_difference": (
            None if reference is None else max_phase_difference
        ),
        "position_convention": "row i is zero-based genomic position i",
        "overlap_resolution": "midpoint ownership of overlapping windows",
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
