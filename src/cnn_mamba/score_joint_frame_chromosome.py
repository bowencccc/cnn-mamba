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
    parser.add_argument(
        "--save-logits", action="store_true",
        help="Save owned, uncalibrated four-state logits for future calibration",
    )
    temperature_group = parser.add_mutually_exclusive_group()
    temperature_group.add_argument("--temperature", type=float, default=None)
    temperature_group.add_argument("--calibration-json", type=Path, default=None)
    temperature_group.add_argument(
        "--vector-calibration-json", type=Path, default=None,
        help="JSON containing classwise vector_scale and vector_bias",
    )
    args = parser.parse_args()

    class_order = ["non_CDS", "frame_0", "frame_1", "frame_2"]
    vector_scale = vector_bias = None
    if args.vector_calibration_json is not None:
        calibration = json.loads(args.vector_calibration_json.read_text())
        try:
            vector_scale = [float(calibration["vector_scale"][name]) for name in class_order]
            vector_bias = [float(calibration["vector_bias"][name]) for name in class_order]
        except (KeyError, TypeError, ValueError) as error:
            parser.error(f"invalid vector calibration JSON: {error}")
        if not all(np.isfinite(vector_scale)) or not all(np.isfinite(vector_bias)):
            parser.error("vector calibration parameters must be finite")
        temperature = 1.0
        calibration_path = str(args.vector_calibration_json.resolve())
        calibration_method = "vector_scaling"
    elif args.calibration_json is not None:
        calibration = json.loads(args.calibration_json.read_text())
        temperature = float(calibration["temperature"])
        calibration_path = str(args.calibration_json.resolve())
        calibration_method = "scalar_temperature"
    else:
        temperature = 1.0 if args.temperature is None else float(args.temperature)
        calibration_path = None
        calibration_method = "raw" if temperature == 1.0 else "scalar_temperature"
    if temperature <= 0:
        parser.error("temperature must be positive")

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
    saved_logits = None
    if args.save_logits:
        saved_logits = np.lib.format.open_memmap(
            args.output_dir / "joint_frame_logits.npy", mode="w+",
            dtype=np.float32, shape=(chrom_len, 4),
        )

    device = torch.device("cuda")
    model = model_from_checkpoint(checkpoint, device).eval()
    vector_scale_tensor = (
        None if vector_scale is None
        else torch.tensor(vector_scale, dtype=torch.float32, device=device)
    )
    vector_bias_tensor = (
        None if vector_bias is None
        else torch.tensor(vector_bias, dtype=torch.float32, device=device)
    )
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
            raw_logits = outputs[2].float()
            if vector_scale_tensor is not None:
                calibrated_logits = (
                    raw_logits * vector_scale_tensor + vector_bias_tensor
                )
            else:
                calibrated_logits = raw_logits / temperature
            values = calibrated_logits.softmax(-1).cpu().numpy()
            raw_values = raw_logits.cpu().numpy() if saved_logits is not None else None

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
                if saved_logits is not None:
                    saved_logits[destination_start:destination_end] = raw_values[
                        local_index, left:valid_right
                    ]
                expected_position = destination_end
            done = min(batch_start + args.batch_size, len(starts))
            if done % 500 < args.batch_size or done == len(starts):
                print(f"processed={done:,}/{len(starts):,}", flush=True)

    fasta.close()
    if expected_position != chrom_len:
        raise RuntimeError(f"incomplete coverage: {expected_position}/{chrom_len}")
    probabilities.flush()
    argmax.flush()
    if saved_logits is not None:
        saved_logits.flush()
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
        "class_order": class_order,
        "calibration_method": calibration_method,
        "temperature": temperature if vector_scale is None else None,
        "calibration_json": calibration_path,
        "temperature_scope": (
            "joint_frame_head_only" if vector_scale is None else None
        ),
        "vector_scale": (
            None if vector_scale is None else dict(zip(class_order, vector_scale))
        ),
        "vector_bias": (
            None if vector_bias is None else dict(zip(class_order, vector_bias))
        ),
        "calibration_scope": "joint_frame_head_only",
        "coding_probability": "1 - P(non_CDS)",
        "position_convention": "row i is zero-based genomic position i",
        "overlap_resolution": "midpoint ownership of overlapping windows",
        "files": {
            "probabilities": "joint_frame_probabilities.npy",
            "argmax": "joint_frame_argmax.npy",
            "raw_logits": "joint_frame_logits.npy" if saved_logits is not None else None,
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
