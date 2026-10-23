#!/usr/bin/env python3
"""Strip optimizer state from a training checkpoint for portable inference."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--description", required=True)
    args = parser.parse_args()

    source = args.checkpoint.resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    required = ("epoch", "model_state_dict", "config")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        parser.error(f"source checkpoint lacks required fields: {missing}")

    payload = {
        "format": "cnn_mamba_inference_checkpoint_v1",
        "description": args.description,
        "epoch": int(checkpoint["epoch"]),
        "model_state_dict": checkpoint["model_state_dict"],
        "config": checkpoint["config"],
        "mean_AP": checkpoint.get("mean_AP"),
        "best_validation_mean_AP": checkpoint.get("best_validation_mean_AP"),
        "optimizer_step": checkpoint.get("optimizer_step"),
        "source_checkpoint_sha256": sha256(source),
        "optimizer_state_included": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(f"output={args.output}")
    print(f"bytes={args.output.stat().st_size}")
    print(f"sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
