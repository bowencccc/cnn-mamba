#!/usr/bin/env python3
"""Build reference-free high-confidence teacher masks for candidate weighting."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .common import splice_candidate_masks, start_stop_candidate_masks
from .model import model_from_checkpoint


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--splits", default="train,test")
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("threshold must be in (0, 1)")
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]

    files = []
    for split in splits:
        files.extend((split, path) for path in sorted((args.data_root / split).glob("*.npz")))
    if not files:
        raise RuntimeError("no input NPZ files found")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = model_from_checkpoint(checkpoint, torch.device("cuda")).eval()
    totals = {
        name: {"candidates": 0, "eviann_positive": 0, "selected_unlabeled": 0}
        for name in ("donor", "acceptor", "start", "stop")
    }
    completed = 0
    started = time.time()

    for batch_start in range(0, len(files), args.batch_size):
        batch_files = files[batch_start:batch_start + args.batch_size]
        sequences, splice_labels, ss_labels = [], [], []
        for _split, path in batch_files:
            with np.load(path) as data:
                sequences.append(data["sequence"].astype(np.int64, copy=False))
                splice_labels.append(data["labels"].astype(np.int64, copy=False))
                ss_labels.append(data["start_stop_labels"].astype(np.int64, copy=False))
        sequence = torch.from_numpy(np.stack(sequences)).long().cuda()
        splice_label = torch.from_numpy(np.stack(splice_labels)).long().cuda()
        ss_label = torch.from_numpy(np.stack(ss_labels)).long().cuda()
        with torch.inference_mode():
            splice_logits, ss_logits = model(sequence)[:2]
            splice_probability = splice_logits.softmax(-1)
            ss_probability = ss_logits.softmax(-1)
            donor, acceptor = splice_candidate_masks(sequence)
            start, stop = start_stop_candidate_masks(sequence)

            donor_selected = donor & (splice_label == 0) & (
                splice_probability[..., 1] >= args.threshold
            )
            acceptor_selected = acceptor & (splice_label == 0) & (
                splice_probability[..., 2] >= args.threshold
            )
            start_selected = start & (ss_label == 0) & (
                ss_probability[..., 1] >= args.threshold
            )
            stop_selected = stop & (ss_label == 0) & (
                ss_probability[..., 2] >= args.threshold
            )
            selections = {
                "donor": (donor, splice_label == 1, donor_selected),
                "acceptor": (acceptor, splice_label == 2, acceptor_selected),
                "start": (start, ss_label == 1, start_selected),
                "stop": (stop, ss_label == 2, stop_selected),
            }
            for name, (candidate, positive, selected) in selections.items():
                totals[name]["candidates"] += int(candidate.sum().item())
                totals[name]["eviann_positive"] += int(positive.sum().item())
                totals[name]["selected_unlabeled"] += int(selected.sum().item())

            splice_masks = (donor_selected | acceptor_selected).cpu().numpy()
            ss_masks = (start_selected | stop_selected).cpu().numpy()

        for index, (split, source) in enumerate(batch_files):
            destination = args.output_root / split / source.name
            if destination.is_file():
                with np.load(destination) as existing:
                    valid = (
                        existing.get("splice_propensity", np.empty(0)).shape
                        == splice_masks[index].shape
                        and existing.get("start_stop_propensity", np.empty(0)).shape
                        == ss_masks[index].shape
                    )
                if valid:
                    destination.unlink()
            atomic_npz(
                destination,
                splice_propensity=splice_masks[index],
                start_stop_propensity=ss_masks[index],
            )
            completed += 1
        if completed % 500 < len(batch_files) or completed == len(files):
            print(f"processed={completed:,}/{len(files):,}", flush=True)

    payload = {
        "schema": "candidate_teacher_propensity_masks_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "data_root": str(args.data_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "splits": splits,
        "threshold": args.threshold,
        "selection_rule": (
            "canonical candidate, absent from EviAnn label, and frozen-teacher "
            "task probability >= threshold"
        ),
        "reference_labels_used": False,
        "files": len(files),
        "totals_with_overlapping_windows": totals,
        "elapsed_seconds": time.time() - started,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
