#!/usr/bin/env python3
"""Fit a partial-label gene-structure HMM above a frozen CNN--Mamba model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from cnn_mamba.model import model_from_checkpoint
from cnn_mamba.structured_hmm import StructuredGeneHMM, build_partial_constraints


LABEL_KEYS = (
    "labels", "start_stop_labels", "phase_labels", "phase_mask", "cds_labels",
)


def choose_examples(directory, count, crop_length, seed):
    """Choose boundary-containing windows and a deterministic crop in each."""
    files = sorted(Path(directory).glob("*.npz"))
    generator = random.Random(seed)
    generator.shuffle(files)
    selected = []
    for path in files:
        with np.load(path) as data:
            boundaries = np.flatnonzero(
                (data["labels"] > 0) | (data["start_stop_labels"] > 0)
            )
            if boundaries.size == 0:
                continue
            anchor = int(boundaries[generator.randrange(len(boundaries))])
            length = int(data["sequence"].shape[0])
        start = max(0, min(anchor - crop_length // 2, length - crop_length))
        selected.append((path, start))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"requested {count} boundary windows from {directory}, found {len(selected)}"
        )
    return selected


def build_cache(model, examples, crop_length, batch_size, device, amp):
    cached = {key: [] for key in ("sequence", "splice_logits", "start_stop_logits",
                                  "frame_logits", *LABEL_KEYS)}
    started = time.time()
    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            chunk = examples[offset:offset + batch_size]
            full_sequences = []
            loaded = []
            for path, start in chunk:
                with np.load(path) as data:
                    arrays = {key: np.asarray(data[key]) for key in ("sequence", *LABEL_KEYS)}
                full_sequences.append(torch.from_numpy(arrays["sequence"].astype(np.int64)))
                loaded.append((arrays, start))
            sequence = torch.stack(full_sequences).to(device, non_blocking=True)
            with autocast("cuda", enabled=amp):
                splice, start_stop, frame = model(
                    sequence, return_joint_frame=True
                )
            for row, (arrays, start) in enumerate(loaded):
                stop = start + crop_length
                cached["sequence"].append(
                    torch.from_numpy(arrays["sequence"][start:stop].astype(np.int64))
                )
                cached["splice_logits"].append(splice[row, start:stop].float().cpu())
                cached["start_stop_logits"].append(start_stop[row, start:stop].float().cpu())
                cached["frame_logits"].append(frame[row, start:stop].float().cpu())
                for key in LABEL_KEYS:
                    cached[key].append(
                        torch.from_numpy(arrays[key][start:stop].astype(np.int64))
                    )
            print(
                f"cached {min(offset + batch_size, len(examples))}/{len(examples)}",
                flush=True,
            )
    result = {key: torch.stack(values) for key, values in cached.items()}
    result["collection_seconds"] = time.time() - started
    return result


def batches(cache, batch_size, shuffle, seed):
    count = cache["sequence"].shape[0]
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed)) \
        if shuffle else torch.arange(count)
    for start in range(0, count, batch_size):
        indices = order[start:start + batch_size]
        yield {key: value[indices] for key, value in cache.items() if torch.is_tensor(value)}


def evaluate(hmm, cache, batch_size, device):
    total_loss = total_examples = total_constraints = rejected = 0
    with torch.no_grad():
        for batch in batches(cache, batch_size, False, 0):
            batch = {key: value.to(device) for key, value in batch.items()}
            allowed, summary = build_partial_constraints(
                batch["labels"], batch["start_stop_labels"], batch["phase_labels"],
                batch["phase_mask"], batch["cds_labels"],
            )
            emissions = hmm.emissions(
                batch["sequence"], batch["splice_logits"],
                batch["start_stop_logits"], batch["frame_logits"],
            ).double()
            per_example = hmm.log_partition(emissions) - hmm.log_partition(emissions, allowed)
            valid = torch.isfinite(per_example) & (per_example < 5000)
            rejected += int((~valid).sum())
            if valid.any():
                total_loss += float(per_example[valid].sum())
                total_examples += int(valid.sum())
            total_constraints += summary.constrained_bases
    return {
        "partial_nll_per_example": total_loss / max(total_examples, 1),
        "accepted_examples": total_examples,
        "rejected_examples": rejected,
        "constrained_bases": total_constraints,
    }


def train(hmm, train_cache, val_cache, args, device, baseline):
    optimizer = torch.optim.Adam(hmm.parameters(), lr=args.learning_rate)
    history = []
    best_loss = baseline["partial_nll_per_example"]
    best_state = {key: value.detach().cpu().clone()
                  for key, value in hmm.state_dict().items()}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        hmm.train()
        running = examples = rejected = 0
        for batch in batches(train_cache, args.hmm_batch_size, True, args.seed + epoch):
            batch = {key: value.to(device) for key, value in batch.items()}
            allowed, _ = build_partial_constraints(
                batch["labels"], batch["start_stop_labels"], batch["phase_labels"],
                batch["phase_mask"], batch["cds_labels"],
            )
            emissions = hmm.emissions(
                batch["sequence"], batch["splice_logits"],
                batch["start_stop_logits"], batch["frame_logits"],
            ).double()
            per_example = hmm.log_partition(emissions) - hmm.log_partition(emissions, allowed)
            valid = torch.isfinite(per_example) & (per_example < 5000)
            rejected += int((~valid).sum())
            if not valid.any():
                continue
            data_loss = per_example[valid].mean()
            regularization = args.regularization * (
                hmm.frame_log_scale.square() + hmm.frame_bias.square().mean()
            )
            loss = data_loss + regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hmm.parameters(), 10.0)
            optimizer.step()
            running += float(data_loss.detach()) * int(valid.sum())
            examples += int(valid.sum())

        hmm.eval()
        validation = evaluate(hmm, val_cache, args.hmm_batch_size, device)
        row = {
            "epoch": epoch,
            "train_partial_nll": running / max(examples, 1),
            "train_accepted": examples,
            "train_rejected": rejected,
            "val_partial_nll": validation["partial_nll_per_example"],
            "val_accepted": validation["accepted_examples"],
            "val_rejected": validation["rejected_examples"],
            "frame_scale": float(hmm.frame_scale.detach()),
            **{
                f"bias_{name}": value
                for name, value in hmm.calibration_dict()["vector_bias"].items()
            },
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_partial_nll"] < best_loss - args.minimum_delta:
            best_loss = row["val_partial_nll"]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in hmm.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after epoch {epoch}", flush=True)
                break
    hmm.load_state_dict(best_state)
    return history, best_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-length", type=int, default=512)
    parser.add_argument("--train-windows", type=int, default=256)
    parser.add_argument("--val-windows", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=2)
    parser.add_argument("--hmm-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--minimum-delta", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument(
        "--initial-temperature", type=float, default=9.0,
        help="Starting frame temperature; the fitted multiplier is 1/T",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen-backbone inference")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "structured_crop_cache.pt"
    device = torch.device("cuda")

    if args.reuse_cache and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = model_from_checkpoint(saved, device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        train_examples = choose_examples(
            args.data_root / "train", args.train_windows, args.crop_length, args.seed
        )
        val_examples = choose_examples(
            args.data_root / "val", args.val_windows, args.crop_length, args.seed + 1
        )
        payload = {
            "train": build_cache(
                model, train_examples, args.crop_length, args.inference_batch_size,
                device, args.amp,
            ),
            "val": build_cache(
                model, val_examples, args.crop_length, args.inference_batch_size,
                device, args.amp,
            ),
            "checkpoint": str(args.checkpoint.resolve()),
            "train_files": [str(path) for path, _ in train_examples],
            "val_files": [str(path) for path, _ in val_examples],
        }
        torch.save(payload, cache_path)
        del model
        torch.cuda.empty_cache()

    if args.initial_temperature <= 0:
        parser.error("--initial-temperature must be positive")
    hmm = StructuredGeneHMM().to(device)
    with torch.no_grad():
        hmm.frame_log_scale.fill_(math.log(1.0 / args.initial_temperature))
    baseline = evaluate(hmm, payload["val"], args.hmm_batch_size, device)
    history, best_loss = train(
        hmm, payload["train"], payload["val"], args, device, baseline
    )
    final = evaluate(hmm, payload["val"], args.hmm_batch_size, device)
    calibration = hmm.calibration_dict()
    calibration.update({
        "checkpoint": str(args.checkpoint.resolve()),
        "data_root": str(args.data_root.resolve()),
        "split": "EviAnn train/val; positive constraints only",
        "crop_length": args.crop_length,
        "train_windows": args.train_windows,
        "val_windows": args.val_windows,
        "seed": args.seed,
        "baseline_validation": baseline,
        "best_validation_partial_nll": best_loss,
        "final_validation": final,
        "stop_state_coordinate": "third nucleotide; source stop logit shifted +2 bp",
        "acceptor_state_coordinate": "G of AG; source acceptor logit shifted +1 bp",
    })
    (args.output_dir / "structured_frame_calibration.json").write_text(
        json.dumps(calibration, indent=2) + "\n"
    )
    torch.save(
        {"hmm_state_dict": hmm.state_dict(), "calibration": calibration,
         "history": history, "arguments": vars(args)},
        args.output_dir / "best_structured_hmm.pt",
    )
    with (args.output_dir / "history.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps(calibration, indent=2), flush=True)


if __name__ == "__main__":
    main()
