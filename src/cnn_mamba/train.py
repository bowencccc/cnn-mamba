#!/usr/bin/env python3
"""Train the reproducible CNN--Mamba-k7 candidate classifier."""

import argparse
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .model import SpliceMamba
from .common import (
    BinnedPR,
    format_metrics,
    initialize_like_original,
    splice_candidate_masks,
    start_stop_candidate_masks,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "processed" / "w10"
SEED = 42


class PUDataset(Dataset):
    def __init__(self, directory):
        directories = directory if isinstance(directory, (list, tuple)) else [directory]
        self.files = sorted(
            str(Path(item) / name)
            for item in directories
            for name in os.listdir(item)
            if name.endswith(".npz")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as data:
            keys = (
                "sequence",
                "labels",
                "start_stop_labels",
                "splice_ignore",
                "start_stop_ignore",
                "chess_labels",
                "chess_start_stop_labels",
            )
            # Some regenerated baseline-only grids omit PU-ignore arrays. They
            # are unused in baseline mode, so materialize all-false masks while
            # preserving the historical tuple layout.
            sequence = data["sequence"]
            arrays = []
            for key in keys:
                if key in data:
                    value = data[key]
                elif key in {"splice_ignore", "start_stop_ignore"}:
                    value = np.zeros_like(sequence)
                else:
                    raise KeyError(f"missing required array {key!r} in {self.files[index]}")
                arrays.append(torch.from_numpy(value.astype(np.int64, copy=False)))
            return tuple(arrays)


def candidate_loss(logits, labels, sequence, task, ignore=None):
    if task == "splice":
        first, second = splice_candidate_masks(sequence)
    else:
        first, second = start_stop_candidate_masks(sequence)
    positive = labels != 0
    mask = first | second | positive
    if ignore is not None:
        # Never suppress an EviAnn positive, even if annotations conflict.
        mask &= (~ignore.bool()) | positive
    if not mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], labels[mask])


@torch.no_grad()
def evaluate(
    model, loader, device, use_amp, center_left=2500, center_right=7500,
    splice_loss_weight=0.25,
):
    model.eval()
    curves = {name: BinnedPR() for name in ("donor", "acceptor", "start", "stop")}
    totals = {"loss": 0.0, "splice": 0.0, "ss": 0.0}
    batches = 0
    for batch in loader:
        sequence = batch[0].to(device, non_blocking=True)
        chess_splice = batch[5].to(device, non_blocking=True)
        chess_ss = batch[6].to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            splice_logits, ss_logits = model(sequence)
            splice_loss = candidate_loss(
                splice_logits, chess_splice, sequence, "splice"
            )
            ss_loss = candidate_loss(
                ss_logits, chess_ss, sequence, "start_stop"
            )
            loss = splice_loss_weight * splice_loss + ss_loss
        totals["loss"] += float(loss.item())
        totals["splice"] += float(splice_loss.item())
        totals["ss"] += float(ss_loss.item())
        batches += 1
        splice_probability = splice_logits.softmax(-1)
        ss_probability = ss_logits.softmax(-1)
        gt, ag = splice_candidate_masks(sequence)
        atg, stop = start_stop_candidate_masks(sequence)
        center = torch.zeros_like(sequence, dtype=torch.bool)
        center[..., center_left:center_right] = True
        for name, probability, labels, mask, cls in (
            ("donor", splice_probability, chess_splice, gt & center, 1),
            ("acceptor", splice_probability, chess_splice, ag & center, 2),
            ("start", ss_probability, chess_ss, atg & center, 1),
            ("stop", ss_probability, chess_ss, stop & center, 2),
        ):
            curves[name].add(probability[..., cls][mask].float(), labels[mask] == cls)
    metrics = {name: curve.metrics() for name, curve in curves.items()}
    losses = {
        "loss": totals["loss"] / batches,
        "splice_loss": totals["splice"] / batches,
        "start_stop_loss": totals["ss"] / batches,
    }
    return metrics, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--mode", choices=("baseline", "ignore"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--splice-loss-weight", type=float, default=0.25,
        help="Weight applied to splice loss; start/stop loss has weight 1.0.",
    )
    parser.add_argument(
        "--architecture", choices=("cnn_mamba", "mamba"), default="cnn_mamba"
    )
    parser.add_argument("--cnn-kernel-size", type=int, default=7)
    parser.add_argument("--window-size", type=int, default=10000)
    parser.add_argument("--stride", type=int, default=5000)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Recompute CNN-Mamba blocks during backward to reduce activation memory.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-name", default="CHESS")
    parser.add_argument("--test-name", default="chr1")
    parser.add_argument(
        "--all-splits-as-train", action="store_true",
        help=(
            "Train a final model on train/val/test together. The epoch count must "
            "already have been selected by an earlier held-out experiment."
        ),
    )
    parser.add_argument(
        "--include-test-in-train",
        action="store_true",
        help=(
            "Merge the test folder into train while retaining the original "
            "validation folder for model selection. No test evaluation is run."
        ),
    )
    args = parser.parse_args()
    if args.all_splits_as_train and args.include_test_in_train:
        parser.error(
            "--all-splits-as-train and --include-test-in-train are mutually exclusive"
        )
    if args.stride != args.window_size // 2:
        parser.error("this experiment requires stride = window_size / 2")
    center_left = (args.window_size - args.stride) // 2
    center_right = center_left + args.stride
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    output_dir = (args.output_dir or (
        ROOT / f"checkpoints_hsap_eviann_pu_pilot_{args.mode}"
    )).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("eviann_pu_pilot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(output_dir / "training_log.txt")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    if args.all_splits_as_train:
        train_data = PUDataset([args.data_root / name for name in ("train", "val", "test")])
        val_data = test_data = None
    elif args.include_test_in_train:
        train_data = PUDataset([args.data_root / name for name in ("train", "test")])
        val_data = PUDataset(args.data_root / "val")
        test_data = None
    else:
        train_data = PUDataset(args.data_root / "train")
        val_data = PUDataset(args.data_root / "val")
        test_data = PUDataset(args.data_root / "test")
    generator = torch.Generator().manual_seed(SEED)
    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=generator,
        drop_last=True, **loader_kwargs
    )
    val_loader = None if val_data is None else DataLoader(
        val_data, batch_size=args.batch_size * 2, shuffle=False, **loader_kwargs
    )
    test_loader = None if test_data is None else DataLoader(
        test_data, batch_size=args.batch_size * 2, shuffle=False, **loader_kwargs
    )

    model = SpliceMamba(
        d_model=192,
        d_state=64,
        n_layers=8,
        dropout=0.1,
        architecture=args.architecture,
        cnn_kernel_size=args.cnn_kernel_size,
    ).to(device)
    model.gradient_checkpointing = args.gradient_checkpointing
    initialize_like_original(model)
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if name.endswith(".bias") or "norm" in name.lower() or "embedding" in name.lower() else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 1e-4}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=args.amp)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.epochs * steps_per_epoch
    warmup = max(20, min(200, total_steps // 10))

    def scheduled_lr(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))

    history, best_score, optimizer_step, start_epoch = [], -1.0, 0, 1
    last_checkpoint = output_dir / "last_model.pt"
    if args.resume and last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        if args.amp and "scaler_state_dict" in saved:
            scaler.load_state_dict(saved["scaler_state_dict"])
        if "data_generator_state" in saved:
            generator.set_state(saved["data_generator_state"])
        history = saved.get("history", [])
        best_score = max(
            (row.get("mean_AP", -1.0) for row in history if row.get("mean_AP") is not None),
            default=-1.0,
        )
        optimizer_step = int(saved.get("optimizer_step", saved["epoch"] * steps_per_epoch))
        start_epoch = int(saved["epoch"]) + 1

    logger.info(
        f"mode={args.mode} device={torch.cuda.get_device_name(0)} train={len(train_data):,} "
        f"val={len(val_data) if val_data is not None else 0:,} "
        f"test={len(test_data) if test_data is not None else 0:,} epochs={args.epochs} "
        f"start_epoch={start_epoch}"
    )
    run_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"loss": 0.0, "splice": 0.0, "ss": 0.0}
        accumulated = 0
        epoch_start = time.time()
        for batch_index, batch in enumerate(train_loader, 1):
            sequence, splice_labels, ss_labels, splice_ignore, ss_ignore = (
                tensor.to(device, non_blocking=True) for tensor in batch[:5]
            )
            lr = scheduled_lr(optimizer_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            use_ignore = args.mode == "ignore"
            with autocast("cuda", enabled=args.amp):
                splice_logits, ss_logits = model(sequence)
                splice_loss = candidate_loss(
                    splice_logits, splice_labels, sequence, "splice",
                    splice_ignore if use_ignore else None,
                )
                ss_loss = candidate_loss(
                    ss_logits, ss_labels, sequence, "start_stop",
                    ss_ignore if use_ignore else None,
                )
                loss = args.splice_loss_weight * splice_loss + ss_loss
            if not torch.isfinite(loss):
                logger.warning(
                    f"non-finite loss at epoch={epoch} batch={batch_index}; "
                    "clearing gradients and skipping batch"
                )
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                continue
            scaled = loss / args.grad_accum
            (scaler.scale(scaled) if args.amp else scaled).backward()
            accumulated += 1
            if accumulated == args.grad_accum:
                if args.amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if args.amp:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                optimizer_step += 1
            totals["loss"] += float(loss.item())
            totals["splice"] += float(splice_loss.item())
            totals["ss"] += float(ss_loss.item())
            if batch_index % 500 == 0:
                logger.info(
                    f"epoch={epoch} batch={batch_index:,}/{len(train_loader):,} "
                    f"loss={totals['loss']/batch_index:.5f} lr={lr:.2e}"
                )
        if accumulated:
            if args.amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if args.amp:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

        if args.all_splits_as_train:
            metrics, validation_losses = None, None
        else:
            metrics, validation_losses = evaluate(
                model, val_loader, device, args.amp, center_left, center_right,
                args.splice_loss_weight,
            )
        mean_ap = None if metrics is None else sum(metrics[name]["AP"] for name in metrics) / 4
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / len(train_loader),
            "train_splice_loss": totals["splice"] / len(train_loader),
            "train_start_stop_loss": totals["ss"] / len(train_loader),
            "validation_loss": None if validation_losses is None else validation_losses["loss"],
            "validation_splice_loss": (
                None if validation_losses is None else validation_losses["splice_loss"]
            ),
            "validation_start_stop_loss": (
                None if validation_losses is None else validation_losses["start_stop_loss"]
            ),
            "validation_chess": metrics,
            "validation_reference": metrics,
            "mean_AP": mean_ap,
            "epoch_seconds": time.time() - epoch_start,
        }
        history.append(row)
        if metrics is None:
            logger.info(
                f"Epoch {epoch}: train loss={row['train_loss']:.5f}; "
                "no held-out evaluation (final all-chromosome training)"
            )
        else:
            logger.info(
                f"Epoch {epoch}: train loss={row['train_loss']:.5f}; "
                f"validation loss={row['validation_loss']:.5f}; "
                f"mean {args.reference_name} AP={mean_ap:.4f}\n    "
                + format_metrics(metrics).replace("\n", "\n    ")
            )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "data_generator_state": generator.get_state(),
            "optimizer_step": optimizer_step,
            "history": history,
            "mean_AP": mean_ap,
            "config": vars(args) | {
                "data_root": str(args.data_root), "output_dir": str(output_dir),
                "d_model": 192, "d_state": 64, "n_layers": 8, "dropout": 0.1,
                "regime": "chrxtrain" if args.include_test_in_train else "heldout",
            },
        }
        torch.save(checkpoint, output_dir / "last_model.pt")
        epoch_dir = output_dir / "epoch_checkpoints"
        epoch_dir.mkdir(exist_ok=True)
        epoch_width = max(3, len(str(args.epochs)))
        torch.save(checkpoint, epoch_dir / f"epoch_{epoch:0{epoch_width}d}.pt")
        if args.all_splits_as_train or mean_ap > best_score:
            best_score = mean_ap
            torch.save(checkpoint, output_dir / "best_model.pt")

    test_metrics = None
    test_losses = None
    if test_loader is not None:
        saved = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model_state_dict"])
        model.to(device)
        test_metrics, test_losses = evaluate(
            model, test_loader, device, args.amp, center_left, center_right,
            args.splice_loss_weight,
        )
    result = {
        "mode": args.mode,
        "best_validation_mean_AP": None if args.all_splits_as_train else best_score,
        "history": history,
        "chr1_chess": test_metrics,
        "test_reference": test_metrics,
        "test_losses": test_losses,
        "elapsed_seconds": time.time() - run_start,
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if test_metrics is not None:
        logger.info(f"{args.test_name} {args.reference_name}\n" + format_metrics(test_metrics))
    logger.info(f"finished in {(time.time() - run_start)/3600:.2f} h")


if __name__ == "__main__":
    main()
