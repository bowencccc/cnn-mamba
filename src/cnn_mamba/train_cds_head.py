#!/usr/bin/env python3
"""Fit a frozen-backbone CDS/non-CDS probe on an existing phase model."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .common import BinnedPR
from .model import SpliceMamba


SEED = 42


class CDSDataset(Dataset):
    def __init__(self, directories):
        if not isinstance(directories, (list, tuple)):
            directories = [directories]
        self.files = sorted(
            str(Path(directory) / name)
            for directory in directories
            for name in os.listdir(directory)
            if name.endswith(".npz")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with np.load(self.files[index]) as data:
            missing = {
                key for key in ("sequence", "cds_labels", "reference_cds_labels")
                if key not in data
            }
            if missing:
                raise KeyError(
                    f"missing {sorted(missing)} in {self.files[index]}; "
                    "regenerate windows with current prepare_droso"
                )
            return tuple(
                torch.from_numpy(data[key].astype(np.int64, copy=False))
                for key in ("sequence", "cds_labels", "reference_cds_labels")
            )


def make_model(base):
    config = base["config"]
    model = SpliceMamba(
        d_model=config.get("d_model", 192),
        d_state=config.get("d_state", 64),
        n_layers=config.get("n_layers", 8),
        dropout=config.get("dropout", 0.1),
        architecture=config.get("architecture", "cnn_mamba"),
        cnn_kernel_size=config.get("cnn_kernel_size", 7),
        frame_dilation=config.get("frame_dilation", 0),
        phase_auxiliary=config.get("phase_aux_weight", 0.0) > 0,
        cds_auxiliary=True,
    )
    incompatible = model.load_state_dict(base["model_state_dict"], strict=False)
    expected_missing = {"cds_head.weight", "cds_head.bias"}
    if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    torch.nn.init.trunc_normal_(model.cds_head.weight, std=0.02)
    torch.nn.init.zeros_(model.cds_head.bias)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("cds_head."))
    return model


@torch.no_grad()
def evaluate(model, loader, device, use_amp, center_left, center_right):
    model.eval()
    curve = BinnedPR()
    loss_sum = 0.0
    positives = 0
    bases = 0
    for sequence, _eviann_cds, reference_cds in loader:
        sequence = sequence.to(device, non_blocking=True)
        reference_cds = reference_cds.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            cds_logits = model(sequence, return_cds=True)[-1]
            loss = F.cross_entropy(
                cds_logits[:, center_left:center_right].reshape(-1, 2),
                reference_cds[:, center_left:center_right].reshape(-1),
            )
        labels = reference_cds[:, center_left:center_right].bool()
        scores = cds_logits[:, center_left:center_right].softmax(-1)[..., 1]
        curve.add(scores.float().reshape(-1), labels.reshape(-1))
        loss_sum += float(loss.item())
        positives += int(labels.sum())
        bases += labels.numel()
    return loss_sum / len(loader), curve.metrics(), positives / bases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--include-test-in-train", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.epochs < 1 or args.early_stopping_patience < 1:
        parser.error("epochs and early-stopping patience must be positive")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_cds_head")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler = logging.FileHandler(args.output_dir / "training_log.txt")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    base = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    config = base["config"]
    if float(config.get("phase_aux_weight", 0.0)) <= 0:
        raise RuntimeError("base checkpoint must contain a trained phase head")
    window = int(config["window_size"])
    stride = int(config["stride"])
    center_left = (window - stride) // 2
    center_right = center_left + stride
    train_dirs = [args.data_root / "train"]
    if args.include_test_in_train:
        train_dirs.append(args.data_root / "test")
    train_data = CDSDataset(train_dirs)
    val_data = CDSDataset(args.data_root / "val")
    loader_kwargs = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        generator=generator, drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size * 2, shuffle=False, **loader_kwargs,
    )

    device = torch.device("cuda")
    model = make_model(base).to(device)
    optimizer = torch.optim.AdamW(
        model.cds_head.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = GradScaler(enabled=args.amp)
    history = []
    best_ap = -1.0
    epochs_without_improvement = 0
    stopped_epoch = None
    optimizer_step = 0
    started = time.time()
    logger.info(
        f"device={torch.cuda.get_device_name(0)} train={len(train_data):,} "
        f"val={len(val_data):,} batch={args.batch_size} lr={args.lr:g} "
        f"base_epoch={base.get('epoch')} phase_weight="
        f"{config.get('phase_aux_weight')} frozen_backbone=True"
    )

    for epoch in range(1, args.epochs + 1):
        # Keep every frozen dropout layer deterministic while optimizing the probe.
        model.eval()
        model.cds_head.train()
        epoch_started = time.time()
        loss_sum = 0.0
        positives = 0
        bases = 0
        for batch_index, (sequence, eviann_cds, _reference_cds) in enumerate(
            train_loader, 1
        ):
            sequence = sequence.to(device, non_blocking=True)
            eviann_cds = eviann_cds.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp):
                cds_logits = model(sequence, return_cds=True)[-1]
                loss = F.cross_entropy(
                    cds_logits.reshape(-1, 2), eviann_cds.reshape(-1)
                )
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer_step += 1
            loss_sum += float(loss.item())
            positives += int(eviann_cds.sum())
            bases += eviann_cds.numel()
            if batch_index % 500 == 0:
                logger.info(
                    f"epoch={epoch} batch={batch_index:,}/{len(train_loader):,} "
                    f"cds_loss={loss_sum / batch_index:.5f}"
                )

        validation_loss, validation_metrics, validation_fraction = evaluate(
            model, val_loader, device, args.amp, center_left, center_right
        )
        validation_ap = float(validation_metrics["AP"])
        improved = validation_ap > best_ap + args.early_stopping_min_delta
        if improved:
            best_ap = validation_ap
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "train_cds_loss": loss_sum / len(train_loader),
            "validation_cds_loss": validation_loss,
            "train_cds_fraction": positives / bases,
            "validation_reference_cds_fraction": validation_fraction,
            "validation_cds": validation_metrics,
            "validation_cds_AP": validation_ap,
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(row)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer_step": optimizer_step,
            "history": history,
            "best_validation_cds_AP": best_ap,
            "epochs_without_improvement": epochs_without_improvement,
            "base_checkpoint": str(args.base_checkpoint.resolve()),
            "config": config | {
                "data_root": str(args.data_root),
                "output_dir": str(args.output_dir),
                "cds_auxiliary": True,
                "cds_aux_weight": 1.0,
                "cds_head_only": True,
                "cds_label_source": "EviAnn CDS interval union",
                "cds_validation_source": "FlyBase CDS interval union",
                "include_test_in_train": args.include_test_in_train,
            },
        }
        torch.save(checkpoint, args.output_dir / "last_model.pt")
        if improved:
            torch.save(checkpoint, args.output_dir / "best_model.pt")
        logger.info(
            f"Epoch {epoch}: train CDS loss={row['train_cds_loss']:.5f}; "
            f"validation CDS loss={validation_loss:.5f}; AP={validation_ap:.6f}; "
            f"F1={validation_metrics['F1']:.6f}; "
            f"P={validation_metrics['P']:.6f}; R={validation_metrics['R']:.6f}"
        )
        if epochs_without_improvement >= args.early_stopping_patience:
            stopped_epoch = epoch
            logger.info(
                f"Early stopping at epoch {epoch}; best validation CDS "
                f"AP={best_ap:.6f}"
            )
            break

    # The probe experiment must not alter any parameter from the source model.
    final_state = model.state_dict()
    for name, expected in base["model_state_dict"].items():
        if not torch.equal(final_state[name].detach().cpu(), expected):
            raise RuntimeError(f"frozen parameter changed: {name}")
    result = {
        "schema": "frozen_backbone_cds_head_training_v1",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "best_validation_cds_AP": best_ap,
        "history": history,
        "stopped_epoch": stopped_epoch,
        "frozen_parameters_unchanged": True,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    logger.info(f"finished in {(time.time() - started) / 3600:.2f} h")


if __name__ == "__main__":
    main()
