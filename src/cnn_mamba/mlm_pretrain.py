#!/usr/bin/env python3
"""Self-supervised masked-base pretraining for the CNN--Mamba backbone."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sys
import time

import numpy as np
import pysam
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from .common import initialize_like_original
from .human_annotations import CHR_TO_NC
from .model import SpliceMamba
from .train import cosine_learning_rate


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FASTA = ROOT / "data/raw/human_grch38.fa"
ENCODE = np.full(256, 4, dtype=np.uint8)
for _base, _value in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    ENCODE[ord(_base)] = _value
COMPLEMENT = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)
BACKBONE_PREFIXES = ("embedding.", "input_proj.", "layers.", "norms.", "final_norm.")


class GenomeWindowDataset(Dataset):
    """Fixed full-length windows fetched lazily from an indexed FASTA."""

    def __init__(self, fasta_path, accessions, window_size, stride,
                 reverse_complement=False, min_acgt_fraction=0.5):
        self.fasta_path = str(Path(fasta_path).resolve())
        self.window_size = int(window_size)
        self.reverse_complement = bool(reverse_complement)
        self.minimum_acgt = int(math.ceil(window_size * min_acgt_fraction))
        self._fasta = None
        with pysam.FastaFile(self.fasta_path) as fasta:
            self.windows = [
                (accession, start)
                for accession in accessions
                for start in range(
                    0,
                    fasta.get_reference_length(accession) - self.window_size + 1,
                    stride,
                )
            ]

    def __len__(self):
        return len(self.windows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fasta"] = None
        return state

    def __getitem__(self, index):
        if self._fasta is None:
            self._fasta = pysam.FastaFile(self.fasta_path)
        # Assembly gaps can make an indexed primary-chromosome window nearly
        # all N. Deterministically redirect those entries rather than feeding
        # empty MLM targets or paying for a 3-Gb manifest scan at every run.
        for attempt in range(101):
            accession, start = self.windows[index]
            text = self._fasta.fetch(
                accession, start, start + self.window_size
            ).upper()
            if len(text) != self.window_size:
                raise RuntimeError(f"short FASTA fetch: {accession}:{start}")
            encoded_array = ENCODE[
                np.frombuffer(text.encode("ascii"), dtype=np.uint8)
            ].copy()
            if int(np.count_nonzero(encoded_array < 4)) >= self.minimum_acgt:
                break
            index = (index + 104_729) % len(self.windows)
        else:
            raise RuntimeError("could not locate an A/C/G/T-rich genome window")
        encoded = torch.from_numpy(encoded_array).long()
        if self.reverse_complement and bool(torch.rand(()) < 0.5):
            encoded = COMPLEMENT[encoded.flip(0)]
        return encoded


class CNNMambaMLM(nn.Module):
    """Six-token encoder with a four-base masked reconstruction head."""

    def __init__(self, d_model=192, d_state=64, n_layers=8, dropout=0.1,
                 cnn_kernel_size=7):
        super().__init__()
        self.backbone = SpliceMamba(
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            architecture="cnn_mamba",
            cnn_kernel_size=cnn_kernel_size,
            vocab_size=6,
        )
        for head in (self.backbone.splice_head, self.backbone.start_stop_head):
            head.requires_grad_(False)
        self.mlm_head = nn.Linear(d_model, 4)
        initialize_like_original(self.backbone)
        nn.init.trunc_normal_(self.mlm_head.weight, std=0.02)
        nn.init.zeros_(self.mlm_head.bias)

    def forward(self, sequence):
        return self.mlm_head(self.backbone.encode(sequence))

    def transfer_state_dict(self):
        """Return strict downstream backbone weights; discard MASK/head state."""
        transferred = {}
        for name, value in self.backbone.state_dict().items():
            if not name.startswith(BACKBONE_PREFIXES):
                continue
            if name == "embedding.weight":
                value = value[:5]
            transferred[name] = value.detach().cpu().clone()
        return transferred


def corrupt_for_mlm(sequence, mask_rate=0.15, span_length=3, generator=None):
    """Apply fixed-length span masking and BERT's 80/10/10 corruption rule."""
    if not 0 < mask_rate < 1:
        raise ValueError("mask_rate must be in (0, 1)")
    if span_length < 1:
        raise ValueError("span_length must be positive")
    start_probability = 1.0 - (1.0 - mask_rate) ** (1.0 / span_length)
    starts = torch.rand(
        sequence.shape, device=sequence.device, generator=generator
    ) < start_probability
    selected = starts.clone()
    for offset in range(1, span_length):
        selected[:, offset:] |= starts[:, :-offset]
    selected &= sequence < 4
    targets = torch.full_like(sequence, -100)
    targets[selected] = sequence[selected]
    corrupted = sequence.clone()
    choices = torch.rand(
        sequence.shape, device=sequence.device, generator=generator
    )
    replace_mask = selected & (choices < 0.8)
    replace_random = selected & (choices >= 0.8) & (choices < 0.9)
    corrupted[replace_mask] = 5
    random_bases = torch.randint(
        0, 4, sequence.shape, device=sequence.device, generator=generator
    )
    corrupted[replace_random] = random_bases[replace_random]
    return corrupted, targets


def evaluate(model, loader, device, amp, mask_rate, span_length, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    nll_sum = correct = masked = 0
    with torch.inference_mode():
        for sequence in loader:
            sequence = sequence.to(device, non_blocking=True)
            inputs, targets = corrupt_for_mlm(
                sequence, mask_rate, span_length, generator
            )
            count = int((targets != -100).sum())
            if count == 0:
                continue
            with autocast("cuda", enabled=amp):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, 4), targets.reshape(-1),
                    ignore_index=-100,
                )
            chosen = targets != -100
            nll_sum += float(loss) * count
            correct += int((logits.argmax(-1)[chosen] == targets[chosen]).sum())
            masked += count
    if masked == 0:
        raise RuntimeError("validation produced no masked A/C/G/T bases")
    return {"loss": nll_sum / masked, "accuracy": correct / masked,
            "masked_bases": masked}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-chrom", default="chr22")
    parser.add_argument("--window-size", type=int, default=10_000)
    parser.add_argument("--stride", type=int, default=10_000)
    parser.add_argument("--train-windows-per-epoch", type=int, default=50_000)
    parser.add_argument("--validation-windows", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--span-length", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.validation_chrom not in CHR_TO_NC:
        parser.error(f"unknown primary chromosome: {args.validation_chrom}")
    if args.window_size < 1 or args.stride < 1:
        parser.error("window size and stride must be positive")
    if args.train_windows_per_epoch < 1 or args.validation_windows < 1:
        parser.error("window limits must be positive")
    if args.batch_size < 1 or args.grad_accum < 1:
        parser.error("batch size and gradient accumulation must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("human_mlm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(output / "training_log.txt")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    train_chroms = [
        chrom for chrom in CHR_TO_NC if chrom != args.validation_chrom
    ]
    train_data = GenomeWindowDataset(
        args.fasta,
        [CHR_TO_NC[chrom] for chrom in train_chroms],
        args.window_size,
        args.stride,
        reverse_complement=True,
    )
    validation_all = GenomeWindowDataset(
        args.fasta,
        [CHR_TO_NC[args.validation_chrom]],
        args.window_size,
        args.stride,
        reverse_complement=False,
    )
    validation_count = min(args.validation_windows, len(validation_all))
    validation_indices = np.linspace(
        0, len(validation_all) - 1, validation_count, dtype=np.int64
    ).tolist()
    validation_data = Subset(validation_all, validation_indices)
    sample_count = min(args.train_windows_per_epoch, len(train_data))
    data_generator = torch.Generator().manual_seed(args.seed)
    sampler = RandomSampler(
        train_data,
        replacement=False,
        num_samples=sample_count,
        generator=data_generator,
    )
    loader_kwargs = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size * 2,
        shuffle=False,
        **loader_kwargs,
    )

    model = CNNMambaMLM().to(device)
    model.backbone.gradient_checkpointing = args.gradient_checkpointing
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if (
            name.endswith(".bias") or "norm" in name.lower()
            or "embedding" in name.lower()
        ) else decay
        target.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    scaler = GradScaler("cuda", enabled=args.amp)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    schedule_steps = args.epochs * steps_per_epoch
    warmup_steps = max(20, int(round(0.05 * schedule_steps)))

    history = []
    best_loss = math.inf
    epochs_without_improvement = 0
    optimizer_step = 0
    start_epoch = 1
    last_path = output / "last_model.pt"
    if args.resume and last_path.is_file():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        history = saved["history"]
        best_loss = float(saved["best_validation_loss"])
        epochs_without_improvement = int(saved["epochs_without_improvement"])
        optimizer_step = int(saved["optimizer_step"])
        start_epoch = int(saved["epoch"]) + 1
        if "data_generator_state" in saved:
            data_generator.set_state(saved["data_generator_state"])

    logger.info(
        f"device={torch.cuda.get_device_name(0)} train_chromosomes="
        f"{','.join(train_chroms)} validation_chromosome={args.validation_chrom} "
        f"available_train_windows={len(train_data):,} sampled_per_epoch={sample_count:,} "
        f"validation_windows={len(validation_data):,} window={args.window_size:,} "
        f"stride={args.stride:,} batch={args.batch_size}x{args.grad_accum} "
        f"mask_rate={args.mask_rate:g} span_length={args.span_length} "
        f"epochs={args.epochs} start_epoch={start_epoch}"
    )
    started = time.time()
    stopped_epoch = None
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_nll_sum = train_correct = train_masked = 0
        epoch_started = time.time()
        for batch_index, sequence in enumerate(train_loader, 1):
            sequence = sequence.to(device, non_blocking=True)
            inputs, targets = corrupt_for_mlm(
                sequence, args.mask_rate, args.span_length
            )
            count = int((targets != -100).sum())
            with autocast("cuda", enabled=args.amp):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, 4), targets.reshape(-1),
                    ignore_index=-100,
                )
            scaler.scale(loss / args.grad_accum).backward()
            chosen = targets != -100
            train_nll_sum += float(loss.detach()) * count
            train_correct += int(
                (logits.detach().argmax(-1)[chosen] == targets[chosen]).sum()
            )
            train_masked += count
            update = batch_index % args.grad_accum == 0 or batch_index == len(train_loader)
            if update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters()
                     if parameter.requires_grad],
                    1.0,
                )
                lr = cosine_learning_rate(
                    optimizer_step, args.lr, schedule_steps, warmup_steps
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            if batch_index % 1_000 == 0:
                logger.info(
                    f"epoch={epoch} batch={batch_index:,}/{len(train_loader):,} "
                    f"train_loss={train_nll_sum / max(train_masked, 1):.5f} "
                    f"train_acc={train_correct / max(train_masked, 1):.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3g}"
                )

        validation = evaluate(
            model,
            validation_loader,
            device,
            args.amp,
            args.mask_rate,
            args.span_length,
            args.seed + 10_000,
        )
        train_loss = train_nll_sum / train_masked
        train_accuracy = train_correct / train_masked
        improved = validation["loss"] < (
            best_loss - args.early_stopping_min_delta
        )
        if improved:
            best_loss = validation["loss"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_masked_bases": train_masked,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_masked_bases": validation["masked_bases"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(row)
        logger.info(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"train_acc={train_accuracy:.4f} val_loss={validation['loss']:.6f} "
            f"val_acc={validation['accuracy']:.4f} improved={improved} "
            f"seconds={row['epoch_seconds']:.1f}"
        )
        checkpoint = {
            "schema": "cnn_mamba_human_mlm_v1",
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "backbone_state_dict": model.transfer_state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "data_generator_state": data_generator.get_state(),
            "optimizer_step": optimizer_step,
            "history": history,
            "best_validation_loss": best_loss,
            "epochs_without_improvement": epochs_without_improvement,
            "config": {
                **vars(args),
                "fasta": str(args.fasta.resolve()),
                "output_dir": str(output),
                "train_chromosomes": train_chroms,
                "d_model": 192,
                "d_state": 64,
                "n_layers": 8,
                "dropout": 0.1,
                "cnn_kernel_size": 7,
                "mask_token": 5,
                "base_order": ["A", "C", "G", "T", "N", "MASK"],
            },
        }
        torch.save(checkpoint, last_path)
        if improved:
            torch.save(checkpoint, output / "best_model.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_epoch = epoch
            logger.info(
                f"early stopping at epoch {epoch}; best validation loss="
                f"{best_loss:.6f}"
            )
            break

    result = {
        "schema": "cnn_mamba_human_mlm_results_v1",
        "best_validation_loss": best_loss,
        "history": history,
        "early_stopped": stopped_epoch is not None,
        "stopped_epoch": stopped_epoch or history[-1]["epoch"],
        "elapsed_seconds": time.time() - started,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / ".complete.json").write_text(json.dumps({
        "schema": "cnn_mamba_human_mlm_complete_v1",
        "best_checkpoint": str((output / "best_model.pt").resolve()),
        "best_validation_loss": best_loss,
        "stopped_epoch": result["stopped_epoch"],
    }, indent=2) + "\n")
    logger.info(
        f"complete best_validation_loss={best_loss:.6f} "
        f"stopped_epoch={result['stopped_epoch']}"
    )


if __name__ == "__main__":
    main()
