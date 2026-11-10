"""Masked-base CDS diagnostic for the Human MLM belief-state experiment.

Every evaluated CDS nucleotide is replaced by the MLM MASK token in its own
input example.  The model is compared with two matched bidirectional k-mer
models fitted only on the probe-training chromosomes: one ignores codon phase
and one conditions its counts on reference phase 0/1/2.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .mlm_pretrain import CNNMambaMLM
from .probe_belief import (
    CLEAN_CHROMS, TRAIN_CHROMS, discover_windows, verify_window_manifest,
)
from .probe_features import sha256_file


def valid_cds_positions(data, left=2500, right=7500):
    cds = data["reference_cds_labels"][left:right].astype(bool)
    valid = data["reference_phase_mask"][left:right].astype(bool)
    sequence = data["sequence"]
    selected = np.flatnonzero(cds & valid & (sequence[left:right] < 4)) + left
    return selected, data["reference_phase_labels"][selected].astype(np.int8)


def context_codes(sequence, positions, order):
    relative = np.concatenate((
        np.arange(-order, 0, dtype=np.int64),
        np.arange(1, order + 1, dtype=np.int64),
    ))
    context = sequence[positions[:, None] + relative[None, :]].astype(np.int64)
    powers = 5 ** np.arange(context.shape[1], dtype=np.int64)
    return (context * powers).sum(1)


def fit_kmer_models(files, order, output):
    states = 5 ** (2 * order)
    background = np.ones((states, 4), dtype=np.float64)
    periodic = np.ones((3, states, 4), dtype=np.float64)
    phase_counts = np.zeros(3, dtype=np.int64)
    started = time.time()
    for index, path in enumerate(files, 1):
        with np.load(path) as data:
            positions, phases = valid_cds_positions(data)
            positions = positions[
                (positions >= order) & (positions < len(data["sequence"]) - order)
            ]
            if not len(positions):
                continue
            # Re-read phases after the boundary filter to preserve alignment.
            phases = data["reference_phase_labels"][positions].astype(np.int64)
            sequence = data["sequence"].astype(np.int64, copy=False)
            targets = sequence[positions]
            codes = context_codes(sequence, positions, order)
        np.add.at(background, (codes, targets), 1)
        np.add.at(periodic, (phases, codes, targets), 1)
        phase_counts += np.bincount(phases, minlength=3)
        if index % 5000 == 0 or index == len(files):
            print(f"[kmer] {index}/{len(files)} windows", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "kmer_models.npz", background=background,
        periodic=periodic, phase_counts=phase_counts,
    )
    metadata = {
        "order_per_side": order,
        "context_definition": (
            f"{order} transcript-oriented bases on each side; center omitted"
        ),
        "alphabet_size": 5,
        "pseudocount_per_base": 1.0,
        "windows": len(files),
        "phase_counts": phase_counts.tolist(),
        "elapsed_seconds": time.time() - started,
    }
    (output / "kmer_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def load_mlm(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = CNNMambaMLM(
        d_model=int(config.get("d_model", 192)),
        d_state=int(config.get("d_state", 64)),
        n_layers=int(config.get("n_layers", 8)),
        dropout=float(config.get("dropout", 0.1)),
        cnn_kernel_size=int(config.get("cnn_kernel_size", 7)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def update_accumulator(accumulator, name, phases, probabilities, targets):
    predicted = probabilities.argmax(1)
    chosen = probabilities[np.arange(len(targets)), targets]
    for phase in range(3):
        selected = phases == phase
        if not selected.any():
            continue
        row = accumulator[(name, phase)]
        row["positions"] += int(selected.sum())
        row["nll_sum"] += float(-np.log(np.maximum(chosen[selected], 1e-300)).sum())
        row["correct"] += int((predicted[selected] == targets[selected]).sum())


def rows_from_accumulator(accumulator):
    rows = []
    for model in ("mlm", "kmer_background", "kmer_periodic"):
        total_positions = total_nll = total_correct = 0
        for phase in range(3):
            item = accumulator[(model, phase)]
            count = item["positions"]
            rows.append({
                "model": model, "phase": phase, "positions": count,
                "cross_entropy_nats": item["nll_sum"] / count,
                "accuracy": item["correct"] / count,
            })
            total_positions += count
            total_nll += item["nll_sum"]
            total_correct += item["correct"]
        rows.append({
            "model": model, "phase": "all", "positions": total_positions,
            "cross_entropy_nats": total_nll / total_positions,
            "accuracy": total_correct / total_positions,
        })
    return rows


def evaluate(
    files, checkpoint_path, kmer_path, order, output, batch_size,
    max_positions_per_phase=0, seed=42,
):
    saved = np.load(kmer_path)
    background = saved["background"]
    periodic = saved["periodic"]
    background /= background.sum(1, keepdims=True)
    periodic /= periodic.sum(2, keepdims=True)
    device = torch.device("cuda")
    model, checkpoint = load_mlm(checkpoint_path, device)
    accumulator = {
        (name, phase): {"positions": 0, "nll_sum": 0.0, "correct": 0}
        for name in ("mlm", "kmer_background", "kmer_periodic")
        for phase in range(3)
    }
    available = np.zeros(3, dtype=np.int64)
    selected_counts = np.zeros(3, dtype=np.int64)
    rng = np.random.default_rng(seed)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for file_number, path in enumerate(files, 1):
        with np.load(path) as data:
            positions, phases = valid_cds_positions(data)
            keep = (positions >= order) & (positions < len(data["sequence"]) - order)
            positions, phases = positions[keep], phases[keep]
            available += np.bincount(phases, minlength=3)
            if max_positions_per_phase:
                chosen_parts = []
                for phase in range(3):
                    candidates = np.flatnonzero(phases == phase)
                    need = max_positions_per_phase - int(selected_counts[phase])
                    if need > 0 and len(candidates):
                        take = min(need, len(candidates))
                        chosen_parts.append(rng.choice(candidates, take, replace=False))
                if not chosen_parts:
                    continue
                chosen = np.concatenate(chosen_parts)
                positions, phases = positions[chosen], phases[chosen]
            selected_counts += np.bincount(phases, minlength=3)
            sequence = data["sequence"].astype(np.int64, copy=True)
        targets_all = sequence[positions].astype(np.int64)
        codes_all = context_codes(sequence, positions, order)
        update_accumulator(
            accumulator, "kmer_background", phases,
            background[codes_all], targets_all,
        )
        update_accumulator(
            accumulator, "kmer_periodic", phases,
            periodic[phases, codes_all], targets_all,
        )
        for start in range(0, len(positions), batch_size):
            local_positions = positions[start:start + batch_size]
            local_phases = phases[start:start + batch_size]
            targets = targets_all[start:start + batch_size]
            inputs = np.repeat(sequence[None, :], len(local_positions), axis=0)
            inputs[np.arange(len(local_positions)), local_positions] = 5
            tensor = torch.from_numpy(inputs).to(device=device, dtype=torch.long)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                logits = model(tensor)
                selected_logits = logits[
                    torch.arange(len(local_positions), device=device),
                    torch.as_tensor(local_positions, device=device),
                ].float()
                probabilities = selected_logits.softmax(-1).cpu().numpy()
            update_accumulator(
                accumulator, "mlm", local_phases, probabilities, targets,
            )
        if file_number % 25 == 0 or file_number == len(files):
            print(
                f"[masked CDS] {file_number}/{len(files)} windows; "
                f"positions={selected_counts.tolist()}", flush=True,
            )
        if max_positions_per_phase and np.all(
            selected_counts >= max_positions_per_phase
        ):
            break
    rows = rows_from_accumulator(accumulator)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "masked_cds_metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    config = {
        "schema": "cnn_mamba_mlm_masked_cds_diagnostic_v1",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "evaluation_chromosomes": list(CLEAN_CHROMS),
        "label_validity": "reference_phase_mask and reference_cds_labels",
        "single_mask_per_example": True,
        "mask_token": 5,
        "batch_size": batch_size,
        "kmer_order_per_side": order,
        "kmer_model": str(kmer_path.resolve()),
        "max_positions_per_phase": max_positions_per_phase,
        "available_positions_encountered": available.tolist(),
        "evaluated_positions": selected_counts.tolist(),
        "seed": seed,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2], text=True,
        ).strip(),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return rows, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--windows-root", type=Path, default=Path("data/processed/human_joint_frame_w10_chr1held"))
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mlm_pretrain/human_grch38_w10_v1/best_model.pt"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/belief_probe_v1/C_mlm_cds"))
    parser.add_argument("--result-root", type=Path, default=Path("results/belief_probe_v1/C_mlm_cds"))
    parser.add_argument("--stages", default="fit,eval")
    parser.add_argument("--kmer-order", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-positions-per-phase", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    resolve = lambda value: value if value.is_absolute() else (root / value).resolve()
    windows = resolve(args.windows_root)
    checkpoint = resolve(args.checkpoint)
    artifacts = resolve(args.artifact_root)
    results = resolve(args.result_root)
    stages = {item.strip() for item in args.stages.split(",") if item.strip()}
    verification = verify_window_manifest(windows)
    train_files = discover_windows(windows, TRAIN_CHROMS)
    clean_files = discover_windows(windows, CLEAN_CHROMS)
    if "fit" in stages:
        fit_kmer_models(train_files, args.kmer_order, artifacts)
    kmer_path = artifacts / "kmer_models.npz"
    if "eval" in stages:
        rows, config = evaluate(
            clean_files, checkpoint, kmer_path, args.kmer_order, results,
            args.batch_size, args.max_positions_per_phase, args.seed,
        )
        config["window_manifest_verification"] = verification
        config["training_chromosomes"] = list(TRAIN_CHROMS)
        config["training_windows"] = len(train_files)
        config["evaluation_windows"] = len(clean_files)
        (results / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
