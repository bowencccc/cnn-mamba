"""Run frozen-backbone belief-state probes for coding frame (Experiment A).

Selections are chromosome explicit and saved independently of features.  The
same sampled positions are reused by MLM, random, supervised and one-hot
controls.  Features are written as float16 NumPy memmaps under ``artifacts``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from .probe_features import (
    BackboneFeatureHooks, FEATURE_NAMES, FP16_STORAGE_SCALES,
    load_probe_backbone, sha256_file,
)
from .probe_metrics import fit_linear_probe, four_state_metrics, predict_probabilities


TRAIN_CHROMS = ("2", "3", "4", "6", "7", "8") + tuple(map(str, range(11, 22))) + ("X", "Y")
TUNE_CHROMS = ("5", "9", "10")
CLEAN_CHROMS = ("22",)
TEST_CHROMS = ("1",)
REQUIRED_WINDOW_ARRAYS = {
    "sequence", "phase_labels", "phase_mask", "reference_phase_labels",
    "reference_phase_mask", "cds_labels", "reference_cds_labels",
}


def chromosome(path: Path):
    return path.stem.split("_", 1)[0]


def discover_windows(root: Path, chroms):
    wanted = set(chroms)
    files = []
    for split in ("train", "val", "test"):
        files.extend(p for p in (root / split).glob("*.npz") if chromosome(p) in wanted)
    return sorted(files)


def verify_window_manifest(root: Path):
    """Fail before probing if the reused window geometry is not uniform."""
    manifest_path = root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "window_size": 10000, "stride": 5000,
        "region_overlap_bp": 10000, "max_windows_per_chrom_strand": 0,
    }
    observed = {name: manifest.get(name) for name in expected}
    if observed != expected:
        raise RuntimeError(
            f"window parameter mismatch; expected={expected}, observed={observed}"
        )
    per_split, probes = {}, {}
    for split in ("train", "val", "test"):
        counts = {}
        representatives = {}
        files = sorted((root / split).glob("*.npz"))
        for path in files:
            parts = path.stem.split("_")
            if len(parts) != 3 or parts[1] not in {"plus", "minus"}:
                raise RuntimeError(f"unrecognized window filename: {path}")
            chrom, strand, start_text = parts
            start = int(start_text)
            if start % expected["stride"]:
                raise RuntimeError(f"off-stride window: {path}")
            counts[chrom] = counts.get(chrom, 0) + 1
            representatives.setdefault((chrom, strand), path)
        stated = {
            key.removeprefix("chr"): value
            for key, value in manifest["summary"][split]["by_chrom"].items()
        }
        if counts != stated:
            raise RuntimeError(
                f"manifest/file count mismatch in {split}: stated={stated}, actual={counts}"
            )
        for (chrom, strand), path in representatives.items():
            with np.load(path) as data:
                missing = REQUIRED_WINDOW_ARRAYS - set(data.files)
                shapes = {key: tuple(data[key].shape) for key in REQUIRED_WINDOW_ARRAYS if key in data}
            if missing or any(shape != (10000,) for shape in shapes.values()):
                raise RuntimeError(
                    f"invalid representative window {path}: missing={missing}, shapes={shapes}"
                )
            probes[f"{split}:chr{chrom}:{strand}"] = str(path.resolve())
        per_split[split] = {f"chr{key}": value for key, value in counts.items()}
    if "chr2" not in manifest["splits"]["val"] or "chr2" in manifest["splits"]["train"]:
        raise RuntimeError("expected existing chr2 windows in val only before probe reassignment")
    return {
        "status": "passed", "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "parameters": observed, "actual_window_counts": per_split,
        "representative_array_checks": len(probes),
        "chr2_reassignment": "existing val/chr2 files are included in probe train without moving or regenerating",
    }


def labels_four_state(path, left=2500, right=7500):
    with np.load(path) as data:
        cds = data["reference_cds_labels"][left:right].astype(bool)
        phase = data["reference_phase_labels"][left:right].astype(np.int8)
        valid_phase = data["reference_phase_mask"][left:right].astype(bool)
    valid = (~cds) | valid_phase
    labels = np.zeros(right - left, dtype=np.int8)
    labels[cds & valid_phase] = phase[cds & valid_phase] + 1
    return labels, valid


def make_selection(files, target, balanced, seed, max_per_window, output):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    file_indices, offsets, labels_out = [], [], []
    quotas = np.full(4, target // 4, dtype=np.int64) if balanced else None
    # Natural-frequency samples should span many independent windows rather
    # than exhaust the first few files in a shuffled list.
    natural_cap = min(max_per_window, max(1, target // 1000))
    counts = np.zeros(4, dtype=np.int64)
    for file_index in order:
        labels, valid = labels_four_state(files[file_index])
        if balanced:
            for cls in range(4):
                need = int(quotas[cls] - counts[cls])
                if need <= 0:
                    continue
                candidates = np.flatnonzero(valid & (labels == cls))
                take = min(need, max_per_window, len(candidates))
                if take:
                    chosen = rng.choice(candidates, take, replace=False)
                    file_indices.append(np.full(take, file_index, dtype=np.int32))
                    offsets.append((chosen + 2500).astype(np.int16))
                    labels_out.append(np.full(take, cls, dtype=np.int8))
                    counts[cls] += take
            if np.all(counts >= quotas):
                break
        else:
            need = target - int(counts.sum())
            if need <= 0:
                break
            candidates = np.flatnonzero(valid)
            take = min(need, natural_cap, len(candidates))
            if take:
                chosen = rng.choice(candidates, take, replace=False)
                selected_labels = labels[chosen]
                file_indices.append(np.full(take, file_index, dtype=np.int32))
                offsets.append((chosen + 2500).astype(np.int16))
                labels_out.append(selected_labels)
                counts += np.bincount(selected_labels, minlength=4)
    if balanced and np.any(counts < quotas):
        raise RuntimeError(f"could not meet class quotas: got={counts} target={quotas}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "positions.npz",
        file_index=np.concatenate(file_indices),
        offset=np.concatenate(offsets),
        label=np.concatenate(labels_out),
    )
    (output / "files.json").write_text(json.dumps([str(p.resolve()) for p in files], indent=2) + "\n")
    metadata = {
        "positions": int(counts.sum()), "class_counts": counts.tolist(),
        "balanced": balanced, "seed": seed, "max_per_window": max_per_window,
        "windows_available": len(files),
        "windows_sampled": len(set(np.concatenate(file_indices).tolist())),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_selection(path):
    arrays = np.load(path / "positions.npz")
    files = [Path(x) for x in json.loads((path / "files.json").read_text())]
    return files, arrays["file_index"], arrays["offset"], arrays["label"]


def extract_backbone_features(
    selection, kind, checkpoint, output, device="cuda", window_batch_size=4,
    overwrite=False, backbone_seed=42,
):
    files, file_index, offsets, labels = load_selection(selection)
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "features.npy"
    metadata_path = output / "metadata.json"
    checkpoint_hash = None if checkpoint is None else sha256_file(checkpoint)
    if not overwrite and feature_path.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text())
        expected_shape = [len(labels), len(FEATURE_NAMES), 192]
        if (
            existing.get("backbone") == kind
            and existing.get("positions") == len(labels)
            and existing.get("shape") == expected_shape
            and existing.get("checkpoint_sha256") == checkpoint_hash
            and (
                kind != "random" or existing.get("backbone_seed") == backbone_seed
            )
            and list(np.load(feature_path, mmap_mode="r").shape) == expected_shape
        ):
            print(f"[{kind}] reuse complete features: {feature_path}", flush=True)
            return existing
    destination = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float16,
        shape=(len(labels), len(FEATURE_NAMES), 192),
    )
    if kind == "random":
        torch.manual_seed(backbone_seed)
    model = load_probe_backbone(kind, checkpoint).to(device)
    started = time.time()
    order = np.argsort(file_index, kind="stable")
    boundaries = np.flatnonzero(np.diff(file_index[order])) + 1
    groups = np.split(order, boundaries)
    with BackboneFeatureHooks(model) as hooks:
        for first in range(0, len(groups), window_batch_size):
            batch_groups = groups[first:first + window_batch_size]
            batch_indices = [int(file_index[rows[0]]) for rows in batch_groups]
            sequences = []
            for index in batch_indices:
                with np.load(files[index]) as data:
                    sequences.append(
                        torch.from_numpy(data["sequence"].astype(np.int64, copy=False))
                    )
            sequence = torch.stack(sequences).to(device)
            outputs = hooks.collect(sequence)
            scales = torch.tensor(
                FP16_STORAGE_SCALES, device=device, dtype=outputs["input"].dtype
            )[None, :, None]
            for batch_index, (index, rows) in enumerate(zip(batch_indices, batch_groups)):
                pos = torch.from_numpy(offsets[rows].astype(np.int64)).to(device)
                stacked = torch.stack(
                    [outputs[name][batch_index, pos] for name in FEATURE_NAMES], dim=1,
                )
                stored = (stacked * scales).half()
                if not torch.isfinite(stored).all():
                    raise FloatingPointError(
                        "non-finite float16 feature after storage scaling: "
                        f"{files[index]}"
                    )
                destination[rows] = stored.cpu().numpy()
            done = min(first + window_batch_size, len(groups))
            if done % 100 < window_batch_size or done == len(groups):
                destination.flush()
                print(f"[{kind}] {done}/{len(groups)} windows", flush=True)
    destination.flush()
    meta = {
        "backbone": kind, "checkpoint": None if checkpoint is None else str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "positions": len(labels), "feature_names": list(FEATURE_NAMES),
        "dtype": "float16", "shape": list(destination.shape),
        "per_layer_storage_scales": dict(zip(FEATURE_NAMES, FP16_STORAGE_SCALES)),
        "storage_scaling_note": "fixed positive layer scaling preserves linear decodability",
        "elapsed_seconds": time.time() - started,
        "window_batch_size": window_batch_size,
        "backbone_seed": backbone_seed if kind == "random" else None,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
    }
    metadata_path.write_text(json.dumps(meta, indent=2) + "\n")
    del model, destination
    torch.cuda.empty_cache()
    return meta


def onehot_features(selection, radius, output):
    files, file_index, offsets, labels = load_selection(selection)
    width = 2 * radius + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    destination = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float16, shape=(len(labels), width * 4)
    )
    lookup = np.concatenate(
        [np.eye(4, dtype=np.float16), np.zeros((1, 4), dtype=np.float16)], axis=0,
    )
    relative = np.arange(-radius, radius + 1, dtype=np.int64)
    order = np.argsort(file_index, kind="stable")
    boundaries = np.flatnonzero(np.diff(file_index[order])) + 1
    for rows in np.split(order, boundaries):
        index = int(file_index[rows[0]])
        with np.load(files[index]) as data:
            sequence = data["sequence"]
        indices = offsets[rows].astype(np.int64)[:, None] + relative[None, :]
        bases = sequence[indices]
        destination[rows] = lookup[bases].reshape(len(rows), -1)
    destination.flush()


def feature_view(array, layer):
    if layer == "concat_all":
        return array.reshape(len(array), -1)
    return array[:, FEATURE_NAMES.index(layer), :]


def contiguous_feature_view(array, layer):
    """Materialize strided per-layer memmap views once for repeated epochs."""
    view = feature_view(array, layer)
    return view if view.flags.c_contiguous else np.ascontiguousarray(view)


def select_and_evaluate(
    train_x, train_y, eval_features, eval_labels, input_dim, weight_decays,
    seed, epochs,
):
    candidates = []
    for wd in weight_decays:
        probe = fit_linear_probe(
            train_x, train_y, input_dim, 4, wd, seed=seed, epochs=epochs,
        )
        tune_p = predict_probabilities(probe, eval_features["tune"])
        tune_m = four_state_metrics(tune_p, eval_labels["tune"])
        candidates.append((tune_m["balanced_accuracy_4class"], wd, probe))
    _, wd, probe = max(candidates, key=lambda x: x[0])
    metrics = {}
    for split in ("tune", "chr22", "chr1"):
        probabilities = predict_probabilities(probe, eval_features[split])
        metrics[split] = four_state_metrics(probabilities, eval_labels[split])
    return wd, metrics


def run_probes(
    artifact_root, result_root, backbones, weight_decays, seed, epochs,
    onehot_radii=(15, 50, 150),
):
    result_root.mkdir(parents=True, exist_ok=True)
    rows = []
    layers = list(FEATURE_NAMES) + ["concat_all"]
    labels = {
        split: load_selection(artifact_root / "selections" / split)[3]
        for split in ("train", "tune", "chr22", "chr1")
    }
    for backbone in backbones:
        arrays = {
            split: np.load(artifact_root / "features" / backbone / split / "features.npy", mmap_mode="r")
            for split in labels
        }
        for layer in layers:
            train_x = contiguous_feature_view(arrays["train"], layer)
            eval_x = {
                split: contiguous_feature_view(arrays[split], layer)
                for split in ("tune", "chr22", "chr1")
            }
            wd, evaluated = select_and_evaluate(
                train_x, labels["train"], eval_x, labels, train_x.shape[1],
                weight_decays, seed, epochs,
            )
            for split, metrics in evaluated.items():
                rows.append({"backbone": backbone, "layer": layer, "evaluation": split,
                             "weight_decay": wd, **metrics})
            if backbone == "mlm":
                shuffled = np.random.default_rng(seed).permutation(labels["train"])
                shuffled_wd, shuffled_eval = select_and_evaluate(
                    train_x, shuffled, eval_x, labels, train_x.shape[1],
                    weight_decays, seed, epochs,
                )
                for split, metrics in shuffled_eval.items():
                    rows.append({"backbone": "shuffled", "layer": layer,
                                 "evaluation": split, "weight_decay": shuffled_wd,
                                 **metrics})
            print(f"[probe A] backbone={backbone} layer={layer} wd={wd:g}", flush=True)
            del train_x, eval_x
    for radius in onehot_radii:
        arrays = {
            split: np.load(
                artifact_root / "features" / f"onehot_{radius}" / split / "features.npy",
                mmap_mode="r",
            ) for split in labels
        }
        wd, evaluated = select_and_evaluate(
            arrays["train"], labels["train"],
            {split: arrays[split] for split in ("tune", "chr22", "chr1")},
            labels, arrays["train"].shape[1], weight_decays, seed, epochs,
        )
        for split, metrics in evaluated.items():
            rows.append({"backbone": "onehot", "layer": f"onehot_{radius}",
                         "evaluation": split, "weight_decay": wd, **metrics})
        print(f"[probe A] onehot radius={radius} wd={wd:g}", flush=True)
    fields = list(rows[0])
    with (result_root / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    plot_metrics(rows, result_root)
    return rows


def plot_metrics(rows, output):
    evaluation = "chr22"
    layer_order = list(FEATURE_NAMES) + ["concat_all"]
    metrics = [
        ("conditional_frame_accuracy_3class", "Conditional CDS frame accuracy"),
        ("conditional_frame_macro_f1_3class", "Conditional CDS frame macro-F1"),
        ("coding_auprc", "Coding vs noncoding AUPRC"),
        ("balanced_accuracy_4class", "4-class balanced accuracy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        for backbone in ("mlm", "random", "supervised", "shuffled"):
            selected = {r["layer"]: r for r in rows if r["evaluation"] == evaluation and r["backbone"] == backbone}
            if selected:
                axis.plot(
                    range(len(layer_order)), [selected[x][metric] for x in layer_order],
                    marker="o", linewidth=1.5, label=backbone,
                )
        for row in rows:
            if row["evaluation"] == evaluation and row["backbone"] == "onehot":
                axis.axhline(row[metric], linestyle="--", linewidth=1,
                             label=row["layer"])
        axis.set_title(title)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xticks(range(len(layer_order)), layer_order, rotation=55, ha="right")
    axes[1, 1].set_xticks(range(len(layer_order)), layer_order, rotation=55, ha="right")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(output / "metrics_vs_layer.png", dpi=220)
    fig.savefig(output / "metrics_vs_layer.pdf")
    plt.close(fig)


def git_commit(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--windows-root", type=Path, default=Path("data/processed/human_joint_frame_w10_chr1held"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/belief_probe_v1/A_frame"))
    parser.add_argument("--result-root", type=Path, default=Path("results/belief_probe_v1/A_frame"))
    parser.add_argument("--mlm-checkpoint", type=Path, default=Path("runs/mlm_pretrain/human_grch38_w10_v1/best_model.pt"))
    parser.add_argument("--supervised-checkpoint", type=Path, default=Path("models/human_joint_frame_beta010_chr1held_epoch4_inference.pt"))
    parser.add_argument("--stages", default="sample,features,probe")
    parser.add_argument("--backbones", default="mlm,random,supervised")
    parser.add_argument("--train-positions", type=int, default=2_000_000)
    parser.add_argument("--tune-positions", type=int, default=500_000)
    parser.add_argument("--evaluation-positions", type=int, default=2_000_000)
    parser.add_argument("--max-per-window", type=int, default=2500)
    parser.add_argument("--feature-window-batch-size", type=int, default=4)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-epochs", type=int, default=8)
    parser.add_argument("--onehot-radii", default="15,50,150")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    windows = (root / args.windows_root).resolve() if not args.windows_root.is_absolute() else args.windows_root
    artifacts = (root / args.artifact_root).resolve() if not args.artifact_root.is_absolute() else args.artifact_root
    results = (root / args.result_root).resolve() if not args.result_root.is_absolute() else args.result_root
    mlm = (root / args.mlm_checkpoint).resolve() if not args.mlm_checkpoint.is_absolute() else args.mlm_checkpoint
    supervised = (root / args.supervised_checkpoint).resolve() if not args.supervised_checkpoint.is_absolute() else args.supervised_checkpoint
    stages = {x.strip() for x in args.stages.split(",") if x.strip()}
    backbones = [x.strip() for x in args.backbones.split(",") if x.strip()]
    onehot_radii = tuple(int(x) for x in args.onehot_radii.split(",") if x.strip())
    selection_specs = {
        "train": (TRAIN_CHROMS, args.train_positions, True),
        "tune": (TUNE_CHROMS, args.tune_positions, False),
        "chr22": (CLEAN_CHROMS, args.evaluation_positions, False),
        "chr1": (TEST_CHROMS, args.evaluation_positions, False),
    }
    window_verification = verify_window_manifest(windows)
    selection_meta = {}
    if "sample" in stages:
        for index, (name, (chroms, target, balanced)) in enumerate(selection_specs.items()):
            files = discover_windows(windows, chroms)
            selection_meta[name] = make_selection(
                files, target, balanced, args.seed + index, args.max_per_window,
                artifacts / "selections" / name,
            )
    else:
        for name in selection_specs:
            selection_meta[name] = json.loads(
                (artifacts / "selections" / name / "metadata.json").read_text()
            )
    if "features" in stages:
        for backbone in backbones:
            checkpoint = mlm if backbone == "mlm" else supervised if backbone == "supervised" else None
            for split in selection_specs:
                extract_backbone_features(
                    artifacts / "selections" / split, backbone, checkpoint,
                    artifacts / "features" / backbone / split,
                    window_batch_size=args.feature_window_batch_size,
                    overwrite=args.overwrite_features,
                    backbone_seed=args.seed,
                )
        for radius in onehot_radii:
            for split in selection_specs:
                onehot_features(
                    artifacts / "selections" / split, radius,
                    artifacts / "features" / f"onehot_{radius}" / split / "features.npy",
                )
    if "probe" in stages:
        run_probes(
            artifacts, results, backbones, [1e-4, 1e-3, 1e-2],
            args.seed, args.probe_epochs, onehot_radii,
        )
    results.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": "cnn_mamba_belief_probe_A_v1", "git_commit": git_commit(root),
        "seed": args.seed, "window_size": 10000, "stride": 5000,
        "owned_interval": [2500, 7500],
        "chromosomes": {"train": list(TRAIN_CHROMS), "weight_decay_selection": list(TUNE_CHROMS),
                        "clean_evaluation": list(CLEAN_CHROMS), "second_test": list(TEST_CHROMS)},
        "selection": selection_meta, "backbones": backbones,
        "window_manifest_verification": window_verification,
        "checkpoints": {"mlm": {"path": str(mlm), "sha256": sha256_file(mlm)},
                        "supervised": {"path": str(supervised), "sha256": sha256_file(supervised)}},
        "weight_decays": [1e-4, 1e-3, 1e-2], "probe_epochs": args.probe_epochs,
        "feature_window_batch_size": args.feature_window_batch_size,
        "onehot_radii": list(onehot_radii),
        "metric_definitions": {
            "joint_4class_cds_accuracy": (
                "four-state argmax evaluated at valid CDS positions; a noncoding "
                "prediction is an error"
            ),
            "conditional_frame_accuracy_3class": (
                "argmax over frame logits 0/1/2 only, evaluated at valid CDS "
                "positions"
            ),
        },
        "supervised_chr2_note": "chr2 was in validation, not training, for this checkpoint",
    }
    (results / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
