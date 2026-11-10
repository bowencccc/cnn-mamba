"""Experiment B: frozen-backbone probes for CDS offset and coding-intron phase."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .intron_phase_labels import (
    PHASE_SCHEMA, CodingIntronIndex, coding_introns_from_gtf, parse_window_name,
    window_intron_tracks,
)
from .probe_belief import (
    CLEAN_CHROMS, TEST_CHROMS, TRAIN_CHROMS, TUNE_CHROMS,
    contiguous_feature_view, discover_windows, extract_backbone_features, git_commit,
    onehot_features, verify_window_manifest,
)
from .probe_features import FEATURE_NAMES, sha256_file
from .probe_metrics import fit_linear_probe, predict_probabilities, seven_state_metrics


DISTANCE_BINS = (
    ("0-50", 0, 50),
    ("50-200", 50, 200),
    ("200-1000", 200, 1000),
    ("1000-5000", 1000, 5000),
    (">=5000", 5000, None),
)


class SidecarStore:
    """Materialize only the window-aligned intron tracks touched by sampling."""

    def __init__(self, gtf: Path, output_root: Path, window_size=10000):
        self.gtf = Path(gtf)
        self.output_root = Path(output_root)
        self.window_size = int(window_size)
        self.index = CodingIntronIndex(coding_introns_from_gtf(self.gtf))
        self.written = 0
        self.reused = 0

    def path_for(self, window_path: Path):
        return self.output_root / window_path.parent.name / window_path.name

    def ensure(self, window_path: Path):
        destination = self.path_for(window_path)
        if destination.is_file():
            with np.load(destination) as data:
                required = {
                    "intron_phase_schema",
                    "intron_phase_labels", "intron_phase_mask",
                    "distance_to_donor", "distance_to_acceptor",
                    "intron_fits_window", "intron_start", "intron_end",
                }
                if (
                    required <= set(data.files)
                    and str(data["intron_phase_schema"]) == PHASE_SCHEMA
                ):
                    self.reused += 1
                    return destination
        chrom, strand, start = parse_window_name(window_path)
        introns = self.index.overlapping(
            chrom, strand, start, start + self.window_size,
        )
        tracks = window_intron_tracks(introns, start, self.window_size, strand)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.tmp")
        # Passing a file handle prevents NumPy from appending another suffix.
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **tracks)
        temporary.replace(destination)
        self.written += 1
        return destination


def validate_sidecar_alignment(sidecar_root, windows_root, max_windows_per_strand=500):
    """Verify intron phase equals the saved phase at the CDS base after acceptor."""
    report = {}
    for strand_name in ("plus", "minus"):
        matches = total = checked = 0
        for sidecar in Path(sidecar_root).glob(f"*/*_{strand_name}_*.npz"):
            with np.load(sidecar) as intron:
                if (
                    "intron_phase_schema" not in intron
                    or str(intron["intron_phase_schema"]) != PHASE_SCHEMA
                ):
                    continue
                phase = intron["intron_phase_labels"]
                acceptor = intron["distance_to_acceptor"]
                boundary = intron["intron_start"]
                window_path = Path(windows_root) / sidecar.parent.name / sidecar.name
                with np.load(window_path) as window:
                    indices = np.flatnonzero(
                        (acceptor[:-1] == 0) & (phase[:-1] >= 0)
                        & (boundary[:-1] >= 0)
                        & window["reference_phase_mask"][1:].astype(bool)
                    )
                    matches += int((
                        phase[indices] == window["reference_phase_labels"][indices + 1]
                    ).sum())
                    total += len(indices)
            checked += 1
            if checked >= max_windows_per_strand:
                break
        if total == 0 or matches != total:
            raise RuntimeError(
                f"sidecar/reference phase alignment failed for {strand_name}: "
                f"matches={matches}, total={total}, windows={checked}"
            )
        report[strand_name] = {
            "windows_checked": checked, "acceptors_checked": total,
            "matches": matches, "fraction": matches / total,
        }
    return report


def labels_seven_state(window_path, sidecar_path, left=2500, right=7500):
    """Return unambiguous 7-state CHESS labels for one owned interval.

    ``reference_phase_mask`` is the only authority for whether a CDS phase is
    valid.  Bases that are CDS in one isoform and coding intron in another, or
    whose intron phase conflicts across transcripts, are excluded rather than
    silently becoming noncoding.
    """
    with np.load(window_path) as data:
        cds = data["reference_cds_labels"][left:right].astype(bool)
        cds_phase = data["reference_phase_labels"][left:right].astype(np.int8)
        cds_valid = data["reference_phase_mask"][left:right].astype(bool)
    with np.load(sidecar_path) as data:
        intron_phase = data["intron_phase_labels"][left:right].astype(np.int8)
        intron_valid = data["intron_phase_mask"][left:right].astype(bool)
        donor = data["distance_to_donor"][left:right].astype(np.int32)
        acceptor = data["distance_to_acceptor"][left:right].astype(np.int32)
        fits = data["intron_fits_window"][left:right].astype(bool)
        intron_start = data["intron_start"][left:right].astype(np.int32)
        intron_end = data["intron_end"][left:right].astype(np.int32)

    intron_present = intron_phase != -1
    overlap = cds & intron_present
    labels = np.full(right - left, -1, dtype=np.int8)
    background = (~cds) & (~intron_present)
    valid_cds = cds & cds_valid & (~overlap)
    valid_intron = (~cds) & intron_valid
    labels[background] = 0
    labels[valid_cds] = cds_phase[valid_cds] + 1
    labels[valid_intron] = intron_phase[valid_intron] + 4
    valid = labels >= 0
    return labels, valid, {
        "distance_to_donor": donor,
        "distance_to_acceptor": acceptor,
        "intron_fits_window": fits,
        "intron_start": intron_start,
        "intron_end": intron_end,
    }


def make_selection(
    files, target, balanced, seed, max_per_window, output, sidecars: SidecarStore,
):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    file_indices, offsets, labels_out = [], [], []
    extras = {name: [] for name in (
        "distance_to_donor", "distance_to_acceptor", "intron_fits_window",
        "intron_start", "intron_end",
    )}
    quotas = None
    if balanced:
        quotas = np.full(7, target // 7, dtype=np.int64)
        quotas[:target % 7] += 1
    natural_cap = min(max_per_window, max(1, target // 1000))
    counts = np.zeros(7, dtype=np.int64)

    def append(index, chosen, selected_labels, metadata):
        take = len(chosen)
        file_indices.append(np.full(take, index, dtype=np.int32))
        offsets.append((chosen + 2500).astype(np.int16))
        labels_out.append(selected_labels.astype(np.int8, copy=False))
        for name in extras:
            extras[name].append(metadata[name][chosen])

    for file_index in order:
        sidecar = sidecars.ensure(files[file_index])
        labels, valid, metadata = labels_seven_state(files[file_index], sidecar)
        if balanced:
            for cls in range(7):
                need = int(quotas[cls] - counts[cls])
                if need <= 0:
                    continue
                candidates = np.flatnonzero(valid & (labels == cls))
                take = min(need, max_per_window, len(candidates))
                if take:
                    chosen = rng.choice(candidates, take, replace=False)
                    append(
                        file_index, chosen, np.full(take, cls, dtype=np.int8), metadata,
                    )
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
                append(file_index, chosen, selected_labels, metadata)
                counts += np.bincount(selected_labels, minlength=7)
    if balanced and np.any(counts < quotas):
        raise RuntimeError(f"could not meet 7-state quotas: got={counts}, target={quotas}")
    if not file_indices:
        raise RuntimeError("selection produced no valid positions")
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "file_index": np.concatenate(file_indices),
        "offset": np.concatenate(offsets),
        "label": np.concatenate(labels_out),
        **{name: np.concatenate(values) for name, values in extras.items()},
    }
    np.savez_compressed(output / "positions.npz", **arrays)
    (output / "files.json").write_text(
        json.dumps([str(path.resolve()) for path in files], indent=2) + "\n"
    )
    metadata = {
        "positions": int(counts.sum()), "class_counts": counts.tolist(),
        "balanced": balanced, "seed": seed, "max_per_window": max_per_window,
        "windows_available": len(files),
        "windows_sampled": len(set(arrays["file_index"].tolist())),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_selection(path):
    arrays = np.load(path / "positions.npz")
    files = [Path(item) for item in json.loads((path / "files.json").read_text())]
    return files, arrays


def select_and_evaluate(
    train_x, train_y, evaluation_features, evaluation_labels,
    input_dim, weight_decays, seed, epochs,
):
    candidates = []
    for weight_decay in weight_decays:
        probe = fit_linear_probe(
            train_x, train_y, input_dim, 7, weight_decay, seed=seed, epochs=epochs,
        )
        probabilities = predict_probabilities(probe, evaluation_features["tune"])
        metric = seven_state_metrics(probabilities, evaluation_labels["tune"])
        candidates.append((metric["balanced_accuracy_7class"], weight_decay, probe))
    _, weight_decay, probe = max(candidates, key=lambda item: item[0])
    probabilities = {
        split: predict_probabilities(probe, features)
        for split, features in evaluation_features.items()
    }
    metrics = {
        split: seven_state_metrics(value, evaluation_labels[split])
        for split, value in probabilities.items()
    }
    return weight_decay, metrics, probabilities


def unique_sampled_introns(files, arrays, mask):
    rows = np.flatnonzero(mask)
    if not len(rows):
        return 0
    chrom_names = sorted({parse_window_name(path)[0] for path in files})
    chrom_to_code = {name: index for index, name in enumerate(chrom_names)}
    per_file_chrom = np.asarray(
        [chrom_to_code[parse_window_name(path)[0]] for path in files], dtype=np.int16,
    )
    per_file_strand = np.asarray(
        [parse_window_name(path)[1] == "-" for path in files], dtype=np.int8,
    )
    file_index = arrays["file_index"][rows]
    keys = np.rec.fromarrays([
        per_file_chrom[file_index], per_file_strand[file_index],
        arrays["intron_start"][rows], arrays["intron_end"][rows],
    ], names="chrom,strand,start,end")
    return int(len(np.unique(keys)))


def prepare_distance_specs(selection_dir, labels):
    files, archive = load_selection(selection_dir)
    arrays = {name: archive[name] for name in archive.files}
    archive.close()
    intron = labels >= 4
    fitting = arrays["intron_fits_window"].astype(bool)
    unambiguous = (arrays["intron_start"] >= 0) & (arrays["intron_end"] >= 0)
    specs = []
    for boundary, key in (
        ("donor", "distance_to_donor"), ("acceptor", "distance_to_acceptor")
    ):
        distance = arrays[key]
        for label, lower, upper in DISTANCE_BINS:
            selected = intron & fitting & unambiguous & (distance >= lower)
            if upper is not None:
                selected &= distance < upper
            positions = int(selected.sum())
            specs.append({
                "boundary": boundary,
                "distance_bin": label, "positions": positions,
                "sampled_unique_introns": unique_sampled_introns(files, arrays, selected),
                "mask": selected,
            })
    return specs


def distance_rows(probabilities, labels, specs, backbone, layer, evaluation):
    predicted = probabilities.argmax(1)
    rows = []
    for spec in specs:
        selected = spec["mask"]
        positions = spec["positions"]
        rows.append({
            "backbone": backbone, "layer": layer,
            "evaluation": evaluation, "boundary": spec["boundary"],
            "distance_bin": spec["distance_bin"], "positions": positions,
            "sampled_unique_introns": spec["sampled_unique_introns"],
            "intron_phase_accuracy": (
                float((predicted[selected] == labels[selected]).mean())
                if positions else float("nan")
            ),
        })
    return rows


def run_probes(
    artifact_root, result_root, backbones, weight_decays, seed, epochs,
    onehot_radii=(15, 50, 150),
):
    result_root.mkdir(parents=True, exist_ok=True)
    layers = list(FEATURE_NAMES) + ["concat_all"]
    labels = {}
    for split in ("train", "tune", "chr22", "chr1"):
        _files, archive = load_selection(artifact_root / "selections" / split)
        labels[split] = archive["label"].copy()
        archive.close()
    distance_specs = {
        split: prepare_distance_specs(
            artifact_root / "selections" / split, labels[split],
        ) for split in ("tune", "chr22", "chr1")
    }
    metric_rows, binned_rows = [], []
    for backbone in backbones:
        arrays = {
            split: np.load(
                artifact_root / "features" / backbone / split / "features.npy",
                mmap_mode="r",
            ) for split in labels
        }
        for layer in layers:
            train_x = contiguous_feature_view(arrays["train"], layer)
            evaluation_x = {
                split: contiguous_feature_view(arrays[split], layer)
                for split in ("tune", "chr22", "chr1")
            }
            weight_decay, evaluated, probabilities = select_and_evaluate(
                train_x, labels["train"], evaluation_x, labels,
                train_x.shape[1], weight_decays, seed, epochs,
            )
            for split, metrics in evaluated.items():
                metric_rows.append({
                    "backbone": backbone, "layer": layer, "evaluation": split,
                    "weight_decay": weight_decay, **metrics,
                })
                binned_rows.extend(distance_rows(
                    probabilities[split], labels[split], distance_specs[split],
                    backbone, layer, split,
                ))
            if backbone == "mlm":
                shuffled = np.random.default_rng(seed).permutation(labels["train"])
                shuffled_wd, shuffled_metrics, shuffled_probabilities = select_and_evaluate(
                    train_x, shuffled, evaluation_x, labels,
                    train_x.shape[1], weight_decays, seed, epochs,
                )
                for split, metrics in shuffled_metrics.items():
                    metric_rows.append({
                        "backbone": "shuffled", "layer": layer,
                        "evaluation": split, "weight_decay": shuffled_wd, **metrics,
                    })
                    binned_rows.extend(distance_rows(
                        shuffled_probabilities[split], labels[split], distance_specs[split],
                        "shuffled", layer, split,
                    ))
            print(
                f"[probe B] backbone={backbone} layer={layer} "
                f"wd={weight_decay:g}", flush=True,
            )
            del train_x, evaluation_x
    for radius in onehot_radii:
        arrays = {
            split: np.load(
                artifact_root / "features" / f"onehot_{radius}" / split / "features.npy",
                mmap_mode="r",
            ) for split in labels
        }
        weight_decay, evaluated, probabilities = select_and_evaluate(
            arrays["train"], labels["train"],
            {split: arrays[split] for split in ("tune", "chr22", "chr1")},
            labels, arrays["train"].shape[1], weight_decays, seed, epochs,
        )
        for split, metrics in evaluated.items():
            layer = f"onehot_{radius}"
            metric_rows.append({
                "backbone": "onehot", "layer": layer, "evaluation": split,
                "weight_decay": weight_decay, **metrics,
            })
            binned_rows.extend(distance_rows(
                probabilities[split], labels[split], distance_specs[split],
                "onehot", layer, split,
            ))
        print(f"[probe B] onehot radius={radius} wd={weight_decay:g}", flush=True)
    for filename, rows in (
        ("metrics.tsv", metric_rows), ("distance_binned_metrics.tsv", binned_rows)
    ):
        with (result_root / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    plot_metrics(metric_rows, binned_rows, result_root)
    return metric_rows, binned_rows


def plot_metrics(metrics, binned, output):
    layer_order = list(FEATURE_NAMES) + ["concat_all"]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for backbone in ("mlm", "random", "supervised", "shuffled"):
        selected = {
            row["layer"]: row for row in metrics
            if row["evaluation"] == "chr22" and row["backbone"] == backbone
        }
        if selected:
            axis.plot(
                range(len(layer_order)),
                [selected[layer]["intron_phase_accuracy"] for layer in layer_order],
                marker="o", linewidth=1.5, label=backbone,
            )
    majority = next(
        row["intron_majority_baseline"] for row in metrics
        if row["evaluation"] == "chr22" and row["backbone"] == "mlm"
    )
    axis.axhline(majority, color="black", linestyle=":", label="majority")
    for row in metrics:
        if row["evaluation"] == "chr22" and row["backbone"] == "onehot":
            axis.axhline(
                row["intron_phase_accuracy"], linestyle="--", linewidth=1,
                label=row["layer"],
            )
    axis.set(
        xlabel="Frozen representation", ylabel="Coding-intron phase accuracy",
        ylim=(0, 1.02),
    )
    axis.set_xticks(range(len(layer_order)), layer_order, rotation=50, ha="right")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "intron_phase_accuracy_vs_layer.png", dpi=220)
    fig.savefig(output / "intron_phase_accuracy_vs_layer.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    bin_names = [item[0] for item in DISTANCE_BINS]
    for axis, boundary in zip(axes, ("donor", "acceptor")):
        for backbone in ("mlm", "random", "supervised"):
            selected = {
                row["distance_bin"]: row for row in binned
                if row["evaluation"] == "chr22"
                and row["boundary"] == boundary
                and row["layer"] == "concat_all"
                and row["backbone"] == backbone
            }
            axis.plot(
                range(len(bin_names)),
                [selected[name]["intron_phase_accuracy"] for name in bin_names],
                marker="o", label=backbone,
            )
        axis.axhline(majority, color="black", linestyle=":", label="majority")
        axis.set_title(f"Distance from {boundary}")
        axis.set_xticks(range(len(bin_names)), bin_names, rotation=35, ha="right")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Coding-intron phase accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output / "intron_phase_accuracy_by_distance.png", dpi=220)
    fig.savefig(output / "intron_phase_accuracy_by_distance.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--windows-root", type=Path, default=Path("data/processed/human_joint_frame_w10_chr1held"))
    parser.add_argument("--gtf", type=Path, default=Path("data/raw/human_chess3.1.3.gtf"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/belief_probe_v1/B_intron_phase"))
    parser.add_argument("--result-root", type=Path, default=Path("results/belief_probe_v1/B_intron_phase"))
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
    resolve = lambda path: path if path.is_absolute() else (root / path).resolve()
    windows, gtf = resolve(args.windows_root), resolve(args.gtf)
    artifacts, results = resolve(args.artifact_root), resolve(args.result_root)
    mlm, supervised = resolve(args.mlm_checkpoint), resolve(args.supervised_checkpoint)
    stages = {item.strip() for item in args.stages.split(",") if item.strip()}
    backbones = [item.strip() for item in args.backbones.split(",") if item.strip()]
    onehot_radii = tuple(int(item) for item in args.onehot_radii.split(",") if item)
    specs = {
        "train": (TRAIN_CHROMS, args.train_positions, True),
        "tune": (TUNE_CHROMS, args.tune_positions, False),
        "chr22": (CLEAN_CHROMS, args.evaluation_positions, False),
        "chr1": (TEST_CHROMS, args.evaluation_positions, False),
    }
    window_verification = verify_window_manifest(windows)
    selection_meta = {}
    if "sample" in stages:
        sidecars = SidecarStore(gtf, artifacts / "sidecars")
        for index, (name, (chroms, target, balanced)) in enumerate(specs.items()):
            selection_meta[name] = make_selection(
                discover_windows(windows, chroms), target, balanced,
                args.seed + index, args.max_per_window,
                artifacts / "selections" / name, sidecars,
            )
        sidecar_meta = {
            "source_gtf": str(gtf), "source_gtf_sha256": sha256_file(gtf),
            "written": sidecars.written, "reused": sidecars.reused,
            "phase_convention": (
                "downstream CDS first transcript-oriented state from the existing "
                "build_phase_track: (3 - downstream GTF phase) % 3 on both strands"
            ),
        }
        (artifacts / "sidecars" / "manifest.json").write_text(
            json.dumps(sidecar_meta, indent=2) + "\n"
        )
    else:
        for name in specs:
            selection_meta[name] = json.loads(
                (artifacts / "selections" / name / "metadata.json").read_text()
            )
        sidecar_meta = json.loads((artifacts / "sidecars" / "manifest.json").read_text())
    sidecar_meta["reference_phase_alignment"] = validate_sidecar_alignment(
        artifacts / "sidecars", windows,
    )
    (artifacts / "sidecars" / "manifest.json").write_text(
        json.dumps(sidecar_meta, indent=2) + "\n"
    )
    if "features" in stages:
        for backbone in backbones:
            checkpoint = mlm if backbone == "mlm" else supervised if backbone == "supervised" else None
            for split in specs:
                extract_backbone_features(
                    artifacts / "selections" / split, backbone, checkpoint,
                    artifacts / "features" / backbone / split,
                    window_batch_size=args.feature_window_batch_size,
                    overwrite=args.overwrite_features,
                    backbone_seed=args.seed,
                )
        for radius in onehot_radii:
            for split in specs:
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
        "schema": "cnn_mamba_belief_probe_B_v1", "git_commit": git_commit(root),
        "seed": args.seed, "window_size": 10000, "stride": 5000,
        "owned_interval": [2500, 7500],
        "chromosomes": {
            "train": list(TRAIN_CHROMS),
            "weight_decay_selection": list(TUNE_CHROMS),
            "clean_evaluation": list(CLEAN_CHROMS),
            "second_test": list(TEST_CHROMS),
        },
        "selection": selection_meta, "sidecars": sidecar_meta,
        "window_manifest_verification": window_verification,
        "backbones": backbones,
        "checkpoints": {
            "mlm": {"path": str(mlm), "sha256": sha256_file(mlm)},
            "supervised": {"path": str(supervised), "sha256": sha256_file(supervised)},
        },
        "weight_decays": [1e-4, 1e-3, 1e-2],
        "probe_epochs": args.probe_epochs,
        "feature_window_batch_size": args.feature_window_batch_size,
        "onehot_radii": list(onehot_radii),
        "label_validity": "CDS states require reference_phase_mask; conflicts and CDS/intron overlaps dropped",
        "supervised_chr2_note": "actual checkpoint training used 195072 windows and chr2 was validation, not training",
    }
    (results / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
