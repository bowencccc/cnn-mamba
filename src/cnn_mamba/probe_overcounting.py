"""Experiment E: residual correlation and evidence-overcounting diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from .intron_phase_labels import open_text
from .probe_features import sha256_file


def log_odds_chunk(source, start, end, frame, input_kind):
    values = np.asarray(source[start:end])
    if input_kind == "logits":
        return values[:, frame].astype(np.float64) - values[:, 3].astype(np.float64)
    eps = 1e-7
    return np.log(np.maximum(values[:, frame], eps)) - np.log(
        np.maximum(values[:, 3], eps)
    )


def segmentwise_autocorrelation(source, segments, frame, input_kind, max_lag):
    """Aggregate autocovariance after centering every segment separately.

    No pair is allowed to cross a CDS/noncoding segment boundary.  This avoids
    both concatenation artifacts and correlation caused only by different
    segment-level means.
    """
    numerator = np.zeros(max_lag + 1, dtype=np.float64)
    pairs = np.zeros(max_lag + 1, dtype=np.float64)
    for start, end in segments:
        if end <= start:
            continue
        values = log_odds_chunk(source, start, end, frame, input_kind)
        centered = values - values.mean()
        fft_size = 1 << (2 * len(values) - 1).bit_length()
        fx = np.fft.rfft(centered, fft_size)
        usable = min(max_lag + 1, len(values))
        numerator[:usable] += np.fft.irfft(
            fx * fx.conj(), fft_size,
        )[:usable]
        pairs[:usable] += np.arange(len(values), len(values) - usable, -1)
    covariance = numerator / np.maximum(pairs, 1.0)
    return covariance / max(covariance[0], np.finfo(float).eps), pairs


def integrated_positive_time(correlation):
    positive = []
    for value in correlation[1:]:
        if value <= 0:
            break
        positive.append(value)
    return float(1.0 + 2.0 * np.sum(positive))


def e_fold_lag(correlation, stride=1, pairs=None):
    """Return the first 1/e crossing and whether it was right-censored."""
    threshold = np.exp(-1)
    if pairs is None:
        valid_indices = range(1, len(correlation))
    else:
        pairs = np.asarray(pairs)
        if len(pairs) != len(correlation):
            raise ValueError("pair counts and correlation must have equal length")
        valid_indices = np.flatnonzero(pairs[1:] > 0) + 1
    last_valid = 0
    for index in valid_indices:
        last_valid = int(index)
        value = correlation[index]
        if value <= threshold:
            return int(index * stride), False
    return int(last_valid * stride), True


def chromosome_segments(gtf, chrom, length, strand="+"):
    """Return unique CDS features and contiguous complement segments."""
    cds = set()
    with open_text(gtf) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip().split("\t")
            if (
                len(fields) < 8 or fields[0] != chrom or fields[2] != "CDS"
                or fields[6] != strand
            ):
                continue
            start, end = int(fields[3]) - 1, int(fields[4])
            if start < length:
                cds.add((max(0, start), min(length, end)))
    cds = sorted(cds)
    merged = []
    for start, end in cds:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    noncoding = []
    previous = 0
    for start, end in merged:
        if start > previous:
            noncoding.append((previous, start))
        previous = max(previous, end)
    if previous < length:
        noncoding.append((previous, length))
    return cds, noncoding


def length_summary(dataset, region, segments):
    lengths = np.asarray([end - start for start, end in segments], dtype=np.int64)
    if not len(lengths):
        raise RuntimeError(f"no {region} segments for {dataset}")
    return {
        "dataset": dataset, "region": region, "segments": int(len(lengths)),
        "total_segment_nt": int(lengths.sum()), "minimum_nt": int(lengths.min()),
        "p25_nt": float(np.percentile(lengths, 25)),
        "median_nt": float(np.median(lengths)),
        "mean_nt": float(lengths.mean()),
        "p75_nt": float(np.percentile(lengths, 75)),
        "p90_nt": float(np.percentile(lengths, 90)),
        "p95_nt": float(np.percentile(lengths, 95)),
        "maximum_nt": int(lengths.max()),
    }


def run_autocorrelation(
    probability_path, reference_gtf, chrom, output, max_lag=10000,
    input_kind="probabilities", chunk_size=1_000_000, dataset="drosophila",
):
    source = np.load(probability_path, mmap_mode="r")
    if source.ndim != 2 or source.shape[1] != 4:
        raise ValueError(f"expected [length,4], got {source.shape}")
    cds_segments, noncoding_segments = chromosome_segments(
        reference_gtf, chrom, len(source), "+",
    )
    region_segments = {"CDS": cds_segments, "noncoding": noncoding_segments}
    length_rows = [
        length_summary(dataset, region, segments)
        for region, segments in region_segments.items()
    ]
    rows, curves, frame_curves = [], {}, {}
    for region, segments in region_segments.items():
        per_frame = []
        per_frame_codon_iat = []
        for frame in range(3):
            corr, pairs = segmentwise_autocorrelation(
                source, segments, frame, input_kind, max_lag,
            )
            per_frame.append(corr)
            frame_curves[(region, frame)] = corr
            codon_correlation = corr[::3]
            codon_iat = integrated_positive_time(codon_correlation)
            per_frame_codon_iat.append(codon_iat)
            codon_e_fold, codon_e_fold_censored = e_fold_lag(
                codon_correlation, stride=3, pairs=pairs[::3],
            )
            raw_e_fold, raw_e_fold_censored = e_fold_lag(corr, pairs=pairs)
            rows.append({
                "dataset": dataset, "region": region, "frame": frame,
                "integrated_positive_autocorrelation_time": integrated_positive_time(corr),
                "codon_stride_iat_codons": codon_iat,
                "codon_stride_iat_nt": 3.0 * codon_iat,
                "codon_stride_e_fold_nt": codon_e_fold,
                "codon_stride_e_fold_right_censored": int(codon_e_fold_censored),
                "raw_e_fold_nt": raw_e_fold,
                "raw_e_fold_right_censored": int(raw_e_fold_censored),
                "lag1": float(corr[1]),
                "segments": len(segments),
                "segment_nt_with_multiplicity": int(sum(e - s for s, e in segments)),
                "pairs_at_max_lag": int(pairs[-1]),
            })
        curves[region] = np.mean(per_frame, axis=0)
        codon_e_folds = [
            e_fold_lag(corr[::3], stride=3, pairs=pairs[::3])
            for corr in per_frame
        ]
        raw_e_folds = [e_fold_lag(corr, pairs=pairs) for corr in per_frame]
        rows.append({
            "dataset": dataset, "region": region, "frame": "mean",
            "integrated_positive_autocorrelation_time": integrated_positive_time(curves[region]),
            "codon_stride_iat_codons": float(np.mean(per_frame_codon_iat)),
            "codon_stride_iat_nt": float(3.0 * np.mean(per_frame_codon_iat)),
            "codon_stride_e_fold_nt": float(np.mean([x[0] for x in codon_e_folds])),
            "codon_stride_e_fold_right_censored": int(any(x[1] for x in codon_e_folds)),
            "raw_e_fold_nt": float(np.mean([x[0] for x in raw_e_folds])),
            "raw_e_fold_right_censored": int(any(x[1] for x in raw_e_folds)),
            "lag1": float(curves[region][1]),
            "segments": len(segments),
            "segment_nt_with_multiplicity": int(sum(e - s for s, e in segments)),
            "pairs_at_max_lag": int(pairs[-1]),
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "autocorrelation_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    with (output / "region_length_distribution.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(length_rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(length_rows)
    fig, axis = plt.subplots(figsize=(6, 4.5))
    lag = np.arange(max_lag + 1)
    colors = {"CDS": "C0", "noncoding": "C1"}
    for region, curve in curves.items():
        for frame in range(3):
            axis.plot(
                lag, frame_curves[(region, frame)], color=colors[region],
                alpha=0.22, linewidth=0.8,
            )
        axis.plot(lag, curve, color=colors[region], linewidth=2, label=f"{region} mean")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Lag (nt)", ylabel="Mean log-odds autocorrelation",
             xlim=(0, max_lag))
    axis.legend(frameon=False); axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "autocorrelation.png", dpi=220)
    fig.savefig(output / "autocorrelation.pdf")
    plt.close(fig)
    return rows


def validation_coding_prior(windows_root, chromosomes=("3L",)):
    phase_counts = np.zeros(3, dtype=np.int64)
    noncoding = 0
    wanted = set(chromosomes)
    for split in ("train", "val", "test"):
        for path in (Path(windows_root) / split).glob("*.npz"):
            if path.stem.split("_", 1)[0] not in wanted:
                continue
            with np.load(path) as data:
                cds = data["reference_cds_labels"][2500:7500].astype(bool)
                phase_valid = data["reference_phase_mask"][2500:7500].astype(bool)
                phase = data["reference_phase_labels"][2500:7500]
            coding = cds & phase_valid
            phase_counts += np.bincount(phase[coding], minlength=3)
            noncoding += int((~cds).sum())
    coding = int(phase_counts.sum())
    total = coding + noncoding
    return {
        "coding_positions": coding, "noncoding_positions": noncoding,
        "phase_positions": phase_counts.tolist(),
        "phase_fraction_within_coding": (phase_counts / coding).tolist(),
        "empirical_class_prior": [
            noncoding / total, *(phase_counts / total).tolist(),
        ],
        "coding_fraction": coding / total,
        "log_prior_noncoding_over_coding": float(np.log(noncoding / coding)),
    }


def group_balanced_prior_correction(
    windows_root, training_chromosomes, target_chromosomes,
    noncoding_weight=0.1, temperature=9.0,
):
    """Per-frame logit correction after temperature scaling.

    ``group_balanced_joint_frame_loss`` gives total coding risk mass 1 and
    noncoding risk mass ``noncoding_weight``.  Within coding, the phase prior
    remains empirical.  The converter applies bias after dividing logits by T,
    so the target-vs-training prior log-ratio is divided by T as well.
    """
    train = validation_coding_prior(windows_root, training_chromosomes)
    target = validation_coding_prior(windows_root, target_chromosomes)
    q = np.asarray(train["phase_fraction_within_coding"], dtype=np.float64)
    implicit = np.concatenate((
        [noncoding_weight / (1.0 + noncoding_weight)],
        q / (1.0 + noncoding_weight),
    ))
    target_prior = np.asarray(target["empirical_class_prior"], dtype=np.float64)
    correction = np.log(target_prior / implicit)
    per_frame_bias = (correction[1:] - correction[0]) / temperature
    return {
        "noncoding_weight": noncoding_weight,
        "temperature": temperature,
        "training_chromosomes": list(training_chromosomes),
        "target_chromosomes": list(target_chromosomes),
        "training_observed": train,
        "implicit_training_prior": implicit.tolist(),
        "target_observed": target,
        "unscaled_class_log_correction": correction.tolist(),
        "per_frame_bias_after_temperature": per_frame_bias.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--reference-gtf", type=Path, required=True)
    parser.add_argument("--chrom", default="NC_004354.4")
    parser.add_argument("--windows-root", type=Path)
    parser.add_argument("--validation-chromosomes", default="3L")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-lag", type=int, default=10000)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--input-kind", choices=("probabilities", "logits"), default="probabilities")
    parser.add_argument("--dataset", default="drosophila")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    rows = run_autocorrelation(
        args.probabilities, args.reference_gtf, args.chrom, args.output,
        args.max_lag, args.input_kind, args.chunk_size, args.dataset,
    )
    config = {
        "probabilities": str(args.probabilities.resolve()),
        "probabilities_sha256": sha256_file(args.probabilities),
        "reference_gtf": str(args.reference_gtf.resolve()),
        "input_kind": args.input_kind, "dataset": args.dataset,
        "max_lag": args.max_lag,
        "autocorrelation": (
            "within each unique GTF CDS feature or contiguous noncoding "
            "complement segment; per-segment mean removed; no cross-segment pairs"
        ),
        "e_fold_right_censoring": (
            "reported explicitly when correlation never crossed 1/e before "
            "the last lag having at least one within-segment pair"
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip(),
        "seed": 42,
        "sampled_positions": int(np.load(args.probabilities, mmap_mode="r").shape[0]),
        "representation": "four-state joint_frame_head output",
    }
    if args.checkpoint:
        config["checkpoint"] = str(args.checkpoint.resolve())
        config["checkpoint_sha256"] = sha256_file(args.checkpoint)
    if args.windows_root:
        config["validation_prior"] = validation_coding_prior(
            args.windows_root,
            tuple(x.strip() for x in args.validation_chromosomes.split(",")),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
