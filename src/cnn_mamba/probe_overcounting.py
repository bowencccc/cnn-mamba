"""Experiment E: residual correlation and evidence-overcounting diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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


def masked_autocorrelation(
    source, mask, frame, input_kind, max_lag, chunk_size=1_000_000,
):
    """Chunked autocorrelation using pairs that remain inside one region.

    Chunking bounds memory for Human chr1.  Pairs crossing a chunk boundary are
    omitted (at most ``max_lag/chunk_size`` of pairs at a given short lag), and
    this approximation is recorded in the run config.
    """
    total = 0.0
    count = 0
    for start in range(0, len(mask), chunk_size):
        end = min(start + chunk_size, len(mask))
        local_mask = mask[start:end]
        values = log_odds_chunk(source, start, end, frame, input_kind)
        total += float(values[local_mask].sum())
        count += int(local_mask.sum())
    mean = total / max(count, 1)
    numerator = np.zeros(max_lag + 1, dtype=np.float64)
    pairs = np.zeros(max_lag + 1, dtype=np.float64)
    for start in range(0, len(mask), chunk_size):
        end = min(start + chunk_size, len(mask))
        local_mask = mask[start:end].astype(np.float64)
        values = log_odds_chunk(source, start, end, frame, input_kind)
        centered = (values - mean) * local_mask
        fft_size = 1 << (2 * len(values) - 1).bit_length()
        fx = np.fft.rfft(centered, fft_size)
        fm = np.fft.rfft(local_mask, fft_size)
        usable = min(max_lag + 1, len(values))
        numerator[:usable] += np.fft.irfft(
            fx * fx.conj(), fft_size,
        )[:usable]
        pairs[:usable] += np.fft.irfft(
            fm * fm.conj(), fft_size,
        )[:usable]
    covariance = numerator / np.maximum(pairs, 1.0)
    return covariance / max(covariance[0], np.finfo(float).eps), pairs


def integrated_positive_time(correlation):
    positive = []
    for value in correlation[1:]:
        if value <= 0:
            break
        positive.append(value)
    return float(1.0 + 2.0 * np.sum(positive))


def e_fold_lag(correlation, stride=1):
    threshold = np.exp(-1)
    for index, value in enumerate(correlation[1:], 1):
        if value <= threshold:
            return int(index * stride)
    return int((len(correlation) - 1) * stride)


def chromosome_cds_mask(gtf, chrom, length, strand="+"):
    """Union CDS mask from a GTF/GFF without species-specific name filters."""
    mask = np.zeros(length, dtype=np.bool_)
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
                mask[max(0, start):min(length, end)] = True
    return mask


def run_autocorrelation(
    probability_path, reference_gtf, chrom, output, max_lag=10000,
    input_kind="probabilities", chunk_size=1_000_000, dataset="drosophila",
):
    source = np.load(probability_path, mmap_mode="r")
    if source.ndim != 2 or source.shape[1] != 4:
        raise ValueError(f"expected [length,4], got {source.shape}")
    coding = chromosome_cds_mask(reference_gtf, chrom, len(source), "+")
    rows, curves, frame_curves = [], {}, {}
    for region, mask in (("CDS", coding), ("noncoding", ~coding)):
        per_frame = []
        per_frame_codon_iat = []
        for frame in range(3):
            corr, pairs = masked_autocorrelation(
                source, mask, frame, input_kind, max_lag, chunk_size,
            )
            per_frame.append(corr)
            frame_curves[(region, frame)] = corr
            codon_correlation = corr[::3]
            codon_iat = integrated_positive_time(codon_correlation)
            per_frame_codon_iat.append(codon_iat)
            rows.append({
                "dataset": dataset, "region": region, "frame": frame,
                "integrated_positive_autocorrelation_time": integrated_positive_time(corr),
                "codon_stride_iat_codons": codon_iat,
                "codon_stride_iat_nt": 3.0 * codon_iat,
                "codon_stride_e_fold_nt": e_fold_lag(codon_correlation, stride=3),
                "lag1": float(corr[1]), "valid_positions": int(mask.sum()),
                "minimum_pairs_at_reported_lags": int(pairs.min()),
            })
        curves[region] = np.mean(per_frame, axis=0)
        rows.append({
            "dataset": dataset, "region": region, "frame": "mean",
            "integrated_positive_autocorrelation_time": integrated_positive_time(curves[region]),
            "codon_stride_iat_codons": float(np.mean(per_frame_codon_iat)),
            "codon_stride_iat_nt": float(3.0 * np.mean(per_frame_codon_iat)),
            "codon_stride_e_fold_nt": float(np.mean([
                e_fold_lag(corr[::3], stride=3) for corr in per_frame
            ])),
            "lag1": float(curves[region][1]), "valid_positions": int(mask.sum()),
            "minimum_pairs_at_reported_lags": int(pairs.min()),
        })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "autocorrelation_summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
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
    mean_lengths = {
        row["region"]: row["codon_stride_iat_codons"]
        for row in rows if row["frame"] == "mean"
    }
    comparison = []
    for region, length in mean_lengths.items():
        for temperature in (7.0, 9.0):
            comparison.append({
                "dataset": dataset, "region": region,
                "decorrelation_length_codons": length,
                "reference_temperature": temperature,
                "length_over_temperature": length / temperature,
            })
    with (output / "decorrelation_length_vs_temperature.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(comparison)
    fig, axis = plt.subplots(figsize=(5.5, 4.2))
    names = list(mean_lengths)
    axis.bar(names, [mean_lengths[name] for name in names], label="codon-stride IAT")
    axis.axhline(7, color="C2", linestyle="--", label="T=7")
    axis.axhline(9, color="C3", linestyle=":", label="T=9")
    axis.set(ylabel="Effective correlated codons / temperature", title=dataset)
    axis.set_yscale("log")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "decorrelation_length_vs_temperature.png", dpi=220)
    fig.savefig(output / "decorrelation_length_vs_temperature.pdf")
    plt.close(fig)
    return rows


def validation_coding_prior(windows_root, chromosomes=("3L",)):
    coding = noncoding = 0
    wanted = set(chromosomes)
    for split in ("train", "val", "test"):
        for path in (Path(windows_root) / split).glob("*.npz"):
            if path.stem.split("_", 1)[0] not in wanted:
                continue
            with np.load(path) as data:
                cds = data["reference_cds_labels"][2500:7500].astype(bool)
                valid = (~cds) | data["reference_phase_mask"][2500:7500].astype(bool)
            coding += int((cds & valid).sum())
            noncoding += int((~cds & valid).sum())
    return {
        "coding_positions": coding, "noncoding_positions": noncoding,
        "coding_fraction": coding / (coding + noncoding),
        "log_prior_noncoding_over_coding": float(np.log(noncoding / coding)),
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
        "max_lag": args.max_lag, "chunk_size": args.chunk_size,
        "chunk_boundary_pair_omission_upper_fraction": args.max_lag / args.chunk_size,
        "best_temperature_comparison": [7, 9],
    }
    if args.windows_root:
        config["validation_prior"] = validation_coding_prior(
            args.windows_root,
            tuple(x.strip() for x in args.validation_chromosomes.split(",")),
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
