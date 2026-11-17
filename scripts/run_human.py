#!/usr/bin/env python3
"""Train or reuse Human CNN--Mamba-k7, then score/evaluate chr1+."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def call(command):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], check=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=("chr1held", "chr1train"), required=True)
    parser.add_argument("--stages", default="train,score,auprc")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--experiment-tag", default="")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--training-mode", choices=("baseline", "ignore"), default="baseline"
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--scheduler-epochs", type=int)
    parser.add_argument("--frame-dilation", type=int, default=0)
    parser.add_argument("--joint-frame-aux-weight", type=float, default=0.0)
    parser.add_argument("--joint-noncoding-weight", type=float, default=0.1)
    parser.add_argument("--record-gradient-norms", action="store_true")
    parser.add_argument("--use-start-stop-weights", action="store_true")
    parser.add_argument("--start-stop-weight-root", type=Path)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--score-batch", type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0001)
    args = parser.parse_args()
    stages = [value.strip() for value in args.stages.split(",") if value.strip()]
    unknown = set(stages) - {"train", "score", "auprc"}
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")

    config = json.loads((root / "configs/human_cnn_mamba_k7.json").read_text())
    if args.window_size is not None:
        config["window_bp"] = args.window_size
    if args.stride is not None:
        config["stride_bp"] = args.stride
    if args.micro_batch is not None:
        config["micro_batch"] = args.micro_batch
    if args.grad_accum is not None:
        config["grad_accum"] = args.grad_accum
    if args.score_batch is not None:
        config["score_batch"] = args.score_batch
    raw = root / "data/raw"
    processed = args.data_root or root / "data/processed/human_refseq_w10_chr1held"
    run_root = root / "runs/human"
    score_root = root / "artifacts/scores/human"
    if args.experiment_tag:
        run_root /= args.experiment_tag
        score_root /= args.experiment_tag
    run_dir = run_root / args.regime
    checkpoint = run_dir / "best_model.pt"
    external = root / "artifacts/checkpoints/human/chr1train_cnn_mamba_k7/best_model.pt"
    score_dir = score_root / args.regime / "chr1_plus"

    if "train" in stages:
        manifest_path = processed / "split_manifest.json"
        reference_name = "RefSeq"
        if manifest_path.is_file():
            reference_name = json.loads(manifest_path.read_text()).get(
                "reference_name", "CHESS"
            )
        command = [sys.executable, "-m", "cnn_mamba.train",
                   "--data-root", processed, "--mode", args.training_mode,
                   "--output-dir", run_dir,
                   "--epochs", str(args.epochs or config["epochs"]),
                   "--batch-size", str(config["micro_batch"]),
                   "--grad-accum", str(config["grad_accum"]),
                   "--workers", str(args.workers), "--architecture", "cnn_mamba",
                   "--cnn-kernel-size", str(config["cnn_kernel_size"]),
                   "--window-size", str(config["window_bp"]),
                   "--stride", str(config["stride_bp"]),
                   "--reference-name", reference_name, "--test-name", "chr1",
                   "--frame-dilation", str(args.frame_dilation),
                   "--joint-frame-aux-weight", str(args.joint_frame_aux_weight),
                   "--joint-noncoding-weight", str(args.joint_noncoding_weight),
                   "--early-stopping-patience", str(args.early_stopping_patience),
                   "--early-stopping-min-delta", str(args.early_stopping_min_delta)]
        if args.scheduler_epochs is not None:
            command.extend(("--scheduler-epochs", str(args.scheduler_epochs)))
        if args.record_gradient_norms:
            command.append("--record-gradient-norms")
        if args.use_start_stop_weights:
            command.append("--use-start-stop-weights")
        if args.start_stop_weight_root is not None:
            command.extend(("--start-stop-weight-root", args.start_stop_weight_root))
        if args.gradient_checkpointing:
            command.append("--gradient-checkpointing")
        if config.get("amp"):
            command.extend(("--amp", "--amp-dtype", config.get("amp_dtype", "bf16")))
        if config.get("dropout") is not None:
            command.extend(("--dropout", str(config["dropout"])))
        if args.regime == "chr1train":
            command.append("--include-test-in-train")
        if (run_dir / "last_model.pt").is_file():
            command.append("--resume")
        call(command)
    elif not checkpoint.is_file():
        if (
            args.regime == "chr1train"
            and not args.experiment_tag
            and args.joint_frame_aux_weight == 0
            and external.is_file()
        ):
            checkpoint = external
        else:
            parser.error(f"checkpoint not found: {checkpoint}")

    if "score" in stages:
        call([sys.executable, "-m", "cnn_mamba.score_chromosome",
              "--checkpoint", checkpoint, "--fasta", raw / "human_grch38.fa",
              "--chrom", "NC_000001.11", "--display-chrom", "chr1",
              "--output-dir", score_dir, "--batch-size", str(config["score_batch"])])
    if "auprc" in stages:
        call([sys.executable, "-m", "cnn_mamba.evaluate_candidates",
              "--species", "human", "--score-dir", score_dir,
              "--reference-gtf", raw / "human_refseq_grch38_p14.gff.gz",
              "--fasta", raw / "human_grch38.fa", "--chrom", "NC_000001.11",
              "--reference-chrom", "chr1", "--output", score_dir / "candidate_auprc.tsv"])


if __name__ == "__main__":
    main()
