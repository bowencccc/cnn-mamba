#!/usr/bin/env python3
"""Run selected stages for one window/regime; suitable for one process per GPU."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def call(command):
    print("+ " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(x) for x in command], check=True)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", choices=("w5", "w10", "w20", "w30", "w40", "w80"), required=True)
    parser.add_argument("--regime", choices=("heldout", "chrxtrain"), required=True)
    parser.add_argument("--stages", default="train,score,auprc",
                        help="comma list: train,score,auprc,export,uniann")
    parser.add_argument("--uniann-root", type=Path,
                        default=Path(os.environ.get("UNIANN_ROOT", root.parent / "UniAnn")))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads((root / "configs/droso_window_grid.json").read_text())
    cell = next(row for row in config["cells"] if row["id"] == args.cell)
    stages = [x.strip() for x in args.stages.split(",") if x.strip()]
    unknown = set(stages) - {"train", "score", "auprc", "export", "uniann"}
    if unknown:
        parser.error(f"unknown stages: {sorted(unknown)}")

    data = root / "data"
    run_dir = root / "runs" / args.regime / args.cell
    checkpoint = run_dir / "best_model.pt"
    score_dir = root / "artifacts" / "scores" / args.regime / args.cell
    raw = data / "raw"

    if "train" in stages:
        command = [sys.executable, "-m", "cnn_mamba.train", "--data-root",
                   data / "processed" / args.cell, "--mode", "baseline",
                   "--architecture", "cnn_mamba", "--cnn-kernel-size", "7",
                   "--epochs", str(config["epochs"]), "--batch-size", str(cell["micro_batch"]),
                   "--grad-accum", str(cell["grad_accum"]), "--window-size", str(cell["window_bp"]),
                   "--stride", str(cell["stride_bp"]), "--workers", str(args.workers),
                   "--output-dir", run_dir, "--reference-name", "FlyBase", "--test-name", "chrX"]
        if args.regime == "chrxtrain":
            command.append("--include-test-in-train")
        if cell.get("gradient_checkpointing"):
            command.append("--gradient-checkpointing")
        if (run_dir / "last_model.pt").is_file():
            command.append("--resume")
        call(command)
    elif not checkpoint.is_file():
        external = root / "artifacts/checkpoints" / args.regime / args.cell / "best_model.pt"
        if external.is_file():
            checkpoint = external
        else:
            parser.error(f"checkpoint not found: {checkpoint} or {external}")

    if "score" in stages:
        call([sys.executable, "-m", "cnn_mamba.score_chromosome", "--checkpoint", checkpoint,
              "--output-dir", score_dir, "--fasta", raw / "dmel_genome.fa",
              "--chrom", "NC_004354.4", "--display-chrom", "chrX",
              "--batch-size", str(cell["score_batch"])])
    if "auprc" in stages:
        call([sys.executable, "-m", "cnn_mamba.evaluate_candidates", "--score-dir", score_dir,
              "--reference-gtf", raw / "dmel_reference.gtf", "--fasta", raw / "dmel_genome.fa",
              "--chrom", "NC_004354.4", "--output", score_dir / "candidate_auprc.tsv"])
    legacy = score_dir / "chrX_plus_scores.tsv"
    if "export" in stages or "uniann" in stages:
        call([sys.executable, "-m", "cnn_mamba.export_uniann_scores", "--score-dir", score_dir,
              "--output", legacy, "--chrom", "X", "--fasta", raw / "dmel_genome.fa",
              "--accession", "NC_004354.4"])
    if "uniann" in stages:
        call([sys.executable, "-m", "cnn_mamba.uniann_evaluate", "--uniann-root", args.uniann_root,
              "--fasta", raw / "dmel_chrX.fa", "--psauron", raw / "psauron_score_droso.csv",
              "--scores", legacy, "--reference-gtf", raw / "dmel_chrX_plus_CDS_reference.gtf",
              "--eviann-gff", raw / "dmel_chrX_plus_EviAnn_CDS.gff",
              "--output-dir", root / "work" / "uniann" / args.cell])


if __name__ == "__main__":
    main()
