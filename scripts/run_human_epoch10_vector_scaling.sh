#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/bowencccc/cnn_mamba_k7_portable
PYTHON=/home/bowencccc/miniconda3/envs/mamba3_env/bin/python
CHECKPOINT="$PROJECT/runs/human/human_joint_frame_beta010_es5_v1/chr1train/epoch_checkpoints/epoch_010.pt"
DATA_ROOT="$PROJECT/data/processed/human_joint_frame_w10_chr1held"
CALIBRATION_DIR="$PROJECT/results/human_joint_frame_beta010_es5_v1/chr1train/epoch10_vector_scaling"
OUTPUT_DIR="$PROJECT/artifacts/joint_frame_predictions/human_joint_frame_beta010_es5_v1/chr1train/epoch10_vector_scaled/chr1_plus"
FASTA=/home/bowencccc/ssm_splice_site/GCF_000001405.40_GRCh38.p14_genomic.fna
LOG="$CALIBRATION_DIR/run.log"

mkdir -p "$CALIBRATION_DIR" "$OUTPUT_DIR"
exec > >(tee -a "$LOG") 2>&1
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[$(date --iso-8601=seconds)] collecting validation logits"
"$PYTHON" -m cnn_mamba.calibrate_joint_frame \
  --checkpoint "$CHECKPOINT" \
  --data-root "$DATA_ROOT" \
  --split val \
  --output-dir "$CALIBRATION_DIR" \
  --batch-size 1 \
  --workers 2 \
  --samples-per-window 32 \
  --seed 42 \
  --save-sample

echo "[$(date --iso-8601=seconds)] fitting vector scaling"
"$PYTHON" scripts/fit_joint_frame_vector_scaling.py \
  --sample "$CALIBRATION_DIR/calibration_sample.npz" \
  --metadata-json "$CALIBRATION_DIR/temperature.json" \
  --output-dir "$CALIBRATION_DIR" \
  --regularization 1e-4

echo "[$(date --iso-8601=seconds)] scoring chr1 with vector scaling"
"$PYTHON" -m cnn_mamba.score_joint_frame_chromosome \
  --checkpoint "$CHECKPOINT" \
  --fasta "$FASTA" \
  --chrom NC_000001.11 \
  --display-chrom chr1 \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 1 \
  --vector-calibration-json "$CALIBRATION_DIR/vector_scaling.json" \
  --save-logits

echo "[$(date --iso-8601=seconds)] exporting per-base TSV"
"$PYTHON" -m cnn_mamba.export_joint_frame_tsv \
  --probabilities "$OUTPUT_DIR/joint_frame_probabilities.npy" \
  --argmax "$OUTPUT_DIR/joint_frame_argmax.npy" \
  --output "$OUTPUT_DIR/chr1_plus_joint_frame_probabilities.tsv" \
  --chrom chr1 \
  --strand +

echo "[$(date --iso-8601=seconds)] complete"
