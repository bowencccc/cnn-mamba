#!/usr/bin/env bash
# Fit Drosophila four-state temperature scaling and export calibrated chrX+ scores.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
wait_session="${CNN_MAMBA_WAIT_TMUX_SESSION:-}"
checkpoint="$repo_dir/runs/joint_frame_grid_v1/b010/chrxtrain/w10/best_model.pt"
data_root="$repo_dir/data/processed/cds_gating/w10_frame"
calibration_dir="$repo_dir/results/joint_frame_grid_v1/b010/chrxtrain/w10/temperature_scaling"
output_dir="$repo_dir/artifacts/joint_frame_predictions/joint_frame_grid_v1/b010/chrxtrain/w10/chrX_plus_temperature_scaled"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

if [[ -n "$wait_session" ]]; then
    echo "waiting for tmux session: $wait_session"
    while tmux has-session -t "$wait_session" 2>/dev/null; do
        sleep 30
    done
fi

echo "fitting Drosophila validation temperature"
"$python_bin" -m cnn_mamba.calibrate_joint_frame \
    --checkpoint "$checkpoint" \
    --data-root "$data_root" \
    --split val \
    --output-dir "$calibration_dir" \
    --batch-size "${CNN_MAMBA_BATCH_SIZE:-8}" \
    --workers "${CNN_MAMBA_WORKERS:-4}" \
    --samples-per-window "${CNN_MAMBA_CALIBRATION_SAMPLES_PER_WINDOW:-32}" \
    --seed 42 \
    --save-sample

echo "scoring calibrated Drosophila chrX+ four-state probabilities"
"$python_bin" -m cnn_mamba.score_joint_frame_chromosome \
    --checkpoint "$checkpoint" \
    --fasta data/raw/dmel_genome.fa \
    --chrom NC_004354.4 \
    --display-chrom chrX \
    --calibration-json "$calibration_dir/temperature.json" \
    --output-dir "$output_dir" \
    --batch-size "${CNN_MAMBA_SCORE_BATCH_SIZE:-8}"

echo "exporting one-row-per-base TSV"
"$python_bin" -m cnn_mamba.export_joint_frame_tsv \
    --probabilities "$output_dir/joint_frame_probabilities.npy" \
    --argmax "$output_dir/joint_frame_argmax.npy" \
    --output "$output_dir/chrX_plus_joint_frame_probabilities.tsv" \
    --chrom chrX \
    --strand + \
    --chunk-size "${CNN_MAMBA_EXPORT_CHUNK_SIZE:-250000}"

echo "temperature-scaled Drosophila chrX+ export complete"
