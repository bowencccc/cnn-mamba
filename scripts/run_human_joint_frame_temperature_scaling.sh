#!/usr/bin/env bash
# Fit scalar temperature scaling on CHESS validation chromosomes (never chr1).

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

exec "$python_bin" -m cnn_mamba.calibrate_joint_frame \
    --checkpoint runs/human/human_joint_frame_beta010_v1/chr1held/best_model.pt \
    --data-root data/processed/human_joint_frame_w10_chr1held \
    --split val \
    --output-dir results/human_joint_frame_beta010_v1/temperature_scaling \
    --batch-size "${CNN_MAMBA_BATCH_SIZE:-8}" \
    --workers "${CNN_MAMBA_WORKERS:-4}" \
    --samples-per-window "${CNN_MAMBA_CALIBRATION_SAMPLES_PER_WINDOW:-32}" \
    --seed 42 \
    --save-sample
