#!/bin/bash
# Controlled 10 kb chrX-held-out frame ablation on one GPU.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
data_root="$repo_dir/data/processed/frame_ablation/w10_frame"
config="$repo_dir/configs/droso_window_early_stopping.json"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

variants=(frame_d3 phase_aux combined)
frame_dilations=(3 0 3)
phase_weights=(0 0.1 0.1)

for index in "${!variants[@]}"; do
    variant="${variants[$index]}"
    "$python_bin" scripts/run_cell.py \
        --config "$config" \
        --experiment-tag "frame_ablation_v1/$variant" \
        --cell w10 \
        --regime heldout \
        --stages train,score,auprc \
        --workers "${CNN_MAMBA_WORKERS:-4}" \
        --data-root "$data_root" \
        --frame-dilation "${frame_dilations[$index]}" \
        --phase-aux-weight "${phase_weights[$index]}"
done

"$python_bin" scripts/summarize_frame_ablation.py
