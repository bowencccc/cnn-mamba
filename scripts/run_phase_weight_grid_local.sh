#!/bin/bash
# Train/evaluate the controlled phase-loss grid sequentially on one GPU.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
data_root="$repo_dir/data/processed/frame_ablation/w10_frame"
config="$repo_dir/configs/droso_window_early_stopping.json"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

# lambda=0.10 already exists for both regimes and is reused by the summary.
labels=(p003 p005 p020)
weights=(0.03 0.05 0.20)

for regime in heldout chrxtrain; do
    if [[ "$regime" == "heldout" ]]; then
        stages="train,score,auprc"
    else
        stages="train,score,auprc,export,uniann"
    fi
    for index in "${!labels[@]}"; do
        label="${labels[$index]}"
        weight="${weights[$index]}"
        "$python_bin" scripts/run_cell.py \
            --config "$config" \
            --experiment-tag "phase_weight_grid_v1/$label" \
            --cell w10 \
            --regime "$regime" \
            --stages "$stages" \
            --workers "${CNN_MAMBA_WORKERS:-4}" \
            --data-root "$data_root" \
            --frame-dilation 3 \
            --phase-aux-weight "$weight" \
            --uniann-root "$uniann_root"
    done
done

"$python_bin" scripts/summarize_phase_weight_grid.py
