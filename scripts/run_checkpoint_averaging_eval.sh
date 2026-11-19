#!/usr/bin/env bash
# Evaluate epoch-averaged WD=1e-3 seed-44 checkpoints without retraining.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
checkpoint_dir="$repo_dir/runs/coleoptera_w1_weight_decay_v1/wd1em3/seed44/chrxtrain/w1/epoch_checkpoints"
average_dir="$repo_dir/runs/coleoptera_w1_weight_decay_v1/wd1em3/seed44/checkpoint_averages"

export CNN_MAMBA_REPO="$repo_dir"
export CNN_MAMBA_PYTHON="$python_bin"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$average_dir"

average_and_evaluate() {
    local name="$1"
    local epochs="$2"
    shift 2
    local output="$average_dir/$name.pt"
    local checkpoint_args=()
    local epoch
    for epoch in "$@"; do
        checkpoint_args+=("$checkpoint_dir/epoch_$(printf '%03d' "$epoch").pt")
    done
    if [[ ! -s "$output" ]]; then
        "$python_bin" -m cnn_mamba.average_checkpoints \
            --checkpoints "${checkpoint_args[@]}" --output "$output"
    fi
    CHECKPOINT="$output" \
    OUTPUT_TAG="coleoptera_w1_weight_decay_v1/wd1em3/seed44_checkpoint_average/$name" \
    SELECTION_LABEL="checkpoint_arithmetic_average" \
    SOURCE_EPOCHS="$epochs" \
        "$repo_dir/scripts/evaluate_codon_selected_checkpoint.sh"
}

average_and_evaluate avg_epoch04_05 4,5 4 5
average_and_evaluate avg_epoch04_06 4,5,6 4 5 6
average_and_evaluate avg_epoch03_05 3,4,5 3 4 5
