#!/usr/bin/env bash
# Evaluate epoch-4--6 averages for the remaining weight-1.0 seeds.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
export CNN_MAMBA_REPO="$repo_dir"
export CNN_MAMBA_PYTHON="$python_bin"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

for seed in 42 43; do
    checkpoint_dir="$repo_dir/runs/coleoptera_w1_weight_decay_v1/wd1em3/seed${seed}/chrxtrain/w1/epoch_checkpoints"
    average_dir="$repo_dir/runs/coleoptera_w1_negative_weight_v1/wd1em3/n100/seed${seed}_gap_selection/checkpoint_averages"
    average="$average_dir/avg_epoch04_06.pt"
    mkdir -p "$average_dir"
    if [[ ! -s "$average" ]]; then
        "$python_bin" -m cnn_mamba.average_checkpoints \
            --checkpoints "$checkpoint_dir/epoch_004.pt" \
                          "$checkpoint_dir/epoch_005.pt" \
                          "$checkpoint_dir/epoch_006.pt" \
            --output "$average"
    fi
    CHECKPOINT="$average" \
    OUTPUT_TAG="coleoptera_w1_negative_weight_v1/wd1em3/n100/seed${seed}_gap_selection/avg_epoch04_06" \
    SELECTION_LABEL="checkpoint_arithmetic_average" SOURCE_EPOCHS="4,5,6" \
        "$repo_dir/scripts/evaluate_codon_selected_checkpoint.sh"
done
