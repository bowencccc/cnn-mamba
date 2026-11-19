#!/usr/bin/env bash
# Single-seed dropout=0.05 experiment after negative-weight selection.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
tag="coleoptera_w1_dropout005_v1/wd1em3/n100/seed44"
run_dir="$repo_dir/runs/$tag/chrxtrain/w1"
checkpoint_dir="$run_dir/epoch_checkpoints"
average_dir="$repo_dir/runs/$tag/gap_selection/checkpoint_averages"
average="$average_dir/avg_epoch03_05.pt"

export CNN_MAMBA_REPO="$repo_dir"
export CNN_MAMBA_PYTHON="$python_bin"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

if [[ ! -s "$run_dir/results.json" ]]; then
    "$python_bin" scripts/run_cell.py \
        --config "$repo_dir/configs/droso_coleoptera_w1_dropout005.json" \
        --experiment-tag "$tag" --cell w1 --regime chrxtrain \
        --seed 44 --weight-decay 1e-3 --stages train --workers 4 \
        --data-root "$repo_dir/data/processed/coleoptera_eviann_w1_v1/w1" \
        --frame-dilation 3 --joint-frame-aux-weight 0.1 \
        --joint-noncoding-weight 0.1 --candidate-negative-weight 1.0
fi

mkdir -p "$average_dir"
for epoch in 3 4 5; do
    checkpoint="$checkpoint_dir/epoch_$(printf '%03d' "$epoch").pt"
    if [[ ! -s "$checkpoint" ]]; then
        echo "required checkpoint missing: $checkpoint" >&2
        exit 1
    fi
    CHECKPOINT="$checkpoint" \
    OUTPUT_TAG="$tag/gap_selection/epoch$(printf '%02d' "$epoch")" \
    SELECTION_LABEL="raw_epoch" SOURCE_EPOCHS="$epoch" \
        "$repo_dir/scripts/evaluate_codon_selected_checkpoint.sh"
done

if [[ ! -s "$average" ]]; then
    "$python_bin" -m cnn_mamba.average_checkpoints \
        --checkpoints "$checkpoint_dir/epoch_003.pt" \
                      "$checkpoint_dir/epoch_004.pt" \
                      "$checkpoint_dir/epoch_005.pt" \
        --output "$average"
fi

CHECKPOINT="$average" \
OUTPUT_TAG="$tag/gap_selection/avg_epoch03_05" \
SELECTION_LABEL="checkpoint_arithmetic_average" SOURCE_EPOCHS="3,4,5" \
    "$repo_dir/scripts/evaluate_codon_selected_checkpoint.sh"
