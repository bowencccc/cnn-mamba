#!/usr/bin/env bash
# Fixed-WD candidate-negative-weight experiment with downstream gap-locus selection.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
config="$repo_dir/configs/droso_coleoptera_w1_weight_decay.json"
data_root="$repo_dir/data/processed/coleoptera_eviann_w1_v1/w1"
experiment_root="coleoptera_w1_negative_weight_v1/wd1em3"

export CNN_MAMBA_REPO="$repo_dir"
export CNN_MAMBA_PYTHON="$python_bin"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

label_for_weight() {
    case "$1" in
        1.0) echo n100 ;;
        0.7) echo n070 ;;
        0.5) echo n050 ;;
        *) return 2 ;;
    esac
}

train_weight() {
    local weight="$1"
    local label
    label="$(label_for_weight "$weight")"
    local tag="$experiment_root/$label/seed44"
    local result="$repo_dir/runs/$tag/chrxtrain/w1/results.json"
    if [[ -s "$result" ]]; then
        echo "training already complete: $result"
        return
    fi
    "$python_bin" scripts/run_cell.py \
        --config "$config" --experiment-tag "$tag" \
        --cell w1 --regime chrxtrain --seed 44 \
        --weight-decay 1e-3 --stages train --workers 4 \
        --data-root "$data_root" --frame-dilation 3 \
        --joint-frame-aux-weight 0.1 --joint-noncoding-weight 0.1 \
        --candidate-negative-weight "$weight"
}

checkpoint_dir_for_weight() {
    local weight="$1"
    if [[ "$weight" == "1.0" ]]; then
        echo "$repo_dir/runs/coleoptera_w1_weight_decay_v1/wd1em3/seed44/chrxtrain/w1/epoch_checkpoints"
    else
        local label
        label="$(label_for_weight "$weight")"
        echo "$repo_dir/runs/$experiment_root/$label/seed44/chrxtrain/w1/epoch_checkpoints"
    fi
}

evaluate_checkpoint() {
    local checkpoint="$1"
    local output_tag="$2"
    local label="$3"
    local epochs="$4"
    CHECKPOINT="$checkpoint" OUTPUT_TAG="$output_tag" \
    SELECTION_LABEL="$label" SOURCE_EPOCHS="$epochs" \
        "$repo_dir/scripts/evaluate_codon_selected_checkpoint.sh"
}

# The weight-1.0 control already exists. Train only the two new settings.
train_weight 0.7
train_weight 0.5

for weight in 1.0 0.7 0.5; do
    label="$(label_for_weight "$weight")"
    checkpoint_dir="$(checkpoint_dir_for_weight "$weight")"
    output_root="$experiment_root/$label/seed44_gap_selection"
    for epoch in 4 5 6; do
        checkpoint="$checkpoint_dir/epoch_$(printf '%03d' "$epoch").pt"
        if [[ ! -s "$checkpoint" ]]; then
            echo "required checkpoint missing: $checkpoint" >&2
            exit 1
        fi
        evaluate_checkpoint \
            "$checkpoint" "$output_root/epoch$(printf '%02d' "$epoch")" \
            "raw_epoch" "$epoch"
    done

    average_dir="$repo_dir/runs/$experiment_root/$label/seed44_gap_selection/checkpoint_averages"
    average="$average_dir/avg_epoch04_06.pt"
    if [[ ! -s "$average" ]]; then
        "$python_bin" -m cnn_mamba.average_checkpoints \
            --checkpoints "$checkpoint_dir/epoch_004.pt" \
                          "$checkpoint_dir/epoch_005.pt" \
                          "$checkpoint_dir/epoch_006.pt" \
            --output "$average"
    fi
    evaluate_checkpoint \
        "$average" "$output_root/avg_epoch04_06" \
        "checkpoint_arithmetic_average" "4,5,6"
done
