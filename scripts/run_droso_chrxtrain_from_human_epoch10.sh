#!/usr/bin/env bash
# Drosophila chrX-in-training control initialized from the human epoch-10 model.
# Every training/data setting matches joint_frame_grid_v1/b010/chrxtrain/w10.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
data_root="$repo_dir/data/processed/cds_gating/w10_frame"
human_checkpoint="$repo_dir/runs/human/human_joint_frame_beta010_es5_v1/chr1train/best_model.pt"
output_dir="$repo_dir/runs/droso_from_human_pretrain_v1/chrxtrain/w10"

mkdir -p "$output_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

command=(
    "$python_bin" -m cnn_mamba.train
    --data-root "$data_root"
    --mode baseline
    --output-dir "$output_dir"
    --epochs 30
    --batch-size 4
    --grad-accum 2
    --workers 4
    --lr 0.0003
    --splice-loss-weight 0.25
    --early-stopping-patience 3
    --early-stopping-min-delta 0.0001
    --architecture cnn_mamba
    --cnn-kernel-size 7
    --frame-dilation 3
    --phase-aux-weight 0
    --cds-aux-weight 0
    --cds-boundary-radius 0
    --cds-far-loss-weight 0.1
    --joint-frame-aux-weight 0.1
    --joint-noncoding-weight 0.1
    --window-size 10000
    --stride 5000
    --record-gradient-norms
    --reference-name FlyBase
    --test-name chrX
    --include-test-in-train
)

if [[ -f "$output_dir/last_model.pt" ]]; then
    command+=(--resume)
else
    command+=(--init-checkpoint "$human_checkpoint")
fi

printf 'Command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}" 2>&1 | tee -a "$output_dir/launcher.log"
