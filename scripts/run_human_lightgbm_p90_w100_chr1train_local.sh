#!/usr/bin/env bash
# Train chr1-included four-state CNN--Mamba on 100-kb windows with LightGBM-filtered start/stop labels.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
launcher_log="$repo_dir/results/human_lgbm_p90_w100_joint_frame_es5_v1/launcher.log"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
mkdir -p "$(dirname "$launcher_log")"
exec > >(tee -a "$launcher_log") 2>&1

echo "[$(date -Is)] launcher start"

"$repo_dir/scripts/prepare_human_lightgbm_p90_w100.sh"

exec "$python_bin" scripts/run_human.py \
    --regime chr1train \
    --stages train \
    --experiment-tag human_lgbm_p90_w100_joint_frame_es5_v1 \
    --data-root data/processed/human_lgbm_p90_w100_refseq_chr1held \
    --training-mode baseline \
    --epochs 30 \
    --scheduler-epochs 5 \
    --workers "${CNN_MAMBA_WORKERS:-4}" \
    --window-size 100000 \
    --stride 50000 \
    --micro-batch 1 \
    --grad-accum 8 \
    --gradient-checkpointing \
    --frame-dilation 3 \
    --joint-frame-aux-weight 0.1 \
    --joint-noncoding-weight 0.1 \
    --start-stop-weight-root data/weights/human_lgbm_p90_w100_refseq_chr1held \
    --early-stopping-patience 3 \
    --early-stopping-min-delta 0.0001
