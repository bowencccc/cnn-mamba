#!/usr/bin/env bash
# Train chr1-included Human model with soft-weighted protein_only start/stop.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

exec "$python_bin" scripts/run_human.py \
    --regime chr1train \
    --stages train \
    --experiment-tag human_task_soft030_w10_refseq_es5_v1 \
    --data-root data/processed/human_task_soft030_w10_refseq_chr1held \
    --training-mode ignore \
    --epochs 30 \
    --scheduler-epochs 5 \
    --workers "${CNN_MAMBA_WORKERS:-4}" \
    --frame-dilation 3 \
    --joint-frame-aux-weight 0.1 \
    --joint-noncoding-weight 0.1 \
    --use-start-stop-weights \
    --early-stopping-patience 3 \
    --early-stopping-min-delta 0.0001 \
    --record-gradient-norms
