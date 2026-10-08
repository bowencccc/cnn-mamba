#!/usr/bin/env bash
# Human chr1-held-out four-state pretraining, matched to the five-epoch baseline.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

exec "$python_bin" scripts/run_human.py \
    --regime chr1held \
    --stages train \
    --experiment-tag human_joint_frame_beta010_v1 \
    --data-root data/processed/human_joint_frame_w10_chr1held \
    --epochs 5 \
    --workers "${CNN_MAMBA_WORKERS:-4}" \
    --frame-dilation 3 \
    --joint-frame-aux-weight 0.1 \
    --joint-noncoding-weight 0.1 \
    --record-gradient-norms
