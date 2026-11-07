#!/usr/bin/env bash
# Build Human task-specific labels with 0.3-weight protein_only start/stop sites.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

exec "$python_bin" -m cnn_mamba.prepare_human \
    --output-dir data/processed/human_task_soft030_w10_refseq_chr1held \
    --reference-annotation data/raw/human_refseq_grch38_p14.gff.gz \
    --max-windows-per-chrom-strand 0 \
    --region-overlap-bp 10000 \
    --include-phase-labels \
    --label-policy task_soft \
    --medium-start-stop-weight 0.3
