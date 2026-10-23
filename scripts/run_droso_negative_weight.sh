#!/usr/bin/env bash
# Run one leakage-safe held-out-chrX negative-weight experiment at a time.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 ALPHA (one of: 0.5 0.25 0.1)" >&2
    exit 2
fi

alpha="$1"
case "$alpha" in
    0.5) label="n050" ;;
    0.25) label="n025" ;;
    0.1) label="n010" ;;
    *) echo "unsupported alpha: $alpha" >&2; exit 2 ;;
esac

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
tag="incomplete_label_v1/negative_weight/$label"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

"$python_bin" scripts/run_cell.py \
    --config "$repo_dir/configs/droso_window_early_stopping.json" \
    --experiment-tag "$tag" \
    --cell w10 \
    --regime heldout \
    --stages train,score,auprc \
    --workers "${CNN_MAMBA_WORKERS:-4}" \
    --data-root "$repo_dir/data/processed/cds_gating/w10_frame" \
    --frame-dilation 3 \
    --joint-frame-aux-weight 0.1 \
    --joint-noncoding-weight 0.1 \
    --candidate-negative-weight "$alpha" \
    --record-gradient-norms

score_dir="$repo_dir/artifacts/scores/$tag/heldout/w10"
result_dir="$repo_dir/results/$tag/heldout/w10"
"$python_bin" -m cnn_mamba.evaluate_annotation_gap \
    --score-dir "$score_dir" \
    --eviann-gff "$repo_dir/data/raw/dmel_eviann_pseudo_label.gff" \
    --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
    --fasta "$repo_dir/data/raw/dmel_genome.fa" \
    --chrom NC_004354.4 \
    --strand + \
    --output "$result_dir/annotation_gap_metrics.tsv"
