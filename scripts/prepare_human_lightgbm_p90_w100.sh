#!/usr/bin/env bash
# Build 100-kb Human windows and cross-fitted LightGBM P90 start/stop masks.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
data_root="$repo_dir/data/processed/human_lgbm_p90_w100_refseq_chr1held"
weight_root="$repo_dir/data/weights/human_lgbm_p90_w100_refseq_chr1held"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

if [[ ! -f "$data_root/split_manifest.json" ]]; then
    "$python_bin" -m cnn_mamba.prepare_human \
        --output-dir "$data_root" \
        --eviann-gff data/raw/GCF_000001405.40_GRCh38.p14_genomic.noalpsnopatches.fna.pseudo_label.gff.gz \
        --reference-annotation data/raw/human_refseq_grch38_p14.gff.gz \
        --max-windows-per-chrom-strand 0 \
        --window-size 100000 \
        --stride 50000 \
        --region-overlap-bp 100000 \
        --include-phase-labels \
        --label-policy all_eviann \
        --overwrite
fi

if [[ ! -f "$weight_root/summary.json" ]]; then
    "$python_bin" scripts/prepare_human_lightgbm_start_stop_filter.py \
        --data-root "$data_root" \
        --lightgbm-dir results/human_site_quality_lightgbm_v1 \
        --output-root "$weight_root"
fi
