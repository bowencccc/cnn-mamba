#!/usr/bin/env bash
# Train and evaluate one chrX-included incomplete-label experiment.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 ALPHA (one of: 0.5 0.3 0.25 0.1 0.03 0.01)" >&2
    exit 2
fi

alpha="$1"
case "$alpha" in
    0.5) label="n050" ;;
    0.3) label="n030" ;;
    0.25) label="n025" ;;
    0.1) label="n010" ;;
    0.03) label="n003" ;;
    0.01) label="n001" ;;
    *) echo "unsupported alpha: $alpha" >&2; exit 2 ;;
esac

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"
tag="incomplete_label_v2/negative_weight/$label"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

"$python_bin" scripts/run_cell.py \
    --config "$repo_dir/configs/droso_window_early_stopping.json" \
    --experiment-tag "$tag" \
    --cell w10 \
    --regime chrxtrain \
    --stages train,score,auprc,export \
    --workers "${CNN_MAMBA_WORKERS:-4}" \
    --data-root "$repo_dir/data/processed/cds_gating/w10_frame" \
    --frame-dilation 3 \
    --joint-frame-aux-weight 0.1 \
    --joint-noncoding-weight 0.1 \
    --candidate-negative-weight "$alpha" \
    --record-gradient-norms \
    --uniann-root "$uniann_root"

score_dir="$repo_dir/artifacts/scores/$tag/chrxtrain/w10"
result_dir="$repo_dir/results/$tag/chrxtrain/w10"
prediction_dir="$repo_dir/artifacts/joint_frame_predictions/$tag/chrxtrain/w10/chrX_plus"

"$python_bin" -m cnn_mamba.evaluate_annotation_gap \
    --score-dir "$score_dir" \
    --eviann-gff "$repo_dir/data/raw/dmel_eviann_pseudo_label.gff" \
    --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
    --fasta "$repo_dir/data/raw/dmel_genome.fa" \
    --chrom NC_004354.4 \
    --strand + \
    --output "$result_dir/annotation_gap_metrics.tsv"

"$python_bin" -m cnn_mamba.score_joint_frame_chromosome \
    --checkpoint "$repo_dir/runs/$tag/chrxtrain/w10/best_model.pt" \
    --fasta "$repo_dir/data/raw/dmel_genome.fa" \
    --chrom NC_004354.4 \
    --display-chrom chrX \
    --output-dir "$prediction_dir" \
    --batch-size 8

"$python_bin" -m cnn_mamba.evaluate_joint_frame \
    --prediction-dir "$prediction_dir" \
    --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
    --fasta "$repo_dir/data/raw/dmel_genome.fa" \
    --chrom NC_004354.4 \
    --output-dir "$result_dir/joint_frame"

"$python_bin" -m cnn_mamba.export_joint_frame_tsv \
    --probabilities "$prediction_dir/joint_frame_probabilities.npy" \
    --argmax "$prediction_dir/joint_frame_argmax.npy" \
    --output "$prediction_dir/chrX_plus_joint_frame_probabilities.tsv" \
    --chrom chrX \
    --strand +

"$python_bin" scripts/run_incomplete_label_uniann.py \
    --label "$label" \
    --temperature 9

"$python_bin" scripts/plot_single_run_loss.py \
    --results-json "$repo_dir/runs/$tag/chrxtrain/w10/results.json" \
    --output-dir "$result_dir/loss" \
    --prefix chrxtrain_train_validation_loss
