#!/usr/bin/env bash
# Resume downstream evaluation after a completed chrX-included training run.

set -euo pipefail

label="${1:-n050}"
repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
tag="incomplete_label_v2/negative_weight/$label"
score_dir="$repo_dir/artifacts/scores/$tag/chrxtrain/w10"
result_dir="$repo_dir/results/$tag/chrxtrain/w10"
prediction_dir="$repo_dir/artifacts/joint_frame_predictions/$tag/chrxtrain/w10/chrX_plus"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

"$python_bin" -m cnn_mamba.evaluate_annotation_gap \
    --score-dir "$score_dir" \
    --eviann-gff "$repo_dir/data/raw/dmel_eviann_pseudo_label.gff" \
    --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
    --fasta "$repo_dir/data/raw/dmel_genome.fa" \
    --chrom NC_004354.4 --strand + \
    --output "$result_dir/annotation_gap_metrics.tsv"

"$python_bin" -m cnn_mamba.score_joint_frame_chromosome \
    --checkpoint "$repo_dir/runs/$tag/chrxtrain/w10/best_model.pt" \
    --fasta "$repo_dir/data/raw/dmel_genome.fa" \
    --chrom NC_004354.4 --display-chrom chrX \
    --output-dir "$prediction_dir" --batch-size 8

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
    --chrom chrX --strand +

"$python_bin" scripts/run_incomplete_label_uniann.py \
    --label "$label" --temperature 9

"$python_bin" scripts/plot_single_run_loss.py \
    --results-json "$repo_dir/runs/$tag/chrxtrain/w10/results.json" \
    --output-dir "$result_dir/loss" \
    --prefix chrxtrain_train_validation_loss
