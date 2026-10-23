#!/usr/bin/env bash
# Train and evaluate the matched chrX-included model using Coleoptera EviAnn labels.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
source_gff="$repo_dir/data/raw/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pseudo_label.coleoptera.gff.gz"
processed_root="$repo_dir/data/processed/coleoptera_eviann_v1"
data_root="$processed_root/w10"
chrx_gff="$processed_root/chrX_plus_EviAnn_CDS.gff"
tag="coleoptera_eviann_v1/retrained"
run_dir="$repo_dir/runs/$tag/chrxtrain/w10"
score_dir="$repo_dir/artifacts/scores/$tag/chrxtrain/w10"
result_dir="$repo_dir/results/$tag/chrxtrain/w10"
prediction_dir="$repo_dir/artifacts/joint_frame_predictions/$tag/chrxtrain/w10/chrX_plus"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

if [[ ! -f "$processed_root/.complete.json" ]]; then
    "$python_bin" -m cnn_mamba.prepare_droso \
        --config "$repo_dir/configs/droso_coleoptera_w10.json" \
        --fasta "$repo_dir/data/raw/dmel_genome.fa" \
        --eviann-gff "$source_gff" \
        --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
        --output-root "$processed_root" \
        --include-phase-labels
fi

if [[ ! -f "$chrx_gff" ]]; then
    "$python_bin" scripts/filter_eviann_cds.py \
        --input "$source_gff" --output "$chrx_gff" \
        --chrom NC_004354.4 --strand +
fi

"$python_bin" scripts/run_cell.py \
    --config "$repo_dir/configs/droso_coleoptera_w10.json" \
    --experiment-tag "$tag" \
    --cell w10 --regime chrxtrain \
    --stages train,score,auprc,export \
    --workers "${CNN_MAMBA_WORKERS:-4}" \
    --data-root "$data_root" \
    --frame-dilation 3 \
    --joint-frame-aux-weight 0.1 \
    --joint-noncoding-weight 0.1 \
    --candidate-negative-weight 1.0

"$python_bin" -m cnn_mamba.evaluate_annotation_gap \
    --score-dir "$score_dir" \
    --eviann-gff "$source_gff" \
    --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
    --fasta "$repo_dir/data/raw/dmel_genome.fa" \
    --chrom NC_004354.4 --strand + \
    --output "$result_dir/annotation_gap_metrics.tsv"

"$python_bin" -m cnn_mamba.score_joint_frame_chromosome \
    --checkpoint "$run_dir/best_model.pt" \
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
    --tag "$tag" --temperature 9 --eviann-gff "$chrx_gff"

"$python_bin" scripts/plot_single_run_loss.py \
    --results-json "$run_dir/results.json" \
    --output-dir "$result_dir/loss" \
    --prefix chrxtrain_train_validation_loss

echo "[$(date -Is)] Coleoptera EviAnn experiment complete"
