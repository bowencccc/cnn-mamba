#!/usr/bin/env bash
# Evaluate a checkpoint selected from an existing run.
# This script does not retrain and writes to a separate namespace so the
# original four-task-mean selected checkpoint and outputs remain untouched.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"

checkpoint="${CHECKPOINT:-$repo_dir/runs/coleoptera_w1_weight_decay_v1/wd1em3/seed44/chrxtrain/w1/epoch_checkpoints/epoch_004.pt}"
tag="${OUTPUT_TAG:-coleoptera_w1_weight_decay_v1/wd1em3/seed44_codon_monitor_epoch4}"
selection_label="${SELECTION_LABEL:-posthoc_epoch4_codon_monitor}"
source_epochs="${SOURCE_EPOCHS:-4}"
data_root="$repo_dir/data/processed/coleoptera_eviann_w1_v1/w1"
fasta="$repo_dir/data/raw/dmel_genome.fa"
chrx_fasta="$repo_dir/data/raw/dmel_chrX.fa"
reference="$repo_dir/data/raw/dmel_chrX_plus_CDS_reference.gtf"
eviann="$repo_dir/data/processed/coleoptera_eviann_v1/chrX_plus_EviAnn_CDS.gff"

score_dir="$repo_dir/artifacts/scores/$tag/chrxtrain/w1"
result_dir="$repo_dir/results/$tag/chrxtrain/w1"
calibration_dir="$result_dir/calibration"
vector_dir="$result_dir/vector_scaling"
prediction_dir="$repo_dir/artifacts/joint_frame_predictions/$tag/chrxtrain/w1/chrX_plus_vector_scaled"
coding_tsv="$prediction_dir/chrX_plus_joint_frame_probabilities_vector.tsv"
work_dir="$repo_dir/work/uniann/$tag/chrxtrain/w1/vector_uniann_builtin_header"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
mkdir -p "$score_dir" "$result_dir"

if [[ ! -s "$score_dir/chrX_plus_scores.tsv" ]]; then
    "$python_bin" -m cnn_mamba.score_chromosome \
        --checkpoint "$checkpoint" --output-dir "$score_dir" \
        --fasta "$fasta" --chrom NC_004354.4 --display-chrom chrX \
        --batch-size 64
    "$python_bin" -m cnn_mamba.evaluate_candidates \
        --score-dir "$score_dir" \
        --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
        --fasta "$fasta" --chrom NC_004354.4 \
        --output "$score_dir/candidate_auprc.tsv"
    "$python_bin" -m cnn_mamba.export_uniann_scores \
        --score-dir "$score_dir" --output "$score_dir/chrX_plus_scores.tsv" \
        --chrom X --fasta "$fasta" --accession NC_004354.4
fi

if [[ ! -s "$calibration_dir/calibration_sample.npz" ]]; then
    "$python_bin" -m cnn_mamba.calibrate_joint_frame \
        --checkpoint "$checkpoint" --data-root "$data_root" --split val \
        --output-dir "$calibration_dir" --batch-size 64 --workers 4 \
        --samples-per-window 32 --seed 42 --bins 20 --save-sample
fi
if [[ ! -s "$vector_dir/vector_scaling.json" ]]; then
    "$python_bin" -m cnn_mamba.fit_vector_scaling \
        --sample "$calibration_dir/calibration_sample.npz" \
        --metadata-json "$calibration_dir/temperature.json" \
        --output-dir "$vector_dir" --regularization 0.0001
fi
if [[ ! -s "$prediction_dir/joint_frame_probabilities.npy" ]]; then
    "$python_bin" -m cnn_mamba.score_joint_frame_chromosome \
        --checkpoint "$checkpoint" --fasta "$fasta" \
        --chrom NC_004354.4 --display-chrom chrX \
        --output-dir "$prediction_dir" --batch-size 64 \
        --vector-calibration-json "$vector_dir/vector_scaling.json"
fi
if [[ ! -s "$coding_tsv.complete.json" ]]; then
    "$python_bin" -m cnn_mamba.export_joint_frame_tsv \
        --probabilities "$prediction_dir/joint_frame_probabilities.npy" \
        --argmax "$prediction_dir/joint_frame_argmax.npy" \
        --output "$coding_tsv" --chrom chrX --strand +
fi

if [[ ! -s "$work_dir/locus_metrics.tsv" ]]; then
    "$python_bin" -m cnn_mamba.uniann_evaluate \
        --uniann-root "$uniann_root" --fasta "$chrx_fasta" \
        --coding-frame "$coding_tsv" --require-cnn-mamba-tsv-header \
        --scores "$score_dir/chrX_plus_scores.tsv" \
        --reference-gtf "$reference" --eviann-gff "$eviann" \
        --output-dir "$work_dir"
fi

git -C "$uniann_root" rev-parse HEAD > "$work_dir/uniann_commit.txt"
printf 'checkpoint\t%s\n' "$checkpoint" > "$result_dir/selection.tsv"
printf 'selection_label\t%s\n' "$selection_label" >> "$result_dir/selection.tsv"
printf 'source_epochs\t%s\n' "$source_epochs" >> "$result_dir/selection.tsv"
cat "$score_dir/candidate_auprc.tsv"
cat "$work_dir/locus_metrics.tsv"
cat "$work_dir/gap_locus_metrics.tsv"
