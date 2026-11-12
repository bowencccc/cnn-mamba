#!/usr/bin/env bash
# Produce the Human Rodentia-evidence chr1+ inputs consumed by downstream
# Viterbi/CRF experiments. The promoted checkpoint is never retrained here.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
checkpoint="$repo_dir/artifacts/checkpoints/current_best/human_rodentia/best_model.pt"
data_root="$repo_dir/data/processed/human_rodentia_joint_frame_w10_refseq_chr1held"
fasta="$repo_dir/data/raw/human_grch38.fa"
reference="$repo_dir/data/raw/human_refseq_grch38_p14.gff.gz"
score_dir="$repo_dir/artifacts/scores/human_rodentia_joint_frame_beta010_es5_v1/chr1train/chr1_plus"
calibration_dir="$repo_dir/results/human_rodentia_joint_frame_beta010_es5_v1/chr1train/vector_scaling"
frame_dir="$repo_dir/artifacts/joint_frame_predictions/human_rodentia_joint_frame_beta010_es5_v1/chr1train/chr1_plus_vector"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

"$python_bin" -m cnn_mamba.score_chromosome \
  --checkpoint "$checkpoint" --fasta "$fasta" \
  --chrom NC_000001.11 --display-chrom chr1 \
  --output-dir "$score_dir" --batch-size "${CNN_MAMBA_SCORE_BATCH:-4}"

"$python_bin" -m cnn_mamba.export_uniann_scores \
  --score-dir "$score_dir" --output "$score_dir/chr1_plus_scores.tsv" \
  --chrom chr1 --fasta "$fasta" --accession NC_000001.11

"$python_bin" -m cnn_mamba.evaluate_candidates \
  --species human --score-dir "$score_dir" --reference-gtf "$reference" \
  --fasta "$fasta" --chrom NC_000001.11 --reference-chrom chr1 \
  --output "$score_dir/candidate_auprc.tsv"

if [[ ! -s "$calibration_dir/calibration_sample.npz" ]]; then
  "$python_bin" -m cnn_mamba.calibrate_joint_frame \
    --checkpoint "$checkpoint" --data-root "$data_root" --split val \
    --output-dir "$calibration_dir" --batch-size "${CNN_MAMBA_SCORE_BATCH:-4}" \
    --workers "${CNN_MAMBA_WORKERS:-4}" --samples-per-window 32 \
    --seed 42 --bins 20 --save-sample
fi
if [[ ! -s "$calibration_dir/vector_scaling.json" ]]; then
  "$python_bin" -m cnn_mamba.fit_vector_scaling \
    --sample "$calibration_dir/calibration_sample.npz" \
    --metadata-json "$calibration_dir/temperature.json" \
    --output-dir "$calibration_dir" --regularization 0.0001
fi

"$python_bin" -m cnn_mamba.score_joint_frame_chromosome \
  --checkpoint "$checkpoint" --fasta "$fasta" \
  --chrom NC_000001.11 --display-chrom chr1 \
  --vector-calibration-json "$calibration_dir/vector_scaling.json" \
  --save-logits --output-dir "$frame_dir" \
  --batch-size "${CNN_MAMBA_SCORE_BATCH:-4}"

"$python_bin" -m cnn_mamba.current_best
