#!/usr/bin/env bash
# Re-evaluate the retrained Coleoptera-evidence model through the direct,
# header-preserving CNN--Mamba TSV -> unchanged UniAnn interface.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"
prediction_dir="$repo_dir/artifacts/joint_frame_predictions/coleoptera_eviann_v1/retrained/chrxtrain/w10/chrX_plus_vector_default"
coding_tsv="$prediction_dir/chrX_plus_joint_frame_probabilities.tsv"
score_tsv="$repo_dir/artifacts/scores/coleoptera_eviann_v1/retrained/chrxtrain/w10/chrX_plus_scores.tsv"
output_dir="$repo_dir/work/uniann/coleoptera_eviann_v1/retrained/vector_builtin_t3_headered_v1"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

"$python_bin" -m cnn_mamba.export_joint_frame_tsv \
  --probabilities "$prediction_dir/joint_frame_probabilities.npy" \
  --argmax "$prediction_dir/joint_frame_argmax.npy" \
  --output "$coding_tsv" \
  --chrom chrX \
  --strand +

"$python_bin" -m cnn_mamba.uniann_evaluate \
  --uniann-root "$uniann_root" \
  --fasta "$repo_dir/data/raw/dmel_chrX.fa" \
  --coding-frame "$coding_tsv" \
  --require-cnn-mamba-tsv-header \
  --scores "$score_tsv" \
  --reference-gtf "$repo_dir/data/raw/dmel_chrX_plus_CDS_reference.gtf" \
  --eviann-gff "$repo_dir/data/processed/coleoptera_eviann_v1/chrX_plus_EviAnn_CDS.gff" \
  --output-dir "$output_dir"

echo "[$(date --iso-8601=seconds)] corrected Coleoptera evaluation complete"
