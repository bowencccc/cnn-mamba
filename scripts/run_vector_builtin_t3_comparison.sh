#!/usr/bin/env bash
set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"
chrx_fasta="$repo_dir/data/raw/dmel_chrX.fa"
reference="$repo_dir/data/raw/dmel_chrX_plus_CDS_reference.gtf"
eviann="$repo_dir/data/raw/dmel_chrX_plus_EviAnn_CDS.gff"
log_dir="$repo_dir/results/vector_builtin_t3_comparison_v1"

mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/launcher.log") 2>&1
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

evaluate_one() {
  local label="$1"
  local prediction_dir="$2"
  local site_scores="$3"
  local work_dir="$4"
  local coding_tsv="$prediction_dir/chrX_plus_joint_frame_probabilities.tsv"

  echo "[$(date --iso-8601=seconds)] $label: exporting header-compatible TSV"
  if [[ ! -s "$coding_tsv.complete.json" ]]; then
    "$python_bin" -m cnn_mamba.export_joint_frame_tsv \
      --probabilities "$prediction_dir/joint_frame_probabilities.npy" \
      --argmax "$prediction_dir/joint_frame_argmax.npy" \
      --output "$coding_tsv" \
      --chrom chrX \
      --strand +
  fi

  "$python_bin" -m cnn_mamba.uniann_evaluate \
    --uniann-root "$uniann_root" \
    --fasta "$chrx_fasta" \
    --coding-frame "$coding_tsv" \
    --require-cnn-mamba-tsv-header \
    --scores "$site_scores" \
    --reference-gtf "$reference" \
    --eviann-gff "$eviann" \
    --output-dir "$work_dir"
}

evaluate_one \
  w10 \
  "$repo_dir/artifacts/joint_frame_predictions/joint_frame_grid_v1/b010/chrxtrain/w10/chrX_plus_vector_scaled" \
  "$repo_dir/artifacts/scores/joint_frame_grid_v1/b010/chrxtrain/w10/chrX_plus_scores.tsv" \
  "$repo_dir/work/uniann/vector_builtin_t3_comparison_v1/w10"

for cell in w0p5 w1 w2 w5; do
  evaluate_one \
    "$cell" \
    "$repo_dir/artifacts/joint_frame_predictions/droso_small_window_v1_four_state_vector/chrxtrain/$cell/chrX_plus" \
    "$repo_dir/artifacts/scores/droso_small_window_v1/chrxtrain/$cell/chrX_plus_scores.tsv" \
    "$repo_dir/work/uniann/droso_small_window_v1_four_state_vector_builtin_t3/$cell"
done

echo "[$(date --iso-8601=seconds)] vector + built-in-equivalent comparison complete"
