#!/usr/bin/env bash
set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"
fasta="$repo_dir/data/raw/dmel_genome.fa"
chrx_fasta="$repo_dir/data/raw/dmel_chrX.fa"
reference="$repo_dir/data/raw/dmel_chrX_plus_CDS_reference.gtf"
eviann="$repo_dir/data/raw/dmel_chrX_plus_EviAnn_CDS.gff"
log_dir="$repo_dir/results/vector_builtin_t3_comparison_v1"

mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/launcher.log") 2>&1
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

chrom_len="$(awk '$1 == "NC_004354.4" {print $2}' "$fasta.fai")"
if [[ -z "$chrom_len" ]]; then
  echo "Could not determine NC_004354.4 length" >&2
  exit 1
fi

evaluate_one() {
  local label="$1"
  local prediction_dir="$2"
  local site_scores="$3"
  local work_dir="$4"
  local emission_dir="$repo_dir/artifacts/uniann_emissions/vector_builtin_t3_comparison_v1/$label"
  local emission="$emission_dir/out.ps.txt"
  local manifest="$emission_dir/out.ps.txt.manifest.json"

  echo "[$(date --iso-8601=seconds)] $label: generating built-in-equivalent T=3+gate emissions"
  mkdir -p "$emission_dir"
  if [[ ! -s "$manifest" ]]; then
    "$python_bin" scripts/stream_joint_frame_uniann_emissions.py convert \
      --index "$prediction_dir/joint_frame_probabilities.npy" \
      --start0 0 \
      --end0 "$chrom_len" \
      --out-ps "$emission" \
      --out-fasta "$chrx_fasta" \
      --seqid NC_004354.4 \
      --manifest "$manifest"
  fi

  "$python_bin" -m cnn_mamba.uniann_evaluate \
    --uniann-root "$uniann_root" \
    --fasta "$chrx_fasta" \
    --coding-frame "$prediction_dir/joint_frame_probabilities.npy" \
    --precomputed-emissions "$emission" \
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
