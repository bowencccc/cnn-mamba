#!/usr/bin/env bash
set -euo pipefail

# Evaluate the 5-kb seed-43 model using its own four-state head.  This pipeline
# deliberately does not accept or read a PSAURON file.
repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"
tag="droso_small_window_seed43_four_state_t9"
cell="w5"

checkpoint="$repo_dir/runs/droso_small_window_seed43/chrxtrain/$cell/best_model.pt"
site_scores="$repo_dir/artifacts/scores/droso_small_window_seed43/chrxtrain/$cell/chrX_plus_scores.tsv"
prediction_dir="$repo_dir/artifacts/joint_frame_predictions/$tag/chrxtrain/$cell/chrX_plus"
emission="$prediction_dir/out.ps.txt"
emission_manifest="$prediction_dir/out.ps.txt.manifest.json"
work_dir="$repo_dir/work/uniann/$tag/$cell"
log_dir="$repo_dir/results/$tag"
fasta="$repo_dir/data/raw/dmel_genome.fa"
chrx_fasta="$repo_dir/data/raw/dmel_chrX.fa"

mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/launcher.log") 2>&1
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

chrom_len="$(awk '$1 == "NC_004354.4" {print $2}' "$fasta.fai")"
if [[ -z "$chrom_len" ]]; then
  echo "Could not determine NC_004354.4 length from $fasta.fai" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] scoring seed-43 four-state head at T=9"
"$python_bin" -m cnn_mamba.score_joint_frame_chromosome \
  --checkpoint "$checkpoint" \
  --fasta "$fasta" \
  --chrom NC_004354.4 \
  --display-chrom chrX \
  --output-dir "$prediction_dir" \
  --batch-size 128 \
  --temperature 9

echo "[$(date --iso-8601=seconds)] converting CNN-Mamba probabilities to UniAnn emissions"
"$python_bin" scripts/stream_joint_frame_no_gate_emissions.py convert \
  --index "$prediction_dir/joint_frame_probabilities.npy" \
  --start0 0 \
  --end0 "$chrom_len" \
  --out-ps "$emission" \
  --out-fasta "$chrx_fasta" \
  --seqid NC_004354.4 \
  --manifest "$emission_manifest"

echo "[$(date --iso-8601=seconds)] running strict UniAnn evaluation without PSAURON"
"$python_bin" -m cnn_mamba.uniann_evaluate \
  --uniann-root "$uniann_root" \
  --fasta "$chrx_fasta" \
  --coding-frame "$prediction_dir/joint_frame_probabilities.npy" \
  --precomputed-emissions "$emission" \
  --scores "$site_scores" \
  --reference-gtf "$repo_dir/data/raw/dmel_chrX_plus_CDS_reference.gtf" \
  --eviann-gff "$repo_dir/data/raw/dmel_chrX_plus_EviAnn_CDS.gff" \
  --output-dir "$work_dir"

echo "[$(date --iso-8601=seconds)] complete: $work_dir/locus_metrics.tsv"
