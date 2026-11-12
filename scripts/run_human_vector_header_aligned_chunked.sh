#!/usr/bin/env bash
# Re-run Human chr1 deployment evaluation with the vector-scaled four-state
# probabilities and the retained-header-equivalent frame mapping.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
chunk_pipeline="${HUMAN_CHUNK_PIPELINE_ROOT:-$repo_dir/../human_chr1_oracle_uniann_minimal_20260824/full_chunked}"
output_dir="$repo_dir/results/human_joint_frame_beta010_es5_v1/chr1train/epoch10_vector_header_aligned_v1"
score_root="$repo_dir/artifacts/scores/human_joint_frame_beta010_es5_v1/chr1train"
frame_dir="$repo_dir/artifacts/joint_frame_predictions/human_joint_frame_beta010_es5_v1/chr1train/epoch10_vector_scaled/chr1_plus"

# The external chunk runner predates the four-state head and retains option
# names containing "psauron". Both arguments below point to CNN--Mamba vector-
# scaled four-state outputs; this run does not use a PSAURON model or scores.

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

if [[ ! -f "$chunk_pipeline/run_model_pipeline.py" ]]; then
  echo "Missing Human chunk pipeline: $chunk_pipeline/run_model_pipeline.py" >&2
  exit 1
fi

"$python_bin" "$chunk_pipeline/run_model_pipeline.py" \
  --score-tsv "$score_root/chr1_plus_scores.tsv" \
  --score-column prob \
  --score-metadata "$score_root/chr1_plus/metadata.json" \
  --output-dir "$output_dir" \
  --analysis-label human_chr1train_epoch10_vector_header_aligned_v1 \
  --provenance-class diagnostic_leaky_anchor \
  --stage all \
  --variants baseline \
  --emission-workers "${HUMAN_EMISSION_WORKERS:-2}" \
  --psauron-index "$frame_dir/joint_frame_probabilities.npy" \
  --converter "$repo_dir/scripts/stream_joint_frame_header_aligned_emissions.py" \
  --psauron-csv "$frame_dir/chr1_plus_joint_frame_probabilities.tsv" \
  --uniann-bin "${UNIANN_ROOT:-$repo_dir/../UniAnn}/bin" \
  --publish-experiment-root "$output_dir/published" \
  --publish-score-kind raw \
  --publish-mode vector_header_aligned

echo "[$(date --iso-8601=seconds)] corrected Human vector evaluation complete"
