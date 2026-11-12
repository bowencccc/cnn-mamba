#!/usr/bin/env bash
set -euo pipefail
repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
"$python_bin" -m cnn_mamba.probe_overcounting \
  --probabilities artifacts/joint_frame_predictions/joint_frame_grid_v1/b010/chrxtrain/w10/chrX_plus/joint_frame_probabilities.npy \
  --reference-gtf data/raw/dmel_reference.gtf \
  --windows-root data/processed/cds_gating/w10_frame \
  --checkpoint runs/joint_frame_grid_v1/b010/chrxtrain/w10/best_model.pt \
  --dataset drosophila \
  --output results/belief_probe_v1/E_overcounting/drosophila "$@"

"$python_bin" -m cnn_mamba.probe_overcounting \
  --probabilities artifacts/joint_frame_predictions/human_joint_frame_beta010_es5_v1/chr1train/epoch10_vector_scaled/chr1_plus/joint_frame_logits.npy \
  --input-kind logits \
  --reference-gtf data/raw/human_chess3.1.3.gtf \
  --chrom chr1 \
  --dataset human_chr1 \
  --checkpoint runs/human/human_joint_frame_beta010_es5_v1/chr1train/epoch_checkpoints/epoch_010.pt \
  --output results/belief_probe_v1/E_overcounting/human_chr1 "$@"

exec "$python_bin" scripts/run_belief_probe_E_grid.py
