#!/usr/bin/env bash
# Fit the structured HMM calibration above the explicitly promoted checkpoint.

set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "human" && "$1" != "dmel" ) ]]; then
  echo "usage: $0 human|dmel" >&2
  exit 2
fi
species="$1"
repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
stable="$repo_dir/artifacts/current_best/viterbi/$species"
output="$repo_dir/results/structured_hmm_current_best/$species"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
"$python_bin" -m cnn_mamba.current_best --species "$species"
"$python_bin" scripts/train_structured_hmm_calibration.py \
  --checkpoint "$stable/best_model.pt" --data-root "$stable/windows" \
  --output-dir "$output" --initial-temperature 9 --amp \
  --train-windows "${VITERBI_TRAIN_WINDOWS:-256}" \
  --val-windows "${VITERBI_VAL_WINDOWS:-64}" \
  --crop-length "${VITERBI_CROP_LENGTH:-512}" \
  --epochs "${VITERBI_EPOCHS:-30}" --patience "${VITERBI_PATIENCE:-5}"

echo "current-best $species structured calibration: $output"
