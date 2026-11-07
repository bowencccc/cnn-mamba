#!/usr/bin/env bash
set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
config="$repo_dir/configs/droso_small_window_v1.json"
data_root="$repo_dir/data/processed/droso_small_window_v1"
tag="droso_small_window_v1"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"
log_dir="$repo_dir/results/$tag"

mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/launcher.log") 2>&1
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

echo "[$(date --iso-8601=seconds)] preparing 0.5/1/2/5-kb datasets"
"$python_bin" -m cnn_mamba.prepare_droso \
  --config "$config" \
  --output-root "$data_root" \
  --include-phase-labels

for cell in w0p5 w1 w2 w5; do
  echo "[$(date --iso-8601=seconds)] starting $cell chrX-included pipeline"
  stages="train,score,auprc,export,uniann"
  if [[ -s "$repo_dir/runs/$tag/chrxtrain/$cell/results.json" ]]; then
    echo "[$(date --iso-8601=seconds)] $cell training already complete; reusing best_model.pt"
    stages="score,auprc,export,uniann"
  fi
  "$python_bin" scripts/run_cell.py \
    --config "$config" \
    --experiment-tag "$tag" \
    --cell "$cell" \
    --regime chrxtrain \
    --stages "$stages" \
    --workers "${CNN_MAMBA_WORKERS:-4}" \
    --data-root "$data_root/$cell" \
    --frame-dilation 3 \
    --joint-frame-aux-weight 0.1 \
    --joint-noncoding-weight 0.1 \
    --record-gradient-norms \
    --uniann-root "$uniann_root"
done

echo "[$(date --iso-8601=seconds)] all small-window pipelines complete"
