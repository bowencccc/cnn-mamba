#!/bin/bash
# Run the complete early-stopping Drosophila grid sequentially on one GPU.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"
cells=(w5 w10 w20 w30 w40 w80)

cd "$repo_dir"

for regime in heldout chrxtrain; do
    if [[ "$regime" == "heldout" ]]; then
        stages="train,score,auprc"
    else
        stages="train,score,export,uniann"
    fi
    for cell in "${cells[@]}"; do
        "$python_bin" scripts/run_cell.py \
            --config configs/droso_window_early_stopping.json \
            --experiment-tag early_stopping_v1 \
            --cell "$cell" \
            --regime "$regime" \
            --stages "$stages" \
            --workers "${CNN_MAMBA_WORKERS:-4}" \
            --uniann-root "$uniann_root"
    done
done
