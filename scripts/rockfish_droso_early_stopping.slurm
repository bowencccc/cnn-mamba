#!/bin/bash
#SBATCH --job-name=dmel-cm-es
#SBATCH --output=dmel-cm-es-%A_%a.log
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-11%6

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$SLURM_SUBMIT_DIR}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
uniann_root="${UNIANN_ROOT:-$HOME/UniAnn}"
cells=(w5 w10 w20 w30 w40 w80)

cd "$repo_dir"

if [[ ! -d data/processed/w5/train || ! -d data/processed/w80/test ]]; then
    echo "Missing generated Drosophila windows under data/processed." >&2
    echo "Run the cnn-mamba-prepare-droso command in README first." >&2
    exit 2
fi

if (( SLURM_ARRAY_TASK_ID < 6 )); then
    regime="heldout"
    cell="${cells[$SLURM_ARRAY_TASK_ID]}"
    stages="train,score,auprc"
else
    regime="chrxtrain"
    cell="${cells[$((SLURM_ARRAY_TASK_ID - 6))]}"
    stages="train,score,export,uniann"
fi

"$python_bin" scripts/run_cell.py \
    --config configs/droso_window_early_stopping.json \
    --experiment-tag early_stopping_v1 \
    --cell "$cell" \
    --regime "$regime" \
    --stages "$stages" \
    --workers "${SLURM_CPUS_PER_TASK:-4}" \
    --uniann-root "$uniann_root"
