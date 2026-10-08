#!/bin/bash
#SBATCH --job-name=human-joint-frame
#SBATCH --output=human-joint-frame-%j.log
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$SLURM_SUBMIT_DIR}"
python_bin="${CNN_MAMBA_PYTHON:-python}"

cd "$repo_dir"

if [[ ! -d data/processed/human_joint_frame_w10_chr1held/train ]]; then
    echo "Missing Human four-state windows." >&2
    echo "Generate them with the command documented in README.md." >&2
    exit 2
fi

CNN_MAMBA_REPO="$repo_dir" \
CNN_MAMBA_PYTHON="$python_bin" \
CNN_MAMBA_WORKERS="${SLURM_CPUS_PER_TASK:-4}" \
    bash scripts/run_human_joint_frame_pretrain_local.sh
