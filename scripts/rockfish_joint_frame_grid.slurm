#!/bin/bash
#SBATCH --job-name=dmel-joint-frame
#SBATCH --output=dmel-joint-frame-%A_%a.log
#SBATCH --partition=a100
#SBATCH --qos=qos_gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-5%6

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$SLURM_SUBMIT_DIR}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
uniann_root="${UNIANN_ROOT:-$HOME/UniAnn}"
data_root="$repo_dir/data/processed/cds_gating/w10_frame"
config="$repo_dir/configs/droso_window_early_stopping.json"

labels=(b003 b010 b030)
weights=(0.03 0.10 0.30)

if (( SLURM_ARRAY_TASK_ID < 3 )); then
    regime=heldout
    index=$SLURM_ARRAY_TASK_ID
    stages=train,score,auprc
else
    regime=chrxtrain
    index=$((SLURM_ARRAY_TASK_ID - 3))
    stages=train,score,auprc,export,uniann
fi

label="${labels[$index]}"
weight="${weights[$index]}"
tag="joint_frame_grid_v1/$label"

cd "$repo_dir"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -d "$data_root/train" || ! -d "$data_root/val" || ! -d "$data_root/test" ]]; then
    echo "Missing phase/CDS training windows under $data_root." >&2
    echo "Generate them with the command documented in README.md." >&2
    exit 2
fi

"$python_bin" scripts/run_cell.py \
    --config "$config" \
    --experiment-tag "$tag" \
    --cell w10 \
    --regime "$regime" \
    --stages "$stages" \
    --workers "${SLURM_CPUS_PER_TASK:-4}" \
    --data-root "$data_root" \
    --frame-dilation 3 \
    --joint-frame-aux-weight 0.1 \
    --joint-noncoding-weight "$weight" \
    --record-gradient-norms \
    --uniann-root "$uniann_root"

prediction_dir="$repo_dir/artifacts/joint_frame_predictions/$tag/$regime/w10/chrX_plus"
"$python_bin" -m cnn_mamba.score_joint_frame_chromosome \
    --checkpoint "$repo_dir/runs/$tag/$regime/w10/best_model.pt" \
    --fasta "$repo_dir/data/raw/dmel_genome.fa" \
    --chrom NC_004354.4 --display-chrom chrX \
    --output-dir "$prediction_dir" --batch-size 8

"$python_bin" -m cnn_mamba.evaluate_joint_frame \
    --prediction-dir "$prediction_dir" \
    --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
    --fasta "$repo_dir/data/raw/dmel_genome.fa" \
    --chrom NC_004354.4 \
    --output-dir "$repo_dir/results/$tag/$regime/w10/joint_frame"

"$python_bin" scripts/plot_single_run_loss.py \
    --results-json "$repo_dir/runs/$tag/$regime/w10/results.json" \
    --output-dir "$repo_dir/results/$tag/$regime/w10/loss" \
    --prefix "${regime}_train_validation_loss"
