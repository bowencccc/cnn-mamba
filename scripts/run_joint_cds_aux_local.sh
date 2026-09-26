#!/bin/bash
# Full controlled phase=0.10 + CDS=0.10 experiment on one GPU.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
data_root="$repo_dir/data/processed/cds_gating/w10_frame"
config="$repo_dir/configs/droso_window_early_stopping.json"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"
tag="joint_cds_aux_v1/lambda_010"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

for regime in heldout chrxtrain; do
    if [[ "$regime" == "heldout" ]]; then
        stages="train,score,auprc"
    else
        stages="train,score,auprc,export,uniann"
    fi
    "$python_bin" scripts/run_cell.py \
        --config "$config" \
        --experiment-tag "$tag" \
        --cell w10 \
        --regime "$regime" \
        --stages "$stages" \
        --workers "${CNN_MAMBA_WORKERS:-4}" \
        --data-root "$data_root" \
        --frame-dilation 3 \
        --phase-aux-weight 0.1 \
        --cds-aux-weight 0.1 \
        --uniann-root "$uniann_root"

    phase_dir="$repo_dir/artifacts/phase_predictions/$tag/$regime/w10/chrX_plus"
    "$python_bin" -m cnn_mamba.score_gated_phase_chromosome \
        --checkpoint "$repo_dir/runs/$tag/$regime/w10/best_model.pt" \
        --fasta "$repo_dir/data/raw/dmel_genome.fa" \
        --chrom NC_004354.4 --display-chrom chrX \
        --output-dir "$phase_dir" --batch-size 8 --write-raw-phase
    "$python_bin" -m cnn_mamba.evaluate_gated_phase \
        --raw-phase "$phase_dir/raw_phase_probabilities.npy" \
        --gated-dir "$phase_dir" \
        --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
        --fasta "$repo_dir/data/raw/dmel_genome.fa" \
        --chrom NC_004354.4 \
        --output-dir "$repo_dir/results/$tag/$regime/w10/phase"

    "$python_bin" scripts/plot_single_run_loss.py \
        --results-json "$repo_dir/runs/$tag/$regime/w10/results.json" \
        --output-dir "$repo_dir/results/$tag/$regime/w10/loss" \
        --prefix "${regime}_train_validation_loss"
done

"$python_bin" scripts/summarize_joint_cds_aux.py
