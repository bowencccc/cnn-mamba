#!/bin/bash
# Four-state non-CDS/frame-0/frame-1/frame-2 auxiliary-head grid.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
data_root="$repo_dir/data/processed/cds_gating/w10_frame"
config="$repo_dir/configs/droso_window_early_stopping.json"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

for spec in "b003 0.03" "b010 0.10" "b030 0.30"; do
    read -r label beta <<< "$spec"
    tag="joint_frame_grid_v1/$label"
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
            --joint-frame-aux-weight 0.1 \
            --joint-noncoding-weight "$beta" \
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
    done
done

"$python_bin" scripts/summarize_joint_frame_grid.py
