#!/usr/bin/env bash
# Score Human chr1+ with the validation-fitted temperature and export legacy TSV.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
wait_session="${CNN_MAMBA_WAIT_TMUX_SESSION:-}"
output_dir="$repo_dir/artifacts/joint_frame_predictions/human_joint_frame_beta010_v1/chr1held/chr1_plus_temperature_scaled"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

if [[ -n "$wait_session" ]]; then
    echo "waiting for tmux session: $wait_session"
    while tmux has-session -t "$wait_session" 2>/dev/null; do
        sleep 30
    done
fi

echo "starting calibrated chr1+ four-state scoring"
"$python_bin" -m cnn_mamba.score_joint_frame_chromosome \
    --checkpoint models/human_joint_frame_beta010_chr1held_epoch4_inference.pt \
    --fasta data/raw/human_grch38.fa \
    --chrom NC_000001.11 \
    --display-chrom chr1 \
    --calibration-json results/human_joint_frame_beta010_v1/temperature_scaling/temperature.json \
    --output-dir "$output_dir" \
    --batch-size "${CNN_MAMBA_SCORE_BATCH_SIZE:-8}"

echo "exporting one-row-per-base TSV"
"$python_bin" -m cnn_mamba.export_joint_frame_tsv \
    --probabilities "$output_dir/joint_frame_probabilities.npy" \
    --argmax "$output_dir/joint_frame_argmax.npy" \
    --output "$output_dir/chr1_plus_joint_frame_probabilities.tsv" \
    --chrom chr1 \
    --strand + \
    --chunk-size "${CNN_MAMBA_EXPORT_CHUNK_SIZE:-250000}"

echo "temperature-scaled chr1+ export complete"
