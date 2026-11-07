#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
uniann_root="/home/bowencccc/UniAnn"
fasta="$root/data/raw/dmel_chrX.fa"
reference="$root/data/raw/dmel_chrX_plus_CDS_reference.gtf"
eviann="$root/data/raw/dmel_chrX_plus_EviAnn_CDS.gff"
emission_root="$root/artifacts/uniann_emissions/droso_small_window_v1_four_state_raw_t1"
work_root="$root/work/uniann/droso_small_window_v1_four_state_raw_t1"

for key in w0p5 w1 w2 w5 w10; do
    if [[ "$key" == "w10" ]]; then
        probability_dir="$root/artifacts/joint_frame_predictions/joint_frame_grid_v1/b010/chrxtrain/w10/chrX_plus"
        score_file="$root/artifacts/scores/joint_frame_grid_v1/b010/chrxtrain/w10/chrX_plus_scores.tsv"
        calibration_args=(--expected-calibration-method raw --output-temperature 1)
    else
        probability_dir="$root/artifacts/joint_frame_predictions/droso_small_window_v1_four_state_t9/chrxtrain/$key/chrX_plus"
        score_file="$root/artifacts/scores/droso_small_window_v1/chrxtrain/$key/chrX_plus_scores.tsv"
        calibration_args=(--expected-calibration-method scalar_temperature --expected-temperature 9 --output-temperature 1)
    fi

    emission_dir="$emission_root/$key"
    emission_file="$emission_dir/out.ps.txt"
    manifest="$emission_dir/out.ps.txt.complete.json"
    mkdir -p "$emission_dir"
    if [[ ! -s "$manifest" ]]; then
        "$python_bin" "$root/scripts/stream_joint_frame_no_gate_emissions.py" convert \
            --index "$probability_dir/joint_frame_probabilities.npy" \
            --start0 0 \
            --end0 23542271 \
            --out-ps "$emission_file" \
            --out-fasta "$fasta" \
            --seqid NC_004354.4 \
            --manifest "$manifest" \
            "${calibration_args[@]}"
    fi

    if [[ ! -s "$work_root/$key/locus_metrics.tsv" ]]; then
        "$python_bin" -m cnn_mamba.uniann_evaluate \
            --uniann-root "$uniann_root" \
            --fasta "$fasta" \
            --coding-frame "$probability_dir/joint_frame_probabilities.npy" \
            --precomputed-emissions "$emission_file" \
            --scores "$score_file" \
            --reference-gtf "$reference" \
            --eviann-gff "$eviann" \
            --output-dir "$work_root/$key"
    fi
done
