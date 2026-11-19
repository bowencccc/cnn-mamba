#!/usr/bin/env bash
# Validation-only 3x3 AdamW weight-decay sweep. The best checkpoint from each
# run is retained; downstream chromosome scoring and UniAnn are intentionally
# deferred until the validation mean/variance identifies the useful settings.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
config="$repo_dir/configs/droso_coleoptera_w1_weight_decay.json"
source_gff="$repo_dir/data/raw/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pseudo_label.coleoptera.gff.gz"
processed_root="$repo_dir/data/processed/coleoptera_eviann_w1_v1"
data_root="$processed_root/w1"
fasta="$repo_dir/data/raw/dmel_genome.fa"
reference="$repo_dir/data/raw/dmel_reference.gtf"
seeds="${SEEDS:-42 43 44}"
weight_decays="${WEIGHT_DECAYS:-1e-4 1e-3 1e-2}"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
mkdir -p results/coleoptera_w1_weight_decay_v1

if [[ ! -f "$processed_root/.complete.json" ]]; then
    "$python_bin" -m cnn_mamba.prepare_droso \
        --config "$config" --fasta "$fasta" --eviann-gff "$source_gff" \
        --reference-gtf "$reference" --output-root "$processed_root" \
        --include-phase-labels
fi

for wd in $weight_decays; do
    tag_wd="$(printf '%s' "$wd" | tr '.-' 'pm')"
    for seed in $seeds; do
        tag="coleoptera_w1_weight_decay_v1/wd${tag_wd}/seed${seed}"
        run_dir="$repo_dir/runs/$tag/chrxtrain/w1"
        if [[ -s "$run_dir/results.json" ]]; then
            echo "[$(date -Is)] $tag already complete"
            continue
        fi
        echo "[$(date -Is)] training $tag"
        "$python_bin" scripts/run_cell.py \
            --config "$config" --experiment-tag "$tag" \
            --cell w1 --regime chrxtrain --seed "$seed" \
            --weight-decay "$wd" --stages train --workers 4 \
            --data-root "$data_root" --frame-dilation 3 \
            --joint-frame-aux-weight 0.1 --joint-noncoding-weight 0.1 \
            --candidate-negative-weight 1.0
    done
done

"$python_bin" "$repo_dir/scripts/summarize_weight_decay_sweep.py" \
    --run-root "$repo_dir/runs/coleoptera_w1_weight_decay_v1" \
    --output "$repo_dir/results/coleoptera_w1_weight_decay_v1/validation_summary.tsv"
