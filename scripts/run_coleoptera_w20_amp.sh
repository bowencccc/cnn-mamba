#!/usr/bin/env bash
# Coleoptera EviAnn labels, 20 kb windows, chrX-included, CUDA AMP enabled.
# Same recipe as run_coleoptera_eviann_experiment.sh except window size and AMP.
# CNN_MAMBA_AMP_DTYPE=bf16|fp16 (default bf16; fp16 overflowed at epoch 5 in v1).

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
config="$repo_dir/configs/droso_coleoptera_w20_amp.json"
source_gff="$repo_dir/data/raw/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pseudo_label.coleoptera.gff.gz"
processed_root="$repo_dir/data/processed/coleoptera_eviann_w20_amp_v1"
amp_dtype="${CNN_MAMBA_AMP_DTYPE:-bf16}"
tag="${CNN_MAMBA_TAG:-coleoptera_eviann_w20_amp_${amp_dtype}_v2}"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

if [[ ! -f "$processed_root/.complete.json" ]]; then
    "$python_bin" -m cnn_mamba.prepare_droso \
        --config "$config" \
        --fasta "$repo_dir/data/raw/dmel_genome.fa" \
        --eviann-gff "$source_gff" \
        --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
        --output-root "$processed_root" \
        --include-phase-labels
fi

"$python_bin" scripts/run_cell.py \
    --config "$config" \
    --experiment-tag "$tag" \
    --cell w20 --regime chrxtrain \
    --stages "${CNN_MAMBA_STAGES:-train,score,auprc}" \
    --workers "${CNN_MAMBA_WORKERS:-4}" \
    --data-root "$processed_root/w20" \
    --frame-dilation 3 \
    --joint-frame-aux-weight 0.1 \
    --joint-noncoding-weight 0.1 \
    --candidate-negative-weight 1.0 \
    --amp --amp-dtype "$amp_dtype"

echo "[$(date -Is)] Coleoptera w20 AMP ($amp_dtype) experiment complete: $tag"
