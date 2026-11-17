#!/usr/bin/env bash
# Train the matched Human chr1-included four-state model using Rodentia EviAnn labels.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
source_gff="$repo_dir/data/raw/GCF_000001405.40_GRCh38.p14_genomic.noalpsnopatches.fna.pseudo_label.rodentia.gff.gz"
data_root="$repo_dir/data/processed/human_rodentia_joint_frame_w10_refseq_chr1held"
tag="${CNN_MAMBA_TAG:-human_rodentia_joint_frame_beta010_es5_v1}"
log_dir="$repo_dir/results/$tag"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/launcher.log") 2>&1

if [[ ! -f "$data_root/split_manifest.json" ]]; then
    "$python_bin" -m cnn_mamba.prepare_human \
        --fasta "$repo_dir/data/raw/human_grch38.fa" \
        --eviann-gff "$source_gff" \
        --reference-annotation "$repo_dir/data/raw/human_refseq_grch38_p14.gff.gz" \
        --output-dir "$data_root" \
        --max-windows-per-chrom-strand 0 \
        --region-overlap-bp 10000 \
        --include-phase-labels \
        --label-policy all_eviann
fi

"$python_bin" scripts/run_human.py \
    --regime chr1train \
    --stages train,score,auprc \
    --experiment-tag "$tag" \
    --data-root "$data_root" \
    --epochs 30 \
    --scheduler-epochs 5 \
    --workers "${CNN_MAMBA_WORKERS:-4}" \
    --frame-dilation 3 \
    --joint-frame-aux-weight 0.1 \
    --joint-noncoding-weight 0.1 \
    --early-stopping-patience 3 \
    --early-stopping-min-delta 0.0001 \
    --record-gradient-norms

echo "[$(date -Is)] Human Rodentia-EviAnn chr1-included experiment complete"
