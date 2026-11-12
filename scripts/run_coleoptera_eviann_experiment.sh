#!/usr/bin/env bash
# Train and evaluate the matched chrX-included model using Coleoptera EviAnn
# labels, with the canonical coding-frame recipe of
# configs/workflow.droso.example.json:
#
#   prepare (10 kb, union labels) -> train (bf16 AMP from the w10 config) ->
#   chrX+ score / AUPRC / UniAnn site export -> annotation gap ->
#   calibrate-joint-frame (3L validation sample) -> fit-vector-scaling ->
#   score-joint-frame with the vector calibration -> joint-frame metrics ->
#   export-joint-frame (header kept) -> uniann-eval
#   --require-cnn-mamba-tsv-header (UniAnn's own built-in transform) ->
#   UniAnn-only and EviAnn+UniAnn strict locus metrics.
#
# Joint-softmax T=9 (scripts/run_incomplete_label_uniann.py) is no longer used.
#
# env: CNN_MAMBA_PYTHON, CNN_MAMBA_TAG (default coleoptera_eviann_v1/retrained),
#      UNIANN_ROOT (default ../UniAnn; must support -c, i.e. >= d0f9c65),
#      UNIANN_UPDATE=1 to fetch + check out origin/main and rebuild UniAnn
#      before evaluation (records the commit used).

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
uniann_root="${UNIANN_ROOT:-$repo_dir/../UniAnn}"
source_gff="$repo_dir/data/raw/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pseudo_label.coleoptera.gff.gz"
config="$repo_dir/configs/droso_coleoptera_w10.json"
processed_root="$repo_dir/data/processed/coleoptera_eviann_v1"
data_root="$processed_root/w10"
chrx_gff="$processed_root/chrX_plus_EviAnn_CDS.gff"
fasta="$repo_dir/data/raw/dmel_genome.fa"
reference_gtf="$repo_dir/data/raw/dmel_reference.gtf"
chrx_fasta="$repo_dir/data/raw/dmel_chrX.fa"
chrx_reference="$repo_dir/data/raw/dmel_chrX_plus_CDS_reference.gtf"
tag="${CNN_MAMBA_TAG:-coleoptera_eviann_v1/retrained}"
batch_size=8
run_dir="$repo_dir/runs/$tag/chrxtrain/w10"
score_dir="$repo_dir/artifacts/scores/$tag/chrxtrain/w10"
result_dir="$repo_dir/results/$tag/chrxtrain/w10"
vector_dir="$result_dir/vector_scaling"
prediction_dir="$repo_dir/artifacts/joint_frame_predictions/$tag/chrxtrain/w10/chrX_plus_vector"
coding_tsv="$prediction_dir/chrX_plus_joint_frame_probabilities.tsv"
uniann_dir="$result_dir/uniann_vector"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
log() { echo "[$(date -Is)] $*"; }

if [[ ! -f "$processed_root/.complete.json" ]]; then
    "$python_bin" -m cnn_mamba.prepare_droso \
        --config "$config" --fasta "$fasta" \
        --eviann-gff "$source_gff" --reference-gtf "$reference_gtf" \
        --output-root "$processed_root" --include-phase-labels
fi

if [[ ! -f "$chrx_gff" ]]; then
    "$python_bin" scripts/filter_eviann_cds.py \
        --input "$source_gff" --output "$chrx_gff" \
        --chrom NC_004354.4 --strand +
fi

stages="train,score,auprc,export"
[[ -s "$run_dir/results.json" ]] && stages="score,auprc,export"
"$python_bin" scripts/run_cell.py \
    --config "$config" --experiment-tag "$tag" \
    --cell w10 --regime chrxtrain \
    --stages "$stages" \
    --workers "${CNN_MAMBA_WORKERS:-4}" \
    --data-root "$data_root" \
    --frame-dilation 3 \
    --joint-frame-aux-weight 0.1 \
    --joint-noncoding-weight 0.1 \
    --candidate-negative-weight 1.0

"$python_bin" -m cnn_mamba.evaluate_annotation_gap \
    --score-dir "$score_dir" --eviann-gff "$source_gff" \
    --reference-gtf "$reference_gtf" --fasta "$fasta" \
    --chrom NC_004354.4 --strand + \
    --output "$result_dir/annotation_gap_metrics.tsv"

log "vector scaling of the four-state head (3L validation sample)"
if [[ ! -s "$vector_dir/calibration_sample.npz" ]]; then
    "$python_bin" -m cnn_mamba.calibrate_joint_frame \
        --checkpoint "$run_dir/best_model.pt" --data-root "$data_root" \
        --split val --output-dir "$vector_dir" \
        --batch-size "$batch_size" --workers 4 \
        --samples-per-window 32 --seed 42 --bins 20 --save-sample
fi
if [[ ! -s "$vector_dir/vector_scaling.json" ]]; then
    "$python_bin" -m cnn_mamba.fit_vector_scaling \
        --sample "$vector_dir/calibration_sample.npz" \
        --metadata-json "$vector_dir/temperature.json" \
        --output-dir "$vector_dir" --regularization 0.0001
fi
if [[ ! -s "$prediction_dir/joint_frame_probabilities.npy" ]]; then
    "$python_bin" -m cnn_mamba.score_joint_frame_chromosome \
        --checkpoint "$run_dir/best_model.pt" --fasta "$fasta" \
        --chrom NC_004354.4 --display-chrom chrX \
        --vector-calibration-json "$vector_dir/vector_scaling.json" \
        --output-dir "$prediction_dir" --batch-size "$batch_size"
fi

"$python_bin" -m cnn_mamba.evaluate_joint_frame \
    --prediction-dir "$prediction_dir" \
    --reference-gtf "$reference_gtf" --fasta "$fasta" \
    --chrom NC_004354.4 \
    --output-dir "$result_dir/joint_frame_vector"

# Header kept: UniAnn counts rows; see docs/UNIANN_FRAME_TSV_CONTRACT.md.
if [[ ! -s "$coding_tsv.complete.json" ]]; then
    "$python_bin" -m cnn_mamba.export_joint_frame_tsv \
        --probabilities "$prediction_dir/joint_frame_probabilities.npy" \
        --argmax "$prediction_dir/joint_frame_argmax.npy" \
        --output "$coding_tsv" --chrom chrX --strand +
fi

if [[ "${UNIANN_UPDATE:-0}" == 1 ]]; then
    git -C "$uniann_root" fetch -q origin
    git -C "$uniann_root" checkout -q --detach origin/main
    head="$(git -C "$uniann_root" rev-parse HEAD)"
    built="$(cat "$uniann_root/bin/.built_commit" 2>/dev/null || true)"
    if [[ "$head" != "$built" ]]; then
        log "rebuilding UniAnn at ${head:0:7}"
        (cd "$uniann_root" && ./install.sh > "$uniann_root/install.log" 2>&1)
        grep -q "All done" "$uniann_root/install.log" || { echo "UniAnn build failed" >&2; exit 1; }
        echo "$head" > "$uniann_root/bin/.built_commit"
    fi
fi
log "UniAnn $(git -C "$uniann_root" rev-parse --short HEAD 2>/dev/null || echo unknown) on the vector-scaled TSV"
"$python_bin" -m cnn_mamba.uniann_evaluate \
    --uniann-root "$uniann_root" --fasta "$chrx_fasta" \
    --coding-frame "$coding_tsv" --require-cnn-mamba-tsv-header \
    --scores "$score_dir/chrX_plus_scores.tsv" \
    --reference-gtf "$chrx_reference" --eviann-gff "$chrx_gff" \
    --output-dir "$uniann_dir"
git -C "$uniann_root" rev-parse HEAD > "$uniann_dir/uniann_commit.txt" 2>/dev/null || true

"$python_bin" scripts/plot_single_run_loss.py \
    --results-json "$run_dir/results.json" \
    --output-dir "$result_dir/loss" \
    --prefix chrxtrain_train_validation_loss

log "Coleoptera EviAnn experiment complete"
