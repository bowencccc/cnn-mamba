#!/usr/bin/env bash
# Coleoptera EviAnn w10 chrX-included model -> canonical vector-scaling UniAnn
# evaluation (configs/workflow.droso.example.json steps sample-calibration-data
# .. evaluate-uniann), after the original training/scoring setup of
# scripts/run_coleoptera_eviann_experiment.sh.
#
# Per run: run_cell train,score,auprc,export (original w10 config) ->
# annotation-gap -> calibrate-joint-frame (3L val sample) -> fit-vector-scaling
# -> score-joint-frame with vector calibration -> export-joint-frame (header
# kept) -> latest UniAnn (pulled every run) with its built-in transform ->
# uniann-eval --require-cnn-mamba-tsv-header (UniAnn-only and EviAnn+UniAnn).
#
# env: CNN_MAMBA_PYTHON, CNN_MAMBA_TAG (default coleoptera_eviann_v1/homelab_retrain),
#      UNIANN_LATEST_ROOT (default ~/UniAnn-latest), UNIANN_BRANCH (default origin/main),
#      CNN_MAMBA_AMP_DTYPE (bf16|fp16; unset = original fp32 training)

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
uniann_root="${UNIANN_LATEST_ROOT:-$HOME/UniAnn-latest}"
uniann_branch="${UNIANN_BRANCH:-origin/main}"
tag="${CNN_MAMBA_TAG:-coleoptera_eviann_v1/homelab_retrain}"
config="$repo_dir/configs/droso_coleoptera_w10.json"
source_gff="$repo_dir/data/raw/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pseudo_label.coleoptera.gff.gz"
processed_root="$repo_dir/data/processed/coleoptera_eviann_v1"
data_root="$processed_root/w10"
chrx_eviann="$processed_root/chrX_plus_EviAnn_CDS.gff"
fasta="$repo_dir/data/raw/dmel_genome.fa"
chrx_fasta="$repo_dir/data/raw/dmel_chrX.fa"
reference_gtf="$repo_dir/data/raw/dmel_reference.gtf"
chrx_reference="$repo_dir/data/raw/dmel_chrX_plus_CDS_reference.gtf"
cell=w10
batch_size=8
amp_args=()
[[ -n "${CNN_MAMBA_AMP_DTYPE:-}" ]] && amp_args=(--amp --amp-dtype "$CNN_MAMBA_AMP_DTYPE")

run_dir="runs/$tag/chrxtrain/$cell"
score_dir="artifacts/scores/$tag/chrxtrain/$cell"
result_dir="results/$tag/chrxtrain/$cell"
vector_dir="$result_dir/vector_scaling"
prediction_dir="artifacts/joint_frame_predictions/$tag/chrxtrain/$cell/chrX_plus_vector"
coding_tsv="$prediction_dir/chrX_plus_joint_frame_probabilities.tsv"
uniann_dir="$result_dir/uniann_vector"

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
log() { echo "[$(date -Is)] $*"; }

refresh_uniann() {
    git -C "$uniann_root" fetch -q origin
    git -C "$uniann_root" checkout -q --detach "$uniann_branch"
    local head built=""
    head="$(git -C "$uniann_root" rev-parse HEAD)"
    [[ -f "$uniann_root/bin/.built_commit" ]] && built="$(cat "$uniann_root/bin/.built_commit")"
    if [[ "$head" != "$built" ]]; then
        log "UniAnn updated ${built:0:7} -> ${head:0:7}; rebuilding"
        (cd "$uniann_root" && ./install.sh > "$uniann_root/install.log" 2>&1)
        grep -q "All done" "$uniann_root/install.log" || { echo "UniAnn build failed" >&2; exit 1; }
        echo "$head" > "$uniann_root/bin/.built_commit"
    fi
    uniann_commit="$head"
    log "UniAnn: $uniann_root @ ${uniann_commit:0:7} ($(git -C "$uniann_root" log -1 --format=%s))"
}

# Let a run_cell started by an earlier launcher for this tag finish first.
while pgrep -f "run_cell.py.*--experiment-tag $tag( |$)" > /dev/null; do
    log "waiting for in-flight run_cell.py ($tag)"
    sleep 120
done

if [[ ! -s "$score_dir/chrX_plus_scores.tsv" ]]; then
    stages="train,score,auprc,export"
    [[ -s "$run_dir/results.json" ]] && stages="score,auprc,export"
    log "run_cell stages=$stages"
    "$python_bin" scripts/run_cell.py \
        --config "$config" --experiment-tag "$tag" \
        --cell "$cell" --regime chrxtrain \
        --stages "$stages" --workers "${CNN_MAMBA_WORKERS:-4}" \
        --data-root "$data_root" \
        --frame-dilation 3 --joint-frame-aux-weight 0.1 \
        --joint-noncoding-weight 0.1 --candidate-negative-weight 1.0 \
        "${amp_args[@]}"
fi

if [[ ! -s "$result_dir/annotation_gap_metrics.tsv" ]]; then
    log "annotation gap"
    "$python_bin" -m cnn_mamba.evaluate_annotation_gap \
        --score-dir "$score_dir" --eviann-gff "$source_gff" \
        --reference-gtf "$reference_gtf" --fasta "$fasta" \
        --chrom NC_004354.4 --strand + \
        --output "$result_dir/annotation_gap_metrics.tsv"
fi

log "vector scaling (3L validation sample)"
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
# Header kept: UniAnn counts rows; see docs/UNIANN_FRAME_TSV_CONTRACT.md.
if [[ ! -s "$coding_tsv.complete.json" ]]; then
    "$python_bin" -m cnn_mamba.export_joint_frame_tsv \
        --probabilities "$prediction_dir/joint_frame_probabilities.npy" \
        --argmax "$prediction_dir/joint_frame_argmax.npy" \
        --output "$coding_tsv" --chrom chrX --strand +
fi

refresh_uniann
log "UniAnn (built-in transform on vector-scaled TSV) + strict gffcompare"
"$python_bin" -m cnn_mamba.uniann_evaluate \
    --uniann-root "$uniann_root" --fasta "$chrx_fasta" \
    --coding-frame "$coding_tsv" --require-cnn-mamba-tsv-header \
    --scores "$score_dir/chrX_plus_scores.tsv" \
    --reference-gtf "$chrx_reference" --eviann-gff "$chrx_eviann" \
    --output-dir "$uniann_dir"
echo "$uniann_commit" > "$uniann_dir/uniann_commit.txt"

if [[ -s "$run_dir/results.json" && -f scripts/plot_single_run_loss.py ]]; then
    "$python_bin" scripts/plot_single_run_loss.py \
        --results-json "$run_dir/results.json" \
        --output-dir "$result_dir/loss" \
        --prefix chrxtrain_train_validation_loss || true
fi
log "w10 vector-scaling UniAnn evaluation complete"
