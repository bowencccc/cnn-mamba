#!/usr/bin/env bash
# Seed sweep: Coleoptera EviAnn labels, 1 kb windows, chrX-included (chrxtrain).
# Per seed: train -> chrX+ score/AUPRC/export -> four-state vector scaling fitted
# on the 3L validation split -> latest UniAnn (git pull of origin/main before
# every seed's UniAnn step) with its own built-in coding-frame scaling applied
# to the vector-scaled probabilities -> strict gffcompare, UniAnn-only and
# EviAnn+UniAnn. Finally summarises mean/SD across seeds.
#
# env: CNN_MAMBA_PYTHON, SEEDS (default "42 43 44 45 46"),
#      UNIANN_LATEST_ROOT (git worktree tracking origin/main; default ~/UniAnn-latest),
#      UNIANN_BRANCH (default origin/main),
#      SWEEP_TAG (default coleoptera_w1_seed_sweep_v1),
#      SWEEP_CONFIG (default configs/droso_coleoptera_w1_seeds.json)

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
uniann_root="${UNIANN_LATEST_ROOT:-$HOME/UniAnn-latest}"
uniann_branch="${UNIANN_BRANCH:-origin/main}"
seeds="${SEEDS:-42 43 44 45 46}"
config="$repo_dir/${SWEEP_CONFIG:-configs/droso_coleoptera_w1_seeds.json}"
source_gff="$repo_dir/data/raw/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pseudo_label.coleoptera.gff.gz"
processed_root="$repo_dir/data/processed/coleoptera_eviann_w1_v1"
data_root="$processed_root/w1"
chrx_eviann="$repo_dir/data/processed/coleoptera_eviann_v1/chrX_plus_EviAnn_CDS.gff"
fasta="$repo_dir/data/raw/dmel_genome.fa"
chrx_fasta="$repo_dir/data/raw/dmel_chrX.fa"
reference="$repo_dir/data/raw/dmel_chrX_plus_CDS_reference.gtf"
sweep="${SWEEP_TAG:-coleoptera_w1_seed_sweep_v1}"
cell=w1
batch_size=64

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
mkdir -p "results/$sweep"

log() { echo "[$(date -Is)] $*"; }

# Pull the newest UniAnn and rebuild only when the commit changed.
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

# If an earlier launcher was replaced mid-run, let its training finish first.
while pgrep -f "run_cell.py.*--experiment-tag $sweep/" > /dev/null; do
    log "waiting for an in-flight run_cell.py from a previous launcher"
    sleep 120
done

chrom_len="$(awk '$1 == "NC_004354.4" {print $2}' "$fasta.fai")"
[[ -n "$chrom_len" ]] || { echo "NC_004354.4 missing from $fasta.fai" >&2; exit 1; }

if [[ ! -f "$processed_root/.complete.json" ]]; then
    log "preparing 1 kb windows"
    "$python_bin" -m cnn_mamba.prepare_droso \
        --config "$config" --fasta "$fasta" --eviann-gff "$source_gff" \
        --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
        --output-root "$processed_root" --include-phase-labels
fi
if [[ ! -s "$chrx_eviann" ]]; then
    mkdir -p "$(dirname "$chrx_eviann")"
    "$python_bin" scripts/filter_eviann_cds.py --input "$source_gff" \
        --output "$chrx_eviann" --chrom NC_004354.4 --strand +
fi

for seed in $seeds; do
    tag="$sweep/seed$seed"
    run_dir="runs/$tag/chrxtrain/$cell"
    score_dir="artifacts/scores/$tag/chrxtrain/$cell"
    result_dir="results/$tag/chrxtrain/$cell"
    calibration_dir="$result_dir/calibration"
    vector_dir="$result_dir/vector_scaling"
    prediction_dir="artifacts/joint_frame_predictions/$tag/chrxtrain/$cell/chrX_plus_vector_scaled"
    coding_tsv="$prediction_dir/chrX_plus_joint_frame_probabilities_vector.tsv"
    work_dir="work/uniann/$tag/chrxtrain/$cell/vector_uniann_builtin_header"

    if [[ -s "$work_dir/locus_metrics.tsv" ]]; then
        log "seed $seed already complete; skipping"
        continue
    fi

    stages="train,score,auprc,export"
    [[ -s "$run_dir/results.json" ]] && stages="score,auprc,export"
    log "seed $seed: run_cell stages=$stages"
    "$python_bin" scripts/run_cell.py \
        --config "$config" --experiment-tag "$tag" \
        --cell "$cell" --regime chrxtrain --seed "$seed" \
        --stages "$stages" --workers "${CNN_MAMBA_WORKERS:-4}" \
        --data-root "$data_root" \
        --frame-dilation 3 --joint-frame-aux-weight 0.1 \
        --joint-noncoding-weight 0.1 --candidate-negative-weight 1.0

    log "seed $seed: vector scaling (fit on validation split)"
    if [[ ! -s "$calibration_dir/calibration_sample.npz" ]]; then
        "$python_bin" -m cnn_mamba.calibrate_joint_frame \
            --checkpoint "$run_dir/best_model.pt" --data-root "$data_root" \
            --split val --output-dir "$calibration_dir" \
            --batch-size "$batch_size" --workers 4 \
            --samples-per-window 32 --seed 42 --bins 20 --save-sample
    fi
    if [[ ! -s "$vector_dir/vector_scaling.json" ]]; then
        "$python_bin" -m cnn_mamba.fit_vector_scaling \
            --sample "$calibration_dir/calibration_sample.npz" \
            --metadata-json "$calibration_dir/temperature.json" \
            --output-dir "$vector_dir" --regularization 0.0001
    fi
    if [[ ! -s "$prediction_dir/joint_frame_probabilities.npy" ]]; then
        "$python_bin" -m cnn_mamba.score_joint_frame_chromosome \
            --checkpoint "$run_dir/best_model.pt" --fasta "$fasta" \
            --chrom NC_004354.4 --display-chrom chrX \
            --output-dir "$prediction_dir" --batch-size "$batch_size" \
            --vector-calibration-json "$vector_dir/vector_scaling.json"
    fi
    # Keep the TSV header: UniAnn counts rows, and the header row supplies the
    # +1 phase offset (docs/UNIANN_FRAME_TSV_CONTRACT.md). Never strip it.
    if [[ ! -s "$coding_tsv.complete.json" ]]; then
        "$python_bin" -m cnn_mamba.export_joint_frame_tsv \
            --probabilities "$prediction_dir/joint_frame_probabilities.npy" \
            --argmax "$prediction_dir/joint_frame_argmax.npy" \
            --output "$coding_tsv" --chrom chrX --strand +
    fi

    refresh_uniann
    log "seed $seed: UniAnn (built-in coding-frame scaling) + strict gffcompare"
    "$python_bin" -m cnn_mamba.uniann_evaluate \
        --uniann-root "$uniann_root" --fasta "$chrx_fasta" \
        --coding-frame "$coding_tsv" --require-cnn-mamba-tsv-header \
        --scores "$score_dir/chrX_plus_scores.tsv" \
        --reference-gtf "$reference" --eviann-gff "$chrx_eviann" \
        --output-dir "$work_dir"
    echo "$uniann_commit" > "$work_dir/uniann_commit.txt"
    log "seed $seed complete (UniAnn ${uniann_commit:0:7})"
done

"$python_bin" scripts/summarize_seed_sweep.py --sweep "$sweep" --cell "$cell" \
    --output-dir "results/$sweep"
log "seed sweep complete"
