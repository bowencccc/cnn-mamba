#!/usr/bin/env bash
# Post-hoc Gap-locus evaluation of saved epoch checkpoints on a held-out
# validation chromosome (3L, plus strand) and on the deployment chromosome
# (chrX, plus strand). No retraining. Answers: does selecting the epoch by
# 3L Gap F1 pick a better chrX checkpoint than selecting by 3L mean AP?
#
# Per (seed, epoch): vector scaling fitted on the 3L validation windows (as in
# the seed sweep), then for each chromosome: site scores + candidate AUPRC +
# UniAnn score export, joint-frame scoring, frame TSV (header kept), then
# UniAnn + strict gffcompare (+ gap_locus_metrics.tsv) in the background.
#
# 3L inputs: data/processed/dmel_3l_plus_eval_v1 (RefSeq CDS reference built
# like scripts/make_dmel_chrx_uniann_reference.sh, EviAnn CDS via
# scripts/filter_eviann_cds.py, NT_037436.4 FASTA).
#
# env: CNN_MAMBA_PYTHON, SOURCE_SWEEP (runs/<sweep> with epoch_checkpoints),
#      OUTPUT_TAG, SEEDS, EPOCHS, UNIANN_ROOT (already built; not refreshed here),
#      PROCESSED_ROOT (training windows; val split used for vector scaling),
#      UNIANN_JOBS (max concurrent UniAnn runs, default 6)

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-python}"
uniann_root="${UNIANN_ROOT:-$HOME/UniAnn-latest}"
source_sweep="${SOURCE_SWEEP:-coleoptera_w1_seed_sweep_fixed15_perisoform_v1}"
out_tag="${OUTPUT_TAG:-coleoptera_w1_gap_selection_perisoform_v1}"
seeds="${SEEDS:-42 43 44 45 46}"
epochs="${EPOCHS:-3 4 5 6 7 8 9 10}"
uniann_jobs="${UNIANN_JOBS:-6}"
data_root="$repo_dir/${PROCESSED_ROOT:-data/processed/coleoptera_eviann_w1_perisoform_v1}/w1"
fasta="$repo_dir/data/raw/dmel_genome.fa"
eval3l="$repo_dir/data/processed/dmel_3l_plus_eval_v1"
batch_size=64

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

log() { echo "[$(date -Is)] $*"; }
uniann_commit="$(git -C "$uniann_root" rev-parse HEAD)"

# name accession display_chrom uniann_chrom fasta reference eviann
chromosomes=(
    "chrX NC_004354.4 chrX X data/raw/dmel_chrX.fa data/raw/dmel_chrX_plus_CDS_reference.gtf data/processed/coleoptera_eviann_v1/chrX_plus_EviAnn_CDS.gff"
    "chr3L NT_037436.4 chr3L 3L $eval3l/dmel_3L.fa $eval3l/dmel_3L_plus_CDS_reference.gtf $eval3l/3L_plus_EviAnn_CDS.gff"
)

throttle() { while (( $(jobs -rp | wc -l) >= uniann_jobs )); do wait -n || true; done; }

for seed in $seeds; do
    for epoch in $epochs; do
        ep="$(printf '%03d' "$epoch")"
        checkpoint="runs/$source_sweep/seed$seed/chrxtrain/w1/epoch_checkpoints/epoch_$ep.pt"
        [[ -s "$checkpoint" ]] || { log "missing $checkpoint"; exit 1; }
        tag="$out_tag/seed$seed/epoch$ep"
        result_dir="results/$tag"
        calibration_dir="artifacts/joint_frame_predictions/$tag/calibration"  # ZFS; samples are ~15 MB
        vector_dir="$result_dir/vector_scaling"
        mkdir -p "$result_dir"

        if [[ ! -s "$vector_dir/vector_scaling.json" ]]; then
            log "seed $seed epoch $epoch: vector scaling (3L validation windows)"
            "$python_bin" -m cnn_mamba.calibrate_joint_frame \
                --checkpoint "$checkpoint" --data-root "$data_root" --split val \
                --output-dir "$calibration_dir" --batch-size "$batch_size" --workers 4 \
                --samples-per-window 32 --seed 42 --bins 20 --save-sample > "$result_dir/calibrate.log" 2>&1
            "$python_bin" -m cnn_mamba.fit_vector_scaling \
                --sample "$calibration_dir/calibration_sample.npz" \
                --metadata-json "$calibration_dir/temperature.json" \
                --output-dir "$vector_dir" --regularization 0.0001 > "$result_dir/vector.log" 2>&1
        fi

        for spec in "${chromosomes[@]}"; do
            read -r name acc display uchrom cfasta reference eviann <<< "$spec"
            score_dir="artifacts/scores/$tag/$name"
            prediction_dir="artifacts/joint_frame_predictions/$tag/${name}_plus_vector_scaled"
            coding_tsv="$prediction_dir/${name}_plus_joint_frame_probabilities_vector.tsv"
            work_dir="work/uniann/$tag/$name/vector_uniann_builtin_header"
            [[ -s "$work_dir/gap_locus_metrics.tsv" ]] && continue

            if [[ ! -s "$score_dir/${name}_plus_scores.tsv" ]]; then
                log "seed $seed epoch $epoch $name: site scores"
                "$python_bin" -m cnn_mamba.score_chromosome --checkpoint "$checkpoint" \
                    --output-dir "$score_dir" --fasta "$fasta" --chrom "$acc" \
                    --display-chrom "$display" --batch-size "$batch_size" > "$result_dir/score_$name.log" 2>&1
                "$python_bin" -m cnn_mamba.evaluate_candidates --score-dir "$score_dir" \
                    --reference-gtf data/raw/dmel_reference.gtf --fasta "$fasta" --chrom "$acc" \
                    --output "$score_dir/candidate_auprc.tsv" >> "$result_dir/score_$name.log" 2>&1
                "$python_bin" -m cnn_mamba.export_uniann_scores --score-dir "$score_dir" \
                    --output "$score_dir/${name}_plus_scores.tsv" --chrom "$uchrom" \
                    --fasta "$fasta" --accession "$acc" >> "$result_dir/score_$name.log" 2>&1
            fi
            if [[ ! -s "$prediction_dir/joint_frame_probabilities.npy" ]]; then
                log "seed $seed epoch $epoch $name: joint-frame scores"
                "$python_bin" -m cnn_mamba.score_joint_frame_chromosome --checkpoint "$checkpoint" \
                    --fasta "$fasta" --chrom "$acc" --display-chrom "$display" \
                    --output-dir "$prediction_dir" --batch-size "$batch_size" \
                    --vector-calibration-json "$vector_dir/vector_scaling.json" > "$result_dir/frame_$name.log" 2>&1
            fi
            # Keep the TSV header: UniAnn counts rows, and the header row supplies the
            # +1 phase offset (docs/UNIANN_FRAME_TSV_CONTRACT.md). Never strip it.
            if [[ ! -s "$coding_tsv.complete.json" ]]; then
                "$python_bin" -m cnn_mamba.export_joint_frame_tsv \
                    --probabilities "$prediction_dir/joint_frame_probabilities.npy" \
                    --argmax "$prediction_dir/joint_frame_argmax.npy" \
                    --output "$coding_tsv" --chrom "$display" --strand + >> "$result_dir/frame_$name.log" 2>&1
            fi

            throttle
            log "seed $seed epoch $epoch $name: UniAnn (background)"
            (
                "$python_bin" -m cnn_mamba.uniann_evaluate --uniann-root "$uniann_root" \
                    --fasta "$cfasta" --coding-frame "$coding_tsv" --require-cnn-mamba-tsv-header \
                    --scores "$score_dir/${name}_plus_scores.tsv" --reference-gtf "$reference" \
                    --eviann-gff "$eviann" --output-dir "$work_dir" > "$result_dir/uniann_$name.log" 2>&1 \
                    && echo "$uniann_commit" > "$work_dir/uniann_commit.txt" \
                    && log "seed $seed epoch $epoch $name: UniAnn done" \
                    || log "seed $seed epoch $epoch $name: UniAnn FAILED (see $result_dir/uniann_$name.log)"
            ) &
        done
    done
done
wait
log "gap selection evaluation complete"
