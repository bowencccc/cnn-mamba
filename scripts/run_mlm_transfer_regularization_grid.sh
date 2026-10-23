#!/usr/bin/env bash
# Sequential two-day grid testing regularized fine-tuning after Human MLM.

set -euo pipefail

repo_dir="${CNN_MAMBA_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
python_bin="${CNN_MAMBA_PYTHON:-/home/bowencccc/miniconda3/envs/mamba3_env/bin/python}"
mlm_checkpoint="$repo_dir/runs/mlm_pretrain/human_grch38_w10_v1/best_model.pt"
data_root="$repo_dir/data/processed/cds_gating/w10_frame"
summary_dir="$repo_dir/results/mlm_transfer_regularization_v1/summary"
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"

run_variant() {
    local name="$1"
    local backbone_multiplier="$2"
    local freeze_layers="$3"
    local freeze_epochs="$4"
    local tag="mlm_transfer_regularization_v1/$name"
    local run_dir="$repo_dir/runs/$tag/chrxtrain/w10"
    local score_dir="$repo_dir/artifacts/scores/$tag/chrxtrain/w10"
    local result_dir="$repo_dir/results/$tag/chrxtrain/w10"
    local prediction_dir="$repo_dir/artifacts/joint_frame_predictions/$tag/chrxtrain/w10/chrX_plus"

    echo "[$(date -Is)] starting $name multiplier=$backbone_multiplier freeze=${freeze_layers}x${freeze_epochs}"
    "$python_bin" scripts/run_cell.py \
        --config "$repo_dir/configs/droso_window_early_stopping.json" \
        --experiment-tag "$tag" \
        --cell w10 --regime chrxtrain \
        --stages train,score,auprc,export \
        --workers "${CNN_MAMBA_WORKERS:-4}" \
        --data-root "$data_root" \
        --frame-dilation 3 \
        --joint-frame-aux-weight 0.1 \
        --joint-noncoding-weight 0.1 \
        --candidate-negative-weight 1.0 \
        --init-backbone-checkpoint "$mlm_checkpoint" \
        --backbone-lr-multiplier "$backbone_multiplier" \
        --freeze-pretrained-layers "$freeze_layers" \
        --freeze-pretrained-epochs "$freeze_epochs"

    "$python_bin" -m cnn_mamba.evaluate_annotation_gap \
        --score-dir "$score_dir" \
        --eviann-gff "$repo_dir/data/raw/dmel_eviann_pseudo_label.gff" \
        --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
        --fasta "$repo_dir/data/raw/dmel_genome.fa" \
        --chrom NC_004354.4 --strand + \
        --output "$result_dir/annotation_gap_metrics.tsv"

    "$python_bin" -m cnn_mamba.score_joint_frame_chromosome \
        --checkpoint "$run_dir/best_model.pt" \
        --fasta "$repo_dir/data/raw/dmel_genome.fa" \
        --chrom NC_004354.4 --display-chrom chrX \
        --output-dir "$prediction_dir" --batch-size 8

    "$python_bin" -m cnn_mamba.evaluate_joint_frame \
        --prediction-dir "$prediction_dir" \
        --reference-gtf "$repo_dir/data/raw/dmel_reference.gtf" \
        --fasta "$repo_dir/data/raw/dmel_genome.fa" \
        --chrom NC_004354.4 \
        --output-dir "$result_dir/joint_frame"

    "$python_bin" -m cnn_mamba.export_joint_frame_tsv \
        --probabilities "$prediction_dir/joint_frame_probabilities.npy" \
        --argmax "$prediction_dir/joint_frame_argmax.npy" \
        --output "$prediction_dir/chrX_plus_joint_frame_probabilities.tsv" \
        --chrom chrX --strand +

    "$python_bin" scripts/run_incomplete_label_uniann.py \
        --tag "$tag" --temperature 9

    "$python_bin" scripts/plot_single_run_loss.py \
        --results-json "$run_dir/results.json" \
        --output-dir "$result_dir/loss" \
        --prefix chrxtrain_train_validation_loss

    "$python_bin" scripts/summarize_mlm_transfer_regularization.py \
        --output-dir "$summary_dir"
    echo "[$(date -Is)] completed $name"
}

run_variant backbone_lr010 0.1 0 0
run_variant freeze4_e2_lr010 0.1 4 2
run_variant freeze6_e2_lr010 0.1 6 2

# Choose one local refinement using the primary EviAnn+UniAnn strict locus F1.
read -r refine_multiplier refine_layers refine_epochs < <(
    "$python_bin" - <<'PY'
import csv
from pathlib import Path
root = Path("results/mlm_transfer_regularization_v1")
names = ("backbone_lr010", "freeze4_e2_lr010", "freeze6_e2_lr010")
scores = {}
for name in names:
    path = root / name / "chrxtrain/w10/uniann_joint_t9/strict_locus_metrics.tsv"
    rows = {row["method"]: row for row in csv.DictReader(path.open(), delimiter="\t")}
    scores[name] = float(rows["EviAnn+UniAnn"]["f1"])
best = max(scores, key=scores.get)
if best == "backbone_lr010":
    print("0.03 0 0")
elif best.startswith("freeze4"):
    print("0.1 4 1")
else:
    print("0.1 6 1")
PY
)
run_variant refinement "$refine_multiplier" "$refine_layers" "$refine_epochs"

echo "[$(date -Is)] all MLM transfer regularization experiments completed"
