#!/usr/bin/env python3
"""Cross-fit LightGBM models for exact RefSeq agreement of EviAnn boundaries."""

import argparse
import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np

from cnn_mamba.human_annotations import EVIANN_GFF, FASTA, REFSEQ_GFF
from cnn_mamba.human_site_quality import (
    CHROM_FOLDS, average_precision, build_site_quality_examples,
    calibration_bins, cross_fitted_threshold_metrics, threshold_for_precision,
)


def write_examples(path, examples, names, scores):
    fields = [
        "task", "chrom", "strand", "position_0based", "fold", "label",
        "exact_refseq_match", "comparable_refseq_locus", "oof_score",
        "support_transcript_ids",
    ] + names
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row, score in zip(examples, scores):
            output = {key: row[key] for key in fields[:8]}
            output["oof_score"] = float(score)
            output["support_transcript_ids"] = row["support_transcript_ids"]
            output.update(zip(names, row["features"].tolist()))
            writer.writerow(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eviann-gff", type=Path, default=EVIANN_GFF)
    parser.add_argument("--refseq", type=Path, default=REFSEQ_GFF)
    parser.add_argument("--fasta", type=Path, default=FASTA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))

    print("Building leakage-safe site features and labels...", flush=True)
    by_task, names = build_site_quality_examples(
        args.eviann_gff, args.refseq, args.fasta
    )
    import lightgbm as lgb

    summary = {
        "definition": {
            "unit": "unique EviAnn start/stop site",
            "positive": (
                "exact chromosome/strand/type/coordinate match to any RefSeq "
                "protein-coding transcript"
            ),
            "negative": (
                "no exact match, but at least one supporting EviAnn CDS span "
                "has >=50% overlap of the shorter span with a same-strand "
                "RefSeq coding transcript"
            ),
            "unlabeled": "no conservatively comparable RefSeq coding locus",
            "feature_source": "EviAnn annotation, ORF QC, support, and DNA only",
        },
        "chromosome_folds": [list(values) for values in CHROM_FOLDS],
        "features": names,
        "tasks": {},
    }
    for task, examples in by_task.items():
        print(f"Training {task}: {len(examples):,} unique sites", flush=True)
        matrix = np.stack([row["features"] for row in examples])
        labels = np.asarray([row["label"] for row in examples], dtype=np.int8)
        folds = np.asarray([row["fold"] for row in examples], dtype=np.int8)
        scores = np.full(len(examples), np.nan, dtype=np.float64)
        importance = np.zeros(len(names), dtype=np.float64)
        fold_rows = []
        model_dir = args.output_dir / f"{task}_models"
        model_dir.mkdir(exist_ok=True)
        for fold, held_chroms in enumerate(CHROM_FOLDS):
            train_mask = (labels >= 0) & (folds != fold)
            held_mask = folds == fold
            held_labeled = held_mask & (labels >= 0)
            dataset = lgb.Dataset(
                matrix[train_mask], label=labels[train_mask],
                feature_name=names, free_raw_data=False,
            )
            booster = lgb.train(
                {
                    "objective": "binary",
                    "metric": "binary_logloss",
                    "learning_rate": 0.03,
                    "num_leaves": 31,
                    "min_data_in_leaf": 50,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                    "lambda_l2": 1.0,
                    "verbosity": -1,
                    "seed": 42,
                    "feature_fraction_seed": 42,
                    "bagging_seed": 42,
                    "num_threads": args.threads,
                },
                dataset,
                num_boost_round=args.num_boost_round,
            )
            scores[held_mask] = booster.predict(matrix[held_mask])
            importance += booster.feature_importance(importance_type="gain")
            booster.save_model(str(model_dir / f"fold_{fold}.txt"))
            fold_rows.append({
                "fold": fold,
                "chromosomes": list(held_chroms),
                "train_labeled": int(train_mask.sum()),
                "held_labeled": int(held_labeled.sum()),
                "held_positive": int(labels[held_labeled].sum()),
                "held_AP": average_precision(
                    labels[held_labeled], scores[held_labeled]
                ),
            })
            print(
                f"  fold {fold} {','.join(held_chroms)} "
                f"AP={fold_rows[-1]['held_AP']:.4f}", flush=True
            )
        if np.isnan(scores).any():
            raise RuntimeError(f"missing out-of-fold scores for {task}")
        labeled = labels >= 0
        task_summary = {
            "sites": len(examples),
            "labeled": int(labeled.sum()),
            "positive": int(labels[labeled].sum()),
            "negative": int(labeled.sum() - labels[labeled].sum()),
            "unlabeled": int((labels < 0).sum()),
            "positive_prevalence": float(labels[labeled].mean()),
            "oof_average_precision": average_precision(
                labels[labeled], scores[labeled]
            ),
            "baseline_complete_non_low_quality_AP": average_precision(
                labels[labeled], matrix[labeled, names.index(
                    "any_complete_non_low_quality"
                )],
            ),
            "folds": fold_rows,
            "precision_operating_points": [
                threshold_for_precision(labels[labeled], scores[labeled], target)
                for target in (0.80, 0.85, 0.90, 0.95)
            ],
            "cross_fitted_precision_operating_points": [
                cross_fitted_threshold_metrics(
                    labels[labeled], scores[labeled], folds[labeled], target
                )
                for target in (0.80, 0.85, 0.90, 0.95)
            ],
            "calibration_bins": calibration_bins(
                labels[labeled], scores[labeled], bins=10
            ),
            "feature_importance_gain": [
                {"feature": name, "gain": float(gain)}
                for name, gain in sorted(
                    zip(names, importance), key=lambda item: item[1], reverse=True
                )
            ],
        }
        summary["tasks"][task] = task_summary
        write_examples(
            args.output_dir / f"{task}_site_scores.tsv.gz",
            examples, names, scores,
        )
        print(
            f"{task} OOF AP={task_summary['oof_average_precision']:.4f}; "
            f"metadata baseline={task_summary['baseline_complete_non_low_quality_AP']:.4f}",
            flush=True,
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(f"Wrote {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
