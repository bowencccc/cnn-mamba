#!/usr/bin/env python3
"""Zero-shot evaluation of Human EviAnn-quality LightGBMs on Drosophila."""

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from cnn_mamba.dmel_site_quality import build_dmel_site_quality_examples
from cnn_mamba.human_site_quality import average_precision, threshold_for_precision


ROOT = Path(__file__).resolve().parents[1]


def write_scores(path, examples, scores):
    fields = [
        "task", "chrom", "strand", "position_0based", "label",
        "exact_refseq_match", "comparable_refseq_locus", "human_lgbm_score",
        "support_transcript_ids",
    ]
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row, score in zip(examples, scores):
            writer.writerow({
                **{key: row[key] for key in fields if key in row},
                "human_lgbm_score": float(score),
            })


def metrics_at_threshold(labels, scores, threshold):
    selected = scores >= threshold
    tp = int(labels[selected].sum())
    count = int(selected.sum())
    positives = int(labels.sum())
    return {
        "threshold": float(threshold), "selected": count,
        "true_positive": tp,
        "precision": float(tp / count) if count else None,
        "recall": float(tp / positives) if positives else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eviann", type=Path, default=ROOT / "data/raw/dmel_eviann_pseudo_label.gff")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/raw/dmel_reference.gtf")
    parser.add_argument("--fasta", type=Path, default=ROOT / "data/raw/dmel_genome.fa")
    parser.add_argument("--human-model-dir", type=Path, default=ROOT / "results/human_site_quality_lightgbm_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/human_lightgbm_on_dmel_v1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import lightgbm as lgb
    examples_by_task, names = build_dmel_site_quality_examples(
        args.eviann, args.reference, args.fasta
    )
    human_summary = json.loads((args.human_model_dir / "summary_corrected.json").read_text())
    output = {
        "experiment": "zero-shot Human LightGBM ensemble applied to D. melanogaster",
        "models_retrained_on_drosophila": False,
        "features": len(names), "tasks": {},
    }
    for task, examples in examples_by_task.items():
        matrix = np.stack([row["features"] for row in examples])
        models = [
            lgb.Booster(model_file=str(path))
            for path in sorted((args.human_model_dir / f"{task}_models").glob("fold_*.txt"))
        ]
        if not models:
            raise RuntimeError(f"no Human {task} models found")
        scores = np.mean([model.predict(matrix) for model in models], axis=0)
        labels = np.asarray([row["label"] for row in examples], dtype=np.int8)
        labeled = labels >= 0
        y, s = labels[labeled], scores[labeled]
        baseline_feature = names.index("any_complete_non_low_quality")
        baseline = matrix[labeled, baseline_feature]
        human_90 = next(
            row for row in human_summary["tasks"][task]["precision_operating_points"]
            if row["target_precision"] == 0.9
        )["threshold"]
        output["tasks"][task] = {
            "sites": len(examples), "labeled": int(labeled.sum()),
            "positive": int(y.sum()), "negative": int(len(y) - y.sum()),
            "unlabeled": int((labels < 0).sum()),
            "positive_prevalence": float(y.mean()),
            "zero_shot_average_precision": average_precision(y, s),
            "complete_non_low_quality_AP": average_precision(y, baseline),
            "human_90_precision_threshold_on_dmel": metrics_at_threshold(y, s, human_90),
            "dmel_oracle_90_precision_point": threshold_for_precision(y, s, 0.9),
        }
        write_scores(args.output_dir / f"{task}_scores.tsv.gz", examples, scores)
        print(task, json.dumps(output["tasks"][task], indent=2), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
