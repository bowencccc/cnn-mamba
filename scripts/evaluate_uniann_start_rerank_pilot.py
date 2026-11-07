#!/usr/bin/env python3
"""Compare current UniAnn starts with NetStart reranking on predicted chains."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.metadata, sep="\t")
    pred = pd.read_csv(args.predictions)
    frame = pred.merge(meta, on="entry_line", how="inner", validate="many_to_one")
    frame["is_true_start"] = frame.atg_position == frame.true_atg_position_1based
    counts = frame.groupby("entry_line").is_true_start.sum()
    if not (counts == 1).all():
        raise RuntimeError(f"Expected one true ATG per entry: {counts.value_counts().to_dict()}")
    top = frame.loc[frame.groupby("entry_line").preds.idxmax()].copy()
    top["netstart_top1_exact"] = top.is_true_start.astype(int)
    ranks = frame.groupby("entry_line", group_keys=False).apply(
        lambda group: 1 + int((group.preds > group.loc[group.is_true_start, "preds"].iloc[0]).sum()),
        include_groups=False,
    ).rename("true_rank")
    per_tx = meta.merge(top[["entry_line", "netstart_top1_exact", "preds"]], on="entry_line")
    per_tx = per_tx.merge(ranks, left_on="entry_line", right_index=True)
    per_tx.to_csv(args.output_dir / "per_transcript.tsv", sep="\t", index=False)
    summary = {
        "evaluable_transcripts": int(len(meta)),
        "current_uniann_start_exact": int(meta.uniann_start_exact.sum()),
        "current_uniann_top1_accuracy": float(meta.uniann_start_exact.mean()),
        "netstart_start_exact": int(per_tx.netstart_top1_exact.sum()),
        "netstart_top1_accuracy": float(per_tx.netstart_top1_exact.mean()),
        "netstart_candidate_auprc": float(average_precision_score(frame.is_true_start, frame.preds)),
        "netstart_median_true_rank": float(per_tx.true_rank.median()),
        "fixed_by_netstart": int(((per_tx.uniann_start_exact == 0) & (per_tx.netstart_top1_exact == 1)).sum()),
        "broken_by_netstart": int(((per_tx.uniann_start_exact == 1) & (per_tx.netstart_top1_exact == 0)).sum()),
        "scope": "conditional on a predicted multi-exon chain containing exactly one RefSeq ATG start after 5-prime extension",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([summary]).to_csv(args.output_dir / "summary.tsv", sep="\t", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
