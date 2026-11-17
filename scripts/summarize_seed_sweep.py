#!/usr/bin/env python3
"""Summarise a chrxtrain seed sweep: per-seed table plus mean/SD/min/max/range.

Collects, for every runs/<sweep>/seed*/chrxtrain/<cell> run:
  * best validation mean AP, best epoch, stopped epoch, training hours
    (runs/.../results.json)
  * chrX+ candidate AUPRC for donor/acceptor/start/stop and their mean
    (artifacts/scores/.../candidate_auprc.tsv)
  * strict locus Sn/Pr/F1 for UniAnn-only and EviAnn+UniAnn
    (work/uniann/.../vector_uniann_builtin_header/locus_metrics.tsv), plus the
    UniAnn commit each seed actually ran (uniann_commit.txt)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SITES = ("donor", "acceptor", "start", "stop")


def collect(sweep: str, cell: str, checkpoint: str = "best") -> pd.DataFrame:
    sub = "last" if checkpoint == "last" else ""
    rows = []
    for seed_dir in sorted((ROOT / "runs" / sweep).glob("seed*")):
        seed = int(re.sub(r"\D", "", seed_dir.name))
        tag = f"{sweep}/{seed_dir.name}"
        row: dict[str, object] = {"seed": seed}
        results = seed_dir / "chrxtrain" / cell / "results.json"
        if results.is_file():
            r = json.loads(results.read_text())
            history = r.get("history") or []
            best = max(history, key=lambda h: h.get("mean_AP") or -1) if history else {}
            row.update({
                "best_epoch": best.get("epoch"),
                "stopped_epoch": r.get("stopped_epoch"),
                "train_hours": round(r.get("elapsed_seconds", 0) / 3600, 2),
                "val_mean_AP": r.get("best_validation_mean_AP"),
            })
        auprc = ROOT / "artifacts/scores" / tag / "chrxtrain" / cell / sub / "candidate_auprc.tsv"
        if auprc.is_file():
            a = pd.read_csv(auprc, sep="\t").set_index("task")["AUPRC"]
            for site in SITES:
                row[f"chrX_AUPRC_{site}"] = a.get(site)
            row["chrX_AUPRC_mean"] = float(a.reindex(SITES).mean())
        uniann_dir = ROOT / "work/uniann" / tag / "chrxtrain" / cell / sub / "vector_uniann_builtin_header"
        locus = uniann_dir / "locus_metrics.tsv"
        commit = uniann_dir / "uniann_commit.txt"
        if commit.is_file():
            row["uniann_commit"] = commit.read_text().strip()[:7]
        if locus.is_file():
            m = pd.read_csv(locus, sep="\t").set_index("method")
            for method, prefix in (("UniAnn", "UniAnn"), ("EviAnn+UniAnn", "EviAnn+UniAnn")):
                if method in m.index:
                    for metric in ("sn", "pr", "f1", "predicted_loci", "matched_loci"):
                        row[f"{prefix}_{metric}"] = m.at[method, metric]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("seed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--cell", default="w1")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best",
                        help="which evaluated checkpoint to summarise")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    per_seed = collect(args.sweep, args.cell, args.checkpoint)
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "per_seed.tsv", sep="\t", index=False, float_format="%.6g")

    numeric = per_seed.drop(columns=["seed"]).select_dtypes("number")
    stats = pd.DataFrame({
        "n": numeric.count(),
        "mean": numeric.mean(),
        "sd": numeric.std(ddof=1),
        "min": numeric.min(),
        "max": numeric.max(),
    })
    stats["range"] = stats["max"] - stats["min"]
    stats.index.name = "metric"
    stats.to_csv(out / "seed_stats.tsv", sep="\t", float_format="%.6g")
    (out / "provenance.json").write_text(json.dumps({
        "sweep": args.sweep, "cell": args.cell, "regime": "chrxtrain",
        "checkpoint": args.checkpoint,
        "seeds": per_seed["seed"].tolist(),
        "uniann_commits": sorted(set(per_seed.get("uniann_commit", pd.Series(dtype=str)).dropna())),
        "uniann_policy": "latest origin/main pulled before every seed's UniAnn step",
        "uniann_scaling": "four-state vector scaling (fit on 3L validation), then "
                          "UniAnn's own built-in coding-frame scaling (convert_coding_frame.pl)",
        "caveat": "chrxtrain: chrX is in the training set, so chrX AUPRC and UniAnn "
                  "locus metrics are not held-out estimates.",
    }, indent=2) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(per_seed.to_string(index=False))
        print()
        print(stats.to_string())


if __name__ == "__main__":
    main()
