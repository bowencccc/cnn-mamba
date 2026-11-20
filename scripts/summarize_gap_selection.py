#!/usr/bin/env python3
"""Summarise scripts/run_gap_selection_eval.sh: per (seed, epoch) 3L/chrX metrics,
and chrX outcomes of each 3L-based epoch-selection rule.
usage: summarize_gap_selection.py [OUTPUT_TAG] [SOURCE_SWEEP]"""
import csv, json, sys
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
tag = sys.argv[1] if len(sys.argv) > 1 else "coleoptera_w1_gap_selection_perisoform_v1"
src = sys.argv[2] if len(sys.argv) > 2 else "coleoptera_w1_seed_sweep_fixed15_perisoform_v1"
rows = []
for seed_dir in sorted((ROOT / "work/uniann" / tag).glob("seed*")):
    seed = int(seed_dir.name[4:])
    hist = json.loads((ROOT / "runs" / src / seed_dir.name / "chrxtrain/w1/results.json").read_text())["history"]
    ap = {h["epoch"]: h["mean_AP"] for h in hist}
    for ep_dir in sorted(seed_dir.glob("epoch*")):
        epoch = int(ep_dir.name[5:])
        row = {"seed": seed, "epoch": epoch, "3L_mean_AP": ap[epoch]}
        for chrom, short in (("chr3L", "3L"), ("chrX", "X")):
            w = ep_dir / chrom / "vector_uniann_builtin_header"
            loc = {r["method"]: float(r["f1"]) for r in csv.DictReader(open(w / "locus_metrics.tsv"), delimiter="\t")}
            gap = next(csv.DictReader(open(w / "gap_locus_metrics.tsv"), delimiter="\t"))
            row |= {f"{short}_UniAnn_f1": loc["UniAnn"], f"{short}_EU_f1": loc["EviAnn+UniAnn"],
                    f"{short}_gap_f1": float(gap["gap_f1"])}
        rows.append(row)
df = pd.DataFrame(rows).sort_values(["seed", "epoch"])
out = ROOT / "results" / tag
df.to_csv(out / "per_epoch.tsv", sep="\t", index=False, float_format="%.4f")

rules = {"3L mean AP": "3L_mean_AP", "3L Gap F1": "3L_gap_f1", "3L EviAnn+UniAnn F1": "3L_EU_f1",
         "oracle: chrX Gap F1": "X_gap_f1", "oracle: chrX EviAnn+UniAnn F1": "X_EU_f1"}
sel = []
for rule, col in rules.items():
    for seed, g in df.groupby("seed"):
        r = g.loc[g[col].idxmax()]
        sel.append({"rule": rule, "seed": seed, "epoch": int(r.epoch), "X_gap_f1": r.X_gap_f1,
                    "X_EU_f1": r.X_EU_f1, "X_UniAnn_f1": r.X_UniAnn_f1})
sel = pd.DataFrame(sel)
sel.to_csv(out / "selection_per_seed.tsv", sep="\t", index=False, float_format="%.4f")
summary = sel.groupby("rule", sort=False).agg(
    epochs=("epoch", lambda x: ",".join(map(str, x))),
    X_gap_f1_mean=("X_gap_f1", "mean"), X_gap_f1_sd=("X_gap_f1", "std"),
    X_EU_f1_mean=("X_EU_f1", "mean"), X_EU_f1_sd=("X_EU_f1", "std"),
    X_UniAnn_f1_mean=("X_UniAnn_f1", "mean"), X_UniAnn_f1_sd=("X_UniAnn_f1", "std"))
summary.to_csv(out / "selection_summary.tsv", sep="\t", float_format="%.4f")
print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
print("\nwithin-seed Spearman across epochs (mean over seeds):")
for a, b in [("3L_gap_f1", "X_gap_f1"), ("3L_EU_f1", "X_EU_f1"), ("3L_mean_AP", "X_gap_f1"), ("3L_mean_AP", "X_EU_f1")]:
    rho = [spearmanr(g[a], g[b])[0] for _, g in df.groupby("seed")]
    print(f"  {a:>11} vs {b:<9} rho = {sum(rho)/len(rho):+.2f}  per seed {[round(x, 2) for x in rho]}")
