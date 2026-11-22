#!/usr/bin/env python3
"""Checkpoint-averaging figure for the weight-decay 1e-3 runs (cybertron).

(a, b) seed 44: single epochs 4/5/6 vs arithmetic averages 3-5, 4-5, 4-6;
       EviAnn+UniAnn strict locus F1 and corrected Gap-locus F1 on chrX+.
(c, d) seeds 42-44: checkpoint selected by 3L validation mean AP vs the
       epochs 4-6 average; EviAnn+UniAnn F1 and chrX+ mean candidate AUPRC.
Output: results/coleoptera_w1_weight_decay_v1/wd1em3/figures/checkpoint_averaging.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
EVIANN_ONLY_F1 = 73.6

neg = pd.read_csv(ROOT / "results/coleoptera_w1_negative_weight_v1/negative_weight_gap_summary.tsv", sep="\t")
avg = pd.read_csv(ROOT / "results/coleoptera_w1_weight_decay_v1/wd1em3/checkpoint_averaging_summary.tsv", sep="\t")
three = pd.read_csv(ROOT / "results/coleoptera_w1_negative_weight_v1/n100_three_seed_gap_summary.tsv", sep="\t")
sel = pd.read_csv(ROOT / "results/coleoptera_w1_weight_decay_v1/wd1em3/per_seed.tsv", sep="\t")

raw = neg[(neg.setting == 1.0) & neg.checkpoint.str.startswith("epoch")].sort_values("checkpoint")
seed44 = pd.concat([
    pd.DataFrame({"label": [f"epoch {int(c[5:])}" for c in raw.checkpoint], "kind": "single",
                  "union_f1": raw.union_f1, "gap_f1": raw.gap_f1,
                  "selected": [c == "epoch06" for c in raw.checkpoint]}),
    pd.DataFrame({"label": [f"avg {int(m[9:11])}–{int(m[12:14])}" for m in avg.model], "kind": "average",
                  "union_f1": avg.union_f1, "gap_f1": avg.gap_f1, "selected": False}),
], ignore_index=True)
seed44 = pd.concat([seed44[seed44.kind == "single"],
                    seed44[seed44.kind == "average"].sort_values("label")], ignore_index=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(1, 4, figsize=(17, 4.9), gridspec_kw={"width_ratios": [1.35, 1.35, 1, 1]})


def seed44_panel(ax, col, ylabel, title, fmt, ref=None):
    xs = range(len(seed44))
    colors = [MUTED if k == "single" else BLUE for k in seed44.kind]
    ax.bar(xs, seed44[col], width=0.6, color=colors, edgecolor="white", linewidth=2, zorder=2)
    lo = seed44[col].min()
    span = seed44[col].max() - lo
    ax.set_ylim(lo - span * 1.2, seed44[col].max() + span * 0.45)
    for x, (v, s) in enumerate(zip(seed44[col], seed44.selected)):
        ax.text(x, v + span * 0.04, fmt(v), ha="center", va="bottom", fontsize=9, color=INK)
    labels = [l + ("\n(mean-AP pick)" if s else "") for l, s in zip(seed44.label, seed44.selected)]
    ax.set_xticks(list(xs), labels, fontsize=9)
    ax.axvline(2.5, color=GRID, lw=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)


seed44_panel(axes[0], "union_f1", "Strict locus F1 (%)", "(a) Seed 44: EviAnn+UniAnn F1", lambda v: f"{v:.1f}")
seed44_panel(axes[1], "gap_f1", "Gap-locus F1", "(b) Seed 44: Gap-locus F1", lambda v: f"{v:.3f}")

paired = sel[["seed", "best_epoch", "EviAnn+UniAnn_f1", "chrX_AUPRC_mean"]].merge(
    three[["seed", "union_f1", "mean_AP"]], on="seed")


def paired_panel(ax, a, b, ylabel, title, fmt, ref=None):
    for _, r in paired.iterrows():
        ax.plot([0, 1], [r[a], r[b]], color=GRID, lw=1.5, zorder=1)
        ax.scatter([0], [r[a]], s=40, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3)
        ax.scatter([1], [r[b]], s=40, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(-0.08, r[a], f"{int(r.seed)} (ep {int(r.best_epoch)})", ha="right", va="center",
                fontsize=8, color=INK2)
    for x, c, col in ((0, ORANGE, a), (1, BLUE, b)):
        m, s = paired[col].mean(), paired[col].std()
        ax.errorbar([x + 0.14], [m], yerr=[s], fmt="o", color=c, ms=8, capsize=4, lw=2, zorder=4)
        ax.text(x + 0.24, m, fmt(m) + "\n±" + fmt(s), va="center", fontsize=9, color=INK)
    if ref is not None:
        ax.axhline(ref, color=MUTED, ls="--", lw=1)
    ax.set_xticks([0, 1], ["single checkpoint\n(3L mean AP)", "average\nepochs 4–6"], fontsize=9)
    ax.set_xlim(-0.75, 1.65)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)


paired_panel(axes[2], "EviAnn+UniAnn_f1", "union_f1", "Strict locus F1 (%)",
             "(c) Seeds 42–44: EviAnn+UniAnn F1", lambda v: f"{v:.2f}")
paired_panel(axes[3], "chrX_AUPRC_mean", "mean_AP", "Mean AUPRC",
             "(d) Seeds 42–44: chrX+ AUPRC (4 sites)", lambda v: f"{v:.3f}")

handles = [plt.Rectangle((0, 0), 1, 1, color=MUTED, label="single epoch"),
           plt.Rectangle((0, 0), 1, 1, color=BLUE, label="averaged checkpoint"),
           plt.Line2D([], [], marker="o", ls="", color=ORANGE, label="checkpoint selected by 3L validation mean AP")]
fig.legend(handles=handles, frameon=False, ncol=3, fontsize=9.5, loc="upper left", bbox_to_anchor=(0.005, 0.93))
fig.suptitle("Checkpoint averaging, weight decay 1e-3 (union windows, lr 1.5e-4, early stopping patience 2, "
             "dropout 0, chrX-included training; UniAnn 08eaf15, strict gffcompare on chrX+)",
             x=0.01, ha="left", fontsize=12, color=INK, y=0.995)
fig.text(0.01, 0.005,
         "Averages are arithmetic means of model weights. The 4–6 range was chosen on seed 44 by chrX+ Gap F1, "
         "then applied to seeds 42 and 43 (whose mean-AP-selected epoch was 5, so 4–6 = selected epoch ± 1). "
         "chrX is in the training set; chrX+ values are not held-out. Gap F1 for seeds 42/43 single checkpoints was not computed. "
         "Seed 44 epoch 6 scores 84.7 in (a) (negative-weight grid re-evaluation) and 84.3 in (c) (original sweep evaluation); same weights.",
         fontsize=8.5, color=INK2, ha="left", va="bottom", wrap=True)
fig.tight_layout(rect=(0, 0.06, 1, 0.88))
out = ROOT / "results/coleoptera_w1_weight_decay_v1/wd1em3/figures"
out.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(out / f"checkpoint_averaging.{ext}", dpi=200)
print(f"wrote {out}/checkpoint_averaging.png")
