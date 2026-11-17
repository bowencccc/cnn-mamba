#!/usr/bin/env python3
"""Report figures for the Coleoptera EviAnn 1 kb (w1) chrxtrain seed sweep.

Figure 1 (seed_sweep_dropout0): five seeds, dropout 0 -
  (a) chrX+ candidate AUPRC per site, (b) strict locus Sn/Pr/F1 for UniAnn and
  EviAnn+UniAnn with the EviAnn-only reference, (c) per-seed F1 against the
  early-stopping epoch.
Figure 2 (dropout_comparison): seeds 42-44 with dropout 0.1 vs 0.

Reads results/<sweep>/per_seed.tsv written by scripts/summarize_seed_sweep.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SITES = [("donor", "Donor"), ("acceptor", "Acceptor"), ("start", "Start"),
         ("stop", "Stop"), ("mean", "Mean")]
# EviAnn-only strict locus metrics on chrX+ (model independent; same reference
# and gffcompare settings as the UniAnn rows).
EVIANN_ONLY = {"sn": 62.3, "pr": 89.9, "f1": 73.6}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def strip(ax, x, values, color, jitter=0.09, seed=0):
    rng = np.random.default_rng(seed)
    xs = x + rng.uniform(-jitter, jitter, len(values))
    ax.scatter(xs, values, s=34, color=color, alpha=0.55, edgecolor="white",
               linewidth=1.2, zorder=3)
    mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
    ax.errorbar(x + 0.24, mean, yerr=sd, fmt="o", color=color, ms=6,
                capsize=3, elinewidth=1.6, capthick=1.6, zorder=4)
    return mean, sd


def figure_sweep(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1.25, 1]})

    ax = axes[0]
    for i, (key, label) in enumerate(SITES):
        mean, sd = strip(ax, i, df[f"chrX_AUPRC_{key}"].to_numpy(), BLUE, seed=i)
        ax.text(i + 0.24, mean + sd + 0.006, f"{mean:.3f}", ha="center",
                va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(SITES)), [s[1] for s in SITES])
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0.80, 0.97)
    ax.set_title("(a) chrX+ candidate AUPRC", loc="left", fontsize=11)

    ax = axes[1]
    metrics = [("sn", "Sn"), ("pr", "Pr"), ("f1", "F1")]
    groups = [("UniAnn", "UniAnn"), ("EviAnn+UniAnn", "EviAnn+UniAnn")]
    for g, (prefix, title) in enumerate(groups):
        for m, (key, label) in enumerate(metrics):
            x = g * 3.6 + m
            mean, sd = strip(ax, x, df[f"{prefix}_{key}"].to_numpy(), BLUE,
                             seed=10 * g + m)
            ax.text(x + 0.24, mean + sd + 0.4, f"{mean:.1f}", ha="center",
                    va="bottom", fontsize=8.5, color=INK)
        ax.text(g * 3.6 + 1, 90.3, title, ha="center", fontsize=10, color=INK)
    ax.axhline(EVIANN_ONLY["f1"], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(6.55, EVIANN_ONLY["f1"] - 0.5, f"EviAnn only F1 {EVIANN_ONLY['f1']}",
            ha="right", va="top", fontsize=8.5, color=INK2)
    ax.set_xticks([g * 3.6 + m for g in range(2) for m in range(3)],
                  [lab for _ in groups for _, lab in metrics])
    ax.set_xlim(-0.5, 6.7)
    ax.set_ylim(70, 92)
    ax.set_ylabel("Strict locus metric (%)")
    ax.set_title("(b) chrX+ gene models vs RefSeq", loc="left", fontsize=11)

    ax = axes[2]
    for prefix, marker in (("UniAnn", "o"), ("EviAnn+UniAnn", "s")):
        ax.scatter(df["best_epoch"], df[f"{prefix}_f1"], s=46, marker=marker,
                   color=BLUE if prefix == "UniAnn" else "white",
                   edgecolor=BLUE, linewidth=1.6, zorder=3, label=prefix)
    for _, r in df.iterrows():
        right = int(r.seed) == 43  # avoid seed 46 one epoch to the left
        ax.annotate(f"seed {int(r.seed)}", (r.best_epoch, r["UniAnn_f1"]),
                    xytext=(8, 0) if right else (0, -13),
                    textcoords="offset points",
                    ha="left" if right else "center",
                    va="center" if right else "baseline",
                    fontsize=8, color=INK2)
    ax.set_xlabel("Best epoch (early stopping on 3L mean AP)")
    ax.set_ylabel("Strict locus F1 (%)")
    ax.set_ylim(72, 89)
    ax.set_xlim(2, 15)
    ax.legend(frameon=False, loc="center right", fontsize=9)
    ax.set_title("(c) Per-seed F1 vs stopping epoch", loc="left", fontsize=11)

    n = len(df)
    fig.suptitle(
        f"Coleoptera EviAnn labels, 1 kb windows, chrX-included training, dropout 0 "
        f"({n} seeds: {', '.join(str(s) for s in df.seed)})",
        x=0.01, ha="left", fontsize=12.5, color=INK, y=0.995)
    fig.text(0.01, 0.005,
             "Points: individual seeds; marker with bar: mean ± SD. UniAnn uses "
             "vector-scaled four-state probabilities (fit on 3L). chrX is in the "
             "training set, so chrX+ values are not held-out estimates.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"seed_sweep_dropout0.{ext}", dpi=220)
    plt.close(fig)


def spread(values, gap):
    """Label y positions: the values, pushed apart to at least `gap`."""
    order = np.argsort(values)
    ys = np.asarray(values, dtype=float)[order]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + gap)
    shift = (np.asarray(values)[order] - ys).mean()
    out = np.empty_like(ys)
    out[order] = ys + shift
    return out


def figure_dropout(d01: pd.DataFrame, d0: pd.DataFrame, out: Path) -> None:
    seeds = sorted(set(d01.dropna(subset=["UniAnn_f1"]).seed) & set(d0.seed))
    a = d01.set_index("seed").loc[seeds]
    b = d0.set_index("seed").loc[seeds]
    panels = [("chrX_AUPRC_mean", "chrX+ mean AUPRC", "{:.3f}"),
              ("UniAnn_f1", "UniAnn F1 (%)", "{:.1f}"),
              ("EviAnn+UniAnn_f1", "EviAnn+UniAnn F1 (%)", "{:.1f}")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (col, title, fmt) in zip(axes, panels):
        lo = min(a[col].min(), b[col].min())
        hi = max(a[col].max(), b[col].max())
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        gap = 0.075 * (hi - lo + 2 * pad)
        left = spread([a.at[s, col] for s in seeds], gap)
        right = spread([b.at[s, col] for s in seeds], gap)
        for i, s in enumerate(seeds):
            ax.plot([0, 1], [a.at[s, col], b.at[s, col]], color=MUTED, lw=1.2,
                    zorder=1)
            ax.scatter([0], [a.at[s, col]], s=60, color=ORANGE, edgecolor="white",
                       linewidth=1.5, zorder=3)
            ax.scatter([1], [b.at[s, col]], s=60, color=BLUE, edgecolor="white",
                       linewidth=1.5, zorder=3)
            ax.text(-0.08, left[i], f"seed {s}  {fmt.format(a.at[s, col])}",
                    ha="right", va="center", fontsize=8.5, color=INK2)
            ax.text(1.08, right[i], f"{fmt.format(b.at[s, col])}  seed {s}",
                    ha="left", va="center", fontsize=8.5, color=INK2)
        ax.set_xticks([0, 1], ["dropout 0.1", "dropout 0"])
        ax.set_xlim(-0.95, 1.95)
        ax.set_title(title, loc="left", fontsize=11)
    fig.suptitle("Dropout 0.1 vs 0, same seeds and data (1 kb, chrX-included)",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.text(0.01, 0.01,
             "With dropout 0.1, seeds 43 and 44 collapse in eval mode "
             "(residual dropout in layers 4-5 shifts activation variance); "
             "dropout 0 removes the failure.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"dropout_comparison.{ext}", dpi=220)
    plt.close(fig)


SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


DEFAULT_LOSS_NOTE = (
    "Large dot: best epoch (checkpoint kept); runs stop after 3 epochs "
    "without validation mean AP improvement. Training loss is the "
    "weighted multi-task loss in train mode.")


def figure_loss(sweep: str, cell: str, out: Path, name: str, title: str,
                note: str = DEFAULT_LOSS_NOTE, mark_last: bool = False) -> None:
    """Train loss, validation loss and validation mean AP per epoch, all seeds."""
    runs = []
    for seed_dir in sorted((ROOT / "runs" / sweep).glob("seed*")):
        path = seed_dir / "chrxtrain" / cell / "results.json"
        if path.is_file():
            r = json.loads(path.read_text())
            h = pd.DataFrame(r["history"])
            runs.append((int(seed_dir.name[4:]), h))
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [("train_loss", "(a) Training loss"),
              ("validation_loss", "(b) Validation loss (3L)"),
              ("mean_AP", "(c) Validation mean RefSeq AP (3L)")]
    for k, (seed, h) in enumerate(runs):
        color = SEED_COLORS[k % len(SEED_COLORS)]
        best = h.loc[h["mean_AP"].idxmax()]
        for ax, (col, _) in zip(axes, panels):
            ax.plot(h["epoch"], h[col], color=color, lw=2, label=f"seed {seed}",
                    zorder=2)
            ax.scatter([best["epoch"]], [best[col]], s=64, color=color,
                       edgecolor="white", linewidth=2, zorder=3)
            if mark_last:
                last = h.iloc[-1]
                ax.scatter([last["epoch"]], [last[col]], s=64, color="white",
                           edgecolor=color, linewidth=2, zorder=3)
    for ax, (col, t) in zip(axes, panels):
        ax.set_title(t, loc="left", fontsize=11)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel("Loss")
    axes[2].set_ylabel("Mean AP")
    if mark_last:
        handles, labels = axes[2].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=9.5, ncol=len(labels),
                   loc="upper left", bbox_to_anchor=(0.005, 0.935))
        top = 0.88
    else:
        axes[2].legend(frameon=False, fontsize=9, loc="lower right")
        top = 0.94
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.text(0.01, 0.01, note, fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, top))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def figure_fixed_results(base: pd.DataFrame, best: pd.DataFrame,
                         last: pd.DataFrame, out: Path) -> None:
    """Early stopping vs fixed-15 (best val checkpoint) vs fixed-15 (epoch 15)."""
    conds = [("Early stopping\n(30-ep cosine)", base, MUTED),
             ("Fixed 15 ep\nbest val epoch", best, BLUE),
             ("Fixed 15 ep\nepoch 15", last, ORANGE)]
    panels = [("chrX_AUPRC_mean", "Mean AUPRC", "(a) chrX+ candidate AUPRC (mean of 4 sites)", 0.006),
              ("UniAnn_f1", "Strict locus F1 (%)", "(b) UniAnn gene models", 0.5),
              ("EviAnn+UniAnn_f1", "Strict locus F1 (%)", "(c) EviAnn+UniAnn gene models", 0.5)]
    seeds = sorted(set(base.seed) & set(best.seed) & set(last.seed))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (col, ylab, title, pad) in zip(axes, panels):
        tab = {i: d.set_index("seed").loc[seeds, col] for i, (_, d, _) in enumerate(conds)}
        # same jitter as strip() so the pairing lines end on the dots
        jx = {i: i + np.random.default_rng(i).uniform(-0.09, 0.09, len(seeds)) for i in (1, 2)}
        for k, s in enumerate(seeds):
            ax.plot([jx[1][k], jx[2][k]], [tab[1][s], tab[2][s]], color=GRID, lw=1.3, zorder=1)
        for i, (label, d, color) in enumerate(conds):
            mean, sd = strip(ax, i, tab[i].to_numpy(), color, seed=i)
            ax.text(i + 0.24, mean + sd + pad, f"{mean:.3f}" if col.startswith("chrX") else f"{mean:.1f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
        if col == "EviAnn+UniAnn_f1" or col == "UniAnn_f1":
            ax.axhline(EVIANN_ONLY["f1"], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
            ax.text(-0.45, EVIANN_ONLY["f1"] - 0.25, f"EviAnn only F1 {EVIANN_ONLY['f1']}",
                    ha="left", va="top", fontsize=8.5, color=INK2)
        ax.set_xticks(range(3), [c[0] for c in conds])
        ax.set_xlim(-0.5, 2.6)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].set_ylim(0.80, 0.92)
    axes[1].set_ylim(66, 80)
    axes[2].set_ylim(72, 89)
    fig.suptitle(
        "Coleoptera EviAnn labels, 1 kb windows, chrX-included training, dropout 0: "
        f"early stopping vs fixed 15 epochs (seeds {', '.join(map(str, seeds))})",
        x=0.01, ha="left", fontsize=12.5, color=INK, y=0.995)
    fig.text(0.01, 0.005,
             "Points: individual seeds; marker with bar: mean ± SD; grey lines pair the "
             "same fixed-15 run's best-validation and final checkpoints. Early stopping is a "
             "separate run per seed. chrX is in the training set, so chrX+ values are not held-out.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fixed15_results.{ext}", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="coleoptera_w1_seed_sweep_dropout0_v1")
    parser.add_argument("--dropout01-sweep", default="coleoptera_w1_seed_sweep_v1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--fixed-epoch-loss", action="store_true",
        help="only draw loss curves for a fixed-epoch sweep (no early stopping)",
    )
    parser.add_argument(
        "--fixed-epoch-results", action="store_true",
        help="compare a fixed-epoch sweep (best and last checkpoints) with the "
             "early-stopping sweep given by --baseline-sweep",
    )
    parser.add_argument("--baseline-sweep", default="coleoptera_w1_seed_sweep_dropout0_v1")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="peak lr of the fixed-epoch sweep (labels only; floor is 0.1x)")
    args = parser.parse_args()
    if args.fixed_epoch_results:
        res = ROOT / "results" / args.sweep
        out = args.output_dir or res / "figures"
        out.mkdir(parents=True, exist_ok=True)
        read = lambda p: pd.read_csv(p, sep="\t").dropna(subset=["UniAnn_f1"])
        figure_fixed_results(read(ROOT / "results" / args.baseline_sweep / "per_seed.tsv"),
                             read(res / "per_seed.tsv"), read(res / "last_epoch" / "per_seed.tsv"), out)
        print(f"wrote figures to {out}")
        return
    if args.fixed_epoch_loss:
        out = args.output_dir or ROOT / "results" / args.sweep / "figures"
        out.mkdir(parents=True, exist_ok=True)
        n = len(list((ROOT / "runs" / args.sweep).glob("seed*/chrxtrain/w1/results.json")))
        lr = f"{args.lr:.1e}".replace("e-0", "e-")
        floor = f"{args.lr * 0.1:.1e}".replace("e-0", "e-")
        figure_loss(
            args.sweep, "w1", out,
            "loss_curves_fixed15" if args.lr == 3e-4 else f"loss_curves_fixed15_lr{lr}",
            "Training and validation curves, 1 kb, chrX-included, dropout 0, "
            f"fixed 15 epochs, peak lr {lr} ({n} seeds)",
            note=("Filled dot: best validation epoch; open circle: final epoch "
                  f"(epoch 15). No early stopping; cosine lr {lr} decays to {floor} by epoch "
                  "15. Training loss is the epoch-mean weighted multi-task loss."),
            mark_last=True)
        print(f"wrote figures to {out}")
        return
    d0 = pd.read_csv(ROOT / "results" / args.sweep / "per_seed.tsv", sep="\t")
    d0 = d0.dropna(subset=["UniAnn_f1"])
    out = args.output_dir or ROOT / "results" / args.sweep / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figure_sweep(d0, out)
    d01_path = ROOT / "results" / args.dropout01_sweep / "per_seed.tsv"
    if d01_path.is_file():
        figure_dropout(pd.read_csv(d01_path, sep="\t"), d0, out)
    figure_loss(args.sweep, "w1", out, "loss_curves_dropout0",
                "Training and validation curves, 1 kb, chrX-included, dropout 0 "
                f"({len(d0)} seeds)")
    figure_loss(args.dropout01_sweep, "w1", out, "loss_curves_dropout01",
                "Training and validation curves, 1 kb, chrX-included, dropout 0.1")
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
