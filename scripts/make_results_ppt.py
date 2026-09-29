#!/usr/bin/env python3
"""Create an English results deck for the CNN--Mamba-k7 experiments."""

from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "presentations"
STEM = "cnn_mamba_drosophila_results"

BG = "#F5F7FB"
INK = "#172A3A"
MUTED = "#607184"
GRID = "#DCE3EC"
WHITE = "#FFFFFF"
BLUE = "#3478F6"
TEAL = "#159A8A"
ORANGE = "#F59E0B"
RED = "#E45756"
PURPLE = "#7C4DFF"
GREEN = "#54A24B"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,
})


def add_box(fig, x, y, w, h, text, color=WHITE, edge=GRID, fontsize=15,
            weight="normal", text_color=INK, radius=0.018, lw=1.4,
            align="center", zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        transform=fig.transFigure, facecolor=color, edgecolor=edge,
        linewidth=lw, zorder=zorder,
    )
    fig.add_artist(patch)
    tx = x + w / 2 if align == "center" else x + 0.018
    ha = "center" if align == "center" else "left"
    fig.text(
        tx, y + h / 2, text, ha=ha, va="center", fontsize=fontsize,
        weight=weight, color=text_color, zorder=zorder + 1,
        linespacing=1.25, wrap=True,
    )
    return patch


def add_arrow(fig, x1, y1, x2, y2, color=MUTED, lw=2.0,
              connectionstyle="arc3"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=14, linewidth=lw,
        color=color, connectionstyle=connectionstyle, zorder=4,
    )
    fig.add_artist(arrow)


def add_pill(fig, x, y, text, color=BLUE, width=None, fontsize=10):
    width = width or max(0.07, 0.0085 * len(text) + 0.025)
    add_box(
        fig, x, y, width, 0.035, text, color=color, edge=color,
        fontsize=fontsize, weight="bold", text_color=WHITE,
        radius=0.016, lw=0.0,
    )


def add_bullets(fig, x, y, lines, fontsize=15, gap=0.073,
                bullet_color=BLUE):
    current = y
    for line in lines:
        fig.add_artist(Circle(
            (x, current + 0.004), 0.006, transform=fig.transFigure,
            facecolor=bullet_color, edgecolor="none", zorder=3,
        ))
        fig.text(
            x + 0.018, current, line, ha="left", va="center",
            fontsize=fontsize, color=INK, linespacing=1.25,
            wrap=True, zorder=4,
        )
        current -= gap


def add_image(fig, path, rect, border=False):
    axis = fig.add_axes(rect, zorder=2)
    axis.imshow(plt.imread(path))
    axis.axis("off")
    if border:
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(GRID)
            spine.set_linewidth(1.0)
    return axis


def base_slide(title, number, kicker=None, subtitle=None):
    fig = plt.figure(figsize=(13.333, 7.5), dpi=144, facecolor=BG)
    fig.add_artist(FancyBboxPatch(
        (0, 0.965), 1, 0.035, boxstyle="square,pad=0",
        transform=fig.transFigure, facecolor=BLUE, edgecolor="none",
    ))
    if kicker:
        fig.text(0.055, 0.915, kicker.upper(), fontsize=10.5,
                 weight="bold", color=BLUE, va="center")
        title_y = 0.858
    else:
        title_y = 0.88
    title_size = 26 if len(title) < 52 else 23 if len(title) < 66 else 21
    fig.text(0.055, title_y, title, fontsize=title_size, weight="bold",
             color=INK, va="center")
    if subtitle:
        fig.text(0.055, title_y - 0.051, subtitle, fontsize=12.2,
                 color=MUTED, va="center")
    fig.text(0.055, 0.028, "Frame-aware CNN–Mamba for gene-signal prediction",
             fontsize=8.5, color=MUTED)
    fig.text(0.955, 0.028, f"{number:02d}", ha="right",
             fontsize=9, color=MUTED, weight="bold")
    return fig


def style_axis(axis):
    axis.set_facecolor(WHITE)
    axis.grid(axis="y", alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=10)


def slide_title():
    fig = plt.figure(figsize=(13.333, 7.5), dpi=144, facecolor=INK)
    fig.add_artist(FancyBboxPatch(
        (0.055, 0.16), 0.012, 0.68, boxstyle="square,pad=0",
        transform=fig.transFigure, facecolor=ORANGE, edgecolor="none",
    ))
    fig.text(0.105, 0.72, "Frame-aware CNN–Mamba", fontsize=40,
             weight="bold", color=WHITE, va="center")
    fig.text(0.105, 0.60,
             "Gene-signal prediction and strict locus reconstruction",
             fontsize=26, weight="bold", color="#DCE8F7", va="top")
    fig.text(0.105, 0.44,
             "Drosophila chrX benchmark • UniAnn integration • human transfer gap",
             fontsize=15, color="#AFC2D9")
    add_pill(fig, 0.105, 0.285, "CNN–MAMBA-k7", BLUE, 0.145, 10)
    add_pill(fig, 0.275, 0.285, "FRAME DILATION 3", TEAL, 0.17, 10)
    add_pill(fig, 0.47, 0.285, "PHASE AUXILIARY", PURPLE, 0.17, 10)
    add_pill(fig, 0.665, 0.285, "STRICT GFFCOMPARE", ORANGE, 0.19, 10)
    fig.text(0.105, 0.11,
             "Architecture ablations, window-size search, auxiliary-loss grids, and end-to-end annotation",
             fontsize=12.2, color="#86A2BF")
    return fig


def slide_evaluation_design():
    fig = base_slide(
        "Two evaluation regimes answer different questions", 2,
        kicker="Experimental design",
        subtitle="Held-out chrX measures generalization; chrX-in-training supports UniAnn assembly",
    )
    add_box(fig, 0.06, 0.51, 0.40, 0.27,
            "MODEL EVALUATION\n\nTrain: 2L, 2R, 3R, 4, Y\nValidation: 3L\nTest: chrX (never seen in training)",
            color="#EAF1FF", edge=BLUE, fontsize=16, weight="bold")
    add_box(fig, 0.54, 0.51, 0.40, 0.27,
            "ANNOTATION EVALUATION\n\nchrX is allowed in training\nScore every canonical chrX+ candidate\nRun UniAnn and strict gffcompare",
            color="#FFF3DC", edge=ORANGE, fontsize=16, weight="bold")
    add_arrow(fig, 0.20, 0.48, 0.20, 0.38, color=BLUE)
    add_arrow(fig, 0.68, 0.48, 0.68, 0.38, color=ORANGE)
    add_box(fig, 0.06, 0.18, 0.40, 0.17,
            "Candidate-level metrics\nExact AUPRC and PR–Sn curves\nDonor • acceptor • start • stop",
            color=WHITE, edge=BLUE, fontsize=14.5, weight="bold")
    add_box(fig, 0.54, 0.18, 0.40, 0.17,
            "Gene-model metrics\ngffread -M → gffcompare --strict-match -e 0\nSn • Pr • F1 at locus level",
            color=WHITE, edge=ORANGE, fontsize=14.5, weight="bold")
    fig.text(0.50, 0.105,
             "The two regimes must not be mixed: chrX-in-training scores are not evidence of held-out generalization.",
             ha="center", fontsize=12.5, color=RED, weight="bold")
    return fig


def slide_architecture():
    fig = base_slide(
        "From sequence context to frame-aware multi-task supervision", 3,
        kicker="Model",
        subtitle="The shared backbone is unchanged; frame and auxiliary heads reshape what it learns",
    )
    boxes = [
        (0.055, 0.57, 0.12, "DNA tokens\nA/C/G/T/N", BLUE),
        (0.215, 0.57, 0.14, "Embedding\n5 → 192", BLUE),
        (0.395, 0.57, 0.20, "8 × CNN–BiMamba3\nkernel size 7", TEAL),
        (0.635, 0.57, 0.14, "Final\nLayerNorm", PURPLE),
    ]
    for x, y, w, label, color in boxes:
        add_box(fig, x, y, w, 0.14, label, color=WHITE, edge=color,
                fontsize=14, weight="bold")
    for left, right in zip(boxes[:-1], boxes[1:]):
        add_arrow(fig, left[0] + left[2], 0.64, right[0] - 0.01, 0.64)

    add_arrow(fig, 0.775, 0.64, 0.825, 0.73, color=BLUE,
              connectionstyle="arc3,rad=-0.12")
    add_box(fig, 0.83, 0.67, 0.13, 0.13,
            "Splice head\nBG / donor / acceptor",
            color="#EAF1FF", edge=BLUE, fontsize=11.5, weight="bold")
    add_box(fig, 0.83, 0.49, 0.13, 0.13,
            "Start–stop head\nBG / start / stop",
            color="#FFF3DC", edge=ORANGE, fontsize=11.5, weight="bold")
    add_arrow(fig, 0.70, 0.56, 0.70, 0.47, color=TEAL)
    add_box(fig, 0.61, 0.35, 0.18, 0.11,
            "Frame branch\nConv1D dilation = 3",
            color="#DDF4F0", edge=TEAL, fontsize=13, weight="bold")
    add_arrow(fig, 0.79, 0.405, 0.825, 0.55, color=ORANGE,
              connectionstyle="arc3,rad=-0.12")
    add_arrow(fig, 0.79, 0.405, 0.83, 0.365, color=PURPLE,
              connectionstyle="arc3,rad=0.10")
    add_box(fig, 0.83, 0.30, 0.13, 0.13,
            "Phase head\nphase 0 / 1 / 2",
            color="#F0EDFF", edge=PURPLE, fontsize=12, weight="bold")

    add_box(fig, 0.06, 0.14, 0.47, 0.18,
            "Training objective\n\nL = 0.25 Lsplice + Lstart/stop + 0.10 Lphase",
            color=WHITE, edge=GRID, fontsize=17, weight="bold")
    fig.text(0.58, 0.22,
             "Phase labels follow annotated CDS across introns.\nThe auxiliary head is used during training; UniAnn receives\nonly donor, acceptor, start, and stop candidate scores.",
             fontsize=13, color=MUTED, linespacing=1.45, va="center")
    return fig


def slide_frame_ablation():
    fig = base_slide(
        "Frame and phase supervision drive the largest gain", 4,
        kicker="Held-out chrX",
        subtitle="Exact positive-strand candidate AUPRC; chrX excluded from training",
    )
    data = pd.read_csv(
        ROOT / "results/frame_ablation_v1/exact_chrx_plus_auprc.tsv",
        sep="\t",
    )
    axis = fig.add_axes([0.055, 0.17, 0.60, 0.59])
    tasks = ["donor", "acceptor", "start", "stop"]
    x = np.arange(4)
    width = 0.19
    colors = ["#9D9D9D", "#72B7B2", ORANGE, BLUE]
    for index, row in data.iterrows():
        axis.bar(
            x + (index - 1.5) * width,
            [100 * row[task] for task in tasks],
            width, label=row["model"], color=colors[index],
        )
    axis.set_xticks(x, ["Donor", "Acceptor", "Start", "Stop"])
    axis.set_ylim(45, 100)
    axis.set_ylabel("AUPRC (%)")
    axis.legend(fontsize=9, ncol=2, frameon=True)
    style_axis(axis)

    add_box(fig, 0.70, 0.57, 0.25, 0.16,
            "START\n71.8 → 85.1\n+13.3 points",
            color="#FFF3DC", edge=ORANGE, fontsize=16, weight="bold")
    add_box(fig, 0.70, 0.36, 0.25, 0.16,
            "STOP\n53.4 → 88.2\n+34.9 points",
            color="#EAF1FF", edge=BLUE, fontsize=16, weight="bold")
    add_box(fig, 0.70, 0.15, 0.25, 0.16,
            "FOUR-TASK MEAN\n78.6 → 91.0\n+12.4 points",
            color="#DDF4F0", edge=TEAL, fontsize=16, weight="bold")
    return fig


def slide_window_size():
    fig = base_slide(
        "Longer windows are not the primary performance driver", 5,
        kicker="Window-size ablation",
        subtitle="Same CNN–Mamba-k7 setup; early stopping applied independently",
    )
    exact = pd.read_csv(
        ROOT / "results/early_stopping_v1/plots/chrx_heldout_auprc/exact_auprc.tsv",
        sep="\t",
    )
    mean_ap = exact.groupby("window_kb", sort=True)["AUPRC"].mean() * 100
    locus = pd.read_csv(
        ROOT / "results/early_stopping_v1/plots/uniann_locus/"
        "early_stopping_uniann_eviann_combined_locus_metrics_with_frame.tsv",
        sep="\t",
    )
    plain = locus[
        (locus.method == "EviAnn+UniAnn")
        & (~locus.frame_auxiliary.astype(bool))
    ].set_index("window_kb").sort_index()
    windows = mean_ap.index.to_numpy()

    axis = fig.add_axes([0.07, 0.20, 0.59, 0.55])
    axis.plot(windows, mean_ap, "o-", linewidth=2.5, markersize=7,
              color=GREEN, label="Mean held-out AUPRC")
    axis.plot(windows, plain.loc[windows, "f1"], "s-", linewidth=2.5,
              markersize=7, color=BLUE, label="EviAnn + UniAnn locus F1")
    axis.scatter([10], [90.5378244], s=180, marker="*", color=ORANGE,
                 edgecolor=INK, linewidth=0.6, zorder=5,
                 label="10 kb + frame/phase")
    axis.set_xticks(windows)
    axis.set_ylim(76, 92)
    axis.set_xlabel("Window size (kb)")
    axis.set_ylabel("Score (%)")
    axis.legend(fontsize=9, frameon=True, loc="center right")
    style_axis(axis)

    add_box(fig, 0.71, 0.55, 0.23, 0.16,
            "Best raw mean AUPRC\n20 kb: 78.96%",
            color="#DDF4F0", edge=GREEN, fontsize=16, weight="bold")
    add_box(fig, 0.71, 0.34, 0.23, 0.16,
            "Best final locus F1\n10 kb + frame/phase: 90.54%",
            color="#FFF3DC", edge=ORANGE, fontsize=15, weight="bold")
    fig.text(0.70, 0.19,
             "Increasing context alone is not monotonic.\nBiological supervision matters more than\nwindow length.",
             fontsize=11.5, color=MUTED, weight="bold", linespacing=1.35)
    return fig


def slide_phase_weight():
    fig = base_slide(
        "Phase-loss weight controls a recall–precision trade-off", 6,
        kicker="Auxiliary-loss grid",
        subtitle="Architecture fixed; only λphase changes",
    )
    data = pd.read_csv(
        ROOT / "results/phase_weight_grid_v1/all_metrics_wide.tsv", sep="\t"
    ).sort_values("phase_weight")
    x = data.phase_weight.to_numpy()
    axis = fig.add_axes([0.07, 0.20, 0.58, 0.55])
    axis.plot(x, data.uniann_f1, "o-", linewidth=2.5, markersize=7,
              color=ORANGE, label="UniAnn")
    axis.plot(x, data.eviann_uniann_f1, "s-", linewidth=2.5, markersize=7,
              color=BLUE, label="EviAnn + UniAnn")
    axis.set_xticks(x)
    axis.set_ylim(71.5, 92)
    axis.set_xlabel("Phase auxiliary weight")
    axis.set_ylabel("Strict locus F1 (%)")
    axis.legend(fontsize=10, frameon=True)
    style_axis(axis)
    axis.annotate("Best UniAnn", (0.03, 79.1847), xytext=(14, 18),
                  textcoords="offset points", fontsize=10, color=ORANGE,
                  weight="bold", arrowprops=dict(arrowstyle="->", color=ORANGE))
    axis.annotate("Best combined", (0.10, 90.5378), xytext=(8, -28),
                  textcoords="offset points", fontsize=10, color=BLUE,
                  weight="bold", arrowprops=dict(arrowstyle="->", color=BLUE))

    add_box(fig, 0.70, 0.56, 0.25, 0.16,
            "λphase = 0.03\nUniAnn Sn / Pr / F1\n80.3 / 78.1 / 79.18",
            color="#FFF3DC", edge=ORANGE, fontsize=14.5, weight="bold")
    add_box(fig, 0.70, 0.34, 0.25, 0.16,
            "λphase = 0.10\nEviAnn + UniAnn Sn / Pr / F1\n91.6 / 89.5 / 90.54",
            color="#EAF1FF", edge=BLUE, fontsize=14.5, weight="bold")
    fig.text(0.71, 0.19,
             "λphase = 0.20 is consistently worse,\nindicating over-regularization.",
             fontsize=13, color=RED, weight="bold", linespacing=1.4)
    return fig


def slide_dense_cds():
    fig = base_slide(
        "Dense CDS supervision improves recall—but not final precision", 7,
        kicker="CDS auxiliary head",
        subtitle="The CDS head is highly accurate, yet its signal is largely redundant downstream",
    )
    data = pd.read_csv(
        ROOT / "results/cds_weight_grid_v1/all_metrics_wide.tsv", sep="\t"
    ).sort_values("cds_weight")
    x = data.cds_weight.to_numpy()
    axis = fig.add_axes([0.06, 0.19, 0.54, 0.56])
    axis.plot(x, data.uniann_f1, "o-", linewidth=2.5, markersize=7,
              color=ORANGE, label="UniAnn")
    axis.plot(x, data.combined_f1, "s-", linewidth=2.5, markersize=7,
              color=BLUE, label="EviAnn + UniAnn")
    axis.set_xticks(x)
    axis.set_ylim(75, 92)
    axis.set_xlabel("Dense CDS auxiliary weight")
    axis.set_ylabel("Strict locus F1 (%)")
    axis.legend(fontsize=9.5, frameon=True)
    style_axis(axis)

    add_box(fig, 0.64, 0.61, 0.31, 0.13,
            "CDS prediction itself is easy\nAP = 0.986–0.990",
            color="#DDF4F0", edge=TEAL, fontsize=15, weight="bold")
    add_box(fig, 0.64, 0.43, 0.31, 0.13,
            "λCDS = 0.03 improves UniAnn\nF1: 77.15 → 78.04  |  Sn: 77.1 → 79.0",
            color="#FFF3DC", edge=ORANGE, fontsize=13.5, weight="bold")
    add_box(fig, 0.64, 0.25, 0.31, 0.13,
            "But final combined F1 falls\n90.54 → 90.02",
            color="#FDE9E8", edge=RED, fontsize=15, weight="bold")
    fig.text(0.65, 0.145,
             "Additional true loci mostly overlap EviAnn.\nNovel loci increase, and λCDS = 0.10 consumes\n~30–40% of the weighted loss.",
             fontsize=10.8, color=MUTED, weight="bold", linespacing=1.3)
    return fig


def slide_best_locus():
    fig = base_slide(
        "The frame-aware model improves end-to-end gene reconstruction", 8,
        kicker="Strict locus benchmark",
        subtitle="10 kb, dilation 3, λphase = 0.10; chrX included for UniAnn scoring",
    )
    metrics = ["Sensitivity", "Precision", "F1"]
    uniann = [77.1, 77.2, 77.1499676]
    combined = [91.6, 89.5, 90.5378244]
    axis = fig.add_axes([0.07, 0.20, 0.51, 0.55])
    x = np.arange(3)
    width = 0.34
    bars1 = axis.bar(x - width / 2, uniann, width, color=ORANGE, label="UniAnn")
    bars2 = axis.bar(x + width / 2, combined, width, color=BLUE,
                     label="EviAnn + UniAnn")
    axis.bar_label(bars1, fmt="%.1f", padding=3, fontsize=10)
    axis.bar_label(bars2, fmt="%.1f", padding=3, fontsize=10)
    axis.set_xticks(x, metrics)
    axis.set_ylim(65, 96)
    axis.set_ylabel("Locus-level score (%)")
    axis.legend(fontsize=10, frameon=True)
    style_axis(axis)

    columns = ["Method", "Predicted", "Matched", "Missed", "Novel"]
    values = [
        ["UniAnn", "1,068", "825", "41", "28"],
        ["EviAnn + UniAnn", "1,080", "980", "20", "35"],
    ]
    table_axis = fig.add_axes([0.62, 0.48, 0.33, 0.25])
    table_axis.axis("off")
    table = table_axis.table(cellText=values, colLabels=columns,
                             cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.08, 2.0)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(WHITE)
        cell.set_linewidth(2)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#EAF1FF" if row == 2 else "#FFF3DC")
    add_box(fig, 0.64, 0.20, 0.29, 0.18,
            "Primary outcome\n\nEviAnn + UniAnn\nstrict-locus F1 = 90.54%",
            color=WHITE, edge=BLUE, fontsize=17, weight="bold")
    return fig


def slide_human_gap():
    fig = base_slide(
        "The human transfer gap is concentrated at translation boundaries", 9,
        kicker="Generalization",
        subtitle="Current human chr1 benchmark; candidate-level AUPRC",
    )
    human = pd.read_csv(
        ROOT / "results/human_cnn_mamba_k7/chr1_plus_auprc.tsv", sep="\t"
    )
    dmel = pd.read_csv(
        ROOT / "results/frame_ablation_v1/exact_chrx_plus_auprc.tsv", sep="\t"
    ).query("model == 'Combined'").iloc[0]
    tasks = ["donor", "acceptor", "start", "stop"]
    x = np.arange(4)
    axis = fig.add_axes([0.07, 0.20, 0.60, 0.55])
    width = 0.34
    axis.bar(x - width / 2, [100 * dmel[t] for t in tasks], width,
             color=BLUE, label="Drosophila chrX (frame-aware)")
    axis.bar(x + width / 2, 100 * human.AUPRC.to_numpy(), width,
             color=RED, label="Human chr1 (current model)")
    axis.set_xticks(x, ["Donor", "Acceptor", "Start", "Stop"])
    axis.set_ylim(0, 100)
    axis.set_ylabel("AUPRC (%)")
    axis.legend(fontsize=9.5, frameon=True)
    style_axis(axis)

    add_box(fig, 0.72, 0.58, 0.23, 0.13,
            "Splice sites remain usable\nDonor 87.5%  |  Acceptor 84.4%",
            color="#EAF1FF", edge=BLUE, fontsize=14, weight="bold")
    add_box(fig, 0.72, 0.38, 0.23, 0.13,
            "Translation boundaries collapse\nStart 38.3%  |  Stop 20.3%",
            color="#FDE9E8", edge=RED, fontsize=14, weight="bold")
    fig.text(0.72, 0.22,
             "This motivates explicit coding-frame and\nboundary supervision. The cross-species\nbars are diagnostic, not a controlled pair.",
             fontsize=10.8, color=MUTED, weight="bold", linespacing=1.35)
    return fig


def slide_boundary_next():
    fig = base_slide(
        "Next experiment: concentrate CDS supervision near boundaries", 10,
        kicker="Ongoing",
        subtitle="Reduce redundant gradients from easy CDS interiors",
    )
    axis = fig.add_axes([0.07, 0.47, 0.86, 0.25])
    axis.set_xlim(-300, 300)
    axis.set_ylim(0, 1)
    axis.axvspan(-300, -128, color="#E9EDF3")
    axis.axvspan(-128, 128, color="#F9D99B")
    axis.axvspan(128, 300, color="#E9EDF3")
    axis.axvline(0, color=RED, linewidth=3)
    axis.text(0, 0.73, "CDS 0↔1 transition", ha="center", va="center",
              fontsize=15, weight="bold", color=RED)
    axis.text(0, 0.34, "boundary loss weight = 1.0", ha="center",
              fontsize=14, weight="bold", color=INK)
    axis.text(-215, 0.34, "far loss\nweight = 0.1", ha="center",
              fontsize=13, color=MUTED, weight="bold")
    axis.text(215, 0.34, "far loss\nweight = 0.1", ha="center",
              fontsize=13, color=MUTED, weight="bold")
    axis.set_xticks([-128, 0, 128], ["−128 bp", "boundary", "+128 bp"])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    add_box(fig, 0.07, 0.18, 0.39, 0.18,
            "Boundary-focused CDS loss\n\nL_CDS = mean(CE_boundary)\n+ 0.1 mean(CE_far)",
            color=WHITE, edge=ORANGE, fontsize=14.5, weight="bold")
    add_box(fig, 0.54, 0.18, 0.39, 0.18,
            "Controlled setting\n\nλCDS = 0.03 • 10 kb / 5 kb stride\nphase = 0.10 • dilation 3 • seed 42",
            color=WHITE, edge=BLUE, fontsize=15, weight="bold")
    fig.text(0.50, 0.105,
             "Hypothesis: retain the UniAnn sensitivity gain while reducing novel loci and recovering precision. Results pending.",
             ha="center", fontsize=12.5, color=TEAL, weight="bold")
    return fig


def slide_pr_curves():
    fig = base_slide(
        "Candidate-level PR–Sn curves support the held-out conclusion", 11,
        kicker="Supplementary result",
        subtitle="Phase-loss grid on chrX+; chrX excluded from training",
    )
    add_image(
        fig,
        ROOT / "results/phase_weight_grid_v1/heldout_four_task_prsn_curves.png",
        [0.035, 0.10, 0.63, 0.69],
    )
    add_box(fig, 0.69, 0.54, 0.25, 0.14,
            "Splice-site curves are nearly saturated",
            color="#EAF1FF", edge=BLUE, fontsize=14, weight="bold")
    add_box(fig, 0.69, 0.35, 0.25, 0.14,
            "Phase weight separates start/stop behavior",
            color="#FFF3DC", edge=ORANGE, fontsize=14, weight="bold")
    add_box(fig, 0.69, 0.16, 0.25, 0.14,
            "λphase = 0.20 degrades all four tasks",
            color="#FDE9E8", edge=RED, fontsize=14, weight="bold")
    return fig


def slide_takeaways():
    fig = base_slide(
        "Take-home messages", 12,
        kicker="Conclusions",
        subtitle="What is established, and what remains to be tested",
    )
    cards = [
        (0.06, 0.56, "1", "Frame awareness works",
         "Phase supervision produces the largest gains,\nespecially for start and stop codons.", BLUE),
        (0.52, 0.56, "2", "Context length is secondary",
         "Larger windows do not improve candidate or\nlocus metrics monotonically.", GREEN),
        (0.06, 0.28, "3", "Downstream metrics matter",
         "AUPRC gains do not automatically improve strict\nlocus precision after EviAnn merging.", ORANGE),
        (0.52, 0.28, "4", "Human transfer remains difficult",
         "The largest transfer failure is at translation\nboundaries, motivating boundary-focused training.", RED),
    ]
    for x, y, number, title, body, color in cards:
        add_box(fig, x, y, 0.40, 0.20, "", color=WHITE, edge=color, lw=1.8)
        fig.add_artist(Circle((x + 0.05, y + 0.14), 0.025,
                              transform=fig.transFigure, facecolor=color,
                              edgecolor="none", zorder=4))
        fig.text(x + 0.05, y + 0.14, number, ha="center", va="center",
                 color=WHITE, fontsize=14, weight="bold", zorder=5)
        fig.text(x + 0.09, y + 0.145, title, ha="left", va="center",
                 fontsize=16, weight="bold", color=INK)
        fig.text(x + 0.05, y + 0.075, body, ha="left", va="center",
                 fontsize=11.2, color=MUTED, linespacing=1.3)
    fig.text(0.50, 0.145,
             "Best completed Drosophila endpoint: EviAnn + UniAnn strict-locus F1 = 90.54%",
             ha="center", fontsize=17, weight="bold", color=INK)
    fig.text(0.50, 0.095,
             "Ongoing: chromosome-aware boundary-weighted CDS auxiliary training",
             ha="center", fontsize=12.5, weight="bold", color=TEAL)
    return fig


def make_contact_sheet(paths, output):
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((640, 360), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (660, 390), "white")
        canvas.paste(image, ((660 - image.width) // 2, 10))
        thumbs.append(canvas)
    rows = (len(thumbs) + 1) // 2
    sheet = Image.new("RGB", (1320, 390 * rows), GRID)
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((index % 2) * 660, (index // 2) * 390))
    sheet.save(output, quality=92)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pptx_path = OUT / f"{STEM}.pptx"
    pdf_path = OUT / f"{STEM}.pdf"
    preview_path = OUT / f"{STEM}_contact_sheet.png"
    builders = [
        slide_title,
        slide_evaluation_design,
        slide_architecture,
        slide_frame_ablation,
        slide_window_size,
        slide_phase_weight,
        slide_dense_cds,
        slide_best_locus,
        slide_human_gap,
        slide_boundary_next,
        slide_pr_curves,
        slide_takeaways,
    ]

    presentation = Presentation()
    presentation.slide_width = Inches(13.333)
    presentation.slide_height = Inches(7.5)
    presentation.core_properties.title = (
        "Frame-aware CNN–Mamba for gene-signal prediction"
    )
    presentation.core_properties.subject = (
        "Drosophila chrX evaluation, UniAnn integration, and human transfer"
    )
    presentation.core_properties.author = "CNN–Mamba project"

    with tempfile.TemporaryDirectory(prefix="cnn_mamba_results_ppt_") as temp_name:
        temp = Path(temp_name)
        pngs = []
        with PdfPages(pdf_path) as pdf:
            for index, builder in enumerate(builders, 1):
                fig = builder()
                png = temp / f"slide_{index:02d}.png"
                fig.savefig(png, dpi=144, facecolor=fig.get_facecolor(),
                            edgecolor="none")
                pdf.savefig(fig, facecolor=fig.get_facecolor(), edgecolor="none")
                plt.close(fig)
                pngs.append(png)

                slide = presentation.slides.add_slide(presentation.slide_layouts[6])
                slide.shapes.add_picture(
                    str(png), 0, 0,
                    width=presentation.slide_width,
                    height=presentation.slide_height,
                )
        presentation.save(pptx_path)
        make_contact_sheet(pngs, preview_path)

    print(pptx_path)
    print(pdf_path)
    print(preview_path)


if __name__ == "__main__":
    main()
