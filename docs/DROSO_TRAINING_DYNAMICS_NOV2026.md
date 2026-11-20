# Drosophila chrX-included deployment: training dynamics and model selection (Nov 2026)

Summary of the experiments run 2026-11-17 to 2026-11-20 on HomeLab (fixed-epoch
learning-rate sweeps, per-isoform windows, Gap-locus selection on 3L) and on
cybertron (weight decay, checkpoint averaging, negative weight, dropout 0.05;
see `DROSO_CHRX_DEPLOYMENT_SELECTION.md` and `DROSO_GAP_LOCUS_SELECTION.md`).

## Common setting

1 kb windows on a 500 bp grid, Coleoptera-evidence EviAnn labels, chrX included
in training (deployment regime, not held-out chrX), CNN--Mamba-k7 (4.4M
parameters), bf16, effective batch 80, dropout 0, AdamW (betas 0.9/0.95, weight
decay 1e-4 unless stated), gradient clipping 1.0, linear warmup of 200 steps
(~13% of the 1,558-step union epoch) then cosine to 0.1x peak. Validation is
3L, scored as the mean of donor/acceptor/start/stop AP against RefSeq.
Downstream: four-state vector scaling fitted on the 3L validation windows,
UniAnn `08eaf15` with its built-in coding-frame transform, strict `gffcompare`
on chrX+ (1,070 reference loci; EviAnn alone matches 667, leaving 403 gap loci).
Seeds 42--46 unless stated. All values are mean +/- SD over seeds.

## 1. Fixed 15 epochs: the model overfits after epoch 5--8 at every learning rate

Fixed-15 runs (no early stopping) were evaluated at the best-validation epoch
and at epoch 15. Reference: early stopping (30-epoch cosine horizon, patience
3, peak lr 3e-4) gives chrX+ AUPRC 0.893 +/- 0.010, UniAnn F1 76.5 +/- 1.1,
EviAnn+UniAnn F1 85.3 +/- 1.0 (best epoch 8 +/- 3.5).

| Peak lr | Windows | Best epoch | AUPRC best -> ep15 | UniAnn F1 best -> ep15 | EviAnn+UniAnn F1 best -> ep15 |
|---|---|---:|---|---|---|
| 3e-4 | union | 7.2 | 0.899 -> 0.840 | 76.0 +/- 1.3 -> 72.1 +/- 2.6 | 84.8 +/- 1.2 -> 79.5 +/- 2.7 |
| 1.5e-4 | union | 4.6 | 0.888 -> 0.830 | 73.1 +/- 3.6 -> 70.1 +/- 0.8 | 84.1 +/- 0.4 -> 77.8 +/- 0.7 |
| 1e-4 | union | 5.8 | 0.886 -> 0.836 | 74.2 +/- 0.8 -> 69.8 +/- 0.5 | 83.9 +/- 0.5 -> 77.5 +/- 0.4 |
| 3e-4 | per-isoform | 6.2 | 0.895 -> 0.884 | 75.8 +/- 0.5 -> 74.2 +/- 2.2 | 85.0 +/- 0.6 -> 81.5 +/- 2.3 |

- Validation loss rises after epoch 5--8 in every sweep while training loss
  keeps falling. Lowering the learning rate makes runs more consistent but not
  better: it does not remove the overfitting, it only slows the approach to it.
- Replaying early stopping (patience 2 or 3, min_delta 1e-4) on the fixed-15
  histories selects the global-best epoch for every 1.5e-4 and 1e-4 run; at
  3e-4 only seed 44 with patience 2 differs (epoch 4 instead of 7).
- The low UniAnn outlier at 1.5e-4 is seed 44 (best epoch 3, UniAnn F1 67.0);
  without it the mean is 74.7 +/- 1.5.
- Interpretation: RefSeq sites missing from EviAnn are trained as negatives.
  Broad sequence rules are learned in the first few epochs; later epochs
  increasingly memorise EviAnn omissions. Early stopping is a necessary
  regulariser here, not a convenience.

Figures: `results/coleoptera_w1_fixed15_lr_comparison/`,
`results/coleoptera_w1_seed_sweep_fixed15*_v1/figures/`.

## 2. Per-isoform windows: same peak, more stable

Union windows (`prepare_droso.py`) merge all EviAnn coding transcripts of a
locus into one label track (isoform phase conflicts masked). Per-isoform
windows (`prepare_droso_per_isoform.py`, requested by A. Zimin) tile each
transcript separately on the same grid; labels come from that transcript plus
other loci, and exact duplicates are removed. Training windows: 139,195 vs
124,655 (+11.7%; 11,747 grid positions carry more than one label version).
Validation windows are unchanged. `scripts/check_per_isoform_windows.py`
verifies that positions, sequence and reference arrays match the union data and
that the union of label versions equals the union labels.

- Best-validation checkpoint: tie with union (EviAnn+UniAnn 85.0 vs 84.8), but
  the seed SD halves (0.6 vs 1.2; UniAnn 0.5 vs 1.3).
- Epoch 15: clearly better (EviAnn+UniAnn 81.5 vs 79.5, AUPRC 0.884 +/- 0.007 vs
  0.840 +/- 0.024). Overfitting still happens but is slower; per-isoform labels
  act as a mild label-level augmentation.
- Paired by seed, the epoch-15 gain is positive in 4 of 5 seeds; the
  best-epoch difference is within noise.

Figures: `results/coleoptera_w1_fixed15_perisoform_comparison/`.

## 3. cybertron: weight decay, averaging, negative weight, dropout

From `DROSO_CHRX_DEPLOYMENT_SELECTION.md` and `DROSO_GAP_LOCUS_SELECTION.md`
(union windows, lr 1.5e-4, early stopping patience 2, seeds 42--44):

- Weight decay 1e-3 is best (validation mean AP 0.892 +/- 0.009 vs 0.886 +/-
  0.018 at 1e-4; 1e-2 hurts start codons). EviAnn+UniAnn 84.8 +/- 0.5 at the
  mean-AP-selected checkpoint.
- Averaging epochs 4--6: EviAnn+UniAnn 86.30 +/- 0.18, Gap F1 0.348 +/- 0.006.
  The 4--6 range was chosen by chrX Gap F1 (see section 4 for why this is
  optimistic).
- Candidate negative weight 0.7 or 0.5: worse than 1.0 (seed 44).
- Dropout 0.05: much worse (EviAnn+UniAnn 80.5, sensitivity drops).

## 4. Is Gap-locus F1 a better selection metric than 3L mean AP?

Gap F1 targets the deployment objective (recovering reference loci EviAnn
misses), but computing it on chrX selects on the reporting chromosome. To
test a clean version, `scripts/run_gap_selection_eval.sh` ran UniAnn on 3L+
(validation; RefSeq 1,351 loci, EviAnn alone matches 822) and chrX+ for every
per-isoform fixed-15 checkpoint at epochs 3--10 (5 seeds, 80 UniAnn runs,
no retraining).

chrX+ outcome of each selection rule (epoch chosen per seed within 3--10):

| Selected by | Epochs | chrX Gap F1 | chrX EviAnn+UniAnn F1 |
|---|---|---:|---:|
| **3L mean AP** (current) | 7, 6, 5, 8, 5 | **0.323 +/- 0.015** | **85.03 +/- 0.61** |
| 3L Gap F1 | 5, 6, 8, 9, 5 | 0.320 +/- 0.025 | 84.84 +/- 1.44 |
| 3L EviAnn+UniAnn F1 | 10, 6, 8, 9, 5 | 0.316 +/- 0.023 | 84.75 +/- 1.39 |
| oracle: chrX itself | 5, 7, 5, 9, 3 | 0.332 +/- 0.012 | 85.49 +/- 0.67 |

Within-seed Spearman correlation across epochs with chrX Gap F1: 3L mean AP
+0.67 (positive in all seeds), 3L Gap F1 +0.43 (negative for seed 42).

- 3L mean AP is the better selector. Locus-level metrics on ~1,000 loci are
  too noisy to rank neighbouring epochs; site AP over hundreds of thousands of
  candidates is smooth.
- Even oracle selection on chrX gains only ~0.45 F1 over mean-AP selection.
  Numbers selected on chrX (including the 86.3 above) carry roughly that much
  optimistic bias and should be compared with mean-AP-selected results.
- UniAnn-alone F1 keeps rising with epoch (seed 42: 73.8 -> 78.4) while
  EviAnn+UniAnn and Gap F1 do not: the extra correct UniAnn models are mostly
  loci EviAnn already has.

Results: `results/coleoptera_w1_gap_selection_perisoform_v1/`
(`per_epoch.tsv`, `selection_summary.tsv`).

**Bug fix.** `uniann_evaluate.exact_reference_loci` counted matched loci by
refmap gene name. On 3L+ two genes can share one gffcompare locus (e.g. tipE /
CG18675), so the count disagreed with `gffcompare` stats (824 vs 822) and the
consistency check failed. It now clusters reference transcripts by exon
overlap, reproducing gffcompare exactly on both chromosomes. chrX+ results are
unchanged (e.g. 230 recovered gap loci for seed 44 epoch 5).

## Current recommendation

- Training data: per-isoform windows.
- Selection: early stopping on 3L mean AP (patience 2--3), not on chrX metrics.
  Keep Gap F1 and strict locus F1 as reported downstream metrics.
- Weight decay 1e-3, candidate negative weight 1.0, dropout 0.

## Next steps

1. **Combine the positive findings with clean selection:** per-isoform windows
   + weight decay 1e-3, 5 seeds, all epoch checkpoints kept. Compare (a) the
   3L-mean-AP best epoch and (b) an average of the best epoch +/- 1 chosen by
   3L mean AP, so averaging is selected without chrX.
2. **Report without selection bias:** if a chrX-based rule is kept, split the
   403 RefSeq gap loci into calibration and reporting halves, as proposed in
   `DROSO_CHRX_DEPLOYMENT_SELECTION.md`.
3. **Longest-CDS windows** (`coleoptera_eviann_w1_longestcds_v1`) are prepared
   but never trained; a third labelling strategy to compare with union and
   per-isoform if label construction remains of interest.
