# Drosophila chrX-included deployment model selection

## Objective and regime

The deployment experiment intentionally trains on EviAnn annotations from
Drosophila chrX and then runs CNN--Mamba, UniAnn, and the EviAnn+UniAnn union on
chrX+. Its objective is not held-out chromosome generalization. It asks whether
a model trained from incomplete EviAnn labels can recover structures supported
by RefSeq but absent from EviAnn in the intended annotation workflow.

The distinction between labels is essential:

- EviAnn is the training annotation;
- RefSeq is the external evaluation annotation;
- an EviAnn-unlabeled canonical candidate is not a confirmed biological
  negative;
- chrX is therefore correctly included in the deployment training set, but the
  result must be named `chrX-included deployment`, not held-out chrX.

## Why four-task mean AUPRC is not the primary selection metric

Overall donor/acceptor/start/stop AUPRC rewards prediction of RefSeq sites that
EviAnn already contains. Those sites do not add information to the final union.
Site-level AUPRC also does not measure whether splice sites, codons, and coding
frame emissions form a compatible gene model after UniAnn's nonlinear scoring
and dynamic programming.

Keep the four PR curves and AUPRC values as diagnostics. Select a deployment
checkpoint by its downstream recovery of EviAnn annotation gaps.

## Gap-locus metric

Run strict `gffcompare` for EviAnn alone and for EviAnn+UniAnn. Let:

- `R` be the set of reference loci;
- `M_E` be reference loci exactly matched by EviAnn;
- `M_U` be reference loci exactly matched by EviAnn+UniAnn;
- `L_E` and `L_U` be the respective numbers of predicted loci.

Define:

```text
recovered_gap_loci = |M_U - M_E|
available_gap_loci = |R - M_E|
added_loci         = max(L_U - L_E, 0)

gap_sensitivity = recovered_gap_loci / available_gap_loci
gap_precision   = recovered_gap_loci / added_loci
gap_F1          = harmonic_mean(gap_precision, gap_sensitivity)
```

Also require baseline retention
`|M_U intersect M_E| / |M_E| >= 0.995`. Report ordinary union locus Sn, Pr,
and F1 beside this gap-specific metric. If two checkpoints have practically
equal gap F1, prefer the earlier checkpoint or the one with higher gap
precision.

For a publication-grade estimate that is independent of model selection, the
training set may still contain all chrX EviAnn labels while external RefSeq
gap loci are divided into calibration and reporting subsets. This separates
reference-informed checkpoint selection from final reporting without changing
the intended EviAnn deployment regime.

## Weight-decay and seed experiment

The controlled experiment used 1 kb windows, 500 bp stride, chrX included,
Coleoptera-evidence EviAnn labels, CNN--Mamba-k7, BF16, effective batch size 80,
learning rate `1.5e-4`, dropout 0, maximum 15 epochs, early-stopping patience 2,
and seeds 42/43/44. Only AdamW weight decay changed.

| Weight decay | Validation mean AP | Seed SD | Mean best epoch |
|---:|---:|---:|---:|
| `1e-4` | 0.886259 | 0.018257 | 5.33 |
| `1e-3` | **0.891639** | **0.009168** | 5.33 |
| `1e-2` | 0.881644 | 0.012803 | 6.33 |

`1e-3` gave the best mean and approximately halved the seed standard
deviation relative to `1e-4`. `1e-2` was too strong, especially for start
codons. The selected `1e-3` runs produced mean EviAnn+UniAnn locus F1 84.8267
with SD 0.5390 when checkpoints were chosen by four-task validation mean AP.

## Checkpoint-selection counterfactual

Within the existing `weight_decay=1e-3`, seed-44 run, four-task validation mean
AP selected epoch 6. A codon-oriented validation rule selected epoch 4. Both
checkpoints were independently vector-calibrated and evaluated with the same
headered coding-frame TSV, UniAnn built-in transform, UniAnn commit `08eaf15`,
`gffread -M`, and strict `gffcompare`.

EviAnn alone exactly matched 667 of 1,070 reference loci, leaving 403 gap loci.

| Metric | Epoch 6 | Epoch 4 |
|---|---:|---:|
| chrX four-task mean AUPRC | **0.888860** | 0.873298 |
| Gap loci recovered | 202 | **231** |
| Gap precision | **84.87%** | 81.34% |
| Gap sensitivity | 50.12% | **57.32%** |
| Gap F1 | 63.03% | **67.25%** |
| EviAnn+UniAnn locus F1 | 84.2630 | **85.2253** |

This is direct evidence that overall candidate AUPRC and the deployment
objective are not perfectly aligned. Replacing only seed 44 with the
gap-aligned checkpoint changes the three-seed mean EviAnn+UniAnn F1 from
84.8267 to 85.1475 and its SD from 0.5390 to 0.2382. This counterfactual is
supportive, not a reason to tune an arbitrary site-level weighting formula on
one seed; the primary comparison should be downstream gap-locus F1.

## Why validation peaks after five or six epochs

The 1 kb training split contains approximately 124,655 windows when chrX is
included. With effective batch size 80, one epoch is about 1,558 optimizer
updates, so five to six epochs already correspond to roughly 7,800--9,350
updates. Fast convergence is therefore not itself a failure.

More importantly, real RefSeq sites missing from EviAnn are presented to the
supervised loss as negatives. Early training can learn broad sequence rules;
later training can increasingly memorize EviAnn-specific omissions. Early
stopping is an appropriate regularizer for this incomplete-label setting.

Recommended order of follow-up experiments:

1. save every epoch, cap training at about 10 epochs, and select with downstream
   gap-locus F1 subject to the baseline-retention guard;
2. evaluate weight-averaged checkpoints from epochs 4--5, 4--6, and 3--5;
3. keep weight decay `1e-3` and test canonical-candidate negative weights
   `1.0`, `0.7`, and `0.5`, leaving EviAnn positives and ordinary background
   unchanged;
4. only after that, test one small regularizer at a time, such as dropout 0.05,
   label smoothing 0.02, or effective batch size 40;
5. align the learning-rate schedule with the observed useful training horizon,
   for example maximum 10 epochs and an 8-epoch cosine schedule.

The aim is not to force training to last longer. It is to reduce confidence in
unlabeled canonical candidates as negatives and select the point before the
model memorizes annotation incompleteness.

## Versioned outputs

- `results/coleoptera_w1_weight_decay_v1/validation_summary.tsv`
- `results/coleoptera_w1_weight_decay_v1/weight_decay_seed_distribution.png`
- `results/coleoptera_w1_weight_decay_v1/wd1em3/per_seed.tsv`
- `results/coleoptera_w1_weight_decay_v1/wd1em3/seed_stats.tsv`
- `results/coleoptera_w1_weight_decay_v1/wd1em3/checkpoint_selection_comparison.tsv`
- `results/coleoptera_w1_weight_decay_v1/wd1em3/checkpoint_selection_README.md`
