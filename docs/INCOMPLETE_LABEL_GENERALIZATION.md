# Generalizing beyond incomplete EviAnn annotations

## Problem definition

The observed training label is whether EviAnn annotated a site (`s`), while
the desired biological target is whether the reference annotation considers
the site real (`y`). An unlabeled canonical candidate is therefore not a
confirmed negative:

```text
s = 1  -> high-confidence EviAnn positive
s = 0  -> mixture of true negatives and real sites missed by EviAnn
```

Ordinary cross-entropy incorrectly trains every `s = 0` candidate as negative.
The model can consequently learn `P(EviAnn detects the site | sequence)` rather
than `P(the site is real | sequence)`. EviAnn's selection process is unlikely
to be random: weak, unusual, or poorly supported sites can be systematically
underrepresented. This is a selected-at-random PU problem, not a standard
fully supervised classification problem.

## Leakage-safe evaluation

For held-out chromosomes, split reference positives into:

1. `Reference intersection EviAnn`: supported/easier positives.
2. `Reference minus EviAnn`: missed positives that define the main objective.
3. `Canonical candidates minus Reference`: evaluation negatives.

Report overall AUPRC, supported-site AUPRC, missed-site AUPRC, missed-site
recall at fixed precision, and downstream UniAnn strict locus Sn/Pr/F1. Never
use held-out reference labels to construct training weights or pseudo-labels.
The chrX-in-training runs remain downstream annotation diagnostics, not honest
generalization estimates.

## Ordered experiment plan

All ablations keep CNN-Mamba-k7, 10 kb windows, 5 kb stride, seed 42, batch
`4 x 2`, learning rate, scheduler, four-state head (`lambda=0.1`, `beta=0.1`),
early stopping, and preprocessing fixed unless explicitly stated.

1. **Unlabeled-negative weighting.** Sweep candidate negative weight
   `alpha = 1.0, 0.5, 0.25, 0.1`. EviAnn positives retain weight 1.0. Select
   with held-out reference mean AUPRC, but judge success primarily on
   `Reference minus EviAnn` recovery at controlled precision. Run downstream
   chrX-in-training UniAnn only for a promising setting.
2. **Propensity-aware soft weighting.** Estimate which unlabeled candidates
   are likely EviAnn false negatives from leakage-safe structural/evidence
   features, then reduce their negative weight or assign soft targets.
3. **Out-of-fold self-training.** Generate pseudo-positives only from models
   that did not train on the target fold; require agreement across models and
   motif/frame/structure constraints to limit confirmation bias.
4. **Transfer retention controls.** If cross-species initialization remains of
   interest, compare zero-shot scoring, frozen-backbone training, and
   discriminative learning rates. Full fine-tuning at the original learning
   rate can erase transferred features.

## Decision rule

An experiment is useful only if it increases recovery of
`Reference minus EviAnn` sites without an unacceptable precision loss and the
gain survives strict downstream gene-model evaluation. An increase driven
only by sites already present in EviAnn is not considered improved
generalization.
