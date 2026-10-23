# Structured HMM calibration pilot

This experiment freezes the Drosophila chrX-included CNN--Mamba-k7
four-state checkpoint and fits only a scalar joint-frame logit multiplier plus
four centered class biases. Splice and start/stop logits are not rescaled.

The structured layer has 15 states: intergenic, three intron phases, three
exon phases, start, stop, three donor phases, and three acceptor phases. The
loss is `log Z(all paths) - log Z(paths compatible with EviAnn positives)`;
unannotated positions are not labelled negative.

## Pilot configuration

- checkpoint: `runs/joint_frame_grid_v1/b010/chrxtrain/w10/best_model.pt`
- data: `data/processed/cds_gating/w10_frame`
- train/validation windows: 256/64
- structured crop length: 512 bp
- initial joint-frame temperature: 9
- early stopping patience: 5

## Result

The epoch-0 T=9 calibration had validation partial NLL 3.0641. No trained
epoch improved it; the closest was epoch 2 at 3.2875, so early stopping
restored epoch 0. The retained calibration is therefore exactly scalar T=9
with zero class bias.

A fixed-temperature diagnostic showed that the positive-only partial NLL
kept decreasing through T=20 (1.8413). This is a degeneracy: with no negative
or reference-gap supervision, making the frame distribution more uniform
makes annotated paths easier to retain but does not select the temperature
that maximizes downstream UniAnn accuracy.

Because the retained output is identical to the existing T=9 score file, its
strict locus results are unchanged:

| Method | Sn | Pr | F1 |
|---|---:|---:|---:|
| UniAnn | 77.9 | 79.9 | 78.8873 |
| EviAnn+UniAnn | 90.7 | 90.1 | 90.3990 |

The main conclusion is that the state topology is usable, but positive-only
partial likelihood is not by itself a suitable calibration-selection
objective for the incomplete-EviAnn setting. A next structured experiment
needs an additional anchor, such as held-out reference-gap positives,
propensity-weighted reliable negatives, or direct optimization of a
differentiable surrogate for downstream structure quality.
