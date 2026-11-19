# Drosophila chrX deployment selection experiments

## Scope

These experiments use the 1 kb CNN--Mamba-k7 deployment regime in which chrX
is included in EviAnn-supervised training and is subsequently scored on the
chrX plus strand.  Unless stated otherwise, the fixed settings are seed 44,
AdamW weight decay `1e-3`, candidate negative weight `1.0`, dropout `0`, 500 bp
stride, four-state auxiliary weight `0.1`, noncoding weight `0.1`, and frame
dilation 3.  Each downstream run uses vector scaling followed by UniAnn's
built-in transform and strict `gffcompare` evaluation.

All epoch checkpoints are retained.  Checkpoint averaging is an arithmetic
mean of compatible `model_state_dict` floating tensors; optimizer and RNG
states are deliberately omitted from the averaged inference checkpoint.

## Gap-locus metric

EviAnn alone has 1,070 reference loci, 740 predicted loci, and 667 strict
matched loci on this benchmark.  Therefore 403 reference loci are available
for recovery.

For a UniAnn experiment:

- recovered Gap loci are strict reference loci matched by EviAnn+UniAnn but
  not by EviAnn alone;
- Gap sensitivity is recovered loci divided by 403;
- Gap precision is recovered loci divided by all loci predicted by UniAnn
  alone after `gffread -M`;
- Gap F1 is the harmonic mean of those two values.

Do **not** use `union_predicted_loci - eviann_predicted_loci` as the precision
denominator.  `gffread -M` can merge EviAnn and UniAnn loci, so that net count
difference is only a diagnostic and can be smaller than the number of
recovered loci.  Earlier exploratory reports using that denominator (for
example Gap F1 0.6977 for the seed-44 baseline) are superseded.  The corrected
seed-44 value is 0.3419.  Standard UniAnn and EviAnn+UniAnn strict locus
precision, sensitivity, and F1 were never affected by this correction.

## Results

### Checkpoint averaging (seed 44)

| Average | Gap P | Gap Sn | Gap F1 | EviAnn+UniAnn F1 |
|---|---:|---:|---:|---:|
| epochs 4--6 | 0.2370 | 0.6129 | **0.3419** | **86.14** |
| epochs 3--5 | 0.2329 | 0.5931 | 0.3345 | 85.98 |
| epochs 4--5 | 0.2314 | 0.6030 | 0.3345 | 85.60 |

Epochs 4--6 are the preferred averaging range.

### Candidate negative weight (seed 44)

Each weight was allowed to select the best of raw epochs 4--6 and their
4--6 average by corrected Gap F1.

| Negative weight | Selected checkpoint | Gap P | Gap Sn | Gap F1 | EviAnn+UniAnn F1 |
|---:|---|---:|---:|---:|---:|
| **1.0** | epochs 4--6 average | 0.2370 | 0.6129 | **0.3419** | **86.14** |
| 0.7 | epoch 5 | 0.2215 | 0.5732 | 0.3195 | 85.03 |
| 0.5 | epoch 6 | 0.2135 | 0.5658 | 0.3100 | 84.20 |

Downweighting EviAnn-unlabeled canonical candidates did not improve this
deployment benchmark.

### Three seeds at negative weight 1.0

All seeds use epochs 4--6 averaging.

| Seed | Gap P | Gap Sn | Gap F1 | EviAnn+UniAnn F1 |
|---:|---:|---:|---:|---:|
| 42 | 0.2402 | 0.6377 | 0.3489 | 86.25 |
| 43 | 0.2435 | 0.6501 | 0.3543 | 86.50 |
| 44 | 0.2370 | 0.6129 | 0.3419 | 86.14 |
| mean +/- SD | 0.2402 +/- 0.0032 | 0.6336 +/- 0.0190 | 0.3484 +/- 0.0062 | 86.30 +/- 0.18 |

### Dropout 0.05 (seed 44)

The dropout run early-stopped at epoch 5, so downstream selection used raw
epochs 3--5 and their 3--5 average.  Its best Gap checkpoint was epoch 5.

| Dropout | Selected checkpoint | Gap P | Gap Sn | Gap F1 | EviAnn+UniAnn F1 |
|---:|---|---:|---:|---:|---:|
| **0** | epochs 4--6 average | 0.2370 | 0.6129 | **0.3419** | **86.14** |
| 0.05 | epoch 5 | 0.1641 | 0.3424 | 0.2219 | 80.46 |

Dropout 0.05 substantially reduced sensitivity and should not be adopted.

## Current recommendation

Use weight decay `1e-3`, candidate negative weight `1.0`, dropout `0`, and an
arithmetic average of epochs 4--6.  Candidate mean AP is retained as a
diagnostic, but deployment selection should prioritize corrected Gap-locus F1
and report the standard strict EviAnn+UniAnn locus metrics alongside it.
