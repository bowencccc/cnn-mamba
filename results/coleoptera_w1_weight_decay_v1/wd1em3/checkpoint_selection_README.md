# Checkpoint-selection comparison

This is a post-hoc comparison within the existing `weight_decay=1e-3`,
`seed=44` run. No model was retrained. The four-task-mean rule selected epoch
6. The codon-centric rule selected epoch 4 using:

`0.7 * harmonic_mean(start_AP, stop_AP) + 0.15 * donor_AP + 0.15 * acceptor_AP`

Both checkpoints were independently fitted with four-state vector scaling on
the 3L validation split, then evaluated on chrX+ with the same score export,
headered coding-frame TSV contract, UniAnn built-in transform, UniAnn commit
`08eaf1595faf469069b9fd28a42973b84ce1e27e`, `gffread -M`, and strict
`gffcompare` pipeline.

The codon-centric checkpoint reduced chrX mean four-task AUPRC from 0.888860
to 0.873298, but increased UniAnn locus F1 from 73.5593 to 73.9037 and
EviAnn+UniAnn locus F1 from 84.2630 to 85.2253. Replacing only seed 44 in the
three-seed aggregate (seeds 42 and 43 select the same epoch under both rules)
changes mean EviAnn+UniAnn F1 from 84.8267 to 85.1475 and its sample SD from
0.5390 to 0.2382. This is evidence that four-task mean AUPRC is not perfectly
aligned with the downstream objective, but it is not yet enough to tune a new
selection formula on chrX.

EviAnn alone exactly matched 667 of 1,070 reference loci. Relative to those
existing matches, epoch 6 recovered 202 of the 403 available gap loci (gap
precision 84.87%, sensitivity 50.12%, F1 63.03%), whereas epoch 4 recovered
231 (gap precision 81.34%, sensitivity 57.32%, F1 67.25%). The recommended
deployment selection metric is therefore downstream gap-locus F1 with a
baseline-retention guard, rather than the provisional codon-weighted site
formula. See `docs/DROSO_CHRX_DEPLOYMENT_SELECTION.md`.

Checkpoint SHA256:

- epoch 6: `b5493568eddd654322ba9d50c462f5096c7307836050d5c5fb59d94584b9a202`
- epoch 4: `515fd91361b4777b5f20a9ba05e9fc9689572065c5d01fb5ba2a2429690dd4fb`
