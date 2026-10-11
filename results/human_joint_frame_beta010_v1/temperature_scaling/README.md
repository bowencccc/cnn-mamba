# Human four-state temperature scaling

This calibration uses the chr1-held-out Human four-state model's best
checkpoint (epoch 4). A single scalar temperature was fitted against CHESS
labels on the validation chromosomes (chr2/5/9/10/22); chr1 was not used.
Both strands were included. Only each 10 kb window's owned 5 kb center was
eligible, avoiding duplicated overlap bases. CDS bases with ambiguous phase
were excluded.

The fitted empirical-NLL temperature is **1.228817774**. The separate
group-balanced diagnostic optimum is 1.235624335, close enough to show that the
conclusion is not driven by the natural noncoding prevalence alone.

| Metric | Before (T=1) | After (T=1.2288) |
|---|---:|---:|
| NLL | 0.330856 | 0.322067 |
| Brier score | 0.166834 | 0.165977 |
| ECE | 0.020826 | 0.004903 |
| Mean max probability | 0.908164 | 0.883610 |
| Median max probability | 0.988912 | 0.971418 |
| Fraction max probability >= 0.99 | 0.487940 | 0.323018 |
| Fraction max probability >= 0.999 | 0.138237 | 0.009921 |
| Accuracy | 0.887338 | 0.887338 |

The calibration sample contains 2,193,082 bases: 2,144,725 noncoding and
48,357 coding. Scalar temperature scaling preserved the predicted class at
every sampled base (`argmax_changes = 0`). Consequently, argmax accuracy and
rank-based AUPRC/PR curves do not change; calibrated probabilities and metrics
that use fixed probability thresholds do change.

Files:

- `temperature.json`: complete provenance, temperatures, metrics and
  reliability bins;
- `calibration_metrics.tsv`: compact before/after table;
- `calibration.png`: reliability, confidence distribution and metric plots;
- `calibration_sample.npz`: ignored by Git; reusable raw logits and labels;
- `run.log`: inference progress and terminal summary.
