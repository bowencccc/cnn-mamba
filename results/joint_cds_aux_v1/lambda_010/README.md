# Joint CDS auxiliary experiment (`lambda_CDS = 0.10`)

Controlled comparison against the existing 10 kb CNN--Mamba-k7 model with
frame dilation 3 and phase-loss weight 0.10. The new model is trained from
scratch with

`L = 0.25 L_splice + L_start/stop + 0.10 L_phase + 0.10 L_CDS`.

The chrX-held-out model selected epoch 14 and stopped at epoch 17. The
chrX-in-training model selected epoch 12 and stopped at epoch 15.

## Candidate AUPRC

| Regime | Task | Phase auxiliary | + CDS auxiliary | Difference |
| --- | --- | ---: | ---: | ---: |
| Held-out | Donor | 0.955915 | 0.958579 | +0.002664 |
| Held-out | Acceptor | 0.952746 | 0.955750 | +0.003004 |
| Held-out | Start | 0.850984 | 0.864165 | +0.013181 |
| Held-out | Stop | 0.882287 | 0.881493 | -0.000794 |
| Held-out | **Mean** | **0.910483** | **0.914997** | **+0.004514** |
| chrX in training | Donor | 0.963165 | 0.961872 | -0.001293 |
| chrX in training | Acceptor | 0.957509 | 0.954217 | -0.003292 |
| chrX in training | Start | 0.881102 | 0.871672 | -0.009429 |
| chrX in training | Stop | 0.907998 | 0.900708 | -0.007290 |
| chrX in training | **Mean** | **0.927443** | **0.922117** | **-0.005326** |

## Strict locus results

| Method | Model | Sn | Pr | F1 |
| --- | --- | ---: | ---: | ---: |
| UniAnn | Phase auxiliary | 77.1 | 77.2 | 77.150 |
| UniAnn | + CDS auxiliary | 77.2 | 75.1 | 76.136 |
| EviAnn+UniAnn | Phase auxiliary | 91.6 | 89.5 | 90.538 |
| EviAnn+UniAnn | + CDS auxiliary | 91.4 | 87.8 | 89.564 |

The joint CDS loss improves honest held-out mean AUPRC, especially start codons,
but reduces the chrX-in-training candidate metrics and both strict locus F1
scores. At weight 0.10 it should therefore not replace the current production
model. The separate CDS head remains useful for calibrated genome-wide phase
output: held-out CDS AUPRC is 0.986013 and gated phase macro-AUPRC is 0.977377.
