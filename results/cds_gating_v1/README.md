# Frozen CDS-head gating experiment

This experiment starts from the chrX-in-training CNN--Mamba-k7 checkpoint with
frame dilation 3 and phase-loss weight 0.10. Every existing parameter is frozen;
only a new linear CDS/non-CDS head is trained. Training uses EviAnn CDS unions
from train chromosomes plus chrX, while early stopping uses FlyBase CDS unions
on chromosome 3L. The run stopped at epoch 6 and selected epoch 3.

## Validation CDS head

| Metric | Value |
| --- | ---: |
| CDS AUPRC | 0.985418 |
| CDS F1 | 0.944934 |
| Precision | 0.951122 |
| Sensitivity | 0.938826 |

## chrX+ full-chromosome phase evaluation

Gated scores are defined as
`P(CDS) * P(phase_k | CDS)`. Metrics use 4,096-bin bounded-memory AUPRC and the
FlyBase reference, excluding 1,343 bases with conflicting annotated phases.

| Target | Raw phase AUPRC | CDS-gated AUPRC | Difference |
| --- | ---: | ---: | ---: |
| Phase 0 | 0.688398 | 0.969885 | +0.281487 |
| Phase 1 | 0.722618 | 0.971460 | +0.248842 |
| Phase 2 | 0.775487 | 0.968967 | +0.193480 |
| **Macro** | **0.728834** | **0.970104** | **+0.241270** |

On chrX+, the CDS head has AUPRC 0.977001 and best-threshold F1 0.940117.
The four-way `non-CDS / phase-0 / phase-1 / phase-2` joint argmax has macro-F1
0.947299. The old raw phase probabilities are bit-identical to the source run
(maximum absolute difference 0.0), and all source-model parameters were verified
unchanged.

Because chrX participates in CDS-head training, the chrX+ numbers are an
in-training functional test rather than an unbiased held-out estimate. Gating
changes genome-wide phase-vs-non-CDS ranking, but multiplying all three phase
probabilities by the same CDS probability cannot change the winning phase at a
given base and does not by itself alter splice/start/stop or UniAnn output.
