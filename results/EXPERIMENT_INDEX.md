# Experiment result index

Git tracks compact, report-ready outputs: TSV/JSON summaries, PR--Sn curves,
loss plots, strict locus plots and presentations. Large checkpoints, raw score
arrays, per-base predictions and UniAnn temporary files are intentionally
external.

## Core experiments

- `window_ablation/`: legacy 5/10/20/30/40/80 kb window comparison.
- `early_stopping_v1/`: early-stopped window comparison for held-out chrX and
  chrX-in-training/UniAnn evaluation.
- `frame_ablation_v1/` and `phase_weight_grid_v1/`: dilation-3 coding-frame
  branch and masked three-class phase-loss experiments.
- `cds_gating_v1/` and `joint_cds_aux_v1/`: frozen and jointly trained binary
  CDS heads.
- `cds_weight_grid_v1/`: dense CDS auxiliary weights 0.03 and 0.05 with
  shared-backbone gradient diagnostics.
- `cds_boundary_r128_v1/`: CDS loss concentrated within 128 bp of annotated
  CDS boundaries.
- `joint_frame_grid_v1/`: four-state `non_CDS/frame0/frame1/frame2` auxiliary
  head with noncoding weights 0.03, 0.10 and 0.30.
- `human_cnn_mamba_k7/`: current Human pipeline results.
- `droso_human_pretrain_comparison_v1/`: random initialization versus Human
  MLM-pretrained initialization on the matched Drosophila task.
- `mlm_transfer_v1/` and `mlm_transfer_regularization_v1/`: Human MLM transfer
  and retention controls (backbone LR multiplier and staged freezing).
- `incomplete_label_v1/`, `incomplete_label_v2/`, and `incomplete_label_v3/`:
  candidate-negative weighting and propensity-aware incomplete-label tests.
- `frame_scaling_extended_grid_v1/`: fixed-temperature, learned vector-scaling
  and UniAnn downstream calibration comparisons.
- `coleoptera_eviann_v1/`: Dmel annotation/training experiment using
  Coleoptera proteins as EviAnn evidence.
- `evidence_source_comparison_v1/`: report-ready Drosophila-versus-Coleoptera
  evidence PR--Sn curves, EviAnn operating points, four-state metrics and
  strict locus comparison.

For the latest four-state experiment, `joint_frame_grid_v1/summary/` contains
the cross-model candidate AUPRC table, held-out and chrX-in-training PR--Sn
figures, CDS/frame metrics, and UniAnn/EviAnn+UniAnn strict locus comparison.
The best four-state UniAnn-only setting was noncoding weight 0.10: Sn 80.4,
Pr 79.6, F1 80.00.

## Protein-evidence source experiment

The matched comparison holds the 10 kb CNN--Mamba-k7 architecture, chrX-in-
training regime and RefSeq chrX+ evaluation fixed while changing the EviAnn
protein evidence source. Mean four-task candidate AUPRC was 0.932868 with the
original Drosophila evidence and 0.907035 with Coleoptera evidence. Matched
EviAnn+UniAnn strict locus F1 was 90.399 and 83.585, respectively.

The cross-annotation diagnostic is also informative: augmenting the sparse
Coleoptera-evidence EviAnn GFF with the stronger Drosophila-evidence model gave
strict locus Sn 87.2, Pr 88.3 and F1 87.747. This exceeded retraining on the
sparser Coleoptera pseudo-labels, consistent with false-negative label noise.
