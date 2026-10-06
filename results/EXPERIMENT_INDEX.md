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

For the latest four-state experiment, `joint_frame_grid_v1/summary/` contains
the cross-model candidate AUPRC table, held-out and chrX-in-training PR--Sn
figures, CDS/frame metrics, and UniAnn/EviAnn+UniAnn strict locus comparison.
The best four-state UniAnn-only setting was noncoding weight 0.10: Sn 80.4,
Pr 79.6, F1 80.00.
