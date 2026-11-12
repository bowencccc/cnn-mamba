# UniAnn header-contract audit and rerun

Date: 2026-11-12

This audit follows the collaborator-facing interface exactly: the exported
four-state TSV keeps its single standard header, site scores are unchanged,
UniAnn performs its built-in transform, predictions are merged with
`gffread -M`, and locus metrics use strict `gffcompare`.

## What required rerunning

Runs that converted the four-state NPY directly into a headerless UniAnn
emission file with the legacy `(position - phase - 1) mod 3` mapping were
phase-misaligned. The affected report-ready groups were:

- Drosophila 500 bp, 1 kb, 2 kb, 5 kb, and 10 kb vector + built-in window
  comparison;
- Coleoptera-evidence retrained-model vector + built-in evaluation;
- Human chr1 vector + built-in chunked evaluation.

The older small-window raw and joint-T9 **no-gate** files also used the legacy
headerless mapping. They are retained only as noncanonical diagnostics and are
not used in the corrected default report. Replacing them with a direct TSV run
would additionally enable UniAnn's built-in transform and gate, so it would be
a different calibration condition rather than a clean rerun. The maintained
default is vector scaling plus the built-in transform.

The Drosophila, Coleoptera, and Human reruns are complete.

## What was not invalidated

- model training, train/validation loss, checkpoints, and early stopping;
- donor/acceptor/start/stop candidate scores, AUPRC, and PR--Sn curves;
- PSAURON coding-frame experiments;
- UniAnn runs whose converter consumed the original TSV including its header;
- calibration fitting itself (raw, scalar temperature, or vector scaling).

The bug was at the four-state-to-UniAnn row/frame interface, not in the model
or the probability files.

The Human chunk orchestrator has legacy command-line options named
`--psauron-index` and `--psauron-csv`. In this rerun they receive the
CNN--Mamba vector-scaled four-state NPY and TSV, respectively; PSAURON is not
used. This is the chr1-included deployment model, so its chr1 result is labeled
`diagnostic_leaky_anchor`, not a held-out generalization estimate.

## Canonical files

- Corrected Drosophila metrics: `corrected_strict_locus_metrics.tsv`
- Affected/unaffected inventory: `audit_status.tsv`
- Old-versus-corrected comparison: `old_vs_corrected.tsv`
- Corrected Drosophila plot:
  `../droso_small_window_v1_four_state_t9/report/drosophila_window_test_vector_scaling_plus_uniann_builtin_locus_metrics.png`
- Contract and derivation: `../../docs/UNIANN_FRAME_TSV_CONTRACT.md`

Human chr1 (chr1-included deployment diagnostic) gives UniAnn Sn/Pr/F1
46.4/42.1/44.146 and EviAnn+UniAnn 90.7/68.8/78.247. EviAnn-only is
88.8/73.8/80.608, so augmentation still does not improve Human locus F1.
