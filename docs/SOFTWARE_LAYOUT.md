# Software and research-artifact boundary

## Stable software

- `src/cnn_mamba/`: installable implementation and public CLIs.
- `cnn-mamba`: unified command dispatcher.
- `cnn-mamba workflow`: versioned JSON workflow runner.
- `configs/*.example.json`: portable examples containing no server paths.
- `tests/`: unit and smoke tests for reusable behavior.

Changes in this layer should be species/configuration driven, tested, and free
of usernames, cluster names, and absolute data paths.

## External inputs and generated state

- `data/raw/`: logical input paths; large files are not tracked by Git.
- `data/processed/`: generated training windows.
- `runs/`: resumable training state.
- `artifacts/`: checkpoints, score arrays, and model outputs.
- `work/`: temporary downstream-tool workspaces.

These directories are not software APIs. Exact external inputs are described by
`external_manifest.tsv` and can be verified with `scripts/check_external.py`.

## Research record

- `scripts/`: historical experiment launchers and figure generators.
- `results/`: compact tables, plots, configuration records, and provenance.

Research scripts may encode an experiment's model/checkpoint names, but must use
repository-relative paths and environment variables for external software. New
reusable logic must be promoted into `src/cnn_mamba/` rather than copied between
experiment scripts.

## Evaluation regimes

The public examples name the two regimes explicitly:

- `heldout`: test chromosome excluded from training; use for candidate AUPRC.
- `deployment`: test chromosome included in training while validation remains
  separate; use for the intended EviAnn-to-reference UniAnn augmentation case.

These regimes must not share output directories or be compared without their
split being reported.

## Coding-frame calibration

The maintained UniAnn workflow fits classwise vector scaling on the validation
split and applies it to the four-state logits before softmax. UniAnn-compatible
emissions then explicitly apply the selected built-in-equivalent transform:
independent binary-odds scaling at `T=3` and the frame/non-CDS gate. Both stages
are written into output metadata; the older joint-softmax `T=9` recipe remains
only as a historical experiment.
