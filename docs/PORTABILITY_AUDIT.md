# Portability audit

Audit date: 2026-11-11

## Findings before cleanup

- The installable package under `src/cnn_mamba/` contained no user-home paths.
- `scripts/` contained 110 Python/Bash/Perl files mixing stable workflows,
  one-off experiments, cluster launchers, and plots.
- 26 scripts referenced `/home/bowencccc`, a particular Conda interpreter, or
  a sibling UniAnn/AltAnn checkout directly.
- UniAnn union evaluation concatenated inputs before format normalization. A
  mixed GTF/GFF3 union could therefore silently omit one input.
- There was no single top-level CLI or versioned multi-stage configuration.

## Changes in version 0.2.0

- Added the unified `cnn-mamba` CLI and a JSON workflow runner.
- Added `cnn-mamba doctor` for environment, CUDA, dependency, and UniAnn checks.
- Added an explicit UniAnn-emission builder; calibration and gating choices are
  recorded instead of being hidden in an experiment script.
- Removed user/server-specific absolute paths from all supported source,
  scripts, and configurations. External tools now use `UNIANN_ROOT`,
  `ALTANN_ROOT`, and `CNN_MAMBA_PYTHON`.
- Fixed EviAnn+UniAnn union construction by normalizing each source separately
  to GTF before concatenation and `gffread -M`.
- Documented the software/research-artifact boundary and the heldout/deployment
  distinction.

## Intentionally retained provenance

Files under `results/` and `external_manifest.tsv` can contain original absolute
paths. Those strings document where an old run or external file came from; they
are not runtime defaults and do not need to exist on a new machine.

## Remaining limitations

- Historical experiment scripts still encode experiment names, checkpoint
  layouts, chromosome accessions, and biological split choices. They are
  reproducibility records, not general APIs.
- Human whole-chromosome UniAnn currently requires chunked orchestration. The
  existing Human chunk runner should be promoted into the package before it is
  advertised as a stable cross-species command.
- Mamba3 and causal-conv1d installation remains CUDA/PyTorch-version sensitive.
- Unit tests cover portable logic; full training and chromosome-wide tests need
  a GPU and external genomes, so they are not suitable for lightweight CI.
