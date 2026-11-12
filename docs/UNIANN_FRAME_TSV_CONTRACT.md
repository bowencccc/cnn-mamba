# CNN--Mamba joint-frame TSV contract for UniAnn

## Required rule

Keep the first line of every exported joint-frame TSV and pass the file to
UniAnn unchanged. Do not remove the header.

The exact required header is:

```text
chrom	position_1based	strand	frame0_probability	frame1_probability	frame2_probability	noncoding_probability	predicted_state
```

This rule applies to vector-scaled, scalar-temperature, and raw four-state TSVs.
The TSV remains a human-readable, one-data-row-per-base file; no change to
UniAnn is required.

## Why the header matters

CNN--Mamba defines phase `i` as codon nucleotide offset 0, 1, or 2. For a
zero-based genomic position `n`, that state belongs to UniAnn genomic frame:

```text
(n - i) mod 3
```

UniAnn commit `2f5c0ab` converts input row number `r` with:

```text
(r - i - 1) mod 3
```

Because UniAnn counts rather than parses the position column, the retained TSV
header is consumed as row 0 and makes the first nucleotide row have `r=1`.
This supplies the phase offset expected by the established direct-TSV
workflow. It also means that this particular UniAnn converter does not provide
a coordinate-aware interpretation of the TSV: it emits one placeholder row
for the header, consumes nucleotide rows with a one-row offset, and does not
use `position_1based`. This behavior is intentionally preserved because it is
the collaborator-facing workflow and reproduces the validated strict-locus
result. Removing the header instead rotates all three frame emissions and
produces the failed regression below.

This is an explicit compatibility contract, not permission to add arbitrary
lines. Exactly one header row is required, followed by one-based positions
starting at 1.

## Reproduced regression

The 10 kb Drosophila chrX-included beta-0.1 model with vector scaling,
unchanged site scores, UniAnn built-in transform, and strict comparison gives:

| Condition | Method | Sensitivity | Precision | F1 |
|---|---|---:|---:|---:|
| Header retained | UniAnn | 80.3 | 79.8 | 80.05 |
| Header retained | EviAnn+UniAnn | 91.2 | 90.2 | 90.70 |
| Header removed | UniAnn | 51.5 | 68.2 | 58.69 |
| Header removed | EviAnn+UniAnn | 87.6 | 90.5 | 89.03 |

The header-removed values are phase-misaligned and must not be reported as the
vector-scaling result.

The corrected Drosophila window comparison (vector scaling, unchanged four
site scores, UniAnn built-in transform, and strict locus comparison) is:

| Window | UniAnn Sn | UniAnn Pr | UniAnn F1 | EviAnn+UniAnn Sn | EviAnn+UniAnn Pr | EviAnn+UniAnn F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 500 bp | 80.0 | 76.4 | 78.16 | 91.7 | 86.8 | 89.18 |
| 1 kb | 79.6 | 77.2 | 78.38 | 91.3 | 87.8 | 89.52 |
| 2 kb | 81.1 | 78.2 | 79.62 | 91.9 | 88.5 | 90.17 |
| 5 kb | 70.7 | 73.5 | 72.07 | 89.6 | 89.7 | 89.65 |
| 10 kb | 80.3 | 79.8 | 80.05 | 91.2 | 90.2 | 90.70 |

These rows replace only the earlier NPY-to-headerless-emission window results.
Training losses, candidate-level AUPRC/PR--Sn curves, and any UniAnn run that
consumed the original one-header TSV are not changed by this correction.

## Canonical commands

Use `cnn-mamba export-joint-frame` to create the TSV. It always writes the
required header and records the contract in the adjacent `.complete.json`.

Use `cnn-mamba uniann-eval --require-cnn-mamba-tsv-header` for new experiments.
The wrapper checks the exact header, first one-based position, probability
range, probability sum, and sentinel metadata before invoking UniAnn. It fails
instead of silently accepting a stripped file.

When running `uniann.sh` directly for a collaborator, pass the exported TSV
without preprocessing:

```bash
"$UNIANN_ROOT/bin/uniann.sh" \
  -f NC_004354.4.fa \
  -c chrX_plus_joint_frame_probabilities_vector.tsv \
  -s chrX_plus_scores.tsv
```

## Migration to another server

After pulling the repository update, do not rewrite or strip an existing TSV.
The adjacent sentinel is recommended for newly exported files but is not
required for an older TSV that passes the header validator. Before a run,
verify:

```bash
head -n 2 chrX_plus_joint_frame_probabilities_vector.tsv
```

The first line must match the header above and the second line must begin with
`chrX`, position `1`, and the intended strand.

Historical runs that generated `out.ps.txt` directly from NPY files must be
audited for their frame mapping before reuse. The reliable discriminator is
the recorded mapping: CNN--Mamba phases require `(position-model_phase) mod 3`.
The coordinate-safe Human chunk adapter records schema
`joint_frame_header_aligned_uniann_emissions_v1` and uses that mapping; the
legacy schema `joint_frame_uniann_emissions_v1` uses the headerless mapping and
must not be used to represent a collaborator-facing headered-TSV run.
