# Human EviAnn evidence reliability decision

The agreement analysis uses exact canonical CDS donor, acceptor, start, and
stop positions on RefSeq validation chromosomes `chr2/5/9/10/22`. It does not
use chr1. `LOW_QUALITY_PROTEIN` is a flag overlapping the EviAnn evidence
categories, so training sites must be assigned to the strongest supporting
transcript stratum before weighting.

The locked first-pass positive-label reliability weights are:

| Evidence stratum | donor | acceptor | start | stop |
|---|---:|---:|---:|---:|
| complete, high quality | 1.00 | 1.00 | 1.00 | 1.00 |
| complete, low quality | 0.80 | 0.80 | 0.65 | 0.70 |
| protein-only, high quality | 0.85 | 0.85 | 0.70 | 0.65 |
| protein-only, low quality | 0.70 | 0.70 | 0.55 | 0.65 |
| transcript-only | 0.15 | 0.15 | 0.05 | 0.05 |

For a site supported by multiple transcripts, use the maximum applicable
weight. These weights apply only to positive labels; they do not solve the
positive-unlabeled false-negative problem and must not be interpreted as task
loss coefficients.

See `site_agreement.tsv`, `exclusive_site_support_agreement.tsv`,
`recommended_positive_weights.tsv`, and `metadata.json` for the measurements
and exact definitions.
