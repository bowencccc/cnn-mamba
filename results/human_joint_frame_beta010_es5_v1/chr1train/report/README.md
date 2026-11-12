# Human chr1 four-state + UniAnn report

This report uses the CNN--Mamba-k7 four-state auxiliary-head model trained with
chr1 included, selected at epoch 10. It is therefore a deployment/fit
diagnostic, not a held-out chromosome generalization result.

## Candidate-level evaluation

All four curves cover every canonical plus-strand candidate on GRCh38 chr1.
Reference and EviAnn sites are derived from plus-strand CDS models, deduplicated
across isoforms, and restricted to the same canonical candidate vocabulary.
The reference is `NC_000001.11.p.CDS.uniq.gff` (RefSeq CDS models).

| Task | AUPRC | Model best PR | Model best SN | Model best F1 | EviAnn PR | EviAnn SN | EviAnn F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Donor | 0.8715 | 85.8% | 81.4% | 83.6% | 93.6% | 92.9% | 93.3% |
| Acceptor | 0.8486 | 85.4% | 77.6% | 81.3% | 92.3% | 91.7% | 92.0% |
| Start | 0.4200 | 57.2% | 37.1% | 45.1% | 58.9% | 68.8% | 63.5% |
| Stop | 0.4141 | 60.7% | 39.6% | 48.0% | 61.1% | 70.3% | 65.3% |

Model operating points maximize F1 on this same chr1 target and are descriptive,
not independently selected deployment thresholds.

## Annotation-level evaluation

Human chr1 is decoded in 50 gene-aware chunks with 1 Mb context. Authenticated
PSAURON emissions are reused, predictions are assigned by transcript-span
midpoint, artificial context-edge predictions are rejected, and duplicates are
merged with `gffread -M`. All 1,020 RefSeq plus-strand CDS loci are eligible.
Comparisons use `gffcompare --strict-match -e 0`.

| Method | Locus PR | Locus SN | Locus F1 | Matching loci | Query loci |
|---|---:|---:|---:|---:|---:|
| EviAnn | 73.8% | 88.8% | 80.6% | 906 | 1,216 |
| UniAnn | 42.1% | 45.5% | 43.7% | 464 | 1,108 |
| EviAnn + UniAnn | 68.2% | 90.6% | 77.8% | 924 | 1,308 |

These corrected union results normalize EviAnn and UniAnn separately to the
same annotation syntax before concatenation and `gffread -M`. The earlier
direct GTF/GFF3 concatenation caused the UniAnn records to be ignored and
incorrectly made the union identical to EviAnn alone.
