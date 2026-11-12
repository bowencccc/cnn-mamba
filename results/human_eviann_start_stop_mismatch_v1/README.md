# Human EviAnn start/stop mismatch audit

All comparisons use exact canonical motif-start coordinates on GRCh38.p14. Same-locus mapping is strand-aware and based on maximum exon overlap with a RefSeq coding gene.

| Task | EviAnn sites | Exact any RefSeq | Exact MANE Select | Mismatches mapped to same locus | Within 30 nt (mapped) | Within 300 nt (mapped) | Spliced distance divisible by 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| start | 43,015 | 59.4% | 40.6% | 76.2% | 2.2% | 18.4% | 78.8% (6,233 evaluable) |
| stop | 39,329 | 60.4% | 46.8% | 72.5% | 2.0% | 14.1% | 11.2% (2,866 evaluable) |

`Exact MANE Select` is a deliberately narrower target than exact agreement with any RefSeq isoform. A site matching another RefSeq isoform but not MANE is evidence of isoform choice rather than an incorrect coordinate.

Genomic distance modulo 3 must not be treated as a reading-frame test when an intron lies between sites. The report therefore also supplies spliced-transcript distance where it can be computed on a supporting RefSeq transcript.

Files:

- `all_eviann_start_stop_sites.tsv.gz`: every unique EviAnn start/stop site and its RefSeq/MANE classification.
- `mismatched_start_stop_sites.tsv.gz`: exact-RefSeq mismatches with nearest-boundary distances.
- `scope_summary.tsv`: all-primary, validation-chromosome and chr1 summaries.
- `distance_bucket_summary.tsv`: mismatch distance bins.
- `evidence_stratum_summary.tsv`: complete/protein-only/high/low-quality stratification.
- `human_eviann_start_stop_mismatch_distances.png`: report figure.
- `metadata.json`: definitions, inputs and parsing counts.
