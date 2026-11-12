# Human EviAnn class-code annotation audit

Source annotation:
`GCF_000001405.40_GRCh38.p14_genomic.noalpsnopatches.fna.pseudo_label.gff.gz`.
The audit uses exact, unique canonical CDS sites matched to RefSeq by primary
chromosome, strand, and 0-based motif-start coordinate. The primary comparison
is Human chr1 positive strand, matching the deployment evaluation scope.

## Annotation composition

- 115,034 mRNA records; 114,659 have CDS; 22,916 coding loci.
- Evidence: 66,125 `complete`, 47,967 `protein_only`, and 942
  `transcript_only` transcripts.
- Class codes: 67,122 `=`, 41,248 `k`, 6,599 `j`, 43 `m`, and 22 `n`.
- 2,905 transcripts carry `LOW_QUALITY_PROTEIN`.

## chr1+ start/stop results

| Selection | Start Pr | Start Sn | Start F1 | Stop Pr | Stop Sn | Stop F1 |
|---|---:|---:|---:|---:|---:|---:|
| All isoforms (union) | 59.4 | 68.1 | 63.4 | 61.3 | 69.9 | 65.3 |
| Longest CDS per locus | 69.6 | 42.5 | 52.7 | 78.0 | 51.1 | 61.7 |
| Longest complete `=`; else complete `k` | 81.7 | 31.8 | 45.8 | 91.2 | 38.1 | 53.8 |
| Complete `=`/`k`; otherwise longest-CDS locus fallback | 70.4 | 43.0 | 53.4 | 79.8 | 52.3 | 63.2 |
| LightGBM OOF P90 | 79.3 | 49.8 | 61.2 | 79.0 | 59.7 | **68.0** |

The LightGBM row is leakage-controlled: chr1 uses predictions from the model
that held out the chr1 chromosome fold, and its P90 threshold was fitted using
only the other folds. Thresholds are 0.6529115 for start and 0.4725645 for
stop. Scores are restricted to sites still present in the new class-code
annotation before comparison with the same RefSeq truth set.

The strict complete-class policy selects 14,686 of 22,916 coding loci and
excludes 8,230 loci (35.9%): 7,505 have no complete transcript and 725 have a
complete transcript but no `=` or `k` model. It therefore produces very clean
start/stop labels but loses too much coverage to be used alone.

The locus-fallback policy is the practical class-aware replacement for the old
longest-CDS-per-locus rule. Relative to longest CDS, it improves start from
69.6/42.5/52.7 to 70.4/43.0/53.4 and stop from 78.0/51.1/61.7 to
79.8/52.3/63.2 (Pr/Sn/F1), while retaining one selected model for every coding
locus. The gain is real but modest.

LightGBM gives the strongest overall filtered result in this comparison. For
start it reaches 79.3/49.8/61.2, combining much higher precision than longest
CDS with substantially more sensitivity than strict complete `=`/`k`. For
stop it gives the best F1, 68.0. Unlike the per-locus policies, however, this is
a site-level filter and does not guarantee one complete transcript per locus.

## Interpretation

Class code is useful as a confidence feature, especially when combined with
`evidence=complete`; it is not sufficient as a hard inclusion rule. The large
precision increase of the strict policy comes with a larger sensitivity loss.
For CNN--Mamba training, the recommended first comparison is therefore:

1. old longest-CDS-per-locus labels;
2. complete `=`/`k` with longest-CDS fallback at otherwise uncovered loci.

Only start/stop labels need to change in that controlled experiment; splice
labels, architecture, split, optimizer, and all other settings should remain
fixed.

Files:

- `site_prsn.tsv`: all four tasks, four scopes, selection policies, and
  individual class-code groups.
- `evidence_class_distribution.tsv`: evidence-by-class transcript/locus counts.
- `excluded_loci.tsv.gz`: loci rejected by strict complete `=`/`k` selection.
- `summary.json`: provenance and headline counts.
- `chr1_plus_start_stop_selection_prsn.png`/`.pdf`: report-ready comparison.
