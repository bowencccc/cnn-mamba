# Data layout

Large inputs are not stored in Git. Copy or symlink them under `data/raw/`, or
pass their paths directly to the command-line tools.

## Minimum Drosophila training set

```text
data/raw/dmel_genome.fa
data/raw/dmel_genome.fa.fai
data/raw/dmel_eviann_pseudo_label.gff
data/raw/dmel_reference.gtf
```

## Additional Drosophila UniAnn evaluation files

```text
data/raw/dmel_chrX.fa
data/raw/dmel_chrX_plus_CDS_reference.gtf
data/raw/dmel_chrX_plus_EviAnn_CDS.gff
```

The maintained four-state workflow does not require PSAURON. Its vector-scaled
`non-CDS/frame0/frame1/frame2` probabilities are used as UniAnn's coding-frame
input. Drosophila and Human PSAURON files remain in the external manifest only
for reproducing legacy experiments.

## Minimum Human training set

```text
data/raw/human_grch38.fa
data/raw/human_grch38.fa.fai
data/raw/human_eviann_pseudo_label.gff
data/raw/human_refseq_grch38_p14.gff.gz
```

CHESS is optional and is used only by experiments that explicitly request it.

`external_manifest.tsv` records the exact source files, sizes and SHA256 hashes
used in the published experiments. `scripts/check_external.py` validates that
exact inventory; it is expected to report mismatches for different assembly or
annotation releases.

Generated NPZ windows are written under `data/processed/` and are ignored by
Git.
