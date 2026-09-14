# External data layout

Large inputs are deliberately not tracked by Git. Put files at these logical
paths (real files or symlinks are both accepted):

```
data/raw/dmel_genome.fa
data/raw/dmel_genome.fa.fai
data/raw/dmel_eviann_pseudo_label.gff
data/raw/dmel_reference.gtf
data/raw/dmel_chrX.fa
data/raw/psauron_score_droso.csv
data/raw/dmel_chrX_plus_CDS_reference.gtf
data/raw/dmel_chrX_plus_EviAnn_CDS.gff
```

The exact source files, byte sizes and SHA256 hashes from the original server
are recorded in `external_manifest.tsv`. Run `python scripts/check_external.py`
after filling the paths. `cnn-mamba-prepare-droso` creates NPZ windows under
`data/processed/<window-id>/{train,val,test}`; those generated files are also
ignored by Git.
