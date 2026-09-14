# Completed window-size experiment

These files were copied from
`/home/bowencccc/ssm_splice_site/droso_window_grid_legacy_recipe_20260906` on
2026-09-13. The configuration changed window/stride and memory batch geometry
while retaining the original CNN–Mamba-k7 recipe and five epochs.

- `summary/results.tsv`: 5–40 kb exact held-out AUPRC and original downstream metrics.
- `summary/w80_results.tsv`: completed 80 kb extension.
- `latest_uniann_results.tsv`: latest UniAnn strict results for the 5--40 kb cells;
  the 80 kb row is in `summary/w80_results.tsv`.
- `plots/`: PR–Sn curves, AUPRC, training-loss, score distributions and locus plots.
- `training/`: `results.json` histories and text logs; checkpoints are external.
- `gffcompare/`: compact strict `gffcompare` stats files.

The latest strict locus results are:

| Window | UniAnn Sn | UniAnn Pr | UniAnn F1 | Combined Sn | Combined Pr | Combined F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 5 kb | 74.2 | 80.4 | 77.18 | 89.3 | 91.3 | 90.29 |
| 10 kb | 75.0 | 79.8 | 77.33 | 89.8 | 90.8 | 90.30 |
| 20 kb | 68.6 | 78.0 | 73.00 | 88.2 | 91.5 | 89.82 |
| 30 kb | 68.4 | 77.9 | 72.84 | 87.3 | 91.0 | 89.11 |
| 40 kb | 70.0 | 78.5 | 74.01 | 88.7 | 91.4 | 90.03 |
| 80 kb | 65.5 | 74.8 | 69.84 | 88.2 | 91.5 | 89.82 |

UniAnn source commit: `91477a69e1a949fed082c4662b348d9a7a91ca2b`.
The 80 kb combined stats are under `gffcompare/combined/w80`; other cells are
under `gffcompare/runs/<cell>`.
