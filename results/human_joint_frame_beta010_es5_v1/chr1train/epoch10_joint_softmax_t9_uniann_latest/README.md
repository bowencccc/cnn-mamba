# Human chr1 epoch-10 joint-softmax T=9 evaluation

This run uses the chr1-included Human four-state model at epoch 10.

## Inputs and transformations

- Checkpoint: `runs/human/human_joint_frame_beta010_es5_v1/chr1train/epoch_checkpoints/epoch_010.pt`
- Four-task donor/acceptor/start/stop scores: raw, without temperature scaling
- Four-state head: joint softmax of `logits / 9`
- UniAnn revision: `2f5c0ab8fdb2e848f5538f3a1f741cb9c4c67e07`
- Latest UniAnn coding-frame conversion: independent binary-odds temperature 3 on each of the four T=9 probabilities, followed by the UniAnn frame/noncoding emission construction
- Evaluation reference: processed RefSeq chr1 plus-strand CDS annotation, 1,020 eligible loci
- Comparison: `gffcompare --strict-match -e 0`

The raw four-task score metadata names `chr1train/best_model.pt`. Both that
checkpoint and `epoch_checkpoints/epoch_010.pt` report epoch 10, and their 213
model-state tensors have the same content SHA256
`33df0153d93624cd5f1489564ffa18492e9ea63db830358fae2b94c93f8b1d1c`.
They are therefore the same model weights despite different whole-file hashes.

## Strict locus results with the corrected union pipeline

| Query | Sensitivity (%) | Precision (%) | F1 (%) | Matching | Reference | Query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EviAnn only | 88.8 | 73.8 | 80.608 | 906 | 1,020 | 1,216 |
| UniAnn only | 45.5 | 42.1 | 43.734 | 464 | 1,020 | 1,108 |
| EviAnn + UniAnn | 90.6 | 68.2 | 77.820 | 924 | 1,020 | 1,308 |

The zero change in the last row is an union-construction artifact, not a true
zero contribution from UniAnn. The EviAnn source was GTF while the stitched
UniAnn source was GFF3. Literal concatenation mixed both attribute syntaxes in
one file, so `gffread` selected GTF parsing from the first records and ignored
the appended GFF3 transcripts. A controlled ablation showed that generated
`RLOC_*` identifiers were not the cause: normalizing both inputs to either GTF
or GFF3 retained the same 5,539 transcripts in 1,308 loci.

Relative to EviAnn alone this adds 18 matching loci, but also 104 additional
novel loci, so sensitivity rises by 1.8 points while precision falls by 5.6
points. The formal summary was regenerated after the union implementation was
fixed.

## Main outputs

- Strict summary: `strict_evaluation/locus_summary.tsv`
- Stitched UniAnn GFF: `stitched/baseline.bdae5c083e49629bd304/prediction.stitched.merged.gff`
- EviAnn+UniAnn merged GFF: `unions/baseline.eviann_plus_uniann.88ed23fb7556d94ef122/combined.merged.gff`
- EviAnn-only merged GFF: `unions/eviann_only.3695497bdf861651cfdd/combined.merged.gff`

The chromosome-wide score and probability files are stored outside this result
directory under `artifacts/`; see the parent project handoff for their paths.
