# Human four-state beta=0.10: chr1 held out

This is the completed Cybertron five-epoch pretraining run, distinct from the
earlier Rockfish `human_joint_frame_w10_v1` early-stopping run. The model was
trained from scratch with chr1 excluded from training.

## Configuration

- CNN-Mamba-k7, 10,000 bp windows, 5,000 bp stride;
- eight bidirectional Mamba3 layers, `d_model=192`, `d_state=64`;
- dilation-3 frame branch and four-state non-CDS/frame-0/frame-1/frame-2 head;
- `0.25 * splice_loss + start_stop_loss + 0.1 * joint_frame_loss`;
- joint-frame noncoding within-head weight 0.1;
- seed 42, micro-batch 4, gradient accumulation 2, FP32;
- five fixed epochs; best checkpoint selected by validation mean candidate AP;
- train/validation/test windows: 195,072 / 68,555 / 26,252;
- chr1 excluded from training and used as the test chromosome.

Best epoch: **4**. Validation mean candidate AP: **0.686643**.

## chr1 test-window results

Evaluation covers the owned center of eligible chr1 windows on both strands.
AP is the trainer's 4,096-bin bounded-memory estimate. F1/P/R thresholds are
selected on this evaluation set and are not fixed deployment thresholds.

| Task | AP | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| donor | 0.926373 | 0.882554 | 0.898709 | 0.866970 |
| acceptor | 0.891359 | 0.849896 | 0.859236 | 0.840757 |
| start | 0.527933 | 0.521988 | 0.597167 | 0.463621 |
| stop | 0.481112 | 0.523158 | 0.668610 | 0.429683 |
| joint CDS | 0.911444 | 0.889752 | 0.898563 | 0.881112 |

Mean candidate AP over donor/acceptor/start/stop: **0.706695**.

The associated validation-fitted scalar temperature is **1.228817774**. It
reduced ECE from 0.020826 to 0.004903 and the fraction of bases with maximum
probability at least 0.999 from 13.82% to 0.99%, with zero argmax changes.
Calibration artifacts are in `../temperature_scaling/`.

## Rockfish test

After pulling the repository and materializing the Human external inputs:

```bash
python -m cnn_mamba.score_chromosome \
  --checkpoint models/human_joint_frame_beta010_chr1held_epoch4_inference.pt \
  --fasta data/raw/human_grch38.fa \
  --chrom NC_000001.11 --display-chrom chr1 \
  --output-dir artifacts/scores/human_joint_frame_beta010_v1/chr1held/chr1_plus

python -m cnn_mamba.evaluate_candidates \
  --species human \
  --score-dir artifacts/scores/human_joint_frame_beta010_v1/chr1held/chr1_plus \
  --reference-gtf data/raw/human_chess3.1.3.gtf \
  --fasta data/raw/human_grch38.fa \
  --chrom NC_000001.11 --reference-chrom chr1 \
  --output results/human_joint_frame_beta010_v1/chr1held/chr1_plus_candidate_auprc.tsv
```

For calibrated four-state per-base output, pass the same checkpoint and
`../temperature_scaling/temperature.json` to
`python -m cnn_mamba.score_joint_frame_chromosome`.

The published checkpoint is inference-only. The complete optimizer-bearing
checkpoint remains outside Git; see `models/README.md` for hashes.
