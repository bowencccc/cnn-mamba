# Human 10 kb four-state model: chr1 held out

Training job **31705646** (`human4-train`) completed successfully on Rockfish
on 2026-10-08 (Slurm `COMPLETED`, exit `0:0`). Slurm elapsed time was
25 h 45 min 29 s; the training program recorded 25.72 h.
This report covers training and its final test-window evaluation. Completion
of training does not establish completion of downstream chromosome scoring,
four-class evaluation, or UniAnn stages.

## Configuration and data

- CNN-k7 + bidirectional Mamba3, dilation-3 frame branch, and a four-state
  non-CDS/frame-0/frame-1/frame-2 auxiliary head; candidate heads remain active.
- Window/stride: 10,000/5,000 bp; seed 42; batch 4; gradient accumulation 2;
  AMP disabled; one NVIDIA A100-PCIE-40GB; four data-loader workers.
- Joint-frame auxiliary weight 0.1; non-CDS weight 0.1.
- EviAnn training labels and CHESS 3.1.3 evaluation labels, both strands.
- Train: chr2/3/4/6/7/8/11–21/X/Y; validation: chr5/9/10/22; test: chr1.
- Train/validation/test windows: 218,005 / 45,630 / 26,241.
  Full split and preparation metadata: [split_manifest.json](split_manifest.json).
- Maximum 30 epochs; early-stopping patience 3; minimum validation mean-AP
  improvement 0.0001. Best epoch **4**, validation mean AP **0.417442**.
  Training stopped after epoch **7** following three epochs without improvement.

## Final test-window results

The trainer reloaded `best_model.pt` (epoch 4) for these chr1 results.
Evaluation uses the central 5,000 bases of each eligible 10,000-base window
on both strands. Candidate AP and Joint CDS AP use the 4,096-bin approximation.
F1/P/R are reported at the thresholds selected by this evaluation, not at
independently fixed deployment thresholds. Joint CDS is `1 - P(non-CDS)`;
it does not measure the accuracy of the three individual frames.

| Task | AP | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| donor | 0.7820 | 0.7311 | 0.7372 | 0.7252 |
| acceptor | 0.6990 | 0.6750 | 0.6823 | 0.6680 |
| start | 0.1648 | 0.2365 | 0.2249 | 0.2493 |
| stop | 0.0612 | 0.1273 | 0.1309 | 0.1239 |
| joint_cds | 0.6460 | 0.6253 | 0.6078 | 0.6438 |

Mean candidate AP (donor/acceptor/start/stop): **0.426755**.
Full precision, thresholds and counts: [test_window_metrics.tsv](test_window_metrics.tsv).
Epoch history: [training_history.tsv](training_history.tsv).
Raw trainer output: [results.json](results.json) and [training_log.txt](training_log.txt).

## Comparison with the previous human model

The [previous human result](../../human_cnn_mamba_k7/README.md) reported AP
0.8751 / 0.8437 / 0.3828 / 0.2032 for donor / acceptor / start / stop
(mean 0.5762), numerically higher than this run. It included chr1 in training
and evaluated exact AP across canonical candidates on the full chr1 positive
strand. This run excludes chr1 from training and reports binned AP on selected
test-window centers from both strands. Data preparation and validation splits
also differ. These numbers are **not a controlled architecture comparison**
and do not establish that four-state supervision helps or hurts.
A fair comparison requires the same training data/split and the same evaluation
candidate set and method for both models.

## Provenance and scope

[training_command.txt](training_command.txt) preserves the actual launch command;
[pipeline_config.json](pipeline_config.json) records pipeline settings.
Its `evaluation_strand: "+"` describes downstream chromosome evaluation,
not the trainer's both-strand test-window metrics reported here.
[provenance.json](provenance.json) records Slurm status and source hashes.
The training checkout contained local modifications on top of `3c8f52c`;
that base commit alone does not reproduce the run. This publication archives
results and metadata, not those implementation changes. Checkpoints and large
arrays remain external under `runs/human_joint_frame_w10_v1/chr1held` and the
Rockfish data storage links.
