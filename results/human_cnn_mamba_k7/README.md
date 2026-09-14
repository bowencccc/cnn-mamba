# Existing Human CNN--Mamba-k7 result

The staged checkpoint is the 10 kb window / 5 kb stride CNN--Mamba-k7 model
trained for five epochs with chr1 included in training. Its exact CHESS 3.1.3
evaluation over all canonical chr1 positive-strand candidates was:

| Task | AUPRC |
|---|---:|
| Donor | 0.875113932 |
| Acceptor | 0.843740368 |
| Start | 0.382820293 |
| Stop | 0.203225204 |
| Arithmetic mean | 0.576224949 |

Because chr1 was included in training, this is a downstream/fit diagnostic,
not an unbiased chromosome-held-out generalization result. `training/` contains
the complete five-epoch history and log. `training_data_manifest.json` records
the exact 195,072/68,555/26,252 train/validation/test windows used. Large NPZ
windows and chr1 score arrays were omitted because they can be regenerated.
