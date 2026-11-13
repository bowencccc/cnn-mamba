# Current-best inputs for downstream Viterbi experiments

This is the promoted, evidence-source-controlled interface for structured
decoder experiments. Human always means the Rodentia-evidence EviAnn
annotation; Dmel always means the Coleoptera-evidence EviAnn annotation. A run
made with another annotation must not overwrite these entries.

## Promoted models

| Species | Evidence | Regime | Best epoch | Validation mean AP | Model | Chromosome outputs |
|---|---|---|---:|---:|---|---|
| Human | Rodentia proteins | chr1 included | 3 | 0.669506 | ready | ready |
| Dmel | Coleoptera proteins | chrX included | 21 | 0.897269 | ready | ready |

The Human value is the best fully completed epoch. Its original job stopped in
the middle of epoch 4; it is a usable checkpoint but not a completed
early-stopping run. The four task values used for selection are recorded in
`checkpoint_validation_metrics.tsv`. The promoted epoch-3 checkpoint was
subsequently scored over chr1+ and all decoder inputs completed on 2026-11-13.

Human chr1+ candidate AUPRC against RefSeq is:

| Task | AUPRC |
|---|---:|
| Donor | 0.837593 |
| Acceptor | 0.802664 |
| Start | 0.354366 |
| Stop | 0.322378 |

The exact table is in `human_chr1_candidate_auprc.tsv`. Validation-fitted
vector scaling reduced four-state NLL from 0.226524 to 0.019961; see
`human_vector_scaling_metrics.tsv`.

The promoted Dmel vector-scaled output has these corrected, header-preserving
strict locus results:

| Method | Sn | Pr | F1 |
|---|---:|---:|---:|
| UniAnn | 72.9 | 76.4 | 74.6090 |
| EviAnn+UniAnn | 82.7 | 86.2 | 84.4137 |

## Stable interface

Run this whenever a repository/data bundle is installed, or after changing the
explicit promotion in `configs/viterbi_current_best.json`:

```bash
cnn-mamba current-best
```

It creates `artifacts/current_best/viterbi/{human,dmel}/` symlinks and a
machine-readable `manifest.json`. Required assets are the evidence annotation,
prepared windows and checkpoint. Chromosome-wide predictions are tracked
separately, which allows the command to report `model_ready=true` while keeping
`prediction_ready=false` until every decoder input exists.

To regenerate the Human chr1+ scores, raw four-state logits and vector-scaled
probabilities without retraining:

```bash
CNN_MAMBA_PYTHON=/path/to/mamba3_env/bin/python \
  scripts/run_human_rodentia_current_best_outputs.sh
```

This requires a CUDA GPU and about 12 GB for the chr1 four-state arrays,
candidate arrays and exported site-score TSV, plus temporary space. It does not
write the approximately 19 GB per-base probability TSV unless that file is
requested separately.

To fit the current structured-HMM calibration above either promoted model:

```bash
scripts/run_current_best_viterbi_calibration.sh human
scripts/run_current_best_viterbi_calibration.sh dmel
```

Outputs are written to `results/structured_hmm_current_best/{human,dmel}/`.
Changing the promoted checkpoint requires a new output directory or removal of
the old structured cache; never reuse a crop cache made from another model.
