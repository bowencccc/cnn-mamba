# CNN–Mamba-k7 portable pipeline

This repository is the cleaned, portable version of the current Drosophila and
Human CNN–Mamba-k7 work. It covers the complete path from raw
FASTA/annotations to training windows, model training, chromosome-wide
candidate scoring, exact AUPRC, codon/frame prediction, UniAnn input export,
strict `gffcompare`, and EviAnn+UniAnn evaluation.

Small result tables, plots, training histories and strict-match statistics are
tracked in `results/`. Large biological inputs, generated NPZ windows, raw
score arrays and PyTorch checkpoints are excluded from Git and described in
`external_manifest.tsv` with exact byte sizes and SHA256 hashes.

## Public software interface

The stable interface is the installed `cnn-mamba` command. The many files under
`scripts/` are retained research recipes and figure generators; they are not
the public API.

```bash
cnn-mamba --help
cnn-mamba doctor --uniann-root /path/to/UniAnn
cnn-mamba prepare-droso --help
cnn-mamba train --help
cnn-mamba score --help
cnn-mamba uniann-eval --help
```

Multi-stage runs can be defined without editing source files. Copy the example
JSON file, change only its variables, inspect the generated commands, then run it:

```bash
cp configs/workflow.droso.example.json configs/my_workflow.json
export UNIANN_ROOT=/path/to/UniAnn
cnn-mamba workflow configs/my_workflow.json --dry-run
cnn-mamba workflow configs/my_workflow.json
```

Every workflow writes per-stage logs and a machine-readable `run.json` under
`workflow_runs/<UTC timestamp>/`. Paths support `${PROJECT_ROOT}` and ordinary
environment variables such as `${UNIANN_ROOT}`. No packaged source or supported
script contains a user-specific `/home/...` path. See
[`docs/SOFTWARE_LAYOUT.md`](docs/SOFTWARE_LAYOUT.md) for the boundary between
software, external data, generated state, and historical experiments.

The example makes coding-frame calibration explicit. The default recipe fits
and applies four-class **vector scaling** on the validation split. It then
reproduces the selected UniAnn input behavior explicitly: independent
binary-odds scaling at `T=3` followed by the frame/non-CDS gate. These are two
different transformations; both are recorded in metadata and the emission
manifest. The older joint-softmax `T=9` experiment is retained only as a
historical comparison and is not the software default.

## Reproducibility boundary

The preserved recipe is:

- model: 8-layer bidirectional Mamba3, `d_model=192`, `d_state=64`;
- a depthwise/pointwise CNN before every Mamba block, kernel size 7;
- two primary 3-class heads: background/donor/acceptor and
  background/start/stop;
- an optional dilation-3 codon-aware residual branch feeding the start/stop and
  coding-frame heads;
- an optional four-state auxiliary head in class order
  `non_CDS/frame0/frame1/frame2`;
- candidate-masked cross entropy plus the optional four-state loss:
  `0.25 * splice_loss + start_stop_loss + lambda_frame * frame_loss`;
- AdamW, learning rate `3e-4`, cosine decay, and an explicitly recorded random
  seed (42 unless overridden);
- 50% overlapping windows; experiment configs set gradient accumulation so
  comparisons can use a matched effective number of bases per optimizer step;
- validation on chromosome 3L and held-out candidate evaluation on chrX+;
- UniAnn evaluation uses chrX included in training by design, followed by
  `gffread -M` and `gffcompare --strict-match -e 0`.

The window experiment is a legacy-recipe ablation, not a compute-normalized
comparison: larger windows receive fewer optimizer updates in five epochs.

## Current published Drosophila models

Two compact inference checkpoints are tracked under `models/`:

- `drosophila_joint_frame_beta010_chrxtrain_epoch23_inference.pt`: current
  best four-state Drosophila-evidence model for UniAnn augmentation;
- `drosophila_coleoptera_evidence_joint_frame_beta010_chrxtrain_epoch21_inference.pt`:
  matched model trained from Coleoptera-protein-evidence EviAnn pseudo-labels.

Both use CNN--Mamba-k7, 10 kb windows, 5 kb stride, dilation 3, four-state
weight 0.1 and noncoding weight 0.1, with chrX included for the downstream
annotation setting. Full resumable checkpoints remain external. See
`models/README.md` and `results/evidence_source_comparison_v1/` for provenance,
PR--Sn curves, EviAnn operating points and strict locus results.

## 1. Install

The recorded environment used Python 3.10, PyTorch 2.6.0+cu124 and a Git build
of Mamba at commit `be0303b971cd79a4fbbcfc411a01b33f6b0e3602`. Its package
metadata says `mamba-ssm==2.3.1`, but the PyPI 2.3.1 source is not equivalent:
the pinned Git commit is required because it contains `Mamba3`.

```bash
conda env create -f environment.yml
conda activate cnn_mamba_k7
pip install -e . --no-deps --no-build-isolation
python scripts/smoke_model.py
```

For a newer CUDA stack, install a matching PyTorch build first, then compile or
install `causal-conv1d` and `mamba-ssm` against it. Do not assume a wheel built
for a different PyTorch/CUDA combination is compatible.

If installing into an already-created environment, install the exact Mamba3
source with:

```bash
pip install --no-build-isolation \
  "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@be0303b971cd79a4fbbcfc411a01b33f6b0e3602"
```

UniAnn is a separate dependency. The recorded downstream results use commit
`91477a69e1a949fed082c4662b348d9a7a91ca2b`:

```bash
git clone https://github.com/alekseyzimin/UniAnn.git
cd UniAnn
git checkout 91477a69e1a949fed082c4662b348d9a7a91ca2b
cd src && make clean && make
cd .. && ./install.sh
export UNIANN_ROOT="$PWD"
```

## 2. Stage large inputs

See `data/README.md` and `external_manifest.tsv`. On the original server, this
creates local symlinks without copying files. The complete Drosophila + Human
inventory is about 7.11 GiB; individual groups can be materialized separately:

```bash
bash scripts/link_current_server.sh
python scripts/check_external.py --fast
python scripts/check_external.py             # full SHA256 verification
```

On another server, copy or symlink your files to the same logical paths. The
`original_path` column records where each exact file came from; it is provenance,
not a path that must exist on the new server.

Materialize only the Human group (about 5.93 GiB, including full Human
PSAURON) with:

```bash
bash scripts/materialize_external.sh /path/to/human_external_bundle human
```

## 3. Generate training windows

This reads the whole-genome FASTA, EviAnn pseudo-label GFF and reference GTF
once, then creates all configured window sizes:

```bash
cnn-mamba-prepare-droso \
  --config configs/droso_window_grid.json \
  --fasta data/raw/dmel_genome.fa \
  --eviann-gff data/raw/dmel_eviann_pseudo_label.gff \
  --reference-gtf data/raw/dmel_reference.gtf \
  --output-root data/processed
```

Splits are fixed: train = 2L/2R/3R/4/Y, validation = 3L, test = X. Both strands
are used for training; downstream candidate scoring is chrX positive strand.

## 4. Train, score and evaluate

One command runs a single cell and is safe to assign to one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_cell.py \
  --cell w10 --regime heldout --stages train,score,auprc
```

To train cells concurrently on a multi-GPU server, start one process per GPU,
for example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_cell.py --cell w5  --regime heldout &
CUDA_VISIBLE_DEVICES=1 python scripts/run_cell.py --cell w10 --regime heldout &
CUDA_VISIBLE_DEVICES=2 python scripts/run_cell.py --cell w20 --regime heldout &
CUDA_VISIBLE_DEVICES=3 python scripts/run_cell.py --cell w40 --regime heldout &
wait
```

The default stages produce exact sklearn AUPRC for donor, acceptor, start and
stop over every canonical chrX+ candidate. To reuse a staged checkpoint without
training, omit `train`; `run_cell.py` automatically looks under
`artifacts/checkpoints/<regime>/<cell>/best_model.pt`.

Training writes per-epoch train and validation total/splice/start-stop losses to
`results.json` and to the checkpoint history. Complete resumable checkpoints for
every epoch are retained under `runs/<regime>/<cell>/epoch_checkpoints/`, while
`best_model.pt` and `last_model.pt` keep their original meanings. Runs trained
with all splits combined have no held-out validation set, so their validation
loss fields are `null`.

The historical loss remains the default: `0.25 * splice_loss +
start_stop_loss`. Controlled loss-weight runs can use
`--splice-loss-weight 1.0`; the same weight is applied to both training and
validation loss calculations and is recorded in each checkpoint's config.

### Early-stopping window rerun

The new 5/10/20/30/40/80 kb rerun has a separate configuration so it cannot
silently resume or overwrite the legacy five-epoch experiment. It uses at most
30 epochs and stops after three consecutive epochs without a validation mean-AP
increase greater than `0.0001`. Validation mean AP is the arithmetic mean over
donor, acceptor, start and stop; chromosome 3L remains the validation split.
Every completed epoch is retained as a checkpoint.

On Rockfish, first generate the window files if `data/processed/w*/` is absent,
using the command in section 3. Then submit the 12-job array (six held-out
models for chrX+ AUPRC and six chrX-in-training models for UniAnn):

```bash
cd ~/cnn_mamba_k7_portable
sbatch --account=<PI_NAME>_gpu scripts/rockfish_droso_early_stopping.slurm
```

Set `CNN_MAMBA_PYTHON` to the fully qualified Python executable if `python` is
not the prepared CNN--Mamba environment. Set `UNIANN_ROOT` if UniAnn is not at
`~/UniAnn`. Outputs are namespaced under `early_stopping_v1` in `runs/`,
`artifacts/scores/`, and `work/uniann/`.

### Coding-frame ablation

The controlled 10 kb chrX-held-out ablation compares the existing baseline
with a dilation-3 residual CNN used only by the start/stop head, a masked
three-class CDS-phase auxiliary loss (weight `0.1`), and both changes together.
All other split, seed, optimizer, loss and early-stopping settings are unchanged.
CDS phase is a training target, never a model input; bases outside annotated CDS
and positions where transcript phases conflict are masked.

```bash
cnn-mamba-prepare-droso \
  --config configs/droso_frame_w10.json \
  --output-root data/processed/frame_ablation \
  --include-phase-labels

CNN_MAMBA_PYTHON=/path/to/python \
  bash scripts/run_droso_frame_ablation_local.sh
```

The script trains the three new arms, scores every canonical chrX+ candidate,
computes exact AUPRC, and writes the final table and plot under
`results/frame_ablation_v1/`. The baseline is reused from
`early_stopping_v1/heldout/w10`.

To compare phase-loss weights `0.03/0.05/0.10/0.20` across both held-out and
chrX-in-training regimes, including exact PR--Sn curves and strict UniAnn locus
metrics, run:

```bash
CNN_MAMBA_PYTHON=/path/to/python \
  bash scripts/run_phase_weight_grid_local.sh
```

The completed summary is written to `results/phase_weight_grid_v1/`. Existing
weight-0.10 checkpoints and scores are reused; the other six models are trained
from scratch with identical splits, seed and early-stopping settings.

Per-base phase probabilities from a phase-auxiliary checkpoint can be exported
for an entire chromosome strand:

```bash
python -m cnn_mamba.score_phase_chromosome \
  --checkpoint /path/to/best_model.pt \
  --fasta data/raw/dmel_genome.fa \
  --chrom NC_004354.4 \
  --display-chrom chrX \
  --output-dir artifacts/phase_predictions/chrX_plus
```

`phase_probabilities.npy` has shape `[chromosome_length, 3]` in class order
phase 0/1/2, and row `i` corresponds to zero-based genomic position `i` on the
plus strand. The auxiliary head was not supervised outside non-conflicting CDS,
so probabilities outside coding sequence are emitted but are not calibrated;
the head also does not enforce a global 0-1-2 path.

To convert the arrays to a one-row-per-base TSV with one-based coordinates:

```bash
python -m cnn_mamba.export_phase_tsv \
  --probabilities artifacts/phase_predictions/chrX_plus/phase_probabilities.npy \
  --argmax artifacts/phase_predictions/chrX_plus/phase_argmax.npy \
  --chrom chrX --strand + \
  --output artifacts/phase_predictions/chrX_plus/chrX_plus_phase_probabilities.tsv
```

### Frozen CDS-head gating experiment

The CDS-gating probe preserves every parameter in an existing phase model and
fits only a new linear CDS/non-CDS head. Independent CDS labels are the union of
all annotated CDS intervals, so bases with conflicting isoform phases remain
valid CDS positives. Regenerate the 10 kb labels and train the probe with:

```bash
cnn-mamba-prepare-droso \
  --config configs/droso_frame_w10.json \
  --output-root data/processed/cds_gating \
  --include-phase-labels

python -m cnn_mamba.train_cds_head \
  --data-root data/processed/cds_gating/w10_frame \
  --base-checkpoint /path/to/phase_0.10/best_model.pt \
  --output-dir runs/cds_gating_v1/frozen_probe/chrxtrain/w10 \
  --include-test-in-train
```

Chromosome-wide inference exports `P(CDS)` and the hierarchical joint scores
`P(CDS) * P(phase_k | CDS)` for phase 0/1/2:

```bash
python -m cnn_mamba.score_gated_phase_chromosome \
  --checkpoint runs/cds_gating_v1/frozen_probe/chrxtrain/w10/best_model.pt \
  --fasta data/raw/dmel_genome.fa --chrom NC_004354.4 \
  --output-dir artifacts/phase_predictions/cds_gating_v1/chrX_plus
```

To run the controlled joint-loss grid at `lambda_CDS=0.03/0.05`, including
held-out and chrX-in-training models, strict UniAnn evaluation, and per-loss
shared-core-backbone gradient diagnostics:

```bash
CNN_MAMBA_PYTHON=/path/to/python \
  bash scripts/run_cds_weight_grid_local.sh
```

The gradient probe uses the same fixed phase-containing training window for
every model. On the first batch of each epoch it measures each unweighted loss
with `torch.autograd.grad`, reports the norm after applying that loss's
coefficient, and restores RNG state so the diagnostic does not alter training.
Heads and the dilation-3 frame branch are excluded from the shared-core norm.

### Four-state coding-frame head

The current model has three related components that should not be conflated:

```text
DNA -> embedding -> 8 x CNN--BiMamba3 -> shared per-base representation
                                      |-> splice head: background/donor/acceptor
                                      `-> dilation-3 codon-aware branch
                                           |-> start/stop head: background/start/stop
                                           `-> four-state auxiliary head:
                                               non_CDS/frame0/frame1/frame2
```

The **start/stop codon head** is a primary three-class candidate head. Its loss
is evaluated only at canonical start/stop candidates and annotated positives.
The **codon-aware branch** is not a classifier: it is a residual depthwise CNN
with dilation 3, so adjacent kernel taps sample bases three nucleotides apart.
The **four-state head** is a dense auxiliary classifier used to teach the
shared representation coding status and reading frame. It does not replace the
start/stop head.

The four-state head supersedes the earlier masked three-class phase head. The
class order used internally is `non_CDS/frame0/frame1/frame2`. Each annotated
CDS segment's GFF/GTF phase is converted to a strand-aware per-base `0,1,2`
track. Intronic and intergenic bases are labeled `non_CDS`; bases assigned
different coding frames by overlapping isoforms are masked rather than forced
to an arbitrary state. This is supervised annotation-derived information, not
an input feature and not a globally decoded reading-frame path.

Coding and noncoding cross-entropies are averaged separately:

```text
frame_loss = CE(coding frame0/1/2 bases)
           + beta * CE(non_CDS bases)

total_loss = 0.25 * splice_loss
           + start_stop_loss
           + lambda_frame * frame_loss
```

The main completed model uses `lambda_frame=0.1`, `beta=0.1`, and dilation 3.
The grid also compares `beta=0.03/0.10/0.30` while leaving both primary heads
and their loss weights unchanged. The four-state probabilities can be
temperature- or vector-calibrated after training without altering the raw
donor/acceptor/start/stop scores.

Run all six cells sequentially on one local GPU with:

```bash
CNN_MAMBA_PYTHON=/path/to/python \
  bash scripts/run_joint_frame_grid_local.sh
```

On Rockfish, generate the phase/CDS windows once and submit the six independent
GPU cells as a Slurm array:

```bash
cnn-mamba-prepare-droso \
  --config configs/droso_frame_w10.json \
  --output-root data/processed/cds_gating \
  --include-phase-labels

sbatch --account=<PI_NAME>_gpu scripts/rockfish_joint_frame_grid.slurm
```

After the whole array succeeds, create the cross-model figures and tables:

```bash
python scripts/summarize_joint_frame_grid.py
```

Export a completed four-state chromosome prediction as a one-row-per-base TSV
in `frame0/frame1/frame2/noncoding` column order with one-based coordinates:

```bash
python -m cnn_mamba.export_joint_frame_tsv \
  --probabilities artifacts/joint_frame_predictions/joint_frame_grid_v1/b010/chrxtrain/w10/chrX_plus/joint_frame_probabilities.npy \
  --argmax artifacts/joint_frame_predictions/joint_frame_grid_v1/b010/chrxtrain/w10/chrX_plus/joint_frame_argmax.npy \
  --output artifacts/joint_frame_predictions/joint_frame_grid_v1/b010/chrxtrain/w10/chrX_plus/chrX_plus_joint_frame_probabilities.tsv \
  --chrom chrX --strand +
```

The completed local run is versioned under `results/joint_frame_grid_v1/`.
Among the four-state models, noncoding weight `0.10` gave the best strict
UniAnn-only locus result (Sn 80.4, Pr 79.6, F1 80.00). The phase-only baseline
remained narrowly best after combining EviAnn and UniAnn (F1 90.54 versus
90.34 for the four-state 0.10 model).

For a reproducible alternate seed, pass it explicitly to the end-to-end driver;
the value is stored in every checkpoint:

```bash
python scripts/run_cell.py \
  --config configs/droso_small_window_v1.json \
  --experiment-tag droso_small_window_seed43 \
  --cell w5 --regime chrxtrain --seed 43 \
  --frame-dilation 3 --joint-frame-aux-weight 0.1 \
  --joint-noncoding-weight 0.1 --stages train,score,auprc
```

## 5. Run UniAnn and combined strict evaluation

This deliberately uses the `chrxtrain` checkpoint, exports the six-column
legacy score TSV, runs UniAnn, merges duplicate transcripts, and computes both
UniAnn-only and EviAnn+UniAnn locus metrics:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_cell.py \
  --cell w10 --regime chrxtrain \
  --stages score,auprc,export,uniann \
  --uniann-root "$UNIANN_ROOT"
```

Outputs are written to `work/uniann/<cell>/locus_metrics.tsv`. The exact UniAnn
inputs are `dmel_chrX.fa`, `psauron_score_droso.csv`, the model score TSV, and
`dmel_chrX_plus_CDS_reference.gtf`; the optional combined call adds
`dmel_chrX_plus_EviAnn_CDS.gff`.

## Existing results

`results/window_ablation/` contains all compact outputs from the completed
5/10/20/30/40/80 kb experiment. The best honest held-out mean AUPRC in this
legacy grid was 20 kb (0.7949); the best UniAnn-only locus F1 was 10 kb
(77.33 using the latest UniAnn), and EviAnn+UniAnn at 10 kb was Sn 89.8 / Pr
90.8 / F1 90.30. See `results/window_ablation/README.md` for provenance and
file routing.

The newer CDS auxiliary, boundary-weighted CDS, and four-state frame experiments
are stored in `results/cds_weight_grid_v1/`, `results/cds_boundary_r128_v1/`,
and `results/joint_frame_grid_v1/`. These compact tables, JSON summaries, loss
plots, PR--Sn figures and strict locus comparisons are tracked by Git. Raw
chromosome score arrays, per-base predictions, training checkpoints and UniAnn
working directories remain external.

## Human pipeline

Completed Rockfish four-state training (job 31705646, 2026-10-08):
[chr1-held-out results and evaluation caveats](results/human_joint_frame_w10_v1/chr1held/README.md).
Best epoch 4; early stopping at epoch 7; mean test-window candidate AP 0.426755.


The Human external group contains GRCh38, NCBI RefSeq GRCh38.p14, CHESS 3.1.3, the Human EviAnn
pseudo-label GFF, full Human PSAURON scores, and the current 10 kb
CNN--Mamba-k7 chr1-in-training checkpoint. Generated NPZ windows and chr1 score
arrays are omitted and can be rebuilt.

```bash
cnn-mamba-prepare-human \
  --fasta data/raw/human_grch38.fa \
  --eviann-gff data/raw/human_eviann_pseudo_label.gff \
  --reference-annotation data/raw/human_refseq_grch38_p14.gff.gz \
  --output-dir data/processed/human_refseq_w10_chr1held \
  --max-windows-per-chrom-strand 0 \
  --region-overlap-bp 10000
```

Train an honest chr1-held-out model:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_human.py \
  --regime chr1held --stages train,score,auprc
```

For the Human four-state `noncoding/frame0/frame1/frame2` pretraining run,
generate the EviAnn training and RefSeq reference phase/CDS arrays while keeping
the original 10 kb windows and chromosome split:

```bash
cnn-mamba-prepare-human \
  --fasta data/raw/human_grch38.fa \
  --eviann-gff data/raw/human_eviann_pseudo_label.gff \
  --reference-annotation data/raw/human_refseq_grch38_p14.gff.gz \
  --output-dir data/processed/human_joint_frame_w10_refseq_chr1held \
  --max-windows-per-chrom-strand 0 \
  --region-overlap-bp 10000 \
  --include-phase-labels

CNN_MAMBA_PYTHON=/path/to/python \
  bash scripts/run_human_joint_frame_pretrain_local.sh
```

This run holds out chr1, uses frame dilation 3, joint-head weight 0.1 and
noncoding within-head weight 0.1. It retains the original Human baseline's five
epochs and effective batch of eight windows. On Rockfish, submit the same run
with `scripts/rockfish_human_joint_frame_pretrain.slurm` after generating or
transferring the processed windows.

Train the matched chr1-included model from scratch while retaining the same
validation chromosomes for model selection:

```bash
CNN_MAMBA_PYTHON=/path/to/python \
  bash scripts/run_human_joint_frame_chr1train_local.sh
```

This merges the original chr1 test windows into training via
`--include-test-in-train`; it does not evaluate chr1 as held out and does not
initialize from the chr1-held-out checkpoint.

#### Four-state temperature scaling

Fit one positive scalar temperature on the RefSeq validation chromosomes from
the chr1-held-out split. The fitter samples only the non-overlapping, owned
5 kb center of each 10 kb validation window and excludes CDS bases with
ambiguous reference phase. Chr1 is never used to choose the temperature.

```bash
CNN_MAMBA_PYTHON=/path/to/python \
  bash scripts/run_human_joint_frame_temperature_scaling.sh
```

The fitted transformation is `softmax(joint_frame_logits / T)`. It is applied
only to the four-state head, preserves every per-base argmax exactly, and does
not change the splice or start/stop heads. `temperature.json` records NLL,
Brier score, ECE, confidence/extreme-probability summaries, and a diagnostic
temperature using the training loss's group-balanced objective. The standard
empirical-validation NLL temperature is the one used downstream.

Apply the saved temperature during chromosome-wide four-state scoring with:

```bash
python -m cnn_mamba.score_joint_frame_chromosome \
  --checkpoint runs/human/human_joint_frame_beta010_v1/chr1held/best_model.pt \
  --fasta data/raw/human_grch38.fa --chrom chr1 --display-chrom chr1 \
  --calibration-json results/human_joint_frame_beta010_v1/temperature_scaling/temperature.json \
  --output-dir artifacts/joint_frame_predictions/human_beta010/chr1_plus_calibrated
```

Reuse and evaluate the existing chr1-in-training checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_human.py \
  --regime chr1train --stages score,auprc
```

## Repository map

```text
src/cnn_mamba/model.py                 model definition/checkpoint loader
src/cnn_mamba/prepare_droso.py         raw annotations -> NPZ windows
src/cnn_mamba/train.py                 training and validation
src/cnn_mamba/score_chromosome.py      all canonical candidate scores
src/cnn_mamba/evaluate_candidates.py   exact held-out AUPRC
src/cnn_mamba/export_uniann_scores.py  legacy UniAnn score TSV
src/cnn_mamba/uniann_evaluate.py       UniAnn + strict locus comparison
src/cnn_mamba/score_joint_frame_chromosome.py  four-state per-base inference
src/cnn_mamba/evaluate_joint_frame.py  CDS/frame evaluation for four-state head
scripts/run_cell.py                    end-to-end per-GPU driver
scripts/rockfish_joint_frame_grid.slurm  six-cell Rockfish Slurm array
external_manifest.tsv                  external data/checkpoint inventory
results/window_ablation/               versioned compact results
```
