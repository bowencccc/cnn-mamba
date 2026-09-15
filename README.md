# CNN–Mamba-k7 portable pipeline

This repository is the cleaned, portable version of the current Drosophila
CNN–Mamba-k7 work. It covers the complete path from raw FASTA/annotations to
training windows, model training, chromosome-wide candidate scoring, exact
AUPRC, UniAnn input export, strict `gffcompare`, and EviAnn+UniAnn evaluation.

Small result tables, plots, training histories and strict-match statistics are
tracked in `results/`. Large biological inputs, generated NPZ windows, raw
score arrays and PyTorch checkpoints are excluded from Git and described in
`external_manifest.tsv` with exact byte sizes and SHA256 hashes.

## Reproducibility boundary

The preserved recipe is:

- model: 8-layer bidirectional Mamba3, `d_model=192`, `d_state=64`;
- a depthwise/pointwise CNN before every Mamba block, kernel size 7;
- two 3-class heads: background/donor/acceptor and background/start/stop;
- candidate-masked cross entropy, `0.25 * splice_loss + start_stop_loss`;
- AdamW, learning rate `3e-4`, cosine decay, seed 42, five epochs;
- 50% overlapping windows and eight windows per optimizer update;
- validation on chromosome 3L and held-out candidate evaluation on chrX+;
- UniAnn evaluation uses chrX included in training by design, followed by
  `gffread -M` and `gffcompare --strict-match -e 0`.

The window experiment is a legacy-recipe ablation, not a compute-normalized
comparison: larger windows receive fewer optimizer updates in five epochs.

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

## Human pipeline

The Human external group contains GRCh38, CHESS 3.1.3, the Human EviAnn
pseudo-label GFF, full Human PSAURON scores, and the current 10 kb
CNN--Mamba-k7 chr1-in-training checkpoint. Generated NPZ windows and chr1 score
arrays are omitted and can be rebuilt.

```bash
cnn-mamba-prepare-human \
  --fasta data/raw/human_grch38.fa \
  --eviann-gff data/raw/human_eviann_pseudo_label.gff \
  --chess-gtf data/raw/human_chess3.1.3.gtf \
  --output-dir data/processed/human_w10_chr1held \
  --max-windows-per-chrom-strand 0 \
  --region-overlap-bp 10000
```

Train an honest chr1-held-out model:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_human.py \
  --regime chr1held --stages train,score,auprc
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
scripts/run_cell.py                    end-to-end per-GPU driver
external_manifest.tsv                  external data/checkpoint inventory
results/window_ablation/               versioned compact results
```
