# CNN–Mamba

CNN–Mamba predicts donor, acceptor, start-codon and stop-codon sites from DNA.
An auxiliary four-state head predicts `non-CDS/frame0/frame1/frame2` and supplies
coding-frame probabilities to UniAnn.

The supported pipeline is:

```text
FASTA + EviAnn labels + reference annotation
  -> window generation
  -> CNN–Mamba training
  -> chromosome-wide site scores
  -> candidate AUPRC
  -> vector-scaled four-state probabilities
  -> UniAnn
  -> strict locus evaluation
```

## 1. Installation

### Requirements

- Linux
- Python 3.10
- NVIDIA GPU with a CUDA-compatible driver
- Conda or Mamba
- Git
- A C/C++ compiler and `make` only if running UniAnn

The recorded environment uses PyTorch 2.6.0 with CUDA 12.4. The model requires
the pinned Git version of `mamba-ssm`; the PyPI package named 2.3.1 does not
contain the same Mamba3 implementation.

```bash
git clone https://github.com/bowencccc/cnn-mamba.git
cd cnn-mamba

conda env create -f environment.yml
conda activate cnn_mamba_k7
python -m pip install -e . --no-deps --no-build-isolation

cnn-mamba doctor
python scripts/smoke_model.py
```

`cnn-mamba doctor` reports the Python, PyTorch, Mamba and CUDA installation.
The smoke test performs a small GPU forward/backward pass.

If the Conda environment already exists, install the required Mamba3 source
with:

```bash
python -m pip install --no-build-isolation \
  "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@be0303b971cd79a4fbbcfc411a01b33f6b0e3602"
python -m pip install -e . --no-deps --no-build-isolation
```

### Optional: install UniAnn

UniAnn is required only for transcript/locus prediction. Training, site scoring
and candidate AUPRC do not require it.

```bash
git clone https://github.com/alekseyzimin/UniAnn.git
cd UniAnn
git checkout 2f5c0ab8fdb2e848f5538f3a1f741cb9c4c67e07
cd src
make clean && make
cd ..
./install.sh
export UNIANN_ROOT="$PWD"
cd /path/to/cnn-mamba

cnn-mamba doctor --uniann-root "$UNIANN_ROOT"
```

## 2. Required data

Large genomes and annotations are not stored in Git. Files may be copied or
symbolically linked into `data/raw/`. Commands also accept paths outside the
repository.

### Drosophila: training and held-out candidate evaluation

| Logical path | Purpose |
|---|---|
| `data/raw/dmel_genome.fa` | D. melanogaster genome FASTA |
| `data/raw/dmel_genome.fa.fai` | FASTA index |
| `data/raw/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pseudo_label.coleoptera.gff.gz` | Current Dmel EviAnn training labels, generated with Coleoptera protein evidence |
| `data/raw/dmel_reference.gtf` | reference labels and held-out evaluation |

### Additional files for Drosophila UniAnn evaluation

| Logical path | Purpose |
|---|---|
| `data/raw/dmel_chrX.fa` | chrX FASTA with sequence name `NC_004354.4` |
| `data/raw/dmel_chrX_plus_CDS_reference.gtf` | chrX+ strict-match reference |
| `data/processed/coleoptera_eviann_v1/chrX_plus_EviAnn_CDS.gff` | chrX+ annotation filtered from the Coleoptera-evidence EviAnn GFF for the union |

The current pipeline does **not** require PSAURON. It uses the model's
vector-scaled four-state probabilities as UniAnn's coding-frame input.
`psauron_score_droso.csv` is retained in `external_manifest.tsv` only for
legacy comparisons.

### Human data

| Logical path | Purpose |
|---|---|
| `data/raw/human_grch38.fa` | GRCh38.p14 genome FASTA |
| `data/raw/human_grch38.fa.fai` | FASTA index |
| `data/raw/GCF_000001405.40_GRCh38.p14_genomic.noalpsnopatches.fna.pseudo_label.rodentia.gff.gz` | Current Human EviAnn training labels, generated with Rodentia protein evidence |
| `data/raw/human_refseq_grch38_p14.gff.gz` | RefSeq validation and test reference |

CHESS and Human PSAURON files are optional historical inputs, not requirements
for the current training pipeline.

Example setup using symbolic links:

```bash
mkdir -p data/raw
ln -s /data/genomes/dmel.fa data/raw/dmel_genome.fa
ln -s /data/genomes/dmel.fa.fai data/raw/dmel_genome.fa.fai
ln -s /data/annotations/dmel_coleoptera_eviann.gff.gz \
  data/raw/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pseudo_label.coleoptera.gff.gz
ln -s /data/annotations/dmel_reference.gtf data/raw/dmel_reference.gtf
```

`external_manifest.tsv` records the exact files used in the published
experiments, including sizes and SHA256 hashes. To reproduce those exact inputs:

```bash
python scripts/check_external.py --group all --fast  # existence and size
python scripts/check_external.py --group all         # also verify SHA256
```

These checks are intentionally strict. They will report a mismatch when using
a different genome or annotation release.

## 3. Recommended Drosophila workflow

The versioned workflow is the simplest way to run the complete pipeline.

```bash
cp configs/workflow.droso.example.json configs/workflow.droso.local.json
```

Edit only the `variables` block at the top of the copied file if your inputs do
not use the logical paths above. Then inspect and run it:

```bash
export UNIANN_ROOT=/path/to/UniAnn

cnn-mamba workflow configs/workflow.droso.local.json --dry-run
cnn-mamba workflow configs/workflow.droso.local.json
```

The workflow performs:

1. 10 kb window generation with 5 kb stride;
2. chrX-held-out training and four-task AUPRC;
3. a second model with chrX included for the deployment/UniAnn setting;
4. validation-set vector scaling of the four-state head;
5. chromosome-wide site and four-state scoring;
6. UniAnn, `gffread -M`, and strict `gffcompare`;
7. separate UniAnn and EviAnn+UniAnn locus metrics.

Important outputs:

```text
runs/software_example/heldout/best_model.pt
runs/software_example/deployment/best_model.pt
results/software_example/heldout_candidate_auprc.tsv
results/software_example/vector_scaling/vector_scaling.json
results/software_example/uniann/locus_metrics.tsv
workflow_runs/<timestamp>/run.json
```

The default coding-frame recipe is vector scaling followed by UniAnn's
built-in-equivalent binary-odds transform (`T=3`) and non-CDS gate. The four
primary donor/acceptor/start/stop scores are not scaled.

### Required UniAnn TSV header convention

The exported `*_joint_frame_probabilities.tsv` header is part of the tested
CNN--Mamba/UniAnn interface. Pass that TSV to UniAnn **with its header intact**.
Do not use `tail -n +2`, `sed 1d`, or another command that removes it.

CNN--Mamba labels `frame0/frame1/frame2` as codon-nucleotide offsets. UniAnn
commit `2f5c0ab` counts coding-score input rows instead of reading the TSV
position column. The single header row supplies the required +1 counter offset,
giving the effective mapping `(zero_based_position - model_phase) mod 3`.
Removing the header rotates every exon-frame emission by one state and can
silently change strict locus results. The canonical workflow validates this
contract with `--require-cnn-mamba-tsv-header` before running UniAnn.

This convention does not change the TSV format or UniAnn. It records and tests
the exact interface used by the collaborator-facing workflow. See
[`docs/UNIANN_FRAME_TSV_CONTRACT.md`](docs/UNIANN_FRAME_TSV_CONTRACT.md) for the
derivation, regression values, and migration guidance.

## 4. Run the main steps manually

Use the unified `cnn-mamba` command. Run `cnn-mamba COMMAND --help` for all
options.

### Generate 10 kb windows

```bash
cnn-mamba prepare-droso \
  --config configs/droso_frame_w10.json \
  --fasta data/raw/dmel_genome.fa \
  --eviann-gff data/raw/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.pseudo_label.coleoptera.gff.gz \
  --reference-gtf data/raw/dmel_reference.gtf \
  --output-root data/processed/droso \
  --include-phase-labels
```

This creates:

```text
data/processed/droso/w10_frame/train/
data/processed/droso/w10_frame/val/
data/processed/droso/w10_frame/test/
```

The Drosophila split is train = `2L/2R/3R/4/Y`, validation = `3L`, and test =
`X`. Both strands are used in training.

### Train a chrX-held-out model

```bash
cnn-mamba train \
  --data-root data/processed/droso/w10_frame \
  --mode baseline \
  --architecture cnn_mamba \
  --cnn-kernel-size 7 \
  --frame-dilation 3 \
  --joint-frame-aux-weight 0.1 \
  --joint-noncoding-weight 0.1 \
  --window-size 10000 \
  --stride 5000 \
  --epochs 30 \
  --early-stopping-patience 3 \
  --early-stopping-min-delta 0.0001 \
  --amp --amp-dtype bf16 \
  --output-dir runs/droso/heldout
```

`--amp --amp-dtype bf16` is the recommended precision: about 1.2x (10 kb,
micro-batch 4) to 1.4x (20 kb) faster than fp32 on an RTX A4500, with less
memory and no GradScaler. Avoid `--amp-dtype fp16`: it produced non-finite
forward passes and false early stopping in 20 kb Drosophila training.

### Score chrX+ and calculate AUPRC

```bash
cnn-mamba score \
  --checkpoint runs/droso/heldout/best_model.pt \
  --fasta data/raw/dmel_genome.fa \
  --chrom NC_004354.4 \
  --display-chrom chrX \
  --output-dir artifacts/scores/droso/heldout

cnn-mamba auprc \
  --score-dir artifacts/scores/droso/heldout \
  --reference-gtf data/raw/dmel_reference.gtf \
  --fasta data/raw/dmel_genome.fa \
  --chrom NC_004354.4 \
  --output results/droso/heldout_candidate_auprc.tsv
```

The output reports exact AUPRC for donor, acceptor, start and stop over all
canonical chrX+ candidates.

### Train the deployment model used with UniAnn

Use the same training command with these changes:

```bash
  --include-test-in-train \
  --output-dir runs/droso/deployment
```

This model includes chrX in training while retaining chromosome 3L for model
selection. It is appropriate for the intended EviAnn-to-reference augmentation
setting, but it is not a held-out chrX generalization test.

Vector calibration and UniAnn emission construction require several connected
steps. Use `configs/workflow.droso.example.json` as the canonical executable
recipe instead of manually copying those commands.

For a manual run, first export the headered TSV and then pass it directly to
the evaluation wrapper:

```bash
cnn-mamba export-joint-frame \
  --probabilities artifacts/joint_frame_predictions/droso/deployment/joint_frame_probabilities.npy \
  --argmax artifacts/joint_frame_predictions/droso/deployment/joint_frame_argmax.npy \
  --output artifacts/joint_frame_predictions/droso/deployment/chrX_plus_joint_frame_probabilities.tsv \
  --chrom chrX \
  --strand +

cnn-mamba uniann-eval \
  --uniann-root "$UNIANN_ROOT" \
  --fasta data/raw/dmel_chrX.fa \
  --coding-frame artifacts/joint_frame_predictions/droso/deployment/chrX_plus_joint_frame_probabilities.tsv \
  --require-cnn-mamba-tsv-header \
  --scores artifacts/scores/droso/deployment/chrX_plus_scores.tsv \
  --reference-gtf data/raw/dmel_chrX_plus_CDS_reference.gtf \
  --eviann-gff data/raw/dmel_chrX_plus_EviAnn_CDS.gff \
  --output-dir results/droso/deployment/uniann
```

The wrapper deliberately leaves UniAnn unchanged. UniAnn performs its normal
built-in binary-odds transform and non-CDS gate on the headered TSV.

## 5. Use a published checkpoint without training

Inference-only checkpoints are included under `models/`. For example:

```bash
cnn-mamba score \
  --checkpoint models/drosophila_joint_frame_beta010_chrxtrain_epoch23_inference.pt \
  --fasta data/raw/dmel_genome.fa \
  --chrom NC_004354.4 \
  --display-chrom chrX \
  --output-dir artifacts/scores/published_droso_model
```

This checkpoint includes chrX in training. Use it for deployment tests, not as
a held-out chrX result. See `models/README.md` for checkpoint provenance.

## 6. Human training

Generate RefSeq-labeled Human windows:

```bash
cnn-mamba prepare-human \
  --fasta data/raw/human_grch38.fa \
  --eviann-gff data/raw/GCF_000001405.40_GRCh38.p14_genomic.noalpsnopatches.fna.pseudo_label.rodentia.gff.gz \
  --reference-annotation data/raw/human_refseq_grch38_p14.gff.gz \
  --output-dir data/processed/human_refseq_w10_chr1held \
  --window-size 10000 \
  --stride 5000 \
  --max-windows-per-chrom-strand 0 \
  --region-overlap-bp 10000 \
  --include-phase-labels
```

The default Human split is listed in `configs/human_cnn_mamba_k7.json`: chr1 is
test; chr2/5/9/10/22 are validation; the remaining configured chromosomes are
training. Train it with the same `cnn-mamba train` options used above, changing
only `--data-root` and `--output-dir`.

Add `--include-test-in-train` only when intentionally training the chr1-included
deployment model. Do not report that run as held-out chr1 performance.

## 7. Evaluation regimes

| Regime | Test chromosome in training? | Use |
|---|---:|---|
| `heldout` | No | model generalization and candidate AUPRC |
| `deployment` | Yes | EviAnn+UniAnn augmentation in the intended use case |

Always report the regime with the result. The two regimes answer different
questions and should not be compared as if they were the same test.

## 8. Current-best inputs for downstream Viterbi code

Downstream structured-decoder experiments use two fixed evidence sources:

- Human: Rodentia-evidence EviAnn annotation;
- Dmel: Coleoptera-evidence EviAnn annotation.

The versioned registry is `configs/viterbi_current_best.json`. It promotes a
checkpoint explicitly by validation mean AP; it never guesses from file dates.
Materialize stable links after copying results to a server or after promoting a
new experiment:

```bash
cnn-mamba current-best
```

Viterbi code can then use paths that do not change between experiments:

```text
artifacts/current_best/viterbi/human/best_model.pt
artifacts/current_best/viterbi/human/windows/
artifacts/current_best/viterbi/human/eviann_annotation.gff.gz
artifacts/current_best/viterbi/dmel/best_model.pt
artifacts/current_best/viterbi/dmel/windows/
artifacts/current_best/viterbi/dmel/site_scores.tsv
artifacts/current_best/viterbi/dmel/joint_frame_probabilities.npy
```

The command reports `model_ready` and `prediction_ready` separately. It removes
a stale managed link when its promoted source is missing, so downstream code
cannot silently consume an older annotation/model combination. See
[`results/viterbi_current_best/README.md`](results/viterbi_current_best/README.md)
for the promoted metrics and exact commands.

## 9. Repository layout

```text
src/cnn_mamba/     reusable model, training, scoring and evaluation code
configs/           model, data and workflow configurations
models/            small published inference checkpoints
scripts/           historical experiments and figure-generation recipes
data/raw/          external genomes and annotations; ignored by Git
data/processed/    generated NPZ windows; ignored by Git
runs/              training checkpoints; ignored by Git
artifacts/         chromosome scores and probabilities; ignored by Git
results/           compact final tables and figures
work/              UniAnn and other temporary working directories
```

For implementation boundaries, see `docs/SOFTWARE_LAYOUT.md`. For exact
published data/checkpoint hashes, see `external_manifest.tsv`. Historical
experiments are indexed in `results/EXPERIMENT_INDEX.md` and `scripts/README.md`.
