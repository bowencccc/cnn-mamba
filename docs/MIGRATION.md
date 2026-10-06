# Moving to another GPU server

1. Push/clone this Git repository (tens of MB including compact figures; no
   biological data or model weights are included).
2. Install the environment and UniAnn at the pinned commit from `README.md`.
3. Either fill `data/raw/` and `artifacts/checkpoints/` using
   `external_manifest.tsv`, or materialize files on this server first:

   ```bash
   bash scripts/materialize_external.sh /path/to/external_bundle all
   rsync -avh --info=progress2 /path/to/external_bundle/ newserver:/path/to/repo/
   ```

   Available groups are `drosophila` (about 1.18 GiB), `human` (about
   5.93 GiB, including full Human PSAURON), or `all` (about 7.11 GiB).

4. On the destination, run `python scripts/check_external.py` to verify all
   files byte-for-byte.
5. Generate `data/processed` locally on the destination. Generated NPZ windows
   need not cross the network unless regeneration time matters more than
   transfer size.
6. Start one `scripts/run_cell.py` process per GPU. Each cell has its own output
   directory, so cells can run concurrently. Do not run the same cell/regime
   twice at the same time.

For code-only work, omit the checkpoint group. For evaluation from existing
weights, transfer both raw and checkpoint groups but skip window generation and
training.

## Rockfish quick start

The Git repository contains code plus compact result tables/figures, but not
raw genomes, annotations, NPZ windows, checkpoints, chromosome-wide arrays or
UniAnn work directories. On Rockfish:

```bash
git clone git@github.com:bowencccc/cnn-mamba.git
cd cnn-mamba

# If the previously transferred bundle is at this location:
rsync -a ~/cnn_mamba_k7_transfer/external/ ./
python scripts/check_external.py --fast
```

Generate derived windows on Rockfish rather than transferring them. For the
four-state frame experiment:

```bash
cnn-mamba-prepare-droso \
  --config configs/droso_frame_w10.json \
  --output-root data/processed/cds_gating \
  --include-phase-labels

sbatch --account=<PI_NAME>_gpu scripts/rockfish_joint_frame_grid.slurm
```

The six array tasks are independent: tasks 0--2 hold out chrX and tasks 3--5
include chrX for UniAnn evaluation. Do not use the include-chrX models as an
honest held-out generalization estimate.
