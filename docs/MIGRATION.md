# Moving to another GPU server

1. Push/clone this Git repository (about 5 MB; no biological data or model
   weights are included).
2. Install the environment and UniAnn at the pinned commit from `README.md`.
3. Either fill `data/raw/` and `artifacts/checkpoints/` using
   `external_manifest.tsv`, or materialize files on this server first:

   ```bash
   bash scripts/materialize_external.sh /path/to/external_bundle all
   rsync -avh --info=progress2 /path/to/external_bundle/ newserver:/path/to/repo/
   ```

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
