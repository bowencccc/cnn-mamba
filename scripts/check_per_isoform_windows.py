#!/usr/bin/env python3
"""Consistency checks of per-isoform windows against the union windows.

For every split in {train, test} and every grid position:
  * the set of (strand, start) positions equals the union dataset's;
  * every version's EviAnn positives (donor/acceptor/start/stop) are a subset of
    the union labels, and the union over versions equals the union labels;
  * sequence and reference arrays are identical to the union window;
  * where the union codon phase is defined (no isoform conflict), every version's
    defined phase agrees with it.
usage: check_per_isoform_windows.py UNION_CELL_DIR PER_ISOFORM_CELL_DIR
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

union_dir, iso_dir = map(Path, sys.argv[1:3])
for split in ("train", "test"):
    groups = defaultdict(list)
    for f in (iso_dir / split).glob("*.npz"):
        groups[f.stem.rsplit("_", 1)[0]].append(f)
    union = {f.stem: f for f in (union_dir / split).glob("*.npz")}
    assert set(groups) == set(union), (split, len(set(groups) ^ set(union)))
    n_multi = n_phase_checked = 0
    for key, files in groups.items():
        with np.load(union[key]) as u:
            U = {k: u[k] for k in u.files}
        cover = {k: np.zeros_like(U[k], dtype=bool) for k in ("labels", "start_stop_labels")}
        for f in files:
            with np.load(f) as d:
                for k in ("sequence", "chess_labels", "chess_start_stop_labels",
                          "reference_phase_labels", "reference_phase_mask", "reference_cds_labels"):
                    assert np.array_equal(d[k], U[k]), (f, k)
                for k in ("labels", "start_stop_labels"):
                    pos = d[k] > 0
                    assert np.array_equal(d[k][pos], U[k][pos]), (f, k, "not a subset of union")
                    cover[k] |= pos
                ok = U["phase_mask"] & d["phase_mask"]
                assert np.array_equal(d["phase_labels"][ok], U["phase_labels"][ok]), (f, "phase")
                n_phase_checked += int(ok.sum())
                assert not (d["cds_labels"] & ~U["cds_labels"]).any(), (f, "cds outside union")
        for k in cover:
            assert np.array_equal(cover[k], U[k] > 0), (key, k, "union of versions != union")
        n_multi += len(files) > 1
    print(f"{split}: {len(groups):,} positions, {sum(map(len, groups.values())):,} windows, "
          f"{n_multi:,} positions with >1 label version, {n_phase_checked:,} phase bases checked: OK")
