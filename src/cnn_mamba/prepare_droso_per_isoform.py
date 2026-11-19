#!/usr/bin/env python3
"""Per-isoform 1 kb training windows for the Drosophila EviAnn experiments.

``prepare_droso`` merges all EviAnn coding transcripts into regions, places one
window per grid position, and labels it with the union of every transcript
(codon-phase conflicts masked). This variant instead tiles every EviAnn coding
transcript separately on the same global stride grid and labels each window
with that transcript only, then removes exact duplicates:

* windows: for each transcript, every grid start whose window intersects the
  transcript span (first exon start .. last exon end), as in ``window_starts``;
* EviAnn labels (donor/acceptor, start/stop, codon phase, CDS): the transcript
  itself plus all transcripts of *other* EviAnn loci (union; phase conflicts
  among them masked). Alternative isoforms of the same locus are not added, so
  a site used only by another isoform of this locus is background here;
* deduplication: a window is kept once per (strand, start, EviAnn label arrays);
  windows whose labels are identical across isoforms collapse to one, windows
  whose labels differ are all kept;
* reference (RefSeq) arrays are the usual all-transcript union;
* ``val`` (3L) is copied unchanged from a union dataset so validation and
  checkpoint selection are identical to the union experiments. ``train`` and
  ``test`` (chrX, trained on in the chrxtrain regime) use per-isoform windows.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import bisect
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pysam

import cnn_mamba.prepare_droso as pd_

SPLITS_PER_ISOFORM = ("train", "test")


def parse_loci_and_phase(path: Path):
    """Return {tid: locus} and {tid: [(start0, end0, phase)]} for EviAnn mRNAs."""
    accessions = set(pd_.CHROMS.values())
    locus, phase = {}, defaultdict(list)
    with pd_.open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip().split("\t")
            if len(f) < 9 or f[0] not in accessions:
                continue
            attrs = pd_.gff_attrs(f[8])
            if f[2] == "mRNA" and attrs.get("ID"):
                locus[attrs["ID"]] = attrs.get("Parent") or attrs.get("geneID") or attrs["ID"]
            elif f[2] == "CDS" and f[7] in {"0", "1", "2"} and f[6] in {"+", "-"}:
                for tid in attrs.get("Parent", "").split(","):
                    if tid:
                        phase[tid].append((int(f[3]) - 1, int(f[4]), int(f[7])))
    return locus, phase


def local_phase(records, w0: int, width: int, strand: str):
    """``build_phase_track`` restricted to [w0, w0 + width) (genomic order)."""
    track = np.full(width, -1, dtype=np.int8)
    cds = np.zeros(width, dtype=np.bool_)
    for start, end, ph in records:
        lo, hi = max(start, w0), min(end, w0 + width)
        if hi <= lo:
            continue
        initial = (3 - ph) % 3
        offsets = np.arange(end - start, dtype=np.int64)
        if strand == "-":
            offsets = offsets[::-1]
        values = ((initial + offsets) % 3).astype(np.int8)[lo - start:hi - start]
        view = track[lo - w0:hi - w0]
        unknown = view == -1
        conflict = (view >= 0) & (view != values)
        view[unknown] = values[unknown]
        view[conflict] = -2
        cds[lo - w0:hi - w0] = True
    return track, cds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument("--eviann-gff", type=Path, required=True)
    p.add_argument("--reference-gtf", type=Path, required=True)
    p.add_argument("--union-root", type=Path, required=True,
                   help="existing prepare_droso output root whose <cell>/val is copied")
    p.add_argument("--output-root", type=Path, required=True)
    a = p.parse_args()
    pd_.FASTA, pd_.EVIANN_GFF, pd_.REFERENCE_GTF = (
        a.fasta.resolve(), a.eviann_gff.resolve(), a.reference_gtf.resolve())
    config = json.loads(a.config.read_text())
    t0 = time.time()

    print("parsing annotations", flush=True)
    meta, exons, cds, _ = pd_.parse_eviann(pd_.EVIANN_GFF, "all")
    locus, phase = parse_loci_and_phase(pd_.EVIANN_GFF)
    tids = sorted(t for t in cds if cds[t] and t in meta)
    missing_locus = [t for t in tids if t not in locus]
    assert not missing_locus, f"transcripts without mRNA locus: {missing_locus[:5]}"

    # per-transcript sites, extracted with the exact prepare_droso rules
    own = {}
    rejected = defaultdict(int)
    for t in tids:
        s, r = pd_.extract_sites({t: meta[t]}, {t: exons[t]} if t in exons else {}, {t: cds[t]})
        own[t] = s
        for k, v in r.items():
            rejected[k] += v
    # site -> loci providing it, per (site type, chrom, strand)
    by_loci = {n: defaultdict(lambda: defaultdict(set)) for n in pd_.SITE_TYPES}
    for t in tids:
        for n in pd_.SITE_TYPES:
            for chrom, strand, pos in own[t][n]:
                by_loci[n][(chrom, strand)][pos].add(locus[t])
    sorted_pos = {n: {k: sorted(v) for k, v in d.items()} for n, d in by_loci.items()}
    # CDS phase records of every transcript, per (chrom, strand), sorted by start
    recs = defaultdict(list)
    for t in tids:
        chrom, strand = meta[t]
        for s, e, ph in phase.get(t, []):
            recs[(chrom, strand)].append((s, e, ph, locus[t]))
    rec_starts, max_len = {}, {}
    for k in recs:
        recs[k].sort()
        rec_starts[k] = [r[0] for r in recs[k]]
        max_len[k] = max(r[1] - r[0] for r in recs[k])

    r_meta, r_exons, r_cds, explicit = pd_.parse_reference(pd_.REFERENCE_GTF)
    r_sites, r_rejected = pd_.extract_sites(r_meta, r_exons, r_cds, explicit)
    r_index = pd_.index_sites(r_sites)
    r_phase = pd_.parse_cds_phase_records(pd_.REFERENCE_GTF)

    completed = []
    fasta = pysam.FastaFile(str(pd_.FASTA))
    for cell in config["cells"]:
        W, stride = int(cell["window_bp"]), int(cell["stride_bp"])
        out = a.output_root / cell["id"]
        if out.exists() and any(out.iterdir()):
            raise RuntimeError(f"refusing non-empty output: {out}")
        summary = {}
        for split in SPLITS_PER_ISOFORM:
            sd = out / split
            sd.mkdir(parents=True, exist_ok=True)
            st = {"windows": 0, "positions": 0, "multi_label_positions": 0,
                  "transcript_windows_before_dedup": 0, "skipped_N": 0,
                  "by_chrom": {}, "eviann_positive_bases": defaultdict(int)}
            for name in pd_.SPLITS[split]:
                chrom = pd_.CHROMS[name]
                L = fasta.get_reference_length(chrom)
                ref_tracks = {
                    s: (pd_.build_phase_track(r_phase[chrom][s], L, s),
                        pd_.build_cds_track(r_phase[chrom][s], L)) for s in ("+", "-")}
                n_chrom = 0
                for strand in ("+", "-"):
                    key = (chrom, strand)
                    r_local = {n: r_index[n][chrom][strand] for n in pd_.SITE_TYPES}
                    by_start = defaultdict(list)  # grid start -> transcripts covering it
                    for t in tids:
                        if meta[t] != key:
                            continue
                        span = exons.get(t) or cds[t]
                        region = [(min(x[0] for x in span), max(x[1] for x in span))]
                        for w0 in pd_.window_starts(region, L, W, stride):
                            by_start[w0].append(t)
                    for w0 in sorted(by_start):
                        text = fasta.fetch(chrom, w0, w0 + W).upper()
                        if len(text) != W or text.count("N") > W // 2:
                            st["skipped_N"] += 1
                            continue
                        r_splice, r_ss = pd_.labels_for_window(w0, strand, r_local, W)
                        r_ph = ref_tracks[strand][0][w0:w0 + W]
                        r_cd = ref_tracks[strand][1][w0:w0 + W]
                        seq_text = text.translate(pd_.COMPLEMENT)[::-1] if strand == "-" else text
                        sequence = pd_.ENCODE[np.frombuffer(seq_text.encode("ascii"), dtype=np.uint8)]
                        # other-loci records overlapping the window
                        rs = recs.get(key, [])
                        lo_i = bisect.bisect_left(rec_starts.get(key, []), w0 - max_len.get(key, 0))
                        hi_i = bisect.bisect_left(rec_starts.get(key, []), w0 + W)
                        window_recs = [r for r in rs[lo_i:hi_i] if r[1] > w0]
                        seen = set()
                        st["positions"] += 1
                        for t in by_start[w0]:
                            st["transcript_windows_before_dedup"] += 1
                            lt = locus[t]
                            indexed = {}
                            for n in pd_.SITE_TYPES:
                                pos = sorted_pos[n].get(key, [])
                                i0 = bisect.bisect_left(pos, w0)
                                i1 = bisect.bisect_right(pos, w0 + W)
                                sel = {q for q in pos[i0:i1] if by_loci[n][key][q] - {lt}}
                                sel |= {q for c, s_, q in own[t][n] if w0 <= q <= w0 + W}
                                indexed[n] = sorted(sel)
                            e_splice, e_ss = pd_.labels_for_window(w0, strand, indexed, W)
                            ph_recs = [(s, e, ph) for s, e, ph, l in window_recs if l != lt]
                            ph_recs += [(s, e, ph) for s, e, ph in phase.get(t, [])]
                            e_ph, e_cd = local_phase(ph_recs, w0, W, strand)
                            r_ph_w, r_cd_w = r_ph, r_cd
                            if strand == "-":
                                e_ph, e_cd = e_ph[::-1], e_cd[::-1]
                                r_ph_w, r_cd_w = r_ph[::-1], r_cd[::-1]
                            digest = hashlib.sha1(b"".join(
                                np.ascontiguousarray(x).tobytes()
                                for x in (e_splice, e_ss, e_ph, e_cd))).hexdigest()
                            if digest in seen:
                                continue
                            seen.add(digest)
                            k = len(seen) - 1
                            fn = f"{name}_{'plus' if strand == '+' else 'minus'}_{w0}_{k}.npz"
                            np.savez_compressed(
                                sd / fn, sequence=sequence, labels=e_splice,
                                start_stop_labels=e_ss, chess_labels=r_splice,
                                chess_start_stop_labels=r_ss,
                                phase_labels=np.maximum(e_ph, 0), phase_mask=e_ph >= 0,
                                reference_phase_labels=np.maximum(r_ph_w, 0),
                                reference_phase_mask=r_ph_w >= 0,
                                cds_labels=e_cd, reference_cds_labels=r_cd_w)
                            for n, arr, cls in (("donor", e_splice, 1), ("acceptor", e_splice, 2),
                                                ("start", e_ss, 1), ("stop", e_ss, 2)):
                                st["eviann_positive_bases"][n] += int((arr == cls).sum())
                            st["windows"] += 1
                            n_chrom += 1
                        if len(seen) > 1:
                            st["multi_label_positions"] += 1
                st["by_chrom"][name] = n_chrom
                print(f"[{cell['id']}:{split}] {name}: {n_chrom:,}", flush=True)
            st["eviann_positive_bases"] = dict(st["eviann_positive_bases"])
            summary[split] = st
        # validation: copy the union windows unchanged
        src_val = a.union_root / cell["id"] / "val"
        shutil.copytree(src_val, out / "val")
        n_val = sum(1 for _ in (out / "val").glob("*.npz"))
        union_manifest = json.loads((a.union_root / cell["id"] / "split_manifest.json").read_text())
        summary["val"] = {**union_manifest["summary"]["val"], "copied_from": str(src_val.resolve())}
        assert n_val == int(summary["val"]["windows"]), (n_val, summary["val"]["windows"])
        manifest = {
            "schema": "droso_window_cell_per_isoform_v1", "cell": cell["id"],
            "window_size": W, "stride": stride,
            "window_policy": (
                "train/test: every EviAnn coding transcript tiled separately on the global "
                "stride grid; labels = that transcript + all transcripts of other loci "
                "(alternative isoforms of the same locus excluded); exact duplicates "
                "(strand, start, EviAnn label arrays) removed. val: copied from union dataset."),
            "splits": pd_.SPLITS, "accessions": pd_.CHROMS, "summary": summary,
            "union_root": str(a.union_root.resolve()),
            "sources": {
                "fasta": {"path": str(pd_.FASTA), "sha256": pd_.sha256(pd_.FASTA)},
                "eviann": {"path": str(pd_.EVIANN_GFF), "sha256": pd_.sha256(pd_.EVIANN_GFF)},
                "reference": {"path": str(pd_.REFERENCE_GTF), "sha256": pd_.sha256(pd_.REFERENCE_GTF)},
            },
        }
        pd_.atomic_json(out / "split_manifest.json", manifest)
        completed.append({"cell": cell["id"], "windows": {s: summary[s]["windows"] for s in summary}})
    fasta.close()
    pd_.atomic_json(a.output_root / ".complete.json", {
        "schema": "droso_window_generation_per_isoform_complete_v1", "cells": completed,
        "eviann_transcripts": len(tids), "eviann_loci": len({locus[t] for t in tids}),
        "eviann_rejected_per_transcript": dict(rejected), "reference_rejected": r_rejected,
        "elapsed_seconds": time.time() - t0,
    })
    print(json.dumps(completed, indent=2), flush=True)


if __name__ == "__main__":
    main()
