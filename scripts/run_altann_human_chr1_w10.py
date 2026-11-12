#!/usr/bin/env python3
"""Run AltAnn K=10 on the existing Human chr1 T=9 UniAnn chunks."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "results/human_joint_frame_beta010_es5_v1/chr1train/"
    "epoch10_joint_softmax_t9_uniann_latest"
)
WORK = ROOT / "work/altann/human_chr1_w10_epoch10_t9_k10"
RESULTS = ROOT / "results/altann_human_chr1_w10_epoch10_t9_k10"
ALTANN_ROOT = Path(os.environ.get("ALTANN_ROOT", ROOT.parent / "AltAnn"))
UNIANN_ROOT = Path(os.environ.get("UNIANN_ROOT", ROOT.parent / "UniAnn"))
ALTANN = ALTANN_ROOT / "bin/altann"
UNIANN = UNIANN_ROOT / "bin/uniann.sh"
GFFREAD = UNIANN_ROOT / "bin/gffread"
GFFCOMPARE = UNIANN_ROOT / "bin/gffcompare"
CHROMOSOME_LENGTH = 248_956_422
SEQID = "NC_000001.11"
# AltAnn's optional best-GFF validation found different rerun-Viterbi global
# paths in these chunks, despite exact UniAnn feature reproduction. They are
# decoded in documented rerun mode without supplying the validation GFF.
ALTANN_BASELINE_MISMATCH_CHUNKS = [
    "chunk_029", "chunk_033", "chunk_040", "chunk_041", "chunk_049",
]


def run(command: list[object], *, cwd: Path, stdout: Path, stderr: Path) -> None:
    with stdout.open("w") as out, stderr.open("w") as err:
        subprocess.run(
            [str(value) for value in command], cwd=cwd, stdout=out,
            stderr=err, check=True,
        )


def ensure_link(source: Path, destination: Path) -> None:
    source = source.resolve()
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"Refusing to replace {destination}")
    destination.symlink_to(source)


def feature_signature(path: Path) -> list[tuple[str, ...]]:
    signature = []
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            signature.append(tuple(fields[:8]))
    return signature


def process_chunk(chunk: dict[str, object], threads: int) -> dict[str, object]:
    chunk_id = str(chunk["id"])
    source_chunk = SOURCE / "chunks" / chunk_id
    work = WORK / "chunks" / chunk_id
    work.mkdir(parents=True, exist_ok=True)
    fasta = Path(str(chunk["fasta"]))
    emissions = source_chunk / "emissions/out.ps.txt"
    scores = source_chunk / "scores/baseline.tsv"
    expected = SOURCE / "runs" / chunk_id / "baseline/prediction.raw.gff"
    ensure_link(fasta, work / "genome.fa")
    ensure_link(emissions, work / "out.ps.txt")
    ensure_link(scores, work / "scores.txt")

    baseline = work / "genome.fa.uniann.gff"
    expected_signature = feature_signature(expected)
    if not expected_signature:
        # Centromeric chunks can have no donor with a positive transformed
        # score. UniAnn's wrapper stops before writing transition tables, and
        # the established chunk pipeline records a valid empty annotation.
        # AltAnn represents the same input with empty sparse transition files.
        if not baseline.is_file():
            baseline.write_text("##gff-version 3\n")
        for name in ("out.gt.txt", "out.ag.txt", "out.atg.txt", "out.stop.txt"):
            (work / name).touch()
    elif not baseline.is_file():
        run(
            [UNIANN, "-f", "genome.fa", "-c", "out.ps.txt", "-s", "scores.txt", "-n"],
            cwd=work,
            stdout=work / "uniann.stdout.log",
            stderr=work / "uniann.stderr.log",
        )
    if feature_signature(baseline) != expected_signature:
        raise RuntimeError(f"UniAnn baseline structure differs for {chunk_id}")

    output = work / "altann_k10.all_context.gff3"
    if not output.is_file():
        run(
            [
                ALTANN, "decode", "--fasta", fasta,
                "--emissions", work / "out.ps.txt",
                "--gt", work / "out.gt.txt", "--ag", work / "out.ag.txt",
                "--atg", work / "out.atg.txt", "--stop", work / "out.stop.txt",
                "--rerun-viterbi", "--k", 10, "--short-cds-filter",
                "--threads", threads, "--seqid", SEQID,
                "--offset", int(chunk["context_start0"]),
                "--sequence-length", CHROMOSOME_LENGTH,
                # Retain the whole mapped context here. Exact asymmetric core
                # ownership is applied during stitching below.
                "--segment-overlap", 0, "--segment-margin", 0,
                "--output", output,
            ],
            cwd=work,
            stdout=work / "altann.stdout.log",
            stderr=work / "altann.stderr.log",
        )
    metadata = json.loads(Path(str(output) + ".json").read_text())
    return {
        "chunk_id": chunk_id,
        "output": str(output),
        "seconds": metadata["seconds"],
    }


def transcript_blocks(path: Path):
    block: list[str] = []
    start = end = None
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if fields[2] in ("transcript", "mRNA"):
                if block:
                    yield start, end, block
                block = [line]
                start, end = int(fields[3]), int(fields[4])
            else:
                block.append(line)
    if block:
        yield start, end, block


def stitch(chunks: list[dict[str, object]]) -> tuple[Path, dict[str, int]]:
    output = WORK / "altann_k10.stitched.raw.gff3"
    counts = {"retained": 0, "rejected_midpoint": 0, "rejected_edge": 0}
    with output.open("w") as destination:
        destination.write("##gff-version 3\n")
        for chunk in chunks:
            source = WORK / "chunks" / str(chunk["id"]) / "altann_k10.all_context.gff3"
            core_start = int(chunk["core_start0"])
            core_end = int(chunk["core_end0"])
            context_start = int(chunk["context_start0"])
            context_end = int(chunk["context_end0"])
            for start1, end1, block in transcript_blocks(source):
                midpoint0 = ((start1 - 1) + end1) // 2
                if not core_start <= midpoint0 < core_end:
                    counts["rejected_midpoint"] += 1
                    continue
                if (
                    (context_start > 0 and start1 <= context_start + 1)
                    or (context_end < CHROMOSOME_LENGTH and end1 >= context_end)
                ):
                    counts["rejected_edge"] += 1
                    continue
                destination.writelines(block)
                counts["retained"] += 1
    return output, counts


def concatenate(inputs: list[Path], output: Path, *, gff_header: bool = True) -> None:
    with output.open("w") as destination:
        if gff_header:
            destination.write("##gff-version 3\n")
        for path in inputs:
            with path.open() as source:
                for line in source:
                    if not line.startswith("#"):
                        destination.write(line)


def strict_compare(query: Path, name: str, reference: Path) -> Path:
    evaluation = WORK / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    merged = evaluation / f"{name}.merged.gff"
    run(
        [GFFREAD, "-M", query, "-o", merged], cwd=evaluation,
        stdout=evaluation / f"{name}.gffread.stdout.log",
        stderr=evaluation / f"{name}.gffread.stderr.log",
    )
    prefix = evaluation / name
    run(
        [GFFCOMPARE, "--strict-match", "-e", 0, "-r", reference, merged, "-o", prefix],
        cwd=evaluation,
        stdout=evaluation / f"{name}.gffcompare.stdout.log",
        stderr=evaluation / f"{name}.gffcompare.stderr.log",
    )
    return evaluation / f"{name}.stats"


def parse_locus(path: Path) -> dict[str, float | int]:
    text = path.read_text()
    locus = re.search(r"Locus level:\s*([0-9.]+)\s*\|\s*([0-9.]+)", text)
    query = re.search(r"Query mRNAs\s*:\s*(\d+) in\s*(\d+) loci", text)
    if locus is None or query is None:
        raise RuntimeError(f"Could not parse {path}")
    sensitivity, precision = map(float, locus.groups())
    return {
        "sensitivity": sensitivity,
        "precision": precision,
        "f1": 2 * sensitivity * precision / (sensitivity + precision),
        "query_transcripts": int(query.group(1)),
        "query_loci": int(query.group(2)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=8)
    args = parser.parse_args()
    started = time.time()
    manifest = json.loads((SOURCE / "chunk_manifest.json").read_text())
    chunks = manifest["chunks"]
    WORK.mkdir(parents=True, exist_ok=True)
    completed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_chunk, chunk, args.threads_per_worker): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            record = future.result()
            completed.append(record)
            print(f"[{len(completed)}/{len(chunks)}] {record['chunk_id']}", flush=True)

    stitched, stitch_counts = stitch(chunks)
    eviann = SOURCE / "shared/eviann.chr1.plus.cds.raw.gtf"
    eviann_normalized = WORK / "eviann.normalized.gtf"
    altann_normalized = WORK / "altann_k10.normalized.gtf"
    run(
        [GFFREAD, "-T", eviann, "-o", eviann_normalized], cwd=WORK,
        stdout=WORK / "eviann.normalize.stdout.log",
        stderr=WORK / "eviann.normalize.stderr.log",
    )
    run(
        [GFFREAD, "-T", stitched, "-o", altann_normalized], cwd=WORK,
        stdout=WORK / "altann.normalize.stdout.log",
        stderr=WORK / "altann.normalize.stderr.log",
    )
    union = WORK / "eviann_altann_k10.concat.gtf"
    concatenate([eviann_normalized, altann_normalized], union, gff_header=False)
    reference = SOURCE / "strict_evaluation/reference.full.eligible.merged.gtf"
    altann_stats = strict_compare(stitched, "altann_k10", reference)
    union_stats = strict_compare(union, "eviann_altann_k10", reference)

    old_rows = {
        row["query_mode"]: row
        for row in json.loads(
            (SOURCE / "strict_evaluation/SUMMARY.complete.json").read_text()
        )["rows"]
    }
    rows = [
        {"method": "UniAnn", **{
            "sensitivity": old_rows["uniann_only"]["locus_sensitivity_percent"],
            "precision": old_rows["uniann_only"]["locus_precision_percent"],
            "f1": old_rows["uniann_only"]["locus_f1_percent"],
            "query_transcripts": old_rows["uniann_only"]["query_transcripts"],
            "query_loci": old_rows["uniann_only"]["query_loci"],
        }},
        {"method": "AltAnn K=10", **parse_locus(altann_stats)},
        {"method": "EviAnn + UniAnn", **{
            "sensitivity": old_rows["eviann_plus_uniann"]["locus_sensitivity_percent"],
            "precision": old_rows["eviann_plus_uniann"]["locus_precision_percent"],
            "f1": old_rows["eviann_plus_uniann"]["locus_f1_percent"],
            "query_transcripts": old_rows["eviann_plus_uniann"]["query_transcripts"],
            "query_loci": old_rows["eviann_plus_uniann"]["query_loci"],
        }},
        {"method": "EviAnn + AltAnn K=10", **parse_locus(union_stats)},
    ]
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "locus_metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    (RESULTS / "provenance.json").write_text(json.dumps({
        "source": str(SOURCE.resolve()),
        "model": "Human four-state epoch 10, chr1 included, 10 kb/5 kb stride",
        "frame_scaling": "joint-softmax T=9",
        "psauron_used": False,
        "altann_commit": subprocess.check_output(
            ["git", "-C", ALTANN_ROOT, "rev-parse", "HEAD"], text=True
        ).strip(),
        "k": 10,
        "short_cds_filter": True,
        "chunks": len(chunks),
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "stitch_counts": stitch_counts,
        "altann_global_path_mismatch_chunks": ALTANN_BASELINE_MISMATCH_CHUNKS,
        "elapsed_seconds": time.time() - started,
        "metrics": rows,
    }, indent=2) + "\n")
    print(RESULTS / "locus_metrics.tsv")


if __name__ == "__main__":
    main()
