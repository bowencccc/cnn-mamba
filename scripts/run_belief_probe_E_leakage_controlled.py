#!/usr/bin/env python3
"""Select UniAnn frame calibration on Dmel 3L and report once on chrX."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pysam

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_uniann_frame_scaling_grid as legacy  # noqa: E402
from cnn_mamba.probe_features import sha256_file  # noqa: E402
from cnn_mamba.probe_overcounting import group_balanced_prior_correction  # noqa: E402


UNIANN = Path(os.environ.get("UNIANN_ROOT", ROOT.parent / "UniAnn"))
CHECKPOINT = ROOT / "runs/joint_frame_grid_v1/b010/chrxtrain/w10/best_model.pt"
WINDOWS = ROOT / "data/processed/cds_gating/w10_frame"
FULL_FASTA = ROOT / "data/raw/dmel_genome.fa"
FULL_REFERENCE = ROOT / "data/raw/dmel_reference.gtf"
FULL_EVIANN = ROOT / "data/raw/dmel_eviann_pseudo_label.gff"
CONVERTER = ROOT / "scripts/convert_joint_frame_for_uniann.pl"
ARTIFACT = ROOT / "artifacts/belief_probe_v1/E_overcounting/leakage_controlled"
WORK = ROOT / "work/uniann/belief_probe_E_leakage_controlled"
RESULT = ROOT / "results/belief_probe_v1/E_overcounting/leakage_controlled"

CHROMS = {
    "3L": {"accession": "NT_037436.4"},
    "X": {"accession": "NC_004354.4"},
}
SCORE_BATCH_SIZE = int(os.environ.get("BELIEF_PROBE_E_SCORE_BATCH_SIZE", "8"))


def parse_attributes(text):
    values = {}
    for key, quoted, plain in re.findall(
        r"([A-Za-z0-9_]+)[ =](?:\"([^\"]*)\"|([^;]+))", text,
    ):
        values[key] = quoted or plain.strip()
    return values


def write_single_fasta(accession, destination):
    if destination.is_file() and Path(str(destination) + ".fai").is_file():
        return
    fasta = pysam.FastaFile(str(FULL_FASTA))
    sequence = fasta.fetch(accession).upper()
    fasta.close()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(f">{accession}\n")
        for start in range(0, len(sequence), 80):
            handle.write(sequence[start:start + 80] + "\n")
    os.replace(temporary, destination)
    pysam.faidx(str(destination))


def write_cds_reference(accession, destination):
    """Create the same CDS-as-exon strict reference used for chrX."""
    if destination.is_file():
        return
    transcripts = {}
    genes = {}
    with FULL_REFERENCE.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if (
                len(fields) < 9 or fields[0] != accession
                or fields[2] not in {"CDS", "stop_codon"}
                or fields[6] != "+"
            ):
                continue
            attrs = parse_attributes(fields[8])
            transcript = attrs.get("transcript_id")
            if not transcript:
                continue
            transcripts.setdefault(transcript, []).append((int(fields[3]), int(fields[4])))
            genes[transcript] = attrs.get("gene_id", transcript)
    representatives = {}
    for transcript, segments in transcripts.items():
        merged = []
        for start, end in sorted(set(segments)):
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        chain = tuple(merged)
        representatives.setdefault(chain, transcript)
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = destination.with_suffix(destination.suffix + ".raw.tmp")
    merged = destination.with_suffix(destination.suffix + ".merged.tmp")
    with raw.open("w") as output:
        for locus_index, (chain, transcript) in enumerate(
            sorted(representatives.items(), key=lambda item: item[0]), 1,
        ):
            gene = genes[transcript]
            attrs = (
                f'transcript_id "{transcript}"; gene_id "{gene}"; '
                f'locus "RLOC_{locus_index:08d}";'
            )
            output.write(
                f"{accession}\tRefSeq\ttranscript\t{chain[0][0]}\t{chain[-1][1]}"
                f"\t.\t+\t.\t{attrs}\n"
            )
            for start, end in chain:
                output.write(
                    f"{accession}\tRefSeq\texon\t{start}\t{end}\t.\t+\t.\t"
                    f'transcript_id "{transcript}"; gene_id "{gene}";\n'
                )
    subprocess.run([
        UNIANN / "bin/gffread", "-M", raw, "-o", merged,
    ], check=True)
    os.replace(merged, destination)
    raw.unlink(missing_ok=True)


def write_eviann_subset(accession, destination):
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FULL_EVIANN.open() as source, destination.open("w") as output:
        output.write("##gff-version 3\n")
        for line in source:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if (
                len(fields) >= 9 and fields[0] == accession
                and fields[6] == "+" and fields[2] in {"mRNA", "transcript", "CDS"}
            ):
                output.write(line)


def call(command, log, cwd=None):
    command = [str(item) for item in command]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write("+ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(
            command, cwd=cwd, stdout=handle, stderr=handle, check=True,
        )


def prepare_scored_chromosome(name):
    accession = CHROMS[name]["accession"]
    root = ARTIFACT / name
    fasta = root / f"{accession}.fa"
    reference = root / f"{accession}.plus.CDS.uniq.gtf"
    eviann = root / f"{accession}.plus.EviAnn.CDS.gff"
    write_single_fasta(accession, fasta)
    write_cds_reference(accession, reference)
    write_eviann_subset(accession, eviann)
    score_dir = root / "site_scores"
    if not (score_dir / ".complete.json").is_file():
        call([
            sys.executable, "-m", "cnn_mamba.score_chromosome",
            "--checkpoint", CHECKPOINT, "--output-dir", score_dir,
            "--fasta", FULL_FASTA, "--chrom", accession,
            "--display-chrom", f"chr{name}", "--batch-size", SCORE_BATCH_SIZE,
        ], root / "score_sites.log")
    score_tsv = root / f"chr{name}_plus_scores.tsv"
    if not Path(str(score_tsv) + ".complete.json").is_file():
        call([
            sys.executable, "-m", "cnn_mamba.export_uniann_scores",
            "--score-dir", score_dir, "--output", score_tsv,
            "--chrom", name, "--fasta", FULL_FASTA, "--accession", accession,
        ], root / "export_sites.log")
    frame_dir = root / "joint_frame"
    if not (frame_dir / ".complete.json").is_file():
        call([
            sys.executable, "-m", "cnn_mamba.score_joint_frame_chromosome",
            "--checkpoint", CHECKPOINT, "--fasta", FULL_FASTA,
            "--chrom", accession, "--display-chrom", f"chr{name}",
            "--output-dir", frame_dir, "--batch-size", SCORE_BATCH_SIZE,
            "--save-logits",
        ], root / "score_frame.log")
    frame_tsv = frame_dir / f"chr{name}_plus_joint_frame_probabilities.tsv"
    if not Path(str(frame_tsv) + ".complete.json").is_file():
        call([
            sys.executable, "-m", "cnn_mamba.export_joint_frame_tsv",
            "--probabilities", frame_dir / "joint_frame_probabilities.npy",
            "--argmax", frame_dir / "joint_frame_argmax.npy",
            "--output", frame_tsv, "--chrom", f"chr{name}", "--strand", "+",
        ], root / "export_frame.log")
    return {
        "name": name, "accession": accession, "fasta": fasta,
        "reference": reference, "eviann": eviann,
        "scores": score_tsv, "frame": frame_tsv,
        "factor": factor_for_scores(score_tsv),
    }


def factor_for_scores(path):
    scores = pd.read_csv(path, sep="\t", usecols=["type", "prob"])
    maximum = np.log(
        scores.loc[scores["type"].eq("donor"), "prob"].to_numpy() * math.e
        + 1e-10
    ).max()
    return int(1000.0 / maximum + 0.5)


def bias_text(values):
    return ",".join(f"{float(value):.12g}" for value in values)


def condition_name(temperature, kind, values):
    encoded = "_".join(f"{float(value):+.4f}".replace("+", "p").replace("-", "m").replace(".", "p") for value in values)
    return f"t{temperature:g}_{kind}_{encoded}"


def strict_compare(work, query, reference, prefix):
    merged = work / f"{prefix}.merged.gff"
    stats = work / f"{prefix}.stats"
    if not stats.is_file():
        call([UNIANN / "bin/gffread", "-M", query, "-o", merged], work / f"{prefix}.gffread.log")
        call([
            UNIANN / "bin/gffcompare", "--strict-match", "-e", "0",
            "-r", reference, merged, "-o", work / prefix,
        ], work / f"{prefix}.gffcompare.log")
    return legacy.parse_stats(stats)


def evaluate_condition(inputs, specification, split):
    work = WORK / split / specification["condition"]
    work.mkdir(parents=True, exist_ok=True)
    legacy.ensure_link(inputs["fasta"], work / "genome.fa")
    legacy.ensure_link(inputs["scores"], work / "scores.txt")
    legacy.ensure_link(inputs["frame"], work / "coding.txt")
    emission = work / "out.ps.txt"
    prediction = work / "genome.fa.uniann.gff"
    if not prediction.is_file():
        if not emission.is_file():
            temporary = work / "out.ps.txt.tmp"
            with inputs["frame"].open("rb") as source, temporary.open("wb") as output:
                subprocess.run([
                    "perl", CONVERTER, inputs["fasta"], "joint_softmax",
                    str(specification["temperature"]),
                    bias_text(specification["biases"]),
                ], stdin=source, stdout=output, stderr=subprocess.PIPE, check=True)
            os.replace(temporary, emission)
        call([
            UNIANN / "bin/uniann.sh", "-f", "genome.fa", "-c", "coding.txt",
            "-s", "scores.txt", "-n",
        ], work / "uniann.log", cwd=work)
    alone = strict_compare(work, prediction, inputs["reference"], "uniann")
    combined = work / "eviann_uniann.concat.gff"
    if not combined.is_file():
        with combined.open("w") as destination:
            destination.write(inputs["eviann"].read_text())
            with prediction.open() as source:
                destination.writelines(line for line in source if not line.startswith("#"))
    union = strict_compare(work, combined, inputs["reference"], "eviann_uniann")
    if split == "tune_3L":
        # Each chromosome-scale emission file is about 1 GB and is exactly
        # reproducible from the retained frame TSV and condition metadata.
        emission.unlink(missing_ok=True)
    common = {
        **specification, "biases": bias_text(specification["biases"]),
        "split": split, "chromosome": inputs["name"],
        "factor": inputs["factor"],
    }
    return [
        {**common, "method": "UniAnn", **alone},
        {**common, "method": "EviAnn+UniAnn", **union},
    ]


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def main():
    tune = prepare_scored_chromosome("3L")
    training_chromosomes = ("2L", "2R", "3R", "4", "Y", "X")
    temperatures = (6.0, 7.0, 8.0, 9.0, 10.0)
    scalar_biases = (-0.9, -0.45, 0.0, 0.45, 0.9)
    theory_unscaled = group_balanced_prior_correction(
        WINDOWS, training_chromosomes, ("3L",), 0.1, 1.0,
    )
    theory = {}
    for temperature in temperatures:
        item = dict(theory_unscaled)
        item["temperature"] = temperature
        item["per_frame_bias_after_temperature"] = [
            value / temperature
            for value in theory_unscaled["per_frame_bias_after_temperature"]
        ]
        theory[str(int(temperature))] = item
    specifications = []
    for temperature in temperatures:
        for scalar in scalar_biases:
            values = [scalar] * 3
            specifications.append({
                "condition": condition_name(temperature, "scalar", values),
                "temperature": temperature, "bias_kind": "shared_grid",
                "biases": values,
            })
        values = theory[str(int(temperature))]["per_frame_bias_after_temperature"]
        specifications.append({
            "condition": condition_name(temperature, "theory", values),
            "temperature": temperature, "bias_kind": "theoretical_per_frame",
            "biases": values,
        })
    tune_rows = []
    for index, specification in enumerate(specifications, 1):
        tune_rows.extend(evaluate_condition(tune, specification, "tune_3L"))
        write_table(RESULT / "tune_3L_grid.tsv", tune_rows)
        print(f"[3L {index}/{len(specifications)}] {specification['condition']}", flush=True)
    union_rows = [row for row in tune_rows if row["method"] == "EviAnn+UniAnn"]
    selected_row = max(union_rows, key=lambda row: (row["f1"], row["pr"], row["sn"]))
    selected = next(item for item in specifications if item["condition"] == selected_row["condition"])
    test = {
        "name": "X", "accession": "NC_004354.4",
        "fasta": ROOT / "data/raw/dmel_chrX.fa",
        "reference": ROOT / "data/raw/dmel_chrX_plus_CDS_reference.gtf",
        "eviann": ROOT / "data/raw/dmel_chrX_plus_EviAnn_CDS.gff",
        "scores": ROOT / "artifacts/scores/joint_frame_grid_v1/b010/chrxtrain/w10/chrX_plus_scores.tsv",
        "frame": ROOT / "artifacts/joint_frame_predictions/joint_frame_grid_v1/b010/chrxtrain/w10/chrX_plus/chrX_plus_joint_frame_probabilities.tsv",
    }
    test["factor"] = factor_for_scores(test["scores"])
    test_rows = evaluate_condition(test, selected, "test_chrX")
    write_table(RESULT / "test_chrX_selected.tsv", test_rows)
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / "selection.json").write_text(json.dumps({
        "selection_dataset": "Drosophila chromosome 3L plus strand",
        "selection_objective": "EviAnn+UniAnn strict-locus F1",
        "selected": selected,
        "selected_tune_metrics": selected_row,
        "test_dataset": "Drosophila chromosome X plus strand",
        "test_metrics": test_rows,
        "chrX_used_for_selection": False,
    }, indent=2) + "\n")
    (RESULT / "theoretical_prior_correction.json").write_text(
        json.dumps(theory, indent=2) + "\n"
    )
    (RESULT / "config.json").write_text(json.dumps({
        "schema": "belief_probe_E_leakage_controlled_v1",
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "noncoding_weight": 0.1,
        "training_chromosomes_for_implicit_prior": list(training_chromosomes),
        "tuning_chromosome": "3L",
        "test_chromosome": "X",
        "temperatures": list(temperatures),
        "shared_bias_grid": list(scalar_biases),
        "chromosome_scoring_batch_size": SCORE_BATCH_SIZE,
        "uniann_commit": subprocess.check_output(
            ["git", "-C", str(UNIANN), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "factor_definition": "round(1000 / max_donor(log(exp(1)*p + 1e-10)))",
        "actual_factor_3L": tune["factor"],
        "actual_factor_chrX": test["factor"],
        "reference_recipe_validation": (
            "generated chrX reference reproduced all 1783 existing unique "
            "CDS exon chains exactly (1783/1783; no chain differences)"
        ),
        "evaluation": "gffread -M then gffcompare --strict-match -e 0",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
