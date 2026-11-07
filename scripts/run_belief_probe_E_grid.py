#!/usr/bin/env python3
"""Experiment E: extend coding-bias around the empirical class-prior odds."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_uniann_frame_scaling_grid as base  # noqa: E402
from cnn_mamba.probe_overcounting import validation_coding_prior  # noqa: E402


WORK_ROOT = ROOT / "work/uniann/belief_probe_E_prior_grid"
RESULT_ROOT = ROOT / "results/belief_probe_v1/E_overcounting"
WINDOWS_ROOT = ROOT / "data/processed/cds_gating/w10_frame"


def condition_name(bias):
    sign = "m" if bias < 0 else "p"
    return f"joint_t9_bias_{sign}{abs(bias):.6f}".replace(".", "p")


def write_rows(rows):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    output = RESULT_ROOT / "extended_coding_bias_grid.tsv"
    temporary = output.with_suffix(".tsv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)


def cleanup(work):
    for name in (
        "out.ps.txt", "out.ps.txt.complete.json", "out.gt.txt", "out.ag.txt",
        "out.atg.txt", "out.stop.txt",
    ):
        path = work / name
        if path.is_file():
            path.unlink()


def main():
    for required in (
        base.FASTA, base.REFERENCE, base.EVIANN, base.SITE_SCORES,
        base.FRAME_TSV, base.CONVERTER,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    prior = validation_coding_prior(WINDOWS_ROOT, ("3L",))
    log_odds = prior["log_prior_noncoding_over_coding"]
    # Test both signs because the converter's positive bias boosts coding,
    # while natural-prior correction of class-balanced logits has the opposite
    # sign.  The exact magnitude requested by the brief is included.
    biases = sorted(set([
        -log_odds, -log_odds / 2, -log_odds / 4,
        0.0, 0.03,
        log_odds / 4, log_odds / 2, 3 * log_odds / 4, log_odds,
    ]))
    uniann_commit = subprocess.check_output(
        ["git", "-C", str(base.UNIANN), "rev-parse", "HEAD"], text=True,
    ).strip()
    rows = []
    for index, bias in enumerate(biases, 1):
        started = time.time()
        condition = condition_name(bias)
        work = WORK_ROOT / condition
        work.mkdir(parents=True, exist_ok=True)
        base.ensure_link(base.FASTA, work / "genome.fa")
        base.ensure_link(base.SITE_SCORES, work / "scores.txt")
        base.ensure_link(base.FRAME_TSV, work / "coding.txt")
        prediction = work / "genome.fa.uniann.gff"
        if not prediction.is_file():
            emission = work / "out.ps.txt"
            temporary = work / "out.ps.txt.tmp"
            if not emission.is_file():
                with base.FRAME_TSV.open("rb") as source, temporary.open("wb") as output:
                    base.log_run(
                        ["perl", base.CONVERTER, base.FASTA.resolve(),
                         "joint_softmax", 9.0, bias],
                        work, work / "emission_conversion.log",
                        stdin=source, stdout=output,
                    )
                os.replace(temporary, emission)
            base.log_run(
                [base.UNIANN / "bin/uniann.sh", "-f", "genome.fa", "-c",
                 "coding.txt", "-s", "scores.txt", "-n"],
                work, work / "uniann.log",
            )
        alone = base.parse_stats(base.strict_compare(work, prediction, "uniann"))
        combined = work / "eviann_uniann.concat.gff"
        if not combined.is_file():
            with combined.open("w") as destination:
                with base.EVIANN.open() as source:
                    destination.writelines(source)
                with prediction.open() as source:
                    destination.writelines(
                        line for line in source if not line.startswith("#")
                    )
        merged = base.parse_stats(base.strict_compare(work, combined, "eviann_uniann"))
        common = {
            "condition": condition, "temperature": 9.0,
            "coding_bias": bias,
            "prior_log_noncoding_over_coding": log_odds,
        }
        for method, metrics in (("UniAnn", alone), ("EviAnn+UniAnn", merged)):
            rows.append({**common, "method": method, **metrics})
        write_rows(rows)
        (work / "provenance.json").write_text(json.dumps({
            **common, "grid_index": index, "grid_size": len(biases),
            "prior": prior, "bias_semantics": "positive boosts all three coding states",
            "uniann_commit": uniann_commit,
            "frame_probabilities": str(base.FRAME_TSV.resolve()),
            "site_scores": str(base.SITE_SCORES.resolve()),
            "reference": str(base.REFERENCE.resolve()),
            "elapsed_seconds": time.time() - started,
            "metrics": {"UniAnn": alone, "EviAnn+UniAnn": merged},
        }, indent=2) + "\n")
        cleanup(work)
        print(
            f"[{index}/{len(biases)}] bias={bias:.6f} "
            f"combined Sn={merged['sn']:.1f} Pr={merged['pr']:.1f} "
            f"F1={merged['f1']:.4f}", flush=True,
        )
    (RESULT_ROOT / "coding_prior.json").write_text(json.dumps(prior, indent=2) + "\n")


if __name__ == "__main__":
    main()
