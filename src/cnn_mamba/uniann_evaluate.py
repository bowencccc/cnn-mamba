"""Run UniAnn and strict locus evaluation, optionally merged with EviAnn."""

import argparse
import json
from pathlib import Path
import re
import subprocess

import pandas as pd

from .uniann_compat import validate_header_compatible_joint_frame_tsv


def run(command, cwd, log):
    command = [str(x) for x in command]
    print("+ " + " ".join(command), flush=True)
    with Path(log).open("a") as handle:
        handle.write("+ " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def link(source, target):
    source, target = Path(source).resolve(), Path(target)
    if target.is_symlink() and target.resolve() == source:
        return
    if target.exists() or target.is_symlink():
        raise RuntimeError(f"refusing to replace existing input: {target}")
    target.symlink_to(source)


def parse_stats(path):
    text = Path(path).read_text()
    pair = re.search(r"Locus level:\s*([0-9.]+)\s*\|\s*([0-9.]+)", text)
    query = re.search(r"Query mRNAs\s*:\s*\d+ in\s*(\d+) loci", text)
    matched = re.search(r"Matching loci:\s*(\d+)", text)
    missed = re.search(r"Missed loci:\s*(\d+)/", text)
    novel = re.search(r"Novel loci:\s*(\d+)/", text)
    if not all((pair, query, matched, missed, novel)):
        raise RuntimeError(f"could not parse locus metrics from {path}")
    sn, pr = map(float, pair.groups())
    return {"sn": sn, "pr": pr, "f1": 2 * sn * pr / (sn + pr),
            "predicted_loci": int(query.group(1)), "matched_loci": int(matched.group(1)),
            "missed_loci": int(missed.group(1)), "novel_loci": int(novel.group(1))}


def strict_compare(gffread, gffcompare, reference, query, prefix, cwd):
    merged = Path(cwd) / f"{prefix}.merged.gff"
    run([gffread, "-M", query, "-o", merged], cwd, Path(cwd) / f"{prefix}.gffread.log")
    output = Path(cwd) / prefix
    run([gffcompare, "--strict-match", "-e", "0", "-r", reference, merged,
         "-o", output], cwd, Path(cwd) / f"{prefix}.gffcompare.log")
    return Path(str(output) + ".stats")


def normalize_gtf(gffread, source, destination, cwd):
    """Normalize GFF3 or GTF independently before combining annotations."""
    run([gffread, "-T", source, "-o", destination], cwd,
        Path(cwd) / f"{Path(destination).stem}.normalize.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniann-root", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument(
        "--coding-frame", "--psauron", dest="coding_frame", type=Path,
        required=True,
        help=(
            "Per-base coding-frame input. --psauron is retained only as a "
            "deprecated command-line alias."
        ),
    )
    parser.add_argument(
        "--precomputed-emissions", type=Path,
        help=(
            "Validated out.ps.txt emissions generated from the coding-frame "
            "input; skips UniAnn's built-in converter."
        ),
    )
    parser.add_argument(
        "--require-cnn-mamba-tsv-header", action="store_true",
        help=(
            "Validate and preserve the CNN-Mamba four-state TSV header before "
            "passing the file directly to UniAnn. Do not combine with "
            "--precomputed-emissions."
        ),
    )
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--reference-gtf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eviann-gff", type=Path,
                        help="plus-strand EviAnn GFF to evaluate EviAnn+UniAnn")
    args = parser.parse_args()
    if args.require_cnn_mamba_tsv_header and args.precomputed_emissions:
        parser.error(
            "--require-cnn-mamba-tsv-header is for direct TSV conversion and "
            "cannot be combined with --precomputed-emissions"
        )
    coding_frame_contract = None
    if args.require_cnn_mamba_tsv_header:
        try:
            coding_frame_contract = validate_header_compatible_joint_frame_tsv(
                args.coding_frame
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
    required = [args.fasta, args.coding_frame, args.scores, args.reference_gtf]
    if args.precomputed_emissions:
        required.append(args.precomputed_emissions)
    if args.eviann_gff:
        required.append(args.eviann_gff)
    for path in required:
        if not path.is_file():
            parser.error(f"required input does not exist: {path}")

    bindir = args.uniann_root.resolve() / "bin"
    uniann, gffread, gffcompare = bindir / "uniann.sh", bindir / "gffread", bindir / "gffcompare"
    for path in (uniann, gffread, gffcompare):
        if not path.is_file():
            parser.error(f"missing UniAnn executable: {path}")

    work = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    local_fasta = work / "genome.fa"
    local_coding = work / "coding.txt"
    local_scores = work / "scores.txt"
    link(args.fasta, local_fasta)
    link(args.coding_frame, local_coding)
    link(args.scores, local_scores)
    if args.precomputed_emissions:
        link(args.precomputed_emissions, work / "out.ps.txt")
    raw = Path(str(local_fasta) + ".uniann.gff")
    # Current UniAnn uses -c/--coding for the coding-frame probability file.
    run([uniann, "-f", local_fasta.name, "-c", local_coding.name,
         "-s", local_scores.name, "-n"],
        work, work / "uniann.log")
    alone_stats = strict_compare(gffread, gffcompare, args.reference_gtf.resolve(), raw,
                                 "uniann", work)
    rows = [{"method": "UniAnn", **parse_stats(alone_stats)}]

    if args.eviann_gff:
        # Do not concatenate GTF and GFF3 directly. gffread selects one parser
        # from the first records and can silently ignore the other syntax.
        eviann_gtf = work / "eviann.normalized.gtf"
        uniann_gtf = work / "uniann.normalized.gtf"
        normalize_gtf(gffread, args.eviann_gff.resolve(), eviann_gtf, work)
        normalize_gtf(gffread, raw, uniann_gtf, work)
        combined = work / "eviann_uniann.concat.gtf"
        with combined.open("w") as output:
            for normalized in (eviann_gtf, uniann_gtf):
                with normalized.open() as source:
                    output.writelines(
                        line for line in source if not line.startswith("#")
                    )
        stats = strict_compare(gffread, gffcompare, args.reference_gtf.resolve(), combined,
                               "eviann_uniann", work)
        rows.append({"method": "EviAnn+UniAnn", **parse_stats(stats)})

    frame = pd.DataFrame(rows)
    frame.to_csv(work / "locus_metrics.tsv", sep="\t", index=False, float_format="%.9g")
    inputs = {"fasta": args.fasta, "coding_frame": args.coding_frame,
              "scores": args.scores, "reference_gtf": args.reference_gtf}
    if args.precomputed_emissions:
        inputs["precomputed_emissions"] = args.precomputed_emissions
    if args.eviann_gff:
        inputs["eviann_gff"] = args.eviann_gff
    provenance = {"uniann_root": str(args.uniann_root.resolve()),
                  "union_normalization": "each source normalized to GTF before concatenation",
                  "inputs": {key: str(value.resolve()) for key, value in inputs.items()},
                  "metrics": rows}
    if coding_frame_contract is not None:
        provenance["coding_frame_input_contract"] = coding_frame_contract
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
