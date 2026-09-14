"""Run UniAnn and strict locus evaluation, optionally merged with EviAnn."""

import argparse
import json
from pathlib import Path
import re
import subprocess

import pandas as pd


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniann-root", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--psauron", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--reference-gtf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eviann-gff", type=Path,
                        help="plus-strand EviAnn GFF to evaluate EviAnn+UniAnn")
    args = parser.parse_args()
    required = [args.fasta, args.psauron, args.scores, args.reference_gtf]
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
    local_psauron = work / "psauron_score.csv"
    local_scores = work / "scores.txt"
    link(args.fasta, local_fasta)
    link(args.psauron, local_psauron)
    link(args.scores, local_scores)
    raw = Path(str(local_fasta) + ".uniann.gff")
    run([uniann, "-f", local_fasta.name, "-p", local_psauron.name, "-s", local_scores.name],
        work, work / "uniann.log")
    alone_stats = strict_compare(gffread, gffcompare, args.reference_gtf.resolve(), raw,
                                 "uniann", work)
    rows = [{"method": "UniAnn", **parse_stats(alone_stats)}]

    if args.eviann_gff:
        combined = work / "eviann_uniann.concat.gff"
        with combined.open("w") as output:
            with args.eviann_gff.open() as source:
                output.writelines(source)
            with raw.open() as source:
                output.writelines(line for line in source if not line.startswith("#"))
        stats = strict_compare(gffread, gffcompare, args.reference_gtf.resolve(), combined,
                               "eviann_uniann", work)
        rows.append({"method": "EviAnn+UniAnn", **parse_stats(stats)})

    frame = pd.DataFrame(rows)
    frame.to_csv(work / "locus_metrics.tsv", sep="\t", index=False, float_format="%.9g")
    inputs = {"fasta": args.fasta, "psauron": args.psauron,
              "scores": args.scores, "reference_gtf": args.reference_gtf}
    if args.eviann_gff:
        inputs["eviann_gff"] = args.eviann_gff
    provenance = {"uniann_root": str(args.uniann_root.resolve()),
                  "inputs": {key: str(value.resolve()) for key, value in inputs.items()},
                  "metrics": rows}
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
