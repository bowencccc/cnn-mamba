#!/usr/bin/env python3
"""Extract transcript/CDS records for one chromosome and strand from EviAnn."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path


def open_input(path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--strand", choices=("+", "-"), required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with open_input(args.input) as source, args.output.open("w") as destination:
        destination.write("##gff-version 3\n")
        for line in source:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if (
                len(fields) >= 9 and fields[0] == args.chrom
                and fields[6] == args.strand and fields[2] in {"mRNA", "CDS"}
            ):
                destination.write(line)
                records += 1
    print(f"wrote {records:,} mRNA/CDS records to {args.output}")


if __name__ == "__main__":
    main()
