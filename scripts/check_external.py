#!/usr/bin/env python3
"""Check staged external inputs/checkpoints against the recorded manifest."""

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=("raw", "checkpoint", "all"), default="all")
    parser.add_argument("--fast", action="store_true", help="check existence and size but skip SHA256")
    args = parser.parse_args()
    failures = 0
    with (root / "external_manifest.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if args.group != "all" and row["group"] != args.group:
                continue
            path = root / row["logical_path"]
            if not path.is_file():
                print(f"MISSING  {row['logical_path']}")
                failures += 1
                continue
            size_ok = path.stat().st_size == int(row["bytes"])
            hash_ok = True if args.fast else sha256(path) == row["sha256"]
            status = "OK" if size_ok and hash_ok else "MISMATCH"
            print(f"{status:8s} {row['logical_path']}")
            failures += status != "OK"
    if failures:
        raise SystemExit(f"{failures} external artifact(s) missing or mismatched")


if __name__ == "__main__":
    main()
