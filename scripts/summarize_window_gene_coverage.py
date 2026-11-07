#!/usr/bin/env python3
"""Summarize complete EviAnn gene/transcript containment by window size."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


def open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def parse_attrs(text: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in text.split(";") if "=" in item)


def parse_features(path: Path, accessions: set[str]):
    features = {"gene": [], "transcript": []}
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 9 or fields[0] not in accessions or fields[6] not in {"+", "-"}:
                continue
            feature = fields[2]
            if feature == "gene":
                kind = "gene"
            elif feature in {"mRNA", "transcript"}:
                kind = "transcript"
            else:
                continue
            attrs = parse_attrs(fields[8])
            identifier = attrs.get("ID") or attrs.get("transcript_id") or (
                f"{fields[0]}:{fields[3]}-{fields[4]}:{fields[6]}"
            )
            features[kind].append(
                (identifier, fields[0], fields[6], int(fields[3]) - 1, int(fields[4]))
            )
    return features


def load_windows(cell_dir: Path, accessions: dict[str, str]):
    windows = defaultdict(list)
    for split in ("train", "val", "test"):
        for path in (cell_dir / split).glob("*.npz"):
            stem = path.stem
            chrom, orientation, start_text = stem.rsplit("_", 2)
            strand = "+" if orientation == "plus" else "-"
            windows[(split, accessions[chrom], strand)].append(int(start_text))
    for starts in windows.values():
        starts.sort()
    return windows


def summarize_group(windows, features, window_size: int, splits: set[str]):
    selected = {
        key: starts for key, starts in windows.items() if key[0] in splits
    }
    total_windows = sum(len(starts) for starts in selected.values())
    contained_windows = {
        key: bytearray(len(starts)) for key, starts in selected.items()
    }
    captured_features = set()
    containment_pairs = 0
    eligible_features = 0

    for identifier, chrom, strand, feature_start, feature_end in features:
        matching_keys = [
            key for key in selected if key[1] == chrom and key[2] == strand
        ]
        if not matching_keys:
            continue
        eligible_features += 1
        captured = False
        for key in matching_keys:
            starts = selected[key]
            left = bisect.bisect_left(starts, feature_end - window_size)
            right = bisect.bisect_right(starts, feature_start)
            if right <= left:
                continue
            captured = True
            containment_pairs += right - left
            marks = contained_windows[key]
            for index in range(left, right):
                marks[index] = 1
        if captured:
            captured_features.add((identifier, chrom, strand))

    windows_with_complete = sum(sum(marks) for marks in contained_windows.values())
    return {
        "total_windows": total_windows,
        "windows_with_complete": windows_with_complete,
        "window_fraction": windows_with_complete / total_windows if total_windows else 0.0,
        "containment_pairs": containment_pairs,
        "eligible_features": eligible_features,
        "captured_features": len(captured_features),
        "feature_fraction": len(captured_features) / eligible_features if eligible_features else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cells = ["w0p5", "w1", "w2", "w5"]
    first_manifest = json.loads(
        (args.processed_root / cells[0] / "split_manifest.json").read_text()
    )
    accessions = first_manifest["accessions"]
    annotation = Path(first_manifest["sources"]["eviann"]["path"])
    features = parse_features(annotation, set(accessions.values()))
    groups = {
        "effective_train_chrX_included": {"train", "test"},
        "validation": {"val"},
        "all_unique": {"train", "val", "test"},
    }

    rows = []
    for cell in cells:
        cell_dir = args.processed_root / cell
        manifest = json.loads((cell_dir / "split_manifest.json").read_text())
        window_size = int(manifest["window_size"])
        windows = load_windows(cell_dir, accessions)
        for group, splits in groups.items():
            for feature_kind in ("gene", "transcript"):
                stats = summarize_group(
                    windows, features[feature_kind], window_size, splits
                )
                rows.append(
                    {
                        "cell": cell,
                        "window_bp": window_size,
                        "stride_bp": int(manifest["stride"]),
                        "group": group,
                        "feature": feature_kind,
                        **stats,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
