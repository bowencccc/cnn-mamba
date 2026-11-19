#!/usr/bin/env python3
"""Summarize validation trajectories from the weight-decay sweep."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for result in sorted(args.run_root.glob("wd*/seed*/chrxtrain/w1/results.json")):
        run_dir = result.parent
        payload = json.loads(result.read_text())
        best_checkpoint = run_dir / "best_model.pt"
        last_checkpoint = run_dir / "last_model.pt"
        import torch
        best_saved = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
        last_saved = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        config = best_saved["config"]
        history = last_saved["history"]
        best = max(
            (entry for entry in history if entry.get("validation_reference") is not None),
            key=lambda entry: sum(
                entry["validation_reference"][task]["AP"]
                for task in ("donor", "acceptor", "start", "stop")
            ) / 4,
        )
        metrics = best["validation_reference"]
        rows.append({
            "weight_decay": float(config["weight_decay"]),
            "seed": int(config["seed"]),
            "best_epoch": int(best["epoch"]),
            "stopped_epoch": int(last_saved["epoch"]),
            "val_mean_AP": sum(metrics[t]["AP"] for t in ("donor", "acceptor", "start", "stop")) / 4,
            **{f"val_{t}_AP": metrics[t]["AP"] for t in ("donor", "acceptor", "start", "stop")},
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["weight_decay", "seed", "best_epoch", "stopped_epoch", "val_mean_AP",
              "val_donor_AP", "val_acceptor_AP", "val_start_AP", "val_stop_AP"]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
