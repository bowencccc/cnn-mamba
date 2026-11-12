"""Materialize stable links to the promoted inputs for downstream decoders.

The experiment registry is deliberately explicit.  A checkpoint or chromosome
output is never discovered by newest modification time: it must first be
promoted in the versioned JSON configuration.  This prevents a partially
written run, or a result made with the wrong EviAnn evidence source, from being
silently consumed by Viterbi/CRF experiments.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


SCHEMA = "cnn_mamba_viterbi_current_best_v1"
EXPECTED_EVIDENCE = {"human": "rodentia", "dmel": "coleoptera"}


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path) -> tuple[dict[str, object], Path]:
    path = path.expanduser().resolve()
    config = json.loads(path.read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError(f"unsupported registry schema: {config.get('schema')!r}")
    project_root = Path(config.get("project_root", "."))
    if not project_root.is_absolute():
        project_root = path.parent / project_root
    project_root = project_root.resolve()
    species = config.get("species")
    if not isinstance(species, dict) or set(species) != set(EXPECTED_EVIDENCE):
        raise ValueError("registry must define exactly the human and dmel entries")
    for name, evidence in EXPECTED_EVIDENCE.items():
        entry = species[name]
        if not isinstance(entry, dict) or str(entry.get("evidence_source", "")).lower() != evidence:
            raise ValueError(f"{name} evidence_source must be {evidence!r}")
        if not isinstance(entry.get("assets"), dict):
            raise ValueError(f"{name}.assets must be an object")
    return config, project_root


def _source(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not target.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink path: {target}")
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source, target_is_directory=source.is_dir())
    os.replace(temporary, target)


def materialize(
    config: dict[str, object], project_root: Path, output_root: Path,
    selected_species: set[str] | None = None,
) -> dict[str, object]:
    output_root = output_root.expanduser()
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root = output_root.resolve()
    selected_species = selected_species or set(EXPECTED_EVIDENCE)
    records: dict[str, object] = {}
    failures: list[str] = []

    for name, entry in config["species"].items():
        if name not in selected_species:
            continue
        destination = output_root / name
        assets = {}
        missing_required = []
        missing_optional = []
        for asset_name, specification in entry["assets"].items():
            if not isinstance(specification, dict):
                raise ValueError(f"{name}.{asset_name} must be an object")
            source = _source(project_root, str(specification["source"]))
            required = bool(specification.get("required", True))
            link_name = str(specification.get("link", asset_name))
            target = destination / link_name
            exists = source.exists()
            record = {
                "source": str(source),
                "link": str(target),
                "required": required,
                "available": exists,
            }
            if exists:
                _atomic_symlink(source, target)
                if source.is_file():
                    record["bytes"] = source.stat().st_size
                    if specification.get("sha256", False):
                        record["sha256"] = sha256_file(source)
            else:
                if target.is_symlink():
                    target.unlink()
                (missing_required if required else missing_optional).append(asset_name)
            assets[asset_name] = record

        model_ready = not missing_required
        prediction_assets = entry.get("prediction_assets", [])
        prediction_ready = model_ready and all(
            assets.get(asset, {}).get("available", False) for asset in prediction_assets
        )
        if missing_required:
            failures.append(f"{name}: {', '.join(missing_required)}")
        records[name] = {
            "evidence_source": entry["evidence_source"],
            "selection": entry.get("selection", {}),
            "model_ready": model_ready,
            "prediction_ready": prediction_ready,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "assets": assets,
        }

    manifest = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry": str(config.get("registry_path", "configs/viterbi_current_best.json")),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "species": records,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    if failures:
        raise FileNotFoundError("missing required current-best assets: " + "; ".join(failures))
    return manifest


def _print_summary(manifest: dict[str, object]) -> None:
    for name, entry in manifest["species"].items():
        selection = entry.get("selection", {})
        epoch = selection.get("checkpoint_epoch", "?")
        metric = selection.get("validation_mean_ap", "?")
        print(
            f"{name:5s} evidence={entry['evidence_source']:10s} "
            f"epoch={epoch} validation_mean_AP={metric} "
            f"model_ready={entry['model_ready']} "
            f"prediction_ready={entry['prediction_ready']}"
        )
        if entry["missing_optional"]:
            print("      missing optional: " + ", ".join(entry["missing_optional"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stable current-best links for downstream Viterbi code."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/viterbi_current_best.json")
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("artifacts/current_best/viterbi"),
    )
    parser.add_argument("--species", action="append", choices=sorted(EXPECTED_EVIDENCE))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    config, project_root = load_registry(args.config)
    config["registry_path"] = str(args.config)
    manifest = materialize(
        config, project_root, args.output_root,
        set(args.species) if args.species else None,
    )
    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        _print_summary(manifest)


if __name__ == "__main__":
    main()
