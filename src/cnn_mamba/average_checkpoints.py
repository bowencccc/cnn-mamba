"""Create an inference checkpoint by arithmetic averaging model weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arithmetic-average model_state_dict tensors from compatible checkpoints."
    )
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = [path.resolve() for path in args.checkpoints]
    if len(sources) < 2:
        parser.error("at least two checkpoints are required")
    if len(set(sources)) != len(sources):
        parser.error("checkpoint paths must be unique")
    for path in sources:
        if not path.is_file():
            parser.error(f"checkpoint does not exist: {path}")

    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in sources]
    states = [checkpoint["model_state_dict"] for checkpoint in checkpoints]
    keys = list(states[0])
    for path, state in zip(sources[1:], states[1:]):
        if list(state) != keys:
            raise ValueError(f"model_state_dict keys differ in {path}")

    averaged = {}
    for key in keys:
        tensors = [state[key] for state in states]
        reference = tensors[0]
        for path, tensor in zip(sources[1:], tensors[1:]):
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise ValueError(f"incompatible tensor {key!r} in {path}")
        if reference.is_floating_point() or reference.is_complex():
            accumulator_dtype = torch.complex128 if reference.is_complex() else torch.float64
            value = sum(tensor.to(accumulator_dtype) for tensor in tensors) / len(tensors)
            averaged[key] = value.to(reference.dtype)
        else:
            if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                raise ValueError(f"non-floating buffer {key!r} differs across checkpoints")
            averaged[key] = reference.clone()

    # Keep model construction and inference metadata from the latest source,
    # but deliberately omit optimizer/RNG state: an averaged checkpoint is an
    # inference artifact and must not be mistaken for a resumable training state.
    latest_index = max(
        range(len(checkpoints)), key=lambda index: int(checkpoints[index].get("epoch", -1))
    )
    latest = checkpoints[latest_index]
    output_checkpoint = {
        key: value
        for key, value in latest.items()
        if key not in {
            "model_state_dict",
            "optimizer_state_dict",
            "scaler_state_dict",
            "data_generator_state",
        }
    }
    output_checkpoint["model_state_dict"] = averaged
    provenance = {
        "method": "arithmetic_mean_model_state_dict",
        "n_checkpoints": len(sources),
        "source_epochs": [int(checkpoint.get("epoch", -1)) for checkpoint in checkpoints],
        "sources": [
            {"path": str(path), "sha256": sha256(path)} for path in sources
        ],
        "resumable_training_checkpoint": False,
    }
    output_checkpoint["checkpoint_averaging"] = provenance

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(output_checkpoint, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(json.dumps({**provenance, "output_sha256": sha256(output)}, indent=2) + "\n")
    print(output)
    print(sidecar)


if __name__ == "__main__":
    main()
