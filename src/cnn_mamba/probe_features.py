"""Frozen CNN--Mamba backbone loading and residual-stream feature hooks.

The MLM checkpoint already contains the five-token ``backbone_state_dict``
produced by :meth:`CNNMambaMLM.transfer_state_dict`; that method removes the
MASK-token embedding before saving.  This module consumes that exact transfer
artifact and deliberately never loads or evaluates any task head or the
optional dilation-3 frame branch.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
from pathlib import Path

import torch

from .common import initialize_like_original
from .model import SpliceMamba, model_from_checkpoint


FEATURE_NAMES = ("input",) + tuple(f"layer{i}" for i in range(1, 9)) + ("final_norm",)
# The supervised checkpoint's layer-8 residual can exceed 65,504.  A fixed,
# positive per-layer scale is an invertible linear reparameterization, so it
# preserves exactly which labels a linear probe can decode while permitting
# the required float16 storage.
FP16_STORAGE_SCALES = (1.0,) * 8 + (1.0 / 32.0, 1.0)
BACKBONE_PREFIXES = ("embedding.", "input_proj.", "layers.", "final_norm.")


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def new_backbone() -> SpliceMamba:
    model = SpliceMamba(
        d_model=192, d_state=64, n_layers=8, dropout=0.1,
        architecture="cnn_mamba", cnn_kernel_size=7,
    )
    return model


def load_probe_backbone(kind: str, checkpoint: Path | None = None) -> SpliceMamba:
    """Load an MLM, random, or supervised backbone for frozen probing."""
    kind = kind.lower()
    if kind == "random":
        model = new_backbone()
        initialize_like_original(model)
    elif kind == "mlm":
        if checkpoint is None:
            raise ValueError("MLM backbone requires a checkpoint")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "backbone_state_dict" not in saved:
            raise KeyError(f"{checkpoint} has no backbone_state_dict")
        model = new_backbone()
        initialize_like_original(model)
        state = saved["backbone_state_dict"]
        expected = {
            name for name in model.state_dict()
            if name.startswith(BACKBONE_PREFIXES)
        }
        if set(state) != expected:
            raise RuntimeError(
                "MLM backbone key mismatch: "
                f"missing={sorted(expected - set(state))} "
                f"unexpected={sorted(set(state) - expected)}"
            )
        mismatch = {
            name: (tuple(state[name].shape), tuple(model.state_dict()[name].shape))
            for name in expected
            if state[name].shape != model.state_dict()[name].shape
        }
        if mismatch:
            raise RuntimeError(f"MLM backbone shape mismatch: {mismatch}")
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected MLM keys: {incompatible.unexpected_keys}")
    elif kind == "supervised":
        if checkpoint is None:
            raise ValueError("supervised backbone requires a checkpoint")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = model_from_checkpoint(saved)
    else:
        raise ValueError(f"unknown backbone kind: {kind}")

    model.eval()
    model.requires_grad_(False)
    return model


class BackboneFeatureHooks(AbstractContextManager):
    """Capture input projection, all eight blocks, and final norm outputs.

    Hook tensors retain shape ``(batch, length, 192)``.  ``concat_all`` is
    constructed on request by concatenating the ten captured representations.
    The model is frozen and evaluated through ``encode`` only.
    """

    def __init__(self, model: SpliceMamba):
        if len(model.layers) != 8 or model.architecture != "cnn_mamba":
            raise ValueError("belief probes require the 8-layer CNN--Mamba backbone")
        self.model = model
        self.outputs: dict[str, torch.Tensor] = {}
        self.handles = []
        self.handles.append(model.input_proj.register_forward_hook(self._hook("input")))
        for index, layer in enumerate(model.layers, 1):
            self.handles.append(layer.register_forward_hook(self._hook(f"layer{index}")))
        self.handles.append(model.final_norm.register_forward_hook(self._hook("final_norm")))

    def _hook(self, name):
        def capture(_module, _inputs, output):
            self.outputs[name] = output.detach()
        return capture

    @torch.inference_mode()
    def collect(self, sequence: torch.Tensor, include_concat: bool = False):
        self.outputs.clear()
        self.model.encode(sequence)
        missing = set(FEATURE_NAMES) - set(self.outputs)
        if missing:
            raise RuntimeError(f"feature hooks did not fire: {sorted(missing)}")
        result = {name: self.outputs[name] for name in FEATURE_NAMES}
        if include_concat:
            result["concat_all"] = torch.cat(
                [result[name] for name in FEATURE_NAMES], dim=-1
            )
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def backbone_parameter_snapshot(model: SpliceMamba):
    """Small exact snapshot used to assert that probe training stays frozen."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def assert_backbone_unchanged(model: SpliceMamba, snapshot) -> None:
    current = model.state_dict()
    changed = [name for name, value in snapshot.items() if not torch.equal(value, current[name].cpu())]
    if changed:
        raise AssertionError(f"frozen backbone changed: {changed[:10]}")
