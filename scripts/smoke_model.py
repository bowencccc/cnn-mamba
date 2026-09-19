#!/usr/bin/env python3
"""Verify Mamba3 provenance and exercise its CUDA forward/backward when available."""

import importlib.metadata as metadata
import inspect
import json
from pathlib import Path

import torch
from mamba_ssm import Mamba3
from cnn_mamba.model import SpliceMamba


EXPECTED_COMMIT = "be0303b971cd79a4fbbcfc411a01b33f6b0e3602"


def main():
    distribution = metadata.distribution("mamba-ssm")
    direct_url = Path(distribution._path) / "direct_url.json"
    provenance = json.loads(direct_url.read_text()) if direct_url.is_file() else {}
    installed_commit = provenance.get("vcs_info", {}).get("commit_id")
    print(f"Mamba3 source: {inspect.getsourcefile(Mamba3)}")
    print(f"mamba-ssm metadata version: {distribution.version}")
    print(f"mamba git commit: {installed_commit or 'not recorded'}")
    if installed_commit and installed_commit != EXPECTED_COMMIT:
        raise RuntimeError(
            f"wrong mamba commit: {installed_commit}; expected {EXPECTED_COMMIT}"
        )

    model = SpliceMamba(d_model=16, d_state=8, n_layers=0, dropout=0.0,
                        architecture="cnn_mamba", cnn_kernel_size=7)
    splice, start_stop = model(torch.randint(0, 5, (2, 101)))
    assert splice.shape == start_stop.shape == (2, 101, 3)
    print("CPU wrapper shape test passed")

    if not torch.cuda.is_available():
        print("CUDA unavailable; Mamba3 kernel test skipped")
        return
    device = torch.device("cuda")
    layer = Mamba3(d_model=64, d_state=16, headdim=64).to(device)
    x = torch.randn(2, 256, 64, device=device, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"Mamba3 CUDA forward/backward passed on {torch.cuda.get_device_name(0)}")

    frame_model = SpliceMamba(
        d_model=64, d_state=16, n_layers=1, dropout=0.1,
        architecture="cnn_mamba", cnn_kernel_size=7,
        frame_dilation=3, phase_auxiliary=True,
    ).to(device)
    sequence = torch.randint(0, 5, (2, 256), device=device)
    outputs = frame_model(sequence, return_phase=True)
    assert all(output.shape == (2, 256, 3) for output in outputs)
    sum(output.square().mean() for output in outputs).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in frame_model.parameters()
    )
    print("Frame-branch plus phase-head CUDA forward/backward passed")


if __name__ == "__main__":
    main()
