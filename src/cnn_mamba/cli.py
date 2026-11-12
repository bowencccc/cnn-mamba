"""Unified, stable command-line entry point for the public pipeline."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys

from . import __version__
from .commands import COMMAND_MODULES


def _tool(path: Path) -> dict[str, object]:
    return {"path": str(path), "available": path.is_file() and os.access(path, os.X_OK)}


def doctor_report(uniann_root: Path | None = None) -> dict[str, object]:
    """Return a machine-readable installation report without mutating state."""
    packages = {}
    for name in ("numpy", "pandas", "pysam", "scikit-learn", "torch", "mamba-ssm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    cuda = {"available": False, "device_count": 0, "devices": []}
    try:
        import torch
        cuda["available"] = bool(torch.cuda.is_available())
        cuda["device_count"] = int(torch.cuda.device_count())
        cuda["devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except Exception as error:  # Doctor should report broken installs, not crash.
        cuda["error"] = f"{type(error).__name__}: {error}"

    root_value = uniann_root or (
        Path(os.environ["UNIANN_ROOT"]) if "UNIANN_ROOT" in os.environ else None
    )
    uniann = None
    if root_value is not None:
        root_value = root_value.expanduser().resolve()
        uniann = {
            "root": str(root_value),
            "uniann": _tool(root_value / "bin" / "uniann.sh"),
            "gffread": _tool(root_value / "bin" / "gffread"),
            "gffcompare": _tool(root_value / "bin" / "gffcompare"),
        }

    return {
        "cnn_mamba": __version__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": packages,
        "cuda": cuda,
        "external_commands": {
            name: shutil.which(name) for name in ("git", "perl")
        },
        "uniann": uniann,
    }


def _print_doctor(report: dict[str, object]) -> None:
    print(f"CNN-Mamba {report['cnn_mamba']}")
    print(f"Python {report['python']} ({report['executable']})")
    for name, version in report["packages"].items():
        print(f"{name:16s} {version or 'MISSING'}")
    cuda = report["cuda"]
    print(f"CUDA             {'available' if cuda['available'] else 'not available'}")
    for index, device in enumerate(cuda.get("devices", [])):
        print(f"  GPU {index}: {device}")
    if report["uniann"] is None:
        print("UniAnn           not checked (set UNIANN_ROOT or pass --uniann-root)")
    else:
        for name in ("uniann", "gffread", "gffcompare"):
            row = report["uniann"][name]
            print(f"{name:16s} {'OK' if row['available'] else 'MISSING'}  {row['path']}")


def _delegate(command: str, arguments: list[str]) -> int:
    module = importlib.import_module(COMMAND_MODULES[command])
    previous = sys.argv
    try:
        sys.argv = [f"cnn-mamba {command}", *arguments]
        module.main()
    finally:
        sys.argv = previous
    return 0


def main() -> int:
    commands = ", ".join(COMMAND_MODULES)
    parser = argparse.ArgumentParser(
        prog="cnn-mamba",
        description="CNN--Mamba genomic site prediction and annotation pipeline.",
        epilog=f"Pipeline commands: {commands}. Use 'cnn-mamba COMMAND --help'.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("command", nargs="?", help="pipeline command, workflow, or doctor")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0
    if args.command in COMMAND_MODULES:
        return _delegate(args.command, args.arguments)
    if args.command == "workflow":
        from .workflow import main as workflow_main
        previous = sys.argv
        try:
            sys.argv = ["cnn-mamba workflow", *args.arguments]
            return workflow_main()
        finally:
            sys.argv = previous
    if args.command == "doctor":
        doctor_parser = argparse.ArgumentParser(prog="cnn-mamba doctor")
        doctor_parser.add_argument("--uniann-root", type=Path)
        doctor_parser.add_argument("--json", action="store_true")
        doctor_args = doctor_parser.parse_args(args.arguments)
        report = doctor_report(doctor_args.uniann_root)
        if doctor_args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_doctor(report)
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
