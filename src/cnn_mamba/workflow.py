"""Configuration-driven runner for reproducible multi-stage workflows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .commands import COMMAND_MODULES


VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand(value: str, variables: dict[str, str]) -> str:
    """Expand ${NAME} from workflow variables and then the environment."""
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in variables:
            return variables[name]
        if name in os.environ:
            return os.environ[name]
        raise ValueError(f"undefined workflow variable: {name}")
    return VARIABLE.sub(replace, value)


def load_workflow(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    with path.open() as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("workflow schema_version must be 1")
    steps = config.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("workflow must contain a non-empty steps array")

    raw_root = str(config.get("project_root", path.parent))
    preliminary = {"CONFIG_DIR": str(path.parent)}
    project_root = Path(expand(raw_root, preliminary)).expanduser()
    if not project_root.is_absolute():
        project_root = path.parent / project_root
    project_root = project_root.resolve()
    variables = {
        "CONFIG_DIR": str(path.parent),
        "PROJECT_ROOT": str(project_root),
    }
    for name, value in config.get("variables", {}).items():
        variables[str(name)] = expand(str(value), variables)

    normalized_steps = []
    names = set()
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            raise ValueError(f"step {index} is not an object")
        name = str(step.get("name", f"step_{index}"))
        command = str(step.get("command", ""))
        if name in names:
            raise ValueError(f"duplicate step name: {name}")
        if command not in COMMAND_MODULES:
            raise ValueError(f"step {name} has unsupported command: {command}")
        arguments = step.get("args", [])
        if not isinstance(arguments, list) or not all(isinstance(x, str) for x in arguments):
            raise ValueError(f"step {name} args must be an array of strings")
        names.add(name)
        normalized_steps.append({
            "name": name,
            "command": command,
            "module": COMMAND_MODULES[command],
            "args": [expand(item, variables) for item in arguments],
            "enabled": bool(step.get("enabled", True)),
        })
    return {
        "config_path": str(path),
        "project_root": str(project_root),
        "variables": variables,
        "steps": normalized_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a versioned CNN--Mamba JSON workflow."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--stage", action="append", help="run only this named stage; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()

    workflow = load_workflow(args.config)
    selected = set(args.stage or [])
    known = {step["name"] for step in workflow["steps"]}
    unknown = selected - known
    if unknown:
        parser.error(f"unknown stage(s): {', '.join(sorted(unknown))}")

    commands = []
    for step in workflow["steps"]:
        if not step["enabled"] or (selected and step["name"] not in selected):
            continue
        commands.append({
            **step,
            "argv": [sys.executable, "-m", step["module"], *step["args"]],
        })
    for row in commands:
        print(f"[{row['name']}] " + " ".join(row["argv"]), flush=True)
    if args.dry_run:
        return 0

    project_root = Path(workflow["project_root"])
    log_root = (args.log_dir or project_root / "workflow_runs").expanduser().resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = log_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    record = {**workflow, "run_id": run_id, "commands": [], "status": "running"}
    started = time.time()
    try:
        for row in commands:
            step_started = time.time()
            log_path = run_dir / f"{row['name']}.log"
            with log_path.open("w") as log:
                completed = subprocess.run(
                    row["argv"], cwd=project_root, stdout=log,
                    stderr=subprocess.STDOUT, text=True,
                )
            record["commands"].append({
                "name": row["name"], "argv": row["argv"],
                "returncode": completed.returncode,
                "elapsed_seconds": time.time() - step_started,
                "log": str(log_path),
            })
            if completed.returncode:
                raise RuntimeError(
                    f"workflow step {row['name']} failed; see {log_path}"
                )
        record["status"] = "complete"
    except Exception:
        record["status"] = "failed"
        raise
    finally:
        record["elapsed_seconds"] = time.time() - started
        (run_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
