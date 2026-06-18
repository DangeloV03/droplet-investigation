"""
Minimal Slurm submission helpers for json_runner.py.

Reads slurm_config.yml and submits one sbatch job per expanded sweep run.
"""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover - optional until --slurm is used
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None

DEFAULT_CONFIG = "slurm_config.yml"


def load_slurm_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    if yaml is None:
        raise ImportError(
            "PyYAML is required for Slurm submission. Install with: pip install pyyaml"
        ) from _YAML_IMPORT_ERROR

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Slurm config not found: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("partition", "cpu")
    cfg.setdefault("cpus_per_task", 1)
    cfg.setdefault("mem", "4G")
    cfg.setdefault("time", "24:00:00")
    cfg.setdefault("job_name", "droplet_sim")
    cfg.setdefault("report_dir", "/home/$USER/slurm_reports")
    cfg.setdefault("setup_cmds", [])
    cfg.setdefault("project_root", "")
    return cfg


def project_root(cfg: dict[str, Any]) -> Path:
    env_root = os.environ.get("PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    cfg_root = str(cfg.get("project_root", "")).strip()
    if cfg_root:
        return Path(cfg_root).expanduser().resolve()
    return Path.cwd().resolve()


def expand_user_vars(text: str) -> str:
    return os.path.expandvars(os.path.expanduser(text))


def build_batch_script(
    *,
    job_json: Path,
    cfg: dict[str, Any],
    root: Path,
) -> str:
    report_dir = expand_user_vars(str(cfg["report_dir"]))
    job_stem = job_json.stem
    stdout = f"{report_dir}/{job_stem}.out"
    stderr = f"{report_dir}/{job_stem}.err"

    lines = ["#!/bin/bash", "set -euo pipefail", ""]

    directives = {
        "job-name": f"{cfg['job_name']}_{job_stem}",
        "partition": cfg["partition"],
        "cpus-per-task": cfg["cpus_per_task"],
        "mem": cfg["mem"],
        "time": cfg["time"],
        "output": stdout,
        "error": stderr,
    }
    account = cfg.get("account")
    if account:
        directives["account"] = account

    for key, value in directives.items():
        lines.append(f"#SBATCH --{key}={value}")
    lines.append("")

    for cmd in cfg.get("setup_cmds", []):
        lines.append(str(cmd))
    lines.append("")

    rel_job = job_json.resolve().relative_to(root)
    lines.extend(
        [
            f"cd {shlex.quote(str(root))}",
            "export PROJECT_ROOT=" + shlex.quote(str(root)),
            (
                "python json_runner.py --run-job "
                + shlex.quote(str(rel_job))
            ),
            "",
        ]
    )
    return "\n".join(lines)


def submit_job(script_text: str, *, dry_run: bool = False) -> str | None:
    if dry_run:
        print(script_text)
        print("--- dry-run: sbatch not invoked ---")
        return None

    proc = subprocess.run(
        ["sbatch"],
        input=script_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {proc.returncode}):\n{proc.stderr.strip()}"
        )

    line = proc.stdout.strip()
    print(line)
    # Typical output: "Submitted batch job 12345678"
    parts = line.split()
    return parts[-1] if parts else None


def submit_runs(
    job_json_paths: list[Path],
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    dry_run: bool = False,
) -> list[str | None]:
    cfg = load_slurm_config(config_path)
    root = project_root(cfg)

    if not dry_run:
        report_dir = expand_user_vars(str(cfg["report_dir"]))
        os.makedirs(report_dir, exist_ok=True)

    job_ids: list[str | None] = []
    for job_json in job_json_paths:
        script = build_batch_script(job_json=job_json, cfg=cfg, root=root)
        job_ids.append(submit_job(script, dry_run=dry_run))
    return job_ids
