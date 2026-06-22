"""
JSON batch runner for HeteroNVTDrivenChain.

Reads a master JSON config, expands sweep arrays into individual runs,
executes equilibration + production chunks for each, and writes a
timestamped subfolder per run with params, lattice snapshots, and density
time series.

Usage:
    python json_runner.py example_sweep.json
    python json_runner.py delta_f_sweep.json --jobs 4
    python json_runner.py delta_f_sweep.json --slurm
    python json_runner.py --run-job samples/256_negative_drive_df2p85_dm0_epsm2p95.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from cli import SWEEPABLE_KEYS, build_output_basename
from simulation import (
    RunParams,
    load_or_create_geometry,
    make_timestamped_run_dir,
    run_chunked_simulation,
)

RUN_PARAM_KEYS = frozenset(RunParams.__dataclass_fields__.keys())
METADATA_KEYS = frozenset({"beta", "eta"})
SAMPLES_DIR = Path("samples")


def load_master_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    if "fixed" not in cfg:
        raise ValueError("Master JSON must contain a 'fixed' object")
    if "sweep" not in cfg:
        cfg["sweep"] = {}
    return cfg


def validate_keys(fixed: dict[str, Any], sweep: dict[str, Any]) -> None:
    for key in fixed:
        if key not in RUN_PARAM_KEYS and key not in METADATA_KEYS:
            raise ValueError(f"Unknown fixed key {key!r}")
    for key in sweep:
        if key not in SWEEPABLE_KEYS:
            raise ValueError(f"Cannot sweep unknown key {key!r}")
        if not isinstance(sweep[key], list) or len(sweep[key]) == 0:
            raise ValueError(f"sweep.{key} must be a non-empty JSON array")


def expand_runs(fixed: dict[str, Any], sweep: dict[str, Any]) -> list[dict[str, Any]]:
    if not sweep:
        return [{**fixed}]

    keys = sorted(sweep.keys())
    value_lists = [sweep[k] for k in keys]

    runs: list[dict[str, Any]] = []
    for combo in product(*value_lists):
        run = {**fixed}
        for k, v in zip(keys, combo):
            run[k] = v
        runs.append(run)
    return runs


def run_config_to_params(run: dict[str, Any]) -> RunParams:
    return RunParams(**{k: v for k, v in run.items() if k in RUN_PARAM_KEYS})


def execute_single_run(
    run: dict[str, Any],
    initial_state: np.ndarray,
    sweep_keys: list[str],
) -> dict:
    params = run_config_to_params(run)
    output_dir = run.get("output_dir", params.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    label = build_output_basename(run, sweep_keys)
    run_dir = make_timestamped_run_dir(output_dir, label)

    sweep_meta = {k: run[k] for k in sweep_keys}
    result = run_chunked_simulation(
        params,
        initial_state,
        run_dir,
        label=label,
        extra_params={"sweep": sweep_meta},
    )
    result["basename"] = label
    return result


def _worker_run(
    run: dict[str, Any],
    geometry_path: str,
    sweep_keys: list[str],
) -> dict:
    initial_state = np.load(geometry_path).astype(np.uint32)
    return execute_single_run(run, initial_state, sweep_keys)


def write_job_json(run: dict[str, Any], sweep_keys: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    label = build_output_basename(run, sweep_keys)
    path = out_dir / f"{label}.json"
    payload = {
        "run": run,
        "sweep_keys": sweep_keys,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_job_json(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if "run" in payload:
        run = payload["run"]
        sweep_keys = payload.get("sweep_keys", [])
        return run, sweep_keys

    # Backward-compatible: flat run dict only.
    return payload, []


def run_single_job(path: str | Path) -> dict:
    run, sweep_keys = load_job_json(path)
    params = run_config_to_params(run)
    print(f"Job config: {path}")
    print(f"Label: {build_output_basename(run, sweep_keys)}")
    print(f"Loading geometry from {params.initial_npy} ...")
    initial_state = load_or_create_geometry(params)
    return execute_single_run(run, initial_state, sweep_keys)


def _print_result(i: int, n: int, run: dict[str, Any], sweep_keys: list[str], result: dict) -> None:
    sweep_desc = ", ".join(f"{k}={run[k]}" for k in sweep_keys)
    print(f"[{i}/{n}] {sweep_desc or 'fixed params'}")
    print(
        f"  -> {result['run_dir']}\n"
        f"     t={result['final_time']:.4f}, "
        f"N={result['n_before']}->{result['n_after']}"
    )


def run_from_master(
    path: str | Path,
    *,
    jobs: int = 1,
    keep_run_json: bool = False,
    samples_only: bool = False,
    slurm: bool = False,
    slurm_dry_run: bool = False,
    slurm_config: str = "slurm_config.yml",
    samples_dir: Path = SAMPLES_DIR,
) -> None:
    cfg = load_master_config(path)
    fixed = cfg["fixed"]
    sweep = cfg.get("sweep", {})

    validate_keys(fixed, sweep)
    runs = expand_runs(fixed, sweep)
    sweep_keys = sorted(sweep.keys())

    n = len(runs)
    print(f"Master config: {path}")
    print(f"Sweep axes: {sweep_keys or '(none)'}")
    for k in sweep_keys:
        print(f"  {k}: {len(sweep[k])} values")
    print(f"Total runs: {n}\n")

    job_paths = [write_job_json(run, sweep_keys, samples_dir) for run in runs]
    print(f"Wrote {len(job_paths)} job JSON files under {samples_dir}/")

    if samples_only:
        print(f"\n=== Done: {n} job JSON files ready under {samples_dir}/ ===")
        print("Submit with: ./scripts/submit_sweep.sh", path)
        return

    if slurm or slurm_dry_run:
        from slurm_submit import submit_runs

        if slurm_dry_run:
            print("\nDry run: printing batch scripts (no sbatch, no simulations) ...")
        else:
            print("\nSubmitting Slurm jobs ...")
        submit_runs(
            job_paths,
            config_path=slurm_config,
            dry_run=slurm_dry_run,
        )
        if slurm_dry_run:
            print(f"\n=== Dry run done: {n} batch scripts printed, nothing submitted ===")
        else:
            print(f"\n=== Submitted {n} Slurm jobs (see slurm_config.yml report_dir) ===")
        return

    base_params = run_config_to_params(runs[0])
    print(f"\nLoading geometry from {base_params.initial_npy} ...")
    initial_state = load_or_create_geometry(base_params)

    if jobs <= 1:
        for i, run in enumerate(runs, start=1):
            result = execute_single_run(run, initial_state, sweep_keys)
            _print_result(i, n, run, sweep_keys, result)
        print(f"\n=== Done: {n} timestamped run folders under {base_params.output_dir}/ ===")
        return

    jobs = min(jobs, n)
    print(f"Running with {jobs} parallel worker processes ...")

    geometry_path = tempfile.mkstemp(prefix="json_runner_geom_", suffix=".npy")[1]
    np.save(geometry_path, initial_state)

    try:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(_worker_run, run, geometry_path, sweep_keys): (i, run)
                for i, run in enumerate(runs, start=1)
            }
            for future in as_completed(futures):
                i, run = futures[future]
                result = future.result()
                _print_result(i, n, run, sweep_keys, result)
    finally:
        os.remove(geometry_path)

    if not keep_run_json:
        for job_path in job_paths:
            job_path.unlink(missing_ok=True)
        if samples_dir.exists() and not any(samples_dir.iterdir()):
            samples_dir.rmdir()

    print(f"\n=== Done: {n} timestamped run folders under {base_params.output_dir}/ ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand a master JSON sweep config into chunked lattice-gas runs",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="example_sweep.json",
        help="Path to master JSON config (default: example_sweep.json)",
    )
    parser.add_argument(
        "--run-job",
        metavar="JOB_JSON",
        help="Run a single expanded job JSON (used by Slurm workers)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help="Run up to N simulations in parallel locally (default: 1)",
    )
    parser.add_argument(
        "--write-samples-only",
        action="store_true",
        help="Only expand the sweep into samples/*.json (no simulation, no Slurm)",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="(legacy) Submit via Python slurm_submit.py — prefer ./scripts/submit_sweep.sh",
    )
    parser.add_argument(
        "--slurm-dry-run",
        action="store_true",
        help="Print batch scripts without sbatch or running simulations",
    )
    parser.add_argument(
        "--slurm-config",
        default="slurm_config.yml",
        help="Slurm settings file (default: slurm_config.yml)",
    )
    parser.add_argument(
        "--samples-dir",
        default=str(SAMPLES_DIR),
        help="Directory for per-run job JSON files (default: samples/)",
    )
    parser.add_argument(
        "--keep-run-json",
        action="store_true",
        help="Keep per-run job JSON files after a local parallel batch",
    )
    args = parser.parse_args()

    if args.run_job:
        if not os.path.exists(args.run_job):
            print(f"Job JSON not found: {args.run_job}", file=sys.stderr)
            sys.exit(1)
        result = run_single_job(args.run_job)
        print(
            f"\n=== Done ===\n"
            f"{result['run_dir']}\n"
            f"t={result['final_time']:.4f}, "
            f"N={result['n_before']}->{result['n_after']}"
        )
        return

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if args.jobs < 1:
        print("--jobs must be >= 1", file=sys.stderr)
        sys.exit(1)

    run_from_master(
        args.config,
        jobs=args.jobs,
        keep_run_json=args.keep_run_json,
        samples_only=args.write_samples_only,
        slurm=args.slurm or args.slurm_dry_run,
        slurm_dry_run=args.slurm_dry_run,
        slurm_config=args.slurm_config,
        samples_dir=Path(args.samples_dir),
    )


if __name__ == "__main__":
    main()
