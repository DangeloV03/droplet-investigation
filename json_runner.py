"""
JSON batch runner for HeteroNVTDrivenChain.

Reads a master JSON config, expands sweep arrays into individual runs,
executes equilibration + production chunks for each, and writes a
timestamped subfolder per run with params, lattice snapshots, and density
time series.

Usage:
    python json_runner.py example_sweep.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from itertools import product
from pathlib import Path
from typing import Any

from cli import SWEEPABLE_KEYS, build_output_basename
from simulation import (
    RunParams,
    load_or_create_geometry,
    make_timestamped_run_dir,
    run_chunked_simulation,
)

RUN_PARAM_KEYS = frozenset(RunParams.__dataclass_fields__.keys())
METADATA_KEYS = frozenset({"beta", "eta"})


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
    initial_state,
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


def run_from_master(path: str | Path, *, keep_run_json: bool = False) -> None:
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

    base_params = run_config_to_params(runs[0])
    print(f"Loading geometry from {base_params.initial_npy} ...")
    initial_state = load_or_create_geometry(base_params)

    staging_dir = tempfile.mkdtemp(prefix="json_runner_")
    print(f"Staging per-run JSON in {staging_dir} (deleted after batch)\n")

    try:
        for i, run in enumerate(runs, start=1):
            label = build_output_basename(run, sweep_keys)
            run_json_path = os.path.join(staging_dir, f"{label}.json")

            with open(run_json_path, "w", encoding="utf-8") as f:
                json.dump(run, f, indent=2)

            sweep_desc = ", ".join(f"{k}={run[k]}" for k in sweep_keys)
            print(f"[{i}/{n}] {sweep_desc or 'fixed params'}")
            print(f"  staging config: {run_json_path}")

            result = execute_single_run(run, initial_state, sweep_keys)
            print(
                f"  -> {result['run_dir']}\n"
                f"     t={result['final_time']:.4f}, "
                f"N={result['n_before']}->{result['n_after']}"
            )

            if not keep_run_json:
                os.remove(run_json_path)

        print(f"\n=== Done: {n} timestamped run folders under {base_params.output_dir}/ ===")
    finally:
        if not keep_run_json:
            shutil.rmtree(staging_dir, ignore_errors=True)
        else:
            print(f"Per-run staging JSON kept in {staging_dir}")


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
        "--keep-run-json",
        action="store_true",
        help="Keep per-run staging JSON files (for debugging)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    run_from_master(args.config, keep_run_json=args.keep_run_json)


if __name__ == "__main__":
    main()
