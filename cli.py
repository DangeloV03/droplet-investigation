"""
Interactive and batch CLI for HeteroNVTDrivenChain dividing-droplet runs.

Two modes:
  - run_interactive(): prompts for every parameter and runs a single
    simulation via simulation.run_chunked_simulation.
  - run_from_params_file(): reads a simple key=value params file, sweeps
    over one parameter (a comma-separated list of values for one key),
    and runs simulation.run_chunked_simulation once per value.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from simulation import (
    EMPTY,
    RunParams,
    load_or_create_geometry,
    make_seed_geometry,
    make_timestamped_run_dir,
    run_chunked_simulation,
)

PARAM_KEYS = frozenset({
    "bond_energy",
    "delta_f",
    "delta_mu",
    "diffusion_lamda",
    "scheme",
    "concentration",
    "equilibration_time",
    "chunk_time",
    "num_chunks",
    "seed",
    "lattice_size",
    "radius",
    "geometry_seed",
    "initial_npy",
    "output_dir",
    "run_prefix",
})

SWEEPABLE_KEYS = frozenset({
    "bond_energy",
    "delta_f",
    "delta_mu",
    "diffusion_lamda",
    "scheme",
    "concentration",
    "equilibration_time",
    "chunk_time",
    "num_chunks",
    "seed",
})

INT_KEYS = frozenset({"seed", "lattice_size", "radius", "geometry_seed", "num_chunks"})
STR_KEYS = frozenset({"scheme", "initial_npy", "output_dir", "run_prefix"})


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def prompt_str(label: str, default: str | None = None) -> str:
    if default is not None:
        raw = input(f"{label} [{default}]: ").strip()
        return raw if raw else default
    while True:
        raw = input(f"{label}: ").strip()
        if raw:
            return raw
        print("  (required)")


def prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [{default}]: ").strip()
    return float(raw) if raw else default


def prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    return int(raw) if raw else default


def prompt_yes_no(label: str, default: bool = False) -> bool:
    default_str = "y" if default else "n"
    raw = input(f"{label} (y/n) [{default_str}]: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes", "1", "true")


def prompt_scheme(default: str = "negative_drive") -> str:
    print("  Scheme options: homo, negative_drive, positive_drive")
    while True:
        scheme = prompt_str("Scheme", default=default)
        if scheme in ("homo", "negative_drive", "positive_drive"):
            return scheme
        print("  Invalid scheme.")


def format_value_for_filename(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"[^A-Za-z0-9._-]", "", value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return str(value).replace(".", "p").replace("-", "m")
    return str(value)


# ---------------------------------------------------------------------------
# Params file I/O
# ---------------------------------------------------------------------------

def parse_params_file(path: str) -> dict[str, list[str]]:
    params: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{line_no}: expected key=value, got {line!r}")
            key, raw_val = line.split("=", 1)
            key = key.strip()
            if key not in PARAM_KEYS:
                raise ValueError(
                    f"{path}:{line_no}: unknown key {key!r}. "
                    f"Allowed: {', '.join(sorted(PARAM_KEYS))}"
                )
            values = [v.strip() for v in raw_val.split(",") if v.strip()]
            if not values:
                raise ValueError(f"{path}:{line_no}: no values for {key}")
            params[key] = values
    return params


def coerce_value(key: str, raw: str) -> Any:
    if key in STR_KEYS:
        if key == "scheme" and raw not in ("homo", "negative_drive", "positive_drive"):
            raise ValueError(f"Invalid scheme {raw!r}")
        return raw
    if key in INT_KEYS:
        return int(raw)
    return float(raw)


def params_from_parsed(
    parsed: dict[str, list[str]],
    overrides: dict[str, Any] | None = None,
) -> RunParams:
    base: dict[str, Any] = {}
    for key, values in parsed.items():
        if key in RunParams.__dataclass_fields__:
            base[key] = coerce_value(key, values[0])
    if overrides:
        base.update(overrides)
    return RunParams(**{k: v for k, v in base.items() if k in RunParams.__dataclass_fields__})


def sweep_values(parsed: dict[str, list[str]], sweep_key: str) -> list[Any]:
    if sweep_key not in parsed:
        raise ValueError(f"Key {sweep_key!r} not found in params file")
    return [coerce_value(sweep_key, v) for v in parsed[sweep_key]]


def output_basename(prefix: str, sweep_key: str, sweep_value: Any) -> str:
    return f"{prefix}_{sweep_key}{format_value_for_filename(sweep_value)}"


def build_output_basename(run: dict[str, Any], sweep_keys: list[str]) -> str:
    prefix = run.get("run_prefix", "run")
    parts = [prefix]
    for key in sorted(sweep_keys):
        if key in run:
            parts.append(f"{key}{format_value_for_filename(run[key])}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_interactive() -> None:
    print("\n--- Interactive single run ---\n")

    if prompt_yes_no("Create new central-droplet .npy?", default=True):
        lattice_size = prompt_int("Lattice size", default=128)
        concentration = prompt_float("Total concentration (occupied fraction, B+I)", default=0.3)
        radius = prompt_int("Initial droplet radius (sites)", default=35)
        seed_geom = prompt_int("Geometry random seed", default=0)
        initial_npy = prompt_str("Path to save initial geometry", default="initial_geometry.npy")
        try:
            state = make_seed_geometry(
                lattice_size=lattice_size,
                concentration=concentration,
                radius=radius,
                seed=seed_geom,
            )
        except ValueError as exc:
            print(f"  Error: {exc}")
            return
        np.save(initial_npy, state)
        print(f"  Saved initial geometry -> {initial_npy}")
    else:
        initial_npy = prompt_str("Path to initial geometry .npy")
        state = np.load(initial_npy).astype(np.uint32)
        lattice_size = state.shape[0]
        concentration = float(np.count_nonzero(state != EMPTY)) / state.size
        radius = 0  # not meaningful when loading an existing geometry

    output_dir = prompt_str("Output directory", default="results")

    print("\n--- Chain parameters ---")
    params = RunParams(
        bond_energy=prompt_float("bond_energy (epsilon)", default=-2.0),
        delta_f=prompt_float("delta_f", default=-1.0),
        delta_mu=prompt_float("delta_mu", default=0.0),
        diffusion_lamda=prompt_float("diffusion_lamda", default=1.0),
        scheme=prompt_scheme(default="negative_drive"),
        concentration=concentration,
        equilibration_time=prompt_float("Equilibration time (KMC)", default=1000.0),
        chunk_time=prompt_float("Production chunk time (KMC)", default=100.0),
        num_chunks=prompt_int("Number of production chunks", default=100),
        seed=prompt_int("Random seed", default=42),
        lattice_size=lattice_size,
        radius=radius,
        initial_npy=initial_npy,
        output_dir=output_dir,
    )

    run_dir = make_timestamped_run_dir(params.output_dir, params.run_prefix)
    print(f"\nRunning in {run_dir} ...")
    result = run_chunked_simulation(params, state, run_dir, label=params.run_prefix)

    print("\n=== Done ===")
    print(f"Total KMC time: {result['final_time']:.4f}")
    print(f"Particles conserved: {result['n_before']} -> {result['n_after']}")
    print(f"Run folder: {result['run_dir']}")


def run_from_params_file() -> None:
    print("\n--- Batch run from params file ---\n")

    params_path = prompt_str("Path to params .txt file", default="example_params.txt")
    parsed = parse_params_file(params_path)
    base = params_from_parsed(parsed)

    multi_keys = [k for k in SWEEPABLE_KEYS if k in parsed and len(parsed[k]) > 1]
    if not multi_keys:
        sweep_key = prompt_str("Which parameter varies per run?")
    else:
        print("Parameters with multiple values in file:")
        for k in multi_keys:
            print(f"  {k} = {', '.join(parsed[k])}")
        sweep_key = prompt_str("Which parameter varies per run?", default=multi_keys[0])

    if sweep_key not in SWEEPABLE_KEYS:
        raise ValueError(f"Cannot sweep {sweep_key!r}")

    values = sweep_values(parsed, sweep_key)
    prefix = prompt_str("Run name prefix", default=base.run_prefix)
    output_dir = prompt_str("Output directory", default=base.output_dir)

    print(f"\nLoading geometry from {base.initial_npy} ...")
    initial_state = load_or_create_geometry(base)

    print(f"\nRunning {len(values)} simulations (sweeping {sweep_key}) ...")
    for i, value in enumerate(values, start=1):
        run_params = params_from_parsed(parsed, overrides={sweep_key: value})
        run_params.run_prefix = prefix
        run_params.output_dir = output_dir
        label = output_basename(prefix, sweep_key, value)
        run_dir = make_timestamped_run_dir(output_dir, label)

        print(f"\n[{i}/{len(values)}] {sweep_key}={value} -> {run_dir}")
        result = run_chunked_simulation(
            run_params,
            initial_state,
            run_dir,
            label=label,
            extra_params={"sweep_key": sweep_key, "sweep_value": value},
        )
        print(
            f"  t={result['final_time']:.4f}, "
            f"particles {result['n_before']}->{result['n_after']}"
        )

    print(f"\n=== Done: {len(values)} run folders under {output_dir}/ ===")