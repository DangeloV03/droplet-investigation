#!/usr/bin/env python3
"""Generate initial .npy files for the Yongick geometry comparison sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from json_runner import expand_runs
from simulation import YONGICK_GEOMETRY_BUILDERS, YONGICK_GEOMETRY_RADII, yongick_droplet_radius

DEFAULT_CONFIG = REPO_ROOT / "yongick_geometry_sweep.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Master JSON listing sweep axes (lattice_size, geometry_label)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    fixed = cfg["fixed"]
    sweep = cfg.get("sweep", {})
    runs = expand_runs(fixed, sweep)
    concentration = float(fixed["concentration"])
    geometry_seed = int(fixed.get("geometry_seed", 0))

    print(f"Generating {len(runs)} geometry file(s), c={concentration}, seed={geometry_seed}")
    for run in runs:
        lattice_size = int(run["lattice_size"])
        label = run["geometry_label"]
        builder = YONGICK_GEOMETRY_BUILDERS[label]
        state = builder(lattice_size, concentration, geometry_seed)
        path = Path(run["initial_npy"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, state)
        occupied = int(np.count_nonzero(state))
        base_r = YONGICK_GEOMETRY_RADII[label]
        r_note = (
            f"r={yongick_droplet_radius(base_r, lattice_size)}"
            if base_r is not None
            else "uniform"
        )
        print(f"  {lattice_size}² {label} ({r_note}): {occupied} occupied sites -> {path}")


if __name__ == "__main__":
    main()
