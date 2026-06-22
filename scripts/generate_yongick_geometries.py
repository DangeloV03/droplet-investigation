#!/usr/bin/env python3
"""Generate initial .npy files for the Yongick geometry comparison sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulation import YONGICK_GEOMETRY_BUILDERS, yongick_geometry_path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "yongick_geometry_sweep.json"
DEFAULT_OUTDIR = REPO_ROOT / "geometries" / "yongick"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Master JSON listing geometry_label values to build",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help="Directory for generated .npy files",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    fixed = cfg["fixed"]
    labels = cfg.get("sweep", {}).get("geometry_label", sorted(YONGICK_GEOMETRY_BUILDERS))
    lattice_size = int(fixed["lattice_size"])
    concentration = float(fixed["concentration"])
    geometry_seed = int(fixed.get("geometry_seed", 0))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Lattice: {lattice_size}², concentration={concentration}, seed={geometry_seed}")
    for label in labels:
        builder = YONGICK_GEOMETRY_BUILDERS[label]
        state = builder(lattice_size, concentration, geometry_seed)
        path = outdir / f"{label}.npy"
        np.save(path, state)
        occupied = int(np.count_nonzero(state))
        print(f"  {label}: {occupied} occupied sites -> {path}")


if __name__ == "__main__":
    main()
