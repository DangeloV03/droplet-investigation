#!/usr/bin/env python3
"""Build split_droplet_initial_geometry.npy for the split-droplet sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from simulation import count_droplet_sites, make_split_droplet_geometry

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "split_droplet_sweep.json"
DEFAULT_OUTPUT = REPO_ROOT / "split_droplet_initial_geometry.npy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Master JSON with lattice/concentration/radius settings",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the generated .npy file",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        fixed = json.load(f)["fixed"]

    lattice_size = int(fixed["lattice_size"])
    radius = int(fixed["radius"])
    concentration = float(fixed["concentration"])
    geometry_seed = int(fixed.get("geometry_seed", 0))

    ref_sites = count_droplet_sites(lattice_size, radius)
    per_droplet = ref_sites // 2
    total = round(concentration * lattice_size ** 2)

    print(f"Reference central disk (r={radius}): {ref_sites} sites")
    print(f"Each split droplet: {per_droplet} particles")
    print(f"Total lattice particles (c={concentration}): {total}")

    state = make_split_droplet_geometry(
        lattice_size=lattice_size,
        concentration=concentration,
        single_droplet_radius=radius,
        seed=geometry_seed,
    )

    np.save(args.output, state)
    occupied = int(np.count_nonzero(state))
    print(f"Wrote {state.shape} lattice -> {args.output} ({occupied} occupied sites)")


if __name__ == "__main__":
    main()
