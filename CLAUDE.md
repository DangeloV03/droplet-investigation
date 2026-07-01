# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a 2D lattice-gas Kinetic Monte Carlo (KMC) simulation platform for studying nonequilibrium dividing droplet dynamics. The physics: particles on an N×N periodic lattice are one of three species — `0` (Empty), `1` (Inert/Inactive), `2` (Bonding/Active) — and evolve via diffusion hops and nonequilibrium `B ↔ I` interconversions driven by `delta_f` and `delta_mu`.

**This repo is the application layer.** The core KMC engine lives in a sibling crate `../lattice-gas` (Rust, exposed to Python via PyO3/Maturin as the `lattice_gas` package). You cannot run simulations without building that crate first.

## Build & Setup

### 1. Build the Rust/Python extension (required before any Python simulation)
```bash
cd ../lattice-gas
maturin develop --release --features "extension-module"
```
On macOS, you may need `DYLD_LIBRARY_PATH` pointing to your Python env's lib folder if the `.so` fails to load.

### 2. Install Python dependencies
```bash
pip install -r requirements.txt   # numpy scipy matplotlib pyyaml
# maturin must also be installed
```

### 3. Build native Rust verification binaries
```bash
cargo build --release
```

## Running Simulations

### Interactive single run
```bash
python main.py
```
Prompts for all parameters, or reads from a `key=value` params file (see `example_params.txt`).

### Multi-dimensional JSON sweep (local)
```bash
python json_runner.py example_sweep.json
python json_runner.py delta_f_sweep.json --jobs 4   # parallel with 4 workers
```

### Generate Yongick geometry `.npy` files before sweeping
```bash
python scripts/generate_yongick_geometries.py --config yongick_geometry_sweep.json
```
This must be run before any sweep that sets `geometry_label`.

### Rust verification binaries
```bash
cargo run --release --bin test1 -- config.toml   # reaction & phase-separation
cargo run --release --bin test2 -- config.toml   # diffusion MSD
```

## HPC / Slurm

Cluster submission uses two parallel systems; prefer the shell script approach:

**Shell-based (preferred):**
```bash
# Requires scripts/slurm.env (copy from scripts/slurm.env.example and fill in)
./scripts/submit_sweep.sh delta_f_sweep.json
./scripts/submit_sweep.sh --force yongick_geometry_sweep.json  # re-run completed
```

**Python-based (legacy):**
```bash
python json_runner.py delta_f_sweep.json --slurm                # submit
python json_runner.py delta_f_sweep.json --slurm-dry-run        # preview batch scripts
```

Cluster config lives in `slurm_config.yml` (partition, mem, walltime, `setup_cmds` for conda activation). The `project_root` and `report_dir` in that file are Princeton's Della/scratch paths.

## Architecture

### Python layer
- **`simulation.py`** — core engine: geometry builders, `RunParams` dataclass, `run_chunked_simulation()` (equilibration then N production chunks), all I/O (CSV, PNG, NPY).
- **`json_runner.py`** — batch sweep runner: reads a master JSON with `fixed` + `sweep` dicts, Cartesian-expands into per-run job JSONs under `samples/`, then runs locally or submits to Slurm.
- **`cli.py`** — interactive prompts and `SWEEPABLE_KEYS` (keys valid in `sweep` dict).
- **`cluster.py`** — `largest_cluster_stats()` and `far_field_densities()` analysis functions.
- **`slurm_submit.py`** — Python Slurm submission path; shell scripts are preferred.

### JSON sweep format
```json
{
  "fixed": { "bond_energy": -2.95, "lattice_size": 256, ... },
  "sweep": { "delta_f": [-1.0, 0.0, 1.0], "geometry_label": ["single_r25", "homogeneous"] }
}
```
`sweep` values are Cartesian-producted. Only keys in `SWEEPABLE_KEYS` (cli.py) may appear in `sweep`.

### Geometry labels (`geometry_label` field)
The Yongick geometry sweep uses four named initial conditions:
- `single_r25` — one central droplet
- `two_r15` — two droplets left/right
- `nine_r8` — 3×3 grid of nine droplets
- `homogeneous` — uniform random placement

Radii auto-scale with lattice size (reference: r=25 at 256²). Initial `.npy` files live under `geometries/yongick/<lattice_size>/<label>.npy` and must be pre-generated.

### Run output structure
Each run writes a timestamped subfolder `results/<YYYYMMDD_HHMMSS>_<label>/` containing:
`params.json`, `equilibrated.npy/.png`, `final_state.npy/.png`, `density_series.csv/.png`, `cluster_series.csv/.png`, `farfield_series.csv/.png`.

Runs are skipped automatically if a completed folder (containing `cluster_series.csv` + `params.json`) already exists for that label.

### Fixed simulation constants
`BETA = 1.0` and `ETA = 1.0` are fixed by convention throughout; energy scales are absorbed into `bond_energy`, `delta_f`, `delta_mu`.

### Rust crate (`src/`)
`test1.rs` and `test2.rs` are standalone binaries (not a library) that link directly against `../lattice-gas` as a cargo dependency. `src/config.rs` parses `config.toml` for them.
