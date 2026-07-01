"""
Poster simulation runner: equilibrate → drive ON → drive OFF.

Runs a single droplet (single_r25 geometry) through three phases and saves
lattice snapshots (.npy + .png) every snapshot_interval KMC time units.

Local usage:
    python poster_run.py --lattice-size 256
    python poster_run.py --lattice-size 128 --delta-mu-drive 5.0

Slurm usage:
    python poster_run.py --lattice-size 256 --slurm
    python poster_run.py --lattice-size 256 --slurm --dry-run
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from simulation import (
    RunParams,
    build_yongick_geometry,
    make_timestamped_run_dir,
    run_poster_simulation,
    yongick_droplet_radius,
    yongick_geometry_path,
)


# ---------------------------------------------------------------------------
# Default physics parameters (Yongick sweep baseline)
# ---------------------------------------------------------------------------

DEFAULT_BOND_ENERGY = -2.95
DEFAULT_DELTA_F = 1.7337
DEFAULT_SCHEME = "negative_drive"
DEFAULT_CONCENTRATION = 0.05
DEFAULT_DIFFUSION_LAMDA = 100.0
DEFAULT_GEOMETRY_LABEL = "single_r25"
DEFAULT_DELTA_MU_DRIVE = 3.0

DEFAULT_EQ_TIME = 1_000_000.0
DEFAULT_DRIVE_ON_TIME = 500_000.0
DEFAULT_DRIVE_OFF_TIME = 500_000.0
DEFAULT_SNAPSHOT_INTERVAL = 10_000.0


def _label(lattice_size: int, delta_mu_drive: float) -> str:
    dm_str = f"{delta_mu_drive:.1f}".replace(".", "p").replace("-", "m")
    return f"poster_L{lattice_size}_dm{dm_str}"


def _load_or_build_geometry(
    lattice_size: int,
    concentration: float,
    geometry_label: str,
    geometry_root: str,
    seed: int,
) -> np.ndarray:
    npy_path = yongick_geometry_path(geometry_label, lattice_size, root=geometry_root)
    if os.path.exists(npy_path):
        state = np.load(npy_path).astype(np.uint32)
        if state.shape == (lattice_size, lattice_size):
            print(f"  Loaded geometry from {npy_path}")
            return state
        print(f"  {npy_path} is {state.shape}, expected ({lattice_size},{lattice_size}) — rebuilding")

    print(f"  Building {geometry_label} geometry (L={lattice_size}) ...")
    state = build_yongick_geometry(geometry_label, lattice_size, concentration, seed=seed)
    path = Path(npy_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, state)
    print(f"  Saved to {npy_path}")
    return state


def run_local(args: argparse.Namespace) -> None:
    lattice_size: int = args.lattice_size
    delta_mu_drive: float = args.delta_mu_drive

    print(f"=== Poster simulation: L={lattice_size}, delta_mu_drive={delta_mu_drive} ===")

    initial_state = _load_or_build_geometry(
        lattice_size,
        args.concentration,
        args.geometry_label,
        args.geometry_root,
        seed=args.seed,
    )

    params = RunParams(
        bond_energy=args.bond_energy,
        delta_f=args.delta_f,
        delta_mu=0.0,
        diffusion_lamda=args.diffusion_lamda,
        scheme=args.scheme,
        concentration=args.concentration,
        equilibration_time=args.eq_time,
        chunk_time=args.snapshot_interval,
        num_chunks=round(args.drive_on_time / args.snapshot_interval),
        seed=args.seed,
        lattice_size=lattice_size,
        radius=yongick_droplet_radius(25, lattice_size),
        geometry_seed=args.seed,
        initial_npy=yongick_geometry_path(args.geometry_label, lattice_size,
                                          root=args.geometry_root),
        output_dir=args.output_dir,
        run_prefix="poster",
        geometry_label=args.geometry_label,
    )

    run_dir = make_timestamped_run_dir(args.output_dir, _label(lattice_size, delta_mu_drive))

    result = run_poster_simulation(
        params,
        drive_on_delta_mu=delta_mu_drive,
        initial_state=initial_state,
        run_dir=run_dir,
        eq_time=args.eq_time,
        drive_on_time=args.drive_on_time,
        drive_off_time=args.drive_off_time,
        snapshot_interval=args.snapshot_interval,
    )

    print(
        f"\n=== Done ===\n"
        f"  {result['run_dir']}\n"
        f"  eq ended at t={result['equilibration_end_time']:.2f}\n"
        f"  drive ON ended at t={result['drive_on_end_time']:.2f}\n"
        f"  drive OFF ended at t={result['final_time']:.2f}"
    )


def _build_batch_script(args: argparse.Namespace, cfg: dict) -> str:
    """Build a Slurm batch script that re-invokes poster_run.py without --slurm."""
    from slurm_submit import expand_user_vars, project_root

    root = project_root(cfg)
    report_dir = expand_user_vars(str(cfg["report_dir"]))
    job_label = _label(args.lattice_size, args.delta_mu_drive)
    job_name = f"{cfg['job_name']}_{job_label}"[:64]
    stdout = expand_user_vars(str(cfg.get("output", f"{report_dir}/%j.out")))
    stderr = expand_user_vars(str(cfg.get("error", f"{report_dir}/%j.err")))

    lines = ["#!/bin/bash", "set -euo pipefail", ""]
    lines += [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={cfg['partition']}",
        f"#SBATCH --cpus-per-task={cfg['cpus_per_task']}",
        f"#SBATCH --mem={cfg.get('mem', '8G')}",
        f"#SBATCH --time={cfg['time_minutes']}",
        f"#SBATCH --output={stdout}",
        f"#SBATCH --error={stderr}",
    ]
    if cfg.get("account"):
        lines.append(f"#SBATCH --account={cfg['account']}")
    if cfg.get("qos"):
        lines.append(f"#SBATCH --qos={cfg['qos']}")
    lines.append("")

    for cmd in cfg.get("setup_cmds", []):
        lines.append(str(cmd))
    lines.append("")

    poster_cmd = (
        f"python poster_run.py"
        f" --lattice-size {args.lattice_size}"
        f" --delta-mu-drive {args.delta_mu_drive}"
        f" --eq-time {args.eq_time}"
        f" --drive-on-time {args.drive_on_time}"
        f" --drive-off-time {args.drive_off_time}"
        f" --snapshot-interval {args.snapshot_interval}"
        f" --geometry-label {shlex.quote(args.geometry_label)}"
        f" --geometry-root {shlex.quote(args.geometry_root)}"
        f" --output-dir {shlex.quote(args.output_dir)}"
        f" --seed {args.seed}"
        f" --bond-energy {args.bond_energy}"
        f" --delta-f {args.delta_f}"
        f" --concentration {args.concentration}"
        f" --diffusion-lamda {args.diffusion_lamda}"
        f" --scheme {shlex.quote(args.scheme)}"
    )

    lines += [
        f"cd {shlex.quote(str(root))}",
        f"export PROJECT_ROOT={shlex.quote(str(root))}",
        poster_cmd,
        "",
    ]
    return "\n".join(lines)


def run_slurm(args: argparse.Namespace) -> None:
    from slurm_submit import load_slurm_config, expand_user_vars

    cfg = load_slurm_config(args.slurm_config)
    script = _build_batch_script(args, cfg)

    if args.dry_run:
        print(script)
        print("--- dry-run: sbatch not invoked ---")
        return

    report_dir = expand_user_vars(str(cfg["report_dir"]))
    os.makedirs(report_dir, exist_ok=True)

    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".slurm", prefix="poster_sim_",
            delete=False, encoding="utf-8",
        ) as f:
            f.write(script)
            script_path = f.name

        proc = subprocess.run(
            ["sbatch", script_path],
            text=True, capture_output=True, check=False,
        )
    finally:
        if script_path and os.path.exists(script_path):
            os.unlink(script_path)

    if proc.returncode != 0:
        print(f"sbatch failed:\n{proc.stderr.strip()}", file=sys.stderr)
        sys.exit(proc.returncode)

    print(proc.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poster 3-phase simulation: equilibrate → drive on → drive off",
    )

    # Core args
    parser.add_argument("--lattice-size", type=int, required=True,
                        help="Lattice side length (e.g. 256 or 128)")
    parser.add_argument("--delta-mu-drive", type=float, default=DEFAULT_DELTA_MU_DRIVE,
                        help=f"delta_mu during drive-ON phase (default: {DEFAULT_DELTA_MU_DRIVE})")

    # Timing
    parser.add_argument("--eq-time", type=float, default=DEFAULT_EQ_TIME)
    parser.add_argument("--drive-on-time", type=float, default=DEFAULT_DRIVE_ON_TIME)
    parser.add_argument("--drive-off-time", type=float, default=DEFAULT_DRIVE_OFF_TIME)
    parser.add_argument("--snapshot-interval", type=float, default=DEFAULT_SNAPSHOT_INTERVAL,
                        help="KMC time between lattice snapshots (default: 10000)")

    # Physics
    parser.add_argument("--bond-energy", type=float, default=DEFAULT_BOND_ENERGY)
    parser.add_argument("--delta-f", type=float, default=DEFAULT_DELTA_F)
    parser.add_argument("--concentration", type=float, default=DEFAULT_CONCENTRATION)
    parser.add_argument("--diffusion-lamda", type=float, default=DEFAULT_DIFFUSION_LAMDA)
    parser.add_argument("--scheme", default=DEFAULT_SCHEME)

    # Geometry / I/O
    parser.add_argument("--geometry-label", default=DEFAULT_GEOMETRY_LABEL)
    parser.add_argument("--geometry-root", default="geometries/yongick")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)

    # Slurm
    parser.add_argument("--slurm", action="store_true",
                        help="Submit to Slurm instead of running locally")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --slurm: print batch script without submitting")
    parser.add_argument("--slurm-config", default="slurm_config.yml")

    args = parser.parse_args()

    if args.slurm or args.dry_run:
        run_slurm(args)
    else:
        run_local(args)


if __name__ == "__main__":
    main()
