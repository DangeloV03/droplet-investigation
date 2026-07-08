"""
Dissolve runner: drive a pre-equilibrated droplet until it slowly dissolves.

Loads a fixed pre-equilibrated lattice (.npy) as the initial state, turns the
drive ON (delta_mu = 1.0 by default), and runs a single long phase with many
snapshots so the slow dissolution can be watched frame by frame.

By default it runs 120 chunks (10x the poster drive-on phase) at the same
20000-KMC snapshot interval.

Local usage:
    python dissolve_run.py
    python dissolve_run.py --delta-mu-drive 1.0 --num-chunks 120

Slurm usage:
    python dissolve_run.py --slurm
    python dissolve_run.py --slurm --dry-run
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
    make_timestamped_run_dir,
    run_dissolve_simulation,
    yongick_droplet_radius,
)


# ---------------------------------------------------------------------------
# Defaults (Yongick poster baseline physics; drive delta_mu = 1.0)
# ---------------------------------------------------------------------------

DEFAULT_INITIAL_NPY = "dissolve_initial_L128.npy"
DEFAULT_BOND_ENERGY = -2.95
DEFAULT_DELTA_F = 1.7337
DEFAULT_SCHEME = "negative_drive"
DEFAULT_CONCENTRATION = 0.05
DEFAULT_DIFFUSION_LAMDA = 100.0

DEFAULT_DELTA_MU_DRIVE = 1.0
DEFAULT_SNAPSHOT_INTERVAL = 20_000.0
DEFAULT_NUM_CHUNKS = 120  # 10x the ~12-chunk poster drive-on phase


def _label(lattice_size: int, delta_mu_drive: float) -> str:
    dm_str = f"{delta_mu_drive:.1f}".replace(".", "p").replace("-", "m")
    return f"dissolve_L{lattice_size}_dm{dm_str}"


def _load_initial_state(initial_npy: str) -> np.ndarray:
    if not os.path.exists(initial_npy):
        print(f"ERROR: initial state not found: {initial_npy}", file=sys.stderr)
        sys.exit(1)
    state = np.load(initial_npy).astype(np.uint32)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        print(f"ERROR: expected a square 2D lattice, got shape {state.shape}", file=sys.stderr)
        sys.exit(1)
    return state


def _find_existing_run(output_dir: str, label: str) -> str | None:
    """Return the most recent partial run directory for this label, or None."""
    root = Path(output_dir)
    if not root.is_dir():
        return None
    suffix = f"_{label}"
    matches = [d for d in root.iterdir() if d.is_dir() and d.name.endswith(suffix)]
    return str(sorted(matches)[-1]) if matches else None


def run_local(args: argparse.Namespace) -> None:
    initial_state = _load_initial_state(args.initial_npy)
    lattice_size = int(initial_state.shape[0])
    delta_mu_drive: float = args.delta_mu_drive
    label = _label(lattice_size, delta_mu_drive)

    print(f"=== Dissolve simulation: L={lattice_size}, delta_mu_drive={delta_mu_drive} ===")
    print(f"  Initial state: {args.initial_npy}")

    params = RunParams(
        bond_energy=args.bond_energy,
        delta_f=args.delta_f,
        delta_mu=0.0,  # overridden to drive_delta_mu inside run_dissolve_simulation
        diffusion_lamda=args.diffusion_lamda,
        scheme=args.scheme,
        concentration=args.concentration,
        equilibration_time=0.0,
        chunk_time=args.snapshot_interval,
        num_chunks=args.num_chunks,
        seed=args.seed,
        lattice_size=lattice_size,
        radius=yongick_droplet_radius(25, lattice_size),
        geometry_seed=args.seed,
        initial_npy=args.initial_npy,
        output_dir=args.output_dir,
        run_prefix="dissolve",
        geometry_label="preequilibrated",
    )

    existing = _find_existing_run(args.output_dir, label)
    if existing:
        run_dir = existing
        print(f"  Found partial run — resuming: {run_dir}")
    else:
        run_dir = make_timestamped_run_dir(args.output_dir, label)

    result = run_dissolve_simulation(
        params,
        initial_state=initial_state,
        run_dir=run_dir,
        drive_delta_mu=delta_mu_drive,
        num_chunks=args.num_chunks,
        snapshot_interval=args.snapshot_interval,
    )

    print(
        f"\n=== Done ===\n"
        f"  {result['run_dir']}\n"
        f"  {result['num_chunks']} chunks, final t={result['final_time']:.2f}"
    )


def _build_batch_script(args: argparse.Namespace, cfg: dict, lattice_size: int) -> str:
    """Build a Slurm batch script that re-invokes dissolve_run.py without --slurm."""
    from slurm_submit import expand_user_vars

    cfg_root = str(cfg.get("project_root", "")).strip()
    root = Path(cfg_root).expanduser().resolve() if cfg_root else Path.cwd().resolve()
    report_dir = expand_user_vars(str(cfg["report_dir"]))
    job_label = _label(lattice_size, args.delta_mu_drive)
    job_name = f"{cfg['job_name']}_{job_label}"[:64]
    stdout = expand_user_vars(str(cfg.get("output", f"{report_dir}/%j.out")))
    stderr = expand_user_vars(str(cfg.get("error", f"{report_dir}/%j.err")))

    # NOTE: #SBATCH directives must precede the first non-comment, non-blank
    # line or Slurm silently ignores them. Keep `set -euo pipefail` AFTER them.
    lines = ["#!/bin/bash"]
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
    lines.append("set -euo pipefail")
    lines.append("")

    for cmd in cfg.get("setup_cmds", []):
        lines.append(str(cmd))
    lines.append("")

    dissolve_cmd = (
        f"python dissolve_run.py"
        f" --initial-npy {shlex.quote(args.initial_npy)}"
        f" --delta-mu-drive {args.delta_mu_drive}"
        f" --num-chunks {args.num_chunks}"
        f" --snapshot-interval {args.snapshot_interval}"
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
        dissolve_cmd,
        "",
    ]
    return "\n".join(lines)


def run_slurm(args: argparse.Namespace) -> None:
    from slurm_submit import load_slurm_config, expand_user_vars

    lattice_size = int(_load_initial_state(args.initial_npy).shape[0])

    cfg = load_slurm_config(args.slurm_config)
    if args.time_minutes is not None:
        cfg["time_minutes"] = args.time_minutes
    script = _build_batch_script(args, cfg, lattice_size)

    if args.dry_run:
        print(script)
        print("--- dry-run: sbatch not invoked ---")
        return

    report_dir = expand_user_vars(str(cfg["report_dir"]))
    os.makedirs(report_dir, exist_ok=True)

    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".slurm", prefix="dissolve_sim_",
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
        description="Dissolve a pre-equilibrated droplet under a constant drive.",
    )

    parser.add_argument("--initial-npy", default=DEFAULT_INITIAL_NPY,
                        help=f"Pre-equilibrated lattice .npy (default: {DEFAULT_INITIAL_NPY})")
    parser.add_argument("--delta-mu-drive", type=float, default=DEFAULT_DELTA_MU_DRIVE,
                        help=f"delta_mu during the drive (default: {DEFAULT_DELTA_MU_DRIVE})")
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS,
                        help=f"Number of snapshot chunks (default: {DEFAULT_NUM_CHUNKS})")
    parser.add_argument("--snapshot-interval", type=float, default=DEFAULT_SNAPSHOT_INTERVAL,
                        help=f"KMC time between snapshots (default: {DEFAULT_SNAPSHOT_INTERVAL})")

    # Physics
    parser.add_argument("--bond-energy", type=float, default=DEFAULT_BOND_ENERGY)
    parser.add_argument("--delta-f", type=float, default=DEFAULT_DELTA_F)
    parser.add_argument("--concentration", type=float, default=DEFAULT_CONCENTRATION)
    parser.add_argument("--diffusion-lamda", type=float, default=DEFAULT_DIFFUSION_LAMDA)
    parser.add_argument("--scheme", default=DEFAULT_SCHEME)

    # I/O
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)

    # Slurm
    parser.add_argument("--slurm", action="store_true",
                        help="Submit to Slurm instead of running locally")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --slurm: print batch script without submitting")
    parser.add_argument("--slurm-config", default="slurm_config.yml")
    parser.add_argument("--time-minutes", type=int, default=None,
                        help="Override walltime (integer minutes) from slurm_config.yml")

    args = parser.parse_args()

    if args.slurm or args.dry_run:
        run_slurm(args)
    else:
        run_local(args)


if __name__ == "__main__":
    main()
