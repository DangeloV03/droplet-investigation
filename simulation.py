"""
Simulation engine for HeteroNVTDrivenChain dividing-droplet runs.

This module is the reusable "engine": building initial lattice
configurations, running the chunked equilibration + production protocol,
and writing/plotting the resulting density, cluster, and far-field series.
It has no interactive prompts — see cli.py for the interactive/batch runner.

Simulation protocol (matches test1.rs chunk logic):
  1. Equilibration for equilibration_time
  2. num_chunks production segments, each of chunk_time
  3. Density/cluster/far-field measurements recorded after equilibration
     and after each production chunk

Each run writes a timestamped subfolder containing:
  - params.json
  - equilibrated.npy / equilibrated.png
  - final_state.npy / final_state.png
  - density_series.csv / density_series.png
  - cluster_series.csv / cluster_series.png
  - farfield_series.csv / farfield_series.png
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from lattice_gas.markov_chain import HeteroNVTDrivenChain
from lattice_gas.boundary_condition import Periodic
from lattice_gas.ending_criterion import Time
from lattice_gas.simulate import simulate
from lattice_gas import load

from cluster import largest_cluster_stats, far_field_densities

EMPTY, INERT, BONDING = 0, 1, 2

# Fixed by convention: beta sets the overall energy scale (absorbed into
# bond_energy/delta_f/delta_mu instead), and eta=1 means driven and passive
# reaction pathways are weighted equally.
BETA = 1.0
ETA = 1.0


@dataclass
class RunParams:
    bond_energy: float = -2.0
    delta_f: float = -1.0
    delta_mu: float = 0.0
    diffusion_lamda: float = 1.0
    scheme: str = "negative_drive"
    concentration: float = 0.3
    equilibration_time: float = 1000.0
    chunk_time: float = 100.0
    num_chunks: int = 100
    seed: int = 42
    lattice_size: int = 128
    radius: int = 35
    geometry_seed: int = 0
    initial_npy: str = "initial_geometry.npy"
    output_dir: str = "results"
    run_prefix: str = "test1"


@dataclass
class DensityRow:
    chunk: int
    time: float
    rho_bonding: float
    rho_inert: float
    rho_empty: float


@dataclass
class ClusterRow:
    chunk: int
    time: float
    area: int
    perimeter: int
    r_eff: float


@dataclass
class FarFieldRow:
    chunk: int
    time: float
    rho_b_far: float
    rho_i_far: float


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def make_seed_geometry(
    lattice_size: int,
    concentration: float,
    radius: int,
    bonding_fraction: float = 0.85,
    seed: int = 0,
) -> np.ndarray:
    """
    Build an initial configuration: a central circular droplet of the given
    `radius`, filled with a B/I mixture at `bonding_fraction`, plus any
    remaining particles implied by `concentration` scattered randomly over
    the dilute region.

    `concentration` is the total occupied fraction (B + I) of all lattice
    sites: total_particles = round(concentration * lattice_size**2).

    Raises
    ------
    ValueError
        If the droplet itself (number of sites within `radius` of center)
        requires more particles than `concentration` provides.
    """
    rng = np.random.default_rng(seed)
    state = np.zeros((lattice_size, lattice_size), dtype=np.uint32)
    cy = cx = lattice_size // 2

    yy, xx = np.mgrid[0:lattice_size, 0:lattice_size]
    droplet_mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    droplet_area = int(np.count_nonzero(droplet_mask))

    total_particles = round(concentration * lattice_size ** 2)

    if droplet_area > total_particles:
        raise ValueError(
            f"Cannot build a radius-{radius} droplet ({droplet_area} sites) "
            f"at concentration={concentration} on a {lattice_size}x{lattice_size} "
            f"lattice (only {total_particles} total particles available). "
            f"Increase concentration or decrease radius."
        )

    # Fill the droplet with a B/I mixture.
    droplet_species = np.where(
        rng.random(droplet_area) < bonding_fraction, BONDING, INERT
    ).astype(np.uint32)
    state[droplet_mask] = droplet_species

    # Scatter any remaining particles randomly over the dilute region.
    leftover = total_particles - droplet_area
    if leftover > 0:
        outside_indices = np.flatnonzero(~droplet_mask.reshape(-1))
        chosen = rng.choice(outside_indices, size=leftover, replace=False)
        chosen_species = np.where(
            rng.random(leftover) < bonding_fraction, BONDING, INERT
        ).astype(np.uint32)
        flat_state = state.reshape(-1)
        flat_state[chosen] = chosen_species

    return state


def load_or_create_geometry(params: RunParams, create_if_missing: bool = True) -> np.ndarray:
    expected = (params.lattice_size, params.lattice_size)

    if os.path.exists(params.initial_npy):
        state = np.load(params.initial_npy).astype(np.uint32)
        if state.ndim != 2:
            raise ValueError(f"Expected 2D lattice in {params.initial_npy}")
        if state.shape == expected:
            return state
        print(
            f"  {params.initial_npy} is {state.shape}, expected {expected} "
            f"from lattice_size={params.lattice_size} — recreating geometry"
        )

    if not create_if_missing:
        raise FileNotFoundError(params.initial_npy)

    radius = min(params.radius, params.lattice_size // 2)
    state = make_seed_geometry(
        lattice_size=params.lattice_size,
        concentration=params.concentration,
        radius=radius,
        seed=params.geometry_seed,
    )
    np.save(params.initial_npy, state)
    print(f"  Created initial geometry {state.shape} -> {params.initial_npy}")
    return state


def make_timestamped_run_dir(output_dir: str, label: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{stamp}_{label}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def lattice_densities(state: np.ndarray) -> tuple[float, float, float]:
    total = state.size
    rho_b = float(np.count_nonzero(state == BONDING)) / total
    rho_i = float(np.count_nonzero(state == INERT)) / total
    rho_e = float(np.count_nonzero(state == EMPTY)) / total
    return rho_b, rho_i, rho_e


# ---------------------------------------------------------------------------
# Output: lattice snapshots
# ---------------------------------------------------------------------------

# Lattice colors: white = empty, blue = inert (inactive), red = bonding (active).
LATTICE_CMAP = ListedColormap(["#ffffff", "#2166ac", "#d73027"])
LATTICE_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], LATTICE_CMAP.N)


def save_lattice_png(state: np.ndarray, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        state,
        cmap=LATTICE_CMAP,
        norm=LATTICE_NORM,
        interpolation="nearest",
    )
    ax.set_title("Lattice (white=empty, blue=inert, red=bonding)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["empty", "inert", "bonding"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output: density series
# ---------------------------------------------------------------------------

def write_density_csv(path: str, rows: list[DensityRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk", "time", "rho_bonding", "rho_inert", "rho_empty"])
        for row in rows:
            writer.writerow([
                row.chunk,
                f"{row.time:.6f}",
                f"{row.rho_bonding:.6f}",
                f"{row.rho_inert:.6f}",
                f"{row.rho_empty:.6f}",
            ])


def plot_density_series(csv_path: str, png_path: str) -> None:
    times: list[float] = []
    rho_b: list[float] = []
    rho_i: list[float] = []
    rho_e: list[float] = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            rho_b.append(float(row["rho_bonding"]))
            rho_i.append(float(row["rho_inert"]))
            rho_e.append(float(row["rho_empty"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, rho_b, label="bonding (active)", color="#d73027")
    ax.plot(times, rho_i, label="inert (inactive)", color="#2166ac")
    ax.plot(times, rho_e, label="empty", color="#888888")
    ax.set_xlabel("KMC time")
    ax.set_ylabel("Density (fraction of sites)")
    ax.set_title("Species densities vs time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output: cluster series
# ---------------------------------------------------------------------------

def write_cluster_csv(path: str, rows: list[ClusterRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk", "time", "area", "perimeter", "r_eff"])
        for row in rows:
            writer.writerow([
                row.chunk,
                f"{row.time:.6f}",
                row.area,
                row.perimeter,
                f"{row.r_eff:.6f}",
            ])


def plot_cluster_series(csv_path: str, png_path: str, lattice_size: int) -> None:
    times: list[float] = []
    r_eff: list[float] = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            r_eff.append(float(row["r_eff"]))

    r_max = lattice_size / np.sqrt(np.pi)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, r_eff, label="$R_\\mathrm{eff}$ (largest bonding cluster)", color="#d73027")
    ax.axhline(r_max, color="#888888", linestyle="--", label="$L/\\sqrt{\\pi}$ (box-filling)")
    ax.axhline(0, color="#888888", linestyle=":")
    ax.set_xlabel("KMC time")
    ax.set_ylabel("$R_\\mathrm{eff}$ (lattice units)")
    ax.set_title("Effective droplet radius vs time")
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output: far-field series
# ---------------------------------------------------------------------------

def write_farfield_csv(path: str, rows: list[FarFieldRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["chunk", "time", "rho_b_far", "rho_i_far"])
        for row in rows:
            writer.writerow([
                row.chunk,
                f"{row.time:.6f}",
                f"{row.rho_b_far:.6f}",
                f"{row.rho_i_far:.6f}",
            ])


def plot_farfield_series(csv_path: str, png_path: str) -> None:
    times: list[float] = []
    rho_b_far: list[float] = []
    rho_i_far: list[float] = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time"]))
            rho_b_far.append(float(row["rho_b_far"]))
            rho_i_far.append(float(row["rho_i_far"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, rho_b_far, label="$\\rho^v_B$ (bonding, border)", color="#d73027")
    ax.plot(times, rho_i_far, label="$\\rho^v_I$ (inert, border)", color="#2166ac")
    ax.set_xlabel("KMC time")
    ax.set_ylabel("Far-field density")
    ax.set_title("Far-field (border) densities vs time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output: params.json
# ---------------------------------------------------------------------------

def save_params_json(path: str, params: RunParams, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "beta": BETA,
        "eta": ETA,
        **asdict(params),
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Chunked simulation
# ---------------------------------------------------------------------------

def _simulate_phase(
    state: np.ndarray,
    params: RunParams,
    duration: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Run one equilibration or production chunk. Returns (final_state, kmc_time)."""
    chain = HeteroNVTDrivenChain(
        beta=BETA,
        bond_energy=params.bond_energy,
        delta_f=params.delta_f,
        delta_mu=params.delta_mu,
        eta=ETA,
        diffusion_lamda=params.diffusion_lamda,
        scheme=params.scheme,
    )
    boundary = Periodic()
    ending = [Time(duration)]

    tmp_dir = tempfile.mkdtemp(prefix="lattice_gas_")
    try:
        simulate(state, boundary, chain, [], ending, seed, tmp_dir)
        final_state = load.final_state(tmp_dir)
        kmc_time = load.final_time(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir)

    return final_state, kmc_time


def run_chunked_simulation(
    params: RunParams,
    initial_state: np.ndarray,
    run_dir: str,
    *,
    label: str = "run",
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Full equilibration + production protocol. Writes all artifacts under run_dir.
    """
    os.makedirs(run_dir, exist_ok=True)
    n_before = int(np.count_nonzero(initial_state != EMPTY))

    # --- equilibration ---
    print(f"  Equilibrating for t={params.equilibration_time} ...")
    eq_state, eq_time = _simulate_phase(
        initial_state,
        params,
        params.equilibration_time,
        params.seed,
    )
    total_time = eq_time

    eq_npy = os.path.join(run_dir, "equilibrated.npy")
    eq_png = os.path.join(run_dir, "equilibrated.png")
    np.save(eq_npy, eq_state)
    save_lattice_png(eq_state, eq_png)

    density_rows: list[DensityRow] = []
    rb, ri, re = lattice_densities(eq_state)
    density_rows.append(DensityRow(chunk=0, time=total_time, rho_bonding=rb, rho_inert=ri, rho_empty=re))

    cluster_rows: list[ClusterRow] = []
    area, perimeter, r_eff = largest_cluster_stats(eq_state)
    cluster_rows.append(ClusterRow(chunk=0, time=total_time, area=area, perimeter=perimeter, r_eff=r_eff))

    farfield_rows: list[FarFieldRow] = []
    rho_b_far, rho_i_far = far_field_densities(eq_state)
    farfield_rows.append(FarFieldRow(chunk=0, time=total_time, rho_b_far=rho_b_far, rho_i_far=rho_i_far))

    # --- production chunks ---
    current_state = eq_state
    print(f"  Production: {params.num_chunks} chunks × {params.chunk_time} ...")
    for chunk_idx in range(params.num_chunks):
        chunk_seed = params.seed + chunk_idx + 1
        current_state, chunk_time = _simulate_phase(
            current_state,
            params,
            params.chunk_time,
            chunk_seed,
        )
        total_time += chunk_time
        rb, ri, re = lattice_densities(current_state)
        density_rows.append(
            DensityRow(
                chunk=chunk_idx + 1,
                time=total_time,
                rho_bonding=rb,
                rho_inert=ri,
                rho_empty=re,
            )
        )
        area, perimeter, r_eff = largest_cluster_stats(current_state)
        cluster_rows.append(
            ClusterRow(
                chunk=chunk_idx + 1,
                time=total_time,
                area=area,
                perimeter=perimeter,
                r_eff=r_eff,
            )
        )
        rho_b_far, rho_i_far = far_field_densities(current_state)
        farfield_rows.append(
            FarFieldRow(
                chunk=chunk_idx + 1,
                time=total_time,
                rho_b_far=rho_b_far,
                rho_i_far=rho_i_far,
            )
        )
        if chunk_idx % max(1, params.num_chunks // 10) == 0:
            print(f"    chunk {chunk_idx + 1}/{params.num_chunks}, t={total_time:.2f}")

    # --- save outputs ---
    final_npy = os.path.join(run_dir, "final_state.npy")
    final_png = os.path.join(run_dir, "final_state.png")
    np.save(final_npy, current_state)
    save_lattice_png(current_state, final_png)

    csv_path = os.path.join(run_dir, "density_series.csv")
    plot_path = os.path.join(run_dir, "density_series.png")
    write_density_csv(csv_path, density_rows)
    plot_density_series(csv_path, plot_path)

    cluster_csv_path = os.path.join(run_dir, "cluster_series.csv")
    cluster_plot_path = os.path.join(run_dir, "cluster_series.png")
    write_cluster_csv(cluster_csv_path, cluster_rows)
    plot_cluster_series(cluster_csv_path, cluster_plot_path, current_state.shape[0])

    farfield_csv_path = os.path.join(run_dir, "farfield_series.csv")
    farfield_plot_path = os.path.join(run_dir, "farfield_series.png")
    write_farfield_csv(farfield_csv_path, farfield_rows)
    plot_farfield_series(farfield_csv_path, farfield_plot_path)

    params_path = os.path.join(run_dir, "params.json")
    save_params_json(
        params_path,
        params,
        extra={
            "label": label,
            "total_kmc_time": total_time,
            "n_particles_before": n_before,
            "n_particles_after": int(np.count_nonzero(current_state != EMPTY)),
        },
    )

    n_after = int(np.count_nonzero(current_state != EMPTY))
    return {
        "run_dir": run_dir,
        "label": label,
        "params_json": params_path,
        "equilibrated_npy": eq_npy,
        "equilibrated_png": eq_png,
        "final_npy": final_npy,
        "final_png": final_png,
        "density_csv": csv_path,
        "density_png": plot_path,
        "cluster_csv": cluster_csv_path,
        "cluster_png": cluster_plot_path,
        "farfield_csv": farfield_csv_path,
        "farfield_png": farfield_plot_path,
        "final_time": total_time,
        "n_before": n_before,
        "n_after": n_after,
        "params": params,
    }


# Backward-compatible alias used by older callers.
def run_once(params: RunParams, initial_state: np.ndarray, run_dir: str, **kwargs) -> dict:
    return run_chunked_simulation(params, initial_state, run_dir, **kwargs)