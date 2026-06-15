# Droplet Investigation: Kinetic Monte Carlo Simulation

A high-performance simulation platform modeling **nonequilibrium dividing droplet dynamics** in a 2D lattice gas. This repository combines a high-speed stochastic Markov chain engine written in **Rust** with an interactive **Python** harness for running sweeps, executing chunked equilibration-production protocols, and analyzing thermodynamic/morphological statistics.

---

## 🌌 Project Overview

The project simulates particle dynamics on an $N \times N$ periodic lattice. Each lattice site contains one of three states:
- **`0` (Empty)**: Available space.
- **`1` (Inert / Inactive)**: Particles that do not form bonds.
- **`2` (Bonding / Active)**: Particles that attract other bonding particles.

The physics engine is governed by a **Kinetic Monte Carlo (KMC)** Markov Chain (`HeteroNVTDrivenChain`) that handles:
- **Diffusion**: Particles hopping to adjacent empty sites.
- **Interconversions**: Nonequilibrium chemical reactions ($B \leftrightarrow I$) driven by internal energy differences ($\Delta f$) and thermodynamic driving force ($\Delta\mu$).
- **Phase Separation**: Bonding particles form cohesive condensates (droplets) at low temperatures (defined by attraction $\epsilon$).

---

## 🛠️ Repository Architecture

This workspace is part of a two-crate ecosystem:
1. **`lattice-gas` (Rust Crate, located at `../lattice-gas`):** The core Monte Carlo engine, compiled into Python extension bindings using `PyO3` and `Maturin`.
2. **`droplet-investigation` (This repository):** The application layer which includes:
   - **Interactive CLI & Batch Sweeper (`main.py`, `cli.py`):** Runs chunked simulation runs, prompting for options or parsing parameter `.txt` configurations.
   - **Simulation Engine (`simulation.py`):** Drives the equilibration + production chunks protocol, tracks densities, measures cluster boundaries, and writes reports.
   - **Batch Sweep Runner (`json_runner.py`):** Explores multi-dimensional param spaces (Cartesian product) from a JSON config file.
   - **Native Rust Binaries (`src/test1.rs`, `src/test2.rs`):** Compile directly against `lattice_gas` as a cargo dependency for verification and high-throughput validation.

```
droplet-investigation/
├── Cargo.toml                 # Cargo config (binaries for test1/test2)
├── config.toml                 # Settings for Rust verification binaries
├── main.py                     # CLI Entry point (Interactive vs Batch params)
├── cli.py                      # CLI prompts and parameter parsing
├── simulation.py               # Chunked simulation engine & file I/O
├── cluster.py                  # Cluster and far-field analysis functions
├── json_runner.py              # Cartesian sweep runner using JSON files
├── RUST_INTEGRATION_GUIDE.md   # Integration & Python API guide
└── src/
    ├── config.rs               # TOML configuration parser for binaries
    ├── test1.rs                # Rust binary (reaction and phase separation validation)
    └── test2.rs                # Rust binary (diffusion MSD validation)
```

---

## 🚀 Getting Started

### Prerequisites
- **Rust Toolchain** (install via [rustup](https://rustup.rs/))
- **Python 3.8+**
- **Required Python Libraries:**
  ```bash
  pip install numpy scipy matplotlib maturin
  ```

### 1. Build and Link the Rust Core
Compile the Rust library bindings and register them inside your active Python environment:
```bash
cd ../lattice-gas
maturin develop --release --features "extension-module"
```
*(On macOS, if you encounter library loading issues, configure your `DYLD_LIBRARY_PATH` to point to your Python environment's lib folder.)*

### 2. Run Python Simulations
Start the simulation runner to run interactive single runs or parameter sweeps:
```bash
python main.py
```
- **Interactive Mode**: Prompts for lattice parameters, concentrations, seeds, and chain coefficients.
- **Batch Mode**: Reads from a parameter sweep file (e.g. `example_params.txt`) to vary individual variables sequentially.

Alternatively, execute a structured multi-dimensional JSON sweep:
```bash
python json_runner.py example_sweep.json
```
Output results (CSV logs, lattice snapshots, and time-series plots) will be written to timestamped subfolders under `results/`.

### 3. Run Native Rust Verification Binaries
Run validation tests directly in Rust:
```bash
# Chemical Reactions & Phase Separation tests
cargo run --release --bin test1 -- config.toml

# Diffusion Mean Squared Displacement (MSD) tracking
cargo run --release --bin test2 -- config.toml
```

---

## 📊 Outputs & Diagnostics

During simulation execution, the engine divides the run into an initial **equilibration phase** followed by a configurable number of **production chunks**. After each chunk, it records:
1. **Lattice Snapshots (`equilibrated.png`, `final_state.png`)**: Visual representation of the lattice where bonding particles form clusters (white = empty, blue = inert, red = bonding).
2. **Species Densities (`density_series.csv`, `density_series.png`)**: Plots global fraction of sites occupied by $B$, $I$, and Empty species over time.
3. **Droplet Morphology (`cluster_series.csv`, `cluster_series.png`)**: Tracks the area, perimeter, and effective radius $R_\mathrm{eff} = \sqrt{\mathrm{Area}/\pi}$ of the largest active cluster.
4. **Far-Field Boundary Densities (`farfield_series.csv`, `farfield_series.png`)**: Measures concentrations at the lattice boundaries to observe dilution effects.

---

## 🧪 Validation & Verification

### Test 1: Chemical Reaction & Phase Separation Rates (`src/test1.md`)
- **Equilibrium**: Measured $B/I$ ratios at $\Delta\mu=0$ match the Boltzmann factor $e^{-\beta\Delta f}$ with $<1.5\%$ error.
- **Nonequilibrium**: Measured ratios match steady-state rate balances under driving fields.
- **Phase Separation**: Visual phase separation occurs correctly below the critical attraction point $\beta\epsilon_c \approx -1.76$ for the 2D Ising model limit.

### Test 2: Diffusion Mean Squared Displacement (`src/test2.md`)
- Diffusion constants ($D$) measured from single-particle ensemble MSD curves match the input rate parameters ($\lambda/4$) within statistical noise (e.g., error $\approx 1-2\%$ for $\lambda=1$).
