# Guide to Connecting and Running the Rust lattice_gas Module

This document is designed for developers or AI agents working in this repository to explain the integration between the Rust-based simulation library `lattice_gas` and the Python/Rust application `droplet-investigation`.

---

## 1. System Architecture

The simulation is split into two complementary projects:

1. **`lattice-gas` (Rust Crate, located at `../lattice-gas`):**
   - Implements the core Kinetic Monte Carlo (KMC) engine on a periodic lattice.
   - Exposes simulation structures and procedures to Python using **PyO3** and the **numpy** Rust bindings.
   - Exposes a native Rust API for inclusion in other Rust programs.

2. **`droplet-investigation` (Rust & Python Application, located here):**
   - **Rust Binaries (`src/test1.rs` and `src/test2.rs`):** Compile directly against the Rust crate `lattice_gas` as a Cargo dependency. Used for validation, diffusion MSD tracking, and quick performance testing.
   - **Python Engine (`simulation.py`, `cli.py`, `main.py`, `json_runner.py`):** Import the compiled Python bindings (`lattice_gas`) to initialize droplet geometry, execute runs in production/equilibration chunks, analyze cluster properties (effective radius, far-field densities), and plot the final visual outputs.

---

## 2. Setting Up the Environment

### Prerequisites

Ensure you have the following installed on your machine:
- **Rust toolchain** (installed via [rustup](https://rustup.rs/))
- **Python 3.8+**
- **pip** and Python packages: `numpy`, `scipy`, `matplotlib`, and `maturin`

You can install the Python prerequisites using:
```bash
pip install numpy scipy matplotlib maturin
```

---

## 3. Compiling the Rust `lattice_gas` Python Module

To compile the Rust engine into a Python extension module that Python scripts can import:

1. Run the compilation script inside `lattice-gas`:
   ```bash
   cd ../lattice-gas
   ./build-rust-lib.sh
   ```
2. Under the hood, this script runs the following **Maturin** command:
   ```bash
   maturin develop --release -m "./Cargo.toml" --features "extension-module"
   ```
   This compiles the Rust package and registers the `lattice_gas` module in your active Python environment.

### Troubleshooting on macOS
If `cargo test` or library loading fails with `Library not loaded: @rpath/libpython3.11.dylib`, add your Python library directory to the dynamic loader path:
```bash
# Example with Anaconda:
export DYLD_LIBRARY_PATH=/opt/anaconda3/lib:$DYLD_LIBRARY_PATH
```

---

## 4. Python Integration & APIs

Once Maturin installs the module, you can import and call it in Python as follows:

```python
from lattice_gas.markov_chain import HeteroNVTDrivenChain
from lattice_gas.boundary_condition import Periodic
from lattice_gas.ending_criterion import Time
from lattice_gas.simulate import simulate
from lattice_gas import load
```

### Key Python API Components

- **`HeteroNVTDrivenChain(beta, bond_energy, delta_f, delta_mu, eta, diffusion_lamda, scheme)`**
  Initializes the Markov chain state transition parameters:
  - `beta` (float): Temperature parameter $\beta$ (typically fixed at `1.0`).
  - `bond_energy` (float): Active-active particle attraction (e.g. `-2.0`).
  - `delta_f` (float): Internal energy difference between active/inactive states (e.g. `-1.0`).
  - `delta_mu` (float): Nonequilibrium driving parameter (e.g. `0.0`).
  - `eta` (float): Reaction pathway weight (typically fixed at `1.0`).
  - `diffusion_lamda` (float): Diffusion rate constant.
  - `scheme` (str): Drive direction scheme (`"homo"`, `"negative_drive"`, `"positive_drive"`).

- **`Periodic()`**
  Instantiates periodic boundary conditions for the $N \times N$ lattice grid.

- **`Time(duration)`**
  Instantiates the ending criterion based on Kinetic Monte Carlo (KMC) simulation time.

- **`simulate(state, boundary, chain, observs, endings, seed, outdir)`**
  Executes the simulation loop starting from the 2D `ndarray` `state`.
  - Writes intermediate states to a temporary directory (`outdir`).
  
- **`load.final_state(outdir)` & `load.final_time(outdir)`**
  Reads the ending state (2D `uint32` array) and final simulation KMC time from the output directory.

---

## 5. Running Python Simulations

### Option A: Interactive & CLI runs (`main.py`)
Run `main.py` to prompt for run settings or read parameter profiles from a text file:
```bash
python main.py
```
- **Interactive mode:** Interactively asks for grid size, concentration, initial droplet radius, and chain coefficients.
- **Params-file mode:** Prompts for a parameters file (e.g. `example_params.txt`). If a parameter lists multiple values (e.g. `diffusion_lamda=1, 10, 100`), the runner will sweep over those values sequentially.

### Option B: Batch multi-axis sweeps (`json_runner.py`)
Run a batch sweep with a structured JSON config:
```bash
python json_runner.py example_sweep.json
```
- Config structure expects a `"fixed"` object for shared settings, and a `"sweep"` object listing lists of variables (e.g., `diffusion_lamda`, `delta_f`).
- The script automatically expands all combinations (cartesian product) and executes runs in timestamped result folders under `results/`.

---

## 6. Compiling & Running Native Rust Binaries

The `droplet-investigation` repository includes native Rust binaries (`test1` and `test2`) that bypass Python for fast verification runs.

### How to Run:
Ensure you are in the `/Users/d_angel/ReMatch/droplet-investigation` directory and run:

1. **`test1` Binary (Chemical Reactions Validation):**
   Validates equilibrium Boltzmann statistics, nonequilibrium reaction rate balance, and phase separation.
   ```bash
   cargo run --release --bin test1 -- config.toml
   ```
2. **`test2` Binary (Diffusion & Mean Squared Displacement):**
   Calculates single-particle MSD(t) vs time curves and verifies the expected diffusion constant $D = \lambda/4$.
   ```bash
   cargo run --release --bin test2 -- config.toml
   ```

These binaries load settings from a local configuration file such as [config.toml](file:///Users/d_angel/ReMatch/droplet-investigation/config.toml).
