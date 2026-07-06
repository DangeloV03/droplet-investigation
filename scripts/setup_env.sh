#!/bin/bash
# One-time environment setup for droplet-investigation on Della (or any host
# with conda + the sibling ../lattice-gas crate).
#
# What this does (steps that are NOT handled by the Slurm batch scripts, which
# only *activate* an existing `lattice` env via slurm_config.yml setup_cmds):
#   1. create/activate the `lattice` conda env
#   2. pip install Python deps + maturin
#   3. build the lattice_gas PyO3 extension from ../lattice-gas
#
# After running this once, submit jobs with ./scripts/submit_poster.sh — the
# generated Slurm script re-activates the env on the compute node for you.
#
# Usage:
#   ./scripts/setup_env.sh            # env name: lattice, python 3.11
#   ENV_NAME=myenv PY_VERSION=3.12 ./scripts/setup_env.sh

set -euo pipefail

ENV_NAME="${ENV_NAME:-lattice}"
PY_VERSION="${PY_VERSION:-3.11}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_DIR="${REPO_ROOT}/../lattice-gas"

if [[ ! -f "${CRATE_DIR}/Cargo.toml" ]]; then
  echo "ERROR: sibling crate not found at ${CRATE_DIR}" >&2
  echo "Clone lattice-gas next to droplet-investigation and re-run." >&2
  exit 1
fi

# --- conda ------------------------------------------------------------------
# On Della, load the module first: module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "=== Creating conda env '${ENV_NAME}' (python ${PY_VERSION}) ==="
  conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}"
else
  echo "=== Reusing existing conda env '${ENV_NAME}' ==="
fi

conda activate "${ENV_NAME}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# --- Python deps ------------------------------------------------------------
echo "=== Installing Python dependencies ==="
pip install --upgrade pip
pip install -r "${REPO_ROOT}/requirements.txt"
pip install maturin

# --- build the Rust/Python extension ---------------------------------------
echo "=== Building lattice_gas extension (maturin develop --release) ==="
cd "${CRATE_DIR}"
maturin develop --release --features "extension-module"

echo
echo "=== Done. Env '${ENV_NAME}' is ready. ==="
echo "Next: cd ${REPO_ROOT} && ./scripts/submit_poster.sh --dry-run"
