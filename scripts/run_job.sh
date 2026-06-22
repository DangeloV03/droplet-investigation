#!/bin/bash
# Worker script executed on a compute node (no #SBATCH lines here).
# Invoked by submit_one.sh via: sbatch [flags] scripts/run_job.sh samples/foo.json

set -euo pipefail

JOB_JSON="${1:?Usage: run_job.sh samples/your_job.json}"

# Slurm copies this script to /var/spool/slurmd/.../slurm_script, so BASH_SOURCE
# does not point at the repo. submit_one.sh passes DROPLET_REPO_ROOT via --export.
if [[ -n "${DROPLET_REPO_ROOT:-}" ]]; then
  REPO_ROOT="${DROPLET_REPO_ROOT}"
elif [[ -f "$(dirname "${BASH_SOURCE[0]}")/slurm.env" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/slurm.env" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  echo "Cannot locate repo root (missing DROPLET_REPO_ROOT)" >&2
  exit 1
fi

SCRIPT_DIR="${REPO_ROOT}/scripts"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/slurm.env"

cd "${PROJECT_ROOT}"

module load anaconda3/2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lattice
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

echo "=== droplet-investigation job ==="
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Job JSON: ${JOB_JSON}"
echo "Project: ${PROJECT_ROOT}"
echo

python json_runner.py --run-job "${JOB_JSON}"

echo
echo "End: $(date)"
