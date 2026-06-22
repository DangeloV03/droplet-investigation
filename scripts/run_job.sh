#!/bin/bash
# Worker script executed on a compute node (no #SBATCH lines here).
# Invoked by submit_one.sh via: sbatch [flags] scripts/run_job.sh samples/foo.json

set -euo pipefail

JOB_JSON="${1:?Usage: run_job.sh samples/your_job.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
