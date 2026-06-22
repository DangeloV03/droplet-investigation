#!/bin/bash
# Submit one simulation job to Slurm.
#
# Usage:
#   ./scripts/submit_one.sh samples/256_negative_drive_df2_dm0_epsm2p95.json

set -euo pipefail

JOB_JSON="${1:?Usage: submit_one.sh samples/your_job.json}"

if [[ ! -f "${JOB_JSON}" ]]; then
  echo "Job JSON not found: ${JOB_JSON}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ ! -f "${SCRIPT_DIR}/slurm.env" ]]; then
  echo "Missing ${SCRIPT_DIR}/slurm.env — copy scripts/slurm.env.example and edit." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/slurm.env"

mkdir -p "${REPORT_DIR}"

JOB_STEM="$(basename "${JOB_JSON}" .json)"
JOB_NAME="${JOB_NAME_PREFIX}_${JOB_STEM}"
JOB_NAME="${JOB_NAME:0:64}"

RUNNER="${REPO_ROOT}/scripts/run_job.sh"
# Resolve job JSON relative to repo root if needed.
if [[ "${JOB_JSON}" != /* ]]; then
  JOB_JSON="${REPO_ROOT}/${JOB_JSON}"
fi

SBATCH_ARGS=(
  --job-name="${JOB_NAME}"
  --partition="${PARTITION}"
  --cpus-per-task="${CPUS_PER_TASK}"
  --mem="${MEM}"
  --time="${TIME_MINUTES}"
  --output="${REPORT_DIR}/%j.out"
  --error="${REPORT_DIR}/%j.err"
)

if [[ -n "${ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account="${ACCOUNT}")
fi

echo "Submitting ${JOB_STEM} ..."
sbatch "${SBATCH_ARGS[@]}" "${RUNNER}" "${JOB_JSON}"
