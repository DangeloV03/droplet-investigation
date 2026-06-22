#!/bin/bash
# Expand a master sweep JSON into samples/ and submit one Slurm job per point.
#
# Usage:
#   ./scripts/submit_sweep.sh delta_f_sweep.json
#   ./scripts/submit_sweep.sh              # defaults to delta_f_sweep.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MASTER="${1:-delta_f_sweep.json}"

cd "${REPO_ROOT}"

if [[ ! -f "${SCRIPT_DIR}/slurm.env" ]]; then
  echo "Missing ${SCRIPT_DIR}/slurm.env — copy scripts/slurm.env.example and edit." >&2
  exit 1
fi

echo "=== Writing job JSON files from ${MASTER} ==="
python json_runner.py "${MASTER}" --write-samples-only

echo
echo "=== Submitting Slurm jobs ==="
shopt -s nullglob
jobs=(samples/*.json)
if (( ${#jobs[@]} == 0 )); then
  echo "No job JSON files found in samples/" >&2
  exit 1
fi

for job_json in "${jobs[@]}"; do
  "${SCRIPT_DIR}/submit_one.sh" "${job_json}"
done

echo
echo "=== Submitted ${#jobs[@]} jobs. Monitor with: squeue -u \$USER ==="
