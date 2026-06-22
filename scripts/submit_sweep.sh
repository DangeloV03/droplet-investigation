#!/bin/bash
# Expand a master sweep JSON into samples/ and submit one Slurm job per point.
# Skips parameter points that already have a completed folder under results/.
#
# Usage:
#   ./scripts/submit_sweep.sh delta_f_sweep.json
#   ./scripts/submit_sweep.sh --force delta_f_sweep.json
#   ./scripts/submit_sweep.sh              # defaults to delta_f_sweep.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FORCE=()
if [[ "${1:-}" == "--force" ]]; then
  FORCE=(--force)
  shift
fi
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

SUBMITTED=0
SKIPPED=0
for job_json in "${jobs[@]}"; do
  set +e
  "${SCRIPT_DIR}/submit_one.sh" "${FORCE[@]}" "${job_json}"
  rc=$?
  set -e
  if (( rc == 2 )); then
    SKIPPED=$((SKIPPED + 1))
  elif (( rc == 0 )); then
    SUBMITTED=$((SUBMITTED + 1))
  else
    exit "${rc}"
  fi
done

echo
echo "=== Submitted ${SUBMITTED} job(s), skipped ${SKIPPED} duplicate(s). Monitor with: squeue -u \$USER ==="
