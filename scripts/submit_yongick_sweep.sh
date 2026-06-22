#!/bin/bash
# Generate Yongick geometry .npy files, write samples, and submit four runs.
#
# Usage:
#   ./scripts/submit_yongick_sweep.sh
#   ./scripts/submit_yongick_sweep.sh --force

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MASTER="${REPO_ROOT}/yongick_geometry_sweep.json"
SAMPLES_DIR="${REPO_ROOT}/samples_yongick"

FORCE=()
if [[ "${1:-}" == "--force" ]]; then
  FORCE=(--force)
fi

cd "${REPO_ROOT}"

if [[ ! -f "${SCRIPT_DIR}/slurm.env" ]]; then
  echo "Missing ${SCRIPT_DIR}/slurm.env — copy scripts/slurm.env.example and edit." >&2
  exit 1
fi

echo "=== Generating Yongick initial geometries ==="
python "${SCRIPT_DIR}/generate_yongick_geometries.py"

echo
echo "=== Writing job JSON files from yongick_geometry_sweep.json ==="
mkdir -p "${SAMPLES_DIR}"
rm -f "${SAMPLES_DIR}"/*.json
python json_runner.py "${MASTER}" --write-samples-only --samples-dir samples_yongick

echo
echo "=== Submitting Slurm jobs ==="
shopt -s nullglob
jobs=("${SAMPLES_DIR}"/*.json)
if (( ${#jobs[@]} == 0 )); then
  echo "No job JSON files found in samples_yongick/" >&2
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
echo "=== Submitted ${SUBMITTED} Yongick job(s), skipped ${SKIPPED} duplicate(s) ==="
