#!/bin/bash
# Generate Yongick geometry .npy files, write samples, and submit runs.
#
# Usage:
#   ./scripts/submit_yongick_sweep.sh
#   ./scripts/submit_yongick_sweep.sh yongick_64_sweep.json
#   ./scripts/submit_yongick_sweep.sh --force yongick_64_sweep.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FORCE=()
MASTER="${REPO_ROOT}/yongick_geometry_sweep.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=(--force)
      shift
      ;;
    *)
      if [[ "$1" == /* ]]; then
        MASTER="$1"
      else
        MASTER="${REPO_ROOT}/$1"
      fi
      shift
      ;;
  esac
done

CONFIG_STEM="$(basename "${MASTER}" .json)"
SAMPLES_DIR="${REPO_ROOT}/samples_${CONFIG_STEM}"
SAMPLES_ARG="samples_${CONFIG_STEM}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${SCRIPT_DIR}/slurm.env" ]]; then
  echo "Missing ${SCRIPT_DIR}/slurm.env — copy scripts/slurm.env.example and edit." >&2
  exit 1
fi

echo "=== Generating Yongick initial geometries from ${CONFIG_STEM} ==="
python "${SCRIPT_DIR}/generate_yongick_geometries.py" --config "${MASTER}"

echo
echo "=== Writing job JSON files from ${CONFIG_STEM} ==="
mkdir -p "${SAMPLES_DIR}"
rm -f "${SAMPLES_DIR}"/*.json
python json_runner.py "${MASTER}" --write-samples-only --samples-dir "${SAMPLES_ARG}"

echo
echo "=== Submitting Slurm jobs ==="
shopt -s nullglob
jobs=("${SAMPLES_DIR}"/*.json)
if (( ${#jobs[@]} == 0 )); then
  echo "No job JSON files found in ${SAMPLES_ARG}/" >&2
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
echo "=== Submitted ${SUBMITTED} job(s), skipped ${SKIPPED} duplicate(s) (${CONFIG_STEM}) ==="
