#!/bin/bash
# Submit the droplet-dissolve job to Slurm.
#
# Drives the pre-equilibrated droplet (dissolve_initial_L128.npy) with
# delta_mu = 1.0 for 120 snapshot chunks so it dissolves slowly.
#
# Usage:
#   ./scripts/submit_dissolve.sh                        # delta_mu=1.0, 120 chunks
#   ./scripts/submit_dissolve.sh --delta-mu-drive 2.0
#   ./scripts/submit_dissolve.sh --num-chunks 200
#   ./scripts/submit_dissolve.sh --time-minutes 5760    # 4-day walltime
#   ./scripts/submit_dissolve.sh --output-dir dissolve_run
#   ./scripts/submit_dissolve.sh --dry-run              # preview batch script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DM_ARGS=()
CHUNK_ARGS=()
TIME_ARGS=()
OUT_ARGS=()
DRY_RUN=()
while [[ $# -gt 0 ]]; do
  case "${1}" in
    --delta-mu-drive)
      DM_ARGS=(--delta-mu-drive "${2:?Missing value for --delta-mu-drive}")
      shift 2
      ;;
    --num-chunks)
      CHUNK_ARGS=(--num-chunks "${2:?Missing value for --num-chunks}")
      shift 2
      ;;
    --time-minutes)
      TIME_ARGS=(--time-minutes "${2:?Missing value for --time-minutes}")
      shift 2
      ;;
    --output-dir)
      OUT_ARGS=(--output-dir "${2:?Missing value for --output-dir}")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=(--dry-run)
      shift
      ;;
    *)
      echo "Unknown argument: ${1}" >&2
      echo "Usage: $0 [--delta-mu-drive V] [--num-chunks N] [--time-minutes N] [--output-dir DIR] [--dry-run]" >&2
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

echo "=== Submitting droplet-dissolve job ==="
python dissolve_run.py \
  "${DM_ARGS[@]}" \
  "${CHUNK_ARGS[@]}" \
  "${TIME_ARGS[@]}" \
  "${OUT_ARGS[@]}" \
  --slurm \
  "${DRY_RUN[@]}"

if [[ ${#DRY_RUN[@]} -eq 0 ]]; then
  echo
  echo "=== Job submitted. Monitor with: squeue -u \$USER ==="
fi
