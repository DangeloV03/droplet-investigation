#!/bin/bash
# Submit poster simulation jobs for both L=256 and L=128 to Slurm.
#
# Usage:
#   ./scripts/submit_poster.sh                         # delta_mu_drive=3.0
#   ./scripts/submit_poster.sh --delta-mu-drive 5.0
#   ./scripts/submit_poster.sh --dry-run               # preview batch scripts

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse optional flags
DM_ARGS=()
DRY_RUN=()
while [[ $# -gt 0 ]]; do
  case "${1}" in
    --delta-mu-drive)
      DM_ARGS=(--delta-mu-drive "${2:?Missing value for --delta-mu-drive}")
      shift 2
      ;;
    --dry-run)
      DRY_RUN=(--dry-run)
      shift
      ;;
    *)
      echo "Unknown argument: ${1}" >&2
      echo "Usage: $0 [--delta-mu-drive VALUE] [--dry-run]" >&2
      exit 1
      ;;
  esac
done

cd "${REPO_ROOT}"

for L in 128; do
  echo "=== Submitting L=${L} poster job ==="
  python poster_run.py \
    --lattice-size "${L}" \
    "${DM_ARGS[@]}" \
    --slurm \
    "${DRY_RUN[@]}"
done

if [[ ${#DRY_RUN[@]} -eq 0 ]]; then
  echo
  echo "=== Both jobs submitted. Monitor with: squeue -u \$USER ==="
fi
