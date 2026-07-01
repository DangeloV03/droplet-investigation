#!/bin/bash
# Daemon: submit the poster simulation and resubmit automatically on timeout
# until all three phases are complete.
#
# Start in a named tmux session:
#   tmux new -s poster
#   ./scripts/poster_daemon.sh            # L=128, delta_mu=3.0
#   ./scripts/poster_daemon.sh 128 5.0   # custom params
#   Ctrl-B, D   to detach (daemon keeps running)
#   tmux attach -t poster   to reattach
#
# One-liner (detached):
#   tmux new -d -s poster "./scripts/poster_daemon.sh" && tmux attach -t poster

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LATTICE_SIZE="${1:-128}"
DM="${2:-3.0}"
CHECK_INTERVAL=300   # seconds between squeue polls

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── completion check ────────────────────────────────────────────────────────
# Returns path of the completed run dir and exits 0, or exits 1 if not done.
is_complete() {
    python3 - "${LATTICE_SIZE}" "${DM}" <<'PYEOF'
import sys
from pathlib import Path
L, dm = sys.argv[1], sys.argv[2]
dm_str = f"{float(dm):.1f}".replace(".", "p").replace("-", "m")
label = f"poster_L{L}_dm{dm_str}"
root = Path("results")
if not root.is_dir():
    sys.exit(1)
matches = sorted(d for d in root.iterdir()
                 if d.is_dir() and d.name.endswith(label))
for d in reversed(matches):
    if (d / "phase_log.json").exists():
        print(d)
        sys.exit(0)
sys.exit(1)
PYEOF
}

# ── progress snapshot ───────────────────────────────────────────────────────
show_progress() {
    python3 - "${LATTICE_SIZE}" "${DM}" <<'PYEOF'
import sys
from pathlib import Path
L, dm = sys.argv[1], sys.argv[2]
dm_str = f"{float(dm):.1f}".replace(".", "p").replace("-", "m")
label = f"poster_L{L}_dm{dm_str}"
root = Path("results")
if not root.is_dir():
    print("  No results directory yet")
    sys.exit(0)
matches = sorted(d for d in root.iterdir()
                 if d.is_dir() and d.name.endswith(label))
if not matches:
    print("  No run directory found yet")
    sys.exit(0)
d = matches[-1]
print(f"  Run dir: {d}")
for phase, n_total in [("equilibrated", 100), ("drive_on", 50), ("drive_off", 50)]:
    snap_dir = d / "snapshots" / phase
    n = sum(1 for f in snap_dir.glob("state_t*.npy")) if snap_dir.is_dir() else 0
    bar = "#" * (n * 20 // n_total) + "-" * (20 - n * 20 // n_total)
    print(f"  {phase:>13s}: [{bar}] {n:3d}/{n_total}")
PYEOF
}

# ── submit and return job id ────────────────────────────────────────────────
submit_job() {
    local out
    out=$(python poster_run.py \
        --lattice-size "${LATTICE_SIZE}" \
        --delta-mu-drive "${DM}" \
        --slurm 2>&1)
    printf '%s\n' "${out}" >&2
    printf '%s\n' "${out}" \
        | grep -oE 'Submitted batch job [0-9]+' \
        | grep -oE '[0-9]+' \
        || true
}

# ── wait until job is no longer running/pending ─────────────────────────────
# Uses sacct as primary (catches completed jobs squeue drops) with squeue fallback.
job_active() {
    local jid="$1"
    # sacct returns state for completed jobs too; "active" = RUNNING or PENDING
    local state
    state=$(sacct -j "${jid}" --noheader -o State%20 2>/dev/null | head -1 | tr -d ' ')
    if [[ "${state}" == "RUNNING" || "${state}" == "PENDING" || "${state}" == "REQUEUED" ]]; then
        return 0
    fi
    # Fall back to squeue for newly-submitted jobs not yet in sacct
    if squeue -j "${jid}" -h 2>/dev/null | grep -q .; then
        return 0
    fi
    return 1
}

wait_for_job() {
    local jid="$1"
    sleep 30   # give sbatch time to register
    while job_active "${jid}"; do
        sleep "${CHECK_INTERVAL}"
    done
}

# ── main loop ───────────────────────────────────────────────────────────────
log "=== Poster daemon started  L=${LATTICE_SIZE}  delta_mu=${DM} ==="

while true; do
    if dir=$(is_complete 2>/dev/null); then
        log "=== COMPLETE: ${dir} ==="
        show_progress
        break
    fi

    show_progress 2>/dev/null || true

    log "Submitting job ..."
    jid=$(submit_job)

    if [[ -z "${jid}" ]]; then
        log "ERROR: sbatch did not return a job ID — check output above"
        exit 1
    fi

    log "Job ${jid} queued — polling every ${CHECK_INTERVAL}s"
    wait_for_job "${jid}"
    log "Job ${jid} finished"
done
