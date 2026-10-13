#!/usr/bin/env bash
# TartanDrive 1.0: download → extract → build → delete, one archive at a time.
#
# Why a loop instead of "download everything, then process": the release is 2.06 TB
# compressed and expands to ~2.79 TB, so on a volume with a few TB free the
# archives and the bags cannot both be resident. Processing each
# archive to per-anchor artifacts and deleting its raw data keeps the peak under
# ~0.5 TB while the permanent output grows to ~215 GB.
#
# The next archive downloads *while* the current one is being built. Download is the
# bottleneck (~90 GB at ~20 MB/s = 75 min vs ~7 min of parallel builds), so the
# prefetch hides essentially all of the build time.
#
# Resumable: each archive is recorded in STATE_DIR once its bags are built, and a
# re-run skips completed archives. Safe to Ctrl-C and restart.
#
# Self-restarting: the archive pass runs under a bounded supervisor (see the bottom
# of this file). It does NOT survive the session going away — that is a separate
# failure mode and needs the process detached (e.g. setsid).
#
# Measured: 11-14 MB/s, ~88 min per archive, of which ~5 min is building.
#
# Usage (detach it — this takes ~25 h):
#   setsid tools/tartandrive/td1_fetch_build_loop.sh [--jobs N] [--stage-dir DIR] [--out-root DIR]
#   tail -f <out-root>_state/loop.log
set -uo pipefail

# Base URL of the TartanDrive 1.0 archive store (<ENDPOINT>/<archive>.tar.gz). Obtain it
# from the official TartanDrive 1.0 download instructions and export it before running.
ENDPOINT="${TD1_ENDPOINT:?set TD1_ENDPOINT to the TartanDrive 1.0 archive URL (see README)}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

STAGE_DIR="${DATA_ROOT:-datasets}/tartandrive1_stage"
OUT_ROOT="${DATA_ROOT:-datasets}/tartandrive1_processed"
JOBS=6
MIN_FREE_GB=600     # abort before starting an archive if the volume is this tight
FETCH_ATTEMPTS=8    # each attempt resumes with `curl -C -`
FETCH_BACKOFF_S=60
SUPERVISE_ATTEMPTS=40    # bounded restarts of the whole pass
SUPERVISE_BACKOFF_S=120
SUPERVISE_MAX_STALL=3    # consecutive passes completing no archive -> stop

# 22 of the 23 released archives. 20210910_heightmaps_1 is referenced by NO
# trajectory in the NWM release's TartanDrive split, so it is outside the corpus
# this pipeline is built to match.
ARCHIVES=(
  20210826_heightmaps_1 20210826_heightmaps_2
  20210828_heightmaps_1
  20210829_heightmaps_1 20210829_heightmaps_2 20210829_heightmaps_3
  20210902_heightmaps_1 20210902_heightmaps_2 20210902_heightmaps_3
  20210902_heightmaps_4 20210902_heightmaps_5
  20210903_heightmaps_1 20210903_heightmaps_2 20210903_heightmaps_3
  20210903_heightmaps_4 20210903_heightmaps_5 20210903_heightmaps_6
  20210903_heightmaps_7 20210903_heightmaps_8 20210903_heightmaps_9
  20210903_heightmaps_10 20210903_heightmaps_11
)

while [ $# -gt 0 ]; do
  case "$1" in
    --jobs) JOBS="$2"; shift 2 ;;
    --stage-dir) STAGE_DIR="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# State lives OUTSIDE $OUT_ROOT: tools/features/cache_dino_features_reg.py treats
# every directory under its --data-root as a sequence, so a bookkeeping directory
# in there would be picked up as one.
STATE_ROOT="${OUT_ROOT}_state"
STATE_DIR="$STATE_ROOT/done"
FAILED_LOG="$STATE_ROOT/failed_bags"
mkdir -p "$STAGE_DIR" "$OUT_ROOT" "$STATE_DIR"
touch "$FAILED_LOG"
LOG_FILE="$STATE_ROOT/loop.log"
log() { echo "[$(date -u +%F\ %T)] $*" | tee -a "$LOG_FILE"; }
trap 'rc=$?; reap_prefetch 2>/dev/null; log "loop exiting (rc=$rc) — resume by re-running this script"' EXIT

free_gb() { df -BG --output=avail "$STAGE_DIR" | tail -1 | tr -dc '0-9'; }

# Resumable download, retried by US rather than by curl alone.
#
# curl's own --retry gave up on 20210829_heightmaps_3 after 15.7 of 73.7 GB and
# killed a run that was 5 archives in. Every attempt here resumes with `-C -`, so a
# dropped connection costs one backoff instead of the whole transfer, and the
# server's Content-Length is the completion test — a truncated file is caught here
# rather than surfacing as a tar error 20 GB later.
fetch() {   # fetch <archive>  — leaves <stage>/<archive>.tar.gz
  local a="$1" f="$STAGE_DIR/$1.tar.gz" want have attempt
  want=$(curl -sSI "$ENDPOINT/$a.tar.gz" | tr -d '\r' \
         | awk '/^[Cc]ontent-[Ll]ength:/{print $2}')
  for attempt in $(seq 1 "$FETCH_ATTEMPTS"); do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ -n "$want" ] && [ "$have" = "$want" ]; then
      [ "$attempt" -eq 1 ] && log "$a already fully downloaded"
      return 0
    fi
    if [ "$attempt" -gt 1 ]; then
      log "$a: attempt $attempt/$FETCH_ATTEMPTS, resuming at $have / $want bytes"
      sleep "$FETCH_BACKOFF_S"
    fi
    curl -sS --retry 5 --retry-delay 15 -C - -o "$f" "$ENDPOINT/$a.tar.gz"
  done
  have=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ -n "$want" ] && [ "$have" = "$want" ] && return 0
  log "$a: giving up after $FETCH_ATTEMPTS attempts (have $have, want $want)"
  return 1
}

# Kill a prefetch left running by a previous attempt. Without this, a retry would
# start a second curl writing the same file — and an orphaned prefetch outliving
# its parent is exactly what happened in an earlier run (it ran for another 2 h).
reap_prefetch() {
  if [ -n "${PREFETCH_PID:-}" ]; then
    kill "$PREFETCH_PID" 2>/dev/null
    wait "$PREFETCH_PID" 2>/dev/null
    unset PREFETCH_PID
  fi
}

# One pass over the archive list. Returns 0 when everything is built, 1 on a
# retryable failure, 2 on one that retrying cannot fix.
run_once() {
  reap_prefetch
  local A NEXT i AVAIL N_BAGS N_FAIL

for A in "${ARCHIVES[@]}"; do
  if [ -f "$STATE_DIR/$A" ]; then
    log "$A already built — skip"
    continue
  fi

  AVAIL=$(free_gb)
  if [ "$AVAIL" -lt "$MIN_FREE_GB" ]; then
    log "ABORT: only ${AVAIL} GB free on the stage volume (need ${MIN_FREE_GB})"
    return 2
  fi

  # The prefetch started at the end of the previous iteration; wait for it here.
  # A failed prefetch is NOT fatal — fetch() below resumes it. Treating it as fatal
  # once ended a run at 5 of 22 archives.
  if [ -n "${PREFETCH_PID:-}" ]; then
    log "waiting for the prefetch of $A"
    wait "$PREFETCH_PID" || log "prefetch of $A did not finish — resuming it inline"
    unset PREFETCH_PID
  fi
  # No-op when the prefetch already completed; resumes otherwise.
  fetch "$A" || { log "download of $A failed"; return 1; }

  log "extracting $A"
  rm -rf "${STAGE_DIR:?}/$A"
  if ! tar -xzf "$STAGE_DIR/$A.tar.gz" -C "$STAGE_DIR"; then
    # A tar that fails despite a Content-Length match is corrupt on disk, not
    # short. Drop it so the retry re-downloads rather than looping on the same
    # bad bytes.
    log "extract of $A failed — discarding the archive so the retry re-fetches it"
    rm -f "$STAGE_DIR/$A.tar.gz"
    return 1
  fi
  rm -f "$STAGE_DIR/$A.tar.gz"

  # Start the next download now so it overlaps the builds below.
  NEXT=""
  for ((i = 0; i < ${#ARCHIVES[@]}; i++)); do
    if [ "${ARCHIVES[$i]}" = "$A" ] && [ $((i + 1)) -lt ${#ARCHIVES[@]} ]; then
      NEXT="${ARCHIVES[$((i + 1))]}"
    fi
  done
  if [ -n "$NEXT" ] && [ ! -f "$STATE_DIR/$NEXT" ]; then
    log "prefetching $NEXT in the background"
    fetch "$NEXT" &
    PREFETCH_PID=$!
  fi

  # A per-bag failure must NOT kill a 27 h run: TD1 ships degenerate bags (the
  # probe archives held several of 1-2 MB, far too short to carry an anchor), and
  # those legitimately fail to build. They are recorded and skipped; the count is
  # reported so a systematic failure is still visible.
  log "building bags of $A with $JOBS workers"
  find "$STAGE_DIR/$A" -name '*.bag' -printf '%f\n' \
    | sort \
    | xargs -P "$JOBS" -I {} bash -c '
        python -m tools.tartandrive.build --dataset td1 \
          --raw-root "$1" --seq-name "$2/$3" --out-root "$4" \
          || echo "$2/$3" >> "$5"
      ' _ "$STAGE_DIR" "$A" {} "$OUT_ROOT" "$FAILED_LOG"

  N_BAGS=$(find "$STAGE_DIR/$A" -name '*.bag' | wc -l)
  N_FAIL=$(grep -c "^$A/" "$FAILED_LOG" 2>/dev/null); N_FAIL=${N_FAIL:-0}
  log "$A: $((N_BAGS - N_FAIL))/$N_BAGS bags built ($N_FAIL skipped)"

  rm -rf "${STAGE_DIR:?}/$A"
  touch "$STATE_DIR/$A"
  log "$A DONE  ($(ls "$STATE_DIR" | wc -l)/${#ARCHIVES[@]} archives, $(free_gb) GB free)"
done
  return 0
}


# Supervisor. A bare `while true; do ...; done` would spin on an unfixable error
# and hammer the endpoint, so restarts are bounded three ways: a hard failure
# (rc=2, e.g. the volume filling) is not retried at all; attempts are capped; and
# a run that completes NO new archive counts as a stall, three of which stop the
# supervisor. Progress is measured in completed archives, not in exit codes,
# because a partially successful pass is still progress worth continuing from.
#
# NOTE this does not protect against the session going away — the supervisor is a
# loop inside the same shell. Detach the whole thing (e.g. setsid) for that.
completed() { ls "$STATE_DIR" | wc -l; }

attempt=0
stall=0
while :; do
  attempt=$((attempt + 1))
  before=$(completed)
  run_once
  rc=$?
  after=$(completed)

  if [ "$after" -ge "${#ARCHIVES[@]}" ]; then
    log "all ${#ARCHIVES[@]} archives built into $OUT_ROOT"
    log "sequences built: $(find "$OUT_ROOT" -maxdepth 2 -name meta.json | wc -l)"
    log "bags skipped:    $(wc -l < "$FAILED_LOG")  (see $FAILED_LOG)"
    exit 0
  fi
  if [ "$rc" -eq 2 ]; then
    log "attempt $attempt hit a non-retryable failure (rc=2) — stopping"
    exit 2
  fi
  if [ "$after" -eq "$before" ]; then
    stall=$((stall + 1))
    log "attempt $attempt made no progress (stall $stall/$SUPERVISE_MAX_STALL)"
  else
    stall=0
  fi
  if [ "$stall" -ge "$SUPERVISE_MAX_STALL" ]; then
    log "stopping: $SUPERVISE_MAX_STALL consecutive attempts completed no archive"
    exit 1
  fi
  if [ "$attempt" -ge "$SUPERVISE_ATTEMPTS" ]; then
    log "stopping: $SUPERVISE_ATTEMPTS attempts used, $after/${#ARCHIVES[@]} archives built"
    exit 1
  fi
  log "attempt $attempt ended rc=$rc at $after/${#ARCHIVES[@]} archives — "\
"restarting in ${SUPERVISE_BACKOFF_S}s"
  sleep "$SUPERVISE_BACKOFF_S"
done
