#!/bin/bash
#
# Supervised launcher for the combined live-trading server.
#
# WHY: on 2026-06-26 the server was OOM-killed at 09:36 ET (hit the 16GB WSL
# cap) and, because nothing restarts it, it stayed dead all day — the afternoon
# Meta/HTF/Momentum loops never fired and 13 paper positions went unmanaged.
# This wrapper makes the server self-heal:
#
#   1) Memory-friendly glibc allocator env (must be set BEFORE Python starts):
#        MALLOC_ARENA_MAX  — stop glibc spawning one arena per core (RSS bloat).
#        MALLOC_TRIM_THRESHOLD_ — return freed memory to the OS sooner.
#      (combined_server also runs an in-process gc+malloc_trim keeper every 5m.)
#   2) Restart loop with crash-loop backoff — an OOM-kill (exit 137) or any other
#      exit relaunches the server; --start-all re-arms every dashboard on boot.
#   3) A heartbeat watchdog runs alongside and alerts if no live audit file is
#      written during RTH (catches a hung-but-alive process the restart can't).
#
# Usage:
#   scripts/run_live_server.sh                 # paper, trading sessions auto-start
#   LIVE=1 scripts/run_live_server.sh          # route swing/meta/htf/mom to LIVE
#   scripts/run_live_server.sh --no-start-all  # serve pages but don't auto-run
#   scripts/run_live_server.sh --readiness-on-start  # explicitly run guarded cache readiness
#   DATA_READINESS_TIME=22:15 scripts/run_live_server.sh  # evening full shared-data refresh
#   NIGHTLY_TIME=16:45 scripts/run_live_server.sh  # post-refresher collection/enrichment
#   DEALER_RANKER_TIME=15:40 scripts/run_live_server.sh  # override near-close dealer run
#   (any extra args are passed straight through to combined_server)
#
# Stop: Ctrl-C (stops the watchdog and the server together), or kill this script.
# Logs: logs/live_server/server_YYYYMMDD.log  and  .../watchdog_YYYYMMDD.log
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
LOG_DIR="$REPO_ROOT/logs/live_server"
mkdir -p "$LOG_DIR"
DATESTAMP="$(date +%Y%m%d)"
SERVER_LOG="$LOG_DIR/server_${DATESTAMP}.log"
WATCHDOG_LOG="$LOG_DIR/watchdog_${DATESTAMP}.log"

# --- glibc allocator tuning (read once at process start; must be exported here) ---
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"
export PYTHONUNBUFFERED=1

# --- build the combined_server arg list ---------------------------------------
# `--no-start-all` is OUR sentinel (combined_server has no such flag): consume it
# here and keep everything else as passthrough.
START_ALL=1
PASSTHRU=()
for arg in "$@"; do
  if [ "$arg" = "--no-start-all" ]; then START_ALL=0; else PASSTHRU+=("$arg"); fi
done

SERVER_ARGS=()
# Auto-arm continuous dashboards on boot so a restart resumes trading sessions,
# not just serving. The hub intentionally skips one-shot Meta/Momentum manual
# loops; their scheduled 4H passes fire later from combined_server.
if [ "$START_ALL" = "1" ]; then SERVER_ARGS+=("--start-all"); fi
# The structure engine is paper-only and part of the normal supervised live
# stack. Keep the explicit flag here as a belt-and-suspenders guard even though
# combined_server also defaults it on; a caller can still pass
# --no-intraday-structure to opt out.
SERVER_ARGS+=("--intraday-structure")
# Run the dealer-ranked ATM options experiment automatically near the close.
# Defaults stay paper; caller passthrough args later in SERVER_ARGS can override.
SERVER_ARGS+=(
  # Readiness runs in the EVENING, after nightly_market_data.sh has released the
  # shared heavy-job lock (nightly starts 16:45 ET and has been finishing ~21:45).
  # It was at 05:30 ET, which put the one job that authorizes every 4H entry into
  # the few hours where a crash cannot be recovered from: on 2026-07-24 a WSL2
  # crash at 06:07 killed it mid-run, the server came back at 08:18 with the
  # 05:30 slot already past, and the stale stamp blocked all nine planned entries
  # for the whole session. An evening stamp is generated after the session it
  # needs to cover, and the startup catch-up below is the second line of defence.
  # The workflow itself is still guarded and only stamps after every required
  # bars/context/features/matrix stage succeeds, so stale data stays fail-closed.
  "--data-readiness-time" "${DATA_READINESS_TIME:-22:15}"
  "--nightly-time" "${NIGHTLY_TIME:-16:45}"
  "--dealer-ranker-time" "${DEALER_RANKER_TIME:-15:45}"
  "--dealer-ranker-workers" "${DEALER_RANKER_WORKERS:-8}"
  "--dealer-ranker-top-k" "${DEALER_RANKER_TOP_K:-10}"
  "--dealer-ranker-target-notional" "${DEALER_RANKER_TARGET_NOTIONAL:-5000}"
)
# LIVE=1 flips the scheduled loops onto the real-money account.
if [ "${LIVE:-0}" = "1" ]; then
  SERVER_ARGS+=("--meta-ranker-live" "--htf-live" "--momentum-live")
  if [ -n "${LIVE_ENV:-}" ]; then SERVER_ARGS+=("--live-env" "$LIVE_ENV"); fi
fi
# Dealer Ranker live routing is intentionally separate because it is a new
# experiment with option orders. Set only after an explicit live-trading decision.
if [ "${DEALER_RANKER_LIVE:-0}" = "1" ]; then
  SERVER_ARGS+=("--dealer-ranker-live")
fi
# Pass through anything else the caller added (e.g. --env, ports).
SERVER_ARGS+=("${PASSTHRU[@]}")

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [supervisor] $*" | tee -a "$SERVER_LOG"; }

# --- heartbeat watchdog (best-effort; never blocks the server) ----------------
WATCHDOG_PID=""
if [ -f "$REPO_ROOT/scripts/heartbeat_watchdog.py" ]; then
  nohup "$PYTHON" -u scripts/heartbeat_watchdog.py >> "$WATCHDOG_LOG" 2>&1 &
  WATCHDOG_PID=$!
  log "heartbeat watchdog started (PID $WATCHDOG_PID, log $WATCHDOG_LOG)"
fi

# --- WSL crash forensics logger (streams guest dmesg + memory to NTFS) ---------
# The recurring death is a WSL2 *VM* crash that leaves no host-side trace and
# wipes the guest dmesg on restart. This persists the kernel log to /mnt/c so the
# NEXT crash's cause (OOM? panic?) survives. Best-effort; never blocks the server.
CRASHLOG_PID=""
if [ -f "$REPO_ROOT/scripts/wsl_crash_logger.sh" ]; then
  nohup bash scripts/wsl_crash_logger.sh >/dev/null 2>&1 &
  CRASHLOG_PID=$!
  log "WSL crash logger started (PID $CRASHLOG_PID -> C:\\Users\\<you>\\wsl_crashlog)"
fi

cleanup() {
  log "shutdown requested — stopping server and watchdog"
  [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null
  [ -n "${TAIL_PID:-}" ] && kill "$TAIL_PID" 2>/dev/null
  [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null
  [ -n "$CRASHLOG_PID" ] && kill "$CRASHLOG_PID" 2>/dev/null
  pkill -f 'dmesg.*--follow' 2>/dev/null
  exit 0
}
trap cleanup INT TERM

log "supervised launch: MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX MALLOC_TRIM_THRESHOLD_=$MALLOC_TRIM_THRESHOLD_"
log "combined_server args: ${SERVER_ARGS[*]:-(none)}"

# --- restart loop with crash-loop backoff -------------------------------------
fast_fails=0
while true; do
  start_ts=$(date +%s)
  log "starting combined_server (fast_fails=$fast_fails)"
  "$PYTHON" -u -m UI.combined_server "${SERVER_ARGS[@]}" >> "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  # Mirror the server log to this terminal while it runs (set QUIET=1 to skip).
  # `tail --pid` self-exits the moment the server process ends, so the loop
  # never hangs on it.
  TAIL_PID=""
  if [ "${QUIET:-0}" != "1" ]; then
    tail -n 0 -F --pid="$SERVER_PID" "$SERVER_LOG" &
    TAIL_PID=$!
  fi
  wait "$SERVER_PID"
  rc=$?
  [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null
  ran=$(( $(date +%s) - start_ts ))

  if [ "$rc" -eq 0 ]; then
    log "combined_server exited cleanly (rc=0) — not restarting"
    break
  fi
  if [ "$rc" -eq 137 ]; then
    log "combined_server KILLED (rc=137, likely OOM after ${ran}s) — restarting"
  else
    log "combined_server exited rc=$rc after ${ran}s — restarting"
  fi

  # Backoff: if it died within 60s it's crash-looping; otherwise reset.
  if [ "$ran" -lt 60 ]; then
    fast_fails=$((fast_fails + 1))
  else
    fast_fails=0
  fi
  if   [ "$fast_fails" -ge 5 ]; then delay=300
  elif [ "$fast_fails" -ge 3 ]; then delay=60
  elif [ "$fast_fails" -ge 1 ]; then delay=15
  else delay=5
  fi
  log "restarting in ${delay}s"
  sleep "$delay"
done

cleanup
