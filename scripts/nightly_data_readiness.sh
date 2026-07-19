#!/bin/bash
#
# Nightly data-readiness for the shared-universe swing stack.
#
# Guarantees that the THREE shared-universe modules — HTF Swing, Momentum
# Expansion, and Meta Ranker — wake up every trading day caught up to the exact
# same state. They all read the SAME data:
#
#   Data/shared/universe/shared_universe.csv          (the shared universe)
#   Data/shared/bars/{1h,4h,1d}/{TICKER}.parquet      (the shared bars)
#
# so a single refresh keeps all three in lock-step. This job does the heavy work
# that must NOT collide with the live session or the 15:50 Meta Ranker MOC loop,
# so schedule it off-hours (default: pre-open via the combined server).
#
# Entry-critical steps:
#   1) Catch up shared bars for the WHOLE universe (1H/4H/1D), with 429 backoff.
#   2) Refresh shared context bars.
#   3) Rebuild the 4H feature matrix (features_4h.parquet) the HTF Swing dashboard
#      scores off — this is what was going 21 days stale.
#   4) Append the fresh bars to the Meta Ranker rolling matrix.
#
# Daily enrichment (earnings/news/etc.) is intentionally NOT in this critical
# path.  It is bounded and owned by nightly_market_data.sh; a flaky vendor must
# not prevent fresh bars/features/matrix from authorizing new entries.
#
# Usage (manual):  bash scripts/nightly_data_readiness.sh
# Logs:            signals/news/data/processed/data_readiness.log
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
LOG_FILE="$REPO_ROOT/signals/news/data/processed/data_readiness.log"
export PYTHONPATH="$REPO_ROOT"

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

mkdir -p "$(dirname "$LOG_FILE")" "$REPO_ROOT/Data/runtime"

# Never let this full-universe readiness job collide with live startup/open.
# It belongs overnight/pre-open; the live server now has a lighter afternoon
# bars+matrix refresher for 4H decision times.
HHMM="$(TZ=America/New_York date +%H%M)"
DOW="$(TZ=America/New_York date +%u)"
if [ "${ALLOW_LIVE_READINESS:-0}" != "1" ] && [ "$DOW" -le 5 ] && [ "$HHMM" -ge 0745 ] && [ "$HHMM" -lt 1640 ]; then
  {
    echo ""
    echo "[$(ts)] nightly_data_readiness.sh skipped: blocked during live window (${HHMM} ET)"
  } >> "$LOG_FILE" 2>&1
  exit 76
fi

# Idempotency: an off-hours repair completed after the latest session already
# satisfies the morning schedule. Avoid rebuilding a 4 GB feature cache twice.
if [ "${FORCE_DATA_READINESS:-0}" != "1" ] && "$PYTHON" -m core.live_readiness >/dev/null 2>&1; then
  {
    echo ""
    echo "[$(ts)] nightly_data_readiness.sh skipped: current session is already ready"
  } >> "$LOG_FILE" 2>&1
  exit 0
fi

LOCK_FILE="$REPO_ROOT/Data/runtime/live_data_jobs.lock"
OWNS_LOCK=0
clear_owned_lock() {
  if [ "$OWNS_LOCK" = "1" ]; then
    : > "$LOCK_FILE"
  fi
}
trap clear_owned_lock EXIT
if command -v flock >/dev/null 2>&1; then
  # Append mode matters: a losing contender must not truncate the current
  # owner's metadata.  The combined-server wrapper already owns this same lock;
  # detect that exact parent and rely on its lock instead of self-deadlocking.
  exec 9>>"$LOCK_FILE"
  if ! flock -n 9; then
    if grep -q "^combined-server-data-readiness pid=${PPID} " "$LOCK_FILE" 2>/dev/null; then
      PARENT_OWNS_LOCK=1
    else
      {
        echo ""
        echo "[$(ts)] nightly_data_readiness.sh skipped: another heavy data job is already running"
      } >> "$LOG_FILE" 2>&1
      exit 75
    fi
  else
    OWNS_LOCK=1
    printf 'nightly_data_readiness pid=%s started=%s\n' "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_FILE"
  fi
fi

run_timed() {
  local label="$1"
  local seconds="$2"
  shift 2
  echo "[$(ts)] $label (timeout=${seconds}s)"
  timeout --signal=TERM --kill-after=60s "${seconds}s" "$@"
  local rc=$?
  echo "[$(ts)] $label exit=$rc"
  return "$rc"
}

{
  echo ""
  echo "================================================================"
  echo "[$(ts)] nightly_data_readiness.sh starting"
  echo "================================================================"

  STATUS=0

  run_timed "1/4 catch up shared bars (1H/4H/1D, full universe, 429 backoff)" \
    "${READINESS_BARS_TIMEOUT_SECONDS:-7200}" \
    "$PYTHON" -u scripts/catchup_shared_bars.py --workers 4
  STATUS=$?

  if [ "$STATUS" -eq 0 ]; then
    run_timed "2/4 refresh shared context bars (including VIXY)" \
      "${READINESS_CONTEXT_TIMEOUT_SECONDS:-1800}" \
      "$PYTHON" -u scripts/refresh_shared_context_bars.py
    STATUS=$?
  fi

  if [ "$STATUS" -eq 0 ]; then
    echo "[$(ts)] 3/4 rebuild HTF 4H feature matrix (features_4h.parquet)"
  # --force is REQUIRED: without it build_all_features_4h() skips when the
  # combined parquet (and per-ticker parquets) already exist, so the matrix
  # froze and HTF Swing scored a 3-week-old bar. Forcing rebuilds both the
  # per-ticker features and the combined matrix off the freshly caught-up bars.
    run_timed "3/4 build-features" "${READINESS_FEATURES_TIMEOUT_SECONDS:-7200}" \
      "$PYTHON" -u -m strategies.momentum_expansion.main --build-features --force
    STATUS=$?
  fi

  if [ "$STATUS" -eq 0 ]; then
    run_timed "4/4 append fresh bars to the Meta Ranker matrix" \
      "${READINESS_MATRIX_TIMEOUT_SECONDS:-2700}" \
      "$PYTHON" -u signals/meta_context/meta_ranker/update_meta_matrix.py
    STATUS=$?
  fi

  if [ "$STATUS" -eq 0 ]; then
    "$PYTHON" -m core.live_readiness --write-success --job nightly_data_readiness
    echo "[$(ts)] readiness stamp updated"
  else
    echo "[$(ts)] readiness aborted; stamp NOT updated (status=$STATUS)"
  fi
  echo "[$(ts)] nightly_data_readiness.sh complete status=$STATUS"
} >> "$LOG_FILE" 2>&1

exit "$STATUS"
