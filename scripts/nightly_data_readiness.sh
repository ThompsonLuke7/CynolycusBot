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
# Steps:
#   1) Catch up shared bars for the WHOLE universe (1H/4H/1D), with 429 backoff.
#   2) Rebuild the 4H feature matrix (features_4h.parquet) the HTF Swing dashboard
#      scores off — this is what was going 21 days stale.
#   3) Append the fresh bars to the Meta Ranker rolling matrix.
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

if command -v flock >/dev/null 2>&1; then
  exec 9>"$REPO_ROOT/Data/runtime/live_data_jobs.lock"
  if ! flock -n 9; then
    {
      echo ""
      echo "[$(ts)] nightly_data_readiness.sh skipped: another heavy data job is already running"
    } >> "$LOG_FILE" 2>&1
    exit 75
  fi
fi

{
  echo ""
  echo "================================================================"
  echo "[$(ts)] nightly_data_readiness.sh starting"
  echo "================================================================"

  STATUS=0

  echo "[$(ts)] 1/5 catch up shared bars (1H/4H/1D, full universe, 429 backoff)"
  "$PYTHON" -u scripts/catchup_shared_bars.py --workers 4
  rc=$?
  echo "[$(ts)] catchup exit=$rc"
  [ "$rc" -ne 0 ] && STATUS="$rc"

  echo "[$(ts)] 2/5 refresh shared context bars (including VIXY)"
  "$PYTHON" -u scripts/refresh_shared_context_bars.py
  rc=$?
  echo "[$(ts)] context refresh exit=$rc"
  [ "$rc" -ne 0 ] && STATUS="$rc"

  echo "[$(ts)] 3/5 rebuild HTF 4H feature matrix (features_4h.parquet)"
  # --force is REQUIRED: without it build_all_features_4h() skips when the
  # combined parquet (and per-ticker parquets) already exist, so the matrix
  # froze and HTF Swing scored a 3-week-old bar. Forcing rebuilds both the
  # per-ticker features and the combined matrix off the freshly caught-up bars.
  "$PYTHON" -u -m strategies.momentum_expansion.main --build-features --force
  rc=$?
  echo "[$(ts)] build-features exit=$rc"
  [ "$rc" -ne 0 ] && STATUS="$rc"

  echo "[$(ts)] 4/5 heavy daily feeds (earnings-calendar, news-catalyst rescore)"
  # Moved off the per-4H loop (each was ~25-27 min and blew the 60-min feeds
  # budget). They change on a daily cadence, so refresh them once here pre-open.
  "$PYTHON" -u signals/meta_context/meta_ranker/update_feeds.py --daily --timeout 5400
  rc=$?
  echo "[$(ts)] daily feeds exit=$rc"
  [ "$rc" -ne 0 ] && STATUS="$rc"

  echo "[$(ts)] 5/5 append fresh bars to the Meta Ranker matrix"
  "$PYTHON" -u signals/meta_context/meta_ranker/update_meta_matrix.py
  rc=$?
  echo "[$(ts)] meta matrix exit=$rc"
  [ "$rc" -ne 0 ] && STATUS="$rc"

  if [ "$STATUS" -eq 0 ]; then
    "$PYTHON" -m core.live_readiness --write-success --job nightly_data_readiness
    echo "[$(ts)] readiness stamp updated"
  else
    echo "[$(ts)] readiness stamp NOT updated because one or more steps failed (status=$STATUS)"
  fi
  echo "[$(ts)] nightly_data_readiness.sh complete status=$STATUS"
} >> "$LOG_FILE" 2>&1

exit "$STATUS"
