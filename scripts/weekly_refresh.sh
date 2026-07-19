#!/bin/bash
#
# Weekly (Sunday) refresh for the trading stack — run by hand each weekend.
#
# Does the heavy weekly work the nightly job intentionally skips:
#   1) Catch up shared bars for the WHOLE universe (1H/4H/1D), so names promoted
#      during the week get their full history backfilled before they're scored.
#   2) Rebuild the Momentum Expansion weekly universe snapshot (needs #1 first —
#      it scores off daily bars, so new names without bars would be dropped).
#   3) Refresh Meta Ranker feeds + the dynamic theme taxonomy (Claude API $).
#
# It does NOT (and cannot) re-auth Schwab — that needs an interactive browser
# login — so it prints a reminder + the exact command at the end.
#
# Usage:  bash scripts/weekly_refresh.sh
# Logs:   signals/news/data/processed/weekly_refresh.log
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
LOG_FILE="$REPO_ROOT/signals/news/data/processed/weekly_refresh.log"
LOCK_FILE="$REPO_ROOT/Data/runtime/live_data_jobs.lock"
export PYTHONPATH="$REPO_ROOT"

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

mkdir -p "$(dirname "$LOG_FILE")" "$REPO_ROOT/Data/runtime"
OWNS_LOCK=0
clear_owned_lock() {
  if [ "$OWNS_LOCK" = "1" ]; then
    : > "$LOCK_FILE"
  fi
}
trap clear_owned_lock EXIT
if command -v flock >/dev/null 2>&1; then
  exec 9>>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "[$(ts)] weekly_refresh.sh skipped: another heavy data job is already running" | tee -a "$LOG_FILE"
    exit 75
  fi
  OWNS_LOCK=1
  printf 'weekly_refresh pid=%s started=%s\n' "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_FILE"
fi

{
  echo ""
  echo "================================================================"
  echo "[$(ts)] weekly_refresh.sh starting"
  echo "================================================================"
  STATUS=0

  # 1) Backfill bars for the whole universe (incl. newly promoted names).
  echo "[$(ts)] 1/4 catch up shared bars (1H/4H/1D, full universe)"
  timeout --signal=TERM --kill-after=60s "${WEEKLY_BARS_TIMEOUT_SECONDS:-21600}s" \
    "$PYTHON" -u scripts/catchup_shared_bars.py --workers 6
  rc=$?
  echo "[$(ts)] catchup exit=$rc"
  [ "$rc" -ne 0 ] && STATUS="$rc"

  # 2) Momentum weekly universe snapshot (must follow #1 to include new names).
  echo "[$(ts)] 2/4 momentum universe snapshot"
  timeout --signal=TERM --kill-after=60s "${WEEKLY_UNIVERSE_TIMEOUT_SECONDS:-7200}s" \
    "$PYTHON" -u -m strategies.momentum_expansion.main --refresh-universe
  rc=$?
  echo "[$(ts)] momentum snapshot exit=$rc"
  [ "$rc" -ne 0 ] && STATUS="$rc"

  # 3) Full-universe catalyst news backfill. The nightly job only collects the
  #    PRIORITY scope (swing ∪ top-300 momentum, ~1.25k names) to stay fast; this weekly
  #    sweep refreshes news for every eligible name (~2.9k), then runs the same
  #    incremental embed/cluster + rebuilds news_catalyst_signal so the broad
  #    corpus the Meta ranker reads is current. (This is the ~3h tail moved off
  #    the nightly path.)
  echo "[$(ts)] 3/4 full-universe news: collect (scope=full) → embed → signal"
  timeout --signal=TERM --kill-after=60s "${WEEKLY_NEWS_COLLECT_TIMEOUT_SECONDS:-43200}s" \
    "$PYTHON" -u -m scripts.collect_news_scope --scope full
  rc=$?
  echo "[$(ts)] news collect (full) exit=$rc"
  [ "$rc" -ne 0 ] && STATUS="$rc"
  timeout --signal=TERM --kill-after=60s "${WEEKLY_NEWS_PROCESS_TIMEOUT_SECONDS:-21600}s" \
    "$PYTHON" -u -m signals.news.main --stage incremental
  rc=$?
  echo "[$(ts)] news incremental exit=$rc"
  [ "$rc" -ne 0 ] && STATUS="$rc"
  timeout --signal=TERM --kill-after=60s "${WEEKLY_NEWS_SIGNAL_TIMEOUT_SECONDS:-10800}s" \
    "$PYTHON" -u -m signals.meta_context.build_news_signal --incremental-cache
  rc=$?
  echo "[$(ts)] news signal rebuild exit=$rc"
  [ "$rc" -ne 0 ] && STATUS="$rc"

  # 4) Meta Ranker feeds + dynamic themes (Claude $).
  echo "[$(ts)] 4/4 meta feeds + dynamic themes (--weekly, costs Claude \$)"
  timeout --signal=TERM --kill-after=60s "${WEEKLY_FEEDS_TIMEOUT_SECONDS:-21600}s" \
    "$PYTHON" -u signals/meta_context/meta_ranker/update_feeds.py --weekly
  rc=$?
  echo "[$(ts)] meta feeds exit=$rc"
  [ "$rc" -ne 0 ] && STATUS="$rc"

  echo ""
  echo "----------------------------------------------------------------"
  echo "[$(ts)] weekly_refresh.sh DONE status=$STATUS"
  echo ""
  echo "  >>> MANUAL STEP: re-auth Schwab (7-day token; dealer gamma levels"
  echo "      go stale without it). Run this and complete the browser login:"
  echo ""
  echo "        cd $REPO_ROOT"
  echo "        .venv/bin/python -m core.API.Schwab_API.schwab_client --reauth"
  echo ""
  echo "----------------------------------------------------------------"
} > >(tee -a "$LOG_FILE") 2>&1

exit "$STATUS"
