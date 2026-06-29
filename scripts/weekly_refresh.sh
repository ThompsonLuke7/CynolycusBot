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
export PYTHONPATH="$REPO_ROOT"

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

{
  echo ""
  echo "================================================================"
  echo "[$(ts)] weekly_refresh.sh starting"
  echo "================================================================"

  # 1) Backfill bars for the whole universe (incl. newly promoted names).
  echo "[$(ts)] 1/4 catch up shared bars (1H/4H/1D, full universe)"
  "$PYTHON" -u scripts/catchup_shared_bars.py --workers 6
  echo "[$(ts)] catchup exit=$?"

  # 2) Momentum weekly universe snapshot (must follow #1 to include new names).
  echo "[$(ts)] 2/4 momentum universe snapshot"
  "$PYTHON" -u -m strategies.momentum_expansion.main --refresh-universe
  echo "[$(ts)] momentum snapshot exit=$?"

  # 3) Full-universe catalyst news backfill. The nightly job only collects the
  #    PRIORITY scope (swing ∪ momentum, ~1.1k names) to stay fast; this weekly
  #    sweep refreshes news for every eligible name (~2.9k), then runs the same
  #    incremental embed/cluster + rebuilds news_catalyst_signal so the broad
  #    corpus the Meta ranker reads is current. (This is the ~3h tail moved off
  #    the nightly path.)
  echo "[$(ts)] 3/4 full-universe news: collect (scope=full) → embed → signal"
  "$PYTHON" -u -m scripts.collect_news_scope --scope full
  echo "[$(ts)] news collect (full) exit=$?"
  "$PYTHON" -u -m signals.news.main --stage incremental
  echo "[$(ts)] news incremental exit=$?"
  "$PYTHON" -u -m signals.meta_context.build_news_signal --incremental-cache
  echo "[$(ts)] news signal rebuild exit=$?"

  # 4) Meta Ranker feeds + dynamic themes (Claude $).
  echo "[$(ts)] 4/4 meta feeds + dynamic themes (--weekly, costs Claude \$)"
  "$PYTHON" -u signals/meta_context/meta_ranker/update_feeds.py --weekly
  echo "[$(ts)] meta feeds exit=$?"

  echo ""
  echo "----------------------------------------------------------------"
  echo "[$(ts)] weekly_refresh.sh DONE"
  echo ""
  echo "  >>> MANUAL STEP: re-auth Schwab (7-day token; dealer gamma levels"
  echo "      go stale without it). Run this and complete the browser login:"
  echo ""
  echo "        cd $REPO_ROOT"
  echo "        .venv/bin/python -m core.API.Schwab_API.schwab_client --reauth"
  echo ""
  echo "----------------------------------------------------------------"
} 2>&1 | tee -a "$LOG_FILE"
