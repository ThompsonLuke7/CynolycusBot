#!/bin/bash
#
# Start the intraday catalyst poll as a detached background daemon.
#
# Designed to run alongside the combined dashboard — you can launch this once
# at the start of the trading day, then start your dashboard normally. The
# poll keeps running, writing to news/data/processed/live_catalyst_records.parquet,
# until you stop it via:
#
#   scripts/stop_catalyst_poll.sh
#
# or directly:
#
#   kill $(cat /tmp/catalyst_poll.pid)
#
# Logs land in news/data/processed/catalyst_poll.log
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
LOG_DIR="$REPO_ROOT/news/data/processed"
LOG_FILE="$LOG_DIR/catalyst_poll.log"
PID_FILE="/tmp/catalyst_poll.pid"
UNIVERSE="Data/shared/universe/shared_universe.csv"
INTERVAL=${CATALYST_POLL_INTERVAL:-300}
LOOKBACK=${CATALYST_POLL_LOOKBACK:-15}

mkdir -p "$LOG_DIR"

# Don't double-start
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "catalyst poll already running (PID $(cat "$PID_FILE")). Stop it first with scripts/stop_catalyst_poll.sh"
  exit 1
fi

nohup "$PYTHON" -u scripts/intraday_catalyst_poll.py \
  --universe-from "$UNIVERSE" \
  --interval "$INTERVAL" \
  --lookback-minutes "$LOOKBACK" \
  --sources google_news yfinance fed_rss \
  >> "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"
disown $PID

sleep 2
if kill -0 "$PID" 2>/dev/null; then
  echo "catalyst poll started: PID $PID, log $LOG_FILE"
  echo "  interval=${INTERVAL}s lookback=${LOOKBACK}m universe=$UNIVERSE"
  echo "  stop with: scripts/stop_catalyst_poll.sh"
else
  echo "catalyst poll failed to start — check $LOG_FILE"
  rm -f "$PID_FILE"
  exit 1
fi
