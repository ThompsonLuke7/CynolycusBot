#!/bin/bash
#
# Stop the detached intraday catalyst poll started by scripts/start_catalyst_poll.sh
#
PID_FILE="/tmp/catalyst_poll.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "no PID file at $PID_FILE — poll is not running (or was started a different way)"
  exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  sleep 1
  if kill -0 "$PID" 2>/dev/null; then
    echo "PID $PID didn't stop on SIGTERM — sending SIGKILL"
    kill -9 "$PID"
  fi
  echo "catalyst poll stopped (was PID $PID)"
else
  echo "PID $PID not running (stale PID file)"
fi
rm -f "$PID_FILE"
