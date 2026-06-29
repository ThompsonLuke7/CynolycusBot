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

{
  echo ""
  echo "================================================================"
  echo "[$(ts)] nightly_data_readiness.sh starting"
  echo "================================================================"

  echo "[$(ts)] 1/3 catch up shared bars (1H/4H/1D, full universe, 429 backoff)"
  "$PYTHON" -u scripts/catchup_shared_bars.py --workers 4
  echo "[$(ts)] catchup exit=$?"

  echo "[$(ts)] 2/3 rebuild HTF 4H feature matrix (features_4h.parquet)"
  # --force is REQUIRED: without it build_all_features_4h() skips when the
  # combined parquet (and per-ticker parquets) already exist, so the matrix
  # froze and HTF Swing scored a 3-week-old bar. Forcing rebuilds both the
  # per-ticker features and the combined matrix off the freshly caught-up bars.
  "$PYTHON" -u -m strategies.momentum_expansion.main --build-features --force
  echo "[$(ts)] build-features exit=$?"

  echo "[$(ts)] 3/3 append fresh bars to the Meta Ranker matrix"
  "$PYTHON" -u signals/meta_context/meta_ranker/update_meta_matrix.py
  echo "[$(ts)] meta matrix exit=$?"

  echo "[$(ts)] nightly_data_readiness.sh complete"
} >> "$LOG_FILE" 2>&1
