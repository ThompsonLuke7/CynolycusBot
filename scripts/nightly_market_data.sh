#!/bin/bash
#
# Nightly market-data refresh for the catalyst module.
#
# Pulls today's CBOE per-ticker options snapshot (appends to
# cboe_options_summary.parquet) and yesterday's FINRA short-volume CSV
# (appends to a per-day file under signals/news/data/processed/finra_daily/),
# runs ticker discovery, then collects + processes the day's catalyst news and
# rebuilds the per-(ticker, day) news_catalyst_signal the meta-ranker consumes.
#
# Canonical owner: the supervised combined server, after the live refresher and
# 4H loops finish. Safe to re-run manually; the shared heavy-job lock prevents
# overlap with readiness, another nightly run, or the weekly refresh.
# — the CBOE collector de-dupes by (ticker, snapshot_date), and the FINRA
# fetch is idempotent per (date, ticker).
#
# Usage (manual): bash scripts/nightly_market_data.sh
# Schedule:       scripts/run_live_server.sh -> combined server at 16:45 ET
#
# Logs go to signals/news/data/processed/nightly_cron.log
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
LOG_DIR="$REPO_ROOT/signals/news/data/processed"
LOG_FILE="$LOG_DIR/nightly_cron.log"
LOCK_FILE="$REPO_ROOT/Data/runtime/live_data_jobs.lock"
mkdir -p "$LOG_DIR/finra_daily" "$REPO_ROOT/Data/runtime"

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

OWNS_LOCK=0
clear_owned_lock() {
  if [ "$OWNS_LOCK" = "1" ]; then
    : > "$LOCK_FILE"
  fi
}
trap clear_owned_lock EXIT

if command -v flock >/dev/null 2>&1; then
  exec 9>>"$LOCK_FILE"
  if ! flock -w "${NIGHTLY_LOCK_WAIT_SECONDS:-1200}" 9; then
    {
      echo ""
      echo "[$(ts)] nightly_market_data.sh skipped: heavy-job lock unavailable after ${NIGHTLY_LOCK_WAIT_SECONDS:-1200}s"
    } >> "$LOG_FILE" 2>&1
    exit 75
  fi
  OWNS_LOCK=1
  printf 'nightly_market_data pid=%s started=%s\n' "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$LOCK_FILE"
fi

{
  echo ""
  echo "================================================================"
  echo "[$(ts)] nightly_market_data.sh starting"
  echo "================================================================"
  STATUS=0

  # 1) CBOE per-ticker options snapshot (full universe, ~30 min)
  echo "[$(ts)] CBOE snapshot — full universe"
  timeout --signal=TERM --kill-after=60s "${NIGHTLY_CBOE_TIMEOUT_SECONDS:-7200}s" \
    "$PYTHON" -u -m signals.news.main --stage cboe-snapshot
  cboe_exit=$?
  echo "[$(ts)] CBOE snapshot exit=$cboe_exit"
  [ "$cboe_exit" -ne 0 ] && STATUS="$cboe_exit"

  # 1b) Dealer-positioning option snapshots (forward collection). This stores
  #     the Amethyst-ready floors/ceilings/magnets/vega walls plus strike rows.
  #     It is intentionally configurable because the Schwab chain API can rate
  #     limit on a very broad universe.
  if [ "${DEALER_SNAPSHOT_ENABLED:-1}" != "0" ]; then
    echo "[$(ts)] Dealer snapshots — floors/ceilings/magnets/strike ladders"
    dealer_args=(
      --sleep-seconds "${DEALER_SNAPSHOT_SLEEP_SECONDS:-0.25}"
      --strike-window-pct "${DEALER_SNAPSHOT_STRIKE_WINDOW_PCT:-0.50}"
      --min-total-oi "${DEALER_SNAPSHOT_MIN_TOTAL_OI:-5000}"
      --min-option-volume "${DEALER_SNAPSHOT_MIN_OPTION_VOLUME:-2000}"
      --min-adv "${DEALER_SNAPSHOT_MIN_ADV:-50000000}"
      --min-market-cap "${DEALER_SNAPSHOT_MIN_MARKET_CAP:-2000000000}"
    )
    if [ "${DEALER_SNAPSHOT_LIQUIDITY_PREFILTER:-1}" = "0" ]; then
      dealer_args+=(--no-liquidity-prefilter)
    fi
    if [ -n "${DEALER_SNAPSHOT_LIMIT:-}" ]; then
      dealer_args+=(--limit "$DEALER_SNAPSHOT_LIMIT")
    fi
    if [ -n "${DEALER_SNAPSHOT_SCOPES:-}" ]; then
      # shellcheck disable=SC2206
      dealer_scopes=(${DEALER_SNAPSHOT_SCOPES})
      dealer_args+=(--scopes "${dealer_scopes[@]}")
    fi
    timeout --signal=TERM --kill-after=60s "${NIGHTLY_DEALER_TIMEOUT_SECONDS:-14400}s" \
      "$PYTHON" -u -m strategies.dealer_positioning.scripts.capture_historical_snapshots "${dealer_args[@]}"
    dealer_snapshot_exit=$?
    echo "[$(ts)] Dealer snapshots exit=$dealer_snapshot_exit"
    if [ "$dealer_snapshot_exit" -eq 0 ]; then
      echo "[$(ts)] Dealer rankings — swing-potential research ranks"
      timeout --signal=TERM --kill-after=60s "${NIGHTLY_DEALER_RANK_TIMEOUT_SECONDS:-1800}s" \
        "$PYTHON" -u -m strategies.dealer_positioning.scripts.build_dealer_rankings
      dealer_ranking_exit=$?
      echo "[$(ts)] Dealer rankings exit=$dealer_ranking_exit"
    fi
  else
    echo "[$(ts)] Dealer snapshots disabled by DEALER_SNAPSHOT_ENABLED=0"
  fi

  # 2) FINRA prior-trading-day short volume (appends a single day's parquet)
  echo "[$(ts)] FINRA — pulling yesterday's CSV"
  timeout --signal=TERM --kill-after=30s "${NIGHTLY_FINRA_TIMEOUT_SECONDS:-900}s" "$PYTHON" -u <<'PYEOF'
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, ".")
from signals.news.sources import fetch_finra_short_volume_day

target = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)
while target.weekday() >= 5:
    target -= pd.Timedelta(days=1)

out_path = Path(f"signals/news/data/processed/finra_daily/{target.strftime('%Y%m%d')}.parquet")
if out_path.exists():
    print(f"  {out_path} already exists — skipping")
    sys.exit(0)

df = fetch_finra_short_volume_day(target)
if df.empty:
    print(f"  no rows for {target.date()} (likely market holiday)")
    sys.exit(0)

df.to_parquet(out_path, index=False)
print(f"  wrote {len(df):,} rows to {out_path}")
PYEOF
  finra_exit=$?
  echo "[$(ts)] FINRA fetch exit=$finra_exit"
  [ "$finra_exit" -ne 0 ] && STATUS="$finra_exit"

  # 3) Compact: merge any new finra_daily/*.parquet into the consolidated parquet
  echo "[$(ts)] FINRA — compacting daily files into consolidated parquet"
  timeout --signal=TERM --kill-after=30s "${NIGHTLY_COMPACT_TIMEOUT_SECONDS:-1800}s" "$PYTHON" -u <<'PYEOF'
from pathlib import Path
import pandas as pd

daily_dir = Path("signals/news/data/processed/finra_daily")
consolidated = Path("signals/news/data/processed/finra_short_volume.parquet")

dailies = sorted(daily_dir.glob("*.parquet"))
if not dailies:
    print("  no daily files to compact")
    raise SystemExit(0)

frames = []
if consolidated.exists():
    frames.append(pd.read_parquet(consolidated))
for d in dailies:
    frames.append(pd.read_parquet(d))
all_df = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "ticker"])
all_df.to_parquet(consolidated, index=False)
print(f"  consolidated: {len(all_df):,} rows -> {consolidated}")
PYEOF
  finra_compact_exit=$?
  echo "[$(ts)] FINRA compact exit=$finra_compact_exit"
  [ "$finra_compact_exit" -ne 0 ] && STATUS="$finra_compact_exit"

  # 4) Dynamic ticker discovery + promotion gate
  #    Broad Alpaca market screen + catalyst-ledger discovery -> pending queue;
  #    releases names meeting the price/ADV/market-cap/history minimums and
  #    folds promoted tickers into shared_universe.csv.
  echo "[$(ts)] ticker discovery — broad screen + catalyst, promotion gate"
  timeout --signal=TERM --kill-after=60s "${NIGHTLY_DISCOVERY_TIMEOUT_SECONDS:-3600}s" \
    "$PYTHON" -u -m scripts.nightly_ticker_discovery --rebuild-universe
  discovery_exit=$?
  echo "[$(ts)] ticker discovery exit=$discovery_exit"
  [ "$discovery_exit" -ne 0 ] && STATUS="$discovery_exit"

  # 5) Catalyst news — collect headlines for the PRIORITY scope only (the names
  #    the live system actually trades/ranks: swing universe ∪ top-300 momentum).
  #    The broad full-universe sweep is the weekly job's responsibility — that's
  #    the ~3h tail we moved off the nightly path. Breaking news on tradeable
  #    names still flows through the same incremental embed/cluster + signal below.
  echo "[$(ts)] news — collecting headlines (PRIORITY scope: swing ∪ top-300 momentum)"
  timeout --signal=TERM --kill-after=60s "${NIGHTLY_NEWS_COLLECT_TIMEOUT_SECONDS:-14400}s" \
    "$PYTHON" -u -m scripts.collect_news_scope --scope priority
  news_collect_exit=$?
  echo "[$(ts)] news collect exit=$news_collect_exit"
  [ "$news_collect_exit" -ne 0 ] && STATUS="$news_collect_exit"

  # 6) Catalyst news — incremental embed/cluster/refine/label of only-new records
  #    (~minutes; chains embed-new -> cluster-predict -> refine -> label-mature).
  echo "[$(ts)] news — incremental processing (new records only)"
  timeout --signal=TERM --kill-after=60s "${NIGHTLY_NEWS_PROCESS_TIMEOUT_SECONDS:-7200}s" \
    "$PYTHON" -u -m signals.news.main --stage incremental
  news_incr_exit=$?
  echo "[$(ts)] news incremental exit=$news_incr_exit"
  [ "$news_incr_exit" -ne 0 ] && STATUS="$news_incr_exit"

  # 7) Catalyst news — refresh per-record predictions only for new/stale records,
  #    then aggregate to the per-(ticker, day) signal the meta-ranker reads.
  echo "[$(ts)] news — rebuilding news_catalyst_signal.parquet"
  timeout --signal=TERM --kill-after=60s "${NIGHTLY_NEWS_SIGNAL_TIMEOUT_SECONDS:-3600}s" \
    "$PYTHON" -u -m signals.meta_context.build_news_signal --incremental-cache
  news_signal_exit=$?
  echo "[$(ts)] news signal rebuild exit=$news_signal_exit"
  [ "$news_signal_exit" -ne 0 ] && STATUS="$news_signal_exit"

  # 8) Emerging themes — enrich a bounded pending-ticker shortlist, map strong
  #    matches into established themes, cluster novel residuals, and regenerate
  #    the standalone visualizers. Production theme features remain untouched.
  echo "[$(ts)] themes — pending ticker enrichment + provisional discovery"
  timeout --signal=TERM --kill-after=60s "${NIGHTLY_THEME_TIMEOUT_SECONDS:-3600}s" \
    "$PYTHON" -u -m themes.dynamic_theme.emerging --limit 300
  emerging_theme_exit=$?
  echo "[$(ts)] emerging themes exit=$emerging_theme_exit"

  if [ "$emerging_theme_exit" -eq 0 ]; then
    echo "[$(ts)] themes — rebuilding explorer"
    timeout --signal=TERM --kill-after=30s 1200s "$PYTHON" -u -m themes.dynamic_theme.viz.build_theme_explorer
  fi

  # 9) Earnings enrichment is useful context, but it is deliberately outside
  #    the entry-readiness critical path.  Each Yahoo ticker request is isolated
  #    and hard-bounded; this outer deadline caps the whole sweep as well.
  echo "[$(ts)] earnings — bounded equity calendar enrichment (best effort; known ETFs excluded)"
  timeout --signal=TERM --kill-after=60s "${NIGHTLY_EARNINGS_TIMEOUT_SECONDS:-3600}s" \
    "$PYTHON" -u -m signals.news.main --stage earnings-calendar \
      --ticker-timeout "${EARNINGS_TICKER_TIMEOUT_SECONDS:-20}"
  earnings_exit=$?
  echo "[$(ts)] earnings calendar exit=$earnings_exit (non-critical to entry readiness)"

  echo "[$(ts)] nightly_market_data.sh complete status=$STATUS"
} >> "$LOG_FILE" 2>&1

exit "$STATUS"
