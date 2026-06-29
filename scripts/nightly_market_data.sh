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
# Designed to be invoked from cron. Safe to re-run multiple times per day
# — the CBOE collector de-dupes by (ticker, snapshot_date), and the FINRA
# fetch is idempotent per (date, ticker).
#
# Usage (manual): bash scripts/nightly_market_data.sh
# Usage (cron):   30 23 * * 1-5 bash /home/luket/repos/CynolycusBot/scripts/nightly_market_data.sh
#
# Logs go to signals/news/data/processed/nightly_cron.log
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
LOG_DIR="$REPO_ROOT/signals/news/data/processed"
LOG_FILE="$LOG_DIR/nightly_cron.log"
mkdir -p "$LOG_DIR/finra_daily"

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

{
  echo ""
  echo "================================================================"
  echo "[$(ts)] nightly_market_data.sh starting"
  echo "================================================================"

  # 1) CBOE per-ticker options snapshot (full universe, ~30 min)
  echo "[$(ts)] CBOE snapshot — full universe"
  "$PYTHON" -u -m signals.news.main --stage cboe-snapshot
  cboe_exit=$?
  echo "[$(ts)] CBOE snapshot exit=$cboe_exit"

  # 2) FINRA prior-trading-day short volume (appends a single day's parquet)
  echo "[$(ts)] FINRA — pulling yesterday's CSV"
  "$PYTHON" -u <<'PYEOF'
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

  # 3) Compact: merge any new finra_daily/*.parquet into the consolidated parquet
  echo "[$(ts)] FINRA — compacting daily files into consolidated parquet"
  "$PYTHON" -u <<'PYEOF'
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

  # 4) Dynamic ticker discovery + promotion gate
  #    Broad Alpaca market screen + catalyst-ledger discovery -> pending queue;
  #    releases names meeting the price/ADV/market-cap/history minimums and
  #    folds promoted tickers into shared_universe.csv.
  echo "[$(ts)] ticker discovery — broad screen + catalyst, promotion gate"
  "$PYTHON" -u -m scripts.nightly_ticker_discovery --rebuild-universe
  discovery_exit=$?
  echo "[$(ts)] ticker discovery exit=$discovery_exit"

  # 5) Catalyst news — collect headlines for the PRIORITY scope only (the names
  #    the live system actually trades/ranks: swing universe ∪ momentum snapshot).
  #    The broad full-universe sweep is the weekly job's responsibility — that's
  #    the ~3h tail we moved off the nightly path. Breaking news on tradeable
  #    names still flows through the same incremental embed/cluster + signal below.
  echo "[$(ts)] news — collecting headlines (PRIORITY scope: swing ∪ momentum)"
  "$PYTHON" -u -m scripts.collect_news_scope --scope priority
  news_collect_exit=$?
  echo "[$(ts)] news collect exit=$news_collect_exit"

  # 6) Catalyst news — incremental embed/cluster/refine/label of only-new records
  #    (~minutes; chains embed-new -> cluster-predict -> refine -> label-mature).
  echo "[$(ts)] news — incremental processing (new records only)"
  "$PYTHON" -u -m signals.news.main --stage incremental
  news_incr_exit=$?
  echo "[$(ts)] news incremental exit=$news_incr_exit"

  # 7) Catalyst news — refresh per-record predictions only for new/stale records,
  #    then aggregate to the per-(ticker, day) signal the meta-ranker reads.
  echo "[$(ts)] news — rebuilding news_catalyst_signal.parquet"
  "$PYTHON" -u -m signals.meta_context.build_news_signal --incremental-cache
  news_signal_exit=$?
  echo "[$(ts)] news signal rebuild exit=$news_signal_exit"

  # 8) Emerging themes — enrich a bounded pending-ticker shortlist, map strong
  #    matches into established themes, cluster novel residuals, and regenerate
  #    the standalone visualizers. Production theme features remain untouched.
  echo "[$(ts)] themes — pending ticker enrichment + provisional discovery"
  "$PYTHON" -u -m themes.dynamic_theme.emerging --limit 300
  emerging_theme_exit=$?
  echo "[$(ts)] emerging themes exit=$emerging_theme_exit"

  if [ "$emerging_theme_exit" -eq 0 ]; then
    echo "[$(ts)] themes — rebuilding explorer variants and dreamscapes"
    "$PYTHON" -u -m themes.dynamic_theme.viz.build_theme_explorer
    "$PYTHON" -u -m themes.dynamic_theme.viz.build_theme_variants
    "$PYTHON" -u -m themes.dynamic_theme.viz.build_theme_dreamscapes
  fi

  echo "[$(ts)] nightly_market_data.sh complete"
} >> "$LOG_FILE" 2>&1
