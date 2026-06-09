#!/bin/bash
#
# Nightly market-data refresh for the catalyst module.
#
# Pulls today's CBOE per-ticker options snapshot (appends to
# cboe_options_summary.parquet) and yesterday's FINRA short-volume CSV
# (appends to a per-day file under news/data/processed/finra_daily/).
#
# Designed to be invoked from cron. Safe to re-run multiple times per day
# — the CBOE collector de-dupes by (ticker, snapshot_date), and the FINRA
# fetch is idempotent per (date, ticker).
#
# Usage (manual): bash scripts/nightly_market_data.sh
# Usage (cron):   30 23 * * 1-5 bash /home/luket/repos/CynolycusBot/scripts/nightly_market_data.sh
#
# Logs go to news/data/processed/nightly_cron.log
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
LOG_DIR="$REPO_ROOT/news/data/processed"
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
  "$PYTHON" -u -m news.main --stage cboe-snapshot
  cboe_exit=$?
  echo "[$(ts)] CBOE snapshot exit=$cboe_exit"

  # 2) FINRA prior-trading-day short volume (appends a single day's parquet)
  echo "[$(ts)] FINRA — pulling yesterday's CSV"
  "$PYTHON" -u <<'PYEOF'
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, ".")
from news.sources import fetch_finra_short_volume_day

target = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1)
while target.weekday() >= 5:
    target -= pd.Timedelta(days=1)

out_path = Path(f"news/data/processed/finra_daily/{target.strftime('%Y%m%d')}.parquet")
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

daily_dir = Path("news/data/processed/finra_daily")
consolidated = Path("news/data/processed/finra_short_volume.parquet")

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

  echo "[$(ts)] nightly_market_data.sh complete"
} >> "$LOG_FILE" 2>&1
