"""Deterministic high-impact US economic calendar — no API key required.

Generates the dates of the macro events that actually move the broad market, from
their fixed/published public schedules, both historically and forward:

  FOMC  — rate decisions (hard-coded published meeting dates, 2022→2027)
  NFP   — monthly jobs report, first Friday rule (BLS)
  CPI   — monthly inflation print, BLS mid-month (rule-based estimate)

These are market-wide events (not per-ticker), used by the Meta Ranker for
days_to/since_macro_event features. Re-run weekly to roll the forward window:

    python -m signals.events.economic_calendar --build

Output: signals/news/data/processed/economic_calendar.parquet
        schema: date | event | impact | country
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
OUT_PATH = REPO / "signals" / "news" / "data" / "processed" / "economic_calendar.parquet"

# FOMC rate-decision dates (second day of meeting), as published by the Federal
# Reserve. 2022–2026 are final; 2027 is the Fed's announced tentative schedule.
_FOMC_DATES = [
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16", "2027-07-28", "2027-09-22", "2027-11-03", "2027-12-15",
]


def _first_friday(year: int, month: int) -> pd.Timestamp:
    """NFP (jobs report) — released the first Friday of each month, 8:30 ET."""
    d = pd.Timestamp(year=year, month=month, day=1)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def _cpi_estimate(year: int, month: int) -> pd.Timestamp:
    """CPI — BLS releases mid-month; estimate the nearest weekday to the 12th.

    Rule-based (exact BLS dates vary by 1-3 days) — adequate for a market-wide
    'days to inflation print' feature, but not a precise per-day timestamp.
    """
    d = pd.Timestamp(year=year, month=month, day=12)
    if d.weekday() == 5:   # Sat -> Fri
        d -= pd.Timedelta(days=1)
    elif d.weekday() == 6:  # Sun -> Mon
        d += pd.Timedelta(days=1)
    return d


def build_economic_calendar(start: str = "2022-01-01", forward_months: int = 12) -> pd.DataFrame:
    """Assemble the deterministic high-impact macro calendar."""
    start_ts = pd.Timestamp(start)
    end_ts = (pd.Timestamp.now() + pd.DateOffset(months=forward_months)).normalize()

    rows: list[dict] = []
    for d in _FOMC_DATES:
        rows.append({"date": pd.Timestamp(d), "event": "FOMC rate decision", "impact": "high"})

    # NFP + CPI for every month in range
    cur = start_ts.replace(day=1)
    while cur <= end_ts:
        rows.append({"date": _first_friday(cur.year, cur.month), "event": "Nonfarm payrolls (NFP)", "impact": "high"})
        rows.append({"date": _cpi_estimate(cur.year, cur.month), "event": "CPI inflation", "impact": "high"})
        cur += pd.DateOffset(months=1)

    cal = pd.DataFrame(rows)
    cal["country"] = "US"
    cal = cal[(cal["date"] >= start_ts) & (cal["date"] <= end_ts)]
    cal = cal.sort_values("date").drop_duplicates(["date", "event"]).reset_index(drop=True)
    return cal[["date", "event", "impact", "country"]]


def load_economic_calendar(path: Path = OUT_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "event", "impact", "country"])
    return pd.read_parquet(path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build deterministic high-impact US economic calendar")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--forward-months", type=int, default=12)
    args = ap.parse_args()

    if not args.build:
        cal = load_economic_calendar()
        print(f"economic calendar: {len(cal)} events"
              + (f", {cal['date'].min().date()} → {cal['date'].max().date()}" if len(cal) else ""))
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cal = build_economic_calendar(forward_months=args.forward_months)
    cal.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %s  events=%d  range %s → %s  (FOMC %d, NFP+CPI monthly)",
                OUT_PATH, len(cal), cal["date"].min().date(), cal["date"].max().date(),
                (cal["event"] == "FOMC rate decision").sum())


if __name__ == "__main__":
    main()
