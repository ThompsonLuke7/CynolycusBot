"""Nightly: the 30m ranker's daily context rows for the next session (and recent ones).

The live scanner looks the ~45 daily context features (earnings, macro calendar, treasury,
news catalysts, dynamic themes) up as of the bar's date. They came only from the training
export ``data/training_export/daily_context_features.parquet``, which ends 2026-06-03, so
every live decision since then was served June values (decision-state parity, 2026-11-23).

This builds rows with the training export's own function
(``export_for_colab._build_daily_context_features``: as-of-prior-day joins, so a row for
session D uses only sources dated before D) and upserts them into a SEPARATE live file;
the training export stays as built.

    .venv/bin/python -m strategies.multi_ticker_swing.live.refresh_daily_context
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from strategies.multi_ticker_swing.live.context_features import (  # noqa: E402
    LIVE_DAILY_CONTEXT, TRAINING_DAILY_CONTEXT)

logger = logging.getLogger("swing.refresh_daily_context")
UNIVERSE_JSON = REPO / "strategies/multi_ticker_swing/config/trading_universe.json"


def session_dates(asof: pd.Timestamp, back: int) -> list[pd.Timestamp]:
    """The ``back`` sessions up to ``asof`` plus the next session after it (naive dates)."""
    from core.calendar import is_trading_day

    day = pd.Timestamp(asof).normalize()
    nxt = day + pd.Timedelta(days=1)
    while not is_trading_day(nxt.date()):
        nxt += pd.Timedelta(days=1)
    out, d = [], day
    while len(out) < back:
        if is_trading_day(d.date()):
            out.append(d)
        d -= pd.Timedelta(days=1)
    return sorted(out) + [nxt]


def build_rows(tickers: list[str], dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Context rows keyed (ticker, date) in the training export's schema."""
    from strategies.multi_ticker_swing.models import export_for_colab as ex

    keys = pd.DataFrame([(t, d + pd.Timedelta(hours=15)) for t in tickers for d in dates],
                        columns=["ticker", "timestamp"])
    keys["timestamp"] = pd.to_datetime(keys["timestamp"]).dt.tz_localize("UTC")
    ctx, _cols, sources = ex._build_daily_context_features(keys)
    schema = list(pd.read_parquet(TRAINING_DAILY_CONTEXT).columns) if TRAINING_DAILY_CONTEXT.exists() else list(ctx.columns)
    for c in schema:
        if c not in ctx.columns:
            ctx[c] = float("nan")
    ctx = ctx[schema].copy()
    ctx.attrs["sources"] = sources
    return ctx


def upsert(rows: pd.DataFrame, path: Path = LIVE_DAILY_CONTEXT) -> int:
    """Replace (ticker, date) rows in ``path`` with ``rows``; atomic write. Returns total rows."""
    rows = rows.copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.tz_localize(None).dt.normalize()
    if path.exists():
        old = pd.read_parquet(path)
        old["date"] = pd.to_datetime(old["date"]).dt.tz_localize(None).dt.normalize()
        keep = ~old.set_index(["ticker", "date"]).index.isin(rows.set_index(["ticker", "date"]).index)
        rows = pd.concat([old[keep], rows], ignore_index=True)
    rows = rows.sort_values(["ticker", "date"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    rows.to_parquet(tmp, index=False)
    tmp.replace(path)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--asof", default=None, help="NY date; default today")
    ap.add_argument("--back", type=int, default=5, help="recent sessions to (re)build besides the next one")
    ap.add_argument("--out", default=str(LIVE_DAILY_CONTEXT))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asof = pd.Timestamp(args.asof) if args.asof else pd.Timestamp.now(tz="America/New_York").tz_localize(None)
    tickers = sorted({str(t).upper() for t in json.loads(UNIVERSE_JSON.read_text())})
    dates = session_dates(asof, args.back)
    rows = build_rows(tickers, dates)
    if rows.empty:
        logger.error("no context rows built for %s", [d.date() for d in dates])
        return 1
    total = upsert(rows, Path(args.out))
    cover = rows.drop(columns=["ticker", "date"]).notna().mean()
    print(json.dumps({"tickers": len(tickers), "dates": [str(d.date()) for d in dates], "rows": len(rows),
                      "file_rows": total, "out": args.out, "sources": rows.attrs.get("sources"),
                      "lowest_coverage": cover.sort_values().head(5).round(2).to_dict()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
