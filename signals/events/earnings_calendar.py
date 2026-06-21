"""Earnings calendar — historical + forward, for the whole universe.

ONE yfinance call per ticker returns its full earnings history AND its upcoming
scheduled dates (with EPS estimate / actual / surprise), so the same artifact
serves both model training (historical `days_to/since_earnings`) and live trading
(forward `days_to_earnings` capped at the next scheduled report).

Fetch / weekly refresh (idempotent — overwrites with the latest schedule):

    python -m signals.events.earnings_calendar --refresh           # whole universe
    python -m signals.events.earnings_calendar --refresh --tickers NVDA AAPL

Use in feature builders:

    from signals.events.earnings_calendar import load_earnings_calendar, add_earnings_features
    df = add_earnings_features(df, date_col="date")   # adds days_to/since, pre/post flags

Output: signals/news/data/processed/ticker_earnings_calendar.parquet
        schema: ticker | date | eps_estimate | reported_eps | surprise_pct
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
OUT_PATH = REPO / "signals" / "news" / "data" / "processed" / "ticker_earnings_calendar.parquet"

# How far before/after an earnings date the proximity features stay populated.
MAX_EARNINGS_DISTANCE_DAYS = 90
# yfinance history depth: 40 quarters ≈ 10 years back plus the upcoming dates.
_LIMIT = 40


# ── fetch ─────────────────────────────────────────────────────────────────────

def _fetch_one(ticker: str, *, retries: int = 3) -> pd.DataFrame | None:
    """Return a ticker's earnings dates (past + upcoming) or None on failure.

    Retries with backoff — yfinance/Yahoo rate-limits aggressively past a few
    hundred rapid requests, so transient empties are retried before giving up.
    """
    import yfinance as yf
    ed = None
    for attempt in range(retries):
        try:
            ed = yf.Ticker(ticker).get_earnings_dates(limit=_LIMIT)
            if ed is not None and not ed.empty:
                break
        except Exception as exc:
            logger.debug("%s earnings fetch failed (attempt %d): %s", ticker, attempt + 1, exc)
        time.sleep(0.6 * (attempt + 1))
    if ed is None or ed.empty:
        return None
    ed = ed.reset_index()
    date_col = next((c for c in ed.columns if "Date" in c), ed.columns[0])
    out = pd.DataFrame({
        "ticker": ticker,
        "date": pd.to_datetime(ed[date_col], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize(),
        "eps_estimate": pd.to_numeric(ed.get("EPS Estimate"), errors="coerce"),
        "reported_eps": pd.to_numeric(ed.get("Reported EPS"), errors="coerce"),
        "surprise_pct": pd.to_numeric(ed.get("Surprise(%)"), errors="coerce"),
    })
    return out.dropna(subset=["date"]).drop_duplicates(["ticker", "date"])


def fetch_earnings_calendar(
    tickers: list[str],
    *,
    out_path: Path = OUT_PATH,
    max_workers: int = 8,
    throttle: float = 0.0,
    merge: bool = False,
) -> pd.DataFrame:
    """Pull earnings dates for every ticker and write one consolidated parquet.

    merge=True keeps any tickers already in out_path and only fetches the rest —
    use it to fill in tickers that failed an earlier (rate-limited) pass.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tickers = [str(t).upper() for t in dict.fromkeys(tickers)]

    existing = pd.DataFrame()
    if merge and out_path.exists():
        existing = load_earnings_calendar(out_path)
        have = set(existing["ticker"].unique())
        tickers = [t for t in tickers if t not in have]
        logger.info("Merge mode: %d already present, fetching %d missing", len(have), len(tickers))
    logger.info("Fetching earnings calendar for %d tickers (yfinance) ...", len(tickers))

    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            df = fut.result()
            if df is not None and not df.empty:
                frames.append(df)
            else:
                failures.append(t)
            done += 1
            if done % 100 == 0:
                logger.info("  ... %d/%d (%d ok, %d empty/failed)", done, len(tickers), len(frames), len(failures))
            if throttle:
                time.sleep(throttle)

    if not frames and existing.empty:
        logger.error("No earnings data fetched for any ticker")
        return pd.DataFrame(columns=["ticker", "date", "eps_estimate", "reported_eps", "surprise_pct"])

    all_frames = ([existing] if not existing.empty else []) + frames
    cal = (pd.concat(all_frames, ignore_index=True)
           .drop_duplicates(["ticker", "date"])
           .sort_values(["ticker", "date"]).reset_index(drop=True))
    cal.to_parquet(out_path, index=False)
    logger.info(
        "Wrote %s  rows=%d  tickers=%d  range %s → %s  (%d had no data)",
        out_path, len(cal), cal["ticker"].nunique(),
        cal["date"].min().date(), cal["date"].max().date(), len(failures),
    )
    if failures:
        logger.info("No earnings data for: %s%s", ", ".join(failures[:25]), " ..." if len(failures) > 25 else "")
    return cal


# ── load + feature engineering ────────────────────────────────────────────────

def load_earnings_calendar(path: Path = OUT_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "date", "eps_estimate", "reported_eps", "surprise_pct"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df


def add_earnings_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    fwd_window_days: int | None = None,
    calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add point-in-time earnings-proximity features to a (ticker, date) frame.

    Adds:
      days_to_earnings      days until the NEXT scheduled report (capped, the
                            "earnings wall" — large right after a report)
      days_since_earnings   days since the most recent report (capped)
      is_pre_earnings_3d    1 if the next report is within 3 days
      is_post_earnings_3d   1 if the last report was within 3 days
      earnings_in_fwd_window (only if fwd_window_days given) 1 if a report falls
                            inside the model's forward label window — use this to
                            down-weight / exclude event-driven labels.
    """
    cal = calendar if calendar is not None else load_earnings_calendar()
    out = df.copy()
    base = ["days_to_earnings", "days_since_earnings", "is_pre_earnings_3d", "is_post_earnings_3d"]
    for c in base:
        out[c] = np.nan
    if fwd_window_days is not None:
        out["earnings_in_fwd_window"] = np.nan
    if cal.empty:
        return out

    spine = out[[ticker_col, date_col]].copy()
    spine["_d"] = pd.to_datetime(spine[date_col]).dt.tz_localize(None).dt.normalize()
    events = cal.rename(columns={"ticker": ticker_col})[[ticker_col, "date"]].copy()

    nxt = pd.merge_asof(
        spine.sort_values("_d"),
        events.rename(columns={"date": "_next"}).sort_values("_next"),
        left_on="_d", right_on="_next", by=ticker_col, direction="forward",
    )["_next"].to_numpy()
    prv = pd.merge_asof(
        spine.sort_values("_d"),
        events.rename(columns={"date": "_prev"}).sort_values("_prev"),
        left_on="_d", right_on="_prev", by=ticker_col, direction="backward",
    )["_prev"].to_numpy()
    # merge_asof reorders by the sort key; realign to the (sorted) spine index
    order = spine["_d"].sort_values().index
    days_to = pd.Series((pd.to_datetime(nxt) - spine.loc[order, "_d"].to_numpy()) / np.timedelta64(1, "D"), index=order)
    days_since = pd.Series((spine.loc[order, "_d"].to_numpy() - pd.to_datetime(prv)) / np.timedelta64(1, "D"), index=order)
    days_to = days_to.reindex(out.index)
    days_since = days_since.reindex(out.index)

    days_to = days_to.where(days_to <= MAX_EARNINGS_DISTANCE_DAYS)
    days_since = days_since.where(days_since <= MAX_EARNINGS_DISTANCE_DAYS)
    out["days_to_earnings"] = days_to.astype(float)
    out["days_since_earnings"] = days_since.astype(float)
    out["is_pre_earnings_3d"] = ((days_to >= 0) & (days_to <= 3)).astype(float)
    out["is_post_earnings_3d"] = ((days_since >= 0) & (days_since <= 3)).astype(float)
    if fwd_window_days is not None:
        out["earnings_in_fwd_window"] = ((days_to >= 0) & (days_to <= fwd_window_days)).astype(float)
    return out


def _universe() -> list[str]:
    from core.shared_universe.universe import shared_tickers
    return shared_tickers(eligible_only=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Fetch earnings calendar (historical + forward)")
    ap.add_argument("--refresh", action="store_true", help="fetch + overwrite the calendar")
    ap.add_argument("--missing-only", action="store_true", help="keep existing, only fetch tickers not yet present")
    ap.add_argument("--tickers", nargs="*", help="explicit tickers (default: whole universe)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--throttle", type=float, default=0.0, help="seconds between requests (avoid rate limits)")
    args = ap.parse_args()

    if not (args.refresh or args.missing_only):
        cal = load_earnings_calendar()
        print(f"earnings calendar: {len(cal)} rows, {cal['ticker'].nunique() if len(cal) else 0} tickers")
        return

    tickers = [t.upper() for t in args.tickers] if args.tickers else _universe()
    fetch_earnings_calendar(tickers, max_workers=args.workers, throttle=args.throttle, merge=args.missing_only)


if __name__ == "__main__":
    main()
