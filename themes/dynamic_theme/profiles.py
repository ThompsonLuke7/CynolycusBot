"""Company profiles for the theme pipeline: business description + Yahoo industry.

The shared news profile table (``TICKER_PROFILES_PATH``) is a 2026-06-05 snapshot
of 1,338 names. The theme universe has since grown to ~2,900, which left 61% of
tickers with no business description; those names were embedded from headlines
and price co-movement alone, and that is where the catch-all clusters came from.

Profiles fetched here go to a theme-owned SUPPLEMENT file. The shared table is
left alone on purpose: the live news-catalyst scorer reads sector / marketCap /
float / beta from it as model inputs, so widening it would change live scoring.

A profile is a current snapshot, not point-in-time. It is used only to describe
and group companies, never as a dated model feature.

Usage (resumable; each call fetches at most ``--max`` tickers, ~0.7s each):
    .venv/bin/python -m themes.dynamic_theme.profiles --max 400
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

from themes.dynamic_theme.config import (
    PROFILE_RETRY_DAYS,
    TICKER_PROFILES_PATH,
    TICKER_PROFILES_SUPPLEMENT_PATH,
)

logger = logging.getLogger(__name__)

_MIN_SUMMARY_CHARS = 30


def _read(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["ticker"])
    df = pd.read_parquet(path)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def _has_data(df: pd.DataFrame) -> pd.Series:
    summary = df["longBusinessSummary"].fillna("").astype(str).str.len() >= _MIN_SUMMARY_CHARS
    industry = df["industry"].fillna("").astype(str).str.strip().ne("")
    return summary | industry


def load_profiles() -> pd.DataFrame:
    """Shared table + theme supplement, one row per ticker.

    A row with a description or industry beats an empty one; between two usable
    rows the shared table wins; between two empty rows the latest attempt wins
    (so an empty shared row cannot hide a fresh "tried, nothing there" marker).
    """
    frames = [
        f.assign(_src=i)
        for i, f in enumerate((_read(TICKER_PROFILES_PATH), _read(TICKER_PROFILES_SUPPLEMENT_PATH)))
        if not f.empty
    ]
    if not frames:
        return pd.DataFrame(columns=["ticker", "longBusinessSummary", "sector", "industry", "quoteType"])
    out = pd.concat(frames, ignore_index=True)
    for col in ("longBusinessSummary", "sector", "industry", "quoteType", "snapshot_date"):
        if col not in out.columns:
            out[col] = None
    out["_empty"] = ~_has_data(out)
    out["_tried"] = pd.to_datetime(out["snapshot_date"], errors="coerce", utc=True)
    # usable before empty; then shared before supplement for usable rows, newest first for empty ones
    tried_ns = out["_tried"].map(lambda t: t.value if pd.notna(t) else 0)
    out["_order"] = np.where(out["_empty"], -tried_ns, out["_src"])
    out = out.sort_values(["_empty", "_order"], kind="stable").drop_duplicates("ticker", keep="first")
    return out.drop(columns=["_src", "_empty", "_tried", "_order"]).reset_index(drop=True)


def industry_by_ticker() -> dict[str, str]:
    """{ticker: Yahoo industry} for operating companies with a known industry."""
    p = load_profiles()
    ind = p["industry"].fillna("").astype(str).str.strip()
    keep = ind.ne("") & ind.str.lower().ne("none")
    return dict(zip(p.loc[keep, "ticker"], ind[keep]))


def country_by_ticker() -> dict[str, str]:
    """{ticker: Yahoo country of domicile} where known."""
    p = load_profiles()
    if "country" not in p.columns:
        return {}
    c = p["country"].fillna("").astype(str).str.strip()
    return dict(zip(p.loc[c.ne(""), "ticker"], c[c.ne("")]))


def tickers_needing_profiles(tickers: list[str], *, now: pd.Timestamp | None = None) -> list[str]:
    """Tickers with no profile row, or an empty one last tried > PROFILE_RETRY_DAYS ago."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize(None) if now.tzinfo else now
    p = load_profiles()
    have = set(p["ticker"])
    tried = pd.to_datetime(p["snapshot_date"], errors="coerce", utc=True).dt.tz_localize(None)
    stale_empty = set(p.loc[~_has_data(p) & (tried < now - pd.Timedelta(days=PROFILE_RETRY_DAYS)), "ticker"])
    wanted = [str(t).upper() for t in tickers]
    return sorted({t for t in wanted if t not in have or t in stale_empty})


def fetch_missing_profiles(tickers: list[str], *, max_tickers: int, chunk: int = 50) -> int:
    """Fetch up to ``max_tickers`` missing profiles into the supplement file.

    Written after every chunk, so an interrupted run keeps what it fetched.
    Tickers Yahoo returns nothing for get an empty placeholder row (so they are
    not re-requested every run) and are retried after PROFILE_RETRY_DAYS.
    Returns the number of tickers that came back with a usable profile.
    """
    from signals.news.sources import fetch_yfinance_profiles

    todo = tickers_needing_profiles(tickers)[: int(max_tickers)]
    if not todo:
        logger.info("Profiles: nothing to fetch")
        return 0
    logger.info("Profiles: fetching %d (of %d missing)", len(todo), len(tickers_needing_profiles(tickers)))
    got = 0
    for i in range(0, len(todo), chunk):
        batch = todo[i:i + chunk]
        fetched = fetch_yfinance_profiles(batch, progress_every=10_000)
        stamp = pd.Timestamp.now(tz="UTC").normalize()
        returned = set(fetched["ticker"].astype(str).str.upper()) if not fetched.empty else set()
        placeholders = pd.DataFrame({"ticker": [t for t in batch if t not in returned], "snapshot_date": stamp})
        new = pd.concat([f for f in (fetched, placeholders) if not f.empty], ignore_index=True)
        new["ticker"] = new["ticker"].astype(str).str.upper()
        prior = _read(TICKER_PROFILES_SUPPLEMENT_PATH)
        merged = pd.concat([prior[~prior["ticker"].isin(new["ticker"])], new], ignore_index=True)
        tmp = TICKER_PROFILES_SUPPLEMENT_PATH.with_suffix(".tmp.parquet")
        merged.to_parquet(tmp, index=False)
        os.replace(tmp, TICKER_PROFILES_SUPPLEMENT_PATH)
        got += len(returned)
        logger.info("Profiles: %d/%d done, %d usable so far", min(i + chunk, len(todo)), len(todo), got)
    return got


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=400, help="max tickers to fetch this call")
    args = ap.parse_args()
    from themes.dynamic_theme.pipeline import _get_tickers

    tickers = _get_tickers()
    before = len(tickers_needing_profiles(tickers))
    got = fetch_missing_profiles(tickers, max_tickers=args.max)
    after = len(tickers_needing_profiles(tickers))
    print(f"profiles: universe {len(tickers)}, needed {before} -> {after}, usable fetched {got}")


if __name__ == "__main__":
    main()
