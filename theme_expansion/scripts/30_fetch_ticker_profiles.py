"""
Fetch Yahoo sector/industry/business-summary profiles for tickers that are in
the momentum_expansion candidate pool but NOT yet in the theme universe.

Used to scaffold theme assignment for the universe expansion. Output is a
concrete classification source (Yahoo's ~148-industry taxonomy + summary) that
we then map onto the curated theme taxonomy by hand.

Resumable: each ticker is cached as JSON under data/ticker_profiles_cache/ so a
429-interrupted run can be re-run cheaply.

Usage:
    python scripts/30_fetch_ticker_profiles.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))  # repo root, for momentum_expansion import

import config as tcfg  # theme_expansion/config.py
from momentum_expansion.data.universe import get_candidate_pool

CACHE_DIR = ROOT / "data" / "ticker_profiles_cache"
OUT_CSV = ROOT / "data" / "ticker_profiles_new.csv"

FIELDS = ["sector", "sectorKey", "industry", "industryKey", "quoteType", "longBusinessSummary"]
MAX_RETRIES = 5
BASE_DELAY = 2.0
REQUEST_DELAY = 0.5


def new_tickers() -> list[str]:
    members = tcfg.load_theme_memberships()
    theme_tickers = {tcfg.clean_ticker(t) for t in members["ticker"]} - {""}
    mom = set(get_candidate_pool())
    return sorted(mom - theme_tickers)


def fetch_one(ticker: str) -> dict:
    cache = CACHE_DIR / f"{ticker}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    for attempt in range(MAX_RETRIES):
        try:
            info = yf.Ticker(ticker).get_info() or {}
            rec = {"ticker": ticker}
            for f in FIELDS:
                v = info.get(f)
                if f == "longBusinessSummary" and isinstance(v, str):
                    v = v[:400]
                rec[f] = v
            cache.write_text(json.dumps(rec))
            return rec
        except Exception as exc:  # noqa: BLE001 - yfinance raises bare Exceptions
            if attempt == MAX_RETRIES - 1:
                print(f"warning: giving up on {ticker} ({exc})")
                rec = {"ticker": ticker, **{f: None for f in FIELDS}}
                cache.write_text(json.dumps(rec))
                return rec
            delay = BASE_DELAY * (2 ** attempt) + random.uniform(0, BASE_DELAY)
            print(f"retry {ticker} ({attempt + 1}/{MAX_RETRIES}) after {delay:.1f}s: {exc}")
            time.sleep(delay)
    return {"ticker": ticker, **{f: None for f in FIELDS}}


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tickers = new_tickers()
    print(f"fetching profiles for {len(tickers)} new tickers")
    rows = []
    for i, t in enumerate(tickers, 1):
        if not (CACHE_DIR / f"{t}.json").exists():
            time.sleep(REQUEST_DELAY)
        rows.append(fetch_one(t))
        if i % 25 == 0 or i == len(tickers):
            print(f"  {i}/{len(tickers)}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    got = df["sector"].notna().sum()
    print(f"saved {len(df)} rows -> {OUT_CSV} ({got} with sector, {len(df) - got} missing)")


if __name__ == "__main__":
    main()
