from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# Politeness/robustness for yfinance, which is frequently rate-limited (HTTP 429)
# and offers no built-in retry. Tunable via CLI flags in main().
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY = 2.0   # seconds; backoff is base * 2**attempt + jitter
DEFAULT_REQUEST_DELAY = 0.5  # seconds slept between live downloads

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    BENCHMARK_TICKERS,
    DAILY_BARS_PATH,
    END_DATE,
    FILTER_EXCEPTIONS,
    MIN_AVG_DOLLAR_VOLUME_20D,
    MIN_MARKET_CAP,
    MIN_PRICE,
    RAW_DAILY_DIR,
    START_DATE,
    THEME_FILTER_OVERRIDES,
    UNIVERSE_FILTER_PATH,
    clean_ticker,
    ensure_dirs,
    load_theme_memberships,
)


def load_tickers() -> list[str]:
    try:
        from shared_universe.universe import shared_tickers

        tickers = {clean_ticker(t) for t in shared_tickers(eligible_only=True, rebuild=False)}
    except Exception:
        memberships = load_theme_memberships()
        tickers = {clean_ticker(t) for t in memberships["ticker"].dropna()}
    tickers.update(BENCHMARK_TICKERS)
    return sorted(t for t in tickers if t)


def normalize_download(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    out = df.reset_index()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    rename = {"date": "date", "adj_close": "adj_close"}
    out = out.rename(columns=rename)
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = ticker
    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    return out[[c for c in keep if c in out.columns]].sort_values("date")


def _download_with_retry(
    ticker: str,
    start: str,
    end: str | None,
    *,
    max_retries: int,
    base_delay: float,
) -> pd.DataFrame:
    """yf.download with exponential backoff + jitter.

    yfinance raises (rather than returning empty) on rate limits / transient
    network errors, which would otherwise abort the whole universe run. We retry
    a bounded number of times and, on exhaustion, return an empty frame so the
    caller can skip the ticker instead of crashing.
    """
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            return normalize_download(raw, ticker)
        except Exception as exc:  # noqa: BLE001 - yfinance raises bare Exceptions
            if attempt == max_retries - 1:
                print(f"warning: giving up on {ticker} after {max_retries} attempts ({exc})")
                return pd.DataFrame()
            delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
            print(f"retry {ticker} (attempt {attempt + 1}/{max_retries}) after {delay:.1f}s: {exc}")
            time.sleep(delay)
    return pd.DataFrame()


def download_one(
    ticker: str,
    start: str,
    end: str | None,
    force: bool,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    request_delay: float = DEFAULT_REQUEST_DELAY,
) -> pd.DataFrame:
    out_path = RAW_DAILY_DIR / f"{ticker}.parquet"
    start_ts = pd.Timestamp(start).tz_localize(None)
    end_ts = pd.Timestamp(end).tz_localize(None) if end else None
    target_last_date = end_ts - pd.Timedelta(days=1) if end_ts is not None else None
    cached = None
    if out_path.exists() and not force:
        cached = pd.read_parquet(out_path)
        if not cached.empty and "date" in cached.columns:
            cached["date"] = pd.to_datetime(cached["date"]).dt.tz_localize(None)
            cached = cached.loc[cached["date"] >= start_ts].copy()
            last_date = cached["date"].max()
            if target_last_date is None or (pd.notna(last_date) and last_date >= target_last_date):
                return cached
            start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if request_delay:
        time.sleep(request_delay)
    df = _download_with_retry(
        ticker, start, end, max_retries=max_retries, base_delay=base_delay
    )
    if df.empty:
        if cached is not None and not cached.empty:
            return cached
        print(f"warning: no bars for {ticker}")
        return df
    if cached is not None and not cached.empty:
        df = pd.concat([cached, df], ignore_index=True)
        df = df.drop_duplicates(["date", "ticker"], keep="last").sort_values("date")
    df = df.loc[df["date"] >= start_ts].copy()
    df.to_parquet(out_path, index=False)
    return df


def market_cap_for(ticker: str) -> float | None:
    try:
        yticker = yf.Ticker(ticker)
        value = yticker.fast_info.get("market_cap")
    except Exception:
        return None
    if value is None:
        try:
            value = yticker.info.get("marketCap")
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_universe_filter(combined: pd.DataFrame) -> pd.DataFrame:
    df = combined.copy()
    df["px"] = df["adj_close"].fillna(df["close"])
    df["dollar_volume"] = df["px"] * df["volume"]
    df = df.sort_values(["ticker", "date"])
    df["avg_dollar_volume_20d"] = df.groupby("ticker")["dollar_volume"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    latest = df.groupby("ticker").tail(1).copy()
    shared_eligible: set[str] | None = None
    try:
        from shared_universe.universe import shared_tickers

        shared_eligible = set(shared_tickers(eligible_only=True, rebuild=False))
    except Exception:
        shared_eligible = None
    memberships = load_theme_memberships()
    asset_types = (
        memberships[["ticker", "asset_type"]]
        .drop_duplicates(subset=["ticker"])
        .set_index("ticker")["asset_type"]
    )
    ticker_themes = memberships.groupby("ticker")["theme"].agg(list)

    def thresholds_for(ticker: str) -> pd.Series:
        thresholds = {
            "min_price": MIN_PRICE,
            "min_avg_dollar_volume_20d": MIN_AVG_DOLLAR_VOLUME_20D,
            "min_market_cap": MIN_MARKET_CAP,
        }
        for theme in ticker_themes.get(ticker, []):
            override = THEME_FILTER_OVERRIDES.get(theme, {})
            for key, value in override.items():
                if key in thresholds:
                    thresholds[key] = min(thresholds[key], value)
        return pd.Series(thresholds)

    latest["asset_type"] = latest["ticker"].map(asset_types).fillna("stock")
    thresholds = latest["ticker"].apply(thresholds_for)
    latest = pd.concat([latest, thresholds], axis=1)
    latest["market_cap"] = float("nan") if shared_eligible is not None else latest["ticker"].map(market_cap_for)
    latest["is_exception"] = latest["ticker"].isin(FILTER_EXCEPTIONS)
    latest["passes_price"] = latest["px"] > latest["min_price"]
    latest["passes_adv20"] = latest["avg_dollar_volume_20d"] > latest["min_avg_dollar_volume_20d"]
    latest["market_cap_missing"] = latest["market_cap"].isna()
    latest["passes_market_cap"] = latest["asset_type"].eq("etf") | (
        latest["market_cap"].fillna(0.0) > latest["min_market_cap"]
    )
    if shared_eligible is not None:
        latest["passes_market_cap"] = latest["ticker"].isin(shared_eligible)
        latest["is_eligible"] = latest["ticker"].isin(shared_eligible)
    else:
        latest["is_eligible"] = latest["is_exception"] | (
            latest["passes_price"] & latest["passes_adv20"] & latest["passes_market_cap"]
        )
    out = latest[
        [
            "ticker",
            "asset_type",
            "date",
            "px",
            "avg_dollar_volume_20d",
            "market_cap",
            "min_price",
            "min_avg_dollar_volume_20d",
            "min_market_cap",
            "market_cap_missing",
            "is_exception",
            "passes_price",
            "passes_adv20",
            "passes_market_cap",
            "is_eligible",
        ]
    ].sort_values("ticker")
    out.to_csv(UNIVERSE_FILTER_PATH, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily bars for theme universe with yfinance.")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--base-delay", type=float, default=DEFAULT_BASE_DELAY)
    parser.add_argument("--request-delay", type=float, default=DEFAULT_REQUEST_DELAY)
    args = parser.parse_args()

    ensure_dirs()
    frames = []
    tickers = load_tickers()
    for i, ticker in enumerate(tickers, 1):
        df = download_one(
            ticker,
            args.start,
            args.end,
            args.force,
            max_retries=args.max_retries,
            base_delay=args.base_delay,
            request_delay=args.request_delay,
        )
        if not df.empty:
            frames.append(df)
        if i % 25 == 0 or i == len(tickers):
            print(f"downloaded/cache checked {i}/{len(tickers)}")

    if not frames:
        raise SystemExit("no daily bars downloaded")
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "ticker"])
    combined = combined.sort_values(["date", "ticker"])
    combined.to_parquet(DAILY_BARS_PATH, index=False)
    universe_filter = build_universe_filter(combined)
    print(f"saved {len(combined):,} rows -> {DAILY_BARS_PATH}")
    print(
        f"eligible universe: {int(universe_filter['is_eligible'].sum())}/{len(universe_filter)} "
        f"-> {UNIVERSE_FILTER_PATH}"
    )


if __name__ == "__main__":
    main()
