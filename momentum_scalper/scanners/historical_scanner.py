"""Historical premarket scanner reconstruction."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_scalper.configs.settings import MINUTE_BARS_DIR, NEWS_DIR, SCANNER_SNAPSHOTS_DIR, ScannerConfig, ensure_data_dirs
from momentum_scalper.utils.io import add_session_columns, normalize_timestamp_column, write_parquet


SNAPSHOT_FIELDS = ["timestamp", "ticker", "scanner_rank", "gap_pct", "premarket_volume", "rvol", "dist_to_hod", "float", "news_flag"]


def load_bars_for_day(day: str, bars_dir: Path = MINUTE_BARS_DIR) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    month = pd.Timestamp(day).strftime("%Y-%m")
    for path in bars_dir.glob(f"ticker=*/{month}.parquet"):
        df = pd.read_parquet(path)
        df = add_session_columns(df)
        part = df[df["date"].eq(pd.Timestamp(day).strftime("%Y-%m-%d"))].copy()
        if not part.empty:
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_news_flags(day: str, news_dir: Path = NEWS_DIR) -> set[str]:
    path = news_dir / f"{pd.Timestamp(day):%Y-%m-%d}.parquet"
    if not path.exists():
        return set()
    news = pd.read_parquet(path)
    return set(news.get("ticker", pd.Series(dtype=str)).astype(str).str.upper())


def reconstruct_premarket_scanner(
    bars: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    news_tickers: set[str] | None = None,
    config: ScannerConfig = ScannerConfig(),
) -> pd.DataFrame:
    bars = add_session_columns(bars)
    if bars.empty:
        return pd.DataFrame(columns=SNAPSHOT_FIELDS)
    news_tickers = news_tickers or set()
    meta = metadata.copy() if metadata is not None and not metadata.empty else pd.DataFrame(columns=["ticker", "float"])
    if "ticker" in meta.columns:
        meta["ticker"] = meta["ticker"].astype(str).str.upper()

    rows: list[pd.DataFrame] = []
    for ts, snap in bars[bars["is_premarket"]].groupby("timestamp", sort=True):
        hist = bars[(bars["timestamp"] <= ts) & (bars["is_premarket"])]
        grouped = hist.groupby("ticker", as_index=False).agg(
            premarket_volume=("volume", "sum"),
            premarket_high=("high", "max"),
            last_price=("close", "last"),
            first_price=("open", "first"),
            avg_volume=("volume", "mean"),
        )
        prior_close = bars[bars["timestamp"] < hist["timestamp"].min()].groupby("ticker")["close"].last()
        grouped["prior_close"] = grouped["ticker"].map(prior_close)
        grouped["prior_close"] = grouped["prior_close"].fillna(grouped["first_price"])
        grouped["gap_pct"] = (grouped["last_price"] / grouped["prior_close"] - 1.0) * 100.0
        grouped["rvol"] = grouped["premarket_volume"] / grouped["avg_volume"].replace(0, np.nan)
        grouped["dist_to_hod"] = (grouped["premarket_high"] / grouped["last_price"] - 1.0) * 100.0
        grouped["news_flag"] = grouped["ticker"].isin(news_tickers)
        if not meta.empty:
            grouped = grouped.merge(meta[["ticker", "float"]].drop_duplicates("ticker"), on="ticker", how="left")
        else:
            grouped["float"] = np.nan

        filt = (
            (grouped["gap_pct"] > config.gap_pct_min)
            & (grouped["premarket_volume"] > config.premarket_volume_min)
            & (grouped["rvol"] > config.relative_volume_min)
            & (grouped["last_price"].between(config.min_price, config.max_price))
            & (grouped["float"].isna() | (grouped["float"] < config.max_float))
        )
        if config.require_news:
            filt &= grouped["news_flag"]
        out = grouped[filt].copy()
        if out.empty:
            continue
        out["timestamp"] = ts
        out["scanner_rank"] = out[["gap_pct", "rvol"]].rank(ascending=False, method="first").mean(axis=1).rank(method="first").astype(int)
        rows.append(out[SNAPSHOT_FIELDS])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=SNAPSHOT_FIELDS)


def build_daily_snapshot(day: str, output_dir: Path = SCANNER_SNAPSHOTS_DIR) -> Path:
    ensure_data_dirs()
    bars = load_bars_for_day(day)
    snapshot = reconstruct_premarket_scanner(bars, news_tickers=load_news_flags(day))
    return write_parquet(snapshot, output_dir / f"{pd.Timestamp(day):%Y-%m-%d}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical scanner snapshot")
    parser.add_argument("--day", required=True)
    args = parser.parse_args()
    path = build_daily_snapshot(args.day)
    print(f"wrote scanner snapshot to {path}")


if __name__ == "__main__":
    main()
