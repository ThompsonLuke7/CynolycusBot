"""Build context backtest universe, timestamp rows, and forward labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from meta_context.config import (
    CONTEXT_BACKTEST_TIMESTAMPS_PATH,
    CONTEXT_BACKTEST_UNIVERSE_PATH,
    CONTEXT_FORWARD_LABELS_PATH,
    ensure_dirs,
)


def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "timestamp" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "timestamp"})
        else:
            for candidate in ("time", "t", "date"):
                if candidate in out.columns:
                    out = out.rename(columns={candidate: "timestamp"})
                    break
    if "timestamp" not in out.columns:
        raise ValueError("Bars must have a timestamp column or DatetimeIndex.")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def build_context_backtest_universe(
    *,
    universe_csv: Path | str | None = None,
    include_swing: bool = True,
    include_momentum: bool = True,
    include_funds: bool = False,
    include_blacklist: bool = False,
    output_path: Path | str = CONTEXT_BACKTEST_UNIVERSE_PATH,
    limit: int | None = None,
) -> pd.DataFrame:
    """Build a combined context universe from swing and momentum-expansion sources."""
    ensure_dirs()
    from multi_ticker_swing.config.pipeline_config import TRADING_BLACKLIST, UNIVERSE_CSV

    frames: list[pd.DataFrame] = []
    if include_swing:
        path = Path(universe_csv) if universe_csv else UNIVERSE_CSV
        swing = pd.read_csv(path)
        if "cap_bucket" in swing.columns and "market_cap_bucket" not in swing.columns:
            swing = swing.rename(columns={"cap_bucket": "market_cap_bucket"})
        if "asset_type" in swing.columns and "type" not in swing.columns:
            swing = swing.rename(columns={"asset_type": "type"})
        for col, default in (
            ("sector", "Unknown"),
            ("market_cap_bucket", "Unknown"),
            ("type", "Stock"),
            ("notes", "multi_ticker_swing universe"),
        ):
            if col not in swing.columns:
                swing[col] = default
        swing["in_multi_ticker_swing"] = True
        swing["in_momentum_expansion"] = False
        frames.append(swing[["ticker", "sector", "market_cap_bucket", "type", "notes", "in_multi_ticker_swing", "in_momentum_expansion"]])

    if include_momentum:
        from momentum_expansion.data.universe import load_candidate_metadata

        momentum = load_candidate_metadata()
        if not momentum.empty:
            for col, default in (
                ("sector", "Unknown"),
                ("market_cap_bucket", "Unknown"),
                ("type", "Stock"),
                ("notes", "momentum_expansion universe"),
            ):
                if col not in momentum.columns:
                    momentum[col] = default
            momentum["in_multi_ticker_swing"] = False
            momentum["in_momentum_expansion"] = True
            frames.append(momentum[["ticker", "sector", "market_cap_bucket", "type", "notes", "in_multi_ticker_swing", "in_momentum_expansion"]])

    if not frames:
        df = pd.DataFrame(columns=["ticker", "sector", "market_cap_bucket", "type", "notes"])
    else:
        df = pd.concat(frames, ignore_index=True)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    df = df.loc[df["ticker"].ne("")].copy()
    if not include_funds and "type" in df.columns:
        asset_type = df["type"].astype(str).str.upper()
        df = df.loc[~asset_type.isin({"ETF", "ETN", "FUND", "INDEX"})].copy()
    if not include_blacklist:
        blacklist = {x.upper().replace("$", "") for x in TRADING_BLACKLIST}
        df = df.loc[~df["ticker"].isin(blacklist)].copy()

    def first_known(values: pd.Series, default: str = "Unknown") -> str:
        for value in values:
            text = str(value).strip()
            if text and text.lower() != "nan" and text != "Unknown":
                return text
        return default

    rows = []
    for ticker, group in df.groupby("ticker", sort=True):
        in_swing = bool(group["in_multi_ticker_swing"].any())
        in_momentum = bool(group["in_momentum_expansion"].any())
        rows.append(
            {
                "ticker": ticker,
                "sector": first_known(group["sector"]),
                "market_cap_bucket": first_known(group["market_cap_bucket"]),
                "type": "ETF" if group["type"].astype(str).str.upper().eq("ETF").any() else first_known(group["type"], "Stock"),
                "notes": first_known(group["notes"], ""),
                "in_multi_ticker_swing": in_swing,
                "in_momentum_expansion": in_momentum,
                "universe_sources": "|".join(
                    source
                    for source, enabled in (
                        ("multi_ticker_swing", in_swing),
                        ("momentum_expansion", in_momentum),
                    )
                    if enabled
                ),
            }
        )
    df = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    if limit:
        df = df.head(int(limit)).copy()
    df.to_csv(output_path, index=False)
    return df


def load_cached_bars_for_universe(
    universe: pd.DataFrame,
    *,
    raw_dir: Path | str | None = None,
    include_momentum_fallback: bool = True,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load cached context bars, preferring swing 30m and falling back to momentum 4h."""
    from multi_ticker_swing.config.pipeline_config import RAW_30M_DIR
    from momentum_expansion.config.momentum_config import RAW_4H_DIR

    base = Path(raw_dir) if raw_dir else RAW_30M_DIR
    rows = []
    start_ts = pd.Timestamp(start, tz="UTC") if start else None
    end_ts = pd.Timestamp(end, tz="UTC") if end else None
    for ticker in universe["ticker"].astype(str).str.upper():
        path = base / f"{ticker}.parquet"
        bar_timeframe = "30m"
        if not path.exists() and include_momentum_fallback and raw_dir is None:
            path = RAW_4H_DIR / f"{ticker}.parquet"
            bar_timeframe = "4h"
        if not path.exists():
            continue
        df = _ensure_timestamp_column(pd.read_parquet(path))
        if start_ts is not None:
            df = df.loc[df["timestamp"] >= start_ts]
        if end_ts is not None:
            df = df.loc[df["timestamp"] <= end_ts]
        if df.empty:
            continue
        df["ticker"] = ticker
        df["bar_timeframe"] = bar_timeframe
        keep = [c for c in ("timestamp", "ticker", "open", "high", "low", "close", "volume") if c in df.columns]
        keep.append("bar_timeframe")
        rows.append(df[keep])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["timestamp", "ticker", "close"])


def load_cached_30m_bars_for_universe(
    universe: pd.DataFrame,
    *,
    raw_dir: Path | str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias; now loads 30m with momentum 4h fallback by default."""
    return load_cached_bars_for_universe(universe, raw_dir=raw_dir, start=start, end=end)


def build_context_backtest_timestamps(
    bars: pd.DataFrame,
    *,
    output_path: Path | str = CONTEXT_BACKTEST_TIMESTAMPS_PATH,
) -> pd.DataFrame:
    """Create timestamp/ticker rows for event/news feature generation."""
    ensure_dirs()
    if bars.empty:
        out = pd.DataFrame(columns=["timestamp", "ticker"])
    else:
        out = bars[["timestamp", "ticker"]].drop_duplicates().sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    out.to_parquet(output_path, index=False)
    return out


def build_forward_performance_labels(
    bars: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5, 10),
    expansion_threshold: float = 0.10,
    output_path: Path | str = CONTEXT_FORWARD_LABELS_PATH,
) -> pd.DataFrame:
    """Build forward bar-return labels for every timestamp/ticker row."""
    ensure_dirs()
    if bars.empty:
        out = pd.DataFrame()
        out.to_parquet(output_path, index=False)
        return out
    df = bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace("$", "", regex=False)
    df = df.dropna(subset=["timestamp", "close"]).sort_values(["ticker", "timestamp"])
    rows = []
    for ticker, group in df.groupby("ticker", sort=True):
        close = group["close"].astype(float).to_numpy()
        timestamps = group["timestamp"].to_numpy()
        for i, ts in enumerate(timestamps):
            row = {"timestamp": pd.Timestamp(ts), "ticker": ticker}
            future = close[i + 1 : i + 1 + max(horizons)]
            if len(future) == 0 or close[i] == 0:
                continue
            rets = future / close[i] - 1.0
            for horizon in horizons:
                row[f"forward_{horizon}bar_return"] = float(rets[horizon - 1]) if len(rets) >= horizon else np.nan
            row["max_forward_return"] = float(np.nanmax(rets))
            row["max_drawdown"] = float(np.nanmin(rets))
            row["expansion_label"] = float(row["max_forward_return"] >= expansion_threshold)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(output_path, index=False)
    return out


def build_all_context_backtest_inputs(
    *,
    universe_csv: Path | str | None = None,
    raw_dir: Path | str | None = None,
    start: str | None = None,
    end: str | None = None,
    include_swing: bool = True,
    include_momentum: bool = True,
    include_funds: bool = False,
    include_blacklist: bool = False,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    universe = build_context_backtest_universe(
        universe_csv=universe_csv,
        include_swing=include_swing,
        include_momentum=include_momentum,
        include_funds=include_funds,
        include_blacklist=include_blacklist,
        limit=limit,
    )
    bars = load_cached_bars_for_universe(universe, raw_dir=raw_dir, start=start, end=end)
    timestamps = build_context_backtest_timestamps(bars)
    labels = build_forward_performance_labels(bars)
    return universe, timestamps, labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Build news/events backtest universe, timestamps, and forward labels.")
    parser.add_argument("--universe-csv", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--swing-only", action="store_true")
    parser.add_argument("--momentum-only", action="store_true")
    parser.add_argument("--include-funds", action="store_true")
    parser.add_argument("--include-blacklist", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    include_swing = not args.momentum_only
    include_momentum = not args.swing_only
    universe, timestamps, labels = build_all_context_backtest_inputs(
        universe_csv=args.universe_csv,
        raw_dir=args.raw_dir,
        start=args.start,
        end=args.end,
        include_swing=include_swing,
        include_momentum=include_momentum,
        include_funds=args.include_funds,
        include_blacklist=args.include_blacklist,
        limit=args.limit,
    )
    print(f"universe={len(universe)} timestamps={len(timestamps)} labels={len(labels)}")
    print(f"universe_path={CONTEXT_BACKTEST_UNIVERSE_PATH}")
    print(f"timestamps_path={CONTEXT_BACKTEST_TIMESTAMPS_PATH}")
    print(f"labels_path={CONTEXT_FORWARD_LABELS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
