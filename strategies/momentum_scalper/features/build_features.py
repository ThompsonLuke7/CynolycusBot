"""Feature generator for scanner states."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.momentum_scalper.configs.settings import FEATURES_PATH, MINUTE_BARS_DIR, SCANNER_SNAPSHOTS_DIR, ensure_data_dirs
from strategies.momentum_scalper.scanners.historical_scanner import load_bars_for_day
from strategies.momentum_scalper.utils.io import add_session_columns, normalize_timestamp_column, write_parquet


def _slope(values: pd.Series) -> float:
    y = values.dropna().to_numpy(dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def _ticker_features(hist: pd.DataFrame, row: pd.Series) -> dict:
    last = hist.iloc[-1]
    close = float(last["close"])
    volume = pd.to_numeric(hist["volume"], errors="coerce")
    avg_vol = float(volume.rolling(20, min_periods=1).mean().iloc[-1])
    vwap = float(last.get("vwap", np.nan)) if pd.notna(last.get("vwap", np.nan)) else close
    high_so_far = float(hist["high"].max())
    low_so_far = float(hist["low"].min())
    green = (hist["close"] > hist["open"]).astype(int)
    consecutive_green = int(green.iloc[::-1].cumprod().sum())
    returns = hist["close"].pct_change()
    spread_pct = float((hist["high"].iloc[-1] - hist["low"].iloc[-1]) / close * 100.0) if close else np.nan
    premarket = hist[hist["is_premarket"]]
    rth = hist[hist["is_rth"]]
    return {
        "timestamp": row["timestamp"],
        "ticker": row["ticker"],
        "gap_pct": row.get("gap_pct", np.nan),
        "premarket_volume": row.get("premarket_volume", np.nan),
        "premarket_high_break": float(close >= high_so_far),
        "premarket_range": (premarket["high"].max() - premarket["low"].min()) / close if close and not premarket.empty else np.nan,
        "rvol": row.get("rvol", np.nan),
        "volume_spike_ratio": float(volume.iloc[-1] / avg_vol) if avg_vol else np.nan,
        "volume_acceleration": float(volume.diff().tail(3).mean()) if len(volume) >= 3 else 0.0,
        "dollar_volume": float(close * volume.iloc[-1]),
        "dist_to_hod": row.get("dist_to_hod", (high_so_far / close - 1.0) * 100.0 if close else np.nan),
        "float": row.get("float", np.nan),
        "dist_to_vwap": (close / vwap - 1.0) * 100.0 if vwap else np.nan,
        "opening_range_break": float(not rth.empty and close >= rth.head(5)["high"].max()),
        "micro_pullback": float(len(hist) >= 3 and hist["low"].iloc[-1] > hist["low"].iloc[-2] and hist["close"].iloc[-1] > hist["open"].iloc[-1]),
        "bull_flag_tightness": float(hist["close"].tail(5).std() / close) if close else np.nan,
        "flat_top_breakout": float(len(hist) >= 5 and close >= hist["high"].tail(5).max()),
        "spread_pct": spread_pct,
        "trade_count": last.get("trade_count", np.nan),
        "liquidity_score": float(np.log1p(close * volume.iloc[-1]) / max(spread_pct, 0.01)),
        "consecutive_green_candles": consecutive_green,
        "trend_slope": _slope(hist["close"].tail(10)),
        "breakout_velocity": float(returns.tail(3).sum() * 100.0),
        "news_age_minutes": 0.0 if bool(row.get("news_flag", False)) else np.nan,
        "catalyst_type": "news" if bool(row.get("news_flag", False)) else "none",
        "halt_count": row.get("halt_count", 0),
        "runner_rank": row.get("scanner_rank", np.nan),
        "sector_rank": np.nan,
        "relative_volume_rank": np.nan,
    }


def build_features_for_snapshot(snapshot: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    snapshot = normalize_timestamp_column(snapshot)
    bars = add_session_columns(bars)
    rows: list[dict] = []
    for _, row in snapshot.iterrows():
        hist = bars[(bars["ticker"].eq(row["ticker"])) & (bars["timestamp"] <= row["timestamp"])].sort_values("timestamp")
        if not hist.empty:
            rows.append(_ticker_features(hist, row))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["relative_volume_rank"] = df.groupby("timestamp")["rvol"].rank(ascending=False, method="dense")
    return df


def build_features(snapshot_paths: list[Path] | None = None, output: Path = FEATURES_PATH) -> pd.DataFrame:
    ensure_data_dirs()
    paths = snapshot_paths or sorted(SCANNER_SNAPSHOTS_DIR.glob("*.parquet"))
    frames: list[pd.DataFrame] = []
    for path in paths:
        day = path.stem
        snapshot = pd.read_parquet(path)
        bars = load_bars_for_day(day, MINUTE_BARS_DIR)
        part = build_features_for_snapshot(snapshot, bars)
        if not part.empty:
            frames.append(part)
    features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_parquet(features, output)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scalper features")
    parser.add_argument("--snapshots", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=FEATURES_PATH)
    args = parser.parse_args()
    df = build_features(args.snapshots, args.output)
    print(f"wrote {len(df):,} feature rows to {args.output}")


if __name__ == "__main__":
    main()
