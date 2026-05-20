"""Breakout outcome labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from momentum_scalper.configs.settings import FEATURES_PATH, LABELS_PATH, MINUTE_BARS_DIR, ensure_data_dirs
from momentum_scalper.scanners.historical_scanner import load_bars_for_day
from momentum_scalper.utils.io import add_session_columns, normalize_timestamp_column, write_parquet


LABEL_COLUMNS = [
    "timestamp",
    "ticker",
    "hit_2R_before_minus_1R_within_15m",
    "max_forward_return_5m",
    "max_forward_return_15m",
    "max_forward_return_30m",
    "failed_breakout",
    "halt_after_entry",
    "rug_pull",
    "MFE",
    "MAE",
    "time_to_peak",
    "time_to_fail",
]


def _forward_window(bars: pd.DataFrame, ticker: str, ts: pd.Timestamp, minutes: int) -> pd.DataFrame:
    end = ts + pd.Timedelta(minutes=minutes)
    return bars[(bars["ticker"].eq(ticker)) & (bars["timestamp"] > ts) & (bars["timestamp"] <= end)].sort_values("timestamp")


def label_feature_rows(features: pd.DataFrame, bars_by_day: dict[str, pd.DataFrame], risk_pct: float = 1.0) -> pd.DataFrame:
    features = normalize_timestamp_column(features)
    rows: list[dict] = []
    for _, row in features.iterrows():
        ts = row["timestamp"]
        ticker = row["ticker"]
        day = ts.tz_convert("America/New_York").strftime("%Y-%m-%d")
        bars = bars_by_day.get(day, pd.DataFrame())
        hist = bars[(bars["ticker"].eq(ticker)) & (bars["timestamp"] <= ts)].sort_values("timestamp")
        fwd30 = _forward_window(bars, ticker, ts, 30)
        if hist.empty or fwd30.empty:
            continue
        entry = float(hist.iloc[-1]["close"])
        if entry <= 0:
            continue
        risk = entry * (risk_pct / 100.0)
        returns = (fwd30["close"] / entry - 1.0) * 100.0
        highs = (fwd30["high"] / entry - 1.0) * 100.0
        lows = (fwd30["low"] / entry - 1.0) * 100.0
        hit_tp = highs >= (2.0 * risk_pct)
        hit_stop = lows <= (-1.0 * risk_pct)
        first_tp = hit_tp.idxmax() if hit_tp.any() else None
        first_stop = hit_stop.idxmax() if hit_stop.any() else None
        hit_2r = bool(first_tp is not None and (first_stop is None or fwd30.loc[first_tp, "timestamp"] <= fwd30.loc[first_stop, "timestamp"]) and fwd30.loc[first_tp, "timestamp"] <= ts + pd.Timedelta(minutes=15))
        mae = float(lows.min())
        mfe = float(highs.max())
        rows.append(
            {
                "timestamp": ts,
                "ticker": ticker,
                "hit_2R_before_minus_1R_within_15m": hit_2r,
                "max_forward_return_5m": float(_forward_window(bars, ticker, ts, 5)["high"].max() / entry - 1.0) * 100.0 if not _forward_window(bars, ticker, ts, 5).empty else np.nan,
                "max_forward_return_15m": float(_forward_window(bars, ticker, ts, 15)["high"].max() / entry - 1.0) * 100.0 if not _forward_window(bars, ticker, ts, 15).empty else np.nan,
                "max_forward_return_30m": mfe,
                "failed_breakout": bool(mae <= -risk_pct and mfe < risk_pct),
                "halt_after_entry": False,
                "rug_pull": bool(mae <= -2.0 * risk_pct),
                "MFE": mfe,
                "MAE": mae,
                "time_to_peak": float((fwd30.loc[highs.idxmax(), "timestamp"] - ts).total_seconds() / 60.0),
                "time_to_fail": float((fwd30.loc[lows.idxmin(), "timestamp"] - ts).total_seconds() / 60.0),
            }
        )
    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def build_labels(features_path: Path = FEATURES_PATH, output: Path = LABELS_PATH) -> pd.DataFrame:
    ensure_data_dirs()
    features = pd.read_parquet(features_path) if features_path.exists() else pd.DataFrame()
    if features.empty:
        labels = pd.DataFrame(columns=LABEL_COLUMNS)
        write_parquet(labels, output)
        return labels
    features = normalize_timestamp_column(features)
    days = sorted({ts.tz_convert("America/New_York").strftime("%Y-%m-%d") for ts in features["timestamp"]})
    bars_by_day = {day: add_session_columns(load_bars_for_day(day, MINUTE_BARS_DIR)) for day in days}
    labels = label_feature_rows(features, bars_by_day)
    write_parquet(labels, output)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build forward breakout labels")
    parser.add_argument("--features", type=Path, default=FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=LABELS_PATH)
    args = parser.parse_args()
    df = build_labels(args.features, args.output)
    print(f"wrote {len(df):,} labels to {args.output}")


if __name__ == "__main__":
    main()
