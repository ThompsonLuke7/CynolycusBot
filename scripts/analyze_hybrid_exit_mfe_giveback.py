from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze realized ATR capture versus max favorable ATR excursion (MFE) "
            "for the current hybrid exit policy."
        )
    )
    parser.add_argument(
        "--events",
        default="Data/inference/spy/10min/meta/meta_events_hybrid_exit_current.csv",
        help="CSV of entry/exit events for the baseline hybrid execution path.",
    )
    parser.add_argument(
        "--meta-matrix",
        default="Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts_live_2026_03_24.parquet",
        help="Parquet with 10min matrix including ATR.",
    )
    parser.add_argument(
        "--one-min-data",
        default="Data/raw/spy/spy_intraday_1min_live_2026_03_24.parquet",
        help="Raw 1min parquet used to measure intratrade MFE.",
    )
    parser.add_argument(
        "--trades-out",
        default="Data/inference/spy/10min/meta/hybrid_exit_mfe_giveback_trades.csv",
        help="Per-trade MFE/giveback CSV.",
    )
    parser.add_argument(
        "--summary-out",
        default="Data/inference/spy/10min/meta/hybrid_exit_mfe_giveback_summary.csv",
        help="Per-side summary CSV.",
    )
    parser.add_argument(
        "--thresholds-out",
        default="Data/inference/spy/10min/meta/hybrid_exit_mfe_giveback_thresholds.csv",
        help="Threshold-oriented summary CSV for candidate profit-protect rules.",
    )
    return parser.parse_args()


def _load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Events file is empty: {path}")
    required = {"timestamp", "symbol", "event", "price"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Events file missing required columns: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _load_meta_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index(drop=True)
    if "timestamp" not in df.columns:
        raise ValueError(f"Meta matrix missing timestamp column: {path}")
    if "atr" not in df.columns:
        raise ValueError(f"Meta matrix missing atr column: {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "atr", "open", "high", "low", "close"]].copy()


def _load_one_min(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _build_trades(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta_asof = meta.rename(columns={"timestamp": "entry_timestamp", "atr": "entry_atr"}).sort_values("entry_timestamp")
    open_trade: dict[str, dict[str, object] | None] = {"long": None, "short": None}
    trades: list[dict[str, object]] = []

    for row in events.itertuples(index=False):
        event = str(row.event)
        ts = pd.Timestamp(row.timestamp)
        price = float(row.price)
        if event == "enter_long":
            open_trade["long"] = {"entry_timestamp": ts, "entry_price": price, "side": "long", "symbol": row.symbol}
        elif event == "enter_short":
            open_trade["short"] = {"entry_timestamp": ts, "entry_price": price, "side": "short", "symbol": row.symbol}
        elif event == "exit_long" and open_trade["long"] is not None:
            rec = dict(open_trade["long"])
            rec["exit_timestamp"] = ts
            rec["exit_price"] = price
            trades.append(rec)
            open_trade["long"] = None
        elif event == "exit_short" and open_trade["short"] is not None:
            rec = dict(open_trade["short"])
            rec["exit_timestamp"] = ts
            rec["exit_price"] = price
            trades.append(rec)
            open_trade["short"] = None

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df

    trades_df = pd.merge_asof(
        trades_df.sort_values("entry_timestamp"),
        meta_asof,
        on="entry_timestamp",
        direction="backward",
        tolerance=pd.Timedelta("10min"),
    )
    trades_df["hold_minutes"] = (
        (pd.to_datetime(trades_df["exit_timestamp"], utc=True) - pd.to_datetime(trades_df["entry_timestamp"], utc=True))
        .dt.total_seconds()
        .div(60.0)
    )
    trades_df["realized_price_move"] = np.where(
        trades_df["side"].eq("long"),
        pd.to_numeric(trades_df["exit_price"], errors="coerce") - pd.to_numeric(trades_df["entry_price"], errors="coerce"),
        pd.to_numeric(trades_df["entry_price"], errors="coerce") - pd.to_numeric(trades_df["exit_price"], errors="coerce"),
    )
    trades_df["realized_exit_atr"] = trades_df["realized_price_move"] / pd.to_numeric(trades_df["entry_atr"], errors="coerce")
    return trades_df


def _compute_mfe_metrics(trades: pd.DataFrame, one_min: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades

    ts_arr = one_min["timestamp"].to_numpy(dtype="datetime64[ns]")
    high_arr = pd.to_numeric(one_min["high"], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(one_min["low"], errors="coerce").to_numpy(dtype=float)

    mfe_price: list[float] = []
    mfe_atr: list[float] = []
    mfe_timestamp: list[pd.Timestamp] = []
    giveback_atr: list[float] = []
    capture_ratio: list[float] = []
    max_before_exit: list[bool] = []

    for row in trades.itertuples(index=False):
        entry_ts = pd.Timestamp(row.entry_timestamp).to_datetime64()
        exit_ts = pd.Timestamp(row.exit_timestamp).to_datetime64()
        entry_price = float(row.entry_price)
        entry_atr = float(row.entry_atr) if np.isfinite(row.entry_atr) else np.nan

        start_idx = int(np.searchsorted(ts_arr, entry_ts, side="left"))
        end_idx = int(np.searchsorted(ts_arr, exit_ts, side="left"))
        window_high = high_arr[start_idx:end_idx]
        window_low = low_arr[start_idx:end_idx]
        window_ts = ts_arr[start_idx:end_idx]

        best_price = float("nan")
        best_ts = pd.NaT
        best_move = float("nan")

        if len(window_ts) > 0:
            if row.side == "long":
                rel_idx = int(np.nanargmax(window_high)) if np.isfinite(window_high).any() else -1
                if rel_idx >= 0:
                    best_price = float(window_high[rel_idx])
                    best_ts = pd.Timestamp(window_ts[rel_idx], tz="UTC")
                    best_move = best_price - entry_price
            else:
                rel_idx = int(np.nanargmin(window_low)) if np.isfinite(window_low).any() else -1
                if rel_idx >= 0:
                    best_price = float(window_low[rel_idx])
                    best_ts = pd.Timestamp(window_ts[rel_idx], tz="UTC")
                    best_move = entry_price - best_price

        if not np.isfinite(best_move):
            best_move = float(row.realized_price_move)
            best_price = float(row.exit_price)
            best_ts = pd.Timestamp(row.exit_timestamp)

        best_atr = best_move / entry_atr if np.isfinite(entry_atr) and entry_atr > 0.0 else np.nan
        realized_atr = float(row.realized_exit_atr) if np.isfinite(row.realized_exit_atr) else np.nan
        giveback = best_atr - realized_atr if np.isfinite(best_atr) and np.isfinite(realized_atr) else np.nan
        ratio = realized_atr / best_atr if np.isfinite(best_atr) and best_atr > 0.0 and np.isfinite(realized_atr) else np.nan

        mfe_price.append(best_price)
        mfe_timestamp.append(best_ts)
        mfe_atr.append(best_atr)
        giveback_atr.append(giveback)
        capture_ratio.append(ratio)
        max_before_exit.append(bool(pd.notna(best_ts) and pd.Timestamp(best_ts) < pd.Timestamp(row.exit_timestamp)))

    out = trades.copy()
    out["mfe_price"] = mfe_price
    out["mfe_timestamp"] = mfe_timestamp
    out["mfe_atr"] = mfe_atr
    out["giveback_atr"] = giveback_atr
    out["capture_ratio"] = capture_ratio
    out["max_before_exit"] = max_before_exit
    out["winner"] = pd.to_numeric(out["realized_exit_atr"], errors="coerce") > 0.0
    return out


def _quantiles(series: pd.Series, prefix: str) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {f"{prefix}_{name}": np.nan for name in ("p10", "p25", "p50", "p75", "p90")}
    return {
        f"{prefix}_p10": float(s.quantile(0.10)),
        f"{prefix}_p25": float(s.quantile(0.25)),
        f"{prefix}_p50": float(s.quantile(0.50)),
        f"{prefix}_p75": float(s.quantile(0.75)),
        f"{prefix}_p90": float(s.quantile(0.90)),
    }


def _summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for side in ("long", "short"):
        sdf = trades[trades["side"].eq(side)].copy()
        if sdf.empty:
            continue
        realized = pd.to_numeric(sdf["realized_exit_atr"], errors="coerce")
        mfe = pd.to_numeric(sdf["mfe_atr"], errors="coerce")
        giveback = pd.to_numeric(sdf["giveback_atr"], errors="coerce")
        capture = pd.to_numeric(sdf["capture_ratio"], errors="coerce")
        rows.append(
            {
                "side": side,
                "trades": int(len(sdf)),
                "win_rate": float(pd.to_numeric(sdf["winner"], errors="coerce").mean()),
                "avg_realized_exit_atr": float(realized.mean()),
                "avg_mfe_atr": float(mfe.mean()),
                "avg_giveback_atr": float(giveback.mean()),
                "avg_capture_ratio": float(capture.mean()),
                "median_capture_ratio": float(capture.median()),
                "pct_max_before_exit": float(pd.to_numeric(sdf["max_before_exit"], errors="coerce").mean()),
                "pct_giveback_gt_0_50_atr": float((giveback > 0.50).mean()),
                "pct_giveback_gt_1_00_atr": float((giveback > 1.00).mean()),
                **_quantiles(realized, "realized_exit_atr"),
                **_quantiles(mfe, "mfe_atr"),
                **_quantiles(giveback, "giveback_atr"),
                **_quantiles(capture, "capture_ratio"),
            }
        )
    return pd.DataFrame(rows)


def _threshold_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for side in ("long", "short"):
        sdf = trades[trades["side"].eq(side)].copy()
        if sdf.empty:
            continue
        for arm in (1.5, 2.0, 2.5, 3.0):
            arm_df = sdf[pd.to_numeric(sdf["mfe_atr"], errors="coerce") >= arm].copy()
            if arm_df.empty:
                continue
            giveback = pd.to_numeric(arm_df["giveback_atr"], errors="coerce")
            capture = pd.to_numeric(arm_df["capture_ratio"], errors="coerce")
            rows.append(
                {
                    "side": side,
                    "arm_atr": float(arm),
                    "trades_with_mfe_at_least_arm": int(len(arm_df)),
                    "avg_realized_exit_atr": float(pd.to_numeric(arm_df["realized_exit_atr"], errors="coerce").mean()),
                    "avg_mfe_atr": float(pd.to_numeric(arm_df["mfe_atr"], errors="coerce").mean()),
                    "avg_giveback_atr": float(giveback.mean()),
                    "median_giveback_atr": float(giveback.median()),
                    "p75_giveback_atr": float(giveback.quantile(0.75)),
                    "p90_giveback_atr": float(giveback.quantile(0.90)),
                    "avg_capture_ratio": float(capture.mean()),
                    "median_capture_ratio": float(capture.median()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    events = _load_events(Path(args.events))
    meta = _load_meta_matrix(Path(args.meta_matrix))
    one_min = _load_one_min(Path(args.one_min_data))
    trades = _build_trades(events, meta)
    if trades.empty:
        raise SystemExit("No closed trades found in events file.")
    trades = _compute_mfe_metrics(trades, one_min)

    trades_out = Path(args.trades_out)
    summary_out = Path(args.summary_out)
    thresholds_out = Path(args.thresholds_out)
    for path in (trades_out, summary_out, thresholds_out):
        path.parent.mkdir(parents=True, exist_ok=True)

    trades.to_csv(trades_out, index=False)
    summary = _summary(trades)
    summary.to_csv(summary_out, index=False)
    threshold_summary = _threshold_summary(trades)
    threshold_summary.to_csv(thresholds_out, index=False)

    print(summary.to_string(index=False))
    print("\nthreshold_slices:")
    if threshold_summary.empty:
        print("(none)")
    else:
        print(threshold_summary.to_string(index=False))
    print(f"\ntrades_csv={trades_out}")
    print(f"summary_csv={summary_out}")
    print(f"thresholds_csv={thresholds_out}")


if __name__ == "__main__":
    main()
