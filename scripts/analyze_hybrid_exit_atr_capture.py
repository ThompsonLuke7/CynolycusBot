from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze realized ATR capture for the hybrid exit policy using entry-time ATR."
    )
    parser.add_argument(
        "--events",
        default="Data/inference/spy/10min/meta/meta_events_hybrid_exit_next_10m_open.csv",
        help="CSV of entry/exit events for the execution path to analyze.",
    )
    parser.add_argument(
        "--meta-matrix",
        default="Data/inference/spy/10min/debug_matrices_warmup/spy/live_meta_matrix_on_trace_ts_live_2026_03_24.parquet",
        help="Parquet with 10min matrix including ATR.",
    )
    parser.add_argument(
        "--trades-out",
        default="Data/inference/spy/10min/meta/hybrid_exit_atr_capture_trades.csv",
        help="Per-trade ATR capture CSV.",
    )
    parser.add_argument(
        "--summary-out",
        default="Data/inference/spy/10min/meta/hybrid_exit_atr_capture_summary.csv",
        help="Per-side summary CSV.",
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
    trades_df["price_move"] = np.where(
        trades_df["side"].eq("long"),
        pd.to_numeric(trades_df["exit_price"], errors="coerce") - pd.to_numeric(trades_df["entry_price"], errors="coerce"),
        pd.to_numeric(trades_df["entry_price"], errors="coerce") - pd.to_numeric(trades_df["exit_price"], errors="coerce"),
    )
    trades_df["atr_capture"] = trades_df["price_move"] / pd.to_numeric(trades_df["entry_atr"], errors="coerce")
    trades_df["return_frac"] = np.where(
        trades_df["side"].eq("long"),
        pd.to_numeric(trades_df["exit_price"], errors="coerce") / pd.to_numeric(trades_df["entry_price"], errors="coerce") - 1.0,
        (pd.to_numeric(trades_df["entry_price"], errors="coerce") - pd.to_numeric(trades_df["exit_price"], errors="coerce"))
        / pd.to_numeric(trades_df["entry_price"], errors="coerce"),
    )
    trades_df["winner"] = trades_df["atr_capture"] > 0.0
    return trades_df


def _summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for side in ("long", "short"):
        sdf = trades[trades["side"].eq(side)].copy()
        wins = sdf[sdf["winner"]].copy()
        if sdf.empty:
            continue
        capture = pd.to_numeric(sdf["atr_capture"], errors="coerce")
        win_capture = pd.to_numeric(wins["atr_capture"], errors="coerce")
        rows.append(
            {
                "side": side,
                "trades": int(len(sdf)),
                "win_rate": float(sdf["winner"].mean()),
                "avg_atr_capture_all": float(capture.mean()),
                "median_atr_capture_all": float(capture.median()),
                "max_atr_capture_all": float(capture.max()),
                "min_atr_capture_all": float(capture.min()),
                "p90_atr_capture_all": float(capture.quantile(0.90)),
                "avg_atr_capture_winners": float(win_capture.mean()) if not wins.empty else np.nan,
                "median_atr_capture_winners": float(win_capture.median()) if not wins.empty else np.nan,
                "max_atr_capture_winners": float(win_capture.max()) if not wins.empty else np.nan,
                "avg_hold_minutes": float(pd.to_numeric(sdf["hold_minutes"], errors="coerce").mean()),
                "max_hold_minutes": float(pd.to_numeric(sdf["hold_minutes"], errors="coerce").max()),
                "max_vs_avg_all_ratio": float(capture.max() / capture.mean()) if np.isfinite(capture.mean()) and capture.mean() != 0 else np.nan,
                "max_vs_avg_winners_ratio": float(win_capture.max() / win_capture.mean()) if not wins.empty and np.isfinite(win_capture.mean()) and win_capture.mean() != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    events = _load_events(Path(args.events))
    meta = _load_meta_matrix(Path(args.meta_matrix))
    trades = _build_trades(events, meta)
    if trades.empty:
        raise SystemExit("No closed trades found in events file.")

    trades_out = Path(args.trades_out)
    summary_out = Path(args.summary_out)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    trades.to_csv(trades_out, index=False)
    summary = _summary(trades)
    summary.to_csv(summary_out, index=False)

    print(summary.to_string(index=False))
    print(f"\ntrades_csv={trades_out}")
    print(f"summary_csv={summary_out}")


if __name__ == "__main__":
    main()
