from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare hold-through-weekend vs forced Friday close using underlying price history.")
    p.add_argument(
        "--events",
        default="Data/inference/spy/10min/meta/meta_events_1m_directional_execution.csv",
        help="Paired entry/exit event CSV for the policy under test.",
    )
    p.add_argument(
        "--one-min-data",
        default="Data/raw/spy/spy_intraday_1min_live_2026_03_24.parquet",
        help="1m parquet used to source the last Friday regular-hours close.",
    )
    p.add_argument(
        "--tz",
        default="America/New_York",
        help="Display timezone for weekend detection.",
    )
    p.add_argument(
        "--trades-out",
        default="Data/inference/spy/10min/meta/friday_forced_close_trade_compare.csv",
        help="Per-trade weekend comparison CSV.",
    )
    p.add_argument(
        "--summary-out",
        default="Data/inference/spy/10min/meta/friday_forced_close_summary.csv",
        help="Summary CSV.",
    )
    return p.parse_args()


def pair_trades(events: pd.DataFrame) -> pd.DataFrame:
    open_pos: dict[str, dict[str, object] | None] = {"long": None, "short": None}
    rows: list[dict[str, object]] = []
    for _, r in events.sort_values("timestamp").iterrows():
        ev = str(r["event"])
        side = "long" if "long" in ev else "short"
        if ev.startswith("enter_"):
            open_pos[side] = {
                "entry_ts": r["timestamp"],
                "entry_ts_ny": r["ts_ny"],
                "entry_price": float(r["price"]),
            }
        elif ev.startswith("exit_") and open_pos.get(side) is not None:
            op = open_pos[side]
            rows.append(
                {
                    "side": side,
                    "entry_ts": op["entry_ts"],
                    "entry_ts_ny": op["entry_ts_ny"],
                    "entry_price": op["entry_price"],
                    "exit_ts": r["timestamp"],
                    "exit_ts_ny": r["ts_ny"],
                    "exit_price": float(r["price"]),
                }
            )
            open_pos[side] = None
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    tz = ZoneInfo(args.tz)

    events = pd.read_csv(args.events, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    events["ts_ny"] = events["timestamp"].dt.tz_convert(tz)
    trades = pair_trades(events)
    if trades.empty:
        raise SystemExit("no_paired_trades")

    trades["entry_date_ny"] = pd.to_datetime(trades["entry_ts_ny"]).dt.date
    trades["exit_date_ny"] = pd.to_datetime(trades["exit_ts_ny"]).dt.date
    trades["days_held"] = (pd.to_datetime(trades["exit_date_ny"]) - pd.to_datetime(trades["entry_date_ny"])).dt.days
    trades["crosses_weekend"] = trades["days_held"] >= 3
    weekend = trades.loc[trades["crosses_weekend"]].copy()
    if weekend.empty:
        raise SystemExit("no_weekend_trades")

    one = pd.read_parquet(args.one_min_data)
    one["timestamp"] = pd.to_datetime(one["timestamp"], utc=True)
    one["ts_ny"] = one["timestamp"].dt.tz_convert(tz)
    one["date_ny"] = one["ts_ny"].dt.date
    last_rth = (
        one.sort_values("timestamp")
        .groupby("date_ny")
        .tail(1)[["date_ny", "timestamp", "ts_ny", "close"]]
        .rename(
            columns={
                "timestamp": "friday_close_ts_utc",
                "ts_ny": "friday_close_ts_ny",
                "close": "friday_close_price",
            }
        )
    )

    weekend["friday_date_ny"] = weekend["entry_ts_ny"].dt.date
    weekend = weekend.merge(last_rth, left_on="friday_date_ny", right_on="date_ny", how="left")

    weekend["actual_move"] = weekend.apply(
        lambda r: (r["exit_price"] - r["entry_price"]) if r["side"] == "long" else (r["entry_price"] - r["exit_price"]),
        axis=1,
    )
    weekend["forced_friday_move"] = weekend.apply(
        lambda r: (r["friday_close_price"] - r["entry_price"]) if r["side"] == "long" else (r["entry_price"] - r["friday_close_price"]),
        axis=1,
    )
    weekend["weekend_hold_edge"] = weekend["actual_move"] - weekend["forced_friday_move"]
    weekend["hold_better"] = weekend["weekend_hold_edge"] > 0

    trades_out = Path(args.trades_out)
    trades_out.parent.mkdir(parents=True, exist_ok=True)
    weekend.to_csv(trades_out, index=False)

    summary = pd.DataFrame(
        [
            {
                "weekend_trade_count": int(len(weekend)),
                "avg_actual_move": float(weekend["actual_move"].mean()),
                "median_actual_move": float(weekend["actual_move"].median()),
                "avg_forced_friday_move": float(weekend["forced_friday_move"].mean()),
                "median_forced_friday_move": float(weekend["forced_friday_move"].median()),
                "avg_weekend_hold_edge": float(weekend["weekend_hold_edge"].mean()),
                "median_weekend_hold_edge": float(weekend["weekend_hold_edge"].median()),
                "hold_better_count": int((weekend["weekend_hold_edge"] > 0).sum()),
                "forced_better_count": int((weekend["weekend_hold_edge"] < 0).sum()),
            }
        ]
    )
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)

    print(summary.to_string(index=False))
    print("\nSaved:")
    print(trades_out)
    print(summary_out)


if __name__ == "__main__":
    main()
