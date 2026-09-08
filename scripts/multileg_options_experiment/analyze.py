#!/usr/bin/env python
"""Validate every leg, run paired eight-session P&L, and write result artifacts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.multileg_options_experiment.core import (  # noqa: E402
    LegSpec,
    structure_return,
    validate_leg_bars,
    week_block_ci,
)

DATA_DIR = REPO / "research/multileg_options_experiment/data"
PATHS = DATA_DIR / "structure_paths.jsonl"
BARS_DIR = REPO / "Data/shared/bars/1d"
TRADES_OUT = DATA_DIR / "trade_results.parquet"
SUMMARY_OUT = DATA_DIR / "summary.csv"
COVERAGE_OUT = DATA_DIR / "coverage.json"
HALF_SPREADS = (0.0, 0.04, 0.08, 0.12)
HOLD_SESSIONS = 8


def underlying(ticker: str) -> pd.DataFrame | None:
    path = BARS_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path, columns=["timestamp", "close"])
    frame["date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
    return frame.drop_duplicates("date", keep="last").set_index("date").sort_index()


def bar_map(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        day = pd.Timestamp(row["t"]).tz_convert("America/New_York").date().isoformat()
        result[day] = row
    return result


def evaluate_record(record: dict, rejects: Counter) -> list[dict]:
    if record.get("error"):
        rejects["fetch_error"] += 1
        return []
    stock = underlying(record["ticker"])
    if stock is None:
        rejects["no_underlying"] += 1
        return []
    signal = record["signal_date"]
    future = stock.loc[stock.index >= signal]
    if len(future) <= HOLD_SESSIONS:
        rejects["insufficient_underlying_horizon"] += 1
        return []
    allowed_entry_dates = set(future.index[:2])
    target_exit = future.index[HOLD_SESSIONS]
    bars_by_symbol = {symbol: bar_map(rows) for symbol, rows in record["bars"].items()}
    valid_symbols: dict[str, bool] = {}
    for name, rows in record["bars"].items():
        right = next(
            (leg["right"] for legs in record["structures"].values() for leg in legs if leg["symbol"] == name), "C"
        )
        ok, stats = validate_leg_bars(rows, stock, right)
        valid_symbols[name] = ok
        if not ok:
            rejects[f"leg_{stats['reason']}"] += 1

    structures = dict(record["structures"])
    structures.setdefault("long_shares", [{
        "symbol": f"{record['ticker']}_SHARES", "right": "S", "strike": None,
        "quantity": 100.0, "expiry": None,
    }])
    results = []
    for structure_name, raw_legs in structures.items():
        legs = tuple(LegSpec(**item) for item in raw_legs)
        option_legs = [item for item in legs if item.right != "S"]
        if any(not valid_symbols.get(item.symbol, False) for item in option_legs):
            rejects[f"{structure_name}:invalid_leg"] += 1
            continue
        common_entry = allowed_entry_dates.copy()
        for item in option_legs:
            common_entry &= set(bars_by_symbol[item.symbol])
        if not common_entry:
            rejects[f"{structure_name}:no_synchronous_entry"] += 1
            continue
        entry_day = sorted(common_entry)[0]
        if any(target_exit not in bars_by_symbol[item.symbol] for item in option_legs):
            rejects[f"{structure_name}:no_synchronous_exit"] += 1
            continue
        entry_spot = float(stock.loc[entry_day, "close"])
        exit_spot = float(stock.loc[target_exit, "close"])
        entry_prices = {item.symbol: float(bars_by_symbol[item.symbol][entry_day]["c"]) for item in option_legs}
        exit_prices = {item.symbol: float(bars_by_symbol[item.symbol][target_exit]["c"]) for item in option_legs}
        if any(value <= 0 for value in (*entry_prices.values(), *exit_prices.values())):
            rejects[f"{structure_name}:nonpositive_price"] += 1
            continue
        violates_intrinsic = False
        for item in option_legs:
            entry_intrinsic = max(entry_spot - float(item.strike), 0.0) if item.right == "C" else max(float(item.strike) - entry_spot, 0.0)
            exit_intrinsic = max(exit_spot - float(item.strike), 0.0) if item.right == "C" else max(float(item.strike) - exit_spot, 0.0)
            if entry_prices[item.symbol] + 0.01 < entry_intrinsic or exit_prices[item.symbol] + 0.01 < exit_intrinsic:
                violates_intrinsic = True
                break
        if violates_intrinsic:
            rejects[f"{structure_name}:intrinsic_violation"] += 1
            continue
        for spread in HALF_SPREADS:
            ret, detail = structure_return(legs, entry_prices, exit_prices, entry_spot, exit_spot, spread)
            if ret is None:
                rejects[f"{structure_name}:{detail['reason']}"] += 1
                continue
            results.append({
                "trade_key": record["trade_key"], "module": record["module"], "ticker": record["ticker"],
                "week_key": pd.Timestamp(record["signal_date"]).strftime("%G-W%V"),
                "signal_date": signal, "entry_date": entry_day, "exit_date": target_exit,
                "structure": structure_name, "half_spread": spread, "return_on_capital": ret,
                **detail,
            })
    return results


def summarize(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (structure, spread), group in trades.groupby(["structure", "half_spread"]):
        baseline = trades[(trades["structure"] == "long_call") & (trades["half_spread"] == spread)][
            ["trade_key", "return_on_capital"]
        ].rename(columns={"return_on_capital": "baseline_return"})
        paired = group.merge(baseline, on="trade_key", how="inner")
        paired["difference"] = paired["return_on_capital"] - paired["baseline_return"]
        lo, hi = week_block_ci(paired[["week_key", "difference"]]) if len(paired) else (np.nan, np.nan)
        by_module = paired.groupby("module")["difference"].mean()
        rows.append({
            "structure": structure, "half_spread": spread, "n": len(paired),
            "tickers": paired["ticker"].nunique(), "weeks": paired["week_key"].nunique(),
            "mean_return": paired["return_on_capital"].mean(),
            "median_return": paired["return_on_capital"].median(),
            "win_rate": paired["return_on_capital"].gt(0).mean(),
            "p10_return": paired["return_on_capital"].quantile(0.10),
            "baseline_mean": paired["baseline_return"].mean(),
            "paired_mean_difference": paired["difference"].mean(),
            "paired_ci_low": lo, "paired_ci_high": hi,
            "modules_positive": int(by_module.gt(0).sum()), "modules_present": len(by_module),
            "decision_eligible": bool(len(paired) >= 100 and paired["ticker"].nunique() >= 20 and paired["week_key"].nunique() >= 8),
        })
    return pd.DataFrame(rows).sort_values(["half_spread", "paired_mean_difference"], ascending=[True, False])


def main() -> None:
    records = [json.loads(line) for line in PATHS.open() if line.strip()]
    rejects: Counter = Counter()
    results = [row for record in records for row in evaluate_record(record, rejects)]
    trades = pd.DataFrame(results)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(TRADES_OUT, index=False)
    summary = summarize(trades) if not trades.empty else pd.DataFrame()
    summary.to_csv(SUMMARY_OUT, index=False)
    COVERAGE_OUT.write_text(json.dumps({"records": len(records), "rejects": rejects, "result_rows": len(trades)}, indent=2))
    print(f"records={len(records)} result_rows={len(trades)}")
    if not summary.empty:
        cols = ["structure", "half_spread", "n", "tickers", "weeks", "mean_return", "baseline_mean", "paired_mean_difference", "paired_ci_low", "paired_ci_high", "decision_eligible"]
        print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"wrote {TRADES_OUT}, {SUMMARY_OUT}, {COVERAGE_OUT}")


if __name__ == "__main__":
    main()
