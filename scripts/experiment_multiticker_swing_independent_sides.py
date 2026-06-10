from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategies.multi_ticker_swing.backtest.sweep_v4 import (  # noqa: E402
    COMMISSION_PCT,
    CONFIRM_MAX_5M,
    MAX_HOLD_5M,
    TRADE_COLS,
    TickerData,
    compute_metrics,
    find_5m_confirmation,
    load_raw_30m,
    load_raw_5m,
    simulate_ticker_5m,
)
from scripts.plot_multiticker_swing_ticker_action import _infer_trade_times, _load_config, _load_probs  # noqa: E402


ET = ZoneInfo("America/New_York")
DEFAULT_PROBA = Path("strategies/multi_ticker_swing/models/p_swing_probs.parquet")
DEFAULT_UNIVERSE = Path("strategies/multi_ticker_swing/config/trading_universe.json")
DEFAULT_OUT_DIR = Path("UI/swing_audit/independent_sides_20260607")


def _exit_cfg(cfg: dict) -> dict:
    return {
        "name": str(cfg.get("combo", "")),
        "arm_pct": 0.025,
        "giveback_pct": 0.25,
        "sl_atr": float(cfg.get("sl_atr", 4.0) or 0.0),
        "np_n_bars": None if cfg.get("np_n_bars") is None else int(cfg.get("np_n_bars")),
        "np_mfe_atr": None if cfg.get("np_mfe_atr") is None else float(cfg.get("np_mfe_atr")),
    }


def _simulate_side(
    td: TickerData,
    *,
    direction: int,
    entry_threshold: float,
    exit_cfg: dict,
    opposite_exit_threshold: float | None = None,
    min_opposite_hold_5m: int = 6,
) -> list[tuple]:
    if not td.has_5m:
        raise SystemExit("This experiment requires 5m bars.")

    sl_atr_mult = float(exit_cfg.get("sl_atr", 0.0))
    arm_pct = exit_cfg.get("arm_pct")
    giveback_pct = exit_cfg.get("giveback_pct")
    np_n_bars = exit_cfg.get("np_n_bars")
    np_mfe_atr = exit_cfg.get("np_mfe_atr")
    cooldown_ts = np.datetime64(0, "ns")
    results = []

    for i in range(td.n_30m - 1):
        if td.ts_30m[i] <= cooldown_ts:
            continue
        pl = td.p_long_dir[i]
        ps = td.p_short_dir[i]
        if np.isnan(pl):
            continue
        side_prob = pl if direction == 1 else ps
        if side_prob < entry_threshold:
            continue
        atr_val = td.atr_30m[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        conf_price, conf_5m_idx = find_5m_confirmation(td, i, direction, CONFIRM_MAX_5M)
        if conf_price is None:
            continue
        entry_5m_idx = conf_5m_idx + 1
        if entry_5m_idx >= td.n_5m or td.is_first_30min_5m[entry_5m_idx]:
            continue

        entry_price = float(td.open_5m[entry_5m_idx])
        if np.isnan(entry_price) or entry_price <= 0:
            entry_price = conf_price
        sl_price = entry_price - direction * sl_atr_mult * atr_val if sl_atr_mult > 0 else None
        best_price = entry_price
        trail_armed = False
        exit_5m_idx = min(entry_5m_idx + MAX_HOLD_5M, td.n_5m - 1)
        exit_price = float(td.close_5m[exit_5m_idx])
        exit_reason = "time"

        for j in range(entry_5m_idx, min(entry_5m_idx + MAX_HOLD_5M + 1, td.n_5m)):
            bar_h = float(td.high_5m[j])
            bar_l = float(td.low_5m[j])
            bar_c = float(td.close_5m[j])

            if sl_price is not None:
                if direction == 1 and bar_l <= sl_price:
                    exit_5m_idx, exit_price, exit_reason = j, sl_price, "sl"
                    break
                if direction == -1 and bar_h >= sl_price:
                    exit_5m_idx, exit_price, exit_reason = j, sl_price, "sl"
                    break

            if opposite_exit_threshold is not None and (j - entry_5m_idx) >= min_opposite_hold_5m:
                k = min(int(np.searchsorted(td.ts_30m, td.ts_5m[j], side="right")) - 1, td.n_30m - 1)
                if k >= 0:
                    opp = td.p_short_dir[k] if direction == 1 else td.p_long_dir[k]
                    if not np.isnan(opp) and opp >= opposite_exit_threshold:
                        exit_5m_idx, exit_price, exit_reason = j, bar_c, "opp_edge"
                        break

            best_price = max(best_price, bar_h) if direction == 1 else min(best_price, bar_l)
            if arm_pct is not None and giveback_pct is not None:
                move_pct = direction * (best_price - entry_price) / entry_price
                if move_pct >= arm_pct:
                    trail_armed = True
                if trail_armed:
                    peak_profit = direction * (best_price - entry_price)
                    cur_profit = direction * (bar_c - entry_price)
                    floor_profit = peak_profit * (1.0 - giveback_pct)
                    if cur_profit <= floor_profit:
                        exit_5m_idx = j
                        exit_price = entry_price + direction * floor_profit
                        exit_reason = "trail"
                        break

            if np_n_bars is not None and (j - entry_5m_idx) == np_n_bars and not trail_armed:
                mfe_atr_units = direction * (best_price - entry_price) / atr_val
                if mfe_atr_units < np_mfe_atr:
                    exit_5m_idx, exit_price, exit_reason = j, bar_c, "no_progress"
                    break

        exit_5m_ts = td.ts_5m[exit_5m_idx]
        exit_30m_idx = min(int(np.searchsorted(td.ts_30m, exit_5m_ts, side="right")), td.n_30m - 1)
        holding_5m = exit_5m_idx - entry_5m_idx
        net_ret = direction * (exit_price - entry_price) / entry_price - COMMISSION_PCT
        results.append((td.ticker, direction, i, exit_30m_idx, entry_price, exit_price, net_ret, exit_reason, holding_5m))
        cooldown_ts = exit_5m_ts

    return results


def _metrics_row(name: str, trades: pd.DataFrame) -> dict:
    m = compute_metrics(trades)
    return {
        "policy": name,
        "trades": int(m["n_trades"]),
        "longs": int(m["long_n"]),
        "shorts": int(m["short_n"]),
        "win_rate": float(m["win_rate"]),
        "profit_factor": float(m["profit_factor"]),
        "sharpe": float(m["sharpe"]),
        "avg_trade_pp": float(m["avg_pnl_pct"] * 100.0),
        "total_pnl_pp": float(m["total_pnl_pct"] * 100.0),
        "max_dd_pp": float(m["max_dd_pct"]),
        "avg_holding_5m": float(m["avg_holding_bars"]),
    }


def _window_filter(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    out = trades.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    return out[out["entry_time"].between(start, end)].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare single-position vs independent long/short side policies.")
    parser.add_argument("--ticker", default="APLD")
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-06-05")
    parser.add_argument("--proba", type=Path, default=DEFAULT_PROBA)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    ticker = args.ticker.upper()
    cfg = _load_config(ticker, args.universe)
    probs = _load_probs(ticker, args.proba, "all")
    bars30 = load_raw_30m(ticker)
    bars5 = load_raw_5m(ticker)
    td = TickerData(ticker, bars30, bars5, probs)
    exit_cfg = _exit_cfg(cfg)
    threshold = float(cfg.get("entry_threshold", 0.6))
    start = pd.Timestamp(args.start, tz=ET).tz_convert("UTC")
    end = pd.Timestamp(args.end, tz=ET).tz_convert("UTC")

    original = pd.DataFrame(simulate_ticker_5m(td, threshold, CONFIRM_MAX_5M, exit_cfg), columns=TRADE_COLS)
    long_only = pd.DataFrame(_simulate_side(td, direction=1, entry_threshold=threshold, exit_cfg=exit_cfg), columns=TRADE_COLS)
    short_only = pd.DataFrame(_simulate_side(td, direction=-1, entry_threshold=threshold, exit_cfg=exit_cfg), columns=TRADE_COLS)
    independent = pd.concat([long_only, short_only], ignore_index=True).sort_values(["signal_idx", "direction"])
    independent_opp = pd.concat(
        [
            pd.DataFrame(
                _simulate_side(
                    td,
                    direction=1,
                    entry_threshold=threshold,
                    exit_cfg=exit_cfg,
                    opposite_exit_threshold=threshold,
                ),
                columns=TRADE_COLS,
            ),
            pd.DataFrame(
                _simulate_side(
                    td,
                    direction=-1,
                    entry_threshold=threshold,
                    exit_cfg=exit_cfg,
                    opposite_exit_threshold=threshold,
                ),
                columns=TRADE_COLS,
            ),
        ],
        ignore_index=True,
    ).sort_values(["signal_idx", "direction"])

    frames = {
        "single_position_mixed": original,
        "calls_only_independent": long_only,
        "puts_only_independent": short_only,
        "independent_long_short": independent,
        "independent_with_opp_edge_exit": independent_opp,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, trades in frames.items():
        timed = _infer_trade_times(trades, bars30, bars5) if not trades.empty else trades
        windowed = _window_filter(timed, start, end) if not timed.empty else timed
        windowed.to_csv(args.out_dir / f"{ticker.lower()}_{name}_trades.csv", index=False)
        rows.append(_metrics_row(name, windowed))

    summary = pd.DataFrame(rows).sort_values(["profit_factor", "total_pnl_pp"], ascending=False)
    out_path = args.out_dir / f"{ticker.lower()}_independent_side_summary.csv"
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
