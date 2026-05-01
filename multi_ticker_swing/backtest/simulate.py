"""
Event-driven backtest simulator for the multi-ticker swing pipeline.

Signal generation (30-minute):
  - Model probability > threshold for class 2 (long) or class 0 (short)
  - Optional: swing state gate must be active

Execution (10-minute):
  - Entry at the open of the first 10m bar after the 30m signal
  - Up to entry_bars_max bars to find a qualifying entry
  - TP / SL tracked bar-by-bar on 10m OHLCV

Usage:
  python multi_ticker_swing/backtest/simulate.py [--test-start DATE] [--test-end DATE]
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from multi_ticker_swing.config.pipeline_config import (
    BACKTEST_CONFIG,
    BACKTEST_RESULTS_DIR,
    FEATURE_COLUMNS,
    MODEL_PATH,
    RAW_10M_DIR,
    TRAINING_MATRIX,
)
from multi_ticker_swing.data.load_data import load_raw_5m, load_raw_30m, load_training_matrix


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    ticker:       str
    direction:    int          # +1 long, -1 short
    entry_time:   pd.Timestamp
    entry_price:  float
    tp_price:     float
    sl_price:     float
    atr_at_entry: float
    size:         float        # dollar notional


@dataclass
class Trade:
    ticker:       str
    direction:    int
    entry_time:   pd.Timestamp
    exit_time:    pd.Timestamp
    entry_price:  float
    exit_price:   float
    pnl_pct:      float
    pnl_dollar:   float
    exit_reason:  str          # "tp", "sl", "time"
    size:         float


# ---------------------------------------------------------------------------
# 10m data loader (per ticker, test window only)
# ---------------------------------------------------------------------------

def _load_5m_slice(
    ticker: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame | None:
    try:
        df = load_raw_5m(ticker)
        df.columns = [c.lower() for c in df.columns]
        mask = (df.index >= test_start) & (df.index <= test_end)
        return df.loc[mask].copy()
    except FileNotFoundError:
        return None


def _load_30m_slice(
    ticker: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame | None:
    try:
        df = load_raw_30m(ticker)
        df.columns = [c.lower() for c in df.columns]
        mask = (df.index >= test_start) & (df.index <= test_end)
        return df.loc[mask].copy()
    except FileNotFoundError:
        logger.warning("[%s] 30m data missing — no execution data", ticker)
        return None


# ---------------------------------------------------------------------------
# Signal generation from model
# ---------------------------------------------------------------------------

def generate_signals(
    matrix: pd.DataFrame,
    model,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    feature_columns: list[str],
    cfg: dict,
) -> pd.DataFrame:
    """
    Run model on test-period rows and attach probability columns + signal column.

    signal:  +1 = long signal,  -1 = short signal,  0 = no signal
    """
    ts_col = _find_ts_col(matrix)
    mask   = (pd.to_datetime(matrix[ts_col]) >= test_start) & \
             (pd.to_datetime(matrix[ts_col]) <= test_end)
    df = matrix.loc[mask].copy()

    avail = [c for c in feature_columns if c in df.columns]
    X     = df[avail].values.astype(np.float32)

    proba = model.predict_proba(X)    # shape (N, 3): [P(short), P(neutral), P(long)]
    df["prob_short"]   = proba[:, 0]
    df["prob_neutral"] = proba[:, 1]
    df["prob_long"]    = proba[:, 2]

    th_long  = cfg["prob_threshold_long"]
    th_short = cfg["prob_threshold_short"]

    long_signal  = df["prob_long"]  > th_long
    short_signal = df["prob_short"] > th_short

    if cfg.get("swing_gate_required", True) and "p_long_state_gate" in df.columns:
        long_signal  = long_signal  & (df["p_long_state_gate"]  == 1)
        short_signal = short_signal & (df["p_short_state_gate"] == 1)

    df["signal"] = 0
    df.loc[long_signal,  "signal"] = 1
    df.loc[short_signal, "signal"] = -1

    logger.info(
        "Signals: long=%d  short=%d  (test window %s→%s, %d rows)",
        int(long_signal.sum()), int(short_signal.sum()),
        test_start.date(), test_end.date(), len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Per-ticker simulation
# ---------------------------------------------------------------------------

def _simulate_ticker(
    ticker: str,
    signals: pd.DataFrame,           # rows for this ticker, has ts_col, signal, prob_long, prob_short
    df_5m: Optional[pd.DataFrame],   # 5m bars for execution; None → fall back to df_30m
    df_30m: Optional[pd.DataFrame],  # raw 30m OHLCV bars; used for ATR/price and as exec fallback
    cfg: dict,
    equity_curve: list[float],
    equity_times: list[pd.Timestamp],
    initial_equity: float,
    ts_col: str,
) -> list[Trade]:
    """Simulate all trades for one ticker during the test period."""
    trades: list[Trade] = []
    if signals.empty:
        return trades

    # Execution bars: prefer 5m, fall back to raw 30m bars
    if df_5m is not None and not df_5m.empty:
        exec_df = df_5m
        max_holding_bars = 144 * 3   # 432 × 5m ≈ 3 days
    elif df_30m is not None and not df_30m.empty:
        exec_df = df_30m
        max_holding_bars = 24 * 3    # 72 × 30m ≈ 3 days
    else:
        return trades  # no execution data at all

    tp_mult = cfg["tp_atr_mult"]
    sl_mult = cfg["sl_atr_mult"]
    entry_bars_max  = cfg["entry_bars_max"]
    pos_size_pct    = cfg["position_size_pct"]
    commission_pct  = cfg["commission_pct"]

    # Build a timestamp → close lookup from raw 30m bars for ATR/price at signal time
    raw30_close: dict[pd.Timestamp, float] = {}
    raw30_atr14: dict[pd.Timestamp, float] = {}
    if df_30m is not None and not df_30m.empty:
        raw30_close = df_30m["close"].to_dict()
        if "atr_14" in df_30m.columns:
            raw30_atr14 = df_30m["atr_14"].to_dict()

    active: Optional[Position] = None

    exec_times = exec_df.index.tolist()
    exec_lookup = {ts: i for i, ts in enumerate(exec_times)}

    signal_rows = signals.sort_values(ts_col)

    for _, sig_row in signal_rows.iterrows():
        sig_ts  = pd.Timestamp(sig_row[ts_col])
        if sig_ts.tzinfo is None:
            sig_ts = sig_ts.tz_localize("UTC")
        signal  = int(sig_row["signal"])

        # Skip if no signal or already in a position
        if signal == 0 or active is not None:
            continue

        # ATR in absolute price terms at signal time
        # Prefer raw 30m atr_14; fall back to atr_pct_14 × close
        close_at_signal = raw30_close.get(sig_ts, np.nan)
        atr_val = raw30_atr14.get(sig_ts, np.nan)
        if np.isnan(atr_val) and not np.isnan(close_at_signal):
            atr_pct = float(sig_row.get("atr_pct_14", np.nan))
            if not np.isnan(atr_pct) and atr_pct > 0:
                atr_val = atr_pct * close_at_signal

        if np.isnan(atr_val) or atr_val <= 0:
            continue
        if np.isnan(close_at_signal):
            continue

        # -----------------------------------------------------------------
        # Find entry on next 10m bars after signal
        # -----------------------------------------------------------------
        # Locate first exec bar strictly after sig_ts
        entry_idx = None
        for ts in exec_times:
            if ts > sig_ts:
                entry_idx = exec_lookup[ts]
                break

        if entry_idx is None:
            continue

        # Try up to entry_bars_max bars to enter
        entry_ts    = None
        entry_price = None
        for k in range(entry_bars_max):
            idx = entry_idx + k
            if idx >= len(exec_times):
                break
            bar = exec_df.iloc[idx]
            ep  = float(bar.get("open", bar.get("close", np.nan)))
            if not np.isnan(ep):
                entry_ts    = exec_times[idx]
                entry_price = ep
                break

        if entry_ts is None or entry_price is None:
            continue

        # Equity at this point for position sizing
        current_equity = equity_curve[-1] if equity_curve else initial_equity
        notional = current_equity * pos_size_pct

        if signal == 1:    # long
            tp_price = entry_price + tp_mult * atr_val
            sl_price = entry_price - sl_mult * atr_val
        else:              # short
            tp_price = entry_price - tp_mult * atr_val
            sl_price = entry_price + sl_mult * atr_val

        active = Position(
            ticker=ticker,
            direction=signal,
            entry_time=entry_ts,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            atr_at_entry=atr_val,
            size=notional,
        )

        # -----------------------------------------------------------------
        # Track position until TP / SL / time expiry
        # -----------------------------------------------------------------
        exit_ts     = None
        exit_price  = None
        exit_reason = "time"

        for j in range(entry_idx + 1, min(entry_idx + max_holding_bars + 1, len(exec_times))):
            bar = exec_df.iloc[j]
            bar_h = float(bar.get("high",  close_at_signal))
            bar_l = float(bar.get("low",   close_at_signal))
            bar_c = float(bar.get("close", close_at_signal))

            if signal == 1:   # long
                if bar_l <= active.sl_price:
                    exit_ts, exit_price, exit_reason = exec_times[j], active.sl_price, "sl"
                    break
                if bar_h >= active.tp_price:
                    exit_ts, exit_price, exit_reason = exec_times[j], active.tp_price, "tp"
                    break
            else:              # short
                if bar_h >= active.sl_price:
                    exit_ts, exit_price, exit_reason = exec_times[j], active.sl_price, "sl"
                    break
                if bar_l <= active.tp_price:
                    exit_ts, exit_price, exit_reason = exec_times[j], active.tp_price, "tp"
                    break
        else:
            # Time expiry: exit at last bar's close
            last_idx = min(entry_idx + max_holding_bars, len(exec_times) - 1)
            exit_ts    = exec_times[last_idx]
            exit_price = float(exec_df.iloc[last_idx].get("close", entry_price))
            exit_reason = "time"

        if exit_ts is None:
            active = None
            continue

        # PnL
        raw_ret = (exit_price - entry_price) / entry_price * signal
        net_ret = raw_ret - commission_pct
        pnl_dollar = notional * net_ret

        trades.append(Trade(
            ticker=ticker,
            direction=signal,
            entry_time=active.entry_time,
            exit_time=exit_ts,
            entry_price=active.entry_price,
            exit_price=exit_price,
            pnl_pct=net_ret,
            pnl_dollar=pnl_dollar,
            exit_reason=exit_reason,
            size=notional,
        ))
        active = None

    return trades


# ---------------------------------------------------------------------------
# Portfolio-level simulation
# ---------------------------------------------------------------------------

def simulate(
    *,
    matrix_path: Path = TRAINING_MATRIX,
    model_path: Path = MODEL_PATH,
    results_dir: Path = BACKTEST_RESULTS_DIR,
    feature_columns: list[str] = FEATURE_COLUMNS,
    cfg: dict = BACKTEST_CONFIG,
    test_start_str: str = "2024-01-01",
    test_end_str: str   = "2026-04-15",
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full backtest. Returns (trades_df, equity_curve_df).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    trades_path = results_dir / "trades.parquet"
    equity_path = results_dir / "equity_curve.parquet"

    if trades_path.exists() and equity_path.exists() and not force:
        logger.info("Backtest results exist — loading from cache.")
        return pd.read_parquet(trades_path), pd.read_parquet(equity_path)

    # Load model
    if XGBClassifier is None:
        raise ImportError("xgboost required")
    clf = XGBClassifier()
    clf.load_model(str(model_path))
    logger.info("Loaded model from %s", model_path)

    # Load full matrix
    matrix = load_training_matrix(matrix_path)
    ts_col = _find_ts_col(matrix)

    test_start = pd.Timestamp(test_start_str, tz="UTC")
    test_end   = pd.Timestamp(test_end_str, tz="UTC")

    # Generate signals for entire test window (all tickers together)
    signals_all = generate_signals(matrix, clf, test_start, test_end, feature_columns, cfg)

    tickers = signals_all["ticker"].unique().tolist()
    logger.info("Running backtest for %d tickers (%s → %s)", len(tickers), test_start.date(), test_end.date())

    initial_equity = cfg["initial_capital"]
    equity: float  = initial_equity
    equity_times: list[pd.Timestamp] = [test_start]
    equity_vals:  list[float]        = [initial_equity]

    all_trades: list[Trade] = []
    open_positions: dict[str, Position] = {}

    for i, ticker in enumerate(tickers, 1):
        logger.info("  (%d/%d) %s", i, len(tickers), ticker)

        ticker_sigs = signals_all[signals_all["ticker"] == ticker].copy()
        if ticker_sigs.empty:
            continue

        df_5m  = _load_5m_slice(ticker, test_start, test_end)
        df_30m = _load_30m_slice(ticker, test_start, test_end)

        ticker_trades = _simulate_ticker(
            ticker=ticker,
            signals=ticker_sigs,
            df_5m=df_5m,
            df_30m=df_30m,
            cfg=cfg,
            equity_curve=equity_vals,
            equity_times=equity_times,
            initial_equity=initial_equity,
            ts_col=ts_col,
        )
        all_trades.extend(ticker_trades)

    # Build trades DataFrame
    if not all_trades:
        logger.warning("No trades generated in backtest window.")
        trades_df  = pd.DataFrame()
        equity_df  = pd.DataFrame({"timestamp": [test_start], "equity": [initial_equity]})
    else:
        trades_df = pd.DataFrame([vars(t) for t in all_trades]).sort_values("entry_time")

        # Reconstruct equity curve from sorted trades
        equity_series = [initial_equity]
        equity_ts     = [test_start]
        running = initial_equity
        for _, row in trades_df.iterrows():
            running += row["pnl_dollar"]
            equity_series.append(running)
            equity_ts.append(row["exit_time"])
        equity_df = pd.DataFrame({"timestamp": equity_ts, "equity": equity_series})

        # Summary stats
        total_trades = len(trades_df)
        win_rate     = (trades_df["pnl_pct"] > 0).mean()
        avg_win      = trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].mean()
        avg_loss     = trades_df.loc[trades_df["pnl_pct"] <= 0, "pnl_pct"].mean()
        total_pnl    = trades_df["pnl_dollar"].sum()
        max_dd       = _max_drawdown(equity_series)

        logger.info(
            "\n=== Backtest Summary ===\n"
            "  Trades:      %d\n"
            "  Win rate:    %.2f%%\n"
            "  Avg win:     %.2f%%\n"
            "  Avg loss:    %.2f%%\n"
            "  Total PnL:   $%.0f\n"
            "  Max DD:      %.2f%%\n"
            "  Final equity:$%.0f",
            total_trades, win_rate * 100,
            (avg_win  or 0) * 100,
            (avg_loss or 0) * 100,
            total_pnl, max_dd * 100, running,
        )

    trades_df.to_parquet(trades_path, index=False)
    equity_df.to_parquet(equity_path, index=False)
    logger.info("Results saved → %s", results_dir)
    return trades_df, equity_df


def _max_drawdown(equity: list[float]) -> float:
    arr = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd   = (arr - peak) / peak
    return float(dd.min())


def _find_ts_col(df: pd.DataFrame) -> str:
    for name in ("timestamp", "index", "t", "time", "date"):
        if name in df.columns:
            return name
    return df.columns[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-ticker swing backtest.")
    p.add_argument("--test-start", default="2024-01-01")
    p.add_argument("--test-end",   default="2026-04-15")
    p.add_argument("--force",      action="store_true")
    p.add_argument("--model",      default=None, help="Path to model .json (default: models/swing_xgb_model.json)")
    p.add_argument("--features",   default=None, help="Path to newline-delimited feature list .txt (default: FEATURE_COLUMNS from config)")
    p.add_argument("--results-dir", default=None, help="Directory to save results (default: backtest/results)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    kwargs = dict(test_start_str=args.test_start, test_end_str=args.test_end, force=args.force)
    if args.model:
        kwargs["model_path"] = Path(args.model)
    if args.results_dir:
        kwargs["results_dir"] = Path(args.results_dir)
    if args.features:
        feat_path = Path(args.features)
        kwargs["feature_columns"] = [l.strip() for l in feat_path.read_text().splitlines() if l.strip()]
        logger.info("Loaded %d features from %s", len(kwargs["feature_columns"]), feat_path)
    simulate(**kwargs)
