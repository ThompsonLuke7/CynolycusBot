"""
Generate swing state machine label plots for 5 random tickers over the last month
on 1-hour candles.
"""
from __future__ import annotations

import logging
from pathlib import Path
import random

import numpy as np
import pandas as pd

from Data.plots.plots import plot_swing_state_machine_signals
from Features.label_generations import (
    add_pivot_swing_state_machine,
    add_triple_barrier_labels_atr,
)
from Features.feature_sets.custom_indicators import add_fractal_pivots
from multi_ticker_swing.config.pipeline_config import (
    MODULE_ROOT,
    UNIVERSE_CSV,
    RAW_30M_DIR,
    SWING_STATE_CONFIG,
    TRIPLE_BARRIER_CONFIG,
)
from multi_ticker_swing.data.fetch_data import load_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PLOTS_DIR = MODULE_ROOT / "plots" / "swing_label_plots"


def resample_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 30m OHLCV to 1-hour."""
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")

    df_1h = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    df_1h = df_1h.dropna()
    return df_1h


def compute_atr_simple(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Compute ATR for 1-hour data."""
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=length).mean()
    return atr


def build_1h_labels(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """
    Compute swing labels on 1-hour data.

    This replicates the label-building logic from multi_ticker_swing/labels/build_labels.py
    but works on 1-hour candles.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("[%s] missing columns for labels: %s", ticker, missing)
        return None

    # Compute ATR
    df["atr_14"] = compute_atr_simple(df, length=14)

    # Add fractal pivots (OK for labels since they're forward-looking)
    try:
        df = add_fractal_pivots(df)
    except Exception as exc:
        logger.warning("[%s] add_fractal_pivots failed: %s", ticker, exc)
        df["pivot_up"] = 0
        df["pivot_down"] = 0

    # Ensure pivots exist
    if "pivot_up" not in df.columns:
        df["pivot_up"] = 0
    if "pivot_down" not in df.columns:
        df["pivot_down"] = 0

    # Triple-barrier labels
    try:
        df = add_triple_barrier_labels_atr(
            df,
            close_col="close",
            high_col="high",
            low_col="low",
            open_col="open",
            atr_col="atr_14",
            atr_length=TRIPLE_BARRIER_CONFIG["atr_length"],
            k_up=TRIPLE_BARRIER_CONFIG["k_up"],
            k_dn=TRIPLE_BARRIER_CONFIG["k_dn"],
            max_holding=TRIPLE_BARRIER_CONFIG["max_holding"],
            chop_atr_mult=TRIPLE_BARRIER_CONFIG["chop_atr_mult"],
            base_label_col="tb_label",
        )
    except Exception as exc:
        logger.warning("[%s] triple-barrier labels failed: %s", ticker, exc)
        df["tb_label"] = 0

    # Swing state machine (binary fallback)
    try:
        df = add_pivot_swing_state_machine(
            df,
            atr_col="atr_14",
            atr_length=SWING_STATE_CONFIG.get("atr_length", 14),
            threshold=SWING_STATE_CONFIG["threshold"],
            confirm_threshold=SWING_STATE_CONFIG["confirm_threshold"],
            confirm_mult=SWING_STATE_CONFIG["confirm_mult"],
            pending_max_bars=SWING_STATE_CONFIG["pending_max_bars"],
            tp_mult=SWING_STATE_CONFIG["tp_mult"],
            sl_mult=SWING_STATE_CONFIG["sl_mult"],
            max_holding=SWING_STATE_CONFIG["max_holding"],
            cooldown_bars=SWING_STATE_CONFIG["cooldown_bars"],
            session_open_minutes=SWING_STATE_CONFIG["session_open_minutes"],
            session_late_minutes=SWING_STATE_CONFIG["session_late_minutes"],
            allow_binary_fallback=True,
        )
    except Exception as exc:
        logger.warning("[%s] swing state machine failed: %s", ticker, exc)
        df["p_long_state_gate"] = 0
        df["p_short_state_gate"] = 0
        df["p_long_pending"] = 0
        df["p_short_pending"] = 0

    return df


def plot_ticker_1h_labels(ticker: str, days_back: int = 30) -> None:
    """Load 30m data, resample to 1h, compute labels, and plot."""
    logger.info("Processing %s", ticker)

    # Load 30m data
    data_path = RAW_30M_DIR / f"{ticker}.parquet"
    if not data_path.exists():
        logger.warning("[%s] 30m data not found at %s", ticker, data_path)
        return

    df_30m = pd.read_parquet(data_path)

    # Ensure timestamp index
    if isinstance(df_30m.index, pd.DatetimeIndex):
        df_30m["timestamp"] = df_30m.index

    # Filter to last N days
    if "timestamp" in df_30m.columns:
        df_30m["timestamp"] = pd.to_datetime(df_30m["timestamp"])
        # Handle timezone-aware timestamps
        if df_30m["timestamp"].dt.tz is not None:
            cutoff = pd.Timestamp.now(tz=df_30m["timestamp"].dt.tz) - pd.Timedelta(days=days_back)
        else:
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
        df_30m = df_30m[df_30m["timestamp"] >= cutoff]

    if len(df_30m) < 50:
        logger.warning("[%s] Not enough data after filtering (only %d bars)", ticker, len(df_30m))
        return

    # Resample to 1h
    df_1h = resample_to_1h(df_30m)

    if len(df_1h) < 20:
        logger.warning("[%s] Not enough 1h bars after resampling (only %d bars)", ticker, len(df_1h))
        return

    # Build labels
    df_labeled = build_1h_labels(df_1h, ticker)
    if df_labeled is None:
        logger.warning("[%s] Failed to build labels", ticker)
        return

    # Plot
    output_dir = PLOTS_DIR / ticker
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(output_dir / "swing_state_1h_labels.png")

    try:
        plot_swing_state_machine_signals(
            df_labeled,
            long_state_col="p_long_state_gate",
            short_state_col="p_short_state_gate",
            long_pending_col="p_long_pending",
            short_pending_col="p_short_pending",
            tail=None,  # Use all data
            save_path=save_path,
        )
        logger.info("[%s] Plot saved → %s", ticker, save_path)
    except Exception as exc:
        logger.error("[%s] Failed to plot: %s", ticker, exc, exc_info=True)


def main():
    # Load universe
    universe = load_universe(UNIVERSE_CSV)
    tickers = universe["ticker"].tolist()

    # Select 5 random tickers
    random.seed(42)
    selected = random.sample(tickers, min(5, len(tickers)))

    logger.info("Selected tickers: %s", selected)

    for ticker in selected:
        plot_ticker_1h_labels(ticker, days_back=30)

    logger.info("Done! Plots saved to %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
