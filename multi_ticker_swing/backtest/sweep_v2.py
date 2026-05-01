"""
Backtest sweep v2: proper entry confirmation on 5m bars + option-style trailing exits.

Entry logic (mirrors SPY intraday phase4_bodyclose_bodyclose):
  1. 30m bar fires signal (P(long|dir) >= threshold)
  2. On next N 5m bars, wait for breakout confirmation:
     - Long: 5m bar high >= signal bar high AND close > open AND close > signal bar high
     - Short: 5m bar low <= signal bar low AND close < open AND close < signal bar low
  3. Entry at the 5m bar's close price (confirmed breakout price)

Exit logic (option_adaptive_trail style):
  - Trailing arm: wait until underlying move >= arm_pct of entry (proxy for option doubling)
  - Giveback: once armed, exit if profit retraces giveback_pct from peak
  - Hard stop: exit if underlying moves against by sl_atr × ATR
  - Opposite signal exit: exit if opposite P(dir) >= opp_threshold
  - Time decay: exit if no progress after max_bars and underwater

Run:
  python -m multi_ticker_swing.backtest.sweep_v2 [--top-n 100] [--split test]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from multi_ticker_swing.config.pipeline_config import (
    BACKTEST_RESULTS_DIR,
    RAW_30M_DIR,
    RAW_5M_DIR,
    TRADING_BLACKLIST,
    UNIVERSE_CSV,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

PROBA_PATH = Path(__file__).resolve().parents[1] / "models" / "p_swing_probs.parquet"
SWEEP_DIR = BACKTEST_RESULTS_DIR / "sweep_v2"

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
ENTRY_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
CONFIRM_MAX_BARS_5M = [6, 12, 18]  # 30min, 1hr, 1.5hr windows for confirmation

EXIT_STRATEGIES: list[dict] = [
    # Option-style trailing: arm_pct = underlying move to arm trail, giveback_pct = fraction of peak profit to give back
    # For monthly ATM calls (~0.40 delta), 2.5% underlying move ≈ 100% option gain
    {"name": "trail_arm2.5_gb25",  "arm_pct": 0.025, "giveback_pct": 0.25, "sl_atr": 1.0, "opp_exit": None,  "max_days": None},
    {"name": "trail_arm2.5_gb33",  "arm_pct": 0.025, "giveback_pct": 0.33, "sl_atr": 1.0, "opp_exit": None,  "max_days": None},
    {"name": "trail_arm5_gb25",    "arm_pct": 0.050, "giveback_pct": 0.25, "sl_atr": 1.0, "opp_exit": None,  "max_days": None},
    {"name": "trail_arm5_gb33",    "arm_pct": 0.050, "giveback_pct": 0.33, "sl_atr": 1.0, "opp_exit": None,  "max_days": None},
    {"name": "trail_arm2.5_gb25_opp65", "arm_pct": 0.025, "giveback_pct": 0.25, "sl_atr": 1.0, "opp_exit": 0.65, "max_days": None},
    {"name": "trail_arm2.5_gb25_opp60", "arm_pct": 0.025, "giveback_pct": 0.25, "sl_atr": 1.0, "opp_exit": 0.60, "max_days": None},
    # Wider stop
    {"name": "trail_arm2.5_gb25_sl1.5", "arm_pct": 0.025, "giveback_pct": 0.25, "sl_atr": 1.5, "opp_exit": None, "max_days": None},
    {"name": "trail_arm2.5_gb25_sl2.0", "arm_pct": 0.025, "giveback_pct": 0.25, "sl_atr": 2.0, "opp_exit": None, "max_days": None},
    {"name": "trail_arm2.5_gb25_sl2.5", "arm_pct": 0.025, "giveback_pct": 0.25, "sl_atr": 2.5, "opp_exit": None, "max_days": None},
    {"name": "trail_arm2.5_gb25_sl3.0", "arm_pct": 0.025, "giveback_pct": 0.25, "sl_atr": 3.0, "opp_exit": None, "max_days": None},
    {"name": "trail_arm2.5_gb25_sl4.0", "arm_pct": 0.025, "giveback_pct": 0.25, "sl_atr": 4.0, "opp_exit": None, "max_days": None},
    {"name": "trail_arm5_gb25_sl1.5",   "arm_pct": 0.050, "giveback_pct": 0.25, "sl_atr": 1.5, "opp_exit": None, "max_days": None},
    # Time-capped (for weeklies)
    {"name": "trail_arm2.5_gb25_5d", "arm_pct": 0.025, "giveback_pct": 0.25, "sl_atr": 1.0, "opp_exit": None, "max_days": 5},
    {"name": "trail_arm2.5_gb25_10d","arm_pct": 0.025, "giveback_pct": 0.25, "sl_atr": 1.0, "opp_exit": None, "max_days": 10},
    # ATR-based trailing (non-option, for comparison)
    {"name": "atr_trail_1.5_sl1",  "arm_pct": None, "giveback_pct": None, "sl_atr": 1.0, "opp_exit": None, "max_days": None, "trail_atr": 1.5, "trail_arm_atr": 1.0},
    {"name": "atr_trail_2_sl1",    "arm_pct": None, "giveback_pct": None, "sl_atr": 1.0, "opp_exit": None, "max_days": None, "trail_atr": 2.0, "trail_arm_atr": 1.0},
]

MAX_HOLDING_BARS_30M = 195  # ~15 trading days hard cap
COMMISSION_PCT = 0.001
BARS_5M_PER_DAY = 78  # 6.5hr RTH × 12 bars/hr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_proba(split: str = "test") -> pd.DataFrame:
    df = pd.read_parquet(PROBA_PATH)
    df = df[df["split"] == split].copy()
    p_dir = (df["p_long"] + df["p_short"]).clip(lower=1e-8)
    df["p_long_dir"] = df["p_long"] / p_dir
    df["p_short_dir"] = df["p_short"] / p_dir
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def load_raw_30m(ticker: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW_30M_DIR / f"{ticker}.parquet")
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_raw_5m(ticker: str) -> pd.DataFrame | None:
    path = RAW_5M_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(period, min_periods=period).mean().values


def select_top_tickers(proba: pd.DataFrame, top_n: int) -> list[str]:
    tickers = [t for t in proba["ticker"].unique() if (RAW_30M_DIR / f"{t}.parquet").exists()]
    if top_n >= len(tickers):
        return tickers
    dvol = {}
    for t in tickers:
        raw = pd.read_parquet(RAW_30M_DIR / f"{t}.parquet")
        raw.columns = [c.lower() for c in raw.columns]
        dvol[t] = (raw["close"].values * raw["volume"].values).mean()
    return sorted(dvol, key=dvol.get, reverse=True)[:top_n]


# ---------------------------------------------------------------------------
# Pre-built ticker data
# ---------------------------------------------------------------------------

def compute_daily_ema_slope(raw_30m: pd.DataFrame, period: int = 20) -> np.ndarray:
    """
    Compute daily EMA(period) slope aligned to 30m bar indices.
    Uses PREVIOUS day's close to avoid lookahead — safe to use at signal time.
    Returns array of length n_30m: positive = daily uptrend, negative = downtrend, 0 = unknown.
    """
    raw = raw_30m.copy()
    raw["date_et"] = pd.to_datetime(raw["timestamp"]).dt.tz_convert("America/New_York").dt.date
    daily = raw.groupby("date_et")["close"].last().reset_index()
    daily["ema"] = daily["close"].ewm(span=period, adjust=False).mean()
    # Slope using previous day's ema so no lookahead
    daily["slope"] = daily["ema"].diff()
    # Shift by 1: slope known at start of next day (after prior close)
    daily["slope_prev"] = daily["slope"].shift(1)
    date_to_slope = dict(zip(daily["date_et"], daily["slope_prev"]))

    slopes = np.zeros(len(raw), dtype=np.float64)
    for idx, date in enumerate(raw["date_et"]):
        s = date_to_slope.get(date, np.nan)
        slopes[idx] = 0.0 if np.isnan(s) else s
    return slopes


class TickerData:
    __slots__ = ("ticker", "ts_30m", "open_30m", "high_30m", "low_30m", "close_30m",
                 "atr_30m", "daily_ema_slope", "n_30m", "p_long_dir", "p_short_dir",
                 "ts_5m", "open_5m", "high_5m", "low_5m", "close_5m", "vol_5m",
                 "vol_5m_mean20", "n_5m", "has_5m")

    def __init__(self, ticker, raw_30m, raw_5m, proba_ticker):
        self.ticker = ticker
        n = len(raw_30m)
        self.ts_30m = raw_30m["timestamp"].values
        self.open_30m = raw_30m["open"].values.astype(np.float64)
        self.high_30m = raw_30m["high"].values.astype(np.float64)
        self.low_30m = raw_30m["low"].values.astype(np.float64)
        self.close_30m = raw_30m["close"].values.astype(np.float64)
        self.atr_30m = compute_atr(raw_30m)
        self.daily_ema_slope = compute_daily_ema_slope(raw_30m)
        self.n_30m = n

        # Map probabilities to 30m bar indices
        self.p_long_dir = np.full(n, np.nan)
        self.p_short_dir = np.full(n, np.nan)
        ts_to_idx = pd.Series(np.arange(n), index=self.ts_30m)
        mask = np.isin(proba_ticker["timestamp"].values, self.ts_30m)
        matched = proba_ticker.iloc[mask.nonzero()[0]]
        if len(matched) > 0:
            idxs = ts_to_idx.loc[matched["timestamp"].values].values
            self.p_long_dir[idxs] = matched["p_long_dir"].values
            self.p_short_dir[idxs] = matched["p_short_dir"].values

        # 5m data
        if raw_5m is not None and len(raw_5m) > 0:
            self.ts_5m = raw_5m["timestamp"].values
            self.open_5m = raw_5m["open"].values.astype(np.float64)
            self.high_5m = raw_5m["high"].values.astype(np.float64)
            self.low_5m = raw_5m["low"].values.astype(np.float64)
            self.close_5m = raw_5m["close"].values.astype(np.float64)
            raw_vol = raw_5m["volume"].values.astype(np.float64)
            self.vol_5m = raw_vol
            # Rolling 20-bar mean volume — used for volume confirmation filter
            self.vol_5m_mean20 = pd.Series(raw_vol).rolling(20, min_periods=5).mean().values
            self.n_5m = len(raw_5m)
            self.has_5m = True
        else:
            self.ts_5m = np.array([], dtype="datetime64[ns]")
            self.open_5m = np.array([], dtype=np.float64)
            self.high_5m = np.array([], dtype=np.float64)
            self.low_5m = np.array([], dtype=np.float64)
            self.close_5m = np.array([], dtype=np.float64)
            self.vol_5m = np.array([], dtype=np.float64)
            self.vol_5m_mean20 = np.array([], dtype=np.float64)
            self.n_5m = 0
            self.has_5m = False


# ---------------------------------------------------------------------------
# 5m confirmation logic
# ---------------------------------------------------------------------------

def find_5m_confirmation(
    td: TickerData,
    signal_bar_idx: int,
    direction: int,
    max_5m_bars: int,
    min_vol_mult: float = 0.0,
) -> tuple[float | None, int]:
    """
    Look for breakout confirmation on 5m bars after the 30m signal bar.

    For longs: 5m bar high >= signal bar high AND close > open AND close > signal bar high
    For shorts: 5m bar low <= signal bar low AND close < open AND close < signal bar low

    Confirmation must occur within max_5m_bars × 5 minutes of wall-clock time from the
    signal bar close — overnight gaps are excluded (no confirming into the next day's open).

    min_vol_mult: confirming 5m bar volume must be >= min_vol_mult × 20-bar rolling mean.
    Default 0.0 = disabled. Values around 1.2–1.5 filter thin false breakouts but reduce
    trade count and Sharpe — empirically not beneficial with current model.

    Returns (entry_price, 5m_bar_idx) or (None, -1) if no confirmation found.
    """
    if not td.has_5m:
        return None, -1

    signal_ts = td.ts_30m[signal_bar_idx]
    ref_high = td.high_30m[signal_bar_idx]
    ref_low = td.low_30m[signal_bar_idx]

    # Find first 5m bar strictly after signal bar timestamp
    start_5m = np.searchsorted(td.ts_5m, signal_ts, side="right")

    # Hard wall-clock deadline: max_5m_bars × 5 minutes of actual time
    # This prevents confirming into overnight gaps (e.g. signal at close, entry at next open)
    deadline_ts = signal_ts + np.timedelta64(max_5m_bars * 5, "m")

    end_5m = min(start_5m + max_5m_bars, td.n_5m)

    for j in range(start_5m, end_5m):
        if td.ts_5m[j] > deadline_ts:
            break

        h = td.high_5m[j]
        l = td.low_5m[j]
        o = td.open_5m[j]
        c = td.close_5m[j]

        # Volume confirmation: reject thin breakouts below min_vol_mult × rolling mean
        if min_vol_mult > 0.0:
            mean_vol = td.vol_5m_mean20[j]
            if not np.isnan(mean_vol) and mean_vol > 0:
                if td.vol_5m[j] < min_vol_mult * mean_vol:
                    continue

        if direction == 1:
            if h >= ref_high and c > o and c > ref_high:
                return c, j
        else:
            if l <= ref_low and c < o and c < ref_low:
                return c, j

    return None, -1


# ---------------------------------------------------------------------------
# Simulation with confirmation + option trailing
# ---------------------------------------------------------------------------

def simulate_ticker(
    td: TickerData,
    entry_threshold: float,
    confirm_max_5m: int,
    exit_cfg: dict,
    trend_filter: bool = False,
) -> list[tuple]:
    """Returns list of tuples: (ticker, dir, entry_ts_idx, exit_ts_idx, entry_price, exit_price, pnl_pct, exit_reason, holding_bars_30m)"""
    results = []

    sl_atr_mult = exit_cfg["sl_atr"]
    arm_pct = exit_cfg.get("arm_pct")
    giveback_pct = exit_cfg.get("giveback_pct")
    opp_exit_thresh = exit_cfg.get("opp_exit")
    max_days = exit_cfg.get("max_days")
    trail_atr_mult = exit_cfg.get("trail_atr")
    trail_arm_atr = exit_cfg.get("trail_arm_atr")

    max_hold_30m = int(max_days * 13) if max_days else MAX_HOLDING_BARS_30M  # ~13 30m bars/day

    cooldown_ts = np.datetime64(0, "ns")
    n = td.n_30m

    for i in range(n - 1):
        if td.ts_30m[i] <= cooldown_ts:
            continue

        pl = td.p_long_dir[i]
        ps = td.p_short_dir[i]
        if np.isnan(pl):
            continue

        direction = 0
        if pl >= entry_threshold:
            direction = 1
        elif ps >= entry_threshold:
            direction = -1
        if direction == 0:
            continue

        # Daily trend filter: only take longs in uptrend, shorts in downtrend
        # Uses previous day's EMA slope — no lookahead
        if trend_filter:
            slope = td.daily_ema_slope[i]
            if direction == 1 and slope <= 0:
                continue
            if direction == -1 and slope >= 0:
                continue

        atr_val = td.atr_30m[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # --- 5m breakout confirmation ---
        entry_price, confirm_5m_idx = find_5m_confirmation(td, i, direction, confirm_max_5m)
        if entry_price is None:
            continue

        if np.isnan(entry_price) or entry_price <= 0:
            continue

        # --- Exit simulation on 30m bars (starting from next bar after signal) ---
        entry_30m_idx = i + 1
        if entry_30m_idx >= n:
            continue

        sl_price = entry_price - direction * sl_atr_mult * atr_val

        # Trailing state
        best_price = entry_price
        trail_armed = False

        exit_30m_idx = min(entry_30m_idx + max_hold_30m, n - 1)
        exit_price = td.close_30m[exit_30m_idx]
        exit_reason = "time"

        for j in range(entry_30m_idx, min(entry_30m_idx + max_hold_30m + 1, n)):
            bar_h = td.high_30m[j]
            bar_l = td.low_30m[j]
            bar_c = td.close_30m[j]

            # Hard stop loss
            if direction == 1 and bar_l <= sl_price:
                exit_30m_idx, exit_price, exit_reason = j, sl_price, "sl"
                break
            elif direction == -1 and bar_h >= sl_price:
                exit_30m_idx, exit_price, exit_reason = j, sl_price, "sl"
                break

            # Update best price
            if direction == 1:
                best_price = max(best_price, bar_h)
            else:
                best_price = min(best_price, bar_l)

            # Option-style percentage trailing
            if arm_pct is not None and giveback_pct is not None:
                move_pct = direction * (best_price - entry_price) / entry_price
                if move_pct >= arm_pct:
                    trail_armed = True

                if trail_armed:
                    peak_profit = direction * (best_price - entry_price)
                    current_profit = direction * (bar_c - entry_price)
                    floor_profit = peak_profit * (1.0 - giveback_pct)
                    if current_profit <= floor_profit:
                        floor_price = entry_price + direction * floor_profit
                        exit_30m_idx, exit_price, exit_reason = j, floor_price, "trail"
                        break

            # ATR-based trailing (alternative)
            if trail_atr_mult is not None and trail_arm_atr is not None:
                move_atr = direction * (best_price - entry_price) / atr_val
                if move_atr >= trail_arm_atr:
                    trail_armed = True
                if trail_armed:
                    if direction == 1:
                        trail_level = best_price - trail_atr_mult * atr_val
                        if bar_l <= trail_level:
                            exit_30m_idx, exit_price, exit_reason = j, trail_level, "trail"
                            break
                    else:
                        trail_level = best_price + trail_atr_mult * atr_val
                        if bar_h >= trail_level:
                            exit_30m_idx, exit_price, exit_reason = j, trail_level, "trail"
                            break

            # Opposite signal exit
            if opp_exit_thresh is not None:
                opp = td.p_short_dir[j] if direction == 1 else td.p_long_dir[j]
                if not np.isnan(opp) and opp >= opp_exit_thresh:
                    current_profit = direction * (bar_c - entry_price) / entry_price
                    if current_profit > 0:
                        exit_30m_idx, exit_price, exit_reason = j, bar_c, "opp_signal"
                        break

        holding_bars = exit_30m_idx - entry_30m_idx
        raw_ret = direction * (exit_price - entry_price) / entry_price
        net_ret = raw_ret - COMMISSION_PCT

        results.append((
            td.ticker, direction, i, exit_30m_idx,
            entry_price, exit_price, net_ret, exit_reason, holding_bars,
        ))
        cooldown_ts = td.ts_30m[exit_30m_idx]

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {
            "n_trades": 0, "win_rate": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "profit_factor": 0.0, "avg_pnl_pct": 0.0, "total_pnl_pct": 0.0,
            "sharpe": 0.0, "max_dd_pct": 0.0, "avg_holding_bars": 0.0,
            "exit_sl_pct": 0.0, "exit_trail_pct": 0.0, "exit_time_pct": 0.0,
            "exit_opp_pct": 0.0,
            "long_n": 0, "short_n": 0, "long_wr": 0.0, "short_wr": 0.0,
        }

    rets = trades_df["pnl_pct"].values
    wins = rets > 0
    losses = ~wins
    gp = rets[wins].sum() if wins.any() else 0.0
    gl = abs(rets[losses].sum()) if losses.any() else 0.0
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    cum = np.cumsum(rets)
    max_dd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) > 0 else 0.0
    dirs = trades_df["direction"].values
    er = trades_df["exit_reason"].values

    return {
        "n_trades": len(trades_df),
        "win_rate": float(wins.mean()),
        "avg_win_pct": float(rets[wins].mean()) if wins.any() else 0.0,
        "avg_loss_pct": float(rets[losses].mean()) if losses.any() else 0.0,
        "profit_factor": round(pf, 3),
        "avg_pnl_pct": float(rets.mean()),
        "total_pnl_pct": float(rets.sum()),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "avg_holding_bars": float(trades_df["holding_bars"].mean()),
        "exit_sl_pct": float((er == "sl").mean()),
        "exit_trail_pct": float((er == "trail").mean()),
        "exit_time_pct": float((er == "time").mean()),
        "exit_opp_pct": float((er == "opp_signal").mean()),
        "long_n": int((dirs == 1).sum()),
        "short_n": int((dirs == -1).sum()),
        "long_wr": float((rets[dirs == 1] > 0).mean()) if (dirs == 1).any() else 0.0,
        "short_wr": float((rets[dirs == -1] > 0).mean()) if (dirs == -1).any() else 0.0,
    }


def compute_grouped_metrics(trades_df: pd.DataFrame) -> dict:
    uni = pd.read_csv(UNIVERSE_CSV)
    meta = {row["ticker"]: row for _, row in uni.iterrows()}
    trades_df = trades_df.copy()
    trades_df["sector"] = trades_df["ticker"].map(lambda t: meta.get(t, {}).get("sector", "Unknown"))
    trades_df["cap_bucket"] = trades_df["ticker"].map(lambda t: meta.get(t, {}).get("market_cap_bucket", "Unknown"))
    trades_df["asset_type"] = trades_df["ticker"].map(lambda t: meta.get(t, {}).get("type", "Stock"))

    groups = {}
    for col, prefix in [("sector", "sector"), ("cap_bucket", "cap"), ("asset_type", "type")]:
        for val, grp in trades_df.groupby(col):
            if len(grp) >= 5:
                groups[f"{prefix}:{val}"] = compute_metrics(grp)
    for d, grp in trades_df.groupby("direction"):
        groups[f"dir:{'long' if d == 1 else 'short'}"] = compute_metrics(grp)
    return groups


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(
    top_n: int = 100,
    split: str = "test",
    trend_filter: bool = False,
    apply_blacklist: bool = True,
) -> pd.DataFrame:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    # Check 5m data availability
    n_5m_files = len(list(RAW_5M_DIR.glob("*.parquet"))) if RAW_5M_DIR.exists() else 0
    if n_5m_files < 50:
        logger.error("Only %d 5m files found in %s — need at least 50. Fetch 5m data first.", n_5m_files, RAW_5M_DIR)
        return pd.DataFrame()

    logger.info("Loading probabilities (split=%s)...", split)
    proba = load_proba(split)

    logger.info("Selecting top %d tickers by dollar volume...", top_n)
    tickers = select_top_tickers(proba, top_n)

    if apply_blacklist:
        blacklisted = [t for t in tickers if t in TRADING_BLACKLIST]
        tickers = [t for t in tickers if t not in TRADING_BLACKLIST]
        if blacklisted:
            logger.info("Blacklist removed %d tickers: %s", len(blacklisted), blacklisted)
    logger.info("Selected %d tickers", len(tickers))

    logger.info("Pre-building ticker data (30m + 5m)...")
    ticker_data: dict[str, TickerData] = {}
    no_5m = []
    for t in tickers:
        try:
            raw_30m = load_raw_30m(t)
            raw_5m = load_raw_5m(t)
            proba_t = proba[proba["ticker"] == t]
            td = TickerData(t, raw_30m, raw_5m, proba_t)
            if not td.has_5m:
                no_5m.append(t)
                continue
            ticker_data[t] = td
        except Exception as e:
            logger.warning("Skipping %s: %s", t, e)

    logger.info("Built data for %d tickers (%d skipped: no 5m data)", len(ticker_data), len(no_5m))
    if no_5m:
        logger.info("  Missing 5m: %s", ", ".join(no_5m[:10]) + ("..." if len(no_5m) > 10 else ""))

    total_combos = len(ENTRY_THRESHOLDS) * len(CONFIRM_MAX_BARS_5M) * len(EXIT_STRATEGIES)
    logger.info("Running %d combos × %d tickers  [trend_filter=%s]", total_combos, len(ticker_data), trend_filter)

    results = []
    combo_idx = 0

    for entry_thresh in ENTRY_THRESHOLDS:
        for confirm_bars in CONFIRM_MAX_BARS_5M:
            for exit_cfg in EXIT_STRATEGIES:
                combo_idx += 1
                combo_name = f"e{entry_thresh}_c{confirm_bars}_{exit_cfg['name']}"
                t0 = time.time()

                all_rows = []
                for td in ticker_data.values():
                    trades = simulate_ticker(td, entry_thresh, confirm_bars, exit_cfg, trend_filter=trend_filter)
                    all_rows.extend(trades)

                if all_rows:
                    trades_df = pd.DataFrame(all_rows, columns=[
                        "ticker", "direction", "signal_idx", "exit_idx",
                        "entry_price", "exit_price", "pnl_pct", "exit_reason", "holding_bars",
                    ])
                else:
                    trades_df = pd.DataFrame()

                metrics = compute_metrics(trades_df)
                metrics["combo_name"] = combo_name
                metrics["entry_threshold"] = entry_thresh
                metrics["confirm_5m_bars"] = confirm_bars
                metrics["exit_strategy"] = exit_cfg["name"]
                for k in ("arm_pct", "giveback_pct", "sl_atr", "opp_exit", "max_days"):
                    metrics[k] = exit_cfg.get(k)
                results.append(metrics)

                elapsed = time.time() - t0
                logger.info(
                    "(%d/%d) %s — %d trades, WR=%.1f%%, PF=%.2f, Sharpe=%.2f  [%.1fs]",
                    combo_idx, total_combos, combo_name,
                    metrics["n_trades"], metrics["win_rate"] * 100,
                    metrics["profit_factor"], metrics["sharpe"], elapsed,
                )

    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    results_df.to_csv(SWEEP_DIR / "sweep_v2_summary.csv", index=False)
    results_df.to_parquet(SWEEP_DIR / "sweep_v2_summary.parquet", index=False)

    logger.info("\n=== TOP 15 COMBOS BY SHARPE ===")
    for _, row in results_df.head(15).iterrows():
        logger.info(
            "  %-45s  n=%5d  WR=%.1f%%  PF=%.2f  Sharpe=%+.2f  avgPnL=%+.3f%%  hold=%.0f  SL%%=%.0f  trail%%=%.0f  opp%%=%.0f  time%%=%.0f",
            row["combo_name"], row["n_trades"], row["win_rate"] * 100,
            row["profit_factor"], row["sharpe"], row["avg_pnl_pct"] * 100,
            row["avg_holding_bars"],
            row["exit_sl_pct"] * 100, row["exit_trail_pct"] * 100,
            row["exit_opp_pct"] * 100, row["exit_time_pct"] * 100,
        )

    # Grouped analysis on best combo
    if not results_df.empty:
        best = results_df.iloc[0]
        best_exit_cfg = next(e for e in EXIT_STRATEGIES if e["name"] == best["exit_strategy"])
        best_confirm = int(best["confirm_5m_bars"])

        logger.info("\n=== DETAILED ANALYSIS: %s ===", best["combo_name"])

        all_rows = []
        for td in ticker_data.values():
            trades = simulate_ticker(td, best["entry_threshold"], best_confirm, best_exit_cfg, trend_filter=trend_filter)
            all_rows.extend(trades)

        if all_rows:
            best_trades_df = pd.DataFrame(all_rows, columns=[
                "ticker", "direction", "signal_idx", "exit_idx",
                "entry_price", "exit_price", "pnl_pct", "exit_reason", "holding_bars",
            ])
            best_trades_df.to_parquet(SWEEP_DIR / "best_v2_trades.parquet", index=False)

            grouped = compute_grouped_metrics(best_trades_df)
            with open(SWEEP_DIR / "best_v2_grouped.json", "w") as f:
                json.dump(grouped, f, indent=2, default=str)

            for gn, gm in sorted(grouped.items()):
                if gm["n_trades"] >= 10:
                    logger.info(
                        "  %-35s  n=%4d  WR=%.1f%%  PF=%.2f  Sharpe=%+.2f  avgPnL=%+.3f%%",
                        gn, gm["n_trades"], gm["win_rate"] * 100,
                        gm["profit_factor"], gm["sharpe"], gm["avg_pnl_pct"] * 100,
                    )

            # Per-ticker
            ticker_metrics = []
            for ticker, grp in best_trades_df.groupby("ticker"):
                m = compute_metrics(grp)
                m["ticker"] = ticker
                ticker_metrics.append(m)
            ticker_df = pd.DataFrame(ticker_metrics).sort_values("sharpe", ascending=False)
            ticker_df.to_csv(SWEEP_DIR / "best_v2_per_ticker.csv", index=False)

            logger.info("\n=== TOP 15 TICKERS ===")
            for _, row in ticker_df.head(15).iterrows():
                logger.info("  %-6s  n=%3d  WR=%.1f%%  PF=%.2f  Sharpe=%+.2f  totalPnL=%+.2f%%",
                            row["ticker"], row["n_trades"], row["win_rate"] * 100,
                            row["profit_factor"], row["sharpe"], row["total_pnl_pct"] * 100)

            logger.info("\n=== BOTTOM 10 TICKERS ===")
            for _, row in ticker_df.tail(10).iterrows():
                logger.info("  %-6s  n=%3d  WR=%.1f%%  PF=%.2f  Sharpe=%+.2f  totalPnL=%+.2f%%",
                            row["ticker"], row["n_trades"], row["win_rate"] * 100,
                            row["profit_factor"], row["sharpe"], row["total_pnl_pct"] * 100)

    logger.info("\nSweep v2 complete. Results → %s", SWEEP_DIR)
    return results_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--trend-filter", action="store_true", help="Only trade direction aligned with daily EMA(20) slope")
    p.add_argument("--no-blacklist", action="store_true", help="Disable TRADING_BLACKLIST filtering (for diagnostics)")
    p.add_argument("--proba", default=None, help="Path to p_swing_probs parquet (overrides PROBA_PATH)")
    p.add_argument("--out-dir", default=None, help="Override SWEEP_DIR output directory")
    args = p.parse_args()
    if args.proba:
        PROBA_PATH = Path(args.proba)
    if args.out_dir:
        SWEEP_DIR = Path(args.out_dir)
    run_sweep(top_n=args.top_n, split=args.split, trend_filter=args.trend_filter, apply_blacklist=not args.no_blacklist)
