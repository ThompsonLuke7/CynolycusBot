"""
Sweep v3 — ablation study: realistic 5m entry + exit, no-progress timeout, opposite-signal exit.

Changes vs sweep_v2:
  - Exit tracked on 5m bars (not 30m) — intrabar SL/trail fire correctly
  - Entry at OPEN of next 5m bar after confirmation close (not confirming bar's close)
  - No-progress exit: cut trade if underwater with no movement after N 5m bars
  - Opposite-signal exit only fires when already in profit (configurable threshold)
  - Hard ATR stop optional (test with and without)

Ablation grid (fixed arm=2.5%, giveback=25%, threshold=0.60/0.70, confirm=6 bars):
  × sl_atr:            0.0 (none)  |  4.0
  × no_progress_bars:  78 (eod)  |  96 (8h)  |  156 (next_day)  |  None
  × opp_exit:          None  |  (0.60, 0.0%)  |  (0.60, 1.0%)  |  (0.65, 0.0%)  |  (0.65, 1.0%)

Run:
  python multi_ticker_swing/backtest/sweep_v3.py [--proba path] [--out-dir path] [--split test]
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.multi_ticker_swing.config.pipeline_config import (
    BACKTEST_RESULTS_DIR,
    RAW_30M_DIR,
    RAW_5M_DIR,
    TRADING_BLACKLIST,
    UNIVERSE_CSV,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

MODELS_DIR     = Path(__file__).resolve().parents[1] / "models"
PROBA_PATH     = MODELS_DIR / "p_swing_probs.parquet"
SWEEP_DIR      = BACKTEST_RESULTS_DIR / "sweep_v3"
COMMISSION_PCT = 0.001
BARS_5M_DAY    = 78    # 6.5h RTH × 12
MAX_HOLD_5M    = BARS_5M_DAY * 15   # 3-week hard cap

# ---------------------------------------------------------------------------
# Ablation grid
# ---------------------------------------------------------------------------
ENTRY_THRESHOLDS  = [0.60, 0.70]
CONFIRM_MAX_5M    = 6        # fixed at sweep_v2 best

_NP = {                      # no-progress timeout in 5m bars
    "eod":      BARS_5M_DAY,          # end of current trading day (~6.5h)
    "8h":       96,                    # 8 trading hours
    "next_day": BARS_5M_DAY * 2,      # end of next trading day
    "none":     None,
}
_OPP = {                     # (opp_threshold, min_underlying_profit_pct to allow exit)
    "none":       (None,  0.00),
    "opp60_any":  (0.60,  0.00),
    "opp60_1pct": (0.60,  0.01),
    "opp65_any":  (0.65,  0.00),
    "opp65_1pct": (0.65,  0.01),
}

EXIT_STRATEGIES: list[dict] = []
for _sl in (0.0, 4.0):
    for _np_name, _np_val in _NP.items():
        for _opp_name, (_opp_t, _opp_p) in _OPP.items():
            EXIT_STRATEGIES.append({
                "name":               f"sl{_sl}_np{_np_name}_{_opp_name}",
                "arm_pct":            0.025,
                "giveback_pct":       0.25,
                "sl_atr":             _sl,
                "no_progress_bars_5m": _np_val,
                "opp_exit":           _opp_t,
                "opp_min_profit":     _opp_p,
            })
# 2 sl × 4 np × 5 opp = 40 exit strategies × 2 thresholds = 80 combos


# ---------------------------------------------------------------------------
# Data loading  (identical to sweep_v2)
# ---------------------------------------------------------------------------

def load_proba(split: str = "test") -> pd.DataFrame:
    df = pd.read_parquet(PROBA_PATH)
    df = df[df["split"] == split].copy()
    p_dir = (df["p_long"] + df["p_short"]).clip(lower=1e-8)
    df["p_long_dir"]  = df["p_long"]  / p_dir
    df["p_short_dir"] = df["p_short"] / p_dir
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _normalise_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Promote timestamp index to a column if needed, lowercase all column names."""
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns and df.index.name == "timestamp":
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_raw_30m(ticker: str) -> pd.DataFrame:
    return _normalise_raw(pd.read_parquet(RAW_30M_DIR / f"{ticker}.parquet"))


def load_raw_5m(ticker: str) -> pd.DataFrame | None:
    path = RAW_5M_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    return _normalise_raw(pd.read_parquet(path))


def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    prev_c = np.roll(c, 1); prev_c[0] = np.nan
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    return pd.Series(tr).rolling(period, min_periods=period).mean().values


def compute_daily_ema_slope(raw_30m: pd.DataFrame, period: int = 20) -> np.ndarray:
    raw = raw_30m.copy()
    raw["date_et"] = pd.to_datetime(raw["timestamp"]).dt.tz_convert("America/New_York").dt.date
    daily = raw.groupby("date_et")["close"].last().reset_index()
    daily["ema"] = daily["close"].ewm(span=period, adjust=False).mean()
    daily["slope"] = daily["ema"].diff().shift(1)
    date_to_slope = dict(zip(daily["date_et"], daily["slope"]))
    slopes = np.array([date_to_slope.get(d, 0.0) or 0.0 for d in raw["date_et"]], dtype=np.float64)
    return slopes


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
# TickerData  (identical to sweep_v2)
# ---------------------------------------------------------------------------

class TickerData:
    __slots__ = ("ticker", "ts_30m", "open_30m", "high_30m", "low_30m", "close_30m",
                 "atr_30m", "daily_ema_slope", "n_30m", "p_long_dir", "p_short_dir",
                 "ts_5m", "open_5m", "high_5m", "low_5m", "close_5m",
                 "vol_5m", "vol_5m_mean20", "n_5m", "has_5m")

    def __init__(self, ticker, raw_30m, raw_5m, proba_ticker):
        self.ticker = ticker
        n = len(raw_30m)
        self.ts_30m    = raw_30m["timestamp"].values
        self.open_30m  = raw_30m["open"].values.astype(np.float64)
        self.high_30m  = raw_30m["high"].values.astype(np.float64)
        self.low_30m   = raw_30m["low"].values.astype(np.float64)
        self.close_30m = raw_30m["close"].values.astype(np.float64)
        self.atr_30m   = compute_atr(raw_30m)
        self.daily_ema_slope = compute_daily_ema_slope(raw_30m)
        self.n_30m     = n

        self.p_long_dir  = np.full(n, np.nan)
        self.p_short_dir = np.full(n, np.nan)
        ts_to_idx = pd.Series(np.arange(n), index=self.ts_30m)
        mask = np.isin(proba_ticker["timestamp"].values, self.ts_30m)
        matched = proba_ticker.iloc[mask.nonzero()[0]]
        if len(matched) > 0:
            idxs = ts_to_idx.loc[matched["timestamp"].values].values
            self.p_long_dir[idxs]  = matched["p_long_dir"].values
            self.p_short_dir[idxs] = matched["p_short_dir"].values

        if raw_5m is not None and len(raw_5m) > 0:
            self.ts_5m        = raw_5m["timestamp"].values
            self.open_5m      = raw_5m["open"].values.astype(np.float64)
            self.high_5m      = raw_5m["high"].values.astype(np.float64)
            self.low_5m       = raw_5m["low"].values.astype(np.float64)
            self.close_5m     = raw_5m["close"].values.astype(np.float64)
            raw_vol           = raw_5m["volume"].values.astype(np.float64)
            self.vol_5m       = raw_vol
            self.vol_5m_mean20 = pd.Series(raw_vol).rolling(20, min_periods=5).mean().values
            self.n_5m         = len(raw_5m)
            self.has_5m       = True
        else:
            self.ts_5m = self.open_5m = self.high_5m = self.low_5m = \
                self.close_5m = self.vol_5m = self.vol_5m_mean20 = np.array([], dtype=np.float64)
            self.n_5m   = 0
            self.has_5m = False


# ---------------------------------------------------------------------------
# 5m confirmation  (identical to sweep_v2)
# ---------------------------------------------------------------------------

def find_5m_confirmation(
    td: TickerData,
    signal_bar_idx: int,
    direction: int,
    max_5m_bars: int,
) -> tuple[float | None, int]:
    if not td.has_5m:
        return None, -1

    signal_ts = td.ts_30m[signal_bar_idx]
    ref_high  = td.high_30m[signal_bar_idx]
    ref_low   = td.low_30m[signal_bar_idx]

    start_5m    = int(np.searchsorted(td.ts_5m, signal_ts, side="right"))
    deadline_ts = signal_ts + np.timedelta64(max_5m_bars * 5, "m")
    end_5m      = min(start_5m + max_5m_bars, td.n_5m)

    for j in range(start_5m, end_5m):
        if td.ts_5m[j] > deadline_ts:
            break
        h, l, o, c = td.high_5m[j], td.low_5m[j], td.open_5m[j], td.close_5m[j]
        if direction == 1:
            if h >= ref_high and c > o and c > ref_high:
                return c, j
        else:
            if l <= ref_low and c < o and c < ref_low:
                return c, j
    return None, -1


# ---------------------------------------------------------------------------
# Core simulation — 5m exits, next-bar-open entry
# ---------------------------------------------------------------------------

def simulate_ticker_5m(
    td: TickerData,
    entry_threshold: float,
    confirm_max_5m: int,
    exit_cfg: dict,
) -> list[tuple]:
    """
    Signal on 30m bar → 5m breakout confirmation → enter at OPEN of next 5m bar.
    Exit tracked bar-by-bar on 5m: ATR stop, trailing stop, no-progress, opposite signal.
    Falls back to 30m exit if ticker has no 5m data.
    """
    if not td.has_5m:
        # Minimal 30m fallback — entry at close of confirming bar (no 5m available)
        return _simulate_30m_fallback(td, entry_threshold, confirm_max_5m, exit_cfg)

    sl_atr_mult  = float(exit_cfg.get("sl_atr", 0.0))
    arm_pct      = exit_cfg.get("arm_pct")
    giveback_pct = exit_cfg.get("giveback_pct")
    opp_thresh   = exit_cfg.get("opp_exit")
    opp_min_pct  = float(exit_cfg.get("opp_min_profit", 0.0))
    no_prog_bars = exit_cfg.get("no_progress_bars_5m")

    cooldown_ts = np.datetime64(0, "ns")
    n_30m = td.n_30m
    results = []

    for i in range(n_30m - 1):
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

        atr_val = td.atr_30m[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        # 5m breakout confirmation
        conf_price, conf_5m_idx = find_5m_confirmation(td, i, direction, confirm_max_5m)
        if conf_price is None:
            continue

        # Enter at OPEN of NEXT 5m bar after confirmation closes (realistic execution)
        entry_5m_idx = conf_5m_idx + 1
        if entry_5m_idx >= td.n_5m:
            continue
        entry_price = float(td.open_5m[entry_5m_idx])
        if np.isnan(entry_price) or entry_price <= 0:
            entry_price = conf_price  # gap open fallback

        sl_price = (entry_price - direction * sl_atr_mult * atr_val) if sl_atr_mult > 0 else None

        best_price  = entry_price
        trail_armed = False
        exit_5m_idx = min(entry_5m_idx + MAX_HOLD_5M, td.n_5m - 1)
        exit_price  = float(td.close_5m[exit_5m_idx])
        exit_reason = "time"

        for j in range(entry_5m_idx, min(entry_5m_idx + MAX_HOLD_5M + 1, td.n_5m)):
            bar_h = float(td.high_5m[j])
            bar_l = float(td.low_5m[j])
            bar_c = float(td.close_5m[j])

            # ── Hard ATR stop ────────────────────────────────────────────────
            if sl_price is not None:
                if direction == 1 and bar_l <= sl_price:
                    exit_5m_idx, exit_price, exit_reason = j, sl_price, "sl"
                    break
                if direction == -1 and bar_h >= sl_price:
                    exit_5m_idx, exit_price, exit_reason = j, sl_price, "sl"
                    break

            # ── Track best price ─────────────────────────────────────────────
            best_price = max(best_price, bar_h) if direction == 1 else min(best_price, bar_l)

            # ── Trailing stop (option-style % of underlying) ─────────────────
            if arm_pct is not None and giveback_pct is not None:
                move_pct = direction * (best_price - entry_price) / entry_price
                if move_pct >= arm_pct:
                    trail_armed = True
                if trail_armed:
                    peak_profit  = direction * (best_price - entry_price)
                    cur_profit   = direction * (bar_c - entry_price)
                    floor_profit = peak_profit * (1.0 - giveback_pct)
                    if cur_profit <= floor_profit:
                        exit_5m_idx = j
                        exit_price  = entry_price + direction * floor_profit
                        exit_reason = "trail"
                        break

            # ── No-progress exit ─────────────────────────────────────────────
            if no_prog_bars is not None and (j - entry_5m_idx) >= no_prog_bars:
                cur_pct = direction * (bar_c - entry_price) / entry_price
                if cur_pct <= 0.0 and not trail_armed:
                    exit_5m_idx, exit_price, exit_reason = j, bar_c, "no_progress"
                    break

            # ── Opposite-signal exit (only when profitable) ──────────────────
            if opp_thresh is not None:
                bar_ts      = td.ts_5m[j]
                idx_30m     = int(np.searchsorted(td.ts_30m, bar_ts, side="right")) - 1
                if 0 <= idx_30m < n_30m:
                    opp_p   = td.p_short_dir[idx_30m] if direction == 1 else td.p_long_dir[idx_30m]
                    cur_pct = direction * (bar_c - entry_price) / entry_price
                    if not np.isnan(opp_p) and opp_p >= opp_thresh and cur_pct >= opp_min_pct:
                        exit_5m_idx, exit_price, exit_reason = j, bar_c, "opp_signal"
                        break
        else:
            last        = min(entry_5m_idx + MAX_HOLD_5M, td.n_5m - 1)
            exit_5m_idx = last
            exit_price  = float(td.close_5m[last])
            exit_reason = "time"

        exit_5m_ts  = td.ts_5m[exit_5m_idx]
        exit_30m_idx = min(int(np.searchsorted(td.ts_30m, exit_5m_ts, side="right")), n_30m - 1)
        holding_5m  = exit_5m_idx - entry_5m_idx
        net_ret     = direction * (exit_price - entry_price) / entry_price - COMMISSION_PCT

        results.append((td.ticker, direction, i, exit_30m_idx,
                        entry_price, exit_price, net_ret, exit_reason, holding_5m))
        cooldown_ts = exit_5m_ts

    return results


def _simulate_30m_fallback(td, entry_threshold, confirm_max_5m, exit_cfg):
    """Minimal 30m fallback for tickers without 5m data — simplified exit."""
    sl_atr_mult  = float(exit_cfg.get("sl_atr", 0.0))
    arm_pct      = exit_cfg.get("arm_pct")
    giveback_pct = exit_cfg.get("giveback_pct")
    cooldown_ts  = np.datetime64(0, "ns")
    n = td.n_30m
    results = []
    for i in range(n - 1):
        if td.ts_30m[i] <= cooldown_ts:
            continue
        pl, ps = td.p_long_dir[i], td.p_short_dir[i]
        if np.isnan(pl):
            continue
        direction = 1 if pl >= entry_threshold else (-1 if ps >= entry_threshold else 0)
        if direction == 0:
            continue
        atr_val = td.atr_30m[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue
        entry_price = td.close_30m[i]
        sl_price = (entry_price - direction * sl_atr_mult * atr_val) if sl_atr_mult > 0 else None
        best_price, trail_armed = entry_price, False
        exit_idx, exit_price, exit_reason = min(i + 195, n - 1), td.close_30m[min(i + 195, n - 1)], "time"
        for j in range(i + 1, min(i + 196, n)):
            bh, bl, bc = td.high_30m[j], td.low_30m[j], td.close_30m[j]
            if sl_price and ((direction == 1 and bl <= sl_price) or (direction == -1 and bh >= sl_price)):
                exit_idx, exit_price, exit_reason = j, sl_price, "sl"; break
            best_price = max(best_price, bh) if direction == 1 else min(best_price, bl)
            if arm_pct and giveback_pct:
                if direction * (best_price - entry_price) / entry_price >= arm_pct:
                    trail_armed = True
                if trail_armed:
                    pk = direction * (best_price - entry_price)
                    if direction * (bc - entry_price) <= pk * (1 - giveback_pct):
                        exit_idx, exit_price, exit_reason = j, entry_price + direction * pk * (1 - giveback_pct), "trail"; break
        net_ret = direction * (exit_price - entry_price) / entry_price - COMMISSION_PCT
        results.append((td.ticker, direction, i, exit_idx, entry_price, exit_price, net_ret, exit_reason, exit_idx - i))
        cooldown_ts = td.ts_30m[exit_idx]
    return results


# ---------------------------------------------------------------------------
# Metrics  (identical to sweep_v2)
# ---------------------------------------------------------------------------

def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"n_trades": 0, "win_rate": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
                "profit_factor": 0, "avg_pnl_pct": 0, "total_pnl_pct": 0,
                "sharpe": 0, "max_dd_pct": 0, "avg_holding_bars": 0,
                "exit_sl_pct": 0, "exit_trail_pct": 0, "exit_time_pct": 0,
                "exit_opp_pct": 0, "exit_noprog_pct": 0, "long_n": 0, "short_n": 0,
                "long_wr": 0, "short_wr": 0}
    wins  = trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"]
    loss  = trades_df[trades_df["pnl_pct"] <= 0]["pnl_pct"]
    pf    = (wins.sum() / abs(loss.sum())) if len(loss) > 0 and loss.sum() != 0 else float("inf")
    rets  = trades_df["pnl_pct"].values
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    cumul = np.cumsum(rets)
    peak  = np.maximum.accumulate(cumul)
    dd    = (cumul - peak)
    longs  = trades_df[trades_df["direction"] == 1]
    shorts = trades_df[trades_df["direction"] == -1]
    exits  = trades_df["exit_reason"].value_counts(normalize=True)
    return {
        "n_trades":        len(trades_df),
        "win_rate":        (trades_df["pnl_pct"] > 0).mean(),
        "avg_win_pct":     wins.mean() if len(wins) else 0,
        "avg_loss_pct":    loss.mean() if len(loss) else 0,
        "profit_factor":   round(pf, 3),
        "avg_pnl_pct":     rets.mean(),
        "total_pnl_pct":   rets.sum(),
        "sharpe":          round(sharpe, 3),
        "max_dd_pct":      round(float(dd.min() * 100), 2),
        "avg_holding_bars": trades_df["holding_bars"].mean(),
        "exit_sl_pct":     exits.get("sl", 0),
        "exit_trail_pct":  exits.get("trail", 0),
        "exit_time_pct":   exits.get("time", 0),
        "exit_opp_pct":    exits.get("opp_signal", 0),
        "exit_noprog_pct": exits.get("no_progress", 0),
        "long_n":          len(longs),
        "short_n":         len(shorts),
        "long_wr":         (longs["pnl_pct"] > 0).mean() if len(longs) else 0,
        "short_wr":        (shorts["pnl_pct"] > 0).mean() if len(shorts) else 0,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

TRADE_COLS = ["ticker", "direction", "signal_idx", "exit_idx",
              "entry_price", "exit_price", "pnl_pct", "exit_reason", "holding_bars"]

_WORKER_TICKER_DATA: dict[str, TickerData] = {}


def _run_combo_worker(spec: tuple[int, int, float, dict]) -> tuple[int, dict, list[tuple], float]:
    k, n_combos, threshold, ex = spec
    t0 = time.time()
    combo_name = f"e{threshold}_c{CONFIRM_MAX_5M}_{ex['name']}"
    all_trades: list[tuple] = []
    for td in _WORKER_TICKER_DATA.values():
        trades = simulate_ticker_5m(td, threshold, CONFIRM_MAX_5M, ex)
        all_trades.extend(trades)

    tdf = pd.DataFrame(all_trades, columns=TRADE_COLS) if all_trades else pd.DataFrame(columns=TRADE_COLS)
    m = compute_metrics(tdf)
    row = {"combo_name": combo_name, "entry_threshold": threshold,
           "confirm_5m_bars": CONFIRM_MAX_5M, **ex, **m}
    return k, row, all_trades, time.time() - t0


def run_sweep(
    top_n: int = 200,
    split: str = "test",
    apply_blacklist: bool = True,
    jobs: int = 1,
) -> pd.DataFrame:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    n_5m_files = len(list(RAW_5M_DIR.glob("*.parquet"))) if RAW_5M_DIR.exists() else 0
    if n_5m_files < 50:
        logger.error("Only %d 5m files — fetch 5m data first.", n_5m_files)
        return pd.DataFrame()

    logger.info("Loading probabilities (split=%s)...", split)
    proba = load_proba(split)

    tickers = select_top_tickers(proba, top_n)
    if apply_blacklist:
        bl = [t for t in tickers if t in TRADING_BLACKLIST]
        tickers = [t for t in tickers if t not in TRADING_BLACKLIST]
        if bl:
            logger.info("Blacklist removed %d: %s", len(bl), bl)
    logger.info("Selected %d tickers", len(tickers))

    logger.info("Pre-building ticker data...")
    ticker_data: dict[str, TickerData] = {}
    for t in tickers:
        try:
            r30 = load_raw_30m(t)
            r5  = load_raw_5m(t)
            pt  = proba[proba["ticker"] == t]
            ticker_data[t] = TickerData(t, r30, r5, pt)
        except Exception as e:
            logger.warning("[%s] skipped: %s", t, e)
    logger.info("Built %d ticker objects (%d with 5m)", len(ticker_data),
                sum(1 for td in ticker_data.values() if td.has_5m))

    n_combos = len(ENTRY_THRESHOLDS) * len(EXIT_STRATEGIES)
    logger.info("Running %d combos...", n_combos)

    combo_specs = []
    k = 0
    for threshold in ENTRY_THRESHOLDS:
        for ex in EXIT_STRATEGIES:
            k += 1
            combo_specs.append((k, n_combos, threshold, ex))

    all_rows, best_trades, best_sharpe = [], [], -999.0
    global _WORKER_TICKER_DATA
    _WORKER_TICKER_DATA = ticker_data
    if jobs > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=jobs) as pool:
            results_iter = pool.imap_unordered(_run_combo_worker, combo_specs)
            for k, row, all_trades, elapsed in results_iter:
                all_rows.append(row)
                if row["sharpe"] > best_sharpe and row["n_trades"] >= 100:
                    best_sharpe = row["sharpe"]
                    best_trades = all_trades
                logger.info("(%d/%d) %-52s %5d trades  WR=%.1f%%  PF=%.2f  Sharpe=%.2f  [%.1fs]",
                            k, n_combos, row["combo_name"], row["n_trades"],
                            row["win_rate"] * 100, row["profit_factor"], row["sharpe"], elapsed)
    else:
        for spec in combo_specs:
            k, _, _, _ = spec
            k, row, all_trades, elapsed = _run_combo_worker(spec)
            all_rows.append(row)
            if row["sharpe"] > best_sharpe and row["n_trades"] >= 100:
                best_sharpe = row["sharpe"]
                best_trades = all_trades
            logger.info("(%d/%d) %-52s %5d trades  WR=%.1f%%  PF=%.2f  Sharpe=%.2f  [%.1fs]",
                        k, n_combos, row["combo_name"], row["n_trades"],
                        row["win_rate"] * 100, row["profit_factor"], row["sharpe"], elapsed)

    summary_df = pd.DataFrame(all_rows).sort_values("sharpe", ascending=False)
    summary_df.to_csv( SWEEP_DIR / "sweep_v3_summary.csv",    index=False)
    summary_df.to_parquet(SWEEP_DIR / "sweep_v3_summary.parquet", index=False)

    if best_trades:
        pd.DataFrame(best_trades, columns=TRADE_COLS).to_parquet(
            SWEEP_DIR / "best_v3_trades.parquet", index=False)

    # Per-ticker breakdown for best combo
    best_combo = summary_df.iloc[0]["combo_name"]
    best_row   = summary_df.iloc[0]
    best_cfg = {k: best_row[k] for k in
                ["arm_pct", "giveback_pct", "sl_atr", "no_progress_bars_5m",
                 "opp_exit", "opp_min_profit"]}
    best_thr = float(best_row["entry_threshold"])
    per_ticker = []
    for td in ticker_data.values():
        trades = simulate_ticker_5m(td, best_thr, CONFIRM_MAX_5M, best_cfg)
        if not trades:
            continue
        tdf = pd.DataFrame(trades, columns=TRADE_COLS)
        m   = compute_metrics(tdf)
        per_ticker.append({"ticker": td.ticker, **m})
    per_df = pd.DataFrame(per_ticker).sort_values("sharpe", ascending=False)
    per_df.to_csv(SWEEP_DIR / "best_v3_per_ticker.csv", index=False)

    logger.info("\n=== Top 5 combos by Sharpe ===")
    for _, r in summary_df.head(5).iterrows():
        logger.info("  %-52s  trades=%d  WR=%.1f%%  PF=%.3f  Sharpe=%.3f  MaxDD=%.1f%%",
                    r["combo_name"], r["n_trades"], r["win_rate"] * 100,
                    r["profit_factor"], r["sharpe"], r["max_dd_pct"])
    logger.info("Sweep v3 complete. Results → %s", SWEEP_DIR)
    return summary_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-n",       type=int, default=200)
    p.add_argument("--split",       default="test", choices=["train", "val", "test"])
    p.add_argument("--no-blacklist", action="store_true")
    p.add_argument("--proba",       default=None)
    p.add_argument("--out-dir",     default=None)
    p.add_argument("--jobs",        type=int, default=1)
    args = p.parse_args()
    if args.proba:
        PROBA_PATH = Path(args.proba)
    if args.out_dir:
        SWEEP_DIR = Path(args.out_dir)
    run_sweep(top_n=args.top_n, split=args.split, apply_blacklist=not args.no_blacklist, jobs=max(1, args.jobs))
