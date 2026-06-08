"""
Sweep v4 — tier-based ablation: ATR MFE no-progress exit + first-30-min entry filter.

Tickers split into 3 tiers from sweep_v3 per-ticker Sharpe:
  Tier 1 (Strong):   Sharpe ≥ 2.0   (~93 tickers)
  Tier 2 (Marginal): 0 ≤ Sharpe < 2.0 (~61 tickers)
  Tier 3 (Negative): Sharpe < 0      (~34 tickers)

Each tier gets an independent ablation; best config per tier drives live trading.

Changes vs sweep_v3:
  - ATR MFE no-progress: at bar N, if MFE < X×ATR → exit (replaces time-only check)
  - First-30-min entry filter: no new entries in 9:30–10:00 ET; exits unaffected
  - No opposite-signal exit (uniformly hurt in sweep_v3)

Grid per tier (arm=2.5%, giveback=25%, confirm=6 fixed):
  entry threshold: 0.60 | 0.70
  sl_atr:          0.0  | 4.0
  ATR no-progress: None | N∈{48,78} × MFE_thresh∈{-0.25, 0, 0.25, 0.5, 1.0}
  = 2 × 2 × 11 = 44 combos per tier, 132 total

Run:
  python multi_ticker_swing/backtest/sweep_v4.py \
      --proba multi_ticker_swing/models/p_swing_probs.parquet \
      --v3-per-ticker multi_ticker_swing/backtest/results/sweep_v3/best_v3_per_ticker.csv \
      --out-dir multi_ticker_swing/backtest/results/sweep_v4 \
      --split test
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from multi_ticker_swing.config.pipeline_config import (
    BACKTEST_RESULTS_DIR,
    RAW_30M_DIR,
    RAW_5M_DIR,
    TRADING_BLACKLIST,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

MODELS_DIR     = Path(__file__).resolve().parents[1] / "models"
PROBA_PATH     = MODELS_DIR / "p_swing_probs.parquet"
V3_PER_TICKER  = BACKTEST_RESULTS_DIR / "sweep_v3" / "best_v3_per_ticker.csv"
OUT_DIR        = BACKTEST_RESULTS_DIR / "sweep_v4"

COMMISSION_PCT  = 0.001
BARS_5M_DAY     = 78
MAX_HOLD_5M     = BARS_5M_DAY * 15     # 3-week hard cap (seldom hit)

# ---------------------------------------------------------------------------
# Ablation grid
# ---------------------------------------------------------------------------
ENTRY_THRESHOLDS = [0.60, 0.70]
CONFIRM_MAX_5M   = 6   # fixed at sweep_v2/v3 best

_ATR_NP: list[dict] = [{"name": "npnone", "n_bars": None, "mfe_atr": None}]
for _n, _label in ((48, "np48"), (78, "np78")):
    for _t, _tname in ((-0.25, "mfem025"), (0.0, "mfe0"), (0.25, "mfe025"),
                       (0.50, "mfe05"),    (1.0,  "mfe10")):
        _ATR_NP.append({"name": f"{_label}_{_tname}", "n_bars": _n, "mfe_atr": _t})

EXIT_STRATEGIES: list[dict] = []
for _sl in (0.0, 4.0):
    for _np in _ATR_NP:
        EXIT_STRATEGIES.append({
            "name":        f"sl{_sl}_{_np['name']}",
            "arm_pct":     0.025,
            "giveback_pct": 0.25,
            "sl_atr":      _sl,
            "np_n_bars":   _np["n_bars"],
            "np_mfe_atr":  _np["mfe_atr"],
        })
# 2 sl × 11 np = 22 exit strategies × 2 thresholds = 44 combos per tier


# ---------------------------------------------------------------------------
# Data loading
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
    """Promote timestamp index to column if needed, lowercase all column names."""
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
    daily["ema"]   = daily["close"].ewm(span=period, adjust=False).mean()
    daily["slope"] = daily["ema"].diff().shift(1)
    date_to_slope  = dict(zip(daily["date_et"], daily["slope"]))
    slopes = np.array([date_to_slope.get(d, 0.0) or 0.0 for d in raw["date_et"]], dtype=np.float64)
    return slopes


# ---------------------------------------------------------------------------
# TickerData
# ---------------------------------------------------------------------------

class TickerData:
    __slots__ = (
        "ticker", "ts_30m", "open_30m", "high_30m", "low_30m", "close_30m",
        "atr_30m", "daily_ema_slope", "n_30m", "p_long_dir", "p_short_dir",
        "ts_5m", "open_5m", "high_5m", "low_5m", "close_5m",
        "vol_5m", "vol_5m_mean20", "n_5m", "has_5m",
        "is_first_30min_5m",
    )

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
            self.ts_5m         = raw_5m["timestamp"].values
            self.open_5m       = raw_5m["open"].values.astype(np.float64)
            self.high_5m       = raw_5m["high"].values.astype(np.float64)
            self.low_5m        = raw_5m["low"].values.astype(np.float64)
            self.close_5m      = raw_5m["close"].values.astype(np.float64)
            raw_vol            = raw_5m["volume"].values.astype(np.float64)
            self.vol_5m        = raw_vol
            self.vol_5m_mean20 = pd.Series(raw_vol).rolling(20, min_periods=5).mean().values
            self.n_5m          = len(raw_5m)
            self.has_5m        = True

            # Precompute first-30-min mask: 9:30–9:55 ET (RTH open)
            ts_et = pd.DatetimeIndex(raw_5m["timestamp"]).tz_convert("America/New_York")
            self.is_first_30min_5m = np.asarray((ts_et.hour == 9) & (ts_et.minute >= 30))
        else:
            self.ts_5m = self.open_5m = self.high_5m = self.low_5m = \
                self.close_5m = self.vol_5m = self.vol_5m_mean20 = np.array([], dtype=np.float64)
            self.n_5m = 0
            self.has_5m = False
            self.is_first_30min_5m = np.array([], dtype=bool)


# ---------------------------------------------------------------------------
# 5m confirmation  (unchanged from sweep_v3)
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
# Core simulation — 5m exits, next-bar-open entry, ATR MFE no-progress
# ---------------------------------------------------------------------------

def simulate_ticker_5m(
    td: TickerData,
    entry_threshold: float,
    confirm_max_5m: int,
    exit_cfg: dict,
) -> list[tuple]:
    if not td.has_5m:
        return _simulate_30m_fallback(td, entry_threshold, confirm_max_5m, exit_cfg)

    sl_atr_mult  = float(exit_cfg.get("sl_atr", 0.0))
    arm_pct      = exit_cfg.get("arm_pct")
    giveback_pct = exit_cfg.get("giveback_pct")
    np_n_bars    = exit_cfg.get("np_n_bars")       # bars before ATR no-progress check
    np_mfe_atr   = exit_cfg.get("np_mfe_atr")      # MFE threshold in ATR units (can be negative)

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

        conf_price, conf_5m_idx = find_5m_confirmation(td, i, direction, confirm_max_5m)
        if conf_price is None:
            continue

        # Enter at OPEN of next 5m bar after confirmation closes
        entry_5m_idx = conf_5m_idx + 1
        if entry_5m_idx >= td.n_5m:
            continue

        # ── First-30-min entry filter: skip if entry bar is in 9:30–9:55 ET ──
        if td.is_first_30min_5m[entry_5m_idx]:
            continue

        entry_price = float(td.open_5m[entry_5m_idx])
        if np.isnan(entry_price) or entry_price <= 0:
            entry_price = conf_price

        sl_price    = (entry_price - direction * sl_atr_mult * atr_val) if sl_atr_mult > 0 else None
        best_price  = entry_price   # tracks max favorable excursion (MFE)
        trail_armed = False
        exit_5m_idx = min(entry_5m_idx + MAX_HOLD_5M, td.n_5m - 1)
        exit_price  = float(td.close_5m[exit_5m_idx])
        exit_reason = "time"

        for j in range(entry_5m_idx, min(entry_5m_idx + MAX_HOLD_5M + 1, td.n_5m)):
            bar_h = float(td.high_5m[j])
            bar_l = float(td.low_5m[j])
            bar_c = float(td.close_5m[j])

            # ── Hard ATR stop ─────────────────────────────────────────────────
            if sl_price is not None:
                if direction == 1 and bar_l <= sl_price:
                    exit_5m_idx, exit_price, exit_reason = j, sl_price, "sl"
                    break
                if direction == -1 and bar_h >= sl_price:
                    exit_5m_idx, exit_price, exit_reason = j, sl_price, "sl"
                    break

            # ── Track MFE (best_price = max favorable excursion from entry) ───
            best_price = max(best_price, bar_h) if direction == 1 else min(best_price, bar_l)

            # ── Trailing stop (option-style % of underlying) ──────────────────
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

            # ── ATR MFE no-progress: check ONCE at bar np_n_bars ─────────────
            # Exits if MFE hasn't reached np_mfe_atr × ATR after N bars.
            # Negative threshold allows the trade to be slightly underwater.
            if np_n_bars is not None and (j - entry_5m_idx) == np_n_bars and not trail_armed:
                mfe_atr_units = direction * (best_price - entry_price) / atr_val
                if mfe_atr_units < np_mfe_atr:
                    exit_5m_idx, exit_price, exit_reason = j, bar_c, "no_progress"
                    break
        else:
            last        = min(entry_5m_idx + MAX_HOLD_5M, td.n_5m - 1)
            exit_5m_idx = last
            exit_price  = float(td.close_5m[last])
            exit_reason = "time"

        exit_5m_ts   = td.ts_5m[exit_5m_idx]
        exit_30m_idx = min(int(np.searchsorted(td.ts_30m, exit_5m_ts, side="right")), n_30m - 1)
        holding_5m   = exit_5m_idx - entry_5m_idx
        net_ret      = direction * (exit_price - entry_price) / entry_price - COMMISSION_PCT

        results.append((td.ticker, direction, i, exit_30m_idx,
                        entry_price, exit_price, net_ret, exit_reason, holding_5m))
        cooldown_ts = exit_5m_ts

    return results


def _simulate_30m_fallback(td, entry_threshold, confirm_max_5m, exit_cfg):
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
        sl_price    = (entry_price - direction * sl_atr_mult * atr_val) if sl_atr_mult > 0 else None
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
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"n_trades": 0, "win_rate": 0, "avg_win_pct": 0, "avg_loss_pct": 0,
                "profit_factor": 0, "avg_pnl_pct": 0, "total_pnl_pct": 0,
                "sharpe": 0, "max_dd_pct": 0, "avg_holding_bars": 0,
                "exit_sl_pct": 0, "exit_trail_pct": 0, "exit_time_pct": 0,
                "exit_noprog_pct": 0, "long_n": 0, "short_n": 0,
                "long_wr": 0, "short_wr": 0}
    wins  = trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"]
    loss  = trades_df[trades_df["pnl_pct"] <= 0]["pnl_pct"]
    pf    = (wins.sum() / abs(loss.sum())) if len(loss) > 0 and loss.sum() != 0 else float("inf")
    rets  = trades_df["pnl_pct"].values
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    cumul = np.cumsum(rets)
    peak  = np.maximum.accumulate(cumul)
    dd    = cumul - peak
    longs  = trades_df[trades_df["direction"] == 1]
    shorts = trades_df[trades_df["direction"] == -1]
    exits  = trades_df["exit_reason"].value_counts(normalize=True)
    return {
        "n_trades":         len(trades_df),
        "win_rate":         (trades_df["pnl_pct"] > 0).mean(),
        "avg_win_pct":      wins.mean() if len(wins) else 0,
        "avg_loss_pct":     loss.mean() if len(loss) else 0,
        "profit_factor":    round(pf, 3),
        "avg_pnl_pct":      rets.mean(),
        "total_pnl_pct":    rets.sum(),
        "sharpe":           round(sharpe, 3),
        "max_dd_pct":       round(float(dd.min() * 100), 2),
        "avg_holding_bars": trades_df["holding_bars"].mean(),
        "exit_sl_pct":      exits.get("sl", 0),
        "exit_trail_pct":   exits.get("trail", 0),
        "exit_time_pct":    exits.get("time", 0),
        "exit_noprog_pct":  exits.get("no_progress", 0),
        "long_n":           len(longs),
        "short_n":          len(shorts),
        "long_wr":          (longs["pnl_pct"] > 0).mean() if len(longs) else 0,
        "short_wr":         (shorts["pnl_pct"] > 0).mean() if len(shorts) else 0,
    }


# ---------------------------------------------------------------------------
# Per-tier sweep
# ---------------------------------------------------------------------------

TRADE_COLS = ["ticker", "direction", "signal_idx", "exit_idx",
              "entry_price", "exit_price", "pnl_pct", "exit_reason", "holding_bars"]

_WORKER_TICKER_DATA: dict[str, TickerData] = {}


def _run_tier_combo_worker(spec: tuple[str, int, int, list[str], float, dict]) -> tuple[int, dict, list[tuple], float]:
    tier_name, k, n_combos, tickers, threshold, ex = spec
    t0 = time.time()
    combo_name = f"e{threshold}_c{CONFIRM_MAX_5M}_{ex['name']}"
    all_trades: list[tuple] = []
    for t in tickers:
        if t not in _WORKER_TICKER_DATA:
            continue
        trades = simulate_ticker_5m(_WORKER_TICKER_DATA[t], threshold, CONFIRM_MAX_5M, ex)
        all_trades.extend(trades)

    tdf = pd.DataFrame(all_trades, columns=TRADE_COLS) if all_trades else pd.DataFrame(columns=TRADE_COLS)
    m = compute_metrics(tdf)
    row = {"combo_name": combo_name, "tier": tier_name,
           "entry_threshold": threshold, "confirm_5m_bars": CONFIRM_MAX_5M,
           **ex, **m}
    return k, row, all_trades, time.time() - t0


def run_tier_sweep(
    tier_name: str,
    tickers: list[str],
    ticker_data: dict,
    out_dir: Path,
    jobs: int = 1,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tickers = len([t for t in tickers if t in ticker_data])
    n_combos  = len(ENTRY_THRESHOLDS) * len(EXIT_STRATEGIES)
    logger.info("=== %s: %d tickers, %d combos ===", tier_name, n_tickers, n_combos)

    combo_specs = []
    k = 0
    for threshold in ENTRY_THRESHOLDS:
        for ex in EXIT_STRATEGIES:
            k += 1
            combo_specs.append((tier_name, k, n_combos, tickers, threshold, ex))

    all_rows, best_trades, best_sharpe = [], [], -999.0
    global _WORKER_TICKER_DATA
    _WORKER_TICKER_DATA = ticker_data
    if jobs > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=jobs) as pool:
            results_iter = pool.imap_unordered(_run_tier_combo_worker, combo_specs)
            for k, row, all_trades, elapsed in results_iter:
                all_rows.append(row)
                if row["sharpe"] > best_sharpe and row["n_trades"] >= 30:
                    best_sharpe = row["sharpe"]
                    best_trades = all_trades
                logger.info("[%s] (%d/%d) %-52s %4d trades  WR=%.1f%%  PF=%.2f  Sharpe=%.2f  [%.1fs]",
                            tier_name, k, n_combos, row["combo_name"], row["n_trades"],
                            row["win_rate"] * 100, row["profit_factor"], row["sharpe"], elapsed)
    else:
        for spec in combo_specs:
            k, row, all_trades, elapsed = _run_tier_combo_worker(spec)
            all_rows.append(row)
            if row["sharpe"] > best_sharpe and row["n_trades"] >= 30:
                best_sharpe = row["sharpe"]
                best_trades = all_trades
            logger.info("[%s] (%d/%d) %-52s %4d trades  WR=%.1f%%  PF=%.2f  Sharpe=%.2f  [%.1fs]",
                        tier_name, k, n_combos, row["combo_name"], row["n_trades"],
                        row["win_rate"] * 100, row["profit_factor"], row["sharpe"], elapsed)

    summary_df = pd.DataFrame(all_rows).sort_values("sharpe", ascending=False)
    summary_df.to_csv(out_dir / f"summary_{tier_name}.csv", index=False)
    summary_df.to_parquet(out_dir / f"summary_{tier_name}.parquet", index=False)

    if best_trades:
        pd.DataFrame(best_trades, columns=TRADE_COLS).to_parquet(
            out_dir / f"best_trades_{tier_name}.parquet", index=False)

    # Per-ticker breakdown for best combo
    if not summary_df.empty:
        best_row = summary_df.iloc[0]
        best_cfg = {k: best_row[k] for k in
                    ["arm_pct", "giveback_pct", "sl_atr", "np_n_bars", "np_mfe_atr"]}
        best_thr = float(best_row["entry_threshold"])
        per_ticker = []
        for t in tickers:
            if t not in ticker_data:
                continue
            trades = simulate_ticker_5m(ticker_data[t], best_thr, CONFIRM_MAX_5M, best_cfg)
            if not trades:
                continue
            tdf = pd.DataFrame(trades, columns=TRADE_COLS)
            m   = compute_metrics(tdf)
            per_ticker.append({"ticker": t, "combo": best_row["combo_name"], **m})
        if per_ticker:
            per_df = pd.DataFrame(per_ticker).sort_values("sharpe", ascending=False)
            per_df.to_csv(out_dir / f"per_ticker_{tier_name}.csv", index=False)

    logger.info("[%s] Best: %s  Sharpe=%.3f", tier_name,
                summary_df.iloc[0]["combo_name"] if not summary_df.empty else "none",
                best_sharpe)
    return summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sweep(split: str = "test", top_n: int = 200, jobs: int = 1) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tier assignments from sweep_v3 per-ticker results
    if not V3_PER_TICKER.exists():
        logger.error("sweep_v3 per-ticker file not found: %s", V3_PER_TICKER)
        logger.error("Run sweep_v3.py first.")
        return
    v3_pt = pd.read_csv(V3_PER_TICKER)
    tier1_tickers = v3_pt[v3_pt["sharpe"] >= 2.0]["ticker"].tolist()
    tier2_tickers = v3_pt[(v3_pt["sharpe"] >= 0.0) & (v3_pt["sharpe"] < 2.0)]["ticker"].tolist()
    tier3_tickers = v3_pt[v3_pt["sharpe"] < 0.0]["ticker"].tolist()
    logger.info("Tiers from sweep_v3: T1=%d  T2=%d  T3=%d",
                len(tier1_tickers), len(tier2_tickers), len(tier3_tickers))

    # Load probabilities
    logger.info("Loading probabilities (split=%s)...", split)
    proba = load_proba(split)

    # Pre-build ALL ticker data once
    all_tickers = list(set(tier1_tickers + tier2_tickers + tier3_tickers))
    # Also filter against blacklist
    all_tickers = [t for t in all_tickers if t not in TRADING_BLACKLIST]
    tier1_tickers = [t for t in tier1_tickers if t not in TRADING_BLACKLIST]
    tier2_tickers = [t for t in tier2_tickers if t not in TRADING_BLACKLIST]
    tier3_tickers = [t for t in tier3_tickers if t not in TRADING_BLACKLIST]

    logger.info("Pre-building ticker data for %d tickers...", len(all_tickers))
    ticker_data: dict[str, TickerData] = {}
    for t in all_tickers:
        try:
            r30 = load_raw_30m(t)
            r5  = load_raw_5m(t)
            pt  = proba[proba["ticker"] == t]
            ticker_data[t] = TickerData(t, r30, r5, pt)
        except Exception as e:
            logger.warning("[%s] skipped: %s", t, e)
    logger.info("Built %d ticker objects (%d with 5m)", len(ticker_data),
                sum(1 for td in ticker_data.values() if td.has_5m))

    # Run per-tier sweeps
    results = {}
    for tier_name, tickers in (("tier1", tier1_tickers), ("tier2", tier2_tickers), ("tier3", tier3_tickers)):
        tier_dir = OUT_DIR / tier_name
        summary  = run_tier_sweep(tier_name, tickers, ticker_data, tier_dir, jobs=jobs)
        results[tier_name] = summary

    # Save combined summary
    combined = pd.concat(results.values(), ignore_index=True).sort_values(["tier", "sharpe"], ascending=[True, False])
    combined.to_csv(OUT_DIR / "sweep_v4_combined.csv", index=False)

    logger.info("\n=== sweep_v4 complete ===")
    for tier_name, df in results.items():
        if not df.empty:
            r = df.iloc[0]
            logger.info("  %-6s best: %-52s  trades=%d  WR=%.1f%%  PF=%.3f  Sharpe=%.3f",
                        tier_name, r["combo_name"], r["n_trades"],
                        r["win_rate"] * 100, r["profit_factor"], r["sharpe"])
    logger.info("Results → %s", OUT_DIR)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split",        default="test", choices=["train", "val", "test"])
    p.add_argument("--top-n",        type=int, default=200)
    p.add_argument("--proba",        default=None)
    p.add_argument("--v3-per-ticker", default=None)
    p.add_argument("--out-dir",      default=None)
    p.add_argument("--jobs",         type=int, default=1)
    args = p.parse_args()
    if args.proba:
        PROBA_PATH = Path(args.proba)
    if args.v3_per_ticker:
        V3_PER_TICKER = Path(args.v3_per_ticker)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    run_sweep(split=args.split, top_n=args.top_n, jobs=max(1, args.jobs))
