"""
Backtest sweep: test combinations of entry thresholds × exit strategies
on the saved model probabilities (p_swing_probs.parquet) + raw 30m OHLCV.

Uses directional-conditional probability: P(long|dir) = P(long)/(P(long)+P(short)).

Run:
  python -m multi_ticker_swing.backtest.sweep_experiments [--top-n 100] [--split test]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.multi_ticker_swing.config.pipeline_config import (
    BACKTEST_RESULTS_DIR,
    RAW_30M_DIR,
    UNIVERSE_CSV,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

PROBA_PATH = Path(__file__).resolve().parents[1] / "models" / "p_swing_probs.parquet"
SWEEP_RESULTS_DIR = BACKTEST_RESULTS_DIR / "sweep"

# ---------------------------------------------------------------------------
# Sweep parameter grid
# ---------------------------------------------------------------------------
ENTRY_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

EXIT_STRATEGIES: list[dict] = [
    {"name": "tp2_sl1",      "tp_atr": 2.0, "sl_atr": 1.0, "prob_exit": None, "trail_atr": None, "time_bars": None},
    {"name": "tp3_sl1",      "tp_atr": 3.0, "sl_atr": 1.0, "prob_exit": None, "trail_atr": None, "time_bars": None},
    {"name": "tp4_sl1",      "tp_atr": 4.0, "sl_atr": 1.0, "prob_exit": None, "trail_atr": None, "time_bars": None},
    {"name": "tp3_sl1.5",    "tp_atr": 3.0, "sl_atr": 1.5, "prob_exit": None, "trail_atr": None, "time_bars": None},
    {"name": "tp4_sl1.5",    "tp_atr": 4.0, "sl_atr": 1.5, "prob_exit": None, "trail_atr": None, "time_bars": None},
    {"name": "prob_exit_55", "tp_atr": 4.0, "sl_atr": 1.0, "prob_exit": 0.55, "trail_atr": None, "time_bars": None},
    {"name": "prob_exit_60", "tp_atr": 4.0, "sl_atr": 1.0, "prob_exit": 0.60, "trail_atr": None, "time_bars": None},
    {"name": "prob_exit_65", "tp_atr": 4.0, "sl_atr": 1.0, "prob_exit": 0.65, "trail_atr": None, "time_bars": None},
    {"name": "prob_exit_70", "tp_atr": 4.0, "sl_atr": 1.0, "prob_exit": 0.70, "trail_atr": None, "time_bars": None},
    {"name": "trail_1atr",   "tp_atr": None, "sl_atr": 1.0, "prob_exit": None, "trail_atr": 1.0,  "time_bars": None},
    {"name": "trail_1.5atr", "tp_atr": None, "sl_atr": 1.0, "prob_exit": None, "trail_atr": 1.5,  "time_bars": None},
    {"name": "time_2d",      "tp_atr": 3.0, "sl_atr": 1.0, "prob_exit": None, "trail_atr": None, "time_bars": 26},
    {"name": "time_5d",      "tp_atr": 3.0, "sl_atr": 1.0, "prob_exit": None, "trail_atr": None, "time_bars": 65},
    {"name": "time_10d",     "tp_atr": 3.0, "sl_atr": 1.0, "prob_exit": None, "trail_atr": None, "time_bars": 130},
]

MAX_HOLDING_BARS = 195
COMMISSION_PCT = 0.001


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
    path = RAW_30M_DIR / f"{ticker}.parquet"
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    prev_c = np.roll(c, 1)
    prev_c[0] = np.nan
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(period, min_periods=period).mean().values
    return atr


def select_top_tickers(proba: pd.DataFrame, universe_csv: Path, top_n: int) -> list[str]:
    tickers_in_proba = proba["ticker"].unique()
    selected = [t for t in tickers_in_proba if (RAW_30M_DIR / f"{t}.parquet").exists()]
    if top_n >= len(selected):
        return selected

    dvol = {}
    for t in selected:
        raw = pd.read_parquet(RAW_30M_DIR / f"{t}.parquet")
        raw.columns = [c.lower() for c in raw.columns]
        dvol[t] = (raw["close"].values * raw["volume"].values).mean()
    ranked = sorted(dvol, key=dvol.get, reverse=True)
    return ranked[:top_n]


# ---------------------------------------------------------------------------
# Precomputed per-ticker arrays (built once, reused across all combos)
# ---------------------------------------------------------------------------

@dataclass
class TickerData:
    ticker: str
    # Raw 30m arrays (aligned by integer index)
    timestamps: np.ndarray   # datetime64
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    n_bars: int
    # Signal arrays (aligned to raw bar indices; NaN where no proba)
    p_long_dir: np.ndarray   # length = n_bars, NaN for non-signal bars
    p_short_dir: np.ndarray


def build_ticker_data(ticker: str, raw_30m: pd.DataFrame, proba_ticker: pd.DataFrame) -> TickerData:
    n = len(raw_30m)
    ts_arr = raw_30m["timestamp"].values
    o = raw_30m["open"].values.astype(np.float64)
    h = raw_30m["high"].values.astype(np.float64)
    l = raw_30m["low"].values.astype(np.float64)
    c = raw_30m["close"].values.astype(np.float64)
    atr = compute_atr(raw_30m)

    p_long = np.full(n, np.nan)
    p_short = np.full(n, np.nan)

    ts_to_idx = pd.Series(np.arange(n), index=ts_arr)
    proba_ts = proba_ticker["timestamp"].values
    mask = np.isin(proba_ts, ts_arr)
    matched_proba = proba_ticker.iloc[mask.nonzero()[0]]
    if len(matched_proba) > 0:
        idxs = ts_to_idx.loc[matched_proba["timestamp"].values].values
        p_long[idxs] = matched_proba["p_long_dir"].values
        p_short[idxs] = matched_proba["p_short_dir"].values

    return TickerData(
        ticker=ticker, timestamps=ts_arr,
        open_=o, high=h, low=l, close=c, atr=atr,
        n_bars=n, p_long_dir=p_long, p_short_dir=p_short,
    )


# ---------------------------------------------------------------------------
# Fast numpy-based simulation (no iterrows)
# ---------------------------------------------------------------------------

def simulate_ticker_fast(
    td: TickerData,
    entry_threshold: float,
    exit_cfg: dict,
) -> list[tuple]:
    """Returns list of (ticker, dir, entry_idx, exit_idx, entry_price, exit_price, pnl_pct, exit_reason, bars_held)."""
    results = []
    tp_atr_mult = exit_cfg["tp_atr"]
    sl_atr_mult = exit_cfg["sl_atr"]
    prob_exit_thresh = exit_cfg["prob_exit"]
    trail_atr_mult = exit_cfg["trail_atr"]
    time_limit = exit_cfg["time_bars"] or MAX_HOLDING_BARS

    cooldown_idx = -1
    n = td.n_bars

    for i in range(n - 1):
        if i <= cooldown_idx:
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

        atr_val = td.atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        entry_idx = i + 1
        entry_price = td.open_[entry_idx]
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        # Exit levels
        sl_price = entry_price - direction * sl_atr_mult * atr_val
        tp_price = entry_price + direction * tp_atr_mult * atr_val if tp_atr_mult is not None else None

        # Trailing stop state
        best_price = entry_price
        trail_activation = entry_price + direction * atr_val if trail_atr_mult is not None else None
        trail_active = False
        trail_stop = 0.0

        end_idx = min(entry_idx + time_limit + 1, n)
        exit_idx = end_idx - 1
        exit_price = td.close[exit_idx]
        exit_reason = "time"

        for j in range(entry_idx + 1, end_idx):
            bar_h = td.high[j]
            bar_l = td.low[j]

            # Stop loss
            if direction == 1:
                if bar_l <= sl_price:
                    exit_idx, exit_price, exit_reason = j, sl_price, "sl"
                    break
            else:
                if bar_h >= sl_price:
                    exit_idx, exit_price, exit_reason = j, sl_price, "sl"
                    break

            # Take profit
            if tp_price is not None:
                if direction == 1 and bar_h >= tp_price:
                    exit_idx, exit_price, exit_reason = j, tp_price, "tp"
                    break
                elif direction == -1 and bar_l <= tp_price:
                    exit_idx, exit_price, exit_reason = j, tp_price, "tp"
                    break

            # Trailing stop
            if trail_atr_mult is not None:
                if direction == 1:
                    if bar_h > best_price:
                        best_price = bar_h
                    if best_price >= trail_activation:
                        trail_active = True
                        trail_stop = best_price - trail_atr_mult * atr_val
                    if trail_active and bar_l <= trail_stop:
                        exit_idx, exit_price, exit_reason = j, trail_stop, "trail"
                        break
                else:
                    if bar_l < best_price:
                        best_price = bar_l
                    if best_price <= trail_activation:
                        trail_active = True
                        trail_stop = best_price + trail_atr_mult * atr_val
                    if trail_active and bar_h >= trail_stop:
                        exit_idx, exit_price, exit_reason = j, trail_stop, "trail"
                        break

            # Probability exit
            if prob_exit_thresh is not None:
                opp = td.p_short_dir[j] if direction == 1 else td.p_long_dir[j]
                if not np.isnan(opp) and opp >= prob_exit_thresh:
                    exit_idx, exit_price, exit_reason = j, td.close[j], "prob_exit"
                    break

        bars_held = exit_idx - entry_idx
        raw_ret = direction * (exit_price - entry_price) / entry_price
        net_ret = raw_ret - COMMISSION_PCT

        results.append((
            td.ticker, direction, entry_idx, exit_idx,
            entry_price, exit_price, net_ret, exit_reason, bars_held,
        ))
        cooldown_idx = exit_idx

    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty or len(trades_df) == 0:
        return {
            "n_trades": 0, "win_rate": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "profit_factor": 0.0, "avg_pnl_pct": 0.0, "total_pnl_pct": 0.0,
            "sharpe": 0.0, "max_dd_pct": 0.0, "avg_holding_bars": 0.0,
            "exit_tp_pct": 0.0, "exit_sl_pct": 0.0, "exit_time_pct": 0.0,
            "exit_trail_pct": 0.0, "exit_prob_pct": 0.0,
            "long_n": 0, "short_n": 0, "long_wr": 0.0, "short_wr": 0.0,
        }

    n = len(trades_df)
    rets = trades_df["pnl_pct"].values
    wins = rets > 0
    losses = ~wins

    gp = rets[wins].sum() if wins.any() else 0.0
    gl = abs(rets[losses].sum()) if losses.any() else 0.0
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)

    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0

    cum = np.cumsum(rets)
    peak = np.maximum.accumulate(cum)
    max_dd = float((cum - peak).min()) if len(cum) > 0 else 0.0

    dirs = trades_df["direction"].values
    longs_mask = dirs == 1
    shorts_mask = dirs == -1

    er = trades_df["exit_reason"].values

    return {
        "n_trades": n,
        "win_rate": float(wins.mean()),
        "avg_win_pct": float(rets[wins].mean()) if wins.any() else 0.0,
        "avg_loss_pct": float(rets[losses].mean()) if losses.any() else 0.0,
        "profit_factor": round(pf, 3),
        "avg_pnl_pct": float(rets.mean()),
        "total_pnl_pct": float(rets.sum()),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "avg_holding_bars": float(trades_df["holding_bars"].mean()),
        "exit_tp_pct": float((er == "tp").mean()),
        "exit_sl_pct": float((er == "sl").mean()),
        "exit_time_pct": float((er == "time").mean()),
        "exit_trail_pct": float((er == "trail").mean()),
        "exit_prob_pct": float((er == "prob_exit").mean()),
        "long_n": int(longs_mask.sum()),
        "short_n": int(shorts_mask.sum()),
        "long_wr": float((rets[longs_mask] > 0).mean()) if longs_mask.any() else 0.0,
        "short_wr": float((rets[shorts_mask] > 0).mean()) if shorts_mask.any() else 0.0,
    }


def compute_grouped_metrics(trades_df: pd.DataFrame, universe_csv: Path) -> dict:
    uni = pd.read_csv(universe_csv)
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
    universe_csv: Path = UNIVERSE_CSV,
) -> pd.DataFrame:
    SWEEP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading probabilities (split=%s)...", split)
    proba = load_proba(split)

    logger.info("Selecting top %d tickers by dollar volume...", top_n)
    tickers = select_top_tickers(proba, universe_csv, top_n)
    logger.info("Selected %d tickers", len(tickers))

    # Pre-build all TickerData objects (one-time cost)
    logger.info("Pre-building ticker data arrays...")
    ticker_data: dict[str, TickerData] = {}
    for t in tickers:
        try:
            raw = load_raw_30m(t)
            proba_t = proba[proba["ticker"] == t]
            ticker_data[t] = build_ticker_data(t, raw, proba_t)
        except Exception as e:
            logger.warning("Skipping %s: %s", t, e)
    logger.info("Built data for %d tickers", len(ticker_data))

    total_combos = len(ENTRY_THRESHOLDS) * len(EXIT_STRATEGIES)
    logger.info("Running %d combos × %d tickers", total_combos, len(ticker_data))

    results = []
    combo_idx = 0

    for entry_thresh in ENTRY_THRESHOLDS:
        for exit_cfg in EXIT_STRATEGIES:
            combo_idx += 1
            combo_name = f"entry_{entry_thresh}_{exit_cfg['name']}"
            t0 = time.time()

            all_rows = []
            for td in ticker_data.values():
                trades = simulate_ticker_fast(td, entry_thresh, exit_cfg)
                all_rows.extend(trades)

            if all_rows:
                trades_df = pd.DataFrame(all_rows, columns=[
                    "ticker", "direction", "entry_idx", "exit_idx",
                    "entry_price", "exit_price", "pnl_pct", "exit_reason", "holding_bars",
                ])
            else:
                trades_df = pd.DataFrame()

            metrics = compute_metrics(trades_df)
            metrics["combo_name"] = combo_name
            metrics["entry_threshold"] = entry_thresh
            metrics["exit_strategy"] = exit_cfg["name"]
            metrics["tp_atr"] = exit_cfg["tp_atr"]
            metrics["sl_atr"] = exit_cfg["sl_atr"]
            metrics["prob_exit"] = exit_cfg["prob_exit"]
            metrics["trail_atr"] = exit_cfg["trail_atr"]
            metrics["time_bars"] = exit_cfg["time_bars"]
            results.append(metrics)

            elapsed = time.time() - t0
            logger.info(
                "(%d/%d) %s — %d trades, WR=%.1f%%, PF=%.2f, Sharpe=%.2f  [%.1fs]",
                combo_idx, total_combos, combo_name,
                metrics["n_trades"], metrics["win_rate"] * 100,
                metrics["profit_factor"], metrics["sharpe"], elapsed,
            )

    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    results_df.to_csv(SWEEP_RESULTS_DIR / "sweep_summary.csv", index=False)
    results_df.to_parquet(SWEEP_RESULTS_DIR / "sweep_summary.parquet", index=False)

    # Print top 10
    logger.info("\n=== TOP 10 COMBOS BY SHARPE ===")
    for _, row in results_df.head(10).iterrows():
        logger.info(
            "  %s — trades=%d  WR=%.1f%%  PF=%.2f  Sharpe=%.2f  avgPnL=%.3f%%  maxDD=%.1f%%  avgHold=%.0f bars",
            row["combo_name"], row["n_trades"], row["win_rate"] * 100,
            row["profit_factor"], row["sharpe"], row["avg_pnl_pct"] * 100,
            row["max_dd_pct"], row["avg_holding_bars"],
        )

    # Detailed grouped analysis on best combo
    if not results_df.empty:
        best = results_df.iloc[0]
        best_exit_cfg = next(e for e in EXIT_STRATEGIES if e["name"] == best["exit_strategy"])

        logger.info("\n=== DETAILED ANALYSIS: %s ===", best["combo_name"])

        all_rows = []
        for td in ticker_data.values():
            trades = simulate_ticker_fast(td, best["entry_threshold"], best_exit_cfg)
            all_rows.extend(trades)

        if all_rows:
            best_trades_df = pd.DataFrame(all_rows, columns=[
                "ticker", "direction", "entry_idx", "exit_idx",
                "entry_price", "exit_price", "pnl_pct", "exit_reason", "holding_bars",
            ])
            best_trades_df.to_parquet(SWEEP_RESULTS_DIR / "best_combo_trades.parquet", index=False)

            grouped = compute_grouped_metrics(best_trades_df, universe_csv)
            with open(SWEEP_RESULTS_DIR / "best_combo_grouped.json", "w") as f:
                json.dump(grouped, f, indent=2, default=str)

            for gn, gm in sorted(grouped.items()):
                if gm["n_trades"] >= 10:
                    logger.info(
                        "  %-35s  n=%4d  WR=%.1f%%  PF=%.2f  Sharpe=%+.2f  avgPnL=%+.3f%%",
                        gn, gm["n_trades"], gm["win_rate"] * 100,
                        gm["profit_factor"], gm["sharpe"], gm["avg_pnl_pct"] * 100,
                    )

            # Per-ticker breakdown
            ticker_metrics = []
            for ticker, grp in best_trades_df.groupby("ticker"):
                m = compute_metrics(grp)
                m["ticker"] = ticker
                ticker_metrics.append(m)
            ticker_df = pd.DataFrame(ticker_metrics).sort_values("sharpe", ascending=False)
            ticker_df.to_csv(SWEEP_RESULTS_DIR / "best_combo_per_ticker.csv", index=False)

            logger.info("\n=== TOP 15 TICKERS (best combo) ===")
            for _, row in ticker_df.head(15).iterrows():
                logger.info(
                    "  %-6s  n=%3d  WR=%.1f%%  PF=%.2f  Sharpe=%+.2f  totalPnL=%+.2f%%",
                    row["ticker"], row["n_trades"], row["win_rate"] * 100,
                    row["profit_factor"], row["sharpe"], row["total_pnl_pct"] * 100,
                )

            logger.info("\n=== BOTTOM 10 TICKERS (best combo) ===")
            for _, row in ticker_df.tail(10).iterrows():
                logger.info(
                    "  %-6s  n=%3d  WR=%.1f%%  PF=%.2f  Sharpe=%+.2f  totalPnL=%+.2f%%",
                    row["ticker"], row["n_trades"], row["win_rate"] * 100,
                    row["profit_factor"], row["sharpe"], row["total_pnl_pct"] * 100,
                )

    logger.info("\nSweep complete. Results → %s", SWEEP_RESULTS_DIR)
    return results_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = p.parse_args()
    run_sweep(top_n=args.top_n, split=args.split)
