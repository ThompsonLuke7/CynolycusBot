"""
Sweep v5 — SPY regime filter + trail parameter search on Tier 1 & 2 only.

Tier 3 (negative Sharpe) is excluded from live trading and from this sweep.

Builds on the best sweep_v4 baseline configs per tier:
  Tier 1 baseline:  entry_threshold=0.60, sl_atr=4.0, np_n_bars=None  (Sharpe 3.18)
  Tier 2 baseline:  entry_threshold=0.70, sl_atr=4.0, np_n_bars=78, np_mfe_atr=0.25 (Sharpe 1.01)

New dimensions tested on top of each baseline:

  1. SPY regime filter (countertrend veto)
     Only take LONG  entries when SPY p_long_dir  >= spy_min  at signal time.
     Only take SHORT entries when SPY p_short_dir >= spy_min  at signal time.
     Tests: off (0.0), 0.50, 0.55, 0.60

  2. Trail parameter grid
     arm_pct     ∈ {0.015, 0.025, 0.040}   (was fixed at 0.025)
     giveback_pct ∈ {0.20,  0.25,  0.35 }   (was fixed at 0.25 )

Grid: 4 spy_filter × 3 arm × 3 giveback = 36 combos per tier × 2 tiers = 72 total.
Run on val split to avoid burning the test set during parameter search.

Run:
  python multi_ticker_swing/backtest/sweep_v5.py [--split val] [--out-dir ...]
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.multi_ticker_swing.config.pipeline_config import (
    BACKTEST_RESULTS_DIR,
    RAW_30M_DIR,
    RAW_5M_DIR,
    TRADING_BLACKLIST,
)
from strategies.multi_ticker_swing.backtest.sweep_v4 import (
    TickerData,
    compute_metrics,
    find_5m_confirmation,
    load_proba,
    load_raw_30m,
    load_raw_5m,
    TRADE_COLS,
    BARS_5M_DAY,
    MAX_HOLD_5M,
    COMMISSION_PCT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
logger = logging.getLogger(__name__)

V3_PER_TICKER = BACKTEST_RESULTS_DIR / "sweep_v3" / "best_v3_per_ticker.csv"
OUT_DIR       = BACKTEST_RESULTS_DIR / "sweep_v5"

CONFIRM_MAX_5M = 6  # fixed (sweep_v4 best)

# ---------------------------------------------------------------------------
# Tier baselines (best sweep_v4 config per tier)
# ---------------------------------------------------------------------------
TIER_BASELINES = {
    "tier1": {
        "entry_threshold": 0.60,
        "sl_atr":          4.0,
        "np_n_bars":       None,
        "np_mfe_atr":      None,
    },
    "tier2": {
        "entry_threshold": 0.70,
        "sl_atr":          4.0,
        "np_n_bars":       78,
        "np_mfe_atr":      0.25,
    },
}

# ---------------------------------------------------------------------------
# New sweep dimensions
# ---------------------------------------------------------------------------
SPY_FILTER_THRESHOLDS = [0.0, 0.50, 0.55, 0.60]  # 0.0 = off (no filter)
ARM_PCTS              = [0.015, 0.025, 0.040]
GIVEBACK_PCTS         = [0.20,  0.25,  0.35]


def build_grid(tier_name: str) -> list[dict]:
    base = TIER_BASELINES[tier_name]
    combos = []
    for spy_min in SPY_FILTER_THRESHOLDS:
        for arm in ARM_PCTS:
            for gb in GIVEBACK_PCTS:
                spy_tag = f"spy{int(spy_min*100)}" if spy_min > 0 else "spyoff"
                name = f"{spy_tag}_arm{int(arm*1000)}_gb{int(gb*100)}"
                combos.append({
                    "name":           name,
                    "spy_min":        spy_min,
                    "arm_pct":        arm,
                    "giveback_pct":   gb,
                    **base,
                })
    return combos


# ---------------------------------------------------------------------------
# Simulation with SPY regime filter
# ---------------------------------------------------------------------------

def simulate_ticker_v5(
    td: TickerData,
    entry_threshold: float,
    confirm_max_5m: int,
    exit_cfg: dict,
    spy_data: "SpyData | None",
) -> list[tuple]:
    """Like sweep_v4's simulate_ticker_5m but with optional SPY regime filter."""
    if not td.has_5m:
        return []   # skip tickers without 5m data (fallback not used in v5)

    sl_atr_mult  = float(exit_cfg.get("sl_atr", 0.0))
    arm_pct      = exit_cfg["arm_pct"]
    giveback_pct = exit_cfg["giveback_pct"]
    np_n_bars    = exit_cfg.get("np_n_bars")
    np_mfe_atr   = exit_cfg.get("np_mfe_atr")
    spy_min      = float(exit_cfg.get("spy_min", 0.0))

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

        # ── SPY regime filter ───────────────────────────────────────────────
        if spy_min > 0.0 and spy_data is not None:
            spy_ts = td.ts_30m[i]
            if direction == 1:
                spy_p = spy_data.p_long_dir_at(spy_ts)
                if np.isnan(spy_p) or spy_p < spy_min:
                    continue
            else:
                spy_p = spy_data.p_short_dir_at(spy_ts)
                if np.isnan(spy_p) or spy_p < spy_min:
                    continue

        atr_val = td.atr_30m[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        conf_price, conf_5m_idx = find_5m_confirmation(td, i, direction, confirm_max_5m)
        if conf_price is None:
            continue

        entry_5m_idx = conf_5m_idx + 1
        if entry_5m_idx >= td.n_5m:
            continue

        if td.is_first_30min_5m[entry_5m_idx]:
            continue

        entry_price = float(td.open_5m[entry_5m_idx])
        if np.isnan(entry_price) or entry_price <= 0:
            entry_price = conf_price

        sl_price    = (entry_price - direction * sl_atr_mult * atr_val) if sl_atr_mult > 0 else None
        best_price  = entry_price
        trail_armed = False
        exit_5m_idx = min(entry_5m_idx + MAX_HOLD_5M, td.n_5m - 1)
        exit_price  = float(td.close_5m[exit_5m_idx])
        exit_reason = "time"

        for j in range(entry_5m_idx, min(entry_5m_idx + MAX_HOLD_5M + 1, td.n_5m)):
            bar_h = float(td.high_5m[j])
            bar_l = float(td.low_5m[j])
            bar_c = float(td.close_5m[j])

            if sl_price is not None:
                if direction == 1 and bar_l <= sl_price:
                    exit_5m_idx, exit_price, exit_reason = j, sl_price, "sl"; break
                if direction == -1 and bar_h >= sl_price:
                    exit_5m_idx, exit_price, exit_reason = j, sl_price, "sl"; break

            best_price = max(best_price, bar_h) if direction == 1 else min(best_price, bar_l)

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
                    exit_reason = "trail"; break

            if np_n_bars is not None and (j - entry_5m_idx) == np_n_bars and not trail_armed:
                mfe_atr_units = direction * (best_price - entry_price) / atr_val
                if mfe_atr_units < np_mfe_atr:
                    exit_5m_idx, exit_price, exit_reason = j, bar_c, "no_progress"; break
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


# ---------------------------------------------------------------------------
# SPY probability lookup helper
# ---------------------------------------------------------------------------

class SpyData:
    """Fast timestamp → SPY p_long_dir/p_short_dir lookup."""

    def __init__(self, proba_df: pd.DataFrame) -> None:
        spy = proba_df[proba_df["ticker"] == "SPY"].copy()
        spy = spy.sort_values("timestamp")
        self._ts     = spy["timestamp"].values
        self._p_long = spy["p_long_dir"].values if "p_long_dir" in spy.columns else np.full(len(spy), np.nan)
        self._p_short = spy["p_short_dir"].values if "p_short_dir" in spy.columns else np.full(len(spy), np.nan)

    def p_long_dir_at(self, ts: np.datetime64) -> float:
        idx = int(np.searchsorted(self._ts, ts, side="left"))
        if idx < len(self._ts) and self._ts[idx] == ts:
            return float(self._p_long[idx])
        return float("nan")

    def p_short_dir_at(self, ts: np.datetime64) -> float:
        idx = int(np.searchsorted(self._ts, ts, side="left"))
        if idx < len(self._ts) and self._ts[idx] == ts:
            return float(self._p_short[idx])
        return float("nan")


# ---------------------------------------------------------------------------
# Per-tier sweep
# ---------------------------------------------------------------------------

def run_tier_sweep_v5(
    tier_name: str,
    tickers: list[str],
    ticker_data: dict,
    spy_data: SpyData,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = build_grid(tier_name)
    n_tickers = len([t for t in tickers if t in ticker_data])
    n_combos  = len(grid)
    entry_threshold = TIER_BASELINES[tier_name]["entry_threshold"]
    logger.info("=== %s: %d tickers, %d combos ===", tier_name, n_tickers, n_combos)

    all_rows, best_trades, best_sharpe = [], [], -999.0

    for k, combo in enumerate(grid, 1):
        t0 = time.time()
        combo_name = f"{tier_name}_{combo['name']}"
        all_trades = []
        for t in tickers:
            if t not in ticker_data:
                continue
            trades = simulate_ticker_v5(
                ticker_data[t], entry_threshold, CONFIRM_MAX_5M, combo, spy_data,
            )
            all_trades.extend(trades)

        tdf = pd.DataFrame(all_trades, columns=TRADE_COLS) if all_trades else pd.DataFrame(columns=TRADE_COLS)
        m   = compute_metrics(tdf)
        elapsed = time.time() - t0

        row = {
            "combo_name":      combo_name,
            "tier":            tier_name,
            "entry_threshold": entry_threshold,
            "spy_min":         combo["spy_min"],
            "arm_pct":         combo["arm_pct"],
            "giveback_pct":    combo["giveback_pct"],
            "sl_atr":          combo["sl_atr"],
            "np_n_bars":       combo.get("np_n_bars"),
            "np_mfe_atr":      combo.get("np_mfe_atr"),
            **m,
        }
        all_rows.append(row)

        if m["sharpe"] > best_sharpe and m["n_trades"] >= 30:
            best_sharpe = m["sharpe"]
            best_trades = all_trades

        logger.info(
            "[%s] (%2d/%d) %-50s %4d trades  WR=%.1f%%  PF=%.2f  Sharpe=%.3f  [%.1fs]",
            tier_name, k, n_combos, combo_name,
            m["n_trades"], m["win_rate"] * 100, m["profit_factor"], m["sharpe"], elapsed,
        )

    summary_df = pd.DataFrame(all_rows).sort_values("sharpe", ascending=False)
    summary_df.to_csv(out_dir / f"summary_{tier_name}.csv", index=False)
    summary_df.to_parquet(out_dir / f"summary_{tier_name}.parquet", index=False)
    if best_trades:
        pd.DataFrame(best_trades, columns=TRADE_COLS).to_parquet(
            out_dir / f"best_trades_{tier_name}.parquet", index=False)

    logger.info("[%s] Best: %s  Sharpe=%.3f", tier_name,
                summary_df.iloc[0]["combo_name"] if not summary_df.empty else "none", best_sharpe)
    return summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sweep(split: str = "val") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not V3_PER_TICKER.exists():
        logger.error("sweep_v3 per-ticker file not found: %s. Run sweep_v3.py first.", V3_PER_TICKER)
        return

    v3_pt = pd.read_csv(V3_PER_TICKER)
    tier1_tickers = [t for t in v3_pt[v3_pt["sharpe"] >= 2.0]["ticker"].tolist()
                     if t not in TRADING_BLACKLIST]
    tier2_tickers = [t for t in v3_pt[(v3_pt["sharpe"] >= 0.0) & (v3_pt["sharpe"] < 2.0)]["ticker"].tolist()
                     if t not in TRADING_BLACKLIST]
    logger.info("Tiers (T3 excluded): T1=%d  T2=%d", len(tier1_tickers), len(tier2_tickers))

    logger.info("Loading probabilities (split=%s)...", split)
    proba = load_proba(split)
    spy_data = SpyData(proba)
    logger.info("SPY regime data: %d timestamps available.", len(spy_data._ts))

    all_tickers = list(set(tier1_tickers + tier2_tickers))
    logger.info("Pre-building ticker data for %d tickers...", len(all_tickers))
    ticker_data: dict = {}
    for t in all_tickers:
        try:
            r30 = load_raw_30m(t)
            r5  = load_raw_5m(t)
            pt  = proba[proba["ticker"] == t]
            ticker_data[t] = TickerData(t, r30, r5, pt)
        except Exception as exc:
            logger.warning("Failed to build data for %s: %s", t, exc)

    logger.info("Ticker data built for %d / %d tickers.", len(ticker_data), len(all_tickers))

    summaries = []
    for tier_name, tickers in [("tier1", tier1_tickers), ("tier2", tier2_tickers)]:
        s = run_tier_sweep_v5(
            tier_name, tickers, ticker_data, spy_data,
            OUT_DIR / tier_name,
        )
        summaries.append(s)

    combined = pd.concat(summaries, ignore_index=True).sort_values("sharpe", ascending=False)
    combined.to_csv(OUT_DIR / "sweep_v5_combined.csv", index=False)
    logger.info("Saved combined results → %s", OUT_DIR / "sweep_v5_combined.csv")

    # Print top-5 per tier
    for tier_name in ("tier1", "tier2"):
        top = combined[combined["tier"] == tier_name].head(5)
        logger.info("\n=== Top-5 %s ===\n%s", tier_name,
                    top[["combo_name","spy_min","arm_pct","giveback_pct","n_trades",
                          "win_rate","sharpe","total_pnl_pct"]].to_string(index=False))


def main() -> None:
    global OUT_DIR
    p = argparse.ArgumentParser(description="Sweep v5: SPY regime filter + trail params")
    p.add_argument("--split",   default="val", choices=["val", "test"])
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()
    OUT_DIR = Path(args.out_dir)
    run_sweep(split=args.split)


if __name__ == "__main__":
    main()
