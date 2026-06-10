"""
Compare two exit strategies for 30-minute swing labels:
  A) Opposite pivot exit  (current sweep)
  B) ATR trail exit       (mirrors order_policy.py logic, scaled to 30m)

Policy B parameters (scaled from 10m order_policy.py defaults to 30m):
  trail_activate_atr      = 0.75   (arm the trail once price moves 0.75 ATR in our favour)
  trail_atr               = 0.80   (trail distance from peak, pre-TP)
  trail_atr_after_tp      = 0.50   (tighten after TP level hit)
  tp_atr                  = 1.00   (TP level that triggers tighter trail)
  hard_stop_atr           = 2.00   (hard SL — was 0/disabled on 10m, enabled here for 30m)
  stale_no_progress_bars  = 2      (scaled from 20m → 60m = 2×30m bars, no move ≥ 0.35 ATR)
  stale_no_progress_atr   = 0.35
  stale_after_fav_bars    = 3      (scaled from 30m → 90m = 3×30m bars after last peak)
  stale_retrace_atr       = 0.25
  max_hold_bars           = 48     (cap at 24h = 48×30m; safety net)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.spy_intraday.Features.feature_sets.custom_indicators import add_fractal_pivots
from strategies.multi_ticker_swing.config.pipeline_config import MODULE_ROOT, UNIVERSE_CSV, RAW_30M_DIR
from strategies.multi_ticker_swing.data.fetch_data import load_universe
from strategies.multi_ticker_swing.plots.generate_soft_swing_30m_plots import (
    FIRST_IN_RUN_FILTER, LABEL_SHIFT_BARS,
    _shift_pivot_labels_back, _session_dates,
    keep_first_same_side_event_session_reset,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = MODULE_ROOT / "plots" / "exit_comparison"

# ── ATR trail policy parameters (30m-optimised via parameter sweep) ──────────
# Original 10m values → what changed and why:
#   trail_activate_atr : 0.75 → 2.50  (arm only after a real move; 0.75 was
#                                       arming immediately and trailing from
#                                       near-breakeven on 30m bars)
#   trail_atr          : 0.80 → 2.00  (30m bars oscillate ±1 ATR intrabar;
#                                       0.80 was stopping out on noise)
#   trail_atr_after_tp : 0.50 → 0.75  (tighten moderately once TP hit, not
#                                       too tight — ride the momentum further)
#   tp_atr             : 1.00 → 3.00  (matches optimal TP from ATR sweep)
#   hard_stop_atr      : 0.00 → 2.50  (add a hard floor — was disabled on 10m)
#   stale_no_prog_bars : 2    → 6     (2 bars = 60 min was too short;
#                                       6 bars = 3h gives swing time to develop)
#   stale_fav_bars     : 3    → 6     (3 bars = 90 min → 6 bars = 3h)
TRAIL_ACTIVATE_ATR   = 2.50
TRAIL_ATR            = 2.00
TRAIL_ATR_AFTER_TP   = 0.75
TP_ATR               = 3.00
HARD_STOP_ATR        = 2.50
STALE_NO_PROG_BARS   = 6        # 6 × 30m = 3h no progress
STALE_NO_PROG_ATR    = 0.35
STALE_FAV_BARS       = 6        # 6 × 30m = 3h since last peak
STALE_RETRACE_ATR    = 0.25
MAX_HOLD_BARS        = 48       # 24-hour safety cap


class TradeResult(NamedTuple):
    ticker:      str
    side:        str
    entry_bar:   int
    entry_price: float
    atr:         float
    # --- strategy A: opposite pivot ---
    exit_opp_bars:   int
    exit_opp_pnl:    float          # signed ATR P&L
    # --- strategy B: ATR trail ---
    exit_trail_bars: int
    exit_trail_pnl:  float
    exit_trail_reason: str


def _atr_ewm(df: pd.DataFrame, length: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=length, adjust=False).mean()


def _simulate_trail_exit(
    side: str,
    actual_entry: int,      # bar index of fill (next bar open)
    entry_price: float,
    atr: float,
    opp_exit_bar: int,      # bar index where opposite pivot fires (upper bound)
    highs: np.ndarray,
    lows:  np.ndarray,
    closes: np.ndarray,
    opens:  np.ndarray,
    n: int,
) -> tuple[int, float, str]:
    """
    Bar-by-bar ATR trail simulation.
    Returns (exit_bar_index, exit_price, reason).
    """
    peak = entry_price          # best price in our favour seen so far
    tp_seen = False
    bars_no_progress = 0
    bars_since_peak = 0
    last_peak_bar = actual_entry
    cap_bar = min(actual_entry + MAX_HOLD_BARS, n - 1)

    for j in range(actual_entry, cap_bar + 1):
        h, l, c = highs[j], lows[j], closes[j]

        # ── update peak ──────────────────────────────────────────────────────
        if side == "long":
            if h > peak:
                peak = h
                bars_since_peak = 0
                last_peak_bar = j
            else:
                bars_since_peak += 1
            favorable_move = peak - entry_price
            # TP level check
            if not tp_seen and h >= entry_price + TP_ATR * atr:
                tp_seen = True
        else:
            if l < peak:
                peak = l
                bars_since_peak = 0
                last_peak_bar = j
            else:
                bars_since_peak += 1
            favorable_move = entry_price - peak
            if not tp_seen and l <= entry_price - TP_ATR * atr:
                tp_seen = True

        # ── hard stop ────────────────────────────────────────────────────────
        if HARD_STOP_ATR > 0:
            if side == "long" and l <= entry_price - HARD_STOP_ATR * atr:
                return j, entry_price - HARD_STOP_ATR * atr, "hard_stop"
            if side == "short" and h >= entry_price + HARD_STOP_ATR * atr:
                return j, entry_price + HARD_STOP_ATR * atr, "hard_stop"

        # ── ATR trail stop ───────────────────────────────────────────────────
        trail = TRAIL_ATR_AFTER_TP if tp_seen else TRAIL_ATR
        trail_active = (favorable_move >= TRAIL_ACTIVATE_ATR * atr) or tp_seen

        if trail_active and trail > 0:
            if side == "long":
                trail_level = peak - trail * atr
                if l <= trail_level or c <= trail_level:
                    return j, trail_level, "atr_trail"
            else:
                trail_level = peak + trail * atr
                if h >= trail_level or c >= trail_level:
                    return j, trail_level, "atr_trail"

        # ── stale: no progress ───────────────────────────────────────────────
        bars_held = j - actual_entry
        if (bars_held >= STALE_NO_PROG_BARS
                and favorable_move < STALE_NO_PROG_ATR * atr):
            return j, c, "stale_no_progress"

        # ── stale: retrace after favourable move ─────────────────────────────
        if bars_since_peak >= STALE_FAV_BARS:
            if side == "long":
                retrace = peak - c
            else:
                retrace = c - peak
            if retrace >= STALE_RETRACE_ATR * atr:
                return j, c, "stale_retrace"

        # ── opposite pivot fires (max hold ceiling) ───────────────────────────
        if j >= opp_exit_bar:
            fill = opens[min(j + 1, n - 1)] if j + 1 < n else c
            return j, fill, "opposite_pivot"

    # ── safety cap ───────────────────────────────────────────────────────────
    return cap_bar, closes[cap_bar], "max_hold"


def extract_trades_with_both_exits(
    ticker: str, df: pd.DataFrame
) -> list[TradeResult] | None:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df["atr_14"] = _atr_ewm(df)
    df = df.dropna(subset=["atr_14"])

    try:
        df = add_fractal_pivots(df)
    except Exception as exc:
        logger.warning("[%s] fractal pivots failed: %s", ticker, exc)
        return None

    df["pivot_up"]   = df.get("pivot_up",   pd.Series(0, index=df.index)).fillna(0)
    df["pivot_down"] = df.get("pivot_down", pd.Series(0, index=df.index)).fillna(0)

    long_shifted  = _shift_pivot_labels_back(df["pivot_down"], df.index, LABEL_SHIFT_BARS).to_numpy()
    short_shifted = _shift_pivot_labels_back(df["pivot_up"],   df.index, LABEL_SHIFT_BARS).to_numpy()

    lows_arr  = df["low"].to_numpy(dtype=float)
    highs_arr = df["high"].to_numpy(dtype=float)

    if FIRST_IN_RUN_FILTER:
        long_core, short_core, _, _ = keep_first_same_side_event_session_reset(
            long_shifted, short_shifted, df.index,
            lows=lows_arr, highs=highs_arr,
        )
    else:
        long_core, short_core = long_shifted.copy(), short_shifted.copy()

    closes = df["close"].to_numpy(dtype=float)
    opens  = df["open"].to_numpy(dtype=float)
    atrs   = df["atr_14"].to_numpy(dtype=float)
    n      = len(df)

    long_idx  = np.flatnonzero(long_core  == 1)
    short_idx = np.flatnonzero(short_core == 1)

    results: list[TradeResult] = []

    def _process(entry_label_bar: int, side: str) -> None:
        actual_entry = entry_label_bar + 1
        if actual_entry >= n:
            return
        entry_price = opens[actual_entry]
        atr = atrs[entry_label_bar]
        if atr <= 0 or not np.isfinite(atr):
            return

        # Opposite pivot = exit for strategy A
        if side == "long":
            opp_arr = short_idx[short_idx > entry_label_bar]
        else:
            opp_arr = long_idx[long_idx > entry_label_bar]
        if len(opp_arr) == 0:
            return

        opp_label_bar = int(opp_arr[0])
        opp_exit_bar  = min(opp_label_bar + 1, n - 1)
        opp_exit_price = opens[opp_exit_bar]

        if side == "long":
            opp_pnl = (opp_exit_price - entry_price) / atr
        else:
            opp_pnl = (entry_price - opp_exit_price) / atr
        opp_bars = opp_exit_bar - actual_entry

        # Strategy B: ATR trail
        tb, tp, reason = _simulate_trail_exit(
            side, actual_entry, entry_price, atr,
            opp_exit_bar,
            highs_arr, lows_arr, closes, opens, n,
        )
        if side == "long":
            trail_pnl = (tp - entry_price) / atr
        else:
            trail_pnl = (entry_price - tp) / atr
        trail_bars = tb - actual_entry

        results.append(TradeResult(
            ticker=ticker, side=side,
            entry_bar=actual_entry, entry_price=entry_price, atr=atr,
            exit_opp_bars=opp_bars, exit_opp_pnl=opp_pnl,
            exit_trail_bars=trail_bars, exit_trail_pnl=trail_pnl,
            exit_trail_reason=reason,
        ))

    for i in long_idx:
        _process(int(i), "long")
    for i in short_idx:
        _process(int(i), "short")

    return results if results else None


def _stats(pnl: np.ndarray, bars: np.ndarray, label: str) -> dict:
    n = len(pnl)
    wins  = (pnl > 0).sum()
    losses = (pnl < 0).sum()
    return {
        "strategy":     label,
        "n":            n,
        "win_rate":     wins / n,
        "avg_pnl_atr":  pnl.mean(),
        "median_pnl_atr": np.median(pnl),
        "avg_win_atr":  pnl[pnl > 0].mean() if wins else 0.0,
        "avg_loss_atr": pnl[pnl < 0].mean() if losses else 0.0,
        "pf":           pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()) if losses else np.inf,
        "med_bars":     np.median(bars),
        "p90_bars":     np.percentile(bars, 90),
    }


def plot_comparison(df: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. P&L distribution
    ax = axes[0]
    ax.hist(df["exit_opp_pnl"].clip(-4, 6),  bins=60, alpha=0.6,
            color="#1565C0", label="Opposite pivot exit")
    ax.hist(df["exit_trail_pnl"].clip(-4, 6), bins=60, alpha=0.6,
            color="#2E7D32", label="ATR trail exit")
    ax.axvline(df["exit_opp_pnl"].mean(),   color="#1565C0", linestyle="--", linewidth=1.5)
    ax.axvline(df["exit_trail_pnl"].mean(), color="#2E7D32", linestyle="--", linewidth=1.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("P&L distribution (ATR units)")
    ax.set_xlabel("P&L (ATR)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Hold time distribution
    ax = axes[1]
    ax.hist(df["exit_opp_bars"].clip(0, 60)  * 0.5, bins=40, alpha=0.6,
            color="#1565C0", label="Opposite pivot")
    ax.hist(df["exit_trail_bars"].clip(0, 60) * 0.5, bins=40, alpha=0.6,
            color="#2E7D32", label="ATR trail")
    ax.set_title("Hold time distribution (hours)")
    ax.set_xlabel("Hours held")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. ATR trail exit reason breakdown
    ax = axes[2]
    reason_counts = df["exit_trail_reason"].value_counts()
    colors = {
        "atr_trail":         "#2E7D32",
        "hard_stop":         "#C62828",
        "stale_no_progress": "#F57F17",
        "stale_retrace":     "#FF8F00",
        "opposite_pivot":    "#1565C0",
        "max_hold":          "#555555",
    }
    bars_rc = ax.bar(
        range(len(reason_counts)),
        reason_counts.values,
        color=[colors.get(r, "#888888") for r in reason_counts.index],
    )
    ax.set_xticks(range(len(reason_counts)))
    ax.set_xticklabels(reason_counts.index, rotation=25, ha="right", fontsize=9)
    ax.set_title("ATR trail exit reasons")
    ax.set_ylabel("Count")
    for bar, val in zip(bars_rc, reason_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val / len(df):.0%}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "30m Swing Setup: Opposite-Pivot Exit vs ATR Trail Exit\n"
        f"(trail arm={TRAIL_ACTIVATE_ATR}×, trail={TRAIL_ATR}×/{TRAIL_ATR_AFTER_TP}× after TP={TP_ATR}×, "
        f"hard_stop={HARD_STOP_ATR}×, stale={STALE_NO_PROG_BARS}bars/{STALE_FAV_BARS}bars)",
        fontsize=11,
    )
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=180)
    logger.info("Saved → %s", save_path)
    plt.close(fig)


def main() -> None:
    universe = load_universe(UNIVERSE_CSV)
    tickers  = universe["ticker"].tolist()

    all_results: list[TradeResult] = []
    n_tickers = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        data_path = RAW_30M_DIR / f"{ticker}.parquet"
        if not data_path.exists():
            continue

        df = pd.read_parquet(data_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ("timestamp", "time", "date"):
                if col in df.columns:
                    df = df.set_index(col)
                    break
        df.index = pd.to_datetime(df.index)

        recs = extract_trades_with_both_exits(ticker, df)
        if recs:
            all_results.extend(recs)

        if i % 25 == 0 or i == n_tickers:
            logger.info("(%d/%d) — %d trades so far", i, n_tickers, len(all_results))

    if not all_results:
        logger.error("No trades extracted.")
        return

    df = pd.DataFrame(all_results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "exit_comparison_trades.parquet", index=False)

    # ── summary stats ─────────────────────────────────────────────────────────
    opp_stats   = _stats(df["exit_opp_pnl"].to_numpy(),
                         df["exit_opp_bars"].to_numpy(),   "Opposite pivot")
    trail_stats = _stats(df["exit_trail_pnl"].to_numpy(),
                         df["exit_trail_bars"].to_numpy(), "ATR trail")

    print(f"\n{'Strategy':<22} {'N':>7} {'WinRate':>8} {'AvgPnL':>8} {'MedPnL':>8} "
          f"{'AvgWin':>8} {'AvgLoss':>8} {'PF':>7} {'MedHold':>9} {'P90Hold':>9}")
    print("-" * 100)
    for s in (opp_stats, trail_stats):
        print(
            f"{s['strategy']:<22} {s['n']:>7} {s['win_rate']:>8.1%} "
            f"{s['avg_pnl_atr']:>8.3f} {s['median_pnl_atr']:>8.3f} "
            f"{s['avg_win_atr']:>8.3f} {s['avg_loss_atr']:>8.3f} "
            f"{min(s['pf'], 99.9):>7.2f} "
            f"{s['med_bars']*0.5:>8.1f}h {s['p90_bars']*0.5:>8.1f}h"
        )

    print("\n── ATR trail exit reason breakdown ──")
    reason_df = df["exit_trail_reason"].value_counts().reset_index()
    reason_df.columns = ["reason", "count"]
    reason_df["pct"] = reason_df["count"] / len(df)
    for _, row in reason_df.iterrows():
        avg_pnl = df.loc[df["exit_trail_reason"] == row["reason"], "exit_trail_pnl"].mean()
        print(f"  {row['reason']:<25} {row['count']:>7}  ({row['pct']:>6.1%})  avg P&L={avg_pnl:+.3f} ATR")

    plot_comparison(df, OUT_DIR / "exit_comparison_plot.png")
    logger.info("All done → %s", OUT_DIR)


if __name__ == "__main__":
    main()
