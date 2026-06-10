"""
ATR threshold sweep for 30-minute soft swing pivot labels.

For each core pivot across all universe tickers:
  - Long  entry: close[pivot_down bar], exit when next short core pivot fires
  - Short entry: close[pivot_up  bar], exit when next long  core pivot fires
  - MFE/MAE measured in ATR units during the hold window

Then sweeps a grid of TP × SL ATR multiples and reports:
  - Win rate, avg win (ATR), avg loss (ATR), expectancy (ATR), Sharpe-like ratio
  - Histogram of raw swing sizes in ATR
  - Heatmap of expectancy across the TP/SL grid
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from strategies.spy_intraday.Features.feature_sets.custom_indicators import add_fractal_pivots
from strategies.multi_ticker_swing.config.pipeline_config import (
    MODULE_ROOT,
    UNIVERSE_CSV,
    RAW_30M_DIR,
)
from strategies.multi_ticker_swing.data.fetch_data import load_universe
from strategies.multi_ticker_swing.plots.generate_soft_swing_30m_plots import (
    FIRST_IN_RUN_FILTER,
    POSITIVE_WINDOW_BARS,
    AMBIGUOUS_WINDOW_BARS,
    NEIGHBOR_WEIGHT,
    AMBIGUOUS_WEIGHT,
    _session_dates,
    apply_swing_pivot_zone_weights_session_aware,
    keep_first_same_side_event_session_reset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OUT_DIR = MODULE_ROOT / "plots" / "atr_sweep"

# ATR multiples to sweep
TP_VALUES = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
SL_VALUES = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]


# ---------------------------------------------------------------------------
# ATR helper
# ---------------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()


# ---------------------------------------------------------------------------
# Per-ticker trade extraction
# ---------------------------------------------------------------------------

def extract_trades(ticker: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Return a DataFrame of trades with columns:
      side, entry_price, atr_at_entry, mfe_atr, mae_atr, swing_atr,
      bars_held, exit_reason
    where swing_atr = (exit_at_opposite_pivot - entry) / atr   (signed)
    and mfe/mae are the best/worst unrealised excursion before opposite pivot fires.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            logger.warning("[%s] missing column %s", ticker, col)
            return None

    df["atr_14"] = _compute_atr(df, 14)
    df = df.dropna(subset=["atr_14"])

    try:
        df = add_fractal_pivots(df)
    except Exception as exc:
        logger.warning("[%s] fractal pivots failed: %s", ticker, exc)
        return None

    df["pivot_up"]   = df.get("pivot_up",   pd.Series(0, index=df.index)).fillna(0)
    df["pivot_down"] = df.get("pivot_down", pd.Series(0, index=df.index)).fillna(0)

    # Apply same label shift as the plot (core = T-1 within same session)
    from strategies.multi_ticker_swing.plots.generate_soft_swing_30m_plots import (
        _shift_pivot_labels_back, LABEL_SHIFT_BARS,
    )
    raw_long_pivot  = df["pivot_down"].copy()
    raw_short_pivot = df["pivot_up"].copy()
    long_shifted  = _shift_pivot_labels_back(raw_long_pivot,  df.index, LABEL_SHIFT_BARS).to_numpy()
    short_shifted = _shift_pivot_labels_back(raw_short_pivot, df.index, LABEL_SHIFT_BARS).to_numpy()

    timestamps = df.index
    lows_arr  = df["low"].to_numpy(dtype=float)
    highs_arr = df["high"].to_numpy(dtype=float)

    # Apply session-aware first-in-run filter (same as the label plot)
    if FIRST_IN_RUN_FILTER:
        long_core, short_core, _, _ = keep_first_same_side_event_session_reset(
            long_shifted, short_shifted, timestamps,
            lows=lows_arr, highs=highs_arr,
        )
    else:
        long_core, short_core = long_shifted.copy(), short_shifted.copy()

    closes = df["close"].to_numpy(dtype=float)
    opens  = df["open"].to_numpy(dtype=float)
    atrs   = df["atr_14"].to_numpy(dtype=float)
    highs  = highs_arr
    lows   = lows_arr
    n      = len(df)

    long_idx  = np.flatnonzero(long_core  == 1)
    short_idx = np.flatnonzero(short_core == 1)

    records = []

    def _simulate(entry_bar: int, side: str) -> dict | None:
        """Simulate one trade: enter at open of bar AFTER the label bar."""
        actual_entry = entry_bar + 1          # next bar's open
        if actual_entry >= n:
            return None                        # no next bar available
        entry_price = opens[actual_entry]     # realistic fill price
        atr         = atrs[entry_bar]         # ATR known at label bar
        if atr <= 0 or np.isnan(atr):
            return None

        # Find the first opposite pivot strictly after entry
        if side == "long":
            opp_idx_arr = short_idx[short_idx > entry_bar]
        else:
            opp_idx_arr = long_idx[long_idx > entry_bar]

        if len(opp_idx_arr) == 0:
            return None  # no exit found

        # Exit at open of bar after the opposite label fires (symmetric)
        opp_label_bar = int(opp_idx_arr[0])
        exit_actual   = opp_label_bar + 1
        if exit_actual >= n:
            exit_actual = opp_label_bar
        exit_price = opens[exit_actual]
        bars_held  = exit_actual - actual_entry

        # MFE / MAE over the hold window (actual entry → actual exit inclusive)
        window_high = highs[actual_entry:exit_actual + 1].max()
        window_low  = lows[actual_entry:exit_actual + 1].min()

        if side == "long":
            mfe_atr    = (window_high  - entry_price) / atr
            mae_atr    = (entry_price  - window_low)  / atr
            swing_atr  = (exit_price   - entry_price) / atr
        else:
            mfe_atr    = (entry_price  - window_low)  / atr
            mae_atr    = (window_high  - entry_price) / atr
            swing_atr  = (entry_price  - exit_price)  / atr

        return {
            "ticker":       ticker,
            "side":         side,
            "entry_bar":    entry_bar,
            "entry_price":  entry_price,
            "atr_at_entry": atr,
            "mfe_atr":      mfe_atr,
            "mae_atr":      mae_atr,
            "swing_atr":    swing_atr,
            "bars_held":    bars_held,
        }

    for i in long_idx:
        rec = _simulate(int(i), "long")
        if rec:
            records.append(rec)
    for i in short_idx:
        rec = _simulate(int(i), "short")
        if rec:
            records.append(rec)

    if not records:
        return None
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# ATR grid simulation
# ---------------------------------------------------------------------------

def simulate_tp_sl_grid(trades: pd.DataFrame) -> pd.DataFrame:
    """
    For each (TP, SL) pair: check if MFE >= TP before MAE >= SL.
    Returns a flat DataFrame with columns: tp, sl, win_rate, avg_win_atr,
    avg_loss_atr, expectancy_atr, n_trades, profit_factor.
    """
    results = []
    for tp in TP_VALUES:
        for sl in SL_VALUES:
            wins  = trades["mfe_atr"] >= tp
            # only a win if MFE reaches TP before MAE reaches SL
            # approximation: if mfe >= tp AND (mae < sl OR mfe/mae >= tp/sl)
            # Simpler but sound: win if mfe >= tp and mae < sl (TP hit first), or mfe >= tp (MAE check)
            # Best-effort without bar-by-bar: win if mfe_atr >= tp AND mae_atr < sl
            true_wins  = (trades["mfe_atr"] >= tp) & (trades["mae_atr"] < sl)
            true_losses= (trades["mae_atr"] >= sl) & (trades["mfe_atr"] < tp)
            n_total    = len(trades)
            n_win      = true_wins.sum()
            n_loss     = true_losses.sum()
            n_open     = n_total - n_win - n_loss  # neither hit (partial swing)

            win_rate   = n_win / n_total if n_total else 0.0
            avg_win    = tp   # simplified: always full TP if win
            avg_loss   = sl   # simplified: always full SL if loss
            # include partial outcomes at actual swing_atr value
            partial_return = trades.loc[~true_wins & ~true_losses, "swing_atr"].mean() if n_open else 0.0

            expectancy = (
                win_rate * avg_win
                - (n_loss / n_total) * avg_loss
                + (n_open / n_total) * partial_return
            ) if n_total else 0.0

            pf = (n_win * tp) / (n_loss * sl) if n_loss > 0 else np.inf

            results.append({
                "tp": tp, "sl": sl,
                "win_rate": win_rate,
                "avg_win_atr": avg_win,
                "avg_loss_atr": avg_loss,
                "expectancy_atr": expectancy,
                "profit_factor": min(pf, 10.0),
                "n_trades": n_total,
                "n_win": int(n_win),
                "n_loss": int(n_loss),
            })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_swing_size_histogram(trades: pd.DataFrame, save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, side, color in [
        (axes[0], "long",  "#2E7D32"),
        (axes[1], "short", "#C62828"),
    ]:
        sub = trades[trades["side"] == side]
        ax.hist(sub["mfe_atr"].clip(0, 8),  bins=40, color=color,    alpha=0.7, label="MFE (ATR)")
        ax.hist(sub["mae_atr"].clip(0, 8),  bins=40, color="#888888", alpha=0.5, label="MAE (ATR)")
        ax.axvline(sub["mfe_atr"].median(), color=color,    linestyle="--", linewidth=1.5,
                   label=f"median MFE {sub['mfe_atr'].median():.2f}x ATR")
        ax.axvline(sub["mae_atr"].median(), color="#555555", linestyle="--", linewidth=1.5,
                   label=f"median MAE {sub['mae_atr'].median():.2f}x ATR")
        ax.set_title(f"{side.upper()} pivot MFE / MAE distribution (n={len(sub)})")
        ax.set_xlabel("ATR multiples")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("30m Swing Pivot — MFE / MAE until opposite pivot fires", fontsize=13)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=180)
    logger.info("Saved → %s", save_path)
    plt.close(fig)


def plot_expectancy_heatmap(grid: pd.DataFrame, save_path: Path) -> None:
    """2D heatmap: TP (x) vs SL (y) coloured by expectancy."""
    for metric, title, cmap in [
        ("expectancy_atr",  "Expectancy (ATR per trade)",  "RdYlGn"),
        ("win_rate",         "Win Rate",                   "YlGn"),
        ("profit_factor",    "Profit Factor (capped 10x)", "RdYlGn"),
    ]:
        pivot_tbl = grid.pivot(index="sl", columns="tp", values=metric)
        pivot_tbl = pivot_tbl.sort_index(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(
            pivot_tbl.values,
            cmap=cmap,
            aspect="auto",
            norm=mcolors.TwoSlopeNorm(
                vmin=float(pivot_tbl.values.min()),
                vcenter=float(np.nanmedian(pivot_tbl.values)),
                vmax=float(pivot_tbl.values.max()),
            ) if metric != "win_rate" else None,
        )
        plt.colorbar(im, ax=ax, label=title)
        ax.set_xticks(range(len(TP_VALUES)))
        ax.set_xticklabels([f"{v:g}×" for v in TP_VALUES])
        ax.set_yticks(range(len(SL_VALUES)))
        ax.set_yticklabels([f"{v:g}×" for v in sorted(SL_VALUES, reverse=True)])
        ax.set_xlabel("TP (ATR multiple)")
        ax.set_ylabel("SL (ATR multiple)")
        ax.set_title(f"30m Swing Setup — {title}\n(entry at pivot, exit at opposite pivot or TP/SL)")

        # Annotate cells
        for i, sl in enumerate(sorted(SL_VALUES, reverse=True)):
            for j, tp in enumerate(TP_VALUES):
                val = grid.loc[(grid["tp"] == tp) & (grid["sl"] == sl), metric]
                if not val.empty:
                    ax.text(j, i, f"{float(val.iloc[0]):.2f}",
                            ha="center", va="center", fontsize=7.5,
                            color="black")

        plt.tight_layout()
        fname = save_path.parent / f"{save_path.stem}_{metric}.png"
        plt.savefig(fname, bbox_inches="tight", dpi=180)
        logger.info("Saved → %s", fname)
        plt.close(fig)


def print_summary_table(grid: pd.DataFrame) -> None:
    print("\n=== ATR Sweep Summary (sorted by expectancy) ===")
    print(f"{'TP':>6} {'SL':>6} {'WinRate':>9} {'Expectancy':>12} {'ProfitFact':>12} {'N':>8}")
    print("-" * 60)
    for _, row in grid.sort_values("expectancy_atr", ascending=False).head(20).iterrows():
        print(
            f"{row['tp']:>6.2f} {row['sl']:>6.2f} "
            f"{row['win_rate']:>9.1%} "
            f"{row['expectancy_atr']:>12.3f} "
            f"{row['profit_factor']:>12.2f} "
            f"{int(row['n_trades']):>8}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    universe = load_universe(UNIVERSE_CSV)
    tickers  = universe["ticker"].tolist()

    all_trades: list[pd.DataFrame] = []
    n = len(tickers)

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

        trades = extract_trades(ticker, df)
        if trades is not None and len(trades) >= 5:
            all_trades.append(trades)
            if i % 20 == 0 or i == n:
                logger.info("(%d/%d) collected — running total %d trades",
                            i, n, sum(len(t) for t in all_trades))

    if not all_trades:
        logger.error("No trades extracted.")
        return

    trades_all = pd.concat(all_trades, ignore_index=True)
    logger.info("Total trades: %d  (long=%d, short=%d)",
                len(trades_all),
                (trades_all["side"] == "long").sum(),
                (trades_all["side"] == "short").sum())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw trades for inspection
    trades_all.to_parquet(OUT_DIR / "swing_trades_30m.parquet", index=False)
    trades_all.to_csv(OUT_DIR / "swing_trades_30m.csv", index=False)
    logger.info("Trades saved → %s", OUT_DIR / "swing_trades_30m.parquet")

    # Swing size histogram
    plot_swing_size_histogram(trades_all, OUT_DIR / "swing_mfe_mae_histogram.png")

    # TP/SL grid
    grid = simulate_tp_sl_grid(trades_all)
    grid.to_csv(OUT_DIR / "tp_sl_grid.csv", index=False)
    plot_expectancy_heatmap(grid, OUT_DIR / "tp_sl_heatmap.png")
    print_summary_table(grid)

    logger.info("All outputs → %s", OUT_DIR)


if __name__ == "__main__":
    main()
