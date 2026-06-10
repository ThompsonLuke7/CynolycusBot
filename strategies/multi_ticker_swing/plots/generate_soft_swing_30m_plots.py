"""
Generate soft swing label plots for 5 random tickers on 30-minute candles
over the last month.

Uses the same logic as Data/plots/soft_swing_label_plot.py with two
session-aware corrections:

  1. Gap-aware neighbor window: the ±1 bar neighbor zone never crosses an
     overnight session boundary.  If the pivot is the first bar of a new day
     (after a gap), the previous day's closing bar is NOT marked as a neighbor.

  2. Session-reset first-in-run filter: active_side resets to None at every
     new trading day so that a bearish day doesn't suppress a pivot low that
     comes later in the same or the next session.
"""
from __future__ import annotations

import logging
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Data.plots.plots import (
    _apply_time_ticks,
    _compute_marker_offset,
    _compute_time_ticks,
    _draw_day_lines,
    _extract_ohlc,
    _plot_candles,
    _select_plot_window,
)
from strategies.spy_intraday.Features.feature_sets.custom_indicators import add_fractal_pivots
from strategies.multi_ticker_swing.config.pipeline_config import (
    MODULE_ROOT,
    UNIVERSE_CSV,
    RAW_30M_DIR,
)
from strategies.multi_ticker_swing.data.fetch_data import load_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PLOTS_DIR = MODULE_ROOT / "plots" / "soft_swing_30m_plots"

# Weighting parameters (same defaults as the 10m pipeline)
POSITIVE_WINDOW_BARS = 1    # ±1 bar around core gets neighbor_weight
AMBIGUOUS_WINDOW_BARS = 3   # wider band gets ambiguous_weight
NEIGHBOR_WEIGHT = 0.75
AMBIGUOUS_WEIGHT = 0.0
FIRST_IN_RUN_FILTER = True

# Shift core label 1 bar earlier than the fractal pivot.
# Rationale: fractal pivot at T is only CONFIRMED at T+3 (3-bar lookahead).
# The best practical entry was T-1 — the bar immediately before the reversal
# bar. The model should learn to fire there, not at T where it's already late.
# Gap-aware: if T is the first bar of a new session (overnight gap), keep at T
# because T-1 is a pre-gap bar at a completely different price level.
LABEL_SHIFT_BARS = 1

# Follow-through check for visual filled/hollow distinction on core markers.
# "Filled" = price moved at least FOLLOWTHROUGH_MIN_ATR × ATR in the right
# direction within FOLLOWTHROUGH_BARS bars after entry (open of next bar).
# This is a plot-only indicator — it does NOT change training labels.
FOLLOWTHROUGH_MIN_ATR = 1.0   # minimum ATR move required
FOLLOWTHROUGH_BARS    = 12    # look-forward window (12 × 30m = 6 hours)


# ---------------------------------------------------------------------------
# Session-aware helpers
# ---------------------------------------------------------------------------

def _session_dates(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Return an array of date objects (tz-stripped) for each bar."""
    if timestamps.tz is not None:
        return timestamps.tz_convert("America/New_York").date
    return np.array(timestamps.date)


def _shift_pivot_labels_back(
    pivot_series: pd.Series,
    timestamps: pd.DatetimeIndex,
    shift: int = 1,
) -> pd.Series:
    """
    Move each core pivot label `shift` bars earlier, within the same session.
    If a pivot fires on the first bar of a new day (gap open), keep it at T
    so we don't cross the overnight gap backward.
    """
    if shift <= 0:
        return (pivot_series > 0).astype(int)

    dates  = _session_dates(timestamps)
    src    = (pivot_series > 0).to_numpy()
    out    = np.zeros(len(src), dtype=np.int64)
    n      = len(src)

    for i in range(n):
        if not src[i]:
            continue
        target = i - shift
        # Walk backward until we find a bar in the same session (or give up)
        while target < 0 or dates[target] != dates[i]:
            target += 1           # can't go that far back — land at T
            if target >= i:
                target = i
                break
        out[target] = 1

    return pd.Series(out, index=pivot_series.index)


def apply_swing_pivot_zone_weights_session_aware(
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    *,
    positive_window_bars: int,
    ambiguous_window_bars: int,
    neighbor_weight: float,
    ambiguous_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Same as apply_swing_pivot_zone_weights() but the BACKWARD expansion of the
    neighbor / ambiguous windows is capped at session boundaries.

    Rule: if bar[j] and the pivot bar[i] are on different calendar days,
    bar[j] (j < i) is NOT included in any neighbor or ambiguous zone.
    The forward direction is unchanged — a post-gap follow-through bar CAN
    still be marked as a neighbor.
    """
    dates = _session_dates(timestamps)
    n = len(y)
    y_base = (np.asarray(y) == 1).astype(np.int64)
    y_zone = y_base.copy()
    weights = np.ones(n, dtype=np.float32)
    source = np.zeros(n, dtype=np.int8)

    if n == 0:
        return y_zone, weights, source

    pos_w = max(0, int(positive_window_bars))
    amb_w = max(pos_w, max(0, int(ambiguous_window_bars)))
    core_idx = np.flatnonzero(y_base == 1)

    if core_idx.size == 0:
        return y_zone, weights, source

    # --- ambiguous band first (wider) ---
    if amb_w > 0:
        for idx in core_idx:
            pivot_date = dates[idx]
            # backward: only same session
            for j in range(max(0, idx - amb_w), idx):
                if dates[j] == pivot_date:
                    weights[j] = float(ambiguous_weight)
                    source[j] = 3
            # forward: always allowed
            for j in range(idx + 1, min(n, idx + amb_w + 1)):
                weights[j] = float(ambiguous_weight)
                source[j] = 3

    # --- positive (neighbor) band overrides ambiguous ---
    if pos_w > 0:
        for idx in core_idx:
            pivot_date = dates[idx]
            # backward: only same session
            for j in range(max(0, idx - pos_w), idx):
                if dates[j] == pivot_date:
                    y_zone[j] = 1
                    weights[j] = float(neighbor_weight)
                    source[j] = 2
            # forward: always allowed
            for j in range(idx + 1, min(n, idx + pos_w + 1)):
                y_zone[j] = 1
                weights[j] = float(neighbor_weight)
                source[j] = 2

    # core bars override everything
    y_zone[core_idx] = 1
    weights[core_idx] = 1.0
    source[core_idx] = 1
    return y_zone, weights, source


def keep_first_same_side_event_session_reset(
    long_y: np.ndarray,
    short_y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    lows: np.ndarray,
    highs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    First-in-run filter with two reset conditions:

    1. Session boundary — active_side resets at every new trading day.
    2. Price progression — a same-side event is NOT suppressed when the market
       has made a genuinely new extreme since the last kept event:
         • Long:  new pivot low  < last kept long's low  (fresh lower low)
         • Short: new pivot high > last kept short's high (fresh higher high)

    This prevents suppressing a lower low that the market drifted down to after
    an earlier long pivot, which is a new setup, not a duplicate.
    """
    dates = _session_dates(timestamps)
    long_in  = (np.asarray(long_y)  == 1).astype(np.int64)
    short_in = (np.asarray(short_y) == 1).astype(np.int64)

    long_out        = long_in.copy()
    short_out       = short_in.copy()
    suppressed_long  = np.zeros(len(long_in),  dtype=bool)
    suppressed_short = np.zeros(len(short_in), dtype=bool)

    active_side: str | None = None
    last_long_low:  float = np.inf
    last_short_high: float = -np.inf
    prev_date = None

    for idx, (is_long, is_short) in enumerate(zip(long_in == 1, short_in == 1)):
        cur_date = dates[idx]

        # ── session boundary → reset filter ──────────────────────────────────
        if prev_date is not None and cur_date != prev_date:
            active_side = None
        prev_date = cur_date

        if is_long and is_short:          # simultaneous conflict → suppress both
            long_out[idx]  = 0
            short_out[idx] = 0
            suppressed_long[idx]  = True
            suppressed_short[idx] = True
            active_side = None
            continue

        if is_long:
            cur_low = float(lows[idx])
            # Allow if: first long, opposite side was last active, OR new lower low
            if active_side == "long" and cur_low >= last_long_low:
                long_out[idx] = 0
                suppressed_long[idx] = True
            else:
                active_side = "long"
                last_long_low = cur_low

        elif is_short:
            cur_high = float(highs[idx])
            # Allow if: first short, opposite side was last active, OR new higher high
            if active_side == "short" and cur_high <= last_short_high:
                short_out[idx] = 0
                suppressed_short[idx] = True
            else:
                active_side = "short"
                last_short_high = cur_high

    return long_out, short_out, suppressed_long, suppressed_short


# ---------------------------------------------------------------------------
# Follow-through check (plot-only — does NOT affect training labels)
# ---------------------------------------------------------------------------

def _add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Add atr_14 column if not already present."""
    if "atr_14" in df.columns:
        return df
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(length).mean()
    return df


def _compute_followthrough_mask(
    df: pd.DataFrame,
    core_mask: np.ndarray,
    direction: str,          # "long" or "short"
    min_atr: float = FOLLOWTHROUGH_MIN_ATR,
    max_bars: int  = FOLLOWTHROUGH_BARS,
) -> np.ndarray:
    """
    For each bar where core_mask is True, check whether price moved at least
    min_atr × ATR in `direction` within max_bars bars after entry (open[i+1]).

    Returns a boolean array, same length as df, True = followed through.
    """
    n      = len(df)
    highs  = df["high"].to_numpy(float)
    lows   = df["low"].to_numpy(float)
    opens  = df["open"].to_numpy(float)
    atr    = df["atr_14"].to_numpy(float)
    result = np.zeros(n, dtype=bool)

    for i in np.where(core_mask)[0]:
        if i + 1 >= n:
            continue
        ep = opens[i + 1]
        if np.isnan(ep) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        target = ep + min_atr * atr[i] if direction == "long" else ep - min_atr * atr[i]
        end = min(i + 1 + max_bars, n)
        for j in range(i + 1, end):
            if direction == "long"  and highs[j] >= target:
                result[i] = True
                break
            if direction == "short" and lows[j]  <= target:
                result[i] = True
                break
    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_and_filter_30m(ticker: str, days_back: int) -> pd.DataFrame | None:
    data_path = RAW_30M_DIR / f"{ticker}.parquet"
    if not data_path.exists():
        logger.warning("[%s] 30m data not found at %s", ticker, data_path)
        return None

    df = pd.read_parquet(data_path)

    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("timestamp", "time", "date"):
            if col in df.columns:
                df = df.set_index(col)
                break
    df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]

    if df.index.tz is not None:
        cutoff = pd.Timestamp.now(tz=df.index.tz) - pd.Timedelta(days=days_back)
    else:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)
    df = df[df.index >= cutoff]

    if len(df) < 50:
        logger.warning("[%s] only %d bars after filtering — skipping", ticker, len(df))
        return None

    return df


def _compute_swing_labels(
    df: pd.DataFrame,
    ticker: str,
    sequence_count: int = 3,
    label_shift: int = LABEL_SHIFT_BARS,
) -> pd.DataFrame | None:
    df = _add_atr(df)   # needed for follow-through check

    try:
        df = add_fractal_pivots(df, sequence_count=sequence_count)
    except Exception as exc:
        logger.warning("[%s] add_fractal_pivots failed: %s", ticker, exc)
        df["pivot_up"]   = 0
        df["pivot_down"] = 0

    df["pivot_up"]   = df.get("pivot_up",   pd.Series(0, index=df.index)).fillna(0)
    df["pivot_down"] = df.get("pivot_down", pd.Series(0, index=df.index)).fillna(0)

    # Shift core label `label_shift` bars earlier within the same session.
    # pivot_down → long candidate; pivot_up → short candidate
    df["long_swing_label"]  = _shift_pivot_labels_back(df["pivot_down"], df.index, label_shift)
    df["short_swing_label"] = _shift_pivot_labels_back(df["pivot_up"],   df.index, label_shift)
    return df


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_soft_swing_labels_30m(
    df: pd.DataFrame,
    ticker: str,
    save_path: str | Path,
) -> None:
    timestamps = df.index  # used for session-boundary checks

    long_core_full  = df["long_swing_label"].fillna(0).astype(np.int64).to_numpy()
    short_core_full = df["short_swing_label"].fillna(0).astype(np.int64).to_numpy()

    suppressed_run_long_full  = np.zeros(len(df), dtype=bool)
    suppressed_run_short_full = np.zeros(len(df), dtype=bool)

    if FIRST_IN_RUN_FILTER:
        lows_arr  = df["low"].to_numpy(dtype=float)
        highs_arr = df["high"].to_numpy(dtype=float)
        (long_core_full, short_core_full,
         suppressed_run_long_full, suppressed_run_short_full) = (
            keep_first_same_side_event_session_reset(
                long_core_full, short_core_full, timestamps,
                lows=lows_arr, highs=highs_arr,
            )
        )

    long_zone_full, long_w_full, _ = apply_swing_pivot_zone_weights_session_aware(
        long_core_full, timestamps,
        positive_window_bars=POSITIVE_WINDOW_BARS,
        ambiguous_window_bars=AMBIGUOUS_WINDOW_BARS,
        neighbor_weight=NEIGHBOR_WEIGHT,
        ambiguous_weight=AMBIGUOUS_WEIGHT,
    )
    short_zone_full, short_w_full, _ = apply_swing_pivot_zone_weights_session_aware(
        short_core_full, timestamps,
        positive_window_bars=POSITIVE_WINDOW_BARS,
        ambiguous_window_bars=AMBIGUOUS_WINDOW_BARS,
        neighbor_weight=NEIGHBOR_WEIGHT,
        ambiguous_weight=AMBIGUOUS_WEIGHT,
    )

    work = df.copy()
    work["soft_long_core"]        = long_core_full
    work["soft_short_core"]       = short_core_full
    work["soft_long_label"]       = long_zone_full
    work["soft_short_label"]      = short_zone_full
    work["soft_long_weight"]      = long_w_full
    work["soft_short_weight"]     = short_w_full
    work["suppressed_run_long"]   = suppressed_run_long_full
    work["suppressed_run_short"]  = suppressed_run_short_full
    work = _select_plot_window(work, window=None, random_window=False)

    pos, open_y, high_y, low_y, close_y = _extract_ohlc(work)
    marker_offset = _compute_marker_offset(work, high_y, low_y)
    date_index = work.index
    tick_positions, tick_labels = _compute_time_ticks(date_index, pos, max_ticks=20)

    long_core       = work["soft_long_core"].to_numpy(dtype=bool)
    short_core      = work["soft_short_core"].to_numpy(dtype=bool)
    long_zone       = work["soft_long_label"].to_numpy(dtype=bool)
    short_zone      = work["soft_short_label"].to_numpy(dtype=bool)
    long_w          = work["soft_long_weight"].to_numpy(dtype=np.float32)
    short_w         = work["soft_short_weight"].to_numpy(dtype=np.float32)
    long_neighbor   = long_zone  & ~long_core
    short_neighbor  = short_zone & ~short_core
    long_ambiguous  = (~long_zone)  & (long_w  != 1.0) & (long_w  > 0)
    short_ambiguous = (~short_zone) & (short_w != 1.0) & (short_w > 0)
    suppressed_run_long  = work["suppressed_run_long"].to_numpy(dtype=bool)
    suppressed_run_short = work["suppressed_run_short"].to_numpy(dtype=bool)

    fig, (ax_price, ax_weight) = plt.subplots(
        2, 1, figsize=(18, 8), sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.2]},
    )
    _plot_candles(ax_price, pos, open_y, high_y, low_y, close_y)

    long_y_pos          = low_y  - marker_offset * 0.85
    long_neighbor_y_pos = low_y  - marker_offset * 0.55
    long_ambig_y_pos    = low_y  - marker_offset * 0.25
    short_y_pos         = high_y + marker_offset * 0.85
    short_neighbor_y_pos= high_y + marker_offset * 0.55
    short_ambig_y_pos   = high_y + marker_offset * 0.25

    # Follow-through masks for core bars (plot-only, no effect on labels)
    long_ft  = _compute_followthrough_mask(work, long_core,  "long")
    short_ft = _compute_followthrough_mask(work, short_core, "short")
    long_core_hit   = long_core  &  long_ft
    long_core_miss  = long_core  & ~long_ft
    short_core_hit  = short_core &  short_ft
    short_core_miss = short_core & ~short_ft

    if long_neighbor.any():
        ax_price.scatter(pos[long_neighbor], long_neighbor_y_pos[long_neighbor],
                         marker="^", s=45, color="#66BB6A", alpha=0.8,
                         label=f"LONG neighbor weight={NEIGHBOR_WEIGHT:g}", zorder=3)
    if long_core_hit.any():
        ax_price.scatter(pos[long_core_hit], long_y_pos[long_core_hit],
                         marker="^", s=100, color="#1B5E20",
                         label=f"LONG core — followed through (≥{FOLLOWTHROUGH_MIN_ATR}×ATR)", zorder=3.3)
    if long_core_miss.any():
        ax_price.scatter(pos[long_core_miss], long_y_pos[long_core_miss],
                         marker="^", s=90, facecolors="none", edgecolors="#1B5E20",
                         linewidths=1.8, label="LONG core — no follow-through", zorder=3.2)
    if long_ambiguous.any():
        ax_price.scatter(pos[long_ambiguous], long_ambig_y_pos[long_ambiguous],
                         marker="x", s=38, color="#9E9E9E", alpha=0.7,
                         label=f"LONG ambiguous weight={AMBIGUOUS_WEIGHT:g}", zorder=2.8)
    if suppressed_run_long.any():
        ax_price.scatter(pos[suppressed_run_long], long_ambig_y_pos[suppressed_run_long],
                         marker=".", s=34, color="#2E7D32", alpha=0.45,
                         label="LONG suppressed (first-in-run)", zorder=2.9)

    if short_neighbor.any():
        ax_price.scatter(pos[short_neighbor], short_neighbor_y_pos[short_neighbor],
                         marker="v", s=45, color="#EF5350", alpha=0.8,
                         label=f"SHORT neighbor weight={NEIGHBOR_WEIGHT:g}", zorder=3)
    if short_core_hit.any():
        ax_price.scatter(pos[short_core_hit], short_y_pos[short_core_hit],
                         marker="v", s=100, color="#B71C1C",
                         label=f"SHORT core — followed through (≥{FOLLOWTHROUGH_MIN_ATR}×ATR)", zorder=3.3)
    if short_core_miss.any():
        ax_price.scatter(pos[short_core_miss], short_y_pos[short_core_miss],
                         marker="v", s=90, facecolors="none", edgecolors="#B71C1C",
                         linewidths=1.8, label="SHORT core — no follow-through", zorder=3.2)
    if short_ambiguous.any():
        ax_price.scatter(pos[short_ambiguous], short_ambig_y_pos[short_ambiguous],
                         marker="x", s=38, color="#616161", alpha=0.7,
                         label=f"SHORT ambiguous weight={AMBIGUOUS_WEIGHT:g}", zorder=2.8)
    if suppressed_run_short.any():
        ax_price.scatter(pos[suppressed_run_short], short_ambig_y_pos[suppressed_run_short],
                         marker=".", s=34, color="#C62828", alpha=0.45,
                         label="SHORT suppressed (first-in-run)", zorder=2.9)

    ax_weight.axhline(0.0, color="#333333", linewidth=1)
    ax_weight.step(pos, long_zone.astype(float) * long_w, where="mid",
                   color="#2E7D32", label="LONG label * weight")
    ax_weight.step(pos, -short_zone.astype(float) * short_w, where="mid",
                   color="#C62828", label="-SHORT label * weight")
    if long_ambiguous.any():
        ax_weight.scatter(pos[long_ambiguous], np.zeros(long_ambiguous.sum()),
                          marker="x", s=26, color="#9E9E9E", alpha=0.65,
                          label="LONG ambiguous (zero weight)")
    if short_ambiguous.any():
        ax_weight.scatter(pos[short_ambiguous], np.zeros(short_ambiguous.sum()),
                          marker="x", s=26, color="#616161", alpha=0.65,
                          label="SHORT ambiguous (zero weight)")

    _draw_day_lines((ax_price, ax_weight), tick_positions, line_color="#d0d0d0")
    _apply_time_ticks(ax_weight, tick_positions, tick_labels)

    n_lhit = int(long_core_hit.sum());  n_lmiss = int(long_core_miss.sum())
    n_shit = int(short_core_hit.sum()); n_smiss = int(short_core_miss.sum())
    ax_price.set_title(
        f"30m Soft Swing Labels — {ticker}  "
        f"(neighbor ±{POSITIVE_WINDOW_BARS}={NEIGHBOR_WEIGHT:g}, first_in_run={FIRST_IN_RUN_FILTER})\n"
        f"Filled=followed through ≥{FOLLOWTHROUGH_MIN_ATR}×ATR in {FOLLOWTHROUGH_BARS} bars  |  "
        f"Long: {n_lhit} hit / {n_lmiss} miss   Short: {n_shit} hit / {n_smiss} miss"
    )
    ax_price.set_ylabel("Price")
    ax_weight.set_ylabel("label * weight")
    ax_weight.set_xlabel("Date")
    ax_weight.set_ylim(
        min(-1.1, float(np.nanmin(-short_zone.astype(float) * short_w)) - 0.1),
        max(1.1,  float(np.nanmax(long_zone.astype(float)  * long_w))  + 0.1),
    )
    ax_price.grid(True, alpha=0.25)
    ax_weight.grid(True, alpha=0.25)
    ax_price.legend(loc="best", fontsize=8, ncols=2)
    ax_weight.legend(loc="best", fontsize=8, ncols=2)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    logger.info("[%s] saved → %s", ticker, save_path)
    plt.close(fig)

    print(
        f"[{ticker}] long_core={int(long_core.sum())}, "
        f"long_zone={int(long_zone.sum())}, "
        f"suppressed_long={int(suppressed_run_long.sum())}, "
        f"short_core={int(short_core.sum())}, "
        f"short_zone={int(short_zone.sum())}, "
        f"suppressed_short={int(suppressed_run_short.sum())}"
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def plot_ticker(
    ticker: str,
    days_back: int = 30,
    sequence_count: int = 3,
    label_shift: int = LABEL_SHIFT_BARS,
    subdir: str = "soft_swing_30m",
) -> None:
    df = _load_and_filter_30m(ticker, days_back)
    if df is None:
        return
    df = _compute_swing_labels(df, ticker, sequence_count=sequence_count, label_shift=label_shift)
    if df is None:
        return
    output_dir = PLOTS_DIR / ticker
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_soft_swing_labels_30m(df, ticker, save_path=output_dir / f"{subdir}.png")


def main():
    universe = load_universe(UNIVERSE_CSV)
    tickers  = universe["ticker"].tolist()

    random.seed(42)
    selected = random.sample(tickers, min(5, len(tickers)))
    logger.info("Selected tickers: %s", selected)

    # --- seq_count=3, label shifted back 1 bar (primary) ---
    logger.info("=== seq_count=3, label_shift=1 (primary) ===")
    for ticker in selected:
        plot_ticker(ticker, days_back=30, sequence_count=3,
                    label_shift=1, subdir="soft_swing_30m_seq3_shift1")

    # --- seq_count=2 comparison (noisier, earlier confirmation) ---
    logger.info("=== seq_count=2 comparison ===")
    for ticker in selected:
        plot_ticker(ticker, days_back=30, sequence_count=2,
                    label_shift=1, subdir="soft_swing_30m_seq2_shift1")

    logger.info("Done. Plots saved to %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
