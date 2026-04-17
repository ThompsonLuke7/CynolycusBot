"""
Plot the 30m pivot-based swing labels with follow-through distinction.

For each selected ticker, shows the last 30 days of 30m candles with:
  - Solid green  ^  : long  pivot that hit TP   (swing_label = +1)
  - Hollow gray  ^  : long  pivot that failed   (pivot_down=1, swing_label = 0)
  - Solid red    v  : short pivot that hit TP   (swing_label = -1)
  - Hollow gray  v  : short pivot that failed   (pivot_up=1,   swing_label = 0)

The TP and SL levels are drawn as dotted lines from each pivot for the most
recent N_LEVELS pivots so you can visually confirm the follow-through filter.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Features.feature_sets.custom_indicators import add_fractal_pivots
from multi_ticker_swing.config.pipeline_config import (
    MODULE_ROOT,
    RAW_30M_DIR,
    SWING_LABEL_30M_CONFIG,
    UNIVERSE_CSV,
)
from multi_ticker_swing.data.fetch_data import load_universe
from multi_ticker_swing.labels.build_labels import _build_swing_pivot_labels_30m

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PLOTS_DIR = MODULE_ROOT / "plots" / "swing_label_30m_plots"
DAYS_BACK  = 30
N_LEVELS   = 8    # draw TP/SL lines for the most-recent N pivots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def _load_ticker(ticker: str, days_back: int) -> pd.DataFrame | None:
    path = RAW_30M_DIR / f"{ticker}.parquet"
    if not path.exists():
        logger.warning("[%s] raw 30m not found at %s", ticker, path)
        return None

    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        cutoff = pd.Timestamp.now(tz=df.index.tz) - pd.Timedelta(days=days_back)
    else:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_back)

    df = df[df.index >= cutoff].copy()
    if len(df) < 50:
        logger.warning("[%s] only %d bars after filter", ticker, len(df))
        return None
    return df


def _build_labels(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    df = df.copy()
    df["atr_14"] = _compute_atr(df, length=SWING_LABEL_30M_CONFIG["atr_length"])

    try:
        df = add_fractal_pivots(df, sequence_count=SWING_LABEL_30M_CONFIG["sequence_count"])
    except Exception as exc:
        logger.warning("[%s] fractal pivots failed: %s", ticker, exc)
        df["pivot_up"] = 0
        df["pivot_down"] = 0

    if "pivot_up"   not in df.columns: df["pivot_up"]   = 0
    if "pivot_down" not in df.columns: df["pivot_down"] = 0

    try:
        df = _build_swing_pivot_labels_30m(
            df, ticker,
            tp_mult=SWING_LABEL_30M_CONFIG["tp_mult"],
            sl_mult=SWING_LABEL_30M_CONFIG["sl_mult"],
            max_holding=SWING_LABEL_30M_CONFIG["max_holding"],
        )
    except Exception as exc:
        logger.error("[%s] label build failed: %s", ticker, exc)
        return None
    return df


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot_ticker(ticker: str) -> None:
    df = _load_ticker(ticker, DAYS_BACK)
    if df is None:
        return

    df = _build_labels(df, ticker)
    if df is None:
        return

    # ---- compressed x-axis (skip weekend gaps) ----
    n   = len(df)
    pos = np.arange(n)
    o   = df["open"].to_numpy(float)
    h   = df["high"].to_numpy(float)
    l   = df["low"].to_numpy(float)
    c   = df["close"].to_numpy(float)
    atr = df["atr_14"].to_numpy(float)
    s_lbl  = df["swing_label"].to_numpy(float)
    piv_dn = df["pivot_down"].to_numpy(int)
    piv_up = df["pivot_up"].to_numpy(int)

    marker_size  = (h - l).mean() * 0.8

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(22, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    # candles
    for i in range(n):
        bull = c[i] >= o[i]
        body_color = "#26a641" if bull else "#f85149"
        wick_color = "#8b949e"
        ax.plot([pos[i], pos[i]], [l[i], h[i]], color=wick_color, lw=0.7, zorder=1)
        body_lo = min(o[i], c[i])
        body_hi = max(o[i], c[i])
        if body_hi - body_lo < 0.0001:
            body_hi = body_lo + 0.0001
        ax.add_patch(plt.Rectangle(
            (pos[i] - 0.35, body_lo), 0.7, body_hi - body_lo,
            color=body_color, zorder=2,
        ))

    # ---- pivot markers ----
    off = marker_size * 1.0

    # Long pivots
    long_hit    = (piv_dn == 1) & (s_lbl ==  1)
    long_failed = (piv_dn == 1) & (s_lbl ==  0)
    # Short pivots
    short_hit    = (piv_up == 1) & (s_lbl == -1)
    short_failed = (piv_up == 1) & (s_lbl ==  0)

    if long_hit.any():
        ax.scatter(pos[long_hit],  l[long_hit]  - off, marker="^", s=80,
                   color="#26a641", zorder=5, label="Long hit TP")
    if long_failed.any():
        ax.scatter(pos[long_failed], l[long_failed] - off, marker="^", s=60,
                   facecolors="none", edgecolors="#8b949e", linewidths=1.2,
                   zorder=4, label="Long failed")
    if short_hit.any():
        ax.scatter(pos[short_hit],  h[short_hit]  + off, marker="v", s=80,
                   color="#f85149", zorder=5, label="Short hit TP")
    if short_failed.any():
        ax.scatter(pos[short_failed], h[short_failed] + off, marker="v", s=60,
                   facecolors="none", edgecolors="#8b949e", linewidths=1.2,
                   zorder=4, label="Short failed")

    # ---- TP / SL level lines for most-recent N_LEVELS pivots ----
    any_pivot = (piv_dn == 1) | (piv_up == 1)
    pivot_idxs = np.where(any_pivot)[0]
    recent = pivot_idxs[max(0, len(pivot_idxs) - N_LEVELS):]

    tp_mult = SWING_LABEL_30M_CONFIG["tp_mult"]
    sl_mult = SWING_LABEL_30M_CONFIG["sl_mult"]
    max_hold = SWING_LABEL_30M_CONFIG["max_holding"]

    for i in recent:
        if i + 1 >= n or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        ep   = o[i + 1]   # entry = next bar open
        end  = min(i + 1 + max_hold, n - 1)
        xr   = [pos[i + 1], pos[end]]

        if piv_dn[i] == 1:
            tp_lvl = ep + tp_mult * atr[i]
            sl_lvl = ep - sl_mult * atr[i]
            ax.plot(xr, [tp_lvl, tp_lvl], color="#26a641", lw=0.8,
                    ls="--", alpha=0.55, zorder=3)
            ax.plot(xr, [sl_lvl, sl_lvl], color="#f85149", lw=0.8,
                    ls="--", alpha=0.55, zorder=3)
        elif piv_up[i] == 1:
            tp_lvl = ep - tp_mult * atr[i]
            sl_lvl = ep + sl_mult * atr[i]
            ax.plot(xr, [tp_lvl, tp_lvl], color="#f85149", lw=0.8,
                    ls="--", alpha=0.55, zorder=3)
            ax.plot(xr, [sl_lvl, sl_lvl], color="#26a641", lw=0.8,
                    ls="--", alpha=0.55, zorder=3)

    # ---- x-axis date ticks (every ~26 bars ≈ 1 trading day) ----
    tick_step = max(1, n // 20)
    tick_pos  = pos[::tick_step]
    tick_lbl  = [df.index[i].strftime("%m/%d %H:%M") for i in range(0, n, tick_step)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=7, color="#8b949e")
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", colors="#8b949e", labelsize=8)

    # label counts
    n_lh = int(long_hit.sum());    n_lf = int(long_failed.sum())
    n_sh = int(short_hit.sum());   n_sf = int(short_failed.sum())
    total_piv = n_lh + n_lf + n_sh + n_sf
    pct_hit = 100 * (n_lh + n_sh) / max(1, total_piv)

    ax.set_title(
        f"{ticker}  |  30m swing labels  |  last {DAYS_BACK}d  "
        f"|  TP=3.0×ATR  SL=2.0×ATR  max={SWING_LABEL_30M_CONFIG['max_holding']}bars\n"
        f"Long hit={n_lh}  failed={n_lf}   Short hit={n_sh}  failed={n_sf}   "
        f"Follow-through rate={pct_hit:.0f}%",
        color="#c9d1d9", fontsize=9, pad=8,
    )

    legend_patches = [
        mpatches.Patch(color="#26a641", label=f"Long hit TP ({n_lh})"),
        mpatches.Patch(facecolor="none", edgecolor="#8b949e", label=f"Long failed ({n_lf})"),
        mpatches.Patch(color="#f85149", label=f"Short hit TP ({n_sh})"),
        mpatches.Patch(facecolor="none", edgecolor="#8b949e", label=f"Short failed ({n_sf})"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8,
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")

    ax.set_xlim(-1, n)
    plt.tight_layout()

    output_dir = PLOTS_DIR / ticker
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "swing_label_30m.png"
    fig.savefig(save_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("[%s] saved → %s", ticker, save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    universe = load_universe(UNIVERSE_CSV)
    tickers  = universe["ticker"].tolist()

    random.seed(42)
    selected = random.sample(tickers, min(5, len(tickers)))
    logger.info("Selected: %s", selected)

    for ticker in selected:
        _plot_ticker(ticker)

    logger.info("Done. Plots → %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
