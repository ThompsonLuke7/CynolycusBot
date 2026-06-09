"""
Single-name module diagnostic: see what each specialist model says about ONE
ticker, individually, over time -- drawn straight from the Meta Ranker matrix
(so it doubles as a sanity check that the matrix is populated correctly).

Panels (shared time axis):
  1. price (4H close)
  2. momentum expansion score   (shade = top-quintile cross-sectionally)
  3. HTF swing score            (shade = top-quintile cross-sectionally)
  4. theme signal: theme ML score + theme rank (the ticker's primary theme)
  5. confluence: signal_agreement (mom x-rank * htf x-rank)

Run: python meta_context/example_single_name.py [TICKER]   (default MU)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
MATRIX = REPO / "meta_context/meta_ranker/meta_ranker_matrix.parquet"
BARS_4H = REPO / "Data/shared/bars/4h"
OUT = REPO / "meta_context/meta_ranker"


def main(ticker: str = "MU", window_months: int = 12):
    cols = ["theme", "mom_score", "htf_score", "theme_ml_score", "theme_rank",
            "theme_rank_smooth_10", "mom_xs_rank", "htf_xs_rank", "signal_agreement"]
    df = pd.read_parquet(MATRIX, columns=cols, filters=[("ticker", "==", ticker)])
    if df.empty:
        print(f"no rows for {ticker}"); return
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").set_index("timestamp")
    if window_months:
        df = df[df.index >= df.index.max() - pd.DateOffset(months=window_months)]
    theme = df["theme"].dropna().iloc[0] if df["theme"].notna().any() else "?"

    bars = pd.read_parquet(BARS_4H / f"{ticker}.parquet")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    bars = bars.set_index("timestamp").sort_index()
    bars = bars.loc[df.index.min(): df.index.max()]

    fig, axes = plt.subplots(5, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(bars.index, bars["close"], color="#222", lw=0.9)
    axes[0].set_ylabel("price")
    axes[0].set_title(f"{ticker} (Micron) — primary theme: {theme}  "
                      f"(last {window_months} months)", fontsize=11)

    # momentum / HTF shown as CROSS-SECTIONAL PERCENTILE (the actual signal);
    # the raw scores sit in a narrow band so the percentile is what matters.
    axes[1].plot(df.index, df["mom_xs_rank"] * 100, color="#1565C0", lw=1.0)
    axes[1].axhline(80, color="#2E7D32", ls="--", lw=0.8)
    axes[1].fill_between(df.index, 80, df["mom_xs_rank"] * 100,
                         where=(df["mom_xs_rank"] > 0.8).values, color="#2E7D32", alpha=0.18)
    axes[1].set_ylim(0, 100); axes[1].set_ylabel("momentum\nx-sec %ile")

    axes[2].plot(df.index, df["htf_xs_rank"] * 100, color="#6A1B9A", lw=1.0)
    axes[2].axhline(80, color="#2E7D32", ls="--", lw=0.8)
    axes[2].fill_between(df.index, 80, df["htf_xs_rank"] * 100,
                         where=(df["htf_xs_rank"] > 0.8).values, color="#2E7D32", alpha=0.18)
    axes[2].set_ylim(0, 100); axes[2].set_ylabel("HTF swing\nx-sec %ile")

    ax3 = axes[3]
    ax3.plot(df.index, df["theme_ml_score"], color="#00897B", lw=1.2, label="theme ML score")
    ax3.set_ylabel("theme ML\nscore", color="#00897B")
    ax3b = ax3.twinx()
    ax3b.plot(df.index, df["theme_rank"], color="#EF6C00", lw=0.6, alpha=0.35, label="rank (raw)")
    ax3b.plot(df.index, df["theme_rank_smooth_10"], color="#EF6C00", lw=1.6, label="rank (10d smooth)")
    ax3b.set_ylabel("theme rank\n(1=best)", color="#EF6C00")
    ax3b.invert_yaxis()
    ax3b.legend(fontsize=6, loc="upper right")

    axes[4].fill_between(df.index, df["signal_agreement"], color="#C62828", alpha=0.5, step="mid")
    axes[4].set_ylabel("confluence\n(mom x htf)")
    axes[4].set_xlabel("date")

    for ax in axes:
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{ticker}: each specialist module's view (individually) — green shade = "
                 f"top-quintile cross-sectionally that bar", fontsize=10, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = OUT / f"example_{ticker}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"wrote {out}")
    # quick text read
    last = df.dropna(subset=["mom_score"]).iloc[-1]
    print(f"latest ({df.index.max().date()}): mom={last['mom_score']:.3f} (xs {last['mom_xs_rank']:.0%}) | "
          f"htf={last['htf_score'] if pd.notna(last['htf_score']) else float('nan')} | "
          f"theme_ml={last['theme_ml_score']:.3f} | theme_rank={last['theme_rank']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "MU")
