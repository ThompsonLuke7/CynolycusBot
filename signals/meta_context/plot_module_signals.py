"""Plot the raw specialist-module signals for a few tickers BEFORE the Meta Ranker.

For each chosen ticker, draws the last ~6 months of:
  1. price (4H close)
  2. momentum-expansion vs HTF-swing OOF scores  (the two base models)
  3. dynamic-theme context (heat / strength / membership)
  4. news-catalyst signal (score + bull/crash trajectory probabilities)

Tickers are picked at random from the names with ample coverage across ALL four
modules, so the panels actually show every signal working.

    python signals/meta_context/plot_module_signals.py            # 3 random tickers
    python signals/meta_context/plot_module_signals.py --tickers NVDA AAPL XOM
    python signals/meta_context/plot_module_signals.py --seed 7 --n 3

Outputs: signals/meta_context/meta_ranker/module_signals_<TICKER>_last6mo.png
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MATRIX = REPO / "signals/meta_context/meta_ranker/meta_ranker_matrix.parquet"
BARS_4H = REPO / "Data/shared/bars/4h"
OUT_DIR = REPO / "signals/meta_context/meta_ranker"

LOOKBACK_DAYS = 183

_COLS = [
    "timestamp", "ticker", "mom_score", "htf_score",
    "theme_heat_score", "theme_strength", "membership_score",
    "news_catalyst_score", "news_p_bull_steady", "news_p_crash_stayed",
    "dollar_vol_pctile_252",
]


def _load_recent_matrix() -> pd.DataFrame:
    # (timestamp, ticker) are the parquet index — pyarrow restores them as the
    # index, so request only the feature columns then reset_index to recover them.
    feat = [c for c in _COLS if c not in ("timestamp", "ticker")]
    df = pd.read_parquet(MATRIX, columns=feat).reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=LOOKBACK_DAYS)
    return df[df["timestamp"] >= cutoff].copy()


def _well_covered(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker coverage of each module; keep names with ample data everywhere."""
    g = df.groupby("ticker")
    cov = pd.DataFrame({
        "rows": g.size(),
        "htf": g["htf_score"].apply(lambda s: s.notna().mean()),
        "theme": g["theme_heat_score"].apply(lambda s: s.notna().mean()),
        "news": g["news_catalyst_score"].apply(lambda s: s.notna().mean()),
        "liq": g["dollar_vol_pctile_252"].median(),
    })
    keep = cov[(cov["rows"] >= 150) & (cov["htf"] >= 0.5) &
              (cov["theme"] >= 0.8) & (cov["news"] >= 0.25) & (cov["liq"] >= 0.6)]
    return keep.sort_values(["news", "htf"], ascending=False)


def _plot_ticker(ticker: str, mdf: pd.DataFrame) -> Path:
    sub = mdf[mdf["ticker"] == ticker].sort_values("timestamp").set_index("timestamp")

    # price from the 4H bar cache (richer than the matrix, which has no spot price)
    bars_path = BARS_4H / f"{ticker}.parquet"
    price = None
    if bars_path.exists():
        b = pd.read_parquet(bars_path, columns=["timestamp", "close"])
        b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
        b = b[(b["timestamp"] >= sub.index.min()) & (b["timestamp"] <= sub.index.max())]
        price = b.set_index("timestamp")["close"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.4, 1.4, 1.4], "hspace": 0.12})
    fig.suptitle(f"{ticker} — specialist module signals (last 6 months, pre-Meta-Ranker)",
                 fontsize=14, fontweight="bold", y=0.995)

    # 1) price
    ax = axes[0]
    if price is not None and len(price):
        ax.plot(price.index, price.values, color="#111827", lw=1.3)
        ax.fill_between(price.index, price.values, price.min(), color="#111827", alpha=0.05)
    ax.set_ylabel("4H close ($)")
    ax.set_title("Price", loc="left", fontsize=10, color="#555")

    # 2) base models: momentum vs HTF swing (twin axes, different native scales)
    ax = axes[1]
    ax.plot(sub.index, sub["mom_score"], color="#2563eb", lw=1.3, label="momentum score")
    ax.set_ylabel("momentum", color="#2563eb")
    ax.tick_params(axis="y", labelcolor="#2563eb")
    ax2 = ax.twinx()
    ax2.plot(sub.index, sub["htf_score"], color="#d97706", lw=1.3, label="HTF swing score")
    ax2.set_ylabel("HTF swing", color="#d97706")
    ax2.tick_params(axis="y", labelcolor="#d97706")
    ax.set_title("Base models — momentum expansion (long) vs HTF swing (long, ~½ bars covered)",
                 loc="left", fontsize=10, color="#555")

    # 3) dynamic theme context
    ax = axes[2]
    for col, c, lbl in [("theme_heat_score", "#dc2626", "theme heat"),
                        ("theme_strength", "#16a34a", "theme strength"),
                        ("membership_score", "#7c3aed", "membership")]:
        ax.plot(sub.index, sub[col], color=c, lw=1.2, label=lbl)
    ax.set_ylabel("0–1")
    ax.legend(loc="upper left", fontsize=8, ncol=3, frameon=False)
    ax.set_title("Dynamic theme context (its primary theme's heat/strength + membership)",
                 loc="left", fontsize=10, color="#555")

    # 4) news catalyst (sparse, event-driven → markers + trajectory probabilities)
    ax = axes[3]
    nc = sub["news_catalyst_score"].dropna()
    if len(nc):
        ax.scatter(nc.index, nc.values, s=14, color="#0891b2", label="catalyst score", zorder=3)
    for col, c, lbl in [("news_p_bull_steady", "#16a34a", "P(bull steady)"),
                        ("news_p_crash_stayed", "#dc2626", "P(crash stayed)")]:
        s = sub[col].dropna()
        if len(s):
            ax.plot(s.index, s.values, color=c, lw=1.0, alpha=0.8, label=lbl)
    ax.set_ylabel("score / prob")
    ax.legend(loc="upper left", fontsize=8, ncol=3, frameon=False)
    ax.set_title("News-catalyst module (bidirectional — bullish & crash trajectories)",
                 loc="left", fontsize=10, color="#555")

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    for a in axes:
        a.grid(True, alpha=0.18)
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out = OUT_DIR / f"module_signals_{ticker}_last6mo.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", help="explicit tickers (skip random pick)")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    mdf = _load_recent_matrix()
    print(f"matrix window: {mdf['timestamp'].min().date()} → {mdf['timestamp'].max().date()}  "
          f"({mdf['ticker'].nunique()} tickers)")

    if args.tickers:
        chosen = [t.upper() for t in args.tickers]
    else:
        pool = _well_covered(mdf)
        print(f"{len(pool)} tickers have ample coverage across all four modules")
        rng = random.Random(args.seed)
        # sample from the better-covered half so every panel is populated
        head = pool.head(max(args.n * 12, 30)).index.tolist()
        chosen = rng.sample(head, min(args.n, len(head)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for t in chosen:
        path = _plot_ticker(t, mdf)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
