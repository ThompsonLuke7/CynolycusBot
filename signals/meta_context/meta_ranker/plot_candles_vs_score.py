"""
Candles vs Meta-Ranker output.

For each ticker, draws ~9 months of 4H candlesticks with markers on the bars where
the name was in the combo top-K, above two score panels (combo, and the underlying
upside/quality probabilities). Lets you eyeball whether the model fires before moves.

  PYTHONPATH=. python signals/meta_context/meta_ranker/plot_candles_vs_score.py --tickers NVDA MU PLTR
  PYTHONPATH=. python signals/meta_context/meta_ranker/plot_candles_vs_score.py            # default set

Scores are read from /tmp/meta_scored.parquet if present (built by the analysis pass);
otherwise the matrix is scored on the fly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BARS_4H = REPO / "Data/shared/bars/4h"
MATRIX = HERE / "meta_ranker_matrix.parquet"
SCORED_CACHE = Path("/tmp/meta_scored.parquet")
OUT_DIR = HERE / "plots"
LOOKBACK_DAYS = 274
DEFAULT_TICKERS = ["NVDA", "MU", "PLTR", "COIN", "TLRY"]
TOP_K = 10


def _scored() -> pd.DataFrame:
    if SCORED_CACHE.exists():
        df = pd.read_parquet(SCORED_CACHE)
    else:
        import xgboost as xgb
        M = HERE / "models"
        mat = pd.read_parquet(MATRIX).reset_index()
        mat["timestamp"] = pd.to_datetime(mat["timestamp"], utc=True)
        df = mat[["timestamp", "ticker"]].copy()
        for lbl in ("upside", "quality"):
            feats = json.loads((M / lbl / "meta.json").read_text())["feature_names"]
            bst = xgb.Booster(); bst.load_model(str(M / lbl / "meta_ranker_xgb.json"))
            df[f"s_{lbl}"] = bst.predict(xgb.DMatrix(mat[feats].astype("float32").values, feature_names=feats))
        df["s_combo"] = (df.groupby("timestamp")["s_upside"].rank(pct=True)
                         + df.groupby("timestamp")["s_quality"].rank(pct=True)) / 2
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["in_topk"] = df.groupby("timestamp")["s_combo"].rank(ascending=False, method="first") <= TOP_K
    return df


def _candles(ax, b: pd.DataFrame):
    x = mdates.date2num(b.index.to_pydatetime())
    w = (x[1] - x[0]) * 0.7 if len(x) > 1 else 0.05
    up = b.close >= b.open
    for color, mask in [("#16a34a", up), ("#dc2626", ~up)]:
        sel = b[mask]; xs = mdates.date2num(sel.index.to_pydatetime())
        ax.vlines(xs, sel.low, sel.high, color=color, lw=0.6)
        ax.bar(xs, (sel.close - sel.open), width=w, bottom=sel.open, color=color, edgecolor=color)


def _plot(ticker: str, sc: pd.DataFrame) -> Path | None:
    bp = BARS_4H / f"{ticker}.parquet"
    if not bp.exists():
        print(f"  ! {ticker}: no bars"); return None
    s = sc[sc.ticker == ticker].sort_values("timestamp").set_index("timestamp")
    if s.empty:
        print(f"  ! {ticker}: no scores"); return None
    cutoff = s.index.max() - pd.Timedelta(days=LOOKBACK_DAYS)
    s = s[s.index >= cutoff]
    b = pd.read_parquet(bp); b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    b = b.set_index("timestamp").sort_index()
    b = b[(b.index >= s.index.min()) & (b.index <= s.index.max())]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1.2], "hspace": 0.08})
    fig.suptitle(f"{ticker} — 4H candles vs Meta-Ranker combo (markers = in combo top-{TOP_K})",
                 fontsize=13, fontweight="bold")
    ax = axes[0]
    _candles(ax, b)
    hits = s[s.in_topk]
    hit_px = b.reindex(hits.index)["high"]
    ax.scatter(mdates.date2num(hits.index.to_pydatetime()), hit_px.values * 1.02,
               marker="v", color="#2563eb", s=36, zorder=5, label=f"in combo top-{TOP_K}")
    ax.set_ylabel("price"); ax.legend(loc="upper left"); ax.grid(alpha=0.2)

    ax2 = axes[1]
    ax2.plot(s.index, s.s_combo, color="#7c3aed", lw=1.3, label="combo (rank pct)")
    ax2.plot(s.index, s.s_upside, color="#ea580c", lw=0.9, alpha=0.7, label="upside P")
    ax2.plot(s.index, s.s_quality, color="#0891b2", lw=0.9, alpha=0.7, label="quality P")
    ax2.axhline(0.90, color="#9ca3af", ls="--", lw=0.8)
    ax2.set_ylabel("score"); ax2.set_ylim(0, 1.02); ax2.legend(loc="upper left", ncol=3); ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"candles_vs_score_{ticker}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    args = ap.parse_args()
    sc = _scored()
    print(f"scored frame: {len(sc):,} rows; plotting {args.tickers}")
    for t in args.tickers:
        _plot(t, sc)


if __name__ == "__main__":
    main()
