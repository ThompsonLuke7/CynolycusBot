"""
Top-N combo picks for chosen 4H bars — eyeball pick quality.

For each chosen signal bar, plots the top-N combo tickers' 4H candles in a window
around the bar: entry bar marked, the ~10-day forward hold window shaded, and the
realized forward return annotated. One figure per bar (N subplots).

  PYTHONPATH=. python signals/meta_context/meta_ranker/plot_topk_bars.py
  PYTHONPATH=. python signals/meta_context/meta_ranker/plot_topk_bars.py --bars 2025-09-12 2026-01-09 --top 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
BARS_4H = HERE.parents[2] / "Data/shared/bars/4h"
SCORED = Path("/tmp/meta_scored.parquet")
OUT_DIR = HERE / "plots"
PRE_DAYS, POST_DAYS, HOLD_BARS = 55, 30, 25


def _candles(ax, b):
    if b.empty:
        return
    xs = mdates.date2num(b.index.to_pydatetime())
    w = (xs[1] - xs[0]) * 0.7 if len(xs) > 1 else 0.05
    up = b.close >= b.open
    for color, mask in [("#16a34a", up), ("#dc2626", ~up)]:
        s = b[mask]; x = mdates.date2num(s.index.to_pydatetime())
        ax.vlines(x, s.low, s.high, color=color, lw=0.6)
        ax.bar(x, s.close - s.open, width=w, bottom=s.open, color=color, edgecolor=color)


def _plot_bar(ts: pd.Timestamp, sc: pd.DataFrame, n: int):
    bar = sc[sc.timestamp == ts]
    if bar.empty:
        print(f"  ! {ts}: no scored rows"); return
    top = bar.sort_values("s_combo", ascending=False).head(n)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.2))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Top-{n} combo picks @ {ts:%Y-%m-%d %H:%MZ}", fontsize=13, fontweight="bold")
    for ax, (_, row) in zip(axes, top.iterrows()):
        tkr = row.ticker
        bp = BARS_4H / f"{tkr}.parquet"
        if not bp.exists():
            ax.set_title(f"{tkr} (no bars)"); continue
        b = pd.read_parquet(bp); b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
        b = b.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        lo, hi = ts - pd.Timedelta(days=PRE_DAYS), ts + pd.Timedelta(days=POST_DAYS)
        win = b[(b.index >= lo) & (b.index <= hi)]
        _candles(ax, win)
        ax.axvline(mdates.date2num(ts.to_pydatetime()), color="#2563eb", lw=1.4, ls="--")
        # shade the forward hold window
        if ts in b.index:
            pos = b.index.get_loc(ts)
            exit_ts = b.index[min(pos + HOLD_BARS, len(b) - 1)]
            ax.axvspan(mdates.date2num(ts.to_pydatetime()), mdates.date2num(exit_ts.to_pydatetime()),
                       color="#3b82f6", alpha=0.08)
        fr = row.get("fwd_close_return", float("nan"))
        ax.set_title(f"{tkr}  combo={row.s_combo:.3f}  (u={row.s_upside:.2f} q={row.s_quality:.2f})\n"
                     f"realized fwd_close={fr:+.1%}", fontsize=10)
        ax.grid(alpha=0.2); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        for lab in ax.get_xticklabels():
            lab.set_rotation(30); lab.set_ha("right")
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"top{n}_{ts:%Y%m%d_%H%MZ}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"  wrote {out}  ->  {list(top.ticker)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", nargs="*", default=None, help="UTC dates/timestamps; default = 3 auto-picked.")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    sc = pd.read_parquet(SCORED).dropna(subset=["s_combo"])
    sc["timestamp"] = pd.to_datetime(sc["timestamp"], utc=True)
    counts = sc.groupby("timestamp").size()
    full = counts[counts >= 300].index
    if args.bars:
        want = [pd.Timestamp(x, tz="UTC") if len(str(x)) > 10 else pd.Timestamp(x).tz_localize("UTC") for x in args.bars]
        bars = [min(full, key=lambda t: abs((t - w).total_seconds())) for w in want]
    else:
        fl = full.sort_values()
        bars = [fl[int(len(fl) * f)] for f in (0.55, 0.78, 0.995)]  # spread across history incl. recent
    print(f"plotting top-{args.top} for bars: {[str(b) for b in bars]}")
    for ts in bars:
        _plot_bar(ts, sc, args.top)


if __name__ == "__main__":
    main()
