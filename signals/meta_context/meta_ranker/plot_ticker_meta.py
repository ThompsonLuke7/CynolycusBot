"""Overlay one ticker's 4H price with its Meta Ranker score / rank over a window.

Scores the matrix bar-by-bar (so cross-sectional rank is real), then for the
chosen ticker plots price on top and the combo score + universe rank below, with
a table of what was driving the score. Use it to see whether the ranker flagged
a big mover before the move and when it de-ranked.

  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/plot_ticker_meta.py --ticker MU --start 2026-03-15 --end 2026-06-22
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from signals.meta_context.meta_ranker.score import score_frame

HERE = Path(__file__).resolve().parent
BARS_4H = HERE.parents[2] / "Data/shared/bars/4h"
OUT_DIR = HERE / "plots"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default="2026-03-15")
    ap.add_argument("--end", default="2026-06-22")
    ap.add_argument("--matrix", default=str(HERE / "meta_ranker_matrix.parquet"))
    args = ap.parse_args()
    tk = args.ticker.upper()

    m = pd.read_parquet(args.matrix).reset_index()
    m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
    win = m[(m["timestamp"] >= args.start) & (m["timestamp"] <= args.end)].copy()
    print(f"scoring {win['timestamp'].nunique()} bars for {tk} ...")
    sc = score_frame(win)
    sc["rank"] = sc.groupby("timestamp")["s_combo"].rank(ascending=False, method="first")
    t = sc[sc["ticker"] == tk].sort_values("timestamp").copy()
    if t.empty:
        print(f"{tk} not in matrix for that window."); return

    # price
    p = BARS_4H / f"{tk}.parquet"
    px = pd.read_parquet(p, columns=["timestamp", "close"]) if p.exists() else None
    if px is not None:
        px["timestamp"] = pd.to_datetime(px["timestamp"], utc=True)
        px = px[(px["timestamp"] >= args.start) & (px["timestamp"] <= args.end)].sort_values("timestamp")

    # daily breakdown table
    t["day"] = t["timestamp"].dt.date
    daily = t.groupby("day").tail(1)
    print(f"\n=== {tk}: what drove the score (one row/day) ===")
    print(f"{'day':12s} {'rank':>5} {'combo':>6} {'upside':>7} {'qual':>6} {'mom_xr':>7} {'htf_xr':>7} "
          f"{'wThm':>5} {'news':>5} {'fwdMax':>7}")
    for _, r in daily.iterrows():
        def g(c): return r[c] if c in r and pd.notna(r[c]) else float('nan')
        print(f"{str(r['day']):12s} {int(r['rank']):>5d} {g('s_combo'):>6.2f} {g('s_upside'):>7.2f} "
              f"{g('s_quality'):>6.2f} {g('mom_xs_rank'):>7.2f} {g('htf_xs_rank'):>7.2f} "
              f"{g('within_theme_mom_rank'):>5.2f} {g('news_catalyst_score'):>5.2f} {g('fwd_max_return'):>+7.2f}")

    # plot
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, (axp, axs) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    if px is not None and not px.empty:
        axp.plot(px["timestamp"], px["close"], color="#111", lw=1.1)
        top5 = t[t["rank"] <= 5]
        top20 = t[(t["rank"] > 5) & (t["rank"] <= 20)]
        pxi = px.set_index("timestamp")["close"]
        def at(ts):
            idx = pxi.index.get_indexer([ts], method="nearest")[0]
            return pxi.iloc[idx]
        axp.scatter(top20["timestamp"], [at(x) for x in top20["timestamp"]], s=20,
                    color="#f59e0b", label="ranked 6-20", zorder=3)
        axp.scatter(top5["timestamp"], [at(x) for x in top5["timestamp"]], s=34,
                    color="#16a34a", label="ranked top-5", zorder=4)
    axp.set_title(f"{tk} — price with Meta Ranker top-5 / top-20 bars marked")
    axp.set_ylabel("price"); axp.legend(loc="upper left")

    axs.plot(t["timestamp"], t["s_combo"], color="#2563eb", lw=1.0, label="combo score")
    axs.set_ylabel("combo", color="#2563eb"); axs.set_ylim(0, 1.02)
    axr = axs.twinx()
    axr.plot(t["timestamp"], t["rank"], color="#dc2626", lw=0.9, alpha=0.7, label="universe rank")
    axr.axhline(5, color="#16a34a", ls="--", lw=0.7)
    axr.set_ylabel("rank (1=best)", color="#dc2626"); axr.set_ylim(1, 120); axr.invert_yaxis()
    fig.autofmt_xdate(); fig.tight_layout()
    out = OUT_DIR / f"meta_{tk}_{args.start}_{args.end}.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
