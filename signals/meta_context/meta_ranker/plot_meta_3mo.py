"""Three-month review of Meta Ranker top picks vs. what they actually did.

Scores the live matrix, takes the daily top-K combo picks over the trailing
window, computes each pick's realized forward move from the 4H bars, and plots
forward max-return by date (biggest winners labeled). Answers: did the ranker
surface any monster moves, and how did the cohort do overall.

  PYTHONPATH=. .venv/bin/python signals/meta_context/meta_ranker/plot_meta_3mo.py --days 90 --top 5 --hold-bars 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from signals.meta_context.meta_ranker.score import score_frame, DEFAULT_MATRIX

HERE = Path(__file__).resolve().parent
BARS_4H = HERE.parents[2] / "Data/shared/bars/4h"
OUT_DIR = HERE / "plots"


def _fwd_moves(ticker: str, ts: pd.Timestamp, hold_bars: int) -> tuple[float, float] | None:
    p = BARS_4H / f"{ticker}.parquet"
    if not p.exists():
        return None
    b = pd.read_parquet(p, columns=["timestamp", "high", "close"])
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
    b = b.sort_values("timestamp")
    fut = b[b["timestamp"] >= ts]
    if len(fut) < 2:
        return None
    entry = float(fut.iloc[0]["close"])
    if entry <= 0:
        return None
    window = fut.iloc[1 : hold_bars + 1]
    if window.empty:
        return None
    fwd_max = float(window["high"].max()) / entry - 1.0
    fwd_close = float(window.iloc[-1]["close"]) / entry - 1.0
    return fwd_max, fwd_close


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--hold-bars", type=int, default=20, help="~10 sessions at 2 4H bars/day")
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    args = ap.parse_args()

    df = pd.read_parquet(args.matrix).reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=args.days)
    df = df[df["timestamp"] >= cutoff].copy()
    print(f"scoring {df['timestamp'].nunique()} bars since {cutoff.date()} ...")

    scored = score_frame(df)
    scored["rank"] = scored.groupby("timestamp")["s_combo"].rank(ascending=False, method="first")
    picks = scored[scored["rank"] <= args.top][["timestamp", "ticker", "s_combo", "rank"]].copy()

    rows = []
    for _, r in picks.iterrows():
        mv = _fwd_moves(str(r["ticker"]), r["timestamp"], args.hold_bars)
        if mv is None:
            continue
        rows.append({"timestamp": r["timestamp"], "ticker": r["ticker"], "rank": int(r["rank"]),
                     "fwd_max": mv[0], "fwd_close": mv[1]})
    res = pd.DataFrame(rows)
    if res.empty:
        print("no forward data available (4H bars missing for picks).")
        return

    # Only score picks whose forward window is complete (>= hold_bars bars existed).
    mature = res[res["timestamp"] <= res["timestamp"].max() - pd.Timedelta(days=10)]
    hit = (mature["fwd_max"] >= 0.10).mean() if len(mature) else float("nan")
    print(f"\n=== Meta top-{args.top} picks, last {args.days}d "
          f"(forward {args.hold_bars} 4H bars ≈ {args.hold_bars // 2} sessions) ===")
    print(f"  matured picks      : {len(mature)}")
    print(f"  median fwd max     : {mature['fwd_max'].median():+.1%}")
    print(f"  median fwd close   : {mature['fwd_close'].median():+.1%}")
    print(f"  hit-rate (>=+10%)  : {hit:.0%}")
    print("\n  top 15 winners by forward max move:")
    top = mature.sort_values("fwd_max", ascending=False).head(15)
    for _, r in top.iterrows():
        print(f"    {r['timestamp'].date()}  #{r['rank']:<2d} {r['ticker']:<6s} "
              f"max {r['fwd_max']:+.0%}  close {r['fwd_close']:+.0%}")

    # ---- plot ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    sc = ax.scatter(res["timestamp"], res["fwd_max"] * 100, c=res["rank"],
                    cmap="viridis_r", s=28, alpha=0.8, edgecolor="none")
    ax.axhline(0, color="#888", lw=0.8)
    ax.axhline(10, color="#16a34a", lw=0.8, ls="--", label="+10% target")
    # label the biggest winners
    for _, r in res.sort_values("fwd_max", ascending=False).head(12).iterrows():
        ax.annotate(r["ticker"], (r["timestamp"], r["fwd_max"] * 100),
                    fontsize=8, xytext=(0, 4), textcoords="offset points", ha="center")
    cbar = fig.colorbar(sc, ax=ax); cbar.set_label("pick rank (1 = top)")
    ax.set_title(f"Meta Ranker top-{args.top} picks — forward max move "
                 f"(~{args.hold_bars // 2} sessions), last {args.days}d")
    ax.set_ylabel("forward max return (%)"); ax.set_xlabel("signal bar")
    ax.legend(loc="upper left")
    fig.autofmt_xdate(); fig.tight_layout()
    out = OUT_DIR / f"meta_top{args.top}_fwd_{args.days}d.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
